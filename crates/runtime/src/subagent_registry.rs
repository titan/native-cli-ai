//! Subagent task registry + read-only control consumer (P1 of
//! `docs/subagent-task-lifecycle.md` §5-P1, §6 invariants).
//!
//! The registry is an **in-memory projection** of child-session lifecycle
//! events — populated live by `spawn_child_session` and re-derived at resume
//! by folding the event log (`Supervisor::resume`). It never writes another
//! session's json: the control consumer answers `task_status`/`task_result`
//! from the registry plus `SessionStore::load` (read-only), preserving the
//! single-writer invariant.

use crate::session_store::SessionStore;
use nca_common::event::{AgentEvent, EventEnvelope};
use nca_common::message::{ContentPart, MessageContent, Role};
use nca_common::session::{ChildSessionState, SessionState};
use nca_core::agent_driver::InboxItem;
use nca_core::tools::subagent_control::{SubagentControlRequest, SubagentControlResponse};
use std::collections::HashSet;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// One tracked child (subagent) task.
#[derive(Debug, Clone)]
pub struct SubagentRegistryEntry {
    pub session_id: String,
    pub parent_session_id: String,
    pub task: String,
    pub state: ChildSessionState,
    pub generation: u64,
    pub alias: Option<String>,
    pub workspace: String,
    pub branch: Option<String>,
    pub worktree_path: Option<String>,
    pub result_summary: Option<String>,
    /// Live cancel handle of a running child (`AgentLoop::cancel_handle`).
    /// `None` for terminal tasks (cleared by [`SubagentRegistry::record_terminal`])
    /// and for entries re-derived from the event log at resume — handles are
    /// runtime-only state, never persisted.
    pub cancel_flag: Option<Arc<AtomicBool>>,
    /// Live inbox sender of a running child (`Supervisor::inbox_sender`);
    /// `task_message` steering is queued through it. Cleared on terminal.
    pub inbox_tx: Option<mpsc::Sender<InboxItem>>,
    /// Reason recorded by the last `task_cancel` request, folded into the
    /// terminal `result_summary` ("cancelled: <reason>") and cleared when
    /// the task goes terminal.
    pub cancel_reason: Option<String>,
}

/// In-memory projection of spawned child tasks, keyed by child session id
/// and kept in spawn order for `list()`. Cheap to rebuild from the event
/// log; never persisted directly.
#[derive(Default)]
pub struct SubagentRegistry {
    entries: Mutex<Vec<SubagentRegistryEntry>>,
    /// Session ids with an in-flight mutating control operation (see
    /// [`SubagentRegistry::try_acquire_lease`]).
    control_leases: Arc<Mutex<HashSet<String>>>,
}

impl SubagentRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a freshly spawned child in `Running` state. Idempotent: a
    /// duplicate spawn event (e.g. live call + log fold overlap) is skipped.
    pub fn record_spawned(
        &self,
        parent_session_id: &str,
        child_session_id: &str,
        task: &str,
        workspace: String,
        branch: Option<String>,
    ) {
        let mut entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        if entries.iter().any(|e| e.session_id == child_session_id) {
            return;
        }
        entries.push(SubagentRegistryEntry {
            session_id: child_session_id.to_string(),
            parent_session_id: parent_session_id.to_string(),
            task: task.to_string(),
            state: ChildSessionState::Running,
            generation: 0,
            alias: None,
            workspace,
            branch,
            worktree_path: None,
            result_summary: None,
            cancel_flag: None,
            inbox_tx: None,
            cancel_reason: None,
        });
    }

    /// Record a live child's control handles (cancel flag + inbox sender)
    /// right after spawn. No-op for ids the registry does not track.
    pub fn record_handles(
        &self,
        child_session_id: &str,
        cancel_flag: Arc<AtomicBool>,
        inbox_tx: mpsc::Sender<InboxItem>,
    ) {
        let mut entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        let Some(entry) = entries
            .iter_mut()
            .find(|e| e.session_id == child_session_id)
        else {
            return;
        };
        entry.cancel_flag = Some(cancel_flag);
        entry.inbox_tx = Some(inbox_tx);
    }

    /// Record the reason of a `task_cancel` request against a tracked child.
    /// No-op for unknown ids; a `None` reason clears a previously recorded
    /// one only by explicit request (cancel-without-reason is legitimate).
    pub fn record_cancel_requested(&self, child_session_id: &str, reason: Option<String>) {
        let mut entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        let Some(entry) = entries
            .iter_mut()
            .find(|e| e.session_id == child_session_id)
        else {
            return;
        };
        entry.cancel_reason = reason;
    }

    /// Drop a child's live control handles (and any spent cancel reason).
    /// Called automatically by [`Self::record_terminal`] — terminal tasks
    /// have no live handles — and safe to call directly/idempotently.
    pub fn clear_handles(&self, child_session_id: &str) {
        let mut entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        let Some(entry) = entries
            .iter_mut()
            .find(|e| e.session_id == child_session_id)
        else {
            return;
        };
        entry.cancel_flag = None;
        entry.inbox_tx = None;
        entry.cancel_reason = None;
    }

    /// Fold a terminal transition for a tracked child. Unknown ids are
    /// ignored (the registry only tracks children it saw spawn). A `None`
    /// result_summary never clobbers a previously recorded summary — the
    /// richer `ChildSessionStatusChanged` may arrive before the coarser
    /// `ChildSessionCompleted`.
    pub fn record_terminal(
        &self,
        child_session_id: &str,
        state: ChildSessionState,
        result_summary: Option<String>,
    ) {
        let mut entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        let Some(entry) = entries
            .iter_mut()
            .find(|e| e.session_id == child_session_id)
        else {
            return;
        };
        entry.state = state;
        if result_summary.is_some() {
            entry.result_summary = result_summary;
        }
        // Terminal tasks have no live handles; the cancel reason has been
        // folded into the terminal result_summary by the caller.
        entry.cancel_flag = None;
        entry.inbox_tx = None;
        entry.cancel_reason = None;
    }

    /// Apply one event-log envelope: `ChildSessionSpawned` records the
    /// spawn, `ChildSessionStatusChanged` updates tracked fields, and
    /// `ChildSessionCompleted` folds the terminal state (without
    /// downgrading a more-specific `ChildSessionStatusChanged` already
    /// recorded — see [`Self::record_terminal`]). All other events are
    /// ignored.
    pub fn apply_envelope(&self, envelope: &EventEnvelope) {
        match &envelope.event {
            AgentEvent::ChildSessionSpawned {
                parent_session_id,
                child_session_id,
                task,
                workspace,
                branch,
            } => self.record_spawned(
                parent_session_id,
                child_session_id,
                task,
                workspace.display().to_string(),
                branch.clone(),
            ),
            AgentEvent::ChildSessionStatusChanged {
                child_session_id,
                state,
                generation,
                alias,
                result_summary,
                ..
            } => {
                let mut entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
                let Some(entry) = entries
                    .iter_mut()
                    .find(|e| e.session_id == *child_session_id)
                else {
                    return;
                };
                entry.state = *state;
                entry.generation = *generation;
                if alias.is_some() {
                    entry.alias = alias.clone();
                }
                if result_summary.is_some() {
                    entry.result_summary = result_summary.clone();
                }
            }
            AgentEvent::ChildSessionCompleted {
                child_session_id,
                status,
                ..
            } => self.record_terminal(
                child_session_id,
                ChildSessionState::from_spawn_status(status),
                None,
            ),
            _ => {}
        }
    }

    /// Look up one entry by child session id.
    pub fn get(&self, id: &str) -> Option<SubagentRegistryEntry> {
        let entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        entries.iter().find(|e| e.session_id == id).cloned()
    }

    /// Resolve a control-request target by exact session id first, then by
    /// parent-scoped alias.
    ///
    /// - Exact id match always wins (an alias may collide with another
    ///   task's id; the id is the unambiguous handle).
    /// - Otherwise entries whose `alias` equals `id_or_alias`: none →
    ///   `Ok(None)`, exactly one → `Ok(Some)`, more than one →
    ///   `Err("ambiguous alias '<x>' matches N tasks")`.
    pub fn resolve(&self, id_or_alias: &str) -> Result<Option<SubagentRegistryEntry>, String> {
        let entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(entry) = entries.iter().find(|e| e.session_id == id_or_alias) {
            return Ok(Some(entry.clone()));
        }
        let matches: Vec<SubagentRegistryEntry> = entries
            .iter()
            .filter(|e| e.alias.as_deref() == Some(id_or_alias))
            .cloned()
            .collect();
        match matches.len() {
            0 => Ok(None),
            1 => Ok(matches.into_iter().next()),
            n => Err(format!("ambiguous alias '{id_or_alias}' matches {n} tasks")),
        }
    }

    /// All entries in spawn order.
    pub fn list(&self) -> Vec<SubagentRegistryEntry> {
        let entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        entries.clone()
    }

    /// Try to acquire the single in-flight mutating control lease for a
    /// task (cancel/revive). Returns `None` when another control operation
    /// is already in flight; the lease is released when the guard drops.
    ///
    /// Simplified per `docs/subagent-task-lifecycle.md` §4: one lease per
    /// task id, no upstream `statusUncertain`/liveness-reconciliation
    /// machinery — the consumer serializes requests on one channel anyway,
    /// so the lease only guards against overlapping *asynchronous* control
    /// sequences (e.g. revive-while-cancelling).
    pub fn try_acquire_lease(&self, session_id: &str) -> Option<ControlLease> {
        let mut held = self
            .control_leases
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if held.contains(session_id) {
            return None;
        }
        held.insert(session_id.to_string());
        Some(ControlLease {
            session_id: session_id.to_string(),
            held: Arc::clone(&self.control_leases),
        })
    }
}

/// Guard for a task's single in-flight mutating control operation
/// (cancel/revive). Released on drop; acquire via
/// [`SubagentRegistry::try_acquire_lease`].
pub struct ControlLease {
    session_id: String,
    held: Arc<Mutex<HashSet<String>>>,
}

impl Drop for ControlLease {
    fn drop(&mut self) {
        let mut held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        held.remove(&self.session_id);
    }
}

/// Last non-empty assistant message text from a session state, full text
/// (joins `ContentPart::Text` parts; non-`Parts` content → `event_preview()`).
/// Skips trailing tool/empty messages: a child that ended mid-tool-call or
/// with a whitespace reply yields the previous real answer, or `None`.
pub fn last_assistant_text(state: &SessionState) -> Option<String> {
    for message in state.messages.iter().rev() {
        if !matches!(message.role, Role::Assistant) {
            continue;
        }
        let text = match &message.content {
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
            other => other.event_preview(),
        };
        if !text.trim().is_empty() {
            return Some(text);
        }
    }
    None
}

/// Derive a response entry from a loaded session json (registry miss
/// fallback): lifecycle state from the persisted `SessionStatus`, identity
/// fields from the meta. The task text is not persisted on the child
/// session, so it reports `None`.
fn response_from_state(session_id: &str, state: &SessionState) -> SubagentControlResponse {
    SubagentControlResponse {
        session_id: session_id.to_string(),
        state: ChildSessionState::from_session_status(state.meta.status.clone()),
        task: None,
        workspace: Some(state.meta.workspace.display().to_string()),
        branch: state.meta.branch.clone(),
        result_summary: state.meta.session_summary.clone(),
        output: None,
        note: None,
        ok: true,
        error_message: None,
        generation: None,
    }
}

/// Placeholder reply for P2 write-side operations (`Message`/`Cancel`/
/// `Revive`): the wire types land in chunk A, but executing them needs the
/// control lease (chunk B/C) — fail loudly instead of pretending success.
fn operation_not_available(session_id: &str) -> SubagentControlResponse {
    SubagentControlResponse::unknown(
        session_id,
        "task control operation is not available in this build",
    )
}

/// Consume `task_status`/`task_result` control requests against the
/// registry, with a read-only `SessionStore::load` fallback for ids the
/// registry never saw (e.g. spawned before this process started). Spawned
/// by `Supervisor::create`; replies ride per-request oneshot channels.
///
/// **Invariant:** never calls `session_store.save` — introspection is
/// strictly read-only (single-writer preserved).
pub fn subagent_control_consumer(
    mut control_rx: mpsc::Receiver<SubagentControlRequest>,
    registry: Arc<SubagentRegistry>,
    session_store: SessionStore,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(request) = control_rx.recv().await {
            match request {
                SubagentControlRequest::Status { session_id, reply } => {
                    let response = match registry.get(&session_id) {
                        Some(entry) => SubagentControlResponse {
                            session_id: entry.session_id,
                            state: entry.state,
                            task: Some(entry.task),
                            workspace: Some(entry.workspace),
                            branch: entry.branch,
                            result_summary: entry.result_summary,
                            output: None,
                            note: None,
                            ok: true,
                            error_message: None,
                            generation: None,
                        },
                        None => match session_store.load(&session_id).await {
                            Ok(state) => response_from_state(&session_id, &state),
                            Err(_) => SubagentControlResponse::unknown(
                                &session_id,
                                "unknown subagent task id",
                            ),
                        },
                    };
                    let _ = reply.send(response);
                }
                SubagentControlRequest::Result { session_id, reply } => {
                    let response = match registry.get(&session_id) {
                        Some(entry) => {
                            build_result_response(
                                &session_id,
                                entry.state,
                                entry.result_summary,
                                &session_store,
                            )
                            .await
                        }
                        None => match session_store.load(&session_id).await {
                            Ok(state) => {
                                let state = &state;
                                let mut response = response_from_state(&session_id, state);
                                if !response.state.is_terminal() {
                                    response.note = Some("task is still running".into());
                                } else {
                                    response.output = last_assistant_text(state);
                                    if response.output.is_none() {
                                        response.note = Some(
                                            "task finished but has no assistant message on record"
                                                .into(),
                                        );
                                    }
                                }
                                response
                            }
                            Err(_) => SubagentControlResponse::unknown(
                                &session_id,
                                "unknown subagent task id",
                            ),
                        },
                    };
                    let _ = reply.send(response);
                }
                // P2 chunk A placeholders: the write-side operations are
                // executed by the control lease + revive path in chunk B/C;
                // until then every request fails loudly (never silently
                // succeeds against a live child).
                SubagentControlRequest::Message {
                    session_id, reply, ..
                } => {
                    let _ = reply.send(operation_not_available(&session_id));
                }
                SubagentControlRequest::Cancel {
                    session_id, reply, ..
                } => {
                    let _ = reply.send(operation_not_available(&session_id));
                }
                SubagentControlRequest::Revive {
                    session_id, reply, ..
                } => {
                    let _ = reply.send(operation_not_available(&session_id));
                }
            }
        }
    })
}

/// `task_result` reply for a registry-known task: running tasks report
/// `output: None` + note; terminal tasks load the child json read-only and
/// return the last assistant message.
async fn build_result_response(
    session_id: &str,
    state: ChildSessionState,
    result_summary: Option<String>,
    session_store: &SessionStore,
) -> SubagentControlResponse {
    let mut response = SubagentControlResponse {
        session_id: session_id.to_string(),
        state,
        task: None,
        workspace: None,
        branch: None,
        result_summary,
        output: None,
        note: None,
        ok: true,
        error_message: None,
        generation: None,
    };
    if !state.is_terminal() {
        response.note = Some("task is still running".into());
        return response;
    }
    match session_store.load(session_id).await {
        Ok(child) => {
            response.output = last_assistant_text(&child);
            if response.output.is_none() {
                response.note = Some("task finished but has no assistant message on record".into());
            }
        }
        Err(e) => {
            response.note = Some(format!(
                "task finished but its session record could not be loaded: {e}"
            ));
        }
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use nca_common::event::EventEnvelope;
    use nca_common::message::Message;
    use nca_common::session::{SessionMeta, SessionStatus};
    use std::sync::atomic::AtomicBool;
    use tokio::sync::oneshot;

    fn spawned_envelope(id: u64, child: &str) -> EventEnvelope {
        EventEnvelope::new(
            id,
            AgentEvent::ChildSessionSpawned {
                parent_session_id: "parent".into(),
                child_session_id: child.into(),
                task: format!("task-{child}"),
                workspace: std::path::PathBuf::from("/tmp/ws"),
                branch: Some(format!("nca/{child}")),
            },
        )
    }

    fn status_changed_envelope(id: u64, child: &str, state: ChildSessionState) -> EventEnvelope {
        EventEnvelope::new(
            id,
            AgentEvent::ChildSessionStatusChanged {
                parent_session_id: "parent".into(),
                child_session_id: child.into(),
                state,
                generation: 1,
                alias: Some("fixer".into()),
                result_summary: Some(format!("summary-{child}")),
            },
        )
    }

    fn completed_envelope(id: u64, child: &str, status: &str) -> EventEnvelope {
        EventEnvelope::new(
            id,
            AgentEvent::ChildSessionCompleted {
                parent_session_id: "parent".into(),
                child_session_id: child.into(),
                status: status.into(),
            },
        )
    }

    #[test]
    fn record_spawned_then_get_returns_running() {
        let registry = SubagentRegistry::new();
        registry.record_spawned(
            "parent",
            "c1",
            "do things",
            "/ws".into(),
            Some("nca/c1".into()),
        );
        let entry = registry.get("c1").expect("entry recorded");
        assert_eq!(entry.state, ChildSessionState::Running);
        assert_eq!(entry.parent_session_id, "parent");
        assert_eq!(entry.task, "do things");
        assert_eq!(entry.workspace, "/ws");
        assert_eq!(entry.branch.as_deref(), Some("nca/c1"));
        assert_eq!(entry.result_summary, None);
        assert!(registry.get("missing").is_none());
    }

    #[test]
    fn record_spawned_is_idempotent() {
        let registry = SubagentRegistry::new();
        registry.record_spawned("p", "c1", "first", "/ws".into(), None);
        registry.record_spawned("p", "c1", "second", "/ws".into(), None);
        let entry = registry.get("c1").expect("entry");
        assert_eq!(entry.task, "first", "duplicate spawn must be skipped");
        assert_eq!(registry.list().len(), 1);
    }

    #[test]
    fn record_terminal_updates_state_and_summary() {
        let registry = SubagentRegistry::new();
        registry.record_spawned("p", "c1", "task", "/ws".into(), None);
        registry.record_terminal("c1", ChildSessionState::Failed, Some("boom".into()));
        let entry = registry.get("c1").expect("entry");
        assert_eq!(entry.state, ChildSessionState::Failed);
        assert_eq!(entry.result_summary.as_deref(), Some("boom"));
    }

    #[test]
    fn record_terminal_unknown_id_is_ignored() {
        let registry = SubagentRegistry::new();
        registry.record_terminal("ghost", ChildSessionState::Completed, Some("x".into()));
        assert!(registry.get("ghost").is_none());
        assert!(registry.list().is_empty());
    }

    #[test]
    fn apply_envelope_folds_spawned_then_completed() {
        let registry = SubagentRegistry::new();
        registry.apply_envelope(&spawned_envelope(1, "c1"));
        registry.apply_envelope(&completed_envelope(2, "c1", "completed"));
        let entry = registry.get("c1").expect("entry");
        assert_eq!(entry.state, ChildSessionState::Completed);
        assert_eq!(entry.result_summary, None);

        // error status maps to Failed
        registry.apply_envelope(&spawned_envelope(3, "c2"));
        registry.apply_envelope(&completed_envelope(4, "c2", "error"));
        assert_eq!(
            registry.get("c2").expect("c2").state,
            ChildSessionState::Failed
        );
    }

    #[test]
    fn completed_after_status_changed_preserves_result_summary() {
        // The richer StatusChanged carries the result_summary; the later,
        // coarser Completed must not clobber it back to None.
        let registry = SubagentRegistry::new();
        registry.apply_envelope(&spawned_envelope(1, "c1"));
        registry.apply_envelope(&status_changed_envelope(
            2,
            "c1",
            ChildSessionState::Completed,
        ));
        registry.apply_envelope(&completed_envelope(3, "c1", "completed"));
        let entry = registry.get("c1").expect("entry");
        assert_eq!(entry.state, ChildSessionState::Completed);
        assert_eq!(entry.result_summary.as_deref(), Some("summary-c1"));
        assert_eq!(entry.alias.as_deref(), Some("fixer"));
        assert_eq!(entry.generation, 1);
    }

    #[test]
    fn status_changed_for_unknown_child_is_ignored() {
        let registry = SubagentRegistry::new();
        registry.apply_envelope(&status_changed_envelope(
            1,
            "ghost",
            ChildSessionState::Running,
        ));
        assert!(registry.list().is_empty());
    }

    #[test]
    fn list_preserves_spawn_order() {
        let registry = SubagentRegistry::new();
        for child in ["a", "b", "c"] {
            registry.record_spawned("p", child, "t", "/ws".into(), None);
        }
        // Terminal transitions must not reorder entries.
        registry.record_terminal("a", ChildSessionState::Completed, Some("done".into()));
        let ids: Vec<String> = registry.list().into_iter().map(|e| e.session_id).collect();
        assert_eq!(ids, vec!["a".to_string(), "b".into(), "c".into()]);
    }

    // ------------------------------------------------------------------
    // P2 chunk B — live handles, alias resolution, control lease
    // ------------------------------------------------------------------

    #[test]
    fn record_handles_stores_cancel_flag_and_inbox() {
        let registry = SubagentRegistry::new();
        registry.record_spawned("p", "c1", "t", "/ws".into(), None);
        let flag = Arc::new(AtomicBool::new(false));
        let (inbox_tx, _inbox_rx) = mpsc::channel(16);
        registry.record_handles("c1", flag.clone(), inbox_tx);

        let entry = registry.get("c1").expect("entry");
        let stored = entry.cancel_flag.expect("cancel flag recorded");
        assert!(Arc::ptr_eq(&stored, &flag));
        assert!(entry.inbox_tx.is_some(), "inbox sender recorded");
        assert_eq!(entry.cancel_reason, None);
    }

    #[test]
    fn record_handles_unknown_id_is_noop() {
        let registry = SubagentRegistry::new();
        registry.record_handles(
            "ghost",
            Arc::new(AtomicBool::new(false)),
            mpsc::channel(16).0,
        );
        assert!(registry.get("ghost").is_none());
        assert!(registry.list().is_empty());
    }

    #[test]
    fn record_cancel_requested_sets_reason_and_terminal_clears_it() {
        let registry = SubagentRegistry::new();
        registry.record_spawned("p", "c1", "t", "/ws".into(), None);
        registry.record_cancel_requested("c1", Some("wrong branch".into()));
        assert_eq!(
            registry.get("c1").expect("entry").cancel_reason.as_deref(),
            Some("wrong branch")
        );
        // Unknown id is a no-op.
        registry.record_cancel_requested("ghost", Some("x".into()));
        assert!(registry.get("ghost").is_none());

        // Terminal fold clears the live handles AND the spent reason.
        registry.record_handles("c1", Arc::new(AtomicBool::new(false)), mpsc::channel(16).0);
        registry.record_terminal(
            "c1",
            ChildSessionState::Cancelled,
            Some("cancelled: wrong branch".into()),
        );
        let entry = registry.get("c1").expect("entry after terminal");
        assert_eq!(entry.state, ChildSessionState::Cancelled);
        assert!(
            entry.cancel_flag.is_none(),
            "terminal tasks have no live handles"
        );
        assert!(entry.inbox_tx.is_none(), "terminal tasks have no inbox");
        assert_eq!(
            entry.cancel_reason, None,
            "reason is folded into the summary, then cleared"
        );
        assert_eq!(
            entry.result_summary.as_deref(),
            Some("cancelled: wrong branch")
        );
    }

    #[test]
    fn clear_handles_is_idempotent_and_safe_for_unknown() {
        let registry = SubagentRegistry::new();
        registry.record_spawned("p", "c1", "t", "/ws".into(), None);
        registry.record_handles("c1", Arc::new(AtomicBool::new(false)), mpsc::channel(16).0);
        registry.clear_handles("c1");
        registry.clear_handles("c1");
        registry.clear_handles("ghost");
        let entry = registry.get("c1").expect("entry");
        assert!(entry.cancel_flag.is_none());
        assert!(entry.inbox_tx.is_none());
    }

    /// Give a tracked child an alias (mirrors a `ChildSessionStatusChanged`
    /// fold carrying one).
    fn aliased(registry: &SubagentRegistry, child: &str, alias: &str) {
        registry.apply_envelope(&EventEnvelope::new(
            1,
            AgentEvent::ChildSessionStatusChanged {
                parent_session_id: "parent".into(),
                child_session_id: child.into(),
                state: ChildSessionState::Running,
                generation: 0,
                alias: Some(alias.into()),
                result_summary: None,
            },
        ));
    }

    #[test]
    fn resolve_exact_session_id_wins_over_alias() {
        let registry = SubagentRegistry::new();
        registry.record_spawned("p", "c1", "t", "/ws".into(), None);
        registry.record_spawned("p", "c2", "t", "/ws".into(), None);
        aliased(&registry, "c2", "c1"); // alias collides with c1's id

        let entry = registry
            .resolve("c1")
            .expect("no ambiguity")
            .expect("found");
        assert_eq!(entry.session_id, "c1", "exact id match must win");
    }

    #[test]
    fn resolve_by_unique_alias() {
        let registry = SubagentRegistry::new();
        registry.record_spawned("p", "c1", "t", "/ws".into(), None);
        aliased(&registry, "c1", "fixer");

        let entry = registry
            .resolve("fixer")
            .expect("no ambiguity")
            .expect("found");
        assert_eq!(entry.session_id, "c1");
        // The id still resolves too.
        assert_eq!(
            registry
                .resolve("c1")
                .expect("id")
                .expect("found")
                .session_id,
            "c1"
        );
    }

    #[test]
    fn resolve_unknown_returns_none() {
        let registry = SubagentRegistry::new();
        assert!(registry.resolve("ghost").expect("no error").is_none());
    }

    #[test]
    fn resolve_ambiguous_alias_errors_with_count() {
        let registry = SubagentRegistry::new();
        registry.record_spawned("p", "c1", "t", "/ws".into(), None);
        registry.record_spawned("p", "c2", "t", "/ws".into(), None);
        registry.record_spawned("p", "c3", "t", "/ws".into(), None);
        aliased(&registry, "c1", "fixer");
        aliased(&registry, "c2", "fixer");
        aliased(&registry, "c3", "fixer");

        let err = registry
            .resolve("fixer")
            .expect_err("3 aliases must be ambiguous");
        assert!(
            err.contains("ambiguous alias 'fixer'") && err.contains("3 tasks"),
            "error must name the alias and the count: {err}"
        );
    }

    #[test]
    fn control_lease_is_exclusive_and_released_on_drop() {
        let registry = SubagentRegistry::new();
        registry.record_spawned("p", "c1", "t", "/ws".into(), None);

        let lease = registry
            .try_acquire_lease("c1")
            .expect("first acquisition succeeds");
        assert!(
            registry.try_acquire_lease("c1").is_none(),
            "second acquisition while held must fail"
        );
        drop(lease);
        assert!(
            registry.try_acquire_lease("c1").is_some(),
            "lease must be released on drop"
        );
    }

    #[test]
    fn control_lease_per_session_not_global() {
        let registry = SubagentRegistry::new();
        registry.record_spawned("p", "c1", "t", "/ws".into(), None);
        registry.record_spawned("p", "c2", "t", "/ws".into(), None);
        let _lease_a = registry.try_acquire_lease("c1").expect("c1");
        assert!(
            registry.try_acquire_lease("c2").is_some(),
            "leases are per-task; c2 is independent"
        );
    }

    fn session_state(messages: Vec<Message>, status: SessionStatus) -> SessionState {
        let now = chrono::Utc::now();
        SessionState {
            meta: SessionMeta {
                id: "c1".into(),
                created_at: now,
                updated_at: now,
                workspace: "/tmp/ws".into(),
                model: "m".into(),
                status,
                pid: None,
                socket_path: None,
                worktree_path: None,
                branch: None,
                base_branch: None,
                parent_session_id: None,
                child_session_ids: Vec::new(),
                inherited_summary: None,
                spawn_reason: None,
                session_summary: None,
                session_title: None,
                orchestration: None,
                agent_name: None,
            },
            messages,
            total_input_tokens: 0,
            total_output_tokens: 0,
            estimated_cost_usd: 0.0,
        }
    }

    #[test]
    fn last_assistant_text_picks_last_non_empty_skipping_trailing_noise() {
        // Trailing tool message + whitespace assistant must be skipped.
        let state = session_state(
            vec![
                Message::user("do it"),
                Message::assistant("first answer"),
                Message::assistant("   \n  "),
                Message::tool("call-1", "tool output"),
                Message::assistant("final answer"),
                Message::tool("call-2", "trailing tool output"),
            ],
            SessionStatus::Completed,
        );
        assert_eq!(last_assistant_text(&state).as_deref(), Some("final answer"));
    }

    #[test]
    fn last_assistant_text_joins_text_parts() {
        let assistant_parts = Message {
            role: Role::Assistant,
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "line one".into(),
                },
                ContentPart::Image {
                    media_type: "image/png".into(),
                    path: "a.png".into(),
                },
                ContentPart::Text {
                    text: "line two".into(),
                },
            ]),
            tool_call_id: None,
            tool_calls: None,
            reasoning_content: None,
        };
        let state = session_state(vec![assistant_parts], SessionStatus::Completed);
        assert_eq!(
            last_assistant_text(&state).as_deref(),
            Some("line one\nline two")
        );
    }

    #[test]
    fn last_assistant_text_none_when_no_assistant_messages() {
        let state = session_state(vec![Message::user("hi")], SessionStatus::Completed);
        assert_eq!(last_assistant_text(&state), None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn control_consumer_status_from_registry_and_unknown() {
        let (tx, rx) = mpsc::channel(4);
        let registry = Arc::new(SubagentRegistry::new());
        registry.record_spawned("p", "c1", "build it", "/ws".into(), Some("nca/c1".into()));
        // Store with no sessions: unknown ids must fail through it.
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SessionStore::new(dir.path());
        let _consumer = subagent_control_consumer(rx, registry, store);

        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(SubagentControlRequest::Status {
            session_id: "c1".into(),
            reply: reply_tx,
        })
        .await
        .expect("send");
        let resp = reply_rx.await.expect("reply");
        assert!(resp.ok);
        assert_eq!(resp.state, ChildSessionState::Running);
        assert_eq!(resp.task.as_deref(), Some("build it"));
        assert_eq!(resp.workspace.as_deref(), Some("/ws"));

        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(SubagentControlRequest::Status {
            session_id: "ghost".into(),
            reply: reply_tx,
        })
        .await
        .expect("send");
        let resp = reply_rx.await.expect("reply");
        assert!(!resp.ok);
        assert_eq!(
            resp.error_message.as_deref(),
            Some("unknown subagent task id")
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn control_consumer_status_falls_back_to_store() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SessionStore::new(dir.path());
        let state = session_state(
            vec![Message::user("hi"), Message::assistant("done")],
            SessionStatus::Completed,
        );
        store.save(&state).await.expect("save");

        let (tx, rx) = mpsc::channel(4);
        let registry = Arc::new(SubagentRegistry::new());
        let _consumer = subagent_control_consumer(rx, registry, store);

        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(SubagentControlRequest::Status {
            session_id: "c1".into(),
            reply: reply_tx,
        })
        .await
        .expect("send");
        let resp = reply_rx.await.expect("reply");
        assert!(resp.ok);
        assert_eq!(resp.state, ChildSessionState::Completed);
        assert_eq!(resp.task, None, "task text is not persisted on the child");
        assert_eq!(resp.workspace.as_deref(), Some("/tmp/ws"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn control_consumer_result_running_vs_terminal() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SessionStore::new(dir.path());
        // A terminal child json with a real assistant message.
        let state = session_state(
            vec![Message::user("hi"), Message::assistant("final words")],
            SessionStatus::Completed,
        );
        store.save(&state).await.expect("save");

        let (tx, rx) = mpsc::channel(4);
        let registry = Arc::new(SubagentRegistry::new());
        // Live entry: still running.
        registry.record_spawned("p", "running-1", "t", "/ws".into(), None);
        // Terminal entry pointing at the saved json.
        registry.record_spawned("p", "c1", "t", "/ws".into(), None);
        registry.record_terminal("c1", ChildSessionState::Completed, Some("ok".into()));
        let _consumer = subagent_control_consumer(rx, registry, store);

        // Running → note, no output.
        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(SubagentControlRequest::Result {
            session_id: "running-1".into(),
            reply: reply_tx,
        })
        .await
        .expect("send");
        let resp = reply_rx.await.expect("reply");
        assert!(resp.ok);
        assert_eq!(resp.state, ChildSessionState::Running);
        assert_eq!(resp.output, None);
        assert_eq!(resp.note.as_deref(), Some("task is still running"));

        // Terminal → full last assistant message.
        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(SubagentControlRequest::Result {
            session_id: "c1".into(),
            reply: reply_tx,
        })
        .await
        .expect("send");
        let resp = reply_rx.await.expect("reply");
        assert!(resp.ok);
        assert_eq!(resp.output.as_deref(), Some("final words"));
        assert_eq!(resp.result_summary.as_deref(), Some("ok"));

        // Unknown id → error.
        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(SubagentControlRequest::Result {
            session_id: "ghost".into(),
            reply: reply_tx,
        })
        .await
        .expect("send");
        let resp = reply_rx.await.expect("reply");
        assert!(!resp.ok);
        assert!(resp.error_message.is_some());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn control_consumer_write_operations_reply_not_available_yet() {
        // P2 chunk A: the Message/Cancel/Revive wire types exist, but this
        // build's consumer cannot execute them (the control lease lands in
        // chunk B) — every write op must fail loudly instead of pretending
        // success, even for a registry-known, still-running child.
        let (tx, rx) = mpsc::channel(8);
        let registry = Arc::new(SubagentRegistry::new());
        registry.record_spawned("p", "c1", "t", "/ws".into(), None);
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SessionStore::new(dir.path());
        let _consumer = subagent_control_consumer(rx, registry, store);

        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(SubagentControlRequest::Message {
            session_id: "c1".into(),
            text: "pivot to tests".into(),
            reply: reply_tx,
        })
        .await
        .expect("send");
        let resp = reply_rx.await.expect("reply");
        assert!(!resp.ok);
        assert_eq!(resp.session_id, "c1");
        assert_eq!(
            resp.error_message.as_deref(),
            Some("task control operation is not available in this build")
        );

        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(SubagentControlRequest::Cancel {
            session_id: "c1".into(),
            reason: Some("wrong branch".into()),
            reply: reply_tx,
        })
        .await
        .expect("send");
        let resp = reply_rx.await.expect("reply");
        assert!(!resp.ok);
        assert_eq!(
            resp.error_message.as_deref(),
            Some("task control operation is not available in this build")
        );

        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(SubagentControlRequest::Revive {
            session_id: "c1".into(),
            prompt: "finish the tests".into(),
            reply: reply_tx,
        })
        .await
        .expect("send");
        let resp = reply_rx.await.expect("reply");
        assert!(!resp.ok);
        assert_eq!(
            resp.error_message.as_deref(),
            Some("task control operation is not available in this build")
        );
    }
}
