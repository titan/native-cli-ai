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
    /// Specialist agent name the child was spawned with (revive rebuilds
    /// its config routing from this).
    pub specialist: Option<String>,
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
            specialist: None,
        });
    }

    /// Attach a parent-scoped alias to a tracked child (spawn path).
    /// The `ChildSessionSpawned` event has no alias field, so the alias is
    /// applied post-record and surfaced through the running
    /// `ChildSessionStatusChanged` event (which DOES carry it, keeping the
    /// registry re-derivable from the event log at resume). No-op for
    /// unknown ids; a blank alias is ignored.
    pub fn set_alias(&self, child_session_id: &str, alias: Option<&str>) {
        let Some(alias) = alias.filter(|a| !a.trim().is_empty()) else {
            return;
        };
        let mut entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        let Some(entry) = entries
            .iter_mut()
            .find(|e| e.session_id == child_session_id)
        else {
            return;
        };
        entry.alias = Some(alias.to_string());
    }

    /// Record the child's worktree path on its entry (spawn path). Needed
    /// by `task_revive` to reuse the retained worktree (cancel never deletes
    /// it) and by tests pinning worktree identity across generations.
    pub fn set_worktree(&self, child_session_id: &str, worktree_path: Option<String>) {
        let mut entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        let Some(entry) = entries
            .iter_mut()
            .find(|e| e.session_id == child_session_id)
        else {
            return;
        };
        entry.worktree_path = worktree_path;
    }

    /// Record the specialist name a child was spawned with, so `task_revive`
    /// can rebuild the child config with the same routing treatment
    /// (`apply_child_routing`).
    pub fn set_specialist(&self, child_session_id: &str, specialist: Option<String>) {
        let mut entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        let Some(entry) = entries
            .iter_mut()
            .find(|e| e.session_id == child_session_id)
        else {
            return;
        };
        entry.specialist = specialist;
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

    /// Fold a revive transition: the entry flips back to `Running` with a
    /// bumped `generation` and a clean slate — the old `result_summary`/
    /// `cancel_reason` are spent, live handles were already cleared by the
    /// terminal fold and are re-recorded by the caller right after the
    /// resumed supervisor exists. Returns the new generation, or `None`
    /// for ids the registry does not track.
    ///
    /// Alias, specialist, and worktree path SURVIVE — revive reuses the
    /// retained worktree and keeps addressing the task by its alias.
    pub fn record_revive(&self, child_session_id: &str) -> Option<u64> {
        let mut entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        let entry = entries
            .iter_mut()
            .find(|e| e.session_id == child_session_id)?;
        entry.state = ChildSessionState::Running;
        entry.generation += 1;
        entry.result_summary = None;
        entry.cancel_reason = None;
        Some(entry.generation)
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

    /// One-line hint of the tasks this registry tracks, appended to
    /// unknown-id errors so the model can self-correct without another
    /// round-trip. Empty registry says so plainly; otherwise entries are
    /// rendered in [`Self::list`] order (spawn order, oldest-first) as
    /// `<session_id> (alias <alias>) [state]` with lowercase state words,
    /// capped at the 6 most recent (`… +N earlier` when older entries are
    /// elided) and hard-truncated to ~400 chars.
    pub(crate) fn known_tasks_hint(&self) -> String {
        let entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        if entries.is_empty() {
            return "no subagent tasks are registered in this session".to_string();
        }
        let skip = entries.len().saturating_sub(6);
        let mut parts: Vec<String> = entries[skip..]
            .iter()
            .map(|entry| {
                let mut rendered = entry.session_id.clone();
                if let Some(alias) = &entry.alias {
                    rendered.push_str(" (alias ");
                    rendered.push_str(alias);
                    rendered.push(')');
                }
                rendered.push_str(" [");
                rendered.push_str(state_tag(entry.state));
                rendered.push(']');
                rendered
            })
            .collect();
        if skip > 0 {
            parts.push(format!("… +{skip} earlier"));
        }
        nca_core::agent::truncate_str(&format!("known tasks: {}", parts.join(", ")), 400)
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

/// Lowercase state word for the known-tasks hint (model-facing prose —
/// never `Debug` formatting).
fn state_tag(state: ChildSessionState) -> &'static str {
    match state {
        ChildSessionState::Pending => "pending",
        ChildSessionState::Running => "running",
        ChildSessionState::Completed => "completed",
        ChildSessionState::Cancelled => "cancelled",
        ChildSessionState::Failed => "failed",
    }
}

/// Unknown-id control error: the offending id plus the registry's
/// known-tasks hint, so the model can self-correct. Every control path
/// (`task_status`/`task_result`/`task_message`/`task_cancel`/
/// `task_revive`) reports resolution misses through this one helper for a
/// consistent contract.
pub(crate) fn unknown_task_error(registry: &SubagentRegistry, id: &str) -> String {
    format!(
        "unknown subagent task id '{id}'; {}",
        registry.known_tasks_hint()
    )
}

/// Ambiguous-alias control error: the original resolve message as the
/// prefix plus the known-tasks hint, so the model can disambiguate by
/// exact session id.
pub(crate) fn ambiguous_task_error(registry: &SubagentRegistry, message: String) -> String {
    format!("{message}; {}", registry.known_tasks_hint())
}

/// Consume control requests (`task_status`/`task_result`/`task_message`/
/// `task_cancel`/`task_revive`) against the registry,
/// with a read-only `SessionStore::load` fallback for ids the registry
/// never saw (e.g. spawned before this process started). Spawned by
/// `Supervisor::create`; replies ride per-request oneshot channels.
/// `event_tx` (the parent's bounded event channel) receives
/// `ChildMessageQueued` envelopes for accepted/refused steering attempts.
///
/// **Invariant:** never calls `session_store.save` — introspection and
/// control signaling are strictly non-persisting (single-writer preserved:
/// cancel flips the child's own in-memory flag; the child's supervisor
/// persists its own terminal state).
pub fn subagent_control_consumer(
    mut control_rx: mpsc::Receiver<SubagentControlRequest>,
    registry: Arc<SubagentRegistry>,
    session_store: SessionStore,
    config: nca_common::config::NcaConfig,
    workspace_root: std::path::PathBuf,
    event_tx: Option<mpsc::Sender<AgentEvent>>,
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
                                unknown_task_error(&registry, &session_id),
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
                                unknown_task_error(&registry, &session_id),
                            ),
                        },
                    };
                    let _ = reply.send(response);
                }
                SubagentControlRequest::Message {
                    session_id,
                    text,
                    reply,
                } => {
                    let (response, parent_id, accepted) =
                        handle_message_request(&registry, &session_id, text).await;
                    if let (Some(tx), Some(parent)) = (event_tx.as_ref(), parent_id) {
                        // Bounded channel: try_send only (never block the
                        // control loop on a slow UI consumer).
                        let _ = tx.try_send(AgentEvent::ChildMessageQueued {
                            parent_session_id: parent,
                            child_session_id: session_id.clone(),
                            accepted,
                        });
                    }
                    let _ = reply.send(response);
                }
                SubagentControlRequest::Cancel {
                    session_id,
                    reason,
                    reply,
                } => {
                    let response = handle_cancel_request(&registry, &session_id, reason).await;
                    let _ = reply.send(response);
                }
                // P2 C2: `task_revive` runs LONG (cancel-wait + resume + a
                // full child turn) — execute it on its own tokio task so
                // this loop keeps serving status/result/message/cancel;
                // the reply rides the request's oneshot from inside that
                // task. The lease is held for the whole sequence inside
                // `handle_revive_request`.
                SubagentControlRequest::Revive {
                    session_id,
                    prompt,
                    reply,
                } => {
                    let registry = Arc::clone(&registry);
                    let config = config.clone();
                    let workspace_root = workspace_root.clone();
                    let event_tx = event_tx.clone();
                    tokio::spawn(async move {
                        let response = crate::subagent::handle_revive_request(
                            registry,
                            config,
                            workspace_root,
                            event_tx,
                            None,
                            session_id,
                            prompt,
                        )
                        .await;
                        let _ = reply.send(response);
                    });
                }
            }
        }
    })
}

/// Execute a `task_message` steering request against the registry.
/// Returns `(reply, parent_session_id, accepted)`; the parent id is `None`
/// when the target could not be attributed to a parent (unknown id), in
/// which case no `ChildMessageQueued` event is emitted.
async fn handle_message_request(
    registry: &SubagentRegistry,
    session_id: &str,
    text: String,
) -> (SubagentControlResponse, Option<String>, bool) {
    let entry = match registry.resolve(session_id) {
        Ok(Some(entry)) => entry,
        Ok(None) => {
            return (
                SubagentControlResponse::unknown(
                    session_id,
                    unknown_task_error(registry, session_id),
                ),
                None,
                false,
            );
        }
        Err(message) => {
            return (
                SubagentControlResponse::unknown(
                    session_id,
                    ambiguous_task_error(registry, message),
                ),
                None,
                false,
            );
        }
    };
    let parent = Some(entry.parent_session_id.clone());
    let base = |state, note: Option<String>, ok: bool| SubagentControlResponse {
        session_id: entry.session_id.clone(),
        state,
        task: None,
        workspace: None,
        branch: None,
        result_summary: None,
        output: None,
        note,
        ok,
        error_message: None,
        generation: None,
    };
    if entry.state != ChildSessionState::Running {
        return (
            base(entry.state, Some("task is not running".into()), false),
            parent,
            false,
        );
    }
    let Some(inbox_tx) = entry.inbox_tx else {
        return (
            base(entry.state, Some("task is not running".into()), false),
            parent,
            false,
        );
    };
    let (response, accepted) = match inbox_tx.try_send(InboxItem::Steering { text }) {
        Ok(()) => (
            base(
                entry.state,
                Some("queued for delivery at the child's next step boundary".into()),
                true,
            ),
            true,
        ),
        Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => (
            base(
                entry.state,
                Some("child inbox is full (16) — retry later".into()),
                false,
            ),
            false,
        ),
        Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => (
            base(entry.state, Some("child inbox closed".into()), false),
            false,
        ),
    };
    (response, parent, accepted)
}

/// Execute a `task_cancel` request: acquire the control lease, flip the
/// child's cooperative cancel flag, record the reason, and release the
/// lease immediately — the abort itself is asynchronous (the child's
/// stream/tool loops poll the flag every 25–50ms).
async fn handle_cancel_request(
    registry: &SubagentRegistry,
    session_id: &str,
    reason: Option<String>,
) -> SubagentControlResponse {
    let entry = match registry.resolve(session_id) {
        Ok(Some(entry)) => entry,
        Ok(None) => {
            return SubagentControlResponse::unknown(
                session_id,
                unknown_task_error(registry, session_id),
            );
        }
        Err(message) => {
            return SubagentControlResponse::unknown(
                session_id,
                ambiguous_task_error(registry, message),
            );
        }
    };
    let base = |state, note: Option<String>, ok: bool| SubagentControlResponse {
        session_id: entry.session_id.clone(),
        state,
        task: None,
        workspace: None,
        branch: None,
        result_summary: None,
        output: None,
        note,
        ok,
        error_message: None,
        generation: None,
    };
    let Some(_lease) = registry.try_acquire_lease(&entry.session_id) else {
        return base(
            entry.state,
            Some("another control operation is in flight for this task".into()),
            false,
        );
    };
    let Some(cancel_flag) = entry.cancel_flag else {
        return base(entry.state, Some("task is not running".into()), false);
    };
    if entry.state != ChildSessionState::Running {
        return base(entry.state, Some("task is not running".into()), false);
    }
    cancel_flag.store(true, std::sync::atomic::Ordering::SeqCst);
    registry.record_cancel_requested(&entry.session_id, reason);
    let response = base(
        entry.state,
        Some(
            "cancel requested; the child aborts cooperatively at its next poll (≤50ms). \
             Worktree and branch are retained for revive."
                .into(),
        ),
        true,
    );
    // The lease guarded only the flag-set; release it now (explicit drop
    // documents that the abort itself is asynchronous and un-leased).
    drop(_lease);
    response
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

/// Seed registry entries from persisted session lineage
/// (`SessionMeta::child_session_ids`) — the resume path's second recovery
/// channel after the event-log envelope fold. For each lineage id the
/// registry does not already track, the child's json is loaded read-only
/// and recorded as a projection: identity fields from the child meta,
/// state mapped from its persisted `SessionStatus` (folded straight to a
/// terminal-truthful entry — handles are `None` by construction, no live
/// child supervisor exists at resume, so a seeded `Running` entry can
/// still be inspected but never signalled). Nothing is written back — the
/// child json stays owned by whichever supervisor spawned it
/// (single-writer invariant).
///
/// Ids with no json on disk (or an unreadable one) are skipped silently:
/// there is nothing truthful to seed, and the unknown-id hint from
/// [`unknown_task_error`] explains the gap to the model.
pub(crate) async fn seed_registry_from_lineage(
    registry: &SubagentRegistry,
    session_store: &SessionStore,
    parent_session_id: &str,
    child_session_ids: &[String],
) {
    for id in child_session_ids {
        if registry.get(id).is_some() {
            continue;
        }
        let Ok(state) = session_store.load(id).await else {
            continue;
        };
        registry.record_spawned(
            state
                .meta
                .parent_session_id
                .as_deref()
                .unwrap_or(parent_session_id),
            id,
            state.meta.spawn_reason.as_deref().unwrap_or(""),
            state.meta.workspace.display().to_string(),
            state.meta.branch.clone(),
        );
        registry.record_terminal(
            id,
            ChildSessionState::from_session_status(state.meta.status.clone()),
            state.meta.session_summary.clone(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nca_common::config::NcaConfig;
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

    #[test]
    fn known_tasks_hint_empty_registry_says_so() {
        assert_eq!(
            SubagentRegistry::new().known_tasks_hint(),
            "no subagent tasks are registered in this session"
        );
    }

    #[test]
    fn known_tasks_hint_lists_ids_aliases_and_state_words() {
        let registry = SubagentRegistry::new();
        registry.record_spawned("p", "c1", "t", "/ws".into(), None);
        registry.record_spawned("p", "c2", "t", "/ws".into(), None);
        registry.set_alias("c2", Some("fixer"));
        registry.record_terminal("c1", ChildSessionState::Completed, None);
        assert_eq!(
            registry.known_tasks_hint(),
            "known tasks: c1 [completed], c2 (alias fixer) [running]"
        );
    }

    #[test]
    fn known_tasks_hint_caps_at_six_most_recent() {
        let registry = SubagentRegistry::new();
        for n in 1..=8 {
            registry.record_spawned("p", &format!("child-{n:02}"), "t", "/ws".into(), None);
        }
        let hint = registry.known_tasks_hint();
        assert!(hint.starts_with("known tasks: child-03"), "got: {hint}");
        assert!(hint.contains("child-08"), "latest entry must show: {hint}");
        assert!(hint.ends_with("… +2 earlier"), "got: {hint}");
        assert!(!hint.contains("child-01") && !hint.contains("child-02"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn seed_registry_from_lineage_projects_persisted_children() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SessionStore::new(dir.path());

        // A completed child json on disk with lineage + summary fields set.
        let mut child = session_state(vec![Message::user("hi")], SessionStatus::Completed);
        child.meta.id = "child-1".into();
        child.meta.parent_session_id = Some("parent-1".into());
        child.meta.spawn_reason = Some("write the tests".into());
        child.meta.session_summary = Some("all good".into());
        store.save(&child).await.expect("save child");

        let registry = SubagentRegistry::new();
        // An id the registry already tracks must be left untouched.
        registry.record_spawned("parent-1", "tracked-1", "live task", "/ws".into(), None);

        seed_registry_from_lineage(
            &registry,
            &store,
            "parent-1",
            &[
                "child-1".to_string(),
                "no-json-child".to_string(),
                "tracked-1".to_string(),
            ],
        )
        .await;

        // child-1: seeded as a terminal-only projection from the json.
        let entry = registry.get("child-1").expect("seeded from lineage");
        assert_eq!(entry.state, ChildSessionState::Completed);
        assert_eq!(entry.task, "write the tests", "task text from spawn_reason");
        assert_eq!(entry.parent_session_id, "parent-1");
        assert_eq!(entry.workspace, "/tmp/ws");
        assert_eq!(entry.result_summary.as_deref(), Some("all good"));
        assert!(
            entry.cancel_flag.is_none(),
            "seeded entries never carry live handles"
        );
        assert!(
            entry.inbox_tx.is_none(),
            "seeded entries never carry an inbox"
        );

        // No json on disk → nothing truthful to seed → stays unknown.
        assert!(registry.get("no-json-child").is_none());

        // Already-tracked id → not overwritten by the seeding.
        let tracked = registry.get("tracked-1").expect("still tracked");
        assert_eq!(tracked.task, "live task");
        assert_eq!(tracked.state, ChildSessionState::Running);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn seed_registry_from_lineage_falls_back_to_passed_parent_id() {
        // Child json without parent_session_id (legacy): the passed parent
        // id is used so the entry stays attributable.
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SessionStore::new(dir.path());
        let child = session_state(Vec::new(), SessionStatus::Cancelled);
        store.save(&child).await.expect("save child");

        let registry = SubagentRegistry::new();
        seed_registry_from_lineage(&registry, &store, "fallback-parent", &["c1".to_string()]).await;

        let entry = registry.get("c1").expect("seeded");
        assert_eq!(entry.parent_session_id, "fallback-parent");
        assert_eq!(entry.state, ChildSessionState::Cancelled);
        assert_eq!(entry.task, "", "missing spawn_reason seeds empty task text");
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
    fn record_revive_bumps_generation_and_resets_terminal_fields() {
        // P2 C2: revive flips a terminal entry back to Running with a
        // bumped generation and a clean slate — the old summary/reason are
        // spent, and fresh handles arrive right after via record_handles.
        let registry = SubagentRegistry::new();
        registry.record_spawned("p", "c1", "t", "/ws".into(), None);
        registry.set_alias("c1", Some("fixer-x"));
        registry.set_worktree("c1", Some("/ws/.nca/worktrees/c1".into()));
        registry.set_specialist("c1", Some("fixer".into()));
        registry.record_cancel_requested("c1", Some("wrong branch".into()));
        registry.record_terminal(
            "c1",
            ChildSessionState::Cancelled,
            Some("cancelled: wrong branch".into()),
        );

        let generation = registry.record_revive("c1").expect("revive generation");
        assert_eq!(generation, 1, "first revive bumps 0 → 1");
        let entry = registry.get("c1").expect("entry");
        assert_eq!(entry.state, ChildSessionState::Running);
        assert_eq!(entry.generation, 1);
        assert_eq!(entry.result_summary, None, "old summary is spent");
        assert_eq!(entry.cancel_reason, None);
        assert_eq!(entry.alias.as_deref(), Some("fixer-x"), "alias survives");
        assert_eq!(entry.specialist.as_deref(), Some("fixer"));
        assert_eq!(
            entry.worktree_path.as_deref(),
            Some("/ws/.nca/worktrees/c1")
        );

        let generation = registry.record_revive("c1").expect("second revive");
        assert_eq!(generation, 2, "generation keeps counting up");
        assert_eq!(registry.get("c1").expect("entry").generation, 2);
    }

    #[test]
    fn record_revive_unknown_id_is_none() {
        let registry = SubagentRegistry::new();
        assert!(registry.record_revive("ghost").is_none());
    }

    #[test]
    fn setters_ignore_unknown_ids_and_blank_alias() {
        let registry = SubagentRegistry::new();
        registry.set_alias("ghost", Some("x"));
        registry.set_worktree("ghost", Some("/wt".into()));
        registry.set_specialist("ghost", Some("fixer".into()));
        assert!(registry.get("ghost").is_none(), "no entry is created");

        registry.record_spawned("p", "c1", "t", "/ws".into(), None);
        registry.set_alias("c1", Some("   "));
        assert_eq!(registry.get("c1").expect("entry").alias, None);
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
        let _consumer = subagent_control_consumer(
            rx,
            registry,
            store,
            NcaConfig::default(),
            std::path::PathBuf::from("/tmp/nca-registry-test-ws"),
            None,
        );

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
        let error = resp.error_message.expect("error");
        assert!(
            error.starts_with("unknown subagent task id 'ghost';"),
            "error must name the offending id: {error}"
        );
        assert!(
            error.contains("c1"),
            "hint must list the registered task: {error}"
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
        let _consumer = subagent_control_consumer(
            rx,
            registry,
            store,
            NcaConfig::default(),
            std::path::PathBuf::from("/tmp/nca-registry-test-ws"),
            None,
        );

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
        let _consumer = subagent_control_consumer(
            rx,
            registry,
            store,
            NcaConfig::default(),
            std::path::PathBuf::from("/tmp/nca-registry-test-ws"),
            None,
        );

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
    async fn control_consumer_revive_unknown_id_is_unknown() {
        // P2 C2: revive is live in the consumer — an unknown id replies
        // unknown (the stub "not available" arm is gone).
        let (tx, rx) = mpsc::channel(8);
        let registry = Arc::new(SubagentRegistry::new());
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SessionStore::new(dir.path());
        let _consumer = subagent_control_consumer(
            rx,
            registry,
            store,
            NcaConfig::default(),
            std::path::PathBuf::from("/tmp/nca-registry-test-ws"),
            None,
        );

        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(SubagentControlRequest::Revive {
            session_id: "ghost".into(),
            prompt: "finish the tests".into(),
            reply: reply_tx,
        })
        .await
        .expect("send");
        let resp = reply_rx.await.expect("reply");
        assert!(!resp.ok);
        let error = resp.error_message.expect("error");
        assert!(
            error.starts_with("unknown subagent task id 'ghost';"),
            "error must name the offending id: {error}"
        );
        assert!(
            error.contains("no subagent tasks are registered"),
            "empty-registry hint must say so plainly: {error}"
        );
    }

    // ------------------------------------------------------------------
    // P2 chunk B — task_message / task_cancel through the consumer
    // ------------------------------------------------------------------

    /// Registry with one child in `state`, carrying real live handles bound
    /// to a fresh bounded inbox channel.
    fn running_child_with_inbox(
        registry: &SubagentRegistry,
        id: &str,
        cap: usize,
    ) -> (Arc<AtomicBool>, mpsc::Receiver<InboxItem>) {
        registry.record_spawned("parent", id, "t", "/ws".into(), None);
        let flag = Arc::new(AtomicBool::new(false));
        let (inbox_tx, inbox_rx) = mpsc::channel(cap);
        registry.record_handles(id, flag.clone(), inbox_tx);
        (flag, inbox_rx)
    }

    async fn send_message(
        tx: &mpsc::Sender<SubagentControlRequest>,
        id: &str,
        text: &str,
    ) -> SubagentControlResponse {
        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(SubagentControlRequest::Message {
            session_id: id.into(),
            text: text.into(),
            reply: reply_tx,
        })
        .await
        .expect("send");
        reply_rx.await.expect("reply")
    }

    async fn send_cancel(
        tx: &mpsc::Sender<SubagentControlRequest>,
        id: &str,
        reason: Option<&str>,
    ) -> SubagentControlResponse {
        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(SubagentControlRequest::Cancel {
            session_id: id.into(),
            reason: reason.map(String::from),
            reply: reply_tx,
        })
        .await
        .expect("send");
        reply_rx.await.expect("reply")
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn message_to_running_child_delivers_steering_and_emits_event() {
        let (tx, rx) = mpsc::channel(8);
        let (event_tx, mut event_rx) = mpsc::channel(16);
        let registry = Arc::new(SubagentRegistry::new());
        let (_flag, mut inbox_rx) = running_child_with_inbox(&registry, "c1", 16);
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SessionStore::new(dir.path());
        let _consumer = subagent_control_consumer(
            rx,
            registry.clone(),
            store,
            NcaConfig::default(),
            std::path::PathBuf::from("/tmp/nca-registry-test-ws"),
            Some(event_tx),
        );

        let resp = send_message(&tx, "c1", "pivot to tests").await;
        assert!(resp.ok, "steering to a running child must be accepted");
        assert_eq!(resp.state, ChildSessionState::Running);
        assert_eq!(
            resp.note.as_deref(),
            Some("queued for delivery at the child's next step boundary")
        );

        match inbox_rx.try_recv() {
            Ok(InboxItem::Steering { text }) => assert_eq!(text, "pivot to tests"),
            other => panic!("steering must land on the child inbox: {other:?}"),
        }

        match event_rx.try_recv() {
            Ok(AgentEvent::ChildMessageQueued {
                parent_session_id,
                child_session_id,
                accepted,
            }) => {
                assert_eq!(parent_session_id, "parent");
                assert_eq!(child_session_id, "c1");
                assert!(accepted);
            }
            other => panic!("ChildMessageQueued must be emitted on success: {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn message_to_terminal_child_reports_not_running_with_accepted_false() {
        let (tx, rx) = mpsc::channel(8);
        let registry = Arc::new(SubagentRegistry::new());
        registry.record_spawned("parent", "c1", "t", "/ws".into(), None);
        registry.record_terminal("c1", ChildSessionState::Completed, Some("done".into()));
        let (event_tx, mut event_rx) = mpsc::channel(16);
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SessionStore::new(dir.path());
        let _consumer = subagent_control_consumer(
            rx,
            registry,
            store,
            NcaConfig::default(),
            std::path::PathBuf::from("/tmp/nca-registry-test-ws"),
            Some(event_tx),
        );

        let resp = send_message(&tx, "c1", "steer").await;
        assert!(!resp.ok);
        assert_eq!(resp.state, ChildSessionState::Completed);
        assert_eq!(resp.note.as_deref(), Some("task is not running"));

        match event_rx.try_recv() {
            Ok(AgentEvent::ChildMessageQueued { accepted, .. }) => assert!(!accepted),
            other => panic!("ChildMessageQueued must be emitted on failure too: {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn message_to_full_inbox_reports_full_and_retry_hint() {
        let (tx, rx) = mpsc::channel(8);
        let registry = Arc::new(SubagentRegistry::new());
        // Capacity-1 inbox, pre-filled: the steering try_send must hit Full.
        let (_flag, _inbox_rx) = running_child_with_inbox(&registry, "c1", 1);
        registry
            .get("c1")
            .and_then(|e| e.inbox_tx)
            .expect("handle")
            .try_send(InboxItem::Steering {
                text: "occupant".into(),
            })
            .expect("pre-fill");
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SessionStore::new(dir.path());
        let _consumer = subagent_control_consumer(
            rx,
            registry,
            store,
            NcaConfig::default(),
            std::path::PathBuf::from("/tmp/nca-registry-test-ws"),
            None,
        );

        let resp = send_message(&tx, "c1", "overflow").await;
        assert!(!resp.ok);
        assert_eq!(
            resp.note.as_deref(),
            Some("child inbox is full (16) — retry later")
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn message_to_closed_inbox_reports_closed() {
        let (tx, rx) = mpsc::channel(8);
        let registry = Arc::new(SubagentRegistry::new());
        // Running state but the receiving half is already dropped.
        registry.record_spawned("parent", "c1", "t", "/ws".into(), None);
        let (inbox_tx, inbox_rx) = mpsc::channel(16);
        registry.record_handles("c1", Arc::new(AtomicBool::new(false)), inbox_tx);
        drop(inbox_rx);
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SessionStore::new(dir.path());
        let _consumer = subagent_control_consumer(
            rx,
            registry,
            store,
            NcaConfig::default(),
            std::path::PathBuf::from("/tmp/nca-registry-test-ws"),
            None,
        );

        let resp = send_message(&tx, "c1", "steer").await;
        assert!(!resp.ok);
        assert_eq!(resp.note.as_deref(), Some("child inbox closed"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn message_unknown_id_is_unknown() {
        let (tx, rx) = mpsc::channel(8);
        let registry = Arc::new(SubagentRegistry::new());
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SessionStore::new(dir.path());
        let _consumer = subagent_control_consumer(
            rx,
            registry,
            store,
            NcaConfig::default(),
            std::path::PathBuf::from("/tmp/nca-registry-test-ws"),
            None,
        );

        let resp = send_message(&tx, "ghost", "steer").await;
        assert!(!resp.ok);
        let error = resp.error_message.expect("error");
        assert!(
            error.starts_with("unknown subagent task id 'ghost';"),
            "error must name the offending id: {error}"
        );
        assert!(
            error.contains("no subagent tasks are registered"),
            "empty-registry hint must say so plainly: {error}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancel_running_child_sets_flag_records_reason_and_replies() {
        let (tx, rx) = mpsc::channel(8);
        let registry = Arc::new(SubagentRegistry::new());
        let (flag, _inbox_rx) = running_child_with_inbox(&registry, "c1", 16);
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SessionStore::new(dir.path());
        let _consumer = subagent_control_consumer(
            rx,
            registry.clone(),
            store,
            NcaConfig::default(),
            std::path::PathBuf::from("/tmp/nca-registry-test-ws"),
            None,
        );

        let resp = send_cancel(&tx, "c1", Some("wrong branch")).await;
        assert!(resp.ok, "cancel of a running child must be accepted");
        assert_eq!(resp.state, ChildSessionState::Running);
        let note = resp.note.as_deref().expect("note");
        assert!(
            note.contains("cancel requested") && note.contains("revive"),
            "note must explain the cooperative abort and retention: {note}"
        );
        assert!(flag.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(
            registry.get("c1").and_then(|e| e.cancel_reason).as_deref(),
            Some("wrong branch")
        );

        // The lease is released immediately after the flag-set: a second
        // cancel must NOT report an in-flight conflict.
        let resp2 = send_cancel(&tx, "c1", None).await;
        assert!(
            resp2.ok,
            "lease must be free for a follow-up cancel: {resp2:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancel_terminal_child_reports_not_running() {
        let (tx, rx) = mpsc::channel(8);
        let registry = Arc::new(SubagentRegistry::new());
        registry.record_spawned("parent", "c1", "t", "/ws".into(), None);
        registry.record_terminal("c1", ChildSessionState::Cancelled, Some("cancelled".into()));
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SessionStore::new(dir.path());
        let _consumer = subagent_control_consumer(
            rx,
            registry,
            store,
            NcaConfig::default(),
            std::path::PathBuf::from("/tmp/nca-registry-test-ws"),
            None,
        );

        let resp = send_cancel(&tx, "c1", None).await;
        assert!(!resp.ok);
        assert_eq!(resp.state, ChildSessionState::Cancelled);
        assert_eq!(resp.note.as_deref(), Some("task is not running"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancel_unknown_id_is_unknown() {
        let (tx, rx) = mpsc::channel(8);
        let registry = Arc::new(SubagentRegistry::new());
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SessionStore::new(dir.path());
        let _consumer = subagent_control_consumer(
            rx,
            registry,
            store,
            NcaConfig::default(),
            std::path::PathBuf::from("/tmp/nca-registry-test-ws"),
            None,
        );

        let resp = send_cancel(&tx, "ghost", None).await;
        assert!(!resp.ok);
        let error = resp.error_message.expect("error");
        assert!(
            error.starts_with("unknown subagent task id 'ghost';"),
            "error must name the offending id: {error}"
        );
        assert!(
            error.contains("no subagent tasks are registered"),
            "empty-registry hint must say so plainly: {error}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancel_while_lease_held_reports_in_flight() {
        let (tx, rx) = mpsc::channel(8);
        let registry = Arc::new(SubagentRegistry::new());
        let (flag, _inbox_rx) = running_child_with_inbox(&registry, "c1", 16);
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SessionStore::new(dir.path());
        let _consumer = subagent_control_consumer(
            rx,
            registry.clone(),
            store,
            NcaConfig::default(),
            std::path::PathBuf::from("/tmp/nca-registry-test-ws"),
            None,
        );

        // Hold the lease externally (simulating an in-flight control op):
        // the cancel must be refused, and must NOT touch the flag.
        let _lease = registry.try_acquire_lease("c1").expect("lease");
        let resp = send_cancel(&tx, "c1", None).await;
        assert!(!resp.ok);
        assert_eq!(
            resp.note.as_deref(),
            Some("another control operation is in flight for this task")
        );
        assert!(!flag.load(std::sync::atomic::Ordering::SeqCst));
    }
}
