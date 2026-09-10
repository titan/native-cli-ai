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
use nca_core::tools::subagent_control::{SubagentControlRequest, SubagentControlResponse};
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
}

/// In-memory projection of spawned child tasks, keyed by child session id
/// and kept in spawn order for `list()`. Cheap to rebuild from the event
/// log; never persisted directly.
#[derive(Default)]
pub struct SubagentRegistry {
    entries: Mutex<Vec<SubagentRegistryEntry>>,
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
        });
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

    /// All entries in spawn order.
    pub fn list(&self) -> Vec<SubagentRegistryEntry> {
        let entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        entries.clone()
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
    }
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
}
