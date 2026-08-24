use crate::session_store::SessionStore;
use nca_common::config::NcaConfig;
use nca_common::event::{AgentCommand, AgentEvent, EndReason, EventEnvelope, QuestionSelection};
use nca_core::approval::ApprovalVerdict;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot, watch};

pub(crate) type ApprovalPendingMap = Arc<Mutex<HashMap<String, oneshot::Sender<ApprovalVerdict>>>>;
pub(crate) type QuestionPendingMap =
    Arc<Mutex<HashMap<String, oneshot::Sender<QuestionSelection>>>>;
type EventFanoutCallback = Box<dyn Fn(&EventEnvelope) + Send>;

/// Maps a child session event to a parent-visible activity line (sidebar + transcript).
pub(crate) fn map_child_event_for_parent_broadcast(
    child_session_id: &str,
    event: &AgentEvent,
) -> Option<AgentEvent> {
    match event {
        AgentEvent::ToolCallStarted { tool, input, .. } => Some(AgentEvent::ChildSessionActivity {
            child_session_id: child_session_id.to_string(),
            phase: tool.clone(),
            detail: tool_input_one_line(input),
        }),
        AgentEvent::Checkpoint { phase, detail, .. } => Some(AgentEvent::ChildSessionActivity {
            child_session_id: child_session_id.to_string(),
            phase: phase.clone(),
            detail: truncate_child_detail(detail, 120),
        }),
        AgentEvent::ChildSessionSpawned { task, .. } => Some(AgentEvent::ChildSessionActivity {
            child_session_id: child_session_id.to_string(),
            phase: "nested_subagent".to_string(),
            detail: truncate_child_detail(task, 120),
        }),
        AgentEvent::Error { message } => Some(AgentEvent::ChildSessionActivity {
            child_session_id: child_session_id.to_string(),
            phase: "error".to_string(),
            detail: truncate_child_detail(message, 160),
        }),
        _ => None,
    }
}

fn truncate_child_detail(s: &str, max_chars: usize) -> String {
    let t = s.trim();
    if t.chars().count() <= max_chars {
        t.to_string()
    } else {
        format!(
            "{}…",
            t.chars()
                .take(max_chars.saturating_sub(1))
                .collect::<String>()
        )
    }
}

fn tool_input_one_line(input: &serde_json::Value) -> String {
    if let Some(s) = input.as_str() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(s) {
            return tool_input_one_line(&v);
        }
        return truncate_child_detail(s, 120);
    }
    if let Some(cmd) = input.get("command").and_then(|v| v.as_str()) {
        return truncate_child_detail(cmd, 120);
    }
    if let Some(p) = input
        .get("path")
        .or_else(|| input.get("file_path"))
        .and_then(|v| v.as_str())
    {
        return truncate_child_detail(p, 120);
    }
    let s = serde_json::to_string(input).unwrap_or_default();
    truncate_child_detail(&s, 120)
}

/// Optional durability barrier for `run_turn`: the fanout sends the
/// `turn_id` of each `TurnCompleted` it has committed (flush + fsync) to the
/// log. The supervisor's `run_turn` waits on this watch so a turn's events
/// are durable before it returns. See `Supervisor::await_turn_commit`.
pub(crate) type TurnCommitTx = watch::Sender<u64>;

/// Spawns the event fanout task: writes events to disk as `EventEnvelope`
/// (via [`crate::event_log::EventLogWriter`], which seeds ids from the
/// existing log), broadcasts over IPC, renders to the provided callback,
/// and fsync-commits at every `TurnCompleted` before signalling `commit_tx`.
/// Bounded graceful drain of a [`spawn_event_fanout`] task (P2 Phase C §4).
/// Callers must first drop every event-channel sender (supervisor + consumer
/// clones) so the fanout's loop ends, commits on close, and exits; this
/// awaits that exit with a liveness timeout, aborting on expiry.
pub(crate) async fn drain_event_fanout(task: &mut tokio::task::JoinHandle<()>, label: &str) {
    if tokio::time::timeout(std::time::Duration::from_secs(5), &mut *task)
        .await
        .is_err()
    {
        tracing::error!("event fanout drain for {label} timed out; aborting");
        task.abort();
    }
}

pub fn spawn_event_fanout(
    mut event_rx: mpsc::Receiver<AgentEvent>,
    log_path: PathBuf,
    ipc_tx: Option<tokio::sync::broadcast::Sender<String>>,
    on_event: Option<EventFanoutCallback>,
    parent_forward: Option<(String, mpsc::Sender<AgentEvent>)>,
    commit_tx: Option<TurnCommitTx>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut writer = crate::event_log::EventLogWriter::open(&log_path).await;

        while let Some(event) = event_rx.recv().await {
            let id = writer.next_id();
            if let Some((ref child_id, ref ptx)) = parent_forward
                && let Some(fwd) = map_child_event_for_parent_broadcast(child_id, &event)
            {
                let _ = ptx.send(fwd).await;
            }
            let envelope = EventEnvelope::new(id, event);
            if let Some(ref tx) = ipc_tx {
                let line = serde_json::to_string(&envelope).unwrap_or_default();
                let _ = tx.send(line);
            }

            if let Err(e) = writer.append(&envelope).await {
                tracing::error!("failed to append event {} to log: {}", id, e);
            }

            // Liveness over false durability: the commit barrier fires even
            // if the append failed — a stuck barrier would be worse than a
            // missing line (the tolerant reader skips gaps).
            if let AgentEvent::TurnCompleted { turn_id, .. } = &envelope.event {
                if let Err(e) = writer.commit().await {
                    tracing::error!("event-log commit at turn {turn_id} failed: {e}");
                }
                if let Some(tx) = &commit_tx {
                    let _ = tx.send(*turn_id);
                }
            }

            if let Some(ref cb) = on_event {
                cb(&envelope);
            }
        }

        // All senders dropped (owning supervisor gone): flush + fsync
        // anything still buffered (e.g. `SessionEnded`) so the log is
        // durable on graceful close instead of resting in the page cache.
        // Log-and-continue on error, matching the TurnCompleted handling.
        if let Err(e) = writer.commit().await {
            tracing::error!("event-log commit at channel close failed: {e}");
        }
    })
}

/// Spawns a task that consumes IPC commands and resolves approvals/cancellation.
pub fn spawn_command_consumer(
    command_rx: mpsc::UnboundedReceiver<AgentCommand>,
    approval_pending: Option<ApprovalPendingMap>,
    question_pending: Option<QuestionPendingMap>,
    cancel_tx: Option<oneshot::Sender<()>>,
) -> tokio::task::JoinHandle<()> {
    spawn_command_consumer_with_store(
        command_rx,
        approval_pending,
        question_pending,
        cancel_tx,
        None,
        None,
        None,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionControlCommand {
    Cancel,
    Shutdown,
}

/// Extended command consumer with optional event fanout, prompt forwarding, and session control.
pub fn spawn_command_consumer_with_store(
    mut command_rx: mpsc::UnboundedReceiver<AgentCommand>,
    approval_pending: Option<ApprovalPendingMap>,
    question_pending: Option<QuestionPendingMap>,
    cancel_tx: Option<oneshot::Sender<()>>,
    event_tx: Option<mpsc::Sender<AgentEvent>>,
    prompt_tx: Option<mpsc::UnboundedSender<String>>,
    control_tx: Option<mpsc::UnboundedSender<SessionControlCommand>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut cancel = cancel_tx;
        while let Some(cmd) = command_rx.recv().await {
            match cmd {
                AgentCommand::ApproveToolCall { call_id } => {
                    if let Some(ref p) = approval_pending
                        && let Ok(mut m) = p.lock()
                        && let Some(tx) = m.remove(&call_id)
                    {
                        let _ = tx.send(ApprovalVerdict::Approved);
                    }
                }
                AgentCommand::DenyToolCall { call_id } => {
                    if let Some(ref p) = approval_pending
                        && let Ok(mut m) = p.lock()
                        && let Some(tx) = m.remove(&call_id)
                    {
                        let _ = tx.send(ApprovalVerdict::Denied);
                    }
                }
                AgentCommand::Cancel => {
                    if let Some(tx) = cancel.take() {
                        let _ = tx.send(());
                    }
                    if let Some(ref tx) = control_tx {
                        let _ = tx.send(SessionControlCommand::Cancel);
                    } else if let Some(ref tx) = event_tx {
                        let _ = tx
                            .send(AgentEvent::SessionEnded {
                                reason: EndReason::Cancelled,
                            })
                            .await;
                    }
                }
                AgentCommand::Shutdown => {
                    if let Some(tx) = cancel.take() {
                        let _ = tx.send(());
                    }
                    if let Some(ref tx) = control_tx {
                        let _ = tx.send(SessionControlCommand::Shutdown);
                    } else if let Some(ref tx) = event_tx {
                        let _ = tx
                            .send(AgentEvent::SessionEnded {
                                reason: EndReason::UserExit,
                            })
                            .await;
                    }
                    break;
                }
                AgentCommand::SendMessage { content } => {
                    if let Some(ref tx) = prompt_tx {
                        let _ = tx.send(content);
                    } else if let Some(ref tx) = event_tx {
                        // UI-only echo; no `messages.push` here, so it is deliberately NOT a
                        // `MessageRecorded` (never replayed). See p2 design doc.
                        let _ = tx
                            .send(AgentEvent::MessageReceived {
                                role: "user".into(),
                                content,
                                steering: false,
                            })
                            .await;
                    }
                }
                AgentCommand::AnswerQuestion {
                    question_id,
                    selection,
                } => {
                    if let Some(ref qp) = question_pending
                        && let Ok(mut m) = qp.lock()
                        && let Some(tx) = m.remove(&question_id)
                    {
                        let _ = tx.send(selection);
                    }
                }
            }
        }
    })
}

/// Get the last session ID from `.nca/.last_session`, if it exists and is valid.
/// Falls back to finding the most recently updated session in the sessions directory.
pub async fn get_last_session_id(
    config: &NcaConfig,
    workspace_root: &Path,
) -> anyhow::Result<Option<String>> {
    use crate::last_session::LastSessionStore;

    // First, try the explicit last-session pointer
    let store = LastSessionStore::new(workspace_root.join(&config.session.last_session_file));
    match store.load().await {
        Ok(Some(id)) => {
            // Verify the session still exists on disk.
            let session_store = SessionStore::new(workspace_root.join(&config.session.history_dir));
            match session_store.load(&id).await {
                Ok(_) => return Ok(Some(id)),
                Err(_) => {
                    // Session file missing or corrupted; clear the stale pointer.
                    let _ = store.clear().await;
                }
            }
        }
        Ok(None) => {
            // No pointer file - fall through to scan sessions dir
        }
        Err(e) => {
            tracing::warn!("failed to load last session pointer: {}", e);
            // Fall through to scan sessions dir
        }
    }

    // Fallback: find the most recently updated session in the sessions directory
    let session_store = SessionStore::new(workspace_root.join(&config.session.history_dir));
    let ids = match session_store.list().await {
        Ok(ids) => ids,
        Err(e) => {
            tracing::debug!("failed to list sessions: {}", e);
            return Ok(None);
        }
    };

    let mut latest: Option<(String, chrono::DateTime<chrono::Utc>)> = None;
    for id in ids {
        match session_store.load(&id).await {
            Ok(session) => {
                let should_replace = latest
                    .as_ref()
                    .map(|(_, updated_at)| session.meta.updated_at > *updated_at)
                    .unwrap_or(true);
                if should_replace {
                    latest = Some((session.meta.id, session.meta.updated_at));
                }
            }
            Err(_) => continue,
        }
    }

    if let Some((id, _)) = latest {
        // Update the last-session pointer for future runs
        let _ = store.save(&id).await;
        Ok(Some(id))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};
    use nca_common::event::AgentCommand;
    use nca_common::message::Message;
    use nca_common::session::{SessionMeta, SessionState, SessionStatus};
    use std::fs;

    fn write_session_for_test(
        workspace: &std::path::Path,
        id: &str,
        updated_at: chrono::DateTime<Utc>,
        model: &str,
        status: SessionStatus,
    ) {
        let sessions_dir = workspace.join(".nca").join("sessions");
        std::fs::create_dir_all(&sessions_dir).expect("create sessions dir");

        let session = SessionState {
            meta: SessionMeta {
                id: id.to_string(),
                created_at: updated_at - Duration::minutes(1),
                updated_at,
                workspace: workspace.to_path_buf(),
                model: model.to_string(),
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
            messages: vec![Message::user("hello")],
            total_input_tokens: 0,
            total_output_tokens: 0,
            estimated_cost_usd: 0.0,
        };

        let json = serde_json::to_string_pretty(&session).expect("serialize session");
        fs::write(sessions_dir.join(format!("{id}.json")), json).expect("write session");
    }

    #[tokio::test]
    async fn get_last_session_id_falls_back_to_most_recent() {
        let temp = tempfile::tempdir().expect("tempdir");
        let workspace = temp.path();
        let now = Utc::now();

        // Write sessions WITHOUT .last_session file
        write_session_for_test(
            workspace,
            "session-oldest",
            now - Duration::minutes(10),
            "MiniMax-M2.5",
            SessionStatus::Completed,
        );
        write_session_for_test(
            workspace,
            "session-middle",
            now - Duration::minutes(5),
            "MiniMax-M2.5",
            SessionStatus::Completed,
        );
        write_session_for_test(
            workspace,
            "session-newest",
            now,
            "MiniMax-M2.5",
            SessionStatus::Running,
        );

        let config = nca_common::config::NcaConfig::default();
        let session_id = get_last_session_id(&config, workspace)
            .await
            .expect("get_last_session_id should succeed")
            .expect("should find a session");

        // Should find the most recent session
        assert_eq!(session_id, "session-newest");

        // The .last_session file should now be updated
        let last_session_path = workspace.join(".nca").join(".last_session");
        assert!(
            last_session_path.exists(),
            ".last_session should be created"
        );
        let content = std::fs::read_to_string(&last_session_path).unwrap();
        assert_eq!(content.trim(), "session-newest");
    }

    #[tokio::test]
    async fn send_message_forwards_prompt_to_session_queue() {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let (prompt_tx, mut prompt_rx) = mpsc::unbounded_channel();
        let (control_tx, _control_rx) = mpsc::unbounded_channel();

        let task = spawn_command_consumer_with_store(
            cmd_rx,
            None,
            None,
            None,
            None,
            Some(prompt_tx),
            Some(control_tx),
        );

        cmd_tx
            .send(AgentCommand::SendMessage {
                content: "hello from ipc".into(),
            })
            .unwrap();

        let received = tokio::time::timeout(std::time::Duration::from_secs(1), prompt_rx.recv())
            .await
            .expect("prompt should be forwarded")
            .expect("prompt channel should remain open");
        assert_eq!(received, "hello from ipc");

        let _ = cmd_tx.send(AgentCommand::Shutdown);
        task.abort();
    }

    #[tokio::test]
    async fn answer_question_resolves_pending_channel() {
        use nca_common::event::QuestionSelection;

        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let pending: Arc<Mutex<HashMap<String, oneshot::Sender<QuestionSelection>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let (tx, rx) = oneshot::channel();
        pending.lock().unwrap().insert("q-1".into(), tx);

        let task = spawn_command_consumer_with_store(
            cmd_rx,
            None,
            Some(pending.clone()),
            None,
            None,
            None,
            None,
        );

        cmd_tx
            .send(AgentCommand::AnswerQuestion {
                question_id: "q-1".into(),
                selection: QuestionSelection::Suggested,
            })
            .unwrap();

        let got = tokio::time::timeout(std::time::Duration::from_secs(1), rx)
            .await
            .expect("timeout")
            .expect("channel");
        assert!(matches!(got, QuestionSelection::Suggested));

        let _ = cmd_tx.send(AgentCommand::Shutdown);
        task.abort();
    }
}
