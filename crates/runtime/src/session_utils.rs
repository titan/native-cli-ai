use crate::session_store::SessionStore;
use nca_common::config::NcaConfig;
use nca_common::event::{AgentCommand, AgentEvent, EndReason, EventEnvelope, QuestionSelection};
use nca_core::approval::ApprovalVerdict;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, watch};

pub(crate) type ApprovalPendingMap = Arc<Mutex<HashMap<String, oneshot::Sender<ApprovalVerdict>>>>;
pub(crate) type QuestionPendingMap =
    Arc<Mutex<HashMap<String, oneshot::Sender<QuestionSelection>>>>;
type EventFanoutCallback = Box<dyn Fn(&EventEnvelope) + Send>;

/// Last-activity heartbeat for the stalled-child watchdog (T1-3): fired
/// by a child session's event fanout for EVERY observed child event —
/// proof of life, regardless of whether the event maps to a
/// parent-visible forward. Typically closes over the parent's
/// [`crate::subagent_registry::SubagentRegistry`] and the child id
/// ([`SubagentRegistry::note_activity`]).
pub type ActivityHeartbeat = Arc<dyn Fn() + Send + Sync>;

/// Parent-forward wiring for a CHILD session's event fanout: the child's
/// identity, the parent's bounded event channel, and (optionally) the
/// T1-3 liveness heartbeat stamped for every observed child event.
#[derive(Clone)]
pub struct ParentForward {
    /// Child session id — tags forwarded activity events and keys the
    /// heartbeat.
    pub child_id: String,
    /// Parent's bounded event channel (forward target).
    pub tx: mpsc::Sender<AgentEvent>,
    /// Liveness heartbeat (T1-3). `None` only in tests that exercise
    /// forwarding without a registry; production child runs always stamp.
    pub heartbeat: Option<ActivityHeartbeat>,
    /// D6 activity projection feed: called with `(phase, detail)` for every
    /// event that maps to a parent-visible activity line, so the parent's
    /// registry can append it to the child's recent-activity ring. Fed from
    /// the same mapping the forward uses — including elided droppables (the
    /// ring is the model-facing record, not the UI channel). `None` when
    /// the run has no registry (tests).
    pub activity_log: Option<ActivityLogCallback>,
}

/// Callback that appends one `(phase, detail)` line to a child's
/// recent-activity ring (D6). Named to keep `ParentForward` readable.
pub type ActivityLogCallback = Arc<dyn Fn(&str, &str) + Send + Sync>;

/// Delivery class for a parent-forwarded child event
/// (event-consumer-liveness, Chunk 1). Decides how the fanout forwards it
/// when the parent's bounded event channel is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ForwardClass {
    /// Activity noise (tool starts, checkpoints). May elide under a stalled
    /// parent; elisions are counted and summarized by an `EventsElided`
    /// forward flushed before the next must-deliver forward.
    Droppable,
    /// Child notices the parent must never miss (errors, nested spawns).
    /// Never elided — the fanout blocks on them (with a stall alarm) until
    /// the parent drains.
    MustDeliver,
}

/// Maps a child session event to a parent-visible activity line (sidebar +
/// transcript), tagged with its [`ForwardClass`]. Mapping semantics are
/// unchanged from the pre-classification version; only the class tag is new.
///
/// Invariant: child must-deliver loss is never acceptable — an elided error
/// or nested-spawn notice would silently hide a child failure from the
/// parent's UI — while activity noise may elide under a stalled parent
/// (the parent sees one `EventsElided` summary instead of the noise).
pub(crate) fn map_child_event_for_parent_broadcast(
    child_session_id: &str,
    event: &AgentEvent,
) -> Option<(AgentEvent, ForwardClass)> {
    let child_activity = |phase: String, detail: String| AgentEvent::ChildSessionActivity {
        child_session_id: child_session_id.to_string(),
        phase,
        detail,
    };
    match event {
        AgentEvent::ToolCallStarted { tool, input, .. } => Some((
            child_activity(tool.clone(), tool_input_one_line(input)),
            ForwardClass::Droppable,
        )),
        AgentEvent::Checkpoint { phase, detail, .. } => Some((
            child_activity(phase.clone(), truncate_child_detail(detail, 120)),
            ForwardClass::Droppable,
        )),
        AgentEvent::ChildSessionSpawned { task, .. } => Some((
            child_activity(
                "nested_subagent".to_string(),
                truncate_child_detail(task, 120),
            ),
            ForwardClass::MustDeliver,
        )),
        AgentEvent::Error { message } => Some((
            child_activity("error".to_string(), truncate_child_detail(message, 160)),
            ForwardClass::MustDeliver,
        )),
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

/// Spawns the event fanout task: writes events to disk as `EventEnvelope`
/// (via [`crate::event_log::EventLogWriter`], which seeds ids from the
/// existing log), broadcasts over IPC, renders to the provided callback,
/// and fsync-commits at every `TurnCompleted` before signalling `commit_tx`.
///
/// Per-event order is **ipc → on_event → append → parent-forward → commit**
/// (live-consumer reorder): the callback fires before the append so
/// live-consumer latency (TUI render, stream renderer) never waits on
/// per-event disk I/O. The durability invariant — append precedes the
/// parent-forward — is untouched: a must-deliver forward parked on a
/// stalled parent can still never stale the child's own log; the parked
/// event is already durable if the process aborts mid-park.
///
/// `step_stall_alarm` arms a liveness watch (alarm-only, never
/// interference): while a `StepStarted` bracket is open, that much event
/// silence fires a synthetic `Checkpoint { phase: "step_stalled" }` through
/// the fanout's own durable path (ipc + on_event + append — never the
/// parent forward, never the event channel), repeating every interval
/// until events resume. `Duration::ZERO` disables the watch entirely.
pub fn spawn_event_fanout(
    mut event_rx: mpsc::Receiver<AgentEvent>,
    log_path: PathBuf,
    ipc_tx: Option<tokio::sync::broadcast::Sender<String>>,
    on_event: Option<EventFanoutCallback>,
    parent_forward: Option<ParentForward>,
    commit_tx: Option<TurnCommitTx>,
    step_stall_alarm: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut writer = crate::event_log::EventLogWriter::open(&log_path).await;

        // Parent-forward elision ledger (event-consumer-liveness Chunk 1).
        // The fanout is a single task, so a plain `u64` needs no atomicity.
        // Droppable forwards that hit a full parent channel elide into this
        // counter; the next must-deliver forward flushes it as one
        // `EventsElided` summary first. Invariant: child must-deliver loss
        // is never acceptable — an elided error or nested-spawn notice would
        // silently hide a child failure from the parent's UI — while
        // activity noise may elide under a stalled parent. A must-deliver
        // forward that blocks parks this loop (and transitively the child's
        // turn) until the parent drains: acceptable by design (rare
        // events), and `send_with_stall_alarm` screams every 30s while it
        // happens instead of parking silently.
        let mut elided_forwards: u64 = 0;

        // Step-liveness watch state (alarm-only). `tokio::time::Instant` —
        // not std — so the watch shares the select! timer's clock (and
        // paused-time tests can drive it). While the loop is parked inside a
        // must-deliver forward the watch cannot fire — that park has its own
        // 30s stall alarm in `send_with_stall_alarm`.
        let mut open_step: Option<(u64 /*turn_id*/, u64 /*step_index*/)> = None;
        let mut stalled_for = Duration::ZERO;
        // Some(now + interval) iff a step bracket is open AND the alarm is
        // enabled; slid forward by every activity event while a bracket is
        // open, re-armed after each alarm fire.
        let mut deadline: Option<tokio::time::Instant> = None;
        // Arm helper: the interval's zero-value disables the watch entirely.
        let arm = |now: tokio::time::Instant| {
            (step_stall_alarm > Duration::ZERO).then(|| now + step_stall_alarm)
        };

        loop {
            tokio::select! {
                maybe = event_rx.recv() => {
                    let Some(event) = maybe else { break };
                    stalled_for = Duration::ZERO;
                    // T1-3 liveness heartbeat: every observed child event
                    // is proof of life, BEFORE any mapping/elision — a
                    // busy child whose activity forwards elide under
                    // parent backpressure must not read as stalled.
                    if let Some(ref forward) = parent_forward
                        && let Some(ref heartbeat) = forward.heartbeat
                    {
                        heartbeat();
                    }
                    let now = tokio::time::Instant::now();
                    match &event {
                        AgentEvent::StepStarted { turn_id, step_index } => {
                            open_step = Some((*turn_id, *step_index));
                            deadline = arm(now);
                        }
                        AgentEvent::StepCompleted { .. }
                        | AgentEvent::StepFailed { .. }
                        | AgentEvent::TurnCompleted { .. } => {
                            open_step = None;
                            deadline = None;
                        }
                        // Activity (deltas, tool notices, …): slides the
                        // deadline while a bracket is open — liveness means
                        // "no events", not "no completions".
                        _ => {
                            if open_step.is_some() {
                                deadline = arm(now);
                            }
                        }
                    }

                    let id = writer.next_id();
                    let envelope = EventEnvelope::new(id, event);

                    if let Some(ref tx) = ipc_tx {
                        let line = serde_json::to_string(&envelope).unwrap_or_default();
                        let _ = tx.send(line);
                    }

                    // Live-consumer reorder: the callback (TUI bridge /
                    // stream renderer) fires BEFORE the append — live
                    // render latency must not wait on per-event disk I/O.
                    // The durability invariant (append precedes the
                    // parent-forward below) is untouched.
                    if let Some(ref cb) = on_event {
                        cb(&envelope);
                    }

                    if let Err(e) = writer.append(&envelope).await {
                        tracing::error!("failed to append event {} to log: {}", id, e);
                    }

                    // Forward AFTER the append (durability reorder): a
                    // must-deliver forward that parks here leaves the event
                    // already durable — abort-safe, never staled.
                    if let Some(ref forward) = parent_forward
                        && let Some((fwd, class)) =
                            map_child_event_for_parent_broadcast(&forward.child_id, &envelope.event)
                    {
                        // D6: feed the registry's activity ring from the same
                        // mapping — BEFORE the elision branch, so a ring entry
                        // exists even when the parent channel was full (the
                        // ring is the durable projection; elision only sheds
                        // the live-UI copy).
                        if let Some(ref log) = forward.activity_log
                            && let AgentEvent::ChildSessionActivity {
                                phase, detail, ..
                            } = &fwd
                        {
                            log(phase, detail);
                        }
                        match class {
                            ForwardClass::Droppable => match forward.tx.try_send(fwd) {
                                Ok(()) => {}
                                Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                                    elided_forwards += 1;
                                }
                                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {}
                            },
                            ForwardClass::MustDeliver => {
                                if elided_forwards > 0 {
                                    nca_core::agent::send_with_stall_alarm(
                                        &forward.tx,
                                        AgentEvent::EventsElided {
                                            count: elided_forwards,
                                        },
                                    )
                                    .await;
                                    elided_forwards = 0;
                                }
                                nca_core::agent::send_with_stall_alarm(&forward.tx, fwd).await;
                            }
                        }
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
                }
                // Step-stall alarm branch. The sleep future is created FRESH
                // each loop iteration inside this async block — never pinned
                // across iterations — so every recomputed deadline takes
                // effect on the next poll and a disabled/None deadline simply
                // never arms. The `if` precondition guarantees the branch (and
                // its `expect`) only run while a deadline is armed.
                _ = async {
                    tokio::time::sleep_until(
                        deadline.expect("precondition arms this branch only with Some"),
                    )
                    .await
                }, if deadline.is_some() => {
                    stalled_for += step_stall_alarm;
                    let (turn, step) =
                        open_step.expect("deadline armed only while a step is open");
                    let detail = format!(
                        "no events for {stalled_for:?} while step {step} of turn {turn} \
                         is open — wedged driver or silent long-running tool?"
                    );
                    tracing::error!(
                        turn,
                        step,
                        stalled_secs = stalled_for.as_secs(),
                        "step_stalled"
                    );
                    // `Checkpoint.turn` is u32 while `turn_id` is u64; the
                    // narrowing cast is display-only metadata.
                    let alarm = AgentEvent::Checkpoint {
                        phase: "step_stalled".to_string(),
                        detail,
                        turn: turn as u32,
                    };
                    // The alarm rides the fanout's own durable path with a
                    // fresh envelope id: ipc + callback + append (same
                    // live-consumer reorder as the main loop). NO parent
                    // forward (a Checkpoint maps to droppable activity
                    // anyway) and NEVER the event channel — this loop owns
                    // its receiver; re-sending would deadlock it.
                    let id = writer.next_id();
                    let envelope = EventEnvelope::new(id, alarm);
                    if let Some(ref tx) = ipc_tx {
                        let line = serde_json::to_string(&envelope).unwrap_or_default();
                        let _ = tx.send(line);
                    }
                    if let Some(ref cb) = on_event {
                        cb(&envelope);
                    }
                    if let Err(e) = writer.append(&envelope).await {
                        tracing::error!("failed to append event {} to log: {}", id, e);
                    }
                    // Re-arm from NOW so the alarm REPEATS at the next full
                    // interval instead of firing in a tight loop.
                    deadline = arm(tokio::time::Instant::now());
                }
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
    use chrono::{Duration as ChronoDuration, Utc};
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
                created_at: updated_at - ChronoDuration::minutes(1),
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
            now - ChronoDuration::minutes(10),
            "MiniMax-M2.5",
            SessionStatus::Completed,
        );
        write_session_for_test(
            workspace,
            "session-middle",
            now - ChronoDuration::minutes(5),
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

    // ── event-consumer-liveness Chunk 1: classified parent forwards ────

    /// T1-3: the fanout fires the liveness heartbeat for EVERY observed
    /// child event — including events that map to NO parent forward at
    /// all (e.g. `TokensStreamed`), so a busy child can never read as
    /// stalled just because its visible activity elided.
    #[tokio::test]
    async fn fanout_fires_heartbeat_for_every_child_event() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log_path = dir.path().join("events.jsonl");
        let (event_tx, event_rx) = mpsc::channel(64);
        let (ptx, _prx) = mpsc::channel(16);
        let beats = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&beats);
        let forward = ParentForward {
            child_id: "child-1".to_string(),
            tx: ptx,
            heartbeat: Some(Arc::new(move || {
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            })),
            activity_log: None,
        };
        let mut task = spawn_event_fanout(
            event_rx,
            log_path,
            None,
            None,
            Some(forward),
            None,
            Duration::ZERO,
        );

        // One forwardable event + two that map to NO forward.
        event_tx
            .send(AgentEvent::ToolCallStarted {
                call_id: "c0".into(),
                tool: "read_file".into(),
                input: serde_json::json!("path"),
            })
            .await
            .expect("send");
        for _ in 0..2 {
            event_tx
                .send(AgentEvent::TokensStreamed { delta: "x".into() })
                .await
                .expect("send");
        }
        drop(event_tx);
        crate::session_utils::drain_event_fanout(&mut task, "heartbeat test").await;

        assert_eq!(
            beats.load(std::sync::atomic::Ordering::SeqCst),
            3,
            "every observed child event fired the heartbeat"
        );
    }

    fn nonempty_log_lines(log_path: &std::path::Path) -> Vec<String> {
        std::fs::read_to_string(log_path)
            .unwrap_or_default()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.to_string())
            .collect()
    }

    /// Droppable forwards elide under a stalled parent instead of parking
    /// the fanout: the loop keeps draining its own channel, every child
    /// event stays durable, and exactly channel-capacity forwards land.
    #[tokio::test]
    async fn fanout_elides_droppable_forwards_when_parent_stalls() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log_path = dir.path().join("events.jsonl");
        let (event_tx, event_rx) = mpsc::channel(64);
        // Stalled parent: capacity 2, receiver never drained during the run.
        let (ptx, mut prx) = mpsc::channel(2);
        let mut task = spawn_event_fanout(
            event_rx,
            log_path.clone(),
            None,
            None,
            Some(ParentForward {
                child_id: "child-1".to_string(),
                tx: ptx,
                heartbeat: None,
                activity_log: None,
            }),
            None,
            Duration::ZERO,
        );

        for i in 0..10 {
            event_tx
                .send(AgentEvent::ToolCallStarted {
                    call_id: format!("c{i}"),
                    tool: "read".to_string(),
                    input: serde_json::json!({ "path": "x" }),
                })
                .await
                .expect("send child event");
        }
        drop(event_tx);

        // Clean exit (not the drain-abort path): the loop never parked on a
        // droppable forward despite the undrained parent.
        let exited = tokio::time::timeout(std::time::Duration::from_secs(5), &mut task).await;
        assert!(
            exited.is_ok(),
            "fanout must drain and exit cleanly despite the stalled parent"
        );

        // Log completeness: all 10 events hit disk.
        let lines = nonempty_log_lines(&log_path);
        assert_eq!(lines.len(), 10, "all events logged: {lines:?}");
        assert!(lines.iter().all(|l| l.contains("ToolCallStarted")));

        // Exactly capacity(2) forwards landed; the other 8 elided.
        assert!(matches!(
            prx.try_recv(),
            Ok(AgentEvent::ChildSessionActivity { .. })
        ));
        assert!(matches!(
            prx.try_recv(),
            Ok(AgentEvent::ChildSessionActivity { .. })
        ));
        assert!(
            prx.try_recv().is_err(),
            "8 forwards elided; nothing else was delivered"
        );
    }

    /// Must-deliver forwards never elide: a stalled parent parks the fanout
    /// AFTER the event's own append (durability reorder: ipc → append →
    /// forward), so the parked event is already durable; recovery flushes
    /// the elision summary ahead of the error forward.
    #[tokio::test]
    async fn fanout_parks_on_must_deliver_forward_then_recovers() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log_path = dir.path().join("events.jsonl");
        let (event_tx, event_rx) = mpsc::channel(64);
        let (ptx, mut prx) = mpsc::channel(1);
        let mut task = spawn_event_fanout(
            event_rx,
            log_path.clone(),
            None,
            None,
            Some(ParentForward {
                child_id: "child-2".to_string(),
                tx: ptx,
                heartbeat: None,
                activity_log: None,
            }),
            None,
            Duration::ZERO,
        );

        // 3 droppable forwards against capacity 1: first lands, two elide.
        for i in 0..3 {
            event_tx
                .send(AgentEvent::ToolCallStarted {
                    call_id: format!("c{i}"),
                    tool: "read".to_string(),
                    input: serde_json::json!({}),
                })
                .await
                .expect("send child event");
        }
        // Must-deliver: the Error's OWN append completes first (durability
        // reorder), THEN the forward parks the loop on the stalled parent
        // (with the stall alarm screaming every 30s — by design, not a
        // deadlock). The parked event is abort-durable either way.
        event_tx
            .send(AgentEvent::Error {
                message: "child exploded".to_string(),
            })
            .await
            .expect("send error event");
        drop(event_tx);

        // Bounded wall wait until all FOUR events are durable — proves the
        // loop drained the droppables AND appended the Error, and is now
        // parked on the Error FORWARD alone.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while nonempty_log_lines(&log_path).len() < 4 && std::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let lines = nonempty_log_lines(&log_path);
        assert_eq!(
            lines.len(),
            4,
            "droppables + Error durable; only the forward is parked: {lines:?}"
        );
        assert!(lines[3].contains("child exploded"));
        assert!(
            !task.is_finished(),
            "fanout must stay parked on the must-deliver forward"
        );

        // Parent finally drains: elision summary flushes first, then the
        // error forward itself — in that order.
        let first = tokio::time::timeout(std::time::Duration::from_secs(2), prx.recv())
            .await
            .expect("first forward")
            .expect("parent channel open");
        assert!(
            matches!(&first, AgentEvent::ChildSessionActivity { phase, .. } if phase == "read")
        );
        let second = tokio::time::timeout(std::time::Duration::from_secs(2), prx.recv())
            .await
            .expect("elision summary")
            .expect("parent channel open");
        assert!(matches!(second, AgentEvent::EventsElided { count: 2 }));
        let third = tokio::time::timeout(std::time::Duration::from_secs(2), prx.recv())
            .await
            .expect("error forward")
            .expect("parent channel open");
        assert!(
            matches!(&third, AgentEvent::ChildSessionActivity { phase, .. } if phase == "error")
        );

        // Park resolved: the loop exits cleanly once its own senders are
        // gone (the Error was already durable at park time).
        let exited = tokio::time::timeout(std::time::Duration::from_secs(5), &mut task).await;
        assert!(exited.is_ok(), "fanout exits after the parent drains");
        let lines = nonempty_log_lines(&log_path);
        assert_eq!(
            lines.len(),
            4,
            "no duplicate append after the park: {lines:?}"
        );
        assert!(lines[3].contains("child exploded"));
    }

    // ── step-watch: alarm-only step-liveness inside the fanout ──────

    /// Bounded real-time wait until the log holds at least `want` non-empty
    /// lines. Needed because the fanout's appends land on tokio's blocking
    /// pool — the paused virtual clock alone cannot order them against a
    /// std::fs read from the test task.
    async fn await_log_lines(log_path: &std::path::Path, want: usize) -> Vec<String> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let lines = nonempty_log_lines(log_path);
            if lines.len() >= want || std::time::Instant::now() > deadline {
                return lines;
            }
            tokio::task::yield_now().await;
        }
    }

    /// Advance virtual time past `interval` and let the runtime poll every
    /// task a timer or channel readied. The +50ms overshoot guarantees the
    /// fanout's earlier deadline fires first (established wake-scheduler
    /// idiom for paused-time tests).
    async fn elapse(interval: Duration) {
        tokio::time::sleep(interval + Duration::from_millis(50)).await;
    }

    fn alarms(lines: &[String]) -> Vec<&String> {
        lines
            .iter()
            .filter(|l| l.contains("step_stalled"))
            .collect()
    }

    #[tokio::test(start_paused = true)]
    async fn step_stall_alarm_fires_at_deadline_and_repeats() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log_path = dir.path().join("events.jsonl");
        let (event_tx, event_rx) = mpsc::channel(8);
        let mut task = spawn_event_fanout(
            event_rx,
            log_path.clone(),
            None,
            None,
            None,
            None,
            Duration::from_secs(600),
        );

        event_tx
            .send(AgentEvent::StepStarted {
                turn_id: 7,
                step_index: 3,
            })
            .await
            .expect("send step started");

        // Silence for the full interval: exactly one alarm at the deadline.
        elapse(Duration::from_secs(600)).await;
        let lines = await_log_lines(&log_path, 2).await;
        assert_eq!(lines.len(), 2, "step + one alarm: {lines:?}");
        assert!(lines[1].contains("step_stalled"), "{lines:?}");
        assert!(
            lines[1].contains("600s"),
            "first alarm reports 600s: {lines:?}"
        );
        assert!(
            lines[1].contains("step 3 of turn 7"),
            "alarm names the open bracket: {lines:?}"
        );

        // Another full interval of silence: the alarm REPEATS at the next
        // interval with the accumulated stall, not in a burst.
        elapse(Duration::from_secs(600)).await;
        let lines = await_log_lines(&log_path, 3).await;
        assert_eq!(lines.len(), 3, "step + two alarms: {lines:?}");
        assert!(
            lines[2].contains("1200s"),
            "second alarm accumulates the stall: {lines:?}"
        );

        drop(event_tx);
        let exited = tokio::time::timeout(Duration::from_secs(5), &mut task).await;
        assert!(exited.is_ok(), "fanout drains and exits cleanly");
    }

    #[tokio::test(start_paused = true)]
    async fn stream_activity_resets_step_stall_deadline() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log_path = dir.path().join("events.jsonl");
        let (event_tx, event_rx) = mpsc::channel(8);
        let mut task = spawn_event_fanout(
            event_rx,
            log_path.clone(),
            None,
            None,
            None,
            None,
            Duration::from_secs(600),
        );

        event_tx
            .send(AgentEvent::StepStarted {
                turn_id: 7,
                step_index: 3,
            })
            .await
            .expect("send step started");

        // Token stream at t≈300 is activity: the deadline moves from t≈600
        // (step start + 600) to t≈900 (delta + 600).
        elapse(Duration::from_secs(300)).await;
        event_tx
            .send(AgentEvent::TokensStreamed {
                delta: "chunk".to_string(),
            })
            .await
            .expect("send delta");

        // t≈700: past the ORIGINAL deadline, before the reset one.
        elapse(Duration::from_secs(400)).await;
        let lines = await_log_lines(&log_path, 2).await;
        assert_eq!(lines.len(), 2, "step + delta only: {lines:?}");
        assert!(
            alarms(&lines).is_empty(),
            "activity must reset the deadline: {lines:?}"
        );

        // Silence past the RESET deadline (t≈900): fires, measured from
        // the delta — 600s of stall, not 900.
        elapse(Duration::from_secs(600)).await;
        let lines = await_log_lines(&log_path, 3).await;
        assert_eq!(lines.len(), 3, "step + delta + one alarm: {lines:?}");
        assert!(
            lines[2].contains("600s"),
            "stall measured from the delta, not the step start: {lines:?}"
        );

        drop(event_tx);
        let exited = tokio::time::timeout(Duration::from_secs(5), &mut task).await;
        assert!(exited.is_ok(), "fanout drains and exits cleanly");
    }

    #[tokio::test(start_paused = true)]
    async fn step_stall_alarm_disarms_on_step_completion() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log_path = dir.path().join("events.jsonl");
        let (event_tx, event_rx) = mpsc::channel(8);
        let mut task = spawn_event_fanout(
            event_rx,
            log_path.clone(),
            None,
            None,
            None,
            None,
            Duration::from_secs(600),
        );

        event_tx
            .send(AgentEvent::StepStarted {
                turn_id: 7,
                step_index: 3,
            })
            .await
            .expect("send step started");
        event_tx
            .send(AgentEvent::StepCompleted {
                turn_id: 7,
                step_index: 3,
                duration_ms: 1,
                had_tool_calls: false,
            })
            .await
            .expect("send step completed");

        // 3× the interval after the bracket closed: the armed→cleared
        // transition must not false-alarm an idle session — the deadline
        // was dropped with the bracket, not merely slid.
        elapse(Duration::from_secs(1_800)).await;
        let lines = await_log_lines(&log_path, 2).await;
        assert_eq!(
            lines.len(),
            2,
            "step start + completion only, no alarm: {lines:?}"
        );
        assert!(alarms(&lines).is_empty(), "closed bracket never alarms");

        drop(event_tx);
        let exited = tokio::time::timeout(Duration::from_secs(5), &mut task).await;
        assert!(exited.is_ok(), "fanout drains and exits cleanly");
    }

    #[tokio::test(start_paused = true)]
    async fn step_stall_alarm_resets_after_activity_resumes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log_path = dir.path().join("events.jsonl");
        let (event_tx, event_rx) = mpsc::channel(8);
        let mut task = spawn_event_fanout(
            event_rx,
            log_path.clone(),
            None,
            None,
            None,
            None,
            Duration::from_secs(600),
        );

        event_tx
            .send(AgentEvent::StepStarted {
                turn_id: 7,
                step_index: 3,
            })
            .await
            .expect("send step started");

        // Silence past the interval: alarm #1 fires, stalled_for = 600s.
        elapse(Duration::from_secs(600)).await;
        let lines = await_log_lines(&log_path, 2).await;
        assert_eq!(lines.len(), 2, "step + one alarm: {lines:?}");
        assert!(
            lines[1].contains("600s"),
            "first alarm reports one interval: {lines:?}"
        );

        // Activity resumes mid-bracket: slides the deadline and resets the
        // stall accumulator.
        event_tx
            .send(AgentEvent::TokensStreamed {
                delta: "chunk".to_string(),
            })
            .await
            .expect("send delta");

        // Silence past ONE fresh interval: alarm #2 fires with stalled_for
        // measured from the activity (600s), not accumulated 2× (1200s) —
        // the counter reset keeps the cadence interval-based.
        elapse(Duration::from_secs(600)).await;
        let lines = await_log_lines(&log_path, 4).await;
        assert_eq!(lines.len(), 4, "step + alarm + delta + alarm: {lines:?}");
        assert_eq!(
            alarms(&lines).len(),
            2,
            "exactly two alarms, cadence stays interval-based: {lines:?}"
        );
        assert!(
            lines[3].contains("600s"),
            "second alarm measures one fresh interval, not 2×: {lines:?}"
        );
        assert!(
            !lines[3].contains("1200s"),
            "stall accumulator reset at the activity: {lines:?}"
        );

        drop(event_tx);
        let exited = tokio::time::timeout(Duration::from_secs(5), &mut task).await;
        assert!(exited.is_ok(), "fanout drains and exits cleanly");
    }

    #[tokio::test(start_paused = true)]
    async fn no_step_stall_alarm_without_open_step() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log_path = dir.path().join("events.jsonl");
        let (event_tx, event_rx) = mpsc::channel(8);
        let mut task = spawn_event_fanout(
            event_rx,
            log_path.clone(),
            None,
            None,
            None,
            None,
            Duration::from_secs(600),
        );

        // A non-bracket event, then silence: no StepStarted ever opened a
        // bracket, so no timer ever arms regardless of the silence length.
        event_tx
            .send(AgentEvent::ToolCallStarted {
                call_id: "c0".to_string(),
                tool: "read".to_string(),
                input: serde_json::json!({}),
            })
            .await
            .expect("send tool event");
        elapse(Duration::from_secs(1_200)).await;

        let lines = await_log_lines(&log_path, 1).await;
        assert_eq!(lines.len(), 1, "only the tool event, no alarm: {lines:?}");
        assert!(alarms(&lines).is_empty());

        drop(event_tx);
        let exited = tokio::time::timeout(Duration::from_secs(5), &mut task).await;
        assert!(exited.is_ok());
    }

    #[tokio::test(start_paused = true)]
    async fn step_stall_alarm_disabled_never_arms() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log_path = dir.path().join("events.jsonl");
        let (event_tx, event_rx) = mpsc::channel(8);
        // Interval 0: the watch is fully disabled — the loop behaves exactly
        // as before the watch existed (no timer ever arms, even with a
        // bracket open).
        let mut task = spawn_event_fanout(
            event_rx,
            log_path.clone(),
            None,
            None,
            None,
            None,
            Duration::ZERO,
        );

        event_tx
            .send(AgentEvent::StepStarted {
                turn_id: 1,
                step_index: 1,
            })
            .await
            .expect("send step started");
        elapse(Duration::from_secs(1_200)).await;

        let lines = await_log_lines(&log_path, 1).await;
        assert_eq!(lines.len(), 1, "no alarm appended when disabled: {lines:?}");
        assert!(alarms(&lines).is_empty());

        drop(event_tx);
        let exited = tokio::time::timeout(Duration::from_secs(5), &mut task).await;
        assert!(exited.is_ok());
    }
}
