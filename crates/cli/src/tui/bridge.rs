//! Event fanout: session log, IPC, and TUI state (no stdout streaming).

use crate::ipc_pending::{ApprovalPendingMap, QuestionPendingMap};
use crate::tui::elm::feedback::TuiFeedbackMsg;
use nca_common::event::{AgentEvent, EventEnvelope};
use nca_common::todo::TodoStatus;
use nca_runtime::event_log::EventLogWriter;
use nca_runtime::ipc::IpcHandle;
use nca_runtime::supervisor;
use nca_runtime::wake_scheduler::WakeScheduler;

struct IpcFanout {
    tx: tokio::sync::broadcast::Sender<String>,
}

/// Disk + IPC + TUI state; starts IPC command consumer when needed.
/// `commit_tx` receives the turn id of each `TurnCompleted` committed
/// (flush + fsync) to the log — the supervisor's `run_turn` barrier.
/// `wake`, when set, receives the P3 todo-gate fold of live
/// `TodosUpdated` events (see [`fold_wake_todos`]).
#[allow(clippy::too_many_arguments)]
pub fn spawn_tui_bridge(
    mut rx: tokio::sync::mpsc::Receiver<AgentEvent>,
    log_path: std::path::PathBuf,
    ipc_handle: Option<IpcHandle>,
    approval_pending: Option<ApprovalPendingMap>,
    question_pending: Option<QuestionPendingMap>,
    feedback_tx: tokio::sync::mpsc::UnboundedSender<TuiFeedbackMsg>,
    commit_tx: Option<tokio::sync::watch::Sender<u64>>,
    wake: Option<WakeScheduler>,
) -> tokio::task::JoinHandle<()> {
    let (event_tx_ipc, command_rx) = match ipc_handle {
        Some(h) => {
            let (etx, crx) = h.into_parts();
            (Some(etx), Some(crx))
        }
        None => (None, None),
    };

    if let Some(crx) = command_rx {
        supervisor::spawn_command_consumer(crx, approval_pending, question_pending, None);
    }

    let ipc = event_tx_ipc.map(|tx| IpcFanout { tx });

    tokio::spawn(async move {
        let mut writer = EventLogWriter::open(&log_path).await;

        while let Some(event) = rx.recv().await {
            // P3 todo gate: live TodosUpdated events feed the wake
            // scheduler. Replay does not rebuild the gate (TodoStore is
            // not replayed), so after a restart the scheduler stays
            // conservative (always wake) — documented limitation.
            fold_wake_todos(&event, wake.as_ref());

            let id = writer.next_id();
            let envelope = EventEnvelope::new(id, event.clone());

            if let Some(ref fan) = ipc {
                let line = serde_json::to_string(&envelope).unwrap_or_default();
                let _ = fan.tx.send(line);
            }

            // Forward to TUI FIRST, before disk I/O. Disk writes are the
            // per-event bottleneck (JSON serialize + async write); doing them
            // before the TUI forward delays every subsequent event in the
            // bounded(256) channel, which can cause the UI to lag behind the
            // agent by hundreds of milliseconds during high-frequency streaming.
            let _ = feedback_tx.send(TuiFeedbackMsg::Agent(event));

            if let Err(e) = writer.append(&envelope).await {
                tracing::error!("failed to append event {} to log: {}", id, e);
            }

            // Durability barrier (P2 Phase B): commit at TurnCompleted and
            // signal the supervisor's run_turn barrier.
            if let AgentEvent::TurnCompleted { turn_id, .. } = &envelope.event {
                if let Err(e) = writer.commit().await {
                    tracing::error!("event-log commit at turn {turn_id} failed: {e}");
                }
                if let Some(tx) = &commit_tx {
                    let _ = tx.send(*turn_id);
                }
            }
        }
    })
}

/// P3 wake todo gate: fold `TodosUpdated` into the scheduler — any todo
/// still `pending`/`in_progress` keeps background wakes armed; an
/// all-completed list (`note_todos(false)`) mutes pending and future
/// wakes (the result stays queryable via `task_status`/`/jobs`).
fn fold_wake_todos(event: &AgentEvent, wake: Option<&WakeScheduler>) {
    if let AgentEvent::TodosUpdated { todos } = event
        && let Some(sched) = wake
    {
        let incomplete = todos
            .iter()
            .any(|t| matches!(t.status, TodoStatus::Pending | TodoStatus::InProgress));
        sched.note_todos(incomplete);
    }
}
