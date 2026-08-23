//! Event fanout: session log, IPC, and TUI state (no stdout streaming).

use crate::ipc_pending::{ApprovalPendingMap, QuestionPendingMap};
use crate::tui::elm::feedback::TuiFeedbackMsg;
use nca_common::event::{AgentEvent, EventEnvelope};
use nca_runtime::event_log::EventLogWriter;
use nca_runtime::ipc::IpcHandle;
use nca_runtime::supervisor;

struct IpcFanout {
    tx: tokio::sync::broadcast::Sender<String>,
}

/// Disk + IPC + TUI state; starts IPC command consumer when needed.
/// `commit_tx` receives the turn id of each `TurnCompleted` committed
/// (flush + fsync) to the log — the supervisor's `run_turn` barrier.
#[allow(clippy::too_many_arguments)]
pub fn spawn_tui_bridge(
    mut rx: tokio::sync::mpsc::Receiver<AgentEvent>,
    log_path: std::path::PathBuf,
    ipc_handle: Option<IpcHandle>,
    approval_pending: Option<ApprovalPendingMap>,
    question_pending: Option<QuestionPendingMap>,
    feedback_tx: tokio::sync::mpsc::UnboundedSender<TuiFeedbackMsg>,
    commit_tx: Option<tokio::sync::watch::Sender<u64>>,
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
