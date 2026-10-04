//! Event fanout: session log, IPC, and TUI state (no stdout streaming).

use crate::ipc_pending::{ApprovalPendingMap, QuestionPendingMap};
use crate::tui::elm::feedback::TuiFeedbackMsg;
use nca_common::event::{AgentEvent, EventEnvelope};
use nca_runtime::ipc::IpcHandle;
use nca_runtime::supervisor;

/// Disk + IPC + TUI state; starts IPC command consumer when needed.
/// `commit_tx` receives the turn id of each `TurnCompleted` committed
/// (flush + fsync) to the log — the supervisor's `run_turn` barrier.
///
/// Thin delegation to [`nca_runtime::session_utils::spawn_event_fanout`]:
/// the TUI feedback channel installs as the `on_event` callback, which the
/// runtime fires between the IPC broadcast and the disk append — so the
/// TUI-first latency contract (UI never waits on per-event disk I/O)
/// survives the delegation, while the bridge gains the runtime fanout's
/// step-liveness watch and commit-on-channel-close durability.
#[allow(clippy::too_many_arguments)]
pub fn spawn_tui_bridge(
    rx: tokio::sync::mpsc::Receiver<AgentEvent>,
    log_path: std::path::PathBuf,
    ipc_handle: Option<IpcHandle>,
    approval_pending: Option<ApprovalPendingMap>,
    question_pending: Option<QuestionPendingMap>,
    feedback_tx: tokio::sync::mpsc::UnboundedSender<TuiFeedbackMsg>,
    commit_tx: Option<tokio::sync::watch::Sender<u64>>,
    step_stall_alarm: std::time::Duration,
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

    // TUI-first: the runtime fanout invokes this callback BEFORE the
    // per-event append, so the Elm model's feedback channel never waits
    // on disk I/O during high-frequency streaming.
    let on_event = Box::new(move |envelope: &EventEnvelope| {
        let _ = feedback_tx.send(TuiFeedbackMsg::Agent(envelope.event.clone()));
    });

    nca_runtime::session_utils::spawn_event_fanout(
        rx,
        log_path,
        event_tx_ipc,
        Some(on_event),
        None, // no parent forward on the TUI path
        commit_tx,
        step_stall_alarm,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Delegation pin (fanout consolidation): a TUI-bridge-shaped session
    /// drives the runtime fanout with the feedback channel installed as
    /// `on_event` — a StepStarted bracket plus one interval of
    /// paused-clock silence must land the step_stalled alarm BOTH in the
    /// log the delegated fanout writes AND on the TUI feedback channel.
    /// Driving idioms mirror the runtime watch tests: +50ms overshoot so
    /// the alarm deadline fires first, then a bounded wall wait for the
    /// blocking-pool fs append to become visible to a std read.
    #[tokio::test(start_paused = true)]
    async fn tui_bridge_delegates_step_watch_to_runtime_fanout() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = dir.path().join("events.jsonl");
        let (event_tx, event_rx) = tokio::sync::mpsc::channel(8);
        let (feedback_tx, mut feedback_rx) =
            tokio::sync::mpsc::unbounded_channel::<TuiFeedbackMsg>();
        let mut task = spawn_tui_bridge(
            event_rx,
            log.clone(),
            None,
            None,
            None,
            feedback_tx,
            None,
            std::time::Duration::from_secs(600),
        );

        event_tx
            .send(AgentEvent::StepStarted {
                turn_id: 7,
                step_index: 1,
            })
            .await
            .expect("send step started");

        tokio::time::sleep(std::time::Duration::from_secs(650)).await;

        // Log observation: step + exactly one alarm line, durable.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let lines = loop {
            let l: Vec<String> = std::fs::read_to_string(&log)
                .unwrap_or_default()
                .lines()
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.to_string())
                .collect();
            if l.len() >= 2 || std::time::Instant::now() > deadline {
                break l;
            }
            tokio::task::yield_now().await;
        };
        assert_eq!(lines.len(), 2, "step + one alarm: {lines:?}");
        assert!(lines[1].contains("step_stalled"), "{lines:?}");

        // Feedback-channel observation: the alarm reached the Elm side as
        // a Checkpoint{step_stalled} Agent event.
        let mut saw_alarm = false;
        while let Ok(msg) = feedback_rx.try_recv() {
            if let TuiFeedbackMsg::Agent(AgentEvent::Checkpoint { phase, .. }) = msg
                && phase == "step_stalled"
            {
                saw_alarm = true;
            }
        }
        assert!(saw_alarm, "alarm must reach the TUI feedback channel");

        drop(event_tx);
        let exited = tokio::time::timeout(std::time::Duration::from_secs(5), &mut task).await;
        assert!(exited.is_ok(), "bridge fanout drains and exits cleanly");
    }
}
