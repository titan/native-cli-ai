//! Stalled-child watchdog (T1-3): periodic parent-side scan of the
//! subagent registry for `Running` children that went silent.
//!
//! Why: [`crate::wake_scheduler::WakeScheduler`] only wakes on a terminal
//! notification. A child that dies without reaching a terminal fold
//! (wedged driver, tokio task lost, silent hang) leaves its registry
//! entry `Running` forever and an unattended background run wedges
//! silently. The watchdog converts "provably no activity for
//! `[subagent] stall_timeout_secs`" into the NORMAL terminal path:
//! `Failed("stalled: …")` through the existing exactly-once registry
//! fold, the `ChildSessionStatusChanged` event on the parent channel,
//! and [`WakeScheduler::notify_terminal`] — debounce, the pause latch,
//! and chained delivery all come free through the existing wake path.
//!
//! Boundaries (spec: core/backend index):
//! - The step-liveness watch (`[session] step_stall_alarm_secs`) is a
//!   CHILD-INTERNAL, alarm-only bracket owned by the event fanout. This
//!   module is a PARENT-side registry scan; it never touches the fanout
//!   watch or its semantics.
//! - The watchdog marks and reports; it never aborts. Flipping the
//!   child's cancel flag is `task_cancel`'s (and kill-all's) business —
//!   a watchdog that killed would conflate "no evidence of life" with
//!   "proof of death" (a legitimately silent 10-minute build must not
//!   be shot on a 300s clock).
//!
//! Skip conditions (a hit requires ALL of):
//! - `state == Running` (terminal entries are naturally deduped — the
//!   exactly-once property comes from the `Running`-only scan plus the
//!   lock-atomic [`crate::subagent_registry::SubagentRegistry::record_terminal_if_running`]);
//! - no pending approval/question ([`SubagentRegistryEntry::interaction_pending`]
//!   — a legal wait, structurally always false for today's
//!   non-interactive children, guarded seam);
//! - no in-flight control lease (`task_cancel`/`task_revive` mid-flight
//!   is legitimate activity — see
//!   [`SubagentRegistry::has_control_lease`]);
//! - `last_activity` older than `stall_timeout_secs` (stamped at
//!   spawn/revive and refreshed by the child's event fanout for EVERY
//!   observed child event — see [`crate::session_utils::ParentForward`]).

use std::sync::Arc;
use std::time::Duration;

use nca_common::event::AgentEvent;
use nca_common::session::ChildSessionState;
use tokio::sync::mpsc;

use crate::subagent_registry::SubagentRegistry;
use crate::wake_scheduler::WakeScheduler;

/// Scan cadence: `clamp(stall_timeout/3, 10s..60s)` — at least three
/// scans per timeout window (a stall is detected within ~⅓ past the
/// bound), never a hot loop, never slower than once a minute.
fn scan_period(stall_timeout: Duration) -> Duration {
    let thirds = stall_timeout.as_secs() / 3;
    Duration::from_secs(thirds.clamp(10, 60))
}

/// Spawn the watchdog scan task (T1-3). `stall_timeout == Duration::ZERO`
/// disables it entirely — no task is spawned. The task holds `Arc` clones
/// of the registry and (optionally) the wake scheduler and parent event
/// channel; it ticks at [`scan_period`] and is parked between ticks, so an
/// idle watchdog costs nothing but a timer.
///
/// Spawned by [`crate::subagent::spawn_subagent_consumer`] (one watchdog
/// per spawn consumer = per parent session), reading the timeout from the
/// live config snapshot once at wiring time. `wake == None` (no wake
/// delivery path — stdio one-shot, service sessions) still folds the
/// registry terminal and emits the status event; only the wake
/// notification is skipped, mirroring how the spawn consumer degrades.
pub fn spawn_stall_watchdog(
    registry: Arc<SubagentRegistry>,
    wake: Option<WakeScheduler>,
    event_tx: Option<mpsc::Sender<AgentEvent>>,
    stall_timeout: Duration,
) -> Option<tokio::task::JoinHandle<()>> {
    if stall_timeout.is_zero() {
        tracing::debug!("[subagent] stall_timeout_secs = 0 — stalled-child watchdog disabled");
        return None;
    }
    let period = scan_period(stall_timeout);
    Some(tokio::spawn(async move {
        let mut ticker = tokio::time::interval(period);
        // After a long parking (laptop sleep, debugger), fire the missed
        // tick once instead of bursting through the backlog.
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            scan_once(&registry, wake.as_ref(), event_tx.as_ref(), stall_timeout).await;
        }
    }))
}

/// One scan pass: fold every stalled `Running` entry to `Failed` and
/// report it. Split from the loop so tests can drive a single pass.
async fn scan_once(
    registry: &SubagentRegistry,
    wake: Option<&WakeScheduler>,
    event_tx: Option<&mpsc::Sender<AgentEvent>>,
    stall_timeout: Duration,
) {
    let now = tokio::time::Instant::now();
    for entry in registry.list() {
        if entry.state != ChildSessionState::Running {
            continue; // Terminal (or pending) entries are naturally deduped.
        }
        if entry.interaction_pending {
            continue; // Legally blocked on a pending approval/question.
        }
        if registry.has_control_lease(&entry.session_id) {
            continue; // A task_cancel/task_revive is mid-flight.
        }
        let idle = now.saturating_duration_since(entry.last_activity);
        if idle < stall_timeout {
            continue;
        }
        let reason = format!("stalled: no activity for {}s", idle.as_secs());
        // Exactly-once transition: the check and the fold share one lock,
        // so a real terminal that raced the scan keeps its true state and
        // re-scans of an already-terminal entry are skipped.
        let Some(after) = registry.record_terminal_if_running(
            &entry.session_id,
            ChildSessionState::Failed,
            Some(reason.clone()),
        ) else {
            continue;
        };
        tracing::warn!(
            child = %entry.session_id,
            alias = ?entry.alias,
            idle_secs = idle.as_secs(),
            "stalled-child watchdog folded a silent Running child to Failed"
        );
        // Same surfacing as the panic-guard terminal fold in
        // `run_prepared_child`: the status event rides the parent channel
        // (must-deliver with a stall alarm — the registry stays
        // re-derivable from the log through `apply_envelope`).
        if let Some(tx) = event_tx {
            nca_core::agent::send_with_stall_alarm(
                tx,
                AgentEvent::ChildSessionStatusChanged {
                    parent_session_id: entry.parent_session_id.clone(),
                    child_session_id: entry.session_id.clone(),
                    state: ChildSessionState::Failed,
                    generation: after.generation,
                    alias: after.alias.clone(),
                    result_summary: Some(reason.clone()),
                },
            )
            .await;
        }
        // Existing wake path — debounce / pause latch / chaining come free.
        if let Some(wake) = wake {
            let child_ref = entry
                .alias
                .clone()
                .unwrap_or_else(|| entry.session_id.clone());
            wake.notify_terminal(&child_ref, "failed", &reason);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scheduler() -> (WakeScheduler, mpsc::UnboundedReceiver<String>) {
        let (tx, rx) = mpsc::unbounded_channel::<String>();
        let trigger: crate::wake_scheduler::WakeTrigger = Arc::new(move |text: &str| {
            let _ = tx.send(text.to_string());
        });
        // Short debounce so the paused test clock can drive delivery.
        (
            WakeScheduler::new(
                true,
                Duration::from_millis(100),
                Duration::from_secs(30),
                trigger,
            ),
            rx,
        )
    }

    async fn elapse(d: Duration) {
        tokio::time::sleep(d).await;
    }

    fn collect(rx: &mut mpsc::UnboundedReceiver<String>) -> Vec<String> {
        let mut out = Vec::new();
        while let Ok(text) = rx.try_recv() {
            out.push(text);
        }
        out
    }

    #[tokio::test(start_paused = true)]
    async fn stalled_child_folds_failed_once_and_wakes_exactly_once() {
        let registry = Arc::new(SubagentRegistry::new());
        registry.record_spawned("p", "c1", "t", "/ws".into(), None);
        registry.set_alias("c1", Some("fixer-1"));
        let (wake, mut wake_rx) = scheduler();
        let (event_tx, mut event_rx) = mpsc::channel(16);

        // Period for a 30s timeout = clamp(10, 10..60) = 10s; the
        // interval's first tick fires immediately (idle = 0 → skip).
        let _watchdog = spawn_stall_watchdog(
            registry.clone(),
            Some(wake),
            Some(event_tx),
            Duration::from_secs(30),
        )
        .expect("timeout > 0 spawns the watchdog");

        // Idle past 30s (ticks at 10/20/30s): the 30s tick is the hit.
        elapse(Duration::from_secs(31)).await;

        let entry = registry.get("c1").expect("entry");
        assert_eq!(entry.state, ChildSessionState::Failed);
        let summary = entry.result_summary.expect("stall reason recorded");
        assert!(
            summary.starts_with("stalled: no activity for 3"),
            "reason carries the idle seconds: {summary}"
        );

        // The status event surfaced on the parent channel.
        match event_rx.try_recv() {
            Ok(AgentEvent::ChildSessionStatusChanged {
                state,
                child_session_id,
                result_summary,
                alias,
                ..
            }) => {
                assert_eq!(state, ChildSessionState::Failed);
                assert_eq!(child_session_id, "c1");
                assert_eq!(alias.as_deref(), Some("fixer-1"));
                assert!(result_summary.is_some());
            }
            other => panic!("expected ChildSessionStatusChanged, got {other:?}"),
        }

        // Exactly one wake, addressed by alias, debounced+delivered.
        elapse(Duration::from_secs(2)).await;
        let fires = collect(&mut wake_rx);
        assert_eq!(fires.len(), 1, "exactly one wake: {fires:?}");
        assert!(
            fires[0].contains("fixer-1") && fires[0].contains("failed"),
            "wake names the alias and state: {}",
            fires[0]
        );

        // Idempotence: further scans never re-transition or re-wake.
        elapse(Duration::from_secs(120)).await;
        assert_eq!(
            registry.get("c1").map(|e| e.state),
            Some(ChildSessionState::Failed)
        );
        assert!(collect(&mut wake_rx).is_empty(), "no second wake");
        assert!(event_rx.try_recv().is_err(), "no second status event");
    }

    #[tokio::test(start_paused = true)]
    async fn child_with_pending_interaction_is_skipped() {
        let registry = Arc::new(SubagentRegistry::new());
        registry.record_spawned("p", "c1", "t", "/ws".into(), None);
        // The legal-wait latch (guarded seam: children are structurally
        // non-interactive today, so only the watchdog contract is proven
        // here — see the module docs).
        registry.set_interaction_pending("c1", true);
        let (wake, mut wake_rx) = scheduler();

        let _watchdog =
            spawn_stall_watchdog(registry.clone(), Some(wake), None, Duration::from_secs(30))
                .expect("watchdog");
        elapse(Duration::from_secs(120)).await;

        assert_eq!(
            registry.get("c1").map(|e| e.state),
            Some(ChildSessionState::Running),
            "interaction-blocked children are never stall-failed"
        );
        assert!(collect(&mut wake_rx).is_empty(), "no wake fired");
    }

    #[tokio::test(start_paused = true)]
    async fn child_with_in_flight_control_lease_is_skipped() {
        let registry = Arc::new(SubagentRegistry::new());
        registry.record_spawned("p", "c1", "t", "/ws".into(), None);
        // A task_cancel/task_revive mid-flight (e.g. revive's bounded
        // cancel-wait) is legitimate activity.
        let _lease = registry.try_acquire_lease("c1").expect("lease");
        let (wake, mut wake_rx) = scheduler();

        let _watchdog =
            spawn_stall_watchdog(registry.clone(), Some(wake), None, Duration::from_secs(30))
                .expect("watchdog");
        elapse(Duration::from_secs(120)).await;

        assert_eq!(
            registry.get("c1").map(|e| e.state),
            Some(ChildSessionState::Running),
            "leased children are never stall-failed"
        );
        assert!(collect(&mut wake_rx).is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn fresh_child_and_heartbeats_defer_the_stall() {
        let registry = Arc::new(SubagentRegistry::new());
        registry.record_spawned("p", "c1", "t", "/ws".into(), None);
        let (wake, mut wake_rx) = scheduler();

        let _watchdog =
            spawn_stall_watchdog(registry.clone(), Some(wake), None, Duration::from_secs(30))
                .expect("watchdog");

        // 20s in: still fresh (idle < 30).
        elapse(Duration::from_secs(20)).await;
        assert_eq!(
            registry.get("c1").map(|e| e.state),
            Some(ChildSessionState::Running)
        );

        // A heartbeat at t=20 restarts the clock: at t=40 the idle time is
        // 20s, not 40s — still running.
        registry.note_activity("c1");
        elapse(Duration::from_secs(20)).await;
        assert_eq!(
            registry.get("c1").map(|e| e.state),
            Some(ChildSessionState::Running),
            "a heartbeat defers the stall window"
        );
        assert!(collect(&mut wake_rx).is_empty());

        // Silence past the full window from the LAST heartbeat: folded.
        elapse(Duration::from_secs(15)).await;
        assert_eq!(
            registry.get("c1").map(|e| e.state),
            Some(ChildSessionState::Failed)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn zero_timeout_disables_the_watchdog() {
        let registry = Arc::new(SubagentRegistry::new());
        registry.record_spawned("p", "c1", "t", "/ws".into(), None);
        let (wake, mut wake_rx) = scheduler();

        assert!(
            spawn_stall_watchdog(registry.clone(), Some(wake), None, Duration::ZERO).is_none(),
            "zero timeout spawns no scan task"
        );
        elapse(Duration::from_secs(600)).await;
        assert_eq!(
            registry.get("c1").map(|e| e.state),
            Some(ChildSessionState::Running),
            "disabled watchdog never scans"
        );
        assert!(collect(&mut wake_rx).is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn real_terminal_racing_the_scan_keeps_its_true_state() {
        // The exactly-once seam: a child that folds a REAL terminal right
        // before the watchdog's pass must not be overwritten to Failed.
        let registry = Arc::new(SubagentRegistry::new());
        registry.record_spawned("p", "c1", "t", "/ws".into(), None);
        let (wake, mut wake_rx) = scheduler();

        // Backdate by letting the clock run past the window without a
        // watchdog running, then fold the real terminal, THEN scan once.
        elapse(Duration::from_secs(120)).await;
        registry.record_terminal("c1", ChildSessionState::Completed, Some("done".into()));

        scan_once(&registry, Some(&wake), None, Duration::from_secs(30)).await;

        assert_eq!(
            registry.get("c1").map(|e| e.state),
            Some(ChildSessionState::Completed),
            "a real terminal survives the scan"
        );
        assert!(
            collect(&mut wake_rx).is_empty(),
            "no wake for the skipped entry"
        );
    }

    #[test]
    fn scan_period_clamps_to_ten_sixty() {
        assert_eq!(
            scan_period(Duration::from_secs(15)),
            Duration::from_secs(10)
        );
        assert_eq!(
            scan_period(Duration::from_secs(30)),
            Duration::from_secs(10)
        );
        assert_eq!(
            scan_period(Duration::from_secs(60)),
            Duration::from_secs(20)
        );
        assert_eq!(
            scan_period(Duration::from_secs(300)),
            Duration::from_secs(60)
        );
        assert_eq!(
            scan_period(Duration::from_secs(3600)),
            Duration::from_secs(60)
        );
    }
}
