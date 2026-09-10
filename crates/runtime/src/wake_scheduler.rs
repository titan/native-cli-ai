//! Wake scheduler (P3): debounced parent wake when a background child task
//! reaches a terminal state.
//!
//! Pure state machine plus a trigger closure — no event subscription, no
//! inbound channels. The CLI wires the trigger to its submit path (TUI:
//! `TuiCmd::Submit(wake_text)` through the command queue that serializes all
//! `run_turn` calls), so a wake is an ordinary parent turn. stdio REPL and
//! one-shot modes run P2 foreground defaults instead (no delivery path).
//!
//! Rollback gate: constructing with `enabled = false` makes every entry
//! point a hard no-op (no task is ever spawned, the trigger never fires).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Fire-and-forget wake delivery: receives the fully rendered wake text.
/// The CLI closure must never be awaited, must not block, and must not
/// panic; delivery failure is the closure's business (e.g. `try_send`
/// semantics inside the TUI cmd-queue closure).
pub type WakeTrigger = Arc<dyn Fn(&str) + Send + Sync>;

/// Shared scheduler state behind the `Clone` handle.
struct Inner {
    enabled: bool,
    interval: Duration,
    trigger: WakeTrigger,
    /// A debounce window is open (a terminal is pending commit).
    in_flight: AtomicBool,
    /// A wake has been delivered and not yet consumed by input; further
    /// terminals are suppressed until the next `note_input`.
    delivered: AtomicBool,
    /// Todo mute fold: `None` = todos never tracked; `Some(false)` = all
    /// todos complete (mute pending and future wakes).
    todos_incomplete: Mutex<Option<bool>>,
    /// Window generation, bumped by every `note_input`. A debounce task
    /// commits only while its generation is still current, so a window
    /// canceled by user input can never steal the reservation of a newer
    /// window and deliver its stale wake text.
    generation: AtomicU64,
}

/// Debounced wake scheduler for background subagent terminals (P3).
///
/// The spawn consumer's background arm calls [`WakeScheduler::notify_terminal`]
/// when a detached child reaches a terminal state (completed / cancelled /
/// failed). After the debounce `interval`, at most one wake is delivered via
/// the trigger — coalescing terminals that land inside the window, honoring
/// the todo mute ([`WakeScheduler::note_todos`]), and deferring to any user
/// input ([`WakeScheduler::note_input`]) that arrives first.
#[derive(Clone)]
pub struct WakeScheduler {
    inner: Arc<Inner>,
}

impl WakeScheduler {
    /// Build a scheduler. `enabled: false` is the rollback gate — every
    /// later entry point returns without spawning a task.
    pub fn new(enabled: bool, interval: Duration, trigger: WakeTrigger) -> Self {
        Self {
            inner: Arc::new(Inner {
                enabled,
                interval,
                trigger,
                in_flight: AtomicBool::new(false),
                delivered: AtomicBool::new(false),
                todos_incomplete: Mutex::new(None),
                generation: AtomicU64::new(0),
            }),
        }
    }

    /// Notify that a background child task reached a terminal state.
    ///
    /// Opens (or joins) the debounce window; after `interval` the wake is
    /// delivered unless muted by the todo gate or superseded by user input.
    /// No-op when disabled, when a wake is already awaiting consumption
    /// (`delivered`), or when a debounce window is already open (rapid
    /// terminals coalesce into the pending wake).
    pub fn notify_terminal(&self, child_ref: &str, state: &str, summary: &str) {
        let inner = &self.inner;
        if !inner.enabled
            || inner.delivered.load(Ordering::SeqCst)
            || inner.in_flight.swap(true, Ordering::SeqCst)
        {
            return;
        }
        // This call owns the window: build the wake text (static template)
        // and spawn the debounce task.
        let inner = Arc::clone(&self.inner);
        let text = format!(
            "[wake] Background task {child_ref} reached {state}: {summary}. Reconcile (task_status or /jobs) and continue."
        );
        let generation = inner.generation.load(Ordering::SeqCst);
        tokio::spawn(async move {
            tokio::time::sleep(inner.interval).await;
            // Todo gate: all todos complete ⇒ mute — the result stays
            // queryable via `task_status` / `/jobs`, but no wake fires.
            if let Ok(todos) = inner.todos_incomplete.lock()
                && *todos == Some(false)
            {
                inner.in_flight.store(false, Ordering::SeqCst);
                return;
            }
            // Stale-window guard: if `note_input` canceled this window and a
            // newer one opened, the reservation belongs to that window's
            // task — this one must not touch it nor deliver stale text.
            if inner.generation.load(Ordering::SeqCst) != generation {
                return;
            }
            // Commit via CAS true→false: a lost race means `note_input`
            // ran during the window — user input supersedes the wake.
            if inner
                .in_flight
                .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
            {
                return;
            }
            (inner.trigger)(&text);
            inner.delivered.store(true, Ordering::SeqCst);
        });
    }

    /// Fold of `TodosUpdated`: `incomplete` is `true` when any todo is
    /// pending or in_progress. `Some(false)` (all complete) mutes pending
    /// and future wakes; `Some(true)` or never-called both allow waking.
    pub fn note_todos(&self, incomplete: bool) {
        if let Ok(mut todos) = self.inner.todos_incomplete.lock() {
            *todos = Some(incomplete);
        }
    }

    /// Called on every Submit (user or wake). Clears both gate flags: any
    /// input consumes/obsoletes a queued wake, cancels a pending debounce,
    /// and re-arms the scheduler for the next terminal. Bumping the window
    /// generation retires any debounce task still parked in a canceled
    /// window so it can never deliver its stale wake text later.
    pub fn note_input(&self) {
        self.inner.generation.fetch_add(1, Ordering::SeqCst);
        self.inner.in_flight.store(false, Ordering::SeqCst);
        self.inner.delivered.store(false, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INTERVAL: Duration = Duration::from_millis(1000);

    /// Build an enabled scheduler whose trigger records every delivered
    /// wake text on an unbounded channel.
    fn scheduler(enabled: bool) -> (WakeScheduler, tokio::sync::mpsc::UnboundedReceiver<String>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let trigger: WakeTrigger = Arc::new(move |text: &str| {
            let _ = tx.send(text.to_string());
        });
        (WakeScheduler::new(enabled, INTERVAL, trigger), rx)
    }

    /// Advance virtual time past `interval` and run everything scheduled up
    /// to it. Empirically (this tokio version): `advance` alone never fires
    /// already-registered timers, and parking on a sleep SHORTER than the
    /// pending deadline never reaches it — the reliable pattern is to park
    /// on a sleep strictly longer than every deadline of interest and let
    /// the paused clock auto-advance through them. Pitfall this encodes:
    /// any await can auto-advance to the timer deadline, so tests must NOT
    /// await between `notify_terminal` and a "during the window" action —
    /// the window is pinned synchronously instead.
    async fn elapse(interval: Duration) {
        tokio::time::sleep(interval + Duration::from_millis(50)).await;
    }

    fn fired(rx: &mut tokio::sync::mpsc::UnboundedReceiver<String>) -> Vec<String> {
        let mut out = Vec::new();
        while let Ok(text) = rx.try_recv() {
            out.push(text);
        }
        out
    }

    #[tokio::test(start_paused = true)]
    async fn fires_exactly_once_after_interval_with_expected_text() {
        let (sched, mut rx) = scheduler(true);
        sched.notify_terminal("fixer-1", "completed", "did the thing");

        // Synchronously after the terminal (virtual time frozen at 0, the
        // debounce task still parked on its sleep): nothing fired yet.
        assert!(
            fired(&mut rx).is_empty(),
            "no fire before the debounce interval"
        );

        // Once the interval elapses: exactly one fire, exact template text.
        elapse(INTERVAL).await;
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "exactly one fire: {fires:?}");
        assert_eq!(
            fires[0],
            "[wake] Background task fixer-1 reached completed: did the thing. Reconcile (task_status or /jobs) and continue."
        );

        // No further fires later.
        elapse(Duration::from_secs(10)).await;
        assert!(fired(&mut rx).is_empty(), "still exactly one fire total");
    }

    #[tokio::test(start_paused = true)]
    async fn two_terminals_within_window_coalesce_into_one_fire() {
        let (sched, mut rx) = scheduler(true);
        // Both terminals land while the debounce window is open (the second
        // call sees in_flight=true and joins the pending wake) — pinned
        // synchronously so the paused clock cannot race past the window.
        sched.notify_terminal("a-1", "completed", "first");
        sched.notify_terminal("b-2", "failed", "second");
        elapse(INTERVAL).await;

        // One fire, carrying the FIRST terminal (the one that opened the
        // window).
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "terminals coalesce: {fires:?}");
        assert!(fires[0].contains("a-1"), "window owner's text: {fires:?}");

        // Even far past the window, no second fire for b-2.
        elapse(Duration::from_secs(10)).await;
        assert!(fired(&mut rx).is_empty(), "no delayed second fire");
    }

    #[tokio::test(start_paused = true)]
    async fn note_input_during_window_cancels_the_pending_wake() {
        let (sched, mut rx) = scheduler(true);
        // The window is open (in_flight=true, timer parked); the user
        // submits before it commits.
        sched.notify_terminal("a-1", "completed", "on it");
        sched.note_input();
        elapse(Duration::from_millis(600)).await;
        assert!(
            fired(&mut rx).is_empty(),
            "input during the window cancels delivery"
        );

        // The scheduler is re-armed: a later terminal fires normally.
        sched.notify_terminal("b-2", "completed", "next");
        elapse(INTERVAL).await;
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "re-armed after input: {fires:?}");
        assert!(fires[0].contains("b-2"));
    }

    #[tokio::test(start_paused = true)]
    async fn delivered_suppresses_until_note_input_then_fires_again() {
        let (sched, mut rx) = scheduler(true);

        // First terminal delivers.
        sched.notify_terminal("a-1", "completed", "first");
        elapse(INTERVAL).await;
        assert_eq!(fired(&mut rx).len(), 1);

        // A second terminal while delivered=true is a no-op — max ONE wake
        // queued at any time.
        sched.notify_terminal("b-2", "completed", "second");
        elapse(Duration::from_secs(10)).await;
        assert!(
            fired(&mut rx).is_empty(),
            "delivered flag suppresses further wakes"
        );

        // After the wake is consumed by input, a new terminal fires again.
        sched.note_input();
        sched.notify_terminal("c-3", "cancelled", "third");
        elapse(INTERVAL).await;
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "fires again after note_input: {fires:?}");
        assert!(fires[0].contains("c-3"));
    }

    #[tokio::test(start_paused = true)]
    async fn todo_gate_mutes_on_all_complete_fires_on_incomplete_or_untracked() {
        // Some(false): all todos complete ⇒ muted.
        let (sched, mut rx) = scheduler(true);
        sched.note_todos(false);
        sched.notify_terminal("a-1", "completed", "muted");
        elapse(INTERVAL).await;
        assert!(fired(&mut rx).is_empty(), "Some(false) mutes the wake");

        // Still muted for a later terminal, until todos change.
        sched.notify_terminal("a-2", "failed", "still muted");
        elapse(INTERVAL).await;
        assert!(fired(&mut rx).is_empty(), "mute persists across terminals");

        // Some(true): any todo pending|in_progress ⇒ fires.
        sched.note_todos(true);
        sched.notify_terminal("b-1", "completed", "busy");
        elapse(INTERVAL).await;
        assert_eq!(fired(&mut rx).len(), 1, "Some(true) un-mutes");

        // None: todos never tracked ⇒ fires (conservative default).
        let (sched, mut rx) = scheduler(true);
        sched.notify_terminal("c-1", "completed", "never tracked");
        elapse(INTERVAL).await;
        assert_eq!(
            fired(&mut rx).len(),
            1,
            "never-called todo fold still fires"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn disabled_scheduler_never_fires_and_spawns_no_task() {
        let (sched, mut rx) = scheduler(false);
        sched.notify_terminal("a-1", "completed", "ignored");
        elapse(Duration::from_secs(30)).await;
        assert!(fired(&mut rx).is_empty(), "disabled scheduler never fires");
        // The disabled early return happens before the in_flight reserve —
        // no debounce task was ever spawned.
        assert!(
            !sched.inner.in_flight.load(Ordering::SeqCst),
            "no window reserved when disabled"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn note_todos_false_arriving_during_window_mutes_pending_wake() {
        let (sched, mut rx) = scheduler(true);
        // The window is open (timer parked at t=0); the todo fold lands
        // INSIDE it, before the debounce commits.
        sched.notify_terminal("a-1", "completed", "pending");
        sched.note_todos(false);
        elapse(INTERVAL).await;
        assert!(
            fired(&mut rx).is_empty(),
            "Some(false) during the window mutes the pending wake"
        );

        // The mute path cleared the window: todos turn incomplete again and
        // a new terminal fires normally.
        sched.note_todos(true);
        sched.notify_terminal("b-1", "completed", "next");
        elapse(INTERVAL).await;
        assert_eq!(fired(&mut rx).len(), 1, "re-armed after in-window mute");
    }
}
