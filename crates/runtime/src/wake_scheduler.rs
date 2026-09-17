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
//!
//! No todo mute gate, by design. An earlier revision suppressed wakes
//! whenever the todo list was all-completed, but `notify_terminal` is only
//! ever called for a live background child, so the gate could only ever
//! suppress real unreconciled terminals — exactly the "promised wake never
//! arrives" bug. Wake frequency is already bounded by the `delivered` flag
//! (at most one queued wake per input).
//!
//! Pause latch (P4): [`WakeScheduler::pause`] DEFERS wake delivery until
//! the next [`WakeScheduler::note_input`] (the Submit choke point) — used
//! by the `wait_for_user` tool so background terminals stay quiet while
//! the orchestrator hands control back to the user. A terminal that lands
//! while paused is held in a deferred-wake slot (first terminal wins,
//! coalescing like the debounce window) instead of dropped, and the next
//! `note_input` flushes it immediately after the user's Submit. There is
//! deliberately NO `resume()`: an inverse that only cleared `paused`
//! would be incomplete (a pending window's debounce task must also be
//! re-evaluated), and a resume that also reset the window flags would
//! just duplicate `note_input`. Un-pausing rides `note_input` alone.

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
    /// Deferred-wake slot (P4): wake text retained while the pause latch
    /// is set. First terminal wins (coalesce), mirroring the debounce
    /// window-owner semantics; flushed by the next `note_input`.
    pending: Mutex<Option<String>>,
    /// Pause latch (P4): while set, no wake may commit — terminals
    /// neither reserve a window nor fire; they are DEFERRED into
    /// `pending`. Cleared by the next `note_input` (there is
    /// intentionally no `resume()`; see the module docs).
    paused: AtomicBool,
    /// Window generation, bumped by every `note_input`. A debounce task
    /// commits only while its generation is still current, so a window
    /// canceled by user input can never steal the reservation of a newer
    /// window — input supersedes the wake's TIMING, not its DELIVERY, so
    /// a canceled-window task still delivers (guarded by `delivered`).
    generation: AtomicU64,
}

/// Debounced wake scheduler for background subagent terminals (P3).
///
/// The spawn consumer's background arm calls [`WakeScheduler::notify_terminal`]
/// when a detached child reaches a terminal state (completed / cancelled /
/// failed). After the debounce `interval`, at most one wake is delivered via
/// the trigger — coalescing terminals that land inside the window, deferring
/// to any user input ([`WakeScheduler::note_input`]) that arrives first
/// (input supersedes the wake's timing, not its delivery), and holding any
/// terminal that lands while paused for flush right after the user's next
/// input.
#[derive(Clone)]
pub struct WakeScheduler {
    inner: Arc<Inner>,
}

/// Store `text` in the deferred-wake slot if it is empty — first terminal
/// wins, mirroring the debounce window-owner semantics. Called only while
/// the pause latch is set; the next `note_input` flushes the slot.
fn defer_pending(inner: &Inner, text: &str) {
    if let Ok(mut pending) = inner.pending.lock()
        && pending.is_none()
    {
        *pending = Some(text.to_string());
    }
}

/// Deliver `text` unless a wake is already queued: the `delivered` flag is
/// the max-one-queued-wake cap, and claiming it both marks this delivery
/// and suppresses any concurrent claimant.
fn deliver_guarded(inner: &Inner, text: &str) {
    if !inner.delivered.swap(true, Ordering::SeqCst) {
        (inner.trigger)(text);
    }
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
                pending: Mutex::new(None),
                paused: AtomicBool::new(false),
                generation: AtomicU64::new(0),
            }),
        }
    }

    /// Notify that a background child task reached a terminal state.
    ///
    /// Opens (or joins) the debounce window; after `interval` the wake is
    /// delivered unless held by the pause latch (deferred into the
    /// pending slot, not dropped) or already superseded by user input
    /// (input supersedes the wake's timing, not its delivery). No-op when
    /// disabled, when a wake is already awaiting consumption (`delivered`
    /// — max one queued wake), or when a debounce window is already open
    /// (rapid terminals coalesce into the pending wake). While paused the
    /// terminal is stored in the deferred-wake slot (first wins) and
    /// reserves NO window; the pause check runs BEFORE the `in_flight`
    /// reserve so a paused scheduler never holds a window open.
    pub fn notify_terminal(&self, child_ref: &str, state: &str, summary: &str) {
        let inner = &self.inner;
        if !inner.enabled || inner.delivered.load(Ordering::SeqCst) {
            return;
        }
        let text = format!(
            "[wake] Background task {child_ref} reached {state}: {summary}. Reconcile (task_status or /jobs) and continue."
        );
        // Pause latch: defer the terminal into the pending slot — no
        // window is reserved and nothing fires; the next `note_input`
        // flushes the wake right after the user's turn.
        if inner.paused.load(Ordering::SeqCst) {
            defer_pending(inner, &text);
            return;
        }
        if inner.in_flight.swap(true, Ordering::SeqCst) {
            return;
        }
        // This call owns the window: spawn the debounce task.
        let inner = Arc::clone(&self.inner);
        let generation = inner.generation.load(Ordering::SeqCst);
        tokio::spawn(async move {
            tokio::time::sleep(inner.interval).await;
            // Pause gate: `pause()` may have landed after this window was
            // reserved (the reserve-just-before-pause race) — defer the
            // wake into the pending slot instead of dropping it, and close
            // the window.
            if inner.paused.load(Ordering::SeqCst) {
                defer_pending(&inner, &text);
                inner.in_flight.store(false, Ordering::SeqCst);
                return;
            }
            // Stale-window guard: if `note_input` canceled this window (a
            // newer one may own the reservation), input supersedes the
            // wake's TIMING, not its DELIVERY — deliver now unless a wake
            // is already queued.
            if inner.generation.load(Ordering::SeqCst) != generation {
                deliver_guarded(&inner, &text);
                return;
            }
            // Commit via CAS true→false: a lost race means `note_input`
            // ran during the window — user input supersedes the wake's
            // timing, so deliver anyway (guarded) instead of dropping.
            if inner
                .in_flight
                .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
            {
                deliver_guarded(&inner, &text);
                return;
            }
            // Pause landed mid-commit (after the CAS closed the window):
            // same defer path — the pending slot holds delivery until the
            // next `note_input`.
            if inner.paused.load(Ordering::SeqCst) {
                defer_pending(&inner, &text);
                return;
            }
            deliver_guarded(&inner, &text);
        });
    }

    /// Pause wake delivery (P4). The caller is the `wait_for_user` tool's
    /// injected hook: the orchestrator is handing control back to the
    /// user, so background-child wakes must be held. Terminals landing
    /// while paused are DEFERRED into the pending slot and flushed by the
    /// next [`WakeScheduler::note_input`] — delivered immediately after
    /// the user's next Submit, so while paused the next Submit is
    /// external (the user's) by construction. Harmless on a disabled
    /// scheduler.
    pub fn pause(&self) {
        self.inner.paused.store(true, Ordering::SeqCst);
    }

    /// Called on every Submit (user or wake). Clears both gate flags: any
    /// input consumes/obsoletes a queued wake, cancels a pending debounce
    /// (timing only — a canceled window's task still delivers, guarded by
    /// `delivered`), re-arms the scheduler for the next terminal, and
    /// releases the P4 pause latch. Then flushes the deferred-wake slot:
    /// the retained terminal's wake enqueues as an ordinary Submit behind
    /// the input being processed, so the model learns about the terminal
    /// right after the user's turn; the flush re-raises `delivered` until
    /// that Submit's own `note_input` consumes it.
    pub fn note_input(&self) {
        self.inner.generation.fetch_add(1, Ordering::SeqCst);
        self.inner.in_flight.store(false, Ordering::SeqCst);
        self.inner.delivered.store(false, Ordering::SeqCst);
        self.inner.paused.store(false, Ordering::SeqCst);
        let flushed = self
            .inner
            .pending
            .lock()
            .ok()
            .and_then(|mut pending| pending.take());
        if let Some(text) = flushed {
            self.inner.delivered.store(true, Ordering::SeqCst);
            (self.inner.trigger)(&text);
        }
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
    async fn note_input_during_window_defers_delivery_instead_of_cancelling() {
        let (sched, mut rx) = scheduler(true);
        // The window is open (in_flight=true, timer parked); the user
        // submits before it commits — pinned synchronously so the paused
        // clock cannot race past the window.
        sched.notify_terminal("a-1", "completed", "on it");
        sched.note_input();
        elapse(INTERVAL).await;
        // Input supersedes the wake's TIMING, not its DELIVERY: exactly one
        // fire, still carrying the window owner's text.
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "input defers delivery: {fires:?}");
        assert!(fires[0].contains("a-1"));

        // Consume the delivered wake (the Submit choke point) and the
        // scheduler is re-armed: a later terminal fires normally.
        sched.note_input();
        sched.notify_terminal("b-2", "completed", "next");
        elapse(INTERVAL).await;
        let fires = fired(&mut rx);
        assert_eq!(
            fires.len(),
            1,
            "re-armed after consuming the wake: {fires:?}"
        );
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

    // ── P4 pause latch ─────────────────────────────────────────────

    #[tokio::test(start_paused = true)]
    async fn terminal_while_paused_defers_and_reserves_no_window() {
        let (sched, mut rx) = scheduler(true);
        sched.pause();
        sched.notify_terminal("a-1", "completed", "while paused");

        // The pause check runs BEFORE the in_flight reserve — a paused
        // scheduler holds no window open (the terminal went to the
        // deferred-wake slot instead).
        assert!(
            !sched.inner.in_flight.load(Ordering::SeqCst),
            "paused terminal must not reserve a debounce window"
        );

        elapse(Duration::from_secs(10)).await;
        assert!(fired(&mut rx).is_empty(), "no delivery while paused");

        // The next note_input flushes the deferred wake immediately.
        sched.note_input();
        elapse(INTERVAL).await;
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "deferred pending flushed: {fires:?}");
        assert!(fires[0].contains("a-1"));
    }

    #[tokio::test(start_paused = true)]
    async fn pending_coalesces_first_terminal_wins() {
        let (sched, mut rx) = scheduler(true);
        sched.pause();
        // Both terminals land while paused: the first owns the pending
        // slot, the second coalesces away (window-owner semantics) —
        // pinned synchronously.
        sched.notify_terminal("a-1", "completed", "first");
        sched.notify_terminal("a-2", "failed", "second");
        sched.note_input();
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "pending slot coalesces: {fires:?}");
        assert!(fires[0].contains("a-1"), "first terminal wins: {fires:?}");

        // Nothing fires later — a-2 was coalesced away.
        elapse(Duration::from_secs(10)).await;
        assert!(fired(&mut rx).is_empty(), "no delayed second fire");
    }

    #[tokio::test(start_paused = true)]
    async fn flushed_pending_rearms_delivered_cap() {
        let (sched, mut rx) = scheduler(true);
        sched.pause();
        sched.notify_terminal("a-1", "completed", "deferred");
        sched.note_input();
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "flushed on note_input: {fires:?}");
        assert!(fires[0].contains("a-1"));

        // The flush re-raises `delivered` until the flushed wake's Submit
        // is consumed: an immediate terminal is suppressed (max one queued
        // wake).
        sched.notify_terminal("b-1", "completed", "capped");
        elapse(Duration::from_secs(10)).await;
        assert!(
            fired(&mut rx).is_empty(),
            "delivered cap holds after the flush"
        );

        // Consume the flushed wake (the Submit choke point); a later
        // terminal fires again.
        sched.note_input();
        sched.notify_terminal("c-1", "cancelled", "after re-arm");
        elapse(INTERVAL).await;
        let fires = fired(&mut rx);
        assert_eq!(
            fires.len(),
            1,
            "re-armed after consuming the flush: {fires:?}"
        );
        assert!(fires[0].contains("c-1"));
    }

    #[tokio::test(start_paused = true)]
    async fn pause_during_open_window_defers_until_note_input() {
        let (sched, mut rx) = scheduler(true);
        // The window is open (timer parked at t=0); pause lands INSIDE it,
        // before the debounce commits — pinned synchronously so the paused
        // clock cannot race past the window.
        sched.notify_terminal("a-1", "completed", "pending");
        sched.pause();
        elapse(INTERVAL).await;
        assert!(
            fired(&mut rx).is_empty(),
            "pause during the window defers the wake"
        );

        // Still paused: a second terminal coalesces into the pending slot
        // and still reserves no window.
        sched.notify_terminal("a-2", "completed", "still paused");
        assert!(
            !sched.inner.in_flight.load(Ordering::SeqCst),
            "paused terminal must not reserve a debounce window"
        );
        elapse(INTERVAL).await;
        assert!(fired(&mut rx).is_empty(), "defer holds until note_input");

        // note_input releases the latch and flushes the deferred wake —
        // the FIRST terminal owns the pending slot.
        sched.note_input();
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "flushed on note_input: {fires:?}");
        assert!(fires[0].contains("a-1"), "first terminal owns the slot");

        // Consume the flushed wake (the Submit choke point); the scheduler
        // is re-armed.
        sched.note_input();
        sched.notify_terminal("b-2", "completed", "after the user spoke");
        elapse(INTERVAL).await;
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "re-armed after note_input: {fires:?}");
        assert!(fires[0].contains("b-2"));
    }

    #[tokio::test(start_paused = true)]
    async fn note_input_un_pauses_the_scheduler() {
        let (sched, mut rx) = scheduler(true);
        sched.pause();
        // Terminal while paused is deferred into the pending slot.
        sched.notify_terminal("a-1", "completed", "deferred");
        elapse(INTERVAL).await;
        assert!(fired(&mut rx).is_empty(), "no wake while paused");

        // note_input releases the latch AND flushes the deferred wake.
        sched.note_input();
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "flushed on note_input: {fires:?}");
        assert!(fires[0].contains("a-1"));

        // Consume the flushed wake; un-paused and un-capped, the next
        // terminal fires normally.
        sched.note_input();
        sched.notify_terminal("c-1", "completed", "un-paused");
        elapse(INTERVAL).await;
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "note_input un-pauses: {fires:?}");
        assert!(fires[0].contains("c-1"));
    }

    #[tokio::test(start_paused = true)]
    async fn pause_on_disabled_scheduler_is_a_harmless_noop() {
        let (sched, mut rx) = scheduler(false);
        sched.pause();
        sched.notify_terminal("a-1", "completed", "ignored");
        elapse(Duration::from_secs(30)).await;
        assert!(fired(&mut rx).is_empty(), "disabled scheduler never fires");
        assert!(
            !sched.inner.in_flight.load(Ordering::SeqCst),
            "no window reserved when disabled"
        );

        // Even the un-pause path stays inert on a disabled scheduler.
        sched.note_input();
        sched.notify_terminal("b-1", "completed", "still disabled");
        elapse(Duration::from_secs(30)).await;
        assert!(fired(&mut rx).is_empty(), "disabled + unpaused never fires");
    }
}
