//! Wake scheduler (P3/P4): debounced parent wake when a background child
//! task reaches a terminal state.
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
//! Unseen-notes queue: every recorded terminal is pushed into ONE bounded
//! queue ([`MAX_WAKE_NOTES`] notes retained, further terminals counted as
//! overflow) and lives there until a delivery drains it — a terminal is
//! never dropped, only deferred. Two delivery paths drain the same queue:
//!
//! - **Debounce commit** (the normal path): the first terminal after a
//!   quiet period opens a window; `interval` later the queued terminals
//!   commit as one wake. Terminals landing inside the window coalesce —
//!   ALL of them are rendered in the composite, not just the window
//!   owner.
//! - **`note_input` flush** (chained delivery): terminals that landed
//!   while a wake was already queued and unconsumed (`delivered`), or
//!   while the pause latch was set, are deferred into the queue; the next
//!   [`WakeScheduler::note_input`] (the Submit choke point) flushes them
//!   as one composite wake queued behind the input being processed.
//!   Chaining is bounded: every Submit consumes at most one wake, so the
//!   queue can surface at most one extra Submit per consume.
//!
//! No todo mute gate, by design. An earlier revision suppressed wakes
//! whenever the todo list was all-completed, but `notify_terminal` is only
//! ever called for a live background child, so the gate could only ever
//! suppress real unreconciled terminals — exactly the "promised wake never
//! arrives" bug. Wake frequency is bounded by the `delivered` flag (at
//! most one queued wake per input).
//!
//! Pause latch (P4, bounded hold): [`WakeScheduler::pause`] — used by
//! the `wait_for_user` tool when the orchestrator hands control back to
//! the user — holds wake delivery for at most `hold`. A terminal that
//! lands while held is pushed into the unseen-notes queue (it reserves
//! no window and fires nothing) and delivers at the EARLIER of
//! (a) the next [`WakeScheduler::note_input`] (the Submit choke point
//! flushes the queue immediately — user input first, wake behind it),
//! or (b) `hold` expiry, where the release task clears the latch and
//! re-arms a debounce window for any stranded notes. The bound kills
//! the unattended wedge: a `wait_for_user` turn in a headless run has
//! no next Submit to ride, so a purely latch-based defer could hold a
//! promised wake forever. A newer pause supersedes an unexpired
//! release task via the pause epoch. There is deliberately NO public
//! `resume()`: an inverse that only cleared `paused` would be
//! incomplete (stranded notes must also be re-armed, which is exactly
//! what the release task's re-arm does), and un-pausing rides
//! `note_input` and hold expiry alone.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Fire-and-forget wake delivery: receives the fully rendered wake text.
/// The CLI closure must never be awaited, must not block, and must not
/// panic; delivery failure is the closure's business (e.g. `try_send`
/// semantics inside the TUI cmd-queue closure).
pub type WakeTrigger = Arc<dyn Fn(&str) + Send + Sync>;

/// Maximum number of unseen terminal notes retained for one composite
/// wake. Terminals beyond the cap are not dropped: they are counted as
/// overflow and rendered as `(+N more)` so the wake still reports the
/// FULL terminal count. A prompt-size bound only — deliberately not a
/// config knob.
const MAX_WAKE_NOTES: usize = 8;

/// One recorded terminal — the building block of a wake. Fields are kept
/// verbatim (summaries arrive pre-truncated upstream; the scheduler never
/// re-truncates).
struct TerminalNote {
    child_ref: String,
    state: String,
    summary: String,
}

/// Bounded unseen-note queue: the single place every recorded terminal
/// lives between arrival and delivery. Notes are kept (not just counted)
/// up to [`MAX_WAKE_NOTES`] entries; further arrivals bump `overflow` so
/// the rendered composite still reports the true total.
struct NoteQueue {
    notes: VecDeque<TerminalNote>,
    overflow: u32,
}

/// Shared scheduler state behind the `Clone` handle.
struct Inner {
    enabled: bool,
    interval: Duration,
    trigger: WakeTrigger,
    /// A debounce window is open (a terminal is pending commit).
    in_flight: AtomicBool,
    /// A wake has been delivered and not yet consumed by input; further
    /// terminals are DEFERRED into the notes queue (not dropped) until
    /// the next `note_input` flushes them as a chained composite wake.
    delivered: AtomicBool,
    /// Unseen terminal notes: every recorded terminal lives here until a
    /// delivery drains it (debounce commit or `note_input` flush).
    notes: Mutex<NoteQueue>,
    /// Pause latch (P4): while set, no wake may commit — terminals
    /// neither reserve a window nor fire; they queue as unseen notes.
    /// Cleared at the earlier of the next `note_input` or the bounded
    /// hold's release task (there is intentionally no public `resume()`;
    /// see the module docs).
    paused: AtomicBool,
    /// Pause epoch, bumped by every `pause()` and `note_input()`. A
    /// release task clears the latch only while its epoch is still
    /// current, so a stale release can never clear a FRESH pause's
    /// latch.
    pause_epoch: AtomicU64,
    /// Bounded pause hold (P4): the maximum time `pause()` may defer
    /// wake delivery before its release task clears the latch and
    /// re-arms delivery.
    hold: Duration,
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
/// failed). Every terminal is recorded in a bounded unseen-notes queue and
/// delivered exactly once — either `interval` later through the debounce
/// commit (coalescing ALL in-window terminals into one composite), or,
/// when a wake is already queued and unconsumed (`delivered`), as a
/// chained composite flush at the next [`WakeScheduler::note_input`]
/// (user input supersedes a wake's timing, never its delivery).
/// Terminals landing while the bounded pause latch is held deliver at
/// the earlier of that same `note_input` flush or the hold's expiry.
#[derive(Clone)]
pub struct WakeScheduler {
    inner: Arc<Inner>,
}

/// Run `f` against the unseen-note queue. Lock poisoning is treated as
/// unlocked: the queue's invariants (bounded length, exact overflow
/// count) hold in every observable state, so a panicking caller cannot
/// corrupt it and must not wedge delivery.
fn with_notes<T>(inner: &Inner, f: impl FnOnce(&mut NoteQueue) -> T) -> T {
    match inner.notes.lock() {
        Ok(mut guard) => f(&mut guard),
        Err(poisoned) => f(&mut poisoned.into_inner()),
    }
}

/// Push one terminal into the unseen-note queue, enforcing the
/// [`MAX_WAKE_NOTES`] cap: beyond the cap the note is COUNTED, not kept —
/// the overflow count keeps the rendered wake honest about the true
/// number of unseen terminals without growing the prompt unboundedly.
fn push_note(inner: &Inner, note: TerminalNote) {
    with_notes(inner, |queue| {
        if queue.notes.len() < MAX_WAKE_NOTES {
            queue.notes.push_back(note);
        } else {
            queue.overflow = queue.overflow.saturating_add(1);
        }
    });
}

/// Drain the unseen-note queue: take every retained note plus the
/// overflow count, resetting both. Returns `None` when nothing was
/// queued (empty notes and zero overflow).
fn drain_notes(inner: &Inner) -> Option<(Vec<TerminalNote>, u32)> {
    with_notes(inner, |queue| {
        if queue.notes.is_empty() && queue.overflow == 0 {
            return None;
        }
        let notes: Vec<TerminalNote> = std::mem::take(&mut queue.notes).into();
        let overflow = std::mem::take(&mut queue.overflow);
        Some((notes, overflow))
    })
}

/// Render the wake text for a drained queue.
///
/// Exactly one note keeps the single-terminal template byte-identical to
/// the historical wake text (pinned by
/// `fires_exactly_once_after_interval_with_expected_text`). Two or more
/// notes render as a composite: `N` counts ALL unseen terminals — the
/// retained notes PLUS the overflow — and each retained note contributes
/// one `ref state: summary` item; overflow appends `(+N more)` right
/// before the list sentence's final period. Summaries arrive
/// pre-truncated upstream and are never re-truncated here.
fn render_wake(notes: &[TerminalNote], overflow: u32) -> String {
    if let [note] = notes {
        return format!(
            "[wake] Background task {} reached {}: {}. Reconcile (task_status or /jobs) and continue.",
            note.child_ref, note.state, note.summary
        );
    }
    let total = notes.len() + overflow as usize;
    let mut text = format!("[wake] {total} background tasks reached terminal states: ");
    for (i, note) in notes.iter().enumerate() {
        if i > 0 {
            text.push_str("; ");
        }
        text.push_str(&note.child_ref);
        text.push(' ');
        text.push_str(&note.state);
        text.push_str(": ");
        text.push_str(&note.summary);
    }
    if overflow > 0 {
        text.push_str(&format!(" (+{overflow} more)"));
    }
    text.push_str(". Reconcile (task_status or /jobs) and continue.");
    text
}

/// The single idempotent delivery path, drain FIRST / claim SECOND:
/// drain the unseen-note queue, then claim the `delivered` flag (max
/// one queued wake per input) and fire the trigger with the rendered
/// composite text. An EMPTY drain returns WITHOUT claiming — a claim
/// taken with nothing to deliver would sit as a phantom wake and
/// strand every later terminal behind it (`delivered` set while idle
/// is only recoverable through `note_input`; the empty-claim wedge). A
/// LOSING claim — another deliverer won the flag while these notes
/// were drained — re-queues the notes at the FRONT of the queue in
/// original order (plus overflow) and returns without firing: the
/// winner's wake is already on its way, and the pushed-back notes ride
/// the next `note_input` chain flush. A losing claimant never strands
/// notes.
fn commit_delivery(inner: &Inner) {
    let Some((notes, overflow)) = drain_notes(inner) else {
        // Nothing to deliver: leave `delivered` untouched so a later
        // terminal can still claim its own wake.
        return;
    };
    if inner.delivered.swap(true, Ordering::SeqCst) {
        // Lost the claim: put the drained notes back at the FRONT (in
        // original order) so they deliver ahead of anything newer.
        with_notes(inner, |queue| {
            for note in notes.into_iter().rev() {
                queue.notes.push_front(note);
            }
            queue.overflow = queue.overflow.saturating_add(overflow);
        });
        return;
    }
    let text = render_wake(&notes, overflow);
    (inner.trigger)(&text);
}

/// Spawn the debounce task for the just-reserved window — the single
/// spawn tail shared by `notify_terminal` (the window owner) and
/// `ensure_armed` (the pause-expiry re-arm). After `interval`: pause
/// gate → stale-generation guard → in-flight CAS → mid-commit pause
/// check → [`commit_delivery`].
fn spawn_debounce(inner: Arc<Inner>) {
    let generation = inner.generation.load(Ordering::SeqCst);
    tokio::spawn(async move {
        tokio::time::sleep(inner.interval).await;
        // Pause gate: `pause()` may have landed after this window was
        // reserved (the reserve-just-before-pause race) — close the
        // window WITHOUT delivering; the notes stay queued and deliver
        // at the earlier of the next `note_input` flush or the pause's
        // hold expiry (whose release task re-arms them).
        if inner.paused.load(Ordering::SeqCst) {
            inner.in_flight.store(false, Ordering::SeqCst);
            return;
        }
        // Stale-window guard: if `note_input` canceled this window (a
        // newer one may own the reservation), input supersedes the
        // wake's TIMING, not its DELIVERY — deliver through the
        // common idempotent path (unless an input-time flush already
        // claimed the notes).
        if inner.generation.load(Ordering::SeqCst) != generation {
            commit_delivery(&inner);
            return;
        }
        // Commit via CAS true→false: a lost race means `note_input`
        // ran during the window — user input supersedes the wake's
        // timing, so deliver anyway through the common path instead
        // of dropping.
        if inner
            .in_flight
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            commit_delivery(&inner);
            return;
        }
        // Pause landed mid-commit (after the CAS closed the window):
        // the window is closed; the notes stay queued for the hold's
        // release task (or the next `note_input` flush, whichever
        // comes first).
        if inner.paused.load(Ordering::SeqCst) {
            return;
        }
        commit_delivery(&inner);
    });
}

/// Open a debounce window for unseen notes stranded with no live
/// window — the pause-expiry recovery path. No-op when a wake is
/// already queued and unconsumed (`delivered`: the next `note_input`
/// chain flush covers the notes), when a window is already open
/// (`in_flight`: its commit drains them), or when nothing is queued.
fn ensure_armed(inner: &Arc<Inner>) {
    if inner.delivered.load(Ordering::SeqCst) || inner.in_flight.load(Ordering::SeqCst) {
        return;
    }
    let has_unseen = with_notes(inner, |queue| !queue.notes.is_empty() || queue.overflow > 0);
    if !has_unseen {
        return;
    }
    if !inner.in_flight.swap(true, Ordering::SeqCst) {
        spawn_debounce(Arc::clone(inner));
    }
}

impl WakeScheduler {
    /// Build a scheduler. `interval` is the terminal→wake debounce
    /// window; `hold` is the bounded `wait_for_user` wake hold — the
    /// maximum time [`WakeScheduler::pause`] may defer delivery before
    /// its release task clears the latch and re-arms a window.
    /// `enabled: false` is the rollback gate — every later entry point
    /// returns without spawning a task.
    pub fn new(enabled: bool, interval: Duration, hold: Duration, trigger: WakeTrigger) -> Self {
        Self {
            inner: Arc::new(Inner {
                enabled,
                interval,
                hold,
                trigger,
                in_flight: AtomicBool::new(false),
                delivered: AtomicBool::new(false),
                notes: Mutex::new(NoteQueue {
                    notes: VecDeque::new(),
                    overflow: 0,
                }),
                paused: AtomicBool::new(false),
                pause_epoch: AtomicU64::new(0),
                generation: AtomicU64::new(0),
            }),
        }
    }

    /// Notify that a background child task reached a terminal state.
    ///
    /// The terminal is ALWAYS recorded first: it is pushed into the
    /// bounded unseen-notes queue, so from here on it lives in exactly
    /// one place until a delivery renders it — nothing is dropped. Then,
    /// in order: while the pause latch is held it reserves no window
    /// (deferred; delivered at the earlier of the next
    /// [`WakeScheduler::note_input`] flush or the hold's expiry);
    /// while a wake is already queued and unconsumed (`delivered`) it is
    /// likewise deferred into the queue and flushed as a chained
    /// composite at the next `note_input` (no window opens, no timer
    /// fires); while a debounce window is open it coalesces — the open
    /// window's commit drains the queue, so the composite renders ALL
    /// in-window terminals, not just the window owner. Otherwise this
    /// call owns the window and spawns the debounce task. No-op when
    /// disabled.
    pub fn notify_terminal(&self, child_ref: &str, state: &str, summary: &str) {
        let inner = &self.inner;
        if !inner.enabled {
            return;
        }
        push_note(
            inner,
            TerminalNote {
                child_ref: child_ref.to_string(),
                state: state.to_string(),
                summary: summary.to_string(),
            },
        );
        // Pause latch: the note stays queued — no window is reserved and
        // nothing fires; delivery comes at the earlier of the next
        // `note_input` flush (right after the user's turn) or the pause's
        // hold expiry (whose release task re-arms a window).
        if inner.paused.load(Ordering::SeqCst) {
            return;
        }
        // Delivered cap: a wake is already enqueued as a Submit and not
        // yet consumed. The note stays queued (deferred, not dropped) and
        // is flushed as a chained composite at the next `note_input`.
        if inner.delivered.load(Ordering::SeqCst) {
            return;
        }
        if inner.in_flight.swap(true, Ordering::SeqCst) {
            // A debounce window is open: its commit drains the queue, so
            // this terminal joins the pending composite.
            return;
        }
        // This call owns the window: spawn the debounce task.
        spawn_debounce(Arc::clone(inner));
    }

    /// Pause wake delivery (P4, bounded hold). The caller is the
    /// `wait_for_user` tool's injected hook: the orchestrator is handing
    /// control back to the user, so background-child wakes are held for
    /// at most `hold`. Terminals landing while held are DEFERRED into
    /// the unseen-notes queue (no window, no fire) and deliver at the
    /// EARLIER of the next [`WakeScheduler::note_input`] (immediate
    /// flush right after the user's Submit) or this hold's expiry (the
    /// release task clears the latch and re-arms a window). A newer
    /// pause supersedes an unexpired release via the pause epoch.
    /// Harmless on a disabled scheduler — the rollback gate skips the
    /// release task, so the latch simply stays set.
    pub fn pause(&self) {
        let inner = &self.inner;
        inner.paused.store(true, Ordering::SeqCst);
        let epoch = inner.pause_epoch.fetch_add(1, Ordering::SeqCst) + 1;
        if !inner.enabled {
            return;
        }
        let task_inner = Arc::clone(inner);
        tokio::spawn(async move {
            tokio::time::sleep(task_inner.hold).await;
            // Epoch guard: a newer pause (or a note_input) superseded
            // this one — leave the FRESH latch set; the fresh pause's
            // own release task owns the re-arm.
            if task_inner.pause_epoch.load(Ordering::SeqCst) == epoch {
                task_inner.paused.store(false, Ordering::SeqCst);
            }
            // Re-arm unconditionally, even when superseded: if
            // `delivered` is set the note_input chain flush covers the
            // notes; if a window is open its commit drains them; if a
            // fresh pause holds, the debounce task's pause gate stays
            // quiet and the fresh release re-arms.
            ensure_armed(&task_inner);
        });
    }

    /// Called on every Submit (user or wake). Clears the gate flags: any
    /// input consumes/obsoletes a queued wake, cancels a pending debounce
    /// (timing only — a canceled window's task still delivers through the
    /// common idempotent path), re-arms the scheduler for the next
    /// terminal, and releases the P4 pause latch (retiring any pending
    /// release task). Then flushes whatever terminals accumulated unseen
    /// — queued while `delivered` (the chained-delivery case) or while
    /// paused — through the SAME claim as the debounce commit: the
    /// composite wake enqueues as an ordinary Submit behind the input
    /// being processed, so the model learns about every terminal right
    /// after this turn. The flush re-raises `delivered` until that
    /// Submit's own `note_input` consumes it (chaining is bounded: one
    /// extra Submit per consume).
    pub fn note_input(&self) {
        let inner = &self.inner;
        inner.generation.fetch_add(1, Ordering::SeqCst);
        inner.in_flight.store(false, Ordering::SeqCst);
        inner.delivered.store(false, Ordering::SeqCst);
        inner.paused.store(false, Ordering::SeqCst);
        // Retire any release task from an older pause: it must not clear
        // a FRESH pause's latch (this input already cleared `paused`
        // itself, above).
        inner.pause_epoch.fetch_add(1, Ordering::SeqCst);
        // The emptiness pre-check keeps a quiet Submit from re-raising
        // `delivered`; the swap claim inside `commit_delivery` is the
        // real double-delivery lock.
        let has_unseen = with_notes(inner, |queue| !queue.notes.is_empty());
        if has_unseen {
            commit_delivery(inner);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INTERVAL: Duration = Duration::from_millis(1000);
    /// Bounded pause hold used by every pause-family test: long enough
    /// to pin quiet-inside-the-window behavior with intermediate
    /// `elapse` steps, short enough to reach expiry quickly.
    const HOLD: Duration = Duration::from_secs(5);

    /// Build an enabled scheduler whose trigger records every delivered
    /// wake text on an unbounded channel.
    fn scheduler(enabled: bool) -> (WakeScheduler, tokio::sync::mpsc::UnboundedReceiver<String>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let trigger: WakeTrigger = Arc::new(move |text: &str| {
            let _ = tx.send(text.to_string());
        });
        (WakeScheduler::new(enabled, INTERVAL, HOLD, trigger), rx)
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
        // call sees in_flight=true and joins the pending composite) —
        // pinned synchronously so the paused clock cannot race past the
        // window.
        sched.notify_terminal("a-1", "completed", "first");
        sched.notify_terminal("b-2", "failed", "second");
        elapse(INTERVAL).await;

        // One COMPOSITE fire carrying BOTH terminals — the joiner is
        // coalesced into the wake, not overwritten.
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "terminals coalesce: {fires:?}");
        assert_eq!(
            fires[0],
            "[wake] 2 background tasks reached terminal states: a-1 completed: first; b-2 failed: second. Reconcile (task_status or /jobs) and continue."
        );

        // Even far past the window, no second fire — both terminals were
        // drained by the composite.
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
        // fire, still carrying the window owner's text (the input-time
        // flush claimed the queued note; the stale debounce task's commit
        // finds the claim taken and stays silent).
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
    async fn delivered_defers_until_note_input_then_chain_flushes() {
        let (sched, mut rx) = scheduler(true);

        // First terminal delivers.
        sched.notify_terminal("a-1", "completed", "first");
        elapse(INTERVAL).await;
        assert_eq!(fired(&mut rx).len(), 1);

        // A second terminal while delivered=true never fires on its own —
        // no window opens, no timer runs. But it is DEFERRED into the
        // unseen-notes queue, not dropped.
        sched.notify_terminal("b-2", "completed", "second");
        elapse(Duration::from_secs(10)).await;
        assert!(
            fired(&mut rx).is_empty(),
            "delivered flag defers further wakes (no timer fire)"
        );

        // The next note_input — the Submit consuming the a-1 wake —
        // chain-flushes b-2 as one Submit queued behind it.
        sched.note_input();
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "chained flush on note_input: {fires:?}");
        assert!(fires[0].contains("b-2"), "flush carries the deferred b-2");

        // Consume the flushed wake; the queue is empty and a quiet
        // note_input raises nothing.
        sched.note_input();
        assert!(
            fired(&mut rx).is_empty(),
            "a quiet note_input fires nothing"
        );

        // Fully re-armed: a fresh terminal fires on its own again.
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
        // unseen-notes queue instead).
        assert!(
            !sched.inner.in_flight.load(Ordering::SeqCst),
            "paused terminal must not reserve a debounce window"
        );

        // Inside the hold window: quiet — no delivery, no timer.
        elapse(Duration::from_secs(2)).await;
        assert!(fired(&mut rx).is_empty(), "no delivery while held");

        // At hold expiry the release task clears the latch and re-arms a
        // window (ensure_armed): the wake commits one INTERVAL later with
        // NO Submit at all — the unattended-run recovery path.
        elapse(HOLD).await;
        let fires = fired(&mut rx);
        assert_eq!(
            fires.len(),
            1,
            "held notes delivered at hold expiry: {fires:?}"
        );
        assert!(fires[0].contains("a-1"));

        // Exactly one delivery — the queue was drained.
        elapse(Duration::from_secs(10)).await;
        assert!(fired(&mut rx).is_empty(), "no delayed second fire");
    }

    #[tokio::test(start_paused = true)]
    async fn pending_accumulates_all_terminals() {
        let (sched, mut rx) = scheduler(true);
        sched.pause();
        // Both terminals land while paused: BOTH are queued as unseen
        // notes (the old first-terminal-wins slot is gone) — pinned
        // synchronously.
        sched.notify_terminal("a-1", "completed", "first");
        sched.notify_terminal("a-2", "failed", "second");
        sched.note_input();
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "one composite flush: {fires:?}");
        assert!(
            fires[0].contains("a-1") && fires[0].contains("a-2"),
            "flush carries BOTH paused terminals: {fires:?}"
        );

        // Nothing fires later — both terminals were drained.
        elapse(Duration::from_secs(10)).await;
        assert!(fired(&mut rx).is_empty(), "no delayed second fire");
    }

    #[tokio::test(start_paused = true)]
    async fn flushed_wake_delivered_period_defers_into_queue() {
        let (sched, mut rx) = scheduler(true);
        sched.pause();
        sched.notify_terminal("a-1", "completed", "deferred");
        sched.note_input();
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "flushed on note_input: {fires:?}");
        assert!(fires[0].contains("a-1"));

        // The flush re-raises `delivered` until the flushed wake's Submit
        // is consumed: an immediate terminal never fires on its own — but
        // it no longer DISAPPEARS: it queues unseen.
        sched.notify_terminal("b-1", "completed", "capped");
        elapse(Duration::from_secs(10)).await;
        assert!(
            fired(&mut rx).is_empty(),
            "no timer fire while the flush is unconsumed"
        );

        // The next note_input (consuming the flushed wake) chain-flushes
        // b-1 as a queued Submit.
        sched.note_input();
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "b-1 flushes as a chained wake: {fires:?}");
        assert!(fires[0].contains("b-1"));

        // Consume the chained wake; a later terminal fires again.
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
    async fn pause_during_open_window_defers_until_hold_expiry() {
        let (sched, mut rx) = scheduler(true);
        // The window is open (timer parked at t=0); pause lands INSIDE it,
        // before the debounce commits — pinned synchronously so the paused
        // clock cannot race past the window.
        sched.notify_terminal("a-1", "completed", "pending");
        sched.pause();
        elapse(INTERVAL).await;
        assert!(
            fired(&mut rx).is_empty(),
            "pause during the window closes it without delivering"
        );

        // Still inside the hold: a second terminal queues as an unseen
        // note and still reserves no window.
        sched.notify_terminal("a-2", "completed", "still held");
        assert!(
            !sched.inner.in_flight.load(Ordering::SeqCst),
            "held terminal must not reserve a debounce window"
        );
        elapse(Duration::from_secs(2)).await;
        assert!(
            fired(&mut rx).is_empty(),
            "hold is quiet until expiry (or note_input)"
        );

        // Hold expiry: the release task clears the latch and re-arms —
        // BOTH deferred terminals ride the composite, no Submit needed.
        elapse(HOLD).await;
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "delivered at hold expiry: {fires:?}");
        assert!(
            fires[0].contains("a-1") && fires[0].contains("a-2"),
            "composite carries both deferred terminals"
        );

        // The delivered wake is consumed by the next note_input; the
        // scheduler is re-armed.
        sched.note_input();
        sched.notify_terminal("b-2", "completed", "after the wake");
        elapse(INTERVAL).await;
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "re-armed after note_input: {fires:?}");
        assert!(fires[0].contains("b-2"));
    }

    #[tokio::test(start_paused = true)]
    async fn note_input_un_pauses_the_scheduler() {
        let (sched, mut rx) = scheduler(true);
        sched.pause();
        // Terminal while paused is deferred into the notes queue.
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

        // The pause's release task (parked since t=0) was retired by the
        // note_input epoch bump: it wakes at HOLD, clears nothing, and
        // its ensure_armed is a no-op — no duplicate fire ever lands.
        elapse(HOLD).await;
        assert!(
            fired(&mut rx).is_empty(),
            "retired release task never duplicates a wake"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn newer_pause_supersedes_unexpired_release() {
        let (sched, mut rx) = scheduler(true);
        sched.pause();
        sched.notify_terminal("a-1", "completed", "held");

        // Inside the first hold a SECOND pause lands: its epoch bump
        // retires the first release task and restarts the hold clock.
        elapse(Duration::from_secs(2)).await;
        sched.pause();

        // The first release (due at t=5s) fires inside this elapse: the
        // epoch check leaves the FRESH latch set. Its ensure_armed opens
        // a window, but the debounce task's pause gate closes it without
        // delivering — still quiet at t≈7.1s.
        elapse(HOLD).await;
        assert!(
            fired(&mut rx).is_empty(),
            "superseded release must not deliver"
        );

        // The fresh pause's own release (due at t≈7.05s) has cleared the
        // latch and armed a window: the wake commits one INTERVAL later.
        elapse(INTERVAL).await;
        let fires = fired(&mut rx);
        assert_eq!(
            fires.len(),
            1,
            "delivered at the FRESH hold's expiry: {fires:?}"
        );
        assert!(fires[0].contains("a-1"));
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
        // The rollback gate also covers the release task: none was
        // spawned, so the latch stays set even far past the hold.
        assert!(
            sched.inner.paused.load(Ordering::SeqCst),
            "disabled pause spawns no release task — latch stays set"
        );

        // Even the un-pause path stays inert on a disabled scheduler.
        sched.note_input();
        sched.notify_terminal("b-1", "completed", "still disabled");
        elapse(Duration::from_secs(30)).await;
        assert!(fired(&mut rx).is_empty(), "disabled + unpaused never fires");
    }

    // ── Composite rendering ────────────────────────────────────────

    #[tokio::test(start_paused = true)]
    async fn composite_rendering_lists_first_eight_and_counts_overflow() {
        let (sched, mut rx) = scheduler(true);
        // Ten terminals inside one debounce window: the queue keeps the
        // FIRST 8 notes and counts the rest as overflow — pinned
        // synchronously.
        for i in 0..10 {
            sched.notify_terminal(&format!("t-{i}"), "completed", &format!("s{i}"));
        }
        elapse(INTERVAL).await;

        // One composite fire: N counts ALL ten terminals (8 listed + 2
        // overflow), the overflow renders as "(+2 more)" right before the
        // list sentence's final period.
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "one composite fire: {fires:?}");
        assert_eq!(
            fires[0],
            "[wake] 10 background tasks reached terminal states: t-0 completed: s0; t-1 completed: s1; t-2 completed: s2; t-3 completed: s3; t-4 completed: s4; t-5 completed: s5; t-6 completed: s6; t-7 completed: s7 (+2 more). Reconcile (task_status or /jobs) and continue."
        );

        // The overflowed terminals were counted and the queue fully
        // drained: nothing fires later.
        elapse(Duration::from_secs(10)).await;
        assert!(fired(&mut rx).is_empty(), "no delayed fire for overflow");
    }

    #[tokio::test(start_paused = true)]
    async fn delivered_terminals_chain_flush_one_submit_per_consume() {
        let (sched, mut rx) = scheduler(true);
        sched.notify_terminal("a-1", "completed", "first");
        elapse(INTERVAL).await;
        assert_eq!(fired(&mut rx).len(), 1);

        // Terminals keep landing while the a-1 wake is unconsumed: none
        // fire on their own, none are lost.
        sched.notify_terminal("b-2", "completed", "second");
        sched.notify_terminal("c-3", "failed", "third");
        elapse(Duration::from_secs(10)).await;
        assert!(
            fired(&mut rx).is_empty(),
            "no timer fire while the wake is unconsumed"
        );

        // ONE consume flushes ALL unseen terminals as ONE composite —
        // chaining adds exactly one extra Submit per consume, not one per
        // terminal.
        sched.note_input();
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "exactly one chained Submit: {fires:?}");
        assert!(
            fires[0].contains("b-2") && fires[0].contains("c-3"),
            "chained flush carries both deferred terminals: {fires:?}"
        );

        // The chain repeats cleanly: a terminal landing during the
        // flushed wake's delivered period defers again and flushes at the
        // next consume — still exactly one Submit.
        sched.notify_terminal("d-4", "cancelled", "fourth");
        elapse(Duration::from_secs(10)).await;
        assert!(fired(&mut rx).is_empty());
        sched.note_input();
        let fires = fired(&mut rx);
        assert_eq!(fires.len(), 1, "second chain flush: {fires:?}");
        assert!(fires[0].contains("d-4"));
    }
}
