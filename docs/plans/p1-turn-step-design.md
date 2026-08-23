# P1 Design — Turn/Step Layering + Single Inbox

> Implements §P1 of `deepseek-harness-adoption.md`. This doc fixes the public
> API contract before any code. Status: approved-for-implementation pending review.

## Problem recap

`run_turn` is a blocking call. While it runs, no task can accept user input on
its behalf (the TUI `cmd_rx` loop is parked inside `run_turn().await`), so
steering and queued prompts must ride side channels (`question_answer_tx`) or
wait for turn end. There is no turn/step event structure for P2/P4 to hang on.

## Public API (contract-first)

### 1. `nca-common` — events (additive, old logs replay)

```rust
pub enum AgentEvent {
    // ... existing variants unchanged ...
    TurnStarted   { #[serde(default)] turn_id: u64 },
    StepStarted   { #[serde(default)] turn_id: u64, #[serde(default)] step_index: u64 },
    StepCompleted { #[serde(default)] turn_id: u64, #[serde(default)] step_index: u64,
                    #[serde(default)] duration_ms: u64, #[serde(default)] had_tool_calls: bool },
    StepFailed    { #[serde(default)] turn_id: u64, #[serde(default)] step_index: u64,
                    #[serde(default)] duration_ms: u64, #[serde(default)] error: String },
    TurnCompleted { #[serde(default)] turn_id: u64, #[serde(default)] duration_ms: u64 }, // +turn_id
    MessageReceived { role: String, content: String,
                      #[serde(default)] steering: bool },  // +steering
}
```

- `turn_id`: monotonic per-`AgentLoop` counter (1-based). **Survives resume**:
  `AgentLoop::set_turn_seq_start(n)` seeds the counter; the supervisor scans
  the event log on resume (max `TurnStarted.turn_id`, fallback 0) so ids stay
  session-unique across restarts. New sessions start at 0 → first turn = 1.
- `StepFailed` closes the bracket when a step errors (stream error, budget,
  pipeline error, empty-after-retries) — no dangling `StepStarted` for P2
  replay / P4 middleware.
- `Checkpoint.turn` (existing, 1-based per `run_turn`) is exactly
  `step_index`; kept as-is, migration deferred to P2.
- All new fields `#[serde(default)]` → old event logs deserialize.
- Match-site updates (add `..` or the field) are compiler-driven: `stream.rs`,
  `tui/elm/components/transcript.rs`, `tui/state.rs`, `tui/elm/model.rs`,
  `session_utils.rs`, plus construction sites in `agent.rs`.

### 2. `nca-core` — inbox + driver (`agent_driver.rs`, new)

```rust
/// Item delivered to a running turn's inbox; claimed at step boundaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InboxItem {
    /// Full user prompt queued while a turn was running. Appended as a user
    /// message at the next step boundary (in arrival order).
    UserPrompt { text: String },
    /// Mid-turn steering guidance. Same delivery as `UserPrompt`, but the
    /// `MessageReceived` event carries `steering: true` for UI distinction.
    Steering { text: String },
}
```

`AgentLoop` additions (all other fields/methods unchanged):

```rust
impl AgentLoop {
    /// Handle for enqueueing prompts/steering into the *running or next* turn.
    /// Bounded (16). `try_send` failure = "inbox full" (caller surfaces it).
    pub fn inbox_sender(&self) -> tokio::sync::mpsc::Sender<InboxItem>;
}
```

- Channel created in `AgentLoop::new` (capacity 16), receiver held on the loop,
  survives across turns (leftovers are claimed at next turn start — see below).
- `TurnDriver` is `pub(crate)` — internal structure, not public API. One
  implementation, no seams.

### 3. Driver semantics (replaces `run_turn_inner`)

```
run_turn(user_input, workspace_root, attachments) -> Result<String>   // signature unchanged
  1. baseline = messages.len(); turn_id = ++counter; emit TurnStarted { turn_id }
  2. drain inbox leftovers (race items from previous turn) THROUGH the same
     claim-and-emit helper as the loop (MessageReceived emitted, steering flag
     per kind) — appended BEFORE the new user message, arrival order
  3. push initial user message, emit MessageReceived (steering=false)   // existing behavior
  4. TurnDriver loop:
       owed = true
       loop {
         claimed = inbox.try_recv-drain()            // step boundary claim
         if !owed && claimed.is_empty() { break }    // turn complete
         append claimed via claim-and-emit (same helper); owed = true
         if cancelled → Err (existing message)       // existing cancel semantics
         step_index += 1; emit StepStarted
         match step() {                              // ONE provider call + its tool pipeline
           ok(had_tool_calls) → emit StepCompleted; owed = tool_calls || retry
           Err(e) → emit StepFailed { error }; return Err(e)
         }
       }
  5. on Err: messages.truncate(baseline) — NOT pop-count (claimed messages sit
     mid-history; popping the tail would strip assistant/tool pairs and orphan
     tool_calls). Residual orphans are repaired by `sanitize_tool_call_pairs`
     on the next request.
  6. emit TurnCompleted { turn_id, duration_ms }, BusyStateChanged::Idle  // existing
```

- `step()` = the body of today's per-iteration code, unchanged in behavior:
  budget check (`max_turns` = step budget per turn, as today), prepare/sanitize/
  compaction, `provider.chat`, stream loop with cancel poll, attachments
  cleanup (first step), empty-response handling (Retry outcome), assistant
  message push, tool pipeline + keepalive, consecutive-failure detection.
- Steering does NOT interrupt an in-flight stream — claimed at boundaries only
  (dsh semantics: injected context waits for claim).
- Budgets: steering-extended steps count against `max_turns` (documented;
  config rename out of scope).

### 4. `nca-runtime` / `nca-cli` wiring

- `Supervisor::inbox_sender()` → delegates to agent. `SessionRuntime::inbox_sender()` likewise.
- Resume seeding: supervisor scans the session's events.jsonl on resume for
  max `TurnStarted.turn_id` and calls `agent.set_turn_seq_start(n)`.
- TUI (Elm): `SideEffectChannels` gains
  `inbox_tx: Option<tokio::sync::mpsc::Sender<InboxItem>>` (same bypass pattern
  as `question_answer_tx`, wired in `repl.rs::run_with_tui`).
  **Routing predicate is an authoritative shared `Arc<AtomicBool>` busy flag**
  (set in the `cmd_rx` loop immediately before `run_turn().await`, cleared on
  return) — NOT the bridged `BusyStateChanged` event, which is documented as
  racy (`repl.rs:2020`). Composer Submit while `busy && !active_question` →
  `try_send` to inbox + queued-count state; Submit while idle → existing
  `TuiCmd::Submit` path unchanged.
- Queued count (status bar hint, driver is authoritative): increment on local
  busy-Submit enqueue, decrement on each `MessageReceived` observed while
  busy (claims), reset on `TurnCompleted`.
- `stream.rs` (ndjson/stream mode): print `StepCompleted` as a dim one-liner
  (`step 2/… tool-calls`) and include `turn_id` in the `TurnCompleted` line;
  `StepFailed` prints like `Error` (dim, one line); `TurnStarted`/`StepStarted`
  no-op (too chatty).
- IPC `SendMessage` during busy: out of scope (unchanged).

## Deviations from plan doc (deliberate, documented)

1. **No `InboxItem::QuestionAnswer`.** The turn task blocks inside the tool
   pipeline awaiting the answer oneshot; it cannot drain its own inbox, so an
   inbox-routed answer deadlocks by construction. Answers keep using the
   existing `QuestionPendingMap` oneshot (`SessionRuntime::submit_question_answer`),
   which already works from REPL, TUI, and IPC. An inbox variant that a broker
   task translates into the same oneshot call is indirection with no new
   capability — rejected (YAGNI).
2. **stdio REPL stays blocking-input.** Concurrent reedline reading during a
   turn is a UX project of its own. Steering surface for P1 = TUI composer +
   the inbox API. Acceptance criterion 3 ("queued input consumed in order") is
   proven at the runtime level (supervisor + scripted provider).
3. **`MessageReceived.content` stays a preview for the initial user message**
   (existing behavior). P2's "model-visible means logged" needs full-content
   surface events; that schema decision (e.g. a distinct surface event or a
   `content_full` field) belongs to P2, where the replay projection is built.
   Claimed inbox items DO emit full text (user-typed, bounded).

## Review record

Oracle review (approve-with-changes) incorporated: truncate-to-baseline
rollback (blocker), `StepFailed` variant, resume-safe `turn_id` seeding,
 claim-and-emit unification for turn-start drain, authoritative busy flag for
 TUI routing, queued-count semantics, `Checkpoint.turn` ≡ `step_index` note.
 Verdict on deviations and YAGNI scope: clean.

## Test matrix (maps to plan acceptance criteria)

| # | Test | File | Criterion |
|---|------|------|-----------|
| T1 | 3-step scripted turn; steering injected during step 2; step-3 request contains steering msg | `crates/core/tests/turn_driver.rs` | 1 |
| T2 | queued UserPrompt claimed in order at boundary | same | 1 |
| T3 | event sequence TurnStarted → Step×N(+Completed, had_tool_calls) → TurnCompleted, turn_id consistent | same | — |
| T4 | leftover inbox item claimed at next `run_turn` start | same | — |
| T5 | old event log serde replay (missing turn_id/steering fields) | same | — |
| T6 | existing agent.rs tests unchanged (no edits) | `crates/core/src/agent.rs` | 2 |
| T7 | supervisor-level: enqueue while turn in flight → consumed in order within that turn | `crates/runtime/tests/inbox.rs` | 3 |
| T8 | question-answer path regression (existing tests only) | existing | 4 |
