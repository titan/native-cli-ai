# P2 Phase B Design — Durable Event Log + Replay-Authoritative Resume

> Implements §P2 Phase B of `deepseek-harness-adoption.md`. Prerequisite:
> Phase A (`p2-event-sourced-session-design.md`) is merged (`746ca7f`).
> Status: implemented (lanes A–C; audit findings and design rationale below).

## Problem

Phase A made the event log a fallback truth; the json snapshot remains
authoritative because the log is not durable at turn boundaries: every writer
is fire-and-forget async (`emit()` awaits the channel send, never the disk
write), so a clean shutdown can drop the last turn's events, and a kill -9 can
drop everything since the last incidental flush.

Additionally, flipping resume to replay-authoritative is blocked by four
event-stream blind spots found in the code audit:

1. **Cancelled turns leave a folded-but-rolled-back bracket.** The loop-top
   cancel check in `TurnDriver::run` (`agent_driver.rs`) returns `Err`
   WITHOUT `StepFailed`; `run_turn` then truncates to baseline and emits
   `TurnCompleted`. Replay's bracket rule ("closed with no `StepFailed`
   ⇒ keep") would fold messages the json dropped. (The mid-stream cancel
   path is already covered: its `Err` propagates to `run()`'s match, which
   emits `StepFailed`.)
2. **Context compaction rewrites history outside the event stream.**
   `perform_auto_summarize` assigns `self.agent.messages = compacted /
   apply_summary(...)` at three sites — model-visible state change with no
   surface event. Replay would resurrect pre-compaction history after resume.
3. **Four duplicated log writers, ids restart per process.** `runtime/
   session_utils.rs::spawn_event_fanout` (service/attach + subagent),
   `cli/stream.rs::spawn_event_fanout_task` (one-shot, `--run`, non-TUI
   REPL), `cli/tui/bridge.rs::spawn_tui_bridge` (TUI). Each keeps its own
   `event_id` counter starting at 0 → duplicate ids in one file after
   restart. The two cli writers also still do the two-call line write
   (torn-tail risk Phase A fixed only in runtime).
4. **No turn-end durability barrier.** No writer flushes or fsyncs, ever.

## Design

### 1. `HistoryReplaced` surface event (common + core)

```rust
pub enum AgentEvent {
    /// The conversation history was wholesale replaced (context compaction:
    /// AI summary or sliding window). The replay projection sets its state
    /// to exactly this payload. Emitted between turn brackets; payload is
    /// system-stripped at the emit site (resume always prepends a fresh
    /// system prompt, so stale prompts must not ride along).
    HistoryReplaced { messages: Vec<Message> },
    ...
}
```

Fold semantics (`core/src/replay.rs`): on `HistoryReplaced`, the committed
base becomes the payload verbatim. A pending open bracket is unaffected
(compaction runs between turns; co-occurrence is unexpected but the turn's
records still belong to its bracket). A later failed bracket still drops only
its own messages — the compaction checkpoint stands.

Emit sites: `supervisor.rs::perform_auto_summarize`, immediately before each
of the three `self.agent.messages = ...` assignments (apply-summary,
AI-failed sliding window, empty-summarize sliding window). Payload =
non-system messages of the new history.

### 2. `StepFailed` on the loop-top cancel path (core)

`TurnDriver::run`'s cancel branch emits `StepFailed { turn_id, step_index,
duration_ms: 0, error: "run cancelled" }` after the existing `Error` event,
before returning `Err`. This makes the bracket rule exact: every `Err` out of
`run()` now leaves a `StepFailed` in the bracket, so replay's drop-bracket
reproduces `truncate(baseline)` in all cases.

### 3. `EventLogWriter` (runtime, public)

New `crates/runtime/src/event_log.rs`:

```rust
/// Append-only JSONL event-log writer with turn-end durability.
pub struct EventLogWriter { /* file handle, next_id */ }
impl EventLogWriter {
    /// Open (create+append), seeding `next_id` from the max envelope id
    /// already in the file (resume-safe ids). Missing file → ids start at 1.
    /// Open failure degrades to a no-op writer (errors surface on append).
    pub async fn open(path: &Path) -> Self;
    pub fn next_id(&mut self) -> u64;
    /// Single `write_all(line + "\n")` — a torn tail can only be a partial
    /// line, which the tolerant reader skips.
    pub async fn append(&mut self, envelope: &EventEnvelope) -> std::io::Result<()>;
    /// flush + fsync — the durability barrier.
    pub async fn commit(&mut self) -> std::io::Result<()>;
}
```

Deviation from the Phase B preview ("buffered"): unbuffered per-event writes,
exactly like today; buffering would compromise line-atomicity for no measured
need (event rate is unchanged). One `commit()` per turn.

All three cli/runtime writers are rewired onto `EventLogWriter` (ids seeded,
single-write, shared implementation). This also fixes the cli writers'
torn-tail two-call writes and id collisions.

### 4. Turn-end commit barrier (watch channel)

- `Supervisor` creates `watch::channel::<u64>(0)` — value = last turn id
  whose `TurnCompleted` envelope is durably committed. Supervisor keeps the
  receiver plus a `wired: Arc<AtomicBool>`; the sender (and a flag clone)
  goes out via `SupervisorHandle::take_turn_commit_tx()`, which sets
  `wired = true`.
- Writers take a `commit_tx: Option<watch::Sender<u64>>` parameter. After
  appending a `TurnCompleted { turn_id, .. }` envelope: `commit()`, then
  `send(turn_id)`. On write/commit error or a disabled file: `tracing::
  error!`/`warn!` and STILL publish (liveness over false durability — resume
  falls back to json, divergence detector fires).
- `Supervisor::run_turn_with_images`: after `agent.run_turn(...)` returns
  (Ok or Err — restructured so the barrier runs on both), if `wired`:
  `before = *rx.borrow()` captured before the turn, then await
  `rx.wait_for(|v| *v > before)` with a 5s timeout. Timeout ⇒
  `tracing::error!` (durability violated; json save still proceeds). Not
  wired (tests, embedders without a writer) ⇒ skip.

Turn ids are monotonic per session (`turn_seq`, seeded on resume), so
`> before` identifies exactly this turn's commit.

### 5. Resume flips to replay-authoritative

`select_resume_messages` decision table (replaces Phase A's):

| Log has `MessageRecorded`? | Replay | Json | Result |
|---|---|---|---|
| yes | non-empty | any (incl. corrupt/empty/divergent) | **Replay wins**, repaired |
| yes | empty | usable | Json (Snapshot) |
| yes | empty | corrupt/empty | Error (Phase A behavior) |
| no (old format) | — | usable | Json (Snapshot) — old logs cannot fold |
| no | — | corrupt/empty | Error (Phase A behavior) |

Both paths keep: fresh-system prepend, `repair_missing_images` on the replay
path. `ResumeMessageSource` gains `ReplayAuthoritative`. Divergence warn
flips polarity: when replay wins and json (non-system, normalized) differs,
warn "replay kept".

Json stays a full snapshot written after every turn (Phase C moves it to
clean shutdown and must then handle the second json writer in
`append_child_to_parent`).

## Non-goals (Phase C)

- Json-at-clean-shutdown-only; removing per-turn json saves.
- Graceful writer close on channel end (`fanout_task.abort()` may drop
  trailing UI-only events like `SessionEnded`; turn data is already
  committed at the barrier). Not model-visible; no resume impact.
- Compact envelope format / log rotation.

## Test matrix

| # | Test | Where |
|---|------|-------|
| T12 | `HistoryReplaced` fold: base := payload; later brackets append | `core/src/replay.rs` |
| T13 | `HistoryReplaced` then failed turn → checkpoint stands, turn dropped | same |
| T14 | loop-top cancel emits `StepFailed`; replay drops the bracket (truncate parity) | `core/tests/turn_driver.rs` |
| T15 | writer id seeding: continues from max existing id; fresh file starts at 1 | `runtime/src/event_log.rs` |
| T16 | fanout commits at `TurnCompleted`: after watch fires, disk file parses with the full bracket; ids strictly increasing | `runtime/tests` |
| T17 | barrier: `run_turn` returns only after the turn's `TurnCompleted` is durable on disk | `runtime/tests` |
| T18 | resume flip: fresh log + divergent json → messages from replay | `runtime` (unit + integration) |
| T19 | old-format log + json → Snapshot path unchanged | `runtime` (unit) |
| T20 | log with `HistoryReplaced` + stale json → resume reflects compacted history | `runtime` |
| T21 | cli writers use `EventLogWriter`: single-write + seeded ids (stream task, TUI bridge) | `cli` |

Phase A's `select_resume_messages_snapshot_wins_when_both_non_empty` becomes
T18's inverse and must be updated: snapshot now wins ONLY when the log is
old-format/empty.

## Rollout order

1. Lane A (core+common): `HistoryReplaced` variant, replay fold, cancel
   `StepFailed` — T12–T14.
2. Lane B (runtime+cli): `EventLogWriter`, three-writer rewire, commit
   barrier + wiring — T15–T17, T21.
3. Lane C (runtime): emit sites for `HistoryReplaced`, resume flip,
   divergence polarity, docs — T18–T20.
