# P2 Phase A Design — Event-Sourced Session (read path)

> Implements §P2 Phase A of `deepseek-harness-adoption.md`. Contract-first, like
> the P1 doc. Status: oracle-reviewed (approve-with-changes, incorporated below).

## Problem recap

`<id>.json` (full snapshot, saved after each turn) and `<id>.events.jsonl`
(event stream) are two parallel truths. Resume reads only the json; the event
log is ignored for state. They can drift, the json can be destroyed by a crash
in `create()`'s save-before-restore window, and there is no way to recover a
session whose json is corrupt.

## Scope (Phase A, revised by oracle review)

Phase A builds the **replay projection** and wires it into resume as a
**corruption fallback + divergence detector**. The json snapshot stays
authoritative until Phase B closes the tail-durability gap (async fanout can
drop the last turn's events on clean shutdown — `emit()` awaits the channel
send, not the disk write).

## Event contract

### `MessageRecorded` (new, the ONLY surface event)

```rust
pub enum AgentEvent {
    /// A model-visible message was pushed to `agent.messages`. The replay
    /// projection folds these, in order, to reconstruct conversation state.
    /// Emitted at every push site inside `run_turn`; NOT for system prompts.
    MessageRecorded {
        message: Message,   // full content, parts, tool_calls, tool_call_id
    },
    ...
}
```

Emit sites (immediately after each `messages.push`):
- initial user message (`agent.rs::run_turn`) — full parts, fixes the
  preview-only gap of `MessageReceived`
- claimed inbox user/steering (`agent_driver.rs::claim_inbox`)
- assistant final text (`step()`, no-tool-calls branch)
- assistant with tool_calls (`step()`)
- tool result messages (`step()`, per-result loop)

NOT emitted for: the system prompt (`set_system_prompt` appends and is called
repeatedly; folding would accumulate stale prompts — no replace semantics
exist), error rollbacks (`messages.truncate(baseline)` emits nothing; replay
drops the turn bracket instead).

`MessageReceived` remains unchanged, UI-only. The plan doc's surface whitelist
(`MessageReceived`/`ToolCallCompleted`/`QuestionResolved`) is rejected:
`MessageReceived` has ambiguous provenance (including an IPC `SendMessage`
fallback that never pushes a message — `session_utils.rs`), and
`QuestionResolved` is not a message (the answer reaches history as the
ask_question tool result).

### Replay semantics (`core/src/replay.rs`, new)

```rust
/// Fold surface events into the conversation they produced.
/// Turn-bracket rule: a turn contributes its messages only if its
/// `TurnStarted..TurnCompleted` bracket closed with no `StepFailed`.
pub fn replay_surface_events(envelopes: &[EventEnvelope]) -> Vec<Message>
```

- Keyed on event **type** (`TurnStarted`/`StepFailed`/`TurnCompleted`), never
  on envelope `id` (ids restart per process today).
- Failed turn (any `StepFailed`) → drop the whole bracket. This reproduces
  `run_turn`'s truncate-to-baseline exactly: a failed turn contributes zero
  messages, including its earlier completed steps.
- Crash mid-turn (no `TurnCompleted`) → drop the bracket. Kill-9 resume lands
  on the last completed turn, matching json semantics (saved only after a
  successful turn).
- `MessageRecorded` outside any bracket → dropped (defensive; no push site
  emits there today).
- Old logs (pre-Phase-A, no `MessageRecorded`) → empty projection, no error.
  Old-log **folding is rejected**: the initial user prompt exists only as a
  preview in old logs, so folding would reconstruct wrong history; empty +
  json fallback is strictly more correct.
- Messages are recorded **exactly as pushed** (including `reasoning_content`):
  replay is then ≡ `agent.messages` with no normalization, and reasoning is
  already duplicated in the log via `ReasoningStreamed` deltas. Providers strip
  reasoning on send, so restoring it is provider-inert. (Oracle D7 said
  strip-to-None; recording as-pushed is simpler and strictly more faithful —
  deviation accepted.)

## Resume algorithm (Phase A)

`Supervisor::resume`, in order:

1. Load json (authoritative), as today.
2. Always run `replay_surface_events(read_event_log(path))` (cheap; exercises
   the path; enables 3).
3. If json load failed **or** json messages are empty while the replay is
   non-empty → **replay fallback**: `messages = fresh_system + repaired(replay)`.
   - `fresh_system`: system messages `create()` just pushed. Restored
     messages are filtered to `role != System` first — fixes today's stale
     system prompt restore (`set_system_prompt` pushes; json accumulates them).
   - `repaired`: image parts whose on-disk file no longer exists (deleted by
     `cleanup_processed_attachments` after the message was recorded) collapse
     to the same text placeholder via `MessageContent::strip_image_paths`.
4. Divergence detector: if the log contains any `MessageRecorded` and json
   loaded fine, compare normalized projections (system + reasoning stripped);
   `tracing::warn!` on diff. Free drift alarm, no behavior change.
5. Re-save immediately after restore (before returning). Fixes the
   create()-saves-empty-state window: a crash right after resume no longer
   leaves a wiped json.
6. Token totals: json path unchanged. Replay-fallback path may seed
   `cost_tracker` from the last cumulative `CostUpdated` in the log when a
   setter exists (best-effort; not required for correctness).

## Write-path hardening (1-line, included in Phase A)

`spawn_event_fanout` writes the line and `"\n"` in two `write_all` calls —
combined into one so a torn tail can only ever be a partial line, which the
tolerant reader skips.

## Deviations from plan doc (documented, oracle-approved)

1. **No `surface: bool` on `EventEnvelope`.** With a single surface event type
   the projection matches on type; the flag would be a second derived truth
   with no consumer. Revisit if P3/fork wants a cheap pre-filter.
2. **`MessageRecorded` instead of whitelisting existing events.** See contract
   above.
3. **Json stays authoritative in Phase A; no prefer-replay yet.** Async fanout
   lag can drop the last turn's events on clean shutdown; replay wins only
   after Phase B's flush-at-turn-end. (The plan's Phase A said "resume prefers
   replay" — unsafe pre-B.)
4. **No old-log fold** (preview-only initial prompts make it wrong).
5. **No mtime json-vs-log cache comparison** (YAGNI; both loads are fast).

## Phase B preview (not this phase)

`EventLogWriter` (buffered, line-atomic writes, flush+fsync at turn end),
"model-visible means logged" enforced before `run_turn` returns, resume-safe
envelope ids, then flip resume to replay-authoritative. With flush-at-turn-end,
`TurnCompleted` IS the commit marker — D1's bracket rule needs no separate
persisted marker. Phase C (json only at clean shutdown) must then handle the
second json writer in `append_child_to_parent`.

## Review record (oracle, approve-with-changes — incorporated)

- D1 turn-bracket projection: approved (type-keyed, not id-keyed).
- D2 `MessageRecorded`-only surface set: approved; `QuestionResolved` and
  `ToolCallCompleted` dropped from surface; `MessageReceived` stays UI-only.
- D3 surface flag: cut (YAGNI).
- D4 json-authoritative + replay fallback + divergence warn: approved.
- D5 projection in `core::replay` (pure, common types only), reader in
  runtime: approved. `format_tool_result` export not needed (no fold path).
- D6 totals from last `CostUpdated`: approved (fallback path only).
- D7 replayed reasoning = None: approved.
- Q1 no Phase-B corner-painting; Q2 re-save fix in Phase A; Q3 cross-session
  hazards none live (child logs are separate; parent sees mapped non-surface
  `ChildSessionActivity` only); Q4 tolerant line-skip + single-write hardening.
- Ranked risk #1 (system prompt never logged) → fixed by fresh-system prepend
  on BOTH resume paths (improves today's stale restore).

## Test matrix

| # | Test | Where |
|---|------|-------|
| T1 | multi-step turn (user → assistant+tools → tool results → assistant final): replay ≡ messages minus system, reasoning None | `core/src/replay.rs` |
| T2 | `StepFailed` turn drops ALL its messages (incl. earlier steps) | same |
| T3 | missing `TurnCompleted` (crash cutoff) drops the turn | same |
| T4 | `MessageRecorded` outside a bracket is dropped | same |
| T5 | old log (no `MessageRecorded`) → empty projection, no error | same |
| T6 | emit sites: every model-visible push emits `MessageRecorded` in push order; system prompt does not | `core` (turn_driver tests) |
| T7 | json corrupt + log good → resume via replay fallback (fresh system prepended, missing-image parts repaired) | `runtime/tests` |
| T8 | crash-cutoff log (mid-turn, no TurnCompleted) + corrupt json → resume lands on last completed turn, no orphaned tool_results | `runtime/tests` |
| T9 | resume re-saves immediately: json intact after resume without any turn | `runtime/tests` |
| T10 | divergence warn fires when json and replay differ (log has MessageRecorded) | `runtime/tests` |
| T11 | stale system prompts in json are replaced by the fresh one on resume | `runtime/tests` |
