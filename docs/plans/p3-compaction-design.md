# P3 Design — Compaction Event-Sourcing + Overflow Recovery

> Implements `deepseek-harness-adoption.md` §P3. Prerequisites merged: P1
> (step boundary in `agent_driver.rs`), P2 (event-sourced sessions,
> replay-authoritative resume), P4 (middleware chain — this design mounts
> its first real consumer). Status: **reviewed 2026-08-24 — no P0s;
> findings applied (§Oracle review record)**.

## Problem

Two gaps, one roadmap item:

1. **Compaction is not event-sourced.** Three compaction paths exist and
   none of them leaves an auditable bracket on the event log:
   - per-step smart compaction builds a request view inline in
     `TurnDriver::step` (`agent_driver.rs:295-322`) and emits a single
     informational `ContextCompaction { phase }` event;
   - supervisor pre-turn auto-summarize (`maybe_compact_context`,
     `supervisor.rs:974`) and post-turn summarize
     (`check_and_summarize_context`, `supervisor.rs:1016`) replace
     canonical history via `perform_auto_summarize` (`supervisor.rs:1093`)
     and emit ad-hoc `ContextCompaction { phase: "starting"/"completed" }`
     around it (plus `HistoryReplaced`, which is the actual replay
     checkpoint).
   After the fact you cannot reconstruct what was compacted, when, or why.

2. **Context overflow has no recovery path.** A provider rejects an
   oversized request with an HTTP 4xx that lands in
   `ProviderError::RequestFailed(body)` and bubbles straight to the user:
   `chat()` → chain → `step()` → `TurnDriver::run` → `AgentLoop::run_turn`
   → `Supervisor::run_turn_with_images` → caller. Nothing classifies the
   error, nothing retries, nothing compacts harder.

dsh's answer: compaction is a three-event bracket on the event stream
(`compaction/start → summary → compaction/end`) paired with a surface
replace-op, and overflow recovery runs between the failed step and turn
close: prune tool results first, summarize only as a last resort, then
retry.

## Current state (verified 2026-08-24)

- The only canonical provider call site is `TurnDriver::step` →
  `agent.middleware.call(...)` → `Next::run` → `provider.chat`
  (`agent_driver.rs:328-345`). P4's chain is mounted there; the chain is
  empty by default (`MiddlewareChain::default()`), `AgentLoop` exposes the
  consuming builder `with_middleware` (`agent.rs`), and the single
  `AgentLoop::new` construction site in runtime is `supervisor.rs:567`
  (subagent child sessions also funnel through `Supervisor::create`,
  `subagent.rs:107`).
- P4's documented seam (`docs/plans/p4-middleware-design.md` §1): from a
  `StepRequest` a middleware CAN retry the same request, rewrite the
  **request view**, short-circuit, and catch `chat()`-level
  `ProviderError`s. It CANNOT re-plan against canonical history —
  `StepRequest.messages` is already the post-compaction view and
  `plan_context_view` needs `agent.messages` + mode. "A compaction handle
  (or driver-supplied closure) is a future extension."
- Mid-stream overflow (`StreamChunk::Error`) is NOT interceptable (P4
  documented limitation). DeepSeek/OpenAI/Anthropic reject oversized
  prompts with HTTP 400 **before** streaming starts, so overflow arrives
  as `chat()` `Err` — the interceptable path. Stream-level overflow stays
  out of scope.
- `SmartCompactionMode` is `{ Off, DryRun, On }` (`common/config.rs`) — no
  graded/aggressive level exists.
- `plan_context_view` (`context_view.rs:354`) is already pure
  (clone-in/clone-out); the driver decides what to send. Group logic it
  exposes internally: `partition_groups` (atomic assistant+tool_calls+
  results bundles), `is_compactible_tool` (read/search tool allowlist),
  `group_must_keep` (writes, errors, images, directive user text).
- `ProviderError` (`provider.rs:115-131`): `Configuration | RequestFailed
  | AuthError | RateLimited | ModelNotFound | Other`. No classification.
- Replay (`core/replay.rs`) folds `MessageRecorded` / `HistoryReplaced`
  and brackets on `TurnStarted/StepFailed/TurnCompleted`; informational
  events are ignored — adding `AgentEvent` variants is replay-inert.
- The CLI renders no compaction events at all (zero references to
  `ContextCompaction` in `crates/cli`) — new variants are CLI-inert too.
- `run_turn` failure truncates `agent.messages` to the pre-turn baseline
  (P1 rollback policy; see P4 implementation record M7b note). So by the
  time `Supervisor::run_turn_with_images` sees an `Err`, canonical
  history is already back at the baseline.

## Design

### 1. Event bracket (`common/src/event.rs`)

Two new variants for **replacement-type** compactions only (canonical
summarize, overflow recovery) — per oracle P2-2, routine per-step smart /
dry-run compaction keeps emitting the existing single `ContextCompaction`
event (it already carries `tokens_before/after` + phase; a Start/End pair
per step would double event volume for zero audit value, and a bracket
with no observable middle misleads readers into hunting for one):

```rust
/// Replacement-type compaction started (canonical summarize or overflow
/// recovery). Informational; replay-ignored. Pairs with
/// `ContextCompactionEnd`; the middle of the bracket is the replacement
/// itself (`HistoryReplaced` for canonical paths, nothing for view-prune).
ContextCompactionStart {
    #[serde(default)]
    tokens_before: usize,
    /// Why: "auto_summarize" | "overflow_prune" | "overflow_summarize".
    #[serde(default)]
    reason: String,
},
/// Replacement-type compaction finished. `kv_prefix_broken` marks
/// recoveries that changed the request view or replaced history
/// (cache-cold retry).
ContextCompactionEnd {
    #[serde(default)]
    tokens_after: usize,
    #[serde(default)]
    kv_prefix_broken: bool,
},
```

Bracket map (the "middle" event is what the bracket wraps):

| Path | Start reason | Middle | End `kv_prefix_broken` |
|---|---|---|---|
| per-step smart view / dry run (`step`) | — (single `ContextCompaction`, unchanged) | — | — |
| supervisor auto-summarize (3 entry points) | `auto_summarize` | `HistoryReplaced` | `true` |
| overflow view-prune retry (middleware) | `overflow_prune` | — (view mutation, canonical history untouched) | `true` |
| overflow summarize fallback (supervisor) | `overflow_summarize` | `HistoryReplaced` | `true` |

Emission-site conversions:
- `agent_driver.rs:295-322`: **unchanged** — the existing per-step
  `ContextCompaction` emission stays exactly as is (oracle P2-2).
- `supervisor.rs` `maybe_compact_context` (:995) /
  `check_and_summarize_context` (:1033) / overflow fallback (§5):
  each entry emits `Start { reason }`; `perform_auto_summarize` threads
  the reason and emits `End { tokens_after, kv_prefix_broken: true }` on
  **both** exit paths (AI summary and sliding-window fallback — fixes the
  existing asymmetry where the fallback paths emit `HistoryReplaced` but
  no completion event, `supervisor.rs:1098-1105` and `:1135-1153`).

All fields `#[serde(default)]` (forward-compat for partially-written
tails of the log, same convention as P1/P2 additions).

### 2. Overflow classification (`core/src/provider.rs`)

A method, not a variant (roadmap wording; single normalization choke
point, no serde surface change):

```rust
impl ProviderError {
    /// Whether this error is a context-window overflow rejection
    /// (HTTP 4xx raised before streaming started). Recoverable by
    /// compaction + retry.
    pub fn is_context_overflow(&self) -> bool;
}
```

Case-insensitive substring match over the payloads of `RequestFailed` and
`Other` (all providers route non-401/403/404/429 HTTP bodies to
`RequestFailed(body_text)` via `openai_compat`/`anthropic_compat`
`map_provider_error` — verified). Pattern list — deliberately tight,
false positives trigger destructive recovery:

- `"maximum context length"` — OpenAI / OpenAI-compatible (incl.
  DeepSeek, which is served by `OpenAiCompatProvider`)
- `"context length exceeded"` — OpenAI-compatible alt phrasing
- `"prompt is too long"` — Anthropic-compatible (incl. MiniMax, Kimi)
- `"exceed context limit"` — Anthropic alt phrasing ("input length and
  `max_tokens` exceed context limit")

**Coverage is intentionally partial** (oracle P1-1): providers with
non-English error bodies (e.g. ZhipuAI GLM) may phrase overflow
differently and will not match — the error then passes through
unchanged, which is exactly today's behavior (graceful degradation, no
regression). Patterns are extended only with verified literal strings.
Everything else (auth, rate limit, model-not-found, generic 500s) is not
overflow. Unit-tested with the literal provider error strings, embedded
in realistic JSON bodies, plus a representative non-matching overflow
body asserting `false`.

### 3. Tool-result pruning (`core/src/context_view.rs`)

New pure function, the pressure mechanism for view-level recovery —
**delete** whole old tool groups instead of truncating them (cheaper than
a summarizer call, deterministic, no side-call):

```rust
/// Outcome of a prune pass.
pub struct PruneOutcome {
    pub messages: Vec<Message>,
    /// Number of complete tool groups deleted.
    pub pruned_groups: usize,
}

/// Delete (not truncate) the oldest compactible tool groups from a
/// message list, keeping system messages, `must_keep` groups, and the
/// last `keep_recent_groups` groups intact. Never splits a tool group —
/// deletion removes the assistant carrier plus all its results.
pub fn prune_tool_results(messages: &[Message], keep_recent_groups: usize) -> PruneOutcome;
```

- "Compactible group" reuses the existing predicate: a `ToolGroup` whose
  assistant carrier calls only `is_compactible_tool` tools AND is not
  `must_keep` (writes/errors/images stay). The helpers stay private —
  `prune_tool_results` lives in the same module (`context_view.rs`), no
  visibility changes needed.
- Important non-dead-weight property (oracle Q5): `plan_context_view`
  with mode `On` **truncates** old tool outputs to 400 chars but never
  deletes groups — the compacted view still contains every tool group,
  so deletion always has material. And with `Off`/`DryRun` the view is
  the full history clone (`agent_driver.rs:317-319`) — even more
  prunable material.
- Operates on any message list — in recovery it runs against the request
  view (`StepRequest.messages`), so no canonical-history access is needed
  (this is what makes the middleware arm possible despite the P4 seam
  note).
- Honesty note (KV): deleting from the front breaks the warmed KV prefix
  regardless; the win over summarize is no LLM side-call and a
  deterministic result, not cache preservation. `kv_prefix_broken: true`
  on the End event records it.

### 4. View-level recovery: `OverflowRecoveryMiddleware` (`core/src/middleware.rs`)

P4's first real consumer, per the roadmap's "recovery lives in
middleware.rs, not agent.rs":

```rust
/// Catches context-overflow rejections at the `chat()` seam and retries
/// with a progressively pruned request view. Bounded; escalates the
/// original error unchanged when pruning cannot shrink the view further.
pub struct OverflowRecoveryMiddleware {
    /// Max prune-retry attempts per step (roadmap: at most 2).
    max_retries: u32, // default 2
}
```

Loop (per `call`):

```
 est(m) := context_view::estimate_tokens_for_slice(m)   (already pub)
 attempt = 0; keep_recent ladder = [RECENT_GROUPS_KEEP_FULL (8), 2]
loop:
  match next.clone().run(current_req).await:
    Ok(reply)         => return Ok(reply)
    Err(e) if e.is_context_overflow() && attempt < max_retries:
      emit Start{ tokens_before: est(current.messages), reason: "overflow_prune" }
      pruned = prune_tool_results(current.messages, ladder[attempt])
      emit End{ tokens_after: est(pruned.messages), kv_prefix_broken: pruned.pruned_groups > 0 }
      if pruned.pruned_groups == 0: return Err(e)   // no progress → escalate unchanged
      current.messages = pruned.messages; attempt += 1
    Err(e)            => return Err(e)              // non-overflow → untouched passthrough
```

Properties:
- **Single middleware, outermost practical position.** Wired alone for
  now; roadmap's fuller chain (observability → cost-guard → compaction →
  retry) stays future work. As the only middleware it is trivially
  outermost; when more land, recovery goes innermost-of-policy /
  outermost-of-retry — order documented here so the next chain author
  knows.
- `Next: Clone` (P4 deviation record) is what makes re-entry possible.
- Events flow through `req.event_tx` (informational-only convention, P4
  §4) — the middleware never emits `MessageRecorded`/`HistoryReplaced`;
  it only ever rewrites the request view, never canonical history.
- Empty-result guard: if the view is already minimal (all groups recent
  or `must_keep`), `pruned_groups == 0` → immediate escalation, no
  spin. Bound is additionally enforced by `max_retries`.
- Smart-compaction-mode independence: the middleware is wired regardless
  of `SmartCompactionMode`. `Off` governs routine view building, not
  error recovery — an overflow is still an overflow (oracle Q2 ratified;
  with `Off`/`DryRun` the view is the full history clone, so the
  middleware has *more* prunable material, not less).
- Non-overflow errors: one attempt, no events, byte-identical `Err`
  passthrough.

### 5. Canonical fallback: supervisor summarize + one retry (`runtime/src/supervisor.rs`)

Resolves the P4 seam tension ("re-compaction against canonical history
is not reachable from `StepRequest`") by **not reaching for it**: the
view-level arm lives in the middleware (§4); the canonical arm stays
where canonical authority already lives — the supervisor — and triggers
on the error that bubbles out.

In `run_turn_with_images` (`supervisor.rs:839-880`), replace the bare
`let result = self.agent.run_turn(...)` with (durability ordering per
oracle P1-2: each turn's commit barrier waits **before** the next
mutation of canonical history, so a crash can never leave an uncommitted
failed-turn bracket ahead of a `HistoryReplaced` checkpoint):

```
let before = /* existing commit counter capture */;
let first = self.agent.run_turn(prompt, ...).await;
self.await_turn_commit(before).await;               // ALWAYS, ok or err
let output = match first {
  Ok(out)                      => out,
  Err(e) if e.is_context_overflow():
      emit Start{ tokens_before: stats.estimated_tokens, reason: "overflow_summarize" }
      perform_auto_summarize(reason="overflow_summarize").await   // emits HistoryReplaced + End
      let retry_before = /* capture */;
      let second = self.agent.run_turn(prompt, ...).await;
      self.await_turn_commit(retry_before).await;
      second?
  Err(e)                       => return Err(e),
};
```

- Bounded: exactly **one** summarize-retry per `run_turn_with_images`
  call (oracle P2-3: this cap is the real loop bound — the
  `last_summary_at_tokens` guard is best-effort and is NOT set on the
  empty→sliding-window path, `supervisor.rs:1093-1101`). If the retry
  overflows again, the error reaches the caller.
- Ordering is safe: `run_turn`'s Err already rolled `agent.messages`
  back to the pre-turn baseline (`agent.rs:152-157`, truncate at
  `:207-208` — code-verified by oracle; behavior already pinned by the
  M7b integration test, `core/tests/middleware.rs:400-430`), so the
  summarize operates on the baseline; `HistoryReplaced`'s payload is
  computed from true post-rollback state, which makes the replay
  projection self-correcting even if the failed turn recorded messages
  before dying. Replay folds `[failed turn bracket w/ StepFailed]
  [HistoryReplaced][retry turn]` cleanly (oracle Q3: failed brackets are
  dropped by `replay.rs:46-55`, `HistoryReplaced` resets state verbatim).
- `perform_auto_summarize` already shrinks monotonically (AI summary →
  fallback sliding window → no-op when already small). A no-op summarize
  followed by an identical retry is wasted but harmless and bounded; the
  middleware arm (§4) usually prevents reaching here.
- Rejected alternative: a driver-supplied compaction closure / shared
  `Arc<RwLock<Vec<Message>>>` handle on `StepRequest`. It would break
  the single-writer discipline on `agent.messages` or duplicate
  canonical state across an await boundary, for zero functional gain
  over the two-arm split. The supervisor arm is ~25 lines.

Fallback decision extracted as a pure helper for unit testing (full
supervisor E2E with a mock provider is not reachable —
`SupervisorConfig` builds its provider via `build_provider`, no
injection):

```rust
/// Pure: should a run_turn overflow error trigger the summarize-retry?
pub(crate) fn should_overflow_retry(err: &ProviderError, already_retried: bool) -> bool;
```

### 6. Wiring (`core/src/agent.rs` + `supervisor.rs:567`)

- `AgentLoop` gains `pub fn push_middleware(&mut self, middleware: Arc<dyn
  AgentMiddleware>)` (P4 only added the consuming `with_middleware`
  builder; the supervisor constructs the loop via `new` then configures).
- `supervisor.rs:567`, right after construction:
  `agent.push_middleware(Arc::new(OverflowRecoveryMiddleware::default()));`
  Single site; parent sessions, resumed sessions, and subagent children
  all funnel through `Supervisor::create`.
- `AgentLoop::new` default chain remains **empty** — P4's M1/M8
  invariants and every existing core test fixture are untouched. The
  recovery middleware is a supervisor-level policy, not an AgentLoop
  default.

## What does NOT change

- Canonical-history ownership: driver records, supervisor replaces via
  `HistoryReplaced`. The middleware touches views only.
- `plan_context_view` itself (no plan/apply split — see "Deviation"
  below).
- Keepalive (`keepalive_ping`), `tool_pipeline`, `prepare_messages_for_request`,
  sanitize — all still outside the chain (P4 §3).
- Replay folding set (`MessageRecorded`/`HistoryReplaced` + turn
  brackets) — new events are informational.
- CI clippy gate shape, crate boundaries, config surface (no new config
  keys).

### Deviation from roadmap §P3 (call it out, oracle to ratify)

- **No `plan()`/`apply()` split of `plan_context_view`.** The roadmap
  sketched it pre-P4. Today `plan_context_view` is already pure and the
  driver-side "apply" is two lines (choose view vs clone); the bracket
  events (§1) deliver the auditability the split was meant to enable.
  Splitting would add a type and a call boundary with no behavior change.
  Skip.
- **No `SmartCompactionMode::Aggressive`.** Pressure is realized as the
  prune ladder in §4 (shrinking keep-recent window), not a new config
  mode. A new mode would touch config resolution, docs, and every
  `match mode` site for a knob nothing reads.
- **"现有 supervisor 逻辑下放" reinterpreted.** The roadmap wanted
  `perform_auto_summarize` moved down into core so the recovery loop
  could call it. With recovery split two-arm (§4/§5) the supervisor
  already owns exactly the canonical half — nothing moves. Core gets no
  provider-side-call machinery it cannot test.

## Test matrix

| # | Test | Where |
|---|------|-------|
| C1 | `is_context_overflow`: true for the four literal provider strings, case-insensitive, and embedded in realistic JSON error bodies; false for auth/rate-limit/model-not-found/generic 500 **and for a representative non-matching overflow body** (graceful passthrough) | unit, `provider.rs` |
| C2 | `prune_tool_results`: deletes complete compactible groups (assistant carrier + all results); keeps `must_keep` (write tools, error results, images) + system + recent window; `orphaned_tool_results == 0` | unit, `context_view.rs` |
| C3 | prune ladder: `keep_recent=8` prunes less than `keep_recent=2`; already-minimal list prunes 0 groups | unit, `context_view.rs` |
| C4 | middleware: provider overflows once → prune → retry succeeds; 2 provider calls; attempt-2 messages lack pruned groups; `ContextCompactionStart(overflow_prune)` + `End(kv_prefix_broken=true)` in event stream | unit, `middleware.rs` |
| C5 | middleware: nothing prunable → escalates the original overflow `Err` unchanged (≤2 attempts, no successful retry); non-overflow `Err` passes through with 1 attempt and no events | unit, `middleware.rs` |
| C6 | **roadmap acceptance 1 (integration)**: full `AgentLoop` with middleware wired, scripted provider overflow→success → `run_turn` Ok; Start/End pair on the stream; provider saw the pruned view on attempt 2 | `core/tests/` |
| C7 | **roadmap acceptance 3**: DryRun reports planned `tokens_after` decrease via the existing per-step `ContextCompaction` event (assertions unchanged from today — pin that P3 does not regress it) | unit, existing driver/context_view tests |
| C8 | **roadmap acceptance 2**: prune on a group-straddling fixture deletes whole groups; reuse the `assert_no_orphaned_tool_results` helper shape | unit, `context_view.rs` (C2/C8 may merge) |
| C9 | `should_overflow_retry` decision table (overflow+first → true; overflow+already → false; non-overflow → false) | unit, `supervisor.rs` |
| C10 | event serde: Start/End roundtrip; log tail missing `kv_prefix_broken` still deserializes (`#[serde(default)]`) | unit, `event.rs` |
| C11 | **oracle Q1 sequence**: `prune_tool_results` on a post-summarize fixture (system + summary-as-system + recent tool groups) → no orphans, `must_keep` preserved, the summary message never pruned (assistant-without-tool_calls is `GroupKind::Assistant`, never a `ToolGroup`) | unit, `context_view.rs` |

Existing suites stay green untouched: P4's M1–M9 (empty default chain
unchanged — the middleware is supervisor-wired, not default), turn
driver, phase C, replay, context_view, context_manager.

## Rollout order

Same lane pattern as P2/P4:

1. This design doc + oracle review record → commit (worktrees branch
   from main).
2. **fixer lane** (impl + unit tests C1–C5, C10 + supervisor emission
   conversions + wiring): common events → provider → context_view →
   middleware → agent (`push_middleware`) → supervisor. `feat(core):
   compaction event bracket + overflow recovery (P3)` — spans
   common/core/runtime; doc-sync in the same commit per convention.
3. **tester lane** (independent, spec = this doc): integration C6–C8,
   C11 + any matrix gaps it finds; reports deviations, does not patch
   production code.
4. ponytail pass over the full diff; `fmt` + `clippy --workspace --
   -D warnings` + `test --workspace` (mind `/dev/shm` headroom before
   the workspace run; clean `/dev/shm/debug/incremental` if >85%).
5. Doc sync: `docs/architecture.md` (middleware consumer + event
   bracket), mark P3 implemented in `deepseek-harness-adoption.md` —
   same commit as the code per doc-sync rule.

## Non-goals

- Mid-stream (`StreamChunk::Error`) overflow interception (P4
  limitation stands; providers reject pre-stream).
- dsh's sampling-anchored summarize (KV-stable summaries) — the
  `kv_prefix_broken` flag is the honest 80%.
- Config surface for recovery knobs (bounds are constants; no user
  keys).
- Migrating the driver's routine smart-compaction emission into a
  `CompactionMiddleware` (bracket events make it auditable; moving it
  buys nothing until a second policy needs to compose with it).
- Sliding-window / auto-summarize strategy changes.

## Oracle review record (2026-08-24)

Verdicts: §1 sound · §2 needs change (coverage — fixed) · §3 sound ·
§4 sound · §5 needs change (ordering + test gap — fixed) · §6 sound.
**No P0 findings.** All three roadmap deviations **ratified**.

Findings applied to this doc:

- **P1-1** pattern list under-covers non-OpenAI/Anthropic providers
  (ZhipuAI/MiniMax/Kimi bodies may phrase overflow differently). Fixed
  as explicit graceful degradation (§2 "Coverage is intentionally
  partial") + C1 non-matching-body case — patterns extend only with
  verified literals. Variant-surface claim verified: all providers route
  non-401/403/404/429 bodies to `RequestFailed(body_text)` via
  `openai_compat.rs:278` / `anthropic_compat.rs:288`.
- **P1-2** §5 pseudocode originally dropped `await_turn_commit` — fixed:
  each turn's commit barrier now waits before any further canonical
  mutation (§5). Rollback-to-baseline-on-Err ordering claim code-verified
  (`agent.rs:152-157`, `:207-208`) and already pinned by M7b
  (`core/tests/middleware.rs:400-430`).
- **P2-1** DeepSeek pattern attribution corrected (served by
  `OpenAiCompatProvider`; matches "maximum context length").
- **P2-2 adopted** — per-step smart/dry-run compaction keeps the single
  `ContextCompaction` event; Start/End brackets are for replacement-type
  compactions only. This dropped the entire `agent_driver.rs` emission
  conversion from scope.
- **P2-3** loop-bound claim corrected: the one-retry cap is the real
  bound; `last_summary_at_tokens` is best-effort and unset on the
  empty→sliding-window path (`supervisor.rs:1093-1101`).
- **P2-4** `est()` defined as `context_view::estimate_tokens_for_slice`
  (already `pub`).
- Q1/Q5 confirmed the two-arm split and prune ladder sound: prune
  material always exists (compaction truncates but never deletes
  groups; `Off`/`DryRun` views are full clones); summarized views
  cannot orphan or lose the summary (assistant-without-tool_calls is
  `GroupKind::Assistant`, never a pruned `ToolGroup`) — C11 added.
- Q3 confirmed replay folds `[failed turn][HistoryReplaced][retry turn]`
  cleanly (`replay.rs:46-55`).

Deviations from roadmap §P3 — **all ratified** (oracle wording
compressed):

1. No `plan()`/`apply()` split — `plan_context_view` is already pure
   clone-in/clone-out; the driver "apply" is two lines; brackets deliver
   the auditability the split was meant to enable.
2. No `SmartCompactionMode::Aggressive` — pressure is the prune ladder;
   a new mode touches config/docs/every `match mode` site for a knob
   nothing reads.
3. "Supervisor logic pushed down" reinterpreted — P4's seam note makes
   moving `perform_auto_summarize` into core the wrong call (needs
   `agent.messages` + a provider side-call); two-arm keeps canonical
   authority where it belongs and core gets nothing it cannot test.

Also noted by oracle, no doc change required: `partition_groups` /
`is_compactible_tool` / `group_must_keep` are private — fine as is since
`prune_tool_results` lives in the same module (clarified in §3).
