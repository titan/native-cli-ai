# Middleware Chain Composition — observability → cost-guard → compaction → retry

> Implements the roadmap remainder of `deepseek-harness-adoption.md` §P4
> ("中间件顺序（初始）：observability → cost-guard → compaction → retry").
> Prerequisites merged: P4 seam (`middleware.rs`, M1–M9), P3
> (`OverflowRecoveryMiddleware`, C1–C11). Status: **design** — oracle
> review record at the bottom once run.

## Problem

The chain seam exists and has exactly one production consumer
(`OverflowRecoveryMiddleware`, wired at `supervisor.rs:593`). Three
deferred items are now due, plus one new capability:

1. **Routine smart compaction is still inline in `step()`**
   (`agent_driver.rs:295-322`): the driver builds the request view via
   `plan_context_view` and emits the legacy `ContextCompaction` event.
   P3 explicitly deferred this migration (its Non-goals: "migrating the
   driver's routine smart-compaction emission into a middleware").
   Roadmap acceptance criterion 2 wanted compaction logic in
   `middleware.rs`.
2. **No general retry.** `ProviderError::RateLimited { retry_after_ms }`
   is produced by providers (`provider.rs:123`) and consumed by
   **nothing** — a single rate-limit rejection fails the whole turn.
   The roadmap's chain ends in a retry middleware; P3's recovery only
   handles overflow.
3. **No cost guard.** `CostTracker` accumulates totals
   (`cost.rs`) but no budget exists anywhere in config or code; the
   only budget is `max_turns_per_run` (step count). A session can spend
   unbounded dollars.
4. **No config surface for the chain** — both P3 and P4 Non-goals say
   "config surface for chain composition (programmatic only for now)".
   With three knob-bearing middlewares landing, "now" has arrived.

## Current state (verified 2026-08-24)

- Chain: `MiddlewareChain::call` at the single `chat()` site
  (`agent_driver.rs:328-345`); empty by default; `AgentLoop` builders
  `with_middleware` / `push_middleware` (`agent.rs:104-118`).
- Single construction site for parent, resumed, and subagent-child
  sessions: `supervisor.rs:568-593`.
- `StepRequest { messages, tools, model, workspace_root, turn_id,
  step_index, event_tx }` — `Clone`; middlewares may emit informational
  events but never `MessageRecorded` (P4 §4).
- Inline in `step()` today: smart-compaction view build + legacy
  `ContextCompaction` emission (`agent_driver.rs:295-322`); cost fold on
  `StreamChunk::Usage` → `agent.cost_tracker.add` + `CostUpdated` emit
  (`agent_driver.rs:445-459`); empty-response retry;
  `max_turns` budget check (`agent_driver.rs:266`).
- `plan_context_view(&[Message], SmartCompactionMode) ->
  ContextViewPlan { messages, report }` (`context_view.rs:356`);
  `SmartCompactionMode::{Off,On,DryRun}`, `is_enabled() == !Off`
  (`config.rs:1827`). Config home: `[memory.context]
  smart_compaction_mode`.
- `set_smart_compaction_mode` has exactly one production caller
  (`supervisor.rs:587`, construction-time) and one test caller
  (`core/tests/overflow_recovery.rs:376`, DryRun).
- `RateLimited` handling: none anywhere (grep-clean outside provider
  internals and a negative classification test).
- `CostUpdated` is informational-only; on resume the tracker is re-seeded
  from the last log entry (`supervisor.rs:295`), so budget state survives
  session restarts for free.
- `estimated_cost_usd()` (`cost.rs`) uses hard-coded Sonnet-class rates
  and approximates cache accounting — adequate for a threshold, not an
  invoice. Guard semantics must say so.

## Design

### 1. Three new middlewares in `core/src/middleware.rs`

#### 1a. `CompactionMiddleware` (migration, not new behavior)

Owns `mode: SmartCompactionMode`. Behavior is **verbatim** the current
inline block, relocated:

```rust
async fn call(&self, mut req, next) {
    if !self.mode.is_enabled() { return next.run(req).await; }
    let plan = plan_context_view(&req.messages, self.mode);
    let report = &plan.report;
    if report.tokens_after < report.tokens_before || self.mode == DryRun {
        emit ContextCompaction { phase: "dry_run"|"completed", ..report } via req.event_tx;
    }
    if self.mode == On { req.messages = plan.messages; }
    next.run(req).await
}
```

- The driver stops building `request_messages` and stops emitting
  `ContextCompaction`; `StepRequest.messages` becomes the canonical
  post-`prepare`/`sanitize` history (`agent.messages.clone()` — same
  clone cost as today's Off path).
- Event ordering is preserved: `ContextCompaction` still fires after
  `Checkpoint("provider_request")` and before the provider call. It is
  informational-only, so replay is unaffected.
- **Mode ownership moves** from `AgentLoop` to the middleware:
  `smart_compaction_mode` field + getter/setter on `AgentLoop` are
  removed (single construction-time caller); the DryRun test migrates to
  pushing `CompactionMiddleware::new(SmartCompactionMode::DryRun)`.
- Keepalive note (P4 §3 forward-compat): no regression — the driver
  already sends the divergent view today; the middleware diverges at the
  same point.

#### 1b. `RetryMiddleware` (transient failures, tight classification)

```rust
pub struct RetryMiddleware { max_attempts: u32 }   // retries after the first attempt
```

- **Classification v1: `ProviderError::RateLimited` only.** The variant
  is structurally unambiguous and carries `retry_after_ms` — no body
  sniffing, mirroring the overflow-pattern discipline
  (`provider.rs:130-147`): a false positive wastes a retry, so patterns
  are extended only with verified signals. 5xx/network-substring
  matching is a documented non-goal.
- On `RateLimited { retry_after_ms }` with attempts remaining: sleep
  `min(retry_after_ms, delay_cap_ms)`, then re-run `next` with the
  **unmodified** request (`Next: Clone`). On exhaustion or non-retryable
  errors: pass the error through unchanged.
- Sleep uses `tokio::time::sleep`. **Today both compat parsers emit a
  hard-coded `retry_after_ms: 1000`** (`openai_compat.rs:275-277`,
  `anthropic_compat.rs:285-287`) — no `Retry-After` header is parsed —
  so `retry_delay_cap_ms` is forward-looking protection for when header
  parsing lands (a non-goal here), and the production sleep is a fixed
  1 s. Tests exercise the cap with a scripted provider
  (`retry_after_ms: 600_000`), which is the only way to reach it.

#### 1c. `CostGuardMiddleware` (session spend cap)

- Pre-request check: if accumulated session cost ≥ budget, fail the
  step **loudly** (`ProviderError::Other("session cost budget
  exhausted: $X ≥ $Y (config: [middleware] cost_budget_usd)")`) before
  the provider is called. 0 provider calls after the trip.
- **Semantics: `Err`, not short-circuit `FinalText`.** Budget exhaustion
  is a policy failure the caller must see (CLI surfaces the error;
  headless orchestrators need a non-zero signal). A synthetic answer
  would masquerade as model output. The existing error path covers
  surfacing (`StepFailed { error }`).
- Trip basis: `estimated_cost_usd()` of the accumulated tracker —
  **estimate-grade by construction** (`cost.rs:38-46` hard-codes
  Sonnet-class rates, 3.0/1M in + 15.0/1M out, and its own doc admits
  OpenAI-compatible `input_tokens` includes cached tokens, so cache
  reads are double-counted). For DeepSeek — the primary provider — the
  estimate overstates actual spend roughly **10×**. Consequences made
  explicit on every user-facing surface: the error text reads
  `"estimated session cost budget exhausted: $X ≥ $Y (Sonnet-class rate
  estimate; actual spend may be lower, e.g. ~10× for DeepSeek; config:
  [middleware] cost_budget_usd)"`, and the config field's doc-comment
  states the basis. The key keeps the name `cost_budget_usd` for
  vocabulary consistency with the existing `CostUpdated
  .estimated_cost_usd` surface users already see (rename to
  `cost_budget_estimated_usd` was considered and rejected — see review
  record). A model-aware rate table is a non-goal.
- **Resume determinism:** on a replay-format resume the tracker is
  re-seeded from the last log `CostUpdated` (`supervisor.rs:295-308`),
  so a tripped session **re-trips immediately** on its next provider
  call — terminal until the budget is raised. Spend stays spent
  (monotonic tracker); money cannot be un-spent.
- **Seed gaps (pre-existing, now budget-relevant):** the re-seed drops
  `cache_creation_tokens` (`CostUpdated` has no such field →
  post-resume guard undercounts by the cache-creation component), and
  legacy `Snapshot`-path resumes never seed the tracker (budget resets
  to 0 for old-format logs). Both noted, neither fixed here.
- **No `Arc<Mutex<CostTracker>>`.** `StepRequest` gains a
  `Copy` snapshot:

```rust
/// Session usage accumulated so far (driver-materialized, per request).
#[derive(Clone, Copy, Debug, Default)]
pub struct SessionUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cache_read_tokens: u64,
}
```

  The driver fills it from `agent.cost_tracker` when building each
  `StepRequest` (4 `u64`s, zero staleness across steps). The guard holds
  only `budget_usd: f64` + a pricing fn; ownership of the tracker stays
  driver-internal. The guard computes cost via the same rate table
  `CostTracker::estimated_cost_usd` uses — expose a
  `CostTracker::estimated_cost_for(usage: SessionUsage) -> f64` (or
  make the existing fn take the 4 fields) so there is exactly one rate
  table.

### 2. Composition: `default_chain`, config, wiring

```rust
/// Compose the roadmap's initial chain (fixed order, knob-bearing
/// middlewares conditional on config): cost-guard → compaction →
/// overflow-recovery → retry (innermost, wraps `chat()` directly).
pub fn default_chain(cfg: &MiddlewareConfig) -> MiddlewareChain
```

- **Order is code-fixed** (roadmap order minus observability, see §3) and
  honors P3's documented nesting contract — "recovery goes
  innermost-of-policy / outermost-of-retry"
  (`p3-compaction-design.md`): recovery is *policy* (it changes what is
  sent — prunes), retry is *mechanism* (re-sends the same thing after a
  wait), so retry sits innermost, directly wrapping the terminal. Error
  flow under this order: overflow → passes retry (non-retryable) →
  caught by recovery, which prunes and re-runs `next` (= retry +
  terminal), so each prune rung gets rate-limit protection for free;
  `RateLimited` → passes recovery untouched → caught by retry, which
  sleeps and re-runs the terminal only (prune state preserved — the
  arm-intent of the roadmap's "溢出恢复作为 retry 的一个 arm" realized
  as nesting). No user-reorderable list — reordering is a footgun and
  nobody has a use case; the config surface carries knobs only.
- `cost-guard` pushed only when `cost_budget_usd = Some(_)`; `retry`
  only when `retry_max_attempts > 0`; `compaction` always (Off =
  pass-through, one branch); `overflow-recovery` always.
- Drain accessor: `MiddlewareChain::into_middlewares() ->
  Vec<Arc<dyn AgentMiddleware>>` (private field today) +
  `AgentLoop::extend_middleware(chain: MiddlewareChain)`; `names()`
  iterates in the same order the drain yields (outermost first).

Config (`common/src/config.rs`, new top-level section):

```toml
[middleware]
retry_max_attempts = 2        # 0 disables the retry middleware
retry_delay_cap_ms = 30000    # cap on server-provided retry-after
cost_budget_usd = 5.0         # absent = no guard
```

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiddlewareConfig {
    pub retry_max_attempts: u32,        // default 2
    pub retry_delay_cap_ms: u64,        // default 30_000
    pub cost_budget_usd: Option<f64>,   // default None
}
```

Wiring (`supervisor.rs:593`): replace the single
`push_middleware(OverflowRecoveryMiddleware)` with
`agent.extend_middleware(default_chain(&config.middleware))`. The
construction site stays single — subagent children inherit the full
chain automatically. **Budget semantics under inheritance:** the budget
is *per session* — each child (`subagent.rs:107` funnels through
`Supervisor::create`) gets its own fresh `CostTracker` and its own
independent `cost_budget_usd`; a parent orchestrating N children can
spend up to (N+1)× the configured budget, and a tripped child surfaces
`ProviderError::Other("…budget exhausted…")` as its turn error through
the subagent protocol. No cross-session aggregate limit exists —
intentionally (v1).
`smart_compaction_mode` stays in `[memory]`
(existing home, no migration).

### 3. What is deliberately NOT built: an observability middleware

The roadmap's outermost slot has no gap to fill:

- Per-step brackets + timing already exist and are replay-folded:
  `StepStarted` / `StepCompleted { duration_ms, had_tool_calls }` /
  `StepFailed { duration_ms, error }` (driver-emitted).
- Spend/context telemetry already exists informationally: `CostUpdated`
  (per usage fold), `ContextStatsUpdated { estimated_tokens,
  context_window, usage_percent }`.
- A chain-level duplicate would re-emit the same facts one frame early —
  ceremony without new signal. The one genuinely missing metric,
  provider time-to-first-token (chain-level timing isolates provider
  latency from tool-pipeline time inside `StepCompleted.duration_ms`),
  does not justify a pass-through middleware today; it is listed as a
  future option.

If oracle disagrees, the fallback shape is a trivial outermost recorder
emitting `Checkpoint { phase: "provider_request_size", detail: "≈N
tokens, M tools" }` — but the default recommendation is skip.

### 4. What does NOT change

- P4 §3 exclusions verbatim: keepalive, supervisor auto-summarize,
  `tool_pipeline`, `prepare_messages_for_request`,
  `sanitize_tool_call_pairs` stay outside the chain.
- Mid-stream interception (`StreamChunk::Error`) stays impossible —
  rate limits arrive as `chat()`-level `Err` (`RateLimited`), so retry
  works; mid-stream failures remain out of scope.
- `plan_context_view` itself, prune ladder, `SmartCompactionMode`
  semantics, replay fold set, `AgentEvent` variants (no new ones),
  `AgentLoop::new` signature, empty-default-chain invariant (M1/M8).
- Driver-owned projection discipline (`MessageRecorded` only from the
  driver); the new middlewares emit nothing on the happy path
  (compaction emits the same legacy event it emits today).
- Empty-chain behavior; `AgentLoop::new` signature, empty-default-chain
  invariant (M1/M8).
- **Test-breakage scope of the `SessionUsage` field (explicit):** the
  four literal `StepRequest` constructions in `middleware.rs` tests
  break and get a mechanical `session_usage: SessionUsage::default()` —
  `fn req()` at `middleware.rs:396` (breaks M1–M5) and the C4/C5a/C5b
  literals at `:668`/`:733`/`:780`. M6–M9 live in
  `core/tests/middleware.rs` and never construct `StepRequest`
  literally, so they survive untouched, as do turn_driver / phase_c /
  agent inline suites. The DryRun setter migration
  (`overflow_recovery.rs:376`) remains the only semantic test change.

## Test matrix

Unit (`middleware.rs` inline):

| # | Test |
|---|------|
| K1 | compaction On + compactible history: inner provider sees the pruned view; `ContextCompaction{phase:"completed"}` on `event_tx`; canonical (outer) messages untouched |
| K2 | compaction Off: provider sees canonical, zero events |
| K3 | compaction DryRun: provider sees canonical, event with `phase:"dry_run"` fires |
| R1 | `RateLimited` once → retry succeeds: 2 provider calls, identical request both times |
| R2 | non-retryable error → single attempt, `Err` unchanged |
| R3 | attempts exhausted → `Err` after `1 + max_attempts` calls |
| R4 | retry-after respected but capped: scripted provider says 10 min (`retry_after_ms: 600_000`), cap 30 s → elapsed ≪ 600 s (test-only scenario — production parsers hard-code 1000 ms today) |
| G1 | usage snapshot over budget → `Err` naming the budget, provider never called |
| G2 | under budget → pass-through, provider called once, request untouched |
| G3 | `default_chain` composition: names in order `["cost-guard","compaction","overflow-recovery","retry"]` with all knobs on; guard/retry absent when disabled |
| G4 | cost computed from the shared rate-table fn equals `CostTracker::estimated_cost_usd` for the same usage |

Integration (`core/tests/`, mock `Provider`):

| # | Test |
|---|------|
| W1 | wiring pin: `run_turn` with smart compaction On + RecordingProvider → provider observes the compacted view; legacy `ContextCompaction` in the event stream (relocates today's inline-behavior assertions; fails if `step()` bypasses the chain) |
| W2 | budget trip end-to-end: tiny budget, pre-seeded usage → `run_turn` fails with the budget message, provider 0 calls, P1 rollback leaves the pre-turn baseline |
| W3 | rate-limit end-to-end: provider fails `RateLimited{retry_after_ms: 1}` once → turn succeeds, exactly 2 calls |
| W4 | resume: budget state re-seeded from log `CostUpdated` (existing seed path) — guard still trips on a resumed session. **Scoped to replay-format sessions** (P2 Phase B logs); legacy `Snapshot`-path resumes do not seed the tracker |

Existing suites (turn_driver, overflow_recovery, phase_c, middleware
M1–M9, agent inline) stay green; mechanical fixture updates from the
`SessionUsage` field (§4) touch M1–M5 + C4/C5a/C5b literals, and the
DryRun setter migration touches `overflow_recovery.rs:376` — the only
semantic change.

## Rollout order

1. Design doc committed (this file) **before** dispatching lanes.
2. Oracle review → findings applied → amend doc commit.
3. Fixer lane: `SessionUsage` + `StepRequest` field + rate-table fn
   extraction (`cost.rs`), three middlewares + `default_chain` +
   `names()` + `into_middlewares()`, driver `step()` simplification,
   `AgentLoop` field/method removal + `extend_middleware`, config
   section, supervisor wiring, fixture updates (M1–M5, C4/C5a/C5b,
   DryRun test migration). Unit tests K1–G4.
   `CARGO_TARGET_DIR=<main>/target`, warm artifacts.
4. Tester lane (cross-model): W1–W4 integration; report deviations only.
5. Ponytail pass over the full diff; `cargo fmt --all -- --check`,
   `cargo clippy --workspace -- -D warnings`, `cargo test --workspace`
   (≥900 s timeout, output to file).
6. Commit `feat(core): compose middleware chain (cost-guard, compaction, retry)`; doc-sync same-commit: `docs/architecture.md` module map,
   roadmap §P4 composition note. (No dep changes → tech-stack untouched.)

## Non-goals

- Observability middleware (§3 — driver events already cover it).
- 5xx / network-error retry classification (body sniffing; extend only
  with verified signals).
- Mid-stream (`StreamChunk::Error`) interception or usage folding in
  middleware.
- User-reorderable chain config (order is code-fixed).
- Budget semantics beyond session-accumulated cost (per-turn budgets,
  hard stop-mid-stream).
- Migrating overflow recovery INTO the retry middleware (separate type
  ratified by P3; nesting realizes the "arm" intent).
- Subagent-specific chain overrides.
- `Retry-After` header parsing in the compat parsers (production
  `retry_after_ms` stays a hard-coded 1000 ms; the cap guards the
  future).
- Model-aware rate table for the cost guard (single Sonnet-class table
  stays; error text carries the caveat).

## Open questions — resolved by oracle review (2026-08-24)

All five confirmed as designed:

1. Budget trip = `Err` (not `FinalText`): rollback to baseline is
   *desirable* — replay drops the failed bracket entirely
   (`replay.rs:46-55`), zero `MessageRecorded` residue, resumed sessions
   re-trip deterministically. `FinalText` would pollute model-visible
   history and masquerade as model output.
2. `SessionUsage` snapshot confirmed; `Arc<StdMutex<CostTracker>>`
   rejected (breaks single-writer discipline, locks across `await`,
   hands mutable state to stateless middlewares).
3. `RateLimited`-only v1 confirmed — `RequestFailed` lumps all
   non-401/403/404/429 bodies; 5xx substrings would re-bill the full
   prompt against likely-deterministic failures.
4. Observability skip confirmed sufficient.
5. `default_chain` in core confirmed (runtime placement forfeits
   unit-testability — `SupervisorConfig` has no provider injection
   seam, the known candidate ② limitation).

## Oracle review record (2026-08-24)

Verdicts: §1a sound · §1b needs change (fixed) · §1c needs change
(fixed) · §2 needs change (fixed) · §3 sound · §4 needs change (fixed)
· test matrix needs change (fixed). **No P0.** Both retry/recovery
nestings traced functionally correct; all findings applied:

- **P1-1** `SessionUsage` field breaks the four literal `StepRequest`
  constructions (`middleware.rs:396/668/733/780` → M1–M5 + C4/C5a/C5b);
  "all tests untouched" claim corrected, mechanical fixture updates
  added to rollout. M6–M9 confirmed surviving (never construct
  `StepRequest` literally).
- **P1-2** original draft ordered recovery innermost, silently
  reversing P3's documented "outermost-of-retry" contract. **Adopted
  P3's order** (`cost-guard → compaction → overflow-recovery → retry`)
  with the policy/mechanism rationale; overruling was the alternative.
- **P1-3** pricing realism: Sonnet-class rates overstate DeepSeek ~10×;
  cache double-count on OpenAI-compatible input. Applied as explicit
  caveat in error text + config doc-comment + §1c; key rename
  (`cost_budget_estimated_usd`) considered and **rejected** for
  vocabulary consistency with `CostUpdated.estimated_cost_usd`;
  model-aware rate table listed as non-goal.
- **P2-1** subagent budget = per-session, (N+1)× aggregate possible —
  stated in §2 wiring.
- **P2-2** resume re-seed drops `cache_creation_tokens` — noted in §1c.
- **P2-3** legacy `Snapshot` resume never seeds the tracker — W4 scoped.
- **P2-4** empty-response `StepOutcome::Retry` re-enters `step()` →
  fresh chain traversal, so guard re-checks per re-entry (beneficial)
  and worst case is (1+2)×(1+2) = 9 provider calls per logical step
  (bounded). Noted here.
- **P2-5** `into_middlewares()` drain accessor + `extend_middleware`
  pinned in §2.
- **P2-6** "server-provided retry-after" was factually wrong (both
  parsers hard-code 1000 ms); §1b corrected, cap repositioned as
  forward-looking, R4 marked test-only, header parsing added to
  non-goals.

## Implementation record (2026-08-24)

- Lanes: oracle (design review — record above), fixer (impl + K1–G4
  unit, `d5114d7`), tester (W1–W4 integration; lane hit the 600 s
  timeout with complete green files left in the worktree — recovered,
  validated, and committed by the orchestrator per the lane-timeout
  lesson, `22ee4ef`), ponytail (verdict `net: -25 possible`; test-side
  pair-helper shrink applied as `aef1105`, the
  `SessionUsage::estimated_cost_usd` delete rejected with rationale).
- **Accepted deviation from §2:** `default_chain` takes a second
  parameter `compaction_mode: SmartCompactionMode` — as literally
  specified (middleware-config-only) the configured mode would be
  silently dropped (compaction always Off). The supervisor passes
  `config.memory.context.smart_compaction_mode`; `[memory]` stays the
  mode's config home.
- **W4 exceeded spec:** the tester found the supervisor-level resume
  harness (phase_c/inbox patterns) and wrote the real
  `seed_cost_tracker_from_log` path as
  `crates/runtime/tests/cost_guard_resume.rs` (spec's fallback was a
  core-level essence only, which also landed as `w4_reeseed_*` in
  `core/tests/middleware_composition.rs`).
- Validation at merge: `cargo fmt --all -- --check` ✅, `cargo clippy
  --workspace -- -D warnings` ✅, `cargo test --workspace` **527 passed
  / 0 failed** (511 baseline + 11 unit + 5 integration). One mid-run
  `/dev/shm` quota hit (os error 122, known environment issue) cleared
  by removing `target/debug/incremental`; rerun green.
- Doc sync: `docs/architecture.md` (module map + streaming section),
  roadmap §P4 status note — this commit. No dep changes.
