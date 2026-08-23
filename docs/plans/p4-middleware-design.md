# P4 Design — Waterfall Middleware Layer (tower-style, core)

> Implements `deepseek-harness-adoption.md` §P4. Prerequisites merged: P1
> (`agent_driver.rs` step boundary), P2 (event-sourced sessions — the
> projection invariants below assume it). Status: **implemented
> 2026-08-23** — module + wiring + tests M1–M9 all green (`middleware.rs`,
> `agent_driver.rs` step rewiring, `core/tests/middleware.rs`); oracle
> review record and implementation deviations at the bottom.

## Problem

Every cross-cutting concern that touches a provider request — smart
compaction, cost accounting, telemetry, (future) context-overflow retry —
is inline in `TurnDriver::step` (`agent_driver.rs:259`, the former
`run_turn_inner` body absorbed by P1). Adding a new one means editing the
step function itself; nothing can observe, rewrite, or short-circuit a
request without forking the loop.

dsh's answer is a waterfall: plugins wrap `next()` and can intercept,
rewrite, or short-circuit. P4 ports that seam to Rust as an explicit
chain around the single `provider.chat()` call of a step.

## Current state (verified 2026-08-23)

- The only canonical provider call site is `agent_driver.rs:328-336`
  (`.chat(` at :331, inside `step()`): `agent.provider.chat(&request_messages, &agent.tool_definitions(), &agent.model, self.workspace_root)`.
- Request assembly before it: `prepare_messages_for_request`
  (provider-specific rewrite), `sanitize_tool_call_pairs` (orphan repair),
  smart-compaction view build (`plan_context_view`).
- Cross-cutting logic inline in `step()`: ContextCompaction event emission,
  cost tracking (from `StreamChunk::Usage`), empty-response retry policy,
  consecutive-tool-failure tracking, checkpoints.
- The keepalive path never enters the `chat()` seam: `CacheKeepalive` pings
  via `provider.keepalive_ping(&snapshot)` (`cache_keepalive.rs:330`, a
  distinct `Provider` trait method), and its snapshot is the **canonical**
  `agent.messages` prefix taken at the tool pause — not the compacted
  request-view — so wrapping `chat()` only is exactly right. Supervisor
  auto-summarize side-calls and `provider/*` internals likewise stay
  outside.
- `Provider` is held as `Arc<dyn Provider>` and is runtime-replaceable
  (`AgentLoop::replace_provider`, agent.rs:124).

## Design

### 1. New module `core/src/middleware.rs`

```rust
/// The canonical step request: everything the terminal provider call needs.
#[derive(Clone, Debug)]
pub struct StepRequest {
    /// Request-view messages (post prepare/sanitize/compaction). Not the
    /// canonical history — rewriting affects this request only.
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub model: String,
    pub workspace_root: PathBuf,
    pub turn_id: u64,
    pub step_index: u64,
    /// Informational-event handle. Middlewares MAY emit events, but MUST
    /// NOT emit `MessageRecorded` (projection is driver-owned; see §4).
    /// Enforcement is convention-only (raw sender) — see M9.
    pub event_tx: tokio::sync::mpsc::Sender<AgentEvent>,
}

/// What the chain hands back to the driver.
#[derive(Debug)]
pub enum StepReply {
    /// The provider was called (possibly after middleware-internal
    /// retries): here is its stream.
    Stream(tokio::sync::mpsc::Receiver<StreamChunk>),
    /// Skip the provider entirely; this text is the step's final answer.
    FinalText(String),
}

#[async_trait::async_trait]
pub trait AgentMiddleware: Send + Sync {
    fn name(&self) -> &str;
    async fn call(&self, req: StepRequest, next: Next<'_>)
        -> Result<StepReply, ProviderError>;
}

/// The rest of the chain plus the terminal. Zero-allocation: borrows the
/// chain slice; `run` consumes it.
pub struct Next<'a> { /* middlewares: &'a [Arc<dyn AgentMiddleware>],
                         provider: &'a Arc<dyn Provider> */ }
impl<'a> Next<'a> {
    pub async fn run(self, req: StepRequest) -> Result<StepReply, ProviderError>;
}

#[derive(Default)]
pub struct MiddlewareChain { /* Vec<Arc<dyn AgentMiddleware>> */ }
impl MiddlewareChain {
    pub fn new() -> Self;
    pub fn push(&mut self, middleware: Arc<dyn AgentMiddleware>);
    pub fn is_empty(&self) -> bool;
    /// Run `req` through the chain; index 0 is outermost. Empty chain =
    /// direct terminal call, observably identical to today's provider.chat
    /// (same arguments at the same point in time; `model`/`workspace_root`
    /// are materialized by value instead of by reference).
    pub async fn call(&self, provider: &Arc<dyn Provider>, req: StepRequest)
        -> Result<StepReply, ProviderError>;
}
```

Mechanics (tower-style, no `Rewrite` variant — rewriting is "call `next`
with a modified request"):

- `Next::run`: `split_first()`; head middleware called with the tail as its
  `Next`; empty slice = terminal: `provider.chat(&req.messages, &req.tools,
  &req.model, &req.workspace_root)` wrapped in `StepReply::Stream`.
- The chain never owns the provider — it is passed per call, so
  `replace_provider` keeps working and the chain stays provider-agnostic.
- Mechanics note for implementation: `Next::run` is a plain `async fn`
  (NOT `#[async_trait]` — it must escape the boxing machinery); the trait
  method's `Next<'_>` lifetime desugars correctly under async-trait
  (verified against async-trait 0.1.91: elided lifetimes become `'lifeN`
  with `'lifeN: 'async_trait` + `Send` bounds).
- The three roadmap capabilities map as:
  - **Proceed** = `next.run(req).await` unchanged;
  - **Rewrite** = `next.run(req_with_modified_messages).await`;
  - **ShortCircuit** = return `Ok(StepReply::FinalText(text))` without
    calling `next` (later middlewares and the provider are skipped).
- What this enables for P3 (first consumer): retry-same-request, prune or
  rewrite the **request-view**, short-circuit, and `ProviderError` access
  (`chat()`-level `Err` bubbles back through every layer, so middlewares
  can catch/retry it). **P3's re-compaction against canonical history is
  NOT reachable from `StepRequest` as specified here** — `messages` is the
  post-compaction view and `plan_context_view` needs `agent.messages` +
  `smart_compaction_mode`; a compaction handle (or driver-supplied closure)
  is a future extension. P4 does not promise it.
- **Mid-stream errors (`StreamChunk::Error`) surface after the chain has
  returned and are NOT interceptable in P4** — documented limitation;
  revisit only if P3 needs it (DeepSeek/OpenAI overflow arrives as HTTP
  400 → `chat()` `Err`).
- `StepRequest: Clone` powers retry arms; the dominant cost (messages
  clone) is already paid per step today (`request_messages` build), so no
  new asymptotic cost.

### 2. Wiring into `step()`

Replace `agent_driver.rs:325-334` with a chain call; match on the reply:

```rust
let reply = agent.middleware.call(&agent.provider, StepRequest { … }).await?;
let mut stream = match reply {
    StepReply::Stream(stream) => stream,
    StepReply::FinalText(text) => {
        // Empty short-circuit text is a middleware bug — fail loudly, do
        // NOT record an empty assistant message (empty-response policy
        // exists precisely because empty assistant messages confuse
        // providers and would replay back on resume).
        if text.trim().is_empty() {
            return Err(ProviderError::Other(
                "middleware short-circuited the step with empty final text".into(),
            ));
        }
        // Attachments are real inputs even though the provider never ran:
        // run the same cleanup the stream path does (agent_driver.rs
        // cleanup block), BEFORE recording, so history + disk stay
        // consistent (oracle P1-1).
        if !self.attachments_cleaned {
            crate::agent::cleanup_processed_attachments(
                &mut agent.messages, self.workspace_root, attachments);
            self.attachments_cleaned = true;
        }
        // Replay-safe short-circuit: identical bookkeeping to the normal
        // final-text path — record, push, emit, then FinalText outcome.
        let msg = Message::assistant(text.clone());
        agent.record(&msg).await;
        agent.messages.push(msg);
        agent.emit(AgentEvent::MessageReceived { role: "assistant".into(),
                   content: text.clone(), steering: false }).await;
        return Ok(StepOutcome::FinalText { text, had_tool_calls: false });
    }
};
```

The empty-response retry counter is intentionally NOT applied to
short-circuits: the provider never ran (a short-circuit is not a provider
empty).

`AgentLoop` gains `pub(crate) middleware: MiddlewareChain` (default empty
in `new()`) and a builder:

```rust
/// Append a step-request middleware. First added = outermost (wraps
/// everything added later and the provider call itself).
pub fn with_middleware(mut self, middleware: Arc<dyn AgentMiddleware>) -> Self;
```

(Chain-level `len()` deliberately omitted — nothing needs it; `is_empty()`
suffices.)

No `AgentLoop::new` signature change — supervisor.rs:567 and all test
fixtures are untouched.

### 3. What does NOT traverse the chain (explicit)

- `CacheKeepalive` pings — structurally outside: they call
  `keepalive_ping()`, not `chat()` (see §Current state).
- Supervisor auto-summarize side-calls (not step requests).
- `tool_pipeline` (keeps its own approval → hooks → execute phases;
  wrapping it would break approval semantics — roadmap P4 §2 says the
  same).
- `prepare_messages_for_request` + `sanitize_tool_call_pairs` stay before
  the chain (provider normalization, not policy).
- Subagent child sessions: their own `AgentLoop`s default to an empty
  chain until something wires them.

**Forward-compat note (for the first rewriting middleware author):** the
keepalive snapshot warms the *canonical* `agent.messages` prefix; a
middleware that rewrites `req.messages` sends a divergent next request
and silently forfeits the warmed cache. Harmless in P4 (empty chain);
worth a comment when P3 lands.

### 4. P2 projection invariants preserved

- Only the driver records model-visible messages (`agent.record`). A
  middleware MUST NOT emit `MessageRecorded`; the `event_tx` handle is for
  informational events (Checkpoint, ContextCompaction, Error, …) only.
  **Enforcement is convention-only** (it is a raw sender) — M9 exercises
  the emit surface; the projection stays driver-owned by discipline, the
  same way `record()` call sites are today.
- Short-circuit final text is recorded like any assistant message, so
  resume/replay folds it — no invisible turns.
- No new `AgentEvent` variants in P4 (no serde/replay surface change).

## Test matrix

| # | Test | Where |
|---|------|-------|
| M1 | empty chain → terminal called once; provider observes messages/tools/model/workspace_root untouched | unit, `middleware.rs` inline |
| M2 | three recorder middlewares execute in registration order (outer→in→terminal) | unit |
| M3 | rewrite: middle middleware modifies messages; inner middleware + provider see the rewrite; outer sees the original | unit |
| M4 | short-circuit: outer returns `FinalText` without `next`; inner + provider never run | unit |
| M5 | error path: provider `Err` reaches outer layers; retry-shaped middleware catches, rewrites, second attempt succeeds | unit |
| M6 | wiring pin: real `AgentLoop` + rewriting middleware; two-step turn (tool call → final) → middleware invoked once per step, provider sees rewritten messages both times. **Fails if `step()` bypasses the chain** (diff-verification lesson from Phase C) | integration, `core/tests/middleware.rs` |
| M7 | short-circuit end-to-end: `run_turn` returns middleware text; provider 0 calls; `agent.messages` = user + assistant; event stream has `MessageRecorded`(assistant) + `MessageReceived`(assistant) + `TurnCompleted` | integration |
| M7a | short-circuit with one `ImageAttachment`: temp file removed, no `Image` part survives in `agent.messages` (oracle P1-1) | integration |
| M7b | empty `FinalText("")` → turn fails loudly (`ProviderError::Other` naming the short-circuit), nothing recorded | integration |
| M8 | default (no middleware added): a plain `run_turn` turn still succeeds — guards against default-chain misconstruction | integration |
| M9 | middleware CAN emit an informational event via `req.event_tx` (e.g. `Checkpoint`) — it lands in the event stream; exercises the `event_tx` surface so it does not ship untested | integration |

Existing suites (turn_driver, phase_c, agent inline) must stay green
untouched — they are the empty-chain regression proof.

## Rollout order

Single lane (core only; no runtime/cli callers change):

1. `middleware.rs` with inline unit tests M1–M5.
2. `agent_driver.rs` rewiring + `AgentLoop` field/builder (§2).
3. Integration tests M6–M8 (cross-model tester lane, spec = this doc).
4. Ponytail pass over the full diff; fmt/clippy/test workspace; commit
   (feat(core): waterfall middleware layer around step requests).
5. Doc sync: `docs/architecture.md` core module map + mark P4 implemented
   in `deepseek-harness-adoption.md` (same commit as 4 per doc-sync rule).

## Non-goals

- Migrating inline compaction into a `CompactionMiddleware` — that is P3's
  mounting step, with its own event-bracket semantics to preserve.
- Retry / context-overflow middleware (P3, first consumer).
- Mid-stream (`StreamChunk::Error`) interception.
- Config surface for chain composition (programmatic only for now).
- New `AgentEvent` variants or replay changes.
- Wrapping `tool_pipeline` or keepalive calls (§3).

## Oracle review record (2026-08-23)

Verdicts: §1 sound · §2 needs change (fixed) · §3 needs change (wording,
fixed) · §4 sound. **No P0 findings.**

Findings applied to this doc:

- **P1-1** short-circuit path returned before `cleanup_processed_attachments`
  → attachments + `Image` paths leak on short-circuited turns. Fixed in §2
  (cleanup runs first, mirroring stream-path order); M7a added.
- **P1-2** §1 over-promised the P3 seam ("escalate compaction" is not
  reachable — `StepRequest.messages` is the post-compaction view). Claim
  narrowed + explicit future-extension note; §1 updated.
- **P2-1** keepalive uses `keepalive_ping()` (distinct trait method,
  `cache_keepalive.rs:330`), not `chat()` — bypass is structural, not a
  rule to remember. §Current state + §3 corrected; real reason (canonical
  prefix vs request-view) stated.
- **P2-2** forward-compat note: a rewriting middleware silently forfeits
  keepalive warmth. §3 note added.
- **P2-3** `event_tx` kept (P3 is the declared first consumer) but must not
  ship unexercised → M9 added; convention-only enforcement stated in §4.
- **P2-4** empty `FinalText` semantics were implied → decided: fail loudly
  (`ProviderError::Other`), nothing recorded. §2 + M7b.
- **P2-5** `len()` YAGNI → dropped.
- **P2-6** line refs corrected (`replace_provider` agent.rs:124; call site
  `agent_driver.rs:328-336`); "byte-identical" → "observably identical"
  (model/workspace_root materialize by value).
- Confirmed sound: async-trait 0.1.91 lifetime mechanics for `Next<'_>`
  (elided → `'lifeN: 'async_trait` + `Send`); `Next::run` must be a plain
  `async fn`; `StepRequest: Clone` cost already paid per step today;
  `StepReply::Stream` Receiver moved (never cloned) through layers;
  empty-chain argument ordering identical; M6 is a genuine wiring pin.

## Implementation record (2026-08-23)

- Lanes: fixer (`middleware.rs` + wiring + M1–M5, `afd9969`), tester
  (integration M6–M9, `9e83402`), ponytail pass (lean; one Arc-path nit
  applied, one test-code shrink noted and skipped).
- **Deviation from §1 (accepted):** `Next` derives `Clone` (two borrows,
  zero cost). The spec's "`run` consumes it" made catch-and-retry
  middlewares impossible — `next.run()` consumes the only handle to the
  rest of the chain. Clone is the minimal fix and matches the design's
  own retry intent; documented on the type.
- **M7b clarification (from tester lane):** after a failed short-circuit,
  `run_turn`'s P1 rollback truncates to the pre-turn baseline — a fresh
  agent ends with **empty** messages (not `[user]`). The design's
  "nothing recorded" wording is what holds; the rollback is pre-existing
  P1 policy, unchanged by P4.
- Validation: `cargo fmt --all -- --check`, `cargo clippy --workspace
  -- -D warnings`, `cargo test --workspace` — all green at merge.
