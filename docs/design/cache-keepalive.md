# Design: Provider Prompt-Cache Keepalive

> Status: **Design** · Owner: core · Depends-on: `crates/core/src/agent.rs`, `provider/`
>
> References:
> - Maxim Khailo, *"Your Agentic Workflow's Cache Keepalive Costs 8x Too Much (v2: the interval frontier)"* — measured retention curves and keepalive economics across Anthropic, OpenAI, Gemini, DeepSeek.
> - DeepSeek API Docs — *Context Caching*, *Models & Pricing* (V4).

---

## 1. Problem

Provider prompt caches are the best deal in the API: a cached prefix is billed at
~1/50th–1/120th of the input price and skips most of the prefill latency. But the
cache expires in **minutes**, and an agentic loop is the worst possible user of it.

An agent fires a request, then executes tools or waits on an approval. The pause
often outlives the cache TTL. The follow-up request pays full price and full latency
on a prefix it just processed seconds ago. At agent scale, that's a real line item.

### The pause points in nca

The agent loop (`crates/core/src/agent.rs`, `run_turn_inner`) sends a request,
streams the model output, then enters the tool pipeline:

```
provider.chat()  ──►  stream model output  ──►  tool_pipeline::run_tool_pipeline().await
                                                        │
                                                        │  ← PAUSE: tool execution + approvals
                                                        │     provider & messages are idle here
                                                        ▼
                                               next provider.chat()
```

During `run_tool_pipeline(...).await`, neither `self.provider` nor `self.messages`
is touched. At that moment `self.messages` ends with the assistant's `tool_calls`
message — i.e. **exactly the reusable prefix** the next request will re-send.

Typical pause durations in this codebase:

| Pause source | Typical duration |
|---|---|
| `read_file` / `search_code` / `ast_grep` | < 1 s (below TTL — keepalive buys nothing) |
| `run_validation` (cargo build / test suite) | 1–30 min (above TTL — cache dies) |
| Human approval (`ApprovalPolicy::Ask` → IPC wait) | unbounded (minutes to hours) |

---

## 2. The Economics (measured, not guessed)

### 2.1 Cache retention — where each provider's cache dies

Measured idle survival (no keepalive), from the article:

| Provider | Documented TTL | Measured eviction cliff | Grace period |
|---|---|---|---|
| **Anthropic** | 5 min | 5–6 min | **none** (hard cliff) |
| **DeepSeek V3.2** | — | ~9–10 min | marginal |
| **OpenAI** | 5–10 min | ~30 min (slow eviction) | generous, but not plannable |
| **Gemini** | — | never converges (routing lottery) | unreliable |

**Rule:** assume warmth only up to the shortest documented TTL minus a margin.
Nothing evicted before its TTL; nothing past it is guaranteed.

### 2.2 V4 pricing — why DeepSeek is now the best keepalive target

The article measured DeepSeek **V3.2**, where cache miss was so cheap ($0.022/100k)
that pings cost more than the eviction they prevent — "cost negative, latency play only."

**DeepSeek V4 completely reverses this.** V4 pricing (per 1M tokens, official):

| Model | cache hit | cache miss | hit/miss ratio | output |
|---|---|---|---|---|
| deepseek-v4-flash | $0.0028 | $0.14 | **50×** | $0.28 |
| deepseek-v4-pro | $0.003625 | $0.435 | **120×** | $0.87 |

At 100k prefix (V4-pro):
- one cache hit (ping): 100k × $0.003625/1M = **$0.00036**
- one cache miss (cold re-prefill): 100k × $0.435/1M = **$0.0435**

The break-even horizon is `τ(w/r − 1)`:

| Provider | w/r | τ* | break-even horizon |
|---|---|---|---|
| Anthropic | 12.5× | 4 min | ~46 min |
| OpenAI | ~10× | 8 min | ~72 min |
| DeepSeek V4-pro | **120×** | 8 min | **~16 h** |
| DeepSeek V4-flash | **50×** | 8 min | **~6.5 h** |

A 120× ratio means pings are nearly free. Keepalive on V4-pro pays for itself
after ~120 pings — over 16 hours of pause at 8-minute intervals. In practice,
**any bounded agent pause benefits from keepalive on V4.**

#### Per-pause economics (V4-pro, 100k prefix)

| Pause | pings (τ=8min) | ping cost | vs. let-it-die | verdict |
|---|---|---|---|---|
| 10 min (just past TTL) | 1 | $0.00036 | $0.0435 | **120× ahead** |
| 30 min (cargo build) | 3 | $0.0011 | $0.0435 | **40× ahead** |
| 60 min (test suite) | 7 | $0.0025 | $0.0435 | **17× ahead** |
| 16 h | 120 | $0.0432 | $0.0435 | wash (break-even line) |
| > 16 h | 120+ | > $0.0435 | $0.0435 | **stop — let it die** |

### 2.3 The toxic edge

The keepalive interval τ must stay **below** the provider's retention. Past it,
every ping lands on a dead cache and re-prefills at full price:

> Anthropic at 8-minute pings (τ > 5-min TTL): 3 pings, 3 full prefills,
> $1.334 — **4× the cost of never pinging at all.**

An interval past the TTL isn't wasteful — it's actively toxic. This must be a
hard guardrail, not a configuration suggestion.

### 2.4 The 30-second convention

The community default (ping every 30 s) has a break-even of ~6 minutes. It loses
money at a 10-minute pause on every provider. The article measured it costs
**7.8× more** than a 4-minute interval for the same warmth. The convention isn't
cautious — it's expensive.

---

## 3. Mechanism Overview

A background `CacheKeepalive` task runs during **bounded, agent-driven pauses**
(tool execution, approval waits). It re-sends the exact conversation prefix on a
per-provider timer to refresh the cache TTL. When the pause ends, the task is
cancelled immediately.

**Key insight — the first ping is scheduled after τ\***, not immediately:

- Most tool pauses (`read_file`, `search_code`, `ast_grep`) finish in < 1 s — well
  before the first ping fires. The task is cancelled → **zero pings, zero cost.**
- This "for free" implements the article's rule *"below the eviction point, do not
  bother"* — the timer naturally filters short pauses without predicting their length.

The keepalive is **never** active during inter-turn user-idle time (the user may be
away for hours). The article: *"never keep a dead session warm."*

### What gets pinged

The snapshot taken at pause entry — identical to what the next real request will send:

```
KeepaliveSnapshot {
    messages: Vec<Message>,     // system + history + assistant(tool_calls)
    tools:    Vec<ToolDefinition>,
    model:    String,
    workspace_root: PathBuf,
}
```

The ping sends this prefix with `max_tokens: 1`, drains the stream, discards all
output (text, tool calls, reasoning), and records only `usage`. The ping response
is **never** written to `self.messages`.

---

## 4. Provider Profiles

Per-provider parameters encoded as configuration. The economic table from §2
hard-coded as defaults, overridable via config.

```rust
pub struct KeepaliveProfile {
    /// Master switch. Forced false when interval >= retention (toxic-edge guard).
    pub enabled: bool,
    /// Ping interval τ* = measured retention − margin.
    pub interval: Duration,
    /// Break-even horizon. Past this, stop pinging and let the cache die.
    pub max_pause: Duration,
    /// Minimum prefix size (tokens) to bother keeping warm.
    pub min_prefix_tokens: usize,
    /// What the keepalive buys on this provider.
    pub benefit: KeepaliveBenefit,
}

pub enum KeepaliveBenefit {
    /// Saves real money (high w/r ratio).
    Cost,
    /// Saves only latency (low w/r, but TTFT improvement is valuable).
    LatencyOnly,
    /// No benefit — keepalive disabled.
    None,
}
```

Default table:

```rust
fn default_profile(kind: ProviderKind) -> KeepaliveProfile {
    use ProviderKind::*;
    match kind {
        // 120× hit/miss ratio → pings nearly free, break-even ~16h.
        // DeepSeek is the PRIMARY provider; this is the highest-value profile.
        DeepSeek  => KeepaliveProfile {
            enabled: true,
            interval: Duration::from_secs(480),  // 8 min (TTL ~10 min, 2-min margin)
            max_pause: Duration::from_secs(960 * 60), // 16 h
            min_prefix_tokens: 4000,
            benefit: KeepaliveBenefit::Cost,
        },

        // 12.5× ratio, hard 5-min cliff. Tightest margin — most fragile.
        Anthropic => KeepaliveProfile {
            enabled: true,
            interval: Duration::from_secs(240),  // 4 min (TTL 5 min, 1-min margin)
            max_pause: Duration::from_secs(46 * 60), // 46 min
            min_prefix_tokens: 4000,
            benefit: KeepaliveBenefit::Cost,
        },

        // ~10× ratio, slow eviction (~30 min). Generous interval.
        OpenAi    => KeepaliveProfile {
            enabled: true,
            interval: Duration::from_secs(480),  // 8 min
            max_pause: Duration::from_secs(72 * 60), // ~72 min
            min_prefix_tokens: 4000,
            benefit: KeepaliveBenefit::Cost,
        },

        // Unreliable retention (routing lottery) — don't bother.
        MiniMax | OpenRouter | ZhipuAI | Kimi | Custom => KeepaliveProfile::disabled(),
    }
}
```

### Toxic-edge enforcement

At profile resolution:

```rust
if profile.interval >= retention_estimate(kind) {
    profile.enabled = false;
    tracing::warn!(
        provider = ?kind,
        interval_secs = profile.interval.as_secs(),
        "keepalive disabled: interval >= retention (toxic edge)"
    );
}
```

A misconfigured interval past the TTL is **silently disabled**, never allowed to
burn money on dead-cache re-prefills.

---

## 5. Integration Point

### 5.1 AgentLoop (`crates/core/src/agent.rs`)

In `run_turn_inner`, wrap the tool pipeline call:

```rust
// At this point self.messages = system + history + assistant(tool_calls)
// — exactly the prefix the next request will re-send.
let snapshot = KeepaliveSnapshot {
    messages: self.messages.clone(),
    tools: self.tool_definitions(),
    model: self.model.clone(),
    workspace_root: workspace_root.to_path_buf(),
};

let keepalive = CacheKeepalive::start(
    Arc::clone(&self.provider),    // §6: Box → Arc<dyn Provider>
    snapshot,
    self.keepalive_profile,        // resolved per active provider kind
    self.event_tx.clone(),
);

let pipeline = tool_pipeline::run_tool_pipeline(
    &self.tools,
    &mut self.approval,
    &self.hooks,
    &self.event_tx,
    &self.cancel_flag,
    tool_calls.clone(),
)
.await
.map_err(ProviderError::Other)?;

keepalive.stop().await;  // cancel task + emit summary
```

The keepalive borrows neither `tools` nor `approval` nor `hooks` — all of which are
already borrowed by the pipeline. It only needs `provider` (idle during the pause)
and the snapshot (owned clone). No borrow-conflict.

### 5.2 What does NOT get a keepalive

- **Inter-turn idle** (waiting for user input between `run_turn` calls).
- **The request itself** — keepalive starts only when the model finishes streaming
  and the pipeline begins, and stops before the next `provider.chat()`.
- **Sessions in Plan mode** (read-only, no tool execution pauses of meaningful length).

---

## 6. Provider Trait Change

### 6.1 `Box<dyn Provider>` → `Arc<dyn Provider>`

`Provider` is already `Send + Sync`. The keepalive task needs its own reference to
send pings independently while the main loop is `.await`-ing the pipeline. `Box`
can't be cloned; `Arc` can.

Changes:
- `AgentLoop.provider: Box<dyn Provider>` → `Arc<dyn Provider>`
- `AgentLoop::new(...)` takes `Arc<dyn Provider>`
- `AgentLoop::replace_provider` takes `Arc<dyn Provider>`
- `factory::build_provider` / `build_provider_for` return `Arc<dyn Provider>`
- All call sites in `crates/runtime/` that construct or hold the provider adapt.

### 6.2 Keepalive ping method

```rust
#[async_trait]
pub trait Provider: Send + Sync {
    // ... existing methods ...

    /// Send the snapshot prefix with max_tokens=1 to refresh the provider's
    /// prompt cache. Returns observed usage (for cost tracking).
    ///
    /// Default implementation: send via chat(), drain the stream, discard output.
    /// Providers may override for optimization (e.g. non-streaming ping).
    async fn keepalive_ping(
        &self,
        snapshot: &KeepaliveSnapshot,
    ) -> Result<PingUsage, ProviderError> {
        // Default: build a minimal request, drain stream, extract usage only.
    }
}

pub struct PingUsage {
    pub input_tokens: u64,
    pub cache_read_tokens: u64,   // should be > 0 if cache was warm
    pub cache_miss_tokens: u64,
    pub estimated_cost_usd: f64,
}
```

**OpenAiCompatProvider** (serves DeepSeek, OpenAI, OpenRouter, ZhipuAI):
- Reuses `openai_request_body` with forced `max_tokens: 1`.
- Drains the SSE stream, extracts `usage` (`prompt_cache_hit_tokens` /
  `prompt_cache_miss_tokens` for DeepSeek; `prompt_tokens_details.cached_tokens`
  for OpenAI).
- Discards all text/tool/reasoning deltas.

**AnthropicProvider:**
- Sends the prefix with `cache_control: { type: "ephemeral" }` on the stable prefix
  (system prompt + tools schema).
- `max_tokens: 1`, drain, extract usage.

---

## 7. Keepalive Task Algorithm

```rust
pub struct CacheKeepalive {
    provider: Arc<dyn Provider>,
    snapshot: KeepaliveSnapshot,
    profile: KeepaliveProfile,
    event_tx: tokio::sync::mpsc::Sender<AgentEvent>,
    stop_notify: Arc<tokio::sync::Notify>,
    handle: tokio::task::JoinHandle<()>,
}

impl CacheKeepalive {
    pub fn start(/* ... */) -> Self {
        // Guard 0: disabled or benefit == None → no-op (return a no-op handle)
        if !profile.enabled || matches!(profile.benefit, KeepaliveBenefit::None) {
            return Self::noop();
        }
        // Guard 1: prefix too small to be worth pinging
        if snapshot.est_tokens() < profile.min_prefix_tokens {
            return Self::noop();
        }
        // Guard 2: toxic edge already enforced at profile resolution (§4)

        let stop_notify = Arc::new(Notify::new());
        let handle = tokio::spawn(/* run loop */);
        Self { /* ... */ }
    }

    pub async fn stop(self) {
        self.stop_notify.notify_waiters();
        let _ = self.handle.await;
        // Summary event emitted inside the task before exit
    }
}
```

Task run loop:

```rust
async fn run(self) {
    let start = Instant::now();
    let mut pings = 0u32;
    let mut total_cost = 0.0f64;

    self.emit(CacheKeepaliveStarted {
        interval_secs: profile.interval.as_secs(),
        benefit: profile.benefit,
    });

    loop {
        // Wait for either the next interval or a stop signal.
        tokio::select! {
            _ = self.stop_notify.notified() => {
                self.emit(CacheKeepaliveStopped {
                    reason: StopReason::PauseEnded,
                    pings,
                    total_cost_usd: total_cost,
                });
                return;
            }
            _ = sleep(profile.interval) => {}
        }

        // Guard 3: break-even horizon — stop paying premiums, let cache die.
        if start.elapsed() >= profile.max_pause {
            self.emit(CacheKeepaliveStopped {
                reason: StopReason::BreakEvenExceeded,
                pings,
                total_cost_usd: total_cost,
            });
            return;
        }

        // Send ping. Failures are logged + backoff, never propagated to main loop.
        match self.provider.keepalive_ping(&self.snapshot).await {
            Ok(usage) => {
                pings += 1;
                total_cost += usage.estimated_cost_usd;
                self.emit(CacheKeepalivePing {
                    ping_cost_usd: usage.estimated_cost_usd,
                    cumulative_cost_usd: total_cost,
                    cache_was_warm: usage.cache_read_tokens > 0,
                    elapsed_secs: start.elapsed().as_secs(),
                });
            }
            Err(e) => {
                tracing::warn!(error = %e, "keepalive_ping_failed; backing off");
                // Exponential backoff on failure — don't hammer a failing endpoint.
                // Continue loop; next ping after interval × backoff_multiplier.
            }
        }
    }
}
```

### Guardrails summary

| # | Guard | Where | Effect |
|---|---|---|---|
| 0 | `enabled` / `benefit == None` | `start()` | No task spawned at all |
| 1 | `prefix < min_prefix_tokens` | `start()` | No task spawned |
| 2 | `interval >= retention` (toxic edge) | profile resolution | Profile forced disabled |
| 3 | `elapsed >= max_pause` (break-even) | run loop | Task self-terminates |
| 4 | short pause < τ* | timer | First ping never fires, task cancelled |
| 5 | ping failure | run loop | Logged + backoff, never propagates to main loop |

---

## 8. Observability

### 8.1 Cost tracking (separate bucket)

`CostTracker` already tracks `cache_read_tokens` / `cache_creation_tokens` from
normal request usage. Keepalive ping costs are tracked in a **separate bucket**
so users can see exactly how much the keepalive itself costs:

```rust
// In CostTracker (crates/core/src/cost.rs)
pub struct CostTracker {
    // ... existing fields ...
    pub keepalive_cost_usd: f64,       // NEW: isolated keepalive spend
    pub keepalive_pings: u64,          // NEW: ping count
}
```

### 8.2 Cache hit ratio

```rust
impl CostTracker {
    /// Fraction of input tokens served from cache across the session.
    pub fn cache_hit_ratio(&self) -> f64 {
        let total = self.input_tokens + self.cache_read_tokens;
        if total == 0 { return 0.0; }
        self.cache_read_tokens as f64 / total as f64
    }
}
```

**Mapping note:** DeepSeek reports `prompt_cache_hit_tokens` (hit) and
`prompt_cache_miss_tokens` (miss). The current code (`openai_compat.rs:180-195`)
maps these to `cache_creation_tokens` (miss) / `cache_read_tokens` (hit), and sets
`input_tokens` = `prompt_tokens` (which includes both). This needs verification so
that `cache_hit_ratio()` is accurate.

### 8.3 Events

Three new `AgentEvent` variants:

```rust
CacheKeepaliveStarted {
    interval_secs: u64,
    benefit: KeepaliveBenefit,
},
CacheKeepalivePing {
    ping_cost_usd: f64,
    cumulative_cost_usd: f64,
    cache_was_warm: bool,   // did this ping actually hit?
    elapsed_secs: u64,
},
CacheKeepaliveStopped {
    reason: StopReason,     // PauseEnded | BreakEvenExceeded | Disabled
    pings: u32,
    total_cost_usd: f64,
},
```

TUI can show: `keeping cache warm (deepseek, 8m) · $0.0011 spent · 3 pings`.

### 8.4 Adaptive τ* (future enhancement)

Since we already observe `cache_read_tokens` per request, we can detect eviction:
if a real request shows `cache_read_tokens` dropping sharply, the cache was evicted
despite keepalive → tighten τ*. If hits are stable, the interval is safe. This is
data-driven self-tuning; deferred until we have production telemetry.

---

## 9. Complementary: Prefix Stability (the bigger lever)

Keepalive only preserves cache warmth across pauses. But cache **hits** require the
prefix to be byte-identical to a previously cached unit. Any per-turn variation
busts the cache key — making keepalive pointless (it's keeping a prefix warm that
would never hit anyway).

DeepSeek's disk-cache rules (official docs) require **exact full-prefix match** to
a persisted cache unit. This makes prefix stability critical across **all** turns,
not just during pauses.

### 9.1 Anthropic: add cache breakpoints (currently missing)

`crates/core/src/provider/anthropic.rs` has **zero** `cache_control` markers. Even
with perfect keepalive, Anthropic won't cache the prefix because it's never marked.
Adding `cache_control: { type: "ephemeral" }` on the system prompt + tools schema
is a **larger and more immediate win** than keepalive — it makes caching possible
at all.

### 9.2 Prefix stability audit

The system prompt and tools schema must be byte-identical across turns. Risk areas:

| Risk | Location | Impact |
|---|---|---|
| Dynamic content in system prompt (timestamps, counters) | `harness.rs` `build_system_prompt` | Busts every turn |
| Unstable tool-schema serialization order | `agent.rs:tool_definitions()` → `openai_compat.rs` | Busts if HashMap iteration order varies across restarts |
| Per-turn parameter drift (temperature, max_tokens) | `openai_request_body` | Busts if dynamically adjusted |
| Smart compaction changing history | `agent.rs:plan_context_view` | Compacted view no longer matches cached prefix unit |

The codebase already has awareness of this — `harness.rs` caps the skills index:
*"Cap the skills index so it can't bloat the **cache-stable** system-prompt prefix."*

### 9.3 Prefix hash diagnostic

Add a `cache_prefix_hash` — hash of `[system_prompt, tools_json, first_N_messages]`
— emitted alongside `CostUpdated`. Lets users see whether a turn hit or missed, and
diagnose cache-busting changes.

---

## 10. Implementation Phases

### Phase 1: Foundation (largest immediate win, independent of keepalive)

- **P1.1** Anthropic `cache_control` breakpoints on system prompt + tools schema.
- **P1.2** Prefix stability audit + `cache_prefix_hash` diagnostic.
- **P1.3** `cache_hit_ratio()` in `CostTracker` + verify DeepSeek usage mapping.

These make caching **possible and measurable**. No keepalive yet.

### Phase 2: Keepalive core

- **P2.1** `AgentLoop.provider: Box → Arc<dyn Provider>` + all call-site adaptation.
- **P2.2** `CacheKeepalive` task (`crates/core/src/cache_keepalive.rs`).
- **P2.3** `Provider::keepalive_ping()` default impl + OpenAiCompat override.
- **P2.4** Default provider profiles (§4).
- **P2.5** Events + isolated cost bucket.
- **P2.6** Integration in `agent.rs` `run_turn_inner`.

### Phase 3: Polish

- **P3.1** Anthropic `keepalive_ping()` override with `cache_control`.
- **P3.2** Config surface (`cache_keepalive: "auto" | "off" | "latency"`).
- **P3.3** Adaptive τ* tuning from observed hit rates.
- **P3.4** TUI display of keepalive state.

---

## 11. Risks and Trade-offs

| Risk | Mitigation |
|---|---|
| **Prefix mismatch** between ping snapshot and next real request (compaction, attachment cleanup) | Keepalive pings the exact snapshot; if next request takes a different path (compaction), that turn wouldn't have hit anyway. Acceptable. |
| **Ping failure pollutes main loop** | Ping errors are logged + backoff, never propagated. The main loop is unaffected. |
| **Concurrency on shared provider** | Provider is idle during the pause (main loop is `.await`-ing the pipeline). `Arc` shared, no write contention. |
| **Toxic edge misconfiguration** | Hard guardrail at profile resolution: `interval >= retention → disabled`. |
| **Commons externality** (article's belief: mass keepalive adoption may force providers to meter residency per token-hour) | This is a long-term policy risk, not an implementation correctness issue. The arbitrage exists today. |
| **DeepSeek TTL uncertainty** (~10 min, best-effort) | τ* = 8 min gives 2-min margin. Adaptive tuning (P3.3) can tighten from observed data. |

---

## 12. What this design does NOT do

- **Does not ping during user-idle time.** Sessions waiting for human input are left
  to go cold. Re-warming on the next real request is cheaper than indefinite pinging.
- **Does not predict pause duration.** The first-ping-after-τ* design naturally
  filters short pauses without prediction.
- **Does not cache tool results separately.** The prefix is the full conversation;
  the provider handles prefix matching internally.
- **Does not support Gemini-like unreliable providers.** If retention isn't a
  measurable curve, keepalive can't be tuned, so it stays disabled.
