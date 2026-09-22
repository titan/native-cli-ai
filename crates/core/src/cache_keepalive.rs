//! Prompt-cache keepalive: re-send the conversation prefix during tool-execution
//! pauses to keep the provider's prompt cache warm.
//!
//! See `docs/design/cache-keepalive.md` for the full design and economic analysis.
//!
//! ## How it works
//!
//! When the agent enters a tool-execution pause (the gap between the model's
//! last response and the next request), a background task re-sends the exact
//! conversation prefix on a per-provider timer. Each ping refreshes the
//! provider's cache TTL. When the pause ends, the task is cancelled.
//!
//! The first ping fires **after** the provider's interval τ* (e.g. 8 min for
//! DeepSeek). Most tool pauses finish before that → zero pings, zero cost.
//!
//! ## Guardrails
//!
//! 1. **Prefix too small** — below `min_prefix_tokens`, no task is spawned.
//! 2. **Toxic edge** — if `interval >= retention`, the profile is disabled at
//!    resolution time. An interval past the TTL re-prefills a dead cache at
//!    full price on every ping.
//! 3. **Break-even horizon** — past `max_pause`, the task self-terminates.
//!    Pinging longer costs more than letting the cache die and paying one
//!    re-prefill.
//! 4. **Short pause** — the first ping is τ* away. If the pause ends before
//!    then, the task is cancelled with zero pings.
//! 5. **Ping failure** — logged + skipped, never propagated to the main loop.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use nca_common::config::ProviderKind;
use nca_common::event::AgentEvent;
use nca_common::message::Message;
use nca_common::tool::ToolDefinition;
use tokio::sync::Notify;
use tokio::task::JoinHandle;

use crate::provider::Provider;

// ---------------------------------------------------------------------------
// Snapshot — the prefix to keep warm
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of the conversation prefix, taken at the start of
/// a tool-execution pause. This is exactly what the next real request will
/// re-send, so keeping it warm maximises the cache hit.
#[derive(Clone)]
pub struct KeepaliveSnapshot {
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub model: String,
    pub workspace_root: PathBuf,
}

impl KeepaliveSnapshot {
    /// Rough token estimate (~4 chars/token) to gate against pinging tiny prefixes.
    pub fn est_tokens(&self) -> usize {
        let msg_chars: usize = self.messages.iter().map(|m| m.content.approx_chars()).sum();
        let tool_chars: usize = self
            .tools
            .iter()
            .map(|t| t.name.len() + t.description.len() + t.parameters.to_string().len())
            .sum();
        (msg_chars + tool_chars) / 4
    }
}

/// Usage observed from a single keepalive ping.
#[derive(Debug, Clone, Default)]
pub struct PingUsage {
    pub input_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_creation_tokens: u64,
}

// ---------------------------------------------------------------------------
// Profile — per-provider economics
// ---------------------------------------------------------------------------

/// What the keepalive buys on a given provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeepaliveBenefit {
    /// Saves real money (high cache-hit / cache-miss price ratio).
    Cost,
    /// Saves only latency (TTFT improvement).
    LatencyOnly,
    /// No benefit — keepalive disabled.
    None,
}

/// Per-provider keepalive parameters derived from measured cache retention
/// and pricing economics.
///
/// Defaults come from the measured data in `docs/design/cache-keepalive.md`.
#[derive(Clone)]
pub struct KeepaliveProfile {
    pub enabled: bool,
    /// Ping interval τ* = measured retention − margin.
    pub interval: Duration,
    /// Break-even horizon. Past this, stop pinging and let the cache die.
    pub max_pause: Duration,
    /// Minimum prefix size (est. tokens) to bother keeping warm.
    pub min_prefix_tokens: usize,
    pub benefit: KeepaliveBenefit,
}

impl Default for KeepaliveProfile {
    fn default() -> Self {
        Self {
            enabled: false,
            interval: Duration::from_secs(480),
            max_pause: Duration::from_secs(3600),
            min_prefix_tokens: 4000,
            benefit: KeepaliveBenefit::None,
        }
    }
}

impl KeepaliveProfile {
    pub fn disabled() -> Self {
        Self::default()
    }
}

/// Default keepalive profile for a provider kind, based on measured cache
/// retention curves and V4 pricing.
///
/// **DeepSeek V4**: 120× hit/miss ratio (pro) → pings nearly free, break-even
/// ~16 h. τ* = 8 min (TTL ~10 min, 2-min margin). This is the highest-value
/// profile since DeepSeek is the primary provider.
///
/// **Anthropic**: 12.5× ratio, hard 5-min cliff. τ* = 4 min (tightest margin).
/// Break-even ~46 min.
///
/// **OpenAI**: ~10× ratio, slow eviction (~30 min). τ* = 8 min.
///
/// Other providers: disabled (unreliable retention or unknown economics).
pub fn default_profile(kind: ProviderKind) -> KeepaliveProfile {
    match kind {
        ProviderKind::DeepSeek => KeepaliveProfile {
            enabled: true,
            interval: Duration::from_secs(480), // 8 min (TTL ~10 min)
            max_pause: Duration::from_secs(960 * 60), // 16 h break-even
            min_prefix_tokens: 4000,
            benefit: KeepaliveBenefit::Cost,
        },
        ProviderKind::Anthropic => KeepaliveProfile {
            enabled: true,
            interval: Duration::from_secs(240), // 4 min (TTL 5 min, 1-min margin)
            max_pause: Duration::from_secs(46 * 60), // 46 min break-even
            min_prefix_tokens: 4000,
            benefit: KeepaliveBenefit::Cost,
        },
        ProviderKind::OpenAi => KeepaliveProfile {
            enabled: true,
            interval: Duration::from_secs(480),      // 8 min
            max_pause: Duration::from_secs(72 * 60), // ~72 min break-even
            min_prefix_tokens: 4000,
            benefit: KeepaliveBenefit::Cost,
        },
        ProviderKind::MiniMax
        | ProviderKind::OpenRouter
        | ProviderKind::ZhipuAI
        | ProviderKind::Kimi
        | ProviderKind::Mimo
        | ProviderKind::Custom => KeepaliveProfile::disabled(),
    }
}

/// Estimated cache retention for toxic-edge enforcement.
///
/// If `interval >= retention`, the profile must be disabled: every ping would
/// land on a dead cache and re-prefill at full price.
fn retention_estimate(kind: ProviderKind) -> Duration {
    match kind {
        ProviderKind::DeepSeek => Duration::from_secs(600), // ~10 min
        ProviderKind::Anthropic => Duration::from_secs(300), // 5 min
        ProviderKind::OpenAi => Duration::from_secs(600),   // 10 min (conservative)
        _ => Duration::ZERO,
    }
}

/// Resolve a profile for a provider kind, enforcing the toxic-edge guard.
pub fn resolve_profile(kind: ProviderKind) -> KeepaliveProfile {
    let mut profile = default_profile(kind);
    let retention = retention_estimate(kind);
    if retention > Duration::ZERO && profile.interval >= retention {
        tracing::warn!(
            provider = ?kind,
            interval_secs = profile.interval.as_secs(),
            retention_secs = retention.as_secs(),
            "keepalive disabled: interval >= retention (toxic edge)"
        );
        profile.enabled = false;
    }
    profile
}

// ---------------------------------------------------------------------------
// CacheKeepalive — the background task
// ---------------------------------------------------------------------------

/// Why the keepalive task stopped.
#[derive(Debug, Clone, Copy)]
enum StopReason {
    /// The tool-execution pause ended normally.
    PauseEnded,
    /// The break-even horizon was exceeded.
    BreakEvenExceeded,
}

impl StopReason {
    fn as_str(&self) -> &'static str {
        match self {
            StopReason::PauseEnded => "pause_ended",
            StopReason::BreakEvenExceeded => "break_even_exceeded",
        }
    }
}

/// Handle to a running keepalive task. Drop or [`stop`](Self::stop) it to
/// cancel the pings.
pub struct CacheKeepalive {
    stop_notify: Arc<Notify>,
    handle: Option<JoinHandle<()>>,
}

impl CacheKeepalive {
    /// A no-op keepalive (disabled or guarded out). [`stop`](Self::stop) is
    /// instant.
    fn noop() -> Self {
        Self {
            stop_notify: Arc::new(Notify::new()),
            handle: None,
        }
    }

    /// Start a keepalive task for the given snapshot and profile.
    ///
    /// Returns immediately. The first ping fires after `profile.interval`.
    /// If the profile is disabled or the prefix is too small, returns a no-op.
    pub fn start(
        provider: Arc<dyn Provider>,
        snapshot: KeepaliveSnapshot,
        profile: KeepaliveProfile,
        event_tx: tokio::sync::mpsc::Sender<AgentEvent>,
    ) -> Self {
        // Guard 0: disabled or no benefit
        if !profile.enabled || matches!(profile.benefit, KeepaliveBenefit::None) {
            return Self::noop();
        }
        // Guard 1: prefix too small to be worth pinging
        if snapshot.est_tokens() < profile.min_prefix_tokens {
            tracing::debug!(
                est_tokens = snapshot.est_tokens(),
                min = profile.min_prefix_tokens,
                "keepalive skipped: prefix below minimum"
            );
            return Self::noop();
        }

        let stop_notify = Arc::new(Notify::new());
        let task_notify = stop_notify.clone();

        let handle = tokio::spawn(async move {
            run_keepalive_loop(provider, snapshot, profile, event_tx, task_notify).await;
        });

        Self {
            stop_notify,
            handle: Some(handle),
        }
    }

    /// Signal the task to stop and wait for it to finish.
    pub async fn stop(self) {
        self.stop_notify.notify_one();
        if let Some(handle) = self.handle {
            let _ = handle.await;
        }
    }
}

/// Inner task loop: sleep(interval) → ping → repeat, until stopped or
/// break-even exceeded.
async fn run_keepalive_loop(
    provider: Arc<dyn Provider>,
    snapshot: KeepaliveSnapshot,
    profile: KeepaliveProfile,
    event_tx: tokio::sync::mpsc::Sender<AgentEvent>,
    stop: Arc<Notify>,
) {
    let start = Instant::now();
    let mut pings: u32 = 0;
    let mut total_cache_read: u64 = 0;
    let mut total_cache_creation: u64 = 0;

    let _ = event_tx
        .send(AgentEvent::CacheKeepaliveStarted {
            interval_secs: profile.interval.as_secs(),
            benefit: format!("{:?}", profile.benefit).to_lowercase(),
        })
        .await;

    loop {
        // Wait for either the next interval or a stop signal.
        tokio::select! {
            _ = stop.notified() => {
                emit_stopped(&event_tx, StopReason::PauseEnded, pings, total_cache_read, total_cache_creation).await;
                return;
            }
            _ = tokio::time::sleep(profile.interval) => {}
        }

        // Guard 3: break-even horizon — stop paying premiums, let cache die.
        if start.elapsed() >= profile.max_pause {
            emit_stopped(
                &event_tx,
                StopReason::BreakEvenExceeded,
                pings,
                total_cache_read,
                total_cache_creation,
            )
            .await;
            return;
        }

        // Send ping. Failures are logged + skipped, never propagated.
        match provider.keepalive_ping(&snapshot).await {
            Ok(usage) => {
                pings += 1;
                total_cache_read += usage.cache_read_tokens;
                total_cache_creation += usage.cache_creation_tokens;
                let _ = event_tx
                    .send(AgentEvent::CacheKeepalivePing {
                        pings,
                        cache_read_tokens: usage.cache_read_tokens,
                        cache_creation_tokens: usage.cache_creation_tokens,
                        elapsed_secs: start.elapsed().as_secs(),
                    })
                    .await;
            }
            Err(e) => {
                tracing::warn!(error = %e, "keepalive_ping_failed; will retry next interval");
            }
        }
    }
}

async fn emit_stopped(
    event_tx: &tokio::sync::mpsc::Sender<AgentEvent>,
    reason: StopReason,
    pings: u32,
    total_cache_read: u64,
    total_cache_creation: u64,
) {
    let _ = event_tx
        .send(AgentEvent::CacheKeepaliveStopped {
            reason: reason.as_str().into(),
            pings,
            total_cache_read_tokens: total_cache_read,
            total_cache_creation_tokens: total_cache_creation,
        })
        .await;
}
