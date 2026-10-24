//! Provider fallback (failover): wrap a primary provider with an ordered
//! chain of alternates (`[fallback]` config), retrying a failed step on the
//! next provider of the chain when — and only when — the failure is
//! failover-class AND nothing has been delivered downstream yet.
//!
//! Design contract (the "never fake success" rule, applied to failover):
//!
//! - **Error classification is deliberately tight.** Only 429 (`RateLimited`),
//!   5xx, 408, network/timeout transport errors, content-moderation 400s,
//!   and empty completions fail over. 401/403, 404, other 400s, context
//!   overflow, and configuration errors surface verbatim — retrying those on
//!   a different provider would mask a real problem.
//! - **Mid-stream failures never retry.** Failover is only allowed while
//!   ZERO content chunks (text, reasoning, or tool-use deltas) have been
//!   delivered downstream. Once any content is out, an error is propagated
//!   as-is: retrying would duplicate partial output in the transcript.
//! - **Failover is never silent.** Every switch emits a
//!   [`AgentEvent::ProviderFallback`] (when an event channel was wired at
//!   build time) plus a `tracing::warn`, naming both providers and the
//!   reason.
//! - **Chain exhaustion fails loudly** with a
//!   [`ProviderError::FallbackExhausted`] carrying one line per attempted
//!   provider — except for the empty-completion class, where the empty
//!   stream is forwarded as-is so the driver's existing empty-response
//!   policy (bounded retries, then a loud error) stays in charge.
//! - **Retry budget (chain-global).** `[fallback] max_retries = N` absorbs
//!   failover-class failures 1..N on the current model — redialing the SAME
//!   provider once per absorbed failure, spaced by the shared throttle
//!   clock — and failure N+1 (and every later one, on any model) advances
//!   the chain. Switching models does not re-arm the budget; only a new
//!   external user turn does ([`FallbackProvider::reset_for_new_turn`],
//!   called by the supervisor). `max_retries = 0` (the default) switches
//!   immediately, exactly like the pre-budget behavior. Permanent
//!   usage/quota failures ([`is_permanent_usage_error`]: 402 / billing /
//!   expired-plan wording) never charge the budget and advance at once.
//! - **Exhaustion is sticky within a turn descent.** Once the chain is
//!   spent (error class), the aggregate error is cached: every later
//!   `chat()` on this instance returns it immediately instead of
//!   replaying the whole chain against dead providers.
//! - **Escapes are recorded for child avoidance.** Every switch records the
//!   provider it advanced away from ([`FallbackProvider::escaped_providers`]);
//!   the subagent spawn path drops those providers from freshly built
//!   child fallback chains.

use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use nca_common::config::ProviderKind;
use nca_common::event::AgentEvent;
use nca_common::message::Message;
use nca_common::tool::ToolDefinition;
use tokio::sync::mpsc;
use tokio::time::Instant;

use super::openai_compat::{truncate_bytes_safe, truncate_tail_bytes_safe};
use super::{Provider, ProviderError, StreamChunk};
use crate::cache_keepalive::{KeepaliveSnapshot, PingUsage};

/// Body-substring markers for content-moderation rejections on a 400.
///
/// OpenAI-class APIs reject policy-violating prompts with
/// `code: "content_policy_violation"`; DeepSeek uses a `cyber_policy`-class
/// code; others say "moderation". A 400 matching any marker is treated as
/// transient-for-this-prompt (failover-able): another provider may accept the
/// same prompt. Kept deliberately tight — a false positive routes auth or
/// schema errors to failover; a false negative merely surfaces the error
/// verbatim (graceful degradation).
const CONTENT_MODERATION_MARKERS: [&str; 4] = [
    "content_policy_violation",
    "content policy violation",
    "cyber_policy",
    "moderation",
];

/// Local request-build guard errors (openai_compat refuses to send empty or
/// malformed message lists). Deterministic client-side rejections: retrying
/// the identical request on another provider reproduces the same guard.
const LOCAL_GUARD_PREFIX: &str = "refusing to send";

/// Body/message-substring markers for permanent usage/quota failures:
/// billing-side account states the current model cannot recover from
/// within the turn (omos `isPermanentUsageQuotaError` parity, kept tight).
const PERMANENT_USAGE_MARKERS: [&str; 8] = [
    "insufficient balance",
    "billing",
    "plan expired",
    "package expired",
    "plan has expired",
    "package has expired",
    "quota exhausted",
    "permanently exhausted",
];

/// Whether `err` is failover-class: retrying the step on the next provider
/// of the fallback chain is reasonable.
///
/// - 429 (`RateLimited`): yes — the provider is throttling us specifically.
/// - 5xx / 408 (`Http`): yes — server-side or timeout trouble.
/// - 400 with a content-moderation body: yes — moderation is
///   provider-specific; another provider may accept the prompt.
/// - Network/timeout transport errors (`RequestFailed` chains): yes —
///   includes client response-header timeouts. Context-overflow bodies and
///   local request guards are excluded (retrying identical input elsewhere
///   cannot help).
/// - 401/403 (`AuthError`), 404 (`ModelNotFound`), other 400s,
///   configuration errors: no — surface verbatim.
pub fn is_failoverable(err: &ProviderError) -> bool {
    match err {
        ProviderError::RateLimited { .. } => true,
        ProviderError::Http { status, body } => match *status {
            500..=599 | 408 => true,
            400 => is_content_moderation(body),
            _ => false,
        },
        ProviderError::RequestFailed(_) => !err.is_context_overflow() && !is_local_guard(err),
        ProviderError::AuthError(_)
        | ProviderError::ModelNotFound(_)
        | ProviderError::Configuration(_)
        | ProviderError::FallbackExhausted { .. }
        | ProviderError::Other(_) => false,
    }
}

/// Case-insensitive check of `body` against [`CONTENT_MODERATION_MARKERS`].
fn is_content_moderation(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    CONTENT_MODERATION_MARKERS
        .iter()
        .any(|marker| lower.contains(marker))
}

/// Whether `err` is a permanent usage/quota failure — payment required, an
/// exhausted plan/package, or explicit quota exhaustion. Such failures skip
/// the same-model retry budget entirely (no amount of redialing recovers a
/// dead account) and advance the fallback chain immediately.
///
/// Typed status wins where [`ProviderError`] carries one (HTTP 402);
/// otherwise the markers are matched case-insensitively against the HTTP
/// body or transport message. Kept deliberately tight: a false positive
/// merely switches providers one hop early; a false negative burns one
/// redial on a dead account. Ordinary 429 / short-term rate-limit wording
/// is deliberately NOT permanent.
pub fn is_permanent_usage_error(err: &ProviderError) -> bool {
    match err {
        ProviderError::Http { status: 402, .. } => true,
        ProviderError::Http { body, .. } => contains_permanent_marker(body),
        ProviderError::RequestFailed(msg) => contains_permanent_marker(msg),
        _ => false,
    }
}

/// Case-insensitive check of `text` against [`PERMANENT_USAGE_MARKERS`].
fn contains_permanent_marker(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    PERMANENT_USAGE_MARKERS
        .iter()
        .any(|marker| lower.contains(marker))
}

/// Whether `err` (already known to be `RequestFailed`) is a deterministic
/// local request-build guard rather than a transport failure.
fn is_local_guard(err: &ProviderError) -> bool {
    match err {
        ProviderError::RequestFailed(msg) => msg.starts_with(LOCAL_GUARD_PREFIX),
        _ => false,
    }
}

/// Short failure-class label used in fallback notifications and logs.
fn reason_label(err: &ProviderError) -> &'static str {
    match err {
        ProviderError::RateLimited { .. } => "rate_limited",
        ProviderError::Http { status: 400, body } if is_content_moderation(body) => {
            "content_moderation"
        }
        ProviderError::Http { status, .. } if (500..=599).contains(status) => "server_error",
        ProviderError::Http { status: 408, .. } => "timeout",
        ProviderError::Http { .. } => "http_error",
        ProviderError::RequestFailed(_) => "network_error",
        _ => "error",
    }
}

/// One-line failure record for the aggregate error: `name: error`.
fn fail_line(name: &str, err: &ProviderError) -> String {
    format!("{name}: {}", truncate_bytes_safe(&err.to_string(), 300))
}

/// The chain-exhausted aggregate error: names every attempted provider and
/// why each failed.
fn aggregate(attempted: &[String], failures: &[String]) -> ProviderError {
    ProviderError::FallbackExhausted {
        chain: attempted.join(" → "),
        reasons: if failures.is_empty() {
            "no failure recorded".into()
        } else {
            failures.join("; ")
        },
    }
}

/// Anti-storm throttle shared by the chat()-level loop and the stream
/// forwarder task of one [`FallbackProvider`] instance.
///
/// The first switch of an instance waits `initial_retry_delay_ms`; every
/// subsequent switch is spaced at least `retry_delay_ms` after the previous
/// one (measured switch-initiation to switch-initiation). Uses
/// [`tokio::time::Instant`] so the throttle clock matches the sleep clock
/// (and stays testable under `start_paused`).
#[derive(Clone)]
struct Throttle {
    initial: Duration,
    gap: Duration,
    last_fallback: Arc<Mutex<Option<Instant>>>,
}

impl Throttle {
    fn new(initial_retry_delay_ms: u64, retry_delay_ms: u64) -> Self {
        Self {
            initial: Duration::from_millis(initial_retry_delay_ms),
            gap: Duration::from_millis(retry_delay_ms),
            last_fallback: Arc::new(Mutex::new(None)),
        }
    }

    /// Compute the wait for this switch, mark the switch time, and sleep.
    /// A poisoned lock is recovered from (the clock state is advisory).
    async fn wait_and_mark(&self) {
        let delay = {
            let mut last = match self.last_fallback.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            let delay = match *last {
                None => self.initial,
                Some(previous) => {
                    let elapsed = previous.elapsed();
                    if elapsed >= self.gap {
                        Duration::ZERO
                    } else {
                        self.gap - elapsed
                    }
                }
            };
            *last = Some(Instant::now());
            delay
        };
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
    }
}

/// One entry of the fallback chain: a display name plus the built provider.
/// Fallback providers are called with an empty model string so each uses its
/// own `[provider.<name>]` configured model.
#[derive(Clone)]
pub struct FallbackEntry {
    /// Display name (for events, logs, and the aggregate error).
    pub name: String,
    /// The built provider.
    pub provider: Arc<dyn Provider>,
}

/// Chain-global state shared between the `chat()`-level loop and the spawned
/// stream forwarder of one [`FallbackProvider`] instance: the retry budget,
/// the post-exhaustion cache, and the escaped-provider record.
struct FallbackState {
    /// `[fallback] max_retries` — immutable for the instance. 0 preserves
    /// the switch-immediately behavior.
    max_retries: u32,
    /// Failures absorbed so far in this descent. Chain-global: incremented
    /// for every absorbed failure on ANY model and only cleared by
    /// [`FallbackProvider::reset_for_new_turn`].
    absorbed: AtomicU32,
    /// Post-exhaustion cache: once the chain is spent (error class), the
    /// aggregate error is recorded here and every later `chat()` returns it
    /// without replaying the walk. Cleared only by a new external turn.
    exhausted: Mutex<Option<ProviderError>>,
    /// Providers this instance has escaped (a switch advanced away from
    /// them), as canonical display names. Deliberately NOT cleared by
    /// [`FallbackProvider::reset_for_new_turn`] — a provider the session
    /// already escaped stays avoided for the session's children.
    escaped: Arc<Mutex<HashSet<String>>>,
}

impl FallbackState {
    fn new(max_retries: u32) -> Self {
        Self {
            max_retries,
            absorbed: AtomicU32::new(0),
            exhausted: Mutex::new(None),
            escaped: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Try to charge one same-model absorption against the budget. Returns
    /// `false` when the chain-global budget is already spent (in which case
    /// nothing is charged). Atomic so concurrent `chat()` calls on one
    /// instance can never overshoot the budget.
    fn try_absorb(&self) -> bool {
        self.absorbed
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |absorbed| {
                (absorbed < self.max_retries).then(|| absorbed + 1)
            })
            .is_ok()
    }

    /// Failures absorbed so far in this descent (diagnostics/logs).
    fn absorbed(&self) -> u32 {
        self.absorbed.load(Ordering::SeqCst)
    }

    /// Record the aggregate error of a spent chain (error class only — the
    /// empty-completion class forwards `Done` and stays driver-owned).
    fn record_exhausted(&self, err: ProviderError) {
        *lock_or_recover(&self.exhausted) = Some(err);
    }

    /// The recorded terminal error, when the chain is spent.
    fn cached_exhausted(&self) -> Option<ProviderError> {
        lock_or_recover(&self.exhausted).clone()
    }

    /// Record `from` — the model a switch advanced away from — as escaped,
    /// when its provider is recognizable. This is the same `from` value the
    /// corresponding [`AgentEvent::ProviderFallback`] carries; recording it
    /// here (rather than in the supervisor) keeps the set exactly in sync
    /// with the events without tapping the single-consumer event channel.
    fn record_escaped(&self, from: &str) {
        if let Some(key) = escaped_provider_key(from)
            && !lock_or_recover(&self.escaped).contains(key)
        {
            lock_or_recover(&self.escaped).insert(key.to_string());
        }
    }

    /// Per-external-turn reset: clears the absorbed budget and the
    /// exhaustion cache. The throttle clock (anti-storm, cross-turn by
    /// design) and the escaped-provider record are deliberately untouched.
    fn reset_for_new_turn(&self) {
        self.absorbed.store(0, Ordering::SeqCst);
        *lock_or_recover(&self.exhausted) = None;
    }
}

/// Lock a std `Mutex`, recovering from poison (all guarded state here is
/// advisory and single-writer in practice; a panic elsewhere must not wedge
/// failover).
fn lock_or_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Canonical provider display name for a fallback `from`/entry name, when
/// recognizable — e.g. `ZhipuAI(glm-5.3-flash)` → `ZhipuAI`. Unrecognizable
/// names (test doubles like `Primary`) map to `None` and are not recorded.
fn escaped_provider_key(from: &str) -> Option<&'static str> {
    let base = from.split('(').next()?.trim();
    ProviderKind::parse_display_name(base).map(|kind| kind.display_name())
}

/// The model currently being tried: display name, provider handle, and the
/// model string it is dialed with (the session model for the primary,
/// empty — "use your configured model" — for chain entries).
struct CurrentModel {
    name: String,
    provider: Arc<dyn Provider>,
    model: String,
}

/// A [`Provider`] that delegates to a primary provider and, on
/// failover-class failures with zero delivered content, retries the step on
/// an ordered chain of alternates. Built by
/// [`crate::provider::factory::build_provider_with_events`] when
/// `[fallback] enabled = true`; the [`Provider`] trait itself is unchanged.
///
/// `prepare_messages_for_request` and `keepalive_ping` always delegate to
/// the primary: request normalization follows the session's main provider
/// semantics, and cache keepalives refresh the primary's prompt cache.
pub struct FallbackProvider {
    primary: Arc<dyn Provider>,
    primary_name: String,
    fallbacks: Vec<FallbackEntry>,
    throttle: Throttle,
    state: Arc<FallbackState>,
    event_tx: Option<mpsc::Sender<AgentEvent>>,
}

impl FallbackProvider {
    /// Wrap `primary` with `fallbacks` (in order). `max_retries` is the
    /// chain-global same-model absorption budget (`[fallback] max_retries`,
    /// 0 = switch immediately). Notifications are sent on `event_tx` when
    /// present; without one they degrade to `tracing` only.
    pub fn new(
        primary: Arc<dyn Provider>,
        primary_name: String,
        fallbacks: Vec<FallbackEntry>,
        initial_retry_delay_ms: u64,
        retry_delay_ms: u64,
        max_retries: u32,
        event_tx: Option<mpsc::Sender<AgentEvent>>,
    ) -> Self {
        Self {
            primary,
            primary_name,
            fallbacks,
            throttle: Throttle::new(initial_retry_delay_ms, retry_delay_ms),
            state: Arc::new(FallbackState::new(max_retries)),
            event_tx,
        }
    }

    /// Reset the chain-global retry budget and exhaustion cache for a new
    /// EXTERNAL user turn. The supervisor calls this at each new user
    /// message (see `Supervisor::run_turn_with_images`); internal replays
    /// and wake-injected prompts must NOT reset it — a fresh descent is what
    /// re-arms absorption. The anti-storm throttle clock and the
    /// escaped-provider record are deliberately preserved.
    pub fn reset_for_new_turn(&self) {
        self.state.reset_for_new_turn();
    }

    /// Snapshot of providers this instance has escaped via failover, as
    /// canonical display names. Consulted when spawning child sessions so
    /// their fallback chains avoid providers the parent already left.
    pub fn escaped_providers(&self) -> HashSet<String> {
        lock_or_recover(&self.state.escaped).clone()
    }

    /// Live handle to the escaped-provider set, for spawn consumers that
    /// need a per-spawn snapshot (see
    /// `nca_runtime`'s subagent spawn path). Shares the exact set this
    /// instance records at each switch.
    pub fn escaped_providers_handle(&self) -> Arc<Mutex<HashSet<String>>> {
        Arc::clone(&self.state.escaped)
    }

    /// Display names of the fallback chain, in order (diagnostics/tests).
    pub fn chain_names(&self) -> Vec<&str> {
        self.fallbacks.iter().map(|e| e.name.as_str()).collect()
    }

    /// Spawn the stream forwarder: forwards chunks from the current model
    /// downstream, guarding the zero-content rule, and — while no content
    /// has been delivered — applies the recovery policy ([`recover`]: absorb
    /// on the same model while budget remains, else advance the chain) on
    /// failover-class errors and empty completions.
    #[allow(clippy::too_many_arguments)]
    fn spawn_forwarder(
        current: CurrentModel,
        current_rx: mpsc::Receiver<StreamChunk>,
        remaining: Vec<FallbackEntry>,
        messages: Vec<Message>,
        tools: Vec<ToolDefinition>,
        workspace_root: PathBuf,
        attempted: Vec<String>,
        failures: Vec<String>,
        state: Arc<FallbackState>,
        throttle: Throttle,
        event_tx: Option<mpsc::Sender<AgentEvent>>,
    ) -> mpsc::Receiver<StreamChunk> {
        let (tx, rx) = mpsc::channel(64);
        tokio::spawn(async move {
            let mut current = current;
            let mut current_rx = current_rx;
            let mut remaining: VecDeque<FallbackEntry> = remaining.into();
            let mut attempted = attempted;
            let mut failures = failures;
            // Any delivered text/reasoning/tool-use delta locks out failover.
            let mut content_delivered = false;
            // A zero-content Finish is held back: it belongs to a generation
            // that may be discarded by failover. Flushed before any terminal
            // chunk that reaches the driver.
            let mut held_finish: Option<String> = None;

            loop {
                let Some(chunk) = current_rx.recv().await else {
                    // Inner channel closed without Done/Error (provider bug):
                    // propagate the close downstream, same as pre-fallback.
                    return;
                };
                match chunk {
                    StreamChunk::TextDelta(delta) => {
                        content_delivered = true;
                        flush_finish(&tx, &mut held_finish).await;
                        if tx.send(StreamChunk::TextDelta(delta)).await.is_err() {
                            return;
                        }
                    }
                    StreamChunk::ReasoningDelta(delta) => {
                        // Reasoning is streamed to the UI even though it is
                        // stripped from history — delivered is delivered.
                        content_delivered = true;
                        flush_finish(&tx, &mut held_finish).await;
                        if tx.send(StreamChunk::ReasoningDelta(delta)).await.is_err() {
                            return;
                        }
                    }
                    StreamChunk::ToolUse(call) => {
                        content_delivered = true;
                        flush_finish(&tx, &mut held_finish).await;
                        if tx.send(StreamChunk::ToolUse(call)).await.is_err() {
                            return;
                        }
                    }
                    StreamChunk::Usage { .. } => {
                        // Usage is forwarded even on attempts that later fail
                        // over: both generations were billed, so both count.
                        if tx.send(chunk).await.is_err() {
                            return;
                        }
                    }
                    StreamChunk::Finish { reason } => {
                        if content_delivered {
                            if tx.send(StreamChunk::Finish { reason }).await.is_err() {
                                return;
                            }
                        } else {
                            held_finish = Some(reason);
                        }
                    }
                    StreamChunk::Done => {
                        if content_delivered {
                            flush_finish(&tx, &mut held_finish).await;
                            let _ = tx.send(StreamChunk::Done).await;
                            return;
                        }
                        // Empty completion: recover (advance — the empty
                        // class never charges the retry budget); if the chain
                        // is exhausted, forward the empty stream verbatim so
                        // the driver's empty-response policy fails loudly. No
                        // exhausted-cache record for this class: the driver
                        // owns empty-completion retries.
                        match recover(
                            &mut current,
                            &mut remaining,
                            &messages,
                            &tools,
                            &workspace_root,
                            FailureCause::EmptyCompletion,
                            &mut attempted,
                            &mut failures,
                            &state,
                            &throttle,
                            &event_tx,
                        )
                        .await
                        {
                            Recovered::Stream(next) => {
                                current_rx = next;
                                // Drop the failed generation's held Finish:
                                // its completion never happened.
                                held_finish = None;
                            }
                            Recovered::Exhausted => {
                                flush_finish(&tx, &mut held_finish).await;
                                let _ = tx.send(StreamChunk::Done).await;
                                return;
                            }
                            Recovered::Fatal(err) => {
                                flush_finish(&tx, &mut held_finish).await;
                                let _ = tx.send(StreamChunk::Error(err)).await;
                                return;
                            }
                        }
                    }
                    StreamChunk::Error(err) => {
                        if content_delivered || !is_failoverable(&err) {
                            flush_finish(&tx, &mut held_finish).await;
                            let _ = tx.send(StreamChunk::Error(err)).await;
                            return;
                        }
                        let from = current.name.clone();
                        failures.push(fail_line(&from, &err));
                        match recover(
                            &mut current,
                            &mut remaining,
                            &messages,
                            &tools,
                            &workspace_root,
                            FailureCause::Error(err),
                            &mut attempted,
                            &mut failures,
                            &state,
                            &throttle,
                            &event_tx,
                        )
                        .await
                        {
                            Recovered::Stream(next) => {
                                current_rx = next;
                                // Drop the failed generation's held Finish:
                                // its completion never happened.
                                held_finish = None;
                            }
                            Recovered::Exhausted => {
                                flush_finish(&tx, &mut held_finish).await;
                                let aggregate_err = aggregate(&attempted, &failures);
                                state.record_exhausted(aggregate_err.clone());
                                let _ = tx.send(StreamChunk::Error(aggregate_err)).await;
                                return;
                            }
                            Recovered::Fatal(fatal) => {
                                flush_finish(&tx, &mut held_finish).await;
                                let _ = tx.send(StreamChunk::Error(fatal)).await;
                                return;
                            }
                        }
                    }
                }
            }
        });
        rx
    }
}

/// Emit a held zero-content `Finish` chunk before a terminal chunk.
async fn flush_finish(tx: &mpsc::Sender<StreamChunk>, held: &mut Option<String>) {
    if let Some(reason) = held.take() {
        let _ = tx.send(StreamChunk::Finish { reason }).await;
    }
}

/// Why the current model is being recovered from: a failover-class error
/// (retryable or permanent) or an empty completion (driver-policy class —
/// never absorbed).
enum FailureCause {
    Error(ProviderError),
    EmptyCompletion,
}

impl FailureCause {
    /// Whether this cause may charge the same-model retry budget. Permanent
    /// usage/quota failures never absorb — no redial recovers a dead
    /// account — and the empty-completion class is driver-owned.
    fn absorbable(&self) -> bool {
        match self {
            FailureCause::Error(err) => !is_permanent_usage_error(err),
            FailureCause::EmptyCompletion => false,
        }
    }

    /// Notification reason for the next switch: failure class + short text.
    fn reason(&self) -> String {
        match self {
            FailureCause::Error(err) => format!(
                "{}: {}",
                reason_label(err),
                truncate_tail_bytes_safe(&err.to_string(), 160)
            ),
            FailureCause::EmptyCompletion => "empty completion".to_string(),
        }
    }
}

/// Outcome of the recovery walk.
enum Recovered {
    /// A provider produced a stream (same model after an absorbed failure,
    /// or the next chain entry): continue forwarding from it.
    Stream(mpsc::Receiver<StreamChunk>),
    /// The chain ran out of providers.
    Exhausted,
    /// A provider failed non-failover-class: surface its error verbatim.
    Fatal(ProviderError),
}

/// Recovery policy shared by the `chat()`-level loop and the stream
/// forwarder, applied whenever the CURRENT model just failed with `cause`
/// (its failure line already recorded by the caller):
///
/// - **Absorb while budget remains.** A failover-class, non-permanent
///   failure with `absorbed < max_retries` is absorbed — the budget is
///   CHAIN-GLOBAL, so switching models does not re-arm it — and the SAME
///   provider is redialed once per absorbed failure, spaced by the shared
///   throttle clock. `max_retries = 0` skips straight to advancement
///   (pre-budget behavior). A same-model redial emits no
///   [`AgentEvent::ProviderFallback`]: no provider switch happened.
/// - **Permanent usage/quota failures fast-fail**: no absorption, advance
///   immediately.
/// - **Advance the chain** once absorption is exhausted or disallowed: each
///   switch is throttled, announced (event + log), and records the escaped
///   provider. Failover-class errors from the new model continue the walk
///   (with the same absorb/advance decision); non-failover errors stop it
///   verbatim.
#[allow(clippy::too_many_arguments)]
async fn recover(
    current: &mut CurrentModel,
    remaining: &mut VecDeque<FallbackEntry>,
    messages: &[Message],
    tools: &[ToolDefinition],
    workspace_root: &Path,
    mut cause: FailureCause,
    attempted: &mut Vec<String>,
    failures: &mut Vec<String>,
    state: &FallbackState,
    throttle: &Throttle,
    event_tx: &Option<mpsc::Sender<AgentEvent>>,
) -> Recovered {
    loop {
        if cause.absorbable() && state.try_absorb() {
            throttle.wait_and_mark().await;
            tracing::warn!(
                provider = %current.name,
                absorbed = state.absorbed(),
                budget = state.max_retries,
                "[fallback] absorbing failure; retrying same provider"
            );
            match current
                .provider
                .chat(messages, tools, &current.model, workspace_root)
                .await
            {
                Ok(stream) => return Recovered::Stream(stream),
                Err(err) if is_failoverable(&err) => {
                    failures.push(fail_line(&current.name, &err));
                    cause = FailureCause::Error(err);
                    continue;
                }
                Err(err) => return Recovered::Fatal(err),
            }
        }
        // Budget spent, permanent failure, or empty completion: advance.
        let Some(entry) = remaining.pop_front() else {
            return Recovered::Exhausted;
        };
        throttle.wait_and_mark().await;
        let from = current.name.clone();
        state.record_escaped(&from);
        notify_fallback(event_tx, &from, &entry.name, &cause.reason()).await;
        // Empty model string: each fallback provider uses its own
        // `[provider.<name>]` configured model.
        match entry
            .provider
            .chat(messages, tools, "", workspace_root)
            .await
        {
            Ok(stream) => {
                attempted.push(entry.name.clone());
                *current = CurrentModel {
                    name: entry.name,
                    provider: entry.provider,
                    model: String::new(),
                };
                return Recovered::Stream(stream);
            }
            Err(err) if is_failoverable(&err) => {
                attempted.push(entry.name.clone());
                failures.push(fail_line(&entry.name, &err));
                // The current model is now the failed entry: a later
                // absorbable failure may retry IT (budget permitting), and
                // the next notification (if the walk continues) names THIS
                // failure, not the one that started the walk.
                *current = CurrentModel {
                    name: entry.name,
                    provider: entry.provider,
                    model: String::new(),
                };
                cause = FailureCause::Error(err);
                continue;
            }
            Err(err) => return Recovered::Fatal(err),
        }
    }
}

/// Announce a fallback switch: always a `tracing::warn`, plus an
/// [`AgentEvent::ProviderFallback`] when an event channel is wired. Best
/// effort — a closed channel never blocks the failover path.
async fn notify_fallback(
    event_tx: &Option<mpsc::Sender<AgentEvent>>,
    from: &str,
    to: &str,
    reason: &str,
) {
    tracing::warn!(
        provider_from = from,
        provider_to = to,
        reason = reason,
        "provider_fallback"
    );
    if let Some(tx) = event_tx {
        let _ = tx
            .send(AgentEvent::ProviderFallback {
                from: from.to_string(),
                to: to.to_string(),
                reason: reason.to_string(),
            })
            .await;
    }
}

#[async_trait]
impl Provider for FallbackProvider {
    async fn prepare_messages_for_request(
        &self,
        messages: &mut Vec<Message>,
        workspace_root: &Path,
    ) -> Result<(), ProviderError> {
        self.primary
            .prepare_messages_for_request(messages, workspace_root)
            .await
    }

    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        model: &str,
        workspace_root: &Path,
    ) -> Result<mpsc::Receiver<StreamChunk>, ProviderError> {
        // Sticky exhaustion: once the chain is spent (error class), every
        // later chat() on this instance returns the recorded aggregate
        // immediately — replaying the walk would hammer dead providers.
        // Cleared only by `reset_for_new_turn` (a new external user turn).
        if let Some(err) = self.state.cached_exhausted() {
            return Err(err);
        }
        let mut attempted = vec![self.primary_name.clone()];
        let mut failures: Vec<String> = Vec::new();
        let mut current = CurrentModel {
            name: self.primary_name.clone(),
            provider: Arc::clone(&self.primary),
            model: model.to_string(),
        };

        let current_rx = match current
            .provider
            .chat(messages, tools, model, workspace_root)
            .await
        {
            Ok(stream) => stream,
            Err(err) => {
                if !is_failoverable(&err) {
                    return Err(err);
                }
                failures.push(fail_line(&current.name, &err));
                let mut remaining: VecDeque<FallbackEntry> =
                    self.fallbacks.iter().cloned().collect();
                match recover(
                    &mut current,
                    &mut remaining,
                    messages,
                    tools,
                    workspace_root,
                    FailureCause::Error(err),
                    &mut attempted,
                    &mut failures,
                    &self.state,
                    &self.throttle,
                    &self.event_tx,
                )
                .await
                {
                    Recovered::Stream(stream) => stream,
                    Recovered::Exhausted => {
                        let err = aggregate(&attempted, &failures);
                        self.state.record_exhausted(err.clone());
                        return Err(err);
                    }
                    Recovered::Fatal(err) => return Err(err),
                }
            }
        };

        let remaining: Vec<FallbackEntry> = self.fallbacks.to_vec();
        Ok(Self::spawn_forwarder(
            current,
            current_rx,
            remaining,
            messages.to_vec(),
            tools.to_vec(),
            workspace_root.to_path_buf(),
            attempted,
            failures,
            Arc::clone(&self.state),
            self.throttle.clone(),
            self.event_tx.clone(),
        ))
    }

    async fn keepalive_ping(
        &self,
        snapshot: &KeepaliveSnapshot,
    ) -> Result<PingUsage, ProviderError> {
        self.primary.keepalive_ping(snapshot).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Error classification (spec: minimal set) ----

    #[test]
    fn rate_limited_is_failoverable() {
        assert!(is_failoverable(&ProviderError::RateLimited {
            retry_after_ms: 1_000
        }));
    }

    #[test]
    fn http_5xx_and_408_are_failoverable() {
        for status in [500u16, 502, 503, 504, 408] {
            assert!(
                is_failoverable(&ProviderError::Http {
                    status,
                    body: "upstream error".into()
                }),
                "HTTP {status} must be failover-able"
            );
        }
    }

    #[test]
    fn content_moderation_400_is_failoverable() {
        let bodies = [
            r#"{"error":{"code":"content_policy_violation","message":"Your request was rejected"}}"#,
            r#"{"error":{"code":"cyber_policy","message":"rejected"}}"#,
            r#"{"error":{"message":"Request rejected by the Moderation system"}}"#,
        ];
        for body in bodies {
            assert!(
                is_failoverable(&ProviderError::Http {
                    status: 400,
                    body: body.into()
                }),
                "moderation 400 must be failover-able: {body}"
            );
        }
    }

    #[test]
    fn other_400s_are_not_failoverable() {
        let cases = [
            r#"{"error":{"message":"Invalid schema for function","type":"invalid_request_error"}}"#,
            "Insufficient Balance",
        ];
        for body in cases {
            assert!(
                !is_failoverable(&ProviderError::Http {
                    status: 400,
                    body: body.into()
                }),
                "non-moderation 400 must surface verbatim: {body}"
            );
        }
    }

    #[test]
    fn auth_model_and_config_errors_are_not_failoverable() {
        assert!(!is_failoverable(&ProviderError::AuthError(
            "invalid api key".into()
        )));
        assert!(!is_failoverable(&ProviderError::ModelNotFound(
            "no such model".into()
        )));
        assert!(!is_failoverable(&ProviderError::Configuration(
            "missing key".into()
        )));
        assert!(!is_failoverable(&ProviderError::Http {
            status: 402,
            body: "payment required".into()
        }));
    }

    #[test]
    fn network_errors_are_failoverable_except_overflow_and_local_guards() {
        assert!(is_failoverable(&ProviderError::RequestFailed(
            "error sending request for url (https://api.deepseek.com/chat/completions): operation timed out".into()
        )));
        assert!(is_failoverable(&ProviderError::RequestFailed(
            "DeepSeek stream error: connection reset by peer".into()
        )));
        // Context overflow phrased any of the four known ways.
        assert!(!is_failoverable(&ProviderError::RequestFailed(
            "This model's maximum context length is 65536 tokens".into()
        )));
        assert!(!is_failoverable(&ProviderError::Http {
            status: 400,
            body: "prompt is too long: 200001 tokens > 200000 maximum".into()
        }));
        // Local deterministic guards.
        assert!(!is_failoverable(&ProviderError::RequestFailed(
            "refusing to send a request with an empty messages list".into()
        )));
    }

    // ---- Scripted providers for behavior tests ----

    /// A provider whose `chat()` returns a scripted result per call and
    /// records the model string it was called with.
    struct ScriptedProvider {
        calls: Mutex<Vec<String>>,
        script: Mutex<Vec<ScriptRound>>,
    }

    enum ScriptRound {
        ChatErr(ProviderError),
        Stream(&'static [&'static str]),
    }

    impl ScriptedProvider {
        fn new(script: Vec<ScriptRound>) -> Arc<Self> {
            Arc::new(Self {
                calls: Mutex::new(Vec::new()),
                script: Mutex::new(script),
            })
        }

        fn call_count(&self) -> usize {
            self.calls.lock().map(|c| c.len()).unwrap_or(0)
        }
    }

    #[async_trait]
    impl Provider for ScriptedProvider {
        async fn chat(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
            model: &str,
            _workspace_root: &Path,
        ) -> Result<mpsc::Receiver<StreamChunk>, ProviderError> {
            if let Ok(mut calls) = self.calls.lock() {
                calls.push(model.to_string());
            }
            let round = match self.script.lock() {
                Ok(mut s) if !s.is_empty() => s.remove(0),
                _ => ScriptRound::Stream(&["hello", "\u{2026}done"]),
            };
            match round {
                ScriptRound::ChatErr(err) => Err(err),
                ScriptRound::Stream(texts) => {
                    let (tx, rx) = mpsc::channel(8);
                    let texts: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
                    tokio::spawn(async move {
                        for text in texts {
                            let _ = tx.send(StreamChunk::TextDelta(text)).await;
                        }
                        let _ = tx.send(StreamChunk::Done).await;
                    });
                    Ok(rx)
                }
            }
        }
    }

    fn fallback(
        primary: Arc<dyn Provider>,
        fallbacks: Vec<FallbackEntry>,
        event_tx: Option<mpsc::Sender<AgentEvent>>,
    ) -> FallbackProvider {
        FallbackProvider::new(
            primary,
            "Primary".into(),
            fallbacks,
            0, // initial delay: instant first switch (keeps tests fast)
            0, // retry gap: instant subsequent switches
            0, // max_retries: switch immediately (pre-budget behavior)
            event_tx,
        )
    }

    async fn collect(mut rx: mpsc::Receiver<StreamChunk>) -> Vec<StreamChunk> {
        let mut out = Vec::new();
        while let Some(chunk) = rx.recv().await {
            let done = matches!(chunk, StreamChunk::Done);
            let error = matches!(chunk, StreamChunk::Error(_));
            out.push(chunk);
            if done || error {
                break;
            }
        }
        out
    }

    fn text_of(chunks: &[StreamChunk]) -> String {
        chunks
            .iter()
            .map(|c| match c {
                StreamChunk::TextDelta(t) => t.as_str(),
                _ => "",
            })
            .collect()
    }

    // chat()-level failover: primary 429 → fallback streams.
    #[tokio::test]
    async fn chat_error_fails_over_to_next_provider() {
        let primary =
            ScriptedProvider::new(vec![ScriptRound::ChatErr(ProviderError::RateLimited {
                retry_after_ms: 1_000,
            })]);
        let secondary = ScriptedProvider::new(vec![]);
        let (event_tx, mut event_rx) = mpsc::channel(16);

        let provider = fallback(
            primary.clone(),
            vec![FallbackEntry {
                name: "Secondary".into(),
                provider: secondary.clone(),
            }],
            Some(event_tx),
        );

        let chunks = collect(
            provider
                .chat(&[Message::user("hi")], &[], "primary-model", Path::new("."))
                .await
                .expect("fallback stream"),
        )
        .await;
        assert_eq!(text_of(&chunks), "hello…done");
        // Fallback providers are called with an EMPTY model string so they
        // use their own configured model.
        assert_eq!(secondary.calls.lock().unwrap()[0], "");

        // Notification: exactly one ProviderFallback event, naming both
        // providers with the failure class.
        let mut saw = 0;
        while let Ok(event) = event_rx.try_recv() {
            if let AgentEvent::ProviderFallback { from, to, reason } = event {
                saw += 1;
                assert_eq!(from, "Primary");
                assert_eq!(to, "Secondary");
                assert!(reason.starts_with("rate_limited"), "reason: {reason}");
            }
        }
        assert_eq!(saw, 1, "fallback must never be silent");
    }

    // Non-failover chat() error surfaces verbatim; fallback never called.
    #[tokio::test]
    async fn non_failover_chat_error_surfaces_verbatim() {
        let primary = ScriptedProvider::new(vec![ScriptRound::ChatErr(ProviderError::AuthError(
            "invalid api key".into(),
        ))]);
        let secondary = ScriptedProvider::new(vec![]);
        let provider = fallback(
            primary.clone(),
            vec![FallbackEntry {
                name: "Secondary".into(),
                provider: secondary.clone(),
            }],
            None,
        );
        let err = provider
            .chat(&[Message::user("hi")], &[], "m", Path::new("."))
            .await
            .expect_err("auth error must surface");
        assert!(matches!(err, ProviderError::AuthError(_)));
        assert_eq!(secondary.call_count(), 0);
    }

    // Chain exhaustion at chat() level: aggregate error with reason chain.
    #[tokio::test]
    async fn chat_level_exhaustion_returns_aggregate_error() {
        let primary = ScriptedProvider::new(vec![ScriptRound::ChatErr(ProviderError::Http {
            status: 503,
            body: "upstream unavailable".into(),
        })]);
        let secondary =
            ScriptedProvider::new(vec![ScriptRound::ChatErr(ProviderError::RateLimited {
                retry_after_ms: 1,
            })]);
        let provider = fallback(
            primary,
            vec![FallbackEntry {
                name: "Secondary".into(),
                provider: secondary,
            }],
            None,
        );
        let err = provider
            .chat(&[Message::user("hi")], &[], "m", Path::new("."))
            .await
            .expect_err("chain exhausted");
        match err {
            ProviderError::FallbackExhausted { chain, reasons } => {
                assert_eq!(chain, "Primary → Secondary");
                assert!(reasons.contains("Primary:"), "reasons: {reasons}");
                assert!(reasons.contains("Secondary:"), "reasons: {reasons}");
            }
            other => panic!("expected FallbackExhausted, got {other}"),
        }
    }

    // Mid-stream error with zero delivered content → failover.
    #[tokio::test]
    async fn midstream_error_before_content_fails_over() {
        let (primary_tx, primary_rx) = mpsc::channel(8);
        tokio::spawn(async move {
            let _ = primary_tx
                .send(StreamChunk::Error(ProviderError::RequestFailed(
                    "connection reset by peer".into(),
                )))
                .await;
        });
        let primary = StreamOnceProvider {
            rx: Mutex::new(Some(primary_rx)),
        };
        let secondary = ScriptedProvider::new(vec![]);
        let provider = fallback(
            Arc::new(primary),
            vec![FallbackEntry {
                name: "Secondary".into(),
                provider: secondary,
            }],
            None,
        );
        let chunks = collect(
            provider
                .chat(&[Message::user("hi")], &[], "m", Path::new("."))
                .await
                .expect("stream"),
        )
        .await;
        assert_eq!(text_of(&chunks), "hello…done", "secondary answered");
    }

    // Mid-stream error AFTER content → propagate verbatim, no retry.
    #[tokio::test]
    async fn midstream_error_after_content_never_retries() {
        let (primary_tx, primary_rx) = mpsc::channel(8);
        tokio::spawn(async move {
            let _ = primary_tx
                .send(StreamChunk::TextDelta("partial ".into()))
                .await;
            let _ = primary_tx
                .send(StreamChunk::Error(ProviderError::Http {
                    status: 502,
                    body: "upstream dropped".into(),
                }))
                .await;
        });
        let primary = StreamOnceProvider {
            rx: Mutex::new(Some(primary_rx)),
        };
        let secondary = ScriptedProvider::new(vec![]);
        let provider = fallback(
            Arc::new(primary),
            vec![FallbackEntry {
                name: "Secondary".into(),
                provider: secondary.clone(),
            }],
            None,
        );
        let chunks = collect(
            provider
                .chat(&[Message::user("hi")], &[], "m", Path::new("."))
                .await
                .expect("stream"),
        )
        .await;
        assert_eq!(text_of(&chunks), "partial ", "partial output preserved");
        assert!(
            chunks.iter().any(|c| matches!(c, StreamChunk::Error(_))),
            "error must be propagated"
        );
        assert_eq!(secondary.call_count(), 0, "never retry mid-stream");
    }

    // Empty completion (Done with zero content) → failover; exhaustion
    // forwards the empty stream (driver policy stays in charge).
    #[tokio::test]
    async fn empty_completion_fails_over_then_forwards_on_exhaustion() {
        // Secondary also streams empty → chain exhausted → Done forwarded.
        let (primary_tx, primary_rx) = mpsc::channel(8);
        tokio::spawn(async move {
            let _ = primary_tx.send(StreamChunk::Done).await;
        });
        let primary = StreamOnceProvider {
            rx: Mutex::new(Some(primary_rx)),
        };
        let (secondary_tx, secondary_rx) = mpsc::channel(8);
        tokio::spawn(async move {
            let _ = secondary_tx.send(StreamChunk::Done).await;
        });
        let secondary = StreamOnceProvider {
            rx: Mutex::new(Some(secondary_rx)),
        };
        let provider = fallback(
            Arc::new(primary),
            vec![FallbackEntry {
                name: "Secondary".into(),
                provider: Arc::new(secondary),
            }],
            None,
        );
        let chunks = collect(
            provider
                .chat(&[Message::user("hi")], &[], "m", Path::new("."))
                .await
                .expect("stream"),
        )
        .await;
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
        assert!(
            !chunks.iter().any(|c| matches!(c, StreamChunk::Error(_))),
            "exhausted empty completion must not synthesize an error"
        );
    }

    // Multi-hop walk: primary 429 → secondary 5xx → tertiary succeeds; the
    // second notification must name the SECONDARY's failure class, and the
    // final stream comes from the tertiary.
    #[tokio::test]
    async fn walk_updates_notification_reason_to_latest_failure() {
        let primary =
            ScriptedProvider::new(vec![ScriptRound::ChatErr(ProviderError::RateLimited {
                retry_after_ms: 1_000,
            })]);
        let secondary = ScriptedProvider::new(vec![ScriptRound::ChatErr(ProviderError::Http {
            status: 502,
            body: "upstream dropped".into(),
        })]);
        let tertiary = ScriptedProvider::new(vec![]);
        let (event_tx, mut event_rx) = mpsc::channel(16);

        let provider = fallback(
            primary,
            vec![
                FallbackEntry {
                    name: "Secondary".into(),
                    provider: secondary,
                },
                FallbackEntry {
                    name: "Tertiary".into(),
                    provider: tertiary.clone(),
                },
            ],
            Some(event_tx),
        );

        let chunks = collect(
            provider
                .chat(&[Message::user("hi")], &[], "m", Path::new("."))
                .await
                .expect("tertiary stream"),
        )
        .await;
        assert_eq!(text_of(&chunks), "hello…done");
        assert_eq!(tertiary.call_count(), 1);

        let mut notifications = Vec::new();
        while let Ok(event) = event_rx.try_recv() {
            if let AgentEvent::ProviderFallback { from, to, reason } = event {
                notifications.push((from, to, reason));
            }
        }
        assert_eq!(notifications.len(), 2, "one notification per switch");
        assert_eq!(notifications[0].0, "Primary");
        assert_eq!(notifications[0].1, "Secondary");
        assert!(notifications[0].2.starts_with("rate_limited"));
        assert_eq!(notifications[1].0, "Secondary");
        assert_eq!(notifications[1].1, "Tertiary");
        assert!(
            notifications[1].2.starts_with("server_error"),
            "second hop names the secondary's failure: {:?}",
            notifications[1].2
        );
    }

    // Throttle: first switch immediate (initial=0), later switches spaced.
    #[tokio::test(start_paused = true)]
    async fn throttle_spaces_subsequent_switches() {
        let throttle = Throttle::new(0, 500);
        let start = Instant::now();
        throttle.wait_and_mark().await;
        assert!(start.elapsed() >= Duration::from_millis(0));
        throttle.wait_and_mark().await;
        assert!(
            start.elapsed() >= Duration::from_millis(500),
            "second switch must wait out the gap"
        );
    }

    /// Provider whose chat() hands out one pre-built stream receiver.
    /// Used for mid-stream cases where the stream contents are crafted
    /// by the test.
    struct StreamOnceProvider {
        rx: Mutex<Option<mpsc::Receiver<StreamChunk>>>,
    }

    #[async_trait]
    impl Provider for StreamOnceProvider {
        async fn chat(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
            _model: &str,
            _workspace_root: &Path,
        ) -> Result<mpsc::Receiver<StreamChunk>, ProviderError> {
            match self.rx.lock() {
                Ok(mut guard) => match guard.take() {
                    Some(rx) => Ok(rx),
                    None => Err(ProviderError::Other("no scripted stream".into())),
                },
                Err(poisoned) => Err(ProviderError::Other(format!(
                    "script poisoned: {}",
                    poisoned
                ))),
            }
        }
    }

    // ---- Stream-level failover (cross-model verification) ----

    /// Spawn a channel that emits `chunks` in order, then closes.
    fn chunk_stream(chunks: Vec<StreamChunk>) -> mpsc::Receiver<StreamChunk> {
        let (tx, rx) = mpsc::channel(8);
        tokio::spawn(async move {
            for chunk in chunks {
                if tx.send(chunk).await.is_err() {
                    return;
                }
            }
        });
        rx
    }

    /// A provider that hands out one crafted chunk stream and counts how
    /// many times `chat()` was called. Lets a test drive exact stream
    /// shapes (mid-stream errors, partial content) — no network.
    struct CraftedProvider {
        calls: Mutex<usize>,
        rx: Mutex<Option<mpsc::Receiver<StreamChunk>>>,
    }

    impl CraftedProvider {
        fn new(chunks: Vec<StreamChunk>) -> Arc<Self> {
            Arc::new(Self {
                calls: Mutex::new(0),
                rx: Mutex::new(Some(chunk_stream(chunks))),
            })
        }

        fn call_count(&self) -> usize {
            self.calls.lock().map_or(0, |c| *c)
        }
    }

    #[async_trait]
    impl Provider for CraftedProvider {
        async fn chat(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
            _model: &str,
            _workspace_root: &Path,
        ) -> Result<mpsc::Receiver<StreamChunk>, ProviderError> {
            if let Ok(mut calls) = self.calls.lock() {
                *calls += 1;
            }
            match self.rx.lock() {
                Ok(mut guard) => match guard.take() {
                    Some(rx) => Ok(rx),
                    None => Err(ProviderError::Other("no crafted stream".into())),
                },
                Err(poisoned) => Err(ProviderError::Other(format!(
                    "crafted rx poisoned: {poisoned}"
                ))),
            }
        }
    }

    /// Drain every queued `ProviderFallback` as `(from, to, reason)`.
    fn drain_fallbacks(rx: &mut mpsc::Receiver<AgentEvent>) -> Vec<(String, String, String)> {
        let mut out = Vec::new();
        while let Ok(event) = rx.try_recv() {
            if let AgentEvent::ProviderFallback { from, to, reason } = event {
                out.push((from, to, reason));
            }
        }
        out
    }

    // Scenario 1: the first provider fails at STREAM level with zero
    // delivered content → the chain's second provider continues and answers,
    // and the switch is announced with a user-visible ProviderFallback event.
    #[tokio::test]
    async fn stream_zero_content_error_fails_over_and_announces_switch() {
        let primary = CraftedProvider::new(vec![StreamChunk::Error(ProviderError::RequestFailed(
            "connection reset by peer".into(),
        ))]);
        let secondary = ScriptedProvider::new(vec![]);
        let (event_tx, mut event_rx) = mpsc::channel(16);

        let provider = fallback(
            primary.clone(),
            vec![FallbackEntry {
                name: "Secondary".into(),
                provider: secondary.clone(),
            }],
            Some(event_tx),
        );

        let chunks = collect(
            provider
                .chat(&[Message::user("hi")], &[], "primary-model", Path::new("."))
                .await
                .expect("fallback stream"),
        )
        .await;
        // No Finish chunk was ever produced before the failure, so the
        // delivered stream is exactly the secondary's completion.
        assert_eq!(text_of(&chunks), "hello…done", "secondary answered");
        assert_eq!(primary.call_count(), 1);
        assert_eq!(secondary.call_count(), 1);

        let events = drain_fallbacks(&mut event_rx);
        assert_eq!(events.len(), 1, "failover must never be silent");
        assert_eq!(events[0].0, "Primary");
        assert_eq!(events[0].1, "Secondary");
        assert!(
            events[0].2.starts_with("network_error"),
            "reason names the failure class: {:?}",
            events[0].2
        );
    }

    // Scenario 2: a failure AFTER >=1 delivered content chunk is surfaced
    // verbatim — no retry, no second request, no fallback event.
    #[tokio::test]
    async fn midstream_error_after_content_is_verbatim_and_silent() {
        let primary = CraftedProvider::new(vec![
            StreamChunk::TextDelta("partial ".into()),
            StreamChunk::Error(ProviderError::Http {
                status: 502,
                body: "upstream dropped".into(),
            }),
        ]);
        let secondary = ScriptedProvider::new(vec![]);
        let (event_tx, mut event_rx) = mpsc::channel(16);

        let provider = fallback(
            primary.clone(),
            vec![FallbackEntry {
                name: "Secondary".into(),
                provider: secondary.clone(),
            }],
            Some(event_tx),
        );
        let chunks = collect(
            provider
                .chat(&[Message::user("hi")], &[], "m", Path::new("."))
                .await
                .expect("stream"),
        )
        .await;
        assert_eq!(text_of(&chunks), "partial ", "partial output preserved");
        // The original failure surfaces unchanged (status + body intact).
        match chunks.last() {
            Some(StreamChunk::Error(ProviderError::Http { status, body })) => {
                assert_eq!(*status, 502);
                assert_eq!(body, "upstream dropped");
            }
            other => panic!("expected verbatim 502 error, got {other:?}"),
        }
        assert_eq!(primary.call_count(), 1);
        assert_eq!(secondary.call_count(), 0, "never retry mid-stream");
        assert!(
            drain_fallbacks(&mut event_rx).is_empty(),
            "a mid-stream failure must not announce a switch"
        );
    }

    // Scenario 3: a second failure inside the retry window does not switch
    // immediately — the walk waits out `retry_delay_ms` (initial = 0 makes
    // the FIRST switch instant, isolating the gap on the second switch).
    #[tokio::test(start_paused = true)]
    async fn second_switch_waits_out_retry_delay() {
        let primary =
            ScriptedProvider::new(vec![ScriptRound::ChatErr(ProviderError::RateLimited {
                retry_after_ms: 1,
            })]);
        let secondary =
            ScriptedProvider::new(vec![ScriptRound::ChatErr(ProviderError::RateLimited {
                retry_after_ms: 1,
            })]);
        let tertiary = ScriptedProvider::new(vec![]);
        let provider = FallbackProvider::new(
            primary,
            "Primary".into(),
            vec![
                FallbackEntry {
                    name: "Secondary".into(),
                    provider: secondary,
                },
                FallbackEntry {
                    name: "Tertiary".into(),
                    provider: tertiary.clone(),
                },
            ],
            0,   // first switch: instant
            400, // subsequent switches: 400ms apart
            0,   // max_retries: no same-model absorption
            None,
        );

        let start = Instant::now();
        let chunks = collect(
            provider
                .chat(&[Message::user("hi")], &[], "m", Path::new("."))
                .await
                .expect("tertiary stream"),
        )
        .await;
        assert_eq!(text_of(&chunks), "hello…done");
        assert_eq!(tertiary.call_count(), 1);
        assert!(
            start.elapsed() >= Duration::from_millis(400),
            "second switch must wait out retry_delay_ms, elapsed {:?}",
            start.elapsed()
        );
    }

    // Scenario 4: stream-level exhaustion aggregates every hop's reason into
    // one loud error.
    #[tokio::test]
    async fn stream_chain_exhaustion_aggregates_every_hop_reason() {
        let primary = CraftedProvider::new(vec![StreamChunk::Error(ProviderError::RateLimited {
            retry_after_ms: 7,
        })]);
        let secondary = ScriptedProvider::new(vec![ScriptRound::ChatErr(ProviderError::Http {
            status: 503,
            body: "upstream unavailable".into(),
        })]);
        let tertiary = ScriptedProvider::new(vec![ScriptRound::ChatErr(
            ProviderError::RequestFailed("connection timed out".into()),
        )]);
        let provider = fallback(
            primary.clone(),
            vec![
                FallbackEntry {
                    name: "Secondary".into(),
                    provider: secondary,
                },
                FallbackEntry {
                    name: "Tertiary".into(),
                    provider: tertiary,
                },
            ],
            None,
        );

        let chunks = collect(
            provider
                .chat(&[Message::user("hi")], &[], "m", Path::new("."))
                .await
                .expect("stream"),
        )
        .await;
        match chunks.last() {
            Some(StreamChunk::Error(ProviderError::FallbackExhausted { chain, reasons })) => {
                assert_eq!(chain, "Primary → Secondary → Tertiary");
                for hop in ["Primary:", "Secondary:", "Tertiary:"] {
                    assert!(
                        reasons.contains(hop),
                        "aggregate must name {hop} — reasons: {reasons}"
                    );
                }
            }
            other => panic!("expected FallbackExhausted, got {other:?}"),
        }
        assert_eq!(primary.call_count(), 1);
    }

    // Scenario 5: a stream-level 401 is not failover-class — it surfaces
    // verbatim and the chain is never contacted.
    #[tokio::test]
    async fn stream_auth_error_surfaces_verbatim_and_never_touches_chain() {
        let primary = CraftedProvider::new(vec![StreamChunk::Error(ProviderError::AuthError(
            "invalid api key".into(),
        ))]);
        let secondary = ScriptedProvider::new(vec![]);
        let (event_tx, mut event_rx) = mpsc::channel(16);
        let provider = fallback(
            primary.clone(),
            vec![FallbackEntry {
                name: "Secondary".into(),
                provider: secondary.clone(),
            }],
            Some(event_tx),
        );

        let chunks = collect(
            provider
                .chat(&[Message::user("hi")], &[], "m", Path::new("."))
                .await
                .expect("stream"),
        )
        .await;
        match chunks.last() {
            Some(StreamChunk::Error(ProviderError::AuthError(msg))) => {
                assert_eq!(msg, "invalid api key");
            }
            other => panic!("expected verbatim AuthError, got {other:?}"),
        }
        assert_eq!(primary.call_count(), 1);
        assert_eq!(secondary.call_count(), 0, "auth error must not fail over");
        assert!(
            drain_fallbacks(&mut event_rx).is_empty(),
            "chain untouched → no switch event"
        );
    }

    // ---- T1-2: retry budget, permanent fast-fail, exhaustion, escapes ----

    #[test]
    fn permanent_usage_error_classification() {
        // Typed status wins.
        assert!(is_permanent_usage_error(&ProviderError::Http {
            status: 402,
            body: "payment required".into(),
        }));
        // Body/message markers, case-insensitive.
        assert!(is_permanent_usage_error(&ProviderError::Http {
            status: 400,
            body: "Insufficient Balance".into(),
        }));
        assert!(is_permanent_usage_error(&ProviderError::RequestFailed(
            "billing: card declined".into(),
        )));
        assert!(is_permanent_usage_error(&ProviderError::RequestFailed(
            "coding plan package has expired".into(),
        )));
        assert!(is_permanent_usage_error(&ProviderError::RequestFailed(
            "quota exhausted — permanently exhausted".into(),
        )));
        // Ordinary transient classes stay absorbable/failoverable.
        assert!(!is_permanent_usage_error(&ProviderError::RateLimited {
            retry_after_ms: 1_000
        }));
        assert!(!is_permanent_usage_error(&ProviderError::Http {
            status: 503,
            body: "upstream unavailable".into(),
        }));
        assert!(!is_permanent_usage_error(&ProviderError::RequestFailed(
            "rate limit exceeded, retry later".into(),
        )));
        assert!(!is_permanent_usage_error(&ProviderError::AuthError(
            "invalid api key".into()
        )));
    }

    #[test]
    fn escaped_provider_key_parses_display_and_pinned_names() {
        assert_eq!(escaped_provider_key("DeepSeek"), Some("DeepSeek"));
        assert_eq!(
            escaped_provider_key("ZhipuAI(glm-5.3-flash)"),
            Some("ZhipuAI")
        );
        assert_eq!(escaped_provider_key("MiMo"), Some("MiMo"));
        assert_eq!(
            escaped_provider_key("Primary"),
            None,
            "test doubles parse to none"
        );
        assert_eq!(escaped_provider_key(""), None);
    }

    // Budget = 2: the primary absorbs failures 1 and 2 (redialing itself),
    // failure 3 advances. max_retries = N absorbs N failures.
    #[tokio::test]
    async fn budget_absorbs_failures_then_advances() {
        let primary = ScriptedProvider::new(vec![
            ScriptRound::ChatErr(ProviderError::RateLimited { retry_after_ms: 1 }),
            ScriptRound::ChatErr(ProviderError::RateLimited { retry_after_ms: 1 }),
            ScriptRound::ChatErr(ProviderError::RateLimited { retry_after_ms: 1 }),
        ]);
        let secondary = ScriptedProvider::new(vec![]);
        let provider = FallbackProvider::new(
            primary.clone(),
            "Primary".into(),
            vec![FallbackEntry {
                name: "Secondary".into(),
                provider: secondary.clone(),
            }],
            0,
            0,
            2,
            None,
        );
        let chunks = collect(
            provider
                .chat(&[Message::user("hi")], &[], "m", Path::new("."))
                .await
                .expect("secondary stream"),
        )
        .await;
        assert_eq!(text_of(&chunks), "hello…done");
        assert_eq!(primary.call_count(), 3, "1 initial + 2 absorbed redials");
        assert_eq!(secondary.call_count(), 1, "only the 3rd failure advances");
    }

    // The budget is CHAIN-GLOBAL: after it is spent switching away from the
    // primary, a failure on the SECONDARY advances immediately — it does not
    // get its own N absorptions (omos: "not cleared on a model switch").
    #[tokio::test]
    async fn budget_is_chain_global_not_reset_per_model() {
        let primary = ScriptedProvider::new(vec![
            ScriptRound::ChatErr(ProviderError::RateLimited { retry_after_ms: 1 }),
            ScriptRound::ChatErr(ProviderError::RateLimited { retry_after_ms: 1 }),
            ScriptRound::ChatErr(ProviderError::RateLimited { retry_after_ms: 1 }),
        ]);
        let secondary = ScriptedProvider::new(vec![ScriptRound::ChatErr(ProviderError::Http {
            status: 503,
            body: "upstream unavailable".into(),
        })]);
        let tertiary = ScriptedProvider::new(vec![]);
        let provider = FallbackProvider::new(
            primary,
            "Primary".into(),
            vec![
                FallbackEntry {
                    name: "Secondary".into(),
                    provider: secondary.clone(),
                },
                FallbackEntry {
                    name: "Tertiary".into(),
                    provider: tertiary.clone(),
                },
            ],
            0,
            0,
            2,
            None,
        );
        let chunks = collect(
            provider
                .chat(&[Message::user("hi")], &[], "m", Path::new("."))
                .await
                .expect("tertiary stream"),
        )
        .await;
        assert_eq!(text_of(&chunks), "hello…done");
        assert_eq!(secondary.call_count(), 1, "no fresh budget after a switch");
        assert_eq!(tertiary.call_count(), 1);
    }

    // Permanent usage/quota failures skip absorption entirely: with budget 2
    // the 429 is absorbed once, the billing failure advances immediately.
    #[tokio::test]
    async fn permanent_usage_error_advances_without_absorption() {
        let primary = ScriptedProvider::new(vec![
            ScriptRound::ChatErr(ProviderError::RateLimited { retry_after_ms: 1 }),
            ScriptRound::ChatErr(ProviderError::RequestFailed(
                "402 Payment Required: insufficient balance".into(),
            )),
        ]);
        let secondary = ScriptedProvider::new(vec![]);
        let provider = FallbackProvider::new(
            primary.clone(),
            "Primary".into(),
            vec![FallbackEntry {
                name: "Secondary".into(),
                provider: secondary.clone(),
            }],
            0,
            0,
            2,
            None,
        );
        let chunks = collect(
            provider
                .chat(&[Message::user("hi")], &[], "m", Path::new("."))
                .await
                .expect("secondary stream"),
        )
        .await;
        assert_eq!(text_of(&chunks), "hello…done");
        assert_eq!(
            primary.call_count(),
            2,
            "429 absorbed once; the permanent failure must not redial"
        );
        assert_eq!(secondary.call_count(), 1);
    }

    // A spent chain caches its aggregate error: the next chat() on the same
    // instance returns it without touching any provider again.
    #[tokio::test]
    async fn exhausted_cache_prevents_chain_replay() {
        let primary =
            ScriptedProvider::new(vec![ScriptRound::ChatErr(ProviderError::RateLimited {
                retry_after_ms: 1,
            })]);
        let secondary = ScriptedProvider::new(vec![ScriptRound::ChatErr(ProviderError::Http {
            status: 503,
            body: "upstream unavailable".into(),
        })]);
        let provider = FallbackProvider::new(
            primary.clone(),
            "Primary".into(),
            vec![FallbackEntry {
                name: "Secondary".into(),
                provider: secondary.clone(),
            }],
            0,
            0,
            0,
            None,
        );
        let first = provider
            .chat(&[Message::user("hi")], &[], "m", Path::new("."))
            .await
            .expect_err("chain exhausts");
        assert!(matches!(first, ProviderError::FallbackExhausted { .. }));
        assert_eq!(primary.call_count(), 1);
        assert_eq!(secondary.call_count(), 1);

        let second = provider
            .chat(&[Message::user("again")], &[], "m", Path::new("."))
            .await
            .expect_err("cached exhaustion returns the same verdict");
        match (&first, &second) {
            (
                ProviderError::FallbackExhausted { chain: c1, .. },
                ProviderError::FallbackExhausted { chain: c2, .. },
            ) => assert_eq!(c1, c2),
            other => panic!("expected cached FallbackExhausted, got {other:?}"),
        }
        assert_eq!(primary.call_count(), 1, "exhausted chat() must not redial");
        assert_eq!(
            secondary.call_count(),
            1,
            "exhausted chat() must not replay the chain"
        );
    }

    // A new external turn (reset_for_new_turn) re-arms the budget AND clears
    // the exhaustion cache: the walk replays from the primary with fresh
    // absorption. The escaped-provider set survives the reset.
    #[tokio::test]
    async fn new_turn_reset_restores_absorption() {
        let primary = ScriptedProvider::new(vec![
            ScriptRound::ChatErr(ProviderError::RateLimited { retry_after_ms: 1 }),
            ScriptRound::ChatErr(ProviderError::RateLimited { retry_after_ms: 1 }),
            ScriptRound::ChatErr(ProviderError::RateLimited { retry_after_ms: 1 }),
            // Second descent after the reset: 3 more failures.
            ScriptRound::ChatErr(ProviderError::RateLimited { retry_after_ms: 1 }),
            ScriptRound::ChatErr(ProviderError::RateLimited { retry_after_ms: 1 }),
            ScriptRound::ChatErr(ProviderError::RateLimited { retry_after_ms: 1 }),
        ]);
        let secondary = ScriptedProvider::new(vec![
            ScriptRound::ChatErr(ProviderError::Http {
                status: 503,
                body: "upstream unavailable".into(),
            }),
            ScriptRound::ChatErr(ProviderError::Http {
                status: 503,
                body: "upstream unavailable".into(),
            }),
        ]);
        let provider = FallbackProvider::new(
            primary.clone(),
            "DeepSeek".into(),
            vec![FallbackEntry {
                name: "Kimi".into(),
                provider: secondary.clone(),
            }],
            0,
            0,
            2,
            None,
        );
        let first = provider
            .chat(&[Message::user("hi")], &[], "m", Path::new("."))
            .await;
        assert!(matches!(
            first,
            Err(ProviderError::FallbackExhausted { .. })
        ));
        assert!(primary.call_count() == 3);
        assert_eq!(secondary.call_count(), 1);
        // Kimi failed terminally (chain exhausted) but no switch ever
        // ADVANCED AWAY from it, so it never appears as a `from` — the
        // escaped set mirrors the ProviderFallback events exactly.
        assert_eq!(
            provider.escaped_providers(),
            ["DeepSeek".to_string()].into_iter().collect(),
            "escaped set == the from-values of emitted switches"
        );

        provider.reset_for_new_turn();

        let second = provider
            .chat(&[Message::user("new turn")], &[], "m", Path::new("."))
            .await;
        assert!(matches!(
            second,
            Err(ProviderError::FallbackExhausted { .. })
        ));
        assert_eq!(
            primary.call_count(),
            6,
            "budget re-armed: 3 more absorbed redials"
        );
        assert_eq!(secondary.call_count(), 2);
        assert_eq!(
            provider.escaped_providers(),
            ["DeepSeek".to_string()].into_iter().collect(),
            "escapes survive the turn reset by design"
        );
    }

    // Mid-stream failures never retry — not even same-model absorption: the
    // zero-content rule gates everything.
    #[tokio::test]
    async fn budget_never_absorbs_a_midstream_failure() {
        let primary = CraftedProvider::new(vec![
            StreamChunk::TextDelta("partial ".into()),
            StreamChunk::Error(ProviderError::Http {
                status: 502,
                body: "upstream dropped".into(),
            }),
        ]);
        let secondary = ScriptedProvider::new(vec![]);
        let provider = FallbackProvider::new(
            primary.clone(),
            "Primary".into(),
            vec![FallbackEntry {
                name: "Secondary".into(),
                provider: secondary.clone(),
            }],
            0,
            0,
            3,
            None,
        );
        let chunks = collect(
            provider
                .chat(&[Message::user("hi")], &[], "m", Path::new("."))
                .await
                .expect("stream"),
        )
        .await;
        assert_eq!(text_of(&chunks), "partial ");
        assert_eq!(
            primary.call_count(),
            1,
            "no same-model redial after content"
        );
        assert_eq!(secondary.call_count(), 0, "no switch after content");
    }

    // Empty completions are the driver-policy class: they advance the chain
    // without charging the budget (no same-model redial), and exhaustion
    // forwards Done verbatim — no exhausted-cache record, no synthesized
    // error.
    #[tokio::test]
    async fn empty_completion_advances_without_absorbing() {
        let primary = ScriptedProvider::new(vec![
            ScriptRound::Stream(&[]),
            ScriptRound::Stream(&["never"]),
        ]);
        let secondary = ScriptedProvider::new(vec![ScriptRound::Stream(&[])]);
        let provider = FallbackProvider::new(
            primary.clone(),
            "Primary".into(),
            vec![FallbackEntry {
                name: "Secondary".into(),
                provider: secondary.clone(),
            }],
            0,
            0,
            2,
            None,
        );
        let chunks = collect(
            provider
                .chat(&[Message::user("hi")], &[], "m", Path::new("."))
                .await
                .expect("stream"),
        )
        .await;
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
        assert!(!chunks.iter().any(|c| matches!(c, StreamChunk::Error(_))));
        assert_eq!(
            primary.call_count(),
            1,
            "empty completions never redial the same model"
        );
        assert_eq!(secondary.call_count(), 1);
    }

    // Every switch records the provider it advanced away from — including
    // the primary and pinned-model tier names (canonical display name).
    #[tokio::test]
    async fn switch_records_escaped_providers() {
        let primary =
            ScriptedProvider::new(vec![ScriptRound::ChatErr(ProviderError::RateLimited {
                retry_after_ms: 1,
            })]);
        let secondary = ScriptedProvider::new(vec![]);
        let (event_tx, mut event_rx) = mpsc::channel(16);
        let provider = FallbackProvider::new(
            primary,
            "ZhipuAI(glm-5.3-flash)".into(),
            vec![FallbackEntry {
                name: "Kimi".into(),
                provider: secondary,
            }],
            0,
            0,
            0,
            Some(event_tx),
        );
        let chunks = collect(
            provider
                .chat(&[Message::user("hi")], &[], "m", Path::new("."))
                .await
                .expect("secondary stream"),
        )
        .await;
        assert_eq!(text_of(&chunks), "hello…done");
        let events = drain_fallbacks(&mut event_rx);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, "ZhipuAI(glm-5.3-flash)");
        assert_eq!(
            provider.escaped_providers(),
            ["ZhipuAI".to_string()].into_iter().collect(),
            "pinned tier names canonicalize to the provider"
        );
    }

    // ---- Wire-policy de-pollution across a failover switch ----
    //
    // The production incident shape: primary = GLM (Strip — reasoning is
    // output-only there), fallback = DeepSeek (ToolsGated — reasoning MUST
    // be passed back for tools requests, else the recurring 400: "The
    // `reasoning_content` in the thinking mode must be passed back to the
    // API."). Historically the primary's prepare stripped history IN PLACE,
    // so the DeepSeek failover target received already-polluted messages and
    // reproduced the 400 on a different provider. Since the keep/strip
    // decision moved to per-provider wire serialization (and prepare became
    // a no-op), each chain entry serializes the SAME original history under
    // its OWN policy. This test pins that end-to-end against real
    // OpenAiCompatProviders built by the factory over mock HTTP servers.
    #[tokio::test]
    async fn fallback_switch_serializes_under_target_policy_not_primary() {
        use crate::provider::factory::build_provider_for;
        use crate::provider::test_support::spawn_sse_server_with_body;
        use nca_common::config::ProviderKind;

        let sse_ok = || {
            concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0,\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n",
                "data: [DONE]\n\n"
            )
            .to_string()
        };

        // Primary (ZhipuAI/GLM, Strip): 429 on the first request so the
        // chain advances. Any request it does send must be reasoning-free.
        let glm_body = String::from("{\"error\":{\"message\":\"rate limited\"}}");
        let glm_url = spawn_sse_server_with_body(glm_body, 429, |_request, request_body| {
            assert!(
                !request_body.contains("reasoning_content"),
                "GLM (Strip) request must not echo reasoning_content: {request_body}"
            );
        });
        // Target (DeepSeek, ToolsGated): assert ON the request body that the
        // assistant reasoning_content IS present — the pass-back the tools
        // contract demands.
        let deepseek_url = spawn_sse_server_with_body(sse_ok(), 200, |_request, request_body| {
            let parsed: serde_json::Value =
                serde_json::from_str(request_body).expect("deepseek body is JSON");
            let assistants: Vec<&serde_json::Value> = parsed["messages"]
                .as_array()
                .expect("messages")
                .iter()
                .filter(|m| m["role"] == "assistant")
                .collect();
            assert!(!assistants.is_empty());
            assert!(
                assistants.iter().all(|m| m
                    .get("reasoning_content")
                    .is_some_and(|r| r.as_str() == Some("survivor reasoning"))),
                "DeepSeek failover target must receive reasoning_content under its own ToolsGated policy \
                 (omitting it is the 400: 'The `reasoning_content` in the thinking mode must be passed back to the API.'): {request_body}"
            );
        });

        let mut config = nca_common::config::NcaConfig::default();
        config.provider.zhipuai.api_key = Some("glm-key".into());
        config.provider.zhipuai.base_url = glm_url;
        config.provider.zhipuai.model = "glm-5.3".into();
        config.provider.deepseek.api_key = Some("deepseek-key".into());
        config.provider.deepseek.base_url = deepseek_url;
        config.provider.deepseek.model = "deepseek-flash".into();

        let glm = build_provider_for(&config, ProviderKind::ZhipuAI).expect("glm provider");
        let deepseek =
            build_provider_for(&config, ProviderKind::DeepSeek).expect("deepseek provider");

        let provider = FallbackProvider::new(
            glm,
            "ZhipuAI(glm-5.3)".into(),
            vec![FallbackEntry {
                name: "DeepSeek(deepseek-flash)".into(),
                provider: deepseek,
            }],
            0, // instant switches: keep the test fast
            0,
            0,
            None,
        );

        // Mirror the driver contract: history still carrying reasoning (the
        // session source of truth), a tools-bearing request.
        let mut assistant =
            Message::assistant("prior turn").with_reasoning("survivor reasoning".into());
        assistant.tool_calls = Some(vec![nca_common::message::MessageToolCall {
            id: "call_1".into(),
            name: "lookup".into(),
            arguments: serde_json::json!({}),
        }]);
        let history = vec![
            Message::user("hello"),
            assistant,
            Message::tool("call_1", "result"),
            Message::user("continue"),
        ];
        // The driver calls prepare on a CLONE now, but even the legacy
        // in-place call through the wrapper must not pollute history:
        let mut prepared = history.clone();
        provider
            .prepare_messages_for_request(&mut prepared, Path::new("."))
            .await
            .expect("prepare");
        assert_eq!(
            prepared, history,
            "prepare must not strip history ahead of the chain"
        );

        let tools = vec![ToolDefinition {
            timeout_ms: None,
            name: "lookup".into(),
            description: "Lookup a path".into(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
        }];
        let chunks = collect(
            provider
                .chat(&history, &tools, "glm-5.3", Path::new("."))
                .await
                .expect("deepseek failover stream"),
        )
        .await;
        assert_eq!(text_of(&chunks), "ok");
    }
}
