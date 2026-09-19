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

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use nca_common::event::AgentEvent;
use nca_common::message::Message;
use nca_common::tool::ToolDefinition;
use tokio::sync::mpsc;
use tokio::time::Instant;

use super::openai_compat::truncate_bytes_safe;
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
    event_tx: Option<mpsc::Sender<AgentEvent>>,
}

impl FallbackProvider {
    /// Wrap `primary` with `fallbacks` (in order). Notifications are sent on
    /// `event_tx` when present; without one they degrade to `tracing` only.
    pub fn new(
        primary: Arc<dyn Provider>,
        primary_name: String,
        fallbacks: Vec<FallbackEntry>,
        initial_retry_delay_ms: u64,
        retry_delay_ms: u64,
        event_tx: Option<mpsc::Sender<AgentEvent>>,
    ) -> Self {
        Self {
            primary,
            primary_name,
            fallbacks,
            throttle: Throttle::new(initial_retry_delay_ms, retry_delay_ms),
            event_tx,
        }
    }

    /// Display names of the fallback chain, in order (diagnostics/tests).
    pub fn chain_names(&self) -> Vec<&str> {
        self.fallbacks.iter().map(|e| e.name.as_str()).collect()
    }

    /// Spawn the stream forwarder: forwards chunks from `current`
    /// downstream, guarding the zero-content rule, and — while no content
    /// has been delivered — fails over to `remaining` on failover-class
    /// errors and empty completions.
    #[allow(clippy::too_many_arguments)]
    fn spawn_forwarder(
        current: mpsc::Receiver<StreamChunk>,
        remaining: Vec<FallbackEntry>,
        messages: Vec<Message>,
        tools: Vec<ToolDefinition>,
        workspace_root: PathBuf,
        attempted: Vec<String>,
        failures: Vec<String>,
        throttle: Throttle,
        event_tx: Option<mpsc::Sender<AgentEvent>>,
    ) -> mpsc::Receiver<StreamChunk> {
        let (tx, rx) = mpsc::channel(64);
        tokio::spawn(async move {
            let mut current = current;
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
                let Some(chunk) = current.recv().await else {
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
                        // Empty completion: fail over; if the chain is
                        // exhausted, forward the empty stream verbatim so
                        // the driver's empty-response policy fails loudly.
                        match advance(
                            &mut remaining,
                            &messages,
                            &tools,
                            &workspace_root,
                            "empty completion".to_string(),
                            &mut attempted,
                            &mut failures,
                            &throttle,
                            &event_tx,
                        )
                        .await
                        {
                            AdvanceOutcome::Stream(next) => {
                                current = next;
                                // Drop the failed generation's held Finish:
                                // its completion never happened.
                                held_finish = None;
                            }
                            AdvanceOutcome::Exhausted => {
                                flush_finish(&tx, &mut held_finish).await;
                                let _ = tx.send(StreamChunk::Done).await;
                                return;
                            }
                            AdvanceOutcome::Fatal(err) => {
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
                        let reason = format!(
                            "{}: {}",
                            reason_label(&err),
                            truncate_bytes_safe(&err.to_string(), 160)
                        );
                        let from = attempted.last().cloned().unwrap_or_default();
                        failures.push(fail_line(&from, &err));
                        match advance(
                            &mut remaining,
                            &messages,
                            &tools,
                            &workspace_root,
                            reason,
                            &mut attempted,
                            &mut failures,
                            &throttle,
                            &event_tx,
                        )
                        .await
                        {
                            AdvanceOutcome::Stream(next) => {
                                current = next;
                                // Drop the failed generation's held Finish:
                                // its completion never happened.
                                held_finish = None;
                            }
                            AdvanceOutcome::Exhausted => {
                                flush_finish(&tx, &mut held_finish).await;
                                let _ = tx
                                    .send(StreamChunk::Error(aggregate(&attempted, &failures)))
                                    .await;
                                return;
                            }
                            AdvanceOutcome::Fatal(fatal) => {
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

/// Outcome of trying to obtain a stream from the next provider(s).
enum AdvanceOutcome {
    /// A provider produced a stream: continue forwarding from it.
    Stream(mpsc::Receiver<StreamChunk>),
    /// The chain ran out of providers.
    Exhausted,
    /// A provider failed non-failover-class: surface its error verbatim.
    Fatal(ProviderError),
}

/// Walk `remaining` until a stream is obtained. Each switch is throttled and
/// announced (event + log). Failover-class chat() errors continue the walk;
/// non-failover errors stop it verbatim.
#[allow(clippy::too_many_arguments)]
async fn advance(
    remaining: &mut VecDeque<FallbackEntry>,
    messages: &[Message],
    tools: &[ToolDefinition],
    workspace_root: &Path,
    mut reason: String,
    attempted: &mut Vec<String>,
    failures: &mut Vec<String>,
    throttle: &Throttle,
    event_tx: &Option<mpsc::Sender<AgentEvent>>,
) -> AdvanceOutcome {
    loop {
        let Some(entry) = remaining.pop_front() else {
            return AdvanceOutcome::Exhausted;
        };
        throttle.wait_and_mark().await;
        let from = attempted.last().cloned().unwrap_or_default();
        notify_fallback(event_tx, &from, &entry.name, &reason).await;
        // Empty model string: each fallback provider uses its own
        // `[provider.<name>]` configured model.
        match entry
            .provider
            .chat(messages, tools, "", workspace_root)
            .await
        {
            Ok(stream) => {
                attempted.push(entry.name.clone());
                return AdvanceOutcome::Stream(stream);
            }
            Err(err) if is_failoverable(&err) => {
                attempted.push(entry.name.clone());
                failures.push(fail_line(&entry.name, &err));
                // The next notification (if the walk continues) names THIS
                // failure, not the one that started the walk.
                reason = format!(
                    "{}: {}",
                    reason_label(&err),
                    truncate_bytes_safe(&err.to_string(), 160)
                );
                continue;
            }
            Err(err) => return AdvanceOutcome::Fatal(err),
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
        let mut attempted = vec![self.primary_name.clone()];
        let mut failures: Vec<String> = Vec::new();

        let current = match self
            .primary
            .chat(messages, tools, model, workspace_root)
            .await
        {
            Ok(stream) => stream,
            Err(err) => {
                if !is_failoverable(&err) {
                    return Err(err);
                }
                failures.push(fail_line(&self.primary_name, &err));
                let reason = format!(
                    "{}: {}",
                    reason_label(&err),
                    truncate_bytes_safe(&err.to_string(), 160)
                );
                let mut remaining: VecDeque<FallbackEntry> =
                    self.fallbacks.iter().cloned().collect();
                match advance(
                    &mut remaining,
                    messages,
                    tools,
                    workspace_root,
                    reason,
                    &mut attempted,
                    &mut failures,
                    &self.throttle,
                    &self.event_tx,
                )
                .await
                {
                    AdvanceOutcome::Stream(stream) => stream,
                    AdvanceOutcome::Exhausted => {
                        return Err(aggregate(&attempted, &failures));
                    }
                    AdvanceOutcome::Fatal(err) => return Err(err),
                }
            }
        };

        let remaining: Vec<FallbackEntry> = self.fallbacks.to_vec();
        Ok(Self::spawn_forwarder(
            current,
            remaining,
            messages.to_vec(),
            tools.to_vec(),
            workspace_root.to_path_buf(),
            attempted,
            failures,
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
}
