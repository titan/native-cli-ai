pub mod anthropic;
pub mod anthropic_compat;
#[cfg(test)]
pub(crate) mod conformance;
pub mod custom;
pub mod deepseek;
pub mod factory;
pub mod fallback;
pub mod kimi;
pub mod mimo;
pub mod minimax;
pub mod minimax_vlm;
pub mod openai;
pub mod openai_compat;
pub mod openrouter;
#[cfg(test)]
pub mod test_support;
pub mod validate;
pub mod zhipuai;

use crate::cache_keepalive::{KeepaliveSnapshot, PingUsage};
use std::path::Path;

use async_trait::async_trait;
use nca_common::message::Message;
use nca_common::tool::{ToolCall, ToolDefinition};

/// Parse an HTTP `Retry-After` response header into milliseconds.
///
/// Per RFC 7231 §7.1.3 the header carries either delta-seconds or an
/// HTTP-date (the time at which the client may retry). Returns `None` when
/// the header is absent or unparseable — callers fall back to a fixed
/// delay. A date already in the past clamps to `0` (retry immediately).
pub fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    let raw = headers
        .get("Retry-After")?
        .to_str()
        .ok()?
        .trim()
        .to_string();
    if raw.is_empty() {
        return None;
    }
    // Form 1: delta-seconds (integer). Some gateways emit a decimal value;
    // the fractional part is sub-second noise for a retry hint and is
    // truncated rather than failing the parse.
    if let Ok(seconds) = raw.parse::<u64>() {
        return Some(seconds.saturating_mul(1000));
    }
    if let Some((int_part, frac)) = raw.split_once('.')
        && !int_part.is_empty()
        && int_part.bytes().all(|b| b.is_ascii_digit())
        && !frac.is_empty()
        && frac.bytes().all(|b| b.is_ascii_digit())
        && let Ok(seconds) = int_part.parse::<u64>()
    {
        return Some(seconds.saturating_mul(1000));
    }
    // Form 2: HTTP-date — retry at the named instant. RFC 7231 §7.1.1.1
    // lists three formats: IMF-fixdate (rfc2822-shaped), RFC 850, and
    // asctime. chrono has a dedicated parser for the first; the other two
    // are strptime formats (rfc850's 2-digit year maps to %y). All are
    // interpreted as UTC — HTTP-date mandates GMT.
    if let Ok(target) = chrono::DateTime::parse_from_rfc2822(&raw) {
        return Some(elapsed_ms_until(&target.with_timezone(&chrono::Utc)));
    }
    for format in ["%A, %d-%b-%y %H:%M:%S GMT", "%a %b %e %H:%M:%S %Y"] {
        if let Ok(target) = chrono::NaiveDateTime::parse_from_str(&raw, format) {
            return Some(elapsed_ms_until(&target.and_utc()));
        }
    }
    None
}

/// Milliseconds from now until `target`, clamped at zero (a date already in
/// the past means the server allows an immediate retry).
fn elapsed_ms_until(target: &chrono::DateTime<chrono::Utc>) -> u64 {
    target
        .signed_duration_since(chrono::Utc::now())
        .num_milliseconds()
        .max(0) as u64
}

/// Format an error's full source chain as "msg → source → …" for diagnostics.
///
/// The top-level `Display` of reqwest/hyper errors is often too vague (e.g.
/// "error decoding response body"); the real root cause lives in the source
/// chain.
pub(crate) fn format_error_chain(err: &dyn std::error::Error) -> String {
    let mut parts = vec![err.to_string()];
    let mut current = err.source();
    while let Some(source) = current {
        let msg = source.to_string();
        // Skip duplicate messages that add no information.
        if msg != *parts.last().unwrap() {
            parts.push(msg);
        }
        current = source.source();
    }
    parts.join(" → ")
}

/// Byte-stream error payload consumed by the SSE compat core loops.
///
/// Captures everything the stream parsers log and propagate about a
/// transport-level body-stream failure — the `Display` message, the full
/// source chain, and reqwest's transport classification flags — decoupled
/// from `reqwest` so the "bytes → StreamChunk" core loops in
/// [`openai_compat`](self::openai_compat) / [`anthropic_compat`](self::anthropic_compat)
/// can be unit-tested against plain in-memory byte streams without real HTTP.
#[derive(Debug, Clone)]
pub(crate) struct ByteStreamError {
    /// `Display` of the underlying transport error.
    pub display: String,
    /// Full source chain, pre-formatted by [`format_error_chain`].
    pub chain: String,
    pub is_timeout: bool,
    pub is_connect: bool,
    pub is_request: bool,
    pub is_body: bool,
}

impl From<reqwest::Error> for ByteStreamError {
    fn from(err: reqwest::Error) -> Self {
        Self {
            display: err.to_string(),
            chain: format_error_chain(&err),
            is_timeout: err.is_timeout(),
            is_connect: err.is_connect(),
            is_request: err.is_request(),
            is_body: err.is_body(),
        }
    }
}

/// A streamed chunk from the provider.
#[derive(Debug, Clone)]
pub enum StreamChunk {
    TextDelta(String),
    /// Reasoning/thinking content from providers like DeepSeek R1.
    ReasoningDelta(String),
    ToolUse(ToolCall),
    Usage {
        input_tokens: u64,
        output_tokens: u64,
        cache_creation_tokens: u64,
        cache_read_tokens: u64,
    },
    /// A fatal error occurred mid-stream (e.g. connection reset, idle timeout).
    /// The stream terminates after this chunk. Consumers MUST propagate it as an
    /// error rather than treating any buffered text as a successful assistant reply.
    Error(ProviderError),
    /// Terminal finish reason for the generation (e.g. `"stop"`, `"length"`,
    /// `"tool_calls"`), when the provider reports one. Emitted at most once,
    /// just before [`StreamChunk::Done`].
    ///
    /// Lets consumers distinguish truncation from a clean stop. Critical for
    /// thinking models (e.g. ZhipuAI GLM-5.3, whose thinking cannot be disabled):
    /// they can exhaust the entire `max_tokens` budget on `reasoning_content`
    /// and end with an empty `content` and `finish_reason: "length"` — a
    /// deterministic outcome that retrying with identical parameters cannot fix.
    Finish {
        reason: String,
    },
    Done,
}

/// Abstraction over LLM providers (Anthropic, OpenAI, Gemini, etc.).
#[async_trait]
pub trait Provider: Send + Sync {
    /// Request-time message normalization on a request-scoped view (e.g.
    /// MiniMax `coding_plan/vlm` for images). MUST NOT destructively rewrite
    /// session history — callers pass a clone; the session store retains
    /// `reasoning_content` so wire-policy fixes can self-heal. Default: no-op.
    async fn prepare_messages_for_request(
        &self,
        _messages: &mut Vec<Message>,
        _workspace_root: &Path,
    ) -> Result<(), ProviderError> {
        Ok(())
    }

    /// Send a conversation and receive a streaming response.
    ///
    /// `workspace_root` is used to resolve on-disk image paths embedded in user messages.
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        model: &str,
        workspace_root: &Path,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError>;
    /// Send the snapshot prefix with `max_tokens=1` to refresh the provider's
    /// prompt cache during a tool-execution pause. Returns observed usage for
    /// cost tracking.
    ///
    /// The default implementation returns an error — providers that support
    /// keepalive override this with a direct HTTP request at minimal cost.
    async fn keepalive_ping(
        &self,
        _snapshot: &KeepaliveSnapshot,
    ) -> Result<PingUsage, ProviderError> {
        Err(ProviderError::Other(
            "keepalive_ping not implemented for this provider".into(),
        ))
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum ProviderError {
    #[error("provider configuration error: {0}")]
    Configuration(String),
    #[error("API request failed: {0}")]
    RequestFailed(String),
    /// A non-success HTTP response not covered by a more specific variant
    /// (5xx, most 4xx). Carries the numeric status so failover/retry
    /// policies can classify without parsing the body.
    #[error("provider HTTP {status}: {body}")]
    Http { status: u16, body: String },
    #[error("Authentication error: {0}")]
    AuthError(String),
    #[error("Rate limited, retry after {retry_after_ms}ms")]
    RateLimited { retry_after_ms: u64 },
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    /// A provider fallback chain ran out of providers. `reasons` carries one
    /// line per attempted provider so the root causes stay visible.
    #[error("fallback chain exhausted [{chain}]: {reasons}")]
    FallbackExhausted { chain: String, reasons: String },
    #[error("{0}")]
    Other(String),
}

/// Error-body substrings that identify a context-window overflow rejection
/// (HTTP 4xx raised before streaming started). Deliberately tight — a false
/// positive triggers destructive compaction recovery.
///
/// Coverage is intentionally partial (`docs/plans/p3-compaction-design.md`
/// §2): providers phrasing overflow differently (e.g. non-English bodies)
/// will not match and the error passes through unchanged (graceful
/// degradation). Patterns are extended only with verified literal strings.
const CONTEXT_OVERFLOW_PATTERNS: [&str; 4] = [
    // OpenAI / OpenAI-compatible (incl. DeepSeek, served by OpenAiCompatProvider)
    "maximum context length",
    // OpenAI-compatible alt phrasing
    "context length exceeded",
    // Anthropic-compatible (incl. MiniMax, Kimi)
    "prompt is too long",
    // Anthropic alt phrasing ("input length and `max_tokens` exceed context limit")
    "exceed context limit",
];

impl ProviderError {
    /// Whether this error is a context-window overflow rejection (HTTP 4xx
    /// raised before streaming started). Recoverable by compaction + retry.
    ///
    /// Case-insensitive substring match over the payloads of `RequestFailed`,
    /// `Http`, and `Other` — all providers route non-401/403/404/429 HTTP
    /// bodies to `Http { status, body_text }` via the compat stream parsers
    /// (legacy `RequestFailed(body_text)` bodies still match).
    pub fn is_context_overflow(&self) -> bool {
        let payload = match self {
            ProviderError::RequestFailed(body)
            | ProviderError::Http { body, .. }
            | ProviderError::Other(body) => body,
            _ => return false,
        };
        let lower = payload.to_ascii_lowercase();
        CONTEXT_OVERFLOW_PATTERNS
            .iter()
            .any(|pattern| lower.contains(pattern))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_context_overflow_matches_the_four_literal_patterns() {
        for literal in [
            "This model's maximum context length is 65536 tokens",
            "context length exceeded",
            "prompt is too long: 12000 tokens > 8192 maximum",
            "input length and `max_tokens` exceed context limit: 100000 > 65536",
        ] {
            assert!(
                ProviderError::RequestFailed(literal.into()).is_context_overflow(),
                "must match: {literal}"
            );
        }
    }

    #[test]
    fn is_context_overflow_is_case_insensitive_and_matches_other_variant() {
        assert!(
            ProviderError::RequestFailed(
                "This Model's MAXIMUM CONTEXT LENGTH is 65536 tokens".into()
            )
            .is_context_overflow()
        );
        assert!(ProviderError::Other("Prompt Is TOO LONG".into()).is_context_overflow());
    }

    #[test]
    fn is_context_overflow_matches_inside_realistic_json_bodies() {
        let openai_style = r#"{"error":{"message":"This model's maximum context length is 65536 tokens. However, you requested 70124 tokens (68676 in the messages, 1448 in the completion). Please reduce the length of the messages or completion.","type":"invalid_request_error","param":null,"code":"context_length_exceeded"}}"#;
        assert!(ProviderError::RequestFailed(openai_style.into()).is_context_overflow());

        let anthropic_style = r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 200001 tokens > 200000 maximum"}}"#;
        assert!(ProviderError::RequestFailed(anthropic_style.into()).is_context_overflow());
    }

    #[test]
    fn is_context_overflow_matches_http_variant_bodies() {
        // Non-401/403/404/429 HTTP bodies arrive as `Http { status, body }`
        // since fallback classification needs the status; overflow must
        // still be detected on the 400 path.
        assert!(
            ProviderError::Http {
                status: 400,
                body: "This model's maximum context length is 65536 tokens".into()
            }
            .is_context_overflow()
        );
        assert!(
            !ProviderError::Http {
                status: 500,
                body: "internal server error".into()
            }
            .is_context_overflow()
        );
    }

    #[test]
    fn is_context_overflow_false_for_unrelated_errors() {
        assert!(!ProviderError::AuthError("invalid api key".into()).is_context_overflow());
        // Sentinel-independent since D2: the RateLimited backoff hint may be
        // any server-provided value (parse_retry_after), so the assertion
        // must hold for an arbitrary ms — e.g. one a gateway would send.
        assert!(
            !ProviderError::RateLimited {
                retry_after_ms: 65_000
            }
            .is_context_overflow()
        );
        assert!(!ProviderError::ModelNotFound("no such model".into()).is_context_overflow());
        assert!(
            !ProviderError::RequestFailed("internal server error".into()).is_context_overflow()
        );
        assert!(
            !ProviderError::RequestFailed("upstream connect error".into()).is_context_overflow()
        );
        assert!(!ProviderError::Configuration("missing api key".into()).is_context_overflow());
    }

    #[test]
    fn is_context_overflow_false_for_non_matching_overflow_body() {
        // Graceful passthrough (design §2): a provider phrasing overflow in
        // another language does not match any pattern — the error passes
        // through unchanged, which is exactly the pre-P3 behavior.
        let chinese_body = "错误：输入内容超过了模型的最大上下文窗口，请缩短输入";
        assert!(!ProviderError::RequestFailed(chinese_body.into()).is_context_overflow());
    }

    fn header_map(pairs: &[(&str, &str)]) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        for (name, value) in pairs {
            headers.insert(
                reqwest::header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                reqwest::header::HeaderValue::from_str(value).unwrap(),
            );
        }
        headers
    }

    #[test]
    fn parse_retry_after_integer_seconds_scale_to_ms() {
        assert_eq!(
            parse_retry_after(&header_map(&[("Retry-After", "120")])),
            Some(120_000)
        );
        assert_eq!(
            parse_retry_after(&header_map(&[("Retry-After", "0")])),
            Some(0)
        );
        // Some gateways send a decimal delta; the fraction is truncated.
        assert_eq!(
            parse_retry_after(&header_map(&[("Retry-After", "1.5")])),
            Some(1_000)
        );
    }

    #[test]
    fn parse_retry_after_http_date_yields_delta() {
        // Future date (~1 h out) → delta ≈ 3_600_000 ms with clock slop.
        let target = chrono::Utc::now() + chrono::Duration::hours(1);
        for formatted in [
            target.to_rfc2822(),
            target.format("%a %b %e %H:%M:%S %Y").to_string(),
        ] {
            let parsed = parse_retry_after(&header_map(&[("Retry-After", &formatted)]))
                .unwrap_or_else(|| panic!("must parse: {formatted}"));
            let slop = (parsed as i64 - 3_600_000).abs();
            assert!(slop < 5_000, "delta too far from 1 h: {parsed} ms");
        }
    }

    #[test]
    fn parse_retry_after_http_date_rfc850() {
        // RFC 850 format ("%A, %d-%b-%y %H:%M:%S GMT", 2-digit year).
        let target = chrono::Utc::now() + chrono::Duration::minutes(2);
        let rfc850 = target.format("%A, %d-%b-%y %H:%M:%S GMT").to_string();
        let parsed = parse_retry_after(&header_map(&[("Retry-After", &rfc850)]))
            .expect("rfc850 date must parse");
        let slop = (parsed as i64 - 120_000).abs();
        assert!(slop < 5_000, "delta too far from 2 min: {parsed} ms");
    }

    #[test]
    fn parse_retry_after_past_date_clamps_to_zero() {
        let past = (chrono::Utc::now() - chrono::Duration::hours(1)).to_rfc2822();
        assert_eq!(
            parse_retry_after(&header_map(&[("Retry-After", &past)])),
            Some(0)
        );
    }

    #[test]
    fn parse_retry_after_missing_or_garbage_is_none() {
        assert_eq!(parse_retry_after(&header_map(&[])), None);
        assert_eq!(parse_retry_after(&header_map(&[("Retry-After", "")])), None);
        assert_eq!(
            parse_retry_after(&header_map(&[("Retry-After", "soon")])),
            None
        );
        assert_eq!(
            parse_retry_after(&header_map(&[("Content-Type", "text/plain")])),
            None
        );
    }
}
