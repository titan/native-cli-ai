pub mod anthropic;
pub mod anthropic_compat;
pub mod custom;
pub mod deepseek;
pub mod factory;
pub mod fallback;
pub mod kimi;
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
    /// Rewrite conversation history before an HTTP request (e.g. MiniMax `coding_plan/vlm` for images).
    /// Default: no-op.
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
        assert!(
            !ProviderError::RateLimited {
                retry_after_ms: 1000
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
}
