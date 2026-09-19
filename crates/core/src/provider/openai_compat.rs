use super::{Provider, ProviderError, StreamChunk};
use crate::cache_keepalive::{KeepaliveSnapshot, PingUsage};

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use base64::{Engine, engine::general_purpose::STANDARD as B64};
use futures_util::StreamExt;
use nca_common::message::{ContentPart, Message, MessageContent, Role};
use nca_common::tool::{ToolCall, ToolDefinition};
use serde_json::{Value, json};

/// 单次流式读取的空闲超时。
///
/// 替代 reqwest 的全局总超时：只要持续有 token 流出，任意长的总耗时都不会被掐断；
/// 仅在连接真正静默（无任何字节）超过该阈值时才报错。这对 reasoning/thinking 模型
/// 尤其重要——thinking 阶段总耗时常超过 120s，但 token 间隔很小。
pub(crate) const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(90);

pub fn openai_request_body(
    messages: &[Message],
    tools: &[ToolDefinition],
    model: &str,
    max_tokens: u32,
    temperature: f32,
    workspace_root: &Path,
) -> Result<Value, ProviderError> {
    let mut body = json!({
        "model": model,
        "messages": to_openai_messages(messages, workspace_root)?,
        "stream": true,
        "stream_options": {
            "include_usage": true
        },
        "max_tokens": max_tokens,
        "temperature": temperature,
    });
    // Omit the key entirely when there are no tools: `"tools": null` is
    // rejected by strict OpenAI-compatible gateways (e.g. ZhipuAI 400s the
    // request instead of ignoring the field).
    if !tools.is_empty() {
        body["tools"] = json!(
            tools
                .iter()
                .map(|tool| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        }
                    })
                })
                .collect::<Vec<_>>()
        );
    }
    Ok(body)
}

pub fn spawn_openai_stream(
    response: reqwest::Response,
    provider_name: &'static str,
) -> tokio::sync::mpsc::Receiver<StreamChunk> {
    let mut byte_stream = response.bytes_stream();
    let (tx, rx) = tokio::sync::mpsc::channel(64);

    tokio::spawn(async move {
        let mut buffer = String::new();
        let mut tool_calls: BTreeMap<u64, ToolCallAccumulator> = BTreeMap::new();
        // Last non-null finish_reason seen across chunks (OpenAI-compatible
        // streams repeat it on the final chunk of each choice). Surfaced via
        // `StreamChunk::Finish` so the agent loop can distinguish a truncated
        // generation ("length") from a clean stop.
        let mut finish_reason: Option<String> = None;

        loop {
            // 用单次读取的空闲超时替代 reqwest 全局总超时：只要持续有 token 流出，
            // 整个流可以跑任意长时间；仅在连接静默超过 STREAM_IDLE_TIMEOUT 时报错。
            let item = match tokio::time::timeout(STREAM_IDLE_TIMEOUT, byte_stream.next()).await {
                Ok(Some(result)) => result,
                Ok(None) => break,
                Err(_elapsed) => {
                    let chain = format!(
                        "operation timed out (stream idle for more than {STREAM_IDLE_TIMEOUT:?})"
                    );
                    let buffer_preview = if buffer.is_empty() {
                        String::from("(none)")
                    } else {
                        buffer.chars().take(500).collect()
                    };
                    tracing::error!(
                        provider = provider_name,
                        idle_timeout_secs = STREAM_IDLE_TIMEOUT.as_secs(),
                        buffer_preview = %buffer_preview,
                        "stream_idle_timeout"
                    );
                    let _ = tx
                        .send(StreamChunk::Error(ProviderError::RequestFailed(format!(
                            "{provider_name} stream error: {chain}\nBuffered data before error: {buffer_preview}"
                        ))))
                        .await;
                    return;
                }
            };

            let chunk = match item {
                Ok(chunk) => chunk,
                Err(err) => {
                    // The Display message alone (e.g. "error decoding response
                    // body") is too vague to diagnose intermittent stream
                    // disruptions. Walk the full source chain and include any
                    // buffered response data for diagnostics.
                    let chain = super::format_error_chain(&err);
                    let buffer_preview = if buffer.is_empty() {
                        String::from("(none)")
                    } else {
                        buffer.chars().take(500).collect()
                    };

                    tracing::error!(
                        provider = provider_name,
                        error = %err,
                        error_chain = %chain,
                        is_timeout = err.is_timeout(),
                        is_connect = err.is_connect(),
                        is_request = err.is_request(),
                        is_body = err.is_body(),
                        buffer_preview = %buffer_preview,
                        "stream_byte_error"
                    );

                    let _ = tx
                        .send(StreamChunk::Error(ProviderError::RequestFailed(format!(
                            "{provider_name} stream error: {chain}\nBuffered data before error: {buffer_preview}"
                        ))))
                        .await;
                    return;
                }
            };

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(nl) = buffer.find('\n') {
                let raw = buffer[..nl].to_string();
                buffer.drain(..=nl);
                let line = raw.trim_end_matches('\r').trim();

                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                if !line.starts_with("data:") {
                    continue;
                }

                let data = line["data:".len()..].trim();
                if data == "[DONE]" {
                    flush_openai_tool_calls(&tx, &mut tool_calls).await;
                    if let Some(reason) = finish_reason.take() {
                        let _ = tx.send(StreamChunk::Finish { reason }).await;
                    }
                    let _ = tx.send(StreamChunk::Done).await;
                    return;
                }

                let Ok(event) = serde_json::from_str::<Value>(data) else {
                    continue;
                };

                if let Some(usage) = event.get("usage") {
                    let input_tokens = usage["prompt_tokens"].as_u64().unwrap_or(0);
                    let output_tokens = usage["completion_tokens"].as_u64().unwrap_or(0);

                    // OpenAI format: prompt_tokens_details.cached_tokens
                    // DeepSeek format: prompt_cache_hit_tokens / prompt_cache_miss_tokens
                    let cached_tokens = usage
                        .get("prompt_tokens_details")
                        .and_then(|d| d.get("cached_tokens"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let (cache_creation_tokens, cache_read_tokens) = if cached_tokens > 0 {
                        // OpenAI style: cached_tokens are hits; misses are input - cached
                        (0, cached_tokens)
                    } else if let Some(miss) = usage
                        .get("prompt_cache_miss_tokens")
                        .and_then(|v| v.as_u64())
                    {
                        // DeepSeek style: explicit hit/miss fields
                        let hit = usage
                            .get("prompt_cache_hit_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        (miss, hit)
                    } else {
                        (0, 0)
                    };

                    if input_tokens > 0 || output_tokens > 0 {
                        let _ = tx
                            .send(StreamChunk::Usage {
                                input_tokens,
                                output_tokens,
                                cache_creation_tokens,
                                cache_read_tokens,
                            })
                            .await;
                    }
                }

                let Some(choices) = event["choices"].as_array() else {
                    continue;
                };

                for choice in choices {
                    let delta = &choice["delta"];
                    if let Some(text) = delta["content"].as_str()
                        && !text.is_empty()
                    {
                        let _ = tx.send(StreamChunk::TextDelta(text.to_string())).await;
                    }

                    if let Some(reasoning) = delta["reasoning_content"].as_str()
                        && !reasoning.is_empty()
                    {
                        let _ = tx
                            .send(StreamChunk::ReasoningDelta(reasoning.to_string()))
                            .await;
                    }

                    if let Some(tool_deltas) = delta["tool_calls"].as_array() {
                        for tool_delta in tool_deltas {
                            let index = tool_delta["index"].as_u64().unwrap_or(0);
                            let entry = tool_calls.entry(index).or_default();
                            if let Some(id) = tool_delta["id"].as_str() {
                                entry.id = id.to_string();
                            }
                            if let Some(name) = tool_delta["function"]["name"].as_str() {
                                entry.name.push_str(name);
                            }
                            if let Some(arguments) = tool_delta["function"]["arguments"].as_str() {
                                entry.arguments.push_str(arguments);
                            }
                        }
                    }

                    if let Some(reason) = choice["finish_reason"].as_str()
                        && !reason.is_empty()
                    {
                        finish_reason = Some(reason.to_string());
                    }

                    if choice["finish_reason"].as_str() == Some("tool_calls") {
                        flush_openai_tool_calls(&tx, &mut tool_calls).await;
                    }
                }
            }
        }

        flush_openai_tool_calls(&tx, &mut tool_calls).await;
        if let Some(reason) = finish_reason.take() {
            let _ = tx.send(StreamChunk::Finish { reason }).await;
        }
        let _ = tx.send(StreamChunk::Done).await;
    });

    rx
}

pub fn map_provider_error(status: reqwest::StatusCode, body_text: String) -> ProviderError {
    match status.as_u16() {
        401 | 403 => ProviderError::AuthError(body_text),
        404 => ProviderError::ModelNotFound(body_text),
        429 => ProviderError::RateLimited {
            retry_after_ms: 1000,
        },
        // Everything else keeps the numeric status so failover/retry
        // policies can classify by status class (5xx vs 400) without
        // parsing the body.
        _ => ProviderError::Http {
            status: status.as_u16(),
            body: body_text,
        },
    }
}

#[derive(Default)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

async fn flush_openai_tool_calls(
    tx: &tokio::sync::mpsc::Sender<StreamChunk>,
    tool_calls: &mut BTreeMap<u64, ToolCallAccumulator>,
) {
    use crate::tools::input_repair;

    let drained = std::mem::take(tool_calls);
    for (index, call) in drained {
        if call.name.is_empty() {
            continue;
        }

        if let Ok(input) = serde_json::from_str(&call.arguments) {
            let _ = tx
                .send(StreamChunk::ToolUse(ToolCall {
                    id: if call.id.is_empty() {
                        format!("tool-call-{index}")
                    } else {
                        call.id
                    },
                    name: call.name,
                    input,
                }))
                .await;
        } else if let Some(input) = input_repair::repair_json_string(&call.arguments) {
            // Repaired from stream-level JSON issue (truncation, trailing comma, etc.)
            tracing::warn!(
                tool = %call.name,
                call_id = %call.id,
                index,
                "tool_input_stream_repaired"
            );
            let _ = tx
                .send(StreamChunk::ToolUse(ToolCall {
                    id: if call.id.is_empty() {
                        format!("tool-call-{index}")
                    } else {
                        call.id
                    },
                    name: call.name,
                    input,
                }))
                .await;
        } else {
            // Even stream-level repair failed — emit a tool call with the raw
            // string so the model gets an error it can recover from, rather
            // than having the call silently vanish.
            tracing::warn!(
                tool = %call.name,
                call_id = %call.id,
                index,
                arguments_preview = %truncate_bytes_safe(&call.arguments, 500),
                "tool_input_unparseable"
            );
            let _ = tx
                .send(StreamChunk::ToolUse(ToolCall {
                    id: if call.id.is_empty() {
                        format!("tool-call-{index}")
                    } else {
                        call.id
                    },
                    name: call.name,
                    input: json!({
                        "_error": format!(
                            "Failed to parse tool arguments as JSON. Raw input: {}",
                            truncate_bytes_safe(&call.arguments, 500)
                        )
                    }),
                }))
                .await;
        }
    }
}

/// Truncate `s` to at most `max_bytes` bytes, landing on a UTF-8 char boundary.
///
/// Naive `&s[..len.min(N)]` panics when `N` falls inside a multi-byte
/// character (e.g. an em dash or CJK glyph in tool arguments). This backs off
/// to the nearest preceding boundary so preview snippets never panic on
/// non-ASCII input.
pub(crate) fn truncate_bytes_safe(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn tool_content_string(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(t) => t.clone(),
        MessageContent::Parts(_) => content.to_summary_text(),
    }
}

/// Compact per-message shape summary for error-path diagnostics, e.g.
/// `System(text:4821),User(text:12)` or `User(parts:2)` for array content.
/// Lets a provider-side 400 (ZhipuAI 1213/1214 class) be diagnosed from the
/// log alone without reproducing the turn.
fn messages_shape(messages: &[Message]) -> String {
    messages
        .iter()
        .map(|m| {
            let content = match &m.content {
                MessageContent::Text(t) => format!("text:{}", t.len()),
                MessageContent::Parts(p) => format!("parts:{}", p.len()),
            };
            format!("{:?}({content})", m.role)
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn openai_user_content_value(
    content: &MessageContent,
    workspace_root: &Path,
) -> Result<Value, ProviderError> {
    match content {
        MessageContent::Text(s) => Ok(json!(s)),
        MessageContent::Parts(parts) => {
            let mut blocks = Vec::new();
            for p in parts {
                match p {
                    ContentPart::Text { text } => {
                        blocks.push(json!({
                            "type": "text",
                            "text": text,
                        }));
                    }
                    ContentPart::Image { media_type, path } => {
                        let full = workspace_root.join(path);
                        let bytes = std::fs::read(&full).map_err(|e| {
                            ProviderError::RequestFailed(format!(
                                "failed to read image {}: {e}",
                                full.display()
                            ))
                        })?;
                        let b64 = B64.encode(bytes);
                        let url = format!("data:{media_type};base64,{b64}");
                        blocks.push(json!({
                            "type": "image_url",
                            "image_url": { "url": url }
                        }));
                    }
                }
            }
            // Text-only parts collapse to a plain string: text-model chat
            // endpoints on strict OpenAI-compatible gateways (ZhipuAI among
            // them) reject array-form content for their text models with a
            // generic "prompt not received" 400. Arrays are only emitted when
            // an image block is actually present.
            let has_image = parts.iter().any(|p| matches!(p, ContentPart::Image { .. }));
            if !has_image {
                let text = blocks
                    .iter()
                    .filter_map(|b| b["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                return Ok(json!(text));
            }
            Ok(Value::Array(blocks))
        }
    }
}

fn to_openai_messages(
    messages: &[Message],
    workspace_root: &Path,
) -> Result<Vec<Value>, ProviderError> {
    // Preflight: strict gateways (ZhipuAI error 1213 "未正常接收到prompt参数")
    // return an opaque 400 for empty prompt payloads. Fail loudly here with a
    // precise, local error instead of a provider-side riddle.
    if messages.is_empty() {
        return Err(ProviderError::RequestFailed(
            "refusing to send a request with an empty messages list".into(),
        ));
    }

    let mut out = Vec::new();

    for (index, message) in messages.iter().enumerate() {
        match message.role {
            Role::System => {
                let content = tool_content_string(&message.content);
                if content.trim().is_empty() {
                    return Err(ProviderError::RequestFailed(format!(
                        "refusing to send an empty system message (index {index})"
                    )));
                }
                out.push(json!({
                    "role": "system",
                    "content": content,
                }));
            }
            Role::User => {
                let c = openai_user_content_value(&message.content, workspace_root)?;
                // Empty-payload guard on the WIRE value (not the in-memory
                // representation): whitespace-only text and empty Parts both
                // collapse to an unparsable prompt server-side (ZhipuAI 1213).
                let empty = match &c {
                    Value::String(s) => s.trim().is_empty(),
                    Value::Array(a) => a.is_empty(),
                    _ => false,
                };
                if empty {
                    return Err(ProviderError::RequestFailed(format!(
                        "refusing to send an empty user message (index {index})"
                    )));
                }
                out.push(json!({
                    "role": "user",
                    "content": c,
                }));
            }
            Role::Assistant => {
                let has_tool_calls = message.tool_calls.is_some();
                if message.content.is_empty() && !has_tool_calls {
                    return Err(ProviderError::RequestFailed(format!(
                        "refusing to send an empty assistant message with no tool calls (index {index})"
                    )));
                }
                let mut value = json!({
                    "role": "assistant",
                    "content": if message.content.is_empty() && has_tool_calls {
                        Value::Null
                    } else {
                        openai_user_content_value(&message.content, workspace_root)?
                    },
                });

                if let Some(reasoning) = &message.reasoning_content {
                    value["reasoning_content"] = json!(reasoning);
                }

                if let Some(calls) = &message.tool_calls {
                    value["tool_calls"] = Value::Array(
                        calls
                            .iter()
                            .map(|call| {
                                json!({
                                    "id": call.id,
                                    "type": "function",
                                    "function": {
                                        "name": call.name,
                                        "arguments": serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".into()),
                                    }
                                })
                            })
                            .collect(),
                    );
                }

                out.push(value);
            }
            Role::Tool => out.push(json!({
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": tool_content_string(&message.content),
            })),
        }
    }

    Ok(out)
}

/// Static profile describing the unique aspects of an OpenAI-compatible provider.
pub struct CompatProfile {
    /// Human-readable provider name (for error messages and stream labels).
    pub name: &'static str,
    /// Provider-specific endpoint suffix appended to base_url.
    pub endpoint_suffix: &'static str,
    /// Whether prepare_messages_for_request should strip reasoning_content.
    pub strip_reasoning: bool,
}

/// Generic OpenAI-compatible provider parameterized by a CompatProfile.
pub struct OpenAiCompatProvider {
    client: reqwest::Client,
    name: &'static str,
    model: String,
    max_tokens: u32,
    temperature: f32,
    base_url: String,
    endpoint_suffix: &'static str,
    strip_reasoning: bool,
    /// Vendor-specific request-body override merged into every
    /// chat/keepalive request body (e.g. ZhipuAI's
    /// `thinking: {"type": "enabled" | "disabled"}`).
    ///
    /// `None` (the default) emits no `thinking` field, preserving the plain
    /// OpenAI-compatible wire format.
    thinking: Option<Value>,
}

impl OpenAiCompatProvider {
    pub fn from_config(
        compat: &dyn nca_common::config::OpenAiCompatConfig,
        max_tokens: u32,
        profile: CompatProfile,
        extra_headers: reqwest::header::HeaderMap,
    ) -> Result<Self, ProviderError> {
        let api_key = compat.resolve_api_key().ok_or_else(|| {
            ProviderError::Configuration(format!(
                "missing {} API key; set {} or provide in config",
                profile.name,
                compat.api_key_env()
            ))
        })?;

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&format!("Bearer {api_key}")).map_err(
                |err| {
                    ProviderError::Configuration(format!(
                        "failed to build {} authorization header: {err}",
                        profile.name
                    ))
                },
            )?,
        );
        // Merge any extra headers (e.g. OpenRouter's http-referer, x-title).
        headers.extend(extra_headers);

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .connect_timeout(Duration::from_secs(30))
            .build()
            .map_err(|err| {
                ProviderError::Configuration(format!("failed to build HTTP client: {err}"))
            })?;

        Ok(Self {
            client,
            name: profile.name,
            model: compat.model().to_string(),
            max_tokens,
            temperature: compat.temperature(),
            base_url: compat.base_url().to_string(),
            endpoint_suffix: profile.endpoint_suffix,
            strip_reasoning: profile.strip_reasoning,
            thinking: None,
        })
    }

    /// Attach a vendor-specific `thinking` field merged into every
    /// chat/keepalive request body.
    ///
    /// Consuming builder: `from_config(...)?.with_thinking(json!({ "type": "disabled" }))`.
    pub fn with_thinking(mut self, thinking: Value) -> Self {
        self.thinking = Some(thinking);
        self
    }

    fn endpoint(&self) -> String {
        format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            self.endpoint_suffix
        )
    }
}

#[async_trait::async_trait]
impl Provider for OpenAiCompatProvider {
    async fn prepare_messages_for_request(
        &self,
        messages: &mut Vec<Message>,
        _workspace_root: &Path,
    ) -> Result<(), ProviderError> {
        if self.strip_reasoning {
            for msg in messages.iter_mut() {
                msg.reasoning_content = None;
            }
        }
        Ok(())
    }

    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        model: &str,
        workspace_root: &Path,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError> {
        let model = if model.is_empty() {
            self.model.clone()
        } else {
            model.to_string()
        };

        // Capability-aware clamp on the final model string: raise 128K-class
        // models to the floor, protect smaller output windows from oversize
        // values (hard 400s). keepalive_ping bypasses this (max_tokens = 1).
        let max_tokens = nca_common::model_limits::clamp_max_tokens(&model, self.max_tokens);

        let mut body = openai_request_body(
            messages,
            tools,
            &model,
            max_tokens,
            self.temperature,
            workspace_root,
        )?;
        if let Some(thinking) = &self.thinking {
            body["thinking"] = thinking.clone();
        };

        let response = self
            .client
            .post(self.endpoint())
            .json(&body)
            .send()
            .await
            .map_err(|err| {
                let chain = super::format_error_chain(&err);
                tracing::error!(
                    provider = self.name,
                    model = %model,
                    error = %err,
                    error_chain = %chain,
                    is_timeout = err.is_timeout(),
                    is_connect = err.is_connect(),
                    is_request = err.is_request(),
                    is_body = err.is_body(),
                    "provider_request_failed"
                );
                ProviderError::RequestFailed(chain)
            })?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            tracing::error!(
                provider = self.name,
                model = %model,
                http_status = %status,
                response_preview = %truncate_bytes_safe(&body_text, 500),
                request_messages_shape = %messages_shape(messages),
                request_tools = tools.len(),
                "provider_http_error"
            );
            return Err(map_provider_error(status, body_text));
        }

        Ok(spawn_openai_stream(response, self.name))
    }

    async fn keepalive_ping(
        &self,
        snapshot: &KeepaliveSnapshot,
    ) -> Result<PingUsage, ProviderError> {
        // Build the request with max_tokens=1 to minimise output cost.
        // The input prefix is what we're paying for — it refreshes the cache.
        // Deliberately NOT clamped: the capability clamp would raise this to
        // the 128K floor, billing a full completion just to refresh the cache.
        let mut body = openai_request_body(
            &snapshot.messages,
            &snapshot.tools,
            &snapshot.model,
            1, // max_tokens = 1
            self.temperature,
            &snapshot.workspace_root,
        )?;
        if let Some(thinking) = &self.thinking {
            body["thinking"] = thinking.clone();
        };

        let response = self
            .client
            .post(self.endpoint())
            .json(&body)
            .send()
            .await
            .map_err(|err| {
                let chain = super::format_error_chain(&err);
                ProviderError::RequestFailed(format!("{} keepalive error: {}", self.name, chain))
            })?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            return Err(map_provider_error(status, body_text));
        }

        // Drain the stream, discard all output, extract usage only.
        let mut stream = spawn_openai_stream(response, self.name);
        let mut result = PingUsage::default();

        while let Some(chunk) = stream.recv().await {
            match chunk {
                StreamChunk::Usage {
                    input_tokens,
                    cache_read_tokens,
                    cache_creation_tokens,
                    ..
                } => {
                    result.input_tokens = input_tokens;
                    result.cache_read_tokens = cache_read_tokens;
                    result.cache_creation_tokens = cache_creation_tokens;
                }
                StreamChunk::Error(e) => return Err(e),
                StreamChunk::Done => break,
                _ => {} // discard text, tool calls, reasoning
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_content_is_serialized_by_default() {
        let messages = vec![
            Message::user("hello"),
            Message::assistant("response").with_reasoning("I was thinking...".into()),
            Message::tool("call_1", "result"),
        ];

        let out = to_openai_messages(&messages, std::path::Path::new(".")).expect("messages");
        assert_eq!(out.len(), 3);

        // By default (non-DeepSeek providers), reasoning_content IS serialized
        // because some models require it for multi-turn reasoning continuity.
        let assistant = &out[1];
        assert_eq!(assistant["role"], "assistant");
        assert_eq!(
            assistant["reasoning_content"].as_str().unwrap(),
            "I was thinking..."
        );
        assert_eq!(assistant["content"], "response");
    }

    #[test]
    fn truncate_bytes_safe_never_panics_on_multibyte() {
        // A 500-byte cut lands inside the em dash (bytes 498..501) and panics
        // with a naive `&s[..500]`. Must back off to a char boundary instead.
        let s = format!("{}—{}", "a".repeat(498), "b".repeat(10));
        assert_eq!(s.len(), 511);
        let truncated = truncate_bytes_safe(&s, 500);
        assert!(truncated.len() <= 500);
        assert_eq!(truncated.len(), 498);
        assert_eq!(truncated, "a".repeat(498));
    }

    #[test]
    fn truncate_bytes_safe_returns_input_under_limit() {
        assert_eq!(truncate_bytes_safe("hello", 200), "hello");
        assert_eq!(truncate_bytes_safe("", 500), "");
    }

    // ---- ZhipuAI 1213-class guardrails: prompt payload must deserialize ----

    #[test]
    fn text_only_parts_collapse_to_plain_string_content() {
        let messages = vec![Message::user_with_parts(vec![
            ContentPart::Text {
                text: "hello ".into(),
            },
            ContentPart::Text {
                text: "world".into(),
            },
        ])];
        let out = to_openai_messages(&messages, std::path::Path::new(".")).expect("messages");
        assert!(
            out[0]["content"].is_string(),
            "text-only parts must not serialize as an array: {}",
            out[0]
        );
        assert_eq!(out[0]["content"], "hello \nworld");
    }

    #[test]
    fn image_parts_still_serialize_as_content_blocks() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("img.png"), b"png-bytes").expect("write image");
        let messages = vec![Message::user_with_parts(vec![
            ContentPart::Text {
                text: "look".into(),
            },
            ContentPart::Image {
                media_type: "image/png".into(),
                path: "img.png".into(),
            },
        ])];
        let out = to_openai_messages(&messages, dir.path()).expect("messages");
        let arr = out[0]["content"]
            .as_array()
            .expect("image-bearing parts stay array-form");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[1]["type"], "image_url");
    }

    #[test]
    fn empty_messages_list_rejected_loudly() {
        let err = to_openai_messages(&[], std::path::Path::new(".")).unwrap_err();
        assert!(err.to_string().contains("empty messages list"), "{err}");
    }

    #[test]
    fn blank_user_text_rejected_loudly() {
        // Whitespace-only text and image-less empty parts both collapse to an
        // unparsable prompt server-side (ZhipuAI 1213) — refuse locally.
        let cases = vec![Message::user("   "), Message::user_with_parts(vec![])];
        for message in cases {
            let messages = vec![message];
            let err = to_openai_messages(&messages, std::path::Path::new(".")).unwrap_err();
            assert!(err.to_string().contains("empty user message"), "{err}");
        }
    }

    #[test]
    fn empty_assistant_needs_tool_calls_to_survive() {
        // No tool calls + empty content → loud local error…
        let bad = vec![Message::assistant("")];
        let err = to_openai_messages(&bad, std::path::Path::new(".")).unwrap_err();
        assert!(err.to_string().contains("empty assistant"), "{err}");

        // …but empty content WITH tool calls stays `content: null` (legal).
        let mut with_calls = Message::assistant("");
        with_calls.tool_calls = Some(vec![nca_common::message::MessageToolCall {
            id: "call_1".into(),
            name: "read".into(),
            arguments: json!({}),
        }]);
        let out = to_openai_messages(&[with_calls], std::path::Path::new(".")).expect("messages");
        assert!(out[0]["content"].is_null());
        assert_eq!(out[0]["tool_calls"][0]["function"]["name"], "read");
    }

    #[test]
    fn tools_key_omitted_when_no_tools_not_null() {
        let body = openai_request_body(
            &[Message::user("hello")],
            &[],
            "glm-5.3",
            1_000,
            0.7,
            std::path::Path::new("."),
        )
        .expect("body");
        assert!(
            body.get("tools").is_none(),
            "empty tools must omit the key, not send null: {body}"
        );
    }

    #[test]
    fn tools_array_present_when_tools_exist() {
        let body = openai_request_body(
            &[Message::user("hello")],
            &[ToolDefinition {
                timeout_ms: None,
                name: "lookup".into(),
                description: "Lookup a path".into(),
                parameters: json!({"type": "object", "properties": {}}),
            }],
            "glm-5.3",
            1_000,
            0.7,
            std::path::Path::new("."),
        )
        .expect("body");
        let tools = body["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "lookup");
    }
}
