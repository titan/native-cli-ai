use std::path::Path;
use std::time::Duration;

use base64::{Engine, engine::general_purpose::STANDARD as B64};
use futures_util::StreamExt;
use nca_common::message::{ContentPart, Message, MessageContent, Role};
use nca_common::tool::{ToolCall, ToolDefinition};
use serde_json::{Value, json};

use super::{ByteStreamError, ProviderError, StreamChunk};

/// 单次流式读取的空闲超时（与 `openai_compat::STREAM_IDLE_TIMEOUT` 保持一致）。
///
/// 用 `tokio::time::timeout` 包裹每次 `bytes_stream().next()`，替代 reqwest 的全局
/// 总超时：只要持续有 token 流出，整个流可以跑任意长时间（reasoning/thinking 模型
/// 总耗时常超过 120s）；仅在连接静默超过该阈值时才报错。
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(90);

/// Build the Anthropic Messages API request body with prompt-cache breakpoints.
///
/// Anthropic's prompt caching requires explicit `cache_control` markers to opt
/// the prefix into caching. We place two breakpoints (out of the 4 max):
///
/// 1. **System prompt** — the largest stable block. Wrapping it in a structured
///    text block with `cache_control: { type: "ephemeral" }` makes it
///    cache-eligible. The TTL is 5 minutes (refreshed on each hit).
///
/// 2. **Last tool definition** — the second largest stable block. A breakpoint
///    here caches `system + tools` as a single prefix unit, which is reused on
///    every subsequent turn (tools rarely change mid-session).
///
/// Conversation history grows after the tools, so those breakpoints cover the
/// entire stable prefix. Without these markers Anthropic will **never** cache,
/// even if the prefix is byte-identical — keepalive or not.
pub fn anthropic_request_body(
    messages: &[Message],
    tools: &[ToolDefinition],
    model: &str,
    max_tokens: u32,
    temperature: f32,
    workspace_root: &Path,
) -> Result<Value, ProviderError> {
    let (system, anthropic_messages) = to_anthropic_messages(messages, workspace_root)?;

    // System prompt as a structured text block with cache_control so the
    // prefix is eligible for Anthropic prompt caching.
    let system = system.map(|s| {
        json!([{
            "type": "text",
            "text": s,
            "cache_control": { "type": "ephemeral" }
        }])
    });

    let tools = if tools.is_empty() {
        None
    } else {
        let mut tool_list: Vec<Value> = tools
            .iter()
            .map(|tool| {
                json!({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                })
            })
            .collect();
        // Cache breakpoint on the last tool: caches system + all tools as a
        // single prefix unit reused on every subsequent turn.
        if let Some(last) = tool_list.last_mut() {
            last["cache_control"] = json!({ "type": "ephemeral" });
        }
        Some(Value::Array(tool_list))
    };

    Ok(json!({
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": anthropic_messages,
        "tools": tools,
        "stream": true,
        "temperature": temperature,
    }))
}

/// Spawn the Anthropic-compatible SSE parser over a live HTTP response body.
///
/// Thin shell: adapts the reqwest body into the testable core's stream
/// shape (`Result<chunk, ByteStreamError>` — source chain and transport
/// flags captured eagerly at yield time) and delegates to
/// [`run_anthropic_sse`].
pub fn spawn_anthropic_stream(
    response: reqwest::Response,
    provider_name: &'static str,
) -> tokio::sync::mpsc::Receiver<StreamChunk> {
    let byte_stream = response
        .bytes_stream()
        .map(|item| item.map_err(ByteStreamError::from));
    run_anthropic_sse(byte_stream, provider_name)
}

/// Testable core of the Anthropic-compatible SSE parser: "byte stream →
/// StreamChunk".
///
/// Split out of [`spawn_anthropic_stream`] so conformance tests can feed
/// fixture byte streams (plain `String`/`Vec<u8>` chunks work — anything
/// `AsRef<[u8]>`) without real HTTP. The idle-timeout wrapping, error
/// mapping (source chain + transport flags), and the `stream_idle_timeout` /
/// `stream_byte_error` logging all live here. Events are assembled per the
/// WHATWG SSE spec: lines are extracted from a raw byte buffer (UTF-8 decoded
/// per complete line, so mid-character chunk boundaries cannot corrupt) and
/// dispatched in batch when the terminating blank line arrives — multi-line
/// `data:` values are joined with `\n` as one event, dispatched against the
/// recorded `event:` type.
pub(crate) fn run_anthropic_sse<B>(
    mut byte_stream: impl futures_util::Stream<Item = Result<B, ByteStreamError>>
    + Send
    + 'static
    + Unpin,
    provider_name: &'static str,
) -> tokio::sync::mpsc::Receiver<StreamChunk>
where
    B: AsRef<[u8]> + Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel(64);

    tokio::spawn(async move {
        // Raw-byte line-assembly buffer. UTF-8 decoding happens per COMPLETE
        // extracted line, never per network chunk — a chunk boundary landing
        // inside a multi-byte sequence stays buffered as raw bytes until its
        // line is complete (`0x0A` can never appear inside a multi-byte UTF-8
        // sequence — every lead/continuation byte is >= 0x80 — so slicing at
        // `\n` bytes can never split a code point).
        let mut buffer: Vec<u8> = Vec::new();
        // `data:` field values of the event currently being assembled, per
        // the SSE spec: an event's data may span multiple physical `data:`
        // lines; the values are joined with `\n` and dispatched as ONE event
        // (against `event_type`) when the terminating blank line arrives. An
        // event still incomplete at EOF is discarded.
        let mut data_lines: Vec<String> = Vec::new();
        let mut event_type = String::new();
        let mut tool_id = String::new();
        let mut tool_name = String::new();
        let mut tool_input = String::new();
        // Per-stream ordinal of flushed tool calls — backs the synthetic
        // `tool-call-{n}` id when a compat endpoint omits the block id
        // (mirror of openai_compat's `tool-call-{index}`).
        let mut tool_seq: u64 = 0;
        let mut input_tokens: u64 = 0;
        let mut cache_creation_tokens: u64 = 0;
        let mut cache_read_tokens: u64 = 0;

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
                        String::from_utf8_lossy(&buffer).chars().take(500).collect()
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
                    // disruptions. The adapter captured the full source chain
                    // and transport flags at yield time; include any buffered
                    // response data for diagnostics.
                    let ByteStreamError {
                        display: error_display,
                        chain,
                        is_timeout,
                        is_connect,
                        is_request,
                        is_body,
                    } = err;
                    let buffer_preview = if buffer.is_empty() {
                        String::from("(none)")
                    } else {
                        String::from_utf8_lossy(&buffer).chars().take(500).collect()
                    };

                    tracing::error!(
                        provider = provider_name,
                        error = %error_display,
                        error_chain = %chain,
                        is_timeout,
                        is_connect,
                        is_request,
                        is_body,
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

            buffer.extend_from_slice(chunk.as_ref());

            while let Some(nl) = buffer.iter().position(|&b| b == b'\n') {
                let raw = String::from_utf8_lossy(&buffer[..nl]).into_owned();
                buffer.drain(..=nl);
                let line = raw.trim_end_matches('\r').trim();

                if !line.is_empty() {
                    // Accumulate the event's fields; dispatch happens on the
                    // blank line below.
                    if let Some(event) = line.strip_prefix("event:") {
                        event_type = event.trim().to_string();
                    } else if let Some(data) = line.strip_prefix("data:") {
                        data_lines.push(data.trim().to_string());
                    }
                    // Comment (`:`-prefixed) lines and other SSE fields (`id`,
                    // `retry`) are ignored.
                    continue;
                }

                // Blank line: the current event is complete — dispatch the
                // assembled data per the SSE spec (multi-line `data:` values
                // joined with `\n`) against the recorded event type. Events
                // with no `data:` field (e.g. comment-only) dispatch nothing.
                if data_lines.is_empty() {
                    event_type.clear();
                    continue;
                }
                let data = std::mem::take(&mut data_lines).join("\n");
                if data == "[DONE]" {
                    break;
                }

                let event = match serde_json::from_str::<Value>(&data) {
                    Ok(event) => event,
                    Err(_) => {
                        event_type.clear();
                        continue;
                    }
                };

                match event_type.as_str() {
                    "message_start" => {
                        let usage = &event["message"]["usage"];
                        input_tokens = usage["input_tokens"].as_u64().unwrap_or(0);
                        // Anthropic reports cache tokens in message_start usage.
                        cache_creation_tokens =
                            usage["cache_creation_input_tokens"].as_u64().unwrap_or(0);
                        cache_read_tokens = usage["cache_read_input_tokens"].as_u64().unwrap_or(0);
                    }
                    "content_block_start" => {
                        let block = &event["content_block"];
                        if block["type"].as_str().unwrap_or("") == "tool_use" {
                            tool_id = block["id"].as_str().unwrap_or("").to_string();
                            tool_name = block["name"].as_str().unwrap_or("").to_string();
                            tool_input.clear();
                        }
                    }
                    "content_block_delta" => {
                        let delta = &event["delta"];
                        match delta["type"].as_str().unwrap_or("") {
                            "text_delta" => {
                                if let Some(text) = delta["text"].as_str()
                                    && !text.is_empty()
                                {
                                    let _ = tx.send(StreamChunk::TextDelta(text.to_string())).await;
                                }
                            }
                            "input_json_delta" => {
                                if let Some(partial) = delta["partial_json"].as_str() {
                                    tool_input.push_str(partial);
                                }
                            }
                            "thinking_delta" => {
                                // Anthropic thinking blocks (used by MiniMax and others).
                                if let Some(thinking) = delta["thinking"].as_str()
                                    && !thinking.is_empty()
                                {
                                    let _ = tx
                                        .send(StreamChunk::ReasoningDelta(thinking.to_string()))
                                        .await;
                                }
                            }
                            _ => {}
                        }
                    }
                    "content_block_stop" => {
                        flush_anthropic_tool_call(
                            &tx,
                            &mut tool_id,
                            &mut tool_name,
                            &mut tool_input,
                            &mut tool_seq,
                        )
                        .await;
                    }
                    "message_delta" => {
                        let output_tokens = event["usage"]["output_tokens"].as_u64().unwrap_or(0);
                        if input_tokens > 0 || output_tokens > 0 {
                            let _ = tx
                                .send(StreamChunk::Usage {
                                    input_tokens,
                                    output_tokens,
                                    cache_creation_tokens,
                                    cache_read_tokens,
                                })
                                .await;
                            input_tokens = 0;
                            cache_creation_tokens = 0;
                            cache_read_tokens = 0;
                        }
                    }
                    _ => {}
                }
                event_type.clear();
            }
        }

        // Clean EOF: an event still incomplete (no blank-line terminator) is
        // discarded per the SSE spec; trailing tool state still flushes.
        flush_anthropic_tool_call(
            &tx,
            &mut tool_id,
            &mut tool_name,
            &mut tool_input,
            &mut tool_seq,
        )
        .await;
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

async fn flush_anthropic_tool_call(
    tx: &tokio::sync::mpsc::Sender<StreamChunk>,
    tool_id: &mut String,
    tool_name: &mut String,
    tool_input: &mut String,
    tool_seq: &mut u64,
) {
    use super::openai_compat::truncate_bytes_safe;
    use crate::tools::input_repair;

    if tool_name.is_empty() {
        return;
    }

    // Mirror of openai_compat's `tool-call-{index}`: Anthropic streams carry
    // no index, so synthesize from the per-stream tool-call ordinal.
    let seq = *tool_seq;
    *tool_seq += 1;

    if let Ok(input) = serde_json::from_str(tool_input) {
        let _ = tx
            .send(StreamChunk::ToolUse(ToolCall {
                id: if tool_id.is_empty() {
                    format!("tool-call-{seq}")
                } else {
                    tool_id.clone()
                },
                name: tool_name.clone(),
                input,
            }))
            .await;
    } else if let Some(input) = input_repair::repair_json_string(tool_input) {
        // Repaired from stream-level JSON issue (truncation, trailing comma, etc.)
        tracing::warn!(
            tool = %tool_name,
            call_id = %tool_id,
            seq,
            "tool_input_stream_repaired"
        );
        let _ = tx
            .send(StreamChunk::ToolUse(ToolCall {
                id: if tool_id.is_empty() {
                    format!("tool-call-{seq}")
                } else {
                    tool_id.clone()
                },
                name: tool_name.clone(),
                input,
            }))
            .await;
    } else {
        // Even stream-level repair failed — emit a tool call with the raw
        // string so the model gets an error it can recover from, rather
        // than having the call silently vanish.
        tracing::warn!(
            tool = %tool_name,
            call_id = %tool_id,
            seq,
            arguments_preview = %truncate_bytes_safe(tool_input, 500),
            "tool_input_unparseable"
        );
        let _ = tx
            .send(StreamChunk::ToolUse(ToolCall {
                id: if tool_id.is_empty() {
                    format!("tool-call-{seq}")
                } else {
                    tool_id.clone()
                },
                name: tool_name.clone(),
                input: json!({
                    "_error": format!(
                        "Failed to parse tool arguments as JSON. Raw input: {}",
                        truncate_bytes_safe(tool_input, 500)
                    )
                }),
            }))
            .await;
    }

    tool_id.clear();
    tool_name.clear();
    tool_input.clear();
}

fn tool_content_string(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(t) => t.clone(),
        MessageContent::Parts(_) => content.to_summary_text(),
    }
}

fn user_content_value(
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
                        let data = B64.encode(bytes);
                        blocks.push(json!({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data,
                            }
                        }));
                    }
                }
            }
            Ok(Value::Array(blocks))
        }
    }
}

fn to_anthropic_messages(
    messages: &[Message],
    workspace_root: &Path,
) -> Result<(Option<String>, Vec<Value>), ProviderError> {
    let mut system_parts = Vec::new();
    let mut out = Vec::new();
    let mut i = 0;

    while i < messages.len() && messages[i].role == Role::System {
        match &messages[i].content {
            MessageContent::Text(t) => system_parts.push(t.clone()),
            MessageContent::Parts(_) => system_parts.push(messages[i].content.to_summary_text()),
        }
        i += 1;
    }

    while i < messages.len() {
        let message = &messages[i];
        match message.role {
            Role::User => {
                let content = user_content_value(&message.content, workspace_root)?;
                out.push(json!({
                    "role": "user",
                    "content": content,
                }));
                i += 1;
            }
            Role::Assistant => {
                let mut blocks = Vec::new();
                if let MessageContent::Text(t) = &message.content {
                    if !t.is_empty() {
                        blocks.push(json!({
                            "type": "text",
                            "text": t,
                        }));
                    }
                } else {
                    let v = user_content_value(&message.content, workspace_root)?;
                    if let Value::Array(arr) = v {
                        blocks.extend(arr);
                    }
                }
                if let Some(calls) = &message.tool_calls {
                    for call in calls {
                        blocks.push(json!({
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }));
                    }
                }

                let content_out = if blocks.is_empty() {
                    match &message.content {
                        MessageContent::Text(t) => json!(t),
                        MessageContent::Parts(_) => {
                            user_content_value(&message.content, workspace_root)?
                        }
                    }
                } else {
                    Value::Array(blocks)
                };

                out.push(json!({
                    "role": "assistant",
                    "content": content_out,
                }));
                i += 1;
            }
            Role::Tool => {
                let mut results = Vec::new();
                while i < messages.len() && messages[i].role == Role::Tool {
                    let tool_message = &messages[i];
                    results.push(json!({
                        "type": "tool_result",
                        "tool_use_id": tool_message.tool_call_id.as_deref().unwrap_or(""),
                        "content": tool_content_string(&tool_message.content),
                    }));
                    i += 1;
                }
                out.push(json!({
                    "role": "user",
                    "content": results,
                }));
            }
            Role::System => {
                i += 1;
            }
        }
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };

    Ok((system, out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{Engine, engine::general_purpose::STANDARD as B64};
    use nca_common::message::{ContentPart, Message};
    use tempfile::tempdir;

    /// Drive `flush_anthropic_tool_call` once and collect what it emitted.
    async fn flush_chunks(tool_id: &str, tool_name: &str, tool_input: &str) -> Vec<StreamChunk> {
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let mut id = tool_id.to_string();
        let mut name = tool_name.to_string();
        let mut input = tool_input.to_string();
        let mut seq = 0u64;
        flush_anthropic_tool_call(&tx, &mut id, &mut name, &mut input, &mut seq).await;
        drop(tx);
        let mut chunks = Vec::new();
        while let Some(chunk) = rx.recv().await {
            chunks.push(chunk);
        }
        chunks
    }

    #[tokio::test]
    async fn anthropic_flush_repairs_stream_broken_tool_input_instead_of_dropping() {
        // Trailing comma: invalid JSON as-streamed, but stream-level repair
        // recovers it — same policy as the openai compat parser.
        let chunks = flush_chunks("toolu_1", "lookup", r#"{"path":"src/main.rs",}"#).await;
        assert!(matches!(
            &chunks[..],
            [StreamChunk::ToolUse(call)] if call.id == "toolu_1"
                && call.name == "lookup"
                && call.input == json!({"path": "src/main.rs"})
        ));
    }

    #[tokio::test]
    async fn anthropic_flush_unparseable_tool_input_fails_loudly_not_silently() {
        // Truncated mid-value and irreparable: the ToolUse must still be
        // emitted with an `_error` payload — never silently dropped.
        let chunks = flush_chunks("toolu_2", "write_file", r#"{"path":"src"#).await;
        assert!(matches!(
            &chunks[..],
            [StreamChunk::ToolUse(call)] if call.id == "toolu_2"
                && call.name == "write_file"
                && call.input.get("_error").is_some()
        ));
    }

    #[tokio::test]
    async fn anthropic_flush_synthesizes_id_when_missing() {
        // Anthropic always sends a block id, but compat endpoints may not —
        // mirror openai's `tool-call-{index}` with a per-stream ordinal.
        let chunks = flush_chunks("", "lookup", r#"{"path":"src"}"#).await;
        assert!(matches!(
            &chunks[..],
            [StreamChunk::ToolUse(call)] if call.id == "tool-call-0" && call.name == "lookup"
        ));
    }

    #[tokio::test]
    async fn run_anthropic_sse_parses_frames_from_in_memory_bytes() {
        // The extracted core runs without HTTP: feed SSE bytes directly.
        let frames = concat!(
            "event: content_block_start\n",
            "data: {\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_9\",\"name\":\"lookup\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"src\\\"}\"}}\n\n",
            "event: content_block_stop\n",
            "data: {}\n\n"
        )
        .to_string();
        let byte_stream = futures_util::stream::iter(vec![Ok::<
            String,
            crate::provider::ByteStreamError,
        >(frames)]);
        let mut rx = run_anthropic_sse(byte_stream, "test-anthropic");
        let mut chunks = Vec::new();
        while let Some(chunk) = rx.recv().await {
            chunks.push(chunk);
        }
        assert!(matches!(
            &chunks[..],
            [StreamChunk::ToolUse(call), StreamChunk::Done] if call.id == "toolu_9"
                && call.name == "lookup"
                && call.input == json!({"path": "src"})
        ));
    }

    #[test]
    fn user_multimodal_message_serializes_image_base64_block() {
        let dir = tempdir().unwrap();
        let workspace = dir.path();
        let rel = ".nca/sessions/s1/attachments/x.png";
        let att_dir = workspace.join(".nca/sessions/s1/attachments");
        std::fs::create_dir_all(&att_dir).unwrap();
        let png = B64
            .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
            .unwrap();
        std::fs::write(att_dir.join("x.png"), png).unwrap();

        let messages = vec![Message::user_with_parts(vec![
            ContentPart::Text {
                text: "describe".into(),
            },
            ContentPart::Image {
                media_type: "image/png".into(),
                path: rel.into(),
            },
        ])];

        let body = anthropic_request_body(&messages, &[], "MiniMax-M2.7", 128, 1.0, workspace)
            .expect("body");
        let content = body["messages"][0]["content"].as_array().expect("array");
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["source"]["type"], "base64");
        assert_eq!(content[1]["source"]["media_type"], "image/png");
        assert!(content[1]["source"]["data"].as_str().unwrap().len() > 8);
    }
}
