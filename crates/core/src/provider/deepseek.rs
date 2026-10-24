//! Tests for the DeepSeek provider (served by `OpenAiCompatProvider` with
//! `ReasoningWirePolicy::ToolsGated`, resolved at wire-serialization time).
//!
//! Contract anchor (verified against the official docs, 2026-10-24):
//! when a request carries `tools`, ALL prior assistant messages must pass
//! `reasoning_content` back — omitting it is rejected with a 400:
//! "The `reasoning_content` in the thinking mode must be passed back to the
//! API." Without `tools` the field is ignored server-side. The legacy
//! aliases `deepseek-chat`/`deepseek-reasoner` were decommissioned
//! 2026-07-24; the current models (`deepseek-flash`/`deepseek-v4-pro`) are
//! hybrid, so the decision keys on the request's `tools` alone — never the
//! model name.

#[cfg(test)]
mod tests {
    use crate::provider::openai_compat::{
        CompatProfile, OpenAiCompatProvider, ReasoningWirePolicy,
    };
    use crate::provider::test_support::{
        collect_chunks, spawn_sse_server, spawn_sse_server_with_body,
    };
    use crate::provider::{Provider, StreamChunk};
    use nca_common::config::NcaConfig;
    use nca_common::message::Message;
    use nca_common::tool::ToolDefinition;
    use serde_json::{Value, json};
    use std::path::Path;

    // Mirrors the factory's DEEPSEEK_PROFILE (wire contract must stay in
    // sync): ToolsGated + NO model-name marker — the decision keys on the
    // request's `tools` parameter alone.
    const DEEPSEEK_PROFILE: CompatProfile = CompatProfile {
        name: "DeepSeek",
        endpoint_suffix: "chat/completions",
        reasoning: ReasoningWirePolicy::ToolsGated,
        thinking_model_marker: None,
        send_stream_options: true,
        use_max_completion_tokens: false,
    };

    fn lookup_tool() -> ToolDefinition {
        ToolDefinition {
            timeout_ms: None,
            name: "lookup".into(),
            description: "Lookup a path".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                }
            }),
        }
    }

    /// History shaped like a resumed/multi-round tool session: an assistant
    /// turn WITH reasoning and a tool call, plus a tool-less reasoning turn.
    /// The DeepSeek tools-mode contract covers BOTH assistant messages.
    fn tool_loop_history() -> Vec<Message> {
        let mut with_call = Message::assistant("response").with_reasoning("chain A".into());
        with_call.tool_calls = Some(vec![nca_common::message::MessageToolCall {
            id: "call_1".into(),
            name: "lookup".into(),
            arguments: json!({}),
        }]);
        vec![
            Message::user("hello"),
            with_call,
            Message::tool("call_1", "result"),
            Message::assistant("plain turn").with_reasoning("chain B".into()),
            Message::user("continue"),
        ]
    }

    fn build_provider(config: &NcaConfig) -> OpenAiCompatProvider {
        OpenAiCompatProvider::from_config(
            &config.provider.deepseek,
            config.model.max_tokens,
            DEEPSEEK_PROFILE,
            reqwest::header::HeaderMap::new(),
        )
        .expect("provider")
    }

    /// Assert the serialized request body's assistant messages: every
    /// `reasoning_content` present (with its text) when `expect_keep`, absent
    /// otherwise. Returns the parsed body for extra checks.
    fn assert_wire_reasoning(request_body: &str, expect_keep: bool) -> Value {
        let parsed: Value = serde_json::from_str(request_body).expect("request body is JSON");
        let assistants: Vec<&Value> = parsed["messages"]
            .as_array()
            .expect("messages array")
            .iter()
            .filter(|m| m["role"] == "assistant")
            .collect();
        assert_eq!(assistants.len(), 2, "body: {request_body}");
        for assistant in &assistants {
            if expect_keep {
                let text = assistant["reasoning_content"].as_str().unwrap_or_else(|| {
                    panic!("reasoning_content must be passed back: {assistant}")
                });
                // Verbatim preservation, not a placeholder.
                assert!(
                    text == "chain A" || text == "chain B",
                    "reasoning must be verbatim: {text}"
                );
            } else {
                assert!(
                    assistant.get("reasoning_content").is_none(),
                    "no-tools request must omit reasoning_content (ignored server-side; stripped to save context): {assistant}"
                );
            }
        }
        parsed
    }

    fn sse_ok() -> String {
        concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        )
        .to_string()
    }

    // ---- four-quadrant matrix: model × tools presence --------------------
    //
    // The pass-back predicate is `tools`, NOT the model name: both current
    // models must KEEP reasoning on the wire for tools requests and STRIP it
    // otherwise. This matrix is the regression for the third recurrence of
    // the 400 (the old tests pinned only the decommissioned aliases).

    #[tokio::test]
    async fn deepseek_flash_with_tools_keeps_reasoning_on_the_wire() {
        let base_url = spawn_sse_server_with_body(sse_ok(), 200, |_request, request_body| {
            let parsed = assert_wire_reasoning(request_body, true);
            assert!(
                parsed["tools"].as_array().is_some_and(|t| !t.is_empty()),
                "body: {request_body}"
            );
        });
        let mut config = NcaConfig::default();
        config.provider.deepseek.api_key = Some("deepseek-test-key".into());
        config.provider.deepseek.base_url = base_url;
        config.provider.deepseek.model = "deepseek-flash".into();

        let provider = build_provider(&config);
        let history = tool_loop_history();
        let stream = provider
            .chat(&history, &[lookup_tool()], "", Path::new("."))
            .await
            .expect("chat stream");
        let chunks = collect_chunks(stream).await;
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
    }

    #[tokio::test]
    async fn deepseek_flash_without_tools_strips_reasoning_on_the_wire() {
        let base_url = spawn_sse_server_with_body(sse_ok(), 200, |_request, request_body| {
            let parsed = assert_wire_reasoning(request_body, false);
            assert!(parsed.get("tools").is_none(), "body: {request_body}");
        });
        let mut config = NcaConfig::default();
        config.provider.deepseek.api_key = Some("deepseek-test-key".into());
        config.provider.deepseek.base_url = base_url;
        config.provider.deepseek.model = "deepseek-flash".into();

        let provider = build_provider(&config);
        let history = tool_loop_history();
        let stream = provider
            .chat(&history, &[], "", Path::new("."))
            .await
            .expect("chat stream");
        let chunks = collect_chunks(stream).await;
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
    }

    #[tokio::test]
    async fn deepseek_v4_pro_with_tools_keeps_reasoning_on_the_wire() {
        let base_url = spawn_sse_server_with_body(sse_ok(), 200, |_request, request_body| {
            assert_wire_reasoning(request_body, true);
        });
        let mut config = NcaConfig::default();
        config.provider.deepseek.api_key = Some("deepseek-test-key".into());
        config.provider.deepseek.base_url = base_url;
        config.provider.deepseek.model = "deepseek-v4-pro".into();

        let provider = build_provider(&config);
        let history = tool_loop_history();
        let stream = provider
            .chat(&history, &[lookup_tool()], "", Path::new("."))
            .await
            .expect("chat stream");
        let chunks = collect_chunks(stream).await;
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
    }

    #[tokio::test]
    async fn deepseek_v4_pro_without_tools_strips_reasoning_on_the_wire() {
        let base_url = spawn_sse_server_with_body(sse_ok(), 200, |_request, request_body| {
            assert_wire_reasoning(request_body, false);
        });
        let mut config = NcaConfig::default();
        config.provider.deepseek.api_key = Some("deepseek-test-key".into());
        config.provider.deepseek.base_url = base_url;
        config.provider.deepseek.model = "deepseek-v4-pro".into();

        let provider = build_provider(&config);
        let history = tool_loop_history();
        let stream = provider
            .chat(&history, &[], "", Path::new("."))
            .await
            .expect("chat stream");
        let chunks = collect_chunks(stream).await;
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
    }

    // The predicate must not depend on the caller-supplied model string
    // either: fallback pins and agent profiles may dial any model name on
    // the same provider instance, and tools-mode keep must still hold.
    #[tokio::test]
    async fn tools_gating_ignores_the_caller_supplied_model_string() {
        let base_url = spawn_sse_server_with_body(sse_ok(), 200, |_request, request_body| {
            let parsed = assert_wire_reasoning(request_body, true);
            assert_eq!(parsed["model"], "deepseek-flash", "body: {request_body}");
        });
        let mut config = NcaConfig::default();
        config.provider.deepseek.api_key = Some("deepseek-test-key".into());
        config.provider.deepseek.base_url = base_url;
        // Configured model is v4-pro; the caller pins flash. Either way the
        // tools-mode keep decision is identical.
        config.provider.deepseek.model = "deepseek-v4-pro".into();

        let provider = build_provider(&config);
        let history = tool_loop_history();
        let stream = provider
            .chat(&history, &[lookup_tool()], "deepseek-flash", Path::new("."))
            .await
            .expect("chat stream");
        let chunks = collect_chunks(stream).await;
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
    }

    // ---- streaming sanity (unchanged wire path) ---------------------------

    #[tokio::test]
    async fn deepseek_provider_streams_text_tool_and_usage() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello \"},\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"path\\\":\\\"\"}}]},\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"src\\\"}\"}}]},\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7}}\n\n",
            "data: [DONE]\n\n"
        )
        .to_string();
        let base_url = spawn_sse_server(body, 200, |request| {
            assert_eq!(request.url(), "/chat/completions");
            let auth = request
                .headers()
                .iter()
                .find(|header| header.field.equiv("authorization"))
                .expect("authorization header");
            assert_eq!(auth.value.as_str(), "Bearer deepseek-test-key");
        });

        let mut config = NcaConfig::default();
        config.provider.deepseek.api_key = Some("deepseek-test-key".into());
        config.provider.deepseek.base_url = base_url;

        let provider = build_provider(&config);
        let stream = provider
            .chat(
                &[Message::user("hello")],
                &[lookup_tool()],
                "",
                Path::new("."),
            )
            .await
            .expect("chat stream");

        let chunks = collect_chunks(stream).await;
        assert!(matches!(&chunks[0], StreamChunk::TextDelta(text) if text == "Hello "));
        assert!(
            matches!(&chunks[1], StreamChunk::ToolUse(call) if call.id == "call_1" && call.name == "lookup" && call.input == json!({"path":"src"}))
        );
        assert!(matches!(
            &chunks[2],
            StreamChunk::Usage {
                input_tokens: 11,
                output_tokens: 7,
                ..
            }
        ));
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
    }

    // ---- non-destructive prepare (session history is the source of truth) -

    #[tokio::test]
    async fn prepare_messages_for_request_never_rewrites_history() {
        // The stored history must survive prepare VERBATIM — reasoning and
        // all. The keep/strip decision lives at wire serialization
        // (`assert_wire_reasoning` above pins it per request), so one wrong
        // policy call can never irreversibly delete reasoning from
        // `agent.messages` (the replay/session source of truth) again.
        let mut config = NcaConfig::default();
        config.provider.deepseek.api_key = Some("test".into());
        config.provider.deepseek.model = "deepseek-flash".into();
        let provider = build_provider(&config);

        let history = tool_loop_history();
        let mut messages = history.clone();

        provider
            .prepare_messages_for_request(&mut messages, Path::new("."))
            .await
            .expect("prepare");

        assert_eq!(
            messages, history,
            "prepare must be non-destructive for the OpenAI-compat family"
        );
    }

    // ---- keepalive: same per-request resolution as chat --------------------

    #[tokio::test]
    async fn deepseek_keepalive_body_matches_chat_reasoning_resolution() {
        // Keepalive warms the SAME prefix the next real request will send:
        // with tools in the snapshot the warmed body must carry
        // reasoning_content (byte-identical prefix → cache hit), without
        // tools it must omit it.
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":1}}\n\n",
            "data: [DONE]\n\n"
        )
        .to_string();
        let base_url = spawn_sse_server_with_body(body, 200, |_request, request_body| {
            assert_wire_reasoning(request_body, true);
        });

        let mut config = NcaConfig::default();
        config.provider.deepseek.api_key = Some("deepseek-test-key".into());
        config.provider.deepseek.base_url = base_url;
        config.provider.deepseek.model = "deepseek-flash".into();

        let provider = build_provider(&config);
        let snapshot = crate::cache_keepalive::KeepaliveSnapshot {
            messages: tool_loop_history(),
            tools: vec![lookup_tool()],
            model: "deepseek-flash".into(),
            workspace_root: std::path::PathBuf::from("."),
        };
        let usage = provider
            .keepalive_ping(&snapshot)
            .await
            .expect("keepalive ping");
        assert_eq!(usage.input_tokens, 100);
    }
}
