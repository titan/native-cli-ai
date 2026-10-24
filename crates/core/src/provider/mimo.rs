//! Tests for the Xiaomi MiMo provider (served by `OpenAiCompatProvider`).

#[cfg(test)]
mod tests {
    use crate::cache_keepalive::KeepaliveSnapshot;
    use crate::provider::factory::build_provider_for;
    use crate::provider::openai_compat::{
        CompatProfile, OpenAiCompatProvider, ReasoningWirePolicy,
    };
    use crate::provider::test_support::{collect_chunks, spawn_sse_server_with_body};
    use crate::provider::{Provider, StreamChunk};
    use nca_common::config::{NcaConfig, ProviderKind};
    use nca_common::message::Message;
    use std::path::{Path, PathBuf};

    // Mirrors the factory's MIMO_PROFILE (wire format must stay in sync).
    const MIMO_PROFILE: CompatProfile = CompatProfile {
        name: "mimo",
        endpoint_suffix: "chat/completions",
        reasoning: ReasoningWirePolicy::StripUnlessThinking,
        thinking_model_marker: None,
        send_stream_options: false,
        use_max_completion_tokens: true,
    };

    #[tokio::test]
    async fn mimo_provider_streams_text_and_usage() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello \"},\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"world\"},\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n",
            "data: [DONE]\n\n"
        )
        .to_string();
        let base_url = spawn_sse_server_with_body(body, 200, |request, request_body| {
            assert_eq!(request.url(), "/chat/completions");
            let auth = request
                .headers()
                .iter()
                .find(|header| header.field.equiv("authorization"))
                .expect("authorization header");
            assert_eq!(auth.value.as_str(), "Bearer mimo-test-key");
            // Sanity: the request body is well-formed OpenAI-compat JSON.
            let parsed: serde_json::Value =
                serde_json::from_str(request_body).expect("request body is JSON");
            assert_eq!(parsed["model"], "mimo-v2.6-pro", "body: {request_body}");
        });

        let mut config = NcaConfig::default();
        config.provider.mimo.api_key = Some("mimo-test-key".into());
        config.provider.mimo.base_url = base_url;

        let provider = OpenAiCompatProvider::from_config(
            &config.provider.mimo,
            config.model.max_tokens,
            MIMO_PROFILE,
            reqwest::header::HeaderMap::new(),
        )
        .expect("provider");
        let stream = provider
            .chat(&[Message::user("hello")], &[], "", Path::new("."))
            .await
            .expect("chat stream");

        let chunks = collect_chunks(stream).await;
        assert!(matches!(&chunks[0], StreamChunk::TextDelta(text) if text == "Hello "));
        assert!(matches!(&chunks[1], StreamChunk::TextDelta(text) if text == "world"));
        assert!(matches!(
            &chunks[2],
            StreamChunk::Usage {
                input_tokens: 10,
                output_tokens: 5,
                ..
            }
        ));
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
    }

    #[tokio::test]
    async fn mimo_request_body_disables_thinking_by_default() {
        // enable_thinking defaults to false and MiMo accepts "disabled" (the
        // official quickstart sends it by default): the wire must carry
        // thinking.type = "disabled". max_tokens is the 8192 default at the
        // factory (thinking off → no factory floor), but the capability clamp
        // in chat() raises it to 131072 anyway — mimo-v2.6 is a 128K-class
        // model.
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        )
        .to_string();
        let base_url = spawn_sse_server_with_body(body, 200, |request, request_body| {
            assert_eq!(request.url(), "/chat/completions");
            let parsed: serde_json::Value =
                serde_json::from_str(request_body).expect("request body is JSON");
            assert_eq!(
                parsed["thinking"],
                serde_json::json!({ "type": "disabled" }),
                "body: {request_body}"
            );
            // MiMo wire format: the output cap travels as
            // max_completion_tokens (documented; max_tokens is not), and
            // stream_options is omitted (undocumented — the final usage
            // chunk arrives natively). max_completion_tokens is the 8192
            // default at the factory (thinking off → no factory floor), but
            // the capability clamp in chat() raises it to 131072 anyway —
            // mimo-v2.6 is a 128K-class model.
            assert_eq!(
                parsed["max_completion_tokens"], 131_072,
                "body: {request_body}"
            );
            assert!(parsed.get("max_tokens").is_none(), "body: {request_body}");
            assert!(
                parsed.get("stream_options").is_none(),
                "body: {request_body}"
            );
            assert_eq!(parsed["model"], "mimo-v2.6-pro", "body: {request_body}");
        });

        let mut config = NcaConfig::default();
        config.provider.mimo.api_key = Some("mimo-test-key".into());
        config.provider.mimo.base_url = base_url;

        let provider =
            build_provider_for(&config, ProviderKind::Mimo).expect("build mimo provider");
        let stream = provider
            .chat(
                &[Message::user("hello")],
                &[],
                "", // fall back to the provider's configured model
                Path::new("."),
            )
            .await
            .expect("chat stream");

        let chunks = collect_chunks(stream).await;
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
    }

    #[tokio::test]
    async fn mimo_request_body_enables_thinking_and_floors_max_tokens() {
        // enable_thinking = true: the factory must send thinking.type =
        // "enabled" and floor the 8192 default max_tokens at 131072 —
        // reasoning shares the output cap, so mid-reasoning truncation is a
        // risk whenever thinking is on.
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        )
        .to_string();
        let base_url = spawn_sse_server_with_body(body, 200, |request, request_body| {
            assert_eq!(request.url(), "/chat/completions");
            let parsed: serde_json::Value =
                serde_json::from_str(request_body).expect("request body is JSON");
            assert_eq!(
                parsed["thinking"],
                serde_json::json!({ "type": "enabled" }),
                "body: {request_body}"
            );
            // Default 8192 max_tokens must be floored to 131072 end-to-end,
            // carried as max_completion_tokens (no max_tokens, no
            // stream_options — the MiMo wire format).
            assert_eq!(
                parsed["max_completion_tokens"], 131_072,
                "body: {request_body}"
            );
            assert!(parsed.get("max_tokens").is_none(), "body: {request_body}");
            assert!(
                parsed.get("stream_options").is_none(),
                "body: {request_body}"
            );
            assert_eq!(parsed["model"], "mimo-v2.6-pro", "body: {request_body}");
        });

        let mut config = NcaConfig::default();
        config.provider.mimo.api_key = Some("mimo-test-key".into());
        config.provider.mimo.base_url = base_url;
        config.model.enable_thinking = true;
        // config.model.max_tokens stays at the 8192 default.

        let provider =
            build_provider_for(&config, ProviderKind::Mimo).expect("build mimo provider");
        let stream = provider
            .chat(
                &[Message::user("hello")],
                &[],
                "", // fall back to the provider's configured model
                Path::new("."),
            )
            .await
            .expect("chat stream");

        let chunks = collect_chunks(stream).await;
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
    }

    #[tokio::test]
    async fn wire_strips_reasoning_content_when_thinking_disabled() {
        // MiMo streams DeepSeek-style reasoning_content, and this provider is
        // built via `from_config` WITHOUT `.with_thinking` — `thinking` is
        // None, which StripUnlessThinking treats as thinking-disabled, so
        // the wire body must OMIT reasoning_content (context savings; the
        // pass-back-preserving branch only activates once
        // thinking = {"type": "enabled"}). Since the 10-24 contract fix the
        // decision is made at wire serialization — prepare is a no-op that
        // never rewrites history — so the assertion targets the request body.
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        )
        .to_string();
        let base_url = spawn_sse_server_with_body(body, 200, |_request, request_body| {
            assert!(
                !request_body.contains("reasoning_content"),
                "thinking-disabled MiMo must not send reasoning_content: {request_body}"
            );
        });

        let mut config = NcaConfig::default();
        config.provider.mimo.api_key = Some("test".into());
        config.provider.mimo.base_url = base_url;
        let provider = OpenAiCompatProvider::from_config(
            &config.provider.mimo,
            config.model.max_tokens,
            MIMO_PROFILE,
            reqwest::header::HeaderMap::new(),
        )
        .expect("provider");

        let mut assistant = Message::assistant("response").with_reasoning("thinking...".into());
        assistant.tool_calls = Some(vec![nca_common::message::MessageToolCall {
            id: "call_1".into(),
            name: "read".into(),
            arguments: serde_json::json!({}),
        }]);
        let mut messages = vec![
            Message::user("hello"),
            assistant,
            Message::tool("call_1", "result"),
        ];
        let history = messages.clone();
        provider
            .prepare_messages_for_request(&mut messages, Path::new("."))
            .await
            .expect("prepare");
        assert_eq!(messages, history, "prepare must not rewrite history");

        let stream = provider
            .chat(&messages, &[], "", Path::new("."))
            .await
            .expect("chat stream");
        let chunks = collect_chunks(stream).await;
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
    }

    #[tokio::test]
    async fn wire_keeps_reasoning_content_when_thinking_enabled() {
        // Mirror of `wire_strips_reasoning_content_when_thinking_disabled` for
        // the thinking-enabled branch: MiMo's API requires reasoning_content
        // pass-back on assistant messages carrying tool_calls once
        // thinking = {"type": "enabled"} (missing it is a hard HTTP 400), so
        // StripUnlessThinking must KEEP it on the wire.
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        )
        .to_string();
        let base_url = spawn_sse_server_with_body(body, 200, |_request, request_body| {
            let parsed: serde_json::Value =
                serde_json::from_str(request_body).expect("request body is JSON");
            let assistants: Vec<&serde_json::Value> = parsed["messages"]
                .as_array()
                .expect("messages")
                .iter()
                .filter(|m| m["role"] == "assistant")
                .collect();
            assert!(!assistants.is_empty());
            assert!(
                assistants
                    .iter()
                    .all(|m| m["reasoning_content"].as_str() == Some("thinking...")),
                "thinking-enabled MiMo must pass reasoning_content back verbatim: {request_body}"
            );
        });

        let mut config = NcaConfig::default();
        config.provider.mimo.api_key = Some("test".into());
        config.provider.mimo.base_url = base_url;
        let provider = OpenAiCompatProvider::from_config(
            &config.provider.mimo,
            config.model.max_tokens,
            MIMO_PROFILE,
            reqwest::header::HeaderMap::new(),
        )
        .expect("provider")
        .with_thinking(serde_json::json!({ "type": "enabled" }));

        let mut assistant = Message::assistant("response").with_reasoning("thinking...".into());
        assistant.tool_calls = Some(vec![nca_common::message::MessageToolCall {
            id: "call_1".into(),
            name: "read".into(),
            arguments: serde_json::json!({}),
        }]);
        let messages = vec![
            Message::user("hello"),
            assistant,
            Message::tool("call_1", "result"),
        ];

        let stream = provider
            .chat(&messages, &[], "", Path::new("."))
            .await
            .expect("chat stream");
        let chunks = collect_chunks(stream).await;
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
    }

    #[tokio::test]
    async fn mimo_keepalive_body_merges_thinking_override() {
        // keepalive_ping must merge the same `thinking` override as chat
        // (max_tokens is hardcoded to 1 there). Pins the second merge site in
        // OpenAiCompatProvider so a future refactor can't drop it in one path
        // while keeping it in the other.
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":1}}\n\n",
            "data: [DONE]\n\n"
        )
        .to_string();
        let base_url = spawn_sse_server_with_body(body, 200, |request, request_body| {
            assert_eq!(request.url(), "/chat/completions");
            let parsed: serde_json::Value =
                serde_json::from_str(request_body).expect("request body is JSON");
            assert_eq!(
                parsed["thinking"],
                serde_json::json!({ "type": "disabled" }),
                "body: {request_body}"
            );
            // Keepalive hardcodes max_tokens = 1 to minimise output cost —
            // renamed to max_completion_tokens on the MiMo wire (1 is within
            // the documented [1, 131072] range), with stream_options omitted.
            assert_eq!(parsed["max_completion_tokens"], 1, "body: {request_body}");
            assert!(parsed.get("max_tokens").is_none(), "body: {request_body}");
            assert!(
                parsed.get("stream_options").is_none(),
                "body: {request_body}"
            );
            assert_eq!(parsed["model"], "mimo-v2.6-pro", "body: {request_body}");
        });

        let mut config = NcaConfig::default();
        config.provider.mimo.api_key = Some("mimo-test-key".into());
        config.provider.mimo.base_url = base_url;

        let provider =
            build_provider_for(&config, ProviderKind::Mimo).expect("build mimo provider");
        let snapshot = KeepaliveSnapshot {
            messages: vec![Message::user("hello")],
            tools: vec![],
            model: "mimo-v2.6-pro".into(),
            workspace_root: PathBuf::from("."),
        };
        let usage = provider
            .keepalive_ping(&snapshot)
            .await
            .expect("keepalive ping");
        assert_eq!(usage.input_tokens, 100);
    }
}
