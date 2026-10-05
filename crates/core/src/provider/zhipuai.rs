//! Tests for the ZhipuAI provider (now served by `OpenAiCompatProvider`).

#[cfg(test)]
mod tests {
    use crate::cache_keepalive::KeepaliveSnapshot;
    use crate::provider::factory::build_provider_for;
    use crate::provider::openai_compat::{
        CompatProfile, OpenAiCompatProvider, ReasoningWirePolicy,
    };
    use crate::provider::test_support::{
        collect_chunks, spawn_sse_server, spawn_sse_server_with_body,
    };
    use crate::provider::{Provider, StreamChunk};
    use nca_common::config::{NcaConfig, ProviderKind};
    use nca_common::message::Message;
    use std::path::PathBuf;

    const ZHIPUAI_PROFILE: CompatProfile = CompatProfile {
        name: "zhipuai",
        endpoint_suffix: "chat/completions",
        reasoning: ReasoningWirePolicy::Strip,
        thinking_model_marker: None,
        send_stream_options: true,
        use_max_completion_tokens: false,
    };

    #[tokio::test]
    async fn zhipuai_provider_streams_text_and_usage() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"你好 \"},\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"世界\"},\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n",
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
            assert_eq!(auth.value.as_str(), "Bearer zhipuai-test-key");
        });

        let mut config = NcaConfig::default();
        config.provider.zhipuai.api_key = Some("zhipuai-test-key".into());
        config.provider.zhipuai.base_url = base_url;

        let provider = OpenAiCompatProvider::from_config(
            &config.provider.zhipuai,
            config.model.max_tokens,
            ZHIPUAI_PROFILE,
            reqwest::header::HeaderMap::new(),
        )
        .expect("provider");
        let stream = provider
            .chat(
                &[Message::user("hello")],
                &[],
                "",
                std::path::Path::new("."),
            )
            .await
            .expect("chat stream");

        let chunks = collect_chunks(stream).await;
        assert!(matches!(&chunks[0], StreamChunk::TextDelta(text) if text == "你好 "));
        assert!(matches!(&chunks[1], StreamChunk::TextDelta(text) if text == "世界"));
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
    async fn zhipuai_prepare_strips_reasoning_content() {
        // GLM thinking models (glm-5.x; 5.3 cannot disable thinking) stream
        // `reasoning_content`, but the API rejects the field when it appears
        // inside assistant history on a subsequent request — a plain 400 that
        // kills multi-turn tool loops (subagent runs) from the second turn
        // onward. The profile must strip it before re-upload.
        let mut config = NcaConfig::default();
        config.provider.zhipuai.api_key = Some("zhipuai-test-key".into());
        let provider = OpenAiCompatProvider::from_config(
            &config.provider.zhipuai,
            config.model.max_tokens,
            ZHIPUAI_PROFILE,
            reqwest::header::HeaderMap::new(),
        )
        .expect("provider");

        let mut messages = vec![
            Message::user("hello"),
            Message::assistant("response").with_reasoning("thinking...".into()),
            Message::tool("call_1", "result"),
        ];

        provider
            .prepare_messages_for_request(&mut messages, std::path::Path::new("."))
            .await
            .expect("prepare");

        // reasoning_content must be stripped for ZhipuAI
        assert!(messages[1].reasoning_content.is_none());
        // Other fields unchanged
        assert_eq!(messages[1].content.to_summary_text(), "response");
        // User/tool messages unaffected
        assert!(messages[0].reasoning_content.is_none());
        assert!(messages[2].reasoning_content.is_none());
    }

    #[tokio::test]
    async fn zhipuai_request_body_carries_no_reasoning_content() {
        // Wire-level guard: even when history contains an assistant message
        // with reasoning_content recorded, the serialized request body must
        // not carry the field (this is the exact payload GLM 400s on).
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
            let text = request_body;
            assert!(
                !text.contains("reasoning_content"),
                "request body must not echo reasoning_content: {text}"
            );
            assert!(parsed["messages"][1]["role"] == "assistant", "body: {text}");
        });

        let mut config = NcaConfig::default();
        config.provider.zhipuai.api_key = Some("zhipuai-test-key".into());
        config.provider.zhipuai.base_url = base_url;

        let provider =
            build_provider_for(&config, ProviderKind::ZhipuAI).expect("build zhipuai provider");
        // Mirror the driver contract: prepare_messages_for_request runs on the
        // history BEFORE chat() (agent_driver.rs calls it ahead of every
        // request). The wire-level guarantee is "prepared history never
        // carries reasoning_content", not "chat() re-normalizes internally".
        let mut history = vec![
            Message::user("hello"),
            Message::assistant("prior turn").with_reasoning("secret chain".into()),
            Message::tool("call_1", "result"),
            Message::user("continue"),
        ];
        provider
            .prepare_messages_for_request(&mut history, std::path::Path::new("."))
            .await
            .expect("prepare");
        let stream = provider
            .chat(&history, &[], "", std::path::Path::new("."))
            .await
            .expect("chat stream");

        let chunks = collect_chunks(stream).await;
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
    }

    #[tokio::test]
    async fn zhipuai_reasoning_only_truncation_surfaces_finish_reason() {
        // GLM-5.3 cannot disable thinking: when max_tokens is exhausted
        // mid-reasoning the stream carries only reasoning_content deltas and
        // ends with finish_reason="length" and an empty content. The stream
        // must surface Finish{reason:"length"} so the agent loop can fail fast
        // instead of misreading this as a retryable "empty response".
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"let me think\"},\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"... and think\"},\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"length\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9000,\"completion_tokens\":8192}}\n\n",
            "data: [DONE]\n\n"
        )
        .to_string();
        let base_url = spawn_sse_server(body, 200, |_| {});

        let mut config = NcaConfig::default();
        config.provider.zhipuai.api_key = Some("zhipuai-test-key".into());
        config.provider.zhipuai.base_url = base_url;

        let provider = OpenAiCompatProvider::from_config(
            &config.provider.zhipuai,
            config.model.max_tokens,
            ZHIPUAI_PROFILE,
            reqwest::header::HeaderMap::new(),
        )
        .expect("provider");
        let stream = provider
            .chat(
                &[Message::user("refactor this module")],
                &[],
                "",
                std::path::Path::new("."),
            )
            .await
            .expect("chat stream");

        let chunks = collect_chunks(stream).await;
        assert!(
            !chunks
                .iter()
                .any(|c| matches!(c, StreamChunk::TextDelta(_))),
            "truncated reasoning-only stream must carry no content deltas"
        );
        assert!(matches!(
            &chunks[chunks.len() - 2],
            StreamChunk::Finish { reason } if reason == "length"
        ));
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
    }

    #[tokio::test]
    async fn zhipuai_request_body_disables_thinking_for_glm_5_2() {
        // enable_thinking defaults to false: a model that accepts
        // "disabled" must get it on the wire so the GLM server does not
        // apply its own default (thinking ON, which truncates mid-reasoning
        // under small max_tokens values).
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
            // Policy layering: the factory's thinking floor leaves 8192
            // untouched (thinking off, glm-5.2 not thinking-locked), but the
            // capability clamp in chat() raises it anyway — glm-5.2's output
            // window is 128K, so the 128K-class floor applies regardless of
            // the thinking toggle.
            assert_eq!(parsed["max_tokens"], 131_072, "body: {request_body}");
            assert_eq!(parsed["model"], "glm-5.2", "body: {request_body}");
        });

        let mut config = NcaConfig::default();
        config.provider.zhipuai.api_key = Some("zhipuai-test-key".into());
        config.provider.zhipuai.base_url = base_url;
        config.provider.zhipuai.model = "glm-5.2".into();

        let provider =
            build_provider_for(&config, ProviderKind::ZhipuAI).expect("build zhipuai provider");
        let stream = provider
            .chat(
                &[Message::user("hello")],
                &[],
                "", // fall back to the provider's configured model
                std::path::Path::new("."),
            )
            .await
            .expect("chat stream");

        let chunks = collect_chunks(stream).await;
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
    }

    #[tokio::test]
    async fn zhipuai_request_body_pins_thinking_locked_contract_for_glm_5_3() {
        // GLM-5.3 rejects "disabled" — the factory must send "enabled" and
        // floor the 8192 default max_tokens at 131072 end-to-end.
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
            // model.max_tokens is the 8192 default; the thinking-locked floor
            // must raise it to 131072 before it reaches the wire.
            assert_eq!(parsed["max_tokens"], 131_072, "body: {request_body}");
            assert_eq!(parsed["model"], "glm-5.3", "body: {request_body}");
        });

        let mut config = NcaConfig::default();
        config.provider.zhipuai.api_key = Some("zhipuai-test-key".into());
        config.provider.zhipuai.base_url = base_url;
        config.provider.zhipuai.model = "glm-5.3".into();

        let provider =
            build_provider_for(&config, ProviderKind::ZhipuAI).expect("build zhipuai provider");
        let stream = provider
            .chat(
                &[Message::user("hello")],
                &[],
                "", // fall back to the provider's configured model
                std::path::Path::new("."),
            )
            .await
            .expect("chat stream");

        let chunks = collect_chunks(stream).await;
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
    }

    #[tokio::test]
    async fn zhipuai_wire_sends_enabled_and_floors_when_thinking_on() {
        // enable_thinking = true on a GLM-5.x model that is NOT thinking-locked
        // (glm-5.2): the factory must still send thinking.type="enabled" and
        // floor the 8192 default max_tokens at 131072. Reasoning shares the
        // output cap, so mid-reasoning truncation is a risk whenever thinking
        // is on — even when the model could technically accept "disabled".
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
            // Default 8192 max_tokens must be floored to 131072 end-to-end.
            assert_eq!(parsed["max_tokens"], 131_072, "body: {request_body}");
            assert_eq!(parsed["model"], "glm-5.2", "body: {request_body}");
        });

        let mut config = NcaConfig::default();
        config.provider.zhipuai.api_key = Some("zhipuai-test-key".into());
        config.provider.zhipuai.base_url = base_url;
        config.provider.zhipuai.model = "glm-5.2".into();
        config.model.enable_thinking = true;
        // config.model.max_tokens stays at the 8192 default.

        let provider =
            build_provider_for(&config, ProviderKind::ZhipuAI).expect("build zhipuai provider");
        let stream = provider
            .chat(
                &[Message::user("hello")],
                &[],
                "", // fall back to the provider's configured model
                std::path::Path::new("."),
            )
            .await
            .expect("chat stream");

        let chunks = collect_chunks(stream).await;
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
    }

    #[tokio::test]
    async fn non_zhipuai_providers_emit_no_thinking_field() {
        // The `thinking` field is ZhipuAI-specific wiring: OpenAiCompatProvider
        // defaults `thinking` to None and only the factory's ZhipuAI arm calls
        // with_thinking. Even with the config-level thinking flag on, an OpenAI
        // provider must send the plain OpenAI-compatible body.
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        )
        .to_string();
        let base_url = spawn_sse_server_with_body(body, 200, |request, request_body| {
            assert_eq!(request.url(), "/v1/chat/completions");
            let parsed: serde_json::Value =
                serde_json::from_str(request_body).expect("request body is JSON");
            assert!(
                parsed.get("thinking").is_none(),
                "OpenAI body must not carry a thinking field: {request_body}"
            );
            assert_eq!(parsed["model"], "gpt-4o", "body: {request_body}");
            assert_eq!(parsed["max_tokens"], 8_192, "body: {request_body}");
        });

        let mut config = NcaConfig::default();
        config.provider.openai.api_key = Some("openai-test-key".into());
        config.provider.openai.base_url = base_url;
        config.provider.openai.model = "gpt-4o".into();
        // Adversarial: a globally-enabled thinking flag must not leak a
        // `thinking` key into non-ZhipuAI providers.
        config.model.enable_thinking = true;

        let provider =
            build_provider_for(&config, ProviderKind::OpenAi).expect("build openai provider");
        let stream = provider
            .chat(
                &[Message::user("hello")],
                &[],
                "", // fall back to the provider's configured model
                std::path::Path::new("."),
            )
            .await
            .expect("chat stream");

        let chunks = collect_chunks(stream).await;
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
    }

    #[tokio::test]
    async fn openai_wire_caps_max_tokens_to_model_output_window() {
        // Protective cap end-to-end: gpt-4o's output window is 16384, so a
        // configured 131072 must be clamped down before it reaches the wire
        // (the OpenAI API rejects oversize max_tokens with a hard 400).
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        )
        .to_string();
        let base_url = spawn_sse_server_with_body(body, 200, |request, request_body| {
            assert_eq!(request.url(), "/v1/chat/completions");
            let parsed: serde_json::Value =
                serde_json::from_str(request_body).expect("request body is JSON");
            assert_eq!(parsed["model"], "gpt-4o", "body: {request_body}");
            assert_eq!(parsed["max_tokens"], 16_384, "body: {request_body}");
        });

        let mut config = NcaConfig::default();
        config.provider.openai.api_key = Some("openai-test-key".into());
        config.provider.openai.base_url = base_url;
        config.provider.openai.model = "gpt-4o".into();
        config.model.max_tokens = 131_072;

        let provider =
            build_provider_for(&config, ProviderKind::OpenAi).expect("build openai provider");
        let stream = provider
            .chat(
                &[Message::user("hello")],
                &[],
                "", // fall back to the provider's configured model
                std::path::Path::new("."),
            )
            .await
            .expect("chat stream");

        let chunks = collect_chunks(stream).await;
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
    }

    #[tokio::test]
    async fn zhipuai_keepalive_body_merges_thinking_override() {
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
                serde_json::json!({ "type": "enabled" }),
                "body: {request_body}"
            );
            // Keepalive hardcodes max_tokens = 1 to minimise output cost.
            assert_eq!(parsed["max_tokens"], 1, "body: {request_body}");
            assert_eq!(parsed["model"], "glm-5.3", "body: {request_body}");
            assert!(
                !request_body.contains("reasoning_content"),
                "keepalive must strip reasoning_content (GLM 400s the echoed field): {request_body}"
            );
        });

        let mut config = NcaConfig::default();
        config.provider.zhipuai.api_key = Some("zhipuai-test-key".into());
        config.provider.zhipuai.base_url = base_url;
        config.provider.zhipuai.model = "glm-5.3".into();

        let provider =
            build_provider_for(&config, ProviderKind::ZhipuAI).expect("build zhipuai provider");
        let snapshot = KeepaliveSnapshot {
            // The driver clones this at the tool pause — BEFORE the next
            // turn's prepare — so a fresh assistant reasoning_content is
            // present here. keepalive_ping must strip it itself (GLM 400s
            // the echoed field exactly like the main chat path).
            messages: vec![
                Message::user("hello"),
                Message::assistant("thinking turn")
                    .with_reasoning("must not reach the wire".into()),
            ],
            tools: vec![],
            model: "glm-5.3".into(),
            workspace_root: PathBuf::from("."),
        };
        let usage = provider
            .keepalive_ping(&snapshot)
            .await
            .expect("keepalive ping");
        assert_eq!(usage.input_tokens, 100);
    }
}
