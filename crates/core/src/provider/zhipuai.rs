//! Tests for the ZhipuAI provider (now served by `OpenAiCompatProvider`).

#[cfg(test)]
mod tests {
    use crate::cache_keepalive::KeepaliveSnapshot;
    use crate::provider::factory::build_provider_for;
    use crate::provider::openai_compat::{CompatProfile, OpenAiCompatProvider};
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
        strip_reasoning: false,
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
        // under the 8192 default max_tokens).
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
            // Thinking off + not thinking-locked: configured value stands.
            assert_eq!(parsed["max_tokens"], 8_192, "body: {request_body}");
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
        });

        let mut config = NcaConfig::default();
        config.provider.zhipuai.api_key = Some("zhipuai-test-key".into());
        config.provider.zhipuai.base_url = base_url;
        config.provider.zhipuai.model = "glm-5.3".into();

        let provider =
            build_provider_for(&config, ProviderKind::ZhipuAI).expect("build zhipuai provider");
        let snapshot = KeepaliveSnapshot {
            messages: vec![Message::user("hello")],
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
