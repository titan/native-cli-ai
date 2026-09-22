//! Tests for the Xiaomi MiMo provider (served by `OpenAiCompatProvider`).

#[cfg(test)]
mod tests {
    use crate::cache_keepalive::KeepaliveSnapshot;
    use crate::provider::factory::build_provider_for;
    use crate::provider::openai_compat::{CompatProfile, OpenAiCompatProvider};
    use crate::provider::test_support::{collect_chunks, spawn_sse_server_with_body};
    use crate::provider::{Provider, StreamChunk};
    use nca_common::config::{NcaConfig, ProviderKind};
    use nca_common::message::Message;
    use std::path::{Path, PathBuf};

    const MIMO_PROFILE: CompatProfile = CompatProfile {
        name: "mimo",
        endpoint_suffix: "chat/completions",
        strip_reasoning: true,
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
            assert_eq!(parsed["max_tokens"], 131_072, "body: {request_body}");
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
            // Default 8192 max_tokens must be floored to 131072 end-to-end.
            assert_eq!(parsed["max_tokens"], 131_072, "body: {request_body}");
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
    async fn prepare_strips_reasoning_content() {
        // MiMo streams DeepSeek-style reasoning_content (response-only
        // signal): prepare_messages_for_request must strip it before
        // re-upload, mirroring the DeepSeek profile.
        let mut config = NcaConfig::default();
        config.provider.mimo.api_key = Some("test".into());
        let provider = OpenAiCompatProvider::from_config(
            &config.provider.mimo,
            config.model.max_tokens,
            MIMO_PROFILE,
            reqwest::header::HeaderMap::new(),
        )
        .expect("provider");

        let mut messages = vec![
            Message::user("hello"),
            Message::assistant("response").with_reasoning("thinking...".into()),
            Message::tool("call_1", "result"),
        ];

        provider
            .prepare_messages_for_request(&mut messages, Path::new("."))
            .await
            .expect("prepare");

        // reasoning_content must be stripped for MiMo
        assert!(messages[1].reasoning_content.is_none());
        // Other fields unchanged
        assert_eq!(messages[1].content.to_summary_text(), "response");
        // User/tool messages unaffected
        assert!(messages[0].reasoning_content.is_none());
        assert!(messages[2].reasoning_content.is_none());
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
            // Keepalive hardcodes max_tokens = 1 to minimise output cost.
            assert_eq!(parsed["max_tokens"], 1, "body: {request_body}");
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
