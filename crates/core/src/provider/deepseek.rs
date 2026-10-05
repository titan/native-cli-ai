//! Tests for the DeepSeek provider (now served by `OpenAiCompatProvider` with
//! `ReasoningWirePolicy::StripUnlessThinking` + model-name thinking detection).

#[cfg(test)]
mod tests {
    use crate::provider::openai_compat::{
        CompatProfile, OpenAiCompatProvider, ReasoningWirePolicy,
    };
    use crate::provider::test_support::{collect_chunks, spawn_sse_server};
    use crate::provider::{Provider, StreamChunk};
    use nca_common::config::NcaConfig;
    use nca_common::message::Message;
    use nca_common::tool::ToolDefinition;
    use serde_json::json;
    use std::path::Path;

    const DEEPSEEK_PROFILE: CompatProfile = CompatProfile {
        name: "deepseek",
        endpoint_suffix: "chat/completions",
        reasoning: ReasoningWirePolicy::StripUnlessThinking,
        thinking_model_marker: Some("reasoner"),
        send_stream_options: true,
        use_max_completion_tokens: false,
    };

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

        let provider = OpenAiCompatProvider::from_config(
            &config.provider.deepseek,
            config.model.max_tokens,
            DEEPSEEK_PROFILE,
            reqwest::header::HeaderMap::new(),
        )
        .expect("provider");
        let stream = provider
            .chat(
                &[Message::user("hello")],
                &[ToolDefinition {
                    timeout_ms: None,
                    name: "lookup".into(),
                    description: "Lookup a path".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        }
                    }),
                }],
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

    #[tokio::test]
    async fn prepare_keeps_reasoning_content_for_thinking_models() {
        // DeepSeek's thinking-mode API (2026-10 restructure) REQUIRES
        // reasoning_content pass-back on assistant history — omitting it is a
        // 400 ("Then `reasoning_content` in the thinking mode must be passed
        // back to the API"), killing resumed/multi-round tool sessions whose
        // history was rebuilt by replay or stripped by an earlier turn.
        let mut config = NcaConfig::default();
        config.provider.deepseek.api_key = Some("test".into());
        config.provider.deepseek.model = "deepseek-reasoner".into();
        let provider = OpenAiCompatProvider::from_config(
            &config.provider.deepseek,
            config.model.max_tokens,
            DEEPSEEK_PROFILE,
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

        // thinking model (marker hit): reasoning_content is preserved verbatim
        assert_eq!(
            messages[1].reasoning_content.as_deref(),
            Some("thinking...")
        );
        assert_eq!(messages[1].content.to_summary_text(), "response");
    }

    #[tokio::test]
    async fn prepare_strips_reasoning_content_for_non_thinking_models() {
        // deepseek-chat: the classic contract still holds — reasoning_content
        // is output-only there, and stripping saves ~500 tokens/turn and
        // protects the prefix cache (see 3ed4491).
        let mut config = NcaConfig::default();
        config.provider.deepseek.api_key = Some("test".into());
        config.provider.deepseek.model = "deepseek-chat".into();
        let provider = OpenAiCompatProvider::from_config(
            &config.provider.deepseek,
            config.model.max_tokens,
            DEEPSEEK_PROFILE,
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

        // non-thinking model: reasoning_content must be stripped
        assert!(messages[1].reasoning_content.is_none());
        assert_eq!(messages[1].content.to_summary_text(), "response");
        assert!(messages[0].reasoning_content.is_none());
        assert!(messages[2].reasoning_content.is_none());
    }
}
