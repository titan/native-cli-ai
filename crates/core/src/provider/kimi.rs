use std::path::Path;
use std::time::Duration;

use nca_common::config::{KimiConfig, NcaConfig};
use nca_common::message::Message;
use nca_common::tool::ToolDefinition;
use reqwest::header::{HeaderMap, HeaderValue};

use super::anthropic_compat::{anthropic_request_body, map_provider_error, spawn_anthropic_stream};
use super::{Provider, ProviderError, StreamChunk};

/// LLM provider for Moonshot AI's Kimi for Coding endpoint.
///
/// Wire-compatible with the Anthropic Messages API (`/v1/messages`):
/// uses `x-api-key` + `anthropic-version` headers and the same SSE event
/// stream, so it reuses [`anthropic_compat`] helpers for request body
/// construction and stream parsing.
///
/// Default model is `k3` (Kimi K3 flagship, 1M context, 131K output).
pub struct KimiProvider {
    client: reqwest::Client,
    config: KimiConfig,
    max_tokens: u32,
}

impl KimiProvider {
    pub fn from_config(config: &NcaConfig) -> Result<Self, ProviderError> {
        let kimi = config.provider.kimi.clone();
        let api_key = kimi.resolve_api_key().ok_or_else(|| {
            ProviderError::Configuration(format!(
                "missing Kimi API key; set {} or provide `provider.kimi.api_key` in config",
                kimi.api_key_env
            ))
        })?;

        let mut headers = HeaderMap::new();
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&api_key).map_err(|err| {
                ProviderError::Configuration(format!(
                    "failed to build Kimi x-api-key header: {err}"
                ))
            })?,
        );
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .connect_timeout(Duration::from_secs(30))
            .build()
            .map_err(|err| {
                ProviderError::Configuration(format!("failed to build HTTP client: {err}"))
            })?;

        Ok(Self {
            client,
            config: kimi,
            max_tokens: config.model.max_tokens,
        })
    }

    fn endpoint(&self) -> String {
        format!("{}/v1/messages", self.config.base_url.trim_end_matches('/'))
    }
}

#[async_trait::async_trait]
impl Provider for KimiProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        model: &str,
        workspace_root: &Path,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError> {
        let model = if model.is_empty() {
            self.config.model.clone()
        } else {
            model.to_string()
        };

        // Chat-only capability clamp on the final model string; KimiProvider
        // has no keepalive path. Default model "k3" floors to 131072.
        let max_tokens = nca_common::model_limits::clamp_max_tokens(&model, self.max_tokens);

        let body = anthropic_request_body(
            messages,
            tools,
            &model,
            max_tokens,
            self.config.temperature,
            workspace_root,
        )?;

        let response = self
            .client
            .post(self.endpoint())
            .json(&body)
            .send()
            .await
            .map_err(|err| {
                let chain = super::format_error_chain(&err);
                tracing::error!(
                    provider = "kimi",
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
            return Err(map_provider_error(status, body_text));
        }

        Ok(spawn_anthropic_stream(response, "kimi"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::test_support::{collect_chunks, spawn_sse_server};
    use nca_common::message::Message;
    use nca_common::tool::ToolDefinition;
    use serde_json::json;

    #[tokio::test]
    async fn kimi_provider_streams_text_and_usage() {
        let body = concat!(
            "event: message_start\n",
            "data: {\"message\":{\"usage\":{\"input_tokens\":7}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello from Kimi K3\"}}\n\n",
            "event: message_delta\n",
            "data: {\"usage\":{\"output_tokens\":4}}\n\n"
        )
        .to_string();
        let base_url = spawn_sse_server(body, 200, |request| {
            assert_eq!(request.url(), "/v1/messages");
            assert!(
                request
                    .headers()
                    .iter()
                    .any(|header| header.field.equiv("x-api-key")
                        && header.value.as_str() == "kimi-test-key")
            );
            assert!(
                request
                    .headers()
                    .iter()
                    .any(|header| header.field.equiv("anthropic-version"))
            );
        });

        let mut config = NcaConfig::default();
        config.provider.kimi.api_key = Some("kimi-test-key".into());
        config.provider.kimi.base_url = base_url;

        let provider = KimiProvider::from_config(&config).expect("provider");
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
        assert!(matches!(&chunks[0], StreamChunk::TextDelta(text) if text == "Hello from Kimi K3"));
        assert!(matches!(
            &chunks[1],
            StreamChunk::Usage {
                input_tokens: 7,
                output_tokens: 4,
                ..
            }
        ));
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
    }

    #[tokio::test]
    async fn kimi_provider_streams_tool_use() {
        let body = concat!(
            "event: message_start\n",
            "data: {\"message\":{\"usage\":{\"input_tokens\":0}}}\n\n",
            "event: content_block_start\n",
            "data: {\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_kimi_1\",\"name\":\"read\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"src\\\"}\"}}\n\n",
            "event: content_block_stop\n",
            "data: {}\n\n",
            "event: message_delta\n",
            "data: {\"usage\":{\"output_tokens\":3}}\n\n"
        )
        .to_string();
        let base_url = spawn_sse_server(body, 200, |_| {});

        let mut config = NcaConfig::default();
        config.provider.kimi.api_key = Some("kimi-test-key".into());
        config.provider.kimi.base_url = base_url;

        let provider = KimiProvider::from_config(&config).expect("provider");
        let stream = provider
            .chat(
                &[Message::user("read src")],
                &[ToolDefinition {
                    timeout_ms: None,
                    name: "read".into(),
                    description: "Read a file".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {"path": {"type": "string"}}
                    }),
                }],
                "",
                std::path::Path::new("."),
            )
            .await
            .expect("chat stream");

        let chunks = collect_chunks(stream).await;
        assert!(
            matches!(&chunks[0], StreamChunk::ToolUse(call) if call.id == "toolu_kimi_1" && call.name == "read" && call.input == json!({"path":"src"}))
        );
        assert!(matches!(chunks.last(), Some(StreamChunk::Done)));
    }
}
