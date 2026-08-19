use std::sync::Arc;

use nca_common::config::{NcaConfig, ProviderKind};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

use super::anthropic::AnthropicProvider;
use super::kimi::KimiProvider;
use super::minimax::MiniMaxProvider;
use super::openai_compat::{CompatProfile, OpenAiCompatProvider};
use super::{Provider, ProviderError};

const OPENAI_PROFILE: CompatProfile = CompatProfile {
    name: "OpenAI",
    endpoint_suffix: "v1/chat/completions",
    strip_reasoning: false,
};

const OPENROUTER_PROFILE: CompatProfile = CompatProfile {
    name: "OpenRouter",
    endpoint_suffix: "v1/chat/completions",
    strip_reasoning: false,
};

const ZHIPUAI_PROFILE: CompatProfile = CompatProfile {
    name: "ZhipuAI",
    endpoint_suffix: "chat/completions",
    strip_reasoning: false,
};

const DEEPSEEK_PROFILE: CompatProfile = CompatProfile {
    name: "DeepSeek",
    endpoint_suffix: "chat/completions",
    strip_reasoning: true,
};

/// Build the configured provider for the current workspace (uses `config.provider.default`).
pub fn build_provider(config: &NcaConfig) -> Result<Arc<dyn Provider>, ProviderError> {
    build_provider_for(config, config.provider.default)
}

/// Build a provider for a specific [`ProviderKind`], ignoring `config.provider.default`.
///
/// This is used when an agent profile or skill specifies a different provider than
/// the session default.
pub fn build_provider_for(
    config: &NcaConfig,
    kind: ProviderKind,
) -> Result<Arc<dyn Provider>, ProviderError> {
    match kind {
        ProviderKind::MiniMax => Ok(Arc::new(MiniMaxProvider::from_config(config)?)),
        ProviderKind::OpenRouter => {
            let mut extra = HeaderMap::new();
            if let Some(url) = &config.provider.openrouter.site_url {
                let _ = extra.insert(
                    HeaderName::from_static("http-referer"),
                    HeaderValue::from_str(url).unwrap(),
                );
            }
            if let Some(name) = &config.provider.openrouter.app_name {
                let _ = extra.insert(
                    HeaderName::from_static("x-title"),
                    HeaderValue::from_str(name).unwrap(),
                );
            }
            Ok(Arc::new(OpenAiCompatProvider::from_config(
                &config.provider.openrouter,
                config.model.max_tokens,
                OPENROUTER_PROFILE,
                extra,
            )?))
        }
        ProviderKind::Anthropic => Ok(Arc::new(AnthropicProvider::from_config(config)?)),
        ProviderKind::OpenAi => {
            let extra = HeaderMap::new();
            Ok(Arc::new(OpenAiCompatProvider::from_config(
                &config.provider.openai,
                config.model.max_tokens,
                OPENAI_PROFILE,
                extra,
            )?))
        }
        ProviderKind::ZhipuAI => {
            let extra = HeaderMap::new();
            // Check both model strings that can reach the request body.
            let models = format!(
                "{} {}",
                config.provider.zhipuai.model, config.model.default_model
            );
            let max_tokens = zhipuai_effective_max_tokens(&models, config.model.max_tokens);
            Ok(Arc::new(OpenAiCompatProvider::from_config(
                &config.provider.zhipuai,
                max_tokens,
                ZHIPUAI_PROFILE,
                extra,
            )?))
        }
        ProviderKind::DeepSeek => {
            let extra = HeaderMap::new();
            Ok(Arc::new(OpenAiCompatProvider::from_config(
                &config.provider.deepseek,
                config.model.max_tokens,
                DEEPSEEK_PROFILE,
                extra,
            )?))
        }
        ProviderKind::Kimi => Ok(Arc::new(KimiProvider::from_config(config)?)),
        ProviderKind::Custom => Ok(Arc::new(super::custom::CustomProvider::from_config(
            config,
        )?)),
    }
}

/// Effective `max_tokens` for the ZhipuAI provider.
///
/// GLM-5.3 cannot disable thinking (official docs: `thinking.type` only
/// supports `"enabled"`; requests with `"disabled"` are rejected), and the
/// reasoning budget shares the output cap — ZhipuAI's own coding examples use
/// `max_tokens: 65536`. With nca's global default (8192) the model exhausts the
/// cap mid-reasoning and returns an empty `content` with
/// `finish_reason: "length"`, which surfaces as a hard "empty response"
/// failure. Floor the cap for thinking-locked GLM models so the default
/// configuration works; explicitly larger values pass through untouched, and
/// models that allow disabling thinking (GLM-5.2 and earlier) keep the
/// configured value as-is.
fn zhipuai_effective_max_tokens(model: &str, configured: u32) -> u32 {
    const THINKING_LOCKED_FLOOR: u32 = 65_536;
    // Key on every model string that can end up in the request body: the
    // provider-side model ([provider.zhipuai].model) and the session default
    // ([model].default_model) are normally kept in sync, but a manual TOML can
    // diverge them — flooring on either avoids skipping the floor for the
    // model that actually gets sent.
    let is_thinking_locked = model.to_ascii_lowercase().contains("glm-5.3");
    if is_thinking_locked && configured < THINKING_LOCKED_FLOOR {
        tracing::warn!(
            model = %model,
            configured,
            floor = THINKING_LOCKED_FLOOR,
            "zhipuai model cannot disable thinking; raising max_tokens to avoid \
             mid-reasoning truncation (values >= the floor pass through; lower \
             values are always raised)"
        );
        return THINKING_LOCKED_FLOOR;
    }
    configured
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factory_builds_each_supported_provider_when_configured() {
        for kind in ProviderKind::ALL {
            let mut config = NcaConfig::default();
            config.provider.default = kind;
            match kind {
                ProviderKind::MiniMax => {
                    config.provider.minimax.api_key = Some("minimax-key".into());
                }
                ProviderKind::OpenAi => {
                    config.provider.openai.api_key = Some("openai-key".into());
                }
                ProviderKind::Anthropic => {
                    config.provider.anthropic.api_key = Some("anthropic-key".into());
                }
                ProviderKind::OpenRouter => {
                    config.provider.openrouter.api_key = Some("openrouter-key".into());
                }
                ProviderKind::ZhipuAI => {
                    config.provider.zhipuai.api_key = Some("zhipuai-key".into());
                }
                ProviderKind::DeepSeek => {
                    config.provider.deepseek.api_key = Some("deepseek-key".into());
                }
                ProviderKind::Kimi => {
                    config.provider.kimi.api_key = Some("kimi-key".into());
                }
                ProviderKind::Custom => {
                    config.provider.custom.api_key = Some("custom-key".into());
                    config.provider.custom.base_url = "http://localhost:9".into();
                }
            }

            let provider = build_provider(&config);
            assert!(
                provider.is_ok(),
                "expected provider {:?} to build, got {:?}",
                kind,
                provider.as_ref().err()
            );
        }
    }

    #[test]
    fn factory_fails_loudly_when_selected_provider_is_missing_credentials() {
        let mut config = NcaConfig::default();
        config.provider.default = ProviderKind::OpenAi;
        match build_provider(&config) {
            Ok(_) => panic!("missing credentials should fail"),
            Err(error) => {
                assert!(
                    matches!(error, ProviderError::Configuration(message) if message.contains("missing OpenAI API key"))
                );
            }
        }
    }

    #[test]
    fn build_provider_for_uses_explicit_kind_not_default() {
        let mut config = NcaConfig::default();
        // Default is deepseek, but we request openai explicitly
        config.provider.default = ProviderKind::MiniMax;
        config.provider.minimax.api_key = Some("minimax-key".into());
        config.provider.openai.api_key = Some("openai-key".into());

        // build_provider follows default (minimax)
        assert!(build_provider(&config).is_ok());

        // build_provider_for can override to openai
        let provider = build_provider_for(&config, ProviderKind::OpenAi);
        assert!(provider.is_ok(), "expected openai provider to build");
    }

    #[test]
    fn zhipuai_max_tokens_floored_for_thinking_locked_glm_5_3() {
        // Default 8192 would truncate GLM-5.3 mid-reasoning.
        assert_eq!(zhipuai_effective_max_tokens("glm-5.3", 8_192), 65_536);
        assert_eq!(zhipuai_effective_max_tokens("GLM-5.3", 4_096), 65_536);
    }

    #[test]
    fn zhipuai_max_tokens_respects_explicit_larger_values() {
        assert_eq!(zhipuai_effective_max_tokens("glm-5.3", 98_304), 98_304);
        assert_eq!(zhipuai_effective_max_tokens("glm-5.3", 131_072), 131_072);
    }

    #[test]
    fn zhipuai_max_tokens_untouched_for_thinking_optional_models() {
        // GLM-5.2 and earlier can disable thinking; the configured value stands.
        assert_eq!(zhipuai_effective_max_tokens("glm-5.2", 8_192), 8_192);
        assert_eq!(zhipuai_effective_max_tokens("glm-5-turbo", 8_192), 8_192);
        assert_eq!(zhipuai_effective_max_tokens("glm-4.7-flash", 8_192), 8_192);
    }

    #[test]
    fn zhipuai_floor_matches_glm_5_3_via_either_model_string() {
        // build_provider_for concatenates [provider.zhipuai].model and
        // [model].default_model before matching — a divergent manual TOML that
        // sends glm-5.3 must still get the floor.
        assert_eq!(
            zhipuai_effective_max_tokens("glm-5.2 glm-5.3", 8_192),
            65_536
        );
        assert_eq!(
            zhipuai_effective_max_tokens("glm-5.2 glm-4.7-flash", 8_192),
            8_192
        );
    }
}
