use std::sync::Arc;

use nca_common::config::{NcaConfig, ProviderKind};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::json;

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
            )
            .to_ascii_lowercase();
            let max_tokens = zhipuai_effective_max_tokens(
                &models,
                config.model.max_tokens,
                config.model.enable_thinking,
            );
            // GLM >= 5.3 rejects "disabled"; honor enable_thinking otherwise.
            let thinking_type = if glm_thinking_locked(&models) || config.model.enable_thinking {
                "enabled"
            } else {
                "disabled"
            };
            Ok(Arc::new(
                OpenAiCompatProvider::from_config(
                    &config.provider.zhipuai,
                    max_tokens,
                    ZHIPUAI_PROFILE,
                    extra,
                )?
                .with_thinking(json!({ "type": thinking_type })),
            ))
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
/// GLM-5.3 and newer cannot disable thinking (official docs: `thinking.type`
/// only supports `"enabled"`; requests with `"disabled"` are rejected), and
/// the reasoning budget shares the output cap — the model can exhaust the cap
/// mid-reasoning and return an empty `content` with
/// `finish_reason: "length"`, which surfaces as a hard "empty response"
/// failure.
///
/// The floor equals the documented GLM-5.x output window — 131072 tokens
/// (128K), consistent with `runtime/src/model_limits.rs`; the previous 65536
/// was merely the value used in ZhipuAI's own coding examples. The floor
/// applies when either:
///
/// - any GLM model in `models` is version >= 5.3 (thinking-locked), or
/// - thinking is enabled for any GLM-5.x model (reasoning shares the output
///   cap — the same mid-reasoning truncation trap even when the model could
///   technically disable thinking).
///
/// Explicitly larger values pass through untouched; GLM models that allow
/// disabling thinking (GLM-5.2 and earlier) with thinking off keep the
/// configured value as-is.
fn zhipuai_effective_max_tokens(models: &str, configured: u32, enable_thinking: bool) -> u32 {
    const THINKING_LOCKED_FLOOR: u32 = 131_072;
    // Key on every model string that can end up in the request body: the
    // provider-side model ([provider.zhipuai].model) and the session default
    // ([model].default_model) are normally kept in sync, but a manual TOML can
    // diverge them — flooring on either avoids skipping the floor for the
    // model that actually gets sent.
    let lowered = models.to_ascii_lowercase();
    let needs_floor =
        glm_thinking_locked(&lowered) || (enable_thinking && lowered.contains("glm-5"));
    if needs_floor && configured < THINKING_LOCKED_FLOOR {
        tracing::warn!(
            models = %lowered,
            configured,
            floor = THINKING_LOCKED_FLOOR,
            enable_thinking,
            "zhipuai model spends max_tokens on reasoning; raising max_tokens to avoid \
             mid-reasoning truncation (values >= the floor pass through; lower \
             values are always raised)"
        );
        return THINKING_LOCKED_FLOOR;
    }
    configured
}

/// Whether any GLM model mentioned in `models` (already lowercased) has
/// thinking locked on — i.e. rejects `thinking.type: "disabled"`.
///
/// GLM-5.3 introduced the lock. Scan every `glm-` occurrence for a
/// `glm-<major>[.<minor>]` version and report whether any occurrence parses
/// to at least (5, 3). A missing minor counts as `.0` (`glm-5-turbo` is
/// (5, 0), `glm-6` is (6, 0)); occurrences without a leading digit
/// (`glm-air`) are ignored.
fn glm_thinking_locked(models: &str) -> bool {
    let mut rest = models;
    while let Some(pos) = rest.find("glm-") {
        rest = &rest[pos + "glm-".len()..];
        let Some((major, after_major)) = parse_number_prefix(rest) else {
            // Not a versioned model; keep scanning after this occurrence.
            continue;
        };
        let (minor, after) = match after_major.strip_prefix('.') {
            Some(tail) => match parse_number_prefix(tail) {
                Some((minor, after)) => (minor, after),
                // "glm-5." with no digits after the dot: minor stays 0.
                None => (0, after_major),
            },
            None => (0, after_major),
        };
        if (major, minor) >= (5, 3) {
            return true;
        }
        rest = after;
    }
    false
}

/// Parse the leading ASCII-digit run of `s` as `u32`, returning the value and
/// the remainder of the string after the digits.
///
/// Returns `None` when `s` does not start with a digit or the digit run
/// overflows `u32`.
fn parse_number_prefix(s: &str) -> Option<(u32, &str)> {
    let end = s
        .char_indices()
        .find(|(_, c)| !c.is_ascii_digit())
        .map_or(s.len(), |(i, _)| i);
    if end == 0 {
        return None;
    }
    s[..end].parse::<u32>().ok().map(|value| (value, &s[end..]))
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
    fn glm_thinking_locked_detects_versions() {
        assert!(glm_thinking_locked("glm-5.3"));
        assert!(glm_thinking_locked("glm-5.3-flash"));
        assert!(glm_thinking_locked("glm-5.4"));
        assert!(glm_thinking_locked("glm-6"));
        assert!(glm_thinking_locked("glm-5.2 glm-5.3"));
        assert!(!glm_thinking_locked("glm-5.2"));
        assert!(!glm_thinking_locked("glm-4.7-flash"));
        // No minor → (5, 0), below the 5.3 lock.
        assert!(!glm_thinking_locked("glm-5-turbo"));
        assert!(!glm_thinking_locked("deepseek-v4"));
        assert!(!glm_thinking_locked(""));
    }

    #[test]
    fn zhipuai_max_tokens_floored_for_thinking_locked_glm_5_3() {
        // Default 8192 would truncate GLM-5.3 mid-reasoning.
        assert_eq!(
            zhipuai_effective_max_tokens("glm-5.3", 8_192, false),
            131_072
        );
        assert_eq!(
            zhipuai_effective_max_tokens("GLM-5.3", 4_096, false),
            131_072
        );
        // Values between the old 65_536 floor and the new one are also raised.
        assert_eq!(
            zhipuai_effective_max_tokens("glm-5.3", 98_304, false),
            131_072
        );
    }

    #[test]
    fn zhipuai_max_tokens_floored_for_future_locked_versions() {
        // Version-aware detection, not a hardcoded "glm-5.3" substring.
        assert_eq!(
            zhipuai_effective_max_tokens("glm-5.4", 8_192, false),
            131_072
        );
        assert_eq!(
            zhipuai_effective_max_tokens("glm-5.3-flash", 8_192, false),
            131_072
        );
    }

    #[test]
    fn zhipuai_max_tokens_respects_explicit_larger_values() {
        // 131_072 now equals the floor — values at or above it pass through
        // (only values strictly below the floor are raised).
        assert_eq!(
            zhipuai_effective_max_tokens("glm-5.3", 131_072, false),
            131_072
        );
        assert_eq!(
            zhipuai_effective_max_tokens("glm-5.3", 262_144, false),
            262_144
        );
    }

    #[test]
    fn zhipuai_max_tokens_untouched_for_thinking_optional_models() {
        // GLM-5.2 and earlier can disable thinking; with thinking off the
        // configured value stands.
        assert_eq!(zhipuai_effective_max_tokens("glm-5.2", 8_192, false), 8_192);
        assert_eq!(
            zhipuai_effective_max_tokens("glm-5-turbo", 8_192, false),
            8_192
        );
        assert_eq!(
            zhipuai_effective_max_tokens("glm-4.7-flash", 8_192, false),
            8_192
        );
    }

    #[test]
    fn zhipuai_max_tokens_floored_for_glm_5_with_thinking_enabled() {
        // Any GLM-5.x running with thinking on shares the reasoning budget
        // through the output cap — same mid-reasoning truncation trap.
        assert_eq!(
            zhipuai_effective_max_tokens("glm-5.2", 8_192, true),
            131_072
        );
        assert_eq!(
            zhipuai_effective_max_tokens("glm-5-turbo", 8_192, true),
            131_072
        );
        // The thinking-on floor is keyed to glm-5 only; older GLM keeps the
        // configured value.
        assert_eq!(
            zhipuai_effective_max_tokens("glm-4.7-flash", 8_192, true),
            8_192
        );
    }

    #[test]
    fn glm_thinking_locked_version_scan_edges() {
        // Adversarial edge table for the hand-rolled `glm-<major>[.<minor>]`
        // scan. Inputs are already lowercased per the function's contract.
        let cases: &[(&str, bool)] = &[
            // Locked: any occurrence >= (5, 3).
            ("glm-5.3", true),
            ("glm-5.3-flash", true),
            ("glm-5.4", true),
            // Two-digit minor parses as 10, not 1.
            ("glm-5.10", true),
            // Missing minor counts as .0 — (6, 0) is still >= (5, 3).
            ("glm-6", true),
            // Below the lock.
            ("glm-5", false),       // missing minor = .0 -> (5, 0)
            ("glm-5-turbo", false), // missing minor = .0 -> (5, 0)
            ("glm-5.", false),      // dot with no digits -> minor stays 0
            ("glm-5.2", false),
            ("glm-5.2-air", false),
            ("glm-4.7-flash", false),
            // No digits after `glm-` -> not a versioned model.
            ("glm-air", false),
            ("air", false),
            // Any occurrence wins, in either order.
            ("glm-5.2 glm-5.3", true),
            ("glm-5.3 glm-5.2", true),
            // Substring scan is intended — no word boundaries.
            ("xxglm-5.4xx", true),
            // Pinned actual behavior: leading zeros parse away ("05" -> 5 via
            // u32::from_str), so "glm-05.3" IS treated as locked.
            ("glm-05.3", true),
            // Pinned actual behavior: a digit run that overflows u32 makes
            // parse_number_prefix return None, so the occurrence is treated as
            // unversioned and skipped instead of panicking.
            ("glm-99999999999999999999", false),
        ];
        for (input, expected) in cases {
            assert_eq!(
                glm_thinking_locked(input),
                *expected,
                "glm_thinking_locked({input:?})"
            );
        }
    }

    #[test]
    fn zhipuai_floor_edges_for_thinking_toggle() {
        // (models, configured, enable_thinking) -> effective max_tokens.
        let cases: &[(&str, u32, bool, u32)] = &[
            // Thinking-locked by version alone (>= 5.3), regardless of toggle.
            ("glm-5.10", 8_192, false, 131_072),
            ("glm-6", 8_192, false, 131_072),
            // Not locked (missing minor = .0), but the enable_thinking flag
            // triggers the GLM-5 floor on its own.
            ("glm-5-turbo", 8_192, true, 131_072),
            ("glm-5-turbo", 8_192, false, 8_192),
            // Floor never applies below GLM-5, even with thinking on.
            ("glm-4.7-flash", 4_096, true, 4_096),
            // Boundary: configured == floor passes through untouched (the
            // condition is `configured < floor`, not `<=`).
            ("glm-5.2", 131_072, false, 131_072),
            // Above the floor: thinking on, but the configured value wins.
            ("glm-5.2", 262_144, true, 262_144),
        ];
        for (models, configured, enable_thinking, expected) in cases {
            assert_eq!(
                zhipuai_effective_max_tokens(models, *configured, *enable_thinking),
                *expected,
                "zhipuai_effective_max_tokens({models:?}, {configured}, {enable_thinking})"
            );
        }
    }

    #[test]
    fn zhipuai_floor_matches_glm_5_3_via_either_model_string() {
        // build_provider_for concatenates [provider.zhipuai].model and
        // [model].default_model before matching — a divergent manual TOML that
        // sends glm-5.3 must still get the floor.
        assert_eq!(
            zhipuai_effective_max_tokens("glm-5.2 glm-5.3", 8_192, false),
            131_072
        );
        assert_eq!(
            zhipuai_effective_max_tokens("glm-5.2 glm-4.7-flash", 8_192, false),
            8_192
        );
    }
}
