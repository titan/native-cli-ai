//! Model-specific context window sizes and detection.
//!
//! Different LLM models have vastly different context limits.
//! This module provides detection and defaults for common models.
//!
//! For models routed via [OpenRouter](https://openrouter.ai/models), authoritative
//! per-model `context_length` values are published in the public API:
//! `GET https://openrouter.ai/api/v1/models` (JSON field `context_length` on each entry).
//!
//! Lives in `nca-common` (not `nca-runtime`) so the provider layer in `nca-core`
//! can apply the capability-aware [`clamp_max_tokens`] where the final model
//! string is known.

use serde::{Deserialize, Serialize};

/// Context window limits for various LLM models (in tokens).
/// These are approximate and may vary by API version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelContextLimits {
    /// Model identifier pattern (partial match)
    pub pattern: &'static str,
    /// Context window size in tokens
    pub context_window: usize,
    /// Recommended max tokens for the model
    pub max_output_tokens: usize,
}

/// Known model context windows.
/// Order matters - more specific patterns should come first.
pub const MODEL_CONTEXT_LIMITS: &[ModelContextLimits] = &[
    // Claude 3.7 family
    ModelContextLimits {
        pattern: "claude-3-7",
        context_window: 200_000,
        max_output_tokens: 8192,
    },
    // Claude 3.5 family
    ModelContextLimits {
        pattern: "claude-3-5",
        context_window: 200_000,
        max_output_tokens: 8192,
    },
    // Claude 3 Opus
    ModelContextLimits {
        pattern: "claude-3-opus",
        context_window: 200_000,
        max_output_tokens: 4096,
    },
    // Claude 3 Sonnet
    ModelContextLimits {
        pattern: "claude-3-sonnet",
        context_window: 200_000,
        max_output_tokens: 4096,
    },
    // Claude 3 Haiku
    ModelContextLimits {
        pattern: "claude-3-haiku",
        context_window: 200_000,
        max_output_tokens: 4096,
    },
    // GPT-4o family
    ModelContextLimits {
        pattern: "gpt-4o",
        context_window: 128_000,
        max_output_tokens: 16384,
    },
    // GPT-4.5 / GPT-4 Turbo
    ModelContextLimits {
        pattern: "gpt-4-turbo",
        context_window: 128_000,
        max_output_tokens: 4096,
    },
    // GPT-4
    ModelContextLimits {
        pattern: "gpt-4",
        context_window: 128_000,
        max_output_tokens: 4096,
    },
    // GPT-3.5 Turbo
    ModelContextLimits {
        pattern: "gpt-3.5-turbo",
        context_window: 16_385,
        max_output_tokens: 4096,
    },
    // MiniMax M3 (reasoning model)
    ModelContextLimits {
        pattern: "minimax-m3",
        context_window: 512_000,
        max_output_tokens: 128_000,
    },
    // MiniMax M2.7 — OpenRouter `minimax/minimax-m2.7`: context_length 204_800 (must be before `minimax-m2`)
    ModelContextLimits {
        pattern: "minimax-m2.7",
        context_window: 204_800,
        max_output_tokens: 131_072,
    },
    // OpenRouter slug (config may store full id)
    ModelContextLimits {
        pattern: "minimax/minimax-m2.7",
        context_window: 204_800,
        max_output_tokens: 131_072,
    },
    // MiniMax M2.5 (reasoning model)
    ModelContextLimits {
        pattern: "minimax-m2.5",
        context_window: 100_000,
        max_output_tokens: 8192,
    },
    // MiniMax M2 (not M2.5 / M2.7)
    ModelContextLimits {
        pattern: "minimax-m2",
        context_window: 32_000,
        max_output_tokens: 8192,
    },
    // MiniMax M1
    ModelContextLimits {
        pattern: "minimax-m1",
        context_window: 32_000,
        max_output_tokens: 8192,
    },
    // ZhipuAI GLM-5.3 (text-only, 1M context, 128K output; thinking always enabled)
    ModelContextLimits {
        pattern: "glm-5.3",
        context_window: 1_000_000,
        max_output_tokens: 131_072,
    },
    // ZhipuAI GLM-5.2 (1M context, 128K output)
    ModelContextLimits {
        pattern: "glm-5.2",
        context_window: 1_000_000,
        max_output_tokens: 131_072,
    },
    // ZhipuAI GLM-5 Turbo (coding plan)
    ModelContextLimits {
        pattern: "glm-5-turbo",
        context_window: 200_000,
        max_output_tokens: 128_000,
    },
    // Gemini 1.5 Pro
    ModelContextLimits {
        pattern: "gemini-1.5-pro",
        context_window: 2_000_000,
        max_output_tokens: 8192,
    },
    // Gemini 1.5 Flash
    ModelContextLimits {
        pattern: "gemini-1.5-flash",
        context_window: 1_000_000,
        max_output_tokens: 8192,
    },
    // Gemini 1.5
    ModelContextLimits {
        pattern: "gemini-1.5",
        context_window: 1_000_000,
        max_output_tokens: 8192,
    },
    // Gemini 2.0 Flash
    ModelContextLimits {
        pattern: "gemini-2.0-flash",
        context_window: 1_000_000,
        max_output_tokens: 8192,
    },
    // DeepSeek V4 Pro (1M context, 384K max output)
    ModelContextLimits {
        pattern: "deepseek-v4-pro",
        context_window: 1_000_000,
        max_output_tokens: 393_216,
    },
    // DeepSeek V4 Flash (1M context, 384K max output)
    ModelContextLimits {
        pattern: "deepseek-v4-flash",
        context_window: 1_000_000,
        max_output_tokens: 393_216,
    },
    // DeepSeek V4 catch-all
    ModelContextLimits {
        pattern: "deepseek-v4",
        context_window: 1_000_000,
        max_output_tokens: 393_216,
    },
    // DeepSeek V3
    ModelContextLimits {
        pattern: "deepseek-v3",
        context_window: 64_000,
        max_output_tokens: 8192,
    },
    // DeepSeek R1
    ModelContextLimits {
        pattern: "deepseek-r1",
        context_window: 64_000,
        max_output_tokens: 8192,
    },
    // Qwen 2.5
    ModelContextLimits {
        pattern: "qwen-2.5",
        context_window: 128_000,
        max_output_tokens: 8192,
    },
    // Llama 3.1 405B
    ModelContextLimits {
        pattern: "llama-3.1-405b",
        context_window: 128_000,
        max_output_tokens: 4096,
    },
    // Llama 3.1 70B
    ModelContextLimits {
        pattern: "llama-3.1-70b",
        context_window: 128_000,
        max_output_tokens: 4096,
    },
    // Llama 3.1 family
    ModelContextLimits {
        pattern: "llama-3.1",
        context_window: 128_000,
        max_output_tokens: 4096,
    },
    // Llama 3 family
    ModelContextLimits {
        pattern: "llama-3",
        context_window: 8_192,
        max_output_tokens: 2048,
    },
    // Kimi K3 (Moonshot AI) — 1M context, 131K output.
    // Reached via Kimi for Coding (model id "k3") or OpenRouter ("moonshotai/kimi-k3").
    ModelContextLimits {
        pattern: "kimi-k3",
        context_window: 1_048_576,
        max_output_tokens: 131_072,
    },
    // Alias for the Kimi-for-Coding endpoint's canonical model id "k3"
    // (the dedicated KimiProvider sends exactly this string).
    ModelContextLimits {
        pattern: "k3",
        context_window: 1_048_576,
        max_output_tokens: 131_072,
    },
    // Kimi K2.7 Code (256K context, 32K output) — OpenAI-compat model id,
    // same specs as kimi-for-coding per that entry.
    ModelContextLimits {
        pattern: "kimi-k2.7-code",
        context_window: 262_144,
        max_output_tokens: 32_768,
    },
    // Kimi K2.7 Code (256K context, 32K output) — Highspeed variant shares specs.
    ModelContextLimits {
        pattern: "kimi-for-coding",
        context_window: 262_144,
        max_output_tokens: 32_768,
    },
    // Default for unknown models
    ModelContextLimits {
        pattern: "*",
        context_window: 32_000,
        max_output_tokens: 4096,
    },
];

/// First [`MODEL_CONTEXT_LIMITS`] entry whose pattern occurs in `model`
/// (case-insensitive, first match wins), skipping the `"*"` fallback.
fn lookup(model: &str) -> Option<&'static ModelContextLimits> {
    let model_lower = model.to_lowercase();
    MODEL_CONTEXT_LIMITS
        .iter()
        .find(|limit| limit.pattern != "*" && model_lower.contains(&limit.pattern.to_lowercase()))
}

/// Detect the context window size for a given model name.
pub fn detect_context_window(model: &str) -> usize {
    lookup(model).map_or(32_000, |limit| limit.context_window)
}

/// Detect the max output tokens for a given model name.
pub fn detect_max_output_tokens(model: &str) -> usize {
    lookup(model).map_or(4096, |limit| limit.max_output_tokens)
}

/// Get both context window and max output tokens for a model.
#[derive(Debug, Clone)]
pub struct ModelLimits {
    pub context_window: usize,
    pub max_output_tokens: usize,
}

impl ModelLimits {
    pub fn for_model(model: &str) -> Self {
        Self {
            context_window: detect_context_window(model),
            max_output_tokens: detect_max_output_tokens(model),
        }
    }
}

/// Output-token floor for 128K-class models (DeepSeek V4, GLM-5.x, MiniMax
/// M2.7, Kimi K3). When a model's output window is at least this large, the
/// configured `max_tokens` is raised to at least this value.
pub const FLOOR_MAX_TOKENS: u32 = 131_072;

/// Capability-aware `max_tokens` clamp, applied where the final model string
/// is known (i.e. in `Provider::chat` request bodies).
///
/// Policy, keyed off the matched entry's `max_output_tokens` (cap):
///
/// - cap >= [`FLOOR_MAX_TOKENS`] → floor: `max(configured, FLOOR)` — 128K-class
///   models should not inherit the small global default;
/// - cap < [`FLOOR_MAX_TOKENS`] → protective cap: `min(configured, cap)` —
///   APIs like OpenAI/Anthropic hard-400 on oversize `max_tokens`;
/// - unknown model → `configured` unchanged (conservative passthrough).
///
/// Keepalive pings send `max_tokens = 1` and must never pass through here:
/// raising them to the floor would bill a 128K-token completion just to
/// refresh the input cache, and capping is moot at 1.
pub fn clamp_max_tokens(model: &str, configured: u32) -> u32 {
    let Some(limit) = lookup(model) else {
        return configured;
    };
    // Table values are all far below u32::MAX; the cast cannot truncate.
    let cap = limit.max_output_tokens as u32;
    let effective = if cap >= FLOOR_MAX_TOKENS {
        configured.max(FLOOR_MAX_TOKENS)
    } else {
        configured.min(cap)
    };
    if effective != configured {
        tracing::warn!(
            model = %model,
            configured,
            effective,
            "max_tokens adjusted to model output window"
        );
    }
    effective
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_claude() {
        assert_eq!(detect_context_window("claude-3-7-sonnet-latest"), 200_000);
        assert_eq!(detect_context_window("claude-3-5-sonnet-20241022"), 200_000);
        assert_eq!(detect_context_window("claude-3-opus-20240229"), 200_000);
    }

    #[test]
    fn test_detect_gpt() {
        assert_eq!(detect_context_window("gpt-4o-2024-08-06"), 128_000);
        assert_eq!(detect_context_window("gpt-4o-mini"), 128_000);
        assert_eq!(detect_context_window("gpt-4-turbo-2024-04-09"), 128_000);
    }

    #[test]
    fn test_detect_minimax() {
        assert_eq!(detect_context_window("MiniMax-M2.7"), 204_800);
        assert_eq!(detect_context_window("minimax/minimax-m2.7"), 204_800);
        assert_eq!(detect_context_window("MiniMax-M2.5"), 100_000);
        assert_eq!(detect_context_window("minimax-m2"), 32_000);
    }

    #[test]
    fn test_detect_glm() {
        assert_eq!(detect_context_window("glm-5.3"), 1_000_000);
        assert_eq!(detect_max_output_tokens("glm-5.3"), 131_072);
        assert_eq!(detect_context_window("glm-5.2"), 1_000_000);
        assert_eq!(detect_context_window("glm-5-turbo"), 200_000);
        assert_eq!(detect_max_output_tokens("glm-5-turbo"), 128_000);
    }

    #[test]
    fn test_detect_gemini() {
        assert_eq!(detect_context_window("gemini-1.5-pro-latest"), 2_000_000);
        assert_eq!(detect_context_window("gemini-1.5-flash"), 1_000_000);
    }

    #[test]
    fn test_detect_deepseek_v4() {
        assert_eq!(detect_context_window("deepseek-v4-flash"), 1_000_000);
        assert_eq!(detect_context_window("deepseek-v4-pro"), 1_000_000);
        assert_eq!(detect_max_output_tokens("deepseek-v4-flash"), 393_216);
    }

    #[test]
    fn test_detect_deepseek_legacy() {
        assert_eq!(detect_context_window("deepseek-v3"), 64_000);
        assert_eq!(detect_context_window("deepseek-r1"), 64_000);
    }

    #[test]
    fn test_detect_kimi() {
        // Canonical Kimi-for-Coding id "k3" (dedicated KimiProvider sends it).
        assert_eq!(detect_context_window("k3"), 1_048_576);
        assert_eq!(detect_max_output_tokens("k3"), 131_072);
        assert_eq!(detect_max_output_tokens("moonshotai/kimi-k3"), 131_072);
        // K2.7 Code via OpenAI-compat endpoints shares kimi-for-coding specs.
        assert_eq!(detect_context_window("kimi-k2.7-code"), 262_144);
        assert_eq!(detect_max_output_tokens("kimi-k2.7-code"), 32_768);
        assert_eq!(
            detect_max_output_tokens("kimi-for-coding-highspeed"),
            32_768
        );
    }

    #[test]
    fn test_fallback() {
        assert_eq!(detect_context_window("unknown-model-xyz"), 32_000);
    }

    #[test]
    fn test_model_limits_struct() {
        let limits = ModelLimits::for_model("claude-3-7-sonnet");
        assert_eq!(limits.context_window, 200_000);
        assert_eq!(limits.max_output_tokens, 8192);
    }

    #[test]
    fn test_clamp_max_tokens() {
        // (model, configured) -> expected effective value.
        let cases: &[(&str, u32, u32)] = &[
            // 128K-class floor: configured below the floor is raised to it.
            ("deepseek-v4-flash", 8_192, 131_072),
            ("MiniMax-M2.7", 8_192, 131_072),
            ("glm-5.2", 8_192, 131_072),
            ("k3", 8_192, 131_072),
            // Sub-floor output windows: oversize configured values are capped.
            ("glm-5-turbo", 200_000, 128_000),
            ("gpt-4o", 8_192, 8_192),
            ("gpt-4o", 131_072, 16_384),
            ("claude-3-7-sonnet", 999_999, 8_192),
            ("kimi-k2.7-code", 131_072, 32_768),
            // Unknown models pass through unchanged.
            ("totally-unknown-xyz", 5_000, 5_000),
        ];
        for (model, configured, expected) in cases {
            assert_eq!(
                clamp_max_tokens(model, *configured),
                *expected,
                "clamp_max_tokens({model:?}, {configured})"
            );
        }
    }
}
