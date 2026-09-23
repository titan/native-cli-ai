//! Heuristic vision / multimodal support by provider and model id.

use crate::config::ProviderKind;

/// Whether the active provider+model is treated as supporting **native** image inputs
/// in chat (not MCP OCR fallback).
pub fn model_accepts_native_images(kind: ProviderKind, model: &str) -> bool {
    let m = model.trim().to_ascii_lowercase();
    match kind {
        ProviderKind::MiniMax => {
            // MiniMax M-series on the Anthropic-compatible endpoint supports image blocks.
            !m.is_empty()
        }
        ProviderKind::Anthropic => {
            m.contains("claude-3")
                || m.contains("claude-4")
                || m.contains("claude-opus-4")
                || m.contains("claude-sonnet-4")
        }
        ProviderKind::OpenAi => {
            m.contains("gpt-4o")
                || m.contains("gpt-4-turbo")
                || m.contains("gpt-5")
                || m.contains("o1")
                || m.contains("o3")
                || m.contains("vision")
        }
        ProviderKind::OpenRouter => {
            m.contains("gpt-4o")
                || m.contains("gpt-4-turbo")
                || m.contains("gpt-5")
                || m.contains("claude-3")
                || m.contains("claude-4")
                || m.contains("gemini")
                || m.contains("vision")
                || m.contains("qwen-vl")
        }
        // glm-5.3 is text-only; glm-5.3-flash is the GLM-5 series' first
        // native multimodal model; glm-5.2 and glm-4v/glm-4 accept native images.
        ProviderKind::ZhipuAI => {
            let text_only = m.contains("glm-5.3") && !m.contains("glm-5.3-flash");
            (m.contains("glm-5") && !text_only) || m.contains("glm-4v") || m.contains("glm-4")
        }
        // Official API (2026-09): exactly two models — `deepseek-flash`
        // (DeepSeek-V4.1-Flash, vision ✓ per the Vision guide) and
        // `deepseek-v4-pro` (vision explicitly "Not supported" in the
        // pricing table). Legacy `deepseek-v4-flash` and
        // `deepseek-v4-flash-vision-exp` are retired but still accepted,
        // both served by (multimodal) V4.1-Flash. The old V3-era aliases
        // `deepseek-chat`/`deepseek-reasoner` are no longer documented;
        // if they still resolve server-side they land on the current
        // generation, so they count as multimodal (unknown explicit names
        // stay conservative — the run_turn_with_images gate names the
        // model when it rejects).
        ProviderKind::DeepSeek => {
            let v4_pro_text_only = m.contains("v4-pro") || m.contains("v4.1-pro");
            !v4_pro_text_only
                && (m.contains("chat")
                    || m.contains("reasoner")
                    || m.contains("vl")
                    || m.contains("flash")
                    || deepseek_generation(&m).is_some_and(|v| v >= 4))
        }
        // Kimi for Coding serves k3 on the Anthropic-compatible endpoint, which
        // accepts native image blocks (anthropic_compat serializes them directly;
        // no coding_plan/vlm sidecar needed).
        ProviderKind::Kimi => !m.is_empty(),
        // Xiaomi MiMo v2.6 models are omnimodal (text/image/video/audio input)
        // on the OpenAI-compatible chat completions endpoint.
        ProviderKind::Mimo => m.contains("mimo"),
        // ponytail: custom endpoints vary; assume no native image input until configured otherwise
        ProviderKind::Custom => false,
    }
}

/// Major version number of a `deepseek-v<N>...` style model name, if present.
fn deepseek_generation(model: &str) -> Option<u32> {
    let m = model.as_bytes();
    let mut i = 0;
    while i + 1 < m.len() {
        if m[i] == b'v' && m[i + 1].is_ascii_digit() {
            let start = i + 1;
            let end = m[start..]
                .iter()
                .position(|b| !b.is_ascii_digit())
                .map_or(m.len(), |p| start + p);
            return std::str::from_utf8(&m[start..end]).ok()?.parse().ok();
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimax_default_is_vision() {
        assert!(model_accepts_native_images(
            ProviderKind::MiniMax,
            "MiniMax-M2.5"
        ));
    }

    #[test]
    fn gpt35_is_not_vision_openai() {
        assert!(!model_accepts_native_images(
            ProviderKind::OpenAi,
            "gpt-3.5-turbo"
        ));
    }

    #[test]
    fn kimi_k3_accepts_native_images() {
        assert!(model_accepts_native_images(ProviderKind::Kimi, "k3"));
        assert!(model_accepts_native_images(ProviderKind::Kimi, "kimi-k3"));
    }

    #[test]
    fn glm_5_3_is_text_only_but_flash_is_multimodal() {
        assert!(!model_accepts_native_images(
            ProviderKind::ZhipuAI,
            "glm-5.3"
        ));
        // glm-5.3-flash: GLM-5 series' first native multimodal model
        // (image / video / file input).
        assert!(model_accepts_native_images(
            ProviderKind::ZhipuAI,
            "glm-5.3-flash"
        ));
        assert!(model_accepts_native_images(
            ProviderKind::ZhipuAI,
            "glm-5.2"
        ));
        assert!(model_accepts_native_images(
            ProviderKind::ZhipuAI,
            "glm-4v-flash"
        ));
    }

    #[test]
    fn mimo_v2_6_family_accepts_native_images() {
        // All MiMo v2.6 models are omnimodal.
        assert!(model_accepts_native_images(
            ProviderKind::Mimo,
            "mimo-v2.6-pro"
        ));
        assert!(model_accepts_native_images(
            ProviderKind::Mimo,
            "mimo-v2.6-flash"
        ));
        assert!(model_accepts_native_images(
            ProviderKind::Mimo,
            "mimo-v2.6-pro-ultraspeed"
        ));
        // Empty / non-mimo strings stay conservative.
        assert!(!model_accepts_native_images(ProviderKind::Mimo, ""));
        assert!(!model_accepts_native_images(ProviderKind::Mimo, "other"));
    }

    #[test]
    fn deepseek_v4_onwards_is_multimodal() {
        assert!(model_accepts_native_images(
            ProviderKind::DeepSeek,
            "deepseek-v4"
        ));
        assert!(model_accepts_native_images(
            ProviderKind::DeepSeek,
            "deepseek-v4-flash"
        ));
        // API id for V4.1 Flash — the multimodal successor to V4 Flash.
        assert!(model_accepts_native_images(
            ProviderKind::DeepSeek,
            "deepseek-flash"
        ));
        // Rolling aliases track the current (multimodal) generation.
        assert!(model_accepts_native_images(
            ProviderKind::DeepSeek,
            "deepseek-chat"
        ));
        assert!(model_accepts_native_images(
            ProviderKind::DeepSeek,
            "deepseek-reasoner"
        ));
    }

    #[test]
    fn deepseek_v4_pro_is_text_only() {
        // Official pricing table: vision "Not supported" for deepseek-v4-pro.
        assert!(!model_accepts_native_images(
            ProviderKind::DeepSeek,
            "deepseek-v4-pro"
        ));
        assert!(!model_accepts_native_images(
            ProviderKind::DeepSeek,
            "deepseek-v4.1-pro"
        ));
    }

    #[test]
    fn deepseek_retired_legacy_ids_still_multimodal() {
        // Retired but accepted; served by multimodal V4.1-Flash.
        assert!(model_accepts_native_images(
            ProviderKind::DeepSeek,
            "deepseek-v4-flash"
        ));
        assert!(model_accepts_native_images(
            ProviderKind::DeepSeek,
            "deepseek-v4-flash-vision-exp"
        ));
    }

    #[test]
    fn deepseek_pre_v4_is_text_only() {
        assert!(!model_accepts_native_images(
            ProviderKind::DeepSeek,
            "deepseek-v3"
        ));
        assert!(!model_accepts_native_images(
            ProviderKind::DeepSeek,
            "deepseek-v3.2"
        ));
        assert!(!model_accepts_native_images(
            ProviderKind::DeepSeek,
            "deepseek-r1"
        ));
        // Unknown explicit names stay conservative.
        assert!(!model_accepts_native_images(
            ProviderKind::DeepSeek,
            "deepseek"
        ));
        assert!(!model_accepts_native_images(ProviderKind::DeepSeek, ""));
    }
}
