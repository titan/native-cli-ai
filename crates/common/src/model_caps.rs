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
        ProviderKind::DeepSeek => false, // DeepSeek does not support native image inputs
        ProviderKind::Kimi => false,     // Kimi for Coding: k3 specs don't list native image input
        // ponytail: custom endpoints vary; assume no native image input until configured otherwise
        ProviderKind::Custom => false,
    }
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
}
