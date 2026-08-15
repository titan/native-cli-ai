/// Tracks token usage and estimates cost for a session.
#[derive(Debug, Clone, Default)]
pub struct CostTracker {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cache_read_tokens: u64,
}

impl CostTracker {
    pub fn add(&mut self, input: u64, output: u64, cache_creation: u64, cache_read: u64) {
        self.input_tokens += input;
        self.output_tokens += output;
        self.cache_creation_tokens += cache_creation;
        self.cache_read_tokens += cache_read;
    }

    /// Fraction of cached input tokens served from cache (cache reads vs all
    /// cached tokens). For DeepSeek this is `hit / (hit + miss)`. For Anthropic
    /// this is `read / (read + creation)`.
    ///
    /// A ratio trending toward 1.0 means the prompt prefix is stable and the
    /// provider is successfully reusing cached content. A drop signals prefix
    /// instability or cache eviction — investigate prefix changes.
    pub fn cache_hit_ratio(&self) -> f64 {
        let cached = self.cache_read_tokens + self.cache_creation_tokens;
        if cached == 0 {
            return 0.0;
        }
        self.cache_read_tokens as f64 / cached as f64
    }

    /// Rough cost estimate in USD based on Claude Sonnet pricing.
    ///
    /// NOTE: `input_tokens` from OpenAI-compatible providers (DeepSeek,
    /// OpenAI) includes cached tokens, so they are double-counted here (full
    /// input rate + cache_read rate). A per-model pricing lookup would fix
    /// this, but for now this is a rough estimate only.
    pub fn estimated_cost_usd(&self) -> f64 {
        let input_cost = self.input_tokens as f64 * 3.0 / 1_000_000.0;
        let output_cost = self.output_tokens as f64 * 15.0 / 1_000_000.0;
        let cache_creation_cost = self.cache_creation_tokens as f64 * 3.0 / 1_000_000.0;
        let cache_read_cost = self.cache_read_tokens as f64 * 3.0 / (1_000_000.0 * 50.0);
        input_cost + output_cost + cache_creation_cost + cache_read_cost
    }
}
