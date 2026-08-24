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

    /// Rough cost estimate in USD based on hard-coded Claude-Sonnet-class
    /// rates (3.0/1M input, 15.0/1M output, 3.0/1M cache creation,
    /// 3.0/50M cache read) — the single rate table in the codebase.
    ///
    /// Estimate-grade by construction: OpenAI-compatible `input_tokens`
    /// (DeepSeek, OpenAI) includes cached tokens, so cache reads are
    /// double-counted (full input rate + cache_read rate), and Sonnet rates
    /// overstate actual spend roughly 10× for DeepSeek — the primary
    /// provider. Adequate as a threshold, not an invoice.
    pub fn estimated_cost_for(
        input_tokens: u64,
        output_tokens: u64,
        cache_creation_tokens: u64,
        cache_read_tokens: u64,
    ) -> f64 {
        let input_cost = input_tokens as f64 * 3.0 / 1_000_000.0;
        let output_cost = output_tokens as f64 * 15.0 / 1_000_000.0;
        let cache_creation_cost = cache_creation_tokens as f64 * 3.0 / 1_000_000.0;
        let cache_read_cost = cache_read_tokens as f64 * 3.0 / (1_000_000.0 * 50.0);
        input_cost + output_cost + cache_creation_cost + cache_read_cost
    }

    /// Session cost estimate in USD (delegates to the shared rate table,
    /// [`CostTracker::estimated_cost_for`]). See that fn's caveats.
    pub fn estimated_cost_usd(&self) -> f64 {
        Self::estimated_cost_for(
            self.input_tokens,
            self.output_tokens,
            self.cache_creation_tokens,
            self.cache_read_tokens,
        )
    }
}
