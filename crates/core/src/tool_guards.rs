//! Session-scoped tool-call guards: repeated-call detection.
//!
//! Mirrors dsh's cheap but effective behavioral guard: hashing
//! `tool_name + canonical input` and escalating through Hint → StrongHint →
//! Stop when the agent keeps issuing the *identical* call. Counts live for
//! the lifetime of the guard (one per session), so detection works across
//! tool batches, steps, and turns.

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

/// Maximum number of distinct (tool, input) pairs tracked. Oldest entries are
/// evicted FIFO; survivors keep their running counts across evictions.
const RECENT_CALLS_CAP: usize = 32;

/// Escalation threshold for the first hint.
const HINT_THRESHOLD: u32 = 3;
/// Escalation threshold for the strong hint.
const STRONG_HINT_THRESHOLD: u32 = 5;
/// Escalation threshold for the hard stop.
const STOP_THRESHOLD: u32 = 8;

/// Action the pipeline should take for a given tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepeatAction {
    /// Execute normally.
    Proceed,
    /// Execute, then append this hint to the tool result output.
    Hint(String),
    /// Execute, then append this (stronger) hint to the tool result output.
    StrongHint(String),
    /// Do not execute; return a failed result carrying this message.
    Stop(String),
}

/// Tracks (tool, input) pairs for repeat detection.
///
/// Keys are `hash(tool_name + canonical serde_json serialization of input)`.
/// The recent-calls window is capped at [`RECENT_CALLS_CAP`] entries with FIFO
/// eviction; eviction is **lazy**: dropping a key from the window never resets
/// a survivor's running count, and a key re-recorded after eviction resumes
/// its historical count. Counts live for the lifetime of the guard.
pub struct RepeatCallGuard {
    /// Running call count per canonical key (persists across evictions).
    counts: HashMap<u64, u32>,
    /// Recency window of live keys, capped at [`RECENT_CALLS_CAP`] (FIFO).
    order: VecDeque<u64>,
}

impl RepeatCallGuard {
    /// Create an empty guard.
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    /// Record one call and return the escalation action for it.
    pub fn record(&mut self, tool_name: &str, input: &serde_json::Value) -> RepeatAction {
        let key = canonical_key(tool_name, input);
        let count = {
            let entry = self.counts.entry(key).or_insert(0);
            *entry += 1;
            if *entry == 1 {
                self.order.push_back(key);
            } else if let Some(pos) = self.order.iter().position(|&k| k == key) {
                // Refresh recency so a still-active key is evicted last.
                self.order.remove(pos);
                self.order.push_back(key);
            }
            *entry
        };

        // FIFO eviction bounds the recency window. Lazy: the evicted key's
        // count is intentionally kept so survivors never lose escalation
        // state (contract: eviction must not reset live counts).
        while self.order.len() > RECENT_CALLS_CAP {
            let _ = self.order.pop_front();
        }

        if count >= STOP_THRESHOLD {
            RepeatAction::Stop(format!(
                "[guard] 你已连续第 {count} 次用完全相同的参数调用 `{tool_name}`，该调用已被硬停。\
                 继续重复不会产生新结果：请改变策略——换用其它工具、修改参数、或先阅读之前的工具输出再决定下一步。"
            ))
        } else if count >= STRONG_HINT_THRESHOLD {
            RepeatAction::StrongHint(format!(
                "[guard] ⚠ 你已第 {count} 次用完全相同的参数调用 `{tool_name}`——这几乎肯定是在原地打转。\
                 请立刻停下来分析上一次的输出，或改用不同的方法。"
            ))
        } else if count >= HINT_THRESHOLD {
            RepeatAction::Hint(format!(
                "[guard] 你已第 {count} 次用相同参数调用 `{tool_name}`，结果大概率与之前相同。\
                 建议先检查之前的输出，或尝试不同的方法。"
            ))
        } else {
            RepeatAction::Proceed
        }
    }
}

impl Default for RepeatCallGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// Hash the tool name together with the canonical JSON serialization of the
/// input. `serde_json` Maps are sorted by key, so key order in the original
/// value does not affect the hash.
fn canonical_key(tool_name: &str, input: &serde_json::Value) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    tool_name.hash(&mut hasher);
    hasher.write_u8(0xFF); // delimiter: name cannot collide with serialized JSON
    if let Ok(canonical) = serde_json::to_string(input) {
        canonical.hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn thresholds_escalate_and_stop_persists() {
        let mut guard = RepeatCallGuard::new();
        let input = json!({"path": "x"});
        for _ in 0..2 {
            assert_eq!(guard.record("t", &input), RepeatAction::Proceed);
        }
        assert!(matches!(guard.record("t", &input), RepeatAction::Hint(_)));
        assert!(matches!(guard.record("t", &input), RepeatAction::Hint(_)));
        assert!(matches!(
            guard.record("t", &input),
            RepeatAction::StrongHint(_)
        ));
        assert!(matches!(
            guard.record("t", &input),
            RepeatAction::StrongHint(_)
        ));
        assert!(matches!(
            guard.record("t", &input),
            RepeatAction::StrongHint(_)
        ));
        assert!(matches!(guard.record("t", &input), RepeatAction::Stop(_)));
        assert!(matches!(guard.record("t", &input), RepeatAction::Stop(_)));
    }

    #[test]
    fn canonical_key_ignores_key_order() {
        let a = json!({"a": 1, "b": 2});
        let b = json!({"b": 2, "a": 1});
        assert_eq!(canonical_key("t", &a), canonical_key("t", &b));
        assert_ne!(canonical_key("t", &a), canonical_key("u", &a));
    }
}
