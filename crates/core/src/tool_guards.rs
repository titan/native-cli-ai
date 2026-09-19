//! Session-scoped tool-call guards: repeated-call detection.
//!
//! Mirrors dsh's cheap but effective behavioral guard: hashing
//! `tool_name + canonical input` and escalating through Hint → StrongHint →
//! Stop when the agent keeps issuing the *identical* call. Counts live for
//! the lifetime of the guard (one per session), so detection works across
//! tool batches, steps, and turns.
//!
//! Wait tools (`wait_for_user`) are a separate regime: their contract is
//! "end this turn now", so a repeat within one turn is always degenerate
//! regardless of arguments or output. They are counted by tool name within
//! the current turn (see [`RepeatCallGuard::reset_turn`]) and escalate much
//! sooner — warn on the 2nd call, refuse from the 3rd on.

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

/// Wait tools whose contract is "end this turn now" (P4 wait-guard, after
/// oh-my-opencode-slim's tool-loop-guard #1139). A repeat within one turn is
/// always degenerate: the first result already told the model to stop calling
/// tools, and because the tool returns instantly, every repeat re-triggers a
/// full-context inference pass. Counted by tool name only — arguments and
/// output are irrelevant to the degeneracy.
const WAIT_TOOLS: &[&str] = &["wait_for_user"];
/// Warn (execute, then append the marker warning to the output) on the 2nd
/// call of a wait tool within a turn.
const WAIT_WARN_AT: u32 = 2;
/// Refuse execution from the 3rd call of a wait tool within a turn on.
const WAIT_BLOCK_AT: u32 = 3;

/// Marker leading the warning appended to a repeated wait tool's output.
/// Kept grep-identical to oh-my-opencode-slim's `WAIT_GUARD_MARKER`.
pub const WAIT_GUARD_MARKER: &str = "[REPEATED WAIT TOOL - END TURN]";

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
    /// Wait-tool calls in the CURRENT turn, keyed by tool name. Reset by
    /// [`RepeatCallGuard::reset_turn`] at every turn start; never enters the
    /// generic `counts` map above.
    wait_turn_runs: HashMap<String, u32>,
}

impl RepeatCallGuard {
    /// Create an empty guard.
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
            order: VecDeque::new(),
            wait_turn_runs: HashMap::new(),
        }
    }

    /// Record one call and return the escalation action for it.
    ///
    /// Wait tools ([`WAIT_TOOLS`]) take the dedicated per-turn path
    /// ([`Self::record_wait`]) and never touch the generic identical-call
    /// state.
    pub fn record(&mut self, tool_name: &str, input: &serde_json::Value) -> RepeatAction {
        if is_wait_tool(tool_name) {
            return self.record_wait(tool_name);
        }
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

    /// Record one wait-tool call in the current turn and return its action:
    /// 1st call proceeds, the 2nd executes with the marker warning appended,
    /// and the 3rd on is refused without execution. The refusal composes with
    /// the turn driver's consecutive-failure stop, so a non-compliant model
    /// that keeps re-calling wait is terminated after a bounded number of
    /// refusals instead of looping on full-context inference forever.
    fn record_wait(&mut self, tool_name: &str) -> RepeatAction {
        let runs = self
            .wait_turn_runs
            .entry(tool_name.to_string())
            .or_insert(0);
        *runs += 1;
        let n = *runs;

        if n >= WAIT_BLOCK_AT {
            RepeatAction::Stop(format!(
                "[guard] 拒绝执行 `{tool_name}`：这已是本轮第 {n} 次调用，而它的契约是「结束当前回合」。\
                 结束回合不应再调用 wait，而应直接产出面向用户的最终回复。\
                 请立即停止调用任何工具，用纯文本给出最终答复来结束本轮。"
            ))
        } else if n >= WAIT_WARN_AT {
            RepeatAction::Hint(format!(
                "{WAIT_GUARD_MARKER}\n\
                 你已在本轮第 {n} 次调用 `{tool_name}`，它的结果已经指示你结束回合——重复调用不会推进任何事。\
                 请停止调用工具，直接以纯文本回应用户并结束本轮；用户的下一条新消息会恢复正常续行。"
            ))
        } else {
            RepeatAction::Proceed
        }
    }

    /// Reset per-turn guard state (the wait-tool repeat counts). Called by
    /// the agent loop at every turn start (`TurnStarted` / new user message):
    /// a wait tool is legitimate once per turn, so each new turn restores
    /// first-call behavior. The generic identical-call counts are
    /// intentionally NOT reset — they are session-scoped by contract.
    pub fn reset_turn(&mut self) {
        self.wait_turn_runs.clear();
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

/// Whether `tool_name` is a wait tool on the dedicated per-turn path.
fn is_wait_tool(tool_name: &str) -> bool {
    WAIT_TOOLS.contains(&tool_name)
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

    // ── Wait tools: per-turn counter (P4 wait-guard) ──────────────────────

    fn record_wait(guard: &mut RepeatCallGuard) -> RepeatAction {
        guard.record("wait_for_user", &json!({}))
    }

    #[test]
    fn wait_tool_first_proceeds_second_warns_third_refuses() {
        let mut guard = RepeatCallGuard::new();

        // 1st call in the turn: normal execution.
        assert_eq!(record_wait(&mut guard), RepeatAction::Proceed);

        // 2nd call: still executes, but the hint carries the exact marker.
        let hint = match record_wait(&mut guard) {
            RepeatAction::Hint(msg) => msg,
            other => panic!("2nd wait call must be a Hint, got {other:?}"),
        };
        assert!(
            hint.contains(WAIT_GUARD_MARKER),
            "2nd-call warning must carry the marker: {hint}"
        );

        // 3rd and later calls: refused without execution, and the refusal
        // persists (the count keeps climbing on stopped calls too).
        for n in 3..=4 {
            let stop = match record_wait(&mut guard) {
                RepeatAction::Stop(msg) => msg,
                other => panic!("call #{n} must be refused, got {other:?}"),
            };
            assert!(
                stop.contains("最终回复"),
                "refusal must tell the model to produce a final reply: {stop}"
            );
        }
    }

    #[test]
    fn wait_tool_counter_resets_on_new_turn() {
        let mut guard = RepeatCallGuard::new();
        for _ in 0..3 {
            let _ = record_wait(&mut guard); // burn through warn into refused
        }
        assert!(matches!(record_wait(&mut guard), RepeatAction::Stop(_)));

        // Turn boundary: first call of the new turn proceeds again.
        guard.reset_turn();
        assert_eq!(record_wait(&mut guard), RepeatAction::Proceed);
        assert!(matches!(record_wait(&mut guard), RepeatAction::Hint(_)));
    }

    #[test]
    fn wait_tool_counts_by_name_regardless_of_arguments() {
        // Arguments are irrelevant to the degeneracy: distinct inputs still
        // count together on the wait path.
        let mut guard = RepeatCallGuard::new();
        assert_eq!(
            guard.record("wait_for_user", &json!({"note": "a"})),
            RepeatAction::Proceed
        );
        assert!(matches!(
            guard.record("wait_for_user", &json!({"note": "b"})),
            RepeatAction::Hint(_)
        ));
        assert!(matches!(
            guard.record("wait_for_user", &json!({"note": "b"})),
            RepeatAction::Stop(_)
        ));
    }

    #[test]
    fn generic_path_schedule_is_unchanged_alongside_wait_counting() {
        // Sanity: the generic identical-call schedule (3/5/8) is untouched by
        // the wait regime — a burn-in of wait calls plus a reset must leave
        // `read_file` escalating exactly at its own 3rd identical call.
        let mut guard = RepeatCallGuard::new();
        for _ in 0..3 {
            let _ = guard.record("wait_for_user", &json!({}));
        }
        guard.reset_turn();

        let input = json!({"path": "same"});
        assert_eq!(guard.record("read_file", &input), RepeatAction::Proceed);
        assert_eq!(guard.record("read_file", &input), RepeatAction::Proceed);
        assert!(matches!(
            guard.record("read_file", &input),
            RepeatAction::Hint(_)
        ));
    }

    #[test]
    fn ask_question_keeps_the_generic_schedule() {
        // The wait special case must not leak into ask_question (also
        // interactive, also zero-arg): identical repeats follow the generic
        // 3/5/8 thresholds exactly as before.
        let mut guard = RepeatCallGuard::new();
        let input = json!({});
        assert_eq!(guard.record("ask_question", &input), RepeatAction::Proceed);
        assert_eq!(guard.record("ask_question", &input), RepeatAction::Proceed);
        assert!(matches!(
            guard.record("ask_question", &input),
            RepeatAction::Hint(_)
        ));
        assert!(matches!(
            guard.record("ask_question", &input),
            RepeatAction::Hint(_)
        ));
    }
}
