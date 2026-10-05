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

/// Progress-gated count decay: when a key repeats only after at least this
/// many NOVEL (first-time) calls were recorded since its last occurrence,
/// the repeat is treated as fresh (count resets to 1) instead of climbing.
/// This distinguishes "looping without progress" (alternating A/B/A/B never
/// accrues enough novel calls to decay) from "the same call became useful
/// again after real progress" — e.g. a research session re-running a search
/// after several OTHER queries landed new context. Without decay, the
/// session-lifetime count hard-stops legitimate late repeats at 8.
const PROGRESS_DECAY_NOVEL: u32 = 4;

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
/// a survivor's running count. Counts live for the lifetime of the guard,
/// EXCEPT they are reset by progress-gated decay: a key repeated only after
/// at least [`PROGRESS_DECAY_NOVEL`] novel (first-time) calls since its last
/// occurrence restarts fresh — repeats without intervening progress keep
/// climbing, which is the actual loop signal.
pub struct RepeatCallGuard {
    /// Running call count per canonical key (persists across evictions).
    counts: HashMap<u64, u32>,
    /// Recency window of live keys, capped at [`RECENT_CALLS_CAP`] (FIFO).
    order: VecDeque<u64>,
    /// Monotonic counter of novel (first-time) calls recorded so far — one
    /// tick per new key. Repeats do not tick it. Used for progress-gated
    /// decay: a repeat is "fresh" again only after enough novel work
    /// happened in between (see [`PROGRESS_DECAY_NOVEL`]).
    progress_seq: u32,
    /// Per-key snapshot of `progress_seq` at the key's last record, so the
    /// next repeat can measure how much novel work happened in between.
    key_progress: HashMap<u64, u32>,
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
            progress_seq: 0,
            key_progress: HashMap::new(),
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
            // Progress-gated decay: measure how much NOVEL work happened
            // since this key's last record. Enough intervening novel calls
            // mean the repeat is contextually fresh, not a loop — reset the
            // running count to 1. Too little (alternating retries, immediate
            // re-issues) keeps the count climbing.
            let last_progress = self.key_progress.get(&key).copied().unwrap_or(0);
            if self.progress_seq.saturating_sub(last_progress) >= PROGRESS_DECAY_NOVEL {
                self.counts.insert(key, 0);
            }
            let entry = self.counts.entry(key).or_insert(0);
            let was_novel = *entry == 0;
            *entry += 1;
            if was_novel {
                self.progress_seq = self.progress_seq.saturating_add(1);
            }
            // Snapshot AFTER this record's own novel tick, so the next
            // repeat's delta counts only the novel calls recorded strictly
            // after this one (an off-by-one here double-credits the decay).
            self.key_progress.insert(key, self.progress_seq);
            if let Some(pos) = self.order.iter().position(|&k| k == key) {
                // Refresh recency so a still-active key is evicted last
                // (also dedupes after a decay reset re-bumps a live key).
                self.order.remove(pos);
            }
            self.order.push_back(key);
            *entry
        };

        // FIFO eviction bounds the recency window. Lazy: the evicted key's
        // count is intentionally kept so survivors never lose escalation
        // state to eviction alone (progress-gated decay above is the only
        // count reset short of `reset_session`).
        while self.order.len() > RECENT_CALLS_CAP {
            let _ = self.order.pop_front();
        }

        if count >= STOP_THRESHOLD {
            RepeatAction::Stop(format!(
                "[guard] 你已连续第 {count} 次用完全相同的参数调用 `{tool_name}`，该调用已被硬停。\
                 继续重复不会产生新结果。{}\
                 如果确实没有替代方案，请停止调用任何工具，直接给出说明缺口与已尝试内容的部分结果来结束回合——干净的失败报告远优于原地打转。",
                variation_advice(tool_name)
            ))
        } else if count >= STRONG_HINT_THRESHOLD {
            RepeatAction::StrongHint(format!(
                "[guard] ⚠ 你已第 {count} 次用完全相同的参数调用 `{tool_name}`——这几乎肯定是在原地打转。{}
                 请立刻停下来分析上一次的输出，或改用不同的方法。",
                variation_advice(tool_name)
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

    /// Reset all guard state (generic identical-call counts included).
    ///
    /// Called by the runtime supervisor when a NEW session starts in-process
    /// (`/new`): the same `AgentLoop` is reused, but a fresh session must not
    /// inherit the previous session's escalation state — a call the old
    /// session burned to the Stop threshold would otherwise be hard-stopped
    /// on its first legitimate use in the new session.
    pub fn reset_session(&mut self) {
        *self = Self::new();
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

/// Concrete, tool-specific variation strategies appended to StrongHint and
/// Stop messages. Research-type agents stall on generic "change strategy"
/// advice because they cannot think of an alternative — naming the exact
/// moves for the tools they are looping on converts a dead-end refusal into
/// an actionable redirect (observed: three subagent sessions killed by the
/// guard while re-issuing identical `web_search`/`search_code` calls).
fn variation_advice(tool_name: &str) -> &'static str {
    match tool_name {
        "web_search" => {
            "可尝试的具体替代：换同义关键词、限定 site: 域名、拆分成更窄的子查询、或对已有结果中的 URL 改用 fetch_url 直取原文。"
        }
        "fetch_url" => {
            "可尝试的具体替代：检查 URL 是否拼写正确/协议是否为 https、改用 web_search 先找到正确链接、或抓取该站的其它页面。"
        }
        "search_code" | "ast_grep_search" => {
            "可尝试的具体替代：放宽或收紧正则、换大小写敏感开关、改搜标识符的其它片段、或用 query_symbols/read_file 直接定位。"
        }
        "query_symbols" => {
            "可尝试的具体替代：换更短或更完整的符号名、用 search_code 全文搜索定义处、或 read_file 直接读目标模块。"
        }
        "read_file" | "list_directory" => {
            "可尝试的具体替代：先 list_directory 确认路径存在、检查大小写与相对路径基准、或用 search_code 定位内容后再读。"
        }
        _ => "请改变策略——换用其它工具、修改参数、或先阅读之前的工具输出再决定下一步。",
    }
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

    #[test]
    fn progress_decay_treats_post_progress_repeats_as_fresh() {
        let mut guard = RepeatCallGuard::new();
        let q = json!({"query": "glm reasoning_content"});
        assert_eq!(guard.record("web_search", &q), RepeatAction::Proceed);

        // Burn the key toward escalation WITHOUT intervening novel work.
        for _ in 0..6 {
            let _ = guard.record("web_search", &q);
        }
        // 7 identical calls so far; the 8th would Stop. Instead, do 4 NOVEL
        // calls (distinct queries) — enough progress to decay the count.
        for i in 0..4 {
            let _ = guard.record("web_search", &json!({"query": format!("other {i}")}));
        }
        // The same query again is contextually fresh: count resets to 1.
        assert_eq!(
            guard.record("web_search", &q),
            RepeatAction::Proceed,
            "repeat after 4+ novel calls must be treated as fresh"
        );
    }

    #[test]
    fn decay_requires_enough_novel_work() {
        let mut guard = RepeatCallGuard::new();
        let q = json!({"path": "same"});
        assert_eq!(guard.record("read_file", &q), RepeatAction::Proceed);
        // Only 3 novel calls in between — below the decay threshold.
        for i in 0..3 {
            let _ = guard.record("read_file", &json!({"path": format!("n{i}")}));
        }
        assert_eq!(
            guard.record("read_file", &q),
            RepeatAction::Proceed,
            "second call (count 2) still proceeds"
        );
        // 3 more novel calls, then the repeat lands on count 3 = Hint, not a
        // reset — decay needs >= 4 novel calls between occurrences.
        for i in 3..6 {
            let _ = guard.record("read_file", &json!({"path": format!("n{i}")}));
        }
        assert!(
            matches!(guard.record("read_file", &q), RepeatAction::Hint(_)),
            "insufficient progress must keep the count climbing"
        );
    }

    #[test]
    fn alternating_repeats_without_progress_still_stop() {
        // A/B alternation never accrues novel work between repeats — the
        // classic degenerate loop must still escalate to Stop.
        let mut guard = RepeatCallGuard::new();
        let a = json!({"query": "a"});
        let b = json!({"query": "b"});
        let mut a_stopped = false;
        let mut b_stopped = false;
        for _ in 0..12 {
            if let RepeatAction::Stop(_) = guard.record("web_search", &a) {
                a_stopped = true;
            }
            if let RepeatAction::Stop(_) = guard.record("web_search", &b) {
                b_stopped = true;
            }
        }
        assert!(a_stopped && b_stopped, "alternation must still be stopped");
    }

    #[test]
    fn stop_message_carries_surrender_path_and_tool_advice() {
        let mut guard = RepeatCallGuard::new();
        let q = json!({"query": "stuck"});
        for _ in 0..7 {
            let _ = guard.record("web_search", &q);
        }
        match guard.record("web_search", &q) {
            RepeatAction::Stop(msg) => {
                assert!(
                    msg.contains("site:"),
                    "web_search stop must name concrete alternatives: {msg}"
                );
                assert!(
                    msg.contains("结束回合"),
                    "stop must offer the surrender path (end turn with partial result): {msg}"
                );
            }
            other => panic!("8th identical call must Stop, got {other:?}"),
        }

        // Unknown tools keep a generic (but still actionable) message.
        let mut guard = RepeatCallGuard::new();
        let x = json!({});
        for _ in 0..7 {
            let _ = guard.record("some_custom_tool", &x);
        }
        match guard.record("some_custom_tool", &x) {
            RepeatAction::Stop(msg) => {
                assert!(
                    msg.contains("结束回合"),
                    "generic stop must still offer the surrender path: {msg}"
                );
                assert!(
                    !msg.contains("site:"),
                    "generic stop must not carry web_search-specific advice: {msg}"
                );
            }
            other => panic!("8th identical call must Stop, got {other:?}"),
        }
    }

    #[test]
    fn decay_does_not_duplicate_the_recency_window() {
        // Regression: a decayed key is already in the `order` window; the
        // re-record must refresh its position, not push a second copy (a
        // duplicate entry would corrupt FIFO eviction accounting).
        let mut guard = RepeatCallGuard::new();
        let q = json!({"query": "x"});
        let _ = guard.record("web_search", &q);
        for i in 0..4 {
            let _ = guard.record("web_search", &json!({"query": format!("n{i}")}));
        }
        assert_eq!(guard.record("web_search", &q), RepeatAction::Proceed);
        assert_eq!(
            guard
                .order
                .iter()
                .filter(|&&k| k == canonical_key("web_search", &q))
                .count(),
            1,
            "decayed key must appear exactly once in the recency window"
        );
    }

    #[test]
    fn reset_session_restores_first_call_behavior() {
        // `/new` regression: the guard lives on a reused AgentLoop, so its
        // counts must be cleared at the session boundary. Burn a key past the
        // Stop threshold, reset, and the same key proceeds again from 1.
        let mut guard = RepeatCallGuard::new();
        let input = json!({"path": "x"});
        for _ in 0..8 {
            let _ = guard.record("read_file", &input);
        }
        assert!(matches!(
            guard.record("read_file", &input),
            RepeatAction::Stop(_)
        ));

        guard.reset_session();
        assert_eq!(guard.record("read_file", &input), RepeatAction::Proceed);

        // Wait-tool per-turn state is cleared too: first wait proceeds.
        assert_eq!(
            guard.record("wait_for_user", &json!({})),
            RepeatAction::Proceed
        );
    }
}
