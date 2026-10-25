//! Context management for preventing context window overflow.
//!
//! This module provides:
//! - Token counting and context size tracking
//! - Message windowing (sliding window of recent messages)
//! - Automatic summarization when context approaches limits
//! - Preservation of critical messages (system prompt, memory, etc.)

use nca_common::message::{Message, MessageContent, MessageToolCall, Role};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Name of the tool whose latest snapshot survives auto-summarization.
const TODO_TOOL_NAME: &str = "update_todos";
/// Heading under which the preserved todo snapshot is appended to the
/// summary system message (deterministic content, never LLM-paraphrased).
/// Shared by `apply_summary` and the anchored sliding window so the
/// producer and the resume extractor can never drift apart.
pub(crate) const TODO_PRESERVED_HEADING: &str = "### Current todo list (preserved verbatim)";
/// Heading of the summary system message produced by `apply_summary` (and
/// the summary section of the anchored-window anchor). Also matched by the
/// supervisor's resume extractor to recover the last AI summary.
pub(crate) const SUMMARY_HEADER: &str = "## Conversation Summary (Earlier Context)";
/// Heading of the UNCONDITIONAL base anchor the anchored sliding window
/// inserts whenever it truncates — even with no summary and no todos, the
/// model must learn that content was dropped (the original incident:
/// first compaction + failed summary side-call + no `update_todos` use
/// left a bare window indistinguishable from intact history).
pub(crate) const ANCHOR_TRUNCATED_HEADER: &str = "## Context truncated (sliding window)";
/// Hard cap (chars) on the preserved todo snapshot — truncate-protected so a
/// pathological `update_todos` payload cannot bloat the summary message.
const TODO_PRESERVED_MAX_CHARS: usize = 4_096;
/// Cap for the summary section inside the anchored-window anchor message.
const ANCHOR_SUMMARY_MAX_CHARS: usize = 2_000;

/// The latest `update_todos` interaction in a message slice.
///
/// Anchored on the LAST tool *result* whose matching tool_use called
/// `update_todos` — the result is the proof the replacement was accepted;
/// arguments alone (a resultless, in-flight call) are only used as a
/// degraded fallback. The structured todo list itself lives in the call's
/// `arguments.todos` (the result text is just a one-line confirmation), so
/// the preserved snapshot renders both.
struct TodoSnapshot<'a> {
    call: &'a MessageToolCall,
    result: Option<&'a Message>,
}

fn last_update_todos_snapshot(messages: &[Message]) -> Option<TodoSnapshot<'_>> {
    let mut todo_calls: HashMap<&str, &MessageToolCall> = HashMap::new();
    for msg in messages {
        if let Some(calls) = &msg.tool_calls {
            for call in calls {
                if call.name == TODO_TOOL_NAME {
                    todo_calls.insert(call.id.as_str(), call);
                }
            }
        }
    }
    if todo_calls.is_empty() {
        return None;
    }

    // Latest result wins: later snapshots supersede earlier ones.
    let last_result = messages.iter().rev().find(|msg| {
        msg.role == Role::Tool
            && msg
                .tool_call_id
                .as_deref()
                .is_some_and(|id| todo_calls.contains_key(id))
    });

    match last_result {
        Some(result) => {
            let call = todo_calls.get(result.tool_call_id.as_deref()?)?;
            Some(TodoSnapshot {
                call,
                result: Some(result),
            })
        }
        // No result at all (should not happen — pairs are invariant): fall
        // back to the last call's arguments alone.
        None => {
            let call = messages
                .iter()
                .rev()
                .flat_map(|m| m.tool_calls.iter().flatten())
                .find(|c| c.name == TODO_TOOL_NAME)?;
            Some(TodoSnapshot { call, result: None })
        }
    }
}

/// Render a [`TodoSnapshot`] as deterministic text: the `todos` argument
/// verbatim (pretty JSON) plus the tool result confirmation line.
fn render_todo_snapshot(snapshot: &TodoSnapshot<'_>) -> String {
    let args = snapshot
        .call
        .arguments
        .get("todos")
        .unwrap_or(&snapshot.call.arguments);
    let mut out = serde_json::to_string_pretty(args).unwrap_or_else(|_| args.to_string());
    if let Some(result) = snapshot.result {
        out.push_str("\n\n(tool result: ");
        out.push_str(&result.content.to_summary_text());
        out.push(')');
    }
    out
}

/// Cap a string at `cap` chars, marking truncation.
fn cap_chars(text: &str, cap: usize) -> String {
    if text.chars().count() <= cap {
        return text.to_string();
    }
    let mut out: String = text.chars().take(cap).collect();
    out.push_str("\n...[truncated]");
    out
}

/// Strip every `update_todos` tool_use/tool_result pair except
/// `keep_call_id` — both members of each pair, so the no-orphaned-results
/// invariant is preserved. An assistant carrier that loses its last call
/// keeps its text content with `tool_calls` set to `None`.
fn strip_older_update_todos_pairs(messages: &[Message], keep_call_id: &str) -> Vec<Message> {
    let mut drop_ids: HashSet<&str> = HashSet::new();
    for msg in messages {
        if let Some(calls) = &msg.tool_calls {
            for call in calls {
                if call.name == TODO_TOOL_NAME && call.id != keep_call_id {
                    drop_ids.insert(call.id.as_str());
                }
            }
        }
    }
    if drop_ids.is_empty() {
        return messages.to_vec();
    }

    messages
        .iter()
        .filter_map(|msg| {
            if msg.role == Role::Tool
                && msg
                    .tool_call_id
                    .as_deref()
                    .is_some_and(|id| drop_ids.contains(id))
            {
                return None; // drop the tool_result member
            }
            let has_dropped_call = msg
                .tool_calls
                .as_ref()
                .is_some_and(|calls| calls.iter().any(|c| drop_ids.contains(c.id.as_str())));
            if !has_dropped_call {
                return Some(msg.clone());
            }
            // Drop the tool_use member from the carrier's call list.
            let mut cloned = msg.clone();
            let kept: Vec<MessageToolCall> = cloned
                .tool_calls
                .take()
                .unwrap_or_default()
                .into_iter()
                .filter(|c| !drop_ids.contains(c.id.as_str()))
                .collect();
            cloned.tool_calls = (!kept.is_empty()).then_some(kept);
            Some(cloned)
        })
        .collect()
}

/// Configuration for context management behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextManagerConfig {
    /// Target context size in tokens (rough approximation).
    /// When context exceeds this, we'll start compacting.
    pub context_window_target: usize,
    /// Maximum messages to retain after compaction.
    /// This is the "sliding window" size for recent conversation.
    pub max_retained_messages: usize,
    /// Percentage (0-100) of context window that triggers auto-summarize.
    /// E.g., 75 means start summarizing when at 75% of context_window_target.
    pub auto_summarize_threshold: u8,
    /// Enable automatic context summarization.
    pub enable_auto_summarize: bool,
    /// Maximum tokens per message before truncation during summary.
    pub max_message_chars_for_summary: usize,
}

impl Default for ContextManagerConfig {
    fn default() -> Self {
        Self {
            // Default to ~32k tokens target (rough: chars / 4 ≈ tokens)
            context_window_target: 32_000,
            // Keep last 50 messages + system + memory
            max_retained_messages: 50,
            // Start summarizing at 75% of target
            auto_summarize_threshold: 75,
            enable_auto_summarize: true,
            // Truncate very long messages when preparing for summary
            max_message_chars_for_summary: 10_000,
        }
    }
}

/// Context statistics exposed via the CLI/API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextStats {
    /// Model name
    pub model: String,
    /// Model context window in tokens
    pub context_window: usize,
    /// Estimated total tokens in context
    pub estimated_tokens: usize,
    /// Number of messages in context
    pub message_count: usize,
    /// Percentage of context window used (0-100)
    pub usage_percent: u8,
    /// Whether context is approaching limit
    pub needs_attention: bool,
    /// Whether auto-summarize should trigger
    pub should_summarize: bool,
}

/// Manages conversation context to prevent token overflow.
///
/// The context manager maintains:
/// - A sliding window of recent user/assistant messages
/// - Preservation of system messages (always at start)
/// - Track tool messages that follow the relevant turns
pub struct ContextManager {
    config: ContextManagerConfig,
    model: String,
}

impl ContextManager {
    pub fn new(config: ContextManagerConfig, model: String) -> Self {
        Self { config, model }
    }

    pub fn with_default_config(model: String) -> Self {
        Self::new(ContextManagerConfig::default(), model)
    }

    /// Calculate estimated token count for a message.
    /// Uses a rough approximation: tokens ≈ characters / 4
    pub fn estimate_tokens(message: &Message) -> usize {
        // Tool messages tend to be more token-dense
        let divisor = match message.role {
            Role::Tool => 3.5,
            Role::System => 4.0,
            _ => 4.0,
        };

        let content_tokens = message.content.approx_chars() as f64 / divisor;

        // Add overhead for tool calls
        let tool_call_overhead = message
            .tool_calls
            .as_ref()
            .map(|calls| calls.len() * 50) // ~50 tokens per tool call structure
            .unwrap_or(0);

        (content_tokens as usize) + tool_call_overhead + 10 // ~10 tokens base overhead
    }

    /// Calculate estimated token count for a slice of messages.
    pub fn estimate_tokens_for_slice(messages: &[Message]) -> usize {
        messages.iter().map(Self::estimate_tokens).sum()
    }

    /// Get current context statistics.
    pub fn stats(&self, messages: &[Message]) -> ContextStats {
        let estimated_tokens = Self::estimate_tokens_for_slice(messages);
        let message_count = messages.len();

        let context_window = self.config.context_window_target.max(1);
        let usage_percent =
            ((estimated_tokens as f64 / context_window as f64) * 100.0).min(100.0) as u8;

        let needs_attention = usage_percent >= 80;
        let should_summarize = self.config.enable_auto_summarize
            && usage_percent >= self.config.auto_summarize_threshold;

        ContextStats {
            model: self.model.clone(),
            context_window,
            estimated_tokens,
            message_count,
            usage_percent,
            needs_attention,
            should_summarize,
        }
    }

    /// Count system messages at the start of the list.
    /// System messages should always be preserved.
    fn count_system_messages(messages: &[Message]) -> usize {
        messages
            .iter()
            .take_while(|m| m.role == Role::System)
            .count()
    }

    /// Adjust a cutoff index so it never lands inside a tool_use/tool_result group.
    /// If the message at `cutoff` is a Role::Tool, walk backwards to include the
    /// preceding assistant message that contains the matching tool_calls.
    /// `min_cutoff` prevents walking past system messages.
    fn adjust_cutoff_for_tool_groups(
        messages: &[Message],
        mut cutoff: usize,
        min_cutoff: usize,
    ) -> usize {
        while cutoff > min_cutoff && cutoff < messages.len() && messages[cutoff].role == Role::Tool
        {
            cutoff -= 1;
        }
        cutoff
    }

    /// Check if the context needs compaction.
    pub fn needs_compaction(&self, messages: &[Message]) -> bool {
        let stats = self.stats(messages);
        stats.should_summarize || stats.needs_attention
    }

    /// Compact messages using a sliding window strategy.
    /// Returns the indices of messages to keep.
    ///
    /// Strategy:
    /// 1. Always keep all system messages (at start)
    /// 2. Keep recent messages up to max_retained_messages
    /// 3. Mark older messages for summarization
    pub fn get_compaction_plan(&self, messages: &[Message], system_count: usize) -> CompactionPlan {
        // Calculate how many non-system messages we can keep
        let non_system_count = messages.len().saturating_sub(system_count);
        let keep_non_system = non_system_count.saturating_sub(self.config.max_retained_messages);

        // If we need to compact, find the boundary
        if keep_non_system > 0 {
            // Keep last max_retained_messages non-system messages
            let cutoff_index = messages.len() - self.config.max_retained_messages;

            CompactionPlan {
                keep_indices: (0..cutoff_index)
                    .rev()
                    .take(self.config.max_retained_messages)
                    .collect(),
                summarize_range: Some(SummarizeRange {
                    start: system_count,
                    end: cutoff_index,
                }),
                preserve_system: true,
            }
        } else {
            CompactionPlan {
                keep_indices: (0..messages.len()).collect(),
                summarize_range: None,
                preserve_system: true,
            }
        }
    }

    /// Get messages that should be summarized.
    /// These are the older messages that will be replaced by a summary.
    ///
    /// Older `update_todos` tool_use/tool_result pairs are stripped (both
    /// members, pair integrity preserved); only the latest snapshot stays —
    /// stale snapshots neither bloat nor drift the summarizer input, and the
    /// latest one is re-attached verbatim by [`Self::apply_summary`].
    pub fn get_messages_to_summarize(&self, messages: &[Message]) -> Vec<Message> {
        let system_count = Self::count_system_messages(messages);
        let plan = self.get_compaction_plan(messages, system_count);

        if let Some(range) = plan.summarize_range {
            // Skip system messages, get the middle-old messages
            let start = (range.start).max(system_count);
            let slice = &messages[start..range.end];
            match last_update_todos_snapshot(slice) {
                Some(snapshot) => strip_older_update_todos_pairs(slice, &snapshot.call.id),
                None => slice.to_vec(),
            }
        } else {
            Vec::new()
        }
    }

    /// Apply a summary to the context, replacing old messages.
    /// Returns the new message list with the summary inserted.
    pub fn apply_summary(&self, messages: &[Message], summary: &str) -> Vec<Message> {
        let system_count = Self::count_system_messages(messages);
        let plan = self.get_compaction_plan(messages, system_count);

        // If there's nothing to summarize, just return the original messages
        let Some(range) = plan.summarize_range else {
            return messages.to_vec();
        };

        // Get recent messages to keep (everything after the summarize range).
        // Adjust to avoid splitting tool_use/tool_result groups.
        let recent_start = Self::adjust_cutoff_for_tool_groups(messages, range.end, system_count);

        // T2-3: the latest update_todos snapshot in the summarized range is
        // re-attached to the summary system message verbatim (capped) — the
        // structured todo state survives compaction deterministically,
        // never LLM-paraphrased. Uses the same range as
        // `get_messages_to_summarize` so the preserved snapshot is exactly
        // the one kept there.
        let summarize_start = range.start.max(system_count);
        let todo_block =
            last_update_todos_snapshot(&messages[summarize_start..range.end]).map(|snapshot| {
                format!(
                    "\n\n{}\n\n{}",
                    TODO_PRESERVED_HEADING,
                    cap_chars(&render_todo_snapshot(&snapshot), TODO_PRESERVED_MAX_CHARS)
                )
            });

        // Build new message list: system + summary + recent
        let mut result = Vec::with_capacity(system_count + 10);

        // Add system messages
        result.extend(messages.iter().take(system_count).cloned());

        // Insert summary as a special system message. The todo block alone
        // still warrants a message when the AI returned an empty summary —
        // dropping it would lose the preserved state.
        if !summary.trim().is_empty() || todo_block.is_some() {
            result.push(Message {
                role: Role::System,
                content: MessageContent::Text(format!(
                    "{}\n\n{}{}",
                    SUMMARY_HEADER,
                    summary,
                    todo_block.unwrap_or_default()
                )),
                tool_call_id: None,
                tool_calls: None,
                reasoning_content: None,
            });
        }

        // Add recent messages
        result.extend_from_slice(&messages[recent_start..]);

        result
    }

    /// Get a sliding window of recent messages for context.
    /// Preserves system messages and keeps recent conversation.
    /// Ensures tool_use/tool_result groups are never split.
    pub fn get_sliding_window(
        &self,
        messages: &[Message],
        max_messages: Option<usize>,
    ) -> Vec<Message> {
        let max = max_messages.unwrap_or(self.config.max_retained_messages);
        let system_count = Self::count_system_messages(messages);

        if messages.len() <= max {
            return messages.to_vec();
        }

        // Keep system messages + last (max - system_count) messages
        let keep_count = max.saturating_sub(system_count);
        let cutoff = messages.len() - keep_count;
        let cutoff = Self::adjust_cutoff_for_tool_groups(messages, cutoff, system_count);

        let mut result: Vec<Message> = messages[..system_count].to_vec();
        result.extend_from_slice(&messages[cutoff..]);

        result
    }

    /// Anchored sliding window — the fallback-compaction path's answer to
    /// "the window dropped the thread". Same windowing as
    /// [`Self::get_sliding_window`], plus ONE system anchor message stitched
    /// right after the leading system messages whenever truncation actually
    /// dropped content. The anchor is UNCONDITIONAL on truncation (PRD-1):
    /// the base section always states that content was dropped and why, so
    /// even the worst case — first compaction, failed summary side-call
    /// (`last_summary == None`), no `update_todos` use — still leaves the
    /// model knowing the history has a hole. Optional sections append in
    /// order:
    ///
    /// - [`SUMMARY_HEADER`] + `last_summary` (capped) — a later fallback
    ///   does not erase an earlier AI compaction's memory.
    /// - [`TODO_PRESERVED_HEADING`] + the latest `update_todos` snapshot
    ///   from the DROPPED region only (`&messages[..cutoff]`) — a snapshot
    ///   still inside the window is live history and must not be duplicated.
    ///
    /// No truncation → the input verbatim (no anchor, no system synthesis).
    pub fn get_anchored_sliding_window(
        &self,
        messages: &[Message],
        last_summary: Option<&str>,
    ) -> Vec<Message> {
        let max = self.config.max_retained_messages;
        let system_count = Self::count_system_messages(messages);

        if messages.len() <= max {
            return messages.to_vec();
        }

        let keep_count = max.saturating_sub(system_count);
        let cutoff = messages.len() - keep_count;
        let cutoff = Self::adjust_cutoff_for_tool_groups(messages, cutoff, system_count);

        let dropped = &messages[system_count..cutoff];
        let mut anchor = format!(
            "{}\n\nEarlier conversation ({} messages) was dropped because context \
             summarization was unavailable (provider error). If task context appears to be \
             missing, re-read the task artifacts or ask the user before proceeding.",
            ANCHOR_TRUNCATED_HEADER,
            dropped.len(),
        );
        if let Some(summary) = last_summary.filter(|s| !s.trim().is_empty()) {
            anchor.push_str("\n\n");
            anchor.push_str(SUMMARY_HEADER);
            anchor.push_str("\n\n");
            anchor.push_str(&cap_chars(summary, ANCHOR_SUMMARY_MAX_CHARS));
        }
        if let Some(snapshot) = last_update_todos_snapshot(dropped) {
            anchor.push_str("\n\n");
            anchor.push_str(TODO_PRESERVED_HEADING);
            anchor.push_str("\n\n");
            anchor.push_str(&cap_chars(
                &render_todo_snapshot(&snapshot),
                TODO_PRESERVED_MAX_CHARS,
            ));
        }

        let mut result: Vec<Message> =
            Vec::with_capacity(system_count + 2 + (messages.len() - cutoff));
        result.extend_from_slice(&messages[..system_count]);
        result.push(Message::system(anchor));
        result.extend_from_slice(&messages[cutoff..]);
        result
    }

    /// Truncate very long messages for summary generation.
    pub fn prepare_for_summary(&self, messages: &[Message]) -> Vec<Message> {
        messages
            .iter()
            .map(|m| {
                if m.content.approx_chars() > self.config.max_message_chars_for_summary {
                    let truncated: String = m
                        .content
                        .to_summary_text()
                        .chars()
                        .take(self.config.max_message_chars_for_summary)
                        .collect();
                    Message {
                        role: m.role.clone(),
                        content: MessageContent::Text(format!("{}...[truncated]", truncated)),
                        tool_call_id: m.tool_call_id.clone(),
                        tool_calls: m.tool_calls.clone(),
                        reasoning_content: m.reasoning_content.clone(),
                    }
                } else {
                    m.clone()
                }
            })
            .collect()
    }

    /// Generate a prompt for the AI to summarize conversation.
    pub fn summary_prompt(&self, messages: &[Message]) -> String {
        let prepared = self.prepare_for_summary(messages);
        let stats = self.stats(&prepared);

        format!(
            r#"Please summarize the following conversation concisely.

The summary should:
1. Capture the key topics and goals discussed
2. Note any important decisions or findings
3. Preserve critical context (file paths, variable names, errors, etc.)
4. Be written as if you're continuing the conversation

Keep the summary under 500 words.

Current context stats:
- Messages: {}
- Estimated tokens: {} (target: {})

---

Conversation to summarize:

"#,
            stats.message_count, stats.estimated_tokens, self.config.context_window_target
        )
    }

    /// Get the config.
    pub fn config(&self) -> &ContextManagerConfig {
        &self.config
    }

    /// Update the config.
    pub fn set_config(&mut self, config: ContextManagerConfig) {
        self.config = config;
    }
}

/// Plan for compacting messages.
#[derive(Debug, Clone)]
pub struct CompactionPlan {
    /// Indices of messages to keep
    pub keep_indices: Vec<usize>,
    /// Range of messages to summarize (if any)
    pub summarize_range: Option<SummarizeRange>,
    /// Whether to preserve system messages
    pub preserve_system: bool,
}

/// Range of messages to summarize.
#[derive(Debug, Clone)]
pub struct SummarizeRange {
    pub start: usize,
    pub end: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_message(role: Role, content: &str) -> Message {
        Message {
            role,
            content: MessageContent::Text(content.to_string()),
            tool_call_id: None,
            tool_calls: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn test_estimate_tokens() {
        let msg = make_message(Role::User, "Hello, world!");
        assert!(ContextManager::estimate_tokens(&msg) > 0);
    }

    #[test]
    fn test_stats() {
        let config = ContextManagerConfig {
            context_window_target: 1000,
            max_retained_messages: 10,
            auto_summarize_threshold: 50,
            enable_auto_summarize: true,
            max_message_chars_for_summary: 1000,
        };
        let manager = ContextManager::new(config, "test-model".to_string());

        let messages = vec![
            make_message(Role::System, "You are a helpful assistant."),
            make_message(Role::User, "Hello"),
            make_message(Role::Assistant, "Hi there!"),
        ];

        let stats = manager.stats(&messages);
        assert!(stats.message_count == 3);
        assert!(stats.estimated_tokens > 0);
        assert_eq!(stats.model, "test-model");
        assert_eq!(stats.context_window, 1000); // stats use configured target
    }

    #[test]
    fn test_sliding_window() {
        let config = ContextManagerConfig {
            context_window_target: 32000,
            max_retained_messages: 3,
            auto_summarize_threshold: 75,
            enable_auto_summarize: true,
            max_message_chars_for_summary: 10000,
        };
        let manager = ContextManager::new(config, "test-model".to_string());

        let messages: Vec<Message> = (0..10)
            .map(|i| make_message(Role::User, &format!("Message {}", i)))
            .collect();

        let window = manager.get_sliding_window(&messages, None);
        // System messages + last 3
        assert!(window.len() <= 3);
    }

    #[test]
    fn test_apply_summary() {
        // Use a config that will trigger compaction for 5 messages
        let config = ContextManagerConfig {
            context_window_target: 32_000,
            max_retained_messages: 2, // Only keep 2 messages, forcing compaction
            auto_summarize_threshold: 75,
            enable_auto_summarize: true,
            max_message_chars_for_summary: 10_000,
        };
        let manager = ContextManager::new(config, "test-model".to_string());

        let messages = vec![
            make_message(Role::System, "System"),
            make_message(Role::User, "Hello"),
            make_message(Role::Assistant, "Hi"),
            make_message(Role::User, "How are you?"),
            make_message(Role::Assistant, "I'm fine!"),
        ];

        let summary = "User greeted the assistant and asked how it was doing.";
        let result = manager.apply_summary(&messages, summary);

        // Should have system + summary + recent messages (2)
        // Note: The compaction plan keeps last 2 non-system messages
        assert!(
            result.len() >= 2,
            "Expected at least 2 messages, got {}",
            result.len()
        );
        // The summary should be in a system message
        assert!(
            result
                .iter()
                .any(|m| m.content.to_summary_text().contains("Conversation Summary"))
        );
    }

    use nca_common::message::MessageToolCall;

    fn make_tool_call(id: &str, name: &str) -> MessageToolCall {
        MessageToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: serde_json::json!({}),
        }
    }

    /// Helper: assert no Role::Tool message appears without its matching
    /// assistant tool_use in the preceding assistant message.
    fn assert_no_orphaned_tool_results(messages: &[Message]) {
        let mut expected_tool_ids: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for msg in messages {
            match msg.role {
                Role::Assistant => {
                    expected_tool_ids.clear();
                    if let Some(calls) = &msg.tool_calls {
                        for call in calls {
                            expected_tool_ids.insert(call.id.clone());
                        }
                    }
                }
                Role::Tool => {
                    let id = msg.tool_call_id.as_deref().unwrap_or("");
                    assert!(
                        expected_tool_ids.contains(id),
                        "Orphaned tool_result with id '{}' — no matching tool_use in preceding assistant message. Messages: {:?}",
                        id,
                        messages
                            .iter()
                            .map(|m| (&m.role, m.tool_call_id.as_deref()))
                            .collect::<Vec<_>>()
                    );
                }
                _ => {
                    expected_tool_ids.clear();
                }
            }
        }
    }

    // === Anchored sliding window (fallback-compaction anchors) ===

    fn anchored_manager() -> ContextManager {
        ContextManager::new(
            ContextManagerConfig {
                context_window_target: 32_000,
                max_retained_messages: 4,
                auto_summarize_threshold: 75,
                enable_auto_summarize: true,
                max_message_chars_for_summary: 10_000,
            },
            "test-model".to_string(),
        )
    }

    /// History long enough to force truncation, with an update_todos pair in
    /// the to-be-dropped region and a recent window after it.
    fn anchored_history() -> Vec<Message> {
        vec![
            make_message(Role::System, "system prompt"),
            make_message(Role::User, "kick off"),
            Message::assistant_with_tool_calls(
                "planning",
                vec![make_todo_call("todo-anchored", "anchor-todo-marker")],
            ),
            Message::tool(
                "todo-anchored",
                "Updated 1 todos (0 pending, 1 in_progress, 0 completed)",
            ),
            make_message(Role::Assistant, "working"),
            make_message(Role::User, "mid question"),
            make_message(Role::Assistant, "mid answer"),
            make_message(Role::User, "recent question"),
            make_message(Role::Assistant, "recent answer"),
        ]
    }

    #[test]
    fn anchored_window_merges_summary_and_todo_into_single_anchor_after_system() {
        let manager = anchored_manager();
        let history = anchored_history();
        let window = manager.get_anchored_sliding_window(&history, Some("earlier work summary"));

        // Exactly ONE anchor message, directly after the system prefix,
        // carrying the unconditional base section + summary + todo sections
        // in order.
        let texts: Vec<String> = window.iter().map(|m| m.content.to_summary_text()).collect();
        let sys = texts
            .iter()
            .position(|t| t == "system prompt")
            .expect("system");
        let anchor_idx = texts
            .iter()
            .position(|t| t.contains(ANCHOR_TRUNCATED_HEADER))
            .expect("base anchor present");
        assert_eq!(anchor_idx, sys + 1, "single anchor directly after system");
        assert_eq!(
            window[anchor_idx].role,
            Role::System,
            "anchor is a system message"
        );
        let anchor = &texts[anchor_idx];
        assert!(
            anchor.contains("5 messages"),
            "dropped count in base section: {anchor}"
        );
        let summary_pos = anchor.find(SUMMARY_HEADER).expect("summary section");
        let todo_pos = anchor.find(TODO_PRESERVED_HEADING).expect("todo section");
        assert!(
            summary_pos < todo_pos,
            "summary section before todo section"
        );
        assert!(anchor.contains("earlier work summary"));
        assert!(anchor.contains("anchor-todo-marker"));

        assert_no_orphaned_tool_results(&window);
    }

    #[test]
    fn anchored_window_without_truncation_returns_input_verbatim() {
        let manager = anchored_manager();
        let short = vec![
            make_message(Role::System, "system prompt"),
            make_message(Role::User, "one"),
            make_message(Role::Assistant, "two"),
        ];
        let window = manager.get_anchored_sliding_window(&short, Some("summary"));
        assert_eq!(window.len(), short.len());
        assert_eq!(window, short, "no truncation → no anchors, verbatim input");
    }

    #[test]
    fn anchored_window_base_anchor_is_unconditional_on_truncation() {
        // PRD-1 original incident shape: first compaction (no prior summary)
        // + failed summary side-call (None) + no update_todos use. The base
        // anchor must STILL fire — a bare window indistinguishable from
        // intact history is the failure being fixed.
        let manager = anchored_manager();
        let history = vec![
            make_message(Role::System, "system prompt"),
            make_message(Role::User, "old 1"),
            make_message(Role::Assistant, "old 2"),
            make_message(Role::User, "old 3"),
            make_message(Role::Assistant, "old 4"),
            make_message(Role::User, "old 5"),
            make_message(Role::Assistant, "old 6"),
            make_message(Role::User, "recent question"),
            make_message(Role::Assistant, "recent answer"),
        ];
        let window = manager.get_anchored_sliding_window(&history, None);

        let texts: Vec<String> = window.iter().map(|m| m.content.to_summary_text()).collect();
        let sys = texts
            .iter()
            .position(|t| t == "system prompt")
            .expect("system");
        let anchor_idx = texts
            .iter()
            .position(|t| t.contains(ANCHOR_TRUNCATED_HEADER))
            .expect("base anchor must exist even with None/None inputs");
        assert_eq!(anchor_idx, sys + 1);
        assert!(
            texts[anchor_idx].contains("5 messages"),
            "dropped count must be stated: {}",
            texts[anchor_idx]
        );
        assert!(
            !texts[anchor_idx].contains(SUMMARY_HEADER),
            "no summary section"
        );
        assert!(
            !texts[anchor_idx].contains(TODO_PRESERVED_HEADING),
            "no todo section"
        );
    }

    #[test]
    fn anchored_window_without_summary_still_anchors_todo() {
        let manager = anchored_manager();
        let history = anchored_history();
        let window = manager.get_anchored_sliding_window(&history, None);
        let texts: Vec<String> = window.iter().map(|m| m.content.to_summary_text()).collect();
        let anchor_idx = texts
            .iter()
            .position(|t| t.contains(ANCHOR_TRUNCATED_HEADER))
            .expect("base anchor");
        assert!(!texts[anchor_idx].contains(SUMMARY_HEADER));
        assert!(texts[anchor_idx].contains(TODO_PRESERVED_HEADING));
        assert!(texts[anchor_idx].contains("anchor-todo-marker"));
    }

    #[test]
    fn anchored_window_todo_inside_window_is_not_duplicated() {
        // The update_todos pair sits INSIDE the retained window: the anchor
        // must not fire (the window already holds it live), and the pair must
        // survive intact.
        let manager = anchored_manager();
        let history = vec![
            make_message(Role::System, "system prompt"),
            make_message(Role::User, "old 1"),
            make_message(Role::Assistant, "old 2"),
            make_message(Role::User, "old 3"),
            make_message(Role::Assistant, "old 4"),
            Message::assistant_with_tool_calls(
                "planning",
                vec![make_todo_call("todo-live", "live-todo-marker")],
            ),
            Message::tool("todo-live", "Updated 1 todos"),
            make_message(Role::User, "recent question"),
        ];
        let window = manager.get_anchored_sliding_window(&history, None);

        let texts: Vec<String> = window.iter().map(|m| m.content.to_summary_text()).collect();
        assert!(
            !texts
                .iter()
                .any(|t| t.contains("Current todo list (preserved verbatim)"))
        );
        assert_no_orphaned_tool_results(&window);
        let ids: Vec<&str> = window
            .iter()
            .flat_map(|m| m.tool_calls.iter().flatten())
            .map(|c| c.id.as_str())
            .collect();
        assert!(ids.contains(&"todo-live"), "live pair must stay in window");
    }

    #[test]
    fn anchored_window_never_breaks_tool_pairing() {
        // Force the naive cutoff onto a tool_result message: the adjust walk
        // must keep the pair together with the anchors present.
        let manager = ContextManager::new(
            ContextManagerConfig {
                context_window_target: 32_000,
                max_retained_messages: 3,
                auto_summarize_threshold: 75,
                enable_auto_summarize: true,
                max_message_chars_for_summary: 10_000,
            },
            "test-model".to_string(),
        );
        let history = vec![
            make_message(Role::System, "system prompt"),
            make_message(Role::User, "kick off"),
            Message::assistant_with_tool_calls(
                "planning",
                vec![make_todo_call("todo-anchor-2", "marker")],
            ),
            Message::tool("todo-anchor-2", "Updated 1 todos"),
            Message::assistant_with_tool_calls(
                "reading",
                vec![make_tool_call("read-3", "read_file")],
            ),
            Message::tool("read-3", "file contents"),
            make_message(Role::User, "recent question"),
        ];
        let window = manager.get_anchored_sliding_window(&history, Some("summary"));
        assert_no_orphaned_tool_results(&window);
    }

    #[test]
    fn sliding_window_preserves_tool_use_result_pairs() {
        // max_retained=4: cutoff would land on tool result msg if naive
        let config = ContextManagerConfig {
            context_window_target: 32_000,
            max_retained_messages: 4,
            auto_summarize_threshold: 75,
            enable_auto_summarize: true,
            max_message_chars_for_summary: 10_000,
        };
        let manager = ContextManager::new(config, "test-model".to_string());

        // 10 messages: user, assistant+tool_calls, tool_result, user, ...
        let messages = vec![
            make_message(Role::System, "system prompt"),
            make_message(Role::User, "msg 1"),
            make_message(Role::Assistant, "reply 1"),
            make_message(Role::User, "msg 2"),
            // This is the critical group: assistant calls tool, then tool result
            Message::assistant_with_tool_calls(
                "Let me check",
                vec![make_tool_call("call-1", "read_file")],
            ),
            Message::tool("call-1", "file contents here"),
            // After tool result
            make_message(Role::User, "msg 3"),
            make_message(Role::Assistant, "final reply"),
        ];

        let window = manager.get_sliding_window(&messages, None);

        // The window must never contain an orphaned tool_result
        assert_no_orphaned_tool_results(&window);
    }

    #[test]
    fn apply_summary_preserves_tool_use_result_pairs() {
        // max_retained=3: cutoff lands right on the tool_result
        let config = ContextManagerConfig {
            context_window_target: 32_000,
            max_retained_messages: 3,
            auto_summarize_threshold: 75,
            enable_auto_summarize: true,
            max_message_chars_for_summary: 10_000,
        };
        let manager = ContextManager::new(config, "test-model".to_string());

        let messages = vec![
            make_message(Role::System, "system prompt"),
            make_message(Role::User, "msg 1"),
            make_message(Role::Assistant, "reply 1"),
            make_message(Role::User, "msg 2"),
            // Tool use group that straddles the naive cutoff
            Message::assistant_with_tool_calls(
                "Running tool",
                vec![make_tool_call("call-2", "bash")],
            ),
            Message::tool("call-2", "ok"),
            make_message(Role::User, "msg 3"),
            make_message(Role::Assistant, "done"),
        ];

        let result = manager.apply_summary(&messages, "Earlier conversation summary.");

        assert_no_orphaned_tool_results(&result);
    }

    #[test]
    fn sliding_window_preserves_multi_tool_call_group() {
        // Assistant calls 2 tools → 2 tool results. Window must keep entire group.
        let config = ContextManagerConfig {
            context_window_target: 32_000,
            max_retained_messages: 4,
            auto_summarize_threshold: 75,
            enable_auto_summarize: true,
            max_message_chars_for_summary: 10_000,
        };
        let manager = ContextManager::new(config, "test-model".to_string());

        let messages = vec![
            make_message(Role::System, "system prompt"),
            make_message(Role::User, "msg 1"),
            make_message(Role::Assistant, "reply 1"),
            make_message(Role::User, "msg 2"),
            Message::assistant_with_tool_calls(
                "Using two tools",
                vec![
                    make_tool_call("call-a", "read_file"),
                    make_tool_call("call-b", "grep"),
                ],
            ),
            Message::tool("call-a", "contents a"),
            Message::tool("call-b", "contents b"),
            make_message(Role::User, "msg 3"),
            make_message(Role::Assistant, "all done"),
        ];

        let window = manager.get_sliding_window(&messages, None);

        assert_no_orphaned_tool_results(&window);
    }

    // === T2-3: summarize preserves the latest update_todos snapshot ===

    fn make_todo_call(id: &str, marker: &str) -> MessageToolCall {
        MessageToolCall {
            id: id.to_string(),
            name: "update_todos".into(),
            arguments: serde_json::json!({
                "todos": [
                    {"id": "1", "content": marker, "status": "in_progress"}
                ]
            }),
        }
    }

    /// History whose summarize range holds an OLD and a NEW update_todos
    /// pair, with recent traffic after them.
    fn todo_history() -> Vec<Message> {
        vec![
            make_message(Role::System, "system prompt"),
            make_message(Role::User, "kick off"),
            // OLD todo snapshot (superseded — must be stripped).
            Message::assistant_with_tool_calls(
                "planning v1",
                vec![make_todo_call("todo-old", "stale-todo-marker")],
            ),
            Message::tool(
                "todo-old",
                "Updated 1 todos (0 pending, 1 in_progress, 0 completed)",
            ),
            make_message(Role::Assistant, "working with v1"),
            // NEW todo snapshot (must be preserved verbatim).
            Message::assistant_with_tool_calls(
                "planning v2",
                vec![make_todo_call("todo-new", "fresh-todo-marker")],
            ),
            Message::tool(
                "todo-new",
                "Updated 1 todos (0 pending, 1 in_progress, 0 completed) v2",
            ),
            // A non-todo tool pair that must survive untouched in the input.
            Message::assistant_with_tool_calls(
                "reading",
                vec![make_tool_call("read-1", "read_file")],
            ),
            Message::tool("read-1", "file contents"),
            // Recent window (max_retained = 2).
            make_message(Role::User, "recent question"),
            make_message(Role::Assistant, "recent answer"),
        ]
    }

    fn todo_manager() -> ContextManager {
        ContextManager::new(
            ContextManagerConfig {
                context_window_target: 32_000,
                max_retained_messages: 2,
                auto_summarize_threshold: 75,
                enable_auto_summarize: true,
                max_message_chars_for_summary: 10_000,
            },
            "test-model".to_string(),
        )
    }

    fn summary_message(messages: &[Message]) -> &Message {
        messages
            .iter()
            .find(|m| {
                m.role == Role::System
                    && m.content.to_summary_text().contains("Conversation Summary")
            })
            .expect("summary system message must exist")
    }

    #[test]
    fn summarize_preserves_latest_todo_snapshot_verbatim() {
        let manager = todo_manager();
        let result = manager.apply_summary(&todo_history(), "AI summary text.");

        // No orphaned tool results after compaction.
        assert_no_orphaned_tool_results(&result);

        let summary = summary_message(&result).content.to_summary_text();
        assert!(
            summary.contains("### Current todo list (preserved verbatim)"),
            "todo heading must ride along with the summary; got: {summary}"
        );
        // Verbatim arguments: the exact pretty-printed todos array.
        let expected_json = serde_json::to_string_pretty(&serde_json::json!([
            {"id": "1", "content": "fresh-todo-marker", "status": "in_progress"}
        ]))
        .unwrap();
        assert!(
            summary.contains(&expected_json),
            "the LATEST todos argument must be preserved verbatim; got: {summary}"
        );
        assert!(
            summary.contains(
                "(tool result: Updated 1 todos (0 pending, 1 in_progress, 0 completed) v2)"
            ),
            "the latest tool result line must be preserved; got: {summary}"
        );
        assert!(
            !summary.contains("stale-todo-marker"),
            "only the latest snapshot may be preserved; got: {summary}"
        );
    }

    #[test]
    fn get_messages_to_summarize_strips_older_todo_pairs() {
        let manager = todo_manager();
        let input = manager.get_messages_to_summarize(&todo_history());

        // Pair integrity of the stripped input.
        assert_no_orphaned_tool_results(&input);

        let ids: Vec<&str> = input
            .iter()
            .flat_map(|m| m.tool_calls.iter().flatten())
            .map(|c| c.id.as_str())
            .collect();
        assert!(
            !ids.contains(&"todo-old"),
            "older update_todos tool_use must be stripped; ids: {ids:?}"
        );
        assert!(
            ids.contains(&"todo-new"),
            "the latest update_todos tool_use must stay; ids: {ids:?}"
        );
        assert!(
            ids.contains(&"read-1"),
            "non-todo tool pairs must survive; ids: {ids:?}"
        );

        let texts: Vec<String> = input.iter().map(|m| m.content.to_summary_text()).collect();
        assert!(
            !texts.iter().any(|t| t.contains("stale-todo-marker")),
            "no older snapshot content may reach the summarizer"
        );
        // The old carrier keeps its text with tool_calls cleared.
        assert!(
            texts.iter().any(|t| t.contains("planning v1")),
            "the old carrier's assistant text survives with tool_calls stripped"
        );
    }

    #[test]
    fn summarize_without_todos_leaves_summary_unchanged() {
        let manager = todo_manager();
        // No update_todos anywhere in the summarize range.
        let history = vec![
            make_message(Role::System, "system prompt"),
            make_message(Role::User, "old 1"),
            make_message(Role::Assistant, "old 2"),
            Message::assistant_with_tool_calls(
                "reading",
                vec![make_tool_call("read-9", "read_file")],
            ),
            Message::tool("read-9", "file contents"),
            make_message(Role::User, "recent question"),
            make_message(Role::Assistant, "recent answer"),
        ];

        let result = manager.apply_summary(&history, "plain summary.");
        let summary = summary_message(&result).content.to_summary_text();
        assert_eq!(
            summary, "## Conversation Summary (Earlier Context)\n\nplain summary.",
            "no todo block may be appended when the range has no update_todos"
        );
    }

    #[test]
    fn preserved_todo_snapshot_is_truncation_capped() {
        let manager = todo_manager();
        let huge = "x".repeat(20_000);
        let history = vec![
            make_message(Role::System, "system prompt"),
            make_message(Role::User, "kick off"),
            Message::assistant_with_tool_calls(
                "planning huge",
                vec![MessageToolCall {
                    id: "todo-huge".into(),
                    name: "update_todos".into(),
                    arguments: serde_json::json!({
                        "todos": [
                            {"id": "1", "content": huge, "status": "pending"}
                        ]
                    }),
                }],
            ),
            Message::tool("todo-huge", "Updated 1 todos"),
            make_message(Role::User, "recent question"),
            make_message(Role::Assistant, "recent answer"),
        ];

        let result = manager.apply_summary(&history, "summary.");
        let summary = summary_message(&result).content.to_summary_text();
        assert!(
            summary.contains("...[truncated]"),
            "oversized snapshots must be capped; got {} chars",
            summary.chars().count()
        );
        // Heading + cap + truncation marker — bounded message.
        assert!(
            summary.chars().count() < 4_096 + 512,
            "summary must stay bounded; got {} chars",
            summary.chars().count()
        );
    }

    #[test]
    fn summarize_strips_older_todo_call_from_multi_call_carrier() {
        // An assistant carrier holding BOTH an older update_todos call and a
        // read_file call: only the todo member may be stripped; the read
        // pair must survive intact (extend of the orphan invariant).
        let manager = todo_manager();
        let history = vec![
            make_message(Role::System, "system prompt"),
            make_message(Role::User, "kick off"),
            Message::assistant_with_tool_calls(
                "mixed carrier",
                vec![
                    make_todo_call("todo-mixed", "stale-todo-marker"),
                    make_tool_call("read-2", "read_file"),
                ],
            ),
            Message::tool("todo-mixed", "Updated 1 todos"),
            Message::tool("read-2", "file contents"),
            // NEW todo snapshot after it (this one is preserved).
            Message::assistant_with_tool_calls(
                "planning v2",
                vec![make_todo_call("todo-latest", "fresh-todo-marker")],
            ),
            Message::tool("todo-latest", "Updated 1 todos v2"),
            make_message(Role::User, "recent question"),
            make_message(Role::Assistant, "recent answer"),
        ];

        let input = manager.get_messages_to_summarize(&history);
        assert_no_orphaned_tool_results(&input);

        let mixed_carrier = input
            .iter()
            .find(|m| {
                m.role == Role::Assistant && m.content.to_summary_text().contains("mixed carrier")
            })
            .expect("mixed carrier survives with its text");
        let carrier_ids: Vec<&str> = mixed_carrier
            .tool_calls
            .iter()
            .flatten()
            .map(|c| c.id.as_str())
            .collect();
        assert_eq!(
            carrier_ids,
            vec!["read-2"],
            "only the update_todos member is stripped from the carrier"
        );

        let result = manager.apply_summary(&history, "summary.");
        assert_no_orphaned_tool_results(&result);
        let summary = summary_message(&result).content.to_summary_text();
        assert!(summary.contains("fresh-todo-marker"));
        assert!(!summary.contains("stale-todo-marker"));
    }
}
