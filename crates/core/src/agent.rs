use nca_common::config::SmartCompactionMode;
use nca_common::event::{AgentEvent, BusyState};
use nca_common::message::{ContentPart, ImageAttachment, Message, Role};
use nca_common::tool::ToolDefinition;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::agent_driver::InboxItem;
use crate::approval::ApprovalPolicy;
use crate::cache_keepalive::KeepaliveProfile;
use crate::cost::CostTracker;
use crate::hooks::HookRunner;
use crate::provider::{Provider, ProviderError};
use crate::tool_guards::RepeatCallGuard;
use crate::tools::ToolRegistry;

/// Drives the multi-turn conversation and tool-use loop.
pub struct AgentLoop {
    pub provider: Arc<dyn Provider>,
    pub tools: ToolRegistry,
    pub approval: ApprovalPolicy,
    pub messages: Vec<Message>,
    pub model: String,
    pub cost_tracker: CostTracker,
    pub(crate) event_tx: tokio::sync::mpsc::Sender<AgentEvent>,
    pub(crate) max_turns: u32,
    pub(crate) max_tool_calls_per_turn: u32,
    pub(crate) checkpoint_interval: u32,
    pub(crate) cancel_flag: Arc<AtomicBool>,
    pub(crate) hooks: Option<HookRunner>,
    /// Opt-in provider-request smart compaction (canonical history always kept).
    pub(crate) smart_compaction_mode: SmartCompactionMode,
    /// Start instant per pending tool call_id, for duration tracking.
    pub(crate) tool_start_times: HashMap<String, Instant>,
    /// Prompt-cache keepalive profile (per-provider economics).
    pub(crate) keepalive_profile: KeepaliveProfile,
    /// Session-scoped repeated-call guard (persists across tool batches/turns).
    pub(crate) repeat_guard: RepeatCallGuard,
    /// Sender half of the single inbox (bounded 16). Cloned out via
    /// [`AgentLoop::inbox_sender`] for prompts/steering while a turn runs.
    inbox_tx: tokio::sync::mpsc::Sender<InboxItem>,
    /// Receiver half of the inbox; drained at step boundaries by the turn
    /// driver. Survives across turns (leftovers claimed at next turn start).
    pub(crate) inbox_rx: tokio::sync::mpsc::Receiver<InboxItem>,
    /// Monotonic per-agent turn counter (last emitted `turn_id`). Starts at 0
    /// so the first turn is 1; seeded via [`AgentLoop::set_turn_seq_start`] on
    /// session resume so ids stay session-unique across restarts.
    turn_seq: u64,
}

impl AgentLoop {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provider: Arc<dyn Provider>,
        tools: ToolRegistry,
        approval: ApprovalPolicy,
        model: String,
        event_tx: tokio::sync::mpsc::Sender<AgentEvent>,
        max_turns: u32,
        max_tool_calls_per_turn: u32,
        checkpoint_interval: u32,
        hooks: Option<HookRunner>,
    ) -> Self {
        let (inbox_tx, inbox_rx) = tokio::sync::mpsc::channel(16);
        Self {
            provider,
            tools,
            approval,
            messages: Vec::new(),
            model,
            cost_tracker: CostTracker::default(),
            event_tx,
            max_turns,
            max_tool_calls_per_turn,
            checkpoint_interval,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            hooks,
            smart_compaction_mode: SmartCompactionMode::Off,
            tool_start_times: HashMap::new(),
            keepalive_profile: KeepaliveProfile::disabled(),
            repeat_guard: RepeatCallGuard::new(),
            inbox_tx,
            inbox_rx,
            turn_seq: 0,
        }
    }

    /// Handle for enqueueing prompts/steering into the *running or next*
    /// turn's inbox. Bounded (16); `try_send` failure means "inbox full" and
    /// is surfaced by the caller.
    pub fn inbox_sender(&self) -> tokio::sync::mpsc::Sender<InboxItem> {
        self.inbox_tx.clone()
    }

    /// Seed the turn-id counter (session resume: max `TurnStarted.turn_id`
    /// found in the event log) so ids stay session-unique across restarts.
    pub fn set_turn_seq_start(&mut self, n: u64) {
        self.turn_seq = n;
    }

    pub fn set_smart_compaction_mode(&mut self, mode: SmartCompactionMode) {
        self.smart_compaction_mode = mode;
    }

    pub fn smart_compaction_mode(&self) -> SmartCompactionMode {
        self.smart_compaction_mode
    }

    /// Set the prompt-cache keepalive profile (called by supervisor after
    /// provider construction, using per-provider economics).
    pub fn set_keepalive_profile(&mut self, profile: KeepaliveProfile) {
        self.keepalive_profile = profile;
    }

    /// Add a system prompt once at startup.
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.messages.push(Message::system(prompt));
    }

    /// Replace the LLM provider (e.g. after user switches provider in-session).
    pub fn replace_provider(&mut self, provider: Arc<dyn Provider>) {
        self.provider = provider;
    }

    /// Run one turn: send messages to the provider, execute any tool calls,
    /// and repeat until the provider returns a final text response.
    pub async fn run_turn(
        &mut self,
        user_input: &str,
        workspace_root: &Path,
        attachments: &[ImageAttachment],
    ) -> Result<String, ProviderError> {
        let turn_start = Instant::now();
        self.cancel_flag.store(false, Ordering::SeqCst);
        // Rollback baseline: on failure the history is truncated back to this
        // length (NOT pop-counted) because claimed inbox messages can sit
        // mid-history; popping the tail would strip assistant/tool pairs and
        // orphan tool_calls. Residual orphans are repaired by
        // `sanitize_tool_call_pairs` on the next request.
        let baseline = self.messages.len();
        self.turn_seq += 1;
        let turn_id = self.turn_seq;
        self.emit(AgentEvent::TurnStarted { turn_id }).await;

        // Claim leftover inbox items from a previous turn (arrival order),
        // BEFORE the new user message.
        self.claim_inbox().await;

        let user_msg = if attachments.is_empty() {
            Message::user(user_input)
        } else {
            let mut parts: Vec<ContentPart> = Vec::new();
            let trimmed = user_input.trim();
            if !trimmed.is_empty() {
                parts.push(ContentPart::Text {
                    text: user_input.to_string(),
                });
            } else {
                parts.push(ContentPart::Text {
                    text: "(See attached image(s).)".into(),
                });
            }
            for a in attachments {
                parts.push(ContentPart::Image {
                    media_type: a.media_type.clone(),
                    path: a.path.clone(),
                });
            }
            Message::user_with_parts(parts)
        };
        let preview = user_msg.event_preview();
        self.emit(AgentEvent::MessageRecorded {
            message: user_msg.clone(),
        })
        .await;
        self.messages.push(user_msg);
        self.emit(AgentEvent::MessageReceived {
            role: "user".into(),
            content: preview,
            steering: false,
        })
        .await;

        let result =
            crate::agent_driver::TurnDriver::new(self, workspace_root, turn_id, attachments)
                .run(attachments)
                .await;

        // On failure, truncate back to the baseline so the message history
        // isn't left in a corrupted state (consecutive user messages with no
        // assistant reply confuses providers and causes repeated empty
        // responses).
        if result.is_err() {
            self.messages.truncate(baseline);
        }

        self.emit(AgentEvent::TurnCompleted {
            turn_id,
            duration_ms: turn_start.elapsed().as_millis() as u64,
        })
        .await;
        self.emit(AgentEvent::BusyStateChanged {
            state: BusyState::Idle,
        })
        .await;
        result
    }

    pub(crate) fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools.definitions()
    }

    pub(crate) async fn emit(&self, event: AgentEvent) {
        let _ = self.event_tx.send(event).await;
    }

    pub fn event_sender(&self) -> Option<tokio::sync::mpsc::Sender<AgentEvent>> {
        Some(self.event_tx.clone())
    }

    pub fn request_cancel(&self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
    }

    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        self.cancel_flag.clone()
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancel_flag.load(Ordering::SeqCst)
    }
}

/// Ensure every assistant message carrying `tool_calls` is followed by a
/// matching `tool` message for *each* `tool_call_id`.
///
/// A run interrupted mid-turn — after the assistant message was already pushed
/// but before its tools executed (turn/tool-call budget exceeded, pipeline
/// error, or cancellation) — can leave the history with an assistant message
/// whose `tool_calls` have no corresponding tool results. OpenAI-compatible
/// providers (DeepSeek especially) reject such sequences with:
/// "an assistant message with 'tool_calls' must be followed by tool messages".
///
/// This repairs the history in-place by injecting a synthetic error tool
/// message for every missing `tool_call_id`. The repair is persisted to
/// `self.messages`, so resumed sessions stay valid.
pub(crate) fn sanitize_tool_call_pairs(messages: &mut Vec<Message>) {
    const SYNTHETIC_RESULT: &str = "[tool execution interrupted — a budget or \
        pipeline limit was reached before this call ran; synthetic result \
        inserted to keep the message history valid for the provider]";

    let mut out: Vec<Message> = Vec::with_capacity(messages.len() + 4);
    let mut i = 0;
    while i < messages.len() {
        let expected: Option<Vec<String>> = if messages[i].role == Role::Assistant {
            messages[i]
                .tool_calls
                .as_ref()
                .filter(|calls| !calls.is_empty())
                .map(|calls| calls.iter().map(|c| c.id.clone()).collect())
        } else {
            None
        };

        out.push(messages[i].clone());

        if let Some(expected) = expected {
            let mut seen: HashSet<String> = HashSet::new();
            let mut j = i + 1;
            while j < messages.len() && messages[j].role == Role::Tool {
                if let Some(id) = messages[j].tool_call_id.as_ref()
                    && expected.iter().any(|e| e == id)
                {
                    seen.insert(id.clone());
                }
                out.push(messages[j].clone());
                j += 1;
            }
            for id in &expected {
                if !seen.contains(id) {
                    out.push(Message::tool(id.clone(), SYNTHETIC_RESULT));
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }

    *messages = out;
}

/// Truncate a string to `max_chars` characters, appending "…" if truncated.
pub(crate) fn truncate_str(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars.saturating_sub(1)).collect();
        format!("{truncated}…")
    }
}

/// Maximum bytes for a single tool result before it enters the model's context.
/// ~32KB ≈ 8K tokens — enough for a full file read or a busy search, while
/// preventing one accidental "read this 5 MB log" from blowing the context
/// window before compaction runs.
const MAX_TOOL_OUTPUT_BYTES: usize = 32 * 1024;

/// Head/tail budget for tool output truncation.
const TOOL_TRUNCATE_HEAD: usize = 16 * 1024;
const TOOL_TRUNCATE_TAIL: usize = 16 * 1024;

/// Truncate a tool result string if it exceeds [`MAX_TOOL_OUTPUT_BYTES`].
/// Uses head+tail strategy to keep the beginning and end visible.
fn truncate_tool_output(output: &str) -> String {
    let byte_len = output.len();
    if byte_len <= MAX_TOOL_OUTPUT_BYTES {
        return output.to_string();
    }

    // Find safe UTF-8 boundaries for head and tail.
    let head_end = match output.is_char_boundary(TOOL_TRUNCATE_HEAD) {
        true => TOOL_TRUNCATE_HEAD,
        false => {
            let mut pos = TOOL_TRUNCATE_HEAD;
            while pos > 0 && !output.is_char_boundary(pos) {
                pos -= 1;
            }
            pos
        }
    };

    let tail_start = match output.is_char_boundary(byte_len - TOOL_TRUNCATE_TAIL) {
        true => byte_len - TOOL_TRUNCATE_TAIL,
        false => {
            let mut pos = byte_len - TOOL_TRUNCATE_TAIL;
            while pos < byte_len && !output.is_char_boundary(pos) {
                pos += 1;
            }
            pos
        }
    };

    let truncated_bytes = byte_len - (TOOL_TRUNCATE_HEAD + TOOL_TRUNCATE_TAIL);
    format!(
        "{}\n\n[... truncated {} bytes ({} → {}KB limit) ...]\n\n{}",
        &output[..head_end],
        truncated_bytes,
        byte_len / 1024,
        MAX_TOOL_OUTPUT_BYTES / 1024,
        &output[tail_start..]
    )
}

pub(crate) fn cleanup_processed_attachments(
    messages: &mut [Message],
    workspace_root: &Path,
    attachments: &[ImageAttachment],
) {
    let removed_paths: HashSet<String> = attachments.iter().map(|a| a.path.clone()).collect();
    if removed_paths.is_empty() {
        return;
    }

    for message in messages {
        let _ = message.content.strip_image_paths(&removed_paths);
    }

    for attachment in attachments {
        let full_path = workspace_root.join(&attachment.path);
        if let Err(err) = std::fs::remove_file(&full_path)
            && err.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                "failed to remove processed image attachment {}: {}",
                full_path.display(),
                err
            );
        }
        if let Some(parent) = full_path.parent() {
            let _ = std::fs::remove_dir(parent);
        }
    }
}

pub(crate) fn format_tool_result(result: &nca_common::tool::ToolResult) -> String {
    let raw = if result.success {
        result.output.clone()
    } else {
        result
            .error
            .clone()
            .unwrap_or_else(|| "tool failed".to_string())
    };
    truncate_tool_output(&raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approval::ApprovalPolicy;
    use crate::provider::{Provider, ProviderError, StreamChunk};
    use crate::tools::ToolRegistry;
    use nca_common::config::PermissionConfig;
    use nca_common::message::MessageToolCall;
    use nca_common::tool::ToolDefinition;
    use serde_json::json;
    use std::sync::atomic::AtomicU32;

    /// Scripted provider: each `chat()` call replays the next round of chunks,
    /// then closes the channel (the agent loop treats channel close as end of
    /// stream). Counts calls so tests can assert retry behavior.
    struct ScriptedProvider {
        rounds: Vec<Vec<StreamChunk>>,
        calls: Arc<AtomicU32>,
    }

    impl ScriptedProvider {
        fn new(rounds: Vec<Vec<StreamChunk>>) -> (Self, Arc<AtomicU32>) {
            let calls = Arc::new(AtomicU32::new(0));
            (
                Self {
                    rounds,
                    calls: Arc::clone(&calls),
                },
                calls,
            )
        }
    }

    #[async_trait::async_trait]
    impl Provider for ScriptedProvider {
        async fn chat(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
            _model: &str,
            _workspace_root: &Path,
        ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError> {
            let index = self.calls.fetch_add(1, Ordering::SeqCst) as usize;
            let round = self.rounds.get(index).cloned().unwrap_or_default();
            let (tx, rx) = tokio::sync::mpsc::channel(64);
            tokio::spawn(async move {
                for chunk in round {
                    let _ = tx.send(chunk).await;
                }
            });
            Ok(rx)
        }
    }

    fn test_agent(provider: Arc<dyn Provider>) -> AgentLoop {
        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(256);
        AgentLoop::new(
            provider,
            ToolRegistry::new(),
            ApprovalPolicy::new(PermissionConfig::default()),
            "glm-5.3".into(),
            event_tx,
            10,
            16,
            0,
            None,
        )
    }

    #[tokio::test]
    async fn reasoning_only_length_truncation_fails_fast_without_retry() {
        // GLM-5.3 with max_tokens exhausted mid-thinking: reasoning deltas +
        // usage + finish_reason="length", no content, no tool calls. This is a
        // deterministic truncation — retrying with identical parameters would
        // re-bill the full prompt for the same outcome, so the loop must fail
        // fast (exactly one provider call) with a message pointing at the cap.
        let (provider, calls) = ScriptedProvider::new(vec![vec![
            StreamChunk::ReasoningDelta("thinking very hard".into()),
            StreamChunk::Usage {
                input_tokens: 9_000,
                output_tokens: 8_192,
                cache_creation_tokens: 0,
                cache_read_tokens: 0,
            },
            StreamChunk::Finish {
                reason: "length".into(),
            },
        ]]);
        let mut agent = test_agent(Arc::new(provider));

        let err = agent
            .run_turn("do the thing", Path::new("."), &[])
            .await
            .expect_err("must fail fast");
        let message = err.to_string();
        assert!(
            message.contains("max_tokens"),
            "error must point at the token cap: {message}"
        );
        assert!(
            message.contains("length"),
            "error must name the finish reason: {message}"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "length-truncation is deterministic; blind retries must be skipped"
        );
    }

    #[tokio::test]
    async fn reasoning_only_clean_stop_still_retries_then_reports_diagnostics() {
        // Empty content with finish_reason="stop" is NOT deterministic — the
        // existing bounded retry behavior applies (initial attempt + 2 retries).
        // The final error must now include reasoning diagnostics.
        let round = vec![
            StreamChunk::ReasoningDelta("hmm".into()),
            StreamChunk::Usage {
                input_tokens: 100,
                output_tokens: 5,
                cache_creation_tokens: 0,
                cache_read_tokens: 0,
            },
            StreamChunk::Finish {
                reason: "stop".into(),
            },
        ];
        let (provider, calls) = ScriptedProvider::new(vec![round.clone(), round.clone(), round]);
        let mut agent = test_agent(Arc::new(provider));

        let err = agent
            .run_turn("do the thing", Path::new("."), &[])
            .await
            .expect_err("empty response after retries");
        let message = err.to_string();
        assert!(message.contains("after retries"), "got: {message}");
        assert!(
            message.contains("chars of reasoning"),
            "error must surface reasoning diagnostics: {message}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn finish_chunk_does_not_disturb_normal_text_turn() {
        let (provider, _calls) = ScriptedProvider::new(vec![vec![
            StreamChunk::TextDelta("all ".into()),
            StreamChunk::TextDelta("done".into()),
            StreamChunk::Finish {
                reason: "stop".into(),
            },
        ]]);
        let mut agent = test_agent(Arc::new(provider));

        let text = agent
            .run_turn("hi", Path::new("."), &[])
            .await
            .expect("normal turn");
        assert_eq!(text, "all done");
        assert_eq!(agent.messages.len(), 2, "user + assistant messages");
    }

    fn tc(id: &str) -> MessageToolCall {
        MessageToolCall {
            id: id.to_string(),
            name: "read".to_string(),
            arguments: json!({}),
        }
    }

    #[test]
    fn sanitize_is_noop_when_tool_pairs_complete() {
        let mut msgs = vec![
            Message::user("hi"),
            Message::assistant_with_tool_calls("checking", vec![tc("a"), tc("b")]),
            Message::tool("a", "ra"),
            Message::tool("b", "rb"),
        ];
        let before = msgs.clone();
        sanitize_tool_call_pairs(&mut msgs);
        assert_eq!(msgs, before, "complete pairs must be left untouched");
    }

    #[test]
    fn sanitize_fills_all_missing_tool_results() {
        // assistant emitted 2 tool_calls but a budget error fired before any ran.
        let mut msgs = vec![
            Message::assistant_with_tool_calls("checking", vec![tc("a"), tc("b")]),
            Message::user("continue"),
        ];
        sanitize_tool_call_pairs(&mut msgs);
        // [assistant, synthetic(a), synthetic(b), user]
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[1].role, Role::Tool);
        assert_eq!(msgs[1].tool_call_id.as_deref(), Some("a"));
        assert_eq!(msgs[2].role, Role::Tool);
        assert_eq!(msgs[2].tool_call_id.as_deref(), Some("b"));
        assert_eq!(msgs[3].role, Role::User);
    }

    #[test]
    fn sanitize_fills_only_missing_tool_results() {
        // 2 calls, only "a" got a result before interruption.
        let mut msgs = vec![
            Message::assistant_with_tool_calls("checking", vec![tc("a"), tc("b")]),
            Message::tool("a", "ra"),
            Message::assistant("done?"),
        ];
        sanitize_tool_call_pairs(&mut msgs);
        // [assistant+tc, tool(a), synthetic(b), assistant]
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[1].tool_call_id.as_deref(), Some("a"));
        assert_eq!(msgs[2].role, Role::Tool);
        assert_eq!(msgs[2].tool_call_id.as_deref(), Some("b"));
    }

    #[test]
    fn sanitize_repairs_trailing_orphaned_assistant() {
        // assistant with tool_calls at the very end, no results at all.
        let mut msgs = vec![
            Message::user("do it"),
            Message::assistant_with_tool_calls("running", vec![tc("x")]),
        ];
        sanitize_tool_call_pairs(&mut msgs);
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[2].role, Role::Tool);
        assert_eq!(msgs[2].tool_call_id.as_deref(), Some("x"));
    }
}
