use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::tool::ToolResult;

/// Real-time busy state indicator for CLI rendering.
/// Reflects the current processing state of the agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BusyState {
    /// Agent is idle and ready for input.
    Idle,
    /// Agent is thinking/planning (heavy computation, model reasoning).
    Thinking,
    /// Agent is streaming tokens from LLM.
    Streaming,
    /// Agent is executing a tool.
    ToolRunning,
    /// Agent is waiting for user approval on a tool call.
    ApprovalPending,
    /// Error or blocked state.
    Error,
}

impl BusyState {
    /// Human-readable label for this state.
    pub fn label(&self) -> &'static str {
        match self {
            BusyState::Idle => "idle",
            BusyState::Thinking => "thinking",
            BusyState::Streaming => "streaming",
            BusyState::ToolRunning => "tool",
            BusyState::ApprovalPending => "approval",
            BusyState::Error => "error",
        }
    }
}

/// Envelope for events written to disk, with stable id and timestamp for ordering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventEnvelope {
    #[serde(default)]
    pub id: u64,
    #[serde(default)]
    pub ts: Option<DateTime<Utc>>,
    pub event: AgentEvent,
}

impl EventEnvelope {
    pub fn new(id: u64, event: AgentEvent) -> Self {
        Self {
            id,
            ts: Some(Utc::now()),
            event,
        }
    }
}

/// Events emitted by the agent runtime, broadcast over IPC to the CLI.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AgentEvent {
    SessionStarted {
        session_id: String,
        workspace: PathBuf,
        model: String,
    },
    MessageReceived {
        role: String,
        content: String,
        /// `true` when this message is mid-turn steering guidance claimed
        /// from the inbox (as opposed to the turn's initial user prompt).
        #[serde(default)]
        steering: bool,
    },
    /// A model-visible message was pushed to `agent.messages`. The replay
    /// projection folds these, in order, to reconstruct conversation state.
    /// Emitted at every push site inside `run_turn`; never for system prompts.
    MessageRecorded {
        message: crate::message::Message,
    },
    /// The conversation history was wholesale replaced (context compaction:
    /// AI summary or sliding window). The replay projection sets its state
    /// to exactly this payload. Emitted between turn brackets; payload is
    /// system-stripped at the emit site (resume always prepends a fresh
    /// system prompt, so stale prompts must not ride along).
    HistoryReplaced {
        messages: Vec<crate::message::Message>,
    },
    /// A turn started: emitted at the beginning of `run_turn`. `turn_id` is
    /// monotonic per agent (1-based) and stays session-unique across resume.
    TurnStarted {
        #[serde(default)]
        turn_id: u64,
    },
    /// One agent step started (a single provider call plus its tool
    /// pipeline). `step_index` is 1-based within the turn.
    StepStarted {
        #[serde(default)]
        turn_id: u64,
        #[serde(default)]
        step_index: u64,
    },
    /// One agent step finished successfully. `had_tool_calls` is true when
    /// the step executed at least one tool call (i.e. the turn continues).
    StepCompleted {
        #[serde(default)]
        turn_id: u64,
        #[serde(default)]
        step_index: u64,
        #[serde(default)]
        duration_ms: u64,
        #[serde(default)]
        had_tool_calls: bool,
    },
    /// One agent step errored (stream error, budget exceeded, pipeline
    /// failure, empty response after retries). Closes the `StepStarted`
    /// bracket so replay layers never see a dangling step.
    StepFailed {
        #[serde(default)]
        turn_id: u64,
        #[serde(default)]
        step_index: u64,
        #[serde(default)]
        duration_ms: u64,
        #[serde(default)]
        error: String,
    },
    TokensStreamed {
        delta: String,
    },
    /// Streaming reasoning/thinking content from the provider (e.g., DeepSeek R1, MiniMax M2.5).
    ReasoningStreamed {
        delta: String,
    },
    ToolCallStarted {
        call_id: String,
        tool: String,
        input: serde_json::Value,
    },
    ToolCallCompleted {
        call_id: String,
        output: ToolResult,
        /// Wall-clock execution time of this tool call in milliseconds.
        #[serde(default)]
        duration_ms: u64,
    },
    /// Incremental streamed output from a tool (e.g. live shell output).
    ToolOutputChunk {
        call_id: String,
        delta: String,
    },
    ApprovalRequested {
        call_id: String,
        tool: String,
        description: String,
    },
    ApprovalResolved {
        call_id: String,
        approved: bool,
        /// Pattern being persisted when user chose "always allow".
        #[serde(default, skip_serializing_if = "Option::is_none")]
        allow_pattern: Option<String>,
    },
    CostUpdated {
        input_tokens: u64,
        output_tokens: u64,
        #[serde(default)]
        cache_read_tokens: u64,
        estimated_cost_usd: f64,
    },
    Checkpoint {
        phase: String,
        detail: String,
        turn: u32,
    },
    SessionEnded {
        reason: EndReason,
    },
    Error {
        message: String,
    },
    Response {
        response: AgentResponse,
    },
    ChildSessionSpawned {
        parent_session_id: String,
        child_session_id: String,
        task: String,
        workspace: PathBuf,
        branch: Option<String>,
    },
    ChildSessionCompleted {
        parent_session_id: String,
        child_session_id: String,
        status: String,
    },
    /// Live activity from a child session (tools, checkpoints, nested spawns), for parent UI.
    ChildSessionActivity {
        child_session_id: String,
        /// Short label, e.g. tool name or checkpoint phase.
        phase: String,
        /// One-line detail for the sidebar/transcript.
        detail: String,
    },
    /// User must pick an option, use the suggested answer, or enter custom text.
    /// Emitted when the `ask_question` tool runs; answer via `AgentCommand::AnswerQuestion` or local CLI.
    QuestionRequested {
        question: InteractiveQuestionPayload,
    },
    /// Logged after the user (or orchestrator) answers a question.
    QuestionResolved {
        question_id: String,
        selection: QuestionSelection,
    },
    /// Full replacement of the session todo list (last write wins on replay).
    TodosUpdated {
        todos: Vec<crate::todo::AgentTodo>,
    },
    /// Live context usage statistics for the UI (token count + percentage).
    ContextStatsUpdated {
        estimated_tokens: usize,
        context_window: usize,
        usage_percent: u8,
    },
    /// Warning that context is approaching limit.
    ContextWarning {
        message: String,
    },
    /// Context compaction/summarization event.
    ContextCompaction {
        phase: String,
        message: String,
        /// Estimated tokens before compaction (optional for older logs).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tokens_before: Option<usize>,
        /// Estimated tokens after compaction (optional for older logs).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tokens_after: Option<usize>,
        /// Groups retained in the provider request view.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retained_groups: Option<usize>,
        /// Groups dropped or truncated from the provider request view.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        dropped_groups: Option<usize>,
    },
    /// Replacement-type compaction started (canonical summarize or overflow
    /// recovery). Informational; replay-ignored. Pairs with
    /// [`AgentEvent::ContextCompactionEnd`]; the middle of the bracket is the
    /// replacement itself (`HistoryReplaced` for canonical paths, nothing for
    /// view-prune). Routine per-step smart compaction keeps emitting the
    /// single `ContextCompaction` event above.
    ContextCompactionStart {
        /// Estimated token count of the context before compaction.
        #[serde(default)]
        tokens_before: usize,
        /// Why compaction started: "auto_summarize" | "overflow_prune" |
        /// "overflow_summarize".
        #[serde(default)]
        reason: String,
    },
    /// Replacement-type compaction finished. `kv_prefix_broken` marks
    /// recoveries that changed the request view or replaced history
    /// (cache-cold retry).
    ContextCompactionEnd {
        /// Estimated token count of the context after compaction.
        #[serde(default)]
        tokens_after: usize,
        /// Whether the warmed KV cache prefix was invalidated by the
        /// replacement (always true for current paths).
        #[serde(default)]
        kv_prefix_broken: bool,
    },
    /// Busy state transition (for animated indicator rendering).
    BusyStateChanged {
        state: BusyState,
    },
    /// Prompt-cache keepalive started during a tool-execution pause.
    CacheKeepaliveStarted {
        #[serde(default)]
        interval_secs: u64,
        #[serde(default)]
        benefit: String,
    },
    /// A keepalive ping fired and refreshed the cache.
    CacheKeepalivePing {
        #[serde(default)]
        pings: u32,
        #[serde(default)]
        cache_read_tokens: u64,
        #[serde(default)]
        cache_creation_tokens: u64,
        #[serde(default)]
        elapsed_secs: u64,
    },
    /// Keepalive task stopped (pause ended, break-even exceeded, or disabled).
    CacheKeepaliveStopped {
        #[serde(default)]
        reason: String,
        #[serde(default)]
        pings: u32,
        #[serde(default)]
        total_cache_read_tokens: u64,
        #[serde(default)]
        total_cache_creation_tokens: u64,
    },
    /// A full turn completed — emitted once at the end of `run_turn` with the
    /// total wall-clock duration (user input → final reply) in milliseconds.
    TurnCompleted {
        /// Monotonic turn identifier matching the turn's `TurnStarted` event.
        #[serde(default)]
        turn_id: u64,
        #[serde(default)]
        duration_ms: u64,
    },
}

/// One selectable row shown for an interactive question.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct QuestionOption {
    pub id: String,
    pub label: String,
}

/// Full question payload broadcast on the event bus and shown in the CLI.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InteractiveQuestionPayload {
    pub question_id: String,
    pub call_id: String,
    pub prompt: String,
    pub options: Vec<QuestionOption>,
    #[serde(default = "default_allow_custom")]
    pub allow_custom: bool,
    /// Model-provided default; user can accept via suggested / `/auto-answer`.
    pub suggested_answer: String,
}

fn default_allow_custom() -> bool {
    true
}

/// How the user answered an interactive question.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum QuestionSelection {
    Option { option_id: String },
    Custom { text: String },
    Suggested,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EndReason {
    UserExit,
    Completed,
    Error,
    Cancelled,
}

/// Commands sent over IPC to the agent runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AgentCommand {
    SendMessage {
        content: String,
    },
    ApproveToolCall {
        call_id: String,
    },
    DenyToolCall {
        call_id: String,
    },
    /// Submit an answer for the pending `ask_question` with matching `question_id`.
    AnswerQuestion {
        question_id: String,
        selection: QuestionSelection,
    },
    Cancel,
    Shutdown,
}

/// Responses to query-style IPC messages, sent back as `AgentEvent::Response`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AgentResponse {
    SessionState {
        session: Box<crate::session::SessionState>,
    },
    SessionList {
        sessions: Vec<crate::session::SessionMeta>,
    },
    Error {
        message: String,
    },
    Ok,
}

#[cfg(test)]
mod interactive_question_serde_tests {
    use super::*;

    #[test]
    fn question_requested_roundtrip() {
        let q = InteractiveQuestionPayload {
            question_id: "q-1".into(),
            call_id: "call_1".into(),
            prompt: "Pick one".into(),
            options: vec![
                QuestionOption {
                    id: "a".into(),
                    label: "Alpha".into(),
                },
                QuestionOption {
                    id: "b".into(),
                    label: "Beta".into(),
                },
            ],
            allow_custom: true,
            suggested_answer: "Alpha".into(),
        };
        let ev = AgentEvent::QuestionRequested {
            question: q.clone(),
        };
        let json = serde_json::to_string(&ev).expect("serialize");
        let back: AgentEvent = serde_json::from_str(&json).expect("deserialize");
        match back {
            AgentEvent::QuestionRequested { question } => assert_eq!(question, q),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn message_recorded_roundtrip() {
        let message = crate::message::Message::assistant_with_tool_calls(
            "thinking aloud",
            vec![crate::message::MessageToolCall {
                id: "call_1".into(),
                name: "read_file".into(),
                arguments: serde_json::json!({"path": "README.md"}),
            }],
        )
        .with_reasoning("because".into());
        let ev = AgentEvent::MessageRecorded {
            message: message.clone(),
        };
        let json = serde_json::to_string(&ev).expect("serialize");
        let back: AgentEvent = serde_json::from_str(&json).expect("deserialize");
        match back {
            AgentEvent::MessageRecorded { message: m } => assert_eq!(m, message),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn history_replaced_roundtrip() {
        let messages = vec![
            crate::message::Message::user("summarized"),
            crate::message::Message::assistant("the summary"),
        ];
        let ev = AgentEvent::HistoryReplaced {
            messages: messages.clone(),
        };
        let json = serde_json::to_string(&ev).expect("serialize");
        let back: AgentEvent = serde_json::from_str(&json).expect("deserialize");
        match back {
            AgentEvent::HistoryReplaced { messages: m } => assert_eq!(m, messages),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn reasoning_streamed_roundtrip() {
        let ev = AgentEvent::ReasoningStreamed {
            delta: "thinking...".into(),
        };
        let json = serde_json::to_string(&ev).expect("serialize");
        let back: AgentEvent = serde_json::from_str(&json).expect("deserialize");
        match back {
            AgentEvent::ReasoningStreamed { delta } => assert_eq!(delta, "thinking..."),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn answer_question_command_roundtrip() {
        let cmd = AgentCommand::AnswerQuestion {
            question_id: "q-1".into(),
            selection: QuestionSelection::Suggested,
        };
        let json = serde_json::to_string(&cmd).expect("serialize");
        let back: AgentCommand = serde_json::from_str(&json).expect("deserialize");
        match back {
            AgentCommand::AnswerQuestion {
                question_id,
                selection,
            } => {
                assert_eq!(question_id, "q-1");
                assert!(matches!(selection, QuestionSelection::Suggested));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn todos_updated_roundtrip() {
        let ev = AgentEvent::TodosUpdated {
            todos: vec![crate::todo::AgentTodo {
                id: "1".into(),
                content: "Do thing".into(),
                status: crate::todo::TodoStatus::Pending,
                source: Some(crate::todo::TodoSource::Agent),
            }],
        };
        let json = serde_json::to_string(&ev).expect("serialize");
        let back: AgentEvent = serde_json::from_str(&json).expect("deserialize");
        match back {
            AgentEvent::TodosUpdated { todos } => {
                assert_eq!(todos.len(), 1);
                assert_eq!(todos[0].id, "1");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn context_compaction_start_end_roundtrip() {
        let start = AgentEvent::ContextCompactionStart {
            tokens_before: 12_345,
            reason: "overflow_prune".into(),
        };
        let json = serde_json::to_string(&start).expect("serialize");
        let back: AgentEvent = serde_json::from_str(&json).expect("deserialize");
        match back {
            AgentEvent::ContextCompactionStart {
                tokens_before,
                reason,
            } => {
                assert_eq!(tokens_before, 12_345);
                assert_eq!(reason, "overflow_prune");
            }
            _ => panic!("wrong variant"),
        }

        let end = AgentEvent::ContextCompactionEnd {
            tokens_after: 4_000,
            kv_prefix_broken: true,
        };
        let json = serde_json::to_string(&end).expect("serialize");
        let back: AgentEvent = serde_json::from_str(&json).expect("deserialize");
        match back {
            AgentEvent::ContextCompactionEnd {
                tokens_after,
                kv_prefix_broken,
            } => {
                assert_eq!(tokens_after, 4_000);
                assert!(kv_prefix_broken);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn compaction_end_missing_kv_prefix_broken_still_deserializes() {
        // C10: a partially-written log tail may drop trailing fields;
        // `#[serde(default)]` must tolerate it.
        let raw = r#"{"type":"ContextCompactionEnd","tokens_after":900}"#;
        let back: AgentEvent = serde_json::from_str(raw).expect("deserialize");
        match back {
            AgentEvent::ContextCompactionEnd {
                tokens_after,
                kv_prefix_broken,
            } => {
                assert_eq!(tokens_after, 900);
                assert!(!kv_prefix_broken);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn old_context_compaction_still_deserializes() {
        let raw = r#"{"type":"ContextCompaction","phase":"completed","message":"done"}"#;
        let back: AgentEvent = serde_json::from_str(raw).expect("deserialize");
        match back {
            AgentEvent::ContextCompaction {
                phase,
                tokens_before,
                retained_groups,
                ..
            } => {
                assert_eq!(phase, "completed");
                assert!(tokens_before.is_none());
                assert!(retained_groups.is_none());
            }
            _ => panic!("wrong variant"),
        }
    }
}
