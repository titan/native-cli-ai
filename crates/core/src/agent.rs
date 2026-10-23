use nca_common::event::{AgentEvent, BusyState};
use nca_common::message::{ContentPart, ImageAttachment, Message, Role};
use nca_common::tool::{ToolCall, ToolDefinition};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
    /// Cap on tool calls executed from a single assistant message
    /// (`[session] max_tool_calls_per_message`). Excess calls are dropped
    /// with synthetic error results — see [`plan_tool_batch`]. Default
    /// mirrors the config default; the supervisor wires the config value in.
    pub(crate) max_tool_calls_per_message: usize,
    pub(crate) checkpoint_interval: u32,
    pub(crate) cancel_flag: Arc<AtomicBool>,
    pub(crate) hooks: Option<HookRunner>,
    /// Start instant per pending tool call_id, for duration tracking.
    pub(crate) tool_start_times: HashMap<String, Instant>,
    /// Prompt-cache keepalive profile (per-provider economics).
    pub(crate) keepalive_profile: KeepaliveProfile,
    /// Session-scoped repeated-call guard (persists across tool batches/turns).
    pub(crate) repeat_guard: RepeatCallGuard,
    /// Step-request middleware chain (P4). Empty by default — a bare
    /// `AgentLoop` is observably identical to pre-P4 behavior.
    pub(crate) middleware: crate::middleware::MiddlewareChain,
    /// Droppable events elided so far because the bounded event channel was
    /// full (see [`emit_classified`]). Flushed as one
    /// [`AgentEvent::EventsElided`] summary before the next must-deliver
    /// event; never reset except by that flush.
    pub(crate) elided_events: AtomicU64,
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

/// Build the user message that opens a turn: prompt text plus any image
/// attachments (an empty prompt with images gets a placeholder text part).
/// Shared by [`AgentLoop::run_turn`] and the runtime supervisor's
/// spawn-history projection so both see the identical turn-opening message.
pub fn turn_user_message(user_input: &str, attachments: &[ImageAttachment]) -> Message {
    if attachments.is_empty() {
        Message::user(user_input)
    } else {
        let mut parts: Vec<ContentPart> = Vec::new();
        if user_input.trim().is_empty() {
            parts.push(ContentPart::Text {
                text: "(See attached image(s).)".into(),
            });
        } else {
            parts.push(ContentPart::Text {
                text: user_input.to_string(),
            });
        }
        for a in attachments {
            parts.push(ContentPart::Image {
                media_type: a.media_type.clone(),
                path: a.path.clone(),
            });
        }
        Message::user_with_parts(parts)
    }
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
            max_tool_calls_per_message: nca_common::config::default_max_tool_calls_per_message(),
            checkpoint_interval,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            hooks,
            tool_start_times: HashMap::new(),
            keepalive_profile: KeepaliveProfile::disabled(),
            repeat_guard: RepeatCallGuard::new(),
            middleware: crate::middleware::MiddlewareChain::new(),
            elided_events: AtomicU64::new(0),
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

    /// Append a step-request middleware. First added = outermost (wraps
    /// everything added later and the provider call itself).
    pub fn with_middleware(
        mut self,
        middleware: Arc<dyn crate::middleware::AgentMiddleware>,
    ) -> Self {
        self.middleware.push(middleware);
        self
    }

    /// Append a step-request middleware in place (non-consuming counterpart
    /// of [`AgentLoop::with_middleware`], for construction sites that build
    /// via `new` and then configure). First added = outermost.
    pub fn push_middleware(&mut self, middleware: Arc<dyn crate::middleware::AgentMiddleware>) {
        self.middleware.push(middleware);
    }

    /// Drain a composed [`crate::middleware::MiddlewareChain`] into this
    /// agent's chain, outermost first (e.g.
    /// `crate::middleware::default_chain`). Knob-bearing composition entry
    /// point for the supervisor wiring.
    pub fn extend_middleware(&mut self, chain: crate::middleware::MiddlewareChain) {
        for middleware in chain.into_middlewares() {
            self.middleware.push(middleware);
        }
    }

    /// Seed the turn-id counter (session resume: max `TurnStarted.turn_id`
    /// found in the event log) so ids stay session-unique across restarts.
    pub fn set_turn_seq_start(&mut self, n: u64) {
        self.turn_seq = n;
    }

    /// Set the prompt-cache keepalive profile (called by supervisor after
    /// provider construction, using per-provider economics).
    pub fn set_keepalive_profile(&mut self, profile: KeepaliveProfile) {
        self.keepalive_profile = profile;
    }

    /// Set the per-message tool-call batch cap (`[session]
    /// `max_tool_calls_per_message`). Wired by the supervisor from config;
    /// tests force small values to exercise the guardrail.
    pub fn set_max_tool_calls_per_message(&mut self, cap: usize) {
        self.max_tool_calls_per_message = cap;
    }

    /// Set the session's system prompt, REPLACING any system message
    /// already in the history (single-system-message model).
    ///
    /// This used to be a plain push, so every in-session rebuild (Tab
    /// profile switch, `/mount`) stacked the new persona's system message
    /// on top of the old one — stale personas kept steering (and billing)
    /// every later turn. Replace semantics mirror the resume path
    /// (`select_resume_messages` in `nca-runtime` drops every persisted
    /// system message and prepends exactly one fresh prompt).
    /// Mid-history system notes recorded via [`AgentLoop::record_system_note`]
    /// are likewise superseded by the next rebuild: a rebuild IS a prompt
    /// reset, and the note's content was already observed by the turns that
    /// ran while it was live.
    ///
    /// The first call on a fresh history (no system message present) is an
    /// ordinary prepend and leaves every other message untouched.
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.messages.retain(|m| m.role != Role::System);
        self.messages.insert(0, Message::system(prompt));
    }

    /// Replace the LLM provider (e.g. after user switches provider in-session).
    pub fn replace_provider(&mut self, provider: Arc<dyn Provider>) {
        self.provider = provider;
    }

    /// Reset the session-scoped repeat-call guard. Called by the supervisor's
    /// `reset_for_new_session` (`/new`): a fresh session starts with fresh
    /// identical-call escalation state, since this loop instance outlives the
    /// session it was built for.
    pub fn reset_repeat_guard(&mut self) {
        self.repeat_guard.reset_session();
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

        // New turn = fresh per-turn guard state. The wait-tool repeat counter
        // (`RepeatCallGuard::reset_turn`) is turn-scoped by contract: calling
        // `wait_for_user` once per turn is legitimate, so each new user
        // message restores first-call behavior. The generic identical-call
        // counts are deliberately NOT reset here (session-scoped).
        self.repeat_guard.reset_turn();

        // Claim leftover inbox items from a previous turn (arrival order),
        // BEFORE the new user message.
        self.claim_inbox().await;

        let user_msg = turn_user_message(user_input, attachments);
        let preview = user_msg.event_preview();
        self.record(&user_msg).await;
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
        emit_classified(&self.event_tx, &self.elided_events, event).await;
    }

    /// Record a model-visible message: emit [`AgentEvent::MessageRecorded`]
    /// for the replay projection, then the caller pushes it. Every push site
    /// inside `run_turn` goes through this; system prompts never do.
    pub(crate) async fn record(&self, message: &Message) {
        self.emit(AgentEvent::MessageRecorded {
            message: message.clone(),
        })
        .await;
    }

    /// Append a system-role note to the live history AND the replay
    /// projection (G6 plugin command echo): the model observes the state
    /// change on its next turn and the transcript shows it inline. No turn
    /// is triggered — this is a pure history append between turns.
    pub async fn record_system_note(&mut self, text: &str) {
        self.record(&Message::system(text)).await;
        self.messages.push(Message::system(text));
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

/// Classified event send, the single backpressure policy for agent events
/// (incident 2026-10-02, Chunk 3b).
///
/// - **Droppable** notices ([`AgentEvent::is_droppable_under_backpressure`]:
///   `ToolCallStarted`, stream deltas, `ToolOutputChunk`) use `try_send`;
///   on a full channel the event is ELIDED (dropped) and counted in
///   `elided_events`. A degenerate batch can then no longer park the turn's
///   stream loop on per-call notices — the incident's "7,642 ToolCallStarted
///   then nothing" shape.
/// - **Must-deliver** events block on [`send_with_stall_alarm`] (never
///   dropping, never timing out, alarming while stalled): losing one
///   would corrupt the replay projection (`MessageRecorded`), break bracket
///   parity (`Step*`/`TurnCompleted`/`ToolCallCompleted`), orphan an
///   interactive oneshot (`ApprovalRequested`/`QuestionRequested`), or hide
///   a failure (`Error`). Before sending, any pending elision count is
///   flushed as one [`AgentEvent::EventsElided`] summary so consumers see
///   the gap adjacent to the next guaranteed event.
///
/// Channel-closed behaves like the previous fire-and-forget (`let _ =`).
pub(crate) async fn emit_classified(
    event_tx: &tokio::sync::mpsc::Sender<AgentEvent>,
    elided_events: &AtomicU64,
    event: AgentEvent,
) {
    if event.is_droppable_under_backpressure() {
        match event_tx.try_send(event) {
            Ok(()) => {}
            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                elided_events.fetch_add(1, Ordering::Relaxed);
            }
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {}
        }
        return;
    }
    let pending = elided_events.swap(0, Ordering::Relaxed);
    if pending > 0 {
        send_with_stall_alarm(event_tx, AgentEvent::EventsElided { count: pending }).await;
    }
    send_with_stall_alarm(event_tx, event).await;
}

/// How long a guaranteed event send waits between stall alarms while the
/// consumer refuses to drain the bounded channel.
const EVENT_STALL_ALARM_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);

/// Blocking send that never drops and never times out — but screams at
/// [`EVENT_STALL_ALARM_INTERVAL`] while the consumer refuses to drain.
/// Used by [`emit_classified`] for must-deliver events and by the runtime's
/// event fanout for must-deliver parent forwards.
///
/// Cancel-safety (why `reserve()`, not `send()`): `Sender::send` moves the
/// value into its future, so wrapping it in `timeout` and dropping the
/// future on expiry would LOSE the event — there is no way to get the moved
/// value back. `reserve()` is documented cancel-safe by tokio (a dropped
/// permit acquisition reserves nothing), and the resulting `Permit` sends
/// infallibly with capacity already held. This is the only correct shape
/// for a timeout-wrapped guaranteed send.
///
/// Channel-closed behaves like the fire-and-forget sends elsewhere
/// (`let _ = send()`): return silently.
pub async fn send_with_stall_alarm(
    event_tx: &tokio::sync::mpsc::Sender<AgentEvent>,
    event: AgentEvent,
) {
    let mut stalled_for = std::time::Duration::ZERO;
    loop {
        match tokio::time::timeout(EVENT_STALL_ALARM_INTERVAL, event_tx.reserve()).await {
            Ok(Ok(permit)) => {
                // Capacity is reserved: infallible and instantaneous.
                permit.send(event);
                return;
            }
            // Channel closed — fire-and-forget parity.
            Ok(Err(_closed)) => return,
            Err(_elapsed) => {
                stalled_for += EVENT_STALL_ALARM_INTERVAL;
                tracing::error!(
                    stalled_secs = stalled_for.as_secs(),
                    kind = event_kind_label(&event),
                    "event_channel_stalled: consumer not draining; still waiting (never dropping)"
                );
                // Re-arm: `reserve()` cancelled above reserved nothing, so
                // the event is still solely ours and undelivered.
            }
        }
    }
}

/// Stable discriminant label for diagnostics (the stall alarm): the variant
/// name without payload, taken from the leading identifier of the Debug
/// repr (e.g. `MessageRecorded { .. }` → `"MessageRecorded"`). Only runs
/// on the alarm cadence — once per 30s per stuck send — so the formatting
/// cost is irrelevant.
fn event_kind_label(event: &AgentEvent) -> String {
    let dbg = format!("{event:?}");
    let end = dbg
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(dbg.len());
    dbg[..end].to_string()
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
pub fn sanitize_tool_call_pairs(messages: &mut Vec<Message>) {
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

/// Marker prefix stamped on loop-breaker early-final texts (consecutive
/// tool failures, guard-refusal escalation) so downstream consumers can
/// classify the turn as a pathological stop without string-sniffing the
/// human-readable message. Stamped at position 0 of the final text; the
/// runtime's child-session folding checks `starts_with` on it to map such
/// turns to `error`/`Failed` instead of `completed`.
pub const LOOP_BREAKER_MARKER: &str = "[nca:loop-breaker]";

/// Planned execution shape for one assistant message's tool-call batch:
/// which calls execute, which are intra-batch duplicates of an executor,
/// and which fall beyond the per-message cap.
///
/// Produced by [`plan_tool_batch`] at batch finalization (all tool-use
/// blocks received). Plan-before-record: the caller runs this BEFORE the
/// assistant message is recorded and records only the kept window
/// (executors + duplicates in emission order); `dropped` ids never enter
/// history, so tool_use/tool-result pairing stays balanced by construction.
/// Duplicates receive the executor's result re-bound to their own call_id.
#[derive(Debug, Default)]
pub(crate) struct ToolBatchPlan {
    /// Distinct calls that will execute, in first-occurrence order.
    pub(crate) executors: Vec<ToolCall>,
    /// `(duplicate call_id, index into `executors`)` for every kept call
    /// whose (name, canonical input) matches an earlier kept call.
    pub(crate) duplicates: Vec<(String, usize)>,
    /// Emitted calls beyond `max_tool_calls_per_message`, in emission
    /// order. Never executed and never recorded — trimmed from the
    /// assistant message so the overflow cannot become a history bomb.
    pub(crate) dropped: Vec<ToolCall>,
    /// `(tool name, kept occurrences)` per deduped group with more than one
    /// member, sorted by name for deterministic warning order.
    pub(crate) dedup_groups: Vec<(String, usize)>,
}

/// Plan a buffered tool-call batch for execution: apply the per-message cap
/// in emission order (the first `cap` emitted calls are kept), then dedup
/// identical `(name, input)` pairs within the kept prefix.
///
/// Incident 2026-10-02: a degenerate model emitted ONE assistant message
/// with 7,642 duplicate tool_use blocks; all were accepted and the session
/// wedged. This is the containment choke point — provider-neutral, before
/// any event fan-out or approval.
///
/// Canonical input key: `serde_json` maps are `BTreeMap`-ordered in this
/// workspace (the `preserve_order` feature is NOT enabled), so
/// `serde_json::to_string` is a stable canonical form. If that feature is
/// ever enabled, keys must be sorted manually here.
pub(crate) fn plan_tool_batch(calls: &[ToolCall], cap: usize) -> ToolBatchPlan {
    let mut plan = ToolBatchPlan::default();
    let mut first_seen: HashMap<String, usize> = HashMap::with_capacity(calls.len());
    let mut group_counts: HashMap<String, usize> = HashMap::new();

    for call in calls {
        // The cap applies to the first `cap` emitted calls (emission order);
        // duplicates among them still occupy a kept slot — everything after
        // is dropped from execution.
        if plan.executors.len() + plan.duplicates.len() >= cap {
            plan.dropped.push(call.clone());
            continue;
        }
        let key = format!(
            "{}\u{0}{}",
            call.name,
            serde_json::to_string(&call.input).unwrap_or_else(|_| call.input.to_string())
        );
        match first_seen.get(&key) {
            Some(&executor_index) => {
                plan.duplicates.push((call.id.clone(), executor_index));
            }
            None => {
                first_seen.insert(key.clone(), plan.executors.len());
                plan.executors.push(call.clone());
            }
        }
        *group_counts.entry(key).or_insert(0) += 1;
    }

    plan.dedup_groups = group_counts
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .map(|(key, count)| {
            let name = key
                .split_once('\u{0}')
                .map(|(name, _)| name)
                .unwrap_or(key.as_str());
            (name.to_string(), count)
        })
        .collect();
    plan.dedup_groups.sort();
    plan
}

/// Provider-neutral error payload for synthetic tool results injected by
/// the batch guardrails (the duplicate-rebind fallback in the driver). A
/// JSON string so any provider accepts it on re-upload.
pub(crate) fn guard_tool_result_error(message: &str) -> String {
    serde_json::json!({ "error": message }).to_string()
}

/// Truncate a string to `max_chars` characters, appending "…" if truncated.
/// `pub` for cross-crate use (runtime's subagent registry truncates result
/// summaries with the same bound as the spawn tool's error surfacing).
pub fn truncate_str(s: &str, max_chars: usize) -> String {
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

/// Render a tool result into the model-visible text for a `tool` role
/// message.
///
/// Successful results render the raw `output`. Failed results prefer a
/// non-empty `error`, but when `error` is absent or blank they fall back to
/// the non-empty `output` — some tools attach the real diagnostics to
/// `output` on failure (e.g. the runtime bash tool stores a non-zero exit's
/// stdout there, pinned by
/// `crates/runtime/tests/bash_tool_diagnostics.rs`), and discarding it left
/// the model with nothing but `"tool failed"`, inviting blind identical
/// retries. The literal `"tool failed"` is only the both-empty last resort.
/// The rendered text still passes through [`truncate_tool_output`].
pub(crate) fn format_tool_result(result: &nca_common::tool::ToolResult) -> String {
    let raw = if result.success {
        result.output.clone()
    } else {
        result
            .error
            .clone()
            .filter(|e| !e.trim().is_empty())
            .or_else(|| (!result.output.trim().is_empty()).then(|| result.output.clone()))
            .unwrap_or_else(|| "tool failed".to_string())
    };
    truncate_tool_output(&raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `ToolResult` directly — no tools or I/O needed.
    fn tool_result(
        success: bool,
        output: &str,
        error: Option<&str>,
    ) -> nca_common::tool::ToolResult {
        nca_common::tool::ToolResult {
            call_id: "c1".into(),
            success,
            output: output.into(),
            error: error.map(str::to_string),
            timed_out: false,
        }
    }

    #[test]
    fn failed_result_prefers_non_empty_error() {
        let rendered = format_tool_result(&tool_result(false, "some output", Some("boom")));
        assert_eq!(rendered, "boom");
    }

    #[test]
    fn failed_result_falls_back_to_output_when_error_missing() {
        // The bash-tool failure shape: non-zero exit with stdout stored in
        // `output` and `error: None`. The diagnostics must reach the model.
        let rendered = format_tool_result(&tool_result(false, "boom\n", None));
        assert_eq!(rendered, "boom\n");
    }

    #[test]
    fn failed_result_blank_error_is_ignored() {
        let rendered = format_tool_result(&tool_result(false, "real diagnostics", Some("   ")));
        assert_eq!(rendered, "real diagnostics");
    }

    #[test]
    fn failed_result_with_both_empty_renders_literal() {
        let rendered = format_tool_result(&tool_result(false, "", None));
        assert_eq!(rendered, "tool failed");
        let blanked = format_tool_result(&tool_result(false, "  \n\t", Some("")));
        assert_eq!(blanked, "tool failed");
    }

    #[test]
    fn successful_result_renders_raw_output_unchanged() {
        let rendered = format_tool_result(&tool_result(true, "all good", Some("ignored")));
        assert_eq!(rendered, "all good");
    }

    #[test]
    fn long_output_fallback_is_truncated() {
        let big = "x".repeat(MAX_TOOL_OUTPUT_BYTES + 4096);
        let rendered = format_tool_result(&tool_result(false, &big, None));
        assert!(
            rendered.contains("[... truncated "),
            "overlong fallback output must pass through the truncation guard"
        );
        assert!(
            rendered.len() < MAX_TOOL_OUTPUT_BYTES + 4096,
            "rendered text must be smaller than the raw input"
        );
    }

    #[test]
    fn turn_user_message_shapes() {
        use nca_common::message::MessageContent;

        // Plain text, no attachments → simple text message.
        let plain = turn_user_message("hello", &[]);
        assert!(matches!(&plain.content, MessageContent::Text(t) if t == "hello"));

        let images = [ImageAttachment {
            media_type: "image/png".into(),
            path: "shot.png".into(),
        }];

        // Prompt + images → parts, text first.
        let with_prompt = turn_user_message("look", &images);
        let MessageContent::Parts(parts) = with_prompt.content else {
            panic!("expected parts content")
        };
        assert_eq!(parts.len(), 2);
        assert!(matches!(&parts[0], ContentPart::Text { text } if text == "look"));
        assert!(matches!(&parts[1], ContentPart::Image { .. }));

        // Blank prompt + images → placeholder text part.
        let image_only = turn_user_message("  ", &images);
        let MessageContent::Parts(parts) = image_only.content else {
            panic!("expected parts content")
        };
        assert_eq!(parts.len(), 2);
        assert!(matches!(&parts[0], ContentPart::Text { text } if text.contains("attached image")));
    }
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

    #[test]
    fn set_system_prompt_replaces_stacked_persona() {
        use nca_common::message::MessageContent;

        // Persona-stacking regression: every rebuild (Tab switch, /mount)
        // used to PUSH a new system message, so switching @oracle → @fixer
        // left both personas steering (and billing) every later turn.
        // Replace semantics: exactly one system message — the newest —
        // at the head, non-system history preserved in order.
        let (provider, _calls) = ScriptedProvider::new(Vec::new());
        let mut agent = test_agent(Arc::new(provider));

        agent.set_system_prompt("first persona");
        agent.messages.push(Message::user("hello"));
        agent.messages.push(Message::assistant("hi"));
        agent.messages.push(Message::system("mid-history note"));

        agent.set_system_prompt("second persona");
        agent.set_system_prompt("third persona");

        let system_count = agent
            .messages
            .iter()
            .filter(|m| matches!(m.role, Role::System))
            .count();
        assert_eq!(system_count, 1, "exactly one system message after switches");
        assert_eq!(agent.messages[0].role, Role::System);
        let MessageContent::Text(text) = &agent.messages[0].content else {
            panic!("expected text system prompt");
        };
        assert_eq!(text, "third persona", "the newest prompt wins");
        assert!(!text.contains("first persona"));
        assert!(!text.contains("second persona"));
        // Non-system history survives the rebuild, in order.
        assert_eq!(agent.messages.len(), 3);
        assert_eq!(agent.messages[1].role, Role::User);
        assert_eq!(agent.messages[2].role, Role::Assistant);
    }

    #[test]
    fn set_system_prompt_on_fresh_history_is_plain_prepend() {
        // Initial-call path (no existing system message): behavior identical
        // to the old push — the prompt lands at the head of an empty history.
        let (provider, _calls) = ScriptedProvider::new(Vec::new());
        let mut agent = test_agent(Arc::new(provider));
        agent.set_system_prompt("initial prompt");
        assert_eq!(agent.messages.len(), 1);
        assert_eq!(agent.messages[0].role, Role::System);
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

    // P4 wait-guard turn boundary: the wait-tool repeat counter is
    // turn-scoped. Burning it into the refused state during "turn 1" must
    // NOT leak into the next turn — `run_turn` resets it at `TurnStarted`,
    // so the first `wait_for_user` of the new turn proceeds again.
    #[tokio::test]
    async fn run_turn_resets_wait_tool_guard_at_turn_start() {
        use crate::tool_guards::RepeatAction;

        let (provider, _calls) = ScriptedProvider::new(vec![vec![
            StreamChunk::TextDelta("ok".into()),
            StreamChunk::Finish {
                reason: "stop".into(),
            },
        ]]);
        let mut agent = test_agent(Arc::new(provider));

        // Simulate a degenerate prior turn: wait called 3+ times → refused.
        let input = serde_json::json!({});
        for _ in 0..3 {
            let _ = agent.repeat_guard.record("wait_for_user", &input);
        }
        assert!(
            matches!(
                agent.repeat_guard.record("wait_for_user", &input),
                RepeatAction::Stop(_)
            ),
            "burn-in must reach the refused state"
        );

        agent
            .run_turn("next user message", Path::new("."), &[])
            .await
            .expect("turn must complete");

        // New turn: first wait call is legitimate again; the escalation only
        // restarts on repeats *within* the new turn.
        assert_eq!(
            agent.repeat_guard.record("wait_for_user", &input),
            RepeatAction::Proceed
        );
        assert!(matches!(
            agent.repeat_guard.record("wait_for_user", &input),
            RepeatAction::Hint(_)
        ));
    }

    // `/new` regression seam: the supervisor reuses this AgentLoop across
    // sessions, so `reset_repeat_guard` must clear the session-scoped
    // identical-call counts (the runtime wiring calls it from
    // `reset_for_new_session`).
    #[test]
    fn reset_repeat_guard_clears_session_scoped_counts() {
        use crate::tool_guards::RepeatAction;

        let (provider, _calls) = ScriptedProvider::new(Vec::new());
        let mut agent = test_agent(Arc::new(provider));

        let input = serde_json::json!({"path": "same"});
        for _ in 0..8 {
            let _ = agent.repeat_guard.record("read_file", &input);
        }
        assert!(
            matches!(
                agent.repeat_guard.record("read_file", &input),
                RepeatAction::Stop(_)
            ),
            "burn-in must reach the hard stop"
        );

        agent.reset_repeat_guard();
        assert_eq!(
            agent.repeat_guard.record("read_file", &input),
            RepeatAction::Proceed,
            "fresh session must restore first-call behavior"
        );
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

    fn pct(id: &str, name: &str, input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            input,
        }
    }

    #[test]
    fn plan_tool_batch_cap_window_and_dedup() {
        // cap=3 over [A, A, B, C, A]: the kept window is the first 3
        // emitted calls (A, A, B) — C and the trailing A are dropped
        // regardless of dedup. Executors are the distinct kept calls.
        let calls = vec![
            pct("a1", "read", serde_json::json!({"p": 1})),
            pct("a2", "read", serde_json::json!({"p": 1})),
            pct("b1", "read", serde_json::json!({"p": 2})),
            pct("c1", "search", serde_json::json!({"q": "x"})),
            pct("a3", "read", serde_json::json!({"p": 1})),
        ];
        let plan = plan_tool_batch(&calls, 3);
        assert_eq!(
            plan.executors
                .iter()
                .map(|c| c.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a1", "b1"]
        );
        assert_eq!(plan.duplicates, vec![("a2".to_string(), 0)]);
        assert_eq!(
            plan.dropped
                .iter()
                .map(|c| c.id.as_str())
                .collect::<Vec<_>>(),
            vec!["c1", "a3"]
        );
        assert_eq!(plan.dedup_groups, vec![("read".to_string(), 2)]);
    }

    #[test]
    fn plan_tool_batch_input_key_is_order_insensitive() {
        // Same logical input with different key order must dedup: serde_json
        // maps are BTreeMap-ordered here (no `preserve_order` feature), so
        // to_string is canonical.
        let calls = vec![
            pct(
                "x1",
                "edit",
                serde_json::json!({"path": "a", "content": "b"}),
            ),
            pct(
                "x2",
                "edit",
                serde_json::json!({"content": "b", "path": "a"}),
            ),
        ];
        let plan = plan_tool_batch(&calls, 32);
        assert_eq!(plan.executors.len(), 1);
        assert_eq!(plan.duplicates.len(), 1);
        assert!(plan.dropped.is_empty());
    }

    #[test]
    fn plan_tool_batch_cap_zero_drops_everything() {
        let calls = vec![pct("a", "read", serde_json::json!({}))];
        let plan = plan_tool_batch(&calls, 0);
        assert!(plan.executors.is_empty());
        assert!(plan.duplicates.is_empty());
        assert_eq!(plan.dropped.len(), 1);
    }

    // ── Chunk 3b: backpressure classification of `AgentLoop::emit` ──────

    /// Droppable notices are ELIDED (not blocking, not silent): with a full
    /// bounded channel and no consumer, `try_send` fails, the counter climbs,
    /// and the next must-deliver send first flushes an `EventsElided` summary
    /// so consumers see the gap adjacent to the next guaranteed event.
    #[tokio::test]
    async fn emit_elides_droppable_and_flushes_summary_before_next_must_deliver() {
        let (provider, _calls) = ScriptedProvider::new(Vec::new());
        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(2);
        let agent = AgentLoop::new(
            Arc::new(provider),
            ToolRegistry::new(),
            ApprovalPolicy::new(PermissionConfig::default()),
            "m".into(),
            event_tx,
            10,
            16,
            0,
            None,
        );

        // No consumer: buffer absorbs 2, the next 3 Started are elided.
        // Pre-fix each of these was a blocking `send().await` — this loop
        // would have parked forever on the third (the incident's stream-loop
        // wedge).
        for i in 0..5 {
            agent
                .emit(AgentEvent::ToolCallStarted {
                    call_id: format!("c{i}"),
                    tool: "read".into(),
                    input: serde_json::json!({}),
                })
                .await;
        }
        assert_eq!(
            agent.elided_events.load(Ordering::SeqCst),
            3,
            "exactly the overflow is counted"
        );

        // Consumer appears; the must-deliver send unblocks and carries the
        // summary immediately before itself.
        let consumer = tokio::spawn(async move {
            let mut got = Vec::new();
            while let Some(event) = event_rx.recv().await {
                let done = matches!(event, AgentEvent::TurnStarted { .. });
                got.push(event);
                if done {
                    break;
                }
            }
            got
        });
        agent.emit(AgentEvent::TurnStarted { turn_id: 1 }).await;
        let got = consumer.await.expect("consumer");
        assert_eq!(
            got.len(),
            4,
            "2 buffered + summary + the must-deliver event"
        );
        assert!(matches!(&got[0], AgentEvent::ToolCallStarted { .. }));
        assert!(matches!(&got[1], AgentEvent::ToolCallStarted { .. }));
        assert!(matches!(&got[2], AgentEvent::EventsElided { count: 3 }));
        assert!(matches!(&got[3], AgentEvent::TurnStarted { .. }));
        assert_eq!(
            agent.elided_events.load(Ordering::SeqCst),
            0,
            "summary flush resets the counter"
        );
    }

    /// Must-deliver events survive backpressure intact: a buffer-1 channel
    /// paced by a 1 ms-per-event consumer forces every send through the
    /// blocking path, and all of them arrive — no elision, no summary.
    #[tokio::test]
    async fn emit_never_elides_must_deliver_events() {
        let (provider, _calls) = ScriptedProvider::new(Vec::new());
        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(1);
        let agent = AgentLoop::new(
            Arc::new(provider),
            ToolRegistry::new(),
            ApprovalPolicy::new(PermissionConfig::default()),
            "m".into(),
            event_tx,
            10,
            16,
            0,
            None,
        );
        let events = vec![
            AgentEvent::MessageReceived {
                role: "user".into(),
                content: "hi".into(),
                steering: false,
            },
            AgentEvent::StepStarted {
                turn_id: 1,
                step_index: 1,
            },
            AgentEvent::MessageRecorded {
                message: Message::assistant("reply"),
            },
            AgentEvent::ToolCallCompleted {
                call_id: "c".into(),
                output: nca_common::tool::ToolResult {
                    call_id: "c".into(),
                    success: true,
                    output: "ok".into(),
                    error: None,
                    timed_out: false,
                },
                duration_ms: 1,
            },
            AgentEvent::Error {
                message: "boom".into(),
            },
            AgentEvent::TurnCompleted {
                turn_id: 1,
                duration_ms: 2,
            },
        ];
        let total = events.len();

        let consumer = async {
            let mut got = Vec::new();
            while let Some(event) = event_rx.recv().await {
                let done = matches!(event, AgentEvent::TurnCompleted { .. });
                got.push(event);
                if done {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
            got
        };
        let (_, got) = tokio::join!(
            async {
                for event in events {
                    agent.emit(event).await;
                }
            },
            consumer
        );
        assert_eq!(got.len(), total, "every must-deliver event arrived");
        assert!(
            !got.iter()
                .any(|e| matches!(e, AgentEvent::EventsElided { .. })),
            "nothing was elided — only droppable kinds ever are"
        );
        assert_eq!(agent.elided_events.load(Ordering::SeqCst), 0);
    }

    /// Stall-alarm reserve loop (event-consumer-liveness Chunk 2): against a
    /// full buffer with no consumer, the send stays pending across many
    /// 30s alarm boundaries — no panic, no loss, no duplicate — then
    /// delivers exactly once and returns as soon as the consumer drains.
    /// Paused time makes the multi-boundary stall instantaneous.
    #[tokio::test(start_paused = true)]
    async fn send_with_stall_alarm_survives_stall_and_delivers_exactly_once() {
        use std::time::Duration;

        let (tx, mut rx) = tokio::sync::mpsc::channel::<AgentEvent>(1);
        // Occupy the single buffer slot so `reserve()` has something to wait on.
        tx.send(AgentEvent::Checkpoint {
            phase: "filler".into(),
            detail: String::new(),
            turn: 0,
        })
        .await
        .expect("fill buffer");

        let sender = tx.clone();
        let task = tokio::spawn(async move {
            send_with_stall_alarm(
                &sender,
                AgentEvent::Error {
                    message: "boom".into(),
                },
            )
            .await;
        });

        // Let the spawned task park on `reserve()`, then cross exactly one
        // alarm boundary (interval is 30s; +1s margin).
        tokio::time::sleep(Duration::from_millis(1)).await;
        tokio::time::advance(Duration::from_secs(31)).await;
        tokio::time::sleep(Duration::from_millis(1)).await;
        assert!(
            !task.is_finished(),
            "must stay pending while the consumer refuses to drain"
        );
        // The stalled send must not have pre-empted or duplicated the
        // buffered filler: the buffer still holds exactly it, and the only
        // way to observe that without unblocking the send is that the task
        // is still parked (checked above) — consuming here would wake it.

        // Many more boundaries: the re-arm loop is stable — no panic, no
        // loss, no double-send.
        tokio::time::advance(Duration::from_secs(300)).await;
        tokio::time::sleep(Duration::from_millis(1)).await;
        assert!(!task.is_finished(), "still pending after ~10 more alarms");

        // Consumer drains: the filler first (it was buffered), then exactly
        // one delivery of the guaranteed event.
        let filler = rx.recv().await.expect("filler");
        assert!(matches!(filler, AgentEvent::Checkpoint { .. }));
        let got = rx.recv().await.expect("exactly one delivery");
        assert!(
            matches!(got, AgentEvent::Error { .. }),
            "the original event survives the stall intact"
        );
        assert!(rx.try_recv().is_err(), "no duplicate delivery");

        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("helper returns once delivered")
            .expect("task join");
    }
}
