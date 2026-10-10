//! Turn/Step driver: the layer that owns inbox claiming and the per-turn
//! step loop (`docs/plans/p1-turn-step-design.md` §3).
//!
//! One `run_turn` == one [`TurnDriver`] run == N steps. A step is exactly one
//! provider `chat()` call plus its tool pipeline. Inbox items (queued prompts
//! and steering) are claimed at step boundaries only — never mid-stream.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use nca_common::event::{AgentEvent, BusyState};
use nca_common::message::{ImageAttachment, Message, MessageToolCall};
use nca_common::tool::{ToolCall, ToolResult};
use serde_json::json;

use crate::agent::{AgentLoop, LOOP_BREAKER_MARKER};
use crate::cache_keepalive::{CacheKeepalive, KeepaliveSnapshot};
use crate::hooks::HookEventKind;
use crate::middleware::{StepReply, StepRequest};
use crate::provider::{ProviderError, StreamChunk};
use crate::tool_pipeline;

/// Item delivered to a running turn's inbox; claimed at step boundaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InboxItem {
    /// Full user prompt queued while a turn was running. Appended as a user
    /// message at the next step boundary (in arrival order).
    UserPrompt { text: String },
    /// Mid-turn steering guidance. Same delivery as `UserPrompt`, but the
    /// `MessageReceived` event carries `steering: true` for UI distinction.
    Steering { text: String },
}

/// Outcome of a single step, consumed by the driver loop to decide whether
/// another step is owed.
#[derive(Debug)]
enum StepOutcome {
    /// The provider produced a final answer (or the loop stopped with an
    /// early-final message); the turn ends returning this text.
    FinalText {
        text: String,
        /// Whether the step also executed tool calls (true for the
        /// consecutive-tool-failure early stop).
        had_tool_calls: bool,
    },
    /// Tool calls executed; the conversation continues with another step.
    Continue { had_tool_calls: bool },
    /// Empty response retry: another step is owed without tool calls.
    Retry,
}

impl AgentLoop {
    /// Claim everything currently in the inbox (non-blocking drain) and
    /// append each item as a user message in arrival order, emitting
    /// `MessageReceived` with the per-kind steering flag and full text.
    /// Returns the number of items claimed. Used both at turn start
    /// (leftovers) and at each step boundary.
    pub(crate) async fn claim_inbox(&mut self) -> usize {
        let mut claimed = Vec::new();
        while let Ok(item) = self.inbox_rx.try_recv() {
            claimed.push(item);
        }
        let n = claimed.len();
        for item in claimed {
            let (text, steering) = match item {
                InboxItem::UserPrompt { text } => (text, false),
                InboxItem::Steering { text } => (text, true),
            };
            let msg = Message::user(text.clone());
            self.record(&msg).await;
            self.messages.push(msg);
            self.emit(AgentEvent::MessageReceived {
                role: "user".into(),
                content: text,
                steering,
            })
            .await;
        }
        n
    }
}

/// Drives one turn: claims inbox items at step boundaries and runs steps
/// until nothing is owed. Internal structure, not public API.
pub(crate) struct TurnDriver<'a> {
    agent: &'a mut AgentLoop,
    workspace_root: &'a Path,
    turn_id: u64,
    /// 1-based step counter within the turn; also `Checkpoint.turn`.
    step_index: u64,
    /// Bounded empty-response retries within the turn.
    empty_retries: u32,
    /// Whether processed attachments have already been cleaned (first
    /// completed stream of the turn).
    attachments_cleaned: bool,
    /// Consecutive failing steps on the IDENTICAL-retry track — the batch
    /// fingerprint (tool name + canonical inputs) exactly matches the
    /// previous failing step's, i.e. the model is verbatim-retrying. Stops
    /// infinite retry loops fast (threshold
    /// [`MAX_CONSECUTIVE_TOOL_FAILURES`]), aligned with the repeat guard's
    /// hard-stop semantics for true loops.
    consecutive_tool_failures: u32,
    last_failed_tool: String,
    /// Consecutive failing steps on the VARYING track — the batch
    /// fingerprint keeps changing, i.e. the model is iterating (bash-only
    /// debug workflows: sed → build → test, every step pure `execute_bash`,
    /// legitimately string together 3+ failing steps while fixing). Trips
    /// at [`MAX_VARYING_CONSECUTIVE_TOOL_FAILURES`] with a one-time warning
    /// at [`VARYING_FAILURE_WARN_THRESHOLD`].
    consecutive_varying_failures: u32,
    /// Fingerprint of the most recent failing step's executor batch
    /// (0 = no failing step yet this streak; never consulted as a sentinel
    /// because streak-start is detected via the zeroed counters).
    last_failed_batch_fingerprint: u64,
    /// Whether the varying-track warning already fired for the current
    /// streak (one warning per streak; reset by any non-failed batch).
    varying_warned: bool,
    /// Diagnostic details from the most recent failure (populated by the
    /// `all_failed_same_tool` branch; only read when the max is reached).
    last_failed_output: String,
    last_failed_error: Option<String>,
    /// Consecutive steps in which the repeat-call guard refused at least one
    /// call (`RepeatAction::Stop`). Guard refusals are already delivered to
    /// the model as failed tool results; a model that keeps collecting them
    /// across consecutive steps is ignoring explicit correction and only
    /// burning inference passes (each refused call still costs a full-context
    /// re-prefill). At [`MAX_CONSECUTIVE_GUARD_STOP_STEPS`] the turn is ended
    /// with the refusals spelled out — for a sub-agent that output surfaces
    /// to the parent via the spawn reply / task_result.
    consecutive_guard_stop_steps: u32,
}

const MAX_EMPTY_RETRIES: u32 = 2;
const MAX_CONSECUTIVE_TOOL_FAILURES: u32 = 3;
/// Consecutive VARYING-shape failing steps (batch fingerprint keeps
/// changing — iteration, not verbatim retry) before the turn is force-ended.
/// Higher than the identical threshold: killing a legitimate fix → test →
/// fail iteration at 3 aborted normal workflows (the incident shape). True
/// verbatim loops stay on the fast threshold-3 track (and the repeat guard's
/// 3/5/8 ladder underneath).
const MAX_VARYING_CONSECUTIVE_TOOL_FAILURES: u32 = 5;
/// The varying track emits ONE user-visible warning at this count — a
/// correction opportunity that does not consume the turn.
const VARYING_FAILURE_WARN_THRESHOLD: u32 = 3;
/// Consecutive steps that each contain at least one guard refusal before the
/// turn is force-ended. The guard only Stops a call on its 8th+ identical
/// issue, so one refusal-bearing step already implies seven identical priors;
/// three in a row is unambiguous loop pathology, not legitimate iteration
/// (fix → test loops vary their calls and never reach a Stop).
const MAX_CONSECUTIVE_GUARD_STOP_STEPS: u32 = 3;

impl<'a> TurnDriver<'a> {
    pub(crate) fn new(
        agent: &'a mut AgentLoop,
        workspace_root: &'a Path,
        turn_id: u64,
        attachments: &[ImageAttachment],
    ) -> Self {
        Self {
            agent,
            workspace_root,
            turn_id,
            step_index: 0,
            empty_retries: 0,
            attachments_cleaned: attachments.is_empty(),
            consecutive_tool_failures: 0,
            last_failed_tool: String::new(),
            consecutive_varying_failures: 0,
            last_failed_batch_fingerprint: 0,
            varying_warned: false,
            last_failed_output: String::new(),
            last_failed_error: None,
            consecutive_guard_stop_steps: 0,
        }
    }

    /// Emit `StepCompleted` for the step that just ran.
    async fn emit_step_completed(&mut self, duration_ms: u64, had_tool_calls: bool) {
        self.agent
            .emit(AgentEvent::StepCompleted {
                turn_id: self.turn_id,
                step_index: self.step_index,
                duration_ms,
                had_tool_calls,
            })
            .await;
    }

    /// The driver loop: claim inbox → cancel check → StepStarted → step().
    /// `owed` is true while the conversation still expects another step
    /// (tool calls executed, or an empty-response retry).
    pub(crate) async fn run(
        mut self,
        attachments: &[ImageAttachment],
    ) -> Result<String, ProviderError> {
        let mut owed = true;
        let final_text = loop {
            let claimed = self.agent.claim_inbox().await;
            if claimed > 0 {
                owed = true;
            }
            if !owed && claimed == 0 {
                break String::new();
            }
            if self.agent.is_cancelled() {
                self.agent
                    .emit(AgentEvent::Error {
                        message: "Run cancelled".into(),
                    })
                    .await;
                // Bracket rule parity: run_turn truncates to baseline on Err, so the
                // bracket must carry a StepFailed or replay would fold rolled-back messages.
                self.agent
                    .emit(AgentEvent::StepFailed {
                        turn_id: self.turn_id,
                        step_index: self.step_index,
                        duration_ms: 0,
                        error: "run cancelled".into(),
                    })
                    .await;
                return Err(ProviderError::Other("run cancelled".into()));
            }
            self.step_index += 1;
            self.agent
                .emit(AgentEvent::StepStarted {
                    turn_id: self.turn_id,
                    step_index: self.step_index,
                })
                .await;
            let step_start = Instant::now();
            let outcome = self.step(attachments).await;
            let duration_ms = step_start.elapsed().as_millis() as u64;
            match outcome {
                Ok(StepOutcome::FinalText {
                    text,
                    had_tool_calls,
                }) => {
                    self.emit_step_completed(duration_ms, had_tool_calls).await;
                    break text;
                }
                Ok(StepOutcome::Continue { had_tool_calls }) => {
                    self.emit_step_completed(duration_ms, had_tool_calls).await;
                    owed = had_tool_calls;
                }
                Ok(StepOutcome::Retry) => {
                    self.emit_step_completed(duration_ms, false).await;
                    owed = true;
                }
                Err(e) => {
                    self.agent
                        .emit(AgentEvent::StepFailed {
                            turn_id: self.turn_id,
                            step_index: self.step_index,
                            duration_ms,
                            error: e.to_string(),
                        })
                        .await;
                    return Err(e);
                }
            }
        };

        let agent = self.agent;
        if agent.cost_tracker.input_tokens == 0 && agent.cost_tracker.output_tokens == 0 {
            let estimated_input = (agent
                .messages
                .iter()
                .map(|message| message.content.approx_chars())
                .sum::<usize>()
                / 4) as u64;
            let estimated_output = (final_text.len() / 4) as u64;
            agent
                .cost_tracker
                .add(estimated_input, estimated_output, 0, 0);
            agent
                .emit(AgentEvent::CostUpdated {
                    input_tokens: agent.cost_tracker.input_tokens,
                    output_tokens: agent.cost_tracker.output_tokens,
                    cache_read_tokens: agent.cost_tracker.cache_read_tokens,
                    estimated_cost_usd: agent.cost_tracker.estimated_cost_usd(),
                })
                .await;
        }

        if let Some(hooks) = &agent.hooks {
            let response_preview = crate::agent::truncate_str(&final_text, 300);
            hooks
                .run_best_effort(
                    HookEventKind::TurnComplete,
                    None,
                    &json!({
                        "response_preview": response_preview,
                        "workspace": self.workspace_root.display().to_string(),
                    }),
                )
                .await;
        }

        Ok(final_text)
    }

    /// One step: budget check, compaction, a single provider `chat()` with
    /// the full stream loop, assistant/tool message bookkeeping, and the tool
    /// pipeline. Behavior is preserved verbatim from the former
    /// `run_turn_inner` per-iteration body.
    async fn step(
        &mut self,
        attachments: &[ImageAttachment],
    ) -> Result<StepOutcome, ProviderError> {
        let agent = &mut *self.agent;

        if self.step_index > agent.max_turns as u64 {
            let msg = format!("turn budget exceeded (max {})", agent.max_turns);
            agent
                .emit(AgentEvent::Error {
                    message: msg.clone(),
                })
                .await;
            return Err(ProviderError::Other(msg));
        }

        agent
            .emit(AgentEvent::BusyStateChanged {
                state: BusyState::Thinking,
            })
            .await;
        let turn = self.step_index as u32;
        agent
            .emit(AgentEvent::Checkpoint {
                phase: "provider_request".into(),
                detail: format!("Starting model turn {turn}"),
                turn,
            })
            .await;
        agent
            .provider
            .prepare_messages_for_request(&mut agent.messages, self.workspace_root)
            .await?;
        // Repair any orphaned `tool_calls` left by a previously interrupted
        // turn (budget/pipeline error) so strict providers like DeepSeek don't
        // reject the request with "tool_calls must be followed by tool
        // messages". Persisted to `agent.messages` so resumed sessions stay valid.
        crate::agent::sanitize_tool_call_pairs(&mut agent.messages);

        // The chain owns request-view shaping now (CompactionMiddleware);
        // the driver hands the canonical post-prepare/sanitize history.
        let session_usage = crate::middleware::SessionUsage {
            input_tokens: agent.cost_tracker.input_tokens,
            output_tokens: agent.cost_tracker.output_tokens,
            cache_creation_tokens: agent.cost_tracker.cache_creation_tokens,
            cache_read_tokens: agent.cost_tracker.cache_read_tokens,
        };
        let reply = agent
            .middleware
            .call(
                &agent.provider,
                StepRequest {
                    messages: agent.messages.clone(),
                    tools: agent.tool_definitions(),
                    model: agent.model.clone(),
                    workspace_root: self.workspace_root.to_path_buf(),
                    turn_id: self.turn_id,
                    step_index: self.step_index,
                    session_usage,
                    event_tx: agent.event_tx.clone(),
                },
            )
            .await;
        let reply = match reply {
            Ok(reply) => reply,
            Err(e) => {
                // Pre-stream provider failures (e.g. DeepSeek HTTP 402
                // "Insufficient Balance") must reach event consumers — the
                // TUI transcript and orchestrator LLMs only see `AgentEvent`s,
                // not tracing. The `StepFailed` emitted by `TurnDriver::run`'s
                // `Err` branch remains the last word (replay bracket parity).
                agent
                    .emit(AgentEvent::Error {
                        message: e.to_string(),
                    })
                    .await;
                return Err(e);
            }
        };

        let mut stream = match reply {
            StepReply::Stream(stream) => stream,
            StepReply::FinalText(text) => {
                // Empty short-circuit text is a middleware bug — fail loudly,
                // do NOT record an empty assistant message (the empty-response
                // policy exists precisely because empty assistant messages
                // confuse providers and would replay back on resume).
                if text.trim().is_empty() {
                    return Err(ProviderError::Other(
                        "middleware short-circuited the step with empty final text".into(),
                    ));
                }
                // Attachments are real inputs even though the provider never
                // ran: run the same cleanup the stream path does, BEFORE
                // recording, so history + disk stay consistent (oracle P1-1).
                if !self.attachments_cleaned {
                    crate::agent::cleanup_processed_attachments(
                        &mut agent.messages,
                        self.workspace_root,
                        attachments,
                    );
                    self.attachments_cleaned = true;
                }
                // Replay-safe short-circuit: identical bookkeeping to the
                // normal final-text path — record, push, emit, then FinalText.
                // The empty-response retry counter is intentionally NOT
                // applied: the provider never ran.
                let msg = Message::assistant(text.clone());
                agent.record(&msg).await;
                agent.messages.push(msg);
                agent
                    .emit(AgentEvent::MessageReceived {
                        role: "assistant".into(),
                        content: text.clone(),
                        steering: false,
                    })
                    .await;
                return Ok(StepOutcome::FinalText {
                    text,
                    had_tool_calls: false,
                });
            }
        };

        let mut assistant_text = String::new();
        let mut reasoning_text = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut got_usage = false;
        let mut finish_reason: Option<String> = None;

        let mut cancel_poll = tokio::time::interval(Duration::from_millis(25));
        cancel_poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            let chunk = tokio::select! {
                _ = cancel_poll.tick() => {
                    if agent.is_cancelled() {
                        agent
                            .emit(AgentEvent::Error {
                                message: "Run cancelled while streaming model output".into(),
                            })
                            .await;
                        return Err(ProviderError::Other("run cancelled".into()));
                    }
                    continue;
                }
                chunk = stream.recv() => chunk,
            };
            let Some(chunk) = chunk else {
                break;
            };
            match chunk {
                StreamChunk::TextDelta(delta) => {
                    if assistant_text.is_empty() {
                        agent
                            .emit(AgentEvent::BusyStateChanged {
                                state: BusyState::Streaming,
                            })
                            .await;
                    }
                    assistant_text.push_str(&delta);
                    agent.emit(AgentEvent::TokensStreamed { delta }).await;
                }
                StreamChunk::ReasoningDelta(delta) => {
                    reasoning_text.push_str(&delta);
                    agent.emit(AgentEvent::ReasoningStreamed { delta }).await;
                }
                StreamChunk::ToolUse(call) => {
                    agent
                        .tool_start_times
                        .insert(call.id.clone(), Instant::now());
                    agent
                        .emit(AgentEvent::ToolCallStarted {
                            call_id: call.id.clone(),
                            tool: call.name.clone(),
                            input: call.input.clone(),
                        })
                        .await;
                    tool_calls.push(call);
                }
                StreamChunk::Usage {
                    input_tokens,
                    output_tokens,
                    cache_creation_tokens,
                    cache_read_tokens,
                } => {
                    got_usage = true;
                    agent.cost_tracker.add(
                        input_tokens,
                        output_tokens,
                        cache_creation_tokens,
                        cache_read_tokens,
                    );
                    agent
                        .emit(AgentEvent::CostUpdated {
                            input_tokens: agent.cost_tracker.input_tokens,
                            output_tokens: agent.cost_tracker.output_tokens,
                            cache_read_tokens: agent.cost_tracker.cache_read_tokens,
                            estimated_cost_usd: agent.cost_tracker.estimated_cost_usd(),
                        })
                        .await;
                }
                StreamChunk::Error(err) => {
                    agent
                        .emit(AgentEvent::Error {
                            message: err.to_string(),
                        })
                        .await;
                    return Err(err);
                }
                StreamChunk::Finish { reason } => {
                    finish_reason = Some(reason);
                }
                StreamChunk::Done => break,
            }
        }

        if !self.attachments_cleaned {
            crate::agent::cleanup_processed_attachments(
                &mut agent.messages,
                self.workspace_root,
                attachments,
            );
            self.attachments_cleaned = true;
        }

        if tool_calls.is_empty() {
            if assistant_text.trim().is_empty() {
                // Thinking-locked models (e.g. ZhipuAI GLM-5.3, whose thinking
                // cannot be disabled) can spend the entire max_tokens budget on
                // reasoning_content and finish with finish_reason="length" and
                // an empty content. Retrying with identical parameters almost
                // always reproduces the same truncation while re-billing the
                // full prompt — fail fast with an actionable message instead.
                if finish_reason.as_deref() == Some("length") {
                    let msg = format!(
                        "Provider returned empty response: generation hit the max_tokens cap \
                         while thinking (finish_reason=length, {} chars of reasoning produced, \
                         no content). Raise [model] max_tokens — GLM-5.x thinking models cannot \
                         disable thinking and ZhipuAI coding examples use 65536 — or lower the \
                         reasoning effort.",
                        reasoning_text.chars().count()
                    );
                    agent
                        .emit(AgentEvent::Error {
                            message: msg.clone(),
                        })
                        .await;
                    return Err(ProviderError::Other(msg));
                }
                self.empty_retries += 1;
                if self.empty_retries <= MAX_EMPTY_RETRIES && got_usage {
                    agent
                        .emit(AgentEvent::Error {
                            message: format!(
                                "Provider returned empty response (retry {}/{MAX_EMPTY_RETRIES})",
                                self.empty_retries
                            ),
                        })
                        .await;
                    return Ok(StepOutcome::Retry);
                }
                let diag = format!(
                    "Provider returned empty response with no tool calls ({} chars of \
                     reasoning produced, finish_reason={})",
                    reasoning_text.chars().count(),
                    finish_reason.as_deref().unwrap_or("unknown"),
                );
                agent
                    .emit(AgentEvent::Error {
                        message: diag.clone(),
                    })
                    .await;
                return Err(ProviderError::Other(format!("{diag} after retries",)));
            }
            let mut msg = Message::assistant(assistant_text.clone());
            if !reasoning_text.is_empty() {
                msg = msg.with_reasoning(std::mem::take(&mut reasoning_text));
            }
            agent.record(&msg).await;
            agent.messages.push(msg);
            agent
                .emit(AgentEvent::MessageReceived {
                    role: "assistant".into(),
                    content: assistant_text.clone(),
                    steering: false,
                })
                .await;
            return Ok(StepOutcome::FinalText {
                text: assistant_text,
                had_tool_calls: false,
            });
        }

        // ── Batch guardrails (incident 2026-10-02) ──────────────────────
        // A degenerate model emitted ONE assistant message with 7,642
        // duplicate tool_use blocks; all were accepted and the session
        // wedged. Contain the blast radius here, at batch finalization,
        // with a plan-before-record strategy: the cap and the dedup run
        // BEFORE the assistant message is recorded, and the recorded
        // message carries only the kept window (the first `cap` calls in
        // emission order — executors and duplicates alike). Overflow ids
        // never enter history, so tool_use/tool-result pairing stays
        // balanced by construction and the incident's ~206K-token history
        // bomb (823 KB of tool_use blocks re-uploaded every step) cannot
        // form. One `ToolBatchGuarded` event surfaces the containment to
        // the user (transcript/system line); a one-line guard note on the
        // first kept result teaches the model.
        let plan = crate::agent::plan_tool_batch(&tool_calls, agent.max_tool_calls_per_message);
        if !plan.dropped.is_empty() {
            tracing::warn!(
                total = tool_calls.len(),
                dropped = plan.dropped.len(),
                cap = agent.max_tool_calls_per_message,
                "tool-call batch exceeded max_tool_calls_per_message; overflow calls are trimmed from the recorded message"
            );
        }
        for (tool, occurrences) in &plan.dedup_groups {
            tracing::warn!(
                tool = %tool,
                occurrences,
                "duplicate tool calls within one batch; executed once, result fanned out to duplicates"
            );
        }
        if !plan.dropped.is_empty() || !plan.dedup_groups.is_empty() {
            agent
                .emit(AgentEvent::ToolBatchGuarded {
                    dropped: plan.dropped.len(),
                    deduplicated: plan.dedup_groups.clone(),
                })
                .await;
        }

        // The kept window is a strict emission-order prefix (the plan drops
        // every call past the cap), so the recorded message is the prefix
        // of length `len - dropped.len()`.
        let kept_len = tool_calls.len() - plan.dropped.len();
        let replay_tool_calls = tool_calls[..kept_len]
            .iter()
            .map(|call| MessageToolCall {
                id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.input.clone(),
            })
            .collect();

        let mut msg = Message::assistant_with_tool_calls(assistant_text, replay_tool_calls);
        if !reasoning_text.is_empty() {
            msg = msg.with_reasoning(std::mem::take(&mut reasoning_text));
        }
        agent.record(&msg).await;
        agent.messages.push(msg);

        let executors = plan.executors;
        if executors.len() as u32 > agent.max_tool_calls_per_turn {
            return Err(ProviderError::Other(format!(
                "tool-call budget exceeded in turn {turn} ({} > {})",
                executors.len(),
                agent.max_tool_calls_per_turn
            )));
        }

        // ── Tool pipeline: permission checks + concurrent execution ──
        // Start cache keepalive for the tool-execution pause.
        // agent.messages now ends with the assistant's tool_calls — the
        // exact prefix the next request will re-send. Keeping it warm
        // avoids a full-price re-prefill when the pause exceeds the
        // provider's cache TTL (~10 min for DeepSeek).
        let snapshot = KeepaliveSnapshot {
            messages: agent.messages.clone(),
            tools: agent.tool_definitions(),
            model: agent.model.clone(),
            workspace_root: self.workspace_root.to_path_buf(),
        };
        let keepalive = CacheKeepalive::start(
            Arc::clone(&agent.provider),
            snapshot,
            agent.keepalive_profile.clone(),
            agent.event_tx.clone(),
        );

        let pipeline = tool_pipeline::run_tool_pipeline(
            &agent.tools,
            &mut agent.approval,
            &agent.hooks,
            &agent.event_tx,
            &agent.cancel_flag,
            executors.clone(),
            &mut agent.repeat_guard,
            &self.workspace_root.display().to_string(),
            &agent.elided_events,
        )
        .await
        .map_err(ProviderError::Other)?;

        // Cancel keepalive — the pause is over, next request is imminent.
        keepalive.stop().await;

        let n = pipeline.results.len();

        // Checkpoint
        if agent.checkpoint_interval > 0 && n as u32 >= agent.checkpoint_interval {
            agent
                .emit(AgentEvent::Checkpoint {
                    phase: "tool_execution".into(),
                    detail: format!("Executed {n} tool calls in turn {turn}"),
                    turn,
                })
                .await;
        }

        // Track consecutive pure-retry steps to detect infinite retry loops.
        // A step qualifies when EVERY call was the SAME tool and every result
        // failed — the step produced nothing but failed retries. (Historically
        // this required `tool_calls.len() == 1`, which never fired for models
        // that batch 4–8 calls per step; observed sub-agent runs showed
        // 20+ guard hard-stops with the breaker silent.) Any successful call
        // or a mixed-tool batch resets the streak — a legitimate fix → test →
        // fail iteration always intersperses edits or reads that succeed.
        //
        // Dual-track retry-form discrimination: the step's executor batch is
        // fingerprinted (tool name + canonical inputs); a fingerprint
        // identical to the previous failing step's is a verbatim retry
        // (identical track, threshold 3 — true loops must stop fast), while a
        // changed fingerprint is iteration (varying track, threshold 5, one
        // warning at 3). Bash-only debug workflows legitimately string
        // together failing build/test steps with changing arguments — the
        // incident's false kill.
        let step_single_tool = executors
            .first()
            .is_some_and(|first| executors.iter().all(|c| c.name == first.name));
        let all_failed_same_tool = !pipeline.results.is_empty()
            && pipeline.results.iter().all(|r| !r.success)
            && step_single_tool;
        if all_failed_same_tool {
            let tool_name = &executors[0].name;
            let last_result = &pipeline.results[0];
            let fingerprint = batch_fingerprint(&executors);
            let streak_start =
                self.consecutive_tool_failures == 0 && self.consecutive_varying_failures == 0;
            if streak_start {
                // The streak's form is only revealed by its successor, so the
                // first failing step seeds BOTH tracks: a streak that stays
                // varying trips at 5 (warning at 3), one that repeats
                // verbatim trips at 3 — preserving the old breaker's timing
                // for true loops.
                self.consecutive_tool_failures = 1;
                self.consecutive_varying_failures = 1;
            } else if fingerprint == self.last_failed_batch_fingerprint {
                // Identical retry shape: the model re-issued the same batch.
                self.consecutive_tool_failures += 1;
                self.consecutive_varying_failures = 0;
            } else {
                // Changed shape: iteration form.
                self.consecutive_varying_failures += 1;
                self.consecutive_tool_failures = 0;
            }
            self.last_failed_tool = tool_name.clone();
            self.last_failed_batch_fingerprint = fingerprint;
            // Capture failure details for diagnostics and the final error message.
            self.last_failed_output = last_result.output.clone();
            self.last_failed_error = last_result.error.clone();
            tracing::warn!(
                tool = %tool_name,
                identical_attempt = self.consecutive_tool_failures,
                varying_attempt = self.consecutive_varying_failures,
                max_identical = MAX_CONSECUTIVE_TOOL_FAILURES,
                max_varying = MAX_VARYING_CONSECUTIVE_TOOL_FAILURES,
                output = %crate::agent::truncate_str(&last_result.output, 500),
                error = ?last_result.error,
                "consecutive tool failure detected"
            );
            // One-time warning on the varying track: user-visible and logged,
            // deliberately WITHOUT the loop-breaker classification marker so
            // child-session terminal folding never misreads a warned-but-
            // continuing turn as failed. Same channel (`AgentEvent::Error`)
            // as the breaker's own message — transcript-visible.
            if !self.varying_warned
                && self.consecutive_varying_failures == VARYING_FAILURE_WARN_THRESHOLD
            {
                self.varying_warned = true;
                let warn_msg = format!(
                    "[nca:loop-breaker-warning] Tool `{tool_name}` has failed \
                     {VARYING_FAILURE_WARN_THRESHOLD} times consecutively with varying parameters \
                     — this looks like iteration rather than an infinite loop, so the turn \
                     continues. The breaker will end the turn at \
                     {MAX_VARYING_CONSECUTIVE_TOOL_FAILURES} consecutive varying failures \
                     ({MAX_CONSECUTIVE_TOOL_FAILURES} for identical retries). If you are stuck, \
                     change strategy, inspect the last failure's diagnostics, or end the turn \
                     with a partial result explaining the gap."
                );
                agent.emit(AgentEvent::Error { message: warn_msg }).await;
            }
        } else {
            self.consecutive_tool_failures = 0;
            self.last_failed_tool.clear();
            self.consecutive_varying_failures = 0;
            self.last_failed_batch_fingerprint = 0;
            self.varying_warned = false;
        }

        // Assemble the result set for the KEPT window only: real executor
        // results plus copies re-bound to duplicate call_ids. Overflow ids
        // were trimmed from the recorded message above, so they get no
        // result and no completion — pairing stays balanced by construction
        // (one result per kept call_id) and replay tolerates the unclosed
        // `ToolCallStarted` brackets the streaming loop emitted for them.
        let mut all_results = pipeline.results;
        let executor_count = all_results.len();
        for (duplicate_id, executor_index) in &plan.duplicates {
            debug_assert!(*executor_index < executor_count);
            let mut result = match all_results.get(*executor_index).cloned() {
                Some(result) => result,
                // Invariant breach (duplicate index into a missing executor
                // result): impossible today — the plan only emits indices it
                // itself allocated — but a release-mode fallback must never
                // silently skip: the duplicate id IS in the recorded message
                // and would break pairing without a result (oracle F4).
                None => ToolResult {
                    timed_out: false,
                    call_id: String::new(),
                    success: false,
                    output: String::new(),
                    error: Some(crate::agent::guard_tool_result_error(
                        "internal: duplicate tool-call rebind found no executor result",
                    )),
                },
            };
            result.call_id = duplicate_id.clone();
            all_results.push(result);
        }
        // Trimmed ids got a `ToolCallStarted` during stream assembly but
        // no completion follows; drop their timing entries so the map
        // cannot accumulate across steps.
        for dropped in &plan.dropped {
            agent.tool_start_times.remove(&dropped.id);
        }
        // Model-visible teaching: when the guardrail trimmed anything
        // (overflow dropped and/or duplicates collapsed), the FIRST kept
        // result carries a one-line note so the model sees the recorded
        // batch was smaller than what it emitted.
        if !plan.dropped.is_empty() || !plan.dedup_groups.is_empty() {
            let dedup_total: usize = plan
                .dedup_groups
                .iter()
                .map(|(_, occurrences)| occurrences.saturating_sub(1))
                .sum();
            if let Some(first) = all_results.first_mut() {
                first.output.push_str(&format!(
                    "\n[nca: tool batch guarded — {} overflow call(s) dropped, {} duplicate call(s) collapsed]",
                    plan.dropped.len(),
                    dedup_total
                ));
            }
        }

        for result in all_results {
            let duration_ms = agent
                .tool_start_times
                .remove(&result.call_id)
                .map(|t| t.elapsed().as_millis() as u64)
                .unwrap_or(0);
            let msg = Message::tool(
                result.call_id.clone(),
                crate::agent::format_tool_result(&result),
            );
            agent.record(&msg).await;
            agent.messages.push(msg);
            agent
                .emit(AgentEvent::ToolCallCompleted {
                    call_id: result.call_id.clone(),
                    output: result,
                    duration_ms,
                })
                .await;
        }

        // ── Guard-refusal escalation ─────────────────────────────────────
        // The messages above already delivered each refusal to the model; this
        // counts stop-bearing STEPS so a model that ignores them ends the
        // turn instead of re-billing full-context inference until the 1024-step
        // budget expires. A clean step (no refusals) resets the streak.
        if pipeline.guard_refusals.is_empty() {
            self.consecutive_guard_stop_steps = 0;
        } else {
            self.consecutive_guard_stop_steps += 1;
            if self.consecutive_guard_stop_steps >= MAX_CONSECUTIVE_GUARD_STOP_STEPS {
                let n_refusals = pipeline.guard_refusals.len();
                let mut msg = format!(
                    "{LOOP_BREAKER_MARKER} Repeat-call guard refused calls in {} consecutive steps \
                     (latest step: {n_refusals} refusal(s)) — you keep re-issuing \
                     identical tool calls that were already answered and refused. \
                     Ending the turn to break the loop.\n\
                     Refused calls in the latest step:",
                    self.consecutive_guard_stop_steps
                );
                for (tool, refusal) in pipeline.guard_refusals.iter().take(3) {
                    msg.push_str(&format!(
                        "\n- `{tool}`: {}",
                        crate::agent::truncate_str(refusal, 200)
                    ));
                }
                if pipeline.guard_refusals.len() > 3 {
                    msg.push_str(&format!(
                        "\n- (and {} more in this step)",
                        pipeline.guard_refusals.len() - 3
                    ));
                }
                msg.push_str(
                    "\nTo proceed: vary the parameters, use a different tool, or rely \
                     on the earlier tool outputs already in your context. If you are \
                     a sub-agent, report this limitation and what you already \
                     established in your final output.",
                );
                agent
                    .emit(AgentEvent::Error {
                        message: msg.clone(),
                    })
                    .await;
                return Ok(StepOutcome::FinalText {
                    text: msg,
                    had_tool_calls: true,
                });
            }
        }

        if self.consecutive_tool_failures >= MAX_CONSECUTIVE_TOOL_FAILURES
            || self.consecutive_varying_failures >= MAX_VARYING_CONSECUTIVE_TOOL_FAILURES
        {
            // N is the FIRING track's count: 3 for verbatim retries, 5 for
            // varying-shape iteration. Message format unchanged — the runtime
            // child-session classification (`starts_with` marker check) and
            // breaker_classification.rs depend on it.
            let firing_count = if self.consecutive_tool_failures >= MAX_CONSECUTIVE_TOOL_FAILURES {
                self.consecutive_tool_failures
            } else {
                self.consecutive_varying_failures
            };
            let detail = self
                .last_failed_error
                .as_deref()
                .map(|e| e.to_string())
                .unwrap_or_else(|| crate::agent::truncate_str(&self.last_failed_output, 300));
            let msg = format!(
                "{LOOP_BREAKER_MARKER} Tool `{}` failed {} times consecutively — stopping to avoid infinite loop.\n\nLast failure detail:\n{}",
                self.last_failed_tool, firing_count, detail
            );
            agent
                .emit(AgentEvent::Error {
                    message: msg.clone(),
                })
                .await;
            return Ok(StepOutcome::FinalText {
                text: msg,
                had_tool_calls: true,
            });
        }

        Ok(StepOutcome::Continue {
            had_tool_calls: true,
        })
    }
}

/// Fingerprint a step's executor batch: hash of each executor's
/// (tool name + canonical JSON input), aggregated over the whole batch so
/// batch-shape changes (a call added/removed) count as "varying" too.
/// Same technique as `tool_guards::canonical_key` (serde_json Maps are
/// BTreeMap-ordered here, so `to_string` is a stable canonical form) but
/// implemented locally — the breaker is per-turn and batch-level, the
/// repeat guard is session-scoped with decay; coupling them would tangle
/// two orthogonal mechanisms.
fn batch_fingerprint(executors: &[ToolCall]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for call in executors {
        call.name.hash(&mut hasher);
        hasher.write_u8(0xFF); // delimiter: name cannot collide with serialized JSON
        if let Ok(canonical) = serde_json::to_string(&call.input) {
            canonical.hash(&mut hasher);
        }
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approval::ApprovalPolicy;
    use crate::provider::Provider;
    use crate::tools::ToolRegistry;
    use nca_common::config::{PermissionConfig, PermissionMode};
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Minimal scripted provider (one round per `chat()` call; rounds are
    /// irrelevant here because the loop-top cancel fires before any call).
    struct NoopProvider {
        calls: AtomicU32,
    }

    #[async_trait::async_trait]
    impl Provider for NoopProvider {
        async fn chat(
            &self,
            _messages: &[Message],
            _tools: &[nca_common::tool::ToolDefinition],
            _model: &str,
            _workspace_root: &Path,
        ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let (_tx, rx) = tokio::sync::mpsc::channel(1);
            Ok(rx)
        }
    }

    // T14 — loop-top cancel branch emits BOTH `Error` and `StepFailed`
    // before returning Err, so the replay bracket never folds rolled-back
    // messages. Driven via `TurnDriver::run` directly because `run_turn`
    // resets the cancel flag at entry (the loop-top branch is unreachable
    // from an external test without racing the mid-stream cancel poll).
    #[tokio::test]
    async fn t14_loop_top_cancel_emits_error_and_step_failed() {
        let provider = Arc::new(NoopProvider {
            calls: AtomicU32::new(0),
        });
        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(64);
        let mut agent = AgentLoop::new(
            Arc::clone(&provider) as Arc<dyn Provider>,
            ToolRegistry::new(),
            ApprovalPolicy::new(PermissionConfig {
                mode: PermissionMode::BypassPermissions,
                ..Default::default()
            }),
            "test-model".into(),
            event_tx,
            10,
            16,
            0,
            None,
        );
        agent.cancel_flag.store(true, Ordering::SeqCst);

        let err = TurnDriver::new(&mut agent, Path::new("."), 1, &[])
            .run(&[])
            .await
            .expect_err("cancelled run must return Err");
        assert_eq!(err.to_string(), "run cancelled");
        assert_eq!(
            provider.calls.load(Ordering::SeqCst),
            0,
            "no provider call may happen"
        );

        let mut saw_error = false;
        let mut step_failed: Option<(u64, u64, u64, String)> = None;
        while let Ok(e) = event_rx.try_recv() {
            match e {
                AgentEvent::Error { message } => {
                    assert_eq!(message, "Run cancelled");
                    saw_error = true;
                }
                AgentEvent::StepFailed {
                    turn_id,
                    step_index,
                    duration_ms,
                    error,
                } => step_failed = Some((turn_id, step_index, duration_ms, error)),
                other => panic!("unexpected event: {other:?}"),
            }
        }
        assert!(saw_error, "cancel branch must emit Error");
        let (turn_id, step_index, duration_ms, error) =
            step_failed.expect("cancel branch must also emit StepFailed");
        assert_eq!((turn_id, step_index, duration_ms), (1, 0, 0));
        assert_eq!(error, "run cancelled");
    }

    /// T15 — a pre-stream provider failure (chat() returns Err immediately,
    /// e.g. DeepSeek HTTP 402 "Insufficient Balance") must emit
    /// `AgentEvent::Error` before `run`'s `StepFailed`, so the failure is
    /// visible to event consumers (TUI transcript, orchestrator LLMs) and
    /// not only in tracing logs.
    #[tokio::test]
    async fn t15_pre_stream_provider_error_emits_error_and_step_failed() {
        struct FailProvider;

        #[async_trait::async_trait]
        impl Provider for FailProvider {
            async fn chat(
                &self,
                _messages: &[Message],
                _tools: &[nca_common::tool::ToolDefinition],
                _model: &str,
                _workspace_root: &Path,
            ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError> {
                Err(ProviderError::RequestFailed(
                    "{\"error\":{\"message\":\"Insufficient Balance\"}}".into(),
                ))
            }
        }

        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(64);
        let mut agent = AgentLoop::new(
            Arc::new(FailProvider),
            ToolRegistry::new(),
            ApprovalPolicy::new(PermissionConfig {
                mode: PermissionMode::BypassPermissions,
                ..Default::default()
            }),
            "test-model".into(),
            event_tx,
            10,
            16,
            0,
            None,
        );

        let err = agent
            .run_turn("do it", Path::new("."), &[])
            .await
            .expect_err("pre-stream provider failure must propagate");
        assert!(
            err.to_string().contains("Insufficient Balance"),
            "got: {err}"
        );

        // Collect events in order; find positions of the Error and StepFailed.
        let mut events = Vec::new();
        while let Ok(e) = event_rx.try_recv() {
            events.push(e);
        }
        let error_pos = events.iter().position(|e| {
            matches!(e, AgentEvent::Error { message } if message.contains("Insufficient Balance"))
        });
        let step_failed_pos = events.iter().position(|e| {
            matches!(e, AgentEvent::StepFailed { turn_id, step_index, error, .. }
                    if *turn_id == 1 && *step_index == 1 && error.contains("Insufficient Balance"))
        });
        assert!(
            error_pos.is_some(),
            "Error event must be emitted: {events:?}"
        );
        let step_failed_pos =
            step_failed_pos.expect("StepFailed with turn_id=1 step_index=1 must be emitted");
        assert!(
            error_pos.unwrap() < step_failed_pos,
            "Error must arrive BEFORE StepFailed"
        );
    }

    // ── Loop-breaker generalization + guard-refusal escalation ────────────

    /// Scripted provider: each `chat()` call replays the next round of
    /// chunks, then closes the channel. Counts calls.
    struct RoundsProvider {
        rounds: Vec<Vec<StreamChunk>>,
        calls: Arc<AtomicU32>,
    }

    impl RoundsProvider {
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
    impl Provider for RoundsProvider {
        async fn chat(
            &self,
            _messages: &[Message],
            _tools: &[nca_common::tool::ToolDefinition],
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

    /// Always-failing tool for the pure-retry breaker test.
    struct BoomTool;
    #[async_trait::async_trait]
    impl crate::tools::ToolExecutor for BoomTool {
        fn definition(&self) -> nca_common::tool::ToolDefinition {
            nca_common::tool::ToolDefinition {
                timeout_ms: None,
                name: "boom".into(),
                description: "always fails".into(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            }
        }
        async fn execute(&self, call: &ToolCall) -> nca_common::tool::ToolResult {
            nca_common::tool::ToolResult {
                timed_out: false,
                call_id: call.id.clone(),
                success: false,
                output: String::new(),
                error: Some("kaboom".into()),
            }
        }
    }

    /// Always-succeeding tool for the guard-escalation test (the guard Stops
    /// on repeat COUNT, not on failure — a succeeding tool still gets
    /// hard-stopped on its 8th identical issue).
    struct OkayTool;
    #[async_trait::async_trait]
    impl crate::tools::ToolExecutor for OkayTool {
        fn definition(&self) -> nca_common::tool::ToolDefinition {
            nca_common::tool::ToolDefinition {
                timeout_ms: None,
                name: "okay".into(),
                description: "always succeeds".into(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            }
        }
        async fn execute(&self, call: &ToolCall) -> nca_common::tool::ToolResult {
            nca_common::tool::ToolResult {
                timed_out: false,
                call_id: call.id.clone(),
                success: true,
                output: "fine".into(),
                error: None,
            }
        }
    }

    fn use_call(id: &str, name: &str, input: serde_json::Value) -> StreamChunk {
        StreamChunk::ToolUse(ToolCall {
            id: id.into(),
            name: name.into(),
            input,
        })
    }

    fn breaker_agent(
        provider: Arc<dyn Provider>,
        tool: Box<dyn crate::tools::ToolExecutor>,
    ) -> AgentLoop {
        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(1024);
        let mut tools = ToolRegistry::new();
        tools.register(tool);
        AgentLoop::new(
            provider,
            tools,
            ApprovalPolicy::new(PermissionConfig {
                mode: PermissionMode::BypassPermissions,
                ..Default::default()
            }),
            "test-model".into(),
            event_tx,
            20,
            16,
            0,
            None,
        )
    }

    /// A step whose ENTIRE batch is the same failing tool must count toward
    /// the consecutive-failure breaker — not just single-call steps. Before
    /// the fix, `tool_calls.len() == 1` meant batched retriers (the common
    /// sub-agent shape, 4–8 calls/step) looped forever.
    #[tokio::test]
    async fn same_tool_batched_failures_trip_the_breaker() {
        // 4 rounds × 2 identical boom calls; breaker must fire on round 3.
        let round = || {
            vec![
                use_call("a", "boom", serde_json::json!({"x": 1})),
                use_call("b", "boom", serde_json::json!({"x": 1})),
                StreamChunk::Finish {
                    reason: "tool_calls".into(),
                },
            ]
        };
        let (provider, calls) = RoundsProvider::new(vec![round(), round(), round(), round()]);
        let mut agent = breaker_agent(Arc::new(provider), Box::new(BoomTool));

        let text = agent
            .run_turn("go", Path::new("."), &[])
            .await
            .expect("breaker ends the turn with a final text, not an error");
        assert!(
            text.starts_with(crate::agent::LOOP_BREAKER_MARKER),
            "loop-breaker final text must carry the classification marker: {text}"
        );
        assert!(
            text.contains("failed 3 times consecutively"),
            "breaker message expected: {text}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 3, "fires on the 3rd step");
    }

    /// A model that keeps re-issuing identical calls past the guard's hard
    /// stop gets its turn force-ended after MAX_CONSECUTIVE_GUARD_STOP_STEPS
    /// consecutive refusal-bearing steps. Pre-fix the refusals were invisible
    /// to the driver and the loop ran to the 1024-step budget.
    #[tokio::test]
    async fn persistent_guard_refusals_end_the_turn() {
        // 10 identical `okay` calls: counts 1–7 execute (hints from 3 on),
        // counts 8–10 are guard Stops = 3 consecutive stop-bearing steps.
        let rounds: Vec<_> = (0..10)
            .map(|i| {
                vec![
                    use_call(&format!("c{i}"), "okay", serde_json::json!({"q": 1})),
                    StreamChunk::Finish {
                        reason: "tool_calls".into(),
                    },
                ]
            })
            .collect();
        let (provider, calls) = RoundsProvider::new(rounds);
        let mut agent = breaker_agent(Arc::new(provider), Box::new(OkayTool));

        let text = agent
            .run_turn("go", Path::new("."), &[])
            .await
            .expect("escalation ends the turn with a final text");
        assert!(
            text.starts_with(crate::agent::LOOP_BREAKER_MARKER),
            "loop-breaker final text must carry the classification marker: {text}"
        );
        assert!(
            text.contains("Repeat-call guard refused calls in 3 consecutive steps"),
            "escalation message expected: {text}"
        );
        assert!(
            text.contains("`okay`"),
            "message must name the offending tool: {text}"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            10,
            "exactly the scripted steps"
        );
    }

    // ── Dual-track breaker: varying vs identical retry forms ─────────────

    /// A tool that fails only when `fail` is true — lets one scripted run
    /// interleave failing and succeeding steps to pin reset semantics.
    struct FlakyTool;
    #[async_trait::async_trait]
    impl crate::tools::ToolExecutor for FlakyTool {
        fn definition(&self) -> nca_common::tool::ToolDefinition {
            nca_common::tool::ToolDefinition {
                timeout_ms: None,
                name: "flaky".into(),
                description: "fails when fail=true".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"fail": {"type": "boolean"}, "v": {"type": "integer"}}
                }),
            }
        }
        async fn execute(&self, call: &ToolCall) -> nca_common::tool::ToolResult {
            let fail = call
                .input
                .get("fail")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if fail {
                nca_common::tool::ToolResult {
                    timed_out: false,
                    call_id: call.id.clone(),
                    success: false,
                    output: "nope".into(),
                    error: Some("flaky failure".into()),
                }
            } else {
                nca_common::tool::ToolResult {
                    timed_out: false,
                    call_id: call.id.clone(),
                    success: true,
                    output: "recovered".into(),
                    error: None,
                }
            }
        }
    }

    /// Varying-shape failures must NOT trip at 3 — the incident's false
    /// kill. Three steps with distinct inputs each produce exactly one
    /// `[nca:loop-breaker-warning]` Error event (no marker) and the turn
    /// continues to a normal final answer.
    #[tokio::test]
    async fn varying_failures_warn_at_three_but_do_not_trip() {
        let fail_step = |i: i32| {
            vec![
                use_call(
                    &format!("v{i}",),
                    "flaky",
                    serde_json::json!({"fail": true, "v": i}),
                ),
                StreamChunk::Finish {
                    reason: "tool_calls".into(),
                },
            ]
        };
        let final_round = vec![
            StreamChunk::TextDelta("still iterating".into()),
            StreamChunk::Finish {
                reason: "stop".into(),
            },
        ];
        let (provider, calls) =
            RoundsProvider::new(vec![fail_step(1), fail_step(2), fail_step(3), final_round]);
        let (mut agent, event_rx) = guardrail_agent(Arc::new(provider), Box::new(FlakyTool));
        let mut event_rx = event_rx;

        let text = agent
            .run_turn("go", Path::new("."), &[])
            .await
            .expect("varying failures must not trip the breaker at 3");
        assert_eq!(text, "still iterating");
        assert_eq!(calls.load(Ordering::SeqCst), 4, "all four steps ran");

        let mut warnings = 0;
        while let Ok(event) = event_rx.try_recv() {
            if let AgentEvent::Error { message } = event {
                assert!(
                    !message.starts_with(crate::agent::LOOP_BREAKER_MARKER),
                    "no hard-break event may fire: {message}"
                );
                if message.starts_with("[nca:loop-breaker-warning]") {
                    warnings += 1;
                    assert!(
                        message.contains("varying"),
                        "warning must explain the varying form: {message}"
                    );
                    assert!(
                        message.contains("`flaky`"),
                        "warning must name the tool: {message}"
                    );
                }
            }
        }
        assert_eq!(warnings, 1, "exactly one warning at count 3");
    }

    /// A successful call at step 4 resets the varying streak: the warning
    /// must not re-fire on the NEXT streak's count 2 (warned flag cleared),
    /// and the streak that follows starts from zero. Steps: fail(1), fail(2),
    /// fail(3) → warning; success → reset; fail(4), fail(5) → count 2, no
    /// second warning; final answer.
    #[tokio::test]
    async fn successful_step_resets_varying_streak_and_warned_flag() {
        let fail_step = |i: i32| {
            vec![
                use_call(
                    &format!("v{i}"),
                    "flaky",
                    serde_json::json!({"fail": true, "v": i}),
                ),
                StreamChunk::Finish {
                    reason: "tool_calls".into(),
                },
            ]
        };
        let ok_step = vec![
            use_call("ok", "flaky", serde_json::json!({"fail": false, "v": 99})),
            StreamChunk::Finish {
                reason: "tool_calls".into(),
            },
        ];
        let final_round = vec![
            StreamChunk::TextDelta("recovered".into()),
            StreamChunk::Finish {
                reason: "stop".into(),
            },
        ];
        let (provider, _calls) = RoundsProvider::new(vec![
            fail_step(1),
            fail_step(2),
            fail_step(3),
            ok_step,
            fail_step(4),
            fail_step(5),
            final_round,
        ]);
        let (mut agent, event_rx) = guardrail_agent(Arc::new(provider), Box::new(FlakyTool));
        let mut event_rx = event_rx;

        let text = agent
            .run_turn("go", Path::new("."), &[])
            .await
            .expect("post-reset streak of 2 must not trip");
        assert_eq!(text, "recovered");

        let mut warnings = 0;
        while let Ok(event) = event_rx.try_recv() {
            if let AgentEvent::Error { message } = event
                && message.starts_with("[nca:loop-breaker-warning]")
            {
                warnings += 1;
            }
        }
        assert_eq!(
            warnings, 1,
            "warning fires once for the first streak; the reset clears the warned flag so the second streak warns again only at its own count 3"
        );
    }

    /// Five consecutive varying-shape failures trip the breaker at 5 — the
    /// raised threshold still contains a truly stuck iteration, and the
    /// breaker text carries the marker with N = the firing track's count.
    #[tokio::test]
    async fn five_varying_failures_trip_the_breaker_at_five() {
        let fail_step = |i: i32| {
            vec![
                use_call(
                    &format!("v{i}"),
                    "flaky",
                    serde_json::json!({"fail": true, "v": i}),
                ),
                StreamChunk::Finish {
                    reason: "tool_calls".into(),
                },
            ]
        };
        let (provider, calls) = RoundsProvider::new(vec![
            fail_step(1),
            fail_step(2),
            fail_step(3),
            fail_step(4),
            fail_step(5),
        ]);
        let mut agent = breaker_agent(Arc::new(provider), Box::new(FlakyTool));

        let text = agent
            .run_turn("go", Path::new("."), &[])
            .await
            .expect("varying breaker ends the turn with final text");
        assert!(
            text.starts_with(crate::agent::LOOP_BREAKER_MARKER),
            "hard break must carry the classification marker: {text}"
        );
        assert!(
            text.contains("failed 5 times consecutively"),
            "N is the firing (varying) track's count: {text}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 5, "fires on the 5th step");
    }

    // ── Batch guardrails: cap + intra-batch dedup (incident 2026-10-02) ──
    /// Tool that records every execution's `n` argument and echoes it in the
    /// output — lets tests count executions and compare fan-out content.
    struct CountingTool {
        executions: Arc<std::sync::Mutex<Vec<i64>>>,
    }

    #[async_trait::async_trait]
    impl crate::tools::ToolExecutor for CountingTool {
        fn definition(&self) -> nca_common::tool::ToolDefinition {
            nca_common::tool::ToolDefinition {
                timeout_ms: None,
                name: "count".into(),
                description: "records executions".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"n": {"type": "integer"}}
                }),
            }
        }
        async fn execute(&self, call: &ToolCall) -> nca_common::tool::ToolResult {
            let n = call.input.get("n").and_then(|v| v.as_i64()).unwrap_or(-1);
            self.executions.lock().expect("executions lock").push(n);
            nca_common::tool::ToolResult {
                timed_out: false,
                call_id: call.id.clone(),
                success: true,
                output: format!("done:{n}"),
                error: None,
            }
        }
    }

    fn guardrail_agent(
        provider: Arc<dyn Provider>,
        tool: Box<dyn crate::tools::ToolExecutor>,
    ) -> (AgentLoop, tokio::sync::mpsc::Receiver<AgentEvent>) {
        let (event_tx, event_rx) = tokio::sync::mpsc::channel(1024);
        let mut tools = ToolRegistry::new();
        tools.register(tool);
        let agent = AgentLoop::new(
            provider,
            tools,
            ApprovalPolicy::new(PermissionConfig {
                mode: PermissionMode::BypassPermissions,
                ..Default::default()
            }),
            "test-model".into(),
            event_tx,
            20,
            16,
            0,
            None,
        );
        (agent, event_rx)
    }

    /// `(tool_call_id, content)` for every tool-result message in the
    /// history, in push order.
    fn tool_message_contents(agent: &AgentLoop) -> Vec<(String, String)> {
        use nca_common::message::{MessageContent, Role};
        agent
            .messages
            .iter()
            .filter(|m| m.role == Role::Tool)
            .map(|m| {
                let content = match &m.content {
                    MessageContent::Text(t) => t.clone(),
                    _ => panic!("expected text tool content"),
                };
                (m.tool_call_id.clone().expect("tool call id"), content)
            })
            .collect()
    }

    /// Tallies from draining a guardrail test's event stream.
    struct ToolEventTally {
        started: usize,
        completed: usize,
        /// One entry per `ToolBatchGuarded` event: (dropped, dedup groups).
        guarded: Vec<(usize, Vec<(String, usize)>)>,
    }

    /// Drain the event stream tallying the tool-call bracket
    /// (`ToolCallStarted`/`ToolCallCompleted`) and every
    /// `ToolBatchGuarded` payload. Note: bracket parity only holds for
    /// KEPT calls — `Started` fires during stream assembly for every
    /// incoming block, while completions cover the kept window only
    /// (replay drops unclosed brackets), so capped batches legitimately
    /// tally started > completed. The guarded payloads assert the
    /// surfaced containment notice.
    fn count_tool_events(event_rx: &mut tokio::sync::mpsc::Receiver<AgentEvent>) -> ToolEventTally {
        let mut tally = ToolEventTally {
            started: 0,
            completed: 0,
            guarded: Vec::new(),
        };
        while let Ok(event) = event_rx.try_recv() {
            match event {
                AgentEvent::ToolCallStarted { .. } => tally.started += 1,
                AgentEvent::ToolCallCompleted { .. } => tally.completed += 1,
                AgentEvent::ToolBatchGuarded {
                    dropped,
                    deduplicated,
                } => tally.guarded.push((dropped, deduplicated)),
                _ => {}
            }
        }
        tally
    }

    #[tokio::test]
    async fn tool_batch_capped_trims_recorded_message_and_drops_overflow() {
        // Incident 2026-10-02 shape, scaled down: one assistant message
        // carrying 9 distinct calls with the cap forced to 4. Plan-before-
        // record: the RECORDED assistant message is trimmed to the 4-call
        // kept window; exactly 4 execute; exactly 4 tool results exist
        // (kept ids only — overflow ids get NO results, NO completions,
        // and never enter history); the first kept result carries the
        // guard note; the turn completes normally.
        let mut round1 = Vec::new();
        for i in 0..9 {
            round1.push(use_call(
                &format!("c{i}"),
                "count",
                serde_json::json!({"n": i}),
            ));
        }
        round1.push(StreamChunk::Finish {
            reason: "tool_calls".into(),
        });
        let round2 = vec![
            StreamChunk::TextDelta("all done".into()),
            StreamChunk::Finish {
                reason: "stop".into(),
            },
        ];
        let (provider, calls) = RoundsProvider::new(vec![round1, round2]);
        let executions = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (mut agent, mut event_rx) = guardrail_agent(
            Arc::new(provider),
            Box::new(CountingTool {
                executions: Arc::clone(&executions),
            }),
        );
        agent.set_max_tool_calls_per_message(4);

        let text = agent
            .run_turn("go", Path::new("."), &[])
            .await
            .expect("turn must complete normally despite the oversized batch");
        assert_eq!(text, "all done");

        assert_eq!(
            executions.lock().expect("executions lock").len(),
            4,
            "exactly cap calls may execute"
        );

        // The RECORDED assistant message is trimmed to the kept window —
        // history never carries the overflow, so the incident's ~206K-token
        // history bomb cannot form.
        use nca_common::message::Role;
        let recorded_ids: Vec<String> = agent
            .messages
            .iter()
            .filter(|m| m.role == Role::Assistant)
            .find_map(|m| m.tool_calls.clone())
            .expect("assistant message with tool_calls")
            .into_iter()
            .map(|c| c.id)
            .collect();
        assert_eq!(
            recorded_ids,
            vec!["c0", "c1", "c2", "c3"],
            "kept window is the emission-order prefix, trimmed to cap"
        );

        // Results exist for KEPT ids only — no synthetic results for the
        // trimmed ids (pairing balanced by construction, not synthesis).
        let results = tool_message_contents(&agent);
        let result_ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(result_ids, vec!["c0", "c1", "c2", "c3"]);
        for (id, content) in &results {
            assert!(content.starts_with("done:"), "{id} executed: {content}");
        }
        // Guard note on the FIRST kept result (5 dropped, no dedup);
        // only that one result carries it.
        assert!(
            results[0].1.contains(
                "[nca: tool batch guarded — 5 overflow call(s) dropped, 0 duplicate call(s) collapsed]"
            ),
            "first kept result carries the guard note: {}",
            results[0].1
        );
        for (id, content) in &results[1..] {
            assert!(
                !content.contains("nca: tool batch guarded"),
                "only the first kept result carries the note ({id}): {content}"
            );
        }

        // Bracket shape: Started fired for all 9 incoming blocks during
        // stream assembly (droppable/lossy class; replay drops unclosed
        // brackets); completions cover the 4 kept calls.
        let tally = count_tool_events(&mut event_rx);
        assert_eq!((tally.started, tally.completed), (9, 4));
        // Exactly ONE surfaced containment notice with the dropped count;
        // all inputs are distinct so there is nothing to dedup.
        assert_eq!(
            tally.guarded,
            vec![(5, Vec::new())],
            "one ToolBatchGuarded: 9 - cap(4) = 5 dropped, no dedup groups"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "turn continued to a final-answer step"
        );
    }

    #[tokio::test]
    async fn duplicate_tool_calls_execute_once_and_fan_out_results() {
        // Same (name, input) ×3 plus one distinct call in ONE batch: two
        // executions total; the duplicates receive the executor's result
        // under their own call_ids; four results overall.
        let round1 = vec![
            use_call("dup1", "count", serde_json::json!({"n": 7})),
            use_call("dup2", "count", serde_json::json!({"n": 7})),
            use_call("dup3", "count", serde_json::json!({"n": 7})),
            use_call("uniq", "count", serde_json::json!({"n": 1})),
            StreamChunk::Finish {
                reason: "tool_calls".into(),
            },
        ];
        let round2 = vec![
            StreamChunk::TextDelta("ok".into()),
            StreamChunk::Finish {
                reason: "stop".into(),
            },
        ];
        let (provider, _calls) = RoundsProvider::new(vec![round1, round2]);
        let executions = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (mut agent, mut event_rx) = guardrail_agent(
            Arc::new(provider),
            Box::new(CountingTool {
                executions: Arc::clone(&executions),
            }),
        );

        let text = agent
            .run_turn("go", Path::new("."), &[])
            .await
            .expect("turn completes normally");
        assert_eq!(text, "ok");

        assert_eq!(
            *executions.lock().expect("executions lock"),
            vec![7, 1],
            "first occurrence + distinct call execute, in emission order"
        );

        let results = tool_message_contents(&agent);
        assert_eq!(results.len(), 4, "one result per emitted call_id");
        let by_id: std::collections::HashMap<String, String> = results.into_iter().collect();
        // dup1 is the first kept call in emission order → carries the guard
        // note suffix (dedup-only batch: nothing dropped, 3 occurrences - 1
        // = 2 collapsed). The rebinds clone the result BEFORE the note is
        // appended, so only dup1 sees it.
        assert_eq!(
            by_id.get("dup1").map(String::as_str),
            Some(
                "done:7\n[nca: tool batch guarded — 0 overflow call(s) dropped, 2 duplicate call(s) collapsed]"
            ),
            "first kept result carries the guard note"
        );
        assert_eq!(by_id.get("dup2").map(String::as_str), Some("done:7"));
        assert_eq!(by_id.get("dup3").map(String::as_str), Some("done:7"));
        assert_eq!(by_id.get("uniq").map(String::as_str), Some("done:1"));

        let tally = count_tool_events(&mut event_rx);
        assert_eq!((tally.started, tally.completed), (4, 4));
        // One notice: nothing dropped (4 ≤ default cap 32), one dedup group
        // `count` ×3 collapsed to a single execution.
        assert_eq!(
            tally.guarded,
            vec![(0, vec![("count".to_string(), 3)])],
            "one ToolBatchGuarded: 0 dropped, count ×3 deduplicated"
        );
    }

    /// A clean (refusal-free) step resets the streak: stop/clean/stop/clean/
    /// stop never escalates, and the turn finishes normally.
    #[tokio::test]
    async fn clean_step_resets_the_guard_stop_streak() {
        let in1 = serde_json::json!({"q": 1});
        let in2 = serde_json::json!({"q": 2});
        // 8× in1 (8th = Stop), in2 (clean), in1 (Stop), in2 (clean), in1
        // (Stop) — streak peaks at 1 with resets, would reach 3 without.
        let mut rounds: Vec<Vec<StreamChunk>> = Vec::new();
        for i in 0..8 {
            rounds.push(vec![
                use_call(&format!("a{i}"), "okay", in1.clone()),
                StreamChunk::Finish {
                    reason: "tool_calls".into(),
                },
            ]);
        }
        for (i, input) in [
            (100, in2.clone()),
            (101, in1.clone()),
            (102, in2),
            (103, in1),
        ] {
            rounds.push(vec![
                use_call(&format!("c{i}"), "okay", input),
                StreamChunk::Finish {
                    reason: "tool_calls".into(),
                },
            ]);
        }
        rounds.push(vec![StreamChunk::TextDelta("all good".into())]);
        let (provider, calls) = RoundsProvider::new(rounds);
        let mut agent = breaker_agent(Arc::new(provider), Box::new(OkayTool));

        let text = agent
            .run_turn("go", Path::new("."), &[])
            .await
            .expect("turn must complete normally");
        assert_eq!(text, "all good", "no escalation may fire: {text}");
        assert_eq!(calls.load(Ordering::SeqCst), 13);
    }

    // ── Chunk 3b: root-cause repro + event classification (incident
    // 2026-10-02, revised timeline: everything through the giant
    // MessageRecorded was written, then ZERO completions — the wedge sat in
    // the post-recording phase under a 7,642-call batch) ──────────────────

    /// Tallies for the flood tests: bracket counters, replay-critical
    /// message classes, and the elision accounting.
    #[derive(Default)]
    struct FloodTally {
        started: usize,
        completed: usize,
        tool_result_messages: usize,
        assistant_tool_call_blocks: usize,
        tokens_streamed: usize,
        elided: u64,
        turn_completed: usize,
        /// Whether an `EventsElided` summary was received BEFORE the giant
        /// assistant `MessageRecorded` (adjacency of gap-notice and the next
        /// guaranteed event).
        elided_before_assistant_record: bool,
    }

    /// Drain the event channel forever, tallying by kind. `lead` is an
    /// optional initial stall (the incident's consumer freeze window).
    fn spawn_drainer(
        mut rx: tokio::sync::mpsc::Receiver<AgentEvent>,
        lead: Option<Duration>,
    ) -> tokio::task::JoinHandle<FloodTally> {
        tokio::spawn(async move {
            if let Some(lead) = lead {
                tokio::time::sleep(lead).await;
            }
            let mut tally = FloodTally::default();
            let mut seen_assistant_record = false;
            while let Some(event) = rx.recv().await {
                match event {
                    AgentEvent::ToolCallStarted { .. } => tally.started += 1,
                    AgentEvent::ToolCallCompleted { .. } => tally.completed += 1,
                    AgentEvent::TokensStreamed { .. } => tally.tokens_streamed += 1,
                    AgentEvent::EventsElided { count } => {
                        tally.elided += count;
                        if !seen_assistant_record {
                            tally.elided_before_assistant_record = true;
                        }
                    }
                    AgentEvent::TurnCompleted { .. } => tally.turn_completed += 1,
                    AgentEvent::MessageRecorded { message } => match message.role {
                        nca_common::message::Role::Tool => tally.tool_result_messages += 1,
                        nca_common::message::Role::Assistant => {
                            if let Some(calls) = message.tool_calls
                                && !seen_assistant_record
                                && !calls.is_empty()
                            {
                                seen_assistant_record = true;
                                tally.assistant_tool_call_blocks = calls.len();
                            }
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
            tally
        })
    }

    /// Agent with explicit event-channel buffer and turn tool budget — the
    /// flood tests need production-shaped buffering (256) and, when the cap
    /// is bypassed, a per-turn budget that doesn't reject the giant batch.
    fn flood_agent(
        provider: Arc<dyn Provider>,
        tool: Box<dyn crate::tools::ToolExecutor>,
        event_buffer: usize,
        max_tool_calls_per_turn: u32,
    ) -> (AgentLoop, tokio::sync::mpsc::Receiver<AgentEvent>) {
        let (event_tx, event_rx) = tokio::sync::mpsc::channel(event_buffer);
        let mut tools = ToolRegistry::new();
        tools.register(tool);
        let agent = AgentLoop::new(
            provider,
            tools,
            ApprovalPolicy::new(PermissionConfig {
                mode: PermissionMode::BypassPermissions,
                ..Default::default()
            }),
            "test-model".into(),
            event_tx,
            20,
            max_tool_calls_per_turn,
            0,
            None,
        );
        (agent, event_rx)
    }

    /// The incident's 8-distinct-paths × ~950-repeats shape, N total
    /// duplicate `read_file`-shaped calls across 8 distinct inputs.
    fn incident_round(n: usize) -> Vec<StreamChunk> {
        let paths = [
            "crates/core/src/agent.rs",
            "crates/core/src/agent_driver.rs",
            "crates/core/src/tool_pipeline.rs",
            "crates/core/src/harness.rs",
            "crates/common/src/event.rs",
            "crates/common/src/config.rs",
            "crates/runtime/src/supervisor.rs",
            "crates/cli/src/stream.rs",
        ];
        let mut round = Vec::with_capacity(n + 1);
        for i in 0..n {
            round.push(use_call(
                &format!("call_{i:05}",),
                "count",
                serde_json::json!({"path": paths[i % paths.len()]}),
            ));
        }
        round.push(StreamChunk::Finish {
            reason: "tool_calls".into(),
        });
        round
    }

    /// ROOT-CAUSE PROBE 1 (pipeline exoneration, drained consumer): the
    /// incident's revised timeline — all Started events + the giant
    /// MessageRecorded WERE written, so the consumer was draining; the
    /// wedge appeared afterwards, inside the post-recording phase. This test
    /// keeps the consumer draining (tight recv loop) and bypasses the cap
    /// (10_000) for a 5,000-call duplicate batch. If the tool pipeline
    /// (repeat-guard bookkeeping, approval scan, join_all execution) had a
    /// super-linear or wedging pathology, this test stalls into the 30 s
    /// timeout. Timings are printed for the design.md root-cause note.
    #[tokio::test]
    async fn giant_duplicate_batch_cap_bypassed_completes_when_events_drain() {
        const N: usize = 5_000;
        let round2 = vec![
            StreamChunk::TextDelta("done".into()),
            StreamChunk::Finish {
                reason: "stop".into(),
            },
        ];
        let (provider, calls) = RoundsProvider::new(vec![incident_round(N), round2]);
        let executions = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (mut agent, event_rx) = flood_agent(
            Arc::new(provider),
            Box::new(CountingTool {
                executions: Arc::clone(&executions),
            }),
            256, // production buffer (supervisor.rs wires mpsc::channel(256))
            16,
        );
        agent.set_max_tool_calls_per_message(10_000); // cap BYPASSED
        let drainer = spawn_drainer(event_rx, None);

        let started_at = Instant::now();
        let text = tokio::time::timeout(
            Duration::from_secs(30),
            agent.run_turn("go", Path::new("."), &[]),
        )
        .await
        .expect("drained-consumer giant batch must NOT wedge (see design.md root-cause note)")
        .expect("turn completes");
        let elapsed = started_at.elapsed();
        eprintln!(
            "root-cause probe 1 (5,000 dup calls, cap bypassed, drained consumer): {elapsed:?}"
        );

        assert_eq!(text, "done");
        assert_eq!(calls.load(Ordering::SeqCst), 2, "turn reached final answer");
        assert_eq!(
            executions.lock().expect("executions lock").len(),
            8,
            "intra-batch dedup collapses the flood to the 8 distinct inputs"
        );
        drop(agent); // release the last event_tx clone so the drainer can see close
        let tally = drainer.await.expect("drainer");
        assert_eq!(
            tally.completed, N,
            "every call_id gets a completion (must-deliver)"
        );
        assert_eq!(
            tally.tool_result_messages, N,
            "replay-critical tool MessageRecorded events: zero loss"
        );
        assert_eq!(tally.assistant_tool_call_blocks, N);
        assert_eq!(tally.turn_completed, 1);
    }

    /// ROOT-CAUSE PROBE 2 (distinct flood): 5,000 DISTINCT calls with the cap
    /// and the per-turn budget bypassed — the worst-case pipeline load
    /// (Phase-1 guard+approval per call, join_all over 5,000 executions,
    /// fan-out assembly). Drained consumer; must complete linearly.
    #[tokio::test]
    async fn giant_distinct_batch_cap_bypassed_completes_when_events_drain() {
        const N: usize = 5_000;
        let mut round1 = Vec::with_capacity(N + 1);
        for i in 0..N {
            round1.push(use_call(
                &format!("call_{i:05}"),
                "count",
                serde_json::json!({"n": i}),
            ));
        }
        round1.push(StreamChunk::Finish {
            reason: "tool_calls".into(),
        });
        let round2 = vec![
            StreamChunk::TextDelta("done".into()),
            StreamChunk::Finish {
                reason: "stop".into(),
            },
        ];
        let (provider, _calls) = RoundsProvider::new(vec![round1, round2]);
        let executions = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (mut agent, event_rx) = flood_agent(
            Arc::new(provider),
            Box::new(CountingTool {
                executions: Arc::clone(&executions),
            }),
            256,
            N as u32 + 16, // per-turn budget also bypassed for the probe
        );
        agent.set_max_tool_calls_per_message(10_000);
        let drainer = spawn_drainer(event_rx, None);

        let started_at = Instant::now();
        let text = tokio::time::timeout(
            Duration::from_secs(30),
            agent.run_turn("go", Path::new("."), &[]),
        )
        .await
        .expect("distinct-flood pipeline must stay linear (drained consumer)")
        .expect("turn completes");
        let elapsed = started_at.elapsed();
        eprintln!("root-cause probe 2 (5,000 distinct calls, drained consumer): {elapsed:?}");

        assert_eq!(text, "done");
        assert_eq!(
            executions.lock().expect("executions lock").len(),
            N,
            "no dedup: every distinct call executes"
        );
        drop(agent); // release the last event_tx clone so the drainer can see close
        let tally = drainer.await.expect("drainer");
        assert_eq!(tally.completed, N);
        assert_eq!(tally.tool_result_messages, N);
    }

    /// THE REGRESSION (PRD acceptance): the incident's stall shape — bounded
    /// event channel under flood with a consumer that stops keeping up. The
    /// turn must still terminate with full must-deliver fidelity; only the
    /// droppable Started notices are elided (and accounted). Pre-fix every
    /// event used a blocking send, so the 5,000-Started flood parked the
    /// stream loop on send #257 until the consumer resumed — and a consumer
    /// that never resumed (the 20-minute incident) wedged the turn with
    /// zero completions and no terminal event, exactly as logged.
    #[tokio::test]
    async fn flood_backpressure_elides_started_but_never_must_deliver() {
        const N: usize = 5_000;
        let round2 = vec![
            StreamChunk::TextDelta("done".into()),
            StreamChunk::Finish {
                reason: "stop".into(),
            },
        ];
        let (provider, _calls) = RoundsProvider::new(vec![incident_round(N), round2]);
        let executions = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (mut agent, event_rx) = flood_agent(
            Arc::new(provider),
            Box::new(CountingTool {
                executions: Arc::clone(&executions),
            }),
            256, // production buffer
            16,
        );
        agent.set_max_tool_calls_per_message(10_000); // cap BYPASSED — isolate the classification fix
        // Consumer freezes for 2 s (mid-flood), then drains — the incident's
        // stall window, bounded so the must-deliver backlog can clear.
        let drainer = spawn_drainer(event_rx, Some(Duration::from_secs(2)));

        let started_at = Instant::now();
        let text = tokio::time::timeout(
            Duration::from_secs(30),
            agent.run_turn("go", Path::new("."), &[]),
        )
        .await
        .expect("turn must terminate despite the consumer stall")
        .expect("turn completes");
        let elapsed = started_at.elapsed();
        eprintln!("regression (5,000 dup calls, 2 s consumer stall, cap bypassed): {elapsed:?}");

        assert_eq!(text, "done");
        assert_eq!(executions.lock().expect("executions lock").len(), 8);
        drop(agent); // release the last event_tx clone so the drainer can see close
        let tally = drainer.await.expect("drainer");

        // Must-deliver fidelity: zero loss, brackets closed.
        assert_eq!(tally.completed, N, "completions are never elided");
        assert_eq!(
            tally.tool_result_messages, N,
            "replay state is never elided"
        );
        assert_eq!(tally.assistant_tool_call_blocks, N);
        assert_eq!(tally.turn_completed, 1, "terminal event delivered");

        // Droppable flood: lossy but fully accounted.
        assert!(
            tally.elided > 0,
            "the stalled window must elide Started notices"
        );
        assert!(
            tally.started < N,
            "classification active: not all 5,000 Started notices were delivered"
        );
        assert!(
            tally.elided_before_assistant_record,
            "EventsElided summary must precede the next guaranteed event"
        );
    }

    /// History-size measurement for the oracle review (out-of-scope design
    /// question: trimming the giant assistant message). Builds the exact
    /// incident shape — 7,642 duplicate tool_use blocks over 8 distinct
    /// paths — and reports serialized bytes + estimated tokens.
    #[test]
    fn giant_assistant_message_size_measurement() {
        const N: usize = 7_642;
        let paths = [
            "crates/core/src/agent.rs",
            "crates/core/src/agent_driver.rs",
            "crates/core/src/tool_pipeline.rs",
            "crates/core/src/harness.rs",
            "crates/common/src/event.rs",
            "crates/common/src/config.rs",
            "crates/runtime/src/supervisor.rs",
            "crates/cli/src/stream.rs",
        ];
        let tool_calls: Vec<MessageToolCall> = (0..N)
            .map(|i| MessageToolCall {
                // Realistic provider id length (OpenAI-style `call_…`).
                id: format!("call_{i:05}_a1b2c3d4e5f6g7h8"),
                name: "read_file".into(),
                arguments: serde_json::json!({"path": paths[i % paths.len()]}),
            })
            .collect();
        let msg = Message::assistant_with_tool_calls("", tool_calls);
        let event = AgentEvent::MessageRecorded { message: msg };
        let bytes = serde_json::to_string(&event).expect("serialize").len();
        let est_tokens = bytes / 4; // same heuristic as KeepaliveSnapshot::est_tokens
        eprintln!(
            "incident giant assistant MessageRecorded: {N} tool_use blocks = {bytes} bytes serialized, ~{est_tokens} est. tokens"
        );
        assert!(
            bytes / N >= 100,
            "sanity: each serialized tool_use block costs ≥100 bytes"
        );
    }
}
