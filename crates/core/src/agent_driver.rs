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
use nca_common::tool::ToolCall;
use serde_json::json;

use crate::agent::AgentLoop;
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
    /// Consecutive failures of the same tool — stops infinite retry loops.
    consecutive_tool_failures: u32,
    last_failed_tool: String,
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

        let replay_tool_calls = tool_calls
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

        if tool_calls.len() as u32 > agent.max_tool_calls_per_turn {
            return Err(ProviderError::Other(format!(
                "tool-call budget exceeded in turn {turn} ({} > {})",
                tool_calls.len(),
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
            tool_calls.clone(),
            &mut agent.repeat_guard,
            &self.workspace_root.display().to_string(),
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
        let step_single_tool = tool_calls
            .first()
            .is_some_and(|first| tool_calls.iter().all(|c| c.name == first.name));
        let all_failed_same_tool = !pipeline.results.is_empty()
            && pipeline.results.iter().all(|r| !r.success)
            && step_single_tool;
        if all_failed_same_tool {
            let tool_name = &tool_calls[0].name;
            let last_result = &pipeline.results[0];
            if *tool_name == self.last_failed_tool {
                self.consecutive_tool_failures += 1;
            } else {
                self.last_failed_tool = tool_name.clone();
                self.consecutive_tool_failures = 1;
            }
            // Capture failure details for diagnostics and the final error message.
            self.last_failed_output = last_result.output.clone();
            self.last_failed_error = last_result.error.clone();
            tracing::warn!(
                tool = %tool_name,
                attempt = self.consecutive_tool_failures,
                max_attempts = MAX_CONSECUTIVE_TOOL_FAILURES,
                output = %crate::agent::truncate_str(&last_result.output, 500),
                error = ?last_result.error,
                "consecutive tool failure detected"
            );
        } else {
            self.consecutive_tool_failures = 0;
            self.last_failed_tool.clear();
        }

        for result in pipeline.results {
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
                    "Repeat-call guard refused calls in {} consecutive steps \
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

        if self.consecutive_tool_failures >= MAX_CONSECUTIVE_TOOL_FAILURES {
            let detail = self
                .last_failed_error
                .as_deref()
                .map(|e| e.to_string())
                .unwrap_or_else(|| crate::agent::truncate_str(&self.last_failed_output, 300));
            let msg = format!(
                "Tool `{}` failed {} times consecutively — stopping to avoid infinite loop.\n\nLast failure detail:\n{}",
                self.last_failed_tool, self.consecutive_tool_failures, detail
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
}
