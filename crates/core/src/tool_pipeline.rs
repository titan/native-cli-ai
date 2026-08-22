//! Tool execution pipeline extracted from AgentLoop.
//!
//! Takes a batch of [`ToolCall`]s, runs permission checks (sequential, because
//! approvals may be interactive), executes approved calls concurrently, and
//! returns ordered results. This isolates the "check → approve → execute" flow
//! from the streaming/parser logic in AgentLoop.

use std::sync::atomic::AtomicBool;
use std::time::Duration;

use nca_common::event::AgentEvent;
use nca_common::tool::{PermissionTier, ToolCall, ToolResult};
use serde_json::json;
use tokio::time::MissedTickBehavior;

use crate::approval::{ApprovalPolicy, ApprovalVerdict};
use crate::hooks::{HookEventKind, HookRunner};
use crate::tool_guards::{RepeatAction, RepeatCallGuard};
use crate::tools::ToolRegistry;

/// Outcome of running the tool pipeline on a batch of tool calls.
pub struct PipelineResult {
    /// Ordered tool results (same order as input calls).
    pub results: Vec<ToolResult>,
    /// Events that were emitted during pipeline execution. The caller should
    /// log these if needed but does NOT need to re-emit them — they were
    /// already sent via `event_tx`.
    pub events: Vec<AgentEvent>,
}

/// Run the permission-check / hook / execute pipeline on a batch of tool calls.
///
/// Returns [`PipelineResult`] with ordered results. All events (approval
/// requests/resolutions, tool call started/completed, hooks) are emitted
/// directly via `event_tx` and also collected in `PipelineResult.events`.
///
/// `repeat_guard` is caller-owned and must live for the whole session so
/// repeat detection persists across tool batches, steps, and turns.
pub async fn run_tool_pipeline(
    tools: &ToolRegistry,
    approval: &mut ApprovalPolicy,
    hooks: &Option<HookRunner>,
    event_tx: &tokio::sync::mpsc::Sender<AgentEvent>,
    cancel_flag: &AtomicBool,
    tool_calls: Vec<ToolCall>,
    repeat_guard: &mut RepeatCallGuard,
) -> Result<PipelineResult, String> {
    let mut events = Vec::new();
    let mut emit = |e: AgentEvent| {
        events.push(e.clone());
        // Best-effort send; if the channel is full the event is still recorded.
        let _ = event_tx.try_send(e);
    };

    // ── Phase 1: permission checks (sequential — approvals may be interactive) ──
    enum Ticket {
        Resolved(ToolResult),
        Execute(ToolCall, Option<String>),
    }

    let mut tickets: Vec<Ticket> = Vec::with_capacity(tool_calls.len());

    for call in &tool_calls {
        if cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
            return Err("run cancelled before tool execution".into());
        }

        // Repeat-call guard fires BEFORE the permission check so that even a
        // runaway loop of auto-approved identical calls is escalated. The call
        // is always recorded (even when stopped) so the count keeps climbing.
        let guard_hint = match repeat_guard.record(&call.name, &call.input) {
            RepeatAction::Proceed => None,
            RepeatAction::Hint(msg) | RepeatAction::StrongHint(msg) => Some(msg),
            RepeatAction::Stop(msg) => {
                tickets.push(Ticket::Resolved(ToolResult {
                    call_id: call.id.clone(),
                    success: false,
                    output: String::new(),
                    error: Some(msg),
                    timed_out: false,
                }));
                continue;
            }
        };

        let tier = approval.check(&call.name, &call.input.to_string());

        match tier {
            PermissionTier::Denied => {
                tickets.push(Ticket::Resolved(ToolResult {
                    timed_out: false,
                    call_id: call.id.clone(),
                    success: false,
                    output: String::new(),
                    error: Some(format!("tool `{}` denied by policy", call.name)),
                }));
            }

            PermissionTier::Ask => {
                let description = format!("Tool `{}` requires approval", call.name);
                emit(AgentEvent::ApprovalRequested {
                    call_id: call.id.clone(),
                    tool: call.name.clone(),
                    description: description.clone(),
                });
                if let Some(hooks) = hooks {
                    hooks
                        .run_best_effort(
                            HookEventKind::ApprovalRequested,
                            Some(&call.name),
                            &json!({
                                "call_id": call.id.clone(),
                                "tool": call.name.clone(),
                                "input": call.input.clone(),
                                "description": description,
                            }),
                        )
                        .await;
                }
                let verdict = approval.resolve(call, &description).await;
                let approved = verdict.is_approved();
                let allow_pattern = match &verdict {
                    ApprovalVerdict::AllowPattern(p) => Some(p.clone()),
                    _ => None,
                };
                emit(AgentEvent::ApprovalResolved {
                    call_id: call.id.clone(),
                    approved,
                    allow_pattern: allow_pattern.clone(),
                });
                if let Some(pattern) = allow_pattern {
                    approval.add_session_allow(pattern);
                }

                if approved {
                    let hook_err = match hooks.as_ref() {
                        Some(h) => h
                            .run(
                                HookEventKind::PreToolUse,
                                Some(&call.name),
                                &json!({
                                    "call_id": call.id.clone(),
                                    "tool": call.name.clone(),
                                    "input": call.input.clone(),
                                }),
                            )
                            .await
                            .err(),
                        None => None,
                    };
                    if let Some(reason) = hook_err {
                        tickets.push(Ticket::Resolved(ToolResult {
                            timed_out: false,
                            call_id: call.id.clone(),
                            success: false,
                            output: String::new(),
                            error: Some(reason),
                        }));
                        continue;
                    }
                    tickets.push(Ticket::Execute(call.clone(), guard_hint.clone()));
                } else {
                    if approval.should_fail_on_ask() {
                        let message = format!(
                            "tool `{}` requires approval in headless mode; rerun with a non-interactive permission mode such as `dont-ask` or `bypass-permissions`",
                            call.name
                        );
                        emit(AgentEvent::Error {
                            message: message.clone(),
                        });
                        return Err(message);
                    }
                    tickets.push(Ticket::Resolved(ToolResult {
                        timed_out: false,
                        call_id: call.id.clone(),
                        success: false,
                        output: String::new(),
                        error: Some(format!(
                            "tool `{}` requires approval; request was denied",
                            call.name
                        )),
                    }));
                }
            }

            PermissionTier::Allowed => {
                let hook_err = match hooks.as_ref() {
                    Some(h) => h
                        .run(
                            HookEventKind::PreToolUse,
                            Some(&call.name),
                            &json!({
                                "call_id": call.id.clone(),
                                "tool": call.name.clone(),
                                "input": call.input.clone(),
                            }),
                        )
                        .await
                        .err(),
                    None => None,
                };
                if let Some(reason) = hook_err {
                    tickets.push(Ticket::Resolved(ToolResult {
                        timed_out: false,
                        call_id: call.id.clone(),
                        success: false,
                        output: String::new(),
                        error: Some(reason),
                    }));
                    continue;
                }
                tickets.push(Ticket::Execute(call.clone(), guard_hint.clone()));
            }
        }
    }

    let n = tickets.len();
    let mut results: Vec<Option<ToolResult>> = (0..n).map(|_| None).collect();

    let to_execute: Vec<(usize, ToolCall, Option<String>)> = tickets
        .into_iter()
        .enumerate()
        .filter_map(|(i, t)| match t {
            Ticket::Execute(call, hint) => Some((i, call, hint)),
            Ticket::Resolved(result) => {
                results[i] = Some(result);
                None
            }
        })
        .collect();

    // ── Phase 2: concurrent execution with cancel polling ──────────────
    if !to_execute.is_empty() {
        let mut cancel_poll = tokio::time::interval(Duration::from_millis(50));
        cancel_poll.set_missed_tick_behavior(MissedTickBehavior::Delay);

        // Run tool executions concurrently.  Poll cancel_flag every 50 ms so
        // the user can interrupt long-running tools (e.g. cargo build).
        // Each call with a declared `timeout_ms` is additionally wrapped in a
        // cooperative `tokio::time::timeout` (external processes like bash are
        // killed by their own PTY timeout, not by this wrapper).
        let exec_fut = async {
            let futs = to_execute.iter().map(|(i, call, guard_hint)| {
                let call_id = call.id.clone();
                let tx = event_tx.clone();
                let call = call.clone();
                let guard_hint = guard_hint.clone();
                let timeout_ms = tools.timeout_ms_for(&call.name);
                async move {
                    let progress = crate::tools::ToolProgress::new(call_id, tx);
                    let res = match timeout_ms {
                        Some(ms) => match tokio::time::timeout(
                            std::time::Duration::from_millis(ms),
                            tools.execute_streaming(&call, &progress),
                        )
                        .await
                        {
                            Ok(res) => res,
                            Err(_elapsed) => ToolResult {
                                call_id: call.id.clone(),
                                success: false,
                                output: String::new(),
                                error: Some(format!(
                                    "tool `{}` timed out after {} ms",
                                    call.name, ms
                                )),
                                timed_out: true,
                            },
                        },
                        None => tools.execute_streaming(&call, &progress).await,
                    };
                    // Append the repeat-call hint (if any) to the result output.
                    let res = match guard_hint {
                        Some(hint) if !hint.is_empty() => {
                            let mut res = res;
                            if !res.output.is_empty() {
                                res.output.push('\n');
                            }
                            res.output.push_str(&hint);
                            res
                        }
                        _ => res,
                    };
                    (*i, res)
                }
            });
            futures_util::future::join_all(futs).await
        };

        tokio::pin!(exec_fut);

        let executed: Vec<(usize, ToolResult)> = loop {
            tokio::select! {
                result = exec_fut.as_mut() => break result,
                _ = cancel_poll.tick() => {
                    if cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
                        return Err("run cancelled during tool execution".into());
                    }
                }
            }
        };

        for (i, result) in executed {
            results[i] = Some(result);
        }
    }

    let mut final_results: Vec<ToolResult> = Vec::with_capacity(n);
    for result in results.into_iter().flatten() {
        final_results.push(result);
    }

    // ── Phase 2.5: post-execution hooks ─────────────────────────────────────
    if let Some(hooks) = hooks {
        for result in &final_results {
            let hook_event = if result.success {
                HookEventKind::PostToolUse
            } else {
                HookEventKind::PostToolFailure
            };
            hooks
                .run_best_effort(
                    hook_event,
                    None,
                    &json!({
                        "call_id": result.call_id,
                        "success": result.success,
                        "output": result.output,
                        "error": result.error,
                    }),
                )
                .await;
        }
    }

    Ok(PipelineResult {
        results: final_results,
        events,
    })
}
