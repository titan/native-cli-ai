//! Tool execution pipeline extracted from AgentLoop.
//!
//! Takes a batch of [`ToolCall`]s, runs permission checks (sequential, because
//! approvals may be interactive), executes approved calls concurrently —
//! except interactive tools (see [`ToolRegistry::is_interactive`]), which run
//! strictly one at a time — and returns ordered results. This isolates the
//! "check → approve → execute" flow from the streaming/parser logic in
//! AgentLoop.

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
///
/// `workspace_root` is included in every hook payload as a top-level
/// `workspace` string so hook scripts can identify the source directory.
#[allow(clippy::too_many_arguments)]
pub async fn run_tool_pipeline(
    tools: &ToolRegistry,
    approval: &mut ApprovalPolicy,
    hooks: &Option<HookRunner>,
    event_tx: &tokio::sync::mpsc::Sender<AgentEvent>,
    cancel_flag: &AtomicBool,
    tool_calls: Vec<ToolCall>,
    repeat_guard: &mut RepeatCallGuard,
    workspace_root: &str,
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
                                "workspace": workspace_root,
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
                                    "workspace": workspace_root,
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
                                "workspace": workspace_root,
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

        // Run tool executions concurrently, EXCEPT interactive tools (tools
        // that block awaiting a human answer, e.g. `ask_question`): those run
        // strictly one at a time as barriers between concurrent spans. Every
        // UI surface tracks a single active question — two simultaneous
        // `QuestionRequested` events would overwrite the first in the UI,
        // orphan its oneshot channel, and freeze the turn forever.
        //
        // Poll cancel_flag every 50 ms so the user can interrupt long-running
        // tools (e.g. cargo build). Each call with a declared `timeout_ms` is
        // additionally wrapped in a cooperative `tokio::time::timeout`
        // (external processes like bash are killed by their own PTY timeout,
        // not by this wrapper).
        let exec_fut = async {
            // Shared per-call future: timeout wrap + repeat-hint append.
            let run_call = |(i, call, guard_hint): (usize, ToolCall, Option<String>)| {
                let call_id = call.id.clone();
                let tx = event_tx.clone();
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
                    (i, res)
                }
            };

            let mut executed: Vec<(usize, ToolResult)> = Vec::with_capacity(to_execute.len());
            let mut span: Vec<(usize, ToolCall, Option<String>)> = Vec::new();
            for item in to_execute {
                if !tools.is_interactive(&item.1.name) {
                    span.push(item);
                    continue;
                }
                // Barrier: drain the concurrent span, then run the interactive
                // call alone so at most one human-blocking question is pending.
                executed
                    .extend(futures_util::future::join_all(span.drain(..).map(&run_call)).await);
                executed.push(run_call(item).await);
            }
            executed.extend(futures_util::future::join_all(span.into_iter().map(run_call)).await);
            executed
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
                        "workspace": workspace_root,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolExecutor;
    use nca_common::tool::ToolDefinition;
    use std::sync::{Arc, Mutex};

    /// Tool that logs "start:<id>"/"end:<id>" into a shared log and blocks on
    /// a zero-permit semaphore until the test releases it.
    struct GatedTool {
        name: String,
        log: Arc<Mutex<Vec<String>>>,
        gate: Arc<tokio::sync::Semaphore>,
    }

    #[async_trait::async_trait]
    impl ToolExecutor for GatedTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                timeout_ms: None,
                name: self.name.clone(),
                description: "gated test tool".into(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            }
        }

        async fn execute(&self, call: &ToolCall) -> ToolResult {
            self.log.lock().unwrap().push(format!("start:{}", call.id));
            // Block until released. `forget` keeps the permit from recycling
            // back into the semaphore when this call returns (a recycled permit
            // would release the NEXT gated call without test consent); a permit
            // added before we reach here is not lost — no missed-wakeup deadlock.
            self.gate.acquire().await.expect("gate closed").forget();
            self.log.lock().unwrap().push(format!("end:{}", call.id));
            ToolResult {
                timed_out: false,
                call_id: call.id.clone(),
                success: true,
                output: format!("done:{}", call.id),
                error: None,
            }
        }
    }

    fn call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: name.into(),
            input: serde_json::json!({}),
        }
    }

    async fn run(tools: ToolRegistry, calls: Vec<ToolCall>) -> Result<PipelineResult, String> {
        let mut config = nca_common::config::NcaConfig::default();
        // Bypass so the fake tool names ("instant", "slow_read", …) never hit
        // the approval Ask tier — only ask_question is on the read allowlist.
        config.permissions.mode = nca_common::config::PermissionMode::BypassPermissions;
        let mut approval = ApprovalPolicy::new(config.permissions);
        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(64);
        let cancel = AtomicBool::new(false);
        let mut guard = RepeatCallGuard::new();
        run_tool_pipeline(
            &tools,
            &mut approval,
            &None,
            &event_tx,
            &cancel,
            calls,
            &mut guard,
            "/ws",
        )
        .await
    }

    /// Two `ask_question` calls in one batch must serialize: the second must
    /// not start until the first completed. Pre-fix, `join_all` ran both at
    /// once, the UI's single active-question slot was overwritten, and the
    /// first question's oneshot was orphaned — freezing the turn forever.
    #[tokio::test]
    async fn interactive_calls_serialize_one_at_a_time() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let gate = Arc::new(tokio::sync::Semaphore::new(0));
        let mut tools = ToolRegistry::new();
        tools.register(Box::new(GatedTool {
            name: "ask_question".into(),
            log: log.clone(),
            gate: gate.clone(),
        }));

        let pipeline = tokio::spawn(async move {
            run(
                tools,
                vec![call("q1", "ask_question"), call("q2", "ask_question")],
            )
            .await
        });

        // First question starts; the second must NOT have started yet.
        // Give the (wrong) concurrent path a moment to misbehave.
        tokio::time::sleep(Duration::from_millis(150)).await;
        {
            let l = log.lock().unwrap();
            assert_eq!(
                &*l,
                &*vec!["start:q1".to_string()],
                "only q1 may run: {l:?}"
            );
        }

        // Release q1 → q2 may start only after q1 ended.
        gate.add_permits(1);
        tokio::time::sleep(Duration::from_millis(150)).await;
        {
            let l = log.lock().unwrap();
            assert_eq!(
                &*l,
                &*vec![
                    "start:q1".to_string(),
                    "end:q1".to_string(),
                    "start:q2".to_string()
                ],
                "q2 must start strictly after q1 completed: {l:?}"
            );
        }

        // Release q2 → pipeline finishes with ordered results.
        gate.add_permits(1);
        let result = tokio::time::timeout(Duration::from_secs(5), pipeline)
            .await
            .expect("pipeline must not hang")
            .expect("join must succeed")
            .expect("pipeline must succeed");
        assert!(result.results.iter().all(|r| r.success));
        assert_eq!(result.results[0].call_id, "q1");
        assert_eq!(result.results[1].call_id, "q2");
    }

    /// Non-interactive tools must keep running concurrently (the original
    /// `join_all` behavior): both enter execute before either can finish.
    #[tokio::test]
    async fn non_interactive_calls_still_run_concurrently() {
        // Two tools that must be in flight simultaneously: each waits at a
        // 2-party barrier, so concurrency is proven by completion at all.
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        struct BarrierTool {
            name: String,
            barrier: Arc<tokio::sync::Barrier>,
        }
        #[async_trait::async_trait]
        impl ToolExecutor for BarrierTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    timeout_ms: None,
                    name: self.name.clone(),
                    description: "barrier test tool".into(),
                    parameters: serde_json::json!({"type": "object", "properties": {}}),
                }
            }
            async fn execute(&self, call: &ToolCall) -> ToolResult {
                self.barrier.wait().await;
                ToolResult {
                    timed_out: false,
                    call_id: call.id.clone(),
                    success: true,
                    output: String::new(),
                    error: None,
                }
            }
        }

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(BarrierTool {
            name: "slow_read".into(),
            barrier: barrier.clone(),
        }));
        tools.register(Box::new(BarrierTool {
            name: "slow_search".into(),
            barrier,
        }));

        let result = tokio::time::timeout(
            Duration::from_secs(5),
            run(
                tools,
                vec![call("a", "slow_read"), call("b", "slow_search")],
            ),
        )
        .await
        .expect("must not hang — tools must overlap")
        .expect("pipeline must succeed");
        assert!(result.results.iter().all(|r| r.success));
        assert_eq!(result.results[0].call_id, "a");
        assert_eq!(result.results[1].call_id, "b");
    }

    /// Mixed batch: interactive barriers must not lose or reorder results.
    #[tokio::test]
    async fn mixed_batch_preserves_result_order() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let gate = Arc::new(tokio::sync::Semaphore::new(0));

        struct InstantTool;
        #[async_trait::async_trait]
        impl ToolExecutor for InstantTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    timeout_ms: None,
                    name: "instant".into(),
                    description: "instant test tool".into(),
                    parameters: serde_json::json!({"type": "object", "properties": {}}),
                }
            }
            async fn execute(&self, call: &ToolCall) -> ToolResult {
                ToolResult {
                    timed_out: false,
                    call_id: call.id.clone(),
                    success: true,
                    output: String::new(),
                    error: None,
                }
            }
        }

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(InstantTool));
        tools.register(Box::new(GatedTool {
            name: "ask_question".into(),
            log: log.clone(),
            gate: gate.clone(),
        }));

        let pipeline = tokio::spawn(async move {
            run(
                tools,
                vec![
                    call("a", "instant"),
                    call("q1", "ask_question"),
                    call("b", "instant"),
                    call("q2", "ask_question"),
                ],
            )
            .await
        });

        // Both questions run one after the other; release as they arrive.
        for _ in 0..2 {
            tokio::time::sleep(Duration::from_millis(150)).await;
            gate.add_permits(1);
        }

        let result = tokio::time::timeout(Duration::from_secs(5), pipeline)
            .await
            .expect("must not hang")
            .expect("join must succeed")
            .expect("pipeline must succeed");
        let ids: Vec<&str> = result.results.iter().map(|r| r.call_id.as_str()).collect();
        assert_eq!(ids, vec!["a", "q1", "b", "q2"]);
        assert!(result.results.iter().all(|r| r.success));
    }
}
