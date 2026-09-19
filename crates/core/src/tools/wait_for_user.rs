//! `wait_for_user` (P4): non-blocking turn-end signal for handing control
//! back to the user.
//!
//! Distinct from `ask_question` by construction: no options, no oneshot, no
//! `QuestionRequested` event, no event channel at all. Its only side effect
//! is the injected `pause_hook` (the CLI wires it to
//! `WakeScheduler::pause`), which defers background-subagent wakes until
//! the user's next Submit (a terminal landing while paused is held and
//! delivered right after that Submit, never dropped). Registered
//! `is_interactive` so the tool pipeline runs it strictly alone, last in a
//! batch. Same-turn repeats are guarded separately: see
//! `crate::tool_guards` (warn on the 2nd call in a turn, refuse from the 3rd).

use std::sync::Arc;

use nca_common::tool::{ToolCall, ToolDefinition, ToolResult};

use super::ToolExecutor;

/// Turn-end HITL signal: pause wake delivery and return immediately.
///
/// The hook type mirrors [`crate::provider`] closure conventions: cheap,
/// non-blocking, never panicking. `None` leaves the tool a pure signal
/// (registration sites without a wake scheduler).
pub type PauseHook = Arc<dyn Fn() + Send + Sync>;

/// Tool that hands control back to the user without opening a question.
pub struct WaitForUserTool {
    pause_hook: Option<PauseHook>,
}

impl WaitForUserTool {
    /// Build the tool. `pause_hook` is invoked once per execute (the wake
    /// scheduler's `pause()`); pass `None` when there is no scheduler to
    /// pause — the end-turn signal alone stays meaningful.
    pub fn new(pause_hook: Option<PauseHook>) -> Self {
        Self { pause_hook }
    }
}

#[async_trait::async_trait]
impl ToolExecutor for WaitForUserTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            timeout_ms: None,
            name: "wait_for_user".into(),
            description: "Hand control back to the user and end your turn. Call this ONLY \
                when you are waiting on the USER — a decision, input, review, or approval \
                you need to continue. Do NOT call it when waiting on background subagent \
                tasks: for those, simply end your turn; you will be woken automatically \
                when they complete. While you stand by, background task wakes are held \
                and delivered right after the user's next message. Call it last, alone — never \
                alongside other tool calls in the same batch. It returns immediately, \
                never prompts, and never blocks."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        }
    }

    async fn execute(&self, call: &ToolCall) -> ToolResult {
        if let Some(pause) = &self.pause_hook {
            pause();
        }
        ToolResult {
            timed_out: false,
            call_id: call.id.clone(),
            success: true,
            output: "Standing by for the user. End your turn now; background task wakes \
                are held and will be delivered right after the user's next message."
                .into(),
            error: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn call() -> ToolCall {
        ToolCall {
            id: "call-1".into(),
            name: "wait_for_user".into(),
            input: serde_json::json!({}),
        }
    }

    #[tokio::test]
    async fn executes_immediately_with_expected_output_and_invokes_hook() {
        let invoked = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&invoked);
        let tool = WaitForUserTool::new(Some(Arc::new(move || {
            flag.store(true, Ordering::SeqCst);
        })));

        let res = tool.execute(&call()).await;
        assert!(res.success, "error: {:?}", res.error);
        assert!(res.error.is_none());
        assert!(
            res.output.starts_with("Standing by for the user."),
            "output: {}",
            res.output
        );
        assert!(
            res.output
                .contains("delivered right after the user's next message"),
            "output must state the deferral policy: {}",
            res.output
        );
        assert!(
            invoked.load(Ordering::SeqCst),
            "pause hook must be invoked exactly here"
        );
    }

    #[tokio::test]
    async fn none_hook_still_succeeds() {
        let tool = WaitForUserTool::new(None);
        let res = tool.execute(&call()).await;
        assert!(res.success, "error: {:?}", res.error);
        assert!(res.output.contains("Standing by for the user."));
    }

    // (c) Never emits any event, by construction: `WaitForUserTool` has no
    // `event_tx` field (or any channel), so there is nothing to assert at
    // runtime — the type cannot emit. What CAN regress is blocking: the
    // future must complete on its FIRST poll without awaiting anything
    // (the hook included), so `now_or_never` succeeding pins that there is
    // no await-point between call and result.
    #[tokio::test]
    async fn future_completes_on_first_poll_no_await_points() {
        let tool = WaitForUserTool::new(Some(Arc::new(|| {})));
        let res = futures_util::future::FutureExt::now_or_never(tool.execute(&call()))
            .expect("execute must not await — completes on the first poll");
        assert!(res.success);
    }
}
