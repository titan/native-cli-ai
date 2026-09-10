//! Read-only subagent task introspection tools (P1) and task-control wire
//! types (P2 chunk A of `docs/subagent-task-lifecycle.md`).
//!
//! Each tool sends a [`SubagentControlRequest`] over a bounded mpsc channel
//! to the runtime's `subagent_control_consumer` (which owns the
//! `SubagentRegistry` + `SessionStore` read path) and awaits the oneshot
//! reply — mirroring the `SpawnSubagentTool` wire pattern exactly.

use nca_common::session::ChildSessionState;
use nca_common::tool::{ToolCall, ToolDefinition, ToolResult};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};

use super::ToolExecutor;

/// Request sent from a control tool to the runtime's control consumer.
#[derive(Debug)]
pub enum SubagentControlRequest {
    /// Read-only live status of a spawned subagent task.
    Status {
        session_id: String,
        reply: oneshot::Sender<SubagentControlResponse>,
    },
    /// Fetch the final assistant output of a finished subagent task.
    Result {
        session_id: String,
        reply: oneshot::Sender<SubagentControlResponse>,
    },
    /// Queue steering text for a RUNNING subagent task; delivered at the
    /// child's next step boundary (P2).
    Message {
        session_id: String,
        text: String,
        reply: oneshot::Sender<SubagentControlResponse>,
    },
    /// Cooperatively cancel a running subagent task, retaining its
    /// worktree/branch for a later revive (P2).
    Cancel {
        session_id: String,
        reason: Option<String>,
        reply: oneshot::Sender<SubagentControlResponse>,
    },
    /// Resume a finished/cancelled subagent task with a new prompt in its
    /// retained session+worktree, bumping its generation (P2).
    Revive {
        session_id: String,
        prompt: String,
        reply: oneshot::Sender<SubagentControlResponse>,
    },
}

/// Reply from the runtime's control consumer. Always JSON-serialized into
/// the tool output so the orchestrator model can read every field.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SubagentControlResponse {
    pub session_id: String,
    pub state: ChildSessionState,
    pub task: Option<String>,
    pub workspace: Option<String>,
    pub branch: Option<String>,
    pub result_summary: Option<String>,
    /// Final assistant message (`task_result` on a terminal task).
    pub output: Option<String>,
    /// Human-readable context when output/state needs explanation.
    pub note: Option<String>,
    /// `false` + `error_message` when the id is unknown to the registry
    /// AND the session store.
    #[serde(default)]
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Revive generation of the child task (`task_revive` bumps it; 0 on a
    /// fresh spawn). Absent on P1 responses and pre-revive replies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<u64>,
}

impl SubagentControlResponse {
    /// Error response for an id unknown to both the registry and the store.
    ///
    /// `state` has no truthful value here; `Failed` is used as a terminal
    /// marker so callers never mistake this for a live task.
    pub fn unknown(session_id: &str, message: impl Into<String>) -> Self {
        Self {
            session_id: session_id.to_string(),
            state: ChildSessionState::Failed,
            task: None,
            workspace: None,
            branch: None,
            result_summary: None,
            output: None,
            note: None,
            ok: false,
            error_message: Some(message.into()),
            generation: None,
        }
    }
}

/// Read-only live status of a spawned subagent task (`task_status`).
pub struct TaskStatusTool {
    control_tx: mpsc::Sender<SubagentControlRequest>,
    timeout: Duration,
}

impl TaskStatusTool {
    pub fn new(control_tx: mpsc::Sender<SubagentControlRequest>, timeout: Duration) -> Self {
        Self {
            control_tx,
            timeout,
        }
    }
}

/// Fetch the final assistant output of a finished subagent task
/// (`task_result`).
pub struct TaskResultTool {
    control_tx: mpsc::Sender<SubagentControlRequest>,
    timeout: Duration,
}

impl TaskResultTool {
    pub fn new(control_tx: mpsc::Sender<SubagentControlRequest>, timeout: Duration) -> Self {
        Self {
            control_tx,
            timeout,
        }
    }
}

/// Queue steering text for a RUNNING subagent task (`task_message`).
pub struct TaskMessageTool {
    control_tx: mpsc::Sender<SubagentControlRequest>,
    timeout: Duration,
}

impl TaskMessageTool {
    pub fn new(control_tx: mpsc::Sender<SubagentControlRequest>, timeout: Duration) -> Self {
        Self {
            control_tx,
            timeout,
        }
    }
}

/// Cooperatively cancel a running subagent task (`task_cancel`).
pub struct TaskCancelTool {
    control_tx: mpsc::Sender<SubagentControlRequest>,
    timeout: Duration,
}

impl TaskCancelTool {
    pub fn new(control_tx: mpsc::Sender<SubagentControlRequest>, timeout: Duration) -> Self {
        Self {
            control_tx,
            timeout,
        }
    }
}

/// Resume a finished/cancelled subagent task with a new prompt
/// (`task_revive`).
pub struct TaskReviveTool {
    control_tx: mpsc::Sender<SubagentControlRequest>,
    timeout: Duration,
}

impl TaskReviveTool {
    pub fn new(control_tx: mpsc::Sender<SubagentControlRequest>, timeout: Duration) -> Self {
        Self {
            control_tx,
            timeout,
        }
    }
}

/// Shared execution core: build the request, await the oneshot reply with a
/// bounded timeout, and shape the `ToolResult` from the response.
async fn execute_control(
    tool_name: &str,
    timeout: Duration,
    control_tx: &mpsc::Sender<SubagentControlRequest>,
    make_request: impl FnOnce(
        String,
        oneshot::Sender<SubagentControlResponse>,
    ) -> SubagentControlRequest,
    call: &ToolCall,
) -> ToolResult {
    let session_id = call.input["session_id"].as_str().unwrap_or("").trim();
    if session_id.is_empty() {
        return ToolResult {
            timed_out: false,
            call_id: call.id.clone(),
            success: false,
            output: String::new(),
            error: Some("session_id is required".into()),
        };
    }
    let session_id = session_id.to_string();

    let (reply_tx, reply_rx) = oneshot::channel();
    let request = make_request(session_id, reply_tx);

    if control_tx.send(request).await.is_err() {
        return ToolResult {
            timed_out: false,
            call_id: call.id.clone(),
            success: false,
            output: String::new(),
            error: Some("Subagent task control is not available".into()),
        };
    }

    match tokio::time::timeout(timeout, reply_rx).await {
        Ok(Ok(response)) => {
            let output = serde_json::to_string_pretty(&response).unwrap_or_default();
            let error = if response.ok {
                None
            } else {
                response
                    .error_message
                    .clone()
                    .or_else(|| response.note.clone())
            };
            ToolResult {
                timed_out: false,
                call_id: call.id.clone(),
                success: response.ok,
                output,
                error,
            }
        }
        Ok(Err(_)) => ToolResult {
            timed_out: false,
            call_id: call.id.clone(),
            success: false,
            output: String::new(),
            error: Some(format!(
                "{tool_name}: subagent control dropped the reply channel"
            )),
        },
        Err(_) => ToolResult {
            timed_out: false,
            call_id: call.id.clone(),
            success: false,
            output: String::new(),
            error: Some(format!(
                "{tool_name} timed out after {}ms",
                timeout.as_millis()
            )),
        },
    }
}

#[async_trait::async_trait]
impl ToolExecutor for TaskStatusTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            timeout_ms: None,
            name: "task_status".into(),
            description: "Read-only live status of a spawned subagent task: lifecycle state \
                (running/completed/cancelled/failed), task text, workspace, branch, and result \
                summary. Use the child session id returned by spawn_subagent. Never modifies \
                the task."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Child session id returned by spawn_subagent."
                    }
                },
                "required": ["session_id"]
            }),
        }
    }

    async fn execute(&self, call: &ToolCall) -> ToolResult {
        execute_control(
            "task_status",
            self.timeout,
            &self.control_tx,
            |session_id, reply| SubagentControlRequest::Status { session_id, reply },
            call,
        )
        .await
    }
}

#[async_trait::async_trait]
impl ToolExecutor for TaskResultTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            timeout_ms: None,
            name: "task_result".into(),
            description: "Fetch the final assistant output of a finished subagent task by \
                session id. Returns the child's last assistant message in full; reports \
                'still running' for live tasks. Read-only."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Child session id returned by spawn_subagent."
                    }
                },
                "required": ["session_id"]
            }),
        }
    }

    async fn execute(&self, call: &ToolCall) -> ToolResult {
        execute_control(
            "task_result",
            self.timeout,
            &self.control_tx,
            |session_id, reply| SubagentControlRequest::Result { session_id, reply },
            call,
        )
        .await
    }
}

#[async_trait::async_trait]
impl ToolExecutor for TaskMessageTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            timeout_ms: None,
            name: "task_message".into(),
            description: "Queue steering text to a RUNNING subagent task without interrupting \
                its current turn; delivered at the child's next step boundary. Use the child \
                session id returned by spawn_subagent."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Child session id returned by spawn_subagent."
                    },
                    "text": {
                        "type": "string",
                        "description": "Steering text to queue for the child."
                    }
                },
                "required": ["session_id", "text"]
            }),
        }
    }

    async fn execute(&self, call: &ToolCall) -> ToolResult {
        let text = call.input["text"].as_str().unwrap_or("").trim().to_string();
        if text.is_empty() {
            return ToolResult {
                timed_out: false,
                call_id: call.id.clone(),
                success: false,
                output: String::new(),
                error: Some("text is required".into()),
            };
        }
        execute_control(
            "task_message",
            self.timeout,
            &self.control_tx,
            |session_id, reply| SubagentControlRequest::Message {
                session_id,
                text,
                reply,
            },
            call,
        )
        .await
    }
}

#[async_trait::async_trait]
impl ToolExecutor for TaskCancelTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            timeout_ms: None,
            name: "task_cancel".into(),
            description: "Cooperatively cancel a running subagent task (abort within ~50ms at \
                the next poll). The child's worktree and branch are RETAINED (never rolled \
                back); the task may be revived afterwards with task_revive."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Child session id returned by spawn_subagent."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional reason recorded with the cancellation."
                    }
                },
                "required": ["session_id"]
            }),
        }
    }

    async fn execute(&self, call: &ToolCall) -> ToolResult {
        let reason = call.input["reason"]
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from);
        execute_control(
            "task_cancel",
            self.timeout,
            &self.control_tx,
            |session_id, reply| SubagentControlRequest::Cancel {
                session_id,
                reason,
                reply,
            },
            call,
        )
        .await
    }
}

#[async_trait::async_trait]
impl ToolExecutor for TaskReviveTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            timeout_ms: None,
            name: "task_revive".into(),
            description: "Resume a finished/cancelled subagent task in its retained \
                session+worktree with a new prompt; the reply carries the child's new final \
                output. Bumping generation."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Child session id returned by spawn_subagent."
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The new prompt to run in the revived child session."
                    }
                },
                "required": ["session_id", "prompt"]
            }),
        }
    }

    async fn execute(&self, call: &ToolCall) -> ToolResult {
        let prompt = call.input["prompt"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string();
        if prompt.is_empty() {
            return ToolResult {
                timed_out: false,
                call_id: call.id.clone(),
                success: false,
                output: String::new(),
                error: Some("prompt is required".into()),
            };
        }
        execute_control(
            "task_revive",
            self.timeout,
            &self.control_tx,
            |session_id, reply| SubagentControlRequest::Revive {
                session_id,
                prompt,
                reply,
            },
            call,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolExecutor;

    fn call(session_id: &str) -> ToolCall {
        ToolCall {
            id: "call-1".into(),
            name: "task_status".into(),
            input: serde_json::json!({ "session_id": session_id }),
        }
    }

    /// A control call with an explicit tool name and arbitrary input.
    fn named_call(tool: &str, input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "call-1".into(),
            name: tool.into(),
            input,
        }
    }

    fn ok_response() -> SubagentControlResponse {
        SubagentControlResponse {
            session_id: "child-1".into(),
            state: ChildSessionState::Completed,
            task: Some("do the thing".into()),
            workspace: Some("/tmp/ws".into()),
            branch: Some("nca/child-1".into()),
            result_summary: Some("done".into()),
            output: Some("all done".into()),
            note: None,
            ok: true,
            error_message: None,
            generation: None,
        }
    }

    /// A realistic `task_revive` success reply: terminal again, new final
    /// output, generation bumped past the fresh-spawn 0.
    fn revived_response() -> SubagentControlResponse {
        SubagentControlResponse {
            session_id: "child-1".into(),
            state: ChildSessionState::Completed,
            task: Some("do the thing".into()),
            workspace: Some("/tmp/ws".into()),
            branch: Some("nca/child-1".into()),
            result_summary: Some("done twice".into()),
            output: Some("revived final output".into()),
            note: None,
            ok: true,
            error_message: None,
            generation: Some(2),
        }
    }

    fn unknown_response() -> SubagentControlResponse {
        SubagentControlResponse::unknown("child-404", "unknown subagent task id")
    }

    /// Responder that replies to the next control request.
    fn control_responder(
        mut rx: mpsc::Receiver<SubagentControlRequest>,
        response: SubagentControlResponse,
    ) {
        tokio::spawn(async move {
            if let Some(req) = rx.recv().await {
                let reply = match req {
                    SubagentControlRequest::Status { reply, .. }
                    | SubagentControlRequest::Result { reply, .. }
                    | SubagentControlRequest::Message { reply, .. }
                    | SubagentControlRequest::Cancel { reply, .. }
                    | SubagentControlRequest::Revive { reply, .. } => reply,
                };
                let _ = reply.send(response);
            }
        });
    }

    #[tokio::test]
    async fn task_status_ok_response_is_success_with_state_json() {
        let (tx, rx) = mpsc::channel(1);
        control_responder(rx, ok_response());
        let tool = TaskStatusTool::new(tx, Duration::from_secs(5));
        let result = tool.execute(&call("child-1")).await;
        assert!(result.success);
        assert!(result.error.is_none());
        assert!(
            result.output.contains("\"state\": \"completed\""),
            "output must carry the state: {}",
            result.output
        );
        assert!(result.output.contains("all done"));
    }

    #[tokio::test]
    async fn task_result_ok_response_is_success() {
        let (tx, rx) = mpsc::channel(1);
        control_responder(rx, ok_response());
        let tool = TaskResultTool::new(tx, Duration::from_secs(5));
        let result = tool.execute(&call("child-1")).await;
        assert!(result.success);
        assert!(result.output.contains("\"output\": \"all done\""));
    }

    #[tokio::test]
    async fn unknown_id_surfaces_error_message() {
        let (tx, rx) = mpsc::channel(1);
        control_responder(rx, unknown_response());
        let tool = TaskStatusTool::new(tx, Duration::from_secs(5));
        let result = tool.execute(&call("child-404")).await;
        assert!(!result.success);
        let error = result.error.expect("error must be set when !ok");
        assert!(error.contains("unknown subagent task id"), "got: {error}");
        // The full response still rides in output for model context.
        assert!(result.output.contains("child-404"));
    }

    #[tokio::test]
    async fn send_failure_reports_control_unavailable() {
        // Drop the receiver so `send` fails immediately.
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        let tool = TaskStatusTool::new(tx, Duration::from_secs(5));
        let result = tool.execute(&call("child-1")).await;
        assert!(!result.success);
        assert_eq!(
            result.error.as_deref(),
            Some("Subagent task control is not available")
        );
    }

    #[tokio::test]
    async fn reply_channel_drop_is_reported() {
        let (tx, mut rx) = mpsc::channel(1);
        // Receive the request but drop the reply sender without answering.
        tokio::spawn(async move {
            if let Some(req) = rx.recv().await {
                let reply = match req {
                    SubagentControlRequest::Status { reply, .. }
                    | SubagentControlRequest::Result { reply, .. }
                    | SubagentControlRequest::Message { reply, .. }
                    | SubagentControlRequest::Cancel { reply, .. }
                    | SubagentControlRequest::Revive { reply, .. } => reply,
                };
                drop(reply);
            }
        });
        let tool = TaskStatusTool::new(tx, Duration::from_secs(5));
        let result = tool.execute(&call("child-1")).await;
        assert!(!result.success);
        let error = result.error.expect("error set");
        assert!(error.contains("dropped the reply channel"), "got: {error}");
    }

    #[tokio::test]
    async fn slow_reply_times_out_with_bounded_error() {
        let (tx, mut rx) = mpsc::channel(1);
        // Receive the request, hold the reply, never answer.
        tokio::spawn(async move {
            let _held = rx.recv().await;
            tokio::time::sleep(Duration::from_secs(10)).await;
        });
        let tool = TaskResultTool::new(tx, Duration::from_millis(10));
        let result = tool.execute(&call("child-1")).await;
        assert!(!result.success);
        let error = result.error.expect("error set");
        assert!(error.contains("timed out"), "got: {error}");
    }

    #[tokio::test]
    async fn missing_session_id_fails_fast() {
        let (tx, _rx) = mpsc::channel(1);
        let status = TaskStatusTool::new(tx.clone(), Duration::from_secs(5));
        let result = status.execute(&call("")).await;
        assert!(!result.success);
        assert_eq!(result.error.as_deref(), Some("session_id is required"));

        let result_tool = TaskResultTool::new(tx, Duration::from_secs(5));
        let result = result_tool.execute(&call("  ")).await;
        assert!(!result.success);
        assert_eq!(result.error.as_deref(), Some("session_id is required"));
    }

    #[test]
    fn unknown_response_constructor_shape() {
        let resp = SubagentControlResponse::unknown("x", "nope");
        assert!(!resp.ok);
        assert_eq!(resp.error_message.as_deref(), Some("nope"));
        assert!(resp.state.is_terminal());
        assert!(resp.output.is_none());
        // JSON round-trips (the wire shape models read).
        let json = serde_json::to_string(&resp).expect("serialize");
        let back: SubagentControlResponse = serde_json::from_str(&json).expect("deserialize");
        assert!(!back.ok);
        assert_eq!(back.error_message.as_deref(), Some("nope"));
    }

    // ------------------------------------------------------------------
    // task_message
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn task_message_ok_response_is_success() {
        let (tx, rx) = mpsc::channel(1);
        control_responder(rx, ok_response());
        let tool = TaskMessageTool::new(tx, Duration::from_secs(5));
        let result = tool
            .execute(&named_call(
                "task_message",
                serde_json::json!({ "session_id": "child-1", "text": "pivot to tests" }),
            ))
            .await;
        assert!(result.success);
        assert!(result.error.is_none());
        assert!(
            result.output.contains("\"state\": \"completed\""),
            "output must carry the state: {}",
            result.output
        );
    }

    #[tokio::test]
    async fn task_message_missing_or_blank_text_fails_fast() {
        let (tx, _rx) = mpsc::channel(1);
        let tool = TaskMessageTool::new(tx, Duration::from_secs(5));
        for text in [
            serde_json::Value::Null,
            serde_json::json!(""),
            serde_json::json!("   "),
        ] {
            let result = tool
                .execute(&named_call(
                    "task_message",
                    serde_json::json!({ "session_id": "child-1", "text": text }),
                ))
                .await;
            assert!(!result.success, "blank text must fail: {text:?}");
            assert_eq!(result.error.as_deref(), Some("text is required"));
        }
    }

    // ------------------------------------------------------------------
    // task_cancel
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn task_cancel_ok_response_carries_session_id_and_reason() {
        let (tx, mut rx) = mpsc::channel(1);
        // Responder asserts the request shape, then replies ok.
        tokio::spawn(async move {
            if let Some(req) = rx.recv().await {
                match req {
                    SubagentControlRequest::Cancel {
                        session_id,
                        reason,
                        reply,
                    } => {
                        assert_eq!(session_id, "child-1");
                        assert_eq!(reason.as_deref(), Some("wrong branch"));
                        let _ = reply.send(ok_response());
                    }
                    _ => panic!("expected a Cancel request"),
                }
            }
        });
        let tool = TaskCancelTool::new(tx, Duration::from_secs(5));
        let result = tool
            .execute(&named_call(
                "task_cancel",
                serde_json::json!({ "session_id": "child-1", "reason": "wrong branch" }),
            ))
            .await;
        assert!(result.success);
        assert!(result.error.is_none());
    }

    #[tokio::test]
    async fn task_cancel_unknown_id_surfaces_error_message() {
        let (tx, rx) = mpsc::channel(1);
        control_responder(rx, unknown_response());
        let tool = TaskCancelTool::new(tx, Duration::from_secs(5));
        let result = tool
            .execute(&named_call(
                "task_cancel",
                serde_json::json!({ "session_id": "child-404" }),
            ))
            .await;
        assert!(!result.success);
        let error = result.error.expect("error must be set when !ok");
        assert!(error.contains("unknown subagent task id"), "got: {error}");
        assert!(result.output.contains("child-404"));
    }

    #[tokio::test]
    async fn task_cancel_slow_reply_times_out_with_bounded_error() {
        let (tx, mut rx) = mpsc::channel(1);
        tokio::spawn(async move {
            let _held = rx.recv().await;
            tokio::time::sleep(Duration::from_secs(10)).await;
        });
        let tool = TaskCancelTool::new(tx, Duration::from_millis(10));
        let result = tool
            .execute(&named_call(
                "task_cancel",
                serde_json::json!({ "session_id": "child-1" }),
            ))
            .await;
        assert!(!result.success);
        let error = result.error.expect("error set");
        assert!(error.contains("timed out"), "got: {error}");
    }

    // ------------------------------------------------------------------
    // task_revive
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn task_revive_ok_response_carries_generation_in_output() {
        let (tx, rx) = mpsc::channel(1);
        control_responder(rx, revived_response());
        let tool = TaskReviveTool::new(tx, Duration::from_secs(5));
        let result = tool
            .execute(&named_call(
                "task_revive",
                serde_json::json!({ "session_id": "child-1", "prompt": "finish the tests" }),
            ))
            .await;
        assert!(result.success);
        assert!(result.error.is_none());
        assert!(
            result.output.contains("\"generation\": 2"),
            "the bumped generation must ride the output JSON: {}",
            result.output
        );
        assert!(result.output.contains("revived final output"));
    }

    #[tokio::test]
    async fn task_revive_missing_or_blank_prompt_fails_fast() {
        let (tx, _rx) = mpsc::channel(1);
        let tool = TaskReviveTool::new(tx, Duration::from_secs(5));
        for prompt in [
            serde_json::Value::Null,
            serde_json::json!(""),
            serde_json::json!("  "),
        ] {
            let result = tool
                .execute(&named_call(
                    "task_revive",
                    serde_json::json!({ "session_id": "child-1", "prompt": prompt }),
                ))
                .await;
            assert!(!result.success, "blank prompt must fail: {prompt:?}");
            assert_eq!(result.error.as_deref(), Some("prompt is required"));
        }
    }

    #[tokio::test]
    async fn task_revive_reply_channel_drop_is_reported() {
        let (tx, mut rx) = mpsc::channel(1);
        tokio::spawn(async move {
            if let Some(req) = rx.recv().await {
                let reply = match req {
                    SubagentControlRequest::Status { reply, .. }
                    | SubagentControlRequest::Result { reply, .. }
                    | SubagentControlRequest::Message { reply, .. }
                    | SubagentControlRequest::Cancel { reply, .. }
                    | SubagentControlRequest::Revive { reply, .. } => reply,
                };
                drop(reply);
            }
        });
        let tool = TaskReviveTool::new(tx, Duration::from_secs(5));
        let result = tool
            .execute(&named_call(
                "task_revive",
                serde_json::json!({ "session_id": "child-1", "prompt": "again" }),
            ))
            .await;
        assert!(!result.success);
        let error = result.error.expect("error set");
        assert!(error.contains("dropped the reply channel"), "got: {error}");
    }

    // ------------------------------------------------------------------
    // shared wire shape
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn new_tools_missing_session_id_fail_fast() {
        let (tx, _rx) = mpsc::channel(1);
        let message = TaskMessageTool::new(tx.clone(), Duration::from_secs(5));
        let result = message
            .execute(&named_call(
                "task_message",
                serde_json::json!({ "text": "steer" }),
            ))
            .await;
        assert!(!result.success);
        assert_eq!(result.error.as_deref(), Some("session_id is required"));

        let cancel = TaskCancelTool::new(tx.clone(), Duration::from_secs(5));
        let result = cancel
            .execute(&named_call(
                "task_cancel",
                serde_json::json!({ "reason": "stop" }),
            ))
            .await;
        assert_eq!(result.error.as_deref(), Some("session_id is required"));

        let revive = TaskReviveTool::new(tx, Duration::from_secs(5));
        let result = revive
            .execute(&named_call(
                "task_revive",
                serde_json::json!({ "prompt": "again" }),
            ))
            .await;
        assert_eq!(result.error.as_deref(), Some("session_id is required"));
    }

    #[test]
    fn generation_round_trips_and_skips_when_none() {
        // None → key absent: P1 responses keep their exact wire shape.
        let json = serde_json::to_string(&ok_response()).expect("serialize");
        assert!(!json.contains("generation"), "must skip when None: {json}");

        // Some(2) → present and round-trips.
        let json = serde_json::to_string(&revived_response()).expect("serialize");
        assert!(
            json.contains("\"generation\":2"),
            "must serialize when set: {json}"
        );
        let back: SubagentControlResponse = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.generation, Some(2));

        // P1 JSON (no generation key) still deserializes → None.
        let p1 = serde_json::json!({
            "session_id": "child-1",
            "state": "running",
            "ok": true
        });
        let back: SubagentControlResponse =
            serde_json::from_value(p1).expect("P1 JSON must stay deserializable");
        assert_eq!(back.generation, None);
    }
}
