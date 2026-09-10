//! Read-only subagent task introspection tools (P1 of
//! `docs/subagent-task-lifecycle.md`).
//!
//! `task_status` and `task_result` send a [`SubagentControlRequest`] over a
//! bounded mpsc channel to the runtime's `subagent_control_consumer` (which
//! owns the `SubagentRegistry` + `SessionStore` read path) and await the
//! oneshot reply — mirroring the `SpawnSubagentTool` wire pattern exactly.

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
                    SubagentControlRequest::Status { reply, .. } => reply,
                    SubagentControlRequest::Result { reply, .. } => reply,
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
                    SubagentControlRequest::Status { reply, .. } => reply,
                    SubagentControlRequest::Result { reply, .. } => reply,
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
}
