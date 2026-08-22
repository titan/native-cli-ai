use crate::pty::PtyManager;
use nca_common::event::AgentEvent;
use nca_common::tool::{ToolCall, ToolDefinition, ToolResult};
use nca_core::tools::{ToolExecutor, ToolProgress};
use std::sync::Arc;

/// Runtime-backed bash tool that executes shell commands via PTY.
/// Lives in the runtime crate so the supervisor can register it
/// without depending on the CLI crate.
pub struct RuntimeBashTool {
    pty: Arc<PtyManager>,
}

impl RuntimeBashTool {
    pub fn new(pty: Arc<PtyManager>) -> Self {
        Self { pty }
    }
}

#[async_trait::async_trait]
impl ToolExecutor for RuntimeBashTool {
    fn definition(&self) -> ToolDefinition {
        let cwd = self.pty.workspace_root().display().to_string();
        ToolDefinition {
            timeout_ms: None,
            name: "execute_bash".into(),
            description: format!("Execute a shell command in the workspace (cwd: {cwd})"),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute"
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Command timeout in seconds (default: 30)"
                    }
                },
                "required": ["command"]
            }),
        }
    }

    async fn execute(&self, call: &ToolCall) -> ToolResult {
        let (tx, _rx) = tokio::sync::mpsc::channel::<AgentEvent>(8);
        let progress = ToolProgress::new(call.id.clone(), tx);
        self.execute_streaming(call, &progress).await
    }

    async fn execute_streaming(&self, call: &ToolCall, progress: &ToolProgress) -> ToolResult {
        let command = call.input["command"].as_str().unwrap_or("");
        let timeout_secs = call.input["timeout_secs"].as_u64().unwrap_or(30);

        match self
            .pty
            .exec_streaming(command, timeout_secs, progress)
            .await
        {
            Ok(out) => ToolResult {
                timed_out: false,
                call_id: call.id.clone(),
                success: out.exit_code == 0,
                output: if out.stdout.is_empty() {
                    format!("Command exited with status {}", out.exit_code)
                } else {
                    out.stdout
                },
                error: None,
            },
            Err(err) => ToolResult {
                timed_out: false,
                call_id: call.id.clone(),
                success: false,
                output: String::new(),
                error: Some(err.to_string()),
            },
        }
    }
}
