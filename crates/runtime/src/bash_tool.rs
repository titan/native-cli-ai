use crate::pty::PtyManager;
use nca_common::event::AgentEvent;
use nca_common::tool::{ToolCall, ToolDefinition, ToolResult};
use nca_core::tools::{ToolExecutor, ToolProgress};
use std::sync::Arc;

/// Common signal names for diagnostic surfacing. Deliberately not a full
/// table — the goal is giving the model an actionable cause of death
/// ("killed by signal 9 (SIGKILL)" vs a bare `-1`), and uncommon signals
/// fall back to the bare number.
fn signal_name(sig: i32) -> Option<&'static str> {
    match sig {
        6 => Some("SIGABRT"),
        7 => Some("SIGBUS"),
        8 => Some("SIGFPE"),
        9 => Some("SIGKILL"),
        11 => Some("SIGSEGV"),
        13 => Some("SIGPIPE"),
        15 => Some("SIGTERM"),
        _ => None,
    }
}

/// Synthesize a self-diagnosing output line for a failed command that
/// produced no output. Without this the model sees only
/// `Command exited with status -1` and blindly retries the identical call:
/// it cannot tell an OOM SIGKILL from a missing binary, does not know the
/// timeout ceiling that was in effect, or why nothing was captured.
fn synthesize_failure_diagnostics(
    exit_code: i32,
    signal: Option<i32>,
    timeout_secs: u64,
) -> String {
    let mut out = match signal {
        Some(sig) => {
            let name = match signal_name(sig) {
                Some(n) => format!(" ({n})"),
                None => String::new(),
            };
            format!("Command exited with status {exit_code} (killed by signal {sig}{name})")
        }
        None => format!("Command exited with status {exit_code}"),
    };
    out.push_str(&format!(
        "; no output captured — the command may have died before writing anything \
         (signal death, early exit, or lost output); timeout was {timeout_secs}s"
    ));
    out
}

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
            Ok(out) => {
                let success = out.exit_code == 0;
                // Diagnostics for the failed-no-output case go in `error`, NOT
                // `output`: `format_tool_result` (agent.rs) reads only `error`
                // for failed results (falling back to the literal "tool
                // failed"), so `output`-borne diagnostics would never reach
                // the model. The UI path (ToolCallCompleted / stream.rs)
                // already renders error-bearing failures gracefully, so the
                // text is NOT duplicated into both fields.
                let error = if !success && out.stdout.is_empty() {
                    Some(synthesize_failure_diagnostics(
                        out.exit_code,
                        out.signal,
                        timeout_secs,
                    ))
                } else {
                    None
                };
                ToolResult {
                    timed_out: false,
                    call_id: call.id.clone(),
                    success,
                    output: out.stdout,
                    error,
                }
            }
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
