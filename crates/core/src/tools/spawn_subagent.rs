use nca_common::config::ProviderKind;
use nca_common::tool::{ToolCall, ToolDefinition, ToolResult};
use tokio::sync::{mpsc, oneshot};

use super::ToolExecutor;

/// Request sent from the tool to the runtime to spawn a child session.
#[derive(Debug)]
pub struct SpawnRequest {
    pub task: String,
    pub focus_files: Vec<String>,
    pub use_worktree: bool,
    pub provider_override: Option<ProviderKind>,
    pub model_override: Option<String>,
    /// Optional specialist agent name (e.g. "explorer", "oracle").
    /// When set, the runtime loads the matching agent profile.
    pub specialist: Option<String>,
    pub reply: oneshot::Sender<SpawnResponse>,
}

/// Response from the runtime after spawning a child session.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpawnResponse {
    pub child_session_id: String,
    pub status: String,
    pub output: String,
    pub workspace: String,
    pub branch: Option<String>,
    pub worktree_path: Option<String>,
}

pub struct SpawnSubagentTool {
    spawn_tx: mpsc::Sender<SpawnRequest>,
}

impl SpawnSubagentTool {
    pub fn new(spawn_tx: mpsc::Sender<SpawnRequest>) -> Self {
        Self { spawn_tx }
    }
}

#[async_trait::async_trait]
impl ToolExecutor for SpawnSubagentTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            timeout_ms: None,
            name: "spawn_subagent".into(),
            description: "Spawn a sub-agent that runs as a separate session to handle a specific \
                task in parallel. The sub-agent inherits your conversation context and workspace. \
                Use this to delegate independent tasks (e.g. creating files, running builds) \
                to child agents that work in isolated git worktrees."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "A clear, self-contained description of what the sub-agent should do."
                    },
                    "focus_files": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional list of file paths the sub-agent should focus on."
                    },
                    "use_worktree": {
                        "type": "boolean",
                        "description": "If true, the sub-agent runs in an isolated git worktree branch. Defaults to true."
                    },
                    "provider": {
                        "type": "string",
                        "description": "Optional provider override. Only used when `specialist` is NOT set — a specialist profile's provider is authoritative and takes precedence over this. Use one of the configured provider names."
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model name override. Only used when `specialist` is NOT set — a specialist profile's model is authoritative and takes precedence over this."
                    },
                    "specialist": {
                        "type": "string",
                        "description": "Optional specialist agent name (e.g. 'explorer', 'oracle', 'librarian', 'fixer'). When set, the runtime loads the matching agent profile (provider, model, system prompt) automatically. Do NOT also pass `provider` or `model` — the profile's routing is authoritative and any such override is ignored."
                    }
                },
                "required": ["task"]
            }),
        }
    }

    async fn execute(&self, call: &ToolCall) -> ToolResult {
        let task = call.input["task"].as_str().unwrap_or("").to_string();

        if task.is_empty() {
            return ToolResult {
                timed_out: false,
                call_id: call.id.clone(),
                success: false,
                output: String::new(),
                error: Some("task is required".into()),
            };
        }

        let focus_files: Vec<String> = call.input["focus_files"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let use_worktree = call.input["use_worktree"].as_bool().unwrap_or(true);

        // Parse optional provider/model overrides for per-agent routing.
        let provider_override = call.input["provider"]
            .as_str()
            .and_then(ProviderKind::from_cli_name);
        let model_override = call.input["model"].as_str().map(String::from);
        let specialist = call.input["specialist"]
            .as_str()
            .filter(|s| !s.trim().is_empty())
            .map(String::from);

        let (reply_tx, reply_rx) = oneshot::channel();

        let req = SpawnRequest {
            task,
            focus_files,
            use_worktree,
            provider_override,
            model_override,
            specialist,
            reply: reply_tx,
        };

        if self.spawn_tx.send(req).await.is_err() {
            return ToolResult {
                timed_out: false,
                call_id: call.id.clone(),
                success: false,
                output: String::new(),
                error: Some("Sub-agent spawner is not available".into()),
            };
        }

        match tokio::time::timeout(std::time::Duration::from_secs(600), reply_rx).await {
            Ok(Ok(response)) => {
                let output = serde_json::to_string_pretty(&response).unwrap_or_default();
                let success = response.status == "completed";
                ToolResult {
                    timed_out: false,
                    call_id: call.id.clone(),
                    success,
                    output,
                    error: if success {
                        None
                    } else {
                        // Surface the child's actual failure reason (e.g. the
                        // provider error like "API request failed: …Insufficient
                        // Balance…") in `error` — orchestrator LLMs reading the
                        // spawn tool result key off `error`, not `output`.
                        let reason = crate::agent::truncate_str(response.output.trim(), 300);
                        if reason.is_empty() {
                            Some(format!(
                                "Sub-agent finished with status: {}",
                                response.status
                            ))
                        } else {
                            Some(format!(
                                "Sub-agent finished with status: {} — {}",
                                response.status, reason
                            ))
                        }
                    },
                }
            }
            Ok(Err(_)) => ToolResult {
                timed_out: false,
                call_id: call.id.clone(),
                success: false,
                output: String::new(),
                error: Some("Sub-agent spawner dropped the reply channel".into()),
            },
            Err(_) => ToolResult {
                timed_out: false,
                call_id: call.id.clone(),
                success: false,
                output: String::new(),
                error: Some("Sub-agent timed out after 600 seconds".into()),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolExecutor;

    fn tool_call() -> ToolCall {
        ToolCall {
            id: "call-1".into(),
            name: "spawn_subagent".into(),
            input: serde_json::json!({ "task": "do the thing" }),
        }
    }

    /// Spawn a responder that replies to the next `SpawnRequest` with the
    /// given status/output.
    fn spawn_responder(mut spawn_rx: mpsc::Receiver<SpawnRequest>, response: SpawnResponse) {
        tokio::spawn(async move {
            if let Some(req) = spawn_rx.recv().await {
                let _ = req.reply.send(response);
            }
        });
    }

    fn response(status: &str, output: &str) -> SpawnResponse {
        SpawnResponse {
            child_session_id: "sess-1".into(),
            status: status.into(),
            output: output.into(),
            workspace: "/tmp/ws".into(),
            branch: None,
            worktree_path: None,
        }
    }

    #[tokio::test]
    async fn error_field_carries_child_failure_reason() {
        let (spawn_tx, spawn_rx) = mpsc::channel(1);
        spawn_responder(
            spawn_rx,
            response(
                "error",
                "API request failed: {\"error\":{\"message\":\"Insufficient Balance\"}}",
            ),
        );
        let tool = SpawnSubagentTool::new(spawn_tx);
        let result = tool.execute(&tool_call()).await;
        assert!(!result.success);
        let error = result.error.expect("error must be set on failure");
        assert!(error.contains("error"), "status must appear: {error}");
        assert!(
            error.contains("Insufficient Balance"),
            "child reason must appear: {error}"
        );
    }

    #[tokio::test]
    async fn empty_output_keeps_status_only_and_long_output_is_truncated() {
        // Empty output: status-only message.
        let (spawn_tx, spawn_rx) = mpsc::channel(1);
        spawn_responder(spawn_rx, response("error", "   "));
        let tool = SpawnSubagentTool::new(spawn_tx);
        let result = tool.execute(&tool_call()).await;
        assert_eq!(
            result.error.as_deref(),
            Some("Sub-agent finished with status: error")
        );

        // 5000-char output: error stays within the 300-char truncation bound
        // (plus the fixed prefix).
        let (spawn_tx, spawn_rx) = mpsc::channel(1);
        spawn_responder(spawn_rx, response("error", &"x".repeat(5000)));
        let tool = SpawnSubagentTool::new(spawn_tx);
        let result = tool.execute(&tool_call()).await;
        let error = result.error.expect("error must be set on failure");
        assert!(
            error.chars().count() <= 400,
            "error must be bounded, got {} chars",
            error.chars().count()
        );
        assert!(error.ends_with('…'));
    }
}
