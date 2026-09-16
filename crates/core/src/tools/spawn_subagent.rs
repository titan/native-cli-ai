use nca_common::config::ProviderKind;
use nca_common::message::{ContentPart, ImageAttachment, Message, MessageContent};
use nca_common::tool::{ToolCall, ToolDefinition, ToolResult};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};

use super::ToolExecutor;

/// Upper bound on distinct images auto-forwarded with a spawn request; the
/// most recent are kept when the parent conversation exceeds this.
pub const MAX_FORWARD_IMAGES: usize = 8;

/// Request sent from the tool to the runtime to spawn a child session.
#[derive(Debug)]
pub struct SpawnRequest {
    pub task: String,
    pub focus_files: Vec<String>,
    /// Images collected from the parent's live history at spawn time, plus
    /// any image files the task text or `focus_files` reference (see
    /// `collect_task_image_references`). The runtime forwards them to the
    /// child's first message when the child's routed provider+model accepts
    /// native image input; paths are relative to the parent workspace root.
    pub images: Vec<ImageAttachment>,
    pub use_worktree: bool,
    /// Detached execution: `Some(true)` runs the child detached — the spawn
    /// reply returns immediately with the child session id and the final
    /// output is fetched later via `task_result`; completion auto-wakes the
    /// parent. `None` (flag absent) means inherit the session default,
    /// resolved by the runtime consumer (top-level TUI sessions default to
    /// background; P3). An explicit value always wins.
    pub background: Option<bool>,
    /// Parent-scoped name usable in place of the session id for the
    /// `task_*` control tools (P2).
    pub alias: Option<String>,
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
    /// Live mirror of the parent conversation, refreshed by the supervisor at
    /// each turn start, so image collection sees messages pasted after the
    /// consumer was wired (a wiring-time snapshot would miss them).
    history: Arc<Mutex<Vec<Message>>>,
}

impl SpawnSubagentTool {
    pub fn new(spawn_tx: mpsc::Sender<SpawnRequest>, history: Arc<Mutex<Vec<Message>>>) -> Self {
        Self { spawn_tx, history }
    }
}

/// Collect image attachments from a conversation snapshot, deduplicated by
/// path in chronological order. When more than `cap` distinct images exist,
/// the oldest are dropped and the dropped count is returned.
pub fn collect_recent_images(messages: &[Message], cap: usize) -> (Vec<ImageAttachment>, usize) {
    let mut seen = std::collections::HashSet::new();
    let mut images: Vec<ImageAttachment> = Vec::new();
    for message in messages {
        let MessageContent::Parts(parts) = &message.content else {
            continue;
        };
        for part in parts {
            if let ContentPart::Image { media_type, path } = part
                && seen.insert(path.clone())
            {
                images.push(ImageAttachment {
                    media_type: media_type.clone(),
                    path: path.clone(),
                });
            }
        }
    }
    if images.len() <= cap {
        return (images, 0);
    }
    let dropped = images.len() - cap;
    (images.split_off(dropped), dropped)
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
                        "description": "Optional list of file paths the sub-agent should focus on. Image files (png/jpg/gif/webp/bmp) listed here or mentioned in the task text are attached to the sub-agent's first message, so visual-analysis tasks work by path."
                    },
                    "use_worktree": {
                        "type": "boolean",
                        "description": "If true, the sub-agent runs in an isolated git worktree branch. Defaults to true."
                    },
                    "background": {
                        "type": "boolean",
                        "description": "Detached execution: the reply returns immediately with status \"running\" and the final output is fetched later via task_result; completion auto-wakes the parent (else poll task_status/task_result). Absent = inherit the session default (top-level TUI sessions default to background); an explicit true/false always wins."
                    },
                    "alias": {
                        "type": "string",
                        "description": "Short parent-scoped name usable in place of the long session id for ALL \
                            task_* tools (task_status/task_result/task_message/task_cancel/task_revive). \
                            Strongly recommended for every spawn: the returned session id is a long opaque \
                            string that is easy to mis-copy, while a short alias (e.g. 'fixer-1') is the \
                            reliable handle. Must be unique among this session's live tasks; duplicates \
                            make later alias-addressed calls ambiguous."
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

        // P3 wire field: background is now optional — absent (None) means
        // inherit the session default; the consumer resolves it. A blank
        // alias is treated as absent (never an empty-string alias).
        let background = call.input["background"].as_bool();
        let alias = call.input["alias"]
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from);

        // Parse optional provider/model overrides for per-agent routing.
        let provider_override = call.input["provider"]
            .as_str()
            .and_then(ProviderKind::from_cli_name);
        let model_override = call.input["model"].as_str().map(String::from);
        let specialist = call.input["specialist"]
            .as_str()
            .filter(|s| !s.trim().is_empty())
            .map(String::from);
        // Kept for the output hint below (the request consumes `alias`).
        let alias_hint = alias.clone();

        let (reply_tx, reply_rx) = oneshot::channel();

        let images = match self.history.lock() {
            Ok(messages) => collect_recent_images(&messages, MAX_FORWARD_IMAGES).0,
            Err(_) => Vec::new(),
        };

        let req = SpawnRequest {
            task,
            focus_files,
            images,
            use_worktree,
            background,
            alias,
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
                let mut output = serde_json::to_string_pretty(&response).unwrap_or_default();
                // When the caller set an alias, remind it that the alias is
                // the reliable handle for later task_* calls — the session id
                // is a long opaque string that is easy to mis-copy. Kept as a
                // single trailing line after the JSON so string-based
                // consumers of the JSON body keep working.
                if let Some(alias) = alias_hint.as_deref() {
                    output.push_str(&format!(
                        "\nAddress this task as alias \"{alias}\" (or exact session id) in task_* tools."
                    ));
                }
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
        let tool = SpawnSubagentTool::new(spawn_tx, empty_history());
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
        let tool = SpawnSubagentTool::new(spawn_tx, empty_history());
        let result = tool.execute(&tool_call()).await;
        assert_eq!(
            result.error.as_deref(),
            Some("Sub-agent finished with status: error")
        );

        // 5000-char output: error stays within the 300-char truncation bound
        // (plus the fixed prefix).
        let (spawn_tx, spawn_rx) = mpsc::channel(1);
        spawn_responder(spawn_rx, response("error", &"x".repeat(5000)));
        let tool = SpawnSubagentTool::new(spawn_tx, empty_history());
        let result = tool.execute(&tool_call()).await;
        let error = result.error.expect("error must be set on failure");
        assert!(
            error.chars().count() <= 400,
            "error must be bounded, got {} chars",
            error.chars().count()
        );
        assert!(error.ends_with('…'));
    }

    fn empty_history() -> Arc<Mutex<Vec<Message>>> {
        Arc::new(Mutex::new(Vec::new()))
    }

    fn image_part(path: &str) -> ContentPart {
        ContentPart::Image {
            media_type: "image/png".into(),
            path: path.into(),
        }
    }

    #[test]
    fn collect_recent_images_dedups_and_caps_oldest() {
        let messages = vec![
            Message::user_with_parts(vec![
                ContentPart::Text { text: "a".into() },
                image_part("one.png"),
            ]),
            Message::user("no images here"),
            Message::user_with_parts(vec![image_part("two.png"), image_part("one.png")]),
            Message::user_with_parts(vec![image_part("three.png")]),
        ];

        let (images, dropped) = collect_recent_images(&messages, 2);
        assert_eq!(dropped, 1);
        let paths: Vec<_> = images.iter().map(|i| i.path.as_str()).collect();
        assert_eq!(paths, ["two.png", "three.png"]);

        let (images, dropped) = collect_recent_images(&messages, 8);
        assert_eq!((images.len(), dropped), (3, 0));
    }

    #[tokio::test]
    async fn execute_collects_images_from_history_mirror() {
        let (spawn_tx, mut spawn_rx) = mpsc::channel::<SpawnRequest>(1);
        let history = Arc::new(Mutex::new(vec![Message::user_with_parts(vec![
            image_part("shot.png"),
        ])]));

        let responder = tokio::spawn(async move {
            if let Some(req) = spawn_rx.recv().await {
                let _ = req.reply.send(response("completed", "ok"));
                req.images
            } else {
                Vec::new()
            }
        });

        let tool = SpawnSubagentTool::new(spawn_tx, history);
        let result = tool.execute(&tool_call()).await;
        assert!(result.success);

        let images = responder.await.expect("responder");
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].path, "shot.png");
    }

    #[tokio::test]
    async fn execute_parses_background_and_alias_with_defaults() {
        // Explicit values ride the request.
        let (spawn_tx, mut spawn_rx) = mpsc::channel::<SpawnRequest>(1);
        let capture = tokio::spawn(async move {
            match spawn_rx.recv().await {
                Some(req) => {
                    let _ = req.reply.send(response("completed", "ok"));
                    (req.background, req.alias)
                }
                None => panic!("spawn request must arrive"),
            }
        });
        let tool = SpawnSubagentTool::new(spawn_tx, empty_history());
        let result = tool
            .execute(&ToolCall {
                id: "call-1".into(),
                name: "spawn_subagent".into(),
                input: serde_json::json!({
                    "task": "do the thing",
                    "background": true,
                    "alias": "fixer-2"
                }),
            })
            .await;
        assert!(result.success);
        // JSON body survives and the alias hint rides as a trailing line.
        assert!(
            result.output.contains("child_session_id"),
            "output must still contain the JSON body: {}",
            result.output
        );
        assert!(
            result.output.contains(
                "Address this task as alias \"fixer-2\" (or exact session id) in task_* tools."
            ),
            "alias hint must appear in output: {}",
            result.output
        );
        let (background, alias) = capture.await.expect("capture");
        assert_eq!(background, Some(true));
        assert_eq!(alias.as_deref(), Some("fixer-2"));

        // Defaults: background=None (inherit the session default); a blank
        // alias is treated as absent rather than an empty-string alias.
        let (spawn_tx, mut spawn_rx) = mpsc::channel::<SpawnRequest>(1);
        let capture = tokio::spawn(async move {
            match spawn_rx.recv().await {
                Some(req) => {
                    let _ = req.reply.send(response("completed", "ok"));
                    (req.background, req.alias)
                }
                None => panic!("spawn request must arrive"),
            }
        });
        let tool = SpawnSubagentTool::new(spawn_tx, empty_history());
        let result = tool
            .execute(&ToolCall {
                id: "call-2".into(),
                name: "spawn_subagent".into(),
                input: serde_json::json!({ "task": "do the thing", "alias": "   " }),
            })
            .await;
        assert!(result.success);
        assert!(
            !result.output.contains("Address this task as alias"),
            "no alias hint without an alias: {}",
            result.output
        );
        let (background, alias) = capture.await.expect("capture");
        assert_eq!(
            background, None,
            "absent background must ride as None (inherit), not false"
        );
        assert_eq!(alias, None, "blank alias must be treated as absent");
    }

    #[test]
    fn definition_declares_background_and_alias() {
        let (tx, _rx) = mpsc::channel(1);
        let tool = SpawnSubagentTool::new(tx, empty_history());
        let def = tool.definition();
        let props = def.parameters["properties"]
            .as_object()
            .expect("properties");
        assert_eq!(props["background"]["type"], "boolean", "schema: {props:?}");
        assert_eq!(props["alias"]["type"], "string", "schema: {props:?}");
    }
}
