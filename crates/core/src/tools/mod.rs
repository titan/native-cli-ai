pub mod apply_patch;
pub mod ask_question;
pub mod ast_grep;
pub mod code_intel_tool;
pub mod copy_path;
pub mod create_directory;
pub mod delete_path;
pub mod edit_file;
pub mod fetch_url;
pub mod filesystem;
pub mod git;
pub mod input_repair;
pub mod invoke_skill;
pub mod list_directory;
pub mod mcp;
pub mod move_path;
pub mod rename_path;
pub mod replace_match;
pub mod run_validation;
pub mod search;
pub mod spawn_subagent;
pub mod subagent_control;
pub mod types;
pub mod update_todos;
pub mod web_search;
pub mod write_file;

pub use ask_question::AskQuestionTool;
pub use invoke_skill::InvokeSkillTool;
pub use subagent_control::{
    SubagentControlRequest, SubagentControlResponse, TaskResultTool, TaskStatusTool,
};
pub use update_todos::{TodoStore, UpdateTodosTool, validate_todos};

use nca_common::config::WebConfig;
use nca_common::event::AgentEvent;
use nca_common::tool::{ToolCall, ToolDefinition, ToolResult};
use std::sync::Arc;

use crate::workspace_fs::WorkspaceFs;

/// Tool names that block awaiting a human answer.
///
/// The tool pipeline runs these strictly one at a time (barrier semantics,
/// see `tool_pipeline`): every UI surface tracks a single active question,
/// so two simultaneous `QuestionRequested` events would overwrite the first
/// in the UI, orphan its oneshot, and freeze the turn forever.
const INTERACTIVE_TOOLS: &[&str] = &["ask_question"];

/// Registry of available tools the agent can invoke.
pub struct ToolRegistry {
    tools: Vec<Box<dyn ToolExecutor>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    pub fn register(&mut self, tool: Box<dyn ToolExecutor>) {
        self.tools.push(tool);
    }

    pub fn with_default_readonly_tools(fs: Arc<dyn WorkspaceFs>, web_config: WebConfig) -> Self {
        let mut registry = Self::new();
        registry.register(Box::new(filesystem::ReadFileTool::new(fs.clone())));
        registry.register(Box::new(search::SearchCodeTool::new(fs.clone())));
        registry.register(Box::new(ast_grep::AstGrepSearchTool::new(fs.clone())));
        registry.register(Box::new(list_directory::ListDirectoryTool::new(fs.clone())));
        registry.register(Box::new(git::GitStatusTool::new(fs.clone())));
        registry.register(Box::new(git::GitDiffTool::new(fs)));
        registry.register(Box::new(web_search::WebSearchTool::new(web_config.clone())));
        registry.register(Box::new(fetch_url::FetchUrlTool::new(web_config)));
        registry
    }

    pub fn with_default_full_tools(fs: Arc<dyn WorkspaceFs>, web_config: WebConfig) -> Self {
        let mut registry = Self::with_default_readonly_tools(fs.clone(), web_config);
        registry.register(Box::new(code_intel_tool::CodeIntelTool::new(
            crate::code_intel::FastLocalCodeIntel::new(fs.root()),
        )));
        registry.register(Box::new(write_file::WriteFileTool::new(fs.clone())));
        registry.register(Box::new(create_directory::CreateDirectoryTool::new(
            fs.clone(),
        )));
        registry.register(Box::new(apply_patch::ApplyPatchTool::new(fs.clone())));
        registry.register(Box::new(edit_file::EditFileTool::new(fs.clone())));
        registry.register(Box::new(replace_match::ReplaceMatchTool::new(fs.clone())));
        registry.register(Box::new(ast_grep::AstGrepReplaceTool::new(fs.clone())));
        registry.register(Box::new(rename_path::RenamePathTool::new(fs.clone())));
        registry.register(Box::new(move_path::MovePathTool::new(fs.clone())));
        registry.register(Box::new(copy_path::CopyPathTool::new(fs.clone())));
        registry.register(Box::new(delete_path::DeletePathTool::new(fs.clone())));
        registry.register(Box::new(run_validation::RunValidationTool::new(fs)));
        registry
    }

    /// Declarative timeout (if any) declared by the tool with this name.
    /// Used by the tool pipeline to wrap execution in a cooperative timeout.
    pub fn timeout_ms_for(&self, name: &str) -> Option<u64> {
        self.tools
            .iter()
            .find(|t| t.definition().name == name)
            .and_then(|t| t.definition().timeout_ms)
    }

    /// True when the named tool blocks awaiting a human answer (e.g.
    /// `ask_question`). The tool pipeline serializes these so at most one is
    /// ever pending — every UI surface tracks a single active question.
    pub fn is_interactive(&self, name: &str) -> bool {
        INTERACTIVE_TOOLS.contains(&name)
    }

    /// Remove the named tool from the registry (no-op when absent). Used to
    /// strip interactive tools from sessions that have no user attached to
    /// answer them (child subagent sessions).
    pub fn unregister(&mut self, name: &str) {
        self.tools.retain(|t| t.definition().name != name);
    }

    /// Retain only tools whose name is in `allowed`; remove all others.
    /// Used by agent profiles to enforce tool gating.
    pub fn restrict_to(&mut self, allowed: &[String]) {
        self.tools
            .retain(|t| allowed.iter().any(|name| t.definition().name == *name));
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        let mut seen = std::collections::HashSet::new();
        self.tools
            .iter()
            .filter_map(|t| {
                let def = t.definition();
                if seen.insert(def.name.clone()) {
                    Some(def)
                } else {
                    tracing::warn!("duplicate tool name skipped: {}", def.name);
                    None
                }
            })
            .collect()
    }

    pub async fn execute(&self, call: &ToolCall) -> ToolResult {
        for tool in &self.tools {
            if tool.definition().name == call.name {
                return tool.execute(call).await;
            }
        }

        ToolResult {
            timed_out: false,
            call_id: call.id.clone(),
            success: false,
            output: String::new(),
            error: Some(format!("Unknown tool: {}", call.name)),
        }
    }

    /// Streaming dispatch: like [`execute`](Self::execute) but forwards the
    /// `progress` handle so streaming tools can emit incremental output.
    pub async fn execute_streaming(&self, call: &ToolCall, progress: &ToolProgress) -> ToolResult {
        for tool in &self.tools {
            if tool.definition().name == call.name {
                return tool.execute_streaming(call, progress).await;
            }
        }
        ToolResult {
            timed_out: false,
            call_id: call.id.clone(),
            success: false,
            output: String::new(),
            error: Some(format!("Unknown tool: {}", call.name)),
        }
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle given to a tool so it can stream incremental output to the UI.
///
/// Cheap to clone (clones the channel sender). Tools that do not stream
/// output simply ignore this; their default `execute_streaming` delegates
/// to [`ToolExecutor::execute`].
#[derive(Clone)]
pub struct ToolProgress {
    call_id: String,
    sender: tokio::sync::mpsc::Sender<AgentEvent>,
}

impl ToolProgress {
    pub fn new(call_id: impl Into<String>, sender: tokio::sync::mpsc::Sender<AgentEvent>) -> Self {
        Self {
            call_id: call_id.into(),
            sender,
        }
    }

    /// Emit a chunk of streamed output. Best-effort and non-blocking: if the
    /// event channel is full the chunk is dropped (never blocks tool execution).
    pub fn emit_chunk(&self, delta: &str) {
        let _ = self.sender.try_send(AgentEvent::ToolOutputChunk {
            call_id: self.call_id.clone(),
            delta: delta.to_string(),
        });
    }
}

/// Trait implemented by each tool.
#[async_trait::async_trait]
pub trait ToolExecutor: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    async fn execute(&self, call: &ToolCall) -> ToolResult;

    /// Streaming variant. The default implementation ignores `progress` and
    /// delegates to [`execute`](Self::execute). Override this to emit
    /// incremental output via `progress.emit_chunk(...)` while running.
    async fn execute_streaming(&self, call: &ToolCall, progress: &ToolProgress) -> ToolResult {
        let _ = progress;
        self.execute(call).await
    }
}

// ---------------------------------------------------------------------------
// C2: Typed parameter extraction
// ---------------------------------------------------------------------------

/// Extension trait for extracting typed parameters from a [`ToolCall`].
///
/// Each tool defines a `#[derive(Deserialize)]` struct for its parameters and
/// calls `call.extract_params::<Params>()?` at the top of `execute()`. This
/// replaces the repetitive `call.input["key"].as_str().unwrap_or("")` pattern
/// with a single deserialization call that:
/// - reports missing required fields clearly,
/// - coerces types via serde,
/// - provides compile-time struct shape for tests.
///
/// **Validate-then-repair**: if the initial deserialization fails, the input
/// is passed through [`input_repair::repair_value`] which fixes common LLM
/// tool-calling mistakes (null optional fields, stringified arrays, bare
/// objects where arrays are expected, etc.) before retrying. Valid inputs
/// are never touched.
pub trait ToolCallExt {
    fn extract_params<T: serde::de::DeserializeOwned>(&self) -> Result<T, ToolResult>;
}

impl ToolCallExt for ToolCall {
    fn extract_params<T: serde::de::DeserializeOwned>(&self) -> Result<T, ToolResult> {
        // Phase 1: try direct deserialization (fast path for well-formed inputs).
        if let Ok(params) = serde_json::from_value::<T>(self.input.clone()) {
            return Ok(params);
        }

        // Phase 2: repair then retry (handles ~90% of open-model tool errors).
        let repaired = input_repair::repair_value(&self.input);
        let Ok(params) = serde_json::from_value::<T>(repaired) else {
            return Err(ToolResult {
                timed_out: false,
                call_id: self.id.clone(),
                success: false,
                output: String::new(),
                error: Some(format_tool_param_error(&self.name, &self.input)),
            });
        };
        tracing::info!(
            tool = %self.name,
            call_id = %self.id,
            "tool_input_repaired"
        );
        Ok(params)
    }
}

/// Build a model-readable error message for invalid tool parameters.
///
/// Instead of a raw serde error (which models can't recover from), this
/// surfaces the tool name and the parameter keys that were received.
///
/// When the input carries an `_error` key (set by the streaming layer when
/// the raw arguments could not be parsed as JSON at all), the error value
/// is surfaced directly so the model can see what went wrong and correct
/// it on retry.
fn format_tool_param_error(tool_name: &str, input: &serde_json::Value) -> String {
    // Special case: the streaming layer could not parse the arguments as
    // JSON at all and left an _error sentinel.  Surface the parse failure
    // so the model can self-correct instead of seeing a useless key list.
    if let Some(err_msg) = input
        .as_object()
        .and_then(|m| m.get("_error"))
        .and_then(|v| v.as_str())
    {
        return format!("Invalid parameters for tool `{}`. {err_msg}", tool_name);
    }

    let received_keys = match input.as_object() {
        Some(map) => map.keys().cloned().collect::<Vec<_>>().join(", "),
        None => format!("raw value: {}", input),
    };
    format!(
        "Invalid parameters for tool `{}`. Received keys: [{}]. Please check the tool schema and retry.",
        tool_name, received_keys
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nca_common::tool::ToolCall;

    /// A minimal tool for testing ToolRegistry operations.
    struct StubTool {
        name: String,
    }

    #[async_trait::async_trait]
    impl ToolExecutor for StubTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                timeout_ms: None,
                name: self.name.clone(),
                description: "stub".into(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            }
        }
        async fn execute(&self, _call: &ToolCall) -> ToolResult {
            ToolResult {
                timed_out: false,
                call_id: String::new(),
                success: true,
                output: String::new(),
                error: None,
            }
        }
    }

    fn stub_registry(names: &[&str]) -> ToolRegistry {
        let mut reg = ToolRegistry::new();
        for name in names {
            reg.register(Box::new(StubTool {
                name: (*name).into(),
            }));
        }
        reg
    }

    #[test]
    fn restrict_to_keeps_only_allowed_tools() {
        let mut reg = stub_registry(&["read_file", "search_code", "write_file", "delete_path"]);
        reg.restrict_to(&["read_file".into(), "search_code".into()]);

        let names: Vec<String> = reg.definitions().iter().map(|d| d.name.clone()).collect();
        assert_eq!(names, vec!["read_file", "search_code"]);
    }

    #[test]
    fn restrict_to_with_empty_allowed_removes_all() {
        let mut reg = stub_registry(&["read_file", "write_file"]);
        reg.restrict_to(&[]);

        assert!(reg.definitions().is_empty());
    }

    #[test]
    fn restrict_to_with_unknown_names_removes_all() {
        let mut reg = stub_registry(&["read_file", "write_file"]);
        reg.restrict_to(&["nonexistent".into()]);

        assert!(reg.definitions().is_empty());
    }

    #[test]
    fn is_interactive_marks_ask_question_only() {
        let reg = stub_registry(&["ask_question", "read_file"]);
        assert!(reg.is_interactive("ask_question"));
        assert!(!reg.is_interactive("read_file"));
        assert!(!reg.is_interactive("not_registered"));
    }

    #[test]
    fn unregister_removes_only_the_named_tool() {
        let mut reg = stub_registry(&["ask_question", "read_file", "write_file"]);
        reg.unregister("ask_question");

        let names: Vec<String> = reg.definitions().iter().map(|d| d.name.clone()).collect();
        assert_eq!(names, vec!["read_file", "write_file"]);

        // Unregistering an absent tool is a no-op.
        reg.unregister("ask_question");
        assert_eq!(reg.definitions().len(), 2);
    }

    #[test]
    fn format_error_surfaces_error_sentinel_value() {
        // When the streaming layer can't parse arguments, it sends an _error
        // sentinel.  The error message should surface the parse failure so
        // the model can self-correct, not just show "Received keys: [_error]".
        let input = serde_json::json!({
            "_error": "Failed to parse tool arguments as JSON. Raw input: {bad"
        });
        let msg = format_tool_param_error("write_file", &input);
        assert!(msg.contains("Failed to parse tool arguments as JSON"));
        assert!(!msg.contains("Received keys"));
    }

    #[test]
    fn format_error_shows_keys_for_normal_malformed_input() {
        let input = serde_json::json!({"foo": 1, "bar": 2});
        let msg = format_tool_param_error("write_file", &input);
        assert!(msg.contains("foo"));
        assert!(msg.contains("bar"));
        assert!(msg.contains("Received keys"));
    }
}
