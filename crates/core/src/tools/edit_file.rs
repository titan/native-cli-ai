use std::sync::Arc;

use nca_common::tool::{ToolCall, ToolDefinition, ToolResult};
use serde::Deserialize;

use super::edit_match::{MatchLevel, MatchOutcome, seek_match};
use super::{ToolCallExt, ToolExecutor};
use crate::workspace_fs::{WorkspaceFs, sandbox_error_to_tool_result};

pub struct EditFileTool {
    fs: Arc<dyn WorkspaceFs>,
}

impl EditFileTool {
    pub fn new(fs: Arc<dyn WorkspaceFs>) -> Self {
        Self { fs }
    }
}

#[derive(Deserialize)]
struct Params {
    path: String,
    old_text: String,
    new_text: String,
    #[serde(default)]
    replace_all: bool,
}

#[async_trait::async_trait]
impl ToolExecutor for EditFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            timeout_ms: None,
            name: "edit_file".into(),
            description: "Replace a specific string in an existing file (tiered matching: exact substring, then whitespace/unicode-tolerant line levels, then bounded prefix/suffix rescue; replace_all only applies at the exact level)".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "old_text": { "type": "string" },
                    "new_text": { "type": "string" },
                    "replace_all": { "type": "boolean" }
                },
                "required": ["path", "old_text", "new_text"]
            }),
        }
    }

    async fn execute(&self, call: &ToolCall) -> ToolResult {
        let p: Params = match call.extract_params() {
            Ok(p) => p,
            Err(e) => return e,
        };

        if p.old_text.is_empty() {
            return ToolResult {
                timed_out: false,
                call_id: call.id.clone(),
                success: false,
                output: String::new(),
                error: Some("old_text must not be empty".into()),
            };
        }

        let content = match self.fs.read_file(&p.path).await {
            Ok(c) => c,
            Err(e) => return sandbox_error_to_tool_result(&call.id, e),
        };

        // Tiered matching (edit_match): exact substring first (replace_all
        // semantics preserved there), then fuzzy line levels and the bounded
        // rescue. Fuzzy-level multi-hits are always ambiguous — replace_all
        // never fans out a fuzzy replacement.
        let (updated, count, level_note) = match seek_match(&content, &p.old_text) {
            MatchOutcome::Unique { byte_range, level } => {
                let mut updated = content.clone();
                updated.replace_range(byte_range, &p.new_text);
                let note = (level != MatchLevel::Exact)
                    .then(|| format!("matched at {} level", level.label()));
                (updated, 1, note)
            }
            MatchOutcome::Rescue { byte_range } => {
                let mut updated = content.clone();
                updated.replace_range(byte_range, &p.new_text);
                (
                    updated,
                    1,
                    Some("matched via prefix/suffix rescue".to_string()),
                )
            }
            // Exact-level multi-hit keeps the replace_all contract: replace
            // every exact occurrence (historical behaviour, byte-identical).
            MatchOutcome::Ambiguous {
                level: Some(MatchLevel::Exact),
                candidates,
            } if p.replace_all => (content.replace(&p.old_text, &p.new_text), candidates, None),
            MatchOutcome::Ambiguous {
                level: Some(MatchLevel::Exact),
                candidates,
            } => {
                return ToolResult {
                    timed_out: false,
                    call_id: call.id.clone(),
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "old_text matched {candidates} occurrences; use replace_all or replace_match for a precise edit"
                    )),
                };
            }
            MatchOutcome::Ambiguous {
                level: Some(level),
                candidates,
            } => {
                return ToolResult {
                    timed_out: false,
                    call_id: call.id.clone(),
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "old_text matched {candidates} occurrences at {} level (ambiguous); add surrounding lines to old_text to disambiguate",
                        level.label()
                    )),
                };
            }
            MatchOutcome::Ambiguous {
                level: None,
                candidates,
            } => {
                return ToolResult {
                    timed_out: false,
                    call_id: call.id.clone(),
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "old_text anchors matched {candidates} regions (ambiguous rescue); add surrounding lines to old_text to disambiguate"
                    )),
                };
            }
            MatchOutcome::NotFound => {
                return ToolResult {
                    timed_out: false,
                    call_id: call.id.clone(),
                    success: false,
                    output: String::new(),
                    error: Some("old_text was not found".into()),
                };
            }
        };

        match self.fs.write_file(&p.path, &updated).await {
            Ok(()) => {
                let mut output = format!(
                    "Edited {path} (replaced {count} occurrence{s}",
                    path = p.path,
                    s = if count == 1 { "" } else { "s" }
                );
                if let Some(note) = &level_note {
                    output.push_str(&format!(", {note}"));
                }
                output.push(')');
                ToolResult {
                    timed_out: false,
                    call_id: call.id.clone(),
                    success: true,
                    output,
                    error: None,
                }
            }
            Err(e) => sandbox_error_to_tool_result(&call.id, e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace_fs::RealFs;

    fn make_call(input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "call-1".into(),
            name: "edit_file".into(),
            input,
        }
    }

    #[tokio::test]
    async fn edit_file_rejects_ambiguous_single_replacements() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("main.rs"), "alpha\nalpha\n").unwrap();

        let fs: Arc<dyn WorkspaceFs> = Arc::new(RealFs::new(dir.path().to_path_buf()));
        let tool = EditFileTool::new(fs);
        let result = tool
            .execute(&make_call(serde_json::json!({
                "path": "main.rs",
                "old_text": "alpha",
                "new_text": "beta"
            })))
            .await;

        assert!(!result.success);
        assert!(result.error.unwrap().contains("replace_match"));
    }

    #[tokio::test]
    async fn edit_file_replace_all_reports_replacement_count() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("main.rs"), "alpha\nalpha\n").unwrap();

        let fs: Arc<dyn WorkspaceFs> = Arc::new(RealFs::new(dir.path().to_path_buf()));
        let tool = EditFileTool::new(fs);
        let result = tool
            .execute(&make_call(serde_json::json!({
                "path": "main.rs",
                "old_text": "alpha",
                "new_text": "beta",
                "replace_all": true
            })))
            .await;

        assert!(result.success, "{result:?}");
        assert!(result.output.contains("replaced 2 occurrences"));
        let updated = std::fs::read_to_string(dir.path().join("main.rs")).unwrap();
        assert_eq!(updated, "beta\nbeta\n");
    }

    // =====================================================================
    // Tiered matching (omos matching.ts port — `edit_match`)
    // =====================================================================

    /// Trailing-whitespace drift resolves at trim_end, edits the whole
    /// original line, and reports the level in the success message.
    #[tokio::test]
    async fn edit_file_trailing_whitespace_drift_succeeds_with_level_report() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("main.rs"), "let x = 1;   \n").unwrap();

        let fs: Arc<dyn WorkspaceFs> = Arc::new(RealFs::new(dir.path().to_path_buf()));
        let tool = EditFileTool::new(fs);
        let result = tool
            .execute(&make_call(serde_json::json!({
                "path": "main.rs",
                "old_text": "let x = 1;\n",
                "new_text": "let x = 2;\n"
            })))
            .await;

        assert!(result.success, "{result:?}");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("main.rs")).unwrap(),
            "let x = 2;\n"
        );
        assert!(
            result.output.contains("replaced 1 occurrence"),
            "{result:?}"
        );
        assert!(
            result.output.contains("matched at trim_end level"),
            "{result:?}"
        );
    }

    /// A fuzzy-level multi-hit is ambiguous even with replace_all set —
    /// replace_all fans out only exact-substring matches.
    #[tokio::test]
    async fn edit_file_fuzzy_multi_hit_is_ambiguous_even_with_replace_all() {
        let dir = tempfile::tempdir().unwrap();
        let original = "foo  \nbar\nfoo\nbar  \n";
        std::fs::write(dir.path().join("main.rs"), original).unwrap();

        let fs: Arc<dyn WorkspaceFs> = Arc::new(RealFs::new(dir.path().to_path_buf()));
        let tool = EditFileTool::new(fs);
        let result = tool
            .execute(&make_call(serde_json::json!({
                "path": "main.rs",
                "old_text": "foo\nbar\n",
                "new_text": "X\nY\n",
                "replace_all": true
            })))
            .await;

        assert!(!result.success, "{result:?}");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("main.rs")).unwrap(),
            original
        );
        let error = result.error.unwrap();
        assert!(error.contains("trim_end"), "{error}");
        assert!(error.contains("ambiguous"), "{error}");
    }

    /// The anchored rescue works through edit_file too, with the rescue
    /// noted in the success message.
    #[tokio::test]
    async fn edit_file_rescue_rewrites_drifted_middle() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("main.rs"),
            "fn main() {\n    let x = 1;\n    println!(\"hello\");\n}\n",
        )
        .unwrap();

        let fs: Arc<dyn WorkspaceFs> = Arc::new(RealFs::new(dir.path().to_path_buf()));
        let tool = EditFileTool::new(fs);
        let result = tool
            .execute(&make_call(serde_json::json!({
                "path": "main.rs",
                "old_text": "fn main() {\n    let y = 2;\n    something_else();\n}",
                "new_text": "fn main() {\n    let z = 3;\n}"
            })))
            .await;

        assert!(result.success, "rescue must apply: {result:?}");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("main.rs")).unwrap(),
            "fn main() {\n    let z = 3;\n}\n"
        );
        assert!(
            result.output.contains("matched via prefix/suffix rescue"),
            "{result:?}"
        );
    }
}
