//! Tool that lets the LLM load a skill's full instructions by name.

use crate::skills::{SkillCatalog, SkillFilterHandle};
use crate::tools::ToolExecutor;
use nca_common::tool::{ToolCall, ToolDefinition, ToolResult};
use std::collections::HashSet;
use std::path::PathBuf;

pub struct InvokeSkillTool {
    workspace_root: PathBuf,
    skill_directories: Vec<PathBuf>,
    /// Live per-agent gate (see [`SkillFilterHandle`]). `None` = this tool
    /// instance is ungated (default construction, tests). A handle holding
    /// `None` is likewise ungated; `Some(set)` restricts invocation and the
    /// listed-available set to `set`. Read fresh at every execution so
    /// runtime agent switches re-gate without a registry rebuild.
    skill_filter: Option<SkillFilterHandle>,
}

impl InvokeSkillTool {
    pub fn new(workspace_root: PathBuf, skill_directories: Vec<PathBuf>) -> Self {
        Self {
            workspace_root,
            skill_directories,
            skill_filter: None,
        }
    }

    /// Attach a live per-agent skill gate (supervisor-owned, refreshed on
    /// agent profile switches).
    pub fn with_skill_filter(mut self, filter: SkillFilterHandle) -> Self {
        self.skill_filter = Some(filter);
        self
    }

    /// Read the gate's current effective set. No handle, or a handle holding
    /// `None`, means no filtering. A poisoned lock fails loudly instead of
    /// degrading to "allow everything".
    fn current_filter(&self) -> Result<Option<HashSet<String>>, String> {
        match &self.skill_filter {
            None => Ok(None),
            Some(handle) => handle
                .read()
                .map(|guard| guard.clone())
                .map_err(|_| "invoke_skill skill filter lock poisoned".to_string()),
        }
    }
}

#[async_trait::async_trait]
impl ToolExecutor for InvokeSkillTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            timeout_ms: None,
            name: "invoke_skill".into(),
            description: "Load a skill's full instructions by name. Use this when a task matches \
                an available skill from the skills manifest. Returns the complete skill \
                instructions to follow."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The command name from the skills manifest (e.g., 'brainstorming', 'test-driven-development')"
                    }
                },
                "required": ["skill_name"]
            }),
        }
    }

    async fn execute(&self, call: &ToolCall) -> ToolResult {
        let skill_name = call.input["skill_name"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string();

        if skill_name.is_empty() {
            return ToolResult {
                timed_out: false,
                call_id: call.id.clone(),
                success: false,
                output: String::new(),
                error: Some("skill_name is required".into()),
            };
        }

        let skills = match SkillCatalog::discover(&self.workspace_root, &self.skill_directories) {
            Ok(s) => s,
            Err(e) => {
                return ToolResult {
                    timed_out: false,
                    call_id: call.id.clone(),
                    success: false,
                    output: String::new(),
                    error: Some(format!("Failed to discover skills: {e}")),
                };
            }
        };

        let effective = match self.current_filter() {
            Ok(filter) => filter,
            Err(error) => {
                return ToolResult {
                    timed_out: false,
                    call_id: call.id.clone(),
                    success: false,
                    output: String::new(),
                    error: Some(error),
                };
            }
        };
        let allowed = |name: &str| effective.as_ref().is_none_or(|set| set.contains(name));

        if let Some(skill) = skills
            .iter()
            .find(|s| s.command == skill_name && allowed(&s.command))
        {
            let body = skill.expanded_body();
            ToolResult {
                timed_out: false,
                call_id: call.id.clone(),
                success: true,
                output: format!(
                    "Skill `{}` loaded. Follow these instructions:\n\n{}",
                    skill.command,
                    body.trim()
                ),
                error: None,
            }
        } else {
            let exists = skills.iter().any(|s| s.command == skill_name);
            let available: Vec<&str> = skills
                .iter()
                .filter(|s| allowed(&s.command))
                .map(|s| s.command.as_str())
                .collect();
            let available = if available.is_empty() {
                "(none)".to_string()
            } else {
                available.join(", ")
            };
            let reason = if exists {
                format!(
                    "Skill '{skill_name}' is not available to the active agent profile \
                     (excluded by its skills/skills_add/skills_remove configuration)."
                )
            } else {
                format!("Skill '{skill_name}' not found.")
            };
            ToolResult {
                timed_out: false,
                call_id: call.id.clone(),
                success: false,
                output: String::new(),
                error: Some(format!("{reason} Available skills: {available}")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tool(workspace: &std::path::Path) -> InvokeSkillTool {
        InvokeSkillTool::new(
            workspace.to_path_buf(),
            vec![std::path::PathBuf::from(".nca/skills")],
        )
    }

    fn make_call(skill_name: &str) -> ToolCall {
        ToolCall {
            id: "call-1".into(),
            name: "invoke_skill".into(),
            input: serde_json::json!({ "skill_name": skill_name }),
        }
    }

    #[test]
    fn definition_has_correct_name_and_parameters() {
        let dir = tempfile::tempdir().unwrap();
        let tool = make_tool(dir.path());
        let def = tool.definition();
        assert_eq!(def.name, "invoke_skill");
        assert!(def.description.contains("Load a skill"));
        assert!(def.parameters["properties"]["skill_name"].is_object());
        assert!(
            def.parameters["required"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("skill_name"))
        );
    }

    #[tokio::test]
    async fn returns_expanded_body_for_valid_skill() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join(".nca/skills/my-skill");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: My Skill\ncommand: my-skill\ndescription: A test skill\n---\nDo the thing.\n\nSee ./helper.md for details.\n",
        )
        .unwrap();
        std::fs::write(skill_dir.join("helper.md"), "Helper content.").unwrap();

        let tool = make_tool(dir.path());
        let result = tool.execute(&make_call("my-skill")).await;

        assert!(result.success);
        assert!(result.output.contains("Skill `my-skill` loaded"));
        assert!(result.output.contains("Do the thing."));
        assert!(result.output.contains("===== helper.md ====="));
        assert!(result.output.contains("Helper content."));
    }

    #[tokio::test]
    async fn returns_error_for_unknown_skill() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join(".nca/skills/real-skill");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: Real\ncommand: real-skill\n---\nReal body.\n",
        )
        .unwrap();

        let tool = make_tool(dir.path());
        let result = tool.execute(&make_call("nonexistent")).await;

        assert!(!result.success);
        let err = result.error.unwrap();
        assert!(err.contains("not found"));
        assert!(err.contains("real-skill"));
    }

    #[tokio::test]
    async fn returns_error_for_empty_skill_name() {
        let dir = tempfile::tempdir().unwrap();
        let tool = make_tool(dir.path());
        let result = tool.execute(&make_call("")).await;

        assert!(!result.success);
        assert!(result.error.unwrap().contains("skill_name is required"));
    }

    // === per-agent skill gate tests ===

    fn write_two_skills(dir: &std::path::Path) {
        for command in ["alpha", "beta"] {
            let skill_dir = dir.join(format!(".nca/skills/{command}"));
            std::fs::create_dir_all(&skill_dir).unwrap();
            std::fs::write(
                skill_dir.join("SKILL.md"),
                format!(
                    "---\nname: {command}\ncommand: {command}\ndescription: {command} skill\n---\n{command} body.\n"
                ),
            )
            .unwrap();
        }
    }

    fn gated_tool(dir: &std::path::Path, allowed: &[&str]) -> (InvokeSkillTool, SkillFilterHandle) {
        let handle: SkillFilterHandle = std::sync::Arc::new(std::sync::RwLock::new(Some(
            allowed.iter().map(|s| s.to_string()).collect(),
        )));
        (make_tool(dir).with_skill_filter(handle.clone()), handle)
    }

    #[tokio::test]
    async fn gate_blocks_excluded_skill_and_lists_only_allowed() {
        let dir = tempfile::tempdir().unwrap();
        write_two_skills(dir.path());
        let (tool, _handle) = gated_tool(dir.path(), &["alpha"]);

        let result = tool.execute(&make_call("beta")).await;

        assert!(!result.success);
        let error = result.error.unwrap();
        assert!(
            error.contains("not available to the active agent profile"),
            "gated skill must be denied with the profile reason: {error}"
        );
        assert!(
            error.contains("alpha"),
            "allowed skill must be listed: {error}"
        );
        assert!(!error.contains("beta body"));
        // The error may mention the denied name itself; the LISTING must not
        // offer it. Assert on the "Available skills:" tail only.
        let listing = error.split("Available skills:").nth(1).unwrap();
        assert!(
            !listing.contains("beta"),
            "denied skill must not be offered: {error}"
        );
    }

    #[tokio::test]
    async fn gate_allows_included_skill_to_load() {
        let dir = tempfile::tempdir().unwrap();
        write_two_skills(dir.path());
        let (tool, _handle) = gated_tool(dir.path(), &["alpha"]);

        let result = tool.execute(&make_call("alpha")).await;

        assert!(result.success);
        assert!(result.output.contains("Skill `alpha` loaded"));
        assert!(result.output.contains("alpha body."));
    }

    #[tokio::test]
    async fn gate_none_inside_handle_means_unfiltered() {
        let dir = tempfile::tempdir().unwrap();
        write_two_skills(dir.path());
        // Handle present but holding None (default persona) → no filtering.
        let handle: SkillFilterHandle = std::sync::Arc::new(std::sync::RwLock::new(None));
        let tool = make_tool(dir.path()).with_skill_filter(handle);

        let result = tool.execute(&make_call("beta")).await;

        assert!(result.success);
        assert!(result.output.contains("beta body."));
    }

    #[tokio::test]
    async fn gate_reads_live_so_profile_switches_apply_without_rebuild() {
        let dir = tempfile::tempdir().unwrap();
        write_two_skills(dir.path());
        let (tool, handle) = gated_tool(dir.path(), &["alpha"]);

        // Denied while the gate excludes it…
        assert!(!tool.execute(&make_call("beta")).await.success);

        // …allowed after the supervisor re-writes the handle (agent switch).
        *handle.write().unwrap() = Some(["alpha", "beta"].iter().map(|s| s.to_string()).collect());
        let result = tool.execute(&make_call("beta")).await;
        assert!(result.success);
        assert!(result.output.contains("beta body."));

        // …and cleared entirely (back to @orchestrator).
        *handle.write().unwrap() = None;
        assert!(tool.execute(&make_call("beta")).await.success);
    }

    #[tokio::test]
    async fn gate_empty_set_denies_everything_with_none_listing() {
        let dir = tempfile::tempdir().unwrap();
        write_two_skills(dir.path());
        let (tool, _handle) = gated_tool(dir.path(), &[]);

        let result = tool.execute(&make_call("alpha")).await;

        assert!(!result.success);
        let error = result.error.unwrap();
        assert!(error.contains("not available to the active agent profile"));
        assert!(
            error.contains("Available skills: (none)"),
            "empty gate must list nothing, loudly: {error}"
        );
    }

    #[tokio::test]
    async fn gate_poisoned_lock_fails_loudly() {
        let dir = tempfile::tempdir().unwrap();
        write_two_skills(dir.path());
        let (tool, handle) = gated_tool(dir.path(), &["alpha"]);

        // Poison the lock: panic on a worker thread while a write guard is
        // held — the guard drops during unwind and the lock is poisoned from
        // then on. The tool must fail loudly, never degrade to "allow all".
        let poison_handle = handle.clone();
        let poisoner = std::thread::spawn(move || {
            let _guard = poison_handle.write().unwrap();
            panic!("poison the skill gate lock");
        });
        assert!(
            poisoner.join().is_err(),
            "poisoner thread must have panicked"
        );

        let result = tool.execute(&make_call("alpha")).await;
        assert!(!result.success);
        let error = result.error.unwrap();
        assert!(
            error.contains("poisoned"),
            "poisoned gate must surface a loud error: {error}"
        );
    }
}
