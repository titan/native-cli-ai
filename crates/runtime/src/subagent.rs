use crate::session_store::SessionStore;
use crate::session_utils::spawn_event_fanout;
use crate::supervisor::{AutoDenyHandler, Supervisor, SupervisorConfig};
use nca_common::config::{NcaConfig, ProviderKind};
use nca_common::event::{AgentEvent, EndReason};
use nca_core::approval::ApprovalHandler;
use nca_core::hooks::{HookEventKind, HookRunner};
use nca_core::tools::spawn_subagent::SpawnRequest;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Configuration for spawning a child session.
pub struct ChildSessionConfig {
    pub parent_session_id: String,
    pub task: String,
    pub workspace_root: PathBuf,
    pub config: NcaConfig,
    pub parent_summary: String,
    pub use_worktree: bool,
    pub focus_files: Vec<String>,
    /// Override the LLM provider for this child session.
    pub provider_override: Option<ProviderKind>,
    /// Override the model name for this child session.
    pub model_override: Option<String>,
    /// Optional specialist agent name. When set, the matching agent profile is
    /// loaded to override provider/model/system prompt for this child.
    pub specialist: Option<String>,
}

/// Result of a spawned child session.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChildSessionResult {
    pub child_session_id: String,
    pub status: String,
    pub output: String,
    pub workspace: String,
    pub branch: Option<String>,
    pub worktree_path: Option<String>,
}

/// Build a concise summary of the parent conversation for context inheritance.
pub(crate) fn build_parent_summary(messages: &[nca_common::message::Message]) -> String {
    use nca_common::message::Role;

    let mut summary = String::new();
    let recent: Vec<_> = messages
        .iter()
        .filter(|m| matches!(m.role, Role::User | Role::Assistant | Role::System))
        .collect();

    let window = if recent.len() > 10 {
        &recent[recent.len() - 10..]
    } else {
        &recent
    };

    for msg in window {
        let role = match msg.role {
            Role::User => "User",
            Role::Assistant => "Assistant",
            Role::System => "System",
            Role::Tool => continue,
        };
        let body = msg.content.event_preview();
        let content = if body.len() > 500 {
            let truncated: String = body.chars().take(500).collect();
            format!("{truncated}...")
        } else {
            body
        };
        summary.push_str(&format!("[{role}]: {content}\n\n"));
    }

    summary
}

/// Append a child session ID to the parent session's metadata on disk.
async fn append_child_to_parent(store: &SessionStore, parent_id: &str, child_id: &str) {
    if let Ok(mut parent) = store.load(parent_id).await
        && !parent
            .meta
            .child_session_ids
            .contains(&child_id.to_string())
    {
        parent.meta.child_session_ids.push(child_id.to_string());
        let _ = store.save(&parent).await;
    }
}

/// Spawn a child session that inherits parent context and runs to completion.
/// Returns the result of the child run. This is a blocking async call.
pub async fn spawn_child_session(
    cfg: ChildSessionConfig,
    event_tx: Option<tokio::sync::mpsc::Sender<AgentEvent>>,
) -> Result<ChildSessionResult, String> {
    // Child sessions are non-interactive and already authorized by the parent
    // approval. Elevate to BypassPermissions so sub-agents can write files,
    // run tools, and spawn their own children without being auto-denied.
    let mut child_config = cfg.config.clone();
    child_config.permissions.mode = nca_common::config::PermissionMode::BypassPermissions;

    // Apply provider/model routing. A specialist's agent profile is
    // AUTHORITATIVE for provider/model — explicit provider_override /
    // model_override from the tool call are ignored when a specialist profile
    // exists. This prevents the orchestrator LLM from bypassing the declarative
    // per-specialist routing configured by the user (e.g. echoing a misleading
    // "gpt-4o" example from the spawn_subagent description).
    let specialist_profile = apply_child_routing(
        &mut child_config,
        cfg.specialist.as_deref(),
        cfg.provider_override,
        cfg.model_override.as_deref(),
    );

    // Save specialist system_prompt for context injection (before config is moved).
    let specialist_persona = specialist_profile.and_then(|p| p.system_prompt.clone());

    let mut sup = Supervisor::create(SupervisorConfig {
        config: child_config,
        workspace_root: cfg.workspace_root.clone(),
        safe_mode: false,
        interactive_approvals: false,
        session_id: None,
        approval_handler: Some(Arc::new(AutoDenyHandler) as Arc<dyn ApprovalHandler>),
        orchestration_context: None,
        agent_name: cfg.specialist.clone(),
    })
    .await
    .map_err(|e| e.to_string())?;

    let child_id = sup.session_id.clone();

    sup.set_parent(
        cfg.parent_session_id.clone(),
        Some(cfg.parent_summary.clone()),
        Some(cfg.task.clone()),
    );

    if cfg.use_worktree {
        let wt_mgr = crate::worktree::WorktreeManager::new(&cfg.workspace_root);
        if wt_mgr.is_git_repo() {
            match wt_mgr.create_worktree(&child_id) {
                Ok(info) => {
                    sup.switch_to_worktree(
                        info.worktree_path.clone(),
                        info.branch_name.clone(),
                        info.base_branch.clone(),
                    );
                }
                Err(e) => {
                    tracing::warn!("Failed to create worktree for child session: {e}");
                }
            }
        }
    }

    if let Some(ref tx) = event_tx {
        let _ = tx
            .send(AgentEvent::ChildSessionSpawned {
                parent_session_id: cfg.parent_session_id.clone(),
                child_session_id: child_id.clone(),
                task: cfg.task.clone(),
                workspace: sup.workspace_root.clone(),
                branch: sup.branch.clone(),
            })
            .await;
    }

    let mut context_prompt = format!(
        "You are a sub-agent spawned by a parent session to handle a specific task.\n\n\
         ## Parent Context\n{}\n\n\
         ## Your Task\n{}",
        cfg.parent_summary, cfg.task
    );

    // If a specialist was requested, inject its persona at the beginning.
    if let Some(ref specialist) = cfg.specialist
        && let Some(ref persona) = specialist_persona
        && !persona.trim().is_empty()
    {
        context_prompt = format!(
            "## Specialist Persona: {specialist}\n\n\
             {}\n\n---\n\n{context_prompt}",
            persona.trim()
        );
    }

    if !cfg.focus_files.is_empty() {
        context_prompt.push_str("\n\n## Focus Files\n");
        for f in &cfg.focus_files {
            context_prompt.push_str(&format!("- {f}\n"));
        }
    }

    let mut handle = sup.take_handle();
    let event_rx = handle.take_event_rx();
    let log_path = handle.event_log_path.clone();

    let parent_forward = event_tx.map(|tx| (child_id.clone(), tx));
    let fanout = event_rx.map(|rx| spawn_event_fanout(rx, log_path, None, None, parent_forward));

    let result = sup.run_turn(&context_prompt).await;

    let (status, output) = match result {
        Ok(text) => {
            sup.finish(EndReason::Completed).await;
            ("completed".to_string(), text)
        }
        Err(e) => {
            sup.finish(EndReason::Error).await;
            ("error".to_string(), e.to_string())
        }
    };

    if let Some(f) = fanout {
        f.abort();
    }

    let branch = sup.branch.clone();
    let wt_path = sup.worktree_path.clone().map(|p| p.display().to_string());

    Ok(ChildSessionResult {
        child_session_id: child_id,
        status,
        output,
        workspace: sup.workspace_root.display().to_string(),
        branch,
        worktree_path: wt_path,
    })
}

/// Apply provider/model routing to a child session config.
///
/// When `specialist` resolves to an agent profile, that profile's provider/model
/// are AUTHORITATIVE and the explicit `provider_override`/`model_override` are
/// ignored. This protects the user's declarative per-specialist routing
/// (`[agents.<name>]`) from being bypassed by an orchestrator LLM that echoes
/// misleading example values (e.g. "gpt-4o") from the `spawn_subagent` tool
/// description.
///
/// Explicit overrides are honored only as an escape hatch when no specialist
/// profile exists (or no specialist was requested).
///
/// Returns the matched profile (for system-prompt injection) when a specialist
/// profile was used; `None` otherwise.
fn apply_child_routing(
    config: &mut NcaConfig,
    specialist: Option<&str>,
    provider_override: Option<ProviderKind>,
    model_override: Option<&str>,
) -> Option<nca_common::config::AgentProfileConfig> {
    let profile = specialist.and_then(|s| config.agent_profile(s).cloned());
    if let Some(ref profile) = profile {
        if let Some(provider) = profile.resolve_provider() {
            config.set_default_provider(provider);
        }
        if let Some(ref model) = profile.model {
            let resolved = config.model.resolve_alias(model);
            config.provider.set_model_for_default(resolved);
            config.sync_default_model_from_provider();
        }
    } else {
        if let Some(provider) = provider_override {
            config.set_default_provider(provider);
        }
        if let Some(model) = model_override {
            config
                .provider
                .set_model_for_default(config.model.resolve_alias(model));
            config.sync_default_model_from_provider();
        }
    }
    profile
}

/// Spawns a background task that consumes spawn requests from the sub-agent tool
/// and runs child sessions. Each child session inherits parent context.
pub fn spawn_subagent_consumer(
    mut spawn_rx: mpsc::Receiver<SpawnRequest>,
    parent_session_id: String,
    workspace_root: PathBuf,
    config: NcaConfig,
    parent_messages: Vec<nca_common::message::Message>,
    event_tx: Option<tokio::sync::mpsc::Sender<AgentEvent>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let parent_sessions_dir = workspace_root.join(&config.session.history_dir);
        let parent_summary = build_parent_summary(&parent_messages);

        while let Some(req) = spawn_rx.recv().await {
            let parent_session_id = parent_session_id.clone();
            let workspace_root = workspace_root.clone();
            let config = config.clone();
            let event_tx = event_tx.clone();
            let parent_store = SessionStore::new(parent_sessions_dir.clone());
            let parent_summary = parent_summary.clone();

            let child_cfg = ChildSessionConfig {
                parent_session_id: parent_session_id.clone(),
                task: req.task.clone(),
                workspace_root: workspace_root.clone(),
                config,
                parent_summary,
                use_worktree: req.use_worktree,
                focus_files: req.focus_files,
                provider_override: req.provider_override,
                model_override: req.model_override.clone(),
                specialist: req.specialist.clone(),
            };

            tokio::spawn(async move {
                let hook_runner = {
                    let runner = HookRunner::new(child_cfg.config.hooks.clone());
                    runner.has_any().then_some(runner)
                };
                if let Some(hooks) = &hook_runner {
                    hooks
                        .run_best_effort(
                            HookEventKind::SubagentStart,
                            None,
                            &json!({
                                "parent_session_id": parent_session_id.clone(),
                                "task": child_cfg.task.clone(),
                                "workspace": child_cfg.workspace_root.clone(),
                            }),
                        )
                        .await;
                }
                let result = spawn_child_session(child_cfg, event_tx.clone()).await;
                match result {
                    Ok(res) => {
                        append_child_to_parent(
                            &parent_store,
                            &parent_session_id,
                            &res.child_session_id,
                        )
                        .await;

                        if let Some(ref tx) = event_tx {
                            let _ = tx
                                .send(AgentEvent::ChildSessionCompleted {
                                    parent_session_id: parent_session_id.clone(),
                                    child_session_id: res.child_session_id.clone(),
                                    status: res.status.clone(),
                                })
                                .await;
                        }
                        if let Some(hooks) = &hook_runner {
                            hooks
                                .run_best_effort(
                                    HookEventKind::SubagentStop,
                                    None,
                                    &json!({
                                        "parent_session_id": parent_session_id.clone(),
                                        "child_session_id": res.child_session_id.clone(),
                                        "status": res.status.clone(),
                                    }),
                                )
                                .await;
                        }
                        let response = nca_core::tools::spawn_subagent::SpawnResponse {
                            child_session_id: res.child_session_id,
                            status: res.status,
                            output: res.output,
                            workspace: res.workspace,
                            branch: res.branch,
                            worktree_path: res.worktree_path,
                        };
                        let _ = req.reply.send(response);
                    }
                    Err(e) => {
                        if let Some(hooks) = &hook_runner {
                            hooks
                                .run_best_effort(
                                    HookEventKind::SubagentStop,
                                    None,
                                    &json!({
                                        "parent_session_id": parent_session_id.clone(),
                                        "status": "error",
                                        "error": e.clone(),
                                    }),
                                )
                                .await;
                        }
                        if let Some(ref tx) = event_tx {
                            let _ = tx
                                .send(AgentEvent::Error {
                                    message: format!("Failed to spawn child session: {e}"),
                                })
                                .await;
                        }
                        let response = nca_core::tools::spawn_subagent::SpawnResponse {
                            child_session_id: String::new(),
                            status: "error".into(),
                            output: e,
                            workspace: workspace_root.display().to_string(),
                            branch: None,
                            worktree_path: None,
                        };
                        let _ = req.reply.send(response);
                    }
                }
            });
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nca_common::config::AgentProfileConfig;

    fn config_with_librarian_profile() -> NcaConfig {
        let mut cfg = NcaConfig::default();
        cfg.agents.insert(
            "librarian".to_string(),
            AgentProfileConfig {
                provider: Some(ProviderKind::ZhipuAI),
                model: Some("glm-4.7-flash".to_string()),
                ..Default::default()
            },
        );
        cfg
    }

    #[test]
    fn specialist_profile_ignores_explicit_overrides() {
        // Intent: a user's `[agents.librarian]` routing must win even when the
        // orchestrator LLM echoes a misleading provider/model (e.g. "gpt-4o")
        // from the spawn_subagent tool description. Otherwise the declarative
        // per-specialist routing configured by the user is silently bypassed.
        let mut config = config_with_librarian_profile();
        let profile = apply_child_routing(
            &mut config,
            Some("librarian"),
            Some(ProviderKind::OpenAi),
            Some("gpt-4o"),
        );
        assert!(profile.is_some(), "librarian profile should match");
        assert_eq!(config.provider.default, ProviderKind::ZhipuAI);
        assert_eq!(config.provider.active_model(), "glm-4.7-flash");
        assert_eq!(config.model.default_model, "glm-4.7-flash");
    }

    #[test]
    fn specialist_profile_applied_without_overrides() {
        let mut config = config_with_librarian_profile();
        let profile = apply_child_routing(&mut config, Some("librarian"), None, None);
        assert!(profile.is_some());
        assert_eq!(config.provider.default, ProviderKind::ZhipuAI);
        assert_eq!(config.model.default_model, "glm-4.7-flash");
    }

    #[test]
    fn no_specialist_honors_overrides() {
        let mut config = NcaConfig::default();
        let profile = apply_child_routing(
            &mut config,
            None,
            Some(ProviderKind::ZhipuAI),
            Some("glm-4.7-flash"),
        );
        assert!(profile.is_none(), "no specialist means no profile");
        assert_eq!(config.provider.default, ProviderKind::ZhipuAI);
        assert_eq!(config.provider.active_model(), "glm-4.7-flash");
        assert_eq!(config.model.default_model, "glm-4.7-flash");
    }

    #[test]
    fn unknown_specialist_falls_back_to_overrides() {
        let mut config = NcaConfig::default();
        let profile = apply_child_routing(
            &mut config,
            Some("does-not-exist"),
            Some(ProviderKind::DeepSeek),
            None,
        );
        assert!(profile.is_none(), "unknown specialist matches no profile");
        assert_eq!(config.provider.default, ProviderKind::DeepSeek);
    }
}
