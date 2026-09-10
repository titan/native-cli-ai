use crate::session_utils::spawn_event_fanout;
use crate::supervisor::{AutoDenyHandler, Supervisor, SupervisorConfig};
use nca_common::config::{NcaConfig, ProviderKind};
use nca_common::event::{AgentEvent, EndReason};
use nca_common::message::ImageAttachment;
use nca_common::model_caps::model_accepts_native_images;
use nca_core::approval::ApprovalHandler;
use nca_core::hooks::{HookEventKind, HookRunner};
use nca_core::tools::spawn_subagent::{MAX_FORWARD_IMAGES, SpawnRequest};
use nca_core::workspace_fs::WorkspaceFs;
use serde_json::json;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
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
    /// Images collected from the parent's live history at spawn time.
    /// Resolved by [`prepare_child_images`] before the child's first turn.
    pub images: Vec<ImageAttachment>,
    /// Override the LLM provider for this child session.
    pub provider_override: Option<ProviderKind>,
    /// Override the model name for this child session.
    pub model_override: Option<String>,
    /// Optional specialist agent name. When set, the matching agent profile is
    /// loaded to override provider/model/system prompt for this child.
    pub specialist: Option<String>,
    /// Parent's subagent task registry (P1 read-only introspection). When
    /// set, spawn/terminal lifecycle transitions are folded into it and
    /// surfaced as `ChildSessionStatusChanged` events.
    pub registry: Option<std::sync::Arc<crate::subagent_registry::SubagentRegistry>>,
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
    apply_child_routing(
        &mut child_config,
        cfg.specialist.as_deref(),
        cfg.provider_override,
        cfg.model_override.as_deref(),
    );

    // Forward the parent's images only when the routed provider+model accepts
    // native image input; make paths absolute against the PARENT workspace
    // root so a worktree-rooted child still resolves the parent's `.nca`
    // attachment tree (which does not exist inside the worktree).
    //
    // Images the task itself references (in `task` text or `focus_files`) are
    // attached alongside anything inherited from the parent conversation, so
    // path-based delegation ("analyze /path/to/shot.png") reaches the child
    // the same way a pasted image does. The child's provider then handles
    // them natively or via request-time VLM materialization (MiniMax
    // coding_plan/vlm) — the same mechanism as a user-pasted `/image`.
    let task_images =
        collect_task_image_references(&cfg.task, &cfg.focus_files, &cfg.workspace_root);
    let task_image_count = task_images.len();
    let forwarded = merge_spawn_images(&cfg.images, task_images, &cfg.workspace_root);
    let (images, image_note) = prepare_child_images(
        &forwarded,
        &cfg.workspace_root,
        model_accepts_native_images(
            child_config.provider.default,
            child_config.provider.active_model(),
        ),
    );

    let mut sup = Supervisor::create(SupervisorConfig {
        config: child_config,
        workspace_root: cfg.workspace_root.clone(),
        safe_mode: false,
        interactive_approvals: false,
        session_id: None,
        approval_handler: Some(Arc::new(AutoDenyHandler) as Arc<dyn ApprovalHandler>),
        orchestration_context: None,
        agent_name: cfg.specialist.clone(),
        provider: None,
    })
    .await
    .map_err(|e| e.to_string())?;

    // Child sessions are non-interactive: `QuestionRequested` is NOT
    // forwarded to the parent UI (only activity lines are), so a child that
    // called `ask_question` would block forever on an oneshot nobody can
    // answer — freezing both the child and the parent turn awaiting it.
    // Strip the tool; the model gets a normal "unknown tool" error it can
    // recover from instead of an invisible hang.
    sup.agent_mut().tools.unregister("ask_question");

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

    // P1 read-only introspection: fold the spawn into the parent's registry
    // and surface the lifecycle transition on the same bounded event channel
    // (the registry is also re-derivable from these events at resume).
    let terminal_tx = event_tx.clone();
    if let Some(ref registry) = cfg.registry {
        registry.record_spawned(
            &cfg.parent_session_id,
            &child_id,
            &cfg.task,
            sup.workspace_root.display().to_string(),
            sup.branch.clone(),
        );
        if let Some(ref tx) = event_tx {
            let _ = tx
                .send(AgentEvent::ChildSessionStatusChanged {
                    parent_session_id: cfg.parent_session_id.clone(),
                    child_session_id: child_id.clone(),
                    state: nca_common::session::ChildSessionState::Running,
                    generation: 0,
                    alias: None,
                    result_summary: None,
                })
                .await;
        }
    }

    let mut context_prompt = build_context_prompt(&cfg.parent_summary, &cfg.task, &cfg.focus_files);
    if task_image_count > 0 && !images.is_empty() {
        context_prompt.push_str(&format!(
            "\n\n{task_image_count} image file(s) referenced by the task are attached \
             to this message."
        ));
    }
    if let Some(note) = image_note {
        context_prompt.push_str("\n\n");
        context_prompt.push_str(&note);
    }

    let mut handle = sup.take_handle();
    let event_rx = handle.take_event_rx();
    let log_path = handle.event_log_path.clone();

    let commit_tx = handle.take_turn_commit_tx().map(|(tx, _flag)| tx);
    let parent_forward = event_tx.map(|tx| (child_id.clone(), tx));
    let mut fanout =
        event_rx.map(|rx| spawn_event_fanout(rx, log_path, None, None, parent_forward, commit_tx));

    let result = if images.is_empty() {
        sup.run_turn(&context_prompt).await
    } else {
        sup.run_turn_with_images(&context_prompt, &images).await
    };

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

    // P1 read-only introspection: fold the terminal transition into the
    // parent's registry (bounded summary) and emit the matching
    // `ChildSessionStatusChanged` before the drain — same event channel the
    // spawn events used. Foreground reply behavior is unchanged.
    if let Some(ref registry) = cfg.registry {
        let state = nca_common::session::ChildSessionState::from_spawn_status(&status);
        let result_summary = nca_core::agent::truncate_str(output.trim(), 300);
        let result_summary = if result_summary.is_empty() {
            None
        } else {
            Some(result_summary)
        };
        registry.record_terminal(&child_id, state, result_summary.clone());
        if let Some(ref tx) = terminal_tx {
            let _ = tx
                .send(AgentEvent::ChildSessionStatusChanged {
                    parent_session_id: cfg.parent_session_id.clone(),
                    child_session_id: child_id.clone(),
                    state,
                    generation: 0,
                    alias: None,
                    result_summary,
                })
                .await;
        }
    }

    // Snapshot meta BEFORE dropping the supervisor: `switch_to_worktree`
    // mutates `sup.workspace_root` (supervisor.rs) and the result reads it,
    // while `drop(sup)` moves `sup` out of reach.
    let branch = sup.branch.clone();
    let wt_path = sup.worktree_path.clone().map(|p| p.display().to_string());
    let workspace_root = sup.workspace_root.display().to_string();

    // Graceful drain: dropping `sup` closes the child's event channel (all
    // senders live inside `sup.agent`), letting the fanout drain its buffer
    // and hit the final commit-on-close. Bounded wait — liveness over
    // completeness at shutdown.
    drop(sup);
    if let Some(ref mut f) = fanout {
        crate::session_utils::drain_event_fanout(f, "child session").await;
    }

    Ok(ChildSessionResult {
        child_session_id: child_id,
        status,
        output,
        workspace: workspace_root,
        branch,
        worktree_path: wt_path,
    })
}

/// Build the child's first user message: parent context, task, focus files.
/// Deliberately carries NO specialist persona — the persona lives only in
/// the system prompt via `SupervisorConfig::agent_name`.
fn build_context_prompt(parent_summary: &str, task: &str, focus_files: &[String]) -> String {
    let mut prompt = format!(
        "You are a sub-agent spawned by a parent session to handle a specific task.\n\n\
         ## Parent Context\n{parent_summary}\n\n\
         ## Your Task\n{task}"
    );
    if !focus_files.is_empty() {
        prompt.push_str("\n\n## Focus Files\n");
        for f in focus_files {
            prompt.push_str(&format!("- {f}\n"));
        }
    }
    prompt
}

/// Resolve parent-forwarded images for a child session's first turn.
///
/// - No vision input → all images dropped with an explanatory note, so the
///   child knows why it cannot see the attachments instead of guessing.
/// - Vision input → workspace-relative paths are made absolute against the
///   PARENT workspace root (the child may run in a worktree whose root does
///   not contain the parent's `.nca` attachment tree); references whose file
///   no longer exists are dropped.
fn prepare_child_images(
    images: &[ImageAttachment],
    parent_workspace_root: &Path,
    vision_capable: bool,
) -> (Vec<ImageAttachment>, Option<String>) {
    if images.is_empty() {
        return (Vec::new(), None);
    }
    if !vision_capable {
        return (
            Vec::new(),
            Some(format!(
                "Note: {} image attachment(s) from the parent conversation were not forwarded — \
                 the current model has no vision input.",
                images.len()
            )),
        );
    }
    let mut kept = Vec::with_capacity(images.len());
    let mut missing = 0usize;
    for attachment in images {
        let path = if Path::new(&attachment.path).is_absolute() {
            attachment.path.clone()
        } else {
            parent_workspace_root
                .join(&attachment.path)
                .display()
                .to_string()
        };
        if Path::new(&path).exists() {
            kept.push(ImageAttachment {
                media_type: attachment.media_type.clone(),
                path,
            });
        } else {
            missing += 1;
        }
    }
    let note = match (kept.is_empty(), missing) {
        (true, _) => Some(
            "Note: the parent's image attachment(s) no longer exist on disk; \
             work from the task text alone."
                .to_string(),
        ),
        (false, 0) => None,
        (false, n) => Some(format!(
            "Note: {n} of the parent's image attachment(s) no longer exist on disk and were omitted."
        )),
    };
    (kept, note)
}

/// Media type for an image file path, by extension (case-insensitive).
fn image_media_type(path: &str) -> Option<&'static str> {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())?;
    match ext.as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        "bmp" => Some("image/bmp"),
        _ => None,
    }
}

/// Strip the punctuation chat text tends to wrap around inline paths
/// (backticks, quotes, brackets, trailing punctuation).
fn unwrap_path_token(token: &str) -> &str {
    token.trim_matches(|c: char| {
        matches!(
            c,
            '`' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | ':' | '.' | '!'
        )
    })
}

/// Image files explicitly referenced by a spawn request — in the task text or
/// in `focus_files` — resolved against the parent workspace root.
///
/// Only files that exist on disk are attached: a task like "create
/// assets/logo.png" must not attach anything, and mentions of deleted files
/// are ignored. Absolute and workspace-relative paths both resolve; the
/// returned paths are absolute so a worktree-rooted child still finds the
/// parent's copy (mirroring [`prepare_child_images`]).
pub(crate) fn collect_task_image_references(
    task: &str,
    focus_files: &[String],
    workspace_root: &Path,
) -> Vec<ImageAttachment> {
    let mut candidates: Vec<String> = focus_files.to_vec();
    candidates.extend(
        task.split_whitespace()
            .map(unwrap_path_token)
            .filter(|tok| image_media_type(tok).is_some())
            .map(String::from),
    );

    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for candidate in candidates {
        let Some(media_type) = image_media_type(&candidate) else {
            continue;
        };
        let resolved = if Path::new(&candidate).is_absolute() {
            candidate.clone()
        } else {
            workspace_root.join(&candidate).display().to_string()
        };
        if !Path::new(&resolved).is_file() || !seen.insert(resolved.clone()) {
            continue;
        }
        out.push(ImageAttachment {
            media_type: media_type.to_string(),
            path: resolved,
        });
        if out.len() >= MAX_FORWARD_IMAGES {
            break;
        }
    }
    out
}

/// Merge parent-history images with task-referenced ones, deduplicating by
/// absolute path (history entries are workspace-relative; task references may
/// already be absolute).
///
/// Task references come first: an explicitly named image is spawn intent and
/// survives the combined cap ahead of incidental history forwarding.
pub(crate) fn merge_spawn_images(
    history: &[ImageAttachment],
    task_refs: Vec<ImageAttachment>,
    workspace_root: &Path,
) -> Vec<ImageAttachment> {
    /// Upper bound on the combined set; matches the spirit of
    /// [`MAX_FORWARD_IMAGES`] for a single source.
    const COMBINED_CAP: usize = 16;

    let absolutize = |p: &str| {
        if Path::new(p).is_absolute() {
            p.to_string()
        } else {
            workspace_root.join(p).display().to_string()
        }
    };

    let mut seen: HashSet<String> = HashSet::new();
    for att in &task_refs {
        seen.insert(absolutize(&att.path));
    }
    let mut out = task_refs;
    for att in history {
        if seen.insert(absolutize(&att.path)) {
            out.push(att.clone());
        }
    }
    out.truncate(COMBINED_CAP);
    out
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
/// The matched profile's `system_prompt` persona is applied to the child's
/// system prompt via `SupervisorConfig::agent_name` — not here.
fn apply_child_routing(
    config: &mut NcaConfig,
    specialist: Option<&str>,
    provider_override: Option<ProviderKind>,
    model_override: Option<&str>,
) {
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
}

/// Spawns a background task that consumes spawn requests from the sub-agent tool
/// and runs child sessions. Each child session inherits parent context — both
/// the text summary and any images are taken from the live `parent_history`
/// mirror at spawn time, never from a wiring-time snapshot.
#[allow(clippy::too_many_arguments)]
pub fn spawn_subagent_consumer(
    mut spawn_rx: mpsc::Receiver<SpawnRequest>,
    parent_session_id: String,
    workspace_root: PathBuf,
    config: NcaConfig,
    parent_history: Arc<Mutex<Vec<nca_common::message::Message>>>,
    event_tx: Option<tokio::sync::mpsc::Sender<AgentEvent>>,
    registry: std::sync::Arc<crate::subagent_registry::SubagentRegistry>,
    parent_fs: Arc<dyn WorkspaceFs>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(req) = spawn_rx.recv().await {
            let parent_session_id = parent_session_id.clone();
            let workspace_root = workspace_root.clone();
            // Sync runtime mounts from the parent's live FS state so that
            // paths added via `/mount` during the session are inherited by
            // child sessions. The `config` snapshot captured at consumer
            // creation does not reflect runtime mounts.
            let mut config = config.clone();
            let live_mounts = parent_fs.mounted_paths();
            if config.extra_paths != live_mounts {
                config.extra_paths = live_mounts;
            }
            let event_tx = event_tx.clone();
            let registry = registry.clone();

            // Summary of the parent conversation as of THIS spawn (the
            // supervisor refreshes the mirror at each turn start), so a child
            // spawned in turn N sees context through turn N — not just the
            // session-opening messages.
            let parent_summary = {
                let history = parent_history.lock().unwrap_or_else(|p| p.into_inner());
                build_parent_summary(&history)
            };

            let child_cfg = ChildSessionConfig {
                parent_session_id: parent_session_id.clone(),
                task: req.task.clone(),
                workspace_root: workspace_root.clone(),
                config,
                parent_summary,
                use_worktree: req.use_worktree,
                focus_files: req.focus_files,
                images: req.images.clone(),
                provider_override: req.provider_override,
                model_override: req.model_override.clone(),
                specialist: req.specialist.clone(),
                registry: Some(registry.clone()),
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
                        // Lineage is recorded via ChildSessionSpawned on the
                        // parent's event channel (folded into meta at resume —
                        // P2 Phase C §3); no direct parent-json write here.
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
                                        "workspace": res.workspace.clone(),
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
                                        "workspace": workspace_root.display().to_string(),
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

    /// The child's first user message must NOT inline the specialist
    /// persona — it lives only in the system prompt via
    /// `SupervisorConfig::agent_name` (see the companion test below).
    #[test]
    fn child_context_prompt_carries_no_specialist_persona() {
        let prompt = build_context_prompt(
            "[User]: fix the bug",
            "Do the thing",
            &["src/a.rs".to_string(), "src/b.rs".to_string()],
        );
        assert!(
            !prompt.contains("Specialist Persona"),
            "persona must live only in the system prompt, never in the context prompt"
        );
        assert!(prompt.contains("## Parent Context"));
        assert!(prompt.contains("## Your Task"));
        assert!(prompt.contains("## Focus Files"));
        assert!(prompt.contains("- src/a.rs"));
    }

    #[test]
    fn context_prompt_without_focus_files_omits_section() {
        let prompt = build_context_prompt("summary", "task", &[]);
        assert!(!prompt.contains("## Focus Files"));
    }

    #[test]
    fn agent_name_profile_puts_persona_in_system_prompt_not_context() {
        // Supervisor-side seam: when `agent_name` resolves, the persona is in
        // the child's SYSTEM prompt (via `build_system_prompt_with_agent`) —
        // the counterpart to the context-prompt exclusion above.
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = NcaConfig::default();
        config.agents.insert(
            "librarian".to_string(),
            AgentProfileConfig {
                system_prompt: Some("You are the Librarian, keeper of docs.".to_string()),
                ..Default::default()
            },
        );
        let profile = config.agent_profile("librarian").cloned();

        let system_prompt = nca_core::harness::build_system_prompt_with_agent(
            &config,
            dir.path(),
            &nca_core::plugin::PluginRegistry::new(),
            None,
            profile.as_ref(),
            &[],
        );
        assert!(system_prompt.contains("You are the Librarian, keeper of docs."));
        assert!(
            !system_prompt.contains("Specialist Persona"),
            "persona is the system prompt itself — no inline marker anywhere"
        );
    }

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
        apply_child_routing(
            &mut config,
            Some("librarian"),
            Some(ProviderKind::OpenAi),
            Some("gpt-4o"),
        );
        assert_eq!(config.provider.default, ProviderKind::ZhipuAI);
        assert_eq!(config.provider.active_model(), "glm-4.7-flash");
        assert_eq!(config.model.default_model, "glm-4.7-flash");
    }

    #[test]
    fn specialist_profile_applied_without_overrides() {
        let mut config = config_with_librarian_profile();
        apply_child_routing(&mut config, Some("librarian"), None, None);
        assert_eq!(config.provider.default, ProviderKind::ZhipuAI);
        assert_eq!(config.model.default_model, "glm-4.7-flash");
    }

    #[test]
    fn no_specialist_honors_overrides() {
        let mut config = NcaConfig::default();
        apply_child_routing(
            &mut config,
            None,
            Some(ProviderKind::ZhipuAI),
            Some("glm-4.7-flash"),
        );
        assert_eq!(config.provider.default, ProviderKind::ZhipuAI);
        assert_eq!(config.provider.active_model(), "glm-4.7-flash");
        assert_eq!(config.model.default_model, "glm-4.7-flash");
    }

    #[test]
    fn unknown_specialist_falls_back_to_overrides() {
        let mut config = NcaConfig::default();
        apply_child_routing(
            &mut config,
            Some("does-not-exist"),
            Some(ProviderKind::DeepSeek),
            None,
        );
        assert_eq!(config.provider.default, ProviderKind::DeepSeek);
    }

    fn attachment(path: &str) -> ImageAttachment {
        ImageAttachment {
            media_type: "image/png".into(),
            path: path.into(),
        }
    }

    #[test]
    fn prepare_child_images_empty_is_noop() {
        let (kept, note) = prepare_child_images(&[], Path::new("/ws"), false);
        assert!(kept.is_empty());
        assert!(note.is_none());
    }

    #[test]
    fn prepare_child_images_drops_all_without_vision() {
        let images = vec![attachment("a.png"), attachment("b.png")];
        let (kept, note) = prepare_child_images(&images, Path::new("/ws"), false);
        assert!(kept.is_empty());
        let note = note.expect("note must explain the omission");
        assert!(note.contains("no vision input"), "got: {note}");
        assert!(note.contains('2'), "must count the images: {note}");
    }

    #[test]
    fn prepare_child_images_absolutizes_against_parent_root_and_drops_missing() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("a.png"), b"png").expect("write image");
        let images = vec![attachment("a.png"), attachment("gone.png")];

        let (kept, note) = prepare_child_images(&images, dir.path(), true);

        assert_eq!(kept.len(), 1);
        let expected = dir.path().join("a.png");
        assert_eq!(kept[0].path, expected.display().to_string());
        assert!(Path::new(&kept[0].path).is_absolute());
        let note = note.expect("note must report the missing file");
        assert!(note.contains("no longer exist"), "got: {note}");
    }

    #[test]
    fn task_image_references_found_in_task_and_focus_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("a.png"), b"png").expect("write image");
        std::fs::create_dir_all(dir.path().join("sub")).expect("mkdir");
        std::fs::write(dir.path().join("sub/b.jpg"), b"jpg").expect("write image");
        std::fs::write(dir.path().join("notes.rs"), b"rust").expect("write text");

        let task = "Analyze `a.png` and sub/b.jpg, plus missing.png and notes.rs";
        let refs = collect_task_image_references(task, &["notes.rs".to_string()], dir.path());

        assert_eq!(refs.len(), 2, "only existing image files attach: {refs:?}");
        let png = refs
            .iter()
            .find(|r| r.path.ends_with("a.png"))
            .expect("png");
        assert_eq!(png.media_type, "image/png");
        let jpg = refs
            .iter()
            .find(|r| r.path.ends_with("sub/b.jpg"))
            .expect("jpg");
        assert_eq!(jpg.media_type, "image/jpeg");
        assert!(refs.iter().all(|r| Path::new(&r.path).is_absolute()));
    }

    #[test]
    fn task_image_references_ignore_missing_files_and_dedup() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("shot.png"), b"png").expect("write image");

        // Same file via backticked mention, bare mention, focus_files, and an
        // absolute path — one attachment. Hypothetical files never attach.
        let absolute = dir.path().join("shot.png").display().to_string();
        let task = format!("look at `shot.png`, then shot.png: ({absolute}) and gone.png");
        let refs = collect_task_image_references(
            &task,
            &["shot.png".to_string(), "assets/future.png".to_string()],
            dir.path(),
        );

        assert_eq!(refs.len(), 1, "deduped to one attachment: {refs:?}");
        assert_eq!(refs[0].path, absolute);
    }

    #[test]
    fn merge_spawn_images_prefers_task_refs_and_dedups_by_absolute_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join(".nca/att")).expect("mkdir");
        std::fs::write(dir.path().join(".nca/att/x.png"), b"png").expect("write image");

        // History entry is workspace-relative; task ref points at the same
        // file via an absolute path — merged set must keep one entry.
        let history = vec![attachment(".nca/att/x.png")];
        let task_refs = vec![ImageAttachment {
            media_type: "image/png".into(),
            path: dir.path().join(".nca/att/x.png").display().to_string(),
        }];

        let merged = merge_spawn_images(&history, task_refs, dir.path());

        assert_eq!(merged.len(), 1);
        assert!(merged[0].path.ends_with("x.png"));
    }
}
