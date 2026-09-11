use crate::session_utils::spawn_event_fanout;
use crate::supervisor::{AutoDenyHandler, Supervisor, SupervisorConfig};
use crate::wake_scheduler::WakeScheduler;
use nca_common::config::{NcaConfig, ProviderKind};
use nca_common::event::{AgentEvent, EndReason};
use nca_common::message::ImageAttachment;
use nca_common::model_caps::model_accepts_native_images;
use nca_common::session::ChildSessionState;
use nca_core::approval::ApprovalHandler;
use nca_core::hooks::{HookEventKind, HookRunner};
use nca_core::tools::spawn_subagent::{MAX_FORWARD_IMAGES, SpawnRequest};
use nca_core::workspace_fs::WorkspaceFs;
use serde_json::json;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
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
    /// Parent-scoped alias for the `task_*` control tools (P2). Recorded on
    /// the registry entry and surfaced through the running
    /// `ChildSessionStatusChanged` event (the spawn event itself carries no
    /// alias field).
    pub alias: Option<String>,
    /// Parent's subagent task registry (P1 read-only introspection). When
    /// set, spawn/terminal lifecycle transitions are folded into it and
    /// surfaced as `ChildSessionStatusChanged` events.
    pub registry: Option<std::sync::Arc<crate::subagent_registry::SubagentRegistry>>,
    /// Optional pre-built provider, used verbatim by the child supervisor
    /// (`build_provider` skipped) — test seam mirroring
    /// [`crate::supervisor::SupervisorConfig::provider`]. Production
    /// callers pass `None`.
    pub provider: Option<Arc<dyn nca_core::provider::Provider>>,
    /// Parent's plugin registry (G3). When set, `prepare_child_session`
    /// fires the `subagentDispatch` hook against the PARENT-rooted plugin
    /// instances and appends returned context blocks to the child's task
    /// prompt (host-enforced size cap from `[plugins]`).
    pub plugins: Option<Arc<nca_core::plugin::PluginRegistry>>,
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

/// Wake-text state label for a terminal child: the `ChildSessionState`
/// rendered as its wire word (matches the spec §3 state vocabulary —
/// notably `failed`, not the internal `error` status string).
fn child_state_label(state: ChildSessionState) -> &'static str {
    match state {
        ChildSessionState::Pending => "pending",
        ChildSessionState::Running => "running",
        ChildSessionState::Completed => "completed",
        ChildSessionState::Cancelled => "cancelled",
        ChildSessionState::Failed => "failed",
    }
}

/// Whether a failed child turn was a cooperative cancellation: either the
/// child's cancel flag is set (the registry-visible `task_cancel` handle)
/// or the agent driver surfaced its canonical "run cancelled" error.
fn turn_was_cancelled(err: &nca_core::provider::ProviderError, cancel_flag_set: bool) -> bool {
    cancel_flag_set || err.to_string().contains("run cancelled")
}

/// Terminal `result_summary` for a child task. A cancelled task folds the
/// cancel reason recorded by `task_cancel` into `"cancelled: <reason>"`
/// (plain `"cancelled"` when no reason was recorded); any other status
/// summarizes (and truncates) the turn output, yielding `None` for
/// whitespace-only output.
fn terminal_result_summary(
    status: &str,
    output: &str,
    cancel_reason: Option<&str>,
) -> Option<String> {
    if status == "cancelled" {
        return Some(match cancel_reason {
            Some(reason) => format!("cancelled: {reason}"),
            None => "cancelled".to_string(),
        });
    }
    let summary = nca_core::agent::truncate_str(output.trim(), 300);
    (!summary.is_empty()).then_some(summary)
}

/// Everything `run_prepared_child` needs after the prepare phase: the
/// built context prompt + images, registry/event handles for lifecycle
/// folding, and the generation the run reports in
/// `ChildSessionStatusChanged` events (0 for a fresh spawn; a revived child
/// passes its bumped generation).
pub struct PreparedChild {
    pub child_id: String,
    pub parent_session_id: String,
    pub context_prompt: String,
    pub images: Vec<ImageAttachment>,
    pub generation: u64,
    pub registry: Option<Arc<crate::subagent_registry::SubagentRegistry>>,
    /// Event channel for terminal `ChildSessionStatusChanged` emissions
    /// (clone of the parent's bounded channel).
    pub terminal_tx: Option<mpsc::Sender<AgentEvent>>,
}

/// Remove every parent-only tool from a child session's registry (both the
/// spawn and revive paths).
///
/// Why each family is parent-only:
/// - `ask_question` / `wait_for_user`: a child's `QuestionRequested` is not
///   forwarded to any UI and nobody can answer its oneshot — an invisible
///   hang. `wait_for_user` is defensive only: it is never registered on
///   children (registration is the top-level TUI gate).
/// - `spawn_subagent`: children have no spawn consumer of their own, so a
///   grandchild spawn would park on the oneshot reply for the full ~600s
///   tool window — `spawn_rx` alive but undrained is a REAL hang, not an
///   error.
/// - `task_status`/`task_result`/`task_message`/`task_cancel`/
///   `task_revive`: these resolve against the child's own EMPTY registry
///   and would fail fast with "unknown task id" anyway — stripped as
///   hygiene so the model gets a plain unknown-tool error instead.
///
/// Precedent (ask_question): an unknown-tool error is recoverable; an
/// invisible hang is not.
fn strip_child_only_tools(tools: &mut nca_core::tools::ToolRegistry) {
    const PARENT_ONLY_TOOLS: &[&str] = &[
        "ask_question",
        "wait_for_user",
        "spawn_subagent",
        "task_status",
        "task_result",
        "task_message",
        "task_cancel",
        "task_revive",
    ];
    for name in PARENT_ONLY_TOOLS {
        tools.unregister(name);
    }
}

/// Phase 1 of a child spawn: everything up to (but not including) the
/// first `run_turn` — routing config, image resolution,
/// `Supervisor::create`, parent-only tool strip
/// ([`strip_child_only_tools`]), parent linkage, worktree
/// create+switch, spawn/running events, registry record (handles + alias +
/// specialist + worktree path), and the built context prompt. Returns the
/// live supervisor plus a [`PreparedChild`] for [`run_prepared_child`].
///
/// Split from the old monolithic `spawn_child_session` so a
/// `background: true` spawn can reply IMMEDIATELY after this phase and run
/// the turn detached (P2 §6: the oneshot is never held for the foreground
/// 600s window).
pub async fn prepare_child_session(
    cfg: ChildSessionConfig,
    event_tx: Option<tokio::sync::mpsc::Sender<AgentEvent>>,
) -> Result<(Supervisor, PreparedChild), String> {
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
        provider: cfg.provider,
    })
    .await
    .map_err(|e| e.to_string())?;

    // Parent-only tools never exist on a child: a question nobody can
    // answer would hang the child (and the parent turn awaiting it) forever;
    // a grandchild spawn would park on an undrained oneshot for the full
    // 600s window; the task_* tools would only error against the child's
    // empty registry. The model gets a normal "unknown tool" error it can
    // recover from instead of invisible hangs.
    strip_child_only_tools(&mut sup.agent_mut().tools);

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

    // P1 read-only introspection + P2 live control handles: fold the spawn
    // into the parent's registry, record the child's cancel flag + inbox
    // sender (driving `task_cancel`/`task_message`), and surface the
    // lifecycle transition on the same bounded event channel (the registry
    // is also re-derivable from these events at resume — handles are not,
    // they are runtime-only state).
    let terminal_tx = event_tx.clone();
    if let Some(ref registry) = cfg.registry {
        registry.record_spawned(
            &cfg.parent_session_id,
            &child_id,
            &cfg.task,
            sup.workspace_root.display().to_string(),
            sup.branch.clone(),
        );
        registry.record_handles(&child_id, sup.cancel_handle(), sup.inbox_sender());
        // `ChildSessionSpawned` has no alias/specialist/worktree fields —
        // they are runtime-only entry state, set post-record. The alias
        // rides the running `ChildSessionStatusChanged` event so the
        // registry stays re-derivable from the event log at resume.
        registry.set_alias(&child_id, cfg.alias.as_deref());
        registry.set_specialist(&child_id, cfg.specialist.clone());
        registry.set_worktree(
            &child_id,
            sup.worktree_path.as_ref().map(|p| p.display().to_string()),
        );
        if let Some(ref tx) = event_tx {
            let _ = tx
                .send(AgentEvent::ChildSessionStatusChanged {
                    parent_session_id: cfg.parent_session_id.clone(),
                    child_session_id: child_id.clone(),
                    state: nca_common::session::ChildSessionState::Running,
                    generation: 0,
                    alias: cfg.alias.clone(),
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

    // G3: dispatch-time plugin augmentation. The hook runs HERE — in the
    // parent's process, against parent-rooted plugin instances — so the
    // plugin sees the real workspace root even when the child will live in
    // a worktree. Returned blocks are appended to the task prompt under a
    // host-enforced byte cap (overflow truncates with a marker).
    if let Some(plugins) = cfg.plugins.as_ref()
        && !plugins.is_empty()
    {
        let blocks = plugins.collect_subagent_dispatch(
            cfg.specialist.as_deref().unwrap_or(""),
            &cfg.task,
            cfg.use_worktree,
        );
        if !blocks.is_empty() {
            let cap = cfg.config.plugins.subagent_context_max_bytes;
            let mut used = 0usize;
            let mut appended = String::new();
            for (name, text) in &blocks {
                let block =
                    format!("\n\n<plugin-context source=\"{name}\">\n{text}\n</plugin-context>");
                let budget = cap.saturating_sub(used);
                if block.len() <= budget {
                    used += block.len();
                    appended.push_str(&block);
                } else {
                    // Byte-safe truncation: accumulate chars while the byte
                    // budget holds (chars can be multi-byte).
                    let mut bytes = 0usize;
                    let truncated: String = block
                        .chars()
                        .take_while(|c| {
                            bytes += c.len_utf8();
                            bytes <= budget
                        })
                        .collect();
                    used = cap;
                    appended.push_str(&truncated);
                    appended.push_str("\n<!-- plugin context truncated at host cap -->");
                    tracing::warn!(
                        "subagent dispatch context from {name} truncated at {cap}-byte host cap"
                    );
                    break;
                }
            }
            tracing::info!(
                "subagent dispatch augmented by {} plugin block(s), {used}/{} bytes",
                blocks.len(),
                cap
            );
            context_prompt.push_str(&appended);
        }
    }

    Ok((
        sup,
        PreparedChild {
            child_id,
            parent_session_id: cfg.parent_session_id,
            context_prompt,
            images,
            generation: 0,
            registry: cfg.registry,
            terminal_tx,
        },
    ))
}

/// Phase 2 of a child spawn: run the prepared first turn to a terminal
/// state and fold the lifecycle transitions (registry + events + session
/// json via the child's own `finish`). Returns the terminal result — errors
/// are mapped into `status: "error"`/`"cancelled"`, never propagated.
/// Used verbatim by the foreground path, the background (detached) path,
/// and `task_revive` (with a bumped generation).
pub async fn run_prepared_child(
    mut sup: Supervisor,
    prepared: PreparedChild,
) -> ChildSessionResult {
    let PreparedChild {
        child_id,
        parent_session_id,
        context_prompt,
        images,
        generation,
        registry,
        terminal_tx,
    } = prepared;

    let mut handle = sup.take_handle();
    let event_rx = handle.take_event_rx();
    let log_path = handle.event_log_path.clone();

    let commit_tx = handle.take_turn_commit_tx().map(|(tx, _flag)| tx);
    let parent_forward = terminal_tx.clone().map(|tx| (child_id.clone(), tx));
    let mut fanout =
        event_rx.map(|rx| spawn_event_fanout(rx, log_path, None, None, parent_forward, commit_tx));

    let result = if images.is_empty() {
        sup.run_turn(&context_prompt).await
    } else {
        sup.run_turn_with_images(&context_prompt, &images).await
    };

    // Cancellation classification: `task_cancel` flips the child's cancel
    // flag and the driver aborts cooperatively ("run cancelled") — such a
    // child ends Cancelled (session json + registry + events), never Error.
    // The flag check covers aborts whose error text differs (e.g. a provider
    // error racing the flag); the text check covers a cancelled run whose
    // flag a fresh generation could have cleared.
    let child_cancel_flag = sup.cancel_handle();
    let was_cancelled = match &result {
        Ok(_) => false,
        Err(e) => turn_was_cancelled(e, child_cancel_flag.load(Ordering::SeqCst)),
    };
    let (status, output) = match result {
        Ok(text) => {
            sup.finish(EndReason::Completed).await;
            ("completed".to_string(), text)
        }
        Err(e) if was_cancelled => {
            sup.finish(EndReason::Cancelled).await;
            ("cancelled".to_string(), e.to_string())
        }
        Err(e) => {
            sup.finish(EndReason::Error).await;
            ("error".to_string(), e.to_string())
        }
    };

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

    // P1 read-only introspection + P2 cancel bookkeeping: fold the terminal
    // transition into the parent's registry (bounded summary — a cancelled
    // task folds its recorded reason into "cancelled: <reason>"; live
    // handles + the spent reason are cleared inside `record_terminal`) and
    // emit the matching `ChildSessionStatusChanged`. This runs AFTER the
    // drain so "registry says terminal" implies "child fully drained" —
    // `task_revive` polls exactly that state before resuming the child's
    // session json/event log, and resuming mid-drain could read a truncated
    // replay projection.
    if let Some(ref registry) = registry {
        let state = nca_common::session::ChildSessionState::from_spawn_status(&status);
        let cancel_reason = registry
            .get(&child_id)
            .and_then(|entry| entry.cancel_reason.clone());
        let result_summary = terminal_result_summary(&status, &output, cancel_reason.as_deref());
        registry.record_terminal(&child_id, state, result_summary.clone());
        if let Some(ref tx) = terminal_tx {
            let _ = tx
                .send(AgentEvent::ChildSessionStatusChanged {
                    parent_session_id: parent_session_id.clone(),
                    child_session_id: child_id.clone(),
                    state,
                    generation,
                    alias: registry.get(&child_id).and_then(|e| e.alias),
                    result_summary,
                })
                .await;
        }
    }

    ChildSessionResult {
        child_session_id: child_id,
        status,
        output,
        workspace: workspace_root,
        branch,
        worktree_path: wt_path,
    }
}

/// Spawn a child session that inherits parent context and runs to completion.
/// Returns the result of the child run. This is a blocking async call —
/// the foreground composition of [`prepare_child_session`] +
/// [`run_prepared_child`] (P2 chunk C split; behavior unchanged).
pub async fn spawn_child_session(
    cfg: ChildSessionConfig,
    event_tx: Option<tokio::sync::mpsc::Sender<AgentEvent>>,
) -> Result<ChildSessionResult, String> {
    let (sup, prepared) = prepare_child_session(cfg, event_tx).await?;
    Ok(run_prepared_child(sup, prepared).await)
}

/// Execute a `task_revive` request (P2 C2, §3 "task_revive"): resume a
/// terminal child in its retained session + worktree with a new prompt and
/// run that turn to a new terminal state, bumping the generation.
///
/// Order of operations (port of upstream `task-revive.ts`):
/// 1. resolve id/alias → unknown/ambiguous error reply;
/// 2. acquire the control lease for the WHOLE cancel-then-run sequence
///    (a concurrent cancel/revive reports "in flight" instead);
/// 3. a still-`Running` target is cancelled first (cooperative flag flip)
///    and awaited to terminal — bounded 30s;
/// 4. `Supervisor::resume` folds the child's own json + event log (the
///    single writer of that json is the child's own supervisor — this new
///    instance replaces the finished one), re-attaches the retained
///    worktree, and `ask_question` is stripped like any child;
/// 5. `record_revive` (state=Running, generation+=1) + fresh handles +
///    running `ChildSessionStatusChanged`;
/// 6. the new turn runs through [`run_prepared_child`] with the bumped
///    generation (same terminal mapping as a spawn).
///
/// `provider` is a test seam (used verbatim by the resumed supervisor,
/// mirroring `SupervisorConfig::provider`); production callers pass `None`.
/// Runs LONG — the control consumer invokes it on its own tokio task so
/// its loop keeps serving status/result/message/cancel meanwhile; the
/// reply rides the request's oneshot from inside that task.
pub async fn handle_revive_request(
    registry: Arc<crate::subagent_registry::SubagentRegistry>,
    config: NcaConfig,
    workspace_root: PathBuf,
    event_tx: Option<tokio::sync::mpsc::Sender<AgentEvent>>,
    provider: Option<Arc<dyn nca_core::provider::Provider>>,
    session_id: String,
    prompt: String,
) -> nca_core::tools::subagent_control::SubagentControlResponse {
    use nca_common::session::ChildSessionState;
    use nca_core::tools::subagent_control::SubagentControlResponse;

    let entry = match registry.resolve(&session_id) {
        Ok(Some(entry)) => entry,
        Ok(None) => {
            return SubagentControlResponse::unknown(&session_id, "unknown subagent task id");
        }
        Err(message) => return SubagentControlResponse::unknown(&session_id, message),
    };
    let child_id = entry.session_id.clone();
    let base =
        |state, note: Option<String>, ok: bool, error: Option<String>| SubagentControlResponse {
            session_id: child_id.clone(),
            state,
            task: None,
            workspace: None,
            branch: None,
            result_summary: None,
            output: None,
            note,
            ok,
            error_message: error,
            generation: None,
        };

    // The lease is held for the WHOLE revive (cancel-wait + resume + run):
    // a concurrent cancel/revive must not interleave with this sequence.
    let Some(_lease) = registry.try_acquire_lease(&child_id) else {
        return base(
            entry.state,
            Some("another control operation is in flight for this task".into()),
            false,
            None,
        );
    };

    // A still-running target is cancelled first and awaited to terminal —
    // revive never runs a second supervisor for a live child.
    if entry.state == ChildSessionState::Running {
        if let Some(flag) = entry.cancel_flag.as_ref() {
            flag.store(true, Ordering::SeqCst);
            // Fold a reason so the old child's terminal summary reads as a
            // revive-triggered cancel (mirrors task_cancel bookkeeping).
            registry.record_cancel_requested(&child_id, Some("revive".into()));
        }
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let terminal = registry
                .get(&child_id)
                .map(|e| e.state.is_terminal())
                .unwrap_or(false);
            if terminal {
                break;
            }
            if Instant::now() >= deadline {
                return base(
                    ChildSessionState::Running,
                    Some("cancel before revive did not complete in 30s".into()),
                    false,
                    None,
                );
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    // Rebuild the child config exactly like the spawn path: children run
    // BypassPermissions with the same routing treatment (specialist profile
    // authoritative; meta.agent_name re-applies the persona inside resume).
    let mut child_config = config;
    child_config.permissions.mode = nca_common::config::PermissionMode::BypassPermissions;
    apply_child_routing(&mut child_config, entry.specialist.as_deref(), None, None);

    let mut sup = match Supervisor::resume(
        child_config,
        &workspace_root,
        false,
        false,
        &child_id,
        Some(Arc::new(AutoDenyHandler) as Arc<dyn ApprovalHandler>),
        provider,
    )
    .await
    {
        Ok(sup) => sup,
        Err(e) => {
            // Registry stays terminal — no zombie Running entry.
            return base(
                entry.state,
                None,
                false,
                Some(format!("failed to resume child session: {e}")),
            );
        }
    };

    // Parent-only tool strip, same as the spawn path (see
    // [`strip_child_only_tools`]) — a revived child is just as non-
    // interactive and has no spawn consumer as a fresh one.
    strip_child_only_tools(&mut sup.agent_mut().tools);

    // The retained worktree: `Supervisor::resume` restores the worktree
    // FIELDS from the child's meta but NOT the fs/pty cwd (pinned by
    // `resume_restores_worktree_fields_but_not_fs_root`) — switch
    // explicitly so the revived child's file tools + shell run inside the
    // retained worktree. Cancel never deleted it; revive reuses it as-is.
    if let Some(worktree_path) = sup.worktree_path.clone() {
        let branch = sup.branch.clone().unwrap_or_default();
        let base_branch = sup.base_branch.clone().unwrap_or_default();
        sup.switch_to_worktree(worktree_path, branch, base_branch);
    }

    let Some(generation) = registry.record_revive(&child_id) else {
        return SubagentControlResponse::unknown(&child_id, "unknown subagent task id");
    };
    registry.record_handles(&child_id, sup.cancel_handle(), sup.inbox_sender());
    if let Some(ref tx) = event_tx {
        let _ = tx
            .send(AgentEvent::ChildSessionStatusChanged {
                parent_session_id: entry.parent_session_id.clone(),
                child_session_id: child_id.clone(),
                state: ChildSessionState::Running,
                generation,
                alias: entry.alias.clone(),
                result_summary: None,
            })
            .await;
    }

    let result = run_prepared_child(
        sup,
        PreparedChild {
            child_id: child_id.clone(),
            parent_session_id: entry.parent_session_id.clone(),
            context_prompt: prompt,
            images: Vec::new(),
            generation,
            registry: Some(registry.clone()),
            terminal_tx: event_tx.clone(),
        },
    )
    .await;

    // Lineage event for the resumed run (same channel the spawn path uses —
    // folded into the parent meta at resume).
    if let Some(ref tx) = event_tx {
        let _ = tx
            .send(AgentEvent::ChildSessionCompleted {
                parent_session_id: entry.parent_session_id.clone(),
                child_session_id: child_id.clone(),
                status: result.status.clone(),
            })
            .await;
    }

    let state = ChildSessionState::from_spawn_status(&result.status);
    SubagentControlResponse {
        session_id: child_id.clone(),
        state,
        task: None,
        workspace: Some(result.workspace),
        branch: result.branch,
        result_summary: registry.get(&child_id).and_then(|e| e.result_summary),
        output: Some(result.output),
        note: Some(format!(
            "revived task (generation {generation}) ran to a new terminal state"
        )),
        ok: true,
        error_message: None,
        generation: Some(generation),
    }
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
///
/// `child_provider` is a test seam (injected verbatim into every child,
/// mirroring `SupervisorConfig::provider`); production callers pass `None`.
///
/// `background_default` is the P3 policy default applied to spawns that
/// OMIT the `background` flag (`req.background.unwrap_or(background_default)`);
/// an explicit flag always wins. `wake`, when set, is notified when a
/// DETACHED (background) child reaches a terminal state so an idle parent
/// can be woken — foreground spawns and `task_revive` never notify (their
/// callers already hold the reply).
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
    child_provider: Option<Arc<dyn nca_core::provider::Provider>>,
    background_default: bool,
    wake: Option<WakeScheduler>,
    plugins: Option<Arc<nca_core::plugin::PluginRegistry>>,
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
            let child_provider = child_provider.clone();
            let wake = wake.clone();
            let plugins = plugins.clone();

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
                alias: req.alias.clone(),
                registry: Some(registry.clone()),
                provider: child_provider,
                plugins,
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
                match prepare_child_session(child_cfg, event_tx.clone()).await {
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
                    Ok((sup, prepared)) => {
                        // P3: absent flag inherits the caller's policy
                        // default; an explicit flag always wins.
                        if req.background.unwrap_or(background_default) {
                            // §6 invariant ("600s timeout interplay"): answer
                            // the oneshot IMMEDIATELY after prepare — the
                            // reply channel is consumed HERE and the
                            // detached run below must never send again.
                            // The parent turn ends now; the final output is
                            // fetched later via `task_result`.
                            let response = nca_core::tools::spawn_subagent::SpawnResponse {
                                child_session_id: prepared.child_id.clone(),
                                status: "running".into(),
                                output: String::new(),
                                workspace: sup.workspace_root.display().to_string(),
                                branch: sup.branch.clone(),
                                worktree_path: sup
                                    .worktree_path
                                    .as_ref()
                                    .map(|p| p.display().to_string()),
                            };
                            let _ = req.reply.send(response);
                            let parent = parent_session_id.clone();
                            let tx = event_tx.clone();
                            let hooks = hook_runner.clone();
                            let wake = wake.clone();
                            let child_ref = req
                                .alias
                                .clone()
                                .unwrap_or_else(|| prepared.child_id.clone());
                            tokio::spawn(async move {
                                let result = run_prepared_child(sup, prepared).await;
                                complete_child_request(
                                    &result,
                                    &parent,
                                    tx.as_ref(),
                                    hooks.as_ref(),
                                )
                                .await;
                                // P3 wake hook — BACKGROUND ARM ONLY: the
                                // detached child just reached a terminal
                                // state, so an idle parent is woken with
                                // one debounced reconciled prompt. Never
                                // called from the foreground arm (its
                                // output was returned inline) or
                                // `task_revive` (the parent is mid-turn
                                // holding the reply). Fires for completed,
                                // cancelled, AND failed terminals; the
                                // summary is the one `run_prepared_child`
                                // already folded into the registry.
                                if let Some(wake) = wake {
                                    let summary = registry
                                        .get(&result.child_session_id)
                                        .and_then(|e| e.result_summary.clone())
                                        .unwrap_or_else(|| "(no output)".into());
                                    wake.notify_terminal(
                                        &child_ref,
                                        child_state_label(ChildSessionState::from_spawn_status(
                                            &result.status,
                                        )),
                                        &summary,
                                    );
                                }
                            });
                        } else {
                            // Foreground: unchanged synchronous contract —
                            // await the child, then reply with its terminal
                            // output (the 600s window lives in the tool).
                            let result = run_prepared_child(sup, prepared).await;
                            complete_child_request(
                                &result,
                                &parent_session_id,
                                event_tx.as_ref(),
                                hook_runner.as_ref(),
                            )
                            .await;
                            let response = nca_core::tools::spawn_subagent::SpawnResponse {
                                child_session_id: result.child_session_id,
                                status: result.status,
                                output: result.output,
                                workspace: result.workspace,
                                branch: result.branch,
                                worktree_path: result.worktree_path,
                            };
                            let _ = req.reply.send(response);
                        }
                    }
                }
            });
        }
    })
}

/// Post-run bookkeeping shared by the foreground and background (detached)
/// child paths: the `ChildSessionCompleted` lineage event on the parent's
/// channel + `SubagentStop` hooks. The reply itself is the caller's concern
/// (foreground replies after this; background already replied at spawn).
async fn complete_child_request(
    result: &ChildSessionResult,
    parent_session_id: &str,
    event_tx: Option<&tokio::sync::mpsc::Sender<AgentEvent>>,
    hook_runner: Option<&HookRunner>,
) {
    // Lineage is recorded via ChildSessionSpawned/Completed on the parent's
    // event channel (folded into meta at resume — P2 Phase C §3); no direct
    // parent-json write here.
    if let Some(tx) = event_tx {
        let _ = tx
            .send(AgentEvent::ChildSessionCompleted {
                parent_session_id: parent_session_id.to_string(),
                child_session_id: result.child_session_id.clone(),
                status: result.status.clone(),
            })
            .await;
    }
    if let Some(hooks) = hook_runner {
        hooks
            .run_best_effort(
                HookEventKind::SubagentStop,
                None,
                &json!({
                    "parent_session_id": parent_session_id,
                    "child_session_id": result.child_session_id,
                    "status": result.status,
                    "workspace": result.workspace,
                }),
            )
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nca_common::config::AgentProfileConfig;
    use nca_common::tool::{ToolCall, ToolDefinition, ToolResult};
    use nca_core::provider::ProviderError;
    use nca_core::tools::{ToolExecutor, ToolRegistry};

    #[test]
    fn turn_was_cancelled_detects_flag_and_driver_error() {
        let cancelled_err = ProviderError::Other("run cancelled".into());
        let other_err = ProviderError::Other("provider 502".into());
        assert!(turn_was_cancelled(&cancelled_err, false));
        assert!(
            turn_was_cancelled(&other_err, true),
            "set flag wins even for a provider error"
        );
        assert!(!turn_was_cancelled(&other_err, false));
    }

    #[test]
    fn terminal_result_summary_folds_cancel_reason() {
        assert_eq!(
            terminal_result_summary("cancelled", "run cancelled", Some("wrong branch")).as_deref(),
            Some("cancelled: wrong branch")
        );
        assert_eq!(
            terminal_result_summary("cancelled", "run cancelled", None).as_deref(),
            Some("cancelled")
        );
    }

    #[test]
    fn terminal_result_summary_truncates_output_for_other_statuses() {
        assert_eq!(
            terminal_result_summary("completed", "all done", None).as_deref(),
            Some("all done")
        );
        assert_eq!(
            terminal_result_summary("error", "   ", None),
            None,
            "whitespace-only output yields no summary"
        );
        let long = "x".repeat(500);
        let summary = terminal_result_summary("completed", &long, None).expect("truncated summary");
        assert_eq!(
            summary.chars().count(),
            300,
            "output summaries truncate to 300 chars (299 + ellipsis): {}",
            summary.chars().rev().take(3).collect::<String>()
        );
        assert!(summary.ends_with('…'));
    }

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

    // ── M3: strip_child_only_tools ────────────────────────────────

    /// Named no-op tool so a registry can carry arbitrary tool names.
    struct NamedStubTool(&'static str);

    #[async_trait::async_trait]
    impl ToolExecutor for NamedStubTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                timeout_ms: None,
                name: self.0.into(),
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

    /// Cheapest seam for the strip: a bare registry of named stub tools —
    /// no provider, no AgentLoop, no supervisor, no tokio.
    #[test]
    fn strip_child_only_tools_removes_parent_only_tools_and_keeps_the_rest() {
        let mut tools = ToolRegistry::new();
        for name in [
            "read_file",
            "ask_question",
            "wait_for_user",
            "spawn_subagent",
            "task_status",
            "task_result",
            "task_message",
            "task_cancel",
            "task_revive",
            "write_file",
        ] {
            tools.register(Box::new(NamedStubTool(name)));
        }

        strip_child_only_tools(&mut tools);

        let names: Vec<String> = tools.definitions().iter().map(|d| d.name.clone()).collect();
        assert_eq!(
            names,
            vec!["read_file".to_string(), "write_file".to_string()],
            "only the child-legal tools survive the strip"
        );

        // Idempotent: a second strip (the revive path over an already-stripped
        // registry) is a no-op.
        strip_child_only_tools(&mut tools);
        assert_eq!(tools.definitions().len(), 2);
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
