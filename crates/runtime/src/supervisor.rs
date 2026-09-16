pub(crate) use crate::subagent::build_parent_summary;
pub use crate::subagent::{
    ChildSessionConfig, ChildSessionResult, spawn_child_session, spawn_subagent_consumer,
};

pub(crate) use crate::session_utils::{ApprovalPendingMap, QuestionPendingMap};
pub use crate::session_utils::{
    SessionControlCommand, get_last_session_id, spawn_command_consumer,
    spawn_command_consumer_with_store, spawn_event_fanout,
};

use crate::context_manager::{ContextManager, ContextManagerConfig, ContextStats};
use crate::ipc::{IpcHandle, IpcServer};
use crate::last_session::LastSessionStore;
use crate::memory_store::{MemoryNote, MemoryStore};
use crate::model_limits_api;
use crate::plugin_host::PluginHost;
use crate::pty::PtyManager;
use crate::session_store::SessionStore;
use crate::subagent_registry::{SubagentRegistry, subagent_control_consumer};
use chrono::Utc;
use nca_common::config::{AgentProfileConfig, NcaConfig};
use nca_common::event::{AgentEvent, EndReason, EventEnvelope};
use nca_common::message::{ContentPart, Message, MessageContent, Role};
use nca_common::session::{
    OrchestrationContext, SessionMeta, SessionSnapshot, SessionState, SessionStatus,
};
use nca_core::agent::AgentLoop;
use nca_core::agent_driver::InboxItem;
use nca_core::approval::{ApprovalHandler, ApprovalPolicy, ApprovalVerdict};
use nca_core::cache_keepalive;
use nca_core::harness::build_system_prompt_with_agent;
use nca_core::hooks::{HookEventKind, HookRunner};
use nca_core::middleware::default_chain;
use nca_core::plugin::PluginRegistry;
use nca_core::provider::Provider;
use nca_core::provider::ProviderError;
use nca_core::provider::factory::build_provider;
use nca_core::skills::SkillCatalog;
use nca_core::tools::AskQuestionTool;
use nca_core::tools::InvokeSkillTool;
use nca_core::tools::ToolRegistry;
use nca_core::tools::mcp::load_mcp_tools;
use nca_core::tools::spawn_subagent::{SpawnRequest, SpawnSubagentTool};
use nca_core::tools::subagent_control::{
    SubagentControlRequest, TaskCancelTool, TaskMessageTool, TaskResultTool, TaskReviveTool,
    TaskStatusTool,
};
use nca_core::tools::{TodoStore, UpdateTodosTool};
use nca_core::workspace_fs::{RealFs, WorkspaceFs};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, watch};

/// Reusable runtime supervisor that owns session lifecycle, IPC, event fanout,
/// and command handling.
pub struct Supervisor {
    pub session_id: String,
    pub workspace_root: PathBuf,
    pub model: String,
    pub created_at: chrono::DateTime<Utc>,
    status: SessionStatus,
    pid: Option<u32>,
    socket_path: Option<PathBuf>,
    agent: AgentLoop,
    session_store: SessionStore,
    ipc_handle: Option<IpcHandle>,
    event_rx: Option<mpsc::Receiver<AgentEvent>>,
    approval_pending: Option<ApprovalPendingMap>,
    question_pending: Option<QuestionPendingMap>,
    spawn_rx: Option<mpsc::Receiver<SpawnRequest>>,
    /// Live mirror of this session's conversation, refreshed at each turn
    /// start and read by [`SpawnSubagentTool`] when collecting images for a
    /// child session. `Arc`-shared so the tool sees fresh history.
    spawn_history: Arc<Mutex<Vec<Message>>>,
    /// Read-only projection of child-session lifecycle state (P1:
    /// `task_status`/`task_result` introspection). Rebuilt at resume by
    /// folding the event log; never written to session json from here.
    subagent_registry: Arc<SubagentRegistry>,
    pub(crate) worktree_path: Option<PathBuf>,
    pub(crate) branch: Option<String>,
    pub(crate) base_branch: Option<String>,
    parent_session_id: Option<String>,
    child_session_ids: Vec<String>,
    inherited_summary: Option<String>,
    spawn_reason: Option<String>,
    session_summary: Option<String>,
    session_title: Option<String>,
    orchestration: Option<OrchestrationContext>,
    config: NcaConfig,
    /// Config snapshot without agent-profile overrides applied. Used as the
    /// base when switching agents at runtime via [`apply_agent_profile`].
    base_config: NcaConfig,
    /// Active agent profile (if any). Stored so `reset_for_new_session` can
    /// rebuild the system prompt with the same specialist persona.
    agent_profile: Option<AgentProfileConfig>,
    /// Name of the active agent profile, as last selected (`[agents.<name>]`
    /// key or skill-discovered name). Persisted into `SessionMeta::agent_name`
    /// so `resume` can re-resolve the persona against the current config.
    /// Recorded verbatim, even when the name no longer resolves (resume then
    /// warns and falls back to the default harness prompt).
    active_agent_name: Option<String>,
    hooks: Option<HookRunner>,
    plugins: Arc<PluginRegistry>,
    #[allow(dead_code)] // retained for RAII — drop cleans up child plugin processes.
    plugin_host: Option<PluginHost>,
    /// Plugin names whose disable has already been surfaced to the user
    /// (G7): each plugin is reported once per disable (reset on refresh).
    plugin_disable_reported: HashSet<String>,
    context_manager: ContextManager,
    last_summary_at_tokens: usize,
    fs: Arc<dyn WorkspaceFs>,
    pty: Arc<PtyManager>,
    /// Sender half of the turn-commit watch, handed to the fanout via
    /// [`SupervisorHandle::take_turn_commit_tx`]. `None` once taken.
    turn_commit_tx: Option<(watch::Sender<u64>, Arc<AtomicBool>)>,
    /// Receiver half used by `run_turn`'s durability barrier.
    turn_commit_rx: Option<watch::Receiver<u64>>,
    /// Wiring marker: `true` once a fanout has taken the sender. The barrier
    /// skips waiting when nobody was wired (liveness over false durability).
    turn_commit_wired: Arc<AtomicBool>,
}

/// Configuration for creating a new supervised session.
pub struct SupervisorConfig {
    pub config: NcaConfig,
    pub workspace_root: PathBuf,
    pub safe_mode: bool,
    pub interactive_approvals: bool,
    pub session_id: Option<String>,
    pub approval_handler: Option<Arc<dyn ApprovalHandler>>,
    pub orchestration_context: Option<OrchestrationContext>,
    /// Optional agent profile name. When set, the matching `[agents.<name>]`
    /// profile is loaded and its provider/model/permission/tool overrides are
    /// applied to this session.
    pub agent_name: Option<String>,
    /// Optional pre-built provider. When `Some`, `create` uses it verbatim and
    /// skips `build_provider` entirely (test seam — production passes `None`).
    ///
    /// Construction-only: `apply_agent_profile` and `apply_nca_config` rebuild
    /// the provider from config and discard any injected provider.
    pub provider: Option<Arc<dyn Provider>>,
}

/// A handle returned to callers for interacting with a running supervisor.
/// The supervisor itself runs in a background task; this handle provides
/// the control surface.
pub struct SupervisorHandle {
    pub session_id: String,
    pub workspace_root: PathBuf,
    pub model: String,
    pub socket_path: Option<PathBuf>,
    pub event_log_path: PathBuf,
    event_rx: Option<mpsc::Receiver<AgentEvent>>,
    ipc_handle: Option<IpcHandle>,
    approval_pending: Option<ApprovalPendingMap>,
    question_pending: Option<QuestionPendingMap>,
    spawn_rx: Option<mpsc::Receiver<SpawnRequest>>,
    /// Turn-commit watch sender + wiring flag for the event fanout (P2
    /// Phase B). Wiring marker for the `run_turn` durability barrier.
    turn_commit_tx: Option<(watch::Sender<u64>, Arc<AtomicBool>)>,
}

impl SupervisorHandle {
    pub fn take_event_rx(&mut self) -> Option<mpsc::Receiver<AgentEvent>> {
        self.event_rx.take()
    }

    pub fn take_ipc_handle(&mut self) -> Option<IpcHandle> {
        self.ipc_handle.take()
    }

    pub fn take_approval_pending(&mut self) -> Option<ApprovalPendingMap> {
        self.approval_pending.take()
    }

    pub fn take_question_pending(&mut self) -> Option<QuestionPendingMap> {
        self.question_pending.take()
    }

    pub fn take_spawn_rx(&mut self) -> Option<mpsc::Receiver<SpawnRequest>> {
        self.spawn_rx.take()
    }

    /// Takes the turn-commit watch sender for the event fanout.
    ///
    /// Wiring marker for the `run_turn` durability barrier: taking it marks
    /// the writer as wired (flag set BEFORE returning) so `run_turn` starts
    /// waiting on turn commits as soon as anyone owns the sender.
    pub fn take_turn_commit_tx(&mut self) -> Option<(watch::Sender<u64>, Arc<AtomicBool>)> {
        self.turn_commit_tx.take().map(|(tx, flag)| {
            flag.store(true, Ordering::SeqCst);
            (tx, flag)
        })
    }
}

/// Decide which workspace root a resumed session should use.
///
/// `SessionMeta.workspace` records the canonical path captured at
/// session-create time. When the project directory has since been renamed or
/// moved, that path no longer exists; blindly adopting it makes every later
/// persistence write (`create_dir_all` in `save_workspace_file`, allow-pattern
/// and mount persistence) silently *resurrect the old directory tree* — the
/// "config saved under the old directory" bug. The freshly canonicalized root
/// the session was just loaded from is authoritative: sessions live in
/// `<workspace>/.nca/sessions`, so a successful load proves ownership. The
/// stored path is kept only when it still exists and differs for a real
/// reason (e.g. worktree-linked sessions resumed from another root).
fn resolve_resume_workspace_root(current_root: &Path, stored_root: &Path) -> PathBuf {
    if stored_root != current_root && !stored_root.exists() {
        tracing::info!(
            stale = %stored_root.display(),
            current = %current_root.display(),
            "session workspace path is stale (renamed/moved); adopting current root"
        );
        return current_root.to_path_buf();
    }
    stored_root.to_path_buf()
}

/// Which truth a resumed conversation was restored from (P2 Phase B:
/// `docs/plans/p2-phase-b-design.md` §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResumeMessageSource {
    /// The json snapshot (old-format logs that cannot fold, or a fresh log
    /// whose replay projection is empty).
    Snapshot,
    /// The event-log replay rescued a session whose json would not load.
    ReplayFallback,
    /// The event-log replay is authoritative: the log is fresh-format and
    /// folds to a non-empty projection, so it wins over the json snapshot
    /// (json is a cache; divergence is warned, not trusted).
    ReplayAuthoritative,
}

/// Normalization used for resume selection and divergence comparison:
/// drop system messages, keep everything else as-is (records are exact,
/// reasoning included).
pub(crate) fn normalize_resume_projection(messages: &[Message]) -> Vec<Message> {
    messages
        .iter()
        .filter(|m| m.role != Role::System)
        .cloned()
        .collect()
}

/// Replace image parts whose on-disk file no longer exists (deleted by
/// attachment cleanup after the message was recorded) with text placeholders
/// via [`MessageContent::strip_image_paths`]. Text-only messages untouched.
fn repair_missing_images(messages: Vec<Message>, workspace_root: &Path) -> Vec<Message> {
    let mut repaired = messages;
    for message in repaired.iter_mut() {
        let MessageContent::Parts(parts) = &message.content else {
            continue;
        };
        let missing: HashSet<String> = parts
            .iter()
            .filter_map(|part| match part {
                ContentPart::Image { path, .. } => Some(path.clone()),
                _ => None,
            })
            .filter(|path| !workspace_root.join(path).exists())
            .collect();
        if !missing.is_empty() {
            message.content.strip_image_paths(&missing);
        }
    }
    repaired
}

/// Pure decision core of the resume algorithm
/// (`docs/plans/p2-phase-b-design.md` §5 decision table).
///
/// - Both paths are normalized: system messages from the snapshot/replay are
///   dropped and the fresh system messages `create()` just pushed are
///   prepended (fixes stale/duplicated system prompts in the json).
/// - Replay wins when the log is fresh-format (`log_has_surface_events`:
///   contains `MessageRecorded`) AND folds to a non-empty projection —
///   `ReplayAuthoritative` when the json also loaded, `ReplayFallback` when
///   the json is corrupt/unusable and the replay rescues the session.
/// - Otherwise the json snapshot wins: old-format logs cannot fold (empty
///   projection by construction), and a fresh log whose every turn failed
///   legitimately folds to empty.
/// - The replay path repairs image parts whose file is gone.
pub(crate) fn select_resume_messages(
    snapshot_messages: Option<Vec<Message>>,
    replayed: Vec<Message>,
    log_has_surface_events: bool,
    fresh_system: Vec<Message>,
    workspace_root: &Path,
) -> (Vec<Message>, ResumeMessageSource) {
    let snapshot_non_system: Vec<Message> = snapshot_messages
        .as_deref()
        .map(normalize_resume_projection)
        .unwrap_or_default();

    if log_has_surface_events && !replayed.is_empty() {
        let source = if snapshot_messages.is_some() {
            ResumeMessageSource::ReplayAuthoritative
        } else {
            ResumeMessageSource::ReplayFallback
        };
        let mut out = fresh_system;
        out.extend(repair_missing_images(replayed, workspace_root));
        (out, source)
    } else {
        // Old-format log or empty replay: the json is the only foldable
        // truth. `snapshot_messages == None` here is unreachable from
        // `resume()` (it errors when json is unusable AND replay is empty);
        // defensively yields the fresh system prompt only.
        let mut out = fresh_system;
        out.extend(snapshot_non_system);
        (out, ResumeMessageSource::Snapshot)
    }
}

/// Seed the cost tracker from the last cumulative `CostUpdated` in the event
/// log (best-effort, fallback path only — the json path restores totals from
/// the snapshot).
fn seed_cost_tracker_from_log(agent: &mut AgentLoop, envelopes: &[EventEnvelope]) {
    for envelope in envelopes.iter().rev() {
        if let AgentEvent::CostUpdated {
            input_tokens,
            output_tokens,
            cache_read_tokens,
            ..
        } = &envelope.event
        {
            agent.cost_tracker.input_tokens = *input_tokens;
            agent.cost_tracker.output_tokens = *output_tokens;
            agent.cost_tracker.cache_read_tokens = *cache_read_tokens;
            return;
        }
    }
}

/// Fold lineage at resume (`docs/plans/p2-phase-c-design.md` §3): union the
/// child session ids persisted in the json snapshot (order first) with every
/// `ChildSessionSpawned` id found in the event log (appended), deduplicated.
/// The log-derived ids recover lineage for crashed parents whose json was
/// never re-written after a spawn.
pub(crate) fn fold_child_session_ids(
    meta_ids: &[String],
    envelopes: &[EventEnvelope],
) -> Vec<String> {
    let mut folded = meta_ids.to_vec();
    for envelope in envelopes {
        if let AgentEvent::ChildSessionSpawned {
            child_session_id, ..
        } = &envelope.event
            && !folded.contains(child_session_id)
        {
            folded.push(child_session_id.clone());
        }
    }
    folded
}

/// Persist an approved allow pattern to the workspace config file.
fn persist_allow_pattern(workspace_root: &Path, pattern: String) {
    let root = workspace_root.to_path_buf();
    std::mem::drop(tokio::runtime::Handle::current().spawn_blocking(move || {
        match nca_common::config::NcaConfig::load_for_workspace(&root) {
            Ok(mut config) => {
                if !config.permissions.allow.contains(&pattern) {
                    config.permissions.allow.push(pattern);
                    tracing::debug!("persisted allow pattern to workspace config");
                    if let Err(e) = config.save_workspace_file(&root) {
                        tracing::warn!("failed to persist allow pattern: {e}");
                    }
                }
            }
            Err(e) => {
                tracing::warn!("failed to load config for pattern persistence: {e}");
            }
        }
    }));
}

/// Persist the current set of mounted paths to the workspace config file.
///
/// Mirrors [`persist_allow_pattern`]: loads the freshest config from disk,
/// updates `extra_paths` to match the live `RealFs` state, and writes back
/// via the diff algorithm so only the workspace-local file is touched.
///
/// Awaited by [`Supervisor::mount_path`] / [`Supervisor::unmount_path`] rather
/// than spawned-and-dropped: dropping the join handle meant a fast process exit
/// right after `/mount` could discard the queued task and silently lose the
/// write, and callers had no way to know when the file was durably updated.
async fn persist_mounted_paths(workspace_root: &Path, paths: Vec<PathBuf>) {
    let root = workspace_root.to_path_buf();
    let joined =
        tokio::task::spawn_blocking(
            move || match nca_common::config::NcaConfig::load_for_workspace(&root) {
                Ok(mut config) => {
                    if config.extra_paths != paths {
                        config.extra_paths = paths;
                        tracing::debug!("persisted mounted paths to workspace config");
                        if let Err(e) = config.save_workspace_file(&root) {
                            tracing::warn!("failed to persist mounted paths: {e}");
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("failed to load config for mount persistence: {e}");
                }
            },
        )
        .await;
    if let Err(join_err) = joined {
        tracing::warn!("mounted-path persistence task did not complete: {join_err}");
    }
}

/// Resolve the session's provider: an injected provider wins verbatim, else
/// build from config. `create` uses this so tests can supply a mock `Provider`
/// and skip `build_provider` entirely.
fn resolve_provider(
    injected: Option<Arc<dyn Provider>>,
    config: &NcaConfig,
) -> Result<Arc<dyn Provider>, ProviderError> {
    match injected {
        Some(provider) => Ok(provider),
        None => build_provider(config),
    }
}

impl Supervisor {
    /// Create a new supervised session. This sets up the agent loop, IPC server,
    /// event channels, and persists initial session metadata.
    pub async fn create(cfg: SupervisorConfig) -> Result<Self, ProviderError> {
        let workspace_root = cfg
            .workspace_root
            .canonicalize()
            .map_err(|e| ProviderError::Configuration(format!("invalid workspace root: {e}")))?;

        let mut config = cfg.config;
        if cfg.safe_mode {
            config.permissions.deny.push("execute_bash".into());
        }

        // Seed built-in OMO specialist skills into the user's XDG dir (idempotent).
        // This ensures a fresh `nca` install has the seven specialists available
        // without requiring a manual `nca skills add` step.
        let _seeded = nca_common::builtin_skills::seed_builtin_skills();

        // Auto-register skill-based agent profiles before resolving agent_name.
        register_skill_agents(&mut config, &workspace_root);

        // Snapshot the config BEFORE applying agent-profile overrides. This is
        // used as the base when switching agents at runtime.
        let base_config = config.clone();

        // Resolve the agent profile (if any) and apply its overrides to the config.
        let requested_agent_name = cfg.agent_name.clone();
        let agent_profile = requested_agent_name
            .as_deref()
            .and_then(|name| config.agent_profile(name).cloned());
        if let Some(name) = requested_agent_name.as_deref()
            && agent_profile.is_none()
        {
            tracing::warn!(
                agent = name,
                "agent profile not found; using the default harness prompt"
            );
        }
        if let Some(ref profile) = agent_profile {
            if let Some(provider) = profile.resolve_provider() {
                config.set_default_provider(provider);
            }
            if let Some(ref model) = profile.model {
                let resolved = config.model.resolve_alias(model);
                config.provider.set_model_for_default(resolved);
                config.sync_default_model_from_provider();
            }
            if let Some(mode) = profile.permission_mode {
                config.permissions.mode = mode;
            }
        }

        let provider = resolve_provider(cfg.provider, &config)?;
        let fs: Arc<dyn WorkspaceFs> = Arc::new(RealFs::new(workspace_root.clone()));
        let fs_for_supervisor = fs.clone();
        // Restore mounts persisted in the workspace-local config. A missing or
        // inaccessible path is logged but does not abort startup — the user can
        // re-run `/mount` once the path is available again.
        for extra in &config.extra_paths {
            if let Err(e) = fs_for_supervisor.mount_path(extra) {
                tracing::warn!("failed to restore mount {}: {e}", extra.display());
            }
        }
        let mut tools = if cfg.safe_mode {
            ToolRegistry::with_default_readonly_tools(fs.clone(), config.web.clone())
        } else {
            ToolRegistry::with_default_full_tools(fs.clone(), config.web.clone())
        };
        if !config.mcp.servers.is_empty() && (!cfg.safe_mode || config.mcp.expose_in_safe_mode) {
            match load_mcp_tools(&workspace_root, &config.mcp.servers).await {
                Ok(mcp_tools) => {
                    for tool in mcp_tools {
                        tools.register(tool);
                    }
                }
                Err(error) => tracing::warn!("failed to load MCP tools: {}", error),
            }
        }

        // Wire the P5 Landlock sandbox into every PTY shell execution
        // (resolved once here; per-exec confinement applies only to children).
        // Live mounts (restored above + any runtime `/mount`) are passed so
        // the policy matches file-tool visibility; skill catalog roots are
        // passed read-only so skill-bundled tools stay executable under
        // confinement.
        let pty = PtyManager::new(&workspace_root);
        pty.set_sandbox_config(
            config.permissions.sandbox.clone(),
            &fs.mounted_paths(),
            &SkillCatalog::discovery_roots(&workspace_root, &config.harness.skill_directories),
        );
        let pty = Arc::new(pty);
        let pty_for_supervisor = pty.clone();
        tools.register(Box::new(crate::bash_tool::RuntimeBashTool::new(pty)));

        let (spawn_tx, spawn_rx) = mpsc::channel::<SpawnRequest>(16);
        let spawn_history = Arc::new(Mutex::new(Vec::<Message>::new()));
        let registry = Arc::new(SubagentRegistry::new());
        // P1 read-only introspection: bounded control channel for
        // `task_status`/`task_result`. The consumer (spawned below once the
        // session store exists) is owned by the supervisor — embedders only
        // wire the spawn consumer as before.
        let mut subagent_control_rx = None;
        if !cfg.safe_mode {
            tools.register(Box::new(SpawnSubagentTool::new(
                spawn_tx,
                Arc::clone(&spawn_history),
            )));
            let (control_tx, control_rx) = mpsc::channel::<SubagentControlRequest>(100);
            let control_timeout = Duration::from_millis(config.subagent.result_timeout_ms);
            // Cancel is a flag-flip + reply: bounded tight (§2 wire table —
            // 10s vs the 30s status/result/message budget).
            let cancel_timeout = Duration::from_secs(10);
            tools.register(Box::new(TaskStatusTool::new(
                control_tx.clone(),
                control_timeout,
            )));
            tools.register(Box::new(TaskResultTool::new(
                control_tx.clone(),
                control_timeout,
            )));
            tools.register(Box::new(TaskMessageTool::new(
                control_tx.clone(),
                control_timeout,
            )));
            tools.register(Box::new(TaskCancelTool::new(
                control_tx.clone(),
                cancel_timeout,
            )));
            // Revive runs a FULL child turn (cancel-wait + resume + run) —
            // give it the same 600s budget as a foreground spawn so the
            // two spawn modes share one wall-clock contract (§2 tightened
            // the wire table's "no hard cap for revive" to a bounded one;
            // on timeout the revive keeps running detached and its result
            // stays fetchable via task_result).
            tools.register(Box::new(TaskReviveTool::new(
                control_tx,
                Duration::from_secs(600),
            )));
            subagent_control_rx = Some(control_rx);
        }

        let approval_pending: Option<ApprovalPendingMap>;
        let approval = if cfg.interactive_approvals {
            match cfg.approval_handler {
                Some(handler) => {
                    approval_pending = None;
                    ApprovalPolicy::new(config.permissions.clone())
                        .with_handler(handler)
                        .with_persist({
                            let wr = workspace_root.clone();
                            move |p| persist_allow_pattern(&wr, p)
                        })
                }
                None => {
                    let ipc_handler = IpcApprovalHandler::new();
                    approval_pending = Some(ipc_handler.pending());
                    ApprovalPolicy::new(config.permissions.clone())
                        .with_handler(ipc_handler as Arc<dyn ApprovalHandler>)
                        .with_persist({
                            let wr = workspace_root.clone();
                            move |p| persist_allow_pattern(&wr, p)
                        })
                }
            }
        } else {
            approval_pending = None;
            ApprovalPolicy::new(config.permissions.clone())
                .fail_on_ask()
                .with_handler(Arc::new(AutoDenyHandler) as Arc<dyn ApprovalHandler>)
                .with_persist({
                    let wr = workspace_root.clone();
                    move |p| persist_allow_pattern(&wr, p)
                })
        };

        let (event_tx, event_rx) = mpsc::channel(256);
        let question_pending = Arc::new(Mutex::new(HashMap::new()));
        tools.register(Box::new(AskQuestionTool::new(
            event_tx.clone(),
            question_pending.clone(),
        )));
        tools.register(Box::new(InvokeSkillTool::new(
            workspace_root.clone(),
            config.harness.skill_directories.clone(),
        )));

        let todo_store: TodoStore = Arc::new(Mutex::new(Vec::new()));
        tools.register(Box::new(UpdateTodosTool::new(event_tx.clone(), todo_store)));

        // Apply tool gating from the agent profile (if any).
        if let Some(ref profile) = agent_profile
            && let Some(ref allowed) = profile.allowed_tools
        {
            tools.restrict_to(allowed);
        }

        let session_id = cfg.session_id.unwrap_or_else(generate_session_id);
        let session_store = SessionStore::new(workspace_root.join(&config.session.history_dir));

        // Control consumer answers task_status/task_result/task_message/
        // task_cancel/task_revive against the registry + a read-only store
        // handle (never saves — single-writer invariant,
        // `docs/subagent-task-lifecycle.md` §6). `ChildMessageQueued`
        // envelopes ride the session's own bounded event channel. Revive
        // additionally needs the parent config + workspace root to rebuild
        // the child (resume), so they ride along here.
        if let Some(control_rx) = subagent_control_rx.take() {
            tokio::spawn(subagent_control_consumer(
                control_rx,
                Arc::clone(&registry),
                SessionStore::new(workspace_root.join(&config.session.history_dir)),
                config.clone(),
                workspace_root.clone(),
                Some(event_tx.clone()),
            ));
        }

        let ipc_server = IpcServer::new(&session_id);
        let socket_path = ipc_server.socket_path();
        let ipc_handle = ipc_server
            .start()
            .await
            .map_err(|e| ProviderError::Other(e.to_string()))?;

        let _ = event_tx.try_send(AgentEvent::SessionStarted {
            session_id: session_id.clone(),
            workspace: workspace_root.clone(),
            model: config.model.default_model.clone(),
        });

        let created_at = Utc::now();
        let hook_runner = {
            let runner = HookRunner::new(config.hooks.clone());
            runner.has_any().then_some(runner)
        };

        // Build plugin registry — discover and spawn out-of-process plugins.
        let mut plugin_host = Option::<PluginHost>::None;
        let plugins = if cfg.safe_mode {
            PluginRegistry::new()
        } else {
            let descriptors = crate::plugin_host::discover_plugins();
            if descriptors.is_empty() {
                PluginRegistry::new()
            } else {
                let mut host = PluginHost::with_config(config.plugins.clone());
                let perm_mode = serde_json::to_string(&config.permissions.mode)
                    .unwrap_or_else(|_| "\"default\"".into());
                let perm_mode = perm_mode.trim_matches('"');
                let errors = host
                    .start_all(&descriptors, &workspace_root, &session_id, perm_mode)
                    .await;
                for err in &errors {
                    tracing::warn!("plugin startup error: {err}");
                }
                let reg = host.registry();
                plugin_host = Some(host);
                reg
            }
        };

        // G1: every tool the plugins declared in Hello becomes a callable
        // `PluginTool` in the registry (routing, approval tiers, and timeouts
        // identical to built-ins; execution via the plugin's execute_tool).
        nca_core::tools::plugin_tool::register_plugin_tools(&mut tools, &plugins, &config.plugins);

        // G4: lifecycle notification — plugins received Config during
        // `start_all`; `session_start` tells them the session is live
        // (upstream SessionStart parity; fire-and-forget, never blocks).
        plugins.notify_event(&serde_json::json!({
            "type": "session_start",
            "session_id": session_id,
        }));

        let mut agent = AgentLoop::new(
            provider,
            tools,
            approval,
            config.model.default_model.clone(),
            event_tx.clone(),
            config.session.max_turns_per_run,
            config.session.max_tool_calls_per_turn,
            config.session.checkpoint_interval,
            hook_runner.clone(),
        );
        let system_prompt = build_system_prompt_with_agent(
            &config,
            &workspace_root,
            &plugins,
            cfg.orchestration_context.as_ref(),
            agent_profile.as_ref(),
            &fs.mounted_paths(),
        );
        agent.set_system_prompt(system_prompt);
        agent.set_keepalive_profile(cache_keepalive::resolve_profile(config.provider.default));
        // Middleware chain composition (roadmap initial order, knobs from
        // `[middleware]`, compaction mode from `[memory.context]`):
        // cost-guard → compaction → overflow-recovery → retry. Wired at the
        // single construction site so parent, resumed, and subagent-child
        // sessions all get it; `AgentLoop::new` default chain stays empty
        // (P4 invariant).
        agent.extend_middleware(default_chain(
            &config.middleware,
            config.memory.context.smart_compaction_mode,
        ));

        let context_manager =
            Self::make_context_manager(&config, &config.model.default_model).await;

        // Turn-commit durability wiring (P2 Phase B): the fanout signals each
        // committed TurnCompleted on the watch; run_turn waits for it.
        let (commit_tx, commit_rx) = watch::channel::<u64>(0);
        let turn_commit_wired = Arc::new(AtomicBool::new(false));
        let turn_commit_tx = Some((commit_tx, turn_commit_wired.clone()));

        let sup = Self {
            session_id,
            workspace_root,
            model: config.model.default_model.clone(),
            created_at,
            status: SessionStatus::Running,
            pid: Some(std::process::id()),
            socket_path: Some(socket_path),
            agent,
            session_store,
            ipc_handle: Some(ipc_handle),
            event_rx: Some(event_rx),
            approval_pending,
            question_pending: Some(question_pending),
            spawn_rx: Some(spawn_rx),
            spawn_history,
            subagent_registry: registry,
            worktree_path: None,
            branch: None,
            base_branch: None,
            parent_session_id: None,
            child_session_ids: Vec::new(),
            inherited_summary: None,
            spawn_reason: None,
            session_summary: None,
            session_title: None,
            orchestration: cfg.orchestration_context,
            config,
            base_config,
            agent_profile,
            active_agent_name: requested_agent_name,
            hooks: hook_runner,
            plugins: Arc::new(plugins),
            plugin_host,
            plugin_disable_reported: HashSet::new(),
            context_manager,
            last_summary_at_tokens: 0,
            fs: fs_for_supervisor,
            pty: pty_for_supervisor,
            turn_commit_tx,
            turn_commit_rx: Some(commit_rx),
            turn_commit_wired,
        };
        sup.save().await.map_err(ProviderError::Other)?;
        sup.update_last_session()
            .await
            .map_err(ProviderError::Other)?;
        sup.run_session_hook(HookEventKind::SessionStart, json!(sup.snapshot()))
            .await;
        Ok(sup)
    }

    /// Resume an existing session by loading its state and creating a fresh
    /// IPC server + agent loop.
    pub async fn resume(
        config: NcaConfig,
        workspace_root: &Path,
        safe_mode: bool,
        interactive_approvals: bool,
        session_id: &str,
        approval_handler: Option<Arc<dyn ApprovalHandler>>,
        provider: Option<Arc<dyn Provider>>,
    ) -> Result<Self, ProviderError> {
        // Load the original session state BEFORE create() overwrites the file.
        // create() calls save() with an empty message list, which would destroy
        // the conversation history if we loaded after.
        // A failed json load is NOT fatal here: if the event log can supply a
        // replay projection, the resume proceeds with a corrupt-json rescue.
        let store = SessionStore::new(workspace_root.join(&config.session.history_dir));
        let load_result = store.load(session_id).await;
        let load_error = load_result
            .as_ref()
            .err()
            .map(|e| ProviderError::Other(e.to_string()));
        let loaded = load_result.ok();

        let mut sup = Self::create(SupervisorConfig {
            config: config.clone(),
            workspace_root: workspace_root.to_path_buf(),
            safe_mode,
            interactive_approvals,
            session_id: Some(session_id.into()),
            approval_handler,
            orchestration_context: loaded.as_ref().and_then(|l| l.meta.orchestration.clone()),
            // Restore the persisted agent profile so the specialist persona
            // (prompt, provider/permission overrides, tool gating) survives
            // resume. Threaded through `create`'s pipeline — not a post-hoc
            // `apply_agent_profile` — so an injected provider still wins
            // verbatim and unresolvable names degrade to the default prompt.
            agent_name: loaded.as_ref().and_then(|l| l.meta.agent_name.clone()),
            provider,
        })
        .await?;

        if let Some(loaded) = loaded.as_ref() {
            sup.session_id = loaded.meta.id.clone();
            sup.workspace_root =
                resolve_resume_workspace_root(&sup.workspace_root, &loaded.meta.workspace);
            sup.model = loaded.meta.model.clone();
            sup.agent.model = loaded.meta.model.clone();
            sup.created_at = loaded.meta.created_at;
            sup.status = loaded.meta.status.clone();
            sup.worktree_path = loaded.meta.worktree_path.clone();
            sup.branch = loaded.meta.branch.clone();
            sup.base_branch = loaded.meta.base_branch.clone();
            sup.parent_session_id = loaded.meta.parent_session_id.clone();
            sup.child_session_ids = loaded.meta.child_session_ids.clone();
            sup.inherited_summary = loaded.meta.inherited_summary.clone();
            sup.spawn_reason = loaded.meta.spawn_reason.clone();
            sup.session_summary = loaded.meta.session_summary.clone();
            sup.session_title = loaded.meta.session_title.clone();
            sup.orchestration = loaded.meta.orchestration.clone();
            sup.context_manager = Self::make_context_manager(&sup.config, &sup.model).await;
        }
        sup.pid = Some(std::process::id());
        sup.session_store = store;

        // Event-log replay (P2 Phase A): always read + project — cheap, and it
        // enables the corrupt-json fallback and the divergence detector.
        let envelopes = crate::session_store::read_event_log(&sup.event_log_path());
        let replayed = nca_core::replay::replay_surface_events(&envelopes);

        // Fail only when BOTH truths are gone: json unloadable AND replay empty.
        if loaded.is_none() && replayed.is_empty() {
            return Err(
                load_error.unwrap_or_else(|| ProviderError::Other("session load failed".into()))
            );
        }

        let fresh_system: Vec<Message> = sup
            .agent
            .messages
            .iter()
            .filter(|m| m.role == Role::System)
            .cloned()
            .collect();
        let snapshot_messages = loaded.map(|l| l.messages);
        let snapshot_non_system = snapshot_messages
            .as_deref()
            .map(normalize_resume_projection)
            .unwrap_or_default();
        // Fresh-format log (contains MessageRecorded) ⇒ its projection can
        // be trusted as the resume truth (P2 Phase B §5).
        let log_has_surface_events = envelopes
            .iter()
            .any(|e| matches!(e.event, AgentEvent::MessageRecorded { .. }));

        let (messages, source) = select_resume_messages(
            snapshot_messages,
            replayed.clone(),
            log_has_surface_events,
            fresh_system,
            &sup.workspace_root,
        );
        sup.agent.messages = messages;

        match source {
            ResumeMessageSource::ReplayAuthoritative => {
                // Json loaded but the replay won: warn on any real drift so
                // cache staleness is visible. Compare the REPAIRED replay —
                // attachment cleanup rewrites image parts in the snapshot
                // after the message was recorded, so the raw record
                // legitimately differs; repaired, it must match.
                let repaired = repair_missing_images(replayed.clone(), &sup.workspace_root);
                if normalize_resume_projection(&repaired) != snapshot_non_system {
                    tracing::warn!(
                        "session json and event-log replay diverge (json={} msgs, replay={} msgs); replay kept",
                        snapshot_non_system.len(),
                        replayed.len()
                    );
                }
                seed_cost_tracker_from_log(&mut sup.agent, &envelopes);
            }
            ResumeMessageSource::ReplayFallback => {
                tracing::warn!(
                    "session json unusable; restored {} messages from event-log replay",
                    sup.agent.messages.len()
                );
                seed_cost_tracker_from_log(&mut sup.agent, &envelopes);
            }
            ResumeMessageSource::Snapshot => {
                // Json kept. Only a fresh-format log can meaningfully
                // diverge (old logs fold to empty by construction).
                if log_has_surface_events {
                    let repaired = repair_missing_images(replayed.clone(), &sup.workspace_root);
                    if normalize_resume_projection(&repaired) != snapshot_non_system {
                        tracing::warn!(
                            "session json and event-log replay diverge (json={} msgs, replay={} msgs); json kept",
                            snapshot_non_system.len(),
                            replayed.len()
                        );
                    }
                }
            }
        }

        // Seed the turn-id counter from the persisted event log so turn ids
        // stay session-unique across restarts. Tolerant scan: a missing,
        // empty, or partially-corrupt log never fails the resume.
        let max_turn = envelopes
            .iter()
            .filter_map(|e| match e.event {
                AgentEvent::TurnStarted { turn_id } => Some(turn_id),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        if max_turn > 0 {
            sup.agent.set_turn_seq_start(max_turn);
        }

        // Fold lineage from ChildSessionSpawned envelopes (P2 Phase C §3) so
        // the union of json + log ids persists with the resume save below —
        // recovering children of a parent that crashed before a finish() save.
        sup.child_session_ids = fold_child_session_ids(&sup.child_session_ids, &envelopes);

        // Rebuild the subagent task registry from the same envelopes (P1):
        // the registry is an in-memory projection, so children's latest
        // known lifecycle state is re-derived from the log. Read-only —
        // no session json is written from here (single-writer invariant).
        for envelope in &envelopes {
            sup.subagent_registry.apply_envelope(envelope);
        }

        // Second registry recovery channel: meta lineage (persisted by the
        // parent's own saves) recovers children the envelope fold missed
        // (old-format logs, torn tails). Seeded entries are terminal-only
        // projections read from each child's json — never live handles,
        // never written back (single-writer invariant preserved).
        crate::subagent_registry::seed_registry_from_lineage(
            &sup.subagent_registry,
            &sup.session_store,
            &sup.session_id,
            &sup.child_session_ids,
        )
        .await;

        // Re-save immediately after restore: closes the create()-saves-empty-
        // state window so a crash right after resume no longer wipes the json.
        sup.save().await.map_err(ProviderError::Other)?;
        Ok(sup)
    }

    /// Extract a handle for the caller. The handle provides event_rx, ipc_handle,
    /// approval_pending, and spawn_rx for wiring into stream/command tasks.
    pub fn take_handle(&mut self) -> SupervisorHandle {
        SupervisorHandle {
            session_id: self.session_id.clone(),
            workspace_root: self.workspace_root.clone(),
            model: self.model.clone(),
            socket_path: self.socket_path.clone(),
            event_log_path: self.event_log_path(),
            event_rx: self.event_rx.take(),
            ipc_handle: self.ipc_handle.take(),
            approval_pending: self.approval_pending.take(),
            question_pending: self.question_pending.take(),
            spawn_rx: self.spawn_rx.take(),
            turn_commit_tx: self.turn_commit_tx.take(),
        }
    }

    pub fn event_log_path(&self) -> PathBuf {
        self.session_store
            .sessions_dir()
            .join(format!("{}.events.jsonl", self.session_id))
    }

    /// Shared live mirror of this session's conversation, refreshed at each
    /// turn start by [`Self::run_turn_with_images`]. Wired into the sub-agent
    /// spawn consumer so child summaries and forwarded images reflect the
    /// conversation as of spawn time, not session-create time.
    pub fn spawn_history(&self) -> Arc<Mutex<Vec<Message>>> {
        Arc::clone(&self.spawn_history)
    }

    /// Read-only subagent task registry (P1 introspection). Shared with the
    /// spawn consumer (live fold) and the control consumer
    /// (`task_status`/`task_result` replies); rebuilt at resume from the
    /// event log.
    pub fn subagent_registry(&self) -> Arc<SubagentRegistry> {
        Arc::clone(&self.subagent_registry)
    }

    pub async fn run_turn(&mut self, prompt: &str) -> Result<String, ProviderError> {
        self.run_turn_with_images(prompt, &[]).await
    }

    /// G2: collect per-turn plugin context injections (`userPrompt @6`) and
    /// append them to the outgoing user prompt as tagged blocks. The blocks
    /// ride the user message (Claude Code `additionalContext` model): they
    /// reach the model and the transcript without touching the cache-stable
    /// system prompt. Timeouts are per-plugin RPC budgets
    /// (`[plugins] prompt_hook_timeout_ms`), so a slow plugin can never stall
    /// a turn by more than its budget.
    fn apply_user_prompt_hooks(&self, prompt: &str) -> String {
        if self.plugins.is_empty() {
            return prompt.to_string();
        }
        let hooks = self.plugins.collect_user_prompt_hooks(prompt);
        nca_core::plugin::format_prompt_context_blocks(prompt, &hooks)
    }

    /// Like [`run_turn`], but attaches on-disk images (paths relative to workspace) for vision models.
    pub async fn run_turn_with_images(
        &mut self,
        prompt: &str,
        attachments: &[nca_common::message::ImageAttachment],
    ) -> Result<String, ProviderError> {
        if !attachments.is_empty()
            && !nca_common::model_caps::model_accepts_native_images(
                self.config.provider.default,
                self.model.as_str(),
            )
        {
            return Err(ProviderError::Configuration(format!(
                "native images are not supported for provider {} with model `{}` (pick a vision-capable model or remove image attachments)",
                self.config.provider.default.display_name(),
                self.model
            )));
        }

        // G7: report plugins disabled since the last turn (once each).
        self.surface_plugin_disables();

        // Check context before running turn
        self.maybe_compact_context().await;

        // G2: per-turn plugin prompt hooks (`userPrompt @6`). Returned text
        // is appended to the outgoing user message as tagged context blocks —
        // visible to the model and the transcript, never the system prompt
        // (that prefix is cache-stable). The title below keeps the raw prompt.
        let augmented = self.apply_user_prompt_hooks(prompt);

        // Refresh the spawn-history mirror so `spawn_subagent` calls made
        // during this turn collect images and context from exactly the
        // history the model sees (post-compaction), plus this turn's opening
        // message — a screenshot pasted together with the delegation request
        // must reach the child.
        if let Ok(mut mirror) = self.spawn_history.lock() {
            *mirror = self.agent.messages.clone();
            mirror.push(nca_core::agent::turn_user_message(&augmented, attachments));
        }

        // Durability barrier (P2 Phase B): capture the last committed turn
        // before the turn runs, then wait for this turn's TurnCompleted to be
        // flushed+fsynced to the event log before returning.
        let before = self
            .turn_commit_rx
            .as_ref()
            .map(|rx| *rx.borrow())
            .unwrap_or(0);
        let result = self
            .agent
            .run_turn(&augmented, self.workspace_root.as_path(), attachments)
            .await;
        // Durability ordering (P3 §5): ALWAYS wait for the first turn's
        // commit before any further canonical-history mutation (the overflow
        // summarize below replaces history).
        self.await_turn_commit(before).await;
        let output = match result {
            Ok(out) => out,
            Err(e) if should_overflow_retry(&e, false) => {
                // Canonical fallback arm: summarize + exactly one retry.
                // `run_turn`'s Err already rolled `agent.messages` back to
                // the pre-turn baseline, so the summarize operates on it.
                let stats = self.context_manager.stats(&self.agent.messages);
                if let Some(tx) = self.agent.event_sender() {
                    let _ = tx
                        .send(AgentEvent::ContextCompactionStart {
                            tokens_before: stats.estimated_tokens,
                            reason: "overflow_summarize".into(),
                        })
                        .await;
                }
                if let Err(e) = self.perform_auto_summarize("overflow_summarize").await {
                    tracing::error!("overflow summarize failed: {}", e);
                }
                let retry_before = self
                    .turn_commit_rx
                    .as_ref()
                    .map(|rx| *rx.borrow())
                    .unwrap_or(0);
                let second = self
                    .agent
                    .run_turn(&augmented, self.workspace_root.as_path(), attachments)
                    .await;
                self.await_turn_commit(retry_before).await;
                second?
            }
            Err(e) => return Err(e),
        };

        // Check context after turn
        self.check_and_summarize_context().await;

        // Emit context stats for UI
        self.emit_context_stats().await;

        self.refresh_session_summary();

        // Generate session title from the first user prompt if not yet set.
        // Bounded timeout: title generation is best-effort. A flaky network
        // must not hold the main command loop hostage for the full 90s
        // stream-idle window — if it can't finish quickly, just skip it.
        if self.session_title.is_none() {
            let _ = tokio::time::timeout(
                std::time::Duration::from_secs(20),
                self.generate_session_title(prompt),
            )
            .await;
        }

        self.update_last_session()
            .await
            .map_err(ProviderError::Other)?;
        Ok(output)
    }

    /// Clone of the turn-commit watch receiver (observability for tests and
    /// future UI progress indicators).
    pub fn turn_commit_rx(&self) -> Option<watch::Receiver<u64>> {
        self.turn_commit_rx.clone()
    }

    /// Waits (bounded) until the fanout has committed a turn with id greater
    /// than `before` to the event log. Skips entirely when no fanout was
    /// wired (`take_turn_commit_tx` never called) or the receiver is gone;
    /// times out after 5s with an error log — the barrier must never fail
    /// the turn (liveness over false durability).
    async fn await_turn_commit(&self, before: u64) {
        if !self.turn_commit_wired.load(Ordering::SeqCst) {
            return;
        }
        let Some(rx) = self.turn_commit_rx.clone() else {
            return;
        };
        let mut rx = rx;
        match tokio::time::timeout(
            std::time::Duration::from_secs(5),
            rx.wait_for(|v| *v > before),
        )
        .await
        {
            Ok(Ok(value)) => {
                tracing::debug!("event log committed through turn {}", *value);
            }
            Ok(Err(_)) => {
                tracing::error!(
                    "event-log turn commit channel closed; turn data may not be durable"
                );
            }
            Err(_) => {
                tracing::error!(
                    "event-log turn commit barrier timed out; turn data may not be durable"
                );
            }
        }
    }

    /// Get current context statistics with model info.
    pub fn context_stats(&self) -> ContextStats {
        self.context_manager.stats(&self.agent.messages)
    }

    async fn make_context_manager(config: &NcaConfig, model: &str) -> ContextManager {
        let model_limits = model_limits_api::resolve_model_limits(config, model).await;
        let context_window = if config.memory.context.auto_detect_context_window {
            tracing::debug!(
                "Context window target for {}: {} tokens",
                model,
                model_limits.context_window
            );
            model_limits.context_window
        } else {
            config.memory.context.context_window_target
        };

        let context_config = ContextManagerConfig {
            context_window_target: context_window,
            max_retained_messages: config.memory.context.max_retained_messages,
            auto_summarize_threshold: config.memory.context.auto_summarize_threshold,
            enable_auto_summarize: config.memory.context.enable_auto_summarize,
            max_message_chars_for_summary: 10000,
        };
        ContextManager::new(context_config, model.to_string())
    }

    /// Check if context needs attention or summarization.
    /// Proactively compact context **before** a turn runs.
    /// Unlike `check_and_summarize_context` (post-turn), this must act immediately
    /// to prevent empty-response failures caused by an overflowing context window.
    async fn maybe_compact_context(&mut self) {
        if !self.context_manager.config().enable_auto_summarize {
            return;
        }

        let stats = self.context_manager.stats(&self.agent.messages);

        // Already small enough — nothing to do.
        if !stats.should_summarize {
            return;
        }

        // Don't re-summarize if we already compacted in this session and context
        // has not grown past the previous post-compaction size.
        if self.last_summary_at_tokens > 0 && stats.estimated_tokens <= self.last_summary_at_tokens
        {
            return;
        }

        if let Some(tx) = self.agent.event_sender() {
            let _ = tx
                .send(AgentEvent::ContextCompactionStart {
                    tokens_before: stats.estimated_tokens,
                    reason: "auto_summarize".into(),
                })
                .await;
        }

        if let Err(e) = self.perform_auto_summarize("auto_summarize").await {
            tracing::error!("Pre-turn auto-summarize failed: {}", e);
            self.last_summary_at_tokens = 0;
        }
    }

    /// Check if context should be summarized after a turn and trigger if needed.
    async fn check_and_summarize_context(&mut self) {
        if !self.context_manager.config().enable_auto_summarize {
            return;
        }

        let stats = self.context_manager.stats(&self.agent.messages);

        // Don't summarize if we just summarized
        if self.last_summary_at_tokens > 0 && stats.estimated_tokens < self.last_summary_at_tokens {
            // Context was reduced, reset the flag
            self.last_summary_at_tokens = 0;
        }

        if stats.should_summarize && self.last_summary_at_tokens == 0 {
            // Emit event that summarization is starting
            if let Some(tx) = self.agent.event_sender() {
                let _ = tx
                    .send(AgentEvent::ContextCompactionStart {
                        tokens_before: stats.estimated_tokens,
                        reason: "auto_summarize".into(),
                    })
                    .await;
            }

            // Trigger summarization
            if let Err(e) = self.perform_auto_summarize("auto_summarize").await {
                tracing::error!("Auto-summarize failed: {}", e);
                // Reset so we can try again
                self.last_summary_at_tokens = 0;
            }
        }
    }

    /// Emit current context statistics to the UI via the event bus.
    async fn emit_context_stats(&self) {
        let stats = self.context_manager.stats(&self.agent.messages);
        if let Some(tx) = self.agent.event_sender() {
            let _ = tx
                .send(AgentEvent::ContextStatsUpdated {
                    estimated_tokens: stats.estimated_tokens,
                    context_window: stats.context_window,
                    usage_percent: stats.usage_percent,
                })
                .await;
        }
    }

    /// Checkpoint the post-compaction history into the event log (P2 Phase
    /// B §1): replay-authoritative resume folds `HistoryReplaced` as a state
    /// checkpoint, so the projection must see exactly the state the json
    /// will save. System messages are stripped here — resume always
    /// prepends a fresh system prompt; stale prompts must not ride along.
    async fn emit_history_replaced(&self, new_messages: &[Message]) {
        let Some(tx) = self.agent.event_sender() else {
            return;
        };
        let payload: Vec<Message> = new_messages
            .iter()
            .filter(|m| m.role != Role::System)
            .cloned()
            .collect();
        let _ = tx
            .send(AgentEvent::HistoryReplaced { messages: payload })
            .await;
    }

    /// Perform the actual auto-summarization. `reason` threads the bracket
    /// cause ("auto_summarize" | "overflow_summarize") into diagnostics; the
    /// matching `ContextCompactionStart` is emitted by the caller, and this
    /// method emits `ContextCompactionEnd` on BOTH exit paths (AI summary and
    /// sliding-window fallback, including the early empty-messages path).
    async fn perform_auto_summarize(&mut self, reason: &str) -> Result<(), String> {
        let result = self.perform_auto_summarize_inner(reason).await;
        if result.is_ok() {
            // G4: compact notification on every successful compaction path
            // (upstream re-fires SessionStart on compact — plugins use this
            // to re-inject fresh state on the next turn).
            self.plugins.notify_event(&serde_json::json!({
                "type": "context_compact",
                "session_id": self.session_id,
                "reason": reason,
            }));
        }
        result
    }

    async fn perform_auto_summarize_inner(&mut self, reason: &str) -> Result<(), String> {
        let messages_to_summarize = self
            .context_manager
            .get_messages_to_summarize(&self.agent.messages);

        if messages_to_summarize.is_empty() {
            // Nothing to summarize, use sliding window instead
            let compacted = self
                .context_manager
                .get_sliding_window(&self.agent.messages, None);
            self.emit_history_replaced(&compacted).await;
            self.agent.messages = compacted;
            let tokens_after = self
                .context_manager
                .stats(&self.agent.messages)
                .estimated_tokens;
            if let Some(tx) = self.agent.event_sender() {
                let _ = tx
                    .send(AgentEvent::ContextCompactionEnd {
                        tokens_after,
                        kv_prefix_broken: true,
                    })
                    .await;
            }
            tracing::debug!(reason = reason, "auto-summarize applied sliding window");
            return Ok(());
        }

        // Generate summary prompt
        let summary_prompt = self.context_manager.summary_prompt(&messages_to_summarize);

        // Try to use the AI to summarize. If the provider supports a quick call,
        // we can use it. Otherwise, fall back to extractive summarization.
        match self.summarize_with_ai(&summary_prompt).await {
            Ok(summary) => {
                // Apply the summary
                let replaced = self
                    .context_manager
                    .apply_summary(&self.agent.messages, &summary);
                self.emit_history_replaced(&replaced).await;
                self.agent.messages = replaced;
                self.last_summary_at_tokens = self
                    .context_manager
                    .stats(&self.agent.messages)
                    .estimated_tokens;

                if let Some(tx) = self.agent.event_sender() {
                    let _ = tx
                        .send(AgentEvent::ContextCompactionEnd {
                            tokens_after: self.last_summary_at_tokens,
                            kv_prefix_broken: true,
                        })
                        .await;
                }
            }
            Err(e) => {
                // Fallback: just use sliding window
                tracing::warn!("AI summarization failed, using sliding window: {}", e);
                let compacted = self
                    .context_manager
                    .get_sliding_window(&self.agent.messages, None);
                self.emit_history_replaced(&compacted).await;
                self.agent.messages = compacted;
                self.last_summary_at_tokens = self
                    .context_manager
                    .stats(&self.agent.messages)
                    .estimated_tokens;
                if let Some(tx) = self.agent.event_sender() {
                    let _ = tx
                        .send(AgentEvent::ContextCompactionEnd {
                            tokens_after: self.last_summary_at_tokens,
                            kv_prefix_broken: true,
                        })
                        .await;
                }
            }
        }

        Ok(())
    }

    /// Use AI to generate a summary of the conversation.
    async fn summarize_with_ai(&self, prompt: &str) -> Result<String, String> {
        use nca_common::message::Message;

        let messages = vec![Message::user(prompt)];

        let mut stream = self
            .agent
            .provider
            .chat(&messages, &[], &self.model, self.workspace_root.as_path())
            .await
            .map_err(|e| e.to_string())?;

        // Collect the response
        let mut summary = String::new();
        while let Some(chunk) = stream.recv().await {
            match chunk {
                nca_core::provider::StreamChunk::TextDelta(delta) => {
                    summary.push_str(&delta);
                }
                nca_core::provider::StreamChunk::Done => break,
                nca_core::provider::StreamChunk::Error(err) => {
                    return Err(err.to_string());
                }
                _ => {}
            }
        }

        Ok(summary.trim().to_string())
    }

    pub async fn finish(&mut self, reason: EndReason) {
        self.status = match reason {
            EndReason::Completed | EndReason::UserExit => SessionStatus::Completed,
            EndReason::Error => SessionStatus::Error,
            EndReason::Cancelled => SessionStatus::Cancelled,
        };
        // G4: lifecycle notification before anything tears down (fire-and-forget).
        self.plugins.notify_event(&serde_json::json!({
            "type": "session_end",
            "session_id": self.session_id,
            "reason": format!("{reason:?}"),
        }));
        if let Some(tx) = self.agent.event_sender() {
            let _ = tx.send(AgentEvent::SessionEnded { reason }).await;
        }
        self.refresh_session_summary();
        if self.config.memory.auto_compact_on_finish {
            let _ = self
                .append_memory_note("session-summary", self.session_summary.clone())
                .await;
        }
        self.run_session_hook(
            HookEventKind::SessionEnd,
            json!({
                "reason": format!("{reason:?}"),
                "session": self.snapshot(),
                "workspace": self.workspace_root.display().to_string(),
            }),
        )
        .await;
        let _ = self.save().await;
        // Always update last session on finish so stale pointers are avoided.
        let _ = self.update_last_session().await;
    }

    pub async fn save(&self) -> Result<(), String> {
        let session = self.current_session_state(Utc::now());
        self.session_store
            .save(&session)
            .await
            .map_err(|e| e.to_string())
    }

    /// Mark this session as the last active session for the workspace.
    /// Called on create, resume, run_turn, and finish to keep the pointer fresh.
    pub async fn update_last_session(&self) -> Result<(), String> {
        let store = LastSessionStore::new(
            self.workspace_root
                .join(&self.config.session.last_session_file),
        );
        store
            .save(&self.session_id)
            .await
            .map_err(|e| e.to_string())
    }

    fn current_session_state(&self, updated_at: chrono::DateTime<Utc>) -> SessionState {
        SessionState {
            meta: SessionMeta {
                id: self.session_id.clone(),
                created_at: self.created_at,
                updated_at,
                workspace: self.workspace_root.clone(),
                model: self.model.clone(),
                status: self.status.clone(),
                pid: self.pid,
                socket_path: self.socket_path.clone(),
                worktree_path: self.worktree_path.clone(),
                branch: self.branch.clone(),
                base_branch: self.base_branch.clone(),
                parent_session_id: self.parent_session_id.clone(),
                child_session_ids: self.child_session_ids.clone(),
                inherited_summary: self.inherited_summary.clone(),
                spawn_reason: self.spawn_reason.clone(),
                session_summary: self.session_summary.clone(),
                session_title: self.session_title.clone(),
                orchestration: self.orchestration.clone(),
                agent_name: self.active_agent_name.clone(),
            },
            messages: self.agent.messages.clone(),
            total_input_tokens: self.agent.cost_tracker.input_tokens,
            total_output_tokens: self.agent.cost_tracker.output_tokens,
            estimated_cost_usd: self.agent.cost_tracker.estimated_cost_usd(),
        }
    }

    pub fn snapshot(&self) -> SessionSnapshot {
        self.current_session_state(Utc::now()).snapshot()
    }

    pub fn compact_summary(&self) -> String {
        build_parent_summary(&self.agent.messages)
    }

    pub fn set_session_summary(&mut self, summary: Option<String>) {
        self.session_summary = summary.filter(|summary| !summary.trim().is_empty());
    }

    pub fn set_session_title(&mut self, title: Option<String>) {
        self.session_title = title.filter(|t| !t.trim().is_empty());
    }

    pub fn session_title(&self) -> Option<&str> {
        self.session_title.as_deref()
    }

    /// Generate a concise session title from the first user prompt using the LLM.
    /// Runs asynchronously and does not block the main turn flow.
    pub async fn generate_session_title(&mut self, first_prompt: &str) {
        if self.session_title.is_some() {
            return;
        }
        let prompt = format!(
            "Based on the user's first message below, generate a very short title \
             (at most 20 words, in the same language as the user's message) that \
             summarizes the topic of this coding session. Output ONLY the title, \
             nothing else.\n\nUser's first message:\n{first_prompt}"
        );
        match self.summarize_with_ai(&prompt).await {
            Ok(title) => {
                let cleaned = title
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'')
                    .lines()
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !cleaned.is_empty() {
                    self.set_session_title(Some(cleaned));
                }
            }
            Err(e) => {
                tracing::warn!("failed to generate session title: {e}");
            }
        }
    }

    pub async fn append_memory_note(
        &self,
        kind: &str,
        content: Option<String>,
    ) -> Result<(), String> {
        let content = content
            .map(|content| content.trim().to_string())
            .filter(|content| !content.is_empty())
            .ok_or_else(|| "memory note content is empty".to_string())?;
        let store = MemoryStore::new(self.memory_store_path());
        let note = MemoryNote {
            id: format!("{}-{}", kind, Utc::now().timestamp_millis()),
            created_at: Utc::now(),
            kind: kind.to_string(),
            title: Some(self.session_id.clone()),
            content,
        };
        store
            .append_note(note, self.config.memory.max_notes)
            .await
            .map(|_| ())
    }

    pub fn memory_store_path(&self) -> PathBuf {
        if self.config.memory.file_path.is_absolute() {
            self.config.memory.file_path.clone()
        } else {
            self.workspace_root.join(&self.config.memory.file_path)
        }
    }

    /// Switch the active agent profile at runtime.
    ///
    /// Returns the name of the persona actually in effect after the switch:
    ///
    /// - `Ok(Some(applied_name))` — a named profile was found and applied.
    /// - `Ok(None)` — no named profile is active; the session is on the
    ///   default (@orchestrator) harness persona. Reached via `name: None`,
    ///   `name == Some("orchestrator")` (when unregistered), or an
    ///   unresolvable name. Unresolvable names are deliberately non-fatal
    ///   (a resume with a dead recorded name must not hard-fail), but the
    ///   honest `None` lets callers report the fallback instead of a
    ///   false "switched" success.
    /// - `Err` — the provider rebuild failed.
    ///
    /// This rebuilds the LLM provider if the profile changes provider/model.
    /// An injected test provider (via `SupervisorConfig::provider`) is discarded here.
    pub fn apply_agent_profile(
        &mut self,
        name: Option<&str>,
    ) -> Result<Option<String>, ProviderError> {
        // Start from the clean base config (before any agent overrides).
        let config = self.base_config.clone();
        let profile = name.and_then(|n| config.agent_profile(n).cloned());
        if let Some(name) = name
            && profile.is_none()
        {
            tracing::warn!(
                agent = name,
                "agent profile not found; switching to the default harness prompt"
            );
        }

        // Profile provider/model/permission overrides apply to a transient
        // effective snapshot used only to build the provider and drive the
        // session. They must never leak into `self.config` / `base_config`:
        // those are user-level state that whole-config saves persist, and a
        // baked-in profile model used to resurface in `.nca/config.local.toml`
        // after any later `/model` / `/provider` save.
        let mut effective = config.clone();
        if let Some(ref p) = profile {
            if let Some(provider) = p.resolve_provider() {
                effective.set_default_provider(provider);
            }
            if let Some(ref model) = p.model {
                let resolved = effective.model.resolve_alias(model);
                effective.provider.set_model_for_default(resolved);
                effective.sync_default_model_from_provider();
            }
            if let Some(mode) = p.permission_mode {
                effective.permissions.mode = mode;
            }
        }

        // Rebuild provider if config changed.
        let provider = build_provider(&effective)?;
        self.config = config;
        self.model = effective.model.default_model.clone();
        let m = self.model.clone();
        self.agent.model = m;
        self.agent.replace_provider(provider);
        self.agent
            .set_keepalive_profile(cache_keepalive::resolve_profile(effective.provider.default));
        self.agent.approval.set_mode(effective.permissions.mode);

        // Store profile and rebuild system prompt. `active_agent_name` is
        // assigned HERE, on the success path only: a failed switch above
        // (e.g. `build_provider` error) must not persist the new name — the
        // next resume would re-resolve it into an unbuildable provider and
        // fail loudly on a session whose switch merely failed.
        // Truthful applied-name: Some only when a named profile actually
        // resolved and was applied.
        let applied = match (&profile, name) {
            (Some(_), Some(n)) => Some(n.to_string()),
            _ => None,
        };
        self.agent_profile = profile;
        self.active_agent_name = name.map(str::to_string);
        self.rebuild_system_prompt();
        self.rebuild_context_manager_sync();
        Ok(applied)
    }

    /// Reset for a fresh session: new ID, rebuild system prompt, clear lineage and cost.
    pub fn reset_for_new_session(&mut self) {
        self.session_id = generate_session_id();
        self.agent.messages.clear();
        self.rebuild_system_prompt();
        self.child_session_ids.clear();
        self.parent_session_id = None;
        self.inherited_summary = None;
        self.spawn_reason = None;
        self.session_summary = None;
        self.session_title = None;
        self.agent.cost_tracker = Default::default();
        self.status = SessionStatus::Running;
        self.created_at = Utc::now();
        self.last_summary_at_tokens = 0;
        self.session_store =
            SessionStore::new(self.workspace_root.join(&self.config.session.history_dir));
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn status(&self) -> &SessionStatus {
        &self.status
    }

    pub fn agent(&self) -> &AgentLoop {
        &self.agent
    }

    pub fn agent_mut(&mut self) -> &mut AgentLoop {
        &mut self.agent
    }

    /// Borrow the supervisor's merged config (includes OMO agents registered at startup).
    pub fn config(&self) -> &NcaConfig {
        &self.config
    }

    /// Mutable access to the live config, so CLI edits (e.g. `/set-editor`)
    /// land on the same snapshot later whole-config saves serialize.
    pub fn config_mut(&mut self) -> &mut NcaConfig {
        &mut self.config
    }

    /// Apply a new [`NcaConfig`] and rebuild the active LLM provider (in-session provider switch).
    /// An injected test provider (via `SupervisorConfig::provider`) is discarded here.
    pub fn apply_nca_config(&mut self, mut config: NcaConfig) -> Result<(), ProviderError> {
        let provider = build_provider(&config)?;
        // Live mounts win: callers may pass a snapshot taken before a `/mount`,
        // and adopting it verbatim lets the next whole-config save erase the
        // persisted `extra_paths` (see mount_config_persistence regression).
        let live_mounts = self.fs.mounted_paths();
        if config.extra_paths != live_mounts {
            config.extra_paths = live_mounts;
        }
        self.base_config = config.clone();
        self.config = config;
        self.model = self.config.provider.active_model().to_string();
        let provider_kind = self.config.provider.default;
        let m = self.model.clone();
        let agent = self.agent_mut();
        agent.model = m;
        agent.replace_provider(provider);
        agent.set_keepalive_profile(cache_keepalive::resolve_profile(provider_kind));
        self.rebuild_context_manager_sync();
        Ok(())
    }

    /// Rebuild context_manager from current config (sync, uses configured window target).
    fn rebuild_context_manager_sync(&mut self) {
        let ctx = &self.config.memory.context;
        let window = if ctx.context_window_target > 0 {
            ctx.context_window_target
        } else {
            128_000
        };
        let context_config = ContextManagerConfig {
            context_window_target: window,
            max_retained_messages: ctx.max_retained_messages,
            auto_summarize_threshold: ctx.auto_summarize_threshold,
            enable_auto_summarize: ctx.enable_auto_summarize,
            max_message_chars_for_summary: 10000,
        };
        self.context_manager = ContextManager::new(context_config, self.model.clone());
    }

    /// Rebuild the system prompt from the current config/profile/mounts and
    /// apply it to the agent. Called on profile switch, new session, and mount.
    fn rebuild_system_prompt(&mut self) {
        let mounted = self.fs.mounted_paths();
        let system_prompt = build_system_prompt_with_agent(
            &self.config,
            &self.workspace_root,
            &self.plugins,
            self.orchestration.as_ref(),
            self.agent_profile.as_ref(),
            &mounted,
        );
        self.agent.set_system_prompt(system_prompt);
    }

    // ── Mount management ─────────────────────────────────────────────

    /// Rebuild the PTY sandbox policy from the current config and live
    /// mounts. Called at `create` (after restoring persisted mounts) and after
    /// every `/mount` + `/unmount`, so shell-command visibility tracks
    /// file-tool visibility without a session restart. Honors
    /// `[permissions.sandbox] inherit_mounts` (default on).
    fn refresh_sandbox_policy(&self) {
        self.pty.set_sandbox_config(
            self.config.permissions.sandbox.clone(),
            &self.fs.mounted_paths(),
            &SkillCatalog::discovery_roots(
                &self.workspace_root,
                &self.config.harness.skill_directories,
            ),
        );
    }

    /// Mount an additional directory so tools can access files outside the workspace root.
    ///
    /// Keeps the in-memory `config` / `base_config` `extra_paths` in lockstep
    /// with the live `RealFs` mount list. Without this, any later whole-config
    /// save (`/model`, `/thinking`, `/set-editor`, … — they all serialize
    /// `Supervisor::config`) rewrote `.nca/config.local.toml` from the stale
    /// pre-mount snapshot and silently erased the freshly persisted entry.
    pub async fn mount_path(&mut self, path: &Path) -> Result<(), String> {
        self.fs.mount_path(path).map_err(|e| e.to_string())?;
        let paths = self.fs.mounted_paths();
        self.config.extra_paths = paths.clone();
        self.base_config.extra_paths = paths.clone();
        persist_mounted_paths(&self.workspace_root, paths).await;
        self.rebuild_system_prompt();
        self.refresh_sandbox_policy();
        Ok(())
    }

    /// Unmount a previously mounted directory.
    ///
    /// Mirrors [`Self::mount_path`]: syncs the in-memory `extra_paths` and
    /// awaits persistence, so the removal reaches disk before the caller learns
    /// of success and later config saves cannot resurrect the entry.
    pub async fn unmount_path(&mut self, path: &Path) -> Result<(), String> {
        self.fs.unmount_path(path).map_err(|e| e.to_string())?;
        let paths = self.fs.mounted_paths();
        self.config.extra_paths = paths.clone();
        self.base_config.extra_paths = paths.clone();
        persist_mounted_paths(&self.workspace_root, paths).await;
        self.rebuild_system_prompt();
        self.refresh_sandbox_policy();
        Ok(())
    }

    /// List currently mounted extra paths.
    pub fn mounted_paths(&self) -> Vec<PathBuf> {
        self.fs.mounted_paths()
    }

    /// Return the live filesystem adapter. Used by callers (e.g.
    /// `spawn_subagent_consumer`) to query the current set of mounted paths
    /// at runtime, so that paths added via `/mount` propagate to child sessions.
    pub fn fs(&self) -> Arc<dyn WorkspaceFs> {
        self.fs.clone()
    }

    /// Shared handle to the plugin registry (G3): lets the spawn consumer
    /// fire `subagentDispatch` hooks with the parent-rooted plugin instances.
    pub fn plugin_registry(&self) -> Arc<PluginRegistry> {
        Arc::clone(&self.plugins)
    }

    /// Collect slash commands contributed by plugins (for CLI slash panel).
    pub fn plugin_commands(&self) -> Vec<(String, Vec<String>)> {
        self.plugins.collect_commands()
    }

    /// Dispatch a slash command to plugins (command interception).
    /// Returns `Some((plugin_name, result))` if a plugin handled the command.
    pub fn check_command_before(
        &self,
        command: &str,
        arguments: &str,
    ) -> Option<(String, nca_core::plugin::CommandIntercept)> {
        self.plugins.check_command_before(command, arguments)
    }

    /// G6: echo an intercepted plugin command's output into the LLM
    /// conversation as a system-role note (opt-in per interception via
    /// `CommandIntercept.echo`). Pure history append — no turn is triggered,
    /// so the model observes the state change on the user's next submit.
    pub async fn record_plugin_command_echo(&mut self, plugin: &str, text: &str) {
        let note = format!("[plugin:{plugin}] {text}");
        self.agent.record_system_note(&note).await;
    }

    /// G7: surface plugins that got disabled since the last check as
    /// system lines in the transcript (a disable without notice looks like
    /// plugins silently vanishing). Called at turn start — cheap when clean.
    fn surface_plugin_disables(&mut self) {
        let Some(host) = self.plugin_host.as_ref() else {
            return;
        };
        for (name, reason) in host.disabled_plugins() {
            if self.plugin_disable_reported.insert(name.clone()) {
                let note = format!(
                    "[plugin] {name} disabled ({reason}) — tools and hooks from this plugin are unavailable; /plugin refresh to restart"
                );
                tracing::warn!("{note}");
                if let Some(tx) = self.agent.event_sender() {
                    let _ = tx.try_send(AgentEvent::MessageReceived {
                        role: "system".into(),
                        content: note,
                        steering: false,
                    });
                }
            }
        }
    }

    /// G7 `/plugin refresh`: restart every plugin process, swap the live
    /// registry contents in place (spawn consumers keep observing the same
    /// `Arc<PluginRegistry>`), re-register plugin tools, and rebuild the
    /// system prompt. Returns a human-readable status line.
    pub async fn refresh_plugins(&mut self) -> Result<String, String> {
        let Some(host) = self.plugin_host.as_mut() else {
            return Err("no plugin host: safe mode or no plugins discovered".into());
        };

        // Old plugin tool names must leave the ToolRegistry (fresh instances
        // re-register below; a name that vanished between refreshes would
        // otherwise stay callable-but-dead).
        let old_tool_names: Vec<String> = self
            .plugins
            .iter()
            .into_iter()
            .flat_map(|p| p.tools().into_iter().map(|t| t.name))
            .collect();

        let perm_mode = serde_json::to_string(&self.config.permissions.mode)
            .unwrap_or_else(|_| "\"default\"".into());
        let perm_mode = perm_mode.trim_matches('"');
        let errors = host
            .refresh(&self.workspace_root, &self.session_id, perm_mode)
            .await;

        self.plugins.adopt(host.registry());
        self.plugin_disable_reported.clear();

        {
            let plugin_cfg = self.config.plugins.clone();
            let plugins = Arc::clone(&self.plugins);
            let tools = &mut self.agent_mut().tools;
            for name in &old_tool_names {
                tools.unregister(name);
            }
            nca_core::tools::plugin_tool::register_plugin_tools(tools, &plugins, &plugin_cfg);
        }
        self.rebuild_system_prompt();

        // Fresh handshake sent a new Config — replay the session-start event.
        self.plugins.notify_event(&serde_json::json!({
            "type": "session_start",
            "session_id": self.session_id,
        }));

        let count = self.plugins.len();
        if errors.is_empty() {
            Ok(format!("refreshed: {count} plugin(s) running"))
        } else {
            Ok(format!(
                "refreshed: {count} plugin(s) running; errors: {}",
                errors.join("; ")
            ))
        }
    }

    /// G7 `/plugin status`: plugin names, health, and declared surface.
    pub fn plugin_status(&self) -> Vec<String> {
        let mut lines = Vec::new();
        let disabled: HashMap<String, String> = self
            .plugin_host
            .as_ref()
            .map(|h| h.disabled_plugins().into_iter().collect())
            .unwrap_or_default();
        if self.plugins.is_empty() {
            lines.push("no plugins loaded".into());
            return lines;
        }
        for plugin in self.plugins.iter() {
            let name = plugin.name().to_string();
            let tools = plugin.tools().len();
            let commands = plugin.commands().len();
            let health = match disabled.get(&name) {
                Some(reason) => format!("DISABLED ({reason})"),
                None => "running".to_string(),
            };
            lines.push(format!(
                "{name}: {health}, {tools} tool(s), {commands} command(s)"
            ));
        }
        lines
    }

    pub fn request_cancel(&self) {
        self.agent.request_cancel();
    }

    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        self.agent.cancel_handle()
    }

    pub fn set_worktree_info(
        &mut self,
        worktree_path: PathBuf,
        branch: String,
        base_branch: String,
    ) {
        self.worktree_path = Some(worktree_path);
        self.branch = Some(branch);
        self.base_branch = Some(base_branch);
    }

    /// Switch the supervisor to use a worktree as its workspace root.
    /// Updates the filesystem adapter and PTY manager to point to the new root,
    /// so all file operations and shell commands run inside the worktree directory.
    pub fn switch_to_worktree(
        &mut self,
        worktree_path: PathBuf,
        branch: String,
        base_branch: String,
    ) {
        // Update worktree metadata.
        self.worktree_path = Some(worktree_path.clone());
        self.branch = Some(branch);
        self.base_branch = Some(base_branch);

        // Sync the filesystem root to the worktree path.
        // Since all file tools (write_file, edit_file, etc.) share the same
        // Arc<dyn WorkspaceFs> through their own clones, updating the root in-place
        // immediately redirects all subsequent file operations to the worktree.
        if let Err(e) = self.fs.set_root(worktree_path.clone()) {
            tracing::warn!("failed to update fs root to worktree: {e}");
        }

        // Sync the PTY root so shell commands (execute_bash) also run
        // inside the worktree directory.
        self.pty.set_root(&worktree_path);

        // Update the supervisor's own record of the workspace root.
        self.workspace_root = worktree_path;
    }

    pub fn set_parent(
        &mut self,
        parent_id: String,
        summary: Option<String>,
        reason: Option<String>,
    ) {
        self.parent_session_id = Some(parent_id);
        self.inherited_summary = summary;
        self.spawn_reason = reason;
    }

    pub fn add_child(&mut self, child_id: String) {
        if !self.child_session_ids.contains(&child_id) {
            self.child_session_ids.push(child_id);
        }
    }

    pub fn event_tx(&self) -> Option<tokio::sync::mpsc::Sender<AgentEvent>> {
        self.agent.event_sender()
    }

    /// Handle for enqueueing user prompts / steering into the running or
    /// next turn. Delegates to the agent loop's bounded inbox (capacity 16);
    /// `try_send` failure means "inbox full" and should be surfaced by the
    /// caller.
    pub fn inbox_sender(&self) -> mpsc::Sender<InboxItem> {
        self.agent.inbox_sender()
    }

    pub fn session_store(&self) -> &SessionStore {
        &self.session_store
    }

    fn refresh_session_summary(&mut self) {
        self.set_session_summary(Some(self.compact_summary()));
    }

    async fn run_session_hook(&self, event: HookEventKind, payload: serde_json::Value) {
        if let Some(hooks) = &self.hooks {
            hooks.run_best_effort(event, None, &payload).await;
        }
    }
}

/// Discover plugin descriptors from environment variables + default paths.
///
/// Resolution order:
/// 1. `NCA_PLUGIN_<NAME>_BIN` env var (e.g. `NCA_PLUGIN_MYPLUGIN_BIN`)
/// 2. `<name>-plugin` in PATH
///
/// IPC-based approval handler that waits for approve/deny commands from
/// connected clients (e.g. CLI over the session socket).
pub struct IpcApprovalHandler {
    pending: ApprovalPendingMap,
}

impl IpcApprovalHandler {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            pending: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn pending(&self) -> ApprovalPendingMap {
        self.pending.clone()
    }
}

#[async_trait::async_trait]
impl ApprovalHandler for IpcApprovalHandler {
    async fn resolve(
        &self,
        call: &nca_common::tool::ToolCall,
        _description: &str,
    ) -> ApprovalVerdict {
        let (tx, rx) = oneshot::channel();
        {
            let mut m = self.pending.lock().unwrap();
            m.insert(call.id.clone(), tx);
        }
        match rx.await {
            Ok(verdict) => verdict,
            Err(_) => {
                let mut m = self.pending.lock().unwrap();
                m.remove(&call.id);
                ApprovalVerdict::Denied
            }
        }
    }
}

/// Auto-deny handler for non-interactive sessions.
pub(crate) struct AutoDenyHandler;

#[async_trait::async_trait]
impl ApprovalHandler for AutoDenyHandler {
    async fn resolve(
        &self,
        _call: &nca_common::tool::ToolCall,
        _description: &str,
    ) -> ApprovalVerdict {
        ApprovalVerdict::Denied
    }
}

fn generate_session_id() -> String {
    static SESSION_COUNTER: AtomicU64 = AtomicU64::new(0);
    let counter = SESSION_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("session-{}-{counter}", Utc::now().timestamp_micros())
}

/// Known OMO specialist agent names that auto-register as agent profiles.
///
/// `tester` is split from `fixer` by design: they should use different models
/// for cross-model verification (code written by one model, tested by another).
const SPECIALIST_NAMES: &[&str] = &[
    "explorer",
    "oracle",
    "librarian",
    "designer",
    "fixer",
    "tester",
    "observer",
    "council",
];

/// Check if a skill command name is a known OMO specialist.
fn is_specialist_skill(command: &str) -> bool {
    SPECIALIST_NAMES.contains(&command)
}

/// Scan skill directories and register specialist skills as agent profiles.
///
/// A skill becomes an agent profile candidate if:
/// - It has `agent: true` in its frontmatter, OR
/// - Its command name is a known OMO specialist (`explorer`, `oracle`, etc.)
///
/// Profiles are only inserted if not already defined in the user's config
/// (explicit `[agents.<name>]` takes precedence).
pub(crate) fn register_skill_agents(config: &mut NcaConfig, workspace_root: &Path) {
    let skills = match SkillCatalog::discover(workspace_root, &config.harness.skill_directories) {
        Ok(skills) => skills,
        Err(e) => {
            tracing::debug!("skill discovery for agent registration failed: {e}");
            return;
        }
    };

    for skill in &skills {
        if !skill.agent && !is_specialist_skill(&skill.command) {
            continue;
        }

        // Don't override a profile already defined by the user.
        if config.agents.contains_key(&skill.command) {
            continue;
        }

        let profile = AgentProfileConfig {
            description: skill.description.clone(),
            provider: skill.provider,
            model: skill.model.clone(),
            permission_mode: skill.permission_mode,
            system_prompt: Some(skill.expanded_body()),
            system_prompt_append: None,
            allowed_tools: None,
        };
        config.agents.insert(skill.command.clone(), profile);
    }
}

/// Pure: should a `run_turn` error trigger the supervisor's one-shot
/// summarize-retry (P3 §5)? Exactly one retry per `run_turn_with_images`
/// call; the middleware view-prune arm has already run by the time an
/// overflow reaches here.
pub(crate) fn should_overflow_retry(err: &ProviderError, already_retried: bool) -> bool {
    !already_retried && err.is_context_overflow()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nca_common::session::ChildSessionState;

    /// Offline default config for supervisor construction in tests (mirrors
    /// the integration suites' `offline_config`).
    async fn offline_supervisor(root: &Path) -> Supervisor {
        let mut config = NcaConfig::default();
        config.provider.deepseek.api_key = Some("test-key".into());
        config.permissions.mode = nca_common::config::PermissionMode::BypassPermissions;
        config.memory.context.auto_detect_context_window = false;
        config.memory.context.query_provider_models_api = false;
        config.memory.context.enable_auto_summarize = false;
        Supervisor::create(SupervisorConfig {
            config,
            workspace_root: root.to_path_buf(),
            safe_mode: false,
            interactive_approvals: false,
            session_id: Some("pin-resume-worktree".into()),
            approval_handler: None,
            orchestration_context: None,
            agent_name: None,
            provider: None,
        })
        .await
        .expect("offline supervisor")
    }

    /// P2 C2 finding, pinned: `Supervisor::resume` restores the worktree
    /// FIELDS (`workspace_root`/`worktree_path`/`branch`/`base_branch` from
    /// the child's own meta) but does NOT re-root the filesystem adapter or
    /// PTY manager — `fs.root()` stays the caller's root until an explicit
    /// `switch_to_worktree`. Any revive-style resume MUST therefore call
    /// `switch_to_worktree` explicitly, or the revived child's file tools
    /// and shell commands silently run in the PARENT workspace.
    #[tokio::test(flavor = "multi_thread")]
    async fn resume_restores_worktree_fields_but_not_fs_root() {
        let ws = tempfile::tempdir().expect("tempdir");
        let wt = tempfile::tempdir().expect("worktree tempdir");
        let wt_path = wt.path().canonicalize().expect("canonicalize wt");

        let mut sup = offline_supervisor(ws.path()).await;
        sup.set_worktree_info(wt_path.clone(), "nca/pin".into(), "main".into());
        sup.save().await.expect("persist meta with worktree info");
        drop(sup);

        let mut config = NcaConfig::default();
        config.provider.deepseek.api_key = Some("test-key".into());
        let mut resumed = Supervisor::resume(
            config,
            ws.path(),
            false,
            false,
            "pin-resume-worktree",
            None,
            None,
        )
        .await
        .expect("resume");

        // Fields restored from meta…
        assert_eq!(resumed.worktree_path.as_deref(), Some(wt_path.as_path()));
        assert_eq!(resumed.branch.as_deref(), Some("nca/pin"));
        // …but the fs adapter is still rooted at the caller's workspace.
        let fs_root = resumed.fs.root().canonicalize().expect("canonical fs root");
        assert_ne!(
            fs_root, wt_path,
            "resume must NOT re-root the fs adapter on its own — revive has to \
             call switch_to_worktree explicitly"
        );

        // The explicit switch (what task_revive does) fixes the cwd.
        resumed.switch_to_worktree(wt_path.clone(), "nca/pin".into(), "main".into());
        assert_eq!(
            resumed.fs.root().canonicalize().expect("canonical fs root"),
            wt_path
        );
    }

    fn spawned_envelope(id: u64, child: &str) -> EventEnvelope {
        EventEnvelope::new(
            id,
            AgentEvent::ChildSessionSpawned {
                parent_session_id: "parent".into(),
                child_session_id: child.into(),
                task: "t".into(),
                workspace: PathBuf::from("/ws"),
                branch: None,
            },
        )
    }

    // C9 — overflow summarize-retry decision table.
    #[test]
    fn should_overflow_retry_decision_table() {
        let overflow = ProviderError::RequestFailed(
            "This model's maximum context length is 65536 tokens".into(),
        );
        let other = ProviderError::AuthError("invalid api key".into());
        assert!(
            should_overflow_retry(&overflow, false),
            "first overflow retries"
        );
        assert!(
            !should_overflow_retry(&overflow, true),
            "exactly one retry — second overflow escalates"
        );
        assert!(
            !should_overflow_retry(&other, false),
            "non-overflow never retries"
        );
        assert!(
            !should_overflow_retry(&other, true),
            "non-overflow + already-retried never retries"
        );
    }

    #[test]
    fn fold_child_session_ids_unions_deduped_and_ordered() {
        let meta = vec!["a".to_string(), "b".to_string()];
        let envelopes = vec![
            spawned_envelope(1, "b"), // duplicate of json — deduped
            spawned_envelope(2, "c"), // log-only — appended
            spawned_envelope(3, "a"), // duplicate — deduped
        ];
        let folded = fold_child_session_ids(&meta, &envelopes);
        assert_eq!(
            folded,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn fold_child_session_ids_empty_cases() {
        assert!(fold_child_session_ids(&[], &[]).is_empty());
        assert_eq!(
            fold_child_session_ids(&[], &[spawned_envelope(1, "x")]),
            vec!["x".to_string()]
        );
        assert_eq!(
            fold_child_session_ids(&["y".to_string()], &[]),
            vec!["y".to_string()]
        );
    }

    #[test]
    fn fold_child_session_ids_ignores_other_events() {
        let envelopes = vec![EventEnvelope::new(
            1,
            AgentEvent::TurnStarted { turn_id: 1 },
        )];
        assert_eq!(
            fold_child_session_ids(&["z".to_string()], &envelopes),
            vec!["z".to_string()]
        );
    }

    #[test]
    fn resume_fold_rebuilds_registry_from_envelopes() {
        // P1: the registry is an in-memory projection — resume must rebuild
        // it from the same envelopes used for the lineage fold. A
        // Spawned+Completed pair yields a terminal entry with the mapped
        // state; a lone Spawned stays Running.
        let registry = crate::subagent_registry::SubagentRegistry::new();
        let envelopes = vec![
            spawned_envelope(1, "done-child"),
            EventEnvelope::new(
                2,
                AgentEvent::ChildSessionCompleted {
                    parent_session_id: "parent".into(),
                    child_session_id: "done-child".into(),
                    status: "completed".into(),
                },
            ),
            spawned_envelope(3, "live-child"),
            // A Completed for a child whose Spawned never made the log (torn
            // tail) is intentionally ignored — mirrors `fold_child_session_ids`,
            // which can also only recover ids it saw spawn.
            EventEnvelope::new(
                4,
                AgentEvent::ChildSessionCompleted {
                    parent_session_id: "parent".into(),
                    child_session_id: "ghost-child".into(),
                    status: "completed".into(),
                },
            ),
            spawned_envelope(5, "err-child"),
            EventEnvelope::new(
                6,
                AgentEvent::ChildSessionCompleted {
                    parent_session_id: "parent".into(),
                    child_session_id: "err-child".into(),
                    status: "error".into(),
                },
            ),
        ];
        for envelope in &envelopes {
            registry.apply_envelope(envelope);
        }

        let done = registry.get("done-child").expect("done entry");
        assert_eq!(done.state, ChildSessionState::Completed);
        assert_eq!(done.task, "t");

        let live = registry.get("live-child").expect("live entry");
        assert_eq!(live.state, ChildSessionState::Running);

        let errored = registry.get("err-child").expect("err entry");
        assert_eq!(errored.state, ChildSessionState::Failed);

        // The ghost Completed (no Spawned on record) must NOT create an entry.
        assert!(registry.get("ghost-child").is_none());

        // Spawn order preserved across the fold.
        let ids: Vec<String> = registry.list().into_iter().map(|e| e.session_id).collect();
        assert_eq!(
            ids,
            vec![
                "done-child".to_string(),
                "live-child".into(),
                "err-child".into()
            ]
        );
    }

    #[test]
    fn resume_fold_completed_after_status_changed_keeps_summary() {
        // The richer StatusChanged carries the result summary; a later
        // coarser Completed must not wipe it during the resume fold.
        let registry = crate::subagent_registry::SubagentRegistry::new();
        let envelopes = vec![
            spawned_envelope(1, "c1"),
            EventEnvelope::new(
                2,
                AgentEvent::ChildSessionStatusChanged {
                    parent_session_id: "parent".into(),
                    child_session_id: "c1".into(),
                    state: ChildSessionState::Failed,
                    generation: 0,
                    alias: None,
                    result_summary: Some("provider 502".into()),
                },
            ),
            EventEnvelope::new(
                3,
                AgentEvent::ChildSessionCompleted {
                    parent_session_id: "parent".into(),
                    child_session_id: "c1".into(),
                    status: "error".into(),
                },
            ),
        ];
        for envelope in &envelopes {
            registry.apply_envelope(envelope);
        }
        let entry = registry.get("c1").expect("entry");
        assert_eq!(entry.state, ChildSessionState::Failed);
        assert_eq!(entry.result_summary.as_deref(), Some("provider 502"));
    }

    #[test]
    fn select_resume_messages_replaces_stale_system_prompts_with_fresh() {
        // T11: json carries stale (and duplicated) system prompts; resume must
        // drop them all and prepend only the fresh system message. Old-format
        // log (no surface events) ⇒ json snapshot path.
        let dir = tempfile::tempdir().expect("tempdir");
        let stale1 = Message::system("old prompt v1");
        let stale2 = Message::system("old prompt v2");
        let user = Message::user("hello");
        let fresh = vec![Message::system("fresh prompt")];

        let (msgs, source) = select_resume_messages(
            Some(vec![stale1, user.clone(), stale2, Message::assistant("hi")]),
            Vec::new(),
            false,
            fresh.clone(),
            dir.path(),
        );

        assert_eq!(source, ResumeMessageSource::Snapshot);
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0], fresh[0]);
        assert!(matches!(msgs[1].role, Role::User));
        assert!(matches!(msgs[2].role, Role::Assistant));
    }

    #[test]
    fn select_resume_messages_replay_fallback_repairs_missing_images() {
        // T7 unit half: json unloadable + replay non-empty → fallback with
        // fresh system first, and image parts whose file is gone collapse to
        // a text placeholder.
        let dir = tempfile::tempdir().expect("tempdir");
        let kept_image = dir.path().join("attachments/kept.png");
        std::fs::create_dir_all(dir.path().join("attachments")).expect("mkdir");
        std::fs::write(&kept_image, b"png").expect("write image");

        let user = Message::user_with_parts(vec![
            ContentPart::Text {
                text: "look at these".into(),
            },
            ContentPart::Image {
                media_type: "image/png".into(),
                path: "attachments/kept.png".into(),
            },
            ContentPart::Image {
                media_type: "image/png".into(),
                path: "attachments/deleted.png".into(),
            },
        ]);
        let fresh = vec![Message::system("fresh prompt")];

        let (msgs, source) =
            select_resume_messages(None, vec![user], true, fresh.clone(), dir.path());

        assert_eq!(source, ResumeMessageSource::ReplayFallback);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0], fresh[0]);
        // The kept image file exists → survives; the deleted one becomes a
        // text placeholder.
        let MessageContent::Parts(parts) = &msgs[1].content else {
            panic!("expected parts content");
        };
        assert_eq!(parts.len(), 3);
        assert!(matches!(parts[1], ContentPart::Image { .. }));
        assert!(matches!(&parts[2], ContentPart::Text { text } if text.contains("deleted.png")));
    }

    #[test]
    fn select_resume_messages_empty_snapshot_and_empty_replay_yields_fresh_system_only() {
        let dir = tempfile::tempdir().expect("tempdir");
        let fresh = vec![Message::system("fresh")];
        let (msgs, source) =
            select_resume_messages(Some(Vec::new()), Vec::new(), true, fresh, dir.path());
        assert_eq!(source, ResumeMessageSource::Snapshot);
        assert_eq!(msgs.len(), 1);
        assert!(matches!(msgs[0].role, Role::System));
    }

    #[test]
    fn select_resume_messages_replay_rescues_empty_json() {
        let dir = tempfile::tempdir().expect("tempdir");
        let replay = vec![Message::user("from log")];
        let fresh = vec![Message::system("fresh")];
        let (msgs, source) =
            select_resume_messages(Some(Vec::new()), replay.clone(), true, fresh, dir.path());
        assert_eq!(source, ResumeMessageSource::ReplayAuthoritative);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1], replay[0]);
    }

    #[test]
    fn t18_fresh_log_replay_beats_divergent_json() {
        // Phase B flip: fresh-format log (surface events present) with a
        // non-empty projection wins even when the json is healthy and the
        // two diverge — the json is a cache.
        let dir = tempfile::tempdir().expect("tempdir");
        let snap = vec![Message::user("stale json turn")];
        let replay = vec![Message::user("fresh log turn")];
        let fresh = vec![Message::system("fresh")];
        let (msgs, source) =
            select_resume_messages(Some(snap), replay.clone(), true, fresh, dir.path());
        assert_eq!(source, ResumeMessageSource::ReplayAuthoritative);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1], replay[0]);
    }

    #[test]
    fn t19_old_format_log_keeps_snapshot() {
        // Old logs (pre-Phase-A, no MessageRecorded) fold to empty — the json
        // snapshot must keep winning for them.
        let dir = tempfile::tempdir().expect("tempdir");
        let snap = vec![Message::user("from json")];
        let fresh = vec![Message::system("fresh")];
        let (msgs, source) = select_resume_messages(
            Some(snap.clone()),
            Vec::new(), // old-format log folds to empty
            false,
            fresh,
            dir.path(),
        );
        assert_eq!(source, ResumeMessageSource::Snapshot);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1], snap[0]);
    }

    #[test]
    fn divergence_comparison_fires_only_on_real_difference() {
        // T10 pure half: normalization drops system messages; identical
        // non-system histories compare equal.
        let msgs = vec![Message::system("s"), Message::user("u")];
        let again = vec![Message::user("u"), Message::system("other s")];
        assert_eq!(
            normalize_resume_projection(&msgs),
            normalize_resume_projection(&again)
        );
        let different = vec![Message::user("different")];
        assert_ne!(
            normalize_resume_projection(&msgs),
            normalize_resume_projection(&different)
        );
    }

    #[test]
    fn divergence_comparison_tolerates_attachment_placeholders() {
        // The snapshot stores the post-cleanup placeholder; the event record
        // stores the original image part. Repairing the record (file gone)
        // must make the two compare equal — no spurious divergence warning.
        let dir = tempfile::tempdir().expect("tempdir");
        let snapshot = vec![Message::user(
            "look\n[image processed and removed after send: gone.png]",
        )];
        let recorded = vec![Message::user_with_parts(vec![
            ContentPart::Text {
                text: "look".into(),
            },
            ContentPart::Image {
                media_type: "image/png".into(),
                path: "gone.png".into(),
            },
        ])];
        let repaired = repair_missing_images(recorded, dir.path());
        assert_eq!(
            normalize_resume_projection(&repaired),
            normalize_resume_projection(&snapshot)
        );
    }

    #[test]
    fn read_event_log_parses_envelopes_legacy_and_skips_garbage() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("log.events.jsonl");
        let env_line = serde_json::to_string(&EventEnvelope::new(
            1,
            AgentEvent::TurnStarted { turn_id: 7 },
        ))
        .expect("serialize envelope");
        let legacy = serde_json::to_string(&AgentEvent::TurnStarted { turn_id: 3 })
            .expect("serialize bare event");
        let content = format!(
            "{env_line}\n{legacy}\n{{not json at all\n{{\"id\":2,\"event\":{{\"type\":\"Ga",
            env_line = env_line,
            legacy = legacy,
        );
        std::fs::write(&path, content).expect("write log");

        let envelopes = crate::session_store::read_event_log(&path);
        assert_eq!(envelopes.len(), 2, "garbage/torn lines must be skipped");
        assert!(matches!(
            envelopes[0].event,
            AgentEvent::TurnStarted { turn_id: 7 }
        ));
        assert!(matches!(
            envelopes[1].event,
            AgentEvent::TurnStarted { turn_id: 3 }
        ));
    }

    #[test]
    fn read_event_log_missing_file_is_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(crate::session_store::read_event_log(&dir.path().join("nope.jsonl")).is_empty());
    }

    #[test]
    fn resolve_resume_workspace_root_adopts_current_when_stored_path_vanished() {
        let dir = tempfile::tempdir().expect("tempdir");
        let old = dir.path().join("project-old");
        std::fs::create_dir_all(&old).expect("mkdir");
        let current = dir.path().canonicalize().expect("canonicalize current");

        // Stored path still exists and differs → legacy behavior keeps it
        // (e.g. worktree-linked sessions resumed from another root).
        assert_eq!(resolve_resume_workspace_root(&current, &old), old);

        // Directory renamed: stored path is gone → adopt the current root so
        // later config saves do not resurrect the old tree.
        let renamed = dir.path().join("project-new");
        std::fs::rename(&old, &renamed).expect("rename");
        assert_eq!(resolve_resume_workspace_root(&current, &old), current);

        // Equal paths are a no-op.
        assert_eq!(resolve_resume_workspace_root(&current, &current), current);
    }

    #[test]
    fn is_specialist_skill_recognizes_known_names() {
        assert!(is_specialist_skill("explorer"));
        assert!(is_specialist_skill("oracle"));
        assert!(is_specialist_skill("librarian"));
        assert!(is_specialist_skill("designer"));
        assert!(is_specialist_skill("fixer"));
        assert!(is_specialist_skill("tester"));
        assert!(is_specialist_skill("observer"));
        assert!(is_specialist_skill("council"));
    }

    #[test]
    fn is_specialist_skill_rejects_unknown_names() {
        assert!(!is_specialist_skill("code-reviewer"));
        assert!(!is_specialist_skill("brainstorming"));
        assert!(!is_specialist_skill("random-skill"));
    }

    #[test]
    fn register_skill_agents_registers_specialist_skill() {
        let dir = tempfile::tempdir().expect("tempdir");
        let empty_xdg = tempfile::tempdir().expect("tempdir for xdg isolation");

        // Isolate from real XDG config (which may have explorer/oracle/etc.
        // installed). EnvGuard takes the crate env lock so parallel tests
        // reading XDG_CONFIG_HOME (e.g. sandbox policy derivation) cannot see
        // a half-swapped environment, and restores the ORIGINAL value on
        // drop (the old raw remove_var leaked an unset XDG_CONFIG_HOME into
        // every later test when the runner had one set).
        let _env =
            crate::test_util::EnvGuard::set(&[("XDG_CONFIG_HOME", empty_xdg.path().to_str())]);

        let skill_dir = dir.path().join(".nca/skills/explorer");
        std::fs::create_dir_all(&skill_dir).expect("mkdir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: Explorer\ncommand: explorer\ndescription: Explores code\n---\nYou are an explorer.\n",
        )
        .expect("write");

        let mut config = NcaConfig::default();
        register_skill_agents(&mut config, dir.path());

        let profile = config
            .agent_profile("explorer")
            .expect("profile should exist");
        assert!(
            profile
                .system_prompt
                .as_ref()
                .unwrap()
                .contains("You are an explorer")
        );
        assert_eq!(profile.description.as_deref(), Some("Explores code"));
    }

    #[test]
    fn register_skill_agents_registers_agent_flagged_skill() {
        let dir = tempfile::tempdir().expect("tempdir");
        let skill_dir = dir.path().join(".nca/skills/custom-agent");
        std::fs::create_dir_all(&skill_dir).expect("mkdir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: Custom Agent\ncommand: custom-agent\nagent: true\n---\nCustom persona.\n",
        )
        .expect("write");

        let mut config = NcaConfig::default();
        register_skill_agents(&mut config, dir.path());

        assert!(config.agent_profile("custom-agent").is_some());
    }

    #[test]
    fn register_skill_agents_skips_non_agent_skills() {
        let dir = tempfile::tempdir().expect("tempdir");
        let skill_dir = dir.path().join(".nca/skills/regular-skill");
        std::fs::create_dir_all(&skill_dir).expect("mkdir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: Regular Skill\ncommand: regular-skill\n---\nJust a skill.\n",
        )
        .expect("write");

        let mut config = NcaConfig::default();
        register_skill_agents(&mut config, dir.path());

        assert!(config.agent_profile("regular-skill").is_none());
    }

    #[test]
    fn register_skill_agents_does_not_override_existing_profile() {
        let dir = tempfile::tempdir().expect("tempdir");
        let skill_dir = dir.path().join(".nca/skills/oracle");
        std::fs::create_dir_all(&skill_dir).expect("mkdir");
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: Oracle\ncommand: oracle\n---\nSkill body.\n",
        )
        .expect("write");

        let mut config = NcaConfig::default();
        // Pre-define an oracle profile
        config.agents.insert(
            "oracle".into(),
            AgentProfileConfig {
                system_prompt: Some("User-defined oracle.".into()),
                ..Default::default()
            },
        );
        register_skill_agents(&mut config, dir.path());

        // Should keep user-defined profile
        let profile = config.agent_profile("oracle").expect("profile");
        assert_eq!(
            profile.system_prompt.as_deref(),
            Some("User-defined oracle.")
        );
    }

    struct StubProvider;

    #[async_trait::async_trait]
    impl Provider for StubProvider {
        async fn chat(
            &self,
            _messages: &[Message],
            _tools: &[nca_common::tool::ToolDefinition],
            _model: &str,
            _workspace_root: &Path,
        ) -> Result<tokio::sync::mpsc::Receiver<nca_core::provider::StreamChunk>, ProviderError>
        {
            unreachable!("resolve_provider does not call chat")
        }
    }

    #[test]
    fn resolve_provider_uses_injected_verbatim_else_builds() {
        // (a) injected provider is returned verbatim — `build_provider` skipped.
        let mock: Arc<dyn Provider> = Arc::new(StubProvider);
        let injected = match resolve_provider(Some(mock.clone()), &NcaConfig::default()) {
            Ok(p) => p,
            Err(e) => panic!("injected provider should be used verbatim, got: {e}"),
        };
        assert!(Arc::ptr_eq(&injected, &mock));

        // (b) no injection → builds from config (DeepSeek default needs a key).
        let mut with_key = NcaConfig::default();
        with_key.provider.deepseek.api_key = Some("test-key".into());
        assert!(resolve_provider(None, &with_key).is_ok());

        // (c) no injection + keyless config → loud configuration error.
        // DeepSeek (the default) validates its key lazily at request time, so
        // use OpenAI here — its `from_config` fails loudly on a missing key.
        let mut keyless = NcaConfig::default();
        keyless.provider.default = nca_common::config::ProviderKind::OpenAi;
        let err = match resolve_provider(None, &keyless) {
            Ok(_) => panic!("keyless config should fail to build a provider"),
            Err(e) => e,
        };
        assert!(matches!(err, ProviderError::Configuration(_)));
    }
}
