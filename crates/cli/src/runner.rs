use crate::ipc_pending::{ApprovalPendingMap, QuestionPendingMap};
use nca_common::config::{NcaConfig, PermissionMode, PlanEntry, ProviderKind, SandboxMode};
use nca_common::event::{AgentEvent, EndReason, QuestionSelection};
use nca_common::session::{OrchestrationContext, SessionSnapshot};
use nca_core::agent_driver::InboxItem;
use nca_core::approval::{ApprovalHandler, ApprovalVerdict};
use nca_core::provider::ProviderError;
use nca_core::tools::spawn_subagent::SpawnRequest;
use nca_runtime::ipc::IpcHandle;
use nca_runtime::supervisor::{PlanApplyOutcome, Supervisor, SupervisorConfig, SupervisorHandle};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use tokio::sync::mpsc;

/// Resolve a pending `ask_question` without going through `SessionRuntime` (e.g. TUI side task
/// while `run_turn` is blocked waiting on the same question).
pub fn dispatch_question_answer(
    qp: &Option<QuestionPendingMap>,
    question_id: &str,
    selection: QuestionSelection,
) -> bool {
    let Some(qp) = qp else {
        return false;
    };
    let Ok(mut m) = qp.lock() else {
        return false;
    };
    let Some(tx) = m.remove(question_id) else {
        return false;
    };
    tx.send(selection).is_ok()
}

/// Resolve a pending approval without going through the main command loop.
pub fn dispatch_tool_approval(
    approvals: &Option<ApprovalPendingMap>,
    call_id: &str,
    verdict: ApprovalVerdict,
) -> bool {
    let Some(approvals) = approvals else {
        return false;
    };
    let Ok(mut map) = approvals.lock() else {
        return false;
    };
    let Some(tx) = map.remove(call_id) else {
        return false;
    };
    tx.send(verdict).is_ok()
}

/// Thin CLI wrapper around the runtime `Supervisor`.
/// Keeps the same public API so existing CLI code (repl, main) works unchanged.
pub struct SessionRuntime {
    supervisor: Supervisor,
    handle: Option<SupervisorHandle>,
    question_pending: Option<QuestionPendingMap>,
    safe_mode: bool,
    interactive_approvals: bool,
    /// Approval handler the session was built with (`Some` for the stdio
    /// REPL's [`crate::approval_prompts::InteractiveIpcApprovalHandler`],
    /// `None` for the TUI and for non-interactive sessions, where the
    /// supervisor installs its own IPC/auto-deny handler).
    ///
    /// Kept so a supervisor swap (`/new`, session switch) can hand the SAME
    /// handler to the fresh `Supervisor::create`/`resume`. A swap that passed
    /// `None` instead used to silently downgrade a stdio session to the
    /// IPC-pending approval map — whose receiver has no consumer in stdio
    /// mode, so the first approval-gated tool call hung forever.
    approval_handler: Option<Arc<dyn ApprovalHandler>>,
}

impl SessionRuntime {
    pub fn take_event_rx(&mut self) -> Option<tokio::sync::mpsc::Receiver<AgentEvent>> {
        self.handle.as_mut()?.take_event_rx()
    }

    pub fn event_log_path(&self) -> std::path::PathBuf {
        self.supervisor.event_log_path()
    }

    /// Handle for enqueueing user prompts / steering into the running or
    /// next turn. Delegates to the supervisor's bounded agent inbox; use
    /// `try_send` and surface "inbox full" to the user.
    ///
    /// Wired into the TUI composer's busy-Submit path in the P1 TUI lane;
    /// retained here (not dead) as the public passthrough surface.
    #[allow(dead_code)]
    pub fn inbox_sender(&self) -> tokio::sync::mpsc::Sender<InboxItem> {
        self.supervisor.inbox_sender()
    }

    pub async fn run_turn(&mut self, prompt: &str) -> Result<String, ProviderError> {
        self.supervisor.run_turn(prompt).await
    }

    pub async fn run_turn_with_images(
        &mut self,
        prompt: &str,
        attachments: Vec<nca_common::message::ImageAttachment>,
    ) -> Result<String, ProviderError> {
        self.supervisor
            .run_turn_with_images(prompt, &attachments)
            .await
    }

    pub async fn finish(&mut self, reason: EndReason) {
        self.supervisor.finish(reason).await;
    }

    pub fn take_ipc_handle(&mut self) -> Option<IpcHandle> {
        self.handle.as_mut()?.take_ipc_handle()
    }

    pub fn take_ipc_approval_pending(&mut self) -> Option<ApprovalPendingMap> {
        self.handle.as_mut()?.take_approval_pending()
    }

    /// Pending `ask_question` resolvers (same map the runtime tool waits on).
    pub fn question_pending(&self) -> Option<QuestionPendingMap> {
        self.question_pending.clone()
    }

    /// Submit an answer for the current interactive question (TUI / REPL).
    pub fn submit_question_answer(&self, question_id: &str, selection: QuestionSelection) -> bool {
        dispatch_question_answer(&self.question_pending, question_id, selection)
    }

    /// Accept the model's suggested answer when exactly one question is pending.
    pub fn submit_suggested_answer(&self) -> bool {
        let Some(ref qp) = self.question_pending else {
            return false;
        };
        let Ok(mut m) = qp.lock() else {
            return false;
        };
        let keys: Vec<String> = m.keys().cloned().collect();
        if keys.len() != 1 {
            return false;
        }
        let id = keys[0].clone();
        let Some(tx) = m.remove(&id) else {
            return false;
        };
        tx.send(QuestionSelection::Suggested).is_ok()
    }

    pub fn session_id(&self) -> &str {
        self.supervisor.session_id()
    }

    pub fn model(&self) -> &str {
        &self.supervisor.model
    }

    pub fn workspace_root(&self) -> &std::path::Path {
        &self.supervisor.workspace_root
    }

    pub fn take_spawn_rx(&mut self) -> Option<mpsc::Receiver<SpawnRequest>> {
        self.handle.as_mut()?.take_spawn_rx()
    }

    /// Takes the turn-commit watch sender for the CLI event fanout
    /// (durability barrier wiring, P2 Phase B).
    pub fn take_turn_commit_tx(
        &mut self,
    ) -> Option<(tokio::sync::watch::Sender<u64>, Arc<AtomicBool>)> {
        self.handle.as_mut()?.take_turn_commit_tx()
    }

    /// Live conversation mirror for the sub-agent spawn consumer.
    pub fn spawn_history(&self) -> Arc<std::sync::Mutex<Vec<nca_common::message::Message>>> {
        self.supervisor.spawn_history()
    }

    /// Read-only subagent task registry (P1 `/jobs`, `task_status`/
    /// `task_result` replies). Shared with the spawn consumer.
    pub fn subagent_registry(&self) -> Arc<nca_runtime::subagent_registry::SubagentRegistry> {
        self.supervisor.subagent_registry()
    }

    /// Ghost-children report from the last resume (background tasks that
    /// provably died with the previous parent process). Delegates to
    /// [`nca_runtime::supervisor::Supervisor::take_restart_ghosts`].
    pub fn take_restart_ghosts(&mut self) -> Vec<nca_runtime::supervisor::RestartGhost> {
        self.supervisor.take_restart_ghosts()
    }

    /// Per-server MCP load report from session create (loaded tool counts
    /// or skip reasons). Consumed once by the CLI wiring.
    pub fn take_mcp_report(&mut self) -> Vec<nca_core::tools::mcp::McpServerLoadReport> {
        self.supervisor.take_mcp_report()
    }

    /// Shared plugin registry handle (G3: `subagentDispatch` hooks fire
    /// against the parent-rooted plugin instances).
    pub fn plugin_registry(&self) -> Option<Arc<nca_core::plugin::PluginRegistry>> {
        Some(self.supervisor.plugin_registry())
    }

    /// Escaped-provider handle for the subagent spawn consumer (T1-2 chain
    /// avoidance): providers this session failed over away from, dropped
    /// from freshly spawned children's fallback chains. `None` when
    /// fallback is disabled.
    pub fn fallback_escaped_handle(
        &self,
    ) -> Option<Arc<std::sync::Mutex<std::collections::HashSet<String>>>> {
        self.supervisor.fallback_escaped_handle()
    }

    pub fn set_model(&mut self, model: impl Into<String>) {
        let model = model.into();
        self.supervisor.model = model.clone();
        self.supervisor.agent_mut().model = model;
    }

    /// Post-hoc tool registration for CLI-owned tools (P4 `wait_for_user`).
    /// Delegates straight to the supervisor's live agent registry; the
    /// registration survives agent-profile switches because
    /// `apply_agent_profile` never touches the tool registry (gating via
    /// `restrict_to` runs only at `Supervisor::create`).
    pub fn register_tool(&mut self, tool: Box<dyn nca_core::tools::ToolExecutor>) {
        self.supervisor.agent_mut().tools.register(tool);
    }

    pub fn permission_mode(&self) -> PermissionMode {
        self.supervisor.agent().approval.mode()
    }

    pub fn set_permission_mode(&mut self, mode: PermissionMode) {
        self.supervisor.agent_mut().approval.set_mode(mode);
    }

    /// Switch the active agent profile at runtime.
    ///
    /// Pass `None` to restore the default (no-profile) agent.
    /// Pass `Some("explorer")` to activate the explorer specialist, etc.
    /// Returns the applied profile name, or `None` when the session is on
    /// the default (@orchestrator) persona (unresolvable names included).
    pub fn apply_agent_profile(
        &mut self,
        name: Option<&str>,
    ) -> Result<Option<String>, ProviderError> {
        let applied = self.supervisor.apply_agent_profile(name)?;
        Ok(applied)
    }

    /// List all registered agent profile names (from the supervisor's config,
    /// which includes OMO specialists auto-registered at startup).
    pub fn agent_profile_names(&self) -> Vec<String> {
        self.supervisor
            .config()
            .agent_profile_names()
            .into_iter()
            .map(String::from)
            .collect()
    }

    /// Get the description for a named agent profile (for display in pickers).
    pub fn agent_profile_description(&self, name: &str) -> Option<String> {
        self.supervisor
            .config()
            .agent_profile(name)
            .and_then(|p| p.description.clone())
    }

    /// Switch the active LLM provider and optionally override the model.
    /// Rebuilds the underlying provider connection.
    /// Returns the effective model name after the switch.
    pub fn switch_provider(
        &mut self,
        provider: ProviderKind,
        model_override: Option<&str>,
    ) -> Result<String, ProviderError> {
        let mut cfg = self.supervisor.config().clone();
        cfg.set_default_provider(provider);
        if let Some(model) = model_override {
            cfg.provider
                .set_model_for_default(cfg.model.resolve_alias(model));
        }
        cfg.sync_default_model_from_provider();
        let effective_model = cfg.model.default_model.clone();
        self.supervisor.apply_nca_config(cfg)?;
        Ok(effective_model)
    }

    /// Apply a `/model`-style change to the ACTIVE persona: with a specialist
    /// profile active the model is pinned into that profile's
    /// `[agents.<name>]` entry (surviving switches/resumes/spawns); otherwise
    /// the base-config default-provider flow. Returns the updated profile
    /// name, or `None` for the base flow.
    pub fn apply_model_for_active_agent(
        &mut self,
        model: &str,
    ) -> Result<Option<String>, ProviderError> {
        self.supervisor.set_model_for_active_agent(model)
    }

    /// `/provider`-style switch routed to the active persona (see
    /// [`SessionRuntime::apply_model_for_active_agent`]).
    pub fn apply_provider_for_active_agent(
        &mut self,
        provider: ProviderKind,
    ) -> Result<Option<String>, ProviderError> {
        self.supervisor.set_provider_for_active_agent(provider)
    }

    /// Apply a named model plan (`/plan <name>`): pins each covered agent's
    /// provider/model into `[agents.<name>]` and hot-swaps the active
    /// persona when covered. See [`Supervisor::apply_plan`].
    pub fn apply_plan(&mut self, name: &str) -> Result<PlanApplyOutcome, ProviderError> {
        self.supervisor.apply_plan(name)
    }

    /// Configured model plans (`[plans.<name>]`), plan name → agent entries.
    pub fn plans(&self) -> &BTreeMap<String, BTreeMap<String, PlanEntry>> {
        self.supervisor.plans()
    }

    /// Name of the currently applied model plan, if any.
    pub fn active_plan(&self) -> Option<&str> {
        self.supervisor.active_plan()
    }

    /// Provider kind the active persona routes to (profile override or the
    /// config default) — for pickers marking/fetching the live provider.
    pub fn active_provider(&self) -> ProviderKind {
        self.supervisor.active_provider()
    }

    pub fn request_cancel(&self) {
        self.supervisor.request_cancel();
    }

    /// Live spawn-time config handle shared with the subagent spawn
    /// consumer — see [`nca_runtime::supervisor::Supervisor::live_config`].
    /// Read at each spawn so children inherit the CURRENT provider/model
    /// routing, not the wiring-time snapshot.
    pub fn live_config(&self) -> Arc<std::sync::RwLock<nca_common::config::NcaConfig>> {
        self.supervisor.live_config()
    }

    /// Live workspace-root handle shared with the subagent spawn consumer —
    /// see [`nca_runtime::supervisor::Supervisor::workspace_root_handle`].
    /// Read at each spawn so children are rooted where the session lives
    /// NOW (after a `/new` reset or worktree switch), not where the
    /// consumer was wired.
    pub fn workspace_root_handle(&self) -> Arc<std::sync::RwLock<std::path::PathBuf>> {
        self.supervisor.workspace_root_handle()
    }

    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        self.supervisor.cancel_handle()
    }

    pub fn event_tx(&self) -> Option<tokio::sync::mpsc::Sender<AgentEvent>> {
        self.supervisor.event_tx()
    }

    pub async fn list_session_snapshots(
        &self,
    ) -> Result<Vec<nca_common::session::SessionSnapshot>, String> {
        let store = nca_runtime::session_store::SessionStore::new(
            self.workspace_root()
                .join(&self.supervisor.config().session.history_dir),
        );
        let ids = store.list().await.map_err(|err| err.to_string())?;
        let mut snapshots = Vec::with_capacity(ids.len());
        for id in ids {
            match store.load_snapshot(&id).await {
                Ok(snap) => snapshots.push(snap),
                Err(_) => {
                    // skip unreadable sessions
                }
            }
        }
        Ok(snapshots)
    }

    /// Lines describing tracked subagent tasks (P1 `/jobs`), formatted
    /// `<child_id>  <state>  <task ≤60 chars>  [branch]` in spawn order.
    /// An empty registry yields a single "no tasks" line.
    pub fn list_subagent_jobs(&self) -> Vec<String> {
        let entries = self.supervisor.subagent_registry().list();
        if entries.is_empty() {
            return vec!["No subagent tasks tracked.".into()];
        }
        entries.iter().map(format_job_line).collect()
    }

    /// Live config. The supervisor owns the single authoritative copy;
    /// mutations via `config_mut` and `apply_nca_config` stay visible here.
    pub fn config(&self) -> &NcaConfig {
        self.supervisor.config()
    }

    /// Mutate the live config in place (e.g. `/set-editor`).
    pub fn config_mut(&mut self) -> &mut NcaConfig {
        self.supervisor.config_mut()
    }

    /// Replace merged config and rebuild the provider (fails if API key missing, etc.).
    pub fn apply_nca_config(&mut self, config: NcaConfig) -> Result<(), ProviderError> {
        self.supervisor.apply_nca_config(config)
    }

    pub fn snapshot(&self) -> SessionSnapshot {
        self.supervisor.snapshot()
    }

    /// Current context window usage statistics.
    pub fn context_stats(&self) -> nca_runtime::context_manager::ContextStats {
        self.supervisor.context_stats()
    }

    pub fn compact_summary(&self) -> String {
        self.supervisor.compact_summary()
    }

    pub fn set_session_summary(&mut self, summary: Option<String>) {
        self.supervisor.set_session_summary(summary);
    }

    pub async fn append_memory_note(
        &self,
        kind: &str,
        content: Option<String>,
    ) -> Result<(), String> {
        self.supervisor.append_memory_note(kind, content).await
    }

    pub fn memory_store_path(&self) -> std::path::PathBuf {
        self.supervisor.memory_store_path()
    }

    /// Start a fresh session (D3 supervisor swap).
    ///
    /// Builds a BRAND-NEW supervisor rooted at the process's main workspace
    /// root, finishes the current session (which persists ITS json and emits
    /// `SessionEnded` on the old event channel — the single-writer
    /// discipline keeps saves at create/resume/finish only), and swaps it in.
    /// The swap is what makes the CLI rewire protocol (D3) possible: the
    /// fresh supervisor owns fresh event/spawn/commit channels, so the CLI
    /// can take them and spawn a new bridge writing the NEW session's event
    /// log, while dropping the old senders lets the old fanout drain +
    /// commit-on-close and end the OLD log cleanly.
    ///
    /// Ordering (M5): the new supervisor is created BEFORE the old one is
    /// finished. A failed create (unbuildable provider, MCP handshake
    /// failure, store I/O) therefore leaves the current session exactly as it
    /// was — still `Running`, still usable — instead of stranding the user on
    /// a session that is already `Completed` on disk.
    ///
    /// In-place `reset_for_new_session` (D1/D2 logic) cannot offer the log
    /// roll: its event channel survives the reset with senders held by the
    /// agent loop, tools, and fallback wrapper, so the old bridge never
    /// observes all-senders-dropped and the log path never rolls.
    ///
    /// State carried across the swap (mirroring what an in-place reset
    /// preserved): the active agent persona (`agent_name`), the session
    /// sandbox override, the live permission mode, the approval handler
    /// (B1: dropping it downgraded stdio approvals to an unconsumed IPC map),
    /// and the orchestration context (M1: a headless run must not lose its
    /// orchestrator identity). The config carried over is the BASE config
    /// (M2: profile overrides are re-applied by `create` from
    /// `[agents.<name>]`, so consecutive `/new` swaps cannot stack them).
    /// `derived_from` (D2) is stamped from the previous id. Not carried: the
    /// subagent registry (fresh; the family view recovers lineage from disk
    /// via the `derived_from` chain) and the applied model plan (re-apply with
    /// `/plan`).
    ///
    /// Two consequences worth knowing about (documented, not bugs):
    ///
    /// - Background children spawned BEFORE the `/new` keep running but become
    ///   unreachable from the new session's CONTROL plane: `task_status`/
    ///   `task_result`/`task_cancel`/kill-all read the NEW registry, so they
    ///   cannot see or stop the old children. The data plane keeps flowing —
    ///   the old children's events forward through the old (still-draining)
    ///   channel into the shared Elm feedback stream, and wake notifications
    ///   ride the shared `WakeScheduler` into the new session's Submit queue.
    ///   They stay visible/cancellable via the `/jobs` view and `nca cancel
    ///   <child-id>`.
    /// - `ApprovalPolicy` is rebuilt from the base config, so session-scoped
    ///   "always allow THIS tool" grants (`session_allow`) do not survive a
    ///   `/new` — the same tool may prompt again. (Failing closed is the safe
    ///   direction; persisted allow-patterns still arrive on the next process
    ///   start.)
    pub async fn new_session(&mut self) -> Result<(), String> {
        let prev_id = self.supervisor.session_id().to_string();
        let main_root = self.supervisor.main_workspace_root().to_path_buf();
        let agent_name = self.supervisor.active_agent_name().map(String::from);
        let sandbox_override = self.supervisor.sandbox_override();
        let permission_mode = self.supervisor.agent().approval.mode();
        let config = {
            let mut config = self.supervisor.base_config().clone();
            // `base_config` is the overlay-free snapshot, so it does not see
            // session-scoped `config_mut()` edits. The only such edit today is
            // `/set-editor` (`ui.editor`), which is user intent — never a
            // profile override — so carry it over explicitly instead of
            // silently resetting the editor on every `/new`.
            config.ui = self.supervisor.config().ui.clone();
            config
        };
        let orchestration_context = self.supervisor.orchestration_context();
        let approval_handler = self.approval_handler.clone();

        // Same pre-load announcement as the initial create (MCP handshakes
        // can take tens of seconds on a cold cache — `/new` must not look
        // like a freeze).
        announce_mcp_load(&config);

        let mut supervisor = Supervisor::create(SupervisorConfig {
            config,
            workspace_root: main_root,
            safe_mode: self.safe_mode,
            interactive_approvals: self.interactive_approvals,
            session_id: None,
            approval_handler,
            orchestration_context,
            agent_name,
            provider: None,
        })
        .await
        .map_err(|e| e.to_string())?;

        // The fresh supervisor is up: only now close the old session.
        self.supervisor.finish(EndReason::Completed).await;

        // D2: record the derivation so the family view (and other lineage
        // consumers) can walk back to the ancestor family. Persisted
        // immediately — the family view reads `derived_from` from disk, and
        // the json `create` just wrote does not carry it yet. A persist
        // failure is warned, NOT propagated: the fresh supervisor is already
        // valid and the old one is already finished, so failing here would
        // leave the caller without a session to talk to.
        supervisor.set_derived_from(prev_id);
        if let Err(e) = supervisor.save().await {
            tracing::warn!("new session: persisting derived_from failed: {e}");
        }
        if sandbox_override.is_some() {
            supervisor.set_sandbox_override(sandbox_override);
        }
        supervisor.agent_mut().approval.set_mode(permission_mode);

        let mut handle = supervisor.take_handle();
        let question_pending = handle.take_question_pending();
        self.supervisor = supervisor;
        self.handle = Some(handle);
        self.question_pending = question_pending;
        Ok(())
    }

    /// Switch to a different existing session in-place (no process restart).
    /// Finishes the current session (which persists the json), then resumes
    /// the target session, rebuilding
    /// handle and question-pending channels so the caller can rewire its event loop.
    ///
    /// The approval handler is passed through (B1) so a stdio REPL keeps its
    /// interactive prompt instead of silently falling back to the IPC-pending
    /// map, which has no consumer on that path.
    pub async fn switch_to(&mut self, session_id: &str) -> Result<(), String> {
        self.supervisor.finish(EndReason::Completed).await;

        let mut supervisor = Supervisor::resume(
            self.supervisor.config().clone(),
            &self.supervisor.workspace_root,
            self.safe_mode,
            self.interactive_approvals,
            session_id,
            self.approval_handler.clone(),
            None,
        )
        .await
        .map_err(|e| e.to_string())?;

        let mut handle = supervisor.take_handle();
        let question_pending = handle.take_question_pending();
        self.supervisor = supervisor;
        self.handle = Some(handle);
        self.question_pending = question_pending;
        Ok(())
    }

    // ── Mount management ─────────────────────────────────────────────

    /// Mount an additional directory so tools can access files outside the workspace root.
    pub async fn mount_path(&mut self, path: &std::path::Path) -> Result<(), String> {
        self.supervisor.mount_path(path).await
    }

    /// Unmount a previously mounted directory.
    pub async fn unmount_path(&mut self, path: &std::path::Path) -> Result<(), String> {
        self.supervisor.unmount_path(path).await
    }

    /// List currently mounted extra paths.
    pub fn mounted_paths(&self) -> Vec<std::path::PathBuf> {
        self.supervisor.mounted_paths()
    }

    /// Session-level sandbox mode override (`/sandbox`): `Some(mode)` forces
    /// the given Landlock enforcement mode for this session, `None` restores
    /// the configured `[permissions.sandbox]` mode. Applies immediately and
    /// never touches the persisted config. See
    /// [`nca_runtime::supervisor::Supervisor::set_sandbox_override`].
    pub fn set_sandbox_mode(&mut self, mode: Option<SandboxMode>) {
        self.supervisor.set_sandbox_override(mode);
    }

    /// Live sandbox state for `/sandbox status` and `/config`.
    pub fn sandbox_status(&self) -> nca_runtime::supervisor::SandboxStatus {
        self.supervisor.sandbox_status()
    }

    /// Return the live filesystem adapter (for propagating runtime mounts to subagents).
    pub fn fs(&self) -> Arc<dyn nca_core::workspace_fs::WorkspaceFs> {
        self.supervisor.fs()
    }

    /// Slash commands contributed by plugins (for CLI slash panel and REPL hinter).
    pub fn plugin_commands(&self) -> Vec<(String, Vec<String>)> {
        self.supervisor.plugin_commands()
    }

    /// Dispatch a slash command to plugins (command interception).
    pub fn check_command_before(
        &self,
        command: &str,
        arguments: &str,
    ) -> Option<(String, nca_core::plugin::CommandIntercept)> {
        self.supervisor.check_command_before(command, arguments)
    }

    /// G6: echo an intercepted plugin command's output into the conversation
    /// (opt-in per interception; see `CommandIntercept.echo`).
    pub async fn record_plugin_command_echo(&mut self, plugin: &str, text: &str) {
        self.supervisor
            .record_plugin_command_echo(plugin, text)
            .await;
    }

    /// G7 `/plugin refresh`.
    pub async fn refresh_plugins(&mut self) -> Result<String, String> {
        self.supervisor.refresh_plugins().await
    }

    /// G7 `/plugin status`.
    pub fn plugin_status(&self) -> Vec<String> {
        self.supervisor.plugin_status()
    }
}

/// One `/jobs` line: `id [@alias]  state [gN]  task  [branch]`. The alias
/// rides after the id so users can see the name `task_*` tools accept;
/// the generation appears only when > 0 (a revived task).
pub(crate) fn format_job_line(
    entry: &nca_runtime::subagent_registry::SubagentRegistryEntry,
) -> String {
    let state = format!("{:?}", entry.state).to_ascii_lowercase();
    let task: String = entry.task.chars().take(60).collect();
    let id = match entry.alias.as_deref() {
        Some(alias) => format!("{} @{}", entry.session_id, alias),
        None => entry.session_id.clone(),
    };
    let state = if entry.generation > 0 {
        format!("{state} g{}", entry.generation)
    } else {
        state
    };
    match entry.branch.as_deref() {
        Some(branch) => format!("{id}  {state}  {task}  [{branch}]"),
        None => format!("{id}  {state}  {task}"),
    }
}

/// Pre-load MCP progress line (PRD R2): `Some(line)` when any ENABLED
/// server is configured, `None` otherwise.
///
/// Pure helper so the guard is unit-testable; [`announce_mcp_load`] prints it.
fn mcp_load_announcement(config: &NcaConfig) -> Option<String> {
    let enabled: Vec<&str> = config
        .mcp
        .servers
        .iter()
        .filter(|server| server.enabled)
        .map(|server| server.name.as_str())
        .collect();
    (!enabled.is_empty()).then(|| {
        format!(
            "[mcp] loading {} server(s): {}\u{2026}",
            enabled.len(),
            enabled.join(", ")
        )
    })
}

/// Announce MCP loading BEFORE the supervisor's create/resume handshake
/// runs. The TUI has not entered the alt screen yet, so this stderr line is
/// exactly what the user sees during a slow cold-cache `bunx -y pkg@latest`
/// fetch (non-TTY modes see it on stderr too). Main-session paths only —
/// never the subagent spawn path (children run headless). No-op when no
/// enabled servers exist.
fn announce_mcp_load(config: &NcaConfig) {
    if let Some(line) = mcp_load_announcement(config) {
        // stderr in ALL modes: under `--stream ndjson` stdout must stay pure,
        // and in TUI mode tracing/stderr is redirected away from the
        // alt-screen (tracing only — but this fires mid-session rarely
        // enough, and a stray stderr line beats a multi-second freeze).
        eprintln!("{line}");
    }
}

/// Render the per-server MCP load report as user-facing lines (one per
/// server): loaded tool counts, or the load/skip reason.
///
/// Shared by every session-create surface — the stdio REPL's stderr lines,
/// the TUI's `push_system` transcript lines, and the post-`/new` swap (where
/// the fresh supervisor's report would otherwise be dropped on the floor).
pub fn mcp_report_lines(reports: &[nca_core::tools::mcp::McpServerLoadReport]) -> Vec<String> {
    reports
        .iter()
        .map(|report| match (&report.tool_count, &report.error) {
            (Some(count), _) => format!(
                "[mcp] loaded {} ({} tool{})",
                report.name,
                count,
                if *count == 1 { "" } else { "s" }
            ),
            (None, Some(error)) => {
                format!("[mcp] FAILED {}: {} (skipped)", report.name, error)
            }
            (None, None) => format!("[mcp] FAILED {}: unknown reason (skipped)", report.name),
        })
        .collect()
}

pub async fn build_session_runtime(
    config: NcaConfig,
    workspace_root: &Path,
    safe_mode: bool,
    interactive_approvals: bool,
    session_id: Option<String>,
    ipc_approval_handler: Option<Arc<dyn ApprovalHandler>>,
    orchestration_context: Option<OrchestrationContext>,
) -> Result<SessionRuntime, ProviderError> {
    let approval_handler = ipc_approval_handler;

    // Before `create` is awaited (PRD R2): MCP servers can take tens of
    // seconds on a cold cache — show progress instead of a silent hang.
    announce_mcp_load(&config);

    let mut supervisor = Supervisor::create(SupervisorConfig {
        config,
        workspace_root: workspace_root.to_path_buf(),
        safe_mode,
        interactive_approvals,
        session_id,
        // Cloned: the runtime keeps the handler so a later supervisor swap
        // (`/new`, session switch) can hand the SAME one to `create`/`resume`.
        approval_handler: approval_handler.clone(),
        orchestration_context,
        agent_name: None,
        provider: None,
    })
    .await?;

    let mut handle = supervisor.take_handle();
    let question_pending = handle.take_question_pending();
    Ok(SessionRuntime {
        supervisor,
        handle: Some(handle),
        question_pending,
        safe_mode,
        interactive_approvals,
        approval_handler,
    })
}

pub async fn build_resumed_session_runtime(
    config: NcaConfig,
    workspace_root: &Path,
    safe_mode: bool,
    interactive_approvals: bool,
    session_id: &str,
    approval_handler: Option<Arc<dyn ApprovalHandler>>,
) -> Result<SessionRuntime, ProviderError> {
    // Same pre-load announcement as the create path: `Supervisor::resume`
    // re-runs the create pipeline (MCP handshake included), so a resumed
    // session gets the same during-load visibility.
    announce_mcp_load(&config);
    let mut supervisor = Supervisor::resume(
        config,
        workspace_root,
        safe_mode,
        interactive_approvals,
        session_id,
        approval_handler.clone(),
        None,
    )
    .await?;
    let mut handle = supervisor.take_handle();
    let question_pending = handle.take_question_pending();
    Ok(SessionRuntime {
        supervisor,
        handle: Some(handle),
        question_pending,
        safe_mode,
        interactive_approvals,
        approval_handler,
    })
}

#[cfg(test)]
mod mcp_announcement_tests {
    use super::*;

    /// M1 guard: the pre-load line names every ENABLED server and stays
    /// silent when none exist (all-disabled configs must not print noise).
    #[test]
    fn mcp_load_announcement_lists_enabled_servers_only() {
        let mut config = NcaConfig::default();
        assert_eq!(
            mcp_load_announcement(&config),
            None,
            "no servers configured"
        );

        config.mcp.servers = vec![nca_common::config::McpServerConfig {
            name: "rea".into(),
            enabled: false,
            command: String::new(),
            args: vec![],
            env: Default::default(),
            cwd: None,
        }];
        assert_eq!(
            mcp_load_announcement(&config),
            None,
            "disabled servers must not trigger the announcement"
        );

        config.mcp.servers = vec![
            nca_common::config::McpServerConfig {
                name: "off".into(),
                enabled: false,
                command: String::new(),
                args: vec![],
                env: Default::default(),
                cwd: None,
            },
            nca_common::config::McpServerConfig {
                name: "rea".into(),
                enabled: true,
                command: String::new(),
                args: vec![],
                env: Default::default(),
                cwd: None,
            },
        ];
        let line = mcp_load_announcement(&config).expect("enabled server announces");
        assert!(
            line.contains("1 server(s): rea"),
            "only enabled names appear, got: {line}"
        );
        assert!(
            !line.contains("off"),
            "disabled name must be absent: {line}"
        );
    }
}

#[cfg(test)]
mod job_line_tests {
    use super::format_job_line;
    use nca_common::session::ChildSessionState;
    use nca_runtime::subagent_registry::SubagentRegistryEntry;

    fn entry(
        state: ChildSessionState,
        alias: Option<&str>,
        generation: u64,
    ) -> SubagentRegistryEntry {
        SubagentRegistryEntry {
            session_id: "sess-1".into(),
            parent_session_id: "parent".into(),
            task: "do the thing".into(),
            state,
            generation,
            alias: alias.map(String::from),
            workspace: "/ws".into(),
            branch: Some("nca/sess-1".into()),
            worktree_path: None,
            result_summary: None,
            cancel_flag: None,
            inbox_tx: None,
            cancel_reason: None,
            specialist: None,
            last_activity: tokio::time::Instant::now(),
            interaction_pending: false,
            recent_activity: std::collections::VecDeque::new(),
            accumulated_input_tokens: 0,
            accumulated_output_tokens: 0,
            accumulated_cache_read_tokens: 0,
        }
    }

    #[test]
    fn plain_running_task_shows_id_state_task_branch() {
        let line = format_job_line(&entry(ChildSessionState::Running, None, 0));
        assert_eq!(line, "sess-1  running  do the thing  [nca/sess-1]");
    }

    #[test]
    fn alias_rides_after_the_id() {
        let line = format_job_line(&entry(ChildSessionState::Running, Some("fixer-2"), 0));
        assert!(line.starts_with("sess-1 @fixer-2  running"), "got: {line}");
    }

    #[test]
    fn generation_shown_only_when_revived() {
        let revived = format_job_line(&entry(ChildSessionState::Completed, Some("fixer-2"), 1));
        assert!(revived.contains("completed g1"), "got: {revived}");
        let fresh = format_job_line(&entry(ChildSessionState::Completed, None, 0));
        assert_eq!(fresh, "sess-1  completed  do the thing  [nca/sess-1]");
    }

    #[test]
    fn long_task_text_is_capped_at_60_chars() {
        let mut long = entry(ChildSessionState::Running, None, 0);
        long.task = "x".repeat(200);
        let line = format_job_line(&long);
        assert!(
            line.contains(&"x".repeat(60)),
            "task must be capped: {line}"
        );
        assert!(!line.contains(&"x".repeat(61)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Crate-wide env guard: a module-local mutex here would NOT exclude the
    // env mutations in `repl.rs`'s tests (parallel threads, one process).
    use crate::test_env::TestEnvGuard;
    use crate::tui::elm::subagent_family::build_family_index;
    use nca_common::session::{SessionMeta, SessionState, SessionStatus};
    use nca_runtime::session_store::SessionStore;
    use std::collections::HashMap;

    fn offline_config() -> NcaConfig {
        let mut config = NcaConfig::default();
        config.permissions.mode = PermissionMode::BypassPermissions;
        config.memory.context.auto_detect_context_window = false;
        config.memory.context.query_provider_models_api = false;
        config.memory.context.enable_auto_summarize = false;
        config
    }

    /// Regression: `/mount` must survive a later `/model`-style whole-config
    /// save. SessionRuntime used to keep its own stale `NcaConfig` copy that
    /// `mount_path` never updated, so the next clone→apply→save cycle erased
    /// the persisted `extra_paths` from `.nca/config.local.toml`.
    #[tokio::test(flavor = "multi_thread")]
    async fn mount_survives_model_switch_config_save() {
        let home = tempfile::tempdir().expect("home tempdir");
        let xdg = tempfile::tempdir().expect("xdg tempdir");
        let _env = TestEnvGuard::set(&[
            ("HOME", home.path().to_str().unwrap()),
            ("XDG_CONFIG_HOME", xdg.path().to_str().unwrap()),
            // Hermeticity: default config uses deepseek, which validates its
            // API key eagerly at provider build. Inject a dummy key so the
            // runtime constructs regardless of the host environment.
            ("DEEPSEEK_API_KEY", "dummy"),
        ]);

        let ws = tempfile::tempdir().expect("workspace tempdir");
        let ext = tempfile::tempdir().expect("external dir tempdir");
        let mut rt = build_session_runtime(
            offline_config(),
            ws.path(),
            true,
            false,
            Some("mnt-runner-regression".into()),
            None,
            None,
        )
        .await
        .expect("session runtime builds (dummy key injected via env guard)");

        rt.mount_path(ext.path()).await.expect("mount");

        let expected = vec![ext.path().canonicalize().expect("canonicalize")];
        assert_eq!(
            rt.config().extra_paths,
            expected,
            "runtime-visible config must reflect the mount immediately"
        );

        // `/model` flow: clone the runtime snapshot, tweak, re-apply, save.
        let mut cfg = rt.config().clone();
        cfg.apply_model_override("deepseek-chat");
        rt.apply_nca_config(cfg)
            .expect("apply (dummy key injected via env guard)");
        rt.config()
            .save_workspace_file(ws.path())
            .expect("workspace save");

        let disk = NcaConfig::load_for_workspace(ws.path()).expect("reload");
        assert_eq!(
            disk.extra_paths, expected,
            "REGRESSION: model switch erased the persisted mount from the stale runner snapshot"
        );
    }

    // ── `/new` on the PRODUCTION path (PRD acceptance 1-3) ────────────
    //
    // `/new` no longer calls `Supervisor::reset_for_new_session`; the REPL/TUI
    // go through [`SessionRuntime::new_session`], which finishes the old
    // supervisor and swaps in a BRAND-NEW one rooted at
    // `main_workspace_root`. The reset unit tests in `nca-runtime` therefore
    // exercise code the shipped binary does not run for `/new`. These tests
    // drive the real swap instead.

    /// A worktree-grounded session that has just been RE-OPENED through the
    /// resume path — the state the `nca --resume <id>` run in the incident
    /// started from.
    ///
    /// `meta.workspace` points at the worktree, so `Supervisor::resume`
    /// ADOPTS the worktree as the live root (D1's precondition). The returned
    /// runtime is what `/new` then acts on.
    async fn worktree_grounded_resumed_runtime(
        ws: &Path,
        wt_path: &Path,
        session_id: &str,
    ) -> SessionRuntime {
        let mut rt = build_session_runtime(
            offline_config(),
            ws,
            true,
            false,
            Some(session_id.to_string()),
            None,
            None,
        )
        .await
        .expect("session runtime builds (dummy key injected via env guard)");
        // Ground the session in the worktree exactly as a worktree-scoped
        // session does, then persist: `meta.workspace` = the worktree root.
        rt.supervisor.switch_to_worktree(
            wt_path.to_path_buf(),
            "nca/new-e2e".into(),
            "main".into(),
        );
        rt.supervisor
            .save()
            .await
            .expect("persist worktree-grounded meta");
        // Process exit between the two runs (session json stays on disk).
        rt.finish(EndReason::UserExit).await;
        drop(rt);

        build_resumed_session_runtime(offline_config(), ws, true, false, session_id, None)
            .await
            .expect("resume the worktree-grounded session")
    }

    /// D1 + D3 invariants of a post-`/new` session: main repository root,
    /// cleared worktree metadata, and store/event-log/fs all on that root.
    fn assert_reset_to_main_root(rt: &SessionRuntime, ws_root: &Path, wt_path: &Path) {
        let main_sessions = ws_root.join(".nca/sessions");
        assert_eq!(
            rt.workspace_root()
                .canonicalize()
                .expect("canonical workspace root"),
            ws_root,
            "post-/new workspace_root must be the main repository root"
        );

        let snapshot = rt.snapshot();
        assert!(
            snapshot.worktree_path.is_none(),
            "worktree_path cleared, got {:?}",
            snapshot.worktree_path
        );
        assert!(
            snapshot.branch.is_none(),
            "branch cleared, got {:?}",
            snapshot.branch
        );
        assert!(snapshot.base_branch.is_none(), "base_branch cleared");
        assert_eq!(
            snapshot
                .workspace
                .canonicalize()
                .expect("canonical snapshot workspace"),
            ws_root,
            "the persisted session workspace is the main root"
        );

        // D3: store and event log derive from the SAME root (no split-brain).
        let store_dir = rt
            .supervisor
            .session_store()
            .sessions_dir()
            .canonicalize()
            .expect("canonical store dir");
        assert_eq!(
            store_dir,
            main_sessions
                .canonicalize()
                .expect("canonical main sessions dir"),
            "the session store lives in the MAIN .nca/sessions"
        );
        assert_eq!(
            rt.event_log_path()
                .parent()
                .expect("event log has a parent dir")
                .canonicalize()
                .expect("canonical event log dir"),
            store_dir,
            "event log and session store must share one directory"
        );

        // D1: file tools and shell commands agree on the same root.
        assert_eq!(
            rt.fs().root().canonicalize().expect("canonical fs root"),
            ws_root,
            "the fs adapter is re-rooted to the main repository"
        );

        // No split-brain json under the stale worktree.
        assert!(
            !wt_path
                .join(".nca/sessions")
                .join(format!("{}.json", rt.session_id()))
                .exists(),
            "the new session's json must not be written under the worktree"
        );
    }

    /// Read back the new session's persisted json from the main sessions dir.
    fn load_persisted_session(rt: &SessionRuntime, ws_root: &Path) -> SessionState {
        let path = ws_root
            .join(".nca/sessions")
            .join(format!("{}.json", rt.session_id()));
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        serde_json::from_str(&raw).expect("parse the persisted session json")
    }

    /// Persist a child session json through the real [`SessionStore`] (same
    /// serde shape a spawned subagent writes) so the family scan reads
    /// genuine on-disk session files, not hand-written json text.
    async fn write_child_session(
        store: &SessionStore,
        workspace: &Path,
        id: &str,
        parent: Option<&str>,
        status: SessionStatus,
    ) {
        let now = chrono::Utc::now();
        let state = SessionState {
            meta: SessionMeta {
                id: id.to_string(),
                created_at: now,
                updated_at: now,
                workspace: workspace.to_path_buf(),
                model: "deepseek-chat".into(),
                status,
                pid: None,
                socket_path: None,
                worktree_path: None,
                branch: None,
                base_branch: None,
                parent_session_id: parent.map(str::to_string),
                child_session_ids: Vec::new(),
                inherited_summary: None,
                spawn_reason: Some(format!("task for {id}")),
                session_summary: None,
                session_title: None,
                orchestration: None,
                agent_name: None,
                derived_from: None,
            },
            messages: Vec::new(),
            total_input_tokens: 0,
            total_output_tokens: 0,
            estimated_cost_usd: 0.0,
        };
        store.save(&state).await.expect("save child session json");
    }

    /// PRD acceptance 1 + 3 on the production `/new` path
    /// ([`SessionRuntime::new_session`], the supervisor swap the REPL/TUI
    /// actually call — not the in-place `reset_for_new_session`, which the
    /// shipped binary no longer uses for `/new`).
    #[tokio::test(flavor = "multi_thread")]
    async fn new_session_returns_worktree_grounded_session_to_main_root() {
        let home = tempfile::tempdir().expect("home tempdir");
        let xdg = tempfile::tempdir().expect("xdg tempdir");
        let _env = TestEnvGuard::set(&[
            ("HOME", home.path().to_str().expect("home path")),
            ("XDG_CONFIG_HOME", xdg.path().to_str().expect("xdg path")),
            ("DEEPSEEK_API_KEY", "dummy"),
        ]);
        let ws = tempfile::tempdir().expect("workspace tempdir");
        let wt = tempfile::tempdir().expect("worktree tempdir");
        let ws_root = ws.path().canonicalize().expect("canonical ws");
        let wt_path = wt.path().canonicalize().expect("canonical wt");

        let mut rt =
            worktree_grounded_resumed_runtime(ws.path(), &wt_path, "nse2e-new-session-wt").await;
        let old_id = rt.session_id().to_string();
        assert_eq!(
            rt.workspace_root()
                .canonicalize()
                .expect("canonical resumed root"),
            wt_path,
            "precondition: resume adopted the stored worktree root"
        );
        assert_eq!(
            rt.snapshot().worktree_path.as_deref(),
            Some(wt_path.as_path()),
            "precondition: the worktree metadata was restored"
        );

        // The production `/new`: supervisor swap.
        rt.new_session().await.expect("/new supervisor swap");
        assert_ne!(rt.session_id(), old_id, "the swap rolled the session id");

        assert_reset_to_main_root(&rt, &ws_root, &wt_path);

        // D2: the derivation is persisted, so the family view can walk back.
        let persisted = load_persisted_session(&rt, &ws_root);
        assert_eq!(
            persisted.meta.derived_from.as_deref(),
            Some(old_id.as_str()),
            "/new records the derivation ancestor"
        );
        assert!(
            persisted.meta.worktree_path.is_none()
                && persisted.meta.branch.is_none()
                && persisted.meta.base_branch.is_none(),
            "persisted worktree metadata is cleared: {:?} / {:?} / {:?}",
            persisted.meta.worktree_path,
            persisted.meta.branch,
            persisted.meta.base_branch
        );

        rt.finish(EndReason::UserExit).await;
    }

    /// PRD acceptance 1, deleted-worktree variant: the worktree directory is
    /// gone before `/new` runs. The swap must still succeed and land the new
    /// session on the main root (no error, no panic, no stale-root session).
    #[tokio::test(flavor = "multi_thread")]
    async fn new_session_tolerates_deleted_worktree_dir() {
        let home = tempfile::tempdir().expect("home tempdir");
        let xdg = tempfile::tempdir().expect("xdg tempdir");
        let _env = TestEnvGuard::set(&[
            ("HOME", home.path().to_str().expect("home path")),
            ("XDG_CONFIG_HOME", xdg.path().to_str().expect("xdg path")),
            ("DEEPSEEK_API_KEY", "dummy"),
        ]);
        let ws = tempfile::tempdir().expect("workspace tempdir");
        let wt = tempfile::tempdir().expect("worktree tempdir");
        let ws_root = ws.path().canonicalize().expect("canonical ws");
        let wt_path = wt.path().canonicalize().expect("canonical wt");

        let mut rt =
            worktree_grounded_resumed_runtime(ws.path(), &wt_path, "nse2e-new-session-wt-gone")
                .await;
        assert_eq!(
            rt.workspace_root()
                .canonicalize()
                .expect("canonical resumed root"),
            wt_path,
            "precondition: worktree-grounded session"
        );

        // Delete the worktree out from under the still-open session.
        drop(wt);
        assert!(!wt_path.exists(), "precondition: the worktree dir is gone");

        rt.new_session()
            .await
            .expect("/new must tolerate a deleted worktree dir");
        assert_reset_to_main_root(&rt, &ws_root, &wt_path);

        rt.finish(EndReason::UserExit).await;
    }

    /// PRD acceptance 2, end-to-end: after the real `/new` swap, the family
    /// scan the `/jobs` view runs ([`build_family_index`]) is seeded from the
    /// persisted `derived_from` chain, so the previous root's children (and
    /// their descendants) are still listed, while an unrelated root's subtree
    /// stays filtered out.
    #[tokio::test(flavor = "multi_thread")]
    async fn new_session_keeps_prior_family_visible_to_jobs() {
        let home = tempfile::tempdir().expect("home tempdir");
        let xdg = tempfile::tempdir().expect("xdg tempdir");
        let _env = TestEnvGuard::set(&[
            ("HOME", home.path().to_str().expect("home path")),
            ("XDG_CONFIG_HOME", xdg.path().to_str().expect("xdg path")),
            ("DEEPSEEK_API_KEY", "dummy"),
        ]);
        let ws = tempfile::tempdir().expect("workspace tempdir");
        let wt = tempfile::tempdir().expect("worktree tempdir");
        let ws_root = ws.path().canonicalize().expect("canonical ws");
        let wt_path = wt.path().canonicalize().expect("canonical wt");
        let main_sessions = ws_root.join(".nca/sessions");

        let mut rt =
            worktree_grounded_resumed_runtime(ws.path(), &wt_path, "nse2e-lineage-root").await;
        let old_id = rt.session_id().to_string();

        // Children of the OLD root, a grandchild under one of them, and an
        // unrelated root U with its own child X — all as real persisted
        // session jsons in the main sessions dir.
        let store = SessionStore::new(main_sessions.clone());
        write_child_session(
            &store,
            &ws_root,
            "nse2e-child-1",
            Some(&old_id),
            SessionStatus::Completed,
        )
        .await;
        write_child_session(
            &store,
            &ws_root,
            "nse2e-grandchild-1",
            Some("nse2e-child-1"),
            SessionStatus::Completed,
        )
        .await;
        write_child_session(
            &store,
            &ws_root,
            "nse2e-unrelated-root",
            None,
            SessionStatus::Completed,
        )
        .await;
        write_child_session(
            &store,
            &ws_root,
            "nse2e-unrelated-child",
            Some("nse2e-unrelated-root"),
            SessionStatus::Completed,
        )
        .await;

        rt.new_session().await.expect("/new supervisor swap");

        // Exactly what the `/jobs` view does: scan the MAIN sessions dir from
        // the new root's id.
        let mut ids: Vec<String> =
            build_family_index(&main_sessions, rt.session_id(), &HashMap::new())
                .into_iter()
                .map(|entry| entry.session_id)
                .collect();
        ids.sort();
        assert_eq!(
            ids,
            ["nse2e-child-1", "nse2e-grandchild-1"],
            "the /new session inherits the old root's family (children + descendants) \
             and never the unrelated root's subtree"
        );

        rt.finish(EndReason::UserExit).await;
    }

    // ── B1 / M1 / M5: what the `/new` supervisor swap must carry over ──

    /// B1: a stdio session is built with an interactive approval handler.
    /// The OLD swap passed `approval_handler: None`, which made `create`
    /// install the IPC-pending handler instead — and in stdio mode nobody
    /// consumes that map, so the first approval-gated tool call waited on a
    /// oneshot forever. The handler must survive the swap.
    ///
    /// The observable proxy for "which branch did `create` take" is
    /// `take_ipc_approval_pending()`: the handler branch leaves it empty, the
    /// IPC branch populates it.
    #[tokio::test(flavor = "multi_thread")]
    async fn new_session_carries_the_interactive_approval_handler() {
        let home = tempfile::tempdir().expect("home tempdir");
        let xdg = tempfile::tempdir().expect("xdg tempdir");
        let _env = TestEnvGuard::set(&[
            ("HOME", home.path().to_str().expect("home path")),
            ("XDG_CONFIG_HOME", xdg.path().to_str().expect("xdg path")),
            ("DEEPSEEK_API_KEY", "dummy"),
        ]);
        let ws = tempfile::tempdir().expect("workspace tempdir");
        let handler: Option<Arc<dyn ApprovalHandler>> = Some(
            crate::approval_prompts::InteractiveIpcApprovalHandler::new()
                as Arc<dyn ApprovalHandler>,
        );

        let mut rt = build_session_runtime(
            offline_config(),
            ws.path(),
            true,
            true, // interactive approvals (stdio REPL shape)
            Some("b1-handler-swap".into()),
            handler,
            None,
        )
        .await
        .expect("session runtime builds (dummy key injected via env guard)");
        assert!(
            rt.approval_handler.is_some(),
            "precondition: the runtime kept the handler it was built with"
        );
        assert!(
            rt.take_ipc_approval_pending().is_none(),
            "precondition: the handler branch installs no IPC pending map"
        );

        rt.new_session().await.expect("/new supervisor swap");

        assert!(
            rt.approval_handler.is_some(),
            "the swap must keep the handler on the runtime"
        );
        assert!(
            rt.take_ipc_approval_pending().is_none(),
            "REGRESSION (B1): the swap downgraded to the unconsumed IPC approval \
             map, so approval-gated tool calls hang forever in stdio mode"
        );

        rt.finish(EndReason::UserExit).await;
    }

    /// B1 contrast: a TUI session is built with NO handler, so `create`
    /// installs the IPC-pending map — which the TUI answers through its
    /// approval UI. The swap must reproduce THAT shape too (no accidental
    /// handler carried from nowhere).
    #[tokio::test(flavor = "multi_thread")]
    async fn new_session_keeps_the_ipc_approval_map_for_tui_sessions() {
        let home = tempfile::tempdir().expect("home tempdir");
        let xdg = tempfile::tempdir().expect("xdg tempdir");
        let _env = TestEnvGuard::set(&[
            ("HOME", home.path().to_str().expect("home path")),
            ("XDG_CONFIG_HOME", xdg.path().to_str().expect("xdg path")),
            ("DEEPSEEK_API_KEY", "dummy"),
        ]);
        let ws = tempfile::tempdir().expect("workspace tempdir");

        let mut rt = build_session_runtime(
            offline_config(),
            ws.path(),
            true,
            true,
            Some("b1-ipc-map-swap".into()),
            None, // TUI shape: no handler, IPC map instead
            None,
        )
        .await
        .expect("session runtime builds (dummy key injected via env guard)");
        assert!(
            rt.take_ipc_approval_pending().is_some(),
            "precondition: no handler => IPC pending map"
        );

        rt.new_session().await.expect("/new supervisor swap");
        assert!(
            rt.take_ipc_approval_pending().is_some(),
            "the swapped session must expose the fresh IPC approval map"
        );
        assert!(rt.approval_handler.is_none(), "no handler was configured");

        rt.finish(EndReason::UserExit).await;
    }

    /// M1: a headless orchestrated run must not drop out of its run on
    /// `/new` — the orchestration context (`NCA_ORCH_*`) is carried over.
    #[tokio::test(flavor = "multi_thread")]
    async fn new_session_carries_the_orchestration_context() {
        let home = tempfile::tempdir().expect("home tempdir");
        let xdg = tempfile::tempdir().expect("xdg tempdir");
        let _env = TestEnvGuard::set(&[
            ("HOME", home.path().to_str().expect("home path")),
            ("XDG_CONFIG_HOME", xdg.path().to_str().expect("xdg path")),
            ("DEEPSEEK_API_KEY", "dummy"),
        ]);
        let ws = tempfile::tempdir().expect("workspace tempdir");
        let orchestration = OrchestrationContext {
            orchestrator: Some("paperclip-wrapper".into()),
            run_id: Some("run-42".into()),
            ..Default::default()
        };

        let mut rt = build_session_runtime(
            offline_config(),
            ws.path(),
            true,
            false,
            Some("m1-orch-swap".into()),
            None,
            Some(orchestration),
        )
        .await
        .expect("session runtime builds (dummy key injected via env guard)");

        rt.new_session().await.expect("/new supervisor swap");

        let carried = rt
            .supervisor
            .orchestration_context()
            .expect("M1: orchestration context must survive /new");
        assert_eq!(carried.run_id.as_deref(), Some("run-42"));
        assert_eq!(carried.orchestrator.as_deref(), Some("paperclip-wrapper"));

        rt.finish(EndReason::UserExit).await;
    }

    /// M5: the new supervisor is built BEFORE the old one is finished. A
    /// failed create must therefore leave the current session exactly as it
    /// was — still Running, same id, still able to finish — instead of
    /// stranding the user on a session that is already Completed on disk.
    ///
    /// Failure injection: `<ws>/.nca` is replaced by a regular file, so the
    /// fresh supervisor's initial session-store write cannot create
    /// `<ws>/.nca/sessions/` and `create` fails.
    #[tokio::test(flavor = "multi_thread")]
    async fn new_session_create_failure_leaves_the_old_session_running() {
        let home = tempfile::tempdir().expect("home tempdir");
        let xdg = tempfile::tempdir().expect("xdg tempdir");
        let _env = TestEnvGuard::set(&[
            ("HOME", home.path().to_str().expect("home path")),
            ("XDG_CONFIG_HOME", xdg.path().to_str().expect("xdg path")),
            ("DEEPSEEK_API_KEY", "dummy"),
        ]);
        let ws = tempfile::tempdir().expect("workspace tempdir");

        let mut rt = build_session_runtime(
            offline_config(),
            ws.path(),
            true,
            false,
            Some("m5-create-failure".into()),
            None,
            None,
        )
        .await
        .expect("session runtime builds (dummy key injected via env guard)");
        let old_id = rt.session_id().to_string();

        // Break the fresh supervisor's store: a regular file where the
        // `.nca` directory belongs.
        let nca_dir = ws.path().join(".nca");
        std::fs::remove_dir_all(&nca_dir).expect("drop the .nca dir");
        std::fs::write(&nca_dir, b"not a directory").expect("blocking file");

        let err = rt
            .new_session()
            .await
            .expect_err("create must fail while the store is unwritable");
        assert!(!err.is_empty(), "the failure reason must be surfaced");

        // M5 invariant: the OLD session is untouched — same id, still
        // Running, and its supervisor was never finished.
        assert_eq!(
            rt.session_id(),
            old_id,
            "a failed swap must not roll the session id"
        );
        assert_eq!(
            rt.snapshot().status,
            SessionStatus::Running,
            "a failed swap must leave the current session Running"
        );

        // And it is still usable: unblock the store and swap successfully.
        std::fs::remove_file(&nca_dir).expect("unblock");
        std::fs::create_dir_all(ws.path().join(".nca/sessions")).expect("restore sessions dir");
        rt.new_session()
            .await
            .expect("/new works once the store is writable again");
        assert_ne!(rt.session_id(), old_id, "the retry rolled the session id");

        rt.finish(EndReason::UserExit).await;
    }

    /// M1: the shared MCP report formatter names every server outcome (the
    /// post-`/new` swap reuses it so a freshly reloaded server set is not
    /// silently unreported).
    #[test]
    fn mcp_report_lines_render_every_outcome() {
        use nca_core::tools::mcp::McpServerLoadReport;

        let reports = vec![
            McpServerLoadReport {
                name: "ready".into(),
                tool_count: Some(3),
                error: None,
            },
            McpServerLoadReport {
                name: "single".into(),
                tool_count: Some(1),
                error: None,
            },
            McpServerLoadReport {
                name: "broken".into(),
                tool_count: None,
                error: Some("boom".into()),
            },
            McpServerLoadReport {
                name: "mystery".into(),
                tool_count: None,
                error: None,
            },
        ];
        let lines = mcp_report_lines(&reports);
        assert_eq!(lines.len(), 4);
        assert!(
            lines[0].contains("[mcp] loaded ready (3 tools)"),
            "{lines:?}"
        );
        assert!(
            lines[1].contains("[mcp] loaded single (1 tool)"),
            "{lines:?}"
        );
        assert!(lines[2].contains("FAILED broken: boom"), "{lines:?}");
        assert!(lines[3].contains("FAILED mystery"), "{lines:?}");
    }
}
