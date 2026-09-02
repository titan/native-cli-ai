use crate::ipc_pending::{ApprovalPendingMap, QuestionPendingMap};
use nca_common::config::{NcaConfig, PermissionMode, ProviderKind};
use nca_common::event::{AgentEvent, EndReason, QuestionSelection};
use nca_common::session::{OrchestrationContext, SessionSnapshot};
use nca_core::agent_driver::InboxItem;
use nca_core::approval::{ApprovalHandler, ApprovalVerdict};
use nca_core::provider::ProviderError;
use nca_core::tools::spawn_subagent::SpawnRequest;
use nca_runtime::ipc::IpcHandle;
use nca_runtime::supervisor::{Supervisor, SupervisorConfig, SupervisorHandle};
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

    pub fn set_model(&mut self, model: impl Into<String>) {
        let model = model.into();
        self.supervisor.model = model.clone();
        self.supervisor.agent_mut().model = model;
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

    pub fn request_cancel(&self) {
        self.supervisor.request_cancel();
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

    /// Start a fresh session: finish the current one (which persists the
    /// json — the single-writer discipline keeps saves at
    /// create/resume/finish only), generate a new ID, clear messages.
    pub async fn new_session(&mut self) -> Result<(), String> {
        self.supervisor.finish(EndReason::Completed).await;
        self.supervisor.reset_for_new_session();
        Ok(())
    }

    /// Switch to a different existing session in-place (no process restart).
    /// Finishes the current session (which persists the json), then resumes
    /// the target session, rebuilding
    /// handle and question-pending channels so the caller can rewire its event loop.
    pub async fn switch_to(&mut self, session_id: &str) -> Result<(), String> {
        self.supervisor.finish(EndReason::Completed).await;

        let mut supervisor = Supervisor::resume(
            self.supervisor.config().clone(),
            &self.supervisor.workspace_root,
            self.safe_mode,
            self.interactive_approvals,
            session_id,
            None,
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

    let mut supervisor = Supervisor::create(SupervisorConfig {
        config,
        workspace_root: workspace_root.to_path_buf(),
        safe_mode,
        interactive_approvals,
        session_id,
        approval_handler,
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
    let mut supervisor = Supervisor::resume(
        config,
        workspace_root,
        safe_mode,
        interactive_approvals,
        session_id,
        approval_handler,
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    /// Regression: `/mount` must survive a later `/model`-style whole-config
    /// save. SessionRuntime used to keep its own stale `NcaConfig` copy that
    /// `mount_path` never updated, so the next clone→apply→save cycle erased
    /// the persisted `extra_paths` from `.nca/config.local.toml`.
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    struct TestEnvGuard {
        previous: Vec<(String, Option<std::ffi::OsString>)>,
        _lock: MutexGuard<'static, ()>,
    }

    impl TestEnvGuard {
        fn set(vars: &[(&str, &str)]) -> Self {
            let lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
            let mut previous = Vec::new();
            for (key, value) in vars {
                previous.push((key.to_string(), std::env::var_os(key)));
                // SAFETY: the mutex serializes env mutation within this binary.
                unsafe { std::env::set_var(key, value) };
            }
            Self {
                previous,
                _lock: lock,
            }
        }
    }

    impl Drop for TestEnvGuard {
        fn drop(&mut self) {
            for (key, value) in self.previous.drain(..) {
                // SAFETY: still holding the env mutex.
                match value {
                    Some(value) => unsafe { std::env::set_var(&key, value) },
                    None => unsafe { std::env::remove_var(&key) },
                }
            }
        }
    }

    fn offline_config() -> NcaConfig {
        let mut config = NcaConfig::default();
        config.permissions.mode = PermissionMode::BypassPermissions;
        config.memory.context.auto_detect_context_window = false;
        config.memory.context.query_provider_models_api = false;
        config.memory.context.enable_auto_summarize = false;
        config
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn mount_survives_model_switch_config_save() {
        let home = tempfile::tempdir().expect("home tempdir");
        let xdg = tempfile::tempdir().expect("xdg tempdir");
        let _env = TestEnvGuard::set(&[
            ("HOME", home.path().to_str().unwrap()),
            ("XDG_CONFIG_HOME", xdg.path().to_str().unwrap()),
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
        .expect("session runtime builds (keyless deepseek validates lazily)");

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
            .expect("apply (keyless deepseek validates lazily)");
        rt.config()
            .save_workspace_file(ws.path())
            .expect("workspace save");

        let disk = NcaConfig::load_for_workspace(ws.path()).expect("reload");
        assert_eq!(
            disk.extra_paths, expected,
            "REGRESSION: model switch erased the persisted mount from the stale runner snapshot"
        );
    }
}
