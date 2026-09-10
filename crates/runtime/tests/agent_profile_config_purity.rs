//! Regression: agent-profile provider/model overrides are transient session
//! state and must never leak into the user-level config that whole-config
//! saves persist.
//!
//! The bug: `Supervisor::apply_agent_profile` baked the profile's model into
//! `config.provider.<kind>.model` (the same slot user config uses). Any later
//! `/model` / `/provider` whole-config save then diffed the polluted snapshot
//! against defaults+global and wrote the agent's model into
//! `.nca/config.local.toml`.
//!
//! Run: `cargo test -p nca-runtime --test agent_profile_config_purity`

use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
use nca_common::config::{AgentProfileConfig, NcaConfig, PermissionMode, ProviderKind};
use nca_common::message::Message;
use nca_common::tool::ToolDefinition;
use nca_core::provider::{Provider, ProviderError, StreamChunk};
use nca_runtime::supervisor::{Supervisor, SupervisorConfig};

// ---------------------------------------------------------------------------
// Env isolation (same pattern as mount_config_persistence.rs).
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Scaffolding
// ---------------------------------------------------------------------------

struct UnusedProvider;

#[async_trait]
impl Provider for UnusedProvider {
    async fn chat(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        _model: &str,
        _workspace_root: &Path,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError> {
        unreachable!("profile/config persistence must not call the provider")
    }
}

fn offline_config_with_profile() -> NcaConfig {
    let mut config = NcaConfig::default();
    config.permissions.mode = PermissionMode::BypassPermissions;
    config.memory.context.auto_detect_context_window = false;
    config.memory.context.query_provider_models_api = false;
    config.memory.context.enable_auto_summarize = false;
    // User's own model choice.
    config.provider.deepseek.model = "dsv4-user".into();
    // An agent profile with its own transient model + permission mode.
    // Keyless deepseek builds lazily, so the switch succeeds offline.
    config.agents.insert(
        "specialist".into(),
        AgentProfileConfig {
            provider: Some(ProviderKind::DeepSeek),
            model: Some("dsv4-pro".into()),
            permission_mode: Some(PermissionMode::Plan),
            ..Default::default()
        },
    );
    config
}

async fn create_sup(ws: &Path) -> Supervisor {
    Supervisor::create(SupervisorConfig {
        config: offline_config_with_profile(),
        workspace_root: ws.to_path_buf(),
        safe_mode: true,
        interactive_approvals: false,
        session_id: Some("agent-profile-purity".into()),
        approval_handler: None,
        orchestration_context: None,
        agent_name: None,
        provider: Some(Arc::new(UnusedProvider)),
    })
    .await
    .expect("supervisor create must succeed")
}

#[tokio::test(flavor = "multi_thread")]
async fn agent_profile_model_never_reaches_user_config_or_disk() {
    let home = tempfile::tempdir().expect("home tempdir");
    let xdg = tempfile::tempdir().expect("xdg tempdir");
    let _env = TestEnvGuard::set(&[
        ("HOME", home.path().to_str().unwrap()),
        ("XDG_CONFIG_HOME", xdg.path().to_str().unwrap()),
        // Hermetic: keyless environments must not fail the provider build
        // (same pattern as mount_config_persistence.rs, cf. 19d57f8).
        ("DEEPSEEK_API_KEY", "dummy"),
    ]);

    let ws = tempfile::tempdir().expect("workspace tempdir");
    let mut sup = create_sup(ws.path()).await;

    // ── 1. Switch to the profile: session uses its model/mode, config stays clean.
    sup.apply_agent_profile(Some("specialist"))
        .expect("profile switch (keyless deepseek builds lazily)")
        .expect("profile must resolve");

    assert_eq!(sup.model, "dsv4-pro", "session must run the profile model");
    assert_eq!(
        sup.agent().approval.mode(),
        PermissionMode::Plan,
        "session must use the profile permission mode"
    );
    assert_eq!(
        sup.config().provider.deepseek.model,
        "dsv4-user",
        "profile model must not pollute user-level config"
    );
    assert_eq!(
        sup.config().permissions.mode,
        PermissionMode::BypassPermissions,
        "profile permission mode must not pollute user-level config"
    );

    // ── 2. `/model`-style flow: clone runtime config, set a new model, save.
    let mut cfg = sup.config().clone();
    cfg.apply_model_override("dsv4-next");
    sup.apply_nca_config(cfg)
        .expect("apply (keyless deepseek validates lazily)");
    sup.config()
        .save_workspace_file(ws.path())
        .expect("whole-config workspace save");

    // ── 3. The persisted file holds the user's model, never the profile's.
    // Note: `[agents.specialist] model = "dsv4-pro"` (the profile *definition*)
    // legitimately persists — only the provider slot must stay clean.
    let disk = NcaConfig::load_for_workspace(ws.path()).expect("reload after save");
    assert_eq!(
        disk.provider.deepseek.model, "dsv4-next",
        "user's model choice must be persisted"
    );
    let raw = std::fs::read_to_string(ws.path().join(".nca").join("config.local.toml"))
        .expect("read local config");
    assert!(
        !raw.contains("[provider.deepseek]\nmodel = \"dsv4-pro\""),
        "REGRESSION: agent profile model leaked into the persisted provider slot: {raw}"
    );

    // ── 4. Switching back to the default agent restores the user's model.
    sup.apply_agent_profile(None)
        .expect("switch back to default agent");
    assert_eq!(
        sup.model, "dsv4-next",
        "default agent must run the user's last model, not a stale profile value"
    );
}
