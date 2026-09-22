//! `/model` routed to the ACTIVE persona — regression tests.
//!
//! The bug: the TUI/REPL `/model` flow always wrote the base config's
//! default-provider slot (`apply_model_override`). With a specialist profile
//! active (`[agents.<name>] provider/model` pins), the change was transient —
//! the next `apply_agent_profile` (agent switch, resume) re-pinned the
//! profile's model, and specialist subagent spawns (profile-authoritative
//! routing) never saw it. Net effect: only @orchestrator's model was durably
//! changeable from the UI.
//!
//! Run: `cargo test -p nca-runtime --test model_switch_active_persona`

use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
use nca_common::config::{AgentProfileConfig, NcaConfig, PermissionMode, ProviderKind};
use nca_common::message::Message;
use nca_common::tool::ToolDefinition;
use nca_core::provider::{Provider, ProviderError, StreamChunk};
use nca_runtime::supervisor::{Supervisor, SupervisorConfig};

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
        unreachable!("these tests must not call the provider")
    }
}

/// User-shaped config: base default zhipuai/glm-5.3, specialists pin
/// provider+model (mirrors a real `[agents.*]` setup).
fn user_shaped_config() -> NcaConfig {
    let mut config = NcaConfig::default();
    config.permissions.mode = PermissionMode::BypassPermissions;
    config.memory.context.auto_detect_context_window = false;
    config.memory.context.query_provider_models_api = false;
    config.memory.context.enable_auto_summarize = false;

    config.provider.default = ProviderKind::ZhipuAI;
    config.provider.zhipuai.api_key = Some("zkey".into());
    config.provider.kimi.api_key = Some("kkey".into());
    config.provider.zhipuai.model = "glm-5.3".into();
    config.provider.kimi.model = "k3".into();

    let mk = |provider: ProviderKind, model: &str| AgentProfileConfig {
        provider: Some(provider),
        model: Some(model.into()),
        ..Default::default()
    };
    config
        .agents
        .insert("fixer".into(), mk(ProviderKind::ZhipuAI, "glm-5.3"));
    config
        .agents
        .insert("oracle".into(), mk(ProviderKind::Kimi, "k3"));
    config
}

async fn sup(ws: &Path) -> Supervisor {
    Supervisor::create(SupervisorConfig {
        config: user_shaped_config(),
        workspace_root: ws.to_path_buf(),
        safe_mode: true,
        interactive_approvals: false,
        session_id: Some("model-switch-active-persona".into()),
        approval_handler: None,
        orchestration_context: None,
        agent_name: None,
        provider: Some(Arc::new(UnusedProvider)),
    })
    .await
    .expect("supervisor create")
}

fn env_and_ws() -> (TestEnvGuard, tempfile::TempDir) {
    let home = tempfile::tempdir().expect("home tempdir");
    let xdg = tempfile::tempdir().expect("xdg tempdir");
    let env = TestEnvGuard::set(&[
        ("HOME", home.path().to_str().unwrap()),
        ("XDG_CONFIG_HOME", xdg.path().to_str().unwrap()),
    ]);
    let ws = tempfile::tempdir().expect("workspace tempdir");
    (env, ws)
}

/// The core regression: a `/model` change on an active specialist profile
/// must survive re-switching to that profile (and reach the live config that
/// subagent spawns route against).
#[tokio::test(flavor = "multi_thread")]
async fn profile_model_change_survives_agent_re_switch() {
    let (_env, ws) = env_and_ws();
    let mut sup = sup(ws.path()).await;

    sup.apply_agent_profile(Some("fixer"))
        .expect("switch fixer");
    assert_eq!(sup.model, "glm-5.3");

    let applied = sup
        .set_model_for_active_agent("glm-5.3-flash")
        .expect("model change applies");
    assert_eq!(applied.as_deref(), Some("fixer"), "fixer's profile updated");
    assert_eq!(sup.model, "glm-5.3-flash");

    // Switch away and back: the profile must NOT snap back to glm-5.3.
    sup.apply_agent_profile(Some("oracle"))
        .expect("switch oracle");
    assert_eq!(sup.model, "k3");
    sup.apply_agent_profile(Some("fixer")).expect("switch back");
    assert_eq!(
        sup.model, "glm-5.3-flash",
        "REGRESSION: profile model change lost on agent re-switch"
    );

    // Subagent spawns read the live config's profile entry.
    let live = sup.live_config();
    let live = live.read().unwrap();
    assert_eq!(
        live.agent_profile("fixer").and_then(|p| p.model.as_deref()),
        Some("glm-5.3-flash"),
        "spawned specialists must inherit the updated profile model"
    );
}

/// The change persists through the whole-config workspace save as a profile
/// definition — and still never pollutes the provider slot (config purity).
#[tokio::test(flavor = "multi_thread")]
async fn profile_model_change_persists_and_stays_out_of_provider_slot() {
    let (_env, ws) = env_and_ws();
    let mut sup = sup(ws.path()).await;

    sup.apply_agent_profile(Some("fixer"))
        .expect("switch fixer");
    sup.set_model_for_active_agent("glm-5.3-flash")
        .expect("apply");
    sup.config()
        .save_workspace_file(ws.path())
        .expect("workspace save");

    let disk = NcaConfig::load_for_workspace(ws.path()).expect("reload");
    assert_eq!(
        disk.agent_profile("fixer").and_then(|p| p.model.as_deref()),
        Some("glm-5.3-flash"),
        "profile definition must persist to config.local.toml"
    );
    assert_eq!(
        disk.provider.zhipuai.model, "glm-5.3",
        "user-level provider slot must stay clean (config purity)"
    );
}

/// Cross-provider alias: picking a kimi model while on a zhipuai profile
/// retargets the profile's provider, mirroring the base flow.
#[tokio::test(flavor = "multi_thread")]
async fn cross_provider_alias_retargets_profile_provider() {
    let (_env, ws) = env_and_ws();
    let mut sup = sup(ws.path()).await;

    sup.apply_agent_profile(Some("fixer"))
        .expect("switch fixer");
    sup.set_model_for_active_agent("k3").expect("apply");
    assert_eq!(sup.model, "k3");
    assert_eq!(sup.active_provider(), ProviderKind::Kimi);
    let profile = sup.config().agent_profile("fixer").expect("profile");
    assert_eq!(profile.provider, Some(ProviderKind::Kimi));
    assert_eq!(profile.model.as_deref(), Some("k3"));
}

/// Without an active profile the classic base flow is unchanged.
#[tokio::test(flavor = "multi_thread")]
async fn base_flow_unchanged_on_default_persona() {
    let (_env, ws) = env_and_ws();
    let mut sup = sup(ws.path()).await;

    let applied = sup.set_model_for_active_agent("glm-5.2").expect("apply");
    assert_eq!(applied, None, "no profile active — base flow");
    assert_eq!(sup.model, "glm-5.2");
    assert_eq!(sup.config().provider.zhipuai.model, "glm-5.2");
}

/// `/provider` on an active profile retargets the profile and drops the old
/// model pin (its name belongs to the old provider's namespace).
#[tokio::test(flavor = "multi_thread")]
async fn provider_switch_on_profile_updates_profile() {
    let (_env, ws) = env_and_ws();
    let mut sup = sup(ws.path()).await;

    sup.apply_agent_profile(Some("fixer"))
        .expect("switch fixer");
    let applied = sup
        .set_provider_for_active_agent(ProviderKind::Kimi)
        .expect("apply");
    assert_eq!(applied.as_deref(), Some("fixer"));
    assert_eq!(sup.active_provider(), ProviderKind::Kimi);
    assert_eq!(sup.model, "k3", "inherits the new provider's default model");
    let profile = sup.config().agent_profile("fixer").expect("profile");
    assert_eq!(profile.provider, Some(ProviderKind::Kimi));
    assert_eq!(profile.model, None, "old-provider model pin cleared");
}
