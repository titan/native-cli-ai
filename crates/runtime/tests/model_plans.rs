//! Named model plans (`/plan <name>`) — regression tests.
//!
//! `Supervisor::apply_plan` pins each covered agent's provider/model into its
//! `[agents.<name>]` entry (both config layers, so routing survives switches,
//! resumes, and subagent spawns), records `active_plan`, and hot-swaps the
//! ACTIVE persona when the plan covers it. Agents not listed in the plan keep
//! their current routing — plans are partial overrides, not full resets.
//! Unknown plans fail with an error listing the available names.
//!
//! Run: `cargo test -p nca-runtime --test model_plans`

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
use nca_common::config::{AgentProfileConfig, NcaConfig, PermissionMode, PlanEntry, ProviderKind};
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

/// User-shaped base config: default zhipuai, specialists pin provider+model
/// (mirrors a real `[agents.*]` setup). No plans defined yet.
fn base_config() -> NcaConfig {
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

/// Convenience builder for a single `[plans.*]` entry.
fn entry(provider: Option<ProviderKind>, model: Option<&str>) -> PlanEntry {
    PlanEntry {
        provider,
        model: model.map(str::to_string),
    }
}

async fn sup(ws: &Path, config: NcaConfig) -> Supervisor {
    Supervisor::create(SupervisorConfig {
        config,
        workspace_root: ws.to_path_buf(),
        safe_mode: true,
        interactive_approvals: false,
        session_id: Some("model-plans".into()),
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

/// Core routing contract: applying a plan writes each covered agent's pins
/// into the profile table (both config layers), records `active_plan`, and
/// reports the resulting routes in the outcome.
#[tokio::test(flavor = "multi_thread")]
async fn apply_plan_pins_agents_and_records_active_plan() {
    let (_env, ws) = env_and_ws();
    let mut config = base_config();
    config.plans.insert(
        "economy".into(),
        BTreeMap::from([
            ("oracle".into(), entry(Some(ProviderKind::Kimi), Some("k3"))),
            ("fixer".into(), entry(None, Some("glm-5.2"))),
        ]),
    );
    let mut sup = sup(ws.path(), config).await;

    let outcome = sup.apply_plan("economy").expect("apply economy");

    assert_eq!(outcome.plan, "economy");
    assert_eq!(
        outcome.active_agent_swapped, None,
        "no active persona to hot-swap"
    );
    // `[plans.<name>]` is a BTreeMap → deterministic (sorted) agent order.
    assert_eq!(outcome.changes.len(), 2);
    assert_eq!(outcome.changes[0].agent, "fixer");
    assert_eq!(outcome.changes[0].provider, Some(ProviderKind::ZhipuAI));
    assert_eq!(outcome.changes[0].model.as_deref(), Some("glm-5.2"));
    assert_eq!(outcome.changes[1].agent, "oracle");
    assert_eq!(outcome.changes[1].provider, Some(ProviderKind::Kimi));
    assert_eq!(outcome.changes[1].model.as_deref(), Some("k3"));

    // self.config layer: pins landed on the agent profiles + active_plan set.
    assert_eq!(
        sup.config().agents["fixer"].model.as_deref(),
        Some("glm-5.2")
    );
    assert_eq!(
        sup.config().agents["oracle"].provider,
        Some(ProviderKind::Kimi)
    );
    assert_eq!(sup.config().active_plan.as_deref(), Some("economy"));

    // Live-config mirror: the spawn-time routing table children read.
    {
        let live = sup.live_config();
        let live = live.read().unwrap();
        assert_eq!(live.agents["fixer"].model.as_deref(), Some("glm-5.2"));
        assert_eq!(live.active_plan.as_deref(), Some("economy"));
    }

    // base_config layer: a profile rebuild starts from base_config, so the
    // pins and active_plan must survive switching the persona away.
    sup.apply_agent_profile(None).expect("switch to default");
    assert_eq!(
        sup.config().agents["fixer"].model.as_deref(),
        Some("glm-5.2"),
        "base_config must carry the plan pins"
    );
    assert_eq!(
        sup.config().active_plan.as_deref(),
        Some("economy"),
        "base_config must carry active_plan"
    );
    {
        let live = sup.live_config();
        let live = live.read().unwrap();
        assert_eq!(live.agents["oracle"].provider, Some(ProviderKind::Kimi));
    }
}

/// Model aliases resolve at apply time (via `config.model.resolve_alias`).
#[tokio::test(flavor = "multi_thread")]
async fn plan_model_alias_resolves_at_apply_time() {
    let (_env, ws) = env_and_ws();
    let mut config = base_config();
    config.model.aliases.insert("fast".into(), "glm-5.2".into());
    config.plans.insert(
        "economy".into(),
        BTreeMap::from([("fixer".into(), entry(None, Some("fast")))]),
    );
    let mut sup = sup(ws.path(), config).await;

    sup.apply_plan("economy").expect("apply");
    assert_eq!(
        sup.config().agents["fixer"].model.as_deref(),
        Some("glm-5.2"),
        "the alias must be resolved to the concrete model id"
    );
}

/// A provider-only entry clears the agent's model pin (the old model name
/// belongs to the old provider's namespace).
#[tokio::test(flavor = "multi_thread")]
async fn provider_only_entry_clears_model_pin() {
    let (_env, ws) = env_and_ws();
    let mut config = base_config();
    config.plans.insert(
        "economy".into(),
        BTreeMap::from([("fixer".into(), entry(Some(ProviderKind::Kimi), None))]),
    );
    let mut sup = sup(ws.path(), config).await;

    sup.apply_plan("economy").expect("apply");
    let fixer = &sup.config().agents["fixer"];
    assert_eq!(fixer.provider, Some(ProviderKind::Kimi));
    assert_eq!(fixer.model, None, "provider-only entry drops the model pin");
}

/// A model-only entry keeps the agent's provider pin.
#[tokio::test(flavor = "multi_thread")]
async fn model_only_entry_keeps_provider_pin() {
    let (_env, ws) = env_and_ws();
    let mut config = base_config();
    config.plans.insert(
        "economy".into(),
        BTreeMap::from([("fixer".into(), entry(None, Some("glm-5.0")))]),
    );
    let mut sup = sup(ws.path(), config).await;

    sup.apply_plan("economy").expect("apply");
    let fixer = &sup.config().agents["fixer"];
    assert_eq!(
        fixer.provider,
        Some(ProviderKind::ZhipuAI),
        "provider pin preserved by a model-only entry"
    );
    assert_eq!(fixer.model.as_deref(), Some("glm-5.0"));
}

/// Agents not listed in the plan keep their existing routing.
#[tokio::test(flavor = "multi_thread")]
async fn unlisted_agents_keep_pins() {
    let (_env, ws) = env_and_ws();
    let mut config = base_config();
    config.plans.insert(
        "economy".into(),
        BTreeMap::from([("fixer".into(), entry(None, Some("glm-5.0")))]),
    );
    let mut sup = sup(ws.path(), config).await;

    sup.apply_plan("economy").expect("apply");
    let oracle = &sup.config().agents["oracle"];
    assert_eq!(
        oracle.provider,
        Some(ProviderKind::Kimi),
        "unlisted agent must be untouched"
    );
    assert_eq!(oracle.model.as_deref(), Some("k3"));
}

/// Unknown plan → Configuration error listing the available plan names.
#[tokio::test(flavor = "multi_thread")]
async fn unknown_plan_errors_listing_available_names() {
    let (_env, ws) = env_and_ws();
    let mut config = base_config();
    config.plans.insert(
        "economy".into(),
        BTreeMap::from([("fixer".into(), entry(None, Some("glm-5.0")))]),
    );
    config.plans.insert(
        "premium".into(),
        BTreeMap::from([("oracle".into(), entry(Some(ProviderKind::Kimi), None))]),
    );
    let mut sup = sup(ws.path(), config).await;

    let err = sup.apply_plan("bogus").expect_err("unknown plan must fail");
    let ProviderError::Configuration(msg) = err else {
        panic!("expected ProviderError::Configuration, got {err:?}");
    };
    assert!(msg.contains("bogus"), "error must name the request: {msg}");
    assert!(msg.contains("economy"), "error must list economy: {msg}");
    assert!(msg.contains("premium"), "error must list premium: {msg}");
}

/// Unknown plan against an empty library reports "none configured".
#[tokio::test(flavor = "multi_thread")]
async fn unknown_plan_on_empty_library_reports_none_configured() {
    let (_env, ws) = env_and_ws();
    let mut sup = sup(ws.path(), base_config()).await;

    let err = sup.apply_plan("bogus").expect_err("unknown plan must fail");
    let ProviderError::Configuration(msg) = err else {
        panic!("expected ProviderError::Configuration, got {err:?}");
    };
    assert!(msg.contains("none configured"), "message: {msg}");
}

/// When the ACTIVE persona is covered by the plan, it is hot-swapped and the
/// outcome reports `(agent, resolved-model)`.
#[tokio::test(flavor = "multi_thread")]
async fn active_agent_covered_hot_swaps_persona() {
    let (_env, ws) = env_and_ws();
    let mut config = base_config();
    config.plans.insert(
        "economy".into(),
        BTreeMap::from([("fixer".into(), entry(None, Some("glm-5.2")))]),
    );
    let mut sup = sup(ws.path(), config).await;

    sup.apply_agent_profile(Some("fixer"))
        .expect("activate fixer");
    assert_eq!(sup.model, "glm-5.3");

    let outcome = sup.apply_plan("economy").expect("apply");
    assert_eq!(
        outcome.active_agent_swapped,
        Some(("fixer".to_string(), "glm-5.2".to_string()))
    );
    assert_eq!(
        sup.model, "glm-5.2",
        "active persona hot-swapped to the plan's model"
    );
}

/// When the ACTIVE persona is NOT covered, its model is unchanged and no swap
/// is reported (only the spawn-time routing table is refreshed).
#[tokio::test(flavor = "multi_thread")]
async fn active_agent_uncovered_leaves_persona_untouched() {
    let (_env, ws) = env_and_ws();
    let mut config = base_config();
    config.plans.insert(
        "economy".into(),
        BTreeMap::from([("oracle".into(), entry(Some(ProviderKind::Kimi), Some("k3")))]),
    );
    let mut sup = sup(ws.path(), config).await;

    sup.apply_agent_profile(Some("fixer"))
        .expect("activate fixer");
    assert_eq!(sup.model, "glm-5.3");

    let outcome = sup.apply_plan("economy").expect("apply");
    assert_eq!(outcome.active_agent_swapped, None);
    assert_eq!(sup.model, "glm-5.3", "active persona must stay put");
    // The covered non-active agent's live routing was refreshed for spawns.
    {
        let live = sup.live_config();
        let live = live.read().unwrap();
        assert_eq!(live.agents["oracle"].model.as_deref(), Some("k3"));
    }
}

/// `plans()` / `active_plan()` getters reflect the applied state.
#[tokio::test(flavor = "multi_thread")]
async fn plans_and_active_plan_getters_reflect_state() {
    let (_env, ws) = env_and_ws();
    let mut config = base_config();
    config.plans.insert(
        "economy".into(),
        BTreeMap::from([
            ("fixer".into(), entry(None, Some("glm-5.2"))),
            ("oracle".into(), entry(Some(ProviderKind::Kimi), Some("k3"))),
        ]),
    );
    let mut sup = sup(ws.path(), config).await;

    assert_eq!(sup.active_plan(), None, "no plan active before apply");
    assert!(sup.plans().contains_key("economy"));
    assert_eq!(sup.plans()["economy"].len(), 2);

    sup.apply_plan("economy").expect("apply");
    assert_eq!(sup.active_plan(), Some("economy"));
    assert_eq!(
        sup.plans()["economy"]["fixer"].model.as_deref(),
        Some("glm-5.2")
    );
    assert_eq!(
        sup.plans()["economy"]["oracle"].provider,
        Some(ProviderKind::Kimi)
    );
}
