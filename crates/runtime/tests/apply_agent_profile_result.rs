//! `apply_agent_profile` honest-result contract — runtime integration tests.
//!
//! Pins the `Result<Option<String>, ProviderError>` semantics landed with the
//! honest agent-switch reporting fix:
//!
//! - `Ok(Some(name))` — a registered `[agents.<name>]` profile resolved and
//!   was applied.
//! - `Ok(None)` — the session is on the default (@orchestrator) persona:
//!   reached via `name: None`, an unregistered `Some("orchestrator")`, or an
//!   unresolvable name (non-fatal by design — a resume with a dead recorded
//!   name must not hard-fail).
//! - `Err` — provider rebuild failure (already covered by
//!   `tests/provider_injection.rs` I5; not repeated here).
//!
//! Secondary pin: an unresolvable name still records `active_agent_name`
//! verbatim (resume symmetry) even though it reports `Ok(None)` — proven by
//! finish → resume restoring the verbatim recorded name, which then
//! (correctly) resolves to nothing and warns instead of failing.
//!
//! No network: providers are built with a dummy injected key (see
//! `TestEnvGuard`) and are never called.
//!
//! Run: `cargo test -p nca-runtime --test apply_agent_profile_result`

use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use nca_common::config::{AgentProfileConfig, NcaConfig, PermissionMode};
use nca_common::event::EndReason;
use nca_runtime::session_store::SessionStore;
use nca_runtime::supervisor::{Supervisor, SupervisorConfig};

/// Env isolation (same pattern as agent_profile_config_purity.rs / 19d57f8):
/// the default deepseek provider validates its API key eagerly at build, so
/// keyless environments inject a dummy key for the test's duration.
static ENV_MUTEX: Mutex<()> = Mutex::new(());

struct TestEnvGuard {
    previous: Option<std::ffi::OsString>,
    _lock: MutexGuard<'static, ()>,
}

impl TestEnvGuard {
    fn dummy_deepseek_key() -> Self {
        let lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let previous = std::env::var_os("DEEPSEEK_API_KEY");
        // SAFETY: the mutex serializes env mutation within this binary.
        unsafe { std::env::set_var("DEEPSEEK_API_KEY", "dummy") };
        Self {
            previous,
            _lock: lock,
        }
    }
}

impl Drop for TestEnvGuard {
    fn drop(&mut self) {
        // SAFETY: still holding the env mutex.
        match self.previous.take() {
            Some(value) => unsafe { std::env::set_var("DEEPSEEK_API_KEY", value) },
            None => unsafe { std::env::remove_var("DEEPSEEK_API_KEY") },
        }
    }
}

/// Deterministic offline config with NO API key (mirrors
/// resume_agent_profile.rs::offline_config_no_key).
fn offline_config_no_key() -> NcaConfig {
    let mut config = NcaConfig::default();
    config.permissions.mode = PermissionMode::BypassPermissions;
    config.memory.context.auto_detect_context_window = false;
    config.memory.context.query_provider_models_api = false;
    config.memory.context.enable_auto_summarize = false;
    config
}

fn config_with_registered_agent() -> NcaConfig {
    let mut config = offline_config_no_key();
    config.agents.insert(
        "registered-persona".into(),
        AgentProfileConfig {
            system_prompt: Some("You are REGISTERED-PERSONA.".into()),
            ..Default::default()
        },
    );
    config
}

async fn create_sup(ws: &Path, session_id: &str, config: NcaConfig) -> Supervisor {
    let mut sup = Supervisor::create(SupervisorConfig {
        config,
        workspace_root: ws.to_path_buf(),
        safe_mode: true,
        interactive_approvals: false,
        session_id: Some(session_id.into()),
        approval_handler: None,
        orchestration_context: None,
        agent_name: None,
        provider: None,
    })
    .await
    .expect("supervisor create must succeed");
    // Suppress the best-effort title-generation provider call.
    sup.set_session_title(Some(session_id.into()));
    sup
}

/// A1 — a registered `[agents.x]` profile switch reports the applied name.
#[tokio::test(flavor = "multi_thread")]
async fn a1_registered_profile_returns_ok_some() {
    let _env = TestEnvGuard::dummy_deepseek_key();
    let ws = tempfile::tempdir().expect("tempdir");
    let mut sup = create_sup(ws.path(), "a1", config_with_registered_agent()).await;

    let applied = sup
        .apply_agent_profile(Some("registered-persona"))
        .expect("registered profile switch must succeed");
    assert_eq!(applied.as_deref(), Some("registered-persona"));

    drop(sup);
}

/// A2 — `None`, unregistered `Some("orchestrator")`, and `Some("bogus")` all
/// report `Ok(None)` (default persona) without failing.
#[tokio::test(flavor = "multi_thread")]
async fn a2_default_persona_paths_return_ok_none() {
    let _env = TestEnvGuard::dummy_deepseek_key();
    let ws = tempfile::tempdir().expect("tempdir");
    let mut sup = create_sup(ws.path(), "a2", config_with_registered_agent()).await;

    // name: None → default persona.
    let applied = sup
        .apply_agent_profile(None)
        .expect("None switch must succeed");
    assert_eq!(applied, None);

    // Unregistered "orchestrator" → default persona (the Tab-cycle reset).
    let applied = sup
        .apply_agent_profile(Some("orchestrator"))
        .expect("unregistered orchestrator must succeed");
    assert_eq!(applied, None);

    // Unresolvable name → non-fatal fallback to default persona.
    let applied = sup
        .apply_agent_profile(Some("bogus"))
        .expect("bogus name must not hard-fail");
    assert_eq!(applied, None);

    drop(sup);
}

/// A2b — an unresolvable name reports `Ok(None)` yet still records
/// `active_agent_name` verbatim (resume symmetry): after finish → resume the
/// recorded name survives into the resumed session's meta, where it (again)
/// resolves to nothing and warns instead of failing.
#[tokio::test(flavor = "multi_thread")]
async fn a2b_bogus_name_records_verbatim_for_resume_symmetry() {
    let _env = TestEnvGuard::dummy_deepseek_key();
    let ws = tempfile::tempdir().expect("tempdir");
    let config = config_with_registered_agent();

    let mut sup = create_sup(ws.path(), "a2b", config.clone()).await;
    // Switch to a real profile first, then to a bogus one: the verbatim
    // record must be REPLACED by "bogus" even though nothing resolved.
    sup.apply_agent_profile(Some("registered-persona"))
        .expect("registered switch must succeed");
    let applied = sup
        .apply_agent_profile(Some("bogus"))
        .expect("bogus switch must not hard-fail");
    assert_eq!(applied, None, "bogus reports the honest None");
    sup.finish(EndReason::UserExit).await; // persists meta (agent_name)
    drop(sup);

    // The verbatim recorded name was persisted — proving the switch recorded
    // it despite reporting Ok(None).
    let loaded = SessionStore::new(ws.path().join(".nca/sessions"))
        .load("a2b")
        .await
        .expect("session json must load after finish");
    assert_eq!(loaded.meta.agent_name.as_deref(), Some("bogus"));

    // Resume with the dead recorded name must not hard-fail (R4 already pins
    // the prompt degradation; here we pin the name's survival).
    let sup2 = Supervisor::resume(config, ws.path(), true, false, "a2b", None, None)
        .await
        .expect("resume with a dead recorded name must not hard-fail");
    drop(sup2);
}
