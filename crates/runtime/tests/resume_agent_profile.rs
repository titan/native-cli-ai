//! Restore the active agent profile on resume — runtime integration tests.
//!
//! Spec (authoritative): `docs/plans/resume-agent-profile-design.md`
//! §Test matrix R1–R5. Proves that the active agent profile survives
//! `finish` + `resume`: the specialist system prompt, permission override,
//! and (with the injection seam) an injected mock provider all come back,
//! while a profile deleted between save and resume degrades to the default
//! harness prompt instead of failing.
//!
//! No network: providers are injected mocks or lazily-validated keyless
//! DeepSeek configs (never called). The only bounded wait is the fanout
//! drain in R3.
//!
//! Run: `cargo test -p nca-runtime --test resume_agent_profile`

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use nca_common::config::{AgentProfileConfig, NcaConfig, PermissionMode, ProviderKind};
use nca_common::event::EndReason;
use nca_common::message::{ContentPart, Message, MessageContent, Role};
use nca_common::session::OrchestrationContext;
use nca_common::tool::ToolDefinition;
use nca_core::provider::{Provider, ProviderError, StreamChunk};
use nca_runtime::session_utils::spawn_event_fanout;
use nca_runtime::supervisor::{Supervisor, SupervisorConfig};

const R1_PERSONA: &str = "You are R1-SPECIALIST, keeper of the resumed persona.";

// ---------------------------------------------------------------------------
// Scaffolding (mirrors provider_injection.rs)
// ---------------------------------------------------------------------------

/// Hermeticity: the default deepseek provider validates its API key eagerly
/// at build, so keyless dev environments must inject a dummy key before any
/// `Supervisor::create`/`resume`/`apply_agent_profile` rebuild. Set-once,
/// never restored: no test in this binary asserts missing-key behavior
/// (same intent as 19d57f8).
static DUMMY_KEY: std::sync::OnceLock<()> = std::sync::OnceLock::new();

fn hermetic_dummy_key() {
    DUMMY_KEY.get_or_init(|| {
        // SAFETY: set once before any test assertion can race it; the value
        // is a placeholder that no assertion inspects.
        unsafe { std::env::set_var("DEEPSEEK_API_KEY", "dummy") };
    });
}

/// Deterministic offline config with NO explicit API key — providers build
/// against the injected dummy key and are never called (no network).
fn offline_config_no_key() -> NcaConfig {
    let mut config = NcaConfig::default();
    config.permissions.mode = PermissionMode::BypassPermissions;
    config.memory.context.auto_detect_context_window = false;
    config.memory.context.query_provider_models_api = false;
    config.memory.context.enable_auto_summarize = false;
    config
}

/// A mock provider that returns a fixed text delta and counts its calls.
struct CountingTextProvider {
    calls: AtomicU32,
    text: &'static str,
}

impl CountingTextProvider {
    fn new(text: &'static str) -> Arc<Self> {
        Arc::new(Self {
            calls: AtomicU32::new(0),
            text,
        })
    }

    fn call_count(&self) -> u32 {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Provider for CountingTextProvider {
    async fn chat(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        _model: &str,
        _workspace_root: &Path,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let text = self.text;
        tokio::spawn(async move {
            let _ = tx.send(StreamChunk::TextDelta(text.to_string())).await;
            let _ = tx
                .send(StreamChunk::Finish {
                    reason: "stop".into(),
                })
                .await;
        });
        Ok(rx)
    }
}

async fn create_sup(
    ws: &Path,
    session_id: &str,
    config: NcaConfig,
    agent_name: Option<&str>,
    provider: Option<Arc<dyn Provider>>,
) -> Supervisor {
    hermetic_dummy_key();
    let mut sup = Supervisor::create(SupervisorConfig {
        config,
        workspace_root: ws.to_path_buf(),
        safe_mode: true,
        interactive_approvals: false,
        session_id: Some(session_id.into()),
        approval_handler: None,
        orchestration_context: None,
        agent_name: agent_name.map(str::to_string),
        provider,
    })
    .await
    .expect("supervisor create must succeed");
    // Suppress the best-effort title-generation provider call.
    sup.set_session_title(Some(session_id.into()));
    sup
}

async fn resume_sup(
    ws: &Path,
    session_id: &str,
    config: NcaConfig,
    provider: Option<Arc<dyn Provider>>,
) -> Supervisor {
    hermetic_dummy_key();
    let mut sup = Supervisor::resume(config, ws, true, false, session_id, None, provider)
        .await
        .expect("supervisor resume must succeed");
    sup.set_session_title(Some(session_id.into()));
    sup
}

/// Text of the most recent system message (the active system prompt).
/// `set_system_prompt` REPLACES (single-system-message model), so after a
/// profile switch there is exactly one System message — the live persona.
fn system_prompt_of(sup: &Supervisor) -> String {
    sup.agent()
        .messages
        .iter()
        .rev()
        .find(|m| matches!(m.role, Role::System))
        .map(|m| match &m.content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
        })
        .unwrap_or_default()
}

/// Mirrors provider_injection.rs: take the handle, spawn the event fanout.
async fn wire_fanout(sup: &mut Supervisor) -> (PathBuf, tokio::task::JoinHandle<()>) {
    let mut handle = sup.take_handle();
    let event_rx = handle.take_event_rx().expect("event rx");
    let log_path = handle.event_log_path.clone();
    let commit_tx = handle
        .take_turn_commit_tx()
        .map(|(tx, flag)| {
            flag.store(true, Ordering::SeqCst);
            tx
        })
        .expect("commit tx");
    let fanout = spawn_event_fanout(
        event_rx,
        log_path.clone(),
        None,
        None,
        None,
        Some(commit_tx),
    );
    (log_path, fanout)
}

async fn drain_fanout(sup: Supervisor, fanout: tokio::task::JoinHandle<()>) {
    drop(sup);
    tokio::time::timeout(Duration::from_secs(5), fanout)
        .await
        .expect("fanout must drain and exit within 5s of the sender drop")
        .expect("fanout task must complete without panicking");
}

// ---------------------------------------------------------------------------
// R6 — orchestration context survives resume (system-prompt section)
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn r6_resume_restores_orchestration_context_section() {
    let ws = tempfile::tempdir().expect("tempdir");
    let config = offline_config_no_key();
    hermetic_dummy_key();
    let orchestration = OrchestrationContext {
        orchestrator: Some("probe-wrapper".into()),
        run_id: Some("run-1".into()),
        ..Default::default()
    };

    let mut sup = Supervisor::create(SupervisorConfig {
        config: config.clone(),
        workspace_root: ws.path().to_path_buf(),
        safe_mode: true,
        interactive_approvals: false,
        session_id: Some("r6".into()),
        approval_handler: None,
        orchestration_context: Some(orchestration),
        agent_name: None,
        provider: None,
    })
    .await
    .expect("create with orchestration must succeed");
    sup.set_session_title(Some("r6".into()));
    let created_prompt = system_prompt_of(&sup);
    assert!(
        created_prompt.contains("Execution Context:"),
        "create builds the orchestration section"
    );
    sup.finish(EndReason::Completed).await;
    drop(sup);

    let sup2 = resume_sup(ws.path(), "r6", config, None).await;
    let resumed_prompt = system_prompt_of(&sup2);
    assert!(
        resumed_prompt.contains("Execution Context:"),
        "resumed system prompt must rebuild the orchestration section"
    );
    assert!(resumed_prompt.contains("probe-wrapper"));
    assert!(resumed_prompt.contains("run-1"));
    drop(sup2);
}

// ---------------------------------------------------------------------------
// R1 — create with an agent profile → finish → resume: persona restored
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn r1_resume_restores_profile_prompt_and_permissions() {
    let ws = tempfile::tempdir().expect("tempdir");
    let mut config = offline_config_no_key();
    config.agents.insert(
        "r1-specialist".into(),
        AgentProfileConfig {
            system_prompt: Some(R1_PERSONA.into()),
            permission_mode: Some(PermissionMode::Plan),
            ..Default::default()
        },
    );

    let mut sup = create_sup(ws.path(), "r1", config.clone(), Some("r1-specialist"), None).await;
    // Create-time sanity: the profile resolved (prompt + permission applied).
    assert!(system_prompt_of(&sup).contains(R1_PERSONA));
    assert_eq!(sup.config().permissions.mode, PermissionMode::Plan);
    sup.finish(EndReason::Completed).await;
    drop(sup);

    let sup2 = resume_sup(ws.path(), "r1", config, None).await;
    let prompt = system_prompt_of(&sup2);
    assert!(
        prompt.contains(R1_PERSONA),
        "resumed system prompt must restore the specialist persona"
    );
    assert_eq!(
        sup2.config().permissions.mode,
        PermissionMode::Plan,
        "resumed session must keep the profile's permission override"
    );
    drop(sup2);
}

// ---------------------------------------------------------------------------
// R2 — mid-session apply_agent_profile → finish → resume: persona restored
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn r2_resume_restores_mid_session_profile_switch() {
    let ws = tempfile::tempdir().expect("tempdir");
    let mut config = offline_config_no_key();
    config.agents.insert(
        "r2-persona".into(),
        AgentProfileConfig {
            system_prompt: Some("You are R2-SWITCHED, mid-session persona.".into()),
            ..Default::default()
        },
    );

    // Session starts on the default harness prompt…
    let mut sup = create_sup(ws.path(), "r2", config.clone(), None, None).await;
    assert!(!system_prompt_of(&sup).contains("R2-SWITCHED"));
    // …then the user Tab-cycles to the persona…
    sup.apply_agent_profile(Some("r2-persona"))
        .expect("mid-session switch must succeed");
    assert!(system_prompt_of(&sup).contains("R2-SWITCHED"));
    // …and exits gracefully (the only json save after create/resume).
    sup.finish(EndReason::UserExit).await;
    drop(sup);

    let sup2 = resume_sup(ws.path(), "r2", config, None).await;
    assert!(
        system_prompt_of(&sup2).contains("R2-SWITCHED"),
        "a profile switched mid-session must survive resume"
    );
    drop(sup2);
}

// ---------------------------------------------------------------------------
// R3 — resume with an injected mock + a profile that overrides provider:
//      the mock survives the profile path (seam intact), persona restored
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn r3_resume_with_injected_mock_keeps_profile_and_seam() {
    let ws = tempfile::tempdir().expect("tempdir");
    let sid = "r3-resume";
    let mut config = offline_config_no_key();
    // The profile overrides the provider (OpenAI — eagerly validated). With
    // an injected provider the override must be moot: the mock wins verbatim
    // and no keyless `build_provider` failure can happen at create OR resume.
    config.agents.insert(
        "r3-mock-friendly".into(),
        AgentProfileConfig {
            provider: Some(ProviderKind::OpenAi),
            system_prompt: Some("You are R3-SPECIALIST.".into()),
            ..Default::default()
        },
    );

    let mock1 = CountingTextProvider::new("one");
    let mut sup = create_sup(
        ws.path(),
        sid,
        config.clone(),
        Some("r3-mock-friendly"),
        Some(mock1.clone()),
    )
    .await;
    assert!(system_prompt_of(&sup).contains("R3-SPECIALIST"));
    let (log_path, fanout) = wire_fanout(&mut sup).await;
    let out = sup.run_turn("hello").await.expect("turn 1 succeeds");
    assert_eq!(out, "one");
    sup.finish(EndReason::Completed).await;
    drain_fanout(sup, fanout).await;
    assert!(log_path.exists(), "event log must be written by the fanout");

    // Resume with a fresh mock: the profile's persona AND the seam survive.
    let mock2 = CountingTextProvider::new("two");
    let mut sup2 = resume_sup(ws.path(), sid, config, Some(mock2.clone())).await;
    assert!(
        system_prompt_of(&sup2).contains("R3-SPECIALIST"),
        "resumed profile persona must be restored alongside the injected provider"
    );
    let out2 = sup2
        .run_turn("after resume")
        .await
        .expect("turn 2 succeeds");
    assert_eq!(out2, "two");
    assert_eq!(mock1.call_count(), 1, "mock1 only served the original turn");
    assert_eq!(mock2.call_count(), 1, "resumed session must use mock2");
    drop(sup2);
}

// ---------------------------------------------------------------------------
// R4 — profile deleted between save and resume: default prompt, no failure
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn r4_resume_with_profile_deleted_degrades_to_default() {
    let ws = tempfile::tempdir().expect("tempdir");
    let mut with_profile = offline_config_no_key();
    with_profile.agents.insert(
        "temp-persona".into(),
        AgentProfileConfig {
            system_prompt: Some("You are TEMP-PERSONA.".into()),
            ..Default::default()
        },
    );

    let mut sup = create_sup(ws.path(), "r4", with_profile, Some("temp-persona"), None).await;
    assert!(system_prompt_of(&sup).contains("TEMP-PERSONA"));
    sup.finish(EndReason::Completed).await;
    drop(sup);

    // The profile is gone (skill uninstalled / config edited). Resume must
    // succeed with the default harness prompt — never fail the session.
    let sup2 = resume_sup(ws.path(), "r4", offline_config_no_key(), None).await;
    assert!(
        !system_prompt_of(&sup2).contains("TEMP-PERSONA"),
        "deleted profile must degrade to the default harness prompt"
    );
    assert!(
        system_prompt_of(&sup2)
            .to_lowercase()
            .contains("permission"),
        "default harness prompt should be present"
    );
    drop(sup2);
}

// ---------------------------------------------------------------------------
// R5 — reset_for_new_session keeps the persona and persists its name
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn r5_reset_keeps_persona_and_next_session_resumes_with_it() {
    let ws = tempfile::tempdir().expect("tempdir");
    let mut config = offline_config_no_key();
    config.agents.insert(
        "r5-keeper".into(),
        AgentProfileConfig {
            system_prompt: Some("You are R5-KEEPER.".into()),
            ..Default::default()
        },
    );

    let mut sup = create_sup(ws.path(), "r5-old", config.clone(), Some("r5-keeper"), None).await;
    sup.reset_for_new_session();
    let new_session_id = sup.session_id().to_string();
    assert_ne!(new_session_id, "r5-old");
    assert!(
        system_prompt_of(&sup).contains("R5-KEEPER"),
        "reset must keep the specialist persona"
    );
    sup.finish(EndReason::Completed).await;
    drop(sup);

    // The reset session saved under the new id with the carried-over name.
    let sup2 = resume_sup(ws.path(), &new_session_id, config, None).await;
    assert!(
        system_prompt_of(&sup2).contains("R5-KEEPER"),
        "the post-reset session must resume with the kept persona"
    );
    drop(sup2);
}
