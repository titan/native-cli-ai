//! Provider-injection seam — runtime-level integration tests.
//!
//! Spec (authoritative): `docs/plans/provider-injection-seam-design.md`
//! §Test matrix I1–I5. Exercises `SupervisorConfig::provider` (create) and the
//! new 7th `resume` param, proving that an injected `Arc<dyn Provider>` is used
//! verbatim — `build_provider` is skipped, so no API key is required — and that
//! the seam is discarded (loudly) by `apply_agent_profile`/`apply_nca_config`.
//!
//! No network: the mock provider answers in-process. The only bounded waits are
//! the fanout drains for the resume test.
//!
//! Run: `cargo test -p nca-runtime --test provider_injection`

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use nca_common::config::{AgentProfileConfig, NcaConfig, PermissionMode, ProviderKind};
use nca_common::event::EndReason;
use nca_common::message::Message;
use nca_common::tool::ToolDefinition;
use nca_core::provider::{Provider, ProviderError, StreamChunk};
use nca_runtime::session_utils::spawn_event_fanout;
use nca_runtime::supervisor::{Supervisor, SupervisorConfig};

// ---------------------------------------------------------------------------
// Scaffolding (mirrors cost_guard_resume.rs / phase_c.rs)
// ---------------------------------------------------------------------------

/// Deterministic offline config with NO API key. `create` with an injected
/// provider must succeed because `build_provider` is skipped; `create` with
/// `provider: None` and a keyless config must fail (see I2, which uses OpenAI
/// whose `from_config` validates eagerly — DeepSeek validates lazily).
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
    provider: Option<Arc<dyn Provider>>,
) -> Supervisor {
    let mut sup = Supervisor::create(SupervisorConfig {
        config,
        workspace_root: ws.to_path_buf(),
        safe_mode: true,
        interactive_approvals: false,
        session_id: Some(session_id.into()),
        approval_handler: None,
        orchestration_context: None,
        agent_name: None,
        provider,
    })
    .await
    .expect("supervisor create must succeed");

    // Suppress the best-effort title-generation provider call so the mock's
    // call count reflects exactly the turn's `chat` calls (mirrors
    // cost_guard_resume.rs).
    sup.set_session_title(Some(session_id.into()));
    sup
}

/// Mirrors cost_guard_resume.rs: take the handle (event_rx + commit sender),
/// spawn the fanout, hand back the log path + JoinHandle.
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

/// Graceful close: dropping the supervisor closes every event-channel sender,
/// so the fanout drains its buffer and exits.
async fn drain_fanout(sup: Supervisor, fanout: tokio::task::JoinHandle<()>) {
    drop(sup);
    tokio::time::timeout(Duration::from_secs(5), fanout)
        .await
        .expect("fanout must drain and exit within 5s of the sender drop")
        .expect("fanout task must complete without panicking");
}

// ---------------------------------------------------------------------------
// I1 — create with an injected provider skips `build_provider` (no key needed)
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn i1_create_with_injected_provider_skips_build_provider() {
    let ws = tempfile::tempdir().expect("tempdir");
    let mock = CountingTextProvider::new("mock-ok");
    let mut sup = create_sup(ws.path(), "i1", offline_config_no_key(), Some(mock.clone())).await;

    let out = sup.run_turn("hello").await.expect("turn succeeds");
    assert_eq!(out, "mock-ok");
    assert_eq!(mock.call_count(), 1, "exactly one provider call");
    drop(sup);
}

// ---------------------------------------------------------------------------
// I2 — create with `provider: None` and a keyless config fails loudly
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn i2_create_without_injection_keyless_fails() {
    let ws = tempfile::tempdir().expect("tempdir");
    // OpenAI validates eagerly at build time (DeepSeek validates lazily).
    let mut config = offline_config_no_key();
    config.provider.default = ProviderKind::OpenAi;

    let result = Supervisor::create(SupervisorConfig {
        config,
        workspace_root: ws.path().to_path_buf(),
        safe_mode: true,
        interactive_approvals: false,
        session_id: Some("i2".into()),
        approval_handler: None,
        orchestration_context: None,
        agent_name: None,
        provider: None,
    })
    .await;
    let err = match result {
        Ok(_) => panic!("keyless OpenAI config must fail to build a provider"),
        Err(e) => e,
    };
    assert!(
        matches!(err, ProviderError::Configuration(_)),
        "expected a configuration error, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// I3 — resume with an injected provider uses it on the resumed session
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn i3_resume_uses_injected_provider() {
    let ws = tempfile::tempdir().expect("tempdir");
    let sid = "i3-resume";

    // ── Session 1: create with mock1 → turn 1 → finish ──
    let mock1 = CountingTextProvider::new("one");
    let mut sup = create_sup(ws.path(), sid, offline_config_no_key(), Some(mock1.clone())).await;
    let (log_path, fanout) = wire_fanout(&mut sup).await;
    let out = sup.run_turn("first").await.expect("turn 1 succeeds");
    assert_eq!(out, "one");
    sup.finish(EndReason::Completed).await;
    drain_fanout(sup, fanout).await;

    // The session must be resumable (json + fresh-format log both present).
    assert!(log_path.exists(), "event log must be written by the fanout");

    // ── Resume with mock2 (7th param) → turn 2 drives mock2, not mock1 ──
    let mock2 = CountingTextProvider::new("two");
    let mut sup2 = Supervisor::resume(
        offline_config_no_key(),
        ws.path(),
        true,
        false,
        sid,
        None,
        Some(mock2.clone()),
    )
    .await
    .expect("resume must succeed");

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
// I4 — reset_for_new_session preserves the injected provider
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn i4_reset_for_new_session_preserves_injected_provider() {
    let ws = tempfile::tempdir().expect("tempdir");
    let mock = CountingTextProvider::new("mock-ok");
    let mut sup = create_sup(ws.path(), "i4", offline_config_no_key(), Some(mock.clone())).await;

    let out1 = sup.run_turn("one").await.expect("turn 1 succeeds");
    assert_eq!(out1, "mock-ok");

    sup.reset_for_new_session();
    // `reset_for_new_session` clears the title, so re-suppress title generation.
    sup.set_session_title(Some("i4-reset".into()));

    let out2 = sup.run_turn("two").await.expect("turn 2 succeeds");
    assert_eq!(out2, "mock-ok");
    assert_eq!(mock.call_count(), 2, "injected provider survives reset");
    drop(sup);
}

// ---------------------------------------------------------------------------
// I5 — apply_agent_profile discards the injected provider and fails loudly
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn i5_apply_agent_profile_discards_injected_provider_loudly() {
    let ws = tempfile::tempdir().expect("tempdir");
    // A profile that switches to OpenAI (eager key validation) so the rebuild
    // fails loudly on the missing key rather than silently hitting a network.
    let mut config = offline_config_no_key();
    config.agents.insert(
        "openai-profile".into(),
        AgentProfileConfig {
            provider: Some(ProviderKind::OpenAi),
            ..Default::default()
        },
    );

    let mock = CountingTextProvider::new("unused");
    let mut sup = create_sup(ws.path(), "i5", config, Some(mock.clone())).await;
    assert_eq!(mock.call_count(), 0, "mock is not called during create");

    let err = sup
        .apply_agent_profile(Some("openai-profile"))
        .expect_err("profile switch to a keyless OpenAI must fail");
    assert!(
        matches!(err, ProviderError::Configuration(_)),
        "expected a configuration error, got: {err}"
    );
    assert_eq!(
        mock.call_count(),
        0,
        "the injected mock is discarded — build_provider fails before any call"
    );
    drop(sup);
}
