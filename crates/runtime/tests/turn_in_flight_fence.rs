//! Turn fence (`turn_in_flight`) — runtime integration tests.
//!
//! Pins the explicit fence contract around `Supervisor::run_turn_with_images`
//! that converts the old implicit "&mut self + single cmd consumer"
//! serialization into an executable contract:
//!
//! - F1: `apply_agent_profile` / `apply_nca_config` refuse with a
//!   `Configuration` error ("agent switch deferred: turn in flight") while
//!   the fence is set — and leave the session's persona/config untouched —
//!   then succeed once the fence clears.
//! - F2: the fence is actually driven by the turn lifecycle: set for the
//!   whole duration of a parked (gated) turn, cleared after completion.
//! - F3: panic safety — a provider panic unwinds the turn task and the RAII
//!   guard still clears the fence (a dead turn must never wedge every later
//!   agent switch).
//!
//! F1 drives the flag directly: with today's single-owner API surface a
//! *genuinely* concurrent `apply_*` while `run_turn` holds `&mut self` is
//! structurally impossible — the fence exists precisely so the future second
//! mutation entry (IPC extension, attach protocol) fails loudly instead of
//! silently relying on that exclusivity.
//!
//! No network: providers are injected mocks or built from a dummy-key
//! config and never called against a real endpoint.
//!
//! Run: `cargo test -p nca-runtime --test turn_in_flight_fence`

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use nca_common::config::{AgentProfileConfig, NcaConfig, PermissionMode};
use nca_common::message::{Message, Role};
use nca_common::tool::ToolDefinition;
use nca_core::provider::{Provider, ProviderError, StreamChunk};
use nca_runtime::supervisor::{Supervisor, SupervisorConfig};

/// Deterministic offline config with a dummy deepseek key so the
/// `apply_*` rebuild paths (`build_provider` from base config) succeed
/// without network or environment juggling.
fn offline_config() -> NcaConfig {
    let mut config = NcaConfig::default();
    config.provider.deepseek.api_key = Some("test-key".into());
    config.permissions.mode = PermissionMode::BypassPermissions;
    config.memory.context.auto_detect_context_window = false;
    config.memory.context.query_provider_models_api = false;
    config.memory.context.enable_auto_summarize = false;
    config
}

/// Provider whose first chat parks on a `Notify` gate (provably mid-turn);
/// release answers immediately with `answer`. Later calls answer at once.
struct GatedParkProvider {
    release: Arc<tokio::sync::Notify>,
    answer: &'static str,
    calls: AtomicUsize,
}

impl GatedParkProvider {
    fn new(answer: &'static str) -> Arc<Self> {
        Arc::new(Self {
            release: Arc::new(tokio::sync::Notify::new()),
            answer,
            calls: AtomicUsize::new(0),
        })
    }

    fn release(&self) {
        self.release.notify_one();
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Provider for GatedParkProvider {
    async fn chat(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        _model: &str,
        _workspace_root: &Path,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let release = self.release.clone();
        let answer = self.answer;
        tokio::spawn(async move {
            release.notified().await;
            let _ = tx.send(StreamChunk::TextDelta(answer.to_string())).await;
            let _ = tx
                .send(StreamChunk::Finish {
                    reason: "stop".into(),
                })
                .await;
        });
        Ok(rx)
    }
}

/// Provider that panics inside `chat` — proves the fence survives a turn
/// that dies by unwind.
struct PanickingProvider;

#[async_trait]
impl Provider for PanickingProvider {
    async fn chat(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        _model: &str,
        _workspace_root: &Path,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError> {
        panic!("provider exploded mid-turn");
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
    // Suppress the best-effort title-generation provider call.
    sup.set_session_title(Some(session_id.into()));
    sup
}

/// Bounded poll until `cond` holds — turns are async; observe, never sleep hard.
async fn wait_until(timeout: Duration, mut cond: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + timeout;
    while !cond() {
        if Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    true
}

fn system_prompt_contains(sup: &Supervisor, needle: &str) -> bool {
    sup.agent()
        .messages
        .iter()
        .any(|m| m.role == Role::System && m.content.event_preview().contains(needle))
}

// ---------------------------------------------------------------------------
// F1 — the fence refuses provider/config rebuilds and defers them
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn f1_apply_paths_defer_while_turn_in_flight() {
    let ws = tempfile::tempdir().expect("tempdir");
    let mut config = offline_config();
    config.agents.insert(
        "f1-persona".into(),
        AgentProfileConfig {
            system_prompt: Some("You are F1-PERSONA.".into()),
            ..Default::default()
        },
    );

    let mut sup = create_sup(ws.path(), "f1", config, None).await;
    let flag = sup.turn_in_flight_handle();
    assert!(!sup.is_turn_in_flight(), "fence clear before any turn");

    // Simulate a running turn (see module docs: real concurrency is
    // structurally impossible at today's single-owner API surface).
    flag.store(true, Ordering::SeqCst);

    let err = sup
        .apply_agent_profile(Some("f1-persona"))
        .expect_err("agent switch must defer while a turn is in flight");
    assert!(
        matches!(&err, ProviderError::Configuration(msg) if msg.contains("turn in flight")),
        "fence error must be a Configuration error naming the deferral: {err}"
    );
    assert!(
        err.to_string()
            .ends_with("agent switch deferred: turn in flight"),
        "verbatim fence message (UI surfaces it): {err}"
    );

    let err = sup
        .apply_nca_config(offline_config())
        .expect_err("config switch must defer while a turn is in flight");
    assert!(
        matches!(&err, ProviderError::Configuration(msg) if msg.contains("turn in flight")),
        "config switch shares the fence: {err}"
    );

    // No half-applied switch: the persona never landed while deferred.
    assert!(
        !system_prompt_contains(&sup, "F1-PERSONA"),
        "a deferred switch must not leak the persona"
    );

    // Fence clears → both switches succeed.
    flag.store(false, Ordering::SeqCst);
    let applied = sup
        .apply_agent_profile(Some("f1-persona"))
        .expect("agent switch succeeds once idle");
    assert_eq!(applied.as_deref(), Some("f1-persona"));
    assert!(system_prompt_contains(&sup, "F1-PERSONA"));
    sup.apply_nca_config(offline_config())
        .expect("config switch succeeds once idle");
}

// ---------------------------------------------------------------------------
// F2 — run_turn actually drives the fence for its whole duration
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn f2_run_turn_holds_fence_for_whole_turn() {
    let ws = tempfile::tempdir().expect("tempdir");
    let gated = GatedParkProvider::new("fenced answer");
    let mut sup = create_sup(ws.path(), "f2", offline_config(), Some(gated.clone())).await;
    let flag = sup.turn_in_flight_handle();
    assert!(!flag.load(Ordering::SeqCst));

    let turn = tokio::spawn(async move { sup.run_turn("park mid-turn").await });

    // Provably mid-turn: the provider was entered AND the fence is set.
    let reached = wait_until(Duration::from_secs(10), || {
        gated.call_count() >= 1 && flag.load(Ordering::SeqCst)
    })
    .await;
    assert!(reached, "turn must start: provider entered, fence set");
    assert!(flag.load(Ordering::SeqCst), "fence held while parked");

    gated.release();
    let out = tokio::time::timeout(Duration::from_secs(10), turn)
        .await
        .expect("turn must finish after release")
        .expect("turn task must not panic")
        .expect("turn must succeed");
    assert_eq!(out, "fenced answer");
    assert!(
        !flag.load(Ordering::SeqCst),
        "fence cleared after the turn completes"
    );
}

// ---------------------------------------------------------------------------
// F3 — panic safety: a dead turn releases the fence
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn f3_panicking_turn_still_clears_fence() {
    let ws = tempfile::tempdir().expect("tempdir");
    let mut sup = create_sup(
        ws.path(),
        "f3",
        offline_config(),
        Some(Arc::new(PanickingProvider)),
    )
    .await;
    let flag = sup.turn_in_flight_handle();

    let turn = tokio::spawn(async move { sup.run_turn("boom").await });

    // No mid-flight observation here: the panic races past any poller (the
    // F2 test pins that the flag IS set while a turn runs). What this test
    // pins is the aftermath — the unwind must still release the fence.
    let joined = tokio::time::timeout(Duration::from_secs(10), turn)
        .await
        .expect("panicked turn task must settle");
    assert!(
        joined.as_ref().is_err_and(|e| e.is_panic()),
        "provider panic must unwind the turn task: {joined:?}"
    );
    assert!(
        !flag.load(Ordering::SeqCst),
        "RAII guard clears the fence even on panic unwind — a dead turn \
         must never wedge every later agent switch"
    );
}
