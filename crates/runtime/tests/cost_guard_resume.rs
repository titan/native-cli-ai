//! Middleware chain composition — W4 (resume re-seed), RUNTIME level.
//!
//! Spec (authoritative): `docs/plans/middleware-chain-composition-design.md`
//! §Test matrix W4 + §1c ("Resume determinism"): on a replay-format resume
//! the cost tracker is re-seeded from the last log `CostUpdated`
//! (`supervisor.rs::seed_cost_tracker_from_log`), so a tripped session
//! re-trips immediately on its next provider call. This is the honest
//! integration variant of W4 — the supervisor-level resume harness exists
//! (`inbox.rs::resume_seeds_turn_ids`, `phase_c.rs` T24/T28 patterns), so
//! the real re-seed path is exercised end-to-end rather than simulated.
//! `crates/core/tests/middleware_composition.rs::w4_reeseed_*` carries the
//! core-level essence for the core suite.
//!
//! Flow: create (config carries `[middleware] cost_budget_usd = 1.0`) →
//! turn 1 reports a huge `StreamChunk::Usage` (fold → `CostUpdated` in the
//! log) → finish + drain → resume (fresh `AgentLoop`, tracker re-seeded from
//! the log) → turn 2 trips the guard BEFORE the provider call.
//!
//! No network: provider swapped in after `create()` via
//! `agent_mut().replace_provider(...)` (established pattern). No fixed
//! sleeps; the only waits are the bounded fanout drains.
//!
//! Run: `cargo test -p nca-runtime --test cost_guard_resume`

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use nca_common::config::{NcaConfig, PermissionMode};
use nca_common::event::{AgentEvent, EndReason};
use nca_common::message::Message;
use nca_common::tool::ToolDefinition;
use nca_core::provider::{Provider, ProviderError, StreamChunk};
use nca_runtime::session_store::read_event_log;
use nca_runtime::session_utils::spawn_event_fanout;
use nca_runtime::supervisor::{Supervisor, SupervisorConfig};

// ---------------------------------------------------------------------------
// Scaffolding (mirrors phase_c.rs / inbox.rs)
// ---------------------------------------------------------------------------

/// Deterministic offline config + the cost-guard knob: `create()` wires
/// `default_chain(&config.middleware, …)` at construction, so
/// `cost_budget_usd = Some(1.0)` pushes `CostGuardMiddleware` into the
/// session's chain from the start (and on resume).
fn offline_config_with_budget() -> NcaConfig {
    let mut config = NcaConfig::default();
    config.provider.deepseek.api_key = Some("test-key".into());
    config.permissions.mode = PermissionMode::BypassPermissions;
    config.memory.context.auto_detect_context_window = false;
    config.memory.context.query_provider_models_api = false;
    config.memory.context.enable_auto_summarize = false;
    config.middleware.cost_budget_usd = Some(1.0);
    config
}

async fn create_sup(ws: &Path, session_id: &str) -> Supervisor {
    Supervisor::create(SupervisorConfig {
        config: offline_config_with_budget(),
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
    .expect("supervisor create must succeed with the offline config")
}

/// Mirrors service.rs wiring (phase_c.rs): take the handle (event_rx +
/// commit sender), spawn the fanout, hand back the log path + JoinHandle.
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

/// Graceful close (phase_c.rs): dropping the supervisor closes every
/// event-channel sender, so the fanout drains its buffer and exits.
async fn drain_fanout(sup: Supervisor, fanout: tokio::task::JoinHandle<()>) {
    drop(sup);
    tokio::time::timeout(Duration::from_secs(5), fanout)
        .await
        .expect("fanout must drain and exit within 5s of the sender drop")
        .expect("fanout task must complete without panicking");
}

/// Turn-1 provider: a single usage-heavy text round. Replays on every call
/// (the guard must make the resumed session's call never happen).
struct UsageThenTextProvider {
    calls: AtomicU32,
}

impl UsageThenTextProvider {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            calls: AtomicU32::new(0),
        })
    }

    fn call_count(&self) -> u32 {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Provider for UsageThenTextProvider {
    async fn chat(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        _model: &str,
        _workspace_root: &Path,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        tokio::spawn(async move {
            let _ = tx
                .send(StreamChunk::Usage {
                    input_tokens: 1_000_000, // $3.00 at Sonnet-class rates
                    output_tokens: 0,
                    cache_creation_tokens: 0,
                    cache_read_tokens: 0,
                })
                .await;
            let _ = tx.send(StreamChunk::TextDelta("ok".into())).await;
            let _ = tx
                .send(StreamChunk::Finish {
                    reason: "stop".into(),
                })
                .await;
        });
        Ok(rx)
    }
}

/// Resumed-session provider: must NEVER be called (the re-seeded guard trips
/// pre-call). Fails loudly if it is.
struct MustNotBeCalledProvider {
    calls: AtomicU32,
}

impl MustNotBeCalledProvider {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            calls: AtomicU32::new(0),
        })
    }

    fn call_count(&self) -> u32 {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Provider for MustNotBeCalledProvider {
    async fn chat(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        _model: &str,
        _workspace_root: &Path,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(ProviderError::Other(
            "must never be called: the re-seeded cost guard trips pre-call".into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// W4 — budget state survives resume via the log re-seed
// ---------------------------------------------------------------------------

/// W4 — resume re-seed end-to-end: a session whose first turn burned $3.00
/// of estimated spend (folded from `StreamChunk::Usage` into a durable
/// `CostUpdated` log event) is finished and resumed. The resume rebuilds a
/// FRESH `AgentLoop`; `seed_cost_tracker_from_log` restores the tracker from
/// the last log `CostUpdated` — so the guard re-trips on the resumed
/// session's first provider call, BEFORE the provider is invoked, exactly
/// like the pre-resume session would. Spend stays spent (monotonic tracker).
#[tokio::test(flavor = "multi_thread")]
async fn w4_resume_re_seeds_cost_tracker_from_log_and_guard_retrips() {
    let ws = tempfile::tempdir().expect("tempdir");
    let sid = "w4-cost-guard-resume";

    // ── Session 1: create → turn 1 (expensive usage fold) → finish ──
    let mut sup = create_sup(ws.path(), sid).await;
    let provider1 = UsageThenTextProvider::new();
    sup.agent_mut()
        .replace_provider(Arc::clone(&provider1) as Arc<dyn Provider>);
    sup.set_session_title(Some("w4".into()));

    let (log_path, fanout) = wire_fanout(&mut sup).await;
    let out = sup.run_turn("first").await.expect("turn 1 succeeds");
    assert_eq!(out, "ok");
    assert_eq!(
        sup.agent().cost_tracker.input_tokens,
        1_000_000,
        "the usage fold must accumulate the huge input-token report"
    );
    assert_eq!(
        provider1.call_count(),
        1,
        "exactly one provider call for turn 1"
    );
    sup.finish(EndReason::Completed).await;
    drain_fanout(sup, fanout).await;

    // The durable re-seed source: a CostUpdated envelope carrying the
    // cumulative 1M input tokens must be in the event log.
    let envelopes = read_event_log(&log_path);
    let last_cost_updated = envelopes.iter().rev().find_map(|e| match &e.event {
        AgentEvent::CostUpdated { input_tokens, .. } => Some(*input_tokens),
        _ => None,
    });
    assert_eq!(
        last_cost_updated,
        Some(1_000_000),
        "the log must carry the last cumulative CostUpdated (the re-seed source)"
    );

    // ── Resume: fresh AgentLoop, tracker re-seeded from the log ──
    let mut sup2 = Supervisor::resume(
        offline_config_with_budget(),
        ws.path(),
        true,
        false,
        sid,
        None,
        None,
    )
    .await
    .expect("resume must succeed (healthy json + fresh-format log)");
    assert_eq!(
        sup2.agent().cost_tracker.input_tokens,
        1_000_000,
        "seed_cost_tracker_from_log must restore the cumulative spend on resume"
    );

    // ── Turn 2 on the resumed session: the guard trips pre-call ──
    let provider2 = MustNotBeCalledProvider::new();
    sup2.agent_mut()
        .replace_provider(Arc::clone(&provider2) as Arc<dyn Provider>);
    let (_log_path2, fanout2) = wire_fanout(&mut sup2).await;

    let err = sup2
        .run_turn("after resume")
        .await
        .expect_err("the re-seeded budget must trip the resumed session's first call");
    assert!(
        err.to_string()
            .contains("estimated session cost budget exhausted"),
        "error must name the budget trip: {err}"
    );
    assert_eq!(
        provider2.call_count(),
        0,
        "the guard trips BEFORE the provider — zero calls on the resumed session"
    );

    drain_fanout(sup2, fanout2).await;
}
