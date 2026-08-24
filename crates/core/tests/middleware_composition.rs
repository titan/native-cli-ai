//! Middleware chain composition — integration matrix W1–W4.
//!
//! Spec (authoritative): `docs/plans/middleware-chain-composition-design.md`
//! (§Test matrix, integration row). Unit tests K1–G4 live inline in
//! `src/middleware.rs`; these tests pin the WIRING between a real
//! `AgentLoop`, the three new middlewares (`CompactionMiddleware`,
//! `CostGuardMiddleware`, `RetryMiddleware`), and the scripted provider
//! through `run_turn` — the migration of the driver's former inline
//! compaction block, the real budget-trip flow
//! (`StreamChunk::Usage` → cost tracker → driver-materialized
//! `session_usage` → guard), and the real `RateLimited` retry flow.
//!
//! W4 (resume re-seed) is scoped by the design to replay-format sessions at
//! the supervisor level (`supervisor.rs::seed_cost_tracker_from_log`); the
//! honest integration variant lives in
//! `crates/runtime/tests/cost_guard_resume.rs` (the supervisor-level harness
//! exists — `inbox.rs` / `phase_c.rs` patterns). This file additionally
//! carries the core-level essence of that contract (see `w4_*`).
//!
//! Run: `cargo test -p nca-core --test middleware_composition`

use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use nca_common::config::{PermissionConfig, PermissionMode, SmartCompactionMode};
use nca_common::event::AgentEvent;
use nca_common::message::{Message, MessageToolCall};
use nca_common::tool::ToolDefinition;
use nca_core::agent::AgentLoop;
use nca_core::approval::ApprovalPolicy;
use nca_core::middleware::{CompactionMiddleware, CostGuardMiddleware, RetryMiddleware};
use nca_core::provider::{Provider, ProviderError, StreamChunk};
use nca_core::tools::ToolRegistry;
use serde_json::json;

/// Scripted provider: each `chat()` call replays the next scripted round of
/// `StreamChunk`s, then closes the channel (channel close = end of stream).
/// RECORDS the `&[Message]` slice it was called with, per call index, so
/// tests can assert exactly what the (possibly middleware-rewritten) request
/// contained — mirrors `CapturingScriptedProvider` in `middleware.rs`.
struct RecordingScriptedProvider {
    rounds: Vec<Vec<StreamChunk>>,
    calls: AtomicU32,
    recorded: Arc<Mutex<Vec<Vec<Message>>>>,
}

impl RecordingScriptedProvider {
    fn new(rounds: Vec<Vec<StreamChunk>>) -> Self {
        Self {
            rounds,
            calls: AtomicU32::new(0),
            recorded: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn call_count(&self) -> u32 {
        self.calls.load(Ordering::SeqCst)
    }

    fn recorded(&self) -> Vec<Vec<Message>> {
        self.recorded.lock().unwrap().clone()
    }
}

#[async_trait]
impl Provider for RecordingScriptedProvider {
    async fn chat(
        &self,
        messages: &[Message],
        _tools: &[ToolDefinition],
        _model: &str,
        _workspace_root: &Path,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError> {
        let index = self.calls.fetch_add(1, Ordering::SeqCst) as usize;
        self.recorded.lock().unwrap().push(messages.to_vec());

        let round = self.rounds.get(index).cloned().unwrap_or_default();
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        tokio::spawn(async move {
            for chunk in round {
                let _ = tx.send(chunk).await;
            }
        });
        Ok(rx)
    }
}

/// Scripted provider for the retry path: the FIRST `chat()` call fails with
/// `RateLimited { retry_after_ms: 1 }`, every later call succeeds with a
/// text round. Records the request slice per call so the unmodified-request
/// retry contract can be asserted.
struct RateLimitOnceProvider {
    calls: AtomicU32,
    recorded: Arc<Mutex<Vec<Vec<Message>>>>,
}

impl RateLimitOnceProvider {
    fn new() -> Self {
        Self {
            calls: AtomicU32::new(0),
            recorded: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn call_count(&self) -> u32 {
        self.calls.load(Ordering::SeqCst)
    }

    fn recorded(&self) -> Vec<Vec<Message>> {
        self.recorded.lock().unwrap().clone()
    }
}

#[async_trait]
impl Provider for RateLimitOnceProvider {
    async fn chat(
        &self,
        messages: &[Message],
        _tools: &[ToolDefinition],
        _model: &str,
        _workspace_root: &Path,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError> {
        let index = self.calls.fetch_add(1, Ordering::SeqCst) as usize;
        self.recorded.lock().unwrap().push(messages.to_vec());

        if index == 0 {
            return Err(ProviderError::RateLimited { retry_after_ms: 1 });
        }
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        tokio::spawn(async move {
            let _ = tx.send(StreamChunk::TextDelta("done".into())).await;
            let _ = tx
                .send(StreamChunk::Finish {
                    reason: "stop".into(),
                })
                .await;
        });
        Ok(rx)
    }
}

/// Agent under test. BypassPermissions so any tool calls flow through the
/// pipeline without interactive approval (mirrors the recipe in
/// `middleware.rs` / `overflow_recovery.rs`).
fn test_agent(
    provider: Arc<dyn Provider>,
    event_tx: tokio::sync::mpsc::Sender<AgentEvent>,
) -> AgentLoop {
    AgentLoop::new(
        provider,
        ToolRegistry::new(),
        ApprovalPolicy::new(PermissionConfig {
            mode: PermissionMode::BypassPermissions,
            ..Default::default()
        }),
        "test-model".into(),
        event_tx,
        10, // max_turns (per-turn step budget)
        16, // max_tool_calls_per_turn
        0,  // checkpoint_interval
        None,
    )
}

/// Drain every event buffered after `run_turn` returns.
fn drain_events(mut rx: tokio::sync::mpsc::Receiver<AgentEvent>) -> Vec<AgentEvent> {
    let mut events = Vec::new();
    while let Ok(e) = rx.try_recv() {
        events.push(e);
    }
    events
}

/// Plain-text summaries of every message, in order.
fn texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .map(|m| m.content.to_summary_text())
        .collect()
}

fn read_call(id: &str) -> MessageToolCall {
    MessageToolCall {
        id: id.into(),
        name: "read_file".into(),
        arguments: json!({"path": "a.rs"}),
    }
}

/// Compactible history for the smart-compaction path: system + one OLD
/// `read_file` tool group with a big output (compactible tool, no
/// must-keep markers in the output) + a long recent tail so the tool group
/// falls outside the `RECENT_GROUPS_KEEP_FULL = 8` window and the plan
/// actually shrinks (`tokens_after < tokens_before`).
fn compactible_history() -> Vec<Message> {
    let mut messages = vec![Message::system("sys")];
    messages.push(Message::assistant_with_tool_calls(
        "",
        vec![read_call("c1")],
    ));
    messages.push(Message::tool("c1", "x".repeat(2_000)));
    for i in 0..10 {
        messages.push(Message::user(format!("u{i}")));
        messages.push(Message::assistant(format!("a{i}")));
    }
    messages
}

/// One usage-heavy text round: a huge cumulative input-token report (the
/// fold that materializes the session spend) followed by a final answer.
fn expensive_text_round() -> Vec<StreamChunk> {
    vec![
        StreamChunk::Usage {
            input_tokens: 1_000_000, // $3.00 at Sonnet-class rates
            output_tokens: 0,
            cache_creation_tokens: 0,
            cache_read_tokens: 0,
        },
        StreamChunk::TextDelta("ok".into()),
        StreamChunk::Finish {
            reason: "stop".into(),
        },
    ]
}

/// The cost-guard error prefix both the two-turn trip and the resume
/// re-seed trip must surface (verbatim from `CostGuardMiddleware`).
const BUDGET_ERROR_PREFIX: &str = "estimated session cost budget exhausted";

// ---------------------------------------------------------------------------
// W1 — wiring pin: smart compaction On through a real run_turn
// ---------------------------------------------------------------------------

/// W1 — wiring pin: `AgentLoop` + `CompactionMiddleware::new(On)` pushed.
/// With compactible seeded history, `run_turn` succeeds and the provider
/// observes the COMPACTED view (old tool-group text absent, system + recent
/// tail + the turn prompt present), while the event stream carries the
/// legacy `ContextCompaction { phase: "completed" }` with a token decrease.
/// Fails if `step()` ever bypasses the chain (the inline driver block is
/// gone; the middleware owns the view shaping).
#[tokio::test]
async fn w1_compaction_middleware_shapes_provider_view_through_run_turn() {
    let provider = Arc::new(RecordingScriptedProvider::new(vec![vec![
        StreamChunk::TextDelta("done".into()),
        StreamChunk::Finish {
            reason: "stop".into(),
        },
    ]]));
    let (event_tx, event_rx) = tokio::sync::mpsc::channel(256);

    let mut agent = test_agent(Arc::clone(&provider) as Arc<dyn Provider>, event_tx);
    agent.messages = compactible_history();
    agent.push_middleware(Arc::new(CompactionMiddleware::new(SmartCompactionMode::On)));

    let result = agent
        .run_turn("final prompt", Path::new("."), &[])
        .await
        .expect("compaction-on turn succeeds");
    assert_eq!(result, "done");

    assert_eq!(
        provider.call_count(),
        1,
        "one step ⇒ exactly one provider call behind the chain"
    );
    let recorded = provider.recorded();
    assert_eq!(recorded.len(), 1);

    let view = texts(&recorded[0]);
    assert!(
        !view.iter().any(|t| t.contains(&"x".repeat(2_000))),
        "provider must observe the COMPACTED view — the full 2000-char old tool \
         output must be gone (got {view:?})"
    );
    assert!(
        view.iter().any(|t| t.contains("[truncated")),
        "the compacted tool result carries the truncation marker: {view:?}"
    );
    assert!(
        view.iter().any(|t| t == "sys"),
        "the system message survives the compaction: {view:?}"
    );
    assert!(
        view.iter().any(|t| t == "u9") && view.iter().any(|t| t == "a9"),
        "the recent tail survives the compaction: {view:?}"
    );
    assert!(
        view.iter().any(|t| t == "final prompt"),
        "the turn prompt is part of the request view: {view:?}"
    );

    // Legacy informational event on the stream, with the planned decrease.
    let events = drain_events(event_rx);
    let compactions: Vec<&AgentEvent> = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ContextCompaction { .. }))
        .collect();
    assert_eq!(
        compactions.len(),
        1,
        "exactly one per-step ContextCompaction for the single step: {events:?}"
    );
    match compactions[0] {
        AgentEvent::ContextCompaction {
            phase,
            tokens_before,
            tokens_after,
            dropped_groups,
            ..
        } => {
            assert_eq!(phase, "completed");
            let before = tokens_before.expect("completed reports tokens_before");
            let after = tokens_after.expect("completed reports tokens_after");
            assert!(
                after < before,
                "completed must report the token decrease: {before} -> {after}"
            );
            assert_eq!(
                dropped_groups.expect("completed keeps the group accounting"),
                1,
                "the old tool group is counted as dropped/truncated"
            );
        }
        other => panic!("unexpected event: {other:?}"),
    }

    // Canonical history untouched (view-only rewrite, design §1a).
    assert!(
        texts(&agent.messages)
            .iter()
            .any(|t| t.contains(&"x".repeat(2_000))),
        "canonical agent.messages must retain the full old tool output"
    );
}

// ---------------------------------------------------------------------------
// W2 — budget trip end-to-end (black-box, real usage fold)
// ---------------------------------------------------------------------------

/// W2 — budget trip end-to-end: `AgentLoop` + `CostGuardMiddleware::new(1.0)`.
/// Turn 1 reports a huge `StreamChunk::Usage` (fold → cost tracker →
/// driver-materialized `session_usage` on the next step). Turn 2 fails
/// BEFORE the provider is called with `ProviderError::Other` naming the
/// budget; P1 rollback leaves `agent.messages` at the post-turn-1 baseline
/// (the second user prompt is rolled back).
#[tokio::test]
async fn w2_budget_trip_end_to_end_guard_trips_before_second_call() {
    let provider = Arc::new(RecordingScriptedProvider::new(vec![expensive_text_round()]));
    let (event_tx, event_rx) = tokio::sync::mpsc::channel(256);

    let mut agent = test_agent(Arc::clone(&provider) as Arc<dyn Provider>, event_tx);
    agent.push_middleware(Arc::new(CostGuardMiddleware::new(1.0)));

    // Turn 1: under budget (tracker empty), the expensive usage folds in.
    let first = agent
        .run_turn("first", Path::new("."), &[])
        .await
        .expect("turn 1 is under budget and succeeds");
    assert_eq!(first, "ok");
    assert_eq!(
        agent.cost_tracker.input_tokens, 1_000_000,
        "the usage fold must accumulate the huge input-token report"
    );
    assert_eq!(provider.call_count(), 1, "turn 1 made exactly one call");

    // Turn 2: the accumulated spend trips the guard pre-call.
    let err = agent
        .run_turn("second", Path::new("."), &[])
        .await
        .expect_err("turn 2 must fail: budget exhausted");
    let message = err.to_string();
    assert!(
        message.contains(BUDGET_ERROR_PREFIX),
        "error must name the budget trip: {message}"
    );
    assert!(
        message.contains("cost_budget_usd"),
        "error must point at the config knob: {message}"
    );
    assert!(
        message.contains("~10×"),
        "error must carry the rate-estimate caveat: {message}"
    );

    assert_eq!(
        provider.call_count(),
        1,
        "the guard trips BEFORE the provider — turn 2 must add zero calls"
    );

    // P1 rollback: the failed turn's messages were truncated to the pre-turn
    // baseline (post-turn-1 state). The second user prompt is gone.
    assert_eq!(
        agent.messages,
        vec![Message::user("first"), Message::assistant("ok")],
        "failed turn must roll back to the post-turn-1 baseline: {:?}",
        agent.messages
    );

    // The failure surfaces through the existing error path: StepFailed in
    // the event stream carrying the budget message.
    let events = drain_events(event_rx);
    let step_failed = events.iter().find_map(|e| match e {
        AgentEvent::StepFailed { error, .. } => Some(error.clone()),
        _ => None,
    });
    assert!(
        step_failed
            .as_deref()
            .is_some_and(|e| e.contains(BUDGET_ERROR_PREFIX)),
        "StepFailed must surface the budget error: {step_failed:?}"
    );
}

// ---------------------------------------------------------------------------
// W3 — rate-limit retry end-to-end
// ---------------------------------------------------------------------------

/// W3 — rate-limit end-to-end: `AgentLoop` + `RetryMiddleware::new(2, 50)`.
/// The provider rejects the first `chat()` with
/// `RateLimited { retry_after_ms: 1 }` and succeeds after; `run_turn`
/// succeeds with EXACTLY 2 provider calls and IDENTICAL messages both times
/// (the unmodified-request retry contract, design §1b).
#[tokio::test]
async fn w3_rate_limited_turn_retries_with_identical_request() {
    let provider = Arc::new(RateLimitOnceProvider::new());
    let (event_tx, _event_rx) = tokio::sync::mpsc::channel(256);

    let mut agent = test_agent(Arc::clone(&provider) as Arc<dyn Provider>, event_tx);
    agent.push_middleware(Arc::new(RetryMiddleware::new(2, 50)));

    let result = agent
        .run_turn("go", Path::new("."), &[])
        .await
        .expect("a single RateLimited rejection must be retried to success");
    assert_eq!(result, "done");

    assert_eq!(
        provider.call_count(),
        2,
        "exactly 2 provider calls: 1 rejection + 1 retry"
    );
    let recorded = provider.recorded();
    assert_eq!(recorded.len(), 2, "one recorded messages slice per call");
    assert_eq!(
        recorded[0], recorded[1],
        "the retry must re-send the request UNMODIFIED"
    );
    assert_eq!(
        texts(&recorded[0]),
        vec!["go".to_string()],
        "the request view is the single user prompt"
    );
}

// ---------------------------------------------------------------------------
// W4 — resume re-seed (core-level essence; the supervisor-level path lives
//       in crates/runtime/tests/cost_guard_resume.rs)
// ---------------------------------------------------------------------------

/// W4 — resume re-seed, core-level essence: the design scopes the real
/// re-seed to replay-format sessions at the supervisor level
/// (`supervisor.rs::seed_cost_tracker_from_log` — the last cumulative
/// `CostUpdated` in the event log re-seeds a FRESH `AgentLoop`'s tracker on
/// resume). At core level that path is unreachable, so this pins the
/// contract's essence: a fresh agent whose tracker was populated by prior
/// spend (exactly what the log re-seed reproduces) still trips the guard on
/// its FIRST provider call — 0 provider calls, deterministic re-trip.
///
/// The honest supervisor-level integration test is
/// `crates/runtime/tests/cost_guard_resume.rs::w4_resume_re_seeds_cost_tracker_from_log`
/// (the runtime resume harness exists — `inbox.rs` / `phase_c.rs` patterns).
#[tokio::test]
async fn w4_reeseed_tracker_trips_guard_on_fresh_agent_first_call() {
    // Pre-resume session: turn 1 accumulates the spend.
    let provider_a = Arc::new(RecordingScriptedProvider::new(vec![expensive_text_round()]));
    let (event_tx_a, _event_rx_a) = tokio::sync::mpsc::channel(256);
    let mut agent_a = test_agent(Arc::clone(&provider_a) as Arc<dyn Provider>, event_tx_a);
    agent_a.push_middleware(Arc::new(CostGuardMiddleware::new(1.0)));
    let first = agent_a
        .run_turn("first", Path::new("."), &[])
        .await
        .expect("turn 1 succeeds");
    assert_eq!(first, "ok");
    assert_eq!(agent_a.cost_tracker.input_tokens, 1_000_000);

    // "Resume": a FRESH agent (new provider instance, same guard) whose cost
    // tracker is re-seeded from the prior session's spend — the state
    // `seed_cost_tracker_from_log` reproduces from the last CostUpdated.
    let provider_b = Arc::new(RecordingScriptedProvider::new(vec![expensive_text_round()]));
    let (event_tx_b, _event_rx_b) = tokio::sync::mpsc::channel(256);
    let mut agent_b = test_agent(Arc::clone(&provider_b) as Arc<dyn Provider>, event_tx_b);
    agent_b.push_middleware(Arc::new(CostGuardMiddleware::new(1.0)));
    agent_b.cost_tracker = agent_a.cost_tracker.clone();

    let err = agent_b
        .run_turn("after resume", Path::new("."), &[])
        .await
        .expect_err("a re-seeded tripped session must re-trip on its first provider call");
    assert!(
        err.to_string().contains(BUDGET_ERROR_PREFIX),
        "error must name the budget trip: {err}"
    );
    assert_eq!(
        provider_b.call_count(),
        0,
        "the re-seeded guard trips BEFORE the provider — zero calls"
    );
    assert!(
        agent_b.messages.is_empty(),
        "fresh-agent P1 rollback truncates to the pre-turn baseline (empty): {:?}",
        agent_b.messages
    );
}
