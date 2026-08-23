//! P1 — Turn/step layering + single inbox: RUNTIME wiring contract tests
//! (failing-first).
//!
//! Spec (authoritative): `docs/plans/p1-turn-step-design.md` §4 ("nca-runtime /
//! nca-cli wiring"). Core (`nca_core::agent_driver::InboxItem`,
//! `AgentLoop::inbox_sender`, `AgentLoop::set_turn_seq_start`) is DONE —
//! see `crates/core/tests/turn_driver.rs` for the driver-level contract tests.
//!
//! The RUNTIME wiring does NOT exist yet. This file encodes it and must fail
//! until the fixer implements:
//!   1. `Supervisor::inbox_sender(&self) -> mpsc::Sender<InboxItem>` —
//!      delegates to the agent's inbox (compile-red symbol).
//!   2. Resume turn-id seeding: on resume, scan the session's events.jsonl
//!      for max `TurnStarted.turn_id` and call `AgentLoop::set_turn_seq_start(n)`
//!      so the next `run_turn` emits `TurnStarted { turn_id: n + 1 }`.
//!
//! Failure modes by design:
//!   - `queue_while_turn_running` (Test A) fails to COMPILE (missing
//!     `Supervisor::inbox_sender`).
//!   - `resume_seeds_turn_ids` (Test B) compiles once that symbol lands and
//!     then fails at RUNTIME (turn_id 1 instead of 4) until resume seeding
//!     is implemented.
//!
//! Run: `cargo test -p nca-runtime --test inbox`
//!
//! Construction note: `Supervisor::create` builds a real provider via
//! `build_provider` (no mock seam), so tests construct the supervisor with a
//! deterministic offline config (dummy API key, bypass permissions, no context
//! API / auto-summarize), then swap in a scripted provider via
//! `agent_mut().replace_provider(...)`. No network, no fixed sleeps — the
//! only waits are bounded poll-retry loops (≤2s) plus a signal-gated mock
//! stream that makes the boundary-claim ordering deterministic.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use nca_common::config::{NcaConfig, PermissionMode};
use nca_common::event::{AgentEvent, EventEnvelope};
use nca_common::message::{Message, MessageContent, Role};
use nca_common::tool::{ToolCall, ToolDefinition};
use nca_core::agent_driver::InboxItem;
use nca_core::provider::{Provider, ProviderError, StreamChunk};
use nca_runtime::supervisor::{Supervisor, SupervisorConfig};
use serde_json::json;

// ---------------------------------------------------------------------------
// Test scaffolding
// ---------------------------------------------------------------------------

/// Deterministic offline config: provider construction succeeds with a dummy
/// key, unknown-tool calls flow through the pipeline (bypass permissions),
/// and no context-API fetch / auto-summarize / title-gen side effects can add
/// provider calls or touch the network during a turn.
fn offline_config() -> NcaConfig {
    let mut config = NcaConfig::default();
    config.provider.deepseek.api_key = Some("test-key".into());
    config.permissions.mode = PermissionMode::BypassPermissions;
    config.memory.context.auto_detect_context_window = false;
    config.memory.context.query_provider_models_api = false;
    config.memory.context.enable_auto_summarize = false;
    config
}

async fn create_supervisor(ws: &Path, session_id: Option<String>) -> Supervisor {
    Supervisor::create(SupervisorConfig {
        config: offline_config(),
        workspace_root: ws.to_path_buf(),
        safe_mode: true,
        interactive_approvals: false,
        session_id,
        approval_handler: None,
        orchestration_context: None,
        agent_name: None,
    })
    .await
    .expect("supervisor create must succeed with the offline config")
}

/// Scripted provider mirroring `CapturingScriptedProvider` from the core
/// turn_driver tests: each `chat()` call replays the next scripted round and
/// records the `&[Message]` slice it was called with, per call index.
///
/// One optional round (default round 0) is GATED: its stream stays open until
/// `release_gate()` fires. This is the deterministic substitute for a fixed
/// sleep — the turn is provably in flight (round-1 call started, stream not
/// yet delivered) at the moment the test injects the queued prompt, so the
/// step-boundary claim MUST observe it without any timing luck.
struct ScriptedProvider {
    rounds: Vec<Vec<StreamChunk>>,
    gate_round: Option<usize>,
    calls: AtomicU32,
    recorded: Arc<Mutex<Vec<Vec<Message>>>>,
    release: Arc<tokio::sync::Notify>,
}

impl ScriptedProvider {
    fn new(rounds: Vec<Vec<StreamChunk>>, gate_round: Option<usize>) -> Self {
        Self {
            rounds,
            gate_round,
            calls: AtomicU32::new(0),
            recorded: Arc::new(Mutex::new(Vec::new())),
            release: Arc::new(tokio::sync::Notify::new()),
        }
    }

    fn call_count(&self) -> u32 {
        self.calls.load(Ordering::SeqCst)
    }

    fn recorded(&self) -> Vec<Vec<Message>> {
        self.recorded.lock().unwrap().clone()
    }

    /// Release the gated round's stream (if any). Safe to call unconditionally:
    /// a `Notify` permit with no waiter is simply stored and consumed later.
    fn release_gate(&self) {
        self.release.notify_one();
    }
}

#[async_trait]
impl Provider for ScriptedProvider {
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
        let gated = self.gate_round == Some(index);
        let release = self.release.clone();
        tokio::spawn(async move {
            if gated {
                // Hold the stream open until the test injects the queued
                // prompt — the agent blocks in `stream.recv()` meanwhile.
                release.notified().await;
            }
            for chunk in round {
                let _ = tx.send(chunk).await;
            }
        });
        Ok(rx)
    }
}

/// Extract the plain-text content of every user message, in order.
fn user_texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter(|m| m.role == Role::User)
        .filter_map(|m| match &m.content {
            MessageContent::Text(t) => Some(t.clone()),
            _ => None,
        })
        .collect()
}

fn tool_use_round(id: &str) -> Vec<StreamChunk> {
    vec![StreamChunk::ToolUse(ToolCall {
        id: id.into(),
        name: "echo".into(),
        input: json!({}),
    })]
}

fn text_round(text: &str) -> Vec<StreamChunk> {
    vec![
        StreamChunk::TextDelta(text.into()),
        StreamChunk::Finish {
            reason: "stop".into(),
        },
    ]
}

// ---------------------------------------------------------------------------
// Test A (T7 in the plan test matrix) — enqueue while a turn is in flight;
// the queued prompt is claimed at the step boundary and incorporated into the
// running turn, in order.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn queue_while_turn_running() {
    let ws = tempfile::tempdir().expect("tempdir");

    // Round 1 is gated: the turn provably reaches its first provider call and
    // then parks on the open stream. Round 2 finishes the turn.
    let provider = Arc::new(ScriptedProvider::new(
        vec![tool_use_round("t1"), text_round("done")],
        Some(0),
    ));

    let mut sup = create_supervisor(ws.path(), None).await;
    sup.agent_mut().replace_provider(provider.clone());
    // Skip title generation (it would consume an extra provider round).
    sup.set_session_title(Some("p1-inbox-test".into()));

    // MISSING API (compile-red until the fixer lands it): must delegate to
    // the agent's inbox and return the cloneable sender handle.
    let inbox: tokio::sync::mpsc::Sender<InboxItem> = sup.inbox_sender();

    // Injector task: watch the turn from outside. As soon as round 1 is in
    // flight (call_count == 1), queue a user prompt through the supervisor's
    // inbox, then release the gated stream so the agent reaches the next
    // step boundary. Bounded poll-retry, ≤2s.
    let injector = {
        let provider = provider.clone();
        tokio::spawn(async move {
            let deadline = Instant::now() + Duration::from_secs(2);
            while provider.call_count() < 1 && Instant::now() < deadline {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            assert_eq!(
                provider.call_count(),
                1,
                "round 1 must be in flight before the queued prompt is sent"
            );
            inbox
                .try_send(InboxItem::UserPrompt {
                    text: "queued".into(),
                })
                .expect("capacity-16 inbox must accept the queued prompt");
            provider.release_gate();
        })
    };

    let result = sup
        .run_turn("start the work")
        .await
        .expect("scripted 2-step turn succeeds");
    injector.await.expect("injector task");

    assert_eq!(result, "done", "final step text is the turn result");
    assert_eq!(
        provider.call_count(),
        2,
        "exactly 2 scripted rounds (title gen was pre-set and skipped)"
    );

    let recorded = provider.recorded();
    assert_eq!(recorded.len(), 2, "one recorded messages slice per call");

    let call1_users = user_texts(&recorded[0]);
    assert!(
        !call1_users.iter().any(|t| t == "queued"),
        "round 1 predates the injection — the prompt must not be present yet: {call1_users:?}"
    );

    let call2_users = user_texts(&recorded[1]);
    let pos_start = call2_users
        .iter()
        .position(|t| t == "start the work")
        .expect("initial user prompt retained in round 2");
    let pos_queued = call2_users
        .iter()
        .position(|t| t == "queued")
        .expect("queued prompt must be claimed into the round-2 request");
    assert!(
        pos_start < pos_queued,
        "queued prompt must be appended in arrival order after the initial prompt: {call2_users:?}"
    );
}

// ---------------------------------------------------------------------------
// Test B — resume seeds turn ids: after resuming a session whose event log
// contains TurnStarted{turn_id: 3}, the next run_turn emits
// TurnStarted{turn_id: 4}.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn resume_seeds_turn_ids() {
    let ws = tempfile::tempdir().expect("tempdir");
    let sid = "inbox-resume-seed-test";

    // 1. Create a session so the persisted `{sid}.json` state exists (the
    //    resume path loads it before rebuilding the supervisor).
    let sup = create_supervisor(ws.path(), Some(sid.into())).await;
    drop(sup);

    // 2. Persist an event log mirroring the real on-disk shape exactly:
    //    one `EventEnvelope` JSON line per line (see
    //    `session_utils::spawn_event_fanout` → `EventEnvelope::new`).
    //    Three past turns → the seeding scan must take the MAX turn_id (3).
    let log_path: PathBuf = ws
        .path()
        .join(".nca/sessions")
        .join(format!("{sid}.events.jsonl"));
    let mut lines = String::new();
    for turn_id in [1u64, 2, 3] {
        let envelope = EventEnvelope::new(turn_id, AgentEvent::TurnStarted { turn_id });
        lines.push_str(&serde_json::to_string(&envelope).expect("envelope json"));
        lines.push('\n');
    }
    std::fs::write(&log_path, lines).expect("write events.jsonl");
    let on_disk = std::fs::read_to_string(&log_path).expect("read back");
    assert!(
        on_disk.contains(r#""TurnStarted""#) && on_disk.contains(r#""turn_id":3"#),
        "fixture must contain a TurnStarted line with turn_id 3: {on_disk}"
    );

    // 3. Resume via the supervisor resume path.
    let mut sup = Supervisor::resume(offline_config(), ws.path(), true, false, sid, None)
        .await
        .expect("resume must succeed with the offline config");

    // 4. Swap in a trivial scripted provider (single text round).
    let provider = Arc::new(ScriptedProvider::new(vec![text_round("resumed")], None));
    sup.agent_mut().replace_provider(provider.clone());
    sup.set_session_title(Some("p1-inbox-test".into()));

    // 5. Capture events and run one turn. The collector stops at the turn's
    //    `TurnCompleted`.
    let mut handle = sup.take_handle();
    let mut event_rx = handle.take_event_rx().expect("event rx");
    let collector = tokio::spawn(async move {
        let mut events = Vec::new();
        while let Some(e) = event_rx.recv().await {
            let done = matches!(e, AgentEvent::TurnCompleted { .. });
            events.push(e);
            if done {
                break;
            }
        }
        events
    });

    let result = sup.run_turn("resume me").await.expect("turn succeeds");
    assert_eq!(result, "resumed");

    let events = collector.await.expect("event collector task");
    let started = events
        .iter()
        .find_map(|e| match e {
            AgentEvent::TurnStarted { turn_id } => Some(*turn_id),
            _ => None,
        })
        .expect("the turn must emit TurnStarted");
    assert_eq!(
        started, 4,
        "resume must seed turn ids from the event log (max TurnStarted=3 → next turn = 4); events: {events:?}"
    );
}
