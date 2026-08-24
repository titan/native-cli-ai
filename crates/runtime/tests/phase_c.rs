//! P2 Phase C — single-writer session snapshots + graceful writer close
//! (RUNTIME tests, T22–T28).
//!
//! Spec (authoritative): `docs/plans/p2-phase-c-design.md` §2–§5 + "Test
//! matrix". Harness mirrors `crates/runtime/tests/turn_commit.rs` /
//! `replay_resume.rs`: a real `Supervisor` over a tempdir workspace, a
//! scripted offline `Provider` swapped in after `create()`, and a fanout
//! wired exactly like service.rs. No network, no fixed sleeps.
//!
//! Post-Phase-C discipline under test:
//! - T22: `run_turn` never rewrites the session json (create-time snapshot
//!   is the mid-session truth); the event log carries the turn.
//! - T23: `finish()` is the sole mid-life persistence — full history +
//!   display meta (title, lineage) land in the json.
//! - T24: lineage folds from `ChildSessionSpawned` envelopes at resume;
//!   the crashed-parent case (stale json, log wins) recovers lineage.
//! - T25: graceful close drains the fanout and commits `SessionEnded` —
//!   no torn tail.
//! - T26: buffered events are NOT dropped at close (the drain the old
//!   `abort()` cut).
//! - T27: the sub-agent consumer routes child spawn events through the
//!   parent's event channel into the parent's log.
//! - T28: `switch_to` round-trip replaces history wholesale — message
//!   counts stay exact, no duplication (Supervisor-level, not a TUI test).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use async_trait::async_trait;
use nca_common::config::{NcaConfig, PermissionMode};
use nca_common::event::{AgentEvent, BusyState, EndReason, EventEnvelope};
use nca_common::message::{Message, Role};
use nca_common::session::{SessionMeta, SessionState, SessionStatus};
use nca_common::tool::ToolDefinition;
use nca_core::provider::{Provider, ProviderError, StreamChunk};
use nca_core::tools::spawn_subagent::SpawnRequest;
use nca_core::workspace_fs::{RealFs, WorkspaceFs};
use nca_runtime::session_store::{SessionStore, read_event_log};
use nca_runtime::session_utils::spawn_event_fanout;
use nca_runtime::subagent::spawn_subagent_consumer;
use nca_runtime::supervisor::{Supervisor, SupervisorConfig};
use tokio::sync::{mpsc, oneshot};

// ---------------------------------------------------------------------------
// Shared scaffolding (mirrors turn_commit.rs / replay_resume.rs)
// ---------------------------------------------------------------------------

fn offline_config() -> NcaConfig {
    let mut config = NcaConfig::default();
    config.provider.deepseek.api_key = Some("test-key".into());
    config.permissions.mode = PermissionMode::BypassPermissions;
    config.memory.context.auto_detect_context_window = false;
    config.memory.context.query_provider_models_api = false;
    config.memory.context.enable_auto_summarize = false;
    config
}

fn sessions_dir(ws: &Path) -> PathBuf {
    ws.join(".nca").join("sessions")
}

async fn create_sup(ws: &Path, session_id: &str) -> Supervisor {
    Supervisor::create(SupervisorConfig {
        config: offline_config(),
        workspace_root: ws.to_path_buf(),
        safe_mode: true,
        interactive_approvals: false,
        session_id: Some(session_id.into()),
        approval_handler: None,
        orchestration_context: None,
        agent_name: None,
    })
    .await
    .expect("supervisor create must succeed with the offline config")
}

/// Single-round text provider — one TextDelta + Finish per chat() call.
struct OneShotProvider;

impl OneShotProvider {
    fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}

#[async_trait]
impl Provider for OneShotProvider {
    async fn chat(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        _model: &str,
        _workspace_root: &Path,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError> {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        tokio::spawn(async move {
            let _ = tx.send(StreamChunk::TextDelta("answer".into())).await;
            let _ = tx
                .send(StreamChunk::Finish {
                    reason: "stop".into(),
                })
                .await;
        });
        Ok(rx)
    }
}

/// Mirrors service.rs wiring: take the handle (event_rx + commit sender),
/// spawn the fanout, and hand back the log path + JoinHandle so tests can
/// drain on close. The commit sender marks the turn-commit barrier wired.
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

/// Graceful close: dropping the supervisor closes every event-channel sender
/// (agent + tool clones all live inside it), so the fanout drains its buffer,
/// runs the final commit-on-close, and exits. Bounded wait, like production.
async fn drain_fanout(sup: Supervisor, fanout: tokio::task::JoinHandle<()>) {
    drop(sup);
    tokio::time::timeout(Duration::from_secs(5), fanout)
        .await
        .expect("fanout must drain and exit within 5s of the sender drop")
        .expect("fanout task must complete without panicking");
}

// ---------------------------------------------------------------------------
// T22 — run_turn leaves the json untouched; the log carries the turn.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn t22_run_turn_leaves_json_untouched_log_grows() {
    let ws = tempfile::tempdir().expect("tempdir");
    let mut sup = create_sup(ws.path(), "t22-single-writer").await;
    sup.agent_mut().replace_provider(OneShotProvider::new());
    sup.set_session_title(Some("t22".into()));

    // Snapshot the json right after create() — the only mid-session truth.
    let json_path = sessions_dir(ws.path()).join("t22-single-writer.json");
    let before = std::fs::read(&json_path).expect("json must exist right after create");

    let (log_path, fanout) = wire_fanout(&mut sup).await;
    let log_before = read_event_log(&log_path).len();

    let out = sup.run_turn("hello").await.expect("turn succeeds");
    assert_eq!(out, "answer");

    let after = std::fs::read(&json_path).expect("json still exists");
    assert!(
        before == after,
        "run_turn must NOT rewrite the session json (single-writer discipline, \
         P2 Phase C §2): the create-time snapshot is the mid-session truth \
         (json bytes before={} after={})",
        before.len(),
        after.len()
    );
    let log_after = read_event_log(&log_path);
    assert!(
        log_after.len() > log_before,
        "the event log must grow across the turn (was {log_before}, now {})",
        log_after.len()
    );
    drain_fanout(sup, fanout).await;
}

// ---------------------------------------------------------------------------
// T23 — finish() persists full history + display meta (title, lineage).
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn t23_finish_persists_full_history_and_meta() {
    let ws = tempfile::tempdir().expect("tempdir");
    let sid = "t23-finish-persist";
    let mut sup = create_sup(ws.path(), sid).await;
    sup.agent_mut().replace_provider(OneShotProvider::new());
    sup.set_session_title(Some("t23 session title".into()));
    sup.set_parent(
        "parent-0".into(),
        Some("inherited summary".into()),
        Some("t23 test".into()),
    );
    sup.add_child("child-1".into());

    let (log_path, fanout) = wire_fanout(&mut sup).await;
    sup.run_turn("hello").await.expect("turn succeeds");
    sup.finish(EndReason::Completed).await;

    // Drain so the SessionEnded from finish() is durable before reading.
    drain_fanout(sup, fanout).await;

    let store = SessionStore::new(sessions_dir(ws.path()));
    let state = store
        .load(sid)
        .await
        .expect("json must reload after finish()");
    assert_eq!(state.meta.status, SessionStatus::Completed);
    assert_eq!(
        state.meta.session_title.as_deref(),
        Some("t23 session title"),
        "finish() must persist the session title"
    );
    assert_eq!(
        state.meta.parent_session_id.as_deref(),
        Some("parent-0"),
        "finish() must persist lineage (parent id)"
    );
    assert_eq!(
        state.meta.child_session_ids,
        vec!["child-1".to_string()],
        "finish() must persist lineage (child ids)"
    );
    let roles: Vec<Role> = state.messages.iter().map(|m| m.role.clone()).collect();
    assert_eq!(
        roles,
        vec![Role::System, Role::User, Role::Assistant],
        "finish() must persist the full turn history"
    );
    assert_eq!(state.messages[1], Message::user("hello"));
    assert_eq!(state.messages[2].content.event_preview(), "answer");

    let envelopes = read_event_log(&log_path);
    assert!(
        envelopes.iter().any(|e| matches!(
            e.event,
            AgentEvent::SessionEnded {
                reason: EndReason::Completed
            }
        )),
        "finish()'s SessionEnded(Completed) must be in the event log"
    );
}

// ---------------------------------------------------------------------------
// T24 — resume folds child lineage from the event log (§3).
// ---------------------------------------------------------------------------

/// Write a well-formed session json with the given child lineage and a
/// parent_session_id set (so the reloaded snapshot is a realistic parent).
fn write_session_json(ws: &Path, session_id: &str, child_ids: Vec<String>) {
    std::fs::create_dir_all(sessions_dir(ws)).expect("create sessions dir");
    let now = chrono::Utc::now();
    let state = SessionState {
        meta: SessionMeta {
            id: session_id.to_string(),
            created_at: now,
            updated_at: now,
            workspace: ws.to_path_buf(),
            model: "deepseek-chat".into(),
            status: SessionStatus::Completed,
            pid: None,
            socket_path: None,
            worktree_path: None,
            branch: None,
            base_branch: None,
            parent_session_id: Some("parent-0".into()),
            child_session_ids: child_ids,
            inherited_summary: None,
            spawn_reason: None,
            session_summary: None,
            session_title: None,
            orchestration: None,
        },
        messages: vec![
            Message::user("persisted question"),
            Message::assistant("persisted answer"),
        ],
        total_input_tokens: 0,
        total_output_tokens: 0,
        estimated_cost_usd: 0.0,
    };
    std::fs::write(
        sessions_dir(ws).join(format!("{session_id}.json")),
        serde_json::to_string_pretty(&state).expect("serialize session"),
    )
    .expect("write session json");
}

/// Append envelope-wrapped ChildSessionSpawned events for each child id.
fn write_spawn_log(ws: &Path, session_id: &str, children: &[&str]) {
    std::fs::create_dir_all(sessions_dir(ws)).expect("create sessions dir");
    let mut lines = String::new();
    for (i, child) in children.iter().enumerate() {
        let envelope = EventEnvelope::new(
            (i + 1) as u64,
            AgentEvent::ChildSessionSpawned {
                parent_session_id: session_id.to_string(),
                child_session_id: (*child).to_string(),
                task: "child task".into(),
                workspace: PathBuf::from("/ws"),
                branch: None,
            },
        );
        lines.push_str(&serde_json::to_string(&envelope).expect("serialize envelope"));
        lines.push('\n');
    }
    std::fs::write(
        sessions_dir(ws).join(format!("{session_id}.events.jsonl")),
        lines,
    )
    .expect("write events.jsonl");
}

/// Crashed-parent case: the json lineage is stale (empty — the parent never
/// reached a finish() save after spawning), but the log carries the
/// ChildSessionSpawned envelopes. Resume must fold them into meta.
#[tokio::test(flavor = "multi_thread")]
async fn t24_resume_folds_child_lineage_crashed_parent() {
    let ws = tempfile::tempdir().expect("tempdir");
    let sid = "t24-crashed-parent";
    write_session_json(ws.path(), sid, Vec::new());
    write_spawn_log(ws.path(), sid, &["child-1", "child-2"]);

    let sup = Supervisor::resume(offline_config(), ws.path(), true, false, sid, None)
        .await
        .expect("resume must succeed with a stale json + spawned events in the log");

    // In-memory lineage is folded...
    assert_eq!(
        sup.snapshot().child_session_ids,
        vec!["child-1".to_string(), "child-2".to_string()],
        "resume must fold log-derived child ids into meta (crashed parent)"
    );
    // ...and the resume save persisted it immediately (design §3: fold
    // happens BEFORE the resume save, so a crash right after resume keeps it).
    let state = SessionStore::new(sessions_dir(ws.path()))
        .load(sid)
        .await
        .expect("json reload after resume");
    assert_eq!(
        state.meta.child_session_ids,
        vec!["child-1".to_string(), "child-2".to_string()],
        "the folded lineage must be durable in the json right after resume"
    );
}

/// Union case: json holds ["a"], the log repeats "a" (dupe) and adds "b".
/// The fold must be json-first, log-appended, deduplicated.
#[tokio::test(flavor = "multi_thread")]
async fn t24_resume_folds_child_lineage_union_deduped() {
    let ws = tempfile::tempdir().expect("tempdir");
    let sid = "t24-union";
    write_session_json(ws.path(), sid, vec!["a".to_string()]);
    write_spawn_log(ws.path(), sid, &["a", "b"]);

    let sup = Supervisor::resume(offline_config(), ws.path(), true, false, sid, None)
        .await
        .expect("resume must succeed");
    assert_eq!(
        sup.snapshot().child_session_ids,
        vec!["a".to_string(), "b".to_string()],
        "json ids first, log ids appended, no duplicates"
    );
}

// ---------------------------------------------------------------------------
// T25 — graceful close: drop the supervisor, fanout drains + commits,
//       SessionEnded durable, no torn tail.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn t25_graceful_close_drains_and_commits() {
    let ws = tempfile::tempdir().expect("tempdir");
    let mut sup = create_sup(ws.path(), "t25-graceful-close").await;
    sup.agent_mut().replace_provider(OneShotProvider::new());
    sup.set_session_title(Some("t25".into()));

    let (log_path, fanout) = wire_fanout(&mut sup).await;
    sup.run_turn("hello").await.expect("turn succeeds");
    sup.finish(EndReason::Completed).await;

    // Close the event channel: the supervisor owns every sender (agent +
    // tool clones); the fixture holds no clones. The fanout must drain the
    // buffered SessionEnded and exit — not stall and not drop events.
    drain_fanout(sup, fanout).await;

    let raw = std::fs::read_to_string(&log_path).expect("log readable");
    let line_count = raw.lines().filter(|l| !l.trim().is_empty()).count();
    let envelopes = read_event_log(&log_path);
    assert!(
        !envelopes.is_empty(),
        "log must not be empty after a turn + finish"
    );
    assert_eq!(
        envelopes.len(),
        line_count,
        "every log line must parse as an envelope: no torn tail"
    );
    for pair in envelopes.windows(2) {
        assert!(
            pair[0].id < pair[1].id,
            "envelope ids strictly increasing: {:?}",
            envelopes.iter().map(|e| e.id).collect::<Vec<_>>()
        );
    }
    assert!(
        matches!(
            envelopes.last().map(|e| &e.event),
            Some(AgentEvent::SessionEnded { .. })
        ),
        "SessionEnded must be the final durable event after graceful close"
    );
}

// ---------------------------------------------------------------------------
// T26 — buffered events are not dropped at close: N events sent, sender
//       dropped mid-drain, all N land in the log in order.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn t26_buffered_events_not_dropped_at_close() {
    let ws = tempfile::tempdir().expect("tempdir");
    let log_path = sessions_dir(ws.path()).join("t26-drain.events.jsonl");

    // Raw channel + fanout, no supervisor — pins the fanout's drain logic.
    let (tx, rx) = mpsc::channel(64);
    let fanout = spawn_event_fanout(rx, log_path.clone(), None, None, None, None);

    let events = vec![
        AgentEvent::TurnStarted { turn_id: 1 },
        AgentEvent::MessageRecorded {
            message: Message::user("drain question"),
        },
        AgentEvent::MessageReceived {
            role: "user".into(),
            content: "drain question".into(),
            steering: false,
        },
        AgentEvent::BusyStateChanged {
            state: BusyState::Streaming,
        },
        AgentEvent::ContextStatsUpdated {
            estimated_tokens: 100,
            context_window: 8000,
            usage_percent: 1,
        },
        AgentEvent::Checkpoint {
            phase: "test".into(),
            detail: "mid-drain checkpoint".into(),
            turn: 1,
        },
        AgentEvent::TurnCompleted {
            turn_id: 1,
            duration_ms: 5,
        },
        AgentEvent::SessionEnded {
            reason: EndReason::Completed,
        },
    ];
    // Push everything WITHOUT awaiting the fanout's processing, then drop the
    // sender — everything still buffered must be drained on close.
    for event in &events {
        tx.try_send(event.clone())
            .expect("64-capacity channel must hold the whole batch");
    }
    drop(tx);

    tokio::time::timeout(Duration::from_secs(5), fanout)
        .await
        .expect("fanout must drain and exit after the sender drop")
        .expect("fanout task must complete without panicking");

    let envelopes = read_event_log(&log_path);
    assert_eq!(
        envelopes.len(),
        events.len(),
        "ALL buffered events must be drained to the log on close (the old abort() cut here)"
    );
    for (i, (env, ev)) in envelopes.iter().zip(&events).enumerate() {
        assert_eq!(
            serde_json::to_string(&env.event).expect("serialize envelope event"),
            serde_json::to_string(ev).expect("serialize sent event"),
            "envelope {i} must match the sent event, in order"
        );
    }
    for pair in envelopes.windows(2) {
        assert!(
            pair[0].id < pair[1].id,
            "envelope ids strictly increasing: {:?}",
            envelopes.iter().map(|e| e.id).collect::<Vec<_>>()
        );
    }
}

// ---------------------------------------------------------------------------
// T27 — one-shot consumer wiring: child spawn events reach the parent's
//       event channel and land in the parent's log via its fanout.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn t27_consumer_wiring_routes_child_spawn_into_parent_log() {
    let ws = tempfile::tempdir().expect("tempdir");
    let mut config = offline_config();
    // Point the child's provider at a closed localhost port: create()
    // succeeds (no network at build time) and the first chat() fails fast
    // with connection refused — no real API key, no network, no hang.
    config.provider.deepseek.base_url = "http://127.0.0.1:1".into();

    let (parent_tx, parent_rx) = mpsc::channel(256);
    let log_path = sessions_dir(ws.path()).join("t27-parent.events.jsonl");
    let parent_fanout = spawn_event_fanout(parent_rx, log_path.clone(), None, None, None, None);

    let (spawn_tx, spawn_rx) = mpsc::channel(4);
    let (reply_tx, reply_rx) = oneshot::channel();
    let parent_fs: Arc<dyn WorkspaceFs> = Arc::new(RealFs::new(ws.path().to_path_buf()));

    let consumer = spawn_subagent_consumer(
        spawn_rx,
        "t27-parent".into(),
        ws.path().to_path_buf(),
        config,
        vec![Message::user("parent context")],
        Some(parent_tx),
        parent_fs,
    );

    spawn_tx
        .send(SpawnRequest {
            task: "t27 child task".into(),
            focus_files: Vec::new(),
            use_worktree: false,
            provider_override: None,
            model_override: None,
            specialist: None,
            reply: reply_tx,
        })
        .await
        .expect("spawn request accepted");

    let response = tokio::time::timeout(Duration::from_secs(30), reply_rx)
        .await
        .expect("child reply must arrive (bounded)")
        .expect("reply channel must not be dropped");
    assert_eq!(
        response.status, "error",
        "offline child must fail fast against the closed port, not hang or succeed"
    );

    // Close the consumer loop, then await both tasks so every event the
    // child emitted through the parent channel is durable before reading.
    drop(spawn_tx);
    tokio::time::timeout(Duration::from_secs(5), consumer)
        .await
        .expect("consumer loop must exit after the spawn channel closes")
        .expect("consumer task must complete without panicking");
    tokio::time::timeout(Duration::from_secs(5), parent_fanout)
        .await
        .expect("parent fanout must drain after consumer + request task end")
        .expect("parent fanout task must complete without panicking");

    let envelopes = read_event_log(&log_path);
    let spawned = envelopes
        .iter()
        .filter(|e| matches!(e.event, AgentEvent::ChildSessionSpawned { .. }))
        .count();
    assert_eq!(
        spawned, 1,
        "exactly one ChildSessionSpawned must reach the parent log \
         (one-shot wiring: event_tx must be Some, not None)"
    );
    // Robust to either terminal outcome: assert presence, and that the spawn
    // precedes whichever completion/error the child produced (single FIFO
    // channel: spawn is sent before the child turn even starts).
    let spawned_idx = envelopes
        .iter()
        .position(|e| matches!(e.event, AgentEvent::ChildSessionSpawned { .. }))
        .expect("spawned event present");
    let terminal_idx = envelopes
        .iter()
        .rposition(|e| {
            matches!(
                e.event,
                AgentEvent::ChildSessionCompleted { .. } | AgentEvent::Error { .. }
            )
        })
        .expect("a terminal event (ChildSessionCompleted or Error) must follow the spawn");
    assert!(
        spawned_idx < terminal_idx,
        "spawned event must be logged before the terminal event (FIFO channel)"
    );
}

// ---------------------------------------------------------------------------
// T28 — switch_to round-trip replace invariant: message counts exact on
//       both sessions, no duplication across A → B → A (Supervisor-level).
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn t28_switch_round_trip_replaces_history_without_duplication() {
    let ws = tempfile::tempdir().expect("tempdir");

    // Session A: 2 turns → system + 2 × (user + assistant).
    let sid_a = "t28-session-a";
    let mut sup_a = create_sup(ws.path(), sid_a).await;
    sup_a.agent_mut().replace_provider(OneShotProvider::new());
    sup_a.set_session_title(Some("t28-a".into()));
    let (_, fanout_a) = wire_fanout(&mut sup_a).await;
    sup_a.run_turn("a-first").await.expect("a turn 1");
    sup_a.run_turn("a-second").await.expect("a turn 2");
    let expected_a = sup_a.agent().messages.len();
    sup_a.finish(EndReason::Completed).await;
    drain_fanout(sup_a, fanout_a).await;

    // Session B: 1 turn → system + 1 × (user + assistant).
    let sid_b = "t28-session-b";
    let mut sup_b = create_sup(ws.path(), sid_b).await;
    sup_b.agent_mut().replace_provider(OneShotProvider::new());
    sup_b.set_session_title(Some("t28-b".into()));
    let (_, fanout_b) = wire_fanout(&mut sup_b).await;
    sup_b.run_turn("b-only").await.expect("b turn 1");
    let expected_b = sup_b.agent().messages.len();
    sup_b.finish(EndReason::Completed).await;
    drain_fanout(sup_b, fanout_b).await;

    // Round-trip A → B → A via resume (the core of runner.rs `switch_to`).
    let sup_a2 = Supervisor::resume(offline_config(), ws.path(), true, false, sid_a, None)
        .await
        .expect("resume A");
    assert_eq!(
        sup_a2.agent().messages.len(),
        expected_a,
        "resume A must reproduce A's exact history — no duplication"
    );
    let roles_a: Vec<Role> = sup_a2
        .agent()
        .messages
        .iter()
        .map(|m| m.role.clone())
        .collect();
    assert_eq!(
        roles_a,
        vec![
            Role::System,
            Role::User,
            Role::Assistant,
            Role::User,
            Role::Assistant
        ],
        "resume REPLACES history wholesale (P2 Phase C §5) — exact system + 2-turn shape"
    );
    assert_eq!(
        sup_a2
            .agent()
            .messages
            .iter()
            .filter(|m| **m == Message::user("a-first"))
            .count(),
        1,
        "a-first appears exactly once after resume"
    );
    assert_eq!(
        sup_a2
            .agent()
            .messages
            .iter()
            .filter(|m| **m == Message::user("a-second"))
            .count(),
        1,
        "a-second appears exactly once after resume"
    );

    let sup_b2 = Supervisor::resume(offline_config(), ws.path(), true, false, sid_b, None)
        .await
        .expect("resume B");
    assert_eq!(
        sup_b2.agent().messages.len(),
        expected_b,
        "resume B must reproduce B's exact history"
    );
    let roles_b: Vec<Role> = sup_b2
        .agent()
        .messages
        .iter()
        .map(|m| m.role.clone())
        .collect();
    assert_eq!(roles_b, vec![Role::System, Role::User, Role::Assistant]);

    // Second resume of A after B has been in play: still exact, no
    // accumulation across switches.
    let sup_a3 = Supervisor::resume(offline_config(), ws.path(), true, false, sid_a, None)
        .await
        .expect("resume A again");
    assert_eq!(
        sup_a3.agent().messages.len(),
        expected_a,
        "second resume of A must still be exact — history must not accumulate across switches"
    );
    assert_eq!(
        sup_a3
            .agent()
            .messages
            .iter()
            .filter(|m| **m == Message::user("a-second"))
            .count(),
        1,
        "a-second still appears exactly once after the A → B → A round trip"
    );
}
