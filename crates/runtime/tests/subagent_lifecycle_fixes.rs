//! Subagent lifecycle fixes — integration tests for the three audited
//! correctness gaps (`docs` audit: ghost children, revive lease, cancel
//! truth-telling):
//!
//! 1. **Resume ghost sweep**: a parent that crashed mid-child-run resumes
//!    with `Running` registry ghosts. A child json whose `meta.pid` is a
//!    foreign pid is authoritative death evidence (children are
//!    same-process tokio tasks) — resume folds the ghost to `Failed`,
//!    tombstones the PARENT's event log (so a later resume folds the
//!    tombstone instead of re-ghosting), and reports the ghost + its
//!    retained worktree through `take_restart_ghosts`. Same-pid entries
//!    are NOT swept (they may be live under in-process `switch_to`).
//! 2. **Revive lease narrowing**: the revive control lease is released
//!    before the revived turn runs, so a generation-1 child is
//!    `task_cancel`-able mid-run (previously refused with "another
//!    control operation is in flight" for its whole second life).
//! 3. **Cancel truth-telling**: `task_cancel` against a terminal entry
//!    replies `ok=false` carrying the REAL state and generation (never
//!    the stale `state: Running` + `generation: None` shape), and a
//!    `Running` entry without live handles names the stale-projection
//!    situation instead of pretending to have cancelled something.
//!
//! Hermetic: no network — providers ride the injection seams exactly like
//! `task_revive.rs`/`task_control.rs`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use nca_common::config::{NcaConfig, PermissionMode};
use nca_common::event::{AgentEvent, EventEnvelope};
use nca_common::message::Message;
use nca_common::session::{ChildSessionState, SessionMeta, SessionState, SessionStatus};
use nca_common::tool::ToolDefinition;
use nca_core::provider::{Provider, ProviderError, StreamChunk};
use nca_core::tools::subagent_control::SubagentControlRequest;
use nca_runtime::session_store::SessionStore;
use nca_runtime::subagent::{ChildSessionConfig, handle_revive_request, spawn_child_session};
use nca_runtime::subagent_registry::{SubagentRegistry, subagent_control_consumer};
use nca_runtime::supervisor::Supervisor;
use tokio::sync::{mpsc, oneshot};

/// Terminal summary the resume sweep stamps on ghost children (mirrors
/// `supervisor::GHOST_TERMINAL_SUMMARY`, which is crate-private).
const GHOST_SUMMARY: &str = "interrupted: parent process restarted before completion";

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

fn write_event_log(ws: &Path, session_id: &str, events: Vec<AgentEvent>) {
    std::fs::create_dir_all(sessions_dir(ws)).expect("create sessions dir");
    let mut id = 0u64;
    let mut lines = String::new();
    for event in events {
        id += 1;
        lines.push_str(
            &serde_json::to_string(&EventEnvelope::new(id, event)).expect("serialize envelope"),
        );
        lines.push('\n');
    }
    std::fs::write(
        sessions_dir(ws).join(format!("{session_id}.events.jsonl")),
        lines,
    )
    .expect("write events.jsonl");
}

/// Child session json fixture: `pid` is the death-evidence knob
/// (`u32::MAX` can never be a live pid on Linux), `worktree_path` feeds
/// the orphaned-worktree report.
fn write_child_json(ws: &Path, id: &str, pid: Option<u32>, worktree: Option<&str>) {
    std::fs::create_dir_all(sessions_dir(ws)).expect("create sessions dir");
    let now = chrono::Utc::now();
    let state = SessionState {
        meta: SessionMeta {
            id: id.to_string(),
            created_at: now,
            updated_at: now,
            workspace: ws.to_path_buf(),
            model: "deepseek-chat".into(),
            status: SessionStatus::Running,
            pid,
            socket_path: None,
            worktree_path: worktree.map(PathBuf::from),
            branch: worktree.map(|_| format!("nca/{id}")),
            base_branch: None,
            parent_session_id: Some("parent-1".into()),
            child_session_ids: Vec::new(),
            inherited_summary: None,
            spawn_reason: Some("audit the things".into()),
            session_summary: None,
            session_title: None,
            orchestration: None,
            agent_name: None,
        },
        messages: vec![Message::user("task")],
        total_input_tokens: 0,
        total_output_tokens: 0,
        estimated_cost_usd: 0.0,
    };
    std::fs::write(
        sessions_dir(ws).join(format!("{id}.json")),
        serde_json::to_string_pretty(&state).expect("serialize"),
    )
    .expect("write child json");
}

fn write_parent_json(ws: &Path, session_id: &str) {
    std::fs::create_dir_all(sessions_dir(ws)).expect("create sessions dir");
    let now = chrono::Utc::now();
    let state = SessionState {
        meta: SessionMeta {
            id: session_id.to_string(),
            created_at: now,
            updated_at: now,
            workspace: ws.to_path_buf(),
            model: "deepseek-chat".into(),
            status: SessionStatus::Running,
            pid: Some(u32::MAX),
            socket_path: None,
            worktree_path: None,
            branch: None,
            base_branch: None,
            parent_session_id: None,
            child_session_ids: vec!["ghost-1".into(), "live-2".into(), "done-3".into()],
            inherited_summary: None,
            spawn_reason: None,
            session_summary: None,
            session_title: None,
            orchestration: None,
            agent_name: None,
        },
        messages: vec![Message::user("run the audit")],
        total_input_tokens: 0,
        total_output_tokens: 0,
        estimated_cost_usd: 0.0,
    };
    std::fs::write(
        sessions_dir(ws).join(format!("{session_id}.json")),
        serde_json::to_string_pretty(&state).expect("serialize"),
    )
    .expect("write parent json");
}

fn spawned_event(child: &str) -> AgentEvent {
    AgentEvent::ChildSessionSpawned {
        parent_session_id: "parent-1".into(),
        child_session_id: child.into(),
        task: "audit the things".into(),
        workspace: PathBuf::from("/tmp/ws"),
        branch: None,
    }
}

fn status_event(child: &str, state: ChildSessionState, generation: u64) -> AgentEvent {
    AgentEvent::ChildSessionStatusChanged {
        parent_session_id: "parent-1".into(),
        child_session_id: child.into(),
        state,
        generation,
        alias: (child == "ghost-1").then(|| "auditor".to_string()),
        result_summary: None,
    }
}

// ---------------------------------------------------------------------------
// 1. Resume ghost sweep
// ---------------------------------------------------------------------------

/// A crashed parent leaves a `Running` ghost whose child json was written
/// by a foreign pid. Resume must fold it to `Failed`, tombstone the
/// parent's own event log, report it once via `take_restart_ghosts`
/// (with the retained worktree listed), and NOT re-sweep on the next
/// resume. A same-pid `Running` entry and terminal entries stay untouched.
#[tokio::test(flavor = "multi_thread")]
async fn resume_sweeps_cross_pid_ghost_and_tombstones_parent_log() {
    let ws = tempfile::tempdir().expect("tempdir");
    // Three children: a cross-pid ghost (with a retained worktree), a
    // same-pid running entry (potentially live under switch_to), and an
    // already-terminal child.
    write_parent_json(ws.path(), "parent-1");
    write_child_json(
        ws.path(),
        "ghost-1",
        Some(u32::MAX),
        Some("/tmp/wt-ghost-1"),
    );
    write_child_json(ws.path(), "live-2", Some(std::process::id()), None);
    write_child_json(ws.path(), "done-3", Some(u32::MAX), None);
    write_event_log(
        ws.path(),
        "parent-1",
        vec![
            spawned_event("ghost-1"),
            status_event("ghost-1", ChildSessionState::Running, 0),
            spawned_event("live-2"),
            status_event("live-2", ChildSessionState::Running, 0),
            spawned_event("done-3"),
            status_event("done-3", ChildSessionState::Completed, 0),
            AgentEvent::ChildSessionCompleted {
                parent_session_id: "parent-1".into(),
                child_session_id: "done-3".into(),
                status: "completed".into(),
            },
        ],
    );

    let mut sup = Supervisor::resume(
        offline_config(),
        ws.path(),
        true,
        false,
        "parent-1",
        None,
        None,
    )
    .await
    .expect("resume must succeed");

    let registry = sup.subagent_registry();

    // The ghost converged to Failed with the interruption summary.
    let ghost = registry.get("ghost-1").expect("ghost entry");
    assert_eq!(ghost.state, ChildSessionState::Failed);
    assert_eq!(ghost.result_summary.as_deref(), Some(GHOST_SUMMARY));
    assert!(ghost.cancel_flag.is_none(), "no zombie handle");

    // Same-pid running entry is NOT swept (no death evidence).
    let live = registry.get("live-2").expect("live entry");
    assert_eq!(live.state, ChildSessionState::Running);

    // Terminal entry untouched.
    let done = registry.get("done-3").expect("done entry");
    assert_eq!(done.state, ChildSessionState::Completed);

    // Report: exactly the ghost, with alias + retained worktree listed.
    let ghosts = sup.take_restart_ghosts();
    assert_eq!(ghosts.len(), 1, "only the cross-pid ghost is reported");
    assert_eq!(ghosts[0].child_session_id, "ghost-1");
    assert_eq!(ghosts[0].alias.as_deref(), Some("auditor"));
    assert_eq!(
        ghosts[0].worktree_path.as_deref(),
        Some("/tmp/wt-ghost-1"),
        "the orphaned worktree is listed (never deleted)"
    );

    // Parent log carries the tombstone envelope.
    let envelopes = nca_runtime::session_store::read_event_log(&sup.event_log_path());
    let tombstone = envelopes.iter().find(|e| {
        matches!(
            &e.event,
            AgentEvent::ChildSessionStatusChanged {
                child_session_id, state, result_summary, ..
            }
            if child_session_id == "ghost-1"
                && *state == ChildSessionState::Failed
                && result_summary.as_deref() == Some(GHOST_SUMMARY)
        )
    });
    assert!(tombstone.is_some(), "tombstone envelope must be on disk");

    // A second resume folds the tombstone: still Failed, no re-sweep, no
    // duplicate tombstone.
    drop(registry);
    let mut sup2 = Supervisor::resume(
        offline_config(),
        ws.path(),
        true,
        false,
        "parent-1",
        None,
        None,
    )
    .await
    .expect("second resume");
    let entry = sup2.subagent_registry().get("ghost-1").expect("entry");
    assert_eq!(entry.state, ChildSessionState::Failed);
    assert!(sup2.take_restart_ghosts().is_empty(), "no re-sweep");
    let envelopes2 = nca_runtime::session_store::read_event_log(&sup2.event_log_path());
    let tombstones = envelopes2
        .iter()
        .filter(|e| {
            matches!(
                &e.event,
                AgentEvent::ChildSessionStatusChanged {
                    child_session_id, state, ..
                }
                if child_session_id == "ghost-1" && *state == ChildSessionState::Failed
            )
        })
        .count();
    assert_eq!(tombstones, 1, "exactly one tombstone across resumes");
}

// ---------------------------------------------------------------------------
// 2. Revive lease narrowing
// ---------------------------------------------------------------------------

/// Scripted provider: EVERY call parks on the gate until released (or the
/// run is cancelled — the driver's cancel poll unwedges the stream read).
/// The call counter is the deterministic "provably mid-turn" anchor.
struct GatedProvider {
    answer: String,
    calls: std::sync::atomic::AtomicUsize,
}

impl GatedProvider {
    fn new(answer: &str) -> Arc<Self> {
        Arc::new(Self {
            answer: answer.into(),
            calls: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    fn call_count(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[async_trait]
impl Provider for GatedProvider {
    async fn chat(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        _model: &str,
        _workspace_root: &Path,
    ) -> Result<mpsc::Receiver<StreamChunk>, ProviderError> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let (tx, rx) = mpsc::channel(64);
        let answer = self.answer.clone();
        tokio::spawn(async move {
            // Park: never answered. Cancellation (flag poll) is the only
            // exit — the driver drops the stream receiver.
            std::future::pending::<()>().await;
            let _ = tx.send(StreamChunk::TextDelta(answer)).await;
            let _ = tx
                .send(StreamChunk::Finish {
                    reason: "stop".into(),
                })
                .await;
        });
        Ok(rx)
    }
}

fn control_consumer(
    ws: &Path,
    registry: Arc<SubagentRegistry>,
    event_tx: Option<mpsc::Sender<AgentEvent>>,
) -> mpsc::Sender<SubagentControlRequest> {
    let (tx, rx) = mpsc::channel(16);
    let _consumer = subagent_control_consumer(
        rx,
        registry,
        SessionStore::new(sessions_dir(ws)),
        offline_config(),
        ws.to_path_buf(),
        event_tx,
    );
    tx
}

async fn send_cancel(
    tx: &mpsc::Sender<SubagentControlRequest>,
    id: &str,
) -> nca_core::tools::subagent_control::SubagentControlResponse {
    let (reply_tx, reply_rx) = oneshot::channel();
    tx.send(SubagentControlRequest::Cancel {
        session_id: id.into(),
        reason: Some("wrong direction".into()),
        reply: reply_tx,
    })
    .await
    .expect("control channel live");
    reply_rx.await.expect("cancel reply")
}

/// Revive a child, then cancel the REVIVED generation-1 run mid-turn:
/// the cancel must succeed (the revive released its control lease before
/// the second life started) and the revive reply must reflect the
/// cooperative cancellation.
#[tokio::test(flavor = "multi_thread")]
async fn revived_generation_is_cancellable_mid_run() {
    let ws = tempfile::tempdir().expect("tempdir");
    let registry = Arc::new(SubagentRegistry::new());
    let (event_tx, _event_rx) = mpsc::channel(256);
    let provider = GatedProvider::new("never reached");
    let control_tx = control_consumer(ws.path(), registry.clone(), Some(event_tx.clone()));

    // Generation 0: spawn gated (call #1 parks), provably mid-turn, then
    // cancel it to terminal through the real consumer.
    let child_task = tokio::spawn(spawn_child_session(
        ChildSessionConfig {
            parent_session_id: "parent-1".into(),
            task: "first turn blocks on the gate".into(),
            workspace_root: ws.path().to_path_buf(),
            config: offline_config(),
            parent_summary: "[User]: do the thing".into(),
            use_worktree: false,
            focus_files: Vec::new(),
            images: Vec::new(),
            provider_override: None,
            model_override: None,
            specialist: None,
            alias: Some("fixer-9".into()),
            registry: Some(registry.clone()),
            provider: Some(provider.clone() as Arc<dyn Provider>),
            plugins: None,
        },
        Some(event_tx.clone()),
    ));
    let deadline = Instant::now() + Duration::from_secs(10);
    let child_id = loop {
        if let Some(entry) = registry
            .list()
            .into_iter()
            .find(|e| e.cancel_flag.is_some())
            && provider.call_count() >= 1
        {
            break entry.session_id;
        }
        assert!(Instant::now() < deadline, "spawn must record handles");
        tokio::time::sleep(Duration::from_millis(5)).await;
    };

    let cancel0 = send_cancel(&control_tx, &child_id).await;
    assert!(cancel0.ok, "gen-0 cancel: {cancel0:?}");
    let result = tokio::time::timeout(Duration::from_secs(30), child_task)
        .await
        .expect("cancelled gen-0 child must finish")
        .expect("no panic")
        .expect("session completes");
    assert_eq!(result.status, "cancelled");

    // Revive in-flight: generation 1 runs (call #2 parks mid-turn).
    let revive_task = tokio::spawn(handle_revive_request(
        registry.clone(),
        offline_config(),
        ws.path().to_path_buf(),
        Some(event_tx.clone()),
        Some(provider.clone() as Arc<dyn Provider>),
        child_id.clone(),
        "second life".into(),
    ));

    // Wait for gen-1 to be provably mid-turn: Running at generation 1,
    // provider call #2 landed (past `run_turn`'s flag reset), and the
    // revive's control lease is actually released (the probe acquire
    // succeeding proves it — polling the registry alone can observe the
    // instant between record_handles and the explicit drop).
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let entry = registry.get(&child_id).expect("entry");
        let mid_turn = entry.state == ChildSessionState::Running
            && entry.generation == 1
            && provider.call_count() >= 2;
        if mid_turn && let Some(probe) = registry.try_acquire_lease(&child_id) {
            drop(probe);
            break;
        }
        assert!(
            Instant::now() < deadline,
            "gen-1 must reach mid-turn with the lease released"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    // THE regression: cancelling the revived run must NOT be refused as
    // "control operation in flight" (the lease no longer spans the run).
    let cancel1 = send_cancel(&control_tx, &child_id).await;
    assert!(
        cancel1.ok,
        "gen-1 cancel must succeed while the revived run is live: {cancel1:?}"
    );
    assert_eq!(cancel1.state, ChildSessionState::Running);
    assert_eq!(cancel1.generation, Some(1), "real generation in the reply");

    // The revive sequence completes with the cooperative cancellation of
    // its second life — not a hang, not a stuck lease.
    let reply = tokio::time::timeout(Duration::from_secs(30), revive_task)
        .await
        .expect("revive must finish after its run was cancelled")
        .expect("no panic");
    assert!(reply.ok, "revive reply: {reply:?}");
    assert_eq!(reply.generation, Some(1));
    assert_eq!(reply.state, ChildSessionState::Cancelled);

    let entry = registry.get(&child_id).expect("entry");
    assert_eq!(entry.state, ChildSessionState::Cancelled);
    assert_eq!(entry.generation, 1);
}

// ---------------------------------------------------------------------------
// 3. Cancel truth-telling
// ---------------------------------------------------------------------------

/// `task_cancel` against an already-terminal entry: `ok=false` carrying
/// the REAL state and generation — never the stale `state: Running` /
/// `generation: None` shape the audit flagged.
#[tokio::test(flavor = "multi_thread")]
async fn cancel_of_terminal_entry_reports_real_state_and_generation() {
    let ws = tempfile::tempdir().expect("tempdir");
    let registry = Arc::new(SubagentRegistry::new());
    let control_tx = control_consumer(ws.path(), registry.clone(), None);

    // A gen-1 terminal task (spawn → revive → completed).
    registry.record_spawned("parent-1", "c1", "t", "/ws".into(), None);
    registry.record_revive("c1");
    registry.record_terminal("c1", ChildSessionState::Completed, Some("done".into()));

    let resp = send_cancel(&control_tx, "c1").await;
    assert!(!resp.ok, "cancel of a terminal task must be refused");
    assert_eq!(
        resp.state,
        ChildSessionState::Completed,
        "the reply carries the real state, not a stale Running"
    );
    assert_eq!(
        resp.generation,
        Some(1),
        "the reply carries the real generation (was None before)"
    );
    assert!(resp.note.as_deref().is_some_and(|n| !n.is_empty()));

    // A Running entry WITHOUT live handles (the post-crash fold shape
    // before the ghost sweep): the reply must name the situation — not
    // flip a nonexistent flag and claim success.
    registry.record_spawned("parent-1", "ghost-2", "t", "/ws".into(), None);
    let resp = send_cancel(&control_tx, "ghost-2").await;
    assert!(!resp.ok, "no live handle → nothing to cancel: {resp:?}");
    let note = resp.note.as_deref().expect("note");
    assert!(
        note.contains("no live cancel handle"),
        "note must name the stale-projection situation: {note}"
    );
    assert_eq!(resp.generation, Some(0));
}
