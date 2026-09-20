//! P2 chunk B integration — `task_cancel` mid-turn: a child aborted by its
//! registry-recorded cancel flag ends `cancelled` (session json + registry +
//! events) and its worktree is RETAINED on disk (no rollback).
//!
//! Mirrors the t27 scaffolding (`phase_c.rs`) for the consumer/event wiring
//! and the gated-stream pattern from `inbox.rs`: the child's provider
//! returns a stream that stays open until a `Notify` gate fires, so the
//! turn is provably mid-stream when the cancel lands (deterministic — no
//! fixed sleeps, no timing luck). The cancel itself rides the REAL
//! `subagent_control_consumer` (the same task_cancel path production uses).
//!
//! No network: the provider is injected through the
//! `ChildSessionConfig::provider` test seam (mirroring
//! `SupervisorConfig::provider`), exactly like the scripted providers in
//! `inbox.rs`/`phase_c.rs`.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use nca_common::config::{NcaConfig, PermissionMode};
use nca_common::event::AgentEvent;
use nca_common::message::Message;
use nca_common::session::{ChildSessionState, SessionStatus};
use nca_common::tool::ToolDefinition;
use nca_core::provider::{Provider, ProviderError, StreamChunk};
use nca_core::tools::subagent_control::SubagentControlRequest;
use nca_runtime::session_store::SessionStore;
use nca_runtime::subagent::{ChildSessionConfig, spawn_child_session};
use nca_runtime::subagent_registry::{SubagentRegistry, subagent_control_consumer};
use tokio::sync::{mpsc, oneshot};

fn offline_config() -> NcaConfig {
    let mut config = NcaConfig::default();
    config.provider.deepseek.api_key = Some("test-key".into());
    config.permissions.mode = PermissionMode::BypassPermissions;
    config.memory.context.auto_detect_context_window = false;
    config.memory.context.query_provider_models_api = false;
    config.memory.context.enable_auto_summarize = false;
    config
}

/// A `git init`-ed workspace with one commit (worktree creation requires a
/// real HEAD) in a fresh tempdir.
fn git_workspace() -> tempfile::TempDir {
    let ws = tempfile::tempdir().expect("tempdir");
    let run = |args: &[&str]| {
        let status = std::process::Command::new("git")
            .args(args)
            .current_dir(ws.path())
            .status()
            .expect("git must be installed");
        assert!(status.success(), "git {args:?} must succeed");
    };
    run(&["init", "-q"]);
    run(&["config", "user.email", "test@nca"]);
    run(&["config", "user.name", "nca test"]);
    std::fs::write(ws.path().join("README.md"), "workspace\n").expect("write");
    run(&["add", "."]);
    run(&["commit", "-q", "-m", "init"]);
    ws
}

/// Gated provider: `chat()` resolves immediately with a receiver whose
/// sender task parks on a `Notify` gate before delivering the round — the
/// child's turn is provably mid-stream (blocked in `stream.recv()`) while
/// the gate holds, which is exactly what `task_cancel` interrupts via the
/// 25ms cancel poll in the stream loop. The call counter is the
/// deterministic "turn in flight" anchor (mirrors `inbox.rs`).
struct GatedProvider {
    release: Arc<tokio::sync::Notify>,
    calls: std::sync::atomic::AtomicUsize,
}

impl GatedProvider {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            release: Arc::new(tokio::sync::Notify::new()),
            calls: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    fn release_gate(&self) {
        self.release.notify_one();
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
        let release = self.release.clone();
        tokio::spawn(async move {
            release.notified().await;
            let _ = tx
                .send(StreamChunk::TextDelta("mid-turn answer".into()))
                .await;
            let _ = tx
                .send(StreamChunk::Finish {
                    reason: "stop".into(),
                })
                .await;
        });
        Ok(rx)
    }
}

/// Drain a bounded event channel to completion (all senders dropped) and
/// collect the events.
async fn collect_events(mut rx: mpsc::Receiver<AgentEvent>) -> Vec<AgentEvent> {
    let mut events = Vec::new();
    while let Some(event) = rx.recv().await {
        events.push(event);
    }
    events
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_mid_turn_ends_child_cancelled_and_retains_worktree() {
    let ws = git_workspace();
    let config = offline_config();

    let registry = Arc::new(SubagentRegistry::new());
    let (parent_tx, parent_rx) = mpsc::channel(256);
    let events = collect_events(parent_rx);

    // The REAL control consumer (task_cancel path), wired like the
    // supervisor wires it: registry + read-only store + the parent's event
    // channel.
    let (control_tx, control_rx) = mpsc::channel(16);
    let sessions_dir = ws.path().join(".nca").join("sessions");
    let _consumer = subagent_control_consumer(
        control_rx,
        registry.clone(),
        SessionStore::new(sessions_dir.clone()),
        Arc::new(std::sync::RwLock::new(offline_config())),
        Arc::new(nca_core::workspace_fs::RealFs::new(ws.path().to_path_buf())),
        ws.path().to_path_buf(),
        Some(parent_tx.clone()),
    );
    let store = SessionStore::new(sessions_dir);

    let gated = GatedProvider::new();
    let provider: Arc<dyn Provider> = gated.clone();

    // The child: spawned exactly like `spawn_subagent_consumer` spawns it,
    // with a gated provider blocking mid-turn. Worktree requested so the
    // retention assertion has a directory to check.
    let child_task = tokio::spawn(spawn_child_session(
        ChildSessionConfig {
            parent_session_id: "parent-1".into(),
            task: "block on the gated stream".into(),
            workspace_root: ws.path().to_path_buf(),
            config,
            parent_summary: "[User]: do the thing".into(),
            use_worktree: true,
            focus_files: Vec::new(),
            images: Vec::new(),
            provider_override: None,
            model_override: None,
            specialist: None,
            alias: None,
            registry: Some(registry.clone()),
            provider: Some(provider),
            plugins: None,
        },
        Some(parent_tx.clone()),
    ));

    // Deterministic "mid-turn" anchor, in two steps:
    //  1. the registry carries the child's live cancel handle (recorded
    //     between spawn and run_turn), and
    //  2. the turn is provably in flight (provider call #1 returned and the
    //     stream is parked on the gate) — cancelling BEFORE run_turn starts
    //     would be swallowed by the per-turn flag reset in `run_turn`, so
    //     anchoring on the call keeps this a true mid-stream cancel.
    let deadline = Instant::now() + Duration::from_secs(10);
    let child_id = loop {
        let ready = registry
            .list()
            .into_iter()
            .find(|e| e.cancel_flag.is_some())
            .map(|e| e.session_id);
        if let Some(id) = ready
            && gated.call_count() >= 1
        {
            break id;
        }
        assert!(
            Instant::now() < deadline,
            "child spawn must record its cancel handle and start its turn"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    };
    assert_eq!(
        registry.get(&child_id).map(|e| e.state),
        Some(ChildSessionState::Running)
    );

    // task_cancel through the real consumer: lease → flag-flip → reason.
    let (reply_tx, reply_rx) = oneshot::channel();
    control_tx
        .send(SubagentControlRequest::Cancel {
            session_id: child_id.clone(),
            reason: Some("wrong branch".into()),
            reply: reply_tx,
        })
        .await
        .expect("control channel live");
    let cancel_reply = reply_rx.await.expect("cancel reply");
    assert!(cancel_reply.ok, "cancel must be accepted: {cancel_reply:?}");
    let note = cancel_reply.note.as_deref().expect("note");
    assert!(
        note.contains("cancel requested") && note.contains("retained"),
        "note must explain the cooperative abort + retention: {note}"
    );

    // HOLD the gate: the abort fires on the 25ms cancel poll while the
    // stream stays parked — releasing now would race the stream to
    // completion (chunks arriving in the same instant as the flag make
    // the outcome ambiguous). The release below is cleanup only, after the
    // child provably aborted.

    // The child ends cancelled — bounded wait, no fixed sleep.
    let result = tokio::time::timeout(Duration::from_secs(30), child_task)
        .await
        .expect("child task must finish after cancel")
        .expect("spawn_child_session must not panic")
        .expect("child session must complete, not error out");
    assert_eq!(result.status, "cancelled");
    assert!(
        result.output.contains("run cancelled"),
        "cancel output surfaces the cooperative abort: {}",
        result.output
    );

    // Cleanup release for the still-parked provider sender task (the abort
    // never needed it — the flag poll fired while the stream was held).
    gated.release_gate();

    // Worktree + branch RETAINED on disk (cancel never rolls back).
    let worktree = result
        .worktree_path
        .clone()
        .expect("child ran with use_worktree");
    assert!(
        Path::new(&worktree).is_dir(),
        "worktree must still exist after cancel: {worktree}"
    );
    assert!(result.branch.is_some(), "branch retained for revive");

    // Registry projection: terminal Cancelled, reason folded into the
    // summary, live handles cleared.
    let entry = registry.get(&child_id).expect("entry retained");
    assert_eq!(entry.state, ChildSessionState::Cancelled);
    assert_eq!(
        entry.result_summary.as_deref(),
        Some("cancelled: wrong branch")
    );
    assert!(entry.cancel_flag.is_none(), "terminal: handles cleared");
    assert!(entry.inbox_tx.is_none());

    // Child session json: its OWN supervisor persisted Cancelled
    // (single-writer — the control consumer never wrote it).
    let child_state = store.load(&child_id).await.expect("child json saved");
    assert_eq!(child_state.meta.status, SessionStatus::Cancelled);

    // Parent event stream: the terminal StatusChanged(Cancelled) was
    // emitted before the drain. Dropping both senders closes the channel:
    // parent_tx directly, control_tx via ending the consumer (which holds
    // an event_tx clone).
    drop(parent_tx);
    drop(control_tx);
    let events = tokio::time::timeout(Duration::from_secs(5), events)
        .await
        .expect("event collector must finish");
    let saw_cancelled = events.iter().any(|e| {
        matches!(
            e,
            AgentEvent::ChildSessionStatusChanged {
                state: ChildSessionState::Cancelled,
                ..
            }
        )
    });
    assert!(
        saw_cancelled,
        "ChildSessionStatusChanged(cancelled) must reach the parent stream"
    );
}
