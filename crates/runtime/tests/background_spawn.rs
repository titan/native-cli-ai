//! P2 chunk C integration — `background` spawn: the reply returns
//! IMMEDIATELY after prepare (status "running", §6 invariant: the oneshot is
//! never held for the foreground 600s window) while the child runs detached;
//! `alias` lands in the registry entry and the running
//! `ChildSessionStatusChanged` event. Foreground (background:false) keeps the
//! synchronous contract and returns the child's terminal output. A consumer
//! with NO wake scheduler (stdio/one-shot sessions) forces the foreground
//! contract even for an explicit `background: true` — a detached child there
//! could never wake the idle parent and would orphan behind an undeliverable
//! wake, so the spawn keeps the synchronous reply instead.
//!
//! Same hermetic scaffolding as `task_control.rs`: the child's provider is a
//! gated scripted provider (round 1 parks on a `Notify` gate), injected
//! through the `child_provider` test seam of `spawn_subagent_consumer`
//! (production callers always pass `None`).

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use nca_common::config::{NcaConfig, PermissionMode};
use nca_common::event::AgentEvent;
use nca_common::message::Message;
use nca_common::session::ChildSessionState;
use nca_common::tool::ToolDefinition;
use nca_core::provider::{Provider, ProviderError, StreamChunk};
use nca_core::tools::spawn_subagent::{SpawnRequest, SpawnResponse};
use nca_core::workspace_fs::{RealFs, WorkspaceFs};
use nca_runtime::subagent_registry::SubagentRegistry;
use nca_runtime::supervisor::spawn_subagent_consumer;
use nca_runtime::wake_scheduler::{WakeScheduler, WakeTrigger};
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

/// Scripted gated provider: call 1 parks on the gate (provably mid-turn),
/// every later call answers immediately with `answer`. The call counter is
/// the deterministic "turn in flight" anchor.
struct GatedScriptedProvider {
    release: Arc<tokio::sync::Notify>,
    answer: String,
    calls: std::sync::atomic::AtomicUsize,
}

impl GatedScriptedProvider {
    fn new(answer: &str) -> Arc<Self> {
        Arc::new(Self {
            release: Arc::new(tokio::sync::Notify::new()),
            answer: answer.into(),
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
impl Provider for GatedScriptedProvider {
    async fn chat(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        _model: &str,
        _workspace_root: &Path,
    ) -> Result<mpsc::Receiver<StreamChunk>, ProviderError> {
        let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        let (tx, rx) = mpsc::channel(64);
        let release = self.release.clone();
        let answer = self.answer.clone();
        tokio::spawn(async move {
            if call == 1 {
                // First turn parks mid-stream: only the gate (or a cancel)
                // can end it.
                release.notified().await;
            }
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

fn spawn_request(
    task: &str,
    background: bool,
    alias: Option<&str>,
    reply: oneshot::Sender<SpawnResponse>,
) -> SpawnRequest {
    SpawnRequest {
        task: task.into(),
        focus_files: Vec::new(),
        images: Vec::new(),
        use_worktree: true,
        background: Some(background),
        alias: alias.map(String::from),
        provider_override: None,
        model_override: None,
        specialist: None,
        reply,
    }
}

/// Enabled wake scheduler recording delivered texts on an unbounded channel
/// (same seam as `wake_integration.rs`) — the background test needs a wake
/// path for the spawn consumer to honor `background: true`; the channel is
/// simply ignored (the detached mechanics are the assertion target).
fn wake_channel() -> (WakeScheduler, mpsc::UnboundedReceiver<String>) {
    let (tx, rx) = mpsc::unbounded_channel();
    let trigger: WakeTrigger = Arc::new(move |text: &str| {
        let _ = tx.send(text.to_string());
    });
    (
        WakeScheduler::new(true, Duration::from_millis(50), trigger),
        rx,
    )
}

/// Wire the real spawn consumer with a provider seam + event tap.
struct ConsumerHarness {
    spawn_tx: mpsc::Sender<SpawnRequest>,
    registry: Arc<SubagentRegistry>,
    event_rx: mpsc::Receiver<AgentEvent>,
    _events: tokio::task::JoinHandle<()>,
}

fn wire_consumer(
    ws: &Path,
    provider: Arc<dyn Provider>,
    wake: Option<WakeScheduler>,
) -> ConsumerHarness {
    let registry = Arc::new(SubagentRegistry::new());
    let (spawn_tx, spawn_rx) = mpsc::channel(4);
    let (event_tx, event_rx) = mpsc::channel(256);
    let parent_fs: Arc<dyn WorkspaceFs> = Arc::new(RealFs::new(ws.to_path_buf()));
    let _consumer = spawn_subagent_consumer(
        spawn_rx,
        "parent-1".into(),
        ws.to_path_buf(),
        Arc::new(std::sync::RwLock::new(offline_config())),
        Arc::new(std::sync::Mutex::new(vec![Message::user("parent context")])),
        Some(event_tx),
        registry.clone(),
        parent_fs,
        Some(provider),
        // No wake scheduler wired here: these tests pass explicit flags, so
        // the default is irrelevant; the forced-foreground semantics under a
        // missing wake path are pinned by their own test below.
        false,
        wake,
        None,
    );
    // Collect events on a side task so the tap receiver never blocks the
    // bounded channel; tests scan the collected log at assert time.
    let (tap_tx, tap_rx) = mpsc::channel(1024);
    let _events = tokio::spawn(async move {
        let mut rx = event_rx;
        while let Some(event) = rx.recv().await {
            let _ = tap_tx.send(event).await;
        }
    });
    let event_rx = tap_rx;
    ConsumerHarness {
        spawn_tx,
        registry,
        event_rx,
        _events,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn background_spawn_replies_immediately_and_child_completes_detached() {
    let ws = git_workspace();
    let gated = GatedScriptedProvider::new("detached answer");
    let provider: Arc<dyn Provider> = gated.clone();
    // A wake path is the precondition for honoring `background: true` — wire
    // the scheduler exactly like a TUI session does.
    let (sched, _wake_rx) = wake_channel();
    let mut harness = wire_consumer(ws.path(), provider, Some(sched));

    // Background spawn with an alias.
    let (reply_tx, reply_rx) = oneshot::channel();
    harness
        .spawn_tx
        .send(spawn_request(
            "block then answer",
            true,
            Some("fixer-x"),
            reply_tx,
        ))
        .await
        .expect("spawn request accepted");

    // The reply must arrive while the gate STILL HOLDS — i.e. while the
    // child is provably mid-turn (provider call #1 returned and its stream
    // is parked). Bounded wait; no fixed sleeps.
    let reply = tokio::time::timeout(Duration::from_secs(10), reply_rx)
        .await
        .expect("background reply must be immediate (never held 600s)")
        .expect("reply channel must not be dropped");
    assert_eq!(reply.status, "running", "immediate reply: {reply:?}");
    assert!(!reply.child_session_id.is_empty());
    assert!(reply.output.is_empty(), "no output before completion");
    let child_id = reply.child_session_id.clone();
    assert!(
        reply.branch.is_some() && reply.worktree_path.is_some(),
        "worktree info must be present at reply time: {reply:?}"
    );

    // Deterministic mid-turn proof: the child started its first provider
    // call while the gate is still closed.
    let deadline = Instant::now() + Duration::from_secs(10);
    while gated.call_count() == 0 {
        assert!(Instant::now() < deadline, "child must start its turn");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    // Alias landed in the registry entry while running.
    let entry = harness.registry.get(&child_id).expect("registry entry");
    assert_eq!(entry.alias.as_deref(), Some("fixer-x"));
    assert_eq!(entry.state, ChildSessionState::Running);
    assert!(
        entry.worktree_path.is_some(),
        "worktree path must be recorded for later revive"
    );

    // Release the gate; the detached child completes (bounded wait ≤10s).
    gated.release_gate();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let entry = harness.registry.get(&child_id).expect("registry entry");
        if entry.state.is_terminal() {
            assert_eq!(
                entry.state,
                ChildSessionState::Completed,
                "released gate → child completes"
            );
            break;
        }
        assert!(
            Instant::now() < deadline,
            "detached child must reach a terminal state after gate release"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // The running StatusChanged event carried the alias.
    let mut saw_alias_event = false;
    while let Ok(event) = harness.event_rx.try_recv() {
        if let AgentEvent::ChildSessionStatusChanged {
            child_session_id,
            state: ChildSessionState::Running,
            alias,
            ..
        } = &event
            && child_session_id == &child_id
        {
            assert_eq!(alias.as_deref(), Some("fixer-x"), "event: {event:?}");
            saw_alias_event = true;
        }
    }
    assert!(
        saw_alias_event,
        "running ChildSessionStatusChanged must carry the alias"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn foreground_spawn_still_returns_terminal_child_output() {
    let ws = git_workspace();
    // Call 1 is gated; release it eagerly via a fresh Notify... simplest:
    // answer on every call by pre-releasing after wiring is not possible,
    // so use a provider whose first call is gated but release immediately
    // after sending the request.
    let gated = GatedScriptedProvider::new("foreground answer");
    let provider: Arc<dyn Provider> = gated.clone();
    let harness = wire_consumer(ws.path(), provider, None);

    let (reply_tx, reply_rx) = oneshot::channel();
    harness
        .spawn_tx
        .send(spawn_request("answer now", false, None, reply_tx))
        .await
        .expect("spawn request accepted");
    // The turn parks on the gate; release so the foreground await finishes.
    let releaser = gated.clone();
    tokio::spawn(async move {
        // Wait until the child is provably mid-turn, then release.
        let deadline = Instant::now() + Duration::from_secs(10);
        while releaser.call_count() == 0 {
            assert!(Instant::now() < deadline, "child must start");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        releaser.release_gate();
    });

    let reply = tokio::time::timeout(Duration::from_secs(30), reply_rx)
        .await
        .expect("foreground reply must arrive")
        .expect("reply channel must not be dropped");
    assert_eq!(reply.status, "completed", "foreground: {reply:?}");
    assert_eq!(
        reply.output, "foreground answer",
        "foreground reply carries the child's terminal output verbatim"
    );
}

/// No wake scheduler (stdio/one-shot sessions): an EXPLICIT `background:
/// true` must NOT detach — the child could never wake the idle parent and
/// the process may exit before it finishes. The spawn keeps the synchronous
/// foreground contract instead (reply = terminal status + output).
#[tokio::test(flavor = "multi_thread")]
async fn explicit_background_without_wake_path_forces_foreground() {
    let ws = git_workspace();
    let gated = GatedScriptedProvider::new("forced fg answer");
    let provider: Arc<dyn Provider> = gated.clone();
    // No scheduler wired — the stdio/one-shot wiring shape.
    let harness = wire_consumer(ws.path(), provider.clone(), None);

    let (reply_tx, reply_rx) = oneshot::channel();
    harness
        .spawn_tx
        .send(spawn_request(
            "orphaned without a wake path",
            true,
            None,
            reply_tx,
        ))
        .await
        .expect("spawn request accepted");
    // The child parks mid-turn; release once provably started so the
    // synchronous await finishes (mirrors the foreground test's releaser).
    let releaser = gated.clone();
    tokio::spawn(async move {
        let deadline = Instant::now() + Duration::from_secs(10);
        while releaser.call_count() == 0 {
            assert!(Instant::now() < deadline, "child must start");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        releaser.release_gate();
    });

    let reply = tokio::time::timeout(Duration::from_secs(30), reply_rx)
        .await
        .expect("forced-foreground reply must arrive (never a detached 'running')")
        .expect("reply channel must not be dropped");
    assert_eq!(
        reply.status, "completed",
        "explicit background=true without a wake path stays foreground: {reply:?}"
    );
    assert_eq!(
        reply.output, "forced fg answer",
        "forced-foreground reply carries the terminal output verbatim"
    );
}
