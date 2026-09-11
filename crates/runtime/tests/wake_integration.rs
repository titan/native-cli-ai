//! P3 chunk B integration — background default policy + wake-on-terminal.
//!
//! Covers the consumer-level contract added on top of the chunk A wake
//! scheduler state machine:
//! - an OMITTED `background` flag inherits `background_default` (true →
//!   immediate `status:"running"` reply + detached run + exactly one wake
//!   on terminal, carrying the child ref — alias when set — and the
//!   terminal state; false → P2 foreground semantics, the rollback path);
//! - an explicit `background:false` under a default-true policy keeps the
//!   synchronous foreground contract and NEVER wakes;
//! - `task_cancel` of a running background child wakes with `cancelled`
//!   and the recorded cancel reason folded into the summary;
//! - `task_revive` of a terminal background child runs its new turn with
//!   NO additional wake (the reviving parent is mid-turn holding the
//!   reply);
//! - two background children terminating inside the debounce window
//!   produce exactly ONE wake;
//! - P4 wiring chain: `WaitForUserTool` built with the scheduler's `pause`
//!   closure (exactly how `run_with_tui` wires it) suppresses wakes until
//!   the next `note_input` (the TUI Submit choke point), then exactly one
//!   wake fires for a later terminal.
//!
//! Hermetic, same scaffolding as `background_spawn.rs`/`task_control.rs`:
//! gated scripted providers injected through the `child_provider` seam of
//! `spawn_subagent_consumer` (production callers pass `None`), the REAL
//! `subagent_control_consumer` for cancel, and `handle_revive_request` for
//! revive. Every supervisor here gets an injected provider, so no
//! `DEEPSEEK_API_KEY` env self-injection is needed (`offline_config()`
//! pins a dummy key for config validation paths only).

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use nca_common::config::{NcaConfig, PermissionMode};
use nca_common::event::AgentEvent;
use nca_common::message::Message;
use nca_common::session::ChildSessionState;
use nca_common::tool::{ToolCall, ToolDefinition};
use nca_core::provider::{Provider, ProviderError, StreamChunk};
use nca_core::tools::spawn_subagent::{SpawnRequest, SpawnResponse};
use nca_core::tools::subagent_control::SubagentControlRequest;
use nca_core::tools::{ToolExecutor, WaitForUserTool};
use nca_core::workspace_fs::{RealFs, WorkspaceFs};
use nca_runtime::session_store::SessionStore;
use nca_runtime::subagent::{handle_revive_request, spawn_subagent_consumer};
use nca_runtime::subagent_registry::{SubagentRegistry, subagent_control_consumer};
use nca_runtime::wake_scheduler::{WakeScheduler, WakeTrigger};
use tokio::sync::{mpsc, oneshot};

/// Small real-time debounce interval for the fire-path tests (the wake
/// lands ~interval after the terminal; every wait is a bounded
/// channel/deadline wait, never a fixed sleep).
const WAKE_INTERVAL: Duration = Duration::from_millis(50);

/// Wider window for the coalesce test: both children's terminal hooks must
/// land inside one debounce window despite scheduling jitter.
const COALESCE_INTERVAL: Duration = Duration::from_millis(250);

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
/// every later call answers immediately — the revive test reuses the same
/// instance so its second turn ends deterministically.
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

/// Gated provider serving N gated children through one `child_provider`
/// seam: the first N calls each park on their OWN gate, every later call
/// answers immediately. `release_all()` opens every gate within
/// microseconds — the coalesce test's "two terminals inside one debounce
/// window" anchor.
struct SequencedGatedProvider {
    gates: Vec<Arc<tokio::sync::Notify>>,
    answer: String,
    calls: std::sync::atomic::AtomicUsize,
}

impl SequencedGatedProvider {
    fn new(answer: &str, gates: usize) -> Arc<Self> {
        Arc::new(Self {
            gates: (0..gates)
                .map(|_| Arc::new(tokio::sync::Notify::new()))
                .collect(),
            answer: answer.into(),
            calls: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    fn release_all(&self) {
        for gate in &self.gates {
            gate.notify_one();
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[async_trait]
impl Provider for SequencedGatedProvider {
    async fn chat(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        _model: &str,
        _workspace_root: &Path,
    ) -> Result<mpsc::Receiver<StreamChunk>, ProviderError> {
        let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        let (tx, rx) = mpsc::channel(64);
        let gate = self.gates.get(call - 1).cloned();
        let answer = self.answer.clone();
        tokio::spawn(async move {
            if let Some(gate) = gate {
                gate.notified().await;
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

/// Enabled wake scheduler whose trigger records every delivered wake text
/// on an unbounded channel (mirrors the chunk A unit-test seam).
fn wake_channel(interval: Duration) -> (WakeScheduler, mpsc::UnboundedReceiver<String>) {
    let (tx, rx) = mpsc::unbounded_channel();
    let trigger: WakeTrigger = Arc::new(move |text: &str| {
        let _ = tx.send(text.to_string());
    });
    (WakeScheduler::new(true, interval, trigger), rx)
}

fn spawn_request(
    task: &str,
    background: Option<bool>,
    alias: Option<&str>,
    reply: oneshot::Sender<SpawnResponse>,
) -> SpawnRequest {
    SpawnRequest {
        task: task.into(),
        focus_files: Vec::new(),
        images: Vec::new(),
        use_worktree: true,
        background,
        alias: alias.map(String::from),
        provider_override: None,
        model_override: None,
        specialist: None,
        reply,
    }
}

/// Wire the real spawn consumer with the P3 policy params + event tap.
struct ConsumerHarness {
    spawn_tx: mpsc::Sender<SpawnRequest>,
    registry: Arc<SubagentRegistry>,
    event_tx: mpsc::Sender<AgentEvent>,
    /// Held (not asserted on) so the tap task's forwarding receiver stays
    /// alive — dropping it would end the tap and eventually backpressure
    /// the consumer's bounded event channel.
    #[allow(dead_code)]
    event_rx: mpsc::Receiver<AgentEvent>,
    _events: tokio::task::JoinHandle<()>,
}

fn wire_consumer(
    ws: &Path,
    provider: Arc<dyn Provider>,
    background_default: bool,
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
        offline_config(),
        Arc::new(std::sync::Mutex::new(vec![Message::user("parent context")])),
        Some(event_tx.clone()),
        registry.clone(),
        parent_fs,
        Some(provider),
        background_default,
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
    ConsumerHarness {
        spawn_tx,
        registry,
        event_tx,
        event_rx: tap_rx,
        _events,
    }
}

/// Wire the REAL control consumer (task_cancel path) against the spawn
/// consumer's registry, like the supervisor does.
fn wire_control_consumer(
    ws: &Path,
    registry: Arc<SubagentRegistry>,
    event_tx: Option<mpsc::Sender<AgentEvent>>,
) -> mpsc::Sender<SubagentControlRequest> {
    let (control_tx, control_rx) = mpsc::channel(16);
    let sessions_dir = ws.join(".nca").join("sessions");
    let _consumer = subagent_control_consumer(
        control_rx,
        registry,
        SessionStore::new(sessions_dir),
        offline_config(),
        ws.to_path_buf(),
        event_tx,
    );
    control_tx
}

/// Bounded deadline poll (the established deterministic-anchor pattern —
/// tiny polls inside a hard deadline, not fixed sleeps).
async fn wait_for(predicate: impl Fn() -> bool, what: &str) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !predicate() {
        assert!(Instant::now() < deadline, "{what} must happen");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Bounded wait for the child's registry entry to reach a terminal state.
async fn wait_terminal(registry: &SubagentRegistry, child_id: &str) -> ChildSessionState {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(entry) = registry.get(child_id)
            && entry.state.is_terminal()
        {
            return entry.state;
        }
        assert!(
            Instant::now() < deadline,
            "child {child_id} must reach a terminal state"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Assert NO wake fires within `window` — a channel-timeout wait that
/// doubles as the silence window (comfortably larger than the debounce
/// interval, so a hypothetical late terminal hook would be caught).
async fn assert_silent(rx: &mut mpsc::UnboundedReceiver<String>, window: Duration, context: &str) {
    if let Ok(text) = tokio::time::timeout(window, rx.recv()).await {
        panic!("unexpected wake ({context}): {text:?}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn omitted_flag_with_default_true_runs_background_and_wakes_once() {
    let ws = git_workspace();
    let gated = GatedScriptedProvider::new("bg answer");
    let (sched, mut wake_rx) = wake_channel(WAKE_INTERVAL);
    let harness = wire_consumer(ws.path(), gated.clone(), true, Some(sched));

    // OMITTED background flag: inherits background_default=true.
    let (reply_tx, reply_rx) = oneshot::channel();
    harness
        .spawn_tx
        .send(spawn_request(
            "omitted flag task",
            None,
            Some("fixer-wake"),
            reply_tx,
        ))
        .await
        .expect("spawn request accepted");

    // Immediate running reply — the oneshot is never held for the child.
    let reply = tokio::time::timeout(Duration::from_secs(10), reply_rx)
        .await
        .expect("background reply must be immediate")
        .expect("reply channel must not be dropped");
    assert_eq!(reply.status, "running", "inherited background: {reply:?}");
    let child_id = reply.child_session_id.clone();

    // Deterministic mid-turn anchor, then release: the detached child
    // completes and the wake fires (bounded waits).
    wait_for(|| gated.call_count() >= 1, "child must start its turn").await;
    gated.release_gate();

    let text = tokio::time::timeout(Duration::from_secs(10), wake_rx.recv())
        .await
        .expect("wake must fire after the terminal state")
        .expect("wake channel must not be dropped");
    assert!(
        text.contains("fixer-wake"),
        "wake carries the ALIAS as child ref: {text}"
    );
    assert!(
        text.contains("completed"),
        "wake carries the terminal state: {text}"
    );
    assert_eq!(
        wait_terminal(&harness.registry, &child_id).await,
        ChildSessionState::Completed
    );

    // Exactly ONE wake: nothing further is pending for this child.
    assert!(
        wake_rx.try_recv().is_err(),
        "exactly one wake total, got a second: {text}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn explicit_foreground_with_default_true_never_wakes() {
    let ws = git_workspace();
    let gated = GatedScriptedProvider::new("fg answer");
    let (sched, mut wake_rx) = wake_channel(WAKE_INTERVAL);
    let harness = wire_consumer(ws.path(), gated.clone(), true, Some(sched));

    // Explicit background=false overrides the default-true policy.
    let (reply_tx, reply_rx) = oneshot::channel();
    harness
        .spawn_tx
        .send(spawn_request("fg task", Some(false), None, reply_tx))
        .await
        .expect("spawn request accepted");
    // Release the gate once the child is provably mid-turn so the
    // synchronous await finishes.
    let releaser = gated.clone();
    tokio::spawn(async move {
        wait_for(|| releaser.call_count() >= 1, "child must start").await;
        releaser.release_gate();
    });

    let reply = tokio::time::timeout(Duration::from_secs(30), reply_rx)
        .await
        .expect("foreground reply must arrive")
        .expect("reply channel must not be dropped");
    assert_eq!(reply.status, "completed", "foreground: {reply:?}");
    assert_eq!(
        reply.output, "fg answer",
        "foreground reply carries the terminal output"
    );

    // ZERO wakes after completion — the output was returned inline.
    assert_silent(
        &mut wake_rx,
        WAKE_INTERVAL * 6,
        "foreground reply never wakes",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn omitted_flag_with_default_false_keeps_p2_foreground() {
    let ws = git_workspace();
    let gated = GatedScriptedProvider::new("p2 answer");
    // Rollback wiring: default false, no scheduler — exact P2 semantics.
    let harness = wire_consumer(ws.path(), gated.clone(), false, None);

    let (reply_tx, reply_rx) = oneshot::channel();
    harness
        .spawn_tx
        .send(spawn_request("p2 task", None, None, reply_tx))
        .await
        .expect("spawn request accepted");
    let releaser = gated.clone();
    tokio::spawn(async move {
        wait_for(|| releaser.call_count() >= 1, "child must start").await;
        releaser.release_gate();
    });

    let reply = tokio::time::timeout(Duration::from_secs(30), reply_rx)
        .await
        .expect("foreground reply must arrive")
        .expect("reply channel must not be dropped");
    assert_eq!(
        reply.status, "completed",
        "omitted flag under default=false stays foreground: {reply:?}"
    );
    assert_eq!(reply.output, "p2 answer");
}

#[tokio::test(flavor = "multi_thread")]
async fn cancel_of_running_background_child_wakes_cancelled_with_reason() {
    let ws = git_workspace();
    let gated = GatedScriptedProvider::new("never delivered");
    let (sched, mut wake_rx) = wake_channel(WAKE_INTERVAL);
    let harness = wire_consumer(ws.path(), gated.clone(), true, Some(sched));
    let control_tx = wire_control_consumer(ws.path(), harness.registry.clone(), None);

    // Background child (inherited default), provably mid-turn.
    let (reply_tx, reply_rx) = oneshot::channel();
    harness
        .spawn_tx
        .send(spawn_request(
            "block then get cancelled",
            None,
            Some("fixer-cancel"),
            reply_tx,
        ))
        .await
        .expect("spawn request accepted");
    let reply = tokio::time::timeout(Duration::from_secs(10), reply_rx)
        .await
        .expect("background reply must be immediate")
        .expect("reply channel must not be dropped");
    assert_eq!(reply.status, "running");
    let child_id = reply.child_session_id.clone();
    // Anchor on the provider call: cancelling before run_turn starts would
    // be swallowed by the per-turn flag reset.
    wait_for(|| gated.call_count() >= 1, "child must start its turn").await;

    // task_cancel through the REAL control consumer.
    let (cancel_tx, cancel_rx) = oneshot::channel();
    control_tx
        .send(SubagentControlRequest::Cancel {
            session_id: child_id.clone(),
            reason: Some("wrong branch".into()),
            reply: cancel_tx,
        })
        .await
        .expect("control channel live");
    let cancel_reply = cancel_rx.await.expect("cancel reply");
    assert!(cancel_reply.ok, "cancel must be accepted: {cancel_reply:?}");

    // The wake fires for the CANCELLED terminal with the folded reason.
    let text = tokio::time::timeout(Duration::from_secs(10), wake_rx.recv())
        .await
        .expect("wake must fire after cancellation")
        .expect("wake channel must not be dropped");
    assert!(
        text.contains("fixer-cancel"),
        "wake carries the alias: {text}"
    );
    assert!(text.contains("cancelled"), "wake state: {text}");
    assert!(
        text.contains("wrong branch"),
        "wake summary folds the cancel reason: {text}"
    );
    assert_eq!(
        wait_terminal(&harness.registry, &child_id).await,
        ChildSessionState::Cancelled
    );

    // Hygiene: release the still-parked provider sender task.
    gated.release_gate();
}

#[tokio::test(flavor = "multi_thread")]
async fn revive_of_terminal_background_child_completes_without_new_wake() {
    let ws = git_workspace();
    let provider = GatedScriptedProvider::new("revived answer");
    let (sched, mut wake_rx) = wake_channel(WAKE_INTERVAL);
    let harness = wire_consumer(ws.path(), provider.clone(), true, Some(sched.clone()));

    // Background child runs to completion (wake #1 fires).
    let (reply_tx, reply_rx) = oneshot::channel();
    harness
        .spawn_tx
        .send(spawn_request("first turn", None, None, reply_tx))
        .await
        .expect("spawn request accepted");
    let reply = tokio::time::timeout(Duration::from_secs(10), reply_rx)
        .await
        .expect("background reply must be immediate")
        .expect("reply channel must not be dropped");
    assert_eq!(reply.status, "running");
    let child_id = reply.child_session_id.clone();
    wait_for(|| provider.call_count() >= 1, "child must start its turn").await;
    provider.release_gate();
    let first_wake = tokio::time::timeout(Duration::from_secs(10), wake_rx.recv())
        .await
        .expect("wake #1 must fire after completion")
        .expect("wake channel must not be dropped");
    assert!(first_wake.contains("completed"), "wake #1: {first_wake}");
    assert_eq!(
        wait_terminal(&harness.registry, &child_id).await,
        ChildSessionState::Completed
    );

    // Consume wake #1 the way the CLI's Submit fold does (chunk C), so the
    // scheduler is RE-ARMED — a buggy revive-path hook would now be
    // observable instead of hidden behind the delivered flag.
    sched.note_input();

    // Revive the terminal child with the same provider (call 2 answers
    // immediately — the revived turn ends deterministically).
    let revive = handle_revive_request(
        harness.registry.clone(),
        offline_config(),
        ws.path().to_path_buf(),
        Some(harness.event_tx.clone()),
        Some(provider.clone() as Arc<dyn Provider>),
        child_id.clone(),
        "run the second turn".into(),
    )
    .await;
    assert!(revive.ok, "revive reply: {revive:?}");
    assert_eq!(revive.state, ChildSessionState::Completed);
    assert_eq!(revive.generation, Some(1));

    // NO additional wake: the reviving parent is mid-turn holding the
    // reply — the wake hook belongs to the background spawn arm only.
    assert_silent(
        &mut wake_rx,
        WAKE_INTERVAL * 6,
        "task_revive must never wake",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn two_background_terminals_in_window_coalesce_into_one_wake() {
    let ws = git_workspace();
    let provider = SequencedGatedProvider::new("coalesced answer", 2);
    let (sched, mut wake_rx) = wake_channel(COALESCE_INTERVAL);
    let harness = wire_consumer(ws.path(), provider.clone(), true, Some(sched));

    // Two background children; each parks its first call on its own gate.
    let mut child_ids = Vec::new();
    for alias in ["fixer-a", "fixer-b"] {
        let (reply_tx, reply_rx) = oneshot::channel();
        harness
            .spawn_tx
            .send(spawn_request("coalesce me", None, Some(alias), reply_tx))
            .await
            .expect("spawn request accepted");
        let reply = tokio::time::timeout(Duration::from_secs(10), reply_rx)
            .await
            .expect("background reply must be immediate")
            .expect("reply channel must not be dropped");
        assert_eq!(reply.status, "running", "inherited background: {reply:?}");
        child_ids.push(reply.child_session_id);
    }

    // Both children provably mid-turn, then both gates open within
    // microseconds — both terminals land inside one debounce window.
    wait_for(
        || provider.call_count() >= 2,
        "both children must start their turns",
    )
    .await;
    provider.release_all();

    // Both children terminal (both terminal hooks fired).
    for id in &child_ids {
        assert_eq!(
            wait_terminal(&harness.registry, id).await,
            ChildSessionState::Completed
        );
    }

    // Exactly ONE reconciled wake emerges for the pair.
    let text = tokio::time::timeout(Duration::from_secs(10), wake_rx.recv())
        .await
        .expect("one wake must fire for the coalesced terminals")
        .expect("wake channel must not be dropped");
    assert!(
        text.contains("fixer-a") || text.contains("fixer-b"),
        "wake carries one of the child refs: {text}"
    );
    assert_silent(
        &mut wake_rx,
        COALESCE_INTERVAL * 2,
        "two in-window terminals coalesce into exactly one wake",
    )
    .await;
}

/// P4 wiring chain, without a full TUI: build the tool with the scheduler's
/// `pause` closure exactly as `run_with_tui` does, execute it, and prove
/// the suppression + re-arm lifecycle end to end on real tokio time.
#[tokio::test(flavor = "multi_thread")]
async fn wait_for_user_pauses_wakes_until_next_user_input() {
    let (sched, mut wake_rx) = wake_channel(WAKE_INTERVAL);
    let pause_sched = sched.clone();
    let pause_hook: Arc<dyn Fn() + Send + Sync> = Arc::new(move || pause_sched.pause());
    let tool = WaitForUserTool::new(Some(pause_hook));

    // The tool executes immediately with success and its end-turn guidance.
    let call = ToolCall {
        id: "c1".into(),
        name: "wait_for_user".into(),
        input: serde_json::json!({}),
    };
    let res = tool.execute(&call).await;
    assert!(res.success, "error: {:?}", res.error);
    assert!(res.output.contains("Standing by for the user."));

    // A background terminal while paused: NO wake fires, even far past the
    // debounce window.
    sched.notify_terminal("fixer-x", "completed", "while standing by");
    assert_silent(
        &mut wake_rx,
        WAKE_INTERVAL * 6,
        "wait_for_user suppresses background wakes",
    )
    .await;

    // The user's next Submit (note_input at the choke point) releases the
    // latch and re-arms; a new terminal fires exactly one wake.
    sched.note_input();
    sched.notify_terminal("fixer-y", "completed", "after the user spoke");
    let text = tokio::time::timeout(Duration::from_secs(10), wake_rx.recv())
        .await
        .expect("wake must fire after note_input re-arms")
        .expect("wake channel must not be dropped");
    assert!(text.contains("fixer-y"), "re-armed wake: {text}");
    assert!(
        wake_rx.try_recv().is_err(),
        "exactly one wake after the re-arm"
    );
}
