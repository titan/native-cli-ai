//! Loop-breaker child-session classification — regression tests.
//!
//! A child turn ended by the consecutive-tool-failure breaker returns
//! `Ok(text)` from `run_turn`, so before the `[nca:loop-breaker]` marker it
//! folded into status `"completed"` — the parent's wake note and
//! `task_status` then advertised a pathologically-stopped task as completed.
//! The marker (stamped at position 0 of the early-final text by
//! `nca_core::agent_driver`) lets `run_prepared_child` classify such turns
//! as `"error"` (`ChildSessionState::Failed`), while a normal text-final
//! child stays `"completed"`.
//!
//! No network: the child's provider is a scripted in-process mock injected
//! through the `spawn_subagent_consumer` provider seam; the failing tool is
//! a plain `read_file` on a nonexistent path (deterministic, no side
//! effects). Same hermetic scaffolding as `background_spawn.rs`.
//!
//! Run: `cargo test -p nca-runtime --test breaker_classification`

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use nca_common::config::{NcaConfig, PermissionMode};
use nca_common::message::Message;
use nca_common::session::ChildSessionState;
use nca_common::tool::{ToolCall, ToolDefinition};
use nca_core::agent::LOOP_BREAKER_MARKER;
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

/// Scripted provider that NEVER converges: every `chat()` call issues one
/// `read_file` tool call on a path that does not exist. Three consecutive
/// all-failed same-tool steps trip the driver's loop breaker, ending the
/// child's turn with the marker-prefixed early-final text.
struct AlwaysFailingToolProvider;

#[async_trait]
impl Provider for AlwaysFailingToolProvider {
    async fn chat(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        _model: &str,
        _workspace_root: &Path,
    ) -> Result<mpsc::Receiver<StreamChunk>, ProviderError> {
        let (tx, rx) = mpsc::channel(64);
        tokio::spawn(async move {
            let _ = tx
                .send(StreamChunk::ToolUse(ToolCall {
                    id: "call-missing-read".into(),
                    name: "read_file".into(),
                    input: serde_json::json!({
                        "path": "definitely-missing-9f2c.txt"
                    }),
                }))
                .await;
            let _ = tx
                .send(StreamChunk::Finish {
                    reason: "tool_calls".into(),
                })
                .await;
        });
        Ok(rx)
    }
}

/// Plain text-final provider: one call, one answer, turn completes normally.
struct TextAnswerProvider {
    answer: &'static str,
}

#[async_trait]
impl Provider for TextAnswerProvider {
    async fn chat(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        _model: &str,
        _workspace_root: &Path,
    ) -> Result<mpsc::Receiver<StreamChunk>, ProviderError> {
        let (tx, rx) = mpsc::channel(64);
        let answer = self.answer;
        tokio::spawn(async move {
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
        // No worktree: the classification under test is independent of the
        // child's checkout, and skipping git keeps the test hermetic.
        use_worktree: false,
        background: Some(background),
        alias: alias.map(String::from),
        provider_override: None,
        model_override: None,
        specialist: None,
        reply,
    }
}

/// Enabled wake scheduler recording delivered wake texts (same seam as
/// `background_spawn.rs`); the debounce is short so tests assert quickly.
fn wake_channel() -> (WakeScheduler, mpsc::UnboundedReceiver<String>) {
    let (tx, rx) = mpsc::unbounded_channel();
    let trigger: WakeTrigger = Arc::new(move |text: &str| {
        let _ = tx.send(text.to_string());
    });
    (
        WakeScheduler::new(
            true,
            Duration::from_millis(50),
            Duration::from_secs(60),
            trigger,
        ),
        rx,
    )
}

/// Wire the real spawn consumer with a provider seam (mirrors
/// `background_spawn.rs::wire_consumer`, minus the event tap — these tests
/// assert on the reply, the registry, and the wake texts).
#[allow(clippy::type_complexity)]
fn wire_consumer(
    ws: &Path,
    provider: Arc<dyn Provider>,
    wake: Option<WakeScheduler>,
) -> (mpsc::Sender<SpawnRequest>, Arc<SubagentRegistry>) {
    let registry = Arc::new(SubagentRegistry::new());
    let (spawn_tx, spawn_rx) = mpsc::channel(4);
    let (event_tx, mut event_rx) = mpsc::channel(256);
    let parent_fs: Arc<dyn WorkspaceFs> = Arc::new(RealFs::new(ws.to_path_buf()));
    let _consumer = spawn_subagent_consumer(
        spawn_rx,
        "parent-breaker".into(),
        ws.to_path_buf(),
        Arc::new(std::sync::RwLock::new(offline_config())),
        Arc::new(std::sync::Mutex::new(vec![Message::user("parent context")])),
        Some(event_tx),
        registry.clone(),
        parent_fs,
        Some(provider),
        false,
        wake,
        None,
    );
    // Drain the event tap so the bounded channel never back-pressures the
    // child's lifecycle folding.
    tokio::spawn(async move { while event_rx.recv().await.is_some() {} });
    (spawn_tx, registry)
}

/// Poll the registry until the child reaches a terminal state (bounded).
async fn await_terminal(registry: &SubagentRegistry, child_id: &str) -> ChildSessionState {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let entry = registry.get(child_id).expect("registry entry");
        if entry.state.is_terminal() {
            return entry.state;
        }
        assert!(
            Instant::now() < deadline,
            "child must reach a terminal state"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// The core regression: a foreground child that trips the loop breaker must
/// come back with status `"error"` (not `"completed"`), the registry state
/// `Failed`, and the marker-prefixed breaker text as the output.
#[tokio::test(flavor = "multi_thread")]
async fn breaker_child_classifies_as_failed_not_completed() {
    let ws = tempfile::tempdir().expect("tempdir");
    let provider: Arc<dyn Provider> = Arc::new(AlwaysFailingToolProvider);
    let (spawn_tx, registry) = wire_consumer(ws.path(), provider, None);

    let (reply_tx, reply_rx) = oneshot::channel();
    spawn_tx
        .send(spawn_request(
            "read a file that does not exist",
            false,
            None,
            reply_tx,
        ))
        .await
        .expect("spawn request accepted");

    let reply = tokio::time::timeout(Duration::from_secs(30), reply_rx)
        .await
        .expect("foreground reply must arrive")
        .expect("reply channel must not be dropped");

    assert_eq!(
        reply.status, "error",
        "breaker-stopped child must read as error, not completed: {reply:?}"
    );
    assert!(
        reply.output.starts_with(LOOP_BREAKER_MARKER),
        "output must carry the classification marker at position 0: {}",
        reply.output
    );
    assert!(
        reply.output.contains("failed 3 times consecutively"),
        "breaker text expected: {}",
        reply.output
    );

    let state = await_terminal(&registry, &reply.child_session_id).await;
    assert_eq!(
        state,
        ChildSessionState::Failed,
        "registry/task_status state must be Failed"
    );
}

/// Control: a normal text-final child still classifies as completed — the
/// marker check must not over-match ordinary turns.
#[tokio::test(flavor = "multi_thread")]
async fn normal_child_still_classifies_as_completed() {
    let ws = tempfile::tempdir().expect("tempdir");
    let provider: Arc<dyn Provider> = Arc::new(TextAnswerProvider {
        answer: "did the thing",
    });
    let (spawn_tx, registry) = wire_consumer(ws.path(), provider, None);

    let (reply_tx, reply_rx) = oneshot::channel();
    spawn_tx
        .send(spawn_request("do something", false, None, reply_tx))
        .await
        .expect("spawn request accepted");

    let reply = tokio::time::timeout(Duration::from_secs(30), reply_rx)
        .await
        .expect("foreground reply must arrive")
        .expect("reply channel must not be dropped");

    assert_eq!(reply.status, "completed", "control: {reply:?}");
    assert_eq!(reply.output, "did the thing");

    let state = await_terminal(&registry, &reply.child_session_id).await;
    assert_eq!(state, ChildSessionState::Completed);
}

/// End-to-end over the P3 wake path: a background breaker child must wake
/// the idle parent with the label `failed` — the exact wording that used to
/// (misleadingly) read `completed`.
#[tokio::test(flavor = "multi_thread")]
async fn background_breaker_child_wakes_parent_with_failed_label() {
    let ws = tempfile::tempdir().expect("tempdir");
    let provider: Arc<dyn Provider> = Arc::new(AlwaysFailingToolProvider);
    let (sched, mut wake_rx) = wake_channel();
    let (spawn_tx, registry) = wire_consumer(ws.path(), provider, Some(sched));

    let (reply_tx, reply_rx) = oneshot::channel();
    spawn_tx
        .send(spawn_request(
            "read a file that does not exist",
            true,
            Some("fixer-fail"),
            reply_tx,
        ))
        .await
        .expect("spawn request accepted");

    // Background reply is immediate; the child then runs detached.
    let reply = tokio::time::timeout(Duration::from_secs(10), reply_rx)
        .await
        .expect("background reply must be immediate")
        .expect("reply channel must not be dropped");
    assert_eq!(reply.status, "running", "immediate reply: {reply:?}");
    let child_id = reply.child_session_id.clone();

    let state = await_terminal(&registry, &child_id).await;
    assert_eq!(state, ChildSessionState::Failed);

    // The debounced wake must label the child `failed` and carry the
    // marker-prefixed summary — never `completed`.
    let wake_text = tokio::time::timeout(Duration::from_secs(10), wake_rx.recv())
        .await
        .expect("terminal child must wake the parent")
        .expect("wake channel must not be dropped");
    assert!(
        wake_text.contains("fixer-fail reached failed:"),
        "wake must label the breaker child failed: {wake_text}"
    );
    assert!(
        !wake_text.contains("reached completed"),
        "wake must NOT label the breaker child completed: {wake_text}"
    );
    assert!(
        wake_text.contains(LOOP_BREAKER_MARKER),
        "wake summary must carry the breaker marker: {wake_text}"
    );
}
