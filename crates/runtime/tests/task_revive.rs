//! P2 chunk C integration — `task_revive`: resume a terminal child in its
//! retained session+worktree with a new prompt, bumping its generation.
//!
//! Covers the §3 semantics: running targets are cancelled first (port of
//! upstream `task-revive.ts` order), the lease is held for the whole
//! cancel-then-run sequence, the revived supervisor reuses the retained
//! worktree (the resume-fs-root finding pinned in supervisor.rs tests), and
//! the session json accumulates turns across generations (single-writer: the
//! revived child's own supervisor is the only writer).
//!
//! Hermetic: the scripted gated provider rides the `provider` seams
//! (`ChildSessionConfig::provider` for the spawn, `handle_revive_request`'s
//! provider param for the resume — production passes `None`).

use std::path::{Path, PathBuf};
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
use nca_runtime::subagent::{ChildSessionConfig, handle_revive_request, spawn_child_session};
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

/// Scripted provider: call 1 parks on the gate (provably mid-turn); every
/// later call answers immediately. The call counter is the deterministic
/// "turn in flight" anchor.
struct ScriptedProvider {
    release: Arc<tokio::sync::Notify>,
    answer: String,
    calls: std::sync::atomic::AtomicUsize,
}

impl ScriptedProvider {
    fn new(answer: &str) -> Arc<Self> {
        Arc::new(Self {
            release: Arc::new(tokio::sync::Notify::new()),
            answer: answer.into(),
            calls: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    fn call_count(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[async_trait]
impl Provider for ScriptedProvider {
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

/// Wire the REAL control consumer (task_cancel path) like the supervisor
/// does — P2 C2 extends its signature with the parent config + workspace
/// root (revive needs them to rebuild the child).
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

/// Spawn a gated child (mid-turn on provider call #1) and return
/// (child task, child session id) once the child is provably mid-turn:
/// live handles recorded AND the first provider call has landed.
///
/// The call-count anchor is REQUIRED, not an optimization:
/// `AgentLoop::run_turn` resets the cancel flag at turn start
/// (`agent.rs`: `self.cancel_flag.store(false)`), so a cancel/revive that
/// lands between handle-record and the child's turn-start would have its
/// flag flip silently wiped and the gated child would park forever. Waiting
/// for call #1 proves the child is past the reset, parked inside the stream
/// select where the 25ms cancel poll will observe the flip.
async fn spawn_gated_child(
    ws: &Path,
    registry: Arc<SubagentRegistry>,
    event_tx: mpsc::Sender<AgentEvent>,
    provider: Arc<dyn Provider>,
    scripted: &ScriptedProvider,
    alias: Option<&str>,
) -> (
    tokio::task::JoinHandle<Result<nca_runtime::subagent::ChildSessionResult, String>>,
    String,
) {
    let task = tokio::spawn(spawn_child_session(
        ChildSessionConfig {
            parent_session_id: "parent-1".into(),
            task: "first turn blocks on the gate".into(),
            workspace_root: ws.to_path_buf(),
            config: offline_config(),
            parent_summary: "[User]: do the thing".into(),
            use_worktree: true,
            focus_files: Vec::new(),
            images: Vec::new(),
            provider_override: None,
            model_override: None,
            specialist: None,
            alias: alias.map(String::from),
            registry: Some(registry.clone()),
            provider: Some(provider),
            plugins: None,
        },
        Some(event_tx),
    ));
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(entry) = registry
            .list()
            .into_iter()
            .find(|e| e.cancel_flag.is_some())
            && scripted.call_count() >= 1
        {
            return (task, entry.session_id);
        }
        assert!(
            Instant::now() < deadline,
            "child spawn must record its live handles"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn revive_cancelled_child_runs_new_prompt_in_retained_session() {
    let ws = git_workspace();
    let registry = Arc::new(SubagentRegistry::new());
    let (event_tx, _event_rx) = mpsc::channel(256);
    let provider = ScriptedProvider::new("second answer");
    let control_tx = wire_control_consumer(ws.path(), registry.clone(), Some(event_tx.clone()));

    // Child 1: gated mid-turn, then cancelled through the real consumer.
    let (child_task, child_id) = spawn_gated_child(
        ws.path(),
        registry.clone(),
        event_tx.clone(),
        provider.clone(),
        &provider,
        Some("fixer-x"),
    )
    .await;
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
    assert!(cancel_reply.ok, "{cancel_reply:?}");
    let result = tokio::time::timeout(Duration::from_secs(30), child_task)
        .await
        .expect("cancelled child must finish")
        .expect("no panic")
        .expect("session completes");
    assert_eq!(result.status, "cancelled");

    let store = SessionStore::new(ws.path().join(".nca").join("sessions"));
    let pre_revive = store.load(&child_id).await.expect("child json saved");
    let pre_revive_messages = pre_revive.messages.len();
    let entry = registry.get(&child_id).expect("entry");
    let retained_worktree: PathBuf = entry
        .worktree_path
        .clone()
        .expect("worktree path recorded at spawn")
        .into();
    assert!(retained_worktree.is_dir(), "retained worktree exists");
    assert_eq!(entry.alias.as_deref(), Some("fixer-x"));

    // Revive with a new prompt — the same scripted provider now answers
    // immediately (call #2), deterministically ending the revived turn.
    let reply = handle_revive_request(
        registry.clone(),
        offline_config(),
        ws.path().to_path_buf(),
        Some(event_tx.clone()),
        Some(provider.clone() as Arc<dyn Provider>),
        child_id.clone(),
        "run the second turn".into(),
    )
    .await;
    assert!(reply.ok, "revive reply: {reply:?}");
    assert_eq!(reply.generation, Some(1), "generation bumped 0 → 1");
    assert_eq!(reply.state, ChildSessionState::Completed);
    assert_eq!(
        reply.output.as_deref(),
        Some("second answer"),
        "output is the revived turn's final text"
    );

    // The child json accumulated turns (single-writer: the revived child's
    // own supervisor appended; both turns are on record).
    let revived = store.load(&child_id).await.expect("revived child json");
    assert!(
        revived.messages.len() > pre_revive_messages,
        "messages must grow across the revive: {} → {}",
        pre_revive_messages,
        revived.messages.len()
    );
    let texts: Vec<String> = revived
        .messages
        .iter()
        .map(|m| m.content.event_preview())
        .collect();
    assert!(
        texts.iter().any(|t| t.contains("run the second turn")),
        "the revive prompt is on record: {:?}",
        texts
    );
    assert!(
        texts.iter().any(|t| t.contains("second answer")),
        "the revived answer is on record"
    );
    assert_eq!(revived.meta.status, SessionStatus::Completed);

    // Registry: terminal again at generation 1, worktree identity unchanged.
    let entry = registry.get(&child_id).expect("entry");
    assert_eq!(entry.state, ChildSessionState::Completed);
    assert_eq!(entry.generation, 1);
    assert_eq!(
        entry.worktree_path.as_deref(),
        Some(retained_worktree.display().to_string()).as_deref(),
        "revive reuses the retained worktree (never re-creates)"
    );
    assert!(
        retained_worktree.is_dir(),
        "worktree still exists after revive"
    );
    // Resolving by alias still finds the revived task.
    let by_alias = registry
        .resolve("fixer-x")
        .expect("resolve")
        .expect("found");
    assert_eq!(by_alias.session_id, child_id);
}

#[tokio::test(flavor = "multi_thread")]
async fn revive_running_child_cancels_first_then_revives() {
    let ws = git_workspace();
    let registry = Arc::new(SubagentRegistry::new());
    let (event_tx, _event_rx) = mpsc::channel(256);
    let provider = ScriptedProvider::new("after cancel answer");

    // Child still RUNNING (mid-gate) when the revive lands: the revive must
    // cancel it first, wait for the terminal fold, then resume + run.
    let (child_task, child_id) = spawn_gated_child(
        ws.path(),
        registry.clone(),
        event_tx.clone(),
        provider.clone(),
        &provider,
        None,
    )
    .await;

    let reply = handle_revive_request(
        registry.clone(),
        offline_config(),
        ws.path().to_path_buf(),
        Some(event_tx.clone()),
        Some(provider.clone() as Arc<dyn Provider>),
        child_id.clone(),
        "pivot".into(),
    )
    .await;
    assert!(reply.ok, "revive reply: {reply:?}");
    assert_eq!(reply.generation, Some(1));
    assert_eq!(reply.state, ChildSessionState::Completed);
    assert_eq!(reply.output.as_deref(), Some("after cancel answer"));

    // The first (cancelled) run finished — and there is exactly ONE final
    // state at generation 1.
    let first = tokio::time::timeout(Duration::from_secs(30), child_task)
        .await
        .expect("cancelled child must finish")
        .expect("no panic")
        .expect("session completes");
    assert_eq!(first.status, "cancelled");
    let entry = registry.get(&child_id).expect("entry");
    assert_eq!(entry.state, ChildSessionState::Completed);
    assert_eq!(entry.generation, 1);
    assert_eq!(
        entry.result_summary.as_deref(),
        Some("after cancel answer"),
        "the generation-1 summary is the revived turn's output"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn revive_unknown_id_through_consumer_is_unknown() {
    let ws = tempfile::tempdir().expect("tempdir");
    let registry = Arc::new(SubagentRegistry::new());
    let control_tx = wire_control_consumer(ws.path(), registry, None);

    let (reply_tx, reply_rx) = oneshot::channel();
    control_tx
        .send(SubagentControlRequest::Revive {
            session_id: "ghost".into(),
            prompt: "finish the tests".into(),
            reply: reply_tx,
        })
        .await
        .expect("control channel live");
    let resp = reply_rx.await.expect("revive reply");
    assert!(!resp.ok);
    let error = resp.error_message.expect("error");
    assert!(
        error.starts_with("unknown subagent task id 'ghost';"),
        "error must name the offending id: {error}"
    );
    assert!(
        error.contains("no subagent tasks are registered"),
        "empty-registry hint must say so plainly: {error}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn revive_while_lease_held_reports_in_flight() {
    let ws = tempfile::tempdir().expect("tempdir");
    let registry = Arc::new(SubagentRegistry::new());
    registry.record_spawned("p", "c1", "t", "/ws".into(), None);
    registry.record_terminal("c1", ChildSessionState::Cancelled, Some("c".into()));

    // Simulate an in-flight control operation (e.g. a revive blocked on a
    // gated child): the lease is held for the whole cancel-then-run.
    let _lease = registry.try_acquire_lease("c1").expect("lease");
    let reply = handle_revive_request(
        registry.clone(),
        offline_config(),
        ws.path().to_path_buf(),
        None,
        None,
        "c1".into(),
        "again".into(),
    )
    .await;
    assert!(!reply.ok, "{reply:?}");
    assert_eq!(
        reply.note.as_deref(),
        Some("another control operation is in flight for this task")
    );
    // The registry was not mutated by the refused revive.
    assert_eq!(
        registry.get("c1").map(|e| e.state),
        Some(ChildSessionState::Cancelled)
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn revive_with_unloadable_session_fails_and_registry_stays_terminal() {
    // Registry knows the task, but its session json/event log are gone
    // (e.g. wiped workspace): resume fails loudly and the entry keeps its
    // terminal state — no zombie Running entry.
    let ws = tempfile::tempdir().expect("tempdir");
    let registry = Arc::new(SubagentRegistry::new());
    registry.record_spawned("p", "c1", "t", "/ws".into(), None);
    registry.record_terminal("c1", ChildSessionState::Completed, Some("done".into()));

    let reply = handle_revive_request(
        registry.clone(),
        offline_config(),
        ws.path().to_path_buf(),
        None,
        None,
        "c1".into(),
        "again".into(),
    )
    .await;
    assert!(!reply.ok, "{reply:?}");
    let error = reply.error_message.as_deref().expect("error message");
    assert!(!error.is_empty());
    let entry = registry.get("c1").expect("entry");
    assert_eq!(entry.state, ChildSessionState::Completed);
    assert_eq!(entry.generation, 0, "no generation bump on failed resume");
}
