//! P2 Phase B Lane B — turn-end commit barrier integration tests (T16/T17).
//!
//! Spec: `docs/plans/p2-phase-b-design.md` §3–§4. Harness style mirrors
//! `crates/runtime/tests/inbox.rs`: real `Supervisor` with a deterministic
//! offline config and a scripted provider swapped in after `create()`.
//!
//! T16: one scripted turn leaves the full event bracket (TurnStarted →
//! MessageRecorded user → MessageRecorded assistant → TurnCompleted) in the
//! log with strictly increasing ids.
//!
//! T17: `run_turn` returns only AFTER the TurnCompleted line is durable in
//! the log and the commit watch has fired — no sleeps in the assertion path.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use nca_common::config::{NcaConfig, PermissionMode};
use nca_common::event::AgentEvent;
use nca_common::message::{Message, Role};
use nca_common::tool::ToolDefinition;
use nca_core::provider::{Provider, ProviderError, StreamChunk};
use nca_runtime::session_store::read_event_log;
use nca_runtime::session_utils::spawn_event_fanout;
use nca_runtime::supervisor::{Supervisor, SupervisorConfig};

fn offline_config() -> NcaConfig {
    let mut config = NcaConfig::default();
    config.provider.deepseek.api_key = Some("test-key".into());
    config.permissions.mode = PermissionMode::BypassPermissions;
    config.memory.context.auto_detect_context_window = false;
    config.memory.context.query_provider_models_api = false;
    config.memory.context.enable_auto_summarize = false;
    config
}

/// Single-round text provider (no gating needed here).
struct OneShotProvider {
    _recorded: Mutex<Vec<Vec<Message>>>,
}

impl OneShotProvider {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            _recorded: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait]
impl Provider for OneShotProvider {
    async fn chat(
        &self,
        messages: &[Message],
        _tools: &[ToolDefinition],
        _model: &str,
        _workspace_root: &Path,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError> {
        self._recorded.lock().unwrap().push(messages.to_vec());
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

/// Mirrors service.rs wiring: take handle, take commit sender (marks wired),
/// spawn the fanout on event_rx + log path. Returns the log path and a
/// clone of the supervisor's commit watch receiver for assertions.
async fn wire_fanout(sup: &mut Supervisor) -> (PathBuf, tokio::sync::watch::Receiver<u64>) {
    let watch_rx = sup.turn_commit_rx();
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
    spawn_event_fanout(
        event_rx,
        log_path.clone(),
        None,
        None,
        None,
        Some(commit_tx),
    );
    (log_path, watch_rx.expect("commit rx"))
}

// T16: the log contains the full turn bracket with strictly increasing ids.
#[tokio::test(flavor = "multi_thread")]
async fn turn_bracket_written_with_increasing_ids() {
    let ws = tempfile::tempdir().expect("tempdir");
    let mut sup = Supervisor::create(SupervisorConfig {
        config: offline_config(),
        workspace_root: ws.path().to_path_buf(),
        safe_mode: true,
        interactive_approvals: false,
        session_id: Some("t16-bracket".into()),
        approval_handler: None,
        orchestration_context: None,
        agent_name: None,
    })
    .await
    .expect("supervisor create");
    sup.agent_mut().replace_provider(OneShotProvider::new());
    sup.set_session_title(Some("t16".into()));

    let (log_path, _watch_rx) = wire_fanout(&mut sup).await;
    let out = sup.run_turn("hello").await.expect("turn succeeds");
    assert_eq!(out, "answer");

    let envelopes = read_event_log(&log_path);
    let kinds: Vec<&str> = envelopes
        .iter()
        .map(|e| match &e.event {
            AgentEvent::TurnStarted { .. } => "TurnStarted",
            AgentEvent::MessageRecorded { message } => match message.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                _ => "other",
            },
            AgentEvent::TurnCompleted { .. } => "TurnCompleted",
            _ => "other",
        })
        .collect();
    assert!(
        kinds.contains(&"TurnStarted")
            && kinds.contains(&"user")
            && kinds.contains(&"assistant")
            && kinds.contains(&"TurnCompleted"),
        "full bracket must be present: {kinds:?}"
    );
    for pair in envelopes.windows(2) {
        assert!(
            pair[0].id < pair[1].id,
            "ids strictly increasing: {:?}",
            envelopes.iter().map(|e| e.id).collect::<Vec<_>>()
        );
    }
}

// T17: run_turn returns only after the commit barrier — the log already holds
// TurnCompleted and the watch fired, with no sleeping in the assertion path.
#[tokio::test(flavor = "multi_thread")]
async fn run_turn_returns_after_turn_committed() {
    let ws = tempfile::tempdir().expect("tempdir");
    let mut sup = Supervisor::create(SupervisorConfig {
        config: offline_config(),
        workspace_root: ws.path().to_path_buf(),
        safe_mode: true,
        interactive_approvals: false,
        session_id: Some("t17-barrier".into()),
        approval_handler: None,
        orchestration_context: None,
        agent_name: None,
    })
    .await
    .expect("supervisor create");
    sup.agent_mut().replace_provider(OneShotProvider::new());
    sup.set_session_title(Some("t17".into()));

    let (log_path, watch_rx) = wire_fanout(&mut sup).await;

    // Pre-turn invariant: nothing committed yet.
    assert_eq!(*watch_rx.borrow(), 0);

    sup.run_turn("hello").await.expect("turn succeeds");

    // Barrier proof — checked IMMEDIATELY after run_turn, no sleep/wait.
    assert!(
        *watch_rx.borrow() >= 1,
        "commit watch must have fired before run_turn returned"
    );
    let envelopes = read_event_log(&log_path);
    assert!(
        envelopes
            .iter()
            .any(|e| matches!(e.event, AgentEvent::TurnCompleted { .. })),
        "TurnCompleted must already be durable in the log after run_turn"
    );
}
