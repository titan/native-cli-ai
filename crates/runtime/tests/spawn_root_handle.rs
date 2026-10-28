//! Live workspace-root handle (`Supervisor::workspace_root_handle`) —
//! runtime integration tests (D1b of the `/new` worktree-reset task).
//!
//! The subagent spawn consumer used to capture `workspace_root` as a
//! wiring-time `PathBuf` clone, so children spawned after a `/new` reset
//! (or a worktree switch) inherited a stale root: their persisted
//! `meta.workspace` pointed at the OLD session root, and their `.nca`
//! trees landed in the wrong place. The consumer now shares the
//! supervisor's live `Arc<RwLock<PathBuf>>` handle and reads it at each
//! spawn.
//!
//! These tests pin:
//! - the supervisor-side handle updates (`switch_to_worktree`,
//!   `reset_for_new_session` — covered structurally in supervisor.rs unit
//!   tests; here end-to-end),
//! - the end-to-end contract: a child spawned AFTER the root moved
//!   records the NEW root in its persisted session meta.
//!
//! No network: the child's provider is an injected scripted mock (the
//! `child_provider` test seam — production callers pass `None`).
//!
//! Run: `cargo test -p nca-runtime --test spawn_root_handle`

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use nca_common::config::{NcaConfig, PermissionMode};
use nca_common::message::Message;
use nca_common::tool::ToolDefinition;
use nca_core::provider::{Provider, ProviderError, StreamChunk};
use nca_core::tools::spawn_subagent::SpawnRequest;
use nca_core::workspace_fs::{RealFs, WorkspaceFs};
use nca_runtime::session_store::SessionStore;
use nca_runtime::supervisor::{Supervisor, SupervisorConfig, spawn_subagent_consumer};

fn offline_config() -> NcaConfig {
    let mut config = NcaConfig::default();
    config.provider.deepseek.api_key = Some("test-key".into());
    config.permissions.mode = PermissionMode::BypassPermissions;
    config.memory.context.auto_detect_context_window = false;
    config.memory.context.query_provider_models_api = false;
    config.memory.context.enable_auto_summarize = false;
    config
}

/// One-shot scripted provider: every chat call answers `answer` and stops.
struct OneShotProvider {
    answer: &'static str,
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

async fn create_sup(ws: &Path, session_id: &str) -> Supervisor {
    Supervisor::create(SupervisorConfig {
        config: offline_config(),
        workspace_root: ws.to_path_buf(),
        safe_mode: false,
        interactive_approvals: false,
        session_id: Some(session_id.into()),
        approval_handler: None,
        orchestration_context: None,
        agent_name: None,
        provider: None,
    })
    .await
    .expect("supervisor create must succeed")
}

fn spawn_child_and_wait(
    spawn_tx: &tokio::sync::mpsc::Sender<SpawnRequest>,
) -> impl std::future::Future<Output = nca_core::tools::spawn_subagent::SpawnResponse> + '_ {
    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    let req = SpawnRequest {
        task: "say done".into(),
        focus_files: Vec::new(),
        images: Vec::new(),
        use_worktree: false,
        background: Some(false),
        alias: None,
        provider_override: None,
        model_override: None,
        specialist: None,
        reply: reply_tx,
    };
    async move {
        spawn_tx.send(req).await.expect("spawn request accepted");
        tokio::time::timeout(Duration::from_secs(30), reply_rx)
            .await
            .expect("child reply must arrive (bounded)")
            .expect("reply channel must not close")
    }
}

// ---------------------------------------------------------------------------
// Child spawned after a `/new` reset records the reset (main) root
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn child_spawned_after_reset_records_new_root() {
    let ws = tempfile::tempdir().expect("tempdir");
    let wt = tempfile::tempdir().expect("worktree tempdir");
    let wt_path = wt.path().canonicalize().expect("canonicalize wt");
    let main_root = ws.path().canonicalize().expect("canonicalize ws");

    let mut sup = create_sup(ws.path(), "d1b-parent").await;
    // Ground the session in a worktree, then `/new`-reset back to main —
    // the exact production sequence from the incident.
    sup.switch_to_worktree(wt_path.clone(), "nca/d1b".into(), "main".into());
    sup.reset_for_new_session();

    // Production wiring: the consumer shares the supervisor's live handle
    // (read at each spawn), NOT a clone of the root.
    let (spawn_tx, spawn_rx) = tokio::sync::mpsc::channel(4);
    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(256);
    let parent_fs: Arc<dyn WorkspaceFs> = Arc::new(RealFs::new(ws.path().to_path_buf()));
    let _consumer = spawn_subagent_consumer(
        spawn_rx,
        "d1b-parent".into(),
        sup.workspace_root_handle(),
        sup.live_config(),
        Arc::new(std::sync::Mutex::new(vec![Message::user("parent context")])),
        Some(event_tx),
        sup.subagent_registry(),
        parent_fs,
        Some(Arc::new(OneShotProvider {
            answer: "child done",
        })),
        false,
        None,
        None,
        None,
    );
    // Drain lifecycle events so the bounded channel never backpressures.
    tokio::spawn(async move { while event_rx.recv().await.is_some() {} });

    let response = spawn_child_and_wait(&spawn_tx).await;
    assert_eq!(response.status, "completed", "child ran to completion");

    // The child's persisted meta records the RESET root (main workspace),
    // not the worktree the consumer was "born" under.
    let store = SessionStore::new(main_root.join(".nca/sessions"));
    let snapshot = store
        .load_snapshot(&response.child_session_id)
        .await
        .expect("child session json must be persisted in the MAIN sessions dir");
    assert_eq!(
        snapshot
            .workspace
            .canonicalize()
            .expect("canonical child ws"),
        main_root,
        "child meta.workspace must reflect the post-reset main root"
    );
    // The spawn response reports the same root to the model.
    assert_eq!(
        response.workspace,
        main_root.display().to_string(),
        "spawn response workspace must be the post-reset root"
    );
}
