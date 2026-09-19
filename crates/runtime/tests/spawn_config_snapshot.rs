//! Spawn-time config snapshot (`Supervisor::live_config`) — runtime
//! integration tests.
//!
//! Pins the live-handle contract between the supervisor and the subagent
//! spawn consumer that replaced the old wiring-time `NcaConfig` clone
//! (which froze provider/model routing at consumer creation, so children
//! spawned after an in-session `/model`, `/provider`, or agent switch
//! inherited stale routing):
//!
//! - S1: `apply_agent_profile` / `apply_nca_config` refresh the shared
//!   snapshot after a successful rebuild; profile overrides never leak
//!   into it (the snapshot mirrors `self.config`, the clean base — the
//!   same purity contract as `agent_profile_config_purity.rs`).
//! - S2: end-to-end — a child spawned AFTER an in-session model switch
//!   (apply_nca_config) builds against the NEW config: the child's
//!   persisted session meta carries the new model.
//!
//! No network: the parent builds providers from a dummy-key config and
//! never chats; the child's provider is an injected scripted mock (the
//! `child_provider` test seam — production callers pass `None`).
//!
//! Run: `cargo test -p nca-runtime --test spawn_config_snapshot`

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use nca_common::config::{AgentProfileConfig, NcaConfig, PermissionMode};
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

async fn create_sup(ws: &Path, session_id: &str, config: NcaConfig) -> Supervisor {
    let mut sup = Supervisor::create(SupervisorConfig {
        config,
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
    .expect("supervisor create must succeed");
    sup.set_session_title(Some(session_id.into()));
    sup
}

fn live_model(handle: &std::sync::RwLock<NcaConfig>) -> String {
    let guard = handle
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.model.default_model.clone()
}

// ---------------------------------------------------------------------------
// S1 — apply_* refresh the shared snapshot; profile overrides never leak
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn s1_apply_paths_refresh_live_config_snapshot() {
    let ws = tempfile::tempdir().expect("tempdir");
    let mut config = offline_config();
    config.agents.insert(
        "s1-persona".into(),
        AgentProfileConfig {
            model: Some("s1-profile-model".into()),
            system_prompt: Some("You are S1-PERSONA.".into()),
            ..Default::default()
        },
    );
    let base_model = config.model.default_model.clone();

    let mut sup = create_sup(ws.path(), "s1", config).await;
    let live = sup.live_config();
    assert_eq!(
        live_model(&live),
        base_model,
        "snapshot starts at the created config"
    );

    // Agent switch: the SESSION switches to the profile model, but the
    // spawn snapshot must mirror the clean base — children re-apply
    // routing from their own `specialist` argument at spawn time.
    sup.apply_agent_profile(Some("s1-persona"))
        .expect("agent switch succeeds");
    assert_eq!(
        sup.model, "s1-profile-model",
        "the session itself switched to the profile model"
    );
    assert_eq!(
        live_model(&live),
        base_model,
        "profile overrides must never leak into the spawn snapshot"
    );

    // `/model`-style switch: the snapshot follows.
    let mut config_b = offline_config();
    config_b.provider.set_model_for_default("s2-child-model");
    config_b.sync_default_model_from_provider();
    sup.apply_nca_config(config_b)
        .expect("config switch succeeds");
    assert_eq!(
        live_model(&live),
        "s2-child-model",
        "apply_nca_config must refresh the spawn snapshot"
    );
    assert_eq!(sup.model, "s2-child-model");
}

// ---------------------------------------------------------------------------
// S2 — a child spawned after the switch builds against the NEW config
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn s2_child_spawned_after_switch_uses_new_config() {
    let ws = tempfile::tempdir().expect("tempdir");
    let mut sup = create_sup(ws.path(), "s2-parent", offline_config()).await;

    // Production wiring: the consumer shares the supervisor's live handle.
    let live = sup.live_config();
    let (spawn_tx, spawn_rx) = tokio::sync::mpsc::channel(4);
    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(256);
    let parent_fs: Arc<dyn WorkspaceFs> = Arc::new(RealFs::new(ws.path().to_path_buf()));
    let _consumer = spawn_subagent_consumer(
        spawn_rx,
        "s2-parent".into(),
        ws.path().to_path_buf(),
        live,
        Arc::new(std::sync::Mutex::new(vec![Message::user("parent context")])),
        Some(event_tx),
        sup.subagent_registry(),
        parent_fs,
        // Child provider seam: hermetic scripted provider (production: None).
        Some(Arc::new(OneShotProvider {
            answer: "child done",
        })),
        false,
        None,
        None,
    );
    // Drain lifecycle events so the bounded channel never backpressures.
    tokio::spawn(async move { while event_rx.recv().await.is_some() {} });

    // In-session `/model` AFTER the consumer was wired.
    let mut config_b = offline_config();
    config_b.provider.set_model_for_default("s2-child-model");
    config_b.sync_default_model_from_provider();
    sup.apply_nca_config(config_b)
        .expect("model switch succeeds");

    let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
    spawn_tx
        .send(SpawnRequest {
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
        })
        .await
        .expect("spawn request accepted");

    let response = tokio::time::timeout(Duration::from_secs(30), reply_rx)
        .await
        .expect("child reply must arrive (bounded)")
        .expect("reply channel must not close");
    assert_eq!(response.status, "completed", "child ran to completion");
    assert_eq!(response.output.trim(), "child done");

    // The child's persisted session was built from the NEW config: its
    // meta carries the post-switch model, not the wiring-time snapshot.
    let store = SessionStore::new(ws.path().join(".nca/sessions"));
    let snapshot = store
        .load_snapshot(&response.child_session_id)
        .await
        .expect("child session json must be persisted");
    assert_eq!(
        snapshot.model, "s2-child-model",
        "a child spawned after apply_nca_config must inherit the new routing"
    );
}
