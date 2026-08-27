//! Regression: mid-session `/mount` persistence must survive a later
//! whole-config save, and `/unmount` must remove the entry deterministically.
//!
//! The bug: `Supervisor::mount_path` updated only the live `RealFs` and handed
//! persistence to a dropped join handle. Any subsequent REPL action that
//! clones `Supervisor::config` and calls `save_workspace_file` (`/model`,
//! `/thinking`, `/set-editor`, …) serialized the *stale pre-mount* snapshot,
//! silently erasing the freshly written `extra_paths` — no warnings anywhere.
//! The fix syncs the in-memory config and awaits the blocking write.
//!
//! Run: `cargo test -p nca-runtime --test mount_config_persistence`

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
use nca_common::config::{NcaConfig, PermissionMode};
use nca_common::message::Message;
use nca_common::tool::ToolDefinition;
use nca_core::provider::{Provider, ProviderError, StreamChunk};
use nca_runtime::supervisor::{Supervisor, SupervisorConfig};

// ---------------------------------------------------------------------------
// Env isolation (local copy of the crate-internal test_util pattern; helpers
// are not shared across crates, and this integration binary owns its process).
// ---------------------------------------------------------------------------

static ENV_MUTEX: Mutex<()> = Mutex::new(());

/// RAII guard binding before any env observation; restores on drop.
struct TestEnvGuard {
    previous: Vec<(String, Option<std::ffi::OsString>)>,
    _lock: MutexGuard<'static, ()>,
}

impl TestEnvGuard {
    fn set(vars: &[(&str, &str)]) -> Self {
        let lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let mut previous = Vec::new();
        for (key, value) in vars {
            previous.push((key.to_string(), std::env::var_os(key)));
            // SAFETY: the mutex serializes env mutation within this binary.
            unsafe { std::env::set_var(key, value) };
        }
        Self {
            previous,
            _lock: lock,
        }
    }
}

impl Drop for TestEnvGuard {
    fn drop(&mut self) {
        for (key, value) in self.previous.drain(..) {
            match value {
                // SAFETY: still holding the env mutex.
                Some(value) => unsafe { std::env::set_var(&key, value) },
                None => unsafe { std::env::remove_var(&key) },
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Scaffolding (mirrors provider_injection.rs)
// ---------------------------------------------------------------------------

/// A provider that must never be called by these tests.
struct UnusedProvider;

#[async_trait]
impl Provider for UnusedProvider {
    async fn chat(
        &self,
        _messages: &[Message],
        _tools: &[ToolDefinition],
        _model: &str,
        _workspace_root: &Path,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError> {
        unreachable!("mount/config persistence must not call the provider")
    }
}

fn offline_config_no_key() -> NcaConfig {
    let mut config = NcaConfig::default();
    config.permissions.mode = PermissionMode::BypassPermissions;
    config.memory.context.auto_detect_context_window = false;
    config.memory.context.query_provider_models_api = false;
    config.memory.context.enable_auto_summarize = false;
    config
}

async fn create_sup(ws: &Path, session_id: &str) -> Supervisor {
    Supervisor::create(SupervisorConfig {
        config: offline_config_no_key(),
        workspace_root: ws.to_path_buf(),
        safe_mode: true,
        interactive_approvals: false,
        session_id: Some(session_id.into()),
        approval_handler: None,
        orchestration_context: None,
        agent_name: None,
        provider: Some(Arc::new(UnusedProvider)),
    })
    .await
    .expect("supervisor create must succeed")
}

/// Canonical form the supervisor persists (`RealFs::mount_path` canonicalizes).
fn canonical(p: &Path) -> PathBuf {
    p.canonicalize().expect("canonicalize temp dir")
}

#[tokio::test(flavor = "multi_thread")]
async fn mounted_extra_paths_survive_whole_config_save_and_unmount_removes() {
    let home = tempfile::tempdir().expect("home tempdir");
    let xdg = tempfile::tempdir().expect("xdg tempdir");
    // Isolate BEFORE any config read so global files / ambient env cannot pollute.
    let _env = TestEnvGuard::set(&[
        ("HOME", home.path().to_str().unwrap()),
        ("XDG_CONFIG_HOME", xdg.path().to_str().unwrap()),
    ]);

    let ws = tempfile::tempdir().expect("workspace tempdir");
    let ext = tempfile::tempdir().expect("external dir tempdir");
    let mut sup = create_sup(ws.path(), "mnt-regression").await;

    // ── 1. Mount an external directory ──────────────────────────────────────
    sup.mount_path(ext.path())
        .await
        .expect("mount succeeds for an existing external dir");

    assert_eq!(
        sup.config().extra_paths,
        vec![canonical(ext.path())],
        "in-memory config must reflect the live mount list"
    );

    // Persistence is awaited inside mount_path → deterministic disk read.
    let disk = NcaConfig::load_for_workspace(ws.path()).expect("reload after mount");
    assert_eq!(
        disk.extra_paths,
        vec![canonical(ext.path())],
        "extra_paths must be persisted to the workspace-local file"
    );

    // ── 2. Simulate a later whole-config save (`/model`-style flow) ─────────
    // This is exactly what repl.rs does for those commands: clone the runtime
    // snapshot, tweak an unrelated field, re-apply, then save_workspace_file.
    let mut cfg = sup.config().clone();
    cfg.model.enable_thinking = !cfg.model.enable_thinking;
    sup.apply_nca_config(cfg)
        .expect("keyless deepseek config rebuilds fine (validated lazily)");
    sup.config()
        .save_workspace_file(ws.path())
        .expect("whole-config workspace save");

    let after_save = NcaConfig::load_for_workspace(ws.path()).expect("reload after save");
    assert_eq!(
        after_save.extra_paths,
        vec![canonical(ext.path())],
        "REGRESSION: a later whole-config save erased the persisted mount \
         because the in-memory snapshot was stale"
    );

    // ── 3. Unmount removes from memory AND from disk ────────────────────────
    sup.unmount_path(ext.path()).await.expect("unmount");
    assert!(
        sup.config().extra_paths.is_empty(),
        "in-memory config must drop the unmounted path"
    );
    // Awaited persistence → no polling needed.
    let final_disk = NcaConfig::load_for_workspace(ws.path()).expect("reload after unmount");
    assert!(
        final_disk.extra_paths.is_empty(),
        "unmounted path must be removed from the persisted config"
    );
}
