//! `/sandbox` session-level toggle round-trip (P5 wiring follow-up).
//!
//! Pins the seam the `/sandbox off | on | toggle` command drives:
//! `PtyManager::set_sandbox_config` must flip confinement live — same
//! process, same manager, no restart — and `sandbox_confined()` must
//! report the policy state at each step. Round-trip: Required (confined)
//! → Off (unconfined) → Required (confined again).
//!
//! Kernel-gated: skips when Landlock is unavailable (same guard style as
//! `sandbox_wiring.rs`).

use nca_common::config::{SandboxConfig, SandboxMode};
use nca_common::event::AgentEvent;
use nca_core::tools::ToolProgress;
use nca_runtime::pty::PtyManager;
use nca_runtime::sandbox::backend_supported;

fn progress(name: &str) -> ToolProgress {
    let (tx, _rx) = tokio::sync::mpsc::channel::<AgentEvent>(8);
    ToolProgress::new(name, tx)
}

/// `Required` mode, no extra roots — mirrors `sandbox_wiring.rs`'s fixture
/// so the outside-write probe targets a path that must stay read-only
/// under the wired policy.
fn required_config() -> SandboxConfig {
    SandboxConfig {
        mode: SandboxMode::Required,
        ro_paths: Vec::new(),
        rw_paths: Vec::new(),
        net: true,
        env_allow: nca_common::config::default_sandbox_env_allow(),
        inherit_mounts: true,
        host_audio: false,
        host_dbus_session: false,
        host_xdg_runtime: false,
    }
}

/// `Off` mode — same shape, only the mode differs.
fn off_config() -> SandboxConfig {
    SandboxConfig {
        mode: SandboxMode::Off,
        ..required_config()
    }
}

/// Round-trip: `Required` confines → `Off` releases → `Required` confines
/// again, all on one `PtyManager` without rebuilding it.
///
/// The outside-write probe targets a scratch path under `$TMPDIR` (a default
/// rw root, unlike bare `$HOME`). Under `Required` the policy's rw roots are
/// workspace + system tmp — but the policy also strips the env down to the
/// allowlist, so `$TMPDIR` is unset in the child and the probe falls back to
/// the system `/tmp`, which is NOT in the policy's rw set → write fails.
/// Unconfined, the child inherits `$TMPDIR` → write succeeds. That contrast
/// is what the round-trip pins. (Bare `$HOME` is avoided because the hosting
/// process itself may be Landlock-confined — nca dogfooding — which would
/// make even the unconfined phase fail.)
#[tokio::test(flavor = "multi_thread")]
async fn sandbox_toggle_round_trip_flips_confinement_live() {
    if !backend_supported() {
        eprintln!("SKIP: Landlock unavailable on this kernel");
        return;
    }
    let Some(tmpdir) = std::env::var_os("TMPDIR") else {
        eprintln!("SKIP: TMPDIR unset; cannot build an outside-root probe path");
        return;
    };

    let ws = tempfile::tempdir().expect("tempdir workspace");
    let m = PtyManager::new(ws.path());
    let probe = std::path::Path::new(&tmpdir)
        .join(format!(".nca-sbox-toggle-probe-{}", std::process::id()));
    // Clean any stale file from an earlier run so each phase's assertion
    // is meaningful.
    let _ = std::fs::remove_file(&probe);
    let touch = format!("touch {}", probe.display());

    // (1) Required → confined: policy is Some, outside write fails.
    m.set_sandbox_config(required_config(), &[], &[]);
    assert!(
        m.sandbox_confined(),
        "Required mode must resolve to a confined policy"
    );
    let denied = m
        .exec_streaming(&touch, 10, &progress("required-deny"))
        .await
        .expect("command should spawn");
    assert_ne!(
        denied.exit_code, 0,
        "outside-root write must fail under Required; stdout: {:?}",
        denied.stdout
    );
    assert!(
        !probe.exists(),
        "probe file must not be created while confined"
    );

    // (2) Off → unconfined: policy is None, the same write succeeds.
    m.set_sandbox_config(off_config(), &[], &[]);
    assert!(
        !m.sandbox_confined(),
        "Off mode must clear the confined policy"
    );
    let allowed = m
        .exec_streaming(&touch, 10, &progress("off-allow"))
        .await
        .expect("command should spawn");
    assert_eq!(
        allowed.exit_code, 0,
        "outside-root write must succeed when sandbox is off; stdout: {:?}",
        allowed.stdout
    );
    assert!(
        probe.exists(),
        "probe file must be created while unconfined"
    );
    let _ = std::fs::remove_file(&probe);

    // (3) Required again → confined again (round-trip closes).
    m.set_sandbox_config(required_config(), &[], &[]);
    assert!(
        m.sandbox_confined(),
        "re-applying Required must restore the confined policy"
    );
    let denied_again = m
        .exec_streaming(&touch, 10, &progress("required-deny-again"))
        .await
        .expect("command should spawn");
    assert_ne!(
        denied_again.exit_code, 0,
        "outside-root write must fail again after re-enabling; stdout: {:?}",
        denied_again.stdout
    );
    assert!(
        !probe.exists(),
        "probe file must not be created after re-enabling"
    );
}

/// Ghost-mount boundary: when the policy is `None` (sandbox off), the mount
/// list is irrelevant — passing mounts to `set_sandbox_config` must not
/// resurrect confinement. Light assertion on `sandbox_confined()` only.
#[tokio::test(flavor = "multi_thread")]
async fn sandbox_off_ignores_mount_list() {
    if !backend_supported() {
        eprintln!("SKIP: Landlock unavailable on this kernel");
        return;
    }

    let ws = tempfile::tempdir().expect("tempdir workspace");
    let m = PtyManager::new(ws.path());

    // Off with an empty mount list → unconfined.
    m.set_sandbox_config(off_config(), &[], &[]);
    assert!(
        !m.sandbox_confined(),
        "Off with no mounts must be unconfined"
    );

    // Off WITH a mount list → still unconfined (policy=None; mounts only
    // matter when a policy is built).
    let ghost_mount = std::path::PathBuf::from("/nonexistent/ghost-mount");
    m.set_sandbox_config(off_config(), std::slice::from_ref(&ghost_mount), &[]);
    assert!(
        !m.sandbox_confined(),
        "mount list must not resurrect confinement while sandbox is off"
    );
}
