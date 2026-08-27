//! P5 wiring red phase: sandbox → PTY + CLI contract (TDD).
//!
//! These tests pin the WIRING layer on top of the (already-landed) sandbox
//! backend: a `confine_cmd` helper that attaches a Landlock `pre_exec` to a
//! caller-built `std::process::Command`, and a `PtyManager::set_sandbox_config`
//! setter that makes `exec_streaming` confine its `sh -c` child.
//!
//! Neither API exists yet — this crate intentionally FAILS TO COMPILE
//! (compile-red) until:
//!
//! ```ignore
//! // crates/runtime/src/sandbox.rs
//! pub fn confine_cmd(cmd: std::process::Command, policy: &SandboxPolicy)
//!     -> std::process::Command;
//!
//! // crates/runtime/src/pty.rs
//! impl PtyManager {
//!     pub fn set_sandbox_config(&mut self, cfg: SandboxConfig);
//! }
//! ```
//!
//! Contract notes for the implementer:
//! - `confine_cmd` must keep the caller's argv/cwd/pipes and add only the
//!   Landlock `pre_exec` (applied in the child before exec; parent unconfined).
//! - The PTY path routes through `SandboxPolicy::from_config`, whose default
//!   rw roots are workspace + tmp + cargo home + XDG cache (plan §P5), so the
//!   outside-write assertion targets `/etc` — a path that must stay
//!   read-only under the wired policy. /tmp writes succeed by design.

use std::path::PathBuf;
use std::process::Command as StdCommand;

use nca_common::config::{SandboxConfig, SandboxMode};
use nca_common::event::AgentEvent;
use nca_core::tools::ToolProgress;
use nca_runtime::pty::PtyManager;
use nca_runtime::sandbox::{SandboxPolicy, backend_supported, confine_cmd};

fn progress(name: &str) -> ToolProgress {
    let (tx, _rx) = tokio::sync::mpsc::channel::<AgentEvent>(8);
    ToolProgress::new(name, tx)
}

/// `Required` mode, no extra roots — the policy built from this has the
/// workspace root as its only rw root (see module doc).
fn required_config() -> SandboxConfig {
    SandboxConfig {
        mode: SandboxMode::Required,
        ro_paths: Vec::new(),
        rw_paths: Vec::new(),
        net: true,
        env_allow: nca_common::config::default_sandbox_env_allow(),
        inherit_mounts: true,
    }
}

fn sys_ro_roots() -> Vec<PathBuf> {
    ["/usr", "/bin", "/lib"]
        .iter()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .collect()
}

/// §P5 acceptance 1 (backend level): `confine_cmd` restricts a caller-built
/// command. Writing inside an rw root succeeds; writing outside fails with a
/// permission-denied signal. Kernel-gated: skips when Landlock is unavailable.
#[test]
fn confine_cmd_allows_rw_root_and_denies_outside() {
    if !backend_supported() {
        eprintln!("SKIP: Landlock unavailable on this kernel");
        return;
    }

    let tmp = tempfile::tempdir().expect("tempdir");
    let marker = tmp.path().join("marker");
    let policy = SandboxPolicy {
        ro: sys_ro_roots(),
        rw: vec![tmp.path().to_path_buf()],
        net: true,
    };

    // (1) Write inside the rw root → succeeds, marker written.
    let mut ok_cmd = StdCommand::new("sh");
    ok_cmd
        .arg("-c")
        .arg(format!("echo ok > {}", marker.display()));
    let ok = confine_cmd(ok_cmd, &policy)
        .output()
        .expect("confined command should spawn");
    assert_eq!(
        ok.status.code(),
        Some(0),
        "write inside rw root failed: {:?}",
        ok
    );
    assert_eq!(
        std::fs::read_to_string(&marker).expect("marker readable"),
        "ok\n",
        "marker content"
    );

    // (2) Same command, rw root NOT covering tempdir → must fail with
    //     a permission-denied signal and no marker.
    let mut denied_cmd = StdCommand::new("sh");
    denied_cmd
        .arg("-c")
        .arg(format!("echo ok > {}", marker.display()));
    let denied = confine_cmd(
        denied_cmd,
        &SandboxPolicy {
            ro: sys_ro_roots(),
            rw: Vec::new(),
            net: true,
        },
    )
    .output()
    .expect("confined command should spawn");
    assert_ne!(
        denied.status.code(),
        Some(0),
        "write outside rw roots must fail: {:?}",
        denied
    );
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&denied.stdout),
        String::from_utf8_lossy(&denied.stderr)
    );
    assert!(
        combined.to_lowercase().contains("denied") || combined.contains("EACCES"),
        "output should mention EACCES/permission denied, got: {combined:?}"
    );
}

/// §P5 acceptance 1 (PTY level): `set_sandbox_config(Required)` confines
/// `exec_streaming` to the workspace root. Writes outside fail with a non-zero
/// exit; writes inside the workspace succeed. Kernel-gated (skips when
/// Landlock is unavailable).
#[tokio::test(flavor = "multi_thread")]
async fn pty_required_mode_confines_writes_to_workspace_root() {
    if !backend_supported() {
        eprintln!("SKIP: Landlock unavailable on this kernel");
        return;
    }

    let ws = tempfile::tempdir().expect("tempdir");
    let m = PtyManager::new(ws.path());
    m.set_sandbox_config(required_config(), &[]);

    // (1) Write outside all rw roots (/etc is ro under the wired policy) →
    //     must fail. Clean any stale file from earlier runs so the assertion
    //     below is meaningful. (/tmp is a default rw root per plan §P5, so a
    //     /tmp write would — correctly — succeed and cannot be used here.)
    let outside = "/etc/should-not-exist-p5w";
    let _ = std::fs::remove_file(outside);
    let out = m
        .exec_streaming(&format!("touch {outside}"), 10, &progress("outside-write"))
        .await
        .expect("command should spawn");
    assert_ne!(
        out.exit_code, 0,
        "write outside rw root must fail; stdout: {:?}",
        out.stdout
    );
    assert!(
        !std::path::Path::new(outside).exists(),
        "file outside rw root must not be created"
    );

    // (2) Write inside the workspace root → succeeds.
    let ok = m
        .exec_streaming(
            "touch p5w-inside-ok && test -e p5w-inside-ok",
            10,
            &progress("inside-write"),
        )
        .await
        .expect("command should spawn");
    assert_eq!(
        ok.exit_code, 0,
        "write inside rw root failed: {:?}",
        ok.stdout
    );

    // (3) /tmp is a default rw root: a write there MUST succeed under the
    //     wired policy (guards against over-confinement regressions).
    let tmp_ok = m
        .exec_streaming(
            "touch ${TMPDIR:-/tmp}/p5w-tmp-ok && test -e ${TMPDIR:-/tmp}/p5w-tmp-ok",
            10,
            &progress("tmp-write"),
        )
        .await
        .expect("command should spawn");
    assert_eq!(
        tmp_ok.exit_code, 0,
        "write inside default tmp rw root failed: {:?}",
        tmp_ok.stdout
    );
}

/// `/mount` → sandbox propagation: a path passed to `set_sandbox_config` as
/// a mount becomes rw for subsequent `exec_streaming` commands (default
/// `inherit_mounts`). Also covers the mid-session refresh seam: re-calling
/// `set_sandbox_config` with the new mount list widens the policy without
/// rebuilding the manager. Kernel-gated.
///
/// The mount fixture must live OUTSIDE the default rw roots (workspace,
/// `std::env::temp_dir()`, cargo/cache homes) — a `tempfile::tempdir()` is
/// under /tmp and would be writable regardless of the mount. Bare `$HOME`
/// is deliberately not a default root (secrets), so a scratch dir there is
/// confined until mounted.
#[tokio::test(flavor = "multi_thread")]
async fn pty_mounted_paths_become_rw_after_refresh() {
    if !backend_supported() {
        eprintln!("SKIP: Landlock unavailable on this kernel");
        return;
    }
    let Some(home) = std::env::var_os("HOME") else {
        eprintln!("SKIP: HOME unset; cannot build a mount fixture outside rw roots");
        return;
    };

    let ws = tempfile::tempdir().expect("tempdir workspace");
    let mounted = std::path::Path::new(&home).join(format!(
        ".nca-sbox-mount-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    // Hosting process itself sandboxed (e.g. nca dogfooding: bare $HOME is
    // not an rw root) — cannot build a fixture outside the default rw roots,
    // so the pre/post contrast is unobservable. Skip honestly instead of
    // failing; unconfined environments (CI) run the full assertion.
    if let Err(e) = std::fs::create_dir_all(&mounted) {
        eprintln!("SKIP: cannot create mount fixture under HOME ({e})");
        return;
    }
    let m = PtyManager::new(ws.path());

    // Before the mount: writing inside the (not yet mounted) dir fails.
    m.set_sandbox_config(required_config(), &[]);
    let denied = m
        .exec_streaming(
            &format!("touch {}/pre-mount", mounted.display()),
            10,
            &progress("pre-mount-write"),
        )
        .await
        .expect("command should spawn");
    assert_ne!(denied.exit_code, 0, "pre-mount write must be confined");

    // Simulate `/mount`: refresh the policy with the live mount list.
    m.set_sandbox_config(required_config(), std::slice::from_ref(&mounted));
    let allowed = m
        .exec_streaming(
            &format!(
                "echo ok > {}/post-mount && cat {}/post-mount",
                mounted.display(),
                mounted.display()
            ),
            10,
            &progress("post-mount-write"),
        )
        .await
        .expect("command should spawn");
    assert_eq!(
        allowed.exit_code, 0,
        "post-mount write must succeed: {:?}",
        allowed.stdout
    );

    // Confinement elsewhere is untouched: /etc is still ro.
    let still = m
        .exec_streaming(
            "touch /etc/should-not-exist-p5m",
            10,
            &progress("etc-write"),
        )
        .await
        .expect("command should spawn");
    assert_ne!(still.exit_code, 0, "/etc must stay read-only");

    let _ = std::fs::remove_dir_all(&mounted);
}
