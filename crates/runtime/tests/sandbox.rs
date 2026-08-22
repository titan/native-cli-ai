//! P5 sandbox tests (TDD red phase — `nca_runtime::sandbox` does not exist yet).
//!
//! These tests pin the minimal public API the implementation must provide
//! (see `docs/plans/deepseek-harness-adoption.md` §P5). No implementation is
//! written in this phase: the crate currently fails to compile (compile-red),
//! and the assertions below become the behavioral contract once the module
//! lands.
//!
//! ## Assumed API (implementation contract)
//!
//! ```ignore
//! pub mod sandbox {
//!     pub enum SandboxMode { Auto, Required, Off }
//!     // Re-exported from `nca_common::config` (kebab-case serde); common is
//!     // the single source of truth and runtime re-exports it.
//!
//!     pub struct SandboxPolicy {
//!         pub ro: Vec<PathBuf>,   // read-only roots (system dirs)
//!         pub rw: Vec<PathBuf>,   // read-write roots (workspace, tmp)
//!         pub net: bool,          // true = network allowed (unrestricted)
//!     }
//!
//!     #[derive(Debug, Clone, Copy, PartialEq, Eq)]
//!     pub enum SandboxDecision { Confined, Unconfined }
//!
//!     #[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
//!     pub enum SandboxError {
//!         #[error("sandbox unavailable: Landlock backend not supported")]
//!         SandboxUnavailable,
//!     }
//!
//!     pub struct SandboxOutput {
//!         pub combined: String,   // stdout + stderr
//!         pub exit_code: Option<i32>,
//!     }
//!
//!     /// Decide whether to run confined.
//!     /// - `probe` reports backend support (true = Landlock available).
//!     /// - `warned` is a caller-owned warn-once flag: the first auto-mode
//!     ///   degradation flips it and invokes `warn_sink` exactly once.
//!     ///   Production passes a process-global `AtomicBool`.
//!     /// - `Required` + unsupported  → `Err(SandboxUnavailable)` (fail-closed).
//!     /// - `Auto`    + unsupported  → `Ok(Unconfined)` + warn once.
//!     /// - `Off`                     → `Ok(Unconfined)`, never probes.
//!     /// - any mode + supported     → `Ok(Confined)`.
//!     pub fn resolve(
//!         mode: SandboxMode,
//!         probe: &dyn Fn() -> bool,
//!         warned: &AtomicBool,
//!         warn_sink: &dyn Fn(),
//!     ) -> Result<SandboxDecision, SandboxError>;
//!
//!     /// Real backend probe (Landlock ABI check on this kernel).
//!     pub fn backend_supported() -> bool;
//!
//!     /// Run `cmd` via `sh -c` under a Landlock ruleset built from `policy`.
//!     /// Fails with `SandboxUnavailable` when the backend cannot be established.
//!     pub fn exec_confined(cmd: &str, policy: &SandboxPolicy)
//!         -> Result<SandboxOutput, SandboxError>;
//! }
//! ```

use std::cell::Cell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};

use nca_runtime::sandbox::{
    SandboxDecision, SandboxError, SandboxMode, SandboxOutput, SandboxPolicy, backend_supported,
    exec_confined, resolve,
};

fn warn_counter() -> (Rc<Cell<usize>>, impl Fn()) {
    let warns = Rc::new(Cell::new(0usize));
    let counter = warns.clone();
    (warns, move || counter.set(counter.get() + 1))
}

#[test]
fn required_plus_unsupported_probe_fails_closed_never_unconfined() {
    let warned = AtomicBool::new(false);
    let (warns, sink) = warn_counter();

    let err = resolve(SandboxMode::Required, &|| false, &warned, &sink)
        .expect_err("required + unsupported must fail, not degrade");

    assert!(matches!(err, SandboxError::SandboxUnavailable));
    assert!(
        err.to_string().to_lowercase().contains("unavailable"),
        "error display should be actionable: {err}"
    );
    assert_eq!(warns.get(), 0, "no warn on hard failure");
    assert!(
        !warned.load(Ordering::Relaxed),
        "warn-once flag must not be set"
    );
}

#[test]
fn required_plus_supported_probe_confines() {
    let warned = AtomicBool::new(false);
    let (warns, sink) = warn_counter();

    let decision = resolve(SandboxMode::Required, &|| true, &warned, &sink)
        .expect("required + supported should confine");

    assert_eq!(decision, SandboxDecision::Confined);
    assert_eq!(warns.get(), 0);
    assert!(!warned.load(Ordering::Relaxed));
}

#[test]
fn auto_plus_unsupported_probe_degrades_with_warn_once() {
    let warned = AtomicBool::new(false);
    let (warns, sink) = warn_counter();

    // First resolution: degrade to unconfined, warn exactly once.
    let first = resolve(SandboxMode::Auto, &|| false, &warned, &sink)
        .expect("auto + unsupported degrades instead of failing");
    assert_eq!(first, SandboxDecision::Unconfined);
    assert_eq!(warns.get(), 1, "first degradation warns");
    assert!(warned.load(Ordering::Relaxed), "warn-once flag latched");

    // Second resolution: still unconfined, but warn-once means no second warn.
    let second = resolve(SandboxMode::Auto, &|| false, &warned, &sink)
        .expect("auto + unsupported degrades instead of failing");
    assert_eq!(second, SandboxDecision::Unconfined);
    assert_eq!(warns.get(), 1, "warn-once: no second warning");
}

#[test]
fn auto_plus_supported_probe_confines_without_warning() {
    let warned = AtomicBool::new(false);
    let (warns, sink) = warn_counter();

    let decision = resolve(SandboxMode::Auto, &|| true, &warned, &sink)
        .expect("auto + supported should confine");

    assert_eq!(decision, SandboxDecision::Confined);
    assert_eq!(warns.get(), 0);
    assert!(!warned.load(Ordering::Relaxed));
}

#[test]
fn off_mode_never_probes_and_stays_unconfined() {
    let warned = AtomicBool::new(false);
    let (warns, sink) = warn_counter();

    let decision = resolve(
        SandboxMode::Off,
        &|| panic!("off mode must not probe the backend"),
        &warned,
        &sink,
    )
    .expect("off mode never fails");

    assert_eq!(decision, SandboxDecision::Unconfined);
    assert_eq!(warns.get(), 0);
    assert!(!warned.load(Ordering::Relaxed));
}

#[test]
fn sandbox_policy_fields_are_public_and_constructible() {
    let policy = SandboxPolicy {
        ro: vec![PathBuf::from("/usr"), PathBuf::from("/bin")],
        rw: vec![PathBuf::from("/tmp/work")],
        net: true,
    };
    assert_eq!(
        policy.ro,
        vec![PathBuf::from("/usr"), PathBuf::from("/bin")]
    );
    assert_eq!(policy.rw, vec![PathBuf::from("/tmp/work")]);
    assert!(policy.net);
}

/// Kernel-gated real-Landlock test. `#[ignore]`d so CI never runs it; execute
/// manually on a kernel with Landlock (>= 5.13, e.g. ubuntu-latest 6.8+):
///
/// ```text
/// cargo test -p nca-runtime --test sandbox -- --ignored confined_exec
/// ```
///
/// Verifies §P5 acceptance criterion 1: writing outside the rw roots fails with
/// an EACCES/Permission-denied mention, writing inside succeeds.
#[test]
#[ignore = "kernel-gated: requires Landlock support; run manually on a Landlock-capable kernel"]
fn confined_exec_blocks_writes_outside_rw_roots() {
    if !backend_supported() {
        eprintln!("SKIP: Landlock unavailable on this kernel");
        return;
    }

    let tmp = tempfile::tempdir().expect("tempdir");
    let rw_root = tmp.path().join("rw");
    std::fs::create_dir_all(&rw_root).expect("create rw root");

    // System dirs are read-only roots so /bin/sh and /usr/bin/cat stay exec-able.
    let policy = SandboxPolicy {
        ro: vec![
            PathBuf::from("/usr"),
            PathBuf::from("/bin"),
            PathBuf::from("/lib"),
            PathBuf::from("/lib64"),
            PathBuf::from("/sbin"),
            PathBuf::from("/etc"),
        ],
        rw: vec![rw_root.clone()],
        net: true,
    };

    // Outside the rw roots (tmpdir itself is granted nothing) → must fail
    // with an EACCES / permission-denied mention in the output.
    let blocked = tmp.path().join("blocked.txt");
    let out: SandboxOutput = exec_confined(&format!("echo x > {}", blocked.display()), &policy)
        .expect("confined exec should run and report the failed command");
    assert!(
        out.exit_code != Some(0),
        "write outside rw roots must fail: {out:?}"
    );
    let lower = out.combined.to_lowercase();
    assert!(
        lower.contains("denied") || out.combined.contains("EACCES"),
        "output should mention EACCES/permission denied, got: {:?}",
        out.combined
    );
    assert!(!blocked.exists(), "file must not exist after denied write");

    // Inside the rw root → succeeds.
    let ok_file = rw_root.join("ok.txt");
    let out = exec_confined(
        &format!(
            "echo ok > {} && cat {}",
            ok_file.display(),
            ok_file.display()
        ),
        &policy,
    )
    .expect("confined exec should run");
    assert_eq!(
        out.exit_code,
        Some(0),
        "write inside rw root should succeed"
    );
    assert!(out.combined.contains("ok"), "output: {:?}", out.combined);
    assert!(ok_file.exists(), "file must exist after allowed write");
}
