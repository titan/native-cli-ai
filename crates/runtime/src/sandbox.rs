//! Kernel-level command sandboxing via Landlock (P5).
//!
//! The backend confines a child process to a set of read-only roots (system
//! dirs) and read-write roots (workspace, temp) before `exec`. Restrictions
//! apply only to the confined child; the parent (and the agent runtime) stays
//! unconfined.
//!
//! Mode semantics ([`SandboxMode`]):
//! - `Off`: never probe, never confine.
//! - `Auto` (default): confine when supported; otherwise degrade to unconfined
//!   with a warn-once notice.
//! - `Required`: fail closed with [`SandboxError::SandboxUnavailable`].
//!
//! [`exec_confined`] is a synchronous function (spawn + wait). Future async
//! callers should wrap it in `tokio::task::spawn_blocking`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

pub use nca_common::config::{SandboxConfig, SandboxMode};

/// Filesystem confinement policy for one confined execution.
#[derive(Debug, Clone, Default)]
pub struct SandboxPolicy {
    /// Read-only roots (system dirs: `/usr`, `/bin`, `/lib`, ...).
    pub ro: Vec<PathBuf>,
    /// Read-write roots (workspace, temp dirs).
    pub rw: Vec<PathBuf>,
    /// `true` = network unrestricted (denying network is a separate future
    /// feature; Landlock's net scope only covers abstract UNIX sockets).
    pub net: bool,
}

/// Essential character-device nodes granted read-write access in every
/// policy. Ordinary shell commands and build tooling routinely redirect
/// through `/dev/null` (`2>/dev/null`, `</dev/zero`, ...), and many tools
/// draw entropy from `/dev/urandom`. `/dev` is not covered by any built-in
/// root, so without these nodes common redirections fail with
/// `/dev/null: Permission denied` (Landlock EACCES) into tool output.
const ESSENTIAL_DEVICES: [&str; 6] = [
    "/dev/null",
    "/dev/zero",
    "/dev/full",
    "/dev/random",
    "/dev/urandom",
    "/dev/tty",
];

impl SandboxPolicy {
    /// Build a policy from config: built-in read-only system roots plus
    /// `config.ro_paths`, workspace + temp plus `config.rw_paths`, and the
    /// essential device nodes ([`ESSENTIAL_DEVICES`]).
    pub fn from_config(config: &SandboxConfig, workspace_root: &std::path::Path) -> Self {
        let mut ro: Vec<PathBuf> = [
            "/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/nix", "/opt",
        ]
        .iter()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .collect();
        ro.extend(config.ro_paths.iter().cloned());

        // Git global config chain (read-only): without these, every `git`
        // invocation inside the sandbox fails with rc=128 because git cannot
        // read its global/system config. Follows the cargo/cache derivation
        // pattern: env var → HOME fallback → existence filter.
        // NOTE: `~/.ssh` and the whole `$HOME` are deliberately NOT defaults
        // (they contain secrets and far more than tooling needs) — opt in
        // via `ro_paths` in config.
        let git_xdg_config = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
            .map(|d| d.join("git").join("config"));
        let git_global = std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".gitconfig"));
        ro.extend(
            [git_global, git_xdg_config]
                .into_iter()
                .flatten()
                .filter(|p| p.exists()),
        );

        let mut rw = vec![workspace_root.to_path_buf(), std::env::temp_dir()];
        // Plan §P5 default rw roots: also cargo home and XDG cache so that
        // default-Auto confinement does not break `cargo build`.
        let cargo_home = std::env::var_os("CARGO_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cargo")));
        let cache_home = std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")));
        rw.extend(
            [cargo_home, cache_home]
                .into_iter()
                .flatten()
                .filter(|p| p.exists()),
        );
        rw.extend(config.rw_paths.iter().cloned());
        // POSIX shell substrate: must come last so config `rw_paths` cannot
        // accidentally shadow them, and filtered by existence so non-Linux
        // targets (and stripped containers) skip silently.
        rw.extend(
            ESSENTIAL_DEVICES
                .iter()
                .map(PathBuf::from)
                .filter(|p| p.exists()),
        );

        Self {
            ro,
            rw,
            net: config.net,
        }
    }
}

/// Whether a command will run confined after [`resolve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxDecision {
    /// Run under the Landlock ruleset.
    Confined,
    /// Run without confinement (off mode, or auto-degraded backend).
    Unconfined,
}

/// Errors from sandbox resolution and confined execution.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum SandboxError {
    /// The Landlock backend is unavailable and mode is `required` (fail-closed).
    #[error("sandbox unavailable: Landlock backend not supported")]
    SandboxUnavailable,
    /// Spawning or waiting on the confined process failed.
    #[error("sandbox exec failed: {0}")]
    Exec(String),
}

/// Result of one confined execution.
#[derive(Debug, Clone)]
pub struct SandboxOutput {
    /// Merged stdout + stderr of the confined command.
    pub combined: String,
    /// Exit status code, `None` when the process was killed by a signal.
    pub exit_code: Option<i32>,
}

/// Decide whether to run confined.
///
/// - `probe` reports backend support (true = Landlock available).
/// - `warned` is a caller-owned warn-once flag: the first auto-mode
///   degradation flips it and invokes `warn_sink` exactly once. Production
///   callers pass a process-global `AtomicBool`.
/// - `Required` + unsupported → `Err(SandboxUnavailable)` (fail-closed).
/// - `Auto` + unsupported → `Ok(Unconfined)` + warn once.
/// - `Off` → `Ok(Unconfined)`, never probes.
/// - Any mode + supported → `Ok(Confined)`.
pub fn resolve(
    mode: SandboxMode,
    probe: &dyn Fn() -> bool,
    warned: &AtomicBool,
    warn_sink: &dyn Fn(),
) -> Result<SandboxDecision, SandboxError> {
    match mode {
        SandboxMode::Off => Ok(SandboxDecision::Unconfined),
        SandboxMode::Required => {
            if probe() {
                Ok(SandboxDecision::Confined)
            } else {
                Err(SandboxError::SandboxUnavailable)
            }
        }
        SandboxMode::Auto => {
            if probe() {
                Ok(SandboxDecision::Confined)
            } else {
                // Warn exactly once per process: the first degradation wins
                // the swap from false to true.
                if !warned.swap(true, Ordering::Relaxed) {
                    warn_sink();
                }
                Ok(SandboxDecision::Unconfined)
            }
        }
    }
}

#[cfg(target_os = "linux")]
mod landlock_backend {
    use std::io;
    use std::os::unix::process::CommandExt;
    use std::path::PathBuf;

    use landlock::{
        ABI, Access, AccessFs, PathBeneath, PathFd, Ruleset, RulesetAttr, RulesetCreatedAttr,
    };

    use super::{SandboxError, SandboxOutput, SandboxPolicy};

    /// Landlock ABI we build rulesets against. With `CompatLevel::BestEffort`
    /// (the default), access rights beyond the running kernel's ABI are
    /// silently dropped, so this can be bumped without breaking older kernels.
    const TARGET_ABI: ABI = ABI::V5;

    /// Build and apply the Landlock ruleset to the calling thread.
    ///
    /// Runs inside the child's `pre_exec` closure (after fork, before exec):
    /// the restriction persists across `execve` and confines only the child.
    fn apply_ruleset(ro: Vec<PathBuf>, rw: Vec<PathBuf>, net: bool) -> io::Result<()> {
        fn landlock_err(e: impl std::fmt::Display) -> io::Error {
            io::Error::other(e.to_string())
        }

        let access_all = AccessFs::from_all(TARGET_ABI);
        let access_read = AccessFs::from_read(TARGET_ABI);

        let mut ruleset = Ruleset::default()
            .handle_access(access_all)
            .map_err(landlock_err)?;
        // net = false additionally denies abstract UNIX socket connections
        // (Landlock's only network scope at this ABI); BestEffort compatibility
        // silently drops it on kernels without that scope.
        if !net {
            ruleset = ruleset
                .scope(landlock::Scope::AbstractUnixSocket)
                .map_err(landlock_err)?;
        }

        ruleset
            .create()
            .map_err(landlock_err)?
            .add_rules(ro.into_iter().filter_map(|p| {
                Some(Ok::<landlock::PathBeneath<PathFd>, landlock::RulesetError>(
                    PathBeneath::new(PathFd::new(&p).ok()?, access_read),
                ))
            }))
            .map_err(landlock_err)?
            .add_rules(rw.into_iter().filter_map(|p| {
                Some(Ok::<landlock::PathBeneath<PathFd>, landlock::RulesetError>(
                    PathBeneath::new(PathFd::new(&p).ok()?, access_all),
                ))
            }))
            .map_err(landlock_err)?
            .restrict_self()
            .map_err(landlock_err)?;
        Ok(())
    }

    /// Real backend probe: reports whether this kernel supports Landlock.
    ///
    /// Only builds (does not apply) a minimal ruleset, so probing never
    /// confines the calling process.
    pub fn backend_supported() -> bool {
        Ruleset::default()
            .handle_access(AccessFs::from_all(ABI::V1))
            .and_then(|r| r.create())
            .is_ok()
    }

    /// Attach the Landlock ruleset to a caller-built
    /// [`std::process::Command`] as a `pre_exec` hook: the caller's
    /// argv/cwd/pipes are kept, only the confinement is added. The ruleset is
    /// applied in the child between fork and exec, so the parent stays
    /// unconfined and the restriction persists across `execve`.
    ///
    /// Does NOT probe backend support — callers confine only after
    /// [`super::resolve`] returned [`super::SandboxDecision::Confined`].
    pub fn confine_cmd(
        mut cmd: std::process::Command,
        policy: &SandboxPolicy,
    ) -> std::process::Command {
        let ro = policy.ro.clone();
        let rw = policy.rw.clone();
        let net = policy.net;
        // SAFETY: pre_exec runs between fork and exec in the child; the
        // closure only issues landlock syscalls and returns an io::Error on
        // failure, which aborts the exec.
        unsafe {
            cmd.pre_exec(move || apply_ruleset(ro.clone(), rw.clone(), net));
        }
        cmd
    }

    /// Run `cmd` via `sh -c` under a Landlock ruleset built from `policy`.
    ///
    /// The ruleset is applied in the child's `pre_exec` hook, so the parent
    /// process stays unconfined. Fails with [`SandboxError::SandboxUnavailable`]
    /// when the backend cannot be established.
    pub fn exec_confined(cmd: &str, policy: &SandboxPolicy) -> Result<SandboxOutput, SandboxError> {
        if !backend_supported() {
            return Err(SandboxError::SandboxUnavailable);
        }

        let mut command = std::process::Command::new("sh");
        command.arg("-c").arg(cmd);
        let mut command = confine_cmd(command, policy);

        let output = command
            .output()
            .map_err(|e| SandboxError::Exec(e.to_string()))?;

        let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
        combined.push_str(&String::from_utf8_lossy(&output.stderr));

        Ok(SandboxOutput {
            combined,
            exit_code: output.status.code(),
        })
    }
}

#[cfg(target_os = "linux")]
pub use landlock_backend::{backend_supported, confine_cmd, exec_confined};

#[cfg(not(target_os = "linux"))]
mod fallback_backend {
    use super::{SandboxError, SandboxOutput, SandboxPolicy};

    /// Landlock is Linux-only; non-Linux targets (e.g. apple-darwin release
    /// builds) never support confinement.
    pub fn backend_supported() -> bool {
        false
    }

    /// Identity: there is no backend off Linux, so a command is returned
    /// unchanged. Callers only reach this after `resolve` said `Confined`
    /// via a lying probe (or `Required` mode on a non-Linux host).
    pub fn confine_cmd(
        cmd: std::process::Command,
        _policy: &SandboxPolicy,
    ) -> std::process::Command {
        cmd
    }

    /// Never confinable off Linux; callers hit this only after `resolve`
    /// returned `Confined` via a lying probe (or `Required` mode).
    pub fn exec_confined(
        _cmd: &str,
        _policy: &SandboxPolicy,
    ) -> Result<SandboxOutput, SandboxError> {
        Err(SandboxError::SandboxUnavailable)
    }
}

#[cfg(not(target_os = "linux"))]
pub use fallback_backend::{backend_supported, exec_confined};

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> SandboxPolicy {
        SandboxPolicy {
            ro: vec![PathBuf::from("/usr"), PathBuf::from("/bin")],
            rw: vec![PathBuf::from("/tmp")],
            net: true,
        }
    }

    #[test]
    fn from_config_includes_workspace_and_temp_rw() {
        let config = SandboxConfig {
            mode: SandboxMode::Required,
            ro_paths: vec![PathBuf::from("/opt/tools")],
            rw_paths: vec![PathBuf::from("/data")],
            net: false,
        };
        let p = SandboxPolicy::from_config(&config, std::path::Path::new("/work/root"));

        assert!(p.rw.contains(&PathBuf::from("/work/root")));
        assert!(p.rw.contains(&PathBuf::from("/data")));
        assert!(p.ro.contains(&PathBuf::from("/opt/tools")));
        assert!(!p.net);
    }

    #[test]
    fn from_config_grants_essential_device_nodes() {
        // Ordinary shell redirections (`2>/dev/null`, `</dev/zero`) and
        // entropy reads must not fail EACCES under confinement.
        let p = SandboxPolicy::from_config(&SandboxConfig::default(), std::path::Path::new("/w"));
        for dev in ESSENTIAL_DEVICES {
            if std::path::Path::new(dev).exists() {
                assert!(p.rw.contains(&PathBuf::from(dev)), "missing {dev}");
            }
        }
    }

    #[test]
    fn exec_confined_dev_null_redirect_works_on_supported_kernel() {
        if !backend_supported() {
            eprintln!("SKIP: Landlock unavailable on this kernel");
            return;
        }
        // The regression: a confined command redirecting to /dev/null must
        // not fail with "Permission denied".
        let config = SandboxConfig::default();
        let p = SandboxPolicy::from_config(&config, std::path::Path::new("."));
        let out = exec_confined("echo noisy >/dev/null 2>&1; echo pass", &p)
            .expect("confined exec with /dev/null redirect");
        assert_eq!(out.exit_code, Some(0), "{:?}", out.combined);
        assert!(out.combined.contains("pass"), "{:?}", out.combined);
        assert!(
            !out.combined.to_lowercase().contains("permission denied"),
            "{:?}",
            out.combined
        );
    }

    #[test]
    fn resolve_off_never_probes() {
        let warned = AtomicBool::new(false);
        let d = resolve(
            SandboxMode::Off,
            &|| panic!("off must not probe"),
            &warned,
            &|| panic!("off must not warn"),
        )
        .unwrap();
        assert_eq!(d, SandboxDecision::Unconfined);
    }

    #[test]
    fn resolve_required_unsupported_fails_closed() {
        let warned = AtomicBool::new(false);
        let err = resolve(SandboxMode::Required, &|| false, &warned, &|| {
            panic!("no warn on hard failure")
        })
        .unwrap_err();
        assert_eq!(err, SandboxError::SandboxUnavailable);
        assert!(!warned.load(Ordering::Relaxed));
    }

    #[test]
    fn resolve_auto_unsupported_warns_once() {
        let warned = AtomicBool::new(false);
        let count = std::cell::Cell::new(0usize);
        let sink = || count.set(count.get() + 1);

        assert_eq!(
            resolve(SandboxMode::Auto, &|| false, &warned, &sink).unwrap(),
            SandboxDecision::Unconfined
        );
        assert_eq!(count.get(), 1);
        assert_eq!(
            resolve(SandboxMode::Auto, &|| false, &warned, &sink).unwrap(),
            SandboxDecision::Unconfined
        );
        assert_eq!(count.get(), 1, "warn-once");
    }

    #[test]
    fn resolve_supported_confines() {
        let warned = AtomicBool::new(false);
        assert_eq!(
            resolve(SandboxMode::Auto, &|| true, &warned, &|| {}).unwrap(),
            SandboxDecision::Confined
        );
        assert_eq!(
            resolve(SandboxMode::Required, &|| true, &warned, &|| {}).unwrap(),
            SandboxDecision::Confined
        );
    }

    #[test]
    fn backend_probe_does_not_confine_caller() {
        // Probing must not restrict this (parent) thread: a temp write after
        // the probe still succeeds.
        let _ = backend_supported();
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join("probe-marker"), b"x").expect("write after probe");
    }

    #[test]
    fn exec_confined_echo_roundtrip_on_supported_kernel() {
        if !backend_supported() {
            eprintln!("SKIP: Landlock unavailable on this kernel");
            return;
        }
        let p = policy();
        let out = exec_confined("echo hello-sandbox", &p).expect("confined exec");
        assert_eq!(out.exit_code, Some(0));
        assert!(out.combined.contains("hello-sandbox"), "{:?}", out.combined);
    }
}
