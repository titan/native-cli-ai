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

/// Resolve the host's `$XDG_RUNTIME_DIR` (typically `/run/user/<uid>`).
///
/// Returns `None` when the variable is unset or the path does not exist,
/// which makes every host-session tier a graceful no-op on headless hosts,
/// containers without a runtime dir, or CI. Deliberately no
/// `/run/user/<uid>` synthesis: the env var is the contract, and guessing
/// paths the session did not advertise adds nothing (YAGNI).
pub(crate) fn resolve_xdg_runtime_dir() -> Option<PathBuf> {
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

/// Landlock rw roots backing the host-session access tiers.
///
/// - `host_audio` → `$XDG_RUNTIME_DIR/{pipewire-0, pipewire-0.lock, pulse}`
///   (the PipeWire native socket, its lock, and the PulseAudio compat dir
///   with its `native` socket and `cookie`).
/// - `host_dbus_session` → `$XDG_RUNTIME_DIR/bus` (D-Bus session socket).
/// - `host_xdg_runtime` → the whole `$XDG_RUNTIME_DIR` (subsumes both).
///
/// All entries are existence-filtered, so tiers degrade to nothing on hosts
/// without the corresponding sockets. Note these grants exist for env- and
/// file-based discovery (socket paths, the pulse cookie, directory
/// listings): connecting to an AF_UNIX filesystem socket is not itself
/// mediated by Landlock at the ABI we target — the flags gate what the
/// confined process can *discover and read*, and honestly document that.
fn host_session_rw_grants(cfg: &SandboxConfig) -> Vec<PathBuf> {
    let Some(rt) = resolve_xdg_runtime_dir() else {
        return Vec::new();
    };
    if cfg.host_xdg_runtime {
        // The coarse tier covers the finer ones; granting both is redundant.
        return vec![rt];
    }
    let mut grants = Vec::new();
    if cfg.host_audio {
        grants.extend(["pipewire-0", "pipewire-0.lock"].map(|n| rt.join(n)));
        grants.push(rt.join("pulse"));
    }
    if cfg.host_dbus_session {
        grants.push(rt.join("bus"));
    }
    grants.retain(|p| p.exists());
    grants
}

/// Environment variables the host-session tiers pass through the PTY env
/// allowlist (which otherwise strips everything but the curated list).
///
/// - `host_audio` or `host_xdg_runtime` → `XDG_RUNTIME_DIR=<rt>` so
///   PipeWire/Pulse clients locate their sockets by the standard path
///   instead of falling back to `~/.config/pulse` (which is not writable
///   under confinement and produces the misleading
///   "Failed to create secure directory" error).
/// - `host_dbus_session` → `DBUS_SESSION_BUS_ADDRESS`: the host value when
///   exported, else `unix:path=$XDG_RUNTIME_DIR/bus` — but only when that
///   socket actually exists, so headless hosts never get a dangling address.
///
/// Empty when no tier is enabled or `$XDG_RUNTIME_DIR` cannot be resolved.
/// Unconfined children inherit the full parent environment and need nothing
/// here.
pub(crate) fn host_session_env(cfg: &SandboxConfig) -> Vec<(String, String)> {
    let mut env = Vec::new();
    let Some(rt) = resolve_xdg_runtime_dir() else {
        return env;
    };
    if cfg.host_audio || cfg.host_xdg_runtime {
        env.push(("XDG_RUNTIME_DIR".to_string(), rt.display().to_string()));
    }
    if cfg.host_dbus_session {
        let bus = rt.join("bus");
        if let Some(addr) = std::env::var_os("DBUS_SESSION_BUS_ADDRESS") {
            env.push((
                "DBUS_SESSION_BUS_ADDRESS".to_string(),
                addr.to_string_lossy().into_owned(),
            ));
        } else if bus.exists() {
            env.push((
                "DBUS_SESSION_BUS_ADDRESS".to_string(),
                format!("unix:path={}", bus.display()),
            ));
        }
    }
    env
}

impl SandboxPolicy {
    /// Build a policy from config: built-in read-only system roots plus
    /// `config.ro_paths`, workspace + temp plus `config.rw_paths`, and the
    /// essential device nodes ([`ESSENTIAL_DEVICES`]).
    ///
    /// `mounts` are the live `/mount` paths (session `extra_paths`). When
    /// `config.inherit_mounts` (default) they are granted read-write so
    /// sandboxed shell commands can reach what file tools already can —
    /// `/mount` is an explicit user authorization, and file tools get rw on
    /// mounts, so propagating rw adds no authority the user has not granted.
    /// Non-existent paths are dropped (same existence filter as `ro_paths`).
    ///
    /// `skill_roots` are the skill catalog directories
    /// ([`nca_core::skills::SkillCatalog::discovery_roots`]). They become
    /// read-only roots: skills may ship bundled tools that agent-driven
    /// shell commands execute, and Landlock's read access set includes
    /// execute, so ro keeps them runnable while still denying writes — a
    /// confined command must not be able to mutate the skill definitions
    /// the agent itself follows (file-tool write access is denied there
    /// too, so this matches file-tool visibility exactly). Roots already
    /// under the workspace are redundant with its rw root but harmless.
    ///
    /// Host-session tiers (`config.host_audio`, `host_dbus_session`,
    /// `host_xdg_runtime`) additionally grant rw on the matching
    /// `$XDG_RUNTIME_DIR` entries (see [`host_session_rw_grants`]); they
    /// are opt-in because they expose the host user session (microphone
    /// capture, D-Bus services, Wayland). Caveat for honesty: connecting to
    /// an AF_UNIX filesystem socket is not mediated by Landlock at this
    /// ABI — the tiers make the sockets *reachable by the standard
    /// discovery path* (env var + readable socket/cookie files); they do
    /// not, and cannot, gate the connect() itself.
    pub fn from_config(
        config: &SandboxConfig,
        workspace_root: &std::path::Path,
        mounts: &[PathBuf],
        skill_roots: &[PathBuf],
    ) -> Self {
        let mut ro: Vec<PathBuf> = [
            "/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/nix", "/opt",
        ]
        .iter()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .collect();
        ro.extend(config.ro_paths.iter().cloned());
        // Skill catalog roots (read-only) — see the method doc for the
        // authorization argument. Existence-filtered like every other
        // derived root.
        ro.extend(skill_roots.iter().filter(|p| p.exists()).cloned());

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
        // Mounted paths (rw) — see the method doc for the authorization
        // argument. Skipped entirely when `inherit_mounts = false`.
        if config.inherit_mounts {
            rw.extend(mounts.iter().filter(|p| p.exists()).cloned());
        }
        // Host-session tiers (rw) — audio / D-Bus / full runtime dir. Placed
        // before the device nodes so those keep their must-be-last property;
        // deduped against rw already collected (e.g. host_xdg_runtime when a
        // mount already granted the runtime dir).
        for grant in host_session_rw_grants(config) {
            if !rw.contains(&grant) {
                rw.push(grant);
            }
        }
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
pub use fallback_backend::{backend_supported, confine_cmd, exec_confined};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{EnvGuard, env_read_lock};

    /// Fixture `$XDG_RUNTIME_DIR` populated with every entry the host-session
    /// tiers can grant: the PipeWire socket and lock, the PulseAudio compat
    /// dir, and the D-Bus session socket. Returns the tempdir (keeps the
    /// fixture alive) and its path.
    fn host_session_runtime_dir() -> (tempfile::TempDir, PathBuf) {
        let tmp = tempfile::tempdir().expect("runtime-dir fixture");
        let rt = tmp.path().to_path_buf();
        std::fs::write(rt.join("pipewire-0"), b"").expect("pipewire-0 fixture");
        std::fs::write(rt.join("pipewire-0.lock"), b"").expect("pipewire-0.lock fixture");
        std::fs::create_dir(rt.join("pulse")).expect("pulse fixture");
        std::fs::write(rt.join("bus"), b"").expect("bus fixture");
        (tmp, rt)
    }

    /// rw entries derived from `rt` that live INSIDE it (device nodes like
    /// `/dev/null` are rw too but are unrelated to the runtime dir).
    fn rw_inside(policy: &SandboxPolicy, rt: &std::path::Path) -> Vec<PathBuf> {
        policy
            .rw
            .iter()
            .filter(|p| p.starts_with(rt))
            .cloned()
            .collect()
    }

    fn policy() -> SandboxPolicy {
        SandboxPolicy {
            ro: vec![PathBuf::from("/usr"), PathBuf::from("/bin")],
            rw: vec![PathBuf::from("/tmp")],
            net: true,
        }
    }

    #[test]
    fn from_config_includes_workspace_and_temp_rw() {
        // from_config derives git/cargo/cache paths from HOME/XDG_*: serialize
        // against env-mutating tests (see test_util::ENV_TEST_MUTEX).
        let _env = env_read_lock();
        let config = SandboxConfig {
            mode: SandboxMode::Required,
            ro_paths: vec![PathBuf::from("/opt/tools")],
            rw_paths: vec![PathBuf::from("/data")],
            net: false,
            env_allow: nca_common::config::default_sandbox_env_allow(),
            inherit_mounts: true,
            host_audio: false,
            host_dbus_session: false,
            host_xdg_runtime: false,
        };
        let p = SandboxPolicy::from_config(&config, std::path::Path::new("/work/root"), &[], &[]);

        assert!(p.rw.contains(&PathBuf::from("/work/root")));
        assert!(p.rw.contains(&PathBuf::from("/data")));
        assert!(p.ro.contains(&PathBuf::from("/opt/tools")));
        assert!(!p.net);
    }

    #[test]
    fn from_config_grants_essential_device_nodes() {
        // Ordinary shell redirections (`2>/dev/null`, `</dev/zero`) and
        // entropy reads must not fail EACCES under confinement.
        let _env = env_read_lock();
        let p = SandboxPolicy::from_config(
            &SandboxConfig::default(),
            std::path::Path::new("/w"),
            &[],
            &[],
        );
        for dev in ESSENTIAL_DEVICES {
            if std::path::Path::new(dev).exists() {
                assert!(p.rw.contains(&PathBuf::from(dev)), "missing {dev}");
            }
        }
    }

    #[test]
    fn from_config_grants_git_global_config_paths_but_not_home_or_ssh() {
        // Regression for the sandboxed-git breakage: every `git` invocation
        // reads its global config chain at startup, so `~/.gitconfig` and the
        // XDG git config must be readable under confinement when they exist.
        //
        // The whole test (from_config AND the candidate recomputation below)
        // must see one consistent environment: hold the env lock so a
        // parallel env-mutating test (pty's EnvGuard HOME swap) cannot change
        // HOME between the two reads.
        let _env = env_read_lock();
        let p = SandboxPolicy::from_config(
            &SandboxConfig::default(),
            std::path::Path::new("/w"),
            &[],
            &[],
        );

        let xdg_config_home = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")));
        let candidates = [
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".gitconfig")),
            xdg_config_home.map(|d| d.join("git").join("config")),
        ];
        let mut asserted = 0;
        for path in candidates.into_iter().flatten() {
            if path.exists() {
                assert!(
                    p.ro.contains(&path),
                    "existing git config path must be a read-only root: {}",
                    path.display()
                );
                asserted += 1;
            }
        }
        if asserted == 0 {
            eprintln!(
                "NOTE: no git global config files exist on this machine; ro assertions skipped"
            );
        }

        // Secrets stay out: neither bare $HOME nor ~/.ssh may be granted.
        if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
            assert!(!p.ro.contains(&home), "bare $HOME must not be an ro root");
            assert!(
                !p.ro.contains(&home.join(".ssh")),
                "~/.ssh must not be an ro root"
            );
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
        let _env = env_read_lock();
        let config = SandboxConfig::default();
        let p = SandboxPolicy::from_config(&config, std::path::Path::new("."), &[], &[]);
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
    fn exec_confined_git_config_read_works_on_supported_kernel() {
        if !backend_supported() {
            eprintln!("SKIP: Landlock unavailable on this kernel");
            return;
        }
        if std::process::Command::new("git")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("SKIP: git binary not installed");
            return;
        }
        // Regression: a confined `git` must be able to read its global config
        // chain ($HOME/.gitconfig, XDG git config). Without the policy ro
        // entries, git dies at startup with rc=128 (config-read EACCES) and
        // every sandboxed shell call through git fails.
        let _env = env_read_lock();
        let config = SandboxConfig::default();
        let p = SandboxPolicy::from_config(&config, std::path::Path::new("."), &[], &[]);
        // Run from the temp dir (a rw root in every policy): `git config`
        // still walks the parent directory chain looking for a repo, and
        // outside one it needs only the global/system config chain — exactly
        // the paths this regression covers. Running from the workspace root
        // would make git try to open the workspace's own `.git`, which is
        // confined-rw only when cwd == workspace root (repo-discovery walks
        // above it and fails EACCES on the way up).
        let tmp = std::env::temp_dir();
        let out = exec_confined(
            &format!(
                "cd {} && git config user.name >/dev/null; echo rc=$?",
                tmp.display()
            ),
            &p,
        )
        .expect("confined git exec");
        // `git config user.name` exits 1 when the key is unset and 0 when set;
        // the fatal case is 128 (cannot read config). Either 0 or 1 proves the
        // global config chain was readable under confinement.
        assert!(
            out.combined.contains("rc=0") || out.combined.contains("rc=1"),
            "git must not die with a config-read error (rc=128) under confinement: {:?}",
            out.combined
        );
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

    #[test]
    fn from_config_inherits_mounts_as_rw_by_default() {
        let _env = env_read_lock();
        let mounted = tempfile::tempdir().expect("tempdir");
        let config = SandboxConfig::default();
        assert!(config.inherit_mounts, "default must propagate mounts");

        let p = SandboxPolicy::from_config(
            &config,
            std::path::Path::new("/work/root"),
            &[mounted.path().to_path_buf()],
            &[],
        );
        assert!(
            p.rw.contains(&mounted.path().to_path_buf()),
            "mounted dir must be a rw root"
        );
    }

    #[test]
    fn from_config_inherit_mounts_false_excludes_mounts() {
        let _env = env_read_lock();
        let mounted = tempfile::tempdir().expect("tempdir");
        let config = SandboxConfig {
            inherit_mounts: false,
            ..SandboxConfig::default()
        };

        let p = SandboxPolicy::from_config(
            &config,
            std::path::Path::new("/work/root"),
            &[mounted.path().to_path_buf()],
            &[],
        );
        assert!(
            !p.rw.contains(&mounted.path().to_path_buf()),
            "inherit_mounts = false must keep mounts out of the policy"
        );
    }

    #[test]
    fn from_config_drops_nonexistent_mounts() {
        let _env = env_read_lock();
        let ghost = PathBuf::from("/definitely/not/a/real/mount/point");
        let config = SandboxConfig::default();

        let p = SandboxPolicy::from_config(
            &config,
            std::path::Path::new("/work/root"),
            std::slice::from_ref(&ghost),
            &[],
        );
        assert!(!p.rw.contains(&ghost), "non-existent mount must be dropped");
    }

    #[test]
    fn from_config_grants_skill_roots_read_only_and_drops_missing() {
        let _env = env_read_lock();
        let skills = tempfile::tempdir().expect("tempdir");
        let ghost = PathBuf::from("/definitely/not/a/skill/root");
        let config = SandboxConfig::default();

        let p = SandboxPolicy::from_config(
            &config,
            std::path::Path::new("/work/root"),
            &[],
            &[skills.path().to_path_buf(), ghost.clone()],
        );
        assert!(
            p.ro.contains(&skills.path().to_path_buf()),
            "existing skill root must be a read-only root"
        );
        assert!(
            !p.rw.contains(&skills.path().to_path_buf()),
            "skill roots must NOT be read-write (confined commands must not mutate skill definitions)"
        );
        assert!(
            !p.ro.contains(&ghost),
            "non-existent skill root must be dropped"
        );
    }

    #[test]
    fn exec_confined_mounted_dir_writable_on_supported_kernel() {
        if !backend_supported() {
            eprintln!("SKIP: Landlock unavailable on this kernel");
            return;
        }
        // End-to-end regression for the mount/sandbox split: a path granted
        // via mounts must be writable by a confined shell command, matching
        // what file tools could already do through RealFs. Fixture lives
        // under bare $HOME — deliberately NOT a default rw root (unlike /tmp)
        // — so success proves the mount grant, not a default root.
        let Some(home) = std::env::var_os("HOME") else {
            eprintln!("SKIP: HOME unset");
            return;
        };
        // Skip honestly when the hosting process is itself sandboxed and
        // cannot create the fixture (nca dogfooding).
        let mounted = std::path::Path::new(&home)
            .join(format!(".nca-sbox-unit-mount-{}", std::process::id()));
        if let Err(e) = std::fs::create_dir_all(&mounted) {
            eprintln!("SKIP: cannot create mount fixture under HOME ({e})");
            return;
        }
        let config = SandboxConfig::default();
        let p = SandboxPolicy::from_config(
            &config,
            std::path::Path::new("."),
            std::slice::from_ref(&mounted),
            &[],
        );
        let marker = mounted.join("marker.txt");
        let out = exec_confined(
            &format!("echo ok > {} && cat {}", marker.display(), marker.display()),
            &p,
        )
        .expect("confined exec writing to mounted dir");
        let _ = std::fs::remove_dir_all(&mounted);
        assert_eq!(out.exit_code, Some(0), "{:?}", out.combined);
        assert!(out.combined.contains("ok"), "{:?}", out.combined);
    }

    #[test]
    fn host_audio_grants_exactly_audio_entries() {
        // host_audio must grant exactly the three audio entries — never the
        // runtime dir itself (that is the host_xdg_runtime tier) and never
        // the D-Bus session socket.
        let (_tmp, rt) = host_session_runtime_dir();
        let _guard = EnvGuard::set(&[("XDG_RUNTIME_DIR", Some(rt.to_str().unwrap()))]);
        let config = SandboxConfig {
            host_audio: true,
            ..SandboxConfig::default()
        };
        let p = SandboxPolicy::from_config(&config, std::path::Path::new("/w"), &[], &[]);

        let inside = rw_inside(&p, &rt);
        for entry in ["pipewire-0", "pipewire-0.lock", "pulse"] {
            assert!(
                inside.contains(&rt.join(entry)),
                "host_audio must grant {}: {inside:?}",
                rt.join(entry).display()
            );
        }
        assert!(
            !inside.contains(&rt),
            "host_audio must not grant the runtime dir itself"
        );
        assert!(
            !inside.contains(&rt.join("bus")),
            "host_audio must not grant the D-Bus session socket"
        );
        assert_eq!(
            inside.len(),
            3,
            "exactly the three audio entries, nothing else: {inside:?}"
        );
    }

    #[test]
    fn host_audio_grants_are_existence_filtered() {
        // A socket that does not exist on the host must not appear in the
        // policy (headless hosts, pipewire not running, ...).
        let (_tmp, rt) = host_session_runtime_dir();
        std::fs::remove_file(rt.join("pipewire-0.lock")).expect("remove lock fixture");
        let _guard = EnvGuard::set(&[("XDG_RUNTIME_DIR", Some(rt.to_str().unwrap()))]);
        let config = SandboxConfig {
            host_audio: true,
            ..SandboxConfig::default()
        };
        let p = SandboxPolicy::from_config(&config, std::path::Path::new("/w"), &[], &[]);

        let inside = rw_inside(&p, &rt);
        assert!(
            !inside.contains(&rt.join("pipewire-0.lock")),
            "missing pipewire-0.lock must be filtered out: {inside:?}"
        );
        assert!(inside.contains(&rt.join("pipewire-0")));
        assert!(inside.contains(&rt.join("pulse")));
        assert_eq!(inside.len(), 2, "{inside:?}");
    }

    #[test]
    fn host_dbus_session_grants_bus_only() {
        let (_tmp, rt) = host_session_runtime_dir();
        let _guard = EnvGuard::set(&[("XDG_RUNTIME_DIR", Some(rt.to_str().unwrap()))]);
        let config = SandboxConfig {
            host_dbus_session: true,
            ..SandboxConfig::default()
        };
        let p = SandboxPolicy::from_config(&config, std::path::Path::new("/w"), &[], &[]);

        let inside = rw_inside(&p, &rt);
        assert_eq!(
            inside,
            vec![rt.join("bus")],
            "host_dbus_session must grant the bus socket and nothing else"
        );
    }

    #[test]
    fn host_xdg_runtime_grants_whole_dir_subsuming_finer_tiers() {
        // The coarse tier covers the finer ones: with all three flags set the
        // runtime dir itself must be granted, and the subsumed per-entry
        // grants must not be duplicated alongside it.
        let (_tmp, rt) = host_session_runtime_dir();
        let _guard = EnvGuard::set(&[("XDG_RUNTIME_DIR", Some(rt.to_str().unwrap()))]);
        let config = SandboxConfig {
            host_audio: true,
            host_dbus_session: true,
            host_xdg_runtime: true,
            ..SandboxConfig::default()
        };
        let p = SandboxPolicy::from_config(&config, std::path::Path::new("/w"), &[], &[]);

        let inside = rw_inside(&p, &rt);
        assert!(
            inside.contains(&rt),
            "host_xdg_runtime must grant the whole runtime dir: {inside:?}"
        );
        for subsumed in ["pipewire-0", "pipewire-0.lock", "pulse", "bus"] {
            assert!(
                !inside.contains(&rt.join(subsumed)),
                "host_xdg_runtime subsumes the per-entry grant for {subsumed}: {inside:?}"
            );
        }
    }

    #[test]
    fn host_session_tiers_off_grant_nothing_in_runtime_dir() {
        // Default config (all tiers off): no rw entry may be inside the
        // runtime dir even when the process env advertises one.
        let (_tmp, rt) = host_session_runtime_dir();
        let _guard = EnvGuard::set(&[("XDG_RUNTIME_DIR", Some(rt.to_str().unwrap()))]);
        let p = SandboxPolicy::from_config(
            &SandboxConfig::default(),
            std::path::Path::new("/w"),
            &[],
            &[],
        );

        let inside = rw_inside(&p, &rt);
        assert!(
            inside.is_empty(),
            "no tier enabled => nothing under the runtime dir: {inside:?}"
        );
    }

    #[test]
    fn host_session_tiers_degrade_when_runtime_dir_missing() {
        // XDG_RUNTIME_DIR pointing at a nonexistent path (or unset — covered
        // by the same resolve_xdg_runtime_dir filter) must degrade to no
        // grants and no env, not a panic: graceful headless degradation.
        let ghost = PathBuf::from("/definitely/not/a/real/runtime/dir");
        let _guard = EnvGuard::set(&[("XDG_RUNTIME_DIR", Some(ghost.to_str().unwrap()))]);
        let config = SandboxConfig {
            host_audio: true,
            host_dbus_session: true,
            host_xdg_runtime: true,
            ..SandboxConfig::default()
        };

        let p = SandboxPolicy::from_config(&config, std::path::Path::new("/w"), &[], &[]);
        assert!(
            p.rw.iter().all(|entry| !entry.starts_with(&ghost)),
            "nonexistent runtime dir must yield no grants: {:?}",
            p.rw
        );
        assert!(
            host_session_env(&config).is_empty(),
            "nonexistent runtime dir must yield no env passthrough"
        );
    }

    #[test]
    fn host_session_env_audio_sets_xdg_runtime_dir() {
        // host_audio passes XDG_RUNTIME_DIR through so PipeWire/Pulse clients
        // find their sockets by the standard path.
        let (_tmp, rt) = host_session_runtime_dir();
        let _guard = EnvGuard::set(&[("XDG_RUNTIME_DIR", Some(rt.to_str().unwrap()))]);
        let config = SandboxConfig {
            host_audio: true,
            ..SandboxConfig::default()
        };
        let env = host_session_env(&config);
        assert!(
            env.contains(&("XDG_RUNTIME_DIR".to_string(), rt.display().to_string())),
            "host_audio must pass XDG_RUNTIME_DIR: {env:?}"
        );
    }

    #[test]
    fn host_session_env_dbus_only_does_not_set_xdg_runtime_dir() {
        // host_dbus_session alone must NOT leak XDG_RUNTIME_DIR: the D-Bus
        // address is passed explicitly, so the runtime dir path stays hidden.
        let (_tmp, rt) = host_session_runtime_dir();
        let _guard = EnvGuard::set(&[("XDG_RUNTIME_DIR", Some(rt.to_str().unwrap()))]);
        let config = SandboxConfig {
            host_dbus_session: true,
            ..SandboxConfig::default()
        };
        let env = host_session_env(&config);
        assert!(
            !env.iter().any(|(k, _)| k == "XDG_RUNTIME_DIR"),
            "host_dbus_session alone must not set XDG_RUNTIME_DIR: {env:?}"
        );
    }

    #[test]
    fn host_session_env_dbus_prefers_exported_address_verbatim() {
        // An exported DBUS_SESSION_BUS_ADDRESS wins over synthesis, even
        // though the synthesized unix:path=... would differ.
        let (_tmp, rt) = host_session_runtime_dir();
        let _guard = EnvGuard::set(&[
            ("XDG_RUNTIME_DIR", Some(rt.to_str().unwrap())),
            (
                "DBUS_SESSION_BUS_ADDRESS",
                Some("unix:abstract=/tmp/nca-test-bus"),
            ),
        ]);
        let config = SandboxConfig {
            host_dbus_session: true,
            ..SandboxConfig::default()
        };
        let env = host_session_env(&config);
        assert!(
            env.contains(&(
                "DBUS_SESSION_BUS_ADDRESS".to_string(),
                "unix:abstract=/tmp/nca-test-bus".to_string()
            )),
            "exported address must pass through verbatim: {env:?}"
        );
    }

    #[test]
    fn host_session_env_dbus_synthesizes_when_socket_exists() {
        // No exported address + an existing bus socket → synthesize
        // unix:path=$XDG_RUNTIME_DIR/bus.
        let (_tmp, rt) = host_session_runtime_dir();
        let _guard = EnvGuard::set(&[
            ("XDG_RUNTIME_DIR", Some(rt.to_str().unwrap())),
            ("DBUS_SESSION_BUS_ADDRESS", None),
        ]);
        let config = SandboxConfig {
            host_dbus_session: true,
            ..SandboxConfig::default()
        };
        let env = host_session_env(&config);
        assert!(
            env.contains(&(
                "DBUS_SESSION_BUS_ADDRESS".to_string(),
                format!("unix:path={}", rt.join("bus").display())
            )),
            "bus socket exists => synthesized unix:path address: {env:?}"
        );
    }

    #[test]
    fn host_session_env_dbus_omitted_when_no_socket_and_no_address() {
        // No exported address and no bus socket: emit no DBUS variable at
        // all — headless hosts must never get a dangling address.
        let (_tmp, rt) = host_session_runtime_dir();
        std::fs::remove_file(rt.join("bus")).expect("remove bus fixture");
        let _guard = EnvGuard::set(&[
            ("XDG_RUNTIME_DIR", Some(rt.to_str().unwrap())),
            ("DBUS_SESSION_BUS_ADDRESS", None),
        ]);
        let config = SandboxConfig {
            host_dbus_session: true,
            ..SandboxConfig::default()
        };
        let env = host_session_env(&config);
        assert!(
            !env.iter().any(|(k, _)| k == "DBUS_SESSION_BUS_ADDRESS"),
            "no socket and no address => no DBUS var: {env:?}"
        );
    }
}
