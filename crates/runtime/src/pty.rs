use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use tokio::io::AsyncReadExt;
use tokio::time::{Duration, Instant};

use nca_common::config::{SandboxConfig, default_sandbox_env_allow};
use nca_common::event::AgentEvent;
use nca_core::tools::ToolProgress;

use crate::sandbox::{self, SandboxPolicy};

/// Warn-once flag for auto-mode sandbox degradation (module-level so all
/// `PtyManager` instances share it; the warn lands in `.nca/nca.log` in TUI
/// mode via the tracing file writer).
static SANDBOX_WARNED: AtomicBool = AtomicBool::new(false);

/// Strip `cmd`'s environment down to the allowlist: exact-name matches plus
/// any `LC_*` variable always pass. `PATH` is passed unconditionally (it is
/// structurally required to locate executables), so even an empty allowlist
/// leaves the child a working shell.
fn apply_env_allow(cmd: &mut std::process::Command, allow: &[String]) {
    cmd.env_clear();
    for (key, value) in std::env::vars_os() {
        let name = key.to_string_lossy();
        let allowed = allow.iter().any(|a| *a == name) || name.starts_with("LC_");
        if allowed || name == "PATH" {
            cmd.env(key, value);
        }
    }
}

/// Runs shell commands in their own process group: streams stdout, and on
/// completion or timeout kills the entire process group (clearing any
/// backgrounded survivors) so a lingering child can never pin a worker or
/// starve the TUI input loop.
pub struct PtyManager {
    workspace_root: Mutex<std::path::PathBuf>,
    /// Resolved confinement policy; `None` = run unconfined (sandbox off,
    /// auto-degraded, or never configured).
    sandbox: Mutex<Option<SandboxPolicy>>,
    /// Env names allowed through to confined commands (see
    /// [`Self::set_sandbox_config`]); ignored on the unconfined branch.
    env_allow: Mutex<Vec<String>>,
}

impl PtyManager {
    pub fn new(workspace_root: impl AsRef<Path>) -> Self {
        Self {
            workspace_root: Mutex::new(workspace_root.as_ref().to_path_buf()),
            sandbox: Mutex::new(None),
            env_allow: Mutex::new(default_sandbox_env_allow()),
        }
    }

    /// Configure Landlock confinement for all subsequent `exec_streaming`
    /// calls. Resolves the mode against the backend exactly once per process
    /// (warn-once on auto degradation); `required` + unavailable is logged as
    /// an error and degrades to unconfined rather than breaking every shell
    /// tool call.
    ///
    /// `mounts` are the live `/mount` paths; they become rw roots when
    /// `cfg.inherit_mounts` (default). `skill_roots` are the skill catalog
    /// directories; they become read-only roots so skill-bundled tools stay
    /// executable under confinement. Takes `&self` so the supervisor can
    /// refresh the policy after a runtime `/mount` without rebuilding the
    /// `Arc`-shared manager — each confined child snapshots the policy at
    /// spawn time, so later mounts apply to the next command.
    pub fn set_sandbox_config(
        &self,
        cfg: SandboxConfig,
        mounts: &[PathBuf],
        skill_roots: &[PathBuf],
    ) {
        let decision = sandbox::resolve(
            cfg.mode,
            &sandbox::backend_supported,
            &SANDBOX_WARNED,
            &|| tracing::warn!("Landlock sandbox unavailable; PTY commands run unconfined"),
        );
        let policy = match decision {
            Ok(sandbox::SandboxDecision::Confined) => Some(SandboxPolicy::from_config(
                &cfg,
                &self.workspace_root(),
                mounts,
                skill_roots,
            )),
            Ok(sandbox::SandboxDecision::Unconfined) => None,
            Err(e) => {
                tracing::error!("sandbox required but unavailable: {e}; running unconfined");
                None
            }
        };
        *self.sandbox.lock().expect("sandbox lock poisoned") = policy;
        *self.env_allow.lock().expect("env_allow lock poisoned") = cfg.env_allow.clone();
    }

    pub fn workspace_root(&self) -> std::path::PathBuf {
        self.workspace_root
            .lock()
            .expect("workspace_root lock poisoned")
            .clone()
    }

    /// Update the workspace root for this PTY manager.
    /// All subsequent shell commands will use the new root as their working directory.
    pub fn set_root(&self, path: &Path) {
        *self
            .workspace_root
            .lock()
            .expect("workspace_root lock poisoned") = path.to_path_buf();
    }

    /// Spawn a command in its own process group, stream stdout via `progress`,
    /// and on completion or timeout kill the entire process group (clearing any
    /// backgrounded survivors). This replaces the old wait-then-read_to_end model
    /// that deadlocked when a command backgrounded a long-running process.
    pub async fn exec_streaming(
        &self,
        command: &str,
        timeout_secs: u64,
        progress: &ToolProgress,
    ) -> Result<PtyOutput, PtyError> {
        let root = self.workspace_root();
        let sandbox_policy = self.sandbox.lock().expect("sandbox lock poisoned").clone();
        let env_allow = self
            .env_allow
            .lock()
            .expect("env_allow lock poisoned")
            .clone();
        let mut cmd = if let Some(policy) = sandbox_policy {
            // Confined path: build the command as std::process::Command (same
            // sh -c / cwd / piped stdio / own process group), strip the
            // environment down to the configured allowlist, attach the
            // Landlock pre_exec via confine_cmd, then hand it to tokio.
            let std_cmd = {
                use std::os::unix::process::CommandExt;
                let mut c = std::process::Command::new("sh");
                c.arg("-c")
                    .arg(command)
                    .current_dir(&root)
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .process_group(0);
                apply_env_allow(&mut c, &env_allow);
                sandbox::confine_cmd(c, &policy)
            };
            tokio::process::Command::from(std_cmd)
        } else {
            let mut c = tokio::process::Command::new("sh");
            c.arg("-c")
                .arg(command)
                .current_dir(&root)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                // Make the child the leader of a NEW process group (pgid == child pid),
                // so we can kill the whole group via libc::kill(-pid, SIGKILL).
                .process_group(0);
            c
        };

        let mut child = cmd
            .spawn()
            .map_err(|e| PtyError::SpawnFailed(e.to_string()))?;
        let pid = child.id();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        // Reader task for stdout: streams chunks to `progress` AND returns the
        // full buffer (authoritative). Uses try_send so it never blocks if the
        // display channel is full/dropped.
        let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(64);
        let stdout_task = if let Some(mut out) = stdout {
            let chunk_tx = chunk_tx;
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let mut total: Vec<u8> = Vec::new();
                loop {
                    match out.read(&mut buf).await {
                        Ok(0) => break,
                        Ok(n) => {
                            let _ = chunk_tx.try_send(buf[..n].to_vec());
                            total.extend_from_slice(&buf[..n]);
                        }
                        Err(_) => break,
                    }
                }
                total
            })
        } else {
            drop(chunk_tx);
            tokio::spawn(async { Vec::new() })
        };

        // Reader task for stderr: collect only (not streamed to UI).
        let stderr_task = if let Some(mut err) = stderr {
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let mut total: Vec<u8> = Vec::new();
                loop {
                    match err.read(&mut buf).await {
                        Ok(0) => break,
                        Ok(n) => total.extend_from_slice(&buf[..n]),
                        Err(_) => break,
                    }
                }
                total
            })
        } else {
            tokio::spawn(async { Vec::new() })
        };

        // Drive: stream chunks while waiting for the child to exit OR the overall
        // timeout. We do NOT wait for pipe EOF (that would re-introduce the
        // deadlock when survivors hold the pipe).
        //
        // Once the stdout reader ends (chunk channel closed) we stop polling
        // it: a closed `mpsc::Receiver::recv()` returns `None` on every poll,
        // which would busy-spin and starve `child.wait()` / the timeout (a real
        // bug caught by the regression suite for fast-exiting foreground
        // commands whose stdout closes before the child is reaped).
        let deadline = Instant::now() + Duration::from_secs(timeout_secs);
        let mut timed_out = false;
        let mut exit_code: Option<i32> = None;
        let mut reader_done = false;
        loop {
            if reader_done {
                tokio::select! {
                    status = child.wait() => {
                        if let Ok(s) = status {
                            exit_code = s.code();
                        }
                        break;
                    }
                    _ = tokio::time::sleep_until(deadline) => {
                        timed_out = true;
                        break;
                    }
                }
            } else {
                tokio::select! {
                    chunk = chunk_rx.recv() => match chunk {
                        Some(c) => progress.emit_chunk(&String::from_utf8_lossy(&c)),
                        None => reader_done = true,
                    },
                    status = child.wait() => {
                        if let Ok(s) = status {
                            exit_code = s.code();
                        }
                        break;
                    }
                    _ = tokio::time::sleep_until(deadline) => {
                        timed_out = true;
                        break;
                    }
                }
            }
        }

        // Kill the ENTIRE process group to clear any backgrounded survivors.
        // process_group made child the group leader so pid == pgid.
        if let Some(pid) = pid {
            // SAFETY: libc::kill on a process group is a standard POSIX operation.
            // Returns ESRCH (harmless) if the group already exited.
            unsafe {
                libc::kill(-(pid as i32), libc::SIGKILL);
            }
        }
        // Reap to avoid a zombie. On timeout, reap the now-killed child.
        if timed_out {
            let _ = child.wait().await;
        }

        // Drain reader tasks (they terminate on EOF after the group kill).
        let stdout_buf = stdout_task.await.unwrap_or_default();
        let stderr_buf = stderr_task.await.unwrap_or_default();

        if timed_out {
            return Err(PtyError::Timeout(timeout_secs));
        }

        // Merge stderr into stdout, matching the original logic exactly.
        let mut stdout = String::from_utf8_lossy(&stdout_buf).into_owned();
        let stderr = String::from_utf8_lossy(&stderr_buf).into_owned();
        if !stderr.is_empty() && !stdout.is_empty() {
            stdout.push('\n');
            stdout.push_str(&stderr);
        } else if !stderr.is_empty() {
            stdout = stderr;
        }

        Ok(PtyOutput {
            stdout,
            exit_code: exit_code.unwrap_or(-1),
        })
    }

    /// Run a command without streaming its output (backward-compatible wrapper
    /// around [`Self::exec_streaming`]).
    pub async fn exec(&self, command: &str, timeout_secs: u64) -> Result<PtyOutput, PtyError> {
        let (tx, _rx) = tokio::sync::mpsc::channel::<AgentEvent>(8);
        let progress = ToolProgress::new(String::new(), tx);
        self.exec_streaming(command, timeout_secs, &progress).await
    }
}

#[derive(Debug)]
pub struct PtyOutput {
    pub stdout: String,
    pub exit_code: i32,
}

#[derive(Debug, thiserror::Error)]
pub enum PtyError {
    #[error("Command timed out after {0}s")]
    Timeout(u64),
    #[error("Spawn failed: {0}")]
    SpawnFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::EnvGuard;
    use std::time::Duration;

    fn env_names(cmd: &std::process::Command) -> Vec<String> {
        cmd.get_envs()
            .map(|(k, _)| k.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn apply_env_allow_strips_secrets_keeps_allowlist() {
        let _guard = EnvGuard::set(&[
            ("NCA_TEST_ALLOW_ME", Some("visible")),
            ("NCA_TEST_SECRET", Some("s3cr3t")),
            ("LC_NCA_TEST_LOCALE", Some("xx_YY")),
        ]);

        let mut cmd = std::process::Command::new("sh");
        apply_env_allow(&mut cmd, &["NCA_TEST_ALLOW_ME".to_string()]);
        let names = env_names(&cmd);

        assert!(
            names.iter().any(|n| n == "PATH"),
            "PATH passes unconditionally: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "NCA_TEST_ALLOW_ME"),
            "allowlisted name passes: {names:?}"
        );
        assert!(
            !names.iter().any(|n| n == "NCA_TEST_SECRET"),
            "non-allowlisted name must be stripped: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "LC_NCA_TEST_LOCALE"),
            "LC_* prefix always passes: {names:?}"
        );
    }

    #[test]
    fn apply_env_allow_default_list_keeps_toolchain_vars() {
        let _guard = EnvGuard::set(&[
            ("HOME", Some("/home/test")),
            ("CARGO_HOME", Some("/home/test/.cargo")),
            ("RUSTUP_HOME", Some("/home/test/.rustup")),
            ("NCA_TEST_SECRET", Some("s3cr3t")),
        ]);

        let mut cmd = std::process::Command::new("sh");
        apply_env_allow(&mut cmd, &default_sandbox_env_allow());
        let names = env_names(&cmd);

        for name in ["PATH", "HOME", "CARGO_HOME", "RUSTUP_HOME"] {
            assert!(
                names.iter().any(|n| n == name),
                "default allowlist must pass {name}: {names:?}"
            );
        }
        assert!(
            !names.iter().any(|n| n == "NCA_TEST_SECRET"),
            "default allowlist must strip non-allowlisted vars: {names:?}"
        );
    }

    #[test]
    fn apply_env_allow_empty_list_passes_path_only() {
        let _guard = EnvGuard::set(&[
            ("HOME", Some("/home/test")),
            ("LC_NCA_TEST_LOCALE", Some("xx_YY")),
        ]);

        let mut cmd = std::process::Command::new("sh");
        apply_env_allow(&mut cmd, &[]);
        let names = env_names(&cmd);

        assert!(
            names.iter().any(|n| n == "PATH"),
            "PATH survives even an empty allowlist: {names:?}"
        );
        assert!(
            !names.iter().any(|n| n == "HOME"),
            "empty allowlist strips HOME: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "LC_NCA_TEST_LOCALE"),
            "LC_* passes even with an empty allowlist: {names:?}"
        );
    }

    fn mgr() -> PtyManager {
        PtyManager::new(".")
    }

    fn progress(name: &str) -> ToolProgress {
        let (tx, _rx) = tokio::sync::mpsc::channel::<AgentEvent>(8);
        ToolProgress::new(name, tx)
    }

    /// Regression suite for the shell execution backend. Covers: no deadlock
    /// on backgrounded processes (the original input-lag/worker-block bug),
    /// process-group cleanup on return (policy: 结束即清整组), exit-code
    /// reporting for foreground commands, and concurrent calls on one runtime
    /// (the production tool-pipeline shape).
    #[tokio::test(flavor = "multi_thread")]
    async fn exec_streaming_regression_suite() {
        let m = mgr();

        // (1) A backgrounded long-running process must NOT deadlock. The old
        //     wait-then-read_to_end model blocked forever here because the
        //     backgrounded `sleep` kept the stdout pipe open.
        let res = tokio::time::timeout(
            Duration::from_secs(10),
            m.exec_streaming("sleep 30 & echo started", 60, &progress("no-deadlock")),
        )
        .await;
        assert!(res.is_ok(), "exec_streaming hung (deadlock not fixed)");
        let out = res.unwrap().expect("command should succeed");
        assert!(out.stdout.contains("started"), "stdout: {:?}", out.stdout);
        assert_eq!(out.exit_code, 0);

        // (2) The backgrounded survivor must be killed with its process group
        //     on return. Returning at all requires its inherited pipe
        //     write-end to close, i.e. it was reaped; verify with `kill -0`.
        let out = m
            .exec_streaming("sleep 30 & echo $!", 60, &progress("kill-group"))
            .await
            .expect("command should succeed");
        let pid: i32 = out
            .stdout
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("expected a pid, got: {:?}", out.stdout));
        let probe = std::process::Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .output()
            .expect("probe `kill -0` should run");
        assert!(
            !probe.status.success(),
            "backgrounded pid={pid} still alive; process group was not killed"
        );

        // (3) Foreground commands report their exit code. This also guards
        //     against the closed-channel busy-spin regression: a fast-exiting
        //     command whose stdout closes before the child is reaped must not
        //     wedge the drive loop (once `chunk_rx` returns `None` forever we
        //     stop polling it).
        let ok = m
            .exec_streaming("printf hello", 30, &progress("fg-ok"))
            .await
            .expect("command should succeed");
        assert_eq!(ok.stdout, "hello");
        assert_eq!(ok.exit_code, 0);

        let fail = m
            .exec_streaming("exit 7", 30, &progress("fg-fail"))
            .await
            .expect("spawn should succeed");
        assert_eq!(fail.exit_code, 7);

        // (4) Concurrent tool calls on a single runtime (the production shape:
        //     the tool pipeline runs tools concurrently via join_all). Each
        //     call gets its own process group; neither deadlocks.
        let prog_a = progress("conc-a");
        let prog_b = progress("conc-b");
        let (a, b) = tokio::join!(
            tokio::time::timeout(
                Duration::from_secs(10),
                m.exec_streaming("sleep 30 & echo a", 60, &prog_a),
            ),
            tokio::time::timeout(
                Duration::from_secs(10),
                m.exec_streaming("sleep 30 & echo b", 60, &prog_b),
            ),
        );
        let a = a.expect("concurrent A hung").expect("A should succeed");
        let b = b.expect("concurrent B hung").expect("B should succeed");
        assert!(a.stdout.contains('a'), "A stdout: {:?}", a.stdout);
        assert!(b.stdout.contains('b'), "B stdout: {:?}", b.stdout);
    }
}
