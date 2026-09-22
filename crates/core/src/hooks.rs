use nca_common::config::{HookCommand, HookConfig};
use serde_json::Value;
use tokio::io::AsyncWriteExt;

/// Environment variable carrying the JSON hook payload (same bytes as the
/// stdin delivery), so hook scripts can extract multiple fields without the
/// stdin-draining pitfall: a second `jq`/`cat` reading stdin sees EOF because
/// the first reader consumed the whole stream. `echo "$NCA_HOOK_PAYLOAD" |
/// jq …` works any number of times.
pub const HOOK_PAYLOAD_ENV: &str = "NCA_HOOK_PAYLOAD";

/// Serialized payloads larger than this skip env injection (execve rejects
/// single env strings near 128 KiB with `E2BIG`, which would kill the whole
/// hook — stdin delivery is unaffected). Real payloads stay well below this
/// (`truncate_tool_output` caps tool output at 32 KiB and session snapshots
/// carry metadata only), so the cap is a safety valve, not a tuner.
const HOOK_PAYLOAD_ENV_MAX: usize = 120_000;

#[derive(Debug, Clone)]
pub struct HookRunner {
    config: HookConfig,
}

#[derive(Debug, Clone, Copy)]
pub enum HookEventKind {
    SessionStart,
    SessionEnd,
    PreToolUse,
    PostToolUse,
    PostToolFailure,
    ApprovalRequested,
    SubagentStart,
    SubagentStop,
    TurnComplete,
}

impl HookRunner {
    pub fn new(config: HookConfig) -> Self {
        Self { config }
    }

    pub fn has_any(&self) -> bool {
        !self.config.session_start.is_empty()
            || !self.config.session_end.is_empty()
            || !self.config.pre_tool_use.is_empty()
            || !self.config.post_tool_use.is_empty()
            || !self.config.post_tool_failure.is_empty()
            || !self.config.approval_requested.is_empty()
            || !self.config.subagent_start.is_empty()
            || !self.config.subagent_stop.is_empty()
            || !self.config.turn_complete.is_empty()
    }

    pub async fn run(
        &self,
        event: HookEventKind,
        matcher_value: Option<&str>,
        payload: &Value,
    ) -> Result<(), String> {
        for hook in self.matching_hooks(event, matcher_value) {
            run_hook_command(hook, payload).await?;
        }
        Ok(())
    }

    pub async fn run_best_effort(
        &self,
        event: HookEventKind,
        matcher_value: Option<&str>,
        payload: &Value,
    ) {
        if let Err(error) = self.run(event, matcher_value, payload).await {
            tracing::warn!("hook {:?} failed: {}", event, error);
        }
    }

    fn matching_hooks(
        &self,
        event: HookEventKind,
        matcher_value: Option<&str>,
    ) -> Vec<&HookCommand> {
        hooks_for_event(&self.config, event)
            .iter()
            .filter(|hook| hook_matches(hook.matcher.as_deref(), matcher_value))
            .collect()
    }
}

fn hooks_for_event(config: &HookConfig, event: HookEventKind) -> &[HookCommand] {
    match event {
        HookEventKind::SessionStart => &config.session_start,
        HookEventKind::SessionEnd => &config.session_end,
        HookEventKind::PreToolUse => &config.pre_tool_use,
        HookEventKind::PostToolUse => &config.post_tool_use,
        HookEventKind::PostToolFailure => &config.post_tool_failure,
        HookEventKind::ApprovalRequested => &config.approval_requested,
        HookEventKind::SubagentStart => &config.subagent_start,
        HookEventKind::SubagentStop => &config.subagent_stop,
        HookEventKind::TurnComplete => &config.turn_complete,
    }
}

fn hook_matches(matcher: Option<&str>, value: Option<&str>) -> bool {
    match matcher.map(str::trim) {
        None | Some("") | Some("*") => true,
        Some(matcher) => value
            .map(|value| value == matcher || value.contains(matcher))
            .unwrap_or(false),
    }
}

async fn run_hook_command(hook: &HookCommand, payload: &Value) -> Result<(), String> {
    let payload_json = serde_json::to_string(payload).map_err(|err| err.to_string())?;
    let mut command = tokio::process::Command::new("sh");
    command
        .arg("-c")
        .arg(&hook.command)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if payload_json.len() <= HOOK_PAYLOAD_ENV_MAX {
        command.env(HOOK_PAYLOAD_ENV, &payload_json);
    } else {
        tracing::warn!(
            size = payload_json.len(),
            "hook payload too large for {} injection; delivering via stdin only",
            HOOK_PAYLOAD_ENV
        );
    }
    let mut child = command.spawn().map_err(|err| err.to_string())?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(payload_json.as_bytes())
            .await
            .map_err(|err| err.to_string())?;
    }

    let output = child
        .wait_with_output()
        .await
        .map_err(|err| err.to_string())?;

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();

    if output.status.success() {
        return Ok(());
    }

    let reason = if !stderr.is_empty() {
        stderr.clone()
    } else if !stdout.is_empty() {
        stdout.clone()
    } else {
        format!("hook command exited with status {}", output.status)
    };

    if hook.blocking {
        Err(reason)
    } else {
        // Non-blocking hooks still log failures so users can diagnose why
        // their hooks appear not to fire (e.g. missing jq, mako, etc.).
        tracing::warn!(
            "hook (non-blocking) failed: {}{}",
            reason,
            if stderr.is_empty() && stdout.is_empty() {
                String::new()
            } else {
                format!(" | stderr: {} | stdout: {}", stderr, stdout)
            }
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nca_common::config::HookConfig;
    use serde_json::json;

    /// Run one blocking `turn_complete` hook and return its result.
    async fn run_one(command: &str, payload: &Value) -> Result<(), String> {
        let runner = HookRunner::new(HookConfig {
            turn_complete: vec![HookCommand {
                command: command.to_string(),
                matcher: None,
                blocking: true,
            }],
            ..Default::default()
        });
        runner.run(HookEventKind::TurnComplete, None, payload).await
    }

    /// Env injection delivers the exact same bytes as stdin.
    #[tokio::test]
    async fn env_payload_matches_stdin_payload() {
        let dir = tempfile::tempdir().expect("tempdir");
        let env_out = dir.path().join("env.out");
        let stdin_out = dir.path().join("stdin.out");
        let payload = json!({"tool": "write_file", "workspace": "/ws/proj"});

        let cmd = format!(
            "cat > {}; printf '%s' \"$NCA_HOOK_PAYLOAD\" > {}",
            stdin_out.display(),
            env_out.display()
        );
        run_one(&cmd, &payload).await.expect("hook must succeed");

        let expected = serde_json::to_string(&payload).unwrap();
        assert_eq!(std::fs::read_to_string(&stdin_out).unwrap(), expected);
        assert_eq!(std::fs::read_to_string(&env_out).unwrap(), expected);
    }

    /// The motivating pitfall: stdin only survives ONE reader, while the env
    /// var can be read any number of times. Two successive `wc -c` reads of
    /// `$NCA_HOOK_PAYLOAD` must both see the full payload (with stdin, the
    /// second read would see 0 bytes).
    #[tokio::test]
    async fn env_payload_supports_multiple_readers() {
        let dir = tempfile::tempdir().expect("tempdir");
        let out = dir.path().join("out");
        let payload = json!({"tool": "write_file", "workspace": "/ws/proj"});

        let cmd = format!(
            "a=$(printf '%s' \"$NCA_HOOK_PAYLOAD\" | wc -c); \
             b=$(printf '%s' \"$NCA_HOOK_PAYLOAD\" | wc -c); \
             printf '%s/%s' \"$a\" \"$b\" > {}",
            out.display()
        );
        run_one(&cmd, &payload).await.expect("hook must succeed");

        let expected = serde_json::to_string(&payload).unwrap().len().to_string();
        let got = std::fs::read_to_string(&out).unwrap();
        assert_eq!(got, format!("{expected}/{expected}"));
    }

    /// Oversized payloads skip env injection (execve `E2BIG` guard) but
    /// still arrive on stdin — the hook itself must not fail.
    #[tokio::test]
    async fn oversized_payload_skips_env_but_keeps_stdin() {
        let dir = tempfile::tempdir().expect("tempdir");
        let flag = dir.path().join("flag");
        let stdin_out = dir.path().join("stdin.out");
        let payload = json!({"blob": "x".repeat(HOOK_PAYLOAD_ENV_MAX + 1)});

        let cmd = format!(
            "if [ -n \"$NCA_HOOK_PAYLOAD\" ]; then echo set > {}; else echo unset > {}; fi; \
             cat > {}",
            flag.display(),
            flag.display(),
            stdin_out.display()
        );
        run_one(&cmd, &payload).await.expect("hook must succeed");

        assert_eq!(std::fs::read_to_string(&flag).unwrap(), "unset\n");
        let stdin_bytes = std::fs::read(&stdin_out).unwrap();
        assert_eq!(
            stdin_bytes.len(),
            serde_json::to_string(&payload).unwrap().len(),
            "stdin must still carry the full oversized payload"
        );
    }
}
