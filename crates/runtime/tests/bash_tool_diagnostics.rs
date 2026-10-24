//! Bash-tool failure diagnostics — regression tests for the incident shape:
//!
//! ```text
//! ✗ [nca:loop-breaker] Tool `execute_bash` failed 3 times consecutively
//! Last failure detail:
//! Command exited with status -1
//! ```
//!
//! A signal-killed command (OOM-killer SIGKILL, group-cleanup race, external
//! kill) with no output used to surface only that bare `-1` line — the model
//! had no cause of death, no timeout ceiling, no hint why output was empty,
//! and could only retry verbatim until the loop breaker killed the turn.
//!
//! These tests pin the diagnostics contract through the REAL
//! `RuntimeBashTool` + PTY stack, with no sandbox config so commands run on
//! the unconfined branch (same as production when Landlock resolves off).
//!
//! - signal death: `error` carries the synthesized diagnostics (signal
//!   number + SIGKILL name, no-output note, timeout ceiling) — the field
//!   `format_tool_result` reads for failed results, i.e. what the model
//!   actually sees; `output` stays empty;
//! - non-zero exit WITH output: unchanged — raw stdout in `output`, `error`
//!   stays `None` (no diagnostics appended; the model already has what it
//!   needs to self-correct);
//! - plain success: raw stdout, `success == true`.
//!
//! Run: `cargo test -p nca-runtime --test bash_tool_diagnostics`

use std::sync::Arc;

use nca_common::tool::ToolCall;
use nca_core::tools::ToolExecutor;
use nca_runtime::bash_tool::RuntimeBashTool;
use nca_runtime::pty::PtyManager;

fn bash_call(command: &str) -> ToolCall {
    ToolCall {
        id: "call-diag".into(),
        name: "execute_bash".into(),
        input: serde_json::json!({ "command": command, "timeout_secs": 25 }),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn signal_death_output_carries_cause_timeout_and_no_output_note() {
    let ws = tempfile::tempdir().expect("tempdir workspace");
    let pty = Arc::new(PtyManager::new(ws.path()));
    let tool = RuntimeBashTool::new(Arc::clone(&pty));

    let result = tool.execute(&bash_call("kill -9 $$")).await;

    assert!(!result.success, "signal death must be a failure");
    // Diagnostics live in `error` — the field `format_tool_result` prefers
    // for failed results. `output` stays empty (no duplication).
    let err = result
        .error
        .as_deref()
        .expect("signal death must carry synthesized diagnostics in `error`");
    assert!(
        err.contains("killed by signal 9"),
        "cause of death must be named: {err}"
    );
    assert!(
        err.contains("SIGKILL"),
        "signal name must be included: {err}"
    );
    assert!(
        err.contains("no output captured"),
        "empty output must be explicit: {err}"
    );
    assert!(
        err.contains("timeout was 25s"),
        "the timeout ceiling in effect must be reported: {err}"
    );
    assert!(
        err.contains("status -1"),
        "exit-code sentinel is still reported for continuity: {err}"
    );
    assert!(
        !err.trim().eq("Command exited with status -1"),
        "the bare undiagnosed one-liner must be gone: {err}"
    );
    assert!(
        result.output.is_empty(),
        "`output` must stay empty (diagnostics not duplicated): {:?}",
        result.output
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn nonzero_exit_with_stdout_stays_raw() {
    let ws = tempfile::tempdir().expect("tempdir workspace");
    let pty = Arc::new(PtyManager::new(ws.path()));
    let tool = RuntimeBashTool::new(Arc::clone(&pty));

    let result = tool.execute(&bash_call("echo boom; exit 7")).await;

    assert!(!result.success, "exit 7 is a failure");
    assert!(
        result.output.contains("boom"),
        "real stdout is preserved: {}",
        result.output
    );
    assert!(
        result.error.is_none(),
        "output exists — no diagnostics synthesized: {:?}",
        result.error
    );
    assert!(
        !result.output.contains("timeout was"),
        "timeout note only fires on the empty-output failure path: {}",
        result.output
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn plain_success_path_unchanged() {
    let ws = tempfile::tempdir().expect("tempdir workspace");
    let pty = Arc::new(PtyManager::new(ws.path()));
    let tool = RuntimeBashTool::new(Arc::clone(&pty));

    let result = tool.execute(&bash_call("printf all-good")).await;

    assert!(result.success, "exit 0 must stay success");
    assert_eq!(result.output, "all-good");
}
