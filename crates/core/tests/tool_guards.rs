//! TDD red-phase tests for P6 "工具护栏" (tool guards).
//!
//! Spec: `docs/plans/deepseek-harness-adoption.md` §P6.
//!
//! These tests are intentionally **compile-red** right now: they reference
//! API that does not exist yet and that the implementer must provide:
//!
//! 1. New public module `nca_core::tool_guards` (declared in `lib.rs`):
//!    - `pub struct RepeatCallGuard` with `new()` (recent-calls map capped at
//!      32 entries, FIFO eviction) and `record(&mut self, &str, &serde_json::Value) -> RepeatAction`.
//!    - `pub enum RepeatAction { Proceed, Hint(String), StrongHint(String), Stop(String) }`
//!      deriving `Debug, Clone, PartialEq, Eq`.
//! 2. `nca_common::tool::ToolDefinition` gains `#[serde(default)] pub timeout_ms: Option<u64>`.
//! 3. `nca_common::tool::ToolResult` gains `#[serde(default)] pub timed_out: bool`.
//! 4. `nca_core::tool_pipeline::run_tool_pipeline` wraps Phase-2 execution in
//!    `tokio::time::timeout` when `timeout_ms` is set, returning a failed
//!    `ToolResult` with `timed_out = true` and an error mentioning the tool
//!    name and the numeric timeout value (in ms).
//!
//! Pipeline wiring contract for the implementer (not directly testable here
//! because it changes `run_tool_pipeline`'s signature, which is a design
//! decision): `RepeatAction::Hint/StrongHint(msg)` → append `msg` to the tool
//! result `output`; `RepeatAction::Stop(msg)` → do not execute, return a
//! failed `ToolResult` whose `error` is `msg` (strategy-change advice).
//!
//! Run with: `cargo test -p nca-core --test tool_guards`

use std::sync::atomic::AtomicBool;
use std::time::Duration;

use async_trait::async_trait;
use nca_common::config::{PermissionConfig, PermissionMode};
use nca_common::event::AgentEvent;
use nca_common::tool::{ToolCall, ToolDefinition, ToolResult};
use nca_core::approval::ApprovalPolicy;
use nca_core::hooks::HookRunner;
use nca_core::tool_guards::{RepeatAction, RepeatCallGuard};
use nca_core::tool_pipeline::run_tool_pipeline;
use nca_core::tools::{ToolExecutor, ToolRegistry};
use serde_json::json;
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------
// Stub tool + pipeline harness
// ---------------------------------------------------------------------------

/// A controllable stub tool whose definition declares a `timeout_ms` and whose
/// execution sleeps `sleep_ms` before succeeding.
struct StubTool {
    name: &'static str,
    timeout_ms: Option<u64>,
    sleep_ms: u64,
}

#[async_trait]
impl ToolExecutor for StubTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.to_string(),
            description: "stub tool for P6 guard tests".into(),
            parameters: json!({"type": "object", "properties": {}}),
            timeout_ms: self.timeout_ms,
        }
    }

    async fn execute(&self, call: &ToolCall) -> ToolResult {
        tokio::time::sleep(Duration::from_millis(self.sleep_ms)).await;
        ToolResult {
            call_id: call.id.clone(),
            success: true,
            output: "done".into(),
            error: None,
            timed_out: false,
        }
    }
}

/// Run one batch of tool calls through the real pipeline with an allow-all
/// approval policy (no approvals, no hooks, no cancellation).
async fn run_batch(tools: &ToolRegistry, calls: Vec<ToolCall>) -> Vec<ToolResult> {
    let mut approval = ApprovalPolicy::new(PermissionConfig {
        mode: PermissionMode::BypassPermissions,
        ..Default::default()
    });
    let hooks: Option<HookRunner> = None;
    let (tx, _rx) = mpsc::channel::<AgentEvent>(64);
    let cancel_flag = AtomicBool::new(false);
    let pipeline = run_tool_pipeline(&tools, &mut approval, &hooks, &tx, &cancel_flag, calls)
        .await
        .expect("pipeline must not be cancelled");
    pipeline.results
}

// ---------------------------------------------------------------------------
// RepeatCallGuard: escalation thresholds
// ---------------------------------------------------------------------------

#[test]
fn no_hint_below_third_call() {
    let mut guard = RepeatCallGuard::new();
    let input = json!({"path": "/etc/passwd"});

    assert_eq!(guard.record("read_file", &input), RepeatAction::Proceed); // 1st
    assert_eq!(guard.record("read_file", &input), RepeatAction::Proceed); // 2nd
}

#[test]
fn hint_on_third_stronger_on_fifth_hard_stop_on_eighth() {
    let mut guard = RepeatCallGuard::new();
    let input = json!({"path": "/etc/passwd"});

    assert_eq!(guard.record("read_file", &input), RepeatAction::Proceed); // 1
    assert_eq!(guard.record("read_file", &input), RepeatAction::Proceed); // 2

    let hint = match guard.record("read_file", &input) {
        // 3
        RepeatAction::Hint(msg) => msg,
        other => panic!("3rd call must return Hint, got {other:?}"),
    };
    assert!(!hint.is_empty(), "hint message must be non-empty");

    let _ = guard.record("read_file", &input); // 4 (counted, level unspecified)

    let strong = match guard.record("read_file", &input) {
        // 5
        RepeatAction::StrongHint(msg) => msg,
        other => panic!("5th call must return StrongHint, got {other:?}"),
    };
    assert!(!strong.is_empty(), "strong hint message must be non-empty");
    assert_ne!(
        strong, hint,
        "5th-call message must be stronger than the 3rd-call hint"
    );

    let _ = guard.record("read_file", &input); // 6 (counted, level unspecified)
    let _ = guard.record("read_file", &input); // 7 (counted, level unspecified)

    let stop = match guard.record("read_file", &input) {
        // 8
        RepeatAction::Stop(msg) => msg,
        other => panic!("8th call must hard-stop, got {other:?}"),
    };
    assert!(
        !stop.is_empty(),
        "stop message must carry strategy-change advice"
    );

    // Hard stop persists: the 9th call must not bounce back to execution.
    assert!(
        matches!(guard.record("read_file", &input), RepeatAction::Stop(_)),
        "9th call must stay hard-stopped"
    );
}

#[test]
fn different_inputs_do_not_count_against_each_other() {
    let mut guard = RepeatCallGuard::new();
    let a = json!({"path": "a"});
    let b = json!({"path": "b"});

    assert_eq!(guard.record("read_file", &a), RepeatAction::Proceed); // a=1
    assert_eq!(guard.record("read_file", &b), RepeatAction::Proceed); // b=1
    assert_eq!(guard.record("read_file", &a), RepeatAction::Proceed); // a=2
    match guard.record("read_file", &a) {
        RepeatAction::Hint(_) => {}
        other => panic!("3rd call of input `a` must hint, got {other:?}"), // a=3
    }

    // `b` was called only twice: still below the threshold, unaffected by `a`'s
    // escalation...
    assert_eq!(guard.record("read_file", &b), RepeatAction::Proceed); // b=2
    // ...and escalates only on its OWN 3rd call.
    assert!(
        matches!(guard.record("read_file", &b), RepeatAction::Hint(_)),
        "3rd call of input `b` must hint on its own count"
    );
}

#[test]
fn same_input_under_different_tool_names_is_independent() {
    let mut guard = RepeatCallGuard::new();
    let input = json!({"path": "a"});

    assert_eq!(guard.record("read_file", &input), RepeatAction::Proceed);
    assert_eq!(guard.record("search_code", &input), RepeatAction::Proceed);
    assert_eq!(guard.record("read_file", &input), RepeatAction::Proceed); // read_file=2
    assert_eq!(guard.record("search_code", &input), RepeatAction::Proceed); // search_code=2

    assert!(matches!(
        guard.record("read_file", &input),
        RepeatAction::Hint(_)
    )); // read_file=3
    assert!(matches!(
        guard.record("search_code", &input),
        RepeatAction::Hint(_)
    )); // search_code=3
}

#[test]
fn canonically_equivalent_inputs_count_together() {
    let mut guard = RepeatCallGuard::new();
    let a = json!({"path": "a", "offset": 1});
    let b = json!({"offset": 1, "path": "a"});

    // Same logical JSON object, different key order: must hash to the same
    // canonical key (serde_json Maps sort keys, so serialization is stable).
    assert_eq!(guard.record("read_file", &a), RepeatAction::Proceed);
    assert_eq!(guard.record("read_file", &b), RepeatAction::Proceed);
    assert!(matches!(
        guard.record("read_file", &a),
        RepeatAction::Hint(_)
    ));
}

#[test]
fn recent_calls_evict_oldest_fifo_without_resetting_survivors() {
    let mut guard = RepeatCallGuard::new();

    // Fill the map to its capacity (32) with distinct keys, each called once.
    for i in 0..32 {
        let input = json!({"path": format!("/tmp/f{i}")});
        assert_eq!(
            guard.record("read_file", &input),
            RepeatAction::Proceed,
            "fill call #{i}"
        );
    }

    // A 33rd distinct key evicts the oldest entry (/tmp/f0, FIFO).
    assert_eq!(
        guard.record("read_file", &json!({"path": "/tmp/f33"})),
        RepeatAction::Proceed
    );

    // The evicted entry restarts fresh: its 1st call after eviction is Proceed.
    assert_eq!(
        guard.record("read_file", &json!({"path": "/tmp/f0"})),
        RepeatAction::Proceed,
        "evicted entry must be forgotten, not kept with a stale count"
    );

    // A survivor keeps its running count across the eviction: f1 was called
    // once during the fill; the 2nd call is Proceed and the 3rd escalates,
    // proving the other insertions/evictions did not reset its counter.
    assert_eq!(
        guard.record("read_file", &json!({"path": "/tmp/f1"})),
        RepeatAction::Proceed,
        "survivor 2nd call must still be below threshold"
    );
    assert!(
        matches!(
            guard.record("read_file", &json!({"path": "/tmp/f1"})),
            RepeatAction::Hint(_)
        ),
        "survivor 3rd call must escalate: its count must persist across evictions"
    );
}

// ---------------------------------------------------------------------------
// ToolTimeout: declarative timeout_ms on ToolDefinition
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_with_timeout_times_out_and_reports_timed_out() {
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(StubTool {
        name: "slow_stub",
        timeout_ms: Some(50),
        sleep_ms: 200,
    }));

    let calls = vec![ToolCall {
        id: "c1".into(),
        name: "slow_stub".into(),
        input: json!({}),
    }];
    let results = run_batch(&tools, calls).await;

    let res = &results[0];
    assert!(!res.success, "timed-out tool must be reported as a failure");
    assert!(res.timed_out, "timed-out tool must set timed_out=true");
    let err = res
        .error
        .as_deref()
        .expect("timed-out result must carry an error");
    assert!(
        err.contains("slow_stub"),
        "timeout error must mention the tool name: {err}"
    );
    assert!(
        err.contains("50"),
        "timeout error must mention the timeout value in ms: {err}"
    );
}

#[tokio::test]
async fn tool_without_timeout_is_unaffected() {
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(StubTool {
        name: "no_timeout_stub",
        timeout_ms: None,
        sleep_ms: 60,
    }));

    let calls = vec![ToolCall {
        id: "c1".into(),
        name: "no_timeout_stub".into(),
        input: json!({}),
    }];
    let results = run_batch(&tools, calls).await;

    let res = &results[0];
    assert!(
        res.success,
        "tool without timeout_ms must run to completion"
    );
    assert!(!res.timed_out);
    assert_eq!(res.output, "done");
}

#[tokio::test]
async fn tool_finishing_before_timeout_succeeds() {
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(StubTool {
        name: "fast_stub",
        timeout_ms: Some(100),
        sleep_ms: 10,
    }));

    let calls = vec![ToolCall {
        id: "c1".into(),
        name: "fast_stub".into(),
        input: json!({}),
    }];
    let results = run_batch(&tools, calls).await;

    let res = &results[0];
    assert!(res.success, "tool finishing within timeout must succeed");
    assert!(!res.timed_out);
    assert_eq!(res.output, "done");
}

// ---------------------------------------------------------------------------
// ToolResult.timed_out: serde backward compatibility
// ---------------------------------------------------------------------------

#[test]
fn old_serialized_tool_results_without_timed_out_deserialize_to_false() {
    // With an error field present (pre-P6 serialized shape).
    let old = r#"{"call_id":"c1","success":false,"output":"","error":"boom"}"#;
    let res: ToolResult = serde_json::from_str(old).unwrap();
    assert!(!res.timed_out, "missing timed_out must default to false");

    // With the error field omitted (skip_serializing_if shape).
    let old_no_error = r#"{"call_id":"c2","success":true,"output":"ok"}"#;
    let res: ToolResult = serde_json::from_str(old_no_error).unwrap();
    assert!(!res.timed_out, "missing timed_out must default to false");
}

#[test]
fn timed_out_roundtrips_through_serde() {
    for timed_out in [false, true] {
        let res = ToolResult {
            call_id: "c1".into(),
            success: false,
            output: "stalled".into(),
            error: Some("timeout".into()),
            timed_out,
        };
        let encoded = serde_json::to_string(&res).unwrap();
        let decoded: ToolResult = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.timed_out, timed_out);
        assert_eq!(decoded.call_id, "c1");
        assert_eq!(decoded.error.as_deref(), Some("timeout"));
        assert_eq!(decoded.output, "stalled");
    }
}
