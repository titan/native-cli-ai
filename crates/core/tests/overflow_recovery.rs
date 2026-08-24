//! P3 — Overflow recovery: integration matrix C6, C6b, C7.
//!
//! Spec (authoritative): `docs/plans/p3-compaction-design.md` (§Test matrix).
//! Unit tests C1–C5, C8, C10 live inline in `src/`; these tests pin the
//! WIRING between a real `AgentLoop`, `OverflowRecoveryMiddleware`
//! (pushed via `push_middleware`), and the scripted provider through
//! `run_turn` — the roadmap acceptance conditions.
//!
//! Run: `cargo test -p nca-core --test overflow_recovery`

use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use nca_common::config::{PermissionConfig, PermissionMode, SmartCompactionMode};
use nca_common::event::AgentEvent;
use nca_common::message::{Message, MessageToolCall};
use nca_common::tool::ToolDefinition;
use nca_core::agent::AgentLoop;
use nca_core::approval::ApprovalPolicy;
use nca_core::middleware::{CompactionMiddleware, OverflowRecoveryMiddleware};
use nca_core::provider::{Provider, ProviderError, StreamChunk};
use nca_core::tools::ToolRegistry;
use serde_json::json;

/// Realistic OpenAI-style overflow body (the exact shape all OpenAI-compat
/// providers route to `RequestFailed` — DeepSeek included, design §2).
const OVERFLOW_BODY: &str = r#"{"error":{"message":"This model's maximum context length is 65536 tokens. However, you requested 90124 tokens (88680 in the messages, 1444 in the completion). Please reduce the length of the messages or completion.","type":"invalid_request_error","param":null,"code":"context_length_exceeded"}}"#;

/// Scripted provider: fails the first `fail_calls` calls with a context
/// overflow `RequestFailed`, then replays the scripted round and closes the
/// channel. RECORDS every call's `&[Message]` slice, per call index, so
/// tests can assert exactly what the (middleware-pruned) request contained.
struct OverflowThenOkProvider {
    rounds: Vec<Vec<StreamChunk>>,
    fail_calls: usize,
    calls: AtomicU32,
    recorded: Arc<Mutex<Vec<Vec<Message>>>>,
}

impl OverflowThenOkProvider {
    fn new(rounds: Vec<Vec<StreamChunk>>, fail_calls: usize) -> Self {
        Self {
            rounds,
            fail_calls,
            calls: AtomicU32::new(0),
            recorded: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn call_count(&self) -> u32 {
        self.calls.load(Ordering::SeqCst)
    }

    fn recorded(&self) -> Vec<Vec<Message>> {
        self.recorded.lock().unwrap().clone()
    }
}

#[async_trait]
impl Provider for OverflowThenOkProvider {
    async fn chat(
        &self,
        messages: &[Message],
        _tools: &[ToolDefinition],
        _model: &str,
        _workspace_root: &Path,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError> {
        let index = self.calls.fetch_add(1, Ordering::SeqCst) as usize;
        self.recorded.lock().unwrap().push(messages.to_vec());

        if index < self.fail_calls {
            return Err(ProviderError::RequestFailed(OVERFLOW_BODY.into()));
        }

        // Success round for call `index`: the scripted rounds replay from
        // the first successful call onward (call N's round is
        // `rounds[N - fail_calls]`).
        let round = self
            .rounds
            .get(index.saturating_sub(self.fail_calls))
            .cloned()
            .unwrap_or_default();
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        tokio::spawn(async move {
            for chunk in round {
                let _ = tx.send(chunk).await;
            }
        });
        Ok(rx)
    }
}

/// Agent under test. BypassPermissions so tool calls flow through the
/// pipeline without interactive approval (mirrors `middleware.rs`'s
/// `test_agent`).
fn test_agent(
    provider: Arc<dyn Provider>,
    event_tx: tokio::sync::mpsc::Sender<AgentEvent>,
) -> AgentLoop {
    AgentLoop::new(
        provider,
        ToolRegistry::new(),
        ApprovalPolicy::new(PermissionConfig {
            mode: PermissionMode::BypassPermissions,
            ..Default::default()
        }),
        "test-model".into(),
        event_tx,
        10, // max_turns (per-turn step budget)
        16, // max_tool_calls_per_turn
        0,  // checkpoint_interval
        None,
    )
}

/// Drain every event buffered after `run_turn` returns.
fn drain_events(mut rx: tokio::sync::mpsc::Receiver<AgentEvent>) -> Vec<AgentEvent> {
    let mut events = Vec::new();
    while let Ok(e) = rx.try_recv() {
        events.push(e);
    }
    events
}

fn read_call(id: &str) -> MessageToolCall {
    MessageToolCall {
        id: id.into(),
        name: "read_file".into(),
        arguments: json!({"path": "a.rs"}),
    }
}

/// Prunable history: system + old compactible `read_file` tool groups +
/// a long recent tail, so rung 0 (keep last 8 groups) has material to
/// delete. Seed into `agent.messages` BEFORE the turn: the driver's request
/// view (smart compaction Off by default) is the full history clone, so the
/// middleware sees every group.
fn prunable_history() -> Vec<Message> {
    let mut messages = vec![Message::system("sys")];
    for i in 0..6 {
        messages.push(Message::assistant_with_tool_calls(
            "",
            vec![read_call(&format!("c{i}"))],
        ));
        messages.push(Message::tool(&format!("c{i}"), &format!("read output {i}")));
    }
    for i in 0..9 {
        messages.push(Message::user(format!("u{i}")));
        messages.push(Message::assistant(format!("a{i}")));
    }
    messages
}

fn texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .map(|m| m.content.to_summary_text())
        .collect()
}

/// Pull the compaction-bracket events out of a full event stream, keeping
/// their relative order.
fn bracket_events(events: &[AgentEvent]) -> Vec<(bool, usize, String, bool)> {
    // (is_end, tokens, reason-or-empty, kv_prefix_broken-or-false)
    events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ContextCompactionStart {
                tokens_before,
                reason,
            } => Some((false, *tokens_before, reason.clone(), false)),
            AgentEvent::ContextCompactionEnd {
                tokens_after,
                kv_prefix_broken,
            } => Some((true, *tokens_after, String::new(), *kv_prefix_broken)),
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// C6 — roadmap acceptance 1: full AgentLoop overflow → prune → retry
// ---------------------------------------------------------------------------

/// C6 — acceptance: a real `AgentLoop` with `OverflowRecoveryMiddleware`
/// pushed via `push_middleware` recovers from a first-call overflow: the
/// turn succeeds, the event stream carries the `ContextCompactionStart
/// (overflow_prune)` → `End (kv_prefix_broken=true)` bracket, and the
/// provider's second call received FEWER messages than the first (the
/// pruned view). Canonical `agent.messages` keeps the pruned groups — the
/// middleware rewrites views only.
#[tokio::test]
async fn c6_agent_loop_recovers_overflow_via_prune_retry() {
    let provider = Arc::new(OverflowThenOkProvider::new(
        vec![vec![
            StreamChunk::TextDelta("recovered fine".into()),
            StreamChunk::Finish {
                reason: "stop".into(),
            },
        ]],
        1, // first call overflows, second succeeds
    ));
    let (event_tx, event_rx) = tokio::sync::mpsc::channel(256);

    let mut agent = test_agent(Arc::clone(&provider) as Arc<dyn Provider>, event_tx);
    agent.messages = prunable_history();
    agent.push_middleware(Arc::new(OverflowRecoveryMiddleware::default()));

    let result = agent
        .run_turn("check this", Path::new("."), &[])
        .await
        .expect("overflow must be recovered by pruning + retry");
    assert_eq!(result, "recovered fine");

    assert_eq!(
        provider.call_count(),
        2,
        "exactly 2 provider calls: 1 overflow + 1 pruned retry"
    );

    let recorded = provider.recorded();
    assert_eq!(recorded.len(), 2, "one recorded messages slice per call");
    let attempt1 = texts(&recorded[0]);
    let attempt2 = texts(&recorded[1]);
    assert!(
        attempt1.iter().any(|t| t.contains("read output")),
        "attempt 1 must carry the full history incl. prunable groups"
    );
    assert!(
        !attempt2.iter().any(|t| t.contains("read output")),
        "attempt 2 must lack the pruned tool groups: {attempt2:?}"
    );
    assert!(
        attempt2.iter().any(|t| t == "sys"),
        "attempt 2 keeps the system message"
    );
    assert!(
        attempt2.iter().any(|t| t == "check this"),
        "attempt 2 keeps the user prompt"
    );
    assert!(
        recorded[1].len() < recorded[0].len(),
        "provider's 2nd call must receive FEWER messages than the 1st: {} -> {}",
        recorded[0].len(),
        recorded[1].len()
    );

    // Bracket on the stream: Start(overflow_prune, tokens_before > 0) then
    // End(kv_prefix_broken=true, tokens_after < tokens_before), in order.
    let brackets = bracket_events(&drain_events(event_rx));
    assert_eq!(brackets.len(), 2, "one Start + one End: {brackets:?}");
    let (start_is_end, start_tokens, start_reason, _) = &brackets[0];
    assert!(!*start_is_end);
    assert_eq!(start_reason, "overflow_prune");
    assert!(*start_tokens > 0);
    let (end_is_end, end_tokens, _, kv_broken) = &brackets[1];
    assert!(*end_is_end);
    assert!(*kv_broken, "prune broke the warmed KV prefix");
    assert!(
        *end_tokens < *start_tokens,
        "End reports a smaller view: {start_tokens} -> {end_tokens}"
    );

    // Canonical history untouched by the middleware: the seeded groups
    // survive the turn (rollback only truncates on Err; this turn was Ok).
    let canonical = texts(&agent.messages);
    assert!(
        canonical.iter().any(|t| t.contains("read output")),
        "canonical agent.messages must retain the tool groups (view-only prune)"
    );
}

// ---------------------------------------------------------------------------
// C6b — escalation integration: nothing prunable + always-overflow
// ---------------------------------------------------------------------------

/// C6b — escalation: with a view that has NOTHING prunable (no tool groups),
/// an always-overflowing provider makes `run_turn` return the ORIGINAL
/// overflow error, message preserved. Call count: exactly ONE — the design
/// §4 loop escalates immediately on the first no-progress prune, and the
/// prune itself is a pure local computation, not a provider call (the ≤3
/// roadmap bound is a ceiling on retries, never reached).
#[tokio::test]
async fn c6b_agent_loop_escalates_original_overflow_when_nothing_prunable() {
    let provider = Arc::new(OverflowThenOkProvider::new(
        vec![], // never reached
        usize::MAX,
    ));
    let (event_tx, event_rx) = tokio::sync::mpsc::channel(256);

    let mut agent = test_agent(Arc::clone(&provider) as Arc<dyn Provider>, event_tx);
    // Minimal history: system + plain turns. No ToolGroups ⇒ prune finds 0.
    agent.messages = vec![
        Message::system("sys"),
        Message::user("earlier"),
        Message::assistant("earlier reply"),
    ];
    agent.push_middleware(Arc::new(OverflowRecoveryMiddleware::default()));

    let err = agent
        .run_turn("please", Path::new("."), &[])
        .await
        .expect_err("unrecoverable overflow must surface to the caller");
    assert!(
        err.is_context_overflow(),
        "the surfaced error must be the original overflow classification"
    );
    assert!(
        err.to_string().contains("maximum context length"),
        "the original overflow body must be preserved: {err}"
    );

    // Exact call count: ONE. The design §4 loop escalates immediately on
    // the first no-progress prune — `pruned_groups == 0` returns the
    // original error WITHOUT another `next.run` (the prune is a pure local
    // computation, not a provider call). The "initial + 1 prune attempt"
    // framing counts the prune as a provider interaction, which it is not;
    // the ≤3 roadmap bound is a ceiling on retries, never reached here.
    assert_eq!(
        provider.call_count(),
        1,
        "design §4: initial call, then immediate escalation on 0-group prune \
         (no provider re-entry, no spin)"
    );
    let recorded = provider.recorded();
    assert_eq!(recorded.len(), 1);

    // The bracket is still emitted before escalation (audit value even
    // without a retry), with kv_prefix_broken=false (nothing was pruned).
    let brackets = bracket_events(&drain_events(event_rx));
    assert_eq!(
        brackets.len(),
        2,
        "Start + End before escalation: {brackets:?}"
    );
    assert_eq!(brackets[0].2, "overflow_prune");
    assert!(!brackets[1].3, "no groups pruned ⇒ KV prefix untouched");
}

// ---------------------------------------------------------------------------
// C7 — roadmap acceptance 3: DryRun reporting must not regress
// ---------------------------------------------------------------------------

/// C7 — DryRun: with `SmartCompactionMode::DryRun`, the existing per-step
/// `ContextCompaction { phase: "dry_run" }` event still reports a planned
/// token decrease, and the provider still receives the UNCOMPACTED view
/// (full tool output, byte-for-byte history clone). Pins that P3's bracket
/// events did not disturb the routine per-step emission.
#[tokio::test]
async fn c7_dry_run_reports_decrease_and_provider_gets_uncompacted_view() {
    let big_output = "x".repeat(2_000);
    let provider = Arc::new(OverflowThenOkProvider::new(
        vec![vec![
            StreamChunk::TextDelta("done".into()),
            StreamChunk::Finish {
                reason: "stop".into(),
            },
        ]],
        0, // never overflow
    ));
    let (event_tx, event_rx) = tokio::sync::mpsc::channel(256);

    let mut agent = test_agent(Arc::clone(&provider) as Arc<dyn Provider>, event_tx);
    agent.messages = vec![Message::system("sys")];
    agent.messages.push(Message::assistant_with_tool_calls(
        "",
        vec![read_call("c1")],
    ));
    agent.messages.push(Message::tool("c1", &big_output));
    for i in 0..10 {
        agent.messages.push(Message::user(format!("u{i}")));
        agent.messages.push(Message::assistant(format!("a{i}")));
    }
    agent.push_middleware(Arc::new(CompactionMiddleware::new(
        SmartCompactionMode::DryRun,
    )));

    let result = agent
        .run_turn("final", Path::new("."), &[])
        .await
        .expect("dry-run turn succeeds");
    assert_eq!(result, "done");

    // Provider saw the UNCOMPACTED view: the full 2000-char output must
    // still be present (DryRun plans but sends the history clone).
    let recorded = provider.recorded();
    assert_eq!(recorded.len(), 1, "one step, one call");
    assert!(
        recorded[0]
            .iter()
            .any(|m| m.content.to_summary_text().contains(&big_output)),
        "provider must receive the uncompacted view in DryRun mode"
    );

    // Per-step event: phase "dry_run" with decreasing tokens_before/after.
    let events = drain_events(event_rx);
    let dry_runs: Vec<&AgentEvent> = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::ContextCompaction { phase, .. } if phase == "dry_run"))
        .collect();
    assert_eq!(
        dry_runs.len(),
        1,
        "exactly one per-step dry_run event for the single step"
    );
    match dry_runs[0] {
        AgentEvent::ContextCompaction {
            phase,
            tokens_before,
            tokens_after,
            retained_groups,
            dropped_groups,
            ..
        } => {
            assert_eq!(phase, "dry_run");
            let before = tokens_before.expect("dry_run reports tokens_before");
            let after = tokens_after.expect("dry_run reports tokens_after");
            assert!(
                after < before,
                "dry_run must report the planned decrease: {before} -> {after}"
            );
            assert!(
                retained_groups.is_some() && dropped_groups.is_some(),
                "dry_run keeps the group accounting"
            );
        }
        other => panic!("unexpected event: {other:?}"),
    }
    // And no replacement bracket may appear on the dry-run path (P3 §1:
    // routine per-step compaction keeps the single informational event).
    assert!(
        !events.iter().any(|e| matches!(
            e,
            AgentEvent::ContextCompactionStart { .. } | AgentEvent::ContextCompactionEnd { .. }
        )),
        "DryRun must not emit replacement-type bracket events"
    );
    assert_eq!(
        texts(&agent.messages)
            .iter()
            .filter(|t| t.contains(&big_output))
            .count(),
        1,
        "canonical history keeps the full tool output"
    );
}
