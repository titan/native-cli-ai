//! P4 — Waterfall middleware layer: integration matrix M6–M9.
//!
//! Spec (authoritative): `docs/plans/p4-middleware-design.md` (§Test matrix).
//! Unit tests M1–M5 live inline in `src/middleware.rs`; these tests pin the
//! WIRING between a real `AgentLoop` and the chain through `run_turn`, so a
//! future `TurnDriver::step` refactor that bypasses `agent.middleware`
//! fails loudly here (the diff-verification lesson from Phase C).
//!
//! Run: `cargo test -p nca-core --test middleware`

use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use nca_common::config::{PermissionConfig, PermissionMode};
use nca_common::event::AgentEvent;
use nca_common::message::{ImageAttachment, Message, MessageContent, Role};
use nca_common::tool::{ToolCall, ToolDefinition};
use nca_core::agent::AgentLoop;
use nca_core::approval::ApprovalPolicy;
use nca_core::middleware::{AgentMiddleware, Next, StepReply, StepRequest};
use nca_core::provider::{Provider, ProviderError, StreamChunk};
use nca_core::tools::ToolRegistry;
use serde_json::json;

/// Scripted provider: each `chat()` call replays the next scripted round of
/// `StreamChunk`s, then closes the channel (the agent loop treats channel
/// close as end of stream). Mirrors `turn_driver.rs`'s capturing provider:
/// it RECORDS the `&[Message]` slice it was called with, per call index, so
/// tests can assert exactly what the (possibly middleware-rewritten) request
/// contained.
struct CapturingScriptedProvider {
    rounds: Vec<Vec<StreamChunk>>,
    calls: AtomicU32,
    recorded: Arc<Mutex<Vec<Vec<Message>>>>,
}

impl CapturingScriptedProvider {
    fn new(rounds: Vec<Vec<StreamChunk>>) -> Self {
        Self {
            rounds,
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
impl Provider for CapturingScriptedProvider {
    async fn chat(
        &self,
        messages: &[Message],
        _tools: &[ToolDefinition],
        _model: &str,
        _workspace_root: &Path,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError> {
        let index = self.calls.fetch_add(1, Ordering::SeqCst) as usize;
        self.recorded.lock().unwrap().push(messages.to_vec());

        let round = self.rounds.get(index).cloned().unwrap_or_default();
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        tokio::spawn(async move {
            for chunk in round {
                let _ = tx.send(chunk).await;
            }
        });
        Ok(rx)
    }
}

/// Agent under test. BypassPermissions so the scripted `echo` tool calls flow
/// through the pipeline without interactive approval (mirrors the recipe in
/// `agent_driver.rs` t14 / `turn_driver.rs`).
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

/// Drain every event buffered after `run_turn` returns (the agent emits into
/// a bounded channel; a turn's events are all queued once the turn completes).
fn drain_events(mut rx: tokio::sync::mpsc::Receiver<AgentEvent>) -> Vec<AgentEvent> {
    let mut events = Vec::new();
    while let Ok(e) = rx.try_recv() {
        events.push(e);
    }
    events
}

/// Whether `messages` contains a user message whose text is exactly `text`.
fn has_user_text(messages: &[Message], text: &str) -> bool {
    messages
        .iter()
        .any(|m| m.role == Role::User && m.content.to_summary_text() == text)
}

/// M6 — rewriting middleware: counts invocations and appends a marker user
/// message to the REQUEST view before proceeding (canonical history untouched).
struct MarkerMiddleware {
    invocations: Arc<AtomicU32>,
}

#[async_trait]
impl AgentMiddleware for MarkerMiddleware {
    fn name(&self) -> &str {
        "marker"
    }

    async fn call(&self, mut req: StepRequest, next: Next<'_>) -> Result<StepReply, ProviderError> {
        self.invocations.fetch_add(1, Ordering::SeqCst);
        req.messages.push(Message::user("[mw-marker]"));
        next.run(req).await
    }
}

/// M7/M7a/M7b — short-circuiting middleware: returns `FinalText` without
/// ever calling `next` (provider and inner layers skipped).
struct ShortCircuitMiddleware {
    text: String,
}

#[async_trait]
impl AgentMiddleware for ShortCircuitMiddleware {
    fn name(&self) -> &str {
        "short-circuit"
    }

    async fn call(&self, _req: StepRequest, _next: Next<'_>) -> Result<StepReply, ProviderError> {
        Ok(StepReply::FinalText(self.text.clone()))
    }
}

/// M9 — informational emitter: sends a `Checkpoint` on the shared event
/// handle BEFORE proceeding. Deliberately emits NO `MessageRecorded`
/// (projection is driver-owned by convention, design §4).
struct CheckpointEmitter;

#[async_trait]
impl AgentMiddleware for CheckpointEmitter {
    fn name(&self) -> &str {
        "checkpoint-emitter"
    }

    async fn call(&self, req: StepRequest, next: Next<'_>) -> Result<StepReply, ProviderError> {
        let _ = req
            .event_tx
            .send(AgentEvent::Checkpoint {
                phase: "mw".into(),
                detail: "middleware observed the step request".into(),
                turn: req.turn_id as u32,
            })
            .await;
        next.run(req).await
    }
}

// ---------------------------------------------------------------------------
// M6 — wiring pin: real AgentLoop + rewriting middleware, two-step turn
// ---------------------------------------------------------------------------

/// M6 — wiring pin: a real `AgentLoop::with_middleware` chain is invoked
/// exactly once per step, and BOTH provider calls observe the rewritten
/// request (`[mw-marker]`), while canonical `agent.messages` never contains
/// the marker. Fails if `TurnDriver::step` bypasses `agent.middleware`.
#[tokio::test]
async fn m6_wiring_pin_agent_loop_calls_middleware_chain() {
    // Round 1: a tool call (the unregistered `echo` tool fails in the
    // pipeline, but that still yields a valid tool message — exactly the
    // turn_driver.rs pattern, which keeps the step loop going). Round 2:
    // final text + finish.
    let provider = Arc::new(CapturingScriptedProvider::new(vec![
        vec![StreamChunk::ToolUse(ToolCall {
            id: "t1".into(),
            name: "echo".into(),
            input: json!({}),
        })],
        vec![
            StreamChunk::TextDelta("done".into()),
            StreamChunk::Finish {
                reason: "stop".into(),
            },
        ],
    ]));
    let invocations = Arc::new(AtomicU32::new(0));
    let (event_tx, _event_rx) = tokio::sync::mpsc::channel(256);

    let mut agent = test_agent(Arc::clone(&provider) as Arc<dyn Provider>, event_tx)
        .with_middleware(Arc::new(MarkerMiddleware {
            invocations: Arc::clone(&invocations),
        }));

    let result = agent
        .run_turn("do the thing", Path::new("."), &[])
        .await
        .expect("scripted 2-step turn succeeds");
    assert_eq!(result, "done", "final step text is the turn result");

    assert_eq!(
        invocations.load(Ordering::SeqCst),
        2,
        "middleware must run exactly once per step (2 steps: tool call → final)"
    );
    assert_eq!(
        provider.call_count(),
        2,
        "2 rounds scripted ⇒ exactly 2 provider chat calls, both behind the chain"
    );

    let recorded = provider.recorded();
    assert_eq!(recorded.len(), 2, "one recorded messages slice per call");
    for (i, call) in recorded.iter().enumerate() {
        assert!(
            has_user_text(call, "[mw-marker]"),
            "provider call {i} must observe the middleware-rewritten request: {:?}",
            call.iter()
                .map(|m| m.content.to_summary_text())
                .collect::<Vec<_>>()
        );
    }

    assert!(
        !agent
            .messages
            .iter()
            .any(|m| m.content.to_summary_text() == "[mw-marker]"),
        "canonical agent.messages must never contain the middleware-only marker"
    );
}

// ---------------------------------------------------------------------------
// M7 — short-circuit end-to-end
// ---------------------------------------------------------------------------

/// M7 — short-circuit end-to-end: a middleware that returns `FinalText`
/// without calling `next` becomes the turn's answer. Provider 0 calls;
/// `agent.messages` = [user, assistant]; the event stream carries
/// `MessageRecorded`(assistant), `MessageReceived`(assistant) and
/// `TurnCompleted`.
#[tokio::test]
async fn m7_short_circuit_end_to_end() {
    let provider = Arc::new(CapturingScriptedProvider::new(vec![vec![
        StreamChunk::TextDelta("must never be consumed".into()),
    ]]));
    let (event_tx, event_rx) = tokio::sync::mpsc::channel(256);

    let mut agent = test_agent(Arc::clone(&provider) as Arc<dyn Provider>, event_tx)
        .with_middleware(Arc::new(ShortCircuitMiddleware {
            text: "blocked by policy".into(),
        }));

    let result = agent
        .run_turn("please", Path::new("."), &[])
        .await
        .expect("short-circuit turn succeeds");
    assert_eq!(result, "blocked by policy");
    assert_eq!(
        provider.call_count(),
        0,
        "provider must not be called when the middleware short-circuits"
    );

    assert_eq!(
        agent.messages,
        vec![
            Message::user("please"),
            Message::assistant("blocked by policy"),
        ],
        "short-circuited turn must record user + assistant, replay-safe"
    );

    let events = drain_events(event_rx);
    let recorded_assistant: Vec<&Message> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::MessageRecorded { message } if message.role == Role::Assistant => {
                Some(message)
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        recorded_assistant.len(),
        1,
        "exactly one assistant message recorded"
    );
    assert_eq!(
        recorded_assistant[0].content.to_summary_text(),
        "blocked by policy"
    );

    assert!(
        events.iter().any(|e| matches!(
            e,
            AgentEvent::MessageReceived {
                role,
                content,
                steering,
            } if role == "assistant" && content == "blocked by policy" && !steering
        )),
        "short-circuit final text must surface as a MessageReceived(assistant)"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::TurnCompleted { .. })),
        "run_turn must still emit TurnCompleted after a short-circuit"
    );
}

// ---------------------------------------------------------------------------
// M7a — short-circuit attachment cleanup (oracle P1-1)
// ---------------------------------------------------------------------------

/// M7a — attachment cleanup on short-circuit: with one `ImageAttachment`,
/// the middleware short-circuits, and the on-disk image is removed while no
/// `ContentPart::Image` survives in `agent.messages` (the user message's
/// image path is stripped to a text placeholder, oracle P1-1).
#[tokio::test]
async fn m7a_short_circuit_cleans_up_attachments() {
    let ws = tempfile::tempdir().expect("tempdir workspace");
    let img_dir = ws.path().join("attachments");
    std::fs::create_dir_all(&img_dir).expect("create attachments dir");
    let img_path = img_dir.join("img.png");
    std::fs::write(&img_path, b"fake png bytes").expect("write image file");
    let attachment = ImageAttachment {
        media_type: "image/png".into(),
        path: "attachments/img.png".into(),
    };

    let provider = Arc::new(CapturingScriptedProvider::new(vec![vec![
        StreamChunk::TextDelta("must never be consumed".into()),
    ]]));
    let (event_tx, _event_rx) = tokio::sync::mpsc::channel(256);

    let mut agent = test_agent(Arc::clone(&provider) as Arc<dyn Provider>, event_tx)
        .with_middleware(Arc::new(ShortCircuitMiddleware {
            text: "blocked by policy".into(),
        }));

    let result = agent
        .run_turn("look at this", ws.path(), &[attachment])
        .await
        .expect("short-circuit turn succeeds");
    assert_eq!(result, "blocked by policy");
    assert_eq!(provider.call_count(), 0);

    assert!(
        !img_path.exists(),
        "short-circuited turn must remove the processed image from disk"
    );
    assert!(
        !agent.messages.iter().any(|m| m.content.has_image_parts()),
        "no ContentPart::Image may survive in agent.messages after cleanup"
    );
    match &agent.messages[0].content {
        MessageContent::Text(t) => assert!(
            t.contains("image processed and removed"),
            "user message image path must be replaced by the cleanup placeholder: {t:?}"
        ),
        other => panic!("user message content must collapse to text after cleanup, got {other:?}"),
    }
    // Sanity: the turn still recorded the short-circuit answer.
    assert_eq!(agent.messages.len(), 2, "[user, assistant]");
    assert_eq!(
        agent.messages[1].content.to_summary_text(),
        "blocked by policy"
    );
}

// ---------------------------------------------------------------------------
// M7b — empty short-circuit fails loudly
// ---------------------------------------------------------------------------

/// M7b — empty short-circuit fails loudly: `FinalText("  ")` (whitespace)
/// must make `run_turn` return `Err` naming the empty-final-text
/// short-circuit; no assistant message is recorded and the provider is never
/// called. NOTE: `run_turn`'s P1 failure policy truncates to the baseline
/// captured BEFORE the turn's user message push, so a fresh agent's history
/// rolls back to empty — "nothing recorded" per design M7b, not [user].
#[tokio::test]
async fn m7b_empty_short_circuit_fails_loudly() {
    let provider = Arc::new(CapturingScriptedProvider::new(vec![vec![
        StreamChunk::TextDelta("must never be consumed".into()),
    ]]));
    let (event_tx, event_rx) = tokio::sync::mpsc::channel(256);

    let mut agent = test_agent(Arc::clone(&provider) as Arc<dyn Provider>, event_tx)
        .with_middleware(Arc::new(ShortCircuitMiddleware { text: "   ".into() }));

    let err = agent
        .run_turn("please", Path::new("."), &[])
        .await
        .expect_err("empty short-circuit text must fail the turn loudly");
    assert!(
        err.to_string()
            .contains("middleware short-circuited the step with empty final text"),
        "error must name the empty-final-text short-circuit: {err}"
    );
    assert_eq!(
        provider.call_count(),
        0,
        "provider must not be called for an empty short-circuit"
    );

    assert!(
        agent.messages.is_empty(),
        "P1 rollback truncates to the pre-turn baseline (before the user push), so a fresh \
         agent ends empty — the empty short-circuit must never leave an assistant message: {:?}",
        agent.messages
    );

    let events = drain_events(event_rx);
    assert!(
        !events.iter().any(|e| matches!(
            e,
            AgentEvent::MessageRecorded { message } if message.role == Role::Assistant
        )),
        "no assistant message may be recorded on the empty short-circuit path"
    );
}

// ---------------------------------------------------------------------------
// M8 — default chain (no middleware)
// ---------------------------------------------------------------------------

/// M8 — default chain: a plain `AgentLoop::new` (NO `with_middleware`) with
/// a scripted final-text round must still succeed and record the assistant
/// message — guards against default-chain misconstruction (an empty chain
/// must be observably identical to a bare provider call).
#[tokio::test]
async fn m8_default_chain_runs_plain_turn() {
    let provider = Arc::new(CapturingScriptedProvider::new(vec![vec![
        StreamChunk::TextDelta("default chain ".into()),
        StreamChunk::TextDelta("works".into()),
        StreamChunk::Finish {
            reason: "stop".into(),
        },
    ]]));
    let (event_tx, _event_rx) = tokio::sync::mpsc::channel(256);

    // NOTE: deliberately NO with_middleware — the default empty chain.
    let mut agent = test_agent(Arc::clone(&provider) as Arc<dyn Provider>, event_tx);

    let result = agent
        .run_turn("hi", Path::new("."), &[])
        .await
        .expect("default-chain turn must succeed");
    assert_eq!(result, "default chain works");
    assert_eq!(provider.call_count(), 1, "exactly one provider call");
    assert_eq!(
        agent.messages,
        vec![
            Message::user("hi"),
            Message::assistant("default chain works")
        ],
        "plain turn must record user + assistant"
    );
}

// ---------------------------------------------------------------------------
// M9 — informational event emission via req.event_tx
// ---------------------------------------------------------------------------

/// M9 — informational event emission: a middleware MAY emit events through
/// `req.event_tx` (a `Checkpoint` here) and they land in the agent's event
/// stream — exercising the `event_tx` surface so it does not ship untested.
/// The middleware does NOT emit `MessageRecorded` (projection stays
/// driver-owned by convention).
#[tokio::test]
async fn m9_middleware_informational_event_lands_in_event_stream() {
    let provider = Arc::new(CapturingScriptedProvider::new(vec![vec![
        StreamChunk::TextDelta("done".into()),
        StreamChunk::Finish {
            reason: "stop".into(),
        },
    ]]));
    let (event_tx, event_rx) = tokio::sync::mpsc::channel(256);

    let mut agent = test_agent(Arc::clone(&provider) as Arc<dyn Provider>, event_tx)
        .with_middleware(Arc::new(CheckpointEmitter));

    let result = agent
        .run_turn("hi", Path::new("."), &[])
        .await
        .expect("turn behind an event-emitting middleware succeeds");
    assert_eq!(result, "done");
    assert_eq!(
        provider.call_count(),
        1,
        "middleware proceeded to the provider"
    );

    let events = drain_events(event_rx);
    let mw_checkpoints: Vec<&AgentEvent> = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::Checkpoint { phase, .. } if phase == "mw"))
        .collect();
    assert_eq!(
        mw_checkpoints.len(),
        1,
        "exactly one middleware Checkpoint must land in the event stream"
    );
    match mw_checkpoints[0] {
        AgentEvent::Checkpoint {
            phase,
            detail,
            turn,
        } => {
            assert_eq!(phase, "mw");
            assert_eq!(*turn, 1, "first turn has turn_id 1");
            assert_eq!(detail, "middleware observed the step request");
        }
        other => panic!("unexpected event: {other:?}"),
    }
}
