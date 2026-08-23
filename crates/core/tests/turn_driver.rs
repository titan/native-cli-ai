//! P1 — Turn/Step layering + single inbox: contract tests (failing-first).
//!
//! Spec (authoritative): `docs/plans/p1-turn-step-design.md` (§P1 of
//! `deepseek-harness-adoption.md`).
//!
//! The API under test does NOT exist yet. These tests encode the public
//! contract and must fail to compile (missing symbols/variants) until the
//! fixer implements it:
//!   - `AgentEvent::TurnStarted { turn_id }`, `StepStarted`, `StepCompleted`
//!   - `AgentEvent::TurnCompleted` gains `turn_id`
//!   - `AgentEvent::MessageReceived` gains `steering`
//!   - `nca_core::agent_driver::InboxItem` + `AgentLoop::inbox_sender()`
//!
//! Run: `cargo test -p nca-core --test turn_driver`

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use nca_common::config::{PermissionConfig, PermissionMode};
use nca_common::event::{AgentEvent, EventEnvelope};
use nca_common::message::{Message, MessageContent, Role};
use nca_common::tool::{ToolCall, ToolDefinition};
use nca_core::agent::AgentLoop;
use nca_core::agent_driver::InboxItem;
use nca_core::approval::ApprovalPolicy;
use nca_core::provider::{Provider, ProviderError, StreamChunk};
use nca_core::tools::ToolRegistry;
use serde_json::json;

/// Per-round hook: invoked synchronously at the start of that round's
/// `chat()` call, with the provider so it can push into a cloned inbox sender
/// mid-turn (this is how steering is injected exactly during a chosen step).
type Hook = Box<dyn Fn(&CapturingScriptedProvider) + Send + Sync>;

/// Scripted provider: each `chat()` call replays the next scripted round of
/// `StreamChunk`s, then closes the channel (the agent loop treats channel
/// close as end of stream).
///
/// Unlike a plain scripted provider it also:
///   - RECORDS the `&[Message]` slice it was called with, per call index;
///   - runs an optional per-round hook before streaming, so a test can inject
///     `InboxItem`s into the agent's inbox exactly during step N.
struct CapturingScriptedProvider {
    rounds: Vec<Vec<StreamChunk>>,
    /// `hooks[i]` runs when the i-th `chat()` call starts.
    hooks: Vec<Option<Hook>>,
    calls: AtomicU32,
    /// Recorded messages slice per `chat()` call (call index → messages).
    recorded: Arc<Mutex<Vec<Vec<Message>>>>,
    /// Clone of the `AgentLoop` inbox sender (set after `AgentLoop::new`).
    inbox: Arc<Mutex<Option<tokio::sync::mpsc::Sender<InboxItem>>>>,
    /// Set once a delayed (post-turn) injection has landed — lets tests wait
    /// deterministically for a leftover item before starting the next turn.
    arrived: Arc<AtomicBool>,
}

impl CapturingScriptedProvider {
    fn new(rounds: Vec<Vec<StreamChunk>>, hooks: Vec<Option<Hook>>) -> Self {
        Self {
            rounds,
            hooks,
            calls: AtomicU32::new(0),
            recorded: Arc::new(Mutex::new(Vec::new())),
            inbox: Arc::new(Mutex::new(None)),
            arrived: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Wire the agent's inbox sender in after `AgentLoop::new`.
    fn set_inbox(&self, sender: tokio::sync::mpsc::Sender<InboxItem>) {
        *self.inbox.lock().unwrap() = Some(sender);
    }

    fn inbox_sender(&self) -> tokio::sync::mpsc::Sender<InboxItem> {
        self.inbox
            .lock()
            .unwrap()
            .as_ref()
            .expect("inbox sender must be wired via set_inbox before a turn")
            .clone()
    }

    fn push_to_inbox(&self, item: InboxItem) {
        let _ = self.inbox_sender().try_send(item);
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

        // Per-round hook: may push InboxItems into the running turn's inbox.
        if let Some(Some(hook)) = self.hooks.get(index) {
            hook(self);
        }

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
/// through the pipeline without interactive approval (the tool is unregistered,
/// producing a failed `ToolResult` — which still yields a valid tool message,
/// exactly what the step loop needs to continue).
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

/// Extract the plain-text content of every user message, in order.
fn user_texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter(|m| m.role == Role::User)
        .filter_map(|m| match &m.content {
            MessageContent::Text(t) => Some(t.clone()),
            _ => None,
        })
        .collect()
}

/// Collect events from `event_rx` until `stop_after_turn_completed`
/// `TurnCompleted` events have been received (the driver emits one per turn).
async fn collect_events_until(
    mut event_rx: tokio::sync::mpsc::Receiver<AgentEvent>,
    stop_after_turn_completed: usize,
) -> Vec<AgentEvent> {
    let mut events = Vec::new();
    let mut completed = 0;
    while let Some(e) = event_rx.recv().await {
        let is_tc = matches!(e, AgentEvent::TurnCompleted { .. });
        events.push(e);
        if is_tc {
            completed += 1;
            if completed >= stop_after_turn_completed {
                break;
            }
        }
    }
    events
}

/// Projection of the turn/step event family, for exact-sequence assertions
/// (other events — busy state, checkpoints, tokens — are filtered out).
#[derive(Debug, PartialEq, Eq)]
enum TurnStepEvent {
    TurnStarted(u64),
    StepStarted(u64, u64),
    StepCompleted(u64, u64, bool),
    TurnCompleted(u64),
}

fn turn_step_seq(events: &[AgentEvent]) -> Vec<TurnStepEvent> {
    events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::TurnStarted { turn_id } => Some(TurnStepEvent::TurnStarted(*turn_id)),
            AgentEvent::StepStarted {
                turn_id,
                step_index,
            } => Some(TurnStepEvent::StepStarted(*turn_id, *step_index)),
            AgentEvent::StepCompleted {
                turn_id,
                step_index,
                had_tool_calls,
                ..
            } => Some(TurnStepEvent::StepCompleted(
                *turn_id,
                *step_index,
                *had_tool_calls,
            )),
            AgentEvent::TurnCompleted { turn_id, .. } => {
                Some(TurnStepEvent::TurnCompleted(*turn_id))
            }
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// T1 — steering injected during step 2 lands in the step-3 provider request
// ---------------------------------------------------------------------------

#[tokio::test]
async fn steering_at_step_boundary() {
    let steering_hook: Hook = Box::new(|p| {
        p.push_to_inbox(InboxItem::Steering {
            text: "focus on tests".into(),
        });
    });

    let provider = Arc::new(CapturingScriptedProvider::new(
        vec![
            vec![StreamChunk::ToolUse(ToolCall {
                id: "t1".into(),
                name: "echo".into(),
                input: json!({}),
            })],
            vec![StreamChunk::ToolUse(ToolCall {
                id: "t2".into(),
                name: "echo".into(),
                input: json!({}),
            })],
            vec![
                StreamChunk::TextDelta("done".into()),
                StreamChunk::Finish {
                    reason: "stop".into(),
                },
            ],
        ],
        vec![None, Some(steering_hook), None],
    ));

    let (event_tx, event_rx) = tokio::sync::mpsc::channel(256);
    let mut agent = test_agent(Arc::clone(&provider) as Arc<dyn Provider>, event_tx);
    // The hook fires mid-turn and needs a live inbox sender.
    provider.set_inbox(agent.inbox_sender());

    let collector = tokio::spawn(collect_events_until(event_rx, 1));

    let result = agent
        .run_turn("do the thing", Path::new("."), &[])
        .await
        .expect("scripted 3-step turn succeeds");

    assert_eq!(result, "done", "final step text is the turn result");
    assert_eq!(
        provider.call_count(),
        3,
        "3 rounds scripted ⇒ exactly 3 provider chat calls"
    );

    let recorded = provider.recorded();
    assert_eq!(recorded.len(), 3, "one recorded messages slice per call");
    let call3_users = user_texts(&recorded[2]);
    assert!(
        call3_users.iter().any(|t| t == "focus on tests"),
        "steering text must appear as a user message in the step-3 request: {call3_users:?}"
    );

    let events = collector.await.expect("event collector task");
    let steering: Vec<&AgentEvent> = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::MessageReceived { steering: true, .. }))
        .collect();
    assert_eq!(steering.len(), 1, "exactly one steering message received");
    match steering[0] {
        AgentEvent::MessageReceived { content, .. } => {
            assert_eq!(content, "focus on tests");
        }
        other => panic!("unexpected event: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// T2 — queued UserPrompts claimed in arrival order at the step boundary
// ---------------------------------------------------------------------------

#[tokio::test]
async fn queued_prompt_claimed_in_order() {
    let queue_hook: Hook = Box::new(|p| {
        p.push_to_inbox(InboxItem::UserPrompt {
            text: "first".into(),
        });
        p.push_to_inbox(InboxItem::UserPrompt {
            text: "second".into(),
        });
    });

    let provider = Arc::new(CapturingScriptedProvider::new(
        vec![
            vec![StreamChunk::ToolUse(ToolCall {
                id: "t1".into(),
                name: "echo".into(),
                input: json!({}),
            })],
            vec![StreamChunk::ToolUse(ToolCall {
                id: "t2".into(),
                name: "echo".into(),
                input: json!({}),
            })],
            vec![
                StreamChunk::TextDelta("done".into()),
                StreamChunk::Finish {
                    reason: "stop".into(),
                },
            ],
        ],
        vec![None, Some(queue_hook), None],
    ));

    let (event_tx, _event_rx) = tokio::sync::mpsc::channel(256);
    let mut agent = test_agent(Arc::clone(&provider) as Arc<dyn Provider>, event_tx);
    provider.set_inbox(agent.inbox_sender());

    let result = agent
        .run_turn("do the thing", Path::new("."), &[])
        .await
        .expect("scripted 3-step turn succeeds");

    assert_eq!(result, "done");
    assert_eq!(provider.call_count(), 3);

    let recorded = provider.recorded();
    assert_eq!(recorded.len(), 3);
    let call3_users = user_texts(&recorded[2]);
    let pos_first = call3_users
        .iter()
        .position(|t| t == "first")
        .expect("'first' must be claimed into the step-3 request");
    let pos_second = call3_users
        .iter()
        .position(|t| t == "second")
        .expect("'second' must be claimed into the step-3 request");
    assert!(
        pos_first < pos_second,
        "queued items must be claimed in arrival order: {call3_users:?}"
    );
}

// ---------------------------------------------------------------------------
// T3 — exact turn/step event sequence; turn_id monotonic across turns
// ---------------------------------------------------------------------------

#[tokio::test]
async fn event_sequence() {
    let provider = Arc::new(CapturingScriptedProvider::new(
        vec![
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
            vec![
                StreamChunk::TextDelta("ok".into()),
                StreamChunk::Finish {
                    reason: "stop".into(),
                },
            ],
        ],
        vec![None, None, None],
    ));

    let (event_tx, event_rx) = tokio::sync::mpsc::channel(256);
    let mut agent = test_agent(Arc::clone(&provider) as Arc<dyn Provider>, event_tx);
    provider.set_inbox(agent.inbox_sender());

    // Collect until the SECOND TurnCompleted (two turns on the same agent).
    let collector = tokio::spawn(collect_events_until(event_rx, 2));

    let text1 = agent
        .run_turn("hi", Path::new("."), &[])
        .await
        .expect("turn 1");
    let text2 = agent
        .run_turn("again", Path::new("."), &[])
        .await
        .expect("turn 2");
    assert_eq!(text1, "done");
    assert_eq!(text2, "ok");

    let mut all = collector.await.expect("event collector task");
    let first_tc = all
        .iter()
        .position(|e| matches!(e, AgentEvent::TurnCompleted { .. }))
        .expect("turn 1 must emit TurnCompleted");
    let turn2_events: Vec<AgentEvent> = all.split_off(first_tc + 1);
    let turn1_events = all;

    assert_eq!(
        turn_step_seq(&turn1_events),
        vec![
            TurnStepEvent::TurnStarted(1),
            TurnStepEvent::StepStarted(1, 1),
            TurnStepEvent::StepCompleted(1, 1, true),
            TurnStepEvent::StepStarted(1, 2),
            TurnStepEvent::StepCompleted(1, 2, false),
            TurnStepEvent::TurnCompleted(1),
        ],
        "2-step turn: TurnStarted → Step1(+tool) → Step2(text) → TurnCompleted, turn_id 1"
    );
    assert_eq!(
        turn_step_seq(&turn2_events),
        vec![
            TurnStepEvent::TurnStarted(2),
            TurnStepEvent::StepStarted(2, 1),
            TurnStepEvent::StepCompleted(2, 1, false),
            TurnStepEvent::TurnCompleted(2),
        ],
        "second run_turn on the same agent must get turn_id 2"
    );
}

// ---------------------------------------------------------------------------
// T4 — item injected after the final drain survives into the next turn
// ---------------------------------------------------------------------------

#[tokio::test]
async fn leftover_claimed_next_turn() {
    // The hook fires at the start of the FINAL round's chat. The item must
    // land AFTER that turn's final boundary drain, so it is a genuine
    // leftover rather than an extension of the current turn. The turn's
    // remaining work after the hook is a few channel sends (~µs), so a small
    // delayed send guarantees arrival after the turn ended.
    let late_hook: Hook = Box::new(|p| {
        let sender = p.inbox_sender();
        let arrived = Arc::clone(&p.arrived);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(150)).await;
            let _ = sender.try_send(InboxItem::UserPrompt {
                text: "queued-late".into(),
            });
            arrived.store(true, Ordering::SeqCst);
        });
    });

    let provider = Arc::new(CapturingScriptedProvider::new(
        vec![
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
            vec![
                StreamChunk::TextDelta("ok".into()),
                StreamChunk::Finish {
                    reason: "stop".into(),
                },
            ],
        ],
        vec![None, Some(late_hook), None],
    ));

    let (event_tx, _event_rx) = tokio::sync::mpsc::channel(256);
    let mut agent = test_agent(Arc::clone(&provider) as Arc<dyn Provider>, event_tx);
    provider.set_inbox(agent.inbox_sender());

    let result = agent
        .run_turn("start", Path::new("."), &[])
        .await
        .expect("turn 1");
    assert_eq!(result, "done");
    assert_eq!(
        provider.call_count(),
        2,
        "turn 1 must not extend for the late item"
    );

    // The late item must NOT appear in turn 1's chat calls.
    let recorded = provider.recorded();
    assert_eq!(recorded.len(), 2);
    for (i, call) in recorded.iter().enumerate() {
        let users = user_texts(call);
        assert!(
            !users.iter().any(|t| t == "queued-late"),
            "late item must not be claimed in turn 1 call {i}: {users:?}"
        );
    }

    // Wait deterministically for the delayed injection to land (bounded
    // retry loop — no unbounded sleeps).
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while !provider.arrived.load(Ordering::SeqCst) && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        provider.arrived.load(Ordering::SeqCst),
        "delayed inbox item must arrive before turn 2 starts"
    );

    let result2 = agent
        .run_turn("again", Path::new("."), &[])
        .await
        .expect("turn 2");
    assert_eq!(result2, "ok");

    let recorded = provider.recorded();
    assert_eq!(recorded.len(), 3);
    let call3_users = user_texts(&recorded[2]);
    // Turn-1 history ("start") is retained by design; the contract is that
    // the leftover is claimed at the START of the next turn, i.e. appended
    // before that turn's new user message.
    let tail = call3_users.split_at(call3_users.len().saturating_sub(2)).1;
    assert_eq!(
        tail,
        ["queued-late".to_string(), "again".to_string()],
        "leftover must be claimed at the START of the next turn, before its new user message"
    );
}

// ---------------------------------------------------------------------------
// T5 — old event-log lines (missing new fields) still deserialize
// ---------------------------------------------------------------------------

#[test]
fn old_log_serde_replay() {
    // Old event-log lines predate the turn/step fields: `turn_id` on
    // TurnCompleted and `steering` on MessageReceived must default rather
    // than fail to deserialize. The `"type"` tag is part of the existing
    // internally-tagged schema, so it appears in every real log line.
    let tc: AgentEvent = serde_json::from_str(r#"{"type":"TurnCompleted","duration_ms":123}"#)
        .expect("old TurnCompleted line must deserialize");
    match tc {
        AgentEvent::TurnCompleted {
            turn_id,
            duration_ms,
        } => {
            assert_eq!(turn_id, 0, "missing turn_id must default to 0");
            assert_eq!(duration_ms, 123);
        }
        other => panic!("wrong variant: {other:?}"),
    }

    let mr: AgentEvent =
        serde_json::from_str(r#"{"type":"MessageReceived","role":"user","content":"hi"}"#)
            .expect("old MessageReceived line must deserialize");
    match mr {
        AgentEvent::MessageReceived {
            role,
            content,
            steering,
        } => {
            assert_eq!(role, "user");
            assert_eq!(content, "hi");
            assert!(!steering, "missing steering must default to false");
        }
        other => panic!("wrong variant: {other:?}"),
    }

    // Full on-disk envelope form (the session event log wraps each event).
    let env: EventEnvelope = serde_json::from_str(
        r#"{"id":7,"ts":null,"event":{"type":"TurnCompleted","duration_ms":123}}"#,
    )
    .expect("old envelope line must deserialize");
    match env.event {
        AgentEvent::TurnCompleted { turn_id, .. } => {
            assert_eq!(turn_id, 0, "missing turn_id must default to 0");
        }
        other => panic!("wrong variant: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// T7 — MessageRecorded events exactly mirror agent.messages (P2 Phase A)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn message_recorded_matches_agent_messages() {
    let provider = Arc::new(CapturingScriptedProvider::new(
        vec![
            vec![
                StreamChunk::ReasoningDelta("need tools".into()),
                StreamChunk::ToolUse(ToolCall {
                    id: "t1".into(),
                    name: "echo".into(),
                    input: json!({}),
                }),
            ],
            vec![
                StreamChunk::TextDelta("done".into()),
                StreamChunk::Finish {
                    reason: "stop".into(),
                },
            ],
        ],
        vec![None, None],
    ));

    let (event_tx, event_rx) = tokio::sync::mpsc::channel(256);
    let mut agent = test_agent(Arc::clone(&provider) as Arc<dyn Provider>, event_tx);
    provider.set_inbox(agent.inbox_sender());

    let collector = tokio::spawn(collect_events_until(event_rx, 1));

    let result = agent
        .run_turn("do the thing", Path::new("."), &[])
        .await
        .expect("scripted 2-step turn succeeds");
    assert_eq!(result, "done");

    let events = collector.await.expect("event collector task");
    let recorded: Vec<Message> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::MessageRecorded { message } => Some(message.clone()),
            _ => None,
        })
        .collect();

    let expected: Vec<Message> = agent
        .messages
        .iter()
        .filter(|m| m.role != Role::System)
        .cloned()
        .collect();

    assert_eq!(
        recorded, expected,
        "MessageRecorded sequence must equal agent.messages (sans system), in push order"
    );
    // Sanity: the turn actually exercised the interesting sites — initial
    // user, assistant-with-tool_calls, tool result, assistant final.
    assert_eq!(recorded.len(), 4, "unexpected record count: {recorded:?}");
    assert!(recorded.iter().any(|m| m.tool_calls.is_some()));
    assert!(recorded.iter().any(|m| m.role == Role::Tool));
}

// ---------------------------------------------------------------------------
// T14 (loop-top cancel emits Error + StepFailed) lives as a unit test in
// `agent_driver.rs`: `run_turn` resets the cancel flag at entry, so the
// loop-top branch can only be driven deterministically via `TurnDriver::run`.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// T6 — inbox is bounded (16); overflow is rejected
// ---------------------------------------------------------------------------

#[tokio::test]
async fn inbox_full_is_rejected() {
    let (event_tx, _event_rx) = tokio::sync::mpsc::channel(256);
    let provider = Arc::new(CapturingScriptedProvider::new(vec![], vec![]));
    let agent = test_agent(provider, event_tx);
    let inbox = agent.inbox_sender();

    // While idle nothing drains the inbox: exactly 16 items fit (bounded 16).
    for i in 0..16 {
        inbox
            .try_send(InboxItem::UserPrompt {
                text: format!("m{i}"),
            })
            .expect("capacity-16 inbox must accept item {i}");
    }
    let overflow = inbox.try_send(InboxItem::UserPrompt {
        text: "overflow".into(),
    });
    assert!(
        overflow.is_err(),
        "17th item must be rejected — inbox is bounded at 16"
    );
}
