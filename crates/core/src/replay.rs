//! Replay projection for the event-sourced session read path
//! (`docs/plans/p2-event-sourced-session-design.md` §P2 Phase A).
//!
//! Pure fold over [`EventEnvelope`]s that depends only on `nca_common`
//! types. The event-log reader lives in the runtime crate; this module never
//! performs I/O.

use nca_common::event::{AgentEvent, EventEnvelope};
use nca_common::message::Message;

/// Fold surface events into the conversation they produced.
///
/// Turn-bracket rule: a turn contributes its messages only if its
/// `TurnStarted..TurnCompleted` bracket closed with no `StepFailed`. This
/// reproduces `run_turn`'s truncate-to-baseline rollback exactly: a failed
/// turn contributes zero messages, including its earlier completed steps.
/// A bracket left unclosed (crash mid-turn) is dropped, and a
/// `MessageRecorded` outside any bracket is dropped defensively. Logs that
/// predate `MessageRecorded` fold to an empty projection.
///
/// State-checkpoint semantics: a `HistoryReplaced` event sets the committed
/// base to exactly its payload, verbatim. It is emitted between turn
/// brackets; if a bracket is nevertheless open, it is left untouched (its
/// records still belong to the bracket — no panic, no double-apply), and any
/// later failed bracket still drops only its own messages.
pub fn replay_surface_events(envelopes: &[EventEnvelope]) -> Vec<Message> {
    let mut out = Vec::new();
    // Open bracket: buffered messages + failure flag. `None` = no open turn.
    let mut bracket: Option<(Vec<Message>, bool)> = None;

    for envelope in envelopes {
        match &envelope.event {
            AgentEvent::TurnStarted { .. } => {
                bracket = Some((Vec::new(), false));
            }
            AgentEvent::MessageRecorded { message } => {
                if let Some((buf, _)) = bracket.as_mut() {
                    buf.push(message.clone());
                }
            }
            AgentEvent::HistoryReplaced { messages } => {
                // State checkpoint: the committed base becomes the payload
                // verbatim. An open bracket, if any, is left untouched.
                out = messages.clone();
            }
            AgentEvent::StepFailed { .. } => {
                if let Some((_, failed)) = bracket.as_mut() {
                    *failed = true;
                }
            }
            AgentEvent::TurnCompleted { .. } => {
                if let Some((buf, failed)) = bracket.take()
                    && !failed
                {
                    out.extend(buf);
                }
            }
            _ => {}
        }
    }
    // An unclosed bracket (crash mid-turn) is dropped: `bracket` is simply
    // never flushed.
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(id: u64, event: AgentEvent) -> EventEnvelope {
        EventEnvelope::new(id, event)
    }

    fn envelopes(events: Vec<AgentEvent>) -> Vec<EventEnvelope> {
        events
            .into_iter()
            .enumerate()
            .map(|(i, e)| env(i as u64, e))
            .collect()
    }

    // T1 — multi-step turn folds to the exact pushed messages.
    #[test]
    fn t1_multi_step_turn_replays_exactly() {
        let user = Message::user("list the files");
        let assistant_tools = Message::assistant_with_tool_calls(
            "",
            vec![nca_common::message::MessageToolCall {
                id: "call_1".into(),
                name: "list_directory".into(),
                arguments: serde_json::json!({"path": "."}),
            }],
        )
        .with_reasoning("need to look".into());
        let tool_1 = Message::tool("call_1", "result body");
        let tool_2 = Message::tool("call_2", "other result");
        let final_msg = Message::assistant("here is the listing").with_reasoning("done".into());

        let envs = envelopes(vec![
            AgentEvent::TurnStarted { turn_id: 1 },
            AgentEvent::MessageRecorded {
                message: user.clone(),
            },
            AgentEvent::StepStarted {
                turn_id: 1,
                step_index: 1,
            },
            AgentEvent::MessageRecorded {
                message: assistant_tools.clone(),
            },
            AgentEvent::MessageRecorded {
                message: tool_1.clone(),
            },
            AgentEvent::MessageRecorded {
                message: tool_2.clone(),
            },
            AgentEvent::StepCompleted {
                turn_id: 1,
                step_index: 1,
                duration_ms: 5,
                had_tool_calls: true,
            },
            AgentEvent::MessageRecorded {
                message: final_msg.clone(),
            },
            AgentEvent::StepCompleted {
                turn_id: 1,
                step_index: 2,
                duration_ms: 7,
                had_tool_calls: false,
            },
            AgentEvent::TurnCompleted {
                turn_id: 1,
                duration_ms: 12,
            },
        ]);

        assert_eq!(
            replay_surface_events(&envs),
            vec![user, assistant_tools, tool_1, tool_2, final_msg]
        );
    }

    // T2 — a failed turn drops ALL its messages, including earlier
    // completed steps.
    #[test]
    fn t2_failed_turn_drops_all_its_messages() {
        let envs = envelopes(vec![
            AgentEvent::TurnStarted { turn_id: 1 },
            AgentEvent::MessageRecorded {
                message: Message::user("go"),
            },
            AgentEvent::StepCompleted {
                turn_id: 1,
                step_index: 1,
                duration_ms: 1,
                had_tool_calls: true,
            },
            AgentEvent::MessageRecorded {
                message: Message::assistant("partial"),
            },
            AgentEvent::StepFailed {
                turn_id: 1,
                step_index: 2,
                duration_ms: 2,
                error: "boom".into(),
            },
            AgentEvent::TurnCompleted {
                turn_id: 1,
                duration_ms: 3,
            },
        ]);
        assert!(replay_surface_events(&envs).is_empty());
    }

    // T3 — input ending mid-bracket (crash mid-turn) drops the bracket.
    #[test]
    fn t3_unclosed_bracket_is_dropped() {
        let envs = envelopes(vec![
            AgentEvent::TurnStarted { turn_id: 1 },
            AgentEvent::MessageRecorded {
                message: Message::user("go"),
            },
            AgentEvent::MessageRecorded {
                message: Message::assistant("partial"),
            },
        ]);
        assert!(replay_surface_events(&envs).is_empty());
    }

    // T4 — MessageRecorded outside any bracket is dropped.
    #[test]
    fn t4_orphan_record_is_dropped() {
        let envs = envelopes(vec![
            AgentEvent::MessageRecorded {
                message: Message::user("stray"),
            },
            AgentEvent::TurnStarted { turn_id: 1 },
            AgentEvent::MessageRecorded {
                message: Message::user("kept"),
            },
            AgentEvent::TurnCompleted {
                turn_id: 1,
                duration_ms: 0,
            },
        ]);
        let replayed = replay_surface_events(&envs);
        assert_eq!(replayed.len(), 1);
        assert_eq!(
            replayed[0].content,
            nca_common::message::MessageContent::text("kept")
        );
    }

    // T5 — old-style logs (no MessageRecorded) fold to empty without panic.
    #[test]
    fn t5_old_style_log_folds_to_empty() {
        let envs = envelopes(vec![
            AgentEvent::TurnStarted { turn_id: 1 },
            AgentEvent::MessageReceived {
                role: "user".into(),
                content: "hi".into(),
                steering: false,
            },
            AgentEvent::StepCompleted {
                turn_id: 1,
                step_index: 1,
                duration_ms: 0,
                had_tool_calls: false,
            },
            AgentEvent::TurnCompleted {
                turn_id: 1,
                duration_ms: 0,
            },
        ]);
        assert!(replay_surface_events(&envs).is_empty());
    }

    // T12 — HistoryReplaced sets the committed base; later successful
    // turns extend it.
    #[test]
    fn t12_history_replaced_sets_base() {
        let envs = envelopes(vec![
            AgentEvent::TurnStarted { turn_id: 1 },
            AgentEvent::MessageRecorded {
                message: Message::user("a"),
            },
            AgentEvent::TurnCompleted {
                turn_id: 1,
                duration_ms: 0,
            },
            AgentEvent::HistoryReplaced {
                messages: vec![Message::user("b"), Message::assistant("b2")],
            },
            AgentEvent::TurnStarted { turn_id: 2 },
            AgentEvent::MessageRecorded {
                message: Message::user("c"),
            },
            AgentEvent::TurnCompleted {
                turn_id: 2,
                duration_ms: 0,
            },
        ]);
        assert_eq!(
            replay_surface_events(&envs),
            vec![
                Message::user("b"),
                Message::assistant("b2"),
                Message::user("c")
            ]
        );
    }

    // T13 — a checkpoint stands even when a later turn fails; the failed
    // turn drops only its own messages.
    #[test]
    fn t13_history_replaced_checkpoint_stands_over_failed_turn() {
        let envs = envelopes(vec![
            AgentEvent::HistoryReplaced {
                messages: vec![Message::user("b"), Message::assistant("b2")],
            },
            AgentEvent::TurnStarted { turn_id: 2 },
            AgentEvent::MessageRecorded {
                message: Message::user("c"),
            },
            AgentEvent::StepFailed {
                turn_id: 2,
                step_index: 1,
                duration_ms: 1,
                error: "boom".into(),
            },
            AgentEvent::TurnCompleted {
                turn_id: 2,
                duration_ms: 1,
            },
        ]);
        assert_eq!(
            replay_surface_events(&envs),
            vec![Message::user("b"), Message::assistant("b2")]
        );
    }

    // HistoryReplaced before any turn: base = payload even with no prior
    // bracket.
    #[test]
    fn history_replaced_before_any_turn() {
        let envs = envelopes(vec![AgentEvent::HistoryReplaced {
            messages: vec![Message::user("solo")],
        }]);
        assert_eq!(replay_surface_events(&envs), vec![Message::user("solo")]);
    }

    // HistoryReplaced payload is used verbatim — the fold performs no
    // system filtering (that happens at the emit site).
    #[test]
    fn history_replaced_payload_used_verbatim() {
        let envs = envelopes(vec![AgentEvent::HistoryReplaced {
            messages: vec![Message::system("stale prompt"), Message::user("kept")],
        }]);
        assert_eq!(
            replay_surface_events(&envs),
            vec![Message::system("stale prompt"), Message::user("kept")]
        );
    }
}
