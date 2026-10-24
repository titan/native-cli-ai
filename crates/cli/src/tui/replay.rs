//! Replay persisted `*.events.jsonl` into TUI state or Elm feedback channel.

use crate::tui::elm::feedback::TuiFeedbackMsg;
use nca_common::event::{AgentEvent, EventEnvelope};
use serde_json::Value;
use std::path::Path;

pub(crate) fn parse_json_values_on_line(line: &str) -> Vec<Value> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let de = serde_json::Deserializer::from_str(trimmed);
    for res in de.into_iter::<Value>() {
        match res {
            Ok(v) => out.push(v),
            Err(_) => break,
        }
    }
    out
}

pub(crate) fn value_to_event(v: &Value) -> Option<AgentEvent> {
    if v.get("event").is_some() {
        serde_json::from_value::<EventEnvelope>(v.clone())
            .ok()
            .map(|e| e.event)
    } else {
        serde_json::from_value::<AgentEvent>(v.clone()).ok()
    }
}

/// Skip streaming deltas when replaying; final `MessageReceived` for assistant already has full text.
pub(crate) fn should_skip_on_replay(ev: &AgentEvent) -> bool {
    matches!(
        ev,
        AgentEvent::TokensStreamed { .. } | AgentEvent::ReasoningStreamed { .. }
    )
}

/// Parse a raw `*.events.jsonl` string into replay-ready events.
///
/// Streaming deltas are skipped (same semantics as [`should_skip_on_replay`]);
/// the final `MessageReceived`/`ToolCallCompleted` events already carry the
/// full payload. Shared by the resume path (`replay_events_to_feedback`) and
/// the subagent view's child-transcript reader.
pub(crate) fn parse_events_from_str(raw: &str) -> Vec<AgentEvent> {
    let mut out = Vec::new();
    for line in raw.lines() {
        for v in parse_json_values_on_line(line) {
            if let Some(ev) = value_to_event(&v)
                && !should_skip_on_replay(&ev)
            {
                out.push(ev);
            }
        }
    }
    out
}

/// Replay persisted events from disk into the Elm feedback channel.
/// Events are buffered in the channel and consumed by NcaModel when it starts.
/// Returns the number of events applied.
pub async fn replay_events_to_feedback(
    log_path: &Path,
    tx: &tokio::sync::mpsc::UnboundedSender<TuiFeedbackMsg>,
) -> u64 {
    let Ok(raw) = tokio::fs::read_to_string(log_path).await else {
        return 0;
    };

    let mut applied = 0u64;
    for ev in parse_events_from_str(&raw) {
        let _ = tx.send(TuiFeedbackMsg::Agent(ev));
        applied += 1;
    }

    // Cleanup: reset transient state that may have been left mid-interaction.
    let _ = tx.send(TuiFeedbackMsg::SetStreamingAssistant(None));
    let _ = tx.send(TuiFeedbackMsg::SetActiveApproval(None));
    let _ = tx.send(TuiFeedbackMsg::SetActiveQuestion(None));

    tracing::debug!(path = %log_path.display(), events_applied = applied, "replayed event log into feedback channel");
    applied
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_envelope_and_bare_event() {
        let line = r#"{"id":1,"event":{"type":"MessageReceived","role":"user","content":"hi"}}"#;
        let vals = parse_json_values_on_line(line);
        assert_eq!(vals.len(), 1);
        let ev = value_to_event(&vals[0]).expect("event");
        assert!(matches!(
            ev,
            AgentEvent::MessageReceived {
                steering: false,
                ..
            }
        ));

        let line2 = r#"{"type":"MessageReceived","role":"assistant","content":"yo"}"#;
        let vals2 = parse_json_values_on_line(line2);
        let ev2 = value_to_event(&vals2[0]).expect("event");
        assert!(matches!(
            ev2,
            AgentEvent::MessageReceived {
                steering: false,
                ..
            }
        ));
    }

    #[test]
    fn parses_two_concatenated_json_objects() {
        let line = concat!(
            r#"{"type":"SessionEnded","reason":{"Completed":null}}"#,
            r#"{"type":"SessionStarted","session_id":"s","workspace":".","model":"m"}"#
        );
        let vals = parse_json_values_on_line(line);
        assert_eq!(vals.len(), 2);
    }

    #[test]
    fn parse_events_from_str_skips_deltas_and_keeps_payloads() {
        let user = r#"{"type":"MessageReceived","role":"user","content":"hi"}"#;
        let delta = r#"{"type":"TokensStreamed","delta":"partial"}"#;
        let assistant = r#"{"type":"MessageReceived","role":"assistant","content":"full answer"}"#;
        let raw = format!("{user}\n{delta}\n{assistant}\n");
        let events = parse_events_from_str(&raw);
        assert_eq!(events.len(), 2, "the streamed delta must be skipped");
        assert!(
            matches!(&events[0], AgentEvent::MessageReceived { role, content, .. }
            if role == "user" && content == "hi")
        );
        assert!(
            matches!(&events[1], AgentEvent::MessageReceived { role, content, .. }
            if role == "assistant" && content == "full answer")
        );
    }

    #[test]
    fn skips_reasoning_streamed_on_replay() {
        assert!(should_skip_on_replay(&AgentEvent::ReasoningStreamed {
            delta: "x".into()
        }));
    }

    #[test]
    fn skips_tokens_streamed_on_replay_flag() {
        assert!(should_skip_on_replay(&AgentEvent::TokensStreamed {
            delta: "x".into()
        }));
        assert!(!should_skip_on_replay(&AgentEvent::MessageReceived {
            role: "user".into(),
            content: "a".into(),
            steering: false,
        }));
    }
}
