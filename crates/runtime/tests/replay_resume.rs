//! P2 Phase A — event-log replay fallback + resume hardening (RUNTIME tests).
//!
//! Spec (authoritative): `docs/plans/p2-event-sourced-session-design.md`
//! §"Resume algorithm" + §"Test matrix" (T7–T11).
//!
//! Harness mirrors `crates/runtime/tests/inbox.rs`: deterministic offline
//! config (dummy key, bypass permissions, no context API / auto-summarize),
//! real `Supervisor::resume` over a tempdir workspace. No network, no fixed
//! sleeps. The pure decision core (`select_resume_messages`) and the tolerant
//! reader (`read_event_log`) are unit-tested inline in `supervisor.rs` /
//! `session_store.rs`.

use std::path::Path;

use nca_common::config::{NcaConfig, PermissionMode};
use nca_common::event::{AgentEvent, EventEnvelope};
use nca_common::message::{Message, MessageToolCall, Role};
use nca_common::session::{SessionMeta, SessionState, SessionStatus};
use nca_runtime::supervisor::Supervisor;

fn offline_config() -> NcaConfig {
    let mut config = NcaConfig::default();
    config.provider.deepseek.api_key = Some("test-key".into());
    config.permissions.mode = PermissionMode::BypassPermissions;
    config.memory.context.auto_detect_context_window = false;
    config.memory.context.query_provider_models_api = false;
    config.memory.context.enable_auto_summarize = false;
    config
}

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

fn sessions_dir(ws: &Path) -> std::path::PathBuf {
    ws.join(".nca").join("sessions")
}

/// One complete successful turn: user → assistant(tool_calls) → tool result
/// → assistant final. Serialized as real envelopes, in push order.
fn completed_turn_events(turn_id: u64, user_text: &str, final_text: &str) -> Vec<AgentEvent> {
    vec![
        AgentEvent::TurnStarted { turn_id },
        AgentEvent::MessageRecorded {
            message: Message::user(user_text),
        },
        AgentEvent::MessageRecorded {
            message: Message::assistant_with_tool_calls(
                "",
                vec![MessageToolCall {
                    id: format!("call_{turn_id}"),
                    name: "list_directory".into(),
                    arguments: serde_json::json!({"path": "."}),
                }],
            ),
        },
        AgentEvent::MessageRecorded {
            message: Message::tool(format!("call_{turn_id}"), "file_a\nfile_b"),
        },
        AgentEvent::MessageRecorded {
            message: Message::assistant(final_text),
        },
        AgentEvent::TurnCompleted {
            turn_id,
            duration_ms: 10,
        },
    ]
}

fn write_event_log(ws: &Path, session_id: &str, events: Vec<AgentEvent>) {
    std::fs::create_dir_all(sessions_dir(ws)).expect("create sessions dir");
    let mut id = 0u64;
    let mut lines = String::new();
    for event in events {
        id += 1;
        lines.push_str(
            &serde_json::to_string(&EventEnvelope::new(id, event)).expect("serialize envelope"),
        );
        lines.push('\n');
    }
    std::fs::write(
        sessions_dir(ws).join(format!("{session_id}.events.jsonl")),
        lines,
    )
    .expect("write events.jsonl");
}

/// Corrupt the json snapshot (truncate mid-JSON) while leaving the event log
/// intact — the T7 "json unreadable" setup.
fn write_corrupt_json(ws: &Path, session_id: &str) {
    std::fs::create_dir_all(sessions_dir(ws)).expect("create sessions dir");
    std::fs::write(
        sessions_dir(ws).join(format!("{session_id}.json")),
        r#"{"meta":{"id":"broken"#,
    )
    .expect("write corrupt json");
}

/// A well-formed json snapshot carrying a stale system prompt plus the given
/// non-system messages.
fn write_good_json(ws: &Path, session_id: &str, messages: Vec<Message>) {
    std::fs::create_dir_all(sessions_dir(ws)).expect("create sessions dir");
    let now = chrono::Utc::now();
    let state = SessionState {
        meta: SessionMeta {
            id: session_id.to_string(),
            created_at: now,
            updated_at: now,
            workspace: ws.to_path_buf(),
            model: "deepseek-chat".into(),
            status: SessionStatus::Completed,
            pid: None,
            socket_path: None,
            worktree_path: None,
            branch: None,
            base_branch: None,
            parent_session_id: None,
            child_session_ids: Vec::new(),
            inherited_summary: None,
            spawn_reason: None,
            session_summary: None,
            session_title: None,
            orchestration: None,
        },
        messages,
        total_input_tokens: 0,
        total_output_tokens: 0,
        estimated_cost_usd: 0.0,
    };
    std::fs::write(
        sessions_dir(ws).join(format!("{session_id}.json")),
        serde_json::to_string_pretty(&state).expect("serialize session"),
    )
    .expect("write session json");
}

fn read_json_messages(ws: &Path, session_id: &str) -> Vec<Message> {
    let raw = std::fs::read_to_string(sessions_dir(ws).join(format!("{session_id}.json")))
        .expect("session json must exist after resume");
    let state: SessionState = serde_json::from_str(&raw).expect("session json must be valid");
    state.messages
}

/// Every assistant tool_call message must be followed by its tool result
/// message — replay must never leave an orphaned tool_call.
fn assert_no_orphaned_tool_calls(messages: &[Message]) {
    for (i, m) in messages.iter().enumerate() {
        if m.role == Role::Assistant && m.tool_calls.as_ref().is_some_and(|calls| !calls.is_empty())
        {
            for call in m.tool_calls.iter().flatten() {
                let answered = messages[i + 1..].iter().any(|later| {
                    later.role == Role::Tool
                        && later.tool_call_id.as_deref() == Some(call.id.as_str())
                });
                assert!(
                    answered,
                    "assistant tool_call {} at index {i} has no tool result",
                    call.id
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// T7 — corrupt json + good log → replay fallback, fresh system first,
//      tool pairs intact.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn t7_corrupt_json_resumes_from_event_log_replay() {
    let ws = tempfile::tempdir().expect("tempdir");
    let sid = "p2-t7-corrupt-json";

    write_event_log(
        ws.path(),
        sid,
        [
            completed_turn_events(1, "first question", "first answer"),
            completed_turn_events(2, "second question", "second answer"),
        ]
        .concat(),
    );
    write_corrupt_json(ws.path(), sid);

    let sup = Supervisor::resume(offline_config(), ws.path(), true, false, sid, None)
        .await
        .expect("resume must succeed via event-log fallback despite corrupt json");

    let messages = sup.agent().messages.clone();
    assert!(
        matches!(messages.first().map(|m| &m.role), Some(Role::System)),
        "fresh system prompt must be first, got {:?}",
        messages.first().map(|m| &m.role)
    );
    assert_eq!(
        messages[1..].len(),
        8,
        "two 4-message turns must be replayed"
    );
    assert_eq!(messages[1], Message::user("first question"));
    assert_eq!(messages[4], Message::assistant("first answer"));
    assert_eq!(messages[5], Message::user("second question"));
    assert_eq!(messages[8], Message::assistant("second answer"));
    assert_no_orphaned_tool_calls(&messages);

    // The fallback path re-saved a valid json immediately: re-reading it
    // yields the same agent-visible history.
    let on_disk = read_json_messages(ws.path(), sid);
    assert_eq!(on_disk.len(), messages.len());
}

// ---------------------------------------------------------------------------
// T8 — crash-cutoff log (mid-turn, no TurnCompleted) + corrupt json →
//      resume lands on the last completed turn.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn t8_crash_cutoff_log_lands_on_last_completed_turn() {
    let ws = tempfile::tempdir().expect("tempdir");
    let sid = "p2-t8-crash-cutoff";

    let mut events = completed_turn_events(1, "finished question", "finished answer");
    // Turn 2 crashed mid-flight: TurnStarted + pushes, no TurnCompleted.
    events.push(AgentEvent::TurnStarted { turn_id: 2 });
    events.push(AgentEvent::MessageRecorded {
        message: Message::user("crashed question"),
    });
    events.push(AgentEvent::MessageRecorded {
        message: Message::assistant_with_tool_calls(
            "",
            vec![MessageToolCall {
                id: "call_dangling".into(),
                name: "read_file".into(),
                arguments: serde_json::json!({"path": "x"}),
            }],
        ),
    });
    write_event_log(ws.path(), sid, events);
    write_corrupt_json(ws.path(), sid);

    let sup = Supervisor::resume(offline_config(), ws.path(), true, false, sid, None)
        .await
        .expect("resume from crash-cutoff log");

    let messages = sup.agent().messages.clone();
    assert_eq!(
        messages[1..].len(),
        4,
        "only the completed turn must survive; the cut-off turn is dropped"
    );
    assert_eq!(messages[1], Message::user("finished question"));
    assert_eq!(
        messages.last().map(|m| m.content.event_preview()),
        Some("finished answer".into())
    );
    assert_no_orphaned_tool_calls(&messages);
}

// ---------------------------------------------------------------------------
// T9 — resume re-saves immediately: json non-system messages intact after
//      resume without any turn (create() did not wipe them).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn t9_resume_resaves_without_wiping_json() {
    let ws = tempfile::tempdir().expect("tempdir");
    let sid = "p2-t9-resave";

    write_good_json(
        ws.path(),
        sid,
        vec![
            Message::system("stale system prompt"),
            Message::user("persisted question"),
            Message::assistant("persisted answer"),
        ],
    );
    write_event_log(
        ws.path(),
        sid,
        completed_turn_events(1, "persisted question", "persisted answer"),
    );

    let sup = Supervisor::resume(offline_config(), ws.path(), true, false, sid, None)
        .await
        .expect("resume must succeed");

    // No turn is run. The json on disk must already contain the restored
    // non-system history (the create()-wipes-json window is closed). Phase B:
    // the fresh-format log wins, so the re-saved json mirrors the replay
    // projection (4 messages for this turn fixture), not the stale json body.
    let on_disk = read_json_messages(ws.path(), sid);
    let non_system: Vec<&Message> = on_disk.iter().filter(|m| m.role != Role::System).collect();
    assert_eq!(
        non_system.len(),
        4,
        "json must retain the replayed turn's non-system messages"
    );
    assert_eq!(non_system[0], &Message::user("persisted question"));
    assert_eq!(
        non_system.last().map(|m| m.content.event_preview()),
        Some("persisted answer".into())
    );
    assert_eq!(on_disk.len(), sup.agent().messages.len());
}

// ---------------------------------------------------------------------------
// T11 — stale system prompts in json are replaced by the fresh one on the
//       snapshot path too (old-format log ⇒ json wins, normalization
//       applies to BOTH paths). Phase B note: a fresh-format log would take
//       the replay path — that flip is T18 below.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn t11_stale_system_prompts_replaced_on_snapshot_path() {
    let ws = tempfile::tempdir().expect("tempdir");
    let sid = "p2-t11-stale-system";

    write_good_json(
        ws.path(),
        sid,
        vec![
            Message::system("stale prompt v1"),
            Message::user("q"),
            Message::system("stale prompt v2"),
            Message::assistant("a"),
        ],
    );
    // OLD-format log (no MessageRecorded — pre-Phase-A events only): folds
    // to an empty projection, so the json snapshot path is taken.
    write_event_log(
        ws.path(),
        sid,
        vec![
            AgentEvent::TurnStarted { turn_id: 1 },
            AgentEvent::MessageReceived {
                role: "user".into(),
                content: "q".into(),
                steering: false,
            },
            AgentEvent::TurnCompleted {
                turn_id: 1,
                duration_ms: 0,
            },
        ],
    );

    let sup = Supervisor::resume(offline_config(), ws.path(), true, false, sid, None)
        .await
        .expect("resume must succeed");

    let messages = sup.agent().messages.clone();
    let system_count = messages.iter().filter(|m| m.role == Role::System).count();
    assert_eq!(
        system_count, 1,
        "exactly one (fresh) system message; stale json prompts dropped"
    );
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[1], Message::user("q"));
    assert_eq!(messages[2], Message::assistant("a"));
}

// ---------------------------------------------------------------------------
// T18 — fresh-format log + healthy but DIVERGENT json → replay wins
//       (Phase B flip: json is a cache; `docs/plans/p2-phase-b-design.md` §5).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn t18_fresh_log_replay_beats_divergent_json() {
    let ws = tempfile::tempdir().expect("tempdir");
    let sid = "p2-t18-replay-authoritative";

    // Json says turn 2 ended with a truncated final answer (stale cache);
    // the log records the full turn 2.
    write_good_json(
        ws.path(),
        sid,
        vec![
            Message::user("first question"),
            Message::assistant("first answer"),
            Message::user("second question"),
            Message::assistant("cut-off answ"), // diverges from the log
        ],
    );
    write_event_log(
        ws.path(),
        sid,
        [
            completed_turn_events(1, "first question", "first answer"),
            completed_turn_events(2, "second question", "second answer"),
        ]
        .concat(),
    );

    let sup = Supervisor::resume(offline_config(), ws.path(), true, false, sid, None)
        .await
        .expect("resume must succeed");

    let messages = sup.agent().messages.clone();
    // Replay projection: 2 turns × 4 messages, NOT the json's 4-message pair
    // list with the truncated answer.
    assert_eq!(messages[1..].len(), 8, "replay projection wins");
    assert_eq!(messages[1], Message::user("first question"));
    assert_eq!(messages[8], Message::assistant("second answer"));
    assert_no_orphaned_tool_calls(&messages);
}

// ---------------------------------------------------------------------------
// T20 — log carries a compaction checkpoint (HistoryReplaced) + a stale
//       pre-compaction json → resume reflects the COMPACTED history.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn t20_history_replaced_checkpoint_wins_over_stale_json() {
    let ws = tempfile::tempdir().expect("tempdir");
    let sid = "p2-t20-compaction-checkpoint";

    // Pre-compaction json (would resurrect rolled-up history).
    write_good_json(
        ws.path(),
        sid,
        vec![
            Message::user("old question 1"),
            Message::assistant("old answer 1"),
            Message::user("old question 2"),
            Message::assistant("old answer 2"),
        ],
    );
    // Log: turn 1 recorded, then compaction replaced the history with a
    // summary pair, then turn 2 recorded on top of it.
    write_event_log(
        ws.path(),
        sid,
        vec![
            AgentEvent::TurnStarted { turn_id: 1 },
            AgentEvent::MessageRecorded {
                message: Message::user("old question 1"),
            },
            AgentEvent::MessageRecorded {
                message: Message::assistant("old answer 1"),
            },
            AgentEvent::TurnCompleted {
                turn_id: 1,
                duration_ms: 0,
            },
            AgentEvent::HistoryReplaced {
                messages: vec![
                    Message::user("[summary of earlier turns]"),
                    Message::assistant("[acknowledged]"),
                ],
            },
        ]
        .into_iter()
        .chain(completed_turn_events(
            2,
            "post-compact question",
            "post-compact answer",
        ))
        .collect::<Vec<_>>(),
    );

    let sup = Supervisor::resume(offline_config(), ws.path(), true, false, sid, None)
        .await
        .expect("resume must succeed");

    let messages = sup.agent().messages.clone();
    // Checkpoint (2) + turn 2 (4) — the pre-compaction json history is gone.
    assert_eq!(messages[1..].len(), 6, "compacted history must win");
    assert_eq!(messages[1], Message::user("[summary of earlier turns]"));
    assert_eq!(messages[2], Message::assistant("[acknowledged]"));
    assert_eq!(messages[3], Message::user("post-compact question"));
    assert_eq!(messages[6], Message::assistant("post-compact answer"));
}
