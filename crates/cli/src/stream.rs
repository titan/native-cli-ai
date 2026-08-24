//! Event streaming with beautiful TUI rendering
//!
//! This module provides streaming event rendering with Claude Code-inspired styling.

use crate::format::format_duration;
use crate::ipc_pending::{ApprovalPendingMap, QuestionPendingMap};
use colored::Colorize;
use nca_common::event::{AgentEvent, EventEnvelope, InteractiveQuestionPayload, QuestionSelection};
use nca_runtime::event_log::EventLogWriter;
use nca_runtime::ipc::IpcHandle;
use nca_runtime::supervisor;
use std::io::{self, IsTerminal, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tokio::sync::oneshot;

type EventEnvelopeFn = Arc<dyn Fn(&EventEnvelope) + Send + Sync>;

/// Blocking stdin prompt for `--no-tui` human stream when `ask_question` runs.
fn prompt_question_stdio(q: &InteractiveQuestionPayload) -> io::Result<QuestionSelection> {
    let mut err = io::stderr();
    writeln!(err)?;
    writeln!(err, "[question] {}", q.prompt)?;
    writeln!(err, "  [0] suggested: {}", q.suggested_answer)?;
    for (i, o) in q.options.iter().enumerate() {
        writeln!(err, "  [{}] ({}) {}", i + 1, o.id, o.label)?;
    }
    if q.allow_custom {
        writeln!(err, "  [c] custom text")?;
    }
    write!(
        err,
        "Choice (0, 1–{}{}): ",
        q.options.len(),
        if q.allow_custom { ", c" } else { "" }
    )?;
    err.flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    let t = line.trim();
    if t == "0" || t.eq_ignore_ascii_case("s") {
        return Ok(QuestionSelection::Suggested);
    }
    if q.allow_custom && (t.eq_ignore_ascii_case("c") || t.eq_ignore_ascii_case("custom")) {
        write!(err, "Custom answer> ")?;
        err.flush()?;
        let mut custom = String::new();
        io::stdin().read_line(&mut custom)?;
        let custom = custom.trim().to_string();
        if custom.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "empty custom answer",
            ));
        }
        return Ok(QuestionSelection::Custom { text: custom });
    }
    if let Ok(n) = t.parse::<usize>()
        && n >= 1
        && n <= q.options.len()
    {
        return Ok(QuestionSelection::Option {
            option_id: q.options[n - 1].id.clone(),
        });
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "invalid choice; use 0, 1–n, or c",
    ))
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum StreamMode {
    Off,
    Human,
    Ndjson,
}

/// Real-time streaming stats
struct StreamStats {
    input_tokens: Arc<AtomicU64>,
    output_tokens: Arc<AtomicU64>,
    estimated_cost: Arc<AtomicU64>,
    start_time: Instant,
    reasoning_started_at: std::sync::Mutex<Option<Instant>>,
}

impl Clone for StreamStats {
    fn clone(&self) -> Self {
        Self {
            input_tokens: self.input_tokens.clone(),
            output_tokens: self.output_tokens.clone(),
            estimated_cost: self.estimated_cost.clone(),
            start_time: self.start_time,
            // Mutex is not Clone; each clone starts with its own None.
            reasoning_started_at: std::sync::Mutex::new(None),
        }
    }
}

impl StreamStats {
    fn new() -> Self {
        Self {
            input_tokens: Arc::new(AtomicU64::new(0)),
            output_tokens: Arc::new(AtomicU64::new(0)),
            estimated_cost: Arc::new(AtomicU64::new(0)),
            start_time: Instant::now(),
            reasoning_started_at: std::sync::Mutex::new(None),
        }
    }

    fn update_cost(&self, input: u64, output: u64, cost_cents: u64) {
        self.input_tokens.store(input, Ordering::Relaxed);
        self.output_tokens.store(output, Ordering::Relaxed);
        self.estimated_cost.store(cost_cents, Ordering::Relaxed);
    }

    fn record_output_token(&self) {
        self.output_tokens.fetch_add(1, Ordering::Relaxed);
    }

    fn input_tokens(&self) -> u64 {
        self.input_tokens.load(Ordering::Relaxed)
    }

    fn output_tokens(&self) -> u64 {
        self.output_tokens.load(Ordering::Relaxed)
    }

    fn estimated_cost_usd(&self) -> f64 {
        self.estimated_cost.load(Ordering::Relaxed) as f64 / 100.0
    }

    #[allow(dead_code)]
    #[allow(dead_code)]
    fn elapsed_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }
}

impl Default for StreamStats {
    fn default() -> Self {
        Self::new()
    }
}

struct IpcRebroadcast {
    event_tx: tokio::sync::broadcast::Sender<String>,
}

/// Spawns the stream task: event fanout (disk + IPC + rendering) and command consumer.
#[allow(clippy::too_many_arguments)]
pub fn spawn_stream_task(
    rx: tokio::sync::mpsc::Receiver<AgentEvent>,
    mode: StreamMode,
    log_path: std::path::PathBuf,
    ipc_handle: Option<IpcHandle>,
    approval_pending: Option<ApprovalPendingMap>,
    question_pending: Option<QuestionPendingMap>,
    cancel_tx: Option<oneshot::Sender<()>>,
    commit_tx: Option<tokio::sync::watch::Sender<u64>>,
) -> tokio::task::JoinHandle<()> {
    let qp = question_pending.clone();
    let (event_tx_ipc, command_rx) = match ipc_handle {
        Some(h) => {
            let (etx, crx) = h.into_parts();
            (Some(etx), Some(crx))
        }
        None => (None, None),
    };

    if let Some(crx) = command_rx {
        supervisor::spawn_command_consumer(crx, approval_pending, question_pending, cancel_tx);
    }

    spawn_event_fanout_task(rx, mode, log_path, event_tx_ipc, qp, commit_tx)
}

pub fn spawn_event_fanout_task(
    rx: tokio::sync::mpsc::Receiver<AgentEvent>,
    mode: StreamMode,
    log_path: std::path::PathBuf,
    event_tx_ipc: Option<tokio::sync::broadcast::Sender<String>>,
    question_pending: Option<QuestionPendingMap>,
    commit_tx: Option<tokio::sync::watch::Sender<u64>>,
) -> tokio::task::JoinHandle<()> {
    let stats = StreamStats::new();

    let on_event: Option<EventEnvelopeFn> = match mode {
        StreamMode::Off => None,
        StreamMode::Ndjson => Some(Arc::new(|envelope: &EventEnvelope| {
            if let Ok(line) = serde_json::to_string(envelope) {
                println!("{line}");
            }
        })),
        StreamMode::Human => {
            let stats = stats.clone();
            Some(Arc::new(move |envelope: &EventEnvelope| {
                render_event(&envelope.event, &stats);
            }))
        }
    };

    let ipc_handle_rebuilt = event_tx_ipc.map(|tx| IpcRebroadcast { event_tx: tx });

    tokio::spawn(async move {
        use nca_common::event::EventEnvelope;

        let mut writer = EventLogWriter::open(&log_path).await;
        let mut rx = rx;
        let qp = question_pending;
        while let Some(event) = rx.recv().await {
            let id = writer.next_id();
            let envelope = EventEnvelope::new(id, event.clone());

            if let Some(ref ipc) = ipc_handle_rebuilt {
                let line = serde_json::to_string(&envelope).unwrap_or_default();
                let _ = ipc.event_tx.send(line);
            }

            if let Err(e) = writer.append(&envelope).await {
                tracing::error!("failed to append event {} to log: {}", id, e);
            }

            // Durability barrier (P2 Phase B): commit at TurnCompleted and
            // signal the supervisor's run_turn barrier. Liveness over false
            // durability — fires even if the append above failed.
            if let AgentEvent::TurnCompleted { turn_id, .. } = &envelope.event {
                if let Err(e) = writer.commit().await {
                    tracing::error!("event-log commit at turn {turn_id} failed: {e}");
                }
                if let Some(tx) = &commit_tx {
                    let _ = tx.send(*turn_id);
                }
            }

            if let Some(ref cb) = on_event {
                cb(&envelope);
            }

            if let AgentEvent::QuestionRequested { ref question } = event
                && matches!(mode, StreamMode::Human)
                && std::io::stdin().is_terminal()
                && let Some(ref pending) = qp
            {
                let q = question.clone();
                let qid = q.question_id.clone();
                let pending = pending.clone();
                if let Ok(Ok(sel)) =
                    tokio::task::spawn_blocking(move || prompt_question_stdio(&q)).await
                    && let Ok(mut m) = pending.lock()
                    && let Some(tx) = m.remove(&qid)
                {
                    let _ = tx.send(sel);
                }
            }
        }
    })
}

// Claude Code-inspired color theme
mod theme {
    use colored::Color;

    pub const CLEAR_LINE: &str = "\x1B[2K";

    pub const USER_BG: Color = Color::TrueColor {
        r: 0,
        g: 145,
        b: 191,
    };
    pub const ASSISTANT_BG: Color = Color::TrueColor {
        r: 137,
        g: 87,
        b: 220,
    };
    pub const TOOL_BG: Color = Color::TrueColor {
        r: 58,
        g: 170,
        b: 214,
    };

    pub const SUCCESS: Color = Color::TrueColor {
        r: 63,
        g: 185,
        b: 80,
    };
    pub const ERROR: Color = Color::TrueColor {
        r: 248,
        g: 81,
        b: 73,
    };
    pub const WARNING: Color = Color::TrueColor {
        r: 210,
        g: 153,
        b: 34,
    };

    pub const TEXT: Color = Color::TrueColor {
        r: 220,
        g: 220,
        b: 230,
    };
    pub const TEXT_DIM: Color = Color::TrueColor {
        r: 150,
        g: 150,
        b: 160,
    };
}

/// Render a single event with Claude Code-like styling
fn render_event(event: &AgentEvent, stats: &StreamStats) {
    // Flush pending reasoning duration when a non-reasoning event arrives.
    if !matches!(event, AgentEvent::ReasoningStreamed { .. })
        && let Some(start) = stats.reasoning_started_at.lock().unwrap().take()
    {
        let elapsed_ms = start.elapsed().as_millis() as u64;
        println!(
            "  {}",
            format!("thought · {}", format_duration(elapsed_ms)).color(theme::TEXT_DIM)
        );
    }

    match event {
        AgentEvent::SessionStarted {
            session_id: _,
            model,
            workspace: _,
        } => {
            print!("{}", theme::CLEAR_LINE);
            println!();
            println!(
                "{} {}",
                "Connected to".color(theme::SUCCESS),
                model.color(theme::TEXT)
            );
            println!();
        }
        AgentEvent::TokensStreamed { delta } => {
            print!("{delta}");
            stats.record_output_token();
        }
        AgentEvent::ReasoningStreamed { delta } => {
            // Record start time on first reasoning delta for UI-level timing.
            let mut guard = stats.reasoning_started_at.lock().unwrap();
            if guard.is_none() {
                *guard = Some(Instant::now());
            }
            // Dim reasoning output in human mode (not counted as output tokens).
            print!("{}", delta.color(theme::TEXT_DIM));
        }
        AgentEvent::ToolCallStarted {
            tool,
            input: _,
            call_id: _,
        } => {
            // Ensure any inline reasoning text ends on a clean line before
            // the tool header, otherwise CLEAR_LINE would erase it.
            println!();
            print!("{}", theme::CLEAR_LINE);
            println!();
            println!(
                "  {} {}",
                "⚡".color(theme::TOOL_BG).bold(),
                tool.to_uppercase().color(theme::TOOL_BG)
            );
        }
        AgentEvent::ToolCallCompleted {
            call_id: _,
            output,
            duration_ms,
        } => {
            print!("{}", theme::CLEAR_LINE);
            println!();
            let dur = format_duration(*duration_ms);
            if output.success {
                println!(
                    "  {} {} {}",
                    "✓".color(theme::SUCCESS),
                    "Tool completed".color(theme::TEXT_DIM),
                    format!("· {dur}").color(theme::TEXT_DIM)
                );
            } else {
                let label = output
                    .error
                    .as_deref()
                    .filter(|e| !e.is_empty())
                    .unwrap_or_else(|| {
                        // run_validation and similar tools leave error as
                        // None and put diagnostics in output.
                        if output.output.is_empty() {
                            "Tool failed"
                        } else {
                            // Show last non-empty line as a compact summary.
                            output
                                .output
                                .lines()
                                .rev()
                                .find(|l| !l.trim().is_empty())
                                .unwrap_or("Tool failed")
                        }
                    });
                println!(
                    "  {} {} {}",
                    "✗".color(theme::ERROR),
                    label.color(theme::ERROR),
                    format!("· {dur}").color(theme::TEXT_DIM)
                );
            }
        }
        AgentEvent::ApprovalRequested {
            call_id: _,
            tool,
            description,
        } => {
            print!("{}", theme::CLEAR_LINE);
            println!();
            println!(
                "  {} {}: {}",
                "?".color(theme::WARNING).bold(),
                tool.color(theme::WARNING),
                description.color(theme::TEXT_DIM)
            );
        }
        AgentEvent::ApprovalResolved {
            call_id: _,
            approved,
            allow_pattern: _,
        } => {
            print!("{}", theme::CLEAR_LINE);
            if *approved {
                println!(
                    "  {} {}",
                    "✓".color(theme::SUCCESS),
                    "Approved".color(theme::SUCCESS)
                );
            } else {
                println!(
                    "  {} {}",
                    "✗".color(theme::ERROR),
                    "Denied".color(theme::ERROR)
                );
            }
        }
        AgentEvent::QuestionRequested { question } => {
            print!("{}", theme::CLEAR_LINE);
            println!();
            println!(
                "  {} {}",
                "?".color(theme::WARNING).bold(),
                question.prompt.color(theme::TEXT)
            );
            println!(
                "    {} {}",
                "[0]".color(theme::SUCCESS),
                format!("suggested: {}", question.suggested_answer).color(theme::TEXT_DIM)
            );
            for (i, o) in question.options.iter().enumerate() {
                println!(
                    "    {} {} — {}",
                    format!("[{}]", i + 1).color(theme::TOOL_BG),
                    o.id.color(theme::TEXT_DIM),
                    o.label.color(theme::TEXT)
                );
            }
            if question.allow_custom {
                println!("    {}", "[c] custom text".color(theme::TEXT_DIM));
            }
            println!();
        }
        AgentEvent::QuestionResolved {
            question_id,
            selection,
        } => {
            print!("{}", theme::CLEAR_LINE);
            println!(
                "  {} question {} answered: {:?}",
                "✓".color(theme::SUCCESS),
                question_id.color(theme::TEXT_DIM),
                selection
            );
        }
        AgentEvent::CostUpdated {
            input_tokens,
            output_tokens,
            cache_read_tokens: _,
            estimated_cost_usd,
        } => {
            stats.update_cost(
                *input_tokens,
                *output_tokens,
                (*estimated_cost_usd * 100.0) as u64,
            );
        }
        AgentEvent::ContextStatsUpdated {
            estimated_tokens,
            context_window,
            usage_percent,
        } => {
            let pct_color = if *usage_percent >= 90 {
                theme::ERROR
            } else if *usage_percent >= 70 {
                theme::WARNING
            } else {
                theme::TEXT_DIM
            };
            eprintln!(
                "  {} ctx: {}% ({} / {} tokens)",
                "⊗".color(pct_color),
                usage_percent,
                estimated_tokens,
                context_window,
            );
        }
        AgentEvent::SessionEnded { reason } => {
            print!("{}", theme::CLEAR_LINE);
            println!();
            println!();
            println!("  Session ended ({reason:?})");
            println!(
                "  {}/{} tokens, ${:.4}",
                stats.input_tokens(),
                stats.output_tokens(),
                stats.estimated_cost_usd()
            );
        }
        AgentEvent::Error { message } => {
            print!("{}", theme::CLEAR_LINE);
            println!();
            println!(
                "  {} {}",
                "✗".color(theme::ERROR),
                message.color(theme::ERROR)
            );
        }
        AgentEvent::ChildSessionSpawned {
            child_session_id,
            task,
            ..
        } => {
            print!("{}", theme::CLEAR_LINE);
            println!();
            let short = if child_session_id.len() > 8 {
                &child_session_id[..8]
            } else {
                child_session_id.as_str()
            };
            println!(
                "  {} sub-agent {}… — {}",
                "⚡".color(theme::TOOL_BG).bold(),
                short,
                task.chars().take(100).collect::<String>()
            );
        }
        AgentEvent::ChildSessionActivity {
            child_session_id,
            phase,
            detail,
        } => {
            print!("{}", theme::CLEAR_LINE);
            let short = if child_session_id.len() > 8 {
                &child_session_id[..8]
            } else {
                child_session_id.as_str()
            };
            println!(
                "  {} {}… · {} · {}",
                "↳".color(theme::TOOL_BG),
                short,
                phase.color(theme::TEXT_DIM),
                detail.color(theme::TEXT_DIM)
            );
        }
        AgentEvent::ChildSessionCompleted {
            child_session_id,
            status,
            ..
        } => {
            print!("{}", theme::CLEAR_LINE);
            println!();
            let short = if child_session_id.len() > 8 {
                &child_session_id[..8]
            } else {
                child_session_id.as_str()
            };
            println!(
                "  {} sub-agent {}… {}",
                "✓".color(theme::SUCCESS),
                short,
                status.color(theme::TEXT_DIM)
            );
        }
        AgentEvent::MessageReceived { role, content, .. } => {
            println!();
            let header = match role.as_str() {
                "user" => format!(" {} ", "YOU".to_uppercase())
                    .on_color(theme::USER_BG)
                    .white()
                    .bold(),
                "assistant" => format!(" {} ", "nca")
                    .on_color(theme::ASSISTANT_BG)
                    .white()
                    .bold(),
                _ => format!(" {} ", role.to_uppercase())
                    .on_color(theme::WARNING)
                    .white()
                    .bold(),
            };
            println!("{header}");
            println!();
            for line in content.lines() {
                if line.trim().is_empty() {
                    println!();
                } else if line.starts_with("```") {
                    println!("{}", line.color(theme::TEXT_DIM));
                } else {
                    println!("{}", line.color(theme::TEXT));
                }
            }
        }
        AgentEvent::ContextCompactionStart {
            tokens_before,
            reason,
        } => {
            print!("{}", theme::CLEAR_LINE);
            println!(
                "  {}",
                format!("… compacting context ({reason}) · ~{tokens_before} tokens")
                    .color(theme::TEXT_DIM)
            );
        }
        AgentEvent::ContextCompactionEnd {
            tokens_after,
            kv_prefix_broken,
        } => {
            print!("{}", theme::CLEAR_LINE);
            let kv = if *kv_prefix_broken {
                " · cache prefix broken"
            } else {
                ""
            };
            println!(
                "  {} {}",
                "✓".color(theme::SUCCESS),
                format!("context compacted → ~{tokens_after} tokens{kv}").color(theme::TEXT_DIM)
            );
        }
        AgentEvent::ContextCompaction {
            phase: _,
            message,
            tokens_before,
            tokens_after,
            ..
        } => {
            print!("{}", theme::CLEAR_LINE);
            let stats = match (tokens_before, tokens_after) {
                (Some(before), Some(after)) => format!(" · ~{before} → ~{after} tokens"),
                _ => String::new(),
            };
            println!(
                "  {} {}",
                "✓".color(theme::SUCCESS),
                format!("context compacted · {message}{stats}").color(theme::TEXT_DIM)
            );
        }
        AgentEvent::StepCompleted {
            step_index,
            duration_ms,
            had_tool_calls,
            ..
        } => {
            let dur = format_duration(*duration_ms);
            let tools = if *had_tool_calls {
                " · tool-calls"
            } else {
                ""
            };
            println!(
                "  {}",
                format!("step {step_index} · {dur}{tools}").color(theme::TEXT_DIM)
            );
        }
        AgentEvent::StepFailed { error, .. } => {
            let msg: String = error.chars().take(120).collect();
            println!(
                "  {}",
                format!("step failed · {msg}").color(theme::TEXT_DIM)
            );
        }
        AgentEvent::TurnCompleted {
            turn_id,
            duration_ms,
            ..
        } => {
            let dur = format_duration(*duration_ms);
            println!(
                "  {}",
                format!("turn completed · {dur} · turn {turn_id}").color(theme::TEXT_DIM)
            );
        }
        _ => {}
    }
}

/// Public wrapper for rendering a single event (used by main.rs)
pub fn render_human_event(event: &AgentEvent) {
    let stats = StreamStats::new();
    render_event(event, &stats);
}

#[cfg(test)]
mod tests {
    use super::*;
    use nca_common::event::{InteractiveQuestionPayload, QuestionOption};

    #[test]
    fn render_question_requested_does_not_panic() {
        let ev = AgentEvent::QuestionRequested {
            question: InteractiveQuestionPayload {
                question_id: "q-1".into(),
                call_id: "c".into(),
                prompt: "Test?".into(),
                options: vec![QuestionOption {
                    id: "x".into(),
                    label: "X".into(),
                }],
                allow_custom: true,
                suggested_answer: "X".into(),
            },
        };
        render_human_event(&ev);
    }

    #[test]
    fn render_compaction_events_do_not_panic() {
        render_human_event(&AgentEvent::ContextCompactionStart {
            tokens_before: 12_345,
            reason: "auto_summarize".into(),
        });
        render_human_event(&AgentEvent::ContextCompactionEnd {
            tokens_after: 4_000,
            kv_prefix_broken: true,
        });
        render_human_event(&AgentEvent::ContextCompactionEnd {
            tokens_after: 4_000,
            kv_prefix_broken: false,
        });
        render_human_event(&AgentEvent::ContextCompaction {
            phase: "completed".into(),
            message: "summarized".into(),
            tokens_before: Some(10_000),
            tokens_after: Some(3_000),
            retained_groups: None,
            dropped_groups: None,
        });
        render_human_event(&AgentEvent::ContextCompaction {
            phase: "completed".into(),
            message: "pruned".into(),
            tokens_before: None,
            tokens_after: None,
            retained_groups: None,
            dropped_groups: None,
        });
    }

    // T21: the CLI fanout shares the runtime EventLogWriter semantics — ids
    // seed from the existing log (pre-written envelope id 5 → next 6), and
    // TurnCompleted triggers a commit + watch signal with the turn id.
    #[tokio::test(flavor = "multi_thread")]
    async fn fanout_task_seeds_ids_and_signals_commit() {
        use nca_common::message::Message;
        use nca_runtime::session_store::read_event_log;

        let dir = tempfile::tempdir().expect("tempdir");
        let log = dir.path().join("log.events.jsonl");
        std::fs::write(
            &log,
            format!(
                "{}\n",
                serde_json::to_string(&EventEnvelope::new(
                    5,
                    AgentEvent::TurnStarted { turn_id: 5 },
                ))
                .expect("seed envelope")
            ),
        )
        .expect("seed log");

        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let (commit_tx, commit_rx) = tokio::sync::watch::channel(0u64);
        let task = spawn_event_fanout_task(
            rx,
            StreamMode::Off,
            log.clone(),
            None,
            None,
            Some(commit_tx),
        );
        tx.send(AgentEvent::TurnStarted { turn_id: 6 })
            .await
            .expect("send");
        tx.send(AgentEvent::MessageRecorded {
            message: Message::user("hi"),
        })
        .await
        .expect("send");
        tx.send(AgentEvent::TurnCompleted {
            turn_id: 6,
            duration_ms: 1,
        })
        .await
        .expect("send");
        drop(tx);
        task.await.expect("fanout task");

        let envelopes = read_event_log(&log);
        let ids: Vec<u64> = envelopes.iter().map(|e| e.id).collect();
        assert_eq!(ids, vec![5, 6, 7, 8], "ids continue from the seeded log");
        assert!(matches!(
            envelopes[3].event,
            AgentEvent::TurnCompleted { .. }
        ));
        assert!(*commit_rx.borrow() >= 6, "commit watch carries the turn id");
    }
}
