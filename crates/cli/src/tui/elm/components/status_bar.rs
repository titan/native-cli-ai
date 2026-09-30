//! Status bar component — renders model, branch, tokens, cost, busy state, etc.

use std::time::{Duration, Instant};

use nca_common::event::BusyState;
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::tui::busy_indicator;

/// Theme colors used in the status bar (from shared module, re-exported via searchable_list).
use super::searchable_list::theme;

/// Data needed to render the status bar.
///
/// This is a snapshot of the relevant fields from `TuiSessionState`.
/// In Phase 3, NcaModel will populate this from TuiFeedbackMsg updates.
#[derive(Debug, Clone)]
pub(crate) struct StatusBarData {
    pub version: String,
    pub workspace_dir: String,
    pub model: String,
    pub agent_profile: String,
    pub current_branch: String,
    pub permission_mode: String,
    pub session_id: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: f64,
    pub context_usage_percent: u8,
    pub current_busy_state: BusyState,
    pub busy_state_since: Instant,
    pub active_approval: bool,
    pub active_question: bool,
    /// Cumulative time spent in work states (LLM thinking/streaming + tool
    /// execution) across the session. Approval/user wait time is excluded.
    pub work_elapsed: Duration,
    /// Steering messages queued in the agent inbox during the current turn.
    pub queued: u32,
}

impl Default for StatusBarData {
    fn default() -> Self {
        Self {
            version: String::new(),
            workspace_dir: String::new(),
            model: String::from("unknown"),
            agent_profile: String::from("@orchestrator"),
            current_branch: String::new(),
            permission_mode: String::from("ask"),
            session_id: String::new(),
            input_tokens: 0,
            output_tokens: 0,
            cost_usd: 0.0,
            context_usage_percent: 0,
            current_busy_state: BusyState::Idle,
            busy_state_since: Instant::now(),
            active_approval: false,
            active_question: false,
            work_elapsed: Duration::ZERO,
            queued: 0,
        }
    }
}

/// Status bar component.
///
/// Renders the bottom bar showing model, branch, tokens, cost, busy state, etc.
/// Ported from the inline rendering in `run_blocking()`.
pub(crate) struct StatusBar {
    data: StatusBarData,
    /// Whether sidebar is visible (affects whether tokens/cost show on bar).
    pub(crate) sidebar_visible: bool,
    /// Start of the current contiguous work period, if inside one.
    work_since: Option<Instant>,
}

impl StatusBar {
    pub(crate) fn new() -> Self {
        Self {
            data: StatusBarData::default(),
            sidebar_visible: true,
            work_since: None,
        }
    }

    /// Update the status bar data snapshot.
    pub(crate) fn update_data(&mut self, data: StatusBarData) {
        self.data = data;
    }

    // ── Individual setters for NcaModel feedback routing ──

    pub(crate) fn update_session(&mut self, session_id: &str, model: &str) {
        self.data.session_id = session_id.to_string();
        self.data.model = model.to_string();
    }

    pub(crate) fn update_model(&mut self, model: &str) {
        self.data.model = model.to_string();
    }

    pub(crate) fn update_agent_profile(&mut self, label: &str) {
        self.data.agent_profile = label.to_string();
    }

    pub(crate) fn update_permission_mode(&mut self, mode: &str) {
        self.data.permission_mode = mode.to_string();
    }

    pub(crate) fn update_branch(&mut self, branch: &str) {
        self.data.current_branch = branch.to_string();
    }

    pub(crate) fn set_busy(&mut self, state: BusyState) {
        let now = std::time::Instant::now();
        let was_work = is_work_state(self.data.current_busy_state);
        let is_work = is_work_state(state);
        // Work → non-work closes the contiguous work period: bank it into the
        // session total. Work → work stays open (one period spans e.g.
        // Thinking → Streaming → ToolRunning → Thinking …).
        if was_work
            && !is_work
            && let Some(since) = self.work_since.take()
        {
            self.data.work_elapsed += now.saturating_duration_since(since);
        }
        if is_work && self.work_since.is_none() {
            self.work_since = Some(now);
        }
        self.data.current_busy_state = state;
        if state == BusyState::Idle {
            self.data.busy_state_since = now;
        }
    }

    /// Cumulative session work time: banked periods plus the live one.
    fn total_work(&self) -> Duration {
        self.data.work_elapsed + self.work_since.map_or(Duration::ZERO, |t| t.elapsed())
    }

    /// Reset the work timer (new session / session switch): drop banked time
    /// and any open work period without banking it.
    pub(crate) fn reset_work(&mut self) {
        self.data.work_elapsed = Duration::ZERO;
        self.work_since = None;
    }

    /// Whether the UI is currently busy (non-Idle) — drives the Animation Frame
    /// cadence in `NcaModel::tick`.
    pub(crate) fn is_busy(&self) -> bool {
        self.data.current_busy_state != BusyState::Idle
    }

    pub(crate) fn update_cost(&mut self, input: u64, output: u64, cost: f64) {
        self.data.input_tokens = input;
        self.data.output_tokens = output;
        self.data.cost_usd = cost;
    }

    pub(crate) fn update_context(&mut self, window: usize, usage: usize) {
        let _ = window;
        self.data.context_usage_percent = usage.clamp(0, 100) as u8;
    }

    pub(crate) fn set_active_approval(&mut self, active: bool) {
        self.data.active_approval = active;
    }

    pub(crate) fn update_version(&mut self, version: &str) {
        self.data.version = version.to_string();
    }

    pub(crate) fn update_workspace_dir(&mut self, dir: &str) {
        self.data.workspace_dir = dir.to_string();
    }

    pub(crate) fn set_active_question(&mut self, active: bool) {
        self.data.active_question = active;
    }

    pub(crate) fn set_queued(&mut self, queued: u32) {
        self.data.queued = queued;
    }

    /// Render the status bar into the given area.
    pub(crate) fn render(&mut self, frame: &mut Frame, area: Rect) {
        let d = &self.data;

        // Busy indicator
        let indicator_text =
            busy_indicator::render_indicator(d.current_busy_state, d.busy_state_since);
        let indicator_color = busy_indicator::color_for_state(d.current_busy_state);
        let busy = Span::styled(indicator_text, Style::default().fg(indicator_color));

        // Approval hint
        let approval_hint = if d.active_approval {
            Span::styled(" !approve ", Style::default().fg(theme::ERROR))
        } else {
            Span::raw("")
        };

        // Question hint
        let q_hint = if d.active_question {
            Span::styled(" ?answer ", Style::default().fg(theme::WARN))
        } else {
            Span::raw("")
        };

        // Session work timer (LLM + tools) — replaces the old perm chip. The
        // red BYPASS warning stays: it is a safety hint, not a mode readout.
        let timer_span = Span::styled(
            format!(" ⏱ {} ", format_duration(self.total_work())),
            Style::default().fg(theme::MUTED),
        );
        let bypass_span = toolbar_permission_is_bypass(&d.permission_mode).then(|| {
            Span::styled(
                " BYPASS — tools run without approval ",
                Style::default()
                    .fg(Color::Black)
                    .bg(theme::ERROR)
                    .add_modifier(Modifier::BOLD),
            )
        });

        // Cancel hint (when busy, Esc cancels)
        let cancel_hint_text = " Esc cancel ";
        let cancel_visible = matches!(
            d.current_busy_state,
            BusyState::Thinking | BusyState::Streaming | BusyState::ToolRunning
        );
        let cancel_hint = cancel_visible.then(|| {
            Span::styled(
                cancel_hint_text,
                Style::default()
                    .fg(Color::Black)
                    .bg(theme::WARN)
                    .add_modifier(Modifier::BOLD),
            )
        });

        // Layout: main status content (left) + cancel hint (right)
        let status_rect = if cancel_hint.is_some() && area.width > cancel_hint_text.len() as u16 {
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Min(0),
                    Constraint::Length(cancel_hint_text.len() as u16),
                ])
                .split(area)[0]
        } else {
            area
        };

        // Branch chip
        let branch_text = if d.current_branch.is_empty() {
            String::new()
        } else {
            format!("⎇ {}", d.current_branch)
        };
        let branch_span_style = Style::default()
            .fg(theme::TOOL)
            .add_modifier(Modifier::UNDERLINED);

        // Workspace dir (truncated to fit, Unicode-aware)
        let dir_display = if d.workspace_dir.is_empty() {
            String::new()
        } else {
            let max_dir_w = (area.width as usize).saturating_sub(50);
            let dir = &d.workspace_dir;
            if dir.chars().count() > max_dir_w {
                let mut t: String = dir.chars().take(max_dir_w.saturating_sub(1)).collect();
                t.push('…');
                t
            } else {
                dir.clone()
            }
        };

        // Build status spans
        let mut status_spans = vec![busy, approval_hint, q_hint];
        if d.queued > 0 {
            status_spans.push(Span::styled(
                format!(" queued:{}", d.queued),
                Style::default().fg(theme::MUTED),
            ));
        }
        status_spans.push(Span::raw(" │ "));
        status_spans.push(Span::styled(&d.model, Style::default().fg(theme::USER)));
        status_spans.push(Span::raw(" │ "));
        status_spans.push(Span::styled(
            &d.agent_profile,
            Style::default().fg(theme::ASSISTANT),
        ));
        if !dir_display.is_empty() {
            status_spans.push(Span::raw(" │ "));
            status_spans.push(Span::styled(dir_display, Style::default().fg(theme::MUTED)));
        }
        status_spans.push(Span::raw(" │ "));
        status_spans.push(Span::styled(branch_text, branch_span_style));
        status_spans.push(Span::raw(" │ "));
        status_spans.push(timer_span);
        if let Some(bypass_span) = bypass_span {
            status_spans.push(bypass_span);
        }

        // Tokens/cost/session: show on bar only when sidebar is hidden
        if !self.sidebar_visible {
            status_spans.push(Span::raw(" │ "));
            status_spans.push(Span::styled(
                d.session_id[..8.min(d.session_id.len())].to_string(),
                Style::default().fg(theme::MUTED),
            ));
            status_spans.extend([
                Span::raw(" │ in:"),
                Span::styled(
                    format!("{}", d.input_tokens),
                    Style::default().fg(theme::TEXT),
                ),
                Span::raw(" out:"),
                Span::styled(
                    format!("{}", d.output_tokens),
                    Style::default().fg(theme::TEXT),
                ),
                Span::raw(" │ $"),
                Span::styled(
                    format!("{:.4}", d.cost_usd),
                    Style::default().fg(theme::SUCCESS),
                ),
            ]);
        }

        // Context usage percentage (always shown)
        let ctx_pct_color = if d.context_usage_percent >= 90 {
            theme::ERROR
        } else if d.context_usage_percent >= 70 {
            theme::WARN
        } else {
            theme::MUTED
        };
        status_spans.push(Span::raw(" │ ctx:"));
        status_spans.push(Span::styled(
            format!("{}%", d.context_usage_percent),
            Style::default().fg(ctx_pct_color),
        ));
        if !d.version.is_empty() {
            status_spans.push(Span::raw(" │ "));
            status_spans.push(Span::styled(
                format!("nca v{}", d.version),
                Style::default().fg(theme::MUTED),
            ));
        }

        // Render main bar
        let status = Line::from(status_spans);
        let bar = Paragraph::new(status).style(Style::default().bg(theme::SURFACE));
        frame.render_widget(bar, status_rect);

        // Render cancel hint overlay (right-aligned)
        if let Some(cancel_hint) = cancel_hint {
            let hint_width = cancel_hint_text.len() as u16;
            if area.width > hint_width {
                let hint_rect = Rect::new(
                    area.x + area.width.saturating_sub(hint_width),
                    area.y,
                    hint_width,
                    1,
                );
                let hint_bar = Paragraph::new(Line::from(cancel_hint))
                    .style(Style::default().bg(theme::SURFACE));
                frame.render_widget(hint_bar, hint_rect);
            }
        }
    }
}

/// States that count toward the session work timer: LLM response time plus
/// tool execution. `ApprovalPending` (user wait) and `Error` are excluded.
fn is_work_state(state: BusyState) -> bool {
    matches!(
        state,
        BusyState::Thinking | BusyState::Streaming | BusyState::ToolRunning
    )
}

/// Compact human duration for the status bar: `42s`, `3m07s`, `1h04m`.
fn format_duration(d: Duration) -> String {
    let s = d.as_secs();
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{}h{:02}m", s / 3600, (s % 3600) / 60)
    }
}

fn toolbar_permission_is_bypass(mode: &str) -> bool {
    mode.contains("BypassPermissions")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_duration_tiers() {
        assert_eq!(format_duration(Duration::from_secs(5)), "5s");
        assert_eq!(format_duration(Duration::from_secs(65)), "1m05s");
        assert_eq!(format_duration(Duration::from_secs(3661)), "1h01m");
    }

    // The timer banks contiguous work periods and reopens on the next one,
    // so Thinking → Streaming → ToolRunning counts once and Idle/Approval
    // splits turns into separate periods that all accumulate.
    #[test]
    fn set_busy_accumulates_work_periods() {
        let mut bar = StatusBar::new();
        bar.set_busy(BusyState::Thinking);
        std::thread::sleep(Duration::from_millis(30));
        // Work → work keeps the period open; the hop itself adds nothing.
        bar.set_busy(BusyState::Streaming);
        bar.set_busy(BusyState::ToolRunning);
        std::thread::sleep(Duration::from_millis(30));
        bar.set_busy(BusyState::Idle);
        let first = bar.total_work();
        assert!(first >= Duration::from_millis(60), "got {first:?}");

        // A second turn reopens the period and accumulates on top.
        bar.set_busy(BusyState::Thinking);
        std::thread::sleep(Duration::from_millis(20));
        bar.set_busy(BusyState::Idle);
        assert!(bar.total_work() >= first + Duration::from_millis(20));
    }

    // Reset must zero the timer even with a work period open (a switch
    // away mid-busy discards the open period rather than banking it).
    #[test]
    fn reset_work_zeroes_banked_and_open_period() {
        let mut bar = StatusBar::new();
        bar.set_busy(BusyState::Thinking);
        std::thread::sleep(Duration::from_millis(20));
        bar.set_busy(BusyState::Idle);
        bar.set_busy(BusyState::ToolRunning); // open period
        assert!(bar.total_work() > Duration::ZERO);

        bar.reset_work();
        assert_eq!(bar.total_work(), Duration::ZERO);

        // Still functional after reset: a new period accumulates normally.
        bar.set_busy(BusyState::Thinking);
        std::thread::sleep(Duration::from_millis(15));
        assert!(bar.total_work() >= Duration::from_millis(15));
    }

    // Approval wait is user time, not LLM/tool time: it must pause the timer.
    #[test]
    fn set_busy_excludes_approval_wait() {
        let mut bar = StatusBar::new();
        bar.set_busy(BusyState::Thinking);
        std::thread::sleep(Duration::from_millis(20));
        bar.set_busy(BusyState::ApprovalPending);
        std::thread::sleep(Duration::from_millis(50));
        bar.set_busy(BusyState::ToolRunning);
        std::thread::sleep(Duration::from_millis(20));
        bar.set_busy(BusyState::Idle);
        let total = bar.total_work();
        assert!(total >= Duration::from_millis(40), "got {total:?}");
        assert!(total < Duration::from_secs(5), "approval wait leaked in");
    }
}
