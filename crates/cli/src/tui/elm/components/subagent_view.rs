//! Subagent view modal (P2): family list + per-child read-only transcript.
//!
//! Two focus modes inside one modal:
//! - **List**: every descendant session of the current session (running
//!   first, newest first), one row per child — state icon, lineage indent,
//!   task text, short id, cost. `j/k`/arrows select, Enter opens the
//!   Reader, Tab cycles running children, Esc closes.
//! - **Reader**: the selected child's conversation, parsed from its
//!   `events.jsonl` (replay parser reuse — deltas skipped, final payloads
//!   kept). `j/k`/PgUp/PgDn scroll, Tab cycles running, Esc back to List.
//!
//! Isolation invariants (R8): the reader NEVER touches the parent
//! `TranscriptState` — child content renders from this component's own
//! line model only, and view data comes exclusively from disk + already
//! delivered events. Live tail runs on the model's 40ms tick: only while
//! Reader mode is open AND the viewed child is Running, stat the log and
//! parse the newly appended bytes (bounded reads; the Elm thread is a
//! blocking thread by design).

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use nca_common::event::AgentEvent;
use nca_common::session::ChildSessionState;

use super::searchable_list::{ClearWidget, centered_rect, theme};
use crate::tui::elm::subagent_family::SubagentEntry;
use crate::tui::replay::parse_events_from_str;
use crate::tui::text_utils::truncate;

// ── Constants ────────────────────────────────────────────────────

/// Visible list rows in List mode.
const LIST_MAX_ROWS: usize = 12;
/// Initial-load tail cap: parse at most the last 2MB of a child log so a
/// huge finished log cannot stall a frame on open (design §risks).
const INITIAL_TAIL_BYTES: u64 = 2 * 1024 * 1024;
/// Suffix of the truncated-earlier notice line.
const TRUNCATED_NOTICE: &str = "… log truncated (older events not loaded)";

// ── Actions ──────────────────────────────────────────────────────

/// What a keypress asked the model to do.
#[derive(Debug, PartialEq)]
pub(crate) enum SubagentViewAction {
    None,
    /// Unmount the modal.
    Close,
    /// Rebuild the family index (a child lifecycle event arrived).
    Refresh,
}

// ── Focus modes ──────────────────────────────────────────────────

/// Which sub-mode of the modal has keyboard focus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SubagentMode {
    List,
    Reader,
}

// ── State ────────────────────────────────────────────────────────

pub(crate) struct SubagentViewState {
    mode: SubagentMode,
    /// Family rows (rebuilt on open + child lifecycle events).
    entries: Vec<SubagentEntry>,
    /// Selected row in List mode (index into `entries`).
    selected: usize,
    /// List scroll offset.
    list_scroll: usize,
    /// Reader line scroll, remembered PER CHILD across visits.
    scroll_memory: HashMap<String, usize>,
    /// Currently-open child id in Reader mode.
    reader_session_id: Option<String>,
    /// Reader's rendered lines for the open child.
    reader_lines: Vec<Line<'static>>,
    /// Earlier events were dropped by the tail cap.
    reader_truncated: bool,
    /// Child state snapshot taken when the reader was opened (drives tail).
    reader_child_running: bool,
    /// Per-child read offset for incremental tail (byte offset into the log).
    tail_offsets: HashMap<String, u64>,
    /// Reader scroll offset for the open child (clamped at render;
    /// `usize::MAX` is the bottom-anchor sentinel).
    reader_scroll: usize,
}

impl SubagentViewState {
    pub(crate) fn new() -> Self {
        Self {
            mode: SubagentMode::List,
            entries: Vec::new(),
            selected: 0,
            list_scroll: 0,
            scroll_memory: HashMap::new(),
            reader_session_id: None,
            reader_lines: Vec::new(),
            reader_truncated: false,
            reader_child_running: false,
            tail_offsets: HashMap::new(),
            reader_scroll: 0,
        }
    }

    pub(crate) fn is_open(&self) -> bool {
        true // mounted = open
    }

    /// Read-only access to the merged family rows (tests assert on these).
    pub(crate) fn entries(&self) -> &[SubagentEntry] {
        &self.entries
    }

    /// Open the modal: (re)build the family from disk + live observations.
    /// The caller (`NcaModel`) has already merged its live observations into
    /// `entries` — the view holds only the merged rows.
    pub(crate) fn open(&mut self, entries: Vec<SubagentEntry>) {
        self.entries = entries;
        self.mode = SubagentMode::List;
        self.selected = self.selected.min(self.entries.len().saturating_sub(1));
        self.list_scroll = 0;
        self.close_reader();
    }

    /// Drop reader state (keep per-child scroll/offset memories).
    fn close_reader(&mut self) {
        self.mode = SubagentMode::List;
        self.reader_session_id = None;
        self.reader_lines = Vec::new();
        self.reader_truncated = false;
        self.reader_child_running = false;
    }

    /// Swap in a freshly rebuilt family (child lifecycle event while open).
    /// Keeps selection stable by id; leaves Reader mode alone when the
    /// open child still exists (its live state is refreshed too).
    pub(crate) fn refresh_entries(&mut self, entries: Vec<SubagentEntry>) {
        let prev_id = self
            .entries
            .get(self.selected)
            .map(|e| e.session_id.clone());
        self.entries = entries;
        self.selected = match prev_id {
            Some(id) => self
                .entries
                .iter()
                .position(|e| e.session_id == id)
                .unwrap_or(0),
            None => 0,
        };
        self.selected = self.selected.min(self.entries.len().saturating_sub(1));
        if let Some(id) = self.reader_session_id.as_ref() {
            if !self.entries.iter().any(|e| &e.session_id == id) {
                // The open child vanished (session pruned); fall back to list.
                self.close_reader();
            } else {
                self.sync_reader_running();
            }
        }
    }

    /// Update `reader_child_running` from the current entries.
    fn sync_reader_running(&mut self) {
        if let Some(id) = self.reader_session_id.as_ref() {
            self.reader_child_running = self
                .entries
                .iter()
                .find(|e| &e.session_id == id)
                .map(|e| !e.state.is_terminal())
                .unwrap_or(false);
        }
    }

    /// Whether the 40ms tick should tail the open child's log.
    pub(crate) fn wants_tail(&self) -> bool {
        self.mode == SubagentMode::Reader && self.reader_child_running
    }

    /// The open child's log path, when Reader mode has one.
    fn reader_log_path(&self) -> Option<&PathBuf> {
        let id = self.reader_session_id.as_ref()?;
        self.entries
            .iter()
            .find(|e| &e.session_id == id)
            .map(|e| &e.log_path)
    }

    /// Tick hook (called by the model's 40ms tick when `wants_tail`):
    /// stat the open child's log and parse only the newly appended bytes.
    pub(crate) fn on_tick_tail(&mut self) {
        if !self.wants_tail() {
            return;
        }
        let Some(path) = self.reader_log_path().cloned() else {
            return;
        };
        let Ok(meta) = std::fs::metadata(&path) else {
            return;
        };
        let len = meta.len();
        let id = self.reader_session_id.clone().unwrap_or_default();
        let offset = *self.tail_offsets.get(&id).unwrap_or(&0);
        if len <= offset {
            return; // no growth (or truncate: only forward tails are parsed)
        }
        let mut new_text = match read_from(&path, offset, len) {
            Some(t) => t,
            None => return,
        };
        self.tail_offsets.insert(id.clone(), len);
        // A partial trailing line (writer mid-append): hold it back by
        // advancing the offset only to the last complete newline; the rest
        // is re-read next tick.
        if let Some(last_nl) = new_text.rfind('\n') {
            let consumed = last_nl + 1;
            new_text.truncate(consumed);
            *self.tail_offsets.get_mut(&id).unwrap_or(&mut 0) = offset + consumed as u64;
        } else {
            // No complete line yet: wait for one.
            *self.tail_offsets.get_mut(&id).unwrap_or(&mut 0) = offset;
            return;
        }
        for ev in parse_events_from_str(&new_text) {
            self.reader_lines.extend(event_to_lines(&ev));
        }
    }

    /// Enter Reader mode for the selected row (loads the child log with the
    /// tail cap) — or directly for a given child (Tab cycle from List).
    pub(crate) fn open_reader(&mut self, session_id: Option<&str>) {
        let idx = session_id.and_then(|id| self.entries.iter().position(|e| e.session_id == id));
        let idx = match idx {
            Some(i) => i,
            None => {
                if self.entries.is_empty() {
                    return;
                }
                self.selected.min(self.entries.len() - 1)
            }
        };
        let entry = self.entries[idx].clone();
        self.selected = idx;
        self.mode = SubagentMode::Reader;
        self.reader_session_id = Some(entry.session_id.clone());
        self.reader_child_running = !entry.state.is_terminal();

        let (text, offset, truncated) = load_log_tailed(&entry.log_path);
        self.tail_offsets.insert(entry.session_id.clone(), offset);
        let mut lines = Vec::new();
        if truncated {
            lines.push(Line::from(Span::styled(
                format!(" {TRUNCATED_NOTICE}"),
                Style::default().fg(theme::MUTED),
            )));
        }
        for ev in parse_events_from_str(&text) {
            lines.extend(event_to_lines(&ev));
        }
        if lines.is_empty() && entry.log_path.as_os_str().is_empty() {
            lines.push(Line::from(Span::styled(
                " child not yet persisted — no log to read",
                Style::default().fg(theme::MUTED),
            )));
        }
        self.reader_lines = lines;
        self.reader_truncated = truncated;
        // Restore per-child scroll; bottom-anchored by default (live tail).
        let remembered = self
            .scroll_memory
            .get(&entry.session_id)
            .copied()
            .unwrap_or(usize::MAX);
        self.reader_scroll = remembered;
    }

    /// Save the open child's scroll into memory and return to List mode.
    fn back_to_list(&mut self) {
        if let Some(id) = self.reader_session_id.take() {
            self.scroll_memory.insert(id, self.reader_scroll);
        }
        self.mode = SubagentMode::List;
        self.reader_lines = Vec::new();
        self.reader_child_running = false;
    }

    /// Tab: cycle to the next running child (from List or Reader alike —
    /// in Reader it jumps straight to the next running child's reader).
    fn cycle_running(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        let running: Vec<usize> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| !e.state.is_terminal())
            .map(|(i, _)| i)
            .collect();
        if running.is_empty() {
            return;
        }
        let current = self
            .reader_session_id
            .as_ref()
            .and_then(|id| self.entries.iter().position(|e| &e.session_id == id))
            .unwrap_or(self.selected);
        let pos = running.iter().position(|&i| i == current).unwrap_or(0);
        let next = running[(pos + 1) % running.len()];
        self.selected = next;
        let id = self.entries[next].session_id.clone();
        if self.mode == SubagentMode::Reader {
            self.open_reader(Some(&id));
        }
    }

    // ── Key handling ─────────────────────────────────────────────

    pub(crate) fn handle_key(&mut self, key: KeyEvent, viewport_rows: usize) -> SubagentViewAction {
        match self.mode {
            SubagentMode::List => self.handle_key_list(key),
            SubagentMode::Reader => self.handle_key_reader(key, viewport_rows),
        }
    }

    fn handle_key_list(&mut self, key: KeyEvent) -> SubagentViewAction {
        let n = self.entries.len();
        match (key.code, key.modifiers) {
            (KeyCode::Esc, _) | (KeyCode::Char('q'), KeyModifiers::NONE) => {
                SubagentViewAction::Close
            }
            (KeyCode::Up, _) | (KeyCode::Char('k'), KeyModifiers::NONE) => {
                self.selected = self.selected.saturating_sub(1);
                SubagentViewAction::None
            }
            (KeyCode::Down, _) | (KeyCode::Char('j'), KeyModifiers::NONE) => {
                if n > 0 {
                    self.selected = (self.selected + 1).min(n - 1);
                }
                SubagentViewAction::None
            }
            (KeyCode::Enter, _) => {
                self.open_reader(None);
                SubagentViewAction::None
            }
            (KeyCode::Tab, _) => {
                self.cycle_running();
                SubagentViewAction::None
            }
            _ => SubagentViewAction::None,
        }
    }

    fn handle_key_reader(&mut self, key: KeyEvent, viewport_rows: usize) -> SubagentViewAction {
        let max_scroll = self.reader_lines.len().saturating_sub(1);
        let page = viewport_rows.max(1);
        match (key.code, key.modifiers) {
            (KeyCode::Esc, _) => {
                self.back_to_list();
                SubagentViewAction::None
            }
            (KeyCode::Up, _) | (KeyCode::Char('k'), KeyModifiers::NONE) => {
                self.reader_scroll = self.reader_scroll.saturating_sub(1);
                SubagentViewAction::None
            }
            (KeyCode::Down, _) | (KeyCode::Char('j'), KeyModifiers::NONE) => {
                self.reader_scroll = (self.reader_scroll + 1).min(max_scroll);
                SubagentViewAction::None
            }
            (KeyCode::PageUp, _) => {
                self.reader_scroll = self.reader_scroll.saturating_sub(page);
                SubagentViewAction::None
            }
            (KeyCode::PageDown, _) => {
                self.reader_scroll = (self.reader_scroll + page).min(max_scroll);
                SubagentViewAction::None
            }
            (KeyCode::Home, _) => {
                self.reader_scroll = 0;
                SubagentViewAction::None
            }
            (KeyCode::End, _) => {
                self.reader_scroll = max_scroll;
                SubagentViewAction::None
            }
            (KeyCode::Tab, _) => {
                self.cycle_running();
                SubagentViewAction::None
            }
            _ => SubagentViewAction::None,
        }
    }

    // ── Render ───────────────────────────────────────────────────

    pub(crate) fn render(&mut self, frame: &mut Frame) {
        match self.mode {
            SubagentMode::List => self.render_list(frame),
            SubagentMode::Reader => self.render_reader(frame),
        }
    }

    fn render_list(&mut self, frame: &mut Frame) {
        let area = frame.area();
        let n = self.entries.len();
        let viewport = n.min(LIST_MAX_ROWS);
        // Keep selection visible.
        if self.selected < self.list_scroll {
            self.list_scroll = self.selected;
        } else if viewport > 0 && self.selected >= self.list_scroll + viewport {
            self.list_scroll = self.selected.saturating_sub(viewport - 1);
        }
        self.list_scroll = self.list_scroll.min(n.saturating_sub(viewport));
        let start = self.list_scroll;
        let end = (start + viewport).min(n);

        let popup_area = centered_rect(area, 72, (viewport as u16 + 6).max(10));

        let mut lines: Vec<Line> = Vec::new();
        if n == 0 {
            lines.push(Line::from(Span::styled(
                " no subagent sessions for this session",
                Style::default().fg(theme::MUTED),
            )));
        } else {
            if start > 0 {
                lines.push(muted_line(format!("  ▲ {} more", start)));
            }
            for (vis, idx) in (start..end).enumerate() {
                let e = &self.entries[idx];
                let is_sel = idx == self.selected;
                lines.push(list_row_line(e, is_sel));
                let _ = vis;
            }
            let below = n.saturating_sub(end);
            if below > 0 {
                lines.push(muted_line(format!("  ▼ {below} more")));
            }
        }

        lines.push(Line::default());
        lines.push(muted_line(
            " Enter open · Tab next running · j/k select · Esc close".to_string(),
        ));

        frame.render_widget(ClearWidget, popup_area);
        let title = format!(" subagents ({n}) ");
        let popup = Paragraph::new(Text::from(lines))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(theme::BORDER))
                    .title(Span::styled(title, Style::default().fg(theme::MUTED))),
            )
            .style(Style::default().bg(theme::SURFACE))
            .wrap(Wrap { trim: false });
        frame.render_widget(popup, popup_area);
    }

    fn render_reader(&mut self, frame: &mut Frame) {
        let area = frame.area();
        let popup_area = centered_rect(area, 80, (area.height.saturating_sub(4)).max(10));
        let vis_h = popup_area.height.saturating_sub(2) as usize; // minus borders
        let vis_h = vis_h.max(1);
        let total = self.reader_lines.len();
        let max_scroll = total.saturating_sub(vis_h);
        // Bottom-anchor (usize::MAX sentinel from open_reader / live tail).
        if self.reader_scroll == usize::MAX {
            self.reader_scroll = max_scroll;
        }
        if self.reader_scroll > max_scroll {
            self.reader_scroll = max_scroll;
        }
        let start = self.reader_scroll;
        let end = (start + vis_h).min(total);

        let id = self.reader_session_id.as_deref().unwrap_or("").to_string();
        let short = crate::tui::elm::subagent_family::short_id(&id);
        let live_tag = if self.reader_child_running {
            " · live"
        } else {
            ""
        };

        let mut lines: Vec<Line> = Vec::new();
        for l in &self.reader_lines[start..end] {
            lines.push(l.clone());
        }
        lines.push(Line::default());
        lines.push(muted_line(format!(
            " {}/{}{live_tag} · j/k PgUp/PgDn scroll · Tab next running · Esc back",
            start + 1,
            total.max(start + 1)
        )));

        frame.render_widget(ClearWidget, popup_area);
        let title = format!(" child {short}{live_tag} ");
        let popup = Paragraph::new(Text::from(lines))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(theme::BORDER))
                    .title(Span::styled(title, Style::default().fg(theme::MUTED))),
            )
            .style(Style::default().bg(theme::SURFACE))
            .wrap(Wrap { trim: false });
        frame.render_widget(popup, popup_area);
    }
}

// ── Row/line rendering helpers ───────────────────────────────────

fn muted_line(s: String) -> Line<'static> {
    Line::from(Span::styled(s, Style::default().fg(theme::MUTED)))
}

/// State icon for a child lifecycle state.
fn state_icon(state: ChildSessionState) -> (&'static str, ratatui::style::Color) {
    match state {
        ChildSessionState::Pending => ("○", theme::MUTED),
        ChildSessionState::Running => ("●", theme::SUCCESS),
        ChildSessionState::Completed => ("✓", theme::TOOL),
        ChildSessionState::Cancelled => ("⊘", theme::MUTED),
        ChildSessionState::Failed => ("✗", theme::ERROR),
    }
}

/// One List row: `icon  ↳↳ task…  [short-id]  $0.0123`.
fn list_row_line(e: &SubagentEntry, selected: bool) -> Line<'static> {
    let (icon, icon_color) = state_icon(e.state);
    let indent = "  ".repeat(e.depth);
    let task = truncate(e.task.as_deref().unwrap_or("(no task)"), 42);
    let mut spans = vec![
        Span::styled(
            format!("{icon} {indent}"),
            if selected {
                Style::default().fg(icon_color).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(icon_color)
            },
        ),
        Span::styled(
            task,
            if selected {
                Style::default()
                    .fg(theme::TEXT)
                    .bg(theme::USER)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(theme::TEXT)
            },
        ),
        Span::styled(
            format!("  [{}]", e.short_id),
            Style::default().fg(theme::MUTED),
        ),
    ];
    if let Some(cost) = e.cost_usd.filter(|c| *c > 0.0) {
        spans.push(Span::styled(
            format!("  ${cost:.4}"),
            Style::default().fg(theme::MUTED),
        ));
    }
    Line::from(spans)
}

/// Render one parsed child event into reader lines (opencode mini-inspector
/// shape: role icon + text; tool calls as one-liners).
fn event_to_lines(ev: &AgentEvent) -> Vec<Line<'static>> {
    match ev {
        AgentEvent::MessageReceived { role, content, .. } => {
            let text = truncate(content, 4000);
            let (icon, color) = if role == "user" {
                ("❯", theme::USER)
            } else {
                ("❖", theme::ASSISTANT)
            };
            let mut out = Vec::with_capacity(2);
            out.push(Line::from(Span::styled(
                format!("{icon} {role}"),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            )));
            for l in text.split('\n') {
                out.push(Line::from(Span::styled(
                    format!("  {l}"),
                    Style::default().fg(theme::TEXT),
                )));
            }
            out.push(Line::default());
            out
        }
        AgentEvent::MessageRecorded { message } => {
            // Replay-authoritative projection; render role-tagged text.
            let (role, icon, color) = match message.role {
                nca_common::message::Role::User => ("user", "❯", theme::USER),
                nca_common::message::Role::Assistant => ("assistant", "❖", theme::ASSISTANT),
                nca_common::message::Role::Tool => ("tool", "⚙", theme::TOOL),
                nca_common::message::Role::System => ("system", "·", theme::MUTED),
            };
            let plain = message_content_plain(&message.content);
            let text = truncate(&plain, 4000);
            let mut out = Vec::with_capacity(2);
            out.push(Line::from(Span::styled(
                format!("{icon} {role}"),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            )));
            for l in text.split('\n') {
                out.push(Line::from(Span::styled(
                    format!("  {l}"),
                    Style::default().fg(theme::TEXT),
                )));
            }
            out.push(Line::default());
            out
        }
        AgentEvent::ToolCallStarted { tool, input, .. } => {
            let one = tool_input_one_liner(input);
            vec![Line::from(Span::styled(
                format!("⚙ {tool} {one}"),
                Style::default().fg(theme::TOOL),
            ))]
        }
        AgentEvent::ToolCallCompleted {
            output,
            duration_ms,
            ..
        } => {
            let ok = output.success;
            let head = truncate(
                output
                    .output
                    .lines()
                    .find(|l| !l.trim().is_empty())
                    .unwrap_or(""),
                120,
            );
            let mark = if ok { "✓" } else { "✗" };
            vec![Line::from(Span::styled(
                format!("  {mark} {head} · {duration_ms}ms"),
                if ok {
                    Style::default().fg(theme::MUTED)
                } else {
                    Style::default().fg(theme::ERROR)
                },
            ))]
        }
        AgentEvent::Error { message } => vec![Line::from(Span::styled(
            format!("✗ error: {}", truncate(message, 300)),
            Style::default().fg(theme::ERROR),
        ))],
        AgentEvent::SessionEnded { reason } => vec![Line::from(Span::styled(
            format!("— session ended: {reason:?}"),
            Style::default().fg(theme::MUTED),
        ))],
        AgentEvent::TurnStarted { .. } | AgentEvent::TurnCompleted { .. } => {
            vec![Line::default()]
        }
        // Envelope/lifecycle chatter not part of the mini-inspector shape.
        _ => Vec::new(),
    }
}

/// Compact one-line summary of a tool input for the reader.
fn tool_input_one_liner(input: &serde_json::Value) -> String {
    for key in ["path", "command", "pattern", "query", "task", "url"] {
        if let Some(s) = input.get(key).and_then(|v| v.as_str()) {
            return truncate(s, 80);
        }
    }
    String::new()
}

/// Plain-text projection of a message body (text parts joined; image parts
/// become a placeholder) for the read-only reader.
fn message_content_plain(c: &nca_common::message::MessageContent) -> String {
    use nca_common::message::{ContentPart, MessageContent};
    match c {
        MessageContent::Text(t) => t.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|p| match p {
                ContentPart::Text { text } => text.clone(),
                ContentPart::Image { path, .. } => format!("[image: {path}]"),
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

// ── Log loading helpers ──────────────────────────────────────────

/// Read `len - offset` bytes starting at `offset` (bounded single read).
///
/// Decoded **lossily**: a read whose start lands mid-UTF-8-character (the
/// 2MB tail-cap cut point, or a torn trailing char mid-append) must degrade
/// to U+FFFD, never fail the whole read. With strict `from_utf8`, a cut on
/// a continuation byte yielded an empty string AND parked the tail offset
/// on a permanently-invalid boundary — every later incremental read failed
/// too, so a truncated Chinese-heavy log rendered as nothing, forever.
/// The damaged first bytes always sit in the partial first line, which the
/// callers drain to the first newline (initial load) or hold back past the
/// last newline (tail), so the U+FFFD residue never renders.
fn read_from(path: &PathBuf, offset: u64, len: u64) -> Option<String> {
    let mut f = std::fs::File::open(path).ok()?;
    f.seek(SeekFrom::Start(offset)).ok()?;
    let mut buf = vec![0u8; (len - offset) as usize];
    f.read_exact(&mut buf).ok()?;
    Some(String::from_utf8_lossy(&buf).into_owned())
}

/// Load a child log for the reader with the tail cap: when the log exceeds
/// [`INITIAL_TAIL_BYTES`], read only the last cap bytes (starting after a
/// newline boundary so the first parsed line is complete).
///
/// Returns `(text, next_offset, truncated)` where `next_offset` is where
/// incremental tailing should resume.
fn load_log_tailed(path: &PathBuf) -> (String, u64, bool) {
    let Ok(meta) = std::fs::metadata(path) else {
        return (String::new(), 0, false);
    };
    let len = meta.len();
    if len <= INITIAL_TAIL_BYTES {
        let Ok(text) = std::fs::read_to_string(path) else {
            return (String::new(), 0, false);
        };
        let end = text.len() as u64;
        return (text, end, false);
    }
    let start = len - INITIAL_TAIL_BYTES;
    let mut text = read_from(path, start, len).unwrap_or_default();
    // Drop a possibly-partial first line.
    if let Some(nl) = text.find('\n') {
        text.drain(..=nl);
    }
    let consumed = start + (text.len() as u64);
    (text, consumed, true)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn entry(id: &str, state: ChildSessionState, task: Option<&str>) -> SubagentEntry {
        SubagentEntry {
            session_id: id.to_string(),
            short_id: crate::tui::elm::subagent_family::short_id(id),
            parent_id: "root".to_string(),
            depth: 0,
            state,
            task: task.map(str::to_string),
            log_path: PathBuf::new(),
            cost_usd: Some(0.01),
        }
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn tmpdir() -> tempfile::TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    fn write_log(dir: &std::path::Path, id: &str, body: &str) -> PathBuf {
        let p = dir.join(format!("{id}.events.jsonl"));
        std::fs::write(&p, body).expect("write log");
        p
    }

    const USER_LINE: &str = r#"{"type":"MessageReceived","role":"user","content":"do the thing"}"#;
    const ASSISTANT_LINE: &str =
        r#"{"type":"MessageReceived","role":"assistant","content":"done"}"#;

    // ── list navigation ──────────────────────────────────────────

    #[test]
    fn list_navigation_moves_and_clamps() {
        let mut v = SubagentViewState::new();
        v.open(vec![
            entry("c1", ChildSessionState::Running, Some("a")),
            entry("c2", ChildSessionState::Completed, Some("b")),
        ]);
        assert_eq!(v.selected, 0);
        assert_eq!(
            v.handle_key(key(KeyCode::Down), 10),
            SubagentViewAction::None
        );
        assert_eq!(v.selected, 1);
        // Clamp at the end.
        v.handle_key(key(KeyCode::Down), 10);
        assert_eq!(v.selected, 1);
        v.handle_key(key(KeyCode::Up), 10);
        assert_eq!(v.selected, 0);
        // Clamp at the start.
        v.handle_key(key(KeyCode::Up), 10);
        assert_eq!(v.selected, 0);
    }

    #[test]
    fn esc_in_list_closes_but_esc_in_reader_goes_back() {
        let mut v = SubagentViewState::new();
        v.open(vec![entry("c1", ChildSessionState::Running, None)]);
        assert_eq!(
            v.handle_key(key(KeyCode::Esc), 10),
            SubagentViewAction::Close
        );
        assert_eq!(v.mode, SubagentMode::List, "Close does not mutate mode");

        // Enter → Reader; Esc returns to List, not Close.
        v.handle_key(key(KeyCode::Enter), 10);
        assert_eq!(v.mode, SubagentMode::Reader);
        assert_eq!(
            v.handle_key(key(KeyCode::Esc), 10),
            SubagentViewAction::None
        );
        assert_eq!(v.mode, SubagentMode::List);
        // A second Esc now closes.
        assert_eq!(
            v.handle_key(key(KeyCode::Esc), 10),
            SubagentViewAction::Close
        );
    }

    #[test]
    fn tab_cycles_between_running_children_from_list() {
        let mut v = SubagentViewState::new();
        v.open(vec![
            entry("c1", ChildSessionState::Completed, None),
            entry("c2", ChildSessionState::Running, None),
            entry("c3", ChildSessionState::Running, None),
        ]);
        v.selected = 1;
        v.handle_key(key(KeyCode::Tab), 10);
        assert_eq!(v.selected, 2, "c2 → c3");
        v.handle_key(key(KeyCode::Tab), 10);
        assert_eq!(v.selected, 1, "wraps back to c2");
    }

    #[test]
    fn tab_in_reader_jumps_to_next_running_reader() {
        let dir = tmpdir();
        let mut e1 = entry("c1", ChildSessionState::Running, None);
        e1.log_path = write_log(dir.path(), "c1", &format!("{USER_LINE}\n"));
        let mut e2 = entry("c2", ChildSessionState::Running, None);
        e2.log_path = write_log(dir.path(), "c2", &format!("{ASSISTANT_LINE}\n"));

        let mut v = SubagentViewState::new();
        v.open(vec![e1, e2]);
        v.handle_key(key(KeyCode::Enter), 10); // open c1 reader
        assert_eq!(v.reader_session_id.as_deref(), Some("c1"));
        v.handle_key(key(KeyCode::Tab), 10);
        assert_eq!(
            v.reader_session_id.as_deref(),
            Some("c2"),
            "Tab from Reader opens the next running child's reader"
        );
        assert_eq!(v.mode, SubagentMode::Reader);
    }

    // ── reader: parse → render ───────────────────────────────────

    #[test]
    fn reader_parses_log_into_role_and_tool_lines() {
        let dir = tmpdir();
        let tool_start = r#"{"type":"ToolCallStarted","call_id":"t1","tool":"read_file","input":{"path":"/a/b.rs"}}"#;
        let tool_done = r#"{"type":"ToolCallCompleted","call_id":"t1","duration_ms":5,"output":{"call_id":"t1","success":true,"output":"fn main() {}","timed_out":false}}"#;
        let body = format!("{USER_LINE}\n{tool_start}\n{tool_done}\n{ASSISTANT_LINE}\n");
        let mut e = entry("c1", ChildSessionState::Running, None);
        e.log_path = write_log(dir.path(), "c1", &body);

        let mut v = SubagentViewState::new();
        v.open(vec![e]);
        v.handle_key(key(KeyCode::Enter), 10);

        let rendered: Vec<String> = v
            .reader_lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.to_string())
                    .collect::<String>()
            })
            .collect();
        let joined = rendered.join("\n");
        assert!(joined.contains("❯ user"), "user header: {joined}");
        assert!(joined.contains("do the thing"));
        assert!(
            joined.contains("⚙ read_file /a/b.rs"),
            "tool one-liner: {joined}"
        );
        assert!(joined.contains("✓ fn main()"), "tool result: {joined}");
        assert!(joined.contains("❖ assistant"));
        assert!(joined.contains("done"));
    }

    #[test]
    fn reader_skips_streaming_deltas() {
        let dir = tmpdir();
        let delta = r#"{"type":"TokensStreamed","delta":"partial"}"#;
        let body = format!("{delta}\n{ASSISTANT_LINE}\n");
        let mut e = entry("c1", ChildSessionState::Completed, None);
        e.log_path = write_log(dir.path(), "c1", &body);

        let mut v = SubagentViewState::new();
        v.open(vec![e]);
        v.handle_key(key(KeyCode::Enter), 10);
        let joined = lines_to_string(&v.reader_lines);
        assert!(!joined.contains("partial"), "deltas must not render");
        assert!(joined.contains("done"));
    }

    // ── reader: scroll memory ────────────────────────────────────

    #[test]
    fn reader_remember_scroll_per_child() {
        let dir = tmpdir();
        let mut e1 = entry("c1", ChildSessionState::Completed, None);
        e1.log_path = write_log(dir.path(), "c1", &format!("{USER_LINE}\n"));
        let mut e2 = entry("c2", ChildSessionState::Completed, None);
        e2.log_path = write_log(dir.path(), "c2", &format!("{ASSISTANT_LINE}\n"));

        let mut v = SubagentViewState::new();
        v.open(vec![e1.clone(), e2]);
        v.selected = 0;
        v.handle_key(key(KeyCode::Enter), 10);
        v.reader_scroll = 3;
        v.handle_key(key(KeyCode::Esc), 10); // back to list (saves scroll)

        v.selected = 1;
        v.handle_key(key(KeyCode::Enter), 10); // c2: fresh reader
        assert_eq!(
            v.reader_scroll,
            usize::MAX,
            "c2 has no memory — bottom-anchor sentinel"
        );
        v.handle_key(key(KeyCode::Esc), 10);
        v.reader_scroll = 7;

        v.selected = 0;
        v.handle_key(key(KeyCode::Enter), 10); // c1 again
        assert_eq!(v.reader_scroll, 3, "c1's remembered scroll is restored");
    }

    // ── live tail ────────────────────────────────────────────────

    #[test]
    fn tail_appends_new_events_incrementally() {
        let dir = tmpdir();
        let mut e = entry("c1", ChildSessionState::Running, None);
        e.log_path = write_log(dir.path(), "c1", &format!("{USER_LINE}\n"));

        let mut v = SubagentViewState::new();
        v.open(vec![e.clone()]);
        v.handle_key(key(KeyCode::Enter), 10);
        assert!(v.wants_tail(), "running child + Reader open");
        let before = v.reader_lines.len();

        // Append two events.
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&e.log_path)
            .unwrap();
        writeln!(f, "{ASSISTANT_LINE}").unwrap();
        writeln!(f, r#"{{"type":"Error","message":"boom"}}"#).unwrap();
        drop(f);

        v.on_tick_tail();
        let joined = lines_to_string(&v.reader_lines);
        assert!(joined.contains("done"), "appended assistant event renders");
        assert!(joined.contains("boom"), "appended error event renders");
        assert!(v.reader_lines.len() > before);
        let offset = *v.tail_offsets.get("c1").unwrap();
        assert_eq!(
            offset,
            std::fs::metadata(&e.log_path).unwrap().len(),
            "offset tracks the file end"
        );

        // No growth: idempotent.
        v.on_tick_tail();
        let after = v.reader_lines.len();
        v.on_tick_tail();
        assert_eq!(v.reader_lines.len(), after, "no duplicate appends");
    }

    #[test]
    fn tail_holds_back_partial_trailing_line() {
        let dir = tmpdir();
        let mut e = entry("c1", ChildSessionState::Running, None);
        e.log_path = write_log(dir.path(), "c1", &format!("{USER_LINE}\n"));

        let mut v = SubagentViewState::new();
        v.open(vec![e.clone()]);
        v.handle_key(key(KeyCode::Enter), 10);

        // Append HALF an event line (writer mid-append).
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&e.log_path)
            .unwrap();
        write!(f, r#"{{"type":"Mess"#).unwrap();
        drop(f);

        v.on_tick_tail();
        let joined = lines_to_string(&v.reader_lines);
        assert!(!joined.contains("Mess"), "partial line must not parse");
        // Complete the line: the tail must now include it.
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&e.log_path)
            .unwrap();
        writeln!(f, r#"ageReceived","role":"assistant","content":"late"}}"#).unwrap();
        drop(f);
        v.on_tick_tail();
        let joined = lines_to_string(&v.reader_lines);
        assert!(
            joined.contains("late"),
            "completed line parses on next tick"
        );
    }

    #[test]
    fn no_tail_when_child_finished_or_list_mode() {
        let dir = tmpdir();
        let mut e = entry("c1", ChildSessionState::Completed, None);
        e.log_path = write_log(dir.path(), "c1", &format!("{USER_LINE}\n"));

        let mut v = SubagentViewState::new();
        v.open(vec![e.clone()]);
        assert!(!v.wants_tail(), "list mode never tails");
        v.handle_key(key(KeyCode::Enter), 10);
        assert!(!v.wants_tail(), "finished child never tails");

        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&e.log_path)
            .unwrap();
        writeln!(f, "{ASSISTANT_LINE}").unwrap();
        drop(f);
        v.on_tick_tail();
        assert!(
            !lines_to_string(&v.reader_lines).contains("done"),
            "no tail parse for a finished child"
        );
    }

    // ── refresh on lifecycle events ──────────────────────────────

    #[test]
    fn refresh_keeps_selection_by_id_and_updates_live_state() {
        let mut v = SubagentViewState::new();
        v.open(vec![
            entry("c1", ChildSessionState::Running, None),
            entry("c2", ChildSessionState::Running, None),
        ]);
        v.selected = 1;
        v.refresh_entries(vec![
            entry("c1", ChildSessionState::Completed, None),
            entry("c2", ChildSessionState::Failed, None),
        ]);
        assert_eq!(v.selected, 1, "selection tracks the id, not the index");
        assert_eq!(v.entries[v.selected].state, ChildSessionState::Failed);
    }

    #[test]
    fn refresh_drops_reader_when_child_disappears() {
        let dir = tmpdir();
        let mut e = entry("c1", ChildSessionState::Running, None);
        e.log_path = write_log(dir.path(), "c1", &format!("{USER_LINE}\n"));
        let mut v = SubagentViewState::new();
        v.open(vec![e]);
        v.handle_key(key(KeyCode::Enter), 10);
        assert_eq!(v.mode, SubagentMode::Reader);
        v.refresh_entries(vec![entry("c2", ChildSessionState::Running, None)]);
        assert_eq!(v.mode, SubagentMode::List, "orphaned reader falls back");
    }

    #[test]
    fn refresh_updates_reader_running_flag() {
        let dir = tmpdir();
        let mut e = entry("c1", ChildSessionState::Running, None);
        e.log_path = write_log(dir.path(), "c1", &format!("{USER_LINE}\n"));
        let mut v = SubagentViewState::new();
        v.open(vec![e]);
        v.handle_key(key(KeyCode::Enter), 10);
        assert!(v.wants_tail());
        v.refresh_entries(vec![entry("c1", ChildSessionState::Completed, None)]);
        assert!(!v.wants_tail(), "terminal refresh stops the tail");
    }

    // ── tail cap on initial load ─────────────────────────────────

    #[test]
    fn huge_log_is_tail_capped_with_notice() {
        let dir = tmpdir();
        let mut e = entry("c1", ChildSessionState::Completed, None);
        e.log_path = dir.path().join("c1.events.jsonl");
        // Build a log bigger than the cap: N filler lines then the payload.
        let filler = USER_LINE.replace("do the thing", &"x".repeat(400));
        let mut body = String::new();
        while (body.len() as u64) < INITIAL_TAIL_BYTES + 4096 {
            body.push_str(&filler);
            body.push('\n');
        }
        body.push_str(ASSISTANT_LINE);
        body.push('\n');
        std::fs::write(&e.log_path, &body).unwrap();

        let mut v = SubagentViewState::new();
        v.open(vec![e]);
        v.handle_key(key(KeyCode::Enter), 10);
        assert!(v.reader_truncated);
        let joined = lines_to_string(&v.reader_lines);
        assert!(joined.contains(TRUNCATED_NOTICE));
        assert!(joined.contains("done"), "the recent tail is present");
    }

    // ── Fix B: UTF-8 lossy at the tail-cap cut ───────────────────

    #[test]
    fn truncated_multibyte_log_still_renders_content() {
        // A >2MB log whose cap cut lands INSIDE a multi-byte char: with the
        // old strict `from_utf8` this produced an empty string (whole read
        // lost) and parked the offset on an invalid boundary (every later
        // tail read lost too). Lossy decoding + the partial-first-line drain
        // must render the tail's real content.
        //
        // The cap cut is measured from the END of the file
        // (`start = len - INITIAL_TAIL_BYTES`), so the cut is steered onto a
        // continuation byte by adjusting a small tail pad: the zh lines have
        // period 19 bytes (6×3B chars + \n), and 12 of 19 byte positions in
        // a line are continuation bytes — one pad value in 0..19 must land
        // on one (deterministic, no sleeping on mtime).
        let dir = tmpdir();
        let mut e = entry("c1", ChildSessionState::Completed, None);
        e.log_path = dir.path().join("c1.events.jsonl");

        let zh = "中文日志内容"; // 6 chars × 3 bytes
        let line = format!("{zh}\n"); // 19 bytes per line
        let payload = format!("{ASSISTANT_LINE}\n");
        let cap = INITIAL_TAIL_BYTES as usize;
        let lines: Vec<u8> = line.as_bytes().repeat(cap / line.len() + 64);
        assert!(lines.len() + payload.len() > cap);

        let mut file_bytes = Vec::new();
        for tail_pad in 0..19usize {
            let candidate = [lines.as_slice(), payload.as_bytes(), &vec![b' '; tail_pad]].concat();
            let start = candidate.len() - cap;
            if candidate[start] & 0xC0 == 0x80 {
                file_bytes = candidate;
                break;
            }
        }
        assert!(
            !file_bytes.is_empty(),
            "a tail pad must force a mid-char cut"
        );
        let start = file_bytes.len() - cap;
        assert_eq!(
            file_bytes[start] & 0xC0,
            0x80,
            "cut must land on a continuation byte"
        );
        std::fs::write(&e.log_path, &file_bytes).unwrap();

        let mut v = SubagentViewState::new();
        v.open(vec![e]);
        v.handle_key(key(KeyCode::Enter), 10);
        assert!(v.reader_truncated);
        let joined = lines_to_string(&v.reader_lines);
        assert!(
            joined.contains("done"),
            "the ASCII payload line after the cut must render: {joined}"
        );
        // The cut zh char's stray bytes form the partial first line, which
        // is drained — so no U+FFFD residue should surface in the rendered
        // content lines.
        assert!(
            !joined.lines().any(|l| l.contains('\u{FFFD}')),
            "partial first line is dropped, not rendered: {joined}"
        );
    }

    #[test]
    fn tail_reads_survive_a_torn_trailing_multibyte_char() {
        // A running child whose last append ends mid-UTF-8-character: the
        // incremental tail read must not permanently lose the remainder.
        // With strict from_utf8 the torn read returned None and the offset
        // stayed parked — the completed char never appeared.
        let dir = tmpdir();
        let mut e = entry("c1", ChildSessionState::Running, None);
        e.log_path = write_log(dir.path(), "c1", &format!("{USER_LINE}\n"));

        let mut v = SubagentViewState::new();
        v.open(vec![e.clone()]);
        v.handle_key(key(KeyCode::Enter), 10);

        // Append an assistant event, then 1 byte of a 3-byte char (torn).
        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&e.log_path)
            .unwrap();
        writeln!(f, "{ASSISTANT_LINE}").unwrap();
        f.write_all(&"中".as_bytes()[..1]).unwrap(); // 0xE4 only (torn)
        drop(f);

        v.on_tick_tail();
        let joined = lines_to_string(&v.reader_lines);
        assert!(
            joined.contains("done"),
            "the complete line BEFORE the torn char must render: {joined}"
        );

        // Writer completes the char + rest of the line; the next tick must
        // still see it (offset was not poisoned).
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&e.log_path)
            .unwrap();
        f.write_all(&"中".as_bytes()[1..]).unwrap(); // 0xB8 0xAD
        f.write_all(" 尾\n".as_bytes()).unwrap();
        drop(f);
        v.on_tick_tail();
        let joined = lines_to_string(&v.reader_lines);
        assert!(
            !joined.is_empty(),
            "tail keeps working after a torn multibyte append"
        );
        let final_len = std::fs::metadata(&e.log_path).unwrap().len();
        assert_eq!(
            *v.tail_offsets.get("c1").unwrap(),
            final_len,
            "offset advances past the completed line"
        );
    }

    fn lines_to_string(lines: &[Line<'static>]) -> String {
        lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.to_string())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}
