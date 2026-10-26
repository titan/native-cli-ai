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
//!
//! The Reader's live feel is a TRANSIENT partial line: a second, cheap
//! parse pass over the appended bytes picks `TokensStreamed`/
//! `ReasoningStreamed` deltas out (substring prefilter — deltas are 96%+
//! of the bytes, so a full serde pass is off the table) and renders them
//! as one last line (text + `▌`, or muted italic `thinking…`). The next
//! final event replaces the whole line: deltas never become permanent
//! rows.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::Rect;
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
/// Backward-scan block size: the initial load walks a child log from EOF
/// toward the head in chunks of this many bytes, counting renderable events
/// (deltas don't count — they're the 96% of bytes the parser drops anyway).
const SCAN_BLOCK_BYTES: u64 = 256 * 1024;
/// Initial-load target: stop the backward scan once this many renderable
/// events are in the window. 400 ≈ several screens of transcript — enough
/// to see the conversation without bounded-IO concerns.
const RENDER_TARGET_EVENTS: usize = 400;
/// Hard cap on backward scanning from EOF: a pathological log (e.g. tens of
/// MB of reasoning deltas with almost no renderable events) must not cause
/// unbounded IO on open. Beyond this, older events are simply not loaded.
const MAX_SCAN_BYTES: u64 = 64 * 1024 * 1024;
/// Suffix of the truncated-earlier notice line.
const TRUNCATED_NOTICE: &str = "… log truncated (older events not loaded)";
/// Rolling window kept for the transient streaming line's text: a long
/// thinking stream must not grow the line unboundedly — only its tail is
/// shown.
const PARTIAL_TAIL_WINDOW: usize = 200;

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

// ── Transient streaming tail ─────────────────────────────────────

/// Which live stream a delta belongs to (drives the transient line's look).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TailKind {
    /// `TokensStreamed` — normal text color + `▌` cursor.
    Text,
    /// `ReasoningStreamed` — muted italic behind a `thinking…` prefix.
    Reasoning,
}

/// The transient "now streaming" line: live deltas accumulate here while
/// the child streams; the next final event REPLACES the whole line. Never
/// a permanent `reader_lines` entry (deltas are 96%+ of child-log bytes —
/// making them permanent rows is the row-explosion Fix C removed).
#[derive(Debug, Clone, PartialEq, Eq)]
struct PartialTail {
    kind: TailKind,
    text: String,
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
    /// Transient streaming line for the open child (see [`PartialTail`]).
    /// Rendered one line past the bottom of `reader_lines`, never part of
    /// it; cleared on every final/terminal event and Reader lifecycle hop.
    partial: Option<PartialTail>,
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
            partial: None,
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
        self.partial = None;
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
        let open_id = self.reader_session_id.clone();
        if let Some(id) = open_id.as_ref() {
            if !self.entries.iter().any(|e| &e.session_id == id) {
                // The open child vanished (session pruned); fall back to list.
                self.close_reader();
            } else {
                self.sync_reader_running();
                if !self.reader_child_running {
                    // Terminal state: the stream is over, and with it the
                    // transient partial — even when its closing final event
                    // never reached the log, a live "streaming" line must
                    // not linger next to a completed/failed marker.
                    self.partial = None;
                }
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
        // Bottom-anchor continuity: while the reader sits at (or beyond) the
        // last content edge, new lines keep the view pinned to the bottom —
        // the "live tail" the title advertises. Any deliberate scroll up
        // (reader_scroll < max_scroll) detaches: the user is reading history
        // and appends must not yank the viewport.
        let at_bottom = self.reader_scroll >= self.reader_lines.len().saturating_sub(2);
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
        // Dual-path parse — the ORDERING is the contract: final events
        // first; any assistant message / tool start / stream-terminal
        // event in this tick REPLACES the transient partial (the final's
        // own full text supersedes the deltas that streamed before it);
        // only then are this tick's deltas folded in. When both arrive in
        // one tick, the final wins.
        let finals = parse_events_from_str(&new_text);
        let final_replaces = finals.iter().any(|ev| {
            matches!(
                ev,
                AgentEvent::MessageReceived { .. }
                    | AgentEvent::MessageRecorded { .. }
                    | AgentEvent::ToolCallStarted { .. }
                    | AgentEvent::StepFailed { .. }
                    | AgentEvent::TurnCompleted { .. }
                    | AgentEvent::SessionEnded { .. }
            )
        });
        // A terminal event cuts the stream for the WHOLE tick: deltas that
        // share the tick with `StepFailed`/`TurnCompleted`/`SessionEnded`
        // are (at most) the head of a NEXT turn and must not re-arm the
        // transient line — the acceptance is "terminal ⇒ cleared".
        let terminal_cut = finals.iter().any(|ev| {
            matches!(
                ev,
                AgentEvent::StepFailed { .. }
                    | AgentEvent::TurnCompleted { .. }
                    | AgentEvent::SessionEnded { .. }
            )
        });
        for ev in &finals {
            self.reader_lines.extend(event_to_lines(ev));
        }
        if final_replaces {
            self.partial = None;
        }
        if !terminal_cut {
            self.absorb_stream_deltas(&new_text);
        }
        // Re-anchor after append (see `at_bottom` above): usize::MAX is the
        // bottom-anchor sentinel — render clamps it to max_scroll.
        if at_bottom {
            self.reader_scroll = usize::MAX;
        }
    }

    /// Fold this tick's delta lines into the transient partial line.
    ///
    /// Cheap by design: delta lines are 96%+ of a child log's bytes, so
    /// [`parse_stream_delta`] is a substring prefilter that rejects almost
    /// every line without touching serde; only a hit pays one small
    /// single-line JSON parse for the `delta` field. Same-kind deltas
    /// accumulate ACROSS ticks (the stream spans ticks); a kind switch
    /// resets the text. Deltas never become permanent
    /// [`Self::reader_lines`] entries.
    fn absorb_stream_deltas(&mut self, new_text: &str) {
        let mut pending: Option<PartialTail> = None;
        for line in new_text.lines() {
            let Some((kind, delta)) = parse_stream_delta(line) else {
                continue;
            };
            match &mut pending {
                // Same stream: append.
                Some(p) if p.kind == kind => p.text.push_str(&delta),
                // First delta, or a kind switch (text → reasoning or back):
                // a new stream starts, so the text resets.
                _ => pending = Some(PartialTail { kind, text: delta }),
            }
        }
        if let Some(p) = pending {
            match &mut self.partial {
                // The stream continues across the tick boundary: append.
                Some(cur) if cur.kind == p.kind => cur.text.push_str(&p.text),
                // A new stream (or a final cleared the old one): replace.
                _ => self.partial = Some(p),
            }
        }
        // No delta this tick: keep the existing partial — a mid-stream
        // pause (a tool running between text chunks) must not blank the
        // transient line.
    }

    /// Enter Reader mode for the selected row (loads the child log with the
    /// tail cap) — or directly for a given child (Tab cycle from List).
    pub(crate) fn open_reader(&mut self, session_id: Option<&str>) {
        // Switching children invalidates any transient line immediately.
        self.partial = None;
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
        self.partial = None;
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
        let vis_h = reader_viewport_rows(popup_area);
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
        // The transient streaming line rides ONE line past the permanent
        // content: drawn only while the viewport sits at the bottom edge
        // (`start == max_scroll` after clamping — the sentinel was already
        // resolved above). It never enters `reader_lines`, `max_scroll`, or
        // the per-child scroll memory, so scrolling up to read history
        // hides it without touching any anchor semantics.
        if start >= max_scroll
            && let Some(line) = self.partial.as_ref().map(partial_tail_line)
        {
            lines.push(line);
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

/// Content rows the reader can actually show: popup height minus borders
/// (2) minus the reserved footer (blank separator + key-hint line).
/// Exported so the model can pass the SAME height to `handle_key` for
/// PgUp/PgDn — a page step must match what the viewport shows.
pub(crate) fn reader_viewport_rows(popup_area: Rect) -> usize {
    popup_area.height.saturating_sub(4).max(1) as usize
}

fn muted_line(s: String) -> Line<'static> {
    Line::from(Span::styled(s, Style::default().fg(theme::MUTED)))
}

/// Substring prefilter markers for delta lines — the hot path of
/// [`absorb_stream_deltas`]. A full serde pass per line would re-parse the
/// 96%+ of child-log bytes the replay parser already drops; a substring
/// scan over these static markers rejects almost everything with no
/// allocation and no parse tree. Two spellings: serde_json's compact
/// `"type":"X"` and the pretty `"type": "X"`. Marker safety: no other
/// `AgentEvent` variant serializes a type string containing these markers
/// as a substring.
const TEXT_DELTA_COMPACT: &str = "\"type\":\"TokensStreamed\"";
const TEXT_DELTA_PRETTY: &str = "\"type\": \"TokensStreamed\"";
const REASON_DELTA_COMPACT: &str = "\"type\":\"ReasoningStreamed\"";
const REASON_DELTA_PRETTY: &str = "\"type\": \"ReasoningStreamed\"";

/// Extract a streaming delta from one (complete) log line, iff it is a
/// `TokensStreamed`/`ReasoningStreamed` event. The prefilter is the cost
/// model: allocation-free substring scans; only a hit pays one small
/// single-line JSON parse for the `delta` field.
fn parse_stream_delta(line: &str) -> Option<(TailKind, String)> {
    let kind = if line.contains(TEXT_DELTA_COMPACT) || line.contains(TEXT_DELTA_PRETTY) {
        TailKind::Text
    } else if line.contains(REASON_DELTA_COMPACT) || line.contains(REASON_DELTA_PRETTY) {
        TailKind::Reasoning
    } else {
        return None;
    };
    let v: serde_json::Value = serde_json::from_str(line.trim()).ok()?;
    let delta = v.get("delta")?.as_str()?.to_string();
    Some((kind, delta))
}

/// The rendered transient streaming line: `Text` = normal color + `▌`
/// cursor; `Reasoning` = muted italic behind a `thinking…` prefix. The text
/// is a rolling tail window ([`PARTIAL_TAIL_WINDOW`]) so a long thinking
/// stream cannot stretch the line without bound.
fn partial_tail_line(p: &PartialTail) -> Line<'static> {
    let tail = window_tail(&p.text, PARTIAL_TAIL_WINDOW);
    match p.kind {
        TailKind::Text => Line::from(Span::styled(
            format!("  {tail}▌"),
            Style::default().fg(theme::TEXT),
        )),
        TailKind::Reasoning => Line::from(Span::styled(
            format!("  thinking… {tail}"),
            Style::default()
                .fg(theme::MUTED)
                .add_modifier(Modifier::ITALIC),
        )),
    }
}

/// The last `max_chars` Unicode scalar values of `s` — no panic on char
/// boundaries and no allocation when the text already fits.
fn window_tail(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    s.chars().skip(s.chars().count() - max_chars).collect()
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

/// Load a child log for the reader (Fix C): a chunked backward scan from
/// EOF that bounds the window by RENDERABLE EVENTS, not bytes.
///
/// The old 2MB tail cap was byte-delimited, but 96%+ of child-log bytes are
/// `ReasoningStreamed`/`TokensStreamed` delta lines the parser drops — the
/// byte window was consumed before the skip, so a long thinking-heavy log
/// rendered as the last few fragments (observed: 7.4MB log, 2MB tail → 12
/// renderable events vs 140 in the whole file). The new scan walks
/// [`SCAN_BLOCK_BYTES`] blocks from EOF toward the head, counting events
/// that `event_to_lines` renders non-empty, and stops at
/// [`RENDER_TARGET_EVENTS`] — or at the file head, or at the
/// [`MAX_SCAN_BYTES`] hard cap for pathological logs.
///
/// Per-block counting skips each block's own partial head line (the byte
/// cut); a line straddling a block boundary reappears complete once the
/// older block is prepended, so the final rendered set is never smaller
/// than what the count promised — the count may undercount by ≤1 event per
/// boundary, which only ever makes the scan slightly longer, never shorter.
///
/// Returns `(text, next_offset, truncated)` where `next_offset` is where
/// incremental tailing should resume: the scan always reaches EOF, so this
/// is the file length; only the START boundary changed. `truncated` is
/// `true` when the window starts above the file head (target reached or
/// hard cap) — the caller then shows [`TRUNCATED_NOTICE`].
fn load_log_tailed(path: &PathBuf) -> (String, u64, bool) {
    let Ok(meta) = std::fs::metadata(path) else {
        return (String::new(), 0, false);
    };
    let len = meta.len();
    if len == 0 {
        return (String::new(), 0, false);
    }

    let scan_floor = len.saturating_sub(MAX_SCAN_BYTES);
    let mut window = String::new();
    let mut window_start = len;
    let mut renderable = 0usize;
    let mut block_end = len;

    while block_end > scan_floor {
        let block_start = block_end.saturating_sub(SCAN_BLOCK_BYTES).max(scan_floor);
        let block = read_from(path, block_start, block_end).unwrap_or_default();
        block_end = block_start;

        // Count this block's renderable events from its first complete line
        // (the head line may be cut mid-JSON by the byte boundary; it either
        // reappears complete in the final window or was never a full event).
        let count_from = block.find('\n').map(|nl| nl + 1).unwrap_or(block.len());
        renderable += parse_events_from_str(&block[count_from..])
            .iter()
            .filter(|ev| !event_to_lines(ev).is_empty())
            .count();

        // Prepend the block, keeping its raw head (the straddled line's
        // completion lives here; the one partial line that can remain is at
        // the window head and is drained once below).
        let mut combined = block;
        combined.push_str(&window);
        window = combined;
        window_start = block_start;

        if renderable >= RENDER_TARGET_EVENTS || block_start == 0 {
            break;
        }
    }

    let truncated = window_start > 0;
    if truncated && let Some(nl) = window.find('\n') {
        // Drop the window's partial first line (byte-cut mid-JSON).
        window.drain(..=nl);
    }
    (window, len, truncated)
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
    fn tail_appends_keep_bottom_anchor_but_detached_scrolling_is_left_alone() {
        // Live tail + bottom anchor: while the reader sits at the bottom,
        // appended events keep it pinned there (usize::MAX sentinel,
        // clamped to max_scroll at render). Once the user scrolls up, the
        // viewport must NOT be yanked back down by appends.
        let dir = tmpdir();
        let mut e = entry("c1", ChildSessionState::Running, None);
        e.log_path = write_log(dir.path(), "c1", &format!("{USER_LINE}\n"));

        let mut v = SubagentViewState::new();
        v.open(vec![e.clone()]);
        v.handle_key(key(KeyCode::Enter), 10);

        // At the bottom (sentinel): append re-anchors.
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&e.log_path)
            .unwrap();
        writeln!(f, "{ASSISTANT_LINE}").unwrap();
        drop(f);
        v.on_tick_tail();
        assert_eq!(
            v.reader_scroll,
            usize::MAX,
            "bottom-anchored reader stays anchored across appends"
        );

        // Scrolled up (reading history): append must not move the viewport.
        v.reader_scroll = 0;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&e.log_path)
            .unwrap();
        writeln!(f, "{ASSISTANT_LINE}").unwrap();
        drop(f);
        v.on_tick_tail();
        assert_eq!(
            v.reader_scroll, 0,
            "a detached (scrolled-up) viewport is never yanked by appends"
        );
    }

    #[test]
    fn reader_viewport_rows_reserve_the_footer() {
        // The reader renders vis_h content rows PLUS a blank + hint line
        // inside the popup: reserving only borders (old `height - 2`) pushed
        // vis_h + 2 rows into a vis_h-row area, clipping the last content
        // line and the key hints off screen. `height - 4` reserves the
        // footer so everything fits.
        assert_eq!(reader_viewport_rows(Rect::new(0, 0, 80, 30)), 26);
        assert_eq!(reader_viewport_rows(Rect::new(0, 0, 80, 5)), 1, "min 1");
        assert_eq!(reader_viewport_rows(Rect::new(0, 0, 80, 3)), 1, "min 1");
    }

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
    fn delta_heavy_log_loads_target_renderable_events() {
        // Fix C acceptance: a >2MB log that is >90% streaming-delta lines.
        // The old byte-delimited 2MB cap kept only the last ~10 renderable
        // events (deltas consumed the window before the parser could skip
        // them); the event-counted backward scan must surface the full
        // conversation (all 500 renderable events live before the delta
        // flood, so the whole file is scanned and nothing is dropped).
        let dir = tmpdir();
        let mut e = entry("c1", ChildSessionState::Completed, None);
        e.log_path = dir.path().join("c1.events.jsonl");

        const N_RENDERABLE: usize = 500;
        let mut body = String::new();
        // Renderable traffic first (small byte share)…
        for i in 0..N_RENDERABLE {
            body.push_str(&USER_LINE.replace("do the thing", &format!("msg {i}")));
            body.push('\n');
        }
        // …then the delta flood: >2MB of lines the parser drops.
        let delta = r#"{"type":"ReasoningStreamed","delta":"thinking tokens fill the log here"}"#;
        while body.len() < 3 * 1024 * 1024 {
            body.push_str(delta);
            body.push('\n');
        }
        std::fs::write(&e.log_path, &body).unwrap();
        assert!(
            body.len() > 2 * 1024 * 1024,
            "precondition: exceeds the old byte cap"
        );

        let mut v = SubagentViewState::new();
        v.open(vec![e]);
        v.handle_key(key(KeyCode::Enter), 10);
        // The scan walks the delta flood (each block contributes 0
        // renderable events, so the old byte cap can no longer starve the
        // window), reaches the renderable region, and stops once 400 of the
        // 500 events are in hand: the newest 400 render, the oldest 100
        // stay unloaded — instead of the old ~10 surviving fragments.
        assert!(v.reader_truncated);
        let joined = lines_to_string(&v.reader_lines);
        assert!(joined.contains(TRUNCATED_NOTICE));
        let rendered = v
            .reader_lines
            .iter()
            .filter(|l| !l.spans.is_empty())
            .count();
        assert!(
            rendered >= RENDER_TARGET_EVENTS,
            "window must hold ≥{RENDER_TARGET_EVENTS} renderable events (old logic: <10); got {rendered}"
        );
        assert!(joined.contains("msg 499"), "the newest event renders");
        assert!(
            !joined.contains("msg 0\n"),
            "events older than the target stay unloaded"
        );
    }

    #[test]
    fn render_target_bounds_the_window_with_notice() {
        // More than RENDER_TARGET_EVENTS renderable events: the scan stops
        // at the target and reports truncation (older events not loaded).
        let dir = tmpdir();
        let mut e = entry("c1", ChildSessionState::Completed, None);
        e.log_path = dir.path().join("c1.events.jsonl");

        // Geometry: SHORT lines (~60B) so one 256KB block holds ~4k
        // renderable events — the target is reached inside the LAST block,
        // so the scan stops before the head and reports truncation. (Long
        // lines can never trigger the cap: <400 events fit in a block, so
        // the scan would always reach the head.)
        const N: usize = RENDER_TARGET_EVENTS * 12;
        let mut body = String::new();
        for i in 0..N {
            body.push_str(&USER_LINE.replace("do the thing", &format!("msg {i}")));
            body.push('\n');
        }
        assert!(body.len() as u64 > SCAN_BLOCK_BYTES);
        std::fs::write(&e.log_path, &body).unwrap();

        let mut v = SubagentViewState::new();
        v.open(vec![e]);
        v.handle_key(key(KeyCode::Enter), 10);
        assert!(v.reader_truncated);
        let joined = lines_to_string(&v.reader_lines);
        assert!(joined.contains(TRUNCATED_NOTICE));
        // The newest events are in the window; the very oldest are not.
        let last = format!("msg {}", N - 1);
        assert!(joined.contains(&last), "the newest events must render");
        assert!(
            !joined.contains("msg 0 "),
            "events older than the target stay unloaded"
        );
    }

    #[test]
    fn small_log_loads_fully_without_truncation() {
        // < one scan block: whole file, no notice — identical to the old
        // small-log behavior.
        let dir = tmpdir();
        let mut e = entry("c1", ChildSessionState::Completed, None);
        e.log_path = dir.path().join("c1.events.jsonl");
        let body = format!("{USER_LINE}\n{ASSISTANT_LINE}\n");
        std::fs::write(&e.log_path, &body).unwrap();

        let (text, next_offset, truncated) = load_log_tailed(&e.log_path);
        assert_eq!(text, body, "small log loads verbatim");
        assert_eq!(next_offset, body.len() as u64, "tail resumes at EOF");
        assert!(!truncated);
    }

    #[test]
    fn partial_first_line_at_window_head_is_dropped() {
        // Force truncation with a single-block scan: renderable events past
        // the target make the window start mid-file, and the head line is
        // cut mid-JSON — it must be dropped, not parsed as garbage.
        let dir = tmpdir();
        let mut e = entry("c1", ChildSessionState::Completed, None);
        e.log_path = dir.path().join("c1.events.jsonl");

        // Same short-line geometry as the bounds test: one block holds
        // well over the target, so the window head lands mid-file and its
        // first line is a byte-cut fragment.
        const N: usize = RENDER_TARGET_EVENTS * 12;
        let mut body = String::new();
        for i in 0..N {
            body.push_str(&USER_LINE.replace("do the thing", &format!("msg {i}")));
            body.push('\n');
        }
        assert!(body.len() as u64 > SCAN_BLOCK_BYTES);
        std::fs::write(&e.log_path, &body).unwrap();

        let (text, next_offset, truncated) = load_log_tailed(&e.log_path);
        assert!(
            truncated,
            "window stops at the render target, above the head"
        );
        assert_eq!(next_offset, body.len() as u64, "tail resumes at EOF");
        // The window text starts at a line boundary: the byte-cut partial
        // head line is dropped, so parsing sees only complete lines.
        assert!(
            text.starts_with('{'),
            "window must start on a complete line"
        );
        let parsed = parse_events_from_str(&text);
        assert!(
            parsed.len() >= RENDER_TARGET_EVENTS,
            "window holds at least the render target: {}",
            parsed.len()
        );
        // Every dropped byte belonged to the partial head line: the parsed
        // set is exactly the complete lines in the window.
        assert_eq!(parsed.len(), text.lines().count());
    }

    // ── Fix B: UTF-8 lossy at the tail-cap cut ───────────────────

    #[test]
    fn truncated_multibyte_log_still_renders_content() {
        // (Fix C rework) The backward scan's window head can land mid-UTF-8
        // character inside the delta/zh flood: lossy decoding must degrade
        // the cut to U+FFFD and the partial-first-line drain must keep the
        // residue out of the rendered output — same contract as Fix B, new
        // cut geometry (event-counted block walk instead of a fixed 2MB cap).
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

        // zh filler: no renderable events, >1 block so the scan walks at
        // least one boundary; the USER payload at the end stops the scan on
        // the first block. The window head then sits inside the zh flood at
        // an arbitrary byte offset — steer it onto a continuation byte with
        // a small head pad (line period 19 bytes, 12 of 19 positions are
        // continuation bytes).
        let zh = "中文日志内容"; // 6 chars × 3 bytes
        let line = format!("{zh}\n"); // 19 bytes per line
        // The last block must hold ≥target renderable events by itself, so
        // the scan stops after ONE block and the head lands in the flood.
        let mut payload = String::new();
        for i in 0..RENDER_TARGET_EVENTS {
            payload.push_str(&USER_LINE.replace("do the thing", &format!("zh-test {i}")));
            payload.push('\n');
        }
        payload.push_str(ASSISTANT_LINE);
        payload.push('\n');
        let flood: Vec<u8> = line
            .as_bytes()
            .repeat((SCAN_BLOCK_BYTES / line.len() as u64 + 8) as usize);

        let mut file_bytes = Vec::new();
        for head_pad in 0..19usize {
            let candidate = [&vec![b' '; head_pad], flood.as_slice(), payload.as_bytes()].concat();
            // Window head for a target-1 stop: the scan consumed one block
            // (payload found) — head = len - SCAN_BLOCK_BYTES (approx; any
            // in-flood offset proves the point, steer for continuation).
            let head = candidate.len().saturating_sub(SCAN_BLOCK_BYTES as usize);
            if head > 0 && candidate[head] & 0xC0 == 0x80 {
                file_bytes = candidate;
                break;
            }
        }
        assert!(
            !file_bytes.is_empty(),
            "a head pad must force a mid-char cut"
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

    // ── transient streaming line (partial tail) ─────────────────

    const TOKEN_DELTA: fn(&str) -> String =
        |d: &str| format!(r#"{{"type":"TokensStreamed","delta":"{d}"}}"#);
    const REASON_DELTA: fn(&str) -> String =
        |d: &str| format!(r#"{{"type":"ReasoningStreamed","delta":"{d}"}}"#);

    fn open_running_reader(body: &str) -> (tempfile::TempDir, SubagentViewState, PathBuf) {
        let dir = tmpdir();
        let mut e = entry("c1", ChildSessionState::Running, None);
        e.log_path = write_log(dir.path(), "c1", body);
        let mut v = SubagentViewState::new();
        v.open(vec![e.clone()]);
        v.handle_key(key(KeyCode::Enter), 10);
        assert_eq!(v.mode, SubagentMode::Reader);
        assert!(v.wants_tail());
        (dir, v, e.log_path)
    }

    fn append_line(p: &PathBuf, line: &str) {
        let mut f = std::fs::OpenOptions::new().append(true).open(p).unwrap();
        writeln!(f, "{line}").unwrap();
    }

    #[test]
    fn partial_appears_and_accumulates_from_deltas() {
        let (_dir, mut v, log) = open_running_reader(&format!("{USER_LINE}\n"));
        append_line(&log, &TOKEN_DELTA("Hel"));
        v.on_tick_tail();
        let p = v.partial.clone().expect("first delta mounts the partial");
        assert_eq!(p.kind, TailKind::Text);
        assert_eq!(p.text, "Hel");
        assert!(
            !lines_to_string(&v.reader_lines).contains("Hel"),
            "partial is never a permanent line"
        );

        append_line(&log, &TOKEN_DELTA("lo world"));
        v.on_tick_tail();
        let p = v.partial.as_ref().expect("second delta keeps the partial");
        assert_eq!(p.text, "Hel".to_string() + "lo world");

        // A tick with no new bytes keeps the partial untouched.
        v.on_tick_tail();
        assert_eq!(
            v.partial.as_ref().map(|p| p.text.clone()),
            Some("Hello world".to_string()),
            "an idle tick must not blank the transient line"
        );
    }

    #[test]
    fn partial_kind_switch_resets_text() {
        let (_dir, mut v, log) = open_running_reader(&format!("{USER_LINE}\n"));
        append_line(&log, &REASON_DELTA("plan it "));
        append_line(&log, &TOKEN_DELTA("answer"));
        v.on_tick_tail();
        let p = v.partial.clone().expect("a delta arrived");
        assert_eq!(p.kind, TailKind::Text);
        assert_eq!(
            p.text, "answer",
            "the kind switch (reasoning → text) resets the accumulated text"
        );
    }

    #[test]
    fn partial_switch_between_reasoning_and_text() {
        let (_dir, mut v, log) = open_running_reader(&format!("{USER_LINE}\n"));
        append_line(&log, &TOKEN_DELTA("ans"));
        append_line(&log, &REASON_DELTA("plan"));
        v.on_tick_tail();
        let p = v.partial.clone().expect("a delta arrived");
        assert_eq!(p.kind, TailKind::Reasoning);
        assert_eq!(p.text, "plan");
    }

    #[test]
    fn same_tick_final_event_replaces_the_partial() {
        let (_dir, mut v, log) = open_running_reader(&format!("{USER_LINE}\n"));
        append_line(&log, &TOKEN_DELTA("partial words"));
        v.on_tick_tail();
        assert!(v.partial.is_some());

        // Same tick: the final AND a next-stream delta land together.
        append_line(&log, &ASSISTANT_LINE);
        append_line(&log, &TOKEN_DELTA("next"));
        v.on_tick_tail();
        assert!(v.partial.is_some(), "the next stream's delta re-arms");
        assert_eq!(v.partial.as_ref().unwrap().text, "next");
        let joined = lines_to_string(&v.reader_lines);
        assert!(joined.contains("done"), "the final line renders");

        // Final alone (no trailing delta): the transient line disappears.
        append_line(&log, &TOKEN_DELTA("more"));
        v.on_tick_tail();
        append_line(&log, &ASSISTANT_LINE);
        v.on_tick_tail();
        assert!(v.partial.is_none(), "a final with no next delta clears");
        assert!(lines_to_string(&v.reader_lines).matches("done").count() >= 2);
    }

    #[test]
    fn turn_completed_and_step_failed_clear_the_partial() {
        let (_dir, mut v, log) = open_running_reader(&format!("{USER_LINE}\n"));
        append_line(&log, &REASON_DELTA("deep"));
        v.on_tick_tail();
        assert!(v.partial.is_some());

        let turn_done = r#"{"type":"TurnCompleted","turn_id":1,"duration_ms":5}"#;
        append_line(&log, turn_done);
        v.on_tick_tail();
        assert!(
            v.partial.is_none(),
            "TurnCompleted clears the transient line"
        );

        append_line(&log, &REASON_DELTA("again"));
        v.on_tick_tail();
        assert!(v.partial.is_some());

        let step_failed =
            r#"{"type":"StepFailed","turn_id":1,"step_index":0,"duration_ms":3,"error":"boom"}"#;
        append_line(&log, step_failed);
        v.on_tick_tail();
        assert!(v.partial.is_none(), "StepFailed clears the transient line");

        let ended = r#"{"type":"SessionEnded","reason":{"Completed":null}}"#;
        append_line(&log, &REASON_DELTA("x"));
        append_line(&log, ended);
        v.on_tick_tail();
        assert!(
            v.partial.is_none(),
            "SessionEnded clears the transient line"
        );
    }

    #[test]
    fn terminal_refresh_and_close_clear_the_partial() {
        let (_dir, mut v, _log) = open_running_reader(&format!("{USER_LINE}\n"));
        // Seed a partial directly: only the last tick's new lines matter, so
        // a log-only seed cannot survive a tick boundary (the holdback has
        // already consumed the seed bytes).
        v.partial = Some(PartialTail {
            kind: TailKind::Text,
            text: "live".into(),
        });
        assert!(v.partial.is_some());

        // Child reaches a terminal state via a lifecycle refresh.
        v.refresh_entries(vec![entry("c1", ChildSessionState::Completed, None)]);
        assert!(v.partial.is_none(), "terminal refresh clears the partial");

        // Reader open on a terminal child: never any partial again.
        v.open_reader(None);
        assert!(v.partial.is_none());

        // Reader still on a running child: closing it drops the partial.
        v.refresh_entries(vec![entry("c1", ChildSessionState::Running, None)]);
        v.open_reader(None);
        v.partial = Some(PartialTail {
            kind: TailKind::Text,
            text: "x".into(),
        });
        v.handle_key(key(KeyCode::Esc), 10);
        assert!(v.partial.is_none(), "back to List clears the partial");

        v.handle_key(key(KeyCode::Enter), 10);
        v.partial = Some(PartialTail {
            kind: TailKind::Text,
            text: "y".into(),
        });
        v.close_reader();
        assert!(v.partial.is_none(), "close_reader clears the partial");
    }

    #[test]
    fn long_reasoning_shows_only_the_tail_window() {
        let long: String = "αβγδ".repeat(100); // 400 chars, multibyte
        let (_dir, mut v, log) = open_running_reader(&format!("{USER_LINE}\n"));
        append_line(&log, &REASON_DELTA(&long));
        v.on_tick_tail();
        let p = v.partial.as_ref().expect("delta mounted");
        let line = partial_tail_line(p);
        let text: String = line.spans.iter().map(|s| s.content.to_string()).collect();
        assert!(
            text.contains("thinking…"),
            "reasoning renders behind the prefix: {text}"
        );
        assert_eq!(text.chars().count(), "  thinking… ".chars().count() + 200);
        assert!(text.ends_with('δ'), "the window keeps the stream's tail");
    }

    #[test]
    fn torn_delta_line_is_held_back_until_complete() {
        let (dir, mut v, log) = open_running_reader(&format!("{USER_LINE}\n"));
        // Write half of a delta line (no trailing newline): the JSON is
        // missing its closing `"}` — mid-append torn.
        let torn_head = {
            let full = TOKEN_DELTA("TO");
            let cut = full.len() - "\"}".len();
            full[..cut].to_string()
        };
        let mut f = std::fs::OpenOptions::new().append(true).open(&log).unwrap();
        write!(f, "{torn_head}").unwrap();
        drop(f);
        v.on_tick_tail();
        assert!(
            v.partial.is_none(),
            "a torn (newline-less) delta line must not render this tick"
        );

        // Complete the line: the next tick shows the delta.
        append_line(&log, "\"}");
        v.on_tick_tail();
        let p = v
            .partial
            .as_ref()
            .expect("completed line mounts the partial");
        assert!(
            p.text.contains("TO"),
            "the held-back delta renders: {}",
            p.text
        );
        let _ = dir;
    }

    #[test]
    fn partial_renders_one_line_past_content_only_when_bottom_anchored() {
        // TestBackend render: the transient line is visible when
        // bottom-anchored and hidden when scrolled up.
        let (dir, mut v, log) = open_running_reader(&format!("{USER_LINE}\n"));
        for i in 0..80 {
            append_line(
                &log,
                &format!(r#"{{"type":"MessageReceived","role":"assistant","content":"msg {i}"}}"#),
            );
        }
        append_line(&log, &TOKEN_DELTA("typing now"));
        v.on_tick_tail();
        assert!(v.partial.is_some());

        let render_text = |view: &mut SubagentViewState| -> String {
            let backend = ratatui::backend::TestBackend::new(100, 24);
            let mut terminal = ratatui::Terminal::new(backend).unwrap();
            terminal.draw(|f| view.render(f)).unwrap();
            terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|c| c.symbol().to_string())
                .collect()
        };

        // Bottom-anchored: visible.
        assert!(v.reader_scroll == usize::MAX || v.reader_scroll > 0);
        let text = render_text(&mut v);
        assert!(
            text.contains("typing now"),
            "bottom-anchored reader shows the transient line"
        );
        // And it is transient: never part of the permanent lines.
        assert!(!lines_to_string(&v.reader_lines).contains("typing now"));

        // Scrolled up: hidden, and the scroll state itself is untouched.
        v.handle_key(key(KeyCode::PageUp), 10);
        let scroll_after_key = v.reader_scroll;
        let text = render_text(&mut v);
        assert!(
            !text.contains("typing now"),
            "a scrolled-up reader hides the transient line"
        );
        assert_eq!(v.reader_scroll, scroll_after_key);
        let _ = dir;
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
