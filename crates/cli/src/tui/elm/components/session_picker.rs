//! Session picker popup.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use std::collections::HashMap;

use nca_common::session::{SessionSnapshot, SessionStatus};

use super::searchable_list::{ClearWidget, centered_rect, theme};

// ── State + Actions ──────────────────────────────────────────────

pub(crate) struct SessionPickerState {
    search: String,
    index: usize,
    scroll: usize,
    entries: Vec<SessionSnapshot>,
    /// Lineage depth per entry (aligned with `entries`; computed on `open`).
    depths: Vec<usize>,
    /// Transient guard notice (R10): shown when a running child selection
    /// was refused. Cleared by any navigation/filter key.
    notice: Option<String>,
    current_session_id: String,
}

#[derive(Debug)]
pub(crate) enum SessionPickerAction {
    None,
    Close,
    ResumeSession(String),
}

const SESSION_PICKER_MAX_ROWS: usize = 16;
/// Hard cap on parent-chain walks: guards against corrupt lineage data
/// (e.g. a parent cycle in hand-edited session jsons) so depth computation
/// always terminates. Real nesting is far shallower.
const MAX_LINEAGE_DEPTH: usize = 8;
/// R10 guard notice for a refused running-child selection.
const RUNNING_GUARD_NOTICE: &str = "child session still running — use the subagent view or wait";

/// Compute the lineage depth of every entry by walking the
/// `parent_session_id` chain. Roots (no parent) are depth 0; each resolvable
/// parent hop adds one. A parent missing from `entries` still counts the hop
/// that reached it (an orphan child is visibly a child). The walk is capped
/// at [`MAX_LINEAGE_DEPTH`] so cycles (a↔b) terminate instead of hanging.
pub(crate) fn lineage_depths(entries: &[SessionSnapshot]) -> Vec<usize> {
    let mut index_of: HashMap<&str, usize> = HashMap::with_capacity(entries.len());
    for (i, e) in entries.iter().enumerate() {
        index_of.insert(e.id.as_str(), i);
    }

    (0..entries.len())
        .map(|start| {
            let mut depth = 0usize;
            let mut cur = entries[start].parent_session_id.as_deref();
            while let Some(parent_id) = cur {
                if depth >= MAX_LINEAGE_DEPTH {
                    break;
                }
                depth += 1;
                match index_of.get(parent_id) {
                    Some(&i) => cur = entries[i].parent_session_id.as_deref(),
                    None => break,
                }
            }
            depth
        })
        .collect()
}

/// Display prefix for a lineage depth: one `↳ ` arrow per depth level
/// (depth 0 — a root session — gets no prefix).
pub(crate) fn lineage_prefix(depth: usize) -> String {
    "↳ ".repeat(depth)
}

/// True when selecting this entry must be refused (R10): the session is
/// non-terminal on disk AND is not the session we are already in. Resuming
/// it would violate the single-writer invariant on its session json.
pub(crate) fn selection_blocked(snap: &SessionSnapshot, current_session_id: &str) -> bool {
    snap.status == SessionStatus::Running && snap.id != current_session_id
}

/// Single source of truth for a picker row's display text.
/// `depth` from [`lineage_depths`]; `is_current` marks the live session.
pub(crate) fn row_display(snap: &SessionSnapshot, depth: usize, is_current: bool) -> String {
    let marker = if is_current { " *" } else { "" };
    let prefix = lineage_prefix(depth);
    let badge = if depth > 0 { " [child]" } else { "" };
    // R10 display mirror of `selection_blocked`: the (running, non-current)
    // pair is exactly the refused state; `is_current` is already in hand.
    let running = if snap.status == SessionStatus::Running && !is_current {
        " (running)"
    } else {
        ""
    };
    if let Some(title) = &snap.session_title {
        format!(" {prefix}{title}{marker}{badge}{running}  [{}]", snap.id)
    } else {
        format!(" {prefix}{}{marker}{badge}{running}", snap.id)
    }
}

impl SessionPickerState {
    pub fn new() -> Self {
        Self {
            search: String::new(),
            index: 0,
            scroll: 0,
            entries: Vec::new(),
            depths: Vec::new(),
            notice: None,
            current_session_id: String::new(),
        }
    }

    pub fn open(&mut self, entries: Vec<SessionSnapshot>, current_session_id: String) {
        self.depths = lineage_depths(&entries);
        self.entries = entries;
        self.current_session_id = current_session_id;
        self.search.clear();
        self.index = 0;
        self.scroll = 0;
        self.notice = None;
    }

    pub fn close(&mut self) {
        self.entries.clear();
        self.depths.clear();
        self.current_session_id.clear();
        self.search.clear();
        self.index = 0;
        self.scroll = 0;
        self.notice = None;
    }

    pub fn is_open(&self) -> bool {
        true
    }

    fn filtered_count(&self) -> usize {
        let filter = self.search.to_ascii_lowercase();
        self.entries
            .iter()
            .filter(|s| {
                filter.is_empty()
                    || s.id.to_ascii_lowercase().contains(&filter)
                    || s.session_title
                        .as_ref()
                        .map(|t| t.to_ascii_lowercase().contains(&filter))
                        .unwrap_or(false)
            })
            .count()
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> SessionPickerAction {
        let count = self.filtered_count();
        match (key.code, key.modifiers) {
            (KeyCode::Esc, _) => SessionPickerAction::Close,
            (KeyCode::Up, _) => {
                self.index = self.index.saturating_sub(1);
                self.notice = None;
                SessionPickerAction::None
            }
            (KeyCode::Down, _) => {
                if count > 0 {
                    self.index = (self.index + 1).min(count.saturating_sub(1));
                }
                self.notice = None;
                SessionPickerAction::None
            }
            (KeyCode::Enter, _) => {
                let filter = self.search.to_ascii_lowercase();
                let filtered: Vec<&SessionSnapshot> = self
                    .entries
                    .iter()
                    .filter(|s| {
                        filter.is_empty()
                            || s.id.to_ascii_lowercase().contains(&filter)
                            || s.session_title
                                .as_ref()
                                .map(|t| t.to_ascii_lowercase().contains(&filter))
                                .unwrap_or(false)
                    })
                    .collect();
                let pick = self.index.min(filtered.len().saturating_sub(1));
                if let Some(snap) = filtered.get(pick) {
                    // R10 guard: never `switch_to` a session that is still
                    // running somewhere else — that would break the
                    // single-writer invariant on its session json. Show a
                    // notice instead; the picker stays open.
                    if selection_blocked(snap, &self.current_session_id) {
                        self.notice = Some(RUNNING_GUARD_NOTICE.to_string());
                        return SessionPickerAction::None;
                    }
                    let id = snap.id.clone();
                    return SessionPickerAction::ResumeSession(id);
                }
                SessionPickerAction::None
            }
            (KeyCode::Backspace, _) => {
                self.search.pop();
                self.index = 0;
                self.scroll = 0;
                self.notice = None;
                SessionPickerAction::None
            }
            (KeyCode::Char(c), KeyModifiers::NONE | KeyModifiers::SHIFT) => {
                self.search.push(c);
                self.index = 0;
                self.scroll = 0;
                self.notice = None;
                SessionPickerAction::None
            }
            _ => SessionPickerAction::None,
        }
    }

    pub fn render(&mut self, frame: &mut Frame) {
        let area = frame.area();
        let filter = self.search.to_ascii_lowercase();
        let filtered_indices: Vec<usize> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                filter.is_empty()
                    || s.id.to_ascii_lowercase().contains(&filter)
                    || s.session_title
                        .as_ref()
                        .map(|t| t.to_ascii_lowercase().contains(&filter))
                        .unwrap_or(false)
            })
            .map(|(i, _)| i)
            .collect();

        let n_filtered = filtered_indices.len();
        let viewport_rows = n_filtered.min(SESSION_PICKER_MAX_ROWS);
        let notice_extra = if self.notice.is_some() { 1 } else { 0 };
        let rows = (viewport_rows as u16)
            .saturating_add(8 + notice_extra)
            .max(10);
        let popup_area = centered_rect(area, 56, rows);
        let pick = self.index.min(n_filtered.saturating_sub(1));

        // Adjust scroll
        if pick < self.scroll {
            self.scroll = pick;
        } else if viewport_rows > 0 && pick >= self.scroll + viewport_rows {
            self.scroll = pick.saturating_sub(viewport_rows - 1);
        }
        self.scroll = self.scroll.min(n_filtered.saturating_sub(viewport_rows));
        let list_start = self.scroll;
        let list_end = (list_start + viewport_rows).min(n_filtered);

        let search_display = if self.search.is_empty() {
            "type to filter".to_string()
        } else {
            self.search.clone()
        };

        let mut lines: Vec<Line> = vec![
            Line::from(vec![
                Span::styled(
                    " Search ",
                    Style::default()
                        .fg(theme::MUTED)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(search_display, Style::default().fg(theme::TEXT)),
            ]),
            Line::default(),
        ];

        if filtered_indices.is_empty() {
            lines.push(Line::from(Span::styled(
                " No matching sessions",
                Style::default().fg(theme::MUTED),
            )));
        } else {
            if list_start > 0 {
                lines.push(Line::from(Span::styled(
                    format!("  ▲ {} more", list_start),
                    Style::default().fg(theme::MUTED),
                )));
            }
            for (vis_idx, &filt_idx) in filtered_indices
                .iter()
                .enumerate()
                .skip(list_start)
                .take(list_end.saturating_sub(list_start))
            {
                let snap = &self.entries[filt_idx];
                let depth = self.depths.get(filt_idx).copied().unwrap_or(0);
                let is_current = snap.id == self.current_session_id;
                let display = row_display(snap, depth, is_current);
                let st = if vis_idx == pick {
                    Style::default()
                        .fg(Color::Black)
                        .bg(theme::USER)
                        .add_modifier(Modifier::BOLD)
                } else if selection_blocked(snap, &self.current_session_id) {
                    // Running child elsewhere: amber so the blocked state is
                    // visible before the user hits Enter.
                    Style::default().fg(theme::WARN)
                } else {
                    Style::default().fg(theme::TEXT)
                };
                lines.push(Line::from(Span::styled(display, st)));
            }
            let remaining_below = n_filtered.saturating_sub(list_end);
            if remaining_below > 0 {
                lines.push(Line::from(Span::styled(
                    format!("  ▼ {} more", remaining_below),
                    Style::default().fg(theme::MUTED),
                )));
            }
        }

        // R10 guard notice: one warn-colored line between list and footer.
        // Borrowed, not consumed: the busy-animation redraw re-renders every
        // frame, so a `take()` here flashed the notice for a single frame.
        // Clearing is owned by the navigation/filter/open key paths
        // (`handle_key` / `open`) and stays untouched.
        if let Some(notice) = self.notice.as_ref() {
            lines.push(Line::default());
            lines.push(Line::from(Span::styled(
                format!(" ! {notice}"),
                Style::default().fg(theme::WARN),
            )));
        }

        lines.push(Line::default());
        lines.push(Line::from(Span::styled(
            " Enter resume · Esc close ",
            Style::default().fg(theme::MUTED),
        )));

        frame.render_widget(ClearWidget, popup_area);
        let popup = Paragraph::new(Text::from(lines))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(theme::BORDER))
                    .title(Span::styled(
                        " sessions ",
                        Style::default().fg(theme::MUTED),
                    )),
            )
            .style(Style::default().bg(theme::SURFACE))
            .wrap(Wrap { trim: false });
        frame.render_widget(popup, popup_area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn snap(id: &str, parent: Option<&str>, status: SessionStatus) -> SessionSnapshot {
        let now = Utc::now();
        SessionSnapshot {
            id: id.into(),
            created_at: now,
            updated_at: now,
            workspace: "/tmp/ws".into(),
            model: "m".into(),
            status,
            pid: None,
            socket_path: None,
            worktree_path: None,
            branch: None,
            base_branch: None,
            parent_session_id: parent.map(str::to_string),
            child_session_ids: Vec::new(),
            inherited_summary: None,
            spawn_reason: None,
            session_summary: None,
            session_title: Some(format!("task of {id}")),
            orchestration: None,
            agent_name: None,
            total_input_tokens: 0,
            total_output_tokens: 0,
            estimated_cost_usd: 0.0,
        }
    }

    fn enter() -> KeyEvent {
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)
    }

    fn down() -> KeyEvent {
        KeyEvent::new(KeyCode::Down, KeyModifiers::NONE)
    }

    // ── lineage_depths ────────────────────────────────────────────

    #[test]
    fn lineage_depths_roots_children_and_grandchildren() {
        let entries = vec![
            snap("root", None, SessionStatus::Completed),
            snap("child", Some("root"), SessionStatus::Completed),
            snap("grand", Some("child"), SessionStatus::Running),
        ];
        let depths = lineage_depths(&entries);
        assert_eq!(depths, vec![0, 1, 2]);
    }

    #[test]
    fn lineage_depths_parent_missing_from_list_still_counts_the_hop() {
        // A child whose parent json is gone (deleted/unreadable) must still
        // render as a child — depth 1 — not silently become a root.
        let entries = vec![snap("orphan", Some("ghost"), SessionStatus::Completed)];
        assert_eq!(lineage_depths(&entries), vec![1]);
    }

    #[test]
    fn lineage_depths_cycle_is_capped_not_hung() {
        // Corrupt lineage: a↔b mutual parents. The walk must terminate at
        // the depth cap, never spin forever.
        let entries = vec![
            snap("a", Some("b"), SessionStatus::Running),
            snap("b", Some("a"), SessionStatus::Running),
        ];
        let depths = lineage_depths(&entries);
        assert_eq!(depths, vec![MAX_LINEAGE_DEPTH, MAX_LINEAGE_DEPTH]);
    }

    #[test]
    fn lineage_depths_self_parent_is_capped() {
        let entries = vec![snap("self", Some("self"), SessionStatus::Running)];
        assert_eq!(lineage_depths(&entries), vec![MAX_LINEAGE_DEPTH]);
    }

    #[test]
    fn lineage_depths_deep_chain_is_capped_at_max() {
        let mut entries = vec![snap("root", None, SessionStatus::Completed)];
        for i in 1..=20 {
            let parent = entries[i - 1].id.clone();
            entries.push(snap(
                &format!("n{i}"),
                Some(&parent),
                SessionStatus::Completed,
            ));
        }
        let depths = lineage_depths(&entries);
        assert_eq!(
            depths[20], MAX_LINEAGE_DEPTH,
            "chain deeper than cap is clamped"
        );
        assert_eq!(depths[5], 5, "chains within the cap stay exact");
    }

    #[test]
    fn lineage_prefix_is_arrow_times_depth() {
        assert_eq!(lineage_prefix(0), "");
        assert_eq!(lineage_prefix(1), "↳ ");
        assert_eq!(lineage_prefix(3), "↳ ↳ ↳ ");
    }

    // ── row_display ───────────────────────────────────────────────

    #[test]
    fn row_display_root_has_no_lineage_marks() {
        let s = snap("root", None, SessionStatus::Completed);
        assert_eq!(row_display(&s, 0, false), " task of root  [root]");
    }

    #[test]
    fn row_display_child_gets_indent_and_badge() {
        let s = snap("kid", Some("root"), SessionStatus::Completed);
        assert_eq!(row_display(&s, 1, false), " ↳ task of kid [child]  [kid]");
    }

    #[test]
    fn row_display_nested_child_double_indent() {
        let s = snap("grand", Some("kid"), SessionStatus::Completed);
        assert_eq!(
            row_display(&s, 2, false),
            " ↳ ↳ task of grand [child]  [grand]"
        );
    }

    #[test]
    fn row_display_running_non_current_marks_running() {
        let s = snap("kid", Some("root"), SessionStatus::Running);
        let display = row_display(&s, 1, false);
        assert!(
            display.contains("(running)"),
            "blocked running child must be visibly marked: {display}"
        );
    }

    #[test]
    fn row_display_running_current_has_no_running_mark() {
        let s = snap("me", None, SessionStatus::Running);
        assert!(
            !row_display(&s, 0, true).contains("(running)"),
            "the current session is never marked blocked"
        );
    }

    // ── R10 guard via handle_key ──────────────────────────────────

    #[test]
    fn enter_on_running_child_is_refused_with_notice() {
        let mut p = SessionPickerState::new();
        p.open(
            vec![
                snap("current", None, SessionStatus::Running),
                snap("busy-child", Some("current"), SessionStatus::Running),
            ],
            "current".into(),
        );
        p.index = 1; // select the running child

        match p.handle_key(enter()) {
            SessionPickerAction::ResumeSession(_) => {
                panic!("running child must never emit ResumeSession (single-writer violation)")
            }
            SessionPickerAction::Close => panic!("picker must stay open on refusal"),
            SessionPickerAction::None => {}
        }
        assert_eq!(p.notice.as_deref(), Some(RUNNING_GUARD_NOTICE));
    }

    #[test]
    fn enter_on_finished_child_still_resumes() {
        let mut p = SessionPickerState::new();
        p.open(
            vec![
                snap("current", None, SessionStatus::Running),
                snap("done-child", Some("current"), SessionStatus::Completed),
            ],
            "current".into(),
        );
        p.index = 1;

        match p.handle_key(enter()) {
            SessionPickerAction::ResumeSession(id) => assert_eq!(id, "done-child"),
            other => panic!("finished child must keep the resume path, got {other:?}"),
        }
    }

    #[test]
    fn enter_on_terminal_error_and_cancelled_children_resumes() {
        for status in [SessionStatus::Error, SessionStatus::Cancelled] {
            let label = format!("{status:?}");
            let mut p = SessionPickerState::new();
            p.open(
                vec![
                    snap("current", None, SessionStatus::Running),
                    snap("dead-child", Some("current"), status),
                ],
                "current".into(),
            );
            p.index = 1;
            match p.handle_key(enter()) {
                SessionPickerAction::ResumeSession(id) => assert_eq!(id, "dead-child"),
                other => panic!("{label} child must resume, got {other:?}"),
            }
        }
    }

    #[test]
    fn enter_on_running_current_session_is_allowed() {
        // The current session is Running by definition — selecting it is a
        // no-op resume that must NOT trip the guard.
        let mut p = SessionPickerState::new();
        p.open(
            vec![snap("current", None, SessionStatus::Running)],
            "current".into(),
        );
        match p.handle_key(enter()) {
            SessionPickerAction::ResumeSession(id) => assert_eq!(id, "current"),
            other => panic!("current session must stay selectable, got {other:?}"),
        }
    }

    #[test]
    fn notice_clears_on_navigation() {
        let mut p = SessionPickerState::new();
        p.open(
            vec![
                snap("current", None, SessionStatus::Running),
                snap("busy-child", Some("current"), SessionStatus::Running),
            ],
            "current".into(),
        );
        p.index = 1;
        assert!(matches!(p.handle_key(enter()), SessionPickerAction::None));
        assert!(p.notice.is_some(), "refusal sets the notice");

        assert!(matches!(p.handle_key(down()), SessionPickerAction::None));
        assert!(p.notice.is_none(), "navigation clears the notice");
    }

    #[test]
    fn guard_survives_filtering_to_only_the_running_child() {
        // Filter down to just the running child: Enter must still refuse.
        let mut p = SessionPickerState::new();
        p.open(
            vec![
                snap("current", None, SessionStatus::Running),
                snap("busy-child", Some("current"), SessionStatus::Running),
            ],
            "current".into(),
        );
        p.handle_key(KeyEvent::new(KeyCode::Char('b'), KeyModifiers::NONE));
        // filtered list = [busy-child], index 0 now points at it
        match p.handle_key(enter()) {
            SessionPickerAction::ResumeSession(_) => panic!("filtered view must respect the guard"),
            SessionPickerAction::None => {}
            SessionPickerAction::Close => panic!("picker must stay open"),
        }
        assert!(p.notice.is_some());
    }

    #[test]
    fn depths_are_recomputed_on_each_open_and_cleared_on_close() {
        let mut p = SessionPickerState::new();
        p.open(
            vec![
                snap("root", None, SessionStatus::Completed),
                snap("kid", Some("root"), SessionStatus::Completed),
            ],
            "root".into(),
        );
        assert_eq!(p.depths, vec![0, 1]);

        p.close();
        assert!(p.depths.is_empty(), "close resets lineage state");

        // Reopen with a different family: depths must track the new entries.
        p.open(
            vec![
                snap("other-root", None, SessionStatus::Completed),
                snap("other-kid", Some("other-root"), SessionStatus::Completed),
                snap("other-grand", Some("other-kid"), SessionStatus::Completed),
            ],
            "other-root".into(),
        );
        assert_eq!(p.depths, vec![0, 1, 2]);
    }

    #[test]
    fn open_resets_stale_notice() {
        let mut p = SessionPickerState::new();
        p.open(
            vec![
                snap("current", None, SessionStatus::Running),
                snap("busy", Some("current"), SessionStatus::Running),
            ],
            "current".into(),
        );
        p.index = 1;
        let _ = p.handle_key(enter());
        assert!(p.notice.is_some());

        p.open(
            vec![snap("current", None, SessionStatus::Running)],
            "current".into(),
        );
        assert!(p.notice.is_none(), "a fresh open must not inherit a notice");
    }

    // ── notice persistence across redraws (Fix minor) ────────────

    #[test]
    fn notice_survives_consecutive_renders() {
        // The busy animation re-renders every frame; the notice must persist
        // until an explicit clear (navigation/filter/open), not vanish after
        // the first render. Renders into a real TestBackend buffer and checks
        // the notice text is present on N consecutive renders.
        let mut p = SessionPickerState::new();
        p.open(
            vec![
                snap("current", None, SessionStatus::Running),
                snap("busy-child", Some("current"), SessionStatus::Running),
            ],
            "current".into(),
        );
        p.index = 1;
        assert!(matches!(p.handle_key(enter()), SessionPickerAction::None));
        assert!(p.notice.is_some(), "refusal sets the notice");

        for frame in 0..5 {
            let backend = ratatui::backend::TestBackend::new(80, 24);
            let mut terminal = ratatui::Terminal::new(backend).unwrap();
            terminal.draw(|f| p.render(f)).unwrap();
            let text: String = terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|c| c.symbol().to_string())
                .collect();
            assert!(
                text.contains("still running"),
                "notice visible on render #{frame}: {text}"
            );
        }
        assert!(
            p.notice.is_some(),
            "the notice field survives rendering (not consumed)"
        );

        // The explicit clear paths still work: navigation clears it.
        let _ = p.handle_key(down());
        assert!(p.notice.is_none(), "navigation still clears the notice");
        let backend = ratatui::backend::TestBackend::new(80, 24);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        terminal.draw(|f| p.render(f)).unwrap();
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol().to_string())
            .collect();
        assert!(
            !text.contains("still running"),
            "cleared notice renders nothing"
        );
    }
}
