//! Transcript rendering helpers — pure functions that emit `Line`s from blocks,
//! streaming text, and handle text-selection highlighting.
//!
//! Split from `transcript.rs` to keep state management and rendering separate.

use std::sync::Arc;
use std::time::Instant;

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthChar;

use crate::format::format_duration;
use crate::tui::state::DisplayBlock;

use super::searchable_list::theme;
use super::transcript::{LineAnswerHit, StreamingWrap, TranscriptHit};

// ── Text helpers ──────────────────────────────────────────────────

pub(super) fn char_width(ch: char) -> usize {
    ch.width().unwrap_or(1)
}

/// Wrap a single paragraph (a slice with no embedded `\n`) to `width` columns.
///
/// Pass 1 wraps on word boundaries; pass 2 hard-splits any line that still
/// exceeds `width` (CJK text without spaces). An empty paragraph yields a
/// single empty line. This is the exact per-paragraph core of [`wrap_text`],
/// extracted so [`super::transcript::StreamingWrap`] can wrap one paragraph at
/// a time without re-wrapping the whole buffer.
pub(super) fn wrap_paragraph(p: &str, width: usize) -> Vec<String> {
    if p.is_empty() {
        return vec![String::new()];
    }
    let mut out = Vec::new();
    let mut line = String::new();
    let mut line_w = 0usize;
    for word in p.split_whitespace() {
        let word_w: usize = word.chars().map(char_width).sum();
        if line.is_empty() {
            line = word.to_string();
            line_w = word_w;
        } else if line_w + 1 + word_w <= width {
            line.push(' ');
            line.push_str(word);
            line_w += 1 + word_w;
        } else {
            out.push(line);
            line = word.to_string();
            line_w = word_w;
        }
    }
    if !line.is_empty() {
        out.push(line);
    }
    // Second pass: split any line that still exceeds width (CJK text without spaces).
    let mut final_out = Vec::new();
    for l in out {
        if l.chars().map(char_width).sum::<usize>() <= width {
            final_out.push(l);
        } else {
            let mut cur = String::new();
            let mut cur_w = 0usize;
            for ch in l.chars() {
                let w = char_width(ch);
                if cur_w + w > width {
                    final_out.push(cur);
                    cur = String::new();
                    cur_w = 0;
                }
                cur.push(ch);
                cur_w += w;
            }
            if !cur.is_empty() {
                final_out.push(cur);
            }
        }
    }
    final_out
}

/// Wrap `s` to `width` columns, paragraph by paragraph.
///
/// Byte-for-byte equivalent to the historical single-pass implementation:
/// each `\n`-separated paragraph is delegated to [`wrap_paragraph`], and an
/// all-whitespace (but non-empty) string — which wraps to no lines — is
/// emitted verbatim as a single line.
pub(super) fn wrap_text(s: &str, width: usize) -> Vec<String> {
    if width < 8 {
        return vec![s.to_string()];
    }
    let mut final_out = Vec::new();
    for paragraph in s.split('\n') {
        final_out.extend(wrap_paragraph(paragraph, width));
    }
    if final_out.is_empty() && !s.is_empty() {
        final_out.push(s.to_string());
    }
    final_out
}

/// Lazily wrap a block's primary text at `w`, memoizing the result in
/// `cache` (index-aligned with the transcript's `blocks`).
///
/// Clears the whole cache when the render width changes so stale-width lines
/// are never reused. The cache lives on `TranscriptState` as `content_cache`
/// and `content_cache_width`; keeping it a free function lets the caller hold a
/// disjoint borrow of `self.blocks` while handing out a mutable borrow of the
/// cache.
pub(super) fn cached_block_lines(
    cache: &mut [Option<Arc<Vec<String>>>],
    cache_width: &mut u16,
    idx: usize,
    text: &str,
    w: usize,
) -> Arc<Vec<String>> {
    if usize::from(*cache_width) != w {
        for slot in cache.iter_mut() {
            *slot = None;
        }
        *cache_width = w as u16;
    }
    if let Some(Some(lines)) = cache.get(idx) {
        return Arc::clone(lines);
    }
    let wrapped = Arc::new(wrap_text(text, w));
    if idx < cache.len() {
        cache[idx] = Some(Arc::clone(&wrapped));
    }
    wrapped
}

pub(super) fn wrap_preformatted(text: &str, _width: usize) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        if line.is_empty() {
            out.push(String::new());
            continue;
        }
        out.push(line.to_string());
    }
    out
}

pub(super) fn parse_md_line(line: &str) -> Line<'static> {
    if line.starts_with("```") {
        return Line::from(Span::styled(
            line.to_string(),
            Style::default().fg(theme::MUTED),
        ));
    }
    let mut spans: Vec<Span> = Vec::new();
    let mut rest = line.to_string();
    while !rest.is_empty() {
        if let Some(pos) = rest.find("**") {
            if pos > 0 {
                spans.push(Span::styled(
                    rest[..pos].to_string(),
                    Style::default().fg(theme::TEXT),
                ));
            }
            rest = rest[pos + 2..].to_string();
            if let Some(end) = rest.find("**") {
                spans.push(Span::styled(
                    rest[..end].to_string(),
                    Style::default()
                        .fg(theme::TEXT)
                        .add_modifier(Modifier::BOLD),
                ));
                rest = rest[end + 2..].to_string();
            } else {
                spans.push(Span::raw("**"));
                break;
            }
        } else {
            spans.push(Span::styled(rest, Style::default().fg(theme::TEXT)));
            break;
        }
    }
    Line::from(spans)
}

/// Take the tail of `s` that fits within approximately `max_chars` characters.
/// Preserves line boundaries — never cuts mid-line at the start.
pub(super) fn tail_lines(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let lines: Vec<&str> = s.lines().collect();
    let mut result = Vec::new();
    let mut budget = max_chars;
    for line in lines.iter().rev() {
        let line_len = line.chars().count();
        if result.is_empty() {
            if line_len > budget {
                let start = line_len.saturating_sub(budget);
                result.push(line.chars().skip(start).collect::<String>());
                break;
            }
            budget = budget.saturating_sub(line_len + 1);
            result.push(line.to_string());
        } else if line_len < budget {
            budget = budget.saturating_sub(line_len + 1);
            result.push(line.to_string());
        } else {
            break;
        }
    }
    result.reverse();
    result.join(
        "
",
    )
}

pub(super) fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!(
            "{}…",
            s.chars().take(max.saturating_sub(1)).collect::<String>()
        )
    }
}

/// Compute the display width (columns) of a string, accounting for CJK and tab.
fn display_width(s: &str) -> usize {
    s.chars()
        .map(|c| {
            if c == '\t' {
                4
            } else {
                unicode_width::UnicodeWidthChar::width(c).unwrap_or(1)
            }
        })
        .sum()
}

/// Truncate `s` so that its display width fits within `max_cols`.
fn truncate_to_width(s: &str, max_cols: usize) -> String {
    if display_width(s) <= max_cols {
        return s.to_string();
    }
    let mut result = String::new();
    let mut w = 0usize;
    for c in s.chars() {
        let cw = if c == '\t' {
            4
        } else {
            unicode_width::UnicodeWidthChar::width(c).unwrap_or(1)
        };
        if w + cw > max_cols.saturating_sub(1) {
            result.push('…');
            break;
        }
        result.push(c);
        w += cw;
    }
    result
}

// ── Line counting ────────────────────────────────────────────────

pub(super) fn block_line_count(block: &DisplayBlock, width: usize) -> usize {
    let w = width.max(20);
    match block {
        DisplayBlock::User(content) => 2 + wrap_text(content, w).len() + 1,
        DisplayBlock::Assistant(content) => 2 + wrap_text(content, w).len() + 1,
        DisplayBlock::Thinking {
            content,
            expanded,
            duration_ms: _,
        } => {
            let all = wrap_text(content, w);
            let total = all.len();
            let preview = 3usize;
            let show = if *expanded || total <= preview {
                total
            } else {
                preview
            };
            let mut n = 1 + show + 1;
            if total > preview {
                n += 1;
            }
            n
        }
        DisplayBlock::ToolRunning {
            input,
            streamed_output,
            ..
        } => {
            let cmd = serde_json::from_str::<serde_json::Value>(input)
                .ok()
                .and_then(|v| v.get("command").and_then(|v| v.as_str().map(String::from)));
            let mut n = match cmd {
                Some(c) if !c.is_empty() => 1 + wrap_text(&c, w.saturating_sub(3)).len(),
                _ => 1,
            };
            if !streamed_output.is_empty() {
                let tail = tail_lines(streamed_output, 1000);
                n += wrap_text(&tail, w.saturating_sub(3)).len();
            }
            n
        }
        DisplayBlock::ApprovalPending(req) => {
            2 + wrap_text(&req.description, w).len()
                + 1
                + 1
                + wrap_preformatted_lines_count(&req.input, w)
                + 1
                + 1
        }
        DisplayBlock::ApprovalResolved { .. } => 2,
        DisplayBlock::ToolDone {
            full_output,
            expanded,
            input,
            ..
        } => {
            let all = wrap_text(full_output, w);
            let total = all.len();
            let preview = 3usize;
            let show = if *expanded || total <= preview {
                total
            } else {
                preview
            };
            let mut n = 1 + show;
            if total > preview {
                n += 1;
            }
            let cmd = serde_json::from_str::<serde_json::Value>(input)
                .ok()
                .and_then(|v| v.get("command").and_then(|v| v.as_str().map(String::from)))
                .unwrap_or_default();
            if !cmd.is_empty() {
                n += wrap_text(&cmd, w.saturating_sub(3)).len();
            }
            n + 1
        }
        DisplayBlock::System(s) => wrap_text(s, w).len(),
        DisplayBlock::Question(q) => {
            let mut n = 2 + wrap_text(&q.prompt, w).len() + 1 + q.options.len() + 2;
            if q.allow_custom {
                n += 1;
            }
            n
        }
        DisplayBlock::ErrorLine(_s) => 1, // truncated to width in render
        DisplayBlock::TurnInfo { .. } => 1,
    }
}

fn wrap_preformatted_lines_count(text: &str, width: usize) -> usize {
    if text.is_empty() {
        return 0;
    }
    let mut n = 0usize;
    for source_line in text.lines() {
        if source_line.is_empty() {
            n += 1;
        } else {
            let mut w = 0usize;
            let mut has_content = false;
            for ch in source_line.chars() {
                let cw = char_width(ch);
                if w + cw > width {
                    n += 1;
                    w = 0;
                }
                w += cw;
                has_content = true;
            }
            if has_content {
                n += 1;
            }
        }
    }
    n
}

// ── Column truncation helpers ─────────────────────────────────────

pub(super) fn truncate_by_columns(text: &str, max_cols: usize) -> String {
    let mut col = 0usize;
    let mut result = String::new();
    for c in text.chars() {
        if col >= max_cols {
            break;
        }
        let w = if c == '\t' { 4 } else { char_width(c) };
        result.push(c);
        col += w;
    }
    result
}

pub(super) fn truncate_by_columns_skip(text: &str, skip_cols: usize) -> String {
    let mut col = 0usize;
    let mut skipping = true;
    let mut result = String::new();
    for c in text.chars() {
        if skipping {
            let w = if c == '\t' { 4 } else { char_width(c) };
            col += w;
            if col > skip_cols {
                result.push(c);
                skipping = false;
            }
        } else {
            result.push(c);
        }
    }
    result
}

pub(super) fn plain_text_from_lines(
    lines: &[Line<'_>],
    sel_start: (usize, usize),
    sel_end: (usize, usize),
) -> String {
    let (sl, sc) = sel_start;
    let (el, ec) = sel_end;
    let s = sl.min(lines.len());
    let e = (el + 1).min(lines.len());
    let mut out = String::new();
    for (idx, line) in lines[s..e].iter().enumerate() {
        let global_line = s + idx;
        let full_text: String = line.spans.iter().map(|sp| sp.content.as_ref()).collect();
        if global_line == sl && global_line == el {
            let start_col = sc.min(ec);
            let end_col = sc.max(ec);
            let truncated = truncate_by_columns(&full_text, end_col);
            let remaining = truncate_by_columns_skip(&truncated, start_col);
            out.push_str(&remaining);
        } else if global_line == sl {
            out.push_str(&truncate_by_columns_skip(&full_text, sc));
        } else if global_line == el {
            out.push_str(&truncate_by_columns(&full_text, ec));
        } else {
            out.push_str(&full_text);
        }
        if idx < lines[s..e].len() - 1 {
            out.push('\n');
        }
    }
    out
}

// ── Selection highlight ──────────────────────────────────────────

pub(super) fn apply_selection_highlight(
    lines: Vec<Line<'static>>,
    line_offset: usize,
    sel: Option<((usize, usize), (usize, usize))>,
) -> Vec<Line<'static>> {
    let Some(sel) = sel else {
        return lines;
    };
    let ((sl, sc), (el, ec)) = sel;
    let (lo_line, lo_col, hi_line, hi_col) = if sl < el || (sl == el && sc <= ec) {
        (sl, sc, el, ec)
    } else {
        (el, ec, sl, sc)
    };

    lines
        .into_iter()
        .enumerate()
        .map(|(i, line)| {
            let global = line_offset + i;
            if global < lo_line || global > hi_line {
                return line;
            }
            let line_width: usize = line.spans.iter().map(|s| s.width()).sum();
            let sel_start_col = if global == lo_line {
                lo_col.min(line_width)
            } else {
                0
            };
            let sel_end_col = if global == hi_line {
                hi_col.min(line_width)
            } else {
                line_width
            };
            if sel_start_col >= sel_end_col {
                return line;
            }

            let mut highlighted_spans: Vec<Span<'static>> = Vec::new();
            let mut col_acc: usize = 0;
            for sp in line.spans {
                let span_width = sp.width();
                let span_start = col_acc;
                let span_end = col_acc + span_width;

                if span_end <= sel_start_col || span_start >= sel_end_col {
                    highlighted_spans.push(sp);
                } else if span_start >= sel_start_col && span_end <= sel_end_col {
                    highlighted_spans.push(Span::styled(
                        sp.content,
                        sp.style.bg(theme::USER).fg(Color::Black),
                    ));
                } else {
                    let content = sp.content.as_ref();
                    if span_start < sel_start_col && span_end > sel_end_col {
                        // Three-way split
                        let mut tmp_col = span_start;
                        let left_chars = content
                            .chars()
                            .take_while(|c| {
                                let w = char_width(*c);
                                if tmp_col < sel_start_col {
                                    tmp_col += w;
                                    true
                                } else {
                                    false
                                }
                            })
                            .count();
                        highlighted_spans.push(Span::styled(
                            content.chars().take(left_chars).collect::<String>(),
                            sp.style,
                        ));
                        let mid_chars = content
                            .chars()
                            .skip(left_chars)
                            .take_while(|c| {
                                let w = char_width(*c);
                                if tmp_col < sel_end_col {
                                    tmp_col += w;
                                    true
                                } else {
                                    false
                                }
                            })
                            .count();
                        let mid: String =
                            content.chars().skip(left_chars).take(mid_chars).collect();
                        if !mid.is_empty() {
                            highlighted_spans
                                .push(Span::styled(mid, sp.style.bg(theme::USER).fg(Color::Black)));
                        }
                        let right: String = content.chars().skip(left_chars + mid_chars).collect();
                        if !right.is_empty() {
                            highlighted_spans.push(Span::styled(right, sp.style));
                        }
                    } else if span_start < sel_start_col {
                        let mut tmp_col = span_start;
                        let left_chars = content
                            .chars()
                            .take_while(|c| {
                                let w = char_width(*c);
                                if tmp_col < sel_start_col {
                                    tmp_col += w;
                                    true
                                } else {
                                    false
                                }
                            })
                            .count();
                        highlighted_spans.push(Span::styled(
                            content.chars().take(left_chars).collect::<String>(),
                            sp.style,
                        ));
                        let remaining: String = content.chars().skip(left_chars).collect();
                        highlighted_spans.push(Span::styled(
                            remaining,
                            sp.style.bg(theme::USER).fg(Color::Black),
                        ));
                    } else {
                        let mut tmp_col = span_start;
                        let inside_chars = content
                            .chars()
                            .take_while(|c| {
                                let w = char_width(*c);
                                if tmp_col < sel_end_col {
                                    tmp_col += w;
                                    true
                                } else {
                                    false
                                }
                            })
                            .count();
                        let inside: String = content.chars().take(inside_chars).collect();
                        let outside: String = content.chars().skip(inside_chars).collect();
                        if !inside.is_empty() {
                            highlighted_spans.push(Span::styled(
                                inside,
                                sp.style.bg(theme::USER).fg(Color::Black),
                            ));
                        }
                        if !outside.is_empty() {
                            highlighted_spans.push(Span::styled(outside, sp.style));
                        }
                    }
                }
                col_acc += span_width;
            }
            Line::from(highlighted_spans)
        })
        .collect()
}

// ── emit_* functions (virtualized rendering) ─────────────────────

/// Emit lines for a single block, with skip/take virtualization.
///
/// `content_cache`/`content_cache_width` memoize the wrapped "primary text"
/// of User/Assistant/Thinking/ToolDone/System blocks so a long committed
/// block is not re-wrapped on every frame (see [`cached_block_lines`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_block_lines(
    block: &DisplayBlock,
    bi: usize,
    w: usize,
    content_cache: &mut [Option<Arc<Vec<String>>>],
    content_cache_width: &mut u16,
    lines: &mut Vec<Line<'static>>,
    hits: &mut Vec<LineAnswerHit>,
    skip: usize,
    max_lines: usize,
) {
    let mut emitted = 0usize;
    let mut skipped = 0usize;
    let mut push = |line: Line<'static>, hit: LineAnswerHit| {
        if skipped < skip {
            skipped += 1;
            return;
        }
        if emitted >= max_lines {
            return;
        }
        lines.push(line);
        hits.push(hit);
        emitted += 1;
    };
    match block {
        DisplayBlock::User(content) => {
            push(
                Line::from(vec![Span::styled(
                    " YOU ",
                    Style::default()
                        .fg(Color::Black)
                        .bg(theme::USER)
                        .add_modifier(Modifier::BOLD),
                )]),
                None,
            );
            push(Line::default(), None);
            let wrapped = cached_block_lines(content_cache, content_cache_width, bi, content, w);
            for tl in wrapped.iter() {
                push(
                    Line::from(Span::styled(tl.clone(), Style::default().fg(theme::TEXT))),
                    None,
                );
            }
            push(Line::default(), None);
        }
        DisplayBlock::Assistant(content) => {
            push(
                Line::from(vec![Span::styled(
                    " nca ",
                    Style::default()
                        .fg(Color::Black)
                        .bg(theme::ASSISTANT)
                        .add_modifier(Modifier::BOLD),
                )]),
                None,
            );
            push(Line::default(), None);
            let wrapped = cached_block_lines(content_cache, content_cache_width, bi, content, w);
            for tl in wrapped.iter() {
                push(parse_md_line(tl), None);
            }
            push(Line::default(), None);
        }
        DisplayBlock::ToolRunning {
            name,
            input,
            streamed_output,
            ..
        } => {
            let name_budget = w.saturating_sub(5); // " ⚡ " + " …"
            push(
                Line::from(vec![
                    Span::styled(" ⚡ ", Style::default().fg(theme::TOOL)),
                    Span::styled(
                        format!("{} ", truncate_to_width(name, name_budget)),
                        Style::default()
                            .fg(theme::TOOL)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled("…", Style::default().fg(theme::MUTED)),
                ]),
                None,
            );
            let cmd = serde_json::from_str::<serde_json::Value>(input)
                .ok()
                .and_then(|v| v.get("command").and_then(|v| v.as_str().map(String::from)))
                .unwrap_or_default();
            if !cmd.is_empty() {
                for tl in wrap_text(&cmd, w.saturating_sub(3)) {
                    push(
                        Line::from(Span::styled(
                            format!("   {tl}"),
                            Style::default().fg(theme::MUTED),
                        )),
                        None,
                    );
                }
            }
            if !streamed_output.is_empty() {
                let tail = tail_lines(streamed_output, 1000);
                for tl in wrap_text(&tail, w.saturating_sub(3)) {
                    push(
                        Line::from(Span::styled(
                            format!("   {tl}"),
                            Style::default().fg(theme::MUTED),
                        )),
                        None,
                    );
                }
            }
        }
        DisplayBlock::ApprovalPending(req) => {
            push(
                Line::from(vec![Span::styled(
                    " 🔒 ",
                    Style::default()
                        .fg(theme::WARN)
                        .add_modifier(Modifier::BOLD),
                )]),
                None,
            );
            push(Line::default(), None);
            for tl in wrap_text(&req.description, w) {
                push(
                    Line::from(Span::styled(tl, Style::default().fg(theme::TEXT))),
                    None,
                );
            }
            push(Line::default(), None);
            push(
                Line::from(Span::styled("  input:", Style::default().fg(theme::MUTED))),
                None,
            );
            for tl in wrap_preformatted(&req.input, w) {
                push(
                    Line::from(Span::styled(
                        format!("    {tl}"),
                        Style::default().fg(theme::MUTED),
                    )),
                    None,
                );
            }
            push(Line::default(), None);
            push(
                Line::from(vec![
                    Span::styled("  y/yes ", Style::default().fg(theme::SUCCESS)),
                    Span::styled("approve · ", Style::default().fg(theme::MUTED)),
                    Span::styled("n/no ", Style::default().fg(theme::ERROR)),
                    Span::styled("deny", Style::default().fg(theme::MUTED)),
                ]),
                None,
            );
            push(Line::default(), None);
        }
        DisplayBlock::ApprovalResolved { tool, approved } => {
            let (label, style) = if *approved {
                (
                    " approved ",
                    Style::default().fg(Color::Black).bg(theme::SUCCESS),
                )
            } else {
                (
                    " denied ",
                    Style::default().fg(Color::Black).bg(theme::ERROR),
                )
            };
            let label_width = display_width(label);
            let tool_budget = w.saturating_sub(label_width + 1);
            push(
                Line::from(vec![
                    Span::styled(label, style.add_modifier(Modifier::BOLD)),
                    Span::styled(
                        format!(" {}", truncate_to_width(tool, tool_budget)),
                        Style::default().fg(theme::TEXT),
                    ),
                ]),
                None,
            );
            push(Line::default(), None);
        }
        DisplayBlock::ToolDone {
            name,
            input,
            ok: _,
            detail,
            full_output,
            expanded,
            duration_ms,
        } => {
            let icon = "✓";
            let st = Style::default().fg(theme::SUCCESS);
            let all_l = cached_block_lines(content_cache, content_cache_width, bi, full_output, w);
            let total = all_l.len();
            let preview = 3usize;
            let is_exp = *expanded;
            let show = if is_exp || total <= preview {
                total
            } else {
                preview
            };
            let prefix_width = 3 + name.chars().count() + 3; // " icon " + name + " — "
            let detail_budget = w.saturating_sub(prefix_width);
            push(
                Line::from(vec![
                    Span::styled(format!(" {icon} "), st),
                    Span::styled(
                        name.to_string(),
                        Style::default()
                            .fg(theme::TOOL)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!(" — {}", truncate_to_width(detail, detail_budget)),
                        Style::default().fg(theme::MUTED),
                    ),
                    Span::styled(
                        format!(" · {}", format_duration(*duration_ms)),
                        Style::default().fg(theme::MUTED),
                    ),
                ]),
                None,
            );
            let cmd = serde_json::from_str::<serde_json::Value>(input)
                .ok()
                .and_then(|v| v.get("command").and_then(|v| v.as_str().map(String::from)))
                .unwrap_or_default();
            if !cmd.is_empty() {
                for tl in wrap_text(&cmd, w.saturating_sub(3)) {
                    push(
                        Line::from(Span::styled(
                            format!("   {tl}"),
                            Style::default().fg(theme::MUTED),
                        )),
                        None,
                    );
                }
            }
            for tl in &all_l[..show] {
                push(
                    Line::from(Span::styled(tl.clone(), Style::default().fg(theme::MUTED))),
                    None,
                );
            }
            if total > preview {
                let label: String = if is_exp {
                    format!(" ▾ hide {name} output ")
                } else {
                    format!(" ▸ show {name} output ({}/{}) ", show, total)
                };
                push(
                    Line::from(vec![
                        Span::styled(label, Style::default().fg(theme::TOOL)),
                        Span::styled("(click)", Style::default().fg(theme::MUTED)),
                    ]),
                    Some(TranscriptHit::ToggleToolOutput(bi)),
                );
            }
            push(Line::default(), None);
        }
        DisplayBlock::System(s) => {
            let wrapped = cached_block_lines(content_cache, content_cache_width, bi, s, w);
            for text_line in wrapped.iter() {
                push(
                    Line::from(Span::styled(
                        text_line.clone(),
                        Style::default().fg(theme::WARN),
                    )),
                    None,
                );
            }
        }
        DisplayBlock::Question(q) => {
            push(
                Line::from(vec![
                    Span::styled(
                        " ? ",
                        Style::default().fg(Color::Black).bg(theme::WARN).bold(),
                    ),
                    Span::styled(
                        " question ",
                        Style::default()
                            .fg(theme::WARN)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]),
                None,
            );
            push(Line::default(), None);
            for tl in wrap_text(&q.prompt, w) {
                push(
                    Line::from(Span::styled(tl, Style::default().fg(theme::TEXT))),
                    None,
                );
            }
            push(
                Line::from(vec![Span::styled(
                    {
                        let prefix = "  [0] suggested: ".to_string();
                        let suffix = " (click)".to_string();
                        let budget =
                            w.saturating_sub(display_width(&prefix) + display_width(&suffix));
                        format!(
                            "{}{} {} ",
                            prefix,
                            truncate_to_width(&q.suggested_answer, budget),
                            suffix
                        )
                    },
                    Style::default()
                        .fg(theme::SUCCESS)
                        .add_modifier(Modifier::UNDERLINED),
                )]),
                Some(TranscriptHit::Question(
                    nca_common::event::QuestionSelection::Suggested,
                )),
            );
            for (i, o) in q.options.iter().enumerate() {
                push(
                    Line::from(vec![Span::styled(
                        {
                            let prefix = format!("  [{}] ({}) ", i + 1, o.id);
                            let suffix = " (click)".to_string();
                            let budget =
                                w.saturating_sub(display_width(&prefix) + display_width(&suffix));
                            format!(
                                "{}{} {} ",
                                prefix,
                                truncate_to_width(&o.label, budget),
                                suffix
                            )
                        },
                        Style::default()
                            .fg(theme::TEXT)
                            .add_modifier(Modifier::UNDERLINED),
                    )]),
                    Some(TranscriptHit::Question(
                        nca_common::event::QuestionSelection::Option {
                            option_id: o.id.clone(),
                        },
                    )),
                );
            }
            if q.allow_custom {
                push(
                    Line::from(Span::styled(
                        truncate_to_width("  [c] type your own answer below, then Enter", w),
                        Style::default().fg(theme::MUTED),
                    )),
                    None,
                );
            }
            push(
                Line::from(Span::styled(
                    truncate_to_width(
                        "  Tip: /auto-answer or Enter on empty = suggested · click an option above",
                        w,
                    ),
                    Style::default().fg(theme::MUTED),
                )),
                None,
            );
            push(Line::default(), None);
        }
        DisplayBlock::Thinking {
            content,
            expanded,
            duration_ms,
        } => {
            let all_l = cached_block_lines(content_cache, content_cache_width, bi, content, w);
            let total = all_l.len();
            let is_exp = *expanded;
            let preview = 3usize;
            let show = if is_exp || total <= preview {
                total
            } else {
                preview
            };
            let mut title_spans = vec![Span::styled(
                " 💭 thinking ",
                Style::default().fg(theme::MUTED),
            )];
            if let Some(ms) = duration_ms {
                title_spans.push(Span::styled(
                    format!(" · {}", format_duration(*ms)),
                    Style::default().fg(theme::MUTED),
                ));
            }
            push(Line::from(title_spans), None);
            for tl in &all_l[..show] {
                push(
                    Line::from(Span::styled(tl.clone(), Style::default().fg(theme::MUTED))),
                    None,
                );
            }
            if total > preview {
                let label: String = if is_exp {
                    " ▾ hide thinking ".into()
                } else {
                    format!(" ▸ show thinking ({}/{}) ", show, total)
                };
                push(
                    Line::from(vec![
                        Span::styled(label, Style::default().fg(theme::TOOL)),
                        Span::styled("(click)", Style::default().fg(theme::MUTED)),
                    ]),
                    Some(TranscriptHit::ToggleThinking(bi)),
                );
            }
            push(Line::default(), None);
        }
        DisplayBlock::ErrorLine(s) => {
            let budget = w.saturating_sub(3); // " ✗ "
            push(
                Line::from(Span::styled(
                    format!(" ✗ {}", truncate_to_width(s, budget)),
                    Style::default().fg(theme::ERROR),
                )),
                None,
            );
        }
        DisplayBlock::TurnInfo { duration_ms } => {
            push(
                Line::from(Span::styled(
                    format!("⏱ turn completed · {}", format_duration(*duration_ms)),
                    Style::default().fg(theme::MUTED),
                )),
                None,
            );
        }
    }
}

/// Virtualized streaming reasoning lines.
///
/// `wrap` is the precomputed incremental wrap cache for the reasoning buffer
/// (see [`StreamingWrap`]). Only lines inside the requested `[skip,
/// skip+max_lines)` window are materialized, so a long reasoning stream no
/// longer re-wraps or re-spans the whole buffer on every frame.
pub(super) fn emit_streaming_reasoning_lines(
    wrap: &StreamingWrap,
    expanded: bool,
    started_at: Option<Instant>,
    lines: &mut Vec<Line<'static>>,
    hits: &mut Vec<LineAnswerHit>,
    skip: usize,
    max_lines: usize,
) {
    let total_rl = wrap.len();
    let preview_rl = 5usize;
    let show_rl = if expanded || total_rl <= preview_rl {
        total_rl
    } else {
        preview_rl
    };
    let has_toggle = total_rl > preview_rl;
    // Logical layout: [title] + [show_rl content lines] + [toggle?] + [blank].
    let n_logical = 1 + show_rl + usize::from(has_toggle) + 1;
    let start = skip.min(n_logical);
    let end = skip.saturating_add(max_lines).min(n_logical);
    for idx in start..end {
        if idx == 0 {
            let mut title = vec![
                Span::styled(" 💭 thinking ", Style::default().fg(theme::MUTED)),
                Span::styled("…", Style::default().fg(theme::MUTED)),
            ];
            // Live elapsed timer while thinking; redrawn at the busy animation cadence.
            if let Some(since) = started_at {
                let elapsed_ms = since.elapsed().as_millis() as u64;
                title.push(Span::styled(
                    format!(" · {}", format_duration(elapsed_ms)),
                    Style::default().fg(theme::MUTED),
                ));
            }
            lines.push(Line::from(title));
            hits.push(None);
        } else if idx <= show_rl {
            let rl = wrap.line(idx - 1);
            lines.push(Line::from(Span::styled(
                rl.to_string(),
                Style::default().fg(theme::MUTED),
            )));
            hits.push(None);
        } else if has_toggle && idx == 1 + show_rl {
            let label: String = if expanded {
                " ▾ hide thinking ".into()
            } else {
                format!(" ▸ show thinking ({}/{}) ", show_rl, total_rl)
            };
            lines.push(Line::from(vec![
                Span::styled(label, Style::default().fg(theme::TOOL)),
                Span::styled("(click)", Style::default().fg(theme::MUTED)),
            ]));
            hits.push(Some(TranscriptHit::ToggleStreamingThinking));
        } else {
            lines.push(Line::default());
            hits.push(None);
        }
    }
}

/// Virtualized streaming assistant lines.
///
/// `wrap` is the precomputed incremental wrap cache for the assistant buffer;
/// only the requested `[skip, skip+max_lines)` window is materialized (and only
/// those lines go through `parse_md_line`).
pub(super) fn emit_streaming_assistant_lines(
    wrap: &StreamingWrap,
    lines: &mut Vec<Line<'static>>,
    hits: &mut Vec<LineAnswerHit>,
    skip: usize,
    max_lines: usize,
) {
    // Logical layout: [" nca " header] + [blank] + [content lines].
    // Mirrors the original early return: an empty streaming buffer emits no
    // assistant lines at all.
    if wrap.is_empty() {
        return;
    }
    let total = wrap.len();
    let n_logical = 2 + total;
    let start = skip.min(n_logical);
    let end = skip.saturating_add(max_lines).min(n_logical);
    for idx in start..end {
        if idx == 0 {
            lines.push(Line::from(vec![Span::styled(
                " nca ",
                Style::default()
                    .fg(Color::Black)
                    .bg(theme::ASSISTANT)
                    .add_modifier(Modifier::BOLD),
            )]));
            hits.push(None);
        } else if idx == 1 {
            lines.push(Line::default());
            hits.push(None);
        } else {
            lines.push(parse_md_line(wrap.line(idx - 2)));
            hits.push(None);
        }
    }
}

/// Virtualized empty-state fallback.
pub(super) fn emit_empty_fallback_lines(
    lines: &mut Vec<Line<'static>>,
    hits: &mut Vec<LineAnswerHit>,
    skip: usize,
    max_lines: usize,
) {
    let mut emitted = 0usize;
    let mut skipped = 0usize;
    let mut push = |line: Line<'static>, hit: LineAnswerHit| {
        if skipped < skip {
            skipped += 1;
            return;
        }
        if emitted >= max_lines {
            return;
        }
        lines.push(line);
        hits.push(hit);
        emitted += 1;
    };
    push(
        Line::from(vec![
            Span::styled(
                "nca",
                Style::default()
                    .fg(theme::ASSISTANT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" — session ready", Style::default().fg(theme::MUTED)),
        ]),
        None,
    );
    push(Line::default(), None);
    push(
        Line::from(Span::styled(
            "Tab  agent   Ctrl+V  image   Ctrl+P  commands   !cmd  shell   @path  search   /  inline   PgUp/Dn  scroll\n\n                drag to select  ·  release to copy to clipboard",
            Style::default().fg(theme::MUTED),
        )),
        None,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // `wrap_paragraph` is the extracted per-paragraph core of `wrap_text`.
    // These tests pin that the extraction preserved the historical semantics:
    // wrapping paragraph-by-paragraph and concatenating must equal the
    // whole-string wrap, and the empty/whitespace fallbacks must be unchanged.
    #[test]
    fn wrap_text_equals_concatenated_wrap_paragraph() {
        let curated = [
            String::new(),
            "\n".to_string(),
            "\n\n".to_string(),
            "a\nb\nc".to_string(),
            "e".repeat(20),  // word exactly `width` wide
            "e".repeat(21),  // one over → hard split
            "中".repeat(40), // long CJK run, no spaces
            "no newline here".to_string(),
            "trailing\n".to_string(),
            "\nleading".to_string(),
            "mixed 中文 and ascii words here".to_string(),
            "word ".repeat(20).trim_end().to_string(),
        ];
        for s in &curated {
            for &w in &[20usize, 21, 30, 78] {
                let expected: Vec<String> =
                    s.split('\n').flat_map(|p| wrap_paragraph(p, w)).collect();
                assert_eq!(wrap_text(s, w), expected, "s={s:?} w={w}");
            }
        }
    }

    #[test]
    fn wrap_paragraph_handles_named_boundaries() {
        // Empty paragraph → exactly one empty line.
        assert_eq!(wrap_paragraph("", 20), vec![String::new()]);
        // Whitespace-only paragraph → no lines (the caller applies the
        // verbatim fallback).
        assert_eq!(wrap_paragraph("   ", 20), Vec::<String>::new());
        // A word landing exactly on the boundary is not split.
        assert_eq!(wrap_paragraph(&"e".repeat(20), 20), vec!["e".repeat(20)]);
        // One over the boundary hard-splits into two lines.
        assert_eq!(
            wrap_paragraph(&"e".repeat(21), 20),
            vec!["e".repeat(20), "e".to_string()]
        );
    }

    #[test]
    fn wrap_text_keeps_empty_and_whitespace_fallbacks() {
        assert_eq!(wrap_text("", 80), vec![String::new()]);
        // All-whitespace (non-empty) → verbatim single line, since every
        // paragraph wraps to nothing.
        assert_eq!(wrap_text("   ", 80), vec!["   ".to_string()]);
        assert_eq!(wrap_text(" \n ", 80), vec![" \n ".to_string()]);
        // Sub-8 width bypasses wrapping entirely.
        assert_eq!(wrap_text("abc\ndef", 4), vec!["abc\ndef".to_string()]);
    }

    #[test]
    fn cached_block_lines_memoizes_and_resets_on_width_change() {
        let mut cache = vec![None, None];
        let mut cache_width = 0u16;

        let a = cached_block_lines(&mut cache, &mut cache_width, 0, "hello world", 78);
        assert_eq!(*a, wrap_text("hello world", 78));
        assert_eq!(cache_width, 78);
        let _b = cached_block_lines(&mut cache, &mut cache_width, 1, "second block", 78);
        assert!(cache[0].is_some() && cache[1].is_some());

        // Same width → memoized (identical Arc, no re-wrap).
        let again = cached_block_lines(&mut cache, &mut cache_width, 0, "hello world", 78);
        assert!(
            Arc::ptr_eq(&a, &again),
            "same width must reuse the memoized lines"
        );

        // Width change → invalidate every slot, then re-wrap the requested one.
        let narrow = cached_block_lines(&mut cache, &mut cache_width, 0, "hello world", 20);
        assert_eq!(*narrow, wrap_text("hello world", 20));
        assert_eq!(cache_width, 20);
        assert!(
            !Arc::ptr_eq(&a, &narrow),
            "a width change must not reuse stale lines"
        );
        assert!(
            cache[1].is_none(),
            "a width change must invalidate every slot"
        );
    }
}
