//! Shared text formatting utilities used across TUI components.

use std::borrow::Cow;

use serde_json::Value;

/// Truncate a string to at most `max` characters (Unicode-aware), trimming whitespace.
/// Appends `"…"` when truncated.
pub fn truncate(s: &str, max: usize) -> String {
    let t = s.trim();
    if t.chars().count() <= max {
        t.to_string()
    } else {
        format!(
            "{}…",
            t.chars().take(max.saturating_sub(1)).collect::<String>()
        )
    }
}

/// Return the first 8 characters of a session id (or the full id if shorter).
pub(crate) fn short_session_prefix(id: &str) -> &str {
    if id.len() > 8 { &id[..8] } else { id }
}

/// Remove SGR-1006 mouse-tracking residue (`[<btn;col;rowM` / `...m`) that
/// leaks into the input buffer as plain chars after an escape-sequence
/// desync (the terminal's mouse-report bytes overflow the tty input queue,
/// crossterm fails to re-sync on the partial sequence, and downgrades the
/// remaining bytes to plain `Char` key events). Handles multiple occurrences,
/// including back-to-back ones. Incomplete trailing fragments (e.g. `[<35;72;2`)
/// are deliberately left in place — they are stripped on a later call once the
/// terminating `M`/`m` byte arrives.
///
/// Returns a borrowed `Cow` when nothing was stripped (the overwhelmingly
/// common case), so the per-keystroke call site avoids an allocation.
pub(crate) fn strip_sgr_mouse_residue(s: &str) -> Cow<'_, str> {
    let bytes = s.as_bytes();
    // Lazily-built output; `copied_upto` tracks how much of `s` has already
    // been pushed into it (everything before the last stripped match).
    let mut out: Option<String> = None;
    let mut copied_upto = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if let Some(end) = match_sgr_residue_at(bytes, i) {
            let out = out.get_or_insert_with(|| String::with_capacity(s.len()));
            out.push_str(&s[copied_upto..i]);
            copied_upto = end;
            i = end;
        } else {
            // Byte-wise advance is safe: match positions always start at an
            // ASCII `[`, and multi-byte UTF-8 sequences never contain it.
            i += 1;
        }
    }
    match out {
        Some(mut out) => {
            out.push_str(&s[copied_upto..]);
            Cow::Owned(out)
        }
        None => Cow::Borrowed(s),
    }
}

/// Match an SGR-1006 mouse residue sequence starting at byte `i`:
/// literal `[<`, then three groups of 1-3 ASCII digits separated by `;`,
/// terminated by `M` (press) or `m` (release). Returns the exclusive end
/// byte index of the match, or `None`.
fn match_sgr_residue_at(bytes: &[u8], i: usize) -> Option<usize> {
    if bytes.get(i).copied() != Some(b'[') || bytes.get(i + 1).copied() != Some(b'<') {
        return None;
    }
    let mut j = i + 2;
    for group in 0..3 {
        let start = j;
        while j - start < 3 && bytes.get(j).is_some_and(|b| b.is_ascii_digit()) {
            j += 1;
        }
        if j == start {
            return None; // each group needs at least one digit
        }
        if group < 2 && bytes.get(j).copied() != Some(b';') {
            return None;
        }
        if group < 2 {
            j += 1; // consume the `;`
        }
    }
    match bytes.get(j).copied() {
        Some(b'M') | Some(b'm') => Some(j + 1),
        _ => None,
    }
}

/// Format tool input for display in the transcript.
/// Special-cases `spawn_subagent` for a compact multi-line summary.
pub(crate) fn format_tool_input_for_display(tool: &str, value: &Value) -> String {
    if tool == "spawn_subagent" {
        format_spawn_subagent_input(value)
    } else {
        format_tool_input(value)
    }
}

fn format_spawn_subagent_input(v: &Value) -> String {
    let task = v.get("task").and_then(|t| t.as_str()).unwrap_or("").trim();
    let wt = v
        .get("use_worktree")
        .and_then(|b| b.as_bool())
        .unwrap_or(true);
    let n_focus = v
        .get("focus_files")
        .and_then(|a| a.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    format!(
        "task:\n{}\nworktree: {} · focus_files: {}",
        truncate(task, 500),
        wt,
        n_focus
    )
}

fn format_tool_input(value: &Value) -> String {
    if let Some(raw) = value.as_str()
        && let Ok(parsed) = serde_json::from_str::<Value>(raw)
    {
        return serde_json::to_string_pretty(&parsed).unwrap_or_else(|_| raw.to_string());
    }
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    #[test]
    fn strip_sgr_mouse_residue_table() {
        // Each case: (input, expected output).
        let cases: &[(&str, &str)] = &[
            // Full press/release sequences are stripped completely.
            ("[<35;72;23M", ""),
            ("[<0;1;1M[<0;1;1m", ""),
            // Residue embedded in real text is removed, surrounding text kept.
            ("keep[<35;72;23Mthis", "keepthis"),
            // Incomplete trailing fragment is deliberately left in place —
            // it is stripped on a later call once the terminating M/m arrives.
            ("partial[<35;72;2", "partial[<35;72;2"),
            ("[<0;0;0M", ""),
            // A 4-digit group is NOT an SGR-1006 sequence → left unchanged.
            ("[<1234;1;1M stays", "[<1234;1;1M stays"),
            // Clean input round-trips unchanged.
            ("clean input", "clean input"),
        ];
        for (input, expected) in cases {
            let out = strip_sgr_mouse_residue(input);
            assert_eq!(
                out.as_ref(),
                *expected,
                "input: {input:?} expected: {expected:?}"
            );
        }

        // Borrowing discipline: clean input is returned borrowed (no
        // allocation on the per-keystroke hot path), stripped input is owned.
        let borrowed = strip_sgr_mouse_residue("clean input");
        assert!(matches!(borrowed, Cow::Borrowed(_)));
        let owned = strip_sgr_mouse_residue("[<35;72;23M");
        assert!(matches!(owned, Cow::Owned(_)));
    }
}
