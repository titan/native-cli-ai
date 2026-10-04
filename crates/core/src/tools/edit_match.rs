//! Tiered matcher shared by the string-edit tools (`apply_patch`, `edit_file`).
//!
//! Ported from oh-my-opencode-slim `src/hooks/apply-patch/matching.ts`
//! (`seekMatch` + `rescueByPrefixSuffix`), adapted to nca's byte-range
//! contract:
//!
//! * Level 1 (`Exact`) is a raw substring scan — byte-for-byte identical to
//!   the tools' pre-matcher behaviour, including mid-line matches. A single
//!   hit reproduces the old `content.find(old_text)` range exactly; multiple
//!   hits are ambiguous (fail-closed).
//! * Levels 2–4 (`TrimEnd`, `Trim`, `UnicodeTrim`) are line-window scans:
//!   `old_text` is split into lines and consecutive file lines are compared
//!   after per-level normalization. Each level scans the whole file; a level
//!   only runs when every stricter level missed everywhere, so an early
//!   precise hit always beats a later fuzzy one. More than one hit at the
//!   winning level is [`MatchOutcome::Ambiguous`] — the matcher never
//!   guesses.
//! * Trailing-blank retry: when `old_text` ends on an empty line and every
//!   level missed, the empty line is dropped and the fuzzy ladder runs once
//!   more (omos `resolution.ts` trailing-blank retry — a patch may carry a
//!   final blank line the file does not have).
//! * Rescue: when all levels (and the retry) miss, the first and last lines
//!   of `old_text` are used as anchors, compared at the unicode + trim-end
//!   level. Full trim is deliberately excluded there (omos: stale patches
//!   must not bind across indentation depths). The drifted middle between
//!   the anchors is bounded to `2 * old_lines + 4` lines; a second anchored
//!   candidate pair is ambiguous.
//!
//! Every outcome is a byte range over the *original* content. Replacement is
//! performed by the caller on those raw bytes, so everything outside the
//! matched range stays byte-identical (no normalization is ever written
//! back). Line splitting treats both `"\n"` and `"\r\n"` as terminators, so
//! line-ending drift within a file resolves at the `TrimEnd` level while
//! unrelated terminators survive untouched.

use std::borrow::Cow;
use std::ops::Range;

/// Strictness-ordered matching levels. `Exact` is a raw substring scan; the
/// fuzzy levels are whole-line window comparisons after normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatchLevel {
    /// Raw substring match (the tools' original behaviour).
    Exact,
    /// Trailing whitespace ignored on every line.
    TrimEnd,
    /// Leading and trailing whitespace ignored on every line.
    Trim,
    /// Whitespace-trimmed lines with lookalike Unicode punctuation folded to
    /// ASCII (smart quotes, dashes, NBSP, ellipsis).
    UnicodeTrim,
}

impl MatchLevel {
    /// Stable lowercase name used in tool output messages.
    pub(crate) fn label(self) -> &'static str {
        match self {
            MatchLevel::Exact => "exact",
            MatchLevel::TrimEnd => "trim_end",
            MatchLevel::Trim => "trim",
            MatchLevel::UnicodeTrim => "unicode_trim",
        }
    }
}

/// Result of seeking `old_text` inside file content. All ranges are byte
/// ranges into the original content string.
#[derive(Debug, PartialEq)]
pub(crate) enum MatchOutcome {
    /// Exactly one candidate matched at the given level.
    Unique {
        byte_range: Range<usize>,
        level: MatchLevel,
    },
    /// The prefix/suffix rescue anchored a drifted region (bounded middle).
    Rescue { byte_range: Range<usize> },
    /// Nothing matched at any level, nor the rescue.
    NotFound,
    /// More than one candidate matched — fail-closed, never guess. `level` is
    /// `None` when the ambiguity came from the anchored rescue (two candidate
    /// regions) rather than a level scan.
    Ambiguous {
        level: Option<MatchLevel>,
        candidates: usize,
    },
}

/// Seek `old_text` in `content` using the full ladder:
/// exact substring → trim-end → trim → unicode-trim line windows →
/// trailing-blank retry → anchored prefix/suffix rescue. See the module docs
/// for the level semantics. An empty `old_text` is defensively `NotFound`
/// (both call sites reject it before reaching here).
pub(crate) fn seek_match(content: &str, old_text: &str) -> MatchOutcome {
    if old_text.is_empty() {
        return MatchOutcome::NotFound;
    }

    // Level 1 — Exact: raw substring scan. This is the fast path and is
    // byte-identical to the tools' pre-matcher behaviour (mid-line hits
    // included). Counting every occurrence keeps the historical
    // "matched N times (ambiguous)" contract.
    let mut first_hit: Option<usize> = None;
    let mut exact_count = 0_usize;
    for (offset, _) in content.match_indices(old_text) {
        exact_count += 1;
        if first_hit.is_none() {
            first_hit = Some(offset);
        }
    }
    if exact_count == 1
        && let Some(start) = first_hit
    {
        return MatchOutcome::Unique {
            byte_range: start..start + old_text.len(),
            level: MatchLevel::Exact,
        };
    }
    if exact_count > 1 {
        return MatchOutcome::Ambiguous {
            level: Some(MatchLevel::Exact),
            candidates: exact_count,
        };
    }

    // Levels 2–4 — fuzzy line windows over the whole file.
    let content_spans = split_line_spans(content);
    let mut pattern_spans = split_line_spans(old_text);
    if let Some(outcome) = fuzzy_ladder(content, &content_spans, old_text, &pattern_spans) {
        return outcome;
    }

    // Trailing-blank retry: `old_text` ending on an empty line may carry a
    // blank the file does not have. Drop it and rerun the ladder once.
    if pattern_spans.len() >= 2 && {
        let last = &pattern_spans[pattern_spans.len() - 1];
        last.content_start == last.content_end
    } {
        pattern_spans.pop();
        if let Some(outcome) = fuzzy_ladder(content, &content_spans, old_text, &pattern_spans) {
            return outcome;
        }
    }

    // All levels missed — bounded anchored rescue.
    rescue_by_prefix_suffix(content, &content_spans, old_text, &pattern_spans)
}

/// Run the three fuzzy line-window levels (whole file each). Returns
/// `Some(Unique | Ambiguous)` when a level hit, `None` when every level
/// missed and the caller should continue (retry / rescue).
fn fuzzy_ladder(
    content: &str,
    content_spans: &[LineSpan],
    old_text: &str,
    pattern_spans: &[LineSpan],
) -> Option<MatchOutcome> {
    let n = content_spans.len();
    let m = pattern_spans.len();
    if m == 0 || n < m {
        return None;
    }
    let include_final_terminator = old_text.ends_with('\n');

    for level in [
        MatchLevel::TrimEnd,
        MatchLevel::Trim,
        MatchLevel::UnicodeTrim,
    ] {
        let pattern: Vec<String> = pattern_spans
            .iter()
            .map(|span| {
                normalize_for_level(&old_text[span.content_start..span.content_end], level)
                    .into_owned()
            })
            .collect();
        let mut norms = LazyLineNorms::new(content, content_spans, level);
        let mut hits: Vec<usize> = Vec::new();
        for start in 0..=(n - m) {
            let mut matched = true;
            for (k, pattern_line) in pattern.iter().enumerate() {
                if norms.get(start + k) != pattern_line.as_str() {
                    matched = false;
                    break;
                }
            }
            if matched {
                hits.push(start);
            }
        }
        if hits.len() == 1 {
            return Some(MatchOutcome::Unique {
                byte_range: window_byte_range(content_spans, hits[0], m, include_final_terminator),
                level,
            });
        }
        if hits.len() > 1 {
            return Some(MatchOutcome::Ambiguous {
                level: Some(level),
                candidates: hits.len(),
            });
        }
    }
    None
}

/// Anchored rescue for a fully drifted `old_text`: match the first and last
/// pattern lines (unicode + trim-end normalized — full trim is deliberately
/// excluded so stale patches cannot bind across indentation depths) and
/// accept the region when the drifted middle between the anchors is at most
/// `2 * old_lines + 4` lines. Two anchored candidate pairs are ambiguous.
fn rescue_by_prefix_suffix(
    content: &str,
    content_spans: &[LineSpan],
    old_text: &str,
    pattern_spans: &[LineSpan],
) -> MatchOutcome {
    let m = pattern_spans.len();
    if m == 0 || content_spans.is_empty() {
        return MatchOutcome::NotFound;
    }
    let pattern_line =
        |i: usize| &old_text[pattern_spans[i].content_start..pattern_spans[i].content_end];
    let anchor_norm = |line: &str| normalize_unicode(line.trim_end()).into_owned();
    let left = anchor_norm(pattern_line(0));
    let right = anchor_norm(pattern_line(m - 1));
    let max_middle = 2 * m + 4;

    // The rescue only runs after every level missed, so normalizing the whole
    // file here is the rare path — no laziness needed.
    let normed: Vec<String> = content_spans
        .iter()
        .map(|span| normalize_unicode(&content[span.content_start..span.content_end]).into_owned())
        .collect();
    let n = normed.len();

    // Scan anchor pairs: a left-anchor line followed (within the bounded
    // middle) by a right-anchor line. The first pair wins; a second valid
    // pair is ambiguous (fail-closed, matching omos).
    let mut found: Option<(usize, usize)> = None; // (first line, last line) of the footprint
    for left_index in 0..n {
        if normed[left_index] != left {
            continue;
        }
        let from = left_index + 1;
        for (right_index, line) in normed.iter().enumerate().take(n).skip(from) {
            if *line != right {
                continue;
            }
            let middle = right_index - from;
            if middle > max_middle {
                // Anchored scan is monotonic: every later right anchor is
                // even further away (omos `continue` over sorted hits).
                break;
            }
            if found.is_some() {
                return MatchOutcome::Ambiguous {
                    level: None,
                    candidates: 2,
                };
            }
            found = Some((left_index, right_index));
        }
    }

    let Some((first_line, last_line)) = found else {
        return MatchOutcome::NotFound;
    };
    let start = content_spans[first_line].content_start;
    let last = &content_spans[last_line];
    let end = if last_line + 1 < content_spans.len() {
        // Interior region: consume the final line's terminator so the line
        // after the region survives on its own line.
        content_spans[last_line + 1].content_start
    } else if old_text.ends_with('\n') {
        last.term_end
    } else {
        last.content_end
    };
    MatchOutcome::Rescue {
        byte_range: start..end,
    }
}

/// Byte range covering `count` consecutive lines starting at `start`: from
/// the first line's first content byte through the last line's last content
/// byte, plus the last line's terminator when the pattern itself ends with a
/// line terminator. Interior terminators are always inside the range.
fn window_byte_range(
    spans: &[LineSpan],
    start: usize,
    count: usize,
    include_final_terminator: bool,
) -> Range<usize> {
    let first = &spans[start];
    let last = &spans[start + count - 1];
    let end = if include_final_terminator {
        last.term_end
    } else {
        last.content_end
    };
    first.content_start..end
}

/// One source line: content byte range plus its terminator (`"\n"`, `"\r\n"`,
/// or none at end of file). Terminators never enter comparisons; they only
/// matter when mapping a matched window back to a byte range.
struct LineSpan {
    content_start: usize,
    content_end: usize,
    term_end: usize,
}

/// Split `s` into line spans. `"\n"` terminates a line; a `"\r"` immediately
/// before it is part of the terminator (CRLF), so line-ending drift within a
/// file never leaks `"\r"` into comparisons. An empty string yields no lines.
fn split_line_spans(s: &str) -> Vec<LineSpan> {
    let bytes = s.as_bytes();
    let mut spans = Vec::new();
    let mut start = 0_usize;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'\n' {
            let content_end = if i > start && bytes[i - 1] == b'\r' {
                i - 1
            } else {
                i
            };
            spans.push(LineSpan {
                content_start: start,
                content_end,
                term_end: i + 1,
            });
            start = i + 1;
        }
    }
    if start < s.len() {
        spans.push(LineSpan {
            content_start: start,
            content_end: s.len(),
            term_end: s.len(),
        });
    }
    spans
}

/// Per-line normalization cache: lines are normalized lazily, only when a
/// candidate window actually touches them (omos 25ea8ad2 — never
/// pre-normalize the whole file for the common exact-match path).
struct LazyLineNorms<'a> {
    source: &'a str,
    spans: &'a [LineSpan],
    level: MatchLevel,
    cache: Vec<Option<Cow<'a, str>>>,
}

impl<'a> LazyLineNorms<'a> {
    fn new(source: &'a str, spans: &'a [LineSpan], level: MatchLevel) -> Self {
        Self {
            source,
            spans,
            level,
            cache: vec![None; spans.len()],
        }
    }

    /// Normalized content of line `idx`, computed on first use.
    fn get(&mut self, idx: usize) -> &str {
        if self.cache[idx].is_none() {
            let span = &self.spans[idx];
            let raw = &self.source[span.content_start..span.content_end];
            self.cache[idx] = Some(normalize_for_level(raw, self.level));
        }
        self.cache[idx].as_deref().unwrap_or("")
    }
}

/// Normalize one line at the given level.
fn normalize_for_level<'a>(line: &'a str, level: MatchLevel) -> Cow<'a, str> {
    match level {
        MatchLevel::Exact => Cow::Borrowed(line),
        MatchLevel::TrimEnd => Cow::Borrowed(line.trim_end()),
        MatchLevel::Trim => Cow::Borrowed(line.trim()),
        MatchLevel::UnicodeTrim => normalize_unicode(line.trim()),
    }
}

/// omos `normalizeUnicode`: fold lookalike Unicode punctuation (NBSP, dashes,
/// smart quotes, ellipsis) onto their ASCII lookalikes. Char-level folding —
/// a dependency-free approximation of the omos map, applied only at the
/// loosest level and never written back to disk.
fn normalize_unicode(line: &str) -> Cow<'_, str> {
    if !line.chars().any(|c| map_unicode_char(c).is_some()) {
        return Cow::Borrowed(line);
    }
    let mut out = String::with_capacity(line.len());
    for c in line.chars() {
        match map_unicode_char(c) {
            Some(replacement) => out.push_str(replacement),
            None => out.push(c),
        }
    }
    Cow::Owned(out)
}

/// Single-character lookalike map (omos `UNICODE_MAP`).
fn map_unicode_char(c: char) -> Option<&'static str> {
    match c {
        '\u{00A0}' => Some(" "),               // no-break space
        '\u{2010}'..='\u{2015}' => Some("-"),  // hyphens and dashes
        '\u{2018}'..='\u{201B}' => Some("'"),  // single quotes
        '\u{201C}'..='\u{201F}' => Some("\""), // double quotes
        '\u{2026}' => Some("..."),             // ellipsis
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique(content: &str, old_text: &str) -> (Range<usize>, MatchLevel) {
        match seek_match(content, old_text) {
            MatchOutcome::Unique { byte_range, level } => (byte_range, level),
            other => panic!("expected unique match, got {other:?}"),
        }
    }

    #[test]
    fn exact_substring_single_hit() {
        // Byte-for-byte the old behaviour, including the exact substring range.
        let (range, level) = unique("a\nb\nc\n", "b\n");
        assert_eq!(range, 2..4);
        assert_eq!(level, MatchLevel::Exact);
    }

    #[test]
    fn exact_mid_line_hit_is_kept() {
        let (range, level) = unique("#!/bin/sh\necho hi\n", "hi");
        assert_eq!(range, 15..17);
        assert_eq!(level, MatchLevel::Exact);
    }

    #[test]
    fn exact_substring_multi_hit_is_ambiguous() {
        assert_eq!(
            seek_match("alpha\nalpha\n", "alpha"),
            MatchOutcome::Ambiguous {
                level: Some(MatchLevel::Exact),
                candidates: 2
            }
        );
    }

    #[test]
    fn trailing_whitespace_drift_hits_trim_end() {
        let content = "let x = 1;   \nlet y = 2;\n";
        let old_text = "let x = 1;\n";
        // Exact substring is absent (";" is followed by spaces, not "\n").
        let (range, level) = unique(content, old_text);
        assert_eq!(level, MatchLevel::TrimEnd);
        // The range covers the whole original line content plus terminator.
        assert_eq!(range, 0..14);
        assert_eq!(&content[range], "let x = 1;   \n");
    }

    #[test]
    fn indentation_drift_hits_trim() {
        let content = "fn a() {}\nfn b() {}\n";
        let old_text = "    fn b() {}\n";
        let (range, level) = unique(content, old_text);
        assert_eq!(level, MatchLevel::Trim);
        assert_eq!(range, 10..20);
        assert_eq!(&content[range], "fn b() {}\n");
    }

    #[test]
    fn unicode_punctuation_drift_hits_unicode_trim() {
        let content = "let s = \u{201c}hi\u{201d};\n";
        let old_text = "let s = \"hi\";\n";
        let (range, level) = unique(content, old_text);
        assert_eq!(level, MatchLevel::UnicodeTrim);
        assert_eq!(range, 0..content.len());
    }

    #[test]
    fn crlf_drift_within_a_file_hits_trim_end() {
        // Mixed line endings: "b" is terminated by CRLF, the pattern by LF.
        let content = "a\nb\r\nc\n";
        let old_text = "b\n";
        let (range, level) = unique(content, old_text);
        assert_eq!(level, MatchLevel::TrimEnd);
        assert_eq!(range, 2..5);
        assert_eq!(&content[range], "b\r\n");
    }

    #[test]
    fn crlf_file_with_lf_pattern_hits_trim_end() {
        let content = "alpha\r\nbeta\r\n";
        let old_text = "alpha\nbeta\n";
        let (range, level) = unique(content, old_text);
        assert_eq!(level, MatchLevel::TrimEnd);
        assert_eq!(range, 0..13);
        assert_eq!(&content[range], "alpha\r\nbeta\r\n");
    }

    #[test]
    fn eof_blank_line_retry_drops_trailing_blank() {
        // "c\n\n" is absent; the ladder misses on ["c", ""]; the retry on
        // ["c"] matches at trim_end and consumes the line terminator.
        let (range, level) = unique("a\nb\nc\n", "c\n\n");
        assert_eq!(level, MatchLevel::TrimEnd);
        assert_eq!(range, 4..6);
        assert_eq!(&"a\nb\nc\n"[range], "c\n");
    }

    #[test]
    fn eof_blank_line_retry_never_overrides_exact() {
        // When the blank line IS present the exact substring wins first.
        let (_, level) = unique("c\n\n", "c\n\n");
        assert_eq!(level, MatchLevel::Exact);
    }

    #[test]
    fn fuzzy_multi_hit_is_ambiguous_fail_closed() {
        let content = "foo  \nbar\nfoo\nbar  \n";
        let old_text = "foo\nbar\n";
        assert_eq!(
            seek_match(content, old_text),
            MatchOutcome::Ambiguous {
                level: Some(MatchLevel::TrimEnd),
                candidates: 2
            }
        );
    }

    #[test]
    fn rescue_anchors_drifted_middle() {
        let content = "fn main() {\n    let x = 1;\n    println!(\"hello\");\n}\n";
        let old_text = "fn main() {\n    let y = 2;\n    something_else();\n}";
        match seek_match(content, old_text) {
            MatchOutcome::Rescue { byte_range } => {
                // Footprint spans first anchor line through last anchor line;
                // the file's final terminator is preserved (pattern has none).
                assert_eq!(byte_range, 0..content.len() - 1);
            }
            other => panic!("expected rescue, got {other:?}"),
        }
    }

    #[test]
    fn rescue_rejects_out_of_bound_middle() {
        // 12 drifted lines between the anchors exceeds 2m+4 = 10.
        let mut content = String::from("START\n");
        for i in 0..12 {
            content.push_str(&format!("junk {i}\n"));
        }
        content.push_str("END\n");
        let old_text = "START\nmiddle\nEND";
        assert_eq!(seek_match(&content, old_text), MatchOutcome::NotFound);
    }

    #[test]
    fn rescue_with_two_regions_is_ambiguous() {
        let content = "A\nQ\nB\nA\nR\nB\n";
        let old_text = "A\nX\nB";
        assert_eq!(
            seek_match(content, old_text),
            MatchOutcome::Ambiguous {
                level: None,
                candidates: 2
            }
        );
    }

    #[test]
    fn nothing_matches_is_not_found() {
        assert_eq!(seek_match("a\nb\n", "zzz"), MatchOutcome::NotFound);
    }

    #[test]
    fn pattern_longer_than_file_is_not_found() {
        assert_eq!(seek_match("a\n", "a\nb\nc\n"), MatchOutcome::NotFound);
    }

    #[test]
    fn empty_old_text_is_defensively_not_found() {
        assert_eq!(seek_match("abc", ""), MatchOutcome::NotFound);
    }

    #[test]
    fn line_splitting_treats_crlf_as_terminator() {
        let spans = split_line_spans("a\r\n\r\nb");
        let lines: Vec<&str> = spans
            .iter()
            .map(|s| &"a\r\n\r\nb"[s.content_start..s.content_end])
            .collect();
        assert_eq!(lines, vec!["a", "", "b"]);
        assert_eq!(spans[0].term_end - spans[0].content_end, 2); // CRLF
        assert_eq!(spans[1].term_end - spans[1].content_end, 2); // CRLF
        assert_eq!(spans[2].term_end - spans[2].content_end, 0); // EOF, none
    }
}
