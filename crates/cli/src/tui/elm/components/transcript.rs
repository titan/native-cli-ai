//! Transcript component — renders DisplayBlock items with virtual scrolling,
//! text selection, streaming text, and collapsible blocks.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent, MouseButton, MouseEvent, MouseEventKind};
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph};
use unicode_width::UnicodeWidthChar;

use nca_common::event::{AgentEvent, InteractiveQuestionPayload, QuestionSelection};
use nca_common::tool::ToolResult;

use crate::tui::state::{ApprovalRequest, DisplayBlock};
use crate::tui::text_utils::{format_tool_input_for_display, short_session_prefix, truncate};

// ── Theme (from shared module, re-exported via searchable_list) ───
use super::searchable_list::theme;

use super::transcript_render::{
    apply_selection_highlight, block_line_count, emit_block_lines, emit_empty_fallback_lines,
    emit_streaming_assistant_lines, emit_streaming_reasoning_lines, plain_text_from_lines,
    wrap_paragraph,
};

const MOUSE_SCROLL_LINES: usize = 6;

// ── Public types ──────────────────────────────────────────────────

/// Actions triggered by clicking transcript lines.
#[derive(Debug, Clone)]
pub(crate) enum TranscriptHit {
    Question(QuestionSelection),
    ToggleThinking(usize),
    ToggleStreamingThinking,
    ToggleToolOutput(usize),
}

/// External side-effects from transcript interactions.
#[derive(Debug)]
pub(crate) enum TranscriptAction {
    /// No side-effect.
    None,
    /// User answered a question by clicking an option.
    QuestionAnswer(QuestionSelection),
    /// Copy text to clipboard + push System message about it.
    CopyToClipboard(String),
    /// Push a System message into blocks.
    PushSystem(String),
    /// Push an ErrorLine message into blocks.
    PushError(String),
}

/// Per flattened transcript line: click selects this answer or toggles thinking.
pub(crate) type LineAnswerHit = Option<TranscriptHit>;

// ── BlockLineCache ─────────────────────────────────────────────────

/// Cached per-block line counts + cumulative offsets for fast virtualization.
/// Keyed by (blocks_generation, width) so it auto-invalidates. The cache is
/// also maintained incrementally by `TranscriptState` (append/set one block's
/// height at a time); a full `rebuild` only happens when validity is lost
/// (width change, stale generation, or block-count mismatch).
pub(crate) struct BlockLineCache {
    generation: u64,
    width: u16,
    /// `heights[i]` = line count of `blocks[i]`.  Same length as `blocks`.
    heights: Vec<usize>,
    /// `cum_offsets[i]` = total lines of blocks[0..i].  Length = heights.len() + 1.
    cum_offsets: Vec<usize>,
    /// Count of full `rebuild()` calls (test-only metric; the
    /// incremental-cache invariant tests pin "no rebuild" behavior with it).
    #[cfg(test)]
    rebuild_count: u32,
}

impl BlockLineCache {
    fn new() -> Self {
        Self {
            generation: 0,
            width: 0,
            heights: Vec::new(),
            cum_offsets: vec![0],
            #[cfg(test)]
            rebuild_count: 0,
        }
    }

    /// Returns `true` when the cache is valid for the given generation, width,
    /// and current number of blocks. The length check hardens incremental
    /// maintenance: any mutation path that bypassed the helpers (or a
    /// bulk-extend) makes the cache invalid instead of silently misindexing.
    fn is_valid(&self, g: u64, w: u16, n_blocks: usize) -> bool {
        self.generation == g
            && self.width == w
            && !self.heights.is_empty()
            && self.heights.len() == n_blocks
    }

    /// Rebuild from `blocks`.  Must be called when `is_valid` returns `false`.
    fn rebuild(&mut self, blocks: &[DisplayBlock], g: u64, w: u16) {
        self.generation = g;
        self.width = w;
        #[cfg(test)]
        {
            self.rebuild_count += 1;
        }
        self.heights.clear();
        self.cum_offsets.clear();
        let w_usize = w as usize;
        for b in blocks {
            self.heights.push(block_line_count(b, w_usize));
        }
        self.cum_offsets.reserve(self.heights.len() + 1);
        let mut acc = 0usize;
        self.cum_offsets.push(acc);
        for &h in &self.heights {
            acc += h;
            self.cum_offsets.push(acc);
        }
    }

    /// Total committed-block line count (excluding streaming).
    #[inline]
    fn total(&self) -> usize {
        *self.cum_offsets.last().unwrap_or(&0)
    }

    /// Append the measured height of a newly pushed block.
    ///
    /// Only correct when the cache already covered every prior block
    /// (`is_valid` held before the push). O(1): pushes onto `heights` and
    /// extends `cum_offsets` with `last + h`.
    fn append_height(&mut self, h: usize) {
        self.heights.push(h);
        self.cum_offsets.push(self.total() + h);
    }

    /// Replace one block's height, shifting all subsequent cumulative offsets
    /// by the delta. Integer bookkeeping only — no text re-wrapping of any
    /// other block.
    fn set_height(&mut self, idx: usize, h: usize) {
        if idx >= self.heights.len() {
            return;
        }
        let delta = h as isize - self.heights[idx] as isize;
        self.heights[idx] = h;
        if delta != 0 {
            for off in &mut self.cum_offsets[idx + 1..] {
                *off = off.wrapping_add_signed(delta);
            }
        }
    }

    /// Reset to the empty (never-built) state. Used by `TranscriptState::clear`.
    fn reset(&mut self) {
        self.generation = 0;
        self.width = 0;
        self.heights.clear();
        self.cum_offsets.clear();
        self.cum_offsets.push(0);
    }

    /// Re-sync the cached generation after an incremental maintenance step so
    /// the cache stays valid across the mutation instead of being discarded.
    fn sync_generation(&mut self, g: u64) {
        self.generation = g;
    }
}

// ── StreamingWrap ────────────────────────────────────────────────

/// Incremental word-wrap cache for a streaming text buffer.
///
/// Streaming buffers grow one delta at a time (`TokensStreamed` /
/// `ReasoningStreamed`). Re-wrapping the whole buffer on every delta — once
/// for the line count, again for the visible slice, again for emission — is
/// O(buffer) per frame and starves the input loop in long sessions. This cache
/// wraps only the paragraphs a delta *completes* and re-wraps just the trailing
/// (unterminated) paragraph, so per-delta work is O(delta) and per-frame reads
/// are O(1).
///
/// `len()`/`line(i)` mirror `wrap_text(&text, width)` exactly, including the
/// all-whitespace fallback and the empty-string single empty line.
pub(crate) struct StreamingWrap {
    /// Width the cached lines were wrapped at.
    width: usize,
    /// Raw text accumulated so far; kept in lockstep with the owning streaming
    /// buffer so a width change (resize) can rebuild exactly.
    text: String,
    /// Wrapped lines of every *completed* paragraph (terminated by `\n`).
    finalized: Vec<String>,
    /// Trailing paragraph not yet terminated by `\n`.
    active: String,
    /// `wrap_paragraph(&active, width)`, recomputed on each append.
    active_wrapped: Vec<String>,
}

impl StreamingWrap {
    /// An empty cache (mirrors `wrap_text("", width)` → one empty line).
    pub(crate) fn new() -> Self {
        Self {
            width: 0,
            text: String::new(),
            finalized: Vec::new(),
            active: String::new(),
            active_wrapped: vec![String::new()],
        }
    }

    /// Total wrapped line count.
    pub(crate) fn len(&self) -> usize {
        let n = self.finalized.len() + self.active_wrapped.len();
        if n == 0 && !self.text.is_empty() {
            1
        } else {
            n
        }
    }

    /// Whether the owning streaming buffer is currently empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Wrapped line `i`. Caller must ensure `i < len()`.
    pub(crate) fn line(&self, i: usize) -> &str {
        if self.finalized.is_empty() && self.active_wrapped.is_empty() && !self.text.is_empty() {
            return &self.text;
        }
        if i < self.finalized.len() {
            &self.finalized[i]
        } else {
            &self.active_wrapped[i - self.finalized.len()]
        }
    }

    /// Forget all cached state. Used when the owning streaming buffer is
    /// cleared (`None`).
    pub(crate) fn reset(&mut self) {
        self.width = 0;
        self.text.clear();
        self.finalized.clear();
        self.active.clear();
        self.active_wrapped.clear();
        self.active_wrapped.push(String::new());
    }

    /// Replace the cached raw text without wrapping it; forces a lazy rebuild
    /// on the next read/append. Used when the owning buffer is set externally.
    pub(crate) fn set_text(&mut self, text: &str) {
        self.width = 0;
        self.text.clear();
        self.text.push_str(text);
        self.finalized.clear();
        self.active.clear();
        self.active_wrapped.clear();
        self.active_wrapped.push(String::new());
    }

    /// Rebuild the whole cache from `text` at width `w` (rare: resize or the
    /// first read after an external buffer replacement).
    pub(crate) fn rebuild_from(&mut self, text: &str, w: usize) {
        self.width = w;
        self.text.clear();
        self.text.push_str(text);
        self.finalized.clear();
        self.active.clear();
        self.active_wrapped.clear();
        let segments: Vec<&str> = text.split('\n').collect();
        let (last, complete) = segments
            .split_last()
            .expect("split always yields at least one segment");
        for seg in complete {
            self.finalized.extend(wrap_paragraph(seg, w));
        }
        self.active.push_str(last);
        self.active_wrapped = wrap_paragraph(last, w);
    }

    /// Ensure the cache is wrapped at width `w`, rebuilding from `text` on a
    /// width change. `text` must be the owning buffer's current contents.
    pub(crate) fn ensure(&mut self, text: &str, w: usize) {
        if self.width != w {
            self.rebuild_from(text, w);
        }
    }

    /// Extend the cache with `delta`, wrapping at `w`. Only the paragraphs the
    /// delta completes, plus the trailing partial paragraph, are (re-)wrapped;
    /// a width change falls back to a full rebuild.
    pub(crate) fn append(&mut self, delta: &str, w: usize) {
        self.text.push_str(delta);
        if w != self.width {
            let text = std::mem::take(&mut self.text);
            self.rebuild_from(&text, w);
            return;
        }
        let segments: Vec<&str> = delta.split('\n').collect();
        let (last, complete) = segments
            .split_last()
            .expect("split always yields at least one segment");
        for seg in complete {
            self.active.push_str(seg);
            self.finalized.extend(wrap_paragraph(&self.active, w));
            self.active.clear();
        }
        self.active.push_str(last);
        self.active_wrapped = wrap_paragraph(&self.active, w);
    }
}

/// Bound a `ToolRunning` block's streamed-output buffer.
///
/// `streamed_output` is a pure display buffer (the authoritative full output
/// arrives via `ToolCallCompleted`), so it is safe to keep only the tail. Once
/// it exceeds 8 KiB it is trimmed to roughly the last 4 KiB, snapped forward to
/// a char boundary so no UTF-8 sequence is split.
fn cap_streamed_output(s: &mut String) {
    const CAP: usize = 8192;
    const KEEP: usize = 4096;
    if s.len() <= CAP {
        return;
    }
    let mut start = s.len() - KEEP;
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    s.drain(..start);
}

// ── TranscriptState ──────────────────────────────────────────────

pub(crate) struct TranscriptState {
    // ── Transcript data ──
    pub(crate) blocks: Vec<DisplayBlock>,
    pub(crate) streaming_assistant: Option<String>,
    pub(crate) streaming_reasoning: Option<String>,
    pub(crate) streaming_reasoning_expanded: bool,
    pub(crate) blocks_generation: u64,

    // ── Scroll state ──
    pub(crate) scroll_lines: usize,
    pub(crate) transcript_follow_tail: bool,

    // ── Text selection ──
    pub(crate) transcript_selection: Option<((usize, usize), (usize, usize))>, // ((line, col), (line, col))
    pub(crate) transcript_dragging: bool,
    pub(crate) transcript_drag_anchor: Option<(usize, usize)>,

    // ── Cache ──
    pub(crate) line_cache: BlockLineCache,
    pub(crate) last_visible_hits: Vec<LineAnswerHit>,
    /// Width the line cache was last (re)built/maintained at. Event-time
    /// incremental maintenance (`blocks_pushed`/`block_mutated_at`) measures
    /// at this width so appended heights stay consistent with the cache; a
    /// width change still invalidates via `is_valid` → full rebuild fallback.
    pub(crate) last_width: u16,

    /// child_session_id → index of that child's rolling activity block.
    /// Lets `ChildSessionActivity` update one line in place instead of pushing
    /// a new block per event (which forced O(transcript) cache rebuilds during
    /// parallel subagent runs).
    pub(crate) child_activity_blocks: HashMap<String, usize>,

    /// Index of the rolling compaction-bracket block plus the token count
    /// captured at `ContextCompactionStart`. Lets `ContextCompactionEnd`
    /// replace the in-progress line in place instead of pushing a second
    /// block per bracket (cache stays incremental).
    pub(crate) compaction_block: Option<(usize, usize)>,

    // ── Active question for answer routing ──
    pub(crate) _active_question: Option<InteractiveQuestionPayload>,

    // ── UI-layer timer for reasoning/thinking blocks ──
    pub(crate) reasoning_started_at: Option<Instant>,

    // ── Incremental streaming wrap caches ──
    /// Incremental wrap cache for `streaming_assistant`, so the whole stream is
    /// not re-wrapped on every frame.
    streaming_assistant_wrap: StreamingWrap,
    /// Incremental wrap cache for `streaming_reasoning`.
    streaming_reasoning_wrap: StreamingWrap,

    // ── Committed-block content cache ──
    /// Per-block memoized wrap of the block's primary text (User/Assistant
    /// content, Thinking content, ToolDone full_output, System text), indexed to
    /// match `blocks`. `None` means invalid or wrapped at a stale width.
    content_cache: Vec<Option<Arc<Vec<String>>>>,
    /// Width `content_cache` was last valid at; a render-width change refills it.
    content_cache_width: u16,

    /// call_id → index of the in-flight `ToolRunning` block, so
    /// `ToolOutputChunk` can locate its block in O(1) instead of a reverse scan
    /// of `blocks` on every chunk.
    tool_running_blocks: HashMap<String, usize>,
}

impl TranscriptState {
    pub(crate) fn new() -> Self {
        Self {
            blocks: Vec::new(),
            streaming_assistant: None,
            streaming_reasoning: None,
            streaming_reasoning_expanded: false,
            blocks_generation: 0,
            scroll_lines: 0,
            transcript_follow_tail: true,
            transcript_selection: None,
            transcript_dragging: false,
            transcript_drag_anchor: None,
            line_cache: BlockLineCache::new(),
            last_visible_hits: Vec::new(),
            last_width: 0,
            child_activity_blocks: HashMap::new(),
            compaction_block: None,
            _active_question: None,
            reasoning_started_at: None,
            streaming_assistant_wrap: StreamingWrap::new(),
            streaming_reasoning_wrap: StreamingWrap::new(),
            content_cache: Vec::new(),
            content_cache_width: 0,
            tool_running_blocks: HashMap::new(),
        }
    }

    // ── Event handling ──────────────────────────────────────────

    /// Process an AgentEvent and return any side-effect actions.
    pub(crate) fn apply_event(&mut self, e: &AgentEvent) -> TranscriptAction {
        match e {
            AgentEvent::SessionStarted { .. } => {
                // Model/session/branch are StatusBar concerns. No `blocks`
                // change, so skip the cache-invalidation bump below — it would
                // force an O(transcript) line-height rebuild for nothing.
                return TranscriptAction::None;
            }
            AgentEvent::MessageReceived {
                role,
                content,
                steering,
            } => {
                if role == "user" {
                    self.streaming_assistant = None;
                    self.streaming_assistant_wrap.reset();
                    // Steering messages get a small dim-style prefix marker so
                    // they are visually distinct from the turn's initial prompt.
                    let content = if *steering {
                        format!("⤳ {content}")
                    } else {
                        content.clone()
                    };
                    self.blocks.push(DisplayBlock::User(content));
                    self.blocks_pushed();
                } else if role == "assistant" {
                    self.streaming_assistant = None;
                    self.streaming_assistant_wrap.reset();
                    // Commit any accumulated reasoning before the assistant text.
                    let reasoning = self.streaming_reasoning.take();
                    self.streaming_reasoning_wrap.reset();
                    if let Some(reasoning) = reasoning
                        && !reasoning.trim().is_empty()
                    {
                        let duration_ms = self
                            .reasoning_started_at
                            .take()
                            .map(|t| t.elapsed().as_millis() as u64);
                        self.blocks.push(DisplayBlock::Thinking {
                            content: reasoning,
                            expanded: false,
                            duration_ms,
                        });
                        self.blocks_pushed();
                    }
                    self.blocks.push(DisplayBlock::Assistant(content.clone()));
                    self.blocks_pushed();
                }
            }
            AgentEvent::TokensStreamed { delta } => {
                self.streaming_assistant
                    .get_or_insert_with(String::new)
                    .push_str(delta);
                // Mirror the delta into the incremental wrap cache. Streaming
                // text is measured from the cache (`streaming_assistant_line_
                // count`), so committed blocks/cached heights are unaffected.
                // Returning here skips the generation bump — without this,
                // every token during streaming rebuilds the entire
                // committed-blocks line-height cache at O(total transcript
                // text), the dominant per-frame cost in long sessions that
                // starves the input loop.
                self.streaming_assistant_wrap
                    .append(delta, usize::from(self.last_width.max(20)));
                return TranscriptAction::None;
            }
            AgentEvent::ReasoningStreamed { delta } => {
                let is_start = self.streaming_reasoning.is_none();
                let s = self.streaming_reasoning.get_or_insert_with(String::new);
                s.push_str(delta);
                if is_start {
                    self.reasoning_started_at = Some(Instant::now());
                }
                self.streaming_reasoning_wrap
                    .append(delta, usize::from(self.last_width.max(20)));
                // See TokensStreamed: reasoning is measured outside the cache.
                return TranscriptAction::None;
            }
            AgentEvent::ToolCallStarted {
                call_id,
                tool,
                input,
            } => {
                self.flush_stream_before_tool();
                self.blocks.push(DisplayBlock::ToolRunning {
                    name: tool.clone(),
                    call_id: call_id.clone(),
                    input: format_tool_input_for_display(tool, input),
                    streamed_output: String::new(),
                });
                self.blocks_pushed();
                self.tool_running_blocks
                    .insert(call_id.clone(), self.blocks.len() - 1);
            }
            AgentEvent::ToolOutputChunk { call_id, delta } => {
                // O(1) lookup via the call_id → block-index map; fall back to a
                // reverse scan only if the map is missing the entry.
                let idx = self
                    .tool_running_blocks
                    .get(call_id)
                    .copied()
                    .filter(|&i| {
                        matches!(
                            self.blocks.get(i),
                            Some(DisplayBlock::ToolRunning { call_id: id, .. }) if id == call_id
                        )
                    })
                    .or_else(|| {
                        self.blocks.iter().rposition(|b| {
                            matches!(b, DisplayBlock::ToolRunning { call_id: id, .. } if id == call_id)
                        })
                    });
                if let Some(i) = idx {
                    if let Some(DisplayBlock::ToolRunning {
                        streamed_output, ..
                    }) = self.blocks.get_mut(i)
                    {
                        streamed_output.push_str(delta);
                        cap_streamed_output(streamed_output);
                    }
                    // Re-measure just this block and patch its cached height in
                    // place (no generation bump → the cache stays incremental).
                    // `cap_streamed_output` bounds this re-measure's cost.
                    if self.line_cache.is_valid(
                        self.blocks_generation,
                        self.last_width,
                        self.blocks.len(),
                    ) {
                        let h = block_line_count(&self.blocks[i], usize::from(self.last_width));
                        self.line_cache.set_height(i, h);
                    }
                }
                // CRITICAL: do NOT bump blocks_generation or return a
                // cache-invalidating action. Streaming chunks are
                // high-frequency; rebuilding the line-height cache per
                // chunk would starve the input loop (same rationale as
                // TokensStreamed above).
                return TranscriptAction::None;
            }
            AgentEvent::ToolCallCompleted {
                call_id,
                output,
                duration_ms,
            } => {
                let ok = output.success;
                // On failure, prefer the explicit error string; fall back to the
                // tool's stdout/stderr output (e.g. run_validation leaves error
                // as None and puts diagnostics in output).
                let full_output = if ok {
                    output.output.clone()
                } else {
                    output
                        .error
                        .clone()
                        .filter(|e| !e.is_empty())
                        .unwrap_or_else(|| output.output.clone())
                };
                let detail = truncate(&full_output, 120);
                if let Some(idx) = self.blocks.iter().rposition(|b| {
                    matches!(b, DisplayBlock::ToolRunning { call_id: id, .. } if id == call_id)
                        || matches!(b, DisplayBlock::ApprovalPending(req) if req.call_id == *call_id)
                }) {
                    let name = match &self.blocks[idx] {
                        DisplayBlock::ToolRunning { name, .. } => name.clone(),
                        DisplayBlock::ApprovalPending(req) => req.tool.clone(),
                        _ => "?".into(),
                    };
                    let input = match &self.blocks[idx] {
                        DisplayBlock::ToolRunning { input, .. } => input.clone(),
                        _ => String::new(),
                    };
                    self.blocks[idx] = DisplayBlock::ToolDone {
                        name,
                        input,
                        ok,
                        detail,
                        full_output,
                        expanded: false,
                        duration_ms: *duration_ms,
                    };
                    self.block_mutated_at(idx);
                } else {
                    self.blocks.push(DisplayBlock::ToolDone {
                        name: "?".into(),
                        input: String::new(),
                        ok,
                        detail,
                        full_output,
                        expanded: false,
                        duration_ms: *duration_ms,
                    });
                    self.blocks_pushed();
                }
                self.tool_running_blocks.remove(call_id);
            }
            AgentEvent::ApprovalRequested {
                call_id,
                tool,
                description,
            } => {
                let input = self
                    .blocks
                    .iter()
                    .rev()
                    .find_map(|block| match block {
                        DisplayBlock::ToolRunning {
                            call_id: id, input, ..
                        } if id == call_id => Some(input.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| "{}".into());
                let req = ApprovalRequest {
                    call_id: call_id.clone(),
                    tool: tool.clone(),
                    description: description.clone(),
                    input,
                };
                if let Some(idx) = self.blocks.iter().rposition(
                    |b| matches!(b, DisplayBlock::ToolRunning { call_id: id, .. } if id == call_id),
                ) {
                    self.blocks[idx] = DisplayBlock::ApprovalPending(req);
                    self.block_mutated_at(idx);
                } else {
                    self.blocks.push(DisplayBlock::ApprovalPending(req));
                    self.blocks_pushed();
                }
            }
            AgentEvent::ApprovalResolved {
                call_id,
                approved,
                allow_pattern: _,
            } => {
                // Replace the matching ApprovalPending block in-place so the
                // stale prompt ("y/yes approve · n/no deny") disappears.
                if let Some(idx) = self.blocks.iter().rposition(|block| {
                    matches!(
                        block,
                        DisplayBlock::ApprovalPending(req) if req.call_id == *call_id
                    )
                }) {
                    let tool = match &self.blocks[idx] {
                        DisplayBlock::ApprovalPending(req) => req.tool.clone(),
                        _ => "tool".into(),
                    };
                    self.blocks[idx] = DisplayBlock::ApprovalResolved {
                        tool,
                        approved: *approved,
                    };
                    self.block_mutated_at(idx);
                } else {
                    self.blocks.push(DisplayBlock::ApprovalResolved {
                        tool: "tool".into(),
                        approved: *approved,
                    });
                    self.blocks_pushed();
                }
            }
            AgentEvent::QuestionRequested { question } => {
                self.blocks.push(DisplayBlock::Question(question.clone()));
                self.blocks_pushed();
                self.transcript_follow_tail = true;
            }
            AgentEvent::QuestionResolved {
                question_id,
                selection,
            } => {
                self.blocks.push(DisplayBlock::System(format!(
                    "Answered question {question_id}: {selection:?}"
                )));
                self.blocks_pushed();
            }
            AgentEvent::Error { message } => {
                self.blocks.push(DisplayBlock::ErrorLine(message.clone()));
                self.blocks_pushed();
            }
            AgentEvent::ChildSessionSpawned {
                child_session_id,
                task,
                ..
            } => {
                let short = short_session_prefix(child_session_id);
                self.blocks.push(DisplayBlock::System(format!(
                    "Sub-agent {short}… — {}",
                    truncate(task, 80)
                )));
                let idx = self.blocks.len() - 1;
                self.blocks_pushed();
                self.child_activity_blocks
                    .insert(child_session_id.clone(), idx);
            }
            AgentEvent::ChildSessionActivity {
                child_session_id,
                phase,
                detail,
            } => {
                // Aggregate into ONE rolling block per child instead of pushing
                // a new block per activity event. Each pushed block used to
                // invalidate the whole line-height cache (O(total transcript)
                // re-wrap); with parallel subagents these events arrive in
                // bursts and starved the Elm input poll.
                let short = short_session_prefix(child_session_id);
                let d = truncate(detail, 120);
                let text = format!("↳ {short}… · {phase} · {d}");
                let marker = format!("↳ {short}… ·");
                if let Some(&idx) = self.child_activity_blocks.get(child_session_id)
                    && idx < self.blocks.len()
                    && matches!(&self.blocks[idx], DisplayBlock::System(s) if s.starts_with(&marker))
                {
                    // Still our rolling block → replace in place.
                    self.blocks[idx] = DisplayBlock::System(text);
                    self.block_mutated_at(idx);
                } else {
                    // First activity for this child (the map points at the
                    // "Sub-agent …" spawn block) or the rolling block was
                    // replaced → push a fresh one and (re)register it.
                    self.blocks.push(DisplayBlock::System(text));
                    let idx = self.blocks.len() - 1;
                    self.blocks_pushed();
                    self.child_activity_blocks
                        .insert(child_session_id.clone(), idx);
                }
            }
            AgentEvent::ChildSessionCompleted {
                child_session_id,
                status,
                ..
            } => {
                let short = short_session_prefix(child_session_id);
                self.blocks.push(DisplayBlock::System(format!(
                    "Sub-agent {short}… done: {status}"
                )));
                self.blocks_pushed();
                self.child_activity_blocks.remove(child_session_id);
            }
            AgentEvent::TurnCompleted { duration_ms, .. } => {
                self.blocks.push(DisplayBlock::TurnInfo {
                    duration_ms: *duration_ms,
                });
                self.blocks_pushed();
            }
            AgentEvent::ProviderFallback { from, to, reason } => {
                // Failover is never silent: one dim system line naming both
                // providers and the failure class.
                let reason = truncate(reason, 120);
                self.blocks.push(DisplayBlock::System(format!(
                    "↳ provider fallback · {from} → {to} · {reason}"
                )));
                self.blocks_pushed();
            }
            AgentEvent::ContextCompactionStart {
                tokens_before,
                reason,
            } => {
                // Open the bracket as a rolling in-place block (same idiom as
                // ChildSessionActivity): the End event replaces this line
                // instead of pushing a second block.
                self.blocks.push(DisplayBlock::System(format!(
                    "⧗ compacting context · {reason} · ~{tokens_before} tokens"
                )));
                let idx = self.blocks.len() - 1;
                self.blocks_pushed();
                self.compaction_block = Some((idx, *tokens_before));
            }
            AgentEvent::ContextCompactionEnd {
                tokens_after,
                kv_prefix_broken,
            } => {
                let tokens_before = self.compaction_block.filter(|&(idx, _)| {
                    idx < self.blocks.len()
                        && matches!(&self.blocks[idx], DisplayBlock::System(s) if s.starts_with("⧗ compacting context"))
                });
                let mut text = match tokens_before {
                    Some((_, before)) if before > *tokens_after => {
                        let saved = (before.saturating_sub(*tokens_after)) * 100 / before;
                        format!(
                            "✓ context compacted · ~{before} → ~{tokens_after} tokens (-{saved}%)"
                        )
                    }
                    Some((_, before)) => {
                        format!("✓ context compacted · ~{before} → ~{tokens_after} tokens")
                    }
                    None => format!("✓ context compacted · ~{tokens_after} tokens"),
                };
                if *kv_prefix_broken {
                    text.push_str(" · cache prefix broken");
                }
                match tokens_before {
                    Some((idx, _)) => {
                        self.blocks[idx] = DisplayBlock::System(text);
                        self.block_mutated_at(idx);
                    }
                    None => {
                        self.blocks.push(DisplayBlock::System(text));
                        self.blocks_pushed();
                    }
                }
                self.compaction_block = None;
            }
            AgentEvent::ContextCompaction {
                message,
                tokens_before,
                tokens_after,
                ..
            } => {
                // Legacy single-shot event from old session logs (attach /
                // replay). One fresh block; never touches the rolling slot.
                let stats = match (tokens_before, tokens_after) {
                    (Some(before), Some(after)) => format!(" · ~{before} → ~{after} tokens"),
                    _ => String::new(),
                };
                self.blocks.push(DisplayBlock::System(format!(
                    "✓ context compacted · {message}{stats}"
                )));
                self.blocks_pushed();
            }
            AgentEvent::CostUpdated { .. }
            | AgentEvent::ContextStatsUpdated { .. }
            | AgentEvent::BusyStateChanged { .. }
            | AgentEvent::Checkpoint { .. } => {
                // StatusBar concerns. No `blocks` change — skip cache bump.
                return TranscriptAction::None;
            }
            _ => {}
        }
        // Every branch that mutates `self.blocks` maintains the line cache
        // incrementally via `blocks_pushed`/`block_mutated_at` (which also bump
        // the generation); non-mutating branches early-return above, and the
        // catch-all does not touch `blocks`. Nothing left to do here.
        TranscriptAction::None
    }

    /// Maintain the line cache after appending a block to `self.blocks`.
    ///
    /// Call immediately after EVERY `self.blocks.push(...)`. If the cache was
    /// valid for the pre-push block count at `last_width`, the new block's
    /// height is measured once and appended — no full rebuild. Either way the
    /// blocks generation advances; an invalid or stale cache simply stays
    /// invalid and falls back to the lazy full rebuild at next render (the
    /// pre-existing behavior).
    fn blocks_pushed(&mut self) {
        // Pre-push block count (this runs after the push).
        let was_valid = self.line_cache.is_valid(
            self.blocks_generation,
            self.last_width,
            self.blocks.len() - 1,
        );
        if was_valid && let Some(block) = self.blocks.last() {
            let h = block_line_count(block, self.last_width as usize);
            self.line_cache.append_height(h);
        }
        // Keep the content cache index-aligned with `blocks`.
        self.content_cache.push(None);
        self.blocks_generation = self.blocks_generation.wrapping_add(1);
        if was_valid {
            self.line_cache.sync_generation(self.blocks_generation);
        }
    }

    /// Maintain the line cache after an in-place mutation of `blocks[idx]`.
    ///
    /// Call immediately after EVERY in-place replacement/mutation of a single
    /// block. When the cache is valid for the current block count, only that
    /// block's height is re-measured and subsequent cumulative offsets are
    /// shifted by the delta — integer ops, no re-wrapping of other blocks.
    /// Otherwise just bump the generation (lazy full rebuild at next render).
    fn block_mutated_at(&mut self, idx: usize) {
        let valid =
            self.line_cache
                .is_valid(self.blocks_generation, self.last_width, self.blocks.len());
        if valid && idx < self.blocks.len() {
            let h = block_line_count(&self.blocks[idx], self.last_width as usize);
            self.line_cache.set_height(idx, h);
        }
        // Drop the mutated block's memoized wrapped lines.
        if idx < self.content_cache.len() {
            self.content_cache[idx] = None;
        }
        self.blocks_generation = self.blocks_generation.wrapping_add(1);
        if valid {
            self.line_cache.sync_generation(self.blocks_generation);
        }
    }

    fn flush_stream_before_tool(&mut self) {
        let reasoning = self.streaming_reasoning.take();
        self.streaming_reasoning_wrap.reset();
        if let Some(reasoning) = reasoning
            && !reasoning.trim().is_empty()
        {
            let duration_ms = self
                .reasoning_started_at
                .take()
                .map(|t| t.elapsed().as_millis() as u64);
            self.blocks.push(DisplayBlock::Thinking {
                content: reasoning,
                expanded: false,
                duration_ms,
            });
            self.blocks_pushed();
        }
        let assistant = self.streaming_assistant.take();
        self.streaming_assistant_wrap.reset();
        if let Some(s) = assistant
            && !s.trim().is_empty()
        {
            self.blocks.push(DisplayBlock::Assistant(s));
            self.blocks_pushed();
        }
    }

    pub(crate) fn push_error(&mut self, msg: String) {
        self.blocks.push(DisplayBlock::ErrorLine(msg));
        self.blocks_pushed();
    }

    pub(crate) fn push_system(&mut self, msg: String) {
        self.blocks.push(DisplayBlock::System(msg));
        self.blocks_pushed();
    }

    pub(crate) fn push_blocks(&mut self, blocks: Vec<DisplayBlock>) {
        for block in blocks {
            self.blocks.push(block);
            self.blocks_pushed();
        }
    }

    pub(crate) fn set_streaming_assistant(&mut self, text: Option<String>) {
        self.streaming_assistant = text;
        self.streaming_assistant_wrap.reset();
        if let Some(t) = &self.streaming_assistant {
            self.streaming_assistant_wrap.set_text(t);
        }
    }

    pub(crate) fn set_streaming_reasoning(&mut self, text: Option<String>) {
        self.streaming_reasoning = text;
        self.streaming_reasoning_wrap.reset();
        if let Some(t) = &self.streaming_reasoning {
            self.streaming_reasoning_wrap.set_text(t);
        }
    }

    pub(crate) fn clear(&mut self) {
        self.blocks.clear();
        self.streaming_assistant = None;
        self.streaming_reasoning = None;
        self.streaming_assistant_wrap.reset();
        self.streaming_reasoning_wrap.reset();
        self.streaming_reasoning_expanded = false;
        self.scroll_lines = 0;
        self.transcript_follow_tail = true;
        self.transcript_selection = None;
        self.transcript_dragging = false;
        self.transcript_drag_anchor = None;
        self.child_activity_blocks.clear();
        self.tool_running_blocks.clear();
        self.compaction_block = None;
        self.content_cache.clear();
        self.content_cache_width = 0;
        self.line_cache.reset();
        self.blocks_generation = self.blocks_generation.wrapping_add(1);
    }

    /// Store the current active approval request for rendering.
    pub(crate) fn set_active_approval(&mut self, _req: Option<ApprovalRequest>) {
        // Phase 3a: approval rendering handled by existing DisplayBlock::ApprovalPending
    }

    /// Store the current active question for rendering and answer routing.
    pub(crate) fn set_active_question(&mut self, q: Option<InteractiveQuestionPayload>) {
        // Track for answer routing via active_question_id()
        self._active_question = q;
    }

    /// Return the question_id of the currently active question (for answer routing).
    pub(crate) fn active_question_id(&self) -> Option<String> {
        self._active_question
            .as_ref()
            .map(|q| q.question_id.clone())
    }

    /// Toggle expanded state of a specific ToolDone block.
    pub(crate) fn toggle_tool_output(&mut self, block_index: usize) {
        let toggled = match self.blocks.get_mut(block_index) {
            Some(DisplayBlock::ToolDone { expanded, .. }) => {
                *expanded = !*expanded;
                true
            }
            _ => false,
        };
        if toggled {
            self.block_mutated_at(block_index);
        }
    }

    /// Toggle all ToolDone blocks: expand if any collapsed, collapse all otherwise.
    pub(crate) fn toggle_all_tool_output(&mut self) {
        let any_collapsed = self.blocks.iter().any(|b| {
            matches!(
                b,
                DisplayBlock::ToolDone {
                    expanded: false,
                    ..
                }
            )
        });
        let mut mutated: Vec<usize> = Vec::new();
        for (i, block) in self.blocks.iter_mut().enumerate() {
            if let DisplayBlock::ToolDone { expanded, .. } = block {
                *expanded = any_collapsed;
                mutated.push(i);
            }
        }
        for idx in mutated {
            self.block_mutated_at(idx);
        }
        let msg = if any_collapsed {
            "tool output expanded"
        } else {
            "tool output collapsed"
        };
        self.blocks
            .push(DisplayBlock::System(format!("[tool-output] {msg}")));
        self.blocks_pushed();
    }

    // ── Key handling ─────────────────────────────────────────────

    pub(crate) fn handle_key(&mut self, key: KeyEvent, area: Rect) -> TranscriptAction {
        match key.code {
            KeyCode::PageUp => {
                let th = area.height.saturating_sub(2) as usize;
                let page = th.saturating_sub(1).max(1);
                self.transcript_follow_tail = false;
                self.scroll_lines = self.scroll_lines.saturating_sub(page);
            }
            KeyCode::PageDown => {
                let inner_w = area.width.saturating_sub(2);
                let total = self.total_line_count(inner_w);
                let th = area.height.saturating_sub(2) as usize;
                let max_scroll = total.saturating_sub(th);
                let page = th.saturating_sub(1).max(1);
                self.scroll_lines = (self.scroll_lines + page).min(max_scroll);
                if self.scroll_lines >= max_scroll {
                    self.transcript_follow_tail = true;
                }
            }
            KeyCode::End => {
                self.transcript_follow_tail = true;
            }
            _ => {}
        }
        TranscriptAction::None
    }

    // ── Mouse handling ──────────────────────────────────────────

    pub(crate) fn handle_mouse(
        &mut self,
        event: &MouseEvent,
        area: Rect,
        total_lines: usize,
    ) -> TranscriptAction {
        let content_area = Rect::new(
            area.x + 1,
            area.y + 1,
            area.width.saturating_sub(2),
            area.height.saturating_sub(2),
        );
        let inside = event.column >= content_area.x
            && event.column < content_area.x + content_area.width
            && event.row >= content_area.y
            && event.row < content_area.y + content_area.height;
        if !inside {
            return TranscriptAction::None;
        }

        let th = content_area.height as usize;
        let max_scroll = total_lines.saturating_sub(th);

        let gline = (event.row - content_area.y) as usize + self.scroll_lines;
        let gcol = (event.column - content_area.x) as usize;

        match event.kind {
            MouseEventKind::ScrollUp => {
                self.transcript_selection = None;
                self.transcript_dragging = false;
                self.transcript_follow_tail = false;
                self.scroll_lines = self.scroll_lines.saturating_sub(MOUSE_SCROLL_LINES);
            }
            MouseEventKind::ScrollDown => {
                self.transcript_selection = None;
                self.transcript_dragging = false;
                self.scroll_lines = (self.scroll_lines + MOUSE_SCROLL_LINES).min(max_scroll);
                if self.scroll_lines >= max_scroll {
                    self.transcript_follow_tail = true;
                }
            }
            MouseEventKind::Down(MouseButton::Left) => {
                // Handle transcript hit: toggle thinking or answer question.
                let vis_start = self.scroll_lines;
                let local_idx = gline.saturating_sub(vis_start);
                if local_idx < self.last_visible_hits.len() {
                    match &self.last_visible_hits[local_idx] {
                        Some(TranscriptHit::ToggleThinking(block_idx)) => {
                            let toggled = match self.blocks.get_mut(*block_idx) {
                                Some(DisplayBlock::Thinking { expanded, .. }) => {
                                    *expanded = !*expanded;
                                    true
                                }
                                _ => false,
                            };
                            if toggled {
                                self.block_mutated_at(*block_idx);
                            }
                            self.transcript_selection = None;
                            self.transcript_dragging = false;
                            return TranscriptAction::None;
                        }
                        Some(TranscriptHit::ToggleStreamingThinking) => {
                            self.streaming_reasoning_expanded = !self.streaming_reasoning_expanded;
                            self.transcript_selection = None;
                            self.transcript_dragging = false;
                            return TranscriptAction::None;
                        }
                        Some(TranscriptHit::ToggleToolOutput(block_idx)) => {
                            let toggled = match self.blocks.get_mut(*block_idx) {
                                Some(DisplayBlock::ToolDone { expanded, .. }) => {
                                    *expanded = !*expanded;
                                    true
                                }
                                _ => false,
                            };
                            if toggled {
                                self.block_mutated_at(*block_idx);
                            }
                            self.transcript_selection = None;
                            self.transcript_dragging = false;
                            return TranscriptAction::None;
                        }
                        Some(TranscriptHit::Question(sel)) => {
                            self.transcript_selection = None;
                            self.transcript_dragging = false;
                            return TranscriptAction::QuestionAnswer(sel.clone());
                        }
                        None => {}
                    }
                }
                // No question hit — start a new text selection.
                let click_pos = (gline, gcol);
                self.transcript_selection = Some((click_pos, click_pos));
                self.transcript_drag_anchor = Some((gline, gcol));
                self.transcript_dragging = true;
            }
            MouseEventKind::Drag(MouseButton::Left) if self.transcript_dragging => {
                if let Some((anchor_line, anchor_col)) = self.transcript_drag_anchor {
                    if self.transcript_selection.is_some() {
                        let start = (
                            anchor_line.min(gline),
                            if anchor_line <= gline {
                                anchor_col
                            } else {
                                gcol
                            },
                        );
                        let end = (
                            anchor_line.max(gline),
                            if anchor_line <= gline {
                                gcol
                            } else {
                                anchor_col
                            },
                        );
                        self.transcript_selection = Some((start, end));
                    } else {
                        // First drag attempt after Down — filter spurious large jumps.
                        let line_dist = gline.abs_diff(anchor_line);
                        let col_dist = gcol.abs_diff(anchor_col);
                        if line_dist <= 1 && col_dist <= 3 {
                            let start = (
                                anchor_line.min(gline),
                                if anchor_line <= gline {
                                    anchor_col
                                } else {
                                    gcol
                                },
                            );
                            let end = (
                                anchor_line.max(gline),
                                if anchor_line <= gline {
                                    gcol
                                } else {
                                    anchor_col
                                },
                            );
                            self.transcript_selection = Some((start, end));
                        }
                    }
                } else if let Some(((anchor_line, _), _)) = self.transcript_selection {
                    let start = (anchor_line.min(gline), 0);
                    let end = (anchor_line.max(gline), gcol);
                    self.transcript_selection = Some((start.min(end), start.max(end)));
                }
            }
            MouseEventKind::Up(MouseButton::Left) => {
                self.transcript_dragging = false;
                self.transcript_drag_anchor = None;
                // Auto-copy selected text to clipboard on mouse-up.
                if let Some((sel_start, sel_end)) = self.transcript_selection {
                    let (sl, sc) = sel_start;
                    let (el, ec) = sel_end;
                    if sl < el || (sl == el && sc != ec) {
                        let inner_w = area.width.saturating_sub(2);
                        let all_lines = self.build_all_lines(inner_w);
                        let text = plain_text_from_lines(&all_lines, sel_start, sel_end);
                        let n = text.trim_end_matches('\n').chars().count();
                        match crate::image_attach::copy_text_to_clipboard(&text) {
                            Ok(()) => {
                                return TranscriptAction::PushSystem(format!(
                                    "Copied {n} chars to clipboard"
                                ));
                            }
                            Err(e) => {
                                return TranscriptAction::PushError(format!(
                                    "Clipboard failed: {e}"
                                ));
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        TranscriptAction::None
    }

    // ── Line counting helpers ───────────────────────────────────

    /// Visible logical line count of the streaming reasoning block at width
    /// `w`. Reads the incremental wrap cache in O(1), lazily rebuilding it only
    /// if the render width changed (`&mut self` for that rebuild).
    fn streaming_reasoning_line_count(&mut self, w: usize) -> usize {
        let Some(reasoning) = self.streaming_reasoning.as_deref() else {
            return 0;
        };
        if reasoning.is_empty() {
            return 0;
        }
        self.streaming_reasoning_wrap.ensure(reasoning, w);
        let total_rl = self.streaming_reasoning_wrap.len();
        let preview_rl = 5usize;
        let show_rl = if self.streaming_reasoning_expanded || total_rl <= preview_rl {
            total_rl
        } else {
            preview_rl
        };
        let mut rl = 1 + show_rl + 1;
        if total_rl > preview_rl {
            rl += 1;
        }
        rl
    }

    /// Visible logical line count of the streaming assistant block at width
    /// `w`. O(1) via the incremental wrap cache (see
    /// `streaming_reasoning_line_count`).
    fn streaming_assistant_line_count(&mut self, w: usize) -> usize {
        let Some(stream) = self.streaming_assistant.as_deref() else {
            return 0;
        };
        if stream.is_empty() {
            return 0;
        }
        self.streaming_assistant_wrap.ensure(stream, w);
        2 + self.streaming_assistant_wrap.len()
    }

    pub(crate) fn total_line_count(&mut self, width: u16) -> usize {
        self.last_width = width;
        let w = width.max(20) as usize;
        let mut n = if self
            .line_cache
            .is_valid(self.blocks_generation, width, self.blocks.len())
        {
            self.line_cache.total()
        } else {
            self.line_cache
                .rebuild(&self.blocks, self.blocks_generation, width);
            self.line_cache.total()
        };

        // Streaming reasoning block
        n += self.streaming_reasoning_line_count(w);

        // Streaming assistant block
        n += self.streaming_assistant_line_count(w);

        // Empty state fallback
        if n == 0 && self.blocks.is_empty() {
            n = 4;
        }

        n
    }

    // ── Line building ───────────────────────────────────────────

    fn build_visible_lines(
        &mut self,
        width: u16,
        area_height: usize,
    ) -> (Vec<Line<'static>>, Vec<LineAnswerHit>) {
        self.last_width = width;
        let w = width.max(20) as usize;
        // Ensure cache is up to date.
        if !self
            .line_cache
            .is_valid(self.blocks_generation, width, self.blocks.len())
        {
            self.line_cache
                .rebuild(&self.blocks, self.blocks_generation, width);
        }
        let blocks_total = self.line_cache.total();

        let srl = self.streaming_reasoning_line_count(w);
        let sal = self.streaming_assistant_line_count(w);
        let ef = if blocks_total == 0 && self.blocks.is_empty() {
            4usize
        } else {
            0
        };
        let total = blocks_total + srl + sal + ef;

        let start = self.scroll_lines;
        let end = (start + area_height).min(total);

        if start >= end || start >= total {
            self.last_visible_hits = Vec::new();
            return (Vec::new(), Vec::new());
        }

        let cap = (end - start).min(200);
        let mut lines: Vec<Line<'static>> = Vec::with_capacity(cap);
        let mut hits: Vec<LineAnswerHit> = Vec::with_capacity(cap);
        let mut global_line = 0usize;

        for (bi, block) in self.blocks.iter().enumerate() {
            let bh = self.line_cache.heights[bi];
            let block_end = global_line + bh;
            if block_end <= start {
                global_line = block_end;
                continue;
            }
            if global_line >= end {
                break;
            }
            emit_block_lines(
                block,
                bi,
                w,
                &mut self.content_cache,
                &mut self.content_cache_width,
                &mut lines,
                &mut hits,
                start.saturating_sub(global_line),
                end.saturating_sub(global_line.max(start)),
            );
            global_line = block_end;
        }

        // Streaming reasoning
        if srl > 0 && global_line < end {
            emit_streaming_reasoning_lines(
                &self.streaming_reasoning_wrap,
                self.streaming_reasoning_expanded,
                self.reasoning_started_at,
                &mut lines,
                &mut hits,
                start.saturating_sub(global_line),
                end.saturating_sub(global_line.max(start)),
            );
            global_line += srl;
        }

        // Streaming assistant
        if sal > 0 && global_line < end {
            emit_streaming_assistant_lines(
                &self.streaming_assistant_wrap,
                &mut lines,
                &mut hits,
                start.saturating_sub(global_line),
                end.saturating_sub(global_line.max(start)),
            );
            global_line += sal;
        }

        // Empty fallback
        if ef > 0 && global_line < end {
            emit_empty_fallback_lines(
                &mut lines,
                &mut hits,
                start.saturating_sub(global_line),
                end.saturating_sub(global_line.max(start)),
            );
        }

        self.last_visible_hits = hits.clone();
        (lines, hits)
    }

    /// Build ALL lines (for clipboard copy — non-virtualized).
    fn build_all_lines(&mut self, width: u16) -> Vec<Line<'static>> {
        let w = width.max(20) as usize;
        // Bring the streaming wrap caches up to the current width before the
        // (unconditional) streaming emission below.
        let _ = self.streaming_reasoning_line_count(w);
        let _ = self.streaming_assistant_line_count(w);
        let mut lines: Vec<Line<'static>> = Vec::new();
        let mut hits: Vec<LineAnswerHit> = Vec::new();

        for (bi, block) in self.blocks.iter().enumerate() {
            emit_block_lines(
                block,
                bi,
                w,
                &mut self.content_cache,
                &mut self.content_cache_width,
                &mut lines,
                &mut hits,
                0,
                usize::MAX,
            );
        }

        // Streaming reasoning
        emit_streaming_reasoning_lines(
            &self.streaming_reasoning_wrap,
            self.streaming_reasoning_expanded,
            self.reasoning_started_at,
            &mut lines,
            &mut hits,
            0,
            usize::MAX,
        );
        // Streaming assistant
        emit_streaming_assistant_lines(
            &self.streaming_assistant_wrap,
            &mut lines,
            &mut hits,
            0,
            usize::MAX,
        );

        // Empty fallback
        if lines.is_empty() && self.blocks.is_empty() {
            emit_empty_fallback_lines(&mut lines, &mut hits, 0, usize::MAX);
        }

        lines
    }

    // ── Render ──────────────────────────────────────────────────

    pub(crate) fn render(&mut self, area: Rect) -> Paragraph<'static> {
        let inner_w = area.width.saturating_sub(2);
        let total = self.total_line_count(inner_w);
        let transcript_h = area.height.saturating_sub(2) as usize;

        // Clamp scroll position
        let max_scroll = total.saturating_sub(transcript_h);
        // Self-heal follow-tail: whenever the viewport already shows the tail,
        // re-arm following. The wheel/PageDown handlers only re-arm when an
        // event lands exactly at the bottom *at event time* — but the user
        // typically stops scrolling as soon as the newest content is visible,
        // which can sit one or more notches above the absolute bottom while
        // content keeps streaming. Without this, `transcript_follow_tail`
        // stays disarmed and the view freezes at that offset while new
        // messages arrive below the fold.
        if self.scroll_lines >= max_scroll {
            self.transcript_follow_tail = true;
        }
        if self.transcript_follow_tail || self.scroll_lines > max_scroll {
            self.scroll_lines = max_scroll;
        }

        let (visible_lines, _hits) = self.build_visible_lines(inner_w, transcript_h);

        // Apply selection highlight
        let highlighted =
            apply_selection_highlight(visible_lines, self.scroll_lines, self.transcript_selection);

        let title = format!(" transcript — {total} lines ");

        Paragraph::new(Text::from(highlighted))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(theme::BORDER))
                    .title(Span::styled(title, Style::default().fg(theme::MUTED))),
            )
            .style(Style::default().bg(theme::BG))
    }
}

#[cfg(test)]
mod tests {
    use super::super::transcript_render::{cached_block_lines, wrap_text};
    use super::*;
    use nca_common::event::{AgentEvent, BusyState};

    #[test]
    fn provider_fallback_pushes_dim_system_line() {
        let mut t = TranscriptState::new();
        let before = t.blocks.len();
        t.apply_event(&AgentEvent::ProviderFallback {
            from: "DeepSeek".into(),
            to: "OpenAI".into(),
            reason: "rate_limited: Rate limited, retry after 1000ms".into(),
        });
        assert_eq!(t.blocks.len(), before + 1, "one system line pushed");
        match t.blocks.last() {
            Some(DisplayBlock::System(text)) => {
                assert!(text.contains("DeepSeek → OpenAI"), "line: {text}");
                assert!(text.contains("rate_limited"), "line: {text}");
            }
            other => panic!("expected System block, got {other:?}"),
        }
    }

    // ── Follow-tail self-heal ────────────────────────────────────
    // Regression: users who scroll back up and then return to the newest
    // message expect the view to keep following new output. The wheel handler
    // only re-arms `transcript_follow_tail` when an event lands exactly at
    // the bottom *at event time*; while content streams, the bottom keeps
    // moving and that landing is easily missed. Render must therefore re-arm
    // whenever the viewport already shows the tail.
    #[test]
    fn render_rearms_follow_when_viewport_shows_tail() {
        let mut t = TranscriptState::new();
        for i in 0..12 {
            t.apply_event(&AgentEvent::MessageReceived {
                role: "assistant".into(),
                content: format!("message number {i} with some text"),
                steering: false,
            });
        }
        let area = Rect::new(0, 1, 80, 12); // content viewport: 78 wide, 10 tall
        let max_scroll = t.total_line_count(78) - 10;
        assert!(max_scroll > 0, "fixture: transcript must overflow viewport");

        // User scrolled up (follow disarmed), then returned to the newest
        // message: viewport sits at the bottom, but the last wheel event did
        // not land exactly at the (streaming) bottom, so `transcript_follow_tail`
        // is still false. Render at this position must re-arm follow.
        t.transcript_follow_tail = false;
        t.scroll_lines = max_scroll;
        let _ = t.render(area);
        assert!(
            t.transcript_follow_tail,
            "render must re-arm follow when the viewport shows the tail"
        );

        // New message arrives below the fold → view must follow it.
        t.apply_event(&AgentEvent::MessageReceived {
            role: "assistant".into(),
            content: "a brand new message".into(),
            steering: false,
        });

        let _ = t.render(area);
        let new_max = t.total_line_count(78) - 10;
        assert_eq!(
            t.scroll_lines, new_max,
            "view must follow the tail once the viewport shows it"
        );
        assert!(t.transcript_follow_tail);
    }

    #[test]
    fn render_keeps_position_when_reading_older_content() {
        let mut t = TranscriptState::new();
        for i in 0..12 {
            t.apply_event(&AgentEvent::MessageReceived {
                role: "assistant".into(),
                content: format!("message number {i} with some text"),
                steering: false,
            });
        }
        let area = Rect::new(0, 1, 80, 12);
        let max_scroll = t.total_line_count(78) - 10;

        // User is reading older content, far above the tail.
        t.transcript_follow_tail = false;
        t.scroll_lines = max_scroll.saturating_sub(20);

        t.apply_event(&AgentEvent::MessageReceived {
            role: "assistant".into(),
            content: "another new message".into(),
            steering: false,
        });
        let before = t.scroll_lines;
        let _ = t.render(area);
        assert_eq!(t.scroll_lines, before, "reading position must not jump");
        assert!(
            !t.transcript_follow_tail,
            "must stay disarmed away from tail"
        );
    }

    #[test]
    fn wheel_down_to_bottom_rearms_follow() {
        use crossterm::event::{KeyModifiers, MouseEvent, MouseEventKind};

        let mut t = TranscriptState::new();
        for i in 0..12 {
            t.apply_event(&AgentEvent::MessageReceived {
                role: "assistant".into(),
                content: format!("message number {i} with some text"),
                steering: false,
            });
        }
        let area = Rect::new(0, 1, 80, 12);
        let total = t.total_line_count(78);

        // Pin the viewport to the bottom first (as the app does while
        // following), then scroll up one wheel notch.
        let _ = t.render(area);
        let up = MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 5,
            row: 5,
            modifiers: KeyModifiers::NONE,
        };
        t.handle_mouse(&up, area, total);
        assert!(!t.transcript_follow_tail, "wheel up disarms follow");

        let down = MouseEvent {
            kind: MouseEventKind::ScrollDown,
            column: 5,
            row: 5,
            modifiers: KeyModifiers::NONE,
        };
        t.handle_mouse(&down, area, total);
        assert!(
            t.transcript_follow_tail,
            "wheel down landing at bottom re-arms follow"
        );
    }

    // The BlockLineCache rebuild is O(total committed transcript text). During
    // streaming the agent emits one TokensStreamed/ReasoningStreamed per token,
    // and these only append to `streaming_assistant`/`streaming_reasoning`
    // (measured outside the cache). If they bump `blocks_generation`, every
    // token forces a full rebuild — the dominant per-frame cost in long
    // sessions, which starves the input loop and makes typing feel frozen.
    // This invariant pins the fix: streaming/status events must NOT invalidate
    // the committed-blocks cache, while real block mutations still must.
    #[test]
    fn streaming_and_status_events_skip_block_cache_invalidation() {
        let mut t = TranscriptState::new();

        // Seed one committed block so the cache has something to track.
        t.apply_event(&AgentEvent::MessageReceived {
            role: "assistant".into(),
            content: "hello world".into(),
            steering: false,
        });
        let gen_after_commit = t.blocks_generation;
        assert!(
            !t.blocks.is_empty(),
            "fixture setup: a committed message must produce a block"
        );

        // None of these mutate `blocks` → generation must stay put.
        t.apply_event(&AgentEvent::SessionStarted {
            session_id: "s".into(),
            workspace: "/tmp".into(),
            model: "m".into(),
        });
        t.apply_event(&AgentEvent::TokensStreamed {
            delta: "foo ".into(),
        });
        t.apply_event(&AgentEvent::TokensStreamed {
            delta: "bar".into(),
        });
        t.apply_event(&AgentEvent::ReasoningStreamed {
            delta: "hmm".into(),
        });
        t.apply_event(&AgentEvent::CostUpdated {
            input_tokens: 1,
            output_tokens: 1,
            cache_read_tokens: 0,
            estimated_cost_usd: 0.0,
        });
        t.apply_event(&AgentEvent::BusyStateChanged {
            state: BusyState::Thinking,
        });
        assert_eq!(
            t.blocks_generation, gen_after_commit,
            "streaming/status events must not invalidate the blocks line-height cache"
        );

        // A real committed block must still invalidate the cache.
        t.apply_event(&AgentEvent::MessageReceived {
            role: "user".into(),
            content: "next turn".into(),
            steering: false,
        });
        assert_eq!(
            t.blocks_generation,
            gen_after_commit.wrapping_add(1),
            "committed block changes must still invalidate the cache"
        );
    }

    // ── Incremental line-cache maintenance ─────────────────────
    // `blocks_pushed`/`block_mutated_at` maintain the BlockLineCache one
    // block at a time (append/replace a single measured height) instead of
    // forcing the O(total transcript text) full rebuild on every mutation.
    // These tests pin the invariant that routine transcript growth and
    // in-place mutations stay fully incremental, that width changes still
    // rebuild, and that incremental totals agree with a fresh full rebuild.

    fn spawn_child(t: &mut TranscriptState, id: &str) {
        t.apply_event(&AgentEvent::ChildSessionSpawned {
            parent_session_id: "parent".into(),
            child_session_id: id.into(),
            task: "some task".into(),
            workspace: std::path::PathBuf::from("/tmp"),
            branch: None,
        });
    }

    fn system_blocks(t: &TranscriptState) -> Vec<&DisplayBlock> {
        t.blocks
            .iter()
            .filter(|b| matches!(b, DisplayBlock::System(_)))
            .collect()
    }

    #[test]
    fn block_pushes_append_cache_without_full_rebuild() {
        let mut t = TranscriptState::new();
        t.apply_event(&AgentEvent::MessageReceived {
            role: "assistant".into(),
            content: "hello world".into(),
            steering: false,
        });
        let total_first = t.total_line_count(78);
        let rebuilt = t.line_cache.rebuild_count;
        assert_eq!(
            rebuilt, 1,
            "first measure must build the cache exactly once"
        );

        for i in 0..5 {
            t.apply_event(&AgentEvent::MessageReceived {
                role: "assistant".into(),
                content: format!("follow-up message {i}"),
                steering: false,
            });
        }
        let total_after = t.total_line_count(78);
        assert_eq!(
            t.line_cache.rebuild_count, rebuilt,
            "appended blocks must not trigger a full cache rebuild"
        );
        assert!(
            total_after > total_first,
            "new blocks must actually add lines"
        );

        // Correctness cross-check: a fresh state fed the same events (which
        // builds the cache through the full-rebuild path) must report the
        // identical total — proving incremental maintenance tracks the same
        // heights as a from-scratch measurement.
        let mut fresh = TranscriptState::new();
        fresh.apply_event(&AgentEvent::MessageReceived {
            role: "assistant".into(),
            content: "hello world".into(),
            steering: false,
        });
        for i in 0..5 {
            fresh.apply_event(&AgentEvent::MessageReceived {
                role: "assistant".into(),
                content: format!("follow-up message {i}"),
                steering: false,
            });
        }
        assert_eq!(
            fresh.total_line_count(78),
            total_after,
            "incrementally maintained cache must agree with a fresh full rebuild"
        );
    }

    #[test]
    fn in_place_mutations_use_targeted_cache_update() {
        let mut t = TranscriptState::new();
        t.apply_event(&AgentEvent::ToolCallStarted {
            call_id: "call-1".into(),
            tool: "bash".into(),
            input: serde_json::json!({ "command": "true" }),
        });
        t.total_line_count(78);
        let rebuilt = t.line_cache.rebuild_count;

        // ToolRunning → ToolDone is an in-place replacement of one block.
        t.apply_event(&AgentEvent::ToolCallCompleted {
            call_id: "call-1".into(),
            output: ToolResult {
                timed_out: false,
                call_id: "call-1".into(),
                success: true,
                output: "line1\nline2\nline3\nline4".into(),
                error: None,
            },
            duration_ms: 5,
        });
        let total_collapsed = t.total_line_count(78);
        assert_eq!(
            t.line_cache.rebuild_count, rebuilt,
            "ToolRunning → ToolDone swap must not rebuild the cache"
        );

        // Expanding the ToolDone block is also a single-block mutation.
        let idx = t
            .blocks
            .iter()
            .position(|b| matches!(b, DisplayBlock::ToolDone { .. }))
            .expect("ToolCallCompleted must produce a ToolDone block");
        t.toggle_tool_output(idx);
        let total_expanded = t.total_line_count(78);
        assert_eq!(
            t.line_cache.rebuild_count, rebuilt,
            "toggle_tool_output must not rebuild the cache"
        );
        assert!(
            total_expanded > total_collapsed,
            "expanding the tool output must add lines"
        );
    }

    #[test]
    fn width_change_still_full_rebuilds() {
        let mut t = TranscriptState::new();
        t.apply_event(&AgentEvent::MessageReceived {
            role: "assistant".into(),
            content: "hello world".into(),
            steering: false,
        });
        t.total_line_count(78);
        let rebuilt = t.line_cache.rebuild_count;

        t.total_line_count(40);
        assert_eq!(
            t.line_cache.rebuild_count,
            rebuilt + 1,
            "a width change invalidates the cache and must force a full rebuild"
        );
    }

    #[test]
    fn child_activity_burst_rolls_into_single_block() {
        let mut t = TranscriptState::new();
        spawn_child(&mut t, "child-a-0001");

        for (i, phase) in ["plan", "read", "edit", "validate", "commit"]
            .iter()
            .enumerate()
        {
            t.apply_event(&AgentEvent::ChildSessionActivity {
                child_session_id: "child-a-0001".into(),
                phase: (*phase).into(),
                detail: format!("detail {i}"),
            });
        }
        let systems = system_blocks(&t);
        assert_eq!(
            systems.len(),
            2,
            "spawn banner + ONE rolling activity block"
        );
        assert!(
            matches!(systems[1], DisplayBlock::System(s) if s.contains("commit")),
            "rolling block must show the latest phase"
        );

        // More activities for the same child keep rolling in place.
        for i in 0..3 {
            t.apply_event(&AgentEvent::ChildSessionActivity {
                child_session_id: "child-a-0001".into(),
                phase: format!("phase-{i}"),
                detail: "more".into(),
            });
        }
        let systems = system_blocks(&t);
        assert_eq!(
            systems.len(),
            2,
            "later activities must keep rolling into the single block"
        );
        assert!(
            matches!(systems[1], DisplayBlock::System(s) if s.contains("phase-2")),
            "rolling block must be updated to the newest phase"
        );
        assert_eq!(
            t.child_activity_blocks.get("child-a-0001"),
            Some(&1usize),
            "map must keep pointing at the rolling block"
        );
    }

    #[test]
    fn distinct_children_roll_into_distinct_blocks() {
        let mut t = TranscriptState::new();
        spawn_child(&mut t, "child-a-0001");
        spawn_child(&mut t, "child-b-0002");

        // Interleave activities for the two children.
        for (phase_a, phase_b) in [("a-1", "b-1"), ("a-2", "b-2")] {
            t.apply_event(&AgentEvent::ChildSessionActivity {
                child_session_id: "child-a-0001".into(),
                phase: phase_a.into(),
                detail: String::new(),
            });
            t.apply_event(&AgentEvent::ChildSessionActivity {
                child_session_id: "child-b-0002".into(),
                phase: phase_b.into(),
                detail: String::new(),
            });
        }

        let systems = system_blocks(&t);
        assert_eq!(
            systems.len(),
            4,
            "2 spawn banners + one rolling block per child"
        );
        let a_last = systems
            .iter()
            .filter(|b| matches!(b, DisplayBlock::System(s) if s.contains("a-2")))
            .count();
        let b_last = systems
            .iter()
            .filter(|b| matches!(b, DisplayBlock::System(s) if s.contains("b-2")))
            .count();
        assert_eq!(
            a_last, 1,
            "child A's last phase appears in exactly one block"
        );
        assert_eq!(
            b_last, 1,
            "child B's last phase appears in exactly one block"
        );
        assert_eq!(t.child_activity_blocks.len(), 2);
        assert_ne!(
            t.child_activity_blocks["child-a-0001"], t.child_activity_blocks["child-b-0002"],
            "each child must have its own rolling block index"
        );
    }

    #[test]
    fn child_activity_burst_no_cache_rebuild() {
        let mut t = TranscriptState::new();
        t.apply_event(&AgentEvent::MessageReceived {
            role: "assistant".into(),
            content: "seed".into(),
            steering: false,
        });
        spawn_child(&mut t, "child-a-0001");
        t.total_line_count(78);
        let rebuilt = t.line_cache.rebuild_count;

        for i in 0..10 {
            t.apply_event(&AgentEvent::ChildSessionActivity {
                child_session_id: "child-a-0001".into(),
                phase: format!("phase-{i}"),
                detail: format!("detail {i}"),
            });
        }
        t.total_line_count(78);
        assert_eq!(
            t.line_cache.rebuild_count, rebuilt,
            "a same-child activity burst must stay fully incremental"
        );
        assert_eq!(
            t.blocks.len(),
            3,
            "seed message + spawn banner + one rolling block"
        );
    }

    // ── Context compaction bracket ──

    #[test]
    fn compaction_bracket_collapses_to_single_block() {
        let mut t = TranscriptState::new();
        t.apply_event(&AgentEvent::ContextCompactionStart {
            tokens_before: 10_000,
            reason: "auto_summarize".into(),
        });
        t.apply_event(&AgentEvent::ContextCompactionEnd {
            tokens_after: 3_700,
            kv_prefix_broken: true,
        });
        let systems = system_blocks(&t);
        assert_eq!(systems.len(), 1, "bracket must collapse to ONE block");
        assert!(
            matches!(systems[0], DisplayBlock::System(s) if s.contains("→") && s.contains("3700") && s.contains("cache prefix broken")),
            "summary must include the arrow, tokens_after, and kv note"
        );
        assert!(
            matches!(systems[0], DisplayBlock::System(s) if s.contains("-63%")),
            "reduction percentage must be computed"
        );
        assert!(t.compaction_block.is_none(), "slot cleared after bracket");

        // A second bracket adds exactly one more block.
        t.apply_event(&AgentEvent::ContextCompactionStart {
            tokens_before: 9_000,
            reason: "overflow_prune".into(),
        });
        t.apply_event(&AgentEvent::ContextCompactionEnd {
            tokens_after: 9_500,
            kv_prefix_broken: false,
        });
        assert_eq!(system_blocks(&t).len(), 2, "second bracket adds one block");
    }

    #[test]
    fn compaction_end_without_start_still_renders() {
        let mut t = TranscriptState::new();
        t.apply_event(&AgentEvent::ContextCompactionEnd {
            tokens_after: 4_000,
            kv_prefix_broken: false,
        });
        let systems = system_blocks(&t);
        assert_eq!(systems.len(), 1);
        assert!(matches!(systems[0], DisplayBlock::System(s) if s.contains("4000")));
    }

    #[test]
    fn legacy_context_compaction_renders() {
        let mut t = TranscriptState::new();
        t.apply_event(&AgentEvent::ContextCompaction {
            phase: "completed".into(),
            message: "summarized old turns".into(),
            tokens_before: Some(12_000),
            tokens_after: Some(4_000),
            retained_groups: Some(5),
            dropped_groups: Some(3),
        });
        let systems = system_blocks(&t);
        assert_eq!(systems.len(), 1);
        assert!(
            matches!(systems[0], DisplayBlock::System(s) if s.contains("summarized old turns") && s.contains("→")),
            "legacy event must render message + token stats"
        );
        assert!(
            t.compaction_block.is_none(),
            "legacy event must not touch the rolling slot"
        );
    }

    #[test]
    fn compaction_bracket_no_cache_rebuild() {
        let mut t = TranscriptState::new();
        t.apply_event(&AgentEvent::MessageReceived {
            role: "assistant".into(),
            content: "seed".into(),
            steering: false,
        });
        t.total_line_count(78);
        let rebuilt = t.line_cache.rebuild_count;

        t.apply_event(&AgentEvent::ContextCompactionStart {
            tokens_before: 10_000,
            reason: "auto_summarize".into(),
        });
        t.apply_event(&AgentEvent::ContextCompactionEnd {
            tokens_after: 3_000,
            kv_prefix_broken: true,
        });
        t.total_line_count(78);
        assert_eq!(
            t.line_cache.rebuild_count, rebuilt,
            "the compaction bracket must stay fully incremental"
        );
        assert_eq!(t.blocks.len(), 2, "seed message + one compacted block");
    }

    // ── StreamingWrap incremental wrap == full wrap_text ──────────
    // `StreamingWrap` wraps only the paragraphs a delta completes plus the
    // trailing partial paragraph, so per-delta work is O(delta) instead of
    // O(buffer). These tests pin the invariant that it is byte-for-byte
    // equivalent to calling `wrap_text` on the whole accumulated buffer —
    // including the empty-string and all-whitespace fallbacks and the
    // mid-stream width-change rebuild.

    /// Deterministic LCG (no external deps) for reproducible adversarial cases.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 11
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// Assert the incremental cache mirrors `wrap_text(full, w)` exactly —
    /// both the line count and every individual line (and emptiness).
    fn assert_streaming_wrap_equivalent(wrap: &mut StreamingWrap, full: &str, w: usize) {
        wrap.ensure(full, w);
        let expected = wrap_text(full, w);
        assert_eq!(
            wrap.len(),
            expected.len(),
            "line count mismatch: full={full:?} w={w}"
        );
        for (i, exp) in expected.iter().enumerate() {
            assert_eq!(
                wrap.line(i),
                exp.as_str(),
                "line {i} mismatch: full={full:?} w={w}"
            );
        }
        assert_eq!(
            wrap.is_empty(),
            full.is_empty(),
            "emptiness mismatch: full={full:?}"
        );
    }

    /// Feed a `(width, delta)` sequence through a fresh `StreamingWrap`,
    /// checking equivalence to `wrap_text` after every append (the reader path
    /// also calls `ensure`, so a width-only change is exercised too).
    fn check_delta_sequence(steps: &[(usize, String)]) {
        let mut wrap = StreamingWrap::new();
        let mut full = String::new();
        for (w, delta) in steps {
            wrap.append(delta, *w);
            full.push_str(delta);
            assert_streaming_wrap_equivalent(&mut wrap, &full, *w);
        }
    }

    #[test]
    fn streaming_wrap_matches_wrap_text_for_adversarial_deltas() {
        let w = 30usize;

        // Empty input: exactly one empty line, matching `wrap_text("")`.
        check_delta_sequence(&[(w, String::new())]);

        // CJK with no spaces: hard-split, no word boundaries.
        let cjk = "中文没有空格的长句子需要逐字硬切分";
        let cjk_chars: Vec<char> = cjk.chars().collect();
        check_delta_sequence(&[
            (w, cjk_chars[..5].iter().collect()),
            (w, cjk_chars[5..9].iter().collect()),
            (w, cjk_chars[9..].iter().collect()),
        ]);

        // Multi-byte emoji fed one char at a time (never split mid-scalar).
        let emoji: Vec<(usize, String)> =
            "😀🎉🌟✨🚀".chars().map(|c| (w, c.to_string())).collect();
        check_delta_sequence(&emoji);

        // A word split across two deltas (mid-word / mid-token cut).
        check_delta_sequence(&[
            (w, "supercalifragilistic".to_string()),
            (w, "expialidocious yes".to_string()),
        ]);

        // A word whose width lands exactly on the wrap boundary, plus one over.
        check_delta_sequence(&[(w, "e".repeat(w)), (w, " next".to_string())]);
        check_delta_sequence(&[(w, "e".repeat(w + 1))]);

        // Single-newline deltas, consecutive newlines, empty paragraphs.
        check_delta_sequence(&[
            (w, "\n".to_string()),
            (w, "\n".to_string()),
            (w, "text".to_string()),
        ]);
        check_delta_sequence(&[(w, "a\n\n\nb".to_string())]);
        check_delta_sequence(&[
            (w, "p1\n".to_string()),
            (w, "p2\n".to_string()),
            (w, "\n".to_string()),
        ]);

        // Whitespace-only paragraph (fallback semantics) then real text.
        check_delta_sequence(&[(w, "   ".to_string()), (w, "x".to_string())]);
        check_delta_sequence(&[(w, " \n ".to_string())]);

        // Width changes mid-stream must trigger `rebuild_from`.
        check_delta_sequence(&[
            (
                w,
                "alpha beta gamma delta epsilon zeta eta theta iota kappa".to_string(),
            ),
            (24, " lambda mu nu xi omicron pi rho sigma tau".to_string()),
            (w, " upsilon phi chi psi omega".to_string()),
        ]);

        // A width change observed by a *read* (`ensure`) rather than an append.
        let mut wrap = StreamingWrap::new();
        let text = "alpha beta gamma delta epsilon zeta eta theta";
        wrap.append(text, 30);
        assert_streaming_wrap_equivalent(&mut wrap, text, 20);
        assert_streaming_wrap_equivalent(&mut wrap, text, 30);
    }

    #[test]
    fn streaming_wrap_matches_wrap_text_under_fuzz() {
        const TOKENS: &[&str] = &[
            "hello",
            "world",
            "the quick brown fox jumps over the lazy dog",
            "a",
            "ab",
            "abc",
            "supercalifragilisticexpialidocious",
            "中文没有空格的长句子需要硬切分",
            "汉字内容",
            "日本語のテキスト",
            "😀",
            "🎉🎉",
            "é",
            "naïve café",
            "  ",
            " ",
            "\t",
            "\n",
            "\n\n",
            "\n\n\n",
            "word",
            "= ",
            "->",
            "x",
            "yyyyyyyyyyyyyyyy",
        ];
        const WIDTHS: &[usize] = &[20, 21, 24, 30, 40, 60, 78, 80, 100];

        for case in 0..900u64 {
            let mut rng = Lcg::new(0x9E37_79B9_7F4A_7C15 ^ case.wrapping_mul(0x0001_0000_0001));
            let mut width = WIDTHS[rng.below(WIDTHS.len())];
            let mut full = String::new();
            let mut wrap = StreamingWrap::new();
            let mut steps = 0usize;
            while steps < 12 && full.len() < 320 {
                steps += 1;
                // Occasionally resize mid-stream (forces `rebuild_from`).
                if rng.below(6) == 0 {
                    width = WIDTHS[rng.below(WIDTHS.len())];
                }
                let token: String = match rng.below(10) {
                    0 => "e".repeat(width),
                    1 => "e".repeat(width + 1),
                    2 => "中".repeat(width / 3 + 2),
                    3 => format!("{} f", "e".repeat(width.saturating_sub(2))),
                    _ => TOKENS[rng.below(TOKENS.len())].to_string(),
                };
                // Append whole, or split into two halves at a char boundary so
                // words / multi-byte runs are cut across deltas.
                if token.chars().count() >= 2 && rng.below(3) == 0 {
                    let count = token.chars().count();
                    let at = 1 + rng.below(count - 1);
                    let byte = token
                        .char_indices()
                        .nth(at)
                        .map(|(b, _)| b)
                        .unwrap_or(token.len());
                    let (head, tail) = token.split_at(byte);
                    for piece in [head, tail] {
                        wrap.append(piece, width);
                        full.push_str(piece);
                        assert_streaming_wrap_equivalent(&mut wrap, &full, width);
                    }
                } else {
                    wrap.append(&token, width);
                    full.push_str(&token);
                    assert_streaming_wrap_equivalent(&mut wrap, &full, width);
                }
            }
        }
    }

    // ── StreamingWrap reset at every buffer-clear point ───────────
    // The wrap cache mirrors a specific streaming buffer; when that buffer is
    // cleared (commit, tool flush, external set) the cache must be reset too,
    // or stale wrapped lines would leak into the next streaming turn.

    #[test]
    fn streaming_wrap_is_cleared_at_every_stream_reset_point() {
        let w = 40usize;

        // 1) Committing a user message clears the assistant wrap.
        let mut t = TranscriptState::new();
        t.last_width = w as u16;
        for d in ["alpha ", "beta ", "gamma delta"] {
            t.apply_event(&AgentEvent::TokensStreamed { delta: d.into() });
        }
        assert!(
            !t.streaming_assistant_wrap.is_empty(),
            "wrap must track the live stream"
        );
        t.apply_event(&AgentEvent::MessageReceived {
            role: "user".into(),
            content: "next prompt".into(),
            steering: false,
        });
        assert!(
            t.streaming_assistant_wrap.is_empty(),
            "user commit must reset the assistant wrap"
        );
        // A fresh delta must wrap exactly — no stale lines carried over.
        t.apply_event(&AgentEvent::TokensStreamed {
            delta: "fresh".into(),
        });
        assert_eq!(
            t.streaming_assistant_wrap.len(),
            wrap_text("fresh", w).len()
        );
        assert_eq!(t.streaming_assistant_wrap.line(0), "fresh");

        // 2) Committing an assistant message clears reasoning + assistant wraps.
        let mut t = TranscriptState::new();
        t.last_width = w as u16;
        t.apply_event(&AgentEvent::ReasoningStreamed {
            delta: "thinking hard".into(),
        });
        t.apply_event(&AgentEvent::TokensStreamed {
            delta: "answer text".into(),
        });
        assert!(!t.streaming_reasoning_wrap.is_empty());
        assert!(!t.streaming_assistant_wrap.is_empty());
        t.apply_event(&AgentEvent::MessageReceived {
            role: "assistant".into(),
            content: "final answer".into(),
            steering: false,
        });
        assert!(
            t.streaming_reasoning_wrap.is_empty(),
            "assistant commit must reset the reasoning wrap"
        );
        assert!(
            t.streaming_assistant_wrap.is_empty(),
            "assistant commit must reset the assistant wrap"
        );

        // 3) flush_stream_before_tool clears both wraps.
        let mut t = TranscriptState::new();
        t.last_width = w as u16;
        t.apply_event(&AgentEvent::ReasoningStreamed {
            delta: "reason".into(),
        });
        t.apply_event(&AgentEvent::TokensStreamed {
            delta: "text".into(),
        });
        t.apply_event(&AgentEvent::ToolCallStarted {
            call_id: "c".into(),
            tool: "bash".into(),
            input: serde_json::json!({ "command": "true" }),
        });
        assert!(
            t.streaming_reasoning_wrap.is_empty(),
            "tool flush must reset the reasoning wrap"
        );
        assert!(
            t.streaming_assistant_wrap.is_empty(),
            "tool flush must reset the assistant wrap"
        );

        // 4) External set_streaming_* replaces (Some) / clears (None) the wrap.
        let mut t = TranscriptState::new();
        t.set_streaming_assistant(Some("hello world".into()));
        assert!(!t.streaming_assistant_wrap.is_empty());
        t.streaming_assistant_wrap.ensure("hello world", 78);
        assert_eq!(
            t.streaming_assistant_wrap.len(),
            wrap_text("hello world", 78).len(),
            "a set buffer must wrap exactly on first read"
        );
        t.set_streaming_assistant(None);
        assert!(
            t.streaming_assistant_wrap.is_empty(),
            "set(None) must clear the assistant wrap"
        );
        t.set_streaming_reasoning(Some("deep thought".into()));
        assert!(!t.streaming_reasoning_wrap.is_empty());
        t.set_streaming_reasoning(None);
        assert!(
            t.streaming_reasoning_wrap.is_empty(),
            "set(None) must clear the reasoning wrap"
        );
    }

    // ── Committed-block content cache ─────────────────────────────
    // `content_cache` memoizes each block's wrapped primary text. It must stay
    // index-aligned with `blocks`, invalidate the mutated slot on in-place
    // mutation, and drop *every* slot when the render width changes.

    #[test]
    fn content_cache_stays_aligned_and_invalidates_on_mutation() {
        let mut t = TranscriptState::new();
        let events = [
            AgentEvent::MessageReceived {
                role: "assistant".into(),
                content: "first block text".into(),
                steering: false,
            },
            AgentEvent::MessageReceived {
                role: "assistant".into(),
                content: "second block text".into(),
                steering: false,
            },
            AgentEvent::TokensStreamed {
                delta: "stream".into(),
            },
            AgentEvent::ToolCallStarted {
                call_id: "c".into(),
                tool: "bash".into(),
                input: serde_json::json!({ "command": "x" }),
            },
            AgentEvent::ToolOutputChunk {
                call_id: "c".into(),
                delta: "chunk".into(),
            },
            AgentEvent::ToolCallCompleted {
                call_id: "c".into(),
                output: tool_result("c", "done"),
                duration_ms: 3,
            },
        ];
        for ev in &events {
            t.apply_event(ev);
            assert_eq!(
                t.content_cache.len(),
                t.blocks.len(),
                "content_cache must stay index-aligned with blocks"
            );
        }

        // Populate the cache via the full-build path the renderer uses.
        let _ = t.build_all_lines(78);
        assert_eq!(t.content_cache_width, 78);
        let (idx, text) = t
            .blocks
            .iter()
            .enumerate()
            .find_map(|(i, b)| match b {
                DisplayBlock::Assistant(s) | DisplayBlock::User(s) => Some((i, s.clone())),
                _ => None,
            })
            .expect("fixture must contain a committed text block");
        assert_eq!(
            t.content_cache[idx].as_deref(),
            Some(&wrap_text(&text, 78)),
            "cache hit must hold the wrapped text"
        );

        // In-place mutation drops the slot's memoized lines.
        t.block_mutated_at(idx);
        assert!(
            t.content_cache[idx].is_none(),
            "block_mutated_at must invalidate the mutated slot"
        );
        assert_eq!(t.content_cache.len(), t.blocks.len());

        // A render-width change must invalidate every slot, then re-wrap.
        let other = (0..t.blocks.len()).find(|&i| i != idx && t.content_cache[i].is_some());
        if let Some(other) = other {
            let text_a = text.clone();
            let fresh = cached_block_lines(
                &mut t.content_cache,
                &mut t.content_cache_width,
                idx,
                &text_a,
                40,
            );
            assert_eq!(*fresh, wrap_text(&text_a, 40));
            assert_eq!(t.content_cache_width, 40);
            assert!(
                t.content_cache[other].is_none(),
                "a width change must invalidate every slot, not just the re-wrapped one"
            );
        }
    }

    // ── streamed_output bounding ──────────────────────────────────

    #[test]
    fn streamed_output_is_capped_and_never_split_mid_char() {
        let mut t = TranscriptState::new();
        t.apply_event(&AgentEvent::ToolCallStarted {
            call_id: "c".into(),
            tool: "bash".into(),
            input: serde_json::json!({ "command": "true" }),
        });

        // 16 bytes per chunk (mix of 3- and 4-byte scalars); 30 chunks feed
        // ~24 KiB, well past the 8 KiB cap.
        let chunk = "中文😀内容".repeat(50);
        let mut full = String::new();
        for _ in 0..30 {
            t.apply_event(&AgentEvent::ToolOutputChunk {
                call_id: "c".into(),
                delta: chunk.clone(),
            });
            full.push_str(&chunk);
        }

        let so = streamed_output_of(&t, 0);
        assert!(
            so.len() <= 8192,
            "streamed_output must stay bounded at the 8 KiB cap: {} bytes",
            so.len()
        );
        assert!(!so.is_empty());
        assert!(
            so.len() < full.len(),
            "fixture must actually exercise the cap"
        );
        assert!(
            full.ends_with(&so),
            "the capped buffer must be a clean tail (never cut mid-char)"
        );
    }

    // ── ToolOutputChunk O(1) routing ──────────────────────────────

    #[test]
    fn tool_output_chunk_routes_o1_and_falls_back_on_stale_index() {
        let mut t = TranscriptState::new();
        t.apply_event(&AgentEvent::ToolCallStarted {
            call_id: "a".into(),
            tool: "bash".into(),
            input: serde_json::json!({ "command": "a" }),
        });
        t.apply_event(&AgentEvent::ToolCallStarted {
            call_id: "b".into(),
            tool: "bash".into(),
            input: serde_json::json!({ "command": "b" }),
        });
        let ia = t
            .blocks
            .iter()
            .position(|b| matches!(b, DisplayBlock::ToolRunning { call_id, .. } if call_id.as_str() == "a"))
            .expect("block a");
        let ib = t
            .blocks
            .iter()
            .position(|b| matches!(b, DisplayBlock::ToolRunning { call_id, .. } if call_id.as_str() == "b"))
            .expect("block b");
        assert_eq!(t.tool_running_blocks.get("a"), Some(&ia));
        assert_eq!(t.tool_running_blocks.get("b"), Some(&ib));

        // O(1) hit: the chunk lands on the mapped block.
        t.apply_event(&AgentEvent::ToolOutputChunk {
            call_id: "a".into(),
            delta: "A1".into(),
        });
        assert!(streamed_output_of(&t, ia).contains("A1"));

        // Forge a stale entry pointing at b's slot; the O(1) map must reject
        // it and the reverse-scan fallback must still find a's block.
        t.tool_running_blocks.insert("a".into(), ib);
        t.apply_event(&AgentEvent::ToolOutputChunk {
            call_id: "a".into(),
            delta: "A2".into(),
        });
        let a_out = streamed_output_of(&t, ia);
        assert!(
            a_out.contains("A1") && a_out.contains("A2"),
            "fallback must deliver to the real block: {a_out}"
        );
        assert!(
            streamed_output_of(&t, ib).is_empty(),
            "the unrelated block must not receive a's chunk"
        );

        // Completion evicts the map entry (and only that entry).
        t.apply_event(&AgentEvent::ToolCallCompleted {
            call_id: "a".into(),
            output: tool_result("a", "done"),
            duration_ms: 1,
        });
        assert!(
            !t.tool_running_blocks.contains_key("a"),
            "completion must evict the call_id entry"
        );
        assert!(
            t.tool_running_blocks.contains_key("b"),
            "an unrelated entry must stay"
        );
    }

    // ── ToolOutputChunk height growth without cache invalidation ──

    #[test]
    fn tool_output_chunk_grows_cached_height_without_bumping_generation() {
        let mut t = TranscriptState::new();
        t.apply_event(&AgentEvent::ToolCallStarted {
            call_id: "c".into(),
            tool: "bash".into(),
            input: serde_json::json!({ "command": "run" }),
        });
        let idx = t.blocks.len() - 1;
        let total_before = t.total_line_count(78); // builds the line cache
        assert!(
            t.line_cache
                .is_valid(t.blocks_generation, 78, t.blocks.len()),
            "fixture: cache must be built before chunks arrive"
        );
        let gen0 = t.blocks_generation;
        let h0 = t.line_cache.heights[idx];

        for i in 0..40 {
            t.apply_event(&AgentEvent::ToolOutputChunk {
                call_id: "c".into(),
                delta: format!("output line {i}\n"),
            });
        }

        assert_eq!(
            t.blocks_generation, gen0,
            "ToolOutputChunk must not bump blocks_generation"
        );
        assert!(
            t.line_cache.is_valid(gen0, 78, t.blocks.len()),
            "the incremental height patch must keep the cache valid"
        );
        let h1 = t.line_cache.heights[idx];
        assert!(
            h1 > h0,
            "cached height must grow with streamed output ({h0} -> {h1})"
        );
        let total_after = t.total_line_count(78);
        assert!(
            total_after > total_before,
            "total_line_count must reflect the grown height"
        );
        assert_eq!(
            total_after,
            t.line_cache.total(),
            "with no streaming blocks the total is the committed-block total"
        );
    }

    fn tool_result(call_id: &str, out: &str) -> ToolResult {
        ToolResult {
            timed_out: false,
            call_id: call_id.into(),
            success: true,
            output: out.into(),
            error: None,
        }
    }

    fn streamed_output_of(t: &TranscriptState, idx: usize) -> String {
        match &t.blocks[idx] {
            DisplayBlock::ToolRunning {
                streamed_output, ..
            } => streamed_output.clone(),
            other => panic!("expected ToolRunning at {idx}, got {other:?}"),
        }
    }
}
