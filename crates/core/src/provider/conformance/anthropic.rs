//! Anthropic Messages API conformance fixtures.
//!
//! Mirror of `conformance::mod`'s OpenAI suite for the Anthropic wire format
//! (`event: <type>\ndata: <json>\n\n`). Fixtures are driven through the
//! *production* core [`run_anthropic_sse`] with deterministic chunk-boundary
//! splits (no network, no mock server), so every pinned expectation exercises
//! the shipping parser directly.
//!
//! ## Layout
//!
//! * `fixtures/anthropic/*.txt` — raw on-the-wire SSE bytes (UTF-8),
//!   referenced with [`include_str!`]. A mistyped path is a *compile* error, so
//!   the fixture set can never silently go stale.
//! * [`ANTHROPIC_FIXTURES`] — the manifest; every entry is consumed by at least
//!   one test.
//!
//! ## Known exclusions / gaps (pinned, not fixed)
//!
//! * `stop_reason` / [`StreamChunk::Finish`] is deliberately ignored on the
//!   anthropic path (known exclusion, out of scope). Pinned as the *absence* of
//!   any `Finish` chunk in `fixture_anthropic_simple_text_done`.
//! * GAP: multi-line `data:` payloads are not re-joined per the SSE spec — each
//!   physical `data:` line is parsed as standalone JSON, so both fragments of a
//!   split event are dropped silently
//!   (`fixture_anthropic_multiline_data`).
//! * GAP: `String::from_utf8_lossy` is applied per network chunk *before* lines
//!   are assembled, so a chunk boundary that lands mid-UTF-8-sequence corrupts
//!   the glyph into U+FFFD
//!   (`gap_anthropic_utf8_mid_character_split_is_corrupted`).

use serde_json::json;

use super::{Expect, Split, assert_sequence, split_bytes};
use crate::provider::anthropic_compat::run_anthropic_sse;
use crate::provider::test_support::collect_chunks;
use crate::provider::{ByteStreamError, StreamChunk};

// ---------------------------------------------------------------------------
// Fixture manifest
// ---------------------------------------------------------------------------

const SIMPLE_TEXT_DONE: &str = include_str!("fixtures/anthropic/simple_text_done.txt");
const TOOL_USE_ACCUMULATE: &str = include_str!("fixtures/anthropic/tool_use_accumulate.txt");
const MIXED_TEXT_TOOL_TEXT: &str = include_str!("fixtures/anthropic/mixed_text_tool_text.txt");
const PING_INTERLEAVED: &str = include_str!("fixtures/anthropic/ping_interleaved.txt");
const THINKING_DELTA: &str = include_str!("fixtures/anthropic/thinking_delta.txt");
const USAGE_CACHE_TOKENS: &str = include_str!("fixtures/anthropic/usage_cache_tokens.txt");
const CRLF_ENDINGS: &str = include_str!("fixtures/anthropic/crlf_endings.txt");
const MULTILINE_DATA: &str = include_str!("fixtures/anthropic/multiline_data.txt");
const UTF8_MULTIBYTE_DELTA: &str = include_str!("fixtures/anthropic/utf8_multibyte_delta.txt");
const ERROR_EVENT_IGNORED: &str = include_str!("fixtures/anthropic/error_event_ignored.txt");
const SYNTHETIC_TOOL_ID: &str = include_str!("fixtures/anthropic/synthetic_tool_id.txt");
const DONE_SENTINEL: &str = include_str!("fixtures/anthropic/done_sentinel.txt");

/// The Anthropic-side fixture manifest: `(name, raw wire bytes)`.
///
/// Consumed by `manifest_fixtures_all_terminate_with_done`; individual tests
/// reference the named nearest consts directly.
pub(crate) const ANTHROPIC_FIXTURES: &[(&str, &str)] = &[
    ("simple_text_done", SIMPLE_TEXT_DONE),
    ("tool_use_accumulate", TOOL_USE_ACCUMULATE),
    ("mixed_text_tool_text", MIXED_TEXT_TOOL_TEXT),
    ("ping_interleaved", PING_INTERLEAVED),
    ("thinking_delta", THINKING_DELTA),
    ("usage_cache_tokens", USAGE_CACHE_TOKENS),
    ("crlf_endings", CRLF_ENDINGS),
    ("multiline_data", MULTILINE_DATA),
    ("utf8_multibyte_delta", UTF8_MULTIBYTE_DELTA),
    ("error_event_ignored", ERROR_EVENT_IGNORED),
    ("synthetic_tool_id", SYNTHETIC_TOOL_ID),
    ("done_sentinel", DONE_SENTINEL),
];

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Drive the production anthropic parser core with `bytes` split per `split`,
/// collecting every emitted chunk (up to and including `Done`).
///
/// Shares the [`Split`]/[`split_bytes`] chunking machinery with the OpenAI
/// suite; only the parser core under test differs.
async fn drive_anthropic(bytes: &str, split: Split) -> Vec<StreamChunk> {
    let items: Vec<Result<Vec<u8>, ByteStreamError>> = split_bytes(bytes.as_bytes(), split)
        .into_iter()
        .map(Ok)
        .collect();
    let stream = futures_util::stream::iter(items);
    let rx = run_anthropic_sse(stream, "conformance-anthropic");
    collect_chunks(rx).await
}

// ---------------------------------------------------------------------------
// Fixture tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn manifest_fixtures_all_terminate_with_done() {
    // Every manifest entry must be non-empty and drive to a terminal `Done`
    // (and never emit a `Finish` — the anthropic path has no finish channel).
    for (name, bytes) in ANTHROPIC_FIXTURES {
        assert!(!bytes.is_empty(), "fixture `{name}` is empty");
        let chunks = drive_anthropic(bytes, Split::Whole).await;
        assert!(
            matches!(chunks.last(), Some(StreamChunk::Done)),
            "fixture `{name}` did not terminate with Done: {chunks:?}"
        );
        assert!(
            !chunks
                .iter()
                .any(|c| matches!(c, StreamChunk::Finish { .. })),
            "fixture `{name}` emitted a Finish chunk (anthropic path must not)"
        );
    }
}

#[tokio::test]
async fn fixture_anthropic_simple_text_done() {
    // message_start HOLDS input/cache tokens; message_delta combines them with
    // `usage.output_tokens` into a single Usage chunk.
    //
    // KNOWN EXCLUSION: `stop_reason` is ignored, so NO `Expect::Finish` appears
    // — this test pins its deliberate absence (the `stop_reason`/Finish
    // semantic gap is out of scope).
    let expected = [
        Expect::text("Hello"),
        Expect::text(" world"),
        Expect::Usage {
            input: 10,
            output: 5,
            cache_creation: 0,
            cache_read: 0,
        },
        Expect::Done,
    ];
    for split in [
        Split::Whole,
        Split::PerLine,
        Split::ByteEvery,
        Split::Windows(3),
        Split::PseudoRandom {
            seed: 0x1234_5678,
            min: 1,
            max: 4,
        },
    ] {
        let chunks = drive_anthropic(SIMPLE_TEXT_DONE, split).await;
        assert_sequence("anthropic/simple_text_done", split, &chunks, &expected);
    }
}

#[tokio::test]
async fn fixture_anthropic_tool_use_accumulate() {
    // `content_block_start` sets id/name; three `input_json_delta` fragments
    // accumulate; `content_block_stop` flushes ONE ToolUse with the parsed JSON.
    let expected = [
        Expect::Tool {
            id: "toolu_abc".into(),
            name: "read_file".into(),
            input: json!({ "path": "src/main.rs" }),
        },
        Expect::Usage {
            input: 12,
            output: 20,
            cache_creation: 0,
            cache_read: 0,
        },
        Expect::Done,
    ];
    for split in [
        Split::Whole,
        Split::PerLine,
        Split::ByteEvery,
        Split::Windows(7),
    ] {
        let chunks = drive_anthropic(TOOL_USE_ACCUMULATE, split).await;
        assert_sequence("anthropic/tool_use_accumulate", split, &chunks, &expected);
    }
}

#[tokio::test]
async fn fixture_anthropic_mixed_text_tool_text() {
    // Ordering pin: the ToolUse flushes at its own `content_block_stop`, BEFORE
    // the second text block's deltas.
    let expected = [
        Expect::text("before "),
        Expect::Tool {
            id: "toolu_mid".into(),
            name: "lookup".into(),
            input: json!({ "q": 1 }),
        },
        Expect::text("after"),
        Expect::Usage {
            input: 15,
            output: 9,
            cache_creation: 0,
            cache_read: 0,
        },
        Expect::Done,
    ];
    for split in [
        Split::Whole,
        Split::PerLine,
        Split::ByteEvery,
        Split::Windows(6),
    ] {
        let chunks = drive_anthropic(MIXED_TEXT_TOOL_TEXT, split).await;
        assert_sequence("anthropic/mixed_text_tool_text", split, &chunks, &expected);
    }
}

#[tokio::test]
async fn fixture_anthropic_ping_interleaved() {
    // `event: ping` frames are ignored — output is identical to the same stream
    // without pings (cf. `fixture_anthropic_simple_text_done`).
    let expected = [
        Expect::text("Hello"),
        Expect::text(" world"),
        Expect::Usage {
            input: 10,
            output: 5,
            cache_creation: 0,
            cache_read: 0,
        },
        Expect::Done,
    ];
    for split in [
        Split::Whole,
        Split::PerLine,
        Split::ByteEvery,
        Split::Windows(5),
    ] {
        let chunks = drive_anthropic(PING_INTERLEAVED, split).await;
        assert_sequence("anthropic/ping_interleaved", split, &chunks, &expected);
    }
}

#[tokio::test]
async fn fixture_anthropic_thinking_delta() {
    // `thinking_delta` frames surface as `ReasoningDelta`.
    let expected = [
        Expect::Reasoning("Let me think".into()),
        Expect::Reasoning(" step by step".into()),
        Expect::Usage {
            input: 8,
            output: 4,
            cache_creation: 0,
            cache_read: 0,
        },
        Expect::Done,
    ];
    for split in [
        Split::Whole,
        Split::PerLine,
        Split::ByteEvery,
        Split::Windows(5),
    ] {
        let chunks = drive_anthropic(THINKING_DELTA, split).await;
        assert_sequence("anthropic/thinking_delta", split, &chunks, &expected);
    }
}

#[tokio::test]
async fn fixture_anthropic_usage_cache_tokens() {
    // message_start cache_creation/cache_read tokens are held and re-surfaced
    // (with message_delta's output_tokens) in the combined Usage chunk.
    let expected = [
        Expect::text("cached"),
        Expect::Usage {
            input: 100,
            output: 7,
            cache_creation: 20,
            cache_read: 50,
        },
        Expect::Done,
    ];
    for split in [
        Split::Whole,
        Split::PerLine,
        Split::ByteEvery,
        Split::Windows(5),
    ] {
        let chunks = drive_anthropic(USAGE_CACHE_TOKENS, split).await;
        assert_sequence("anthropic/usage_cache_tokens", split, &chunks, &expected);
    }
}

#[tokio::test]
async fn fixture_anthropic_crlf_endings() {
    // `\r\n` line terminators must be trimmed exactly like `\n`.
    let expected = [
        Expect::text("Hello"),
        Expect::text(" world"),
        Expect::Usage {
            input: 10,
            output: 5,
            cache_creation: 0,
            cache_read: 0,
        },
        Expect::Done,
    ];
    for split in [
        Split::Whole,
        Split::PerLine,
        Split::ByteEvery,
        Split::Windows(2),
    ] {
        let chunks = drive_anthropic(CRLF_ENDINGS, split).await;
        assert_sequence("anthropic/crlf_endings", split, &chunks, &expected);
    }
}

#[tokio::test]
async fn fixture_anthropic_done_sentinel() {
    // A trailing `data: [DONE]` sentinel produces no chunk; the stream still
    // terminates with `Done`.
    let expected = [
        Expect::text("done-sentinel"),
        Expect::Usage {
            input: 4,
            output: 2,
            cache_creation: 0,
            cache_read: 0,
        },
        Expect::Done,
    ];
    for split in [
        Split::Whole,
        Split::PerLine,
        Split::ByteEvery,
        Split::Windows(5),
    ] {
        let chunks = drive_anthropic(DONE_SENTINEL, split).await;
        assert_sequence("anthropic/done_sentinel", split, &chunks, &expected);
    }
}

#[tokio::test]
async fn fixture_anthropic_synthetic_tool_id() {
    // A tool_use block with no `id` (compat endpoint) gets the synthetic
    // `tool-call-{seq}` id from `flush_anthropic_tool_call`.
    let expected = [
        Expect::Tool {
            id: "tool-call-0".into(),
            name: "lookup".into(),
            input: json!({ "path": "src" }),
        },
        Expect::Done,
    ];
    for split in [
        Split::Whole,
        Split::PerLine,
        Split::ByteEvery,
        Split::Windows(4),
    ] {
        let chunks = drive_anthropic(SYNTHETIC_TOOL_ID, split).await;
        assert_sequence("anthropic/synthetic_tool_id", split, &chunks, &expected);
    }
}

// ---------------------------------------------------------------------------
// Pinned gaps (behavior recorded, src NOT changed)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fixture_anthropic_multiline_data() {
    // GAP (pinned): a payload split across physical `data:` lines is NOT
    // re-joined per the SSE spec. Each physical `data:` line is parsed as
    // standalone JSON, so BOTH fragments (`{"type":...,"text":"split` and
    // `ted"}}`) fail to parse and are dropped silently; only the subsequent
    // complete event survives. Structurally identical to the OpenAI-side
    // `fixture_multiline_data` gap.
    let expected = [Expect::text("kept"), Expect::Done];
    for split in [
        Split::Whole,
        Split::PerLine,
        Split::ByteEvery,
        Split::Windows(4),
    ] {
        let chunks = drive_anthropic(MULTILINE_DATA, split).await;
        assert_sequence("anthropic/multiline_data", split, &chunks, &expected);
    }
}

#[tokio::test]
async fn fixture_anthropic_error_event_ignored() {
    // GAP (pinned): `event: error` with a valid JSON error payload falls into
    // the catch-all `_ => {}` arm and vanishes; the stream continues to `Done`
    // with no `StreamChunk::Error`. Silent-drop behavior recorded, not fixed.
    let expected = [
        Expect::text("ok"),
        Expect::Usage {
            input: 3,
            output: 1,
            cache_creation: 0,
            cache_read: 0,
        },
        Expect::Done,
    ];
    for split in [
        Split::Whole,
        Split::PerLine,
        Split::ByteEvery,
        Split::Windows(5),
    ] {
        let chunks = drive_anthropic(ERROR_EVENT_IGNORED, split).await;
        assert_sequence("anthropic/error_event_ignored", split, &chunks, &expected);
    }
}

#[tokio::test]
async fn fixture_anthropic_utf8_events_roundtrip() {
    // Content split across *events* (every event boundary is on a UTF-8
    // boundary) must round-trip intact. Mid-character splits are pinned
    // separately below.
    let expected = [
        Expect::text("你好，"),
        Expect::text("世界 "),
        Expect::text("🚀🎉"),
        Expect::Done,
    ];
    for split in [Split::Whole, Split::PerLine] {
        let chunks = drive_anthropic(UTF8_MULTIBYTE_DELTA, split).await;
        assert_sequence("anthropic/utf8_multibyte_delta", split, &chunks, &expected);
    }
}

#[tokio::test]
async fn gap_anthropic_utf8_mid_character_split_is_corrupted() {
    // GAP (pinned): `run_anthropic_sse` decodes each network chunk with
    // `String::from_utf8_lossy` BEFORE assembling lines. A chunk boundary that
    // lands inside a multi-byte UTF-8 sequence replaces the partial bytes with
    // U+FFFD, so the reassembled text is corrupted. Mirror of the OpenAI-side
    // `gap_utf8_mid_character_split_is_corrupted`. Reported, not fixed.
    let chunks = drive_anthropic(UTF8_MULTIBYTE_DELTA, Split::ByteEvery).await;
    let text: String = chunks
        .iter()
        .filter_map(|chunk| match chunk {
            StreamChunk::TextDelta(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();

    assert!(
        text.contains('\u{FFFD}'),
        "expected lossy replacement chars under byte-level splitting, got {text:?}"
    );
    assert_ne!(text, "你好，世界 🚀🎉");
}
