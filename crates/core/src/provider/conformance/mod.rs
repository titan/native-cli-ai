//! Conformance fixtures + pinning harness for the OpenAI-compatible SSE parser.
//!
//! The Anthropic Messages wire format has a mirrored suite in [`anthropic`]
//! (see that module's docs for its fixtures and pinned gaps).
//!
//! ## Layout
//!
//! * `fixtures/*.txt` — raw on-the-wire SSE bytes (UTF-8), referenced with
//!   [`include_str!`]. A mistyped path is a *compile* error, so the fixture set
//!   can never silently go stale (and `cargo test` always compiles `cfg(test)`,
//!   so the guard rides the normal CI path).
//! * [`OPENAI_FIXTURES`] — the manifest. Every entry is consumed by at least
//!   one test (`include_str!` itself is the reference — no orphan fixtures).
//!
//! The whole module is gated by `#[cfg(test)]` at its declaration site in
//! `provider.rs`, so it never enters a release build.
//!
//! ## Driver
//!
//! Fixtures are driven through the *production* core
//! [`openai_compat::run_openai_sse`] — extracted from `spawn_openai_stream` —
//! with deterministic chunk-boundary splits (no network, no mock server), so
//! every pinned expectation exercises the shipping parser directly.
//!
//! [`real_parser_agrees_with_reference_on_whole_block`] is the end-to-end
//! guard: it pushes every ASCII fixture through the full `spawn_openai_stream`
//! path (tiny-http mock server, reqwest body stream) and asserts the
//! socket-fed output equals the in-memory reference, so the HTTP shell /
//! `ByteStreamError` adaptation layer cannot silently diverge from the core.
//!
//! ## Spec conformance (formerly pinned gaps — both fixed)
//!
//! 1. Multi-line `data:` payloads ARE re-joined per the SSE spec: an event's
//!    `data:` field may span multiple physical lines; the values are joined
//!    with `\n` and dispatched as ONE event when the terminating blank line
//!    arrives (an event still incomplete at EOF is discarded). When the join
//!    lands between JSON tokens the payload parses and the event is recovered
//!    (`fixture_multiline_recoverable`); when it lands inside an open JSON
//!    string literal the payload stays invalid JSON and the event is dropped
//!    (`fixture_multiline_data`).
//! 2. Lines are assembled in a raw byte buffer and each COMPLETE line is
//!    UTF-8 decoded separately (`0x0A` can never appear inside a multi-byte
//!    UTF-8 sequence, so slicing at `\n` bytes never splits a code point), so
//!    a network chunk boundary landing mid-UTF-8-sequence no longer corrupts
//!    the glyph into U+FFFD (`utf8_mid_character_split_roundtrips`).
//!
//! The remaining deliberate exclusion — `stop_reason` /
//! [`StreamChunk::Finish`] is not surfaced on the anthropic path — is
//! documented in the [`anthropic`] module docs.

use serde_json::Value;

mod anthropic;

use crate::provider::openai_compat::run_openai_sse;
use crate::provider::test_support::collect_chunks;
use crate::provider::{ByteStreamError, StreamChunk};

// ---------------------------------------------------------------------------
// Fixture manifest
// ---------------------------------------------------------------------------

const SIMPLE_TEXT_DONE: &str = include_str!("fixtures/simple_text_done.txt");
const MULTILINE_DATA: &str = include_str!("fixtures/multiline_data.txt");
const MULTILINE_RECOVERABLE: &str = include_str!("fixtures/multiline_recoverable.txt");
const CRLF_ENDINGS: &str = include_str!("fixtures/crlf_endings.txt");
const COMMENT_PING: &str = include_str!("fixtures/comment_ping.txt");
const UTF8_MULTIBYTE_DELTA: &str = include_str!("fixtures/utf8_multibyte_delta.txt");
const TOOL_CALL_FRAGMENTS: &str = include_str!("fixtures/tool_call_fragments.txt");
const FINISH_REASON_LAST_NONEMPTY: &str = include_str!("fixtures/finish_reason_last_nonempty.txt");
const USAGE_IN_FINAL: &str = include_str!("fixtures/usage_in_final.txt");

/// The OpenAI-side fixture manifest: `(name, raw wire bytes)`.
///
/// Consumed by the real-vs-reference bridge test; individual tests reference
/// the named nearest consts directly.
pub(crate) const OPENAI_FIXTURES: &[(&str, &str)] = &[
    ("simple_text_done", SIMPLE_TEXT_DONE),
    ("multiline_data", MULTILINE_DATA),
    ("multiline_recoverable", MULTILINE_RECOVERABLE),
    ("crlf_endings", CRLF_ENDINGS),
    ("comment_ping", COMMENT_PING),
    ("utf8_multibyte_delta", UTF8_MULTIBYTE_DELTA),
    ("tool_call_fragments", TOOL_CALL_FRAGMENTS),
    ("finish_reason_last_nonempty", FINISH_REASON_LAST_NONEMPTY),
    ("usage_in_final", USAGE_IN_FINAL),
];

// ---------------------------------------------------------------------------
// Deterministic chunk feeding
// ---------------------------------------------------------------------------

/// How a fixture's bytes are chopped into the byte-stream items the parser
/// consumes. Every variant is fully deterministic (no RNG, no wall-clock).
#[derive(Debug, Clone, Copy)]
enum Split {
    /// The whole fixture arrives as a single item.
    Whole,
    /// One item per line, `\n` included — models a line-buffered transport and
    /// keeps every chunk on a UTF-8 boundary.
    PerLine,
    /// One byte per item — worst-case packet fragmentation (may split UTF-8).
    ByteEvery,
    /// Fixed `n`-byte windows.
    Windows(usize),
    /// Variable 1..=`max`-byte windows from a fixed-seed LCG — models jittery
    /// TCP segmentation reproducibly.
    PseudoRandom { seed: u64, min: usize, max: usize },
}

fn split_bytes(bytes: &[u8], split: Split) -> Vec<Vec<u8>> {
    match split {
        Split::Whole => vec![bytes.to_vec()],
        Split::PerLine => {
            let mut out = Vec::new();
            let mut start = 0;
            for (i, b) in bytes.iter().enumerate() {
                if *b == b'\n' {
                    out.push(bytes[start..=i].to_vec());
                    start = i + 1;
                }
            }
            if start < bytes.len() {
                out.push(bytes[start..].to_vec());
            }
            out
        }
        Split::ByteEvery => bytes.iter().map(|b| vec![*b]).collect(),
        Split::Windows(n) => bytes.chunks(n.max(1)).map(<[u8]>::to_vec).collect(),
        Split::PseudoRandom { seed, min, max } => {
            let (min, max) = (min.max(1), max.max(min.max(1)));
            let mut out = Vec::new();
            let mut state = seed;
            let mut i = 0;
            while i < bytes.len() {
                // SplitMix64 step — deterministic, seed-driven.
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let span = min + (z >> 33) as usize % (max - min + 1);
                let end = (i + span).min(bytes.len());
                out.push(bytes[i..end].to_vec());
                i = end;
            }
            out
        }
    }
}

/// Drive the production parser core with `bytes` split per `split`, collecting
/// every emitted chunk (up to and including `Done`).
async fn drive(bytes: &str, split: Split) -> Vec<StreamChunk> {
    let items: Vec<Result<Vec<u8>, ByteStreamError>> = split_bytes(bytes.as_bytes(), split)
        .into_iter()
        .map(Ok)
        .collect();
    let stream = futures_util::stream::iter(items);
    let rx = run_openai_sse(stream, "conformance");
    collect_chunks(rx).await
}

// ---------------------------------------------------------------------------
// Expected-sequence projection
// ---------------------------------------------------------------------------

/// PartialEq-friendly projection of the chunk variants the fixtures exercise,
/// so failures print a precise sequence diff instead of a type dump.
#[derive(Debug, PartialEq)]
enum Expect {
    Text(String),
    Reasoning(String),
    Tool {
        id: String,
        name: String,
        input: Value,
    },
    Usage {
        input: u64,
        output: u64,
        cache_creation: u64,
        cache_read: u64,
    },
    Finish(String),
    Done,
    Error(String),
}

impl Expect {
    fn text(s: &str) -> Self {
        Self::Text(s.into())
    }

    fn finish(reason: &str) -> Self {
        Self::Finish(reason.into())
    }
}

fn project(chunk: &StreamChunk) -> Expect {
    match chunk {
        StreamChunk::TextDelta(t) => Expect::Text(t.clone()),
        StreamChunk::ReasoningDelta(t) => Expect::Reasoning(t.clone()),
        StreamChunk::ToolUse(call) => Expect::Tool {
            id: call.id.clone(),
            name: call.name.clone(),
            input: call.input.clone(),
        },
        StreamChunk::Usage {
            input_tokens,
            output_tokens,
            cache_creation_tokens,
            cache_read_tokens,
        } => Expect::Usage {
            input: *input_tokens,
            output: *output_tokens,
            cache_creation: *cache_creation_tokens,
            cache_read: *cache_read_tokens,
        },
        StreamChunk::Finish { reason } => Expect::Finish(reason.clone()),
        StreamChunk::Done => Expect::Done,
        StreamChunk::Error(err) => Expect::Error(err.to_string()),
    }
}

fn assert_sequence(name: &str, split: Split, actual: &[StreamChunk], expected: &[Expect]) {
    let got: Vec<Expect> = actual.iter().map(project).collect();
    assert_eq!(got, expected, "fixture `{name}` under split {split:?}");
}

// ---------------------------------------------------------------------------
// Fixture tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fixture_simple_text_done() {
    let expected = [
        Expect::text("Hello"),
        Expect::text(" world"),
        Expect::finish("stop"),
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
        let chunks = drive(SIMPLE_TEXT_DONE, split).await;
        assert_sequence("simple_text_done", split, &chunks, &expected);
    }
}

#[tokio::test]
async fn fixture_crlf_endings() {
    // `\r\n` line terminators must be trimmed exactly like `\n`.
    let expected = [
        Expect::text("Hello"),
        Expect::text(" world"),
        Expect::finish("stop"),
        Expect::Done,
    ];
    for split in [
        Split::Whole,
        Split::PerLine,
        Split::ByteEvery,
        Split::Windows(2),
    ] {
        let chunks = drive(CRLF_ENDINGS, split).await;
        assert_sequence("crlf_endings", split, &chunks, &expected);
    }
}

#[tokio::test]
async fn fixture_comment_ping() {
    // `: ping` keep-alive comment lines are ignored.
    let expected = [
        Expect::text("Hello"),
        Expect::text(" world"),
        Expect::finish("stop"),
        Expect::Done,
    ];
    for split in [
        Split::Whole,
        Split::PerLine,
        Split::ByteEvery,
        Split::Windows(5),
    ] {
        let chunks = drive(COMMENT_PING, split).await;
        assert_sequence("comment_ping", split, &chunks, &expected);
    }
}

#[tokio::test]
async fn fixture_multiline_data() {
    // Per the SSE spec the two `data:` values ARE re-joined with a raw `\n`
    // — but here the join lands INSIDE the open JSON string literal
    // (`..."content":"split` + `\n` + `ted"}}]}`), and a raw control
    // character inside a JSON string is still invalid JSON, so the
    // reassembled event is dropped. Only the subsequent complete event
    // survives. The recoverable variant (newline between JSON tokens) is
    // pinned by `fixture_multiline_recoverable`.
    let expected = [Expect::text("kept"), Expect::Done];
    for split in [
        Split::Whole,
        Split::PerLine,
        Split::ByteEvery,
        Split::Windows(4),
    ] {
        let chunks = drive(MULTILINE_DATA, split).await;
        assert_sequence("multiline_data", split, &chunks, &expected);
    }
}

#[tokio::test]
async fn fixture_multiline_recoverable() {
    // Fixed-behavior pin: when the SSE-spec `\n` join lands BETWEEN JSON
    // tokens (legal JSON whitespace), the reassembled payload parses and a
    // `data:` field split across physical lines dispatches as ONE event.
    let expected = [
        Expect::text("recovered"),
        Expect::text("kept"),
        Expect::Done,
    ];
    for split in [
        Split::Whole,
        Split::PerLine,
        Split::ByteEvery,
        Split::Windows(4),
    ] {
        let chunks = drive(MULTILINE_RECOVERABLE, split).await;
        assert_sequence("multiline_recoverable", split, &chunks, &expected);
    }
}

#[tokio::test]
async fn fixture_utf8_multibyte_delta() {
    // Content split across *events* (every chunk ends on a UTF-8 boundary) must
    // round-trip intact. Mid-character splits are pinned separately below.
    let expected = [
        Expect::text("你好，"),
        Expect::text("世界 "),
        Expect::text("🚀🎉"),
        Expect::finish("stop"),
        Expect::Done,
    ];
    for split in [Split::Whole, Split::PerLine] {
        let chunks = drive(UTF8_MULTIBYTE_DELTA, split).await;
        assert_sequence("utf8_multibyte_delta", split, &chunks, &expected);
    }
}

#[tokio::test]
async fn utf8_mid_character_split_roundtrips() {
    // Fixed-behavior pin: lines are assembled in a raw byte buffer and each
    // COMPLETE line is UTF-8 decoded separately (`0x0A` can never appear
    // inside a multi-byte UTF-8 sequence, so slicing at `\n` bytes never
    // splits a code point). A chunk boundary landing mid-UTF-8-sequence
    // therefore round-trips exactly — no U+FFFD replacement chars even under
    // worst-case byte-per-chunk fragmentation.
    let chunks = drive(UTF8_MULTIBYTE_DELTA, Split::ByteEvery).await;
    let text: String = chunks
        .iter()
        .filter_map(|chunk| match chunk {
            StreamChunk::TextDelta(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();

    assert_eq!(
        text, "你好，世界 🚀🎉",
        "byte-level splitting must round-trip the multibyte text exactly"
    );
}

#[tokio::test]
async fn fixture_tool_call_fragments() {
    // `id` appears only on the first fragment; `name`/`arguments` accumulate;
    // the call is flushed when `finish_reason: "tool_calls"` arrives.
    let expected = [
        Expect::Tool {
            id: "call_abc".into(),
            name: "read_file".into(),
            input: serde_json::json!({ "path": "src/main.rs" }),
        },
        Expect::finish("tool_calls"),
        Expect::Done,
    ];
    for split in [
        Split::Whole,
        Split::PerLine,
        Split::ByteEvery,
        Split::Windows(7),
    ] {
        let chunks = drive(TOOL_CALL_FRAGMENTS, split).await;
        assert_sequence("tool_call_fragments", split, &chunks, &expected);
    }
}

#[tokio::test]
async fn fixture_finish_reason_last_nonempty() {
    // Only the last non-empty `finish_reason` is surfaced ("stop" wins over the
    // earlier "length"; trailing nulls are ignored).
    let expected = [
        Expect::text("a"),
        Expect::text("b"),
        Expect::finish("stop"),
        Expect::Done,
    ];
    for split in [
        Split::Whole,
        Split::PerLine,
        Split::ByteEvery,
        Split::Windows(3),
    ] {
        let chunks = drive(FINISH_REASON_LAST_NONEMPTY, split).await;
        assert_sequence("finish_reason_last_nonempty", split, &chunks, &expected);
    }
}

#[tokio::test]
async fn fixture_usage_in_final() {
    // `stream_options.include_usage` shape: a usage-only event precedes `[DONE]`
    // and surfaces BEFORE the terminal finish reason.
    let expected = [
        Expect::text("hi"),
        Expect::Usage {
            input: 42,
            output: 7,
            cache_creation: 0,
            cache_read: 10,
        },
        Expect::finish("stop"),
        Expect::Done,
    ];
    for split in [
        Split::Whole,
        Split::PerLine,
        Split::ByteEvery,
        Split::Windows(5),
    ] {
        let chunks = drive(USAGE_IN_FINAL, split).await;
        assert_sequence("usage_in_final", split, &chunks, &expected);
    }
}

// ---------------------------------------------------------------------------
// End-to-end guard: socket-fed `spawn_openai_stream` vs the in-memory core
// ---------------------------------------------------------------------------

#[tokio::test]
async fn real_parser_agrees_with_reference_on_whole_block() {
    use std::path::Path;

    use nca_common::config::NcaConfig;
    use nca_common::message::Message;

    use crate::provider::Provider;
    use crate::provider::openai_compat::{CompatProfile, OpenAiCompatProvider};
    use crate::provider::test_support::spawn_sse_server;

    const PROFILE: CompatProfile = CompatProfile {
        name: "conformance",
        endpoint_suffix: "v1/chat/completions",
        strip_reasoning: false,
    };

    for (name, bytes) in OPENAI_FIXTURES {
        // The UTF-8 fixture is excluded: over a real socket the kernel/hyper may
        // split the body mid-character non-deterministically, which would make
        // this comparison flaky. That behavior is pinned deterministically by
        // `utf8_mid_character_split_roundtrips` instead.
        if *name == "utf8_multibyte_delta" {
            continue;
        }

        let base_url = spawn_sse_server((*bytes).to_string(), 200, |_| {});
        let mut config = NcaConfig::default();
        config.provider.openai.api_key = Some("conformance-key".into());
        config.provider.openai.base_url = base_url;

        let provider = OpenAiCompatProvider::from_config(
            &config.provider.openai,
            config.model.max_tokens,
            PROFILE,
            reqwest::header::HeaderMap::new(),
        )
        .expect("provider");

        let stream = provider
            .chat(&[Message::user("conformance")], &[], "", Path::new("."))
            .await
            .expect("chat stream");
        let real = collect_chunks(stream).await;

        let reference = drive(bytes, Split::Whole).await;

        let real_seq: Vec<Expect> = real.iter().map(project).collect();
        let reference_seq: Vec<Expect> = reference.iter().map(project).collect();
        assert_eq!(
            real_seq, reference_seq,
            "socket-fed parser diverged from in-memory core for fixture `{name}`"
        );
    }
}
