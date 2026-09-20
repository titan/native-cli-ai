//! Conformance fixtures + pinning harness for the OpenAI-compatible SSE parser.
//!
//! Phase 1 covers only the OpenAI `chat.completions` wire format. The
//! Anthropic side is deliberately deferred until the sibling extraction lane
//! lands its equivalent core.
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
//! ## Transitional driver (IMPORTANT)
//!
//! The production parser lives inside `openai_compat::spawn_openai_stream`,
//! which is welded to `reqwest::Response` — it has no byte-stream entry point
//! that a unit test can drive with controlled chunk boundaries. The sibling
//! lane is lifting the `bytes → StreamChunk` loop into a testable
//! `pub(crate) async fn run_openai_sse(byte_stream, provider_name) -> Receiver`.
//!
//! Until that core is available on this branch, [`run_openai_sse`] below is a
//! **transitional, behavior-faithful port** of the line loop in
//! `spawn_openai_stream` (the network/timeout branches are intentionally
//! omitted — a pre-loaded fixture stream never idles or errors). When the real
//! core lands, delete the mirror and point [`drive`] at it: the fixture bytes
//! and the expected sequences stay unchanged, only the `use` line moves.
//!
//! [`real_parser_agrees_with_reference_on_whole_block`] is the anti-drift
//! guard: it runs every ASCII fixture through the *real* `spawn_openai_stream`
//! (via the tiny-http mock server) and asserts the real output equals the
//! mirror's output, so the two cannot silently diverge.
//!
//! ## Known gaps (pinned, not fixed — see the two `gap_*` tests)
//!
//! 1. Multi-line `data:` payloads are **not** re-joined per the SSE spec: each
//!    physical `data:` line is parsed as standalone JSON, so a continuation
//!    line is dropped silently.
//! 2. `String::from_utf8_lossy` is applied per network chunk *before* lines are
//!    assembled, so a chunk boundary that lands mid-UTF-8-sequence corrupts the
//!    glyph into U+FFFD.

use std::collections::BTreeMap;

use futures_util::StreamExt;
use futures_util::stream::Stream;
use nca_common::tool::ToolCall;
use serde_json::Value;

use crate::provider::test_support::collect_chunks;
use crate::provider::{ProviderError, StreamChunk};

// ---------------------------------------------------------------------------
// Fixture manifest
// ---------------------------------------------------------------------------

const SIMPLE_TEXT_DONE: &str = include_str!("fixtures/simple_text_done.txt");
const MULTILINE_DATA: &str = include_str!("fixtures/multiline_data.txt");
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

/// Drive the reference parser with `bytes` split per `split`, collecting every
/// emitted chunk (up to and including `Done`).
async fn drive(bytes: &str, split: Split) -> Vec<StreamChunk> {
    let items: Vec<Result<Vec<u8>, ProviderError>> = split_bytes(bytes.as_bytes(), split)
        .into_iter()
        .map(Ok)
        .collect();
    let stream = futures_util::stream::iter(items);
    let rx = run_openai_sse(stream, "conformance").await;
    collect_chunks(rx).await
}

// ---------------------------------------------------------------------------
// Transitional reference core — mirror of openai_compat::spawn_openai_stream
// ---------------------------------------------------------------------------

#[derive(Default)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

/// Emit accumulated tool calls as `StreamChunk::ToolUse` — the happy path only
/// (valid accumulated JSON). The production original additionally has
/// input-repair / raw-passthrough fallbacks; the conformance fixtures all
/// accumulate to valid JSON, so those branches are out of scope here.
async fn flush_tool_calls(
    tx: &tokio::sync::mpsc::Sender<StreamChunk>,
    accumulators: &mut BTreeMap<u64, ToolCallAccumulator>,
) {
    for (index, call) in std::mem::take(accumulators) {
        if call.name.is_empty() {
            continue;
        }
        let Ok(input) = serde_json::from_str::<Value>(&call.arguments) else {
            continue;
        };
        let _ = tx
            .send(StreamChunk::ToolUse(ToolCall {
                id: if call.id.is_empty() {
                    format!("tool-call-{index}")
                } else {
                    call.id
                },
                name: call.name,
                input,
            }))
            .await;
    }
}

/// Transitional, behavior-faithful port of the SSE line loop inside
/// `openai_compat::spawn_openai_stream`.
///
/// SWAP ME: when `openai_compat::run_openai_sse` lands, delete this function
/// and import that one instead (`byte_stream` items become `Bytes`, so map with
/// `.map(|r| r.map(Bytes::from))`). Everything else stays put.
async fn run_openai_sse(
    byte_stream: impl Stream<Item = Result<Vec<u8>, ProviderError>>,
    provider_name: &str,
) -> tokio::sync::mpsc::Receiver<StreamChunk> {
    // Only referenced on the (omitted) error path in the production original.
    let _ = provider_name;

    futures_util::pin_mut!(byte_stream);
    let (tx, rx) = tokio::sync::mpsc::channel(64);

    let mut buffer = String::new();
    let mut tool_calls: BTreeMap<u64, ToolCallAccumulator> = BTreeMap::new();
    let mut finish_reason: Option<String> = None;

    while let Some(item) = byte_stream.next().await {
        let chunk = match item {
            Ok(chunk) => chunk,
            Err(err) => {
                let _ = tx.send(StreamChunk::Error(err)).await;
                return rx;
            }
        };

        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(nl) = buffer.find('\n') {
            let raw = buffer[..nl].to_string();
            buffer.drain(..=nl);
            let line = raw.trim_end_matches('\r').trim();

            if line.is_empty() || line.starts_with(':') {
                continue;
            }
            if !line.starts_with("data:") {
                continue;
            }

            let data = line["data:".len()..].trim();
            if data == "[DONE]" {
                flush_tool_calls(&tx, &mut tool_calls).await;
                if let Some(reason) = finish_reason.take() {
                    let _ = tx.send(StreamChunk::Finish { reason }).await;
                }
                let _ = tx.send(StreamChunk::Done).await;
                return rx;
            }

            let Ok(event) = serde_json::from_str::<Value>(data) else {
                continue;
            };

            if let Some(usage) = event.get("usage") {
                let input_tokens = usage["prompt_tokens"].as_u64().unwrap_or(0);
                let output_tokens = usage["completion_tokens"].as_u64().unwrap_or(0);
                let cached_tokens = usage
                    .get("prompt_tokens_details")
                    .and_then(|d| d.get("cached_tokens"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let (cache_creation_tokens, cache_read_tokens) = if cached_tokens > 0 {
                    (0, cached_tokens)
                } else if let Some(miss) = usage
                    .get("prompt_cache_miss_tokens")
                    .and_then(|v| v.as_u64())
                {
                    let hit = usage
                        .get("prompt_cache_hit_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    (miss, hit)
                } else {
                    (0, 0)
                };

                if input_tokens > 0 || output_tokens > 0 {
                    let _ = tx
                        .send(StreamChunk::Usage {
                            input_tokens,
                            output_tokens,
                            cache_creation_tokens,
                            cache_read_tokens,
                        })
                        .await;
                }
            }

            let Some(choices) = event["choices"].as_array() else {
                continue;
            };

            for choice in choices {
                let delta = &choice["delta"];
                if let Some(text) = delta["content"].as_str()
                    && !text.is_empty()
                {
                    let _ = tx.send(StreamChunk::TextDelta(text.to_string())).await;
                }

                if let Some(reasoning) = delta["reasoning_content"].as_str()
                    && !reasoning.is_empty()
                {
                    let _ = tx
                        .send(StreamChunk::ReasoningDelta(reasoning.to_string()))
                        .await;
                }

                if let Some(tool_deltas) = delta["tool_calls"].as_array() {
                    for tool_delta in tool_deltas {
                        let index = tool_delta["index"].as_u64().unwrap_or(0);
                        let entry = tool_calls.entry(index).or_default();
                        if let Some(id) = tool_delta["id"].as_str() {
                            entry.id = id.to_string();
                        }
                        if let Some(name) = tool_delta["function"]["name"].as_str() {
                            entry.name.push_str(name);
                        }
                        if let Some(arguments) = tool_delta["function"]["arguments"].as_str() {
                            entry.arguments.push_str(arguments);
                        }
                    }
                }

                if let Some(reason) = choice["finish_reason"].as_str()
                    && !reason.is_empty()
                {
                    finish_reason = Some(reason.to_string());
                }

                if choice["finish_reason"].as_str() == Some("tool_calls") {
                    flush_tool_calls(&tx, &mut tool_calls).await;
                }
            }
        }
    }

    flush_tool_calls(&tx, &mut tool_calls).await;
    if let Some(reason) = finish_reason.take() {
        let _ = tx.send(StreamChunk::Finish { reason }).await;
    }
    let _ = tx.send(StreamChunk::Done).await;
    rx
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
    // GAP #1 (pinned): a payload split across physical `data:` lines is NOT
    // re-joined. Both fragments fail standalone JSON parsing and vanish; only
    // the subsequent complete event survives.
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
// Pinned gaps (behavior recorded, src NOT changed)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn gap_utf8_mid_character_split_is_corrupted() {
    // GAP #2 (pinned): `spawn_openai_stream` decodes each network chunk with
    // `String::from_utf8_lossy` BEFORE assembling lines. A chunk boundary that
    // lands inside a multi-byte UTF-8 sequence replaces the partial bytes with
    // U+FFFD, so the reassembled text is corrupted. Reported, not fixed.
    let chunks = drive(UTF8_MULTIBYTE_DELTA, Split::ByteEvery).await;
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

// ---------------------------------------------------------------------------
// Anti-drift bridge: real `spawn_openai_stream` vs the transitional mirror
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
        // `gap_utf8_mid_character_split_is_corrupted` instead.
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
            "real parser diverged from transitional reference for fixture `{name}`"
        );
    }
}
