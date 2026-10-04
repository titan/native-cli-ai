//! Hardened HTTP fetch tool.
//!
//! Semantics ported from oh-my-opencode-slim's smartfetch network layer:
//! bounded streaming reads, binary sniffing on raw bytes, a
//! header → meta → UTF-8 charset chain, and a strict redirect policy.

use std::time::Duration;

use futures_util::StreamExt;
use nca_common::config::WebConfig;
use nca_common::tool::{ToolCall, ToolDefinition, ToolResult};
use scraper::{Html, Selector};

use super::ToolExecutor;

/// Maximum number of redirect hops followed before the request is rejected.
const MAX_REDIRECT_HOPS: usize = 10;

/// Size of the leading byte window used for binary sniffing and meta charset
/// detection (mirrors the omos smartfetch sniff window).
const SNIFF_WINDOW: usize = 2048;

/// Fetches a URL over HTTP(S) and returns normalized text content.
pub struct FetchUrlTool {
    client: reqwest::Client,
    config: WebConfig,
}

impl FetchUrlTool {
    /// Build the tool from web config (timeout, user agent, fetch limits).
    pub fn new(config: WebConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .user_agent(config.user_agent.clone())
            .redirect(redirect_policy())
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { client, config }
    }
}

#[async_trait::async_trait]
impl ToolExecutor for FetchUrlTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            timeout_ms: Some(30_000),
            name: "fetch_url".into(),
            description: "Fetch and normalize the text content of a URL".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": { "type": "string" }
                },
                "required": ["url"]
            }),
        }
    }

    async fn execute(&self, call: &ToolCall) -> ToolResult {
        let url = call.input["url"].as_str().unwrap_or("").trim();
        if url.is_empty() {
            return error_result(call, "url is required".into());
        }

        let response = match self.client.get(url).send().await {
            Ok(response) => response,
            Err(err) => return error_result(call, format!("fetch failed: {err}")),
        };

        let status = response.status();
        if status.is_redirection() {
            // The custom redirect policy refused to follow this chain; explain
            // why and name the offending target.
            return error_result(call, explain_stopped_redirect(&response));
        }
        if !status.is_success() {
            return error_result(call, format!("unexpected status: {status}"));
        }

        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_string();

        // Hard byte cap of max_fetch_chars * 4 (UTF-8 headroom). One extra
        // byte is read so a body that fits the cap exactly is not flagged.
        let byte_cap = self.config.max_fetch_chars.saturating_mul(4);
        let read_limit = byte_cap.saturating_add(1);
        let raw = match read_body_bounded(response, read_limit).await {
            Ok(raw) => raw,
            Err(err) => {
                return error_result(call, format!("failed to read response body: {err}"));
            }
        };
        let truncated = raw.len() > byte_cap;
        let raw = &raw[..raw.len().min(byte_cap)];

        // Binary sniffing happens on the raw bytes, before any decoding —
        // binary content must never be whitespace-folded into the context.
        if looks_like_binary(raw) {
            let size_note = if truncated {
                format!("at least {}", raw.len())
            } else {
                raw.len().to_string()
            };
            let content_type = if content_type.is_empty() {
                "unknown"
            } else {
                content_type.as_str()
            };
            return error_result(
                call,
                format!(
                    "binary content ({size_note} bytes, content-type {content_type}) not loaded into context"
                ),
            );
        }

        let (text, warnings) = decode_body(raw, &content_type);

        let normalized = if content_type.contains("html") || text.contains("<html") {
            normalize_html(&text)
        } else {
            normalize_plain_text(&text)
        };

        let mut output = warnings.join("\n");
        if !warnings.is_empty() {
            output.push('\n');
        }
        output.extend(normalized.chars().take(self.config.max_fetch_chars));
        if truncated {
            output.push_str(&format!(
                "\n\n[fetch_url] truncated: body exceeded the {byte_cap}-byte read limit \
                 (max_fetch_chars * 4); output cut to the first {} characters.",
                self.config.max_fetch_chars
            ));
        }

        ToolResult {
            timed_out: false,
            call_id: call.id.clone(),
            success: true,
            output,
            error: None,
        }
    }
}

fn error_result(call: &ToolCall, message: String) -> ToolResult {
    ToolResult {
        timed_out: false,
        call_id: call.id.clone(),
        success: false,
        output: String::new(),
        error: Some(message),
    }
}

/// Redirect policy: at most [`MAX_REDIRECT_HOPS`] followed hops, http(s)
/// targets only, and no userinfo credentials in redirect URLs.
///
/// Violations `stop()` the chain (reqwest then returns the 3xx response as a
/// success) so the caller can report an error naming the offending target —
/// reqwest's error `Display` does not surface custom policy error payloads.
fn redirect_policy() -> reqwest::redirect::Policy {
    reqwest::redirect::Policy::custom(|attempt| {
        // `previous()` holds one entry per URL already requested in the chain,
        // so its length is the ordinal of the redirect being considered.
        if attempt.previous().len() > MAX_REDIRECT_HOPS {
            return attempt.stop();
        }
        let target = attempt.url();
        let scheme = target.scheme();
        if (scheme != "http" && scheme != "https")
            || !target.username().is_empty()
            || target.password().is_some()
        {
            return attempt.stop();
        }
        attempt.follow()
    })
}

/// Build the error message for a redirect chain the policy refused to follow.
/// The message always names the offending target URL.
fn explain_stopped_redirect(response: &reqwest::Response) -> String {
    let base = response.url();
    let Some(location) = response
        .headers()
        .get(reqwest::header::LOCATION)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|location| !location.is_empty())
    else {
        return format!("unexpected status: {}", response.status());
    };
    match base.join(location) {
        Ok(target) => {
            let scheme = target.scheme();
            if !target.username().is_empty() || target.password().is_some() {
                format!(
                    "redirect target '{target}' rejected: userinfo credentials \
                     are not allowed in redirect URLs"
                )
            } else if scheme != "http" && scheme != "https" {
                format!(
                    "redirect target '{target}' rejected: non-http(s) redirect \
                     scheme '{scheme}'"
                )
            } else {
                format!(
                    "redirect target '{target}' rejected: too many redirects \
                     (limit {MAX_REDIRECT_HOPS} hops)"
                )
            }
        }
        Err(_) => format!("redirect location '{location}' is not a valid URL relative to '{base}'"),
    }
}

/// Stream the response body into memory, stopping after `limit` bytes.
///
/// Reading stops as soon as the limit is reached — the rest of the body is
/// never buffered. The caller passes `cap + 1` so bodies larger than the cap
/// can be detected without reading them whole.
async fn read_body_bounded(
    response: reqwest::Response,
    limit: usize,
) -> Result<Vec<u8>, reqwest::Error> {
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while body.len() < limit {
        match stream.next().await {
            Some(Ok(chunk)) => {
                let take = (limit - body.len()).min(chunk.len());
                body.extend_from_slice(&chunk[..take]);
                if take < chunk.len() {
                    break; // more bytes remain beyond the limit
                }
            }
            Some(Err(err)) => return Err(err),
            None => break,
        }
    }
    Ok(body)
}

/// Binary sniffing on raw bytes (before any decoding): a NUL byte, or control
/// characters (excluding tab/LF/CR/FF) making up at least 2% of the first
/// [`SNIFF_WINDOW`] bytes, marks the body as binary.
fn looks_like_binary(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    if bytes.contains(&0) {
        return true;
    }
    let sample = &bytes[..bytes.len().min(SNIFF_WINDOW)];
    let controls = sample
        .iter()
        .filter(|&&b| b < 32 && !matches!(b, b'\t' | b'\n' | b'\r' | 0x0c))
        .count();
    controls * 100 >= sample.len() * 2
}

/// Decode raw bytes to text via the charset chain: a charset declared in the
/// Content-Type header → a `charset=` declaration in the first 2KB of the
/// body → strict UTF-8.
///
/// Only the UTF-8 family is actually honored; other declared charsets emit a
/// warning and the body is decoded as UTF-8 (strict, then lossy with a
/// warning) so no new decoding dependencies are needed.
fn decode_body(bytes: &[u8], content_type: &str) -> (String, Vec<String>) {
    let mut warnings = Vec::new();

    let declared = find_charset_declaration(content_type).or_else(|| sniff_meta_charset(bytes));

    if let Some(label) = declared.as_deref().filter(|label| !is_utf8_family(label)) {
        warnings.push(format!(
            "[fetch_url] warning: declared charset '{label}' is outside the \
             UTF-8 family; decoded as UTF-8"
        ));
    }

    match std::str::from_utf8(bytes) {
        Ok(text) => (text.to_string(), warnings),
        Err(err) => {
            warnings.push(format!(
                "[fetch_url] warning: response body is not valid UTF-8 ({err}); \
                 decoded with replacement characters"
            ));
            (String::from_utf8_lossy(bytes).into_owned(), warnings)
        }
    }
}

/// True for charset labels whose content decodes correctly as strict UTF-8
/// (ASCII labels are byte-compatible subsets of UTF-8).
fn is_utf8_family(label: &str) -> bool {
    matches!(
        label.trim().to_ascii_lowercase().as_str(),
        "" | "*"
            | "utf-8"
            | "utf8"
            | "utf-8-bom"
            | "unicode-1-1-utf-8"
            | "csutf8"
            | "us-ascii"
            | "ascii"
            | "ansi_x3.4-1968"
            | "iso-ir-6"
    )
}

/// Scan the first [`SNIFF_WINDOW`] raw bytes for a `charset=` declaration
/// such as `<meta charset="utf-8">`. Charset declarations are ASCII, so a
/// lossy decode of the window is safe for matching.
fn sniff_meta_charset(bytes: &[u8]) -> Option<String> {
    let window = &bytes[..bytes.len().min(SNIFF_WINDOW)];
    find_charset_declaration(&String::from_utf8_lossy(window))
}

/// Find the first `charset=<token>` declaration (case-insensitive) in `text`,
/// as used in Content-Type headers and HTML `<meta>` tags. Returns `None`
/// when no plausible token is present.
fn find_charset_declaration(text: &str) -> Option<String> {
    let lowered = text.to_ascii_lowercase();
    let bytes = lowered.as_bytes();
    // to_ascii_lowercase preserves byte offsets, so indices into `lowered`
    // are valid for slicing `text` as well.
    let mut cursor = 0;
    while let Some(offset) = lowered[cursor..].find("charset") {
        let start = cursor + offset;
        let mut i = skip_whitespace(bytes, start + "charset".len());
        if bytes.get(i) != Some(&b'=') {
            cursor = start + "charset".len();
            continue;
        }
        i = skip_whitespace(bytes, i + 1);
        if matches!(bytes.get(i), Some(b'"') | Some(b'\'')) {
            i += 1;
        }
        let token_start = i;
        while let Some(&b) = bytes.get(i) {
            if matches!(
                b,
                b'"' | b'\'' | b';' | b'>' | b'/' | b'=' | b',' | b'\0'..=b' '
            ) {
                break;
            }
            i += 1;
        }
        let token = text.get(token_start..i).map(str::trim).filter(|token| {
            !token.is_empty() && token.len() <= 40 && token.bytes().all(|b| b.is_ascii_graphic())
        });
        if let Some(token) = token {
            return Some(token.to_string());
        }
        cursor = start + "charset".len();
    }
    None
}

fn skip_whitespace(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

fn normalize_html(body: &str) -> String {
    let document = Html::parse_document(body);
    let title_selector = Selector::parse("title").unwrap();
    let body_selector = Selector::parse("body").unwrap();

    let title = document
        .select(&title_selector)
        .next()
        .map(|node| normalize_plain_text(&node.text().collect::<Vec<_>>().join(" ")))
        .filter(|text| !text.is_empty());
    let content = document
        .select(&body_selector)
        .next()
        .map(|node| normalize_plain_text(&node.text().collect::<Vec<_>>().join(" ")))
        .unwrap_or_default();

    match title {
        Some(title) if !content.is_empty() => format!("Title: {title}\n\n{content}"),
        Some(title) => title,
        None => content,
    }
}

fn normalize_plain_text(body: &str) -> String {
    body.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tiny_http::{Header, Request, Response, Server, StatusCode};

    use super::*;

    fn test_config(max_fetch_chars: usize) -> WebConfig {
        WebConfig {
            timeout_secs: 5,
            max_fetch_chars,
            default_search_limit: 5,
            user_agent: "nca-test".into(),
            search_endpoint: "http://127.0.0.1".into(),
        }
    }

    fn make_call(url: &str) -> ToolCall {
        ToolCall {
            id: "call-1".into(),
            name: "fetch_url".into(),
            input: serde_json::json!({ "url": url }),
        }
    }

    /// Spawn a local mock HTTP server; every incoming request is passed to
    /// `handler`. Returns the server's base URL. Tests never touch the real
    /// network — rejected redirect targets are never requested because the
    /// policy stops before following them.
    fn spawn_mock_server<F>(handler: F) -> String
    where
        F: Fn(Request) + Send + 'static,
    {
        let server = Server::http("127.0.0.1:0").expect("start mock server");
        let base_url = match server.server_addr() {
            tiny_http::ListenAddr::IP(addr) => format!("http://{addr}"),
            other => panic!("unsupported listen addr: {other:?}"),
        };
        std::thread::spawn(move || {
            for request in server.incoming_requests() {
                handler(request);
            }
        });
        base_url
    }

    /// Respond with an exact Content-Type header and raw body. Built from
    /// `Response::empty` so no implicit `charset=UTF-8` header is added.
    fn respond_with_body(request: Request, content_type: &str, body: Vec<u8>) {
        let len = body.len();
        let response = Response::empty(StatusCode(200))
            .with_header(
                Header::from_bytes("Content-Type", content_type).expect("content type header"),
            )
            .with_data(Cursor::new(body), Some(len));
        request.respond(response).expect("send response");
    }

    fn respond_text(request: Request, content_type: &str, body: &str) {
        respond_with_body(request, content_type, body.as_bytes().to_vec());
    }

    fn respond_redirect(request: Request, location: &str) {
        let response = Response::empty(StatusCode(302))
            .with_header(Header::from_bytes("Location", location).expect("location header"));
        request.respond(response).expect("send redirect");
    }

    async fn fetch(config: WebConfig, url: &str) -> ToolResult {
        FetchUrlTool::new(config).execute(&make_call(url)).await
    }

    #[tokio::test]
    async fn empty_url_is_rejected() {
        let result = fetch(test_config(1000), "").await;
        assert!(!result.success);
        assert_eq!(result.error.as_deref(), Some("url is required"));
    }

    #[tokio::test]
    async fn non_2xx_status_is_reported() {
        let base = spawn_mock_server(|request| {
            let response = Response::empty(StatusCode(404));
            request.respond(response).expect("send response");
        });
        let result = fetch(test_config(1000), &base).await;
        assert!(!result.success);
        let error = result.error.unwrap();
        assert!(error.contains("404"), "{error}");
    }

    #[tokio::test]
    async fn binary_body_with_nul_byte_is_rejected() {
        let base = spawn_mock_server(|request| {
            let mut body = vec![b'A'; 512];
            body[10] = 0;
            respond_with_body(request, "application/octet-stream", body);
        });
        let result = fetch(test_config(1000), &base).await;
        assert!(!result.success);
        let error = result.error.unwrap();
        assert!(error.contains("binary content"), "{error}");
        assert!(error.contains("not loaded into context"), "{error}");
        assert!(error.contains("application/octet-stream"), "{error}");
        assert!(error.contains("512"), "{error}");
    }

    #[tokio::test]
    async fn binary_body_by_control_char_ratio_is_rejected() {
        let base = spawn_mock_server(|request| {
            // 100 ESC bytes out of 116 total (no NUL anywhere) — the control
            // character ratio alone must trip the binary check.
            let mut body = vec![0x1b; 100];
            body.extend_from_slice(b"plain text tail");
            respond_with_body(request, "text/plain", body);
        });
        let result = fetch(test_config(1000), &base).await;
        assert!(!result.success);
        let error = result.error.unwrap();
        assert!(error.contains("binary content"), "{error}");
    }

    #[tokio::test]
    async fn text_with_tabs_newlines_and_form_feeds_is_not_binary() {
        let base = spawn_mock_server(|request| {
            let body = format!("{}\t\n\r\u{0c}still just text", "\t\n\r\u{0c}".repeat(50));
            respond_text(request, "text/plain", &body);
        });
        let result = fetch(test_config(10_000), &base).await;
        assert!(result.success, "{result:?}");
        assert!(
            result.output.contains("still just text"),
            "{}",
            result.output
        );
    }

    #[tokio::test]
    async fn oversized_body_is_truncated_and_reported() {
        let base = spawn_mock_server(|request| {
            respond_text(request, "text/plain", &"a".repeat(200));
        });
        // byte cap = 16 * 4 = 64; the 200-byte body must be cut at 64.
        let result = fetch(test_config(16), &base).await;
        assert!(result.success, "{result:?}");
        assert!(result.error.is_none());
        assert!(result.output.contains("truncated"), "{}", result.output);
        assert!(result.output.contains("64"), "{}", result.output);
        assert!(result.output.starts_with("aaaa"), "{}", result.output);
    }

    #[tokio::test]
    async fn body_exactly_at_byte_cap_is_not_truncated() {
        let base = spawn_mock_server(|request| {
            respond_text(request, "text/plain", &"b".repeat(64));
        });
        let result = fetch(test_config(16), &base).await;
        assert!(result.success, "{result:?}");
        assert!(!result.output.contains("truncated"), "{}", result.output);
        assert_eq!(result.output, "b".repeat(16));
    }

    #[tokio::test]
    async fn meta_charset_utf8_declared_in_body_is_honored() {
        let base = spawn_mock_server(|request| {
            // No charset in the Content-Type header: the meta declaration
            // inside the first 2KB must be sniffed from the raw bytes.
            respond_text(
                request,
                "text/html",
                "<html><head><meta charset=\"utf-8\"></head><body>héllo wörld</body></html>",
            );
        });
        let result = fetch(test_config(1000), &base).await;
        assert!(result.success, "{result:?}");
        assert!(result.output.contains("héllo wörld"), "{}", result.output);
        assert!(!result.output.contains("warning"), "{}", result.output);
    }

    #[tokio::test]
    async fn non_utf8_charset_warns_and_falls_back_to_utf8() {
        // Declared via the Content-Type header...
        let base = spawn_mock_server(|request| {
            respond_text(
                request,
                "text/html; charset=gb2312",
                "<html><body>héllo</body></html>",
            );
        });
        let result = fetch(test_config(1000), &base).await;
        assert!(result.success, "{result:?}");
        assert!(
            result.output.starts_with("[fetch_url] warning:"),
            "{}",
            result.output
        );
        assert!(result.output.contains("gb2312"), "{}", result.output);
        assert!(result.output.contains("héllo"), "{}", result.output);

        // ...and via a <meta charset=...> declaration when the header has none.
        let base = spawn_mock_server(|request| {
            respond_text(
                request,
                "text/html",
                "<html><head><meta charset='gb2312'></head><body>héllo</body></html>",
            );
        });
        let result = fetch(test_config(1000), &base).await;
        assert!(result.success, "{result:?}");
        assert!(
            result.output.starts_with("[fetch_url] warning:"),
            "{}",
            result.output
        );
        assert!(result.output.contains("gb2312"), "{}", result.output);
        assert!(result.output.contains("héllo"), "{}", result.output);
    }

    #[tokio::test]
    async fn invalid_utf8_without_declaration_decodes_lossy_with_warning() {
        let base = spawn_mock_server(|request| {
            let mut body = b"plain text ".to_vec();
            body.extend_from_slice(&[0xff, 0xfe]);
            body.extend_from_slice(b" tail");
            respond_with_body(request, "text/plain", body);
        });
        let result = fetch(test_config(1000), &base).await;
        assert!(result.success, "{result:?}");
        assert!(
            result.output.starts_with("[fetch_url] warning:"),
            "{}",
            result.output
        );
        assert!(
            result.output.contains("not valid UTF-8"),
            "{}",
            result.output
        );
        assert!(result.output.contains("plain text"), "{}", result.output);
        assert!(result.output.contains('\u{fffd}'), "{}", result.output);
    }

    #[tokio::test]
    async fn html_extraction_keeps_title_and_body_text() {
        let base = spawn_mock_server(|request| {
            respond_text(
                request,
                "text/html; charset=utf-8",
                "<html><head><title>Docs</title></head><body><p>hello world</p></body></html>",
            );
        });
        let result = fetch(test_config(1000), &base).await;
        assert!(result.success, "{result:?}");
        assert!(
            result.output.starts_with("Title: Docs"),
            "{}",
            result.output
        );
        assert!(result.output.contains("hello world"), "{}", result.output);
    }

    #[tokio::test]
    async fn redirect_with_userinfo_credentials_is_rejected() {
        let base = spawn_mock_server(|request| {
            // Never actually requested: the policy stops before following.
            respond_redirect(request, "http://user:secret@example.invalid/private");
        });
        let result = fetch(test_config(1000), &base).await;
        assert!(!result.success);
        let error = result.error.unwrap();
        assert!(
            error.contains("user:secret@example.invalid/private"),
            "{error}"
        );
        assert!(error.contains("userinfo"), "{error}");
    }

    #[tokio::test]
    async fn redirect_to_non_http_scheme_is_rejected() {
        let base = spawn_mock_server(|request| {
            respond_redirect(request, "ftp://example.invalid/file.zip");
        });
        let result = fetch(test_config(1000), &base).await;
        assert!(!result.success);
        let error = result.error.unwrap();
        assert!(error.contains("ftp://example.invalid/file.zip"), "{error}");
        assert!(error.contains("scheme"), "{error}");
    }

    #[tokio::test]
    async fn redirect_loop_stops_at_hop_limit() {
        let hits = Arc::new(AtomicUsize::new(0));
        let handler_hits = Arc::clone(&hits);
        let base = spawn_mock_server(move |request| {
            handler_hits.fetch_add(1, Ordering::SeqCst);
            // Redirect back to this same server: an endless chain.
            respond_redirect(request, "/");
        });
        let result = fetch(test_config(1000), &base).await;
        assert!(!result.success);
        let error = result.error.unwrap();
        assert!(error.contains("too many redirects"), "{error}");
        assert!(error.contains(&base), "{error}");
        // 1 original request + 10 followed hops, then the chain stops.
        assert_eq!(hits.load(Ordering::SeqCst), 11);
    }

    #[tokio::test]
    async fn ordinary_redirect_is_still_followed() {
        let base = spawn_mock_server(|request| {
            if request.url() == "/" {
                respond_redirect(request, "/final");
            } else {
                respond_text(request, "text/plain; charset=utf-8", "landed safely");
            }
        });
        let result = fetch(test_config(1000), &base).await;
        assert!(result.success, "{result:?}");
        assert!(result.output.contains("landed safely"), "{}", result.output);
    }
}
