use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use nca_common::config::WebConfig;
use nca_common::tool::{ToolCall, ToolDefinition, ToolResult};
use scraper::{Html, Selector};
use std::time::Duration;

use super::ToolExecutor;

/// A realistic browser User-Agent.
///
/// Search engines (Bing included) reject requests carrying bot-like
/// identifiers, so `web_search` identifies itself as a browser rather than via
/// `WebConfig::user_agent` (which `fetch_url` keeps for honest identification).
const BROWSER_USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) \
     Chrome/124.0.0.0 Safari/537.36";

pub struct WebSearchTool {
    client: reqwest::Client,
    config: WebConfig,
}

impl WebSearchTool {
    pub fn new(config: WebConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            // The browser UA is set per-request in `execute()`; no client-level
            // UA is set here so the engine-facing identity stays explicit.
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { client, config }
    }
}

#[async_trait::async_trait]
impl ToolExecutor for WebSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "web_search".into(),
            description: "Search the public web and return titles, URLs, and snippets".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "limit": { "type": "integer" }
                },
                "required": ["query"]
            }),
        }
    }

    async fn execute(&self, call: &ToolCall) -> ToolResult {
        let query = call.input["query"].as_str().unwrap_or("").trim();
        let limit = call.input["limit"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or(self.config.default_search_limit)
            .clamp(1, 10);

        if query.is_empty() {
            return ToolResult {
                call_id: call.id.clone(),
                success: false,
                output: String::new(),
                error: Some("query is required".into()),
            };
        }

        let response = self
            .client
            .get(self.config.search_endpoint.as_str())
            .query(&[("q", query)])
            .header(reqwest::header::USER_AGENT, BROWSER_USER_AGENT)
            .header(
                reqwest::header::ACCEPT,
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            )
            .header(reqwest::header::ACCEPT_LANGUAGE, "en-US,en;q=0.9")
            .send()
            .await;

        let body = match response {
            Ok(response) => match response.text().await {
                Ok(body) => body,
                Err(err) => {
                    return ToolResult {
                        call_id: call.id.clone(),
                        success: false,
                        output: String::new(),
                        error: Some(format!("failed to read search response: {err}")),
                    };
                }
            },
            Err(err) => {
                return ToolResult {
                    call_id: call.id.clone(),
                    success: false,
                    output: String::new(),
                    error: Some(format!("search request failed: {err}")),
                };
            }
        };

        let rows = parse_search_results(&body, limit);

        if rows.is_empty() {
            let document = Html::parse_document(&body);
            let fallback = clean_text(&extract_text(&document));
            return ToolResult {
                call_id: call.id.clone(),
                success: false,
                output: fallback.chars().take(self.config.max_fetch_chars).collect(),
                error: Some("no structured search results parsed".into()),
            };
        }

        ToolResult {
            call_id: call.id.clone(),
            success: true,
            output: rows.join("\n"),
            error: None,
        }
    }
}

/// Parse a search engine results page (Bing HTML structure) into formatted rows.
///
/// Each natural result lives in an `<li class="b_algo">` with an `<h2><a>` title
/// link and a snippet paragraph. Result URLs are Bing redirect links
/// (`/ck/a?...&u=a1<base64url>`) and are decoded back to their real destinations.
fn parse_search_results(body: &str, limit: usize) -> Vec<String> {
    let document = Html::parse_document(body);
    let result_selector = Selector::parse("li.b_algo").unwrap();
    let title_selector = Selector::parse("h2 a").unwrap();
    let snippet_selector =
        Selector::parse("p.b_lineclamp2, div.b_caption p, .b_caption p").unwrap();

    let mut rows = Vec::new();
    for result in document.select(&result_selector).take(limit) {
        let title_node = match result.select(&title_selector).next() {
            Some(n) => n,
            None => continue,
        };

        let raw_href = title_node.value().attr("href").unwrap_or_default();
        // Sponsored results live in distinct `b_ad` containers (already excluded by
        // the `b_algo` selector), but `aclick` redirects are an extra safeguard.
        if raw_href.contains("bing.com/aclick") {
            continue;
        }

        let title = clean_text(&title_node.text().collect::<Vec<_>>().join(" "));
        let url = clean_url(raw_href);
        let snippet = result
            .select(&snippet_selector)
            .next()
            .map(|node| clean_text(&node.text().collect::<Vec<_>>().join(" ")))
            .unwrap_or_default();

        if !title.is_empty() && !url.is_empty() {
            rows.push(format!("- {title}\n  URL: {url}\n  Snippet: {snippet}"));
        }
    }
    rows
}

/// Resolve a (possibly Bing-redirected) href to its real destination URL.
///
/// Bing wraps organic links as `https://www.bing.com/ck/a?...&u=a1<base64url>`,
/// where the `u` parameter is the real URL base64url-encoded behind an `a1`
/// prefix. Non-redirect hrefs are returned unchanged.
fn clean_url(href: &str) -> String {
    if !href.contains("/ck/a?") {
        return href.to_string();
    }
    decode_bing_redirect(href).unwrap_or_else(|| href.to_string())
}

/// Decode the `u=a1<base64url>` parameter from a Bing `/ck/a` redirect href.
///
/// The `u` parameter sits in the query string; scraper/html5ever already
/// unescapes HTML entities in attribute values, so `&amp;` is already `&`.
fn decode_bing_redirect(href: &str) -> Option<String> {
    let query = href.split_once('?').map(|(_, q)| q).unwrap_or("");
    let encoded = query.split('&').find_map(|seg| seg.strip_prefix("u=a1"))?;
    // base64url without padding: strip any stray `=` then decode.
    let decoded = URL_SAFE_NO_PAD.decode(encoded.trim_end_matches('=')).ok()?;
    String::from_utf8(decoded).ok()
}

fn extract_text(document: &Html) -> String {
    let body_selector = Selector::parse("body").unwrap();
    document
        .select(&body_selector)
        .next()
        .map(|node| node.text().collect::<Vec<_>>().join(" "))
        .unwrap_or_default()
}

fn clean_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b64url(url: &str) -> String {
        URL_SAFE_NO_PAD.encode(url.as_bytes())
    }

    fn bing_html(results: &[(&str, &str, &str)]) -> String {
        let items: String = results
            .iter()
            .map(|(title, encoded_url, snippet)| {
                format!(
                    r#"<li class="b_algo">
                        <h2><a href="https://www.bing.com/ck/a?!&&p=abc&u=a1{encoded_url}&ntb=1">{title}</a></h2>
                        <div class="b_caption"><p>{snippet}</p></div>
                    </li>"#
                )
            })
            .collect();
        format!(r#"<html><body><ol id="b_results">{items}</ol></body></html>"#)
    }

    #[test]
    fn parses_bing_results_and_decodes_redirect_urls() {
        let html = bing_html(&[
            (
                "Rust Programming",
                &b64url("https://www.rust-lang.org/"),
                "A language empowering everyone",
            ),
            (
                "Cargo Book",
                &b64url("https://doc.rust-lang.org/cargo/"),
                "The Cargo Book",
            ),
        ]);
        let rows = parse_search_results(&html, 10);
        assert_eq!(rows.len(), 2);
        assert!(rows[0].contains("Rust Programming"));
        assert!(rows[0].contains("https://www.rust-lang.org/"));
        assert!(rows[0].contains("A language empowering everyone"));
        assert!(rows[1].contains("Cargo Book"));
        assert!(rows[1].contains("https://doc.rust-lang.org/cargo/"));
    }

    #[test]
    fn limits_results_to_requested_count() {
        let html = bing_html(&[
            ("A", &b64url("https://a.example/"), "sa"),
            ("B", &b64url("https://b.example/"), "sb"),
            ("C", &b64url("https://c.example/"), "sc"),
        ]);
        assert_eq!(parse_search_results(&html, 2).len(), 2);
    }

    #[test]
    fn skips_sponsored_aclick_results() {
        // The aclick link appears inside a b_algo container (defensive case);
        // it must be dropped while the real result is kept.
        let html = r#"<html><body><ol id="b_results">
            <li class="b_algo"><h2><a href="https://www.bing.com/aclick?xyz">Ad</a></h2>
                <div class="b_caption"><p>buy now</p></div></li>
            <li class="b_algo"><h2><a href="https://www.bing.com/ck/a?&u=a1REAL">Real</a></h2>
                <div class="b_caption"><p>real snippet</p></div></li>
        </ol></body></html>"#;
        let rows = parse_search_results(html, 10);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].contains("Real"));
    }

    #[test]
    fn non_redirect_urls_pass_through() {
        assert_eq!(
            clean_url("https://example.com/page"),
            "https://example.com/page"
        );
    }

    #[test]
    fn empty_page_yields_no_rows() {
        assert!(parse_search_results("<html><body></body></html>", 10).is_empty());
    }
}
