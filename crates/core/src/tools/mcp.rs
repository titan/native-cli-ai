use crate::tools::ToolExecutor;
use nca_common::config::{McpConfig, McpServerConfig};
use nca_common::tool::{ToolCall, ToolDefinition, ToolResult};
use rmcp::ServiceExt;
use rmcp::model::{CallToolRequestParams, ClientInfo, Implementation};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

/// Hard cap on the retained stderr tail, bytes. Overflow drops the HEAD and
/// keeps the TAIL (most recent diagnostics win — startup banners are noise,
/// the final panic/error line is the signal).
const STDERR_TAIL_LIMIT: usize = 64 * 1024;

/// Portion of the drained stderr embedded in error text, bytes (kept small
/// so a chatty server cannot flood tool-call error messages).
const STDERR_TAIL_IN_ERROR: usize = 4 * 1024;

/// Outcome of loading one MCP server during startup.
///
/// Aggregated per server so a single failing server degrades to "skipped"
/// instead of failing the whole [`load_mcp_tools`] call.
#[derive(Debug, Clone)]
pub struct McpServerLoadReport {
    /// Configured server name.
    pub name: String,
    /// Number of tools registered on success.
    pub tool_count: Option<usize>,
    /// Error text on failure (already includes the stderr tail when one was
    /// captured).
    pub error: Option<String>,
}

impl McpServerLoadReport {
    fn ok(name: &str, tool_count: usize) -> Self {
        Self {
            name: name.to_string(),
            tool_count: Some(tool_count),
            error: None,
        }
    }

    fn failed(name: &str, error: String) -> Self {
        Self {
            name: name.to_string(),
            tool_count: None,
            error: Some(error),
        }
    }
}

/// Load MCP tools from every enabled server in `config`.
///
/// Per-server Result aggregation: a server that fails to start, times out
/// (see `startup_timeout_ms`), or errors during tool discovery is skipped
/// with a failed [`McpServerLoadReport`] — the remaining servers still
/// register. Never returns `Err` for server-level failures.
pub async fn load_mcp_tools(
    workspace_root: &Path,
    config: &McpConfig,
) -> (Vec<Box<dyn ToolExecutor>>, Vec<McpServerLoadReport>) {
    let mut tools: Vec<Box<dyn ToolExecutor>> = Vec::new();
    let mut reports = Vec::new();
    for server in config.servers.iter().filter(|server| server.enabled) {
        match discover_server_tools(workspace_root, server, config).await {
            Ok(server_tools) => {
                reports.push(McpServerLoadReport::ok(&server.name, server_tools.len()));
                for tool in server_tools {
                    tools.push(Box::new(tool));
                }
            }
            Err(error) => {
                tracing::warn!(server = %server.name, "{error}");
                reports.push(McpServerLoadReport::failed(&server.name, error));
            }
        }
    }
    (tools, reports)
}

#[derive(Clone)]
pub struct McpTool {
    server: McpServerConfig,
    workspace_root: PathBuf,
    tool_name: String,
    description: Option<String>,
    parameters: Value,
    /// Call timeout snapshot (`[mcp] call_timeout_ms`); `0` disables.
    call_timeout_ms: u64,
}

impl McpTool {
    fn prefixed_name(&self) -> String {
        format!("mcp__{}__{}", self.server.name, self.tool_name)
    }
}

#[async_trait::async_trait]
impl ToolExecutor for McpTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            timeout_ms: None,
            name: self.prefixed_name(),
            description: self.description.clone().unwrap_or_else(|| {
                format!("MCP tool `{}` from `{}`", self.tool_name, self.server.name)
            }),
            parameters: self.parameters.clone(),
        }
    }

    async fn execute(&self, call: &ToolCall) -> ToolResult {
        let server = self.server.clone();
        let workspace_root = self.workspace_root.clone();
        let tool_name = self.tool_name.clone();
        let input = call.input.clone();
        let call_id = call.id.clone();
        let call_timeout_ms = self.call_timeout_ms;
        match execute_mcp_call(&workspace_root, &server, &tool_name, input, call_timeout_ms).await {
            Ok(output) => ToolResult {
                timed_out: false,
                call_id,
                success: true,
                output,
                error: None,
            },
            Err(error) => ToolResult {
                timed_out: false,
                call_id,
                success: false,
                output: String::new(),
                error: Some(error),
            },
        }
    }
}

/// Normalize one MCP `inputSchema` into a provider-safe parameter schema.
///
/// MCP servers routinely omit `required` (or send it explicitly as `null`).
/// Constructing the parameters object with `schema.get("required")` serializes
/// the absent case as JSON `null`, and strict providers reject the whole
/// request — DeepSeek answers `Invalid schema for function …: null is not of
/// type "array"` for a single offending tool. Every `null` object member is
/// dropped (recursively, including nested object schemas), then the
/// object-schema invariants providers validate are normalized: `type` is
/// forced to `"object"`, `properties` defaults to `{}`, and `required`
/// defaults to `[]`.
///
/// Null members are only stripped from objects, never from arrays: `"enum":
/// [null, "a"]` is meaningful and must survive intact.
fn normalize_mcp_parameters(schema: &Value) -> Value {
    let mut parameters = match schema {
        Value::Object(map) => Value::Object(map.clone()),
        _ => Value::Object(serde_json::Map::new()),
    };
    strip_null_members(&mut parameters);

    if let Some(map) = parameters.as_object_mut() {
        map.insert("type".to_string(), Value::String("object".to_string()));
        if !matches!(map.get("properties"), Some(Value::Object(_))) {
            map.insert("properties".to_string(), Value::Object(Default::default()));
        }
        if !matches!(map.get("required"), Some(Value::Array(_))) {
            map.insert("required".to_string(), Value::Array(Vec::new()));
        }
    }

    parameters
}

/// Recursively drop `null`-valued members from JSON objects.
fn strip_null_members(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.retain(|_, member| !member.is_null());
            for member in map.values_mut() {
                strip_null_members(member);
            }
        }
        Value::Array(items) => {
            for item in items.iter_mut() {
                strip_null_members(item);
            }
        }
        _ => {}
    }
}

/// Bounded stderr tail buffer shared with the background drain task.
type StderrTail = Arc<Mutex<Vec<u8>>>;

/// Spawn a child process for an MCP server.
///
/// stderr is `Stdio::piped()` — never inherited: an MCP server's stray
/// writes must not corrupt the TUI. A background task drains the pipe into
/// `StderrTail` (bounded, drop-head-keep-tail) so the pipe never fills and
/// deadlocks the child, and the tail is available for error reporting.
fn spawn_mcp_server(
    server: &McpServerConfig,
    workspace_root: &Path,
) -> Result<
    (
        tokio::process::ChildStdout,
        tokio::process::ChildStdin,
        tokio::process::Child,
        StderrTail,
    ),
    String,
> {
    let mut cmd = Command::new(&server.command);
    cmd.args(&server.args)
        .envs(&server.env)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let working_dir = server
        .cwd
        .clone()
        .unwrap_or_else(|| workspace_root.to_path_buf());
    cmd.current_dir(working_dir);

    let mut child = cmd
        .spawn()
        .map_err(|err| format!("failed to start MCP server `{}`: {err}", server.name))?;

    // The take() failures below drop `child` — kill it first so a spawn
    // that succeeded but came up without a stdio pipe never leaks a
    // live process (caller gets Err and cannot clean up itself).
    let stdin = match child.stdin.take() {
        Some(stdin) => stdin,
        None => {
            let _ = child.start_kill();
            return Err(format!("missing stdin for MCP server `{}`", server.name));
        }
    };
    let stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            let _ = child.start_kill();
            return Err(format!("missing stdout for MCP server `{}`", server.name));
        }
    };
    let stderr = match child.stderr.take() {
        Some(stderr) => stderr,
        None => {
            let _ = child.start_kill();
            return Err(format!("missing stderr for MCP server `{}`", server.name));
        }
    };
    let tail: StderrTail = Arc::new(Mutex::new(Vec::new()));
    let drain_tail = Arc::clone(&tail);
    tokio::spawn(async move {
        let mut reader = stderr;
        let mut chunk = [0u8; 4096];
        loop {
            match reader.read(&mut chunk).await {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if let Ok(mut buf) = drain_tail.lock() {
                        buf.extend_from_slice(&chunk[..n]);
                        if buf.len() > STDERR_TAIL_LIMIT {
                            let excess = buf.len() - STDERR_TAIL_LIMIT;
                            buf.drain(..excess);
                        }
                    }
                }
            }
        }
    });

    Ok((stdout, stdin, child, tail))
}

/// Format the captured stderr tail for inclusion in an error message.
/// Empty tails contribute nothing; tails are truncated to
/// [`STDERR_TAIL_IN_ERROR`] bytes.
fn stderr_context(tail: &StderrTail) -> String {
    let snapshot = tail.lock().map(|buf| buf.clone()).unwrap_or_default();
    if snapshot.is_empty() {
        return String::new();
    }
    let start = snapshot.len().saturating_sub(STDERR_TAIL_IN_ERROR);
    let slice = &snapshot[start..];
    let text = String::from_utf8_lossy(slice);
    format!("; stderr tail: {text}")
}

/// Kill the MCP server child and collect its stderr tail for the error
/// path, allowing the background drain a brief grace window first.
///
/// `stderr_context` alone races the drain task: a diagnostic emitted at the
/// timeout instant may not have been read into the tail buffer yet when the
/// error text is formatted. `start_kill` followed by a short sleep lets the
/// drain task consume the pipe's final bytes (EOF after kill) before the
/// snapshot is taken. Callers keep their `client.cancel()` awaits where they
/// exist today — this helper owns only the kill + collect concern.
async fn kill_and_collect_stderr(child: &mut tokio::process::Child, tail: &StderrTail) -> String {
    let _ = child.start_kill();
    tokio::time::sleep(Duration::from_millis(50)).await;
    stderr_context(tail)
}

/// `ms` as an optional `tokio::time::timeout` budget. `0` disables the
/// timeout entirely (legacy behavior).
fn timeout_budget(ms: u64) -> Option<Duration> {
    (ms > 0).then(|| Duration::from_millis(ms))
}

async fn discover_server_tools(
    workspace_root: &Path,
    server: &McpServerConfig,
    config: &McpConfig,
) -> Result<Vec<McpTool>, String> {
    let startup_timeout_ms = config.startup_timeout_ms;
    let (stdout, stdin, mut child, tail) = spawn_mcp_server(server, workspace_root)?;

    let client_info = ClientInfo::new(
        rmcp::model::ClientCapabilities::default(),
        Implementation::new("nca", env!("CARGO_PKG_VERSION")),
    );
    let serve = client_info.serve((stdout, stdin));
    let client = match timeout_budget(startup_timeout_ms) {
        Some(budget) => match tokio::time::timeout(budget, serve).await {
            Ok(Ok(client)) => client,
            Ok(Err(err)) => {
                let stderr = kill_and_collect_stderr(&mut child, &tail).await;
                return Err(format!(
                    "MCP server `{}` init failed: {err}{stderr}",
                    server.name
                ));
            }
            Err(_) => {
                let stderr = kill_and_collect_stderr(&mut child, &tail).await;
                return Err(format!(
                    "MCP server `{}` init timed out after {startup_timeout_ms}ms (server killed; \
                     raise [mcp] startup_timeout_ms if this server is slow to start){stderr}",
                    server.name
                ));
            }
        },
        None => match serve.await {
            Ok(client) => client,
            Err(err) => {
                let stderr = kill_and_collect_stderr(&mut child, &tail).await;
                return Err(format!(
                    "MCP server `{}` init failed: {err}{stderr}",
                    server.name
                ));
            }
        },
    };

    let list = client.list_all_tools();
    let tools = match timeout_budget(startup_timeout_ms) {
        Some(budget) => match tokio::time::timeout(budget, list).await {
            Ok(Ok(tools)) => tools,
            Ok(Err(err)) => {
                let stderr = kill_and_collect_stderr(&mut child, &tail).await;
                let _ = client.cancel().await;
                return Err(format!(
                    "MCP server `{}` list_tools failed: {err}{stderr}",
                    server.name
                ));
            }
            Err(_) => {
                let stderr = kill_and_collect_stderr(&mut child, &tail).await;
                let _ = client.cancel().await;
                return Err(format!(
                    "MCP server `{}` list_tools timed out after {startup_timeout_ms}ms (server \
                     killed){stderr}",
                    server.name
                ));
            }
        },
        None => match list.await {
            Ok(tools) => tools,
            Err(err) => {
                let stderr = kill_and_collect_stderr(&mut child, &tail).await;
                let _ = client.cancel().await;
                return Err(format!(
                    "MCP server `{}` list_tools failed: {err}{stderr}",
                    server.name
                ));
            }
        },
    };

    let result: Vec<McpTool> = tools
        .into_iter()
        .map(|tool| {
            let parameters =
                normalize_mcp_parameters(&Value::Object(tool.input_schema.as_ref().clone()));

            McpTool {
                server: server.clone(),
                workspace_root: workspace_root.to_path_buf(),
                tool_name: tool.name.to_string(),
                description: tool.description.map(|d| d.to_string()),
                parameters,
                call_timeout_ms: config.call_timeout_ms,
            }
        })
        .collect();

    // rmcp handles transport close on cancel; kill child to avoid zombies
    let _ = child.start_kill();
    let _ = client.cancel().await;
    Ok(result)
}

async fn execute_mcp_call(
    workspace_root: &Path,
    server: &McpServerConfig,
    tool_name: &str,
    input: Value,
    call_timeout_ms: u64,
) -> Result<String, String> {
    let (stdout, stdin, mut child, tail) = spawn_mcp_server(server, workspace_root)?;

    let client_info = ClientInfo::new(
        rmcp::model::ClientCapabilities::default(),
        Implementation::new("nca", env!("CARGO_PKG_VERSION")),
    );
    let serve = client_info.serve((stdout, stdin));
    let client = match timeout_budget(call_timeout_ms) {
        Some(budget) => match tokio::time::timeout(budget, serve).await {
            Ok(Ok(client)) => client,
            Ok(Err(err)) => {
                let stderr = kill_and_collect_stderr(&mut child, &tail).await;
                return Err(format!(
                    "MCP server `{}` init failed: {err}{stderr}",
                    server.name
                ));
            }
            Err(_) => {
                let stderr = kill_and_collect_stderr(&mut child, &tail).await;
                return Err(format!(
                    "MCP server `{}` init timed out after {call_timeout_ms}ms (server \
                     killed){stderr}",
                    server.name
                ));
            }
        },
        None => match serve.await {
            Ok(client) => client,
            Err(err) => {
                let stderr = kill_and_collect_stderr(&mut child, &tail).await;
                return Err(format!(
                    "MCP server `{}` init failed: {err}{stderr}",
                    server.name
                ));
            }
        },
    };

    let arguments = input.as_object().cloned().unwrap_or_default();

    let call = client
        .call_tool(CallToolRequestParams::new(tool_name.to_owned()).with_arguments(arguments));
    let result = match timeout_budget(call_timeout_ms) {
        Some(budget) => match tokio::time::timeout(budget, call).await {
            Ok(Ok(result)) => result,
            Ok(Err(err)) => {
                let stderr = kill_and_collect_stderr(&mut child, &tail).await;
                let _ = client.cancel().await;
                return Err(format!(
                    "MCP tool `{}` call failed: {err}{stderr}",
                    tool_name
                ));
            }
            Err(_) => {
                let stderr = kill_and_collect_stderr(&mut child, &tail).await;
                let _ = client.cancel().await;
                return Err(format!(
                    "MCP tool `{}` call timed out after {call_timeout_ms}ms (server killed){stderr}",
                    tool_name
                ));
            }
        },
        None => match call.await {
            Ok(result) => result,
            Err(err) => {
                let stderr = kill_and_collect_stderr(&mut child, &tail).await;
                let _ = client.cancel().await;
                return Err(format!(
                    "MCP tool `{}` call failed: {err}{stderr}",
                    tool_name
                ));
            }
        },
    };

    let _ = child.start_kill();
    let _ = client.cancel().await;

    let output: Vec<String> = result
        .content
        .iter()
        .filter_map(|c| c.as_text().map(|t| t.text.clone()))
        .collect();
    serde_json::to_string(&output).map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hanging_server(name: &str, extra_args: &[&str]) -> McpServerConfig {
        let mut args = vec!["-c".to_string()];
        args.extend(extra_args.iter().map(|s| s.to_string()));
        McpServerConfig {
            name: name.to_string(),
            command: "sh".to_string(),
            args,
            enabled: true,
            env: Default::default(),
            cwd: None,
        }
    }

    /// A hanging server must fail within the timeout budget (not hang the
    /// startup path), the error must carry the timeout hint, and the child
    /// must be killed (no orphan).
    #[tokio::test(flavor = "multi_thread")]
    async fn discover_times_out_on_hanging_server() {
        let dir = tempfile::tempdir().expect("tempdir");
        let server = hanging_server("hang", &["sleep 30"]);
        let mut config = McpConfig::default();
        config.startup_timeout_ms = 500;

        let started = std::time::Instant::now();
        let err = match discover_server_tools(dir.path(), &server, &config).await {
            Ok(_) => panic!("hanging server must fail"),
            Err(err) => err,
        };
        let elapsed = started.elapsed();

        assert!(
            err.contains("timed out"),
            "error must carry the timeout hint, got: {err}"
        );
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "must return well before the 30s sleep elapses, took {elapsed:?}"
        );
    }

    /// stderr written before the hang must surface in the error text — the
    /// bunx-style failure mode where the useful diagnostic lands on stderr.
    #[tokio::test(flavor = "multi_thread")]
    async fn discover_includes_stderr_tail_in_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let server = hanging_server(
            "noisy-hang",
            &["echo 'NCA_TEST_STDERR_MARKER' >&2; sleep 30"],
        );
        let mut config = McpConfig::default();
        config.startup_timeout_ms = 500;

        let err = match discover_server_tools(dir.path(), &server, &config).await {
            Ok(_) => panic!("hanging server must fail"),
            Err(err) => err,
        };
        assert!(
            err.contains("NCA_TEST_STDERR_MARKER"),
            "stderr marker must be embedded in the error, got: {err}"
        );
    }

    /// Per-server aggregation: one failing server must not fail the whole
    /// load — a failing report is recorded and zero tools returned.
    #[tokio::test(flavor = "multi_thread")]
    async fn load_mcp_tools_aggregates_failures_per_server() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = McpConfig::default();
        config.startup_timeout_ms = 300;
        config.servers = vec![hanging_server("rea", &["sleep 30"])];

        let started = std::time::Instant::now();
        let (tools, reports) = load_mcp_tools(dir.path(), &config).await;
        let elapsed = started.elapsed();

        assert!(tools.is_empty(), "no tools from a hanging server");
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].name, "rea");
        assert!(
            reports[0]
                .error
                .as_deref()
                .unwrap_or("")
                .contains("timed out"),
            "report must carry the timeout error, got: {:?}",
            reports[0].error
        );
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "load must return within the budget, took {elapsed:?}"
        );
    }

    /// MCP server schema with no `required` must not leak `null` onto the
    /// wire — strict providers (DeepSeek) reject the whole request with
    /// `null is not of type "array"`, taking every other tool down with it.
    #[test]
    fn normalize_parameters_defaults_missing_required_to_empty_array() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "path": { "type": "string" } }
        });
        let parameters = normalize_mcp_parameters(&schema);

        assert_eq!(parameters["type"], "object");
        assert_eq!(parameters["required"], serde_json::json!([]));
        assert_eq!(parameters["properties"]["path"]["type"], "string");
    }

    /// Explicit `"required": null` (seen from JVM/JS MCP servers) is
    /// normalized the same way as an absent key.
    #[test]
    fn normalize_parameters_rewrites_explicit_null_required() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {},
            "required": null
        });
        let parameters = normalize_mcp_parameters(&schema);
        assert_eq!(parameters["required"], serde_json::json!([]));
    }

    /// A schema with neither `properties` nor `required` (freerouting's
    /// `create_session`) becomes a valid empty object schema.
    #[test]
    fn normalize_parameters_handles_empty_schema() {
        let parameters = normalize_mcp_parameters(&serde_json::json!({}));
        assert_eq!(parameters["type"], "object");
        assert_eq!(parameters["properties"], serde_json::json!({}));
        assert_eq!(parameters["required"], serde_json::json!([]));
    }

    /// Nested object schemas are sanitized too — one leaking `null` inside
    /// `properties` fails validation just as hard as the top level.
    #[test]
    fn normalize_parameters_strips_nested_nulls() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "opts": {
                    "type": "object",
                    "properties": { "depth": { "type": "integer" } },
                    "required": null,
                    "description": null
                }
            },
            "required": ["opts"]
        });
        let parameters = normalize_mcp_parameters(&schema);
        let opts = &parameters["properties"]["opts"];

        assert!(opts.get("required").is_none(), "nested null removed");
        assert!(opts.get("description").is_none());
        assert_eq!(opts["properties"]["depth"]["type"], "integer");
        assert_eq!(parameters["required"], serde_json::json!(["opts"]));
    }

    /// `null` inside an array is semantic (`enum: [null, "a"]`) and must not
    /// be stripped — only null *members* of objects are dropped.
    #[test]
    fn normalize_parameters_keeps_null_inside_arrays() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "mode": { "enum": [null, "wire", "via"] }
            },
            "required": ["mode"]
        });
        let parameters = normalize_mcp_parameters(&schema);
        assert_eq!(
            parameters["properties"]["mode"]["enum"],
            serde_json::json!([null, "wire", "via"])
        );
    }

    /// Rich schema details (descriptions, defaults, enums, nested $defs
    /// references) survive normalization — the parameters are passed through
    /// rather than rebuilt from three keys.
    #[test]
    fn normalize_parameters_preserves_schema_details() {
        let schema = serde_json::json!({
            "type": "object",
            "$defs": { "Layer": { "type": "string" } },
            "properties": {
                "layer": {
                    "type": "string",
                    "description": "Copper layer",
                    "enum": ["F.Cu", "B.Cu"],
                    "default": "F.Cu"
                }
            },
            "required": ["layer"]
        });
        let parameters = normalize_mcp_parameters(&schema);

        assert_eq!(parameters["$defs"]["Layer"]["type"], "string");
        assert_eq!(
            parameters["properties"]["layer"]["description"],
            "Copper layer"
        );
        assert_eq!(parameters["properties"]["layer"]["default"], "F.Cu");
        assert_eq!(parameters["required"], serde_json::json!(["layer"]));
    }

    /// Disabled servers are skipped entirely — no report, no spawn.
    #[tokio::test]
    async fn load_mcp_tools_skips_disabled_servers() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut server = hanging_server("off", &["sleep 30"]);
        server.enabled = false;
        let mut config = McpConfig::default();
        config.startup_timeout_ms = 50;
        config.servers = vec![server];

        let (tools, reports) = load_mcp_tools(dir.path(), &config).await;
        assert!(tools.is_empty());
        assert!(reports.is_empty(), "disabled server must not be reported");
    }
}
