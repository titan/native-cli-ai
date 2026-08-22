use serde::{Deserialize, Serialize};

/// Definition of a tool the agent can invoke.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    /// Optional cooperative execution timeout in milliseconds. When set, the
    /// tool pipeline wraps execution in `tokio::time::timeout` and returns a
    /// failed `ToolResult` with `timed_out = true` on expiry.
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

/// A tool invocation requested by the model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// The result of executing a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: String,
    pub success: bool,
    pub output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// True when this result was produced by the cooperative timeout guard
    /// (not by the tool itself). Kept orthogonal to `success` so replay and
    /// telemetry can distinguish "failed" from "timed out".
    #[serde(default)]
    pub timed_out: bool,
}

/// Permission tier for a tool or command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionTier {
    Allowed,
    Ask,
    Denied,
}
