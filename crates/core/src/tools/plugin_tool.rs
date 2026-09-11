//! Plugin-contributed tools (G1).
//!
//! Each tool a plugin declares in its Hello capabilities is registered here
//! as a [`PluginTool`] — a thin [`ToolExecutor`] adapter holding the tool
//! definition plus a shared handle to the owning plugin. Model-initiated
//! tool calls route through the normal pipeline (approval tiers apply by
//! name, exactly like built-ins: unknown names classify to the `Ask` tier
//! in `Default`/`AcceptEdits` modes) and execute via the plugin's
//! `execute_tool` hook — for remote plugins that is the `executeTool` RPC.
//!
//! Failure modes are model-visible, never silent:
//! - plugin disabled (RPC failure earlier in the session) → clear error;
//! - RPC timeout → error naming the plugin and budget;
//! - plugin error response → the plugin's error text.

use crate::plugin::PluginRegistry;
use nca_common::config::PluginConfig;
use nca_common::tool::{ToolCall, ToolDefinition, ToolResult};
use std::sync::Arc;

use super::ToolExecutor;

/// A single tool contributed by a plugin.
pub struct PluginTool {
    definition: ToolDefinition,
    plugin: Arc<dyn crate::plugin::NcaPlugin>,
    plugin_name: String,
    timeout_ms: u64,
}

impl PluginTool {
    /// Build from a `(definition, plugin)` pair collected via
    /// [`PluginRegistry::collect_tool_implementations`].
    ///
    /// The resolved timeout (declaration override, else the `[plugins]`
    /// default) is surfaced on the definition so the tool pipeline wraps
    /// execution in a cooperative timeout like any built-in.
    pub fn new(
        mut definition: ToolDefinition,
        plugin: Arc<dyn crate::plugin::NcaPlugin>,
        plugin_name: String,
        default_timeout_ms: u64,
    ) -> Self {
        let timeout_ms = definition.timeout_ms.unwrap_or(default_timeout_ms);
        definition.timeout_ms = Some(timeout_ms);
        Self {
            definition,
            plugin,
            plugin_name,
            timeout_ms,
        }
    }
}

#[async_trait::async_trait]
impl ToolExecutor for PluginTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn execute(&self, call: &ToolCall) -> ToolResult {
        let call_id = call.id.clone();
        // The plugin hook is sync-in-async (remote RPC via block_in_place);
        // enforce the timeout here so a hung plugin process can never pin a
        // turn, independent of the pipeline's cooperative wrap.
        let fut = self.plugin.execute_tool(call);
        match tokio::time::timeout(
            std::time::Duration::from_millis(self.timeout_ms.saturating_add(1_000)),
            fut,
        )
        .await
        {
            Ok(result) => ToolResult {
                call_id,
                timed_out: result.timed_out,
                success: result.success,
                output: result.output,
                error: result.error,
            },
            Err(_) => ToolResult {
                timed_out: true,
                call_id,
                success: false,
                output: String::new(),
                error: Some(format!(
                    "plugin tool `{}` timed out after {}ms (plugin `{}` did not answer)",
                    self.definition.name, self.timeout_ms, self.plugin_name
                )),
            },
        }
    }
}

/// Register every declared plugin tool into the registry. Returns how many
/// were registered. Name collisions with built-ins resolve in favor of the
/// earlier registration (built-ins register first; `ToolRegistry` warns and
/// skips duplicates) — plugin tools are expected to be namespaced by
/// convention (`trellis_*`), matching MCP's `mcp__server__tool` precedent.
pub fn register_plugin_tools(
    tools: &mut super::ToolRegistry,
    plugins: &PluginRegistry,
    cfg: &PluginConfig,
) -> usize {
    let impls = plugins.collect_tool_implementations();
    let count = impls.len();
    for (def, plugin) in impls {
        let owner = plugin.name().to_string();
        tools.register(Box::new(PluginTool::new(
            def,
            plugin,
            owner,
            cfg.tool_timeout_ms,
        )));
    }
    if count > 0 {
        tracing::info!("registered {count} plugin tool(s)");
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::NcaPlugin;
    use std::future::Future;
    use std::pin::Pin;

    struct StubPlugin {
        disabled: bool,
    }

    impl NcaPlugin for StubPlugin {
        fn name(&self) -> &str {
            "stub-plugin"
        }

        fn tools(&self) -> Vec<ToolDefinition> {
            vec![ToolDefinition {
                timeout_ms: None,
                name: "stub_greet".into(),
                description: "Greets".into(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            }]
        }

        fn execute_tool(
            &self,
            call: &ToolCall,
        ) -> Pin<Box<dyn Future<Output = ToolResult> + Send + '_>> {
            let disabled = self.disabled;
            let call_id = call.id.clone();
            let name = call.name.clone();
            Box::pin(async move {
                if disabled {
                    return ToolResult {
                        timed_out: false,
                        call_id,
                        success: false,
                        output: String::new(),
                        error: Some("plugin stub-plugin is disabled".into()),
                    };
                }
                ToolResult {
                    timed_out: false,
                    call_id,
                    success: true,
                    output: format!("hello from {name}"),
                    error: None,
                }
            })
        }
    }

    fn call(name: &str) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: name.into(),
            input: serde_json::json!({}),
        }
    }

    #[tokio::test]
    async fn plugin_tool_routes_to_owning_plugin() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(StubPlugin { disabled: false }));

        let mut tools = super::super::ToolRegistry::new();
        let n = register_plugin_tools(&mut tools, &reg, &PluginConfig::default());
        assert_eq!(n, 1);
        assert_eq!(tools.definitions().len(), 1);
        assert_eq!(tools.definitions()[0].name, "stub_greet");
        // Resolved timeout surfaces on the definition (cooperative wrap).
        assert_eq!(tools.definitions()[0].timeout_ms, Some(30_000));

        let result = tools.execute(&call("stub_greet")).await;
        assert!(result.success, "err: {:?}", result.error);
        assert_eq!(result.output, "hello from stub_greet");
    }

    #[tokio::test]
    async fn disabled_plugin_yields_model_visible_error() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(StubPlugin { disabled: true }));

        let mut tools = super::super::ToolRegistry::new();
        register_plugin_tools(&mut tools, &reg, &PluginConfig::default());

        let result = tools.execute(&call("stub_greet")).await;
        assert!(!result.success);
        assert!(result.error.unwrap().contains("disabled"));
    }
}
