//! Out-of-process plugin host ��� discovers, spawns, and communicates with
//! plugin processes via Cap'n Proto binary messages over stdin/stdout pipes.
//!
//! Discovery: nca scans `$XDG_CONFIG_DIR/nca/plugins/` for executables,
//! sorted by filename. Each binary is spawned directly (no `--describe` probe);
//! the plugin sends a Cap'n Proto `Hello` message immediately after spawn.
//!
//! Handshake: Plugin → Hello(name, version, protocol, capabilities)
//!            → Host validates protocol version (major match required)
//!            → Host → Config(workspace_root, session_id, permission_mode)
//!            → Operational (bidirectional Cap'n Proto message exchange)
//!
//! Transport: self-delimiting Cap'n Proto streaming frames over stdin/stdout.
//!            stderr is inherited for plugin diagnostics/logs.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use nca_common::config::{NcaConfig, PluginConfig};
use nca_common::tool::ToolDefinition;
use nca_core::plugin::NcaPlugin;
use nca_core::plugin_capnp::{ParamType, body, hello};
use nca_core::plugin_protocol::{self, FrameKind, PROTOCOL_MAJOR, WireError};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

/// A discovered plugin ready to be spawned.
#[derive(Debug, Clone)]
pub struct PluginDescriptor {
    pub name: String,
    pub binary: PathBuf,
}

/// Manages plugin process lifecycle.
pub struct PluginHost {
    instances: Vec<RemotePluginInstance>,
    next_id: Arc<AtomicU64>,
    config: PluginConfig,
}

struct RemotePluginInstance {
    name: String,
    process: Option<Child>,
    stdin: Option<Arc<Mutex<ChildStdin>>>,
    stdout: Option<Arc<Mutex<ChildStdout>>>,
    capabilities: Option<PluginCapabilities>,
    disabled: Arc<AtomicBool>,
}

/// Parsed capabilities from Hello.
#[derive(Debug, Clone, Default)]
pub struct PluginCapabilities {
    pub tools: Vec<ToolDefinition>,
    pub commands: Vec<String>,
}

// ─── Discovery ───────────────────────────────────────────────────────────────

/// Scan `$XDG_CONFIG_DIR/nca/plugins/` for executable plugin binaries,
/// sorted by filename.
pub fn discover_plugins() -> Vec<PluginDescriptor> {
    let plugins_dir = plugins_dir();
    if !plugins_dir.is_dir() {
        tracing::debug!("no plugins dir at {}", plugins_dir.display());
        return Vec::new();
    }

    let mut entries: Vec<PathBuf> = match std::fs::read_dir(&plugins_dir) {
        Ok(e) => e.flatten().map(|e| e.path()).collect(),
        Err(e) => {
            tracing::warn!("failed to read plugins dir: {e}");
            return Vec::new();
        }
    };

    // Sort by filename — deterministic load order (D7).
    entries.sort();

    let mut descriptors = Vec::new();
    for path in entries {
        if !path.is_file() {
            continue;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(meta) = std::fs::metadata(&path)
                && meta.permissions().mode() & 0o111 == 0
            {
                tracing::debug!("skipping non-executable: {}", path.display());
                continue;
            }
        }

        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        tracing::info!("discovered plugin: {} ({})", name, path.display());
        descriptors.push(PluginDescriptor { name, binary: path });
    }

    descriptors
}

/// Resolve the plugins directory path.
fn plugins_dir() -> PathBuf {
    nca_common::config::xdg_config_dir()
        .map(|d| d.join("nca/plugins"))
        .unwrap_or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".config/nca/plugins"))
                .unwrap_or_else(|| PathBuf::from(".nca/plugins"))
        })
}

// ─── Wire I/O ────────────────────────────────────────────────────────────────

/// Read a single Cap'n Proto message from an async reader.
///
/// Cap'n Proto streaming format: framing words + payload. This reads the raw
/// bytes, then parses via capnp's sync reader.
async fn read_capnp_message<R: AsyncRead + Unpin>(reader: &mut R) -> Result<Vec<u8>, WireError> {
    // Read the first word (8 bytes): segment count - 1 in low 32 bits.
    let mut header = [0u8; 8];
    reader.read_exact(&mut header).await?;

    // Stored value is `segment_count - 1` per Cap'n Proto spec; add 1 to
    // recover the true count.
    let stored_count = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let segment_count = stored_count + 1;

    // Calculate header size (1 word for count + segment_count words for sizes,
    // padded to even total).
    let header_words = (1 + segment_count).div_ceil(2);
    let mut full_header = Vec::with_capacity(header_words * 8);
    full_header.extend_from_slice(&header);

    if header_words > 1 {
        let mut rest = vec![0u8; header_words * 8 - 8];
        reader.read_exact(&mut rest).await?;
        full_header.extend_from_slice(&rest);
    }

    // Parse segment sizes and calculate total payload.
    let mut total_payload_words: usize = 0;
    for i in 0..segment_count {
        let offset = 4 + i * 4;
        let size = u32::from_le_bytes([
            full_header[offset],
            full_header[offset + 1],
            full_header[offset + 2],
            full_header[offset + 3],
        ]) as usize;
        total_payload_words += size;
    }

    // Read payload.
    let mut payload = vec![0u8; total_payload_words * 8];
    if !payload.is_empty() {
        reader.read_exact(&mut payload).await?;
    }

    // Reconstruct full wire message (header + payload).
    full_header.extend_from_slice(&payload);
    Ok(full_header)
}

/// Write a Cap'n Proto message to an async writer.
async fn write_capnp_message(writer: &mut ChildStdin, wire: &[u8]) -> Result<(), WireError> {
    writer.write_all(wire).await?;
    Ok(())
}

// ─── Lifecycle ───────────────────────────────────────────────────────────────

impl Default for PluginHost {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginHost {
    pub fn new() -> Self {
        Self::with_config(PluginConfig::default())
    }

    /// Build with explicit `[plugins]` tuning (timeouts, dispatch cap).
    pub fn with_config(config: PluginConfig) -> Self {
        Self {
            instances: Vec::new(),
            next_id: Arc::new(AtomicU64::new(1)),
            config,
        }
    }

    /// Spawn all given plugin descriptors and perform the Hello→Config handshake.
    pub async fn start_all(
        &mut self,
        descriptors: &[PluginDescriptor],
        workspace_root: &Path,
        session_id: &str,
        permission_mode: &str,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        for desc in descriptors {
            match self
                .spawn_one(desc, workspace_root, session_id, permission_mode)
                .await
            {
                Ok(()) => tracing::info!("plugin started: {}", desc.name),
                Err(e) => {
                    tracing::error!("failed to start plugin {}: {}", desc.name, e);
                    errors.push(format!("{}: {}", desc.name, e));
                }
            }
        }
        errors
    }

    async fn spawn_one(
        &mut self,
        desc: &PluginDescriptor,
        workspace_root: &Path,
        session_id: &str,
        permission_mode: &str,
    ) -> Result<(), String> {
        let mut cmd = Command::new(&desc.binary);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);

        let mut process = cmd.spawn().map_err(|e| format!("spawn: {e}"))?;

        let stdin = process.stdin.take().ok_or_else(|| "no stdin".to_string())?;
        let stdout = process
            .stdout
            .take()
            .ok_or_else(|| "no stdout".to_string())?;

        let stdin = Arc::new(Mutex::new(stdin));
        let stdout = Arc::new(Mutex::new(stdout));
        let disabled = Arc::new(AtomicBool::new(false));

        // Read Hello message with timeout.
        let hello_result = tokio::time::timeout(tokio::time::Duration::from_secs(10), async {
            let mut stdout_guard = stdout.lock().await;
            let raw = read_capnp_message(&mut *stdout_guard).await?;
            parse_hello(&raw)
        })
        .await;

        let capabilities = match hello_result {
            Ok(Ok((caps, protocol_major))) => {
                if protocol_major != PROTOCOL_MAJOR {
                    tracing::warn!(
                        "plugin {} protocol major {} != {}, disabling",
                        desc.name,
                        protocol_major,
                        PROTOCOL_MAJOR
                    );
                    disabled.store(true, Ordering::SeqCst);
                }
                caps
            }
            Ok(Err(e)) => {
                disabled.store(true, Ordering::SeqCst);
                return Err(format!("handshake: {e}"));
            }
            Err(_) => {
                disabled.store(true, Ordering::SeqCst);
                return Err(format!(
                    "plugin {} did not send Hello within 10s",
                    desc.name
                ));
            }
        };

        // Send Config message.
        let global_skills_dir = nca_common::config::xdg_config_dir()
            .map(|d| d.join("nca").join("skills"))
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        let config_wire = plugin_protocol::build_config(
            "0",
            workspace_root.to_str().unwrap_or("."),
            session_id,
            permission_mode,
            &global_skills_dir,
        );
        {
            let mut stdin_guard = stdin.lock().await;
            write_capnp_message(&mut stdin_guard, &config_wire)
                .await
                .map_err(|e| format!("write config: {e}"))?;
        }

        self.instances.push(RemotePluginInstance {
            name: desc.name.clone(),
            process: Some(process),
            stdin: Some(stdin),
            stdout: Some(stdout),
            capabilities: Some(capabilities),
            disabled,
        });

        Ok(())
    }

    /// Build a [`PluginRegistry`] with remote-backed [`NcaPlugin`] implementations.
    pub fn registry(&self) -> nca_core::plugin::PluginRegistry {
        let mut reg = nca_core::plugin::PluginRegistry::new();
        for instance in &self.instances {
            if instance.disabled.load(Ordering::SeqCst) {
                continue;
            }
            if let (Some(stdin), Some(stdout)) = (&instance.stdin, &instance.stdout) {
                let caps = instance.capabilities.clone().unwrap_or_default();
                let remote = RemotePlugin::new(
                    &instance.name,
                    stdin.clone(),
                    stdout.clone(),
                    self.next_id.clone(),
                    instance.disabled.clone(),
                    caps,
                    self.config.clone(),
                );
                reg.register(Box::new(remote));
            }
        }
        reg
    }

    /// Gracefully shut down all plugin processes.
    pub async fn shutdown(&mut self) {
        for instance in &mut self.instances {
            if let Some(stdin) = &instance.stdin {
                let wire = plugin_protocol::build_shutdown("0");
                let mut guard = stdin.lock().await;
                let _ = write_capnp_message(&mut guard, &wire).await;
            }
            if let Some(mut process) = instance.process.take() {
                let _ = process.kill().await;
                let _ = process.wait().await;
            }
        }
        self.instances.clear();
    }
}

impl Drop for PluginHost {
    fn drop(&mut self) {
        for instance in &mut self.instances {
            if let Some(mut p) = instance.process.take() {
                let _ = p.start_kill();
            }
        }
    }
}

// ─── Hello parsing ───────────────────────────────────────────────────────────

/// Parse raw Cap'n Proto bytes as a Hello message.
fn parse_hello(raw: &[u8]) -> Result<(PluginCapabilities, u16), WireError> {
    let mut reader = std::io::BufReader::new(raw);
    plugin_protocol::read_message_then(&mut reader, |msg| {
        let body = msg.get_body()?;
        match body.which() {
            Ok(body::Hello(h)) => {
                let h = h?;
                let major = h.get_protocol()?.get_major();
                let caps = parse_capabilities(&h)?;
                Ok((caps, major))
            }
            Ok(body::Error(e)) => {
                let e = e?;
                Err(WireError::Protocol(e.get_message()?.to_string()?))
            }
            _ => Err(WireError::Protocol(format!(
                "expected Hello, got {}",
                plugin_protocol::body_method_name(&body)
            ))),
        }
    })
}

/// Parse tool declarations from Hello capabilities.
fn parse_capabilities(hello: &hello::Reader<'_>) -> Result<PluginCapabilities, WireError> {
    let caps = hello.get_capabilities()?;
    let mut tools = Vec::new();
    for tool_reader in caps.get_tools()?.iter() {
        let name = tool_reader.get_name()?.to_string()?;
        let description = tool_reader.get_description()?.to_string()?;
        let params = build_json_schema(&tool_reader)?;
        tools.push(ToolDefinition {
            timeout_ms: None,
            name,
            description,
            parameters: params,
        });
    }
    let mut commands = Vec::new();
    for cmd in caps.get_commands()?.iter() {
        commands.push(cmd?.to_string()?);
    }
    Ok(PluginCapabilities { tools, commands })
}

/// Convert Cap'n Proto `ToolParameter` list to JSON Schema for LLM consumption.
fn build_json_schema(
    tool: &nca_core::plugin_capnp::tool_declaration::Reader<'_>,
) -> Result<serde_json::Value, WireError> {
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();

    for param in tool.get_parameters()?.iter() {
        let name = param.get_name()?.to_string()?;
        let json_type = param_type_to_string(param.get_type()?);
        let description = param.get_description()?.to_string()?;

        let mut schema = serde_json::json!({
            "type": json_type,
            "description": description,
        });

        // Add enum values if present.
        if param.has_enum_values() {
            let enums = param.get_enum_values()?;
            if !enums.is_empty() {
                let vals: Vec<String> = enums
                    .iter()
                    .map(|r| {
                        r.and_then(|s| {
                            s.to_str()
                                .map(|s| s.to_string())
                                .map_err(|e| capnp::Error::failed(e.to_string()))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                schema["enum"] = serde_json::Value::Array(
                    vals.into_iter().map(|v| serde_json::json!(v)).collect(),
                );
            }
        }

        if param.get_required() {
            required.push(name.clone());
        }
        properties.insert(name, schema);
    }

    Ok(serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
    }))
}

fn param_type_to_string(pt: ParamType) -> &'static str {
    match pt {
        ParamType::String => "string",
        ParamType::Number => "number",
        ParamType::Integer => "integer",
        ParamType::Boolean => "boolean",
        ParamType::Array => "array",
        ParamType::Object => "object",
        ParamType::Null => "null",
    }
}

// ─── Remote NcaPlugin ────────────────────────────────────────────────────────

pub(crate) struct RemotePlugin {
    name: String,
    stdin: Arc<Mutex<ChildStdin>>,
    stdout: Arc<Mutex<ChildStdout>>,
    next_id: Arc<AtomicU64>,
    disabled: Arc<AtomicBool>,
    capabilities: PluginCapabilities,
    config: PluginConfig,
}

impl RemotePlugin {
    pub fn new(
        name: impl Into<String>,
        stdin: Arc<Mutex<ChildStdin>>,
        stdout: Arc<Mutex<ChildStdout>>,
        next_id: Arc<AtomicU64>,
        disabled: Arc<AtomicBool>,
        capabilities: PluginCapabilities,
        config: PluginConfig,
    ) -> Self {
        Self {
            name: name.into(),
            stdin,
            stdout,
            next_id,
            disabled,
            capabilities,
            config,
        }
    }

    fn alloc_id(&self) -> String {
        self.next_id.fetch_add(1, Ordering::Relaxed).to_string()
    }

    fn is_disabled(&self) -> bool {
        self.disabled.load(Ordering::SeqCst)
    }

    /// Send a Cap'n Proto wire message and read the response.
    ///
    /// This is called from sync `NcaPlugin` trait hooks, which may run on a
    /// tokio worker thread. `block_in_place` moves the worker into the blocking
    /// pool so the inner `handle.block_on` runs on a non-driver thread — without
    /// it, `block_on` would panic ("Cannot start a runtime from within a
    /// runtime"). Requires a multi_thread runtime.
    ///
    /// Frame demultiplexing (G5): the plugin's stdout may interleave frames
    /// the host did not ask for — plugin→host callback arms (`readFile`,
    /// `log`, …) or out-of-order responses. Those are NOT supported; instead
    /// of silently consuming one and desyncing the stream, we skip them
    /// (answering callbacks with an error frame) until the frame whose
    /// envelope `id` matches the request. The whole write→read cycle holds
    /// the stdout lock, so concurrent RPCs on the same plugin serialize
    /// instead of interleaving each other's responses.
    fn rpc_sync(
        &self,
        expected_id: &str,
        wire: Vec<u8>,
        timeout_secs: u64,
    ) -> Result<Vec<u8>, String> {
        if self.is_disabled() {
            return Err(format!("plugin {} is disabled", self.name));
        }

        let handle =
            tokio::runtime::Handle::try_current().map_err(|e| format!("no runtime: {e}"))?;

        let stdin = self.stdin.clone();
        let stdout = self.stdout.clone();
        let disabled = self.disabled.clone();
        let expected_id = expected_id.to_string();
        tokio::task::block_in_place(|| {
            handle.block_on(async move {
                // Hold the stdout lock across write AND read: this serializes
                // whole RPCs per plugin (write→response pairing stays atomic
                // even if two hooks race). Lock order is always stdout→stdin.
                let mut stdout_guard = stdout.lock().await;

                {
                    let mut stdin_guard = stdin.lock().await;
                    write_capnp_message(&mut stdin_guard, &wire)
                        .await
                        .map_err(|e| format!("write: {e}"))?;
                }

                let deadline = tokio::time::Instant::now()
                    + std::time::Duration::from_secs(timeout_secs);
                let mut skipped = 0usize;
                loop {
                    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                    let frame = match tokio::time::timeout(
                        remaining,
                        read_capnp_message(&mut *stdout_guard),
                    )
                    .await
                    {
                        Ok(Ok(data)) => data,
                        Ok(Err(e)) => {
                            disabled.store(true, Ordering::SeqCst);
                            return Err(format!("read: {e}"));
                        }
                        Err(_) => {
                            disabled.store(true, Ordering::SeqCst);
                            return Err(format!("plugin timed out after {timeout_secs}s"));
                        }
                    };

                    match plugin_protocol::classify_frame(&frame) {
                        FrameKind::Response(id) if id == expected_id => return Ok(frame),
                        FrameKind::Response(other) => {
                            skipped += 1;
                            tracing::warn!(
                                "plugin {} sent out-of-order response id={other} (expected {expected_id}); skipping",
                                self.name,
                            );
                        }
                        FrameKind::Callback(id, cb) => {
                            skipped += 1;
                            tracing::warn!(
                                "plugin {} sent unsupported callback {cb}; answering with error",
                                self.name,
                            );
                            let reply =
                                plugin_protocol::build_error(&id, "callbacks are not supported");
                            let mut stdin_guard = stdin.lock().await;
                            if let Err(e) = write_capnp_message(&mut stdin_guard, &reply).await {
                                disabled.store(true, Ordering::SeqCst);
                                return Err(format!("write callback-error: {e}"));
                            }
                        }
                        FrameKind::Error(message) => {
                            return Err(message);
                        }
                        FrameKind::Opaque => {
                            skipped += 1;
                            tracing::warn!(
                                "plugin {} sent unclassifiable frame; skipping",
                                self.name,
                            );
                        }
                    }

                    if skipped >= 8 {
                        disabled.store(true, Ordering::SeqCst);
                        return Err(format!(
                            "plugin {} stream desynced: {skipped} unmatched frames",
                            self.name
                        ));
                    }
                }
            })
        })
    }
}

impl NcaPlugin for RemotePlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn on_system_prompt(&self, _config: &NcaConfig, workspace_root: &Path) -> Option<String> {
        if self.is_disabled() {
            return None;
        }

        let id = self.alloc_id();
        let wire = plugin_protocol::build_message(&id, |body| {
            let mut req = body.reborrow().init_system_prompt();
            req.set_workspace_root(workspace_root.to_str().unwrap_or("."));
        });

        let raw = self
            .rpc_sync(&id, wire, self.config.prompt_hook_timeout_ms.div_ceil(1000))
            .ok()?;

        // Parse the response.
        let mut reader = std::io::BufReader::new(&raw[..]);
        let result = plugin_protocol::read_message_then(&mut reader, |msg| {
            let body = msg.get_body()?;
            match body.which() {
                Ok(body::SystemPromptResult(r)) => {
                    let r = r?;
                    let text = r.get_text()?;
                    if text.is_empty() {
                        Ok(None)
                    } else {
                        Ok(Some(text.to_string()?))
                    }
                }
                _ => Ok(None),
            }
        });

        match result {
            Ok(Some(text)) => Some(text),
            _ => None,
        }
    }

    fn tools(&self) -> Vec<ToolDefinition> {
        if self.is_disabled() {
            return Vec::new();
        }
        self.capabilities.tools.clone()
    }

    /// `executeTool @4` RPC bridge (G1). Model-initiated tool calls on a
    /// plugin-owned tool arrive here through `PluginTool` → this method.
    ///
    /// Arguments are converted from the tool call's JSON object into the
    /// schema's typed KV list; non-scalar values ride as `json` text. The
    /// timeout comes from `[plugins] tool_timeout_ms` (default 30s — file IO
    /// can be slow, unlike prompt hooks).
    fn execute_tool(
        &self,
        call: &nca_common::tool::ToolCall,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = nca_common::tool::ToolResult> + Send + '_>,
    > {
        let call = call.clone();
        Box::pin(async move {
            let id = self.alloc_id();
            let args: Vec<(String, serde_json::Value)> = call
                .input
                .as_object()
                .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default();

            let wire = plugin_protocol::build_message(&id, |body| {
                let mut req = body.reborrow().init_execute_tool();
                req.set_tool_id(&call.name);
                let mut kv_list = req.reborrow().init_args(args.len() as u32);
                for (i, (key, value)) in args.iter().enumerate() {
                    let mut kv = kv_list.reborrow().get(i as u32);
                    kv.set_key(key);
                    match value {
                        serde_json::Value::String(s) => {
                            kv.set_type(nca_core::plugin_capnp::ValueType::String);
                            kv.set_str(s);
                        }
                        serde_json::Value::Bool(b) => {
                            kv.set_type(nca_core::plugin_capnp::ValueType::Boolean);
                            kv.set_bool(*b);
                        }
                        serde_json::Value::Number(n) => {
                            if let Some(i64_val) = n.as_i64() {
                                kv.set_type(nca_core::plugin_capnp::ValueType::Integer);
                                kv.set_int(i64_val);
                            } else {
                                kv.set_type(nca_core::plugin_capnp::ValueType::Number);
                                kv.set_num(n.as_f64().unwrap_or(0.0));
                            }
                        }
                        other => {
                            kv.set_type(nca_core::plugin_capnp::ValueType::Json);
                            kv.set_str(
                                serde_json::to_string(other).unwrap_or_else(|_| "null".into()),
                            );
                        }
                    }
                }
            });

            let timeout_secs = self.config.tool_timeout_ms.div_ceil(1000);
            let result = self.rpc_sync(&id, wire, timeout_secs);
            match result {
                Ok(raw) => {
                    let mut reader = std::io::BufReader::new(&raw[..]);
                    let parsed = plugin_protocol::read_message_then(&mut reader, |msg| {
                        let body = msg.get_body()?;
                        match body.which() {
                            Ok(body::ExecuteToolResult(r)) => {
                                let r = r?;
                                let error = r.get_error()?.to_string()?;
                                Ok(Some(nca_common::tool::ToolResult {
                                    timed_out: false,
                                    call_id: call.id.clone(),
                                    success: r.get_success(),
                                    output: r.get_output()?.to_string()?,
                                    error: (!error.is_empty()).then_some(error),
                                }))
                            }
                            Ok(body::Error(e)) => {
                                let e = e?;
                                Ok(Some(nca_common::tool::ToolResult {
                                    timed_out: false,
                                    call_id: call.id.clone(),
                                    success: false,
                                    output: String::new(),
                                    error: Some(e.get_message()?.to_string()?),
                                }))
                            }
                            _ => Ok(None),
                        }
                    });
                    match parsed {
                        Ok(Some(result)) => result,
                        Ok(None) => nca_common::tool::ToolResult {
                            timed_out: false,
                            call_id: call.id.clone(),
                            success: false,
                            output: String::new(),
                            error: Some(format!(
                                "plugin {} returned an unexpected frame for tool `{}`",
                                self.name, call.name
                            )),
                        },
                        Err(e) => nca_common::tool::ToolResult {
                            timed_out: false,
                            call_id: call.id.clone(),
                            success: false,
                            output: String::new(),
                            error: Some(format!(
                                "plugin {} tool `{}` response undecodable: {e}",
                                self.name, call.name
                            )),
                        },
                    }
                }
                Err(e) => nca_common::tool::ToolResult {
                    timed_out: e.contains("timed out"),
                    call_id: call.id.clone(),
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "plugin {} tool `{}` failed: {e}",
                        self.name, call.name
                    )),
                },
            }
        })
    }

    fn commands(&self) -> Vec<String> {
        if self.is_disabled() {
            return Vec::new();
        }
        self.capabilities.commands.clone()
    }

    /// `userPrompt @6` RPC bridge (G2). Per-turn context injection: the
    /// returned text rides as an ephemeral block on the outgoing user message
    /// (never the system prompt — that prefix is cache-stable). Budget comes
    /// from `[plugins] prompt_hook_timeout_ms`.
    fn on_user_prompt(&self, prompt: &str) -> Option<String> {
        if self.is_disabled() {
            return None;
        }

        let id = self.alloc_id();
        let wire = plugin_protocol::build_message(&id, |body| {
            let mut req = body.reborrow().init_user_prompt();
            req.set_prompt(prompt);
        });

        let raw = self
            .rpc_sync(&id, wire, self.config.prompt_hook_timeout_ms.div_ceil(1000))
            .ok()?;

        let mut reader = std::io::BufReader::new(&raw[..]);
        let result = plugin_protocol::read_message_then(&mut reader, |msg| {
            let body = msg.get_body()?;
            match body.which() {
                Ok(body::UserPromptResult(r)) => {
                    let r = r?;
                    let message = r.get_message()?.to_string()?;
                    Ok((!message.is_empty()).then_some(message))
                }
                _ => Ok(None),
            }
        });

        match result {
            Ok(Some(text)) => Some(text),
            _ => None,
        }
    }

    /// `event @15` fire-and-forget notification (G4). Writes the frame on a
    /// spawned task and never reads a reply (the schema defines no result arm
    /// for `event`) — the main loop can never block on a plugin here. A failed
    /// write means the pipe is broken: the plugin is disabled.
    fn on_event(&self, event: &serde_json::Value) {
        if self.is_disabled() {
            return;
        }
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let id = self.alloc_id();
        let payload = event.to_string();
        let wire = plugin_protocol::build_message(&id, |body| {
            let mut n = body.reborrow().init_event();
            n.set_event_json(&payload);
        });
        let stdin = self.stdin.clone();
        let disabled = self.disabled.clone();
        let name = self.name.clone();
        handle.spawn(async move {
            let mut guard = stdin.lock().await;
            if let Err(e) = write_capnp_message(&mut guard, &wire).await {
                tracing::warn!("plugin {name} event notify failed: {e}");
                disabled.store(true, Ordering::SeqCst);
            }
        });
    }

    fn on_command_execute_before(
        &self,
        command: &str,
        arguments: &str,
    ) -> Option<nca_core::plugin::CommandIntercept> {
        if self.is_disabled() {
            return None;
        }

        let id = self.alloc_id();
        let wire = plugin_protocol::build_message(&id, |body| {
            let mut req = body.reborrow().init_command_execute_before();
            req.set_command(command);
            req.set_session_id("");
            req.set_arguments(arguments);
        });

        let raw = self.rpc_sync(&id, wire, 10).ok()?;

        let mut reader = std::io::BufReader::new(&raw[..]);
        let result = plugin_protocol::read_message_then(&mut reader, |msg| {
            let body = msg.get_body()?;
            match body.which() {
                Ok(body::CommandExecuteBeforeResult(r)) => {
                    let r = r?;
                    Ok(Some(nca_core::plugin::CommandIntercept {
                        handled: r.get_handled(),
                        text: r.get_text()?.to_string()?,
                    }))
                }
                _ => Ok(None),
            }
        });

        match result {
            Ok(Some(intercept)) => Some(intercept),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugins_dir_defaults_to_xdg() {
        let dir = plugins_dir();
        assert!(dir.to_string_lossy().contains("nca"));
        assert!(dir.to_string_lossy().contains("plugins"));
    }

    // Regression: real plugin frames (Hello/Shutdown/Config/...) are standard
    // Cap'n Proto streaming messages produced by `serialize::write_message`.
    // `read_capnp_message` hand-parses them; any drift from the spec silently
    // breaks the plugin handshake. The original bug treated the stored
    // `segment_count - 1` as the true count, so every single-segment Hello was
    // rejected with "invalid segment count" and all plugins got disabled.
    //
    // This round-trip pins the hand parser against capnp's own serializer +
    // deserializer: build a real frame, feed the exact bytes a plugin emits,
    // then confirm the returned buffer is re-readable by capnp's reader.
    #[tokio::test]
    async fn read_capnp_message_round_trips_standard_frame() {
        let wire = plugin_protocol::build_shutdown("42");
        let mut reader = wire.as_slice();
        let raw = read_capnp_message(&mut reader)
            .await
            .expect("must parse a standard Cap'n Proto streaming frame");
        // The returned bytes must be re-readable by capnp's own reader.
        let mut buf = std::io::BufReader::new(&raw[..]);
        let parsed = plugin_protocol::read_message_then(&mut buf, |msg| {
            assert_eq!(msg.get_id()?, "42");
            assert!(matches!(msg.get_body()?.which(), Ok(body::Shutdown(_))));
            Ok(())
        });
        assert!(parsed.is_ok(), "round-trip failed: {parsed:?}");
        // Parser must consume exactly the full frame — no trailing bytes.
        assert_eq!(reader.len(), 0);
    }

    // Integration test: spawn the REAL trellis plugin binary and exercise the
    // full Hello → Config → CommandExecuteBefore round-trip.
    #[tokio::test]
    async fn trellis_plugin_command_execute_before_round_trip() {
        let binary = std::path::Path::new("/home/titan/.config/nca/plugins/trellis");
        if !binary.exists() {
            eprintln!("skipping: trellis plugin not installed");
            return;
        }

        let mut cmd = Command::new(binary);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        let mut process = cmd.spawn().expect("spawn trellis");
        let mut stdin = process.stdin.take().unwrap();
        let mut stdout = process.stdout.take().unwrap();

        // 1. Read Hello.
        let hello_raw = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            read_capnp_message(&mut stdout),
        )
        .await
        .expect("plugin did not send Hello in time")
        .expect("read hello");
        let (caps, proto_major) = parse_hello(&hello_raw).expect("parse hello");
        eprintln!(
            "hello: name commands={:?} proto={}",
            caps.commands, proto_major
        );
        assert!(
            caps.commands.iter().any(|c| c == "trellis:init"),
            "trellis:init not in commands: {:?}",
            caps.commands
        );

        // 2. Send Config.
        let cfg_wire = plugin_protocol::build_config("1", ".", "test-session", "accept-edits", "");
        write_capnp_message(&mut stdin, &cfg_wire)
            .await
            .expect("write config");

        // 3. Send CommandExecuteBefore for "trellis:init".
        let cmd_wire = plugin_protocol::build_message("2", |b| {
            let mut req = b.reborrow().init_command_execute_before();
            req.set_command("trellis:init");
            req.set_session_id("");
            req.set_arguments("");
        });
        write_capnp_message(&mut stdin, &cmd_wire)
            .await
            .expect("write command");

        // 4. Read response.
        let resp_raw = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            read_capnp_message(&mut stdout),
        )
        .await
        .expect("plugin did not respond in time")
        .expect("read response");

        let mut buf = std::io::BufReader::new(&resp_raw[..]);
        let result = plugin_protocol::read_message_then(&mut buf, |msg| {
            let body = msg.get_body()?;
            match body.which() {
                Ok(body::CommandExecuteBeforeResult(r)) => {
                    let r = r?;
                    Ok((r.get_handled(), r.get_text()?.to_string()?))
                }
                _ => Err(WireError::Capnp("unexpected body".into())),
            }
        });
        let (handled, text) = result.expect("parse response");
        eprintln!("response: handled={handled} text={text}");
        assert!(handled, "plugin should handle trellis:init");

        // Cleanup.
        let _ = process.start_kill();
    }

    // Verify ponytail returns handled=false for commands it doesn't own.
    // handled=false does NOT stop the poll — the registry continues to later
    // plugins (see `check_command_before`), so trellis still gets its chance.
    #[tokio::test]
    async fn ponytail_returns_not_handled_for_foreign_command() {
        let binary = std::path::Path::new("/home/titan/.config/nca/plugins/ponytail");
        if !binary.exists() {
            eprintln!("skipping: ponytail plugin not installed");
            return;
        }

        let mut cmd = Command::new(binary);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        let mut process = cmd.spawn().expect("spawn ponytail");
        let mut stdin = process.stdin.take().unwrap();
        let mut stdout = process.stdout.take().unwrap();

        // Read Hello.
        let hello_raw = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            read_capnp_message(&mut stdout),
        )
        .await
        .expect("no hello")
        .expect("read hello");
        let (caps, _) = parse_hello(&hello_raw).expect("parse hello");
        eprintln!("ponytail commands={:?}", caps.commands);

        // Send Config.
        let cfg_wire = plugin_protocol::build_config("1", ".", "test", "accept-edits", "");
        write_capnp_message(&mut stdin, &cfg_wire).await.unwrap();

        // Send CommandExecuteBefore for trellis:init.
        let cmd_wire = plugin_protocol::build_message("2", |b| {
            let mut req = b.reborrow().init_command_execute_before();
            req.set_command("trellis:init");
            req.set_session_id("");
            req.set_arguments("");
        });
        write_capnp_message(&mut stdin, &cmd_wire).await.unwrap();

        // Read response.
        let resp_raw = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            read_capnp_message(&mut stdout),
        )
        .await
        .expect("no response")
        .expect("read response");

        let mut buf = std::io::BufReader::new(&resp_raw[..]);
        let result = plugin_protocol::read_message_then(&mut buf, |msg| {
            let body = msg.get_body()?;
            match body.which() {
                Ok(body::CommandExecuteBeforeResult(r)) => {
                    let r = r?;
                    Ok((r.get_handled(), r.get_text()?.to_string()?))
                }
                _ => Err(WireError::Capnp("unexpected body".into())),
            }
        });
        let (handled, text) = result.expect("parse response");
        eprintln!("ponytail response: handled={handled} text={text:?}");
        assert!(!handled, "ponytail should NOT handle trellis:init");

        let _ = process.start_kill();
    }

    // Integration test: exercise the executeTool RPC against the REAL trellis
    // binary (G1). Trellis declares task-lifecycle tools in Hello; calling the
    // read-only `trellis_get_context` must produce an `executeToolResult` frame
    // (success may be false when the workspace has no `.trellis/` — the wire
    // arm and shape are what this pins).
    #[tokio::test]
    async fn trellis_plugin_execute_tool_round_trip() {
        let binary = std::path::Path::new("/home/titan/.config/nca/plugins/trellis");
        if !binary.exists() {
            eprintln!("skipping: trellis plugin not installed");
            return;
        }

        let mut cmd = Command::new(binary);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        let mut process = cmd.spawn().expect("spawn trellis");
        let mut stdin = process.stdin.take().unwrap();
        let mut stdout = process.stdout.take().unwrap();

        // 1. Hello → tools must include trellis_get_context.
        let hello_raw = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            read_capnp_message(&mut stdout),
        )
        .await
        .expect("plugin did not send Hello in time")
        .expect("read hello");
        let (caps, _) = parse_hello(&hello_raw).expect("parse hello");
        assert!(
            caps.tools.iter().any(|t| t.name == "trellis_get_context"),
            "trellis_get_context not declared: {:?}",
            caps.tools
                .iter()
                .map(|t| t.name.clone())
                .collect::<Vec<_>>()
        );

        // 2. Config.
        let cfg_wire = plugin_protocol::build_config("1", ".", "test-session", "accept-edits", "");
        write_capnp_message(&mut stdin, &cfg_wire)
            .await
            .expect("write config");

        // 3. executeTool with a string arg (KV conversion path).
        let cmd_wire = plugin_protocol::build_message("7", |b| {
            let mut req = b.reborrow().init_execute_tool();
            req.set_tool_id("trellis_get_context");
            let mut kvs = req.reborrow().init_args(1);
            let mut kv = kvs.reborrow().get(0);
            kv.set_key("section");
            kv.set_type(nca_core::plugin_capnp::ValueType::String);
            kv.set_str("tasks");
        });
        write_capnp_message(&mut stdin, &cmd_wire)
            .await
            .expect("write executeTool");

        // 4. Response must classify as a Response frame with our id.
        let resp_raw = tokio::time::timeout(
            std::time::Duration::from_secs(15),
            read_capnp_message(&mut stdout),
        )
        .await
        .expect("plugin did not respond in time")
        .expect("read response");

        assert_eq!(
            plugin_protocol::classify_frame(&resp_raw),
            plugin_protocol::FrameKind::Response("7".to_string()),
            "executeTool must answer with an executeToolResult frame carrying our id"
        );

        let mut buf = std::io::BufReader::new(&resp_raw[..]);
        let result = plugin_protocol::read_message_then(&mut buf, |msg| {
            let body = msg.get_body()?;
            match body.which() {
                Ok(body::ExecuteToolResult(r)) => {
                    let r = r?;
                    Ok((
                        r.get_success(),
                        r.get_output()?.to_string()?,
                        r.get_error()?.to_string()?,
                    ))
                }
                _ => Err(WireError::Capnp("unexpected body".into())),
            }
        });
        let (success, output, error) = result.expect("parse executeToolResult");
        eprintln!("executeTool: success={success} output={output:?} error={error:?}");
        if !success {
            assert!(
                !error.is_empty(),
                "failed tool call must carry an error text"
            );
        }

        let _ = process.start_kill();
    }
}
