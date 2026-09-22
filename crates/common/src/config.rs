use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};

/// Top-level configuration, merged from global, workspace, env, and CLI sources.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NcaConfig {
    pub provider: ProviderConfig,
    pub model: ModelConfig,
    pub permissions: PermissionConfig,
    pub session: SessionConfig,
    pub harness: HarnessConfig,
    pub mcp: McpConfig,
    pub memory: MemoryConfig,
    pub hooks: HookConfig,
    pub web: WebConfig,
    /// Step-request middleware knobs (`[middleware]`): retry, cost guard.
    #[serde(default)]
    pub middleware: MiddlewareConfig,
    /// Provider fallback (failover) chain (`[fallback]`).
    #[serde(default)]
    pub fallback: FallbackConfig,
    /// CLI/TUI preferences (e.g. external editor).
    #[serde(default)]
    pub ui: UiConfig,
    /// Named agent profiles: each can override provider, model, permissions,
    /// system prompt, and tool gating.  Keyed by profile name (e.g. "code-reviewer").
    #[serde(default)]
    pub agents: BTreeMap<String, AgentProfileConfig>,
    /// Directories mounted as additional allowed roots via `/mount`. Persisted
    /// to the workspace-local config so mounts survive restarts. Paths are
    /// canonical (symlinks resolved) at the time they were mounted.
    #[serde(default)]
    pub extra_paths: Vec<PathBuf>,
    /// `[subagent]` — subagent task lifecycle tuning
    /// (P1: read-only introspection).
    #[serde(default)]
    pub subagent: SubagentConfig,
    /// `[plugins]` — out-of-process plugin RPC tuning.
    #[serde(default)]
    pub plugins: PluginConfig,
    /// Named model plans (`[plans.<plan-name>.<agent-name>]`): each plan is a
    /// routing table applied atomically to the agent profiles via `/plan
    /// <name>` (provider-quota failover — swap every agent off a cooling
    /// account in one command).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub plans: BTreeMap<String, BTreeMap<String, PlanEntry>>,
    /// Name of the currently applied model plan. Informational marker (set by
    /// `/plan <name>` and persisted with the workspace config) so plan
    /// listings can mark the active plan.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_plan: Option<String>,
}

impl NcaConfig {
    /// Load config from defaults, global file, workspace file, and environment.
    pub fn load() -> Result<Self, ConfigError> {
        let workspace_root = env::current_dir().map_err(|source| ConfigError::Io {
            action: "read current directory",
            path: PathBuf::from("."),
            source,
        })?;
        Self::load_for_workspace(&workspace_root)
    }

    /// Load config for an explicit workspace root.
    pub fn load_for_workspace(workspace_root: &Path) -> Result<Self, ConfigError> {
        let mut config = Self::default();

        if let Some(path) = global_config_path()
            && path.exists()
        {
            let partial = load_partial(&path)?;
            config.merge(partial);
        }

        let local_path = workspace_config_path(workspace_root);
        if local_path.exists() {
            let partial = load_partial(&local_path)?;
            config.merge(partial);
        }

        config.apply_env();
        Ok(config)
    }

    /// Load only the persisted global config file layered over defaults.
    pub fn load_global_file() -> Result<Self, ConfigError> {
        let mut config = Self::default();
        if let Some(path) = global_config_path()
            && path.exists()
        {
            let partial = load_partial(&path)?;
            config.merge(partial);
        }
        Ok(config)
    }

    /// Load only the persisted workspace-local config layered over defaults.
    pub fn load_workspace_file(workspace_root: &Path) -> Result<Self, ConfigError> {
        let mut config = Self::default();
        let local_path = workspace_config_path(workspace_root);
        if local_path.exists() {
            let partial = load_partial(&local_path)?;
            config.merge(partial);
        }
        Ok(config)
    }

    /// Save the full config as the user's global defaults.
    pub fn save_global(&self) -> Result<(), ConfigError> {
        let path = global_config_path().ok_or(ConfigError::NoHomeDir)?;
        save_config_to_path(self, &path)
    }

    /// Save only the workspace-specific overrides (diff against defaults + global + env).
    ///
    /// This avoids dumping the entire merged config into `.nca/config.local.toml`,
    /// which would override global config values with defaults.
    pub fn save_workspace_file(&self, workspace_root: &Path) -> Result<(), ConfigError> {
        // Base = defaults + global + env (everything except the local file itself).
        let mut base = Self::load_global_file().unwrap_or_default();
        base.apply_env();

        let current_toml =
            toml::Value::try_from(self).map_err(|source| ConfigError::SerializeToml {
                path: workspace_config_path(workspace_root),
                source,
            })?;
        let base_toml =
            toml::Value::try_from(&base).map_err(|source| ConfigError::SerializeToml {
                path: workspace_config_path(workspace_root),
                source,
            })?;

        let diff = match diff_toml_values(&current_toml, &base_toml) {
            Some(d) => d,
            None => {
                // No overrides — remove the local file if it exists.
                let path = workspace_config_path(workspace_root);
                if path.exists() {
                    std::fs::remove_file(&path).map_err(|source| ConfigError::Io {
                        action: "remove empty local config",
                        path,
                        source,
                    })?;
                }
                return Ok(());
            }
        };

        let path = workspace_config_path(workspace_root);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| ConfigError::Io {
                action: "create config directory",
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let raw = toml::to_string_pretty(&diff).map_err(|source| ConfigError::SerializeToml {
            path: path.to_path_buf(),
            source,
        })?;
        std::fs::write(&path, raw).map_err(|source| ConfigError::Io {
            action: "write config file",
            path,
            source,
        })
    }

    /// Remove the workspace-local config file, if present.
    pub fn clear_workspace_file(workspace_root: &Path) -> Result<(), ConfigError> {
        let path = workspace_config_path(workspace_root);
        if !path.exists() {
            return Ok(());
        }
        std::fs::remove_file(&path).map_err(|source| ConfigError::Io {
            action: "remove config file",
            path,
            source,
        })
    }

    fn merge(&mut self, partial: PartialNcaConfig) {
        let provider_changed = partial.provider.is_some();
        if let Some(provider) = partial.provider {
            self.provider.merge(provider);
        }

        if let Some(model) = partial.model {
            self.model.merge(model);
        }

        if let Some(permissions) = partial.permissions {
            self.permissions.merge(permissions);
        }

        if let Some(session) = partial.session {
            self.session.merge(session);
        }
        if let Some(harness) = partial.harness {
            self.harness.merge(harness);
        }
        if let Some(mcp) = partial.mcp {
            self.mcp.merge(mcp);
        }
        if let Some(memory) = partial.memory {
            self.memory.merge(memory);
        }
        if let Some(hooks) = partial.hooks {
            self.hooks.merge(hooks);
        }
        if let Some(web) = partial.web {
            self.web.merge(web);
        }
        if let Some(ui) = partial.ui {
            self.ui.merge(ui);
        }
        if let Some(middleware) = partial.middleware {
            self.middleware.merge(middleware);
        }
        if let Some(fallback) = partial.fallback {
            self.fallback.merge(fallback);
        }
        if let Some(agents) = partial.agents {
            self.merge_agents(agents);
        }
        if let Some(subagent) = partial.subagent {
            self.subagent.merge(subagent);
        }
        if let Some(plugins) = partial.plugins {
            self.plugins.merge(plugins);
        }
        if let Some(plans) = partial.plans {
            self.merge_plans(plans);
        }
        if let Some(active_plan) = partial.active_plan {
            self.active_plan = Some(active_plan);
        }

        if let Some(extra_paths) = partial.extra_paths {
            self.extra_paths = extra_paths;
        }

        if provider_changed {
            self.sync_default_model_from_provider();
        }
    }

    fn apply_env(&mut self) {
        if let Ok(provider) = env::var("NCA_DEFAULT_PROVIDER") {
            self.provider.default = ProviderKind::from_env(&provider);
            self.sync_default_model_from_provider();
        }

        if let Ok(model) = env::var("NCA_MODEL") {
            self.apply_model_override(&model);
        }

        if let Ok(api_key) = env::var("MINIMAX_API_KEY") {
            self.provider.minimax.api_key = Some(api_key);
        }

        if let Ok(base_url) = env::var("MINIMAX_BASE_URL") {
            self.provider.minimax.base_url = base_url;
        }

        if let Ok(model) = env::var("MINIMAX_MODEL") {
            self.provider.minimax.model = model;
        }

        if let Ok(api_key) = env::var("OPENAI_API_KEY") {
            self.provider.openai.api_key = Some(api_key);
        }

        if let Ok(base_url) = env::var("OPENAI_BASE_URL") {
            self.provider.openai.base_url = base_url;
        }

        if let Ok(model) = env::var("OPENAI_MODEL") {
            self.provider.openai.model = model;
        }

        if let Ok(api_key) = env::var("ANTHROPIC_API_KEY") {
            self.provider.anthropic.api_key = Some(api_key);
        }

        if let Ok(base_url) = env::var("ANTHROPIC_BASE_URL") {
            self.provider.anthropic.base_url = base_url;
        }

        if let Ok(model) = env::var("ANTHROPIC_MODEL") {
            self.provider.anthropic.model = model;
        }

        if let Ok(api_key) = env::var("OPENROUTER_API_KEY") {
            self.provider.openrouter.api_key = Some(api_key);
        }

        if let Ok(base_url) = env::var("OPENROUTER_BASE_URL") {
            self.provider.openrouter.base_url = base_url;
        }

        if let Ok(model) = env::var("OPENROUTER_MODEL") {
            self.provider.openrouter.model = model;
        }

        if let Ok(site_url) = env::var("OPENROUTER_SITE_URL") {
            self.provider.openrouter.site_url = Some(site_url);
        }

        if let Ok(app_name) = env::var("OPENROUTER_APP_NAME") {
            self.provider.openrouter.app_name = Some(app_name);
        }

        if let Ok(api_key) = env::var("ZHIPUAI_API_KEY") {
            self.provider.zhipuai.api_key = Some(api_key);
        }

        if let Ok(base_url) = env::var("ZHIPUAI_BASE_URL") {
            self.provider.zhipuai.base_url = base_url;
        }

        if let Ok(model) = env::var("ZHIPUAI_MODEL") {
            self.provider.zhipuai.model = model;
        }

        if let Ok(api_key) = env::var("DEEPSEEK_API_KEY") {
            self.provider.deepseek.api_key = Some(api_key);
        }

        if let Ok(base_url) = env::var("DEEPSEEK_BASE_URL") {
            self.provider.deepseek.base_url = base_url;
        }

        if let Ok(model) = env::var("DEEPSEEK_MODEL") {
            self.provider.deepseek.model = model;
        }

        if let Ok(api_key) = env::var("KIMI_API_KEY") {
            self.provider.kimi.api_key = Some(api_key);
        }

        if let Ok(base_url) = env::var("KIMI_BASE_URL") {
            self.provider.kimi.base_url = base_url;
        }

        if let Ok(model) = env::var("KIMI_MODEL") {
            self.provider.kimi.model = model;
        }

        if let Ok(api_key) = env::var("MIMO_API_KEY") {
            self.provider.mimo.api_key = Some(api_key);
        }

        if let Ok(base_url) = env::var("MIMO_BASE_URL") {
            self.provider.mimo.base_url = base_url;
        }

        if let Ok(model) = env::var("MIMO_MODEL") {
            self.provider.mimo.model = model;
        }

        if let Ok(memory_path) = env::var("NCA_MEMORY_PATH") {
            self.memory.file_path = PathBuf::from(memory_path);
        }

        if let Ok(timeout_secs) = env::var("NCA_WEB_TIMEOUT_SECS")
            && let Ok(timeout_secs) = timeout_secs.parse()
        {
            self.web.timeout_secs = timeout_secs;
        }

        if let Ok(max_fetch_chars) = env::var("NCA_WEB_MAX_FETCH_CHARS")
            && let Ok(max_fetch_chars) = max_fetch_chars.parse()
        {
            self.web.max_fetch_chars = max_fetch_chars;
        }

        if let Ok(search_endpoint) = env::var("NCA_WEB_SEARCH_ENDPOINT") {
            self.web.search_endpoint = search_endpoint;
        }

        self.sync_default_model_from_provider();
    }

    /// Map known alias patterns to the provider they belong to.
    ///
    /// When a user picks a provider-specific alias (e.g. "glm" or "gpt4o") from the
    /// model picker while a different provider is active, the alias must also switch
    /// the default provider so the correct endpoint is used.
    pub fn provider_hint_for_alias(alias: &str) -> Option<ProviderKind> {
        match alias.trim().to_ascii_lowercase().as_str() {
            "minimax" | "m2.7" | "m3" => Some(ProviderKind::MiniMax),
            "kimi" | "kimi-k2" | "kimi-k3" | "k3" | "moonshot" | "moonshot-v1" => {
                Some(ProviderKind::Kimi)
            }
            "openai" | "gpt" | "gpt4o" | "gpt4omini" => Some(ProviderKind::OpenAi),
            "claude" | "claude-sonnet" => Some(ProviderKind::Anthropic),
            "openrouter" => Some(ProviderKind::OpenRouter),
            "zhipuai" | "glm" | "glm5" | "glm-5.2" | "glm-5.3" | "glm-5.3-flash" => {
                Some(ProviderKind::ZhipuAI)
            }
            "mimo" | "xiaomi" | "mimo-pro" | "mimo-flash" | "mimo-ultraspeed" => {
                Some(ProviderKind::Mimo)
            }
            "default"
            | "deepseek"
            | "ds"
            | "deepseek-v4"
            | "dsv4"
            | "dsv4p"
            | "deepseek-v3"
            | "dsv3"
            | "deepseek-r1"
            | "dsr1"
            | "deepseek-flash"
            | "flash"
            | "deepseek-v4.1-flash"
            | "deepseek-v4-1-flash" => Some(ProviderKind::DeepSeek),
            _ => None,
        }
    }

    pub fn apply_model_override(&mut self, raw_model: &str) {
        let resolved = self.model.resolve_alias(raw_model);
        // Switch provider when the alias belongs to a specific provider.
        if let Some(provider) = Self::provider_hint_for_alias(raw_model) {
            self.provider.default = provider;
        }
        self.provider.set_model_for_default(resolved);
        self.sync_default_model_from_provider();
    }

    /// Switch the default LLM provider and keep `default_model` aligned with that provider's model field.
    pub fn set_default_provider(&mut self, provider: ProviderKind) {
        self.provider.default = provider;
        self.sync_default_model_from_provider();
    }

    /// Set the API key stored in config for a provider (workspace save may persist it).
    pub fn set_provider_api_key(&mut self, provider: ProviderKind, key: impl Into<String>) {
        let key = key.into();
        match provider {
            ProviderKind::MiniMax => self.provider.minimax.api_key = Some(key),
            ProviderKind::OpenAi => self.provider.openai.api_key = Some(key),
            ProviderKind::Anthropic => self.provider.anthropic.api_key = Some(key),
            ProviderKind::OpenRouter => self.provider.openrouter.api_key = Some(key),
            ProviderKind::ZhipuAI => self.provider.zhipuai.api_key = Some(key),
            ProviderKind::DeepSeek => self.provider.deepseek.api_key = Some(key),
            ProviderKind::Kimi => self.provider.kimi.api_key = Some(key),
            ProviderKind::Mimo => self.provider.mimo.api_key = Some(key),
            ProviderKind::Custom => self.provider.custom.api_key = Some(key),
        }
    }

    /// Editor command: `NCA_EDITOR`, then `[ui].editor`, then `EDITOR`, then `vim`.
    pub fn effective_editor_command(&self) -> String {
        if let Ok(v) = env::var("NCA_EDITOR") {
            let t = v.trim();
            if !t.is_empty() {
                return t.to_string();
            }
        }
        if let Some(ref e) = self.ui.editor {
            let t = e.trim();
            if !t.is_empty() {
                return t.to_string();
            }
        }
        env::var("EDITOR")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| "vim".to_string())
    }

    pub fn sync_default_model_from_provider(&mut self) {
        self.model.default_model = self.provider.active_model().to_string();
    }

    /// Returns `true` if the first-run onboarding gate should be shown.
    /// Triggers when: onboarding not completed OR all API keys have been removed.
    pub fn needs_onboarding(&self) -> bool {
        !self.ui.onboarding_completed || !self.provider.any_api_key_present()
    }

    /// Merge a map of partial agent profiles into the existing `agents` map.
    /// Each key creates or updates the corresponding agent profile.
    fn merge_agents(&mut self, partials: BTreeMap<String, PartialAgentProfileConfig>) {
        for (name, partial) in partials {
            let existing = self.agents.entry(name).or_default();
            existing.merge(partial);
        }
    }

    /// Merge partial plan tables into the existing `plans` map. Entries
    /// replace per `(plan, agent)` key, so a later layer (workspace over
    /// global) can override individual routes of a plan without redefining
    /// the whole table.
    fn merge_plans(&mut self, partials: BTreeMap<String, BTreeMap<String, PlanEntry>>) {
        for (plan, entries) in partials {
            let existing = self.plans.entry(plan).or_default();
            for (agent, entry) in entries {
                existing.insert(agent, entry);
            }
        }
    }

    /// Look up a named agent profile by name.
    pub fn agent_profile(&self, name: &str) -> Option<&AgentProfileConfig> {
        self.agents.get(name)
    }

    /// List all defined agent profile names.
    pub fn agent_profile_names(&self) -> Vec<&str> {
        self.agents.keys().map(|s| s.as_str()).collect()
    }
}

/// Configuration overrides for a named agent profile.
///
/// Agents can be defined in `config.toml` under `[agents.<name>]` and also
/// in skill frontmatter (`provider:` key).  Each field is optional — missing
/// fields inherit from the global config.
///
/// # Example (config.toml)
///
/// ```toml
/// [agents.code-reviewer]
/// provider = "openai"
/// model = "gpt-4o"
/// permission_mode = "plan"
/// system_prompt_append = "Focus on security and correctness."
/// allowed_tools = ["read", "search", "list_directory"]
/// skills_remove = ["web-search", "commit"]
/// ```
///
/// Skill gating: `skills` (base whitelist), `skills_add`, and
/// `skills_remove` fold into the agent's effective skill set
/// ((base ∪ add) − remove, intersected with discovered skills). When all
/// three are unset the agent sees every discovered skill (no filtering).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentProfileConfig {
    /// Human-readable description shown in agent pickers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Override the LLM provider for this agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderKind>,
    /// Override the model name for this agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Override the permission mode for this agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub permission_mode: Option<PermissionMode>,
    /// Complete system prompt that replaces the built-in harness prompt when set.
    /// Use this for specialist agents with their own persona.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Extra system-prompt text appended when this agent is active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt_append: Option<String>,
    /// If set, only these tools are available (all others are disabled).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<Vec<String>>,
    /// Base skill whitelist (skill command names). `None` = all discovered
    /// skills form the base. Entries not present in discovery are ignored —
    /// a skill that was never discovered cannot be granted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skills: Option<Vec<String>>,
    /// Skills merged into the effective set after the base (dedup; names not
    /// present in discovery are ignored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skills_add: Option<Vec<String>>,
    /// Skills cut from the effective set last. On a name clash with
    /// `skills_add`, remove always wins. Removing an unknown name is a
    /// no-op.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skills_remove: Option<Vec<String>>,
}

impl AgentProfileConfig {
    /// Merge partial overrides into this profile.
    fn merge(&mut self, partial: PartialAgentProfileConfig) {
        if let Some(v) = partial.provider {
            self.provider = Some(v);
        }
        if let Some(v) = partial.model {
            self.model = Some(v);
        }
        if let Some(v) = partial.permission_mode {
            self.permission_mode = Some(v);
        }
        if let Some(v) = partial.description {
            self.description = Some(v);
        }
        if let Some(v) = partial.system_prompt {
            self.system_prompt = Some(v);
        }
        if let Some(v) = partial.system_prompt_append {
            self.system_prompt_append = Some(v);
        }
        if let Some(v) = partial.allowed_tools {
            self.allowed_tools = Some(v);
        }
        if let Some(v) = partial.skills {
            self.skills = Some(v);
        }
        if let Some(v) = partial.skills_add {
            self.skills_add = Some(v);
        }
        if let Some(v) = partial.skills_remove {
            self.skills_remove = Some(v);
        }
    }

    /// Resolve the effective provider: explicit override > alias hint from model > inherit.
    pub fn resolve_provider(&self) -> Option<ProviderKind> {
        self.provider.or_else(|| {
            self.model
                .as_ref()
                .and_then(|m| NcaConfig::provider_hint_for_alias(m))
        })
    }

    /// Resolve the effective model: explicit override > inherit.
    pub fn resolve_model<'a>(&'a self, config_default_model: &'a str) -> &'a str {
        self.model.as_deref().unwrap_or(config_default_model)
    }
}

/// Partial version for deserialization (all fields optional).
#[derive(Debug, Clone, Deserialize, Default)]
struct PartialAgentProfileConfig {
    description: Option<String>,
    provider: Option<ProviderKind>,
    model: Option<String>,
    permission_mode: Option<PermissionMode>,
    system_prompt: Option<String>,
    system_prompt_append: Option<String>,
    allowed_tools: Option<Vec<String>>,
    skills: Option<Vec<String>>,
    skills_add: Option<Vec<String>>,
    skills_remove: Option<Vec<String>>,
}

/// One agent's route inside a model plan (`[plans.<plan>.<agent>]`):
/// an optional provider and/or model override. `None` fields mean "leave
/// that dimension untouched" — a provider-only entry clears the agent's
/// model pin (it inherits the new provider's default model), a model-only
/// entry keeps the agent's provider pin.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlanEntry {
    /// Provider override for this agent inside the plan.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderKind>,
    /// Model override (aliases like `dsv4` resolve at apply time).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiConfig {
    /// Shell command to launch the external editor (e.g. `vim` or `code --wait`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub editor: Option<String>,
    /// Theme name (future: "default", "tokyonight", etc.).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub theme: Option<String>,
    /// Hide hint text in the composer area.
    #[serde(default)]
    pub hide_tips: bool,
    /// Lines per scroll event (default 3).
    #[serde(default = "default_scroll_speed")]
    pub scroll_speed: u16,
    /// Whether the user has completed the first-run onboarding flow.
    #[serde(default)]
    pub onboarding_completed: bool,
}

fn default_scroll_speed() -> u16 {
    3
}

fn default_subagent_result_timeout_ms() -> u64 {
    30_000
}

fn default_subagent_background() -> bool {
    true
}

fn default_wake_enabled() -> bool {
    true
}

fn default_wake_interval_ms() -> u64 {
    1000
}

/// `[subagent.wake]` — parent wake-on-terminal scheduling (P3): when a
/// background child task reaches a terminal state, an idle parent session
/// is woken with a single reconciled prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WakeConfig {
    /// Master switch for wake delivery. `false` is the P3 rollback gate:
    /// background children still run detached, but the parent is never
    /// woken (poll `task_status`/`/jobs` instead).
    #[serde(default = "default_wake_enabled")]
    pub enabled: bool,
    /// Terminal→wake debounce window, milliseconds. Terminal states arriving
    /// within the window coalesce into a single wake.
    #[serde(default = "default_wake_interval_ms")]
    pub interval_ms: u64,
}

impl Default for WakeConfig {
    fn default() -> Self {
        Self {
            enabled: default_wake_enabled(),
            interval_ms: default_wake_interval_ms(),
        }
    }
}

impl WakeConfig {
    fn merge(&mut self, partial: PartialWakeConfig) {
        if let Some(enabled) = partial.enabled {
            self.enabled = enabled;
        }
        if let Some(interval_ms) = partial.interval_ms {
            self.interval_ms = interval_ms;
        }
    }
}

/// `[subagent]` — subagent task lifecycle tuning (P1: read-only
/// introspection via `task_status`/`task_result`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentConfig {
    /// Default for `spawn_subagent` calls that OMIT the `background` flag
    /// (P3): `true` detaches by default in top-level TUI sessions; explicit
    /// flag values always win. `false` restores the P2 foreground default.
    #[serde(default = "default_subagent_background")]
    pub background: bool,
    /// Max wait for a `task_status`/`task_result` reply, milliseconds.
    #[serde(default = "default_subagent_result_timeout_ms")]
    pub result_timeout_ms: u64,
    /// Wake scheduler settings for background children (P3).
    #[serde(default)]
    pub wake: WakeConfig,
}

impl Default for SubagentConfig {
    fn default() -> Self {
        Self {
            background: default_subagent_background(),
            result_timeout_ms: default_subagent_result_timeout_ms(),
            wake: WakeConfig::default(),
        }
    }
}

/// `[plugins]` — tuning for the out-of-process plugin RPC surface.
/// Timeouts are per-RPC; the host never blocks a turn longer than the
/// relevant timeout waiting on a plugin.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PluginConfig {
    /// RPC budget for per-turn prompt hooks (`userPrompt`), milliseconds.
    #[serde(default = "default_plugin_prompt_hook_timeout_ms")]
    pub prompt_hook_timeout_ms: u64,
    /// RPC budget for `executeTool` calls, milliseconds. Plugin tools may
    /// do file IO, so this defaults well above the prompt-hook budget.
    #[serde(default = "default_plugin_tool_timeout_ms")]
    pub tool_timeout_ms: u64,
    /// RPC budget for the subagent dispatch hook, milliseconds.
    #[serde(default = "default_plugin_subagent_dispatch_timeout_ms")]
    pub subagent_dispatch_timeout_ms: u64,
    /// Host-enforced cap on total plugin-appended context per subagent
    /// dispatch, in bytes. Overflow is truncated with a marker.
    #[serde(default = "default_plugin_subagent_context_max_bytes")]
    pub subagent_context_max_bytes: usize,
}

fn default_plugin_prompt_hook_timeout_ms() -> u64 {
    5_000
}
fn default_plugin_tool_timeout_ms() -> u64 {
    30_000
}
fn default_plugin_subagent_dispatch_timeout_ms() -> u64 {
    5_000
}
fn default_plugin_subagent_context_max_bytes() -> usize {
    65_536
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            prompt_hook_timeout_ms: default_plugin_prompt_hook_timeout_ms(),
            tool_timeout_ms: default_plugin_tool_timeout_ms(),
            subagent_dispatch_timeout_ms: default_plugin_subagent_dispatch_timeout_ms(),
            subagent_context_max_bytes: default_plugin_subagent_context_max_bytes(),
        }
    }
}

impl PluginConfig {
    fn merge(&mut self, partial: PartialPluginConfig) {
        if let Some(v) = partial.prompt_hook_timeout_ms {
            self.prompt_hook_timeout_ms = v;
        }
        if let Some(v) = partial.tool_timeout_ms {
            self.tool_timeout_ms = v;
        }
        if let Some(v) = partial.subagent_dispatch_timeout_ms {
            self.subagent_dispatch_timeout_ms = v;
        }
        if let Some(v) = partial.subagent_context_max_bytes {
            self.subagent_context_max_bytes = v;
        }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialPluginConfig {
    prompt_hook_timeout_ms: Option<u64>,
    tool_timeout_ms: Option<u64>,
    subagent_dispatch_timeout_ms: Option<u64>,
    subagent_context_max_bytes: Option<usize>,
}

impl SubagentConfig {
    fn merge(&mut self, partial: PartialSubagentConfig) {
        if let Some(background) = partial.background {
            self.background = background;
        }
        if let Some(result_timeout_ms) = partial.result_timeout_ms {
            self.result_timeout_ms = result_timeout_ms;
        }
        if let Some(wake) = partial.wake {
            self.wake.merge(wake);
        }
    }
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            editor: None,
            theme: None,
            hide_tips: false,
            scroll_speed: default_scroll_speed(),
            onboarding_completed: false,
        }
    }
}

impl UiConfig {
    fn merge(&mut self, partial: PartialUiConfig) {
        if let Some(editor) = partial.editor {
            self.editor = Some(editor);
        }
        if let Some(theme) = partial.theme {
            self.theme = Some(theme);
        }
        if let Some(hide_tips) = partial.hide_tips {
            self.hide_tips = hide_tips;
        }
        if let Some(scroll_speed) = partial.scroll_speed {
            self.scroll_speed = scroll_speed;
        }
        if let Some(onboarding_completed) = partial.onboarding_completed {
            self.onboarding_completed = onboarding_completed;
        }
    }
}

/// Global config file: `$XDG_CONFIG_HOME/nca/config.toml` (default `~/.config/nca/config.toml`).
pub fn global_config_path() -> Option<PathBuf> {
    xdg_config_dir().map(|dir| dir.join("nca/config.toml"))
}

/// XDG config home: `$XDG_CONFIG_HOME` (default `$HOME/.config`).
pub fn xdg_config_dir() -> Option<PathBuf> {
    if let Ok(v) = env::var("XDG_CONFIG_HOME")
        && !v.is_empty()
    {
        return Some(PathBuf::from(v));
    }
    env::var_os("HOME").map(|home| PathBuf::from(home).join(".config"))
}

/// XDG data home: `$XDG_DATA_HOME` (default `$HOME/.local/share`).
pub fn xdg_data_dir() -> Option<PathBuf> {
    if let Ok(v) = env::var("XDG_DATA_HOME")
        && !v.is_empty()
    {
        return Some(PathBuf::from(v));
    }
    env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share"))
}

/// `$XDG_DATA_HOME/nca` when a home directory is resolvable.
pub fn nca_data_dir() -> Option<PathBuf> {
    xdg_data_dir().map(|dir| dir.join("nca"))
}

/// Stable per-workspace id: `{slug}-{hex}` derived from the canonical workspace path.
pub fn workspace_cache_id(workspace_root: &Path) -> Result<(String, PathBuf), WorkspaceCacheError> {
    let canonical =
        workspace_root
            .canonicalize()
            .map_err(|source| WorkspaceCacheError::Canonicalize {
                path: workspace_root.to_path_buf(),
                source,
            })?;
    let path_str = canonical.to_string_lossy();
    let suffix = workspace_path_hash_suffix(path_str.as_ref());
    let slug = workspace_dir_slug(&canonical);
    Ok((format!("{slug}-{suffix}"), canonical))
}

/// `$XDG_DATA_HOME/nca/workspaces/<workspace-id>/`
pub fn workspace_cache_dir(workspace_root: &Path) -> Result<PathBuf, WorkspaceCacheError> {
    let (id, _) = workspace_cache_id(workspace_root)?;
    let data = nca_data_dir().ok_or(WorkspaceCacheError::NoHomeDir)?;
    Ok(data.join("workspaces").join(id))
}

/// Cached CLI index JSON for this workspace.
pub fn workspace_cli_index_path(workspace_root: &Path) -> Result<PathBuf, WorkspaceCacheError> {
    Ok(workspace_cache_dir(workspace_root)?.join("cli-index.json"))
}

fn workspace_dir_slug(path: &Path) -> String {
    let raw = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("workspace")
        .to_ascii_lowercase();
    let mut out = String::new();
    let mut prev_sep = false;
    for c in raw.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
            prev_sep = false;
        } else if !out.is_empty() && !prev_sep {
            out.push('-');
            prev_sep = true;
        }
    }
    let trimmed = out.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "workspace".to_string()
    } else {
        trimmed
    }
}

fn workspace_path_hash_suffix(canonical_path: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(canonical_path.as_bytes());
    let digest = hasher.finalize();
    // 16 hex chars — stable across Rust versions (unlike std::collections::hash_map::DefaultHasher).
    format!("{digest:x}")[..16].to_string()
}

#[derive(Debug, thiserror::Error)]
pub enum WorkspaceCacheError {
    #[error("HOME is not set")]
    NoHomeDir,
    #[error("failed to canonicalize workspace path {path}: {source}")]
    Canonicalize {
        path: PathBuf,
        source: std::io::Error,
    },
}

pub fn workspace_config_path(workspace_root: &Path) -> PathBuf {
    workspace_root.join(".nca").join("config.local.toml")
}

fn load_partial(path: &Path) -> Result<PartialNcaConfig, ConfigError> {
    let raw = std::fs::read_to_string(path).map_err(|source| ConfigError::ReadFile {
        path: path.to_path_buf(),
        source,
    })?;

    toml::from_str(&raw).map_err(|source| ConfigError::ParseToml {
        path: path.to_path_buf(),
        source,
    })
}

fn save_config_to_path(config: &NcaConfig, path: &Path) -> Result<(), ConfigError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| ConfigError::Io {
            action: "create config directory",
            path: parent.to_path_buf(),
            source,
        })?;
    }

    let raw = toml::to_string_pretty(config).map_err(|source| ConfigError::SerializeToml {
        path: path.to_path_buf(),
        source,
    })?;

    std::fs::write(path, raw).map_err(|source| ConfigError::Io {
        action: "write config file",
        path: path.to_path_buf(),
        source,
    })
}

/// Recursively compute the diff of two TOML values.
/// Returns `Some(diff)` containing only the fields that differ,
/// or `None` if they are identical.
fn diff_toml_values(current: &toml::Value, base: &toml::Value) -> Option<toml::Value> {
    if current == base {
        return None;
    }
    match (current, base) {
        (toml::Value::Table(curr), toml::Value::Table(base_t)) => {
            let mut diff_table = toml::map::Map::new();
            for (key, val) in curr.iter() {
                if let Some(base_val) = base_t.get(key) {
                    if let Some(sub_diff) = diff_toml_values(val, base_val) {
                        diff_table.insert(key.clone(), sub_diff);
                    }
                } else {
                    // Key exists in current but not in base → include it.
                    diff_table.insert(key.clone(), val.clone());
                }
            }
            if diff_table.is_empty() {
                None
            } else {
                Some(toml::Value::Table(diff_table))
            }
        }
        (_, _) => Some(current.clone()),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("unable to determine the home directory for global config")]
    NoHomeDir,
    #[error("failed to read config file {path}: {source}")]
    ReadFile {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse config file {path}: {source}")]
    ParseToml {
        path: PathBuf,
        source: toml::de::Error,
    },
    #[error("failed to serialize config file {path}: {source}")]
    SerializeToml {
        path: PathBuf,
        source: toml::ser::Error,
    },
    #[error("failed to {action} at {path}: {source}")]
    Io {
        action: &'static str,
        path: PathBuf,
        source: std::io::Error,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub default: ProviderKind,
    pub minimax: MiniMaxConfig,
    pub openai: OpenAiConfig,
    pub anthropic: AnthropicConfig,
    pub openrouter: OpenRouterConfig,
    pub zhipuai: ZhipuAIConfig,
    pub deepseek: DeepSeekConfig,
    pub kimi: KimiConfig,
    pub mimo: MimoConfig,
    #[serde(default)]
    pub custom: CustomProviderConfig,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            default: ProviderKind::DeepSeek,
            minimax: MiniMaxConfig::default(),
            openai: OpenAiConfig::default(),
            anthropic: AnthropicConfig::default(),
            openrouter: OpenRouterConfig::default(),
            zhipuai: ZhipuAIConfig::default(),
            deepseek: DeepSeekConfig::default(),
            kimi: KimiConfig::default(),
            mimo: MimoConfig::default(),
            custom: CustomProviderConfig::default(),
        }
    }
}

impl ProviderConfig {
    fn merge(&mut self, partial: PartialProviderConfig) {
        if let Some(default) = partial.default {
            self.default = default;
        }

        if let Some(minimax) = partial.minimax {
            self.minimax.merge(minimax);
        }
        if let Some(openai) = partial.openai {
            self.openai.merge(openai);
        }
        if let Some(anthropic) = partial.anthropic {
            self.anthropic.merge(anthropic);
        }
        if let Some(openrouter) = partial.openrouter {
            self.openrouter.merge(openrouter);
        }
        if let Some(zhipuai) = partial.zhipuai {
            self.zhipuai.merge(zhipuai);
        }
        if let Some(deepseek) = partial.deepseek {
            self.deepseek.merge(deepseek);
        }
        if let Some(kimi) = partial.kimi {
            self.kimi.merge(kimi);
        }
        if let Some(mimo) = partial.mimo {
            self.mimo.merge(mimo);
        }
        if let Some(custom) = partial.custom {
            self.custom.merge(custom);
        }
    }

    pub fn active_model(&self) -> &str {
        match self.default {
            ProviderKind::MiniMax => &self.minimax.model,
            ProviderKind::OpenRouter => &self.openrouter.model,
            ProviderKind::Anthropic => &self.anthropic.model,
            ProviderKind::OpenAi => &self.openai.model,
            ProviderKind::ZhipuAI => &self.zhipuai.model,
            ProviderKind::DeepSeek => &self.deepseek.model,
            ProviderKind::Kimi => &self.kimi.model,
            ProviderKind::Mimo => &self.mimo.model,
            ProviderKind::Custom => &self.custom.model,
        }
    }

    pub fn set_model_for_default(&mut self, model: impl Into<String>) {
        self.set_model_for(self.default, model);
    }

    pub fn set_model_for(&mut self, provider: ProviderKind, model: impl Into<String>) {
        let model = model.into();
        match provider {
            ProviderKind::MiniMax => self.minimax.model = model,
            ProviderKind::OpenRouter => self.openrouter.model = model,
            ProviderKind::Anthropic => self.anthropic.model = model,
            ProviderKind::OpenAi => self.openai.model = model,
            ProviderKind::ZhipuAI => self.zhipuai.model = model,
            ProviderKind::DeepSeek => self.deepseek.model = model,
            ProviderKind::Kimi => self.kimi.model = model,
            ProviderKind::Mimo => self.mimo.model = model,
            ProviderKind::Custom => self.custom.model = model,
        }
    }

    pub fn model_for(&self, provider: ProviderKind) -> &str {
        match provider {
            ProviderKind::MiniMax => &self.minimax.model,
            ProviderKind::OpenRouter => &self.openrouter.model,
            ProviderKind::Anthropic => &self.anthropic.model,
            ProviderKind::OpenAi => &self.openai.model,
            ProviderKind::ZhipuAI => &self.zhipuai.model,
            ProviderKind::DeepSeek => &self.deepseek.model,
            ProviderKind::Kimi => &self.kimi.model,
            ProviderKind::Mimo => &self.mimo.model,
            ProviderKind::Custom => &self.custom.model,
        }
    }

    pub fn base_url_for(&self, provider: ProviderKind) -> &str {
        match provider {
            ProviderKind::MiniMax => &self.minimax.base_url,
            ProviderKind::OpenRouter => &self.openrouter.base_url,
            ProviderKind::Anthropic => &self.anthropic.base_url,
            ProviderKind::OpenAi => &self.openai.base_url,
            ProviderKind::ZhipuAI => &self.zhipuai.base_url,
            ProviderKind::DeepSeek => &self.deepseek.base_url,
            ProviderKind::Kimi => &self.kimi.base_url,
            ProviderKind::Mimo => &self.mimo.base_url,
            ProviderKind::Custom => &self.custom.base_url,
        }
    }

    pub fn api_key_env_for(&self, provider: ProviderKind) -> &str {
        match provider {
            ProviderKind::MiniMax => &self.minimax.api_key_env,
            ProviderKind::OpenRouter => &self.openrouter.api_key_env,
            ProviderKind::Anthropic => &self.anthropic.api_key_env,
            ProviderKind::OpenAi => &self.openai.api_key_env,
            ProviderKind::ZhipuAI => &self.zhipuai.api_key_env,
            ProviderKind::DeepSeek => &self.deepseek.api_key_env,
            ProviderKind::Kimi => &self.kimi.api_key_env,
            ProviderKind::Mimo => &self.mimo.api_key_env,
            ProviderKind::Custom => &self.custom.api_key_env,
        }
    }

    pub fn api_key_present_for(&self, provider: ProviderKind) -> bool {
        match provider {
            ProviderKind::MiniMax => self.minimax.resolve_api_key().is_some(),
            ProviderKind::OpenRouter => self.openrouter.resolve_api_key().is_some(),
            ProviderKind::Anthropic => self.anthropic.resolve_api_key().is_some(),
            ProviderKind::OpenAi => self.openai.resolve_api_key().is_some(),
            ProviderKind::ZhipuAI => self.zhipuai.resolve_api_key().is_some(),
            ProviderKind::DeepSeek => self.deepseek.resolve_api_key().is_some(),
            ProviderKind::Kimi => self.kimi.resolve_api_key().is_some(),
            ProviderKind::Mimo => self.mimo.resolve_api_key().is_some(),
            ProviderKind::Custom => self.custom.resolve_api_key().is_some(),
        }
    }

    /// Returns `true` if at least one provider has an API key configured
    /// (either in config or via environment variable).
    pub fn any_api_key_present(&self) -> bool {
        ProviderKind::ALL
            .iter()
            .any(|p| self.api_key_present_for(*p))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ProviderCompatibility {
    OpenAi,
    Anthropic,
}

impl ProviderCompatibility {
    pub fn from_cli_name(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "openai" | "open-ai" => Some(Self::OpenAi),
            "anthropic" | "claude" => Some(Self::Anthropic),
            _ => None,
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::OpenAi => "OpenAI-compatible",
            Self::Anthropic => "Anthropic-compatible",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    MiniMax,
    OpenRouter,
    Anthropic,
    OpenAi,
    ZhipuAI,
    DeepSeek,
    Kimi,
    Mimo,
    Custom,
}

impl ProviderKind {
    pub const ALL: [ProviderKind; 9] = [
        ProviderKind::MiniMax,
        ProviderKind::OpenAi,
        ProviderKind::Anthropic,
        ProviderKind::OpenRouter,
        ProviderKind::ZhipuAI,
        ProviderKind::DeepSeek,
        ProviderKind::Kimi,
        ProviderKind::Mimo,
        ProviderKind::Custom,
    ];

    /// Parse user/CLI input (slash commands, TUI pickers).
    pub fn from_cli_name(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "minimax" | "mini-max" | "minimaxi" => Some(Self::MiniMax),
            "openai" | "open-ai" | "gpt" => Some(Self::OpenAi),
            "anthropic" | "claude" => Some(Self::Anthropic),
            "openrouter" | "open-router" => Some(Self::OpenRouter),
            "zhipuai" | "zhipu" | "glm" | "glm-5" | "glm-5.2" | "glm-5.3" | "glm-5.3-flash" => {
                Some(Self::ZhipuAI)
            }
            "deepseek" => Some(Self::DeepSeek),
            "kimi" | "k3" | "kimi-k3" => Some(Self::Kimi),
            "mimo" | "xiaomi" => Some(Self::Mimo),
            "custom" => Some(Self::Custom),
            _ => None,
        }
    }

    fn from_env(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "openrouter" => Self::OpenRouter,
            "anthropic" => Self::Anthropic,
            "openai" => Self::OpenAi,
            "zhipuai" | "zhipu" | "glm" => Self::ZhipuAI,
            "deepseek" => Self::DeepSeek,
            "kimi" => Self::Kimi,
            "mimo" | "xiaomi" => Self::Mimo,
            "custom" => Self::Custom,
            _ => Self::MiniMax,
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            ProviderKind::MiniMax => "MiniMax",
            ProviderKind::OpenRouter => "OpenRouter",
            ProviderKind::Anthropic => "Anthropic",
            ProviderKind::OpenAi => "OpenAI",
            ProviderKind::ZhipuAI => "ZhipuAI",
            ProviderKind::DeepSeek => "DeepSeek",
            ProviderKind::Kimi => "Kimi",
            ProviderKind::Mimo => "MiMo",
            ProviderKind::Custom => "Custom",
        }
    }

    /// Match [`display_name`](Self::display_name) output (case-insensitive).
    pub fn parse_display_name(s: &str) -> Option<Self> {
        let t = s.trim();
        Self::ALL
            .into_iter()
            .find(|k| k.display_name().eq_ignore_ascii_case(t))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiniMaxConfig {
    pub api_key_env: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub model: String,
    pub temperature: f32,
}

impl Default for MiniMaxConfig {
    fn default() -> Self {
        Self {
            api_key_env: "MINIMAX_API_KEY".into(),
            api_key: None,
            // Anthropic-compatible endpoint (recommended for agentic/coding use).
            // International: https://api.minimax.io/anthropic
            // China:         https://api.minimaxi.com/anthropic
            base_url: "https://api.minimax.io/anthropic".into(),
            model: "MiniMax-M2.7".into(),
            temperature: 0.7,
        }
    }
}

impl MiniMaxConfig {
    pub fn resolve_api_key(&self) -> Option<String> {
        resolve_api_key_value(&self.api_key, &self.api_key_env)
    }

    fn merge(&mut self, partial: PartialMiniMaxConfig) {
        if let Some(api_key_env) = partial.api_key_env {
            self.api_key_env = api_key_env;
        }
        if let Some(api_key) = partial.api_key {
            self.api_key = Some(api_key);
        }
        if let Some(base_url) = partial.base_url {
            self.base_url = base_url;
        }
        if let Some(model) = partial.model {
            self.model = model;
        }
        if let Some(temperature) = partial.temperature {
            self.temperature = temperature;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiConfig {
    pub api_key_env: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub model: String,
    pub temperature: f32,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            api_key_env: "OPENAI_API_KEY".into(),
            api_key: None,
            base_url: "https://api.openai.com".into(),
            model: "gpt-4o-mini".into(),
            temperature: 0.7,
        }
    }
}

impl OpenAiConfig {
    pub fn resolve_api_key(&self) -> Option<String> {
        resolve_api_key_value(&self.api_key, &self.api_key_env)
    }

    fn merge(&mut self, partial: PartialOpenAiConfig) {
        if let Some(api_key_env) = partial.api_key_env {
            self.api_key_env = api_key_env;
        }
        if let Some(api_key) = partial.api_key {
            self.api_key = Some(api_key);
        }
        if let Some(base_url) = partial.base_url {
            self.base_url = base_url;
        }
        if let Some(model) = partial.model {
            self.model = model;
        }
        if let Some(temperature) = partial.temperature {
            self.temperature = temperature;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicConfig {
    pub api_key_env: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub model: String,
    pub temperature: f32,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_key_env: "ANTHROPIC_API_KEY".into(),
            api_key: None,
            base_url: "https://api.anthropic.com".into(),
            model: "claude-3-7-sonnet-latest".into(),
            temperature: 1.0,
        }
    }
}

impl AnthropicConfig {
    pub fn resolve_api_key(&self) -> Option<String> {
        resolve_api_key_value(&self.api_key, &self.api_key_env)
    }

    fn merge(&mut self, partial: PartialAnthropicConfig) {
        if let Some(api_key_env) = partial.api_key_env {
            self.api_key_env = api_key_env;
        }
        if let Some(api_key) = partial.api_key {
            self.api_key = Some(api_key);
        }
        if let Some(base_url) = partial.base_url {
            self.base_url = base_url;
        }
        if let Some(model) = partial.model {
            self.model = model;
        }
        if let Some(temperature) = partial.temperature {
            self.temperature = temperature;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterConfig {
    pub api_key_env: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub model: String,
    pub temperature: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub site_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_name: Option<String>,
}

impl Default for OpenRouterConfig {
    fn default() -> Self {
        Self {
            api_key_env: "OPENROUTER_API_KEY".into(),
            api_key: None,
            base_url: "https://openrouter.ai/api".into(),
            model: "openai/gpt-4o-mini".into(),
            temperature: 0.7,
            site_url: None,
            app_name: None,
        }
    }
}

impl OpenRouterConfig {
    pub fn resolve_api_key(&self) -> Option<String> {
        resolve_api_key_value(&self.api_key, &self.api_key_env)
    }

    fn merge(&mut self, partial: PartialOpenRouterConfig) {
        if let Some(api_key_env) = partial.api_key_env {
            self.api_key_env = api_key_env;
        }
        if let Some(api_key) = partial.api_key {
            self.api_key = Some(api_key);
        }
        if let Some(base_url) = partial.base_url {
            self.base_url = base_url;
        }
        if let Some(model) = partial.model {
            self.model = model;
        }
        if let Some(temperature) = partial.temperature {
            self.temperature = temperature;
        }
        if let Some(site_url) = partial.site_url {
            self.site_url = Some(site_url);
        }
        if let Some(app_name) = partial.app_name {
            self.app_name = Some(app_name);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZhipuAIConfig {
    pub api_key_env: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub model: String,
    pub temperature: f32,
}

impl Default for ZhipuAIConfig {
    fn default() -> Self {
        Self {
            api_key_env: "ZHIPUAI_API_KEY".into(),
            api_key: None,
            base_url: "https://open.bigmodel.cn/api/coding/paas/v4".into(),
            model: "glm-5.3".into(),
            temperature: 0.7,
        }
    }
}

impl ZhipuAIConfig {
    pub fn resolve_api_key(&self) -> Option<String> {
        resolve_api_key_value(&self.api_key, &self.api_key_env)
    }

    fn merge(&mut self, partial: PartialZhipuAIConfig) {
        if let Some(api_key_env) = partial.api_key_env {
            self.api_key_env = api_key_env;
        }
        if let Some(api_key) = partial.api_key {
            self.api_key = Some(api_key);
        }
        if let Some(base_url) = partial.base_url {
            self.base_url = base_url;
        }
        if let Some(model) = partial.model {
            self.model = model;
        }
        if let Some(temperature) = partial.temperature {
            self.temperature = temperature;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekConfig {
    pub api_key_env: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub model: String,
    pub temperature: f32,
}

impl Default for DeepSeekConfig {
    fn default() -> Self {
        Self {
            api_key_env: "DEEPSEEK_API_KEY".into(),
            api_key: None,
            base_url: "https://api.deepseek.com".into(),
            model: "deepseek-flash".into(),
            temperature: 0.7,
        }
    }
}

impl DeepSeekConfig {
    pub fn resolve_api_key(&self) -> Option<String> {
        resolve_api_key_value(&self.api_key, &self.api_key_env)
    }

    fn merge(&mut self, partial: PartialDeepSeekConfig) {
        if let Some(api_key_env) = partial.api_key_env {
            self.api_key_env = api_key_env;
        }
        if let Some(api_key) = partial.api_key {
            self.api_key = Some(api_key);
        }
        if let Some(base_url) = partial.base_url {
            self.base_url = base_url;
        }
        if let Some(model) = partial.model {
            self.model = model;
        }
        if let Some(temperature) = partial.temperature {
            self.temperature = temperature;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KimiConfig {
    pub api_key_env: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub model: String,
    pub temperature: f32,
}

impl Default for KimiConfig {
    fn default() -> Self {
        Self {
            api_key_env: "KIMI_API_KEY".into(),
            api_key: None,
            // Kimi for Coding serves an Anthropic-compatible API.
            // Endpoint is `{base_url}/v1/messages` → https://api.kimi.com/coding/v1/messages
            base_url: "https://api.kimi.com/coding".into(),
            // Kimi K3 flagship: 1M context, 131K output, reasoning + tool use.
            model: "k3".into(),
            temperature: 0.7,
        }
    }
}

impl KimiConfig {
    pub fn resolve_api_key(&self) -> Option<String> {
        resolve_api_key_value(&self.api_key, &self.api_key_env)
    }

    fn merge(&mut self, partial: PartialKimiConfig) {
        if let Some(api_key_env) = partial.api_key_env {
            self.api_key_env = api_key_env;
        }
        if let Some(api_key) = partial.api_key {
            self.api_key = Some(api_key);
        }
        if let Some(base_url) = partial.base_url {
            self.base_url = base_url;
        }
        if let Some(model) = partial.model {
            self.model = model;
        }
        if let Some(temperature) = partial.temperature {
            self.temperature = temperature;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MimoConfig {
    pub api_key_env: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub model: String,
    pub temperature: f32,
}

impl Default for MimoConfig {
    fn default() -> Self {
        Self {
            api_key_env: "MIMO_API_KEY".into(),
            api_key: None,
            // Xiaomi MiMo serves an OpenAI-compatible API.
            // Endpoint is `{base_url}/chat/completions`
            // → https://api.xiaomimimo.com/v1/chat/completions
            base_url: "https://api.xiaomimimo.com/v1".into(),
            // MiMo flagship: 1M context, 131K output, omnimodal, deep thinking.
            model: "mimo-v2.6-pro".into(),
            temperature: 0.7,
        }
    }
}

impl MimoConfig {
    pub fn resolve_api_key(&self) -> Option<String> {
        resolve_api_key_value(&self.api_key, &self.api_key_env)
    }

    fn merge(&mut self, partial: PartialMimoConfig) {
        if let Some(api_key_env) = partial.api_key_env {
            self.api_key_env = api_key_env;
        }
        if let Some(api_key) = partial.api_key {
            self.api_key = Some(api_key);
        }
        if let Some(base_url) = partial.base_url {
            self.base_url = base_url;
        }
        if let Some(model) = partial.model {
            self.model = model;
        }
        if let Some(temperature) = partial.temperature {
            self.temperature = temperature;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomProviderConfig {
    pub api_key_env: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub model: String,
    pub temperature: f32,
    pub compatibility: ProviderCompatibility,
}

impl Default for CustomProviderConfig {
    fn default() -> Self {
        Self {
            api_key_env: "CUSTOM_PROVIDER_API_KEY".into(),
            api_key: None,
            base_url: String::new(),
            model: "custom-model".into(),
            temperature: 0.7,
            compatibility: ProviderCompatibility::OpenAi,
        }
    }
}

impl CustomProviderConfig {
    pub fn resolve_api_key(&self) -> Option<String> {
        resolve_api_key_value(&self.api_key, &self.api_key_env)
    }

    fn merge(&mut self, partial: PartialCustomProviderConfig) {
        if let Some(api_key_env) = partial.api_key_env {
            self.api_key_env = api_key_env;
        }
        if let Some(api_key) = partial.api_key {
            self.api_key = Some(api_key);
        }
        if let Some(base_url) = partial.base_url {
            self.base_url = base_url;
        }
        if let Some(model) = partial.model {
            self.model = model;
        }
        if let Some(temperature) = partial.temperature {
            self.temperature = temperature;
        }
        if let Some(compatibility) = partial.compatibility {
            self.compatibility = compatibility;
        }
    }
}

/// Common interface for OpenAI-compatible provider configs.
/// Shared by OpenAiConfig, OpenRouterConfig, ZhipuAIConfig, DeepSeekConfig,
/// and MimoConfig.
pub trait OpenAiCompatConfig {
    fn resolve_api_key(&self) -> Option<String>;
    fn api_key_env(&self) -> &str;
    fn base_url(&self) -> &str;
    fn model(&self) -> &str;
    fn temperature(&self) -> f32;
}

impl OpenAiCompatConfig for OpenAiConfig {
    fn resolve_api_key(&self) -> Option<String> {
        self.resolve_api_key()
    }
    fn api_key_env(&self) -> &str {
        &self.api_key_env
    }
    fn base_url(&self) -> &str {
        &self.base_url
    }
    fn model(&self) -> &str {
        &self.model
    }
    fn temperature(&self) -> f32 {
        self.temperature
    }
}

impl OpenAiCompatConfig for OpenRouterConfig {
    fn resolve_api_key(&self) -> Option<String> {
        self.resolve_api_key()
    }
    fn api_key_env(&self) -> &str {
        &self.api_key_env
    }
    fn base_url(&self) -> &str {
        &self.base_url
    }
    fn model(&self) -> &str {
        &self.model
    }
    fn temperature(&self) -> f32 {
        self.temperature
    }
}

impl OpenAiCompatConfig for ZhipuAIConfig {
    fn resolve_api_key(&self) -> Option<String> {
        self.resolve_api_key()
    }
    fn api_key_env(&self) -> &str {
        &self.api_key_env
    }
    fn base_url(&self) -> &str {
        &self.base_url
    }
    fn model(&self) -> &str {
        &self.model
    }
    fn temperature(&self) -> f32 {
        self.temperature
    }
}

impl OpenAiCompatConfig for DeepSeekConfig {
    fn resolve_api_key(&self) -> Option<String> {
        self.resolve_api_key()
    }
    fn api_key_env(&self) -> &str {
        &self.api_key_env
    }
    fn base_url(&self) -> &str {
        &self.base_url
    }
    fn model(&self) -> &str {
        &self.model
    }
    fn temperature(&self) -> f32 {
        self.temperature
    }
}

impl OpenAiCompatConfig for MimoConfig {
    fn resolve_api_key(&self) -> Option<String> {
        self.resolve_api_key()
    }
    fn api_key_env(&self) -> &str {
        &self.api_key_env
    }
    fn base_url(&self) -> &str {
        &self.base_url
    }
    fn model(&self) -> &str {
        &self.model
    }
    fn temperature(&self) -> f32 {
        self.temperature
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Active model name �� always derived from `provider.active_model()`, never persisted.
    /// Kept in memory for fast access by the runtime and REPL.
    #[serde(skip)]
    pub default_model: String,
    pub max_tokens: u32,
    pub enable_thinking: bool,
    pub thinking_budget: u32,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub aliases: BTreeMap<String, String>,
    /// Last N used model names for F2 cycling.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recent_models: Vec<String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            default_model: "deepseek-flash".into(),
            max_tokens: 8192,
            enable_thinking: false,
            thinking_budget: 5120,
            aliases: default_model_aliases(),
            recent_models: Vec::new(),
        }
    }
}

impl ModelConfig {
    fn merge(&mut self, partial: PartialModelConfig) {
        if let Some(max_tokens) = partial.max_tokens {
            self.max_tokens = max_tokens;
        }
        if let Some(enable_thinking) = partial.enable_thinking {
            self.enable_thinking = enable_thinking;
        }
        if let Some(thinking_budget) = partial.thinking_budget {
            self.thinking_budget = thinking_budget;
        }
        if let Some(aliases) = partial.aliases {
            self.aliases = aliases;
        }
        if let Some(recent_models) = partial.recent_models {
            self.recent_models = recent_models;
        }
    }

    pub fn resolve_alias(&self, raw: &str) -> String {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return self.default_model.clone();
        }

        let lowered = trimmed.to_ascii_lowercase();
        self.aliases
            .get(&lowered)
            .cloned()
            .unwrap_or_else(|| trimmed.to_string())
    }

    /// Push a model name to the front of the recent list, deduplicating and capping at 8.
    pub fn track_recent_model(&mut self, model: &str) {
        self.recent_models.retain(|m| m != model);
        self.recent_models.insert(0, model.to_string());
        self.recent_models.truncate(8);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PermissionConfig {
    pub mode: PermissionMode,
    pub allow: Vec<String>,
    pub deny: Vec<String>,
    pub ask: Vec<String>,
    /// P5 sandbox configuration (Landlock backend, resolved at exec time).
    #[serde(default)]
    pub sandbox: SandboxConfig,
}

/// Kernel-sandbox enforcement mode.
///
/// - `Auto` (default): confine when the Landlock backend is available, degrade
///   to unconfined with a warn-once notice when it is not.
/// - `Required`: fail closed — commands are refused when the backend is
///   unavailable, never silently run unconfined.
/// - `Off`: never probe, never confine.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SandboxMode {
    #[default]
    Auto,
    Required,
    Off,
}

/// Sandboxing configuration under `[permissions.sandbox]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Enforcement mode (`auto` | `required` | `off`).
    #[serde(default)]
    pub mode: SandboxMode,
    /// Additional read-only roots appended to the built-in system set.
    #[serde(default)]
    pub ro_paths: Vec<PathBuf>,
    /// Additional read-write roots appended to workspace + temp defaults.
    #[serde(default)]
    pub rw_paths: Vec<PathBuf>,
    /// `true` = network stays unrestricted (blocking network is a future item;
    /// the Landlock `net` scope only covers abstract UNIX sockets).
    #[serde(default = "default_sandbox_net")]
    pub net: bool,
    /// Environment variables passed through to sandbox-confined PTY commands.
    /// Exact names are matched; additionally any variable whose name starts
    /// with `LC_` is always allowed. Setting this in config REPLACES the
    /// default list (it does not append). An empty list passes only `PATH`
    /// (structurally required to locate executables).
    #[serde(default = "default_sandbox_env_allow")]
    pub env_allow: Vec<String>,
    /// `true` (default) = paths mounted via `/mount` (persisted as
    /// `extra_paths`) are automatically granted read-write access in the
    /// Landlock policy, so shell commands can reach what file tools already
    /// can. `false` keeps mounts file-tool-only; shell access then requires
    /// an explicit `rw_paths` entry.
    #[serde(default = "default_sandbox_inherit_mounts")]
    pub inherit_mounts: bool,
    /// `true` = grant sandboxed commands access to the host audio stack
    /// (PipeWire/PulseAudio): passes `XDG_RUNTIME_DIR` through the env
    /// allowlist and adds rw grants for `$XDG_RUNTIME_DIR/pipewire-0`,
    /// `pipewire-0.lock`, and `pulse/`. SECURITY CAVEAT: this allows playback,
    /// reading audio output streams, AND — under PipeWire's default policy —
    /// capturing from input devices (microphone). Default `false`.
    #[serde(default)]
    pub host_audio: bool,
    /// `true` = grant sandboxed commands access to the host D-Bus session
    /// bus: passes `DBUS_SESSION_BUS_ADDRESS` through (or synthesizes it)
    /// and adds an rw grant for `$XDG_RUNTIME_DIR/bus`. SECURITY CAVEAT: the
    /// session bus is broad user-service access (notifications, secret
    /// service/keyring, media players, ...). Default `false`.
    #[serde(default)]
    pub host_dbus_session: bool,
    /// `true` = pass `XDG_RUNTIME_DIR` through and grant rw on the whole
    /// `$XDG_RUNTIME_DIR` (subsumes `host_audio` and `host_dbus_session`).
    /// SECURITY CAVEAT: this exposes everything living there — Wayland (GUI
    /// input events), pipewire, pulse, D-Bus session bus, and any other
    /// per-user socket a desktop session keeps. Default `false`.
    #[serde(default)]
    pub host_xdg_runtime: bool,
}

fn default_sandbox_net() -> bool {
    true
}

fn default_sandbox_inherit_mounts() -> bool {
    true
}

/// Default env allowlist for sandbox-confined PTY commands.
pub fn default_sandbox_env_allow() -> Vec<String> {
    [
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TERM",
        "LANG",
        "TZ",
        "TMPDIR",
        "CARGO_HOME",
        "RUSTUP_HOME",
        "CC",
        "CXX",
        "PKG_CONFIG_PATH",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            mode: SandboxMode::Auto,
            ro_paths: Vec::new(),
            rw_paths: Vec::new(),
            net: true,
            env_allow: default_sandbox_env_allow(),
            inherit_mounts: true,
            host_audio: false,
            host_dbus_session: false,
            host_xdg_runtime: false,
        }
    }
}

impl PermissionConfig {
    fn merge(&mut self, partial: PartialPermissionConfig) {
        if let Some(mode) = partial.mode {
            self.mode = mode;
        }
        if let Some(allow) = partial.allow {
            self.allow = allow;
        }
        if let Some(deny) = partial.deny {
            self.deny = deny;
        }
        if let Some(ask) = partial.ask {
            self.ask = ask;
        }
        if let Some(sandbox) = partial.sandbox {
            self.sandbox.merge(sandbox);
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "kebab-case")]
pub enum PermissionMode {
    #[default]
    Default,
    Plan,
    AcceptEdits,
    DontAsk,
    BypassPermissions,
}

impl std::str::FromStr for PermissionMode {
    type Err = String;
    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "default" => Ok(PermissionMode::Default),
            "plan" => Ok(PermissionMode::Plan),
            "accept-edits" | "accept_edits" | "acceptedits" => Ok(PermissionMode::AcceptEdits),
            "dont-ask" | "dont_ask" | "dontask" => Ok(PermissionMode::DontAsk),
            "bypass-permissions" | "bypass_permissions" | "bypasspermissions" => {
                Ok(PermissionMode::BypassPermissions)
            }
            _ => Err(format!("unknown permission mode: {raw}")),
        }
    }
}

impl PermissionMode {
    pub const ALL: &[Self] = &[
        Self::Default,
        Self::Plan,
        Self::AcceptEdits,
        Self::DontAsk,
        Self::BypassPermissions,
    ];

    pub fn index(self) -> usize {
        match self {
            Self::Default => 0,
            Self::Plan => 1,
            Self::AcceptEdits => 2,
            Self::DontAsk => 3,
            Self::BypassPermissions => 4,
        }
    }

    pub fn from_index(idx: usize) -> Self {
        Self::ALL.get(idx).copied().unwrap_or(Self::Default)
    }
}

/// Middleware chain configuration (`[middleware]` section). Order of the
/// chain itself is code-fixed; this carries knobs only.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MiddlewareConfig {
    /// Max retries after a `RateLimited` rejection (0 disables the retry
    /// middleware entirely).
    pub retry_max_attempts: u32,
    /// Upper bound on the server-provided retry-after wait, in
    /// milliseconds. Forward-looking: production parsers hard-code
    /// `retry_after_ms = 1000` today; the cap guards future header
    /// parsing.
    pub retry_delay_cap_ms: u64,
    /// Session spend cap in USD. Estimated with hard-coded Sonnet-class
    /// rates, which overstate actual spend for most providers — roughly
    /// 10× for DeepSeek, the primary provider — and double-count cache
    /// reads on OpenAI-compatible usage. Absent = no cost guard.
    pub cost_budget_usd: Option<f64>,
}

impl Default for MiddlewareConfig {
    fn default() -> Self {
        Self {
            retry_max_attempts: 2,
            retry_delay_cap_ms: 30_000,
            cost_budget_usd: None,
        }
    }
}

impl MiddlewareConfig {
    fn merge(&mut self, partial: PartialMiddlewareConfig) {
        if let Some(retry_max_attempts) = partial.retry_max_attempts {
            self.retry_max_attempts = retry_max_attempts;
        }
        if let Some(retry_delay_cap_ms) = partial.retry_delay_cap_ms {
            self.retry_delay_cap_ms = retry_delay_cap_ms;
        }
        if let Some(cost_budget_usd) = partial.cost_budget_usd {
            self.cost_budget_usd = Some(cost_budget_usd);
        }
    }
}

/// Provider fallback (failover) configuration (`[fallback]` section).
///
/// When enabled, the session's primary provider is wrapped in a
/// [`nca_core::provider::fallback::FallbackProvider`] that retries a failed
/// step against the ordered `chain` of alternate providers (each using its
/// own `[provider.<name>]` credentials and model). Off by default.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct FallbackConfig {
    /// Master switch. `false` (the default) keeps single-provider behavior.
    pub enabled: bool,
    /// Ordered fallback provider names, each parseable by
    /// [`ProviderKind::from_cli_name`] (e.g. `["openai", "zhipuai"]`).
    /// Entries matching the primary provider (or repeated entries) are
    /// skipped at build time — a provider that just failed is not retried
    /// immediately.
    pub chain: Vec<String>,
    /// Delay before the FIRST failover switch of a provider instance, in
    /// milliseconds. Default 0: the first switch is immediate.
    pub initial_retry_delay_ms: u64,
    /// Minimum spacing between consecutive failover switches (anti-storm
    /// throttle), in milliseconds. Default 500.
    pub retry_delay_ms: u64,
}

impl Default for FallbackConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            chain: Vec::new(),
            initial_retry_delay_ms: 0,
            retry_delay_ms: 500,
        }
    }
}

impl FallbackConfig {
    fn merge(&mut self, partial: PartialFallbackConfig) {
        if let Some(enabled) = partial.enabled {
            self.enabled = enabled;
        }
        if let Some(chain) = partial.chain {
            self.chain = chain;
        }
        if let Some(initial_retry_delay_ms) = partial.initial_retry_delay_ms {
            self.initial_retry_delay_ms = initial_retry_delay_ms;
        }
        if let Some(retry_delay_ms) = partial.retry_delay_ms {
            self.retry_delay_ms = retry_delay_ms;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    pub history_dir: PathBuf,
    #[serde(alias = "max_turn_per_run")]
    pub max_turns_per_run: u32,
    pub max_tool_calls_per_turn: u32,
    pub checkpoint_interval: u32,
    /// File that stores the last active session ID for auto-resume.
    pub last_session_file: PathBuf,
    /// Auto-compact when switching away from a session.
    #[serde(default)]
    pub auto_compact_on_finish: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            history_dir: PathBuf::from(".nca/sessions"),
            max_turns_per_run: 1024,
            max_tool_calls_per_turn: 200,
            checkpoint_interval: 5,
            last_session_file: PathBuf::from(".nca/.last_session"),
            auto_compact_on_finish: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessConfig {
    pub built_in_enabled: bool,
    /// Path to a global instructions file (e.g. `$XDG_CONFIG_HOME/nca/AGENTS.md`).
    /// Loaded before the workspace `AGENTS.md`, shared across all projects.
    /// Supports `~` expansion to `$HOME`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_instructions_path: Option<PathBuf>,
    pub project_instructions_path: PathBuf,
    pub local_instructions_path: PathBuf,
    pub skill_directories: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct McpConfig {
    #[serde(default)]
    pub expose_in_safe_mode: bool,
    #[serde(default)]
    pub servers: Vec<McpServerConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub file_path: PathBuf,
    #[serde(default = "default_max_memory_notes")]
    pub max_notes: usize,
    #[serde(default)]
    pub auto_compact_on_finish: bool,
    /// Context management configuration.
    #[serde(default)]
    pub context: ContextConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Target context window size (approximate tokens).
    /// Set to 0 for auto-detection based on model, or specify a custom value.
    /// Auto-detection uses known model context windows.
    #[serde(default)]
    pub context_window_target: usize,
    /// Use model-specific context window detection.
    /// When true, ignores context_window_target and auto-detects from model name.
    #[serde(default = "default_true")]
    pub auto_detect_context_window: bool,
    /// When true with `auto_detect_context_window`, query the active provider's models API
    /// before falling back to built-in tables. OpenRouter's catalog is public; OpenAI and
    /// Anthropic require configured API keys. Set `NCA_SKIP_CONTEXT_API=1` to disable at runtime.
    /// Catalog responses are cached in-process; override TTL with `NCA_CONTEXT_API_CACHE_TTL_SECS`.
    #[serde(default = "default_true")]
    pub query_provider_models_api: bool,
    /// Maximum messages to retain after compaction.
    #[serde(default = "default_max_retained_messages")]
    pub max_retained_messages: usize,
    /// Percentage of context window that triggers auto-summarize (0-100).
    #[serde(default = "default_summarize_threshold")]
    pub auto_summarize_threshold: u8,
    /// Enable automatic context summarization.
    #[serde(default = "default_true")]
    pub enable_auto_summarize: bool,
    /// Opt-in deterministic provider-request compaction.
    /// `off` (default) sends canonical history unchanged.
    /// `dry_run` computes savings diagnostics but still sends the full history.
    /// `on` sends a compact cloned view while persisting canonical history.
    #[serde(default)]
    pub smart_compaction_mode: SmartCompactionMode,
}

/// Provider-request smart compaction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SmartCompactionMode {
    #[default]
    Off,
    DryRun,
    On,
}

impl SmartCompactionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::DryRun => "dry_run",
            Self::On => "on",
        }
    }

    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            context_window_target: 0, // 0 means auto-detect
            auto_detect_context_window: true,
            query_provider_models_api: true,
            max_retained_messages: default_max_retained_messages(),
            auto_summarize_threshold: default_summarize_threshold(),
            enable_auto_summarize: default_true(),
            smart_compaction_mode: SmartCompactionMode::Off,
        }
    }
}

fn default_summarize_threshold() -> u8 {
    75
}

fn default_max_retained_messages() -> usize {
    50
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HookConfig {
    #[serde(default)]
    pub session_start: Vec<HookCommand>,
    #[serde(default)]
    pub session_end: Vec<HookCommand>,
    #[serde(default)]
    pub pre_tool_use: Vec<HookCommand>,
    #[serde(default)]
    pub post_tool_use: Vec<HookCommand>,
    #[serde(default)]
    pub post_tool_failure: Vec<HookCommand>,
    #[serde(default)]
    pub approval_requested: Vec<HookCommand>,
    #[serde(default)]
    pub subagent_start: Vec<HookCommand>,
    #[serde(default)]
    pub subagent_stop: Vec<HookCommand>,
    #[serde(default)]
    pub turn_complete: Vec<HookCommand>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookCommand {
    pub command: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matcher: Option<String>,
    #[serde(default)]
    pub blocking: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebConfig {
    pub timeout_secs: u64,
    pub max_fetch_chars: usize,
    pub default_search_limit: usize,
    pub user_agent: String,
    pub search_endpoint: String,
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 15,
            max_fetch_chars: 25_000,
            default_search_limit: 5,
            user_agent: "nca/0.5 (+https://github.com/user/native-cli-ai)".into(),
            search_endpoint: "https://cn.bing.com/search".into(),
        }
    }
}

impl WebConfig {
    fn merge(&mut self, partial: PartialWebConfig) {
        if let Some(timeout_secs) = partial.timeout_secs {
            self.timeout_secs = timeout_secs;
        }
        if let Some(max_fetch_chars) = partial.max_fetch_chars {
            self.max_fetch_chars = max_fetch_chars;
        }
        if let Some(default_search_limit) = partial.default_search_limit {
            self.default_search_limit = default_search_limit;
        }
        if let Some(user_agent) = partial.user_agent {
            self.user_agent = user_agent;
        }
        if let Some(search_endpoint) = partial.search_endpoint {
            self.search_endpoint = search_endpoint;
        }
    }
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            built_in_enabled: true,
            global_instructions_path: None,
            project_instructions_path: PathBuf::from(".ncarc"),
            local_instructions_path: PathBuf::from(".nca/instructions.md"),
            skill_directories: default_skill_directories(),
        }
    }
}

impl HarnessConfig {
    fn merge(&mut self, partial: PartialHarnessConfig) {
        if let Some(enabled) = partial.built_in_enabled {
            self.built_in_enabled = enabled;
        }
        if let Some(path) = partial.project_instructions_path {
            self.project_instructions_path = path;
        }
        if let Some(path) = partial.local_instructions_path {
            self.local_instructions_path = path;
        }
        if let Some(global_instructions_path) = partial.global_instructions_path {
            self.global_instructions_path = Some(global_instructions_path);
        }
        if let Some(skill_directories) = partial.skill_directories {
            // Expand `~` to `$HOME` for each directory so users can write
            // `skill_directories = ["~/.config/nca/skills"]` in config.toml.
            self.skill_directories = skill_directories
                .into_iter()
                .map(|p| expand_tilde(&p))
                .collect();
        }
    }

    /// Expand `~` at the start of the path to `$HOME`.
    /// Returns the path unchanged if `~` expansion is not applicable.
    pub fn resolve_global_instructions_path(&self) -> Option<PathBuf> {
        let raw = self.global_instructions_path.as_ref()?;
        Some(expand_tilde(raw))
    }
}

impl McpConfig {
    fn merge(&mut self, partial: PartialMcpConfig) {
        if let Some(expose_in_safe_mode) = partial.expose_in_safe_mode {
            self.expose_in_safe_mode = expose_in_safe_mode;
        }
        if let Some(servers) = partial.servers {
            self.servers = servers;
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            file_path: PathBuf::from(".nca/memory.json"),
            max_notes: default_max_memory_notes(),
            auto_compact_on_finish: false,
            context: ContextConfig::default(),
        }
    }
}

impl MemoryConfig {
    fn merge(&mut self, partial: PartialMemoryConfig) {
        if let Some(file_path) = partial.file_path {
            self.file_path = file_path;
        }
        if let Some(max_notes) = partial.max_notes {
            self.max_notes = max_notes;
        }
        if let Some(auto_compact_on_finish) = partial.auto_compact_on_finish {
            self.auto_compact_on_finish = auto_compact_on_finish;
        }
        if let Some(context) = partial.context {
            self.context.merge(context);
        }
    }
}

impl ContextConfig {
    fn merge(&mut self, partial: PartialContextConfig) {
        if let Some(auto_detect) = partial.auto_detect_context_window {
            self.auto_detect_context_window = auto_detect;
        }
        if let Some(context_window_target) = partial.context_window_target {
            self.context_window_target = context_window_target;
        }
        if let Some(max_retained_messages) = partial.max_retained_messages {
            self.max_retained_messages = max_retained_messages;
        }
        if let Some(auto_summarize_threshold) = partial.auto_summarize_threshold {
            self.auto_summarize_threshold = auto_summarize_threshold;
        }
        if let Some(enable_auto_summarize) = partial.enable_auto_summarize {
            self.enable_auto_summarize = enable_auto_summarize;
        }
        if let Some(query_provider_models_api) = partial.query_provider_models_api {
            self.query_provider_models_api = query_provider_models_api;
        }
        if let Some(smart_compaction_mode) = partial.smart_compaction_mode {
            self.smart_compaction_mode = smart_compaction_mode;
        }
    }
}

impl HookConfig {
    fn merge(&mut self, partial: PartialHookConfig) {
        if let Some(session_start) = partial.session_start {
            self.session_start = session_start;
        }
        if let Some(session_end) = partial.session_end {
            self.session_end = session_end;
        }
        if let Some(pre_tool_use) = partial.pre_tool_use {
            self.pre_tool_use = pre_tool_use;
        }
        if let Some(post_tool_use) = partial.post_tool_use {
            self.post_tool_use = post_tool_use;
        }
        if let Some(post_tool_failure) = partial.post_tool_failure {
            self.post_tool_failure = post_tool_failure;
        }
        if let Some(approval_requested) = partial.approval_requested {
            self.approval_requested = approval_requested;
        }
        if let Some(subagent_start) = partial.subagent_start {
            self.subagent_start = subagent_start;
        }
        if let Some(subagent_stop) = partial.subagent_stop {
            self.subagent_stop = subagent_stop;
        }
        if let Some(turn_complete) = partial.turn_complete {
            self.turn_complete = turn_complete;
        }
    }
}

impl SessionConfig {
    fn merge(&mut self, partial: PartialSessionConfig) {
        if let Some(history_dir) = partial.history_dir {
            self.history_dir = history_dir;
        }
        if let Some(max_turns_per_run) = partial.max_turns_per_run {
            self.max_turns_per_run = max_turns_per_run;
        }
        if let Some(max_tool_calls_per_turn) = partial.max_tool_calls_per_turn {
            self.max_tool_calls_per_turn = max_tool_calls_per_turn;
        }
        if let Some(checkpoint_interval) = partial.checkpoint_interval {
            self.checkpoint_interval = checkpoint_interval;
        }
        if let Some(last_session_file) = partial.last_session_file {
            self.last_session_file = last_session_file;
        }
        if let Some(auto_compact_on_finish) = partial.auto_compact_on_finish {
            self.auto_compact_on_finish = auto_compact_on_finish;
        }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialNcaConfig {
    provider: Option<PartialProviderConfig>,
    model: Option<PartialModelConfig>,
    permissions: Option<PartialPermissionConfig>,
    session: Option<PartialSessionConfig>,
    harness: Option<PartialHarnessConfig>,
    mcp: Option<PartialMcpConfig>,
    memory: Option<PartialMemoryConfig>,
    hooks: Option<PartialHookConfig>,
    web: Option<PartialWebConfig>,
    ui: Option<PartialUiConfig>,
    middleware: Option<PartialMiddlewareConfig>,
    fallback: Option<PartialFallbackConfig>,
    agents: Option<BTreeMap<String, PartialAgentProfileConfig>>,
    plans: Option<BTreeMap<String, BTreeMap<String, PlanEntry>>>,
    active_plan: Option<String>,
    extra_paths: Option<Vec<PathBuf>>,
    subagent: Option<PartialSubagentConfig>,
    plugins: Option<PartialPluginConfig>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialSubagentConfig {
    background: Option<bool>,
    result_timeout_ms: Option<u64>,
    wake: Option<PartialWakeConfig>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialWakeConfig {
    enabled: Option<bool>,
    interval_ms: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialUiConfig {
    editor: Option<String>,
    theme: Option<String>,
    hide_tips: Option<bool>,
    scroll_speed: Option<u16>,
    onboarding_completed: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialProviderConfig {
    default: Option<ProviderKind>,
    minimax: Option<PartialMiniMaxConfig>,
    openai: Option<PartialOpenAiConfig>,
    anthropic: Option<PartialAnthropicConfig>,
    openrouter: Option<PartialOpenRouterConfig>,
    zhipuai: Option<PartialZhipuAIConfig>,
    deepseek: Option<PartialDeepSeekConfig>,
    kimi: Option<PartialKimiConfig>,
    mimo: Option<PartialMimoConfig>,
    custom: Option<PartialCustomProviderConfig>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialMiniMaxConfig {
    api_key_env: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    temperature: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialOpenAiConfig {
    api_key_env: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    temperature: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialAnthropicConfig {
    api_key_env: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    temperature: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialOpenRouterConfig {
    api_key_env: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    temperature: Option<f32>,
    site_url: Option<String>,
    app_name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialZhipuAIConfig {
    api_key_env: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    temperature: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialDeepSeekConfig {
    api_key_env: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    temperature: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialKimiConfig {
    api_key_env: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    temperature: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialMimoConfig {
    api_key_env: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    temperature: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialCustomProviderConfig {
    api_key_env: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    temperature: Option<f32>,
    compatibility: Option<ProviderCompatibility>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialModelConfig {
    max_tokens: Option<u32>,
    enable_thinking: Option<bool>,
    thinking_budget: Option<u32>,
    aliases: Option<BTreeMap<String, String>>,
    recent_models: Option<Vec<String>>,
}

impl SandboxConfig {
    fn merge(&mut self, partial: PartialSandboxConfig) {
        if let Some(mode) = partial.mode {
            self.mode = mode;
        }
        if let Some(ro_paths) = partial.ro_paths {
            self.ro_paths = ro_paths;
        }
        if let Some(rw_paths) = partial.rw_paths {
            self.rw_paths = rw_paths;
        }
        if let Some(net) = partial.net {
            self.net = net;
        }
        if let Some(env_allow) = partial.env_allow {
            self.env_allow = env_allow;
        }
        if let Some(inherit_mounts) = partial.inherit_mounts {
            self.inherit_mounts = inherit_mounts;
        }
        if let Some(host_audio) = partial.host_audio {
            self.host_audio = host_audio;
        }
        if let Some(host_dbus_session) = partial.host_dbus_session {
            self.host_dbus_session = host_dbus_session;
        }
        if let Some(host_xdg_runtime) = partial.host_xdg_runtime {
            self.host_xdg_runtime = host_xdg_runtime;
        }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialPermissionConfig {
    mode: Option<PermissionMode>,
    allow: Option<Vec<String>>,
    deny: Option<Vec<String>>,
    ask: Option<Vec<String>>,
    sandbox: Option<PartialSandboxConfig>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialSandboxConfig {
    mode: Option<SandboxMode>,
    ro_paths: Option<Vec<PathBuf>>,
    rw_paths: Option<Vec<PathBuf>>,
    net: Option<bool>,
    env_allow: Option<Vec<String>>,
    inherit_mounts: Option<bool>,
    host_audio: Option<bool>,
    host_dbus_session: Option<bool>,
    host_xdg_runtime: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialSessionConfig {
    history_dir: Option<PathBuf>,
    #[serde(alias = "max_turn_per_run")]
    max_turns_per_run: Option<u32>,
    max_tool_calls_per_turn: Option<u32>,
    checkpoint_interval: Option<u32>,
    last_session_file: Option<PathBuf>,
    auto_compact_on_finish: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialMiddlewareConfig {
    retry_max_attempts: Option<u32>,
    retry_delay_cap_ms: Option<u64>,
    cost_budget_usd: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialFallbackConfig {
    enabled: Option<bool>,
    chain: Option<Vec<String>>,
    initial_retry_delay_ms: Option<u64>,
    retry_delay_ms: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialHarnessConfig {
    built_in_enabled: Option<bool>,
    global_instructions_path: Option<PathBuf>,
    project_instructions_path: Option<PathBuf>,
    local_instructions_path: Option<PathBuf>,
    skill_directories: Option<Vec<PathBuf>>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialMcpConfig {
    expose_in_safe_mode: Option<bool>,
    servers: Option<Vec<McpServerConfig>>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialMemoryConfig {
    file_path: Option<PathBuf>,
    max_notes: Option<usize>,
    auto_compact_on_finish: Option<bool>,
    context: Option<PartialContextConfig>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialContextConfig {
    context_window_target: Option<usize>,
    auto_detect_context_window: Option<bool>,
    query_provider_models_api: Option<bool>,
    max_retained_messages: Option<usize>,
    auto_summarize_threshold: Option<u8>,
    enable_auto_summarize: Option<bool>,
    smart_compaction_mode: Option<SmartCompactionMode>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialHookConfig {
    session_start: Option<Vec<HookCommand>>,
    session_end: Option<Vec<HookCommand>>,
    pre_tool_use: Option<Vec<HookCommand>>,
    post_tool_use: Option<Vec<HookCommand>>,
    post_tool_failure: Option<Vec<HookCommand>>,
    approval_requested: Option<Vec<HookCommand>>,
    subagent_start: Option<Vec<HookCommand>>,
    subagent_stop: Option<Vec<HookCommand>>,
    turn_complete: Option<Vec<HookCommand>>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialWebConfig {
    timeout_secs: Option<u64>,
    max_fetch_chars: Option<usize>,
    default_search_limit: Option<usize>,
    user_agent: Option<String>,
    search_endpoint: Option<String>,
}

fn default_true() -> bool {
    true
}

fn default_max_memory_notes() -> usize {
    128
}

fn default_model_aliases() -> BTreeMap<String, String> {
    BTreeMap::from([
        // DeepSeek (default provider). `deepseek-flash` is the V4.1 Flash API
        // id; the retired `deepseek-v4-flash` id only temporarily routes to it,
        // so the convenience aliases below point at the current model.
        ("default".into(), "deepseek-flash".into()),
        ("deepseek".into(), "deepseek-flash".into()),
        ("ds".into(), "deepseek-flash".into()),
        ("deepseek-v4".into(), "deepseek-flash".into()),
        ("dsv4".into(), "deepseek-flash".into()),
        ("deepseek-flash".into(), "deepseek-flash".into()),
        ("flash".into(), "deepseek-flash".into()),
        ("deepseek-v4.1-flash".into(), "deepseek-flash".into()),
        ("deepseek-v4-1-flash".into(), "deepseek-flash".into()),
        // `dsv4p` historically meant "DeepSeek V4 Pro". V4 Pro is being
        // retired — its requests route to V4.1 Flash until V4.1 Pro ships —
        // so point the alias at the model that actually serves it rather than
        // a going-away id.
        ("dsv4p".into(), "deepseek-flash".into()),
        ("deepseek-v3".into(), "deepseek-chat".into()),
        ("dsv3".into(), "deepseek-chat".into()),
        ("deepseek-r1".into(), "deepseek-reasoner".into()),
        ("dsr1".into(), "deepseek-reasoner".into()),
        // MiniMax
        ("minimax".into(), "MiniMax-M2.7".into()),
        ("m2.7".into(), "MiniMax-M2.7".into()),
        ("m3".into(), "MiniMax-M3".into()),
        // ZhipuAI (GLM)
        ("zhipuai".into(), "glm-5.3".into()),
        ("glm".into(), "glm-5.3".into()),
        ("glm5".into(), "glm-5.3".into()),
        ("glm-5.3".into(), "glm-5.3".into()),
        ("glm-5.3-flash".into(), "glm-5.3-flash".into()),
        ("glm-5.2".into(), "glm-5.2".into()),
        // Kimi (via Anthropic-compatible Kimi for Coding endpoint)
        ("kimi".into(), "k3".into()),
        ("moonshot".into(), "k3".into()),
        ("k3".into(), "k3".into()),
        ("kimi-k3".into(), "k3".into()),
        // Xiaomi MiMo
        ("mimo".into(), "mimo-v2.6-pro".into()),
        ("mimo-pro".into(), "mimo-v2.6-pro".into()),
        ("mimo-flash".into(), "mimo-v2.6-flash".into()),
        ("mimo-ultraspeed".into(), "mimo-v2.6-pro-ultraspeed".into()),
        // OpenAI
        ("openai".into(), "gpt-4o-mini".into()),
        ("gpt4o".into(), "gpt-4o".into()),
        ("gpt4omini".into(), "gpt-4o-mini".into()),
        // Anthropic
        ("claude".into(), "claude-3-7-sonnet-latest".into()),
        ("claude-sonnet".into(), "claude-3-7-sonnet-latest".into()),
        // OpenRouter
        ("openrouter".into(), "openai/gpt-4o-mini".into()),
    ])
}

fn resolve_api_key_value(inline: &Option<String>, env_name: &str) -> Option<String> {
    inline
        .as_deref()
        .filter(|v| !v.trim().is_empty())
        .map(String::from)
        .or_else(|| env::var(env_name).ok())
        .filter(|v| !v.trim().is_empty())
}

fn default_skill_directories() -> Vec<PathBuf> {
    vec![
        PathBuf::from(".nca/skills"),
        PathBuf::from(".claude/skills"),
    ]
}

/// Expand a leading `~` to `$HOME`. Returns the path unchanged if no `~` prefix
/// or if `$HOME` is not set.
fn expand_tilde(path: &Path) -> PathBuf {
    let s = path.to_string_lossy();
    if let Some(rest) = s.strip_prefix("~/")
        && let Ok(home) = env::var("HOME")
    {
        return PathBuf::from(format!("{home}/{rest}"));
    }
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_defaults_to_disabled_empty_chain() {
        let fallback = NcaConfig::default().fallback;
        assert!(!fallback.enabled, "fallback must be opt-in");
        assert!(fallback.chain.is_empty());
        assert_eq!(fallback.initial_retry_delay_ms, 0);
        assert_eq!(fallback.retry_delay_ms, 500);
    }

    #[test]
    fn fallback_parses_full_toml_section() {
        let raw = r#"
            [fallback]
            enabled = true
            chain = ["openai", "zhipuai"]
            initial_retry_delay_ms = 250
            retry_delay_ms = 1000
        "#;
        let partial: PartialNcaConfig = toml::from_str(raw).expect("parse");
        let mut config = NcaConfig::default();
        config.merge(partial);
        assert!(config.fallback.enabled);
        assert_eq!(config.fallback.chain, vec!["openai", "zhipuai"]);
        assert_eq!(config.fallback.initial_retry_delay_ms, 250);
        assert_eq!(config.fallback.retry_delay_ms, 1000);
    }

    #[test]
    fn fallback_partial_merge_is_per_field() {
        // A bare `enabled = true` flips the switch without touching the rest.
        let partial: PartialNcaConfig =
            toml::from_str("[fallback]\nenabled = true\n").expect("parse");
        let mut config = NcaConfig::default();
        config.fallback.chain = vec!["kimi".into()];
        config.merge(partial);
        assert!(config.fallback.enabled);
        assert_eq!(config.fallback.chain, vec!["kimi"], "chain untouched");
        assert_eq!(config.fallback.retry_delay_ms, 500, "delays untouched");
    }

    #[test]
    fn fallback_chain_merge_replaces_not_appends() {
        let partial: PartialNcaConfig =
            toml::from_str("[fallback]\nchain = [\"openai\"]\n").expect("parse");
        let mut config = NcaConfig::default();
        config.fallback.chain = vec!["kimi".into(), "openai".into()];
        config.merge(partial);
        assert_eq!(config.fallback.chain, vec!["openai"], "chain is wholesale");
    }

    #[test]
    fn fallback_roundtrips_through_serde() {
        let mut config = NcaConfig::default();
        config.fallback.enabled = true;
        config.fallback.chain = vec!["openai".into(), "kimi".into()];
        config.fallback.retry_delay_ms = 2_000;
        let json = serde_json::to_string(&config).expect("serialize");
        let back: NcaConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.fallback, config.fallback);
    }

    #[test]
    fn session_accepts_max_turn_per_run_typo_alias() {
        let raw = r#"
            [session]
            max_turn_per_run = 99
        "#;
        let partial: PartialNcaConfig = toml::from_str(raw).expect("parse");
        let session = partial.session.expect("session table");
        assert_eq!(session.max_turns_per_run, Some(99));
    }

    #[test]
    fn subagent_config_defaults_to_30s_result_timeout() {
        let config = NcaConfig::default();
        assert_eq!(config.subagent.result_timeout_ms, 30_000);

        // Field-level serde default must match the Default impl so a
        // partial `[subagent]` table (or a bare `subagent = {}`) resolves
        // to the same value.
        let raw = r#"
            [subagent]
        "#;
        let partial: PartialNcaConfig = toml::from_str(raw).expect("parse");
        let mut config = NcaConfig::default();
        config.merge(partial);
        assert_eq!(config.subagent.result_timeout_ms, 30_000);
    }

    #[test]
    fn subagent_config_partial_merge_overrides_result_timeout() {
        let raw = r#"
            [subagent]
            result_timeout_ms = 5000
        "#;
        let partial: PartialNcaConfig = toml::from_str(raw).expect("parse");
        let mut config = NcaConfig::default();
        config.merge(partial);
        assert_eq!(config.subagent.result_timeout_ms, 5000);

        // Round-trips through Serialize/Deserialize (config file writes).
        let json = serde_json::to_string(&config).expect("serialize");
        let back: NcaConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.subagent.result_timeout_ms, 5000);
    }

    #[test]
    fn subagent_config_p3_defaults_background_true_and_wake_enabled() {
        let config = NcaConfig::default();
        assert!(
            config.subagent.background,
            "background defaults to true (P3 default-on detach)"
        );
        assert!(
            config.subagent.wake.enabled,
            "wake.enabled defaults to true"
        );
        assert_eq!(
            config.subagent.wake.interval_ms, 1000,
            "wake interval defaults to 1s debounce"
        );
        assert_eq!(config.subagent.result_timeout_ms, 30_000);
    }

    #[test]
    fn subagent_config_partial_merge_overrides_nested_wake() {
        let raw = r#"
[subagent]
background = false

[subagent.wake]
enabled = false
interval_ms = 250
"#;
        let partial: PartialNcaConfig = toml::from_str(raw).expect("parse");
        let mut config = NcaConfig::default();
        config.merge(partial);
        assert!(!config.subagent.background);
        assert!(!config.subagent.wake.enabled);
        assert_eq!(config.subagent.wake.interval_ms, 250);
        assert_eq!(
            config.subagent.result_timeout_ms, 30_000,
            "untouched sibling keeps its default"
        );

        // Nested per-field merge: a partial `[subagent.wake]` that sets only
        // `enabled` leaves `interval_ms` at the default.
        let raw = r#"
[subagent.wake]
enabled = false
"#;
        let partial: PartialNcaConfig = toml::from_str(raw).expect("parse");
        let mut config = NcaConfig::default();
        config.merge(partial);
        assert!(!config.subagent.wake.enabled);
        assert_eq!(config.subagent.wake.interval_ms, 1000);
    }

    #[test]
    fn subagent_wake_config_toml_roundtrip() {
        let raw = r#"
[subagent]
background = true
result_timeout_ms = 45000

[subagent.wake]
enabled = false
interval_ms = 500
"#;
        let partial: PartialNcaConfig = toml::from_str(raw).expect("parse");
        let mut config = NcaConfig::default();
        config.merge(partial);
        assert!(config.subagent.background);
        assert_eq!(config.subagent.result_timeout_ms, 45000);
        assert!(!config.subagent.wake.enabled);
        assert_eq!(config.subagent.wake.interval_ms, 500);

        // Round-trips through Serialize/Deserialize (config file writes) —
        // nested wake must survive a full NcaConfig serialization.
        let json = serde_json::to_string(&config).expect("serialize");
        let back: NcaConfig = serde_json::from_str(&json).expect("deserialize");
        assert!(back.subagent.background);
        assert_eq!(back.subagent.result_timeout_ms, 45000);
        assert!(!back.subagent.wake.enabled);
        assert_eq!(back.subagent.wake.interval_ms, 500);
    }

    #[test]
    fn apply_model_override_updates_selected_provider_model() {
        let mut config = NcaConfig::default();
        config.provider.default = ProviderKind::OpenAi;
        config.sync_default_model_from_provider();

        config.apply_model_override("gpt4o");

        assert_eq!(config.provider.openai.model, "gpt-4o");
        assert_eq!(config.model.default_model, "gpt-4o");
        assert_eq!(config.provider.minimax.model, "MiniMax-M2.7");
    }

    #[test]
    fn apply_model_override_switches_provider_for_cross_provider_alias() {
        // Regression: selecting "glm" while DeepSeek is active must switch to
        // ZhipuAI instead of setting DeepSeek's model to glm-5.3.
        let mut config = NcaConfig::default();
        config.provider.default = ProviderKind::DeepSeek;
        config.sync_default_model_from_provider();
        assert_eq!(config.provider.deepseek.model, "deepseek-flash");

        config.apply_model_override("glm");

        assert_eq!(config.provider.default, ProviderKind::ZhipuAI);
        assert_eq!(config.provider.zhipuai.model, "glm-5.3");
        assert_eq!(config.model.default_model, "glm-5.3");
        // DeepSeek model must NOT have been polluted
        assert_eq!(config.provider.deepseek.model, "deepseek-flash");
    }

    #[test]
    fn glm_5_3_flash_switches_provider_and_resolves_verbatim() {
        // "glm-5.3-flash" is a full model id, not an alias to rewrite: it must
        // switch the provider to ZhipuAI and pass through unchanged.
        let mut config = NcaConfig::default();
        config.provider.default = ProviderKind::DeepSeek;
        config.sync_default_model_from_provider();

        config.apply_model_override("glm-5.3-flash");

        assert_eq!(config.provider.default, ProviderKind::ZhipuAI);
        assert_eq!(config.provider.zhipuai.model, "glm-5.3-flash");
        assert_eq!(config.model.default_model, "glm-5.3-flash");
    }

    #[test]
    fn apply_model_override_switches_provider_for_gpt4o_alias() {
        let mut config = NcaConfig::default();
        config.provider.default = ProviderKind::DeepSeek;
        config.sync_default_model_from_provider();

        config.apply_model_override("gpt4o");

        assert_eq!(config.provider.default, ProviderKind::OpenAi);
        assert_eq!(config.provider.openai.model, "gpt-4o");
        assert_eq!(config.model.default_model, "gpt-4o");
        assert_eq!(config.provider.deepseek.model, "deepseek-flash");
    }

    #[test]
    fn apply_model_override_keeps_provider_for_unknown_alias() {
        let mut config = NcaConfig::default();
        config.provider.default = ProviderKind::OpenAi;
        config.sync_default_model_from_provider();

        // A literal model name not in the alias map — no provider switch.
        config.apply_model_override("my-custom-model");

        assert_eq!(config.provider.default, ProviderKind::OpenAi);
        assert_eq!(config.provider.openai.model, "my-custom-model");
    }

    #[test]
    fn provider_hint_for_alias_covers_all_built_in_aliases() {
        // Every alias in the default map that resolves to a provider-specific model
        // must have a provider hint.
        let aliases = default_model_aliases();
        for alias in aliases.keys() {
            // Skipping generic aliases that are not provider-specific (e.g. "default", "coding", "reasoning")
            // is fine — they map to DeepSeek which is the default provider.
            // The important thing is that cross-provider aliases DO have hints.
            let _ = NcaConfig::provider_hint_for_alias(alias);
        }
        // Spot-check key cross-provider aliases
        assert_eq!(
            NcaConfig::provider_hint_for_alias("glm"),
            Some(ProviderKind::ZhipuAI)
        );
        assert_eq!(
            NcaConfig::provider_hint_for_alias("deepseek"),
            Some(ProviderKind::DeepSeek)
        );
        assert_eq!(
            NcaConfig::provider_hint_for_alias("claude"),
            Some(ProviderKind::Anthropic)
        );
        assert_eq!(
            NcaConfig::provider_hint_for_alias("gpt4o"),
            Some(ProviderKind::OpenAi)
        );
        assert_eq!(
            NcaConfig::provider_hint_for_alias("openrouter"),
            Some(ProviderKind::OpenRouter)
        );
        assert_eq!(NcaConfig::provider_hint_for_alias("unknown-model"), None);
    }

    #[test]
    fn deepseek_v41_flash_aliases_resolve_to_canonical_id() {
        // `deepseek-flash` is the V4.1 Flash API id. The convenience aliases and
        // the dotted/dashed spellings third-party catalogs publish must all
        // resolve to it and switch the provider to DeepSeek.
        let config = NcaConfig::default();
        for alias in [
            "default",
            "deepseek",
            "ds",
            "deepseek-v4",
            "dsv4",
            "deepseek-flash",
            "flash",
            "deepseek-v4.1-flash",
            "deepseek-v4-1-flash",
            // `dsv4p` no longer points at the retiring `deepseek-v4-pro` id;
            // V4 Pro requests are served by V4.1 Flash today.
            "dsv4p",
        ] {
            assert_eq!(
                config.model.resolve_alias(alias),
                "deepseek-flash",
                "resolved model for alias {alias:?}"
            );
            assert_eq!(
                NcaConfig::provider_hint_for_alias(alias),
                Some(ProviderKind::DeepSeek),
                "provider hint for alias {alias:?}"
            );
        }
    }

    #[test]
    fn apply_env_supports_openai_anthropic_and_openrouter() {
        let _guard = EnvGuard::set(&[
            ("NCA_DEFAULT_PROVIDER", Some("openrouter")),
            ("OPENAI_API_KEY", Some("openai-key")),
            ("OPENAI_MODEL", Some("gpt-4o")),
            ("ANTHROPIC_API_KEY", Some("anthropic-key")),
            ("ANTHROPIC_MODEL", Some("claude-3-7-sonnet-20250219")),
            ("OPENROUTER_API_KEY", Some("openrouter-key")),
            ("OPENROUTER_MODEL", Some("anthropic/claude-3.7-sonnet")),
            ("OPENROUTER_SITE_URL", Some("https://nca.test")),
            ("OPENROUTER_APP_NAME", Some("Native CLI AI")),
        ]);

        let mut config = NcaConfig::default();
        config.apply_env();

        assert_eq!(config.provider.default, ProviderKind::OpenRouter);
        assert_eq!(
            config.provider.openai.resolve_api_key().as_deref(),
            Some("openai-key")
        );
        assert_eq!(
            config.provider.anthropic.resolve_api_key().as_deref(),
            Some("anthropic-key")
        );
        assert_eq!(
            config.provider.openrouter.resolve_api_key().as_deref(),
            Some("openrouter-key")
        );
        assert_eq!(config.provider.openai.model, "gpt-4o");
        assert_eq!(
            config.provider.anthropic.model,
            "claude-3-7-sonnet-20250219"
        );
        assert_eq!(
            config.provider.openrouter.model,
            "anthropic/claude-3.7-sonnet"
        );
        assert_eq!(
            config.provider.openrouter.site_url.as_deref(),
            Some("https://nca.test")
        );
        assert_eq!(
            config.provider.openrouter.app_name.as_deref(),
            Some("Native CLI AI")
        );
        assert_eq!(config.model.default_model, "anthropic/claude-3.7-sonnet");
    }

    /// Process-global mutex that serializes all env-mutating tests.
    ///
    /// `env::set_var` / `env::remove_var` mutate process-global state and are
    /// unsafe under Rust's default parallel test threads.  Locking this mutex
    /// for the lifetime of each `EnvGuard` prevents two tests from racing on
    /// the same environment variable.
    static ENV_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct EnvGuard {
        previous: Vec<(String, Option<String>)>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl EnvGuard {
        fn set(vars: &[(&str, Option<&str>)]) -> Self {
            // Block until we exclusively own the process environment.
            let lock: std::sync::MutexGuard<'static, ()> =
                ENV_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

            let mut previous = Vec::new();
            for (key, value) in vars {
                previous.push((key.to_string(), env::var(key).ok()));
                match value {
                    Some(value) => unsafe { env::set_var(key, value) },
                    None => unsafe { env::remove_var(key) },
                }
            }
            Self {
                previous,
                _lock: lock,
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (key, value) in self.previous.drain(..) {
                match value {
                    Some(value) => unsafe { env::set_var(&key, value) },
                    None => unsafe { env::remove_var(&key) },
                }
            }
        }
    }

    #[test]
    fn workspace_cache_id_stable_for_same_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (id1, p1) = workspace_cache_id(dir.path()).expect("id");
        let (id2, p2) = workspace_cache_id(dir.path()).expect("id");
        assert_eq!(id1, id2);
        assert_eq!(p1, p2);
        assert!(id1.contains('-'));
        assert!(id1.len() > 16);
    }

    #[test]
    fn ui_editor_roundtrips_through_workspace_file() {
        let _guard = EnvGuard::set(&[("NCA_EDITOR", None), ("EDITOR", None)]);
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = NcaConfig::default();
        config.ui.editor = Some("vim".into());
        config.set_default_provider(ProviderKind::MiniMax);
        config.save_workspace_file(dir.path()).expect("save");

        let loaded = NcaConfig::load_for_workspace(dir.path()).expect("load");
        assert_eq!(loaded.ui.editor.as_deref(), Some("vim"));
        assert_eq!(loaded.effective_editor_command(), "vim");
    }

    #[test]
    fn provider_kind_from_cli_name() {
        assert_eq!(
            ProviderKind::from_cli_name("MINIMAX"),
            Some(ProviderKind::MiniMax)
        );
        assert_eq!(
            ProviderKind::from_cli_name("openai"),
            Some(ProviderKind::OpenAi)
        );
        assert_eq!(ProviderKind::from_cli_name("nope"), None);
    }

    #[test]
    fn onboarding_completed_defaults_to_false() {
        let config = NcaConfig::default();
        assert!(!config.ui.onboarding_completed);
    }

    #[test]
    fn onboarding_completed_merges_from_partial() {
        let mut config = NcaConfig::default();
        let toml_str = r#"
[ui]
onboarding_completed = true
"#;
        let partial: PartialNcaConfig = toml::from_str(toml_str).unwrap();
        config.merge(partial);
        assert!(config.ui.onboarding_completed);
    }

    #[test]
    fn any_api_key_present_returns_false_when_no_keys() {
        let config = config_without_env_keys();
        assert!(!config.provider.any_api_key_present());
    }

    #[test]
    fn any_api_key_present_returns_true_when_one_key_set() {
        let mut config = NcaConfig::default();
        config.provider.openai.api_key = Some("sk-test".into());
        assert!(config.provider.any_api_key_present());
    }

    /// Returns an NcaConfig with env var fallbacks disabled so tests don't
    /// pick up real API keys from the shell environment.
    fn config_without_env_keys() -> NcaConfig {
        let mut config = NcaConfig::default();
        config.provider.minimax.api_key_env = "__NCA_TEST_NONE__".into();
        config.provider.openai.api_key_env = "__NCA_TEST_NONE__".into();
        config.provider.anthropic.api_key_env = "__NCA_TEST_NONE__".into();
        config.provider.openrouter.api_key_env = "__NCA_TEST_NONE__".into();
        config.provider.zhipuai.api_key_env = "__NCA_TEST_NONE__".into();
        config.provider.deepseek.api_key_env = "__NCA_TEST_NONE__".into();
        config.provider.kimi.api_key_env = "__NCA_TEST_NONE__".into();
        config
    }

    #[test]
    fn needs_onboarding_true_when_no_flag_and_no_keys() {
        let config = config_without_env_keys();
        assert!(config.needs_onboarding());
    }

    #[test]
    fn needs_onboarding_false_when_flag_set_and_key_present() {
        let mut config = NcaConfig::default();
        config.ui.onboarding_completed = true;
        config.provider.minimax.api_key = Some("test-key".into());
        assert!(!config.needs_onboarding());
    }

    #[test]
    fn needs_onboarding_true_when_flag_set_but_all_keys_removed() {
        let mut config = config_without_env_keys();
        config.ui.onboarding_completed = true;
        // no keys set — safety net triggers
        assert!(config.needs_onboarding());
    }

    #[test]
    fn needs_onboarding_true_when_key_present_but_flag_not_set() {
        let mut config = NcaConfig::default();
        config.provider.openai.api_key = Some("sk-test".into());
        // onboarding_completed is false
        assert!(config.needs_onboarding());
    }

    #[test]
    fn onboarding_roundtrip_through_toml() {
        let toml_str = r#"
[ui]
onboarding_completed = true

[provider.minimax]
api_key = "test-key"
"#;
        let partial: PartialNcaConfig = toml::from_str(toml_str).unwrap();
        let mut config = NcaConfig::default();
        config.merge(partial);
        assert!(!config.needs_onboarding());
    }

    #[test]
    fn onboarding_triggers_when_key_removed_after_completion() {
        let toml_str = r#"
[ui]
onboarding_completed = true
"#;
        let partial: PartialNcaConfig = toml::from_str(toml_str).unwrap();
        let mut config = config_without_env_keys();
        config.merge(partial);
        assert!(config.needs_onboarding());
    }

    #[test]
    fn hooks_with_empty_arrays_and_later_array_of_tables_parses_correctly() {
        // This mirrors the real config pattern: [hooks] sets some arrays to [],
        // then [[hooks.session_end]] etc. add entries to other arrays.
        let toml_str = r#"
[hooks]
session_start = []
pre_tool_use = []
post_tool_use = []
subagent_start = []
subagent_stop = []

[[hooks.session_end]]
command = "echo session ended"
blocking = false

[[hooks.post_tool_failure]]
command = "echo tool failed"
blocking = false

[[hooks.approval_requested]]
command = "echo approval needed"
blocking = false
"#;
        let partial: PartialNcaConfig = toml::from_str(toml_str).expect("parse hooks config");
        let hooks = partial.hooks.expect("hooks should be present");

        // Empty arrays should be Some([])
        assert_eq!(
            hooks.session_start.as_deref().map(<[HookCommand]>::len),
            Some(0)
        );
        assert_eq!(
            hooks.pre_tool_use.as_deref().map(<[HookCommand]>::len),
            Some(0)
        );
        assert_eq!(
            hooks.post_tool_use.as_deref().map(<[HookCommand]>::len),
            Some(0)
        );
        assert_eq!(
            hooks.subagent_start.as_deref().map(<[HookCommand]>::len),
            Some(0)
        );
        assert_eq!(
            hooks.subagent_stop.as_deref().map(<[HookCommand]>::len),
            Some(0)
        );

        // Arrays with [[...]] entries should have 1 element each
        assert_eq!(
            hooks.session_end.as_deref().map(<[HookCommand]>::len),
            Some(1)
        );
        assert_eq!(
            hooks.post_tool_failure.as_deref().map(<[HookCommand]>::len),
            Some(1)
        );
        assert_eq!(
            hooks
                .approval_requested
                .as_deref()
                .map(<[HookCommand]>::len),
            Some(1)
        );

        // Verify the commands are correct
        assert_eq!(
            hooks.session_end.as_ref().unwrap()[0].command,
            "echo session ended"
        );
        assert_eq!(
            hooks.post_tool_failure.as_ref().unwrap()[0].command,
            "echo tool failed"
        );
        assert_eq!(
            hooks.approval_requested.as_ref().unwrap()[0].command,
            "echo approval needed"
        );
    }

    #[test]
    fn hooks_merge_preserves_array_of_tables_entries() {
        let toml_str = r#"
[hooks]
session_start = []

[[hooks.session_end]]
command = "echo end"
blocking = false

[[hooks.post_tool_failure]]
command = "echo fail"
blocking = false
"#;
        let partial: PartialNcaConfig = toml::from_str(toml_str).expect("parse");
        let mut config = NcaConfig::default();
        config.merge(partial);

        assert!(config.hooks.session_start.is_empty());
        assert_eq!(config.hooks.session_end.len(), 1);
        assert_eq!(config.hooks.session_end[0].command, "echo end");
        assert_eq!(config.hooks.post_tool_failure.len(), 1);
        assert_eq!(config.hooks.post_tool_failure[0].command, "echo fail");
    }

    #[test]
    fn hooks_has_any_detects_entries() {
        let toml_str = r#"
[[hooks.session_end]]
command = "echo end"
blocking = false

[[hooks.post_tool_failure]]
command = "echo fail"
blocking = false
"#;
        let partial: PartialNcaConfig = toml::from_str(toml_str).expect("parse");
        let mut config = NcaConfig::default();
        config.merge(partial);

        // HookRunner has_any checks all fields
        assert!(
            !config.hooks.session_start.is_empty()
                || !config.hooks.session_end.is_empty()
                || !config.hooks.pre_tool_use.is_empty()
                || !config.hooks.post_tool_use.is_empty()
                || !config.hooks.post_tool_failure.is_empty()
                || !config.hooks.approval_requested.is_empty()
                || !config.hooks.subagent_start.is_empty()
                || !config.hooks.subagent_stop.is_empty()
                || !config.hooks.turn_complete.is_empty()
        );
    }

    #[test]
    fn save_workspace_file_only_saves_overrides_not_full_config() {
        // Simulate: user switches provider.  The local file should only contain
        // the provider change, not all defaults.
        let tmp_home = tempfile::tempdir().expect("tempdir");
        let _guard = EnvGuard::set(&[
            ("HOME", Some(tmp_home.path().to_str().unwrap())),
            ("MINIMAX_API_KEY", None),
            ("OPENAI_API_KEY", None),
            ("NCA_EDITOR", None),
            ("EDITOR", None),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");

        // Base = defaults (no global file in test — HOME points to empty tmpdir).
        let mut config = NcaConfig::default();
        config.set_default_provider(ProviderKind::OpenAi);
        config.save_workspace_file(dir.path()).expect("save");

        let local_path = workspace_config_path(dir.path());
        assert!(local_path.exists(), "local config should be written");

        let raw = std::fs::read_to_string(&local_path).expect("read local config");
        // The diff should be minimal — just the provider.default change.
        // default_model is now derived (#[serde(skip)]), so it does NOT appear in persisted config.
        assert!(
            raw.contains("openai"),
            "local config should reference openai: {raw}"
        );
        // It should NOT contain all the provider sections (minimax, anthropic, etc.)
        // since those haven't changed from defaults.
        assert!(
            !raw.contains("anthropic"),
            "local config should NOT contain unchanged provider sections: {raw}"
        );
        assert!(
            !raw.contains("deepseek"),
            "local config should NOT contain unchanged provider sections: {raw}"
        );
        assert!(
            !raw.contains("zhipuai"),
            "local config should NOT contain unchanged provider sections: {raw}"
        );
        // Should NOT contain full config sections like [session], [harness], etc.
        assert!(
            !raw.contains("history_dir"),
            "local config should NOT contain session defaults: {raw}"
        );
        assert!(
            !raw.contains("built_in_enabled"),
            "local config should NOT contain harness defaults: {raw}"
        );
    }

    #[test]
    fn save_workspace_file_removes_local_config_when_no_overrides() {
        let tmp_home = tempfile::tempdir().expect("tempdir");
        let _guard = EnvGuard::set(&[
            ("HOME", Some(tmp_home.path().to_str().unwrap())),
            ("MINIMAX_API_KEY", None),
            ("OPENAI_API_KEY", None),
            ("NCA_EDITOR", None),
            ("EDITOR", None),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");

        // Write a local config first (create .nca/ dir)
        let local_path = workspace_config_path(dir.path());
        std::fs::create_dir_all(local_path.parent().unwrap()).expect("create .nca dir");
        std::fs::write(&local_path, "[ui]\neditor = \"vim\"").expect("write");
        assert!(local_path.exists());

        // Save with default config (no overrides) — should remove the local file.
        let config = NcaConfig::default();
        config.save_workspace_file(dir.path()).expect("save");

        assert!(
            !local_path.exists(),
            "local config should be removed when there are no overrides"
        );
    }

    #[test]
    fn diff_toml_values_identical_returns_none() {
        let val = toml::Value::String("hello".into());
        assert!(diff_toml_values(&val, &val).is_none());
    }

    /// Regression: a stale `model.default_model` in the config file must NOT
    /// overwrite the active provider's model field.  default_model is now derived
    /// (not persisted), so any old value is silently ignored during merge.
    #[test]
    fn stale_default_model_does_not_pollute_active_provider() {
        let toml_str = r#"
[provider]
default = "deepseek"

[model]
default_model = "glm-5.2"
"#;
        let partial: PartialNcaConfig = toml::from_str(toml_str).expect("parse");
        let mut config = NcaConfig::default();
        config.merge(partial);

        // Provider must be deepseek (explicit in config)
        assert_eq!(config.provider.default, ProviderKind::DeepSeek);
        // DeepSeek model must remain the deepseek default — NOT polluted by glm-5.2
        assert_eq!(config.provider.deepseek.model, "deepseek-flash");
        // ZhipuAI model must remain the zhipuai default
        assert_eq!(config.provider.zhipuai.model, "glm-5.3");
        // In-memory default_model is derived from the active provider
        assert_eq!(config.model.default_model, "deepseek-flash");
    }

    #[test]
    fn diff_toml_values_different_scalars() {
        let a = toml::Value::String("a".into());
        let b = toml::Value::String("b".into());
        let diff = diff_toml_values(&a, &b).expect("diff");
        assert_eq!(diff, toml::Value::String("a".into()));
    }

    #[test]
    fn diff_toml_values_nested_tables() {
        let mut curr = toml::map::Map::new();
        curr.insert("x".into(), toml::Value::Integer(1));
        curr.insert("y".into(), toml::Value::String("same".into()));

        let mut base = toml::map::Map::new();
        base.insert("x".into(), toml::Value::Integer(2));
        base.insert("y".into(), toml::Value::String("same".into()));

        let curr_table = toml::Value::Table(curr);
        let base_table = toml::Value::Table(base);

        let diff = diff_toml_values(&curr_table, &base_table).expect("diff");
        if let toml::Value::Table(map) = diff {
            // Only "x" should differ; "y" is the same and should be omitted.
            assert_eq!(map.len(), 1);
            assert_eq!(map.get("x"), Some(&toml::Value::Integer(1)));
        } else {
            panic!("expected table");
        }
    }

    #[test]
    fn global_instructions_path_roundtrips_through_toml() {
        let toml_str = r#"
[harness]
global_instructions_path = "~/.nca/AGENTS.md"
"#;
        let partial: PartialNcaConfig = toml::from_str(toml_str).unwrap();
        let mut config = NcaConfig::default();
        config.merge(partial);
        assert_eq!(
            config.harness.global_instructions_path.as_deref(),
            Some(std::path::Path::new("~/.nca/AGENTS.md")),
        );
    }

    #[test]
    fn global_instructions_path_absent_by_default() {
        let config = NcaConfig::default();
        assert!(config.harness.global_instructions_path.is_none());
        assert!(config.harness.resolve_global_instructions_path().is_none());
    }

    #[test]
    fn expand_tilde_expands_home() {
        let _guard = EnvGuard::set(&[("HOME", Some("/test/home"))]);
        let result = expand_tilde(std::path::Path::new("~/.nca/AGENTS.md"));
        assert_eq!(
            result,
            std::path::PathBuf::from("/test/home/.nca/AGENTS.md")
        );
    }

    #[test]
    fn expand_tilde_leaves_non_tilde_unchanged() {
        let _guard = EnvGuard::set(&[("HOME", Some("/test/home"))]);
        let result = expand_tilde(std::path::Path::new("/absolute/path"));
        assert_eq!(result, std::path::PathBuf::from("/absolute/path"));
        let result2 = expand_tilde(std::path::Path::new("relative/path"));
        assert_eq!(result2, std::path::PathBuf::from("relative/path"));
    }

    #[test]
    fn skill_directories_expand_tilde_on_merge() {
        let _guard = EnvGuard::set(&[("HOME", Some("/test/home"))]);
        let toml_str = r#"
[harness]
skill_directories = ["~/.config/nca/skills", "/abs/path", "rel/path"]
"#;
        let partial: PartialNcaConfig = toml::from_str(toml_str).expect("parse");
        let mut config = NcaConfig::default();
        config.merge(partial);

        let dirs = &config.harness.skill_directories;
        assert_eq!(dirs.len(), 3);
        // ~ expanded to absolute
        assert_eq!(
            dirs[0],
            std::path::PathBuf::from("/test/home/.config/nca/skills")
        );
        // absolute passthrough
        assert_eq!(dirs[1], std::path::PathBuf::from("/abs/path"));
        // relative passthrough (resolved against workspace_root at scan time)
        assert_eq!(dirs[2], std::path::PathBuf::from("rel/path"));
    }

    #[test]
    fn agent_profile_config_parses_from_toml() {
        let toml_str = r#"
[agents.code-reviewer]
provider = "openai"
model = "gpt-4o"
permission_mode = "plan"
system_prompt_append = "Focus on security."
allowed_tools = ["read", "search"]

[agents.reviewer-lite]
model = "claude-sonnet"
"#;
        let partial: PartialNcaConfig = toml::from_str(toml_str).expect("parse");
        assert!(partial.agents.is_some());

        let mut config = NcaConfig::default();
        config.merge(partial);

        let reviewer = config.agent_profile("code-reviewer").unwrap();
        assert_eq!(reviewer.provider, Some(ProviderKind::OpenAi));
        assert_eq!(reviewer.model.as_deref(), Some("gpt-4o"));
        assert_eq!(reviewer.permission_mode, Some(PermissionMode::Plan));
        assert_eq!(
            reviewer.system_prompt_append.as_deref(),
            Some("Focus on security.")
        );
        assert_eq!(reviewer.allowed_tools.as_deref().map(|v| v.len()), Some(2));

        let lite = config.agent_profile("reviewer-lite").unwrap();
        assert_eq!(lite.provider, None);
        assert_eq!(lite.model.as_deref(), Some("claude-sonnet"));
        // Alias hint from model name should resolve provider
        assert_eq!(lite.resolve_provider(), Some(ProviderKind::Anthropic));
    }

    #[test]
    fn agent_profile_resolve_provider_prefers_explicit_over_alias() {
        let profile = AgentProfileConfig {
            provider: Some(ProviderKind::DeepSeek),
            model: Some("gpt-4o".into()),
            ..Default::default()
        };
        // Explicit provider wins over alias hint from model
        assert_eq!(profile.resolve_provider(), Some(ProviderKind::DeepSeek));
    }

    #[test]
    fn agent_profile_resolve_model_falls_back_to_config_default() {
        let profile = AgentProfileConfig {
            provider: Some(ProviderKind::OpenAi),
            model: None,
            ..Default::default()
        };
        assert_eq!(
            profile.resolve_model("deepseek-v4-flash"),
            "deepseek-v4-flash"
        );
    }

    #[test]
    fn agent_profile_names_lists_all() {
        let mut config = NcaConfig::default();
        config.agents.insert(
            "alpha".into(),
            AgentProfileConfig {
                provider: Some(ProviderKind::MiniMax),
                ..Default::default()
            },
        );
        config
            .agents
            .insert("beta".into(), AgentProfileConfig::default());
        let mut names = config.agent_profile_names();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn agent_profile_skills_directives_parse_from_toml() {
        // New optional fields must parse, stay absent by default, and
        // merge per-key across config layers (same as existing fields).
        let partial: PartialNcaConfig = toml::from_str(
            r#"
[agents.gatekeeper]
skills = ["alpha", "beta"]
skills_add = ["gamma"]
skills_remove = ["beta"]
"#,
        )
        .unwrap();
        let mut config = NcaConfig::default();
        config.merge(partial);

        let profile = config.agent_profile("gatekeeper").unwrap();
        assert_eq!(profile.skills, Some(vec!["alpha".into(), "beta".into()]));
        assert_eq!(profile.skills_add, Some(vec!["gamma".into()]));
        assert_eq!(profile.skills_remove, Some(vec!["beta".into()]));

        // Absent keys → None (no gating) — existing configs keep working.
        let bare: PartialNcaConfig =
            toml::from_str("[agents.plain]\nprovider = \"openai\"\n").unwrap();
        let mut config2 = NcaConfig::default();
        config2.merge(bare);
        let plain = config2.agent_profile("plain").unwrap();
        assert_eq!(plain.skills, None);
        assert_eq!(plain.skills_add, None);
        assert_eq!(plain.skills_remove, None);
    }

    #[test]
    fn agent_profile_merge_accumulates() {
        let toml1 = r#"
[agents.reviewer]
provider = "openai"
"#;
        let toml2 = r#"
[agents.reviewer]
model = "gpt-4o"
permission_mode = "plan"

[agents.tester]
provider = "deepseek"
"#;
        let partial1: PartialNcaConfig = toml::from_str(toml1).unwrap();
        let partial2: PartialNcaConfig = toml::from_str(toml2).unwrap();

        let mut config = NcaConfig::default();
        config.merge(partial1);
        config.merge(partial2);

        // reviewer should have both provider from partial1 and model from partial2
        let reviewer = config.agent_profile("reviewer").unwrap();
        assert_eq!(reviewer.provider, Some(ProviderKind::OpenAi));
        assert_eq!(reviewer.model.as_deref(), Some("gpt-4o"));
        assert_eq!(reviewer.permission_mode, Some(PermissionMode::Plan));

        // tester only from partial2
        let tester = config.agent_profile("tester").unwrap();
        assert_eq!(tester.provider, Some(ProviderKind::DeepSeek));
        assert!(tester.model.is_none());
    }

    #[test]
    fn extra_paths_parse_and_merge() {
        let toml_str = r#"
extra_paths = ["/home/user/projects", "/opt/data"]
"#;
        let partial: PartialNcaConfig = toml::from_str(toml_str).expect("parse");
        let mut config = NcaConfig::default();
        assert!(config.extra_paths.is_empty(), "default has no extra paths");
        config.merge(partial);

        assert_eq!(
            config.extra_paths,
            vec![
                PathBuf::from("/home/user/projects"),
                PathBuf::from("/opt/data"),
            ]
        );
    }

    // ---- P5 sandbox config (TDD red phase; types do not exist yet) ----
    //
    // Implementation contract (see docs/plans/deepseek-harness-adoption.md §P5):
    //   - `PermissionConfig` gains `pub sandbox: SandboxConfig`.
    //   - `SandboxConfig { mode: SandboxMode, ro_paths: Vec<PathBuf>,
    //     rw_paths: Vec<PathBuf>, net: bool }` in this module; manual `Default`
    //     with `mode: Auto`, `net: true`, empty path lists.
    //   - `SandboxMode` enum { Auto, Required, Off } with
    //     `#[serde(rename_all = "kebab-case")]`; re-exported by
    //     `nca_runtime::sandbox` (common stays the single source of truth).
    //   - `PartialPermissionConfig` gains `sandbox: Option<PartialSandboxConfig>`
    //     and `PermissionConfig::merge` forwards it.

    #[test]
    fn sandbox_defaults_are_auto_net_with_empty_paths() {
        let config = NcaConfig::default();
        assert_eq!(
            config.permissions.sandbox.mode,
            SandboxMode::Auto,
            "mode defaults to auto (escape hatch for kernels without Landlock)"
        );
        assert!(
            config.permissions.sandbox.net,
            "net defaults to true (blocking network is a separate future feature)"
        );
        assert!(
            config.permissions.sandbox.ro_paths.is_empty(),
            "ro_paths default empty (built-in system roots are added by runtime)"
        );
        assert!(config.permissions.sandbox.rw_paths.is_empty());
    }

    #[test]
    fn sandbox_modes_parse_from_toml() {
        for (raw, expected) in [
            ("auto", SandboxMode::Auto),
            ("required", SandboxMode::Required),
            ("off", SandboxMode::Off),
        ] {
            let toml_str = format!("[permissions.sandbox]\nmode = \"{raw}\"\n");
            let partial: PartialNcaConfig = toml::from_str(&toml_str).expect("parse");
            let mut config = NcaConfig::default();
            config.merge(partial);
            assert_eq!(
                config.permissions.sandbox.mode, expected,
                "mode = {raw} should parse"
            );
        }
    }

    #[test]
    fn sandbox_inherit_mounts_defaults_true_and_parses_false() {
        // Absent key → default true (mounts propagate to the sandbox).
        let partial: PartialNcaConfig =
            toml::from_str("[permissions.sandbox]\nmode = \"off\"\n").expect("parse");
        let mut config = NcaConfig::default();
        config.merge(partial);
        assert!(config.permissions.sandbox.inherit_mounts);

        // Explicit false → honored.
        let partial: PartialNcaConfig =
            toml::from_str("[permissions.sandbox]\ninherit_mounts = false\n").expect("parse");
        let mut config = NcaConfig::default();
        config.merge(partial);
        assert!(!config.permissions.sandbox.inherit_mounts);
    }

    #[test]
    fn sandbox_paths_and_net_parse_as_arrays() {
        let toml_str = r#"
[permissions.sandbox]
mode = "required"
net = false
ro_paths = ["/usr", "/opt/tools"]
rw_paths = ["/home/user/project", "/tmp"]
"#;
        let partial: PartialNcaConfig = toml::from_str(toml_str).expect("parse");
        let mut config = NcaConfig::default();
        config.merge(partial);

        let sandbox = &config.permissions.sandbox;
        assert_eq!(sandbox.mode, SandboxMode::Required);
        assert!(!sandbox.net);
        assert_eq!(
            sandbox.ro_paths,
            vec![PathBuf::from("/usr"), PathBuf::from("/opt/tools")]
        );
        assert_eq!(
            sandbox.rw_paths,
            vec![PathBuf::from("/home/user/project"), PathBuf::from("/tmp")]
        );
    }

    #[test]
    fn sandbox_env_allow_defaults_to_curated_list() {
        let config = NcaConfig::default();
        let allow = &config.permissions.sandbox.env_allow;
        for name in ["PATH", "HOME", "CARGO_HOME"] {
            assert!(
                allow.iter().any(|a| a == name),
                "default env_allow should contain {name}: {allow:?}"
            );
        }
        // The function the serde default points at must agree with the struct default.
        assert_eq!(
            *allow,
            default_sandbox_env_allow(),
            "SandboxConfig::default must use default_sandbox_env_allow()"
        );
    }

    #[test]
    fn sandbox_env_allow_parses_and_replaces_default() {
        let toml_str = r#"
[permissions.sandbox]
env_allow = ["FOO"]
"#;
        let partial: PartialNcaConfig = toml::from_str(toml_str).expect("parse");
        let mut config = NcaConfig::default();
        config.merge(partial);

        let allow = &config.permissions.sandbox.env_allow;
        assert_eq!(
            *allow,
            vec!["FOO".to_string()],
            "explicit list REPLACES the curated default"
        );
        assert!(
            !allow.iter().any(|a| a == "HOME"),
            "default entries must not linger after an explicit env_allow"
        );
    }

    #[test]
    fn sandbox_env_allow_empty_list_means_path_only() {
        let toml_str = r#"
[permissions.sandbox]
env_allow = []
"#;
        let partial: PartialNcaConfig = toml::from_str(toml_str).expect("parse");
        let mut config = NcaConfig::default();
        config.merge(partial);
        assert!(
            config.permissions.sandbox.env_allow.is_empty(),
            "explicit empty list must survive merge (runtime passes PATH only)"
        );
    }

    #[test]
    fn sandbox_env_allow_roundtrip_via_workspace_file() {
        let tmp_home = tempfile::tempdir().expect("tempdir");
        let _guard = EnvGuard::set(&[
            ("HOME", Some(tmp_home.path().to_str().unwrap())),
            ("MINIMAX_API_KEY", None),
            ("OPENAI_API_KEY", None),
            ("NCA_EDITOR", None),
            ("EDITOR", None),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");

        let mut config = NcaConfig::default();
        config.permissions.sandbox.env_allow = vec!["FOO".to_string(), "BAR".to_string()];
        config.save_workspace_file(dir.path()).expect("save");

        let raw =
            std::fs::read_to_string(workspace_config_path(dir.path())).expect("read local config");
        assert!(
            raw.contains("env_allow"),
            "persisted config should contain env_allow: {raw}"
        );

        let reloaded = NcaConfig::load_for_workspace(dir.path()).expect("reload");
        assert_eq!(
            reloaded.permissions.sandbox.env_allow,
            vec!["FOO".to_string(), "BAR".to_string()]
        );
    }

    #[test]
    fn sandbox_roundtrip_via_workspace_file() {
        let tmp_home = tempfile::tempdir().expect("tempdir");
        let _guard = EnvGuard::set(&[
            ("HOME", Some(tmp_home.path().to_str().unwrap())),
            ("MINIMAX_API_KEY", None),
            ("OPENAI_API_KEY", None),
            ("NCA_EDITOR", None),
            ("EDITOR", None),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");

        // Non-default sandbox values must survive a workspace-file roundtrip.
        let mut config = NcaConfig::default();
        config.permissions.sandbox.mode = SandboxMode::Required;
        config.permissions.sandbox.net = false;
        config.permissions.sandbox.ro_paths = vec![PathBuf::from("/opt/tools")];
        config.permissions.sandbox.rw_paths = vec![PathBuf::from("/tmp/work")];
        config.save_workspace_file(dir.path()).expect("save");

        let raw =
            std::fs::read_to_string(workspace_config_path(dir.path())).expect("read local config");
        assert!(
            raw.contains("sandbox"),
            "persisted config should contain [permissions.sandbox]: {raw}"
        );

        let reloaded = NcaConfig::load_for_workspace(dir.path()).expect("reload");
        assert_eq!(reloaded.permissions.sandbox.mode, SandboxMode::Required);
        assert!(!reloaded.permissions.sandbox.net);
        assert_eq!(
            reloaded.permissions.sandbox.ro_paths,
            vec![PathBuf::from("/opt/tools")]
        );
        assert_eq!(
            reloaded.permissions.sandbox.rw_paths,
            vec![PathBuf::from("/tmp/work")]
        );
    }

    #[test]
    fn sandbox_host_session_tiers_default_off() {
        // Host-session tiers expose the host user session (mic capture,
        // D-Bus services, Wayland), so every flag must default to false.
        let sandbox = &NcaConfig::default().permissions.sandbox;
        assert!(!sandbox.host_audio, "host_audio must default to false");
        assert!(
            !sandbox.host_dbus_session,
            "host_dbus_session must default to false"
        );
        assert!(
            !sandbox.host_xdg_runtime,
            "host_xdg_runtime must default to false"
        );
    }

    #[test]
    fn sandbox_host_session_tiers_parse_from_toml() {
        // Combined case: all three keys in one `[permissions.sandbox]` table
        // must land on the merged config.
        let toml_str = r#"
[permissions.sandbox]
host_audio = true
host_dbus_session = true
host_xdg_runtime = true
"#;
        let partial: PartialNcaConfig = toml::from_str(toml_str).expect("parse");
        let mut config = NcaConfig::default();
        config.merge(partial);
        let sandbox = &config.permissions.sandbox;
        assert!(sandbox.host_audio);
        assert!(sandbox.host_dbus_session);
        assert!(sandbox.host_xdg_runtime);

        // Per-key case: one flag set must not touch the others (merge is
        // per-field, not whole-table).
        for (key, is_audio, is_dbus, is_xdg) in [
            ("host_audio", true, false, false),
            ("host_dbus_session", false, true, false),
            ("host_xdg_runtime", false, false, true),
        ] {
            let partial: PartialNcaConfig =
                toml::from_str(&format!("[permissions.sandbox]\n{key} = true\n")).expect("parse");
            let mut config = NcaConfig::default();
            config.merge(partial);
            let sandbox = &config.permissions.sandbox;
            assert_eq!(sandbox.host_audio, is_audio, "{key}");
            assert_eq!(sandbox.host_dbus_session, is_dbus, "{key}");
            assert_eq!(sandbox.host_xdg_runtime, is_xdg, "{key}");
        }
    }

    #[test]
    fn sandbox_host_session_tiers_roundtrip_via_workspace_file() {
        let tmp_home = tempfile::tempdir().expect("tempdir");
        let _guard = EnvGuard::set(&[
            ("HOME", Some(tmp_home.path().to_str().unwrap())),
            ("MINIMAX_API_KEY", None),
            ("OPENAI_API_KEY", None),
            ("NCA_EDITOR", None),
            ("EDITOR", None),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");

        // Opted-in host-session tiers must survive a workspace-file roundtrip.
        let mut config = NcaConfig::default();
        config.permissions.sandbox.host_audio = true;
        config.permissions.sandbox.host_dbus_session = true;
        config.permissions.sandbox.host_xdg_runtime = true;
        config.save_workspace_file(dir.path()).expect("save");

        let raw =
            std::fs::read_to_string(workspace_config_path(dir.path())).expect("read local config");
        assert!(
            raw.contains("host_audio"),
            "persisted config should contain host_audio: {raw}"
        );

        let reloaded = NcaConfig::load_for_workspace(dir.path()).expect("reload");
        assert!(reloaded.permissions.sandbox.host_audio);
        assert!(reloaded.permissions.sandbox.host_dbus_session);
        assert!(reloaded.permissions.sandbox.host_xdg_runtime);
    }

    #[test]
    fn extra_paths_roundtrip_via_workspace_file() {
        let tmp_home = tempfile::tempdir().expect("tempdir");
        let _guard = EnvGuard::set(&[
            ("HOME", Some(tmp_home.path().to_str().unwrap())),
            ("MINIMAX_API_KEY", None),
            ("OPENAI_API_KEY", None),
            ("NCA_EDITOR", None),
            ("EDITOR", None),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");

        // Save config with extra_paths set.
        let config = NcaConfig {
            extra_paths: vec![
                PathBuf::from("/home/user/projects"),
                PathBuf::from("/opt/data"),
            ],
            ..Default::default()
        };
        config.save_workspace_file(dir.path()).expect("save");

        // The local file should contain extra_paths (diff against empty default).
        let local_path = workspace_config_path(dir.path());
        let raw = std::fs::read_to_string(&local_path).expect("read local config");
        assert!(
            raw.contains("extra_paths"),
            "persisted config should contain extra_paths: {raw}"
        );

        // Reload and verify the mount list survives the roundtrip.
        let reloaded = NcaConfig::load_for_workspace(dir.path()).expect("reload");
        assert_eq!(reloaded.extra_paths, config.extra_paths);
    }

    #[test]
    fn empty_extra_paths_not_persisted() {
        let tmp_home = tempfile::tempdir().expect("tempdir");
        let _guard = EnvGuard::set(&[
            ("HOME", Some(tmp_home.path().to_str().unwrap())),
            ("MINIMAX_API_KEY", None),
            ("OPENAI_API_KEY", None),
            ("NCA_EDITOR", None),
            ("EDITOR", None),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");

        // Default config has empty extra_paths — diff against defaults is empty.
        let config = NcaConfig::default();
        config.save_workspace_file(dir.path()).expect("save");

        let local_path = workspace_config_path(dir.path());
        assert!(
            !local_path.exists(),
            "empty extra_paths should not be persisted"
        );
    }

    /// Regression: `/unmount` must remove the path from the persisted config,
    /// not just from the in-memory `RealFs` state. `persist_mounted_paths`
    /// replaces `extra_paths` wholesale (not append), so a subsequent save
    /// with a shorter list must overwrite — not accumulate.
    #[test]
    fn extra_paths_unmount_removes_from_persisted_config() {
        let tmp_home = tempfile::tempdir().expect("tempdir");
        let _guard = EnvGuard::set(&[
            ("HOME", Some(tmp_home.path().to_str().unwrap())),
            ("MINIMAX_API_KEY", None),
            ("OPENAI_API_KEY", None),
            ("NCA_EDITOR", None),
            ("EDITOR", None),
        ]);
        let dir = tempfile::tempdir().expect("tempdir");

        // Step 1: simulate /mount of two paths.
        let config = NcaConfig {
            extra_paths: vec![
                PathBuf::from("/home/user/projects"),
                PathBuf::from("/opt/data"),
            ],
            ..Default::default()
        };
        config.save_workspace_file(dir.path()).expect("save");

        // Step 2: simulate /unmount of one path — load fresh, overwrite the
        // field with the live mounted_paths() snapshot, save back.
        let mut reloaded = NcaConfig::load_for_workspace(dir.path()).expect("reload");
        reloaded.extra_paths = vec![PathBuf::from("/home/user/projects")];
        reloaded
            .save_workspace_file(dir.path())
            .expect("save after unmount");

        // Step 3: reload and verify the unmounted path is gone, not lingering.
        let after = NcaConfig::load_for_workspace(dir.path()).expect("reload after unmount");
        assert_eq!(
            after.extra_paths,
            vec![PathBuf::from("/home/user/projects")],
            "unmounted path must be removed from persisted config"
        );

        // And the local file must no longer reference the unmounted path.
        let raw = std::fs::read_to_string(workspace_config_path(dir.path())).expect("read");
        assert!(
            !raw.contains("/opt/data"),
            "unmounted path must not linger in config file: {raw}"
        );
    }

    // === named model plans: parse, layer merge, serialization ===

    #[test]
    fn plans_parse_from_toml_with_root_active_plan() {
        // Root scalar `active_plan` must precede any [table] header, exactly
        // like a real config.toml.
        let raw = r#"
active_plan = "economy"

[plans.economy.oracle]
provider = "kimi"
model = "k3"

[plans.economy.fixer]
model = "glm-5.2"

[plans.premium.oracle]
provider = "zhipuai"
"#;
        let partial: PartialNcaConfig = toml::from_str(raw).expect("parse");
        let mut config = NcaConfig::default();
        config.merge(partial);

        assert_eq!(config.active_plan.as_deref(), Some("economy"));

        let economy = config.plans.get("economy").expect("economy plan");
        let oracle = economy.get("oracle").expect("oracle entry");
        assert_eq!(oracle.provider, Some(ProviderKind::Kimi));
        assert_eq!(oracle.model.as_deref(), Some("k3"));
        let fixer = economy.get("fixer").expect("fixer entry");
        assert_eq!(fixer.provider, None);
        assert_eq!(fixer.model.as_deref(), Some("glm-5.2"));

        let premium = config.plans.get("premium").expect("premium plan");
        assert_eq!(
            premium.get("oracle").unwrap().provider,
            Some(ProviderKind::ZhipuAI)
        );
        assert_eq!(premium.get("oracle").unwrap().model, None);
    }

    #[test]
    fn plan_layers_merge_per_entry_and_workspace_overrides_active_plan() {
        // Global layer: define the whole economy plan + select it.
        let global: PartialNcaConfig = toml::from_str(
            r#"
active_plan = "economy"

[plans.economy.oracle]
provider = "kimi"
model = "k3"

[plans.economy.fixer]
model = "glm-5.3"
"#,
        )
        .expect("parse global");
        let mut config = NcaConfig::default();
        config.merge(global);
        assert_eq!(config.active_plan.as_deref(), Some("economy"));
        assert_eq!(
            config.plans["economy"]["fixer"].model.as_deref(),
            Some("glm-5.3")
        );

        // Workspace layer: override one route (per (plan, agent) key) and
        // re-point active_plan.
        let workspace: PartialNcaConfig = toml::from_str(
            r#"
active_plan = "premium"

[plans.economy.fixer]
provider = "openai"
model = "gpt-4o"

[plans.premium.fixer]
model = "gpt-4o-mini"
"#,
        )
        .expect("parse workspace");
        config.merge(workspace);

        // Overridden entry replaced…
        assert_eq!(
            config.plans["economy"]["fixer"].provider,
            Some(ProviderKind::OpenAi)
        );
        assert_eq!(
            config.plans["economy"]["fixer"].model.as_deref(),
            Some("gpt-4o")
        );
        // …untouched sibling route preserved.
        assert_eq!(
            config.plans["economy"]["oracle"].model.as_deref(),
            Some("k3"),
            "sibling route must survive a partial override"
        );
        // New plan added by the workspace layer.
        assert_eq!(
            config.plans["premium"]["fixer"].model.as_deref(),
            Some("gpt-4o-mini")
        );
        // Workspace active_plan overrides the global one.
        assert_eq!(config.active_plan.as_deref(), Some("premium"));
    }

    #[test]
    fn plans_serialize_roundtrip_and_skip_when_empty() {
        // Empty plans + no active plan → neither key is emitted.
        let empty = NcaConfig::default();
        let empty_toml = toml::to_string_pretty(&empty).expect("serialize default");
        assert!(
            !empty_toml.contains("plans"),
            "empty plans must be skipped: {empty_toml}"
        );
        assert!(
            !empty_toml.contains("active_plan"),
            "None active_plan must be skipped: {empty_toml}"
        );

        // Populated config survives a TOML roundtrip.
        let mut config = NcaConfig::default();
        config.plans.insert(
            "economy".into(),
            BTreeMap::from([
                (
                    "oracle".into(),
                    PlanEntry {
                        provider: Some(ProviderKind::Kimi),
                        model: Some("k3".into()),
                    },
                ),
                (
                    "fixer".into(),
                    PlanEntry {
                        provider: None,
                        model: Some("glm-5.2".into()),
                    },
                ),
            ]),
        );
        config.active_plan = Some("economy".into());

        let toml_str = toml::to_string_pretty(&config).expect("serialize populated");
        let back: NcaConfig = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(back.plans, config.plans);
        assert_eq!(back.active_plan, config.active_plan);
    }

    #[test]
    fn plan_entry_serialization_skips_none_fields() {
        // provider-only entry: no `model` key emitted.
        let provider_only = PlanEntry {
            provider: Some(ProviderKind::Kimi),
            model: None,
        };
        let s = toml::to_string(&provider_only).expect("serialize provider-only");
        assert!(s.contains("provider"), "provider pin must serialize: {s}");
        assert!(!s.contains("model"), "None model must be skipped: {s}");

        // An all-None entry serializes to nothing.
        let empty = PlanEntry::default();
        assert_eq!(toml::to_string(&empty).expect("serialize empty"), "");
    }
}
