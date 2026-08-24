# Provider-injection seam for `Supervisor`

> Unlocks true supervisor-level E2E tests with a mock `Provider` — no fake
> credentials, no post-hoc `agent_mut().replace_provider()` mutation.
> Follows the middleware-chain oracle record's candidate ②
> (`docs/plans/middleware-chain-composition-design.md` §"Open questions", item
> 5: "runtime placement forfeits unit-testability — `SupervisorConfig` has no
> provider injection seam, the known candidate ② limitation"). Status:
> **design** — oracle review record at the bottom once run.

## Problem

`Supervisor::create` always calls `build_provider(&config)?`
(`supervisor.rs:425`) before constructing the `AgentLoop`. There is no way to
supply a pre-built provider, so every supervisor-level integration test must:

1. Fabricate a *valid-looking* API key so `build_provider` succeeds for the
   configured provider kind (`offline_config()` sets
   `deepseek.api_key = Some("test-key")`; see `inbox.rs:26-28`'s explicit
   "no mock seam" note).
2. Construct the supervisor, then post-hoc swap in the mock via
   `agent_mut().replace_provider(mock)` (established pattern in
   `cost_guard_resume.rs`, `phase_c.rs`, `inbox.rs`, `turn_commit.rs`).

This is fragile in three ways:

- **Couples runtime tests to provider construction.** Adding a provider whose
  `from_config` has a new mandatory field forces every offline config to
  carry it, or `create` fails before the mock ever runs.
- **Silently loses the mock on rebuild.** `apply_agent_profile`
  (`supervisor.rs:1424`) and `apply_nca_config` (`supervisor.rs:1485`) call
  `build_provider(&config)?` + `replace_provider`, discarding any injected
  mock. A test that exercises a profile/config switch mid-session would
  observe a real provider (network) where a mock was expected.
- **No resume path.** `resume` funnels into `create` (building a
  `SupervisorConfig` internally, `supervisor.rs:679`), so the mock must again
  be swapped in post-construction.

## Goal

A single, minimal injection point so that `create` and `resume` accept an
optional pre-built `Arc<dyn Provider>` and, when present, use it verbatim and
**skip `build_provider` entirely**. Production callers pass `None` and are
unchanged.

## Design

### 1. `SupervisorConfig` gains an optional provider field

```rust
pub struct SupervisorConfig {
    pub config: NcaConfig,
    pub workspace_root: PathBuf,
    pub safe_mode: bool,
    pub interactive_approvals: bool,
    pub session_id: Option<String>,
    pub approval_handler: Option<Arc<dyn ApprovalHandler>>,
    pub orchestration_context: Option<OrchestrationContext>,
    pub agent_name: Option<String>,
    /// Optional pre-built provider. When `Some`, `create` uses it verbatim
    /// and skips `build_provider` (test seam; production passes `None`).
    pub provider: Option<Arc<dyn Provider>>,
}
```

**Shape rationale — field over factory-fn or builder-closure:**

- A `Box<dyn FnOnce(&NcaConfig) -> Result<Arc<dyn Provider>, ProviderError>>`
  factory's real value is **re-invocation on rebuild**: a factory would let
  `apply_agent_profile`/`apply_nca_config` call `|_| Ok(mock.clone())` and
  preserve a mock across a profile switch (the §4 footgun). It is *not*
  primarily about customizing derivation from config. That is a future need;
  YAGNI favors the field today, and the trigger to upgrade to a factory is
  exactly "the injected provider must survive a profile/config switch".
- A builder-closure (`Box<dyn FnOnce() -> Arc<dyn Provider>>`) adds a
  deferred-construction indirection with no lazy-construction requirement —
  the provider is needed immediately, synchronously, in `create`.
- `Option<Arc<dyn Provider>>` is exactly the type `AgentLoop::new` already
  takes and exactly what `replace_provider` swaps in — zero impedance
  mismatch, no new trait, no new boxed closure. `Provider: Send + Sync`
  (`provider.rs:77`), so `Arc<dyn Provider>` is a valid field with no
  `Clone`/`Debug` obligations (`SupervisorConfig` has no `#[derive]`). YAGNI
  favors the field.

**Required doc-comments** (executable contract for the §4 footgun, not just
prose in this doc): the `provider` field is documented as construction-only
and discarded by `apply_agent_profile`/`apply_nca_config`, and both `apply_*`
method doc-comments gain the same warning.

### 2. `create` resolves provider from the field

```rust
let provider = match cfg.provider {
    Some(provider) => provider,
    None => build_provider(&config)?,
};
```

`cfg.provider` is moved out (partial move of a non-`Copy` struct field is
fine — `create` already moves `cfg.config`, `cfg.approval_handler`,
`cfg.orchestration_context`, etc. individually). No `build_provider` call, no
credential check, when a provider is supplied.

### 3. `resume` threading

`resume` currently takes 6 positional params and builds a `SupervisorConfig`
internally. Two options:

**Option A (recommended) — add a 7th positional `provider` param:**

```rust
pub async fn resume(
    config: NcaConfig,
    workspace_root: &Path,
    safe_mode: bool,
    interactive_approvals: bool,
    session_id: &str,
    approval_handler: Option<Arc<dyn ApprovalHandler>>,
    provider: Option<Arc<dyn Provider>>,   // NEW
) -> Result<Self, ProviderError> {
    ...
    let mut sup = Self::create(SupervisorConfig {
        config: config.clone(),
        workspace_root: workspace_root.to_path_buf(),
        safe_mode,
        interactive_approvals,
        session_id: Some(session_id.into()),
        approval_handler,
        orchestration_context: None,
        agent_name: None,
        provider,                            // NEW
    })
```

**Option B — refactor `resume` to accept `SupervisorConfig` directly**
(collapsing the two parallel construction surfaces into one). This also fixes
a latent limitation — `resume` passes `orchestration_context: None` and
`agent_name: None` into `create`, so the *agent profile* is lost end-to-end
on resume (`SessionMeta` has no agent field; nothing restores `agent_profile`
or the specialist system prompt). Note: the `orchestration` **field** is NOT
lost — resume restores `sup.orchestration = loaded.meta.orchestration.clone()`
(`supervisor.rs:708`); what's dropped is the orchestration *section of the
system prompt* (built in `create` with `None`). Option B widens the change,
makes `session_id` optional at the type boundary (resume requires one, so it
gains a `.ok_or(...)` error branch), and churns every call site from
positional to struct-literal shape.

**Recommendation: A.** The seam's job is provider injection; folding resume's
signature into `SupervisorConfig` is a separate cleanup with its own blast
radius. The latent agent-profile-on-resume loss is a **user-visible bug**
(specialist persona vanishes on resume) and is filed as its own follow-on
("restore the active agent profile on resume"), not silently fixed inside a
test-infra change. Both options touch the same ~16 resume call sites; A keeps
the churn purely mechanical (`, None` appended).

### 4. Profile/config-switch interaction (non-goal, documented)

`apply_agent_profile` and `apply_nca_config` rebuild the provider from config
and `replace_provider`, so a mock injected at construction is **discarded** on
those paths. This is accepted and documented as out of scope:

- The seam unlocks *construction-time* E2E (create/resume/reset), which is the
  entire need behind candidate ②.
- Profile/config switching is a production behavior that inherently means "a
  new provider is wanted" — preserving a test mock there would require storing
  a provider *factory* alongside the live provider (back to the over-general
  shape rejected in §1).
- `reset_for_new_session` does **not** rebuild the provider (it clears
  messages/cost and keeps `agent.provider`), so injected mocks survive session
  resets — the common E2E loop of create → run → reset → run keeps working.

**Footgun and its guard (P2-2):** with a *keyless* config (the standard test
setup), `apply_agent_profile`/`apply_nca_config` call
`build_provider(&config)?` and **fail loudly** on the missing key — so they do
*not* silently hit the network. The silent-network failure only occurs when
the config happens to carry real credentials (e.g. a dev's `config.local.toml`
leaking into a test run). The guard is threefold: (1) the `provider` field
carries a construction-only doc-comment, (2) both `apply_*` methods document
that they discard an injected provider, (3) an I5 test pins the loud-failure
behavior so the footgun is an executable contract, not prose.

**Keepalive note (P3, no action):** `set_keepalive_profile` derives its
*profile* from `config.provider.default`, but `CacheKeepalive::start` pings
`Arc::clone(&agent.provider)` — the injected mock, not a real endpoint
(`agent_driver.rs:571`). A mock + DeepSeek-default config yields an "enabled"
keepalive that would ping the mock after the idle interval; no network leak.

### 5. What does NOT change

- `build_provider` / `build_provider_for` signatures and behavior.
- `AgentLoop::new`, `replace_provider`, `extend_middleware` — untouched.
- Production call sites (`service.rs`, `runner.rs`, `subagent.rs`) pass
  `None`; subagent children continue to inherit the real config-derived
  provider (injection is not threaded into `spawn_subagent_consumer` — a child
  that needs a mock is constructed directly in a test, not via the spawn
  consumer).
- No new `AgentEvent` variants, no new config section, no dep changes.

## Test matrix

Runtime integration (`crates/runtime/tests/provider_injection.rs`, new,
mirrors the `cost_guard_resume.rs` scaffold):

| # | Test |
|---|------|
| I1 | `create` with `provider: Some(mock)` and **no API key** in config succeeds; `run_turn` drives the mock (mock call count ≥ 1, output observable). Proves `build_provider` is skipped — a config with zero credentials would otherwise fail loudly. |
| I2 | `create` with `provider: None` and no key **fails** (`ProviderError::Configuration`) — pins that production behavior is unchanged and that I1's success is *because of* the injection, not a loose credential check. |
| I3 | `resume` with `provider: Some(mock)` (via the new param) uses the mock on the resumed session; a `MustNotBeCalledProvider`-style guard or call-count assertion proves the real provider is never constructed/used. |
| I4 | `reset_for_new_session` preserves the injected provider: create with mock → reset → `run_turn` still drives the mock. |
| I5 | `apply_agent_profile` (to a profile that changes provider) on a keyless injected-mock session **fails loudly** with `ProviderError::Configuration` (missing key), proving the mock is discarded via `build_provider` — not silently replaced by a network provider. Pins the §4 footgun as an executable contract. |

Unit (inline in `supervisor.rs`, if a `resolve_provider` helper is extracted):

| # | Test |
|---|------|
| U1 | `resolve_provider(Some(p), cfg)` returns `p` verbatim (pointer/type identity, no `build_provider`); `resolve_provider(None, cfg)` returns `build_provider(cfg)`'s result (error propagates unchanged). |

Mechanical note (P2-3): `supervisor.rs` currently names `ProviderError` and
`build_provider` but not the `Provider` trait; the new field requires
`use nca_core::provider::Provider;`.

Existing suites (`phase_c`, `inbox`, `turn_commit`, `cost_guard_resume`,
`replay_resume`) stay green — their call sites gain `, None` mechanically, and
their current fake-key + `replace_provider` scaffolding is *left intact*
(migrating them to the seam is a follow-on, not this change's scope).

## Rollout order

1. Design doc committed (this file) **before** dispatching lanes.
2. Oracle review → findings applied → amend doc commit.
3. Fixer lane: `SupervisorConfig.provider` field + `create` resolution +
   `resume` 7th param + mechanical `, None` at all `SupervisorConfig` /
   `resume` call sites (service, runner, subagent, phase_c, inbox,
   turn_commit, cost_guard_resume, replay_resume) + `Provider` import + U1 +
   the `provider`/`apply_*` doc-comments. Unit green.
   `CARGO_TARGET_DIR=<main>/target`, warm artifacts.
4. Tester lane (cross-model): I1–I4 integration; report deviations only.
5. Ponytail pass over the full diff; `cargo fmt --all -- --check`,
   `cargo clippy --workspace -- -D warnings`, `cargo test --workspace`
   (≥900 s timeout, output to file).
6. Commit `feat(runtime): provider-injection seam for Supervisor`; doc-sync
   same-commit if any architecture note warrants it (no boundary change —
   likely none). No dep changes → tech-stack untouched.

## Non-goals

- Provider *factory* / builder-closure shapes (over-general for the need; §1).
- Refactoring `resume` onto `SupervisorConfig` (separate cleanup; §3).
- Fixing resume's latent agent-profile loss (user-visible bug, filed as its own
  follow-on: "restore the active agent profile on resume"; §3).
- Fixing the orchestration system-prompt-section drop on resume (minor; the
  `orchestration` field itself is restored at `supervisor.rs:708`).
- Persisting an injected provider across `apply_agent_profile` /
  `apply_nca_config` (production rebuild semantics; §4).
- Threading injection into `spawn_subagent_consumer` / `ChildSessionConfig`
  (children keep real providers; tests construct mock children directly).
- Migrating the existing fake-key + `replace_provider` test scaffolding to the
  seam (follow-on; keeps this change mechanical).

## Open questions — resolved by oracle review (2026-08-24)

1. **Field vs factory vs builder-closure → field.** Factory's real value is
   re-invocation on rebuild (surviving a profile switch), which is a future
   need, not today's; upgrade trigger is explicit ("mock must survive a
   profile switch").
2. **Resume threading → Option A (7th positional param).** Acceptable; 16
   `resume` + 9 `SupervisorConfig` call sites, A keeps churn mechanical.
3. **Scope → sound**, with a guard: keyless config fails loudly on
   `apply_*`; only leaked real creds go silent. Guard = doc-comments + I5.
4. **I2 worth keeping** (cheap negative control). I1 already proves
   `build_provider` is skipped; I5 added for the footgun.
5. **No P0/P1.** Partial move of `cfg.provider` is fine (`SupervisorConfig`
   has no `Drop`; `create` already partially moves `cfg.config`/
   `cfg.approval_handler`/etc.); keepalive is mock-bound and safe.

## Oracle review record (2026-08-24)

Verdict: **APPROVE with minor changes.** No P0, no P1. All findings applied:

- **P2-1 (doc accuracy):** corrected the Option-B rationale — the
  `orchestration` field is restored on resume (`supervisor.rs:708`); what's
  lost is the orchestration *system-prompt section*, while the agent profile
  is lost end-to-end. Follow-on re-scoped to "restore the active agent profile
  on resume" (a real user-visible bug).
- **P2-2 (footgun guard):** added the threefold guard — `provider` field
  doc-comment, `apply_*` doc-comments, and I5 loud-failure pin.
- **P2-3 (mechanical):** `use nca_core::provider::Provider;` import noted in
  the test matrix.
- **P3 (note):** keepalive profile is config-derived but pings the injected
  mock; harmless, documented in §4.
