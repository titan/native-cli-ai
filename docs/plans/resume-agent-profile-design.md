# Restore the active agent profile on resume

> Follow-on to the provider-injection seam
> (`docs/plans/provider-injection-seam-design.md` §3 Option B rationale,
> oracle P2-1): the seam's design review surfaced that the agent profile is
> lost end-to-end on resume — a real user-visible bug. Status: **design** —
> oracle review record at the bottom once run.

## Problem

`Supervisor::apply_agent_profile(name)` applies a specialist persona at
runtime — system prompt, provider, model, permission mode, tool gating — and
stores `agent_profile: Option<AgentProfileConfig>` in memory
(`supervisor.rs:86`). But nothing about the active profile is persisted:

- `SessionMeta` has no agent field (`common/src/session.rs:11`).
- `resume` passes `agent_name: None` into its internal `create`
  (`supervisor.rs:708`), so the profile pipeline (register skills → resolve
  profile → apply overrides → tool gating → system prompt) runs with no
  profile.
- `reset_for_new_session` deliberately *keeps* the persona (its doc-comment:
  "rebuild the system prompt with the same specialist persona"), which makes
  the resume loss an inconsistency, not a policy.

**User-visible symptom:** Tab-cycle to `explorer` (or pick any `[agents.*]`
profile), run turns, exit, `nca --resume` — the session comes back with the
default harness prompt, default permission mode, and full tool set. The
persona silently vanished. Same for skill-based child sessions
(`subagent.rs:115` passes `agent_name: cfg.specialist`): a resumed child
loses its specialist prompt.

## Design

### 1. Persist the profile *name* in `SessionMeta`

```rust
/// Active agent profile name (`[agents.<name>]` or skill-discovered) at the
/// time this session was last saved. Resume re-resolves it against the
/// current config.
#[serde(default, skip_serializing_if = "Option::is_none")]
pub agent_name: Option<String>,
```

**Name, not `AgentProfileConfig`:** profiles are config-derived and
skill-based ones are re-registered at every startup
(`register_skill_agents` in `create`, `supervisor.rs:419`). Persisting the
struct would freeze a stale persona definition across config edits; the name
re-resolves to whatever the *current* config says, and explicit
`[agents.<name>]` entries keep beating skill-discovered ones (existing
precedence, `register_skill_agents_does_not_override_existing_profile`).
Serde compat is the established `default + skip_serializing_if` pattern used
by every optional `SessionMeta` field — old session jsons deserialize with
`agent_name == None`.

### 2. `Supervisor` tracks the selected name

New field `active_agent_name: Option<String>` — the verbatim record of the
last selection:

- `create`: set from `cfg.agent_name` (even if it fails to resolve — see
  §4).
- `apply_agent_profile(name)`: records on the **success path only** —
  `self.active_agent_name = name.map(Into::into)` co-located with
  `self.agent_profile = profile`, i.e. *after* the fallible
  `build_provider(&config)?`. Assigning before the `?` would poison the
  snapshot: a failed switch (e.g. I5's keyless provider) would persist the
  new name while `agent_profile` is unchanged, and the next resume's
  create-threading would resolve that name → unbuildable provider →
  **resume fails loudly** on a session whose switch merely failed. `None`
  means "back to default harness", which is what gets persisted.
- `reset_for_new_session`: **keeps** it, mirroring `agent_profile` retention
  (persona carries into the next session in the same process).
- `current_session_state`: writes it into `SessionMeta.agent_name`, so every
  existing `save()` site persists it — create, resume (re-save), and
  `finish`. (`run_turn` does **not** save the json — see Crash-window below.)
- Mirrored into `SessionSnapshot` (built field-by-field in
  `SessionState::snapshot()`): every other `SessionMeta` field propagates
  there, and orchestration surfaces / hook payloads should see the agent.

### 3. Resume threads the name through `create`'s existing pipeline

```rust
agent_name: loaded.as_ref().and_then(|l| l.meta.agent_name.clone()),
```

replacing the hardcoded `None` at `supervisor.rs:708`. This deliberately
re-uses `create`'s full profile application (provider/model/permission
overrides, tool gating, system prompt, skills re-registration) rather than
calling `apply_agent_profile` post-create, because:

- `apply_agent_profile` calls `build_provider(&config)` unconditionally and
  **discards an injected provider** (documented seam contract). Routing
  through `create` keeps `resolve_provider` semantics: an injected mock
  survives resume *with* its profile's prompt/permissions restored — no
  keyless `build_provider` failure mid-resume (keeps injection tests I3/I5
  coherent).
- The overrides and the system prompt are built in one place; no second,
  subtly-different code path for "profile at construction" vs "profile
  restored".

**Model precedence note:** `resume` still overwrites `sup.model`/
`sup.agent.model` from `loaded.meta.model` *after* `create` applies profile
overrides (existing behavior, `supervisor.rs:717-718`). That is correct here:
`meta.model` was persisted from the profile-adjusted `self.model` at save
time, so the session keeps the model it was actually using, while the
profile supplies provider/permissions/tools/prompt. Known latent divergence
(documented, not synced): if the profile's `model` was edited between save
and resume, `sup.config.model.default_model` keeps the profile's new model
while `sup.model` keeps the persisted one. Benign today — chat uses
`agent.model`, and `apply_agent_profile`/`apply_nca_config` rebuild from
`base_config` — but newly *reachable* via this fix; revisit if a consumer
starts reading `config.model.default_model` for live chat.

### 4. Unresolvable name at resume: warn + default, never fail

If the profile was removed (skill uninstalled, config edited) between save
and resume, `config.agent_profile(name)` yields `None` and `create` falls
back to the default harness prompt. This is the desired graceful path, but
currently silent; `create` gains a `tracing::warn!` (naming the profile)
when `cfg.agent_name` is `Some` and does not resolve. `apply_agent_profile`
gains the symmetric warn when a `Some` name fails to resolve (pre-existing
false-success UX — REPL prints "Switched to @bogus" — stays out of scope,
but the warn makes the dead name visible at switch time too). Recording the
name verbatim (§2) keeps create and resume symmetric: an unresolvable name
no-ops identically in both.

**Distinct case — name resolves but its provider cannot build:** the
profile's `provider` override is applied to `config` and then
`resolve_provider(cfg.provider, &config)?` runs. With production
`provider: None` and an unbuildable override (e.g. key removed since the
session ran), **resume fails loudly** with `ProviderError::Configuration`.
This is intended and identical to `create` with the same profile: loud beats
silently resuming with a half-applied persona + default provider. The user
recovers by fixing the config or resuming with one that resolves. (With an
injected provider the override is moot — the mock wins verbatim.)

## What does NOT change

- `apply_agent_profile` / `apply_nca_config` runtime semantics and the
  injection-seam discard contract (I5 keeps pinning it).
- `SessionMeta` serde shape beyond the one added optional field.
- The orchestration system-prompt-section drop on resume (separate known gap,
  explicitly out of scope in the seam design).
- Provider rebuild semantics: a resumed profile's `provider` override is
  applied by `create` via `resolve_provider` — production (`provider: None`)
  builds it from config as today.

## Crash-window non-goal

The json snapshot is single-writer: written only at create, resume
(re-save), and finish — `run_turn` calls `update_last_session()` only, never
`save()` (AGENTS.md invariant). So a mid-session profile switch persists only
at the next graceful `finish()` (or a later resume re-save). The loss window
for a crash is *switch → finish* — potentially the whole remainder of the
session. Closing it would require a new `AgentEvent` variant
(`AgentProfileChanged`) folded from the event log at resume (the
`fold_child_session_ids` pattern), touching the shared event enum, TUI, IPC,
and replay. Not worth that surface for a soft loss (persona reverts on resume;
the conversation itself is event-log durable); filed as a follow-up trigger
only if a mid-session crash loses a switch in practice.

## Test matrix

Integration (`crates/runtime/tests/resume_agent_profile.rs`, mirrors the
`provider_injection.rs` scaffold):

| # | Test |
|---|------|
| R1 | `create(agent_name = P)` where P has a distinctive `system_prompt` + `permission_mode` → `finish` → `resume` → system prompt contains P's persona; permission mode is P's. |
| R2 | Mid-session `apply_agent_profile(P)` → `finish()` (the only json save after create/resume — `run_turn` never saves) → drop → `resume` → persona restored. |
| R3 | `resume` with injected mock provider on a session whose `agent_name` is set → mock still used (seam intact through the profile path); no keyless `build_provider` failure even when P overrides `provider`. |
| R4 | Profile deleted from config between save and resume → resume succeeds, default harness prompt, no profile. |
| R5 | `reset_for_new_session` keeps the persona: create with P → reset → system prompt still P's; `active_agent_name` survives into the next save. |

Unit (inline):

| # | Test |
|---|------|
| U1 | Legacy `SessionMeta` json without `agent_name` deserializes to `None` (serde compat). |
| U2 | `current_session_state` carries `active_agent_name` into `SessionMeta`. |

## Rollout order

1. Design doc committed (this file) before dispatching lanes.
2. Oracle review → findings applied → amend doc commit.
3. Implementation (orchestrator-direct if the fixer lane is still failing):
   `SessionMeta.agent_name`, `Supervisor.active_agent_name`, resume
   threading, warn-on-unresolved.
4. Integration tests (tester lane, cross-model; fallback direct).
5. Ponytail pass over the diff; `cargo fmt --all -- --check`,
   `cargo clippy --workspace -- -D warnings`, `cargo test --workspace`
   (mind `/dev/shm` quota; ≥900 s timeout, output to file).
6. Commit `feat(runtime): restore active agent profile on resume`. No
   boundary/deps change → tech-stack/architecture docs untouched.

## Open questions — resolved by oracle review (2026-08-24)

1. **Name vs struct → name.** Frozen structs stale across config edits, can
deserialize unbuildable providers, and bypass startup re-registration +
`[agents.*]`-beats-skill precedence. Resume fidelity = "what the user has
configured now".
2. **Resume via `create` vs post-create apply → create-threading.** Post-create
`apply_agent_profile` discards injected providers and runs after the
meta-restore block (ambiguous ordering). Caveat adopted: a resolvable profile
with an unbuildable provider now fails resume loudly (see §4) — identical to
create, intended.
3. **Verbatim recording → keep, with the name in the warn.** Both choices
converge to the default harness; verbatim preserves intent, keeps the warn
firing on every resume, and never silently forgets. Hard requirement adopted:
assign only on the success path (P1-2 below).
4. **Crash-window → event-fold stays out,** but the original premise was wrong
(there is no per-turn json save; single-writer at create/resume/finish). Loss
window is switch → finish, which makes the fold *less* attractive, not more.
5. **meta.model precedence → persisted model wins,** off-by-one line ref fixed
(717-718). The `config.model.default_model` divergence is documented (§3),
not synced — no consumer reads it for live chat today.

Additional findings adopted: `SessionSnapshot` mirrors `agent_name` (P3-1);
`apply_agent_profile` gains the symmetric unresolvable-name warn (P3-3, warn
only — the REPL false-success message is out of scope); child-session persona
duplication (`context_prompt` + system prompt) noted as a follow-up (P3-2,
pre-existing at create time, this fix merely makes both copies survive resume).

## Oracle review record (2026-08-24)

Verdict: **APPROVE WITH REQUIRED CORRECTIONS.** No P0. Core mechanism
(name-persistence + create-threading) confirmed correct. All findings applied:

- **P1-1 (factual):** "per-turn save inside `run_turn`" was false — line 837
  is `resume`'s re-save; `run_turn_with_images` calls `update_last_session()`
  only. Crash-window rewritten (loss window = switch → finish); R2 rewritten
  to `finish()` before drop.
- **P1-2 (ordering hazard):** `active_agent_name` must be assigned only on the
  success path of `apply_agent_profile`, co-located with
  `self.agent_profile = profile` — otherwise a failed switch poisons the
  snapshot and the next resume fails loudly on the unbuildable provider.
  Adopted as a stated requirement in §2.
- **P2-1:** resolvable-name + unbuildable-provider resume failure is new
  behavior vs today (resume never applied profiles); adopted as intended and
  documented in §4.
- **P2-2:** `config.model.default_model` divergence documented in §3; not
  synced (benign today, no live-chat consumer).
- **P3-1:** `SessionSnapshot` mirrors the field (was silently omitted in the
  original proposal).
- **P3-2:** child persona duplication (context_prompt + system prompt) —
  pre-existing; follow-up, not fixed here.
- **P3-3:** unresolvable-name warn added symmetrically to
  `apply_agent_profile`; REPL "Switched to @bogus" false-success message
  stays out of scope.
