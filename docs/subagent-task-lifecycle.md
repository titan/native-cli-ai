# Async Task Lifecycle for nca Subagents

Status: P2 implemented (task_status/task_result/task_message/task_cancel/
task_revive + background spawn + SubagentRegistry with live handles,
control lease, alias resolution, and generation-tracked revive).
P3–P4 pending. Researched against
oh-my-opencode-slim 2.2.18 (`task`/`task_result`/`task_status`/`task_message`/
`task_cancel`/`task_revive`, Background Job Board, orchestrator wake
scheduler, `wait_for_user`).

## 1. Capability Mapping

| Upstream concept | nca today | Gap |
|---|---|---|
| `task(background:true)` | `spawn_subagent` tool → `SpawnRequest` → `spawn_subagent_consumer` → `tokio::spawn` per child (`crates/runtime/src/subagent.rs:250+`); in-batch children already run concurrently | Tool **blocks** on `oneshot` reply up to 600s (`crates/core/src/tools/spawn_subagent.rs::execute`); no `background` flag; parent turn cannot end while children run |
| `task_result` | `SpawnResponse.output` returned synchronously; child json at `<ws>/.nca/sessions/<id>.json` holds `messages` + `SessionStatus` | No read-only tool to fetch a **background** child's final assistant message after the parent turn ended |
| `task_status` | `ChildSessionCompleted{status}` event + `SessionMeta.status` (`crates/common/src/session.rs`) | No live `busy/idle` distinction; no registry handle to reach a running child's state |
| `task_message` | `Supervisor::inbox_sender()` → `InboxItem::{UserPrompt,Steering}`, claimed at step boundaries (`crates/core/src/agent_driver.rs`, capacity 16) | Mechanism exists; no tool exposing it to the orchestrator |
| `task_cancel` | `AgentLoop.cancel_flag: Arc<AtomicBool>` (`crates/core/src/agent.rs:31`), `cancel_handle()` at `:258`, polled in tool pipeline (50ms) and stream loop (25ms) | No tool, no registry handle to reach a child's flag, no retained-session bookkeeping |
| `task_revive` | `Supervisor::resume` folds json + event log (`crates/runtime/src/supervisor.rs:720`); worktree retained on disk (`worktree.rs::create_worktree`) | No "resume same child with new prompt" path; child `finish`/drop tears down the supervisor |
| Background Job Board | lineage only: `ChildSessionSpawned`/`ChildSessionCompleted` + `fold_child_session_ids` (`supervisor.rs`) | No state registry, aliases, generations, terminal-unreconciled, or lease-based single-writer |
| orchestrator wake scheduler | `inbox_sender` can enqueue a wake prompt as `InboxItem::UserPrompt` | No timer/hook to trigger wake when a child goes terminal |
| `wait_for_user` | `ask_question` (`ToolRegistry::is_interactive` barrier, `tool_pipeline.rs` Phase 2) | No "pause wake, no options, end turn" HITL signal |

**Key enablers already present:** synchronous await makes `task_result`
trivial in the foreground path; concurrent `join_all`
(`crates/core/src/tool_pipeline.rs` Phase 2) gives in-batch parallelism;
`core::replay`/`Supervisor::resume` is the revive machinery; `inbox_sender`
is the wake channel; `cancel_flag` is the abort mechanism.

## 2. Target Architecture

### common (types/events/config)
- `crates/common/src/event.rs`: add
  - `ChildSessionStatusChanged { child_session_id, state, generation, alias, result_summary }`
  - `ChildMessageQueued { child_session_id, accepted: bool }`
  - `OrchestratorWake { reason }` (informational, replay-ignored)
- `crates/common/src/session.rs`: add `ChildSessionState { Pending, Running,
  Completed, Cancelled, Failed }` (fold into `SessionStatus` for terminal
  mapping). Extend `SessionMeta` with `generation: u64` (default 0) and
  `parent_alias: Option<String>`.
- `crates/common/src/config.rs`: add `SubagentConfig { background: bool,
  wake: WakeConfig { enabled, interval_ms }, wall_clock_timeout_ms:
  Option<u64>, result_timeout_ms }`.

### core (tools/protocol)
- `crates/core/src/tools/spawn_subagent.rs`: add `background: bool` to
  `SpawnRequest` and the schema. Add `alias`/`task_id` for P2 reuse.
- New `crates/core/src/tools/subagent_control.rs`:
  `SubagentControlRequest` enum (`Status{id}`, `Result{id}`, `Message{id,text}`,
  `Cancel{id,reason}`, `Revive{id,prompt}`) + `SubagentControlResponse`; each
  operation is a `ToolExecutor` (`task_status`, `task_result`, `task_message`,
  `task_cancel`, `task_revive`) sending over one
  `mpsc::Sender<SubagentControlRequest>` with a `oneshot` reply. This mirrors
  the existing `SpawnRequest` pattern exactly.
- `crates/core/src/tools/wait_for_user.rs`: `wait_for_user` tool (see §3).
- `ToolRegistry::is_interactive` must continue to gate `ask_question` only;
  `wait_for_user` is non-interactive (no oneshot) but must still run as a
  turn-ending barrier — mark it interactive for serialization safety.

### runtime
- `crates/runtime/src/subagent.rs`: split `spawn_child_session` (blocking,
  existing) from new `run_child_session_background` returning a `ChildHandle
  { session_id, generation, cancel_flag: Arc<AtomicBool>, inbox_tx, event_rx,
  join_handle, worktree_path, branch }`. Child runs its own
  `spawn_event_fanout` (existing) with `parent_forward` for activity + status
  events; on terminal, emits `ChildSessionCompleted` +
  `ChildSessionStatusChanged`, then notifies the registry.
- New `crates/runtime/src/subagent_registry.rs`: `SubagentRegistry { map:
  HashMap<session_id, RegistryEntry>, leases: HashMap<session_id, Lease>,
  counters }`. `RegistryEntry { state, generation, alias, parent_session_id,
  agent, description, cancel_flag, inbox_tx, worktree_path, result_summary }`.
  Port a **simplified** lease concept from upstream `background-job-board.ts`
  (single `ControlLease` per session_id/generation for cancel/revive/message)
  — no `statusUncertain`/liveness-reconciliation machinery.
- New `crates/runtime/src/wake_scheduler.rs`: on `ChildSessionStatusChanged`
  to terminal, if `wake.enabled` and parent has incomplete todos,
  `tokio::time::sleep(interval)` then
  `inbox_sender().try_send(InboxItem::UserPrompt { static wake text })`. One
  in-flight wake per parent (reserve/commit gate).
- `crates/runtime/src/supervisor.rs`: own an `Arc<SubagentRegistry>`; expose
  it + `wake_scheduler` to the CLI; wire a `SubagentControlConsumer`
  (analogous to `spawn_subagent_consumer`) that resolves control requests
  against the registry and child handles. Do **not** write parent json from
  any of these (preserve single-writer).

### cli
- `crates/cli/src/repl.rs`: register the five control tools +
  `wait_for_user`; wire `SubagentControlConsumer`; expose
  `ChildSessionStatusChanged` into the existing Elm feedback channel.
- `crates/cli/src/tui/elm/`: job-board panel fed by status events
  (best-effort, `is_dirty` guarded). CLI `/jobs` command reads
  `SubagentRegistry::list`.

### Wire/IPC
- All registry→UI traffic is `AgentEvent` over the existing event fanout
  (bounded). Control requests ride a dedicated `mpsc::channel(100)`; replies
  via `oneshot` with `result_timeout_ms` (default 30s for
  status/result/message, 10s for cancel, no hard cap for revive's first
  turn). No new Unix-socket command variants required in P1–P3.

## 3. Semantics Decisions

- **States:** `pending → running → completed | cancelled | failed`, plus
  terminal. `failed` maps `SessionStatus::Error`; `cancelled` maps
  `Cancelled`. A "retained" child is any terminal state with worktree
  preserved (cancel **never** deletes; `remove_worktree` only on explicit
  parent close).
- **`task_cancel`:** acquire a control lease; set the child's
  `cancel_flag = true` (cooperative abort within 25–50ms, `agent_driver.rs`
  stream loop / `tool_pipeline.rs`). The child's `run_turn` returns
  `Err("run cancelled")`; its `finish(EndReason::Cancelled)` runs; worktree/
  branch retained; `result_summary = "cancelled: <reason>"`. No rollback.
- **`task_revive`:** resolve retained child by id/alias; if still `running`,
  run cancel first (port upstream `task-revive.ts` order). Then
  `Supervisor::resume(config, workspace_root, ..., child_id, ...)` (folds
  json+log, `supervisor.rs:720`), reuse the existing worktree path (do
  **not** re-`create_worktree`), bump `generation`, and `run_turn(new_prompt)`.
  Reply carries the child's new terminal output.
- **Background spawn:** `background: bool` on `spawn_subagent` (not a new
  tool). When `true`, `run_child_session_background` returns
  `{child_session_id, state:"running"}` immediately and the tool's oneshot is
  answered now (not held for 600s). Foreground keeps the existing 600s
  synchronous contract.
- **Parent turn end + wake:** background children run on detached tokio
  tasks; the parent model ends its turn normally (its `TurnCompleted` still
  fsyncs via the existing fanout barrier). On child terminal, the spawn
  consumer's background arm (and only that arm) calls the wake scheduler,
  which delivers the static wake text as a **Submit through the CLI
  cmd-queue** (`TuiCmd::Submit` → the single loop that serializes all
  `run_turn` calls). This replaces the earlier `inbox_sender()`/
  `InboxItem::UserPrompt` sketch: an idle parent is parked on
  `cmd_rx.recv()` and never claims inbox items — inbox alone cannot start a
  turn — so cmd-queue delivery keeps turn serialization in one place (the
  busy flag stays truthful) with no double-delivery (the wake text IS the
  next `run_turn`'s prompt, preserving "the next `run_turn` claims it at
  turn start"). Wakes land BETWEEN turns, never mid-turn (steering during
  a busy turn already has its own `InboxItem::Steering` path). stdio REPL
  and one-shot modes have no cmd queue and run foreground defaults
  (documented limitation: no wake delivery path). No `wait_for_user` is
  used for background completion.
- **`wait_for_user`:** a **new** tool, not `ask_question` reuse — it has no
  options/oneshot and must not emit `QuestionRequested`. It returns
  `state: waiting_for_user` + guidance, and calls `wake_scheduler.pause()`.
  It is registered `is_interactive` (barrier) so it runs last in a batch; it
  preserves the one-active-question invariant trivially because it never
  opens a question channel.

## 4. What NOT to Port (YAGNI)

- opencode host internals: `getClient`, `session.abort/prompt/promptAsync/
  messages`, SDK `ToolContext.sessionID/agent`, `task map`/
  `session-runtime-status` liveness snapshots (`tools/cancel-task.ts`,
  `utils/session-runtime-status.ts`). nca already has `cancel_flag` +
  `SessionStore` + `inbox_sender` as native equivalents.
- Plugin loader / hook admission (`tool.execute.before`),
  `ctx.tool.transform`, tmux/cmux multiplexer + pane lifecycle, shell-rc
  installer (`src/cli/background-subagents.ts`), background concurrency
  admission caps (`backgroundJobs.concurrency`) — nca in-batch parallelism
  is bounded by the pipeline and the registry already prevents overlapping
  write ownership by convention.
- `statusUncertain`/`markStopped`/liveness reconciliation and the full lease
  taxonomy (`background-job-store.ts`) — over-engineered for a
  single-process supervisor; keep only the cancel/revive/message lease.
- Wake fingerprint cap + `continueOnIdle` beta + `BackgroundJobSupervisor`
  timers — defer; a single reserve/commit wake gate suffices. Wall-clock
  timeout is opt-in and only if P3 usage shows stalls.
- Aliased session reuse beyond `parent-scoped alias` (skip per-agent LRU
  `maxReusablePerAgent` in P2; add only if asked).

## 5. Phased Plan

### P1 — Introspection + result-fetch (S)
- `common`: `ChildSessionState`, `ChildSessionStatusChanged` event,
  `SubagentConfig` (minimal).
- `runtime`: `SubagentRegistry` populated from `ChildSessionSpawned`/
  `ChildSessionCompleted` events + `SessionStore::load` for terminal state;
  `task_status`/`task_result` read-only control consumer.
- `core`: `task_status`, `task_result` tools + `SubagentControlRequest` wire
  type.
- `cli`: register tools; `/jobs` list.
- **Tests:** unit — registry folds events; `task_result` returns last
  assistant message from child json; `task_status` reports `running`/
  terminal. No mock network.
- **Rollback:** additive only; foreground spawn untouched.

### P2 — cancel/revive (M)
- Refactor `spawn_child_session` → `run_child_session_background`; add
  `background` + `alias`.
- `core`: `task_cancel`, `task_revive`, `task_message`; `background` schema.
- `runtime`: control lease (single per id/generation); `task_message` →
  `inbox_tx.try_send(InboxItem::Steering)`; `task_revive` →
  `Supervisor::resume` + worktree reuse + `generation` bump.
- **Tests:** cancel mid-turn (gated provider) ends child as `cancelled` and
  keeps worktree; revive reuses retained json and emits a new turn; message
  queues without interrupting a running turn.
- **Rollback:** medium — touches spawn path; keep blocking path behind
  `background==false`.

### P3 — background jobs + wake (L)
- `background` default-on for orchestrator (`[subagent] background`,
  default true; omitted flag inherits it, explicit flag always wins);
  background spawn replies immediately (`status:"running"`) and runs the
  child detached; completion notification → `ChildSessionStatusChanged`.
- `wake_scheduler.rs`: idle resume after child terminal; debounced static
  wake text delivered through the CLI cmd-queue as a Submit (see §3 — not
  `inbox_sender`, per the implemented design).
- Wall-clock timeout: **DEFERRED** (spec §4 already deferred it; no config
  field shipped) — revisit only if usage shows stalls.
- `OrchestratorWake` event: **CUT** (redundant — the wake text is
  self-identifying and lands in the transcript via the wake turn itself;
  add later only if observability demands it).
- `wait_for_user`: **remains P4** (not folded into P3; no pause API ships
  without a caller).
- **Tests:** integration — parent turn ends while child runs; wake fires on
  terminal; no double-wake (reserve/commit); bounded channels never grow
  unbounded. Landed: `tests/wake_integration.rs` (default-on + wake-once,
  explicit-foreground no-wake, rollback default-off, cancel wake with
  reason, revive no-wake, in-window coalescing).
- **Rollback:** highest — gate behind `SubagentConfig.background`/
  `wake.enabled`; ship foreground fallback.

### P4 — wait_for_user (S)
- `core`: `wait_for_user` tool + `wake_scheduler.pause()`.
- `runtime`/`cli`: register + prompt guidance (orchestrator system prompt
  update).
- **Tests:** tool emits no `QuestionRequested`; wake suppressed until next
  external user message.
- **Rollback:** trivial; can fold into P3 if cheap.

## 6. Risks & Invariants

- **Single-writer session json:** registry/control consumer never call
  `Supervisor::save` on another session; result-fetch is read-only via
  `SessionStore::load`. Child writes its own json only via its own
  `finish`. Preserved.
- **fsync-at-TurnCompleted:** background children run their own
  `spawn_event_fanout` + commit barrier (`session_utils.rs`); the parent's
  turn commit is unaffected. Preserved.
- **One active question:** `wait_for_user` never opens a `QuestionRequested`;
  wake prompts are `InboxItem`s, not questions; `is_interactive` barrier
  still serializes `ask_question`. Preserved.
- **Bounded channels:** control `mpsc(100)`; inbox stays at 16; status
  events use `try_send`. Preserved.
- **Worktree cleanup on cancel:** cancel never calls `remove_worktree`;
  branch retained for revive; cleanup only on explicit close. Preserved
  (matches upstream "no rollback").
- **Approval flow for children:** children already run `BypassPermissions` +
  stripped `ask_question` (`subagent.rs:80,137`); cancel/revive/message
  don't re-enter approval. Preserved.
- **600s timeout interplay:** `background:true` answers the oneshot
  immediately and drops the reply channel — no 600s wait; foreground retains
  it. Must be made explicit in `spawn_subagent.rs::execute` to avoid the
  timeout branch firing.
- **Lineage/resume:** registry entries must be re-derivable from the event
  log after crash (emit `ChildSessionStatusChanged`/`ChildSessionSpawned`
  into the log; registry is an in-memory projection, mirroring
  `fold_child_session_ids`). Preserved.
- **Wake hook scope (P3):** the terminal wake hook fires ONLY in the spawn
  consumer's background (detached) arm. Foreground spawns never wake (their
  output was returned inline) and `task_revive` never wakes (the reviving
  parent is mid-turn holding the reply). Preserved.

## 7. Estimated Size

| Phase | Size | Scope |
|---|---|---|
| P1 | S (~250 LOC) | 2 tools, registry read path, 1 event |
| P2 | M (~800 LOC) | background spawn, 3 control tools, leases, resume/revive |
| P3 | L (~1200 LOC) | wake scheduler, completion wiring, optional wall-clock, integration tests |
| P4 | S (~120 LOC) | 1 tool, pause hook, prompt text |

Total ≈ 2.4k LOC plus tests, concentrated in `runtime` (registry/wake) and
`core` (tool wire types); `cli` surface stays thin.
