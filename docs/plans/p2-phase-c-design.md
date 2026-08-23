# P2 Phase C Design — Single-Writer Session Snapshots + Graceful Writer Close

> Implements §"Non-goals (Phase C)" of `p2-phase-b-design.md`. Prerequisite:
> Phase B is merged (`6013ced`): the event log is durable at turn boundaries
> (commit barrier) and resume is replay-authoritative (`select_resume_messages`).
> Status: design (pre-implementation).

## Problem

Phase B made the event log the sole durable truth mid-session, but the json
snapshot is still written from four kinds of sites that violate a
single-writer discipline:

1. **Per-turn save.** `Supervisor::run_turn_with_images` ends with
   `self.save()` (`supervisor.rs:868`) — a full json rewrite after every
   turn. With the turn-commit barrier this buys nothing: the log already
   holds the turn durably, and resume prefers the log anyway. It costs a
   full serialize+write per turn and widens every crash window.
2. **Second writer.** `append_child_to_parent` (`subagent.rs:81`) runs in
   the parent's spawn-consumer task and does load-mutate-save on the
   **parent's** json while the parent supervisor may save concurrently —
   a lost-update race ( whichever writer runs last erases the other's
   meta changes).
3. **Redundant call-site saves.** `Runner::new_session` / `switch_to`
   (`runner.rs:314,324`) and repl `/compact`, `/new` (`repl.rs:870,1346`)
   call `save()` immediately after `finish()` — which already saves.
4. **Unglamorous close.** `fanout_task.abort()` (`service.rs:255`,
   `subagent.rs:203`) terminates the fanout loop with the channel buffer
   un-drained (trailing UI events dropped), and the graceful-drain path
   (senders dropped → `recv() == None`) never calls a final `commit()`:
   `SessionEnded` rests in the OS page cache.

Audit classification of every session-json save site (2026-08-23):

| Site | Class |
|---|---|
| `supervisor.rs:616` (`create`) | keep — bootstrap |
| `supervisor.rs:779` (`resume`) | keep — closes empty-state window (Phase B) |
| `supervisor.rs:868` (`run_turn`) | **remove** (item 1) |
| `supervisor.rs:1186` (`finish`) | keep — the clean-shutdown save |
| `subagent.rs:89` (`append_child_to_parent`) | **remove** (item 2) |
| `runner.rs:314,324`; `repl.rs:870,1346` | **remove** (item 3) |
| `session_utils.rs:299` (`cleanup_stale_sessions`) | keep — owner dead by definition |
| `main.rs:1344` (`nca cancel`) | keep — external, owner dying |

Also found while auditing: the **one-shot CLI path** (`main.rs:951`) passes
`None` as the spawn-consumer's `event_tx`, so `ChildSessionSpawned` /
`ChildSessionCompleted` never reach the session log in one-shot mode —
today `append_child_to_parent`'s direct json write is the only lineage
record there. REPL (`repl.rs:146,1605`) and service (`service.rs:168`)
wire the parent's own channel correctly.

## Design

### 1. Write-discipline contract

A session's `<id>.json` is written **only by its owning `Supervisor`**,
only at:

- `create()` — bootstrap,
- `resume()` — restore-close window,
- `finish()` — clean shutdown.

External maintenance writers (`cleanup_stale_sessions`, `nca cancel`)
touch only owner-dead sessions. Everything model-visible mid-session
lives exclusively in the event log.

**Accepted consequence:** the non-cosmetic mid-session readers of json meta
are `nca attach` and `nca cancel` (`main.rs:1281-1284`, `:1326-1338`) —
and both depend only on **write-once-stable** fields (`pid`,
`socket_path`, set at `create`/`resume`, never changed mid-session), so
nothing breaks. `query_session_state` / `cleanup_stale_sessions` are
currently uncalled in the tree. What does go stale for a crashed (never-
finished) session is display meta: title, status, `updated_at`, cost
totals, lineage — frozen at the last bootstrap/shutdown write. That is
cosmetic for pickers; resume correctness is unaffected because the log
is authoritative and lineage folds from it (§3). Cost display for
crashed sessions also goes stale; `seed_cost_tracker_from_log` already
restores real totals on resume. `get_last_session_id`'s `updated_at`
fallback sort degrades for crashed sessions, but the `.last_session`
pointer (refreshed by the kept `update_last_session()` calls)
short-circuits it in the normal path.

### 2. Remove per-turn and redundant saves

- Delete `self.save()` from `run_turn_with_images`; keep
  `update_last_session()` (pointer file, not session json).
- `Runner::new_session` / `switch_to`: drop the explicit `save()` —
  `finish()` one line above already persisted.
- repl `/compact`: drop `runtime.save()` — the summary meta persists at
  the next `finish()`; the memory note is separately durable.
- repl `/new`: drop `runtime.save()` — `new_session()`'s `finish()`
  persists.
- `Runner::save()` (`runner.rs:102`) loses its last callers → remove it
  and `Supervisor`-facing plumbing that becomes dead (check `event_tx()`
  stays — §3 needs it).
- Update any comment that still describes per-turn json saves (the
  barrier-timeout comment at `supervisor.rs:884-885` says only "the
  barrier must never fail the turn" — verify no stale json-save
  references remain) and any test that asserts json contents mid-session
  (`runtime/tests/turn_commit.rs` title assertions) to assert
  post-`finish()` or read the log.

### 3. Lineage: fold from events, delete the second writer

- **Fix the one-shot gap first:** `main.rs` passes
  `runtime.event_tx()` to `spawn_subagent_consumer` (mirrors repl/service
  wiring). This makes the parent's log carry `ChildSessionSpawned` /
  `ChildSessionCompleted` in all modes — the precondition for deleting
  the direct write.
- **Delete** `append_child_to_parent` and its call site; the consumer no
  longer constructs `parent_store`.
- **Fold at resume:** in `Supervisor::resume`, next to
  `seed_cost_tracker_from_log` (`supervisor.rs:737/744` — i.e. **before**
  the resume save at `:779`, so the folded lineage persists immediately,
  not at some later `finish()`), union
  `loaded.meta.child_session_ids` with every
  `ChildSessionSpawned { child_session_id, .. }` in the read envelopes
  (json order first, log-derived ids appended, deduped). Assign to
  `sup.child_session_ids`; the resume save then persists it.
  Both are envelope-derived seeds; `core::replay` stays messages-only.
- `ChildSessionCompleted` is already on the same channel and lands in
  the log; the fold ignores it (status is display-only, not meta).

Crashed-parent case: json lineage stale (missing children), log has the
spawn events → fold recovers lineage on resume. One-shot case: after the
wiring fix, identical. Durability note: `ChildSessionSpawned` is durable
only at the next `TurnCompleted` commit or graceful drain — a parent that
spawns and crashes mid-turn can still lose the record (existing log
durability model, not a Phase C regression). For `event_tx = None`
configs (tests, embedders) **json lineage is lost** after the deletion —
today `append_child_to_parent` still wrote it; after Phase C lineage
exists only where the event stream does. The only production `None`
caller (`main.rs:951`) is fixed in this phase.

### 4. Graceful writer close

- **Fanout loop:** on `event_rx.recv() == None` (all senders dropped),
  call `writer.commit()` once, then exit. `SessionEnded` (and anything
  buffered behind it) becomes durable.
- **`spawn_child_session`** (`subagent.rs:203`): replace `f.abort()`
  with channel-close drain — snapshot `branch`, `worktree_path`, **and
  `workspace_root`** first (`switch_to_worktree` mutates
  `sup.workspace_root` at `supervisor.rs:1551`; the return reads it at
  `subagent.rs:230`), then `drop(sup)` (the child's event-channel
  senders all live inside `sup.agent` — `supervisor.rs:479/489/550`;
  `parent_forward` is the *parent's* channel, `commit_tx` is a watch
  channel — so dropping `sup` closes the channel), then a **bounded**
  `f.await` (5s timeout, matching the Phase B barrier timeout at
  `supervisor.rs:895`; on timeout `tracing::error!` + abort — liveness
  over completeness at shutdown).
- **`service.rs:254-262` — close ORDER matters (oracle P0):** the
  parent's event channel has three sender holders: the supervisor (incl.
  tool clones), `subagent_task` (clone via `event_tx` at `service.rs:168`)
  and `command_task` (clone at `service.rs:185`). Dropping only the
  supervisor does **not** close the channel — the fanout drain would
  stall to the 5s timeout on every service shutdown. Correct order:
  1. `supervisor.finish(reason).await`
  2. abort `command_task` **and** `subagent_task` (frees their
     `event_tx` clones; neither holds durable state)
  3. `drop(supervisor)`
  4. bounded `fanout_task.await`
  No deadlock: `finish()` awaits nothing the fanout produces; the
  buffered `SessionEnded` is drained before `recv() == None`.
- REPL/TUI paths already drop the supervisor on exit → drain happens
  implicitly; the new commit-on-drain covers them.

### 5. Attach / switch double-restore — verification only

Phase B's flip changed *what* resume restores from, not how: the replay
projection still **replaces** history (`select_resume_messages` returns a
fresh vector, assigned wholesale at `supervisor.rs:720`; nothing appends
into a live `agent.messages`). T28 pins that Supervisor-level structural
invariant — it is not a TUI test and does not cover the Elm
`reset_session_state` + `replay_events_to_feedback` path
(`repl.rs:1871-1873`); pre-existing transcript-display concerns on switch
(if any) are out of scope — they predate replay-authoritative resume.

## Test matrix

| # | Test | Where |
|---|------|-------|
| T22 | `run_turn` leaves the json untouched (bytes identical pre/post turn); log grows | `runtime/tests` |
| T23 | `finish()` persists full history + meta (title, lineage) to json | `runtime/tests` |
| T24 | child spawn lands `ChildSessionSpawned`(+`Completed`) in the **parent** log; parent resume folds `child_session_ids`; crashed-parent case (stale json, log wins) recovers lineage | `runtime/tests` |
| T25 | graceful close: drop supervisor → fanout drains and exits; log parses fully incl. `SessionEnded`; no torn tail | `runtime/tests` |
| T26 | buffered events are not dropped at close: N events sent, sender dropped mid-drain → all N in log | `runtime/tests` (fanout unit) |
| T27 | one-shot consumer wiring passes a real `event_tx` (assert via existing one-shot path test or a wiring unit test) | `cli` |
| T28 | `switch_to` round-trip: message counts exact on both sessions — pins the Supervisor-level replace invariant (not a TUI-layer test) | `runtime/tests` |

Existing `turn_commit.rs` title assertions move from mid-turn to
post-`finish()` (they exist to pin `current_session_state` content, which
`finish` now solely persists).

## Rollout order

1. **Lane A (runtime + cli):** §2 + §3 — supervisor.rs, subagent.rs,
   runner.rs, repl.rs, main.rs. T22–T24, T27, T28. Sequential after
   design review.
2. **Lane B (runtime):** §4 — session_utils.rs fanout, subagent.rs
   close site, service.rs close ordering. T25, T26. Runs after Lane A
   (shared subagent.rs).

Doc sync (same commit as Lane A): `docs/architecture.md` Persistence
section (json written at create/resume/finish only; lineage fold) and
AGENTS.md Sessions line.

## Non-goals

- Signal handlers (SIGTERM → graceful `finish()`). The event log already
  covers kill-mid-session; json staleness is accepted per §1.
- Folding title/status/cost into `query_session_state` from the log for
  crashed sessions (list-display freshness). Can be added later behind
  the same envelope walk as the lineage fold.
- `event_log_path()` recomputation per call (minor; unrelated).
- Compact envelope format / log rotation (Phase B non-goal, still).

## Oracle review record (2026-08-23)

Verdicts: §1 sound · §2 sound · §3 sound (two clarifications applied) ·
§4 **needs change** (fixed) · §5 sound as scoped (reframed).

Findings applied to this doc:

- **P0** service.rs: parent event channel has three sender holders
  (supervisor incl. tool clones, `subagent_task` `service.rs:168`,
  `command_task` `service.rs:185`) — abort order specified in §4.
- **P0** subagent.rs close: snapshot must include `workspace_root`
  (mutated by `switch_to_worktree` `supervisor.rs:1551`, read at
  `subagent.rs:230`).
- **P1** fold placement before the resume save (`:779`), not "next
  finish()" — §3 updated.
- **P1** `event_tx=None` configs lose json lineage after deletion
  (accurate wording, §3).
- **P2** non-cosmetic readers are `attach`/`cancel` on write-once-stable
  `pid`/`socket_path`; `query_session_state`/`cleanup_stale_sessions`
  currently uncalled — §1 rewritten.
- **P2** T28 reframed as a Supervisor-level structural pin (§5).
- **P3** barrier comment bullet verified: current comment has no
  json-save clause; bullet reduced to "verify no stale references".
- Confirmed: `append_child_to_parent` race is real (no lock, no
  temp+rename in `SessionStore::save`); lineage fold is
  order-independent (union+dedup); no deadlock in close ordering;
  §4 does not interact with the Phase B turn-commit barrier.
