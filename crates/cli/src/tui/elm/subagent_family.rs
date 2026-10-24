//! Family index for the subagent view: discovers the current session's
//! descendant sessions (direct children + nested) from the sessions dir on
//! disk.
//!
//! Pure functions over `std::fs` — the Elm thread is a blocking thread, the
//! data set is bounded (dozens of small session jsons), and the read happens
//! only on view open + child lifecycle events (design doc §P2 plumbing).
//! Unparseable files are skipped, never fatal (tolerant scan).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use nca_common::session::{ChildSessionState, SessionMeta, SessionStatus};
use serde::Deserialize;
/// One row of the subagent family list.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SubagentEntry {
    pub session_id: String,
    /// First 6-8 chars of the id, for compact display.
    pub short_id: String,
    pub parent_id: String,
    /// 0 for direct children, 1+ for nested descendants.
    pub depth: usize,
    pub state: ChildSessionState,
    /// Spawn task / reason text (from session json `spawn_reason`), if any.
    pub task: Option<String>,
    /// `<sessions_dir>/<id>.events.jsonl`.
    pub log_path: PathBuf,
    /// Known cost snapshot (from session json `estimated_cost_usd`), if > 0.
    pub cost_usd: Option<f64>,
}

/// Raw facts extracted from one session json (parse-tolerant subset).
#[derive(Debug, Clone)]
struct SessionFacts {
    id: String,
    parent_session_id: Option<String>,
    status: SessionStatus,
    spawn_reason: Option<String>,
    cost_usd: f64,
}

/// Meta-only projection of a session json: everything the family index needs,
/// nothing it doesn't. Serde skips unknown fields, so the `messages` array —
/// by far the bulk of a session json — is never allocated or parsed. On this
/// repo's real sessions dir (353 files / ~110MB) this is the difference
/// between a multi-hundred-ms `Vec<Message>` construction storm and a cheap
/// metadata pass on the Elm thread.
#[derive(Debug, Deserialize)]
struct MetaOnly {
    meta: SessionMeta,
    #[serde(default)]
    estimated_cost_usd: f64,
}

/// Extract the family-index facts from one session json string.
///
/// Tolerant by design: a json that fails to deserialize (torn write, foreign
/// file) yields `None` and the caller skips it.
fn parse_session_facts(raw: &str) -> Option<SessionFacts> {
    let state: MetaOnly = serde_json::from_str(raw).ok()?;
    Some(SessionFacts {
        id: state.meta.id,
        parent_session_id: state.meta.parent_session_id,
        status: state.meta.status,
        spawn_reason: state.meta.spawn_reason,
        cost_usd: state.estimated_cost_usd,
    })
}

/// Short id for display: first 8 chars (matches `short_session_prefix`'s 8),
/// or the whole id when shorter.
pub(crate) fn short_id(id: &str) -> String {
    id.chars().take(8).collect()
}

/// Sort key state rank: running (0) first, then everything else (1).
fn state_rank(state: ChildSessionState) -> u8 {
    if state.is_terminal() { 1 } else { 0 }
}

/// Extract the embedded timestamp from an nca session id for ordering.
///
/// Session ids are `session-<unix_micros>-<counter>` (see
/// `supervisor::generate_session_id`); the micros prefix is a monotonic-enough
/// creation ordering. Ids without a parseable middle segment sort as 0
/// (oldest) — they are foreign/test ids where ordering doesn't matter.
fn id_timestamp(id: &str) -> i64 {
    let mut parts = id.split('-');
    let _prefix = parts.next();
    parts
        .next()
        .and_then(|seg| seg.parse::<i64>().ok())
        .unwrap_or(0)
}

/// Build the family list for `root_session_id` from disk.
///
/// Scans `sessions_dir` for `*.json` (excluding `*.events.jsonl` and
/// `*.json.tmp` — the torn-write residue `SessionStore::save` guards
/// against), loads each, keeps only descendants of the root session (any
/// depth), and returns a flat list ordered: running first, then descending
/// id timestamp (newest first).
///
/// `running_override` maps child ids observed live in the parent transcript
/// (via `ChildSessionSpawned`/`ChildSessionStatusChanged` events) onto states
/// that take precedence over the session json's persisted status — the json is
/// a cache that can lag the live lifecycle (e.g. a just-spawned child whose
/// first save still says a stale state).
pub(crate) fn build_family_index(
    sessions_dir: &Path,
    root_session_id: &str,
    running_override: &HashMap<String, ChildSessionState>,
) -> Vec<SubagentEntry> {
    let mut cache = FamilyCache::default();
    build_family_index_cached(sessions_dir, root_session_id, running_override, &mut cache)
}

/// mtime-keyed per-file cache for [`build_family_index_cached`].
///
/// Rebuilds fire on every child lifecycle event while the subagent view is
/// open; without a cache each one re-reads + re-parses every session json in
/// the dir. With the cache, the dir listing is still walked (cheap — it is
/// the metadata pass we need for mtimes anyway) but a file whose mtime is
/// unchanged reuses its last facts verbatim: rebuilds cost
/// O(listing + changed files). Unparseable files cache `None` too and are
/// only retried when their mtime moves — e.g. a torn write being rewritten.
/// A file whose mtime goes backwards (clock skew, restore) is treated as
/// changed; a deleted file simply disappears from the listing and its cache
/// entry is dropped with the `retain` in [`Self::prune`].
#[derive(Debug, Default)]
pub(crate) struct FamilyCache {
    entries: HashMap<PathBuf, CacheEntry>,
}

/// One cached file: the observed mtime plus the last parse outcome.
#[derive(Debug)]
struct CacheEntry {
    mtime: SystemTime,
    facts: Option<SessionFacts>,
}

impl FamilyCache {
    /// Drop cache rows whose files vanished from the listing (keeps the map
    /// bounded to the live dir contents across a long-lived view).
    fn prune(&mut self, live: &HashMap<PathBuf, SystemTime>) {
        self.entries.retain(|k, _| live.contains_key(k));
    }

    /// Parse `path` via the cache: reuse the cached facts when `mtime` is
    /// unchanged, re-read + re-parse otherwise (updating the cache).
    fn facts_for(
        &mut self,
        path: &Path,
        mtime: SystemTime,
        raw: Option<String>,
    ) -> Option<SessionFacts> {
        match self.entries.get(path) {
            Some(e) if e.mtime == mtime => return e.facts.clone(),
            _ => {}
        }
        let facts = raw.as_deref().and_then(parse_session_facts);
        self.entries.insert(
            path.to_path_buf(),
            CacheEntry {
                mtime,
                facts: facts.clone(),
            },
        );
        facts
    }
}

/// Cache-aware [`build_family_index`]: identical semantics, but reuses the
/// per-file facts for unchanged mtimes so a rebuild while the view is open
/// costs O(dir listing + changed files) instead of O(all files re-parsed).
/// The caller (the model) owns the cache so it survives across rebuilds;
/// `build_family_index` above is the one-shot entry for tests/first use.
pub(crate) fn build_family_index_cached(
    sessions_dir: &Path,
    root_session_id: &str,
    running_override: &HashMap<String, ChildSessionState>,
    cache: &mut FamilyCache,
) -> Vec<SubagentEntry> {
    let Some(dir_entries) = std::fs::read_dir(sessions_dir).ok() else {
        cache.entries.clear();
        return Vec::new();
    };

    // Pass 1 (metadata only): session jsons + mtimes + fresh contents for
    // the files the cache says changed. `File::open` + `read_to_end` on one
    // handle avoids the double stat of metadata() + read_to_string().
    let mut live: HashMap<PathBuf, SystemTime> = HashMap::new();
    let mut fresh: HashMap<PathBuf, String> = HashMap::new();
    for entry in dir_entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        // Session jsons only: skip event logs and torn-write temp files.
        if !name.ends_with(".json") || name.ends_with(".json.tmp") {
            continue;
        }
        let Ok(mut f) = std::fs::File::open(&path) else {
            continue;
        };
        let Ok(meta) = f.metadata() else {
            continue;
        };
        let Some(mtime) = meta.modified().ok() else {
            continue;
        };
        live.insert(path.clone(), mtime);
        let cached = matches!(cache.entries.get(&path), Some(e) if e.mtime == mtime);
        if !cached {
            let mut buf = String::new();
            if std::io::Read::read_to_string(&mut f, &mut buf).is_ok() {
                fresh.insert(path, buf);
            }
            // A failed read leaves no `fresh` row: `facts_for` caches None
            // under this mtime, matching the tolerant-skip semantics.
        }
    }

    cache.prune(&live);

    // Pass 2: resolve facts (cache hit or fresh parse) for every live file.
    let mut facts: Vec<SessionFacts> = Vec::new();
    for (path, mtime) in &live {
        // Note: an mtime race between listing and open is tolerated the same
        // way as before — worst case we parse a slightly newer file content
        // under a slightly older mtime; the next rebuild self-corrects.
        let raw = fresh.get(path).cloned();
        if let Some(f) = cache.facts_for(path, *mtime, raw) {
            facts.push(f);
        }
    }

    facts_to_family(sessions_dir, root_session_id, running_override, facts)
}

/// Pure core of [`build_family_index`]: facts → family entries.
///
/// Split out so tests can drive lineage/ordering/filtering without touching
/// the filesystem.
fn facts_to_family(
    sessions_dir: &Path,
    root_session_id: &str,
    running_override: &HashMap<String, ChildSessionState>,
    facts: Vec<SessionFacts>,
) -> Vec<SubagentEntry> {
    // parent → children index.
    let by_parent: HashMap<&str, Vec<&SessionFacts>> = {
        let mut map: HashMap<&str, Vec<&SessionFacts>> = HashMap::new();
        for f in &facts {
            if let Some(p) = f.parent_session_id.as_deref() {
                map.entry(p).or_default().push(f);
            }
        }
        map
    };

    // BFS from the root, collecting descendants with depth.
    let mut out: Vec<SubagentEntry> = Vec::new();
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut queue: Vec<(&SessionFacts, usize)> = Vec::new();
    if let Some(children) = by_parent.get(root_session_id) {
        for c in children {
            queue.push((c, 0));
        }
    }
    while let Some((f, depth)) = queue.pop() {
        if !seen.insert(f.id.as_str()) {
            continue; // corrupt lineage cycle: emit each node once
        }
        let live = running_override.get(f.id.as_str());
        let state = live
            .copied()
            .unwrap_or_else(|| ChildSessionState::from_session_status(f.status.clone()));
        out.push(SubagentEntry {
            session_id: f.id.clone(),
            short_id: short_id(&f.id),
            parent_id: f.parent_session_id.clone().unwrap_or_default(),
            depth,
            state,
            task: f.spawn_reason.clone(),
            log_path: sessions_dir.join(format!("{}.events.jsonl", f.id)),
            cost_usd: (f.cost_usd > 0.0).then_some(f.cost_usd),
        });
        if let Some(grandchildren) = by_parent.get(f.id.as_str()) {
            for g in grandchildren {
                queue.push((g, depth + 1));
            }
        }
    }

    // Running first, then newest (id timestamp) first.
    out.sort_by(|a, b| {
        state_rank(a.state)
            .cmp(&state_rank(b.state))
            .then_with(|| id_timestamp(&b.session_id).cmp(&id_timestamp(&a.session_id)))
    });
    out
}

/// Live-observed child facts (from parent-transcript events).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LiveChild {
    pub state: ChildSessionState,
}

/// Merge disk-discovered family entries with live observations.
///
/// Live state always wins over the session json cache for a known id, and a
/// child observed live but not yet on disk (spawned seconds ago, first save
/// still pending) is synthesized as a depth-0 entry so it is listed
/// immediately. Only DIRECT children of the root can be synthesized — the
/// depth of an unobserved nested descendant is unknowable without its json.
/// Re-sorts with the same running-first/newest-first ordering.
pub(crate) fn merge_live_entries(
    disk: Vec<SubagentEntry>,
    root_session_id: &str,
    live: &HashMap<String, LiveChild>,
) -> Vec<SubagentEntry> {
    let mut merged: Vec<SubagentEntry> = Vec::with_capacity(disk.len() + live.len());
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for mut e in disk {
        seen.insert(e.session_id.clone());
        // Live observation beats the persisted cache (cache lag).
        if let Some(l) = live.get(e.session_id.as_str()) {
            e.state = l.state;
        }
        merged.push(e);
    }

    for (id, l) in live {
        if !seen.contains(id) {
            merged.push(SubagentEntry {
                short_id: short_id(id),
                session_id: id.clone(),
                parent_id: root_session_id.to_string(),
                depth: 0,
                state: l.state,
                task: None,
                log_path: PathBuf::new(), // not yet on disk; reader shows a notice
                cost_usd: None,
            });
        }
    }

    merged.sort_by(|a, b| {
        state_rank(a.state)
            .cmp(&state_rank(b.state))
            .then_with(|| id_timestamp(&b.session_id).cmp(&id_timestamp(&a.session_id)))
    });
    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::Write;

    fn session_json(id: &str, parent: Option<&str>, status: &str, task: Option<&str>) -> String {
        let parent_field = parent
            .map(|p| format!("\"parent_session_id\": \"{p}\","))
            .unwrap_or_default();
        let task_field = task
            .map(|t| format!("\"spawn_reason\": \"{}\",", t.replace('"', "\\\"")))
            .unwrap_or_default();
        format!(
            "{{\"meta\": {{\"id\": \"{id}\", \"created_at\": \"2026-10-24T00:00:00Z\", \
             \"updated_at\": \"2026-10-24T00:00:00Z\", \"workspace\": \"/tmp/ws\", \
             \"model\": \"deepseek-chat\", {parent_field} {task_field} \"status\": \"{status}\", \
             \"pid\": null, \"socket_path\": null}}, \"messages\": [], \
             \"total_input_tokens\": 10, \"total_output_tokens\": 5, \"estimated_cost_usd\": 0.01}}"
        )
    }

    fn write_session(dir: &Path, id: &str, json: &str) {
        let path = dir.join(format!("{id}.json"));
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(json.as_bytes()).unwrap();
    }

    fn tmpdir() -> tempfile::TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    #[test]
    fn builds_nested_lineage_with_depth() {
        let dir = tmpdir();
        // root → child-a (running) → grandchild; sibling child-b (completed).
        write_session(
            dir.path(),
            "session-100-0",
            &session_json(
                "session-100-0",
                Some("session-1-0"),
                "running",
                Some("parent task"),
            ),
        );
        write_session(
            dir.path(),
            "session-101-0",
            &session_json(
                "session-101-0",
                Some("session-100-0"),
                "completed",
                Some("nested task"),
            ),
        );
        write_session(
            dir.path(),
            "session-102-0",
            &session_json("session-102-0", Some("session-1-0"), "error", None),
        );

        let family = build_family_index(dir.path(), "session-1-0", &HashMap::new());
        assert_eq!(family.len(), 3, "grandchild included");
        let depths: Vec<(&str, usize)> = family
            .iter()
            .map(|e| (e.session_id.as_str(), e.depth))
            .collect();
        assert!(depths.contains(&("session-100-0", 0)));
        assert!(depths.contains(&("session-101-0", 1)));
        assert!(depths.contains(&("session-102-0", 0)));
    }

    #[test]
    fn filters_non_family_sessions() {
        let dir = tmpdir();
        // An unrelated tree: root2 → other-child, plus a rootless top-level.
        write_session(
            dir.path(),
            "session-200-0",
            &session_json("session-200-0", Some("session-2-0"), "running", None),
        );
        write_session(
            dir.path(),
            "session-201-0",
            &session_json("session-201-0", None, "completed", None),
        );

        let family = build_family_index(dir.path(), "session-1-0", &HashMap::new());
        assert!(
            family.is_empty(),
            "sessions of other roots and top-levels must be excluded"
        );
    }

    #[test]
    fn running_first_then_newest_timestamp_desc() {
        let dir = tmpdir();
        write_session(
            dir.path(),
            "session-100-0",
            &session_json("session-100-0", Some("session-1-0"), "completed", None),
        );
        write_session(
            dir.path(),
            "session-105-0",
            &session_json("session-105-0", Some("session-1-0"), "running", None),
        );
        write_session(
            dir.path(),
            "session-103-0",
            &session_json("session-103-0", Some("session-1-0"), "completed", None),
        );

        let family = build_family_index(dir.path(), "session-1-0", &HashMap::new());
        let ids: Vec<&str> = family.iter().map(|e| e.session_id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["session-105-0", "session-103-0", "session-100-0"],
            "running first, then newest first"
        );
        assert_eq!(family[0].state, ChildSessionState::Running);
        assert_eq!(family[1].state, ChildSessionState::Completed);
        assert_eq!(family[2].state, ChildSessionState::Completed);
    }

    #[test]
    fn skips_bad_json_and_ignores_event_logs() {
        let dir = tmpdir();
        write_session(
            dir.path(),
            "session-100-0",
            &session_json("session-100-0", Some("session-1-0"), "running", None),
        );
        // Torn/garbage json: skipped, not fatal.
        write_session(dir.path(), "session-101-0", "{ not json at all");
        // Event log and temp residue: wrong suffixes, ignored.
        std::fs::write(dir.path().join("session-100-0.events.jsonl"), "{}\n").unwrap();
        std::fs::write(dir.path().join("session-102-0.json.tmp"), "{}").unwrap();

        let family = build_family_index(dir.path(), "session-1-0", &HashMap::new());
        assert_eq!(family.len(), 1);
        assert_eq!(family[0].session_id, "session-100-0");
        assert_eq!(
            family[0].log_path,
            dir.path().join("session-100-0.events.jsonl")
        );
    }

    #[test]
    fn missing_dir_is_empty_not_error() {
        let family = build_family_index(
            Path::new("/nonexistent-nca-dir"),
            "session-1-0",
            &HashMap::new(),
        );
        assert!(family.is_empty());
    }

    #[test]
    fn live_override_beats_persisted_status() {
        let dir = tmpdir();
        // Persisted json still says running (cache lag), but the parent
        // transcript saw the terminal event.
        write_session(
            dir.path(),
            "session-100-0",
            &session_json("session-100-0", Some("session-1-0"), "running", None),
        );
        let mut live = HashMap::new();
        live.insert("session-100-0".to_string(), ChildSessionState::Failed);
        let family = build_family_index(dir.path(), "session-1-0", &live);
        assert_eq!(family[0].state, ChildSessionState::Failed);
    }

    #[test]
    fn carries_task_and_cost_fields() {
        let dir = tmpdir();
        write_session(
            dir.path(),
            "session-100-0",
            &session_json(
                "session-100-0",
                Some("session-1-0"),
                "completed",
                Some("do the thing"),
            ),
        );
        let family = build_family_index(dir.path(), "session-1-0", &HashMap::new());
        assert_eq!(family[0].task.as_deref(), Some("do the thing"));
        assert_eq!(family[0].cost_usd, Some(0.01));
        assert_eq!(family[0].short_id, "session-");
    }

    #[test]
    fn pending_live_state_keeps_running_rank() {
        // `Pending` (spawned, not yet observed running) is non-terminal, so it
        // must sort with the running group, not after finished children.
        let dir = tmpdir();
        write_session(
            dir.path(),
            "session-100-0",
            &session_json("session-100-0", Some("session-1-0"), "completed", None),
        );
        write_session(
            dir.path(),
            "session-101-0",
            &session_json("session-101-0", Some("session-1-0"), "completed", None),
        );
        let mut live = HashMap::new();
        live.insert("session-100-0".to_string(), ChildSessionState::Pending);
        let family = build_family_index(dir.path(), "session-1-0", &live);
        assert_eq!(family[0].session_id, "session-100-0");
        assert_eq!(family[0].state, ChildSessionState::Pending);
    }

    #[test]
    fn merge_live_synthesizes_unpersisted_direct_child() {
        // A child spawned seconds ago: no session json on disk yet, but the
        // parent transcript saw the spawn. It must appear as a depth-0 row.
        let dir = tmpdir();
        write_session(
            dir.path(),
            "session-100-0",
            &session_json("session-100-0", Some("session-1-0"), "completed", None),
        );
        let disk = build_family_index(dir.path(), "session-1-0", &HashMap::new());

        let mut live = HashMap::new();
        live.insert(
            "session-999-0".to_string(),
            LiveChild {
                state: ChildSessionState::Running,
            },
        );
        let merged = merge_live_entries(disk, "session-1-0", &live);
        assert_eq!(merged.len(), 2);
        // Running synthesized child ranks first.
        assert_eq!(merged[0].session_id, "session-999-0");
        assert_eq!(merged[0].depth, 0);
        assert_eq!(merged[0].state, ChildSessionState::Running);
        assert!(merged[0].log_path.as_os_str().is_empty(), "no disk log yet");
    }

    #[test]
    fn merge_live_state_beats_disk_cache_and_reorders() {
        let dir = tmpdir();
        write_session(
            dir.path(),
            "session-100-0",
            &session_json("session-100-0", Some("session-1-0"), "running", None),
        );
        write_session(
            dir.path(),
            "session-101-0",
            &session_json("session-101-0", Some("session-1-0"), "completed", None),
        );
        let disk = build_family_index(dir.path(), "session-1-0", &HashMap::new());
        assert_eq!(
            disk[0].session_id, "session-100-0",
            "disk says running first"
        );

        // Live: 100 just failed, 101 (older) still fine — 100 must drop into
        // the finished group even though it is newer.
        let mut live = HashMap::new();
        live.insert(
            "session-100-0".to_string(),
            LiveChild {
                state: ChildSessionState::Failed,
            },
        );
        let merged = merge_live_entries(disk, "session-1-0", &live);
        assert_eq!(merged[0].session_id, "session-101-0");
        assert_eq!(merged[0].state, ChildSessionState::Completed);
        assert_eq!(merged[1].state, ChildSessionState::Failed);
    }

    // ── Meta-only projection (Fix A) ─────────────────────────────

    #[test]
    fn meta_only_parse_skips_messages_but_keeps_facts() {
        // The bulk of a real session json is `messages`. The MetaOnly
        // projection must extract the family facts without needing it —
        // serde skips unknown fields, so no Vec<Message> is ever built.
        let raw = r#"{
            "meta": {
                "id": "session-100-0",
                "created_at": "2026-10-24T00:00:00Z",
                "updated_at": "2026-10-24T00:00:00Z",
                "workspace": "/tmp/ws",
                "model": "deepseek-chat",
                "parent_session_id": "session-1-0",
                "spawn_reason": "fix the bug",
                "status": "completed",
                "pid": null,
                "socket_path": null
            },
            "messages": [
                {"role": "user", "content": "huge payload that would dominate parse time"},
                {"role": "assistant", "content": "…"}
            ],
            "total_input_tokens": 123,
            "total_output_tokens": 45,
            "estimated_cost_usd": 0.42,
            "future_unknown_field": {"anything": [1, 2, 3]}
        }"#;
        let facts = parse_session_facts(raw).expect("facts parse");
        assert_eq!(facts.id, "session-100-0");
        assert_eq!(facts.parent_session_id.as_deref(), Some("session-1-0"));
        assert_eq!(facts.status, SessionStatus::Completed);
        assert_eq!(facts.spawn_reason.as_deref(), Some("fix the bug"));
        assert_eq!(facts.cost_usd, 0.42);
    }

    #[test]
    fn meta_only_missing_cost_defaults_to_zero() {
        // Old/foreign jsons without estimated_cost_usd must parse (default 0)
        // and produce a None cost entry, same as before the projection.
        let raw = r#"{
            "meta": {
                "id": "x", "created_at": "2026-10-24T00:00:00Z",
                "updated_at": "2026-10-24T00:00:00Z", "workspace": "/w",
                "model": "m", "status": "running", "pid": null, "socket_path": null
            },
            "messages": []
        }"#;
        let facts = parse_session_facts(raw).expect("parses without cost field");
        assert_eq!(facts.cost_usd, 0.0);
    }

    // ── mtime-keyed cache (Fix A) ────────────────────────────────

    #[test]
    fn cached_rebuild_skips_unchanged_files_and_picks_up_changes() {
        let dir = tmpdir();
        write_session(
            dir.path(),
            "session-100-0",
            &session_json("session-100-0", Some("session-1-0"), "running", None),
        );

        let mut cache = FamilyCache::default();
        let first =
            build_family_index_cached(dir.path(), "session-1-0", &HashMap::new(), &mut cache);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].state, ChildSessionState::Running);
        assert_eq!(cache.entries.len(), 1, "the file is cached");

        // Rebuild with the SAME bytes on disk: a bad-json rewrite would fail
        // the same way, but a GOOD json cached must not be re-derived from a
        // corrupted view — assert semantic equivalence instead: identical
        // output, cache row still present with unchanged mtime.
        let second =
            build_family_index_cached(dir.path(), "session-1-0", &HashMap::new(), &mut cache);
        assert_eq!(second, first, "cache hit returns identical entries");
        assert_eq!(cache.entries.len(), 1);

        // Now the file changes on disk (mtime moves): the rebuild must see
        // the NEW content, not the cached row.
        // Wait out the mtime granularity floor so the rewrite observably bumps it.
        std::thread::sleep(std::time::Duration::from_millis(20));
        write_session(
            dir.path(),
            "session-100-0",
            &session_json("session-100-0", Some("session-1-0"), "completed", None),
        );
        let third =
            build_family_index_cached(dir.path(), "session-1-0", &HashMap::new(), &mut cache);
        assert_eq!(third.len(), 1);
        assert_eq!(
            third[0].state,
            ChildSessionState::Completed,
            "mtime change forces a re-read: {third:?}"
        );
    }

    #[test]
    fn cache_bad_json_is_retried_only_when_mtime_changes() {
        let dir = tmpdir();
        write_session(dir.path(), "session-100-0", "{ not json");

        let mut cache = FamilyCache::default();
        let first =
            build_family_index_cached(dir.path(), "session-1-0", &HashMap::new(), &mut cache);
        assert!(first.is_empty(), "bad json is skipped, not fatal");
        assert_eq!(cache.entries.len(), 1, "the failure is cached too");

        // Same mtime: the cached None wins — a rewrite in place with the same
        // mtime cannot resurrect the file.
        let second =
            build_family_index_cached(dir.path(), "session-1-0", &HashMap::new(), &mut cache);
        assert!(second.is_empty());

        // The torn write completes (mtime moves): now it parses.
        std::thread::sleep(std::time::Duration::from_millis(20));
        write_session(
            dir.path(),
            "session-100-0",
            &session_json("session-100-0", Some("session-1-0"), "completed", None),
        );
        let third =
            build_family_index_cached(dir.path(), "session-1-0", &HashMap::new(), &mut cache);
        assert_eq!(third.len(), 1, "repaired json parses on the mtime bump");
    }

    #[test]
    fn cache_drops_rows_for_deleted_files_and_survives_dir_removal() {
        let dir = tmpdir();
        write_session(
            dir.path(),
            "session-100-0",
            &session_json("session-100-0", Some("session-1-0"), "running", None),
        );
        let mut cache = FamilyCache::default();
        let _ = build_family_index_cached(dir.path(), "session-1-0", &HashMap::new(), &mut cache);
        assert_eq!(cache.entries.len(), 1);

        std::fs::remove_file(dir.path().join("session-100-0.json")).unwrap();
        let after =
            build_family_index_cached(dir.path(), "session-1-0", &HashMap::new(), &mut cache);
        assert!(after.is_empty());
        assert!(cache.entries.is_empty(), "prune drops deleted files");

        // Missing dir: empty result, cache cleared (not a silent stale reuse).
        let mut cache2 = FamilyCache::default();
        let _ = build_family_index_cached(
            Path::new("/nonexistent-nca-dir"),
            "session-1-0",
            &HashMap::new(),
            &mut cache2,
        );
        assert!(cache2.entries.is_empty());
    }

    #[test]
    fn one_shot_and_cached_builds_agree_on_a_mixed_dir() {
        // Equivalence contract: the cached path must produce exactly what the
        // one-shot (uncached) path produces, on first call AND on a warm
        // cache, across good/bad/foreign files.
        let dir = tmpdir();
        write_session(
            dir.path(),
            "session-100-0",
            &session_json("session-100-0", Some("session-1-0"), "running", Some("a")),
        );
        write_session(
            dir.path(),
            "session-101-0",
            &session_json("session-101-0", Some("session-100-0"), "completed", None),
        );
        write_session(dir.path(), "session-102-0", "{ torn");
        std::fs::write(dir.path().join("session-103-0.events.jsonl"), "{}\n").unwrap();
        write_session(
            dir.path(),
            "session-104-0",
            &session_json("session-104-0", Some("session-9-9"), "running", None),
        );

        let uncached = build_family_index(dir.path(), "session-1-0", &HashMap::new());
        let mut cache = FamilyCache::default();
        let cold =
            build_family_index_cached(dir.path(), "session-1-0", &HashMap::new(), &mut cache);
        let warm =
            build_family_index_cached(dir.path(), "session-1-0", &HashMap::new(), &mut cache);
        assert_eq!(uncached, cold, "cold cached build matches one-shot");
        assert_eq!(uncached, warm, "warm cache build matches one-shot");
        assert_eq!(
            uncached.len(),
            2,
            "grandchild included, torn/foreign excluded"
        );
    }
}
