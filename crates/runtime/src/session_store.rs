use nca_common::event::{AgentEvent, EventEnvelope};
use nca_common::session::{SessionSnapshot, SessionState};
use std::path::{Path, PathBuf};

/// Persists and loads session state to/from disk.
pub struct SessionStore {
    sessions_dir: PathBuf,
}

impl SessionStore {
    pub fn new(sessions_dir: impl AsRef<Path>) -> Self {
        Self {
            sessions_dir: sessions_dir.as_ref().to_path_buf(),
        }
    }

    pub fn sessions_dir(&self) -> &Path {
        &self.sessions_dir
    }

    /// Persist the session json atomically: write to a sibling temp file,
    /// fsync it, rename over the target, then best-effort fsync the parent
    /// directory.
    ///
    /// A crash mid-save can therefore never leave a torn `<id>.json` — the
    /// worst case is a leftover `<id>.json.tmp`, which `list()` ignores
    /// (`.tmp` suffix) and the next save overwrites. The `events.jsonl` log
    /// remains the authoritative record; this json is a single-writer cache
    /// owned by the session's `Supervisor` (the fixed tmp name relies on
    /// that single-writer convention — no two writers may race a save of
    /// the same session).
    pub async fn save(&self, session: &SessionState) -> Result<(), SessionStoreError> {
        let path = self.sessions_dir.join(format!("{}.json", session.meta.id));
        let tmp_path = self
            .sessions_dir
            .join(format!("{}.json.tmp", session.meta.id));
        let json = serde_json::to_string_pretty(session)
            .map_err(|e| SessionStoreError::Serialize(e.to_string()))?;

        tokio::fs::create_dir_all(&self.sessions_dir)
            .await
            .map_err(|e| SessionStoreError::Io(e.to_string()))?;

        // Phase 1: durable write of the temp file. On failure the target is
        // untouched; clean the partial temp file up best-effort.
        if let Err(e) = write_and_sync(&tmp_path, json.as_bytes()).await {
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return Err(SessionStoreError::Io(e.to_string()));
        }

        // Phase 2: atomic replace. `rename(2)` on the same directory is
        // atomic — readers see either the old or the new file, never a mix.
        if let Err(e) = tokio::fs::rename(&tmp_path, &path).await {
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return Err(SessionStoreError::Io(e.to_string()));
        }

        // Phase 3: best-effort directory fsync so the rename itself is
        // durable. Failure here is not fatal — the file content is already
        // committed; only the directory entry could lag across a hard crash.
        let dir = self.sessions_dir.clone();
        match tokio::task::spawn_blocking(move || {
            std::fs::File::open(&dir).and_then(|d| d.sync_all())
        })
        .await
        {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                tracing::warn!(error = %e, dir = %self.sessions_dir.display(), "session dir fsync failed (best-effort)");
            }
            Err(e) => {
                tracing::warn!(error = %e, "session dir fsync task failed (best-effort)");
            }
        }

        Ok(())
    }

    pub async fn load(&self, session_id: &str) -> Result<SessionState, SessionStoreError> {
        let path = self.sessions_dir.join(format!("{session_id}.json"));
        let json = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| SessionStoreError::Io(e.to_string()))?;

        serde_json::from_str(&json).map_err(|e| SessionStoreError::Deserialize(e.to_string()))
    }

    pub async fn load_snapshot(
        &self,
        session_id: &str,
    ) -> Result<SessionSnapshot, SessionStoreError> {
        self.load(session_id)
            .await
            .map(|session| session.snapshot())
    }

    pub async fn list(&self) -> Result<Vec<String>, SessionStoreError> {
        let mut ids = Vec::new();
        if !self.sessions_dir.exists() {
            return Ok(ids);
        }
        let mut entries = tokio::fs::read_dir(&self.sessions_dir)
            .await
            .map_err(|e| SessionStoreError::Io(e.to_string()))?;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| SessionStoreError::Io(e.to_string()))?
        {
            if let Some(name) = entry.file_name().to_str()
                && let Some(id) = name.strip_suffix(".json")
            {
                ids.push(id.to_string());
            }
        }

        Ok(ids)
    }
}

/// Read a session's `events.jsonl` into envelopes. Tolerant of corruption:
/// unparseable or torn-tail lines are skipped, not fatal. Legacy bare
/// `AgentEvent` lines (pre-envelope) are wrapped in an envelope. A missing
/// file yields an empty vec.
pub fn read_event_log(path: &Path) -> Vec<EventEnvelope> {
    let Ok(content) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| {
            // Envelope-shaped lines are the normal case; fall back to a bare
            // AgentEvent (pre-envelope log format) and wrap it. Anything else
            // (garbage or a torn tail) is skipped, never fatal.
            serde_json::from_str::<EventEnvelope>(line)
                .ok()
                .or_else(|| {
                    serde_json::from_str::<AgentEvent>(line)
                        .ok()
                        .map(|ev| EventEnvelope::new(0, ev))
                })
        })
        .collect()
}

/// Create `path`, write `bytes`, and fsync the file before returning.
/// Used by [`SessionStore::save`] for the durable temp-file phase of its
/// atomic write.
async fn write_and_sync(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;
    let mut file = tokio::fs::File::create(path).await?;
    file.write_all(bytes).await?;
    file.sync_all().await?;
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum SessionStoreError {
    #[error("IO error: {0}")]
    Io(String),
    #[error("Serialization error: {0}")]
    Serialize(String),
    #[error("Deserialization error: {0}")]
    Deserialize(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use nca_common::message::Message;
    use nca_common::session::{SessionMeta, SessionStatus};

    fn session_state(id: &str, marker: &str) -> SessionState {
        let now = chrono::Utc::now();
        SessionState {
            meta: SessionMeta {
                id: id.into(),
                created_at: now,
                updated_at: now,
                workspace: "/tmp/ws".into(),
                model: "m".into(),
                status: SessionStatus::Running,
                pid: None,
                socket_path: None,
                worktree_path: None,
                branch: None,
                base_branch: None,
                parent_session_id: None,
                child_session_ids: Vec::new(),
                inherited_summary: None,
                spawn_reason: None,
                session_summary: None,
                session_title: None,
                orchestration: None,
                agent_name: None,
            },
            messages: vec![Message::user(marker)],
            total_input_tokens: 0,
            total_output_tokens: 0,
            estimated_cost_usd: 0.0,
        }
    }

    /// Every `*.tmp` path left in the sessions dir (should always be empty
    /// after a save, on both the success and failure paths).
    fn tmp_residue(dir: &Path) -> Vec<std::ffi::OsString> {
        std::fs::read_dir(dir)
            .expect("read sessions dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name())
            .filter(|n| n.to_string_lossy().ends_with(".tmp"))
            .collect()
    }

    #[tokio::test]
    async fn save_roundtrips_content_and_leaves_no_tmp_residue() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SessionStore::new(dir.path());

        store
            .save(&session_state("s1", "first write"))
            .await
            .expect("first save");
        // Overwrite path: the second save renames over the first atomically.
        store
            .save(&session_state("s1", "second write"))
            .await
            .expect("overwrite save");

        let loaded = store.load("s1").await.expect("reload after save");
        assert_eq!(
            loaded.messages[0].content.to_summary_text(),
            "second write",
            "the target must hold the latest content"
        );
        assert!(
            tmp_residue(dir.path()).is_empty(),
            "no .tmp residue after success path"
        );
    }

    #[tokio::test]
    async fn save_failure_before_rename_preserves_target_and_cleans_tmp() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SessionStore::new(dir.path());

        // Commit a good state first.
        store
            .save(&session_state("s2", "good content"))
            .await
            .expect("seed save");

        // Force the temp-write phase to fail: a directory at the tmp path
        // makes File::create fail with EISDIR — deterministic on Unix.
        std::fs::create_dir(dir.path().join("s2.json.tmp")).expect("block tmp path");

        let err = store
            .save(&session_state("s2", "broken content"))
            .await
            .expect_err("save must fail when the temp write fails");
        assert!(matches!(err, SessionStoreError::Io(_)), "got: {err:?}");

        // The committed target is untouched and still parses.
        let loaded = store.load("s2").await.expect("target must survive");
        assert_eq!(loaded.messages[0].content.to_summary_text(), "good content");

        // Cleanup removed our temp FILE; the injected blocker was a
        // directory (remove_file cannot delete it — expected). Remove it and
        // confirm no other tmp residue exists.
        std::fs::remove_dir(dir.path().join("s2.json.tmp")).expect("unblock");
        assert!(tmp_residue(dir.path()).is_empty());
    }

    #[tokio::test]
    async fn save_rename_failure_cleans_tmp_and_keeps_target() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SessionStore::new(dir.path());

        // rename(file, dir) fails with EISDIR — the atomic-rename phase
        // errors out while the target stays intact.
        let target = dir.path().join("s3.json");
        std::fs::create_dir(&target).expect("dir target");
        std::fs::write(target.join("keep"), "x").expect("non-empty dir");

        let err = store
            .save(&session_state("s3", "any"))
            .await
            .expect_err("save must fail when rename fails");
        assert!(matches!(err, SessionStoreError::Io(_)), "got: {err:?}");

        assert!(target.is_dir(), "target (a directory) must be untouched");
        assert!(
            tmp_residue(dir.path()).is_empty(),
            "failed rename must clean its temp file"
        );
    }
}
