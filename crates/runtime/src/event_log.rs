//! Append-only event-log writer with turn-end durability commits.
//!
//! P2 Phase B lane B (`docs/plans/p2-phase-b-design.md` §3): the fanout
//! previously opened the log with raw `OpenOptions` and restarted event ids
//! at 1 on every process start, colliding with ids already on disk. This
//! writer owns the log file, seeds `next_id` from the existing log, and adds
//! an explicit `commit` (flush + fsync) used by the fanout at
//! `TurnCompleted` so a turn's events are durable before `run_turn` returns.

use std::io;
use std::path::Path;

use nca_common::event::EventEnvelope;
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncWriteExt;

/// Appending writer for a session's `events.jsonl`.
///
/// Degrades gracefully: if the file cannot be opened, the writer becomes a
/// no-op (warn-once) instead of failing the event fanout — liveness over
/// false durability. Event ids are seeded from the existing log so ids stay
/// unique across process restarts.
pub struct EventLogWriter {
    file: Option<File>,
    next_id: u64,
    disabled_logged: bool,
}

impl EventLogWriter {
    /// Opens (creating if needed) the log at `path` and seeds the id counter
    /// from the existing content: `next_id` is `max(envelope.id) + 1`.
    /// Legacy bare-event lines are wrapped with id 0 and never raise the max.
    /// On open failure the writer is degraded (`file = None`) and all later
    /// appends are no-ops.
    pub async fn open(path: &Path) -> Self {
        if let Some(parent) = path.parent() {
            // Best-effort: the OpenOptions below surfaces real failures.
            let _ = std::fs::create_dir_all(parent);
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await;
        let existing = crate::session_store::read_event_log(path);
        let max_id = existing.iter().map(|e| e.id).max().unwrap_or(0);
        match file {
            Ok(file) => Self {
                file: Some(file),
                next_id: max_id.saturating_add(1),
                disabled_logged: false,
            },
            Err(e) => {
                tracing::warn!(
                    "event log unavailable ({}); session will not persist events",
                    e
                );
                Self {
                    file: None,
                    next_id: max_id.saturating_add(1),
                    disabled_logged: true,
                }
            }
        }
    }

    /// Returns the current event id and advances the counter by one.
    pub fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    /// Appends one envelope as a single JSON line.
    ///
    /// The line and its newline are written with ONE `write_all`: a torn
    /// tail can only ever be a partial line, which the tolerant reader
    /// (`read_event_log`) skips. Two separate writes could leave a
    /// complete-but-newline-less line that still parses as an envelope.
    /// A degraded (unopened) writer is a no-op that returns `Ok(())`
    /// after a warn-once notice.
    pub async fn append(&mut self, envelope: &EventEnvelope) -> io::Result<()> {
        let Some(file) = self.file.as_mut() else {
            if !self.disabled_logged {
                tracing::warn!("event log writer disabled; dropping event to disk");
                self.disabled_logged = true;
            }
            return Ok(());
        };
        let line = serde_json::to_string(envelope)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let mut buf = line.into_bytes();
        buf.push(b'\n');
        file.write_all(&buf).await
    }

    /// Flushes and fsyncs the log so appended events survive a crash.
    /// A degraded writer is a no-op returning `Ok(())`.
    pub async fn commit(&mut self) -> io::Result<()> {
        let Some(file) = self.file.as_mut() else {
            return Ok(());
        };
        file.flush().await?;
        file.sync_all().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nca_common::event::AgentEvent;

    #[tokio::test]
    async fn next_id_seeds_from_existing_log() {
        // T15: an existing envelope id 7 plus a legacy bare-event line (id 0
        // after wrapping) → next id 8.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("log.events.jsonl");
        let env = serde_json::to_string(&EventEnvelope::new(
            7,
            AgentEvent::TurnStarted { turn_id: 1 },
        ))
        .expect("envelope json");
        let legacy = serde_json::to_string(&AgentEvent::SessionEnded {
            reason: nca_common::event::EndReason::Completed,
        })
        .expect("legacy json");
        std::fs::write(&path, format!("{env}\n{legacy}\n")).expect("write log");

        let mut writer = EventLogWriter::open(&path).await;
        assert_eq!(writer.next_id(), 8);
    }

    #[tokio::test]
    async fn next_id_starts_at_one_for_missing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut writer = EventLogWriter::open(&dir.path().join("missing.jsonl")).await;
        assert_eq!(writer.next_id(), 1);
    }

    #[tokio::test]
    async fn append_commit_roundtrip() {
        // T15: append + commit, drop, read back two parseable envelopes with
        // ids 1 and 2.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("log.events.jsonl");
        let mut writer = EventLogWriter::open(&path).await;
        let id1 = writer.next_id();
        writer
            .append(&EventEnvelope::new(
                id1,
                AgentEvent::TurnStarted { turn_id: 1 },
            ))
            .await
            .expect("append 1");
        let id2 = writer.next_id();
        writer
            .append(&EventEnvelope::new(
                id2,
                AgentEvent::TurnCompleted {
                    turn_id: 1,
                    duration_ms: 5,
                },
            ))
            .await
            .expect("append 2");
        writer.commit().await.expect("commit");
        drop(writer);

        let envelopes = crate::session_store::read_event_log(&path);
        assert_eq!(envelopes.len(), 2);
        assert_eq!(envelopes[0].id, 1);
        assert_eq!(envelopes[1].id, 2);
        assert!(matches!(
            envelopes[1].event,
            AgentEvent::TurnCompleted { .. }
        ));
    }
}
