use std::sync::Arc;

use nca_common::tool::{ToolCall, ToolDefinition, ToolResult};
use serde::Deserialize;

use super::{ToolCallExt, ToolExecutor};
use crate::workspace_fs::{WorkspaceFs, sandbox_error_to_tool_result};

pub struct ApplyPatchTool {
    fs: Arc<dyn WorkspaceFs>,
}

impl ApplyPatchTool {
    pub fn new(fs: Arc<dyn WorkspaceFs>) -> Self {
        Self { fs }
    }
}

#[derive(Deserialize)]
struct Params {
    path: String,
    edits: Vec<PatchEdit>,
}

#[derive(Deserialize)]
struct PatchEdit {
    old_text: String,
    new_text: String,
}

#[async_trait::async_trait]
impl ToolExecutor for ApplyPatchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            timeout_ms: None,
            name: "apply_patch".into(),
            description: "Apply one or more exact string replacements to a file".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_text": { "type": "string" },
                                "new_text": { "type": "string" }
                            },
                            "required": ["old_text", "new_text"]
                        }
                    }
                },
                "required": ["path", "edits"]
            }),
        }
    }

    async fn execute(&self, call: &ToolCall) -> ToolResult {
        let p: Params = match call.extract_params() {
            Ok(p) => p,
            Err(e) => return e,
        };

        if p.edits.is_empty() {
            return ToolResult {
                timed_out: false,
                call_id: call.id.clone(),
                success: false,
                output: String::new(),
                error: Some("edits array must not be empty".into()),
            };
        }

        // Degenerate request guard, matching sibling `edit_file`/`replace_match`:
        // without it, `"".matches("")` counts 1 on an empty file and silently
        // inserts `new_text`.
        if let Some((i, _)) = p
            .edits
            .iter()
            .enumerate()
            .find(|(_, e)| e.old_text.is_empty())
        {
            return ToolResult {
                timed_out: false,
                call_id: call.id.clone(),
                success: false,
                output: String::new(),
                error: Some(format!("edit {}: old_text must not be empty", i + 1)),
            };
        }

        let mut content = match self.fs.read_file(&p.path).await {
            Ok(c) => c,
            Err(e) => return sandbox_error_to_tool_result(&call.id, e),
        };

        let mut applied = 0_usize;
        let mut errors = Vec::new();

        for (i, edit) in p.edits.iter().enumerate() {
            let count = content.matches(&edit.old_text).count();
            match count {
                0 => errors.push(format!("edit {}: old_text not found", i + 1)),
                1 => {
                    if let Some(idx) = content.find(&edit.old_text) {
                        content.replace_range(idx..idx + edit.old_text.len(), &edit.new_text);
                        applied += 1;
                    }
                }
                n => errors.push(format!(
                    "edit {}: old_text matched {n} times (ambiguous)",
                    i + 1
                )),
            }
        }

        let error_msg = if errors.is_empty() {
            None
        } else {
            Some(errors.join("; "))
        };

        // Atomicity: a multi-edit request is all-or-nothing. Edits are staged
        // against the in-memory buffer only; the file is written (through the
        // existing write path, which preserves mode bits) exclusively when every
        // edit applied cleanly, so a failed request leaves the file
        // byte-for-byte unchanged.
        if !errors.is_empty() {
            return ToolResult {
                timed_out: false,
                call_id: call.id.clone(),
                success: false,
                output: format!(
                    "Applied {applied}/{} edits to {path}; file left unchanged",
                    p.edits.len(),
                    path = p.path
                ),
                error: error_msg,
            };
        }

        match self.fs.write_file(&p.path, &content).await {
            Ok(()) => ToolResult {
                timed_out: false,
                call_id: call.id.clone(),
                success: true,
                output: format!(
                    "Applied {applied}/{} edits to {path}",
                    p.edits.len(),
                    path = p.path
                ),
                error: None,
            },
            Err(e) => sandbox_error_to_tool_result(&call.id, e),
        }
    }
}

// ---------------------------------------------------------------------------
// Boundary-fidelity tests
// ---------------------------------------------------------------------------
//
// These translate the four bug classes that OMO 2.2.18 → 2.2.21 hardened in its
// patch codec (`src/hooks/apply-patch/{codec,prepared-changes}.ts`) onto nca's
// apply_patch contract. nca's tool is *not* a unified-diff applier: it takes a
// whole `{ path, edits: [{ old_text, new_text }] }` JSON request and performs
// exact-string replacements over a UTF-8 file. Several OMO concepts have no
// nca analogue and are skipped in place (see notes below).
//
// OMO bug classes → nca translation:
//   ① empty text        → empty `new_text` deletion / empty `old_text` guard /
//                          zero-byte truncation (no phantom blank line).
//                          NOTE: "Add File" file *creation* does not exist in
//                          nca (the path must already exist) — skipped.
//   ② rollback fidelity → on any edit failure the file must be left byte-for-byte
//                          intact (OMO snapshots raw bytes + mode and restores).
//                          NOTE: OMO mode *tracking* (move transfers mode) does
//                          not exist in nca — skipped; in-place write keeps mode.
//   ③ unterminated Add  → a patch that cannot be fully applied must not be
//                          reported as silent success with content dropped.
//                          NOTE: nca receives whole JSON, not a patch *stream*,
//                          so mid-stream truncation is not representable —
//                          translated to malformed/missing-field requests.
//   ④ hunk ordering /   → edits apply in array order; a text region referenced
//     double-consume      twice must not be silently double-applied.
//
// RED convention: genuine gaps vs the OMO contract are written as real tests
// asserting the *correct* (OMO) behaviour but marked `#[ignore]`, so the default
// `cargo test -p nca-core` run stays green and compiles, while the gap is
// reproducible with `cargo test -p nca-core -- --ignored`. Each ignored test
// carries `RED:` in its reason string.
//
// UPDATE: all three RED tests have since been resolved — multi-edit writes are
// now atomic, empty `old_text` is unconditionally rejected, and chained
// (evolving-buffer) semantics was ratified as an explicit product decision and
// pinned by a positive test. The suite currently carries zero ignored tests.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace_fs::RealFs;

    fn tool_over(dir: &tempfile::TempDir) -> ApplyPatchTool {
        let fs: Arc<dyn WorkspaceFs> = Arc::new(RealFs::new(dir.path().to_path_buf()));
        ApplyPatchTool::new(fs)
    }

    fn make_call(path: &str, edits: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "call-1".into(),
            name: "apply_patch".into(),
            input: serde_json::json!({ "path": path, "edits": edits }),
        }
    }

    async fn apply(dir: &tempfile::TempDir, rel: &str, edits: serde_json::Value) -> ToolResult {
        tool_over(dir).execute(&make_call(rel, edits)).await
    }

    fn read(dir: &tempfile::TempDir, rel: &str) -> String {
        std::fs::read_to_string(dir.path().join(rel)).unwrap()
    }

    // =====================================================================
    // ① Empty text semantics
    // =====================================================================

    /// Deleting a full line via empty `new_text` must not leave a phantom
    /// blank line (OMO codec.ts:329 — "drop only the terminator's empty
    /// element, retaining unterminated lines").
    #[tokio::test]
    async fn empty_new_text_deletes_line_without_phantom_line() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "a\nb\nc\n").unwrap();

        let r = apply(
            &dir,
            "f.txt",
            serde_json::json!([{ "old_text": "b\n", "new_text": "" }]),
        )
        .await;

        assert!(r.success, "{r:?}");
        assert_eq!(read(&dir, "f.txt"), "a\nc\n");
    }

    /// Replacing the entire file with empty text yields a genuine 0-byte file
    /// (OMO: empty Add File contents = 0 lines), not a lone newline.
    #[tokio::test]
    async fn empty_new_text_truncates_file_to_zero_bytes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "abc\n").unwrap();

        let r = apply(
            &dir,
            "f.txt",
            serde_json::json!([{ "old_text": "abc\n", "new_text": "" }]),
        )
        .await;

        assert!(r.success, "{r:?}");
        assert_eq!(read(&dir, "f.txt"), "");
    }

    /// An empty `old_text` is a degenerate request; sibling `edit_file` rejects
    /// it outright ("old_text must not be empty"), and `apply_patch` guards it
    /// unconditionally before any file I/O — regardless of file content.
    #[tokio::test]
    async fn empty_old_text_on_nonempty_file_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "abc").unwrap();

        let r = apply(
            &dir,
            "f.txt",
            serde_json::json!([{ "old_text": "", "new_text": "X" }]),
        )
        .await;

        assert!(!r.success, "{r:?}");
        assert_eq!(read(&dir, "f.txt"), "abc");
    }

    /// On an *empty* file the same empty `old_text` must also be rejected.
    /// Without an explicit guard, `"".matches("")` counts 1 on zero-byte
    /// content and silently inserts `new_text`, diverging from the non-empty
    /// case and from `edit_file`. The guard rejects it on any content, and the
    /// file is left untouched.
    #[tokio::test]
    async fn empty_old_text_on_empty_file_must_be_rejected() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "").unwrap();

        let r = apply(
            &dir,
            "f.txt",
            serde_json::json!([{ "old_text": "", "new_text": "X" }]),
        )
        .await;

        assert!(!r.success, "empty old_text should be rejected: {r:?}");
        assert_eq!(read(&dir, "f.txt"), "");
    }

    /// A missing target file is reported — nca has no "Add File" creation
    /// path, so this must fail loudly rather than silently creating a file.
    #[tokio::test]
    async fn missing_target_file_is_reported_not_created() {
        let dir = tempfile::tempdir().unwrap();

        let r = apply(
            &dir,
            "new.txt",
            serde_json::json!([{ "old_text": "a", "new_text": "b" }]),
        )
        .await;

        assert!(!r.success, "{r:?}");
        assert!(!dir.path().join("new.txt").exists());
    }

    /// An edit against an empty file finds nothing and reports it (0 matches),
    /// leaving the file empty.
    #[tokio::test]
    async fn nonempty_edit_on_empty_file_is_not_found() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "").unwrap();

        let r = apply(
            &dir,
            "f.txt",
            serde_json::json!([{ "old_text": "a", "new_text": "b" }]),
        )
        .await;

        assert!(!r.success, "{r:?}");
        assert_eq!(read(&dir, "f.txt"), "");
    }

    // =====================================================================
    // ② Rollback / fidelity
    // =====================================================================

    /// OMO snapshots every touched file (raw bytes + mode) and restores it
    /// when any change fails (prepared-changes.ts:190-231). nca stages all
    /// edits against the in-memory buffer and writes only when every edit
    /// applied cleanly: a failed multi-edit request leaves the file
    /// byte-for-byte unchanged.
    #[tokio::test]
    async fn partial_failure_must_be_atomic() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "keep\nfoo\n").unwrap();

        let r = apply(
            &dir,
            "f.txt",
            serde_json::json!([
                { "old_text": "foo", "new_text": "bar" },
                { "old_text": "missing", "new_text": "x" }
            ]),
        )
        .await;

        assert!(!r.success, "{r:?}");
        assert_eq!(
            read(&dir, "f.txt"),
            "keep\nfoo\n",
            "failed patch must not leave partial changes"
        );
    }

    /// When every edit fails, the file must be left exactly as it was.
    #[tokio::test]
    async fn all_edits_fail_leave_file_bytes_intact() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "orig\n").unwrap();

        let r = apply(
            &dir,
            "f.txt",
            serde_json::json!([{ "old_text": "nope", "new_text": "x" }]),
        )
        .await;

        assert!(!r.success, "{r:?}");
        assert_eq!(read(&dir, "f.txt"), "orig\n");
    }

    /// OMO preserves permission bits on write (prepared-changes.ts uses temp +
    /// rename with an explicit chmod). nca rewrites in place, which keeps the
    /// inode mode — assert that a successful apply does not clear mode bits.
    #[cfg(unix)]
    #[tokio::test]
    async fn successful_apply_preserves_unix_mode_bits() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("f.sh");
        std::fs::write(&file, "#!/bin/sh\necho hi\n").unwrap();
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o600)).unwrap();

        let r = apply(
            &dir,
            "f.sh",
            serde_json::json!([{ "old_text": "hi", "new_text": "bye" }]),
        )
        .await;
        assert!(r.success, "{r:?}");

        let mode = std::fs::metadata(&file).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "mode bits must survive an in-place rewrite");
    }

    /// Binary (non-UTF-8) content must be rejected cleanly and left untouched
    /// — never partially decoded/corrupted (OMO reads raw bytes: prepared-
    /// changes.ts:190-193).
    #[tokio::test]
    async fn binary_target_is_rejected_without_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let bytes: Vec<u8> = vec![0x00, 0xff, 0xfe, 0x80, 0x01, 0x7f];
        std::fs::write(dir.path().join("blob.bin"), &bytes).unwrap();

        let r = apply(
            &dir,
            "blob.bin",
            serde_json::json!([{ "old_text": "\u{fffd}", "new_text": "y" }]),
        )
        .await;

        assert!(!r.success, "invalid UTF-8 must fail loudly: {r:?}");
        assert_eq!(std::fs::read(dir.path().join("blob.bin")).unwrap(), bytes);
    }

    /// A successful edit must preserve every other byte exactly — CRLF line
    /// endings, non-ASCII text, and a missing trailing newline all survive.
    #[tokio::test]
    async fn successful_apply_preserves_unrelated_content_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let original = "line1\r\nline2\r\nunrelated \u{00e9}\u{4e2d}";
        std::fs::write(dir.path().join("f.txt"), original).unwrap();

        let r = apply(
            &dir,
            "f.txt",
            serde_json::json!([{ "old_text": "line2", "new_text": "LINE2" }]),
        )
        .await;

        assert!(r.success, "{r:?}");
        assert_eq!(
            read(&dir, "f.txt"),
            "line1\r\nLINE2\r\nunrelated \u{00e9}\u{4e2d}"
        );
    }

    // =====================================================================
    // ③ Unterminated / no-silent-success
    // =====================================================================

    /// A partially-applied request must not claim success; the error and the
    /// applied count are surfaced. (The resulting file mutation is the ② gap.)
    #[tokio::test]
    async fn partial_success_is_not_reported_as_success_and_reports_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "foo\n").unwrap();

        let r = apply(
            &dir,
            "f.txt",
            serde_json::json!([
                { "old_text": "foo", "new_text": "bar" },
                { "old_text": "missing", "new_text": "x" }
            ]),
        )
        .await;

        assert!(!r.success, "{r:?}");
        assert!(r.error.is_some(), "failure must carry an error: {r:?}");
        assert!(
            r.output.contains("Applied 1/2"),
            "applied count must be reported: {r:?}"
        );
    }

    /// A truncated/malformed edit object (missing `new_text`) is rejected at
    /// param extraction and must not touch the file — the closest nca analogue
    /// of an unterminated Add File (content received but incomplete).
    #[tokio::test]
    async fn edit_missing_new_text_field_is_rejected_before_write() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "aaa").unwrap();

        let r = apply(&dir, "f.txt", serde_json::json!([{ "old_text": "aaa" }])).await;

        assert!(!r.success, "{r:?}");
        assert_eq!(read(&dir, "f.txt"), "aaa");
    }

    /// A malformed edit object (missing `old_text`) is likewise rejected.
    #[tokio::test]
    async fn edit_missing_old_text_field_is_rejected_before_write() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "aaa").unwrap();

        let r = apply(&dir, "f.txt", serde_json::json!([{ "new_text": "bbb" }])).await;

        assert!(!r.success, "{r:?}");
        assert_eq!(read(&dir, "f.txt"), "aaa");
    }

    /// An empty `edits` array is rejected loudly rather than being a silent
    /// no-op success.
    #[tokio::test]
    async fn empty_edits_array_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "aaa").unwrap();

        let r = apply(&dir, "f.txt", serde_json::json!([])).await;

        assert!(!r.success, "{r:?}");
        assert!(
            r.error.as_deref().unwrap_or("").contains("not be empty"),
            "{r:?}"
        );
    }

    // =====================================================================
    // ④ Ordering / double-consume
    // =====================================================================

    /// Two edits targeting the same original text: the first consumes it, the
    /// second finds nothing and fails — a shared region is never silently
    /// applied twice (OMO ④). Because any failed edit aborts the whole request
    /// atomically, the file keeps its original bytes ("dup\n", not "one\n").
    #[tokio::test]
    async fn duplicate_old_text_applies_once_and_second_edit_fails() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "dup\n").unwrap();

        let r = apply(
            &dir,
            "f.txt",
            serde_json::json!([
                { "old_text": "dup", "new_text": "one" },
                { "old_text": "dup", "new_text": "two" }
            ]),
        )
        .await;

        assert!(!r.success, "second edit must fail: {r:?}");
        assert_eq!(read(&dir, "f.txt"), "dup\n");
    }

    /// Non-overlapping edits are order-independent — applying them in either
    /// array order yields the same file.
    #[tokio::test]
    async fn independent_edits_are_order_independent() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "aaa\nbbb\n").unwrap();
        std::fs::write(dir.path().join("b.txt"), "aaa\nbbb\n").unwrap();

        let forward = apply(
            &dir,
            "a.txt",
            serde_json::json!([
                { "old_text": "aaa", "new_text": "XXX" },
                { "old_text": "bbb", "new_text": "YYY" }
            ]),
        )
        .await;
        let reversed = apply(
            &dir,
            "b.txt",
            serde_json::json!([
                { "old_text": "bbb", "new_text": "YYY" },
                { "old_text": "aaa", "new_text": "XXX" }
            ]),
        )
        .await;

        assert!(
            forward.success && reversed.success,
            "{forward:?} {reversed:?}"
        );
        assert_eq!(read(&dir, "a.txt"), "XXX\nYYY\n");
        assert_eq!(read(&dir, "b.txt"), "XXX\nYYY\n");
    }

    /// Documents the current chaining model: edits are evaluated against the
    /// evolving buffer, so a later edit can consume text produced by an earlier
    /// one ("x" → "y" → "z"). This is the flip side of the RED test below.
    #[tokio::test]
    async fn sequential_edits_apply_against_evolving_buffer() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "x\n").unwrap();

        let r = apply(
            &dir,
            "f.txt",
            serde_json::json!([
                { "old_text": "x", "new_text": "y" },
                { "old_text": "y", "new_text": "z" }
            ]),
        )
        .await;

        assert!(r.success, "{r:?}");
        assert_eq!(read(&dir, "f.txt"), "z\n");
    }

    /// Pins the intentional *chained* (evolving-buffer) semantics of apply_patch:
    /// each edit is matched against the buffer as mutated by the edits before
    /// it, so a later edit may consume text that only an earlier edit
    /// introduced ("base" → "base zzz" → "base !"). This is a deliberate
    /// product decision: unlike a unified-diff applier — where every hunk is
    /// computed independently against the original content and such an edit
    /// would be reported as not-found — apply_patch edits form a chain. See
    /// `sequential_edits_apply_against_evolving_buffer` for the minimal case.
    #[tokio::test]
    async fn later_edit_may_match_text_introduced_by_earlier_edit_chained_semantics() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "base\n").unwrap();

        let r = apply(
            &dir,
            "f.txt",
            serde_json::json!([
                { "old_text": "base", "new_text": "base zzz" },
                { "old_text": "zzz", "new_text": "!" }
            ]),
        )
        .await;

        assert!(r.success, "chained edits must both apply: {r:?}");
        assert_eq!(read(&dir, "f.txt"), "base !\n");
    }
}
