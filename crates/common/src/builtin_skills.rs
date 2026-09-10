//! Built-in OMO (oh-my-opencode-slim) specialist skills.
//!
//! These seven specialist personas are embedded into the `nca` binary at
//! compile time via `include_str!`. On startup, [`seed_builtin_skills`]
//! writes any that are missing into the user's XDG skills directory so they
//! are discovered by [`SkillCatalog`] and auto-registered as agent profiles
//! by [`register_skill_agents`].
//!
//! Seeding has a force-update contract, tracked by a seed-state manifest at
//! `<skills_root>/.seed-state.json` that records the SHA-256 of the content
//! `nca` last seeded for each skill:
//!
//! - missing files are written;
//! - files that are still byte-identical to what we last seeded are refreshed
//!   in place when the embedded copy has changed (upgrades reach existing
//!   installs without churn when nothing changed);
//! - files with no manifest entry, or whose bytes no longer match the recorded
//!   hash, are treated as user-modified or foreign and are never touched.
//!
//! Users can therefore still override any of these by editing or replacing
//! `~/.config/nca/skills/<name>/SKILL.md`.

use crate::config::xdg_config_dir;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

/// Name of the seed-state sidecar stored in the skills root, next to the
/// skill folders. Not a `SKILL.md`, so skill discovery ignores it.
const SEED_STATE_FILE: &str = ".seed-state.json";

/// `(name, embedded SKILL.md content)` pairs for the OMO specialists.
///
/// `tester` is deliberately split from `fixer`: Fixer writes production code,
/// Tester writes tests — and they should be configured with different models
/// for cross-model verification.
pub const BUILTIN_SPECIALIST_SKILLS: &[(&str, &str)] = &[
    ("explorer", include_str!("../skills/explorer/SKILL.md")),
    ("oracle", include_str!("../skills/oracle/SKILL.md")),
    ("librarian", include_str!("../skills/librarian/SKILL.md")),
    ("designer", include_str!("../skills/designer/SKILL.md")),
    ("fixer", include_str!("../skills/fixer/SKILL.md")),
    ("tester", include_str!("../skills/tester/SKILL.md")),
    ("observer", include_str!("../skills/observer/SKILL.md")),
    ("council", include_str!("../skills/council/SKILL.md")),
];

/// `(name, embedded SKILL.md content)` pairs for built-in workflow skills.
///
/// Unlike [`BUILTIN_SPECIALIST_SKILLS`], these are **not** agent personas —
/// they are methodology prompts (e.g. evidence-path planning) discovered as
/// normal skills but never registered as agent profiles, because they lack
/// `agent: true` and are not in the specialist-name list.
pub const BUILTIN_WORKFLOW_SKILLS: &[(&str, &str)] = &[(
    "verification-planning",
    include_str!("../skills/verification-planning/SKILL.md"),
)];

/// Ensure built-in skills exist (and stay current) in the user's XDG skills dir.
///
/// Writes files that are missing and refreshes files that are byte-identical
/// to our last seed when the embedded copy has changed. Files that look
/// user-modified or foreign (no seed-state entry, or hash mismatch) are never
/// overwritten. See the [module docs](self) for the full contract.
///
/// Returns the number of skills written, counting both newly-seeded and
/// refreshed files (0 if nothing changed).
pub fn seed_builtin_skills() -> usize {
    let Some(config_dir) = xdg_config_dir() else {
        return 0;
    };
    let skills_root = config_dir.join("nca/skills");
    seed_into(&skills_root)
}

/// Seed-state manifest: skill name → hex SHA-256 of the content last seeded.
type SeedState = BTreeMap<String, String>;

/// Hex-encoded SHA-256 digest of `bytes`.
fn content_hash(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// Load the seed-state manifest; an unreadable or malformed file is treated
/// as "no state" (existing files are then never overwritten — the safe
/// direction) with a warning.
fn load_seed_state(path: &Path) -> SeedState {
    let Ok(text) = std::fs::read_to_string(path) else {
        return SeedState::new();
    };
    match serde_json::from_str(&text) {
        Ok(state) => state,
        Err(e) => {
            tracing::warn!("ignoring malformed seed state {}: {e}", path.display());
            SeedState::new()
        }
    }
}

/// Persist the seed-state manifest. Failures only warn: seeding itself must
/// not fail because bookkeeping could not be written.
fn save_seed_state(path: &Path, state: &SeedState) {
    match serde_json::to_string_pretty(state) {
        Ok(text) => {
            if let Err(e) = std::fs::write(path, text) {
                tracing::warn!("failed to write seed state {}: {e}", path.display());
            }
        }
        Err(e) => tracing::warn!("failed to serialize seed state: {e}"),
    }
}

/// Seed built-in skills into a specific directory (testable variant).
///
/// Counts both newly-written and refreshed files as "written"; see
/// [`seed_builtin_skills`] for the full contract.
fn seed_into(skills_root: &Path) -> usize {
    let state_path = skills_root.join(SEED_STATE_FILE);
    let mut state = load_seed_state(&state_path);
    let mut written = 0;
    let mut state_dirty = false;

    // Specialists first, then workflow skills. Both follow the same
    // seed / refresh / never-touch-user-edits contract.
    for (name, content) in BUILTIN_SPECIALIST_SKILLS
        .iter()
        .chain(BUILTIN_WORKFLOW_SKILLS)
    {
        let dest = skills_root.join(name).join("SKILL.md");
        let embedded_hash = content_hash(content.as_bytes());

        if !dest.exists() {
            if let Some(parent) = dest.parent()
                && let Err(e) = std::fs::create_dir_all(parent)
            {
                tracing::warn!(
                    "failed to create builtin skill dir {}: {e}",
                    parent.display()
                );
                continue;
            }
            match std::fs::write(&dest, content) {
                Ok(()) => {
                    written += 1;
                    state.insert((*name).to_string(), embedded_hash);
                    state_dirty = true;
                }
                Err(e) => tracing::warn!("failed to seed builtin skill {}: {e}", dest.display()),
            }
            continue;
        }

        // Dest exists: refresh only if it is byte-identical to what we last
        // seeded (the user has not customized it). No manifest entry, or a
        // hash mismatch, means user-modified or foreign — leave it alone.
        let Some(recorded) = state.get(*name) else {
            continue;
        };
        let existing = match std::fs::read(&dest) {
            Ok(bytes) => bytes,
            Err(e) => {
                tracing::warn!(
                    "failed to read existing builtin skill {}: {e}",
                    dest.display()
                );
                continue;
            }
        };
        if content_hash(&existing) != *recorded {
            continue;
        }
        if embedded_hash == *recorded {
            // Already current — no write, no churn.
            continue;
        }
        match std::fs::write(&dest, content) {
            Ok(()) => {
                written += 1;
                state.insert((*name).to_string(), embedded_hash);
                state_dirty = true;
            }
            Err(e) => tracing::warn!("failed to refresh builtin skill {}: {e}", dest.display()),
        }
    }

    if state_dirty {
        save_seed_state(&state_path, &state);
    }
    if written > 0 {
        tracing::info!(
            "seeded {written} builtin skills into {}",
            skills_root.display()
        );
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_skills_have_valid_frontmatter() {
        for (name, content) in BUILTIN_SPECIALIST_SKILLS
            .iter()
            .chain(BUILTIN_WORKFLOW_SKILLS)
        {
            assert!(
                content.starts_with("---\n"),
                "{name}: missing frontmatter delimiter"
            );
            let rest = &content[4..];
            let end = rest
                .find("\n---\n")
                .unwrap_or_else(|| panic!("{name}: missing closing frontmatter delimiter"));
            let fm = &rest[..end];
            assert!(
                fm.contains("command:"),
                "{name}: frontmatter missing command field"
            );
        }
    }

    #[test]
    fn seed_into_writes_missing_skills() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("skills");
        let n = seed_into(&root);

        // All nine should be written (8 specialists + 1 workflow skill).
        assert_eq!(n, 9, "all builtin skills should be seeded");
        for (name, content) in BUILTIN_SPECIALIST_SKILLS
            .iter()
            .chain(BUILTIN_WORKFLOW_SKILLS)
        {
            let dest = root.join(name).join("SKILL.md");
            assert!(dest.exists(), "{name}/SKILL.md should exist");
            let written = std::fs::read_to_string(&dest).unwrap();
            assert_eq!(written, *content, "{name}: content mismatch");
        }
        // A seed-state manifest is created alongside the skill folders.
        assert!(root.join(SEED_STATE_FILE).exists());
    }

    #[test]
    fn seed_into_does_not_overwrite_existing_without_manifest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("skills");

        // Pre-create a custom explorer skill with no seed-state entry.
        let explorer = root.join("explorer/SKILL.md");
        std::fs::create_dir_all(explorer.parent().unwrap()).unwrap();
        std::fs::write(&explorer, "my custom explorer").unwrap();

        let n = seed_into(&root);

        // Only eight should be written (explorer already exists; 9 - 1).
        assert_eq!(n, 8);
        // The custom explorer must be untouched.
        assert_eq!(
            std::fs::read_to_string(&explorer).unwrap(),
            "my custom explorer"
        );
    }

    #[test]
    fn seed_into_does_not_overwrite_existing_with_entryless_manifest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("skills");

        // Manifest exists but has no entry for this skill (foreign file).
        let explorer = root.join("explorer/SKILL.md");
        std::fs::create_dir_all(explorer.parent().unwrap()).unwrap();
        std::fs::write(&explorer, "my custom explorer").unwrap();
        std::fs::write(
            root.join(SEED_STATE_FILE),
            serde_json::to_string(&serde_json::json!({ "other": "00" })).unwrap(),
        )
        .unwrap();

        assert_eq!(seed_into(&root), 8);
        assert_eq!(
            std::fs::read_to_string(&explorer).unwrap(),
            "my custom explorer"
        );
    }

    #[test]
    fn seed_into_is_idempotent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("skills");

        assert_eq!(seed_into(&root), 9, "first run seeds all");
        assert_eq!(seed_into(&root), 0, "second run seeds nothing");
    }

    #[test]
    fn seed_into_updates_stale_seeded_skill() {
        // Simulate a prior nca version: everything was seeded once, then
        // explorer's file and manifest entry hold "old" content this run no
        // longer embeds.
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("skills");
        assert_eq!(seed_into(&root), 9, "first run seeds all");

        let old_content = "---\nname: Explorer\ncommand: explorer\n---\nold body\n";
        let explorer = root.join("explorer/SKILL.md");
        std::fs::write(&explorer, old_content).unwrap();
        let mut state: SeedState = serde_json::from_str(
            &std::fs::read_to_string(root.join(SEED_STATE_FILE)).unwrap(),
        )
        .unwrap();
        state.insert(
            "explorer".to_string(),
            content_hash(old_content.as_bytes()),
        );
        std::fs::write(
            root.join(SEED_STATE_FILE),
            serde_json::to_string(&state).unwrap(),
        )
        .unwrap();

        // This run ships different embedded content for explorer → refresh it.
        assert_eq!(seed_into(&root), 1, "only the stale explorer is refreshed");
        let refreshed = std::fs::read_to_string(&explorer).unwrap();
        assert_eq!(
            refreshed,
            BUILTIN_SPECIALIST_SKILLS
                .iter()
                .find(|(name, _)| *name == "explorer")
                .unwrap()
                .1
        );
    }

    #[test]
    fn seed_into_does_not_overwrite_user_modified_skill() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("skills");

        assert_eq!(seed_into(&root), 9, "first run seeds all");

        // User edits one skill after seeding.
        let fixer = root.join("fixer/SKILL.md");
        let mut modified = std::fs::read_to_string(&fixer).unwrap();
        modified.push_str("\n<!-- my local tweak -->\n");
        std::fs::write(&fixer, &modified).unwrap();

        assert_eq!(seed_into(&root), 0, "user-modified file is not touched");
        assert_eq!(std::fs::read_to_string(&fixer).unwrap(), modified);
    }

    #[test]
    fn seed_state_records_hash_after_refresh() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("skills");

        // Old seeded content + matching manifest entry, as a prior version left it.
        let old_content = "---\nname: Council\ncommand: council\n---\nold body\n";
        let council = root.join("council/SKILL.md");
        std::fs::create_dir_all(council.parent().unwrap()).unwrap();
        std::fs::write(&council, old_content).unwrap();
        std::fs::write(
            root.join(SEED_STATE_FILE),
            serde_json::to_string(&SeedState::from([(
                "council".to_string(),
                content_hash(old_content.as_bytes()),
            )]))
            .unwrap(),
        )
        .unwrap();

        seed_into(&root);

        // Manifest now records the hash of the *embedded* content we wrote.
        let state: SeedState =
            serde_json::from_str(&std::fs::read_to_string(root.join(SEED_STATE_FILE)).unwrap())
                .unwrap();
        let embedded = BUILTIN_SPECIALIST_SKILLS
            .iter()
            .find(|(name, _)| *name == "council")
            .unwrap()
            .1;
        assert_eq!(
            state.get("council").unwrap(),
            &content_hash(embedded.as_bytes())
        );

        // And a subsequent run is a no-op.
        assert_eq!(seed_into(&root), 0);
    }
}
