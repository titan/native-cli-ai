//! Built-in OMO (oh-my-opencode-slim) specialist skills.
//!
//! These seven specialist personas are embedded into the `nca` binary at
//! compile time via `include_str!`. On startup, [`seed_builtin_skills`]
//! writes any that are missing into the user's XDG skills directory so they
//! are discovered by [`SkillCatalog`] and auto-registered as agent profiles
//! by [`register_skill_agents`].
//!
//! Users can override any of these by placing their own `SKILL.md` at
//! `~/.config/nca/skills/<name>/SKILL.md` — seeding never overwrites existing
//! files.

use crate::config::xdg_config_dir;
use std::path::Path;

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

/// Ensure built-in specialist skills exist in the user's XDG skills directory.
///
/// Writes only files that are missing — never overwrites existing ones. This
/// makes seeding idempotent and respects user customizations.
///
/// Returns the number of skills written (0 if all already present).
pub fn seed_builtin_skills() -> usize {
    let Some(config_dir) = xdg_config_dir() else {
        return 0;
    };
    let skills_root = config_dir.join("nca/skills");
    seed_into(&skills_root)
}

/// Seed built-in skills into a specific directory (testable variant).
fn seed_into(skills_root: &Path) -> usize {
    let mut written = 0;
    // Specialists first, then workflow skills. Both follow the same
    // idempotent, never-overwrite contract.
    for (name, content) in BUILTIN_SPECIALIST_SKILLS
        .iter()
        .chain(BUILTIN_WORKFLOW_SKILLS)
    {
        let dest = skills_root.join(name).join("SKILL.md");
        if dest.exists() {
            continue;
        }
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
            Ok(()) => written += 1,
            Err(e) => tracing::warn!("failed to seed builtin skill {}: {e}", dest.display()),
        }
    }
    if written > 0 {
        tracing::info!(
            "seeded {written} builtin specialist skills into {}",
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
    }

    #[test]
    fn seed_into_does_not_overwrite_existing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("skills");

        // Pre-create a custom explorer skill.
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
    fn seed_into_is_idempotent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("skills");

        assert_eq!(seed_into(&root), 9, "first run seeds all");
        assert_eq!(seed_into(&root), 0, "second run seeds nothing");
    }
}
