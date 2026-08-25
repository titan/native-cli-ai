//! Shared test utilities for env-sensitive tests.
//!
//! `std::env::set_var` / `remove_var` mutate PROCESS-GLOBAL state while the
//! default test harness runs tests on parallel threads. The lock/guard pair
//! below serializes env mutation against env-derived reads across every
//! module in this crate (the same pattern exists in `nca_common::config`
//! tests; test helpers are not shared across crates).

use std::sync::{Mutex, MutexGuard};

/// Serializes env mutation AND env-derived reads across this crate's tests.
/// Without it, an env-mutating test (e.g. pty's `EnvGuard::set(("HOME", ..))`)
/// races any parallel test that reads the same variable — observed as the
/// `from_config_grants_git_global_config_paths_but_not_home_or_ssh` flake:
/// `SandboxPolicy::from_config` derived `~/.gitconfig` from a temporarily
/// replaced `HOME`, then the test body re-read the restored one.
static ENV_TEST_MUTEX: Mutex<()> = Mutex::new(());

/// RAII guard: snapshots and sets vars on creation, restores on drop, holding
/// [`ENV_TEST_MUTEX`] for its whole lifetime. Bind it BEFORE any code that
/// observes the environment. Use for tests that MUTATE env vars.
pub(crate) struct EnvGuard {
    previous: Vec<(String, Option<std::ffi::OsString>)>,
    _lock: MutexGuard<'static, ()>,
}

impl EnvGuard {
    pub(crate) fn set(vars: &[(&str, Option<&str>)]) -> Self {
        // Block until we exclusively own the process environment.
        let lock = ENV_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let mut previous = Vec::new();
        for (key, value) in vars {
            previous.push((key.to_string(), std::env::var_os(key)));
            match value {
                // SAFETY: the mutex above serializes env mutation within the
                // tests of this crate.
                Some(value) => unsafe { std::env::set_var(key, value) },
                None => unsafe { std::env::remove_var(key) },
            }
        }
        Self {
            previous,
            _lock: lock,
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, value) in self.previous.drain(..) {
            match value {
                // SAFETY: still holding the env mutex.
                Some(value) => unsafe { std::env::set_var(&key, value) },
                None => unsafe { std::env::remove_var(&key) },
            }
        }
    }
}

/// Hold [`ENV_TEST_MUTEX`] without mutating anything: for tests that only
/// READ env-derived state (e.g. `SandboxPolicy::from_config` reading
/// `HOME`/`XDG_*`) and must not overlap an env-mutating test.
///
/// Do NOT combine with [`EnvGuard`] in the same test — that would deadlock.
pub(crate) fn env_read_lock() -> MutexGuard<'static, ()> {
    ENV_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner())
}
