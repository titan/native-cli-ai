//! Crate-wide, test-only serialization for process-environment mutation.
//!
//! The process environment is global mutable state, but `cargo test` runs a
//! crate's tests on parallel threads inside ONE process. A per-module
//! `static Mutex` therefore does not serialize anything across modules: each
//! `#[cfg(test)] mod tests` gets its own lock, so two modules that both touch
//! `DEEPSEEK_API_KEY` still race.
//!
//! That race was a real, intermittent `cargo test --workspace` failure:
//! `repl::tests` and `runner::tests` each kept their own mutex, so a
//! `runner` guard's `Drop` could `remove_var("DEEPSEEK_API_KEY")` while a
//! `repl` test was between "set the key" and `Supervisor::create` reading
//! it → `provider configuration error: missing DeepSeek API key`.
//!
//! One crate-wide lock plus one shared guard is the only correct shape. This
//! module is compiled for test builds only (`#[cfg(test)] mod test_env;` in
//! `main.rs`) — it must never leak into the shipped binary.

use std::ffi::OsString;
use std::sync::{Mutex, MutexGuard};

/// The single lock every env-mutating test in this crate must hold. Using any
/// other lock (or none) re-introduces the cross-module race described above.
static ENV_MUTEX: Mutex<()> = Mutex::new(());

/// RAII environment override: sets the requested variables on construction,
/// restores their exact previous values (including "unset") on drop, and holds
/// the crate-wide [`ENV_MUTEX`] for its whole lifetime so no other
/// env-mutating test can observe or clobber the override in between.
///
/// Hold it for as long as the test depends on the values — for the whole test
/// body, typically via `let _env = TestEnvGuard::set(&[...]);` — because the
/// lock is released at drop.
pub(crate) struct TestEnvGuard {
    previous: Vec<(String, Option<OsString>)>,
    _lock: MutexGuard<'static, ()>,
}

impl TestEnvGuard {
    /// Acquire the crate-wide env lock and apply `vars` (restored on drop).
    pub(crate) fn set(vars: &[(&str, &str)]) -> Self {
        // A poisoned lock means another test panicked while mutating the env;
        // recover rather than cascading the failure into every later test.
        let lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let mut previous = Vec::new();
        for (key, value) in vars {
            previous.push((key.to_string(), std::env::var_os(key)));
            // SAFETY: the crate-wide mutex serializes env mutation within this
            // test binary, and the pre-values are restored on drop.
            unsafe { std::env::set_var(key, value) };
        }
        Self {
            previous,
            _lock: lock,
        }
    }
}

impl Drop for TestEnvGuard {
    fn drop(&mut self) {
        for (key, value) in self.previous.drain(..) {
            // SAFETY: still holding the crate-wide env mutex.
            match value {
                Some(value) => unsafe { std::env::set_var(&key, value) },
                None => unsafe { std::env::remove_var(&key) },
            }
        }
    }
}
