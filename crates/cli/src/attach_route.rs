//! Attach routing for `nca attach <session>` / `nca logs <session> -f`
//! (P1 R2: the no-hang contract).
//!
//! Attaching used to hang silently on any session whose IPC socket was
//! dead-but-present (finished children keep their socket file until the
//! runtime dir is cleaned, and child sockets were historically never even
//! wired to broadcast). The contract here is a three-branch decision:
//!
//! 1. **Terminal-state session** (`SessionStatus != Running` in the session
//!    json) — never connect; print the durable event log instead. The log
//!    is the authoritative record, so nothing is lost.
//! 2. **Connect failure or timeout** (dead/stale socket) — fall back to
//!    printing the log.
//! 3. **Connected** — stream live envelopes, but bound each silent wait:
//!    after [`SILENCE_CHECK`] with no events, re-check the session json; a
//!    now-terminal session falls back to the log, a still-running one keeps
//!    waiting (a quiet live child is legal — tool steps can be silent for
//!    minutes).
//!
//! The decision inputs are deliberately cheap and pure so the routing is
//! unit-testable without a socket (see tests at the bottom of this file).

use nca_common::session::SessionStatus;
use std::time::Duration;

/// How long a connect attempt may take before we treat the socket as dead.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

/// How long a connected stream may stay silent before we re-check the
/// session json for a terminal state.
pub const SILENCE_CHECK: Duration = Duration::from_secs(5);

/// Routing decision for an attach attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttachRoute {
    /// Stream live envelopes from the session's IPC socket.
    Stream,
    /// Print the session's event log and exit (never blocks).
    PrintLog,
}

/// Initial routing decision, made after loading the session json but before
/// touching the socket: a terminal-state session never connects (branch 1).
///
/// A session json that cannot be loaded is NOT treated as terminal — an
/// in-flight session's json exists from create time on, but the fallback
/// for a missing json is the caller's (it prints the log when there is
/// nothing to connect to anyway).
pub fn route_initial(status: Option<SessionStatus>) -> AttachRoute {
    match status {
        Some(s) if s != SessionStatus::Running => AttachRoute::PrintLog,
        _ => AttachRoute::Stream,
    }
}

/// Routing decision after a connect attempt: failure or timeout falls back
/// to the log (branch 2), a successful connect streams (branch 3).
pub fn route_after_connect(connect_ok: bool) -> AttachRoute {
    if connect_ok {
        AttachRoute::Stream
    } else {
        AttachRoute::PrintLog
    }
}

/// Routing decision when a connected stream has gone silent for
/// [`SILENCE_CHECK`]: re-read the session json; terminal ⇒ print the log
/// (the TurnCompleted envelope has already been fsynced, so the log is
/// complete), still-running ⇒ keep waiting (quiet live children are legal).
pub fn route_on_silence(status: Option<SessionStatus>) -> AttachRoute {
    match status {
        Some(s) if s != SessionStatus::Running => AttachRoute::PrintLog,
        _ => AttachRoute::Stream,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_status_routes_to_log_without_connecting() {
        for status in [
            SessionStatus::Completed,
            SessionStatus::Error,
            SessionStatus::Cancelled,
        ] {
            assert_eq!(
                route_initial(Some(status.clone())),
                AttachRoute::PrintLog,
                "terminal {status:?} must never connect"
            );
        }
    }

    #[test]
    fn running_or_unknown_status_attempts_the_stream() {
        assert_eq!(
            route_initial(Some(SessionStatus::Running)),
            AttachRoute::Stream
        );
        // Unknown (unparseable/absent json) — let the connect attempt decide.
        assert_eq!(route_initial(None), AttachRoute::Stream);
    }

    #[test]
    fn failed_or_timed_out_connect_falls_back_to_log() {
        assert_eq!(route_after_connect(false), AttachRoute::PrintLog);
    }

    #[test]
    fn successful_connect_streams() {
        assert_eq!(route_after_connect(true), AttachRoute::Stream);
    }

    #[test]
    fn silence_on_terminal_session_falls_back_to_log() {
        assert_eq!(
            route_on_silence(Some(SessionStatus::Completed)),
            AttachRoute::PrintLog
        );
        assert_eq!(
            route_on_silence(Some(SessionStatus::Cancelled)),
            AttachRoute::PrintLog
        );
    }

    #[test]
    fn silence_on_running_session_keeps_waiting() {
        // A quiet live child is legal — the stream must not give up.
        assert_eq!(
            route_on_silence(Some(SessionStatus::Running)),
            AttachRoute::Stream
        );
        // Unknown state: prefer liveness (keep waiting) over a wrong exit.
        assert_eq!(route_on_silence(None), AttachRoute::Stream);
    }
}
