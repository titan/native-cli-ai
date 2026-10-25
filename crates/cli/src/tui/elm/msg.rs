//! Messages processed by the Elm update loop.

use crossterm::event::{KeyEvent, MouseEvent};
use nca_common::event::{AgentEvent, QuestionSelection};

use crate::tui::app::TuiCmd;

/// All messages that flow through the Elm architecture update loop.
#[derive(Debug)]
pub(crate) enum Msg {
    // ── Crossterm events ──────────────────────────────────────────
    /// Keyboard input event.
    Key(KeyEvent),
    /// Mouse input event.
    Mouse(MouseEvent),
    /// Pasted text from bracketed paste.
    Paste(String),
    /// Terminal was resized.
    Resize(u16, u16),

    // ── Agent events (from bridge via channel) ─────────────────────
    /// Raw agent event from the runtime, processed by update() into state changes.
    Agent(AgentEvent),

    // ── Internal TUI signals ─────────────────────────────────────
    /// Command from TUI to external runtime (submit, branch switch, etc.).
    Cmd(TuiCmd),
    /// Direct question answer — bypasses cmd_tx because run_turn is blocked.
    QuestionAnswer(QuestionSelection),
    /// Emergency kill-all (Ctrl+X K, T2-1): batch-cancel every Running
    /// subagent NOW. Rides its own side channel (`SideEffectChannels::
    /// kill_all_tx`), never `cmd_tx` — the whole point is working while
    /// the cmd_rx loop is blocked inside a long `run_turn`.
    KillAllSubagents,
    /// Open the command palette (Ctrl+P). Local-only: mounting the popup
    /// happens directly in the model so it works while the cmd_rx loop is
    /// blocked inside `run_turn` (thinking/tool execution).
    OpenCommandPalette,
    /// Open the subagent view locally (`/jobs` typed while a turn runs —
    /// the queued TuiCmd would only be handled after the turn ends, so
    /// the Elm model intercepts the slash command and opens the modal
    /// directly, same bypass rationale as `OpenCommandPalette`).
    OpenSubagentViewLocal,
    /// Raw input for the active question — parsed and routed by the model.
    QuestionSubmit(String),
    /// Raw input for the active approval — parsed and routed by the model.
    ApprovalSubmit(String),
    /// Quick approval: approve (Ctrl+Y), deny (Enter with "n"), or always-allow (Ctrl+U).
    ApprovalQuickAnswer { approved: bool, always_allow: bool },
    /// Force a redraw (e.g. after busy indicator animation tick).
    Redraw,
    /// Request application exit.
    Quit,
}
