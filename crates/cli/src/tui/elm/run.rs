//! NcaModel-based event loop replacing run_blocking.

use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::AtomicBool;

use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use super::feedback::TuiFeedbackMsg;
use super::model::{NcaModel, SideEffectChannels};
use super::msg::Msg;
use crate::tui::app::{ApprovalAnswer, TerminalGuard, restore_terminal, setup_terminal};
use nca_common::event::{InteractiveQuestionPayload, QuestionSelection};

/// Install SIGTERM/SIGHUP handlers that restore the terminal before exit.
///
/// A hard `kill` of the TUI process previously leaked the alternate screen
/// and mouse-tracking sequences (`?1000/?1002/?1006`): the shell is back in
/// cooked+echo mode and echoes raw SGR mouse reports (`^[[<34;98;32…`) on
/// every move until `reset`. `setup_terminal` has already run by the time
/// this is called, so restoring here is symmetric. One task per signal —
/// deliberately simple; SIGKILL remains un-interceptable (known residual).
/// Runs on the async runtime from the blocking TUI thread: requires an
/// active tokio runtime handle, callers on a bare thread must not use this.
fn install_terminal_signal_handlers() {
    for (kind, exit_code) in [
        (
            tokio::signal::unix::SignalKind::terminate(),
            130, // SIGINT-conventional 128+2-style interrupt exit
        ),
        (tokio::signal::unix::SignalKind::hangup(), 129), // 128+SIGHUP(1)
    ] {
        let mut handle = match tokio::signal::unix::signal(kind) {
            Ok(handle) => handle,
            Err(err) => {
                tracing::warn!(
                    ?kind,
                    "failed to install terminal-restore signal handler: {err}"
                );
                continue;
            }
        };
        tokio::spawn(async move {
            handle.recv().await;
            restore_terminal();
            std::process::exit(exit_code);
        });
    }
}

/// Parameters for initializing the NcaModel event loop.
pub(crate) struct NcaModelParams {
    pub session_id: String,
    pub model: String,
    pub agent_label: String,
    pub permission_mode: String,
    pub workspace_root: std::path::PathBuf,
    /// Sessions dir for the subagent view's disk-side family scan (P2):
    /// `<workspace_root>/<config.session.history_dir>`.
    pub sessions_dir: std::path::PathBuf,
    pub skill_dirs: Vec<std::path::PathBuf>,
    pub plugin_commands: Vec<(String, Vec<String>)>,
}

/// Run the NcaModel-based TUI event loop.
///
/// This is the replacement for `app.rs::run_blocking()`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_nca_model(
    feedback_rx: tokio::sync::mpsc::UnboundedReceiver<TuiFeedbackMsg>,
    cmd_tx: tokio::sync::mpsc::UnboundedSender<Msg>,
    question_answer_tx: Option<
        tokio::sync::mpsc::UnboundedSender<(String, nca_common::event::QuestionSelection)>,
    >,
    approval_answer_tx: Option<tokio::sync::mpsc::UnboundedSender<ApprovalAnswer>>,
    // B3: refreshable slots (not plain handles) — the session wiring rewrites
    // them after every supervisor swap (`/new`, session switch), so cancels,
    // steering and kill-all always target the CURRENT session.
    cancel_flag: Arc<StdMutex<Option<Arc<AtomicBool>>>>,
    active_question_id: Arc<StdMutex<Option<String>>>,
    active_question_payload: Arc<StdMutex<Option<InteractiveQuestionPayload>>>,
    active_approval_payload: Arc<StdMutex<Option<crate::tui::state::ApprovalRequest>>>,
    staged_images: Arc<StdMutex<Vec<nca_common::message::ImageAttachment>>>,
    inbox_tx: Arc<StdMutex<Option<tokio::sync::mpsc::Sender<nca_core::agent_driver::InboxItem>>>>,
    busy_flag: Arc<AtomicBool>,
    kill_all_tx: Option<tokio::sync::mpsc::UnboundedSender<()>>,
    params: NcaModelParams,
) -> anyhow::Result<()> {
    // Setup terminal
    let mut terminal = setup_terminal()?;
    // RAII: restore on ANY exit path (normal, `?` error, panic-unwind). The
    // previous explicit `restore_terminal()` after the loop was bypassed by
    // `tick()?`, leaving the terminal in raw mode + bracketed paste on error.
    let _restore_guard = TerminalGuard;
    // SIGTERM/SIGHUP: the RAII guard cannot cover external kills — restore
    // the terminal from a signal handler before exiting (mouse-tracking
    // leak). Best-effort: registration failure is logged, never fatal.
    install_terminal_signal_handlers();

    // Create NcaModel
    let mut nca_model = NcaModel::new_with_session(
        feedback_rx,
        cmd_tx,
        SideEffectChannels {
            question_answer_tx,
            approval_answer_tx,
            cancel_flag,
            active_question_id,
            active_question_payload,
            active_approval_payload,
            staged_images,
            inbox_tx,
            busy_flag,
            kill_all_tx,
        },
        params.sessions_dir,
        params.workspace_root.clone(),
        params.session_id.clone(),
    );

    // Initialize composer with slash entries and workspace files
    nca_model
        .components
        .composer
        .state_mut()
        .load_slash_entries(
            &params.workspace_root,
            &params.skill_dirs,
            &params.plugin_commands,
        );
    nca_model
        .components
        .composer
        .state_mut()
        .load_workspace_files(&params.workspace_root);

    // Initialize status bar
    nca_model
        .components
        .status_bar
        .update_session(&params.session_id, &params.model);
    nca_model
        .components
        .status_bar
        .update_agent_profile(&params.agent_label);
    nca_model
        .components
        .status_bar
        .update_permission_mode(&params.permission_mode);

    // Set version and workspace dir on status bar
    let dir_name = params
        .workspace_root
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| params.workspace_root.display().to_string());
    nca_model
        .components
        .status_bar
        .update_version(env!("CARGO_PKG_VERSION"));
    nca_model
        .components
        .status_bar
        .update_workspace_dir(&dir_name);

    // Main event loop
    loop {
        nca_model.tick(&mut terminal)?;
        if nca_model.quit {
            break;
        }
    }

    Ok(())
}
