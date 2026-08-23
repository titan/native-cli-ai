//! NcaModel-based event loop replacing run_blocking.

use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::AtomicBool;

use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use super::feedback::TuiFeedbackMsg;
use super::model::{NcaModel, SideEffectChannels};
use super::msg::Msg;
use crate::tui::app::{ApprovalAnswer, TerminalGuard, setup_terminal};
use nca_common::event::{InteractiveQuestionPayload, QuestionSelection};

/// Parameters for initializing the NcaModel event loop.
pub(crate) struct NcaModelParams {
    pub session_id: String,
    pub model: String,
    pub agent_label: String,
    pub permission_mode: String,
    pub workspace_root: std::path::PathBuf,
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
    cancel_flag: Option<Arc<AtomicBool>>,
    active_question_id: Arc<StdMutex<Option<String>>>,
    active_question_payload: Arc<StdMutex<Option<InteractiveQuestionPayload>>>,
    active_approval_payload: Arc<StdMutex<Option<crate::tui::state::ApprovalRequest>>>,
    staged_images: Arc<StdMutex<Vec<nca_common::message::ImageAttachment>>>,
    inbox_tx: Option<tokio::sync::mpsc::Sender<nca_core::agent_driver::InboxItem>>,
    busy_flag: Arc<AtomicBool>,
    params: NcaModelParams,
) -> anyhow::Result<()> {
    // Setup terminal
    let mut terminal = setup_terminal()?;
    // RAII: restore on ANY exit path (normal, `?` error, panic-unwind). The
    // previous explicit `restore_terminal()` after the loop was bypassed by
    // `tick()?`, leaving the terminal in raw mode + bracketed paste on error.
    let _restore_guard = TerminalGuard;

    // Create NcaModel
    let mut nca_model = NcaModel::new(
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
        },
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
