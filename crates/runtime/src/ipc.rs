use nca_common::event::{AgentCommand, EventEnvelope};
use std::path::PathBuf;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{broadcast, mpsc};

/// IPC server that broadcasts AgentEvents and receives AgentCommands
/// over a Unix domain socket.
pub struct IpcServer {
    socket_path: PathBuf,
}

impl IpcServer {
    pub fn new(session_id: &str) -> Self {
        let runtime_dir = std::env::var("XDG_RUNTIME_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/tmp"));
        let socket_path = runtime_dir.join("nca").join(format!("{session_id}.sock"));
        Self { socket_path }
    }

    /// Test-only constructor with an explicit socket path: the production
    /// constructor derives it from `$XDG_RUNTIME_DIR`, which tests must not
    /// mutate process-wide.
    #[cfg(test)]
    fn at_socket_path(socket_path: PathBuf) -> Self {
        Self { socket_path }
    }

    pub fn socket_path(&self) -> PathBuf {
        self.socket_path.clone()
    }

    /// Start listening for client connections.
    pub async fn start(&self) -> Result<IpcHandle, IpcError> {
        if let Some(parent) = self.socket_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|err| IpcError::ConnectionFailed(err.to_string()))?;
        }
        if self.socket_path.exists() {
            let _ = tokio::fs::remove_file(&self.socket_path).await;
        }

        let listener = UnixListener::bind(&self.socket_path)
            .map_err(|err| IpcError::ConnectionFailed(err.to_string()))?;
        let (event_tx, _) = broadcast::channel::<String>(256);
        let accept_event_tx = event_tx.clone();
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let socket_path = self.socket_path.clone();

        let accept_task = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let event_rx = accept_event_tx.subscribe();
                let command_tx = command_tx.clone();
                tokio::spawn(handle_connection(stream, event_rx, command_tx));
            }
            let _ = tokio::fs::remove_file(socket_path).await;
        });

        Ok(IpcHandle {
            socket_path: self.socket_path.clone(),
            event_tx,
            command_rx: Some(command_rx),
            accept_task: Some(accept_task),
        })
    }
}

/// Handle to a started IPC server: the event broadcast half, the command
/// receiver, and the accept loop that owns the socket file.
///
/// Dropping a handle that was never split (via [`IpcHandle::into_parts`])
/// aborts the accept loop and unlinks the socket file. That matters for
/// supervisor swaps: a session replaced by `/new` (or a session switch)
/// leaves its handle untaken, and without this the accept task plus its
/// `<session-id>.sock` file would outlive the session for the life of the
/// process — one leaked socket per `/new`. Split handles are "disarmed":
/// the bridge/command-consumer parts own the live session, so `Drop` must
/// leave the listener alone (that cleanup stays at process exit, unchanged).
pub struct IpcHandle {
    socket_path: PathBuf,
    event_tx: broadcast::Sender<String>,
    command_rx: Option<mpsc::UnboundedReceiver<AgentCommand>>,
    /// Accept-loop task; `None` once detached by [`IpcHandle::into_parts`].
    accept_task: Option<tokio::task::JoinHandle<()>>,
}

impl Drop for IpcHandle {
    fn drop(&mut self) {
        // Disarmed by `into_parts`: the split parts (event bridge + command
        // consumer) own the live session, and their cleanup is process exit.
        if self.socket_path.as_os_str().is_empty() {
            return;
        }
        if let Some(task) = self.accept_task.take() {
            task.abort();
        }
        let path = std::mem::take(&mut self.socket_path);
        // A single unlink syscall (no allocation, no disk spin-up) — the
        // async alternative would need an await point, which `Drop` cannot
        // provide. NotFound is the normal case (the accept loop already
        // unlinked on a clean shutdown).
        if let Err(err) = std::fs::remove_file(&path)
            && err.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                path = %path.display(),
                "failed to remove session socket on IPC handle drop: {err}"
            );
        }
    }
}

impl IpcHandle {
    pub fn socket_path(&self) -> &PathBuf {
        &self.socket_path
    }

    pub async fn broadcast(&self, event: &EventEnvelope) -> Result<(), IpcError> {
        let line = serde_json::to_string(event)
            .map_err(|err| IpcError::ConnectionFailed(err.to_string()))?;
        let _ = self.event_tx.send(line);
        Ok(())
    }

    pub async fn recv_command(&mut self) -> Option<AgentCommand> {
        self.command_rx.as_mut()?.recv().await
    }

    /// Split into parts for separate tasks: event broadcast and command receiver.
    ///
    /// Disarms the handle: the split parts take over the live session, so the
    /// [`Drop`] cleanup (abort accept loop + unlink socket) must not run when
    /// this value is dropped at the end of the call.
    pub fn into_parts(
        mut self,
    ) -> (
        broadcast::Sender<String>,
        mpsc::UnboundedReceiver<AgentCommand>,
    ) {
        // Detach the accept loop (dropping the handle detaches, not aborts it)
        // and mark the handle disarmed by clearing the socket path.
        drop(self.accept_task.take());
        self.socket_path = PathBuf::new();
        let command_rx = self
            .command_rx
            .take()
            .unwrap_or_else(|| mpsc::unbounded_channel::<AgentCommand>().1);
        (self.event_tx.clone(), command_rx)
    }
}

/// IPC client for connecting to a running session socket (events, approvals, shutdown).
pub struct IpcClient {
    socket_path: PathBuf,
}

impl IpcClient {
    pub fn new(socket_path: PathBuf) -> Self {
        Self { socket_path }
    }

    pub async fn connect(&self) -> Result<mpsc::Receiver<EventEnvelope>, IpcError> {
        let stream = UnixStream::connect(&self.socket_path)
            .await
            .map_err(|err| IpcError::ConnectionFailed(err.to_string()))?;
        let (tx, rx) = mpsc::channel(128);
        tokio::spawn(async move {
            let reader = BufReader::new(stream);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if let Ok(event) = serde_json::from_str::<EventEnvelope>(&line)
                    && tx.send(event).await.is_err()
                {
                    break;
                }
            }
        });
        Ok(rx)
    }

    pub async fn send_command(&self, cmd: &AgentCommand) -> Result<(), IpcError> {
        let mut stream = UnixStream::connect(&self.socket_path)
            .await
            .map_err(|err| IpcError::ConnectionFailed(err.to_string()))?;
        let line = serde_json::to_string(cmd)
            .map_err(|err| IpcError::ConnectionFailed(err.to_string()))?;
        stream
            .write_all(line.as_bytes())
            .await
            .map_err(|err| IpcError::ConnectionFailed(err.to_string()))?;
        stream
            .write_all(b"\n")
            .await
            .map_err(|err| IpcError::ConnectionFailed(err.to_string()))?;
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IpcError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
}

async fn handle_connection(
    stream: UnixStream,
    mut event_rx: broadcast::Receiver<String>,
    command_tx: mpsc::UnboundedSender<AgentCommand>,
) {
    let (reader, mut writer) = stream.into_split();
    let read_task = tokio::spawn(async move {
        let mut lines = BufReader::new(reader).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if let Ok(command) = serde_json::from_str::<AgentCommand>(&line) {
                let _ = command_tx.send(command);
            }
        }
    });

    let write_task = tokio::spawn(async move {
        while let Ok(line) = event_rx.recv().await {
            if writer.write_all(line.as_bytes()).await.is_err() {
                break;
            }
            if writer.write_all(b"\n").await.is_err() {
                break;
            }
        }
    });

    let _ = tokio::join!(read_task, write_task);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A handle that is dropped WITHOUT being split (every supervisor swapped
    /// out by `/new` leaves one behind) must abort its accept loop and unlink
    /// its socket file — otherwise each `/new` leaks one socket plus one
    /// accept task for the life of the process.
    #[tokio::test(flavor = "multi_thread")]
    async fn dropped_unsplit_handle_unlinks_socket_and_stops_accepting() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("dropped.sock");
        let server = IpcServer::at_socket_path(path.clone());
        let handle = server.start().await.expect("ipc start");
        assert!(path.exists(), "precondition: the socket is bound");

        drop(handle);

        assert!(
            !path.exists(),
            "dropping an unsplit handle must unlink {}",
            path.display()
        );
        // The accept loop is gone: nothing is listening any more.
        assert!(
            UnixStream::connect(&path).await.is_err(),
            "the accept loop must be aborted, not left listening"
        );
    }

    /// `into_parts` DISARMS the handle: the bridge + command consumer own the
    /// live session, so dropping the split parts must not tear down the
    /// listener or unlink the socket (that cleanup is process-exit territory).
    #[tokio::test(flavor = "multi_thread")]
    async fn split_handle_keeps_socket_alive() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("split.sock");
        let server = IpcServer::at_socket_path(path.clone());
        let handle = server.start().await.expect("ipc start");

        let (event_tx, command_rx) = handle.into_parts();
        assert!(
            path.exists(),
            "splitting must not unlink the live session's socket"
        );
        // Still accepting: a client can connect to the live session.
        let _client = UnixStream::connect(&path).await.expect("connect");

        drop((event_tx, command_rx));
        assert!(
            path.exists(),
            "cleanup of a split handle stays at process exit"
        );
    }
}
