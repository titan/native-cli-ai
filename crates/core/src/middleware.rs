//! Waterfall middleware layer around step requests (P4,
//! `docs/plans/p4-middleware-design.md`).
//!
//! Every step's terminal provider `chat()` call is wrapped by an explicit
//! chain of [`AgentMiddleware`]s. A middleware can observe, rewrite, or
//! short-circuit the request by wrapping `next` — tower-style, with no
//! separate rewrite variant ("rewrite" = call `next` with a modified
//! request). The chain never owns the provider (it is passed per call), so
//! `AgentLoop::replace_provider` keeps working.
//!
//! Explicitly outside the chain (`§3` of the design): keepalive pings
//! (`keepalive_ping`, not `chat`), supervisor auto-summarize side-calls,
//! `tool_pipeline`, and provider normalization (`prepare_messages_for_request`,
//! `sanitize_tool_call_pairs`) — all of which stay driver-owned.

use std::path::PathBuf;
use std::sync::Arc;

use nca_common::event::AgentEvent;
use nca_common::message::Message;
use nca_common::tool::ToolDefinition;

use crate::provider::{Provider, ProviderError, StreamChunk};

/// The canonical step request: everything the terminal provider call needs.
#[derive(Clone, Debug)]
pub struct StepRequest {
    /// Request-view messages (post prepare/sanitize/compaction). Not the
    /// canonical history — rewriting affects this request only.
    pub messages: Vec<Message>,
    /// Tool definitions exposed to the provider for this step.
    pub tools: Vec<ToolDefinition>,
    /// Model identifier the provider is asked to use.
    pub model: String,
    /// Workspace root the provider/tools resolve relative paths against.
    pub workspace_root: PathBuf,
    /// Id of the running turn (for correlation in events/telemetry).
    pub turn_id: u64,
    /// 1-based step index within the turn.
    pub step_index: u64,
    /// Informational-event handle. Middlewares MAY emit events, but MUST
    /// NOT emit `MessageRecorded` (projection is driver-owned; see design
    /// §4). Enforcement is convention-only (raw sender).
    pub event_tx: tokio::sync::mpsc::Sender<AgentEvent>,
}

/// What the chain hands back to the driver.
#[derive(Debug)]
pub enum StepReply {
    /// The provider was called (possibly after middleware-internal
    /// retries): here is its stream.
    Stream(tokio::sync::mpsc::Receiver<StreamChunk>),
    /// Skip the provider entirely; this text is the step's final answer.
    FinalText(String),
}

/// A step-request middleware: wraps the rest of the chain plus the terminal
/// provider call. First registered middleware runs outermost.
#[async_trait::async_trait]
pub trait AgentMiddleware: Send + Sync {
    /// Stable middleware name (diagnostics/telemetry).
    fn name(&self) -> &str;
    /// Handle the request, then either proceed (`next.run(req)`), rewrite
    /// (`next.run(modified_req)`), or short-circuit (return
    /// [`StepReply::FinalText`] without calling `next`).
    async fn call(&self, req: StepRequest, next: Next<'_>) -> Result<StepReply, ProviderError>;
}

/// The rest of the chain plus the terminal provider. Zero-allocation:
/// borrows the chain slice; `run` consumes it. `Clone` is cheap (two
/// borrows) and powers retry-shaped middlewares, which re-run the rest of
/// the chain after a caught `ProviderError`.
#[derive(Clone)]
pub struct Next<'a> {
    /// Remaining middlewares (head is the next to run).
    middlewares: &'a [Arc<dyn AgentMiddleware>],
    /// Terminal provider; passed per call so the chain stays
    /// provider-agnostic and runtime provider replacement keeps working.
    provider: &'a Arc<dyn Provider>,
}

impl<'a> Next<'a> {
    /// Run the remaining chain against `req`. Empty slice = terminal
    /// `provider.chat(...)` wrapped in [`StepReply::Stream`].
    ///
    /// Plain `async fn` (NOT `#[async_trait]`) — it must escape the boxing
    /// machinery so the trait method's `Next<'_>` lifetime desugars
    /// correctly under async-trait.
    pub async fn run(self, req: StepRequest) -> Result<StepReply, ProviderError> {
        match self.middlewares.split_first() {
            Some((head, tail)) => {
                head.call(
                    req,
                    Next {
                        middlewares: tail,
                        provider: self.provider,
                    },
                )
                .await
            }
            None => {
                let stream = self
                    .provider
                    .chat(&req.messages, &req.tools, &req.model, &req.workspace_root)
                    .await?;
                Ok(StepReply::Stream(stream))
            }
        }
    }
}

/// An ordered chain of step middlewares; index 0 is outermost.
#[derive(Default)]
pub struct MiddlewareChain {
    middlewares: Vec<Arc<dyn AgentMiddleware>>,
}

impl MiddlewareChain {
    /// Create an empty chain.
    pub fn new() -> Self {
        Self {
            middlewares: Vec::new(),
        }
    }

    /// Append a middleware (last pushed = innermost, wraps the provider
    /// call directly).
    pub fn push(&mut self, middleware: Arc<dyn AgentMiddleware>) {
        self.middlewares.push(middleware);
    }

    /// Whether the chain holds no middlewares.
    pub fn is_empty(&self) -> bool {
        self.middlewares.is_empty()
    }

    /// Run `req` through the chain; index 0 is outermost. Empty chain =
    /// direct terminal call, observably identical to a bare `provider.chat`
    /// (same arguments at the same point in time; `model`/`workspace_root`
    /// are materialized by value instead of by reference).
    pub async fn call(
        &self,
        provider: &Arc<dyn Provider>,
        req: StepRequest,
    ) -> Result<StepReply, ProviderError> {
        Next {
            middlewares: &self.middlewares,
            provider,
        }
        .run(req)
        .await
    }
}

/// Prune ladder (P3 design §4): first retry keeps the last 8 groups full
/// (matching `plan_context_view`'s recent window), the second is more
/// aggressive (last 2).
const PRUNE_LADDER: [usize; 2] = [crate::context_view::RECENT_GROUPS_KEEP_FULL, 2];

/// Catches context-overflow rejections at the `chat()` seam and retries with
/// a progressively pruned request view (P3,
/// `docs/plans/p3-compaction-design.md` §4). Bounded; escalates the original
/// error unchanged when pruning cannot shrink the view further. Non-overflow
/// errors pass through untouched (single attempt, no events).
///
/// Wired alone for now; as the only middleware it is trivially outermost.
/// When a fuller chain lands, recovery goes innermost-of-policy /
/// outermost-of-retry.
pub struct OverflowRecoveryMiddleware {
    /// Max prune-retry attempts per step (roadmap: at most 2).
    max_retries: u32,
}

impl Default for OverflowRecoveryMiddleware {
    fn default() -> Self {
        Self { max_retries: 2 }
    }
}

#[async_trait::async_trait]
impl AgentMiddleware for OverflowRecoveryMiddleware {
    fn name(&self) -> &str {
        "overflow-recovery"
    }

    async fn call(&self, mut req: StepRequest, next: Next<'_>) -> Result<StepReply, ProviderError> {
        let mut attempt: u32 = 0;
        loop {
            match next.clone().run(req.clone()).await {
                Ok(reply) => return Ok(reply),
                Err(e) if e.is_context_overflow() && attempt < self.max_retries => {
                    let tokens_before =
                        crate::context_view::estimate_tokens_for_slice(&req.messages);
                    let _ = req
                        .event_tx
                        .send(AgentEvent::ContextCompactionStart {
                            tokens_before,
                            reason: "overflow_prune".into(),
                        })
                        .await;
                    let pruned = crate::context_view::prune_tool_results(
                        &req.messages,
                        PRUNE_LADDER[attempt as usize],
                    );
                    let tokens_after =
                        crate::context_view::estimate_tokens_for_slice(&pruned.messages);
                    let _ = req
                        .event_tx
                        .send(AgentEvent::ContextCompactionEnd {
                            tokens_after,
                            kv_prefix_broken: pruned.pruned_groups > 0,
                        })
                        .await;
                    // No progress possible → escalate the original error unchanged.
                    if pruned.pruned_groups == 0 {
                        return Err(e);
                    }
                    req.messages = pruned.messages;
                    attempt += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::Path;
    use std::sync::Mutex;

    /// What a recorder observed: middleware name + message texts seen.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Observation {
        who: String,
        messages: Vec<String>,
    }

    /// Recording provider: counts calls, captures the full argument set.
    struct RecordingProvider {
        calls: Mutex<Vec<(Vec<String>, Vec<String>, String, PathBuf)>>,
        fail_first: bool,
    }

    impl RecordingProvider {
        fn new() -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                fail_first: false,
            }
        }
    }

    fn text_stream() -> tokio::sync::mpsc::Receiver<StreamChunk> {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        tokio::spawn(async move {
            let _ = tx.send(StreamChunk::TextDelta("ok".into())).await;
            let _ = tx.send(StreamChunk::Done).await;
        });
        rx
    }

    #[async_trait::async_trait]
    impl Provider for RecordingProvider {
        async fn chat(
            &self,
            messages: &[Message],
            tools: &[ToolDefinition],
            model: &str,
            workspace_root: &Path,
        ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError> {
            if self.fail_first && self.calls.lock().unwrap().is_empty() {
                self.calls.lock().unwrap().push((
                    messages
                        .iter()
                        .map(|m| m.content.to_summary_text())
                        .collect(),
                    tools.iter().map(|t| t.name.clone()).collect(),
                    model.to_string(),
                    workspace_root.to_path_buf(),
                ));
                return Err(ProviderError::Other("first attempt fails".into()));
            }
            self.calls.lock().unwrap().push((
                messages
                    .iter()
                    .map(|m| m.content.to_summary_text())
                    .collect(),
                tools.iter().map(|t| t.name.clone()).collect(),
                model.to_string(),
                workspace_root.to_path_buf(),
            ));
            Ok(text_stream())
        }
    }

    /// Pass-through recorder middleware.
    struct Recorder {
        name: &'static str,
        log: Arc<Mutex<Vec<Observation>>>,
    }

    #[async_trait::async_trait]
    impl AgentMiddleware for Recorder {
        fn name(&self) -> &str {
            self.name
        }

        async fn call(&self, req: StepRequest, next: Next<'_>) -> Result<StepReply, ProviderError> {
            self.log.lock().unwrap().push(Observation {
                who: self.name.into(),
                messages: req
                    .messages
                    .iter()
                    .map(|m| m.content.to_summary_text())
                    .collect(),
            });
            next.run(req).await
        }
    }

    /// Middle middleware that rewrites messages (M3).
    struct Rewriter {
        log: Arc<Mutex<Vec<Observation>>>,
    }

    #[async_trait::async_trait]
    impl AgentMiddleware for Rewriter {
        fn name(&self) -> &str {
            "rewriter"
        }

        async fn call(
            &self,
            mut req: StepRequest,
            next: Next<'_>,
        ) -> Result<StepReply, ProviderError> {
            self.log.lock().unwrap().push(Observation {
                who: "rewriter".into(),
                messages: req
                    .messages
                    .iter()
                    .map(|m| m.content.to_summary_text())
                    .collect(),
            });
            for msg in &mut req.messages {
                *msg = Message::user(format!("{} [rewritten]", msg.content.to_summary_text()));
            }
            next.run(req).await
        }
    }

    /// Short-circuiting middleware (M4): never calls `next`.
    struct ShortCircuit {
        ran: Arc<Mutex<bool>>,
        text: String,
    }

    #[async_trait::async_trait]
    impl AgentMiddleware for ShortCircuit {
        fn name(&self) -> &str {
            "short-circuit"
        }

        async fn call(
            &self,
            _req: StepRequest,
            _next: Next<'_>,
        ) -> Result<StepReply, ProviderError> {
            *self.ran.lock().unwrap() = true;
            Ok(StepReply::FinalText(self.text.clone()))
        }
    }

    /// Retry middleware (M5): retries once after a provider error, appending
    /// a marker message so the second attempt is distinguishable.
    struct RetryOnce {
        saw_error: Arc<Mutex<Option<String>>>,
    }

    #[async_trait::async_trait]
    impl AgentMiddleware for RetryOnce {
        fn name(&self) -> &str {
            "retry-once"
        }

        async fn call(&self, req: StepRequest, next: Next<'_>) -> Result<StepReply, ProviderError> {
            match next.clone().run(req.clone()).await {
                Ok(reply) => Ok(reply),
                Err(e) => {
                    *self.saw_error.lock().unwrap() = Some(e.to_string());
                    let mut retry = req;
                    retry.messages.push(Message::user("retry marker"));
                    next.run(retry).await
                }
            }
        }
    }

    fn req(messages: Vec<Message>, tools: Vec<ToolDefinition>) -> StepRequest {
        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(16);
        StepRequest {
            messages,
            tools,
            model: "test-model".into(),
            workspace_root: PathBuf::from("/tmp/nca-p4-test"),
            turn_id: 1,
            step_index: 1,
            event_tx,
        }
    }

    /// Construct directly: `parameters` is a JSON Schema object.
    fn tool_def(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.into(),
            description: format!("tool {name}"),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
            timeout_ms: None,
        }
    }

    async fn extract_text(mut rx: tokio::sync::mpsc::Receiver<StreamChunk>) -> String {
        let mut out = String::new();
        while let Some(chunk) = rx.recv().await {
            match chunk {
                StreamChunk::TextDelta(d) => out.push_str(&d),
                StreamChunk::Error(e) => panic!("stream error: {e}"),
                StreamChunk::Done => break,
                _ => {}
            }
        }
        out
    }

    fn texts(messages: &[Message]) -> Vec<String> {
        messages
            .iter()
            .map(|m| m.content.to_summary_text())
            .collect()
    }

    // M1 — empty chain: terminal called once, observes everything untouched.
    #[tokio::test]
    async fn m1_empty_chain_calls_terminal_once_with_untouched_request() {
        let provider = Arc::new(RecordingProvider::new());
        let dyn_provider: Arc<dyn Provider> = provider.clone();
        let chain = MiddlewareChain::new();
        assert!(chain.is_empty());

        let reply = chain
            .call(
                &dyn_provider,
                req(vec![Message::user("hello")], vec![tool_def("read_file")]),
            )
            .await
            .expect("empty chain must succeed");

        match reply {
            StepReply::Stream(rx) => assert_eq!(extract_text(rx).await, "ok"),
            StepReply::FinalText(t) => panic!("expected Stream, got FinalText({t})"),
        }

        let calls = provider.calls.lock().unwrap();
        assert_eq!(calls.len(), 1, "provider called exactly once");
        let (msgs, tools, model, ws) = &calls[0];
        assert_eq!(msgs, &vec!["hello".to_string()]);
        assert_eq!(tools, &vec!["read_file".to_string()]);
        assert_eq!(model, "test-model");
        assert_eq!(ws, &PathBuf::from("/tmp/nca-p4-test"));
    }

    // M2 — three recorders run in registration order outer→in→terminal.
    #[tokio::test]
    async fn m2_middlewares_execute_in_registration_order() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(RecordingProvider::new());
        let dyn_provider: Arc<dyn Provider> = provider.clone();
        let mut chain = MiddlewareChain::new();
        for name in ["outer", "middle", "inner"] {
            chain.push(Arc::new(Recorder {
                name,
                log: Arc::clone(&log),
            }));
        }
        assert!(!chain.is_empty());

        chain
            .call(&dyn_provider, req(vec![Message::user("hi")], vec![]))
            .await
            .expect("chain must succeed");

        let log = log.lock().unwrap();
        assert_eq!(
            log.iter().map(|o| o.who.as_str()).collect::<Vec<_>>(),
            vec!["outer", "middle", "inner"],
            "outer runs first, inner last"
        );
    }

    // M3 — middle rewrites; inner + provider see the rewrite, outer sees the original.
    #[tokio::test]
    async fn m3_rewrite_visible_inward_only() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(RecordingProvider::new());
        let dyn_provider: Arc<dyn Provider> = provider.clone();
        let mut chain = MiddlewareChain::new();
        chain.push(Arc::new(Recorder {
            name: "outer",
            log: Arc::clone(&log),
        }));
        chain.push(Arc::new(Rewriter {
            log: Arc::clone(&log),
        }));
        chain.push(Arc::new(Recorder {
            name: "inner",
            log: Arc::clone(&log),
        }));

        chain
            .call(&dyn_provider, req(vec![Message::user("original")], vec![]))
            .await
            .expect("chain must succeed");

        let log = log.lock().unwrap();
        let outer = &log[0];
        assert_eq!(outer.who, "outer");
        assert_eq!(outer.messages, vec!["original".to_string()]);

        let inner = &log[2];
        assert_eq!(inner.who, "inner");
        assert_eq!(inner.messages, vec!["original [rewritten]".to_string()]);

        let calls = provider.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, vec!["original [rewritten]".to_string()]);
    }

    // M4 — outer short-circuits: inner + provider never run.
    #[tokio::test]
    async fn m4_short_circuit_skips_inner_and_provider() {
        let provider = Arc::new(RecordingProvider::new());
        let dyn_provider: Arc<dyn Provider> = provider.clone();
        let ran = Arc::new(Mutex::new(false));
        let mut chain = MiddlewareChain::new();
        chain.push(Arc::new(ShortCircuit {
            ran: Arc::clone(&ran),
            text: "from middleware".into(),
        }));
        chain.push(Arc::new(Recorder {
            name: "inner",
            log: Arc::new(Mutex::new(Vec::new())),
        }));

        let reply = chain
            .call(&dyn_provider, req(vec![Message::user("hi")], vec![]))
            .await
            .expect("short-circuit must succeed");

        match reply {
            StepReply::FinalText(t) => assert_eq!(t, "from middleware"),
            StepReply::Stream(_) => panic!("expected FinalText"),
        }
        assert!(*ran.lock().unwrap(), "short-circuit middleware ran");
        assert!(
            provider.calls.lock().unwrap().is_empty(),
            "provider must not be called on short-circuit"
        );
    }

    // M5 — provider error reaches the middleware; retry succeeds on attempt 2.
    #[tokio::test]
    async fn m5_provider_error_reaches_middleware_and_retry_succeeds() {
        let provider = Arc::new(RecordingProvider {
            calls: Mutex::new(Vec::new()),
            fail_first: true,
        });
        let dyn_provider: Arc<dyn Provider> = provider.clone();
        let saw_error = Arc::new(Mutex::new(None));
        let mut chain = MiddlewareChain::new();
        chain.push(Arc::new(RetryOnce {
            saw_error: Arc::clone(&saw_error),
        }));

        let reply = chain
            .call(&dyn_provider, req(vec![Message::user("attempt")], vec![]))
            .await
            .expect("retry must succeed on second attempt");

        assert!(
            matches!(reply, StepReply::Stream(_)),
            "second attempt returns the provider stream"
        );
        let err = saw_error.lock().unwrap().clone();
        assert_eq!(
            err.as_deref(),
            Some("first attempt fails"),
            "chat()-level Err must reach the middleware"
        );
        let calls = provider.calls.lock().unwrap();
        assert_eq!(calls.len(), 2, "provider called exactly twice");
        assert_eq!(
            calls[1].0,
            vec!["attempt".to_string(), "retry marker".into()]
        );
    }

    // --- P3: OverflowRecoveryMiddleware (C4/C5) ---

    /// Scripted overflow provider: fails the first `fail_calls` calls with a
    /// context-overflow error body, then succeeds. Records every call's
    /// message texts.
    struct OverflowThenOkProvider {
        calls: Mutex<Vec<Vec<String>>>,
        fail_calls: usize,
    }

    #[async_trait::async_trait]
    impl Provider for OverflowThenOkProvider {
        async fn chat(
            &self,
            messages: &[Message],
            _tools: &[ToolDefinition],
            _model: &str,
            _workspace_root: &Path,
        ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError> {
            let seen: Vec<String> = messages
                .iter()
                .map(|m| m.content.to_summary_text())
                .collect();
            self.calls.lock().unwrap().push(seen);
            if self.calls.lock().unwrap().len() <= self.fail_calls {
                return Err(ProviderError::RequestFailed(
                    "{\"error\":{\"message\":\"This model's maximum context length is 65536 tokens. However, you requested 90124 tokens.\"}}".into(),
                ));
            }
            Ok(text_stream())
        }
    }

    fn read_call(id: &str) -> nca_common::message::MessageToolCall {
        nca_common::message::MessageToolCall {
            id: id.into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "a.rs"}),
        }
    }

    /// Fixture with prunable material: system + old compactible tool groups +
    /// a long recent tail, so rung 0 (keep last 8 groups) has something to delete.
    fn overflow_fixture() -> Vec<Message> {
        let mut messages = vec![Message::system("sys")];
        for i in 0..6 {
            messages.push(Message::assistant_with_tool_calls(
                "",
                vec![read_call(&format!("c{i}"))],
            ));
            messages.push(Message::tool(&format!("c{i}"), &format!("read output {i}")));
        }
        for i in 0..9 {
            messages.push(Message::user(format!("u{i}")));
            messages.push(Message::assistant(format!("a{i}")));
        }
        messages
    }

    // C4 — overflow once → prune → retry succeeds.
    #[tokio::test]
    async fn overflow_recovery_prunes_and_retries_successfully() {
        let provider = Arc::new(OverflowThenOkProvider {
            calls: Mutex::new(Vec::new()),
            fail_calls: 1,
        });
        let dyn_provider: Arc<dyn Provider> = provider.clone();
        let mut chain = MiddlewareChain::new();
        chain.push(Arc::new(OverflowRecoveryMiddleware::default()));

        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel::<AgentEvent>(16);
        let request = StepRequest {
            messages: overflow_fixture(),
            tools: vec![],
            model: "test-model".into(),
            workspace_root: PathBuf::from("/tmp/nca-p3-test"),
            turn_id: 1,
            step_index: 1,
            event_tx,
        };

        let reply = chain
            .call(&dyn_provider, request)
            .await
            .expect("pruned retry must succeed");
        assert!(matches!(reply, StepReply::Stream(_)));

        let calls = provider.calls.lock().unwrap();
        assert_eq!(calls.len(), 2, "provider called exactly twice");
        assert!(
            !calls[1].iter().any(|t| t.contains("read output")),
            "attempt-2 messages must lack the pruned tool groups: {:?}",
            calls[1]
        );
        assert!(calls[1].iter().any(|t| t == "sys"), "system kept");

        // Bracket events on the channel: Start(overflow_prune) then
        // End(kv_prefix_broken=true).
        let mut saw_start = false;
        let mut saw_end = false;
        while let Ok(ev) = event_rx.try_recv() {
            match ev {
                AgentEvent::ContextCompactionStart {
                    tokens_before,
                    reason,
                } => {
                    assert!(tokens_before > 0);
                    assert_eq!(reason, "overflow_prune");
                    saw_start = true;
                }
                AgentEvent::ContextCompactionEnd {
                    kv_prefix_broken, ..
                } => {
                    assert!(kv_prefix_broken);
                    saw_end = true;
                }
                _ => {}
            }
        }
        assert!(saw_start, "ContextCompactionStart emitted");
        assert!(saw_end, "ContextCompactionEnd emitted");
    }

    // C5a — nothing prunable → escalates the original overflow error after
    // one attempt (prune made no progress).
    #[tokio::test]
    async fn overflow_recovery_escalates_when_nothing_prunable() {
        let provider = Arc::new(OverflowThenOkProvider {
            calls: Mutex::new(Vec::new()),
            fail_calls: 5, // always fails
        });
        let dyn_provider: Arc<dyn Provider> = provider.clone();
        let mut chain = MiddlewareChain::new();
        chain.push(Arc::new(OverflowRecoveryMiddleware::default()));

        let (event_tx, _event_rx) = tokio::sync::mpsc::channel::<AgentEvent>(16);
        let request = StepRequest {
            // Minimal view: no tool groups → pruned_groups == 0 on rung 0.
            messages: vec![Message::system("sys"), Message::user("hi")],
            tools: vec![],
            model: "test-model".into(),
            workspace_root: PathBuf::from("/tmp/nca-p3-test"),
            turn_id: 1,
            step_index: 1,
            event_tx,
        };

        let err = chain
            .call(&dyn_provider, request)
            .await
            .expect_err("must escalate");
        assert!(err.is_context_overflow(), "original error unchanged");
        let calls = provider.calls.lock().unwrap();
        assert_eq!(
            calls.len(),
            1,
            "no-progress prune escalates after one call (≤2 bound)"
        );
        // The bracket (Start + End with kv_prefix_broken=false) is emitted
        // before the escalation decision — audit value even without retry.
    }

    // C5b — non-overflow error: single attempt, no events, untouched Err.
    #[tokio::test]
    async fn non_overflow_error_passes_through_untouched() {
        struct AlwaysAuthFail;
        #[async_trait::async_trait]
        impl Provider for AlwaysAuthFail {
            async fn chat(
                &self,
                _messages: &[Message],
                _tools: &[ToolDefinition],
                _model: &str,
                _workspace_root: &Path,
            ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, ProviderError> {
                Err(ProviderError::AuthError("invalid api key".into()))
            }
        }
        let dyn_provider: Arc<dyn Provider> = Arc::new(AlwaysAuthFail);
        let mut chain = MiddlewareChain::new();
        chain.push(Arc::new(OverflowRecoveryMiddleware::default()));

        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel::<AgentEvent>(16);
        let request = StepRequest {
            messages: overflow_fixture(),
            tools: vec![],
            model: "test-model".into(),
            workspace_root: PathBuf::from("/tmp/nca-p3-test"),
            turn_id: 1,
            step_index: 1,
            event_tx,
        };

        let err = chain
            .call(&dyn_provider, request)
            .await
            .expect_err("auth error must pass through");
        assert!(matches!(err, ProviderError::AuthError(_)));
        assert!(
            event_rx.try_recv().is_err(),
            "non-overflow errors must not emit events"
        );
    }
}
