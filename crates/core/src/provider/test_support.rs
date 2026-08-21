use std::thread;

use tiny_http::{Header, Request, Response, Server, StatusCode};

use super::StreamChunk;

pub fn spawn_sse_server<F>(body: String, status: u16, assert_request: F) -> String
where
    F: FnOnce(&Request) + Send + 'static,
{
    spawn_sse_server_with_body(body, status, |request, _request_body| {
        assert_request(request)
    })
}

/// Like [`spawn_sse_server`], but the assertion closure also receives the
/// decoded request body — for pinning wire-contract fields (e.g. a provider's
/// `thinking` override or floored `max_tokens`) on the exact JSON sent.
pub fn spawn_sse_server_with_body<F>(body: String, status: u16, assert_request: F) -> String
where
    F: FnOnce(&Request, &str) + Send + 'static,
{
    let server = Server::http("127.0.0.1:0").expect("start mock server");
    let base_url = match server.server_addr() {
        tiny_http::ListenAddr::IP(addr) => format!("http://{addr}"),
        other => panic!("unsupported listen addr: {other:?}"),
    };

    thread::spawn(move || {
        let mut request = server.recv().expect("receive request");
        let mut request_bytes = Vec::new();
        request
            .as_reader()
            .read_to_end(&mut request_bytes)
            .expect("read request body");
        let request_body = String::from_utf8_lossy(&request_bytes).into_owned();
        assert_request(&request, &request_body);
        let response = Response::from_string(body)
            .with_status_code(StatusCode(status))
            .with_header(
                Header::from_bytes("Content-Type", "text/event-stream")
                    .expect("content type header"),
            );
        request.respond(response).expect("send response");
    });

    base_url
}

pub async fn collect_chunks(mut rx: tokio::sync::mpsc::Receiver<StreamChunk>) -> Vec<StreamChunk> {
    let mut chunks = Vec::new();
    while let Some(chunk) = rx.recv().await {
        let done = matches!(chunk, StreamChunk::Done);
        chunks.push(chunk);
        if done {
            break;
        }
    }
    chunks
}
