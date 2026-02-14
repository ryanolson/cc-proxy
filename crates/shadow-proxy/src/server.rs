//! Axum HTTP server: router, listener, graceful shutdown.

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use bytes::Bytes;
use tracing::Instrument;

use crate::config::ProxyConfig;
use crate::openinference;
use crate::proxy::correlation;
use crate::proxy::primary;
use crate::proxy::shadow::ShadowDispatcher;

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    pub config: ProxyConfig,
    pub primary_client: reqwest::Client,
    pub shadow_dispatcher: ShadowDispatcher,
}

/// Build and run the HTTP server.
pub async fn run(state: AppState) -> anyhow::Result<()> {
    let listen_addr = state.config.server.listen_address.clone();

    let app = Router::new()
        .route("/v1/messages", post(handle_messages))
        .route("/v1/messages/count_tokens", post(handle_count_tokens))
        .route("/health", get(handle_health))
        .fallback(handle_fallback)
        .with_state(Arc::new(state));

    let listener = tokio::net::TcpListener::bind(&listen_addr).await?;
    tracing::info!(address = %listen_addr, "Shadow proxy listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("Shadow proxy shut down gracefully");
    Ok(())
}

/// Main handler for POST /v1/messages.
///
/// 1. Generate correlation ID
/// 2. Optionally rewrite the model field (if `model_override` is configured)
/// 3. Clone body for shadow dispatch
/// 4. Fire-and-forget shadow requests
/// 5. Forward original request to Anthropic and stream response back
async fn handle_messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let correlation_id = correlation::generate_id();

    // Rewrite request body: model override + max_tokens default
    let body = match rewrite_request_body(&body, state.config.primary.model_override.as_deref()) {
        Ok(rewritten) => rewritten,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to rewrite request body, forwarding unchanged");
            body
        }
    };

    // Parse the (possibly rewritten) request body as untyped JSON — resilient
    // to unknown content block types (thinking, citations, server_tool_use, etc.)
    let parsed: Option<serde_json::Value> = serde_json::from_slice(&body).ok();

    let model = parsed
        .as_ref()
        .and_then(|v| v.get("model").and_then(|m| m.as_str()))
        .unwrap_or("unknown")
        .to_string();
    let is_streaming = parsed
        .as_ref()
        .and_then(|v| v.get("stream").and_then(|s| s.as_bool()))
        .unwrap_or(false);

    let span = shadow_tracing::proxy_request_span!(&correlation_id, &model);

    // Set OpenInference request attributes on the root span so Phoenix
    // shows the trace as kind=LLM with visible I/O at the top level.
    if let Some(ref req) = parsed {
        openinference::set_request_attributes(&span, req);

        // Typed validation sidecar: detect Anthropic type drift and emit
        // structured OTLP attributes queryable in Phoenix.
        let report = crate::convert::validation::validate_request(&body, req);
        report.emit(&span);
    }

    async {
        // Fire shadow requests (non-blocking, fire-and-forget)
        state.shadow_dispatcher.dispatch_all(&body, &correlation_id);

        // Forward to Anthropic (blocking, returns the streaming response)
        let url = format!("{}/v1/messages", state.config.primary.upstream_base_url);

        // Pass the root span so TeeBody can set response attributes on it
        // when streaming completes, keeping the span open until then.
        let root_span = tracing::Span::current();

        primary::forward_to_anthropic(
            &state.primary_client,
            &url,
            &headers,
            body,
            &correlation_id,
            is_streaming,
            root_span,
        )
        .await
    }
    .instrument(span)
    .await
}

/// Rewrite fields in a JSON request body before forwarding.
///
/// - Replaces `model` with `new_model` (if Some)
/// - Sets `max_tokens` to 16384 if absent or null
///
/// Uses `serde_json::Value` for a minimal parse-and-patch so that all other
/// fields (including unknown/future ones) are preserved exactly.
fn rewrite_request_body(body: &Bytes, new_model: Option<&str>) -> Result<Bytes, serde_json::Error> {
    let mut value: serde_json::Value = serde_json::from_slice(body)?;
    if let Some(obj) = value.as_object_mut() {
        if let Some(model) = new_model {
            obj.insert(
                "model".to_string(),
                serde_json::Value::String(model.to_string()),
            );
        }
        // Default max_tokens to 16384 if not set
        if !obj.contains_key("max_tokens") || obj.get("max_tokens").is_some_and(|v| v.is_null()) {
            obj.insert(
                "max_tokens".to_string(),
                serde_json::Value::Number(16384.into()),
            );
        }
    }
    let rewritten = serde_json::to_vec(&value)?;
    Ok(Bytes::from(rewritten))
}

/// Catch-all fallback handler for any request not matching explicit routes.
///
/// Forwards the request to Anthropic unchanged (no shadow dispatch).
/// Supports all HTTP methods (GET, POST, DELETE, etc.).
async fn handle_fallback(State(state): State<Arc<AppState>>, request: Request) -> Response {
    let correlation_id = correlation::generate_id();

    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let query = request
        .uri()
        .query()
        .map(|q| format!("?{q}"))
        .unwrap_or_default();
    let url = format!("{}{path}{query}", state.config.primary.upstream_base_url);

    let headers = request.headers().clone();
    let body = match axum::body::to_bytes(request.into_body(), 10 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            tracing::error!(error = %e, "Failed to read request body");
            return (StatusCode::BAD_REQUEST, "failed to read request body").into_response();
        }
    };

    primary::forward_raw(
        &state.primary_client,
        method,
        &url,
        &headers,
        body,
        &correlation_id,
    )
    .await
}

/// Stub handler for POST /v1/messages/count_tokens.
///
/// Some clients call this endpoint for pre-flight token estimation. When the
/// upstream doesn't implement it, we return a rough estimate (chars/4) to
/// unblock the client.
async fn handle_count_tokens(body: Bytes) -> Response {
    let input_tokens = serde_json::from_slice::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| {
            // Sum up character lengths of all message content
            v.get("messages")
                .and_then(|m| m.as_array())
                .map(|msgs| {
                    msgs.iter()
                        .filter_map(|msg| msg.get("content").and_then(|c| c.as_str()))
                        .map(|s| s.len())
                        .sum::<usize>()
                })
        })
        .unwrap_or(100)
        / 4; // rough chars-to-tokens ratio

    let resp = serde_json::json!({ "input_tokens": input_tokens });
    (StatusCode::OK, axum::Json(resp)).into_response()
}

/// Health check endpoint.
async fn handle_health() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

/// Wait for SIGINT (Ctrl+C) for graceful shutdown.
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C signal handler");
    tracing::info!("Shutdown signal received, draining connections...");
}
