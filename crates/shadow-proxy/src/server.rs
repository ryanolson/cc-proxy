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
use crate::convert::types::AnthropicCreateMessageRequest;
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
/// 2. Clone body for shadow dispatch
/// 3. Fire-and-forget shadow requests
/// 4. Forward original request to Anthropic and stream response back
async fn handle_messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let correlation_id = correlation::generate_id();

    // Parse the request body once for model name, stream flag, and OpenInference attributes
    let parsed: Option<AnthropicCreateMessageRequest> = serde_json::from_slice(&body).ok();

    let model = parsed
        .as_ref()
        .map(|r| r.model.clone())
        .unwrap_or_else(|| "unknown".to_string());
    let is_streaming = parsed.as_ref().is_some_and(|r| r.stream);

    let span = shadow_tracing::proxy_request_span!(&correlation_id, &model);

    // Set OpenInference request attributes on the root span so Phoenix
    // shows the trace as kind=LLM with visible I/O at the top level.
    if let Some(ref req) = parsed {
        openinference::set_request_attributes(&span, req);
    }

    async {
        // Fire shadow requests (non-blocking, fire-and-forget)
        state.shadow_dispatcher.dispatch_all(&body, &correlation_id);

        // Forward to Anthropic (blocking, returns the streaming response)
        let url = format!("{}/v1/messages", state.config.primary.upstream_base_url);

        // Pass the root span so TeeBody can set response attributes on it
        // when streaming completes, keeping the span open until then.
        let root_span = tracing::Span::current();

        if parsed.is_some() {
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
        } else {
            // Fallback: forward without OpenInference attributes
            primary::forward_raw(
                &state.primary_client,
                axum::http::Method::POST,
                &url,
                &headers,
                body,
                &correlation_id,
            )
            .await
        }
    }
    .instrument(span)
    .await
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
