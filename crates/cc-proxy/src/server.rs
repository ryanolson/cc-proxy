//! Axum HTTP server: router, listener, graceful shutdown.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use bytes::Bytes;
use tracing::Instrument;

use crate::config::ProxyConfig;
use crate::mode::{ProxyMode, RuntimeMode};
use crate::openinference;
use crate::proxy::correlation;
use crate::proxy::primary;
use crate::proxy::compare::CompareDispatcher;
use crate::stats::ProxyStats;

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    pub config: ProxyConfig,
    pub primary_client: reqwest::Client,
    pub compare_dispatcher: CompareDispatcher,
    pub stats: ProxyStats,
    pub mode: RuntimeMode,
    pub tracing_enabled: Arc<AtomicBool>,
}

/// Build and run the HTTP server.
pub async fn run(state: AppState) -> anyhow::Result<()> {
    let listen_addr = state.config.server.listen_address.clone();

    let app = Router::new()
        .route("/v1/messages", post(handle_messages))
        .route("/health", get(handle_health))
        .route("/api/stats", get(handle_get_stats))
        .route("/api/mode", get(handle_get_mode).put(handle_set_mode))
        .route("/api/tracing", get(handle_get_tracing).put(handle_set_tracing))
        .fallback(handle_fallback)
        .with_state(Arc::new(state));

    let listener = tokio::net::TcpListener::bind(&listen_addr).await?;
    tracing::info!(address = %listen_addr, "cc-proxy listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("cc-proxy shut down gracefully");
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
    let body = match rewrite_request_body(&body, state.config.model_override.as_deref()) {
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

    let span = cc_tracing::proxy_request_span!(&correlation_id, &model);

    // Set OpenInference request attributes on the root span so Phoenix
    // shows the trace as kind=LLM with visible I/O at the top level.
    if let Some(ref req) = parsed {
        openinference::set_request_attributes(&span, req);

        // Typed validation sidecar: detect Anthropic type drift and emit
        // structured OTLP attributes queryable in Phoenix.
        let report = crate::convert::validation::validate_request(&body, req);
        report.emit(&span);
    }

    // Increment request counter
    state.stats.inc_requests();

    async {
        let current_mode = state.mode.get();

        match current_mode {
            ProxyMode::TargetOnly => {
                // Forward request directly to the target endpoint
                return primary::forward_to_target(
                    &state.primary_client,
                    state.config.target.url.as_deref().unwrap_or(""),
                    &headers,
                    body,
                    &correlation_id,
                    is_streaming,
                    tracing::Span::current(),
                    state.stats.clone(),
                )
                .await;
            }
            ProxyMode::Compare => {
                // Fire-and-forget compare request to target, then forward to Anthropic
                state.compare_dispatcher.dispatch(body.clone(), correlation_id.clone());
            }
            ProxyMode::AnthropicOnly => {
                // Check if anthropic-only mode is allowed
                if !state.config.anthropic_only_allowed {
                    return (
                        StatusCode::FORBIDDEN,
                        axum::Json(serde_json::json!({
                            "error": "anthropic-only mode is not enabled; restart with --allow-anthropic-only"
                        })),
                    )
                        .into_response();
                }
            }
        }

        // Forward to Anthropic (blocking, returns the streaming response)
        let url = format!("{}/v1/messages", state.config.passthrough.url);

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
            state.stats.clone(),
        )
        .await
    }
    .instrument(span)
    .await
}

/// Rewrite fields in a JSON request body before forwarding.
///
/// - Replaces `model` with `new_model` (if Some)
/// - Sets `max_tokens` to 65536 if absent or null
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
        // Default max_tokens to 65536 if not set (ZAI recommends 65536 for coding)
        if !obj.contains_key("max_tokens") || obj.get("max_tokens").is_some_and(|v| v.is_null()) {
            obj.insert(
                "max_tokens".to_string(),
                serde_json::Value::Number(65536.into()),
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
    // In target mode, never hit the passthrough upstream — the target is the only backend.
    if state.mode.get() == ProxyMode::TargetOnly {
        let path = request.uri().path().to_string();
        tracing::debug!(path = %path, "Dropping unmatched route in target mode (no passthrough)");
        return (StatusCode::NOT_FOUND, "not found").into_response();
    }

    let correlation_id = correlation::generate_id();

    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let query = request
        .uri()
        .query()
        .map(|q| format!("?{q}"))
        .unwrap_or_default();
    let url = format!("{}{path}{query}", state.config.passthrough.url);

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

/// GET /api/stats — return current proxy statistics.
async fn handle_get_stats(State(state): State<Arc<AppState>>) -> Response {
    axum::Json(state.stats.snapshot()).into_response()
}

/// GET /api/mode — return the current proxy operating mode.
async fn handle_get_mode(State(state): State<Arc<AppState>>) -> Response {
    axum::Json(serde_json::json!({ "mode": state.mode.get() })).into_response()
}

/// PUT /api/mode — set the proxy operating mode.
async fn handle_set_mode(
    State(state): State<Arc<AppState>>,
    axum::Json(payload): axum::Json<serde_json::Value>,
) -> Response {
    let mode_str = match payload.get("mode").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(serde_json::json!({ "error": "missing 'mode' field" })),
            )
                .into_response();
        }
    };

    let mode: ProxyMode = match serde_json::from_value(serde_json::Value::String(mode_str.to_string())) {
        Ok(m) => m,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(serde_json::json!({
                    "error": "invalid mode, expected: target, compare, or anthropic-only"
                })),
            )
                .into_response();
        }
    };

    // Block anthropic-only mode unless explicitly allowed at launch
    if mode == ProxyMode::AnthropicOnly && !state.config.anthropic_only_allowed {
        return (
            StatusCode::FORBIDDEN,
            axum::Json(serde_json::json!({
                "error": "anthropic-only mode is disabled; restart with --allow-anthropic-only"
            })),
        )
            .into_response();
    }

    state.mode.set(mode);
    tracing::info!(mode = %mode_str, "Proxy mode changed");
    axum::Json(serde_json::json!({ "mode": mode })).into_response()
}

/// GET /api/tracing — return whether trace logging is enabled.
async fn handle_get_tracing(State(state): State<Arc<AppState>>) -> Response {
    let enabled = state.tracing_enabled.load(Ordering::Relaxed);
    axum::Json(serde_json::json!({ "enabled": enabled })).into_response()
}

/// PUT /api/tracing — toggle trace logging on or off.
async fn handle_set_tracing(
    State(state): State<Arc<AppState>>,
    axum::Json(payload): axum::Json<serde_json::Value>,
) -> Response {
    let enabled = match payload.get("enabled").and_then(|v| v.as_bool()) {
        Some(b) => b,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(serde_json::json!({ "error": "missing 'enabled' boolean field" })),
            )
                .into_response();
        }
    };

    state.tracing_enabled.store(enabled, Ordering::Relaxed);
    tracing::info!(enabled = enabled, "Trace logging toggled");
    axum::Json(serde_json::json!({ "enabled": enabled })).into_response()
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
