//! Axum HTTP server: router, listener, graceful shutdown.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::extract::{Path, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use bytes::Bytes;
use tracing::Instrument;

use crate::config::{ProxyConfig, TargetConfig};
use crate::mode::{ProxyMode, RuntimeMode};
use crate::models::{ModelRegistry, RouteTarget};
use crate::openinference;
use crate::proxy::compare::CompareDispatcher;
use crate::proxy::correlation;
use crate::proxy::primary;
use crate::stats::ProxyStats;

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    pub config: ProxyConfig,
    pub primary_client: reqwest::Client,
    pub compare_dispatcher: CompareDispatcher,
    pub stats: ProxyStats,
    pub mode: RuntimeMode,
    pub model_registry: ModelRegistry,
    pub tracing_enabled: Arc<AtomicBool>,
}

/// Build and run the HTTP server.
pub async fn run(state: AppState) -> anyhow::Result<()> {
    let listen_addr = state.config.server.listen_address.clone();

    let app = Router::new()
        .route("/v1/messages", post(handle_messages))
        .route("/v1/models", get(handle_list_models))
        .route("/v1/models/{model_id}", get(handle_get_model))
        .route("/health", get(handle_health))
        .route("/api/stats", get(handle_get_stats))
        .route("/api/mode", get(handle_get_mode).put(handle_set_mode))
        .route(
            "/api/tracing",
            get(handle_get_tracing).put(handle_set_tracing),
        )
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
/// Routing is model-based:
/// 1. Extract model name from request body
/// 2. Resolve via ModelRegistry → Local or Anthropic
/// 3. Local models: apply target defaults, forward to target URL
/// 4. Anthropic models: forward body as-is to Anthropic passthrough
///
/// The runtime mode modifies behavior:
/// - `target` (default): model-based routing as above
/// - `compare`: model-based routing for primary + shadow to target for Anthropic requests
/// - `anthropic-only`: ALL requests → Anthropic (rejects local model requests)
async fn handle_messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let correlation_id = correlation::generate_id();

    // Parse the original request body as untyped JSON — resilient
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

        // In anthropic-only mode, reject requests for local models
        if current_mode == ProxyMode::AnthropicOnly {
            if let RouteTarget::Local { .. } = state.model_registry.resolve(&model) {
                return (
                    StatusCode::BAD_REQUEST,
                    axum::Json(serde_json::json!({
                        "error": format!(
                            "Model '{}' is a local model but proxy is in anthropic-only mode. \
                             Use an Anthropic model or switch proxy mode.",
                            model
                        )
                    })),
                )
                    .into_response();
            }

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

        // Resolve routing target from model name
        let route = state.model_registry.resolve(&model);

        match route {
            RouteTarget::Local { target_url, .. } => {
                // Build rewritten body for local target (apply model override + target defaults)
                let target_body = match apply_local_defaults(
                    &body,
                    state.config.model_override.as_deref(),
                    &state.config.target,
                ) {
                    Ok(rewritten) => rewritten,
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to rewrite target body, forwarding unchanged");
                        body.clone()
                    }
                };

                tracing::info!(
                    model = %model,
                    target_url = %target_url,
                    "Routing to local model"
                );

                primary::forward_to_target(
                    &state.primary_client,
                    &target_url,
                    &headers,
                    target_body,
                    &correlation_id,
                    is_streaming,
                    tracing::Span::current(),
                    state.stats.clone(),
                )
                .await
            }
            RouteTarget::Anthropic => {
                // In compare mode, also fire-and-forget to the default target
                if current_mode == ProxyMode::Compare {
                    let target_body = match apply_local_defaults(
                        &body,
                        state.config.model_override.as_deref(),
                        &state.config.target,
                    ) {
                        Ok(rewritten) => rewritten,
                        Err(e) => {
                            tracing::warn!(error = %e, "Failed to rewrite target body for compare");
                            body.clone()
                        }
                    };
                    state
                        .compare_dispatcher
                        .dispatch(target_body, correlation_id.clone());
                }

                // Forward original unmodified body to Anthropic
                let url = format!("{}/v1/messages", state.config.passthrough.url);

                tracing::info!(
                    model = %model,
                    "Routing to Anthropic"
                );

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
        }
    }
    .instrument(span)
    .await
}

/// Apply model override and target config defaults to a request body.
///
/// - Replaces `model` with `new_model` (if Some)
/// - Sets `max_tokens`, `temperature`, `top_p` from target config defaults (if absent in request)
///
/// Only used for local-model-bound traffic. Anthropic passthrough uses the original body.
fn apply_local_defaults(
    body: &Bytes,
    new_model: Option<&str>,
    target: &TargetConfig,
) -> Result<Bytes, serde_json::Error> {
    let mut value: serde_json::Value = serde_json::from_slice(body)?;
    if let Some(obj) = value.as_object_mut() {
        if let Some(model) = new_model {
            obj.insert(
                "model".to_string(),
                serde_json::Value::String(model.to_string()),
            );
        }
        if let Some(max_tokens) = target.max_tokens {
            if !obj.contains_key("max_tokens") || obj.get("max_tokens").is_some_and(|v| v.is_null())
            {
                obj.insert(
                    "max_tokens".to_string(),
                    serde_json::Value::Number(max_tokens.into()),
                );
            }
        }
        if let Some(temperature) = target.temperature {
            if !obj.contains_key("temperature") {
                obj.insert("temperature".to_string(), serde_json::json!(temperature));
            }
        }
        if let Some(top_p) = target.top_p {
            if !obj.contains_key("top_p") {
                obj.insert("top_p".to_string(), serde_json::json!(top_p));
            }
        }
    }
    let rewritten = serde_json::to_vec(&value)?;
    Ok(Bytes::from(rewritten))
}

/// GET /v1/models — list available models.
///
/// Returns locally-registered models from the ModelRegistry. Merges with
/// Anthropic's model list when passthrough is available (best-effort, 5s timeout).
async fn handle_list_models(State(state): State<Arc<AppState>>) -> Response {
    let local_models = state.model_registry.list_models();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut data: Vec<serde_json::Value> = local_models
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.id,
                "display_name": m.display_name.as_deref().unwrap_or(&m.id),
                "type": "model",
                "created_at": now,
            })
        })
        .collect();

    // Best-effort: merge Anthropic's model list (5s timeout)
    if let Ok(anthropic_models) = fetch_anthropic_models(&state).await {
        data.extend(anthropic_models);
    }

    let first_id = data.first().and_then(|d| d.get("id").and_then(|v| v.as_str().map(String::from)));
    let last_id = data.last().and_then(|d| d.get("id").and_then(|v| v.as_str().map(String::from)));

    axum::Json(serde_json::json!({
        "data": data,
        "has_more": false,
        "first_id": first_id,
        "last_id": last_id,
    }))
    .into_response()
}

/// GET /v1/models/:model_id — get a specific model.
async fn handle_get_model(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Response {
    // Check local registry first
    let local_models = state.model_registry.list_models();
    if let Some(m) = local_models.iter().find(|m| m.id == model_id) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        return axum::Json(serde_json::json!({
            "id": m.id,
            "display_name": m.display_name.as_deref().unwrap_or(&m.id),
            "type": "model",
            "created_at": now,
        }))
        .into_response();
    }

    // Try Anthropic
    let url = format!("{}/v1/models/{}", state.config.passthrough.url, model_id);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_default();

    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            if let Ok(body) = resp.json::<serde_json::Value>().await {
                return axum::Json(body).into_response();
            }
        }
        _ => {}
    }

    (
        StatusCode::NOT_FOUND,
        axum::Json(serde_json::json!({
            "error": format!("Model '{}' not found", model_id)
        })),
    )
        .into_response()
}

/// Fetch Anthropic's model list (best-effort, 5s timeout).
async fn fetch_anthropic_models(state: &AppState) -> Result<Vec<serde_json::Value>, ()> {
    let url = format!("{}/v1/models", state.config.passthrough.url);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(|_| ())?;

    let resp = client.get(&url).send().await.map_err(|_| ())?;
    if !resp.status().is_success() {
        return Err(());
    }

    let body: serde_json::Value = resp.json().await.map_err(|_| ())?;
    body.get("data")
        .and_then(|d| d.as_array().cloned())
        .ok_or(())
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

    let mode: ProxyMode =
        match serde_json::from_value(serde_json::Value::String(mode_str.to_string())) {
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
