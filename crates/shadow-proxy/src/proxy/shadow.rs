//! Shadow request dispatcher.
//!
//! Fires converted requests at shadow models via LiteLLM. All shadow requests
//! are fire-and-forget: failures never affect the primary path.
//!
//! All parsing uses `serde_json::Value` — no typed Anthropic structs. This
//! makes shadow dispatch resilient to unknown content block types. Unknown
//! types are logged at info level in the conversion layer so they can be
//! identified and protocol support added incrementally.

use std::sync::Arc;
use std::time::Instant;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use tokio::sync::Semaphore;
use tracing::Instrument;

use super::correlation::CORRELATION_HEADER;
use crate::config::ShadowConfig;
use crate::convert::anthropic_to_openai::anthropic_to_openai;
use crate::openinference;

/// Dispatches shadow requests to configured models via LiteLLM.
#[derive(Clone)]
pub struct ShadowDispatcher {
    client: reqwest::Client,
    semaphore: Arc<Semaphore>,
    config: ShadowConfig,
}

impl ShadowDispatcher {
    /// Create a new dispatcher with the given config and HTTP client.
    pub fn new(client: reqwest::Client, config: ShadowConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent));
        Self {
            client,
            semaphore,
            config,
        }
    }

    /// Dispatch shadow requests to all configured models (fire-and-forget).
    ///
    /// Each model gets its own tokio task. Semaphore limits concurrency.
    /// Failures are logged as warnings and never propagated.
    ///
    /// Quota-check requests (single short user message, no tools, low max_tokens)
    /// are skipped to avoid noisy shadow traces.
    pub fn dispatch_all(&self, request_bytes: &[u8], correlation_id: &str) {
        // Parse as untyped JSON — resilient to unknown content block types
        let value: serde_json::Value = match serde_json::from_slice(request_bytes) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "Failed to parse request JSON for shadow dispatch");
                return;
            }
        };

        // Skip quota-check requests
        if Self::is_quota_check(&value) {
            tracing::debug!(
                correlation_id = %correlation_id,
                "Skipping shadow dispatch for quota-check request"
            );
            return;
        }

        // Convert to OpenAI format once (from Value — never fails on unknown blocks)
        let openai_body = match anthropic_to_openai(&value) {
            Ok(body) => body,
            Err(e) => {
                tracing::warn!(error = %e, "Failed to convert request to OpenAI format");
                return;
            }
        };

        for model in &self.config.models {
            let client = self.client.clone();
            let semaphore = self.semaphore.clone();
            let model = model.clone();
            let correlation_id = correlation_id.to_string();
            let litellm_url = self.config.litellm_url.clone();
            let litellm_api_key = self.config.litellm_api_key.clone();
            let timeout_secs = self.config.timeout_secs;

            // Override the model field
            let mut body = openai_body.clone();
            body["model"] = serde_json::json!(model);

            tokio::spawn(async move {
                let span = shadow_tracing::shadow_request_span!(&correlation_id, &model);

                async {
                    // Try to acquire semaphore permit
                    let _permit = match semaphore.try_acquire() {
                        Ok(permit) => permit,
                        Err(_) => {
                            tracing::warn!(
                                model = %model,
                                "Shadow semaphore full, skipping request"
                            );
                            return;
                        }
                    };

                    let start = Instant::now();

                    // Build the request
                    let mut req_builder = client
                        .post(&litellm_url)
                        .header("content-type", "application/json")
                        .header(CORRELATION_HEADER, &correlation_id);

                    if !litellm_api_key.is_empty() {
                        req_builder = req_builder.header(
                            "authorization",
                            format!("Bearer {}", litellm_api_key),
                        );
                    }

                    let body_str = match serde_json::to_string(&body) {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::warn!(error = %e, model = %model, "Failed to serialize shadow request");
                            return;
                        }
                    };

                    // Set OpenInference attributes so Phoenix renders
                    // shadow_request as an LLM span with visible I/O.
                    let current_span = tracing::Span::current();
                    openinference::set_shadow_request_attributes(
                        &current_span,
                        &body,
                    );

                    // Record the request body and message count on the span
                    // so we can verify the full history is being sent.
                    let msg_count = body["messages"]
                        .as_array()
                        .map(|a| a.len())
                        .unwrap_or(0);
                    current_span
                        .record("shadow.request_message_count", msg_count);
                    current_span
                        .record("shadow.request_body", &body_str);

                    req_builder = req_builder.body(body_str);

                    // Send with timeout
                    let result = tokio::time::timeout(
                        std::time::Duration::from_secs(timeout_secs),
                        req_builder.send(),
                    )
                    .await;

                    let latency = start.elapsed().as_millis() as u64;
                    tracing::Span::current().record("latency_ms", latency);

                    match result {
                        Ok(Ok(resp)) => {
                            let status = resp.status().as_u16();
                            tracing::Span::current().record("status", status);

                            // Read full response body
                            match resp.text().await {
                                Ok(response_body) => {
                                    // Try to extract token usage from the response
                                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&response_body) {
                                        if let Some(usage) = parsed.get("usage") {
                                            if let Some(input) = usage.get("prompt_tokens").and_then(|v| v.as_u64()) {
                                                tracing::Span::current().record("input_tokens", input);
                                            }
                                            if let Some(output) = usage.get("completion_tokens").and_then(|v| v.as_u64()) {
                                                tracing::Span::current().record("output_tokens", output);
                                            }
                                        }
                                    }

                                    // Set OpenInference response attributes
                                    openinference::set_shadow_response_attributes(
                                        &tracing::Span::current(),
                                        &response_body,
                                    );

                                    // Store full response body as span attribute
                                    tracing::Span::current().record("shadow.response_body", &response_body);

                                    tracing::info!(
                                        model = %model,
                                        status = status,
                                        latency_ms = latency,
                                        "Shadow request complete"
                                    );
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        error = %e,
                                        model = %model,
                                        "Failed to read shadow response body"
                                    );
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            tracing::Span::current().record("status", 0_u16);
                            tracing::warn!(
                                error = %e,
                                model = %model,
                                latency_ms = latency,
                                "Shadow request failed"
                            );
                        }
                        Err(_) => {
                            tracing::Span::current().record("status", 0_u16);
                            tracing::warn!(
                                model = %model,
                                latency_ms = latency,
                                timeout_secs = timeout_secs,
                                "Shadow request timed out"
                            );
                        }
                    }
                }
                .instrument(span)
                .await;
            });
        }

        drop(openai_body);
    }

    /// Send a request to the first configured shadow model and return the response.
    ///
    /// Used in `ShadowOnly` mode where the proxy returns the shadow model's response
    /// instead of forwarding to Anthropic. Note: the response is in OpenAI format.
    pub async fn dispatch_and_respond(
        &self,
        request_bytes: &[u8],
        correlation_id: &str,
    ) -> Response {
        let value: serde_json::Value = match serde_json::from_slice(request_bytes) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "Failed to parse request JSON for shadow-only dispatch");
                return (StatusCode::BAD_REQUEST, "invalid request JSON").into_response();
            }
        };

        let mut openai_body = match anthropic_to_openai(&value) {
            Ok(body) => body,
            Err(e) => {
                tracing::warn!(error = %e, "Failed to convert request to OpenAI format");
                return (StatusCode::BAD_REQUEST, "conversion error").into_response();
            }
        };

        let model = match self.config.models.first() {
            Some(m) => m.clone(),
            None => {
                return (StatusCode::BAD_GATEWAY, "no shadow models configured").into_response();
            }
        };

        openai_body["model"] = serde_json::json!(model);

        // Try to acquire semaphore permit (non-blocking)
        let _permit = match self.semaphore.try_acquire() {
            Ok(permit) => permit,
            Err(_) => {
                tracing::warn!("Shadow semaphore full, rejecting shadow-only request");
                return (StatusCode::SERVICE_UNAVAILABLE, "shadow dispatch at capacity")
                    .into_response();
            }
        };

        let body_str = match serde_json::to_string(&openai_body) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "Failed to serialize shadow request");
                return (StatusCode::INTERNAL_SERVER_ERROR, "serialization error").into_response();
            }
        };

        let mut req_builder = self
            .client
            .post(&self.config.litellm_url)
            .header("content-type", "application/json")
            .header(CORRELATION_HEADER, correlation_id)
            .body(body_str);

        if !self.config.litellm_api_key.is_empty() {
            req_builder = req_builder.header(
                "authorization",
                format!("Bearer {}", self.config.litellm_api_key),
            );
        }

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(self.config.timeout_secs),
            req_builder.send(),
        )
        .await;

        match result {
            Ok(Ok(resp)) => {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                let axum_status =
                    StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
                (
                    axum_status,
                    [("content-type", "application/json")],
                    body,
                )
                    .into_response()
            }
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "Shadow-only request failed");
                (StatusCode::BAD_GATEWAY, "shadow upstream error").into_response()
            }
            Err(_) => {
                tracing::warn!("Shadow-only request timed out");
                (StatusCode::GATEWAY_TIMEOUT, "shadow upstream timeout").into_response()
            }
        }
    }

    /// Returns the converted OpenAI body that would be sent for a given request.
    /// Exposed for testing only.
    #[cfg(test)]
    pub fn convert_for_test(request_bytes: &[u8]) -> Option<serde_json::Value> {
        let value: serde_json::Value = serde_json::from_slice(request_bytes).ok()?;
        anthropic_to_openai(&value).ok()
    }

    /// Returns true if the request looks like a Claude Code quota check:
    /// - Exactly one message with role `user`
    /// - No tools defined
    /// - `max_tokens` <= 32
    fn is_quota_check(req: &serde_json::Value) -> bool {
        let messages = match req.get("messages").and_then(|v| v.as_array()) {
            Some(m) => m,
            None => return false,
        };
        let max_tokens = req
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(u64::MAX);
        let has_tools = req
            .get("tools")
            .and_then(|v| v.as_array())
            .is_some_and(|t| !t.is_empty());
        let single_user_msg =
            messages.len() == 1 && messages[0].get("role").and_then(|v| v.as_str()) == Some("user");

        single_user_msg && !has_tools && max_tokens <= 32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Build a realistic multi-turn Anthropic request resembling a Claude Code session.
    fn multi_turn_request() -> serde_json::Value {
        json!({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 8096,
            "system": "You are a coding assistant.",
            "stream": true,
            "messages": [
                {"role": "user", "content": "What files are in src?"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "t1", "name": "list_files", "input": {"path": "src"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "main.rs\nlib.rs"}
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Found main.rs and lib.rs."}
                ]},
                {"role": "user", "content": "Read main.rs"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t2", "name": "read_file", "input": {"path": "src/main.rs"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t2", "content": "fn main() {}"}
                ]}
            ],
            "tools": [
                {"name": "list_files", "description": "List files", "input_schema": {"type": "object"}},
                {"name": "read_file", "description": "Read a file", "input_schema": {"type": "object"}}
            ]
        })
    }

    #[test]
    fn test_shadow_dispatch_preserves_full_history() {
        let req = multi_turn_request();
        let bytes = serde_json::to_vec(&req).unwrap();

        let oai = ShadowDispatcher::convert_for_test(&bytes).expect("conversion should succeed");

        let msgs = oai["messages"]
            .as_array()
            .expect("messages should be an array");

        // 1 system + 7 conversation messages = 8
        assert_eq!(
            msgs.len(),
            8,
            "Shadow request must include full history. Got {} messages: {:#?}",
            msgs.len(),
            msgs
        );

        // Verify every role is present in the expected order
        let roles: Vec<&str> = msgs.iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(
            roles,
            vec![
                "system",
                "user",
                "assistant",
                "tool",
                "assistant",
                "user",
                "assistant",
                "tool"
            ]
        );
    }

    #[test]
    fn test_shadow_dispatch_message_count_matches_input() {
        // Verify that for N input messages, we get exactly N+1 output messages
        // (the +1 is for the system prompt converted to a system message).
        let req = multi_turn_request();
        let input_msg_count = req["messages"].as_array().unwrap().len();
        let bytes = serde_json::to_vec(&req).unwrap();

        let oai = ShadowDispatcher::convert_for_test(&bytes).unwrap();
        let output_msg_count = oai["messages"].as_array().unwrap().len();

        // Input has 7 messages + 1 system = 8 output messages
        assert_eq!(
            output_msg_count,
            input_msg_count + 1, // +1 for system prompt
            "Output message count ({output_msg_count}) should equal input ({input_msg_count}) + 1 (system)"
        );
    }

    #[tokio::test]
    async fn test_shadow_dispatch_sends_full_body_to_endpoint() {
        use std::future::IntoFuture;
        use std::sync::Arc;
        use tokio::sync::Mutex;

        // Set up a mock HTTP server that captures request bodies
        let captured: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
        let captured_clone = captured.clone();

        let app = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(move |body: axum::body::Bytes| {
                let captured = captured_clone.clone();
                async move {
                    if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&body) {
                        captured.lock().await.push(val);
                    }
                    axum::http::StatusCode::OK
                }
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(axum::serve(listener, app).into_future());

        // Create dispatcher pointing at our mock server
        let config = ShadowConfig {
            litellm_url: format!("http://{addr}/v1/chat/completions"),
            litellm_api_key: String::new(),
            models: vec!["test-model".to_string()],
            max_concurrent: 10,
            timeout_secs: 5,
        };
        let client = reqwest::Client::new();
        let dispatcher = ShadowDispatcher::new(client, config);

        // Dispatch a multi-turn request
        let req = multi_turn_request();
        let bytes = serde_json::to_vec(&req).unwrap();
        dispatcher.dispatch_all(&bytes, "test-correlation-id");

        // Wait for the spawned task to complete
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Verify the captured request
        let bodies = captured.lock().await;
        assert_eq!(bodies.len(), 1, "Expected exactly one shadow request");

        let sent = &bodies[0];
        let msgs = sent["messages"].as_array().expect("messages should exist");

        // Full history: 1 system + 7 conversation = 8
        assert_eq!(
            msgs.len(),
            8,
            "Shadow HTTP request must contain full history. Got {} messages",
            msgs.len()
        );

        // Model should be overridden to the shadow model
        assert_eq!(sent["model"], "test-model");

        // Stream should be false (shadows are non-streaming)
        assert_eq!(sent["stream"], false);
    }

    #[test]
    fn test_quota_check_skipped() {
        let req = json!({
            "model": "claude-sonnet-4",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}]
        });
        assert!(ShadowDispatcher::is_quota_check(&req));
    }

    #[test]
    fn test_multi_turn_not_quota_check() {
        let req = multi_turn_request();
        assert!(!ShadowDispatcher::is_quota_check(&req));
    }
}
