//! Shadow request dispatcher.
//!
//! Fires converted requests at shadow models via LiteLLM. All shadow requests
//! are fire-and-forget: failures never affect the primary path.

use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Semaphore;
use tracing::Instrument;

use super::correlation::CORRELATION_HEADER;
use crate::config::ShadowConfig;
use crate::convert::anthropic_to_openai::anthropic_to_openai;
use crate::convert::types::AnthropicCreateMessageRequest;

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
    pub fn dispatch_all(&self, request_bytes: &[u8], correlation_id: &str) {
        // Parse the request body once
        let anthropic_req: AnthropicCreateMessageRequest =
            match serde_json::from_slice(request_bytes) {
                Ok(req) => req,
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to parse request for shadow dispatch");
                    return;
                }
            };

        // Convert to OpenAI format once
        let openai_body = match anthropic_to_openai(&anthropic_req) {
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

        // Fix: the openai_body variable was modified by reference in the loop,
        // but we clone it each time. Suppress the unused warning.
        drop(openai_body);
    }
}
