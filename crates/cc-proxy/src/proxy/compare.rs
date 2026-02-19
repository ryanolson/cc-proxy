//! Compare dispatcher: fire-and-forget Anthropic-format POST to target URL.
//!
//! The target (GLM/MiniMax) speaks Anthropic API format natively — no conversion
//! needed. The same rewritten request bytes are forwarded directly to
//! `target_url/v1/messages`.
//!
//! All compare requests are fire-and-forget: failures never affect the primary
//! path. Errors are logged as warnings and never propagated.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::sync::Semaphore;
use tracing::Instrument;

use super::correlation::CORRELATION_HEADER;

/// Dispatches compare requests to a target endpoint that speaks Anthropic format.
#[derive(Clone)]
pub struct CompareDispatcher {
    client: reqwest::Client,
    semaphore: Arc<Semaphore>,
    target_url: String,
    timeout: Duration,
}

impl CompareDispatcher {
    /// Create a new dispatcher.
    ///
    /// - `target_url`: base URL of the target (e.g. `https://glm.example.com`)
    /// - `timeout_secs`: per-request timeout in seconds
    /// - `max_concurrent`: semaphore capacity for in-flight compare requests
    /// - `client`: shared reqwest client
    pub fn new(
        target_url: String,
        timeout_secs: u64,
        max_concurrent: usize,
        client: reqwest::Client,
    ) -> Self {
        let semaphore = Arc::new(Semaphore::new(max_concurrent));
        Self {
            client,
            semaphore,
            target_url,
            timeout: Duration::from_secs(timeout_secs),
        }
    }

    /// Fire-and-forget: spawns a tokio task to POST the request bytes to the
    /// target and returns immediately.
    pub fn dispatch(&self, request_bytes: Bytes, correlation_id: String) {
        let client = self.client.clone();
        let semaphore = self.semaphore.clone();
        let url = format!("{}/v1/messages", self.target_url);
        let timeout = self.timeout;

        tokio::spawn(async move {
            let span = tracing::info_span!(
                "compare_request",
                correlation_id = %correlation_id,
                latency_ms = tracing::field::Empty,
                status = tracing::field::Empty,
            );

            async {
                // Non-blocking acquire — drop if at capacity
                let _permit = match semaphore.try_acquire_owned() {
                    Ok(permit) => permit,
                    Err(_) => {
                        tracing::warn!(
                            correlation_id = %correlation_id,
                            "Compare semaphore full, dropping request"
                        );
                        return;
                    }
                };

                let start = Instant::now();

                let result = tokio::time::timeout(
                    timeout,
                    client
                        .post(&url)
                        .header("content-type", "application/json")
                        .header(CORRELATION_HEADER, &correlation_id)
                        .body(request_bytes)
                        .send(),
                )
                .await;

                let latency = start.elapsed().as_millis() as u64;
                tracing::Span::current().record("latency_ms", latency);

                match result {
                    Ok(Ok(resp)) => {
                        let status = resp.status().as_u16();
                        tracing::Span::current().record("status", status);

                        match resp.bytes().await {
                            Ok(body) => {
                                // Try to extract token usage from Anthropic-format response
                                if let Ok(parsed) =
                                    serde_json::from_slice::<serde_json::Value>(&body)
                                {
                                    if let Some(usage) = parsed.get("usage") {
                                        let input =
                                            usage.get("input_tokens").and_then(|v| v.as_u64());
                                        let output =
                                            usage.get("output_tokens").and_then(|v| v.as_u64());
                                        tracing::info!(
                                            status = status,
                                            latency_ms = latency,
                                            input_tokens = ?input,
                                            output_tokens = ?output,
                                            "Compare request complete"
                                        );
                                        return;
                                    }
                                }

                                tracing::info!(
                                    status = status,
                                    latency_ms = latency,
                                    "Compare request complete"
                                );
                            }
                            Err(e) => {
                                tracing::warn!(
                                    error = %e,
                                    status = status,
                                    latency_ms = latency,
                                    "Failed to read compare response body"
                                );
                            }
                        }
                    }
                    Ok(Err(e)) => {
                        tracing::Span::current().record("status", 0_u16);
                        tracing::warn!(
                            error = %e,
                            latency_ms = latency,
                            "Compare request failed"
                        );
                    }
                    Err(_) => {
                        tracing::Span::current().record("status", 0_u16);
                        tracing::warn!(latency_ms = latency, "Compare request timed out");
                    }
                }
            }
            .instrument(span)
            .await;
        });
    }
}
