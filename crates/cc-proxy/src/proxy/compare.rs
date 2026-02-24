//! Compare dispatcher: fire-and-forget Anthropic-format POST to target URL.
//!
//! The target  speaks Anthropic API format natively — no conversion
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
use crate::openinference;

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
    /// target and returns immediately. Logs `total_latency_ms` (includes full
    /// body read) alongside the existing `latency_ms` (TTFB).
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
                total_latency_ms = tracing::field::Empty,
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
                                // Record total latency including full body read
                                let total_latency = start.elapsed().as_millis() as u64;
                                tracing::Span::current()
                                    .record("total_latency_ms", total_latency);

                                // Extract token usage — handle both JSON and SSE formats
                                let (input, output) = extract_usage(&body);

                                // Set OpenInference response attributes so Phoenix
                                // captures the full GLM-5 response text
                                let is_streaming = body
                                    .windows(7)
                                    .any(|w| w == b"event: ");
                                openinference::set_response_attributes(
                                    &tracing::Span::current(),
                                    &body,
                                    is_streaming,
                                );

                                tracing::info!(
                                    status = status,
                                    latency_ms = latency,
                                    total_latency_ms = total_latency,
                                    input_tokens = ?input,
                                    output_tokens = ?output,
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

/// Extract input/output token counts from a response body.
///
/// Handles both formats:
/// - Non-streaming JSON: `{"usage": {"input_tokens": N, "output_tokens": N}}`
/// - SSE stream: parses `message_start` and `message_delta` events
fn extract_usage(body: &[u8]) -> (Option<u64>, Option<u64>) {
    // Try non-streaming JSON first
    if let Ok(parsed) = serde_json::from_slice::<serde_json::Value>(body) {
        if let Some(usage) = parsed.get("usage") {
            return (
                usage.get("input_tokens").and_then(|v| v.as_u64()),
                usage.get("output_tokens").and_then(|v| v.as_u64()),
            );
        }
    }

    // Fall back to SSE parsing
    let body_str = match std::str::from_utf8(body) {
        Ok(s) => s,
        Err(_) => return (None, None),
    };

    let mut input_tokens: Option<u64> = None;
    let mut output_tokens: Option<u64> = None;

    for chunk in body_str.split("\n\n") {
        let mut event_type = None;
        let mut data_str = None;

        for line in chunk.lines() {
            if let Some(et) = line.strip_prefix("event: ") {
                event_type = Some(et.trim());
            } else if let Some(d) = line.strip_prefix("data: ") {
                data_str = Some(d.trim());
            }
        }

        let data_str = match data_str {
            Some(d) => d,
            None => continue,
        };

        let data: serde_json::Value = match serde_json::from_str(data_str) {
            Ok(v) => v,
            Err(_) => continue,
        };

        match event_type {
            Some("message_start") => {
                if let Some(usage) = data.get("message").and_then(|m| m.get("usage")) {
                    if let Some(n) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
                        if n > 0 {
                            input_tokens = Some(n);
                        }
                    }
                }
            }
            Some("message_delta") => {
                if let Some(usage) = data.get("usage") {
                    if let Some(n) = usage.get("output_tokens").and_then(|v| v.as_u64()) {
                        output_tokens = Some(n);
                    }
                    // GLM sends input_tokens in message_delta instead of message_start
                    if input_tokens.is_none() {
                        if let Some(n) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
                            if n > 0 {
                                input_tokens = Some(n);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    (input_tokens, output_tokens)
}
