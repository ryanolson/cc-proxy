//! Raw byte forwarding to Anthropic's API.
//!
//! The primary path streams bytes verbatim: no parsing, no transformation.
//! This ensures zero fidelity loss for SSE formatting, field ordering, etc.
//!
//! A `TeeBody` wrapper captures a copy of response bytes for OpenInference
//! attribute extraction without affecting the stream sent to the client.

use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Instant;

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures_core::Stream;
use tracing::Instrument;

use super::correlation::CORRELATION_HEADER;
use crate::openinference;
use crate::stats::ProxyStats;

/// Headers that should NOT be forwarded (hop-by-hop headers).
const HOP_BY_HOP_HEADERS: &[&str] = &[
    "host",
    "connection",
    "transfer-encoding",
    "keep-alive",
    "upgrade",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
];

/// A stream wrapper that passes through bytes unchanged while accumulating a
/// copy of all data. When the inner stream completes, it calls
/// `set_response_attributes()` on the held tracing span and drops the span
/// clone (which closes the OTel span).
///
/// Also records timing attributes on the root span:
/// - `ttft_ms`: milliseconds from `start` to first chunk received
/// - `total_duration_ms`: milliseconds from `start` to stream end
struct TeeBody {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    buffer: Arc<Mutex<Vec<u8>>>,
    span: tracing::Span,
    is_streaming: bool,
    stats: Option<ProxyStats>,
    /// When the upstream request was sent (used to compute timing attributes).
    start: Instant,
    /// Whether the first chunk has been seen (to record ttft_ms exactly once).
    first_chunk_seen: bool,
}

impl Stream for TeeBody {
    type Item = Result<Bytes, reqwest::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                // Record time-to-first-token exactly once: the moment the first
                // upstream byte arrives after the request was sent.
                if !self.first_chunk_seen {
                    self.first_chunk_seen = true;
                    self.span
                        .record("ttft_ms", self.start.elapsed().as_millis() as u64);
                }
                if let Ok(mut buf) = self.buffer.lock() {
                    buf.extend_from_slice(&chunk);
                }
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => {
                // Record total end-to-end streaming duration (request sent → last byte).
                self.span
                    .record("total_duration_ms", self.start.elapsed().as_millis() as u64);

                // Stream complete — set response attributes
                if let Ok(buf) = self.buffer.lock() {
                    openinference::set_response_attributes(&self.span, &buf, self.is_streaming);
                    if let Some(ref stats) = self.stats {
                        extract_and_record_stats(stats, &buf, self.is_streaming);
                    }
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Forward the raw request body to Anthropic and stream the response back.
///
/// Used by the `/v1/messages` handler. The caller constructs the full URL
/// from `upstream_base_url`. The `root_span` is the parent `proxy_request`
/// span — TeeBody holds a clone of it so OpenInference response attributes
/// are set on the root trace (keeping it open until streaming completes).
#[allow(clippy::too_many_arguments)]
pub async fn forward_to_anthropic(
    client: &reqwest::Client,
    url: &str,
    headers: &HeaderMap,
    body: Bytes,
    correlation_id: &str,
    is_streaming: bool,
    root_span: tracing::Span,
    stats: ProxyStats,
) -> Response {
    let host = url
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .split('/')
        .next()
        .unwrap_or(url);
    let span = cc_tracing::primary_forward_span!(correlation_id, host);
    let start = Instant::now();

    async {
        // Build the upstream request
        let mut req_builder = client
            .post(url)
            .body(body)
            .header("content-type", "application/json")
            .header(CORRELATION_HEADER, correlation_id);

        // Forward non-hop-by-hop headers from the original request
        for (name, value) in headers.iter() {
            let name_str = name.as_str().to_lowercase();
            if HOP_BY_HOP_HEADERS.contains(&name_str.as_str()) {
                continue;
            }
            // Skip content-type and correlation header (already set above)
            if name_str == "content-type" || name_str == CORRELATION_HEADER {
                continue;
            }
            // Skip content-length — reqwest sets it from the actual body,
            // and the body may have changed size (e.g. model_override rewrite)
            if name_str == "content-length" {
                continue;
            }
            req_builder = req_builder.header(name, value);
        }

        // Send the request
        let upstream_result = req_builder.send().await;

        build_response(
            upstream_result,
            start,
            correlation_id,
            is_streaming,
            &root_span,
            Some(stats),
        )
    }
    .instrument(span)
    .await
}

/// Forward the raw request body to a target endpoint and stream the response back.
///
/// Used in `TargetOnly` mode. Identical to `forward_to_anthropic` except:
/// - Hits `target_base_url/v1/messages` instead of Anthropic
/// - Does NOT forward the `x-api-key` header (target handles auth separately)
#[allow(clippy::too_many_arguments)]
pub async fn forward_to_target(
    client: &reqwest::Client,
    target_base_url: &str,
    headers: &HeaderMap,
    body: Bytes,
    correlation_id: &str,
    is_streaming: bool,
    root_span: tracing::Span,
    stats: ProxyStats,
) -> Response {
    let url = format!("{}/v1/messages", target_base_url);
    let host = target_base_url
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .split('/')
        .next()
        .unwrap_or(target_base_url);
    let span = cc_tracing::primary_forward_span!(correlation_id, host);
    let start = Instant::now();

    async {
        // Build the upstream request
        let mut req_builder = client
            .post(&url)
            .body(body)
            .header("content-type", "application/json")
            .header(CORRELATION_HEADER, correlation_id);

        // Forward non-hop-by-hop headers, but skip x-api-key (target handles auth)
        for (name, value) in headers.iter() {
            let name_str = name.as_str().to_lowercase();
            if HOP_BY_HOP_HEADERS.contains(&name_str.as_str()) {
                continue;
            }
            if name_str == "content-type" || name_str == CORRELATION_HEADER {
                continue;
            }
            if name_str == "content-length" {
                continue;
            }
            if name_str == "x-api-key" {
                continue;
            }
            req_builder = req_builder.header(name, value);
        }

        let upstream_result = req_builder.send().await;

        build_response(
            upstream_result,
            start,
            correlation_id,
            is_streaming,
            &root_span,
            Some(stats),
        )
    }
    .instrument(span)
    .await
}

/// Forward any request (any HTTP method) to upstream and stream the response back.
///
/// Used by the catch-all fallback handler for endpoints other than `/v1/messages`.
pub async fn forward_raw(
    client: &reqwest::Client,
    method: Method,
    url: &str,
    headers: &HeaderMap,
    body: Bytes,
    correlation_id: &str,
) -> Response {
    let span = tracing::info_span!(
        "passthrough_forward",
        correlation_id = %correlation_id,
        method = %method,
        url = %url,
        status = tracing::field::Empty,
        latency_ms = tracing::field::Empty,
    );
    let start = Instant::now();

    async {
        // Build the upstream request with the original method
        let mut req_builder = client
            .request(method, url)
            .body(body)
            .header(CORRELATION_HEADER, correlation_id);

        // Forward non-hop-by-hop headers from the original request
        for (name, value) in headers.iter() {
            let name_str = name.as_str().to_lowercase();
            if HOP_BY_HOP_HEADERS.contains(&name_str.as_str()) {
                continue;
            }
            if name_str == CORRELATION_HEADER {
                continue;
            }
            req_builder = req_builder.header(name, value);
        }

        // Send the request
        let upstream_result = req_builder.send().await;

        build_response_simple(upstream_result, start, correlation_id)
    }
    .instrument(span)
    .await
}

/// Build an axum Response from the upstream reqwest result, streaming the body
/// through a `TeeBody` that captures bytes for OpenInference response attributes.
fn build_response(
    upstream_result: Result<reqwest::Response, reqwest::Error>,
    start: Instant,
    correlation_id: &str,
    is_streaming: bool,
    span: &tracing::Span,
    stats: Option<ProxyStats>,
) -> Response {
    let upstream_resp = match upstream_result {
        Ok(resp) => resp,
        Err(e) => {
            let latency = start.elapsed().as_millis() as u64;
            tracing::Span::current().record("latency_ms", latency);
            tracing::Span::current().record("status", 502_u16);

            if e.is_timeout() {
                tracing::error!(error = %e, "Upstream timeout");
                return (StatusCode::GATEWAY_TIMEOUT, "upstream timeout").into_response();
            }
            tracing::error!(error = %e, "Upstream connection error");
            return (StatusCode::BAD_GATEWAY, "upstream connection error").into_response();
        }
    };

    let status = upstream_resp.status();
    let latency = start.elapsed().as_millis() as u64;
    tracing::Span::current().record("latency_ms", latency);
    tracing::Span::current().record("status", status.as_u16());

    tracing::info!(
        status = status.as_u16(),
        latency_ms = latency,
        "Forward complete"
    );

    // Build the response, streaming the upstream body through TeeBody
    let mut response_builder = Response::builder()
        .status(StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY));

    // Forward response headers from upstream
    for (name, value) in upstream_resp.headers().iter() {
        let name_str = name.as_str().to_lowercase();
        if HOP_BY_HOP_HEADERS.contains(&name_str.as_str()) {
            continue;
        }
        response_builder = response_builder.header(name, value);
    }

    // Add correlation ID to response
    response_builder = response_builder.header(
        CORRELATION_HEADER,
        HeaderValue::from_str(correlation_id)
            .unwrap_or_else(|_| HeaderValue::from_static("unknown")),
    );

    // Capture upstream request ID (Anthropic sets x-request-id; targets may too)
    if let Some(req_id) = upstream_resp
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
    {
        span.record("anthropic_request_id", req_id);
    }

    // Wrap the upstream byte stream in TeeBody to capture output for OpenInference
    let tee = TeeBody {
        inner: Box::pin(upstream_resp.bytes_stream()),
        buffer: Arc::new(Mutex::new(Vec::new())),
        span: span.clone(),
        is_streaming,
        stats,
        start,
        first_chunk_seen: false,
    };
    let body = Body::from_stream(tee);

    response_builder.body(body).unwrap_or_else(|e| {
        tracing::error!(error = %e, "Failed to build response");
        (StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response()
    })
}

/// Build an axum Response without TeeBody (used by passthrough/fallback).
fn build_response_simple(
    upstream_result: Result<reqwest::Response, reqwest::Error>,
    start: Instant,
    correlation_id: &str,
) -> Response {
    let upstream_resp = match upstream_result {
        Ok(resp) => resp,
        Err(e) => {
            let latency = start.elapsed().as_millis() as u64;
            tracing::Span::current().record("latency_ms", latency);
            tracing::Span::current().record("status", 502_u16);

            if e.is_timeout() {
                tracing::error!(error = %e, "Upstream timeout");
                return (StatusCode::GATEWAY_TIMEOUT, "upstream timeout").into_response();
            }
            tracing::error!(error = %e, "Upstream connection error");
            return (StatusCode::BAD_GATEWAY, "upstream connection error").into_response();
        }
    };

    let status = upstream_resp.status();
    let latency = start.elapsed().as_millis() as u64;
    tracing::Span::current().record("latency_ms", latency);
    tracing::Span::current().record("status", status.as_u16());

    tracing::info!(
        status = status.as_u16(),
        latency_ms = latency,
        "Forward complete"
    );

    // Build the response, streaming the upstream body verbatim
    let mut response_builder = Response::builder()
        .status(StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY));

    // Forward response headers from upstream
    for (name, value) in upstream_resp.headers().iter() {
        let name_str = name.as_str().to_lowercase();
        if HOP_BY_HOP_HEADERS.contains(&name_str.as_str()) {
            continue;
        }
        response_builder = response_builder.header(name, value);
    }

    // Add correlation ID to response
    response_builder = response_builder.header(
        CORRELATION_HEADER,
        HeaderValue::from_str(correlation_id)
            .unwrap_or_else(|_| HeaderValue::from_static("unknown")),
    );

    // Stream the body verbatim
    let body = Body::from_stream(upstream_resp.bytes_stream());

    response_builder.body(body).unwrap_or_else(|e| {
        tracing::error!(error = %e, "Failed to build response");
        (StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response()
    })
}

/// Extract token usage and tool call counts from the Anthropic response and record to stats.
fn extract_and_record_stats(stats: &ProxyStats, response_bytes: &[u8], is_streaming: bool) {
    if is_streaming {
        extract_streaming_stats(stats, response_bytes);
    } else {
        extract_nonstreaming_stats(stats, response_bytes);
    }
}

fn extract_nonstreaming_stats(stats: &ProxyStats, response_bytes: &[u8]) {
    let body: serde_json::Value = match serde_json::from_slice(response_bytes) {
        Ok(v) => v,
        Err(_) => return,
    };

    if let Some(usage) = body.get("usage") {
        if let Some(input) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
            stats.add_input_tokens(input);
        }
        if let Some(output) = usage.get("output_tokens").and_then(|v| v.as_u64()) {
            stats.add_output_tokens(output);
        }
    }

    if let Some(content) = body.get("content").and_then(|v| v.as_array()) {
        let tool_count = content
            .iter()
            .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_use"))
            .count() as u64;
        if tool_count > 0 {
            stats.add_tool_calls(tool_count);
        }
    }
}

fn extract_streaming_stats(stats: &ProxyStats, response_bytes: &[u8]) {
    let body_str = match std::str::from_utf8(response_bytes) {
        Ok(s) => s,
        Err(_) => return,
    };

    // Guard against double-counting input_tokens: Anthropic sends them in
    // message_start, GLM sends them in message_delta. Some endpoints may send
    // both. We take input_tokens from whichever event delivers them first.
    let mut input_tokens_seen = false;

    for event_chunk in body_str.split("\n\n") {
        let mut event_type = None;
        let mut data_str = None;

        for line in event_chunk.lines() {
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
                    if let Some(input) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
                        stats.add_input_tokens(input);
                        input_tokens_seen = true;
                    }
                }
            }
            Some("message_delta") => {
                if let Some(usage) = data.get("usage") {
                    if let Some(output) = usage.get("output_tokens").and_then(|v| v.as_u64()) {
                        stats.add_output_tokens(output);
                    }
                    // Fallback: some targets (e.g. GLM) report input_tokens here
                    // instead of message_start. Only record if not already seen.
                    if !input_tokens_seen {
                        if let Some(input) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
                            stats.add_input_tokens(input);
                            input_tokens_seen = true;
                        }
                    }
                }
            }
            Some("content_block_start") => {
                if let Some(cb) = data.get("content_block") {
                    if cb.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                        stats.add_tool_calls(1);
                    }
                }
            }
            _ => {}
        }
    }
}
