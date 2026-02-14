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
struct TeeBody {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    buffer: Arc<Mutex<Vec<u8>>>,
    span: tracing::Span,
    is_streaming: bool,
}

impl Stream for TeeBody {
    type Item = Result<Bytes, reqwest::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                if let Ok(mut buf) = self.buffer.lock() {
                    buf.extend_from_slice(&chunk);
                }
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => {
                // Stream complete — set response attributes
                if let Ok(buf) = self.buffer.lock() {
                    openinference::set_response_attributes(&self.span, &buf, self.is_streaming);
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
pub async fn forward_to_anthropic(
    client: &reqwest::Client,
    url: &str,
    headers: &HeaderMap,
    body: Bytes,
    correlation_id: &str,
    is_streaming: bool,
    root_span: tracing::Span,
) -> Response {
    let span = shadow_tracing::primary_forward_span!(correlation_id);
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

    // Wrap the upstream byte stream in TeeBody to capture output for OpenInference
    let tee = TeeBody {
        inner: Box::pin(upstream_resp.bytes_stream()),
        buffer: Arc::new(Mutex::new(Vec::new())),
        span: span.clone(),
        is_streaming,
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
