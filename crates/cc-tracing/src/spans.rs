//! Span builder helpers for shadow-proxy instrumentation.

/// Create a tracing span for the top-level proxy request.
///
/// Usage: `let _span = proxy_request_span!(correlation_id, model).entered();`
///
/// Timing fields recorded later by `TeeBody`:
/// - `ttft_ms`: milliseconds from request send to first streaming chunk
/// - `total_duration_ms`: milliseconds from request send to stream end
///
/// Upstream identity:
/// - `anthropic_request_id`: `x-request-id` from the upstream response headers
#[macro_export]
macro_rules! proxy_request_span {
    ($correlation_id:expr, $model:expr) => {
        tracing::info_span!(
            "proxy_request",
            correlation_id = %$correlation_id,
            original_model = %$model,
            ttft_ms = tracing::field::Empty,
            total_duration_ms = tracing::field::Empty,
            anthropic_request_id = tracing::field::Empty,
        )
    };
}

/// Create a tracing span for the primary Anthropic forward.
#[macro_export]
macro_rules! primary_forward_span {
    ($correlation_id:expr, $target:expr) => {
        tracing::info_span!(
            "primary_forward",
            correlation_id = %$correlation_id,
            target = %$target,
            status = tracing::field::Empty,
            latency_ms = tracing::field::Empty,
        )
    };
}

/// Create a tracing span for a compare request to the shadow target.
#[macro_export]
macro_rules! compare_request_span {
    ($correlation_id:expr, $model:expr) => {
        tracing::info_span!(
            "shadow_request",
            correlation_id = %$correlation_id,
            model = %$model,
            latency_ms = tracing::field::Empty,
            input_tokens = tracing::field::Empty,
            output_tokens = tracing::field::Empty,
            status = tracing::field::Empty,
            shadow.request_body = tracing::field::Empty,
            shadow.request_message_count = tracing::field::Empty,
            shadow.response_body = tracing::field::Empty,
        )
    };
}
