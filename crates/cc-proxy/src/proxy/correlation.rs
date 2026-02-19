//! Correlation ID generation for request tracing.

use uuid::Uuid;

/// The header name used to correlate primary and shadow requests.
pub const CORRELATION_HEADER: &str = "x-shadow-request-id";

/// Generate a new correlation ID (UUID v4).
pub fn generate_id() -> String {
    Uuid::new_v4().to_string()
}
