//! Reusable OTLP tracing library for shadow-proxy and related services.

pub mod config;
pub mod otlp;
pub mod spans;

pub use config::{OtlpProtocol, TracingConfig};
pub use otlp::{init_tracing, TracingGuard};
