//! Tracing configuration types.

use serde::Deserialize;

/// Configuration for the OpenTelemetry tracing subsystem.
#[derive(Debug, Clone, Deserialize)]
pub struct TracingConfig {
    /// The service name reported to the OTLP collector.
    #[serde(default = "default_service_name")]
    pub service_name: String,

    /// OTLP collector endpoint (e.g. "http://phoenix:4317").
    /// When `None`, OTLP export is disabled and only fmt logging is used.
    #[serde(default)]
    pub otlp_endpoint: Option<String>,

    /// Transport protocol for OTLP export.
    #[serde(default)]
    pub protocol: OtlpProtocol,

    /// Log level filter (e.g. "info", "debug", "shadow_proxy=debug,info").
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

/// OTLP transport protocol.
#[derive(Debug, Clone, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OtlpProtocol {
    #[default]
    Grpc,
    Http,
}

fn default_service_name() -> String {
    "shadow-proxy".to_string()
}

fn default_log_level() -> String {
    "info".to_string()
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            service_name: default_service_name(),
            otlp_endpoint: None,
            protocol: OtlpProtocol::default(),
            log_level: default_log_level(),
        }
    }
}
