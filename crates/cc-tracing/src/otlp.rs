//! OTLP exporter setup and TracingGuard.

use anyhow::Result;
use opentelemetry::trace::TracerProvider;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

use crate::config::{OtlpProtocol, TracingConfig};

/// RAII guard that shuts down the tracer provider on drop.
pub struct TracingGuard {
    provider: Option<SdkTracerProvider>,
}

impl Drop for TracingGuard {
    fn drop(&mut self) {
        if let Some(ref mut provider) = self.provider {
            if let Err(e) = provider.shutdown() {
                eprintln!("Failed to shutdown tracer provider: {e}");
            }
        }
    }
}

/// Initialize the tracing subsystem with OTLP export and fmt logging.
///
/// If the OTLP exporter fails to initialize (e.g. endpoint unreachable),
/// falls back to fmt-only tracing so the proxy can start without a collector.
///
/// Returns a [`TracingGuard`] that must be held for the lifetime of the application
/// to ensure traces are flushed on shutdown.
pub fn init_tracing(config: &TracingConfig) -> TracingGuard {
    // Build env filter
    let env_filter =
        EnvFilter::try_new(&config.log_level).unwrap_or_else(|_| EnvFilter::new("info"));

    // Skip OTLP entirely if no endpoint is configured
    let endpoint = match &config.otlp_endpoint {
        Some(url) => url.clone(),
        None => {
            tracing_subscriber::registry()
                .with(
                    tracing_subscriber::fmt::layer()
                        .with_target(true)
                        .with_writer(std::io::stderr),
                )
                .with(env_filter)
                .init();
            return TracingGuard { provider: None };
        }
    };

    match try_init_with_otlp(config, &endpoint, env_filter) {
        Ok(guard) => guard,
        Err(e) => {
            // OTLP failed — fall back to fmt-only so the proxy still starts
            let env_filter =
                EnvFilter::try_new(&config.log_level).unwrap_or_else(|_| EnvFilter::new("info"));

            tracing_subscriber::registry()
                .with(
                    tracing_subscriber::fmt::layer()
                        .with_target(true)
                        .with_writer(std::io::stderr),
                )
                .with(env_filter)
                .init();

            tracing::warn!(
                error = %e,
                endpoint = %endpoint,
                "OTLP exporter failed to initialize, running with fmt-only tracing"
            );

            TracingGuard { provider: None }
        }
    }
}

/// Try to initialize tracing with OTLP export. Returns Err if the exporter
/// cannot be built.
fn try_init_with_otlp(config: &TracingConfig, endpoint: &str, env_filter: EnvFilter) -> Result<TracingGuard> {
    // Build OTLP exporter
    let otlp_exporter = match config.protocol {
        OtlpProtocol::Grpc => opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .build()?,
        OtlpProtocol::Http => opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_endpoint(endpoint)
            .build()?,
    };

    // Build tracer provider with batch exporter
    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(otlp_exporter)
        .with_resource(
            opentelemetry_sdk::Resource::builder_empty()
                .with_service_name(config.service_name.clone())
                .build(),
        )
        .build();

    let tracer = provider.tracer(config.service_name.clone());

    // Assemble the tracing subscriber
    tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(tracer))
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_writer(std::io::stderr),
        )
        .with(env_filter)
        .init();

    tracing::info!(
        endpoint = %endpoint,
        service = %config.service_name,
        protocol = ?config.protocol,
        "OpenTelemetry OTLP tracing initialized"
    );

    Ok(TracingGuard {
        provider: Some(provider),
    })
}
