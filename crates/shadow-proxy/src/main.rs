//! Shadow Proxy: transparent proxy that shadows Claude Code requests to
//! alternative models via LiteLLM while forwarding to Anthropic unchanged.

mod config;
mod convert;
mod mode;
mod openinference;
mod proxy;
mod server;
mod stats;

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use config::ProxyConfig;
use mode::{ProxyMode, RuntimeMode};
use proxy::shadow::ShadowDispatcher;
use server::AppState;
use stats::ProxyStats;

fn main() -> anyhow::Result<()> {
    // Determine config path
    let config_path = {
        let args: Vec<String> = std::env::args().collect();
        // Check for --config flag first
        args.iter()
            .position(|a| a == "--config")
            .and_then(|i| args.get(i + 1).cloned())
            // Fall back to positional arg
            .or_else(|| args.get(1).filter(|a| !a.starts_with('-')).cloned())
            .or_else(|| std::env::var("SHADOW_PROXY_CONFIG").ok())
            .unwrap_or_else(|| "shadow-proxy.toml".to_string())
    };

    // Load configuration
    let config = ProxyConfig::load(&config_path)?;

    // Build the tokio runtime first — tonic gRPC exporter needs a reactor context
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    runtime.block_on(async {
        // Initialize tracing (OTLP export is optional — falls back to fmt-only)
        let _tracing_guard = shadow_tracing::init_tracing(&config.tracing);

        tracing::info!(
            config_path = %config_path,
            listen_address = %config.server.listen_address,
            upstream_base = %config.primary.upstream_base_url,
            shadow_models = ?config.shadow.models,
            "Starting shadow-proxy"
        );

        run(config).await
    })
}

async fn run(config: ProxyConfig) -> anyhow::Result<()> {
    // Build primary HTTP client
    let primary_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(config.primary.timeout_secs))
        .build()?;

    // Build shadow HTTP client (separate client with its own timeout)
    let shadow_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(config.shadow.timeout_secs))
        .build()?;

    // Build shadow dispatcher
    let shadow_dispatcher = ShadowDispatcher::new(shadow_client, config.shadow.clone());

    // Build stats and mode
    let stats = ProxyStats::new();
    let mode = RuntimeMode::new(ProxyMode::Both);

    // Build app state
    let state = AppState {
        config,
        primary_client,
        shadow_dispatcher,
        stats,
        mode,
        tracing_enabled: Arc::new(AtomicBool::new(true)),
    };

    // Run the server
    server::run(state).await
}
