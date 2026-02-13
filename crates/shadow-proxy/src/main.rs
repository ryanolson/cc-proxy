//! Shadow Proxy: transparent proxy that shadows Claude Code requests to
//! alternative models via LiteLLM while forwarding to Anthropic unchanged.

mod config;
mod convert;
mod openinference;
mod proxy;
mod server;

use std::time::Duration;

use config::ProxyConfig;
use proxy::shadow::ShadowDispatcher;
use server::AppState;

fn main() -> anyhow::Result<()> {
    // Determine config path
    let config_path = std::env::args()
        .nth(1)
        .or_else(|| {
            // Check for --config flag
            let args: Vec<String> = std::env::args().collect();
            args.iter()
                .position(|a| a == "--config")
                .and_then(|i| args.get(i + 1).cloned())
        })
        .or_else(|| std::env::var("SHADOW_PROXY_CONFIG").ok())
        .unwrap_or_else(|| "shadow-proxy.toml".to_string());

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

    // Build app state
    let state = AppState {
        config,
        primary_client,
        shadow_dispatcher,
    };

    // Run the server
    server::run(state).await
}
