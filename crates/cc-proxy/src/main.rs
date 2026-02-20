//! cc-proxy: model gateway for routing Claude Code to self-hosted Anthropic-format deployments.

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
use proxy::compare::CompareDispatcher;
use server::AppState;
use stats::ProxyStats;

fn main() -> anyhow::Result<()> {
    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let config_path = args
        .iter()
        .position(|a| a == "--config")
        .and_then(|i| args.get(i + 1).cloned())
        .or_else(|| args.get(1).filter(|a| !a.starts_with('-')).cloned())
        .or_else(|| std::env::var("CC_PROXY_CONFIG").ok())
        .unwrap_or_else(|| "cc-proxy.toml".to_string());

    let target_url_override = args
        .iter()
        .position(|a| a == "--target-url")
        .and_then(|i| args.get(i + 1).cloned());

    let model_override = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1).cloned());

    let allow_anthropic_only = args.iter().any(|a| a == "--allow-anthropic-only");

    // Load configuration
    let mut config = ProxyConfig::load(&config_path)?;

    // Apply CLI overrides (take precedence over TOML and env vars)
    if let Some(url) = target_url_override {
        config.target.url = Some(url);
    }
    if model_override.is_some() {
        config.model_override = model_override;
    }
    config.anthropic_only_allowed = allow_anthropic_only;

    // Build the tokio runtime first — tonic gRPC exporter needs a reactor context
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    runtime.block_on(async {
        // Initialize tracing (OTLP export is optional — falls back to fmt-only)
        let _tracing_guard = cc_tracing::init_tracing(&config.tracing);

        tracing::info!(
            config_path = %config_path,
            listen_address = %config.server.listen_address,
            passthrough_url = %config.passthrough.url,
            target_url = ?config.target.url,
            "Starting cc-proxy"
        );

        run(config).await
    })
}

async fn run(config: ProxyConfig) -> anyhow::Result<()> {
    // Build Anthropic HTTP client
    let primary_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(config.passthrough.timeout_secs))
        .build()?;

    // Build target HTTP client (separate client with its own timeout)
    let target_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(config.target.timeout_secs))
        .build()?;

    // Build compare dispatcher
    let compare_dispatcher = CompareDispatcher::new(
        config.target.url.clone().unwrap_or_default(),
        config.target.timeout_secs,
        config.target.max_concurrent,
        target_client,
    );

    // Build stats and mode
    let stats = ProxyStats::new();
    let initial_mode = match config.default_mode.as_str() {
        "target" => ProxyMode::TargetOnly,
        "anthropic-only" => ProxyMode::AnthropicOnly,
        "compare" => ProxyMode::Compare,
        _ => ProxyMode::TargetOnly,
    };
    let mode = RuntimeMode::new(initial_mode);

    // Build app state
    let state = AppState {
        config,
        primary_client,
        compare_dispatcher,
        stats,
        mode,
        tracing_enabled: Arc::new(AtomicBool::new(true)),
    };

    // Run the server
    server::run(state).await
}
