//! Configuration types and loading logic.

use cc_tracing::TracingConfig;
use figment::providers::{Env, Format, Toml};
use figment::Figment;
use serde::Deserialize;

/// Top-level proxy configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ProxyConfig {
    pub server: ServerConfig,
    pub passthrough: PassthroughConfig,
    pub target: TargetConfig,
    pub tracing: TracingConfig,
    #[serde(default = "default_mode")]
    pub default_mode: String,

    /// Override the model field in `/v1/messages` request bodies.
    /// Set via CLI `--model`, not from TOML.
    #[serde(skip)]
    pub model_override: Option<String>,

    /// Whether `anthropic-only` mode is permitted at runtime.
    /// Set via CLI `--allow-anthropic-only`, not from TOML.
    #[serde(skip)]
    pub anthropic_only_allowed: bool,
}

/// Server listen configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_listen_address")]
    pub listen_address: String,
}

/// Passthrough upstream configuration (used only in `compare` and `anthropic-only` modes).
#[derive(Debug, Clone, Deserialize)]
pub struct PassthroughConfig {
    #[serde(default = "default_passthrough_url")]
    pub url: String,

    #[serde(default = "default_true")]
    pub passthrough_auth: bool,

    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

/// Target endpoint configuration (primary destination in `target` mode).
#[derive(Debug, Clone, Deserialize)]
pub struct TargetConfig {
    /// Target URL, set via CLI `--target-url`. Not stored in TOML.
    #[serde(skip)]
    pub url: Option<String>,

    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,

    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,

    /// Optional default temperature for target requests (applied if absent in request).
    pub temperature: Option<f64>,

    /// Optional default top_p for target requests (applied if absent in request).
    pub top_p: Option<f64>,

    /// Optional default max_tokens for target requests (applied if absent/null in request).
    pub max_tokens: Option<u64>,
}

fn default_mode() -> String {
    "target".to_string()
}

fn default_listen_address() -> String {
    "0.0.0.0:3080".to_string()
}

fn default_passthrough_url() -> String {
    "https://api.anthropic.com".to_string()
}

fn default_true() -> bool {
    true
}

fn default_timeout() -> u64 {
    300
}

fn default_max_concurrent() -> usize {
    50
}

impl ProxyConfig {
    /// Load configuration from TOML file and environment variables.
    ///
    /// Priority (highest to lowest):
    /// 1. Environment variables (CC_ prefix, __ for nesting)
    /// 2. TOML config file
    /// 3. Defaults
    pub fn load(config_path: &str) -> anyhow::Result<Self> {
        let config: ProxyConfig = Figment::new()
            .merge(Toml::file(config_path))
            .merge(Env::prefixed("CC_").split("__"))
            .extract()?;

        Ok(config)
    }
}
