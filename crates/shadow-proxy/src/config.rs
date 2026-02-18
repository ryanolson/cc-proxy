//! Configuration types and loading logic.

use figment::providers::{Env, Format, Toml};
use figment::Figment;
use serde::Deserialize;
use shadow_tracing::TracingConfig;

/// Top-level proxy configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ProxyConfig {
    pub server: ServerConfig,
    pub primary: PrimaryConfig,
    pub shadow: ShadowConfig,
    pub tracing: TracingConfig,
    #[serde(default = "default_mode")]
    pub default_mode: String,
}

/// Server listen configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_listen_address")]
    pub listen_address: String,
}

/// Primary upstream (Anthropic) configuration.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct PrimaryConfig {
    #[serde(default = "default_upstream_base_url")]
    pub upstream_base_url: String,

    #[serde(default = "default_true")]
    pub passthrough_auth: bool,

    #[serde(default = "default_primary_timeout")]
    pub timeout_secs: u64,

    /// Override the model field in `/v1/messages` request bodies.
    /// When set, replaces whatever model the client sends with this value.
    #[serde(default)]
    pub model_override: Option<String>,
}

/// Shadow model dispatch configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ShadowConfig {
    #[serde(default = "default_litellm_url")]
    pub litellm_url: String,

    #[serde(default)]
    pub litellm_api_key: String,

    #[serde(default)]
    pub models: Vec<String>,

    #[serde(default = "default_shadow_timeout")]
    pub timeout_secs: u64,

    #[serde(default = "default_max_concurrent")]
    pub max_concurrent: usize,
}

fn default_mode() -> String {
    "both".to_string()
}

fn default_listen_address() -> String {
    "0.0.0.0:3080".to_string()
}

fn default_upstream_base_url() -> String {
    "https://api.anthropic.com".to_string()
}

fn default_true() -> bool {
    true
}

fn default_primary_timeout() -> u64 {
    300
}

fn default_litellm_url() -> String {
    "http://localhost:4000/v1/chat/completions".to_string()
}

fn default_shadow_timeout() -> u64 {
    120
}

fn default_max_concurrent() -> usize {
    50
}

impl ProxyConfig {
    /// Load configuration from TOML file and environment variables.
    ///
    /// Priority (highest to lowest):
    /// 1. Environment variables (SHADOW_ prefix, __ for nesting)
    /// 2. TOML config file
    /// 3. Defaults
    pub fn load(config_path: &str) -> anyhow::Result<Self> {
        let mut config: ProxyConfig = Figment::new()
            .merge(Toml::file(config_path))
            .merge(Env::prefixed("SHADOW_").split("__"))
            .extract()?;

        // Direct env var overrides for sensitive values
        if let Ok(key) = std::env::var("SHADOW_LITELLM_API_KEY") {
            config.shadow.litellm_api_key = key;
        }
        if let Ok(url) = std::env::var("SHADOW_LITELLM_URL") {
            config.shadow.litellm_url = url;
        }

        Ok(config)
    }
}
