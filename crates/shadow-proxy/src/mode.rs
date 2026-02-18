//! Runtime proxy mode toggle.
//!
//! Controls which paths are active: Anthropic-only, shadow-only, or both.
//! Lock-free atomic — mode is read on every request hot path.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Operating mode for the proxy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[repr(u8)]
pub enum ProxyMode {
    AnthropicOnly = 0,
    ShadowOnly = 1,
    Both = 2,
}

impl ProxyMode {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => ProxyMode::AnthropicOnly,
            1 => ProxyMode::ShadowOnly,
            2 => ProxyMode::Both,
            _ => ProxyMode::Both,
        }
    }
}

/// Thread-safe runtime mode. Cheap to clone (Arc).
#[derive(Clone)]
pub struct RuntimeMode {
    inner: Arc<AtomicU8>,
}

impl RuntimeMode {
    pub fn new(mode: ProxyMode) -> Self {
        Self {
            inner: Arc::new(AtomicU8::new(mode as u8)),
        }
    }

    pub fn get(&self) -> ProxyMode {
        ProxyMode::from_u8(self.inner.load(Ordering::Relaxed))
    }

    pub fn set(&self, mode: ProxyMode) {
        self.inner.store(mode as u8, Ordering::Relaxed);
    }
}
