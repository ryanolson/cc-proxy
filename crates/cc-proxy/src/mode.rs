//! Runtime proxy mode toggle.
//!
//! Controls which paths are active: Anthropic-only, target-only, or compare.
//! Lock-free atomic — mode is read on every request hot path.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Operating mode for the proxy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ProxyMode {
    #[serde(rename = "anthropic-only")]
    AnthropicOnly = 0,
    #[serde(rename = "target")]
    TargetOnly = 1,
    #[serde(rename = "compare")]
    Compare = 2,
}

impl ProxyMode {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => ProxyMode::AnthropicOnly,
            1 => ProxyMode::TargetOnly,
            2 => ProxyMode::Compare,
            _ => ProxyMode::Compare,
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
