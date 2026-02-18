//! Atomic proxy statistics counters.
//!
//! Lock-free counters for tracking request volume and token usage.
//! All atomics use `Relaxed` ordering — these are monotonic display counters
//! with no synchronization requirements.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::Serialize;

struct StatsInner {
    total_requests: AtomicU64,
    input_tokens: AtomicU64,
    output_tokens: AtomicU64,
    tool_calls: AtomicU64,
}

/// Thread-safe atomic proxy statistics. Cheap to clone (Arc).
#[derive(Clone)]
pub struct ProxyStats {
    inner: Arc<StatsInner>,
}

/// Snapshot of current stats values, serializable to JSON.
#[derive(Debug, Serialize)]
pub struct StatsSnapshot {
    pub total_requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub tool_calls: u64,
}

impl ProxyStats {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(StatsInner {
                total_requests: AtomicU64::new(0),
                input_tokens: AtomicU64::new(0),
                output_tokens: AtomicU64::new(0),
                tool_calls: AtomicU64::new(0),
            }),
        }
    }

    pub fn inc_requests(&self) {
        self.inner.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_input_tokens(&self, n: u64) {
        self.inner.input_tokens.fetch_add(n, Ordering::Relaxed);
    }

    pub fn add_output_tokens(&self, n: u64) {
        self.inner.output_tokens.fetch_add(n, Ordering::Relaxed);
    }

    pub fn add_tool_calls(&self, n: u64) {
        self.inner.tool_calls.fetch_add(n, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            total_requests: self.inner.total_requests.load(Ordering::Relaxed),
            input_tokens: self.inner.input_tokens.load(Ordering::Relaxed),
            output_tokens: self.inner.output_tokens.load(Ordering::Relaxed),
            tool_calls: self.inner.tool_calls.load(Ordering::Relaxed),
        }
    }
}
