//! Proxy routing: primary forwarding, shadow dispatch, compare dispatch, and correlation.

pub mod compare;
pub mod correlation;
pub mod primary;

// shadow.rs is retained for reference but no longer compiled —
// its types (ShadowDispatcher, ShadowConfig) have been replaced by
// CompareDispatcher + TargetConfig.
