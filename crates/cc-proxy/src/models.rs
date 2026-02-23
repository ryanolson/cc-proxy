//! Model registry for routing requests to local targets or Anthropic.
//!
//! Built once at startup from config + CLI args. Immutable after construction
//! so the hot path is a lock-free HashMap lookup.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// A locally-served model definition from config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDef {
    /// Model identifier (must match the `model` field in API requests).
    pub id: String,

    /// Human-readable name for /v1/models responses.
    #[serde(default)]
    pub display_name: Option<String>,

    /// Target URL for this model. If None, uses the global `--target-url`.
    #[serde(default)]
    pub target_url: Option<String>,
}

/// Routing decision for a single request.
#[allow(dead_code)]
pub enum RouteTarget {
    /// Route to a local model at this base URL.
    Local {
        model_def: ModelDef,
        target_url: String,
    },
    /// Route to Anthropic passthrough.
    Anthropic,
}

/// Immutable registry of locally-served models.
///
/// Wrapped in `Arc` and stored in `AppState`. No locks — the HashMap is
/// read-only after construction.
#[derive(Clone)]
pub struct ModelRegistry {
    inner: Arc<Inner>,
}

struct Inner {
    /// Map from model ID to definition.
    models: HashMap<String, ModelDef>,
    /// Fallback target URL from `--target-url` for models that don't specify one.
    default_target_url: Option<String>,
}

impl ModelRegistry {
    /// Build a new registry from config model definitions and a default target URL.
    pub fn new(models: Vec<ModelDef>, default_target_url: Option<String>) -> Self {
        let map: HashMap<String, ModelDef> = models
            .into_iter()
            .map(|m| (m.id.clone(), m))
            .collect();

        Self {
            inner: Arc::new(Inner {
                models: map,
                default_target_url,
            }),
        }
    }

    /// Resolve a model name to a routing target.
    ///
    /// Returns `Local` if the model is in the registry (with its target URL,
    /// falling back to the default target URL). Returns `Anthropic` otherwise.
    pub fn resolve(&self, model_id: &str) -> RouteTarget {
        match self.inner.models.get(model_id) {
            Some(def) => {
                let url = def
                    .target_url
                    .as_deref()
                    .or(self.inner.default_target_url.as_deref());

                match url {
                    Some(u) => RouteTarget::Local {
                        model_def: def.clone(),
                        target_url: u.to_string(),
                    },
                    None => {
                        tracing::warn!(
                            model = model_id,
                            "Local model has no target_url and no --target-url default; falling back to Anthropic"
                        );
                        RouteTarget::Anthropic
                    }
                }
            }
            None => RouteTarget::Anthropic,
        }
    }

    /// List all locally-registered models (for /v1/models).
    pub fn list_models(&self) -> Vec<&ModelDef> {
        self.inner.models.values().collect()
    }

    /// Number of registered local models.
    pub fn len(&self) -> usize {
        self.inner.models.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_local_model_with_own_url() {
        let reg = ModelRegistry::new(
            vec![ModelDef {
                id: "glm-5-fp8".into(),
                display_name: Some("GLM-5 FP8".into()),
                target_url: Some("http://glm:8000".into()),
            }],
            None,
        );

        match reg.resolve("glm-5-fp8") {
            RouteTarget::Local { target_url, .. } => {
                assert_eq!(target_url, "http://glm:8000");
            }
            RouteTarget::Anthropic => panic!("expected Local"),
        }
    }

    #[test]
    fn resolve_local_model_with_default_url() {
        let reg = ModelRegistry::new(
            vec![ModelDef {
                id: "glm-5-fp8".into(),
                display_name: None,
                target_url: None,
            }],
            Some("http://default:8000".into()),
        );

        match reg.resolve("glm-5-fp8") {
            RouteTarget::Local { target_url, .. } => {
                assert_eq!(target_url, "http://default:8000");
            }
            RouteTarget::Anthropic => panic!("expected Local"),
        }
    }

    #[test]
    fn resolve_unknown_model_goes_to_anthropic() {
        let reg = ModelRegistry::new(
            vec![ModelDef {
                id: "glm-5-fp8".into(),
                display_name: None,
                target_url: Some("http://glm:8000".into()),
            }],
            None,
        );

        match reg.resolve("claude-sonnet-4-20250514") {
            RouteTarget::Anthropic => {}
            RouteTarget::Local { .. } => panic!("expected Anthropic"),
        }
    }

    #[test]
    fn resolve_local_model_no_urls_falls_back_to_anthropic() {
        let reg = ModelRegistry::new(
            vec![ModelDef {
                id: "glm-5-fp8".into(),
                display_name: None,
                target_url: None,
            }],
            None, // no default either
        );

        match reg.resolve("glm-5-fp8") {
            RouteTarget::Anthropic => {}
            RouteTarget::Local { .. } => panic!("expected Anthropic fallback"),
        }
    }

    #[test]
    fn list_models_returns_all() {
        let reg = ModelRegistry::new(
            vec![
                ModelDef { id: "a".into(), display_name: None, target_url: None },
                ModelDef { id: "b".into(), display_name: None, target_url: None },
            ],
            None,
        );
        assert_eq!(reg.len(), 2);
        assert_eq!(reg.list_models().len(), 2);
    }
}
