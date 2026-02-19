//! Typed validation sidecar for detecting Anthropic type drift.
//!
//! Attempts typed deserialization as a **validation gate** (not a conversion path).
//! When unknown block types appear or the request shape diverges, emits structured
//! OTLP span attributes that are queryable in Phoenix.
//!
//! Two detection layers:
//! - **Layer 1 (high severity):** Typed parse fails — request shape has diverged
//! - **Layer 2 (medium severity):** Typed parse succeeds but contains `Other` variants

use opentelemetry::{Key, Value};
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use super::types::{AnthropicContentBlock, AnthropicCreateMessageRequest, AnthropicMessageContent};

/// Severity of a validation finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationSeverity {
    /// Typed parse completely failed — the request shape has diverged.
    High,
    /// Typed parse succeeded but contains unknown content block types.
    Medium,
}

impl ValidationSeverity {
    fn as_str(&self) -> &'static str {
        match self {
            ValidationSeverity::High => "high",
            ValidationSeverity::Medium => "medium",
        }
    }
}

/// A single validation finding.
#[derive(Debug, Clone)]
pub struct ValidationFinding {
    pub severity: ValidationSeverity,
    /// e.g. "typed_parse_failure" or "unknown_content_block"
    pub category: String,
    /// Human-readable description.
    pub message: String,
    /// The unknown block type name (if applicable).
    pub block_type: Option<String>,
    /// Index in the `messages` array where this was found.
    pub message_index: Option<usize>,
    /// Role of the message containing this finding.
    pub role: Option<String>,
}

/// Aggregated validation results for a single request.
#[derive(Debug)]
pub struct ValidationReport {
    /// Whether the typed parse succeeded at all.
    pub typed_parse_succeeded: bool,
    /// Individual findings.
    pub findings: Vec<ValidationFinding>,
    /// Deduplicated list of unknown block type names.
    pub unknown_block_types: Vec<String>,
}

/// Validate an Anthropic request by attempting typed deserialization and
/// cross-referencing with the raw `Value` to recover unknown type names.
pub fn validate_request(request_bytes: &[u8], raw_value: &serde_json::Value) -> ValidationReport {
    match serde_json::from_slice::<AnthropicCreateMessageRequest>(request_bytes) {
        Err(e) => {
            // Layer 1: typed parse completely failed
            ValidationReport {
                typed_parse_succeeded: false,
                findings: vec![ValidationFinding {
                    severity: ValidationSeverity::High,
                    category: "typed_parse_failure".to_string(),
                    message: format!("Typed deserialization failed: {e}"),
                    block_type: None,
                    message_index: None,
                    role: None,
                }],
                unknown_block_types: Vec::new(),
            }
        }
        Ok(typed) => {
            // Layer 2: typed parse succeeded — look for Other variants
            let mut findings = Vec::new();
            let mut unknown_types = Vec::new();

            for (msg_idx, msg) in typed.messages.iter().enumerate() {
                let blocks = match &msg.content {
                    AnthropicMessageContent::Blocks { content } => content,
                    AnthropicMessageContent::Text { .. } => continue,
                };

                for (block_idx, block) in blocks.iter().enumerate() {
                    if !matches!(block, AnthropicContentBlock::Other) {
                        continue;
                    }

                    // Cross-reference with the raw Value to recover the type name
                    let type_name = raw_value
                        .get("messages")
                        .and_then(|msgs| msgs.get(msg_idx))
                        .and_then(|m| m.get("content"))
                        .and_then(|c| c.get(block_idx))
                        .and_then(|b| b.get("type"))
                        .and_then(|t| t.as_str())
                        .unwrap_or("unknown")
                        .to_string();

                    let role = format!("{:?}", msg.role).to_lowercase();

                    findings.push(ValidationFinding {
                        severity: ValidationSeverity::Medium,
                        category: "unknown_content_block".to_string(),
                        message: format!(
                            "Unknown content block type \"{type_name}\" in {role} message at index {msg_idx}"
                        ),
                        block_type: Some(type_name.clone()),
                        message_index: Some(msg_idx),
                        role: Some(role),
                    });

                    if !unknown_types.contains(&type_name) {
                        unknown_types.push(type_name);
                    }
                }
            }

            ValidationReport {
                typed_parse_succeeded: true,
                findings,
                unknown_block_types: unknown_types,
            }
        }
    }
}

impl ValidationReport {
    /// Emit OTLP span attributes and tracing events for this report.
    pub fn emit(&self, span: &Span) {
        // Always set the parse status
        span.set_attribute(
            "shadow.validation.typed_parse_ok",
            Value::Bool(self.typed_parse_succeeded),
        );

        let finding_count = self.findings.len() as i64;
        span.set_attribute("shadow.validation.finding_count", Value::I64(finding_count));

        if !self.unknown_block_types.is_empty() {
            span.set_attribute(
                Key::from_static_str("shadow.validation.unknown_block_types"),
                Value::String(self.unknown_block_types.join(",").into()),
            );
        }

        // Max severity
        let max_severity = self
            .findings
            .iter()
            .map(|f| match f.severity {
                ValidationSeverity::High => 2,
                ValidationSeverity::Medium => 1,
            })
            .max();

        if let Some(sev) = max_severity {
            let sev_str = if sev >= 2 { "high" } else { "medium" };
            span.set_attribute(
                Key::from_static_str("shadow.validation.max_severity"),
                Value::String(sev_str.into()),
            );
        }

        // Structured findings as JSON
        if !self.findings.is_empty() {
            let findings_json: Vec<serde_json::Value> = self
                .findings
                .iter()
                .map(|f| {
                    let mut obj = serde_json::Map::new();
                    obj.insert(
                        "severity".to_string(),
                        serde_json::Value::String(f.severity.as_str().to_string()),
                    );
                    obj.insert(
                        "category".to_string(),
                        serde_json::Value::String(f.category.clone()),
                    );
                    obj.insert(
                        "message".to_string(),
                        serde_json::Value::String(f.message.clone()),
                    );
                    if let Some(ref bt) = f.block_type {
                        obj.insert(
                            "block_type".to_string(),
                            serde_json::Value::String(bt.clone()),
                        );
                    }
                    if let Some(idx) = f.message_index {
                        obj.insert(
                            "message_index".to_string(),
                            serde_json::Value::Number(serde_json::Number::from(idx)),
                        );
                    }
                    if let Some(ref role) = f.role {
                        obj.insert("role".to_string(), serde_json::Value::String(role.clone()));
                    }
                    serde_json::Value::Object(obj)
                })
                .collect();

            if let Ok(json_str) = serde_json::to_string(&findings_json) {
                span.set_attribute(
                    Key::from_static_str("shadow.validation.findings_json"),
                    Value::String(json_str.into()),
                );
            }
        }

        // Emit tracing events for each finding
        for finding in &self.findings {
            match finding.severity {
                ValidationSeverity::High => {
                    tracing::warn!(
                        category = %finding.category,
                        message = %finding.message,
                        block_type = finding.block_type.as_deref().unwrap_or(""),
                        message_index = finding.message_index,
                        role = finding.role.as_deref().unwrap_or(""),
                        "Validation finding (high severity)"
                    );
                }
                ValidationSeverity::Medium => {
                    tracing::info!(
                        category = %finding.category,
                        message = %finding.message,
                        block_type = finding.block_type.as_deref().unwrap_or(""),
                        message_index = finding.message_index,
                        role = finding.role.as_deref().unwrap_or(""),
                        "Validation finding (medium severity)"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_request_no_findings() {
        let json = r#"{
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]}
            ]
        }"#;
        let raw: serde_json::Value = serde_json::from_str(json).unwrap();
        let report = validate_request(json.as_bytes(), &raw);

        assert!(report.typed_parse_succeeded);
        assert!(report.findings.is_empty());
        assert!(report.unknown_block_types.is_empty());
    }

    #[test]
    fn test_unknown_content_blocks_detected() {
        let json = r#"{
            "model": "claude-opus-4-6",
            "max_tokens": 16384,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "let me consider..."},
                    {"type": "text", "text": "Here is my answer"},
                    {"type": "server_tool_use", "id": "st_1", "name": "web_search"},
                    {"type": "citations", "citations": []}
                ]
            }]
        }"#;
        let raw: serde_json::Value = serde_json::from_str(json).unwrap();
        let report = validate_request(json.as_bytes(), &raw);

        assert!(report.typed_parse_succeeded);
        assert_eq!(report.findings.len(), 3);
        assert_eq!(
            report.unknown_block_types,
            vec!["thinking", "server_tool_use", "citations"]
        );

        // All findings should be medium severity
        for finding in &report.findings {
            assert_eq!(finding.severity, ValidationSeverity::Medium);
            assert_eq!(finding.category, "unknown_content_block");
        }

        // First finding should have the correct type name and index
        assert_eq!(report.findings[0].block_type, Some("thinking".to_string()));
        assert_eq!(report.findings[0].message_index, Some(0));
        assert_eq!(report.findings[0].role, Some("assistant".to_string()));
    }

    #[test]
    fn test_unknown_blocks_deduplicated() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "..."},
                        {"type": "text", "text": "answer"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "more thinking..."},
                        {"type": "text", "text": "another answer"}
                    ]
                }
            ]
        }"#;
        let raw: serde_json::Value = serde_json::from_str(json).unwrap();
        let report = validate_request(json.as_bytes(), &raw);

        assert!(report.typed_parse_succeeded);
        assert_eq!(report.findings.len(), 2); // two instances
        assert_eq!(report.unknown_block_types, vec!["thinking"]); // deduplicated
    }

    #[test]
    fn test_string_content_no_findings() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hello, world!"}
            ]
        }"#;
        let raw: serde_json::Value = serde_json::from_str(json).unwrap();
        let report = validate_request(json.as_bytes(), &raw);

        assert!(report.typed_parse_succeeded);
        assert!(report.findings.is_empty());
    }

    #[test]
    fn test_invalid_request_high_severity() {
        // Missing required field "max_tokens"
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}]
        }"#;
        let raw: serde_json::Value = serde_json::from_str(json).unwrap();
        let report = validate_request(json.as_bytes(), &raw);

        assert!(!report.typed_parse_succeeded);
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].severity, ValidationSeverity::High);
        assert_eq!(report.findings[0].category, "typed_parse_failure");
        assert!(report.findings[0].message.contains("max_tokens"));
    }

    #[test]
    fn test_emit_no_panic() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "..."},
                    {"type": "text", "text": "answer"}
                ]
            }]
        }"#;
        let raw: serde_json::Value = serde_json::from_str(json).unwrap();
        let report = validate_request(json.as_bytes(), &raw);

        let span = tracing::info_span!("test_validation");
        report.emit(&span);
    }

    #[test]
    fn test_emit_clean_request_no_panic() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}]
        }"#;
        let raw: serde_json::Value = serde_json::from_str(json).unwrap();
        let report = validate_request(json.as_bytes(), &raw);

        let span = tracing::info_span!("test_validation_clean");
        report.emit(&span);
    }

    #[test]
    fn test_mixed_known_and_unknown_blocks() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "tool_result", "tool_use_id": "t1", "content": "result"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "..."},
                        {"type": "text", "text": "answer"},
                        {"type": "tool_use", "id": "t2", "name": "bash", "input": {}}
                    ]
                }
            ]
        }"#;
        let raw: serde_json::Value = serde_json::from_str(json).unwrap();
        let report = validate_request(json.as_bytes(), &raw);

        assert!(report.typed_parse_succeeded);
        // Only the "thinking" block is unknown
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].block_type, Some("thinking".to_string()));
        assert_eq!(report.findings[0].message_index, Some(1));
        assert_eq!(report.findings[0].role, Some("assistant".to_string()));
    }
}
