//! Anthropic Messages API types.
//!
//! Extracted from dynamo-llm's anthropic types for standalone use.
//!
//! Used at runtime by the validation module (`convert::validation`) which
//! attempts typed deserialization as a sidecar to detect Anthropic type drift.
//! The primary conversion path still uses `serde_json::Value` for resilience.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Custom deserializers
// ---------------------------------------------------------------------------

/// Deserialize `system` from either a plain string or an array of text blocks.
fn deserialize_system_prompt<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum SystemPrompt {
        Text(String),
        Blocks(Vec<SystemBlock>),
    }

    #[derive(Deserialize)]
    struct SystemBlock {
        text: String,
    }

    let maybe: Option<SystemPrompt> = Option::deserialize(deserializer)?;
    Ok(maybe.map(|sp| match sp {
        SystemPrompt::Text(s) => s,
        SystemPrompt::Blocks(blocks) => blocks
            .into_iter()
            .map(|b| b.text)
            .collect::<Vec<_>>()
            .join("\n"),
    }))
}

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// Top-level request body for `POST /v1/messages`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicCreateMessageRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<AnthropicMessage>,

    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_system_prompt"
    )]
    pub system: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    #[serde(default)]
    pub stream: bool,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
}

/// A single message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: AnthropicRole,
    #[serde(flatten)]
    pub content: AnthropicMessageContent,
}

/// The role of a message sender.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicRole {
    User,
    Assistant,
}

/// Message content — either a plain string or an array of content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
    Text { content: String },
    Blocks { content: Vec<AnthropicContentBlock> },
}

/// A single content block within a message.
///
/// Unknown block types (e.g. `thinking`, `server_tool_use`, `citations`) are
/// captured by the `Other` variant so deserialization never fails on new or
/// unexpected Anthropic content block types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: AnthropicImageSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<ToolResultContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    /// Catch-all for unknown content block types.
    #[serde(other)]
    Other,
}

/// Image source for image content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

/// Content of a `tool_result` block — either a plain string or an array of content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

/// A tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

/// Tool choice specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicToolChoice {
    Named(AnthropicToolChoiceNamed),
    Simple(AnthropicToolChoiceSimple),
}

/// Simple tool choice modes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicToolChoiceSimple {
    #[serde(rename = "type")]
    pub choice_type: AnthropicToolChoiceMode,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicToolChoiceMode {
    Auto,
    Any,
    None,
    Tool,
}

/// Named tool choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicToolChoiceNamed {
    #[serde(rename = "type")]
    pub choice_type: AnthropicToolChoiceMode,
    pub name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_simple_message() {
        let json =
            r#"{"model":"test","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "test");
        assert_eq!(req.max_tokens, 100);
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn test_deserialize_system_string() {
        let json = r#"{"model":"test","max_tokens":100,"system":"Be helpful","messages":[{"role":"user","content":"Hi"}]}"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.system, Some("Be helpful".to_string()));
    }

    #[test]
    fn test_deserialize_system_blocks() {
        let json = r#"{"model":"test","max_tokens":100,"system":[{"type":"text","text":"Block one"},{"type":"text","text":"Block two"}],"messages":[{"role":"user","content":"Hi"}]}"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.system, Some("Block one\nBlock two".to_string()));
    }

    #[test]
    fn test_deserialize_content_blocks() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "tool_result", "tool_use_id": "tool_1", "content": "result text"}
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        match &req.messages[0].content {
            AnthropicMessageContent::Blocks { content } => {
                assert_eq!(content.len(), 2);
            }
            _ => panic!("expected blocks content"),
        }
    }

    #[test]
    fn test_deserialize_unknown_content_blocks() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "let me think..."},
                    {"type": "text", "text": "Here is my answer"},
                    {"type": "server_tool_use", "id": "st_1", "name": "web_search"},
                    {"type": "citations", "citations": []}
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            AnthropicMessageContent::Blocks { content } => {
                assert_eq!(content.len(), 4);
                assert!(matches!(&content[0], AnthropicContentBlock::Other));
                assert!(matches!(&content[1], AnthropicContentBlock::Text { .. }));
                assert!(matches!(&content[2], AnthropicContentBlock::Other));
                assert!(matches!(&content[3], AnthropicContentBlock::Other));
            }
            _ => panic!("expected blocks content"),
        }
    }

    #[test]
    fn test_deserialize_tool_use() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "get_weather",
                    "input": {"location": "SF"}
                }]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            AnthropicMessageContent::Blocks { content } => {
                assert!(matches!(&content[0], AnthropicContentBlock::ToolUse { .. }));
            }
            _ => panic!("expected blocks content"),
        }
    }
}
