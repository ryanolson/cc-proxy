//! Anthropic -> OpenAI chat completions format conversion.
//!
//! Converts an `AnthropicCreateMessageRequest` to a `serde_json::Value`
//! representing an OpenAI-compatible chat completions request, suitable
//! for sending to LiteLLM.

use anyhow::Result;
use serde_json::{json, Value};

use super::types::*;

/// Convert an Anthropic Messages API request to OpenAI chat completions JSON.
///
/// The output targets `serde_json::Value` to avoid depending on typed OpenAI
/// structs. LiteLLM accepts valid OpenAI-format JSON.
pub fn anthropic_to_openai(req: &AnthropicCreateMessageRequest) -> Result<Value> {
    let mut messages = Vec::new();

    // Prepend system message if present
    if let Some(system_text) = &req.system {
        messages.push(json!({
            "role": "system",
            "content": system_text
        }));
    }

    // Convert each message
    for msg in &req.messages {
        match (&msg.role, &msg.content) {
            // User with plain text
            (AnthropicRole::User, AnthropicMessageContent::Text { content }) => {
                messages.push(json!({
                    "role": "user",
                    "content": content
                }));
            }
            // User with content blocks
            (AnthropicRole::User, AnthropicMessageContent::Blocks { content: blocks }) => {
                convert_user_blocks(blocks, &mut messages);
            }
            // Assistant with plain text
            (AnthropicRole::Assistant, AnthropicMessageContent::Text { content }) => {
                messages.push(json!({
                    "role": "assistant",
                    "content": content
                }));
            }
            // Assistant with content blocks (may contain tool_use)
            (AnthropicRole::Assistant, AnthropicMessageContent::Blocks { content: blocks }) => {
                convert_assistant_blocks(blocks, &mut messages);
            }
        }
    }

    // Build the request
    let mut request = json!({
        "model": req.model,
        "messages": messages,
        "max_completion_tokens": req.max_tokens,
        "stream": false,
    });

    // Optional fields
    if let Some(temp) = req.temperature {
        request["temperature"] = json!(temp);
    }
    if let Some(top_p) = req.top_p {
        request["top_p"] = json!(top_p);
    }
    if let Some(ref stop_seqs) = req.stop_sequences {
        request["stop"] = json!(stop_seqs);
    }
    if let Some(ref tools) = req.tools {
        request["tools"] = convert_tools(tools);
    }
    if let Some(ref tool_choice) = req.tool_choice {
        request["tool_choice"] = convert_tool_choice(tool_choice);
    }

    Ok(request)
}

/// Convert user-role content blocks into chat completion messages.
fn convert_user_blocks(blocks: &[AnthropicContentBlock], messages: &mut Vec<Value>) {
    let mut text_parts = Vec::new();

    for block in blocks {
        match block {
            AnthropicContentBlock::Text { text } => {
                text_parts.push(text.clone());
            }
            AnthropicContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                // Flush accumulated text first
                if !text_parts.is_empty() {
                    let combined = text_parts.join("");
                    messages.push(json!({
                        "role": "user",
                        "content": combined
                    }));
                    text_parts.clear();
                }
                messages.push(json!({
                    "role": "tool",
                    "content": content.clone().unwrap_or_default(),
                    "tool_call_id": tool_use_id
                }));
            }
            AnthropicContentBlock::Image { .. } => {
                text_parts.push("[image]".to_string());
            }
            AnthropicContentBlock::ToolUse { .. } => {
                // tool_use in a user message is unexpected, skip
            }
        }
    }

    // Flush remaining text
    if !text_parts.is_empty() {
        let combined = text_parts.join("");
        messages.push(json!({
            "role": "user",
            "content": combined
        }));
    }
}

/// Convert assistant-role content blocks into chat completion messages.
fn convert_assistant_blocks(blocks: &[AnthropicContentBlock], messages: &mut Vec<Value>) {
    let mut text_content = String::new();
    let mut tool_calls = Vec::new();

    for block in blocks {
        match block {
            AnthropicContentBlock::Text { text } => {
                text_content.push_str(text);
            }
            AnthropicContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": serde_json::to_string(input).unwrap_or_default()
                    }
                }));
            }
            _ => {}
        }
    }

    let mut msg = json!({ "role": "assistant" });

    if !text_content.is_empty() {
        msg["content"] = json!(text_content);
    }
    if !tool_calls.is_empty() {
        msg["tool_calls"] = json!(tool_calls);
    }

    messages.push(msg);
}

/// Convert Anthropic tools to OpenAI-format tools.
fn convert_tools(tools: &[AnthropicTool]) -> Value {
    let converted: Vec<Value> = tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            })
        })
        .collect();
    json!(converted)
}

/// Convert Anthropic tool_choice to OpenAI-format tool_choice.
fn convert_tool_choice(tc: &AnthropicToolChoice) -> Value {
    match tc {
        AnthropicToolChoice::Simple(simple) => match simple.choice_type {
            AnthropicToolChoiceMode::Auto => json!("auto"),
            AnthropicToolChoiceMode::Any => json!("required"),
            AnthropicToolChoiceMode::None => json!("none"),
            AnthropicToolChoiceMode::Tool => json!("auto"),
        },
        AnthropicToolChoice::Named(named) => {
            json!({
                "type": "function",
                "function": {
                    "name": named.name
                }
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_request() -> AnthropicCreateMessageRequest {
        AnthropicCreateMessageRequest {
            model: "claude-sonnet-4".into(),
            max_tokens: 1024,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
                content: AnthropicMessageContent::Text {
                    content: "Hello!".into(),
                },
            }],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: true,
            metadata: None,
            tools: None,
            tool_choice: None,
        }
    }

    #[test]
    fn test_simple_conversion() {
        let req = simple_request();
        let oai = anthropic_to_openai(&req).unwrap();

        assert_eq!(oai["model"], "claude-sonnet-4");
        assert_eq!(oai["max_completion_tokens"], 1024);
        assert_eq!(oai["stream"], false); // Always non-streaming for shadows
        assert_eq!(oai["messages"].as_array().unwrap().len(), 1);
        assert_eq!(oai["messages"][0]["role"], "user");
        assert_eq!(oai["messages"][0]["content"], "Hello!");
    }

    #[test]
    fn test_system_prompt_prepended() {
        let mut req = simple_request();
        req.system = Some("You are helpful.".into());
        let oai = anthropic_to_openai(&req).unwrap();

        let msgs = oai["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "You are helpful.");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn test_tool_use_conversion() {
        let req = AnthropicCreateMessageRequest {
            model: "test".into(),
            max_tokens: 100,
            messages: vec![
                AnthropicMessage {
                    role: AnthropicRole::User,
                    content: AnthropicMessageContent::Text {
                        content: "Weather?".into(),
                    },
                },
                AnthropicMessage {
                    role: AnthropicRole::Assistant,
                    content: AnthropicMessageContent::Blocks {
                        content: vec![AnthropicContentBlock::ToolUse {
                            id: "tool_123".into(),
                            name: "get_weather".into(),
                            input: serde_json::json!({"location": "SF"}),
                        }],
                    },
                },
                AnthropicMessage {
                    role: AnthropicRole::User,
                    content: AnthropicMessageContent::Blocks {
                        content: vec![AnthropicContentBlock::ToolResult {
                            tool_use_id: "tool_123".into(),
                            content: Some("72F and sunny".into()),
                            is_error: None,
                        }],
                    },
                },
            ],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
        };

        let oai = anthropic_to_openai(&req).unwrap();
        let msgs = oai["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert!(msgs[1]["tool_calls"].is_array());
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "tool_123");
    }

    #[test]
    fn test_tools_and_tool_choice_conversion() {
        let mut req = simple_request();
        req.tools = Some(vec![AnthropicTool {
            name: "get_weather".into(),
            description: Some("Get weather info".into()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }),
        }]);
        req.tool_choice = Some(AnthropicToolChoice::Simple(AnthropicToolChoiceSimple {
            choice_type: AnthropicToolChoiceMode::Auto,
        }));

        let oai = anthropic_to_openai(&req).unwrap();
        assert!(oai["tools"].is_array());
        assert_eq!(oai["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(oai["tool_choice"], "auto");
    }

    #[test]
    fn test_named_tool_choice() {
        let mut req = simple_request();
        req.tool_choice = Some(AnthropicToolChoice::Named(AnthropicToolChoiceNamed {
            choice_type: AnthropicToolChoiceMode::Tool,
            name: "search".into(),
        }));

        let oai = anthropic_to_openai(&req).unwrap();
        assert_eq!(oai["tool_choice"]["type"], "function");
        assert_eq!(oai["tool_choice"]["function"]["name"], "search");
    }

    #[test]
    fn test_stop_sequences() {
        let mut req = simple_request();
        req.stop_sequences = Some(vec!["STOP".into(), "END".into()]);

        let oai = anthropic_to_openai(&req).unwrap();
        let stop = oai["stop"].as_array().unwrap();
        assert_eq!(stop.len(), 2);
        assert_eq!(stop[0], "STOP");
    }

    #[test]
    fn test_optional_params() {
        let mut req = simple_request();
        req.temperature = Some(0.7);
        req.top_p = Some(0.9);

        let oai = anthropic_to_openai(&req).unwrap();
        let temp = oai["temperature"].as_f64().unwrap();
        assert!((temp - 0.7).abs() < 0.001, "temperature: {temp}");
        let top_p = oai["top_p"].as_f64().unwrap();
        assert!((top_p - 0.9).abs() < 0.001, "top_p: {top_p}");
    }

    #[test]
    fn test_image_block_placeholder() {
        let req = AnthropicCreateMessageRequest {
            model: "test".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
                content: AnthropicMessageContent::Blocks {
                    content: vec![
                        AnthropicContentBlock::Text {
                            text: "Look at this: ".into(),
                        },
                        AnthropicContentBlock::Image {
                            source: AnthropicImageSource {
                                source_type: "base64".into(),
                                media_type: "image/png".into(),
                                data: "abc123".into(),
                            },
                        },
                    ],
                },
            }],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
        };

        let oai = anthropic_to_openai(&req).unwrap();
        let msgs = oai["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["content"], "Look at this: [image]");
    }
}
