//! Anthropic -> OpenAI chat completions format conversion.
//!
//! Converts an Anthropic Messages API request (`serde_json::Value`) to an
//! OpenAI-compatible chat completions request, suitable for sending to LiteLLM.
//!
//! All parsing uses `serde_json::Value` — no typed Anthropic structs. This
//! makes the conversion resilient to unknown content block types (thinking,
//! server_tool_use, citations, etc.). Unknown block types are logged at info
//! level so they can be identified and explicitly handled in the future.

use anyhow::Result;
use serde_json::{json, Value};

/// Known content block types we can convert to OpenAI format.
const KNOWN_BLOCK_TYPES: &[&str] = &["text", "image", "tool_use", "tool_result"];

/// Convert an Anthropic Messages API request to OpenAI chat completions JSON.
///
/// Accepts a `serde_json::Value` (the raw parsed JSON body) so deserialization
/// never fails on unknown content block types.
pub fn anthropic_to_openai(req: &Value) -> Result<Value> {
    let mut messages = Vec::new();

    // Prepend system message if present (supports both string and block array forms)
    if let Some(system) = req.get("system") {
        let system_text = match system {
            Value::String(s) => s.clone(),
            Value::Array(blocks) => blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        if !system_text.is_empty() {
            messages.push(json!({
                "role": "system",
                "content": system_text
            }));
        }
    }

    // Convert each message
    if let Some(msgs) = req.get("messages").and_then(|v| v.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = msg.get("content");

            match content {
                // Plain string content
                Some(Value::String(text)) => {
                    messages.push(json!({ "role": role, "content": text }));
                }
                // Array of content blocks
                Some(Value::Array(blocks)) => {
                    if role == "user" {
                        convert_user_blocks(blocks, &mut messages);
                    } else {
                        convert_assistant_blocks(blocks, &mut messages);
                    }
                }
                _ => {}
            }
        }
    }

    // Build the request
    let model = req
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let max_tokens = req
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(4096);

    let mut request = json!({
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "stream": false,
    });

    // Optional fields
    if let Some(temp) = req.get("temperature") {
        request["temperature"] = temp.clone();
    }
    if let Some(top_p) = req.get("top_p") {
        request["top_p"] = top_p.clone();
    }
    if let Some(stop_seqs) = req.get("stop_sequences") {
        request["stop"] = stop_seqs.clone();
    }
    if let Some(tools) = req.get("tools").and_then(|v| v.as_array()) {
        request["tools"] = convert_tools(tools);
    }
    if let Some(tc) = req.get("tool_choice") {
        request["tool_choice"] = convert_tool_choice(tc);
    }

    Ok(request)
}

/// Convert user-role content blocks into chat completion messages.
fn convert_user_blocks(blocks: &[Value], messages: &mut Vec<Value>) {
    let mut text_parts = Vec::new();

    for block in blocks {
        let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match block_type {
            "text" => {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    text_parts.push(text.to_string());
                }
            }
            "tool_result" => {
                // Flush accumulated text first
                if !text_parts.is_empty() {
                    let combined = text_parts.join("");
                    messages.push(json!({
                        "role": "user",
                        "content": combined
                    }));
                    text_parts.clear();
                }

                let tool_use_id = block
                    .get("tool_use_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                // tool_result content can be a string or array of content blocks
                let content_str = match block.get("content") {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Array(arr)) => arr
                        .iter()
                        .filter_map(|b| {
                            if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                                b.get("text").and_then(|t| t.as_str()).map(String::from)
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(""),
                    _ => String::new(),
                };

                messages.push(json!({
                    "role": "tool",
                    "content": content_str,
                    "tool_call_id": tool_use_id
                }));
            }
            "image" => {
                text_parts.push("[image]".to_string());
            }
            other => {
                if !KNOWN_BLOCK_TYPES.contains(&other) && !other.is_empty() {
                    tracing::info!(
                        block_type = %other,
                        role = "user",
                        "Skipping unknown content block type in Anthropic->OpenAI conversion"
                    );
                }
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
fn convert_assistant_blocks(blocks: &[Value], messages: &mut Vec<Value>) {
    let mut text_content = String::new();
    let mut tool_calls = Vec::new();

    for block in blocks {
        let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match block_type {
            "text" => {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    text_content.push_str(text);
                }
            }
            "tool_use" => {
                let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let input = block.get("input").cloned().unwrap_or(json!({}));

                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": serde_json::to_string(&input).unwrap_or_default()
                    }
                }));
            }
            other => {
                if !KNOWN_BLOCK_TYPES.contains(&other) && !other.is_empty() {
                    tracing::info!(
                        block_type = %other,
                        role = "assistant",
                        "Skipping unknown content block type in Anthropic->OpenAI conversion"
                    );
                }
            }
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
fn convert_tools(tools: &[Value]) -> Value {
    let converted: Vec<Value> = tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                    "description": tool.get("description").cloned().unwrap_or(Value::Null),
                    "parameters": tool.get("input_schema").cloned().unwrap_or(json!({}))
                }
            })
        })
        .collect();
    json!(converted)
}

/// Convert Anthropic tool_choice to OpenAI-format tool_choice.
fn convert_tool_choice(tc: &Value) -> Value {
    let choice_type = tc.get("type").and_then(|v| v.as_str()).unwrap_or("auto");

    match choice_type {
        "auto" => json!("auto"),
        "any" => json!("required"),
        "none" => json!("none"),
        "tool" => {
            if let Some(name) = tc.get("name").and_then(|v| v.as_str()) {
                json!({
                    "type": "function",
                    "function": { "name": name }
                })
            } else {
                json!("auto")
            }
        }
        _ => json!("auto"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_request() -> Value {
        json!({
            "model": "claude-sonnet-4",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": true
        })
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
    fn test_system_prompt_string() {
        let mut req = simple_request();
        req["system"] = json!("You are helpful.");
        let oai = anthropic_to_openai(&req).unwrap();

        let msgs = oai["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "You are helpful.");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn test_system_prompt_blocks() {
        let mut req = simple_request();
        req["system"] = json!([
            {"type": "text", "text": "Block one"},
            {"type": "text", "text": "Block two"}
        ]);
        let oai = anthropic_to_openai(&req).unwrap();

        let msgs = oai["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "Block one\nBlock two");
    }

    #[test]
    fn test_tool_use_conversion() {
        let req = json!({
            "model": "test",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tool_123", "name": "get_weather", "input": {"location": "SF"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tool_123", "content": "72F and sunny"}
                ]}
            ]
        });

        let oai = anthropic_to_openai(&req).unwrap();
        let msgs = oai["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert!(msgs[1]["tool_calls"].is_array());
        assert_eq!(msgs[1]["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "tool_123");
    }

    #[test]
    fn test_tools_and_tool_choice_conversion() {
        let mut req = simple_request();
        req["tools"] = json!([{
            "name": "get_weather",
            "description": "Get weather info",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }]);
        req["tool_choice"] = json!({"type": "auto"});

        let oai = anthropic_to_openai(&req).unwrap();
        assert!(oai["tools"].is_array());
        assert_eq!(oai["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(oai["tool_choice"], "auto");
    }

    #[test]
    fn test_named_tool_choice() {
        let mut req = simple_request();
        req["tool_choice"] = json!({"type": "tool", "name": "search"});

        let oai = anthropic_to_openai(&req).unwrap();
        assert_eq!(oai["tool_choice"]["type"], "function");
        assert_eq!(oai["tool_choice"]["function"]["name"], "search");
    }

    #[test]
    fn test_stop_sequences() {
        let mut req = simple_request();
        req["stop_sequences"] = json!(["STOP", "END"]);

        let oai = anthropic_to_openai(&req).unwrap();
        let stop = oai["stop"].as_array().unwrap();
        assert_eq!(stop.len(), 2);
        assert_eq!(stop[0], "STOP");
    }

    #[test]
    fn test_optional_params() {
        let mut req = simple_request();
        req["temperature"] = json!(0.7);
        req["top_p"] = json!(0.9);

        let oai = anthropic_to_openai(&req).unwrap();
        let temp = oai["temperature"].as_f64().unwrap();
        assert!((temp - 0.7).abs() < 0.001, "temperature: {temp}");
        let top_p = oai["top_p"].as_f64().unwrap();
        assert!((top_p - 0.9).abs() < 0.001, "top_p: {top_p}");
    }

    #[test]
    fn test_image_block_placeholder() {
        let req = json!({
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this: "},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc123"}}
                ]
            }]
        });

        let oai = anthropic_to_openai(&req).unwrap();
        let msgs = oai["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["content"], "Look at this: [image]");
    }

    #[test]
    fn test_unknown_block_types_skipped() {
        let req = json!({
            "model": "test",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "citations", "citations": []},
                    {"type": "text", "text": " world"}
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "let me think..."},
                    {"type": "text", "text": "Here is my answer"},
                    {"type": "server_tool_use", "id": "st_1", "name": "web_search"}
                ]}
            ]
        });

        let oai = anthropic_to_openai(&req).unwrap();
        let msgs = oai["messages"].as_array().unwrap();

        // User message: text blocks joined, citations skipped
        assert_eq!(msgs[0]["content"], "Hello world");

        // Assistant message: only "text" extracted, thinking/server_tool_use skipped
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "Here is my answer");
    }

    #[test]
    fn test_tool_result_with_array_content() {
        let req = json!({
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "tool_1",
                    "content": [
                        {"type": "text", "text": "Result line 1"},
                        {"type": "text", "text": "Result line 2"}
                    ]
                }]
            }]
        });

        let oai = anthropic_to_openai(&req).unwrap();
        let msgs = oai["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "tool");
        assert_eq!(msgs[0]["content"], "Result line 1Result line 2");
    }
}
