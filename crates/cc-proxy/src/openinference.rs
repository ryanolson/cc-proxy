//! OpenInference semantic attribute helpers for Phoenix/Arize compatibility.
//!
//! Sets standard OpenInference attributes on OTel spans so Phoenix renders them
//! as proper LLM traces with visible I/O, token counts, and tool calls.
//!
//! All request parsing uses `serde_json::Value` to be resilient to unknown
//! content block types (thinking, citations, server_tool_use, etc.) that
//! would cause typed deserialization to fail.

use opentelemetry::{Key, Value};
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;

/// Helper to set a string attribute on a span.
fn set_str(span: &Span, key: impl Into<Key>, value: impl Into<String>) {
    span.set_attribute(key, Value::String(value.into().into()));
}

/// Helper to set an i64 attribute on a span.
fn set_i64(span: &Span, key: impl Into<Key>, value: i64) {
    span.set_attribute(key, Value::I64(value));
}

/// Extract clean text content from an Anthropic message object.
///
/// Handles both string content and array-of-blocks content. For blocks,
/// extracts only `text` type blocks — tool_result, tool_use, thinking etc.
/// are skipped to keep the display clean.
fn extract_message_text(msg: &serde_json::Value) -> String {
    if let Some(content) = msg.get("content") {
        if let Some(s) = content.as_str() {
            return s.to_string();
        }
        if let Some(blocks) = content.as_array() {
            let parts: Vec<&str> = blocks
                .iter()
                .filter_map(|b| {
                    if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                        b.get("text").and_then(|t| t.as_str())
                    } else {
                        None
                    }
                })
                .collect();
            return parts.join("\n");
        }
    }
    String::new()
}

/// Set OpenInference request attributes on the root span from raw JSON.
///
/// Extracts model, messages, tools, and invocation parameters from the
/// request body Value. Unknown fields/block types are silently skipped.
pub fn set_request_attributes(span: &Span, req: &serde_json::Value) {
    set_str(span, "openinference.span.kind", "LLM");
    set_str(span, "llm.system", "anthropic");

    if let Some(model) = req.get("model").and_then(|v| v.as_str()) {
        set_str(span, "llm.model_name", model);
    }

    // input.value = text content of the last user message (the actual prompt).
    // Using clean text instead of raw JSON avoids tool_result XML noise in Phoenix.
    if let Some(messages) = req.get("messages").and_then(|v| v.as_array()) {
        if let Some(last_user) = messages.iter().rev().find(|m| {
            m.get("role").and_then(|v| v.as_str()) == Some("user")
        }) {
            let input_text = extract_message_text(last_user);
            if !input_text.is_empty() {
                set_str(span, "input.value", input_text);
            }
        }
    }

    // llm.invocation_parameters
    let mut params = serde_json::Map::new();
    if let Some(v) = req.get("max_tokens") {
        params.insert("max_tokens".to_string(), v.clone());
    }
    if let Some(v) = req.get("temperature") {
        params.insert("temperature".to_string(), v.clone());
    }
    if let Some(v) = req.get("top_p") {
        params.insert("top_p".to_string(), v.clone());
    }
    if let Some(v) = req.get("top_k") {
        params.insert("top_k".to_string(), v.clone());
    }
    if let Some(v) = req.get("stop_sequences") {
        params.insert("stop_sequences".to_string(), v.clone());
    }
    if let Ok(params_str) = serde_json::to_string(&params) {
        set_str(span, "llm.invocation_parameters", params_str);
    }

    // Flatten input messages with system prompt as index 0
    let mut msg_idx: usize = 0;

    // System prompt can be a string or an array of {type: "text", text: "..."} blocks
    if let Some(system) = req.get("system") {
        let system_text = if let Some(s) = system.as_str() {
            Some(s.to_string())
        } else if let Some(arr) = system.as_array() {
            let parts: Vec<&str> = arr
                .iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join("\n"))
            }
        } else {
            None
        };

        if let Some(text) = system_text {
            let prefix = format!("llm.input_messages.{msg_idx}.message");
            set_str(span, format!("{prefix}.role"), "system");
            set_str(span, format!("{prefix}.content"), text);
            msg_idx += 1;
        }
    }

    if let Some(messages) = req.get("messages").and_then(|v| v.as_array()) {
        for msg in messages {
            let prefix = format!("llm.input_messages.{msg_idx}.message");

            if let Some(role) = msg.get("role").and_then(|v| v.as_str()) {
                set_str(span, format!("{prefix}.role"), role);
            }

            // Content: either a string or an array of content blocks
            if let Some(content) = msg.get("content") {
                if let Some(text) = content.as_str() {
                    set_str(span, format!("{prefix}.content"), text);
                } else if let Some(blocks) = content.as_array() {
                    let mut text_parts = Vec::new();
                    let mut tool_idx: usize = 0;

                    for block in blocks {
                        match block.get("type").and_then(|v| v.as_str()) {
                            Some("text") => {
                                if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                                    text_parts.push(t);
                                }
                            }
                            Some("tool_use") => {
                                let tc_prefix =
                                    format!("{prefix}.tool_calls.{tool_idx}.tool_call.function");
                                if let Some(name) = block.get("name").and_then(|v| v.as_str()) {
                                    set_str(span, format!("{tc_prefix}.name"), name);
                                }
                                if let Some(input) = block.get("input") {
                                    if let Ok(args) = serde_json::to_string(input) {
                                        set_str(span, format!("{tc_prefix}.arguments"), args);
                                    }
                                }
                                tool_idx += 1;
                            }
                            Some("tool_result") => {
                                if let Some(c) = block.get("content").and_then(|v| v.as_str()) {
                                    text_parts.push(c);
                                }
                            }
                            // Skip unknown block types (thinking, citations, etc.)
                            _ => {}
                        }
                    }

                    if !text_parts.is_empty() {
                        set_str(span, format!("{prefix}.content"), text_parts.join("\n"));
                    }
                }
            }

            msg_idx += 1;
        }
    }

    // llm.tools.N.tool.json_schema
    if let Some(tools) = req.get("tools").and_then(|v| v.as_array()) {
        for (i, tool) in tools.iter().enumerate() {
            if let Ok(schema_str) = serde_json::to_string(tool) {
                set_str(span, format!("llm.tools.{i}.tool.json_schema"), schema_str);
            }
        }
    }
}

/// Set OpenInference request attributes on a shadow span from an OpenAI-format body.
///
/// The shadow request has already been converted from Anthropic to OpenAI format,
/// so the message structure is `{role, content}` with optional `tool_calls`.
#[allow(dead_code)]
pub fn set_shadow_request_attributes(span: &Span, req: &serde_json::Value) {
    set_str(span, "openinference.span.kind", "LLM");

    if let Some(model) = req.get("model").and_then(|v| v.as_str()) {
        set_str(span, "llm.model_name", model);
    }

    // input.value = JSON string of messages array
    if let Some(messages) = req.get("messages") {
        if let Ok(messages_json) = serde_json::to_string(messages) {
            set_str(span, "input.value", messages_json);
        }
    }

    // llm.invocation_parameters
    let mut params = serde_json::Map::new();
    if let Some(v) = req.get("max_completion_tokens") {
        params.insert("max_completion_tokens".to_string(), v.clone());
    }
    if let Some(v) = req.get("temperature") {
        params.insert("temperature".to_string(), v.clone());
    }
    if let Some(v) = req.get("top_p") {
        params.insert("top_p".to_string(), v.clone());
    }
    if let Some(v) = req.get("stop") {
        params.insert("stop".to_string(), v.clone());
    }
    if let Ok(params_str) = serde_json::to_string(&params) {
        set_str(span, "llm.invocation_parameters", params_str);
    }

    // Flatten input messages into OpenInference llm.input_messages
    if let Some(msgs) = req.get("messages").and_then(|v| v.as_array()) {
        for (idx, msg) in msgs.iter().enumerate() {
            let prefix = format!("llm.input_messages.{idx}.message");

            if let Some(role) = msg.get("role").and_then(|v| v.as_str()) {
                set_str(span, format!("{prefix}.role"), role);
            }

            if let Some(content) = msg.get("content").and_then(|v| v.as_str()) {
                set_str(span, format!("{prefix}.content"), content);
            }

            // Tool calls (assistant messages)
            if let Some(tool_calls) = msg.get("tool_calls").and_then(|v| v.as_array()) {
                for (tc_idx, tc) in tool_calls.iter().enumerate() {
                    let tc_prefix = format!("{prefix}.tool_calls.{tc_idx}.tool_call.function");
                    if let Some(func) = tc.get("function") {
                        if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                            set_str(span, format!("{tc_prefix}.name"), name);
                        }
                        if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                            set_str(span, format!("{tc_prefix}.arguments"), args);
                        }
                    }
                }
            }
        }
    }

    // llm.tools
    if let Some(tools) = req.get("tools").and_then(|v| v.as_array()) {
        for (i, tool) in tools.iter().enumerate() {
            if let Ok(schema_str) = serde_json::to_string(tool) {
                set_str(span, format!("llm.tools.{i}.tool.json_schema"), schema_str);
            }
        }
    }
}

/// Set OpenInference response attributes on a shadow span from an OpenAI-format response.
///
/// Shadow responses are always non-streaming OpenAI chat completions.
#[allow(dead_code)]
pub fn set_shadow_response_attributes(span: &Span, response_body: &str) {
    let body: serde_json::Value = match serde_json::from_str(response_body) {
        Ok(v) => v,
        Err(_) => return,
    };

    set_str(span, "output.value", response_body);

    // Extract first choice
    let choice = match body.get("choices").and_then(|v| v.get(0)) {
        Some(c) => c,
        None => return,
    };

    if let Some(msg) = choice.get("message") {
        if let Some(role) = msg.get("role").and_then(|v| v.as_str()) {
            set_str(span, "llm.output_messages.0.message.role", role);
        }

        if let Some(content) = msg.get("content").and_then(|v| v.as_str()) {
            set_str(span, "llm.output_messages.0.message.content", content);
        }

        if let Some(tool_calls) = msg.get("tool_calls").and_then(|v| v.as_array()) {
            for (idx, tc) in tool_calls.iter().enumerate() {
                let tc_prefix =
                    format!("llm.output_messages.0.message.tool_calls.{idx}.tool_call.function");
                if let Some(func) = tc.get("function") {
                    if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                        set_str(span, format!("{tc_prefix}.name"), name);
                    }
                    if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                        set_str(span, format!("{tc_prefix}.arguments"), args);
                    }
                }
            }
        }
    }

    // Token counts (OpenAI format: prompt_tokens, completion_tokens)
    if let Some(usage) = body.get("usage") {
        if let Some(input) = usage.get("prompt_tokens").and_then(|v| v.as_i64()) {
            set_i64(span, "llm.token_count.prompt", input);
        }
        if let Some(output) = usage.get("completion_tokens").and_then(|v| v.as_i64()) {
            set_i64(span, "llm.token_count.completion", output);
        }
    }
}

/// Set OpenInference response attributes on the span from the full response bytes.
///
/// For non-streaming: parses as a single JSON object.
/// For streaming: reassembles SSE events into the response shape.
///
/// All parsing uses `serde_json::Value`; failures are logged and skipped.
pub fn set_response_attributes(span: &Span, response_bytes: &[u8], is_streaming: bool) {
    if is_streaming {
        set_streaming_response_attributes(span, response_bytes);
    } else {
        set_nonstreaming_response_attributes(span, response_bytes);
    }
}

/// Parsed response data extracted from either JSON or SSE response bodies.
/// Separated from span-setting so it can be tested independently.
#[derive(Debug, Default, PartialEq)]
struct ParsedResponse {
    role: Option<String>,
    text_content: String,
    tool_calls: Vec<ParsedToolCall>,
    input_tokens: Option<i64>,
    output_tokens: Option<i64>,
}

#[derive(Debug, PartialEq)]
struct ParsedToolCall {
    name: String,
    arguments: String,
}

/// Parse a non-streaming Anthropic JSON response body.
fn parse_nonstreaming_response(response_bytes: &[u8]) -> Option<ParsedResponse> {
    let body: serde_json::Value = serde_json::from_slice(response_bytes).ok()?;

    let role = body.get("role").and_then(|v| v.as_str()).map(String::from);

    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    if let Some(content_arr) = body.get("content").and_then(|v| v.as_array()) {
        for block in content_arr {
            match block.get("type").and_then(|v| v.as_str()) {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                        text_parts.push(text.to_string());
                    }
                }
                Some("tool_use") => {
                    let name = block
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let arguments = block
                        .get("input")
                        .and_then(|v| serde_json::to_string(v).ok())
                        .unwrap_or_default();
                    tool_calls.push(ParsedToolCall { name, arguments });
                }
                _ => {}
            }
        }
    }

    let (input_tokens, output_tokens) = if let Some(usage) = body.get("usage") {
        (
            usage.get("input_tokens").and_then(|v| v.as_i64()),
            usage.get("output_tokens").and_then(|v| v.as_i64()),
        )
    } else {
        (None, None)
    };

    Some(ParsedResponse {
        role,
        text_content: text_parts.join(""),
        tool_calls,
        input_tokens,
        output_tokens,
    })
}

/// Parse a streaming SSE Anthropic response body.
fn parse_streaming_response(response_bytes: &[u8]) -> Option<ParsedResponse> {
    let body_str = std::str::from_utf8(response_bytes).ok()?;

    let mut input_tokens: Option<i64> = None;
    let mut output_tokens: Option<i64> = None;
    let mut role: Option<String> = None;

    struct ContentBlock {
        block_type: String,
        text: String,
        tool_name: String,
        tool_args_json: String,
    }

    let mut blocks: Vec<ContentBlock> = Vec::new();

    for event_chunk in body_str.split("\n\n") {
        let mut event_type = None;
        let mut data_str = None;

        for line in event_chunk.lines() {
            if let Some(et) = line.strip_prefix("event: ") {
                event_type = Some(et.trim());
            } else if let Some(d) = line.strip_prefix("data: ") {
                data_str = Some(d.trim());
            }
        }

        let data_str = match data_str {
            Some(d) => d,
            None => continue,
        };

        let data: serde_json::Value = match serde_json::from_str(data_str) {
            Ok(v) => v,
            Err(_) => continue,
        };

        match event_type {
            Some("message_start") => {
                if let Some(msg) = data.get("message") {
                    if let Some(r) = msg.get("role").and_then(|v| v.as_str()) {
                        role = Some(r.to_string());
                    }
                    if let Some(usage) = msg.get("usage") {
                        if let Some(it) = usage.get("input_tokens").and_then(|v| v.as_i64()) {
                            input_tokens = Some(it);
                        }
                    }
                }
            }
            Some("content_block_start") => {
                if let Some(cb) = data.get("content_block") {
                    let bt = cb
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("text")
                        .to_string();
                    let tool_name = cb
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    blocks.push(ContentBlock {
                        block_type: bt,
                        text: String::new(),
                        tool_name,
                        tool_args_json: String::new(),
                    });
                }
            }
            Some("content_block_delta") => {
                if let Some(idx) = data.get("index").and_then(|v| v.as_u64()) {
                    let idx = idx as usize;
                    if let Some(block) = blocks.get_mut(idx) {
                        if let Some(delta) = data.get("delta") {
                            match delta.get("type").and_then(|v| v.as_str()) {
                                Some("text_delta") => {
                                    if let Some(t) = delta.get("text").and_then(|v| v.as_str()) {
                                        block.text.push_str(t);
                                    }
                                }
                                Some("input_json_delta") => {
                                    if let Some(pj) =
                                        delta.get("partial_json").and_then(|v| v.as_str())
                                    {
                                        block.tool_args_json.push_str(pj);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            Some("message_delta") => {
                if let Some(usage) = data.get("usage") {
                    if let Some(ot) = usage.get("output_tokens").and_then(|v| v.as_i64()) {
                        output_tokens = Some(ot);
                    }
                    if input_tokens.is_none() {
                        if let Some(it) = usage.get("input_tokens").and_then(|v| v.as_i64()) {
                            input_tokens = Some(it);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in &blocks {
        match block.block_type.as_str() {
            "text" => {
                if !block.text.is_empty() {
                    text_parts.push(block.text.as_str());
                }
            }
            "tool_use" => {
                tool_calls.push(ParsedToolCall {
                    name: block.tool_name.clone(),
                    arguments: block.tool_args_json.clone(),
                });
            }
            _ => {}
        }
    }

    Some(ParsedResponse {
        role,
        text_content: text_parts.join(""),
        tool_calls,
        input_tokens,
        output_tokens,
    })
}

/// Apply a ParsedResponse to a span as OpenInference attributes.
fn apply_response_to_span(span: &Span, parsed: &ParsedResponse) {
    if !parsed.text_content.is_empty() {
        set_str(span, "output.value", &parsed.text_content);
        set_str(
            span,
            "llm.output_messages.0.message.content",
            &parsed.text_content,
        );
    }
    if let Some(ref r) = parsed.role {
        set_str(span, "llm.output_messages.0.message.role", r.clone());
    }
    for (idx, tc) in parsed.tool_calls.iter().enumerate() {
        let prefix =
            format!("llm.output_messages.0.message.tool_calls.{idx}.tool_call.function");
        set_str(span, format!("{prefix}.name"), &tc.name);
        if !tc.arguments.is_empty() {
            set_str(span, format!("{prefix}.arguments"), &tc.arguments);
        }
    }
    if let Some(it) = parsed.input_tokens {
        set_i64(span, "llm.token_count.prompt", it);
    }
    if let Some(ot) = parsed.output_tokens {
        set_i64(span, "llm.token_count.completion", ot);
    }
}

fn set_nonstreaming_response_attributes(span: &Span, response_bytes: &[u8]) {
    match parse_nonstreaming_response(response_bytes) {
        Some(parsed) => apply_response_to_span(span, &parsed),
        None => {
            tracing::warn!("Failed to parse non-streaming response for OpenInference attributes");
        }
    }
}

fn set_streaming_response_attributes(span: &Span, response_bytes: &[u8]) {
    match parse_streaming_response(response_bytes) {
        Some(parsed) => apply_response_to_span(span, &parsed),
        None => {
            tracing::warn!("Non-UTF8 streaming response, skipping OpenInference attributes");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── extract_message_text ──────────────────────────────────────────

    #[test]
    fn extract_text_from_string_content() {
        let msg: serde_json::Value =
            serde_json::from_str(r#"{"role": "user", "content": "Hello world"}"#).unwrap();
        assert_eq!(extract_message_text(&msg), "Hello world");
    }

    #[test]
    fn extract_text_from_blocks_skips_tool_result() {
        let msg: serde_json::Value = serde_json::from_str(r#"{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "<is_displaying_contents>true</is_displaying_contents>"},
                {"type": "text", "text": "What did you find?"}
            ]
        }"#).unwrap();
        assert_eq!(extract_message_text(&msg), "What did you find?");
    }

    #[test]
    fn extract_text_from_blocks_skips_thinking() {
        let msg: serde_json::Value = serde_json::from_str(r#"{
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me consider..."},
                {"type": "text", "text": "Here's my answer."}
            ]
        }"#).unwrap();
        assert_eq!(extract_message_text(&msg), "Here's my answer.");
    }

    #[test]
    fn extract_text_empty_when_no_text_blocks() {
        let msg: serde_json::Value = serde_json::from_str(r#"{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "some result"}
            ]
        }"#).unwrap();
        assert_eq!(extract_message_text(&msg), "");
    }

    #[test]
    fn extract_text_joins_multiple_text_blocks() {
        let msg: serde_json::Value = serde_json::from_str(r#"{
            "role": "user",
            "content": [
                {"type": "text", "text": "First part"},
                {"type": "text", "text": "Second part"}
            ]
        }"#).unwrap();
        assert_eq!(extract_message_text(&msg), "First part\nSecond part");
    }

    // ── parse_nonstreaming_response ───────────────────────────────────

    #[test]
    fn nonstreaming_response_extracts_text_and_tokens() {
        let body = br#"{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello!"},
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}}
            ],
            "usage": {"input_tokens": 100, "output_tokens": 50}
        }"#;
        let parsed = parse_nonstreaming_response(body).unwrap();
        assert_eq!(parsed.role.as_deref(), Some("assistant"));
        assert_eq!(parsed.text_content, "Hello!");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "bash");
        assert_eq!(parsed.input_tokens, Some(100));
        assert_eq!(parsed.output_tokens, Some(50));
    }

    #[test]
    fn nonstreaming_response_text_is_clean_not_raw_json() {
        let body = br#"{
            "role": "assistant",
            "content": [{"type": "text", "text": "Clean text here"}],
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }"#;
        let parsed = parse_nonstreaming_response(body).unwrap();
        // Must be clean text, NOT the raw JSON body
        assert_eq!(parsed.text_content, "Clean text here");
        assert!(!parsed.text_content.contains(r#""type""#));
        assert!(!parsed.text_content.contains(r#""role""#));
    }

    #[test]
    fn nonstreaming_response_invalid_json_returns_none() {
        assert!(parse_nonstreaming_response(b"not json").is_none());
    }

    // ── parse_streaming_response ──────────────────────────────────────

    #[test]
    fn streaming_response_reassembles_text() {
        let body = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"role\":\"assistant\",\"usage\":{\"input_tokens\":25}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":10}}\n\n";
        let parsed = parse_streaming_response(body).unwrap();

        assert_eq!(parsed.role.as_deref(), Some("assistant"));
        // Must be reassembled text, NOT raw SSE body
        assert_eq!(parsed.text_content, "Hello world");
        assert!(!parsed.text_content.contains("event:"));
        assert!(!parsed.text_content.contains("data:"));
        assert_eq!(parsed.input_tokens, Some(25));
        assert_eq!(parsed.output_tokens, Some(10));
    }

    #[test]
    fn streaming_response_extracts_tool_use() {
        let body = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"role\":\"assistant\",\"usage\":{\"input_tokens\":50}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"bash\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"cmd\\\": \"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"ls\\\"}\"}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":20}}\n\n";
        let parsed = parse_streaming_response(body).unwrap();

        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "bash");
        assert_eq!(parsed.tool_calls[0].arguments, r#"{"cmd": "ls"}"#);
        assert_eq!(parsed.input_tokens, Some(50));
        assert_eq!(parsed.output_tokens, Some(20));
    }

    #[test]
    fn streaming_response_input_tokens_in_message_delta() {
        // Some models report input_tokens in message_delta, not message_start
        let body = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"role\":\"assistant\",\"usage\":{}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":42,\"output_tokens\":5}}\n\n";
        let parsed = parse_streaming_response(body).unwrap();

        assert_eq!(parsed.text_content, "Hi");
        assert_eq!(parsed.input_tokens, Some(42));
        assert_eq!(parsed.output_tokens, Some(5));
    }

    #[test]
    fn streaming_response_invalid_utf8_returns_none() {
        assert!(parse_streaming_response(&[0xff, 0xfe]).is_none());
    }

    #[test]
    fn streaming_response_bad_json_events_skipped() {
        let body = b"event: message_start\ndata: {bad json}\n\nevent: content_block_delta\ndata: also bad\n\n";
        let parsed = parse_streaming_response(body).unwrap();
        // Should parse successfully but find nothing useful
        assert_eq!(parsed.text_content, "");
        assert!(parsed.input_tokens.is_none());
    }

    // ── set_request_attributes (smoke tests — span setting still needs OTel) ──

    #[test]
    fn set_request_attributes_no_panic() {
        let json = r#"{
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 8096,
            "system": "Be helpful",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Hi there!"},
                    {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}}
                ]}
            ],
            "tools": [{"name": "bash", "description": "Run bash", "input_schema": {"type": "object"}}],
            "temperature": 0.7,
            "stream": true
        }"#;
        let req: serde_json::Value = serde_json::from_str(json).unwrap();
        let span = tracing::info_span!("test");
        set_request_attributes(&span, &req);
    }

    #[test]
    fn set_request_attributes_unknown_blocks_no_panic() {
        let json = r#"{
            "model": "claude-opus-4-6",
            "max_tokens": 16384,
            "system": [{"type": "text", "text": "You are helpful."}],
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Let me consider..."},
                    {"type": "text", "text": "Hi!"},
                    {"type": "server_tool_use", "id": "st1", "name": "web_search", "input": {}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "st1", "content": "results here"},
                    {"type": "text", "text": "What did you find?"}
                ]}
            ],
            "stream": true
        }"#;
        let req: serde_json::Value = serde_json::from_str(json).unwrap();
        let span = tracing::info_span!("test");
        set_request_attributes(&span, &req);
    }

    #[test]
    fn set_response_attributes_no_panic() {
        let body = br#"{"role": "assistant", "content": [{"type": "text", "text": "Hi"}], "usage": {"input_tokens": 10, "output_tokens": 5}}"#;
        let span = tracing::info_span!("test");
        set_response_attributes(&span, body, false);
    }

    #[test]
    fn set_response_attributes_invalid_no_panic() {
        let span = tracing::info_span!("test");
        set_response_attributes(&span, b"not json", false);
        set_response_attributes(&span, &[0xff, 0xfe], true);
    }

    // ── shadow request/response (smoke tests) ─────────────────────────

    #[test]
    fn shadow_request_no_panic() {
        let json = r#"{"model": "deepseek-v3.2", "messages": [{"role": "user", "content": "Hello"}]}"#;
        let req: serde_json::Value = serde_json::from_str(json).unwrap();
        let span = tracing::info_span!("test");
        set_shadow_request_attributes(&span, &req);
    }

    #[test]
    fn shadow_response_no_panic() {
        let body = r#"{"choices": [{"message": {"role": "assistant", "content": "Hi"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}"#;
        let span = tracing::info_span!("test");
        set_shadow_response_attributes(&span, body);
    }

    #[test]
    fn shadow_response_invalid_no_panic() {
        let span = tracing::info_span!("test");
        set_shadow_response_attributes(&span, "not json");
    }
}
