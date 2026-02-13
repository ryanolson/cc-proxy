//! OpenInference semantic attribute helpers for Phoenix/Arize compatibility.
//!
//! Sets standard OpenInference attributes on OTel spans so Phoenix renders them
//! as proper LLM traces with visible I/O, token counts, and tool calls.

use opentelemetry::{Key, Value};
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::convert::types::{
    AnthropicContentBlock, AnthropicCreateMessageRequest, AnthropicMessageContent, AnthropicRole,
};

/// Helper to set a string attribute on a span.
fn set_str(span: &Span, key: impl Into<Key>, value: impl Into<String>) {
    span.set_attribute(key, Value::String(value.into().into()));
}

/// Helper to set an i64 attribute on a span.
fn set_i64(span: &Span, key: impl Into<Key>, value: i64) {
    span.set_attribute(key, Value::I64(value));
}

/// Set OpenInference request attributes on the current span.
///
/// Called before sending to Anthropic with the parsed request body.
pub fn set_request_attributes(span: &Span, req: &AnthropicCreateMessageRequest) {
    set_str(span, "openinference.span.kind", "LLM");
    set_str(span, "llm.system", "anthropic");
    set_str(span, "llm.model_name", req.model.clone());

    // input.value = JSON string of messages
    if let Ok(messages_json) = serde_json::to_string(&req.messages) {
        set_str(span, "input.value", messages_json);
    }

    // llm.invocation_parameters
    let mut params = serde_json::Map::new();
    params.insert(
        "max_tokens".to_string(),
        serde_json::Value::Number(req.max_tokens.into()),
    );
    if let Some(temp) = req.temperature {
        if let Some(n) = serde_json::Number::from_f64(temp as f64) {
            params.insert("temperature".to_string(), serde_json::Value::Number(n));
        }
    }
    if let Some(top_p) = req.top_p {
        if let Some(n) = serde_json::Number::from_f64(top_p as f64) {
            params.insert("top_p".to_string(), serde_json::Value::Number(n));
        }
    }
    if let Some(top_k) = req.top_k {
        params.insert(
            "top_k".to_string(),
            serde_json::Value::Number(top_k.into()),
        );
    }
    if let Some(ref stop) = req.stop_sequences {
        if let Ok(v) = serde_json::to_value(stop) {
            params.insert("stop_sequences".to_string(), v);
        }
    }
    if let Ok(params_str) = serde_json::to_string(&params) {
        set_str(span, "llm.invocation_parameters", params_str);
    }

    // Flatten input messages with system prompt as index 0
    let mut msg_idx: usize = 0;

    if let Some(ref system) = req.system {
        let prefix = format!("llm.input_messages.{msg_idx}.message");
        set_str(span, format!("{prefix}.role"), "system");
        set_str(span, format!("{prefix}.content"), system.clone());
        msg_idx += 1;
    }

    for msg in &req.messages {
        let prefix = format!("llm.input_messages.{msg_idx}.message");
        let role = match msg.role {
            AnthropicRole::User => "user",
            AnthropicRole::Assistant => "assistant",
        };
        set_str(span, format!("{prefix}.role"), role);

        match &msg.content {
            AnthropicMessageContent::Text { content } => {
                set_str(span, format!("{prefix}.content"), content.clone());
            }
            AnthropicMessageContent::Blocks { content } => {
                let mut text_parts = Vec::new();
                let mut tool_idx: usize = 0;

                for block in content {
                    match block {
                        AnthropicContentBlock::Text { text } => {
                            text_parts.push(text.as_str());
                        }
                        AnthropicContentBlock::ToolUse { name, input, .. } => {
                            let tc_prefix = format!(
                                "{prefix}.tool_calls.{tool_idx}.tool_call.function"
                            );
                            set_str(span, format!("{tc_prefix}.name"), name.clone());
                            if let Ok(args) = serde_json::to_string(input) {
                                set_str(span, format!("{tc_prefix}.arguments"), args);
                            }
                            tool_idx += 1;
                        }
                        AnthropicContentBlock::ToolResult {
                            content: Some(c), ..
                        } => {
                            text_parts.push(c.as_str());
                        }
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

    // llm.tools.N.tool.json_schema
    if let Some(ref tools) = req.tools {
        for (i, tool) in tools.iter().enumerate() {
            if let Ok(schema_str) = serde_json::to_string(tool) {
                set_str(span, format!("llm.tools.{i}.tool.json_schema"), schema_str);
            }
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

fn set_nonstreaming_response_attributes(span: &Span, response_bytes: &[u8]) {
    let body: serde_json::Value = match serde_json::from_slice(response_bytes) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to parse non-streaming response for OpenInference attributes");
            return;
        }
    };

    // output.value = full response body string
    if let Ok(body_str) = std::str::from_utf8(response_bytes) {
        set_str(span, "output.value", body_str);
    }

    // Role
    if let Some(role) = body.get("role").and_then(|v| v.as_str()) {
        set_str(span, "llm.output_messages.0.message.role", role);
    }

    // Content blocks
    if let Some(content_arr) = body.get("content").and_then(|v| v.as_array()) {
        let mut text_parts = Vec::new();
        let mut tool_idx: usize = 0;

        for block in content_arr {
            match block.get("type").and_then(|v| v.as_str()) {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                        text_parts.push(text.to_string());
                    }
                }
                Some("tool_use") => {
                    let tc_prefix = format!(
                        "llm.output_messages.0.message.tool_calls.{tool_idx}.tool_call.function"
                    );
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
                _ => {}
            }
        }

        if !text_parts.is_empty() {
            set_str(
                span,
                "llm.output_messages.0.message.content",
                text_parts.join(""),
            );
        }
    }

    // Token counts
    if let Some(usage) = body.get("usage") {
        if let Some(input) = usage.get("input_tokens").and_then(|v| v.as_i64()) {
            set_i64(span, "llm.token_count.prompt", input);
        }
        if let Some(output) = usage.get("output_tokens").and_then(|v| v.as_i64()) {
            set_i64(span, "llm.token_count.completion", output);
        }
    }
}

fn set_streaming_response_attributes(span: &Span, response_bytes: &[u8]) {
    let body_str = match std::str::from_utf8(response_bytes) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "Non-UTF8 streaming response, skipping OpenInference attributes");
            return;
        }
    };

    // Track accumulated state across SSE events
    let mut input_tokens: Option<i64> = None;
    let mut output_tokens: Option<i64> = None;
    let mut role: Option<String> = None;

    struct ContentBlock {
        block_type: String,
        text: String,
        tool_name: Option<String>,
        tool_args_json: String,
    }

    let mut blocks: Vec<ContentBlock> = Vec::new();

    // Parse SSE events: split on double newlines
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
                    let tool_name = if bt == "tool_use" {
                        cb.get("name")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    } else {
                        None
                    };
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
                                    if let Some(t) =
                                        delta.get("text").and_then(|v| v.as_str())
                                    {
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
                }
            }
            _ => {}
        }
    }

    // Set span attributes from accumulated state
    set_str(span, "output.value", body_str);

    if let Some(r) = role {
        set_str(span, "llm.output_messages.0.message.role", r);
    }

    let mut text_parts = Vec::new();
    let mut tool_call_idx: usize = 0;

    for block in &blocks {
        match block.block_type.as_str() {
            "text" => {
                if !block.text.is_empty() {
                    text_parts.push(block.text.as_str());
                }
            }
            "tool_use" => {
                let tc_prefix = format!(
                    "llm.output_messages.0.message.tool_calls.{tool_call_idx}.tool_call.function"
                );
                if let Some(ref name) = block.tool_name {
                    set_str(span, format!("{tc_prefix}.name"), name.clone());
                }
                if !block.tool_args_json.is_empty() {
                    set_str(
                        span,
                        format!("{tc_prefix}.arguments"),
                        block.tool_args_json.clone(),
                    );
                }
                tool_call_idx += 1;
            }
            _ => {}
        }
    }

    if !text_parts.is_empty() {
        set_str(
            span,
            "llm.output_messages.0.message.content",
            text_parts.join(""),
        );
    }

    if let Some(it) = input_tokens {
        set_i64(span, "llm.token_count.prompt", it);
    }
    if let Some(ot) = output_tokens {
        set_i64(span, "llm.token_count.completion", ot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_request_attributes_no_panic() {
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
            "top_p": 0.9,
            "stream": true
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        let span = tracing::info_span!("test");
        set_request_attributes(&span, &req);
    }

    #[test]
    fn test_set_nonstreaming_response_no_panic() {
        let body = br#"{
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello!"},
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}}
            ],
            "usage": {"input_tokens": 100, "output_tokens": 50}
        }"#;
        let span = tracing::info_span!("test");
        set_response_attributes(&span, body, false);
    }

    #[test]
    fn test_set_streaming_response_no_panic() {
        let body = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"role\":\"assistant\",\"usage\":{\"input_tokens\":25}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":10}}\n\n";
        let span = tracing::info_span!("test");
        set_response_attributes(&span, body, true);
    }

    #[test]
    fn test_streaming_tool_use_no_panic() {
        let body = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"role\":\"assistant\",\"usage\":{\"input_tokens\":50}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"bash\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"cmd\\\": \"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"ls\\\"}\"}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":20}}\n\n";
        let span = tracing::info_span!("test");
        set_response_attributes(&span, body, true);
    }

    #[test]
    fn test_invalid_json_response_no_panic() {
        let body = b"not json at all";
        let span = tracing::info_span!("test");
        set_response_attributes(&span, body, false);
    }

    #[test]
    fn test_invalid_sse_events_no_panic() {
        let body = b"event: message_start\ndata: {bad json}\n\nevent: content_block_delta\ndata: also bad\n\n";
        let span = tracing::info_span!("test");
        set_response_attributes(&span, body, true);
    }
}
