# cc-proxy

A model gateway that routes Claude Code traffic to self-hosted model deployments (GLM-5, MiniMax2.5) that speak the Anthropic Messages API format natively.

## Modes

| Mode | Behavior | Default |
|------|----------|---------|
| `target` | Forward all requests to the configured target. Return target response. | ✓ |
| `compare` | Forward to passthrough upstream (return response) + fire-and-forget to target for side-by-side logging. | |
| `anthropic-only` | Forward only to passthrough upstream. Requires `--allow-anthropic-only` at launch. | |

Modes can be toggled at runtime without restart:
```bash
curl -X PUT http://localhost:3080/api/mode \
  -H "Content-Type: application/json" \
  -d '{"mode":"compare"}'
```

## Quick Start

```bash
cargo build --release

# Route all Claude Code traffic to your target model
./target/release/cc-proxy \
  --config cc-proxy.toml \
  --target-url https://<your-model-endpoint> \
  --model glm-5-fp8

# Launch Claude Code through the proxy
ANTHROPIC_BASE_URL=http://localhost:3080 claude --model glm-5-fp8
```

## CLI Args

| Flag | Description |
|------|-------------|
| `--config <path>` | TOML config file (default: `cc-proxy.toml`, env: `CC_PROXY_CONFIG`) |
| `--target-url <url>` | Target endpoint. Not stored in config. |
| `--model <name>` | Force model for ALL requests, including subagents (Haiku/Sonnet rewritten). |
| `--allow-anthropic-only` | Required to enable `anthropic-only` mode at runtime. |

## Configuration

```toml
# cc-proxy.toml
default_mode = "target"

[server]
listen_address = "0.0.0.0:3080"

[target]
# url set via --target-url (not stored here)
timeout_secs = 300
max_concurrent = 50

[passthrough]
# Used only in `compare` and `anthropic-only` modes (requires --allow-anthropic-only)
url = "https://api.anthropic.com"
timeout_secs = 300
passthrough_auth = true

[tracing]
service_name = "cc-proxy"
# otlp_endpoint = "http://localhost:4317"  # optional — omit to disable OTLP
log_level = "info"
```

Env vars override with `CC_` prefix and `__` for nesting:
```bash
CC_SERVER__LISTEN_ADDRESS=0.0.0.0:3081
CC_DEFAULT_MODE=compare
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/messages` | Main proxy endpoint |
| `GET /health` | Health check |
| `GET /api/stats` | Token usage counters |
| `GET/PUT /api/mode` | Get or set runtime mode |
| `GET/PUT /api/tracing` | Toggle trace logging |
| `*` (fallback) | Any other path forwarded to passthrough upstream unchanged |

## Token Counting

`GET /api/stats` returns cumulative counters from the primary response path:
```json
{"total_requests": 42, "input_tokens": 12500, "output_tokens": 8300, "tool_calls": 7}
```

In `compare` mode, target tokens are logged in traces but not included in stats (stats reflect what was returned to the client).

## Tracing with Phoenix

cc-proxy exports OpenTelemetry spans to any OTLP collector. [Arize Phoenix](https://phoenix.arize.com) is the recommended local collector — it provides a UI for inspecting LLM traces with token counts, TTFT, and full message I/O.

### Start Phoenix

```bash
# Requires uv (https://docs.astral.sh/uv/)
tmux new-session -d -s phoenix
tmux send-keys -t phoenix 'uv run --with arize-phoenix phoenix serve' Enter
# UI at http://localhost:6006 — OTLP gRPC on :4317
```

### Start cc-proxy with OTLP enabled

Use `cc-proxy.local.toml` which has `otlp_endpoint = "http://localhost:4317"` configured:

```bash
./target/release/cc-proxy \
  --config cc-proxy.local.toml \
  --target-url https://<your-model-endpoint> \
  --model glm-5-fp8
```

Confirm OTLP connected — proxy logs should show:
```
INFO cc_tracing::otlp: OpenTelemetry OTLP tracing initialized endpoint=http://localhost:4317
```

### Span attributes

Each `proxy_request` root span includes:

| Attribute | Description |
|-----------|-------------|
| `ttft_ms` | Time to first token (ms from request send to first response chunk) |
| `total_duration_ms` | End-to-end streaming duration (ms) |
| `anthropic_request_id` | Upstream `x-request-id` header for cross-referencing |
| `llm.token_count.prompt` | Input token count |
| `llm.token_count.completion` | Output token count |
| `llm.input_messages` | Full request message array |
| `llm.output_messages` | Full response content |
| `llm.invocation_parameters` | max_tokens, temperature, top_p |

### Export traces programmatically

```python
# uv run --with arize-phoenix python3 export_traces.py
import phoenix as px
import json

client = px.Client(endpoint="http://localhost:6006")
df = client.get_spans_dataframe(project_name="default")

# Filter to root spans only
root = df[df["name"] == "proxy_request"]
root.to_json("traces.json", orient="records", date_format="iso", indent=2)
print(f"Exported {len(root)} traces")
```

## Building

```bash
cargo build --release          # release (use this for running)
cargo build                    # debug (fast iteration)
cargo test --workspace         # tests
cargo clippy --workspace       # lint
```
