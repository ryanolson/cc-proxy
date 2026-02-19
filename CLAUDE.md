# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
cargo build --release                # Release build (always use for running)
cargo build                          # Debug build (for fast iteration)
cargo test --workspace               # Run all tests
cargo test --package cc-proxy        # Tests for proxy crate only
cargo test --package cc-tracing      # Tests for tracing crate only
cargo test <test_name>               # Run a single test by name
cargo clippy --workspace             # Lint all crates
cargo fmt --all -- --check           # Check formatting
cargo fmt --all                      # Auto-format
```

The binary is `cc-proxy`. Always run from the repo root (config defaults to `cc-proxy.toml`):
```bash
./target/release/cc-proxy --config cc-proxy.toml \
  --target-url https://<glm-endpoint> --model glm-5-fp8
```

Config path can also be set via `CC_PROXY_CONFIG` env var.

## Architecture

cc-proxy is a **model gateway** that routes Claude Code traffic to self-hosted model deployments (GLM-5, MiniMax2.5) that speak the Anthropic Messages API format natively. No LiteLLM or format conversion is needed — the target receives and returns the same Anthropic-format JSON.

### Workspace Crates

- **`crates/cc-proxy`** — Main binary. HTTP server, proxy logic, mode routing.
- **`crates/cc-tracing`** — Reusable OpenTelemetry/OTLP tracing library.

### Operating Modes

| Mode | Behavior |
|------|----------|
| `target` | Forward all requests to configured target (GLM/MiniMax). Return target response. **Default.** |
| `compare` | Forward to passthrough upstream (return response) + fire-and-forget to target for comparison logging. |
| `anthropic-only` | Forward only to passthrough upstream. Requires `--allow-anthropic-only` at launch — returns 403 on mode switch if flag not set. |

Modes can be toggled at runtime via `PUT /api/mode` without restart. `anthropic-only` is blocked unless `--allow-anthropic-only` was passed at launch.

### CLI Arguments

| Flag | Description |
|------|-------------|
| `--config <path>` | Path to TOML config file (default: `cc-proxy.toml`) |
| `--target-url <url>` | Target endpoint (GLM/MiniMax). Sets `config.target.url`. Never committed to config. |
| `--model <name>` | Force model for ALL requests, including subagent requests (Haiku, Sonnet). |
| `--allow-anthropic-only` | Required to enable `anthropic-only` mode at runtime. |

### Request Flow

1. `POST /v1/messages` arrives → `server.rs` generates correlation ID (UUID v4)
2. `rewrite_request_body`: applies `--model` override (if set), sets default `max_tokens`/`temperature`/`top_p`
3. Checks runtime mode:
   - **`target`** (default): Forward raw bytes to `target_url/v1/messages`, return response directly
   - **`compare`**: Forward to passthrough upstream (synchronous) + fire-and-forget raw bytes to target for logging
   - **`anthropic-only`**: Forward to passthrough upstream only (gated behind `--allow-anthropic-only`)
4. **Catch-all fallback**: Unmatched routes forwarded to passthrough upstream unchanged (any HTTP method).

### Key Modules (cc-proxy crate)

- `config.rs` — `ProxyConfig` with `[target]`, `[passthrough]`, `[server]`, `[tracing]` sections
- `server.rs` — Axum routes, `AppState`, body rewriting, mode dispatch
- `proxy/primary.rs` — Zero-fidelity byte forwarding (`forward_to_anthropic`, `forward_to_target`, `forward_raw`)
- `proxy/compare.rs` — `CompareDispatcher`: fire-and-forget Anthropic-format POST to target URL
- `proxy/correlation.rs` — `x-shadow-request-id` header, UUID generation
- `mode.rs` — `ProxyMode` enum (TargetOnly, Compare, AnthropicOnly), lock-free `RuntimeMode` wrapper
- `convert/` — Anthropic↔OpenAI conversion (not in hot path — target speaks Anthropic natively)

### Key Modules (cc-tracing crate)

- `otlp.rs` — OTLP exporter init, `TracingGuard` RAII. Falls back to fmt-only if `otlp_endpoint` absent or unreachable.
- `spans.rs` — Macros: `proxy_request_span!`, `primary_forward_span!`, `compare_request_span!`

### Design Principles

- **`target` mode is the primary path** — passthrough (Anthropic) is only used in `compare`/`anthropic-only`
- `--target-url` and `--model` are CLI-only — not stored in config files, no secrets in TOML
- `--model` override applies to ALL requests including subagents (Haiku/Sonnet rewritten at request boundary)
- `anthropic-only` mode requires explicit opt-in at launch (`--allow-anthropic-only`)
- Compare dispatch uses same rewritten bytes as target path — no conversion needed

## Configuration

```toml
# cc-proxy.toml
default_mode = "target"

[server]
listen_address = "0.0.0.0:3080"

[target]
# url set via --target-url (not stored here — no secrets in config)
timeout_secs = 300
max_concurrent = 50

[passthrough]
# Used only in `compare` and `anthropic-only` modes (requires --allow-anthropic-only)
url = "https://api.anthropic.com"
timeout_secs = 300
passthrough_auth = true

[tracing]
service_name = "cc-proxy"
# otlp_endpoint = "http://localhost:4317"  # optional — omit to disable OTLP entirely
log_level = "info"
```

Env vars override with `CC_` prefix and `__` for nesting (e.g., `CC_SERVER__LISTEN_ADDRESS`).

## Verification

```bash
# Start proxy (target mode, model override active)
./target/release/cc-proxy --config cc-proxy.toml \
  --target-url https://<glm-endpoint> --model glm-5-fp8

# Verify default mode is target
curl -s http://localhost:3080/api/mode  # {"mode":"target"}

# Verify anthropic-only is blocked without flag
curl -X PUT http://localhost:3080/api/mode \
  -H "Content-Type: application/json" \
  -d '{"mode":"anthropic-only"}'
# Returns 403

# Run tests
cargo test --package cc-proxy
cargo clippy --workspace
```

## Integration Testing Workflow

This is the required workflow to validate the full proxy pipeline including model override and subagent redirection.

**1. Start the proxy** (tmux session `cc-proxy`, run from repo root):
```bash
cd /Users/mkosec/work/cc-proxy
./target/release/cc-proxy --config cc-proxy.toml \
  --target-url https://<glm-endpoint> --model glm-5-fp8
```

**2. Start Claude Code** (tmux session `claude-test`, must cd to cc-proxy first):
```bash
cd /Users/mkosec/work/cc-proxy
unset CLAUDECODE && ANTHROPIC_BASE_URL=http://localhost:3080 claude --model glm-5-fp8
```
- `unset CLAUDECODE` is required to bypass nested session guard when launched from another Claude Code session
- `--model glm-5-fp8` must be present — without it, subagent calls default to Anthropic models
- `ANTHROPIC_BASE_URL` redirects all Claude Code traffic through the proxy

**3. Validate model override:**
- Send a prompt — proxy logs should show `original_model=glm-5-fp8` and `target=vvagias-glm5.stg.astra.nvidia.com`
- Send an Explore prompt — this triggers parallel Haiku subagent calls; proxy logs must show all of them with `original_model=glm-5-fp8` (never `claude-haiku-*`)

**4. Verify no Anthropic leakage:**
- Proxy logs must contain only `proxy_request` / `primary_forward` entries
- There must be NO `passthrough_forward` lines — in `target` mode the fallback handler returns 404 locally; nothing hits `api.anthropic.com`

**5. Check token counts:**
```bash
curl -s http://localhost:3080/api/stats | python3 -m json.tool
# {"total_requests": N, "input_tokens": N, "output_tokens": N, "tool_calls": N}
```
- `input_tokens` will be non-zero (GLM reports these in `message_delta`, proxy handles both locations)
- `output_tokens` and `tool_calls` should reflect the Explore run (many parallel tool calls)

**Known behaviour:**
- `unset CLAUDECODE` is required when launching from inside an existing Claude Code session — without it the CLI refuses to start
- `--model glm-5-fp8` on the `claude` command sets the *initial* model; `--model` on the proxy rewrites ALL requests including subagents — both are needed
- GLM reports `input_tokens` in the `message_delta` SSE event, not `message_start` (unlike Anthropic)

## Tracing with Phoenix

OpenTelemetry spans are exported to any OTLP collector. [Arize Phoenix](https://phoenix.arize.com) is the recommended local collector — it provides a UI for inspecting LLM traces with token counts, TTFT, and full message I/O.

### 1. Start Phoenix (tmux session `phoenix`)

```bash
tmux new-session -d -s phoenix
tmux send-keys -t phoenix 'uv run --with arize-phoenix phoenix serve' Enter
# UI at http://localhost:6006 — OTLP gRPC on :4317
```

Confirm Phoenix is up:
```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:6006
# 200
```

### 2. Start cc-proxy with OTLP enabled

Use `cc-proxy.local.toml` which has `otlp_endpoint = "http://localhost:4317"` configured:

```bash
tmux new-session -d -s cc-proxy
tmux send-keys -t cc-proxy 'cd /Users/mkosec/work/cc-proxy-tracing && ./target/release/cc-proxy --config cc-proxy.local.toml --target-url https://<glm-endpoint> --model glm-5-fp8' Enter
```

Confirm OTLP connected — proxy logs should show:
```
INFO cc_tracing::otlp: OpenTelemetry OTLP tracing initialized endpoint=http://localhost:4317
```

### 3. Send traffic and inspect traces

Open the Phoenix UI at **http://localhost:6006** to browse spans. Each `proxy_request` root span includes `ttft_ms`, `total_duration_ms`, token counts, and full message I/O.

### 4. Export traces programmatically

```python
# uv run --with arize-phoenix python3 export_traces.py
import phoenix as px
import json

client = px.Client(endpoint="http://localhost:6006")
df = client.get_spans_dataframe(project_name="default")

root = df[df["name"] == "proxy_request"]
root.to_json("traces.json", orient="records", date_format="iso", indent=2)
print(f"Exported {len(root)} traces")
```

## Docker

Multi-stage build with `rust:1.82` builder and `debian:bookworm-slim` runtime. Exposes port 3080.
