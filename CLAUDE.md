# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
cargo build                          # Build all crates
cargo build --release                # Release build
cargo test                           # Run all tests
cargo test --package shadow-proxy    # Tests for proxy crate only
cargo test --package shadow-tracing  # Tests for tracing crate only
cargo test <test_name>               # Run a single test by name
cargo clippy --workspace             # Lint all crates
cargo fmt --all -- --check           # Check formatting
cargo fmt --all                      # Auto-format
```

The binary is `shadow-proxy`. Run it with:
```bash
cargo run --bin shadow-proxy -- --config shadow-proxy.toml
```

Config path can also be set via `SHADOW_PROXY_CONFIG` env var.

## Architecture

Shadow Proxy is a transparent proxy that intercepts Claude Code requests to Anthropic's API. It forwards requests unchanged to Anthropic (primary path) while simultaneously dispatching converted copies to alternative models via LiteLLM (shadow path) for comparison.

### Workspace Crates

- **`crates/shadow-proxy`** — Main binary. HTTP server, proxy logic, protocol conversion.
- **`crates/shadow-tracing`** — Reusable OpenTelemetry/OTLP tracing library.

### Request Flow

1. `POST /v1/messages` arrives → `server.rs` handler generates a correlation ID (UUID v4)
2. **Shadow path** (fire-and-forget): `ShadowDispatcher` spawns tokio tasks per model, converts Anthropic→OpenAI format, sends to LiteLLM. Failures are logged as warnings and never affect the primary path. Concurrency is bounded by a semaphore (`max_concurrent`, non-blocking `try_acquire`).
3. **Primary path** (synchronous): `proxy/primary.rs` streams raw bytes verbatim to/from Anthropic — no parsing, preserving SSE formatting and field ordering.
4. **Catch-all fallback**: Any request not matching `/v1/messages` or `/health` is forwarded to Anthropic unchanged (any HTTP method). No shadow dispatch. This covers `/v1/messages/count_tokens`, `/v1/models`, etc.

### Key Modules (shadow-proxy crate)

- `config.rs` — Config loading via Figment (TOML + env var overrides with `SHADOW_` prefix)
- `server.rs` — Axum routes (`/v1/messages`, `/health`), catch-all fallback, AppState, graceful shutdown
- `proxy/primary.rs` — Zero-fidelity byte forwarding to Anthropic (`forward_to_anthropic` for messages, `forward_raw` for catch-all), hop-by-hop header filtering
- `proxy/shadow.rs` — `ShadowDispatcher` with semaphore-bounded fire-and-forget dispatch
- `proxy/correlation.rs` — `x-shadow-request-id` header, UUID generation
- `convert/anthropic_to_openai.rs` — Anthropic Messages API → OpenAI chat completions conversion
- `convert/types.rs` — Serde types for Anthropic protocol (custom deserializers for system prompt variants)

### Key Modules (shadow-tracing crate)

- `otlp.rs` — OTLP exporter init (gRPC or HTTP), `TracingGuard` for RAII shutdown. Falls back to fmt-only if OTLP endpoint is unreachable.
- `spans.rs` — Macros: `proxy_request_span!`, `primary_forward_span!`, `shadow_request_span!`

### Design Principles

- Primary path must never be affected by shadow failures
- Separate HTTP clients with independent timeouts for primary (300s) and shadow (120s)
- Anthropic request parsed once, converted once, reused across all shadow models
- Non-blocking semaphore acquire — requests are skipped (not queued) when at capacity

## Configuration

All config in `shadow-proxy.toml` (K8s) or `shadow-proxy.local.toml` (local dev). Env vars override with `SHADOW_` prefix (e.g., `SHADOW_SERVER__LISTEN_ADDRESS`). Default listen address: `0.0.0.0:3080`. `upstream_base_url` defaults to `https://api.anthropic.com`.

## Docker

Multi-stage build with `rust:1.82` builder and `debian:bookworm-slim` runtime. Exposes port 3080.
