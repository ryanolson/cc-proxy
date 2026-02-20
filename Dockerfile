FROM rust:1-bookworm AS builder

WORKDIR /app
COPY . .
RUN cargo build --release --bin cc-proxy

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/cc-proxy /usr/local/bin/cc-proxy

ENV CC_PROXY_CONFIG=/etc/cc-proxy/cc-proxy.toml

EXPOSE 3080

ENTRYPOINT ["cc-proxy"]
