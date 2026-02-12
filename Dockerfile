FROM rust:1.82-bookworm AS builder

WORKDIR /app
COPY . .
RUN cargo build --release --bin shadow-proxy

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/shadow-proxy /usr/local/bin/shadow-proxy
COPY shadow-proxy.toml /etc/shadow-proxy/shadow-proxy.toml

ENV SHADOW_PROXY_CONFIG=/etc/shadow-proxy/shadow-proxy.toml

EXPOSE 3080

ENTRYPOINT ["shadow-proxy"]
