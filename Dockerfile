FROM rust:1.85-slim-bookworm

RUN apt-get update && apt-get install -y curl zstd ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://ollama.com | sh

WORKDIR /app
COPY Cargo.toml Cargo.lock index.html Modelfile start.sh ./
COPY src ./src

RUN cargo build --release && cp target/release/mon-ia-rust /app/mon-ia-rust && rm -rf target

ENV OLLAMA_MODELS=/tmp/ollama_models
RUN mkdir -p /app/sessions /app/documents && chmod -R 777 /app /tmp

EXPOSE 7860
CMD ["./start.sh"]
