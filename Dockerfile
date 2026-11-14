FROM rust:latest
RUN apt-get update && apt-get install -y pkg-config libssl-dev python3 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY Cargo.toml Cargo.lock index.html ./
COPY src ./src
RUN cargo build --release && cp target/release/mon-ia-rust /app/mon-ia-rust && rm -rf target
RUN mkdir -p /app/sessions /app/documents
EXPOSE 10000
CMD ["./mon-ia-rust"]
