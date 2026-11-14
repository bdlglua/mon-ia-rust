FROM rust:1.85-slim-bookworm
WORKDIR /app
COPY Cargo.toml Cargo.lock index.html ./
COPY src ./src
RUN cargo build --release && cp target/release/mon-ia-rust /app/mon-ia-rust && rm -rf target
RUN mkdir -p /app/sessions /app/documents
EXPOSE 10000
CMD ["./mon-ia-rust"]
