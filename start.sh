#!/bin/bash
ollama serve &
sleep 4

ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama create bdl -f Modelfile

export PORT=7860
./mon-ia-rust
