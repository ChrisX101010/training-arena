#!/bin/bash
cd "$(dirname "$0")"
source ../venv/bin/activate 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if ! pgrep -f "ollama serve" > /dev/null 2>&1; then
    echo "Starting Ollama..."
    ollama serve > /dev/null 2>&1 &
    sleep 2
fi

python arena.py "$@"
