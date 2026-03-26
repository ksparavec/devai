#!/bin/bash
# Ollama model management helper for the devai-ollama container

usage() {
    cat <<'EOF'
Usage: ollama.sh <command> [args...]

Manage models and GPU in the devai-ollama container.

Model commands:
  pull <model>    Pull a model (e.g., ollama.sh pull qwen3.5:9b)
  list            List downloaded models
  rm <model>      Remove a model from disk
  loaded          Show models currently loaded in GPU
  load <model>    Load a model into GPU
  unload [model]  Unload model from GPU (all if no model specified)

GPU commands:
  gpu             Show GPU status (memory, power, utilization)
  top             Live GPU monitor (nvtop)

Container commands:
  status          Show container status (default)
  stop            Stop the Ollama container
  logs            Follow container logs

Examples:
  ollama.sh pull qwen3.5:27b
  ollama.sh load qwen3.5:9b
  ollama.sh unload
  ollama.sh gpu
  ollama.sh top
EOF
    exit 0
}

if command -v podman &>/dev/null; then
    RUNTIME=podman
elif command -v docker &>/dev/null; then
    RUNTIME=docker
else
    echo "Error: neither podman nor docker found" >&2
    exit 1
fi

OLLAMA_CONTAINER=devai-ollama

case "${1:-status}" in
    pull)   shift; $RUNTIME exec "$OLLAMA_CONTAINER" ollama pull "$@" ;;
    list)   $RUNTIME exec "$OLLAMA_CONTAINER" ollama list ;;
    rm)     shift; $RUNTIME exec "$OLLAMA_CONTAINER" ollama rm "$@" ;;
    loaded) $RUNTIME exec "$OLLAMA_CONTAINER" ollama ps ;;
    load)   shift; $RUNTIME exec "$OLLAMA_CONTAINER" ollama run "$1" /bye ;;
    unload)
        shift
        if [ -n "$1" ]; then
            $RUNTIME exec "$OLLAMA_CONTAINER" ollama stop "$1"
        else
            $RUNTIME exec "$OLLAMA_CONTAINER" sh -c \
                'ollama ps | tail -n +2 | awk "{print \$1}" | while read m; do [ -n "$m" ] && ollama stop "$m"; done'
        fi
        ;;
    gpu)    nvidia-smi ;;
    top)    nvtop ;;
    stop)   $RUNTIME stop "$OLLAMA_CONTAINER" ;;
    status) $RUNTIME inspect -f '{{.State.Status}}' "$OLLAMA_CONTAINER" 2>/dev/null || echo "not running" ;;
    logs)   $RUNTIME logs -f "$OLLAMA_CONTAINER" ;;
    --help|-h|help) usage ;;
    *)      usage ;;
esac
