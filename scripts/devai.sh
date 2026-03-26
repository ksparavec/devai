#!/bin/bash
set -e

# DevAI Lab launcher — run from any git repo
# Container name is derived from the git repository name
# Container is removed automatically on exit

usage() {
    cat <<'EOF'
Usage: devai.sh [--cpu] [--] [command...]

Start a DevAI Lab container for the current git repository.
Defaults to GPU image. Container is removed on exit.

Options:
  --cpu       Use CPU image instead of GPU
  --help      Show this help message

Environment variables (also read from .env in current directory):
  IMAGE_NAME            Base image name (default: devai-lab)
  OLLAMA_DEFAULT_MODEL  Model for ollama-chat (default: llama3.2)
  PORT                  Starting port for JupyterLab (auto-detected)
  HOME_VOLUME           Persistent home directory path
  HOST_HOME_DIR         Host home dir — enables .gitconfig/.ssh in container

Examples:
  devai.sh              # Start GPU container with JupyterLab
  devai.sh --cpu        # Start CPU container with JupyterLab
  devai.sh -- /bin/bash # Start GPU container with shell
EOF
    exit 0
}

# Parse arguments
USE_GPU=true
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)    USE_GPU=false; shift ;;
        --help)   usage ;;
        --)       shift; PASSTHROUGH_ARGS=("$@"); break ;;
        *)        PASSTHROUGH_ARGS=("$@"); break ;;
    esac
done

# Detect container runtime
if command -v podman &>/dev/null; then
    CONTAINER_RUNTIME=podman
elif command -v docker &>/dev/null; then
    CONTAINER_RUNTIME=docker
else
    echo "Error: neither podman nor docker found" >&2
    exit 1
fi

# Defaults (same as .env)
IMAGE_NAME=devai-lab
CONTAINER_USER=devai
PORT=8888
HOST_IP=$(hostname -I | awk '{print $1}')
HOST_HOME_DIR=$HOME
HOME_VOLUME=$HOME/devai-home
DEVAI_NETWORK=devai-net
OLLAMA_CONTAINER=devai-ollama
OLLAMA_HOST=http://${OLLAMA_CONTAINER}:11434

# Source .env from current directory (same as Makefile's -include .env)
[ -f .env ] && . .env

# Detect git repo name for unique container naming
REPO_NAME=$(git rev-parse --show-toplevel 2>/dev/null | xargs basename) || {
    echo "Error: not inside a git repository" >&2
    exit 1
}

# Select image and GPU flags
GPU_FLAGS=()
if [ "$USE_GPU" = true ]; then
    IMAGE_NAME="${IMAGE_NAME}-gpu"
    if [ "$CONTAINER_RUNTIME" = "podman" ]; then
        GPU_FLAGS+=(--device nvidia.com/gpu=all --security-opt=label=disable)
    else
        GPU_FLAGS+=(--gpus all)
    fi
fi
CONTAINER_NAME="${IMAGE_NAME}-${REPO_NAME}"

# Find first free port starting from configured default
find_free_port() {
    local port=${1:-8888}
    while ss -tln | grep -q ":${port} "; do
        port=$((port + 1))
    done
    echo "$port"
}
PORT=$(find_free_port "$PORT")

# User switching: only needed for docker (rootless podman root = host user)
USER_ENV=()
if [ "$CONTAINER_RUNTIME" != "podman" ]; then
    USER_ENV+=(-e "USER_ID=$(id -u)" -e "GROUP_ID=$(id -g)")
fi

# Proxy runtime env (same as Makefile PROXY_RUN_ENV)
PROXY_RUN_ENV=(
    -e "HTTP_PROXY=${HTTP_PROXY:-}"
    -e "HTTPS_PROXY=${HTTPS_PROXY:-}"
    -e "NO_PROXY=${NO_PROXY:-}"
    -e "http_proxy=${HTTP_PROXY:-}"
    -e "https_proxy=${HTTPS_PROXY:-}"
    -e "no_proxy=${NO_PROXY:-}"
)

# Mount host config files to staging dir for entrypoint to copy
# (same as Makefile HOME_MOUNT_ARG, conditional on HOST_HOME_DIR)
HOME_MOUNT_ARG=()
if [ -n "$HOST_HOME_DIR" ]; then
    [ -f "$HOME/.gitconfig" ] && HOME_MOUNT_ARG+=(-v "$HOME/.gitconfig:/tmp/host-config/.gitconfig:ro")
    [ -d "$HOME/.ssh" ] && HOME_MOUNT_ARG+=(-v "$HOME/.ssh:/tmp/host-config/.ssh:ro")
fi

WORK_DIR=$(readlink -f .)

JUPYTER_TOKEN="${JUPYTER_TOKEN:-devai}"
SSL_DIR="${HOME_VOLUME}/.jupyter/ssl"
if [ -f "$SSL_DIR/${HOST_IP}.pem" ] || [ -f "$SSL_DIR/cert.pem" ]; then
    PROTO=https
else
    PROTO=http
fi
echo "Starting ${CONTAINER_NAME}..."
echo "  Image:       ${IMAGE_NAME}"
echo "  Work dir:    ${WORK_DIR}"
echo "  JupyterLab:  ${PROTO}://${HOST_IP}:${PORT}/lab?token=${JUPYTER_TOKEN}"

exec "$CONTAINER_RUNTIME" run -it --rm \
    --name "$CONTAINER_NAME" \
    "${GPU_FLAGS[@]}" \
    --network "$DEVAI_NETWORK" \
    --add-host=host.containers.internal:host-gateway \
    "${PROXY_RUN_ENV[@]}" \
    -e "OLLAMA_HOST=${OLLAMA_HOST}" \
    -e "OLLAMA_URL=${OLLAMA_HOST}" \
    -e "OLLAMA_DEFAULT_MODEL=${OLLAMA_DEFAULT_MODEL:-llama3.2}" \
    -e "JUPYTER_TOKEN=${JUPYTER_TOKEN:-devai}" \
    "${USER_ENV[@]}" \
    -e "CONTAINER_USER=${CONTAINER_USER}" \
    -e "HOST_IP=${HOST_IP}" \
    -e "PORT=${PORT}" \
    -p "0.0.0.0:${PORT}:8888" \
    -v "${HOME_VOLUME}:/home/${CONTAINER_USER}" \
    "${HOME_MOUNT_ARG[@]}" \
    -v "${WORK_DIR}:/home/${CONTAINER_USER}/work" \
    "$IMAGE_NAME" \
    "${PASSTHROUGH_ARGS[@]}"
