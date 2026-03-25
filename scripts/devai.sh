#!/bin/bash
set -e

# DevAI Lab launcher — run from any git repo
# Container name is derived from the git repository name
# Container is removed automatically on exit

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
OLLAMA_HOST=http://host.containers.internal:11434

# Source .env from current directory (same as Makefile's -include .env)
[ -f .env ] && . .env

# Detect git repo name for unique container naming
REPO_NAME=$(git rev-parse --show-toplevel 2>/dev/null | xargs basename) || {
    echo "Error: not inside a git repository" >&2
    exit 1
}
# Use GPU image (same as Makefile run-gpu appends -gpu to IMAGE_NAME)
IMAGE_NAME="${IMAGE_NAME}-gpu"
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

# GPU runtime flags (same as Makefile GPU_FLAGS)
GPU_FLAGS=()
if [ "$CONTAINER_RUNTIME" = "podman" ]; then
    GPU_FLAGS+=(--device nvidia.com/gpu=all --security-opt=label=disable)
else
    GPU_FLAGS+=(--gpus all)
fi

echo "Starting ${CONTAINER_NAME}..."
echo "  Work dir:    ${WORK_DIR}"
echo "  JupyterLab:  http://${HOST_IP}:${PORT}/lab"

exec "$CONTAINER_RUNTIME" run -it --rm \
    --name "$CONTAINER_NAME" \
    "${GPU_FLAGS[@]}" \
    --add-host=host.containers.internal:host-gateway \
    "${PROXY_RUN_ENV[@]}" \
    -e "OLLAMA_HOST=${OLLAMA_HOST}" \
    "${USER_ENV[@]}" \
    -e "CONTAINER_USER=${CONTAINER_USER}" \
    -e "HOST_IP=${HOST_IP}" \
    -e "PORT=${PORT}" \
    -p "0.0.0.0:${PORT}:8888" \
    -v "${HOME_VOLUME}:/home/${CONTAINER_USER}" \
    "${HOME_MOUNT_ARG[@]}" \
    -v "${WORK_DIR}:/home/${CONTAINER_USER}/work" \
    "$IMAGE_NAME" \
    "$@"
