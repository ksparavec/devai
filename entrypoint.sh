#!/bin/bash
set -e

USERNAME=${CONTAINER_USER:-devai}
HOME_DIR="/home/$USERNAME"
WORK_DIR="$HOME_DIR/work"

echo "Initializing container (UID: $(id -u))..."

# Create directories if missing (works for both root and non-root)
mkdir -p "$HOME_DIR" "$WORK_DIR" \
         "$HOME_DIR/.local/share/jupyter/runtime" \
         "$HOME_DIR/.jupyter"

export HOME="$HOME_DIR"

# Navigate to work dir or home dir
if [ -d "$WORK_DIR" ]; then
    cd "$WORK_DIR"
else
    cd "$HOME_DIR"
fi

# Prepare the command
CMD=("$@")

# Inject custom display URL if HOST_IP is set and we are running jupyter
if [ -n "$HOST_IP" ] && [ "${CMD[0]}" = "jupyter" ]; then
    TARGET_PORT=${PORT:-8888}
    CMD+=("--ServerApp.custom_display_url=http://$HOST_IP:$TARGET_PORT")
fi

echo "Exec command: ${CMD[*]}"
exec "${CMD[@]}"
