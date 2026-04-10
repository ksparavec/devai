#!/bin/bash
set -e

USERNAME=${CONTAINER_USER:-devai}
HOME_DIR="/home/$USERNAME"
WORK_DIR="$HOME_DIR/work"

# Export lowercase proxy aliases for tools that require them
if [ -n "${HTTP_PROXY:-}" ]; then
    export http_proxy="${HTTP_PROXY}"
    export https_proxy="${HTTPS_PROXY:-}"
    export no_proxy="${NO_PROXY:-}"
    echo "Proxy configured: ${HTTP_PROXY}"
fi

# Remap devai user/group to match host UID/GID (only when running as root)
if [ "$(id -u)" = "0" ] && [ -n "${USER_ID:-}" ]; then
    TARGET_UID=${USER_ID}
    TARGET_GID=${GROUP_ID:-$USER_ID}

    # Ensure group exists with correct GID
    CURRENT_GID=$(getent group "$USERNAME" 2>/dev/null | cut -d: -f3 || true)
    if [ -n "$CURRENT_GID" ] && [ "$CURRENT_GID" != "$TARGET_GID" ]; then
        # Group exists but wrong GID — free target GID if taken, then reassign
        BLOCKING_GROUP=$(getent group "$TARGET_GID" 2>/dev/null | cut -d: -f1 || true)
        if [ -n "$BLOCKING_GROUP" ] && [ "$BLOCKING_GROUP" != "$USERNAME" ]; then
            groupmod -g 99999 "$BLOCKING_GROUP"
        fi
        groupmod -g "$TARGET_GID" "$USERNAME"
    elif [ -z "$CURRENT_GID" ]; then
        # Group doesn't exist — free target GID if taken, then create
        BLOCKING_GROUP=$(getent group "$TARGET_GID" 2>/dev/null | cut -d: -f1 || true)
        if [ -n "$BLOCKING_GROUP" ]; then
            groupmod -n "$USERNAME" "$BLOCKING_GROUP"
        else
            groupadd -g "$TARGET_GID" "$USERNAME"
        fi
    fi

    # Ensure user exists with correct UID
    CURRENT_UID=$(id -u "$USERNAME" 2>/dev/null || true)
    if [ -n "$CURRENT_UID" ] && [ "$CURRENT_UID" != "$TARGET_UID" ]; then
        # User exists but wrong UID — free target UID if taken, then reassign
        BLOCKING_USER=$(getent passwd "$TARGET_UID" 2>/dev/null | cut -d: -f1 || true)
        if [ -n "$BLOCKING_USER" ] && [ "$BLOCKING_USER" != "$USERNAME" ]; then
            usermod -u 99999 "$BLOCKING_USER"
        fi
        usermod -u "$TARGET_UID" -g "$TARGET_GID" "$USERNAME"
    elif [ -z "$CURRENT_UID" ]; then
        # User doesn't exist — free target UID if taken, then create
        BLOCKING_USER=$(getent passwd "$TARGET_UID" 2>/dev/null | cut -d: -f1 || true)
        if [ -n "$BLOCKING_USER" ]; then
            usermod -l "$USERNAME" -d "$HOME_DIR" "$BLOCKING_USER"
            usermod -g "$TARGET_GID" "$USERNAME"
        else
            useradd -u "$TARGET_UID" -g "$TARGET_GID" -m -s /bin/bash "$USERNAME"
        fi
    fi

    # Create directories and fix ownership
    mkdir -p "$HOME_DIR" "$WORK_DIR" \
             "$HOME_DIR/.local/share/jupyter/runtime" \
             "$HOME_DIR/.jupyter"
    chown -R "$TARGET_UID:$TARGET_GID" "$HOME_DIR/.local" "$HOME_DIR/.jupyter"
    chown "$TARGET_UID:$TARGET_GID" "$HOME_DIR"

    # Copy host config files from staging mount and fix ownership
    if [ -d /tmp/host-config ]; then
        for item in /tmp/host-config/.*; do
            [ "$(basename "$item")" = "." ] || [ "$(basename "$item")" = ".." ] && continue
            cp -a "$item" "$HOME_DIR/"
        done
        chown -R "$TARGET_UID:$TARGET_GID" "$HOME_DIR/.gitconfig" "$HOME_DIR/.ssh" 2>/dev/null || true
    fi

    echo "Running as $USERNAME (UID:$TARGET_UID GID:$TARGET_GID)"
else
    echo "Running as $(id -un) (UID:$(id -u))"

    # Create directories if missing
    mkdir -p "$HOME_DIR" "$WORK_DIR" \
             "$HOME_DIR/.local/share/jupyter/runtime" \
             "$HOME_DIR/.jupyter"

    # Copy host config files from staging mount (rootless podman: root = host user)
    if [ -d /tmp/host-config ]; then
        for item in /tmp/host-config/.*; do
            [ "$(basename "$item")" = "." ] || [ "$(basename "$item")" = ".." ] && continue
            cp -a "$item" "$HOME_DIR/"
        done
    fi
fi


# Seed agent config files if not already present in persistent home
if [ ! -f "$HOME_DIR/.codex/config.toml" ] && [ -f /etc/devai/codex-config.toml ]; then
    mkdir -p "$HOME_DIR/.codex"
    cp /etc/devai/codex-config.toml "$HOME_DIR/.codex/config.toml"
fi

# Prepare the command
CMD=("$@")

# Inject Jupyter settings if running jupyter
if [ "${CMD[0]}" = "jupyter" ]; then
    TARGET_PORT=${PORT:-8888}
    CERT_DIR="$HOME_DIR/.jupyter/ssl"
    if [ -f "$CERT_DIR/cert.pem" ] || [ -f "$CERT_DIR/${HOST_IP}.pem" ]; then
        # Use mkcert certs if available (named by IP or as cert.pem)
        CERTFILE="${CERT_DIR}/${HOST_IP}.pem"
        KEYFILE="${CERT_DIR}/${HOST_IP}-key.pem"
        [ -f "$CERTFILE" ] || CERTFILE="$CERT_DIR/cert.pem"
        [ -f "$KEYFILE" ] || KEYFILE="$CERT_DIR/key.pem"
        CMD+=("--ServerApp.certfile=$CERTFILE" "--ServerApp.keyfile=$KEYFILE")
        PROTO=https
    else
        PROTO=http
    fi
    if [ -n "$HOST_IP" ]; then
        CMD+=("--ServerApp.custom_display_url=$PROTO://$HOST_IP:$TARGET_PORT")
    fi
    if [ -n "${JUPYTER_TOKEN:-}" ]; then
        CMD+=("--IdentityProvider.token=$JUPYTER_TOKEN")
    fi
fi

# Switch to non-root user via gosu if running as root
if [ "$(id -u)" = "0" ] && [ -n "${USER_ID:-}" ]; then
    exec gosu "$USERNAME" bash -c 'export HOME="'"$HOME_DIR"'" && cd "'"$WORK_DIR"'" 2>/dev/null || cd "'"$HOME_DIR"'" && exec "$@"' -- "${CMD[@]}"
fi

# Fallback: running as non-root already
export HOME="$HOME_DIR"
cd "$WORK_DIR" 2>/dev/null || cd "$HOME_DIR"
echo "Exec command: ${CMD[*]}"
exec "${CMD[@]}"
