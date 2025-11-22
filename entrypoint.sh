#!/bin/bash
set -e

# Read configuration from env or defaults
UID_VAL=${USER_ID:-1000}
GID_VAL=${GROUP_ID:-1000}
USERNAME=${CONTAINER_USER:-devai}

echo "Initializing container for user '$USERNAME' (UID: $UID_VAL, GID: $GID_VAL)..."

# Determine if we are running as root
AM_I_ROOT=$(id -u)

# Set home and work directories
HOME_DIR="/home/$USERNAME"
WORK_DIR="$HOME_DIR/work"

if [ "$AM_I_ROOT" -eq 0 ]; then
    # Running as root
    
    # Create group if it doesn't exist
    if ! getent group "$GID_VAL" > /dev/null 2>&1; then
        groupadd -g "$GID_VAL" "$USERNAME"
    else
        echo "Group ID $GID_VAL already exists."
    fi

    # Create user if it doesn't exist
    if ! id -u "$UID_VAL" > /dev/null 2>&1; then
        echo "Creating user $USERNAME..."
        useradd -u "$UID_VAL" -g "$GID_VAL" -M -s /bin/bash "$USERNAME"
    else
        echo "User ID $UID_VAL already exists."
    fi
    
    # If the home directory doesn't exist, create it
    if [ ! -d "$HOME_DIR" ]; then
        mkdir -p "$HOME_DIR"
    fi
    
    # Create the work directory if it doesn't exist
    if [ ! -d "$WORK_DIR" ]; then
        mkdir -p "$WORK_DIR"
    fi
    
    # Ensure ownership
    chown "$UID_VAL:$GID_VAL" "$HOME_DIR" "$WORK_DIR"
fi

# Export HOME explicitely just in case
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

# Switch to the user and execute the command
echo "Exec command: ${CMD[*]}"

if [ "$AM_I_ROOT" -eq 0 ]; then
    exec gosu "$UID_VAL:$GID_VAL" "${CMD[@]}"
else
    exec "${CMD[@]}"
fi