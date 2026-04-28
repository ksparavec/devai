#!/bin/sh
# Container-stdout collector for devai-* services.
#
# Spawns one `podman logs --follow` per target service against the host's
# podman socket and appends the stream (with timestamps) to per-service
# files under /logs. When a target container is removed and recreated
# (e.g. `make probe` cycling devai-ollama between VRAM bands), the
# follower exits, the loop sleeps 5s, and reconnects to the new container
# under the same name. Logs persist across container recreations because
# the output volume is host-backed.
#
# Inputs (env):
#   LOG_TARGETS  — space-separated list of container names to follow.
#                  Default: every devai-* service that exists at startup.
#   LOG_DIR      — output directory inside the container (default /logs).
#
# This deliberately uses no third-party log daemon: just podman + sh.
# Errors don't stop the script — followers keep retrying forever so the
# logger survives compose-up of new services.

set -u
LOG_DIR="${LOG_DIR:-/logs}"
mkdir -p "$LOG_DIR"
PODMAN="podman --remote --url unix:///run/podman/podman.sock"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

discover_targets() {
  # Names of every running devai-* container at this moment.
  $PODMAN ps --all --format '{{.Names}}' 2>/dev/null \
    | grep '^devai-' | grep -v '^devai-logger$' || true
}

if [ -z "${LOG_TARGETS:-}" ]; then
  LOG_TARGETS="$(discover_targets | tr '\n' ' ')"
fi
if [ -z "$LOG_TARGETS" ]; then
  echo "[$(stamp)] [logger] no devai-* containers found yet; will start"
  echo "[$(stamp)] [logger] following the well-known set instead"
  LOG_TARGETS="devai-ollama devai-router devai-open-webui devai-webui-proxy"
fi

echo "[$(stamp)] [logger] following: $LOG_TARGETS"

follow() {
  name=$1
  out="$LOG_DIR/$name.log"
  echo "[$(stamp)] [logger] starting follower for $name -> $out" >> "$out"
  while true; do
    # `--follow` blocks until the container ends or the stream errors.
    # `--timestamps` prepends RFC3339Nano per line so we can correlate
    # across log files.
    $PODMAN logs --follow --timestamps "$name" >> "$out" 2>&1 || true
    echo "[$(stamp)] [logger] follower for $name exited; retry in 5s" >> "$out"
    sleep 5
  done
}

for name in $LOG_TARGETS; do
  follow "$name" &
done

# Stay foreground so the container doesn't exit. `wait` blocks on all
# background followers; if any exits the loop catches it and respawns.
wait
