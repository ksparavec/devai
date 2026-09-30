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
# Hard per-service size cap. When a log exceeds this between reconnects, the
# head is dropped and the most-recent half kept -- bounds a crash-loop
# (which reconnects constantly) from filling the volume. Override via env.
LOG_MAX_BYTES="${LOG_MAX_BYTES:-524288000}" # 500 MiB
mkdir -p "$LOG_DIR"
PODMAN="podman --remote --url unix:///run/podman/podman.sock"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# rotate_if_big drops the head of an oversized log, keeping the most-recent
# half. Only called between reconnects (no follower FD open on the file), so
# an in-place tail+replace is safe.
rotate_if_big() {
  f=$1
  [ -f "$f" ] || return 0
  sz=$(wc -c < "$f" 2>/dev/null || echo 0)
  if [ "$sz" -gt "$LOG_MAX_BYTES" ]; then
    tmp="$f.tmp.$$"
    if tail -c "$((LOG_MAX_BYTES / 2))" "$f" > "$tmp" 2>/dev/null; then
      mv "$tmp" "$f"
      echo "[$(stamp)] [logger] truncated $f (was $sz bytes; kept last $((LOG_MAX_BYTES / 2)))" >> "$f"
    else
      rm -f "$tmp" 2>/dev/null || true
    fi
  fi
}

# last_podman_stamp prints the newest `podman logs --timestamps` stamp in a
# persisted log (RFC3339, with or without fractional seconds), or nothing
# when the file is absent or holds none. The logger's own "[...] [logger]"
# lines are bracketed and never match. Read from the END (tac), so the cost
# does not grow with the file.
last_podman_stamp() {
  [ -f "$1" ] || return 0
  tac "$1" 2>/dev/null \
    | grep -Eo -m1 '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})' \
    || true
}

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
  # Includes the two HF backends: they start as `sleep infinity`
  # placeholders and are RECREATED by the router on first request, so
  # a logger that started before either was ever used would otherwise
  # never follow the engine whose crash logs matter most.
  LOG_TARGETS="devai-ollama devai-router devai-vllm devai-sglang devai-laya-trainer devai-open-webui devai-webui-proxy"
fi

echo "[$(stamp)] [logger] following: $LOG_TARGETS"

follow() {
  name=$1
  out="$LOG_DIR/$name.log"
  echo "[$(stamp)] [logger] starting follower for $name -> $out" >> "$out"
  # On the first connect, resume from the newest stamp already in the file:
  # `--follow` without --since streams the container's WHOLE history, so every
  # logger restart (each `make cache-up` recreates it) appended that history
  # again -- the persisted router log held 3,245 launch lines for 742
  # distinct launches (2026-09-27 re-analysis). --since is inclusive at the
  # stamp's precision (whole seconds here), so up to one second of lines can
  # repeat at the seam; none are lost. An empty file (or none) still gets the
  # full history. After a disconnect we set since to the disconnect time, so
  # a reconnect streams ONLY new lines rather than re-reading the backlog on
  # every transient stream drop or container recreate.
  since="$(last_podman_stamp "$out")"
  while true; do
    rotate_if_big "$out"
    if [ -n "$since" ]; then set -- --since "$since"; else set --; fi
    # `--follow` blocks until the container ends or the stream errors.
    # `--timestamps` prepends RFC3339Nano per line so we can correlate
    # across log files.
    #
    # devai-pipelock emits a benign net.ErrClosed line ("use of closed network
    # connection") on every proxied request teardown -- pipelock has no config
    # knob to suppress it (see docs/pipelock.md). Drop it from the persisted log
    # for that one service; all other streams pass through unfiltered.
    if [ "$name" = "devai-pipelock" ]; then
      $PODMAN logs --follow --timestamps "$@" "$name" 2>&1 \
        | grep -v --line-buffered 'use of closed network connection' >> "$out" || true
    else
      $PODMAN logs --follow --timestamps "$@" "$name" >> "$out" 2>&1 || true
    fi
    since="$(stamp)"
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
