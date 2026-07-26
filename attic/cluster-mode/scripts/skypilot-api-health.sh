#!/usr/bin/env bash
# Smoke test for devai-skypilot-api-server.
#
# Pokes /api/v1/version and (when available) `sky check` from inside
# the container. Used by `make skypilot-check` and as a manual sanity
# check after `make skypilot-up`.

set -euo pipefail

PORT="${SKYPILOT_API_PORT:-46580}"
HOST="${SKYPILOT_API_HOST:-localhost}"
URL="http://${HOST}:${PORT}"

echo ">>> skypilot-api-health: probing ${URL}"

# /api/v1/version: cheap liveness probe.
http_code=$(curl -fsS -o /tmp/sky-version.body -w '%{http_code}' "${URL}/api/v1/version" || true)
if [[ "${http_code}" != "200" ]]; then
    echo "FAIL: /api/v1/version returned HTTP ${http_code}" >&2
    cat /tmp/sky-version.body >&2 || true
    exit 1
fi
echo "  /api/v1/version: $(cat /tmp/sky-version.body)"

# Optional: enabled-cloud check via the container's own sky CLI.
# Bails cleanly when no creds are mounted (still a successful health
# probe -- the SERVER is up, just hasn't been pointed at any cloud).
if command -v podman >/dev/null 2>&1 && podman ps --format '{{.Names}}' | grep -q '^devai-skypilot-api-server$'; then
    echo "  sky check (inside container):"
    podman exec devai-skypilot-api-server sky check 2>&1 | head -20 || true
fi

echo ">>> skypilot-api-health OK"
