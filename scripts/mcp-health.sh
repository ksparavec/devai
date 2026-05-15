#!/usr/bin/env bash
# Smoke test for devai-mcp-gateway.
#
# Hits /health on the gateway's host port (default 8088), then enumerates
# the loaded server catalog. Used by `make mcp-test` and as a manual
# sanity check after `make cache-up --profile mcp`.
#
# Exit code: 0 on success, 1 on any HTTP failure or unexpected response.

set -euo pipefail

PORT="${MCP_PORT:-8088}"
HOST="${MCP_HOST:-localhost}"
URL="http://${HOST}:${PORT}"

echo ">>> mcp-health: probing ${URL}"

# /health: returns "OK" or a JSON status block depending on gateway version.
http_code=$(curl -fsS -o /tmp/mcp-health.body -w '%{http_code}' "${URL}/health" || true)
if [[ "${http_code}" != "200" ]]; then
    echo "FAIL: /health returned HTTP ${http_code}" >&2
    cat /tmp/mcp-health.body >&2 || true
    exit 1
fi
echo "  /health: OK"

# /servers: enumeration endpoint. Some gateway builds expose this as
# /api/v1/servers; tolerate either.
for path in /servers /api/v1/servers; do
    http_code=$(curl -fsS -o /tmp/mcp-servers.body -w '%{http_code}' "${URL}${path}" || true)
    if [[ "${http_code}" == "200" ]]; then
        n=$(python3 -c "import json,sys; d=json.load(open('/tmp/mcp-servers.body')); print(len(d) if isinstance(d, list) else len(d.get('servers') or []))" 2>/dev/null || echo "?")
        echo "  ${path}: ${n} server(s) loaded"
        break
    fi
done

echo ">>> mcp-health OK"
