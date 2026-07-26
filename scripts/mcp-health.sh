#!/usr/bin/env bash
# Liveness probe for devai-mcp-gateway.
#
# This is deliberately a LIVENESS check, not a functional one. It answers
# "is the process up and listening", nothing more.
#
# Read this before trusting it: on 2026-07-25 the gateway returned
# HTTP 200 from /health while serving ZERO tools -- its catalog had
# parsed to an empty registry and no server was enabled. A green
# /health told the operator nothing at all. The functional check is
# `make mcp-test` (tests/test-mcp.sh), which performs a real MCP
# handshake, asserts a tool-count floor and makes a real tools/call.
# Use that to answer "does it work".
#
# Exit code: 0 when the gateway is listening, 1 otherwise.

set -euo pipefail

PORT="${MCP_PORT:-8088}"
HOST="${MCP_HOST:-127.0.0.1}"
URL="http://${HOST}:${PORT}"

echo ">>> mcp-health: probing ${URL}"

http_code=$(curl -fsS -o /tmp/mcp-health.body -w '%{http_code}' "${URL}/health" || true)
if [[ "${http_code}" != "200" ]]; then
    echo "FAIL: /health returned HTTP ${http_code}" >&2
    cat /tmp/mcp-health.body >&2 || true
    echo "      Is the gateway up? 'make mcp-up'." >&2
    exit 1
fi
echo "  /health: listening"

# There is no catalog-enumeration endpoint to probe here. This script
# used to try /servers and /api/v1/servers; on the pinned gateway both
# return 307 (they redirect to /mcp -- they do not exist), so that check
# could never have succeeded and quietly printed nothing when it failed.
# Enumerating the catalog requires a full MCP handshake against /mcp with
# the bearer token, which is what tests/test-mcp.sh does.

echo ">>> mcp-health: gateway is listening."
echo "    This does NOT mean it serves any tools -- run 'make mcp-test' for that."
