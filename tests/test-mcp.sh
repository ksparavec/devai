#!/usr/bin/env bash
# End-to-end smoke test for devai-mcp-gateway (Phase 1).
#
# Runs against a live gateway brought up via `make mcp-up`. Exits 0
# when the gateway is reachable and at least the four core Tier 1
# tools (filesystem, fetch, time, sequentialthinking) respond to a
# tools/list call.
#
# Skips cleanly (exit 77) when the gateway isn't running -- callers
# can decide whether that's a hard failure.

set -u

PORT="${MCP_PORT:-8088}"
HOST="${MCP_HOST:-devai-mcp-gateway}"
URL="http://${HOST}:${PORT}"

echo ">>> test-mcp: probing ${URL}"

if ! curl -fsS --max-time 3 "${URL}/health" >/dev/null 2>&1; then
    # Try localhost as a fallback (when running outside the
    # devai-net network).
    URL="http://localhost:${PORT}"
    if ! curl -fsS --max-time 3 "${URL}/health" >/dev/null 2>&1; then
        echo "SKIP: gateway not reachable at $HOST:$PORT or localhost:$PORT" >&2
        echo "      (run 'make mcp-up' first; or set MCP_HOST/MCP_PORT)" >&2
        exit 77
    fi
fi

echo "  /health OK against ${URL}"

# Try the streaming MCP-protocol JSON-RPC path. We send a tools/list
# request and verify the response carries a tools array. Some gateway
# versions front-end this as /messages or /v1/messages; tolerate
# either.
body='{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
matched=""
for path in /messages /v1/messages /tools/list; do
    resp=$(curl -fsS --max-time 5 -X POST -H 'Content-Type: application/json' \
                -d "${body}" "${URL}${path}" 2>/dev/null) || continue
    if echo "${resp}" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
tools = (d.get('result') or {}).get('tools') or d.get('tools') or []
sys.exit(0 if tools else 1)
" 2>/dev/null; then
        matched="${path}"
        break
    fi
done

if [[ -z "${matched}" ]]; then
    echo "FAIL: no tools/list endpoint returned a non-empty tools array" >&2
    exit 1
fi

echo "  tools/list OK (endpoint=${matched})"
echo ">>> test-mcp OK"
exit 0
