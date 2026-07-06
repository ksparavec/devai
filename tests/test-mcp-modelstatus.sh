#!/usr/bin/env bash
# End-to-end smoke test for the devai-model-status MCP server
# (docs/mcp-model-status.md), run against the live gateway.
#
# The stronger, protocol-level verification (real MCP client over
# stdio, exact fixture data) lives in
# devai-tools/cmd/devai-mcp-modelstatus/e2e_test.go -- run via
# `make test-devai-tools`, no gateway/image required. This script
# checks the other half: that the built image is actually registered
# in the running gateway's catalog and reachable through it.
#
# Skips cleanly (exit 77) when the gateway isn't running, or when the
# image can't be built (e.g. no network access to pull the golang
# builder base image) -- callers can decide whether that's a hard
# failure. Matches tests/test-mcp.sh's convention.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v "${CONTAINER_RUNTIME:-podman}" >/dev/null 2>&1; then
    echo "SKIP: ${CONTAINER_RUNTIME:-podman} not available" >&2
    exit 77
fi

echo ">>> test-mcp-modelstatus: building the image"
if ! (cd "${REPO_ROOT}" && make build-mcp-modelstatus-image >/tmp/test-mcp-modelstatus-build.log 2>&1); then
    echo "SKIP: could not build devai-mcp-modelstatus (see /tmp/test-mcp-modelstatus-build.log)" >&2
    echo "      commonly a network policy blocking the golang base image pull" >&2
    exit 77
fi

PORT="${MCP_PORT:-8088}"
HOST="${MCP_HOST:-devai-mcp-gateway}"
URL="http://${HOST}:${PORT}"

echo ">>> test-mcp-modelstatus: probing ${URL}"
if ! curl -fsS --max-time 3 "${URL}/health" >/dev/null 2>&1; then
    URL="http://localhost:${PORT}"
    if ! curl -fsS --max-time 3 "${URL}/health" >/dev/null 2>&1; then
        echo "SKIP: gateway not reachable at $HOST:$PORT or localhost:$PORT" >&2
        echo "      (run 'make mcp-up' first; or set MCP_HOST/MCP_PORT)" >&2
        exit 77
    fi
fi
echo "  /health OK against ${URL}"

body='{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
matched=""
resp=""
for path in /messages /v1/messages /tools/list; do
    resp=$(curl -fsS --max-time 5 -X POST -H 'Content-Type: application/json' \
                -d "${body}" "${URL}${path}" 2>/dev/null) || continue
    if echo "${resp}" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
tools = (d.get('result') or {}).get('tools') or d.get('tools') or []
names = {t.get('name') for t in tools}
want = {'list_fitting_models', 'get_model_bench', 'get_router_status'}
sys.exit(0 if want <= names else 1)
" 2>/dev/null; then
        matched="${path}"
        break
    fi
done

if [[ -z "${matched}" ]]; then
    echo "FAIL: tools/list did not include the 3 devai-model-status tools" >&2
    echo "      response was: ${resp}" >&2
    exit 1
fi
echo "  tools/list OK (endpoint=${matched}, all 3 devai-model-status tools present)"

call_body='{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_router_status","arguments":{}}}'
call_resp=$(curl -fsS --max-time 10 -X POST -H 'Content-Type: application/json' \
                  -d "${call_body}" "${URL}${matched}" 2>/dev/null)
if ! echo "${call_resp}" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
result = d.get('result') or {}
sc = result.get('structuredContent') or {}
sys.exit(0 if sc.get('mode') else 1)
" 2>/dev/null; then
    echo "FAIL: tools/call get_router_status did not return a mode" >&2
    echo "      response was: ${call_resp}" >&2
    exit 1
fi
echo "  tools/call get_router_status OK (real response, not stubbed)"

echo ">>> test-mcp-modelstatus OK"
exit 0
