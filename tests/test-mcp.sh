#!/usr/bin/env bash
# End-to-end MCP gateway test: real handshake, real tools/list, real
# tools/call.
#
# This replaces a version that probed a /health endpoint the gateway does
# not serve and skipped (exit 77) whenever that probe failed -- which was
# unconditionally, because the service had never successfully started.
# That test reported green for months against a gateway that exposed ZERO
# tools, because everything else it checked was the shape of YAML files.
#
# The rules here, learned from that failure:
#   1. Skip ONLY when the gateway is not running. A reachable gateway
#      that exposes no tools must FAIL, loudly.
#   2. Assert a non-zero tool floor, not just "some JSON came back".
#   3. Actually invoke a tool. Enumeration passing while invocation is
#      broken is a real and observed state.
#
# Usage: bash tests/test-mcp.sh    (or `make mcp-test`)
set -uo pipefail

GATEWAY_HOST="${MCP_GATEWAY_HOST:-127.0.0.1}"
GATEWAY_PORT="${MCP_PORT:-8088}"
URL="http://${GATEWAY_HOST}:${GATEWAY_PORT}/mcp"
CONTAINER="${MCP_CONTAINER:-devai-mcp-gateway}"
RUNTIME="${CONTAINER_RUNTIME:-podman}"

# Minimum tools expected from the enabled server set. Deliberately well
# under the 134 observed on 2026-07-25 so ordinary upstream churn does not
# fail the build, but far enough above zero to catch the "catalog parsed
# empty" and "no --servers flag" failures that motivated this rewrite.
MIN_TOOLS="${MCP_MIN_TOOLS:-40}"

pass() { echo "  PASS  $*"; }
fail() { echo "  FAIL  $*" >&2; FAILURES=$((FAILURES + 1)); }
FAILURES=0

echo ">>> test-mcp: ${URL}"

# --- Precondition: is the gateway running at all? ---------------------
if ! "$RUNTIME" inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "SKIP: MCP gateway not running (no ${CONTAINER} container)." >&2
  echo "      Start it with 'make mcp-up', then re-run." >&2
  exit 77
fi

# --- Bearer token -----------------------------------------------------
# The gateway mints a token at startup and prints it once. A containerised
# run has no other way to obtain it, so read it back from the log.
TOKEN="${MCP_GATEWAY_TOKEN:-}"
if [ -z "$TOKEN" ]; then
  TOKEN="$("$RUNTIME" logs "$CONTAINER" 2>&1 | grep -oE 'Bearer [A-Za-z0-9]+' | tail -1 | awk '{print $2}')"
fi
if [ -z "$TOKEN" ]; then
  echo "FAIL: could not obtain a bearer token from ${CONTAINER} logs." >&2
  echo "      Set MCP_GATEWAY_TOKEN=... to supply one explicitly." >&2
  exit 1
fi

HDRS=(-H "Authorization: Bearer ${TOKEN}"
      -H 'Content-Type: application/json'
      -H 'Accept: application/json, text/event-stream')

# --- initialize -------------------------------------------------------
HDR_FILE="$(mktemp)"; trap 'rm -f "$HDR_FILE"' EXIT
INIT_BODY='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"devai-test-mcp","version":"1"}}}'
INIT_OUT="$(curl -sS -D "$HDR_FILE" --max-time 30 -X POST "$URL" "${HDRS[@]}" -d "$INIT_BODY")"

if echo "$INIT_OUT" | grep -q '"serverInfo"'; then
  pass "initialize handshake"
else
  fail "initialize handshake did not return serverInfo: ${INIT_OUT:0:200}"
  echo; echo "${FAILURES} check(s) failed."; exit 1
fi

SESSION="$(grep -i '^mcp-session-id:' "$HDR_FILE" | tr -d '\r' | awk '{print $2}')"
if [ -n "$SESSION" ]; then
  pass "session id issued"
  HDRS+=(-H "Mcp-Session-Id: ${SESSION}")
else
  fail "no Mcp-Session-Id header on initialize"
fi

curl -sS --max-time 15 -X POST "$URL" "${HDRS[@]}" \
     -d '{"jsonrpc":"2.0","method":"notifications/initialized"}' >/dev/null

# --- tools/list -------------------------------------------------------
# Responses are SSE-framed ("event: message\ndata: {...}"), so strip the
# data: prefix and take the first JSON object.
TOOLS_JSON="$(curl -sS --max-time 60 -X POST "$URL" "${HDRS[@]}" \
              -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
              | sed 's/^data: //' | grep -m1 '^{')"

TOOL_COUNT="$(printf '%s' "$TOOLS_JSON" | python3 -c \
  'import sys,json; print(len(json.load(sys.stdin).get("result",{}).get("tools",[])))' 2>/dev/null || echo 0)"

if [ "${TOOL_COUNT:-0}" -ge "$MIN_TOOLS" ]; then
  pass "tools/list returned ${TOOL_COUNT} tools (floor ${MIN_TOOLS})"
else
  fail "tools/list returned ${TOOL_COUNT} tools, expected at least ${MIN_TOOLS}."
  echo "        A count of 0 means the catalog parsed empty or --servers is missing." >&2
fi

# --- first-party server present --------------------------------------
# devai-model-status is the only server this repo builds. If the
# --additional-catalog merge regresses, its tools vanish while every
# upstream one keeps working -- so assert it by name.
for tool in list_fitting_models get_model_bench get_router_status; do
  if printf '%s' "$TOOLS_JSON" | grep -q "\"${tool}\""; then
    pass "first-party tool present: ${tool}"
  else
    fail "first-party tool missing: ${tool} (is localhost/devai-mcp-modelstatus:latest built?)"
  fi
done

# --- tools/call -------------------------------------------------------
# Enumeration can succeed while invocation fails (container spawn, pull
# policy, stdio framing), so exercise one real call. list_fitting_models
# reads caches baked into the image and needs no network, which makes it
# the right choice for a hermetic test.
CALL_JSON="$(curl -sS --max-time 60 -X POST "$URL" "${HDRS[@]}" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"list_fitting_models","arguments":{"vram_gb":24,"context":32768}}}' \
  | sed 's/^data: //' | grep -m1 '^{')"

if printf '%s' "$CALL_JSON" | python3 -c \
   'import sys,json; d=json.load(sys.stdin); c=d.get("result",{}).get("content",[]); sys.exit(0 if c and c[0].get("text") else 1)' 2>/dev/null; then
  pass "tools/call list_fitting_models returned content"
else
  fail "tools/call list_fitting_models returned no content: ${CALL_JSON:0:200}"
fi

echo
if [ "$FAILURES" -eq 0 ]; then
  echo "All MCP gateway checks passed (${TOOL_COUNT} tools)."
  exit 0
fi
echo "${FAILURES} check(s) failed."
exit 1
