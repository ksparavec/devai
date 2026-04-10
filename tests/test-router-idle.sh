#!/bin/bash
# Integration test for gpu-arbiter idle timeout
# Restarts the router with a short IDLE_TIMEOUT, runs the test,
# then restores the original timeout.
#
# Prerequisites: make cache-up (infrastructure running), GPU available
# Usage: ./tests/test-router-idle.sh

RUNTIME="${CONTAINER_RUNTIME:-podman}"
COMPOSE="$RUNTIME compose -f deploy/docker-compose.yaml"
PASS=0
FAIL=0

GREEN='\033[32m'
RED='\033[31m'
YELLOW='\033[33m'
NC='\033[0m'

pass() { ((PASS++)); echo -e "  ${GREEN}PASS${NC} $1"; }
fail() { ((FAIL++)); echo -e "  ${RED}FAIL${NC} $1: $2"; }
info() { echo -e "${YELLOW}$1${NC}"; }

router_curl() {
    $RUNTIME exec devai-open-webui curl -sf --max-time "${2:-10}" \
        -H "Content-Type: application/json" \
        "http://router:11434$1" ${3:+-d "$3"} 2>&1
}

router_curl_long() {
    $RUNTIME exec devai-open-webui curl -s --max-time "${2:-200}" \
        -H "Content-Type: application/json" \
        "http://router:11434$1" -d "$3" 2>&1
}

# ============================================================================
info "=== Idle timeout test ==="
# ============================================================================

SHORT_TIMEOUT=15

info "  Restarting router with IDLE_TIMEOUT=${SHORT_TIMEOUT}s..."
VLLM_IDLE_TIMEOUT=$SHORT_TIMEOUT $COMPOSE up -d router >/dev/null 2>&1
sleep 2

# Verify router is healthy
health=$(router_curl /health)
if ! echo "$health" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok'" 2>/dev/null; then
    fail "router health after restart" "$health"
    echo ""
    echo "========================================"
    echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}"
    echo "========================================"
    exit 1
fi

info "  Starting vLLM via NVFP4 request (cold start, up to 5min)..."
resp=$(router_curl_long /v1/chat/completions 300 \
    '{"model":"NVIDIA-Nemotron-Nano-9B-v2-NVFP4","messages":[{"role":"user","content":"hi"}],"max_tokens":5}')
if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'NVFP4' in d['model']" 2>/dev/null; then
    pass "vLLM started for idle timeout test"
else
    fail "vLLM start for idle test" "$resp"
fi

WAIT_TIME=$((SHORT_TIMEOUT + 35))
info "  Waiting ${WAIT_TIME}s for idle timeout (${SHORT_TIMEOUT}s timeout + 30s poll interval + margin)..."
sleep $WAIT_TIME

vllm_status=$($RUNTIME inspect -f '{{.State.Status}}' devai-vllm 2>/dev/null)
health=$(router_curl /health)
backend=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin)['active_backend'])" 2>/dev/null)
vllm_flag=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin)['vllm_running'])" 2>/dev/null)

if [ "$vllm_status" != "running" ] && [ "$backend" = "ollama" ] && [ "$vllm_flag" = "False" ]; then
    pass "vLLM auto-stopped after idle timeout"
else
    fail "idle timeout" "container=$vllm_status backend=$backend vllm_running=$vllm_flag"
fi

# ============================================================================
info "  Restoring router with default IDLE_TIMEOUT..."
# ============================================================================
$COMPOSE up -d router >/dev/null 2>&1

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================"
echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}"
echo "========================================"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
