#!/bin/bash
# Integration tests for gpu-arbiter router (stable, Ollama-only)
# Prerequisites: make cache-up (infrastructure running)
#
# Usage: ./tests/test-router.sh
#
# For vLLM/GPU tests (flaky due to cold start timing), see: test-router-vllm.sh

RUNTIME="${CONTAINER_RUNTIME:-podman}"
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
info "=== Test 1: Router health check ==="
# ============================================================================

health=$(router_curl /health)
if echo "$health" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok'" 2>/dev/null; then
    pass "health endpoint returns ok"
else
    fail "health endpoint" "$health"
fi

# ============================================================================
info "=== Test 2: Model listing (/v1/models) ==="
# ============================================================================

models=$(router_curl /v1/models)
if echo "$models" | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d['data'])>0" 2>/dev/null; then
    count=$(echo "$models" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data']))")
    pass "/v1/models returns $count models"
else
    fail "/v1/models" "$models"
fi

# ============================================================================
info "=== Test 3: Model listing (/api/tags includes vLLM) ==="
# ============================================================================

tags=$(router_curl /api/tags)
ollama_count=$(echo "$tags" | python3 -c "import sys,json; d=json.load(sys.stdin); print(sum(1 for m in d['models'] if 'NVFP4' not in m['name']))" 2>/dev/null)
vllm_count=$(echo "$tags" | python3 -c "import sys,json; d=json.load(sys.stdin); print(sum(1 for m in d['models'] if 'NVFP4' in m['name']))" 2>/dev/null)
if [ "$vllm_count" -gt 0 ] 2>/dev/null; then
    pass "/api/tags includes vLLM models ($ollama_count ollama + $vllm_count vllm)"
else
    fail "/api/tags vLLM models" "vllm_count=$vllm_count"
fi

# ============================================================================
info "=== Test 4: GGUF model via Ollama (OpenAI API) ==="
# ============================================================================

resp=$(router_curl_long /v1/chat/completions 60 \
    '{"model":"qwen3.5:9b","messages":[{"role":"user","content":"Say hi"}],"max_tokens":5}')
if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['model']=='qwen3.5:9b'" 2>/dev/null; then
    pass "GGUF model via /v1/chat/completions"
else
    fail "GGUF /v1/ routing" "$resp"
fi

health=$(router_curl /health)
backend=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin)['active_backend'])")
if [ "$backend" = "ollama" ]; then
    pass "active backend is ollama"
else
    fail "active backend" "expected ollama, got $backend"
fi

# ============================================================================
info "=== Test 5: Streaming (SSE via OpenAI API) ==="
# ============================================================================

resp=$($RUNTIME exec devai-open-webui curl -s --max-time 30 \
    -H "Content-Type: application/json" \
    "http://router:11434/v1/chat/completions" \
    -d '{"model":"qwen3.5:9b","messages":[{"role":"user","content":"Count to 3"}],"max_tokens":20,"stream":true}' 2>&1)
if echo "$resp" | grep -q "data:"; then
    pass "OpenAI SSE streaming works"
else
    fail "SSE streaming" "no SSE data received"
fi

# ============================================================================
info "=== Test 6: Non-POST requests pass through to Ollama ==="
# ============================================================================

resp=$(router_curl /api/tags)
if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'models' in d" 2>/dev/null; then
    pass "GET /api/tags passes through to Ollama"
else
    fail "passthrough" "$resp"
fi

# ============================================================================
info "=== Test 7: Empty model name routes to Ollama ==="
# ============================================================================

resp=$(router_curl_long /v1/chat/completions 30 \
    '{"model":"","messages":[{"role":"user","content":"Say hi"}],"max_tokens":5}')
if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'error' in d or 'choices' in d or 'model' in d" 2>/dev/null; then
    pass "empty model name handled without crash"
else
    fail "empty model" "$resp"
fi

# ============================================================================
info "=== Test 8: Missing model field routes to Ollama ==="
# ============================================================================

resp=$(router_curl_long /v1/chat/completions 30 \
    '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":5}')
if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'error' in d or 'choices' in d or 'model' in d" 2>/dev/null; then
    pass "missing model field handled without crash"
else
    fail "missing model" "$resp"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================"
echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}"
echo "========================================"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
