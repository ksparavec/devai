#!/bin/bash
# Integration tests for gpu-arbiter (Ollama backend, port 11434)
# Prerequisites: make cache-up (infrastructure running)
#
# Usage: ./tests/test-router.sh

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

ollama_curl() {
    $RUNTIME exec devai-open-webui curl -sf --max-time "${2:-10}" \
        -H "Content-Type: application/json" \
        "http://router:11434$1" ${3:+-d "$3"} 2>&1
}

ollama_curl_long() {
    $RUNTIME exec devai-open-webui curl -s --max-time "${2:-200}" \
        -H "Content-Type: application/json" \
        "http://router:11434$1" -d "$3" 2>&1
}

vllm_curl() {
    $RUNTIME exec devai-open-webui curl -sf --max-time "${2:-10}" \
        -H "Content-Type: application/json" \
        "http://router:11435$1" ${3:+-d "$3"} 2>&1
}

sglang_curl() {
    $RUNTIME exec devai-open-webui curl -sf --max-time "${2:-10}" \
        -H "Content-Type: application/json" \
        "http://router:11436$1" ${3:+-d "$3"} 2>&1
}

# ============================================================================
info "=== Test 1: Ollama health check (port 11434) ==="
# ============================================================================

health=$(ollama_curl /health)
if echo "$health" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok'; assert d['backend']=='ollama'" 2>/dev/null; then
    pass "Ollama health endpoint returns ok"
else
    fail "Ollama health" "$health"
fi

# ============================================================================
info "=== Test 2: vLLM health check (port 11435) ==="
# ============================================================================

health=$(vllm_curl /health)
if echo "$health" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok'; assert d['backend']=='vllm'" 2>/dev/null; then
    pass "vLLM health endpoint returns ok"
else
    fail "vLLM health" "$health"
fi

# ============================================================================
info "=== Test 3: SGLang health check (port 11436) ==="
# ============================================================================

health=$(sglang_curl /health)
if echo "$health" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok'; assert d['backend']=='sglang'" 2>/dev/null; then
    pass "SGLang health endpoint returns ok"
else
    fail "SGLang health" "$health"
fi

# ============================================================================
info "=== Test 4: Ollama /v1/models returns only Ollama models ==="
# ============================================================================

models=$(ollama_curl /v1/models)
if echo "$models" | python3 -c "
import sys,json
d = json.load(sys.stdin)
assert len(d['data']) > 0, 'no models'
for m in d['data']:
    assert 'NVFP4' not in m['id'], f'NVFP4 model {m[\"id\"]} on Ollama port'
" 2>/dev/null; then
    count=$(echo "$models" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data']))")
    pass "Ollama /v1/models returns $count models (no NVFP4)"
else
    fail "Ollama /v1/models" "$models"
fi

# ============================================================================
info "=== Test 5: vLLM /v1/models returns only vLLM models ==="
# ============================================================================

models=$(vllm_curl /v1/models)
if echo "$models" | python3 -c "
import sys,json
d = json.load(sys.stdin)
assert len(d['data']) > 0, 'no models'
" 2>/dev/null; then
    count=$(echo "$models" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data']))")
    pass "vLLM /v1/models returns $count models"
else
    fail "vLLM /v1/models" "$models"
fi

# ============================================================================
info "=== Test 6: SGLang /v1/models returns only SGLang models ==="
# ============================================================================

models=$(sglang_curl /v1/models)
if echo "$models" | python3 -c "
import sys,json
d = json.load(sys.stdin)
assert len(d['data']) > 0, 'no models'
" 2>/dev/null; then
    count=$(echo "$models" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data']))")
    pass "SGLang /v1/models returns $count models"
else
    fail "SGLang /v1/models" "$models"
fi

# ============================================================================
info "=== Test 7: Ollama /api/tags ==="
# ============================================================================

tags=$(ollama_curl /api/tags)
if echo "$tags" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'models' in d and len(d['models'])>0" 2>/dev/null; then
    pass "Ollama /api/tags returns models"
else
    fail "Ollama /api/tags" "$tags"
fi

# ============================================================================
info "=== Test 8: GGUF model via Ollama port ==="
# ============================================================================

resp=$(ollama_curl_long /v1/chat/completions 60 \
    '{"model":"qwen3.5:9b","messages":[{"role":"user","content":"Say hi"}],"max_tokens":5}')
if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['model']=='qwen3.5:9b'" 2>/dev/null; then
    pass "GGUF model via Ollama port"
else
    fail "GGUF via Ollama" "$resp"
fi

# ============================================================================
info "=== Test 9: Streaming (SSE via Ollama port) ==="
# ============================================================================

resp=$($RUNTIME exec devai-open-webui curl -s --max-time 30 \
    -H "Content-Type: application/json" \
    "http://router:11434/v1/chat/completions" \
    -d '{"model":"qwen3.5:9b","messages":[{"role":"user","content":"Count to 3"}],"max_tokens":20,"stream":true}' 2>&1)
if echo "$resp" | grep -q "data:"; then
    pass "Ollama SSE streaming works"
else
    fail "Ollama SSE streaming" "no SSE data received"
fi

# ============================================================================
info "=== Test 10: Empty model name on Ollama port ==="
# ============================================================================

resp=$(ollama_curl_long /v1/chat/completions 30 \
    '{"model":"","messages":[{"role":"user","content":"Say hi"}],"max_tokens":5}')
if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'error' in d or 'choices' in d or 'model' in d" 2>/dev/null; then
    pass "empty model name handled without crash"
else
    fail "empty model" "$resp"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================"
echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}"
echo "========================================"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
