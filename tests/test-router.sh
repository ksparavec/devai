#!/bin/bash
# Integration tests for gpu-arbiter router
# Prerequisites: make cache-up (infrastructure running)
# Note: NVFP4 tests require ~90s per model cold start
#
# Usage: ./tests/test-router.sh
#        VLLM_IDLE_TIMEOUT=15 ./tests/test-router.sh  # fast idle test

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
info "=== Test 5: NVFP4 model triggers vLLM auto-start ==="
# ============================================================================

info "  Requesting NVFP4 model (vLLM cold start ~90s)..."
resp=$(router_curl_long /v1/chat/completions 200 \
    '{"model":"NVIDIA-Nemotron-Nano-9B-v2-NVFP4","messages":[{"role":"user","content":"Write a haiku"}],"max_tokens":50}')
if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'NVFP4' in d['model']" 2>/dev/null; then
    pass "NVFP4 model via /v1/chat/completions"
else
    fail "NVFP4 /v1/ routing" "$resp"
fi

# Keep vLLM alive for next tests
router_curl_long /v1/chat/completions 30 \
    '{"model":"NVIDIA-Nemotron-Nano-9B-v2-NVFP4","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' >/dev/null 2>&1

health=$(router_curl /health)
backend=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin)['active_backend'])")
vllm_running=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin)['vllm_running'])")
if [ "$backend" = "vllm" ] && [ "$vllm_running" = "True" ]; then
    pass "active backend switched to vllm"
else
    fail "backend switch" "backend=$backend vllm_running=$vllm_running"
fi

# ============================================================================
info "=== Test 6: Ollama API translation (/api/chat → vLLM) ==="
# ============================================================================

info "  Testing /api/chat with NVFP4 model (non-streaming)..."
resp=$(router_curl_long /api/chat 30 \
    '{"model":"NVIDIA-Nemotron-Nano-9B-v2-NVFP4","messages":[{"role":"user","content":"hi"}],"stream":false}')
if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['done']==True; assert 'content' in d['message']" 2>/dev/null; then
    pass "/api/chat non-streaming returns Ollama format"
else
    fail "/api/chat translation" "$resp"
fi

info "  Testing /api/chat streaming..."
resp=$($RUNTIME exec devai-open-webui curl -s --max-time 30 \
    -H "Content-Type: application/json" \
    "http://router:11434/api/chat" \
    -d '{"model":"NVIDIA-Nemotron-Nano-9B-v2-NVFP4","messages":[{"role":"user","content":"count to 3"}],"stream":true}' 2>&1)
if echo "$resp" | python3 -c "import sys,json; lines=[json.loads(l) for l in sys.stdin if l.strip()]; assert any(l.get('done') for l in lines)" 2>/dev/null; then
    pass "/api/chat streaming returns Ollama NDJSON format"
else
    fail "/api/chat streaming" "$(echo "$resp" | head -1)"
fi

# ============================================================================
info "=== Test 7: vLLM model switch ==="
# ============================================================================

info "  Switching to different NVFP4 model (container recreation ~90s)..."
resp=$(router_curl_long /v1/chat/completions 200 \
    '{"model":"nvidia-Llama-3.1-8B-Instruct-NVFP4","messages":[{"role":"user","content":"hi"}],"max_tokens":5}')
if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'Llama' in d['model']" 2>/dev/null; then
    pass "vLLM model switch worked"
else
    fail "vLLM model switch" "$resp"
fi

health=$(router_curl /health)
model=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin).get('vllm_model',''))")
if [ "$model" = "nvidia-Llama-3.1-8B-Instruct-NVFP4" ]; then
    pass "health shows correct vLLM model"
else
    fail "vLLM model in health" "expected nvidia-Llama-3.1-8B-Instruct-NVFP4, got $model"
fi

# ============================================================================
info "=== Test 8: Switch back to Ollama (auto-stops vLLM) ==="
# ============================================================================

info "  Requesting GGUF model while vLLM is active..."
resp=$(router_curl_long /v1/chat/completions 60 \
    '{"model":"qwen3.5:9b","messages":[{"role":"user","content":"Say hi"}],"max_tokens":5}')
if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['model']=='qwen3.5:9b'" 2>/dev/null; then
    pass "GGUF model works after switching from vLLM"
else
    fail "GGUF after vLLM" "$resp"
fi

sleep 5
vllm_status=$($RUNTIME inspect -f '{{.State.Status}}' devai-vllm 2>/dev/null)
if [ "$vllm_status" != "running" ]; then
    pass "vLLM container stopped after switching to Ollama"
else
    fail "vLLM stop" "vLLM still running"
fi

health=$(router_curl /health)
backend=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin)['active_backend'])")
if [ "$backend" = "ollama" ]; then
    pass "active backend restored to ollama"
else
    fail "backend restore" "expected ollama, got $backend"
fi

# ============================================================================
info "=== Test 9: Streaming (SSE via OpenAI API) ==="
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
info "=== Test 10: Non-POST requests pass through to Ollama ==="
# ============================================================================

resp=$(router_curl /api/tags)
if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'models' in d" 2>/dev/null; then
    pass "GET /api/tags passes through to Ollama"
else
    fail "passthrough" "$resp"
fi

# ============================================================================
info "=== Test 11: Idle timeout auto-stops vLLM ==="
# ============================================================================

info "  Starting vLLM via NVFP4 request..."
resp=$(router_curl_long /v1/chat/completions 200 \
    '{"model":"NVIDIA-Nemotron-Nano-9B-v2-NVFP4","messages":[{"role":"user","content":"hi"}],"max_tokens":5}')
if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'NVFP4' in d['model']" 2>/dev/null; then
    pass "vLLM started for idle timeout test"
else
    fail "vLLM start for idle test" "$resp"
fi

info "  Waiting for idle timeout (IDLE_TIMEOUT + 30s poll interval)..."
sleep 50

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
# Summary
# ============================================================================
echo ""
echo "========================================"
echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}"
echo "========================================"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
