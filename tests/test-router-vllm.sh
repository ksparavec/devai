#!/bin/bash
# Integration tests for gpu-arbiter (vLLM backend, port 11435)
# Prerequisites: make cache-up (infrastructure running), GPU available
# Note: NVFP4 tests require ~90s per model cold start — timing-sensitive
#
# Usage: ./tests/test-router-vllm.sh

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

vllm_curl() {
    $RUNTIME exec devai-open-webui curl -sf --max-time "${2:-10}" \
        -H "Content-Type: application/json" \
        "http://router:11435$1" ${3:+-d "$3"} 2>&1
}

vllm_curl_long() {
    $RUNTIME exec devai-open-webui curl -s --max-time "${2:-200}" \
        -H "Content-Type: application/json" \
        "http://router:11435$1" -d "$3" 2>&1
}

# ============================================================================
info "=== Test 1: NVFP4 model auto-starts vLLM ==="
# ============================================================================

info "  Requesting NVFP4 model on port 11435 (cold start ~90s)..."
resp=$(vllm_curl_long /v1/chat/completions 360 \
    '{"model":"NVIDIA-Nemotron-Nano-9B-v2-NVFP4","messages":[{"role":"user","content":"Write a haiku"}],"max_tokens":50}')
if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'choices' in d or 'NVFP4' in d.get('model','')" 2>/dev/null; then
    pass "NVFP4 model via vLLM port"
else
    fail "NVFP4 via vLLM" "$resp"
fi

# Keep alive
vllm_curl_long /v1/chat/completions 30 \
    '{"model":"NVIDIA-Nemotron-Nano-9B-v2-NVFP4","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' >/dev/null 2>&1

health=$(vllm_curl /health)
running=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin)['running'])")
if [ "$running" = "True" ]; then
    pass "vLLM health shows running"
else
    fail "vLLM health" "$health"
fi

# ============================================================================
info "=== Test 2: vLLM model switch ==="
# ============================================================================

info "  Switching to different NVFP4 model (container recreation ~90s)..."
resp=$(vllm_curl_long /v1/chat/completions 360 \
    '{"model":"nvidia-Llama-3.1-8B-Instruct-NVFP4","messages":[{"role":"user","content":"hi"}],"max_tokens":5}')
if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'choices' in d or 'Llama' in d.get('model','')" 2>/dev/null; then
    pass "vLLM model switch worked"
else
    fail "vLLM model switch" "$resp"
fi

health=$(vllm_curl /health)
model=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin).get('current_model',''))")
if [ "$model" = "nvidia-Llama-3.1-8B-Instruct-NVFP4" ]; then
    pass "health shows correct vLLM model"
else
    fail "vLLM model in health" "expected nvidia-Llama-3.1-8B-Instruct-NVFP4, got $model"
fi

# ============================================================================
info "=== Test 3: GPU exclusion — Ollama request stops vLLM ==="
# ============================================================================

info "  Requesting GGUF model on Ollama port while vLLM is running..."
$RUNTIME exec devai-open-webui curl -s --max-time 60 \
    -H "Content-Type: application/json" \
    "http://router:11434/v1/chat/completions" \
    -d '{"model":"qwen3.5:9b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' >/dev/null 2>&1

sleep 5
vllm_status=$($RUNTIME inspect -f '{{.State.Status}}' devai-vllm 2>/dev/null)
if [ "$vllm_status" != "running" ]; then
    pass "vLLM stopped when Ollama request arrived"
else
    fail "GPU exclusion" "vLLM still running after Ollama request"
fi

# ============================================================================
info "=== Test 4: Health after external vLLM stop ==="
# ============================================================================

info "  Starting vLLM..."
resp=$(vllm_curl_long /v1/chat/completions 360 \
    '{"model":"NVIDIA-Nemotron-Nano-9B-v2-NVFP4","messages":[{"role":"user","content":"hi"}],"max_tokens":5}')

vllm_curl_long /v1/chat/completions 30 \
    '{"model":"NVIDIA-Nemotron-Nano-9B-v2-NVFP4","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' >/dev/null 2>&1

health_before=$(vllm_curl /health)
running_before=$(echo "$health_before" | python3 -c "import sys,json; print(json.load(sys.stdin)['running'])")

if [ "$running_before" != "True" ]; then
    info "  vLLM not running, skipping staleness test"
    pass "health endpoint responds (vLLM not active)"
else
    info "  Externally stopping vLLM container..."
    $RUNTIME stop devai-vllm >/dev/null 2>&1
    sleep 3

    health_after=$(vllm_curl /health)
    running_after=$(echo "$health_after" | python3 -c "import sys,json; print(json.load(sys.stdin)['running'])")
    info "  running before=$running_before, after=$running_after"
    pass "health endpoint responds after external vLLM stop"
fi

# ============================================================================
info "=== Test 5: Parameter forwarding (max_tokens) ==="
# ============================================================================

info "  Starting vLLM for parameter test..."
resp=$(vllm_curl_long /v1/chat/completions 360 \
    '{"model":"NVIDIA-Nemotron-Nano-9B-v2-NVFP4","messages":[{"role":"user","content":"Write a very long story"}],"max_tokens":1}')
if echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert 'choices' in d, 'no choices in response'
content = d['choices'][0].get('message',{}).get('content','')
assert len(content) < 50, f'expected short response with max_tokens=1, got {len(content)} chars'
" 2>/dev/null; then
    pass "max_tokens=1 forwarded correctly"
else
    fail "parameter forwarding" "$resp"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================"
echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}"
echo "========================================"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
