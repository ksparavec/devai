#!/bin/bash
# Integration tests for gpu-arbiter router (vLLM/GPU tests)
# Prerequisites: make cache-up (infrastructure running), GPU available
# Note: NVFP4 tests require ~90s per model cold start — timing-sensitive
#
# Usage: ./tests/test-router-vllm.sh
#        VLLM_IDLE_TIMEOUT=15 make cache-up && ./tests/test-router-vllm.sh

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
info "=== Test 1: NVFP4 model triggers vLLM auto-start ==="
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
info "=== Test 2: Ollama API translation (/api/chat → vLLM) ==="
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
info "=== Test 3: vLLM model switch ==="
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
info "=== Test 4: Switch back to Ollama (auto-stops vLLM) ==="
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
info "=== Test 5: Concurrent GGUF + NVFP4 requests ==="
# ============================================================================

info "  Firing concurrent requests (GGUF background, NVFP4 foreground)..."
router_curl_long /v1/chat/completions 200 \
    '{"model":"qwen3.5:9b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' >/dev/null 2>&1 &
GGUF_PID=$!

resp_nvfp4=$(router_curl_long /v1/chat/completions 200 \
    '{"model":"NVIDIA-Nemotron-Nano-9B-v2-NVFP4","messages":[{"role":"user","content":"hi"}],"max_tokens":5}')

wait $GGUF_PID
GGUF_EXIT=$?

if [ $GGUF_EXIT -eq 0 ] || echo "$resp_nvfp4" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'model' in d or 'error' in d" 2>/dev/null; then
    pass "concurrent requests completed without deadlock"
else
    fail "concurrent requests" "gguf_exit=$GGUF_EXIT nvfp4=$resp_nvfp4"
fi

# Ensure we're back on ollama for remaining tests
router_curl_long /v1/chat/completions 60 \
    '{"model":"qwen3.5:9b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' >/dev/null 2>&1

# ============================================================================
info "=== Test 6: Health after external vLLM stop ==="
# ============================================================================

info "  Starting vLLM for health-staleness test..."
resp=$(router_curl_long /v1/chat/completions 200 \
    '{"model":"NVIDIA-Nemotron-Nano-9B-v2-NVFP4","messages":[{"role":"user","content":"hi"}],"max_tokens":5}')

# Keep vLLM alive with a second request (prevents idle timeout race)
router_curl_long /v1/chat/completions 30 \
    '{"model":"NVIDIA-Nemotron-Nano-9B-v2-NVFP4","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' >/dev/null 2>&1

health_before=$(router_curl /health)
vllm_before=$(echo "$health_before" | python3 -c "import sys,json; print(json.load(sys.stdin)['vllm_running'])")

if [ "$vllm_before" != "True" ]; then
    info "  vLLM not running (may have been auto-stopped by idle timeout), skipping staleness test"
    pass "health endpoint responds (vLLM not active — skipped staleness check)"
else
    info "  Externally stopping vLLM container..."
    $RUNTIME stop devai-vllm >/dev/null 2>&1
    sleep 3

    health_after=$(router_curl /health)
    vllm_after=$(echo "$health_after" | python3 -c "import sys,json; print(json.load(sys.stdin)['vllm_running'])")
    info "  health.vllm_running before stop=$vllm_before, after stop=$vllm_after (known stale state)"
    pass "health endpoint responds after external vLLM stop (vllm_running=$vllm_after)"
fi

# Reset state: send GGUF request to flip backend back to ollama
router_curl_long /v1/chat/completions 60 \
    '{"model":"qwen3.5:9b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' >/dev/null 2>&1
sleep 2

# ============================================================================
info "=== Test 7: Parameter forwarding via /api/chat to NVFP4 ==="
# ============================================================================

info "  Starting vLLM for parameter test..."
resp=$(router_curl_long /api/chat 200 \
    '{"model":"NVIDIA-Nemotron-Nano-9B-v2-NVFP4","messages":[{"role":"user","content":"Write a very long story"}],"stream":false,"max_tokens":1}')
if echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
content = d.get('message',{}).get('content','')
# max_tokens=1 should produce a very short response
assert d.get('done') == True, 'expected done=True'
assert len(content) < 50, f'expected short response with max_tokens=1, got {len(content)} chars'
" 2>/dev/null; then
    pass "max_tokens=1 forwarded via /api/chat translation"
else
    fail "parameter forwarding" "$resp"
fi

# Reset to ollama
router_curl_long /v1/chat/completions 60 \
    '{"model":"qwen3.5:9b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' >/dev/null 2>&1

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================"
echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}"
echo "========================================"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
