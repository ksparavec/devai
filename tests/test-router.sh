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
info "=== Test 5: vLLM /v1/models returns OpenAI list shape ==="
# ============================================================================
# Symmetric with Test 6 — `data: null` is a valid empty state when no
# vLLM model has a fitting probe entry. Shape is what we test here;
# actual chat completions are exercised by tests/test-router-vllm.sh.

models=$(vllm_curl /v1/models)
if echo "$models" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert d.get('object') == 'list', f\"object={d.get('object')!r}\"
data = d.get('data')
assert data is None or isinstance(data, list), f'data not list/null: {type(data).__name__}'
" 2>/dev/null; then
    count=$(echo "$models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('data') or []))")
    pass "vLLM /v1/models returns $count models (valid OpenAI list shape)"
else
    fail "vLLM /v1/models" "$models"
fi

# ============================================================================
info "=== Test 6: SGLang /v1/models returns OpenAI list shape ==="
# ============================================================================
# The router synthesizes SGLang rows from .sglang-reasoning-cache.json.
# When the cache has no fitting entries (e.g. SGLang+FP4 fails on this
# image), `data` is null — which is a VALID empty state, not a router
# bug. Test passes when the response shape is correct (object: "list");
# entry count is informational.

models=$(sglang_curl /v1/models)
if echo "$models" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert d.get('object') == 'list', f\"object={d.get('object')!r}\"
data = d.get('data')
assert data is None or isinstance(data, list), f'data not list/null: {type(data).__name__}'
" 2>/dev/null; then
    count=$(echo "$models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('data') or []))")
    pass "SGLang /v1/models returns $count models (valid OpenAI list shape)"
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
# Pick the SMALLEST Ollama tag by weight bytes — cold-start time scales
# with weight transfer + GPU copy, so smallest = fastest round-trip
# under the curl timeout. After test-probe-ollama-idempotent cycles
# devai-ollama, /api/tags transiently returns size=0 entries while
# disk indexing finishes (~5-10s); poll until at least one entry has
# size>0 before deciding. Falls through to alphabetic-first only after
# the timeout to avoid a flaky "no tags" failure when the daemon is
# in the middle of indexing.
TEST_MODEL=""
for attempt in 1 2 3 4 5 6 7 8 9 10; do
    TEST_MODEL=$(ollama_curl /api/tags | python3 -c "
import sys, json
d = json.load(sys.stdin)
models = d.get('models', []) or []
sized = [m for m in models if int(m.get('size') or 0) > 0]
if sized:
    sized.sort(key=lambda m: int(m['size']))
    print(sized[0]['name'])
" 2>/dev/null)
    [ -n "$TEST_MODEL" ] && break
    sleep 2
done
if [ -z "$TEST_MODEL" ]; then
    # Last-ditch: pick whatever's there even with size=0.
    TEST_MODEL=$(ollama_curl /api/tags | python3 -c "
import sys, json
d = json.load(sys.stdin)
models = d.get('models', []) or []
print(sorted(m['name'] for m in models)[0] if models else '')
" 2>/dev/null)
fi
if [ -z "$TEST_MODEL" ]; then
    fail "GGUF via Ollama" "no Ollama tags downloaded — run 'make model-pull' first"
else
    # Cold-start budget: even the smallest GGUF can take 60-120s to
    # warm up after a fresh container cycle. 240s gives reasonable
    # headroom without blocking the suite for an unreasonable time.
    resp=$(ollama_curl_long /v1/chat/completions 240 \
        "{\"model\":\"$TEST_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":5}")
    if echo "$resp" | TEST_MODEL="$TEST_MODEL" python3 -c "
import os, sys, json
d = json.load(sys.stdin)
assert d['model'] == os.environ['TEST_MODEL'], f\"model={d.get('model')!r} want {os.environ['TEST_MODEL']!r}\"
" 2>/dev/null; then
        pass "GGUF model via Ollama port ($TEST_MODEL)"
    else
        fail "GGUF via Ollama" "$resp"
    fi
fi

# ============================================================================
info "=== Test 9: Streaming (SSE via Ollama port) ==="
# ============================================================================
# Reuses TEST_MODEL discovered above so the test stays in lockstep with
# whatever's on disk.

if [ -z "$TEST_MODEL" ]; then
    fail "Ollama SSE streaming" "no Ollama tags downloaded"
else
    resp=$($RUNTIME exec devai-open-webui curl -s --max-time 30 \
        -H "Content-Type: application/json" \
        "http://router:11434/v1/chat/completions" \
        -d "{\"model\":\"$TEST_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count to 3\"}],\"max_tokens\":20,\"stream\":true}" 2>&1)
    if echo "$resp" | grep -q "data:"; then
        pass "Ollama SSE streaming works ($TEST_MODEL)"
    else
        fail "Ollama SSE streaming" "no SSE data received"
    fi
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
