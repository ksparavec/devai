#!/usr/bin/env bash
# Live integration tests for the SGLang backend through gpu-arbiter (port 11436).
#
# Mirror of tests/test-router-vllm.sh — same six-test structure, same
# cache-driven model discovery, same skip semantics. Skips cleanly
# (exit 0) when no fitting SGLang model is in the cache, which is the
# expected state on hardware where the upstream lmsysorg/sglang image's
# flashinfer FP4 path can't compile sm120 kernels (see docs/backends.md
# evidence-kind taxonomy).
#
# Prerequisites:
#   - `make cache-up`
#   - At least one HF model in deploy/.sglang-reasoning-cache.json with
#     fits=true at the host VRAM band
#   - GPU available
#
# Wall time: ~3-5 minutes when SGLang is loadable; <1 second when not.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${CONTAINER_RUNTIME:-podman}"
EXEC_HOST="${TEST_EXEC_HOST:-devai-open-webui}"
ROUTER_INTERNAL="${TEST_ROUTER_HOST:-router}"
SGLANG_CACHE="$REPO_ROOT/deploy/.sglang-reasoning-cache.json"
OLLAMA_CACHE="$REPO_ROOT/deploy/.ollama-reasoning-cache.json"
HOST_VRAM_GB="${HOST_VRAM_GB:-${GPU_MEMORY_GB:-24}}"
COLD_START_S=${COLD_START_S:-600}
WARM_CHAT_S=${WARM_CHAT_S:-30}

GREEN='\033[32m'
RED='\033[31m'
YELLOW='\033[33m'
GRAY='\033[2m'
NC='\033[0m'

# Same cleanup as test-router-vllm.sh — see comments there.
restore_placeholders() {
    for c in devai-vllm devai-sglang; do
        $RUNTIME rm -f "$c" >/dev/null 2>&1 || true
    done
    cd "$REPO_ROOT" && make cache-up >/dev/null 2>&1 || true
}
trap restore_placeholders EXIT

PASS=0
FAIL=0
SKIP=0
pass() { ((PASS++)); echo -e "  ${GREEN}PASS${NC} $1"; }
fail() { ((FAIL++)); echo -e "  ${RED}FAIL${NC} $1: $2"; }
skip() { ((SKIP++)); echo -e "  ${GRAY}SKIP${NC} $1: $2"; }
info() { echo -e "${YELLOW}$1${NC}"; }

discover_sglang_models() {
    python3 - "$SGLANG_CACHE" "$HOST_VRAM_GB" <<'PY'
import json, sys
cache_path, vram = sys.argv[1], int(sys.argv[2])
try:
    with open(cache_path) as fh:
        cache = json.load(fh)
except (OSError, json.JSONDecodeError):
    sys.exit(0)
fits = []
for entry in cache.values():
    if not isinstance(entry, dict) or not entry.get("aliases"):
        continue
    if entry.get("capability") in ("error", "unsupported_arch"):
        continue
    band = (entry.get("probes") or {}).get(str(vram))
    if not isinstance(band, dict):
        continue
    smallest_ctx = None
    smallest_total = None
    for ctx_str, cell in band.items():
        if not isinstance(cell, dict) or not cell.get("fits"):
            continue
        try:
            ctx_val = int(ctx_str)
        except ValueError:
            continue
        total = cell.get("actual_vram_gb") or 0
        if smallest_ctx is None or ctx_val < smallest_ctx:
            smallest_ctx = ctx_val
            smallest_total = total
    if smallest_ctx:
        fits.append((smallest_total or 0, entry["aliases"][0], smallest_ctx))
fits.sort()
for _, name, ctx in fits:
    print(f"{name}\t{ctx}")
PY
}

discover_ollama_model() {
    python3 - "$OLLAMA_CACHE" "$HOST_VRAM_GB" <<'PY'
import json, sys
cache_path, vram = sys.argv[1], int(sys.argv[2])
try:
    with open(cache_path) as fh:
        cache = json.load(fh)
except (OSError, json.JSONDecodeError):
    sys.exit(0)
for entry in cache.values():
    if not isinstance(entry, dict) or not entry.get("aliases"):
        continue
    if entry.get("capability") in ("error", "unsupported_arch"):
        continue
    band = (entry.get("probes") or {}).get(str(vram))
    if not isinstance(band, dict):
        continue
    for ctx_str, cell in band.items():
        if isinstance(cell, dict) and cell.get("fully_on_gpu"):
            print(entry["aliases"][0])
            sys.exit(0)
PY
}

curl_post() {
    local path="$1" timeout="$2" body="$3"
    $RUNTIME exec "$EXEC_HOST" curl -s --max-time "$timeout" \
        -H "Content-Type: application/json" \
        "http://$ROUTER_INTERNAL:11436$path" \
        -d "$body" 2>&1
}

curl_get() {
    local path="$1" timeout="${2:-10}"
    $RUNTIME exec "$EXEC_HOST" curl -sf --max-time "$timeout" \
        "http://$ROUTER_INTERNAL:11436$path" 2>&1
}

curl_get_ollama_chat() {
    local timeout="$1" body="$2"
    $RUNTIME exec "$EXEC_HOST" curl -s --max-time "$timeout" \
        -H "Content-Type: application/json" \
        "http://$ROUTER_INTERNAL:11434/v1/chat/completions" \
        -d "$body" 2>&1
}

assert_chat_ok() {
    local label="$1" resp="$2"
    if echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert 'choices' in d, f'no choices in response: {list(d.keys())}'
assert d['choices'], 'choices array empty'
content = (d['choices'][0].get('message') or {}).get('content', '')
assert isinstance(content, str), f'content not string: {type(content).__name__}'
" 2>/dev/null; then
        pass "$label"
        return 0
    fi
    fail "$label" "${resp:0:300}"
    return 1
}

info "=== SGLang router integration tests ==="
info "  router exec via:    $EXEC_HOST → $ROUTER_INTERNAL:11436"
info "  sglang cache:       $SGLANG_CACHE"
info "  host vram:          ${HOST_VRAM_GB}G"

models_table="$(discover_sglang_models)"
if [ -z "$models_table" ]; then
    info ""
    skip "all tests" "no fitting SGLang models in cache"
    info "  This is the expected state when the upstream lmsysorg/sglang"
    info "  image can't compile FP4 kernels (see docs/backends.md). Once"
    info "  SGLang loads cleanly here, run \`make probe-sglang\` and rerun."
    echo ""
    echo "========================================"
    echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${GRAY}${SKIP} skipped${NC}"
    echo "========================================"
    exit 0
fi

PRIMARY_NAME=$(echo "$models_table" | head -1 | cut -f1)
PRIMARY_CTX=$(echo "$models_table" | head -1 | cut -f2)
SECONDARY_NAME=$(echo "$models_table" | sed -n '2p' | cut -f1)
SECONDARY_CTX=$(echo "$models_table" | sed -n '2p' | cut -f2)

info "  primary model:      $PRIMARY_NAME @ $PRIMARY_CTX"
if [ -n "$SECONDARY_NAME" ]; then
    info "  secondary model:    $SECONDARY_NAME @ $SECONDARY_CTX"
fi

info ""
info "=== Test 1: cold start + chat round-trip on port 11436 ==="
resp=$(curl_post /v1/chat/completions "$COLD_START_S" \
    "{\"model\":\"${PRIMARY_NAME}@${PRIMARY_CTX}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with a single word.\"}],\"max_tokens\":16}")
assert_chat_ok "SGLang serves chat completion" "$resp"

info ""
info "=== Test 2: /health JSON shape ==="
health=$(curl_get /health)
if echo "$health" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert d.get('backend') == 'sglang', f'wrong backend: {d.get(\"backend\")}'
assert d.get('status') == 'ok'
assert 'running' in d
assert 'current_model' in d
" 2>/dev/null; then
    pass "/health returns expected fields"
else
    fail "/health" "${health:0:200}"
fi

info ""
info "=== Test 3: <name>@<ctx> recreates on context change ==="
ALT_CTX=$((PRIMARY_CTX / 2))
if [ "$ALT_CTX" -lt 4096 ]; then ALT_CTX=4096; fi
if [ "$ALT_CTX" = "$PRIMARY_CTX" ]; then
    skip "ctx-switch" "primary ctx already at minimum (4K)"
else
    info "  switching $PRIMARY_NAME from ${PRIMARY_CTX} → ${ALT_CTX}..."
    resp=$(curl_post /v1/chat/completions "$COLD_START_S" \
        "{\"model\":\"${PRIMARY_NAME}@${ALT_CTX}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with a single word.\"}],\"max_tokens\":16}")
    assert_chat_ok "SGLang serves at the new ctx" "$resp"
fi

info ""
info "=== Test 4: model switch ==="
if [ -z "$SECONDARY_NAME" ]; then
    skip "model-switch" "fewer than 2 fitting SGLang models cached"
else
    info "  switching to $SECONDARY_NAME @ $SECONDARY_CTX..."
    resp=$(curl_post /v1/chat/completions "$COLD_START_S" \
        "{\"model\":\"${SECONDARY_NAME}@${SECONDARY_CTX}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with a single word.\"}],\"max_tokens\":16}")
    if assert_chat_ok "SGLang model switch worked" "$resp"; then
        health=$(curl_get /health)
        cur=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin).get('current_model',''))" 2>/dev/null || echo '?')
        if [ "$cur" = "$SECONDARY_NAME" ]; then
            pass "/health.current_model reflects the switch"
        else
            fail "/health.current_model" "expected $SECONDARY_NAME, got '$cur'"
        fi
    fi
fi

info ""
info "=== Test 5: GPU exclusion (Ollama request stops SGLang) ==="
ollama_model=$(discover_ollama_model)
if [ -z "$ollama_model" ]; then
    skip "gpu-exclusion" "no fitting Ollama model in cache"
else
    info "  hitting Ollama with $ollama_model — should drain SGLang..."
    curl_get_ollama_chat 60 \
        "{\"model\":\"$ollama_model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}" \
        >/dev/null 2>&1
    sleep 5
    health=$(curl_get /health)
    running=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin)['running'])" 2>/dev/null || echo '?')
    if [ "$running" = "False" ]; then
        pass "SGLang drained (health.running=False) after Ollama request"
    else
        fail "GPU exclusion" "SGLang health still reports running=$running"
    fi
fi

info ""
info "=== Test 6: max_tokens forwarding ==="
info "  re-warming primary model for parameter test..."
resp=$(curl_post /v1/chat/completions "$COLD_START_S" \
    "{\"model\":\"${PRIMARY_NAME}@${PRIMARY_CTX}\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a long story.\"}],\"max_tokens\":1}")
if echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert 'choices' in d
content = (d['choices'][0].get('message') or {}).get('content', '')
assert len(content) < 80, f'expected ≤80 chars at max_tokens=1, got {len(content)}'
" 2>/dev/null; then
    pass "max_tokens=1 produces short response"
else
    fail "max_tokens forwarding" "${resp:0:300}"
fi

echo ""
echo "========================================"
echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${GRAY}${SKIP} skipped${NC}"
echo "========================================"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
