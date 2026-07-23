#!/usr/bin/env bash
# Live integration tests for the vLLM backend through gpu-arbiter (port 11435).
#
# Drives real chat completions through the router and verifies:
#   1. Cold container start + chat round-trip
#   2. /health JSON shape after a successful launch
#   3. <name>@<ctx> override triggers a context-only recreate (Phase 0)
#   4. Model switch triggers a model recreate (when ≥ 2 models cached)
#   5. GPU exclusion: an Ollama request drains vLLM
#   6. Parameter forwarding: max_tokens=1 produces a short response
#
# Prerequisites:
#   - `make cache-up` (router + Ollama + vLLM placeholder running)
#   - At least one HF model in deploy/.vllm-reasoning-cache.json with
#     fits=true at the host VRAM band
#   - GPU available
#
# The test reads the live cache to choose models — no hardcoded names.
# When fewer cached models exist than a particular test needs, that
# test is reported as SKIP (not FAIL). When the vllm cache has zero
# fitting entries, every test is skipped and the script exits 0.
#
# Wall time: ~3-5 minutes (one cold start ≈ 60s; ctx + model switches
# add one cold start each).

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${CONTAINER_RUNTIME:-podman}"
EXEC_HOST="${TEST_EXEC_HOST:-devai-open-webui}"
ROUTER_INTERNAL="${TEST_ROUTER_HOST:-router}"
VLLM_CACHE="$REPO_ROOT/deploy/.vllm-reasoning-cache.json"
OLLAMA_CACHE="$REPO_ROOT/deploy/.ollama-reasoning-cache.json"
HOST_VRAM_GB="${HOST_VRAM_GB:-${GPU_MEMORY_GB:-24}}"
# The probe cache records what FITS, not what is DOWNLOADED -- on this host
# it advertises 16 fitting models while 5 have weights. Selecting an absent
# one tests the store, not the router, so discovery filters on the store the
# router actually launches from.
VLLM_MODELS_DIR="${VLLM_MODELS_DIR:-/var/cache/devai/vllm}"

# Tunable timeouts. Cold vLLM start with CUDA-graph capture is usually
# 30-90s on consumer NVFP4 weights, but the router's stopOtherBackends
# + containerStop + containerRemove + containerRecreate + waitForHealthy
# chain can stretch the wall time during the very first request when
# transitioning from the `sleep infinity` placeholder to a real workload.
# 600s default gives generous headroom; override via env when iterating.
COLD_START_S=${COLD_START_S:-600}
WARM_CHAT_S=${WARM_CHAT_S:-30}

GREEN='\033[32m'
RED='\033[31m'
YELLOW='\033[33m'
GRAY='\033[2m'
NC='\033[0m'

# Tests trigger router-driven recreates that leave the live vLLM
# container (with the dynamic entrypoint) behind. Compose can't reuse
# `devai-vllm`/`devai-sglang` on a subsequent `cache-up` while those
# are around, so always wipe + restore the placeholders on exit.
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

# ── Cache discovery ──────────────────────────────────────────────────────────
#
# Pick the smallest fitting model at the host VRAM band so cold starts
# are fast. When two are needed (model-switch test), pick two distinct
# ones. Returns names one per line; empty stdout means nothing fits.

discover_vllm_models() {
    python3 - "$VLLM_CACHE" "$HOST_VRAM_GB" "$VLLM_MODELS_DIR" <<'PY'
import json, os, sys
cache_path, vram = sys.argv[1], int(sys.argv[2])
store = sys.argv[3] if len(sys.argv) > 3 else ""
# When the store is not visible at all (CI, a container without the bind),
# fall back to the un-gated list rather than reporting "nothing fits" --
# same degradation the router's checkModelWeights uses.
store_visible = bool(store) and os.path.isdir(store)
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
    for ctx_str, cell in band.items():
        if not isinstance(cell, dict) or not cell.get("fits"):
            continue
        try:
            ctx_val = int(ctx_str)
        except ValueError:
            continue
        if smallest_ctx is None or ctx_val < smallest_ctx:
            smallest_ctx = ctx_val
    if smallest_ctx is not None:
        name = entry["aliases"][0]
        if store_visible and not os.path.isdir(os.path.join(store, name)):
            continue
        # Sort key: weight bytes on disk, NOT actual_vram_gb (which is
        # the engine's pre-allocated pool — same across NVFP4 and BF16
        # models at fixed gpu_memory_utilization). Cold-start time
        # scales with weight transfer + GPU copy, so smallest weights
        # = fastest swap = most reliable test under tight timeouts.
        weight_gb = float(entry.get("size_gb") or 0.0)
        fits.append((weight_gb, entry["aliases"][0], smallest_ctx))
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

# ── HTTP helpers (curl from inside devai-open-webui to use devai-net) ────────

curl_post() {
    local path="$1" timeout="$2" body="$3"
    $RUNTIME exec "$EXEC_HOST" curl -s --max-time "$timeout" \
        -H "Content-Type: application/json" \
        "http://$ROUTER_INTERNAL:11435$path" \
        -d "$body" 2>&1
}

curl_get() {
    local path="$1" timeout="${2:-10}"
    $RUNTIME exec "$EXEC_HOST" curl -sf --max-time "$timeout" \
        "http://$ROUTER_INTERNAL:11435$path" 2>&1
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
    # Structured-reasoning models (Qwen3, etc.) return content=null and
    # put their output into message.reasoning_content / .reasoning when
    # max_tokens is tight. Both shapes prove the round-trip; the test is
    # 'backend reachable + producing output', not 'content field is set'.
    if echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert 'choices' in d, f'no choices in response: {list(d.keys())}'
assert d['choices'], 'choices array empty'
msg = d['choices'][0].get('message') or {}
candidates = [
    msg.get('content'),
    msg.get('reasoning_content'),
    msg.get('reasoning'),
    msg.get('refusal'),
]
texts = [c for c in candidates if isinstance(c, str) and c.strip()]
assert texts, f'no non-empty content/reasoning/refusal in message: keys={sorted(msg.keys())}'
" 2>/dev/null; then
        pass "$label"
        return 0
    fi
    fail "$label" "${resp:0:300}"
    return 1
}

# ── Discovery + setup ────────────────────────────────────────────────────────

info "=== vLLM router integration tests ==="
info "  router exec via:    $EXEC_HOST → $ROUTER_INTERNAL:11435"
info "  vllm cache:         $VLLM_CACHE"
info "  host vram:          ${HOST_VRAM_GB}G"

models_table="$(discover_vllm_models)"
if [ -z "$models_table" ]; then
    info ""
    skip "all tests" "no fitting vLLM models in cache (run \`make probe-vllm\` first)"
    echo ""
    echo "========================================"
    echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${GRAY}${SKIP} skipped${NC}"
    echo "========================================"
    exit 0
fi

# Parse: first model + ctx, second model + ctx (if any)
PRIMARY_NAME=$(echo "$models_table" | head -1 | cut -f1)
PRIMARY_CTX=$(echo "$models_table" | head -1 | cut -f2)
SECONDARY_NAME=$(echo "$models_table" | sed -n '2p' | cut -f1)
SECONDARY_CTX=$(echo "$models_table" | sed -n '2p' | cut -f2)

info "  primary model:      $PRIMARY_NAME @ $PRIMARY_CTX"
if [ -n "$SECONDARY_NAME" ]; then
    info "  secondary model:    $SECONDARY_NAME @ $SECONDARY_CTX"
else
    info "  secondary model:    (none — model-switch test will skip)"
fi

# ── Test 1: cold start + chat round-trip ─────────────────────────────────────

info ""
info "=== Test 1: cold start + chat round-trip on port 11435 ==="
info "  cold start budget: ${COLD_START_S}s"
info "  (typical: 30-90s; budget allows for placeholder transition + recreate)"

# When this test runs after another test that triggered a vLLM recreate
# (test-e2e or a trap-induced cache-up), the placeholder→workload
# transition can leave the router and container briefly out of sync —
# the first request after such a transition occasionally returns empty
# (typically a proxy 502 from a not-yet-ready upstream). One retry,
# with a 5s settle in between, absorbs that transient without masking
# real failures (a true broken state still fails on the second try).
do_test1_chat() {
    curl_post /v1/chat/completions "$COLD_START_S" \
        "{\"model\":\"${PRIMARY_NAME}@${PRIMARY_CTX}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with a single word.\"}],\"max_tokens\":16}"
}
sleep 3
resp=$(do_test1_chat)
if [ -z "$resp" ] || ! echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert 'choices' in d and d['choices']
" 2>/dev/null; then
    info "  first attempt returned empty/invalid; sleeping 5s and retrying once..."
    sleep 5
    resp=$(do_test1_chat)
fi
assert_chat_ok "vLLM serves chat completion" "$resp"

# ── Test 2: /health shape after warmup ───────────────────────────────────────

info ""
info "=== Test 2: /health JSON shape ==="
health=$(curl_get /health)
if echo "$health" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert d.get('backend') == 'vllm', f'wrong backend: {d.get(\"backend\")}'
assert d.get('status') == 'ok', f'status not ok: {d.get(\"status\")}'
assert 'running' in d, 'missing running field'
assert 'current_model' in d, 'missing current_model field'
" 2>/dev/null; then
    pass "/health returns expected fields"
else
    fail "/health" "${health:0:200}"
fi

# ── Test 3: ctx-only switch triggers recreate (Phase 0 currentContext) ──────

info ""
info "=== Test 3: <name>@<ctx> recreates on context change ==="
# A different ctx on the same model must recreate the container (the
# router tracks currentContext after Phase 0). We can detect the
# recreate by watching health.current_model briefly hit "" between
# stop and re-launch — or, more reliably, by triggering a different
# ctx and asserting the chat still succeeds (proves the new container
# was launched with --max-model-len matching the override).
ALT_CTX=$((PRIMARY_CTX / 2))
if [ "$ALT_CTX" -lt 4096 ]; then ALT_CTX=4096; fi
if [ "$ALT_CTX" = "$PRIMARY_CTX" ]; then
    skip "ctx-switch" "primary ctx already at minimum (4K)"
else
    info "  switching $PRIMARY_NAME from ${PRIMARY_CTX} → ${ALT_CTX}..."
    resp=$(curl_post /v1/chat/completions "$COLD_START_S" \
        "{\"model\":\"${PRIMARY_NAME}@${ALT_CTX}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with a single word.\"}],\"max_tokens\":16}")
    assert_chat_ok "vLLM serves at the new ctx" "$resp"
fi

# ── Test 4: model switch (only when 2+ models cached) ───────────────────────

info ""
info "=== Test 4: model switch ==="
if [ -z "$SECONDARY_NAME" ]; then
    skip "model-switch" "fewer than 2 fitting vLLM models cached"
else
    info "  switching to $SECONDARY_NAME @ $SECONDARY_CTX..."
    resp=$(curl_post /v1/chat/completions "$COLD_START_S" \
        "{\"model\":\"${SECONDARY_NAME}@${SECONDARY_CTX}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with a single word.\"}],\"max_tokens\":16}")
    if assert_chat_ok "vLLM model switch worked" "$resp"; then
        health=$(curl_get /health)
        cur=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin).get('current_model',''))" 2>/dev/null || echo '?')
        if [ "$cur" = "$SECONDARY_NAME" ]; then
            pass "/health.current_model reflects the switch"
        else
            fail "/health.current_model" "expected $SECONDARY_NAME, got '$cur'"
        fi
    fi
fi

# ── Test 5: GPU exclusion — Ollama request stops vLLM ───────────────────────

info ""
info "=== Test 5: GPU exclusion (Ollama request stops vLLM) ==="
ollama_model=$(discover_ollama_model)
if [ -z "$ollama_model" ]; then
    skip "gpu-exclusion" "no fitting Ollama model in cache"
else
    info "  hitting Ollama with $ollama_model — should drain vLLM..."
    curl_get_ollama_chat 60 \
        "{\"model\":\"$ollama_model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}" \
        >/dev/null 2>&1
    sleep 5
    vllm_status=$($RUNTIME inspect -f '{{.State.Status}}' devai-vllm 2>/dev/null || echo absent)
    # The router replaces the live container with the sleeping placeholder
    # via stopOtherBackends → containerStop. After the Ollama request,
    # devai-vllm should either be `exited`/`stopped`, the placeholder
    # `sleep infinity` (Status=running but the inference workload is gone),
    # or completely absent. The smoke test's purpose is "vLLM is not
    # actively serving" — checked indirectly by health.running being false.
    health=$(curl_get /health)
    running=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin)['running'])" 2>/dev/null || echo '?')
    if [ "$running" = "False" ]; then
        pass "vLLM drained (health.running=False) after Ollama request"
    else
        fail "GPU exclusion" "vLLM health still reports running=$running"
    fi
fi

# ── Test 6: parameter forwarding ─────────────────────────────────────────────

info ""
info "=== Test 6: max_tokens forwarding ==="
info "  re-warming primary model for parameter test..."
resp=$(curl_post /v1/chat/completions "$COLD_START_S" \
    "{\"model\":\"${PRIMARY_NAME}@${PRIMARY_CTX}\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a long story about dragons.\"}],\"max_tokens\":1}")
if echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert 'choices' in d, 'no choices'
msg = d['choices'][0].get('message') or {}
# Structured-reasoning models put output in reasoning/reasoning_content
# when content is short or null. Sum lengths across all standard payload
# fields — max_tokens=1 should still produce ≤80 chars total regardless
# of which field carries it.
total = sum(
    len(s) for s in (
        msg.get('content'),
        msg.get('reasoning_content'),
        msg.get('reasoning'),
    ) if isinstance(s, str)
)
assert total < 80, f'expected ≤80 chars total at max_tokens=1, got {total}'
" 2>/dev/null; then
    pass "max_tokens=1 produces short response"
else
    fail "max_tokens forwarding" "${resp:0:300}"
fi

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${GRAY}${SKIP} skipped${NC}"
echo "========================================"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
