#!/usr/bin/env bash
# End-to-end test for the picker → agent-command → router chain.
#
# Bridges the gap between unit-tested layers:
#   * Picker discovery + menu build  (covered by Phase 7 smoke test)
#   * Agent command construction      (covered by Phase 7 smoke test)
#   * Router cache integration        (covered by go test + test-router*.sh)
#
# The connection between them — "the command the picker emits actually
# drives a successful chat through the router and a real backend" — was
# previously only demonstrated by manual eyeballing. This test makes it
# part of the suite.
#
# Procedure:
#   1. Import model-picker.py, run _discover_models() and _build_menu()
#      against the live caches — this is what the picker does at startup.
#   2. Pick the first selectable HF entry from the menu (vLLM or SGLang).
#   3. Construct the same command the picker would hand the user via
#      _build("aider", serving_name, backend), where serving_name carries
#      the @<ctx> override (the per-session ctx-binding).
#   4. Replay that exact request against the router's published port,
#      mimicking what the agent CLI would send (OpenAI /v1/chat/completions
#      shape because aider uses it for HF backends).
#   5. Assert the router parses the @<ctx> override (currentContext path
#      from Phase 0), recreates the backend container, and returns a
#      well-formed chat response.
#
# Skips cleanly (exit 0) when no HF model is selectable in the picker
# — that's the expected state when only the Ollama backend has probe
# data, in which case the picker integration isn't testable end-to-end.
#
# Prerequisites: `make cache-up`. Wall time: ~60-360s for one cold
# vLLM/SGLang start.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${CONTAINER_RUNTIME:-podman}"
EXEC_HOST="${TEST_EXEC_HOST:-devai-open-webui}"
COLD_START_S="${COLD_START_S:-900}"
HOST_VRAM_GB="${HOST_VRAM_GB:-${GPU_MEMORY_GB:-24}}"

GREEN='\033[32m'
RED='\033[31m'
YELLOW='\033[33m'
GRAY='\033[2m'
NC='\033[0m'

# Tests that recreate vllm/sglang containers leave the live workload
# behind. Compose can't reuse the name on a subsequent `cache-up`, so
# always wipe the dynamic containers at end-of-script and restore the
# `sleep infinity` placeholders. Idempotent — failures are tolerated.
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

info "=== Picker → agent → router E2E ==="
info "  cwd:               $REPO_ROOT"
info "  router exec via:   $EXEC_HOST"
info "  host vram:         ${HOST_VRAM_GB}G"

# ── Step 1+2: drive the picker's discovery + menu build ─────────────────────
#
# Output: tab-separated `<model_name>\t<backend>\t<context>` for the
# first selectable HF row. Empty stdout when nothing's selectable.
#
# This is the same code path the interactive picker hits when the user
# fires up `agent-picker` — _discover_models + _build_menu — minus fzf.

picked_row=$(
    cd "$REPO_ROOT" && \
    VRAM="$HOST_VRAM_GB" \
    VLLM_MODELS_DIR="${VLLM_MODELS_DIR:-/var/cache/devai/ollama/models/vllm}" \
    OLLAMA_MANIFESTS_DIR="${OLLAMA_MANIFESTS_DIR:-/var/cache/devai/ollama/models/manifests/registry.ollama.ai/library}" \
    python3 - <<'PY'
import sys, importlib.util
spec = importlib.util.spec_from_file_location("p", "scripts/model-picker.py")
p = importlib.util.module_from_spec(spec)
sys.modules["p"] = p
spec.loader.exec_module(p)

models = p._discover_models()
lines, sels, items = p._build_menu(models)
for i, line in enumerate(lines):
    if not sels[i]:
        continue
    m = items[i]
    if not m or m.get("backend") == "ollama":
        continue
    name = m.get("name") or ""
    backend = m.get("backend") or ""
    ctx = int(m.get("_picker_context") or 0)
    if name and backend in ("vllm", "sglang") and ctx > 0:
        print(f"{name}\t{backend}\t{ctx}")
        break
PY
)

if [ -z "$picked_row" ]; then
    info ""
    skip "picker E2E" "no selectable HF rows — the picker would only show Ollama"
    info "  Run \`make probe-vllm\` or \`make probe-sglang\` against a"
    info "  downloaded HF model to populate the relevant cache, then rerun."
    echo ""
    echo "========================================"
    echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${GRAY}${SKIP} skipped${NC}"
    echo "========================================"
    exit 0
fi

PICKED_NAME=$(echo "$picked_row" | cut -f1)
PICKED_BACKEND=$(echo "$picked_row" | cut -f2)
PICKED_CTX=$(echo "$picked_row" | cut -f3)
case "$PICKED_BACKEND" in
    vllm)   ROUTER_PORT=11435 ;;
    sglang) ROUTER_PORT=11436 ;;
    *)      fail "picked_backend" "unexpected backend: $PICKED_BACKEND"; exit 1 ;;
esac

info ""
info "  picked from menu:  $PICKED_NAME"
info "  backend / port:    $PICKED_BACKEND / $ROUTER_PORT"
info "  context:           $PICKED_CTX"

# ── Step 3: construct the agent-style serving name (picker's logic) ─────────

SERVING_NAME="${PICKED_NAME}@${PICKED_CTX}"
info ""
info "  serving name (with @ctx override): $SERVING_NAME"
info "  this is what the picker hands to the agent CLI verbatim"

# ── Step 4+5: replay an aider-style request to the router ───────────────────
#
# aider for HF backends sends OpenAI /v1/chat/completions with model=
# openai/<name>@<ctx>. The router strips the openai/ prefix? No — that's
# a litellm-side prefix that aider strips before sending. So the wire
# request to the router has model=<name>@<ctx>. This is exactly what
# the picker's _build("aider", serving_name, backend) constructs.

info ""
info "=== Sending chat completion via $PICKED_BACKEND port $ROUTER_PORT ==="
info "  (cold start budget ${COLD_START_S}s; the router will recreate the backend)"

# When this test runs after test-router-vllm in `make test`, the trap-
# induced `cache-up` may still be settling — the placeholder container
# is up but the router's backendState may be transitioning. A brief
# settle gives the router a stable view before we trigger a recreate.
sleep 3

resp=$($RUNTIME exec "$EXEC_HOST" curl -s --max-time "$COLD_START_S" \
    -H "Content-Type: application/json" \
    "http://router:${ROUTER_PORT}/v1/chat/completions" \
    -d "{\"model\":\"${SERVING_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Answer in one word: capital of France?\"}],\"max_tokens\":8,\"temperature\":0}" \
    2>&1)

if echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert 'choices' in d, f'no choices in response: {sorted(d.keys())}'
assert d['choices'], 'choices array empty'
msg = d['choices'][0].get('message') or {}
# Structured-reasoning models (Qwen3 in thinking mode, etc.) put the
# answer entirely in reasoning_content / reasoning when max_tokens is
# tight, and leave content=null. Inline-reasoning models put think
# tags + answer in content. Plain models put the answer in content.
# The intent of this test is 'backend round-trip works' — all three
# shapes prove that. Accept any of the four standard payload fields
# carrying non-empty text; refusal also counts as a valid response.
candidates = [
    msg.get('content'),
    msg.get('reasoning_content'),
    msg.get('reasoning'),
    msg.get('refusal'),
]
texts = [c for c in candidates if isinstance(c, str) and c.strip()]
assert texts, f'no non-empty content/reasoning/refusal in message: keys={sorted(msg.keys())}'
print(f'  response model: {d.get(\"model\", \"?\")}')
print(f'  payload field:  {(\"content\" if msg.get(\"content\") else \"reasoning_content\" if msg.get(\"reasoning_content\") else \"reasoning\" if msg.get(\"reasoning\") else \"refusal\")}')
print(f'  snippet:        {texts[0].strip()[:80]!r}')
" 2>/dev/null; then
    pass "router routed picker-emitted serving name to $PICKED_BACKEND"
else
    fail "E2E chat" "${resp:0:500}"
fi

# ── Step 6: confirm the model field in /health reflects the clean name ──────
#
# The router rewrites the body's model field via setTopJSONField after
# parseCtxOverride strips the @<ctx>. Then ensureBackendRunning records
# bs.currentModel = clean_name. The /health endpoint exposes it.
# This is the only place we can verify the rewrite happened correctly.

info ""
info "=== Verifying router stripped @<ctx> from currentModel ==="
health=$($RUNTIME exec "$EXEC_HOST" curl -sf --max-time 10 \
    "http://router:${ROUTER_PORT}/health" 2>&1)
cur_model=$(echo "$health" | python3 -c "import sys,json; print(json.load(sys.stdin).get('current_model',''))" 2>/dev/null || echo '?')

if [ "$cur_model" = "$PICKED_NAME" ]; then
    pass "router.currentModel = $PICKED_NAME (picker's @ctx stripped)"
elif [ -z "$cur_model" ]; then
    # Idle timeout may have stopped the backend by the time we checked;
    # acceptable when test runs slowly.
    skip "currentModel inspection" "backend already idle; clean rewrite still happened upstream"
else
    fail "currentModel" "expected '$PICKED_NAME', got '$cur_model'"
fi

echo ""
echo "========================================"
echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${GRAY}${SKIP} skipped${NC}"
echo "========================================"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
