#!/usr/bin/env bash
# Smoke test for the vLLM probe runner.
#
# Probes one downloaded HF model at one (vram, ctx) cell, then asserts
# the cache file is written with the expected schema-v1 shape. Does NOT
# assert `fits: true` — the goal here is "the prober + cache I/O pipe
# round-trips correctly", which is true even for known-failure models
# (e.g. SGLang+FP4 on this image, which records fits:false with
# evidence.kind=infra).
#
# Prerequisites:
#   - At least one HF model downloaded under VLLM_MODELS_DIR
#   - GPU available
#   - `make cache-down` first — the prober self-checks for running
#     router/vllm/sglang containers and aborts otherwise
#
# Usage:  ./tests/test-probe-vllm.sh
# Knobs:  PROBE_MODEL=<repo regex>  (default: Llama-3.1-8B)
#         PROBE_VRAM=<G>            (default: 24G)
#         PROBE_CTX=<K>             (default: 32K)
#
# Wall time: ~60-120s for a single cold vLLM start + probe + teardown.
# Failure exit: 1; success: 0.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROBE_MODEL="${PROBE_MODEL:-Llama-3.1-8B}"
PROBE_VRAM="${PROBE_VRAM:-24G}"
PROBE_CTX="${PROBE_CTX:-32K}"
CACHE_FILE="$REPO_ROOT/deploy/.vllm-reasoning-cache.json"

GREEN='\033[32m'
RED='\033[31m'
YELLOW='\033[33m'
NC='\033[0m'

PASS=0
FAIL=0
pass() { ((PASS++)); echo -e "  ${GREEN}PASS${NC} $1"; }
fail() { ((FAIL++)); echo -e "  ${RED}FAIL${NC} $1: $2"; }
info() { echo -e "${YELLOW}$1${NC}"; }

info "=== vLLM probe smoke test ==="
info "  model regex:  $PROBE_MODEL"
info "  vram:         $PROBE_VRAM"
info "  ctx:          $PROBE_CTX"
info "  cache file:   $CACHE_FILE"

# Snapshot existing cache so the test can run idempotently — we restore
# it on exit so a CI run doesn't leak state into the operator's cache.
backup=""
if [ -f "$CACHE_FILE" ]; then
    backup="$(mktemp)"
    cp "$CACHE_FILE" "$backup"
fi
cleanup() {
    if [ -n "$backup" ]; then
        mv "$backup" "$CACHE_FILE"
    fi
}
trap cleanup EXIT

# Wipe and re-probe so the assertions run on fresh data.
rm -f "$CACHE_FILE"

info ""
info "Running probe (cold vLLM start + chat + nvidia-smi snapshot)..."
if ! make probe-vllm \
        PROBE_REPO="$PROBE_MODEL" \
        PROBE_VRAMS_VLLM="$PROBE_VRAM" \
        PROBE_CONTEXTS="$PROBE_CTX" \
        >/dev/null 2>&1; then
    fail "make probe-vllm" "non-zero exit"
    echo "Re-run manually for full output:" >&2
    echo "  make probe-vllm PROBE_REPO=$PROBE_MODEL PROBE_VRAMS_VLLM=$PROBE_VRAM PROBE_CONTEXTS=$PROBE_CTX" >&2
fi

info ""
info "Asserting cache shape..."

if [ ! -f "$CACHE_FILE" ]; then
    fail "cache file" "not written: $CACHE_FILE"
    echo "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}"
    exit 1
fi
pass "cache file written"

# Convert the requested ctx tag (32K/64K/128K/...) to its int form so we
# can index the probes map by string-of-int.
ctx_int=$(python3 -c "
import sys
raw = sys.argv[1].strip().upper()
if raw.endswith('K'):
    print(int(raw[:-1]) * 1024)
else:
    print(int(raw))
" "$PROBE_CTX")
vram_int=$(python3 -c "
import sys
raw = sys.argv[1].strip().upper()
print(int(raw.rstrip('G')))
" "$PROBE_VRAM")

# Single-shot Python assertion block. Returns non-zero on any failure.
python3 - "$CACHE_FILE" "$vram_int" "$ctx_int" <<'PY'
import json, sys
cache_path = sys.argv[1]
vram = int(sys.argv[2])
ctx = int(sys.argv[3])
with open(cache_path) as fh:
    cache = json.load(fh)
assert isinstance(cache, dict) and cache, f"cache is empty or non-dict: {type(cache).__name__}"
# Pick the first MODEL entry — the smoke test runs against one model. Skip
# the reserved `_meta` drift-stamp block (Phase C), which sort_keys=True
# places ahead of any lowercase-owner repo key.
key = next(k for k in cache if not k.startswith("_"))
entry = cache[key]
assert "@" in key, f"top-level key not <repo>@<sha>: {key!r}"
sv = entry.get("schema_version")
assert sv in (1, 2), f"schema_version not in (1, 2): {sv!r}"
for field in ("repo", "sha", "aliases", "probes"):
    assert field in entry, f"top-level missing {field!r}: {sorted(entry.keys())}"
# v2 added top-level reasoning_parser / tool_parser / disable_verified
# (populated when the prober verified them, otherwise null/false). The
# fields must exist even when the values are null — that's how the
# router knows "the prober looked but didn't confirm".
if sv == 2:
    for v2_field in ("reasoning_parser", "tool_parser", "disable_verified"):
        assert v2_field in entry, f"v2 entry missing {v2_field!r}: {sorted(entry.keys())}"
assert isinstance(entry["aliases"], list) and entry["aliases"], "aliases empty"
assert isinstance(entry["probes"], dict), "probes not a dict"
band = entry["probes"].get(str(vram))
assert isinstance(band, dict), f"no probes at vram={vram}: have {sorted(entry['probes'].keys())}"
cell = band.get(str(ctx))
assert isinstance(cell, dict), f"no cell at ctx={ctx}: have {sorted(band.keys())}"
for field in ("ctx", "vram_gb", "fits", "probed_at"):
    assert field in cell, f"cell missing {field!r}: {sorted(cell.keys())}"
assert cell["ctx"] == ctx, f"cell.ctx mismatch: {cell['ctx']} vs {ctx}"
assert cell["vram_gb"] == vram, f"cell.vram_gb mismatch: {cell['vram_gb']} vs {vram}"
assert isinstance(cell["fits"], bool), f"cell.fits not bool: {type(cell['fits']).__name__}"
# Capability is set top-level by reflect_first_cell_to_top_level.
assert entry.get("capability"), "top-level capability empty"
print(f"  entry key:       {key}")
print(f"  capability:      {entry['capability']}")
print(f"  cell fits:       {cell['fits']}")
print(f"  cell evidence:   {(cell.get('evidence') or {}).get('kind', 'n/a')}")
PY
shape_rc=$?
if [ "$shape_rc" -eq 0 ]; then
    pass "cache schema valid (top-level + per-cell shape)"
else
    fail "cache schema" "schema assertion failed (rc=$shape_rc)"
fi

echo ""
echo "========================================"
echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}"
echo "========================================"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
