#!/usr/bin/env bash
# Phase 1 byte-identical-cache verification.
#
# After the probe-core refactor (helpers moved to scripts/_probe_core.py),
# re-running `make probe` against an already-populated Ollama cache must
# produce zero content drift. The prober skips cached (vram, ctx) cells
# unless --force, so this is a fast check (~30s) that catches any
# regression in:
#
#   * cache I/O (load/save round-trip with sort_keys)
#   * the implied-spill builder closure (shape preservation)
#   * the canonical-capability update logic
#   * alias reconciliation
#
# Test: snapshot cache → run `make probe` → diff snapshot vs current.
# Diff must be empty modulo `first_probed_at` / `last_probed_at`
# timestamps on entries that had a fresh arch-metadata lookup
# (when an entry's max_context wasn't already populated).
#
# Skips when no Ollama probe cache exists (`make probe` hasn't been
# run yet); fails on any unexpected diff.
#
# Wall time: ~10-30s. No GPU launches when cache is fully populated.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_FILE="$REPO_ROOT/deploy/.ollama-reasoning-cache.json"

GREEN='\033[32m'
RED='\033[31m'
YELLOW='\033[33m'
GRAY='\033[2m'
NC='\033[0m'

PASS=0
FAIL=0
SKIP=0
pass() { ((PASS++)); echo -e "  ${GREEN}PASS${NC} $1"; }
fail() { ((FAIL++)); echo -e "  ${RED}FAIL${NC} $1: $2"; }
skip() { ((SKIP++)); echo -e "  ${GRAY}SKIP${NC} $1: $2"; }
info() { echo -e "${YELLOW}$1${NC}"; }

info "=== Ollama probe idempotency check ==="

if [ ! -f "$CACHE_FILE" ]; then
    skip "all" "no $CACHE_FILE — run \`make probe\` first"
    echo ""
    echo "========================================"
    echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${GRAY}${SKIP} skipped${NC}"
    echo "========================================"
    exit 0
fi

snapshot=$(mktemp)
cp "$CACHE_FILE" "$snapshot"
trap 'rm -f "$snapshot"' EXIT

info "  cache snapshot:    $snapshot"
info "  entries:           $(python3 -c "import json; print(len(json.load(open('$CACHE_FILE'))))")"
info ""
info "Running \`make probe\` (incremental — should skip every cached cell)..."

cd "$REPO_ROOT"
if ! make probe >/tmp/probe-idempotent.log 2>&1; then
    fail "make probe" "non-zero exit (see /tmp/probe-idempotent.log)"
    echo ""
    echo "========================================"
    echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${GRAY}${SKIP} skipped${NC}"
    echo "========================================"
    exit 1
fi

info ""
info "Diffing snapshot vs current cache (timestamps allowed to drift)..."

# Per-entry comparison ignoring `first_probed_at` / `last_probed_at` —
# these update whenever the prober touches an entry, even if no
# probe cell changed. Anything else is a real drift.
python3 - "$snapshot" "$CACHE_FILE" <<'PY'
import json, sys

with open(sys.argv[1]) as fh: before = json.load(fh)
with open(sys.argv[2]) as fh: after = json.load(fh)

TS_KEYS = {"first_probed_at", "last_probed_at", "probed_at"}

def strip_timestamps(obj):
    if isinstance(obj, dict):
        return {k: strip_timestamps(v) for k, v in obj.items() if k not in TS_KEYS}
    if isinstance(obj, list):
        return [strip_timestamps(v) for v in obj]
    return obj

a = strip_timestamps(before)
b = strip_timestamps(after)

if a == b:
    print(f"  unchanged: {len(after)} entries (timestamps allowed to drift)")
    sys.exit(0)

# Drift detected — find which entries differ.
keys_before = set(a.keys())
keys_after = set(b.keys())
added = keys_after - keys_before
removed = keys_before - keys_after
changed = [k for k in (keys_before & keys_after) if a[k] != b[k]]

print(f"  DRIFT DETECTED:", file=sys.stderr)
if added:
    print(f"    added entries: {sorted(added)}", file=sys.stderr)
if removed:
    print(f"    removed entries: {sorted(removed)}", file=sys.stderr)
if changed:
    print(f"    changed entries: {sorted(changed)[:5]}", file=sys.stderr)
    for k in sorted(changed)[:1]:
        before_keys = set(a[k].keys()) if isinstance(a[k], dict) else set()
        after_keys = set(b[k].keys()) if isinstance(b[k], dict) else set()
        ks_added = after_keys - before_keys
        ks_removed = before_keys - after_keys
        ks_modified = [
            kk for kk in (before_keys & after_keys)
            if a[k][kk] != b[k][kk]
        ]
        print(f"      {k}: +{sorted(ks_added)} -{sorted(ks_removed)} ~{sorted(ks_modified)}",
              file=sys.stderr)
sys.exit(1)
PY
diff_rc=$?

if [ "$diff_rc" -eq 0 ]; then
    pass "cache content unchanged (idempotent)"
else
    fail "idempotency" "cache drifted on second run; see stderr above"
fi

echo ""
echo "========================================"
echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${GRAY}${SKIP} skipped${NC}"
echo "========================================"

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
