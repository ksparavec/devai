#!/usr/bin/env bash
# Smoke test for the bench harness.
#
# Picks one tool-callable vLLM model the probe cache says fits at the
# host's VRAM band, runs every task type at a tiny subset size (n=5
# correctness, n=10 latency), then asserts the bench cache gained a
# row with the expected fields populated.
#
# Prerequisites:
#   - `make cache-up` is up (router + ollama; vllm/sglang as placeholders)
#   - At least one fitting HF model in deploy/.vllm-reasoning-cache.json
#   - Lab image rebuilt after appending inspect-ai to requirements-base.txt
#
# Usage:  ./tests/test-bench-smoke.sh
# Knobs:  BENCH_REPO=<regex>  (default: Qwen3-8B-NVFP4 — known mode=auto)
#
# Wall time: ~3-6 min including a vLLM cold start.
# Failure exit: 1; success: 0.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BENCH_REPO="${BENCH_REPO:-Qwen3-8B-NVFP4}"
CACHE="$REPO_ROOT/deploy/.bench-cache.json"

echo ">>> bench smoke: BENCH_REPO=$BENCH_REPO"

# Snapshot pre-state so we can detect a fresh write.
before_keys=$(python3 -c "
import json
try:
    d = json.load(open('$CACHE'))
    print(' '.join(sorted(d.keys())))
except FileNotFoundError:
    print('')
")

# Run the smoke target. Make-driven so the image + GPU flags match
# what an operator would invoke.
if ! BENCH_REPO="$BENCH_REPO" make -s test-bench-smoke; then
    echo "FAIL: make test-bench-smoke exited non-zero" >&2
    exit 1
fi

# Confirm the cache was written and the row carries every expected
# field group (tasks, metrics with VRAM and TTFT).
python3 <<PY || exit 1
import json, sys
try:
    d = json.load(open("$CACHE"))
except FileNotFoundError:
    print(f"FAIL: $CACHE not found", file=sys.stderr); sys.exit(1)

# Find the row that matches BENCH_REPO. The probe cache key is
# repo@sha; we accept any key whose 'model' alias matches the regex.
import re
rx = re.compile(r"$BENCH_REPO")
matches = [k for k, v in d.items() if isinstance(v, dict) and rx.search(v.get("model", "") + " " + k)]
if not matches:
    print(f"FAIL: no row matched {rx.pattern!r}", file=sys.stderr); sys.exit(1)
row = d[matches[0]]
tasks = row.get("tasks") or {}
metrics = row.get("metrics") or {}

probs = []
if not any(k.startswith("gsm8k_") for k in tasks):     probs.append("missing gsm8k task")
if not any(k.startswith("humaneval_") for k in tasks): probs.append("missing humaneval task")
if not any(k.startswith("tools_use") for k in tasks):  probs.append("missing tools_use task")
if "leak_probe" not in tasks:                          probs.append("missing leak_probe task")
if metrics.get("peak_vram_gb", 0) <= 0:                probs.append("peak_vram_gb not populated")
if metrics.get("ttft_ms_first") is None:               probs.append("ttft_ms_first not populated")
# Cold start should be > steady p50 — vLLM warm-up dominates first req.
first = metrics.get("ttft_ms_first") or 0
p50   = metrics.get("ttft_ms_steady_p50") or 0
if p50 > 0 and first <= p50:
    probs.append(f"ttft_ms_first ({first}) <= steady_p50 ({p50}); cold-start signal lost")

if probs:
    for p in probs: print(f"FAIL: {p}", file=sys.stderr)
    sys.exit(1)

print(f"OK: row {matches[0]!r} populated.")
print(f"  tasks: {sorted(tasks.keys())}")
print(f"  ttft_ms_first={metrics.get('ttft_ms_first')}  steady_p50={metrics.get('ttft_ms_steady_p50')}  peak_vram_gb={metrics.get('peak_vram_gb')}")
PY

echo ">>> bench smoke OK"
exit 0
