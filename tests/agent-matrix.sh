#!/usr/bin/env bash
# DevAI agent smoke-test matrix — ollama only.
#
# For each agent (claude, aider, codex), fires a one-shot "say hi" prompt
# at the router's ollama port and classifies the outcome.
#
# Outcomes:
#   PASS — agent exited 0 within timeout AND produced a non-empty reply
#   FAIL — non-zero exit, timeout, or empty reply (reason captured to log)
#   SKIP — agent has no headless mode
#
# Exit code: 0 only if every non-skip cell is PASS. 1 otherwise.
#
# Usage (from host):  make test-agents
#
# Env overrides:
#   CELL_TIMEOUT=30        per-cell timeout in seconds
#   PROMPT="say hi"        prompt sent to each agent
#   ROUTER=devai-router    hostname of the router

set -uo pipefail

# 60s gives the cold-start cell (first agent to pull the model into VRAM)
# enough headroom; warm cells finish in 5-15s.
CELL_TIMEOUT="${CELL_TIMEOUT:-60}"
# Instruction-style prompt that small reasoning models handle in <5s.
# We assert the reply contains EXPECT_TOKEN — that's a stricter check than
# "non-empty output" (which flagged false-passes when agents echoed
# instructions or printed slash-command errors).
PROMPT="${PROMPT:-reply with the single word PONG}"
EXPECT_TOKEN="${EXPECT_TOKEN:-PONG}"
ROUTER="${ROUTER:-devai-router}"
PORT=11434
LOG_DIR="${LOG_DIR:-/tmp/agent-matrix-logs}"
OLLAMA_MANIFESTS="/var/cache/devai/ollama/models/manifests/registry.ollama.ai/library"

mkdir -p "$LOG_DIR"

if [ -t 1 ]; then
    GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'
    DIM=$'\033[2m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
else
    GREEN=""; RED=""; YELLOW=""; DIM=""; BOLD=""; RESET=""
fi

# ── Pick smallest downloaded ollama model with structured reasoning ─────────
# Earlier versions walked the manifests blindly and could pick a base/text
# model (e.g. llama3.2:3b-text-fp16) that has no chat template, producing
# spurious FAILs. We now require the probe-derived `reasoning.capability ==
# "structured"` from active-models.yaml, falling back to the unfiltered
# manifest scan only if the active catalog isn't present.
pick_smallest_ollama() {
    if [ -f /etc/devai/active-models.yaml ]; then
        python3 - <<'PY' 2>/dev/null
import json, os, sys
try:
    import yaml
except ImportError:
    sys.exit(0)
data = yaml.safe_load(open("/etc/devai/active-models.yaml")) or {}
manifests = "/var/cache/devai/ollama/models/manifests/registry.ollama.ai/library"
best = None
best_size = None
for m in data.get("models") or []:
    if "ollama" not in (m.get("backend") or []): continue
    cap = ((m.get("reasoning") or {}).get("capability") or "")
    if cap != "structured": continue
    name = m.get("name", "")
    if ":" not in name: continue
    lib, tag = name.split(":", 1)
    f = os.path.join(manifests, lib, tag)
    if not os.path.isfile(f): continue
    try:
        layers = json.loads(open(f).read()).get("layers", [])
        size = sum(int(L.get("size", 0)) for L in layers)
    except Exception:
        continue
    if best is None or size < best_size:
        best = name
        best_size = size
if best:
    print(best)
PY
        return 0
    fi
    [ -d "$OLLAMA_MANIFESTS" ] || return 1
    python3 - <<'PY' 2>/dev/null
import json, os
base = "/var/cache/devai/ollama/models/manifests/registry.ollama.ai/library"
best = None
best_size = None
for lib in sorted(os.listdir(base)):
    libp = os.path.join(base, lib)
    if not os.path.isdir(libp): continue
    for tag in sorted(os.listdir(libp)):
        f = os.path.join(libp, tag)
        if not os.path.isfile(f): continue
        try:
            data = json.loads(open(f).read())
            size = sum(int(L.get("size", 0)) for L in data.get("layers", []))
        except Exception:
            continue
        if best is None or size < best_size:
            best = f"{lib}:{tag}"
            best_size = size
if best:
    print(best)
PY
}

# ── Per-agent invocations ───────────────────────────────────────────────────
# Each function: $1=model  $2=log_file
# Returns 0 on success exit; 99 = skip; anything else = fail.

run_claude() {
    local model="$1" log="$2"
    ANTHROPIC_BASE_URL="http://$ROUTER:$PORT" \
    ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-local}" \
        timeout "$CELL_TIMEOUT" \
        claude -p "$PROMPT" --model "$model" >"$log" 2>&1
}

run_aider() {
    local model="$1" log="$2"
    OLLAMA_API_BASE="http://$ROUTER:$PORT" \
        timeout "$CELL_TIMEOUT" aider \
            --model "ollama_chat/$model" --no-stream --no-git \
            --yes-always --no-auto-commits --no-show-model-warnings \
            --message "$PROMPT" >"$log" 2>&1
}

run_codex() {
    local model="$1" log="$2"
    timeout "$CELL_TIMEOUT" codex exec \
        --oss --local-provider router-ollama \
        --skip-git-repo-check \
        -c "model=\"$model\"" \
        "$PROMPT" >"$log" 2>&1
}


# ── Cell evaluator ──────────────────────────────────────────────────────────
evaluate_cell() {
    local agent="$1" model="$2"
    local log="$LOG_DIR/${agent}.log"
    : >"$log"

    local fn="run_$agent"
    declare -f "$fn" >/dev/null || { echo "SKIP|no runner"; return; }

    local start=$SECONDS
    "$fn" "$model" "$log"
    local rc=$?
    local elapsed=$((SECONDS - start))

    if [ "$rc" = "99" ]; then echo "SKIP|interactive only"; return; fi
    if [ "$rc" = "124" ]; then echo "FAIL|timeout ${CELL_TIMEOUT}s"; return; fi
    if [ "$rc" -ne 0 ]; then
        local first_err
        first_err=$(grep -m1 -iE "error|exception|traceback|failed" "$log" | head -c 80)
        echo "FAIL|exit=$rc ${first_err:-no_error_in_log}"
        return
    fi
    if ! grep -q '[[:graph:]]' "$log"; then
        echo "FAIL|empty reply"
        return
    fi
    if ! grep -qi "$EXPECT_TOKEN" "$log"; then
        echo "FAIL|reply lacked '$EXPECT_TOKEN'"
        return
    fi
    echo "PASS|${elapsed}s"
}

# ── Main ────────────────────────────────────────────────────────────────────
echo
echo "${BOLD}DevAI agent smoke matrix — ollama only${RESET}"
echo "  prompt:  \"$PROMPT\""
echo "  timeout: ${CELL_TIMEOUT}s/cell"
echo "  router:  http://$ROUTER:$PORT"
echo "  logs:    $LOG_DIR"
echo

MODEL=$(pick_smallest_ollama || true)
if [ -z "$MODEL" ]; then
    echo "${RED}error: no structured-capability ollama model found.${RESET}" >&2
    echo "       Looked in /etc/devai/active-models.yaml (preferred) and" >&2
    echo "       $OLLAMA_MANIFESTS (fallback)." >&2
    echo "       Pull at least one structured model: make model-select DOWNLOAD=1" >&2
    exit 1
fi
echo "  model:   $MODEL"
echo

declare -i pass=0 fail=0 skip=0
declare -A RESULT

for agent in claude aider codex; do
    cell=$(evaluate_cell "$agent" "$MODEL")
    RESULT["$agent"]="$cell"
    status="${cell%%|*}"
    detail="${cell#*|}"
    case "$status" in
        PASS) printf "  %-14s ${GREEN}PASS${RESET}  %s\n" "$agent" "$detail"; pass=$((pass+1)) ;;
        FAIL) printf "  %-14s ${RED}FAIL${RESET}  %s\n" "$agent" "$detail"; fail=$((fail+1)) ;;
        SKIP) printf "  %-14s ${DIM}skip  %s${RESET}\n" "$agent" "$detail"; skip=$((skip+1)) ;;
    esac
done

echo
echo "  ${BOLD}summary:${RESET} ${GREEN}$pass pass${RESET} · ${RED}$fail fail${RESET} · ${DIM}$skip skip${RESET}"
echo

if [ "$fail" -gt 0 ]; then
    echo "  ${YELLOW}failed cells — first error from each log:${RESET}"
    for key in "${!RESULT[@]}"; do
        cell="${RESULT[$key]}"
        if [[ "$cell" == FAIL* ]]; then
            log="$LOG_DIR/${key}.log"
            echo "  ${RED}✗${RESET} $key  ($log)"
            grep -m1 -iE "error|exception|traceback|failed" "$log" 2>/dev/null \
                | sed 's/^/      /' | head -c 200
            echo
        fi
    done | sort
    exit 1
fi
exit 0
