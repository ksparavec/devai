#!/usr/bin/env bash
# DevAI agent × backend smoke-test matrix.
#
# Runs inside the lab container (where all agents are installed). For each
# (agent, backend) pair we have a downloaded model for, fires a one-shot
# "say hi" prompt at the router and classifies the outcome.
#
# Outcomes:
#   PASS — agent exited 0 within timeout AND produced a non-empty reply
#   FAIL — non-zero exit, timeout, or empty reply (reason captured to log)
#   SKIP — no model on disk for this backend, or agent has no headless mode
#
# Exit code: 0 only if every populated cell is PASS. 1 otherwise.
#
# Usage (from host):
#   make test-agents
# Or directly inside container:
#   /var/cache/devai/tests/agent-matrix.sh
#
# Env overrides:
#   CELL_TIMEOUT=30        per-cell timeout in seconds
#   PROMPT="say hi"        prompt sent to each agent
#   ROUTER=devai-router    hostname of the router

set -uo pipefail

PROMPT="${PROMPT:-say hi in five words}"
ROUTER="${ROUTER:-devai-router}"
LOG_DIR="${LOG_DIR:-/tmp/agent-matrix-logs}"

# Per-backend timeouts. vLLM/SGLang cold starts (loading the model into VRAM)
# regularly take 30-90s on first request after the router stops them, so a
# global 30s cap would mistake startup for a hang.
TIMEOUT_OLLAMA="${TIMEOUT_OLLAMA:-30}"
TIMEOUT_VLLM="${TIMEOUT_VLLM:-180}"
TIMEOUT_SGLANG="${TIMEOUT_SGLANG:-180}"

cell_timeout() {
    case "$1" in
        ollama) echo "$TIMEOUT_OLLAMA" ;;
        vllm)   echo "$TIMEOUT_VLLM" ;;
        sglang) echo "$TIMEOUT_SGLANG" ;;
    esac
}

OLLAMA_MANIFESTS="/var/cache/devai/ollama/models/manifests/registry.ollama.ai/library"
VLLM_DIR="/var/cache/devai/ollama/models/vllm"

mkdir -p "$LOG_DIR"

# ── Colour helpers ──────────────────────────────────────────────────────────
if [ -t 1 ]; then
    GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'
    DIM=$'\033[2m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
else
    GREEN=""; RED=""; YELLOW=""; DIM=""; BOLD=""; RESET=""
fi

# ── Discovery: smallest downloaded model per backend ────────────────────────
pick_smallest_ollama() {
    [ -d "$OLLAMA_MANIFESTS" ] || return 1
    # Walk lib/tag manifests; sum layer sizes via python (already in image).
    python3 - <<'PY' 2>/dev/null
import json, os, sys
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

pick_smallest_hf() {
    [ -d "$VLLM_DIR" ] || return 1
    python3 - <<'PY' 2>/dev/null
import os
base = "/var/cache/devai/ollama/models/vllm"
best = None
best_size = None
for name in sorted(os.listdir(base)):
    d = os.path.join(base, name)
    if not (os.path.isdir(d) and os.path.isfile(os.path.join(d, "config.json"))):
        continue
    total = 0
    for root, _, files in os.walk(d):
        for f in files:
            try: total += os.path.getsize(os.path.join(root, f))
            except OSError: pass
    if best is None or total < best_size:
        best = name
        best_size = total
if best:
    print(best)
PY
}

# ── Per-agent invocations ───────────────────────────────────────────────────
# Each function: $1=backend  $2=model_name  $3=log_file
# Returns 0 on PASS, 1 on FAIL. SKIP is signalled by exit code 99.

run_claude() {
    local backend="$1" model="$2" log="$3"
    local port; port=$(backend_port "$backend") || return 99
    local args=(claude -p "$PROMPT" --model "$model")
    if [ "$backend" != "ollama" ]; then
        ANTHROPIC_BASE_URL="http://$ROUTER:$port" \
            timeout "$(cell_timeout "$backend")" "${args[@]}" >"$log" 2>&1
    else
        # Claude doesn't speak ollama-style; route via the OpenAI-compat shim.
        ANTHROPIC_BASE_URL="http://$ROUTER:$port" \
            timeout "$(cell_timeout "$backend")" "${args[@]}" >"$log" 2>&1
    fi
}

run_aider() {
    local backend="$1" model="$2" log="$3"
    local port; port=$(backend_port "$backend") || return 99
    if [ "$backend" = "ollama" ]; then
        OLLAMA_API_BASE="http://$ROUTER:$port" \
            timeout "$(cell_timeout "$backend")" aider \
                --model "ollama_chat/$model" --no-stream --no-git \
                --yes-always --no-auto-commits --no-show-model-warnings \
                --message "$PROMPT" >"$log" 2>&1
    else
        timeout "$(cell_timeout "$backend")" aider \
            --model "openai/$model" \
            --openai-api-base "http://$ROUTER:$port/v1" \
            --openai-api-key local \
            --no-stream --no-git --yes-always --no-auto-commits \
            --no-show-model-warnings \
            --message "$PROMPT" >"$log" 2>&1
    fi
}

run_codex() {
    local backend="$1" model="$2" log="$3"
    OPENAI_API_KEY=local timeout "$(cell_timeout "$backend")" codex exec \
        --oss --local-provider "router-$backend" \
        --skip-git-repo-check \
        -c "model=\"$model\"" \
        "$PROMPT" >"$log" 2>&1
}

run_interpreter() {
    local backend="$1" model="$2" log="$3"
    local port; port=$(backend_port "$backend") || return 99
    if [ "$backend" = "ollama" ]; then
        echo "$PROMPT" | timeout "$(cell_timeout "$backend")" interpreter \
            --model "ollama/$model" -y --offline --stdin >"$log" 2>&1
    else
        echo "$PROMPT" | timeout "$(cell_timeout "$backend")" interpreter \
            --model "openai/$model" \
            --api_base "http://$ROUTER:$port/v1" \
            --api_key local -y --offline --stdin >"$log" 2>&1
    fi
}

run_late() {
    # LATE is a TUI with no headless mode in this build.
    return 99
}

backend_port() {
    case "$1" in
        ollama) echo 11434 ;;
        vllm)   echo 11435 ;;
        sglang) echo 11436 ;;
        *) return 1 ;;
    esac
}

# ── Cell evaluation ─────────────────────────────────────────────────────────
# Pass if exit 0 AND log has at least one non-blank line beyond known prologue.
# Fail with reason otherwise. Skip via exit 99.
evaluate_cell() {
    local agent="$1" backend="$2" model="$3"
    local log="$LOG_DIR/${agent}__${backend}.log"
    : >"$log"

    local fn="run_$agent"
    if ! declare -f "$fn" >/dev/null; then
        echo "SKIP|no runner"
        return
    fi

    local start=$SECONDS
    backend="$backend" "$fn" "$backend" "$model" "$log"
    local rc=$?
    local elapsed=$((SECONDS - start))

    if [ "$rc" = "99" ]; then
        echo "SKIP|no headless mode"
        return
    fi
    if [ "$rc" = "124" ]; then
        echo "FAIL|timeout $(cell_timeout "$backend")s"
        return
    fi
    if [ "$rc" -ne 0 ]; then
        local first_err
        first_err=$(grep -m1 -iE "error|exception|traceback|failed" "$log" \
                    | head -c 80)
        echo "FAIL|exit=$rc ${first_err:-no_error_in_log}"
        return
    fi
    # Check for any non-blank line in the log (model produced output).
    if ! grep -q '[[:graph:]]' "$log"; then
        echo "FAIL|empty reply"
        return
    fi
    echo "PASS|${elapsed}s"
}

# ── Main ────────────────────────────────────────────────────────────────────
echo
echo "${BOLD}DevAI agent × backend matrix${RESET}"
echo "  prompt:   \"$PROMPT\""
echo "  timeouts: ollama=${TIMEOUT_OLLAMA}s  vllm=${TIMEOUT_VLLM}s  sglang=${TIMEOUT_SGLANG}s"
echo "  router:   http://$ROUTER:1143{4,5,6}"
echo "  logs:     $LOG_DIR"
echo

# Pick one model per backend (smallest available). vllm and sglang share the
# HF cache directory so they get the same model.
OLLAMA_MODEL=$(pick_smallest_ollama || true)
HF_MODEL=$(pick_smallest_hf || true)

declare -A MODEL_FOR
[ -n "$OLLAMA_MODEL" ] && MODEL_FOR[ollama]="$OLLAMA_MODEL"
[ -n "$HF_MODEL" ]     && MODEL_FOR[vllm]="$HF_MODEL"
[ -n "$HF_MODEL" ]     && MODEL_FOR[sglang]="$HF_MODEL"

echo "  models picked:"
for b in ollama vllm sglang; do
    if [ -n "${MODEL_FOR[$b]:-}" ]; then
        echo "    $b → ${MODEL_FOR[$b]}"
    else
        echo "    $b → ${DIM}(no downloaded model){$RESET}"
    fi
done
echo

AGENTS=(claude aider codex interpreter late)
BACKENDS=(ollama vllm sglang)

# Header row
printf "  %-14s" ""
for b in "${BACKENDS[@]}"; do
    printf " %-22s" "$b"
done
echo

declare -i pass_count=0 fail_count=0 skip_count=0
declare -A RESULT

for agent in "${AGENTS[@]}"; do
    printf "  %-14s" "$agent"
    for backend in "${BACKENDS[@]}"; do
        model="${MODEL_FOR[$backend]:-}"
        if [ -z "$model" ]; then
            cell="SKIP|no model"
        else
            cell=$(evaluate_cell "$agent" "$backend" "$model")
        fi
        status="${cell%%|*}"
        detail="${cell#*|}"
        RESULT["$agent/$backend"]="$cell"
        case "$status" in
            PASS) printf " ${GREEN}%-22s${RESET}" "PASS  $detail"; pass_count=$((pass_count+1)) ;;
            FAIL) printf " ${RED}%-22s${RESET}"   "FAIL  $detail"; fail_count=$((fail_count+1)) ;;
            SKIP) printf " ${DIM}%-22s${RESET}"   "skip  $detail"; skip_count=$((skip_count+1)) ;;
            *)    printf " %-22s" "?     $detail" ;;
        esac
    done
    echo
done

echo
echo "  ${BOLD}summary:${RESET} ${GREEN}$pass_count pass${RESET} · ${RED}$fail_count fail${RESET} · ${DIM}$skip_count skip${RESET}"
echo

if [ "$fail_count" -gt 0 ]; then
    echo "  ${YELLOW}failed cells — first error from each log:${RESET}"
    for key in "${!RESULT[@]}"; do
        cell="${RESULT[$key]}"
        if [[ "$cell" == FAIL* ]]; then
            agent="${key%/*}"
            backend="${key#*/}"
            log="$LOG_DIR/${agent}__${backend}.log"
            echo "  ${RED}✗${RESET} $key  ($log)"
            grep -m1 -iE "error|exception|traceback|failed" "$log" \
                 2>/dev/null | sed 's/^/      /' | head -c 200
            echo
        fi
    done | sort
    exit 1
fi
exit 0
