#!/usr/bin/env bash
# Matrix test — (model × wire protocol × scenario) against the running router.
#
# Complements tests/agent-matrix.sh (which exercises full agent CLIs against
# one model). This one drives the router's three wire protocols directly
# via curl, against a representative slice of the active probe cache, so
# every regression we've actually hit is covered:
#
#   basic   simple chat round-trip                            → HTTP 200 + content
#   ctx     options.num_ctx loads at exactly that context     → /api/ps cross-check
#   tools   tool definition accepted                          → HTTP 200 (no
#                                                                "does not support
#                                                                tools" 400)
#   think_auto  structured model + auto policy                → response has
#                                                                thinking field
#   think_off   structured + disable_verified + off policy    → response lacks
#                                                                thinking
#
# Wire protocols:
#   /api/chat                Aider via ollama_chat/, OI via ollama/
#   /v1/chat/completions     Codex, Aider via openai/
#   /v1/messages             Claude Code
#
# Run from the host:  make test-models
# All curl traffic goes through devai-open-webui (already on devai-net,
# already has curl) — no extra container, no host port mapping needed.
#
# Errors propagate; the script is fail-safe. Final exit is 0 only when
# every non-skip cell passes.

set -o pipefail
# Deliberately NOT `set -u` — the optional-arg defaults around curl_via and
# the cache-lookup helpers trip on bash 5.x's stricter handling of "${4:-}"
# inside `local`. The PASS/FAIL tracking is robust against missing args.

RUNTIME="${CONTAINER_RUNTIME:-podman}"
ROUTER="${ROUTER:-devai-router:11434}"
OLLAMA="${OLLAMA:-devai-ollama:11434}"
PROBE_CACHE="${PROBE_CACHE:-deploy/.ollama-reasoning-cache.json}"
HOST_VRAM="${HOST_VRAM:-24}"
TEST_CTX="${TEST_CTX:-32768}"     # per-session ctx the ctx scenario asserts
LOAD_TIMEOUT="${LOAD_TIMEOUT:-180}"

PASS=0; FAIL=0; SKIP=0
declare -a FAIL_LINES=()

if [ -t 1 ]; then
    G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'
    D=$'\033[2m'; B=$'\033[1m'; N=$'\033[0m'
else
    G=""; R=""; Y=""; D=""; B=""; N=""
fi

cell_pass() {
    PASS=$((PASS + 1))
    printf '  %sPASS%s  %-32s  %-22s  %s\n' "$G" "$N" "$1" "$2" "$3"
}
cell_fail() {
    FAIL=$((FAIL + 1))
    printf '  %sFAIL%s  %-32s  %-22s  %s\n' "$R" "$N" "$1" "$2" "$3"
    FAIL_LINES+=("$1 / $2 / $3")
}
cell_skip() {
    SKIP=$((SKIP + 1))
    printf '  %sSKIP%s  %-32s  %-22s  %s\n' "$Y" "$N" "$1" "$2" "$3"
}
hdr() { printf '\n%s%s%s\n' "$B" "$1" "$N"; }

curl_via() {
    # POST to $ROUTER (default) or another host:port via devai-open-webui.
    # $1=path  $2=body  $3=timeout(s)  $4=optional extra header (-H)
    local path="$1" body="$2" timeout="${3:-60}" hdr="${4:-}"
    local target="${5:-$ROUTER}"
    if [ -n "$hdr" ]; then
        $RUNTIME exec devai-open-webui curl -sS --max-time "$timeout" \
            -w '\n__HTTP__%{http_code}' \
            -H "Content-Type: application/json" \
            -H "$hdr" \
            "http://$target$path" -d "$body" 2>&1
    else
        $RUNTIME exec devai-open-webui curl -sS --max-time "$timeout" \
            -w '\n__HTTP__%{http_code}' \
            -H "Content-Type: application/json" \
            "http://$target$path" -d "$body" 2>&1
    fi
}

curl_get() {
    # GET helper for endpoints like /api/ps and /api/tags that don't accept
    # a body. Same target/timeout convention as curl_via.
    local path="$1" timeout="${2:-10}" target="${3:-$ROUTER}"
    $RUNTIME exec devai-open-webui curl -sS --max-time "$timeout" \
        -w '\n__HTTP__%{http_code}' \
        "http://$target$path" 2>&1
}

http_code() { sed -n 's/^__HTTP__\([0-9]*\)$/\1/p' <<< "$1"; }
http_body() { sed '/^__HTTP__[0-9]*$/d' <<< "$1"; }

# ── Pick representative test models ────────────────────────────────────────
# One per (family, capability) bucket from the cache; only digests with a
# fully_on_gpu probe at the host VRAM band. Capability comes from the
# top-level field. We sample at most 5 models so the suite finishes in
# minutes; override TEST_MODELS=name1,name2,... to pin specific tags.

if [ -n "${TEST_MODELS:-}" ]; then
    IFS=',' read -ra MODELS <<< "$TEST_MODELS"
else
    if ! command -v jq >/dev/null 2>&1; then
        echo "error: jq required for model selection (or set TEST_MODELS)" >&2
        exit 2
    fi
    if [ ! -f "$PROBE_CACHE" ]; then
        echo "error: $PROBE_CACHE not found — run 'make probe' first" >&2
        exit 2
    fi
    # Exhaustive: every probed digest where at least one (vram, ctx) cell
    # at the host VRAM band fits fully on GPU. We test ONE alias per
    # digest (the first sorted name); aliases share weights and behave
    # identically, so testing all aliases would just multiply runtime
    # without finding new bugs. Eligible digests with capability=error
    # at the canonical level are excluded — they have no fitting cell.
    mapfile -t MODELS < <(jq -r --argjson vram "$HOST_VRAM" '
        [ to_entries[]
          | .value
          | select(.schema_version == 3 and .capability != "error")
          | { alias: .aliases[0],
              has_fit: ((.probes[$vram | tostring] // {}) |
                        to_entries | any(.value.fully_on_gpu == true)) }
          | select(.has_fit)
        ] | sort_by(.alias) | .[] | .alias
    ' "$PROBE_CACHE")
fi

if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "error: no eligible models found in $PROBE_CACHE" >&2
    exit 2
fi

# Each model load takes ~5-30s cold; the matrix runs ~5 scenarios × 3
# protocols per model. A full sweep across every eligible digest is
# typically 30-60 min on a 24G card. Print a heads-up so the operator
# isn't surprised by the wall time.
hdr "matrix: ${#MODELS[@]} model(s) × 3 protocol(s) × up to 5 scenarios"
echo "  models ($HOST_VRAM GB band, exhaustive over fitting digests):"
for m in "${MODELS[@]}"; do echo "    - $m"; done
echo "  router:  $ROUTER"
echo "  ctx:     $TEST_CTX (per-session via options.num_ctx on /api/chat)"
echo "  expect:  ~30-60 min for a full sweep; pin TEST_MODELS=... to subset"

# Capability lookup helpers (read from cache).
cap_of() {
    jq -r --arg name "$1" '
        to_entries[].value
        | select((.aliases // []) | index($name))
        | .capability
    ' "$PROBE_CACHE" | head -1
}
disable_verified_of() {
    jq -r --arg name "$1" '
        to_entries[].value
        | select((.aliases // []) | index($name))
        | (.disable_verified // false)
    ' "$PROBE_CACHE" | head -1
}

# NOTE: this file used to carry an ensure_ctx_variant() helper that minted
# `<parent>-ctx<N>` Modelfile siblings via /api/create, with a comment saying
# "the picker creates these on launch". That stopped being true at 3a98ed0.
# CLAUDE.md is explicit: derived ctx tags are inert leftovers -- the prober
# skips them (_CTX_VARIANT_RE) and the picker filters them (_ctx_tag), so
# they carry no probe row, hence no per-tier KV-cache dtype. Serving one is
# actively broken for a mixed-KV model: with no tier to resolve, the router
# falls back to the declared max (221184) at f16 KV and the load OOMs.
# Measured on the reference host, qwen3.6:35b-a3b-mtp-q4_K_M:
#   parent, no num_ctx           -> loaded ctx=131072 (its q8_0 tier), HTTP 200
#   parent + options.num_ctx=32K -> loaded ctx=32768,                 HTTP 200
#   derived -ctx32768 tag        -> launch at 221184 f16 -> cudaMalloc OOM, 400
# Per-session context is therefore driven the documented way: the parent tag
# plus options.num_ctx on /api/chat, which the router's setNumCtx honours.

# ── Scenarios ───────────────────────────────────────────────────────────────

scenario_basic() {
    local model="$1" path="$2"
    local body
    case "$path" in
        /api/chat)
            body=$(printf '{"model":"%s","messages":[{"role":"user","content":"hi"}],"stream":false,"options":{"num_predict":2}}' "$model")
            ;;
        /v1/chat/completions)
            body=$(printf '{"model":"%s","messages":[{"role":"user","content":"hi"}],"max_tokens":2}' "$model")
            ;;
        /v1/messages)
            body=$(printf '{"model":"%s","messages":[{"role":"user","content":"hi"}],"max_tokens":2}' "$model")
            ;;
    esac
    local out code
    out=$(curl_via "$path" "$body" "$LOAD_TIMEOUT")
    code=$(http_code "$out")
    if [ "$code" != "200" ]; then
        cell_fail "$model" "basic $path" "HTTP $code"
        return
    fi
    cell_pass "$model" "basic $path" "HTTP 200"
}

scenario_tools() {
    local model="$1" path="$2"
    local cap; cap="$(cap_of "${model%-ctx*}")"
    if [ "$cap" = "unsupported" ] || [ "$cap" = "error" ] || [ -z "$cap" ]; then
        cell_skip "$model" "tools $path" "capability=$cap"
        return
    fi
    local body
    case "$path" in
        /api/chat|/v1/chat/completions)
            body=$(printf '{"model":"%s","messages":[{"role":"user","content":"hi"}],"max_tokens":2,"options":{"num_predict":2},"tools":[{"type":"function","function":{"name":"echo","description":"echo","parameters":{"type":"object","properties":{"x":{"type":"string"}}}}}]}' "$model")
            ;;
        /v1/messages)
            body=$(printf '{"model":"%s","messages":[{"role":"user","content":"hi"}],"max_tokens":2,"tools":[{"name":"echo","description":"echo","input_schema":{"type":"object","properties":{"x":{"type":"string"}}}}]}' "$model")
            ;;
    esac
    local out code body_text
    out=$(curl_via "$path" "$body" "$LOAD_TIMEOUT")
    code=$(http_code "$out")
    body_text=$(http_body "$out")
    if [ "$code" != "200" ]; then
        if grep -q "does not support tools" <<< "$body_text"; then
            cell_fail "$model" "tools $path" "HTTP $code (tool capability missing — RENDERER/PARSER?)"
        else
            cell_fail "$model" "tools $path" "HTTP $code"
        fi
        return
    fi
    cell_pass "$model" "tools $path" "HTTP 200"
}

scenario_ctx() {
    local model="$1"
    # Force a fresh load so /api/ps reflects this model.
    $RUNTIME rm -f devai-ollama >/dev/null 2>&1 && \
      $COMPOSE_UP_OLLAMA >/dev/null 2>&1 || true
    until $RUNTIME exec devai-ollama ollama list >/dev/null 2>&1; do sleep 1; done
    local out code
    local req
    # options.num_ctx IS the per-session context mechanism on /api/chat: the
    # router's setNumCtx clamps it to the probed ceiling and otherwise passes
    # it through, so a value below the ceiling is what actually gets loaded.
    req=$(printf '{"model":"%s","messages":[{"role":"user","content":"hi"}],"stream":false,"options":{"num_predict":2,"num_ctx":%d}}' \
                 "$model" "$TEST_CTX")
    out=$(curl_via /api/chat "$req" "$LOAD_TIMEOUT")
    code=$(http_code "$out")
    if [ "$code" != "200" ]; then
        cell_fail "$model" "ctx /api/chat" "HTTP $code"
        return
    fi
    local ps_out actual
    ps_out=$(curl_get /api/ps 5 "$OLLAMA")
    actual=$(http_body "$ps_out" | jq -r --arg n "$model" '
        .models[]? | select(.name == $n)
        | (.context_length // .details.context_length // 0)
    ' 2>/dev/null || echo "")
    if [ "$actual" = "$TEST_CTX" ]; then
        cell_pass "$model" "ctx /api/chat" "loaded ctx=$actual"
    else
        cell_fail "$model" "ctx /api/chat" "expected ctx=$TEST_CTX, got $actual"
    fi
}

scenario_think() {
    local model="$1" path="$2" policy="$3"
    local cap; cap="$(cap_of "${model%-ctx*}")"
    if [ "$cap" != "structured" ]; then
        cell_skip "$model" "think_$policy $path" "capability=$cap"
        return
    fi
    if [ "$policy" = "off" ]; then
        local dv; dv="$(disable_verified_of "${model%-ctx*}")"
        if [ "$dv" != "true" ]; then
            cell_skip "$model" "think_off $path" "disable_verified=$dv"
            return
        fi
        # Ollama 0.21.1's Anthropic compat layer (/v1/messages) has no
        # field to suppress thinking — the model's renderer/parser keeps
        # emitting thinking content regardless of policy. Skip with a
        # clear reason rather than fail; the limitation is upstream and
        # we'd need to translate /v1/messages → /api/chat in the router
        # to make `off` honoured here.
        if [ "$path" = "/v1/messages" ]; then
            cell_skip "$model" "think_off $path" "Anthropic compat: no disable field"
            return
        fi
    fi
    local body
    case "$path" in
        /api/chat)
            body=$(printf '{"model":"%s","messages":[{"role":"user","content":"hi"}],"stream":false,"options":{"num_predict":4}}' "$model")
            ;;
        /v1/chat/completions)
            body=$(printf '{"model":"%s","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' "$model")
            ;;
        /v1/messages)
            body=$(printf '{"model":"%s","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' "$model")
            ;;
    esac
    local out code body_text has_think
    out=$(curl_via "$path" "$body" "$LOAD_TIMEOUT" "X-DevAI-Reasoning: $policy")
    code=$(http_code "$out")
    body_text=$(http_body "$out")
    if [ "$code" != "200" ]; then
        cell_fail "$model" "think_$policy $path" "HTTP $code"
        return
    fi
    case "$path" in
        /api/chat)
            has_think=$(jq -r '
                ((.message.thinking // "") | length > 0)
                or ((.message.content // "") | test("<think>"))
            ' <<< "$body_text" 2>/dev/null || echo "false")
            ;;
        /v1/chat/completions)
            has_think=$(jq -r '
                (.choices[0].message.reasoning // "" | length > 0)
                or (.choices[0].message.content // "" | test("<think>"))
            ' <<< "$body_text" 2>/dev/null || echo "false")
            ;;
        /v1/messages)
            has_think=$(jq -r '
                .content // [] | any(.type == "thinking")
            ' <<< "$body_text" 2>/dev/null || echo "false")
            ;;
    esac
    case "$policy" in
        auto)
            if [ "$has_think" = "true" ]; then
                cell_pass "$model" "think_auto $path" "thinking present"
            else
                cell_fail "$model" "think_auto $path" "thinking missing"
            fi
            ;;
        off)
            if [ "$has_think" != "true" ]; then
                cell_pass "$model" "think_off $path" "thinking suppressed"
            else
                cell_fail "$model" "think_off $path" "thinking still present"
            fi
            ;;
    esac
}

# ── Drive the matrix ───────────────────────────────────────────────────────

PROTOCOLS=("/api/chat" "/v1/chat/completions" "/v1/messages")

# Used by scenario_ctx to recreate ollama; building once here so the inner
# loop is fast.
COMPOSE_UP_OLLAMA="$RUNTIME compose -f deploy/docker-compose.yaml up -d ollama"

for parent in "${MODELS[@]}"; do
    hdr "── $parent ──"

    # The parent tag, NOT a derived -ctx sibling: only the parent has a probe
    # row, and that row is what supplies the per-tier KV-cache dtype.
    for path in "${PROTOCOLS[@]}"; do
        scenario_basic "$parent" "$path"
        scenario_tools "$parent" "$path"
        scenario_think "$parent" "$path" "auto"
        scenario_think "$parent" "$path" "off"
    done

    scenario_ctx "$parent"
done

# ── Summary ────────────────────────────────────────────────────────────────
hdr "summary"
echo "  ${G}PASS${N} = $PASS"
echo "  ${R}FAIL${N} = $FAIL"
echo "  ${Y}SKIP${N} = $SKIP"
if [ "$FAIL" -gt 0 ]; then
    echo
    echo "  failed cells:"
    for line in "${FAIL_LINES[@]}"; do
        echo "    - $line"
    done
    exit 1
fi
exit 0
