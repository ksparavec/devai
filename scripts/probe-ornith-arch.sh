#!/usr/bin/env bash
# Arch-support probe for the Ornith-1.0 35B (qwen3_5_moe, multimodal MoE).
#
# WHY: on a 24G card the 35B does not fit at bf16 (~70 GB) or FP8 (~35 GB),
# and only GGUF Q4_K_M (~20 GiB) fits at all. Before spending a ~20-70 GB
# download -- or any FP4/NVFP4 quantization effort -- on the 35B, we need to
# know whether the backends even SUPPORT the qwen3_5_moe + vision arch. A
# plain fit probe cannot answer that cleanly: an unsupported arch and a
# too-big model both "fail to load", and we must tell them apart.
#
# TIER 1 (this script, CHEAP -- no weight download): the devai-vllm /
# devai-sglang / devai-ollama containers run the REAL backend images (as
# `sleep infinity` placeholders until the router recreates them). We exec
# into them and inspect the installed model registry / source for the arch
# class string. If the arch is not registered, no amount of VRAM will load
# it -- stop here. If it IS registered, the remaining question is fit/OOM.
#
# TIER 2 (DEFINITIVE, needs the weights + `make cache-down`): add the 35B
# repo under the `ornith` family in scripts/model-families.yaml, then run
# `make probe-vllm` / `make probe-sglang`. The project's failure classifier
# (scripts/_probe_core.py) already distinguishes `unsupported_arch` from
# `oom_startup` and records it in deploy/.model-status.json. See the echoed
# guidance at the end of this script and docs/backends.md.
#
# Exit code: 0 = arch registered on >=1 serving backend (vLLM/SGLang);
#            1 = not found on any reachable backend;
#            3 = inconclusive (backend containers not running / introspection
#                errored). Run `make cache-up` first.

set -euo pipefail

# Runtime + container names (override via env). CONTAINER_RUNTIME falls back
# to deploy/.env then podman, matching the rest of the repo.
_env_runtime="$(grep -hE '^CONTAINER_RUNTIME=' deploy/.env .env 2>/dev/null \
    | head -1 | cut -d= -f2- | tr -d '"'"'"' ' || true)"
RT="${CONTAINER_RUNTIME:-${_env_runtime:-podman}}"

VLLM_CONTAINER="${VLLM_CONTAINER:-devai-vllm}"
SGLANG_CONTAINER="${SGLANG_CONTAINER:-devai-sglang}"
OLLAMA_CONTAINER="${OLLAMA_CONTAINER:-devai-ollama}"

# What we are probing for. Defaults to the Ornith-1.0-35B arch; override to
# reuse this script for any other architecture.
ARCH_CLASS="${ARCH_CLASS:-Qwen3_5MoeForConditionalGeneration}"
ARCH_TOKEN="${ARCH_TOKEN:-qwen3_5_moe}"   # model_type / gguf general.architecture
REPO="${REPO:-deepreinforce-ai/Ornith-1.0-35B}"

# Case-insensitive extended regex matching either the HF arch class or the
# model_type token.
_PAT="${ARCH_CLASS}|${ARCH_TOKEN}"

serving_supported=0   # incremented when vLLM or SGLang registers the arch
reachable=0           # incremented per backend container that was up

echo ">>> probe-ornith-arch: arch=${ARCH_CLASS} (token=${ARCH_TOKEN})"
echo ">>> runtime=${RT}  repo=${REPO}"
echo

_is_up() { "${RT}" ps --format '{{.Names}}' 2>/dev/null | grep -qx "$1"; }

# Grep the installed package's model directory for the arch string. Prints the
# matching files (if any) and returns 0 when at least one match is found.
_grep_models_dir() {
    local container="$1" import_name="$2" subdir="$3"
    "${RT}" exec "${container}" bash -lc '
        set -e
        D=$(python3 -c "import '"${import_name}"', os; print(os.path.dirname('"${import_name}"'.__file__))" 2>/dev/null) || exit 3
        grep -RilE "'"${_PAT}"'" "$D/'"${subdir}"'" 2>/dev/null || true
    '
}

probe_backend() {
    # $1 label, $2 container, $3 python import name, $4 models subdir
    local label="$1" container="$2" import_name="$3" subdir="$4"
    echo "--- ${label} (${container}) ---"
    if ! _is_up "${container}"; then
        echo "  SKIP: container not running (start with: make cache-up)"
        echo
        return
    fi
    reachable=$((reachable + 1))

    local hits
    hits="$(_grep_models_dir "${container}" "${import_name}" "${subdir}" || true)"
    if [[ -n "${hits}" ]]; then
        echo "  SUPPORTED: arch found in ${import_name} source:"
        echo "${hits}" | sed 's/^/    /'
        serving_supported=$((serving_supported + 1))
    else
        echo "  NOT FOUND: no ${import_name} model file references ${ARCH_TOKEN}/${ARCH_CLASS}"
        echo "  -> this backend/image build cannot serve the arch (no VRAM fix applies)"
    fi
    echo
}

# --- vLLM ------------------------------------------------------------------
probe_backend "vLLM" "${VLLM_CONTAINER}" "vllm" "model_executor/models"
# Secondary confirmation via the public registry API (tolerant of version drift).
if _is_up "${VLLM_CONTAINER}"; then
    "${RT}" exec "${VLLM_CONTAINER}" python3 -c "
try:
    from vllm.model_executor.models.registry import ModelRegistry
    archs = sorted(ModelRegistry.get_supported_archs())
    hits = [a for a in archs if 'qwen3_5' in a.lower()]
    print('  vLLM registry qwen3_5* archs:', hits or '(none)')
except Exception as e:
    print('  vLLM registry API check skipped:', type(e).__name__, e)
" 2>/dev/null || echo "  vLLM registry API check skipped (exec error)"
    echo
fi

# --- SGLang ----------------------------------------------------------------
probe_backend "SGLang" "${SGLANG_CONTAINER}" "sglang" "srt/models"

# --- Ollama / llama.cpp (GGUF path, heuristic only) ------------------------
echo "--- Ollama / llama.cpp (${OLLAMA_CONTAINER}) ---"
if _is_up "${OLLAMA_CONTAINER}"; then
    echo "  HEURISTIC: scanning the ollama binary for embedded qwen3 arch tokens"
    "${RT}" exec "${OLLAMA_CONTAINER}" bash -lc '
        BIN=$(command -v ollama 2>/dev/null || echo /usr/bin/ollama)
        strings "$BIN" 2>/dev/null | grep -iE "qwen3.?5?.?moe|qwen3_moe|qwen3moe" \
            | sort -u | head || true
    ' | sed 's/^/    /' || true
    echo "  NOTE: llama.cpp arch support for qwen3_5_moe is bleeding-edge (cf. the"
    echo "        qwen3.6 family note re: llama.cpp PR #22673). A definitive answer"
    echo "        needs a real GGUF load; the strings scan is only a proxy."
else
    echo "  SKIP: container not running"
fi
echo

# --- Verdict ---------------------------------------------------------------
echo ">>> Tier 1 verdict:"
if (( reachable == 0 )); then
    echo "  INCONCLUSIVE: no backend containers were up. Run 'make cache-up' and retry."
    _rc=3
elif (( serving_supported > 0 )); then
    echo "  SUPPORTED on ${serving_supported} serving backend(s)."
    echo "  Arch parses -> the only open question is FIT. Proceed to Tier 2 below to"
    echo "  measure real load VRAM / OOM behaviour, or consider an NVFP4 quant"
    echo "  (~19-20 GiB weights) to make the MoE fit a 24G card."
    _rc=0
else
    echo "  NOT SUPPORTED on any reachable serving backend (vLLM/SGLang)."
    echo "  The qwen3_5_moe + vision arch is not registered in these image builds;"
    echo "  no VRAM/quant change will help until the backends add it. Re-run after a"
    echo "  backend image bump (see deploy/backend-flags.yaml + make verify-backend-flags)."
    _rc=1
fi

cat <<'TIER2'

>>> Tier 2 (DEFINITIVE fit + arch classification -- needs weights + cache-down):
  1. make cache-down
  2. Temporarily add the 35B under the `ornith` family in
     scripts/model-families.yaml, e.g.:
         hf_repos:
           - deepreinforce-ai/Ornith-1.0-35B        # bf16 safetensors (~70 GB)
           - deepreinforce-ai/Ornith-1.0-35B-FP8    # FP8 (~35 GB)
     and/or a gguf_repos entry for deepreinforce-ai/Ornith-1.0-35B-GGUF.
  3. make catalog-regen
  4. make probe-vllm   # and: make probe-sglang
  5. Read the outcome -- the classifier tags the row:
       - unsupported_arch  -> backend genuinely cannot serve it (arch gap)
       - oom / oom_startup -> arch OK, just too big for 24G (FP4 would help)
     via: make model-status   (deploy/.model-status.json)
  Revert the temporary model-families.yaml edit afterwards if the arch is
  unsupported or the model does not fit.
TIER2

exit "${_rc}"
