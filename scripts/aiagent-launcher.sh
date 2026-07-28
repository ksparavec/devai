#!/usr/bin/env bash
# aiagent shell launcher -- the devai picker's "AIAgent (shell)" agent.
#
# aiagent is a CLI the user drives explicitly, so instead of exec'ing the agent
# we configure the router endpoint + model in the environment and drop the user
# into an interactive bash shell. The user then runs, for example:
#     aiagent doctor        # verify the router
#     aiagent chat          # interactive chat with the configured model
#     aiagent run <skill>   # run a skill / pipeline once
#     aiagent --help        # full command surface
#
# Env contract (aiagent resolves: env > TOML > devai-env > defaults):
#   AIAGENT_API_BASE  OpenAI-compatible base URL, INCLUDING the /v1 suffix
#   AIAGENT_API_KEY   ignored by local backends, but the client requires a value
#   AIAGENT_MODEL     router model string (Ollama tag, or <name>@<ctx> for vLLM)
#
# We deliberately do NOT set AIAGENT_CONTEXT. aiagent turns it into a
# `<model>@<ctx>` control-surface suffix, which is redundant and can double up:
# Ollama's /v1 surface uses the router's global OLLAMA_CONTEXT_LENGTH (per-request
# ctx is ignored), and vLLM's ctx already rides in AIAGENT_MODEL's own `@<ctx>`
# (composed by the picker) -- so also setting AIAGENT_CONTEXT would risk
# `<name>@<ctx>@<ctx>`. (aiagent <= v0.1.1 additionally composed `@<ctx>` BEFORE
# `::<reasoning>`, which the router mis-parsed into an Ollama "invalid model
# name"; fixed in v0.1.2, devitops-com/aiagent#3.) See docs/aiagent.md.
#
# GPU policy (DEVAI_AIAGENT_GPU): the default "router-only" hides the GPU so
# aiagent cannot contend with the router-loaded model for VRAM -- all compute
# flows through the router's OpenAI endpoint. "share" leaves the GPU visible so
# aiagent may run its own CUDA code, accepting the OOM risk against the loaded
# model. The picker sets this from a sub-modal; a pre-set env value wins and
# suppresses the prompt.
#
# Precedence for every knob below: caller env (picker) > these defaults >
# hard fallback.

set -u

# Resolve the base URL, then ensure it ends in EXACTLY one /v1. The picker's
# _build sets AIAGENT_API_BASE directly (already /v1-suffixed), so this fallback
# only runs for standalone `aiagent-shell` use. OLLAMA_HOST / the hard fallback
# carry no /v1, but an OPENAI_BASE_URL already does (OpenAI-SDK convention, and
# this project's own -- model-picker.py setdefaults it WITH /v1), so append
# conditionally rather than blindly -- otherwise a preset OPENAI_BASE_URL=.../v1
# yields /v1/v1 -> HTTP 404.
if [ -z "${AIAGENT_API_BASE:-}" ]; then
    _base="${OPENAI_BASE_URL:-${OLLAMA_HOST:-http://devai-router:11434}}"
    _base="${_base%/}"
    case "${_base}" in
        */v1) AIAGENT_API_BASE="${_base}" ;;
        *)    AIAGENT_API_BASE="${_base}/v1" ;;
    esac
fi
: "${AIAGENT_API_KEY:=local}"
: "${AIAGENT_MODEL:=${OLLAMA_DEFAULT_MODEL:-qwen3.5:9b}}"
export AIAGENT_API_BASE AIAGENT_API_KEY AIAGENT_MODEL

gpu_mode="${DEVAI_AIAGENT_GPU:-router-only}"
case "${gpu_mode}" in
    share)
        gpu_note="shared with router (aiagent may use the GPU directly; OOM risk)"
        ;;
    *)
        # Router-only default: hide the GPU so an accidental local model load
        # inside aiagent cannot OOM the router-held model on the 24 GB card.
        export CUDA_VISIBLE_DEVICES=""
        gpu_mode="router-only"
        gpu_note="router-only (GPU hidden; all compute via the router)"
        ;;
esac

# Banner on stderr so it never pollutes a piped stdout.
cat >&2 <<EOF

  aiagent shell -- a DSPy agent CLI you drive yourself.
    endpoint : ${AIAGENT_API_BASE}
    model    : ${AIAGENT_MODEL}
    context  : router-managed (Ollama: global; vLLM: the @ctx in the model tag)
    gpu      : ${gpu_note}
    try:  aiagent doctor   |   aiagent chat   |   aiagent --help
    (type 'exit' to leave the shell and return to the picker/host)

EOF

# Test / preview hook: print the resolved environment and exit without opening
# an interactive shell. Exercised by tests/python/test_aiagent_picker.py.
if [ -n "${DEVAI_AIAGENT_SHELL_DEBUG:-}" ]; then
    echo "AIAGENT_API_BASE=${AIAGENT_API_BASE}"
    echo "AIAGENT_API_KEY=${AIAGENT_API_KEY}"
    echo "AIAGENT_MODEL=${AIAGENT_MODEL}"
    echo "AIAGENT_CONTEXT=${AIAGENT_CONTEXT:-}"
    echo "DEVAI_AIAGENT_GPU=${gpu_mode}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
    exit 0
fi

exec bash -i
