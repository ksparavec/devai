#!/usr/bin/env bash
# LATE launcher — applies defaults so `late` works standalone, but honours
# any OPENAI_* vars already set by the caller (e.g. model-picker).
#
# Precedence:
#   caller env  >  wrapper defaults  >  hard fallback
#
# Defaults route to the in-cluster router on the Ollama OpenAI-compat port.

: "${OPENAI_BASE_URL:=${OLLAMA_HOST:-http://devai-router:11434}/v1}"
: "${OPENAI_API_KEY:=local}"
: "${OPENAI_MODEL:=${OLLAMA_DEFAULT_MODEL:-qwen3.5:9b}}"
export OPENAI_BASE_URL OPENAI_API_KEY OPENAI_MODEL

exec /usr/local/bin/late-bin "$@"
