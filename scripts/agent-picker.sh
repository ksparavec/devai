#!/bin/bash
# Interactive agent picker for devai shell sessions
# Falls back to bash after timeout or if no agent is selected

DEFAULT_MODEL="${OLLAMA_DEFAULT_MODEL:-qwen3.5:9b}"

AGENTS=(
  "claude-ollama|Claude Code (Ollama)|ollama launch claude --model $DEFAULT_MODEL --yes"
  "claude-vllm|Claude Code (vLLM)|ANTHROPIC_BASE_URL=http://devai-router:11435 claude --model Qwen3.5-9B-NVFP4"
  "claude-sglang|Claude Code (SGLang)|ANTHROPIC_BASE_URL=http://devai-router:11436 claude --model Qwen3.5-9B-NVFP4"
  "aider|Aider (Ollama)|aider-launcher"
  "bash|Bash Shell|bash"
)

echo ""
echo "  DevAI Agent Picker"
echo "  ─────────────────────"
for i in "${!AGENTS[@]}"; do
  IFS='|' read -r id name cmd <<< "${AGENTS[$i]}"
  printf "  [%d] %s\n" "$((i+1))" "$name"
done
echo ""
read -t 10 -p "  Select [1-${#AGENTS[@]}] (default: bash in 10s): " choice
echo ""

# Default to bash (last entry) on timeout or empty
if [ -z "$choice" ]; then
  choice=${#AGENTS[@]}
fi

idx=$((choice - 1))
if [ "$idx" -ge 0 ] && [ "$idx" -lt "${#AGENTS[@]}" ]; then
  IFS='|' read -r id name cmd <<< "${AGENTS[$idx]}"
  echo "  Starting $name..."
  IFS=' ' read -ra cmd_arr <<< "$cmd"
  exec "${cmd_arr[@]}"
else
  echo "  Invalid selection, starting bash..."
  exec bash
fi
