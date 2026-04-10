#!/bin/bash
# Interactive agent picker for devai shell sessions
# Falls back to bash after timeout or if no agent is selected

DEFAULT_MODEL="${OLLAMA_DEFAULT_MODEL:-qwen3.5:9b}"

AGENTS=(
  "claude|Claude Code|ollama launch claude --model $DEFAULT_MODEL --yes"
  "codex|Codex|ollama launch codex --model $DEFAULT_MODEL --yes"
  "aider|Aider|aider-launcher"
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
