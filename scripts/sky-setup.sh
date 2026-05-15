#!/usr/bin/env bash
# First-launch helper for SkyPilot in the devai lab.
#
# Run once inside the lab container after credentials are mounted via
# $HOME (per the existing devai pattern -- ~/.aws/, ~/.config/gcloud/,
# ~/.config/sky/ flow through automatically).
#
# Output:
#   - Detected credential surfaces (which clouds Sky will see).
#   - `sky check` enabled-cloud summary.
#   - Next-step hint (run a hello-world spin-up via your CLI agent).
#
# Per docs/plans/skypilot-agent-skill.md Phase 1 deliverable.

set -euo pipefail

if ! command -v sky >/dev/null 2>&1; then
    echo "ERROR: sky CLI not on PATH." >&2
    echo "       The lab image installs it during build from the wheel" >&2
    echo "       cache populated by 'make fetch-cli'. If you skipped that," >&2
    echo "       install it now with:" >&2
    echo "         uv pip install --system 'skypilot[aws,gcp,azure,kubernetes,slurm,runpod,lambda]'" >&2
    exit 1
fi

echo "=== sky CLI ==="
sky --version || true
echo

echo "=== Detected credential surfaces ==="
for path in \
    "${HOME}/.aws/credentials" \
    "${HOME}/.config/gcloud/application_default_credentials.json" \
    "${HOME}/.config/sky" \
    "${HOME}/.runpod/config.toml" \
    "${HOME}/.lambdacloud" \
    "${HOME}/.kube/config"; do
    if [[ -e "${path}" ]]; then
        echo "  found: ${path}"
    else
        echo "  ABSENT: ${path}"
    fi
done
echo

echo "=== sky check ==="
sky check 2>&1 || true
echo

cat <<'NEXT'
Next steps:

  1. If `sky check` shows zero enabled clouds, set up at least one
     cloud's credentials -- see docs/skypilot-user-guide.md.
  2. Verify provisioning with a hello-world job:

       sky launch --cloud runpod --gpus 3090:1 -- echo hello
       sky down --all -y

  3. From a CLI agent (Claude Code, Codex, Gemini CLI), the
     SkyPilot Agent Skill is pre-installed -- ask the agent
     "what GPUs are available right now across my clouds?" or
     "spin up a 3090 on the cheapest cloud, run train.py, copy
     results back, then shut it down."

Cost guidance: cloud GPU jobs can spend $1-50/hour depending on
instance type. Use `sky cost-report` to see your total spend, and
keep `sky status` running while jobs are live.
NEXT
