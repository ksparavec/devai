#!/usr/bin/env bash
# Cloud-init entrypoint for SkyPilot-launched worker VMs.
#
# Per docs/plans/gpu-arbiter-cluster-mode.md decision 11: defines the
# env-var contract that the SkyPilot fleet provisioner consumes. A
# SkyPilot task spec sets these vars as part of the launch, then the
# bootstrap image starts and this script execs the arbiter.
#
# Required env:
#   DEVAI_HEAD_URL          full URL of the head's cluster control
#                           plane (e.g. https://head.example/)
#   DEVAI_WORKER_TOKEN_FILE path to the bearer token (typically
#                           /run/devai/cluster-token, populated by
#                           SkyPilot's secret-mount mechanism)
# Recommended:
#   DEVAI_WORKER_NAME       unique-per-fleet name; defaults to the
#                           container hostname
#   DEVAI_LIFECYCLE         ephemeral|persistent (default ephemeral
#                           for SkyPilot-launched workers)
#   GPU_MEMORY_GB           VRAM the head should advertise for this
#                           worker; default 24
#   DEVAI_GPU_TYPE          short name for the head's routing logic
#                           (e.g. "RTX4000", "A100", "H100"); default
#                           "unknown"
#   DEVAI_BACKENDS          comma-separated list; default
#                           "ollama,vllm,sglang"
#   DEVAI_WORKER_INBOUND_PORT  port the worker listens on for
#                              forwarded requests (default 11444)
#
# Optional:
#   DEVAI_WORKER_HOST       hostname the head should use to reach
#                           this worker (overrides $(hostname))

set -euo pipefail

if [[ -z "${DEVAI_HEAD_URL:-}" ]]; then
    echo "FATAL: DEVAI_HEAD_URL is required" >&2
    exit 2
fi

TOKEN_FILE="${DEVAI_WORKER_TOKEN_FILE:-/run/devai/cluster-token}"
if [[ ! -r "${TOKEN_FILE}" ]]; then
    echo "FATAL: token file ${TOKEN_FILE} is not readable" >&2
    exit 2
fi

# Normalise lifecycle (we accept either case from the operator).
DEVAI_LIFECYCLE="${DEVAI_LIFECYCLE:-ephemeral}"
DEVAI_LIFECYCLE="${DEVAI_LIFECYCLE,,}"
case "${DEVAI_LIFECYCLE}" in
    ephemeral|persistent) ;;
    *) echo "FATAL: DEVAI_LIFECYCLE must be ephemeral or persistent (got ${DEVAI_LIFECYCLE})" >&2; exit 2 ;;
esac
export DEVAI_LIFECYCLE

# Default the worker name to the container hostname so two
# SkyPilot-launched workers in the same head get distinct names
# without per-launch config.
export DEVAI_WORKER_NAME="${DEVAI_WORKER_NAME:-$(hostname)}"

echo "[worker-cloud-init] starting"
echo "  head:      ${DEVAI_HEAD_URL}"
echo "  name:      ${DEVAI_WORKER_NAME}"
echo "  lifecycle: ${DEVAI_LIFECYCLE}"
echo "  gpu_type:  ${DEVAI_GPU_TYPE:-unknown}"
echo "  vram_gb:   ${GPU_MEMORY_GB:-24}"
echo "  backends:  ${DEVAI_BACKENDS:-ollama,vllm,sglang}"
echo "  inbound:   :${DEVAI_WORKER_INBOUND_PORT:-11444}"

exec /usr/local/bin/gpu-arbiter --mode=worker
