#!/usr/bin/env bash
# Verifies the GPU-vendor overlay (docs/gpu-vendors.md): flipping
# DEVAI_GPU_VENDOR via devai-gpu-vendor changes the rendered compose
# config's device string + backend image tags, both directions.
#
# Builds devai-gpu-vendor with the local Go toolchain if present
# (matching tests/test-backup-restore.sh's convention); skips with
# exit 77 if neither Go nor a container runtime is available.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${CONTAINER_RUNTIME:-podman}"
if ! command -v "${RUNTIME}" >/dev/null 2>&1; then
    if command -v docker >/dev/null 2>&1; then
        RUNTIME=docker
    else
        echo "SKIP: neither podman nor docker available" >&2
        exit 77
    fi
fi
if ! command -v go >/dev/null 2>&1; then
    echo "SKIP: no local Go toolchain (use 'make build-gpu-vendor-tool' + adapt PATH)" >&2
    exit 77
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT

BIN="${WORKDIR}/devai-gpu-vendor"
ENV_FILE="${WORKDIR}/.env"

echo ">>> test-gpu-vendor: building devai-gpu-vendor"
(cd "${REPO_ROOT}/devai-tools" && go build -o "${BIN}" ./cmd/devai-gpu-vendor)

render() {
    # env -u: compose resolves ${VAR} from the OS environment FIRST and only
    # falls back to --env-file, so an inherited value silently wins over the
    # file under test. The Makefile exports VLLM_IMAGE (see its comment at
    # the VLLM_IMAGE ?= line -- host-run probers need it), which meant this
    # test passed standalone and failed under `make test-gpu-vendor`,
    # rendering .env's pinned image instead of the ROCm one just written.
    # Unset exactly the four vars devai-gpu-vendor manages so ENV_FILE is
    # authoritative for them and this asserts the .env -> compose path.
    env -u DEVAI_GPU_VENDOR -u DEVAI_GPU_DEVICE -u VLLM_IMAGE -u SGLANG_IMAGE \
        "${RUNTIME}" compose --env-file "${ENV_FILE}" \
        -f "${REPO_ROOT}/deploy/docker-compose.yaml" config
}

echo ">>> flipping to amd"
"${BIN}" --env-file "${ENV_FILE}" --vendor amd
rendered=$(render)
echo "${rendered}" | grep -q 'amd.com/gpu=all' || { echo "FAIL: amd.com/gpu=all not in rendered config" >&2; exit 1; }
echo "${rendered}" | grep -q 'vllm-openai-rocm' || { echo "FAIL: ROCm vLLM image not in rendered config" >&2; exit 1; }
echo "${rendered}" | grep -q 'sglang:latest-rocm' || { echo "FAIL: ROCm SGLang image not in rendered config" >&2; exit 1; }
if echo "${rendered}" | grep -q 'nvidia.com/gpu=all'; then
    echo "FAIL: nvidia.com/gpu=all still present after flipping to amd" >&2
    exit 1
fi
echo "  OK: amd.com/gpu=all + ROCm images present, nvidia.com/gpu=all absent"

echo ">>> flipping back to nvidia"
"${BIN}" --env-file "${ENV_FILE}" --vendor nvidia
rendered=$(render)
echo "${rendered}" | grep -q 'nvidia.com/gpu=all' || { echo "FAIL: nvidia.com/gpu=all not in rendered config" >&2; exit 1; }
echo "${rendered}" | grep -q 'vllm-openai:v0.22.1-x86_64-cu129-ubuntu2404' || { echo "FAIL: original vLLM image not restored" >&2; exit 1; }
echo "${rendered}" | grep -q 'sglang:v0.5.16-cu130' || { echo "FAIL: original SGLang image not restored" >&2; exit 1; }
if echo "${rendered}" | grep -q 'amd.com/gpu=all'; then
    echo "FAIL: amd.com/gpu=all still present after flipping back to nvidia" >&2
    exit 1
fi
echo "  OK: original NVIDIA values restored"

echo ">>> test-gpu-vendor OK"
exit 0
