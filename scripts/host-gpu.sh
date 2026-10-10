#!/usr/bin/env bash
# host-gpu.sh -- the host check (platform-rebuild Phase 3, M15): prints the
# CDI device string devai's containers get, or nothing when they run on the
# CPU. One image runs both ways; this decides which.
#
#   DEVAI_GPU_DEVICE in the environment   printed as it is; set but EMPTY
#                                         forces the CPU
#   `nvidia-smi -L` lists a GPU and the   nvidia.com/gpu=all
#   CDI spec names nvidia.com/gpu=all
#   anything else                         nothing
#
# The CDI spec is read through `nvidia-ctk cdi list`, which reads the same
# spec directories podman does (/etc/cdi, /var/run/cdi). Exits 0 in every
# case: no GPU is an answer, not an error.
#
# Used by the Makefile (DEVAI_GPU_DEVICE, unless set in the environment or on
# the command line) and by bin/devai-agent (staged in ~/.devai by
# `make install`). Pinned by tests/python/test_host_gpu.py.
set -u

if [ -n "${DEVAI_GPU_DEVICE+x}" ]; then
    [ -n "$DEVAI_GPU_DEVICE" ] && printf '%s\n' "$DEVAI_GPU_DEVICE"
    exit 0
fi

nvidia-smi -L 2>/dev/null | grep -q '^GPU ' || exit 0
nvidia-ctk cdi list 2>/dev/null | grep -qx 'nvidia.com/gpu=all' || exit 0
echo "nvidia.com/gpu=all"
