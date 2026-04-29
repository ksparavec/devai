#!/usr/bin/env python3
"""Probe downloaded vLLM models per (VRAM band, context tier).

Each probe launches a vLLM server with a specific model + context cap,
waits for /health, sends a deterministic chat request, then snapshots
GPU memory via nvidia-smi. Inline `<think>` markers in the response
classify capability. Top-level fields are populated on the first
observed cell; subsequent cells only record their fit verdict.

Cache file: deploy/.vllm-reasoning-cache.json (schema v1, repo+sha keyed).
See scripts/_probe_hf_common.py for the shared cache shape.

Hard pre-condition: devai-router, devai-vllm, and devai-sglang must be
stopped — the prober launches devai-vllm-probe with explicit GPU
exclusivity. Run `make cache-down` first.

Errors propagate verbatim. No exception swallowing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from _probe_hf_common import (  # noqa: E402  — local import after sys.path fix
    BackendSpec,
    build_argparser,
    run_probe_pass,
)


VLLM_RESERVE_GB = 2.0  # mirrors gpu-arbiter/main.go memFraction
DEFAULT_VLLM_IMAGE = os.environ.get(
    "VLLM_IMAGE", "docker.io/vllm/vllm-openai:latest-cu130-ubuntu2404"
)


def vllm_command_args(
    model_name: str, max_ctx: int, host_frac: float
) -> list[str]:
    """Build the vLLM serve arguments. Mirrors gpu-arbiter vllmEntrypoint
    minus the tool-parser flags (Phase 0 made those parameter-driven and
    the prober omits them — tool detection is a follow-on).
    """
    return [
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", f"/models/{model_name}",
        "--host", "0.0.0.0",
        "--port", "11434",
        "--tensor-parallel-size", "1",
        "--max-model-len", str(max_ctx),
        "--gpu-memory-utilization", f"{host_frac:.4f}",
        "--enable-prefix-caching",
        "--trust-remote-code",
        "--served-model-name", model_name,
    ]


SPEC = BackendSpec(
    name="vllm",
    image=DEFAULT_VLLM_IMAGE,
    container_name="devai-vllm-probe",
    probe_port=18000,
    cache_path=REPO_ROOT / "deploy" / ".vllm-reasoning-cache.json",
    reserve_gb=VLLM_RESERVE_GB,
    entrypoint="python3",
    build_args=vllm_command_args,
)


def main() -> None:
    ap = build_argparser(SPEC, __doc__)
    run_probe_pass(SPEC, ap.parse_args())


if __name__ == "__main__":
    main()
