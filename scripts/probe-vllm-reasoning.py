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
from _probe_load import run_load_probe_pass  # noqa: E402


VLLM_RESERVE_GB = 2.0  # mirrors gpu-arbiter/main.go memFraction
DEFAULT_VLLM_IMAGE = os.environ.get(
    "VLLM_IMAGE", "docker.io/vllm/vllm-openai:v0.22.1-x86_64-cu129-ubuntu2404"
)


def vllm_command_args(
    model_name: str, max_ctx: int, host_frac: float,
    *,
    reasoning_parser: str | None = None,
    tool_parser: str | None = None,
    reasoning_parser_plugin: str | None = None,
    tool_parser_plugin: str | None = None,
    speculative_config: str | None = None,
) -> list[str]:
    """Build the vLLM serve arguments. Mirrors gpu-arbiter vllmEntrypoint.

    Parser flags are appended only when the catalog row's `parsers.vllm`
    block supplied a value. Omitting them keeps the launch in inline /
    no-tool mode (same shape as a model with no curated hints).

    When the parser name resolves to a custom plugin via
    ``deploy/vllm-plugins.json``, the corresponding ``*_parser_plugin``
    kwarg carries the in-container absolute path. The plugin flag must
    precede the parser-name flag — vLLM resolves parser names at the
    point ``--tool-call-parser`` is evaluated, so the plugin module has
    to be loaded by then.

    ``speculative_config`` -- when non-empty, append
    ``--speculative-config <json>`` to enable multi-token-prediction.
    Mirrors the router's vllmEntrypoint emission so probe-time and
    serve-time launches use the same flag shape (and therefore the
    same memory math).
    """
    args = [
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", f"/models/{model_name}",
        "--host", "0.0.0.0",
        "--port", "11434",
        "--tensor-parallel-size", "1",
        "--max-model-len", str(max_ctx),
        # KV-cache dtype for THIS probe pass (default fp8: halves KV
        # memory vs fp16 -- on 24 GiB that's what makes 128K+ contexts
        # on 18 GiB NVFP4 weights fit). Overridable per pass via
        # PROBE_KV_CACHE_TYPE (e.g. `auto` to measure unquantized KV);
        # each probed cell is stamped kv_cache_type so the router
        # reproduces the measured dtype at serve time -- fit is only
        # valid under the dtype it was measured with.
        "--kv-cache-dtype", KV_CACHE_DTYPE,
        "--gpu-memory-utilization", f"{host_frac:.4f}",
        "--enable-prefix-caching",
        "--trust-remote-code",
        "--served-model-name", model_name,
    ]
    if reasoning_parser_plugin:
        args.extend(["--reasoning-parser-plugin", reasoning_parser_plugin])
    if reasoning_parser:
        args.extend(["--reasoning-parser", reasoning_parser])
    if tool_parser_plugin:
        args.extend(["--tool-parser-plugin", tool_parser_plugin])
    if tool_parser:
        args.extend(["--enable-auto-tool-choice", "--tool-call-parser", tool_parser])
    if speculative_config:
        args.extend(["--speculative-config", speculative_config])
    return args


# KV dtype this probe pass launches every cell with. Historical default
# is fp8 (all pre-field cache cells were measured under it); a
# PROBE_KV_CACHE_TYPE=auto pass measures unquantized KV for models with
# VRAM slack.
KV_CACHE_DTYPE = os.environ.get("PROBE_KV_CACHE_TYPE") or "fp8"

SPEC = BackendSpec(
    name="vllm",
    image=DEFAULT_VLLM_IMAGE,
    container_name="devai-vllm-probe",
    probe_port=18000,
    cache_path=REPO_ROOT / "deploy" / ".vllm-reasoning-cache.json",
    reserve_gb=VLLM_RESERVE_GB,
    entrypoint="python3",
    build_args=vllm_command_args,
    supports_plugins=True,
    kv_cache_dtype=KV_CACHE_DTYPE,
)


def main() -> None:
    ap = build_argparser(SPEC, __doc__)
    args = ap.parse_args()
    if args.load:
        run_load_probe_pass(SPEC, args)
    else:
        run_probe_pass(SPEC, args)


if __name__ == "__main__":
    main()
