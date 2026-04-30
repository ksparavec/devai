#!/usr/bin/env python3
"""Probe downloaded SGLang models per (VRAM band, context tier).

Mirrors probe-vllm-reasoning.py via the shared
scripts/_probe_hf_common.py scaffold. Each probe launches an SGLang
server with `--mem-fraction-static`, polls /health, then sends a
deterministic /v1/chat/completions request and snapshots GPU memory.

Cache file: deploy/.sglang-reasoning-cache.json (schema v1, repo+sha
keyed). Same shape as the vLLM cache; consumers (router, picker)
read both.

SGLang differences from vLLM:
  - Memory utilisation flag: `--mem-fraction-static` (vLLM:
    `--gpu-memory-utilization`) — same semantics, different name.
  - Context flag: `--context-length` (vLLM: `--max-model-len`).
  - Reserve budget: 3.0 GB (vLLM: 2.0 GB) — RadixAttention tree +
    CUDA graphs. Mirrors gpu-arbiter/main.go memFraction.
  - Default served-model name is the directory basename, not the
    catalog row name (no --served-model-name flag in the router's
    sglangEntrypoint either). The probe sends `model: <basename>` in
    the chat body, which here equals the catalog `name` — so the
    same field works.

Hard pre-condition: devai-router, devai-vllm, and devai-sglang must
be stopped — the prober launches devai-sglang-probe with explicit
GPU exclusivity. Run `make cache-down` first.

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


SGLANG_RESERVE_GB = 3.0  # mirrors gpu-arbiter/main.go memFraction
DEFAULT_SGLANG_IMAGE = os.environ.get(
    "SGLANG_IMAGE", "docker.io/lmsysorg/sglang:v0.5.10.post1-cu130"
)


def sglang_command_args(
    model_name: str, max_ctx: int, host_frac: float,
    *, reasoning_parser: str | None = None, tool_parser: str | None = None,
) -> list[str]:
    """Build SGLang launch arguments. Mirrors gpu-arbiter sglangEntrypoint.

    Parser flags are appended only when the catalog row's
    `parsers.sglang` block supplied a value. Omitting them keeps the
    launch in inline / no-tool mode. Both flag names verified against
    the v0.5.10.post1-cu130 image — see deploy/backend-flags.yaml.
    """
    args = [
        "-m", "sglang.launch_server",
        "--model-path", f"/models/{model_name}",
        "--host", "0.0.0.0",
        "--port", "11434",
        "--tp", "1",
        "--mem-fraction-static", f"{host_frac:.4f}",
        "--context-length", str(max_ctx),
        "--trust-remote-code",
    ]
    if reasoning_parser:
        args.extend(["--reasoning-parser", reasoning_parser])
    if tool_parser:
        args.extend(["--tool-call-parser", tool_parser])
    return args


SPEC = BackendSpec(
    name="sglang",
    image=DEFAULT_SGLANG_IMAGE,
    container_name="devai-sglang-probe",
    probe_port=18001,
    cache_path=REPO_ROOT / "deploy" / ".sglang-reasoning-cache.json",
    reserve_gb=SGLANG_RESERVE_GB,
    entrypoint="python3",
    build_args=sglang_command_args,
)


def main() -> None:
    ap = build_argparser(SPEC, __doc__)
    run_probe_pass(SPEC, ap.parse_args())


if __name__ == "__main__":
    main()
