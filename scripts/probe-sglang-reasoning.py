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
from _probe_load import run_load_probe_pass  # noqa: E402


SGLANG_RESERVE_GB = 3.0  # mirrors gpu-arbiter/main.go memFraction
DEFAULT_SGLANG_IMAGE = os.environ.get(
    "SGLANG_IMAGE", "docker.io/lmsysorg/sglang:v0.5.10.post1-cu130"
)


def sglang_command_args(
    model_name: str, max_ctx: int, host_frac: float,
    *,
    reasoning_parser: str | None = None,
    tool_parser: str | None = None,
    reasoning_parser_plugin: str | None = None,
    tool_parser_plugin: str | None = None,
    speculative_config: str | None = None,
    kv_cache_dtype: str | None = None,
) -> list[str]:
    """Build SGLang launch arguments. Mirrors gpu-arbiter sglangEntrypoint.

    Parser flags are appended only when the catalog row's
    `parsers.sglang` block supplied a value. Omitting them keeps the
    launch in inline / no-tool mode. Both flag names verified against
    the v0.5.10.post1-cu130 image — see deploy/backend-flags.yaml.

    The ``*_parser_plugin`` kwargs are accepted but ignored: SGLang
    registers parsers via Python imports, not file-path args. They're
    in the signature for parity with vllm_command_args so the shared
    probe driver can call both backends through the same kwargs shape.

    ``speculative_config`` is accepted but **discarded** for SGLang.
    NVFP4 loading itself is no longer broken: --disable-piecewise-cuda-graph
    (added to the launch args below) unblocks it -- before that flag every
    NVFP4 cell crashed a Dynamo graph break at modelopt_quant.py:1482 during
    piecewise CUDA-graph warmup. But the SGLang MTP / --speculative-* path is
    not yet validated on this fleet, so emitting it now would only produce
    noise. The kwarg stays in the signature for parity with vllm_command_args.
    To enable SGLang MTP probing later, emit the --speculative-* family
    (pinned in deploy/backend-flags.yaml).

    ``kv_cache_dtype`` -- explicit KV dtype for THIS launch. None means
    "use the pass default" (``KV_CACHE_DTYPE``, empty = engine default).
    The load probe passes the dtype the target cell was fit-probed under
    so its serving numbers describe the dtype the cell advertises.
    """
    del reasoning_parser_plugin, tool_parser_plugin, speculative_config
    dtype = KV_CACHE_DTYPE if kv_cache_dtype is None else kv_cache_dtype
    args = [
        "-m", "sglang.launch_server",
        "--model-path", f"/models/{model_name}",
        "--host", "0.0.0.0",
        "--port", "11434",
        "--tp", "1",
        "--mem-fraction-static", f"{host_frac:.4f}",
        "--context-length", str(max_ctx),
        "--trust-remote-code",
        # SGLang v0.5.10 enables piecewise CUDA graph by default, which
        # torch.compiles the forward; Dynamo then can't trace flashinfer's
        # FP4 JIT path (modelopt_quant.py:1482 -> fp4_quantize -> a
        # subprocess/threading.Lock) and every NVFP4 load crashes with a
        # graph break. Disabling it runs FP4 eager (JITs fine) -- the
        # engine's documented workaround. Must match gpu-arbiter
        # sglangEntrypoint so probe and serve launch identically.
        "--disable-piecewise-cuda-graph",
    ]
    # KV-cache dtype for THIS launch. SGLang's default is auto (no flag,
    # unquantized); a PROBE_KV_CACHE_TYPE=fp8_e5m2/fp8_e4m3 pass measures
    # quantized-KV fit and stamps the cell so serve time reproduces it.
    # Empty = no flag = engine default (unchanged).
    if dtype:
        args.extend(["--kv-cache-dtype", dtype])
    # SGLang's analogue of vLLM's --max-num-seqs. Emitted for the same
    # reason: the router always passes it, the prober used to omit it, so
    # probe-time and serve-time reservations diverged. Before the parser
    # flags, matching gpu-arbiter sglangEntrypoint's ordering.
    if MAX_RUNNING_REQUESTS > 0:
        args.extend(["--max-running-requests", str(MAX_RUNNING_REQUESTS)])
    if reasoning_parser:
        args.extend(["--reasoning-parser", reasoning_parser])
    if tool_parser:
        args.extend(["--tool-call-parser", tool_parser])
    return args


# Concurrent-request cap this pass launches with, mirroring the router's
# MAX_CONCURRENT_REQUESTS (default 32). SGLang sizes its CUDA-graph
# capture set off this, so a probe that omits it does not measure the
# configuration that actually serves. 0 omits the flag (engine default),
# matching the router's own guard. Shares PROBE_MAX_NUM_SEQS with the
# vLLM prober so one knob moves both backends together.
MAX_RUNNING_REQUESTS = int(os.environ.get("PROBE_MAX_NUM_SEQS") or "32")

# KV dtype this probe pass enforces; empty = SGLang engine default
# (auto/unquantized), matching every pre-field cache cell.
KV_CACHE_DTYPE = os.environ.get("PROBE_KV_CACHE_TYPE") or ""

SPEC = BackendSpec(
    name="sglang",
    image=DEFAULT_SGLANG_IMAGE,
    container_name="devai-sglang-probe",
    probe_port=18001,
    cache_path=REPO_ROOT / "deploy" / ".sglang-reasoning-cache.json",
    reserve_gb=SGLANG_RESERVE_GB,
    entrypoint="python3",
    build_args=sglang_command_args,
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
