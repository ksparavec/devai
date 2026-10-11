#!/usr/bin/env python3
"""Probe the vLLM-devai backend: the home-built, HyperQwen-patched vLLM.

Same engine, launch argv, KV dtypes and model store as probe-vllm-reasoning.py
(it reuses that script's SPEC wholesale); its own name, image, probe container
and cache file. The cache is separate because every cell is stamped with the
image it was measured on, and this backend's image is not vllm's -- mixing
them in one cache is what the drift check treats as invalid. The vLLM version
is the build record scripts/build-vllm.sh built the image from
(VLLM_DEVAI_IMAGE selects the tag). Every launch carries
FLASHINFER_CUBINS_REPOSITORY=file:///nonexistent/ (_probe_hf_common
NO_RUNTIME_CUBINS_ENV, plan 4v).

Usage is identical to probe-vllm-reasoning.py (`make probe-vllm-devai`, same
PROBE_* knobs). The two prepared Qwen3.8 checkpoints only serve here.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _probe_hf_common import build_argparser, run_probe_pass  # noqa: E402
from _probe_load import run_load_probe_pass  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "probe_vllm_reasoning", REPO_ROOT / "scripts" / "probe-vllm-reasoning.py")
_vllm = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("probe_vllm_reasoning", _vllm)
_spec.loader.exec_module(_vllm)

DEFAULT_IMAGE = os.environ.get("VLLM_DEVAI_IMAGE", "docker.io/devai/vllm-devai:latest")

# The 0.30.0 backend's engine args: prefix-cache retention kept dense
# (plan 4v's retention rule, adopted 2026-10-07; docs/backends.md "vLLM
# 0.30.0"). Emitted after vLLM's own builder and before the recovery flags
# _probe_hf_common appends -- the same constant in the same place as
# gpu-arbiter's engineArgDefaults("vllm-devai"), pinned by
# tests/python/test_vllm_devai_backend.py. Stock vLLM 0.22.1 has no such
# flag, so probe-vllm-reasoning.py does not emit it.
PREFIX_CACHE_RETENTION_ARGS = ["--prefix-cache-retention-interval", "None"]


def vllm_devai_command_args(*args, **kwargs) -> list[str]:
    return list(_vllm.SPEC.build_args(*args, **kwargs)) + PREFIX_CACHE_RETENTION_ARGS


SPEC = replace(
    _vllm.SPEC,
    name="vllm-devai",
    build_args=vllm_devai_command_args,
    image=DEFAULT_IMAGE,
    container_name="devai-vllm-devai-probe",
    cache_path=REPO_ROOT / "deploy" / ".vllm-devai-reasoning-cache.json",
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
