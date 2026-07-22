#!/usr/bin/env python3
"""VRAM fit estimator for the devai model catalog.

Prints, for every (model, backend) combination in deploy/models.yaml,
an estimate of total VRAM = weights + KV cache + overhead, and whether
it fits in a given VRAM budget at a given context length.

Architecture is read from the downloaded HF config.json for vLLM/SGLang
models (authoritative). For Ollama-only entries, architecture is derived
from a sibling NVFP4 entry with the same base model when available, or
from a built-in known-architectures table; otherwise we fall back to a
weight-based heuristic and flag the row as approximate.

Usage:
    scripts/vram-fit.py                            # 24 GB VRAM, 128K context, fp16 KV
    scripts/vram-fit.py --vram 48                  # 48 GB GPU
    scripts/vram-fit.py --context 32768            # short context
    scripts/vram-fit.py --kv-dtype fp8             # half KV footprint
    scripts/vram-fit.py --family gemma4            # filter by name substring
    scripts/vram-fit.py --fits-only                # hide rows that don't fit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

# Constants calibrated against vLLM launch behaviour on 24 GB GPU.
CUDA_GRAPH_GB = 1.0     # CUDA graphs + compiled kernels
ACTIVATIONS_GB = 0.5    # peak activation memory during forward
OVERHEAD_GB = CUDA_GRAPH_GB + ACTIVATIONS_GB

KV_BYTES = {"fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}

# Known architectures for model families we don't have config.json for.
# Keyed by substring match against the model name.
# Fields: layers, kv_heads, head_dim, k_eq_v (Gemma-style half-KV optimization).
_ARCH_FALLBACKS: list[tuple[str, dict]] = [
    # Gemma 4 E-series (effective params, smaller arch than the params suggest)
    ("gemma4:e2b", {"layers": 26, "kv_heads": 4, "head_dim": 256, "k_eq_v": True}),
    ("gemma4:e4b", {"layers": 30, "kv_heads": 4, "head_dim": 256, "k_eq_v": True}),
]


@dataclass
class Arch:
    layers: int
    kv_heads: int
    head_dim: int
    k_eq_v: bool = False
    source: str = ""

    def kv_per_token_bytes(self, kv_dtype: str) -> float:
        copies = 1 if self.k_eq_v else 2           # K and V, or just one when K=V
        return copies * self.layers * self.kv_heads * self.head_dim * KV_BYTES[kv_dtype]


def parse_size_gb(s: str) -> float:
    """Parse '16 GB', '7.2G', '23' → float GB."""
    s = s.strip().upper().rstrip("B").strip()
    if s.endswith("G"):
        s = s[:-1].strip()
    return float(s)


def arch_from_config(model_dir: Path) -> Arch | None:
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return None
    with cfg_path.open() as fh:
        c = json.load(fh)
    t = c.get("text_config", c)  # Gemma 4 nests; other HF models usually don't
    try:
        layers = int(t["num_hidden_layers"])
        kv_heads = int(t.get("num_key_value_heads", t["num_attention_heads"]))
        head_dim = int(
            t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
        )
    except (KeyError, ValueError, ZeroDivisionError):
        return None
    return Arch(
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        k_eq_v=bool(t.get("attention_k_eq_v", False)),
        source=f"config.json ({model_dir.name})",
    )


def arch_from_fallback_table(model_name: str) -> Arch | None:
    for pattern, arch in _ARCH_FALLBACKS:
        if pattern in model_name:
            return Arch(**arch, source=f"built-in table ({pattern})")
    return None


def _norm_base(name: str) -> str:
    """Collapse to a quantization/backend-agnostic base key.
    'gemma4:26b-a4b-it-q4_K_M' → 'gemma4 26b a4b'
    'Gemma-4-26B-A4B-it-NVFP4' → 'gemma 4 26b a4b'
    Further flattened to alnum-only so the two above compare equal enough.
    """
    s = name.lower()
    for suffix in (":q4_k_m", ":q8_0", ":q6_k", ":bf16", ":it",
                   "-it-nvfp4", "-nvfp4", "-it", "-instruct"):
        s = s.replace(suffix, "")
    # alnum-only so 'gemma4' and 'gemma-4-' match
    return "".join(ch for ch in s if ch.isalnum())


def arch_from_sibling(model_name: str, all_models: list[dict], vllm_dir: Path) -> Arch | None:
    """Ollama entries often share a base model with a downloaded vLLM
    sibling (same weights, different container format). Match by a
    normalized base key, then read that sibling's config.json."""
    base = _norm_base(model_name)
    if not base:
        return None
    for m in all_models:
        other = m["name"]
        if other == model_name:
            continue
        other_base = _norm_base(other)
        if base == other_base or base in other_base or other_base in base:
            arch = arch_from_config(vllm_dir / other)
            if arch:
                arch.source = f"inferred from {other} config.json"
                return arch
    return None


def resolve_arch(m: dict, all_models: list[dict], vllm_dir: Path) -> Arch | None:
    name = m["name"]
    # Preferred: inline arch from the generated catalog (real data
    # fetched at catalog-regen time from each repo's config.json).
    inline = m.get("arch")
    if isinstance(inline, dict):
        try:
            return Arch(
                layers=int(inline["layers"]),
                kv_heads=int(inline["kv_heads"]),
                head_dim=int(inline["head_dim"]),
                k_eq_v=bool(inline.get("k_eq_v", False)),
                source=m.get("arch_source", "models.yaml (inline)"),
            )
        except (KeyError, ValueError, TypeError):
            pass
    # Fallback: local config.json on disk.
    if {"vllm", "sglang"} & set(m.get("backend", [])):
        a = arch_from_config(vllm_dir / name)
        if a:
            return a
    # Fallback: sibling match.
    a = arch_from_sibling(name, all_models, vllm_dir)
    if a:
        return a
    # Last resort.
    return arch_from_fallback_table(name)


def heuristic_vram(weight_gb: float, context_tokens: int, kv_dtype: str) -> float:
    """Rough fallback: kv_bytes_per_token ≈ weight_gb * 6 KB (calibrated
    against Llama/Qwen dense models at fp16). Scales with kv dtype."""
    kv_bytes_per_tok = weight_gb * 6 * 1024 * (KV_BYTES[kv_dtype] / 2)
    kv_gb = (kv_bytes_per_tok * context_tokens) / (1024 ** 3)
    return weight_gb + kv_gb + OVERHEAD_GB


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vram", type=float, default=float(os.environ.get("GPU_MEMORY_GB", 24)))
    ap.add_argument("--context", type=int, default=int(os.environ.get("MAX_CONTEXT_LEN", 131072)))
    ap.add_argument("--kv-dtype", choices=list(KV_BYTES), default="fp16")
    ap.add_argument("--family", default="")
    ap.add_argument("--models-yaml", default="deploy/models.yaml")
    ap.add_argument("--vllm-dir", default=os.environ.get("VLLM_MODELS_DIR", "/var/cache/devai/vllm"))
    ap.add_argument("--fits-only", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.models_yaml)
    if not cfg_path.is_file():
        # Try relative to the script's repo root.
        alt = Path(__file__).resolve().parent.parent / "deploy" / "models.yaml"
        if alt.is_file():
            cfg_path = alt
        else:
            sys.exit(f"error: {args.models_yaml} not found")
    cfg = yaml.safe_load(cfg_path.read_text())
    models = cfg.get("models", [])
    vllm_dir = Path(args.vllm_dir)

    ctx_k = args.context // 1024
    print()
    print(f"  VRAM budget: {args.vram:g} GB  ·  Context: {ctx_k}K tokens  ·  "
          f"KV dtype: {args.kv_dtype}  ·  Overhead: {OVERHEAD_GB:g} GB (cuda graphs + activations)")
    print()
    print(f"  {'MODEL':<38s} {'BACKEND':<8s} {'WEIGHTS':>9s} {'KV':>9s} "
          f"{'TOTAL':>9s}  FIT  NOTE")
    print(f"  {'-'*108}")

    rows_shown = 0
    for m in models:
        name = m["name"]
        if args.family and args.family.lower() not in name.lower():
            continue
        try:
            weight_gb = parse_size_gb(m["size"])
        except (KeyError, ValueError):
            continue

        arch = resolve_arch(m, models, vllm_dir)
        if arch:
            kv_gb = (arch.kv_per_token_bytes(args.kv_dtype) * args.context) / (1024 ** 3)
            note = arch.source
        else:
            total_heuristic = heuristic_vram(weight_gb, args.context, args.kv_dtype)
            kv_gb = total_heuristic - weight_gb - OVERHEAD_GB
            note = "heuristic (no arch available)"

        total = weight_gb + kv_gb + OVERHEAD_GB

        for backend in m.get("backend", []):
            fits = total <= args.vram
            if args.fits_only and not fits:
                continue
            status = "✓" if fits else "✗"
            print(f"  {name:<38s} {backend:<8s} "
                  f"{weight_gb:>7.1f}G {kv_gb:>7.1f}G {total:>7.1f}G  "
                  f"{status:<3s}  {note}")
            rows_shown += 1

    if rows_shown == 0:
        print("  (no rows matched)")
    print()


if __name__ == "__main__":
    main()
