#!/usr/bin/env python3
"""Declare int8 vocabulary/MTP tensors correctly inside a non-int4 checkpoint.

HyperQwen's preparation scripts (github.com/syv-ai/HyperQwen, prepare/) convert
embed_tokens, lm_head and the mtp.* module to int8 group-128 pack-quantized
tensors. The tensor conversion is independent of the body's format, but each
new config group is built by deep-copying the body's `group_0`. On an NVFP4
body that copy keeps `format: nvfp4-pack-quantized` and the 4-bit float
`input_activations` block -- both wrong for int8 weight-only tensors.

vLLM 0.28 resolves the quantization format PER GROUP, so the repair is purely
declarative: every integer group-quantized group is rewritten to the
definition that is known to load (what the same scripts leave in a prepared
int4 checkpoint), keeping its targets and bit width. The body group, the
top-level format and the ignore list are left alone.

Usage: mixed_quant_groups.py <model-dir>
Rewrites <model-dir>/config.json; the previous file is kept as
config.json.bak-mixed.
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

INT_GROUP_FORMAT = "pack-quantized"

# Weight scheme of an int group, as written into a prepared int4 checkpoint
# that vLLM 0.28 loads and serves. `num_bits` is filled in per group.
INT_GROUP_WEIGHTS = {
    "actorder": None,
    "block_structure": None,
    "dynamic": False,
    "group_size": 128,
    "num_bits": 8,
    "observer": "memoryless_minmax",
    "observer_kwargs": {},
    "scale_dtype": None,
    "strategy": "group",
    "symmetric": True,
    "type": "int",
    "zp_dtype": None,
}

# Tensors the preparation quantizes. If one is still named in `ignore`, vLLM
# loads it as unquantized and the int8 tensors on disk do not match.
_MUST_NOT_BE_IGNORED = ("lm_head",)


# Targets the preparation writes int8 tensors for. Upstream's streaming
# script marks its groups `type: int, strategy: group`; the three per-tensor
# scripts clone the body group and change only `num_bits` and `targets`, so
# on a float body their groups still SAY float. The targets are the reliable
# signal: these three are int8 group-128 whatever the group claims.
_PREPARED_TARGETS = (".*lm_head$", ".*embed_tokens$", "^mtp\\..*")


def _is_int_group(group: dict) -> bool:
    weights = group.get("weights") or {}
    if weights.get("type") == "int" and weights.get("strategy") == "group":
        return True
    targets = [t.replace("re:", "", 1) for t in (group.get("targets") or [])]
    return bool(targets) and all(t in _PREPARED_TARGETS for t in targets)


def normalise_int_groups(quant_config: dict) -> dict:
    """Return a copy of `quant_config` with every int group made weight-only
    pack-quantized. Raises ValueError when there is nothing to normalise or
    when a quantized tensor is still in the ignore list."""
    out = copy.deepcopy(quant_config)
    groups = out.get("config_groups") or {}
    int_groups = [name for name, g in groups.items() if _is_int_group(g)]
    if not int_groups:
        raise ValueError(
            "no int group-quantized config group found -- was the checkpoint "
            "prepared (quant_heads_stream.py / quant_lm_head.py ...) first?")
    ignored = [t for t in _MUST_NOT_BE_IGNORED if t in (out.get("ignore") or [])]
    if ignored:
        raise ValueError(
            f"{ignored} still in quantization_config.ignore: vLLM would load "
            "them as unquantized")
    for name in int_groups:
        old = groups[name]
        groups[name] = {
            "format": INT_GROUP_FORMAT,
            "input_activations": None,
            "output_activations": None,
            "targets": list(old["targets"]),
            "weights": dict(INT_GROUP_WEIGHTS, num_bits=old["weights"]["num_bits"]),
        }
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    cfg_path = Path(argv[1]) / "config.json"
    config = json.loads(cfg_path.read_text())
    try:
        fixed = normalise_int_groups(config["quantization_config"])
    except (KeyError, ValueError) as e:
        print(f"error: {cfg_path}: {e}", file=sys.stderr)
        return 1
    shutil.copy(cfg_path, cfg_path.with_name("config.json.bak-mixed"))
    cfg_path.write_text(json.dumps(dict(config, quantization_config=fixed), indent=2))
    for name, group in fixed["config_groups"].items():
        w = group["weights"]
        print(f"  {name}: {group.get('format')} {w['type']}{w['num_bits']} "
              f"targets={group['targets']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
