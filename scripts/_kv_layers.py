"""How many of a model's layers actually hold a KV cache.

The fit formula costs KV per layer, and `num_hidden_layers` is the wrong
multiplier for hybrid architectures: linear-attention, Mamba and MLP-only
layers keep constant-size state (or none), not a per-token cache. Counting
them overstated KV 4x for Qwen3_5ForConditionalGeneration and 14x for
NemotronHForCausalLM -- enough to classify fitting models as too_big.

Two upstream spellings are recognised, both taken from real config.json
files of checkpoints served here:

  layer_types              list[str], one entry per layer (Qwen3.5/3.6/3.8,
                           Gemma 4, gpt-oss)
  hybrid_override_pattern  str, one char per layer; `*` = attention,
                           `M` = Mamba, `-` = MLP, `E` = MoE expert
                           (Nemotron-H). Counted as "everything except the
                           known KV-free chars", not as "only *".

Sliding-window layers DO hold a KV cache -- a bounded one -- so they are
counted. Costing them as full is conservative and a separate correction.

Anything unrecognised or internally inconsistent falls back to
`num_hidden_layers`, i.e. the old behaviour: this is upstream metadata, and
a wrong guess in the cheap direction would make every context look free.

Imported by generate-catalog.py and vram-fit.py. NOT by model-picker.py,
which runs bind-mounted inside older images and must not grow imports (see
tests/python/test_picker_container_imports.py); the picker only reads the
`kv_layers` value this module produces, from the catalog arch block.
"""

from __future__ import annotations

# What keeps no per-token KV cache, in each spelling. Both are deliberately
# DENYLISTS: anything not named here is counted, never silently dropped, so
# a layer kind this file has not met yet errs toward "holds KV".
KV_FREE_LAYER_TYPES = frozenset({"linear_attention"})
KV_FREE_PATTERN_CHARS = frozenset("M-E")   # Nemotron-H: Mamba, MLP, MoE expert


def kv_layer_count(text_cfg: dict) -> int:
    """Layers holding a KV cache, given a (text_)config dict.

    Returns `num_hidden_layers` when the config declares no per-layer
    layout, or declares one that does not describe exactly that many layers.
    """
    layers = int(text_cfg["num_hidden_layers"])

    layer_types = text_cfg.get("layer_types")
    if isinstance(layer_types, list) and len(layer_types) == layers:
        counted = sum(1 for t in layer_types if t not in KV_FREE_LAYER_TYPES)
        return counted if counted > 0 else layers

    pattern = text_cfg.get("hybrid_override_pattern")
    if isinstance(pattern, str) and len(pattern) == layers:
        counted = sum(1 for c in pattern if c not in KV_FREE_PATTERN_CHARS)
        return counted if counted > 0 else layers

    return layers
