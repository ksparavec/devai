"""KV cache is costed only on the layers that actually hold one.

The fit formula multiplied the per-token KV cost by `num_hidden_layers`.
That is wrong for hybrid architectures, where most layers keep no KV
cache at all:

  - Qwen3_5ForConditionalGeneration (Qwen3.5 / 3.6 / 3.8, Ornith) declares
    `layer_types`; 3 layers in 4 are `linear_attention` and hold constant
    recurrent state, not a per-token cache. Qwen3.8-27B: 64 layers, 16 with
    KV -- a 4x overcount.
  - NemotronHForCausalLM declares `hybrid_override_pattern`, where only `*`
    is an attention layer (`M` = Mamba, `-` = MLP). Nemotron-Nano-9B-v2:
    56 layers, 4 with KV -- a 14x overcount.

The overcount was not cosmetic. It pushed rows past the VRAM budget, so
`make model-fit FAMILY=qwen3.8` called all 14 rows "too large", and an
enumerating `--download` then wrote too_big verdicts into the exclusion
ledger for models that fit -- hiding them from the probe, which is the
only thing that could have proved the estimate wrong.

Sliding-window layers are deliberately still counted: they DO hold a KV
cache, merely a bounded one. Treating them as full is the conservative
status quo and a separate correction.

Fixture values are copied from the real config.json files under
/var/cache/devai/vllm on 2026-09-19.

Stdlib unittest only; no container, no network, no GPU.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import _kv_layers  # noqa: E402


def _load(filename: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, _SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


gc = _load("generate-catalog.py", "generate_catalog_kvl")
sm = _load("select-models.py", "select_models_kvl")
mp = _load("model-picker.py", "model_picker_kvl")
vf = _load("vram-fit.py", "vram_fit_kvl")

# Qwen3.8-27B text_config: 64 layers, every 4th is full attention.
_QWEN38_TYPES = (["linear_attention"] * 3 + ["full_attention"]) * 16
_QWEN38_CFG = {
    "num_hidden_layers": 64, "num_attention_heads": 24,
    "num_key_value_heads": 4, "head_dim": 256, "hidden_size": 5120,
    "layer_types": _QWEN38_TYPES,
}
# NVIDIA-Nemotron-Nano-9B-v2: 56 entries, 4 of them `*` (attention).
_NEMOTRON_PATTERN = "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"
_NEMOTRON_CFG = {
    "num_hidden_layers": 56, "num_attention_heads": 40,
    "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 4480,
    "hybrid_override_pattern": _NEMOTRON_PATTERN,
}
_DENSE_CFG = {
    "num_hidden_layers": 32, "num_attention_heads": 32,
    "num_key_value_heads": 8, "hidden_size": 4096,
}


class KvLayerCountTest(unittest.TestCase):
    def test_dense_model_counts_every_layer(self):
        self.assertEqual(_kv_layers.kv_layer_count(_DENSE_CFG), 32)

    def test_linear_attention_layers_hold_no_kv(self):
        self.assertEqual(_kv_layers.kv_layer_count(_QWEN38_CFG), 16)

    def test_sliding_window_layers_still_count(self):
        # gpt-oss-20b: a sliding layer holds a bounded cache, not none.
        cfg = {"num_hidden_layers": 24,
               "layer_types": ["sliding_attention", "full_attention"] * 12}
        self.assertEqual(_kv_layers.kv_layer_count(cfg), 24)

    def test_unknown_layer_type_is_counted_conservatively(self):
        cfg = {"num_hidden_layers": 4,
               "layer_types": ["full_attention", "some_future_type"] * 2}
        self.assertEqual(_kv_layers.kv_layer_count(cfg), 4)

    def test_nemotron_pattern_counts_only_attention_entries(self):
        self.assertEqual(_kv_layers.kv_layer_count(_NEMOTRON_CFG), 4)

    def test_nemotron_moe_expert_layers_hold_no_kv(self):
        # nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4, fetched 2026-09-19:
        # `E` is a MoE expert (feed-forward) layer. 52 layers, 6 attention.
        cfg = {"num_hidden_layers": 52, "hybrid_override_pattern":
               "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"}
        self.assertEqual(_kv_layers.kv_layer_count(cfg), 6)

    def test_unknown_pattern_char_is_counted_conservatively(self):
        # Same rule as an unknown layer_types entry: only the chars KNOWN to
        # be KV-free are dropped. Counting just `*` would silently treat a
        # future attention variant as free -- the dangerous direction.
        cfg = {"num_hidden_layers": 7, "hybrid_override_pattern": "M-*S*-M"}
        self.assertEqual(_kv_layers.kv_layer_count(cfg), 3)

    def test_all_kv_free_pattern_falls_back_to_depth(self):
        cfg = {"num_hidden_layers": 6, "hybrid_override_pattern": "M-M-MM"}
        self.assertEqual(_kv_layers.kv_layer_count(cfg), 6)

    def test_pattern_disagreeing_with_depth_is_ignored(self):
        cfg = {"num_hidden_layers": 56, "hybrid_override_pattern": "M*M-"}
        self.assertEqual(_kv_layers.kv_layer_count(cfg), 56)

    def test_layer_types_disagreeing_with_depth_is_ignored(self):
        cfg = {"num_hidden_layers": 64,
               "layer_types": ["linear_attention", "full_attention"]}
        self.assertEqual(_kv_layers.kv_layer_count(cfg), 64)

    def test_zero_kv_layers_falls_back_to_depth(self):
        # Never let malformed upstream metadata zero the KV term: that would
        # make every context look free.
        cfg = {"num_hidden_layers": 8, "layer_types": ["linear_attention"] * 8}
        self.assertEqual(_kv_layers.kv_layer_count(cfg), 8)


class CatalogArchTest(unittest.TestCase):
    def test_hybrid_arch_records_kv_layers(self):
        arch = gc.arch_from_config({"text_config": _QWEN38_CFG}, "src")
        self.assertEqual(arch.layers, 64)
        self.assertEqual(arch.kv_layers, 16)

    def test_hybrid_arch_yaml_carries_kv_layers(self):
        arch = gc.arch_from_config({"text_config": _QWEN38_CFG}, "src")
        self.assertIn("layers: 64", arch.to_yaml())
        self.assertIn("kv_layers: 16", arch.to_yaml())

    def test_dense_arch_yaml_is_unchanged(self):
        # Dense rows must not churn in deploy/models.yaml.
        arch = gc.arch_from_config(_DENSE_CFG, "src")
        self.assertEqual(arch.kv_layers, 32)
        self.assertNotIn("kv_layers", arch.to_yaml())


class ReaderFormulaTest(unittest.TestCase):
    _HYBRID = {"layers": 64, "kv_layers": 16, "kv_heads": 4,
               "head_dim": 256, "k_eq_v": False}
    _LEGACY = {"layers": 64, "kv_heads": 4, "head_dim": 256, "k_eq_v": False}

    def test_select_models_costs_only_kv_layers(self):
        # 2 copies * 16 layers * 4 heads * 256 dim * 1 byte = 32 KiB/token.
        self.assertEqual(sm.kv_per_token_bytes(self._HYBRID, "fp8"), 32768)

    def test_select_models_without_kv_layers_keeps_old_behaviour(self):
        self.assertEqual(sm.kv_per_token_bytes(self._LEGACY, "fp8"), 131072)

    def test_picker_costs_only_kv_layers(self):
        # 32 KiB/token * 131072 tokens = 4 GiB exactly.
        self.assertAlmostEqual(mp._hf_kv_gb(self._HYBRID, 131072), 4.0)

    def test_picker_without_kv_layers_keeps_old_behaviour(self):
        self.assertAlmostEqual(mp._hf_kv_gb(self._LEGACY, 131072), 16.0)

    def test_vram_fit_costs_only_kv_layers(self):
        arch = vf.Arch(layers=64, kv_heads=4, head_dim=256, kv_layers=16)
        self.assertEqual(arch.kv_per_token_bytes("fp8"), 32768)

    def test_vram_fit_without_kv_layers_keeps_old_behaviour(self):
        arch = vf.Arch(layers=64, kv_heads=4, head_dim=256)
        self.assertEqual(arch.kv_per_token_bytes("fp8"), 131072)


class Qwen38FitsTest(unittest.TestCase):
    """The regression that started this: the row must not read as too big."""

    def test_nvfp4_27b_fits_24g_at_32k_on_vllm(self):
        row = {"name": "Qwen3.8-27B-MTP-NVFP4", "size": "19.1G",
               "backend": ["vllm"],
               "arch": {"layers": 64, "kv_layers": 16, "kv_heads": 4,
                        "head_dim": 256, "k_eq_v": False}}
        dtype = sm.resolve_kv_dtype(row, sm.KV_DTYPE_PER_BACKEND)
        self.assertLessEqual(sm.estimate_total_gb(row, 32768, dtype), 24.0)


if __name__ == "__main__":
    unittest.main()
