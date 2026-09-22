"""An NVFP4 body with int8 vocabulary tensors: the config must say so, per group.

Experiment 2026-09-21: apply HyperQwen's checkpoint preparation (int8 g128
embed_tokens / lm_head / mtp.*) to an NVFP4 checkpoint, to keep Blackwell's
native FP4 kernels for the body while freeing the ~3 GiB the bf16 vocabulary
tensors cost. HyperQwen's scripts convert the TENSORS correctly whatever the
body is, but they build each new config group by deep-copying the body's
group_0. For an int4 body that is harmless. For an NVFP4 body the copy keeps

    format: nvfp4-pack-quantized          (these tensors are pack-quantized)
    input_activations: {4-bit float ...}  (these tensors are weight-only)

so vLLM would look for FP4 global scales and activation quantization on
tensors that carry neither. vLLM 0.28 resolves the format PER GROUP
(compressed_tensors.py: "use the per-layer format if defined, otherwise use
global format"), so the fix is purely declarative: each int8 group gets the
definition that is already known to load -- the one HyperQwen's scripts leave
in a prepared int4 checkpoint, which serves here.

Pure JSON transformation; nothing is loaded, nothing touches a GPU.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "mixed_quant_groups", REPO_ROOT / "scripts" / "mixed_quant_groups.py")
mq = importlib.util.module_from_spec(_SPEC)
sys.modules["mixed_quant_groups"] = mq
_SPEC.loader.exec_module(mq)

_NVFP4_BODY = {
    "format": "nvfp4-pack-quantized",
    "input_activations": {"num_bits": 4, "type": "float", "dynamic": "local",
                          "group_size": 16, "strategy": "tensor_group"},
    "output_activations": None,
    "targets": ["Linear"],
    "weights": {"num_bits": 4, "type": "float", "strategy": "tensor_group",
                "group_size": 16, "symmetric": True, "dynamic": False,
                "scale_dtype": "torch.float8_e4m3fn", "zp_dtype": None},
}


def _cloned(bits: int, targets: list[str]) -> dict:
    """What HyperQwen's quant_heads_stream.py writes on an NVFP4 body."""
    g = copy.deepcopy(_NVFP4_BODY)
    g["targets"] = targets
    g["weights"].update(num_bits=bits, symmetric=True, zp_dtype=None,
                        group_size=128, strategy="group", type="int")
    return g


def _prepared_config(mtp_bits: int = 8) -> dict:
    return {
        "format": "nvfp4-pack-quantized",
        "quant_method": "compressed-tensors",
        "ignore": ["model.visual.blocks.0.attn.qkv", "mtp.layers.0.input_layernorm"],
        "config_groups": {
            "group_0": copy.deepcopy(_NVFP4_BODY),
            "group_1": _cloned(8, ["re:.*lm_head$"]),
            "group_2": _cloned(8, ["re:.*embed_tokens$"]),
            "group_3": _cloned(mtp_bits, ["re:^mtp\\..*"]),
        },
    }


class NormaliseTest(unittest.TestCase):
    def test_int8_groups_become_weight_only_pack_quantized(self) -> None:
        out = mq.normalise_int_groups(_prepared_config())
        for name in ("group_1", "group_2", "group_3"):
            g = out["config_groups"][name]
            self.assertEqual(g["format"], "pack-quantized", name)
            self.assertIsNone(g["input_activations"], name)
            self.assertEqual(g["weights"]["type"], "int", name)
            self.assertEqual(g["weights"]["strategy"], "group", name)
            self.assertEqual(g["weights"]["group_size"], 128, name)
            self.assertTrue(g["weights"]["symmetric"], name)
            self.assertIsNone(g["weights"]["scale_dtype"], name)

    def test_groups_cloned_by_the_per_tensor_scripts_are_recognised_too(self) -> None:
        # quant_lm_head.py / quant_embed.py / quant_mtp.py copy the body
        # group and change only num_bits + targets, so on an NVFP4 body the
        # int8 lm_head group still says `type: float`. The targets tell.
        cfg = _prepared_config()
        for g in ("group_1", "group_2", "group_3"):
            cfg["config_groups"][g]["weights"].update(type="float", strategy="tensor_group", group_size=16)
        out = mq.normalise_int_groups(cfg)
        for g in ("group_1", "group_2", "group_3"):
            self.assertEqual(out["config_groups"][g]["weights"]["type"], "int", g)
            self.assertEqual(out["config_groups"][g]["format"], "pack-quantized", g)

    def test_targets_and_bit_width_are_preserved(self) -> None:
        out = mq.normalise_int_groups(_prepared_config(mtp_bits=4))
        self.assertEqual(out["config_groups"]["group_1"]["targets"], ["re:.*lm_head$"])
        self.assertEqual(out["config_groups"]["group_3"]["targets"], ["re:^mtp\\..*"])
        self.assertEqual(out["config_groups"]["group_1"]["weights"]["num_bits"], 8)
        self.assertEqual(out["config_groups"]["group_3"]["weights"]["num_bits"], 4)

    def test_the_nvfp4_body_group_is_left_exactly_as_it_was(self) -> None:
        cfg = _prepared_config()
        out = mq.normalise_int_groups(cfg)
        self.assertEqual(out["config_groups"]["group_0"], _NVFP4_BODY)
        self.assertEqual(out["format"], "nvfp4-pack-quantized")
        self.assertEqual(out["ignore"], cfg["ignore"])

    def test_the_input_is_not_mutated(self) -> None:
        cfg = _prepared_config()
        before = copy.deepcopy(cfg)
        mq.normalise_int_groups(cfg)
        self.assertEqual(cfg, before)

    def test_it_is_idempotent(self) -> None:
        once = mq.normalise_int_groups(_prepared_config())
        self.assertEqual(mq.normalise_int_groups(once), once)

    def test_an_already_correct_int4_checkpoint_is_unchanged_in_meaning(self) -> None:
        # The prepared AutoRound config: every group already pack-quantized.
        body = {"format": "pack-quantized", "input_activations": None,
                "output_activations": None, "targets": ["Linear"],
                "weights": dict(mq.INT_GROUP_WEIGHTS, num_bits=4)}
        cfg = {"format": "pack-quantized", "ignore": [],
               "config_groups": {"group_0": body,
                                 "group_1": dict(copy.deepcopy(body), targets=["re:.*lm_head$"])}}
        out = mq.normalise_int_groups(cfg)
        self.assertEqual(out["config_groups"]["group_0"], body)
        self.assertEqual(out["config_groups"]["group_1"]["format"], "pack-quantized")

    def test_nothing_to_normalise_is_an_error(self) -> None:
        # Running this on an UNPREPARED checkpoint means the conversion step
        # was skipped; saying "done" would hide that.
        cfg = {"format": "nvfp4-pack-quantized", "ignore": ["lm_head"],
               "config_groups": {"group_0": copy.deepcopy(_NVFP4_BODY)}}
        with self.assertRaises(ValueError) as cm:
            mq.normalise_int_groups(cfg)
        self.assertIn("no int", str(cm.exception))

    def test_a_vocab_tensor_still_in_the_ignore_list_is_an_error(self) -> None:
        # `ignore` wins over every group: the int8 lm_head would be loaded as
        # if it were unquantized.
        cfg = _prepared_config()
        cfg["ignore"].append("lm_head")
        with self.assertRaises(ValueError) as cm:
            mq.normalise_int_groups(cfg)
        self.assertIn("lm_head", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
