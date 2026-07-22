"""Picker FORMAT column derives a short quant label from config.json.

Regression: an in-house llm-compressor NVFP4 checkpoint sets
`quantization_config.quant_method = "compressed-tensors"` (the LIBRARY name)
and `quantization_config.format = "nvfp4-pack-quantized"` (the real format).
The picker must show `NVFP4` (<=6 chars), not `COMPRESSED-TENSORS`.

NVIDIA ModelOpt checkpoints instead set `quant_algo` directly, and unquantized
checkpoints fall back to the name token / dtype.

Stdlib unittest only. Run with:
    python3 -m unittest tests.python.test_model_picker_format
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_picker():
    spec = importlib.util.spec_from_file_location(
        "_picker_under_test_format",
        str(REPO_ROOT / "scripts" / "model-picker.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


PICKER = _load_picker()


class TestHfFormatLabel(unittest.TestCase):
    def _label(self, name: str, cfg: dict) -> str:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / name
            d.mkdir()
            (d / "config.json").write_text(json.dumps(cfg))
            return PICKER._hf_format_label(d)

    def test_compressed_tensors_nvfp4_shows_nvfp4(self) -> None:
        # The reported bug: quant_method is the library, format is the format.
        lbl = self._label(
            "Ornith-1.0-9B-NVFP4",
            {"quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "nvfp4-pack-quantized",
            }},
        )
        self.assertEqual(lbl, "NVFP4")
        self.assertLessEqual(len(lbl), 6)

    def test_modelopt_quant_algo_wins(self) -> None:
        lbl = self._label(
            "Qwen3-8B-NVFP4",
            {"quantization_config": {"quant_algo": "NVFP4",
                                     "quant_method": "modelopt"}},
        )
        self.assertEqual(lbl, "NVFP4")

    def test_compressed_tensors_unknown_format_falls_back_to_name(self) -> None:
        # format="float-quantized" carries no named token; the FP8 in the dir
        # name is the fallback source.
        lbl = self._label(
            "SomeModel-FP8",
            {"quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "float-quantized",
            }},
        )
        self.assertEqual(lbl, "FP8")

    def test_plain_bf16_dtype(self) -> None:
        lbl = self._label("Ornith-1.0-9B", {"torch_dtype": "bfloat16"})
        self.assertEqual(lbl, "BF16")
        self.assertLessEqual(len(lbl), 6)


if __name__ == "__main__":
    unittest.main()
