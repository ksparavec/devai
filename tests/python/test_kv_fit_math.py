"""Backend-aware KV-cache cost in the fit planner.

scripts/select-models.py used to cost every candidate's KV cache at fp16
(2 bytes/element) regardless of which engine would serve it. That is not
what the router launches: containerRecreate passes --kv-cache-dtype
resolved from the probe cell, and an unstamped cell decodes to fp8 (see
gpu-arbiter resolveKVCacheType and docs/backends.md). On this host 20 of
21 vLLM cells and 20 of 20 SGLang cells are unstamped, so nearly every HF
row was costed at twice the KV it is actually served with -- which pushes
rows over the VRAM budget and quietly removes them from the download
candidate set, so they are never downloaded and never probed.

Ollama is the opposite case: it defaults to fp16 KV and only uses q8_0 on
the specific tiers whose probe cell says so, so ollama-only rows must
keep the 2-byte assumption.

Stdlib unittest only; no container, no network, no GPU.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = REPO_ROOT / "scripts" / "select-models.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("select_models", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["select_models"] = mod
    spec.loader.exec_module(mod)
    return mod


sm = _load_module()

# A GQA arch with round numbers so the KV term is easy to reason about:
# 2 copies * 32 layers * 8 kv_heads * 128 head_dim = 65536 elements/token.
# At 1 byte that is 64 KiB/token; at 2 bytes, 128 KiB/token.
_ARCH = {"layers": 32, "kv_heads": 8, "head_dim": 128, "k_eq_v": False}
_ELEMS_PER_TOKEN = 2 * 32 * 8 * 128


def _row(backends: list[str], size: str = "8.0G") -> dict:
    return {"name": "test-model", "size": size, "arch": _ARCH,
            "backend": backends}


class KVBytesTableTest(unittest.TestCase):
    def test_quantized_ollama_types_are_present(self):
        """q8_0/q4_0 are what the Ollama probe cells actually stamp."""
        for dtype in ("q8_0", "q4_0"):
            self.assertIn(dtype, sm.KV_BYTES)

    def test_auto_is_two_bytes(self):
        """vLLM/SGLang spell unquantized KV 'auto'; it is not free."""
        self.assertEqual(sm.KV_BYTES["auto"], 2.0)

    def test_relative_sizes(self):
        self.assertEqual(sm.KV_BYTES["fp8"], sm.KV_BYTES["fp16"] / 2)
        self.assertEqual(sm.KV_BYTES["q4_0"], sm.KV_BYTES["q8_0"] / 2)


class ResolveKVDtypeTest(unittest.TestCase):
    def test_hf_rows_default_to_fp8(self):
        for backends in (["vllm"], ["sglang"], ["vllm", "sglang"]):
            self.assertEqual(
                sm.resolve_kv_dtype(_row(backends), sm.KV_DTYPE_PER_BACKEND),
                "fp8", f"backends={backends}")

    def test_ollama_only_rows_default_to_fp16(self):
        self.assertEqual(
            sm.resolve_kv_dtype(_row(["ollama"]), sm.KV_DTYPE_PER_BACKEND),
            "fp16")

    def test_mixed_row_uses_the_hf_dtype(self):
        """A row both Ollama and vLLM can serve is costed at the vLLM
        dtype, matching how vram_breakdown already picks vLLM overhead."""
        self.assertEqual(
            sm.resolve_kv_dtype(_row(["ollama", "vllm"]), sm.KV_DTYPE_PER_BACKEND),
            "fp8")

    def test_explicit_dtype_overrides_everything(self):
        """--kv-dtype is an operator override and must win for every row."""
        for backends in (["vllm"], ["ollama"]):
            self.assertEqual(
                sm.resolve_kv_dtype(_row(backends), "q4_0"), "q4_0")


class VramBreakdownTest(unittest.TestCase):
    def test_hf_kv_is_half_of_ollama_kv_for_the_same_arch(self):
        ctx = 131072
        hf = sm.vram_breakdown(_row(["vllm"]), ctx, sm.KV_DTYPE_PER_BACKEND)
        oll = sm.vram_breakdown(_row(["ollama"]), ctx, sm.KV_DTYPE_PER_BACKEND)
        self.assertAlmostEqual(hf["kv_gb"] * 2, oll["kv_gb"], places=1)

    def test_kv_gb_matches_the_hand_computed_value(self):
        ctx = 131072
        got = sm.vram_breakdown(_row(["vllm"]), ctx, sm.KV_DTYPE_PER_BACKEND)
        want = (_ELEMS_PER_TOKEN * 1.0 * ctx) / (1024 ** 3)
        self.assertAlmostEqual(got["kv_gb"], round(want, 2), places=2)

    def test_reported_dtype_is_the_one_costed_not_the_flag(self):
        """The breakdown is printed to the operator, so it must name the
        dtype actually used rather than echo the sentinel back."""
        out = sm.vram_breakdown(_row(["vllm"]), 32768, sm.KV_DTYPE_PER_BACKEND)
        self.assertEqual(out["kv_dtype"], "fp8")
        self.assertNotEqual(out["kv_dtype"], sm.KV_DTYPE_PER_BACKEND)

    def test_explicit_fp16_reproduces_the_old_behaviour(self):
        """Back-compat: the previous default is still reachable and still
        produces the previous numbers."""
        ctx = 65536
        got = sm.vram_breakdown(_row(["vllm"]), ctx, "fp16")
        want = (_ELEMS_PER_TOKEN * 2.0 * ctx) / (1024 ** 3)
        self.assertAlmostEqual(got["kv_gb"], round(want, 2), places=2)
        self.assertEqual(got["kv_dtype"], "fp16")

    def test_archless_row_still_uses_the_conservative_fallback(self):
        """No arch means no KV computation; the 256 KB/token worst case
        must not silently start scaling with dtype."""
        row = {"name": "x", "size": "8.0G", "backend": ["vllm"]}
        a = sm.vram_breakdown(row, 32768, sm.KV_DTYPE_PER_BACKEND)
        b = sm.vram_breakdown(row, 32768, "fp16")
        self.assertEqual(a["kv_gb"], b["kv_gb"])


class LabelTest(unittest.TestCase):
    def test_sentinel_renders_readably(self):
        label = sm.kv_dtype_label(sm.KV_DTYPE_PER_BACKEND)
        self.assertIn("fp8", label)
        self.assertIn("fp16", label)

    def test_explicit_dtype_renders_verbatim(self):
        self.assertEqual(sm.kv_dtype_label("q8_0"), "q8_0")


class PickerParityTest(unittest.TestCase):
    """The picker keeps its own copy of this formula for the '*' cells.
    It is only ever used on the HF path, so it must use the HF dtype."""

    def test_picker_hf_kv_matches_select_models_fp8(self):
        spec = importlib.util.spec_from_file_location(
            "model_picker", REPO_ROOT / "scripts" / "model-picker.py")
        mp = importlib.util.module_from_spec(spec)
        sys.modules["model_picker"] = mp
        spec.loader.exec_module(mp)

        ctx = 131072
        picker_kv = mp._hf_kv_gb(_ARCH, ctx)
        planner_kv = sm.vram_breakdown(
            _row(["vllm"]), ctx, sm.KV_DTYPE_PER_BACKEND)["kv_gb"]
        self.assertAlmostEqual(picker_kv, planner_kv, places=1)


if __name__ == "__main__":
    unittest.main()
