"""Unit tests for scripts/generate-catalog.py.

Network-free surface only: the GGUF tag-token derivation (`_gguf_tag_token`)
and a shape guard on the hand-maintained `ornith` family block in
scripts/model-families.yaml. The live HF / Ollama catalog build is exercised
by `make catalog-regen`, not here.

Stdlib unittest only. Run with:
    python3 -m unittest tests.python.test_generate_catalog
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = REPO_ROOT / "scripts" / "generate-catalog.py"
_FAMILIES_YAML = REPO_ROOT / "scripts" / "model-families.yaml"


def _load_module():
    """Load the hyphenated script as an importable module.

    Registering it in sys.modules is required so `dataclasses` can resolve
    the string annotations produced by `from __future__ import annotations`.
    """
    spec = importlib.util.spec_from_file_location("generate_catalog", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generate_catalog"] = mod
    spec.loader.exec_module(mod)
    return mod


gc = _load_module()


class TestGgufTagToken(unittest.TestCase):
    """`_gguf_tag_token` strips everything up to the rightmost size token."""

    def test_uppercase_dense_regression(self) -> None:
        # unsloth ships uppercase `27B` -- must keep working.
        self.assertEqual(
            gc._gguf_tag_token("Qwen3.5-27B-UD-Q3_K_XL.gguf"), "ud-q3_k_xl"
        )

    def test_uppercase_moe_active_expert_notation(self) -> None:
        # `A3B` / `A4B` (active-experts) is the optional-letter branch.
        self.assertEqual(
            gc._gguf_tag_token("Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf"), "ud-q3_k_xl"
        )
        self.assertEqual(
            gc._gguf_tag_token("gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf"),
            "it-ud-q3_k_xl",
        )

    def test_lowercase_size_token_stripped(self) -> None:
        # deepreinforce-ai Ornith ships lowercase `9b`. Before the [Bb] fix
        # the size token was never stripped and the whole stem leaked into
        # the tag (e.g. `ornith-1.0-9b-q4_k_m`).
        self.assertEqual(
            gc._gguf_tag_token("ornith-1.0-9b-Q4_K_M.gguf"), "q4_k_m"
        )
        self.assertEqual(
            gc._gguf_tag_token("ornith-1.0-9b-Q5_K_M.gguf"), "q5_k_m"
        )
        self.assertEqual(gc._gguf_tag_token("ornith-1.0-9b-Q6_K.gguf"), "q6_k")
        self.assertEqual(gc._gguf_tag_token("ornith-1.0-9b-Q8_0.gguf"), "q8_0")
        self.assertEqual(gc._gguf_tag_token("ornith-1.0-9b-bf16.gguf"), "bf16")

    def test_full_tag_composition(self) -> None:
        # The catalog row name is f"{family}:{tag_prefix}-{token}".
        token = gc._gguf_tag_token("ornith-1.0-9b-Q4_K_M.gguf")
        self.assertEqual(f"ornith:9b-{token}", "ornith:9b-q4_k_m")


def _ornith_family() -> dict:
    with _FAMILIES_YAML.open() as fh:
        data = yaml.safe_load(fh)
    for fam in data["families"]:
        if fam.get("name") == "ornith":
            return fam
    raise AssertionError("ornith family not found in model-families.yaml")


class TestOrnithFamily(unittest.TestCase):
    """Guard the hand-maintained 9B Ornith entry against accidental drift."""

    def test_arch_ref_and_thinking(self) -> None:
        fam = _ornith_family()
        self.assertEqual(fam["arch_ref"], "deepreinforce-ai/Ornith-1.0-9B")
        self.assertTrue(fam["thinking"])

    def test_hf_bf16_safetensors_row(self) -> None:
        fam = _ornith_family()
        self.assertIn("deepreinforce-ai/Ornith-1.0-9B", fam["hf_repos"])

    def test_gguf_repo_and_include_ladder(self) -> None:
        fam = _ornith_family()
        gguf = fam["gguf_repos"]
        self.assertEqual(len(gguf), 1)
        entry = gguf[0]
        self.assertEqual(entry["repo"], "deepreinforce-ai/Ornith-1.0-9B-GGUF")
        self.assertEqual(entry["tag_prefix"], "9b")
        self.assertEqual(
            entry["include"], ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "bf16"]
        )

    def test_parsers_match_qwen3_5_arch(self) -> None:
        # Model-card-prescribed parsers for the qwen3_5 arch.
        fam = _ornith_family()
        self.assertEqual(fam["parsers"]["vllm"]["reasoning"], "qwen3")
        self.assertEqual(fam["parsers"]["vllm"]["tool"], "qwen3_xml")
        self.assertEqual(fam["parsers"]["sglang"]["reasoning"], "qwen3")
        self.assertEqual(fam["parsers"]["sglang"]["tool"], "qwen")

    def test_no_speculative_mtp_block(self) -> None:
        # Intentionally MTP-less: the vocab-248320 draft lm_head would OOM
        # at load on 24G. Assert no mtp block sneaks onto the hf entry.
        fam = _ornith_family()
        for spec in fam["hf_repos"]:
            if isinstance(spec, dict):
                self.assertNotIn("mtp", spec)


if __name__ == "__main__":
    unittest.main()
