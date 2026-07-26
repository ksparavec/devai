"""Unit tests for scripts/catalog-discover.py (read-only lineage discovery).

Covers the deterministic, network-free surface: version + family-name
parsing, repo version extraction (version vs size), the structural
line-membership filter, lineage grouping, discover-block parsing and its
override semantics, candidate classification, and a fully network-stubbed
run() so the end-to-end pipeline is exercised offline. The live HF /
Ollama queries are covered by `make catalog-discover`, not here.

Stdlib unittest only. Run with:
    python3 -m unittest tests.python.test_catalog_discover
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = REPO_ROOT / "scripts" / "catalog-discover.py"


def _load_module():
    """Load the hyphenated script as an importable module.

    Registering it in sys.modules is required so `dataclasses` can resolve
    the string annotations produced by `from __future__ import annotations`.
    """
    spec = importlib.util.spec_from_file_location("catalog_discover", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["catalog_discover"] = mod
    spec.loader.exec_module(mod)
    return mod


cd = _load_module()


class TestVersionParsing(unittest.TestCase):
    def test_parse_version_basic(self) -> None:
        self.assertEqual(cd.parse_version("3"), (3,))
        self.assertEqual(cd.parse_version("3.5"), (3, 5))
        self.assertEqual(cd.parse_version("3.10"), (3, 10))

    def test_parse_version_rejects_non_numeric(self) -> None:
        self.assertIsNone(cd.parse_version("r1"))
        self.assertIsNone(cd.parse_version("v2"))
        self.assertIsNone(cd.parse_version(""))

    def test_tuple_ordering_beats_float(self) -> None:
        # The whole reason for tuples: 3.10 must sort ABOVE 3.5, not equal 3.1.
        self.assertTrue((3,) < (3, 5) < (3, 10) < (4,))
        self.assertGreater(cd.parse_version("3.10"), cd.parse_version("3.5"))

    def test_version_str_roundtrip(self) -> None:
        self.assertEqual(cd.version_str((3, 5)), "3.5")
        self.assertEqual(cd.version_str((4,)), "4")


class TestFamilyLineageParsing(unittest.TestCase):
    CASES = {
        "qwen3": ("qwen", (3,), ""),
        "qwen3.5": ("qwen", (3, 5), ""),
        "qwen3.6": ("qwen", (3, 6), ""),
        "qwen3-coder": ("qwen", (3,), "-coder"),
        "gemma4": ("gemma", (4,), ""),
        "llama3.1": ("llama", (3, 1), ""),
        "llama3.2": ("llama", (3, 2), ""),
        "nemotron-3-nano": ("nemotron", (3,), "-nano"),
        # Irregular -> no derivable numeric version (need a discover: block).
        "gpt-oss": ("gpt-oss", None, ""),
        "deepseek-r1-distill": ("deepseek-r1-distill", None, ""),
        "nemotron-nano-v2": ("nemotron-nano-v2", None, ""),
        "diffusiongemma": ("diffusiongemma", None, ""),
    }

    def test_all_cases(self) -> None:
        for name, expected in self.CASES.items():
            with self.subTest(name=name):
                self.assertEqual(cd.parse_family_lineage(name), expected)

    def test_lineage_key_groups_same_brand_suffix(self) -> None:
        k1 = cd.lineage_key("qwen", "")
        k2 = cd.lineage_key("qwen", "-coder")
        self.assertEqual(k1, cd.lineage_key("qwen", ""))
        self.assertNotEqual(k1, k2)


class TestRepoVersionExtraction(unittest.TestCase):
    CASES = {
        ("Qwen3.6-35B-A3B-NVFP4", "qwen"): (3, 6),
        ("Qwen3-8B-NVFP4", "qwen"): (3,),
        ("Qwen3.5-122B-A10B-NVFP4", "qwen"): (3, 5),
        ("Llama-3.1-8B-Instruct-NVFP4", "llama"): (3, 1),
        ("gemma-4-26b-a4b-it", "gemma"): (4,),
        ("NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4", "nemotron"): (3,),
        # A size must never be mistaken for a version.
        ("gpt-oss-20b", "gpt-oss"): None,
        ("diffusiongemma-26B-A4B-it-NVFP4", "diffusiongemma"): None,
        ("NVIDIA-Nemotron-Nano-9B-v2-NVFP4", "nemotron"): None,
    }

    def test_all_cases(self) -> None:
        for (base, brand), expected in self.CASES.items():
            with self.subTest(base=base):
                self.assertEqual(cd.extract_repo_version(base, brand), expected)


class TestSizeToken(unittest.TestCase):
    def test_size_tokens(self) -> None:
        for tok in ("8b", "27b", "235b", "a3b", "a10b", "0.6b", "e2b"):
            self.assertTrue(cd.is_size_token(tok), tok)

    def test_non_size_tokens(self) -> None:
        for tok in ("vl", "next", "coder", "instruct", "nvfp4", "3"):
            self.assertFalse(cd.is_size_token(tok), tok)


class TestCleanLineageMember(unittest.TestCase):
    def _lin(self, brand: str, suffix: str = "") -> object:
        return cd.Lineage(key=f"{brand}|{suffix}", brand=brand, suffix=suffix)

    def test_accepts_real_members(self) -> None:
        qwen = self._lin("qwen")
        for base in ("Qwen3-8B-NVFP4", "Qwen3.6-35B-A3B-NVFP4",
                     "Qwen3.5-122B-A10B-NVFP4", "Qwen3.6-35B-A3B-MTP-GGUF"):
            self.assertTrue(cd.is_clean_lineage_member(base, qwen), base)
        self.assertTrue(cd.is_clean_lineage_member(
            "Llama-3.1-8B-Instruct-NVFP4", self._lin("llama")))
        self.assertTrue(cd.is_clean_lineage_member(
            "gemma-4-e2b-it", self._lin("gemma")))
        self.assertTrue(cd.is_clean_lineage_member(
            "Qwen3-Coder-30B-A3B-Instruct-FP4", self._lin("qwen", "-coder")))

    def test_accepts_decimal_sizes(self) -> None:
        # The tail tokenizer must keep '0.6' glued so the size matches.
        qwen = self._lin("qwen")
        for base in ("Qwen3-0.6B-NVFP4", "Qwen3-1.7B-NVFP4",
                     "Qwen3-1.5B", "Qwen3-0.5B-Instruct"):
            self.assertTrue(cd.is_clean_lineage_member(base, qwen), base)

    def test_accepts_org_self_prefix(self) -> None:
        # A single prefix token related to the author is an org prefix.
        self.assertTrue(cd.is_clean_lineage_member(
            "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
            self._lin("nemotron", "-nano"), author="nvidia"))
        self.assertTrue(cd.is_clean_lineage_member(
            "Meta-Llama-3.1-8B-Instruct", self._lin("llama"),
            author="meta-llama"))

    def test_rejects_foreign_product_lines(self) -> None:
        qwen = self._lin("qwen")
        for base in (
            "Qwen3-VL-235B-A22B-Instruct-NVFP4",   # vision
            "Qwen3-Next-80B-A3B-Instruct-NVFP4",   # different arch
            "Qwen3-Embedding-8B",                  # embedding model
            "Qwen3-Guard-8B",                      # guard model
            "Qwen3-Coder-30B-A3B-Instruct-FP4",    # sibling lineage, not base
            "DeepSeek-R1-0528-Qwen3-8B-GGUF",      # distill (multi-token prefix)
            "KVzap-mlp-Qwen3-8B",                  # research artifact
        ):
            self.assertFalse(cd.is_clean_lineage_member(base, qwen), base)

    def test_rejects_finetune_brand_even_from_trusted_author(self) -> None:
        # A single foreign brand word (not the author) must NOT pass as an
        # org prefix, even when the publisher is a trusted author.
        self.assertFalse(cd.is_clean_lineage_member(
            "OpenMath2-Llama3.1-8B", self._lin("llama"), author="nvidia"))
        self.assertFalse(cd.is_clean_lineage_member(
            "AceReason-Llama-3.1-8B", self._lin("llama"), author="nvidia"))
        self.assertFalse(cd.is_clean_lineage_member(
            "Dolphin-Qwen3-8B", self._lin("qwen"), author="cognitivecomputations"))

    def test_rejects_wrong_sublineage_version_marker(self) -> None:
        # 9B-v2 belongs to a different sub-line; '-3-' nano lineage rejects it.
        self.assertFalse(cd.is_clean_lineage_member(
            "NVIDIA-Nemotron-Nano-9B-v2-NVFP4", self._lin("nemotron", "-nano"),
            author="nvidia"))


class TestCandidateNextVersions(unittest.TestCase):
    def test_minor_lineage(self) -> None:
        self.assertEqual(
            cd.candidate_next_versions([(3, 5), (3, 6)], minor_steps=3, major_steps=1),
            [(3, 7), (3, 8), (3, 9), (4,)])

    def test_integer_lineage(self) -> None:
        self.assertEqual(
            cd.candidate_next_versions([(4,)], minor_steps=2, major_steps=1),
            [(5,), (6,), (7,)])

    def test_mixed_minor_and_bare_integer_still_probes_minors(self) -> None:
        # A minor-versioned lineage that gained a bare-integer release
        # (llama 3.1/3.2 + 4) must still probe 4.1/4.2, not only 5/6.
        got = cd.candidate_next_versions([(3, 1), (3, 2), (4,)],
                                         minor_steps=2, major_steps=2)
        self.assertIn((4, 1), got)
        self.assertIn((4, 2), got)
        self.assertIn((5,), got)

    def test_empty(self) -> None:
        self.assertEqual(cd.candidate_next_versions([]), [])


class TestAsInt(unittest.TestCase):
    def test_coerces_or_zeroes(self) -> None:
        self.assertEqual(cd._as_int(5), 5)
        self.assertEqual(cd._as_int(None), 0)
        self.assertEqual(cd._as_int("NaN"), 0)
        self.assertEqual(cd._as_int("12.5"), 0)


class TestVramEstimation(unittest.TestCase):
    def test_parse_param_count_total_not_active(self) -> None:
        self.assertEqual(cd.parse_param_count("Qwen3-8B-NVFP4"), 8.0)
        self.assertEqual(cd.parse_param_count("Qwen3.5-35B-A3B-NVFP4"), 35.0)
        self.assertEqual(cd.parse_param_count("Qwen3.5-397B-A17B-NVFP4"), 397.0)
        self.assertEqual(cd.parse_param_count("Llama-3.1-405B-Instruct-FP8"), 405.0)
        self.assertEqual(cd.parse_param_count("Qwen3-0.6B"), 0.6)

    def test_parse_param_count_none_when_no_size(self) -> None:
        self.assertIsNone(cd.parse_param_count("gpt-oss"))
        self.assertIsNone(cd.parse_param_count("Qwen3-Coder"))

    def test_estimate_vram_format_aware(self) -> None:
        # Same params, very different VRAM by quant.
        self.assertAlmostEqual(cd.estimate_vram_gb(35.0, "NVFP4"), 19.25, places=2)
        self.assertAlmostEqual(cd.estimate_vram_gb(35.0, "BF16"), 70.0, places=2)
        self.assertAlmostEqual(cd.estimate_vram_gb(405.0, "FP8"), 445.5, places=2)
        # Unknown format -> conservative full-precision assumption.
        self.assertAlmostEqual(cd.estimate_vram_gb(8.0, "?"), 16.0, places=2)
        self.assertIsNone(cd.estimate_vram_gb(None, "NVFP4"))

    def test_band_floor_is_gpu_relative(self) -> None:
        lin = cd.Lineage(key="qwen|", brand="qwen", suffix="")
        lin.tracked_repos = {"nvidia/qwen3.5-35b-a3b-nvfp4"}  # ~19.25 GB
        floor, ceiling = lin.vram_band_gb(24.0, 1.25, 0.5)
        self.assertAlmostEqual(floor, 12.0, places=2)          # 50% of 24
        self.assertAlmostEqual(ceiling, 24.0, places=2)        # capped at GPU

    def test_band_family_relative_ceiling_below_gpu(self) -> None:
        """Ceiling is the family max at the SAME context the candidates are
        costed at. 35B NVFP4 = 19.25 GB of weights, +4.0 GB KV at the 32K
        default = 23.25. Costing the ceiling weights-only while candidates
        carry KV would make every candidate read as oversized."""
        lin = cd.Lineage(key="qwen|", brand="qwen", suffix="")
        lin.tracked_repos = {"nvidia/qwen3.5-35b-a3b-nvfp4"}
        floor, ceiling = lin.vram_band_gb(24.0, 1.0, 0.5)      # tolerance 1.0
        self.assertAlmostEqual(ceiling, 23.25, places=2)

    def test_band_ceiling_weights_only_at_zero_ctx(self) -> None:
        """ctx=0 reproduces the pre-KV behaviour exactly."""
        lin = cd.Lineage(key="qwen|", brand="qwen", suffix="")
        lin.tracked_repos = {"nvidia/qwen3.5-35b-a3b-nvfp4"}
        _, ceiling = lin.vram_band_gb(24.0, 1.0, 0.5, 0)
        self.assertAlmostEqual(ceiling, 19.25, places=2)

    def test_kv_term_grows_with_context(self) -> None:
        w = 10.0
        self.assertEqual(cd.estimate_kv_gb(w, 0), 0.0)
        k32 = cd.estimate_kv_gb(w, 32768)
        k128 = cd.estimate_kv_gb(w, 131072)
        self.assertGreater(k32, 0.0)
        self.assertAlmostEqual(k128, k32 * 4, places=3)

    def test_kv_term_grows_with_model_size(self) -> None:
        """Bigger models have more layers/KV heads, so more KV per token."""
        small = cd.estimate_kv_gb(5.0, 32768)
        large = cd.estimate_kv_gb(25.0, 32768)
        self.assertGreater(large, small)

    def test_estimate_includes_kv_when_context_given(self) -> None:
        weights = cd.estimate_vram_gb(9.0, "NVFP4", 0)
        withkv = cd.estimate_vram_gb(9.0, "NVFP4", 32768)
        self.assertGreater(withkv, weights)
        self.assertAlmostEqual(
            withkv, weights + cd.estimate_kv_gb(weights, 32768), places=3)

    def test_unparseable_params_stay_none_with_context(self) -> None:
        self.assertIsNone(cd.estimate_vram_gb(None, "NVFP4", 32768))

    def test_small_family_ceiling_falls_back_to_gpu(self) -> None:
        # Family max ~4.4 GB -> family ceiling 5.5 < floor 12 -> empty band;
        # fall back to GPU budget as ceiling so the floor still applies.
        lin = cd.Lineage(key="llama|", brand="llama", suffix="")
        lin.tracked_repos = {"nvidia/llama-3.1-8b-instruct-nvfp4"}  # ~4.4 GB
        floor, ceiling = lin.vram_band_gb(24.0, 1.25, 0.5)
        self.assertAlmostEqual(floor, 12.0, places=2)
        self.assertAlmostEqual(ceiling, 24.0, places=2)

    def test_band_no_tracked_size_uses_gpu_budget(self) -> None:
        lin = cd.Lineage(key="gpt-oss|", brand="gpt-oss", suffix="")
        lin.tracked_repos = {"openai/gpt-oss"}   # no parseable size
        floor, ceiling = lin.vram_band_gb(24.0, 1.25, 0.5)
        self.assertEqual((floor, ceiling), (12.0, 24.0))


class TestBaseDetection(unittest.TestCase):
    def test_name_says_base(self) -> None:
        self.assertTrue(cd.is_base_model("Qwen3.5-9B-Base", True))
        self.assertTrue(cd.is_base_model("Llama-3.1-8B-pretrain", True))

    def test_name_says_instruct_overrides_missing_tag(self) -> None:
        # Conservative: a named-instruct quant survives even with no tag
        # (some NVFP4 quants strip the conversational tag).
        self.assertFalse(cd.is_base_model("Qwen3-8B-Instruct-NVFP4", False))
        self.assertFalse(cd.is_base_model("Llama-3.1-8B-IT", False))

    def test_falls_back_to_tag(self) -> None:
        self.assertTrue(cd.is_base_model("Qwen3-8B-DMS-8x", False))   # no tag
        self.assertFalse(cd.is_base_model("Qwen3-8B-DMS-8x", True))   # tagged


class TestHfWeightGb(unittest.TestCase):
    def setUp(self) -> None:
        self._orig = cd.http_json

    def tearDown(self) -> None:
        cd.http_json = self._orig

    def test_sums_safetensors_excludes_mirrors(self) -> None:
        GB = 1024 ** 3
        cd.http_json = lambda url, timeout: {"siblings": [
            {"rfilename": "model-00001.safetensors", "size": 8 * GB},
            {"rfilename": "model-00002.safetensors", "size": 8 * GB},
            {"rfilename": "original/consolidated.pth", "size": 30 * GB},  # mirror
            {"rfilename": "README.md", "size": 1000},                     # non-weight
        ]}
        self.assertAlmostEqual(cd.hf_weight_gb("x/y", 5), 16.0, places=2)

    def test_none_on_failure(self) -> None:
        def boom(url, timeout):
            raise RuntimeError("network down")
        cd.http_json = boom
        self.assertIsNone(cd.hf_weight_gb("x/y", 5))


class TestDiscoverBlock(unittest.TestCase):
    def test_defaults_for_absent(self) -> None:
        cfg = cd.parse_discover_block(None)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.hf_authors, [])
        self.assertIsNone(cfg.name_regex)
        self.assertIsNone(cfg.min_version)

    def test_full_block(self) -> None:
        cfg = cd.parse_discover_block({
            "enabled": True,
            "hf_authors": ["nvidia", "apolo13x"],
            "name_regex": "Qwen3",
            "min_version": "3.5",
            "ollama_names": ["qwen-next"],
        })
        self.assertEqual(cfg.hf_authors, ["nvidia", "apolo13x"])
        self.assertEqual(cfg.name_regex, "Qwen3")
        self.assertEqual(cfg.min_version, (3, 5))
        self.assertEqual(cfg.ollama_names, ["qwen-next"])

    def test_min_version_quoted_string_dodges_float(self) -> None:
        # '3.10' quoted -> (3, 10); a bare YAML float 3.10 would be 3.1.
        cfg = cd.parse_discover_block({"min_version": "3.10"})
        self.assertEqual(cfg.min_version, (3, 10))

    def test_enabled_false(self) -> None:
        self.assertFalse(cd.parse_discover_block({"enabled": False}).enabled)

    def test_malformed_ignored(self) -> None:
        cfg = cd.parse_discover_block({"hf_authors": "notalist", "name_regex": 5})
        self.assertEqual(cfg.hf_authors, [])
        self.assertIsNone(cfg.name_regex)

    def test_invalid_name_regex_dropped_not_crash(self) -> None:
        # A bad regex must be dropped at parse time, not blow up later.
        cfg = cd.parse_discover_block({"name_regex": "[invalid("})
        self.assertIsNone(cfg.name_regex)


class TestBuildLineages(unittest.TestCase):
    FAMILIES = [
        {"name": "qwen3", "ollama_repos": ["qwen3"],
         "hf_repos": ["nvidia/Qwen3-8B-NVFP4", "nvidia/Qwen3-30B-A3B-NVFP4"]},
        {"name": "qwen3.5", "ollama_repos": ["qwen3.5"],
         "hf_repos": [{"repo": "apolo13x/Qwen3.5-27B-NVFP4"}]},
        {"name": "qwen3.6",
         "hf_repos": ["sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"]},
        {"name": "qwen3-coder",
         "hf_repos": ["NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4"]},
        {"name": "gpt-oss", "hf_repos": ["openai/gpt-oss-20b"]},
    ]

    def setUp(self) -> None:
        self.lins = cd.build_lineages(self.FAMILIES)

    def test_qwen_versions_grouped(self) -> None:
        qwen = self.lins[cd.lineage_key("qwen", "")]
        self.assertEqual(sorted(qwen.family_names), ["qwen3", "qwen3.5", "qwen3.6"])
        self.assertEqual(qwen.hf_versions, {(3,), (3, 5), (3, 6)})
        self.assertIn("nvidia", qwen.hf_authors)
        self.assertIn("apolo13x", qwen.hf_authors)

    def test_coder_is_its_own_lineage(self) -> None:
        self.assertIn(cd.lineage_key("qwen", "-coder"), self.lins)
        coder = self.lins[cd.lineage_key("qwen", "-coder")]
        self.assertEqual(coder.family_names, ["qwen3-coder"])

    def test_version_to_family_map(self) -> None:
        qwen = self.lins[cd.lineage_key("qwen", "")]
        self.assertEqual(qwen.version_to_family[(3, 6)], "qwen3.6")

    def test_gpt_oss_has_no_version(self) -> None:
        gpt = self.lins[cd.lineage_key("gpt-oss", "")]
        self.assertEqual(gpt.all_versions, set())


class TestEffectiveAuthors(unittest.TestCase):
    def test_discover_replaces_auto(self) -> None:
        lin = cd.Lineage(key="qwen|", brand="qwen", suffix="")
        lin.hf_authors = {"nvidia", "apolo13x", "ykarout"}
        # No override -> auto-derived authors.
        self.assertEqual(cd.effective_authors(lin),
                         ["apolo13x", "nvidia", "ykarout"])
        # Override RESTRICTS to the vetted set.
        lin.discover.hf_authors = ["nvidia"]
        self.assertEqual(cd.effective_authors(lin), ["nvidia"])


class TestClassifyCandidate(unittest.TestCase):
    def _qwen(self) -> object:
        lin = cd.Lineage(key="qwen|", brand="qwen", suffix="")
        lin.hf_versions = {(3,), (3, 5), (3, 6)}
        lin.version_to_family = {(3,): "qwen3", (3, 5): "qwen3.5", (3, 6): "qwen3.6"}
        return lin

    def test_same_maps_to_family(self) -> None:
        klass, mapping = cd.classify_candidate((3, 6), self._qwen())
        self.assertEqual(klass, "SAME")
        self.assertIn("qwen3.6", mapping)

    def test_newer_proposes_new_family(self) -> None:
        klass, mapping = cd.classify_candidate((3, 7), self._qwen())
        self.assertEqual(klass, "NEWER")
        self.assertIn("qwen3.7", mapping)

    def test_gap_below_max(self) -> None:
        lin = cd.Lineage(key="qwen|", brand="qwen", suffix="")
        lin.hf_versions = {(3,), (3, 6)}  # 3.5 missing
        klass, _ = cd.classify_candidate((3, 5), lin)
        self.assertEqual(klass, "GAP")


class TestRepoMatchesLineage(unittest.TestCase):
    def test_strict_applies_structural_filter(self) -> None:
        qwen = cd.Lineage(key="qwen|", brand="qwen", suffix="")
        name_re = cd.re.compile(r"(?i)qwen\d")
        self.assertTrue(cd.repo_matches_lineage(
            "Qwen3-8B-NVFP4", qwen, name_re, strict=True))
        self.assertFalse(cd.repo_matches_lineage(
            "Qwen3-VL-8B", qwen, name_re, strict=True))

    def test_non_strict_trusts_regex(self) -> None:
        qwen = cd.Lineage(key="qwen|", brand="qwen", suffix="")
        name_re = cd.re.compile(r"(?i)qwen\d")
        # When the operator supplies a custom regex we skip the structural
        # filter -- VL passes because the regex matched.
        self.assertTrue(cd.repo_matches_lineage(
            "Qwen3-VL-8B", qwen, name_re, strict=False))


class TestEndToEndStubbed(unittest.TestCase):
    """Run the full pipeline with HF/Ollama network calls stubbed."""

    FAMILIES = [
        {"name": "llama3.1", "ollama_repos": ["llama3.1"],
         "hf_repos": ["nvidia/Llama-3.1-8B-Instruct-NVFP4"]},
        {"name": "llama3.2", "ollama_repos": ["llama3.2"]},
    ]

    def setUp(self) -> None:
        self._orig_hf = cd.hf_search_author
        self._orig_status = cd.http_status
        self._orig_weight = cd.hf_weight_gb

        def fake_hf(author, query, limit, timeout):
            if author != "nvidia":
                return []
            return [
                {"id": "nvidia/Llama-3.3-24B-Instruct-NVFP4",  # NEWER, in range (~13GB)
                 "tags": ["fp4"], "likes": 30, "downloads": 9000},
                {"id": "nvidia/Llama-3.3-24B-Base-NVFP4",      # in range (~13GB) but BASE
                 "tags": ["fp4"], "likes": 8, "downloads": 3000},
                {"id": "nvidia/Llama-3.3-8B-Instruct-NVFP4",   # NEWER, undersized (~4GB)
                 "tags": ["fp4"], "likes": 12, "downloads": 5000},
                {"id": "nvidia/Llama-3.3-70B-Instruct-NVFP4",  # NEWER, oversized (~38GB)
                 "tags": ["fp4"], "likes": 45, "downloads": 17228},
                {"id": "nvidia/Llama-3.1-405B-Instruct-FP8",   # SAME, oversized
                 "tags": ["fp8", "conversational"], "likes": 16, "downloads": 4748},
                {"id": "nvidia/Llama-3.1-8B-Instruct-NVFP4",  # already tracked
                 "tags": ["fp4"], "likes": 1, "downloads": 1},
                {"id": "nvidia/Llama-3.1-VL-8B",  # foreign line -> filtered
                 "tags": [], "likes": 0, "downloads": 0},
            ]

        def fake_status(url, timeout):
            return 200 if url.endswith("/llama4/tags") else 404

        cd.hf_search_author = fake_hf
        cd.http_status = fake_status
        cd.hf_weight_gb = lambda repo, timeout: None  # no '?' fetch in this stub

    def tearDown(self) -> None:
        cd.hf_search_author = self._orig_hf
        cd.http_status = self._orig_status
        cd.hf_weight_gb = self._orig_weight

    def _run(self):
        return cd.run(self.FAMILIES, family_filter=None, do_hf=True,
                      do_ollama=True, hf_limit=50, ollama_probe=3, timeout=5)

    def test_tracked_and_foreign_excluded(self) -> None:
        [r] = self._run()
        repos = {c.repo for c in r["hf"]}
        self.assertNotIn("nvidia/Llama-3.1-8B-Instruct-NVFP4", repos)  # tracked
        self.assertNotIn("nvidia/Llama-3.1-VL-8B", repos)             # foreign line

    def test_band_classification(self) -> None:
        # Small family -> band [12, 24] GB. 24B ~13 in range, 8B ~4 under,
        # 70B ~38 / 405B over.
        [r] = self._run()
        by_repo = {c.repo: c for c in r["hf"]}
        self.assertTrue(by_repo["nvidia/Llama-3.3-24B-Instruct-NVFP4"].in_range)
        self.assertTrue(by_repo["nvidia/Llama-3.3-8B-Instruct-NVFP4"].undersized)
        self.assertTrue(by_repo["nvidia/Llama-3.3-70B-Instruct-NVFP4"].oversized)
        self.assertTrue(by_repo["nvidia/Llama-3.1-405B-Instruct-FP8"].oversized)

    def test_base_classification(self) -> None:
        [r] = self._run()
        by_repo = {c.repo: c for c in r["hf"]}
        self.assertTrue(by_repo["nvidia/Llama-3.3-24B-Base-NVFP4"].base)
        self.assertTrue(by_repo["nvidia/Llama-3.3-24B-Base-NVFP4"].in_range)
        self.assertFalse(by_repo["nvidia/Llama-3.3-24B-Instruct-NVFP4"].base)

    def test_report_hides_out_of_band_and_base_by_default(self) -> None:
        results = self._run()
        report = cd.render_report(results, do_hf=True, do_ollama=False)
        self.assertIn("Llama-3.3-24B-Instruct-NVFP4", report)  # in band, instruct
        self.assertNotIn("Llama-3.3-24B-Base", report)         # base hidden
        self.assertNotIn("Llama-3.3-70B", report)              # too big
        self.assertNotIn("Llama-3.3-8B", report)               # too small
        for tok in ("too big", "too small", "base"):
            self.assertIn(tok, report)
        big = cd.render_report(results, do_hf=True, do_ollama=False,
                               include_oversized=True)
        self.assertIn("Llama-3.3-70B", big)
        small = cd.render_report(results, do_hf=True, do_ollama=False,
                                 include_undersized=True)
        self.assertIn("Llama-3.3-8B", small)
        with_base = cd.render_report(results, do_hf=True, do_ollama=False,
                                     include_base=True)
        self.assertIn("Llama-3.3-24B-Base", with_base)

    def test_same_classified(self) -> None:
        [r] = self._run()
        same = [c for c in r["hf"] if c.klass == "SAME"]
        self.assertIn("nvidia/Llama-3.1-405B-Instruct-FP8",
                      {c.repo for c in same})

    def test_ollama_existing_flagged(self) -> None:
        [r] = self._run()
        existing = [c for c in r["ollama"] if c.status == 200]
        self.assertIn("llama4", {c.library for c in existing})

    def test_report_is_marked_read_only(self) -> None:
        results = self._run()
        report = cd.render_report(results, do_hf=True, do_ollama=True)
        self.assertIn("read-only", report)
        self.assertIn("NO auto-edits", report)
        # JSON path renders without error too.
        cd.to_json(results, do_hf=True, do_ollama=True)


class TestRenderOllama(unittest.TestCase):
    """render_report's Ollama block, built from synthetic candidates."""

    def _result(self, ollama):
        lin = cd.Lineage(key="qwen|", brand="qwen", suffix="")
        lin.family_names = ["qwen3.5"]
        lin.ollama_versions = {(3, 5)}
        return [{"lineage": lin, "hf": [], "ollama": ollama}]

    def test_outage_is_signalled_not_silent(self) -> None:
        # Every probe failed (status 0) -> must warn, not read as "none".
        ol = [cd.OllamaCandidate("qwen3.6", (3, 6), 0, "SAME", "x"),
              cd.OllamaCandidate("qwen3.7", (3, 7), 0, "NEWER", "y")]
        report = cd.render_report(self._result(ol), do_hf=False, do_ollama=True)
        self.assertIn("probe(s) failed", report)
        self.assertNotIn("EXISTS", report)

    def test_explicit_names_do_not_crowd_out_numeric_watch(self) -> None:
        # version=() EXPLICIT 404s must not steal the numeric pending slots.
        ol = [
            cd.OllamaCandidate("qwen-next", (), 404, "EXPLICIT", "z"),
            cd.OllamaCandidate("qwen3.6", (3, 6), 404, "NEWER", "a"),
            cd.OllamaCandidate("qwen3.7", (3, 7), 404, "NEWER", "b"),
        ]
        report = cd.render_report(self._result(ol), do_hf=False, do_ollama=True)
        self.assertIn("qwen3.6", report)   # numeric watch survives
        self.assertIn("qwen-next", report)  # explicit still shown


class TestAddYaml(unittest.TestCase):
    """The model-families.yaml mutation path (the only writer)."""

    MIN_YAML = (
        "families:\n"
        "  - name: qwen3.6\n"
        "    # curated comment that must survive\n"
        "    hf_repos:\n"
        "      - sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP\n"
        "    arch_ref: sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP\n"
        "    thinking: true\n"
        "\n"
        "  - name: qwen3-coder\n"
        "    hf_repos:\n"
        "      - NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4\n"
        "    arch_ref: NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4\n"
        "    thinking: false\n"
    )

    def _comment_lines(self, text):
        import re as _re
        return [l for l in text.split("\n") if _re.match(r"^\s*#", l)]

    def test_append_to_existing_key_preserves_comments(self) -> None:
        out = cd.insert_repo_entry(
            self.MIN_YAML, "qwen3.6", "hf_repos",
            cd.format_entry("nvidia/Qwen3.6-35B-A3B-NVFP4", "hf_repos"))
        data = yaml.safe_load(out)   # still valid YAML
        self.assertTrue(cd._repo_in_family(
            data, "qwen3.6", "hf_repos", "nvidia/Qwen3.6-35B-A3B-NVFP4"))
        # comment survived, and was not duplicated
        self.assertEqual(self._comment_lines(self.MIN_YAML),
                         self._comment_lines(out))
        # the existing entry is still there too
        self.assertTrue(cd._repo_in_family(
            data, "qwen3.6", "hf_repos",
            "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"))

    def test_create_missing_key(self) -> None:
        out = cd.insert_repo_entry(
            self.MIN_YAML, "qwen3.6", "ollama_repos",
            cd.format_entry("qwen3.6", "ollama_repos"))
        data = yaml.safe_load(out)
        self.assertTrue(cd._repo_in_family(data, "qwen3.6", "ollama_repos",
                                           "qwen3.6"))

    def test_insert_into_right_family_only(self) -> None:
        out = cd.insert_repo_entry(
            self.MIN_YAML, "qwen3-coder", "hf_repos",
            cd.format_entry("NVFP4/Qwen3-Coder-480B-A35B-Instruct-FP4", "hf_repos"))
        data = yaml.safe_load(out)
        names = {f["name"]: f for f in data["families"]}
        self.assertEqual(len(names["qwen3-coder"]["hf_repos"]), 2)
        self.assertEqual(len(names["qwen3.6"]["hf_repos"]), 1)  # untouched

    def test_unknown_family_raises(self) -> None:
        with self.assertRaises(KeyError):
            cd.insert_repo_entry(self.MIN_YAML, "nonexistent", "hf_repos",
                                 ["      - x/y"])


class TestGgufEntry(unittest.TestCase):
    def test_tag_prefix(self) -> None:
        self.assertEqual(cd._gguf_tag_prefix("Qwen3.6-35B-A3B-GGUF"), "35b-a3b")
        self.assertEqual(cd._gguf_tag_prefix("Qwen3.5-27B-GGUF"), "27b")
        self.assertIsNone(cd._gguf_tag_prefix("gpt-oss"))

    def test_gguf_entry_uses_safe_empty_include(self) -> None:
        entry = cd.format_entry("unsloth/Qwen3.6-35B-A3B-GGUF", "gguf_repos")
        joined = "\n".join(entry)
        self.assertIn("- repo: unsloth/Qwen3.6-35B-A3B-GGUF", joined)
        self.assertIn("tag_prefix: 35b-a3b", joined)
        self.assertIn("include: []", joined)   # safe placeholder, emits nothing

    def test_is_gguf_repo(self) -> None:
        self.assertTrue(cd._is_gguf_repo("Qwen3.5-9B-GGUF"))
        self.assertTrue(cd._is_gguf_repo("Qwen3.5-9B-Q4_K_M"))       # k-quant
        self.assertTrue(cd._is_gguf_repo("Qwen3.5-9B-UD-IQ3_XXS"))   # imatrix quant
        self.assertFalse(cd._is_gguf_repo("Qwen3-8B-NVFP4"))         # safetensors
        self.assertFalse(cd._is_gguf_repo("Qwen3.5-9B-Instruct"))    # no quant marker


class TestPlanAdd(unittest.TestCase):
    FAMILIES = [
        {"name": "qwen3", "ollama_repos": ["qwen3"],
         "hf_repos": ["nvidia/Qwen3-8B-NVFP4"]},
        {"name": "qwen3.5",
         "hf_repos": ["ykarout/Qwen3.5-9B-NVFP4"]},
        {"name": "qwen3.6",
         "hf_repos": ["sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"]},
        {"name": "qwen3-coder",
         "hf_repos": ["NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4"]},
    ]

    def setUp(self) -> None:
        self.lin = cd.build_lineages(self.FAMILIES)
        self.tracked = {r for l in self.lin.values() for r in l.tracked_repos}

    def test_hf_maps_to_family(self) -> None:
        p = cd.plan_add("nvidia/Qwen3.6-35B-A3B-NVFP4", self.lin, self.tracked)
        self.assertEqual(p["family"], "qwen3.6")
        self.assertEqual(p["list_key"], "hf_repos")

    def test_gguf_goes_to_gguf_repos(self) -> None:
        p = cd.plan_add("unsloth/Qwen3.6-27B-GGUF", self.lin, self.tracked)
        self.assertEqual((p["family"], p["list_key"]), ("qwen3.6", "gguf_repos"))

    def test_ollama_name_maps(self) -> None:
        p = cd.plan_add("qwen3.6", self.lin, self.tracked)
        self.assertEqual((p["family"], p["list_key"]), ("qwen3.6", "ollama_repos"))

    def test_coder_routes_to_coder_family(self) -> None:
        p = cd.plan_add("NVFP4/Qwen3-Coder-480B-A35B-Instruct-FP4",
                        self.lin, self.tracked)
        self.assertEqual(p["family"], "qwen3-coder")

    def test_already_tracked_refused(self) -> None:
        p = cd.plan_add("nvidia/Qwen3-8B-NVFP4", self.lin, self.tracked)
        self.assertIn("error", p)

    def test_already_tracked_ollama_lib_refused(self) -> None:
        # qwen3 lib is in the qwen3 family's ollama_repos -> must refuse,
        # not append a duplicate. tracked_all must include ollama_libs.
        tracked = cd.all_tracked(self.lin)
        p = cd.plan_add("qwen3", self.lin, tracked)
        self.assertIn("error", p)

    def test_gguf_by_quant_marker_routes_to_gguf(self) -> None:
        # No '-GGUF' token, only a k-quant marker -> still gguf_repos.
        p = cd.plan_add("bartowski/Qwen3.5-9B-Q4_K_M", self.lin, self.tracked)
        self.assertEqual(p["list_key"], "gguf_repos")

    def test_new_version_refused(self) -> None:
        # 3.9 has no family -> would be a new family -> out of scope.
        p = cd.plan_add("nvidia/Qwen3.9-30B-A3B-NVFP4", self.lin, self.tracked)
        self.assertIn("error", p)


class TestDoAddWrites(unittest.TestCase):
    def test_writes_and_validates(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model-families.yaml"
            p.write_text(TestAddYaml.MIN_YAML)
            rc = cd.do_add("nvidia/Qwen3.6-35B-A3B-NVFP4", p, assume_yes=True)
            self.assertEqual(rc, 0)
            data = yaml.safe_load(p.read_text())
            self.assertTrue(cd._repo_in_family(
                data, "qwen3.6", "hf_repos", "nvidia/Qwen3.6-35B-A3B-NVFP4"))

    def test_refuses_already_tracked_no_write(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model-families.yaml"
            p.write_text(TestAddYaml.MIN_YAML)
            before = p.read_text()
            rc = cd.do_add("sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP", p,
                           assume_yes=True)
            self.assertEqual(rc, 1)
            self.assertEqual(p.read_text(), before)   # untouched

    OLLAMA_YAML = (
        "families:\n"
        "  - name: qwen3\n"
        "    ollama_repos:\n"
        "      - qwen3\n"
        "    hf_repos:\n"
        "      - nvidia/Qwen3-8B-NVFP4\n"
        "    arch_ref: nvidia/Qwen3-8B-NVFP4\n"
    )

    def test_refuses_already_tracked_ollama_lib_no_duplicate(self) -> None:
        # Regression: re-adding an already-tracked ollama lib must NOT append
        # a duplicate (the lib lives in ollama_libs, not tracked_repos).
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model-families.yaml"
            p.write_text(self.OLLAMA_YAML)
            before = p.read_text()
            rc = cd.do_add("qwen3", p, assume_yes=True)
            self.assertEqual(rc, 1)
            self.assertEqual(p.read_text(), before)   # byte-identical, no dup
            data = yaml.safe_load(p.read_text())
            q3 = [f for f in data["families"] if f["name"] == "qwen3"][0]
            self.assertEqual(q3["ollama_repos"], ["qwen3"])   # still one entry


class TestValidationDefense(unittest.TestCase):
    def test_rejects_duplicate_insert(self) -> None:
        # Defense in depth: even if a duplicate slips in, validation fails closed.
        dup = (
            "families:\n"
            "  - name: x\n"
            "    hf_repos:\n"
            "      - a/Foo\n"
            "      - a/Foo\n"   # duplicate
        )
        ok, msg = cd._validate_insertion(dup, "x", "hf_repos", "a/Foo")
        self.assertFalse(ok)
        self.assertIn("already present", msg)

    def test_accepts_single_occurrence(self) -> None:
        good = (
            "families:\n"
            "  - name: x\n"
            "    hf_repos:\n"
            "      - a/Foo\n"
        )
        ok, _ = cd._validate_insertion(good, "x", "hf_repos", "a/Foo")
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
