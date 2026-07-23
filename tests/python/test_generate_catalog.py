"""Unit tests for scripts/generate-catalog.py.

Network-free surface only: the GGUF tag-token derivation (`_gguf_tag_token`),
a shape guard on the hand-maintained `ornith` family block in
scripts/model-families.yaml, and the row-loss guard that refuses to overwrite
deploy/models.yaml with a truncated catalog (every upstream call stubbed).
The live HF / Ollama catalog build is exercised by `make catalog-regen`, not
here.

Stdlib unittest only. Run with:
    python3 -m unittest tests.python.test_generate_catalog
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
import urllib.error
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
    """Guard the hand-maintained 9B Ornith entry against accidental drift.

    GGUF via Ollama plus ONE curated HF repo: the upstream bf16
    safetensors OOMs at vLLM model load on a 24G card (measured
    2026-07-14, dropped in d729033), but the in-house NVFP4 quant
    (ksparavec/Ornith-1.0-9B-NVFP4) fits and serves at 256K, so it is
    the only hf_repos row and carries the qwen3-family parsers.
    """

    def test_arch_ref_and_thinking(self) -> None:
        fam = _ornith_family()
        self.assertEqual(fam["arch_ref"], "deepreinforce-ai/Ornith-1.0-9B")
        self.assertTrue(fam["thinking"])

    def test_nvfp4_is_only_hf_repo_with_parsers(self) -> None:
        # Upstream bf16 safetensors OOMs on 24G (dropped in d729033);
        # the in-house NVFP4 quant is the single HF row and needs the
        # qwen3-family parsers for vLLM/SGLang serving.
        fam = _ornith_family()
        self.assertEqual(fam["hf_repos"], ["ksparavec/Ornith-1.0-9B-NVFP4"])
        parsers = fam["parsers"]
        self.assertEqual(
            parsers["vllm"], {"reasoning": "qwen3", "tool": "qwen3_xml"}
        )
        self.assertEqual(
            parsers["sglang"], {"reasoning": "qwen3", "tool": "qwen"}
        )

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

    def test_no_speculative_mtp_block(self) -> None:
        # Intentionally MTP-less: the vocab-248320 draft lm_head would OOM
        # at load on 24G. No gguf_repos entry should carry an mtp block.
        fam = _ornith_family()
        for entry in fam["gguf_repos"]:
            self.assertNotIn("mtp", entry)


_MIN_CONFIG = {
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "hidden_size": 512,
}
_PREVIOUS_CATALOG = "# the good catalog that must survive a failed run\n"


class TestRowLossGuard(unittest.TestCase):
    """deploy/models.yaml is rewritten whole, so a run that lost rows to an
    upstream failure must refuse to write and exit 1 -- otherwise a transient
    HF/Ollama outage silently deletes models from the system."""

    _PATCHED = ("FAMILIES_YAML", "OUTPUT_YAML", "hf_weight_bytes", "hf_config",
                "hf_repo_sha", "_hf_blobs", "ollama_tags",
                "ollama_manifest_size", "hf_gguf_files")

    def setUp(self) -> None:
        self._saved = {k: getattr(gc, k) for k in self._PATCHED}
        gc._row_loss.clear()
        gc._permanent_skips.clear()
        self._tmp = tempfile.TemporaryDirectory()
        d = Path(self._tmp.name)
        gc.FAMILIES_YAML = d / "families.yaml"
        gc.OUTPUT_YAML = d / "models.yaml"
        gc.OUTPUT_YAML.write_text(_PREVIOUS_CATALOG)
        self._families({"name": "fam", "arch_ref": "org/ref",
                        "hf_repos": ["org/A", "org/B"]})
        gc.hf_config = lambda repo: dict(_MIN_CONFIG)
        gc.hf_repo_sha = lambda repo: "abcdef123456"
        gc._hf_blobs = lambda repo: {"tags": ["conversational"]}
        gc.hf_weight_bytes = lambda repo: 8 * 1024 ** 3

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            setattr(gc, k, v)
        gc._row_loss.clear()
        gc._permanent_skips.clear()
        self._tmp.cleanup()

    def _families(self, *fams: dict) -> None:
        gc.FAMILIES_YAML.write_text(yaml.safe_dump({"families": list(fams)}))

    def _main(self, *argv: str) -> tuple[int, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = gc.main(list(argv))
        return rc, err.getvalue()

    @staticmethod
    def _http_error(code: int, msg: str = "boom") -> urllib.error.HTTPError:
        return urllib.error.HTTPError("http://upstream/x", code, msg, None, None)

    def test_clean_run_writes_the_catalog(self) -> None:
        rc, _ = self._main()
        self.assertEqual(rc, 0)
        text = gc.OUTPUT_YAML.read_text()
        self.assertIn('name: "A"', text)
        self.assertIn('name: "B"', text)

    def test_hf_size_failure_refuses_to_write(self) -> None:
        def flaky(repo):
            if repo == "org/B":
                raise urllib.error.URLError("timed out")
            return 8 * 1024 ** 3

        gc.hf_weight_bytes = flaky
        rc, err = self._main()
        self.assertEqual(rc, 1)
        self.assertEqual(gc.OUTPUT_YAML.read_text(), _PREVIOUS_CATALOG)
        self.assertIn("REFUSING", err)

    def test_repo_with_no_weight_files_is_a_legitimate_skip(self) -> None:
        # Deterministic upstream content, not a fetch failure: the row is
        # dropped and the (smaller) catalog is still written.
        gc.hf_weight_bytes = lambda repo: 0 if repo == "org/B" else 8 * 1024 ** 3
        rc, _ = self._main()
        self.assertEqual(rc, 0)
        text = gc.OUTPUT_YAML.read_text()
        self.assertIn('name: "A"', text)
        self.assertNotIn('name: "B"', text)

    def test_platform_gated_ollama_tag_is_a_legitimate_skip(self) -> None:
        self._families({"name": "fam", "arch_ref": "org/ref",
                        "ollama_repos": ["lib"]})
        gc.ollama_tags = lambda lib: ["9b-q4_K_M", "9b-q8_0"]

        def manifest(lib, tag):
            if tag == "9b-q8_0":   # macOS-only build -> HTTP 412
                raise urllib.error.HTTPError("u", 412, "gated", None, None)
            return 5 * 1024 ** 3

        gc.ollama_manifest_size = manifest
        rc, _ = self._main()
        self.assertEqual(rc, 0)
        text = gc.OUTPUT_YAML.read_text()
        self.assertIn("lib:9b-q4_K_M", text)
        self.assertNotIn("lib:9b-q8_0", text)

    def test_ollama_tag_listing_failure_refuses_to_write(self) -> None:
        # The whole library's tags vanish -> truncated catalog, not a skip.
        self._families({"name": "fam", "arch_ref": "org/ref",
                        "ollama_repos": ["lib"]})

        def boom(lib):
            raise urllib.error.URLError("registry down")

        gc.ollama_tags = boom
        rc, err = self._main()
        self.assertEqual(rc, 1)
        self.assertEqual(gc.OUTPUT_YAML.read_text(), _PREVIOUS_CATALOG)
        self.assertIn("Ollama tag list failed", err)

    def test_gguf_listing_failure_refuses_to_write(self) -> None:
        self._families({"name": "fam", "arch_ref": "org/ref",
                        "gguf_repos": [{"repo": "org/G", "tag_prefix": "9b"}]})

        def boom(repo):
            raise urllib.error.URLError("hf down")

        gc.hf_gguf_files = boom
        rc, err = self._main()
        self.assertEqual(rc, 1)
        self.assertEqual(gc.OUTPUT_YAML.read_text(), _PREVIOUS_CATALOG)
        self.assertIn("HF GGUF listing failed", err)

    def test_family_without_arch_ref_refuses_to_write(self) -> None:
        # Every row of that family is lost -- the largest truncation there is.
        self._families({"name": "fam", "arch_ref": "org/ref",
                        "hf_repos": ["org/A"]},
                       {"name": "broken", "hf_repos": ["org/C"]})
        rc, err = self._main()
        self.assertEqual(rc, 1)
        self.assertEqual(gc.OUTPUT_YAML.read_text(), _PREVIOUS_CATALOG)
        self.assertIn("no arch_ref", err)

    # ── permanent (deterministic) vs transient failures ──────────────────
    #
    # `make catalog-discover-add` writes repos into model-families.yaml and
    # upstream authors delete / rename / gate them. A permanently-gone repo
    # must NOT wedge regeneration: it is a deterministic skip, exactly like a
    # 412 platform-gated Ollama tag. Only failures a retry could fix block
    # the write.

    def _dead_hf_size(self, code: int):
        def fetch(repo):
            if repo == "org/B":
                raise self._http_error(code, "gone")
            return 8 * 1024 ** 3
        return fetch

    def test_hf_404_removed_repo_is_a_deterministic_skip(self) -> None:
        gc.hf_weight_bytes = self._dead_hf_size(404)
        rc, err = self._main()
        self.assertEqual(rc, 0)
        text = gc.OUTPUT_YAML.read_text()
        self.assertIn('name: "A"', text)
        self.assertNotIn('name: "B"', text)
        self.assertIn("GONE (HTTP 404)", err)
        self.assertNotIn("REFUSING", err)

    def test_hf_410_401_403_are_deterministic_skips(self) -> None:
        # 410 Gone (retired repo), 401/403 (gated repo, no token) are all
        # permanent for this host -- retrying forever helps nobody.
        for code in (410, 401, 403):
            with self.subTest(code=code):
                gc._row_loss.clear()
                gc._permanent_skips.clear()
                gc.OUTPUT_YAML.write_text(_PREVIOUS_CATALOG)
                gc.hf_weight_bytes = self._dead_hf_size(code)
                rc, err = self._main()
                self.assertEqual(rc, 0)
                self.assertIn('name: "A"', gc.OUTPUT_YAML.read_text())
                self.assertIn(f"GONE (HTTP {code})", err)

    def test_transient_http_codes_refuse_to_write(self) -> None:
        # 408 / 429 / 5xx are the retry-later class: still a hard block.
        for code in (408, 429, 500, 502, 503):
            with self.subTest(code=code):
                gc._row_loss.clear()
                gc._permanent_skips.clear()
                gc.OUTPUT_YAML.write_text(_PREVIOUS_CATALOG)
                gc.hf_weight_bytes = self._dead_hf_size(code)
                rc, err = self._main()
                self.assertEqual(rc, 1)
                self.assertEqual(gc.OUTPUT_YAML.read_text(), _PREVIOUS_CATALOG)
                self.assertIn("REFUSING", err)

    def test_ollama_manifest_404_is_a_deterministic_skip(self) -> None:
        self._families({"name": "fam", "arch_ref": "org/ref",
                        "ollama_repos": ["lib"]})
        gc.ollama_tags = lambda lib: ["9b-q4_K_M", "9b-q8_0"]

        def manifest(lib, tag):
            if tag == "9b-q8_0":       # tag pulled from the registry
                raise self._http_error(404, "not found")
            return 5 * 1024 ** 3

        gc.ollama_manifest_size = manifest
        rc, err = self._main()
        self.assertEqual(rc, 0)
        text = gc.OUTPUT_YAML.read_text()
        self.assertIn("lib:9b-q4_K_M", text)
        self.assertNotIn("lib:9b-q8_0", text)
        self.assertIn("GONE (HTTP 404)", err)

    def test_ollama_tag_listing_404_is_a_deterministic_skip(self) -> None:
        # The library page itself is gone -- the surviving family rows must
        # still reach the catalog.
        self._families({"name": "fam", "arch_ref": "org/ref",
                        "hf_repos": ["org/A"], "ollama_repos": ["lib"]})

        def gone(lib):
            raise self._http_error(404, "no such library")

        gc.ollama_tags = gone
        rc, err = self._main()
        self.assertEqual(rc, 0)
        self.assertIn('name: "A"', gc.OUTPUT_YAML.read_text())
        self.assertIn("GONE (HTTP 404)", err)
        self.assertIn("Ollama tag list failed", err)

    def test_gguf_listing_404_is_a_deterministic_skip(self) -> None:
        self._families({"name": "fam", "arch_ref": "org/ref",
                        "hf_repos": ["org/A"],
                        "gguf_repos": [{"repo": "org/G", "tag_prefix": "9b"}]})

        def gone(repo):
            raise self._http_error(404, "renamed")

        gc.hf_gguf_files = gone
        rc, err = self._main()
        self.assertEqual(rc, 0)
        self.assertIn('name: "A"', gc.OUTPUT_YAML.read_text())
        self.assertIn("HF GGUF listing failed", err)
        self.assertIn("GONE (HTTP 404)", err)

    def test_permanent_skip_does_not_mask_a_transient_failure(self) -> None:
        # One dead repo AND one flaky repo -> the flaky one still blocks.
        self._families({"name": "fam", "arch_ref": "org/ref",
                        "hf_repos": ["org/A", "org/B", "org/C"]})

        def mixed(repo):
            if repo == "org/B":
                raise self._http_error(404, "gone")
            if repo == "org/C":
                raise urllib.error.URLError("timed out")
            return 8 * 1024 ** 3

        gc.hf_weight_bytes = mixed
        rc, err = self._main()
        self.assertEqual(rc, 1)
        self.assertEqual(gc.OUTPUT_YAML.read_text(), _PREVIOUS_CATALOG)
        self.assertIn("GONE (HTTP 404)", err)
        self.assertIn("REFUSING", err)

    def test_allow_partial_writes_despite_a_transient_failure(self) -> None:
        # Operator escape hatch: they inspected the loss and accept the
        # smaller catalog. The loss is still itemised on stderr.
        gc.hf_weight_bytes = self._dead_hf_size(503)
        rc, err = self._main("--allow-partial")
        self.assertEqual(rc, 0)
        text = gc.OUTPUT_YAML.read_text()
        self.assertIn('name: "A"', text)
        self.assertNotIn('name: "B"', text)
        self.assertIn("--allow-partial", err)
        self.assertIn("HF size fetch failed: org/B", err)
        self.assertNotIn("REFUSING", err)


class TestPermanentHttpCodeClassifier(unittest.TestCase):
    """`_permanent_http_code` is the single place the two classes split."""

    def test_permanent_codes(self) -> None:
        for code in (401, 403, 404, 410):
            exc = urllib.error.HTTPError("u", code, "m", None, None)
            self.assertEqual(gc._permanent_http_code(exc), code)

    def test_transient_and_non_http_errors(self) -> None:
        for code in (408, 412, 429, 500, 502, 503, 504):
            exc = urllib.error.HTTPError("u", code, "m", None, None)
            self.assertIsNone(gc._permanent_http_code(exc))
        self.assertIsNone(gc._permanent_http_code(urllib.error.URLError("x")))
        self.assertIsNone(gc._permanent_http_code(TimeoutError("x")))
        self.assertIsNone(gc._permanent_http_code(ValueError("bad json")))


if __name__ == "__main__":
    unittest.main()
