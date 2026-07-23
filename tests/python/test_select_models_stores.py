"""scripts/select-models.py: weight-store resolution and gguf disk detection.

Two defects this pins:

1. HF weights were always written to (and looked for in) the vLLM store,
   even for models only SGLang serves. devai-vllm and devai-sglang mount
   SEPARATE volumes, so an SGLang-probed model resolved to a path with no
   weights at serve time. The fix is a two-parter -- an opt-in `--hf-store`
   redirect (never an implicit second copy of hundreds of GB) plus
   `sglang_weight_gaps`, which makes the advertised-but-absent state loud.

2. `is_downloaded` had no `source == "gguf"` branch, so every gguf-sourced
   row read as "not on disk" forever and was re-downloaded on every run --
   and `shadow_ollama_tags` flagged the very tag `pull_gguf` had just
   registered as a prunable hand-made alias.

Stdlib unittest only. The real /var/cache/devai stores are never touched:
every path constant is redirected at a tmpdir.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = REPO_ROOT / "scripts" / "select-models.py"


def _load_module():
    """Load the hyphenated script as an importable module."""
    spec = importlib.util.spec_from_file_location("select_models", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["select_models"] = mod
    spec.loader.exec_module(mod)
    return mod


sm = _load_module()


def _hf_row(name: str, repo: str, sha: str, backends: list[str]) -> dict:
    return {"name": name, "source": "hf", "repo": repo, "sha": sha,
            "backend": backends}


def _cache(entries: dict) -> "sm.HFProbeCaches":
    return sm.HFProbeCaches(vllm={}, sglang=entries)


def _fits_entry(ctx: int = 32768) -> dict:
    return {"probes": {"24": {str(ctx): {"fits": True}}}}


class StoreRedirectTest(unittest.TestCase):
    """--hf-store picks the volume every HF path helper resolves against."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.vllm = root / "vllm"
        self.sglang = root / "sglang"
        (self.vllm / "Present-NVFP4").mkdir(parents=True)
        (self.vllm / "Present-NVFP4" / "config.json").write_text("{}")
        self.sglang.mkdir()
        self._saved_stores = sm.HF_STORES
        self._saved_active = sm.HF_STORE
        sm.HF_STORES = {"vllm": self.vllm, "sglang": self.sglang}
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        sm.HF_STORES = self._saved_stores
        sm.HF_STORE = self._saved_active
        self._tmp.cleanup()

    def test_default_store_is_vllm(self) -> None:
        # Unchanged behaviour for every existing caller: no --hf-store means
        # the vLLM volume, exactly as before.
        sm.HF_STORE = "vllm"
        self.assertEqual(sm.hf_store_dir(), self.vllm)

    def test_on_disk_is_per_store(self) -> None:
        sm.HF_STORE = "vllm"
        self.assertTrue(sm.hf_on_disk("Present-NVFP4"))
        # Same model, SGLang store: absent. This is the whole defect --
        # the two volumes never see each other's weights.
        sm.HF_STORE = "sglang"
        self.assertFalse(sm.hf_on_disk("Present-NVFP4"))

    def test_reclaim_bytes_follows_the_active_store(self) -> None:
        row = {"source": "hf", "name": "Present-NVFP4"}
        sm.HF_STORE = "vllm"
        self.assertGreater(sm.reclaim_bytes(row), 0)
        sm.HF_STORE = "sglang"
        self.assertEqual(sm.reclaim_bytes(row), 0)

    def test_pull_targets_the_active_store(self) -> None:
        seen: list[list[str]] = []
        saved = sm.subprocess.call
        sm.subprocess.call = lambda argv, **kw: (seen.append(argv), 0)[1]
        self.addCleanup(lambda: setattr(sm.subprocess, "call", saved))
        sm.HF_STORE = "sglang"
        sm.pull_hf("New-NVFP4", "org/New-NVFP4")
        self.assertIn(str(self.sglang / "New-NVFP4"), seen[0])
        sm.HF_STORE = "vllm"
        sm.pull_hf("New-NVFP4", "org/New-NVFP4")
        self.assertIn(str(self.vllm / "New-NVFP4"), seen[1])


class SGLangWeightGapTest(unittest.TestCase):
    """A row the SGLang cache says fits, with no weights, must be reported."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.sglang = Path(self._tmp.name) / "sglang"
        self.sglang.mkdir(parents=True)
        self._saved = sm.SGLANG_MODELS
        sm.SGLANG_MODELS = self.sglang
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        sm.SGLANG_MODELS = self._saved
        self._tmp.cleanup()

    def _place(self, name: str) -> None:
        (self.sglang / name).mkdir(parents=True)
        (self.sglang / name / "config.json").write_text("{}")

    def test_fitting_but_absent_is_a_gap(self) -> None:
        row = _hf_row("A-NVFP4", "org/A", "deadbeef", ["vllm", "sglang"])
        gaps = sm.sglang_weight_gaps([row], _cache({"org/A@deadbeef": _fits_entry()}))
        self.assertEqual(gaps, ["A-NVFP4"])

    def test_no_gap_when_weights_present(self) -> None:
        self._place("A-NVFP4")
        row = _hf_row("A-NVFP4", "org/A", "deadbeef", ["vllm", "sglang"])
        gaps = sm.sglang_weight_gaps([row], _cache({"org/A@deadbeef": _fits_entry()}))
        self.assertEqual(gaps, [])

    def test_no_gap_when_cache_has_no_fitting_cell(self) -> None:
        row = _hf_row("A-NVFP4", "org/A", "deadbeef", ["vllm", "sglang"])
        entry = {"probes": {"24": {"32768": {"fits": False}}}}
        self.assertEqual(
            sm.sglang_weight_gaps([row], _cache({"org/A@deadbeef": entry})), [])

    def test_no_gap_when_row_does_not_advertise_sglang(self) -> None:
        row = _hf_row("A-NVFP4", "org/A", "deadbeef", ["vllm"])
        self.assertEqual(
            sm.sglang_weight_gaps([row], _cache({"org/A@deadbeef": _fits_entry()})),
            [])

    def test_unprobed_row_is_not_a_gap(self) -> None:
        # No cache entry at all -> nothing is advertised, nothing to repair.
        row = _hf_row("A-NVFP4", "org/A", "deadbeef", ["sglang"])
        self.assertEqual(sm.sglang_weight_gaps([row], _cache({})), [])

    def test_ollama_and_gguf_rows_are_ignored(self) -> None:
        rows = [{"name": "q:1b", "source": "ollama", "backend": ["ollama"]},
                {"name": "g:1b", "source": "gguf", "backend": ["ollama"]}]
        self.assertEqual(sm.sglang_weight_gaps(rows, _cache({})), [])


class GgufDiskDetectionTest(unittest.TestCase):
    """`ollama create` registers gguf rows under the catalog tag."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.manifests = Path(self._tmp.name) / "library"
        (self.manifests / "ornith").mkdir(parents=True)
        (self.manifests / "ornith" / "9b-q4_k_m").write_text("{}")
        self._saved = sm.OLLAMA_MANIFESTS
        sm.OLLAMA_MANIFESTS = self.manifests
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        sm.OLLAMA_MANIFESTS = self._saved
        self._tmp.cleanup()

    def test_registered_gguf_row_reads_as_downloaded(self) -> None:
        row = {"name": "ornith:9b-q4_k_m", "source": "gguf"}
        self.assertTrue(sm.is_downloaded(row))

    def test_unregistered_gguf_row_reads_as_missing(self) -> None:
        row = {"name": "ornith:9b-q8_0", "source": "gguf"}
        self.assertFalse(sm.is_downloaded(row))

    def test_registered_gguf_row_is_not_a_shadow_tag(self) -> None:
        catalog = [{"name": "ornith:9b-q4_k_m", "source": "gguf"}]
        self.assertEqual(sm.shadow_ollama_tags(catalog), [])

    def test_genuine_alias_is_still_a_shadow_tag(self) -> None:
        # A tag on disk that no catalog row claims stays prunable.
        (self.manifests / "ornith" / "mine").write_text("{}")
        catalog = [{"name": "ornith:9b-q4_k_m", "source": "gguf"}]
        self.assertEqual(sm.shadow_ollama_tags(catalog), ["ornith:mine"])

    def test_delete_routes_gguf_rows_through_ollama_rm(self) -> None:
        seen: list[list[str]] = []
        saved = sm.subprocess.call
        sm.subprocess.call = lambda argv, **kw: (seen.append(argv), 0)[1]
        self.addCleanup(lambda: setattr(sm.subprocess, "call", saved))
        sm.delete({"name": "ornith:9b-q4_k_m", "source": "gguf"})
        self.assertEqual(seen[0][-3:], ["ollama", "rm", "ornith:9b-q4_k_m"])


if __name__ == "__main__":
    unittest.main()
