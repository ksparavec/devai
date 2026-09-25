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
import os
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
        self._saved = sm.SGLANG_STORE
        sm.SGLANG_STORE = self.sglang
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        sm.SGLANG_STORE = self._saved
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


class StorageLayoutTest(unittest.TestCase):
    """ONE root, THREE backend stores, every path written out in full.

    GGUF staging used to be `VLLM_MODELS.parent / "_gguf"`. That was only
    correct while the vLLM store happened to sit inside the Ollama tree. The
    2026-07-17 storage refactor moved the vLLM store to its own directory,
    the derived path silently followed it to /var/cache/devai/_gguf, and the
    first GGUF pull afterwards (2026-09-19) downloaded 43 GiB to a place the
    Ollama container cannot see, then failed at `ollama create`. A path that
    is stated cannot drift like that; a path that is computed from another
    backend's path can. So nothing here may be derived or relocatable.
    """

    def test_root_is_var_cache_devai(self) -> None:
        self.assertEqual(sm.DEVAI_ROOT, Path("/var/cache/devai"))

    def test_the_three_backend_stores(self) -> None:
        self.assertEqual(sm.OLLAMA_STORE, Path("/var/cache/devai/ollama"))
        self.assertEqual(sm.VLLM_STORE, Path("/var/cache/devai/vllm"))
        self.assertEqual(sm.SGLANG_STORE, Path("/var/cache/devai/sglang"))

    def test_the_laya_store(self) -> None:
        # Not a backend store for an engine: the laya trainer's base
        # checkpoints, datasets and runs. A plain directory by owner decision
        # (2026-09-24), but written out like the others all the same.
        self.assertEqual(sm.LAYA_STORE, Path("/var/cache/devai/laya"))
        self.assertEqual(sm.LAYA_BASE, Path("/var/cache/devai/laya/base"))
        self.assertEqual(sm.LAYA_STAGING, Path("/var/cache/devai/laya/.staging"))
        self.assertEqual(sm.LAYA_CATALOG, REPO_ROOT / "deploy" / "laya-models.yaml")

    def test_hf_stores_are_exactly_the_vllm_and_sglang_stores(self) -> None:
        # Three HF backends, TWO directories: vllm-devai (the home-built
        # vLLM image) serves the vLLM store. No fourth directory exists.
        self.assertEqual(sm.HF_STORES,
                         {"vllm": sm.VLLM_STORE, "sglang": sm.SGLANG_STORE,
                          "vllm-devai": sm.VLLM_STORE})
        self.assertEqual(len(set(sm.HF_STORES.values())), 2)

    def test_gguf_staging_is_inside_the_ollama_store(self) -> None:
        self.assertEqual(sm.GGUF_STAGING,
                         Path("/var/cache/devai/ollama/models/_gguf"))
        # The property that actually matters: devai-ollama mounts only
        # OLLAMA_STORE, so staging must map to an in-container path.
        self.assertEqual(sm.to_container_path(sm.GGUF_STAGING),
                         "/root/.ollama/models/_gguf")

    def test_ollama_manifests_and_blobs_are_inside_the_ollama_store(self) -> None:
        self.assertEqual(sm.OLLAMA_MANIFESTS_ROOT,
                         Path("/var/cache/devai/ollama/models/manifests"))
        self.assertEqual(sm.OLLAMA_BLOBS,
                         Path("/var/cache/devai/ollama/models/blobs"))
        self.assertEqual(
            sm.OLLAMA_MANIFESTS,
            Path("/var/cache/devai/ollama/models/manifests/"
                 "registry.ollama.ai/library"))

    def test_environment_cannot_relocate_a_store(self) -> None:
        # These variables used to move the stores. A store that can be moved
        # from outside is a store nobody can point at with confidence.
        moved = {"VLLM_MODELS_DIR": "/tmp/elsewhere/vllm",
                 "SGLANG_MODELS_DIR": "/tmp/elsewhere/sglang",
                 "OLLAMA_HOST_ROOT": "/tmp/elsewhere/ollama",
                 "OLLAMA_MANIFESTS_DIR": "/tmp/elsewhere/manifests",
                 "DEVAI_ROOT": "/tmp/elsewhere"}
        saved = {k: os.environ.get(k) for k in moved}
        os.environ.update(moved)
        try:
            spec = importlib.util.spec_from_file_location("sm_env", _SCRIPT)
            fresh = importlib.util.module_from_spec(spec)
            # @dataclass resolves cls.__module__ through sys.modules, so the
            # module has to be registered before its body runs.
            sys.modules["sm_env"] = fresh
            spec.loader.exec_module(fresh)
        finally:
            sys.modules.pop("sm_env", None)
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertEqual(fresh.DEVAI_ROOT, Path("/var/cache/devai"))
        self.assertEqual(fresh.OLLAMA_STORE, Path("/var/cache/devai/ollama"))
        self.assertEqual(fresh.VLLM_STORE, Path("/var/cache/devai/vllm"))
        self.assertEqual(fresh.SGLANG_STORE, Path("/var/cache/devai/sglang"))
        self.assertEqual(fresh.GGUF_STAGING,
                         Path("/var/cache/devai/ollama/models/_gguf"))
        self.assertEqual(fresh.LAYA_STORE, Path("/var/cache/devai/laya"))

    def test_makefile_does_not_pretend_to_steer_the_stores(self) -> None:
        # The script ignores these variables, so a recipe that passes them
        # to it tells the reader a lie: that the stores can be moved from
        # the Makefile. Only the two recipes that run select-models.py are
        # checked -- the probers, bench and compose still read them.
        mk = (REPO_ROOT / "Makefile").read_text()
        for target in ("model-fit:", "model-pull:"):
            block = mk.split("\n" + target, 1)[1].split("\n\n", 1)[0]
            # What the recipe EXECUTES: drop the target line and `@#` comments.
            recipe = "\n".join(
                ln for ln in block.splitlines()[1:]
                if not ln.lstrip().startswith(("@#", "#")))
            self.assertIn("scripts/select-models.py", recipe, target)
            for var in ("VLLM_MODELS_DIR", "SGLANG_MODELS_DIR",
                        "OLLAMA_HOST_ROOT", "DEVAI_ROOT"):
                self.assertNotIn(var, recipe,
                                 f"{target} passes {var} to a script that "
                                 f"ignores it")

    def test_no_store_path_is_derived_with_parent(self) -> None:
        # `.parent` is how the staging path drifted. The one legitimate use
        # locates the repo from this file's own location, which is not a
        # backend store.
        offenders = [
            f"{n}: {line.strip()}"
            for n, line in enumerate(_SCRIPT.read_text().splitlines(), 1)
            if ".parent" in line
            and not line.lstrip().startswith("#")
            and "REPO_ROOT = Path(__file__)" not in line
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
