"""GGUF imports get their RENDERER / PARSER from Ollama's registry, not a guess.

`pull_gguf` writes a Modelfile with RENDERER and PARSER directives; without
working ones Ollama treats the import as a raw completion engine and refuses
tool calls. The script used to write the catalog FAMILY name for both, on the
stated assumption that "the renderer/parser names are 1:1 with the family
name". That held for every family until qwen3.8, whose official library
config says:

    renderer = "qwen3.8"      parser = "qwen3.5"

So `PARSER qwen3.8` named a parser that does not exist. `ollama create`
still printed "success"; the model came up with capability `completion`
only, and a tool call returned 400 "does not support tools" -- on Ollama
0.31.1 AND on 0.34.2, so upgrading the runtime did not help. Measured
2026-09-19.

The fix: the catalog generator reads the real names once per family from one
tag of the family's Ollama library and stamps them on that family's gguf
rows; pull_gguf uses the row's values. A family with no Ollama library has
nothing to consult and keeps the family-name fallback.

Stdlib unittest only; no network, no container.
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
_SCRIPTS = REPO_ROOT / "scripts"


def _load(filename: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, _SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


gc = _load("generate-catalog.py", "generate_catalog_directives")
sm = _load("select-models.py", "select_models_directives")

_MIN_CONFIG = {"num_hidden_layers": 4, "num_attention_heads": 4,
               "num_key_value_heads": 2, "hidden_size": 64}


class RegistryLookupTest(unittest.TestCase):
    """ollama_modelfile_directives reads manifest -> config blob."""

    def setUp(self) -> None:
        self._saved = gc._http_json
        self.addCleanup(lambda: setattr(gc, "_http_json", self._saved))

    def test_reads_renderer_and_parser_from_the_config_blob(self) -> None:
        seen: list[str] = []

        def fake(url: str, accept: str = "", timeout: int = 25) -> dict:
            seen.append(url)
            if "/manifests/" in url:
                return {"config": {"digest": "sha256:abc"}, "layers": []}
            return {"renderer": "qwen3.8", "parser": "qwen3.5"}

        gc._http_json = fake
        self.assertEqual(gc.ollama_modelfile_directives("qwen3.8", "27b-q4_K_M"),
                         ("qwen3.8", "qwen3.5"))
        self.assertTrue(seen[0].endswith("/qwen3.8/manifests/27b-q4_K_M"))
        self.assertTrue(seen[1].endswith("/qwen3.8/blobs/sha256:abc"))

    def test_absent_fields_come_back_as_none(self) -> None:
        gc._http_json = lambda url, accept="", timeout=25: (
            {"config": {"digest": "sha256:abc"}} if "/manifests/" in url else {})
        self.assertEqual(gc.ollama_modelfile_directives("old", "7b"),
                         (None, None))


class CatalogStampsGgufRowsTest(unittest.TestCase):
    _PATCHED = ("FAMILIES_YAML", "OUTPUT_YAML", "hf_config", "ollama_tags",
                "ollama_manifest_size", "hf_gguf_files",
                "ollama_modelfile_directives")

    def setUp(self) -> None:
        self._saved = {k: getattr(gc, k) for k in self._PATCHED}
        gc._row_loss.clear()
        gc._permanent_skips.clear()
        self._tmp = tempfile.TemporaryDirectory()
        d = Path(self._tmp.name)
        gc.FAMILIES_YAML = d / "families.yaml"
        gc.OUTPUT_YAML = d / "models.yaml"
        gc.OUTPUT_YAML.write_text("models: []\n")
        gc.hf_config = lambda repo: dict(_MIN_CONFIG)
        gc.ollama_manifest_size = lambda lib, tag: 16 * 1024 ** 3
        gc.hf_gguf_files = lambda repo: [
            {"filename": "Model-27B-UD-Q4_K_S.gguf", "size_bytes": 14 * 1024 ** 3}]
        self.lookups: list[tuple[str, str]] = []

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            setattr(gc, k, v)
        gc._row_loss.clear()
        gc._permanent_skips.clear()
        self._tmp.cleanup()

    def _run(self, family: dict) -> tuple[int, dict, str]:
        gc.FAMILIES_YAML.write_text(yaml.safe_dump({"families": [family]}))
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = gc.main([])
        rows = {r["name"]: r for r in
                (yaml.safe_load(gc.OUTPUT_YAML.read_text()) or {}).get("models") or []}
        return rc, rows, err.getvalue()

    def _family(self, **extra) -> dict:
        fam = {"name": "qwen3.8", "arch_ref": "org/ref",
               "gguf_repos": [{"repo": "unsloth/Model-27B-GGUF",
                               "tag_prefix": "27b", "include": ["UD-Q4_K_S"]}]}
        fam.update(extra)
        return fam

    def test_gguf_rows_carry_the_registry_names(self) -> None:
        gc.ollama_tags = lambda lib: ["27b-q4_K_M", "27b-q8_0"]

        def lookup(lib: str, tag: str):
            self.lookups.append((lib, tag))
            return ("qwen3.8", "qwen3.5")

        gc.ollama_modelfile_directives = lookup
        rc, rows, err = self._run(self._family(ollama_repos=["qwen3.8"]))
        self.assertEqual(rc, 0, err)
        row = rows["qwen3.8:27b-ud-q4_k_s"]
        self.assertEqual((row["renderer"], row["parser"]), ("qwen3.8", "qwen3.5"))
        # ONE lookup per family, not one per tag: the registry is slow.
        self.assertEqual(len(self.lookups), 1)

    def test_platform_gated_tag_is_skipped_for_the_next_one(self) -> None:
        gc.ollama_tags = lambda lib: ["27b-mlx", "27b-q4_K_M"]

        def lookup(lib: str, tag: str):
            self.lookups.append((lib, tag))
            if tag == "27b-mlx":
                raise urllib.error.HTTPError("u", 412, "gated", None, None)
            return ("qwen3.8", "qwen3.5")

        gc.ollama_modelfile_directives = lookup
        rc, rows, err = self._run(self._family(ollama_repos=["qwen3.8"]))
        self.assertEqual(rc, 0, err)
        self.assertEqual(rows["qwen3.8:27b-ud-q4_k_s"]["parser"], "qwen3.5")

    def test_family_without_an_ollama_library_is_left_alone(self) -> None:
        gc.ollama_tags = lambda lib: self.fail("no library to list")
        gc.ollama_modelfile_directives = lambda lib, tag: self.fail("no lookup")
        rc, rows, err = self._run(self._family())
        self.assertEqual(rc, 0, err)
        row = rows["qwen3.8:27b-ud-q4_k_s"]
        self.assertNotIn("renderer", row)
        self.assertNotIn("parser", row)

    def test_failed_lookup_refuses_to_write_the_catalog(self) -> None:
        # A gguf row without its names would be imported on the family-name
        # guess -- the very failure this exists to prevent. Refuse instead.
        gc.ollama_tags = lambda lib: ["27b-q4_K_M"]

        def lookup(lib: str, tag: str):
            raise TimeoutError("registry timed out")

        gc.ollama_modelfile_directives = lookup
        rc, _rows, _err = self._run(self._family(ollama_repos=["qwen3.8"]))
        self.assertEqual(rc, 1)
        self.assertEqual(gc.OUTPUT_YAML.read_text(), "models: []\n")


class PullGgufUsesTheRowTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.stage = Path(self._tmp.name) / "ollama" / "models" / "_gguf"
        self._saved = (sm.GGUF_STAGING, sm.OLLAMA_STORE, sm.subprocess.call)
        sm.GGUF_STAGING = self.stage
        sm.OLLAMA_STORE = Path(self._tmp.name) / "ollama"
        self.addCleanup(self._restore)

        def fake_call(argv, **kw):
            if argv[1] == "download":      # hf download <repo> <file> --local-dir <d>
                d = Path(argv[argv.index("--local-dir") + 1])
                d.mkdir(parents=True, exist_ok=True)
                (d / argv[3]).write_bytes(b"gguf")
            return 0

        sm.subprocess.call = fake_call

    def _restore(self) -> None:
        sm.GGUF_STAGING, sm.OLLAMA_STORE, sm.subprocess.call = self._saved

    def _modelfile(self) -> str:
        return (self.stage / "unsloth_M-GGUF" / "Modelfile.M.gguf").read_text()

    def test_row_names_are_written_not_the_family_name(self) -> None:
        sm.pull({"name": "qwen3.8:27b-x", "source": "gguf", "family": "qwen3.8",
                 "repo": "unsloth/M-GGUF", "gguf_filename": "M.gguf",
                 "renderer": "qwen3.8", "parser": "qwen3.5"})
        mf = self._modelfile()
        self.assertIn("RENDERER qwen3.8\n", mf)
        self.assertIn("PARSER qwen3.5\n", mf)
        self.assertNotIn("PARSER qwen3.8", mf)

    def test_row_without_names_falls_back_to_the_family(self) -> None:
        sm.pull({"name": "ornith:9b-x", "source": "gguf", "family": "ornith",
                 "repo": "unsloth/M-GGUF", "gguf_filename": "M.gguf"})
        mf = self._modelfile()
        self.assertIn("RENDERER ornith\n", mf)
        self.assertIn("PARSER ornith\n", mf)


if __name__ == "__main__":
    unittest.main()
