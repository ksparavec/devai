"""Picker joins on-disk Ollama tags to the catalog case-insensitively.

Ollama canonicalizes a k-quant tag's case on `ollama create` (a catalog
`ornith:9b-q4_k_m` is stored on disk as `ornith:9b-q4_K_M`), while
generate-catalog lowercases every gguf-derived tag. Ollama tags are
case-insensitive, so `_discover_models` must still attach the catalog
family/purpose metadata to the on-disk (canonicalized-case) row -- otherwise
gguf_repos k-quant rows show a blank family column.

Stdlib unittest only. Run with:
    python3 -m unittest tests.python.test_model_picker_ollama_case
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
        "_picker_under_test_ollama_case",
        str(REPO_ROOT / "scripts" / "model-picker.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


PICKER = _load_picker()


class TestOllamaCaseInsensitiveCatalogJoin(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)

        # On-disk manifest tag: Ollama's canonicalized (mixed) case.
        manif = root / "manifests" / "ornith"
        manif.mkdir(parents=True)
        (manif / "9b-q4_K_M").write_text("{}")

        # Catalog: lowercase name, exactly as generate-catalog emits.
        cat = root / "models.yaml"
        cat.write_text(
            'models:\n'
            '  - name: "ornith:9b-q4_k_m"\n'
            '    family: ornith\n'
            '    source: gguf\n'
            '    size: "5.24 GB"\n'
            '    purpose: "ornith gguf row"\n'
        )

        # Probe cache: alias mirrors the on-disk (mixed-case) tag.
        cache = root / ".ollama-reasoning-cache.json"
        cache.write_text(json.dumps({
            "9f73f59457a2f1e1": {
                "schema_version": 3,
                "aliases": ["ornith:9b-q4_K_M"],
                "capability": "structured",
                "probes": {"24": {"32768": {"fits": True, "fully_on_gpu": True}}},
            }
        }))

        self._saved = {}
        patches = {
            "_OLLAMA_MANIFESTS": str(root / "manifests"),
            "_CATALOG_PATHS": [str(cat)],
            "_PROBE_CACHE_PATHS": [str(cache)],
            "_VLLM_PROBE_CACHE_PATHS": [str(root / "no-vllm")],
            "_SGLANG_PROBE_CACHE_PATHS": [str(root / "no-sglang")],
            "_VLLM_DIR": str(root / "no-vllm-dir"),
        }
        for attr, val in patches.items():
            self._saved[attr] = getattr(PICKER, attr)
            setattr(PICKER, attr, val)
        # Avoid parsing the stub manifest for its byte size.
        self._saved["_ollama_disk_size_gb"] = PICKER._ollama_disk_size_gb
        PICKER._ollama_disk_size_gb = lambda lib, tag: 5.24

    def tearDown(self) -> None:
        for attr, val in self._saved.items():
            setattr(PICKER, attr, val)
        self.tmp.cleanup()

    def test_lowercase_catalog_joins_canonicalized_ondisk_tag(self) -> None:
        # Sanity: the exact-case catalog lookup misses (this was the bug).
        catalog = PICKER._load_catalog()
        self.assertIsNone(catalog.get("ornith:9b-q4_K_M"))

        rows = PICKER._discover_models()
        ornith = [r for r in rows if r["name"].lower().startswith("ornith:")]
        self.assertEqual(len(ornith), 1)
        row = ornith[0]
        self.assertEqual(row["name"], "ornith:9b-q4_K_M")   # on-disk tag
        # The fix: catalog family/purpose still attach despite the case diff.
        self.assertEqual(row["family"], "ornith")
        self.assertEqual(row["purpose"], "ornith gguf row")


if __name__ == "__main__":
    unittest.main()
