"""The picker's recipe badge (plan Phase 17a, from odysseus CB-3).

A row whose model has an official vLLM launch recipe
(deploy/recipe-seeds.yaml, `make fetch-vllm-recipes`) shows `rcp` in the RCP
column beside TOOLS, and its preview names the recipe. The TOOLS value
itself is untouched: it stays the benched score or the probe-verified parser
presence, because a recipe only seeds a probe. Only vLLM-engine rows carry
the badge (a recipe is vLLM's command). The picker reads the YAML itself and
imports no new module: it runs bind-mounted inside older images
(tests/python/test_picker_container_imports.py).

Stdlib unittest only; no fzf.
"""

from __future__ import annotations

import importlib.util
import re
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PICKER = REPO_ROOT / "scripts" / "model-picker.py"


def _load_picker():
    spec = importlib.util.spec_from_file_location("model_picker_recipe_badge", PICKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SEEDS = """\
version: 1
commit: 0123456789abcdef0123456789abcdef01234567
seeds:
  M-NVFP4:
    recipe: models/Org/M.yaml
    commit: 0123456789abcdef0123456789abcdef01234567
    match: base-name
    min_vllm_version: "0.17.0"
    parsers: {tool: hermes}
    engine_flags: []
    engine_env: {}
    vllm_version: {vllm: "0.22.1", vllm-devai: "0.28.0"}
    applies_to: {vllm: true, vllm-devai: true}
  New-NVFP4:
    recipe: models/Org/New.yaml
    commit: 0123456789abcdef0123456789abcdef01234567
    match: exact
    min_vllm_version: "0.25.0"
    parsers: {}
    engine_flags: []
    engine_env: {}
    vllm_version: {vllm: "0.22.1", vllm-devai: "0.28.0"}
    applies_to: {vllm: false, vllm-devai: true}
  Old-Seed:
    recipe: models/Org/Old.yaml
    commit: 0123456789abcdef0123456789abcdef01234567
    match: exact
    min_vllm_version: "0.17.0"
    parsers: {}
    engine_flags: []
    engine_env: {}
"""


def _row(backend: str, name: str = "M-NVFP4") -> dict:
    return {"name": name, "backend": backend, "capability": "structured",
            "vram": {}, "details": {"quantization": "NVFP4"}}


class RecipeBadgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mp = _load_picker()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / "recipe-seeds.yaml"
        path.write_text(SEEDS)
        self.mp._RECIPE_SEED_PATHS = [str(path)]
        self.mp._RECIPE_SEEDS = None

    def test_a_vllm_row_with_a_recipe_carries_the_badge(self) -> None:
        for backend in ("vllm", "vllm-devai"):
            with self.subTest(backend=backend):
                self.assertIsNotNone(self.mp._recipe_for(_row(backend)))
                self.assertRegex(self.mp._format_model_row(_row(backend), 1), r"\brcp\b")

    def test_other_rows_do_not(self) -> None:
        self.assertIsNone(self.mp._recipe_for(_row("sglang")))
        self.assertIsNone(self.mp._recipe_for(_row("ollama")))
        self.assertIsNone(self.mp._recipe_for(_row("vllm", name="Other")))
        self.assertNotRegex(self.mp._format_model_row(_row("sglang"), 1), r"\brcp\b")

    def test_the_badge_shows_only_where_the_seed_applies(self) -> None:
        # min 0.25.0: not on stock vLLM 0.22.1, yes on build-vllm.sh's 0.28.0.
        self.assertNotRegex(self.mp._format_model_row(_row("vllm", "New-NVFP4"), 1), r"\brcp\b")
        self.assertRegex(self.mp._format_model_row(_row("vllm-devai", "New-NVFP4"), 1), r"\brcp\b")
        line = self.mp._recipe_line(_row("vllm", "New-NVFP4"))
        self.assertIn("does NOT apply here: this backend runs vLLM 0.22.1", line)
        # A seed written before `applies_to` existed shows no badge, and says why.
        self.assertNotRegex(self.mp._format_model_row(_row("vllm", "Old-Seed"), 1), r"\brcp\b")
        self.assertIn("run make fetch-vllm-recipes", self.mp._recipe_line(_row("vllm", "Old-Seed")))

    def test_the_header_names_the_column_beside_tools(self) -> None:
        lines, _, _ = self.mp._build_menu([_row("vllm")])
        self.assertRegex(lines[0], r"TOOLS\S*\s+RCP\s+MTP")
        self.assertIn("RCP = vLLM recipe applies", lines[2])

    def test_the_preview_names_the_recipe_as_a_seed(self) -> None:
        line = self.mp._recipe_line(_row("vllm-devai"))
        self.assertIn("models/Org/M.yaml at 0123456789ab", line)
        self.assertIn("not probe-verified", line)
        self.assertEqual(self.mp._recipe_line(_row("sglang")), "")

    def test_a_malformed_or_missing_file_means_no_badges(self) -> None:
        bad = Path(self.tmp.name) / "bad.yaml"
        bad.write_text("version: [\n")
        binary = Path(self.tmp.name) / "binary.yaml"
        binary.write_bytes(b"\xff\xfe\x00version")
        for paths in ([str(bad)], [str(binary)], [str(Path(self.tmp.name) / "absent.yaml")]):
            self.mp._RECIPE_SEED_PATHS = paths
            self.mp._RECIPE_SEEDS = None
            self.assertIsNone(self.mp._recipe_for(_row("vllm")))

    def test_container_path_first_and_no_new_import(self) -> None:
        mp = _load_picker()
        self.assertEqual(mp._RECIPE_SEED_PATHS[0], "/etc/devai/recipe-seeds.yaml")
        self.assertTrue(mp._RECIPE_SEED_PATHS[1].endswith("deploy/recipe-seeds.yaml"))
        src = PICKER.read_text()
        self.assertIsNone(re.search(r"^\s*(import|from)\s+_vllm_recipes", src, re.M))


if __name__ == "__main__":
    unittest.main()
