"""Codex must be able to address every backend the picker offers.

The picker launches Codex with `--oss --local-provider router-<backend>`,
and Codex resolves that name against `[model_providers.*]` in
$CODEX_HOME/config.toml and nothing else. That file is seeded by the
entrypoint ONCE onto the persistent home volume and never overwritten, so
when the fourth backend (vllm-devai, 449270a) was added to `_BACKENDS` the
seed gained nothing an existing home would ever see: every Codex launch on
a vllm-devai row died with

    Error loading configuration: Model provider `router-vllm-devai` not found

(2026-09-22). Two things pin that shut:

1. `_write_codex_providers` re-syncs the `router-*` tables from `_BACKENDS`
   at every launch -- ours are replaced, everything else in the operator's
   file is preserved byte for byte -- mirroring `_write_opencode_providers`.
2. Both seed files (config/codex/config.toml, config/opencode/opencode.json)
   declare every `_BACKENDS` entry on its own router port, so a fresh home
   and a re-synced one agree.

Stdlib unittest only; no cache, no container.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "model_picker_codex", REPO_ROOT / "scripts" / "model-picker.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_picker_codex"] = mod
    spec.loader.exec_module(mod)
    return mod


PICKER = _load()

# The seed exactly as it shipped before vllm-devai existed, i.e. what every
# home created before 2026-09-22 still carries.
STALE_SEED = '''# DevAI Codex configuration.
# (operator comment that must survive)

[model_providers.router-ollama]
name = "Ollama via DevAI router"
base_url = "http://devai-router:11434/v1"

[model_providers.router-vllm]
name = "vLLM via DevAI router"
base_url = "http://devai-router:11435/v1"

[model_providers.router-sglang]
name = "SGLang via DevAI router"
base_url = "http://devai-router:11436/v1"

# Pre-trust the default workdir so codex doesn't prompt on first launch
# inside a fresh shell. Codex will still prompt for any other directory.
[projects."/home/devai/work"]
trust_level = "trusted"
'''


class _CodexHome:
    """Point CODEX_HOME at a temp dir for the duration of a test."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._prev = os.environ.get("CODEX_HOME")
        os.environ["CODEX_HOME"] = self._tmp.name
        return Path(self._tmp.name) / "config.toml"

    def __exit__(self, *exc):
        if self._prev is None:
            os.environ.pop("CODEX_HOME", None)
        else:
            os.environ["CODEX_HOME"] = self._prev
        self._tmp.cleanup()


def _port(url: str) -> str:
    return url.rsplit(":", 1)[1].split("/", 1)[0]


class TestWriteCodexProviders(unittest.TestCase):
    def test_stale_seed_gains_every_backend(self) -> None:
        with _CodexHome() as cfg_path:
            cfg_path.write_text(STALE_SEED)
            PICKER._write_codex_providers()
            cfg = tomllib.loads(cfg_path.read_text())
        self.assertEqual(
            sorted(cfg["model_providers"]),
            sorted(f"router-{b}" for b in PICKER._BACKENDS))
        self.assertIn("router-vllm-devai", cfg["model_providers"])

    def test_each_provider_points_at_its_own_router_port(self) -> None:
        with _CodexHome() as cfg_path:
            PICKER._write_codex_providers()
            cfg = tomllib.loads(cfg_path.read_text())
        for bname, (_label, _reason, port) in PICKER._BACKENDS.items():
            url = cfg["model_providers"][f"router-{bname}"]["base_url"]
            self.assertEqual(_port(url), str(port), f"{bname}: {url}")
            self.assertTrue(url.endswith("/v1"), url)

    def test_operator_content_is_preserved_byte_for_byte(self) -> None:
        with _CodexHome() as cfg_path:
            cfg_path.write_text(STALE_SEED)
            PICKER._write_codex_providers()
            text = cfg_path.read_text()
            cfg = tomllib.loads(text)
        self.assertIn("# (operator comment that must survive)", text)
        self.assertIn('[projects."/home/devai/work"]\ntrust_level = "trusted"', text)
        self.assertEqual(cfg["projects"]["/home/devai/work"]["trust_level"], "trusted")
        # The comment block that documents [projects] sits right after the
        # last router table. The first version of the writer dropped it
        # with that table (found in review, 2026-09-22).
        self.assertIn("# Pre-trust the default workdir so codex doesn't prompt on first launch\n"
                      "# inside a fresh shell. Codex will still prompt for any other directory.\n"
                      '[projects."/home/devai/work"]', text)

    def test_shipped_seed_round_trips_with_its_comments(self) -> None:
        seed = (REPO_ROOT / "config" / "codex" / "config.toml").read_text()
        with _CodexHome() as cfg_path:
            cfg_path.write_text(seed)
            PICKER._write_codex_providers()
            text = cfg_path.read_text()
        self.assertEqual(tomllib.loads(text), tomllib.loads(seed))
        for line in seed.splitlines():
            if line.startswith("#"):
                self.assertIn(line, text, f"comment lost: {line!r}")

    def test_comment_inside_our_table_goes_with_it(self) -> None:
        with _CodexHome() as cfg_path:
            cfg_path.write_text(
                '[model_providers.router-vllm]\n# stale note about our table\n'
                'name = "old"\nbase_url = "http://devai-router:1/v1"\n')
            PICKER._write_codex_providers()
            text = cfg_path.read_text()
        self.assertNotIn("stale note", text)

    def test_trailing_operator_comment_after_our_tables_is_kept(self) -> None:
        with _CodexHome() as cfg_path:
            cfg_path.write_text(STALE_SEED.split("# Pre-trust")[0]
                                + "# operator note at the very end\n")
            PICKER._write_codex_providers()
            text = cfg_path.read_text()
        self.assertIn("# operator note at the very end", text)

    def test_single_quoted_table_keys_are_ours_too(self) -> None:
        with _CodexHome() as cfg_path:
            cfg_path.write_text(
                "[model_providers.'router-vllm']\nname = \"old\"\n"
                'base_url = "http://devai-router:1/v1"\n')
            PICKER._write_codex_providers()
            cfg = tomllib.loads(cfg_path.read_text())
        self.assertEqual(
            _port(cfg["model_providers"]["router-vllm"]["base_url"]),
            str(PICKER._BACKENDS["vllm"][2]))

    def test_inline_router_entry_under_bare_table_fails_loud(self) -> None:
        # `[model_providers]` with `router-vllm = {...}` inline is not a
        # header we own, so it survives, collides with our appended table
        # and must be refused rather than written.
        inline = ('[model_providers]\n'
                  'router-vllm = { name = "x", base_url = "http://devai-router:1/v1" }\n')
        with _CodexHome() as cfg_path:
            cfg_path.write_text(inline)
            with self.assertRaises(SystemExit):
                PICKER._write_codex_providers()
            self.assertEqual(cfg_path.read_text(), inline)

    def test_replaces_our_tables_rather_than_accumulating(self) -> None:
        stale = STALE_SEED + '''
[model_providers.router-gone]
name = "a backend that no longer exists"
base_url = "http://devai-router:19999/v1"

[model_providers.mine]
name = "the operator's own provider"
base_url = "http://example.invalid/v1"
'''
        with _CodexHome() as cfg_path:
            cfg_path.write_text(stale)
            PICKER._write_codex_providers()
            first = cfg_path.read_text()
            PICKER._write_codex_providers()
            second = cfg_path.read_text()
            cfg = tomllib.loads(second)
        self.assertNotIn("router-gone", cfg["model_providers"])
        self.assertIn("mine", cfg["model_providers"])
        self.assertEqual(cfg["model_providers"]["mine"]["base_url"],
                         "http://example.invalid/v1")
        self.assertEqual(first, second, "the rewrite must be idempotent")
        self.assertEqual(second.count("[model_providers.router-vllm]"), 1)

    def test_quoted_table_keys_are_ours_too(self) -> None:
        with _CodexHome() as cfg_path:
            cfg_path.write_text(
                '[model_providers."router-vllm"]\nname = "old"\n'
                'base_url = "http://devai-router:1/v1"\n')
            PICKER._write_codex_providers()
            cfg = tomllib.loads(cfg_path.read_text())
        self.assertEqual(
            _port(cfg["model_providers"]["router-vllm"]["base_url"]),
            str(PICKER._BACKENDS["vllm"][2]))

    def test_absent_file_is_created(self) -> None:
        with _CodexHome() as cfg_path:
            self.assertFalse(cfg_path.exists())
            PICKER._write_codex_providers()
            cfg = tomllib.loads(cfg_path.read_text())
        self.assertIn("router-ollama", cfg["model_providers"])

    def test_unparseable_result_is_refused_not_written(self) -> None:
        broken = 'this = is = not toml\n'
        with _CodexHome() as cfg_path:
            cfg_path.write_text(broken)
            with self.assertRaises(SystemExit):
                PICKER._write_codex_providers()
            self.assertEqual(cfg_path.read_text(), broken)

    def test_build_codex_syncs_before_launch(self) -> None:
        with _CodexHome() as cfg_path:
            cmd = PICKER._build("codex", "Qwen3.8-27B-MTP-devai-NVFP4::mtp@118784",
                                "vllm-devai")
            cfg = tomllib.loads(cfg_path.read_text())
        self.assertIn("--local-provider", cmd)
        provider = cmd[cmd.index("--local-provider") + 1]
        self.assertEqual(provider, "router-vllm-devai")
        self.assertIn(provider, cfg["model_providers"])

    def test_build_codex_declares_the_context_window(self) -> None:
        # Codex's catalog does not know our ids and falls back to NO
        # window; `model_context_window` is honoured (2026-09-22, 0.155.1,
        # RUST_LOG=debug shows context_window=118784). The catalog
        # warning itself cannot be silenced.
        prev = os.environ.get("CONTEXT")
        try:
            os.environ["CONTEXT"] = "118784"
            with _CodexHome():
                cmd = PICKER._build("codex", "m@118784", "vllm-devai")
            self.assertIn("model_context_window=118784", cmd)
            self.assertEqual(cmd[cmd.index("model_context_window=118784") - 1], "-c")
            os.environ.pop("CONTEXT", None)
            with _CodexHome():
                cmd = PICKER._build("codex", "m@118784", "vllm-devai")
            self.assertFalse(any(a.startswith("model_context_window") for a in cmd))
        finally:
            if prev is None:
                os.environ.pop("CONTEXT", None)
            else:
                os.environ["CONTEXT"] = prev


class TestSeedsDeclareEveryBackend(unittest.TestCase):
    def test_codex_seed(self) -> None:
        cfg = tomllib.loads((REPO_ROOT / "config" / "codex" / "config.toml").read_text())
        for bname, (_label, _reason, port) in PICKER._BACKENDS.items():
            url = cfg["model_providers"][f"router-{bname}"]["base_url"]
            self.assertEqual(_port(url), str(port), f"{bname}: {url}")

    def test_opencode_seed(self) -> None:
        cfg = json.loads((REPO_ROOT / "config" / "opencode" / "opencode.json").read_text())
        for bname, (_label, _reason, port) in PICKER._BACKENDS.items():
            url = cfg["provider"][f"router-{bname}"]["options"]["baseURL"]
            self.assertEqual(_port(url), str(port), f"{bname}: {url}")


if __name__ == "__main__":
    unittest.main()
