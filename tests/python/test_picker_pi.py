"""Pi is wired like OpenCode: its declared model list MIRRORS the vetted set.

`_write_pi_models` rebuilds the `router-*` providers in
~/.pi/agent/models.json from _BACKENDS and the vetted set at every launch
(replace, never merge -- the file lives on the persistent home volume) and
preserves everything else. Pi-specific on top of that:

- The chosen id is ALWAYS declared exactly. Pi treats `--model` as a
  pattern and fuzzy-matches one that is not an exact declared id, reading a
  trailing `:<level>` as a thinking level -- on 0.87.1 `qwen3.5:high`
  silently became `qwen3.5:9b-q8_0`.
- Each declared id carries a `contextWindow` where one is known (the
  `@<ctx>` suffix, or the tier the picker resolved for the chosen model):
  Pi defaults to 128K and compacts against that.
- The seed and `_build` both use one provider per router port.

Stdlib unittest only; no probe/bench cache required.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "model_picker_pi", REPO_ROOT / "scripts" / "model-picker.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_picker_pi"] = mod
    spec.loader.exec_module(mod)
    return mod


PICKER = _load()


class _PiHome:
    """Point PI_CODING_AGENT_DIR (and CONTEXT) at test values."""

    def __init__(self, context: str | None = None) -> None:
        self._context = context

    def __enter__(self) -> Path:
        self._tmp = tempfile.TemporaryDirectory()
        self._prev = {k: os.environ.get(k) for k in ("PI_CODING_AGENT_DIR", "CONTEXT")}
        os.environ["PI_CODING_AGENT_DIR"] = self._tmp.name
        if self._context is None:
            os.environ.pop("CONTEXT", None)
        else:
            os.environ["CONTEXT"] = self._context
        return Path(self._tmp.name) / "models.json"

    def __exit__(self, *exc) -> None:
        for k, v in self._prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()


VETTED = {
    "ollama": ["qwen3.6:27b-q4_K_M"],
    "vllm": ["Qwen3-8B-NVFP4@32768", "gpt-oss-20b@131072"],
    "sglang": ["gpt-oss-20b@131072"],
}


def _ids(cfg: dict, provider: str) -> list[str]:
    return [m["id"] for m in cfg["providers"][provider]["models"]]


def _windows(cfg: dict, provider: str) -> dict[str, int | None]:
    return {m["id"]: m.get("contextWindow") for m in cfg["providers"][provider]["models"]}


class TestWritePiModels(unittest.TestCase):
    def test_declares_every_backend_as_its_own_provider(self) -> None:
        with _PiHome() as path:
            PICKER._write_pi_models(VETTED, "vllm", "Qwen3-8B-NVFP4@32768", 32768)
            cfg = json.loads(path.read_text())
        self.assertEqual(
            sorted(p for p in cfg["providers"] if p.startswith("router-")),
            ["router-ollama", "router-sglang", "router-vllm", "router-vllm-devai"])
        self.assertEqual(sorted(_ids(cfg, "router-vllm")),
                         ["Qwen3-8B-NVFP4@32768", "gpt-oss-20b@131072"])
        self.assertEqual(_ids(cfg, "router-sglang"), ["gpt-oss-20b@131072"])
        for prov in cfg["providers"].values():
            self.assertEqual(prov["api"], "openai-completions")
            self.assertEqual(prov["apiKey"], "local")

    def test_each_provider_points_at_its_own_router_port(self) -> None:
        with _PiHome() as path:
            PICKER._write_pi_models(VETTED, "vllm", "Qwen3-8B-NVFP4@32768", 32768)
            cfg = json.loads(path.read_text())
        for bname, (_label, _reason, port) in PICKER._BACKENDS.items():
            self.assertEqual(cfg["providers"][f"router-{bname}"]["baseUrl"],
                             f"http://devai-router:{port}/v1")

    def test_replaces_rather_than_accumulates(self) -> None:
        with _PiHome() as path:
            PICKER._write_pi_models(VETTED, "vllm", "Qwen3-8B-NVFP4@32768", 32768)
            shrunk = {**VETTED, "vllm": ["Qwen3-8B-NVFP4@32768"]}
            PICKER._write_pi_models(shrunk, "vllm", "Qwen3-8B-NVFP4@32768", 32768)
            cfg = json.loads(path.read_text())
        self.assertEqual(_ids(cfg, "router-vllm"), ["Qwen3-8B-NVFP4@32768"])

    def test_preserves_user_config(self) -> None:
        with _PiHome() as path:
            path.write_text(json.dumps({
                "modelOverrides": {"x": {}},
                "providers": {
                    "my-openai": {"baseUrl": "https://example.invalid/v1",
                                  "models": [{"id": "gpt-4"}]},
                    "router-vllm": {"models": [{"id": "stale-model@1"}]},
                },
            }))
            PICKER._write_pi_models(VETTED, "vllm", "Qwen3-8B-NVFP4@32768", 32768)
            cfg = json.loads(path.read_text())
        self.assertEqual(cfg["modelOverrides"], {"x": {}})
        self.assertEqual(cfg["providers"]["my-openai"]["models"], [{"id": "gpt-4"}])
        self.assertNotIn("stale-model@1", _ids(cfg, "router-vllm"))

    def test_chosen_id_declared_exactly_even_with_suffixes(self) -> None:
        """An undeclared `--model` is fuzzy-matched by Pi -- never allow it."""
        chosen = "qwen3.6:27b-q4_K_M::nothink"
        with _PiHome() as path:
            PICKER._write_pi_models(VETTED, "ollama", chosen, 131072)
            cfg = json.loads(path.read_text())
        self.assertIn(chosen, _ids(cfg, "router-ollama"))
        # Only on the chosen backend.
        self.assertNotIn(chosen, _ids(cfg, "router-vllm"))

    def test_empty_vetted_still_declares_chosen_and_every_provider(self) -> None:
        """A failed vetted scan (`{}`) degrades to the chosen model only."""
        with _PiHome() as path:
            PICKER._write_pi_models({}, "vllm", "Surprise-Model@32768", 32768)
            cfg = json.loads(path.read_text())
        self.assertEqual(_ids(cfg, "router-vllm"), ["Surprise-Model@32768"])
        self.assertEqual(_ids(cfg, "router-ollama"), [])
        self.assertIn("baseUrl", cfg["providers"]["router-ollama"])

    def test_context_windows(self) -> None:
        with _PiHome() as path:
            PICKER._write_pi_models(VETTED, "ollama", "qwen3.6:27b-q4_K_M", 65536)
            cfg = json.loads(path.read_text())
        # Chosen: the tier the picker resolved.
        self.assertEqual(_windows(cfg, "router-ollama"),
                         {"qwen3.6:27b-q4_K_M": 65536})
        # HF ids: their own @<ctx>.
        self.assertEqual(_windows(cfg, "router-vllm"),
                         {"Qwen3-8B-NVFP4@32768": 32768, "gpt-oss-20b@131072": 131072})

    def test_bare_unchosen_ollama_id_keeps_pi_default(self) -> None:
        with _PiHome() as path:
            PICKER._write_pi_models(VETTED, "vllm", "Qwen3-8B-NVFP4@32768", 32768)
            cfg = json.loads(path.read_text())
        self.assertEqual(_windows(cfg, "router-ollama"), {"qwen3.6:27b-q4_K_M": None})

    def test_survives_corrupt_config(self) -> None:
        with _PiHome() as path:
            path.write_text("{not json at all")
            PICKER._write_pi_models(VETTED, "vllm", "Qwen3-8B-NVFP4@32768", 32768)
            cfg = json.loads(path.read_text())
        self.assertIn("router-vllm", cfg["providers"])


class TestBuild(unittest.TestCase):
    def setUp(self) -> None:
        self._orig = PICKER._vetted_catalog
        PICKER._vetted_catalog = lambda: VETTED

    def tearDown(self) -> None:
        PICKER._vetted_catalog = self._orig

    def test_command_and_declared_window(self) -> None:
        with _PiHome(context="118784") as path:
            cmd = PICKER._build("pi", "Qwen3.8-27B::mtp@118784", "vllm-devai")
            cfg = json.loads(path.read_text())
        self.assertEqual(cmd, ["pi", "--provider", "router-vllm-devai",
                               "--model", "Qwen3.8-27B::mtp@118784"])
        self.assertEqual(_windows(cfg, "router-vllm-devai"),
                         {"Qwen3.8-27B::mtp@118784": 118784})

    def test_agent_offered_in_menu(self) -> None:
        row = next(a for a in PICKER._AGENTS if a[0] == "pi")
        self.assertEqual(len(row), 3)


class TestSeed(unittest.TestCase):
    def test_seed_declares_every_backend(self) -> None:
        cfg = json.loads((REPO_ROOT / "config" / "pi" / "models.json").read_text())
        for bname, (_label, _reason, port) in PICKER._BACKENDS.items():
            prov = cfg["providers"][f"router-{bname}"]
            self.assertEqual(prov["baseUrl"], f"http://devai-router:{port}/v1")
            self.assertEqual(prov["api"], "openai-completions")


if __name__ == "__main__":
    unittest.main()
