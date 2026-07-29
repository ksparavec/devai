"""OpenCode's declared model list must MIRROR the picker's vetted set.

Two separate problems, both fixed here.

1. **It accumulated forever.** `_ensure_opencode_model` merged the chosen
   model into ~/.config/opencode/opencode.json and never removed anything.
   That config lives on the persistent HOME_VOLUME, so a model later
   dropped by a bench verdict, excluded by the ledger, or deleted from disk
   stayed selectable in `opencode models` indefinitely -- the exact
   "router must never advertise anything unvetted" rule, violated one
   config layer above the router. Declaring the list must REPLACE the
   router-* providers, not merge into them.

2. **It declared one model at a time.** Only the just-picked model was ever
   registered, so a session could not switch models even though OpenCode
   supports it and the router serves every vetted row. All three backends
   are declared as separate providers.

User-authored config (other providers, unrelated top-level keys) is
preserved -- we own the `router-*` provider ids and nothing else.

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
        "model_picker_opencode", REPO_ROOT / "scripts" / "model-picker.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_picker_opencode"] = mod
    spec.loader.exec_module(mod)
    return mod


PICKER = _load()


class _ConfigHome:
    """Point XDG_CONFIG_HOME at a temp dir for the duration of a test."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._prev = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = self._tmp.name
        return Path(self._tmp.name) / "opencode" / "opencode.json"

    def __exit__(self, *exc):
        if self._prev is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = self._prev
        self._tmp.cleanup()


VETTED = {
    "ollama": ["qwen3.6:27b-q4_K_M"],
    "vllm": ["Qwen3-8B-NVFP4@32768", "gpt-oss-20b@131072"],
    "sglang": ["gpt-oss-20b@131072"],
}


class TestWriteOpencodeProviders(unittest.TestCase):
    def test_declares_every_backend_as_its_own_provider(self) -> None:
        with _ConfigHome() as cfg_path:
            PICKER._write_opencode_providers(VETTED, "vllm", "Qwen3-8B-NVFP4@32768")
            cfg = json.loads(cfg_path.read_text())
        self.assertEqual(
            sorted(p for p in cfg["provider"] if p.startswith("router-")),
            ["router-ollama", "router-sglang", "router-vllm"])
        self.assertEqual(
            sorted(cfg["provider"]["router-vllm"]["models"]),
            ["Qwen3-8B-NVFP4@32768", "gpt-oss-20b@131072"])
        self.assertEqual(
            sorted(cfg["provider"]["router-sglang"]["models"]),
            ["gpt-oss-20b@131072"])

    def test_each_provider_points_at_its_own_router_port(self) -> None:
        with _ConfigHome() as cfg_path:
            PICKER._write_opencode_providers(VETTED, "vllm", "Qwen3-8B-NVFP4@32768")
            cfg = json.loads(cfg_path.read_text())
        ports = {
            pid: prov["options"]["baseURL"].rsplit(":", 1)[1].split("/")[0]
            for pid, prov in cfg["provider"].items()
        }
        # One port per backend -- the port IS the backend selector.
        self.assertEqual(len(set(ports.values())), 3, ports)

    def test_replaces_rather_than_accumulates(self) -> None:
        """A model that drops out of the vetted set must disappear."""
        with _ConfigHome() as cfg_path:
            PICKER._write_opencode_providers(VETTED, "vllm", "Qwen3-8B-NVFP4@32768")
            first = json.loads(cfg_path.read_text())
            self.assertIn("gpt-oss-20b@131072", first["provider"]["router-vllm"]["models"])

            shrunk = {**VETTED, "vllm": ["Qwen3-8B-NVFP4@32768"]}
            PICKER._write_opencode_providers(shrunk, "vllm", "Qwen3-8B-NVFP4@32768")
            second = json.loads(cfg_path.read_text())

        self.assertEqual(
            sorted(second["provider"]["router-vllm"]["models"]),
            ["Qwen3-8B-NVFP4@32768"])
        self.assertNotIn("gpt-oss-20b@131072",
                         second["provider"]["router-vllm"]["models"])

    def test_preserves_user_config(self) -> None:
        with _ConfigHome() as cfg_path:
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(json.dumps({
                "theme": "gruvbox",
                "provider": {
                    "my-openai": {"npm": "@ai-sdk/openai", "models": {"gpt-4": {}}},
                    "router-vllm": {"models": {"stale-model@1": {}}},
                },
            }))
            PICKER._write_opencode_providers(VETTED, "vllm", "Qwen3-8B-NVFP4@32768")
            cfg = json.loads(cfg_path.read_text())

        # Untouched: user's own key and their own provider.
        self.assertEqual(cfg["theme"], "gruvbox")
        self.assertIn("my-openai", cfg["provider"])
        self.assertIn("gpt-4", cfg["provider"]["my-openai"]["models"])
        # Ours: rebuilt, stale entry gone.
        self.assertNotIn("stale-model@1", cfg["provider"]["router-vllm"]["models"])

    def test_chosen_model_always_declared_even_if_absent_from_vetted(self) -> None:
        """Launch must not fail on a cache/pick disagreement.

        OpenCode rejects an undeclared id outright ("model is not valid").
        If the vetted scan somehow misses the model the user just picked,
        declaring it anyway costs nothing and keeps the session working --
        the router still applies its own allowlist.
        """
        with _ConfigHome() as cfg_path:
            PICKER._write_opencode_providers({}, "vllm", "Surprise-Model@32768")
            cfg = json.loads(cfg_path.read_text())
        self.assertIn("Surprise-Model@32768",
                      cfg["provider"]["router-vllm"]["models"])

    def test_empty_backend_declares_no_models_not_a_broken_provider(self) -> None:
        with _ConfigHome() as cfg_path:
            PICKER._write_opencode_providers(
                {"ollama": [], "vllm": ["a@1"], "sglang": []}, "vllm", "a@1")
            cfg = json.loads(cfg_path.read_text())
        self.assertEqual(cfg["provider"]["router-ollama"]["models"], {})
        self.assertIn("baseURL", cfg["provider"]["router-ollama"]["options"])

    def test_survives_corrupt_config(self) -> None:
        with _ConfigHome() as cfg_path:
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text("{not json at all")
            PICKER._write_opencode_providers(VETTED, "vllm", "Qwen3-8B-NVFP4@32768")
            cfg = json.loads(cfg_path.read_text())
        self.assertIn("router-vllm", cfg["provider"])


class TestVettedIdShape(unittest.TestCase):
    """Ids must be what the router's allowlist actually accepts."""

    def test_ollama_ids_are_bare_hf_ids_carry_ctx(self) -> None:
        rows = [
            {"name": "qwen3.6:27b-q4_K_M", "backend": "ollama",
             "_picker_context": 131072},
            {"name": "Qwen3-8B-NVFP4", "backend": "vllm",
             "_picker_context": 32768},
            {"name": "gpt-oss-20b", "backend": "sglang",
             "_picker_context": 131072},
        ]
        out = PICKER._vetted_ids_by_backend(rows)
        # Ollama rides bare: the router deliberately will not move an
        # Ollama tier for a bare name, so per-ctx ids would invite thrash.
        self.assertEqual(out["ollama"], ["qwen3.6:27b-q4_K_M"])
        self.assertEqual(out["vllm"], ["Qwen3-8B-NVFP4@32768"])
        self.assertEqual(out["sglang"], ["gpt-oss-20b@131072"])

    def test_same_name_on_two_backends_is_kept_on_both(self) -> None:
        """The menu dedups vLLM/SGLang; the provider list must not."""
        rows = [
            {"name": "gpt-oss-20b", "backend": "vllm", "_picker_context": 131072},
            {"name": "gpt-oss-20b", "backend": "sglang", "_picker_context": 131072},
        ]
        out = PICKER._vetted_ids_by_backend(rows)
        self.assertEqual(out["vllm"], ["gpt-oss-20b@131072"])
        self.assertEqual(out["sglang"], ["gpt-oss-20b@131072"])


if __name__ == "__main__":
    unittest.main()
