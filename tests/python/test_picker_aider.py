"""Aider must see the whole vetted model list, spanning all three backends.

Before this, the picker passed a single `--model` plus a GLOBAL
`--openai-api-base`. One base URL means one router port means one backend
per session: switching models in-session was impossible, and the model the
user picked at the menu was the only one aider ever knew about.

Aider supports per-model endpoints via `--model-settings-file`
(`extra_params.api_base`), so the fix declares every vetted model with its
own backend's URL. Schema verified by reading the INSTALLED aider, not
assumed:

  * `register_models` yaml.safe_load()s a LIST and calls
    `ModelSettings(**d)` -- so unknown keys raise, and only real fields
    (`name`, `extra_params`, ...) may appear. We emit JSON into the .yml:
    JSON is a subset of YAML, and model names like `qwen3.6:27b-q4_K_M`
    carry colons that are genuinely unsafe unquoted in bare YAML.
  * `register_litellm_models` json5-loads a MAP of model-name -> metadata
    and merges it into litellm's `local_model_metadata`.
  * litellm metadata keys taken from a real `litellm.model_cost` entry:
    `max_input_tokens`, `max_output_tokens`, `litellm_provider`, `mode`,
    `input_cost_per_token`, `output_cost_per_token`.

The Ollama/HF prefix asymmetry is deliberate -- see
test_ollama_keeps_ollama_chat_prefix.

Stdlib unittest only; no probe/bench cache and no running router required.
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
        "model_picker_aider", REPO_ROOT / "scripts" / "model-picker.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_picker_aider"] = mod
    spec.loader.exec_module(mod)
    return mod


PICKER = _load()

VETTED = {
    "ollama": ["qwen3.6:27b-q4_K_M"],
    "vllm": ["Qwen3-8B-NVFP4@32768", "gpt-oss-20b@131072"],
    # gpt-oss-20b COLLIDES with vLLM on purpose (it really is served by
    # both here); Ornith is SGLang-only so cross-backend assertions have a
    # model that cannot be satisfied by the vLLM entry.
    "sglang": ["gpt-oss-20b@131072", "Ornith-1.0-9B-NVFP4@131072"],
}


class _Home:
    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._prev = os.environ.get("HOME")
        os.environ["HOME"] = self._tmp.name
        return Path(self._tmp.name)

    def __exit__(self, *exc):
        if self._prev is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = self._prev
        self._tmp.cleanup()


class TestAiderModelFiles(unittest.TestCase):
    def test_settings_is_a_list_of_valid_modelsettings_keys(self) -> None:
        # register_models does ModelSettings(**d): an unknown key is a
        # TypeError at aider startup, so the writer may only emit real
        # dataclass field names.
        allowed = {
            "name", "edit_format", "weak_model_name", "use_repo_map",
            "send_undo_reply", "lazy", "overeager", "reminder",
            "examples_as_sys_msg", "extra_params", "cache_control",
            "caches_by_default", "use_system_prompt", "use_temperature",
            "streaming", "editor_model_name", "editor_edit_format",
            "reasoning_tag", "remove_reasoning", "system_prompt_prefix",
            "accepts_settings",
        }
        with _Home():
            _meta, settings = PICKER._write_aider_model_files(
                VETTED, "vllm", "Qwen3-8B-NVFP4@32768")
            entries = json.loads(Path(settings).read_text())
        self.assertIsInstance(entries, list)
        self.assertTrue(entries)
        for e in entries:
            self.assertLessEqual(set(e) - allowed, set(), f"bad keys in {e}")

    def test_every_model_carries_its_own_backend_api_base(self) -> None:
        with _Home():
            _meta, settings = PICKER._write_aider_model_files(
                VETTED, "vllm", "Qwen3-8B-NVFP4@32768")
            entries = json.loads(Path(settings).read_text())
        by_name = {e["name"]: e for e in entries}
        vllm_base = by_name["openai/Qwen3-8B-NVFP4@32768"]["extra_params"]["api_base"]
        sglang_base = by_name[
            "openai/Ornith-1.0-9B-NVFP4@131072"]["extra_params"]["api_base"]
        # Both declared, on DIFFERENT ports -- that is the whole point: one
        # session, several backends, which a global --openai-api-base could
        # never express.
        self.assertNotEqual(vllm_base, sglang_base)
        self.assertTrue(vllm_base.endswith("/v1"))
        _l, _u, sglang_port = PICKER._BACKENDS["sglang"]
        self.assertIn(str(sglang_port), sglang_base)

    def test_ollama_api_base_has_no_v1_suffix(self) -> None:
        """litellm appends /api/chat to the ollama base, so /v1 would 404.

        `litellm/llms/ollama/chat/transformation.py:240` builds
        `f"{api_base}/api/chat"`. A `/v1` on the base produces
        `/v1/api/chat`, which the router has no route for.
        """
        with _Home():
            _meta, settings = PICKER._write_aider_model_files(
                VETTED, "ollama", "qwen3.6:27b-q4_K_M")
            entries = json.loads(Path(settings).read_text())
        by_name = {e["name"]: e for e in entries}
        ollama_base = by_name[
            "ollama_chat/qwen3.6:27b-q4_K_M"]["extra_params"]["api_base"]
        self.assertFalse(ollama_base.endswith("/v1"), ollama_base)
        # HF backends DO need /v1 -- they speak the OpenAI wire.
        self.assertTrue(
            by_name["openai/gpt-oss-20b@131072"]["extra_params"]["api_base"]
            .endswith("/v1"))

    def test_ollama_keeps_ollama_chat_prefix(self) -> None:
        """Not cosmetic: it is the only path that gets per-session ctx.

        The router injects `options.num_ctx` on Ollama's NATIVE /api/chat
        only; upstream Ollama ignores num_ctx on the /v1 compat surface.
        Declaring Ollama rows as `openai/` would route them through /v1 and
        silently drop per-session context control, so they stay on
        litellm's ollama_chat provider while HF rows use openai/.
        """
        with _Home():
            _meta, settings = PICKER._write_aider_model_files(
                VETTED, "ollama", "qwen3.6:27b-q4_K_M")
            names = {e["name"] for e in json.loads(Path(settings).read_text())}
        self.assertIn("ollama_chat/qwen3.6:27b-q4_K_M", names)
        self.assertNotIn("openai/qwen3.6:27b-q4_K_M", names)

    def test_metadata_is_a_map_with_litellm_keys(self) -> None:
        with _Home():
            meta, _settings = PICKER._write_aider_model_files(
                VETTED, "vllm", "Qwen3-8B-NVFP4@32768")
            data = json.loads(Path(meta).read_text())
        self.assertIsInstance(data, dict)
        entry = data["openai/Qwen3-8B-NVFP4@32768"]
        self.assertEqual(entry["max_input_tokens"], 32768)
        self.assertEqual(entry["litellm_provider"], "openai")
        self.assertEqual(entry["mode"], "chat")
        # Local inference is free; a nonzero cost makes aider nag about spend.
        self.assertEqual(entry["input_cost_per_token"], 0)
        self.assertEqual(entry["output_cost_per_token"], 0)

    def test_context_comes_from_the_pinned_ctx_not_a_default(self) -> None:
        with _Home():
            meta, _s = PICKER._write_aider_model_files(VETTED, "vllm", "x@1")
            data = json.loads(Path(meta).read_text())
        self.assertEqual(data["openai/gpt-oss-20b@131072"]["max_input_tokens"], 131072)

    def test_rewrites_rather_than_accumulates(self) -> None:
        with _Home():
            meta, _s = PICKER._write_aider_model_files(
                VETTED, "vllm", "Qwen3-8B-NVFP4@32768")
            self.assertIn("openai/gpt-oss-20b@131072",
                          json.loads(Path(meta).read_text()))
            shrunk = {**VETTED, "vllm": ["Qwen3-8B-NVFP4@32768"], "sglang": []}
            PICKER._write_aider_model_files(shrunk, "vllm", "Qwen3-8B-NVFP4@32768")
            data = json.loads(Path(meta).read_text())
        self.assertNotIn("openai/gpt-oss-20b@131072", data)


class TestBackendCollision(unittest.TestCase):
    """One aider id cannot mean two backends.

    `gpt-oss-20b@131072` is served by BOTH vLLM and SGLang on this fleet,
    and both map to `openai/gpt-oss-20b@131072`. aider has no provider
    namespace to disambiguate with (unlike OpenCode's router-<backend>/),
    so without explicit resolution the later write silently wins the
    api_base -- picking SGLang at the menu and being served by vLLM.
    """

    def test_collision_resolves_to_the_picked_backend(self) -> None:
        with _Home():
            _m, settings = PICKER._write_aider_model_files(
                VETTED, "sglang", "gpt-oss-20b@131072")
            entries = json.loads(Path(settings).read_text())
        matching = [e for e in entries if e["name"] == "openai/gpt-oss-20b@131072"]
        self.assertEqual(len(matching), 1, "duplicate ids reach aider")
        _l, _u, sglang_port = PICKER._BACKENDS["sglang"]
        self.assertIn(str(sglang_port), matching[0]["extra_params"]["api_base"])

    def test_collision_defaults_to_vllm_when_unpicked(self) -> None:
        with _Home():
            _m, settings = PICKER._write_aider_model_files(
                VETTED, "vllm", "Qwen3-8B-NVFP4@32768")
            entries = json.loads(Path(settings).read_text())
        matching = [e for e in entries if e["name"] == "openai/gpt-oss-20b@131072"]
        self.assertEqual(len(matching), 1)
        _l, _u, vllm_port = PICKER._BACKENDS["vllm"]
        self.assertIn(str(vllm_port), matching[0]["extra_params"]["api_base"])

    def test_no_duplicate_names_anywhere(self) -> None:
        with _Home():
            _m, settings = PICKER._write_aider_model_files(
                VETTED, "sglang", "gpt-oss-20b@131072")
            names = [e["name"] for e in json.loads(Path(settings).read_text())]
        self.assertEqual(len(names), len(set(names)), names)


class TestAiderBuild(unittest.TestCase):
    def test_build_passes_both_files_and_no_global_base(self) -> None:
        with _Home():
            cmd = PICKER._build("aider", "Qwen3-8B-NVFP4@32768", "vllm")
        self.assertIn("--model-metadata-file", cmd)
        self.assertIn("--model-settings-file", cmd)
        self.assertIn("openai/Qwen3-8B-NVFP4@32768", cmd)
        # A global base would re-pin the session to one backend.
        self.assertNotIn("--openai-api-base", cmd)

    def test_build_ollama_uses_ollama_chat(self) -> None:
        with _Home():
            cmd = PICKER._build("aider", "qwen3.6:27b-q4_K_M", "ollama")
        self.assertIn("ollama_chat/qwen3.6:27b-q4_K_M", cmd)


if __name__ == "__main__":
    unittest.main()
