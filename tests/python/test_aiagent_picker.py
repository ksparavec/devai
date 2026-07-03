"""aiagent shell agent: picker wiring + launcher GPU policy.

The picker's "AIAgent (shell)" agent does NOT exec aiagent. Instead it
configures the router endpoint + model in the environment and drops the user
into a bash shell (aiagent-launcher.sh, installed as `aiagent-shell`) where the
user runs `aiagent ...` themselves. These tests pin:

  * aiagent is offered in the agent menu;
  * _build("aiagent", ...) sets the AIAGENT_* env contract (API base INCLUDING
    /v1, per aiagent's README) + OpenAI-compat fallbacks and returns
    ["aiagent-shell"], for both the Ollama (11434) and vLLM (11435) ports;
  * the GPU sub-modal honors a pre-set DEVAI_AIAGENT_GPU env value without
    prompting (env-flag override);
  * the launcher applies CUDA_VISIBLE_DEVICES per mode (router-only hides the
    GPU; share leaves it visible).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LAUNCHER = REPO_ROOT / "scripts" / "aiagent-launcher.sh"


def _load_picker():
    spec = importlib.util.spec_from_file_location(
        "_picker_under_test_aiagent",
        str(REPO_ROOT / "scripts" / "model-picker.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


PICKER = _load_picker()

# Env keys the picker's aiagent branch / GPU helper mutate. Snapshotted and
# restored around every test so ordering can't leak state.
_MUTATED_KEYS = (
    "AIAGENT_API_BASE",
    "AIAGENT_API_KEY",
    "AIAGENT_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "DEVAI_AIAGENT_GPU",
)


class _EnvIsolated(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {k: os.environ.get(k) for k in _MUTATED_KEYS}
        for k in _MUTATED_KEYS:
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestAgentMenu(_EnvIsolated):
    def test_aiagent_offered(self) -> None:
        ids = [a[0] for a in PICKER._AGENTS]
        self.assertIn("aiagent", ids)
        # Each agent row stays a 3-tuple (id, display, description).
        row = next(a for a in PICKER._AGENTS if a[0] == "aiagent")
        self.assertEqual(len(row), 3)


class TestBuildContract(_EnvIsolated):
    def test_ollama_backend_wiring(self) -> None:
        cmd = PICKER._build("aiagent", "qwen3.6:27b-q4_K_M", "ollama")
        self.assertEqual(cmd, ["aiagent-shell"])
        # README devai contract: base URL includes /v1.
        self.assertEqual(
            os.environ["AIAGENT_API_BASE"], "http://devai-router:11434/v1"
        )
        self.assertEqual(os.environ["AIAGENT_MODEL"], "qwen3.6:27b-q4_K_M")
        self.assertEqual(os.environ["AIAGENT_API_KEY"], "local")
        # OpenAI-compat fallbacks so sibling SDK tools inherit the router.
        self.assertEqual(
            os.environ["OPENAI_BASE_URL"], "http://devai-router:11434/v1"
        )
        self.assertEqual(os.environ["OPENAI_API_KEY"], "local")

    def test_vllm_backend_port_and_ctx_suffix(self) -> None:
        # vLLM serving name carries the @<ctx> control-surface suffix; the
        # picker passes it through verbatim (aiagent parses it itself).
        cmd = PICKER._build("aiagent", "Qwen3-14B-NVFP4@32768", "vllm")
        self.assertEqual(cmd, ["aiagent-shell"])
        self.assertEqual(
            os.environ["AIAGENT_API_BASE"], "http://devai-router:11435/v1"
        )
        self.assertEqual(os.environ["AIAGENT_MODEL"], "Qwen3-14B-NVFP4@32768")

    def test_openai_fallbacks_not_clobbered(self) -> None:
        os.environ["OPENAI_BASE_URL"] = "http://preset.example/v1"
        os.environ["OPENAI_API_KEY"] = "preset-key"
        PICKER._build("aiagent", "qwen3.6:27b-q4_K_M", "ollama")
        # setdefault must not overwrite a value the user already exported.
        self.assertEqual(os.environ["OPENAI_BASE_URL"], "http://preset.example/v1")
        self.assertEqual(os.environ["OPENAI_API_KEY"], "preset-key")


class TestGpuModeResolution(_EnvIsolated):
    def test_env_share_overrides_prompt(self) -> None:
        os.environ["DEVAI_AIAGENT_GPU"] = "share"
        self.assertEqual(PICKER._resolve_aiagent_gpu_mode(), "share")

    def test_env_router_only_variants(self) -> None:
        for val in ("router-only", "router_only", "ROUTERONLY", " Router-Only "):
            os.environ["DEVAI_AIAGENT_GPU"] = val
            self.assertEqual(
                PICKER._resolve_aiagent_gpu_mode(), "router-only", msg=val
            )

    def test_apply_noop_for_other_agents(self) -> None:
        # No prompt, no env mutation for a non-aiagent agent.
        self.assertTrue(PICKER._apply_aiagent_gpu("claude"))
        self.assertIsNone(os.environ.get("DEVAI_AIAGENT_GPU"))

    def test_apply_records_mode_for_aiagent(self) -> None:
        os.environ["DEVAI_AIAGENT_GPU"] = "share"
        self.assertTrue(PICKER._apply_aiagent_gpu("aiagent"))
        self.assertEqual(os.environ["DEVAI_AIAGENT_GPU"], "share")


class TestLauncherGpuPolicy(unittest.TestCase):
    """Exercise aiagent-launcher.sh via its DEVAI_AIAGENT_SHELL_DEBUG hook."""

    def _run(self, env: dict[str, str]) -> dict[str, str]:
        base = {
            "PATH": os.environ.get("PATH", ""),
            "DEVAI_AIAGENT_SHELL_DEBUG": "1",
            "AIAGENT_API_BASE": "http://devai-router:11434/v1",
            "AIAGENT_MODEL": "qwen3.6:27b-q4_K_M",
        }
        base.update(env)
        out = subprocess.run(
            ["bash", str(LAUNCHER)],
            env=base,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        parsed: dict[str, str] = {}
        for line in out.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                parsed[k] = v
        return parsed

    def test_router_only_hides_gpu(self) -> None:
        got = self._run({"DEVAI_AIAGENT_GPU": "router-only"})
        self.assertEqual(got["DEVAI_AIAGENT_GPU"], "router-only")
        self.assertEqual(got["CUDA_VISIBLE_DEVICES"], "")

    def test_default_is_router_only(self) -> None:
        # No DEVAI_AIAGENT_GPU set -> safe default hides the GPU.
        got = self._run({})
        self.assertEqual(got["DEVAI_AIAGENT_GPU"], "router-only")
        self.assertEqual(got["CUDA_VISIBLE_DEVICES"], "")

    def test_share_keeps_gpu_visible(self) -> None:
        got = self._run(
            {"DEVAI_AIAGENT_GPU": "share", "CUDA_VISIBLE_DEVICES": "0"}
        )
        self.assertEqual(got["DEVAI_AIAGENT_GPU"], "share")
        self.assertEqual(got["CUDA_VISIBLE_DEVICES"], "0")

    def test_context_not_mapped_from_picker(self) -> None:
        # The launcher deliberately does NOT set AIAGENT_CONTEXT: aiagent turns
        # it into a `<model>@<ctx>` suffix composed BEFORE `::<reasoning>`, but
        # the router only strips `@<ctx>` when it is LAST -- so the mis-ordered
        # `@<ctx>` survives into the model name and Ollama rejects it ("invalid
        # model name"). Context is router-managed (Ollama global / vLLM @ctx tag).
        got = self._run({"CONTEXT": "32768"})
        self.assertEqual(got.get("AIAGENT_CONTEXT", ""), "")


class TestLauncherApiBaseNormalization(unittest.TestCase):
    """Standalone AIAGENT_API_BASE fallback must end in exactly one /v1."""

    def _run_raw(self, env: dict[str, str]) -> dict[str, str]:
        # Deliberately does NOT inject AIAGENT_API_BASE, so the launcher's
        # fallback composition is exercised. Start from a clean env so a
        # host-exported OPENAI_BASE_URL/OLLAMA_HOST can't leak in.
        base = {"PATH": os.environ.get("PATH", ""), "DEVAI_AIAGENT_SHELL_DEBUG": "1"}
        base.update(env)
        out = subprocess.run(
            ["bash", str(LAUNCHER)], env=base,
            capture_output=True, text=True, check=True,
        ).stdout
        parsed: dict[str, str] = {}
        for line in out.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                parsed[k] = v
        return parsed

    def test_openai_base_with_v1_not_doubled(self) -> None:
        got = self._run_raw({"OPENAI_BASE_URL": "http://devai-router:11435/v1"})
        self.assertEqual(got["AIAGENT_API_BASE"], "http://devai-router:11435/v1")

    def test_ollama_host_gets_single_v1(self) -> None:
        got = self._run_raw({"OLLAMA_HOST": "http://devai-router:11434"})
        self.assertEqual(got["AIAGENT_API_BASE"], "http://devai-router:11434/v1")

    def test_trailing_slash_normalized(self) -> None:
        got = self._run_raw({"OPENAI_BASE_URL": "http://devai-router:11435/v1/"})
        self.assertEqual(got["AIAGENT_API_BASE"], "http://devai-router:11435/v1")

    def test_hard_fallback_when_nothing_set(self) -> None:
        got = self._run_raw({})
        self.assertEqual(got["AIAGENT_API_BASE"], "http://devai-router:11434/v1")


if __name__ == "__main__":
    unittest.main()
