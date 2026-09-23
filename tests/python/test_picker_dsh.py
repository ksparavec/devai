"""DeepSeek Harness (dsh) is a picker agent served as a browser UI.

What these pin, each learned against dsh 0.1.5-rc.2:

- Providers go into the `llm-pi-ai` entry of the web profile's own
  cordis.patch.yml. A patch layer REPLACES an entry's whole config, so an
  overlay entry would hide providers added in the Settings UI, and a second
  `llm-pi-ai` instance fails boot (DUPLICATE_DIRECTORY). The `router-*`
  providers are rebuilt at every launch; everything else in the file --
  other providers, other entries, `!!js` expressions, the header comment --
  survives the rewrite.
- The default model goes into settings.yaml's `agent-default-model`.
- The shipped overlay disables both upload paths and binds 0.0.0.0:3080
  (the CLI refuses `--host 0.0.0.0`; the config schema accepts it).
- Only `make lab-cpu|lab-gpu` publish DSH_PORT; the menu hides the agent
  elsewhere and the launcher refuses without it.
- Node >= 22.19: dsh's entry point runs only `if (import.meta.main)`, which
  22.16 lacks, so it exited 0 having done nothing.

Stdlib unittest only; no dsh install needed.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "model_picker_dsh", REPO_ROOT / "scripts" / "model-picker.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_picker_dsh"] = mod
    spec.loader.exec_module(mod)
    return mod


PICKER = _load()

HEADER = (
    "# Your patch layer for this dsh profile, applied after every bundle layer:\n"
    "# a top-level YAML array of loader patch entries.\n"
)

USER_PATCH = HEADER + """\
- id: llm-pi-ai
  config:
    providers:
      my-gateway:
        api: openai-completions
        baseURL: !!js process.env.GW_URL
        models:
          - id: gw-model
      router-vllm:
        baseURL: http://stale:1/v1
        models:
          - id: stale@1
- id: webserver
  config:
    port: !!js 3080 + 0
"""

VETTED = {
    "ollama": ["qwen3.6:27b-q4_K_M"],
    "vllm": ["Qwen3-8B-NVFP4@32768", "gpt-oss-20b@131072"],
    "sglang": [],
}


class _DshHome:
    """A temp DSH_HOME whose web profile already exists (no dsh needed)."""

    def __init__(self, patch: str | None = USER_PATCH, settings: str | None = None) -> None:
        self._patch = patch
        self._settings = settings

    def __enter__(self) -> Path:
        self._tmp = tempfile.TemporaryDirectory()
        home = Path(self._tmp.name)
        self._prev = os.environ.get("DSH_HOME")
        os.environ["DSH_HOME"] = str(home)
        if self._patch is not None:
            prof = home / "profiles" / "web"
            prof.mkdir(parents=True)
            (prof / "cordis.patch.yml").write_text(self._patch)
        if self._settings is not None:
            (home / "settings.yaml").write_text(self._settings)
        return home

    def __exit__(self, *exc) -> None:
        if self._prev is None:
            os.environ.pop("DSH_HOME", None)
        else:
            os.environ["DSH_HOME"] = self._prev
        self._tmp.cleanup()


def _entries(home: Path) -> list:
    text = (home / "profiles" / "web" / "cordis.patch.yml").read_text()
    return yaml.load(text, Loader=PICKER._DshLoader)


def _providers(home: Path) -> dict:
    llm = next(e for e in _entries(home) if e.get("id") == "llm-pi-ai")
    return llm["config"]["providers"]


class TestWriteDshConfig(unittest.TestCase):
    def test_router_providers_replaced_user_provider_kept(self) -> None:
        with _DshHome() as home:
            PICKER._write_dsh_config(VETTED, "vllm", "Qwen3-8B-NVFP4@32768", 32768)
            prov = _providers(home)
        self.assertIn("my-gateway", prov)
        self.assertEqual(
            [m["id"] for m in prov["router-vllm"]["models"]],
            ["Qwen3-8B-NVFP4@32768", "gpt-oss-20b@131072"])
        self.assertEqual(prov["router-vllm"]["baseURL"], "http://devai-router:11435/v1")
        self.assertEqual(prov["router-vllm"]["api"], "openai-completions")
        self.assertEqual(prov["router-vllm"]["apiKeyEnv"], "DEVAI_ROUTER_API_KEY")
        self.assertEqual(prov["router-vllm"]["displayName"], "vLLM via DevAI router")

    def test_backends_without_models_are_left_out(self) -> None:
        with _DshHome() as home:
            PICKER._write_dsh_config(VETTED, "vllm", "Qwen3-8B-NVFP4@32768", 32768)
            prov = _providers(home)
        self.assertNotIn("router-sglang", prov)
        self.assertNotIn("router-vllm-devai", prov)
        self.assertIn("router-ollama", prov)

    def test_js_expressions_other_entries_and_header_survive(self) -> None:
        with _DshHome() as home:
            PICKER._write_dsh_config(VETTED, "vllm", "Qwen3-8B-NVFP4@32768", 32768)
            text = (home / "profiles" / "web" / "cordis.patch.yml").read_text()
            entries = _entries(home)
        self.assertTrue(text.startswith(HEADER), text[:200])
        llm = next(e for e in entries if e.get("id") == "llm-pi-ai")
        url = llm["config"]["providers"]["my-gateway"]["baseURL"]
        self.assertIsInstance(url, PICKER._YamlTagged)
        self.assertEqual(url.tag, "tag:yaml.org,2002:js")
        self.assertEqual(url.value, "process.env.GW_URL")
        self.assertIn("!!js", text)
        web = next(e for e in entries if e.get("id") == "webserver")
        self.assertEqual(web["config"]["port"].value, "3080 + 0")

    def test_context_windows(self) -> None:
        with _DshHome() as home:
            PICKER._write_dsh_config(VETTED, "ollama", "qwen3.6:27b-q4_K_M::nothink", 65536)
            prov = _providers(home)
        windows = {m["id"]: m.get("contextWindow") for m in prov["router-ollama"]["models"]}
        # The chosen id is declared exactly, with the tier the picker resolved;
        # a bare Ollama id keeps dsh's own default.
        self.assertEqual(windows, {"qwen3.6:27b-q4_K_M": None,
                                   "qwen3.6:27b-q4_K_M::nothink": 65536})
        vllm = {m["id"]: m.get("contextWindow") for m in prov["router-vllm"]["models"]}
        self.assertEqual(vllm["gpt-oss-20b@131072"], 131072)

    def test_default_model_written_other_settings_kept(self) -> None:
        with _DshHome(settings="theme:\n  mode: dark\nagent-default-model:\n"
                               "  provider: deepseek-official\n  model: deepseek-flash\n"
                               "  reasoningEffort: max\n") as home:
            PICKER._write_dsh_config(VETTED, "vllm", "Qwen3-8B-NVFP4@32768", 32768)
            settings = yaml.safe_load((home / "settings.yaml").read_text())
        self.assertEqual(settings["theme"], {"mode": "dark"})
        self.assertEqual(settings["agent-default-model"],
                         {"provider": "router-vllm", "model": "Qwen3-8B-NVFP4@32768"})

    def test_missing_llm_entry_is_created(self) -> None:
        with _DshHome(patch=HEADER + "[]\n") as home:
            PICKER._write_dsh_config(VETTED, "vllm", "Qwen3-8B-NVFP4@32768", 32768)
            prov = _providers(home)
        self.assertEqual(set(prov), {"router-ollama", "router-vllm"})

    def test_non_list_patch_is_refused(self) -> None:
        with _DshHome(patch="providers: {}\n"):
            with self.assertRaises(SystemExit):
                PICKER._write_dsh_config(VETTED, "vllm", "m@1", 1)

    def test_uninitialized_profile_asks_dsh_then_refuses_when_it_cannot(self) -> None:
        failed = subprocess.CompletedProcess(args=[], returncode=127, stdout="", stderr="dsh: not found")
        with _DshHome(patch=None):
            with mock.patch.object(PICKER.subprocess, "run", return_value=failed) as run:
                with self.assertRaises(SystemExit):
                    PICKER._write_dsh_config(VETTED, "vllm", "m@1", 1)
        self.assertEqual(run.call_args[0][0], ["dsh", "--profile", "web", "--dump-config"])


class TestAgentWiring(unittest.TestCase):
    def test_build_returns_the_web_launcher(self) -> None:
        with _DshHome(), mock.patch.object(PICKER, "_vetted_catalog", return_value=VETTED):
            self.assertEqual(PICKER._build("dsh", "Qwen3-8B-NVFP4@32768", "vllm"), ["dsh-web"])

    def test_menu_hides_dsh_without_a_published_port(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DSH_PORT", None)
            self.assertNotIn("dsh", [a[0] for a in PICKER._offered_agents()])
            os.environ["DSH_PORT"] = "3080"
            self.assertIn("dsh", [a[0] for a in PICKER._offered_agents()])


class TestShippedFiles(unittest.TestCase):
    def test_overlay_disables_uploads_and_binds_all_interfaces(self) -> None:
        entries = yaml.load((REPO_ROOT / "config" / "dsh" / "devai-overlay.yml").read_text(),
                            Loader=PICKER._DshLoader)
        by_id = {e["id"]: e for e in entries}
        self.assertTrue(by_id["session-telemetry-otel"]["disabled"])
        self.assertTrue(by_id["session-log-deepseek"]["disabled"])
        self.assertEqual(by_id["webserver"]["config"]["host"], "0.0.0.0")
        self.assertEqual(by_id["webserver"]["config"]["port"], 3080)
        # The overlay must never own the providers entry (see module doc).
        self.assertNotIn("llm-pi-ai", by_id)

    def test_launcher_needs_the_published_port(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "DSH_PORT"}
        r = subprocess.run(["bash", str(REPO_ROOT / "scripts" / "dsh-web-launcher.sh")],
                           env=env, capture_output=True, text=True, check=False)
        self.assertEqual(r.returncode, 1)
        self.assertIn("lab-gpu", r.stderr)

    def test_launcher_command(self) -> None:
        env = {**os.environ, "DSH_PORT": "3099", "HOST_IP": "192.0.2.7",
               "DEVAI_DSH_SHELL_DEBUG": "1"}
        r = subprocess.run(["bash", str(REPO_ROOT / "scripts" / "dsh-web-launcher.sh")],
                           env=env, capture_output=True, text=True, check=True)
        cmd = r.stdout
        self.assertIn("--profile web", cmd)
        self.assertIn("--patch /etc/devai/dsh-overlay.yml", cmd)
        self.assertIn("--no-open", cmd)
        self.assertIn("--trusted-host 192.0.2.7:3099", cmd)

    def test_launcher_prints_the_published_and_the_localhost_url(self) -> None:
        """dsh prints its in-container address; the launcher must hand the
        browser the published one, plus the localhost form that dsh's
        Settings pages require (they refuse any non-loopback page address)."""
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "dsh"
            fake.write_text(
                "#!/bin/sh\n"
                "echo 'dsh web: http://127.0.0.1:3080/?token=T0k (LAN: http://10.89.0.5:3080/?token=T0k)'\n")
            fake.chmod(0o755)
            env = {**os.environ, "PATH": f"{tmp}:{os.environ['PATH']}",
                   "DSH_PORT": "3099", "HOST_IP": "192.0.2.7"}
            r = subprocess.run(["bash", str(REPO_ROOT / "scripts" / "dsh-web-launcher.sh")],
                               env=env, capture_output=True, text=True, check=True)
        self.assertIn("dsh web: http://192.0.2.7:3099/?token=T0k\n", r.stdout)
        self.assertIn("http://localhost:3099/?token=T0k", r.stdout)
        self.assertIn("ssh -L 3099:localhost:3099", r.stdout)
        self.assertNotIn("10.89.0.5", r.stdout)
        self.assertNotIn("127.0.0.1:3080", r.stdout)

    def test_lab_targets_publish_the_dsh_port(self) -> None:
        mk = (REPO_ROOT / "Makefile").read_text()
        for target in ("lab-cpu", "lab-gpu"):
            body = mk.split(f"\n{target}:", 1)[1].split("\n\n", 1)[0]
            self.assertIn("-p 0.0.0.0:$(DSH_PORT):3080", body, target)
            self.assertIn("-e DSH_PORT=$(DSH_PORT)", body, target)
        self.assertRegex(mk, r"(?m)^DSH_VERSION \?= \d+\.\d+\.\d+")

    def test_base_node_is_new_enough(self) -> None:
        m = re.search(r"NODE_VERSION=(\d+)\.(\d+)\.(\d+)",
                      (REPO_ROOT / "deploy" / "Dockerfile.base").read_text())
        self.assertIsNotNone(m)
        self.assertGreaterEqual(tuple(int(x) for x in m.groups()), (22, 19, 0))

    def test_lab_image_installs_dsh(self) -> None:
        df = (REPO_ROOT / "deploy" / "Dockerfile.lab").read_text()
        self.assertIn("cp -a /var/cache/bin/dsh /usr/local/lib/dsh", df)
        self.assertIn("COPY config/dsh/devai-overlay.yml /etc/devai/dsh-overlay.yml", df)
        self.assertIn("COPY scripts/dsh-web-launcher.sh /usr/local/bin/dsh-web", df)
        self.assertIn("ENV DSH_TELEMETRY_MODE=DISABLED", df)


if __name__ == "__main__":
    unittest.main()
