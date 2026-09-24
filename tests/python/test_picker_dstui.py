"""dstui (DeepSeek Harness terminal UI) is a picker agent on the lab's own dsh.

Operator requirement (2026-09-23): the TUI and the web UI run EXACTLY the same
backend -- the lab's npm dsh at /usr/local/bin/dsh, never an SDK-bundled
runtime -- so the picker always passes `--dsh-bin /usr/local/bin/dsh`.

dstui spawns its own dsh over stdio (the SDK profile) rather than attaching to
the web server, so the router providers reach it as a devai-owned patch file,
rewritten whole at every launch, next to devai's overlay (uploads off).
Installed like aiagent: a makeself bundle from `make fetch-cli`, extracted to
the isolated prefix /opt/dstui (it brings its own Python).

Stdlib unittest only.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "model_picker_dstui", REPO_ROOT / "scripts" / "model-picker.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_picker_dstui"] = mod
    spec.loader.exec_module(mod)
    return mod


PICKER = _load()

VETTED = {
    "ollama": ["qwen3.8:27b-mtp-q4_K_M"],
    "vllm": ["Qwen3-8B-NVFP4@32768"],
}


class _ConfigHome:
    def __init__(self, context: str | None = None) -> None:
        self._context = context

    def __enter__(self) -> Path:
        self._tmp = tempfile.TemporaryDirectory()
        self._prev = {k: os.environ.get(k) for k in ("XDG_CONFIG_HOME", "CONTEXT")}
        os.environ["XDG_CONFIG_HOME"] = self._tmp.name
        if self._context is None:
            os.environ.pop("CONTEXT", None)
        else:
            os.environ["CONTEXT"] = self._context
        return Path(self._tmp.name) / "devai" / "dstui-router.yml"

    def __exit__(self, *exc) -> None:
        for k, v in self._prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()


class TestDstuiCommand(unittest.TestCase):
    def _build(self, name: str, backend: str, context: str = "131072"):
        with _ConfigHome(context) as patch_path, \
                mock.patch.object(PICKER, "_vetted_catalog", return_value=VETTED):
            cmd = PICKER._build("dstui", name, backend)
            import yaml
            entries = yaml.load(patch_path.read_text(), Loader=PICKER._DshLoader)
            return cmd, patch_path, entries

    def test_always_runs_the_labs_own_dsh(self) -> None:
        cmd, _, _ = self._build("qwen3.8:27b-mtp-q4_K_M", "ollama")
        self.assertEqual(cmd[0], "dstui")
        self.assertEqual(cmd[cmd.index("--dsh-bin") + 1], "/usr/local/bin/dsh")

    def test_provider_model_and_patches(self) -> None:
        cmd, patch_path, _ = self._build("Qwen3-8B-NVFP4::nothink@32768", "vllm", "32768")
        self.assertEqual(cmd[cmd.index("--provider") + 1], "router-vllm")
        self.assertEqual(cmd[cmd.index("-m") + 1], "Qwen3-8B-NVFP4::nothink@32768")
        patches = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--patch"]
        self.assertEqual(patches, ["/etc/devai/dsh-overlay.yml", str(patch_path)])

    def test_patch_declares_the_router_providers_with_the_chosen_model(self) -> None:
        _, _, entries = self._build("Qwen3-8B-NVFP4::nothink@32768", "vllm", "32768")
        self.assertEqual([e["id"] for e in entries], ["llm-pi-ai"])
        prov = entries[0]["config"]["providers"]
        self.assertEqual(set(prov), {"router-ollama", "router-vllm"})
        models = {m["id"]: m.get("contextWindow") for m in prov["router-vllm"]["models"]}
        self.assertEqual(models["Qwen3-8B-NVFP4::nothink@32768"], 32768)
        self.assertEqual(prov["router-vllm"]["apiKeyEnv"], "DEVAI_ROUTER_API_KEY")

    def test_router_key_is_provided(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEVAI_ROUTER_API_KEY", None)
            self._build("qwen3.8:27b-mtp-q4_K_M", "ollama")
            self.assertEqual(os.environ["DEVAI_ROUTER_API_KEY"], "local")

    def test_offered_everywhere(self) -> None:
        """A terminal UI needs no published port, unlike the web agent."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DSH_PORT", None)
            self.assertIn("dstui", [a[0] for a in PICKER._offered_agents()])


class TestImageAndFetch(unittest.TestCase):
    def test_fetch_cli_downloads_the_release_installer(self) -> None:
        mk = (REPO_ROOT / "Makefile").read_text()
        self.assertIn(
            "https://github.com/ksparavec/dstui/releases/latest/download/dstui-install.sh", mk)
        self.assertIn("$(ETAG_DIR)/dstui.etag", mk)

    def test_lab_installs_it_to_an_isolated_prefix(self) -> None:
        df = (REPO_ROOT / "deploy" / "Dockerfile.lab").read_text()
        self.assertIn("DSTUI_PREFIX=/opt/dstui sh /var/cache/bin/dstui-install.sh", df)
        self.assertIn("ln -sf /opt/dstui/bin/dstui /usr/local/bin/dstui", df)

    def test_jupyterlab_card(self) -> None:
        src = (REPO_ROOT / "packages" / "jupyter-ai-launchers" / "src" / "index.ts").read_text()
        self.assertIn("command: 'model-picker --agent dstui'", src)


if __name__ == "__main__":
    unittest.main()
