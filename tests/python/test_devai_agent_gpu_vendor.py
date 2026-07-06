"""bin/devai-agent: GPU-vendor overlay (docs/gpu-vendors.md).

devai-agent deliberately does not read the repo's .env (state lives in
~/.devai/preferences.yaml, independent of repo cwd -- see its own
docstring), so its GPU vendor knob is a `gpu_vendor` preference field
threaded through gpu_flags(), separate from the Makefile/.env-driven
DEVAI_GPU_DEVICE path gpu-arbiter and docker-compose.yaml use. These
tests pin gpu_flags()'s vendor-conditional device string and the
DEFAULTS dict carrying the new field.
"""

from __future__ import annotations

import importlib.util
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LAUNCHER = REPO_ROOT / "bin" / "devai-agent"


def _load_launcher():
    # bin/devai-agent has no .py suffix, so spec_from_file_location can't
    # infer a loader -- construct one explicitly (unlike model-picker.py,
    # which other tests in this suite load via spec_from_file_location
    # directly since it does have the extension).
    loader = SourceFileLoader("_devai_agent_under_test", str(LAUNCHER))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


class TestGPUFlagsVendorOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_launcher()

    def test_cpu_only_ignores_vendor(self):
        self.assertEqual(self.mod.gpu_flags("podman", True, "amd"), [])
        self.assertEqual(self.mod.gpu_flags("podman", True, "nvidia"), [])

    def test_podman_defaults_to_nvidia(self):
        self.assertEqual(
            self.mod.gpu_flags("podman", False),
            ["--device", "nvidia.com/gpu=all", "--security-opt=label=disable"],
        )

    def test_podman_nvidia_explicit(self):
        self.assertEqual(
            self.mod.gpu_flags("podman", False, "nvidia"),
            ["--device", "nvidia.com/gpu=all", "--security-opt=label=disable"],
        )

    def test_podman_amd(self):
        self.assertEqual(
            self.mod.gpu_flags("podman", False, "amd"),
            ["--device", "amd.com/gpu=all", "--security-opt=label=disable"],
        )

    def test_docker_ignores_vendor(self):
        # Docker's --gpus all doesn't take a CDI vendor string; ROCm under
        # Docker is out of scope for this pass (see docs/gpu-vendors.md).
        self.assertEqual(self.mod.gpu_flags("docker", False, "amd"), ["--gpus", "all"])
        self.assertEqual(self.mod.gpu_flags("docker", False, "nvidia"), ["--gpus", "all"])

    def test_defaults_dict_carries_gpu_vendor(self):
        self.assertEqual(self.mod.DEFAULTS["gpu_vendor"], "nvidia")


if __name__ == "__main__":
    unittest.main()
