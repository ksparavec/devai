"""bin/devai-agent is NVIDIA-only, follows the host check, and an old preferences.yaml still loads.

The AMD/ROCm overlay moved to attic/amd-rocm/ on 2026-10-07 (platform-rebuild
Phase 1, M11). It took the launcher's `--gpu-vendor` flag and `gpu_vendor`
preference with it; the GPU device is NVIDIA's CDI string again. A
~/.devai/preferences.yaml written by an older launcher still carries
`gpu_vendor: nvidia` (the old default was written on every exit), so reading
one must not fail and must drop the key, which the next write then omits.

Stdlib unittest only; reads and writes temp files, starts nothing.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import tempfile
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_launcher():
    loader = SourceFileLoader("devai_agent_nvidia_only",
                              str(REPO_ROOT / "bin" / "devai-agent"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


class GPUFlagsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_launcher()

    def test_no_device_has_no_gpu_flags(self):
        # The host check printed nothing: the same image runs on the CPU.
        self.assertEqual(self.mod.gpu_flags("podman", ""), [])
        self.assertEqual(self.mod.gpu_flags("docker", ""), [])

    def test_podman_gets_the_host_checks_device(self):
        self.assertEqual(
            self.mod.gpu_flags("podman", "nvidia.com/gpu=all"),
            ["--device", "nvidia.com/gpu=all", "--security-opt=label=disable"],
        )
        self.assertEqual(self.mod.gpu_flags("podman", "nvidia.com/gpu=0")[1], "nvidia.com/gpu=0")

    def test_docker_gets_gpus_all(self):
        self.assertEqual(self.mod.gpu_flags("docker", "nvidia.com/gpu=all"), ["--gpus", "all"])

    def test_one_lab_image_and_no_cpu_flag(self):
        self.assertEqual(self.mod.LAB_IMAGE, "devai-lab")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.mod.make_parser().parse_args(["--cpu"])

    def test_host_check_is_run_and_a_missing_one_refused(self):
        with tempfile.TemporaryDirectory() as td:
            check = Path(td) / "host-gpu.sh"
            self.mod.HOST_GPU_SRC = check
            with self.assertRaises(SystemExit) as cm:
                self.mod.host_gpu_device()
            self.assertIn("make install", str(cm.exception))
            check.write_text("#!/bin/sh\necho nvidia.com/gpu=all\n")
            check.chmod(0o755)
            self.assertEqual(self.mod.host_gpu_device(), "nvidia.com/gpu=all")
            check.write_text("#!/bin/sh\nexit 0\n")
            self.assertEqual(self.mod.host_gpu_device(), "")

    def test_no_gpu_vendor_flag_or_default(self):
        self.assertNotIn("gpu_vendor", self.mod.DEFAULTS)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.mod.make_parser().parse_args(["--gpu-vendor", "amd"])


class RetiredPreferenceTest(unittest.TestCase):
    def test_old_preferences_with_gpu_vendor_load_and_drop_it(self):
        mod = _load_launcher()
        with tempfile.TemporaryDirectory() as td:
            mod.CONFIG_DIR = Path(td)
            mod.SESSIONS_DIR = Path(td) / "sessions"
            mod.PREFS_FILE = Path(td) / "preferences.yaml"
            mod.PREFS_FILE.write_text(yaml.safe_dump(
                {"vram": 16, "last_agent": "codex", "gpu_vendor": "amd"},
                sort_keys=False))

            prefs = mod.read_prefs()
            self.assertNotIn("gpu_vendor", prefs)
            self.assertEqual(prefs["vram"], 16)
            self.assertEqual(prefs["last_agent"], "codex")

            mod.write_prefs(prefs)
            self.assertNotIn("gpu_vendor", yaml.safe_load(mod.PREFS_FILE.read_text()))


if __name__ == "__main__":
    unittest.main()
