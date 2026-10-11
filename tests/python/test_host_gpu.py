"""scripts/host-gpu.sh and its use by the Makefile (platform-rebuild Phase 3, M15).

The host check prints the CDI device string when an NVIDIA GPU is usable
(`nvidia-smi -L` lists one AND the CDI spec names nvidia.com/gpu=all), and
nothing otherwise; DEVAI_GPU_DEVICE in the environment overrides it, empty
forcing the CPU. The Makefile takes DEVAI_GPU_DEVICE from it unless the
variable comes from the environment or the command line, and attaches a
device only when the result is non-empty.

Runs against fake nvidia-smi / nvidia-ctk on a PATH that holds nothing else
the script could find, so the host's real GPU does not leak in.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "host-gpu.sh"


class HostGpuTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.bin = Path(self._tmp.name) / "bin"
        self.bin.mkdir()
        # Only what the script needs: bash, grep, printf/echo are builtins.
        for tool in ("bash", "grep"):
            (self.bin / tool).symlink_to(shutil.which(tool))

    def _fake(self, name: str, output: str, rc: int = 0) -> None:
        f = self.bin / name
        f.write_text(f"#!{shutil.which('bash')}\nprintf '%s' '{output}'\nexit {rc}\n")
        f.chmod(0o755)

    def _run(self, **env: str) -> str:
        full = {"PATH": str(self.bin), **env}
        r = subprocess.run([str(self.bin / "bash"), str(SCRIPT)], env=full,
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def _gpu_host(self) -> None:
        self._fake("nvidia-smi", "GPU 0: NVIDIA RTX PRO 4000 Blackwell (UUID: GPU-x)\n")
        self._fake("nvidia-ctk", "nvidia.com/gpu=0\nnvidia.com/gpu=all\n")

    def test_gpu_and_cdi_spec_give_the_device(self) -> None:
        self._gpu_host()
        self.assertEqual(self._run(), "nvidia.com/gpu=all\n")

    def test_no_nvidia_smi_is_a_cpu_host(self) -> None:
        self._fake("nvidia-ctk", "nvidia.com/gpu=all\n")
        self.assertEqual(self._run(), "")

    def test_nvidia_smi_listing_no_gpu_is_a_cpu_host(self) -> None:
        self._fake("nvidia-smi", "No devices found.\n", rc=6)
        self._fake("nvidia-ctk", "nvidia.com/gpu=all\n")
        self.assertEqual(self._run(), "")

    def test_gpu_without_a_cdi_spec_is_a_cpu_host(self) -> None:
        # podman could not attach the device, so the GPU is not usable.
        self._fake("nvidia-smi", "GPU 0: x\n")
        self._fake("nvidia-ctk", "")
        self.assertEqual(self._run(), "")
        (self.bin / "nvidia-ctk").unlink()
        self.assertEqual(self._run(), "")

    def test_environment_overrides_and_empty_forces_cpu(self) -> None:
        self._gpu_host()
        self.assertEqual(self._run(DEVAI_GPU_DEVICE=""), "")
        self.assertEqual(self._run(DEVAI_GPU_DEVICE="nvidia.com/gpu=0"), "nvidia.com/gpu=0\n")


class MakefileUsesTheHostCheckTest(unittest.TestCase):
    def _make_n(self, *args: str, env_extra: dict | None = None,
                hide_gpu: bool = False) -> str:
        env = dict(os.environ)
        env.pop("DEVAI_GPU_DEVICE", None)
        env.update(env_extra or {})
        if hide_gpu:
            with tempfile.TemporaryDirectory() as tmp:
                # PATH without nvidia-smi / nvidia-ctk, with what make and the
                # recipe's $(shell ...) calls need.
                for tool in ("bash", "sh", "grep", "awk", "hostname", "cat", "md5sum",
                             "cut", "readlink", "podman", "id", "uname"):
                    path = shutil.which(tool)
                    if path:
                        (Path(tmp) / tool).symlink_to(path)
                env["PATH"] = tmp
                return self._run(env, args)
        return self._run(env, args)

    @staticmethod
    def _run(env: dict, args: tuple) -> str:
        r = subprocess.run([shutil.which("make"), "-n", "--no-print-directory", *args],
                           cwd=REPO, env=env, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise AssertionError(r.stderr)
        return r.stdout

    def test_no_gpu_means_no_device(self) -> None:
        out = self._make_n("lab", hide_gpu=True)
        self.assertNotIn("--device", out)
        self.assertNotIn("--gpus", out)

    def test_empty_on_the_command_line_forces_cpu(self) -> None:
        self.assertNotIn("--device", self._make_n("lab", "DEVAI_GPU_DEVICE="))

    def test_empty_in_the_environment_forces_cpu(self) -> None:
        self.assertNotIn("--device", self._make_n("lab", env_extra={"DEVAI_GPU_DEVICE": ""}))

    def test_an_explicit_device_is_used(self) -> None:
        out = self._make_n("lab", "DEVAI_GPU_DEVICE=nvidia.com/gpu=0")
        self.assertIn("--device nvidia.com/gpu=0 --security-opt=label=disable", out)

    def test_makefile_wiring(self) -> None:
        mk = (REPO / "Makefile").read_text()
        self.assertIn("DEVAI_GPU_DEVICE := $(shell $(CURDIR)/scripts/host-gpu.sh)", mk)
        self.assertNotIn("DEVAI_GPU_DEVICE ?=", mk)
        self.assertNotIn("$(or $(DEVAI_GPU_DEVICE),nvidia.com/gpu=all)", mk)
        for gone in ("build-base-cpu", "build-cpu", "lab-cpu", "shell-cpu",
                     "build-base-gpu", "build-gpu", "lab-gpu", "shell-gpu"):
            self.assertNotRegex(mk, rf"(?m)^{gone}:", gone)
        for target in ("build-base", "build-lab", "lab", "shell"):
            self.assertRegex(mk, rf"(?m)^{target}:.*## ", target)


if __name__ == "__main__":
    unittest.main()
