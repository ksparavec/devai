"""The lab's Python is pinned exactly, and torch is installed in its own step.

Python: the lab moved from 3.13 (minor pin, "latest patch at build time") to an
EXACT pin, 3.14.7, on 2026-09-24 by operator decision. One variable drives the
base image and the SkyPilot wheel pre-fetch; a pre-fetch for a different
Python than the image's would leave the offline `--no-index` SkyPilot install
without matching wheels.

Torch: the CPU and ROCm builds used to pass PyTorch's index together with the
whole requirement set. uv's default index strategy then takes every package
that index carries (numpy, jinja2, certifi, idna, fsspec, ...) from it ONLY, and
the resolver backtracked the rest of the lab to old releases -- measured on the
3.13 CPU image: JupyterLab 4.1.6 (2024-04) instead of 4.6.4, inspect-ai 0.3.69
instead of 0.3.268, 33 packages behind a plain resolve. On 3.14 that same
backtracking reached releases with no 3.14 build and the resolve failed
outright. Torch now comes from PyTorch's index in its own step and the
requirements resolve against PyPI alone; the installed torch satisfies every
later `torch>=` requirement, so it is kept (verified on 3.14.7: 2.14.0+cpu kept,
JupyterLab 4.6.4, inspect-ai 0.3.268).

Stdlib unittest only.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MAKEFILE = (REPO_ROOT / "Makefile").read_text()
BASE = (REPO_ROOT / "deploy" / "Dockerfile.base").read_text()
LAB = (REPO_ROOT / "deploy" / "Dockerfile.lab").read_text()


def _requirements_step() -> str:
    """The RUN instruction that installs requirements-base.txt."""
    start = LAB.index("COPY requirements-base.txt")
    end = LAB.index("\n\n", start)
    return LAB[start:end]


class PythonPinTest(unittest.TestCase):
    def test_makefile_pins_an_exact_3_14_release(self) -> None:
        m = re.search(r"(?m)^PYTHON_VERSION = (\S+)$", MAKEFILE)
        self.assertIsNotNone(m)
        self.assertRegex(m.group(1), r"^3\.14\.\d+$")

    def test_base_default_matches_the_makefile(self) -> None:
        make = re.search(r"(?m)^PYTHON_VERSION = (\S+)$", MAKEFILE).group(1)
        self.assertIn(f"ARG PYTHON_VERSION={make}", BASE)

    def test_skypilot_wheels_follow_the_pin(self) -> None:
        self.assertIn("--python-version $(PYTHON_VERSION)", MAKEFILE)
        self.assertNotRegex(MAKEFILE, r"--python-version 3\.\d")

    def test_skypilot_closure_is_resolved_by_uv_and_downloaded_without_resolving(self) -> None:
        """pip's resolver gives up on this closure for Python 3.14
        (`resolution-too-deep`, measured 2026-09-24) while uv resolves it in
        seconds: uv pins the closure, pip only downloads the pinned wheels."""
        self.assertIn("$(CACHE_DIR)/pip/bin/uv pip compile -q -", MAKEFILE)
        self.assertIn('pip download --no-deps -r "$$SKY_TMP/.download.txt"', MAKEFILE)

    def test_skypilot_resolve_is_pinned_to_latest_with_a_hostlist_wheel(self) -> None:
        """SkyPilot 0.13.0's slurm extra needs python-hostlist, which ships only
        as source; a binary-only resolve silently fell back to 0.11.1 while the
        stamp claimed 0.13.0. The wheel is built first, and the resolve asks for
        exactly ==LATEST, so a fallback now fails instead of passing quietly."""
        self.assertIn('pip wheel -q --no-deps python-hostlist -w "$$SKY_TMP"', MAKEFILE)
        self.assertIn(
            '"skypilot[aws,gcp,azure,kubernetes,slurm,runpod,lambda]==$$LATEST"', MAKEFILE)
        self.assertIn('--find-links "$$SKY_TMP"', MAKEFILE)

    def test_skypilot_stamp_includes_the_python_version(self) -> None:
        """Cached wheels for another Python must count as stale: the lab's
        SkyPilot install is offline and only WARNS on failure, so reusing
        them would silently ship an image without `sky`."""
        self.assertIn('echo "$$LATEST py$(PYTHON_VERSION)" > $(ETAG_DIR)/skypilot.version', MAKEFILE)
        self.assertIn('[ "$$LATEST py$(PYTHON_VERSION)" = "$$CACHED" ]', MAKEFILE)


class SkypilotLayerTest(unittest.TestCase):
    def test_skypilot_layer_is_keyed_on_the_wheel_set(self) -> None:
        """The wheels arrive by bind mount, which podman's layer cache ignores;
        without a content key a changed wheel set was silently not installed
        (2026-09-24: cache held 0.13.0, image kept 0.11.1)."""
        self.assertIn("--build-arg SKY_HASH=$(SKY_HASH)", MAKEFILE)
        self.assertRegex(MAKEFILE, r"(?m)^SKY_HASH = \$\(shell ls \$\(CACHE_DIR\)/pip/wheels/skypilot")
        self.assertIn("ARG SKY_HASH=unknown", LAB)
        self.assertIn('echo "skypilot wheels: SKY_HASH=$SKY_HASH"', LAB)


class TorchStepTest(unittest.TestCase):
    def test_torch_is_installed_before_and_apart_from_the_requirements(self) -> None:
        step = _requirements_step()
        torch_at = step.index("torch torchvision torchaudio")
        reqs_at = step.index("-r /tmp/requirements-base.txt")
        self.assertLess(torch_at, reqs_at)
        # The requirements install names no index: PyPI alone.
        reqs_cmd = step[step.rindex("uv pip install", 0, reqs_at):reqs_at]
        self.assertNotIn("index-url", reqs_cmd)
        self.assertNotIn("torch", reqs_cmd)

    def test_no_extra_index_anywhere_in_the_step(self) -> None:
        self.assertNotIn("--extra-index-url", _requirements_step())

    def test_cpu_and_rocm_torch_come_from_their_own_index(self) -> None:
        step = _requirements_step()
        self.assertIn("--index-url https://download.pytorch.org/whl/cpu", step)
        self.assertIn("--index-url https://download.pytorch.org/whl/rocm6.4", step)


if __name__ == "__main__":
    unittest.main()
