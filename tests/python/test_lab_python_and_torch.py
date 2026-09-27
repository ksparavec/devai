"""The lab's Python is pinned exactly, and torch is installed in its own step.

Python: the lab moved from 3.13 (minor pin, "latest patch at build time") to an
EXACT pin, 3.14.7, on 2026-09-24 by operator decision. One variable drives the
base image and the lock resolves (`make lab-lock`); a lock resolved for another
Python than the image's could name wheels the image cannot install.

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

Pinning (2026-09-27, operator decision): torch/torchvision/torchaudio are exact
in requirements-torch.txt, and everything else comes from requirements-lab.lock
-- the hash-locked closure of requirements-base.txt, resolved against that torch
-- installed with --no-deps --require-hashes and followed by `uv pip check`.
inspect-ai in particular is pinned at 0.3.271, the release the bench harness
fixes were checked against.

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


LOCK = (REPO_ROOT / "requirements-lab.lock").read_text()
TORCH_PINS = (REPO_ROOT / "requirements-torch.txt").read_text()
TORCH_STACK = ("torch", "torchvision", "torchaudio")


def _locked() -> dict[str, str]:
    return dict(re.findall(r"(?m)^([A-Za-z0-9_.\-]+)==(\S+) \\$", LOCK))


def _requirements_step() -> str:
    """The RUN instruction that installs the torch pins and the lock."""
    start = LAB.index("COPY requirements-torch.txt requirements-lab.lock")
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

    def test_locks_are_resolved_for_the_pinned_python(self) -> None:
        self.assertIn("--python-version $(PYTHON_VERSION)", MAKEFILE)
        self.assertNotRegex(MAKEFILE, r"--python-version 3\.\d")


class TorchStepTest(unittest.TestCase):
    def test_torch_is_installed_before_and_apart_from_the_lock(self) -> None:
        step = _requirements_step()
        torch_at = step.index("-r /tmp/requirements-torch.txt")
        lock_at = step.index("-r /tmp/requirements-lab.lock")
        self.assertLess(torch_at, lock_at)
        # The lock install names no index: PyPI alone.
        lock_cmd = step[step.rindex("uv pip install", 0, lock_at):lock_at]
        self.assertNotIn("index-url", lock_cmd)

    def test_no_extra_index_anywhere_in_the_step(self) -> None:
        self.assertNotIn("--extra-index-url", _requirements_step())

    def test_cpu_and_rocm_torch_come_from_their_own_index(self) -> None:
        step = _requirements_step()
        self.assertIn("--index-url https://download.pytorch.org/whl/cpu", step)
        self.assertIn("--index-url https://download.pytorch.org/whl/rocm6.4", step)

    def test_torch_stack_is_pinned_exactly(self) -> None:
        for name in TORCH_STACK:
            self.assertRegex(TORCH_PINS, rf"(?m)^{name}==\d+\.\d+\.\d+$")


class LockTest(unittest.TestCase):
    def test_lock_is_installed_hash_checked_without_resolving(self) -> None:
        step = _requirements_step()
        self.assertIn(
            "uv pip install --system --no-deps --require-hashes -r /tmp/requirements-lab.lock",
            step)
        self.assertIn("uv pip check --system", step)
        self.assertLess(step.index("-r /tmp/requirements-lab.lock"),
                        step.index("uv pip check --system"))

    def test_requirements_base_is_no_longer_installed_directly(self) -> None:
        # It is the lock's INPUT; installing it would re-resolve, unpinned.
        self.assertNotRegex(LAB, r"(?m)^(COPY|RUN)[^\n]*requirements-base\.txt")
        self.assertNotIn("-r /tmp/requirements-base.txt", LAB)

    def test_every_locked_package_has_an_exact_version_and_a_hash(self) -> None:
        entries = re.split(r"(?m)^(?=[A-Za-z0-9])", LOCK)
        entries = [e for e in entries if not e.startswith("#") and e.strip()]
        self.assertGreater(len(entries), 100)
        for e in entries:
            self.assertRegex(e.splitlines()[0], r"^[A-Za-z0-9_.\-]+==\S+ \\$", msg=e[:80])
            self.assertIn("--hash=sha256:", e, msg=e[:80])

    def test_inspect_ai_is_pinned_at_0_3_271(self) -> None:
        base = (REPO_ROOT / "requirements-base.txt").read_text()
        self.assertRegex(base, r"(?m)^inspect-ai==0\.3\.271$")
        self.assertEqual(_locked().get("inspect-ai"), "0.3.271")

    def test_torch_stack_and_skypilot_are_not_in_the_lab_lock(self) -> None:
        locked = _locked()
        for name in TORCH_STACK + ("triton", "skypilot"):
            self.assertNotIn(name, locked)
        self.assertFalse([n for n in locked if n.startswith("nvidia-")])

    def test_lock_is_resolved_against_the_torch_pins(self) -> None:
        self.assertIn("pip compile requirements-base.txt -c requirements-torch.txt", MAKEFILE)
        for name in TORCH_STACK:
            self.assertIn(f"--no-emit-package {name}", MAKEFILE)


if __name__ == "__main__":
    unittest.main()
