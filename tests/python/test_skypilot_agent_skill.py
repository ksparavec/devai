"""Tests for the skypilot-agent-skill scaffold.

Covers:
  - SkyPilot is pinned (requirements-skypilot.in) and hash-locked
    (requirements-skypilot.lock, `make lab-lock`).
  - Dockerfile.lab installs it into its own venv (/opt/skypilot), not the
    lab's Python, and links only `sky` onto PATH.
  - scripts/sky-setup.sh exists and is bash-syntax-valid.
  - docs/skypilot-user-guide.md covers the operator surface.

Real PyPI fetches and image rebuilds are out of scope for unit
tests -- they require network access and ~30 min of build time.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


EXTRAS = "skypilot[aws,gcp,azure,kubernetes,slurm,runpod,lambda]"


class TestSkypilotLock(unittest.TestCase):
    def test_version_is_pinned_with_the_broad_cloud_extras(self) -> None:
        # Per skypilot-agent-skill plan decision 2: broad set so the
        # lab image can drive any cloud the operator has creds for.
        src = (REPO_ROOT / "requirements-skypilot.in").read_text()
        self.assertRegex(src, r"(?m)^" + re.escape(EXTRAS) + r"==\d+\.\d+\.\d+$")

    def test_lock_pins_that_version_with_hashes(self) -> None:
        src = (REPO_ROOT / "requirements-skypilot.in").read_text()
        version = re.search(re.escape(EXTRAS) + r"==(\S+)", src).group(1)
        lock = (REPO_ROOT / "requirements-skypilot.lock").read_text()
        self.assertRegex(lock, r"(?m)^skypilot==" + re.escape(version) + r" \\$")
        self.assertIn("--hash=sha256:", lock)

    def test_make_lab_lock_writes_it(self) -> None:
        make = (REPO_ROOT / "Makefile").read_text()
        self.assertIn("pip compile requirements-skypilot.in", make)
        self.assertIn("-o requirements-skypilot.lock", make)

    def test_the_old_offline_wheel_prefetch_is_gone(self) -> None:
        # It fetched the NEWEST release, i.e. nothing was pinned.
        make = (REPO_ROOT / "Makefile").read_text()
        self.assertNotIn("pip/wheels/skypilot", make)
        self.assertNotIn("SKY_HASH", make)


class TestDockerfileLabSkypilotInstall(unittest.TestCase):
    def setUp(self) -> None:
        self.text = (REPO_ROOT / "deploy" / "Dockerfile.lab").read_text()

    def test_installed_into_its_own_venv_from_the_lock(self) -> None:
        self.assertIn('uv venv --python "$(command -v python3)" /opt/skypilot', self.text)
        self.assertIn(
            "uv pip install --python /opt/skypilot/bin/python --no-deps --require-hashes",
            self.text)
        self.assertIn("-r /tmp/requirements-skypilot.lock", self.text)
        self.assertIn("uv pip check --python /opt/skypilot/bin/python", self.text)
        self.assertIn("ln -s /opt/skypilot/bin/sky /usr/local/bin/sky", self.text)

    def test_never_installed_into_the_lab_python(self) -> None:
        # SkyPilot's caps (click <8.2, websocket-client ==1.3.3, ...)
        # downgraded lab packages when it shared the system Python.
        self.assertNotRegex(self.text, r"uv pip install --system[^\n]*skypilot")
        self.assertNotIn("/var/cache/wheels/skypilot", self.text)


class TestSkySetupScript(unittest.TestCase):
    def test_exists_and_executable(self) -> None:
        path = REPO_ROOT / "scripts" / "sky-setup.sh"
        self.assertTrue(path.is_file())
        # Mode bit check (0o100) on POSIX; skip on Windows.
        st = path.stat()
        self.assertTrue(st.st_mode & 0o100, msg="sky-setup.sh not executable")

    def test_bash_syntax_valid(self) -> None:
        path = REPO_ROOT / "scripts" / "sky-setup.sh"
        r = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(r.returncode, 0, msg=r.stderr)

    def test_handles_sky_missing_with_clear_error(self) -> None:
        text = (REPO_ROOT / "scripts" / "sky-setup.sh").read_text()
        self.assertIn("sky CLI not on PATH", text)
        self.assertIn("/opt/skypilot", text)


class TestDocsSkypilotUserGuide(unittest.TestCase):
    def setUp(self) -> None:
        self.path = REPO_ROOT / "docs" / "skypilot-user-guide.md"

    def test_present(self) -> None:
        self.assertTrue(self.path.is_file())

    def test_covers_required_sections(self) -> None:
        text = self.path.read_text()
        for header in (
            "What you get",
            "First-time setup",
            "Per-cloud credential setup",
            "Hello-world",
            "Driving SkyPilot from an AI agent",
            "Cost guidance",
            "Troubleshooting",
        ):
            self.assertIn(
                header, text, msg=f"docs/skypilot-user-guide.md missing: {header}"
            )

    def test_links_to_sibling_plan(self) -> None:
        text = self.path.read_text()
        self.assertIn("skypilot-fleet-provisioner.md", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
