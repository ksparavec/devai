"""Tests for the skypilot-agent-skill scaffold.

Covers:
  - Makefile fetch-cli has the SkyPilot wheel-download block.
  - CACHE_BUILD_ARGS conditionally mounts the SkyPilot wheel dir.
  - Dockerfile.lab installs SkyPilot via uv pip --offline --find-links.
  - scripts/sky-setup.sh exists and is bash-syntax-valid.
  - docs/skypilot-user-guide.md covers the operator surface.

Real PyPI fetches and image rebuilds are out of scope for unit
tests -- they require network access and ~30 min of build time.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class TestMakefileFetchCli(unittest.TestCase):
    def setUp(self) -> None:
        self.text = (REPO_ROOT / "Makefile").read_text()

    def test_skypilot_wheel_block_present(self) -> None:
        self.assertIn("skypilot.version", self.text)
        self.assertIn("pip/wheels/skypilot", self.text)
        # The wheel fetch uses `python3 -m pip download` (not `uv pip
        # download`, which has no `download` subcommand) with
        # `--only-binary=:all:` so only stable pre-built wheels are
        # pulled. See commit 91d8e15 (repair SkyPilot wheel fetch).
        self.assertIn("python3 -m pip download", self.text)
        self.assertIn("--only-binary=:all:", self.text)

    def test_broad_cloud_extras(self) -> None:
        # Per skypilot-agent-skill plan decision 2: broad set so the
        # lab image can drive any cloud the operator has creds for.
        for extra in ("aws", "gcp", "azure", "kubernetes", "slurm", "runpod", "lambda"):
            self.assertIn(extra, self.text, msg=f"missing extra: {extra}")

    def test_cache_build_args_includes_skypilot_wheel_mount(self) -> None:
        # The mount is conditional on the wheel dir existing on the
        # host -- so the build doesn't fail when fetch-cli was skipped.
        self.assertIn("/var/cache/wheels/skypilot", self.text)
        self.assertIn(
            "$(if $(wildcard $(CACHE_DIR)/pip/wheels/skypilot)", self.text
        )


class TestDockerfileLabSkypilotInstall(unittest.TestCase):
    def setUp(self) -> None:
        self.text = (REPO_ROOT / "deploy" / "Dockerfile.lab").read_text()

    def test_uv_pip_install_offline_block(self) -> None:
        self.assertIn("uv pip install --system --offline", self.text)
        self.assertIn("--find-links /var/cache/wheels/skypilot", self.text)
        # The exact extras string must match the fetch-cli block.
        self.assertIn(
            "'skypilot[aws,gcp,azure,kubernetes,slurm,runpod,lambda]'",
            self.text,
        )

    def test_install_is_optional(self) -> None:
        # Per the plan: the build still succeeds when the wheel cache
        # is empty (a CI/firewalled environment with no fetch-cli run).
        self.assertIn(
            "/var/cache/wheels/skypilot", self.text
        )
        # Skip-block lives inside an `if [ -d ... ]` guard. re.search
        # with DOTALL because the block spans multiple lines.
        import re
        self.assertIsNotNone(
            re.search(
                r"if \[ -d /var/cache/wheels/skypilot \].*?else.*?skipping",
                self.text,
                re.DOTALL,
            )
        )


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
        self.assertIn("uv pip install", text)


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
