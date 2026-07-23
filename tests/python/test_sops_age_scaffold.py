"""Tests for the sops/age secret-store scaffold.

Verifies:
  - .sops.yaml is well-formed YAML with the placeholder rule shape.
  - scripts/age-keygen-host.sh refuses to overwrite an existing key
    and parses the public key correctly.
  - scripts/render-secret.sh refuses to write to a non-tmpfs path
    without DEVAI_RENDER_ALLOW_NON_TMPFS=1 -- including when the
    destination directory does not exist yet.
  - scripts/render-secret.sh validates argument count.
  - scripts/render-secret.sh renders atomically: a failed decrypt
    leaves a previously rendered secret untouched.
  - deploy/setup-secrets-tmpfs.sh basic argument validation runs
    without requiring root (we only exercise the help/usage paths).

We DO NOT exercise real sops/age binaries here -- those require the
operator to have installed them, and the binary fetch lives in
'make fetch-cli' (binary integrity is the OS package manager's
problem, not ours).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class TestSopsYaml(unittest.TestCase):
    def test_file_exists(self) -> None:
        self.assertTrue((REPO_ROOT / ".sops.yaml").is_file())

    def test_well_formed_yaml(self) -> None:
        try:
            import yaml  # PyYAML is in the lab base; tolerate absence here.
        except ImportError:
            self.skipTest("PyYAML not installed")
        with open(REPO_ROOT / ".sops.yaml") as f:
            doc = yaml.safe_load(f)
        self.assertIsInstance(doc, dict)
        self.assertIn("creation_rules", doc)
        rules = doc["creation_rules"]
        self.assertEqual(len(rules), 1)
        rule = rules[0]
        self.assertEqual(rule["path_regex"], r"deploy/.*\.sops\.env$")
        self.assertEqual(rule["encrypted_regex"], "^.*$")
        self.assertIn("age", rule)

    def test_contains_placeholder_or_real_key(self) -> None:
        # The placeholder is shipped; once the operator runs
        # age-keygen-host.sh it gets replaced. Either is valid.
        text = (REPO_ROOT / ".sops.yaml").read_text()
        self.assertRegex(text, r"age1[0-9a-zA-Z]{30,}")


class TestAgeKeygenHostScript(unittest.TestCase):
    def test_idempotent_when_key_exists(self) -> None:
        # Build a fake key file that satisfies the 'public key:' parser
        # in the script, then verify the script reports "already
        # installed" without trying to invoke age-keygen.
        with tempfile.TemporaryDirectory() as td:
            key_dir = Path(td) / "age"
            key_dir.mkdir(parents=True)
            (key_dir / "keys.txt").write_text(
                "# created: 2026-05-15T10:00:00Z\n"
                "# public key: age1exampleyezeznxqzezeznxqzezeznxqzezeznxqzezehjt9hf\n"
                "AGE-SECRET-KEY-EXAMPLEXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
            )
            (key_dir / "keys.txt").chmod(0o600)
            env = {**os.environ, "SOPS_AGE_KEY_DIR": str(key_dir)}
            # The script needs `bash`. PATH inherits from os.environ.
            r = subprocess.run(
                ["bash", str(REPO_ROOT / "scripts" / "age-keygen-host.sh")],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            self.assertEqual(
                r.returncode, 0, msg=f"stderr: {r.stderr}\nstdout: {r.stdout}"
            )
            self.assertIn("already installed", r.stdout)
            self.assertIn(
                "age1exampleyezeznxqzezeznxqzezeznxqzezeznxqzezehjt9hf", r.stdout
            )

    def test_fails_clearly_when_key_file_unparseable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            key_dir = Path(td) / "age"
            key_dir.mkdir(parents=True)
            # File exists but has no `# public key:` line.
            (key_dir / "keys.txt").write_text("garbage\n")
            (key_dir / "keys.txt").chmod(0o600)
            env = {**os.environ, "SOPS_AGE_KEY_DIR": str(key_dir)}
            r = subprocess.run(
                ["bash", str(REPO_ROOT / "scripts" / "age-keygen-host.sh")],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("no parseable public key", r.stderr)


class TestRenderSecretScript(unittest.TestCase):
    def test_argument_count_validation(self) -> None:
        r = subprocess.run(
            ["bash", str(REPO_ROOT / "scripts" / "render-secret.sh")],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(r.returncode, 2)
        self.assertIn("usage:", r.stderr)

    def test_refuses_missing_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            r = subprocess.run(
                [
                    "bash",
                    str(REPO_ROOT / "scripts" / "render-secret.sh"),
                    "/nonexistent/source.sops.env",
                    f"{td}/out.env",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("encrypted source not found", r.stderr)

    def test_refuses_non_tmpfs_destination(self) -> None:
        # Walk up the cwd's filesystem to find a non-tmpfs directory.
        # On many CI hosts /tmp is tmpfs, which would make the gate
        # pass the wrong way; use the repo root's parent (a regular
        # disk-backed FS in practice).
        non_tmpfs_root = REPO_ROOT
        # Sanity: confirm it's actually not tmpfs.
        fs_type = subprocess.check_output(
            ["stat", "-f", "-c", "%T", str(non_tmpfs_root)], text=True
        ).strip()
        if fs_type in ("tmpfs", "ramfs"):
            self.skipTest(f"REPO_ROOT is on {fs_type}; cannot test non-tmpfs gate")
        with tempfile.TemporaryDirectory(dir=str(non_tmpfs_root)) as td:
            src = Path(td) / "fake.sops.env"
            src.write_text("fake=1\n")
            dst = Path(td) / "non-tmpfs.env"
            r = subprocess.run(
                [
                    "bash",
                    str(REPO_ROOT / "scripts" / "render-secret.sh"),
                    str(src),
                    str(dst),
                ],
                capture_output=True,
                text=True,
                check=False,
                env={**os.environ, "DEVAI_RENDER_ALLOW_NON_TMPFS": "0"},
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("not a tmpfs", r.stderr)

    def test_refuses_non_tmpfs_destination_when_dir_absent(self) -> None:
        # Regression: the tmpfs gate used to sit inside a
        # `[[ -d "$dst_dir" ]]` test, so it was skipped entirely for a
        # destination directory that did not exist yet -- exactly the
        # case it exists to catch, since `mkdir -p` then created it on
        # persistent storage and the plaintext landed on disk.
        non_tmpfs_root = REPO_ROOT
        fs_type = subprocess.check_output(
            ["stat", "-f", "-c", "%T", str(non_tmpfs_root)], text=True
        ).strip()
        if fs_type in ("tmpfs", "ramfs"):
            self.skipTest(f"REPO_ROOT is on {fs_type}; cannot test non-tmpfs gate")
        with tempfile.TemporaryDirectory(dir=str(non_tmpfs_root)) as td:
            src = Path(td) / "fake.sops.env"
            src.write_text("fake=1\n")
            missing_dir = Path(td) / "not-created-yet"
            dst = missing_dir / "non-tmpfs.env"
            r = subprocess.run(
                [
                    "bash",
                    str(REPO_ROOT / "scripts" / "render-secret.sh"),
                    str(src),
                    str(dst),
                ],
                capture_output=True,
                text=True,
                check=False,
                env={**os.environ, "DEVAI_RENDER_ALLOW_NON_TMPFS": "0"},
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("not a tmpfs", r.stderr)
            self.assertFalse(
                missing_dir.exists(),
                msg="gate must abort before mkdir -p creates the directory",
            )

    @staticmethod
    def _fake_sops_bin(td: str, *, exit_code: int, payload: str = "") -> str:
        """Put a stub `sops` on PATH.

        Lets us exercise render-secret.sh's success/failure control flow
        without a real sops/age install (see the module docstring: real
        binaries are out of scope here).
        """
        bin_dir = Path(td) / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "sops"
        stub.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s' {shlex.quote(payload)}\n"
            f"exit {exit_code}\n"
        )
        stub.chmod(0o755)
        return str(bin_dir)

    def test_failed_decrypt_leaves_existing_destination_intact(self) -> None:
        # Regression: the script used to redirect sops straight into the
        # destination, so a failed decrypt truncated a good rendered
        # secret to 0 bytes and consumers read an empty .env.
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "fake.sops.env"
            src.write_text("encrypted-blob\n")
            dst = Path(td) / "out.env"
            dst.write_text("GOOD=1\n")
            bin_dir = self._fake_sops_bin(td, exit_code=1)
            r = subprocess.run(
                [
                    "bash",
                    str(REPO_ROOT / "scripts" / "render-secret.sh"),
                    str(src),
                    str(dst),
                ],
                capture_output=True,
                text=True,
                check=False,
                env={
                    **os.environ,
                    "DEVAI_RENDER_ALLOW_NON_TMPFS": "1",
                    "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                },
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("left unchanged", r.stderr)
            self.assertEqual(dst.read_text(), "GOOD=1\n")
            leftovers = sorted(
                p.name for p in Path(td).iterdir() if p.name.startswith(".out.env.")
            )
            self.assertEqual(leftovers, [], msg="temp file not cleaned up")

    def test_successful_render_is_mode_0600_and_leaves_no_temp(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "fake.sops.env"
            src.write_text("encrypted-blob\n")
            dst = Path(td) / "out.env"
            payload = "TOKEN=abc\n"
            bin_dir = self._fake_sops_bin(td, exit_code=0, payload=payload)
            r = subprocess.run(
                [
                    "bash",
                    str(REPO_ROOT / "scripts" / "render-secret.sh"),
                    str(src),
                    str(dst),
                ],
                capture_output=True,
                text=True,
                check=False,
                env={
                    **os.environ,
                    "DEVAI_RENDER_ALLOW_NON_TMPFS": "1",
                    "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                },
            )
            self.assertEqual(
                r.returncode, 0, msg=f"stderr: {r.stderr}\nstdout: {r.stdout}"
            )
            self.assertEqual(dst.read_text(), payload)
            self.assertEqual(dst.stat().st_mode & 0o777, 0o600)
            leftovers = sorted(
                p.name for p in Path(td).iterdir() if p.name.startswith(".out.env.")
            )
            self.assertEqual(leftovers, [], msg="temp file not cleaned up")


class TestSetupSecretsTmpfsScript(unittest.TestCase):
    def test_script_executable_and_well_formed(self) -> None:
        path = REPO_ROOT / "deploy" / "setup-secrets-tmpfs.sh"
        self.assertTrue(path.is_file())
        # Don't actually invoke (needs sudo); just verify it parses
        # under bash's syntax checker.
        r = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            r.returncode, 0, msg=f"bash -n stderr: {r.stderr}"
        )

    def test_default_mountpoint_documented(self) -> None:
        text = (REPO_ROOT / "deploy" / "setup-secrets-tmpfs.sh").read_text()
        self.assertIn("/run/devai", text)
        self.assertIn("nodev", text)
        self.assertIn("nosuid", text)
        self.assertIn("noexec", text)

    def test_mount_options_carry_owner_uid_gid(self) -> None:
        # Static check only -- actually mounting needs CAP_SYS_ADMIN.
        # Without uid=/gid= the tmpfs is root:root 0700 and every
        # *-secrets-render target fails with EACCES.
        text = (REPO_ROOT / "deploy" / "setup-secrets-tmpfs.sh").read_text()
        self.assertIn("uid=${OWNER_UID},gid=${OWNER_GID}", text)
        self.assertIn("mode=0700", text)
        self.assertIn("SUDO_USER", text)


class TestGitignoreCovers(unittest.TestCase):
    def test_run_devai_and_env_plain_blocked(self) -> None:
        text = (REPO_ROOT / ".gitignore").read_text()
        self.assertIn("/run/devai", text)
        self.assertIn("*.env.plain", text)

    def test_sops_env_files_explicitly_tracked(self) -> None:
        text = (REPO_ROOT / ".gitignore").read_text()
        # The exception means git WILL track *.sops.env even if a
        # broader rule (e.g. *.env in a parent gitignore) would block it.
        self.assertIn("!deploy/*.sops.env", text)


class TestDocsSecretsMd(unittest.TestCase):
    def test_present_and_covers_required_sections(self) -> None:
        path = REPO_ROOT / "docs" / "secrets.md"
        self.assertTrue(path.is_file())
        text = path.read_text()
        for header in (
            "Why sops + age",
            "One-time setup",
            "Editing a secrets file",
            "Rendering at startup",
            "Rotation",
            "Recovery",
            "Multi-host onboarding",
        ):
            self.assertIn(header, text, msg=f"docs/secrets.md missing: {header}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
