"""Tests for skypilot-fleet-provisioner Phase 1 deliverables.

Covers:
  - docker-compose.yaml gains the devai-skypilot-api-server service
    with the right image pin, port, profile, secrets mount.
  - skypilot-state named volume is declared.
  - deploy/skypilot-api.env documents SKYPILOT_API_PORT.
  - deploy/skypilot-credentials.sops.env.example documents the 3
    expected secrets (RUNPOD_API_KEY, LAMBDA_API_KEY,
    SKYPILOT_API_TOKEN).
  - scripts/skypilot-api-health.sh exists and is bash-syntax-valid.
  - Makefile targets present in .PHONY.
  - docs/skypilot.md covers the operator surface.

Live cloud provisioning test (sky launch + sky down against RunPod)
is deferred to E2E -- requires real credentials and ~$1 budget.
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

EXPECTED_SECRETS = {"RUNPOD_API_KEY", "LAMBDA_API_KEY", "SKYPILOT_API_TOKEN"}


def _yaml():
    try:
        import yaml
        return yaml
    except ImportError:
        return None


class TestComposeService(unittest.TestCase):
    def setUp(self) -> None:
        self.yaml = _yaml()
        if self.yaml is None:
            self.skipTest("PyYAML not installed")
        with open(REPO_ROOT / "deploy" / "docker-compose.yaml") as f:
            self.compose = self.yaml.safe_load(f)
        self.svc = (self.compose.get("services") or {}).get("skypilot-api-server")

    def test_service_present(self) -> None:
        self.assertIsNotNone(self.svc, "skypilot-api-server service missing")

    def test_image_pinned(self) -> None:
        img = self.svc["image"]
        self.assertNotIn(":latest", img)
        self.assertRegex(img, r":\d+\.\d+\.\d+")
        self.assertIn("berkeleyskypilot/skypilot", img)

    def test_container_name_matches_devai_convention(self) -> None:
        self.assertEqual(self.svc["container_name"], "devai-skypilot-api-server")

    def test_publishes_46580(self) -> None:
        ports = self.svc["ports"]
        self.assertTrue(any(":46580" in str(p) for p in ports))

    def test_port_published_on_loopback_only(self) -> None:
        # `sky api start --deploy` binds 0.0.0.0 inside the container and
        # ships no authentication: POST /api/v1/launch is anonymous and
        # spends real cloud money. Loopback bind is the only thing
        # keeping it off the LAN.
        for p in self.svc["ports"]:
            with self.subTest(port=p):
                self.assertTrue(
                    str(p).startswith("127.0.0.1:"),
                    msg=f"skypilot-api-server port not loopback-bound: {p}",
                )

    def test_cluster_profile(self) -> None:
        self.assertIn("cluster", self.svc.get("profiles") or [])

    def test_named_state_volume(self) -> None:
        vols = self.svc["volumes"]
        skypilot_state = [v for v in vols if v.startswith("skypilot-state:")]
        self.assertEqual(len(skypilot_state), 1)

    def test_scoped_cred_mounts_present(self) -> None:
        # Cloud creds still reach the container, but via narrowly-scoped
        # paths rather than the whole host $HOME.
        vols = self.svc["volumes"]
        for target in ("/root/.aws", "/root/.config/gcloud", "/root/.config/sky"):
            with self.subTest(target=target):
                self.assertTrue(
                    any(f":{target}:" in v for v in vols),
                    msg=f"no cred mount for {target} (volumes: {vols})",
                )

    def test_cred_mounts_are_read_only(self) -> None:
        vols = self.svc["volumes"]
        for v in vols:
            if v.startswith("${HOME}"):
                with self.subTest(volume=v):
                    self.assertTrue(
                        v.endswith(":ro"),
                        msg=f"host-$HOME-derived mount is not read-only: {v}",
                    )

    def test_no_whole_home_mount(self) -> None:
        # A `${HOME}:/root:rw` bind also exposed ~/.ssh and
        # ~/.config/sops/age/keys.txt -- the age key that decrypts every
        # deploy/*.sops.env.
        vols = self.svc["volumes"]
        self.assertNotIn(
            "${HOME}:/root:rw", vols,
            msg="whole-$HOME bind is back; it leaks ~/.ssh and the age key",
        )
        for v in vols:
            with self.subTest(volume=v):
                self.assertFalse(
                    v.startswith("${HOME}:"),
                    msg=f"whole-$HOME bind is back: {v}",
                )

    def test_ssh_and_sops_never_mounted(self) -> None:
        # Hard constraint: docs/secrets.md's recovery posture depends on
        # the age private key never entering a container.
        forbidden = (".ssh", "/sops", "/age")
        for v in self.svc["volumes"]:
            for seg in forbidden:
                with self.subTest(volume=v, segment=seg):
                    self.assertNotIn(
                        seg, v,
                        msg=f"forbidden path segment {seg!r} in mount {v!r}",
                    )

    def test_secrets_mount_with_dev_null_default(self) -> None:
        vols = self.svc["volumes"]
        secrets = [v for v in vols if "/secrets/.env" in v]
        self.assertEqual(len(secrets), 1)
        # Defaults to /dev/null so a Phase 1 install without
        # rendered creds boots cleanly.
        self.assertIn("SKYPILOT_CREDENTIALS_FILE", secrets[0])

    def test_command_runs_sky_api_start(self) -> None:
        cmd = self.svc["command"]
        joined = " ".join(cmd)
        self.assertIn("sky", joined)
        self.assertIn("api", joined)
        self.assertIn("start", joined)

    def test_skypilot_state_volume_declared(self) -> None:
        vols_top = self.compose.get("volumes") or {}
        self.assertIn("skypilot-state", vols_top)


class TestSkypilotApiEnv(unittest.TestCase):
    def test_documents_port(self) -> None:
        path = REPO_ROOT / "deploy" / "skypilot-api.env"
        self.assertTrue(path.is_file())
        text = path.read_text()
        self.assertIn("SKYPILOT_API_PORT", text)
        self.assertIn("46580", text)


class TestSkypilotCredentialsExample(unittest.TestCase):
    def test_present(self) -> None:
        path = REPO_ROOT / "deploy" / "skypilot-credentials.sops.env.example"
        self.assertTrue(path.is_file())

    def test_documents_all_secrets(self) -> None:
        text = (REPO_ROOT / "deploy" / "skypilot-credentials.sops.env.example").read_text()
        for name in EXPECTED_SECRETS:
            self.assertRegex(
                text, rf"(?m)^{name}=", msg=f"example missing {name}"
            )


class TestSkypilotHealthScript(unittest.TestCase):
    def test_exists_executable_valid(self) -> None:
        path = REPO_ROOT / "scripts" / "skypilot-api-health.sh"
        self.assertTrue(path.is_file())
        self.assertTrue(path.stat().st_mode & 0o100, msg="not executable")
        r = subprocess.run(["bash", "-n", str(path)],
                           capture_output=True, text=True, check=False)
        self.assertEqual(r.returncode, 0, msg=r.stderr)

    def test_probes_version_endpoint(self) -> None:
        text = (REPO_ROOT / "scripts" / "skypilot-api-health.sh").read_text()
        self.assertIn("/api/v1/version", text)


class TestMakefileSkypilotTargets(unittest.TestCase):
    def setUp(self) -> None:
        self.text = (REPO_ROOT / "Makefile").read_text()

    def test_targets_present(self) -> None:
        for target in (
            "skypilot-up:",
            "skypilot-down:",
            "skypilot-check:",
            "skypilot-secrets-render:",
        ):
            self.assertRegex(
                self.text, rf"(?m)^{re.escape(target)}",
                msg=f"missing target: {target}",
            )

    def test_secrets_render_depends_on_secrets_tmpfs(self) -> None:
        self.assertRegex(
            self.text,
            r"(?m)^skypilot-secrets-render:.*secrets-tmpfs",
        )

    def test_targets_in_phony(self) -> None:
        phony_lines = [
            line for line in self.text.splitlines()
            if line.startswith(".PHONY:")
        ]
        joined = " ".join(phony_lines)
        for target in (
            "skypilot-up", "skypilot-down",
            "skypilot-check", "skypilot-secrets-render",
        ):
            self.assertIn(
                target, joined, msg=f"{target} not in any .PHONY",
            )


class TestDocsSkypilotMd(unittest.TestCase):
    def test_present(self) -> None:
        self.assertTrue((REPO_ROOT / "docs" / "skypilot.md").is_file())

    def test_covers_required_sections(self) -> None:
        text = (REPO_ROOT / "docs" / "skypilot.md").read_text()
        for header in (
            "Status snapshot",
            "Bring-up",
            "Volume layout",
            "Endpoints",
            "Phase 2 preview",
            "Cost guidance",
            "Troubleshooting",
        ):
            self.assertIn(header, text, msg=f"docs/skypilot.md missing: {header}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
