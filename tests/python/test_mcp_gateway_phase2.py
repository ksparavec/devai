"""Tests for MCP Gateway Phase 2 deliverables.

Covers:
  - deploy/mcp-servers.yaml carries the 4 Tier 2 entries with
    {secret: NAME} references.
  - deploy/mcp-secrets.sops.env.example documents the 4 expected
    secret names (operators see what to encrypt).
  - docker-compose.yaml mounts /run/devai/mcp-secrets.env (or its
    /dev/null fallback) and emits --secrets to the gateway.
  - Makefile mcp-secrets-render target wires through the shared
    sops/age scaffold.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

TIER2_SERVERS = {"github-official", "firecrawl", "hugging-face", "context7"}

EXPECTED_SECRETS = {
    "GITHUB_TOKEN",
    "FIRECRAWL_API_KEY",
    "HF_TOKEN",
    "CONTEXT7_API_KEY",
}


def _yaml():
    try:
        import yaml
        return yaml
    except ImportError:
        return None


class TestMcpServersYamlTier2(unittest.TestCase):
    def setUp(self) -> None:
        self.yaml = _yaml()
        if self.yaml is None:
            self.skipTest("PyYAML not installed")
        with open(REPO_ROOT / "deploy" / "mcp-servers.yaml") as f:
            self.doc = self.yaml.safe_load(f)

    def test_tier2_servers_present(self) -> None:
        names = {s.get("name") for s in self.doc["servers"]}
        missing = TIER2_SERVERS - names
        self.assertEqual(missing, set(), msg=f"Tier 2 servers missing: {missing}")

    def test_each_tier2_references_a_secret(self) -> None:
        servers = {s["name"]: s for s in self.doc["servers"] if s.get("name") in TIER2_SERVERS}
        for name, srv in servers.items():
            with self.subTest(server=name):
                env = srv.get("environment") or []
                self.assertGreater(len(env), 0, msg=f"{name}: no env entries")
                # At least one env entry must reference {secret: ...}.
                self.assertTrue(
                    any("{secret:" in str(e) for e in env),
                    msg=f"{name}: no {{secret: NAME}} reference in env",
                )

    def test_secret_names_match_expected_set(self) -> None:
        seen = set()
        for srv in self.doc["servers"]:
            if srv.get("name") not in TIER2_SERVERS:
                continue
            for entry in srv.get("environment", []) or []:
                m = re.search(r"\{secret:\s*([A-Z0-9_]+)\}", str(entry))
                if m:
                    seen.add(m.group(1))
        self.assertEqual(
            seen, EXPECTED_SECRETS,
            msg=f"secret-name set drift: seen={seen}",
        )


class TestMcpSecretsExample(unittest.TestCase):
    def test_present(self) -> None:
        path = REPO_ROOT / "deploy" / "mcp-secrets.sops.env.example"
        self.assertTrue(path.is_file())

    def test_documents_all_four_names(self) -> None:
        text = (REPO_ROOT / "deploy" / "mcp-secrets.sops.env.example").read_text()
        for name in EXPECTED_SECRETS:
            self.assertRegex(
                text,
                rf"(?m)^{name}=",
                msg=f"example missing {name}",
            )


class TestComposeSecretsMount(unittest.TestCase):
    def setUp(self) -> None:
        self.yaml = _yaml()
        if self.yaml is None:
            self.skipTest("PyYAML not installed")
        with open(REPO_ROOT / "deploy" / "docker-compose.yaml") as f:
            self.compose = self.yaml.safe_load(f)
        self.svc = self.compose["services"]["mcp-gateway"]

    def test_secrets_mount_present(self) -> None:
        vols = self.svc["volumes"]
        secrets_vols = [v for v in vols if "/secrets/.env" in v]
        self.assertEqual(len(secrets_vols), 1)
        # MCP_SECRETS_FILE env var with /dev/null default keeps the
        # entry harmless when secrets aren't rendered.
        self.assertIn("MCP_SECRETS_FILE", secrets_vols[0])

    def test_secrets_command_flag_present(self) -> None:
        cmd = self.svc["command"]
        secrets_flag = [c for c in cmd if "--secrets" in c]
        self.assertEqual(len(secrets_flag), 1)
        # Must accept MCP_SECRETS_PATH override or default to /dev/null.
        self.assertIn("MCP_SECRETS_PATH", secrets_flag[0])


class TestMakefileMcpSecretsRender(unittest.TestCase):
    def setUp(self) -> None:
        self.text = (REPO_ROOT / "Makefile").read_text()

    def test_target_present(self) -> None:
        self.assertRegex(
            self.text,
            r"(?m)^mcp-secrets-render:.*secrets-tmpfs",
            msg="mcp-secrets-render target missing or doesn't depend on secrets-tmpfs",
        )

    def test_target_calls_render_secret(self) -> None:
        # Search inside the target body.
        m = re.search(
            r"(?m)^mcp-secrets-render:.*?\n\n",
            self.text,
            re.DOTALL,
        )
        self.assertIsNotNone(m)
        body = m.group(0)
        self.assertIn("scripts/render-secret.sh", body)
        self.assertIn("deploy/mcp-secrets.sops.env", body)
        self.assertIn("/run/devai/mcp-secrets.env", body)

    def test_target_in_phony_list(self) -> None:
        # .PHONY is split across many lines; just confirm the symbol
        # appears on a .PHONY line somewhere.
        phony_lines = [
            line for line in self.text.splitlines()
            if line.startswith(".PHONY:")
        ]
        joined = " ".join(phony_lines)
        self.assertIn(
            "mcp-secrets-render", joined,
            msg=f"mcp-secrets-render not in any .PHONY (lines: {phony_lines})",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
