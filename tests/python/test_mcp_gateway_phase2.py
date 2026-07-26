"""Tests for the MCP gateway's secret-bearing servers.

Rewritten 2026-07-25 alongside test_mcp_gateway_phase1. The Tier 2
server definitions this used to assert against are gone: we no longer
hand-maintain third-party server entries at all, so there is no local
`{secret: NAME}` syntax to check. Those servers now come from Docker's
official catalog, which declares its own secret names.

Two of the four former "Tier 2" servers turned out to need no secret at
all -- upstream converted hugging-face and context7 to `type: remote`
endpoints that connect anonymously (verified live: 4 tools and 2 tools
respectively, with no credentials).

What remains testable statically is the secret PLUMBING: the mount, the
--secrets flag, and the render target. The secret names themselves are
upstream's, and are recorded here so a drifting example file is caught.

NOTE: the end-to-end secrets path is UNVERIFIED. github-official and
firecrawl enumerate their tools without credentials, but no tool has
been invoked with a real token.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Servers that genuinely need a credential to DO anything. Both come
# from the upstream catalog; we only switch them on.
SECRET_BEARING_SERVERS = {"github-official", "firecrawl"}

# Servers that used to be listed as secret-bearing but are not: upstream
# serves both as anonymous remote endpoints now.
NO_LONGER_SECRET_BEARING = {"hugging-face", "context7"}

# Secret names as the UPSTREAM catalog declares them. The gateway looks
# secrets up by catalog secret name, so the old GITHUB_TOKEN-style keys
# would silently never resolve -- the same class of quiet failure that
# made the whole gateway look shipped while serving zero tools.
EXPECTED_SECRETS = {
    "github.personal_access_token",
    "firecrawl.api_key",
}


def _yaml():
    try:
        import yaml
        return yaml
    except ImportError:
        return None


class TestSecretBearingServersEnabled(unittest.TestCase):
    """The secret-bearing servers must still be in the --servers list; the
    secrets plumbing below is pointless if they are not enabled."""

    def setUp(self) -> None:
        self.yaml = _yaml()
        if self.yaml is None:
            self.skipTest("PyYAML not installed")
        with open(REPO_ROOT / "deploy" / "docker-compose.yaml") as f:
            compose = self.yaml.safe_load(f)
        svc = (compose.get("services") or {}).get("mcp-gateway") or {}
        flag = [c for c in (svc.get("command") or []) if c.startswith("--servers=")]
        self.enabled = set(flag[0].split("=", 1)[1].split(",")) if flag else set()

    def test_secret_bearing_servers_are_enabled(self) -> None:
        missing = SECRET_BEARING_SERVERS - self.enabled
        self.assertEqual(missing, set(), msg=f"not enabled: {missing}")

    def test_anonymous_remote_servers_are_enabled_too(self) -> None:
        """These need no secret, so they must work on an install that has
        never run mcp-secrets-render."""
        missing = NO_LONGER_SECRET_BEARING - self.enabled
        self.assertEqual(missing, set(), msg=f"not enabled: {missing}")


class TestMcpSecretsExample(unittest.TestCase):
    def test_present(self) -> None:
        path = REPO_ROOT / "deploy" / "mcp-secrets.sops.env.example"
        self.assertTrue(path.is_file())

    def test_documents_upstream_secret_names(self) -> None:
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
        # The flag must name the container-side path the secrets volume
        # binds to. An env-var indirection here (the old
        # `--secrets=${MCP_SECRETS_PATH:-/dev/null}`) silently kept the
        # gateway pointed at /dev/null for any operator who followed
        # docs/mcp.md and only set MCP_SECRETS_FILE.
        self.assertEqual(secrets_flag[0], "--secrets=/secrets/.env")

    def test_secrets_flag_target_matches_mount_target(self) -> None:
        # Guards the exact drift F5 was: flag path and mount path must
        # agree, whatever they are.
        secrets_flag = [c for c in self.svc["command"] if "--secrets" in c][0]
        flag_path = secrets_flag.split("=", 1)[1]
        # Source sides carry `${VAR:-default}`, so ':'-splitting is
        # unreliable; match the target as a ':'-delimited segment instead.
        vols = self.svc["volumes"]
        self.assertTrue(
            any(f":{flag_path}:" in v or v.endswith(f":{flag_path}") for v in vols),
            msg=f"--secrets={flag_path} names no bind-mount target "
                f"(volumes: {vols})",
        )


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
