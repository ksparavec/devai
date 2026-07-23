"""Tests for MCP Gateway Phase 1 deliverables.

Covers:
  - deploy/mcp-servers.yaml is well-formed and lists the 10 Tier 1
    servers (no Tier 2 entries enabled).
  - deploy/docker-compose.yaml gains the devai-mcp-gateway service
    with the right image, ports, mounts, and security flags.
  - deploy/mcp-gateway.env documents MCP_PORT.
  - scripts/mcp-health.sh and tests/test-mcp.sh exist and are
    bash-syntax-valid.
  - docs/mcp.md covers the operator-facing surface.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# 10 Tier 1 servers, per docs/plans/mcp-gateway.md.
TIER1_SERVERS = {
    "filesystem",
    "git",
    "sqlite",
    "fetch",
    "memory",
    "time",
    "sequentialthinking",
    "duckduckgo",
    "arxiv",
    "wikipedia",
}

TIER2_SERVERS = {
    "github-official",
    "firecrawl",
    "hugging-face",
    "context7",
}


def _yaml() -> object:
    try:
        import yaml
        return yaml
    except ImportError:
        return None


class TestMcpServersYaml(unittest.TestCase):
    def setUp(self) -> None:
        self.yaml = _yaml()
        if self.yaml is None:
            self.skipTest("PyYAML not installed")
        self.path = REPO_ROOT / "deploy" / "mcp-servers.yaml"

    def test_file_exists(self) -> None:
        self.assertTrue(self.path.is_file())

    def test_well_formed_yaml(self) -> None:
        with open(self.path) as f:
            doc = self.yaml.safe_load(f)
        self.assertIsInstance(doc, dict)
        self.assertEqual(doc.get("schemaVersion"), 1)
        self.assertIsInstance(doc.get("servers"), list)

    def test_all_tier1_present(self) -> None:
        with open(self.path) as f:
            doc = self.yaml.safe_load(f)
        names = {s.get("name") for s in doc["servers"]}
        missing = TIER1_SERVERS - names
        self.assertEqual(
            missing, set(), msg=f"Tier 1 servers missing: {missing}"
        )

    def test_tier1_present(self) -> None:
        # Phase 1 invariant: every Tier 1 server stays catalog-listed
        # even after Phase 2 added Tier 2 entries above.
        with open(self.path) as f:
            doc = self.yaml.safe_load(f)
        names = {s.get("name") for s in doc["servers"]}
        # All 10 Tier 1 still there.
        missing_t1 = TIER1_SERVERS - names
        self.assertEqual(
            missing_t1, set(),
            msg=f"Tier 1 servers regressed after Phase 2 added Tier 2: {missing_t1}",
        )

    def test_every_server_has_image_and_description(self) -> None:
        with open(self.path) as f:
            doc = self.yaml.safe_load(f)
        for s in doc["servers"]:
            with self.subTest(server=s.get("name")):
                self.assertIsInstance(s.get("image"), str)
                # localhost/ covers first-party servers built locally rather
                # than pulled from a registry (e.g. devai-model-status --
                # see docs/mcp-model-status.md).
                self.assertTrue(s["image"].startswith(("docker.io/", "ghcr.io/", "mcp/", "localhost/")))
                self.assertIsInstance(s.get("description"), str)
                self.assertGreater(len(s["description"]), 10)


class TestDockerComposeMcpService(unittest.TestCase):
    def setUp(self) -> None:
        self.yaml = _yaml()
        if self.yaml is None:
            self.skipTest("PyYAML not installed")
        with open(REPO_ROOT / "deploy" / "docker-compose.yaml") as f:
            self.compose = self.yaml.safe_load(f)
        self.svc = (self.compose.get("services") or {}).get("mcp-gateway")

    def test_service_present(self) -> None:
        self.assertIsNotNone(self.svc, "mcp-gateway service missing")

    def test_image_pinned_not_latest(self) -> None:
        img = self.svc["image"]
        self.assertNotIn(":latest", img)
        # Must be a versioned tag, not a moving alias.
        self.assertRegex(img, r":v?\d+\.\d+\.\d+")
        self.assertIn("mcp-gateway", img)

    def test_container_name_matches_devai_convention(self) -> None:
        self.assertEqual(self.svc["container_name"], "devai-mcp-gateway")

    def test_port_publish(self) -> None:
        ports = self.svc["ports"]
        # Default MCP_PORT=8088 mapped to container 8088.
        self.assertTrue(any(":8088" in str(p) for p in ports))

    def test_port_published_on_loopback_only(self) -> None:
        # The gateway holds a read-write podman socket, so an
        # unauthenticated MCP call is equivalent to host-root
        # container-create. Every published port must be bound to
        # 127.0.0.1, never 0.0.0.0.
        for p in self.svc["ports"]:
            with self.subTest(port=p):
                self.assertTrue(
                    str(p).startswith("127.0.0.1:"),
                    msg=f"mcp-gateway port not loopback-bound: {p}",
                )

    def test_podman_socket_mounted(self) -> None:
        # Documents the reason the loopback bind above is the mitigation:
        # the socket itself cannot be made :ro -- the gateway must create
        # containers through it.
        vols = self.svc["volumes"]
        sock = [v for v in vols if "docker.sock" in v]
        self.assertEqual(len(sock), 1)
        self.assertFalse(
            sock[0].endswith(":ro"),
            msg="gateway needs a rw socket; if this ever becomes ro, "
                "revisit the loopback-bind rationale",
        )

    def test_catalog_mount_read_only(self) -> None:
        vols = self.svc["volumes"]
        catalog = [v for v in vols if "catalog.yaml" in v]
        self.assertEqual(len(catalog), 1)
        self.assertTrue(catalog[0].endswith(":ro"))

    def test_no_new_privileges(self) -> None:
        sec = self.svc.get("security_opt") or []
        self.assertIn("no-new-privileges:true", sec)

    def test_block_secrets_flag(self) -> None:
        cmd = self.svc.get("command") or []
        self.assertIn("--block-secrets", cmd)

    def test_mcp_profile_set(self) -> None:
        # Decision 5 says start by default, but the compose profile
        # gate keeps it opt-in until we wire it into 'make cache-up'.
        # The 'mcp' profile is explicit; 'make mcp-up' uses it.
        profiles = self.svc.get("profiles") or []
        self.assertIn("mcp", profiles)


class TestMcpGatewayEnv(unittest.TestCase):
    def test_documents_mcp_port(self) -> None:
        path = REPO_ROOT / "deploy" / "mcp-gateway.env"
        self.assertTrue(path.is_file())
        text = path.read_text()
        self.assertIn("MCP_PORT", text)
        self.assertIn("8088", text)


class TestScriptsBashSyntax(unittest.TestCase):
    def test_mcp_health_parses(self) -> None:
        path = REPO_ROOT / "scripts" / "mcp-health.sh"
        self.assertTrue(path.is_file())
        r = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(r.returncode, 0, msg=r.stderr)

    def test_test_mcp_parses(self) -> None:
        path = REPO_ROOT / "tests" / "test-mcp.sh"
        self.assertTrue(path.is_file())
        r = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(r.returncode, 0, msg=r.stderr)


class TestDocsMcp(unittest.TestCase):
    def test_present_and_covers_required_sections(self) -> None:
        path = REPO_ROOT / "docs" / "mcp.md"
        self.assertTrue(path.is_file())
        text = path.read_text()
        for header in (
            "Bring-up",
            "Tier 1 server catalog",
            "Client configurations",
            "Security model",
            "Phase 2",
            "Troubleshooting",
        ):
            self.assertIn(header, text, msg=f"docs/mcp.md missing: {header}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
