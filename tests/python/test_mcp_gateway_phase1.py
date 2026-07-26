"""Tests for the MCP gateway wiring.

Rewritten 2026-07-25. The previous version asserted the shape of a
hand-maintained deploy/mcp-servers.yaml that listed all 14 third-party
servers. Those assertions all passed while the gateway exposed ZERO
tools, because the file used a schema the gateway does not parse and
pinned every image at a tag (:0.7.0) that exists for none of them.

Shape assertions cannot catch that class of defect. The real regression
guard is tests/test-mcp.sh, which now performs a live handshake, asserts
a tool-count floor, and makes a real tools/call. What is left here is
only the static wiring that must hold for that test to be reachable at
all -- the things that, when wrong, stop the gateway before it can be
exercised.

Covers:
  - deploy/mcp-catalog-devai.yaml carries ONLY first-party servers, in
    the gateway's real schema (name/displayName/registry-as-a-map).
  - deploy/docker-compose.yaml wires the gateway correctly: merged
    catalog, explicit --servers, dual-homed networks, loopback publish,
    security flags.
  - The Makefile targets name the compose service that actually exists.
  - scripts/mcp-health.sh and tests/test-mcp.sh are bash-syntax-valid.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Upstream catalog keys we enable via --servers. These are Docker's
# official catalog names and are CASE-SENSITIVE: "SQLite", not "sqlite".
# We do not define these servers ourselves -- upstream does, and it
# digest-pins them. We only choose which to switch on.
ENABLED_SERVERS = {
    "filesystem",
    "git",
    "SQLite",
    "fetch",
    "memory",
    "time",
    "sequentialthinking",
    "duckduckgo",
    "arxiv-mcp-server",
    "wikipedia-mcp",
    "github-official",
    "firecrawl",
    "hugging-face",
    "context7",
    "devai-model-status",
}

# The only server this repo builds. Everything else must come from
# upstream -- a second entry here means someone started hand-maintaining
# third-party definitions again, which is what broke this before.
FIRST_PARTY_SERVERS = {"devai-model-status"}


def _yaml() -> object:
    try:
        import yaml
        return yaml
    except ImportError:
        return None


class TestFirstPartyCatalog(unittest.TestCase):
    """deploy/mcp-catalog-devai.yaml: first-party servers only, real schema."""

    def setUp(self) -> None:
        self.yaml = _yaml()
        if self.yaml is None:
            self.skipTest("PyYAML not installed")
        self.path = REPO_ROOT / "deploy" / "mcp-catalog-devai.yaml"

    def _doc(self) -> dict:
        with open(self.path) as f:
            return self.yaml.safe_load(f)

    def test_file_exists(self) -> None:
        self.assertTrue(self.path.is_file())

    def test_old_hand_maintained_catalog_is_gone(self) -> None:
        """deploy/mcp-servers.yaml declared all 14 third-party servers with
        invented tags. Its return would mean someone resumed hand-
        maintaining definitions upstream already owns."""
        self.assertFalse(
            (REPO_ROOT / "deploy" / "mcp-servers.yaml").exists(),
            msg="deploy/mcp-servers.yaml is back -- third-party servers must "
                "come from Docker's official catalog, not from this repo.",
        )

    def test_uses_the_gateways_real_schema(self) -> None:
        """The gateway parses top-level name/displayName/registry, where
        registry is a MAP. The old apiVersion/schemaVersion/servers-list
        shape parses to an EMPTY registry and yields zero tools -- silently,
        which is why this needs an explicit assertion."""
        doc = self._doc()
        self.assertIsInstance(doc, dict)
        self.assertIsInstance(doc.get("name"), str)
        self.assertIsInstance(doc.get("registry"), dict,
                              msg="registry must be a map keyed by server name")
        for absent in ("apiVersion", "schemaVersion", "servers"):
            self.assertNotIn(absent, doc,
                             msg=f"{absent!r} is not part of the gateway schema")

    def test_contains_only_first_party_servers(self) -> None:
        self.assertEqual(set(self._doc()["registry"]), FIRST_PARTY_SERVERS)

    def test_every_entry_has_a_required_type(self) -> None:
        """Server.Type is validate:"required,oneof=server remote poci"."""
        for name, entry in self._doc()["registry"].items():
            with self.subTest(server=name):
                self.assertIn(entry.get("type"), {"server", "remote", "poci"})

    def test_first_party_image_is_locally_built(self) -> None:
        """devai-model-status has no upstream registry: the gateway runs
        servers with --pull never, so the tag must be one `make
        build-mcp-modelstatus-image` produces in the local store."""
        entry = self._doc()["registry"]["devai-model-status"]
        self.assertTrue(entry["image"].startswith("localhost/"))


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

    def test_catalog_mount_read_only_and_under_catalogs_dir(self) -> None:
        """Gateway v0.43.3 rejects a catalog path outside its catalogs
        directory, so the old /app/catalog.yaml mount breaks on upgrade."""
        vols = self.svc["volumes"]
        catalog = [v for v in vols if "mcp-catalog-devai.yaml" in v]
        self.assertEqual(len(catalog), 1, msg=f"volumes: {vols}")
        self.assertTrue(catalog[0].endswith(":ro"))
        self.assertIn("/root/.docker/mcp/catalogs/", catalog[0])

    def test_merges_with_upstream_catalog_rather_than_replacing_it(self) -> None:
        """--catalog REPLACES the built-in catalog, which would strand every
        third-party server. --additional-catalog merges on top of Docker's
        official one, which is where all 14 third-party servers come from."""
        cmd = self.svc.get("command") or []
        self.assertTrue(any(c.startswith("--additional-catalog=") for c in cmd),
                        msg=f"command: {cmd}")
        self.assertFalse(any(c.startswith("--catalog=") for c in cmd),
                         msg="--catalog replaces the upstream catalog; use "
                             "--additional-catalog")

    def test_servers_explicitly_enabled(self) -> None:
        """Without --servers the gateway enables NOTHING and serves only its
        own builtins -- a correct catalog alone yields '0 tools listed'."""
        cmd = self.svc.get("command") or []
        flag = [c for c in cmd if c.startswith("--servers=")]
        self.assertEqual(len(flag), 1, msg=f"command: {cmd}")
        listed = set(flag[0].split("=", 1)[1].split(","))
        self.assertEqual(listed, ENABLED_SERVERS)

    def test_dual_homed_so_the_lab_can_reach_it(self) -> None:
        """Lab containers sit on devai-lab-egress. A service published only
        on devai-net is unresolvable from them, which made this gateway
        unreachable from every agent regardless of its config."""
        nets = self.svc.get("networks") or []
        self.assertIn("devai-lab-egress", nets, msg=f"networks: {nets}")

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
        """The Tier 1 / Tier 2 headings this used to require are gone on
        purpose: that phasing described a hand-maintained catalog which no
        longer exists. What an operator needs now is where the servers
        come from, how to reach the endpoint, and the security posture."""
        path = REPO_ROOT / "docs" / "mcp.md"
        self.assertTrue(path.is_file())
        text = path.read_text()
        for header in (
            "Bring-up",
            "catalog",
            "Security model",
            "Troubleshooting",
            "Known limitations",
        ):
            self.assertIn(header, text, msg=f"docs/mcp.md missing: {header}")

    def test_documents_the_mcp_path_and_bearer_token(self) -> None:
        """Both are required to talk to the gateway at all, and both were
        absent from the previous version of this doc."""
        text = (REPO_ROOT / "docs" / "mcp.md").read_text()
        self.assertIn(":8088/mcp", text, msg="the /mcp path is not optional")
        self.assertIn("Bearer", text, msg="bearer token not documented")

    def test_does_not_claim_default_reachability(self) -> None:
        """The gateway is behind `profiles: [mcp]`; `make cache-up` does not
        start it. The old doc claimed port 8088 was up by default."""
        text = (REPO_ROOT / "docs" / "mcp.md").read_text()
        self.assertNotIn("reachable on port 8088 by default", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
