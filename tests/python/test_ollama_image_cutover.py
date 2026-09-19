"""devai-ollama runs the host-built image everywhere, not just at first start.

The Ollama container is created in TWO places: by compose at `make cache-up`,
and by the router, which RECREATES it (containerRecreate) whenever a pinned
KV-cache tier is requested. They must name the same image. If only compose
is switched, the stack comes up on the new image and the first mixed-KV
request silently puts it back on the stock ollama/ollama image -- an older
Ollama, reverted with no error anywhere.

The router only ever sees what compose hands it, so the image has to be
forwarded explicitly (the same gap DEVAI_GPU_DEVICE had until 2026-07-27:
see docs/gpu-vendors.md). The existing "every knob the router reads is
forwarded" test polices DEVAI_* names only, which is how OLLAMA_IMAGE went
unchecked.

Stdlib unittest only; reads files, starts nothing.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COMPOSE = (REPO_ROOT / "deploy" / "docker-compose.yaml").read_text()
ROUTER = (REPO_ROOT / "gpu-arbiter" / "main.go").read_text()
MAKEFILE = (REPO_ROOT / "Makefile").read_text()

IMAGE = "localhost/devai-ollama:latest"
STOCK = "ollama/ollama:latest"


def _service(name: str) -> str:
    """The compose block for one service KEY (`ollama`, `router` -- not the
    container_name) up to the next 2-space key."""
    m = re.search(rf"^  {re.escape(name)}:\n(.*?)(?=^  \S|\Z)", COMPOSE,
                  re.M | re.S)
    assert m, f"service {name} not found in compose"
    return m.group(1)


class OllamaImageCutoverTest(unittest.TestCase):
    def test_compose_runs_the_host_built_image(self) -> None:
        self.assertIn(f"image: ${{OLLAMA_IMAGE:-{IMAGE}}}",
                      _service("ollama"))

    def test_compose_forwards_the_image_to_the_router(self) -> None:
        # Without this the router recreates devai-ollama from ITS default.
        self.assertIn(f"- OLLAMA_IMAGE=${{OLLAMA_IMAGE:-{IMAGE}}}",
                      _service("router"))

    def test_router_default_is_the_same_image(self) -> None:
        # Defence in depth: a router started without the variable must not
        # fall back to a different image than compose uses.
        m = re.search(r'env\("OLLAMA_IMAGE",\s*"([^"]+)"\)', ROUTER)
        self.assertIsNotNone(m, "router no longer reads OLLAMA_IMAGE")
        self.assertEqual(m.group(1), IMAGE)

    def test_makefile_exports_the_image_like_the_other_backends(self) -> None:
        # So a .env override reaches compose, exactly as VLLM_IMAGE does.
        self.assertRegex(MAKEFILE, rf"(?m)^OLLAMA_IMAGE \?= {re.escape(IMAGE)}$")
        self.assertRegex(MAKEFILE, r"(?m)^export OLLAMA_IMAGE$")

    def test_the_stock_image_is_named_nowhere_that_launches_a_container(self) -> None:
        self.assertNotIn(STOCK, _service("ollama"))
        self.assertNotIn(STOCK, _service("router"))
        self.assertNotRegex(ROUTER, rf'env\("OLLAMA_IMAGE",\s*"[^"]*{re.escape(STOCK)}"')


if __name__ == "__main__":
    unittest.main()
