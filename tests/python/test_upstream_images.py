"""Every image devai runs or builds on is built here (platform-rebuild Phase 2).

The only image pulled from outside is Debian, pinned by digest in the
Makefile's DEBIAN_IMAGE. This guard reads every place an image is named --
deploy/docker-compose.yaml, every Dockerfile* under deploy/ and tests/
(subdirectories included), and the Makefile's image variables (`*_IMAGE`,
`*_IMAGE_TAG`, `BASE_IMAGES`) -- and allows only:

  - localhost/devai-*   an image built here
  - the DEBIAN_IMAGE pin, exactly (so a bump must update every Dockerfile)
  - scratch             in a FROM (the router's base)

Anything else fails with the file and line. `*_IMAGE_NAME` variables are not
image references but the tags builds write (-t devai-lab), and a
`$(VAR)` reference is checked where VAR is defined.

TRANSITIONAL names what later phases replace. Each phase removes its
entries; an entry that no longer matches anything fails too, so the list
cannot outlive what it excuses, and Phase 14 asserts it is empty. A phase
that adds an image adds it under the same rule.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

# key -> the phase that removes it. Keys: "Makefile <VAR>" for a Makefile
# variable (a compose `${VAR:-default}` counts as VAR), "<path> <ARG>" for a
# Dockerfile ARG default, "compose <service>" for a literal compose image.
TRANSITIONAL = {
    "Makefile VLLM_IMAGE": "Phase 4 (devai-engines)",
    "Makefile VLLM_DEVAI_IMAGE": "Phase 4 (devai-engines)",
    "Makefile VLLM_DEVAI_IMAGE_TAG": "Phase 4 (devai-engines)",
    "Makefile SGLANG_IMAGE": "Phase 4 (devai-engines)",
    "Makefile QUANT_IMAGE": "Phase 4 (devai-engines)",
    "compose logger": "Phase 6 (the logger moves into devai-operator)",
}

_MAKE_VAR = re.compile(r"^([A-Z][A-Z0-9_]*(?:_IMAGE|_IMAGE_TAG|_IMAGES))\s*[?:]?=\s*(.*?)\s*$")
_MAKE_REF = re.compile(r"^\$\([A-Z][A-Z0-9_]*\)$")
_COMPOSE_SERVICE = re.compile(r"^  ([a-z][a-z0-9-]*):\s*$")
_COMPOSE_IMAGE = re.compile(r"^\s+image:\s*(\S+)\s*$")
_COMPOSE_VAR = re.compile(r"^\$\{([A-Z][A-Z0-9_]*):-(.+)\}$")
_DF_FROM = re.compile(r"^FROM\s+(?:--platform=\S+\s+)?(\S+)(?:\s+AS\s+(\S+))?\s*$", re.I)
_DF_ARG = re.compile(r"^ARG\s+([A-Z][A-Z0-9_]*IMAGE[A-Z0-9_]*)=(\S+)\s*$")
# COPY --from=<x> and RUN --mount=...,from=<x>: an image when <x> is not a stage.
_DF_FROM_OPT = re.compile(r"(?:--from=|[,\s]from=)([^\s,]+)")


def debian_pin() -> str:
    mk = (REPO / "Makefile").read_text()
    return re.search(r"^DEBIAN_IMAGE = (\S+)$", mk, re.M).group(1)


def dockerfiles() -> list[Path]:
    found = []
    for top in ("deploy", "tests"):
        found += [p for p in (REPO / top).rglob("Dockerfile*") if p.is_file()]
    return sorted(found)


def references() -> list[tuple[str, str, str]]:
    """(where, key, image) for every image named in the three sources."""
    refs = []
    for n, line in enumerate((REPO / "Makefile").read_text().splitlines(), 1):
        m = _MAKE_VAR.match(line)
        if not m or m.group(2).startswith("$(shell"):
            continue
        for token in m.group(2).split():
            if not _MAKE_REF.match(token):
                refs.append((f"Makefile:{n}", f"Makefile {m.group(1)}", token))

    service = None
    compose = REPO / "deploy" / "docker-compose.yaml"
    for n, line in enumerate(compose.read_text().splitlines(), 1):
        if m := _COMPOSE_SERVICE.match(line):
            service = m.group(1)
        if m := _COMPOSE_IMAGE.match(line):
            image = m.group(1).strip("\"'")
            if v := _COMPOSE_VAR.match(image):
                refs.append((f"deploy/docker-compose.yaml:{n}", f"Makefile {v.group(1)}", v.group(2)))
            else:
                refs.append((f"deploy/docker-compose.yaml:{n}", f"compose {service}", image))

    for df in dockerfiles():
        rel = df.relative_to(REPO).as_posix()
        stages = set()
        for n, line in enumerate(df.read_text().splitlines(), 1):
            if m := _DF_ARG.match(line.strip()):
                refs.append((f"{rel}:{n}", f"{rel} {m.group(1)}", m.group(2)))
            elif m := _DF_FROM.match(line.strip()):
                image, stage = m.group(1), m.group(2)
                if not (image.startswith("${") or image.startswith("$") or image.lower() in stages):
                    refs.append((f"{rel}:{n}", f"{rel} FROM", image))
                if stage:
                    stages.add(stage.lower())
            elif line.strip().upper().startswith(("COPY", "RUN", "ADD")):
                for image in _DF_FROM_OPT.findall(line):
                    if not (image.isdigit() or image.startswith("$") or image.lower() in stages):
                        refs.append((f"{rel}:{n}", f"{rel} --from", image))
    return refs


def allowed(image: str, pin: str, key: str) -> bool:
    if image.startswith("localhost/devai-") or image == pin:
        return True
    return image == "scratch" and key.endswith(" FROM")


class UpstreamImagesTest(unittest.TestCase):
    def test_only_images_built_here_and_the_debian_pin(self):
        pin = debian_pin()
        bad = [f"{where}: {image}" for where, key, image in references()
               if not allowed(image, pin, key) and key not in TRANSITIONAL]
        self.assertEqual(bad, [], "images not built here (allowed: localhost/devai-*, "
                         f"{pin}, scratch; or a TRANSITIONAL entry)")

    def test_transitional_entries_are_still_used(self):
        pin = debian_pin()
        used = {key for _, key, image in references() if not allowed(image, pin, key)}
        stale = sorted(set(TRANSITIONAL) - used)
        self.assertEqual(stale, [], "TRANSITIONAL entries whose image is gone: remove them")

    def test_debian_pin_is_a_digest(self):
        self.assertRegex(debian_pin(), r"^docker\.io/library/debian:trixie-slim@sha256:[0-9a-f]{64}$")

    def test_every_debian_build_gets_the_pin(self):
        mk = (REPO / "Makefile").read_text()
        for df in ("Dockerfile.base", "Dockerfile.ollama", "Dockerfile.vllm",
                   "Dockerfile.apt-cache", "Dockerfile.pipelock"):
            target = re.search(rf"(?ms)^(build-[a-z-]+):[^\n]*\n(?:\t[^\n]*\n)*?\t\t-f deploy/{re.escape(df)} ", mk)
            self.assertIsNotNone(target, df)
            self.assertIn("--build-arg BASE_IMAGE=$(DEBIAN_IMAGE)", target.group(0), target.group(1))

    def test_pull_images_pulls_only_debian_as_a_base(self):
        # The CUDA base left with Phase 3; compose's transitional images
        # come in through CACHE_IMAGES.
        mk = (REPO / "Makefile").read_text()
        self.assertRegex(mk, r"(?m)^BASE_IMAGES = \$\(DEBIAN_IMAGE\)$")
        self.assertNotIn("GPU_BASE_IMAGE", mk)

    def test_the_guard_sees_every_source(self):
        # A parser that silently matches nothing would pass everything.
        keys = {key for _, key, _ in references()}
        for key in ("Makefile DEBIAN_IMAGE", "compose apt-cache", "compose pipelock",
                    "compose router", "compose logger", "Makefile VLLM_IMAGE",
                    "deploy/Dockerfile.router FROM", "deploy/slurm/Dockerfile BASE_IMAGE",
                    "tests/slurm/Dockerfile.control-test BASE_IMAGE"):
            self.assertIn(key, keys)

    def test_from_options_are_seen(self):
        # COPY --from=<image> pulls an image as surely as FROM does.
        line = "COPY --from=docker.io/library/golang:1 /x /x"
        self.assertEqual(_DF_FROM_OPT.findall(line), ["docker.io/library/golang:1"])
        self.assertEqual(_DF_FROM_OPT.findall("RUN --mount=type=bind,from=ghcr.io/x/y,target=/y true"),
                         ["ghcr.io/x/y"])

    def test_apt_cache_is_not_on_the_lab_network(self):
        # apt-cacher-ng proxies any host (CONNECT and plain http); on
        # devai-lab-egress it was a way out of the locked lab around pipelock.
        compose = (REPO / "deploy" / "docker-compose.yaml").read_text()
        block = re.search(r"(?ms)^  apt-cache:\n(.*?)(?=^  [a-z])", compose).group(1)
        self.assertRegex(block, r"(?m)^    networks: \[default\]$")
        mk = (REPO / "Makefile").read_text()
        self.assertNotIn("apt-cache", re.search(r"(?m)^PIPELOCK_NO_PROXY = (.*)$", mk).group(1))
        agent = (REPO / "bin" / "devai-agent").read_text()
        self.assertNotIn("apt-cache", re.search(r'(?m)^NO_PROXY_HOSTS = "(.*)"$', agent).group(1))

    def test_registry_mirror_is_gone(self):
        compose = (REPO / "deploy" / "docker-compose.yaml").read_text()
        self.assertNotIn("registry-cache", compose)
        self.assertFalse((REPO / "deploy" / "registry-config.yaml").exists())
        conf = (REPO / "deploy" / "registries.conf").read_text().splitlines()
        self.assertEqual([l for l in conf if "mirror" in l and not l.startswith("#")], [])
        mk = (REPO / "Makefile").read_text()
        for var in ("CACHE_SERVICES", "PIPELOCK_NO_PROXY"):
            line = re.search(rf"(?ms)^{var} = (.*?)(?<!\\)$", mk).group(1)
            self.assertNotIn("registry-cache", line, var)
        self.assertNotIn("registry-cache", (REPO / "bin" / "devai-agent").read_text())


if __name__ == "__main__":
    unittest.main()
