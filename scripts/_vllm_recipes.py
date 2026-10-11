"""Shared helpers for vLLM's published launch recipes (plan Phase 17a).

vllm-project/recipes publishes one YAML per model (`models/<org>/<repo>.yaml`)
with the vendor's launch command: base args and env, tool-calling and
reasoning parser args, per-precision variants with a VRAM minimum and extra
flags, and the oldest vLLM the recipe works on (`min_vllm_version`).

  scripts/fetch-vllm-recipes.py  fetches and normalises them into
                                 deploy/recipe-seeds.yaml (one entry per
                                 catalog row with a recipe)
  scripts/recipe-apply.py        writes one row's reviewed flags into
                                 deploy/recovery-flags.generated.json
  scripts/_probe_hf_common.py    reads a seed's parsers as the third step of
                                 the parser order (after curated and
                                 card-derived)
  scripts/model-picker.py        shows a recipe badge; it reads the YAML
                                 itself and does NOT import this module (it
                                 runs bind-mounted inside older images)

Standard library only, plus PyYAML where it is installed; every reader here
fails open (no seeds) rather than raising.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SEEDS_PATH = REPO_ROOT / "deploy" / "recipe-seeds.yaml"
GENERATED_FLAGS_PATH = REPO_ROOT / "deploy" / "recovery-flags.generated.json"
BUILD_VLLM_SH = REPO_ROOT / "scripts" / "build-vllm.sh"
SEEDS_VERSION = 1

# The backend that runs the vLLM build-vllm.sh produces. `vllm-devai` until
# platform-rebuild Phase 4 step 4e renames it to `vllm` (4e also rewrites the
# groups recipe-apply wrote before it).
SERVICE_BACKEND = "vllm-devai"

_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)(?:\.(\d+))?")

# The forms build-vllm.sh has used or will use for its version:
#   VLLM_VERSION="${VLLM_VERSION:-0.28.0}"   (today: a default)
#   : "${VLLM_VERSION:=0.30.0}"              (an assign-default form)
#   VLLM_VERSION=0.28.0 / VLLM_VERSION="0.28.0"   (a plain constant)
_BUILD_VERSION_RES = (
    re.compile(r'^\s*(?:export\s+)?VLLM_VERSION="?\$\{VLLM_VERSION:-([0-9][0-9A-Za-z.+-]*)\}"?', re.M),
    re.compile(r'^\s*:\s*"?\$\{VLLM_VERSION:=([0-9][0-9A-Za-z.+-]*)\}"?', re.M),
    re.compile(r'^\s*(?:export\s+)?VLLM_VERSION="?([0-9][0-9A-Za-z.+-]*)"?\s*(?:#.*)?$', re.M),
)


def parse_version(value: object) -> tuple[int, int, int] | None:
    """`0.28.0` -> (0, 28, 0); `0.17` -> (0, 17, 0); `nightly` / junk -> None.

    A pre-release suffix (`0.31.0rc1`) is read as its release; that errs
    towards admitting a recipe one release early, which the probe -- the
    final authority -- then confirms or refutes.
    """
    if not isinstance(value, str):
        return None
    m = _VERSION_RE.match(value.strip())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))


def version_admits(min_version: object, version: object) -> bool:
    """True when `version` is at least `min_version`. An unparsable value on
    either side (`nightly`, missing) admits nothing: the gate exists so a
    recipe never injects a flag the vLLM in service rejects."""
    lo, have = parse_version(min_version), parse_version(version)
    return lo is not None and have is not None and have >= lo


def service_vllm_version(build_script: Path = BUILD_VLLM_SH) -> str | None:
    """The vLLM version scripts/build-vllm.sh builds by default, or None."""
    try:
        text = build_script.read_text()
    except OSError:
        return None
    for pattern in _BUILD_VERSION_RES:
        m = pattern.search(text)
        if m:
            return m.group(1)
    return None


def image_vllm_version(image: str | None) -> str | None:
    """`docker.io/vllm/vllm-openai:v0.22.1-x86_64-...` -> `0.22.1`; None when
    the tag names no version (`:latest`)."""
    if not image or ":" not in image.rsplit("/", 1)[-1]:
        return None
    tag = image.rsplit(":", 1)[1]
    m = re.match(r"^v?(\d+\.\d+\.\d+)", tag)
    return m.group(1) if m else None


def engine_vllm_version(backend: str, image: str | None = None,
                        *, pinned: bool = False) -> str | None:
    """The vLLM version a launch runs, as far as it can be known here: the
    image tag's, else, for the backend that runs build-vllm.sh's build, that
    script's default (its image is tagged `latest`). A `pinned` image (a
    curated recovery `image`) is some other build: its tag alone counts.
    None when unknown (the caller then seeds nothing)."""
    tagged = image_vllm_version(image)
    if tagged or pinned or backend != SERVICE_BACKEND:
        return tagged
    return service_vllm_version()


COMPOSE_FILE = REPO_ROOT / "deploy" / "docker-compose.yaml"
CURATED_FLAGS_PATH = REPO_ROOT / "deploy" / "recovery-flags.json"
# The vLLM-engine backends a recipe can apply to.
VLLM_BACKENDS = ("vllm", "vllm-devai")


def stock_vllm_image(compose: Path | None = None) -> str | None:
    """The stock vLLM image in service: $VLLM_IMAGE (the Makefile exports
    it from .env), else the router's compose default."""
    image = os.environ.get("VLLM_IMAGE")
    if image:
        return image
    try:
        m = re.search(r"VLLM_IMAGE=\$\{VLLM_IMAGE:-([^}]+)\}",
                      (compose or COMPOSE_FILE).read_text())
    except OSError:
        return None
    return m.group(1) if m else None


def curated_image(model: str, backend: str, path: Path | None = None) -> str | None:
    """The curated recovery entry's `image` pin for `model` on `backend`
    (contract C2 scoping), or None."""
    try:
        entry = (json.loads((path or CURATED_FLAGS_PATH).read_text())
                 .get("models") or {}).get(model)
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(entry, dict) or not isinstance(entry.get("image"), str):
        return None
    backends = entry.get("backends")
    engine = "vllm" if backend in VLLM_BACKENDS else backend
    if isinstance(backends, list) and backend not in backends and engine not in backends:
        return None
    return entry["image"] or None


def backend_vllm_version(model: str, backend: str) -> str | None:
    """The vLLM a launch of `model` on `backend` runs, as far as it is known
    on this host: a curated `image` pin, else the backend's image
    ($VLLM_IMAGE / the compose default for `vllm`, $VLLM_DEVAI_IMAGE for
    `vllm-devai`), else build-vllm.sh's default for `vllm-devai`."""
    pin = curated_image(model, backend)
    if pin:
        return engine_vllm_version(backend, pin, pinned=True)
    image = stock_vllm_image() if backend == "vllm" else os.environ.get("VLLM_DEVAI_IMAGE")
    return engine_vllm_version(backend, image)


def load_seeds(path: Path = SEEDS_PATH) -> dict[str, dict]:
    """{catalog row name: seed} from deploy/recipe-seeds.yaml; {} when the
    file is missing, unreadable, not the expected shape, or PyYAML is not
    installed. Never raises."""
    try:
        import yaml
    except ImportError:
        return {}
    try:
        doc = yaml.safe_load(path.read_text())
    except (OSError, ValueError, yaml.YAMLError):
        return {}
    version = doc.get("version") if isinstance(doc, dict) else None
    if type(version) is not int or version != SEEDS_VERSION:
        return {}
    seeds = doc.get("seeds")
    if not isinstance(seeds, dict):
        return {}
    return {name: seed for name, seed in seeds.items() if isinstance(seed, dict)}


def seed_parsers(seed: dict | None, engine: str, version: str | None) -> dict[str, str]:
    """A seed's parsers ({"tool": ..., "reasoning": ...}, either may be
    absent) when they may be used on a launch of `engine` at `version`:
    vLLM only (a recipe names vLLM's parser plugins), and only when the
    recipe's min_vllm_version admits that version. {} otherwise."""
    if not seed or engine != "vllm":
        return {}
    if not version_admits(seed.get("min_vllm_version"), version):
        return {}
    parsers = seed.get("parsers") or {}
    if not isinstance(parsers, dict):
        return {}
    return {k: v for k, v in parsers.items()
            if k in ("tool", "reasoning") and isinstance(v, str) and v}
