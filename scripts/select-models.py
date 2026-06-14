#!/usr/bin/env python3
"""Diagnostic + downloader for deploy/models.yaml entries.

Reads the catalog from scripts/generate-catalog.py and the probe cache
from scripts/probe-ollama-reasoning.py, then prints which models fit
at the chosen (VRAM, CONTEXT) and (with --download) pulls missing
best-fit candidates. Does NOT write any output file — the probe cache
is the single source of truth for the router and picker. The old
deploy/active-models.yaml is gone; this script just queries.

Usage:
    scripts/select-models.py                         # all families, host VRAM
    scripts/select-models.py --family gemma4
    scripts/select-models.py --vram 16 --context 32768
    scripts/select-models.py --download              # pull missing best-fit
    scripts/select-models.py --prune                 # delete on-disk strays

Errors (network, disk, subprocess) propagate verbatim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from _capability import Capability  # noqa: E402
from _contexts import effective_targets as _ctx_effective_targets  # noqa: E402

CATALOG = REPO_ROOT / "deploy" / "models.yaml"
PROBE_CACHE = REPO_ROOT / "deploy" / ".ollama-reasoning-cache.json"
VLLM_PROBE_CACHE = REPO_ROOT / "deploy" / ".vllm-reasoning-cache.json"
SGLANG_PROBE_CACHE = REPO_ROOT / "deploy" / ".sglang-reasoning-cache.json"

# vLLM/SGLang serve models fully in VRAM — weights + full KV cache +
# CUDA graphs + activations. Ollama (llama.cpp) has a much smaller
# footprint: no CUDA graphs, KV cache is allocated lazily per-request,
# and it spills to CPU RAM when weights approach VRAM. We apply the
# strict accounting only to vLLM/SGLang.
VLLM_OVERHEAD_GB = 1.5       # CUDA graphs (~1) + activations (~0.5)
OLLAMA_OVERHEAD_GB = 0.5     # small buffer for runtime state

KV_BYTES = {"fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}
DEFAULT_CANDIDATE_CONTEXTS = "32768,65536,131072,262144"

OLLAMA_MANIFESTS = Path(
    os.environ.get("OLLAMA_MANIFESTS_DIR",
                   "/var/cache/devai/ollama/models/manifests/registry.ollama.ai/library")
)
VLLM_MODELS = Path(
    os.environ.get("VLLM_MODELS_DIR",
                   "/var/cache/devai/ollama/models/vllm")
)
OLLAMA_CONTAINER = os.environ.get("OLLAMA_CONTAINER", "devai-ollama")
CONTAINER_RUNTIME = os.environ.get("CONTAINER_RUNTIME", "podman")
HF_CLI = os.environ.get("HF_CLI", "hf")

# Host → container path mapping for the devai-ollama service.
# deploy/docker-compose.yaml mounts `/var/cache/devai/ollama:/root/.ollama`,
# so any host path under that root has a deterministic in-container twin.
# Used by `ollama create -w <dir>` so the daemon can resolve the GGUF file
# referenced by FROM in the Modelfile.
OLLAMA_HOST_ROOT = Path(
    os.environ.get("OLLAMA_HOST_ROOT", "/var/cache/devai/ollama")
)
OLLAMA_CONTAINER_ROOT = Path(
    os.environ.get("OLLAMA_CONTAINER_ROOT", "/root/.ollama")
)


def to_container_path(host_path: Path) -> str:
    """Map a host path under OLLAMA_HOST_ROOT to its devai-ollama path.

    Raises ValueError when host_path is outside the mounted root, since
    `ollama create` would not be able to read the file in that case.
    """
    resolved = host_path.resolve()
    try:
        rel = resolved.relative_to(OLLAMA_HOST_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"path {resolved} is not under {OLLAMA_HOST_ROOT} — devai-ollama "
            f"cannot see it. Set OLLAMA_HOST_ROOT or move the file."
        ) from exc
    return str(OLLAMA_CONTAINER_ROOT / rel)


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_size_gb(s: str) -> float:
    s = s.strip().upper().rstrip("B").strip()
    if s.endswith("G"):
        s = s[:-1].strip()
    return float(s)


def kv_per_token_bytes(arch: dict, kv_dtype: str) -> int:
    copies = 1 if arch.get("k_eq_v") else 2
    return (copies
            * int(arch["layers"])
            * int(arch["kv_heads"])
            * int(arch["head_dim"])
            * KV_BYTES[kv_dtype])


def vram_breakdown(model: dict, context: int, kv_dtype: str) -> dict:
    """Return the per-component VRAM breakdown for this model.

    Account for weights + full KV cache at the requested context + a
    runtime overhead. We use the same formula for all backends now:
    Ollama (llama.cpp) *can* spill KV to CPU when overcommitted, but
    that's exactly the slow-path users complain about — better to
    exclude such models from the active set up front.

    Per-backend overhead:
      vLLM/SGLang: VLLM_OVERHEAD_GB (CUDA graphs + activations)
      Ollama:      OLLAMA_OVERHEAD_GB (smaller runtime buffer)
    """
    weight_gb = parse_size_gb(model["size"])
    arch = model.get("arch")
    if not arch:
        # No arch means we can't compute KV. Be conservative: assume
        # worst-case 256 KB/token (rough upper bound for 13B-class models).
        kv_gb = (256 * 1024 * context) / (1024 ** 3)
    else:
        kv_gb = (kv_per_token_bytes(arch, kv_dtype) * context) / (1024 ** 3)

    backends = model.get("backend", [])
    is_ollama_only = ("ollama" in backends
                      and "vllm" not in backends
                      and "sglang" not in backends)
    overhead = OLLAMA_OVERHEAD_GB if is_ollama_only else VLLM_OVERHEAD_GB

    return {
        "weights_gb": round(weight_gb, 2),
        "kv_gb": round(kv_gb, 2),
        "overhead_gb": overhead,
        "total_gb": round(weight_gb + kv_gb + overhead, 2),
        "context": context,
        "kv_dtype": kv_dtype,
    }


def estimate_total_gb(model: dict, context: int, kv_dtype: str) -> float:
    """Back-compat wrapper around vram_breakdown."""
    return vram_breakdown(model, context, kv_dtype)["total_gb"]


def is_ollama_only(model: dict) -> bool:
    backends = model.get("backend", [])
    return ("ollama" in backends
            and "vllm" not in backends
            and "sglang" not in backends)


def is_latest_tag(model: dict) -> bool:
    return str(model.get("name", "")).endswith(":latest")


def name_priority(model: dict) -> tuple[bool, int]:
    """Prefer explicit tags over moving aliases when choosing one trial."""
    name = str(model.get("name", ""))
    return (not name.endswith(":latest"), len(name))


def context_label(context: int) -> str:
    return f"{context // 1024}K" if context >= 1024 else str(context)


def parse_context_list(raw: str) -> list[int]:
    if raw.strip().lower() in ("", "none", "off", "0"):
        return []
    out: list[int] = []
    for part in raw.split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item.endswith("k"):
            value = int(item[:-1]) * 1024
        else:
            value = int(item)
        if value > 0 and value not in out:
            out.append(value)
    return out


def parse_context_value(raw: str) -> int:
    values = parse_context_list(raw)
    if len(values) != 1:
        raise argparse.ArgumentTypeError(
            "expected one context value, e.g. 32768 or 32K"
        )
    return values[0]


# ── Disk detection ───────────────────────────────────────────────────────────

def ollama_on_disk(name: str) -> bool:
    """Ollama stores manifests at manifests/registry.ollama.ai/library/<lib>/<tag>."""
    if ":" not in name:
        return False
    lib, tag = name.split(":", 1)
    return (OLLAMA_MANIFESTS / lib / tag).is_file()


def hf_on_disk(display_name: str) -> bool:
    """HF/NVFP4 models live at VLLM_MODELS/<display_name>/config.json."""
    return (VLLM_MODELS / display_name / "config.json").is_file()


def is_downloaded(model: dict) -> bool:
    source = model.get("source")
    if source == "ollama":
        return ollama_on_disk(model["name"])
    if source == "hf":
        return hf_on_disk(model["name"])
    return False


# ── Download ─────────────────────────────────────────────────────────────────

def pull_ollama(name: str) -> None:
    print(f"  ollama pull {name} ...", flush=True)
    rc = subprocess.call(
        [CONTAINER_RUNTIME, "exec", OLLAMA_CONTAINER, "ollama", "pull", name]
    )
    if rc != 0:
        sys.exit(f"error: ollama pull {name} failed with rc={rc}")


def pull_hf(display_name: str, repo: str) -> None:
    target = VLLM_MODELS / display_name
    target.mkdir(parents=True, exist_ok=True)
    print(f"  hf download {repo} → {target} ...", flush=True)
    rc = subprocess.call([HF_CLI, "download", repo, "--local-dir", str(target)])
    if rc != 0:
        sys.exit(f"error: hf download {repo} failed with rc={rc}")


# Staging for GGUF blobs imported into Ollama via Modelfile. Lives next to
# VLLM_MODELS so we don't pollute Ollama's blob store with files that are
# already absorbed there. Path is informational — Ollama hardlinks/copies
# the file into its blob store on `ollama create`.
GGUF_STAGING = VLLM_MODELS.parent / "_gguf"


def pull_gguf(display_name: str, repo: str, filename: str, family: str) -> None:
    """Download one GGUF file from HF, register it in Ollama via Modelfile.

    Three steps:
      1. hf download <repo> <filename> --local-dir <staging>/<repo-slug>
      2. write Modelfile: FROM <filename> + RENDERER <family> + PARSER <family>
      3. ollama create <display_name> -f Modelfile (inside devai-ollama)

    The RENDERER and PARSER directives are critical for capability
    detection. Without them the imported model is treated as a raw
    completion engine and Ollama refuses tool / thinking calls
    ("does not support tools" on /v1/messages, /v1/chat/completions).
    The renderer/parser names are 1:1 with the family name in our
    catalog (qwen3.5, gemma4, nemotron-3-nano, …) — confirmed by
    inspecting registry-served models of those families.

    `display_name` is the catalog tag (e.g. `qwen3.5:27b-ud-q3_k_xl`).
    Once `ollama create` completes, the GGUF bytes are absorbed into
    Ollama's blob store and the staging file becomes redundant — we
    keep it as a readable artifact for re-import after a blob-store wipe.
    """
    repo_slug = repo.replace("/", "_")
    target_dir = GGUF_STAGING / repo_slug
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / filename
    print(f"  hf download {repo} {filename} → {target_file} ...", flush=True)
    rc = subprocess.call(
        [HF_CLI, "download", repo, filename, "--local-dir", str(target_dir)]
    )
    if rc != 0:
        sys.exit(f"error: hf download {repo} {filename} failed with rc={rc}")
    if not target_file.is_file():
        sys.exit(
            f"error: hf download claimed success but {target_file} is missing"
        )
    modelfile = target_dir / f"Modelfile.{filename}"
    modelfile.write_text(
        f"FROM {filename}\n"
        f"RENDERER {family}\n"
        f"PARSER {family}\n"
    )
    container_dir = to_container_path(target_dir)
    print(f"  ollama create {display_name} -f {modelfile.name} "
          f"(in {container_dir}) ...", flush=True)
    rc = subprocess.call([
        CONTAINER_RUNTIME, "exec", "-w", container_dir, OLLAMA_CONTAINER,
        "ollama", "create", display_name, "-f", modelfile.name,
    ])
    if rc != 0:
        sys.exit(f"error: ollama create {display_name} failed with rc={rc}")


def pull(model: dict) -> None:
    src = model.get("source")
    if src == "ollama":
        pull_ollama(model["name"])
    elif src == "hf":
        pull_hf(model["name"], model["repo"])
    elif src == "gguf":
        if not model.get("repo") or not model.get("gguf_filename"):
            sys.exit(f"error: gguf source row {model['name']} is missing "
                     f"repo or gguf_filename")
        family = model.get("family") or ""
        if not family:
            sys.exit(f"error: gguf source row {model['name']} is missing family")
        pull_gguf(model["name"], model["repo"], model["gguf_filename"], family)
    else:
        sys.exit(f"error: unknown source '{src}' for {model['name']}")


# ── Deletion ─────────────────────────────────────────────────────────────────

def _dir_bytes(p: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def reclaim_bytes(model: dict) -> int:
    """Return the number of bytes that would be freed by deleting this model."""
    if model["source"] == "hf":
        target = VLLM_MODELS / model["name"]
        return _dir_bytes(target) if target.is_dir() else 0
    if model["source"] == "ollama":
        # Weight layer is the big one; read it from the manifest file.
        name = model["name"]
        if ":" not in name:
            return 0
        lib, tag = name.split(":", 1)
        manifest = OLLAMA_MANIFESTS / lib / tag
        if not manifest.is_file():
            return 0
        import json
        try:
            data = json.loads(manifest.read_text())
        except Exception:
            return 0
        return sum(int(L.get("size", 0)) for L in (data.get("layers") or []))
    return 0


def delete_hf(display_name: str) -> None:
    target = VLLM_MODELS / display_name
    if not target.is_dir():
        return
    print(f"  rm -rf {target} ...", flush=True)
    subprocess.check_call(["rm", "-rf", str(target)])


def delete_ollama(name: str) -> None:
    # `ollama rm` inside the container handles manifest + blob refcounts.
    print(f"  ollama rm {name} ...", flush=True)
    rc = subprocess.call(
        [CONTAINER_RUNTIME, "exec", OLLAMA_CONTAINER, "ollama", "rm", name]
    )
    if rc != 0:
        sys.exit(f"error: ollama rm {name} failed with rc={rc} "
                 f"(is the ollama container running?)")


def delete(model: dict) -> None:
    if model["source"] == "ollama":
        delete_ollama(model["name"])
    elif model["source"] == "hf":
        delete_hf(model["name"])
    else:
        sys.exit(f"error: unknown source for {model['name']}")


# ── Shadow / orphan detection ────────────────────────────────────────────────

def shadow_ollama_tags(catalog_models: list[dict]) -> list[str]:
    """Return ollama <library>:<tag> entries that exist on disk but are
    not in the full catalog (e.g. hand-made aliases from `ollama cp`).

    These are the reason `ollama rm` of a catalog tag often reclaims no
    space: a shadow alias still references the shared blobs."""
    catalog_names = {m["name"] for m in catalog_models
                     if m.get("source") == "ollama"}
    found: list[str] = []
    if not OLLAMA_MANIFESTS.exists():
        return found
    for lib_dir in sorted(OLLAMA_MANIFESTS.iterdir()):
        if not lib_dir.is_dir():
            continue
        for tag_file in sorted(lib_dir.iterdir()):
            if not tag_file.is_file():
                continue
            name = f"{lib_dir.name}:{tag_file.name}"
            if name not in catalog_names:
                found.append(name)
    return found


def orphan_blob_gb() -> float:
    """Return total GB of on-disk blobs not referenced by any manifest."""
    import json
    blobs_dir = OLLAMA_MANIFESTS.parent.parent / "blobs"
    if not blobs_dir.is_dir():
        return 0.0
    referenced: set[str] = set()
    for root, _dirs, files in os.walk(OLLAMA_MANIFESTS.parent.parent / "manifests"):
        for f in files:
            p = Path(root) / f
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            for layer in (data.get("layers") or []) + [data.get("config") or {}]:
                d = layer.get("digest")
                if d:
                    referenced.add(d)
    total = 0
    for b in blobs_dir.iterdir():
        digest = b.name.replace("-", ":", 1)
        if digest not in referenced:
            try:
                total += b.stat().st_size
            except OSError:
                pass
    return total / (1024 ** 3)


# ── Output ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProbeCache:
    """v2 cache view: digest-keyed entries plus a name→digest reverse index.

    Entries are produced by scripts/probe-ollama-reasoning.py and migrated
    forward from v1 on first read. A row in the cache is the canonical
    record for one set of weights (one digest); aliases sharing those
    weights are listed in `entry["aliases"]`.
    """

    by_digest: dict
    by_name: dict

    def lookup(self, name: str) -> dict:
        digest = self.by_name.get(name)
        if not digest:
            return {}
        return self.by_digest.get(digest) or {}

    def digest_of(self, name: str) -> str:
        return self.by_name.get(name) or ""

    def aliases_for(self, name: str) -> list[str]:
        entry = self.lookup(name)
        return list(entry.get("aliases") or [])


def load_probe_cache() -> ProbeCache:
    """Read deploy/.ollama-reasoning-cache.json and build the v3 view.

    Cache schema v3 is digest-keyed with probes nested by VRAM band
    (see scripts/probe-ollama-reasoning.py). v1 and v2 entries are
    migrated in-memory using the prober's helper; the on-disk file is
    not rewritten from here (`make probe` does that). Missing file →
    every model resolves to capability=unknown.
    """
    if not PROBE_CACHE.is_file():
        return ProbeCache(by_digest={}, by_name={})
    import json
    try:
        raw = json.loads(PROBE_CACHE.read_text())
    except (OSError, json.JSONDecodeError):
        return ProbeCache(by_digest={}, by_name={})
    by_digest: dict
    if all(
        isinstance(v, dict) and v.get("schema_version") == 3
        for v in raw.values()
    ):
        by_digest = {
            (entry.get("digest") or k): entry
            for k, entry in raw.items()
            if isinstance(entry, dict)
        }
    else:
        # In-memory migration to v3; on-disk file is unchanged.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_probe_module",
            REPO_ROOT / "scripts" / "probe-ollama-reasoning.py",
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_probe_module"] = mod  # frozen-dataclass workaround
        spec.loader.exec_module(mod)
        by_digest, _, _ = mod.ensure_v3(raw)
    by_name: dict = {}
    for digest, entry in by_digest.items():
        for alias in entry.get("aliases") or []:
            by_name[alias] = digest
    return ProbeCache(by_digest=by_digest, by_name=by_name)


def lookup_probe(model_name: str, cache: ProbeCache) -> dict:
    """Return the digest entry for an Ollama model name, or {}."""
    return cache.lookup(model_name)


def lookup_capability(model_name: str, cache: ProbeCache) -> tuple[str, str | None]:
    """Return (capability, disable_verified-as-string-or-None) for an Ollama
    model name."""
    rec = lookup_probe(model_name, cache)
    if not rec:
        return Capability.UNKNOWN, None
    cap = rec.get("capability", Capability.UNKNOWN)
    disable = rec.get("disable_verified")
    if isinstance(disable, bool):
        return cap, "true" if disable else "false"
    if disable == "error":
        return cap, "error"
    return cap, None


@dataclass(frozen=True)
class HFProbeCaches:
    """vLLM and SGLang probe caches, keyed by `<repo>@<sha>`.

    Schema v1 — see scripts/_probe_hf_common.py for the full shape.
    Each backend's cache is independent; missing files resolve to {}.
    """

    vllm: dict
    sglang: dict

    def cache_for(self, backend: str) -> dict:
        if backend == "vllm":
            return self.vllm
        if backend == "sglang":
            return self.sglang
        return {}

    def lookup(self, repo: str, sha: str, backend: str) -> dict:
        if not (repo and sha):
            return {}
        return self.cache_for(backend).get(f"{repo}@{sha}") or {}

    def has_working_probe(self, backend: str) -> bool:
        """True iff *any* row in the backend's cache has a fits=true cell.

        Used as a coarse gate: when a backend has at least one model
        confirmed loadable, HF rows for that backend become eligible
        trial candidates (the user can probe specific rows to confirm
        their fit later — the formula path provides the size estimate
        in the meantime).
        """
        for entry in self.cache_for(backend).values():
            if not isinstance(entry, dict):
                continue
            if entry.get("capability") in (Capability.ERROR, Capability.UNSUPPORTED_ARCH):
                continue
            for band in (entry.get("probes") or {}).values():
                if not isinstance(band, dict):
                    continue
                for cell in band.values():
                    if isinstance(cell, dict) and cell.get("fits"):
                        return True
        return False


def load_hf_probe_caches() -> HFProbeCaches:
    """Read both HF probe cache files. Missing or malformed → empty dict."""
    import json

    def _read(path: Path) -> dict:
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    return HFProbeCaches(
        vllm=_read(VLLM_PROBE_CACHE),
        sglang=_read(SGLANG_PROBE_CACHE),
    )


def hf_probe_at_context(
    entry: dict, vram_gb: int, context: int,
) -> dict | None:
    """Return the HF cache cell at (vram, ctx) or None.

    Mirrors probe_at_context for the schema v1 shape: cells live under
    `entry["probes"][<vram>][<ctx>]` and carry `fits` instead of
    `fully_on_gpu`. The effective ctx is min(context, max_context).
    """
    if not entry:
        return None
    max_ctx = int(entry.get("max_context") or 0)
    eff_ctx = min(context, max_ctx) if max_ctx else context
    band = (entry.get("probes") or {}).get(str(int(vram_gb)))
    if not isinstance(band, dict):
        return None
    rec = band.get(str(eff_ctx))
    return rec if isinstance(rec, dict) else None


def probe_at_context(entry: dict, vram_gb: int, context: int) -> dict | None:
    """Return the per-(vram, ctx) probe cell from a v3 digest entry.

    The effective context is min(context, max_context). The model's
    design ceiling is a hard physical limit — asking a 128K-only model
    to run at 256K just runs at 128K. Returns None when the cache has
    no cell at that (vram, eff_ctx) — i.e. `make probe` hasn't been
    run for that band/tier yet, or the tier was capped above the
    model's max.
    """
    if not entry:
        return None
    max_ctx = int(entry.get("max_context") or 0)
    eff_ctx = min(context, max_ctx) if max_ctx else context
    band = (entry.get("probes") or {}).get(str(int(vram_gb)))
    if not isinstance(band, dict):
        return None
    rec = band.get(str(eff_ctx))
    if isinstance(rec, dict):
        return rec
    return None


def print_active_set(rows: list["Row"], vram: float, context: int) -> None:
    """Render the eligible (fits + fully_on_gpu) set as a summary table.

    Replaces the old `write_active` YAML emit with a print-only diagnostic.
    The probe cache is the source of truth; downstream consumers (router,
    picker) read it directly. This view is informational.
    """
    eligible = [r for r in rows if r.active_eligible]
    print()
    print(f"  ── Eligible at VRAM={vram:g} GB / CONTEXT={context} "
          f"({len(eligible)} model(s)) ──")
    if not eligible:
        print("    (no models fit at this (VRAM, CONTEXT) — run `make probe` "
              "to populate the cache)")
        return
    print(f"    {'NAME':<42s}  {'FAMILY':<14s}  {'TOTAL GB':>9s}  {'CTX':>6s}  CAP")
    print(f"    {'-' * 96}")
    for r in eligible:
        m = r.model
        v = m.get("vram") or {}
        eff_ctx = int(v.get("context") or 0)
        cap = v.get("context_capability") or Capability.UNKNOWN
        total = v.get("total_gb") or 0.0
        family = m.get("family", "?")
        ctx_label = f"{eff_ctx // 1024}K" if eff_ctx >= 1024 else str(eff_ctx)
        print(f"    {m['name']:<42s}  {family:<14s}  {total:>8.2f}G  "
              f"{ctx_label:>6s}  {cap}")


# ── Main ─────────────────────────────────────────────────────────────────────

@dataclass
class Row:
    model: dict
    total_gb: float
    fits: bool
    downloaded: bool
    candidate: bool = False
    suppress_reason: str = ""
    active_eligible: bool = False
    # (family, backend, ctx) cells this row was selected to fill. Populated
    # by assign_cell_candidates when the row is a chosen candidate.
    cells_filled: list = field(default_factory=list)


_PROBED_SOURCES = ("probe", "hf-probe")


def active_eligible(row: Row) -> bool:
    """Return whether this row is allowed into the active set.

    A row is active iff it fits, is downloaded, AND its measurement
    came from a real probe (Ollama or vLLM/SGLang) — not the analytic
    formula. The fully_on_gpu/fits flag must be true.
    """
    if not (row.fits and row.downloaded):
        return False
    vram = row.model.get("vram") or {}
    if vram.get("source") not in _PROBED_SOURCES:
        return False
    return bool(vram.get("fully_on_gpu", True))


def trusted_for_family_gate(row: Row) -> bool:
    """Return whether this row can suppress smaller same-family candidates.

    Trusted == fits AND (not yet downloaded, or probed-and-loadable).
    Formula-only downloaded rows do not gate the family — they could
    still spill at runtime, so a smaller probed alternative shouldn't
    be hidden behind them.
    """
    if not row.fits:
        return False
    if not row.downloaded:
        return True
    vram = row.model.get("vram") or {}
    return (
        vram.get("source") in _PROBED_SOURCES
        and bool(vram.get("fully_on_gpu", True))
    )


def candidate_signature(row: Row) -> tuple:
    """Estimated identity for aliases before a model has been downloaded."""
    model = row.model
    arch = model.get("arch") or {}
    return (
        model.get("family", ""),
        round(row.total_gb, 1),
        str(model.get("size", "")),
        arch.get("layers"),
        arch.get("kv_heads"),
        arch.get("head_dim"),
        bool(arch.get("k_eq_v")),
    )


def _parse_param_b(label: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*b", label.lower())
    return float(m.group(1)) if m else 0.0


def param_size_hint(model: dict) -> float:
    details = model.get("details") or {}
    if details.get("param_size"):
        parsed = _parse_param_b(str(details["param_size"]))
        if parsed:
            return parsed
    name = str(model.get("name", ""))
    matches = re.findall(r"(?<![a-z0-9])(\d+(?:\.\d+)?)b(?![a-z0-9])", name.lower())
    if matches:
        return float(matches[0])
    return 0.0


def row_quality(row: Row) -> float:
    """Approximate model scale independent of KV allocation.

    Prefer parameter-count hints over disk size: a 27B Q4 candidate is
    usually the larger family tier than a 9B BF16 candidate even though
    the 9B file can be larger.
    """
    params_b = param_size_hint(row.model)
    if params_b:
        return params_b
    vram = row.model.get("vram") or {}
    if vram.get("weights_overhead_gb") is not None:
        return float(vram["weights_overhead_gb"])
    if vram.get("weights_gb") is not None:
        return float(vram["weights_gb"])
    try:
        return parse_size_gb(str(row.model.get("size", "0")))
    except ValueError:
        return 0.0


def trial_score(row: Row, vram: float) -> tuple[float, bool, int]:
    """Higher is better: closest fit from below, then explicit tag."""
    unused_gb = max(0.0, vram - row.total_gb)
    return (-unused_gb, *name_priority(row.model))


def _is_eligible_backend(row: Row, hf_caches: HFProbeCaches) -> bool:
    """A row's backend is eligible if it's Ollama (always probed) or an
    HF backend that has at least one fits=true entry in its cache.

    The gate is coarse: a working probe somewhere in the vLLM cache
    proves vLLM serves at least one model, which is enough signal to
    download other vLLM rows speculatively. Fine-grained per-row checks
    happen at picker time, when the user actually picks a model.
    """
    if is_ollama_only(row.model):
        return True
    backends = row.model.get("backend") or []
    for b in ("vllm", "sglang"):
        if b in backends and hf_caches.has_working_probe(b):
            return True
    return False


# Quantization rank table — higher = higher precision. Used as the
# secondary tiebreak in cell_quality(). Strings are uppercased before
# lookup; longer keys take precedence (`Q4_K_M` beats `Q4`). Values are
# loosely calibrated against effective bits-per-weight.
_QUANT_RANK = {
    # full precision
    "F32": 32.0, "FP32": 32.0,
    "F16": 16.0, "FP16": 16.0, "BF16": 16.0, "HALF": 16.0,
    # 8-bit
    "Q8_0": 8.0, "Q8_K": 8.0,
    "INT8": 8.0, "FP8": 8.0, "MXFP8": 8.0,
    # 6-bit
    "Q6_K": 6.0,
    # 5-bit
    "Q5_K_M": 5.5, "Q5_K_S": 5.3, "Q5_1": 5.1, "Q5_0": 5.0,
    # 4-bit
    "Q4_K_M": 4.5, "Q4_K_S": 4.3, "Q4_1": 4.1, "Q4_0": 4.0,
    "INT4": 4.0, "NVFP4": 4.0, "FP4": 4.0, "MXFP4": 4.0,
    "AWQ": 4.0, "GPTQ": 4.0,
    # 3-bit
    "Q3_K_L": 3.7, "Q3_K_M": 3.5, "Q3_K_S": 3.3,
    "IQ3_XXS": 3.0, "IQ3_M": 3.2, "IQ3_S": 3.1,
    "UD-Q3_K_XL": 3.6, "UD-IQ3_XXS": 3.0,
    # 2-bit
    "Q2_K": 2.0, "IQ2_M": 2.2, "IQ2_S": 2.1, "IQ2_XS": 2.0, "IQ2_XXS": 1.9,
}

# Module-level cached sort of quant tokens by length descending so the
# substring search prefers `Q5_K_M` over `Q5_K`, etc.
_QUANT_TOKENS = sorted(_QUANT_RANK.keys(), key=lambda k: -len(k))


def quant_rank(model: dict) -> float:
    """Numeric rank of a model's quantization. Higher = higher precision.

    Lookup priority:
      1. Probe-derived `details.quantization` (Ollama).
      2. Token in catalog `name` (matches the longest token first so
         `Q4_K_M` wins over a bare `Q4`).
      3. Default 4.0 — assume mid-range when unknown.
    """
    details = model.get("details") or {}
    q = (details.get("quantization") or "").upper().strip()
    if q and q in _QUANT_RANK:
        return _QUANT_RANK[q]
    name = str(model.get("name") or "").upper()
    for token in _QUANT_TOKENS:
        if token in name:
            return _QUANT_RANK[token]
    return 4.0


def cell_quality(row: Row) -> tuple[float, float, float, int]:
    """Sort key for picking the best variant within a (family, backend, ctx)
    cell. Higher is better.

      1. param_size_hint (params dominate; 14B > 7B regardless of quant).
      2. quant_rank (14B-Q8 beats 14B-Q4).
      3. total_gb (closer-to-budget fit wins ties).
      4. name_priority[1] (longer / more-explicit tag breaks ties).
    """
    return (
        param_size_hint(row.model),
        quant_rank(row.model),
        row.total_gb,
        name_priority(row.model)[1],
    )


def _row_probe_entry(
    row: Row, backend: str,
    probe_cache: ProbeCache, hf_caches: HFProbeCaches,
) -> dict:
    """Return the top-level probe entry for (model, backend) or {}.

    Ollama uses its digest cache; vLLM/SGLang use the repo+sha HF caches.
    Empty dict when the model wasn't probed for this backend at all.
    """
    m = row.model
    if backend == "ollama":
        if not is_ollama_only(m):
            return {}
        return lookup_probe(m["name"], probe_cache)
    if backend in ("vllm", "sglang"):
        if is_ollama_only(m):
            return {}
        return hf_caches.lookup(m.get("repo") or "", m.get("sha") or "", backend)
    return {}


def _row_probe_cell(
    row: Row, backend: str, vram_band: int, ctx: int,
    probe_cache: ProbeCache, hf_caches: HFProbeCaches,
) -> dict | None:
    """Return the probe cell at (model, backend, vram, ctx), or None if
    no measurement exists at that exact tier."""
    entry = _row_probe_entry(row, backend, probe_cache, hf_caches)
    if not entry:
        return None
    if backend == "ollama":
        return probe_at_context(entry, vram_band, ctx)
    return hf_probe_at_context(entry, vram_band, ctx)


def _row_probe_passed(
    row: Row, backend: str, vram_band: int, ctx: int,
    probe_cache: ProbeCache, hf_caches: HFProbeCaches,
) -> bool:
    """True iff the probe explicitly confirmed (model, backend, vram, ctx)
    fits and runs fully on GPU. Used to decide if a cell is already
    filled by a downloaded variant — pending-probe variants do NOT count."""
    cell = _row_probe_cell(row, backend, vram_band, ctx, probe_cache, hf_caches)
    if not cell:
        return False
    if backend == "ollama":
        return bool(cell.get("fully_on_gpu", False))
    return bool(cell.get("fits", False))


def _row_probe_rejected(
    row: Row, backend: str, vram_band: int, ctx: int,
    probe_cache: ProbeCache, hf_caches: HFProbeCaches,
) -> bool:
    """True iff the probe explicitly rejected (model, backend, vram, ctx).

    Rejection reasons:
      - top-level capability is `error` or `unsupported_arch` (the model
        can't load on this backend at any tier);
      - this exact (vram, ctx) cell exists with fits=false (HF) or
        fully_on_gpu=false (Ollama).
    Cells absent from the cache do NOT count as rejection — they're just
    unprobed at that tier.
    """
    entry = _row_probe_entry(row, backend, probe_cache, hf_caches)
    if entry.get("capability") in (Capability.ERROR, Capability.UNSUPPORTED_ARCH):
        return True
    cell = _row_probe_cell(row, backend, vram_band, ctx, probe_cache, hf_caches)
    if not cell:
        return False
    if backend == "ollama":
        return cell.get("fully_on_gpu") is False
    return cell.get("fits") is False


def _model_file_id(model: dict) -> tuple:
    """Identity for download dedup. Two catalog rows referencing the same
    on-disk file share an id (e.g., one HF safetensors directory backs
    both vLLM and SGLang cells — pull once, two probes can run later)."""
    if is_ollama_only(model):
        return ("ollama", model.get("name"))
    return ("hf", model.get("repo"))


def _backends_for(model: dict) -> list[str]:
    """Return the backends ('ollama', 'vllm', 'sglang') a catalog row
    advertises. Each backend is its own cell axis in the matrix."""
    out: list[str] = []
    for b in model.get("backend") or []:
        if b in ("ollama", "vllm", "sglang"):
            out.append(b)
    return out


def assign_cell_candidates(
    models: list[dict],
    contexts: list[int],
    kv_dtype: str,
    min_total: float,
    vram_budget: float,
    probe_cache: ProbeCache,
    hf_caches: HFProbeCaches,
    max_per_cell: int = 1,
) -> tuple[list[Row], dict[tuple, dict], dict[int, list[Row]]]:
    """Build the (family, backend, ctx) cell matrix and select trial
    candidates per cell.

    Per cell:
      - skip if any downloaded variant has a probe-passed cell here;
      - drop probe-rejected variants (capability error or fits=false at
        this exact (vram, ctx));
      - rank surviving variants by `cell_quality` (params → quant rank
        → total VRAM) and pick the top `max_per_cell`.

    Returns:
      - candidate_rows: file-deduplicated list of Rows to download.
        Each row has `r.candidate = True` set as a side effect, and
        `r.cells_filled` lists the cells the row was selected for.
      - cell_index: {(family, backend, ctx): {"status": ..., "row": ...}}
        with status ∈ {filled, candidate, pending_probe, no_options}.
      - rows_by_ctx: per-context Row lists (for downstream display).
    """
    vram_band = int(round(vram_budget))
    rows_by_ctx: dict[int, list[Row]] = {}
    for ctx in contexts:
        rows_by_ctx[ctx] = build_rows(
            models, ctx, kv_dtype, min_total, vram_budget,
            probe_cache, hf_caches,
        )

    cell_index: dict[tuple, dict] = {}

    for ctx, rows in rows_by_ctx.items():
        # Group fitting rows by (family, backend) for this ctx
        bucket: dict[tuple[str, str], list[Row]] = {}
        for row in rows:
            if not row.fits:
                continue
            family = row.model.get("family") or ""
            for backend in _backends_for(row.model):
                if backend in ("vllm", "sglang") and not hf_caches.has_working_probe(backend):
                    continue
                bucket.setdefault((family, backend), []).append(row)

        for (family, backend), cell_rows in bucket.items():
            cell_key = (family, backend, ctx)

            # Cell-fill check: any downloaded variant probe-passed at this cell?
            filled_by: Row | None = None
            for r in cell_rows:
                if r.downloaded and _row_probe_passed(
                    r, backend, vram_band, ctx, probe_cache, hf_caches,
                ):
                    filled_by = r
                    break
            if filled_by is not None:
                cell_index[cell_key] = {"status": "filled", "row": filled_by}
                continue

            # Drop probe-rejected variants for this exact cell
            usable = [
                r for r in cell_rows
                if not _row_probe_rejected(
                    r, backend, vram_band, ctx, probe_cache, hf_caches,
                )
            ]

            missing = [r for r in usable if not r.downloaded]
            on_disk_unprobed = [r for r in usable if r.downloaded]

            if not missing:
                # No new download needed. Cell may resolve once the
                # on-disk variant gets probed at this tier.
                if on_disk_unprobed:
                    on_disk_unprobed.sort(key=cell_quality, reverse=True)
                    cell_index[cell_key] = {
                        "status": "pending_probe", "row": on_disk_unprobed[0],
                    }
                else:
                    cell_index[cell_key] = {"status": "no_options", "row": None}
                continue

            # Rank by quality — params first, then quant precision
            missing.sort(key=cell_quality, reverse=True)
            picked = missing[:max(0, max_per_cell)]
            if not picked:
                cell_index[cell_key] = {"status": "no_options", "row": None}
                continue
            cell_index[cell_key] = {"status": "candidate", "row": picked[0]}
            for r in picked:
                r.candidate = True
                r.cells_filled.append(cell_key)

    # File-level dedup of the candidate list — two cells targeting the
    # same on-disk file (e.g., the vLLM and SGLang versions of one HF
    # repo) trigger one download.
    seen: set = set()
    deduped: list[Row] = []
    for cell_key, info in cell_index.items():
        if info.get("status") != "candidate":
            continue
        row = info["row"]
        fid = _model_file_id(row.model)
        if fid in seen:
            continue
        seen.add(fid)
        deduped.append(row)

    return deduped, cell_index, rows_by_ctx


def _hf_lookup_with_priority(
    m: dict, hf_caches: HFProbeCaches,
) -> tuple[dict, str | None]:
    """Find the first HF backend (vllm > sglang) with a non-failed entry
    for this catalog row. Returns (entry, backend) or ({}, None).
    """
    backends = m.get("backend") or []
    repo = m.get("repo") or ""
    sha = (m.get("sha") or "").strip()
    for b in ("vllm", "sglang"):
        if b not in backends:
            continue
        entry = hf_caches.lookup(repo, sha, b)
        if entry and entry.get("capability") not in (Capability.ERROR, Capability.UNSUPPORTED_ARCH):
            return entry, b
    return {}, None


def build_rows(
    models: list[dict],
    context: int,
    kv_dtype: str,
    min_total: float,
    vram_budget: float,
    probe_cache: ProbeCache,
    hf_caches: HFProbeCaches,
) -> list[Row]:
    """One Row per catalog entry, annotated with the measurement at CONTEXT.

    Sources of `vram` data, in priority order:
      1. Ollama digest cache  (source == "probe")
      2. vLLM/SGLang HF cache (source == "hf-probe")
      3. Analytic formula     (source == "formula") — fallback when no
         cache has a row, or the cache has no cell at (vram, ctx).

    The HF caches are consulted in priority vllm > sglang for rows
    declaring multiple backends. The first backend with a non-failed
    entry wins; future work could surface both verdicts to the picker.
    """
    rows: list[Row] = []
    for original in models:
        m = dict(original)
        model_is_ollama_only = is_ollama_only(m)

        probe_entry = lookup_probe(m["name"], probe_cache) if model_is_ollama_only else {}
        vram_gb = int(round(vram_budget))
        probe_record = (
            probe_at_context(probe_entry, vram_gb, context)
            if probe_entry else None
        )

        if probe_record:
            total = float(probe_record.get("actual_total_gb") or 0.0)
            vram_gb = float(probe_record.get("actual_vram_gb") or total)
            eff_ctx = int(probe_record.get("ctx") or probe_record.get("actual_context") or context)
            fully_on_gpu = bool(probe_record.get("fully_on_gpu", False))
            ctx_capability = str(probe_record.get("capability") or Capability.UNKNOWN)
            details = {}
            if probe_entry.get("param_size_label"):
                details["param_size"] = probe_entry["param_size_label"]
            if probe_entry.get("quantization"):
                details["quantization"] = probe_entry["quantization"]
            if details:
                m["details"] = details
            breakdown = {
                "source": "probe",
                "total_gb": round(total, 2),
                "vram_gb": round(vram_gb, 2),
                "fully_on_gpu": fully_on_gpu,
                "context": eff_ctx,
                "max_context": probe_entry.get("max_context"),
                "context_capability": ctx_capability,
            }
        elif model_is_ollama_only and probe_entry:
            # We have a digest entry but no probe at this context — most
            # likely the user changed CONTEXT after the last probe run.
            # Carry the context-less data so the picker can still show
            # alternative tiers; mark un-fitting so we don't emit it.
            total = 0.0
            breakdown = {
                "source": "probe-missing",
                "total_gb": 0.0,
                "vram_gb": 0.0,
                "fully_on_gpu": False,
                "context": context,
                "max_context": probe_entry.get("max_context"),
                "context_capability": Capability.UNKNOWN,
            }
        elif not model_is_ollama_only:
            hf_entry, hf_backend = _hf_lookup_with_priority(m, hf_caches)
            hf_record = (
                hf_probe_at_context(hf_entry, vram_gb, context)
                if hf_entry else None
            )
            if hf_record:
                vram_actual = float(hf_record.get("actual_vram_gb") or 0.0)
                eff_ctx = int(hf_record.get("actual_context") or hf_record.get("ctx") or context)
                # Mirror the router's synthesizeHFFromCache + the picker's
                # _vram_from_hf_probe gate: a cell that loaded (fits=true)
                # but OOMed under a near-full-context request
                # (serving_ok=false, set by `make probe-load-*`) is NOT
                # serveable at that ctx, so model-fit / model-pull must not
                # report it as fitting. serving_ok absent (load probe never
                # ran) keeps the fit-only verdict (legacy behaviour).
                fits_flag = (bool(hf_record.get("fits", False))
                             and hf_record.get("serving_ok") is not False)
                ctx_capability = str(hf_record.get("capability") or hf_entry.get("capability") or Capability.UNKNOWN)
                # vLLM/SGLang load weights+KV+CUDA-graphs into the static
                # pool — actual_vram_gb is the most honest "total" we have.
                total = vram_actual or float(hf_entry.get("max_context") and parse_size_gb(m.get("size", "0")) or 0.0)
                breakdown = {
                    "source": "hf-probe",
                    "backend": hf_backend,
                    "total_gb": round(total, 2),
                    "vram_gb": round(vram_actual, 2),
                    "fully_on_gpu": fits_flag,
                    "context": eff_ctx,
                    "max_context": hf_entry.get("max_context"),
                    "context_capability": ctx_capability,
                }
            elif hf_entry:
                # Cache knows the model but not at this (vram, ctx). Same
                # signal as Ollama's "probe-missing" — show as un-fitting
                # so the user knows to re-probe at this tier.
                total = 0.0
                breakdown = {
                    "source": "hf-probe-missing",
                    "backend": hf_backend,
                    "total_gb": 0.0,
                    "vram_gb": 0.0,
                    "fully_on_gpu": False,
                    "context": context,
                    "max_context": hf_entry.get("max_context"),
                    "context_capability": Capability.UNKNOWN,
                }
            else:
                formula = vram_breakdown(m, context, kv_dtype)
                total = formula["total_gb"]
                breakdown = {**formula, "source": "formula"}
        else:
            formula = vram_breakdown(m, context, kv_dtype)
            total = formula["total_gb"]
            breakdown = {**formula, "source": "formula"}
        m["vram"] = breakdown
        row = Row(
            model=m,
            total_gb=float(total),
            fits=(min_total <= total <= vram_budget) if total > 0 else False,
            downloaded=is_downloaded(m),
        )
        row.active_eligible = active_eligible(row)
        rows.append(row)
    rows.sort(key=lambda r: (not r.fits, r.total_gb))
    return rows


def _format_cell(info: dict, width: int) -> str:
    """Render one matrix cell as a fixed-width string with status glyph."""
    glyph_by_status = {
        "filled": "✓", "candidate": "↓", "pending_probe": "◌",
        "no_options": "-",
    }
    status = info.get("status") if info else "no_options"
    glyph = glyph_by_status.get(status, "?")
    if status == "no_options" or not info or not info.get("row"):
        return f"{glyph:<{width}s}"
    name = info["row"].model.get("name") or "?"
    # Truncate from the LEFT — quant suffix is more informative than
    # the family prefix when names get crowded.
    body = f"{glyph} {name}"
    if len(body) > width:
        body = f"{glyph} …{name[-(width - 4):]}"
    return f"{body:<{width}s}"


def print_cell_matrix(
    cell_index: dict[tuple, dict],
    contexts: list[int],
    vram_budget: float,
    kv_dtype: str,
) -> None:
    """Render the (family, backend) × ctx matrix the selector built.

    One row per (family, backend) pair, one column per context. Cells
    show the variant filling that cell (downloaded + probed, marked ✓),
    the trial candidate the selector would pull (marked ↓), an on-disk
    variant awaiting probe (◌), or no fitting option (-).
    """
    if not cell_index or not contexts:
        return
    families = sorted({k[0] for k in cell_index.keys()})
    backends_seen = sorted({k[1] for k in cell_index.keys()},
                           key=lambda b: ("ollama", "vllm", "sglang").index(b)
                           if b in ("ollama", "vllm", "sglang") else 99)

    col_w = 22
    name_w = 18
    backend_w = 7

    def header_line() -> str:
        cols = "  ".join(f"{context_label(c):<{col_w}s}" for c in contexts)
        return (f"  {'FAMILY':<{name_w}s}  {'BACKEND':<{backend_w}s}  {cols}")

    print(f"  ── Cell matrix: vram={vram_budget:g} GB, kv={kv_dtype}, "
          f"contexts=[{', '.join(context_label(c) for c in contexts)}] ──")
    print()
    print(header_line())
    print(f"  {'-' * (name_w + backend_w + 4 + (col_w + 2) * len(contexts) - 2)}")

    for family in families:
        printed_header = False
        for backend in backends_seen:
            # Skip a (family, backend) row entirely if the matrix has no
            # cell at all for this pair (e.g., HF-only family + ollama).
            if not any((family, backend, c) in cell_index for c in contexts):
                continue
            cells = "  ".join(
                _format_cell(cell_index.get((family, backend, c)), col_w)
                for c in contexts
            )
            label = "" if printed_header else family
            print(f"  {label:<{name_w}s}  {backend:<{backend_w}s}  {cells}")
            printed_header = True
    print()
    print("  Legend:  ✓ filled (probed)   ↓ candidate (will pull)   "
          "◌ pending probe   - no option")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", default="")
    ap.add_argument("--name", default="",
                    help="Pull a specific catalog row by its `name` field, "
                         "bypassing the (VRAM, context) matrix entirely. "
                         "Use to fetch a single HF model whose fit-data is "
                         "not yet in the probe cache. Exits after pull.")
    ap.add_argument("--vram", type=float,
                    default=float(os.environ.get("GPU_MEMORY_GB", "24")))
    # --context: single-context override. When omitted, the selector
    # iterates the matrix from --contexts (default 32K/64K/128K/256K).
    # The MAX_CONTEXT_LEN env still seeds a default, but the matrix
    # mode is the right answer for cross-backend benchmarking.
    ap.add_argument("--context", type=parse_context_value, default=None,
                    help="Single context (tokens or 32K/64K notation) to plan "
                         "against. Omit for matrix mode iterating --contexts.")
    ap.add_argument("--contexts",
                    default=os.environ.get("CONTEXTS", DEFAULT_CANDIDATE_CONTEXTS),
                    help="Comma-separated contexts for matrix mode "
                         "(default 32K,64K,128K,256K). Ignored when --context "
                         "is set.")
    ap.add_argument("--kv-dtype", choices=list(KV_BYTES), default="fp16")
    ap.add_argument("--min-vram-fraction", type=float,
                    default=float(os.environ.get("MIN_VRAM_FRACTION", "0.5")),
                    help="Drop models whose total VRAM is less than this "
                         "fraction of --vram (default 0.5). Reduces clutter "
                         "from variants too small to be worth the GPU. Set "
                         "to 0 to disable.")
    ap.add_argument("--download", action="store_true",
                    help="Pull bounded trial candidates that are not yet on disk")
    ap.add_argument("--max-downloads", type=int,
                    default=int(os.environ.get("DOWNLOAD_LIMIT", "1")),
                    help="Maximum trial candidates to pull per "
                         "(family, backend, ctx) cell (default "
                         "$DOWNLOAD_LIMIT or 1).")
    ap.add_argument("--download-report", type=Path,
                    help="Write names actually pulled, one per line")
    ap.add_argument("--prune", action="store_true",
                    help="Delete on-disk models that don't fit. Scoped by --family.")
    ap.add_argument("--prune-shadows", action="store_true",
                    help="Also delete Ollama tags on disk that aren't in the "
                         "full catalog (hand-made aliases that hold blobs alive).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan only — no downloads, no deletions")
    ap.add_argument("--verbose", action="store_true",
                    default=os.environ.get("VERBOSE", "").lower() in ("1", "true", "yes"),
                    help="Print every catalog entry, including too-small and "
                         "too-large rows. Default (off): only fitting models "
                         "are listed; suppressed rows are summarised in the "
                         "footer. Prunable on-disk rows are always shown.")
    args = ap.parse_args()

    if not CATALOG.is_file():
        sys.exit(f"error: {CATALOG} does not exist — run `make catalog-regen` first")
    cfg = yaml.safe_load(CATALOG.read_text()) or {}
    models = cfg.get("models", []) or []

    # --name: short-circuit pull-by-exact-name. Skips the matrix entirely.
    # Useful for HF-backed rows whose vLLM/SGLang fit-data is not yet in
    # the probe cache and would therefore be skipped by the trial-candidate
    # selector. Pulls the named row and exits.
    if args.name:
        match = [m for m in models if m.get("name") == args.name]
        if not match:
            sys.exit(f"error: no model named '{args.name}' in catalog")
        if not args.download:
            sys.exit("error: --name requires --download (the Makefile passes it)")
        if not args.dry_run:
            print(f"  → {args.name}  ({match[0].get('source', '?')})")
            pull(match[0])
        else:
            print(f"  --dry-run: would pull {args.name}")
        return

    if args.family:
        models = [m for m in models if m.get("family") == args.family]
    if not models:
        sys.exit(f"error: no models in catalog match family='{args.family}'")

    # Resolve the context list: --context overrides to single-ctx mode;
    # otherwise the matrix from --contexts is iterated.
    if args.context is not None:
        contexts = [args.context]
    else:
        contexts = parse_context_list(args.contexts)
    if not contexts:
        sys.exit("error: at least one context is required (--context or --contexts)")

    # Display ctx for the per-row table — when in matrix mode we pick
    # the largest context as the "primary" view; smaller contexts are
    # more inclusive (more rows fit) but the user usually cares about
    # the highest tier first.
    display_ctx = max(contexts)

    min_total = args.vram * max(0.0, args.min_vram_fraction)
    probe_cache = load_probe_cache()
    hf_caches = load_hf_probe_caches()

    trial_candidates, cell_index, rows_by_ctx = assign_cell_candidates(
        models, contexts, args.kv_dtype, min_total, args.vram,
        probe_cache, hf_caches, max_per_cell=args.max_downloads,
    )
    rows = rows_by_ctx[display_ctx]
    missing = [r for r in rows if r.fits and not r.downloaded]
    missing_bytes_gb = sum(parse_size_gb(r.model["size"]) for r in missing)
    candidate_bytes_gb = sum(parse_size_gb(r.model["size"]) for r in trial_candidates)
    # Prune candidates: on disk AND outside the [min_total, args.vram] window —
    # either too large to fit OR below the explicit MIN_VRAM_FRACTION floor.
    # Both ends represent models the user has opted out of at current settings.
    prunable = [r for r in rows
                if r.downloaded and (r.total_gb > args.vram
                                     or r.total_gb < min_total)]

    # ── Print plan ───────────────────────────────────────────────────────
    print()
    filter_note = f"family={args.family}  " if args.family else ""
    ctx_note = (f"context={args.context}" if args.context is not None
                else f"contexts=[{', '.join(context_label(c) for c in contexts)}]")
    print(f"  [select] {filter_note}vram={args.vram:g} GB  "
          f"min={min_total:.1f} GB ({args.min_vram_fraction:g}×)  "
          f"{ctx_note}  kv={args.kv_dtype}  "
          f"max_per_cell={args.max_downloads}"
          + ("  (dry-run)" if args.dry_run else ""))
    print()

    # Matrix view — one row per (family, backend), one column per context.
    print_cell_matrix(cell_index, contexts, args.vram, args.kv_dtype)

    if not args.verbose:
        # Verbose mode below adds the per-row table at display_ctx for debug.
        pass
    else:
        print(f"  Per-row view at ctx={context_label(display_ctx)}:")
        print()
    if args.verbose:
        print(f"  {'MODEL':<42s} {'SRC':<7s} {'WEIGHTS':>8s} {'SIZE':>8s} "
              f"{'CTX':>5s}  {'CAP':<11s} FIT  DISK  ACTION")
        print(f"  {'-'*120}")
        for r in rows:
            m = r.model
            size = m.get("size", "?")
            too_large = r.total_gb > args.vram
            too_small = r.total_gb < min_total
            fit_mark = "✓" if r.fits else ("↓" if too_small else "✗")
            disk_mark = "✓" if r.downloaded else "·"
            if too_large and r.downloaded:
                if args.prune:
                    action = "→ PRUNE (on disk, too large)"
                else:
                    action = "on disk but skipped (pass --prune to delete)"
            elif too_small and r.downloaded:
                if args.prune:
                    action = (f"→ PRUNE (on disk, too small for "
                              f"--min-vram-fraction={args.min_vram_fraction:g})")
                else:
                    action = (f"on disk but skipped, too small for floor "
                              f"({min_total:.0f} GB) — pass --prune to delete")
            elif too_large:
                action = "skip (too large)"
            elif too_small:
                action = (f"skip (too small, <{min_total:.0f} GB; lower "
                          f"--min-vram-fraction to include)")
            elif r.downloaded:
                v = r.model.get("vram") or {}
                if r.active_eligible:
                    action = "already on disk → active"
                elif is_ollama_only(r.model) and v.get("source") != "probe":
                    action = "on disk, needs probe before active"
                elif is_ollama_only(r.model) and not v.get("fully_on_gpu", True):
                    action = "on disk but skipped (probe showed CPU spill)"
                else:
                    action = "on disk but skipped"
            elif r.candidate and args.dry_run:
                cells_n = len(r.cells_filled)
                action = f"would download candidate (fills {cells_n} cell(s))"
            elif r.candidate and args.download:
                action = "→ DOWNLOAD (candidate)"
            elif r.candidate:
                action = "candidate (pass --download to pull)"
            elif not is_ollama_only(r.model):
                action = "skip (backend dormant)"
            elif args.dry_run or args.download:
                action = "missing, not selected for trial"
            else:
                action = "missing, not selected for trial"
            v = m.get("vram") or {}
            if v.get("source") in ("probe", "hf-probe"):
                vram_str = f"{r.total_gb:>6.1f}G"
                ctx_k = (v.get("context") or 0) // 1024
                ctx_str = f"{ctx_k}K"
            elif v.get("source") == "formula":
                vram_str = f"{r.total_gb:>5.1f}G*"
                ctx_k = (v.get("context") or 0) // 1024
                ctx_str = f"{ctx_k}K"
            else:
                vram_str = "—"
                ctx_str = "—"
            if v.get("source") in ("probe", "hf-probe"):
                cap = str(v.get("context_capability") or Capability.UNKNOWN)
                if not v.get("fully_on_gpu", True):
                    cap = Capability.ERROR
            else:
                cap = Capability.UNKNOWN
            print(f"  {m['name']:<42s} {m['source']:<7s} {size:>8s} "
                  f"{vram_str:>8s} {ctx_str:>5s}  {cap:<11s} "
                  f"{fit_mark:<3s}  {disk_mark:<4s}  {action}")
        print()
    fit_count = sum(1 for r in rows if r.fits)
    too_small_count = sum(1 for r in rows if r.total_gb < min_total)
    too_large_count = sum(1 for r in rows if r.total_gb > args.vram)
    on_disk_count = sum(1 for r in rows if r.fits and r.downloaded)
    active_count = sum(1 for r in rows if r.active_eligible)
    prunable_bytes = sum(reclaim_bytes(r.model) for r in prunable)
    prunable_gb = prunable_bytes / (1024 ** 3)
    cell_filled = sum(1 for v in cell_index.values() if v.get("status") == "filled")
    cell_candidate = sum(1 for v in cell_index.values() if v.get("status") == "candidate")
    cell_pending = sum(1 for v in cell_index.values() if v.get("status") == "pending_probe")
    cell_empty = sum(1 for v in cell_index.values() if v.get("status") == "no_options")

    print(f"  {fit_count} of {len(rows)} variants fit at ctx="
          f"{context_label(display_ctx)} in [{min_total:.1f}, {args.vram:g}] GB / "
          f"{args.kv_dtype} KV "
          f"(skipped: {too_small_count} too small, {too_large_count} too large).")
    print(f"  matrix: {len(cell_index)} cell(s)  ·  {cell_filled} filled  "
          f"·  {cell_candidate} candidate  ·  {cell_pending} pending probe  "
          f"·  {cell_empty} no option")
    print(f"  {active_count} active  ·  {on_disk_count} fitting on disk @ "
          f"{context_label(display_ctx)}  ·  "
          f"{len(missing)} fitting-but-missing (~{missing_bytes_gb:.1f} GB total).")
    print(f"  {len(trial_candidates)} trial download candidate(s) "
          f"(~{candidate_bytes_gb:.1f} GB; limit {args.max_downloads}/cell, "
          f"file-deduplicated).")
    if prunable:
        n_large = sum(1 for r in prunable if r.total_gb > args.vram)
        n_small = len(prunable) - n_large
        breakdown = []
        if n_large:
            breakdown.append(f"{n_large} too large")
        if n_small:
            breakdown.append(f"{n_small} too small")
        print(f"  {len(prunable)} on-disk but skipped "
              f"({', '.join(breakdown)}; "
              f"~{prunable_gb:.1f} GB reclaimable with --prune).")
    print()

    # Shadow aliases cross families (e.g. `nemotron70b` vs `nemotron`), so
    # always scan the full catalog regardless of --family. The active-set
    # write still respects --family; shadow pruning is a global operation.
    full_catalog = yaml.safe_load(CATALOG.read_text()).get("models", [])
    shadows = shadow_ollama_tags(full_catalog)
    if shadows:
        print(f"  {len(shadows)} shadow Ollama tag(s) not in catalog "
              f"(use --prune-shadows to delete): "
              f"{', '.join(shadows[:5])}"
              + (" ..." if len(shadows) > 5 else ""))
    orphan_gb = orphan_blob_gb()
    if orphan_gb > 0.1:
        print(f"  {orphan_gb:.1f} GB of unreferenced blobs on disk "
              f"(manual cleanup — see ollama blobs dir).")
    print()

    # ── Download only if requested ───────────────────────────────────────
    downloaded_names: list[str] = []
    if args.download_report and not args.dry_run:
        args.download_report.write_text("")
    if trial_candidates and args.download and not args.dry_run:
        print(f"  --download: pulling {len(trial_candidates)} trial candidate(s) ...")
        pulled: set[str] = set()
        for r in trial_candidates:
            name = r.model["name"]
            if name in pulled:
                continue
            pulled.add(name)
            print(f"\n  → {r.model['name']}  ({r.model['source']})")
            pull(r.model)
            r.downloaded = True
            downloaded_names.append(name)
        if args.download_report:
            args.download_report.write_text("\n".join(downloaded_names) + "\n")
        for r in rows:
            r.active_eligible = active_eligible(r)
        print()

    # ── Prune only if requested ──────────────────────────────────────────
    if prunable and args.prune and not args.dry_run:
        print(f"  --prune: deleting {len(prunable)} on-disk variant(s) "
              f"(~{prunable_gb:.1f} GB) ...")
        for r in prunable:
            print(f"\n  ✗ {r.model['name']}  ({r.model['source']})")
            delete(r.model)
            r.downloaded = False
        for r in rows:
            r.active_eligible = active_eligible(r)
        print()

    # ── Prune shadow Ollama tags if requested ────────────────────────────
    if shadows and args.prune_shadows and not args.dry_run:
        print(f"  --prune-shadows: deleting {len(shadows)} Ollama "
              f"tag(s) not in catalog ...")
        for name in shadows:
            print(f"  ✗ {name}")
            delete_ollama(name)
        print()

    # ── Eligible set summary ────────────────────────────────────────────
    # The probe cache is the source of truth; this is informational only.
    # Router and picker read deploy/.ollama-reasoning-cache.json directly.
    print_active_set(rows, args.vram, display_ctx)
    print()


if __name__ == "__main__":
    main()
