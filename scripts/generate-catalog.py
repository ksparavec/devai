#!/usr/bin/env python3
"""Regenerate deploy/models.yaml from upstream data.

Reads scripts/model-families.yaml (the only hand-maintained input),
queries HuggingFace Hub and Ollama registry for every variant in
every family, and writes deploy/models.yaml with:

    name       – repo basename (HF) or library:tag (Ollama)
    family     – family name from model-families.yaml
    backend    – [ollama] | [vllm, sglang]
    repo       – HF repo (HF entries only)
    size       – weights size in GB (from HF API weights files, or sum
                 of Ollama manifest layer sizes)
    arch       – { layers, kv_heads, head_dim, k_eq_v } from the
                 repo's own config.json if available, else from the
                 family's arch_ref config.json
    purpose    – short auto-generated description

No hand-entered sizes. No hand-entered architectures. Run again any
time; it is idempotent modulo upstream changes.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
FAMILIES_YAML = REPO_ROOT / "scripts" / "model-families.yaml"
OUTPUT_YAML = REPO_ROOT / "deploy" / "models.yaml"

HF_API = "https://huggingface.co/api/models"
HF_RAW = "https://huggingface.co"
OLLAMA_REGISTRY = "https://registry.ollama.ai/v2/library"
OLLAMA_WEB = "https://ollama.com/library"
USER_AGENT = "devai-catalog-generator/1.0"


# ── HTTP ─────────────────────────────────────────────────────────────────────

def _http_json(url: str, accept: str = "application/json", timeout: int = 25) -> dict:
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": accept}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _http_text(url: str, timeout: int = 25) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


# ── HuggingFace ──────────────────────────────────────────────────────────────

# Per-run memo of /api/models/{repo}?blobs=true responses. The HF API
# returns size-per-file AND the repo's `main` commit sha in one call;
# caching avoids duplicate fetches when both pieces are needed by
# downstream Entry construction.
_hf_blobs_cache: dict[str, dict] = {}


def _hf_blobs(repo: str) -> dict:
    if repo not in _hf_blobs_cache:
        _hf_blobs_cache[repo] = _http_json(f"{HF_API}/{repo}?blobs=true")
    return _hf_blobs_cache[repo]


_HF_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pth")

# Subdirectories some repos use to ship duplicate / alternate-format
# copies of the same weights. vLLM and SGLang load only the top-level
# safetensors set, so summing these would overstate the on-disk
# footprint by 2–4×. Examples:
#   openai/gpt-oss-20b:  ships MXFP4 at root + bf16 mirror under
#                         original/ + Apple-Metal copy under metal/.
#   meta-llama/*:        consolidated/ holds a single-file PyTorch
#                         checkpoint duplicating the safetensors set.
_HF_NON_PRIMARY_DIRS = ("original/", "metal/", "consolidated/")


def hf_weight_bytes(repo: str) -> int:
    """Sum the loaded-weight footprint reported by the HF API.

    Strategy:
      1. If `model.safetensors.index.json` is present at repo root,
         only count safetensors shards it references (authoritative —
         that's what the loader uses).
      2. Otherwise sum top-level weight files (suffixes in
         `_HF_WEIGHT_SUFFIXES`), excluding any path under a non-primary
         mirror directory (see `_HF_NON_PRIMARY_DIRS`).
    """
    data = _hf_blobs(repo)
    siblings = data.get("siblings", []) or []

    # Path 1: authoritative shard list from the safetensors index.
    index_files = [
        f for f in siblings
        if f.get("rfilename") == "model.safetensors.index.json"
    ]
    if index_files:
        try:
            import json as _json
            import urllib.request as _ur
            url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json"
            with _ur.urlopen(url, timeout=15) as r:
                index = _json.loads(r.read())
            shards = set((index.get("weight_map") or {}).values())
            if shards:
                return sum(
                    (f.get("size") or 0)
                    for f in siblings
                    if f.get("rfilename") in shards
                )
        except Exception:
            # Network or parse failure → fall through to path 2.
            pass

    # Path 2: top-level weight files, no mirror subdirs.
    total = 0
    for f in siblings:
        rfn = f.get("rfilename") or ""
        if not rfn.endswith(_HF_WEIGHT_SUFFIXES):
            continue
        if any(rfn.startswith(d) for d in _HF_NON_PRIMARY_DIRS):
            continue
        total += f.get("size") or 0
    return total


def hf_repo_sha(repo: str) -> str:
    """Return a stable short identifier for the repo's current `main`.

    Preferred: first 12 chars of the HF API's top-level `sha` (the git
    commit on the resolved revision). When the API response lacks a sha
    field — older proxies, mirror layers, certain private endpoints —
    fall back to a deterministic fingerprint over the weight-blob list
    so the cache key remains stable across runs of the same upstream
    state. The fingerprint is prefixed `f-` so it can never collide
    with a real (hex) sha.
    """
    import hashlib

    data = _hf_blobs(repo)
    sha = (data.get("sha") or "").strip()
    if sha:
        return sha[:12]
    weight_files = sorted(
        (f.get("rfilename", ""), int(f.get("size") or 0))
        for f in data.get("siblings", [])
        if f.get("rfilename", "").endswith(_HF_WEIGHT_SUFFIXES)
    )
    digest = hashlib.sha256(repr((repo, weight_files)).encode()).hexdigest()
    return "f-" + digest[:10]


def hf_config(repo: str) -> dict:
    return json.loads(_http_text(f"{HF_RAW}/{repo}/raw/main/config.json"))


def hf_gguf_files(repo: str) -> list[dict]:
    """List every .gguf file in an HF repo with its size in bytes.

    Returns [{"filename": "...gguf", "size_bytes": <int>}, ...]. Used by
    the gguf_repos source kind to enumerate one catalog row per quant.
    """
    data = _http_json(f"{HF_API}/{repo}?blobs=true")
    out: list[dict] = []
    for f in data.get("siblings", []):
        name = f.get("rfilename", "")
        if name.endswith(".gguf"):
            out.append({"filename": name, "size_bytes": int(f.get("size") or 0)})
    return out


# ── Ollama ───────────────────────────────────────────────────────────────────

# Quantization markers that distinguish a real model variant from a
# bare-size alias / `:latest` / `:cloud` placeholder. Tags that don't
# carry one of these tokens collapse onto the same blob as a sibling
# that does, so emitting them in the catalog produces shadow rows.
# Recognised: q[0-9] / iq[0-9] (llama.cpp k-quants), bf16 / fp16 / f16 /
# f32 (full-precision), mxfp[0-9] (Microsoft FP4/FP8), nvfp[0-9] (NVIDIA
# NVFP4/FP8), int[0-9] (INT4/INT8). Case-insensitive substring match.
_OLLAMA_QUANT_RE = re.compile(
    r"q[0-9]|iq[0-9]|bf16|fp16|f16|f32|mxfp[0-9]|nvfp[0-9]|int[0-9]",
    re.IGNORECASE,
)


def _is_aliased_ollama_tag(tag: str) -> bool:
    """True when the tag is a moving alias / placeholder rather than a
    real downloadable variant.

    Drops:
      - `latest`                     (moving alias)
      - any tag with no quantization marker (bare-size aliases like
        `9b`, `27b`, `e2b`, routing variants like `30b-instruct`,
        platform placeholders like `30b-cloud`, family-branch aliases
        like `phi4-reasoning:plus`)
    """
    if tag == "latest":
        return True
    return not _OLLAMA_QUANT_RE.search(tag)


def ollama_tags(library: str) -> list[str]:
    """Scrape tag names from ollama.com (the /v2/<x>/tags/list endpoint
    is not publicly exposed and returns 404).

    Filters out aliases / placeholders via `_is_aliased_ollama_tag` so
    the generated catalog never re-emits the bare-size and `:latest`
    tags we cleaned out of the local Ollama daemon.
    """
    html = _http_text(f"{OLLAMA_WEB}/{library}/tags")
    pattern = re.compile(rf"{re.escape(library)}:[A-Za-z0-9_.\-]+")
    seen: dict[str, None] = {}
    for m in pattern.findall(html):
        tag = m.split(":", 1)[1]
        if _is_aliased_ollama_tag(tag):
            continue
        seen.setdefault(tag, None)
    return list(seen)


_OLLAMA_ACCEPT = ",".join([
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.index.v1+json",
])


def ollama_manifest_size(library: str, tag: str) -> int:
    manifest = _http_json(
        f"{OLLAMA_REGISTRY}/{library}/manifests/{tag}", accept=_OLLAMA_ACCEPT
    )
    layers = manifest.get("layers") or []
    return sum(int(layer.get("size", 0)) for layer in layers)


# ── Architecture ─────────────────────────────────────────────────────────────

@dataclass
class Arch:
    layers: int
    kv_heads: int
    head_dim: int
    k_eq_v: bool
    source: str

    def to_yaml(self) -> str:
        keq = "true" if self.k_eq_v else "false"
        return (f"{{ layers: {self.layers}, kv_heads: {self.kv_heads}, "
                f"head_dim: {self.head_dim}, k_eq_v: {keq} }}")


def arch_from_config(cfg: dict, source: str) -> Arch:
    t = cfg.get("text_config", cfg)
    layers = int(t["num_hidden_layers"])
    kv_heads = int(t.get("num_key_value_heads", t["num_attention_heads"]))
    head_dim = int(
        t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
    )
    k_eq_v = bool(t.get("attention_k_eq_v", False))
    return Arch(layers, kv_heads, head_dim, k_eq_v, source)


# ── Entry construction ───────────────────────────────────────────────────────

@dataclass
class Entry:
    name: str
    family: str
    backend: list[str]
    repo: str | None
    size_gb: float
    arch: Arch
    source_kind: str  # "hf" | "ollama" | "gguf"
    thinking: bool = False  # family-level pre-probe hint; final capability
                            # is determined at runtime by
                            # scripts/probe-ollama-reasoning.py
    gguf_filename: str | None = None  # set on source_kind == "gguf"
    sha: str | None = None  # short git sha (12 chars) for source_kind == "hf";
                            # cache key for the vLLM/SGLang probe — `f-...`
                            # prefix indicates a fingerprint fallback
    parsers: dict | None = None  # curated vLLM/SGLang parser hints from the
                                 # family's `parsers:` block. Shape:
                                 # {"vllm": {"reasoning": ..., "tool": ...},
                                 #  "sglang": {"reasoning": ..., "tool": ...}}.
                                 # Sub-fields optional. None when the family
                                 # declared nothing — probers fall back to
                                 # inline/no-tool mode.
    conversational: bool | None = None  # presence of HF API tag
                                        # "conversational" — instruct/chat
                                        # tuned. None for non-HF rows.


def _gb(bytes_: int) -> float:
    return bytes_ / (1024 ** 3)


def _entry_hf(repo: str, family: str, fallback_arch: Arch,
              thinking: bool, parsers: dict | None) -> Entry | None:
    try:
        size_bytes = hf_weight_bytes(repo)
    except Exception as e:
        print(f"  [warn] HF size: {repo}: {e}", file=sys.stderr)
        return None
    if size_bytes == 0:
        print(f"  [warn] HF {repo}: no weight files — skipping", file=sys.stderr)
        return None
    try:
        arch = arch_from_config(hf_config(repo), f"{repo}/config.json")
    except Exception as e:
        print(f"  [info] HF {repo}: no usable config.json, "
              f"using family arch_ref ({e})", file=sys.stderr)
        arch = fallback_arch
    # sha capture shares the cached /api/models/{repo}?blobs=true response
    # with hf_weight_bytes — single network round-trip per repo.
    try:
        sha = hf_repo_sha(repo)
    except Exception as e:
        print(f"  [warn] HF sha: {repo}: {e}", file=sys.stderr)
        sha = None
    # The HF API tag set distinguishes instruct/chat models from base
    # ones via the `conversational` tag. More reliable than checking
    # tokenizer_config.json's chat_template (NVIDIA's NVFP4 quants
    # routinely strip it). Same cached response as size + sha — no
    # extra network round-trip.
    try:
        hf_tags = _hf_blobs(repo).get("tags") or []
        conversational = "conversational" in hf_tags
    except Exception as e:
        print(f"  [warn] HF tags: {repo}: {e}", file=sys.stderr)
        conversational = None
    return Entry(
        name=repo.split("/")[-1],
        family=family,
        backend=["vllm", "sglang"],
        repo=repo,
        size_gb=_gb(size_bytes),
        arch=arch,
        source_kind="hf",
        thinking=thinking,
        sha=sha,
        parsers=_normalize_parsers(parsers),
        conversational=conversational,
    )


def _normalize_parsers(parsers: dict | None) -> dict | None:
    """Validate and normalize the family's `parsers:` block.

    Accepts only the documented shape and known sub-keys; drops empty
    entries so the generated catalog row is minimal. Returns None when
    the input has no usable content.
    """
    if not parsers or not isinstance(parsers, dict):
        return None
    out: dict = {}
    for backend in ("vllm", "sglang"):
        block = parsers.get(backend)
        if not isinstance(block, dict):
            continue
        kept: dict = {}
        for key in ("reasoning", "tool"):
            val = block.get(key)
            if isinstance(val, str) and val.strip():
                kept[key] = val.strip()
        if kept:
            out[backend] = kept
    return out or None


def _gguf_tag_token(filename: str) -> str:
    """Derive a stable Ollama tag suffix from a .gguf filename.

    Strips everything up to and including the rightmost SIZE-in-B token,
    then lowercases. Recognises both plain `<digits>B` and
    `<letter><digits>B` shapes — the latter covers MoE notation like
    `A4B` / `A3B` (active-experts size). Without that, filenames like
    `gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf` would only strip up to `26B`,
    leaving `A4B` in the token and producing duplicated tag prefixes
    (`26b-a4b-a4b-it-...`).

    Examples:
      `Qwen3.5-27B-UD-Q3_K_XL.gguf`            → `ud-q3_k_xl`
      `Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf`        → `ud-q3_k_xl`
      `gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf`     → `it-ud-q3_k_xl`
      `NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q3_K_XL.gguf`
                                               → `reasoning-ud-q3_k_xl`
    """
    import re
    stem = filename
    if stem.endswith(".gguf"):
        stem = stem[: -len(".gguf")]
    # `[A-Z]?` makes the optional letter prefix recognise A4B / A3B / etc.
    matches = list(re.finditer(
        r"(?<![A-Za-z0-9])([A-Z]?\d+(?:\.\d+)?B)(?=[-_]|$)", stem,
    ))
    if matches:
        stem = stem[matches[-1].end():].lstrip("-_")
    return stem.lower() or "unspec"


def _entry_gguf(repo: str, file_meta: dict, family: str, arch: Arch,
                tag_prefix: str, thinking: bool) -> Entry | None:
    """One catalog row per .gguf file inside an HF repo.

    `tag_prefix` anchors the local Ollama tag (e.g. "27b") so the
    final name is `<family>:<tag_prefix>-<quant_token>`. Architecture
    comes from the family `arch_ref` — quantization doesn't change
    the architecture, so the same arch applies to every quant.
    """
    size_bytes = int(file_meta.get("size_bytes") or 0)
    if size_bytes == 0:
        print(f"  [warn] gguf {repo}/{file_meta.get('filename')}: zero size — "
              f"skipping", file=sys.stderr)
        return None
    quant_token = _gguf_tag_token(file_meta["filename"])
    tag = f"{tag_prefix}-{quant_token}" if tag_prefix else quant_token
    return Entry(
        name=f"{family}:{tag}",
        family=family,
        backend=["ollama"],          # GGUF runs on the ollama backend
        repo=repo,
        size_gb=_gb(size_bytes),
        arch=arch,
        source_kind="gguf",
        thinking=thinking,
        gguf_filename=file_meta["filename"],
    )


def _entry_ollama(library: str, tag: str, family: str, arch: Arch,
                  thinking: bool) -> Entry | None:
    try:
        size_bytes = ollama_manifest_size(library, tag)
    except urllib.error.HTTPError as e:
        if e.code == 412:
            return None  # platform-gated (macOS-only)
        print(f"  [warn] Ollama manifest {library}:{tag}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [warn] Ollama manifest {library}:{tag}: {e}", file=sys.stderr)
        return None
    if size_bytes == 0:
        return None
    return Entry(
        name=f"{library}:{tag}",
        family=family,
        backend=["ollama"],
        repo=None,
        size_gb=_gb(size_bytes),
        arch=arch,
        source_kind="ollama",
        thinking=thinking,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    with FAMILIES_YAML.open() as fh:
        fams = yaml.safe_load(fh)["families"]

    all_entries: list[Entry] = []
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()

    for fam in fams:
        name = fam["name"]
        thinking = bool(fam.get("thinking", False))
        parsers = fam.get("parsers")
        parser_label = "—" if not parsers else ",".join(sorted(parsers.keys()))
        print(f"\n── family: {name}  (thinking-hint={thinking}, "
              f"parsers={parser_label})")
        arch_ref = fam.get("arch_ref")
        if not arch_ref:
            print(f"  [error] family {name} has no arch_ref — skipping",
                  file=sys.stderr)
            continue
        print(f"  fetching arch from {arch_ref} ...")
        fam_arch = arch_from_config(hf_config(arch_ref),
                                    f"{arch_ref}/config.json")

        for repo in fam.get("hf_repos") or []:
            print(f"  HF: {repo} ... ", end="", flush=True)
            e = _entry_hf(repo, name, fam_arch, thinking, parsers)
            if e:
                print(f"{e.size_gb:.1f} GB  arch={e.arch.layers}L/"
                      f"{e.arch.kv_heads}kv/{e.arch.head_dim}h")
                all_entries.append(e)

        for lib in fam.get("ollama_repos") or []:
            try:
                tags = ollama_tags(lib)
            except Exception as e:
                print(f"  [warn] Ollama tags for {lib}: {e}", file=sys.stderr)
                tags = []
            skipped = 0
            for tag in tags:
                e = _entry_ollama(lib, tag, name, fam_arch, thinking)
                if e is None:
                    skipped += 1
                    continue
                all_entries.append(e)
                print(f"  ollama: {lib}:{tag} → {e.size_gb:.1f} GB")
            if skipped:
                print(f"  (skipped {skipped} tag(s): platform-gated or "
                      f"unavailable)")

        for spec in fam.get("gguf_repos") or []:
            repo = spec.get("repo")
            if not repo:
                print(f"  [warn] gguf_repos entry missing 'repo' — skipping",
                      file=sys.stderr)
                continue
            tag_prefix = (spec.get("tag_prefix") or "").strip("-_")
            # include semantics:
            #   key absent             → include every .gguf in the repo
            #   key present, list []   → include nothing (safe placeholder)
            #   key present, list[..] → only files whose lowered filename
            #                            contains any of these tokens
            include_raw = spec.get("include")
            include_filter: list[str] | None
            if "include" in spec:
                include_filter = [s.strip().lower() for s in (include_raw or []) if s]
            else:
                include_filter = None
            filter_label = (
                "all" if include_filter is None
                else (f"{include_filter}" if include_filter else "<empty — none emitted>")
            )
            print(f"  GGUF: {repo} (tag_prefix={tag_prefix or '<none>'}, "
                  f"include={filter_label}) ...")
            if include_filter == []:
                continue
            try:
                files = hf_gguf_files(repo)
            except Exception as e:
                print(f"  [warn] HF GGUF list {repo}: {e}", file=sys.stderr)
                continue
            kept = 0
            for fmeta in files:
                fname = fmeta["filename"]
                if include_filter is not None:
                    fname_lc = fname.lower()
                    if not any(token in fname_lc for token in include_filter):
                        continue
                e = _entry_gguf(repo, fmeta, name, fam_arch, tag_prefix, thinking)
                if e is None:
                    continue
                all_entries.append(e)
                print(f"    gguf: {e.name} → {e.size_gb:.1f} GB ({fname})")
                kept += 1
            if not kept:
                print(f"  [info] {repo}: filter matched 0 files (no entries)")

    # ── Write deploy/models.yaml ─────────────────────────────────────────
    lines: list[str] = []
    lines.append("# DevAI model catalog — AUTO-GENERATED.")
    lines.append("#")
    lines.append("# Regenerate with: make catalog-regen")
    lines.append("# Source of truth:  scripts/model-families.yaml")
    lines.append(f"# Generated at:     {now}")
    lines.append("#")
    lines.append("# Every 'size' and 'arch' below was fetched live from the")
    lines.append("# upstream provider at generation time. Do not hand-edit.")
    lines.append("")
    lines.append("models:")
    for e in all_entries:
        backend_inline = "[" + ", ".join(e.backend) + "]"
        # Auto-generated purpose string — not hand-entered. Downstream tools
        # (ollama-list, vllm-list, model-picker) expect this key.
        purpose = (f"{e.family} · {e.size_gb:.1f} GB · "
                   f"{e.arch.layers}L/{e.arch.kv_heads}kv/{e.arch.head_dim}h"
                   + (" · k=v" if e.arch.k_eq_v else ""))
        lines.append(f'  - name: "{e.name}"')
        lines.append(f"    family: {e.family}")
        lines.append(f"    backend: {backend_inline}")
        if e.repo:
            lines.append(f'    repo: "{e.repo}"')
        lines.append(f'    source: {e.source_kind}')
        if e.sha:
            lines.append(f'    sha: "{e.sha}"')
        if e.gguf_filename:
            lines.append(f'    gguf_filename: "{e.gguf_filename}"')
        lines.append(f'    size: "{e.size_gb:.2f} GB"')
        lines.append(f"    arch: {e.arch.to_yaml()}")
        lines.append(f'    arch_source: "{e.arch.source}"')
        lines.append(f'    purpose: "{purpose}"')
        # `conversational: true|false` from the HF API tag set —
        # consumed by the picker to label tuning style (IT vs BASE).
        # Omitted on non-HF rows (where it'd always be None).
        if e.conversational is not None:
            lines.append(f'    conversational: {str(e.conversational).lower()}')
        # Note: a `thinking:` field used to be written here as a family-level
        # pre-probe hint, but no consumer reads it — capability is determined
        # at runtime by scripts/probe-ollama-reasoning.py and recorded in
        # active-models.yaml under `reasoning.capability`. Field removed to
        # avoid the impression that catalog metadata can override the probe.
        if e.parsers:
            # Curated parser hints from scripts/model-families.yaml. The
            # probe drivers consume these to launch with the correct
            # --reasoning-parser / --tool-call-parser flags; the router
            # reads the probe-confirmed values back from the cache. Only
            # emitted for HF entries — Ollama handles parsing natively.
            lines.append("    parsers:")
            for backend in ("vllm", "sglang"):
                block = e.parsers.get(backend)
                if not block:
                    continue
                lines.append(f"      {backend}:")
                for key in ("reasoning", "tool"):
                    if key in block:
                        lines.append(f"        {key}: {block[key]}")
        lines.append("")

    OUTPUT_YAML.write_text("\n".join(lines))
    print(f"\n  wrote {OUTPUT_YAML} with {len(all_entries)} entries")


if __name__ == "__main__":
    main()
