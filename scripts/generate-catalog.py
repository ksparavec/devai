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

def hf_weight_bytes(repo: str) -> int:
    """Sum all weight file sizes from the HF API (safetensors/bin/pth only)."""
    data = _http_json(f"{HF_API}/{repo}?blobs=true")
    return sum(
        (f.get("size") or 0)
        for f in data.get("siblings", [])
        if f.get("rfilename", "").endswith((".safetensors", ".bin", ".pth"))
    )


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

def ollama_tags(library: str) -> list[str]:
    """Scrape tag names from ollama.com (the /v2/<x>/tags/list endpoint
    is not publicly exposed and returns 404)."""
    import re
    html = _http_text(f"{OLLAMA_WEB}/{library}/tags")
    pattern = re.compile(rf"{re.escape(library)}:[A-Za-z0-9_.\-]+")
    seen: dict[str, None] = {}
    for m in pattern.findall(html):
        seen.setdefault(m.split(":", 1)[1], None)
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


def _gb(bytes_: int) -> float:
    return bytes_ / (1024 ** 3)


def _entry_hf(repo: str, family: str, fallback_arch: Arch,
              thinking: bool) -> Entry | None:
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
    return Entry(
        name=repo.split("/")[-1],
        family=family,
        backend=["vllm", "sglang"],
        repo=repo,
        size_gb=_gb(size_bytes),
        arch=arch,
        source_kind="hf",
        thinking=thinking,
    )


def _gguf_tag_token(filename: str) -> str:
    """Derive a stable Ollama tag suffix from a .gguf filename.

    `Qwen3.5-27B-UD-Q3_K_XL.gguf` → `ud-q3_k_xl`. We strip the model-name
    prefix (everything up to and including the size-in-B token) and the
    `.gguf` suffix, then lowercase. `_` and `-` are kept as-is. Falls
    back to a sanitised lowercase of the full stem when no size token
    is present.
    """
    import re
    stem = filename
    if stem.endswith(".gguf"):
        stem = stem[: -len(".gguf")]
    # Find the rightmost size-in-B token (e.g. "27B", "9B", "1.5B"). Quant
    # tokens always come AFTER it in canonical filenames.
    matches = list(re.finditer(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?B)(?=[-_]|$)", stem))
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
        print(f"\n── family: {name}  (thinking-hint={thinking})")
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
            e = _entry_hf(repo, name, fam_arch, thinking)
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
        if e.gguf_filename:
            lines.append(f'    gguf_filename: "{e.gguf_filename}"')
        lines.append(f'    size: "{e.size_gb:.2f} GB"')
        lines.append(f"    arch: {e.arch.to_yaml()}")
        lines.append(f'    arch_source: "{e.arch.source}"')
        lines.append(f'    purpose: "{purpose}"')
        # Note: a `thinking:` field used to be written here as a family-level
        # pre-probe hint, but no consumer reads it — capability is determined
        # at runtime by scripts/probe-ollama-reasoning.py and recorded in
        # active-models.yaml under `reasoning.capability`. Field removed to
        # avoid the impression that catalog metadata can override the probe.
        lines.append("")

    OUTPUT_YAML.write_text("\n".join(lines))
    print(f"\n  wrote {OUTPUT_YAML} with {len(all_entries)} entries")


if __name__ == "__main__":
    main()
