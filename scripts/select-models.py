#!/usr/bin/env python3
"""Select and optionally download models from deploy/models.yaml.

Reads the catalog produced by scripts/generate-catalog.py and filters
entries by VRAM / CONTEXT / KV dtype requirements. For each fitting
entry, checks whether it is already on disk; if not and --dry-run is
not set, downloads it. Always writes deploy/active-models.yaml with
the set of models that are both fitting AND downloaded — this is the
catalog the router reads at /etc/devai/models.yaml.

Usage:
    scripts/select-models.py                         # all families, 24 GB default
    scripts/select-models.py --family gemma4
    scripts/select-models.py --vram 48 --context 65536 --kv-dtype fp8
    scripts/select-models.py --dry-run               # plan only, no downloads

Re-runnable: new fitters added, stale entries removed from active set.

Errors (network, disk, subprocess) propagate verbatim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG = REPO_ROOT / "deploy" / "models.yaml"
ACTIVE = REPO_ROOT / "deploy" / "active-models.yaml"

# vLLM/SGLang serve models fully in VRAM — weights + full KV cache +
# CUDA graphs + activations. Ollama (llama.cpp) has a much smaller
# footprint: no CUDA graphs, KV cache is allocated lazily per-request,
# and it spills to CPU RAM when weights approach VRAM. We apply the
# strict accounting only to vLLM/SGLang.
VLLM_OVERHEAD_GB = 1.5       # CUDA graphs (~1) + activations (~0.5)
OLLAMA_OVERHEAD_GB = 0.5     # small buffer for runtime state

KV_BYTES = {"fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}

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
    """Return the per-component VRAM breakdown for this model's backend.

    vLLM/SGLang: strict — weights + full KV + CUDA graphs + activations.
    Ollama: lean — weights + small runtime buffer (llama.cpp handles KV
    allocation dynamically and can spill to CPU, so its GPU footprint
    is dominated by active weights, not worst-case KV)."""
    weight_gb = parse_size_gb(model["size"])
    backends = model.get("backend", [])
    if "ollama" in backends and "vllm" not in backends and "sglang" not in backends:
        return {
            "weights_gb": round(weight_gb, 2),
            "kv_gb": 0.0,
            "overhead_gb": OLLAMA_OVERHEAD_GB,
            "total_gb": round(weight_gb + OLLAMA_OVERHEAD_GB, 2),
            "context": context,
            "kv_dtype": kv_dtype,
        }
    arch = model.get("arch")
    if not arch:
        # No arch means we can't compute KV. Be conservative: assume
        # worst-case 256 KB/token (rough upper bound for 13B-class models).
        kv_gb = (256 * 1024 * context) / (1024 ** 3)
    else:
        kv_gb = (kv_per_token_bytes(arch, kv_dtype) * context) / (1024 ** 3)
    return {
        "weights_gb": round(weight_gb, 2),
        "kv_gb": round(kv_gb, 2),
        "overhead_gb": VLLM_OVERHEAD_GB,
        "total_gb": round(weight_gb + kv_gb + VLLM_OVERHEAD_GB, 2),
        "context": context,
        "kv_dtype": kv_dtype,
    }


def estimate_total_gb(model: dict, context: int, kv_dtype: str) -> float:
    """Back-compat wrapper around vram_breakdown."""
    return vram_breakdown(model, context, kv_dtype)["total_gb"]


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


def pull(model: dict) -> None:
    if model["source"] == "ollama":
        pull_ollama(model["name"])
    elif model["source"] == "hf":
        pull_hf(model["name"], model["repo"])
    else:
        sys.exit(f"error: unknown source '{model.get('source')}' for {model['name']}")


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

def write_active(models: list[dict], vram: float, context: int, kv_dtype: str) -> None:
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    lines: list[str] = []
    lines.append("# Active model catalog — AUTO-GENERATED by scripts/select-models.py.")
    lines.append("#")
    lines.append(f"# Selection criteria: VRAM={vram:g} GB  CONTEXT={context}"
                 f"  KV={kv_dtype}")
    lines.append(f"# Generated at:       {now}")
    lines.append("#")
    lines.append("# Only entries below are served by the ollama/vllm/sglang proxies.")
    lines.append("# Regenerate: make model-select [FAMILY=...] [VRAM=...] [CONTEXT=...] [KV=...]")
    lines.append("# Full catalog lives in deploy/models.yaml.")
    lines.append("")
    lines.append("models:")
    if not models:
        lines.append("  []")
    for m in models:
        backend_inline = "[" + ", ".join(m["backend"]) + "]"
        lines.append(f'  - name: "{m["name"]}"')
        lines.append(f"    family: {m.get('family', '')}")
        lines.append(f"    backend: {backend_inline}")
        if m.get("repo"):
            lines.append(f'    repo: "{m["repo"]}"')
        lines.append(f'    source: {m["source"]}')
        lines.append(f'    size: "{m["size"]}"')
        arch = m.get("arch") or {}
        if arch:
            keq = "true" if arch.get("k_eq_v") else "false"
            lines.append(
                f"    arch: {{ layers: {arch['layers']}, "
                f"kv_heads: {arch['kv_heads']}, "
                f"head_dim: {arch['head_dim']}, k_eq_v: {keq} }}"
            )
        if m.get("purpose"):
            lines.append(f'    purpose: "{m["purpose"]}"')
        v = m.get("vram") or {}
        if v:
            lines.append(
                f"    vram: {{ "
                f"weights_gb: {v['weights_gb']}, "
                f"kv_gb: {v['kv_gb']}, "
                f"overhead_gb: {v['overhead_gb']}, "
                f"total_gb: {v['total_gb']}, "
                f"context: {v['context']}, "
                f"kv_dtype: {v['kv_dtype']} }}"
            )
        lines.append("")
    ACTIVE.write_text("\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────────────────

@dataclass
class Row:
    model: dict
    total_gb: float
    fits: bool
    downloaded: bool


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", default="")
    ap.add_argument("--vram", type=float,
                    default=float(os.environ.get("GPU_MEMORY_GB", "24")))
    ap.add_argument("--context", type=int,
                    default=int(os.environ.get("MAX_CONTEXT_LEN", "131072")))
    ap.add_argument("--kv-dtype", choices=list(KV_BYTES), default="fp16")
    ap.add_argument("--download", action="store_true",
                    help="Also pull fitting models that are not yet on disk")
    ap.add_argument("--prune", action="store_true",
                    help="Delete on-disk models that don't fit. Scoped by --family.")
    ap.add_argument("--prune-shadows", action="store_true",
                    help="Also delete Ollama tags on disk that aren't in the "
                         "full catalog (hand-made aliases that hold blobs alive).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan only — no downloads, no deletions, no active-models.yaml write")
    args = ap.parse_args()

    if not CATALOG.is_file():
        sys.exit(f"error: {CATALOG} does not exist — run `make catalog-regen` first")
    cfg = yaml.safe_load(CATALOG.read_text()) or {}
    models = cfg.get("models", []) or []
    if args.family:
        models = [m for m in models if m.get("family") == args.family]
    if not models:
        sys.exit(f"error: no models in catalog match family='{args.family}'")

    rows: list[Row] = []
    for m in models:
        breakdown = vram_breakdown(m, args.context, args.kv_dtype)
        m["vram"] = breakdown            # persist into catalog row for write_active
        rows.append(Row(
            model=m,
            total_gb=breakdown["total_gb"],
            fits=breakdown["total_gb"] <= args.vram,
            downloaded=is_downloaded(m),
        ))
    rows.sort(key=lambda r: (not r.fits, r.total_gb))

    missing = [r for r in rows if r.fits and not r.downloaded]
    missing_bytes_gb = sum(parse_size_gb(r.model["size"]) for r in missing)
    # Prune candidates: on disk AND doesn't fit (within current filter scope).
    prunable = [r for r in rows if not r.fits and r.downloaded]
    # Default: active = fits AND on-disk. Downloads happen only if --download.
    active = [r.model for r in rows if r.fits and r.downloaded]

    # ── Print plan ───────────────────────────────────────────────────────
    print()
    filter_note = f"family={args.family}  " if args.family else ""
    print(f"  [select] {filter_note}vram={args.vram:g} GB  "
          f"context={args.context}  kv={args.kv_dtype}"
          + ("  (dry-run)" if args.dry_run else ""))
    print()
    print(f"  {'MODEL':<48s} {'SRC':<7s} {'SIZE':>8s} {'TOTAL':>9s}"
          f"  FIT  ON-DISK  ACTION")
    print(f"  {'-'*110}")
    for r in rows:
        m = r.model
        size = m.get("size", "?")
        fit_mark = "✓" if r.fits else "✗"
        disk_mark = "✓" if r.downloaded else "·"
        if not r.fits and r.downloaded:
            if args.prune:
                action = "→ PRUNE (on disk, too large)"
            else:
                action = "on disk but skipped (pass --prune to delete)"
        elif not r.fits:
            action = "skip (too large)"
        elif r.downloaded:
            action = "already on disk → active"
        elif args.dry_run:
            action = "would download (use --download)"
        elif args.download:
            action = "→ DOWNLOAD"
        else:
            action = "missing (pass --download to pull)"
        print(f"  {m['name']:<48s} {m['source']:<7s} {size:>8s} "
              f"{r.total_gb:>7.1f}G  {fit_mark:<3s}  "
              f"{disk_mark:<7s}  {action}")
    print()
    fit_count = sum(1 for r in rows if r.fits)
    on_disk_count = sum(1 for r in rows if r.fits and r.downloaded)
    prunable_bytes = sum(reclaim_bytes(r.model) for r in prunable)
    prunable_gb = prunable_bytes / (1024 ** 3)
    print(f"  {fit_count} of {len(rows)} variants fit in {args.vram:g} GB "
          f"@ {args.context} ctx / {args.kv_dtype} KV.")
    print(f"  {on_disk_count} already on disk  ·  {len(missing)} missing "
          f"(~{missing_bytes_gb:.1f} GB if downloaded).")
    if prunable:
        print(f"  {len(prunable)} on-disk but too large "
              f"(~{prunable_gb:.1f} GB reclaimable with --prune).")

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
    if missing and args.download and not args.dry_run:
        print(f"  --download: pulling {len(missing)} missing variant(s) ...")
        for r in missing:
            print(f"\n  → {r.model['name']}  ({r.model['source']})")
            pull(r.model)
            r.downloaded = True
        active = [r.model for r in rows if r.fits and r.downloaded]
        print()

    # ── Prune only if requested ──────────────────────────────────────────
    if prunable and args.prune and not args.dry_run:
        print(f"  --prune: deleting {len(prunable)} on-disk variant(s) "
              f"(~{prunable_gb:.1f} GB) ...")
        for r in prunable:
            print(f"\n  ✗ {r.model['name']}  ({r.model['source']})")
            delete(r.model)
            r.downloaded = False
        print()

    # ── Prune shadow Ollama tags if requested ────────────────────────────
    if shadows and args.prune_shadows and not args.dry_run:
        print(f"  --prune-shadows: deleting {len(shadows)} Ollama "
              f"tag(s) not in catalog ...")
        for name in shadows:
            print(f"  ✗ {name}")
            delete_ollama(name)
        print()

    # ── Write active catalog ─────────────────────────────────────────────
    if args.dry_run:
        print(f"  (dry-run: {ACTIVE.name} NOT written)")
    else:
        write_active(active, args.vram, args.context, args.kv_dtype)
        print(f"  wrote {ACTIVE}  ({len(active)} active entries)")
    print()


if __name__ == "__main__":
    main()
