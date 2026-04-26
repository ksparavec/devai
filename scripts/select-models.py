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
PROBE_CACHE = REPO_ROOT / "deploy" / ".ollama-reasoning-cache.json"

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

def load_probe_cache() -> dict:
    """Load reasoning capability probe results keyed by '<model>@<digest>'.

    Cache is written by scripts/probe-ollama-reasoning.py against the live
    ollama runtime. We index by short-form digest to match. If the file
    isn't present (probe never ran), every model gets capability=unknown.
    """
    if not PROBE_CACHE.is_file():
        return {}
    import json
    try:
        return json.loads(PROBE_CACHE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def lookup_probe(model_name: str, cache: dict) -> dict:
    """Return the cached probe record for a model name, or {} if absent."""
    for key, rec in cache.items():
        if key.startswith(model_name + "@") and rec.get("name") == model_name:
            return rec
    return {}


def lookup_capability(model_name: str, cache: dict) -> tuple[str, str | None]:
    """Return (capability, disable_verified-as-string-or-None) for an Ollama
    model name."""
    rec = lookup_probe(model_name, cache)
    if not rec:
        return "unknown", None
    cap = rec.get("capability", "unknown")
    disable = rec.get("disable_verified")
    if isinstance(disable, bool):
        return cap, "true" if disable else "false"
    return cap, None


def interpolate_total_gb(probe_rec: dict, context: int) -> tuple[float | None, int]:
    """Compute (total_gb, effective_context) from probe coefficients.

    effective_context = min(context, max_context). The model's design
    ceiling is a hard physical limit — a 128K-only model asked to run
    at 256K just runs at 128K with the corresponding VRAM. This is NOT
    a runtime clamp like OLLAMA_CONTEXT_LENGTH; it's a fact about the
    model's architecture.

    Returns (None, 0) when the probe lacks usable coefficients.
    """
    weights_gb = probe_rec.get("weights_overhead_gb")
    kv_pt = probe_rec.get("kv_per_token_bytes")
    if weights_gb is None or kv_pt is None:
        return None, 0
    max_ctx = probe_rec.get("max_context") or 0
    eff_ctx = min(context, max_ctx) if max_ctx else context
    kv_gb = (kv_pt * eff_ctx) / (1024**3)
    return round(weights_gb + kv_gb, 2), eff_ctx


def write_active(models: list[dict], vram: float, context: int, kv_dtype: str) -> None:
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    cache = load_probe_cache()
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
    lines.append("# Reasoning capability comes from a runtime probe of the live ollama")
    lines.append("# (scripts/probe-ollama-reasoning.py); for vLLM/SGLang it stays unknown")
    lines.append("# until those backends get their own probe.")
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
        # Top-level context: the router reads this as the per-model
        # MAX_CONTEXT_LEN override (gpu-arbiter computes launch params
        # from it). It's the same effective context recorded inside the
        # nested `vram` block — the duplicate is intentional: the router
        # only parses the top level, the picker only re-uses the nested
        # coefficients.
        v_for_ctx = m.get("vram") or {}
        eff_ctx = v_for_ctx.get("context")
        if eff_ctx:
            lines.append(f"    context: {eff_ctx}")
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
        # Reasoning capability — runtime-probed for ollama, unknown for others.
        if "ollama" in m.get("backend", []):
            cap, disable = lookup_capability(m["name"], cache)
            probe_rec = lookup_probe(m["name"], cache)
        else:
            cap, disable = "unknown", None
            probe_rec = {}
        parts = [f"capability: {cap}"]
        if disable is not None:
            parts.append(f"disable_verified: {disable}")
        lines.append(f"    reasoning: {{ {', '.join(parts)} }}")
        # MoE info — present only when the probe found expert fields in
        # /api/show. Dense models simply omit the block.
        ec = probe_rec.get("experts_total")
        eu = probe_rec.get("experts_used")
        if ec is not None and eu is not None:
            lines.append(f"    moe: {{ experts_total: {ec}, experts_used: {eu} }}")
        # Display details from /api/show — authoritative substitutes for
        # parsing the tag suffix. Picker renders these as columns.
        det_parts = []
        if probe_rec.get("param_size_label"):
            det_parts.append(f'param_size: "{probe_rec["param_size_label"]}"')
        if probe_rec.get("quantization"):
            det_parts.append(f'quantization: "{probe_rec["quantization"]}"')
        if det_parts:
            lines.append(f"    details: {{ {', '.join(det_parts)} }}")
        v = m.get("vram") or {}
        if v.get("source") == "probe":
            spill = "false" if v.get("fully_on_gpu", True) else "true"
            parts = [
                "source: probe",
                f"total_gb: {v['total_gb']}",
                f"vram_gb: {v.get('vram_gb', v['total_gb'])}",
                f"context: {v['context']}",
                f"spilled_to_cpu: {spill}",
            ]
            # Coefficients (when both points succeeded) let the picker
            # recompute total_gb at any user-chosen context without
            # re-running select-models.
            if v.get("weights_overhead_gb") is not None:
                parts.append(f"weights_overhead_gb: {v['weights_overhead_gb']}")
            if v.get("kv_per_token_bytes") is not None:
                parts.append(f"kv_per_token_bytes: {v['kv_per_token_bytes']}")
            if v.get("max_context"):
                parts.append(f"max_context: {v['max_context']}")
            lines.append(f"    vram: {{ {', '.join(parts)} }}")
        elif v:
            lines.append(
                f"    vram: {{ source: formula, "
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
    ap.add_argument("--min-vram-fraction", type=float,
                    default=float(os.environ.get("MIN_VRAM_FRACTION", "0.5")),
                    help="Drop models whose total VRAM is less than this "
                         "fraction of --vram (default 0.5). Reduces clutter "
                         "from variants too small to be worth the GPU. Set "
                         "to 0 to disable.")
    ap.add_argument("--download", action="store_true",
                    help="Also pull fitting models that are not yet on disk")
    ap.add_argument("--prune", action="store_true",
                    help="Delete on-disk models that don't fit. Scoped by --family.")
    ap.add_argument("--prune-shadows", action="store_true",
                    help="Also delete Ollama tags on disk that aren't in the "
                         "full catalog (hand-made aliases that hold blobs alive).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan only — no downloads, no deletions, no active-models.yaml write")
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
    if args.family:
        models = [m for m in models if m.get("family") == args.family]
    if not models:
        sys.exit(f"error: no models in catalog match family='{args.family}'")

    # Use the user's --context exactly for every backend. No clamping
    # to OLLAMA_CONTEXT_LENGTH or any other runtime cap — the planner's
    # job is to compute fit at the requested context, not to second-
    # guess it.
    min_total = args.vram * max(0.0, args.min_vram_fraction)
    probe_cache = load_probe_cache()

    rows: list[Row] = []
    for m in models:
        backends = m.get("backend", [])
        is_ollama_only = ("ollama" in backends
                          and "vllm" not in backends
                          and "sglang" not in backends)
        ctx = args.context

        # Prefer the live probe's coefficients when present — interpolate
        # to the user's chosen context. The formula is conservative;
        # observed allocation at the requested context is the truth.
        # vLLM/SGLang have no probe yet, so they keep using the formula.
        probe = lookup_probe(m["name"], probe_cache) if is_ollama_only else {}
        interp_total, eff_ctx = interpolate_total_gb(probe, ctx) if probe else (None, 0)
        if interp_total is not None:
            total = interp_total
            low = probe.get("actual_low") or {}
            breakdown = {
                "source": "probe",
                "total_gb": total,
                # vram_gb tracks the on-GPU portion. For in-budget models
                # everything stays on GPU, so total ≡ vram. The LOW probe's
                # fully_on_gpu flag reflects spill behaviour at small ctx —
                # not authoritative at the user's larger ctx, but the only
                # signal we have without re-probing.
                "vram_gb": total,
                "fully_on_gpu": low.get("fully_on_gpu", True),
                "context": eff_ctx,
                "weights_overhead_gb": probe.get("weights_overhead_gb"),
                "kv_per_token_bytes": probe.get("kv_per_token_bytes"),
                "max_context": probe.get("max_context"),
            }
        else:
            formula = vram_breakdown(m, ctx, args.kv_dtype)
            total = formula["total_gb"]
            breakdown = {**formula, "source": "formula"}
        m["vram"] = breakdown
        rows.append(Row(
            model=m,
            total_gb=total,
            fits=(min_total <= total <= args.vram),
            downloaded=is_downloaded(m),
        ))
    rows.sort(key=lambda r: (not r.fits, r.total_gb))

    missing = [r for r in rows if r.fits and not r.downloaded]
    missing_bytes_gb = sum(parse_size_gb(r.model["size"]) for r in missing)
    # Prune candidates: on disk AND outside the [min_total, args.vram] window —
    # either too large to fit OR below the explicit MIN_VRAM_FRACTION floor.
    # Both ends represent models the user has opted out of at current settings.
    prunable = [r for r in rows
                if r.downloaded and (r.total_gb > args.vram
                                     or r.total_gb < min_total)]
    # Default: active = fits AND on-disk. Downloads happen only if --download.
    active = [r.model for r in rows if r.fits and r.downloaded]

    # ── Print plan ───────────────────────────────────────────────────────
    print()
    filter_note = f"family={args.family}  " if args.family else ""
    print(f"  [select] {filter_note}vram={args.vram:g} GB  "
          f"min={min_total:.1f} GB ({args.min_vram_fraction:g}×)  "
          f"context={args.context}  kv={args.kv_dtype}"
          + ("  (dry-run)" if args.dry_run else ""))
    print()
    print(f"  {'MODEL':<42s} {'SRC':<7s} {'WEIGHTS':>8s} {'SIZE':>8s} "
          f"{'CTX':>5s}  {'CAP':<11s} FIT  DISK  ACTION")
    print(f"  {'-'*120}")
    suppressed_small = 0
    suppressed_large = 0
    for r in rows:
        m = r.model
        size = m.get("size", "?")
        too_large = r.total_gb > args.vram
        too_small = r.total_gb < min_total
        # Default view shows only fitting models. Non-fitting rows are
        # suppressed UNLESS verbose is on OR they're on-disk and prunable
        # (which is destructive — user must see what's being deleted).
        if not args.verbose and not r.fits:
            visible = r.downloaded and args.prune
            if not visible:
                if too_small:
                    suppressed_small += 1
                else:
                    suppressed_large += 1
                continue
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
            action = "already on disk → active"
        elif args.dry_run:
            action = "would download (use --download)"
        elif args.download:
            action = "→ DOWNLOAD"
        else:
            action = "missing (pass --download to pull)"
        v = m.get("vram") or {}
        if v.get("source") == "probe":
            vram_str = f"{r.total_gb:>6.1f}G"
            ctx_k = (v.get("context") or 0) // 1024
            ctx_str = f"{ctx_k}K"
        elif v.get("source") == "formula":
            vram_str = f"{r.total_gb:>5.1f}G*"     # * = estimated
            ctx_k = (v.get("context") or 0) // 1024
            ctx_str = f"{ctx_k}K"
        else:
            vram_str = "—"
            ctx_str = "—"
        # Capability is from the probe; for non-ollama or unprobed models
        # it's "unknown".
        if "ollama" in m.get("backend", []):
            cap, _ = lookup_capability(m["name"], probe_cache)
        else:
            cap = "unknown"
        print(f"  {m['name']:<42s} {m['source']:<7s} {size:>8s} "
              f"{vram_str:>8s} {ctx_str:>5s}  {cap:<11s} "
              f"{fit_mark:<3s}  {disk_mark:<4s}  {action}")
    print()
    fit_count = sum(1 for r in rows if r.fits)
    too_small_count = sum(1 for r in rows if r.total_gb < min_total)
    too_large_count = sum(1 for r in rows if r.total_gb > args.vram)
    on_disk_count = sum(1 for r in rows if r.fits and r.downloaded)
    prunable_bytes = sum(reclaim_bytes(r.model) for r in prunable)
    prunable_gb = prunable_bytes / (1024 ** 3)
    print(f"  {fit_count} of {len(rows)} variants fit in "
          f"[{min_total:.1f}, {args.vram:g}] GB "
          f"@ {args.context} ctx / {args.kv_dtype} KV  "
          f"(skipped: {too_small_count} too small, {too_large_count} too large).")
    hidden = suppressed_small + suppressed_large
    if hidden and not args.verbose:
        print(f"  {hidden} non-fitting rows hidden "
              f"({suppressed_small} too small, {suppressed_large} too large) "
              f"— run with VERBOSE=1 to see them.")
    print(f"  {on_disk_count} already on disk  ·  {len(missing)} missing "
          f"(~{missing_bytes_gb:.1f} GB if downloaded).")
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
