#!/usr/bin/env python3
"""Two-step picker for DevAI: pick a downloaded model, then pick an agent.

Discovery is disk-only — the picker walks the host model cache that is
mounted into the container and lists exactly what is on disk. The yaml
catalog (deploy/models.yaml), active catalog, and Ollama probe cache are
used only to enrich display metadata and rank the best rows per family,
context, and reasoning/offload label.

Layout scanned:
    OLLAMA_MANIFESTS_DIR/<library>/<tag>          → ollama tag <library>:<tag>
    VLLM_MODELS_DIR/<name>/config.json            → HF model <name>

Usage:
    model-picker                  Step 1: model · Step 2: agent
    model-picker --agent claude   Pre-select agent, only show step 1
    model-picker --agent gemini   Shortcut: launch gemini directly
    model-picker --agent bash     Shortcut: drop to bash immediately

Errors propagate verbatim. No exception swallowing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


# ── Configuration ────────────────────────────────────────────────────────────

_CATALOG_PATHS = [
    "/etc/devai/models.yaml",
    str(Path(__file__).resolve().parent.parent / "deploy" / "models.yaml"),
]

# Active catalog with precomputed VRAM breakdown (written by select-models.py).
# We read this for display only — never to decide what's downloaded.
_ACTIVE_CATALOG_PATHS = [
    "/etc/devai/active-models.yaml",
    str(Path(__file__).resolve().parent.parent / "deploy" / "active-models.yaml"),
]

_PROBE_CACHE_PATHS = [
    "/etc/devai/.ollama-reasoning-cache.json",
    str(Path(__file__).resolve().parent.parent / "deploy" / ".ollama-reasoning-cache.json"),
]

_ROUTER = os.environ.get("DEVAI_ROUTER_HOST", "devai-router")
_VLLM_DIR = os.environ.get("VLLM_MODELS_DIR", "/var/cache/devai/ollama/models/vllm")
_OLLAMA_MANIFESTS = os.environ.get(
    "OLLAMA_MANIFESTS_DIR",
    "/var/cache/devai/ollama/models/manifests/registry.ollama.ai/library",
)
# Default backend per source. Ollama-source models are ollama-only.
# HF-source models can run on vllm or sglang; we pick vllm by default.
# Override via env DEVAI_HF_BACKEND=sglang.
_HF_BACKEND = os.environ.get("DEVAI_HF_BACKEND", "vllm")

def _parse_context_value(raw: str) -> int:
    value = raw.strip().lower()
    if value.endswith("k"):
        return int(value[:-1]) * 1024
    return int(value)


def _parse_context_list(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        ctx = _parse_context_value(value)
        if ctx > 0 and ctx not in out:
            out.append(ctx)
    return out


# The model step always shows the standard context tiers. The selected row
# stores its tier in CONTEXT before launching the agent, so wrappers that
# honor CONTEXT can use the same budget the picker displayed.
_CONTEXT_CHOICES = _parse_context_list(
    os.environ.get("PICKER_CONTEXTS", "32768,65536,131072,262144")
)
_DEFAULT_CONTEXT = _parse_context_value(
    os.environ.get("CONTEXT", os.environ.get("MAX_CONTEXT_LEN", "131072"))
)
if not _CONTEXT_CHOICES:
    _CONTEXT_CHOICES = [_DEFAULT_CONTEXT]

# VRAM budget for in-budget filtering. Models whose interpolated total
# exceeds this are dropped (they only landed in active-models.yaml under
# a different VRAM/CONTEXT pairing). Defaults to GPU_MEMORY_GB so picker
# defaults match select-models defaults.
_VRAM_BUDGET = float(os.environ.get("VRAM", os.environ.get("GPU_MEMORY_GB", "24")))

#                     label        reason                                              port
_BACKENDS: dict[str, tuple[str, str, int]] = {
    "ollama": ("Ollama", "GGUF quantized — wide compatibility, CPU+GPU",              11434),
    "vllm":   ("vLLM",   "NVFP4 tensor cores — high throughput, paged attention",     11435),
    "sglang": ("SGLang", "NVFP4 tensor cores — RadixAttention, multi-turn optimized", 11436),
}

#                  id             display name          description
_AGENTS: list[tuple[str, str, str]] = [
    ("claude",      "Claude Code",       "AI coding assistant with agentic terminal"),
    ("aider",       "Aider",             "Git-aware pair programming"),
    ("codex",       "Codex",             "OpenAI terminal coding agent"),
    ("late",        "LATE",              "Lightweight AI Terminal Environment — ephemeral subagents"),
    ("interpreter", "Open Interpreter",  "Natural language computer control"),
]

# ANSI helpers
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


# ── Catalog (display metadata only) ──────────────────────────────────────────

def _load_catalog() -> dict[str, dict]:
    """Optional metadata lookup. Returns {} if no yaml found — discovery
    still works, rows just lose declared size / purpose / arch / family /
    reasoning capability."""
    for path in _CATALOG_PATHS:
        if os.path.isfile(path):
            with open(path) as fh:
                data = yaml.safe_load(fh) or {}
            return {m["name"]: m for m in data.get("models", [])}
    return {}


def _load_active_entries() -> dict[str, dict]:
    """Lookup of name → full active-models.yaml entry (vram, reasoning, …).

    Written by scripts/select-models.py. Returns {} if not mounted.
    """
    for path in _ACTIVE_CATALOG_PATHS:
        if os.path.isfile(path):
            with open(path) as fh:
                data = yaml.safe_load(fh) or {}
            return {m["name"]: m for m in data.get("models", []) or []
                    if isinstance(m, dict)}
    return {}


def _load_probe_records() -> dict[str, dict]:
    """Lookup of model name → v2 digest entry from the probe cache.

    Schema v2 keys are digests; each entry carries an `aliases` list and
    a `probes` map keyed by stringified context. Every alias resolves to
    the same digest entry here so a `name`-based lookup downstream stays
    one-call. Legacy v1 caches (name@digest keys) are migrated in-memory
    on the fly using the prober's helper — the on-disk file is left as-is.
    """
    for path in _PROBE_CACHE_PATHS:
        if not os.path.isfile(path):
            continue
        with open(path) as fh:
            data = json.load(fh) or {}
        if not data:
            return {}
        if not all(
            isinstance(v, dict) and v.get("schema_version") == 2
            for v in data.values()
        ):
            data = _migrate_in_memory(data)
        records: dict[str, dict] = {}
        for digest, entry in data.items():
            if not isinstance(entry, dict):
                continue
            for alias in entry.get("aliases") or []:
                records[alias] = entry
        return records
    return {}


def _migrate_in_memory(raw: dict) -> dict:
    """Run probe-ollama-reasoning's v1→v2 conversion without touching disk."""
    import importlib.util
    import sys
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    spec = importlib.util.spec_from_file_location(
        "_probe_module", here / "probe-ollama-reasoning.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    migrated, _ = mod.ensure_v2(raw)
    return migrated


# ── Disk discovery ───────────────────────────────────────────────────────────

def _ollama_disk_size_gb(library: str, tag: str) -> float:
    """Read ollama manifest, sum layer sizes."""
    manifest = Path(_OLLAMA_MANIFESTS) / library / tag
    try:
        data = json.loads(manifest.read_text())
        total = sum(int(L.get("size", 0)) for L in data.get("layers", []))
        return total / (1024 ** 3)
    except (OSError, ValueError, KeyError):
        return 0.0


def _hf_disk_size_gb(model_dir: Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(model_dir):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / (1024 ** 3)


def _details_from_probe(probe: dict) -> dict:
    details: dict = {}
    if probe.get("param_size_label"):
        details["param_size"] = probe["param_size_label"]
    if probe.get("quantization"):
        details["quantization"] = probe["quantization"]
    return details


def _moe_from_probe(probe: dict) -> dict:
    if probe.get("experts_total") is None or probe.get("experts_used") is None:
        return {}
    return {
        "experts_total": probe["experts_total"],
        "experts_used": probe["experts_used"],
    }


def _vram_from_probe(probe: dict) -> dict:
    """Pass through the v2 entry's per-context probes for the picker.

    `_vram_at` reads `probes[str(min(ctx, max_context))]` from this dict.
    """
    if not probe:
        return {}
    out: dict = {"source": "probe"}
    if probe.get("max_context") is not None:
        out["max_context"] = probe["max_context"]
    if probe.get("probes"):
        out["probes"] = probe["probes"]
    return out


def _capability_from_probe(probe: dict, fallback: str = "unknown") -> str:
    """Read the canonical capability from a v2 entry.

    Probe driver records the capability of the smallest fitting tier as
    the canonical (top-level) value, so a model that's structured at 32K
    but spills at 256K stays labelled `structured` here. Per-tier
    capability is read directly from `probes[ctx].capability` when the
    picker needs to filter by tier-specific behaviour.
    """
    if not probe:
        return fallback
    return str(probe.get("capability") or fallback)


def _discover_models() -> list[dict]:
    """Walk the cache dirs and return one entry per model on disk.

    Disk is the source of truth for existence. Catalog metadata (purpose,
    declared size) and per-model active data (VRAM breakdown + runtime-
    probed reasoning capability) are layered on top when available — never
    recomputed here.
    """
    catalog = _load_catalog()
    active = _load_active_entries()
    probes = _load_probe_records()
    out: list[dict] = []

    # Ollama: <library>/<tag> manifest files
    base = Path(_OLLAMA_MANIFESTS)
    if base.is_dir():
        for lib_dir in sorted(base.iterdir()):
            if not lib_dir.is_dir():
                continue
            for tag_file in sorted(lib_dir.iterdir()):
                if not tag_file.is_file():
                    continue
                name = f"{lib_dir.name}:{tag_file.name}"
                meta = catalog.get(name, {})
                disk_gb = _ollama_disk_size_gb(lib_dir.name, tag_file.name)
                act = active.get(name) or {}
                probe = probes.get(name) or {}
                active_cap = (act.get("reasoning") or {}).get("capability", "unknown")
                cap = _capability_from_probe(probe, active_cap)
                active_details = act.get("details") or {}
                probe_details = _details_from_probe(probe)
                out.append({
                    "name": name,
                    "source": "ollama",
                    "backend": "ollama",
                    "size": meta.get("size") or f"{disk_gb:.2f} GB",
                    "purpose": meta.get("purpose", ""),
                    "vram": _vram_from_probe(probe) or act.get("vram"),
                    "family": meta.get("family", ""),
                    "capability": cap,
                    "probe": probe,
                    "moe": _moe_from_probe(probe) or act.get("moe"),
                    "details": probe_details or active_details,
                })

    # HF/vLLM/SGLang: <name>/config.json
    vbase = Path(_VLLM_DIR)
    if vbase.is_dir():
        for d in sorted(vbase.iterdir()):
            if not (d.is_dir() and (d / "config.json").is_file()):
                continue
            name = d.name
            meta = catalog.get(name, {})
            disk_gb = _hf_disk_size_gb(d)
            # HF models can run on vllm or sglang; default per env.
            backend = _HF_BACKEND if _HF_BACKEND in ("vllm", "sglang") else "vllm"
            act = active.get(name) or {}
            cap = (act.get("reasoning") or {}).get("capability", "unknown")
            out.append({
                "name": name,
                "source": "hf",
                "backend": backend,
                "size": meta.get("size") or f"{disk_gb:.2f} GB",
                "purpose": meta.get("purpose", ""),
                "vram": act.get("vram"),
                "family": meta.get("family", ""),
                "capability": cap,
                "moe": act.get("moe"),
            })

    return out


def _check_cache_visible() -> None:
    """Bail early with a clear message if the cache mount is missing."""
    if not (os.path.isdir(_OLLAMA_MANIFESTS) or os.path.isdir(_VLLM_DIR)):
        sys.exit(
            f"error: model cache not visible inside the container.\n"
            f"  Looked for: {_OLLAMA_MANIFESTS}\n"
            f"          and {_VLLM_DIR}\n"
            f"  Re-run via `make shell-gpu` / `make lab-gpu` (or equivalent CPU "
            f"target) so the host cache dir is mounted read-only."
        )


# ── fzf wrapper ──────────────────────────────────────────────────────────────

def _fzf(lines: list[str], header: str,
         selectable: list[bool] | None = None) -> int | None:
    """Run fzf; return selected line index or None on cancel.

    `selectable[i]=False` marks line i as a non-selectable section header.
    Index sentinel `--` is used for those; if user lands on one fzf still
    returns it but we re-prompt.
    """
    if selectable is None:
        selectable = [True] * len(lines)
    indexed = []
    for i, line in enumerate(lines):
        tag = str(i) if selectable[i] else "--"
        indexed.append(f"{tag}\t{line}")
    while True:
        try:
            result = subprocess.run(
                [
                    "fzf", "--reverse", "--no-sort", "--ansi", "--no-info",
                    "--delimiter", "\t", "--with-nth", "2..",
                    "--header", header,
                    "--pointer", "▶", "--prompt", "  ",
                    "--margin", "1,2",
                    "--color", "pointer:3,header:4,hl:3,hl+:3,gutter:-1",
                ],
                input="\n".join(indexed),
                stdout=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            return _numbered_fallback(lines, header, selectable)
        if result.returncode != 0:
            return None
        tag = result.stdout.strip().split("\t", 1)[0]
        if tag == "--":
            # User landed on a section header — re-prompt.
            continue
        return int(tag)


def _numbered_fallback(lines: list[str], header: str,
                       selectable: list[bool] | None = None) -> int | None:
    if selectable is None:
        selectable = [True] * len(lines)
    print(f"\n  {header}\n")
    nums = [None] * len(lines)
    n = 0
    for i, line in enumerate(lines):
        if selectable[i]:
            n += 1
            nums[i] = n
            print(f"  [{n:>2}] {line}")
        else:
            print(f"       {line}")
    print()
    pick = int(input(f"  Select [1-{n}]: "))
    for i, k in enumerate(nums):
        if k == pick:
            return i
    return None


# ── Row formatters ───────────────────────────────────────────────────────────

_CAP_GLYPH = {
    "structured":  "●",  # clean reasoning
    "inline":      "◐",  # reasoning leaks into content
    "unsupported": "·",  # no reasoning
    "unknown":     "?",  # not yet probed
    "error":       "✗",  # probe failed
}

_REASONING_LABEL = {
    "structured": "Native reasoning",
    "inline": "Inline reasoning",
    "unsupported": "No reasoning",
}

_STATUS_ORDER = {
    "native": 0,
    "inline": 1,
    "none": 2,
    "cpu_offload": 3,
}


def _context_label(context: int) -> str:
    return f"{context // 1024}K" if context >= 1024 else str(context)


def _vram_at(v: dict, context: int) -> tuple[float | None, int]:
    """Compute (total_gb, effective_context) for the model at CONTEXT.

    effective_context = min(context, max_context). Probe-backed entries
    return the measurement at that exact tier; when no probe was recorded
    there (e.g., the user picked a non-standard context), the picker
    treats the row as ineligible — there is no interpolation any more.
    Formula entries (vLLM/SGLang dormant path) recompute KV linearly.
    Returns (None, 0) when no usable data is present.
    """
    if not v:
        return None, 0
    exact = _measured_point_at(v, context)
    if exact:
        return round(float(exact.get("actual_total_gb") or 0), 2), int(
            exact.get("ctx") or exact.get("actual_context") or context
        )
    if v.get("source") == "formula" and v.get("total_gb") is not None:
        base_ctx = int(v.get("context") or context)
        if base_ctx <= 0:
            return round(float(v["total_gb"]), 2), context
        max_ctx = v.get("max_context") or 0
        eff_ctx = min(context, max_ctx) if max_ctx else context
        kv_gb = float(v.get("kv_gb") or 0)
        static_gb = float(v["total_gb"]) - kv_gb
        total = static_gb + (kv_gb * eff_ctx / base_ctx)
        return round(total, 2), eff_ctx
    return None, 0


def _measured_point_at(v: dict, context: int) -> dict:
    """Return the v2 probe record at min(context, max_context), or {}.

    Two shapes are accepted:
      - Cache shape: v["probes"] is a dict keyed by str(ctx). We pick the
        record at the effective context.
      - Active-models.yaml shape: v carries a single measurement
        (total_gb, vram_gb, context, spilled_to_cpu) as a flat dict. This
        is the picker's fallback when the probe cache isn't mounted; we
        treat it as a measured point at v["context"].
    """
    max_ctx = int(v.get("max_context") or 0)
    eff_ctx = min(context, max_ctx) if max_ctx else context
    probes = v.get("probes")
    if isinstance(probes, dict) and probes:
        rec = probes.get(str(eff_ctx))
        if isinstance(rec, dict):
            return rec
    if v.get("source") == "probe" and v.get("total_gb") is not None:
        flat_ctx = int(v.get("context") or 0)
        if flat_ctx == eff_ctx:
            return {
                "ctx": flat_ctx,
                "actual_total_gb": float(v["total_gb"]),
                "actual_vram_gb": float(v.get("vram_gb") or v["total_gb"]),
                "fully_on_gpu": not bool(v.get("spilled_to_cpu", False)),
                "actual_context": flat_ctx,
            }
    return {}


def _vram_info_at(v: dict, context: int) -> dict | None:
    exact = _measured_point_at(v, context)
    if exact:
        total = round(float(exact["actual_total_gb"]), 2)
        vram_gb = round(float(exact.get("actual_vram_gb") or total), 2)
        return {
            "total_gb": total,
            "vram_gb": vram_gb,
            "context": int(exact.get("actual_context") or context),
            "fully_on_gpu": bool(exact.get("fully_on_gpu", True))
            and total <= _VRAM_BUDGET,
            "over_budget": total > _VRAM_BUDGET,
            "measured": True,
        }
    total, eff_ctx = _vram_at(v, context)
    if total is None:
        return None
    return {
        "total_gb": total,
        "vram_gb": min(total, _VRAM_BUDGET),
        "context": eff_ctx,
        "fully_on_gpu": (
            not bool(v.get("spilled_to_cpu", False))
            and total <= _VRAM_BUDGET
        ),
        "over_budget": total > _VRAM_BUDGET,
        "measured": False,
    }


def _reasoning_status(m: dict, info: dict) -> tuple[str, str, str] | None:
    if not info.get("fully_on_gpu", True):
        return "cpu_offload", "!", "CPU offload"
    cap = m.get("capability", "unknown")
    if cap == "structured":
        return "native", _CAP_GLYPH[cap], _REASONING_LABEL[cap]
    if cap == "inline":
        return "inline", _CAP_GLYPH[cap], _REASONING_LABEL[cap]
    if cap == "unsupported":
        return "none", _CAP_GLYPH[cap], _REASONING_LABEL[cap]
    return None


def _infer_tuning(name: str) -> str:
    """Best-effort tuning style from the tag suffix. Ollama has no API
    field for this — name suffix is the only signal. Returns short label
    or empty string if unknown."""
    n = name.lower()
    # Order matters: more-specific first.
    if "-instruct" in n or n.endswith(":instruct") or "-it" in n:
        return "IT"
    if "-text" in n or "-base" in n:
        return "BASE"
    if "-chat" in n:
        return "CHAT"
    return ""


def _format_model_row(m: dict) -> str:
    info = m.get("_picker_vram") or {}
    vram_num = "?"
    gpu_num = "?"
    if info:
        suffix = "" if info.get("measured") else "*"
        vram_num = f"{info['total_gb']:.2f}{suffix}"
        gpu_num = f"{info['vram_gb']:.2f}"

    ctx_str = _context_label(int(m.get("_picker_context") or 0))
    glyph = m.get("_picker_glyph", "?")
    status_label = m.get("_picker_status_label", "Unknown")

    # Params column: dense → "9B", MoE → "26B/A4B" (total/active).
    moe = m.get("moe") or {}
    details = m.get("details") or {}
    params_label = details.get("param_size") or "?"
    if moe.get("experts_total"):
        active = ""
        for tok in m["name"].lower().replace(":", "-").split("-"):
            if tok.startswith("a") and tok.endswith("b") and tok[1:-1].replace(".", "").isdigit():
                active = tok.upper()
                break
        params_col = f"{params_label}/{active}" if active else params_label
        type_col = "MoE"
    else:
        params_col = params_label
        type_col = "dense"

    quant = details.get("quantization") or "?"
    tuning = _infer_tuning(m["name"])

    return (
        f"      {ctx_str:>4s}  "
        f"{glyph} {status_label:<16s}  "
        f"{m['name']:<32s}  "
        f"{params_col:>10s}  "
        f"{quant:>8s}  "
        f"{tuning:>5s}  "
        f"{type_col:>5s}  "
        f"{vram_num:>10s}  "
        f"{gpu_num:>8s}"
    )


def _name_priority(m: dict) -> tuple:
    """Tag-name priority for the dedup tiebreak.

    Higher tuple = preferred. Ranking:
      1. anything-but-:latest beats :latest (moving alias, no info)
      2. longer name beats shorter (more explicit -q4_K_M / -a3b suffix)
    """
    name = m["name"]
    return (not name.endswith(":latest"), len(name))


def _parse_param_b(label: str) -> float:
    import re
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", label.lower())
    return float(match.group(1)) if match else 0.0


def _quality(m: dict) -> tuple[float, float, float]:
    """Quality proxy used to rank within a family. Higher tuple wins.

    Tuple components in priority order:
      1. parameter count from /api/show details (probe-backed) — most
         meaningful "scale" signal (a 27B Q4 beats a 9B BF16).
      2. actual VRAM at the picker's context — at fixed param count the
         higher-precision quant uses more memory and is preferred.
      3. raw disk size — last-resort tiebreaker for missing metadata.
    """
    details = m.get("details") or {}
    params_b = _parse_param_b(str(details.get("param_size") or ""))
    info = m.get("_picker_vram") or {}
    vram_total = float(info.get("total_gb") or 0.0)
    try:
        size_gb = float((m.get("size") or "0").split()[0])
    except (ValueError, IndexError):
        size_gb = 0.0
    return (params_b, vram_total, size_gb)


def _build_menu(models: list[dict]) -> tuple[list[str], list[bool], list[dict | None]]:
    """Build (display_lines, selectable_flags, model_per_line) for fzf.

    For each family and standard context tier, show the best model for
    each user-facing status: Native reasoning, Inline reasoning, No
    reasoning, and CPU offload. "Best" means highest model quality proxy,
    with explicit tags preferred over moving :latest aliases.
    """
    # backend → family → (context, status) → model
    grouped: dict[str, dict[str, dict[tuple[int, str], dict]]] = {}
    # Track why models were hidden so the footer can explain. The big
    # categories worth distinguishing today:
    #   - non_ollama_dormant: vLLM/SGLang entries are not part of the
    #     current Ollama reasoning flow.
    #   - missing_vram: no usable coefficients or formula totals
    #   - missing_capability: probed state is unknown/error and not CPU offload
    hidden = {
        "non_ollama_dormant": 0,
        "context_capped": 0,
        "missing_vram": 0,
        "missing_capability": 0,
    }
    for m in models:
        backend = m.get("backend", "ollama")
        if backend != "ollama":
            hidden["non_ollama_dormant"] += 1
            continue
        for ctx in _CONTEXT_CHOICES:
            v = m.get("vram") or {}
            info = _vram_info_at(v, ctx)
            if info is None:
                hidden["missing_vram"] += 1
                continue
            if int(info.get("context") or 0) < ctx:
                hidden["context_capped"] += 1
                continue
            status = _reasoning_status(m, info)
            if status is None:
                hidden["missing_capability"] += 1
                continue
            status_key, glyph, status_label = status
            family = m.get("family") or "(uncategorized)"
            candidate = dict(m)
            candidate["_picker_context"] = ctx
            candidate["_picker_vram"] = info
            candidate["_picker_status"] = status_key
            candidate["_picker_glyph"] = glyph
            candidate["_picker_status_label"] = status_label

            bucket = (ctx, status_key)
            family_rows = grouped.setdefault(backend, {}).setdefault(family, {})
            previous = family_rows.get(bucket)
            if (
                previous is None
                or (_quality(candidate), _name_priority(candidate))
                > (_quality(previous), _name_priority(previous))
            ):
                family_rows[bucket] = candidate

    lines: list[str] = []
    selectable: list[bool] = []
    item_models: list[dict | None] = []

    def emit(text: str, *, selectable_: bool, model: dict | None) -> None:
        lines.append(text)
        selectable.append(selectable_)
        item_models.append(model)

    first_section = True
    # Column header is emitted with a 2-char prefix from emit(); leading
    # spaces align it with data rows.
    column_header = (
        f"      {'CTX':>4s}  "
        f"{'REASONING':<18s}  "
        f"{'TAG':<32s}  "
        f"{'PARAMS':>10s}  "
        f"{'QUANT':>8s}  "
        f"{'TUNE':>5s}  "
        f"{'TYPE':>5s}  "
        f"{'VRAM (GB)':>10s}  "
        f"{'GPU GB':>8s}"
    )
    for backend in ("ollama", "vllm", "sglang"):
        if backend not in grouped:
            continue
        if not first_section:
            emit("", selectable_=False, model=None)
        first_section = False
        label = _BACKENDS[backend][0]
        emit(f"  {_BOLD}── {label} ──{_RESET}",
             selectable_=False, model=None)
        emit(f"  {_BOLD}{column_header}{_RESET}",
             selectable_=False, model=None)
        emit(f"  {_DIM}* = formula estimate (vLLM/SGLang have no probe runner yet){_RESET}",
             selectable_=False, model=None)

        # Order families within this backend by their best selected variant.
        # _quality returns a tuple (params_b, vram_total, size_gb); negate
        # via tuple-component sign to keep "higher quality first".
        def _family_sort_key(kv):
            best = max(_quality(m) for m in kv[1].values())
            return tuple(-x for x in best)

        families = sorted(grouped[backend].items(), key=_family_sort_key)
        for fam, selected in families:
            emit(f"    {_BOLD}{fam}:{_RESET}",
                 selectable_=False, model=None)
            top = sorted(
                selected.values(),
                key=lambda m: (
                    int(m.get("_picker_context") or 0),
                    _STATUS_ORDER.get(str(m.get("_picker_status")), 99),
                ),
            )
            for m in top:
                emit(_format_model_row(m), selectable_=True, model=m)

    # Footer: explain hidden rows so a user with NVFP4 weights on disk
    # (or ollama models the probe rejected) knows why they don't appear.
    total_hidden = sum(hidden.values())
    if total_hidden:
        emit("", selectable_=False, model=None)
        bits: list[str] = []
        if hidden["non_ollama_dormant"]:
            bits.append(f"{hidden['non_ollama_dormant']} vLLM/SGLang (dormant)")
        if hidden["missing_vram"]:
            bits.append(f"{hidden['missing_vram']} missing VRAM data")
        if hidden["context_capped"]:
            bits.append(f"{hidden['context_capped']} below requested context")
        if hidden["missing_capability"]:
            bits.append(f"{hidden['missing_capability']} not probed/probe failed")
        emit(f"  {_DIM}hidden: {', '.join(bits)}  ·  "
             f"see docs/ollama_models.md{_RESET}",
             selectable_=False, model=None)

    return lines, selectable, item_models


def _format_agent_row(agent: tuple[str, str, str]) -> str:
    _aid, name, desc = agent
    return f" {_BOLD}{name:<22s}{_RESET}  {desc}"


# ── Command builder ──────────────────────────────────────────────────────────

def _build(agent_id: str, model_name: str, backend: str) -> list[str]:
    _, _, port = _BACKENDS[backend]
    base = f"http://{_ROUTER}:{port}"

    if agent_id == "claude":
        # Claude Code only knows the Anthropic API. Both Ollama (0.21+) and
        # vLLM expose /v1/messages — point claude at the router on the right
        # backend port. ANTHROPIC_AUTH_TOKEN is required even when the local
        # backend ignores auth: claude refuses to dispatch otherwise.
        os.environ["ANTHROPIC_BASE_URL"] = base
        os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", "local")
        return ["claude", "--model", model_name]

    if agent_id == "aider":
        if backend == "ollama":
            return ["aider", "--model", f"ollama_chat/{model_name}"]
        return [
            "aider", "--model", f"openai/{model_name}",
            "--openai-api-base", f"{base}/v1",
            "--openai-api-key", "local",
        ]

    if agent_id == "codex":
        # Codex 0.124+ requires --oss for non-OpenAI endpoints (chat-completions
        # wire was removed). The built-in `ollama` provider is reserved and
        # hard-codes localhost, so we use custom providers `router-<backend>`
        # defined in $CODEX_HOME/config.toml (seeded by the entrypoint).
        # CODEX_HOME and OPENAI_API_KEY come from image ENV — no overrides
        # needed here.
        return [
            "codex",
            "--oss",
            "--local-provider", f"router-{backend}",
            "-c", f'model="{model_name}"',
        ]

    if agent_id == "late":
        # LATE appends `/v1/chat/completions` itself — OPENAI_BASE_URL must
        # NOT end in `/v1` or requests go to /v1/v1/... → HTTP 404.
        os.environ["OPENAI_BASE_URL"] = base
        os.environ["OPENAI_API_KEY"] = "local"
        os.environ["OPENAI_MODEL"] = model_name
        return ["late"]

    if agent_id == "interpreter":
        if backend == "ollama":
            return ["interpreter", "--model", f"ollama/{model_name}"]
        return [
            "interpreter", "--model", f"openai/{model_name}",
            "--api_base", f"{base}/v1",
            "--api_key", "local",
        ]

    sys.exit(f"error: unknown agent '{agent_id}'")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    agent_filter: str | None = None
    if "--agent" in sys.argv:
        pos = sys.argv.index("--agent")
        if pos + 1 < len(sys.argv):
            agent_filter = sys.argv[pos + 1]

    if agent_filter == "bash":
        os.execvp("bash", ["bash"])
    if agent_filter == "gemini":
        os.execvp("gemini", ["gemini"])

    _check_cache_visible()
    models = _discover_models()
    if not models:
        sys.exit(
            "error: no downloaded models found on disk.\n"
            f"  Scanned: {_OLLAMA_MANIFESTS}\n"
            f"       and {_VLLM_DIR}\n"
            "  Pull models first (e.g. `make ollama-pull`, `make vllm-pull`)."
        )

    # Step 1 — pick a model/context/status row. For each family and
    # standard context tier, the picker shows the best model per
    # user-facing status: Native reasoning, Inline reasoning, No reasoning,
    # and CPU offload.
    lines, selectable, item_models = _build_menu(models)
    if not any(selectable):
        sys.exit(
            f"error: no usable model/context rows on disk for "
            f"≤ {_VRAM_BUDGET:g} GB.\n"
            f"  Run `make model-select` so probe data exists.\n"
            f"  Or raise the VRAM budget: VRAM=48 ...\n"
            f"  Or pull smaller models: make ollama-pull MODEL=…"
        )
    header = (
        f"DevAI  ▸  Step 1/2: pick a model  "
        f"(Ollama · {_context_label(_CONTEXT_CHOICES[0])}-"
        f"{_context_label(_CONTEXT_CHOICES[-1])} ctx · "
        f"≤ {_VRAM_BUDGET:g} GB)"
    )
    idx = _fzf(lines, header, selectable=selectable)
    if idx is None:
        os.execvp("bash", ["bash"])
    model = item_models[idx]
    if model is None:  # defensive — _fzf already filters headers
        os.execvp("bash", ["bash"])
    selected_context = int(model.get("_picker_context") or _DEFAULT_CONTEXT)
    os.environ["CONTEXT"] = str(selected_context)

    # Step 2 — pick an agent (skipped when --agent was passed)
    if agent_filter:
        agent = next((a for a in _AGENTS if a[0] == agent_filter), None)
        if agent is None:
            sys.exit(
                f"error: unknown agent '{agent_filter}' "
                f"(known: {', '.join(a[0] for a in _AGENTS)})"
            )
    else:
        alines = [_format_agent_row(a) for a in _AGENTS]
        header = (
            f"Step 2/2: pick an agent for "
            f"{_BOLD}{model['name']}{_RESET}"
            f" @ {_context_label(selected_context)}"
            f" via {_BACKENDS[model['backend']][0]}"
        )
        idx = _fzf(alines, header)
        if idx is None:
            os.execvp("bash", ["bash"])
        agent = _AGENTS[idx]

    cmd = _build(agent[0], model["name"], model["backend"])
    print(
        f"\n  {_BOLD}{agent[0]}{_RESET}"
        f" → {model['name']} @ {_context_label(selected_context)}"
        f" via {_BACKENDS[model['backend']][0]}\n"
    )
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
