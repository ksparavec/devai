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

import atexit
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import yaml

# _capability lives next to this script in both the repo (scripts/) and
# inside the container (/usr/local/bin/, see Dockerfile.lab). Make the
# import work in both layouts without depending on the caller's CWD or
# PYTHONPATH being right.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _capability import Capability  # noqa: E402


# ── Configuration ────────────────────────────────────────────────────────────

_CATALOG_PATHS = [
    "/etc/devai/models.yaml",
    str(Path(__file__).resolve().parent.parent / "deploy" / "models.yaml"),
]

_PROBE_CACHE_PATHS = [
    "/etc/devai/.ollama-reasoning-cache.json",
    str(Path(__file__).resolve().parent.parent / "deploy" / ".ollama-reasoning-cache.json"),
]

_VLLM_PROBE_CACHE_PATHS = [
    "/etc/devai/.vllm-reasoning-cache.json",
    str(Path(__file__).resolve().parent.parent / "deploy" / ".vllm-reasoning-cache.json"),
]

_SGLANG_PROBE_CACHE_PATHS = [
    "/etc/devai/.sglang-reasoning-cache.json",
    str(Path(__file__).resolve().parent.parent / "deploy" / ".sglang-reasoning-cache.json"),
]

_BENCH_CACHE_PATHS = [
    "/etc/devai/.bench-cache.json",
    str(Path(__file__).resolve().parent.parent / "deploy" / ".bench-cache.json"),
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

# DEVAI_MTP_PREVIEW gates the multi-token-prediction UI (MTP column,
# post-select sub-modal, and `::mtp` suffix emission). Off by default so
# the picker behaves identically to pre-MTP builds during the Phase-3
# rollout; turned on alongside the router's parseMTPOverride wiring in
# Phase 5. See docs/multi-token-prediction.md Sec. 7.2 + the catalog-
# crystalline-beaver plan.
_MTP_PREVIEW = os.environ.get("DEVAI_MTP_PREVIEW", "0").lower() in ("1", "true", "yes", "on")

#                     label        reason                                              port
_BACKENDS: dict[str, tuple[str, str, int]] = {
    "ollama": ("Ollama", "GGUF quantized — wide compatibility, CPU+GPU",              11434),
    "vllm":   ("vLLM",   "NVFP4 tensor cores — high throughput, paged attention",     11435),
    "sglang": ("SGLang", "NVFP4 tensor cores — RadixAttention, multi-turn optimized", 11436),
}

# Backends actually surfaced as picker rows. SGLang stays in _BACKENDS,
# in the router, in `make probe-sglang`, and in select-models.py because
# the infrastructure still works and may be useful for chat-only direct
# access — but its OpenAI-compat surface is too thin for current agent
# flows: gpt-oss community parsers mangle harmony tool/reasoning channels
# (vLLM uses OpenAI's official `openai_gptoss`/`openai` parsers instead),
# Codex 0.124+ /v1/responses tools schema only accepts hosted-tool types
# (rejects `function`), Claude Code's split-model orchestration races on
# the long cold-start path. Re-enable by adding "sglang" back here when
# upstream parser/Responses-API support catches up.
_PICKER_BACKENDS: tuple[str, ...] = ("ollama", "vllm")
_PICKER_HF_BACKENDS: tuple[str, ...] = tuple(
    b for b in _PICKER_BACKENDS if b != "ollama"
)

#                  id             display name          description
_AGENTS: list[tuple[str, str, str]] = [
    ("claude",      "Claude Code",       "AI coding assistant with agentic terminal"),
    ("aider",       "Aider",             "Git-aware pair programming"),
    ("codex",       "Codex",             "OpenAI terminal coding agent"),
    ("late",        "LATE",              "Lightweight AI Terminal Environment — ephemeral subagents"),
    ("interpreter", "Open Interpreter",  "Natural language computer control"),
]

# ANSI helpers — chosen for legibility on a black terminal background.
# `_DIM` uses a 256-colour light-grey rather than `\033[2m` (faint),
# which most terminals render too dim to read on black. Bold remains a
# brightness/weight cue for headers and labels.
_BOLD = "\033[1m"
_DIM = "\033[38;5;245m"     # light-grey foreground (~ #8a8a8a)
_GREEN = "\033[38;5;48m"    # bright green for the PRODUCTION_AGENTIC badge
_RESET = "\033[0m"


# Pick-back-channel: when the host launcher (`bin/devai-agent`) bind-mounts
# `~/.devai/` to `/devai-host`, the picker drops a one-shot JSON file the
# launcher reads after exit so it can persist (model, agent, context) into
# `preferences.yaml`. Silent no-op when the path is absent — this is what
# happens for `make shell-gpu` and any other invocation that doesn't go
# through the launcher.
_PICK_BACK_CHANNEL = Path("/devai-host/.last-pick.json")


def _record_pick(model_name: str, agent_id: str, context: int) -> None:
    if not _PICK_BACK_CHANNEL.parent.is_dir():
        return
    try:
        _PICK_BACK_CHANNEL.write_text(json.dumps({
            "model": model_name,
            "agent": agent_id,
            "context": context,
        }))
    except OSError:
        # Permission or I/O issue is not fatal — the pick still launches.
        pass


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


def _load_probe_records() -> dict[str, dict]:
    """Lookup of model name → v3 digest entry from the probe cache.

    Schema v3 keys are digests; each entry carries an `aliases` list and
    a `probes` map nested by VRAM band then context. Every alias resolves
    to the same digest entry here so a `name`-based lookup downstream
    stays one-call. Legacy v1 / v2 caches are migrated in-memory using
    the prober's helper — the on-disk file is unchanged.
    """
    for path in _PROBE_CACHE_PATHS:
        if not os.path.isfile(path):
            continue
        with open(path) as fh:
            data = json.load(fh) or {}
        if not data:
            return {}
        if not all(
            isinstance(v, dict) and v.get("schema_version") == 3
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


def _load_hf_probe_records(paths: list[str]) -> dict[str, dict]:
    """Lookup of model name → schema-v1 HF probe entry.

    The HF caches (vLLM, SGLang) are repo+sha keyed at top level; each
    entry carries an `aliases` list whose first element is the catalog
    `name` (= directory basename on disk). Index by alias here so the
    picker's name-based discovery loop can find probe data with one
    lookup. Returns {} when no readable cache file exists.
    """
    for path in paths:
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as fh:
                data = json.load(fh) or {}
        except OSError as exc:
            print(f"[picker] HF probe cache {path}: {exc}", file=sys.stderr)
            return {}
        except json.JSONDecodeError as exc:
            # Corrupt cache would otherwise hide every HF model from
            # the picker with no UI indication. Surface the path and
            # the parse error so the operator can decide whether to
            # delete the file (and re-probe) or restore a backup.
            print(
                f"[picker] HF probe cache {path} is CORRUPT: {exc}. "
                f"No HF rows will appear until this is resolved.",
                file=sys.stderr,
            )
            return {}
        records: dict[str, dict] = {}
        for _key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            for alias in entry.get("aliases") or []:
                records[alias] = entry
        return records
    return {}


def _load_bench_records(
    paths: list[str],
) -> dict[tuple[str, str, int], dict]:
    """Lookup of ``(model_name, backend, ctx) -> bench cache row``.

    The bench cache (``deploy/.bench-cache.json``) is keyed at top
    level by ``<repo>@<sha>::<backend>::<ctx>`` (HF) or
    ``<digest>::<backend>::<ctx>`` (Ollama) since schema v3 -- the ctx
    suffix stops different-ctx benches of the same model from
    silently overwriting each other (the same model can differ by
    17+ tok/s between 32K and 128K under MTP).

    The picker indexes by ``(row.model, row.backend, row.context)`` so
    a lookup at the user's chosen ctx hits the matching row without
    silently substituting a different ctx's number.

    Pre-v3 rows (no ``context`` field, or ``context: 0``) emit a
    one-line stderr warning and are treated as "no data at this ctx"
    by callers -- the row is loaded but keyed at ctx=0 so it never
    matches a real picker request. Mirrors the probe-cache loader's
    v1-tolerance pattern at gpu-arbiter/main.go:357.

    Returns ``{}`` when no readable cache exists -- non-fatal: rows
    just won't render badges. The picker is fully usable without
    bench data.
    """
    out: dict[tuple[str, str, int], dict] = {}
    for path in paths:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        n_legacy = 0
        for key, entry in raw.items():
            # Skip _meta and any future top-level harness blocks.
            if not _is_bench_row_key(key):
                continue
            if not isinstance(entry, dict):
                continue
            name = entry.get("model")
            backend = entry.get("backend")
            if not (isinstance(name, str) and isinstance(backend, str)):
                continue
            ctx_raw = entry.get("context")
            try:
                ctx = int(ctx_raw) if ctx_raw is not None else 0
            except (TypeError, ValueError):
                ctx = 0
            if ctx == 0:
                n_legacy += 1
            out[(name, backend, ctx)] = entry
        if n_legacy:
            sys.stderr.write(
                f"  picker: bench cache {path}: {n_legacy} pre-v3 row(s) "
                f"without context -- treated as 'no data at this ctx'. "
                f"Re-run `make bench` to populate.\n"
            )
        return out
    return {}


# Meta-block detection for bench cache iteration. Mirrors the source of
# truth at scripts/bench/_bench_core.py:is_row_key — duplicated to avoid
# making model-picker depend on the bench package (the picker ships as
# a single file inside the lab image).
def _is_bench_row_key(key: object) -> bool:
    return isinstance(key, str) and key != "_meta" and not key.startswith("_")


# Thresholds for the PRODUCTION_AGENTIC badge -- mirrors the formula
# documented in docs/bench-results.md > "Picker-tier recommendations".
# Keep this single source of truth: any change here should also be
# reflected in the doc.
_PRODUCTION_AGENTIC_MIN_TOOLS = 0.9
_PRODUCTION_AGENTIC_MIN_HUMANEVAL = 0.7
_PRODUCTION_AGENTIC_MIN_GSM8K = 0.9
_PRODUCTION_AGENTIC_MAX_LEAK = 0.0
_PRODUCTION_AGENTIC_MAX_VRAM_GB = 23.0


def _bench_score(tasks: dict, prefix: str, key: str) -> float | None:
    """Latest matching bench score by prefix.

    Tasks are stored as e.g. ``gsm8k_subset_100``, ``humaneval_subset_50``,
    ``tools_use_20``; a model may have stale entries from prior runs at
    different ``n``. We pick the row with the latest ``ran_at``, falling
    back to the first match if timestamps are missing.
    """
    best: tuple[str, dict] | None = None
    for tname, tdata in (tasks or {}).items():
        if not (isinstance(tname, str) and tname.startswith(prefix)
                and isinstance(tdata, dict)):
            continue
        if best is None:
            best = (tname, tdata)
            continue
        # Prefer larger ran_at lexicographic (ISO-8601 -> chronological).
        cur_ts = str(tdata.get("ran_at") or "")
        best_ts = str(best[1].get("ran_at") or "")
        if cur_ts > best_ts:
            best = (tname, tdata)
    if best is None:
        return None
    raw = best[1].get(key)
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _picker_scores(bench_row: dict | None) -> dict[str, float | None]:
    """Extract the four picker-sort scores from a bench cache row.

    Returns ``{"tps": float|None, "code": float|None, "reas": float|None,
    "total": float|None}``. Each value is ``None`` when the underlying
    bench task hasn't been recorded yet (so the picker can sort
    unbenched rows to the bottom rather than treating them as zeros).

    Definitions:
      * ``tps``   = ``metrics.tps_sustained_p50``
                    (steady-state decode tokens/sec)
      * ``code``  = ``tasks.humaneval_subset_*.pass@1``
      * ``reas``  = ``2/3 * tools_use_score + 1/3 * gsm8k_score``
                    (weighted blend of agentic-tool correctness and
                    multi-step math reasoning)
      * ``total`` = ``mean(gsm8k, humaneval, tools_use)``
                    (equal-weight quality average; latency-only rows
                    have ``None``)
    """
    if bench_row is None:
        return {"tps": None, "code": None, "reas": None, "total": None}
    tasks = bench_row.get("tasks") or {}
    metrics = bench_row.get("metrics") or {}
    tps_raw = metrics.get("tps_sustained_p50")
    try:
        tps = float(tps_raw) if tps_raw is not None else None
    except (TypeError, ValueError):
        tps = None
    code = _bench_score(tasks, "humaneval_", "pass@1")
    gsm = _bench_score(tasks, "gsm8k_", "score")
    tools = _bench_score(tasks, "tools_use", "score")
    # Reasoning blend: tools-use dominates because agentic workflows
    # care more about tool-call correctness than abstract math, but
    # GSM8K stays in the mix as a multi-step-thinking signal.
    if tools is not None and gsm is not None:
        reas = (2.0 / 3.0) * tools + (1.0 / 3.0) * gsm
    else:
        reas = None
    quality_parts = [v for v in (gsm, code, tools) if v is not None]
    total = sum(quality_parts) / len(quality_parts) if quality_parts else None
    return {"tps": tps, "code": code, "reas": reas, "total": total}


def _is_production_agentic(model: dict, bench_row: dict | None) -> bool:
    """Return True when the (model, bench) pair satisfies every
    PRODUCTION_AGENTIC threshold from docs/bench-results.md.

    Missing bench data -> False (badge is never speculative). Backend
    must be vLLM, weights NVFP4, all four correctness/leak gates
    cleared, and peak VRAM strictly under
    ``_PRODUCTION_AGENTIC_MAX_VRAM_GB``.
    """
    if bench_row is None:
        return False
    if (model.get("backend") or "") != "vllm":
        return False
    fmt = str(((model.get("details") or {}).get("quantization") or "")).upper()
    if fmt != "NVFP4":
        return False
    tasks = bench_row.get("tasks") or {}
    metrics = bench_row.get("metrics") or {}
    tools = _bench_score(tasks, "tools_use", "score")
    he = _bench_score(tasks, "humaneval_", "pass@1")
    gsm = _bench_score(tasks, "gsm8k_", "score")
    leak = (tasks.get("leak_probe") or {}).get("leak_rate")
    peak = metrics.get("peak_vram_gb")
    if None in (tools, he, gsm, leak, peak):
        return False
    return (
        tools >= _PRODUCTION_AGENTIC_MIN_TOOLS
        and he >= _PRODUCTION_AGENTIC_MIN_HUMANEVAL
        and gsm >= _PRODUCTION_AGENTIC_MIN_GSM8K
        and float(leak) <= _PRODUCTION_AGENTIC_MAX_LEAK
        and float(peak) < _PRODUCTION_AGENTIC_MAX_VRAM_GB
    )


def _build_comparison_ctx(candidates: list[dict]) -> dict:
    """Pre-compute per-metric ranks across the candidate list so the
    preview pane can describe each model relative to its peers.

    Uses competition ranking (1, 2, 2, 4) — ties share a rank, the
    next position skips. Models without a TOTAL score are excluded
    from the rankings entirely; their preview will show ``--`` and
    skip the use-cases comparison line.

    Returns ``{"n_benched": int, "n_total": int, "ranks": {metric:
    {id(m): rank}}}``. Empty when no candidate has bench data.
    """
    benched: list[tuple[dict, dict]] = []
    for m in candidates:
        s = m.get("_picker_scores") or {}
        if s.get("total") is not None:
            benched.append((m, s))
    ctx: dict = {"n_benched": len(benched), "n_total": len(candidates)}
    if not benched:
        return ctx
    ranks: dict[str, dict[int, int]] = {}
    for metric in ("tps", "code", "reas", "total"):
        rmap: dict[int, int] = {}
        for m, s in benched:
            v = s.get(metric)
            if v is None:
                continue
            higher = sum(
                1 for _, ss in benched
                if ss.get(metric) is not None and ss.get(metric) > v
            )
            rmap[id(m)] = higher + 1
        ranks[metric] = rmap
    ctx["ranks"] = ranks
    return ctx


_PROBE_MODULE = None  # cached after first migration call


def _load_probe_module():
    """Load probe-ollama-reasoning.py as a module (its filename has dashes
    so it isn't importable directly). Cached at module scope to avoid
    repeated exec_module + sys.path mutation on every cache refresh.
    """
    global _PROBE_MODULE
    if _PROBE_MODULE is not None:
        return _PROBE_MODULE
    import importlib.util
    here = Path(__file__).resolve().parent
    # One-shot sys.path insert; idempotent if the prober itself adds it.
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    spec = importlib.util.spec_from_file_location(
        "_probe_module", here / "probe-ollama-reasoning.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_probe_module"] = mod  # frozen-dataclass workaround
    spec.loader.exec_module(mod)
    _PROBE_MODULE = mod
    return mod


def _migrate_in_memory(raw: dict) -> dict:
    """Run probe-ollama-reasoning's v1/v2 -> v3 conversion without touching disk."""
    mod = _load_probe_module()
    migrated, _, _ = mod.ensure_v3(raw)
    return migrated


# ── Disk discovery ───────────────────────────────────────────────────────────

def _ollama_disk_size_gb(library: str, tag: str) -> float:
    """Read ollama manifest, sum layer sizes."""
    manifest = Path(_OLLAMA_MANIFESTS) / library / tag
    try:
        data = json.loads(manifest.read_text())
        total = sum(int(L.get("size", 0)) for L in data.get("layers", []))
        return total / (1024 ** 3)
    except OSError:
        # Missing manifest / unreadable file is fine -- the model just
        # isn't on disk yet. Silent 0.0 keeps the picker's row count
        # honest (the discovery loop already filters out non-existent
        # tags upstream of here).
        return 0.0
    except (ValueError, KeyError) as exc:
        # Malformed manifest is NOT fine: it means the Ollama daemon
        # wrote something we don't understand, and silently returning
        # 0.0 would let it slide. Warn loudly so the operator notices.
        print(
            f"[picker] malformed Ollama manifest {manifest}: {exc}",
            file=sys.stderr,
        )
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


_DTYPE_LABEL = {
    "bfloat16": "BF16",
    "float16":  "FP16",
    "half":     "FP16",
    "float32":  "FP32",
    "float":    "FP32",
}

_NAME_QUANT_TOKENS = ("NVFP4", "FP8", "INT8", "INT4", "AWQ", "GPTQ", "MARLIN")


def _hf_format_label(model_dir: Path) -> str:
    """Short data-format label for an HF model on disk.

    Priority:
      1. `quantization_config.quant_algo` / `quant_method` — most reliable
         when populated.
      2. Quantization token in the directory name (NVFP4, FP8, AWQ, …) —
         some NVIDIA NVFP4 checkpoints (e.g., Nemotron) ship without a
         `quantization_config` block; the convention is the token in
         the repo/dir name.
      3. `torch_dtype` / `dtype` mapped to BF16 / FP16 / FP32 — the native
         precision when no quantization is applied.
    Returns "?" only when config.json is unreadable. Probe caches don't
    record this for vLLM/SGLang (schema-v2), so config.json + name are
    the authoritative sources.
    """
    try:
        with open(model_dir / "config.json") as fh:
            cfg = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return "?"
    qc = cfg.get("quantization_config") or {}
    algo = qc.get("quant_algo") or qc.get("quant_method")
    if algo:
        return str(algo).upper()
    upper_name = model_dir.name.upper()
    for token in _NAME_QUANT_TOKENS:
        if token in upper_name:
            return token
    dtype = cfg.get("torch_dtype") or cfg.get("dtype")
    if dtype:
        return _DTYPE_LABEL.get(str(dtype).lower(), str(dtype).upper())
    return "?"


def _strip_latest(name: str) -> str:
    """Drop a literal trailing `:latest` Ollama moving-alias suffix.

    Only operates on the tag delimiter — a tag like `8b-latest-tag` is
    left alone. Used for display, the agent launch command, and the
    pick-back-channel record so the user never sees `:latest`.
    """
    suffix = ":latest"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def _display_tag(name: str) -> str:
    """Display-only shortener for the picker's TAG column.

    Strips the literal ``NVIDIA-`` prefix that NVIDIA bakes into its
    HF model directories (e.g.
    ``NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`` -> ``Nemotron-...``) so
    the row doesn't push every other column off-screen. The actual
    ``m["name"]`` is unchanged — agent commands, bench-cache lookups,
    pick-back-channel records, and the preview pane all keep using
    the unabbreviated name.
    """
    base = _strip_latest(name)
    prefix = "NVIDIA-"
    return base[len(prefix):] if base.startswith(prefix) else base


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


# vLLM/SGLang engines pre-allocate `gpu_memory_utilization * total_vram`
# at startup, regardless of the per-session context. So `actual_vram_gb`
# from the probe is constant across cells and useless for "VRAM at this
# context" display. The picker overrides it with a formula breakdown
# (weights + KV(ctx) + overhead) so the user sees per-ctx growth that
# actually reflects the model's resource footprint at that tier.
#
# These constants mirror scripts/select-models.py:51, 65 — keeping them
# in sync is a manual operation but each side has small enough surface
# area that drift is unlikely. See `vram_breakdown` over there for the
# canonical formula.
_KV_BYTES_FP16 = 2
_HF_OVERHEAD_GB = 3.0


def _hf_kv_gb(arch: dict, context: int) -> float:
    """KV cache size in GB for a given (arch, ctx) at fp16."""
    if not arch:
        return 0.0
    copies = 1 if arch.get("k_eq_v") else 2
    bytes_per_token = (
        copies
        * int(arch.get("layers") or 0)
        * int(arch.get("kv_heads") or 0)
        * int(arch.get("head_dim") or 0)
        * _KV_BYTES_FP16
    )
    return (bytes_per_token * context) / (1024 ** 3)


def _parse_size_gb(s: str) -> float:
    if not s:
        return 0.0
    s = s.replace("GB", "").replace("G", "").strip()
    try:
        return float(s.split()[0])
    except (ValueError, IndexError):
        return 0.0


def _params_label_from_name(name: str) -> str:
    """Extract a `<digits>B` (or `<digits>B/<digits>B` for MoE) param
    label from an HF model directory basename.

    Matches the conventional NVIDIA/HF naming where param count appears
    as `<N>B` somewhere in the path: `Qwen3-14B-NVFP4` → `14B`,
    `NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` → `30B/A3B`. Returns "" when
    no recognised tokens appear.
    """
    upper = name.upper()
    total_match = re.search(r"(?<![A-Z0-9])(\d+(?:\.\d+)?)B(?![A-Z0-9])", upper)
    if not total_match:
        return ""
    total = total_match.group(0)  # e.g. "14B"
    # MoE active-experts notation, e.g. "A3B" / "A4B" appearing after the
    # total params token.
    active = ""
    tail = upper[total_match.end():]
    active_match = re.search(r"(?<![A-Z0-9])(A\d+(?:\.\d+)?B)(?![A-Z0-9])", tail)
    if active_match:
        active = active_match.group(1)
    return f"{total}/{active}" if active else total


def _vram_from_hf_probe(probe: dict, arch: dict | None = None,
                        weight_gb: float = 0.0) -> dict:
    """Convert a vLLM/SGLang schema-v1 probe entry to the picker's
    Ollama-shape vram dict, with per-ctx formula override.

    The probe's `fits` flag still gates eligibility — that's the engine's
    truth about whether the model actually loaded. But the displayed
    VRAM number per (vram, ctx) cell is recomputed as
    `weight_gb + KV(arch, ctx) + overhead` so the picker shows growth
    with context. Without this, vLLM/SGLang rows appear flat across all
    contexts (they all read back the engine's pre-allocated pool size).

    When arch or weight_gb is missing, falls back to passing through
    the probe's actual_vram_gb.
    """
    if not probe:
        return {}
    out: dict = {"source": "probe"}
    if probe.get("max_context") is not None:
        out["max_context"] = probe["max_context"]
    raw_probes = probe.get("probes") or {}
    converted: dict = {}
    for vram_key, band in raw_probes.items():
        if not isinstance(band, dict):
            continue
        new_band: dict = {}
        for ctx_key, cell in band.items():
            if not isinstance(cell, dict):
                continue
            new_cell = dict(cell)
            new_cell.setdefault("fully_on_gpu", bool(cell.get("fits", False)))
            try:
                ctx = int(ctx_key)
            except (TypeError, ValueError):
                ctx = 0
            measured_vram = cell.get("actual_vram_gb")
            if measured_vram is not None and cell.get("fits"):
                # Probe truth: the prober launched the engine at this
                # exact (vram_band, ctx) and measured nvidia-smi after
                # the chat round-trip. Use it directly. For vLLM that
                # number is the pre-allocated pool (constant across
                # ctx); for SGLang it's the actual working set (varies
                # by ctx). Both are real and authoritative.
                new_cell["actual_total_gb"] = round(float(measured_vram), 2)
            elif arch and weight_gb > 0 and ctx > 0:
                # Formula fallback for cells without probe measurement
                # (e.g. fits=False, or older cache schemas). Marked so
                # the formatter renders an asterisk + footnote.
                kv_gb = _hf_kv_gb(arch, ctx)
                total = weight_gb + kv_gb + _HF_OVERHEAD_GB
                new_cell["actual_total_gb"] = round(total, 2)
                new_cell["actual_vram_gb"] = round(total, 2)
                new_cell["_formula_override"] = True
            elif measured_vram is not None and "actual_total_gb" not in new_cell:
                new_cell["actual_total_gb"] = measured_vram
            new_band[ctx_key] = new_cell
        converted[vram_key] = new_band
    out["probes"] = converted
    return out


def _resolve_tool_parser(probe: dict, catalog_parsers: dict, backend: str) -> str:
    """Return the canonical vLLM/SGLang tool-call-parser name for a row.

    Priority:
      1. Probe-confirmed value — `entry.tool_parser` in the v2 cache.
         Set only when the probe actually round-tripped a tool call.
      2. Catalog hint from `scripts/model-families.yaml` (parsers.<backend>.tool).
         Curated by hand; the prober consumes it as a launch-flag hint.
      3. "N/A" — no parser known. The router strips `tools`/`tool_choice`
         for this model so chat works but agentic tool calls are absent.
    """
    if isinstance(probe, dict):
        probed = probe.get("tool_parser")
        if probed:
            return str(probed)
    backend_block = (catalog_parsers or {}).get(backend) or {}
    curated = backend_block.get("tool")
    if curated:
        return str(curated)
    return "N/A"


def _capability_from_probe(probe: dict, fallback: str = Capability.UNKNOWN) -> str:
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
    probes = _load_probe_records()
    out: list[dict] = []

    # Ollama: <library>/<tag> manifest files. Skip ctx-variant derived
    # tags — they're a per-session artefact created by the picker via
    # Modelfile to bake `num_ctx` in. The user picks the parent; the
    # picker materialises the variant on launch.
    _ctx_tag = re.compile(r"-ctx\d+$")
    base = Path(_OLLAMA_MANIFESTS)
    if base.is_dir():
        for lib_dir in sorted(base.iterdir()):
            if not lib_dir.is_dir():
                continue
            for tag_file in sorted(lib_dir.iterdir()):
                if not tag_file.is_file():
                    continue
                if _ctx_tag.search(tag_file.name):
                    continue
                name = f"{lib_dir.name}:{tag_file.name}"
                meta = catalog.get(name, {})
                disk_gb = _ollama_disk_size_gb(lib_dir.name, tag_file.name)
                probe = probes.get(name) or {}
                cap = _capability_from_probe(probe, Capability.UNKNOWN)
                probe_details = _details_from_probe(probe)
                out.append({
                    "name": name,
                    "source": "ollama",
                    "backend": "ollama",
                    "size": meta.get("size") or f"{disk_gb:.2f} GB",
                    "purpose": meta.get("purpose", ""),
                    "vram": _vram_from_probe(probe),
                    "family": meta.get("family", ""),
                    "capability": cap,
                    "probe": probe,
                    "catalog_meta": meta,
                    "moe": _moe_from_probe(probe),
                    "details": probe_details,
                    # Ollama negotiates tool calls natively per-request via
                    # `/api/chat`; no engine-startup parser flag is needed.
                    # Show `native` so the user knows tools work for any
                    # tool-trained Ollama model without configuration.
                    "tool_parser": "native",
                })

    # HF/vLLM/SGLang: <name>/config.json. Each backend keeps its own
    # probe cache (schema v1, repo+sha keyed); look up by directory
    # basename, which is also the catalog `name` and the first alias.
    #
    # vLLM and SGLang are NOT 100% interchangeable: architecture and
    # quantization-format support diverge. So we emit ONE row per
    # (file, backend) pair where the backend has a non-error probe.
    # A safetensors checkpoint probed by both backends produces TWO
    # rows that the picker presents in separate sections — the user
    # explicitly chooses which backend to benchmark on.
    #
    # When a backend has no probe at all for a file, we fall back to
    # emitting a single placeholder row using DEVAI_HF_BACKEND so the
    # file still appears (with capability=unknown). _reasoning_status
    # filters those out of the menu, but they're useful for debugging
    # `model-picker --show` style listings.
    vllm_probes = _load_hf_probe_records(_VLLM_PROBE_CACHE_PATHS)
    sglang_probes = _load_hf_probe_records(_SGLANG_PROBE_CACHE_PATHS)
    # Placeholder rows for files with no probe data anywhere are tagged
    # with DEVAI_HF_BACKEND when it is one of the picker-exposed HF
    # backends; otherwise we fall back to the first picker-exposed HF
    # backend (typically "vllm"). Without this gate, setting
    # DEVAI_HF_BACKEND=sglang would still surface placeholder rows
    # tagged sglang even though sglang is filtered out of the picker.
    fallback_backend = (
        _HF_BACKEND if _HF_BACKEND in _PICKER_HF_BACKENDS
        else (_PICKER_HF_BACKENDS[0] if _PICKER_HF_BACKENDS else "vllm")
    )

    def _hf_probes_for(name: str) -> list[tuple[dict, str]]:
        """Return one (probe_entry, backend) pair per backend with a
        non-error probe for this model. Empty list when neither backend
        has a usable entry — the caller emits a single placeholder."""
        out_pairs: list[tuple[dict, str]] = []
        # Iterate only picker-exposed HF backends. The sglang probe
        # cache is still loaded (above) and used by the router; it
        # just doesn't generate menu rows while sglang is filtered out.
        store_by_backend = {"vllm": vllm_probes, "sglang": sglang_probes}
        for backend in _PICKER_HF_BACKENDS:
            store = store_by_backend.get(backend)
            if store is None:
                continue
            entry = store.get(name)
            if not entry:
                continue
            if entry.get("capability") in (Capability.ERROR, Capability.UNSUPPORTED_ARCH):
                continue
            out_pairs.append((entry, backend))
        return out_pairs

    vbase = Path(_VLLM_DIR)
    if vbase.is_dir():
        for d in sorted(vbase.iterdir()):
            if not (d.is_dir() and (d / "config.json").is_file()):
                continue
            name = d.name
            meta = catalog.get(name, {})
            disk_gb = _hf_disk_size_gb(d)
            fmt_label = _hf_format_label(d)
            # Catalog metadata for per-ctx VRAM formula. arch comes from
            # generate-catalog.py walking the HF config.json; weight_gb
            # parsed from the catalog `size` ("9.82 GB" → 9.82).
            arch = meta.get("arch")
            weight_gb = _parse_size_gb(str(meta.get("size") or f"{disk_gb:.2f} GB"))
            # Params label parsed from the directory basename (e.g.,
            # `Qwen3-14B-NVFP4` → `14B`). HF probes don't expose a
            # param_size_label like Ollama's /api/show does, so the
            # display column would otherwise read `?` for every HF row.
            params_label = _params_label_from_name(name)
            details = {"quantization": fmt_label}
            if params_label:
                details["param_size"] = params_label
            common = {
                "name": name,
                "source": "hf",
                "size": meta.get("size") or f"{disk_gb:.2f} GB",
                "purpose": meta.get("purpose", ""),
                "family": meta.get("family", ""),
                "moe": None,
                "details": details,
                # Catalog metadata (incl. `conversational`) is consumed
                # by _tuning_label downstream to populate the TUNE column
                # without resorting to name-pattern guesswork.
                "catalog_meta": meta,
            }
            # Catalog parser hints (curated in scripts/model-families.yaml,
            # propagated into deploy/models.yaml by generate-catalog.py).
            # Used as fallback when the probe didn't (yet) confirm a parser.
            cat_parsers = (meta.get("parsers") or {}) if isinstance(meta, dict) else {}
            pairs = _hf_probes_for(name)
            if pairs:
                # One row per (file, backend) so vLLM and SGLang appear
                # independently in the picker.
                for probe, backend in pairs:
                    out.append({
                        **common,
                        "backend": backend,
                        "vram": _vram_from_hf_probe(probe, arch, weight_gb),
                        "capability": _capability_from_probe(probe, Capability.UNKNOWN),
                        "probe": probe,
                        "tool_parser": _resolve_tool_parser(probe, cat_parsers, backend),
                    })
            else:
                # No working probe on either backend — emit one placeholder
                # row that _reasoning_status() will filter out of the menu.
                out.append({
                    **common,
                    "backend": fallback_backend,
                    "vram": None,
                    "capability": Capability.UNKNOWN,
                    "probe": {},
                    "tool_parser": _resolve_tool_parser({}, cat_parsers, fallback_backend),
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
         selectable: list[bool] | None = None,
         preview_cmd: str | None = None,
         preview_window: str = "right:42%:wrap",
         extra_bindings: list[str] | None = None,
         input_text: str | None = None,
         header_lines: int = 0) -> int | None:
    """Run fzf; return selected line index or None on cancel.

    `selectable[i]=False` marks line i as a non-selectable section header.
    Index sentinel `--` is used for those; if user lands on one fzf still
    returns it but we re-prompt.

    `preview_cmd`, when set, is passed to fzf's `--preview`. The
    delimiter is `\t` and the row tag is field {1}; `--with-nth 2..`
    keeps the tag out of the visible row but available for the preview
    substitution. Header rows produce a blank preview (their tag is
    `--` and the cmd is expected to fail silently for missing files).

    `extra_bindings` adds custom fzf `--bind` clauses (one per list
    entry, fzf syntax). Used for the bench-score sort cycle on ctrl-s.

    `input_text` overrides the default tag-prefixed line stream (used
    when a binding provides its own pre-built input file -- the caller
    can hand fzf the same shape directly).

    `header_lines` (>0) maps to fzf's `--header-lines=N` so the first
    N input lines become a fixed header the cursor cannot navigate
    onto. Used by the model picker so the column header / sort
    indicator / formula note never receive focus -- the preview pane
    always corresponds to a selectable model row.
    """
    if selectable is None:
        selectable = [True] * len(lines)
    indexed = []
    for i, line in enumerate(lines):
        tag = str(i) if selectable[i] else "--"
        indexed.append(f"{tag}\t{line}")
    fzf_input = input_text if input_text is not None else "\n".join(indexed)
    while True:
        try:
            args = [
                "fzf", "--reverse", "--no-sort", "--ansi", "--no-info",
                # --exact disables fzf's default fuzzy matching: a
                # query like "nemo" must appear as a literal substring
                # of the row text, not as scattered characters in
                # order. Without it, "nemo" matched Qwen3-Coder rows
                # via "qwe[n]3-Cod[e]r ... [Mo]E" -- correct fuzzy
                # behaviour, wrong UX for a model picker.
                "--exact",
                "--delimiter", "\t", "--with-nth", "2..",
                "--header", header,
                "--pointer", "▶", "--prompt", "Search: ",
                "--margin", "1,2",
                # Always draw a rounded border around every prompt so the
                # window is clearly delimited on dark terminals. Bright
                # cyan (color 14) matches the existing header tone.
                "--border", "rounded",
                # Colours tuned for a BLACK terminal background.
                # Avoid the dim ANSI 0–7 bank (especially blue/4 and
                # green/2 which vanish on black). Use the bright bank
                # (8–15) for foreground accents and an explicit dark-
                # grey (`bg+:236`) bar for the current line so the
                # cursor row pops without the colours hurting eyes.
                #   fg:15      bright white         normal text
                #   fg+:15     bright white         current row text
                #   bg+:236    dark-grey            current row bar
                #   pointer:11 bright yellow        ▶ glyph
                #   header:14  bright cyan          fzf header line
                #   hl:11      bright yellow        match highlight
                #   hl+:11     bright yellow        match highlight (cur)
                #   prompt:14  bright cyan          search prompt
                #   info:8     bright black (grey)  match counter
                #   gutter:-1  terminal default     left margin column
                #   border:14  bright cyan          window frame
                "--color",
                "fg:15,fg+:15,bg+:236,pointer:11,header:14,"
                "hl:11,hl+:11,prompt:14,info:8,gutter:-1,border:14",
            ]
            if header_lines > 0:
                args.extend(["--header-lines", str(header_lines)])
            if preview_cmd:
                # Preview pane gets its own rounded frame so the split
                # between list and details is visually obvious. Older
                # fzf versions silently ignore the suffix; newer ones
                # render it in the main `border:` colour. The "hidden"
                # spec is left bare -- a hidden window has nothing to
                # frame, and some fzf versions reject the combo.
                pwindow = preview_window
                if pwindow != "hidden" and "border" not in pwindow:
                    pwindow = f"{pwindow}:border-rounded"
                args.extend([
                    "--preview", preview_cmd,
                    "--preview-window", pwindow,
                ])
            for binding in (extra_bindings or []):
                args.extend(["--bind", binding])
            result = subprocess.run(
                args,
                input=fzf_input,
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
    Capability.STRUCTURED:       "●",  # clean reasoning
    Capability.INLINE:           "◐",  # reasoning leaks into content
    # `none` is a probed-clean non-reasoning model (e.g. Llama-3.1
    # working as designed); `unsupported` is a probed-but-no-reasoning
    # outcome (typically a configuration mismatch). Distinct glyphs so
    # the operator can tell the two apart in the preview pane.
    Capability.NONE:             "○",  # non-reasoning by design
    Capability.UNSUPPORTED:      "·",  # no reasoning detected
    Capability.UNKNOWN:          "?",  # not yet probed
    Capability.ERROR:            "✗",  # probe failed
}

_REASONING_LABEL = {
    Capability.STRUCTURED:  "Native reasoning",
    Capability.INLINE:      "Inline reasoning",
    Capability.NONE:        "Non-reasoning model",
    Capability.UNSUPPORTED: "No reasoning",
}

# Inline-reasoning models leak `<think>` blocks into content. They appear
# in the flat list once; the user toggles reasoning ON/OFF via the info
# modal, and the launcher appends `::nothink` when the user picks OFF.
# The router strips that suffix and forces enable_thinking=false.


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
    """Return the v3 probe cell at the picker's VRAM band and effective ctx.

    `v` is the full v3 entry as exposed by `_vram_from_probe(probe)`.
    The probes map is nested: `probes[<vram_gb>][<ctx>]`. We use the
    picker's `_VRAM_BUDGET` env (an int GB) as the band key. If no cell
    is recorded for the (band, ctx) pair we return {} — there is no
    interpolation any more.
    """
    max_ctx = int(v.get("max_context") or 0)
    eff_ctx = min(context, max_ctx) if max_ctx else context
    probes = v.get("probes")
    if not isinstance(probes, dict) or not probes:
        return {}
    band_key = str(int(_VRAM_BUDGET))
    band = probes.get(band_key)
    if not isinstance(band, dict):
        return {}
    rec = band.get(str(eff_ctx))
    if isinstance(rec, dict):
        return rec
    return {}


def _vram_info_at(v: dict, context: int) -> dict | None:
    exact = _measured_point_at(v, context)
    if exact:
        # Cells whose chat probe errored before VRAM was measured have
        # no actual_total_gb — return None so the picker bins them as
        # "missing VRAM data". Cells that loaded but spilled DO carry
        # VRAM (with fully_on_gpu=False) and flow through; the spill
        # filter in _build_menu hides them under the "spilled" bin.
        if exact.get("actual_total_gb") is None:
            return None
        total = round(float(exact["actual_total_gb"]), 2)
        vram_gb = round(float(exact.get("actual_vram_gb") or total), 2)
        # Cells overridden with formula-derived VRAM (HF rows — vLLM
        # and SGLang probes don't measure per-ctx VRAM since the
        # engine pre-allocates a constant pool) flag themselves so the
        # formatter can render an asterisk and footnote.
        is_measured = not bool(exact.get("_formula_override", False))
        return {
            "total_gb": total,
            "vram_gb": vram_gb,
            "context": int(exact.get("actual_context") or context),
            "fully_on_gpu": bool(exact.get("fully_on_gpu", True))
            and total <= _VRAM_BUDGET,
            "over_budget": total > _VRAM_BUDGET,
            "measured": is_measured,
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


def _tuning_label(m: dict) -> str:
    """Return the post-training tuning label for the TUNE column.

    Three reliable signals, OR'd as positive evidence for IT:
      a) Ollama probe `capabilities` (from /api/show): presence of
         `tools` / `thinking` / `vision` ⇒ instruction-tuned.
      b) HF catalog `conversational` flag (from /api/models/{repo}.tags):
         True ⇒ instruction-tuned.
      c) Name suffix pattern (`-instruct`, `:instruct`, `-it`, `-chat`):
         instruction-tuned.

    Negative evidence (BASE) requires NO positive signal AND either:
      - Ollama `capabilities == [completion]` (the daemon explicitly
        says no tools / thinking — strong signal), or
      - HF `conversational == false` AND no IT-style name pattern, or
      - Name pattern `-text` / `-base`.

    The OR logic compensates for NVIDIA's HF quants that ship without
    the `conversational` tag despite obviously being instruct-tuned
    (`Llama-3.1-8B-Instruct-NVFP4` is the canonical example). When all
    sources are silent, returns "" so the column doesn't lie.
    """
    name = m.get("name") or ""
    n = name.lower()
    name_says_it = (
        "-instruct" in n or n.endswith(":instruct")
        or "-it" in n or "-chat" in n
    )
    name_says_base = "-text" in n or "-base" in n

    probe = m.get("probe") or {}
    caps = probe.get("capabilities")
    caps_says_it = isinstance(caps, list) and any(
        c in caps for c in ("tools", "thinking", "vision")
    )
    caps_says_base = (
        isinstance(caps, list) and caps and not caps_says_it
        and "completion" in caps
    )

    catalog_meta = m.get("catalog_meta") or {}
    conv = catalog_meta.get("conversational")

    # Positive evidence wins over negative.
    if caps_says_it or conv is True or name_says_it:
        return "CHAT" if "-chat" in n and not (caps_says_it or conv is True) else "IT"
    if caps_says_base or conv is False or name_says_base:
        return "BASE"
    return ""


def _fmt_score_pct(v: float | None) -> str:
    """0.876 -> '87.6'. None -> '-'. Three-char wide for picker columns."""
    if v is None:
        return "-"
    return f"{v * 100:.1f}"


def _fmt_tps(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.1f}"


def _fmt_leak_pct(m: dict) -> str:
    """leak_rate * 100 as 1-decimal percent, or '-' when there is no
    bench data on this row. Three-char value fits in a 5-char column.
    """
    bench_row = m.get("_picker_bench_row") or {}
    leak_rate = ((bench_row.get("tasks") or {}).get("leak_probe") or {}).get("leak_rate")
    if leak_rate is None:
        return "-"
    return f"{leak_rate * 100:.1f}"


def _format_model_row(m: dict, idx: int = 0) -> str:
    info = m.get("_picker_vram") or {}
    vram_num = "?"
    if info:
        suffix = "" if info.get("measured") else "*"
        vram_num = f"{info['total_gb']:.2f}{suffix}"

    ctx_str = _context_label(int(m.get("_picker_context") or 0))

    # Params column: dense -> "9B", MoE -> "26B/A4B" (total/active).
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
    else:
        params_col = params_label

    display_name = _display_tag(m["name"])
    backend_col = str(m.get("backend") or "?")
    fmt_col = str(details.get("quantization") or "?")
    type_col = "MoE" if _is_moe(m) else "Dense"
    tools_col = "Yes" if _has_tools(m) else "No"
    # MTP column is only rendered when the preview flag is on; until
    # the router's parseMTPOverride wiring lands in Phase 5 the column
    # would be informational-only and the sub-modal would have no
    # downstream effect, so gate them together.
    mtp_col = ("Yes" if _has_mtp(m) else "No") if _MTP_PREVIEW else ""

    # Bench scores -- four columns the user requested. Unbenched rows
    # render '-' so they sort to the bottom but still appear (with all
    # the format/parser/tier metadata visible in the preview pane).
    scores = m.get("_picker_scores") or {}
    tps_col = _fmt_tps(scores.get("tps"))
    code_col = _fmt_score_pct(scores.get("code"))
    reas_col = _fmt_score_pct(scores.get("reas"))
    total_col = _fmt_score_pct(scores.get("total"))
    leak_col = _fmt_leak_pct(m)

    # Line number reflects position in the current sort order, so
    # ctrl-s renumbers the list. Caller passes 1-based ``idx``;
    # ``00.`` is a sentinel used only when the helper is invoked
    # outside ``_build_menu`` (e.g. unit-style tests).
    num_col = f"{idx:02d}."
    mtp_segment = f"{mtp_col:>5s}  " if _MTP_PREVIEW else ""
    return (
        f"{num_col:>3s}  "
        f"{ctx_str:>5s}  "
        f"{display_name:<34s}  "
        f"{backend_col:>7s}  "
        f"{params_col:>10s}  "
        f"{type_col:>6s}  "
        f"{fmt_col:>7s}  "
        f"{tools_col:>5s}  "
        f"{mtp_segment}"
        f"{tps_col:>7s}  "
        f"{code_col:>7s}  "
        f"{reas_col:>7s}  "
        f"{total_col:>7s}  "
        f"{leak_col:>5s}  "
        f"{vram_num:>6s}"
    )


def _parse_param_b(label: str) -> float:
    match = re.search(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*b\b", label.lower())
    return float(match.group(1)) if match else 0.0


def _params_hint(m: dict) -> float:
    """Parameter count in billions. Probe-supplied label preferred;
    falls back to parsing the model name (e.g., `Qwen3-14B-NVFP4` → 14)
    so HF rows aren't penalised when the probe didn't expose param_size."""
    details = m.get("details") or {}
    parsed = _parse_param_b(str(details.get("param_size") or ""))
    if parsed:
        return parsed
    return _parse_param_b(str(m.get("name") or ""))


# ── Capability scoring (flat list ordering) ──────────────────────────────────

_REASONING_SCORE: dict[str, int] = {
    Capability.STRUCTURED:  2,
    Capability.INLINE:      1,
    Capability.NONE:        0,
    Capability.UNSUPPORTED: 0,
    Capability.UNKNOWN:     0,
    Capability.ERROR:       0,
}


def _has_tools(m: dict) -> bool:
    """True when the model can serve tool calls today.

    Ollama negotiates tools per-request natively (parser='native'); HF
    rows are tools-capable only when the probe confirmed a parser AND
    didn't flag `disable_verified` (router strips tools/tool_choice in
    that case to avoid backend rejection).
    """
    parser = str(m.get("tool_parser") or "").strip()
    if not parser or parser.upper() == "N/A":
        return False
    if parser == "native":
        return True
    probe = m.get("probe") or {}
    return not bool(probe.get("disable_verified"))


# Known MoE family substrings -- name-based fallback for cases where
# neither the probe-cache `moe.experts_total` field nor the param_size
# "/A<n>B" notation marks the model. Matched case-insensitively against
# the model name. Extend when new MoE families land.
_MOE_NAME_HINTS: tuple[str, ...] = ("gpt-oss",)


def _is_moe(m: dict) -> bool:
    """Mixture-of-experts detector. Three signals, any of which wins:

    1. ``moe.experts_total`` truthy -- set by the Ollama prober for
       confirmed MoE checkpoints.
    2. ``details.param_size`` contains a slash (e.g. ``30B/A3B``) --
       the convention used by the HF/NVFP4 catalog to encode total
       and active param counts in one string. HF probes don't fill
       ``moe.experts_total`` so this is the load-bearing signal for
       the picker's vLLM rows.
    3. Model name matches one of ``_MOE_NAME_HINTS`` -- catches MoE
       families whose names (e.g. ``gpt-oss-20b``) hide the active
       count and would otherwise read as dense.
    """
    moe = m.get("moe") or {}
    if moe.get("experts_total"):
        return True
    psize = str((m.get("details") or {}).get("param_size") or "")
    if "/" in psize:
        return True
    name_lc = str(m.get("name") or "").lower()
    return any(hint in name_lc for hint in _MOE_NAME_HINTS)


def _mtp_block(m: dict) -> dict | None:
    """Return the catalog's `mtp:` block for this row, or None when
    the row has no MTP variant. Source: catalog_meta attached during
    _discover_models from deploy/models.yaml. Defensive against the
    block being None or non-dict (legacy probe-cache entries pre-
    catalog and operator-edited YAML).
    """
    meta = m.get("catalog_meta") or {}
    if not isinstance(meta, dict):
        return None
    mtp = meta.get("mtp")
    if not isinstance(mtp, dict) or not mtp.get("method"):
        return None
    return mtp


def _has_mtp(m: dict) -> bool:
    """Whether the catalog declares MTP for this row -- drives the
    picker's MTP column ('Yes'/'No') and gates the sub-modal."""
    return _mtp_block(m) is not None


def _ctx_tier(ctx: int) -> int:
    """Index of `ctx` in ascending _CONTEXT_CHOICES. Larger ctx → larger tier."""
    tiers = sorted(_CONTEXT_CHOICES)
    idx = 0
    for i, t in enumerate(tiers):
        if ctx >= t:
            idx = i
    return idx


def _score(m: dict) -> float:
    """Weighted capability score. Higher = better; sort descending.

    Priority (high → low): reasoning > tools > context > MoE > params.
    Weights chosen so each tier dominates the sum of all lower tiers
    given realistic ranges (reasoning 0..2, tools 0..1, ctx_tier 0..3,
    MoE 0..1, params 0..~70).
    """
    cap = str(m.get("capability") or Capability.UNKNOWN)
    reasoning = _REASONING_SCORE.get(cap, 0)
    tools = 1 if _has_tools(m) else 0
    info = m.get("_picker_vram") or {}
    ctx = int(info.get("_picker_ctx") or m.get("_picker_context") or 0)
    moe_flag = 1 if (m.get("moe") or {}).get("experts_total") else 0
    return (
        reasoning * 1_000_000
        + tools * 10_000
        + _ctx_tier(ctx) * 1_000
        + moe_flag * 100
        + _params_hint(m)
    )


def _max_fitting_ctx_info(m: dict) -> dict | None:
    """Largest probe-confirmed context tier that fits fully on GPU at the
    picker's VRAM band. Returns the _vram_info_at-style dict (with an
    extra `_picker_ctx` key) or None if nothing fits.
    """
    v = m.get("vram") or {}
    if not v:
        return None
    for ctx in sorted(_CONTEXT_CHOICES, reverse=True):
        info = _vram_info_at(v, ctx)
        if info is None:
            continue
        if not info.get("fully_on_gpu", False):
            continue
        if int(info.get("context") or 0) < ctx:
            continue
        info = dict(info)
        info["_picker_ctx"] = ctx
        return info
    return None


def _dedup_hf_by_name(models: list[dict]) -> list[dict]:
    """When the same model name has rows for multiple HF backends, keep
    the highest-priority one. vLLM > SGLang. Ollama tag names never
    collide with HF directory names, so Ollama rows pass through
    unchanged.
    """
    hf_priority = {"vllm": 2, "sglang": 1}
    chosen: dict[str, dict] = {}
    for m in models:
        name = m.get("name") or ""
        prev = chosen.get(name)
        if prev is None:
            chosen[name] = m
            continue
        cur_p = hf_priority.get(str(m.get("backend") or ""), 0)
        prev_p = hf_priority.get(str(prev.get("backend") or ""), 0)
        if cur_p > prev_p:
            chosen[name] = m
    return list(chosen.values())


# ── Quant format notes (info modal copy) ────────────────────────────────────

# Per-format one-paragraph explanation. Looked up uppercase. Unknown
# formats render no note (the "Format:" line still shows the marker).
_FORMAT_NOTES: dict[str, str] = {
    "NVFP4": (
        "NVIDIA FP4: 4-bit float (E2M1) with per-block FP8 scales. "
        "About 5x smaller than BF16 and accelerated by Blackwell/Hopper "
        "FP4 tensor cores; quality typically lands within 1-2% of BF16 "
        "when prepared with nvidia/Modelopt."
    ),
    "FP4": (
        "Generic 4-bit float (E2M1). About 4x smaller than BF16; "
        "quality depends heavily on the calibration recipe and per-block "
        "scaling."
    ),
    "MXFP4": (
        "OCP Microscaling FP4: per-32-element block scales, open-standard "
        "successor to vendor FP4 schemes. Runs on FP4-capable tensor "
        "cores (Blackwell, MI350)."
    ),
    "FP8": (
        "8-bit float (E4M3 or E5M2). Half the size of BF16 with marginal "
        "quality loss; commonly paired as the KV-cache format alongside "
        "NVFP4 weights."
    ),
    "BF16": (
        "Brain float 16: full-precision inference reference. 2x the size "
        "of FP8 / 4x of NVFP4; pick when you need the highest-fidelity "
        "baseline and have the VRAM to spare."
    ),
    "FP16": (
        "Half-precision float 16. Same size as BF16 but a narrower "
        "exponent range; legacy choice from pre-Hopper hardware."
    ),
    "F16": (
        "GGUF 16-bit float (BF16 or FP16, depending on the source "
        "tensor). Full-precision GGUF tier; 2x the size of Q8 with "
        "no measurable quality gain on most workloads."
    ),
    "Q8_0": (
        "GGUF 8-bit integer quant. About 50% the size of BF16 with "
        "quality essentially indistinguishable -- the safest GGUF tier "
        "when VRAM allows."
    ),
    "Q6_K": (
        "GGUF 6-bit k-quant. About 38% the size of BF16 with very small "
        "quality loss; a solid middle ground when Q8 doesn't fit."
    ),
    "Q5_K_M": (
        "GGUF 5-bit k-quant (medium variant). About 33% the size of "
        "BF16; small quality cost vs Q8 and noticeably less VRAM."
    ),
    "Q5_K_S": (
        "GGUF 5-bit k-quant (small variant). Slightly smaller than "
        "Q5_K_M with marginally more quality loss."
    ),
    "Q4_K_M": (
        "GGUF 4-bit k-quant (medium variant). About 25% the size of "
        "BF16 -- the most popular size/quality balance for local "
        "inference."
    ),
    "Q4_K_S": (
        "GGUF 4-bit k-quant (small variant). Smaller than Q4_K_M with "
        "a touch more quality loss."
    ),
    "Q3_K_L": (
        "GGUF 3-bit k-quant (large variant). Aggressive size reduction "
        "with noticeable quality drift on reasoning and code tasks."
    ),
    "Q3_K_M": (
        "GGUF 3-bit k-quant (medium variant). Aggressive size reduction "
        "with noticeable quality drift on reasoning and code tasks."
    ),
    "Q3_K_S": (
        "GGUF 3-bit k-quant (small variant). Smaller still; measurable "
        "quality loss on most workloads."
    ),
    "Q3_K_XL": (
        "GGUF 3-bit k-quant (extra-large variant). Slightly larger than "
        "Q3_K_L for a touch more quality at minor VRAM cost."
    ),
    "Q2_K": (
        "GGUF 2-bit k-quant. The smallest GGUF tier; substantial quality "
        "loss on most workloads -- only pick when nothing larger fits."
    ),
    "AWQ": (
        "Activation-aware Weight Quantization: 4-bit weights with "
        "importance-preserving scaling. Generally outperforms GPTQ at "
        "the same bitwidth."
    ),
    "GPTQ": (
        "GPTQ: post-training quantization, typically 4-bit. Older but "
        "well-tested; AWQ usually edges it out on quality."
    ),
}


def _format_quant_note(fmt: str, indent_cols: int = 11, wrap_cols: int = 50) -> str:
    """Per-format explanation block, wrapped + indented to align under
    the "Format:" line in the preview pane. Returns "" for unknown
    formats so the preview just shows the marker without a note.
    """
    if not fmt or fmt == "?":
        return ""
    note = _FORMAT_NOTES.get(fmt.upper())
    if not note:
        return ""
    pad = " " * indent_cols
    wrapped = textwrap.wrap(note, width=wrap_cols)
    return "\n".join(f"{pad}{ln}" for ln in wrapped)


# ── Per-family use-case blurbs (info modal copy) ─────────────────────────────

_FAMILY_USE_CASES: dict[str, str] = {
    "qwen3.5": (
        "Strong coding, math, and structured reasoning. Long-context "
        "retrieval and agentic tool use. Default first choice for most "
        "coding-agent workflows."
    ),
    "qwen3": (
        "Predecessor to Qwen3.5. Inline thinking; solid coding and "
        "instruction-following. Reasonable fallback when the 3.5 weights "
        "are not on disk."
    ),
    "gpt-oss": (
        "OpenAI open-weight reasoning model. Harmony channel cleanly "
        "separates reasoning, tool calls, and answers. Pairs well with "
        "Codex and Claude Code's tool-heavy flows."
    ),
    "llama3.1": (
        "General-purpose chat and instruction-following. Solid baseline "
        "for RAG, summarisation, and assistant-style work where "
        "reasoning is not the primary need."
    ),
    "llama3.2": (
        "Smaller Llama variant. Snappy chat at low VRAM cost; good for "
        "routing/triage agents and edge-style deployments."
    ),
    "gemma4": (
        "Compact Google-tuned model. Low-latency chat in the 8K-128K "
        "range; good for interactive REPL-style work where round-trip "
        "time matters."
    ),
    "deepseek-r1-distill": (
        "Reasoning-distilled from DeepSeek R1. Heavy chain-of-thought; "
        "best for math, code analysis, and competitive-programming-style "
        "problems."
    ),
    "nemotron-nano-v2": (
        "NVIDIA-tuned compact Nemotron. Instruction-tuned with tool "
        "support; good balance of throughput and capability on a single "
        "mid-range GPU."
    ),
    "nemotron-3-nano": (
        "Latest NVIDIA Nano series. Improved reasoning and tool use over "
        "earlier Nano variants; suitable for agentic tasks at modest "
        "VRAM."
    ),
    "nemotron-cascade-2": (
        "NVIDIA Cascade tuning. Strong long-context retrieval and "
        "structured reasoning. Useful when both context and reasoning "
        "matter at once."
    ),
    "nemotron": (
        "Original NVIDIA Nemotron line. Dense Llama-derivative; "
        "conservative choice for stable instruction-following without "
        "surprises."
    ),
    "__default__": (
        "General-purpose model. Use the columns above to pick the right "
        "tradeoff between capability, context, and VRAM headroom."
    ),
}


def _capability_summary_text(
    m: dict,
    reasoning_mode: str = "default",
    comparison: dict | None = None,
) -> str:
    """Body text for the info modal: capabilities + per-model bench
    properties + per-family use case.

    ``comparison`` is the dict returned by ``_build_comparison_ctx``;
    when present and the model has bench scores, a "Model properties"
    section is appended with TPS / CODE / REAS / TOTAL plus per-metric
    rank against the rest of the picker list, peak VRAM headroom,
    steady-state TTFT, and a non-zero leak warning. The "Use cases"
    blurb gets one bench-derived sentence appended when the model is
    a per-metric leader or carries a noteworthy caveat.
    """
    name = _strip_latest(m.get("name") or "")
    backend = str(m.get("backend") or "?")
    info = m.get("_picker_vram") or {}
    ctx = int(info.get("_picker_ctx") or m.get("_picker_context") or 0)
    cap = str(m.get("capability") or Capability.UNKNOWN)
    reason_label = _REASONING_LABEL.get(cap, "Unknown")
    if cap == Capability.INLINE and reasoning_mode == "nothink":
        reason_label = f"{reason_label} (forced OFF for this launch)"
    tools_label = "Yes" if _has_tools(m) else "No"
    type_label = "MoE" if _is_moe(m) else "Dense"
    details = m.get("details") or {}
    fmt = details.get("quantization") or "?"
    params = str(details.get("param_size") or "")
    if not params:
        hint = _params_hint(m)
        params = f"{hint:g}B" if hint else "?"
    parser = str(m.get("tool_parser") or "N/A")
    vram = info.get("total_gb")
    vram_str = f"{vram:.2f} GB" if vram else "?"
    family = (m.get("family") or "").lower()
    blurb = _FAMILY_USE_CASES.get(family, _FAMILY_USE_CASES["__default__"])

    properties_section = _format_model_properties(m, comparison)
    extra_use_case_lines = _extra_use_case_lines(m, comparison)

    fmt_note = _format_quant_note(str(fmt))
    fmt_block = f"Format:    {fmt}\n"
    if fmt_note:
        fmt_block += fmt_note + "\n"
    mtp = _mtp_block(m)
    if mtp:
        drafter = mtp.get("drafter")
        drafter_label = drafter.split("/")[-1] if drafter else "(built-in head)"
        mtp_line = (
            f"MTP:       {mtp.get('method', '?')}  K={mtp.get('num_speculative_tokens', '?')}"
            f"  drafter={drafter_label}\n"
        )
    else:
        mtp_line = "MTP:       not available\n"
    head = (
        f"Model:     {name}\n"
        f"Backend:   {backend}\n"
        f"{fmt_block}"
        f"Params:    {params}    Type: {type_label}\n"
        f"Context:   {_context_label(ctx)} (max fit at {_VRAM_BUDGET:g} GB)\n"
        f"VRAM:      {vram_str}\n"
        f"\n"
        f"Reasoning: {reason_label}\n"
        f"Tools:     {tools_label}    (parser: {parser})\n"
        f"{mtp_line}"
    )
    parts = [head]
    if properties_section:
        parts.append("\n" + properties_section)
    use_cases_body = blurb
    if extra_use_case_lines:
        # Bench-derived sentences sit in their own paragraph: a blank
        # line separates them from the family blurb above, and each
        # one starts flush at column 1 (no leading indent) so the
        # block reads as the next paragraph rather than a sub-bullet.
        use_cases_body = (
            blurb.rstrip()
            + "\n\n"
            + "\n".join(extra_use_case_lines)
        )
    parts.append(f"\nUse cases:\n{use_cases_body}")
    return "".join(parts)


def _format_model_properties(m: dict, comparison: dict | None) -> str:
    """Render the "Model properties" preview section, or an empty
    string when there's no bench data to report.

    Layout (one line each, indented by 2 spaces):
        TPS:    143.8 tok/s    (rank 1/9)
        CODE:   98.0%          (rank 1/9)
        REAS:   99.7%          (rank 1/9)
        TOTAL:  99.0%          (rank 1/9)
        Peak:   22.54 GB       (1.46 GB headroom under 24 GB cap)
        TTFT:   38 ms          (steady-state, post-warmup)
        Leaks:  7.5%           (3 of 40 prompts emitted special tokens)
    """
    scores = m.get("_picker_scores") or {}
    bench_row = m.get("_picker_bench_row") or {}
    if scores.get("total") is None or not bench_row:
        # No bench data for this (model, backend, ctx). Distinguish
        # "model has bench rows at other ctxs but not this one" from
        # "model never benched at all" -- the former is fixable with a
        # targeted `make bench --ctx <N>`, the latter wants a full run.
        ctx = int(m.get("_picker_context") or 0)
        bench_other = m.get("_picker_bench_other_ctxs") or []
        if bench_other:
            tiers = ", ".join(_context_label(c) for c in sorted(bench_other))
            return (
                "Model properties:\n"
                f"  Bench: not available at ctx={_context_label(ctx)} "
                f"(have {tiers}; run `make bench --ctx "
                f"{_context_label(ctx)}` to populate)\n"
            )
        if comparison and comparison.get("n_benched", 0) > 0:
            return (
                "Model properties:\n"
                "  (no bench data for this row -- run `make bench-vllm` "
                "to populate)\n"
            )
        return ""

    ranks = (comparison or {}).get("ranks") or {}
    n = (comparison or {}).get("n_benched", 0)
    rid = id(m)

    def _rank_str(metric: str) -> str:
        r = ranks.get(metric, {}).get(rid)
        if not r or n <= 0:
            return ""
        return f"  (rank {r}/{n})"

    lines: list[str] = ["Model properties:"]
    tps = scores.get("tps")
    if tps is not None:
        lines.append(f"  TPS:    {tps:>6.1f} tok/s{_rank_str('tps')}")
    code = scores.get("code")
    if code is not None:
        lines.append(f"  CODE:   {code * 100:>5.1f}%{_rank_str('code')}")
    reas = scores.get("reas")
    if reas is not None:
        lines.append(f"  REAS:   {reas * 100:>5.1f}%{_rank_str('reas')}")
    total = scores.get("total")
    if total is not None:
        lines.append(f"  TOTAL:  {total * 100:>5.1f}%{_rank_str('total')}")
    metrics = bench_row.get("metrics") or {}
    peak = metrics.get("peak_vram_gb")
    if peak is not None:
        headroom = float(_VRAM_BUDGET) - float(peak)
        lines.append(
            f"  Peak:   {peak:>5.2f} GB"
            f"   ({headroom:.2f} GB headroom under {_VRAM_BUDGET:g} GB cap)"
        )
    ttft_p50 = metrics.get("ttft_ms_steady_p50")
    if ttft_p50 is not None:
        lines.append(f"  TTFT:   {ttft_p50:>5.0f} ms (steady-state, post-warmup)")
    tasks = bench_row.get("tasks") or {}
    leak_probe = tasks.get("leak_probe") or {}
    leak_rate = leak_probe.get("leak_rate")
    n_prompts = leak_probe.get("n_prompts")
    if leak_rate is not None and leak_rate > 0:
        leaked_n = (
            int(round(leak_rate * n_prompts))
            if isinstance(n_prompts, (int, float)) and n_prompts
            else None
        )
        suffix = (
            f"   ({leaked_n} of {n_prompts} prompts emitted special tokens)"
            if leaked_n is not None
            else ""
        )
        lines.append(f"  Leaks:  {leak_rate * 100:>5.1f}%{suffix}")
    return "\n".join(lines) + "\n"


def _extra_use_case_lines(m: dict, comparison: dict | None) -> list[str]:
    """Return at most two bench-derived sentences to append to the
    family use-cases blurb. Sentences are picked to highlight what's
    *useful to a picker user*: the per-metric leader gets a callout,
    leaks earn a caveat. Skipped silently when bench data is absent.
    """
    out: list[str] = []
    if not comparison:
        return out
    scores = m.get("_picker_scores") or {}
    if scores.get("total") is None:
        return out
    ranks = comparison.get("ranks") or {}
    rid = id(m)
    rt = ranks.get("total", {}).get(rid)
    rtps = ranks.get("tps", {}).get(rid)
    rcode = ranks.get("code", {}).get(rid)
    rreas = ranks.get("reas", {}).get(rid)
    n = comparison.get("n_benched", 0)

    if rt == 1:
        out.append(
            f"Top-scoring model in this picker on the equally-weighted "
            f"GSM8K + HumanEval + tools_use blend "
            f"(TOTAL = {scores['total'] * 100:.1f}% across {n} benched models); "
            f"safe default agentic pick on this hardware."
        )
    elif rcode == 1 and rtps == 1:
        out.append(
            "Highest HumanEval pass@1 *and* highest decode TPS in this "
            "picker -- the strongest default for code-heavy workflows."
        )
    elif rtps == 1:
        out.append(
            f"Fastest decoder in this picker "
            f"({scores['tps']:.1f} tok/s steady-state); pick when latency or "
            f"throughput matters more than peak quality."
        )
    elif rcode == 1:
        out.append(
            f"Best HumanEval pass@1 in this picker "
            f"({scores['code'] * 100:.1f}%) -- strong code-completion default."
        )
    elif rreas == 1:
        out.append(
            f"Strongest tools+reasoning blend in this picker "
            f"(REAS = {scores['reas'] * 100:.1f}%); good fit for agentic flows "
            f"that combine tool calls with multi-step thinking."
        )

    bench_row = m.get("_picker_bench_row") or {}
    leak = ((bench_row.get("tasks") or {}).get("leak_probe") or {}).get("leak_rate")
    if isinstance(leak, (int, float)) and leak > 0:
        out.append(
            f"Caveat: leaks chat-template special tokens at "
            f"{leak * 100:.1f}% of prompts; sanitise output for downstream "
            f"consumers or skip for production agentic deployment."
        )
    return out


# Sort modes for the picker's score / context cycle. ctrl-s cycles in
# this order; TOTAL is the default landing mode. CTX sorts by the
# largest probe-confirmed context tier that fits at the picker's VRAM
# band -- useful when the user cares about long-context capacity over
# raw quality scores.
_SORT_MODES: tuple[str, ...] = ("total", "tps", "code", "reas", "ctx")
_SORT_LABELS: dict[str, str] = {
    "total": "TOTAL",
    "tps": "TPS",
    "code": "CODE",
    "reas": "REAS",
    "ctx": "CTX",
}


_SORT_DIRS: tuple[str, ...] = ("desc", "asc")
_SORT_ARROWS: dict[str, str] = {"desc": "▼", "asc": "▲"}


def _sort_key_for_mode(mode: str, sort_dir: str = "desc"):
    """Return a key fn that pushes models with no relevant data to the
    bottom and otherwise sorts the chosen metric in ``sort_dir``.

    Tuple shape: ``(missing_flag, +/- primary_value, -capability_score)``.
    ``missing_flag`` is 1 when the primary value is absent so unbenched
    rows always trail regardless of direction. ``sort_dir="desc"``
    flips the sign so largest values come first; ``"asc"`` keeps the
    natural sign so smallest come first. The capability score is the
    tertiary tie-breaker.
    """
    sign = -1 if sort_dir == "desc" else 1

    def key(m: dict):
        if mode == "ctx":
            ctx = int(m.get("_picker_context") or 0)
            return (
                0 if ctx > 0 else 1,
                ctx * sign,
                -_score(m),
            )
        scores = m.get("_picker_scores") or {}
        v = scores.get(mode)
        return (
            0 if v is not None else 1,
            (v if v is not None else 0.0) * sign,
            -_score(m),
        )
    return key


def _build_candidates(
    models: list[dict],
    bench_records: dict[tuple[str, str, int], dict] | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Decorate every fitting model with picker metadata and return the
    list once. Used by both ``_build_menu`` (sort/render) and the
    main loop's pre-rendered sort-mode files so every mode shares
    the same candidate dict identities (``id(m)`` is stable across
    sort calls, which the tag mapping relies on).

    Second return is the ``hidden`` counter dict so callers can show
    a per-reason "hidden" footer.
    """
    bench_records = bench_records or {}
    hidden = {
        "missing_capability": 0,
        "no_fitting_ctx": 0,
    }
    candidates: list[dict] = []
    for m in models:
        cap = str(m.get("capability") or Capability.UNKNOWN)
        if cap in (Capability.UNKNOWN, Capability.ERROR, Capability.UNSUPPORTED_ARCH):
            hidden["missing_capability"] += 1
            continue
        info = _max_fitting_ctx_info(m)
        if info is None:
            hidden["no_fitting_ctx"] += 1
            continue
        decorated = dict(m)
        decorated["_picker_vram"] = info
        decorated["_picker_context"] = int(info.get("_picker_ctx") or 0)
        decorated["_picker_status"] = cap
        decorated["_picker_glyph"] = _CAP_GLYPH.get(cap, "?")
        decorated["_picker_status_label"] = _REASONING_LABEL.get(cap, "Unknown")
        decorated["_picker_mode"] = "default"
        name_key = str(decorated.get("name") or "")
        backend_key = str(decorated.get("backend") or "")
        ctx_key = int(decorated.get("_picker_context") or 0)
        bench_key = (name_key, backend_key, ctx_key)
        bench_row = bench_records.get(bench_key)
        # Capture other ctx tiers this (model, backend) is benched at so
        # the preview pane can tell the user "have 32K, 128K; missing
        # 64K" instead of the unhelpful "no bench data" when the row
        # exists at a different ctx.
        other_ctxs = sorted(
            c
            for (n, b, c), _row in bench_records.items()
            if n == name_key and b == backend_key and c != ctx_key and c > 0
        )
        decorated["_picker_scores"] = _picker_scores(bench_row)
        decorated["_picker_bench_row"] = bench_row
        decorated["_picker_bench_other_ctxs"] = other_ctxs
        decorated["_picker_agentic"] = _is_production_agentic(decorated, bench_row)
        candidates.append(decorated)
    # HF dedup -- vLLM preferred over SGLang for the same name. Ollama
    # tag names never collide with HF directory names so Ollama rows
    # pass through unchanged.
    candidates = _dedup_hf_by_name(candidates)
    return candidates, hidden


def _build_menu(
    models: list[dict],
    bench_records: dict[tuple[str, str, int], dict] | None = None,
    sort_mode: str = "total",
    sort_dir: str = "desc",
    *,
    _candidates: list[dict] | None = None,
    _hidden: dict[str, int] | None = None,
) -> tuple[list[str], list[bool], list[dict | None]]:
    """Build (display_lines, selectable_flags, model_per_line) for fzf.

    Flat list ordered by ``sort_mode`` (one of ``_SORT_MODES``) in
    ``sort_dir`` direction (``"desc"`` or ``"asc"``). Models without a
    bench row sort to the bottom in every direction. One row per
    model.

    Pass ``_candidates`` / ``_hidden`` from a single ``_build_candidates``
    call when rendering multiple sort modes -- it preserves dict identity
    across modes so tag-based fzf reload mapping stays correct.
    """
    if _candidates is None:
        candidates, hidden = _build_candidates(models, bench_records)
    else:
        candidates = list(_candidates)
        hidden = _hidden or {"missing_capability": 0, "no_fitting_ctx": 0}

    if sort_mode not in _SORT_MODES:
        sort_mode = "total"
    if sort_dir not in _SORT_DIRS:
        sort_dir = "desc"
    candidates.sort(key=_sort_key_for_mode(sort_mode, sort_dir))

    lines: list[str] = []
    selectable: list[bool] = []
    item_models: list[dict | None] = []

    sort_label = _SORT_LABELS.get(sort_mode, sort_mode.upper())
    arrow = _SORT_ARROWS.get(sort_dir, "")

    def _hdr(label: str, mode_key: str) -> str:
        """Append the active-direction arrow to the matching column
        header so the user sees at a glance which column drives the
        current sort and in which direction."""
        return f"{label}{arrow}" if mode_key == sort_mode else label

    mtp_header_segment = f"{'MTP':>5s}  " if _MTP_PREVIEW else ""
    column_header = (
        f"{'##':>3s}  "
        f"{_hdr('CTX', 'ctx'):>5s}  "
        f"{'TAG':<34s}  "
        f"{'BACKEND':>7s}  "
        f"{'PARAMS':>10s}  "
        f"{'TYPE':>6s}  "
        f"{'FORMAT':>7s}  "
        f"{'TOOLS':>5s}  "
        f"{mtp_header_segment}"
        f"{_hdr('TPS', 'tps'):>7s}  "
        f"{_hdr('CODE%', 'code'):>7s}  "
        f"{_hdr('REAS%', 'reas'):>7s}  "
        f"{_hdr('TOTAL%', 'total'):>7s}  "
        f"{'LEAK%':>5s}  "
        f"{'VRAM':>6s}"
    )
    lines.append(f"{_BOLD}{column_header}{_RESET}")
    selectable.append(False)
    item_models.append(None)
    cycle_hint = " > ".join(_SORT_LABELS[m] for m in _SORT_MODES)
    sort_note = (
        f"sort: {_BOLD}{sort_label} {arrow} {sort_dir}{_RESET}{_DIM}  "
        f"(ctrl-s cycles {cycle_hint}; ctrl-r flips direction){_RESET}"
    )
    lines.append(f"  {sort_note}")
    selectable.append(False)
    item_models.append(None)
    note = (
        "* = VRAM formula estimate.  "
        "CODE = HumanEval pass@1.  "
        "REAS = 2/3*tools + 1/3*gsm8k.  "
        "TOTAL = mean."
    )
    lines.append(f"  {_DIM}{note}{_RESET}")
    selectable.append(False)
    item_models.append(None)

    for idx, m in enumerate(candidates, start=1):
        lines.append(_format_model_row(m, idx))
        selectable.append(True)
        item_models.append(m)

    total_hidden = sum(hidden.values())
    if total_hidden:
        lines.append("")
        selectable.append(False)
        item_models.append(None)
        bits: list[str] = []
        if hidden["no_fitting_ctx"]:
            bits.append(
                f"{hidden['no_fitting_ctx']} no context tier fits "
                f"≤ {_VRAM_BUDGET:g} GB"
            )
        if hidden["missing_capability"]:
            bits.append(
                f"{hidden['missing_capability']} not probed/probe failed"
            )
        lines.append(f"  {_DIM}hidden: {', '.join(bits)}{_RESET}")
        selectable.append(False)
        item_models.append(None)

    return lines, selectable, item_models


def _format_agent_row(agent: tuple[str, str, str]) -> str:
    _aid, name, desc = agent
    return f" {_BOLD}{name:<22s}{_RESET}  {desc}"


# ── Command builder ──────────────────────────────────────────────────────────

def _build(agent_id: str, model_name: str, backend: str) -> list[str]:
    _, _, port = _BACKENDS[backend]
    base = f"http://{_ROUTER}:{port}"
    name = model_name

    if agent_id == "claude":
        # Claude Code only knows the Anthropic API. Both Ollama (0.21+) and
        # vLLM expose /v1/messages — point claude at the router on the right
        # backend port. ANTHROPIC_AUTH_TOKEN is required even when the local
        # backend ignores auth: claude refuses to dispatch otherwise.
        os.environ["ANTHROPIC_BASE_URL"] = base
        os.environ.setdefault("ANTHROPIC_AUTH_TOKEN", "local")
        # Claude Code's split-model design uses a smaller/cheaper model for
        # background calls (summarization, prompt rewriting, sub-agent
        # dispatch, file-content compaction). In cloud that's Claude Haiku
        # alongside Sonnet — sensible cost/latency tradeoff. Locally we
        # have one model loaded; switching is a 50-60s cold-start, not a
        # cheap call. Without these overrides Claude Code emits its
        # hardcoded haiku id (e.g. "claude-haiku-4-5-20251001"), the
        # router has no row for it, phantom-launches a vLLM container
        # that never serves, and the foreground turn starves on the 10-
        # minute health timeout. Pin both slots to the picker-chosen
        # model so all calls hit the already-loaded backend.
        os.environ["ANTHROPIC_SMALL_FAST_MODEL"] = name
        os.environ["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = name
        return ["claude", "--model", name]

    if agent_id == "aider":
        if backend == "ollama":
            return ["aider", "--model", f"ollama_chat/{name}"]
        return [
            "aider", "--model", f"openai/{name}",
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
            "-c", f'model="{name}"',
        ]

    if agent_id == "late":
        # LATE appends `/v1/chat/completions` itself — OPENAI_BASE_URL must
        # NOT end in `/v1` or requests go to /v1/v1/... → HTTP 404.
        os.environ["OPENAI_BASE_URL"] = base
        os.environ["OPENAI_API_KEY"] = "local"
        os.environ["OPENAI_MODEL"] = name
        return ["late"]

    if agent_id == "interpreter":
        if backend == "ollama":
            return ["interpreter", "--model", f"ollama/{name}"]
        return [
            "interpreter", "--model", f"openai/{name}",
            "--api_base", f"{base}/v1",
            "--api_key", "local",
        ]

    sys.exit(f"error: unknown agent '{agent_id}'")


# ── Main ─────────────────────────────────────────────────────────────────────

def _resolve_agent(agent_filter: str | None, model: dict) -> tuple[str, str, str] | None:
    """Drive reasoning toggle (inline-reasoning only) → MTP toggle (when
    catalog declares it AND DEVAI_MTP_PREVIEW is on) → agent picker.

    The model details are shown live in the model-list preview pane, so
    there is no separate confirmation step here — Enter on the model row
    in the outer fzf advances directly to this function.

    Returns (agent_id, reasoning_mode, mtp_mode) on launch, or None when
    the user pressed Esc and the caller should re-enter the model list.
    mtp_mode is "off" by default; "on" only when the sub-modal explicitly
    enables it. Always "off" when DEVAI_MTP_PREVIEW is unset.
    """
    reasoning_mode = "default"
    mtp_mode = "off"
    cap = str(model.get("capability") or "")
    if cap == Capability.INLINE:
        toggle_lines = [
            f"  Reasoning ON   {_DIM}(default — model thinks inline){_RESET}",
            f"  Reasoning OFF  {_DIM}(force enable_thinking=false / ::nothink){_RESET}",
        ]
        toggle_header = (
            f"Reasoning mode  ▸  {_BOLD}{_strip_latest(model['name'])}{_RESET}"
            f"   {_DIM}(Esc → back to model list){_RESET}"
        )
        idx = _fzf(toggle_lines, toggle_header)
        if idx is None:
            return None
        if idx == 1:
            reasoning_mode = "nothink"

    # MTP sub-modal mirrors the reasoning one. Two gates: the env-flag
    # rollout switch AND the catalog actually declaring an mtp: block
    # for this row. Without the env flag we silently keep MTP off so
    # the picker behaves identically to pre-MTP builds.
    if _MTP_PREVIEW and _has_mtp(model):
        mtp = _mtp_block(model) or {}
        method = mtp.get("method", "?")
        k = mtp.get("num_speculative_tokens", "?")
        warn = ""
        if cap == Capability.INLINE and reasoning_mode != "nothink":
            warn = (
                f"\n  {_DIM}note: MTP + reasoning on inline-reasoning models will be "
                f"rejected by the router (vllm#34650). Choose ::nothink or MTP OFF.{_RESET}"
            )
        mtp_lines = [
            f"  MTP OFF        {_DIM}(default — vanilla decode, no drafter){_RESET}",
            f"  MTP ON         {_DIM}({method}, K={k} — ~2-3x decode speedup){_RESET}",
        ]
        mtp_header = (
            f"MTP toggle  ▸  {_BOLD}{_strip_latest(model['name'])}{_RESET}"
            f"   {_DIM}(Esc → back to model list){_RESET}{warn}"
        )
        idx = _fzf(mtp_lines, mtp_header)
        if idx is None:
            return None
        if idx == 1:
            mtp_mode = "on"

    if agent_filter:
        agent = next((a for a in _AGENTS if a[0] == agent_filter), None)
        if agent is None:
            sys.exit(
                f"error: unknown agent '{agent_filter}' "
                f"(known: {', '.join(a[0] for a in _AGENTS)})"
            )
        return (agent[0], reasoning_mode, mtp_mode)

    alines = [_format_agent_row(a) for a in _AGENTS]
    mode_notes = []
    if reasoning_mode == "nothink":
        mode_notes.append("no reasoning")
    if mtp_mode == "on":
        mode_notes.append("MTP")
    mode_note = f"  [{', '.join(mode_notes)}]" if mode_notes else ""
    header = (
        f"Pick agent  ▸  {_BOLD}{_strip_latest(model['name'])}{_RESET}"
        f"{mode_note}  @ {_context_label(int(model.get('_picker_context') or 0))}"
        f"  via {_BACKENDS[model['backend']][0]}"
        f"   {_DIM}(Esc → back to model list){_RESET}"
    )
    idx = _fzf(alines, header)
    if idx is None:
        return None
    return (_AGENTS[idx][0], reasoning_mode, mtp_mode)


def main() -> None:
    # Non-interactive entry point. When devai-agent passes --prompt, both the
    # model and the agent are already chosen and there is no human at the
    # picker UI. Bypass the fzf loop entirely and exec the agent with -p.
    noninteractive_prompt = os.environ.get("DEVAI_NONINTERACTIVE_PROMPT")
    if noninteractive_prompt:
        pref_model = os.environ.get("DEVAI_PREF_MODEL")
        pref_agent = os.environ.get("DEVAI_PREF_AGENT")
        if not pref_model or not pref_agent:
            sys.exit(
                "error: DEVAI_NONINTERACTIVE_PROMPT is set but "
                "DEVAI_PREF_MODEL and/or DEVAI_PREF_AGENT is missing. "
                "Call devai-agent with --model and --agent."
            )
        if pref_agent != "claude":
            sys.exit(
                f"error: non-interactive --prompt currently supports "
                f"--agent claude only, got {pref_agent!r}."
            )
        # Backend selection: a per-session context suffix (`@<ctx>`) is the
        # vLLM/SGLang launcher contract; bare names are Ollama. SGLang is
        # not probed in INSTALL.md, so the @<ctx> path always means vLLM
        # in the supported configuration.
        backend = "vllm" if "@" in pref_model else "ollama"
        cmd = _build(pref_agent, pref_model, backend)
        # _build returns ["claude", "--model", name]; splice in the
        # one-shot prompt right after the executable so the rest of the
        # claude CLI flags still parse cleanly.
        cmd = [cmd[0], "-p", noninteractive_prompt, *cmd[1:]]
        os.execvp(cmd[0], cmd)

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

    bench_records = _load_bench_records(_BENCH_CACHE_PATHS)
    # One candidate-decoration pass shared by every (mode, direction)
    # so that ``id(m)`` matches across renders and the tag-to-model
    # mapping for fzf reload stays consistent.
    candidates, hidden = _build_candidates(models, bench_records)
    menus: dict[tuple[str, str], tuple[list[str], list[bool], list[dict | None]]] = {}
    for mode in _SORT_MODES:
        for direction in _SORT_DIRS:
            menus[(mode, direction)] = _build_menu(
                models, bench_records,
                sort_mode=mode, sort_dir=direction,
                _candidates=candidates, _hidden=hidden,
            )
    lines, selectable, item_models = menus[("total", "desc")]
    if not any(selectable):
        sys.exit(
            f"error: no usable model/context rows on disk for "
            f"≤ {_VRAM_BUDGET:g} GB.\n"
            f"  Run `make model-select` so probe data exists.\n"
            f"  Or raise the VRAM budget: VRAM=48 ...\n"
            f"  Or pull smaller models: make ollama-pull MODEL=…"
        )

    header = (
        f"DevAI  ▸  Pick a model  "
        f"(≤ {_VRAM_BUDGET:g} GB · ctrl-s sort · ctrl-r dir · ? preview)"
    )

    # Materialise per-row info files for fzf's preview pane. Each
    # selectable row gets <preview_dir>/<idx>.txt; header rows have
    # tag `--`, so the cat fails silently and the pane shows blank
    # while the user is on a non-data row.
    preview_dir = tempfile.mkdtemp(prefix="devai-picker-")
    atexit.register(shutil.rmtree, preview_dir, ignore_errors=True)
    # Comparison context is computed once over the full candidate list
    # so the per-model preview can describe each row's rank against
    # its peers (TPS / CODE / REAS / TOTAL).
    comparison_ctx = _build_comparison_ctx(candidates)
    # Preview content is keyed off the original (TOTAL-mode) item index,
    # which equals the tag fzf carries for each line. All four sort
    # modes share the same item_models list (only the row order changes
    # within each pre-rendered file), so previews stay correct after
    # ctrl-s reloads.
    for i, m in enumerate(item_models):
        if m is None:
            continue
        Path(preview_dir, f"{i}.txt").write_text(
            _capability_summary_text(m, comparison=comparison_ctx)
        )
    preview_cmd = f"cat {preview_dir}/{{1}}.txt 2>/dev/null"

    # Pre-render every (mode, dir) combination's tag-prefixed input
    # stream. Each render references items via their *original*
    # (TOTAL-desc) index so the preview cmd's {1} field still
    # resolves to the right detail file after a reload.
    base_models = list(item_models)
    base_index: dict[int, int] = {id(m): i for i, m in enumerate(base_models) if m is not None}
    sort_files: dict[tuple[str, str], Path] = {}
    for mode in _SORT_MODES:
        for direction in _SORT_DIRS:
            m_lines, m_selectable, m_items = menus[(mode, direction)]
            rendered: list[str] = []
            for line, sel, m in zip(m_lines, m_selectable, m_items):
                if not sel or m is None:
                    tag = "--"
                else:
                    tag = str(base_index.get(id(m), 0))
                rendered.append(f"{tag}\t{line}")
            path = Path(preview_dir, f"sort-{mode}-{direction}.txt")
            path.write_text("\n".join(rendered))
            sort_files[(mode, direction)] = path

    # Two state files split mode and direction so ctrl-s cycles modes
    # while preserving direction, and ctrl-r flips direction while
    # preserving mode. Initial: TOTAL desc.
    state_mode_path = Path(preview_dir, "sort.mode")
    state_dir_path = Path(preview_dir, "sort.dir")
    state_mode_path.write_text("0")
    state_dir_path.write_text("desc")
    modes_array = " ".join(_SORT_MODES)

    cycle_mode_path = Path(preview_dir, "cycle-sort.sh")
    cycle_mode_path.write_text(
        "#!/bin/bash\n"
        "set -e\n"
        f"state_mode={state_mode_path}\n"
        f"state_dir={state_dir_path}\n"
        f"files_dir={preview_dir}\n"
        f"modes=({modes_array})\n"
        "cur=$(cat \"$state_mode\" 2>/dev/null || echo 0)\n"
        "next=$(( (cur + 1) % ${#modes[@]} ))\n"
        "echo \"$next\" > \"$state_mode\"\n"
        "dir=$(cat \"$state_dir\" 2>/dev/null || echo desc)\n"
        "cat \"$files_dir/sort-${modes[$next]}-${dir}.txt\"\n"
    )
    cycle_mode_path.chmod(0o755)

    cycle_dir_path = Path(preview_dir, "cycle-dir.sh")
    cycle_dir_path.write_text(
        "#!/bin/bash\n"
        "set -e\n"
        f"state_mode={state_mode_path}\n"
        f"state_dir={state_dir_path}\n"
        f"files_dir={preview_dir}\n"
        f"modes=({modes_array})\n"
        "cur=$(cat \"$state_mode\" 2>/dev/null || echo 0)\n"
        "dir=$(cat \"$state_dir\" 2>/dev/null || echo desc)\n"
        "if [ \"$dir\" = \"desc\" ]; then new=asc; else new=desc; fi\n"
        "echo \"$new\" > \"$state_dir\"\n"
        "cat \"$files_dir/sort-${modes[$cur]}-${new}.txt\"\n"
    )
    cycle_dir_path.chmod(0o755)

    # Bindings:
    #   ctrl-s   -- cycle bench-score sort mode (TOTAL > TPS > CODE >
    #               REAS > CTX). Direction is preserved across cycles.
    #   ctrl-r   -- flip sort direction (desc <-> asc). Mode is
    #               preserved across flips.
    #   ?        -- toggle preview pane on/off. Single-char fzf
    #               action; reliable across terminals. Cost: '?' can
    #               no longer be typed into the fuzzy-search query.
    #   ctrl-p   -- alias for ?:toggle-preview.
    # Non-selectable rows (column header / sort note / formula note)
    # are made un-focusable via `--header-lines=3` below, so the
    # preview is always meaningful when visible.
    bindings = [
        f"ctrl-s:reload({cycle_mode_path})",
        f"ctrl-r:reload({cycle_dir_path})",
        "?:toggle-preview",
        "ctrl-p:toggle-preview",
    ]

    while True:
        idx = _fzf(
            lines, header,
            selectable=selectable,
            preview_cmd=preview_cmd,
            extra_bindings=bindings,
            input_text=sort_files[("total", "desc")].read_text(),
            # Column header + sort note + formula note. _build_menu
            # always emits exactly these three at the top; if that
            # ever changes, update this constant in lockstep.
            header_lines=3,
        )
        if idx is None:
            # User cancelled the model list (Ctrl-C / Esc). Exit cleanly so
            # the container's PID 1 terminates and `podman run --rm`
            # reclaims it — same lifecycle as quitting an agent.
            sys.exit(0)
        model = item_models[idx]
        if model is None:  # defensive — _fzf already filters headers
            continue

        decision = _resolve_agent(agent_filter, model)
        if decision is None:
            # Backed out of the info / agent modal. Re-enter the model list.
            continue
        agent_id, reasoning_mode, mtp_mode = decision

        selected_context = int(model.get("_picker_context") or _DEFAULT_CONTEXT)
        os.environ["CONTEXT"] = str(selected_context)
        base_name = _strip_latest(model["name"])
        # `::nothink` rides on the model name only when the user explicitly
        # chose reasoning OFF inside the info modal. The router's
        # parseReasoningOverride strips it and treats the request as
        # policy=off (enable_thinking=false / per-backend disable shape).
        reasoning_suffix = "::nothink" if reasoning_mode == "nothink" else ""
        # `::mtp` rides on the model name when MTP is opted in via the
        # sub-modal. Guarded by the same env flag that gates the column
        # and the sub-modal -- never emitted until Phase-5 router wiring
        # lands. Canonical emit order is `<name>::<reasoning>::<mtp>@<ctx>`
        # so the router's right-to-left parse chain (ctx -> mtp ->
        # reasoning) lines up cleanly.
        mtp_suffix = "::mtp" if (_MTP_PREVIEW and mtp_mode == "on") else ""
        if model["backend"] == "ollama":
            # Ollama: KV is dynamic per request; only the suffixes ride
            # on the model name.
            serving_name = f"{base_name}{reasoning_suffix}{mtp_suffix}"
        else:
            # vLLM / SGLang: order is `<name>::<reasoning>::<mtp>@<ctx>`.
            serving_name = f"{base_name}{reasoning_suffix}{mtp_suffix}@{selected_context}"

        cmd = _build(agent_id, serving_name, model["backend"])
        _record_pick(base_name, agent_id, selected_context)
        notes = []
        if reasoning_suffix:
            notes.append("no reasoning")
        if mtp_suffix:
            notes.append("MTP")
        mode_note = f" [{', '.join(notes)}]" if notes else ""
        print(
            f"\n  {_BOLD}{agent_id}{_RESET}"
            f" → {base_name}{mode_note} @ {_context_label(selected_context)}"
            f" via {_BACKENDS[model['backend']][0]}\n"
        )
        os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Ctrl-C raised between fzf invocations (rare — fzf intercepts
        # SIGINT itself). Exit silently so the container terminates
        # instead of dumping a traceback.
        sys.exit(130)
