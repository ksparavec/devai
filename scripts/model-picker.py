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
_DIM = "\033[38;5;245m"     # light-grey foreground (≈ #8a8a8a)
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
        except (OSError, json.JSONDecodeError):
            return {}
        records: dict[str, dict] = {}
        for _key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            for alias in entry.get("aliases") or []:
                records[alias] = entry
        return records
    return {}


def _migrate_in_memory(raw: dict) -> dict:
    """Run probe-ollama-reasoning's v1/v2 → v3 conversion without touching disk."""
    import importlib.util
    import sys
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    spec = importlib.util.spec_from_file_location(
        "_probe_module", here / "probe-ollama-reasoning.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_probe_module"] = mod  # frozen-dataclass workaround
    spec.loader.exec_module(mod)
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
    import re as _re
    upper = name.upper()
    total_match = _re.search(r"(?<![A-Z0-9])(\d+(?:\.\d+)?)B(?![A-Z0-9])", upper)
    if not total_match:
        return ""
    total = total_match.group(0)  # e.g. "14B"
    # MoE active-experts notation, e.g. "A3B" / "A4B" appearing after the
    # total params token.
    active = ""
    tail = upper[total_match.end():]
    active_match = _re.search(r"(?<![A-Z0-9])(A\d+(?:\.\d+)?B)(?![A-Z0-9])", tail)
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
    probes = _load_probe_records()
    out: list[dict] = []

    # Ollama: <library>/<tag> manifest files. Skip ctx-variant derived
    # tags — they're a per-session artefact created by the picker via
    # Modelfile to bake `num_ctx` in. The user picks the parent; the
    # picker materialises the variant on launch.
    import re as _re
    _ctx_tag = _re.compile(r"-ctx\d+$")
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
                cap = _capability_from_probe(probe, "unknown")
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
            if entry.get("capability") in ("error", "unsupported_arch"):
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
                        "capability": _capability_from_probe(probe, "unknown"),
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
                    "capability": "unknown",
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
                    "--color",
                    "fg:15,fg+:15,bg+:236,pointer:11,header:14,"
                    "hl:11,hl+:11,prompt:14,info:8,gutter:-1",
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
    # `none` and `unsupported` both render as "No reasoning" in the menu.
    # Cache distinguishes them: `none` = model produced a clean answer
    # without a reasoning parser attempted (e.g. Llama-3.1 — working as
    # designed); `unsupported` = parser was attempted but the model
    # didn't emit reasoning content (configuration mismatch).
    "none":        "·",
    "unsupported": "·",
    "unknown":     "?",  # not yet probed
    "error":       "✗",  # probe failed
}

_REASONING_LABEL = {
    "structured": "Native reasoning",
    "inline": "Inline reasoning",
    "none": "No reasoning",
    "unsupported": "No reasoning",
}

# Inline-reasoning models leak `<think>` blocks into content. The picker
# emits TWO rows for them — one that lets the model think (default), one
# that forces thinking off via the router's `::nothink` suffix. The
# router applies enable_thinking=false for inline+off requests.
_INLINE_OFF_LABEL = "Reasoning off"
_INLINE_OFF_GLYPH = "·"

_STATUS_ORDER = {
    "native": 0,
    "inline": 1,
    "inline_off": 2,
    "none": 3,
}

_MODE_ORDER = {"default": 0, "nothink": 1}


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


def _reasoning_variants(m: dict) -> list[tuple[str, str, str, str]]:
    """Map a model's probed capability to (status_key, glyph, label, mode)
    rows for the picker. Returns an empty list for unprobed / error
    capabilities so the caller can hide the row entirely.

    Inline-reasoning models produce TWO rows: a default row that lets the
    model think (router takes no action) and a `::nothink` row that
    forces enable_thinking=false. Structured and unsupported produce one
    row each.
    """
    cap = m.get("capability", "unknown")
    if cap == "structured":
        return [("native", _CAP_GLYPH[cap], _REASONING_LABEL[cap], "default")]
    if cap == "inline":
        return [
            ("inline", _CAP_GLYPH[cap], _REASONING_LABEL[cap], "default"),
            ("inline_off", _INLINE_OFF_GLYPH, _INLINE_OFF_LABEL, "nothink"),
        ]
    if cap in ("unsupported", "none"):
        return [("none", _CAP_GLYPH[cap], _REASONING_LABEL[cap], "default")]
    return []


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


def _format_model_row(m: dict) -> str:
    info = m.get("_picker_vram") or {}
    vram_num = "?"
    if info:
        suffix = "" if info.get("measured") else "*"
        vram_num = f"{info['total_gb']:.2f}{suffix}"

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

    fmt = details.get("quantization") or "?"
    tuning = _tuning_label(m)
    display_name = _strip_latest(m["name"])
    backend_col = str(m.get("backend") or "?")
    parser_col = str(m.get("tool_parser") or "N/A")

    return (
        f"      {ctx_str:>4s}  "
        f"{glyph} {status_label:<16s}  "
        f"{display_name:<32s}  "
        f"{backend_col:<7s}  "
        f"{params_col:>10s}  "
        f"{fmt:>8s}  "
        f"{tuning:>5s}  "
        f"{type_col:>5s}  "
        f"{parser_col:>14s}  "
        f"{vram_num:>10s}"
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


def _quality(m: dict) -> tuple[float, float, float]:
    """Quality proxy used to rank within a family. Higher tuple wins.

    Tuple components in priority order:
      1. parameter count — most meaningful "scale" signal (a 27B Q4
         beats a 9B BF16). Probe-supplied label first, then parsed
         from name when probe lacks the field (HF rows).
      2. actual VRAM at the picker's context — at fixed param count the
         higher-precision quant uses more memory and is preferred.
      3. raw disk size — last-resort tiebreaker for missing metadata.
    """
    params_b = _params_hint(m)
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
    # Track why models were hidden so the footer can explain. Categories:
    #   - context_capped: model's max_context is below the requested tier
    #   - missing_vram: no usable probe cell (band/ctx not yet probed)
    #   - missing_capability: capability is unknown/error/unsupported_arch
    #   - spilled: probe shows fully_on_gpu=false at this (vram, ctx)
    hidden = {
        "context_capped": 0,
        "missing_vram": 0,
        "missing_capability": 0,
        "spilled": 0,
    }
    for m in models:
        backend = m.get("backend", "ollama")
        for ctx in _CONTEXT_CHOICES:
            v = m.get("vram") or {}
            info = _vram_info_at(v, ctx)
            if info is None:
                hidden["missing_vram"] += 1
                continue
            if int(info.get("context") or 0) < ctx:
                hidden["context_capped"] += 1
                continue
            # Eliminate models that don't fit fully on GPU at this
            # (VRAM band, context). The cache record decides — we don't
            # second-guess it.
            if not info.get("fully_on_gpu", False):
                hidden["spilled"] += 1
                continue
            variants = _reasoning_variants(m)
            if not variants:
                hidden["missing_capability"] += 1
                continue
            # Family resolution priority:
            #   1. Catalog metadata (`m["family"]`) — set when the
            #      tag/repo appears in deploy/models.yaml.
            #   2. The Ollama tag's library prefix (text before the
            #      `:` in `<lib>:<tag>`) — handles locally-renamed
            #      tags like `nemotron-3-nano:4b-q4_K_M` that we
            #      created via `ollama cp` and that have no upstream
            #      registry entry. The library prefix matches the
            #      family name 1:1 in this project.
            #   3. "(uncategorized)" — last-resort fallback so the row
            #      still appears in the menu.
            family = m.get("family") or ""
            if not family and m.get("source") == "ollama":
                tag_name = m.get("name") or ""
                if ":" in tag_name:
                    family = tag_name.split(":", 1)[0]
            if not family:
                family = "(uncategorized)"
            family_rows = grouped.setdefault(backend, {}).setdefault(family, {})
            for status_key, glyph, status_label, mode in variants:
                candidate = dict(m)
                candidate["_picker_context"] = ctx
                candidate["_picker_vram"] = info
                candidate["_picker_status"] = status_key
                candidate["_picker_glyph"] = glyph
                candidate["_picker_status_label"] = status_label
                candidate["_picker_mode"] = mode

                # Bucket key includes mode so default and ::nothink rows
                # for the same inline model coexist as separate picks.
                bucket = (ctx, status_key, mode)
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
        f"{'BACKEND':<7s}  "
        f"{'PARAMS':>10s}  "
        f"{'FORMAT':>8s}  "
        f"{'TUNE':>5s}  "
        f"{'TYPE':>5s}  "
        f"{'PARSER':>14s}  "
        f"{'VRAM (GB)':>10s}"
    )
    for backend in _PICKER_BACKENDS:
        if backend not in grouped:
            continue
        if not first_section:
            emit("", selectable_=False, model=None)
        first_section = False
        label = _BACKENDS[backend][0]
        emit(f"  {_BOLD}── {label} ──{_RESET}",
             selectable_=False, model=None)
        # The column header carries its own leading whitespace that
        # matches `_format_model_row` exactly. Don't add a `  ` prefix
        # here — that would shift the header 2 chars right of every
        # data row and the columns would look misaligned.
        emit(f"{_BOLD}{column_header}{_RESET}",
             selectable_=False, model=None)
        if backend == "ollama":
            note = (f"* = formula estimate (no probe cell at this VRAM/ctx — "
                    f"run `make probe-{backend}`)")
        else:
            # vLLM and SGLang engines pre-allocate gpu_memory_utilization ×
            # total_vram, so the probe records the same actual_vram_gb
            # for every (vram, ctx) cell — useless for per-ctx display.
            # The picker overrides with a formula breakdown
            # (weights + KV(arch, ctx) + 3 GB overhead) and marks it `*`.
            # Real measured per-ctx VRAM would require a long-prompt
            # probe with engine memory utilisation tuned per cell — not
            # implemented yet.
            note = (f"* = formula estimate (weights + KV(arch, ctx) + 3 GB; "
                    f"{backend} probe doesn't measure per-ctx VRAM directly)")
        emit(f"  {_DIM}{note}{_RESET}",
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
                    _MODE_ORDER.get(str(m.get("_picker_mode") or "default"), 99),
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
        if hidden["missing_vram"]:
            bits.append(f"{hidden['missing_vram']} missing VRAM data")
        if hidden["context_capped"]:
            bits.append(f"{hidden['context_capped']} below requested context")
        if hidden["spilled"]:
            bits.append(f"{hidden['spilled']} spill to CPU/RAM at this VRAM")
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
        # User cancelled (Ctrl-C / Esc) before picking a model. Exit
        # the picker cleanly so the container's PID 1 (this process)
        # terminates and `podman run --rm` reclaims the container —
        # same lifecycle as quitting an agent. Dropping to bash here
        # would keep the container alive indefinitely.
        sys.exit(0)
    model = item_models[idx]
    if model is None:  # defensive — _fzf already filters headers
        sys.exit(0)
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
            f"{_BOLD}{_strip_latest(model['name'])}{_RESET}"
            f" @ {_context_label(selected_context)}"
            f" via {_BACKENDS[model['backend']][0]}"
        )
        idx = _fzf(alines, header)
        if idx is None:
            # User cancelled at the agent step. Exit cleanly so the
            # container terminates instead of dropping into bash.
            sys.exit(0)
        agent = _AGENTS[idx]

    # Per-session context binding — only for vLLM / SGLang. The router
    # (gpu-arbiter) parses the `@<ctx>` suffix via parseCtxOverride and
    # recreates the backend container with --max-model-len (vLLM) or
    # --context-length (SGLang) set accordingly. KV pool is allocated
    # at startup; the suffix drives that allocation.
    #
    # Ollama is different: KV is allocated dynamically per request, and
    # the picker emits just the parent name. Per-request num_ctx for
    # native /api/chat is injected by the router's setNumCtx (still
    # honoured by Ollama on the native API). Agents that hit Ollama via
    # /v1/chat/completions or /v1/messages get the global
    # OLLAMA_CONTEXT_LENGTH env (an Ollama upstream limitation — the
    # /v1/* paths ignore options.num_ctx).
    base_name = _strip_latest(model["name"])
    # Append `::nothink` for the inline-reasoning forced-off pick. The
    # router's parseReasoningOverride strips this suffix and treats
    # the request as policy=off, injecting enable_thinking=false (or
    # the equivalent per-backend disable shape). The user gets
    # think-disabled output without the model's <think> blocks.
    reasoning_suffix = ""
    if model.get("_picker_mode") == "nothink":
        reasoning_suffix = "::nothink"
    if model["backend"] == "ollama":
        # Ollama: KV is dynamic per request; only the reasoning suffix
        # rides on the model name. No `@<ctx>` needed.
        serving_name = f"{base_name}{reasoning_suffix}"
    else:
        # vLLM / SGLang: order is `<name>::<reasoning>@<ctx>` so the
        # router's parseCtxOverride (which strips trailing @<int>)
        # runs cleanly before parseReasoningOverride.
        serving_name = f"{base_name}{reasoning_suffix}@{selected_context}"

    cmd = _build(agent[0], serving_name, model["backend"])
    _record_pick(base_name, agent[0], selected_context)
    mode_note = " [no reasoning]" if reasoning_suffix else ""
    print(
        f"\n  {_BOLD}{agent[0]}{_RESET}"
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
