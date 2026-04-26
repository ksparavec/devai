#!/usr/bin/env python3
"""Two-step picker for DevAI: pick a downloaded model, then pick an agent.

Discovery is disk-only — the picker walks the host model cache that is
mounted into the container and lists exactly what is on disk. The yaml
catalog (deploy/models.yaml) is consulted only to enrich display metadata
(purpose). It is never used to decide whether a model is "downloaded",
and the picker performs no VRAM-fit math: that gate happens upstream in
`select-models.py` before a model ever reaches disk.

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

# User-chosen context for VRAM display + filtering. Picker re-computes
# total VRAM at this context using probe-derived coefficients
# (weights_overhead_gb + ctx × kv_per_token_bytes / 1024^3). Defaults to
# the same value select-models.py uses so behavior matches what was
# written into active-models.yaml. Override via `CONTEXT=32768 make
# shell-gpu` to widen the choice set, or `CONTEXT=262144` to narrow it.
_CONTEXT = int(os.environ.get("CONTEXT", os.environ.get("MAX_CONTEXT_LEN", "131072")))

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
    thinking-flag."""
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


def _discover_models() -> list[dict]:
    """Walk the cache dirs and return one entry per model on disk.

    Disk is the source of truth for existence. Catalog metadata (purpose,
    declared size) and per-model active data (VRAM breakdown + runtime-
    probed reasoning capability) are layered on top when available — never
    recomputed here.
    """
    catalog = _load_catalog()
    active = _load_active_entries()
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
                cap = (act.get("reasoning") or {}).get("capability", "unknown")
                out.append({
                    "name": name,
                    "source": "ollama",
                    "backend": "ollama",
                    "size": meta.get("size") or f"{disk_gb:.2f} GB",
                    "purpose": meta.get("purpose", ""),
                    "vram": act.get("vram"),
                    "family": meta.get("family", ""),
                    "capability": cap,
                    "moe": act.get("moe"),
                    "details": act.get("details") or {},
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


def _vram_at(v: dict, context: int) -> tuple[float | None, int]:
    """Compute (total_gb, effective_context) for the model at CONTEXT.

    effective_context = min(context, max_context) — the model's design
    ceiling is a hard physical limit, not a runtime knob. A 128K-only
    model asked to run at 256K simply runs at 128K (its max), and the
    picker shows that honestly. This is NOT a runtime clamp like
    OLLAMA_CONTEXT_LENGTH; it's a fact about the model's architecture.

    Returns (None, 0) when no coefficients are present (unprobed model).
    """
    if not v:
        return None, 0
    weights = v.get("weights_overhead_gb")
    kv_pt = v.get("kv_per_token_bytes")
    if weights is None or kv_pt is None:
        return None, 0
    max_ctx = v.get("max_context") or 0
    eff_ctx = min(context, max_ctx) if max_ctx else context
    total = weights + (kv_pt * eff_ctx) / (1024**3)
    return round(total, 2), eff_ctx


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
    v = m.get("vram") or {}
    src = v.get("source")
    total, eff_ctx = _vram_at(v, _CONTEXT)
    if total is None:
        vram_num = "?"
        ctx_str = "?"
    else:
        suffix = "*" if src == "formula" else ""
        vram_num = f"{total:.2f}{suffix}"
        ctx_k = eff_ctx // 1024
        ctx_str = f"{ctx_k}K"

    cap = m.get("capability", "unknown")
    glyph = _CAP_GLYPH.get(cap, _CAP_GLYPH["unknown"])

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
        f"      {glyph} {m['name']:<32s}  "
        f"{params_col:>10s}  "
        f"{quant:>8s}  "
        f"{tuning:>5s}  "
        f"{type_col:>5s}  "
        f"{vram_num:>10s}  "
        f"{ctx_str:>8s}"
    )


def _name_priority(m: dict) -> tuple:
    """Tag-name priority for the dedup tiebreak.

    Higher tuple = preferred. Ranking:
      1. anything-but-:latest beats :latest (moving alias, no info)
      2. longer name beats shorter (more explicit -q4_K_M / -a3b suffix)
    """
    name = m["name"]
    return (not name.endswith(":latest"), len(name))


def _quality(m: dict) -> float:
    """Quality proxy used to rank within a family. Larger weights → better.

    Coefficient-derived `weights_overhead_gb` is the cleanest signal
    (intercept = weights + small runtime overhead, KV-free). Older
    formula-only entries fall back to `weights_gb`, then raw disk size.
    """
    v = m.get("vram") or {}
    if v.get("weights_overhead_gb") is not None:
        return float(v["weights_overhead_gb"])
    if v.get("weights_gb"):
        return float(v["weights_gb"])
    try:
        return float((m.get("size") or "0").split()[0])
    except (ValueError, IndexError):
        return 0.0


# Was capped at 3; the strict-structured filter already cuts the list
# enough that an artificial cap loses real choices. Dedup-by-VRAM-bucket
# below still collapses literal aliases (same digest, different tag).
_TOP_PER_FAMILY = None  # None = no cap


def _build_menu(models: list[dict]) -> tuple[list[str], list[bool], list[dict | None]]:
    """Build (display_lines, selectable_flags, model_per_line) for fzf.

    Shows ALL fitting models, grouped by backend then family. Per row a
    capability glyph (●/◐/·/?/✗) reflects the runtime probe — no filter
    by capability (per docs/ollama_models.md). Within each family, top N
    by quality (weights_gb desc), deduped on weight bucket so the user
    sees distinct sizes rather than 3 quants of the same model.
    """
    # backend → family → [models]. Three strict filters:
    #   1. must have probe-derived coefficients — absence means
    #      the model wasn't probed.
    #   2. capability must be `structured` — runtime probe must have
    #      verified the model exposes its reasoning in a separate
    #      message field. Hides:
    #         · unsupported (no reasoning observed)
    #         ◐ inline      (reasoning leaks into content as <think>…</think>)
    #         ? unknown     (not yet probed — incl. all vLLM/SGLang)
    #         ✗ error       (probe failed; e.g. ollama HTTP 400 on think:)
    #   3. interpolated total at effective_ctx (= min(CONTEXT, max_ctx))
    #      must fit in VRAM_BUDGET. Models with max_ctx < CONTEXT are
    #      shown with their effective context, not hidden.
    grouped: dict[str, dict[str, list[dict]]] = {}
    # Track why models were hidden so the footer can explain. The big
    # categories worth distinguishing today:
    #   - non_ollama_unprobed: every vLLM/SGLang entry (capability=unknown)
    #     because there is no probe runner for those backends yet — they
    #     are also dormant per docs/sidelined-backends.md
    #   - non_structured: ollama models the probe classified as
    #     unsupported / inline / error
    #   - over_budget: structured but interpolated VRAM > VRAM_BUDGET
    #   - missing_coeffs: probe never ran or didn't yield coefficients
    hidden = {
        "non_ollama_unprobed": 0,
        "non_structured": 0,
        "over_budget": 0,
        "missing_coeffs": 0,
    }
    for m in models:
        v = m.get("vram") or {}
        cap = m.get("capability", "unknown")
        backend = m.get("backend", "ollama")
        if not v or v.get("weights_overhead_gb") is None:
            if backend != "ollama":
                hidden["non_ollama_unprobed"] += 1
            else:
                hidden["missing_coeffs"] += 1
            continue
        if cap != "structured":
            if backend != "ollama":
                hidden["non_ollama_unprobed"] += 1
            else:
                hidden["non_structured"] += 1
            continue
        total, _ = _vram_at(v, _CONTEXT)
        if total is None or total > _VRAM_BUDGET:
            hidden["over_budget"] += 1
            continue
        family = m.get("family") or "(uncategorized)"
        grouped.setdefault(backend, {}).setdefault(family, []).append(m)

    lines: list[str] = []
    selectable: list[bool] = []
    item_models: list[dict | None] = []

    def emit(text: str, *, selectable_: bool, model: dict | None) -> None:
        lines.append(text)
        selectable.append(selectable_)
        item_models.append(model)

    first_section = True
    # Column header is emitted with a 2-char prefix from emit(); the leading
    # 6 spaces here align it with data rows whose prefix is "      ● " (6
    # spaces + 1 glyph + 1 space = 8 chars before the TAG field begins).
    column_header = (
        f"      {'TAG':<32s}  "
        f"{'PARAMS':>10s}  "
        f"{'QUANT':>8s}  "
        f"{'TUNE':>5s}  "
        f"{'TYPE':>5s}  "
        f"{'VRAM (GB)':>10s}  "
        f"{'CONTEXT':>8s}"
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

        # Order families within this backend by their best variant.
        families = sorted(
            grouped[backend].items(),
            key=lambda kv: -max(_quality(m) for m in kv[1]),
        )
        for fam, fmodels in families:
            # Dedupe on VRAM bucket (1-decimal). When several tags share
            # the same bucket (almost always literal aliases of one blob —
            # e.g. qwen3.5:9b, qwen3.5:9b-q4_K_M and qwen3.5:latest all
            # map to the same digest) we keep the most INFORMATIVE name:
            #   - never :latest (it's a moving alias, says nothing about
            #     parameters / quantization)
            #   - otherwise prefer the longest name (longer = more
            #     explicit suffixes like -q4_K_M, -a3b, -it-bf16)
            seen: dict[float, dict] = {}
            for m in sorted(fmodels, key=_quality, reverse=True):
                bucket = round(_quality(m), 1)
                if bucket not in seen or _name_priority(m) > _name_priority(seen[bucket]):
                    seen[bucket] = m
            ranked = sorted(seen.values(), key=_quality, reverse=True)
            top = ranked if _TOP_PER_FAMILY is None else ranked[:_TOP_PER_FAMILY]
            emit(f"    {_BOLD}{fam}:{_RESET}",
                 selectable_=False, model=None)
            for m in top:
                emit(_format_model_row(m), selectable_=True, model=m)

    # Footer: explain hidden rows so a user with NVFP4 weights on disk
    # (or ollama models the probe rejected) knows why they don't appear.
    total_hidden = sum(hidden.values())
    if total_hidden:
        emit("", selectable_=False, model=None)
        bits: list[str] = []
        if hidden["non_ollama_unprobed"]:
            bits.append(f"{hidden['non_ollama_unprobed']} vLLM/SGLang (dormant)")
        if hidden["non_structured"]:
            bits.append(f"{hidden['non_structured']} non-structured")
        if hidden["missing_coeffs"]:
            bits.append(f"{hidden['missing_coeffs']} unprobed")
        if hidden["over_budget"]:
            bits.append(f"{hidden['over_budget']} over VRAM budget")
        emit(f"  {_DIM}hidden: {', '.join(bits)}  ·  "
             f"see docs/sidelined-backends.md{_RESET}",
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
        cmd = ["late"]
        if model_name.startswith("Gemma-4-"):
            cmd.append("--gemma-thinking")
        return cmd

    if agent_id == "interpreter":
        # `/no_think` is a Qwen3 hint to skip <think>…</think> reasoning
        # blocks (which OI doesn't render — content goes to reasoning_content).
        # Harmless for non-Qwen models. We pass it as a system-prompt suffix
        # so OI doesn't treat the prompt itself as a slash-command.
        custom = "/no_think"
        if backend == "ollama":
            return [
                "interpreter", "--model", f"ollama/{model_name}",
                "--custom_instructions", custom,
            ]
        return [
            "interpreter", "--model", f"openai/{model_name}",
            "--api_base", f"{base}/v1",
            "--api_key", "local",
            "--custom_instructions", custom,
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

    # Step 1 — pick a model. Top N per family ranked by weight, with the
    # runtime-probed reasoning capability shown as a glyph next to each
    # row. ●=structured ◐=inline ·=unsupported ?=unknown ✗=error.
    lines, selectable, item_models = _build_menu(models)
    if not any(selectable):
        sys.exit(
            f"error: no fitting models on disk at {_CONTEXT // 1024}K ctx, "
            f"≤ {_VRAM_BUDGET:g} GB.\n"
            f"  Try a smaller context: CONTEXT=32768 …\n"
            f"  Or raise the VRAM budget: VRAM=48 …\n"
            f"  Or pull smaller models: make ollama-pull MODEL=…"
        )
    header = (
        f"DevAI  ▸  Step 1/2: pick a model  "
        f"(structured-reasoning · {_CONTEXT // 1024}K ctx · "
        f"≤ {_VRAM_BUDGET:g} GB)"
    )
    idx = _fzf(lines, header, selectable=selectable)
    if idx is None:
        os.execvp("bash", ["bash"])
    model = item_models[idx]
    if model is None:  # defensive — _fzf already filters headers
        os.execvp("bash", ["bash"])

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
            f" via {_BACKENDS[model['backend']][0]}"
        )
        idx = _fzf(alines, header)
        if idx is None:
            os.execvp("bash", ["bash"])
        agent = _AGENTS[idx]

    cmd = _build(agent[0], model["name"], model["backend"])
    print(
        f"\n  {_BOLD}{agent[0]}{_RESET}"
        f" → {model['name']} via {_BACKENDS[model['backend']][0]}\n"
    )
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
