"""Backend-agnostic HF (vLLM/SGLang) probe scaffold.

Shared by scripts/probe-vllm-reasoning.py and
scripts/probe-sglang-reasoning.py. Each backend supplies a
`BackendSpec` (image, reservation, launch-arg builder) and gets a
complete prober — argparse, catalog filter, podman driver, /v1/models
+ /v1/chat/completions probe, nvidia-smi snapshot, fits classifier,
implied-fail propagation, cache writer.

What stays per-backend:
  - The launch command (different module, different flags)
  - The image default
  - The memFraction reservation (vLLM: 2.0 GB, SGLang: 3.0 GB)
  - The default cache filename and probe container name

What is shared (and lives here):
  - Catalog parsing / disk presence check
  - Container lifecycle helpers (podman CLI shellouts)
  - GPU memory snapshot via nvidia-smi
  - OpenAI-shape chat response classifier
  - Failure log pattern matcher
  - Cache schema v1 navigation (ensure_entry, reflect-on-first-cell)
  - The main per-row probe loop with implied-fail propagation
  - Hard precondition: refuse to run when any GPU-owning backend
    container is up (router / vllm / sglang)

Cache schema is identical to vLLM's — see probe-vllm-reasoning.py
docstring. SGLang gets a separate cache file but the same shape.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from _contexts import (  # noqa: E402  — local import after sys.path fix
    BINARY_SEARCH_CONTEXTS,
    binary_search_max_ctx,
    context_label,
    parse_context_list,
    parse_vram_list,
    standard_contexts,
    standard_vram_budgets,
    vram_label,
)
import _card_hints  # noqa: E402
import _vllm_recipes  # noqa: E402
from _capability import Capability, is_terminal  # noqa: E402
from _probe_core import (  # noqa: E402
    has_inline_think_markers,
    http_get,
    http_post,
    image_digest_via_cli,
    load_cache,
    now_iso,
    save_cache,
    smallest_clean_probe,
    row_image_is_stale,
    stamp_image_digest,
    ROW_IMAGE_FIELD,
    backfill_row_images,
    stamp_row_image,
)
from _vllm_plugins import PluginEntry, PluginRegistry, get_registry  # noqa: E402
from _model_status import (  # noqa: E402
    clear as _ledger_clear,
    exclusion_reason as _ledger_reason,
    implied_vram_exclusion as _ledger_implied_vram_exclusion,
    is_excluded as _ledger_is_excluded,
    load_ledger as _load_ledger,
    record_exclusion as _ledger_record,
    save_ledger as _save_ledger,
)


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_CATALOG = REPO_ROOT / "deploy" / "models.yaml"
DEFAULT_MODELS_DIR = os.environ.get(
    "VLLM_MODELS_DIR", "/var/cache/devai/vllm"
)
DEFAULT_PROMPT = (
    "Solve this step by step. Show your reasoning, then state the final number. "
    "Problem: A train travels 60 miles in the first hour, then increases speed "
    "by 20 mph each hour. How far has it traveled after 3 hours?"
)

# Probe B (tool-call) — minimal tool spec that any backend supporting
# OpenAI-style tool calling should be able to invoke. Deterministic,
# parameterless, no model-side guessing.
TOOL_PROBE_PROMPT = "Use the get_time tool to fetch the current time."
TOOL_PROBE_SPEC: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Return the current server time.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]

# Probe C (disable verification) — same prompt as Probe A but with the
# backend-specific switch that should suppress structured reasoning.
# Only meaningful when Probe A produced `structured`; else skipped.
DISABLE_PROBE_PROMPT = DEFAULT_PROMPT
DEFAULT_HOST_VRAM_GB = int(os.environ.get("GPU_MEMORY_GB", "24"))

# Cold start with CUDA graph capture takes 60–300s on consumer hardware.
# Both vLLM and SGLang share this ceiling.
STARTUP_TIMEOUT = float(os.environ.get("HF_PROBE_STARTUP_SECONDS", "600"))
HEALTH_POLL_INTERVAL = 5.0
CHAT_TIMEOUT = float(os.environ.get("HF_PROBE_CHAT_SECONDS", "60"))
MEM_BUDGET_SLACK = 1.05

# Containers that must be down before a probe runs — they all hold the
# single GPU and would race the prober's launch. Keep this list in sync
# with deploy/docker-compose.yaml service names: every container that can
# hold the GPU belongs here. devai-vllm-devai and devai-ollama were missing
# until the laya trainer (devai-laya-trainer) was added (2026-09-24).
MUTEX_CONTAINERS = ("devai-router", "devai-vllm", "devai-vllm-devai", "devai-sglang",
                    "devai-ollama", "devai-laya-trainer")


# ── Recovery flags registry ──────────────────────────────────────────────────
# Per-model launch flags + env overrides shared with the router. The same two
# files gpu-arbiter/recovery_flags.go reads:
#
#   deploy/recovery-flags.json            hand-curated, one entry per model
#                                         (contract C2, _entry_applies_to_backend)
#   deploy/recovery-flags.generated.json  machine-made (plan I25): per model a
#                                         list of groups {backends (required),
#                                         engine_flags, engine_env, source,
#                                         reason, added}; written only by
#                                         scripts/recipe-apply.py today, whole
#                                         file to a temporary name and renamed
#
# For a launch on backend B: the curated entry's flags/env when it applies to
# B, then every generated group whose `backends` names B or B's engine, in
# file order (recovery_overrides). Both files are re-read before each launch
# when either one's device, inode, mtime or size changed (_refresh_recovery).
# A file that fails to parse keeps what was loaded before and warns; so does
# a curated file that disappears after a good load (git checkout replaces a
# file by unlink and create). A missing generated file reads as empty.
#
# The decoding rules are those of recovery_flags.go's header (exact keys; a
# top level that is not an object fails the file; an entry or group with a
# malformed field is skipped whole; a curated non-list `backends` is C2's
# "treat as absent"), and the two readers' verdicts are pinned by the shared
# fixture in tests/fixtures/recovery-flags/, which both test suites load.
RECOVERY_FLAGS_PATH = Path(
    os.environ.get("RECOVERY_FLAGS_REGISTRY", str(REPO_ROOT / "deploy" / "recovery-flags.json"))
)
RECOVERY_GENERATED_PATH = Path(
    os.environ.get("RECOVERY_FLAGS_GENERATED",
                   str(RECOVERY_FLAGS_PATH.parent / "recovery-flags.generated.json"))
)
RECOVERY_GENERATED_VERSION = 1


def _strict_json(text: str) -> object:
    """json.loads, but refusing NaN / Infinity, which Go's decoder rejects."""
    def _refuse(token: str) -> object:
        raise ValueError(f"non-standard JSON constant {token}")
    return json.loads(text, parse_constant=_refuse)


def _is_str_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _is_str_map(value: object) -> bool:
    return isinstance(value, dict) and all(isinstance(v, str) for v in value.values())


def _curated_entry_problem(entry: object) -> str | None:
    """Why a curated entry is skipped whole, or None when it is usable.

    `engine_flags` absent, null or a list of strings; `engine_env` absent,
    null or an object of strings; `image` absent, null or a string; a
    `backends` LIST must hold strings only (a non-list `backends` is C2's
    "treat as absent", decided in _entry_applies_to_backend)."""
    if not isinstance(entry, dict):
        return "not an object"
    flags = entry.get("engine_flags")
    if flags is not None and not _is_str_list(flags):
        return "`engine_flags` must be a list of strings"
    env = entry.get("engine_env")
    if env is not None and not _is_str_map(env):
        return "`engine_env` must be an object of strings"
    image = entry.get("image")
    if image is not None and not isinstance(image, str):
        return "`image` must be a string"
    backends = entry.get("backends")
    if isinstance(backends, list) and not _is_str_list(backends):
        return "`backends` must list strings"
    return None


def _generated_group_problem(group: object) -> str | None:
    """Why a generated group is skipped whole, or None when it is usable:
    `backends` a list of strings (required, no "every backend" group, I25);
    `engine_flags` and `engine_env` as in a curated entry. `source`,
    `reason` and `added` are informational and not checked."""
    if not isinstance(group, dict):
        return "not an object"
    if not _is_str_list(group.get("backends")):
        return "`backends` must be a list of strings"
    flags = group.get("engine_flags")
    if flags is not None and not _is_str_list(flags):
        return "`engine_flags` must be a list of strings"
    env = group.get("engine_env")
    if env is not None and not _is_str_map(env):
        return "`engine_env` must be an object of strings"
    return None


def _read_top_level(path: Path, label: str) -> tuple[dict, dict]:
    """(top-level object, its `models` map) of a recovery file. A JSON
    error, a top level that is not an object (`null` included) or a
    `models` that is neither absent/null nor an object raises ValueError.
    FileNotFoundError propagates; so does any other OSError, which means
    the operator's environment is broken in a way that would otherwise make
    every model launch without its recovery flags -- the prober aborts
    loudly instead of writing fits=false for models that would have fit."""
    try:
        data = _strict_json(path.read_text())
    except ValueError as exc:  # json.JSONDecodeError is a ValueError
        raise ValueError(f"{label} {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} {path}: not a JSON object")
    models = data.get("models")
    if models is None:
        models = {}
    if not isinstance(models, dict):
        raise ValueError(f"{label} {path}: `models` is not an object")
    return data, models


def _read_curated_recovery(path: Path) -> dict[str, dict]:
    """Read deploy/recovery-flags.json -> {model_name: entry}. A malformed
    entry is skipped on its own with a warning; FileNotFoundError and
    ValueError (see _read_top_level) propagate."""
    _, models = _read_top_level(path, "recovery registry")
    out: dict[str, dict] = {}
    for name, entry in models.items():
        problem = _curated_entry_problem(entry)
        if problem:
            print(f"[warn] recovery registry: entry {name!r} skipped ({problem})",
                  file=sys.stderr)
            continue
        out[name] = entry
    return out


def _read_generated_recovery(path: Path) -> dict[str, list[dict]]:
    """Read deploy/recovery-flags.generated.json -> {model_name: [group, ...]}.

    Missing file -> {}. Besides _read_top_level's failures, a `version`
    other than the integer 1 raises ValueError (the caller keeps what it
    had). A model whose value is not a list, or a malformed group, is
    skipped on its own with a warning. Group order is kept: it is the order
    the flags are appended in.
    """
    try:
        data, models = _read_top_level(path, "generated recovery flags")
    except FileNotFoundError:
        return {}
    version = data.get("version")
    if type(version) is not int or version != RECOVERY_GENERATED_VERSION:
        raise ValueError(f"generated recovery flags {path}: `version` must be "
                         f"the integer {RECOVERY_GENERATED_VERSION}")
    out: dict[str, list[dict]] = {}
    for name, groups in models.items():
        if not isinstance(groups, list):
            print(f"[warn] generated recovery flags: {name}: not a list of "
                  f"groups; skipped", file=sys.stderr)
            continue
        kept = []
        for i, group in enumerate(groups):
            problem = _generated_group_problem(group)
            if problem:
                print(f"[warn] generated recovery flags: {name} group {i}: "
                      f"{problem}; group skipped", file=sys.stderr)
                continue
            if "image" in group:
                print(f"[warn] generated recovery flags: {name} group {i}: a "
                      f"generated group may not carry an `image`; ignored",
                      file=sys.stderr)
            kept.append(group)
        if kept:
            out[name] = kept
    return out


def _file_sig(path: Path) -> tuple[int, int, int, int] | None:
    """(device, inode, mtime_ns, size), or None for a missing file. A new
    inode (a writer's rename, git checkout's unlink and create) or an
    in-place write changes it."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size)


_UNSEEN = object()
_RECOVERY_REGISTRY: dict[str, dict] = {}
_RECOVERY_GENERATED: dict[str, list[dict]] = {}
# Signature of each file as last loaded successfully: [curated, generated].
_RECOVERY_SIGS: list = [_UNSEEN, _UNSEEN]


def _refresh_recovery() -> None:
    """Reload whichever recovery file changed since its last good load.

    Called before every launch. A file that fails to parse keeps the
    previously loaded content and warns; its signature is not advanced, so
    the next launch tries again. A curated file that is missing at the
    first load reads as empty (with a warning); one that disappears after a
    good load keeps the previous entries. A missing generated file reverts
    what it added (I25).
    """
    global _RECOVERY_REGISTRY, _RECOVERY_GENERATED
    sig = _file_sig(RECOVERY_FLAGS_PATH)
    if sig != _RECOVERY_SIGS[0]:
        first = _RECOVERY_SIGS[0] is _UNSEEN
        try:
            _RECOVERY_REGISTRY = _read_curated_recovery(RECOVERY_FLAGS_PATH)
            _RECOVERY_SIGS[0] = sig
        except FileNotFoundError:
            if first:
                print(f"[warn] recovery registry {RECOVERY_FLAGS_PATH} not present "
                      f"-- no per-model overrides", file=sys.stderr)
                _RECOVERY_REGISTRY, _RECOVERY_SIGS[0] = {}, None
            else:
                print(f"[warn] recovery registry {RECOVERY_FLAGS_PATH} disappeared; "
                      f"keeping the previously loaded curated entries "
                      f"({len(_RECOVERY_REGISTRY)})", file=sys.stderr)
        except ValueError as exc:
            print(f"[warn] {exc}; keeping the previously loaded curated "
                  f"entries ({len(_RECOVERY_REGISTRY)})", file=sys.stderr)
    sig = _file_sig(RECOVERY_GENERATED_PATH)
    if sig != _RECOVERY_SIGS[1]:
        try:
            _RECOVERY_GENERATED = _read_generated_recovery(RECOVERY_GENERATED_PATH)
            _RECOVERY_SIGS[1] = sig
        except ValueError as exc:
            print(f"[warn] {exc}; keeping the previously loaded generated "
                  f"groups ({len(_RECOVERY_GENERATED)} models)", file=sys.stderr)


_refresh_recovery()


_WARNED_BAD_BACKENDS: set[str] = set()


def _entry_applies_to_backend(
    entry: dict, backend: str, model_name: str = "",
) -> bool:
    """Backend gate for a recovery entry (cross-unit contract C2).

    The agreed semantics, implemented identically by
    gpu-arbiter/recovery_flags.go:

      - key ABSENT           -> applies to ALL backends (backward
        compatible with every entry written before the key existed).
      - key present, ``[]``  -> applies to NO backend (an operator
        writing an empty list means "disable this entry").
      - key present, a list  -> applies only to the named backends --
        vLLM-only flags like ``--language-model-only`` or
        ``--quantization modelopt`` are not valid SGLang arguments and
        would fail the launch outright.
      - key present, NOT a list -> malformed; warn (naming the model)
        and treat as ABSENT rather than silently dropping the entry's
        recovery flags, which on 24G cards typically means an OOM.

    Keep this in sync with the Go side.
    """
    backends = entry.get("backends")
    if backends is None:  # absent, or an explicit null: no restriction
        return True
    if not isinstance(backends, list):
        label = model_name or "<unknown model>"
        if label not in _WARNED_BAD_BACKENDS:
            _WARNED_BAD_BACKENDS.add(label)
            print(f"[warn] recovery registry: {label}: \"backends\" must be a "
                  f"list, got {type(backends).__name__}; treating the entry as "
                  f"applying to all backends", file=sys.stderr)
        return True
    # By name or by engine (kept in sync with recovery_flags.go appliesTo):
    # vLLM CLI flags written for `vllm` apply to the custom vLLM build too;
    # an entry scoped to `vllm-devai` names the only image that serves it.
    return backend in backends or engine_of(backend) in backends


def _recovery_entry(model_name: str, backend: str) -> dict | None:
    """Curated registry entry for (model_name, backend), or None when absent
    or gated out by the entry's `backends` list."""
    entry = _RECOVERY_REGISTRY.get(model_name)
    if not entry or not _entry_applies_to_backend(entry, backend, model_name):
        return None
    return entry


def _flags_env(item: dict) -> tuple[list[str], dict[str, str]]:
    flags = item.get("engine_flags") or []
    env = item.get("engine_env") or {}
    return (list(flags) if isinstance(flags, list) else [],
            dict(env) if isinstance(env, dict) else {})


def recovery_overrides(
    model_name: str, backend: str,
) -> tuple[list[str], dict[str, str]]:
    """Return (extra_flags, extra_env) for model_name on `backend`.

    The curated entry's flags/env when it applies to `backend` (contract
    C2), then every generated group whose `backends` names `backend` or its
    engine, in file order (I25); later env keys win, and the engine takes
    the last of a repeated flag. Both files are re-read first when changed.
    Empty when nothing applies.
    """
    _refresh_recovery()
    flags: list[str] = []
    env: dict[str, str] = {}
    entry = _recovery_entry(model_name, backend)
    if entry:
        f, e = _flags_env(entry)
        flags += f
        env.update(e)
    for group in _RECOVERY_GENERATED.get(model_name, []):
        if _entry_applies_to_backend(group, backend, model_name):
            f, e = _flags_env(group)
            flags += f
            env.update(e)
    return flags, env


def recovery_image(model_name: str, backend: str) -> str | None:
    """Return the per-model container-image override, or None.

    Mirrors the `image` field consumed by gpu-arbiter/recovery_flags.go:
    a model whose recovery entry pins a non-default engine image must be
    PROBED on that same image, otherwise the probe launches the global
    default engine, fails to load the checkpoint, and records a spurious
    fits=false cell -- hiding the model from the picker. Keeps probe-time
    and serve-time launches on the same engine (the registry's
    single-source-of-truth contract). DiffusionGemma is the first user:
    it needs the vLLM "gemma" build, which can't be the global default
    because it regresses Qwen NVFP4 loading.

    Backend-gated the same way as recovery_overrides (contract C2): an
    entry pinning a vLLM-only image must not redirect an SGLang launch.
    Curated file only: a generated group carries no image (I25).
    """
    _refresh_recovery()
    entry = _recovery_entry(model_name, backend)
    if not entry:
        return None
    img = entry.get("image")
    return img if isinstance(img, str) and img else None


# ── Backend spec ─────────────────────────────────────────────────────────────

# Backend NAME -> ENGINE. Mirrors gpu-arbiter's engineOf: names key ports,
# containers, caches and rows; the engine decides behaviour. `vllm-devai` is
# the home-built, HyperQwen-patched vLLM 0.28.0 image on its own port/cache.
_ENGINE_OF = {"vllm-devai": "vllm"}


def engine_of(backend: str) -> str:
    return _ENGINE_OF.get(backend, backend)


def curated_parsers(row: dict, backend: str) -> tuple[str | None, str | None]:
    """(reasoning, tool) from the catalog row's `parsers:` block for a backend.

    The block is keyed by ENGINE (`vllm`, `sglang`): a parser name is an
    engine plugin, and `vllm-devai` runs the same plugins as `vllm`. A block
    under the backend's own name still wins, per key, so a build that ships
    a different plugin can be curated explicitly. Looking this up by NAME
    alone is how both derived rows were probed with no parsers at all on
    2026-09-22 (no `vllm-devai:` block exists) and then served with
    reasoning bleeding into content.
    """
    parsers = row.get("parsers") or {}
    own = parsers.get(backend) or {}
    engine = parsers.get(engine_of(backend)) or {}
    return (own.get("reasoning") or engine.get("reasoning") or None,
            own.get("tool") or engine.get("tool") or None)


def derive_parser_for(name: str, models_dir: Path, backend: str,
                      kind: str) -> str | None:
    """Card-derived parser hint; the hint tables are keyed by engine too."""
    return _card_hints.derive_parser(name, models_dir, engine_of(backend), kind)


# Name -> parser guesses, the fourth step of the parser order (plan Phase
# 17a): used only when the row has no curated block for the engine, the
# checkpoint's own template derives nothing and no recipe seeds a value.
# Labelled "guessed, not probe-verified": the probe is the authority, and a
# guessed TOOL parser is recorded only when the auto-choice call verifies it
# (probe_one_cell, `tool_parser_auto_only`). Ordered, first match wins,
# matched against the lower-cased catalog name.
#
# Each value is one this host's probe caches recorded for that family on
# 2026-10-07 (deploy/.vllm-, .vllm-devai- and .sglang-reasoning-cache.json):
# a tool parser verified in AUTO mode (`tool_mode: auto`), a reasoning
# parser with capability `structured`. Left out on that evidence: the
# Llama-3.x tool parsers (`llama3_json`, SGLang `llama3`) and SGLang's
# `qwen` for the Qwen3.5+ markup, which passed only the forced call; SGLang
# `qwen3_coder`, never verified there; and every plugin-backed parser
# (deploy/vllm-plugins.json).
NAME_PARSER_HEURISTICS: tuple[tuple[str, dict[str, dict[str, str]]], ...] = (
    # First: a distill names its base too (DeepSeek-R1-0528-Qwen3-8B).
    (r"deepseek-r1", {"vllm": {"reasoning": "deepseek_r1"},
                      "sglang": {"reasoning": "deepseek-r1"}}),
    (r"gpt-oss", {"vllm": {"tool": "openai", "reasoning": "openai_gptoss"},
                  "sglang": {"tool": "gpt-oss", "reasoning": "gpt-oss"}}),
    (r"qwen3-?coder", {"vllm": {"tool": "qwen3_coder"}}),
    # Qwen3.5 and later (the qwen3_5 architecture: Qwen3.5/3.6/3.8).
    (r"qwen3\.[5-9]", {"vllm": {"tool": "qwen3_xml", "reasoning": "qwen3"},
                       "sglang": {"reasoning": "qwen3"}}),
    (r"qwen3(?![.\d])", {"vllm": {"tool": "hermes", "reasoning": "qwen3"},
                         "sglang": {"tool": "qwen", "reasoning": "qwen3"}}),
    (r"gemma-?4", {"vllm": {"tool": "gemma4"}}),
)

# The label printed beside a heuristic value; `heuristic` is what the cache
# records as its source.
HEURISTIC_LABEL = "guessed, not probe-verified"
# Sources whose value is a guess: dropped when the row's first launch fails
# and the same context fits without it; a guessed tool parser is recorded
# only on the auto call.
GUESSED_PARSER_SOURCES = frozenset({"recipe", "heuristic"})
# The entry fields that record where a row's parsers came from; restored
# with the band when an exact-context probe misses.
PARSER_PROVENANCE_KEYS = ("tool_parser_source", "reasoning_parser_source",
                          "parser_disagreement", "parser_fallback")
_ABSENT = object()


def heuristic_parser(name: str, backend: str, kind: str) -> str | None:
    """The name heuristic's parser for `kind` ("tool" / "reasoning") on the
    backend's engine, or None. The first rule whose pattern matches the name
    decides; a matching rule with no value for this kind gives None."""
    lowered = (name or "").lower()
    for pattern, by_engine in NAME_PARSER_HEURISTICS:
        if re.search(pattern, lowered):
            return (by_engine.get(engine_of(backend)) or {}).get(kind)
    return None


def recipe_seed_parsers(name: str, backend: str, image: str | None,
                        seeds: dict | None = None) -> dict[str, str]:
    """The recipe seed's parsers for this row (deploy/recipe-seeds.yaml), or
    {}: vLLM engines only, and only when the recipe's min_vllm_version admits
    the vLLM this backend runs (a recipe never injects a parser the pinned
    vLLM lacks). When that version is unknown (an image tagged `latest`),
    nothing is seeded."""
    if seeds is None:
        seeds = _vllm_recipes.load_seeds()
    seed = seeds.get(name)
    if not seed:
        return {}
    # The image the launch will really use: a curated `image` pin wins, and
    # is judged by its own tag only.
    pin = recovery_image(name, backend)
    version = _vllm_recipes.engine_vllm_version(backend, pin or image, pinned=bool(pin))
    return _vllm_recipes.seed_parsers(seed, engine_of(backend), version)


def has_curated_block(row: dict, backend: str) -> bool:
    """True when the catalog row curates parsers for this backend (its own
    name or its engine), even if the block names only one kind."""
    parsers = row.get("parsers") or {}
    return bool(parsers.get(backend) or parsers.get(engine_of(backend)))


def resolve_parsers(
    name: str, models_dir: Path, backend: str,
    curated_reasoning: str | None, curated_tool: str | None,
    *, recipe_parsers: dict[str, str] | None = None,
    ignore_curated: bool = False, curated_block: bool = False,
    rejected: dict[str, str] | None = None,
) -> tuple[str | None, str | None, dict[str, str | None], dict[str, dict]]:
    """Pick each parser in the order of plan Phase 17a, step 1.

    1. the curated `parsers:` block (`curated`);
    2. the value derived from the checkpoint's own chat template
       (`card`, scripts/_card_hints.py, which refuses rather than guesses);
    3. the recipe seed (`recipe`, deploy/recipe-seeds.yaml);
    4. the name heuristic (`heuristic`, "guessed, not probe-verified");
    5. none (no parser flag; the router strips tools).

    Steps 3 and 4 are guesses for a model nobody has curated: with
    `curated_block` (the row has a `parsers:` block for this engine) a kind
    the block leaves out stays a curated omission and they are skipped.
    That keeps decisions like gemma4's measured "no reasoning parser"
    (scripts/model-families.yaml) from being overridden by a recipe.

    `ignore_curated` (PROBE_IGNORE_CURATED_PARSERS=1, plan I24) skips 1, 3
    and 4, so the card-derived parsers alone drive the launch. The probe
    remains the authority: a value it cannot verify is not recorded.

    `rejected` ({kind: parser}, from the row's recorded `parser_fallback`)
    names guesses an earlier run dropped: a recipe or heuristic value equal
    to it is skipped, and the order goes on to the next step.

    Returns (reasoning, tool, sources, disagreements); `sources` maps each
    kind to its step's label or None, `disagreements` records a card
    derivation that names a different parser than the curated block.
    """
    chosen: dict[str, str | None] = {}
    sources: dict[str, str | None] = {}
    disagreements: dict[str, dict] = {}
    seeded = recipe_parsers or {}
    for kind, curated in (("tool", curated_tool),
                          ("reasoning", curated_reasoning)):
        if ignore_curated:
            curated = None
        derived = None
        try:
            derived = derive_parser_for(name, models_dir, backend, kind)
        except Exception as e:  # noqa: BLE001
            # Never let a hint break a probe: no hint is today's
            # behaviour, a crash is not.
            print(f"  {name}: card-hint derivation failed ({kind}): {e}",
                  file=sys.stderr)
        if curated is not None:
            chosen[kind], sources[kind] = curated, "curated"
            if derived and derived != curated:
                # Behaviour is unchanged -- curated still wins -- but a
                # silent disagreement is how curation drifts out of step
                # with the checkpoints it describes.
                print(f"  {name}: WARNING {kind} parser disagreement -- "
                      f"curated={curated!r} but the chat template "
                      f"derives {derived!r}. Serving the curated value.",
                      file=sys.stderr)
                disagreements[kind] = {"curated": curated, "derived": derived}
            continue
        if derived:
            chosen[kind], sources[kind] = derived, "card"
            continue
        if not ignore_curated and not curated_block:
            dropped = (rejected or {}).get(kind)
            if seeded.get(kind) and seeded[kind] != dropped:
                chosen[kind], sources[kind] = seeded[kind], "recipe"
                continue
            guessed = heuristic_parser(name, backend, kind)
            if guessed and guessed == dropped:
                print(f"  {name}: {kind} parser {guessed!r} skipped: an earlier "
                      f"run dropped it (parser_fallback)", file=sys.stderr)
                guessed = None
            if guessed:
                print(f"  {name}: {kind} parser {guessed!r} from the name "
                      f"heuristic ({HEURISTIC_LABEL})", file=sys.stderr)
                chosen[kind], sources[kind] = guessed, "heuristic"
                continue
        chosen[kind], sources[kind] = None, None
    return chosen.get("reasoning"), chosen.get("tool"), sources, disagreements


def ignore_curated_parsers_from_env() -> bool:
    """PROBE_IGNORE_CURATED_PARSERS=1 (the Makefile forwards it; plan I24)."""
    return os.environ.get("PROBE_IGNORE_CURATED_PARSERS", "").strip().lower() in (
        "1", "true", "yes", "on")


@dataclass(frozen=True)
class BackendSpec:
    """Per-backend constants and the launch-arg builder."""
    name: str                   # "vllm" | "vllm-devai" | "sglang"

    @property
    def engine(self) -> str:
        return engine_of(self.name)
    image: str                  # default container image
    container_name: str         # default probe container name
    probe_port: int             # default loopback port
    cache_path: Path            # default cache file
    reserve_gb: float           # memFraction reservation
    entrypoint: str             # container ENTRYPOINT override
    # build_args(model_name, max_ctx, host_frac, *, reasoning_parser,
    # tool_parser, reasoning_parser_plugin, tool_parser_plugin)
    # → CMD list (excluding entrypoint). The two parser-name kwargs are
    # optional — when None, the arg builder must omit the corresponding
    # backend flag so the launch falls back to inline/no-tool mode.
    # The two *_plugin kwargs carry an in-container absolute path to a
    # plugin .py file when the parser name resolved through the
    # vllm-plugins registry; the arg builder must emit the
    # `--*-parser-plugin <path>` flag *before* the matching parser-name
    # flag (vLLM loads plugin files at parser-resolution time). Backends
    # without a plugin model (SGLang) accept the kwargs and ignore them.
    # The `kv_cache_dtype` kwarg names the KV dtype for THIS launch; None
    # means "use the pass default". The load probe passes the dtype the
    # target cell was fit-probed under so serving numbers describe the
    # dtype the cell actually advertises.
    build_args: Callable[..., list[str]] = field(repr=False)
    # When True, parser names are looked up against the vllm-plugins
    # registry; matches inject the bind-mount + --tool-parser-plugin
    # flag. Only vLLM supports this today — SGLang's plugin model uses
    # Python registry imports rather than a file-path arg.
    supports_plugins: bool = False
    # KV-cache dtype the build_args launch enforces for this pass
    # (e.g. "fp8" for vLLM's default pass, "auto" for an unquantized
    # measurement). Stamped as `kv_cache_type` on every successful cell
    # so serve time reproduces the measured dtype — fit is only valid
    # under the dtype it was measured with. "" = engine default, cell
    # left unstamped (legacy shape).
    kv_cache_dtype: str = ""
    # KV dtypes this backend's --kv-cache-dtype actually accepts, taken
    # from the pinned image's --help. The two engines DISAGREE: vLLM
    # accepts a bare `fp8` (= fp8_e4m3), SGLang does not -- its choices are
    # {auto,fp8_e5m2,fp8_e4m3,bf16,bfloat16,fp4_e2m1}. Since
    # PROBE_KV_CACHE_TYPE is a single knob shared by BOTH probers and its
    # canonical vLLM value is `fp8`, one `make probe-sglang
    # PROBE_KV_CACHE_TYPE=fp8` would have every launch rejected by
    # argparse. Empty tuple = do not validate.
    allowed_kv_dtypes: tuple[str, ...] = ()
    # schema_version v2 added reasoning_parser, tool_parser (populated),
    # disable_verified, and per-cell tool/disable verdicts. v1 readers
    # backfill defaults on first read.
    schema_version: int = 2


# ── VRAM math (mirrors gpu-arbiter/main.go memFraction) ──────────────────────

def mem_fraction_for_band(
    model_size_gb: float, band_gb: float, reserve_gb: float
) -> float:
    """Compute backend memory utilisation for a target VRAM band.

    Mirrors gpu-arbiter/main.go memFraction so the prober's
    measurements match what the router will demand at serve time.
    Returns a float in [0.40, 0.95].
    """
    headroom = band_gb - model_size_gb
    if headroom < reserve_gb + 2.0:
        reserve_gb = max(0.5, headroom * 0.3)
    frac = (band_gb - reserve_gb) / band_gb
    return max(0.40, min(0.95, frac))


def host_scaled_fraction(
    model_size_gb: float,
    band_gb: float,
    host_vram_gb: float,
    reserve_gb: float,
) -> float:
    """Translate band-relative fraction to a host-relative fraction.

    Both vLLM (--gpu-memory-utilization) and SGLang
    (--mem-fraction-static) interpret their fraction against the
    *physical* GPU. To simulate a smaller card we scale by
    (band_gb / host_vram_gb). When band == host this is a passthrough.
    """
    band_frac = mem_fraction_for_band(model_size_gb, band_gb, reserve_gb)
    if band_gb >= host_vram_gb:
        return band_frac
    scaled = (band_frac * band_gb) / host_vram_gb
    return max(0.10, min(0.95, scaled))


# ── Catalog filtering ────────────────────────────────────────────────────────

def load_catalog_hf_rows(path: Path, backend_filter: str) -> list[dict]:
    """Parse models.yaml and return source:hf rows where backend_filter
    is in the row's `backend` list (e.g. 'vllm' or 'sglang').

    Uses PyYAML when available, falls back to a small regex parser so
    the prober runs in environments without PyYAML installed.
    """
    text = path.read_text()
    try:
        import yaml
        doc = yaml.safe_load(text)
        models = doc.get("models", []) if isinstance(doc, dict) else []
    except ImportError:
        models = _parse_models_yaml_regex(text)
    out = []
    for m in models:
        # Derived rows (made by `make model-prepare`) live in the HF store
        # and are probed like HF rows; their `repo: devai/<name>` keys them
        # apart from their source in the cache.
        if m.get("source") not in ("hf", "derived"):
            continue
        if backend_filter not in (m.get("backend") or []):
            continue
        out.append(m)
    return out


def _parse_models_yaml_regex(text: str) -> list[dict]:
    """Minimal `- name:` block parser for the auto-generated YAML.

    Used only when PyYAML isn't installed. Recognises the shape from
    generate-catalog.py: every key on its own line, list values inline
    `[a, b]`, scalar values bare or quoted.
    """
    models: list[dict] = []
    current: dict | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("  - name:"):
            if current is not None:
                models.append(current)
            current = {}
            current["name"] = line.split(":", 1)[1].strip().strip('"')
        elif current is not None and line.startswith("    "):
            kv = line.strip()
            if not kv or ":" not in kv:
                continue
            key, _, val = kv.partition(":")
            key = key.strip()
            val = val.strip()
            if key == "backend":
                inner = val.strip("[]")
                current[key] = [s.strip() for s in inner.split(",") if s.strip()]
            elif key == "arch":
                current[key] = val
            else:
                current[key] = val.strip('"')
    if current is not None:
        models.append(current)
    return models


def is_downloaded(model_name: str, models_dir: Path) -> bool:
    """HF/NVFP4 models live at <models_dir>/<name>/config.json."""
    return (models_dir / model_name / "config.json").is_file()


def model_size_gb_from_row(row: dict) -> float:
    raw = (row.get("size") or "").strip().upper()
    raw = raw.rstrip("B").rstrip("G").strip()
    try:
        return float(raw)
    except ValueError:
        return 0.0


def model_kind_from_disk(model_name: str, models_dir: Path) -> str:
    """Inspect <models_dir>/<name>/config.json for MoE markers.

    Returns 'moe' when num_local_experts > 0 (or analogous fields),
    'dense' otherwise. Falls back to 'unknown' on any read error.
    SGLang reserves more memory for expert weights than memFraction
    accounts for, so this hint is useful for post-hoc analysis even
    when capability classification doesn't depend on it.
    """
    cfg_path = models_dir / model_name / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, json.JSONDecodeError):
        return "unknown"
    if not isinstance(cfg, dict):
        return "unknown"
    inner = cfg.get("text_config", cfg) if isinstance(cfg, dict) else cfg
    for key in ("num_local_experts", "num_experts", "moe_num_experts"):
        v = inner.get(key) if isinstance(inner, dict) else None
        if isinstance(v, int) and v > 0:
            return "moe"
    return "dense"


def effective_position_limit(model_name: str, models_dir: Path) -> int | None:
    """Largest token position the model can actually serve, read from its
    config.json -- the HARD architectural ceiling, distinct from a model
    card's advertised context.

    Returns max_position_embeddings (top-level or text_config, whichever
    is larger), widened to factor*original_max_position_embeddings when a
    rope_scaling block declares a larger extended range. None when the
    field is absent (model unaffected by the cap).

    Why this exists: nvidia/Qwen3-8B-NVFP4 ships max_position_embeddings
    =40960 with rope_scaling=null. vLLM loads it at --max-model-len 131072
    (VLLM_ALLOW_LONG_MAX_MODEL_LEN bypasses the startup guard -- it only
    raises the scheduler's max-len, it does NOT add YaRN), so the fit
    probe records fits=true at 131072. But a >40960-token prompt indexes
    RoPE out of range and triggers a CUDA device-side assert at serve
    time (confirmed: Qwen3 model cards + QwenLM/Qwen3#1361, vLLM#17924).
    Capping the probe + max_context at this limit stops the fit probe
    over-promising a context the model cannot serve. gpt-oss-20b
    (max_position_embeddings=131072, YaRN to 131072) is the same story
    one tier up: it must not advertise 256K.
    """
    cfg_path = models_dir / model_name / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(cfg, dict):
        return None
    scopes = [cfg]
    if isinstance(cfg.get("text_config"), dict):
        scopes.append(cfg["text_config"])
    mpe = 0
    rope: dict | None = None
    for s in scopes:
        v = s.get("max_position_embeddings")
        if isinstance(v, int) and v > mpe:
            mpe = v
        r = s.get("rope_scaling")
        if rope is None and isinstance(r, dict):
            rope = r
    if mpe <= 0:
        return None
    limit = mpe
    if isinstance(rope, dict):
        factor = rope.get("factor")
        orig = rope.get("original_max_position_embeddings")
        if isinstance(factor, (int, float)) and isinstance(orig, int) and orig > 0:
            limit = max(limit, int(factor * orig))
    return limit


# ── Container lifecycle (podman CLI) ─────────────────────────────────────────

def assert_no_active_backends(runtime: str) -> None:
    """Refuse to run when any GPU-owning backend container is up.

    The router, vLLM, and SGLang containers share the single GPU. If
    any are running they would race the prober's launch — the user
    must `make cache-down` first.
    """
    for name in MUTEX_CONTAINERS:
        try:
            r = subprocess.run(
                [runtime, "ps", "--filter", f"name=^{name}$",
                 "--format", "{{.Names}}"],
                capture_output=True, text=True, check=False,
            )
        except FileNotFoundError:
            sys.exit(f"error: {runtime} CLI not found in PATH")
        if name in r.stdout:
            sys.exit(
                f"error: {name} is running. Stop it first: `make cache-down`."
            )


# Probe containers this process has launched and not yet torn down, as
# (runtime, name) pairs. Populated by container_run_detached, drained by
# container_remove, and swept by the SIGINT/SIGTERM/atexit hooks installed
# via install_probe_cleanup. An orphan here is a GPU-holding container that
# assert_no_active_backends does NOT look for, so every later probe cell
# would silently measure against the leftover allocation.
_ACTIVE_CONTAINERS: set[tuple[str, str]] = set()

# `podman rm --force` normally returns in well under a second. Bound it so
# teardown -- which runs from a signal handler -- can never hang forever on a
# wedged runtime.
CONTAINER_RM_TIMEOUT = 60.0


def container_remove(runtime: str, name: str) -> None:
    """Force-remove a probe container. Idempotent, bounded, never raises."""
    try:
        subprocess.run(
            [runtime, "rm", "--force", name],
            capture_output=True, check=False, timeout=CONTAINER_RM_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        print(f"  [warn] `{runtime} rm --force {name}` timed out after "
              f"{CONTAINER_RM_TIMEOUT:.0f}s — it may still hold the GPU",
              file=sys.stderr)
    except FileNotFoundError:
        pass
    _ACTIVE_CONTAINERS.discard((runtime, name))


def _teardown_active_containers() -> None:
    """Remove every probe container this process still owns. Idempotent:
    container_remove drops each entry from the registry as it goes, and
    `sorted()` snapshots the set so removal during iteration is safe."""
    for runtime, name in sorted(_ACTIVE_CONTAINERS):
        print(f"  [cleanup] removing probe container {name}", file=sys.stderr)
        container_remove(runtime, name)


_CLEANUP_INSTALLED = False


def install_probe_cleanup() -> None:
    """Guarantee probe containers are torn down on Ctrl-C / SIGTERM / exit.

    Without this, an interrupt during a health poll or a chat call leaves an
    orphaned GPU-holding container behind: assert_no_active_backends only
    looks for devai-router / devai-vllm / devai-sglang, so the leftover is
    invisible and silently contaminates every later probe cell with false
    OOMs / fits=false. Idempotent — safe to call from each pass driver.
    """
    global _CLEANUP_INSTALLED
    if _CLEANUP_INSTALLED:
        return
    _CLEANUP_INSTALLED = True
    atexit.register(_teardown_active_containers)

    def _handler(signum, _frame):
        _teardown_active_containers()
        # Restore the default disposition and re-raise so the process exits
        # with the conventional signal status rather than a bare code.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            # Not on the main thread — atexit still covers normal exit.
            pass


def container_logs(runtime: str, name: str, tail: int = 500) -> str:
    r = subprocess.run(
        [runtime, "logs", "--tail", str(tail), name],
        capture_output=True, text=True, check=False,
    )
    return (r.stdout or "") + (r.stderr or "")


def container_state(runtime: str, name: str) -> str:
    """Return State.Status, or 'absent' on missing/error."""
    r = subprocess.run(
        [runtime, "inspect", "--format", "{{.State.Status}}", name],
        capture_output=True, text=True, check=False,
    )
    if r.returncode != 0:
        return "absent"
    return (r.stdout or "").strip() or "absent"


# Persistent engine caches, mirrored from gpu-arbiter/engine_cache.go: one
# podman named volume per cache directory and backend. vLLM and SGLang
# JIT-compile at first use (FlashInfer's nvcc-built kernels under
# ~/.cache/flashinfer/<version>/, torch.compile under ~/.cache/vllm,
# SGLang under ~/.cache/sglang); with nothing bound there every launch
# re-paid the whole compile (269 s cold start measured 2026-09-22 for
# Qwen3.8-27B-MTP-devai-NVFP4 on vllm-devai). A probe that mounts the same
# volumes as the router warms the cache a serve-time launch then reuses.
# Narrow on purpose -- the SGLang image carries 877 MB of HF cache under
# /root/.cache, which podman would copy into an empty whole-directory
# volume on first use. Keyed by BACKEND so the stock and custom vLLM
# images never share one. vLLM's own cache (torch.compile) is NOT
# persisted, only FlashInfer's: a full cache hit under-measures peak
# activation at startup, the KV pool comes out oversized and the first
# real prompt OOMs (measured 2026-09-23; see gpu-arbiter/engine_cache.go).
# A probe measured against that oversized pool would record a fit the
# engine cannot serve. Pinned to the Go side by
# tests/python/test_engine_cache_volumes.py.
ENGINE_CACHE_VOLUME_PREFIX = "devai-engine-cache-"


def engine_cache_volumes(backend: str) -> list[tuple[str, str]]:
    """(volume name, container path) pairs for an HF backend; [] otherwise."""
    engine = engine_of(backend)
    if engine not in ("vllm", "sglang"):
        return []
    vols = [(f"{ENGINE_CACHE_VOLUME_PREFIX}{backend}-flashinfer", "/root/.cache/flashinfer")]
    if engine == "sglang":
        vols.append((f"{ENGINE_CACHE_VOLUME_PREFIX}{backend}-{engine}", f"/root/.cache/{engine}"))
    return vols


def container_run_args(
    runtime: str,
    name: str,
    image: str,
    publish_port: int,
    models_dir: str,
    env_vars: dict[str, str],
    entrypoint: str,
    command: list[str],
    extra_volumes: list[tuple[str, str, str]] | None = None,
    backend: str | None = None,
) -> list[str]:
    """The `podman run` argv container_run_detached executes; pure, testable.

    `backend` adds that backend's persistent engine-cache volumes (see
    engine_cache_volumes); a named volume takes the same `--volume
    name:dest:mode` form as a bind and is created on demand.
    """
    args = [
        runtime, "run", "--detach",
        "--name", name,
        "--entrypoint", entrypoint,
        "--publish", f"127.0.0.1:{publish_port}:11434",
        "--volume", f"{models_dir}:/models:ro",
        "--device", "nvidia.com/gpu=all",
        "--security-opt", "label=disable",
    ]
    for host, dst, mode in extra_volumes or []:
        args.extend(["--volume", f"{host}:{dst}:{mode}"])
    for vol, dst in engine_cache_volumes(backend) if backend else []:
        args.extend(["--volume", f"{vol}:{dst}:rw"])
    for k, v in env_vars.items():
        args.extend(["--env", f"{k}={v}"])
    args.append(image)
    args.extend(command)
    return args


def container_run_detached(
    runtime: str,
    name: str,
    image: str,
    publish_port: int,
    models_dir: str,
    env_vars: dict[str, str],
    entrypoint: str,
    command: list[str],
    extra_volumes: list[tuple[str, str, str]] | None = None,
    backend: str | None = None,
) -> None:
    """Launch a probe container detached on localhost loopback.

    `--entrypoint` is mandatory: the upstream vllm and sglang images
    ship with their own ENTRYPOINTs that swallow our CMD args (the
    "vllm: error: unrecognized arguments" we hit on the first probe
    run). Replace it explicitly to match the router's libpod spec.

    `extra_volumes` is an optional list of (host_path, container_path,
    mode) tuples — currently used to mount the vllm-plugins directory
    when a model's parser resolved through the plugin registry.
    """
    args = container_run_args(
        runtime, name, image, publish_port, models_dir, env_vars, entrypoint,
        command, extra_volumes=extra_volumes, backend=backend,
    )
    # Register BEFORE launching: a `podman run` that fails partway can still
    # have created the container, and the teardown hooks must know about it.
    _ACTIVE_CONTAINERS.add((runtime, name))
    r = subprocess.run(args, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(
            f"{runtime} run failed (rc={r.returncode}): {r.stderr.strip()}"
        )


# ── nvidia-smi ───────────────────────────────────────────────────────────────

def gpu_memory_used_mb() -> int:
    """Max memory.used across visible GPUs, in MB. 0 when nvidia-smi unavailable."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,nounits,noheader"],
            capture_output=True, text=True, check=False, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    if r.returncode != 0:
        return 0
    values: list[int] = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(line))
        except ValueError:
            continue
    return max(values) if values else 0


# ── Classification ───────────────────────────────────────────────────────────

def _extract_reasoning_content(msg: dict) -> str:
    """Pull the reasoning trace from a chat message regardless of field
    name. vLLM ≥0.19 returns it as `reasoning`; SGLang and older vLLM
    use `reasoning_content`. Future-proof: check both. Returns empty
    string when neither is present.
    """
    if not isinstance(msg, dict):
        return ""
    for key in ("reasoning_content", "reasoning"):
        v = msg.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def response_has_reasoning_content(resp: dict) -> bool:
    """Return True iff the response carries a non-empty reasoning trace
    in either `reasoning_content` (SGLang, older vLLM) or `reasoning`
    (vLLM ≥0.19)."""
    if not isinstance(resp, dict) or "error" in resp:
        return False
    choices = resp.get("choices") or []
    if not choices:
        return False
    return bool(_extract_reasoning_content(choices[0].get("message") or {}))


def response_has_valid_tool_call(resp: dict, expected_tool_name: str) -> bool:
    """Return True iff the response contains a parseable tool call to
    `expected_tool_name`. Tolerates either OpenAI shape (top-level
    `tool_calls`) or providers that nest under `function_call`.
    """
    if not isinstance(resp, dict) or "error" in resp:
        return False
    choices = resp.get("choices") or []
    if not choices:
        return False
    msg = (choices[0].get("message") or {})
    calls = msg.get("tool_calls") or []
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        if fn.get("name") == expected_tool_name:
            return True
    legacy = msg.get("function_call") or {}
    return legacy.get("name") == expected_tool_name


# Launch-argv values that legitimately vary per cell and must NOT be part
# of the fingerprint, or every model and every context tier would hash
# differently and the signal would be noise.
_FINGERPRINT_VALUE_FLAGS = (
    "--model-path", "--model", "--served-model-name",
    "--context-length", "--max-model-len",
    "--mem-fraction-static", "--gpu-memory-utilization",
)


def launch_fingerprint(cmd_args: list[str]) -> str:
    """Stable 12-char hash of the launch argv's SHAPE.

    Per-cell values (model path, served name, context, memory fraction)
    are elided; everything else -- flag names, parser names, dtypes,
    recovery flags, their ORDER -- is hashed. So the fingerprint changes
    exactly when the way we launch changes, and not when we launch a
    different model.

    This is what makes the stale-cell class self-detecting. Seven SGLang
    cells sat in the cache for three weeks recording a permanent `quant`
    verdict that was really an argparse rejection of a flag the lab should
    never have sent; the allow-list fix shipped with no cache
    invalidation, so those verdicts stayed authoritative and were cited in
    three places as genuine arch/quant gaps. A fingerprint per cell turns
    "the way we launch changed" from an archaeology problem into a diff.
    """
    parts: list[str] = []
    skip_next = False
    for tok in cmd_args:
        if skip_next:
            skip_next = False
            parts.append("<elided>")
            continue
        parts.append(tok)
        if tok in _FINGERPRINT_VALUE_FLAGS:
            skip_next = True
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:12]


def validate_kv_dtype(spec: "BackendSpec", dtype: str | None) -> None:
    """Reject a KV dtype this backend's CLI does not accept, before launch.

    Without this the launch fails inside the container with an argparse
    error, which the probe then has to classify from a log -- and which,
    until the `launch_args` kind existed, was misfiled as a permanent
    per-model `quant` verdict. Failing here instead names the knob, the
    backend and the accepted set.

    No cell is stamped with a non-default dtype on this host today, so
    this guard is latent rather than fixing live data -- it stops the
    NEXT `make probe-sglang PROBE_KV_CACHE_TYPE=fp8` from manufacturing a
    fresh batch of poisoned cells.
    """
    if not dtype or not spec.allowed_kv_dtypes:
        return
    if dtype in spec.allowed_kv_dtypes:
        return
    hint = ""
    if dtype == "fp8" and "fp8_e4m3" in spec.allowed_kv_dtypes:
        hint = (" Did you mean fp8_e4m3? vLLM's bare `fp8` is an alias for it, "
                "but SGLang requires the explicit spelling.")
    raise SystemExit(
        f"error: KV cache dtype {dtype!r} is not accepted by {spec.name}.\n"
        f"  accepted: {', '.join(spec.allowed_kv_dtypes)}\n"
        f"  set PROBE_KV_CACHE_TYPE to one of those, or unset it for the "
        f"engine default.{hint}"
    )


def build_disable_thinking_body(backend: str, base: dict) -> dict:
    """Mutate-in-place a chat body to request reasoning suppression.

    vLLM (what applyVLLMPolicy sends):
        reasoning_effort = "none", plus
        top-level chat_template_kwargs.enable_thinking = False
      Until 2026-09-22 the kwarg went under `extra_body`, which vLLM never
      reads (it is the OpenAI Python client's name for "splice into the
      body", not a wire field), so it was discarded and only
      `reasoning_effort` did anything. The "kwarg alone still yields 175
      reasoning tokens" measurement that justified sending both had
      measured that inert form. Re-measured through the router on Qwen3.8
      (vllm-devai): top-level kwarg alone 0 reasoning chars, extra_body
      kwarg alone 121 (= baseline), reasoning_effort=none alone 0. Both
      fields are sent because the router sends both; existing
      disable_verified verdicts were measured with an effective
      reasoning_effort=none alone and remain valid (this body is a
      superset).
    SGLang:
        top-level chat_template_kwargs.enable_thinking = False, plus
        reasoning_effort = "none"

    Both engines read the kwarg at the top level; the probe must send
    exactly what the router sends or it is measuring a fiction.

    SGLang previously got `separate_reasoning = False` here, which made the
    whole probe a tautology: that field stops SGLang SPLITTING `<think>`
    out into `reasoning_content`, and the verification then checked
    whether `reasoning_content` was absent. It could only ever pass.
    Measured across the caches, `disable_verified` was true for 8 of 9
    SGLang reasoning rows and false for 0 of 11 vLLM rows on the same
    checkpoints -- not a capability difference, an artefact. It also sent
    `extra_body`, which is not a field on SGLang's request model and is
    discarded silently.

    `reasoning_effort="none"` is the real lever: SGLang's
    `normalize_reasoning_inputs` expands it to both template key spellings
    (`thinking` and `enable_thinking`). It is sent TOP-LEVEL deliberately
    -- the Harmony guard that rejects "none" for gpt-oss reads the value
    popped out of `chat_template_kwargs`, so nesting it there would 400.

    Both fields are no-ops when the model has no template-level
    thinking switch — the disable verdict is then `False` and the
    router won't emit a disable directive at serve time.
    """
    body = dict(base)
    # Keyed by ENGINE: vllm-devai speaks vLLM's shape. Comparing the
    # backend NAME here sent the base body unchanged for it, so the
    # disable probe measured a model that was never told to stop thinking
    # and recorded disable_verified=false for every derived row (2026-09-22).
    engine = engine_of(backend)
    if engine in ("vllm", "sglang"):
        ctk = dict(body.get("chat_template_kwargs") or {})
        ctk["enable_thinking"] = False
        body["chat_template_kwargs"] = ctk
        body["reasoning_effort"] = "none"
    return body


# Inline reasoning markers. A model that was told not to think and then
# emits one of these in `content` did not honour the directive -- it just
# stopped having its trace split into a separate field.
_INLINE_THINK_MARKERS = ("<think>", "</think>", "<thinking>", "<|channel|>analysis")


def response_has_inline_reasoning(resp: dict) -> bool:
    """True when the assistant's `content` carries an inline reasoning
    trace. Needed to make the disable check falsifiable: absence of
    `reasoning_content` alone proves nothing, because a parsing switch can
    produce that absence while the model thinks exactly as much as before.
    """
    if not isinstance(resp, dict) or "error" in resp:
        return False
    choices = resp.get("choices") or []
    if not choices:
        return False
    msg = choices[0].get("message") or {}
    content = (msg.get("content") or "").lower()
    return any(marker in content for marker in _INLINE_THINK_MARKERS)


def build_enable_thinking_body(backend: str, base: dict) -> dict:
    """Symmetric to build_disable_thinking_body -- flips the same field
    to enable structured reasoning output. Used by Probe A so models
    with chat templates that default `enable_thinking=false` (Gemma 4,
    etc.) emit a reasoning trace under the curated parser.

    Top-level `chat_template_kwargs` on BOTH engines. Until 2026-09-22 this
    sent the kwarg under `extra_body` for both, which neither engine reads,
    so Probe A had always measured the template's own default: a row whose
    template opts thinking OUT by default was classified with thinking off
    and never asked to think. Rows in that class (Gemma-4 family) deserve
    a re-probe now that the request reaches the template.
    """
    body = dict(base)
    engine = engine_of(backend)  # see build_disable_thinking_body
    if engine in ("vllm", "sglang"):
        ctk = dict(body.get("chat_template_kwargs") or {})
        ctk["enable_thinking"] = True
        body["chat_template_kwargs"] = ctk
    if engine == "sglang":
        body["separate_reasoning"] = True
    return body


def classify_chat_response(
    resp: dict, *, reasoning_parser_attempted: str | None = None,
) -> tuple[str, dict]:
    """Map /v1/chat/completions response to (capability, evidence).

    Detection order:
      1. `message.reasoning_content` non-empty → `structured`
         (vLLM and SGLang both populate this when launched with
         `--reasoning-parser <X>` and the model emits a reasoning
         trace; the parser strips the trace from `content`)
      2. reasoning_parser was attempted AND finish_reason == "length"
         AND content is empty → `structured` (model was thinking past
         max_tokens; parser was buffering inside an unclosed `<think>`
         block when generation cut off — the parser was clearly active)
      3. reasoning_parser was attempted AND content starts with the
         tell-tale `\\n\\n` left by a stripped `<think></think>` block
         → `structured` (parser stripped an empty trace; common with
         R1-Distill chat templates)
      4. inline `<think>` markers in `message.content` → `inline`
         (any chat surface, no parser flag required)
      5. otherwise → `unsupported`
    """
    if "error" in resp:
        return Capability.ERROR, {"error": resp["error"]}
    choices = resp.get("choices") or []
    if not choices:
        return Capability.UNSUPPORTED, {"reason": "no choices in response"}
    msg = (choices[0].get("message") or {})
    reasoning = _extract_reasoning_content(msg)
    content = msg.get("content") or ""
    finish_reason = (choices[0].get("finish_reason") or "")

    if reasoning:
        return Capability.STRUCTURED, {
            "reasoning_preview": reasoning[:200],
            "content_preview": content[:120],
            "structured_reason": "reasoning_content_populated",
        }
    if reasoning_parser_attempted and finish_reason == "length" and not content:
        # Model thought past max_tokens; parser buffered inside <think>
        # waiting for </think>. The parser was active — the budget was
        # too low. Classify as structured; the disable probe will
        # confirm wiring.
        return Capability.STRUCTURED, {
            "structured_reason": "thought_past_max_tokens",
            "finish_reason": finish_reason,
        }
    if reasoning_parser_attempted and content.startswith("\n\n"):
        # Parser likely stripped an empty `<think></think>` block,
        # leaving the leading newlines. Distinct from a model that
        # genuinely starts with newlines (rare with our prompt).
        return Capability.STRUCTURED, {
            "structured_reason": "empty_think_stripped",
            "content_preview": content[:120],
        }
    if has_inline_think_markers(content):
        return Capability.INLINE, {"content_preview": content[:200]}
    # Distinguish "model doesn't reason and was never asked to" from
    # "we tried a reasoning parser and got nothing back". Both render
    # as `No reasoning` in the picker, but the cache audit needs to
    # tell apart well-behaved non-reasoning models (Llama-3.1, dense
    # Gemma) from broken parser pairings.
    if (
        not reasoning_parser_attempted
        and finish_reason == "stop"
        and content
    ):
        return Capability.NONE, {
            "none_reason": "clean_answer_no_parser_attempted",
            "content_preview": content[:120],
        }
    return Capability.UNSUPPORTED, {"content_preview": content[:120]}


# Patterns must indicate an architecture *rejection*, not just the
# presence of architecture-related strings. Verified false positive:
# SGLang's startup banner echoes `--trust-remote-code` as
# `trust_remote_code=True`, so that pattern (with or without the =True
# suffix) hits on every nominal SGLang launch and is unusable here.
# Same risk for `auto_map`, which appears in HF config.json parsing
# regardless of error.
#
# Patterns below appear specifically in error/exception traces.
_ARCH_ERROR_PATTERNS = (
    "Model architectures",                 # e.g. "Model architectures ['Talkie...'] are not supported"
    "ValueError: The checkpoint",          # vLLM weight-checkpoint mismatch
    "ValueError: Unrecognized configuration",
    "Unable to load model",
    "Cannot infer model type",
    "is not a supported model type",
)
_QUANT_ERROR_PATTERNS = (
    "quantization is not supported",
    "Unsupported quant",
    "FP8 is not supported",
    # NOTE: bare "GPTQ" and "AWQ kernel" used to live here and were a
    # false-positive factory. The patterns are matched against the WHOLE
    # log, and an argparse usage dump enumerates every choice of
    # --quantization (…awq, gptq, gptq_marlin…). Seven SGLang cells were
    # therefore filed as `kind: quant` -- a permanent per-model verdict --
    # when the actual cause was this lab handing SGLang a vLLM-only
    # recovery flag. All seven recorded matched_pattern: "GPTQ".
    # Anything re-added here must be a phrase that cannot appear in help
    # text.
    "GPTQ quantization is not supported",
    "AWQ kernel is not supported",
)

# An argparse rejection is a defect in OUR launch construction, not a
# verdict about the model. It must be classified BEFORE the arch/quant
# cascade (a usage dump contains enough vocabulary to match several of
# those patterns) and must never produce a terminal capability, or the
# model is written off permanently for a flag we chose to send.
_LAUNCH_ARGS_ERROR_PATTERNS = (
    "unrecognized arguments",
    "error: argument",
    "invalid choice:",
    # vLLM rejects an unknown parser name at startup with a KeyError, not
    # through argparse (`validate_api_server_args`, as read in the vLLM
    # 0.28.0 source by the 2026-10-07 review). Our command line, not the
    # model: without these the failure filed as `infra`.
    "invalid tool call parser",
    "invalid reasoning parser",
)
_OOM_PATTERNS = (
    "CUDA out of memory",
    "torch.cuda.OutOfMemoryError",
    "RuntimeError: Allocator",
    "free; ",
    "no space left",
    # vLLM's KV-pool check (kv_cache_utils._check_enough_kv_cache_memory)
    # reports "does not fit at this ctx" without the words "out of memory":
    # the pool left after weights + activations is smaller than one request
    # at --max-model-len needs. Both branches are genuine memory limits.
    # Filed as `infra` until 2026-09-19 -- run-specific, so it never reached
    # the ledger and a model failing it at every tier was re-probed forever.
    # Keep these SPECIFIC: a bare "larger than" also appears in config
    # errors that really are infra (see test_unknown_is_infra).
    "larger than the available KV cache memory",
    "No available memory for the cache blocks",
)


# Lines that mark the START of the real failure, so the saved excerpt
# captures the root cause and not just the generic "Engine core
# initialization failed" tail that buried it (the gemma-4 case).
_FAILURE_ANCHORS = (
    "Traceback (most recent call last)",
    "ValueError", "RuntimeError", "AssertionError", "KeyError",
    "ImportError", "OSError", "Error:",
)


# Fraction of the generated text that may be a single repeated character
# before the output is judged degenerate. 0.9 is deliberately far from any
# healthy sample: the worst real reasoning trace on this fleet
# (Qwen3.5-9B, 477 coherent output tokens) is nowhere near it, while the
# Ornith failure was thousands of consecutive '!'.
_DEGENERATE_RUN_FRACTION = 0.9
# Below this many characters a "run" is not evidence of anything -- a
# short answer like "!!!" or "..." is legitimate.
_DEGENERATE_MIN_CHARS = 32


def is_degenerate_generation(
    content: str | None,
    reasoning: str | None,
    finish_reason: str | None,
    output_tokens: int | None,
    output_cap: int | None,
) -> tuple[bool, str]:
    """Detect an engine that is answering HTTP but not generating text.

    Returns ``(degenerate, reason)``; ``reason`` is "" when healthy.

    This exists because a dead CUDA kernel does not look like a failure
    over HTTP. `Ornith-1.0-9B-NVFP4` on SGLang raised a device-side assert,
    killed its scheduler, and then kept returning 200s whose content was
    empty and whose reasoning_content was thousands of identical '!'. The
    load probe scored that cell ``serving_ok: true`` because it computed
    ``not failed`` and consulted nothing about what came back.

    Two independent signals, either of which is decisive:

    1. **Empty content that ran to the output cap.** A model with nothing
       to say stops; it does not spend its entire budget saying nothing.
       BOTH conditions are required -- a reasoning model legitimately
       returns empty ``content`` when the answer is still inside a
       ``<think>`` block, and a short answer legitimately stops early.
    2. **A single character repeated across almost the whole output.**
       Catches the same corpse when it terminates before the cap.

    Deliberately NOT a recall check. ``needle_score`` is confounded on this
    fleet -- 3 of its 4 zeros were reasoning models that spent the answer
    budget on a ``<think>`` trace -- so gating on it would hide healthy
    models at a 100% false-positive rate. This predicate asks the narrower
    question "did the engine emit anything meaningful at all", which is
    answerable from the response alone.
    """
    text = (content or "").strip()
    reasoning_text = (reasoning or "").strip()

    if not text and output_cap and output_tokens and output_tokens >= output_cap:
        return True, (
            f"empty content with output at the {output_cap}-token cap "
            f"(finish_reason={finish_reason!r})"
        )

    for label, sample in (("content", text), ("reasoning_content", reasoning_text)):
        if len(sample) < _DEGENERATE_MIN_CHARS:
            continue
        run = _longest_char_run(sample)
        if run / len(sample) >= _DEGENERATE_RUN_FRACTION:
            return True, (
                f"{label} is {run}/{len(sample)} chars of a single repeated "
                f"character (>= {_DEGENERATE_RUN_FRACTION:.0%})"
            )
    return False, ""


def _longest_char_run(s: str) -> int:
    """Length of the longest run of one repeated character in `s`."""
    best = run = 0
    prev = None
    for ch in s:
        run = run + 1 if ch == prev else 1
        prev = ch
        if run > best:
            best = run
    return best


def _failure_excerpt(logs: str, context: int = 120) -> str:
    """Pick the most informative ~`context` lines of a failed-launch log.

    Anchors on the first explicit error/traceback line so the root cause is
    preserved (vLLM prints a long traceback ending in a generic wrapper; the
    real cause is near the top of it). Falls back to the tail when no anchor
    is found.
    """
    lines = logs.strip().splitlines()
    if len(lines) <= context:
        return "\n".join(lines)
    start = next(
        (i for i, ln in enumerate(lines)
         if any(a in ln for a in _FAILURE_ANCHORS)),
        None,
    )
    if start is None:
        return "\n".join(lines[-context:])
    start = max(0, start - 3)
    return "\n".join(lines[start:start + context])


def classify_failure_logs(logs: str) -> dict:
    """Inspect container logs after a failed launch and tag the cause.

    Returns {kind, log_excerpt, matched_pattern?}. kind ∈ {launch_args,
    arch, quant, oom_startup, infra}. The matched pattern is recorded when
    the cause is determined by a keyword match (everything except `infra`)
    so an operator can audit why a particular run was tagged the way it
    was. Pattern matching runs against the FULL log; `_failure_excerpt`
    then saves the root-cause region (not just the tail) for the cache.

    `launch_args` is checked FIRST and deliberately so. It means the
    engine rejected our command line, which says nothing whatsoever about
    the model -- and an argparse usage dump is long, quotes every
    supported quantisation and architecture name, and will happily match
    an arch or quant pattern if given the chance. That is exactly how
    seven SGLang cells acquired a permanent `quant` verdict for a flag
    this lab should never have sent them.
    """
    excerpt = _failure_excerpt(logs)
    lc = logs.lower()
    for pat in _LAUNCH_ARGS_ERROR_PATTERNS:
        if pat.lower() in lc:
            return {"kind": "launch_args", "log_excerpt": excerpt,
                    "matched_pattern": pat}
    for pat in _ARCH_ERROR_PATTERNS:
        if pat.lower() in lc:
            return {"kind": "arch", "log_excerpt": excerpt, "matched_pattern": pat}
    for pat in _QUANT_ERROR_PATTERNS:
        if pat.lower() in lc:
            return {"kind": "quant", "log_excerpt": excerpt, "matched_pattern": pat}
    for pat in _OOM_PATTERNS:
        if pat.lower() in lc:
            return {"kind": "oom_startup", "log_excerpt": excerpt, "matched_pattern": pat}
    return {"kind": "infra", "log_excerpt": excerpt}


# ── Probe driver ─────────────────────────────────────────────────────────────

def _resolve_plugins(
    spec: BackendSpec,
    reasoning_parser: str | None,
    tool_parser: str | None,
) -> tuple[tuple[str, str, str] | None, str | None, str | None]:
    """Look up parser names against the vllm-plugins registry.

    Returns ``(volume, tool_plugin_path, reasoning_plugin_path)`` where:
      - ``volume`` is a (host_dir, container_dir, mode) tuple to add to
        the launch when at least one parser resolved through the
        registry; ``None`` when no plugins are needed.
      - ``*_plugin_path`` is the in-container absolute path to pass via
        ``--*-parser-plugin``; ``None`` when the corresponding parser
        is a built-in (or absent).

    Backends without ``supports_plugins`` get all-``None`` results, so
    SGLang's launch is unchanged regardless of registry contents.
    """
    if not spec.supports_plugins:
        return None, None, None
    registry: PluginRegistry = get_registry()
    if not registry.entries:
        return None, None, None
    tool_entry: PluginEntry | None = registry.lookup(tool_parser)
    reasoning_entry: PluginEntry | None = registry.lookup(reasoning_parser)
    if tool_entry is None and reasoning_entry is None:
        return None, None, None
    # Validate that a tool plugin is registered under kind=tool and a
    # reasoning plugin under kind=reasoning. A mis-tagged registry entry
    # would silently pass the wrong flag — fail loudly instead.
    if tool_entry is not None and tool_entry.kind != "tool":
        raise RuntimeError(
            f"vllm-plugins: parser {tool_entry.name!r} resolved as "
            f"tool parser but registered under kind={tool_entry.kind!r}"
        )
    if reasoning_entry is not None and reasoning_entry.kind != "reasoning":
        raise RuntimeError(
            f"vllm-plugins: parser {reasoning_entry.name!r} resolved as "
            f"reasoning parser but registered under kind={reasoning_entry.kind!r}"
        )
    volume = (str(registry.host_dir), registry.container_dir, "ro")
    tool_path = (
        tool_entry.container_path(registry.container_dir)
        if tool_entry is not None else None
    )
    reasoning_path = (
        reasoning_entry.container_path(registry.container_dir)
        if reasoning_entry is not None else None
    )
    return volume, tool_path, reasoning_path


def run_tool_probe(
    base_url: str, model_name: str, tool_parser: str | None, capability: str,
    *, auto_only: bool = False,
) -> tuple[str | None, dict | None]:
    """Probe B of probe_one_cell: (verified tool parser or None, evidence).

    B1 sends tool_choice="auto"; when no call comes back, B2 forces a named
    call. For a curated or card-derived parser either verifies it. With
    `auto_only` (a guessed recipe / heuristic parser) only B1 does: a named
    tool_choice is satisfied by constrained decoding without the parser,
    so B2 cannot tell a working parser from a wrong one. The B2 result is
    then kept in the evidence with the reason, and nothing is recorded.
    """
    if not tool_parser:
        return None, None
    # Reasoning models burn 500-2000 tokens on chain-of-thought before
    # any tool-call markers. The old 128-token budget guaranteed they
    # never reached the call. 2048 covers typical R1-Distill traces.
    is_reasoning = capability in (Capability.STRUCTURED, Capability.INLINE)
    tool_max_tokens = 2048 if is_reasoning else 128
    auto_body = {
        "model": model_name,
        "messages": [{"role": "user", "content": TOOL_PROBE_PROMPT}],
        "tools": TOOL_PROBE_SPEC,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": tool_max_tokens,
        "stream": False,
    }
    auto_resp = _post_chat(base_url, auto_body)
    if response_has_valid_tool_call(auto_resp, "get_time"):
        return tool_parser, {"verified": True, "mode": "auto"}
    # Force a call so the parser is exercised even when the model
    # wouldn't pick a tool on its own. Keep the same prompt and
    # tools spec; just change tool_choice. Boost the budget once
    # more so reasoning + the forced call both fit.
    forced_body = dict(auto_body)
    forced_body["tool_choice"] = {
        "type": "function",
        "function": {"name": "get_time"},
    }
    forced_body["max_tokens"] = max(tool_max_tokens, 4096)
    forced_resp = _post_chat(base_url, forced_body)
    forced_ok = response_has_valid_tool_call(forced_resp, "get_time")
    if forced_ok and auto_only:
        return None, {
            "verified": False,
            "mode": "forced",
            "auto_only": True,
            "reason": "a guessed (recipe / heuristic) tool parser is recorded "
                      "only when the auto call verifies it; the forced call is "
                      "satisfied by constrained decoding without the parser",
            "auto_response_preview": _short(auto_resp),
            "forced_response_preview": _short(forced_resp),
        }
    if forced_ok:
        return tool_parser, {
            "verified": True,
            "mode": "forced",
            "auto_response_preview": _short(auto_resp),
        }
    return None, {
        "verified": False,
        "auto_response_preview": _short(auto_resp),
        "forced_response_preview": _short(forced_resp),
    }


def probe_one_cell(
    spec: BackendSpec,
    *,
    runtime: str,
    image: str,
    container_name: str,
    probe_port: int,
    models_dir: str,
    model_name: str,
    requested_ctx: int,
    band_gb: int,
    host_vram_gb: float,
    model_size_gb: float,
    prompt: str,
    reasoning_parser: str | None,
    tool_parser: str | None,
    mtp_method: str | None = None,
    mtp_drafter: str | None = None,
    mtp_num_tokens: int | None = None,
    tool_parser_auto_only: bool = False,
) -> dict:
    """Launch the backend once, run up to three HTTP probes against the
    same container, return a single record.

    Probes (all hit the same running server — no relaunch):
      A. Reasoning + fit. Send `prompt` to /v1/chat/completions and
         classify the response. Sets `capability` ∈
         {structured, inline, unsupported, error}.
      B. Tool-call. Only when `tool_parser` is set and the launch
         succeeded. Send TOOL_PROBE_PROMPT with TOOL_PROBE_SPEC; if the
         response carries a parseable tool_call to `get_time`, record
         `tool_parser=<curated-name>`; else null. See run_tool_probe;
         with `tool_parser_auto_only` (a guessed parser) only the auto
         call can verify it.
      C. Disable verification. Only when Probe A produced `structured`.
         Send the same prompt with the backend-specific suppression
         flag. If `reasoning_content` is then absent, record
         `disable_verified=true`.

    Always tears down the container before returning.
    """
    started = time.time()
    host_frac = host_scaled_fraction(
        model_size_gb, band_gb, host_vram_gb, spec.reserve_gb,
    )
    plugin_volume, tool_plugin_path, reasoning_plugin_path = _resolve_plugins(
        spec, reasoning_parser, tool_parser,
    )
    # Multi-token-prediction launch flag. Built JSON-side so the same
    # blob ends up in both probe-time and serve-time (router) launches.
    # SGLang discards this (its MTP path is unvalidated on this fleet -- NVFP4
    # loading itself is fixed via --disable-piecewise-cuda-graph); vLLM
    # appends `--speculative-config <json>` when present.
    speculative_config_json: str | None = None
    if mtp_method:
        spec_payload: dict[str, object] = {
            "method": mtp_method,
            "num_speculative_tokens": int(mtp_num_tokens or 1),
        }
        if mtp_drafter:
            # The drafter directory must already be on disk inside the
            # bound /models tree. Mirror the router's path convention so
            # the same JSON works at probe and serve time.
            spec_payload["model"] = f"/models/{mtp_drafter.split('/')[-1]}"
        speculative_config_json = json.dumps(spec_payload, separators=(",", ":"))
    # Fail before touching the GPU: the two engines accept DIFFERENT KV
    # dtype spellings and PROBE_KV_CACHE_TYPE is one knob shared by both.
    validate_kv_dtype(spec, spec.kv_cache_dtype)
    cmd_args = spec.build_args(
        model_name, requested_ctx, host_frac,
        reasoning_parser=reasoning_parser, tool_parser=tool_parser,
        reasoning_parser_plugin=reasoning_plugin_path,
        tool_parser_plugin=tool_plugin_path,
        speculative_config=speculative_config_json,
        kv_cache_dtype=spec.kv_cache_dtype,
    )

    env_vars = {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"}
    # Per-model recovery overrides (deploy/recovery-flags.json) — shared
    # with the router so probe and serve-time launches see the same flags.
    # Env entries override env_vars defaults on key collision; flags are
    # appended after parser flags (mirrors gpu-arbiter/main.go). Scoped by
    # backend so a vLLM-only entry is not applied to an SGLang launch.
    extra_flags, extra_env = recovery_overrides(model_name, spec.name)
    if extra_flags:
        cmd_args = list(cmd_args) + extra_flags
    if extra_env:
        env_vars = {**env_vars, **extra_env}
    # Per-model image override: probe on the SAME engine the router serves
    # on (mirrors gpu-arbiter buildContainerSpec). Without this a model
    # pinned to a non-default image probes on the wrong engine and records
    # a spurious fits=false cell.
    image_override = recovery_image(model_name, spec.name)
    if image_override and image_override != image:
        print(f"    [recovery] {model_name}: probing on pinned image "
              f"{image_override}", file=sys.stderr)
        image = image_override
    extra_volumes: list[tuple[str, str, str]] = []
    if plugin_volume is not None:
        extra_volumes.append(plugin_volume)
    container_remove(runtime, container_name)
    # Guarantee teardown: an exception (or a Ctrl-C surfacing as
    # KeyboardInterrupt) anywhere between here and the final record
    # would otherwise leave a GPU-holding probe container behind, which
    # assert_no_active_backends does not look for -- every later cell
    # would then measure against the leftover allocation and record
    # false OOMs / fits=false. The signal + atexit hooks installed by
    # install_probe_cleanup cover the paths this finally cannot.
    try:
        try:
            container_run_detached(
                runtime, container_name, image, probe_port, models_dir, env_vars,
                spec.entrypoint, cmd_args, extra_volumes=extra_volumes,
                backend=spec.name,
            )
        except RuntimeError as e:
            return _failure_record(
                ctx=requested_ctx, vram_gb=band_gb, started=started,
                evidence={
                    "kind": "infra", "error": str(e),
                    "reasoning_parser_attempted": reasoning_parser,
                    "tool_parser_attempted": tool_parser,
                },
            )

        base_url = f"http://127.0.0.1:{probe_port}"
        health_deadline = time.time() + STARTUP_TIMEOUT
        healthy = False
        last_err: str | None = None
        while time.time() < health_deadline:
            try:
                http_get(f"{base_url}/health", timeout=5.0)
                healthy = True
                break
            except urllib.error.URLError as e:
                last_err = f"URLError: {e}"
            except (TimeoutError, OSError) as e:
                last_err = f"{type(e).__name__}: {e}"
            except json.JSONDecodeError:
                healthy = True
                break

            # Fail-fast on early exit (arg errors, OOM during model load,
            # missing GPU). Without this check the loop would burn the full
            # STARTUP_TIMEOUT waiting on /health from a dead process.
            state = container_state(runtime, container_name)
            if state in ("exited", "stopped", "absent"):
                last_err = f"container {state} before /health"
                break
            time.sleep(HEALTH_POLL_INTERVAL)

        startup_seconds = round(time.time() - started, 2)
        if not healthy:
            logs = container_logs(runtime, container_name)
            evidence = classify_failure_logs(logs)
            evidence.setdefault("startup_error", last_err or "unknown")
            evidence["reasoning_parser_attempted"] = reasoning_parser
            evidence["tool_parser_attempted"] = tool_parser
            return _failure_record(
                ctx=requested_ctx, vram_gb=band_gb, started=started,
                evidence=evidence, startup_seconds=startup_seconds,
            )

        # Read what the engine actually accepted for max-model-len. Recorded
        # for evidence but no longer used as an early-exit gate — the chat
        # probe below decides whether the cell counts as fits=True. An engine
        # that silently clamps will still answer; an engine that breaks at
        # the requested ctx will fail the chat probe naturally.
        actual_max = 0
        try:
            models_resp = http_get(f"{base_url}/v1/models", timeout=10.0)
            for entry in (models_resp.get("data") or []):
                mm = entry.get("max_model_len")
                if isinstance(mm, int) and mm > actual_max:
                    actual_max = mm
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            actual_max = 0

        # ── Probe A: fit + reasoning ─────────────────────────────────────
        # When a reasoning_parser is being attempted, request the backend's
        # "enable thinking" surface so models with chat templates that
        # default `enable_thinking=false` (newer Qwen3, some Phi-4) still
        # emit a reasoning trace. This mirrors what the router's
        # applyVLLMPolicy / applySGLangPolicy does at serve time. Without it,
        # the probe sends a vanilla request and these models classify as
        # `unsupported` even though the parser would work in production.
        base_chat_body: dict = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            # Reasoning models need room for the full trace before the
            # final answer. 256 tokens often truncates mid-think; the
            # parser then can't bracket a `<think>...</think>` block and
            # silently drops the partial trace. 2048 covers Qwen3-8B's
            # typical CoT for medium-complexity prompts.
            "max_tokens": 2048,
            "stream": False,
        }
        if reasoning_parser:
            base_chat_body = build_enable_thinking_body(spec.name, base_chat_body)
        chat_resp = _post_chat(base_url, base_chat_body)

        used_mb = gpu_memory_used_mb()
        actual_vram_gb = round(used_mb / 1024, 2)

        capability, cap_evidence = classify_chat_response(
            chat_resp, reasoning_parser_attempted=reasoning_parser,
        )
        cap_evidence = dict(cap_evidence)
        cap_evidence["reasoning_parser_attempted"] = reasoning_parser
        cap_evidence["tool_parser_attempted"] = tool_parser
        # Forensic snapshot of the raw Probe A response — full content +
        # reasoning_content (capped) so we can debug parser-content
        # mismatches without re-launching. Replaces the older 200-char
        # previews; classify_chat_response still produces those for the
        # short-form summary.
        try:
            msg = ((chat_resp.get("choices") or [{}])[0].get("message") or {})
            cap_evidence["full_content"] = (msg.get("content") or "")[:2000]
            cap_evidence["full_reasoning_content"] = _extract_reasoning_content(msg)[:2000]
            cap_evidence["finish_reason"] = (chat_resp.get("choices") or [{}])[0].get("finish_reason")
        except (AttributeError, IndexError, TypeError):
            pass

        if capability == Capability.ERROR:
            return _failure_record(
                ctx=requested_ctx, vram_gb=band_gb, started=started,
                startup_seconds=startup_seconds, actual_vram_gb=actual_vram_gb,
                actual_context=actual_max,
                evidence={"kind": "oom_chat", **cap_evidence},
            )

        # Confirmed reasoning parser only when Probe A actually produced
        # `structured`. Any other capability means the curated parser
        # didn't apply (model emits inline or doesn't reason at all).
        reasoning_parser_verified = reasoning_parser if capability == Capability.STRUCTURED else None

        # ── Probe B: tool-call ───────────────────────────────────────────
        # Two-phase verification:
        #   B1 (auto):   tool_choice="auto"  — does the model spontaneously
        #                pick the tool? Useful signal for agents that send
        #                "auto", but reasoning models (R1-Distill et al)
        #                tend to ramble in reasoning_content instead of
        #                calling, so a "no" here doesn't mean the parser is
        #                broken.
        #   B2 (forced): tool_choice={"type":"function","function":{"name":...}}
        #                if B1 didn't yield a call. Forces the model to emit
        #                the tool-call markers so we can verify the parser
        #                actually extracts. Distinguishes "model declines
        #                to call" from "parser can't parse what was emitted".
        # Either path verifying counts as `tool_parser_verified=True` — the
        # router's strip-tools logic only cares whether the parser CAN
        # extract; agents that send explicit tool_choice still need the
        # plugin loaded even if the model wouldn't auto-pick.
        #
        # EXCEPT for a guessed parser (`tool_parser_auto_only`: a recipe
        # seed or the name heuristic). B2 is satisfied by constrained
        # decoding: vLLM 0.28 builds a named-choice call from the
        # constrained JSON without the tool parser, and SGLang skips its
        # detector for a named or required choice in the common case. So
        # B2 cannot tell a working parser from a wrong one, and a row
        # recorded `tool_mode=forced` answers every multi-tool request 400
        # at the router -- worse than stripping tools. A guess is recorded
        # only when B1 verified it.
        tool_parser_verified, tool_evidence = run_tool_probe(
            base_url, model_name, tool_parser, capability,
            auto_only=tool_parser_auto_only)

        # ── Probe C: disable verification ────────────────────────────────
        disable_verified = False
        disable_evidence: dict | None = None
        if capability == Capability.STRUCTURED:
            disable_body = build_disable_thinking_body(spec.name, base_chat_body)
            disable_resp = _post_chat(base_url, disable_body)
            # Falsifiable on BOTH surfaces. Absence of `reasoning_content`
            # alone is not evidence: a parsing switch can produce that
            # absence while the model thinks exactly as much as before and
            # merges the trace into `content`. Requiring no inline
            # `<think>` either means the check can actually fail.
            has_separate = response_has_reasoning_content(disable_resp)
            has_inline = response_has_inline_reasoning(disable_resp)
            if not has_separate and not has_inline and "error" not in disable_resp:
                disable_verified = True
                disable_evidence = {"verified": True}
            else:
                disable_evidence = {
                    "verified": False,
                    "reasoning_content_present": has_separate,
                    "inline_think_present": has_inline,
                    "response_preview": _short(disable_resp),
                }

        rec = {
            "ctx": requested_ctx,
            "vram_gb": band_gb,
            "fits": True,
            "actual_vram_gb": actual_vram_gb,
            "actual_context": actual_max or requested_ctx,
            "capability": capability,
            "reasoning_parser": reasoning_parser_verified,
            "tool_parser": tool_parser_verified,
            "disable_verified": disable_verified,
            "startup_seconds": startup_seconds,
            "probe_seconds": round(time.time() - started, 2),
            "probed_at": now_iso(),
            "evidence": cap_evidence,
        }
        # The launch SHAPE this fit was measured under. A cell measured
        # with a different set of flags is not evidence about today's
        # launch -- see launch_fingerprint.
        rec["launch_fingerprint"] = launch_fingerprint(cmd_args)
        if spec.kv_cache_dtype:
            # Fit is dtype-scoped: the router reproduces this dtype when
            # serving any ctx this cell covers (gpu-arbiter resolveKVCacheType).
            rec["kv_cache_type"] = spec.kv_cache_dtype
        if tool_evidence is not None:
            rec["evidence"]["tool"] = tool_evidence
        if disable_evidence is not None:
            rec["evidence"]["disable"] = disable_evidence
        return rec
    finally:
        container_remove(runtime, container_name)


def _post_chat(base_url: str, body: dict, timeout: float = CHAT_TIMEOUT) -> dict:
    """POST a chat-completion body, normalize errors to `{error: ...}`.

    `timeout` defaults to the fit-probe's short CHAT_TIMEOUT. The load
    probe overrides it with a context-scaled value -- a near-full
    256K-token prefill takes minutes, far longer than 60s, so the short
    default would mis-classify a still-progressing prefill as a serving
    failure.
    """
    try:
        return http_post(
            f"{base_url}/v1/chat/completions", body, timeout=timeout,
        )
    except urllib.error.HTTPError as e:
        raw = e.read() if hasattr(e, "read") else b""
        return {
            "error": f"HTTP {e.code}: {e.reason}",
            "body": raw[:200].decode(errors="replace"),
        }
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _short(resp: dict, n: int = 200) -> str:
    try:
        return json.dumps(resp)[:n]
    except (TypeError, ValueError):
        return str(resp)[:n]


def _failure_record(
    *,
    ctx: int,
    vram_gb: int,
    started: float,
    evidence: dict,
    startup_seconds: float | None = None,
    actual_vram_gb: float | None = None,
    actual_context: int = 0,
) -> dict:
    rec = {
        "ctx": ctx,
        "vram_gb": vram_gb,
        "fits": False,
        "actual_context": actual_context,
        "capability": Capability.ERROR,
        "evidence": evidence,
        "probed_at": now_iso(),
        "probe_seconds": round(time.time() - started, 2),
    }
    if startup_seconds is not None:
        rec["startup_seconds"] = startup_seconds
    if actual_vram_gb is not None:
        rec["actual_vram_gb"] = actual_vram_gb
    return rec


def _is_oom_kind(rec: dict) -> bool:
    if rec.get("fits", False):
        return False
    kind = ((rec.get("evidence") or {}).get("kind") or "")
    return kind in ("oom_startup", "oom_chat")


def _is_arch_kind(rec: dict) -> bool:
    kind = ((rec.get("evidence") or {}).get("kind") or "")
    return kind in ("arch", "quant")


def _latest_cell_with(entry: dict, field: str) -> dict | None:
    """Return the most-recently-probed clean cell whose `field` is truthy.

    Used by refresh_top_level_from_cells so freshly-probed cells with
    new parser info supersede stale cells (older probes often have
    ``tool_parser=None`` / ``reasoning_parser=None`` because they
    pre-date a curated family hint). Returns None when no cell carries
    the field.
    """
    best: dict | None = None
    best_at: str = ""
    for vram_bucket in (entry.get("probes") or {}).values():
        if not isinstance(vram_bucket, dict):
            continue
        cells = (
            [vram_bucket]
            if "capability" in vram_bucket
            else [c for c in vram_bucket.values() if isinstance(c, dict)]
        )
        for cell in cells:
            if not cell.get("fits", False):
                continue
            if not cell.get(field):
                continue
            at = str(cell.get("probed_at") or "")
            if at >= best_at:
                best, best_at = cell, at
    return best


# ── Top-level entry maintenance ──────────────────────────────────────────────

def _carry_forward_terminal(cache: dict, repo: str, new_entry: dict) -> bool:
    """Inherit a sibling sha's TERMINAL verdict for the same repo.

    A re-quant/commit changes the `repo@sha` cache key, so without this an
    `unsupported_arch` model re-probes under the new key every catalog-regen
    (the gemma double-key churn). Only `unsupported_arch` carries -- it does
    not depend on the exact weights. OOM is weight-specific and is re-checked
    on a new sha (plan decision 2); `infra`/`error` are run-specific.
    """
    for k, e in cache.items():
        if not isinstance(e, dict) or k.startswith("_") or e is new_entry:
            continue
        if e.get("repo") == repo and e.get("capability") == Capability.UNSUPPORTED_ARCH:
            new_entry["capability"] = Capability.UNSUPPORTED_ARCH
            new_entry["evidence"] = dict(e.get("evidence") or {})
            new_entry["carried_from_sha"] = e.get("sha")
            return True
    return False


def prune_orphaned_shas(cache: dict, catalog_rows: list[dict]) -> int:
    """Drop `repo@sha` entries stranded by a re-quant/commit.

    An orphan is a cache entry whose `repo` is still in the catalog but at a
    DIFFERENT (current) sha. Pruned ONLY when a current-sha entry for that
    repo already exists in the cache, so the last/only data for a repo is
    never lost (and a carried-forward terminal verdict already lives on the
    current-sha entry by the time this runs). Returns the count pruned.
    """
    current: dict[str, str] = {}
    for row in catalog_rows:
        repo = row.get("repo")
        sha = (row.get("sha") or "").strip()
        if repo and sha:
            current[repo] = sha
    have_current = {
        e.get("repo") for k, e in cache.items()
        if isinstance(e, dict) and not k.startswith("_")
        and e.get("repo") in current and e.get("sha") == current[e["repo"]]
    }
    orphans = [
        k for k, e in cache.items()
        if isinstance(e, dict) and not k.startswith("_")
        and e.get("repo") in current
        and e.get("sha") != current.get(e.get("repo"))
        and e.get("repo") in have_current
    ]
    for k in orphans:
        del cache[k]
    return len(orphans)


def _entry_fits_anywhere(entry: dict) -> bool:
    """True iff any (vram, ctx) cell loaded with fits=true -- i.e. the model
    serves on this host at some tier. Used to un-exclude a recovered model."""
    for band in (entry.get("probes") or {}).values():
        if not isinstance(band, dict):
            continue
        for cell in band.values():
            if isinstance(cell, dict) and cell.get("fits"):
                return True
    return False


_OOM_CELL_KINDS = frozenset({"oom_startup", "oom", "implied_spill"})


def _entry_oom_everywhere(entry: dict) -> bool:
    """True iff the model fits at NO tier and its failures are OOM (not arch).

    The fit probe walks ascending ctx and stops at the first OOM, so an entry
    that fits nowhere and whose cells are OOM-kind genuinely does not fit this
    GPU at any context -> a ledger `oom` exclusion (re-checked on a new sha).
    """
    if _entry_fits_anywhere(entry):
        return False
    saw_oom = False
    for band in (entry.get("probes") or {}).values():
        if not isinstance(band, dict):
            continue
        for cell in band.values():
            if not isinstance(cell, dict) or cell.get("fits"):
                continue
            kind = (cell.get("evidence") or {}).get("kind")
            if kind in _OOM_CELL_KINDS:
                saw_oom = True
            elif kind in ("arch", "quant"):
                return False  # terminal arch/quant owns this, not oom
    return saw_oom


def ensure_entry(
    cache: dict, key: str, repo: str, sha: str, alias: str,
    schema_version: int, model_kind: str, weight_size_gb: float,
) -> dict:
    entry = cache.get(key)
    if entry is None:
        entry = {
            "schema_version": schema_version,
            "repo": repo,
            "sha": sha,
            "aliases": [alias],
            "model_kind": model_kind,
            # Catalog-declared weight size on disk. The router uses this
            # for memFraction launch math; without it, the synthesizer
            # would mis-read actual_vram_gb (which is post-load,
            # weights + KV + CUDA graphs) as the weight size and clamp
            # --max-model-len to a few thousand tokens.
            "size_gb": weight_size_gb,
            "max_context": 0,
            "capability": Capability.UNKNOWN,
            # v2: confirmed parser names from the first probed cell. Null
            # when the curated family hint did not produce a verified
            # round-trip (or when no parsers were curated).
            "reasoning_parser": None,
            "tool_parser": None,
            # v2: True iff the disable-thinking probe produced a response
            # with empty `reasoning_content`. Mirrors Ollama's
            # `disable_verified` field; gates the router's reasoningOff
            # rewrite.
            "disable_verified": False,
            # tool_mode ("auto"|"forced"|None) records HOW the tool
            # parser was verified — `auto` if the model spontaneously
            # called the tool with tool_choice="auto", `forced` if the
            # call only happened with tool_choice={function:{name:...}}.
            # The router uses this to promote tool_choice on incoming
            # requests for `forced` models (single-tool only) or fail
            # multi-tool requests with an actionable error.
            "tool_mode": None,
            "evidence": {},
            "probes": {},
        }
        cache[key] = entry
        # A prior sha of this repo that was terminally unsupported_arch makes
        # this fresh entry inherit the verdict, so run_probe_pass skips it
        # instead of re-probing a known-dead arch after a re-quant.
        _carry_forward_terminal(cache, repo, entry)
    else:
        if alias and alias not in entry.get("aliases", []):
            entry.setdefault("aliases", []).append(alias)
        # Update model_kind if upgraded from unknown
        if entry.get("model_kind") in (None, "unknown") and model_kind != "unknown":
            entry["model_kind"] = model_kind
        # Backfill size_gb on entries written before the field existed.
        if not entry.get("size_gb") and weight_size_gb > 0:
            entry["size_gb"] = weight_size_gb
        # v1 → v2 backfill. New fields default to non-verified so an
        # upgraded cache produces the same router behaviour as v1 until
        # a fresh probe lands.
        entry.setdefault("reasoning_parser", None)
        entry.setdefault("tool_parser", None)
        entry.setdefault("disable_verified", False)
        # tool_mode (auto|forced|None) — added alongside the two-phase
        # tool probe. None for v2 entries written before the field
        # existed; refresh_top_level_from_cells repopulates from
        # cell-level evidence.tool.mode on the next probe.
        entry.setdefault("tool_mode", None)
        # NOTE: schema_version is deliberately NOT bumped here. The v1->v2
        # fields above are backfilled with non-verified defaults; declaring
        # the entry v2 on that basis defeats the router's deliberate v1
        # refusal (which exists precisely because a v1 entry has no probed
        # tool_parser / disable_verified) and silently re-creates the
        # tool-stripping state that refusal guards against. run_probe_pass
        # bumps the version only after fresh cells have actually been
        # measured by this (v2) probe driver.
    return entry


def refresh_top_level_from_cells(entry: dict) -> None:
    """Re-derive top-level capability and parser fields from ALL recorded
    cells. Idempotent — call after every cell write.

    The bug-prone alternative was setting top-level fields imperatively
    from the most-recently-written cell: a 16G/oom_startup cell would
    overwrite a 24G/structured success because each cell's outcome was
    treated as authoritative. The fix is to derive top-level state
    from the corpus of cells, picking the smallest-ctx, smallest-vram
    clean probe as canonical.

    Terminal states are sticky:
      - `unsupported_arch` (backend doesn't recognise the arch — won't
        change without an image bump);
      - `error` with evidence.kind ∈ {arch, quant} (load failed for a
        reason that's tier-independent).
    Non-terminal `error` (kind=oom/infra/clamped_ctx) at one tier does
    NOT downgrade a successful classification at a higher tier — a 16G
    spill doesn't invalidate a 24G fit.

    ...but stickiness yields to a CLEAN CELL. A tier-independent load
    failure and a probe that loaded and produced structured output cannot
    both be true of the same (repo, sha, backend): if the engine answered,
    it supports the arch. Without this escape, a stale terminal verdict
    outlived the evidence that disproved it and `--force` could not clear
    it -- observed 2026-07-25 on ykarout/Qwen3.5-9B-NVFP4 under SGLang,
    which re-probed clean at 256K (structured, rp=qwen3, tp=qwen) while
    the top-level row stayed `error`/kind=quant with both parsers None
    from a July run. The router injects `--reasoning-parser` /
    `--tool-call-parser` from those top-level fields, so the model would
    have been served with no parsers at all -- silently losing tool
    calling and reasoning on a model that probes perfectly.
    """
    cur_cap = entry.get("capability") or Capability.UNKNOWN
    cur_kind = ((entry.get("evidence") or {}).get("kind") or "")
    # Stickier than the global TERMINAL set: an `error` with arch/quant
    # evidence is tier-independent (it'll fail at every VRAM band),
    # whereas an `error` with oom/infra evidence is tier-specific and
    # MUST allow a higher-tier success to overwrite it.
    cur_is_terminal = (
        cur_cap == Capability.UNSUPPORTED_ARCH
        or (cur_cap == Capability.ERROR and cur_kind in ("arch", "quant"))
    )

    smallest = smallest_clean_probe(entry)
    if cur_is_terminal and smallest is None:
        return
    if smallest is not None:
        entry["capability"] = smallest.get("capability") or Capability.UNKNOWN
        ev = smallest.get("evidence") or {}
        if ev:
            entry["evidence"] = ev
        # Parser fields propagate from the most-recently-probed cell that
        # has them populated — NOT from the smallest_clean_probe. Reason:
        # older cells often pre-date a curated parser hint or the
        # two-phase tool probe, so they have `tool_parser=None` /
        # `tool_mode=None` even when newer cells at higher tiers verified
        # the parser cleanly. Picking from `smallest` would shadow the
        # fresh evidence with stale Nones. Walking all cells and taking
        # the latest probed_at lets `--force` on a single cell update
        # the top-level row without requiring a full re-probe matrix.
        rp_cell = _latest_cell_with(entry, "reasoning_parser")
        tp_cell = _latest_cell_with(entry, "tool_parser")
        entry["reasoning_parser"] = (rp_cell or {}).get("reasoning_parser")
        entry["tool_parser"] = (tp_cell or {}).get("tool_parser")
        # tool_mode tracks WHICH tool-choice path verified — `auto` (the
        # model spontaneously called) vs `forced` (only forced-choice
        # round-tripped). Pulled from the same cell as tool_parser so the
        # two stay consistent. Stays None when no cell verified a parser.
        if tp_cell is not None:
            cell_tool_evidence = (tp_cell.get("evidence") or {}).get("tool") or {}
            entry["tool_mode"] = cell_tool_evidence.get("mode")
        else:
            entry["tool_mode"] = None
        # disable_verified is a per-cell verdict that doesn't drift with
        # tier the way capability/parser do; pick the latest cell that
        # ran the disable probe (which only fires on capability=structured).
        dv_cell = _latest_cell_with(entry, "disable_verified") or smallest
        entry["disable_verified"] = bool(dv_cell.get("disable_verified", False))
        # max_context = largest actual_context across ALL clean cells.
        # Using `smallest`'s actual_context here was a long-standing bug:
        # it left max_context pinned to the smallest verified tier (e.g.
        # 32K) even when 256K had also been verified as fitting. The
        # picker's _measured_point_at clamps requested ctx to max_context,
        # so the bug hid every higher-tier row from the menu.
        largest_ctx = 0
        for vram_bucket in (entry.get("probes") or {}).values():
            if not isinstance(vram_bucket, dict):
                continue
            cells = (
                [vram_bucket]
                if "capability" in vram_bucket
                else [c for c in vram_bucket.values() if isinstance(c, dict)]
            )
            for cell in cells:
                if cell.get("capability") in (None, Capability.ERROR):
                    continue
                ac = int(cell.get("actual_context") or 0)
                if ac > largest_ctx:
                    largest_ctx = ac
        # Cap at the model's HARD position limit when known. The fit
        # probe can load (fits=true) at a ctx beyond max_position_embeddings
        # because VLLM_ALLOW_LONG_MAX_MODEL_LEN bypasses vLLM's guard, but
        # the model asserts on any prompt that long at serve time -- so
        # max_context must not advertise it. position_limit is stamped on
        # the entry by run_probe_pass from config.json (see
        # effective_position_limit).
        pos_limit = entry.get("position_limit")
        if isinstance(pos_limit, int) and pos_limit > 0:
            largest_ctx = min(largest_ctx, pos_limit)
        # max_context tracks the largest clean probed actual_context (capped
        # at the position limit). Sync on ANY difference: it GROWS when a
        # higher tier is verified, SHRINKS when the position limit caps an
        # over-promise, and -- crucially for the single-cell binary search --
        # SHRINKS when a full re-probe replaces a multi-cell entry whose stale
        # max exceeded the new lone winner (e.g. 40960 -> 32768). The cell
        # loop above already spans every remaining cell, so largest_ctx is the
        # true max across the current set; a shrink is never a partial-probe
        # artefact. Guard on truthiness so a fits-nowhere entry keeps its max.
        if largest_ctx and largest_ctx != (entry.get("max_context") or 0):
            entry["max_context"] = largest_ctx
        return

    # No clean probe in the corpus — pick the most-severe failure kind
    # to set capability.
    #
    # `launch_args` is severity 0: the engine rejected OUR command line, so
    # the run produced no evidence about the model at all. Giving it any
    # positive severity would write Capability.ERROR, which the router
    # treats as terminal -- permanently writing off a model because this
    # lab sent it a flag it does not accept. That is precisely how seven
    # SGLang cells were condemned. The run must fail loudly (the caller
    # aborts the pass) and leave the previous verdict alone.
    severities = {"arch": 3, "quant": 2, "launch_args": 0}  # others = 1
    chosen_severity = 0
    chosen_ev: dict = {}
    for value in (entry.get("probes") or {}).values():
        if not isinstance(value, dict):
            continue
        cells = [value] if "capability" in value else [
            c for c in value.values() if isinstance(c, dict)
        ]
        for cell in cells:
            ev = cell.get("evidence") or {}
            kind = ev.get("kind")
            if not kind:
                continue
            sev = severities.get(kind, 1)
            if sev > chosen_severity:
                chosen_severity = sev
                chosen_ev = ev
    if chosen_severity == 3:
        entry["capability"] = Capability.UNSUPPORTED_ARCH
        entry["evidence"] = chosen_ev
    elif chosen_severity >= 1:
        entry["capability"] = Capability.ERROR
        entry["evidence"] = chosen_ev
    else:
        entry.setdefault("capability", Capability.UNKNOWN)


# ── Argparse builder ─────────────────────────────────────────────────────────

_ENGINE_MAX_LEN_RE = re.compile(r"estimated maximum model length is (\d+)")


def engine_max_len_estimate(evidence: dict | None) -> int | None:
    """vLLM's own answer to "how much would have fit", parsed from the
    failure evidence of a launch that asked for too much context. The
    engine prints it when the KV budget cannot hold one request at
    --max-model-len ("the estimated maximum model length is N"). None for
    every other kind of failure (a weights OOM says nothing about context).
    """
    if not isinstance(evidence, dict):
        return None
    m = _ENGINE_MAX_LEN_RE.search(str(evidence.get("log_excerpt") or ""))
    return int(m.group(1)) if m else None


def exact_ctx_grid(ctx_exact: int, position_limit: int | None) -> tuple[int, ...]:
    """The one-element search grid for --ctx-exact, or ValueError when the
    value lies past the checkpoint's own ceiling. Refused here rather than
    left to binary_search_max_ctx: the search path turns "every tier above
    position_limit" into an unsupported_arch verdict, which is a statement
    about the checkpoint, not about a mistyped number.
    """
    if ctx_exact <= 0:
        raise ValueError(f"--ctx-exact must be positive, got {ctx_exact}")
    if position_limit is not None and ctx_exact > position_limit:
        raise ValueError(
            f"--ctx-exact {ctx_exact} is above the model's position limit "
            f"{position_limit}; nothing above it can serve")
    return (ctx_exact,)


def _parse_ctx_exact(raw: str) -> int:
    values = parse_context_list(raw)
    if len(values) != 1:
        raise argparse.ArgumentTypeError("--ctx-exact takes exactly one context")
    return values[0]


def build_argparser(spec: BackendSpec, doc: str) -> argparse.ArgumentParser:
    """Build an argparse for a backend prober. Backend-specific defaults
    (cache path, image, container name, port) come from the spec; the
    rest are common across backends.
    """
    ap = argparse.ArgumentParser(description=doc.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=spec.cache_path,
                    help=f"cache file (default {spec.cache_path.relative_to(REPO_ROOT)})")
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG,
                    help="models.yaml path")
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR,
                    help="host directory holding HF model trees")
    ap.add_argument("--image", default=spec.image,
                    help=f"{spec.name} container image")
    ap.add_argument("--container-name", default=spec.container_name,
                    help="probe container name (auto-removed before each launch)")
    ap.add_argument("--probe-port", type=int, default=spec.probe_port,
                    help="localhost loopback port published by the probe container")
    ap.add_argument("--runtime", default=os.environ.get("CONTAINER_RUNTIME", "podman"),
                    help="container runtime CLI (podman or docker)")
    ap.add_argument("--host-vram-gb", type=int, default=DEFAULT_HOST_VRAM_GB,
                    help="physical GPU VRAM (used for fraction scaling)")
    ap.add_argument("--vram", default="",
                    help="comma-separated VRAM bands, e.g. '16G,24G' (default: standard tiers)")
    ap.add_argument("--ctx", default="",
                    help="comma-separated context tiers, e.g. '32K,128K' (default: standard tiers)")
    ap.add_argument("--repo", default="",
                    help="regex filter on catalog row repo")
    ap.add_argument("--ctx-exact", type=_parse_ctx_exact, default=None,
                    help="probe exactly ONE context (e.g. 110K or 112640) with no "
                         "binary search and no 32K rounding, for confirming a "
                         "ceiling computed in advance. Implies --force for the "
                         "targeted rows and requires --repo. A miss leaves the "
                         "existing cell in place and reports the engine's own "
                         "estimate of what would have fit.")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="probe prompt (kept short and deterministic)")
    ap.add_argument("--force", action="store_true",
                    help="re-probe every requested cell even if cached")
    ap.add_argument("--force-arch", action="store_true",
                    help="re-probe top-level arch/capability fields")
    ap.add_argument("--no-cache-write", action="store_true",
                    help="dry run — do not modify the cache")
    ap.add_argument("--ignore-curated-parsers", action="store_true",
                    default=ignore_curated_parsers_from_env(),
                    help="skip the curated parsers: block, the recipe seed and "
                         "the name heuristic, so card-derived parsers alone "
                         "drive the launch (plan I24; default from "
                         "PROBE_IGNORE_CURATED_PARSERS=1)")
    ap.add_argument("--no-mtp", action="store_true",
                    help="skip the per-cell MTP overhead probe for "
                         "catalog rows declaring an `mtp:` block. Halves "
                         "wall-time when MTP overhead is not needed.")
    ap.add_argument("--load", action="store_true",
                    help="run the serving-time LOAD probe instead of the "
                         "fit probe: relaunch each already-fitting cell, "
                         "send a near-full-context request under a VRAM "
                         "sampler, record serving_ok/transient/needle. "
                         "Ascends ctx tiers and stops at the first OOM. "
                         "Requires the fit cache to already exist.")
    ap.add_argument("--needle-depth", type=float, default=0.5,
                    help="fractional depth (0.0=top, 1.0=bottom) at which "
                         "the load probe inserts its recall needle "
                         "(default 0.5)")
    return ap


# ── Main loop ────────────────────────────────────────────────────────────────

def check_ignore_curated(spec: BackendSpec, args: argparse.Namespace) -> bool:
    """I24's switch, refused where it would rewrite production data.

    PROBE_IGNORE_CURATED_PARSERS=1 probes with the card-derived parsers
    alone, so its cells must not replace the curated ones: it is accepted
    only scoped to rows (`--repo`) and with a scratch cache (`--cache` other
    than the backend's own). `--no-cache-write` is refused: it makes a row
    that already fits look fully cached, so the run would probe nothing.
    The exclusion ledger is not written (Phase 4v adds a scratch
    `--ledger`). The --load pass refuses the switch (run_load_probe_pass).
    Read like `no_mtp`: hand-built Namespaces in the tests predate the flag.
    """
    if not getattr(args, "ignore_curated_parsers", False):
        return False
    scratch = Path(args.cache).resolve() != Path(spec.cache_path).resolve()
    if not args.repo or not scratch or getattr(args, "no_cache_write", False):
        sys.exit("error: PROBE_IGNORE_CURATED_PARSERS=1 / --ignore-curated-parsers "
                 "needs --repo <regex> (PROBE_REPO) and a scratch --cache <file> "
                 "other than the backend's own, and refuses --no-cache-write: its "
                 "cells carry card-derived parsers only and must not replace the "
                 "curated ones (plan I24)")
    print("PROBE_IGNORE_CURATED_PARSERS: curated parsers, recipe seeds and "
          "the name heuristic are ignored; only card-derived parsers drive "
          "the launch; the exclusion ledger is not written", file=sys.stderr)
    return True


def run_probe_pass(spec: BackendSpec, args: argparse.Namespace) -> None:
    """Run one probe pass with the given backend spec and argparse args."""
    ignore_curated = check_ignore_curated(spec, args)
    assert_no_active_backends(args.runtime)
    install_probe_cleanup()

    catalog_rows = load_catalog_hf_rows(args.catalog, spec.name)
    if args.repo:
        rx = re.compile(args.repo)
        catalog_rows = [r for r in catalog_rows if rx.search(r.get("repo") or "")]
    if not catalog_rows:
        sys.exit("error: no HF rows match — check --repo filter or models.yaml")

    models_dir = Path(args.models_dir)
    if not models_dir.is_dir():
        sys.exit(f"error: models dir not found: {models_dir}")

    seeds = {} if ignore_curated else _vllm_recipes.load_seeds()

    # Default to ONLY the host VRAM. Probing 16G when the host has 24G
    # produces data nobody on this host can use; the operator can still
    # ask for sub-host bands explicitly via `--vram 16,24`.
    vrams = parse_vram_list(args.vram) if args.vram else [int(args.host_vram_gb)]
    ctxs = parse_context_list(args.ctx) if args.ctx else standard_contexts()
    if not vrams or not ctxs:
        sys.exit("error: --vram or --ctx produced an empty list")
    # The requested tiers CAP the binary search. Without this the search
    # walked the full 32K-multiple grid up to 256K regardless of --ctx /
    # PROBE_CONTEXTS, launching (and OOM-killing) containers at tiers the
    # operator explicitly excluded. Keeping the fine grid *below* the
    # requested ceiling preserves the documented 32K-multiple precision on
    # a default run (whose ceiling is already 256K) while `--ctx 32K` now
    # launches exactly one tier. Requested tiers that are not grid
    # multiples are unioned in so an odd explicit tier is still probed.
    ctx_ceiling = max(ctxs)
    search_grid = tuple(sorted(
        {c for c in BINARY_SEARCH_CONTEXTS if c <= ctx_ceiling} | set(ctxs)
    ))
    # Read like `no_mtp`: hand-built Namespaces in the tests predate the flag.
    ctx_exact = getattr(args, "ctx_exact", None)
    if ctx_exact is not None:
        # Exact mode: one launch at the requested value, nothing else. The
        # position_limit check happens per row below (it is a per-model
        # figure); here only the grid and the scoping rule.
        if not args.repo:
            sys.exit("error: --ctx-exact is a per-model operation; scope it with --repo")
        ctxs = [ctx_exact]
        search_grid = (ctx_exact,)

    cache = load_cache(args.cache)
    # Phase C: record the backend image digest this cache is being probed
    # against so the router can detect a moved tag (drift) that silently
    # invalidates it. Guarded save lands `_meta` even on an all-cached run.
    # Kept in a local as well: `_meta` answers "what image was this cache
    # last touched by", which a PARTIAL re-probe makes misleading on its
    # own. Each row this pass rewrites is stamped individually so an
    # un-reprobed row stays distinguishable -- see stamp_row_image.
    run_image_digest = image_digest_via_cli(args.runtime, args.image)
    # Order matters: attribute legacy rows to the OUTGOING `_meta` digest
    # before the pointer moves, otherwise their true provenance is lost.
    backfilled = backfill_row_images(cache)
    if backfilled:
        print(f"  backfilled:     {backfilled} legacy row(s) attributed to the "
              f"previous image digest", file=sys.stderr)
    stamp_image_digest(
        cache, digest=run_image_digest, image_ref=args.image)
    if not args.no_cache_write:
        save_cache(args.cache, cache)
    ledger = _load_ledger()

    print(f"  prober:         probe-{spec.name}-reasoning", file=sys.stderr)
    print(f"  cache:          {args.cache} ({len(cache)} entries)", file=sys.stderr)
    print(f"  catalog rows:   {len(catalog_rows)} (after --repo filter)",
          file=sys.stderr)
    print(f"  vram bands:     {','.join(vram_label(v) for v in vrams)}",
          file=sys.stderr)
    print(f"  ctx tiers:      {','.join(context_label(c) for c in ctxs)}",
          file=sys.stderr)
    print(f"  ctx search grid:{','.join(context_label(c) for c in search_grid)}",
          file=sys.stderr)
    print(f"  host vram:      {args.host_vram_gb}G", file=sys.stderr)
    print(f"  {spec.name} image:     {args.image}", file=sys.stderr)
    print(file=sys.stderr)

    fresh_probes = 0
    exact_misses = 0
    fully_cached = 0
    skipped_missing = 0
    skipped_arch = 0
    skipped_excluded = 0
    skipped_pos_limit = 0

    for row in catalog_rows:
        name = row.get("name") or ""
        repo = row.get("repo") or ""
        sha = (row.get("sha") or "").strip()
        if not (name and repo and sha):
            print(f"  [skip] catalog row missing fields: name={name!r} "
                  f"repo={repo!r} sha={sha!r}", file=sys.stderr)
            continue

        # Host-local exclusion ledger: a too_big / too_small / unsupported_arch
        # model is skipped SILENTLY (no "not on disk" noise) and never probed.
        # --force-arch re-evaluates a cached unsupported_arch, so honor it here
        # too. Stability rules (vram/sha) live in is_excluded.
        if (not args.force_arch
                and _ledger_is_excluded(ledger, name, spec.name,
                                        host_vram=args.host_vram_gb, sha=sha)):
            skipped_excluded += 1
            continue

        # A VRAM verdict already measured on a roomier backend applies
        # here too, so probing would burn a cold start to rediscover a
        # known answer. One-way and reason-scoped -- see
        # _model_status._VRAM_IMPLIED_BY. Announced rather than silent:
        # this is the one skip whose evidence lives under a DIFFERENT
        # backend, so a reader looking at this backend's ledger would
        # otherwise find no reason for it.
        if not args.force_arch:
            implied = _ledger_implied_vram_exclusion(
                ledger, name, spec.name,
                host_vram=args.host_vram_gb, sha=sha)
            if implied:
                src, why = implied
                print(f"  [skip] {name}: {src} recorded {why!r} at this VRAM, "
                      f"and {spec.name} needs strictly more (reserve "
                      f"{spec.reserve_gb:g} GB, unquantized KV) -- so it "
                      f"cannot fit here either. PROBE_FORCE_ARCH=1 to probe "
                      f"anyway.", file=sys.stderr)
                skipped_excluded += 1
                continue

        if not is_downloaded(name, models_dir):
            print(f"  [skip] {name}: not on disk under {models_dir}",
                  file=sys.stderr)
            skipped_missing += 1
            continue

        kind = model_kind_from_disk(name, models_dir)
        size_gb = model_size_gb_from_row(row)
        key = f"{repo}@{sha}"
        entry = ensure_entry(
            cache, key, repo, sha, name, spec.schema_version, kind, size_gb,
        )

        # Stamp the HARD position limit from config.json so
        # refresh_top_level_from_cells caps max_context at it and the
        # tier loop below never probes a context the model can't serve.
        pos_limit = effective_position_limit(name, models_dir)
        if isinstance(pos_limit, int) and pos_limit > 0:
            entry["position_limit"] = pos_limit

        # Pull curated parser hints for this backend from the catalog
        # row. The probe driver passes these to spec.build_args; the
        # cell record only confirms them when the backend round-trip
        # produces the expected response shape.
        curated_reasoning, curated_tool = curated_parsers(row, spec.name)

        # The parser order (plan Phase 17a): curated, then card-derived
        # (which refuses to guess wherever validation showed the markup
        # does not determine the parser -- qwen3_xml vs qwen3_coder, bare
        # <think>, Gemma-4's prompt-side <|think|>), then the recipe seed,
        # then the labelled name heuristic, then none. A curated value
        # ALWAYS wins unless PROBE_IGNORE_CURATED_PARSERS=1 (I24), which
        # leaves the card derivation alone.
        curated_block = has_curated_block(row, spec.name)
        recipe_parsers = ({} if ignore_curated or curated_block else
                          recipe_seed_parsers(name, spec.name, args.image, seeds))
        # Guesses an earlier run dropped (see _fit_works): skipped again.
        prior_fallback = entry.get("parser_fallback")
        rejected = ({k: v["parser"] for k, v in prior_fallback.items()
                     if isinstance(v, dict) and isinstance(v.get("parser"), str)}
                    if isinstance(prior_fallback, dict) else {})
        (row_reasoning_parser, row_tool_parser, parser_sources,
         disagreements) = resolve_parsers(
            name, models_dir, spec.name, curated_reasoning, curated_tool,
            recipe_parsers=recipe_parsers, ignore_curated=ignore_curated,
            curated_block=curated_block, rejected=rejected,
        )
        # Guessed values a failed first launch may be blamed on; see _fit_works.
        guessed_kinds = [k for k, src in parser_sources.items()
                         if src in GUESSED_PARSER_SOURCES]

        # Provenance. Additive fields only, matching how the LOAD probe
        # augments cells -- no schema bump. Lets a reader tell "a human
        # chose this" from "the checkpoint's own template implied it".
        # Sampling defaults from the checkpoint's generation_config.json.
        # Stamped HERE, host-side, because the bench container mounts
        # scripts/ + deploy/ but deliberately NOT the model weights -- so
        # the runner can read this from /deploy without a weights mount it
        # has no other reason to have.
        try:
            card_sampling = _card_hints.sampling_defaults(models_dir / name)
        except Exception:  # noqa: BLE001
            card_sampling = {}
        if card_sampling:
            entry["card_sampling"] = card_sampling
        else:
            entry.pop("card_sampling", None)

        def _stamp_parser_provenance() -> None:
            # Written only when this run probes the row, so the fields
            # describe the parsers its cells were measured with; a cached
            # row keeps the provenance of the run that measured it.
            entry["tool_parser_source"] = parser_sources.get("tool")
            entry["reasoning_parser_source"] = parser_sources.get("reasoning")
            if disagreements:
                entry["parser_disagreement"] = disagreements
            else:
                entry.pop("parser_disagreement", None)
            if parser_fallback:
                entry["parser_fallback"] = dict(parser_fallback)
            else:
                entry.pop("parser_fallback", None)

        # A recorded fallback stays while it still keeps the same guess out.
        parser_fallback: dict[str, dict] = {
            k: prior_fallback[k] for k, v in rejected.items()
            if not ignore_curated and not curated_block
            and parser_sources.get(k) not in ("curated", "card")
            and v in (recipe_parsers.get(k), heuristic_parser(name, spec.name, k))
        }
        first_launch_pending = True

        # A row measured under a DIFFERENT engine image is not evidence
        # about this one. Unlike launch-fingerprint drift (see the long
        # note further down, deliberately report-only), the image digest
        # is a single value known exactly for this run, so comparing it
        # per row cannot produce a false positive -- and an
        # `unsupported_arch` verdict is precisely the kind that an engine
        # bump overturns. Measured: Gemma-4 was unsupported_arch on
        # v0.5.10.post1 and loads on v0.5.16, but the cached verdict
        # skipped it, so the bump silently rescued nothing.
        entry_image = entry.get(ROW_IMAGE_FIELD)
        stale_image = row_image_is_stale(
            entry, run_image_digest, writable=not args.no_cache_write)

        if args.force_arch or stale_image:
            if stale_image and entry.get("capability") == \
                    Capability.UNSUPPORTED_ARCH:
                print(f"  {name}: cached unsupported_arch was measured on a "
                      f"different engine image — re-evaluating",
                      file=sys.stderr)
            entry["capability"] = Capability.UNKNOWN
            entry["evidence"] = {}

        if (entry.get("capability") == Capability.UNSUPPORTED_ARCH
                and not args.force_arch):
            print(f"  [skip] {name}: unsupported_arch (cached)",
                  file=sys.stderr)
            _ledger_record(ledger, name, spec.name, "unsupported_arch",
                           detail=(entry.get("evidence") or {}).get(
                               "matched_pattern") or "arch load failure",
                           repo=repo, host_vram=args.host_vram_gb, sha=sha)
            skipped_arch += 1
            continue

        first_seen_record: dict | None = None
        arch_or_quant_seen = False

        for vram_gb in vrams:
            if arch_or_quant_seen:
                break
            band = entry.setdefault("probes", {}).setdefault(str(vram_gb), {})

            # A cache entry written by an OLDER schema version must be
            # RE-probed, not merely re-labelled: the router refuses v1
            # entries precisely because they carry no probed parser /
            # disable verdicts, and its own error message tells the
            # operator to run `make probe-vllm` / `make probe-sglang`.
            # Without this bypass a populated v1 band short-circuits, the
            # version bump at the end of this loop (which is gated on
            # freshly probed cells, by design) never runs, and that
            # documented instruction can never succeed without
            # PROBE_FORCE=1. Mirrors --force in also requiring that the
            # result can actually be written.
            stale_schema = (
                entry.get("schema_version", 1) < spec.schema_version
                and not args.no_cache_write
            )
            if stale_schema and band:
                print(f"  {name}: cache entry is schema v"
                      f"{entry.get('schema_version', 1)} < v"
                      f"{spec.schema_version} — re-probing to upgrade",
                      file=sys.stderr)

            # Keep-one-cell: a populated band already holds the single
            # binary-searched result from a prior run. Re-probe only under
            # --force (which rewrites the cell, not appends), when the
            # entry predates the current schema version, when the band
            # holds NO fitting cell, or when the launch shape has changed.
            #
            # "Band is non-empty" was the old condition and it was too
            # weak: a band of nothing but FAILURES counted as complete, so
            # a model that failed once was never retried even after the
            # cause was fixed -- which is precisely how seven SGLang cells
            # kept an argparse-rejection verdict for three weeks after the
            # flag bug was fixed.
            #
            # Launch-fingerprint drift is deliberately NOT an
            # auto-invalidation trigger here. The fingerprint is
            # per-(model, launch shape) -- it includes parser names and
            # per-model recovery flags -- so the prober cannot compute
            # "what today's launch would hash to" for a row without
            # building that row's full argv, and getting it slightly wrong
            # turns a routine `make probe-sglang` into an unrequested
            # fleet-wide re-probe. Drift is REPORTED instead, by
            # `make probe-check`, where a false positive costs a line of
            # output rather than a GPU day. Re-probe with PROBE_FORCE=1.
            has_fitting = any(
                isinstance(c, dict) and c.get("fits") for c in band.values()
            )
            if not has_fitting and band:
                print(f"  {name}: cached band holds no fitting cell — "
                      f"re-probing rather than treating failure as complete",
                      file=sys.stderr)
            if stale_image and band:
                print(f"  {name}: cached cells were measured on engine image "
                      f"{(entry_image or '')[:19]} != {run_image_digest[:19]} "
                      f"— re-probing", file=sys.stderr)
            force_this_row = args.force or ctx_exact is not None
            if (band and has_fitting and not stale_schema and not stale_image
                    and not (force_this_row and not args.no_cache_write)):
                fully_cached += 1
                continue
            # Exact mode keeps the previous band on a miss (see below).
            prior_band = dict(band) if ctx_exact is not None else None
            # ... and the provenance that describes it, restored with it.
            prior_provenance = ({k: entry.get(k, _ABSENT) for k in PARSER_PROVENANCE_KEYS}
                                if ctx_exact is not None else None)

            def _restore_prior() -> None:
                band.update(prior_band or {})
                for k, v in (prior_provenance or {}).items():
                    if v is _ABSENT:
                        entry.pop(k, None)
                    else:
                        entry[k] = v

            band.clear()
            _stamp_parser_provenance()

            def _src(kind: str) -> str:
                src = parser_sources.get(kind)
                if src == "heuristic":
                    return f"(heuristic: {HEURISTIC_LABEL})"
                return f"({src})" if src else ""

            parser_label = (
                f"R={row_reasoning_parser or '—'}{_src('reasoning')} "
                f"T={row_tool_parser or '—'}{_src('tool')}"
            )
            print(f"  {name} @ {vram_label(vram_gb)}: binary-searching max "
                  f"fitting ctx over "
                  f"{context_label(search_grid[0])}-"
                  f"{context_label(search_grid[-1])} [{parser_label}] ...",
                  file=sys.stderr)

            row_mtp = row.get("mtp") if isinstance(row.get("mtp"), dict) else None
            mtp_should_probe = (
                row_mtp is not None
                and engine_of(spec.name) == "vllm"
                and not getattr(args, "no_mtp", False)
            )

            # Binary-search the 32K grid for the largest ctx that FITS
            # (loads on the GPU). probe_one_cell is the per-tier predicate;
            # every launched cell is stashed so we keep exactly the winning
            # (largest fitting) one and discard the rest. Tiers above the
            # model's position_limit are instant-failed with no launch.
            probed_cells: dict[int, dict] = {}

            def _probe_at(ctx: int) -> dict:
                return probe_one_cell(
                    spec,
                    runtime=args.runtime,
                    image=args.image,
                    container_name=args.container_name,
                    probe_port=args.probe_port,
                    models_dir=str(models_dir),
                    model_name=name,
                    requested_ctx=ctx,
                    band_gb=vram_gb,
                    host_vram_gb=args.host_vram_gb,
                    model_size_gb=size_gb,
                    prompt=args.prompt,
                    reasoning_parser=row_reasoning_parser,
                    tool_parser=row_tool_parser,
                    tool_parser_auto_only=(
                        parser_sources.get("tool") in GUESSED_PARSER_SOURCES),
                )

            def _fit_works(ctx: int) -> bool:
                nonlocal row_reasoning_parser, row_tool_parser, first_launch_pending
                rec = _probe_at(ctx)
                if first_launch_pending:
                    first_launch_pending = False
                    if guessed_kinds and not rec.get("fits"):
                        # The row's FIRST launch failed -- at startup, in
                        # Probe A, anywhere -- while it carried a GUESSED
                        # parser (recipe seed or name heuristic). Retry the
                        # same context once without the guesses. Only if
                        # that run fits is the guess blamed: it is dropped
                        # for the rest of the run, recorded in
                        # `parser_fallback`, and skipped by later runs. If
                        # the run without it fails too, the failure is not
                        # the guess's (an OOM, a recovery flag the engine
                        # rejects, ...): the guesses stay and nothing is
                        # recorded. At most one extra launch per row.
                        failed_kind = (rec.get("evidence") or {}).get("kind", "?")
                        saved = (row_reasoning_parser, row_tool_parser)
                        dropped = {k: (row_tool_parser if k == "tool"
                                       else row_reasoning_parser)
                                   for k in guessed_kinds}
                        if "tool" in dropped:
                            row_tool_parser = None
                        if "reasoning" in dropped:
                            row_reasoning_parser = None
                        print(f"    {context_label(ctx):>4s} the first launch with "
                              f"guessed parser(s) {dropped} failed "
                              f"(kind={failed_kind}); retrying without them",
                              file=sys.stderr)
                        retry = _probe_at(ctx)
                        if retry.get("fits"):
                            for kind, value in dropped.items():
                                parser_fallback[kind] = {
                                    "parser": value,
                                    "source": parser_sources.get(kind),
                                    "failure": failed_kind,
                                    "note": "the first launch with it failed and "
                                            "the same context fit without it; "
                                            "later runs skip it",
                                }
                                parser_sources[kind] = None
                            guessed_kinds.clear()
                            _stamp_parser_provenance()
                            rec = retry
                        else:
                            row_reasoning_parser, row_tool_parser = saved
                            print(f"    {context_label(ctx):>4s} failed without them "
                                  f"too (kind={(retry.get('evidence') or {}).get('kind', '?')}); "
                                  f"keeping the guesses", file=sys.stderr)
                probed_cells[ctx] = rec
                if rec.get("fits"):
                    rp = rec.get("reasoning_parser") or "—"
                    tp = rec.get("tool_parser") or "—"
                    dv = "y" if rec.get("disable_verified") else "n"
                    print(f"    {context_label(ctx):>4s} fits  "
                          f"vram={rec.get('actual_vram_gb', '?')} "
                          f"cap={rec.get('capability', '?')} R={rp} T={tp} dis={dv} "
                          f"({rec.get('startup_seconds', 0):.1f}s start)",
                          file=sys.stderr)
                else:
                    ev = rec.get("evidence") or {}
                    print(f"    {context_label(ctx):>4s} FAIL  "
                          f"kind={ev.get('kind', '?')}", file=sys.stderr)
                return bool(rec.get("fits"))

            pos_cap = pos_limit if isinstance(pos_limit, int) and pos_limit > 0 else None
            if ctx_exact is not None:
                try:
                    exact_ctx_grid(ctx_exact, pos_cap)
                except ValueError as e:
                    _restore_prior()
                    sys.exit(f"error: {name}: {e}")
                prior_max = max((int(c) for c, cell in (prior_band or {}).items()
                                 if isinstance(cell, dict) and cell.get("fits")),
                                default=0)
                if prior_max > ctx_exact:
                    print(f"    note: the cached cell is {context_label(prior_max)}; "
                          f"a fit at {context_label(ctx_exact)} will REPLACE it "
                          f"with the smaller value", file=sys.stderr)
            max_ctx = binary_search_max_ctx(
                _fit_works, position_limit=pos_cap, grid=search_grid,
            )

            # An architecture/quant rejection at any probed tier means the
            # model cannot load at all -- keep it as evidence and stop the
            # vram loop (drives the unsupported_arch ledger entry below).
            arch_rec = next(
                (r for r in probed_cells.values() if _is_arch_kind(r)), None
            )

            if arch_rec is not None:
                band[str(arch_rec["ctx"])] = arch_rec
                entry.setdefault("first_probed_at", arch_rec["probed_at"])
                entry["last_probed_at"] = arch_rec["probed_at"]
                stamp_row_image(entry, run_image_digest)
                print(f"    [stop] {name}: "
                      f"{(arch_rec.get('evidence') or {}).get('kind')} — "
                      f"unsupported arch", file=sys.stderr)
                arch_or_quant_seen = True
            elif max_ctx is not None:
                winner = probed_cells[max_ctx]  # may be reset to None below (exact-mode MTP miss)
                # Second pass: MTP overhead measurement on the WINNING cell
                # only (not every probed tier). Only when the catalog
                # declared an MTP block AND we're on vLLM AND not --no-mtp.
                if mtp_should_probe and winner.get("fits"):
                    mtp_method = (row_mtp or {}).get("method")
                    mtp_drafter = (row_mtp or {}).get("drafter")
                    mtp_k = (row_mtp or {}).get("num_speculative_tokens")
                    print(f"    {context_label(max_ctx):>4s}   ... re-probing with "
                          f"MTP ({mtp_method}, K={mtp_k}, drafter={mtp_drafter or 'built-in'})",
                          file=sys.stderr)
                    mtp_rec = probe_one_cell(
                        spec,
                        runtime=args.runtime,
                        image=args.image,
                        # Separate container name so the second pass can't
                        # collide with a half-torn-down baseline container.
                        container_name=f"{args.container_name}-mtp",
                        probe_port=args.probe_port,
                        models_dir=str(models_dir),
                        model_name=name,
                        requested_ctx=max_ctx,
                        band_gb=vram_gb,
                        host_vram_gb=args.host_vram_gb,
                        model_size_gb=size_gb,
                        prompt=args.prompt,
                        reasoning_parser=row_reasoning_parser,
                        tool_parser=row_tool_parser,
                        tool_parser_auto_only=(
                            parser_sources.get("tool") in GUESSED_PARSER_SOURCES),
                        mtp_method=mtp_method,
                        mtp_drafter=mtp_drafter,
                        mtp_num_tokens=mtp_k,
                    )
                    winner["mtp_method"] = mtp_method
                    if mtp_rec.get("fits"):
                        base_vram = float(winner.get("actual_vram_gb", 0) or 0)
                        mtp_vram = float(mtp_rec.get("actual_vram_gb", 0) or 0)
                        winner["mtp_fits"] = True
                        winner["mtp_overhead_gb"] = round(max(0.0, mtp_vram - base_vram), 2)
                        winner["mtp_actual_vram_gb"] = round(mtp_vram, 2)
                    else:
                        winner["mtp_fits"] = False
                        winner["mtp_evidence"] = mtp_rec.get("evidence", {})
                if (ctx_exact is not None and mtp_should_probe
                        and not winner.get("mtp_fits")):
                    # Exact mode on a row that declares MTP: a hit means the
                    # DECLARED configuration works at this ctx. Recording a
                    # fits=True / mtp_fits=False cell here would make the
                    # router refuse `::mtp` for a row whose previous, smaller
                    # cell served it (measured 2026-09-22: 120K fit, MTP did
                    # not, and the 115K MTP cell was lost). Keep the previous
                    # cell and report the MTP pass's own estimate.
                    _restore_prior()
                    est = engine_max_len_estimate(winner.get("mtp_evidence"))
                    hint = (f"; with MTP the engine estimates {est} tokens "
                            f"({context_label(est)}) would fit -- retry with "
                            f"--ctx-exact {est}" if est else "")
                    print(f"    [miss] {name}: {context_label(ctx_exact)} fits "
                          f"without MTP but the MTP pass does not{hint}",
                          file=sys.stderr)
                    exact_misses += 1
                    winner = None
                if winner is not None:
                    band[str(max_ctx)] = winner
                    entry.setdefault("first_probed_at", winner["probed_at"])
                    entry["last_probed_at"] = winner["probed_at"]
                    stamp_row_image(entry, run_image_digest)
                    if first_seen_record is None:
                        first_seen_record = winner
                    fresh_probes += 1
            elif probed_cells and ctx_exact is not None:
                # An exact probe missed. That is not "fits nowhere": the
                # operator's number was a little high. Put the previous cell
                # back untouched, say what the engine itself thinks would
                # fit, and let the run end non-zero so nobody mistakes it
                # for a recorded result.
                miss = probed_cells[ctx_exact]
                _restore_prior()
                est = engine_max_len_estimate(miss.get("evidence"))
                kind = (miss.get("evidence") or {}).get("kind", "?")
                hint = (f"; the engine estimates {est} tokens "
                        f"({context_label(est)}) would fit -- retry with "
                        f"--ctx-exact {est}" if est else
                        f" (kind={kind}; no engine estimate in the evidence)")
                print(f"    [miss] {name}: {context_label(ctx_exact)} does not "
                      f"fit{hint}", file=sys.stderr)
                exact_misses += 1
            elif probed_cells:
                # Fits nowhere (OOM at every probed tier). Keep the smallest
                # probed cell as the failing evidence so _entry_oom_everywhere
                # records the `oom` ledger exclusion below.
                lo = min(probed_cells)
                band[str(lo)] = probed_cells[lo]
                entry.setdefault("first_probed_at", probed_cells[lo]["probed_at"])
                entry["last_probed_at"] = probed_cells[lo]["probed_at"]
                stamp_row_image(entry, run_image_digest)
                fresh_probes += 1
            else:
                # Nothing was ever launched: every grid tier sits above the
                # model's position_limit, so binary_search_max_ctx
                # instant-failed them all. Previously this wrote NO cell, NO
                # ledger entry and printed NOTHING, so the model was silently
                # re-attempted on every future run. Record the reason and
                # exclude it -- a context ceiling below the smallest probed
                # tier is a property of the checkpoint, not of this GPU, so
                # the terminal (vram- and sha-independent) verdict is right.
                detail = (f"position_limit={pos_cap} is below the smallest "
                          f"probed ctx tier {context_label(search_grid[0])}")
                print(f"    [skip] {name}: {detail}", file=sys.stderr)
                _ledger_record(ledger, name, spec.name, "unsupported_arch",
                               detail=detail, repo=repo,
                               host_vram=args.host_vram_gb, sha=sha)
                skipped_pos_limit += 1

            if probed_cells and entry.get("schema_version", 1) < spec.schema_version:
                # Only now is the version bump earned: fresh cells were just
                # measured by this (v2) probe driver, so the entry carries the
                # probed parser / disable verdicts the router's v1 refusal
                # exists to demand. ensure_entry deliberately does NOT bump.
                entry["schema_version"] = spec.schema_version

            refresh_top_level_from_cells(entry)
            if not args.no_cache_write:
                save_cache(args.cache, cache)

        # Top-level capability is re-derived from the full cell corpus
        # via refresh_top_level_from_cells after each cell write. Final
        # save here just persists the last state; terminal states like
        # `unsupported_arch` are sticky inside that helper.
        refresh_top_level_from_cells(entry)
        # Reconcile the host ledger with the fresh probe result:
        #  - terminal arch -> mirror unsupported_arch (sha-stable, gates
        #    download + future probes).
        #  - now fits (e.g. a --force-arch re-probe recovered the model, or a
        #    recovery flag was added) -> clear a stale prober-owned exclusion
        #    so the ledger NEVER hides a model the cache says fits. (Leave a
        #    `manual` operator pin alone.)
        if entry.get("capability") == Capability.UNSUPPORTED_ARCH:
            _ledger_record(ledger, name, spec.name, "unsupported_arch",
                           detail=(entry.get("evidence") or {}).get(
                               "matched_pattern") or "arch load failure",
                           repo=repo, host_vram=args.host_vram_gb, sha=sha)
        elif _entry_fits_anywhere(entry):
            # Recovered/fitting -> clear a stale prober-owned exclusion.
            if _ledger_reason(ledger, name, spec.name) in (
                    "unsupported_arch", "oom"):
                _ledger_clear(ledger, name, spec.name)
        elif _entry_oom_everywhere(entry):
            # Fits at no tier and the failure is OOM -> record `oom`
            # (re-checked on a new sha; decision 2). Stops re-probing a model
            # that does not fit this GPU.
            _ledger_record(ledger, name, spec.name, "oom",
                           detail="OOM at every probed ctx; does not fit GPU",
                           repo=repo, host_vram=args.host_vram_gb, sha=sha)
        if not args.no_cache_write:
            save_cache(args.cache, cache)

    # Drop sha-orphaned entries left by re-quant/commit (terminal verdicts
    # have already been carried forward onto the current-sha entries).
    pruned = prune_orphaned_shas(cache, catalog_rows)
    if pruned and not args.no_cache_write:
        save_cache(args.cache, cache)
    if not args.no_cache_write and not ignore_curated:
        _save_ledger(ledger, host_vram_gb=args.host_vram_gb)

    print(file=sys.stderr)
    by_cap: dict[str, int] = {}
    for k, entry in cache.items():
        if k.startswith("_") or not isinstance(entry, dict):
            continue
        c = entry.get("capability") or Capability.UNKNOWN
        by_cap[c] = by_cap.get(c, 0) + 1
    summary = "  ".join(f"{c}={n}" for c, n in sorted(by_cap.items()))
    print(
        f"  done: {fresh_probes} probe(s); "
        f"{fully_cached} band(s) fully cached; "
        f"{skipped_missing} not on disk; "
        f"{skipped_excluded} ledger-excluded; "
        f"{skipped_pos_limit} below the smallest ctx tier; "
        f"{skipped_arch} skipped as cached unsupported_arch; "
        f"{pruned} orphan sha(s) pruned",
        file=sys.stderr,
    )
    print(f"  capability counts: {summary}", file=sys.stderr)
    if exact_misses:
        sys.exit(f"error: --ctx-exact {context_label(ctx_exact)} missed on "
                 f"{exact_misses} row(s); nothing was recorded for them")
