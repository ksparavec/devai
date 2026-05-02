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
import json
import os
import re
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
    context_label,
    effective_targets,
    parse_context_list,
    parse_vram_list,
    standard_contexts,
    standard_vram_budgets,
    vram_label,
)
from _probe_core import (  # noqa: E402
    has_inline_think_markers,
    http_get,
    http_post,
    load_cache,
    now_iso,
    propagate_implied_fail,
    save_cache,
    smallest_clean_probe,
)
from _vllm_plugins import PluginEntry, PluginRegistry, get_registry  # noqa: E402


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_CATALOG = REPO_ROOT / "deploy" / "models.yaml"
DEFAULT_MODELS_DIR = os.environ.get(
    "VLLM_MODELS_DIR", "/var/cache/devai/ollama/models/vllm"
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
# with deploy/docker-compose.yaml service names.
MUTEX_CONTAINERS = ("devai-router", "devai-vllm", "devai-sglang")


# ── Recovery flags registry ──────────────────────────────────────────────────
# Per-model launch flags + env overrides shared with the router. Same file
# the router reads via gpu-arbiter/recovery_flags.go. Loaded once at module
# import; lookup is a single dict access per probe.
RECOVERY_FLAGS_PATH = Path(
    os.environ.get("RECOVERY_FLAGS_REGISTRY", str(REPO_ROOT / "deploy" / "recovery-flags.json"))
)


def _load_recovery_registry(path: Path) -> dict[str, dict]:
    """Read deploy/recovery-flags.json and return {model_name: entry}.

    Missing file or parse error → empty dict (every model launches without
    extra flags, matching pre-registry behaviour).
    """
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] recovery registry {path}: {exc}", file=sys.stderr)
        return {}
    models = data.get("models")
    if not isinstance(models, dict):
        return {}
    return {name: entry for name, entry in models.items() if isinstance(entry, dict)}


_RECOVERY_REGISTRY: dict[str, dict] = _load_recovery_registry(RECOVERY_FLAGS_PATH)


def recovery_overrides(model_name: str) -> tuple[list[str], dict[str, str]]:
    """Return (extra_flags, extra_env) for model_name. Empty when no entry."""
    entry = _RECOVERY_REGISTRY.get(model_name)
    if not entry:
        return [], {}
    flags = entry.get("engine_flags") or []
    env = entry.get("engine_env") or {}
    if not isinstance(flags, list):
        flags = []
    if not isinstance(env, dict):
        env = {}
    return list(flags), dict(env)


# ── Backend spec ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BackendSpec:
    """Per-backend constants and the launch-arg builder."""
    name: str                   # "vllm" | "sglang"
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
    build_args: Callable[..., list[str]] = field(repr=False)
    # When True, parser names are looked up against the vllm-plugins
    # registry; matches inject the bind-mount + --tool-parser-plugin
    # flag. Only vLLM supports this today — SGLang's plugin model uses
    # Python registry imports rather than a file-path arg.
    supports_plugins: bool = False
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
        if m.get("source") != "hf":
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


def container_remove(runtime: str, name: str) -> None:
    subprocess.run(
        [runtime, "rm", "--force", name],
        capture_output=True, check=False,
    )


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
    for k, v in env_vars.items():
        args.extend(["--env", f"{k}={v}"])
    args.append(image)
    args.extend(command)
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


def build_disable_thinking_body(backend: str, base: dict) -> dict:
    """Mutate-in-place a chat body to request reasoning suppression.

    vLLM (Qwen3-style template):
        extra_body.chat_template_kwargs.enable_thinking = False
    SGLang:
        separate_reasoning = False (top-level field; SGLang's OpenAI-
        compatible layer reads it from the request body)

    Both fields are no-ops when the model has no template-level
    thinking switch — the disable verdict is then `False` and the
    router won't emit a disable directive at serve time.
    """
    body = dict(base)
    if backend == "vllm":
        extra = dict(body.get("extra_body") or {})
        ctk = dict(extra.get("chat_template_kwargs") or {})
        ctk["enable_thinking"] = False
        extra["chat_template_kwargs"] = ctk
        body["extra_body"] = extra
    elif backend == "sglang":
        body["separate_reasoning"] = False
        # SGLang also accepts the chat_template_kwargs path on the same
        # template families it shares with vLLM (Qwen3). Set both so we
        # disable cleanly regardless of which path the runtime honours.
        extra = dict(body.get("extra_body") or {})
        ctk = dict(extra.get("chat_template_kwargs") or {})
        ctk["enable_thinking"] = False
        extra["chat_template_kwargs"] = ctk
        body["extra_body"] = extra
    return body


def build_enable_thinking_body(backend: str, base: dict) -> dict:
    """Symmetric to build_disable_thinking_body — flips the same fields
    to enable structured reasoning output. Used by Probe A so models
    with chat templates that default `enable_thinking=false` (newer
    Qwen3, etc.) emit a reasoning trace under the curated parser.
    """
    body = dict(base)
    if backend == "vllm":
        extra = dict(body.get("extra_body") or {})
        ctk = dict(extra.get("chat_template_kwargs") or {})
        ctk["enable_thinking"] = True
        extra["chat_template_kwargs"] = ctk
        body["extra_body"] = extra
    elif backend == "sglang":
        body["separate_reasoning"] = True
        extra = dict(body.get("extra_body") or {})
        ctk = dict(extra.get("chat_template_kwargs") or {})
        ctk["enable_thinking"] = True
        extra["chat_template_kwargs"] = ctk
        body["extra_body"] = extra
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
        return "error", {"error": resp["error"]}
    choices = resp.get("choices") or []
    if not choices:
        return "unsupported", {"reason": "no choices in response"}
    msg = (choices[0].get("message") or {})
    reasoning = _extract_reasoning_content(msg)
    content = msg.get("content") or ""
    finish_reason = (choices[0].get("finish_reason") or "")

    if reasoning:
        return "structured", {
            "reasoning_preview": reasoning[:200],
            "content_preview": content[:120],
            "structured_reason": "reasoning_content_populated",
        }
    if reasoning_parser_attempted and finish_reason == "length" and not content:
        # Model thought past max_tokens; parser buffered inside <think>
        # waiting for </think>. The parser was active — the budget was
        # too low. Classify as structured; the disable probe will
        # confirm wiring.
        return "structured", {
            "structured_reason": "thought_past_max_tokens",
            "finish_reason": finish_reason,
        }
    if reasoning_parser_attempted and content.startswith("\n\n"):
        # Parser likely stripped an empty `<think></think>` block,
        # leaving the leading newlines. Distinct from a model that
        # genuinely starts with newlines (rare with our prompt).
        return "structured", {
            "structured_reason": "empty_think_stripped",
            "content_preview": content[:120],
        }
    if has_inline_think_markers(content):
        return "inline", {"content_preview": content[:200]}
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
        return "none", {
            "none_reason": "clean_answer_no_parser_attempted",
            "content_preview": content[:120],
        }
    return "unsupported", {"content_preview": content[:120]}


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
    "GPTQ",
    "AWQ kernel",
)
_OOM_PATTERNS = (
    "CUDA out of memory",
    "torch.cuda.OutOfMemoryError",
    "RuntimeError: Allocator",
    "free; ",
    "no space left",
)


def classify_failure_logs(logs: str) -> dict:
    """Inspect container logs after a failed launch and tag the cause.

    Returns {kind, log_excerpt, matched_pattern?}. kind ∈ {arch, quant,
    oom_startup, infra}. The matched pattern is recorded when the cause
    is determined by a keyword match (everything except `infra`) so an
    operator can audit why a particular run was tagged the way it was —
    crucial because pattern matching runs against the full log but only
    the last 30 lines land in the excerpt.
    """
    excerpt = "\n".join(logs.strip().splitlines()[-30:])
    lc = logs.lower()
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
         `tool_parser=<curated-name>`; else null.
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
    cmd_args = spec.build_args(
        model_name, requested_ctx, host_frac,
        reasoning_parser=reasoning_parser, tool_parser=tool_parser,
        reasoning_parser_plugin=reasoning_plugin_path,
        tool_parser_plugin=tool_plugin_path,
    )

    env_vars = {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"}
    # Per-model recovery overrides (deploy/recovery-flags.json) — shared
    # with the router so probe and serve-time launches see the same flags.
    # Env entries override env_vars defaults on key collision; flags are
    # appended after parser flags (mirrors gpu-arbiter/main.go).
    extra_flags, extra_env = recovery_overrides(model_name)
    if extra_flags:
        cmd_args = list(cmd_args) + extra_flags
    if extra_env:
        env_vars = {**env_vars, **extra_env}
    extra_volumes: list[tuple[str, str, str]] = []
    if plugin_volume is not None:
        extra_volumes.append(plugin_volume)
    container_remove(runtime, container_name)
    try:
        container_run_detached(
            runtime, container_name, image, probe_port, models_dir, env_vars,
            spec.entrypoint, cmd_args, extra_volumes=extra_volumes,
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
        container_remove(runtime, container_name)
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

    if capability == "error":
        container_remove(runtime, container_name)
        return _failure_record(
            ctx=requested_ctx, vram_gb=band_gb, started=started,
            startup_seconds=startup_seconds, actual_vram_gb=actual_vram_gb,
            actual_context=actual_max,
            evidence={"kind": "oom_chat", **cap_evidence},
        )

    # Confirmed reasoning parser only when Probe A actually produced
    # `structured`. Any other capability means the curated parser
    # didn't apply (model emits inline or doesn't reason at all).
    reasoning_parser_verified = reasoning_parser if capability == "structured" else None

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
    tool_parser_verified: str | None = None
    tool_evidence: dict | None = None
    if tool_parser:
        # Reasoning models burn 500-2000 tokens on chain-of-thought before
        # any tool-call markers. The old 128-token budget guaranteed they
        # never reached the call. 2048 covers typical R1-Distill traces.
        is_reasoning = capability in ("structured", "inline")
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
            tool_parser_verified = tool_parser
            tool_evidence = {"verified": True, "mode": "auto"}
        else:
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
            if response_has_valid_tool_call(forced_resp, "get_time"):
                tool_parser_verified = tool_parser
                tool_evidence = {
                    "verified": True,
                    "mode": "forced",
                    "auto_response_preview": _short(auto_resp),
                }
            else:
                tool_evidence = {
                    "verified": False,
                    "auto_response_preview": _short(auto_resp),
                    "forced_response_preview": _short(forced_resp),
                }

    # ── Probe C: disable verification ────────────────────────────────
    disable_verified = False
    disable_evidence: dict | None = None
    if capability == "structured":
        disable_body = build_disable_thinking_body(spec.name, base_chat_body)
        disable_resp = _post_chat(base_url, disable_body)
        if not response_has_reasoning_content(disable_resp) and "error" not in disable_resp:
            disable_verified = True
            disable_evidence = {"verified": True}
        else:
            disable_evidence = {
                "verified": False,
                "response_preview": _short(disable_resp),
            }

    container_remove(runtime, container_name)

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
    if tool_evidence is not None:
        rec["evidence"]["tool"] = tool_evidence
    if disable_evidence is not None:
        rec["evidence"]["disable"] = disable_evidence
    return rec


def _post_chat(base_url: str, body: dict) -> dict:
    """POST a chat-completion body, normalize errors to `{error: ...}`."""
    try:
        return http_post(
            f"{base_url}/v1/chat/completions", body, timeout=CHAT_TIMEOUT,
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
        "capability": "error",
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
            "capability": "unknown",
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
        if entry.get("schema_version", 1) < schema_version:
            entry["schema_version"] = schema_version
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
    """
    cur_cap = entry.get("capability") or "unknown"
    cur_kind = ((entry.get("evidence") or {}).get("kind") or "")
    is_terminal = (
        cur_cap == "unsupported_arch"
        or (cur_cap == "error" and cur_kind in ("arch", "quant"))
    )
    if is_terminal:
        return

    smallest = smallest_clean_probe(entry)
    if smallest is not None:
        entry["capability"] = smallest.get("capability") or "unknown"
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
                if cell.get("capability") in (None, "error"):
                    continue
                ac = int(cell.get("actual_context") or 0)
                if ac > largest_ctx:
                    largest_ctx = ac
        if largest_ctx and (entry.get("max_context") or 0) < largest_ctx:
            entry["max_context"] = largest_ctx
        return

    # No clean probe in the corpus — pick the most-severe failure kind
    # to set capability.
    severities = {"arch": 3, "quant": 2}  # other kinds = severity 1
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
        entry["capability"] = "unsupported_arch"
        entry["evidence"] = chosen_ev
    elif chosen_severity >= 1:
        entry["capability"] = "error"
        entry["evidence"] = chosen_ev
    else:
        entry.setdefault("capability", "unknown")


# ── Argparse builder ─────────────────────────────────────────────────────────

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
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="probe prompt (kept short and deterministic)")
    ap.add_argument("--force", action="store_true",
                    help="re-probe every requested cell even if cached")
    ap.add_argument("--force-arch", action="store_true",
                    help="re-probe top-level arch/capability fields")
    ap.add_argument("--no-cache-write", action="store_true",
                    help="dry run — do not modify the cache")
    return ap


# ── Main loop ────────────────────────────────────────────────────────────────

def run_probe_pass(spec: BackendSpec, args: argparse.Namespace) -> None:
    """Run one probe pass with the given backend spec and argparse args."""
    assert_no_active_backends(args.runtime)

    catalog_rows = load_catalog_hf_rows(args.catalog, spec.name)
    if args.repo:
        rx = re.compile(args.repo)
        catalog_rows = [r for r in catalog_rows if rx.search(r.get("repo") or "")]
    if not catalog_rows:
        sys.exit("error: no HF rows match — check --repo filter or models.yaml")

    models_dir = Path(args.models_dir)
    if not models_dir.is_dir():
        sys.exit(f"error: models dir not found: {models_dir}")

    # Default to ONLY the host VRAM. Probing 16G when the host has 24G
    # produces data nobody on this host can use; the operator can still
    # ask for sub-host bands explicitly via `--vram 16,24`.
    vrams = parse_vram_list(args.vram) if args.vram else [int(args.host_vram_gb)]
    ctxs = parse_context_list(args.ctx) if args.ctx else standard_contexts()
    if not vrams or not ctxs:
        sys.exit("error: --vram or --ctx produced an empty list")

    cache = load_cache(args.cache)

    print(f"  prober:         probe-{spec.name}-reasoning", file=sys.stderr)
    print(f"  cache:          {args.cache} ({len(cache)} entries)", file=sys.stderr)
    print(f"  catalog rows:   {len(catalog_rows)} (after --repo filter)",
          file=sys.stderr)
    print(f"  vram bands:     {','.join(vram_label(v) for v in vrams)}",
          file=sys.stderr)
    print(f"  ctx tiers:      {','.join(context_label(c) for c in ctxs)}",
          file=sys.stderr)
    print(f"  host vram:      {args.host_vram_gb}G", file=sys.stderr)
    print(f"  {spec.name} image:     {args.image}", file=sys.stderr)
    print(file=sys.stderr)

    fresh_probes = 0
    fully_cached = 0
    skipped_missing = 0
    skipped_arch = 0

    for row in catalog_rows:
        name = row.get("name") or ""
        repo = row.get("repo") or ""
        sha = (row.get("sha") or "").strip()
        if not (name and repo and sha):
            print(f"  [skip] catalog row missing fields: name={name!r} "
                  f"repo={repo!r} sha={sha!r}", file=sys.stderr)
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

        # Pull curated parser hints for this backend from the catalog
        # row. The probe driver passes these to spec.build_args; the
        # cell record only confirms them when the backend round-trip
        # produces the expected response shape.
        row_parsers = (row.get("parsers") or {}).get(spec.name) or {}
        row_reasoning_parser = row_parsers.get("reasoning") or None
        row_tool_parser = row_parsers.get("tool") or None

        if args.force_arch:
            entry["capability"] = "unknown"
            entry["evidence"] = {}

        if (entry.get("capability") == "unsupported_arch"
                and not args.force_arch):
            print(f"  [skip] {name}: unsupported_arch (cached)",
                  file=sys.stderr)
            skipped_arch += 1
            continue

        first_seen_record: dict | None = None
        arch_or_quant_seen = False

        for vram_gb in vrams:
            if arch_or_quant_seen:
                break
            band = entry.setdefault("probes", {}).setdefault(str(vram_gb), {})
            targets = effective_targets(ctxs, entry.get("max_context") or 0)
            if not targets:
                targets = sorted(set(ctxs))

            missing = [
                t for t in targets
                if str(t) not in band or (args.force and not args.no_cache_write)
            ]
            if not missing:
                fully_cached += 1
                continue

            parser_label = (
                f"R={row_reasoning_parser or '—'} "
                f"T={row_tool_parser or '—'}"
            )
            print(f"  {name} @ {vram_label(vram_gb)}: probing "
                  f"{','.join(context_label(c) for c in missing)} "
                  f"[{parser_label}] ...",
                  file=sys.stderr)

            for ctx in missing:
                rec = probe_one_cell(
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
                )
                band[str(ctx)] = rec
                entry.setdefault("first_probed_at", rec["probed_at"])
                entry["last_probed_at"] = rec["probed_at"]
                if first_seen_record is None:
                    first_seen_record = rec
                # Re-derive top-level capability + parser fields from
                # the full corpus of recorded cells (not just this one).
                # Idempotent — invariant: a successful classification at
                # any (vram, ctx) tier is preserved even if a different
                # tier later fails on oom/infra/clamped_ctx.
                refresh_top_level_from_cells(entry)
                fresh_probes += 1
                if not args.no_cache_write:
                    save_cache(args.cache, cache)

                cap_marker = rec.get("capability", "?")
                if rec.get("fits"):
                    rp = rec.get("reasoning_parser") or "—"
                    tp = rec.get("tool_parser") or "—"
                    dv = "y" if rec.get("disable_verified") else "n"
                    print(f"    {context_label(ctx):>4s} fits  "
                          f"vram={rec.get('actual_vram_gb', '?')} "
                          f"cap={cap_marker} R={rp} T={tp} dis={dv} "
                          f"({rec.get('startup_seconds', 0):.1f}s start)",
                          file=sys.stderr)
                else:
                    ev = rec.get("evidence") or {}
                    print(f"    {context_label(ctx):>4s} FAIL  "
                          f"kind={ev.get('kind', '?')}",
                          file=sys.stderr)

                if _is_arch_kind(rec):
                    print(f"    [stop] {name}: "
                          f"{(rec.get('evidence') or {}).get('kind')} — "
                          f"skipping remaining cells", file=sys.stderr)
                    arch_or_quant_seen = True
                    if not args.no_cache_write:
                        save_cache(args.cache, cache)
                    break

                if _is_oom_kind(rec):
                    def build_implied(
                        larger: int,
                        _ctx=ctx,
                        _vram=vram_gb,
                        _at=rec["probed_at"],
                        _ev_kind=(rec.get("evidence") or {}).get("kind"),
                    ) -> dict:
                        return {
                            "ctx": larger,
                            "vram_gb": _vram,
                            "fits": False,
                            "actual_context": larger,
                            "capability": "error",
                            "evidence": {
                                "kind": "implied_spill",
                                "implied_from_ctx": _ctx,
                                "source_kind": _ev_kind,
                                "error": (
                                    f"implied fail: {context_label(_ctx)} "
                                    f"at {vram_label(_vram)} already "
                                    f"{_ev_kind}"
                                ),
                            },
                            "probed_at": _at,
                            "probe_seconds": 0.0,
                            "implied": True,
                        }

                    new_implied = propagate_implied_fail(
                        vram_band=band,
                        targets=targets,
                        failed_ctx=ctx,
                        force_set=set() if not args.force else set(targets),
                        build_implied_record=build_implied,
                    )
                    if new_implied:
                        entry["last_probed_at"] = rec["probed_at"]
                    if not args.no_cache_write:
                        save_cache(args.cache, cache)
                    break

        # Top-level capability is re-derived from the full cell corpus
        # via refresh_top_level_from_cells after each cell write. Final
        # save here just persists the last state; terminal states like
        # `unsupported_arch` are sticky inside that helper.
        refresh_top_level_from_cells(entry)
        if not args.no_cache_write:
            save_cache(args.cache, cache)

    print(file=sys.stderr)
    by_cap: dict[str, int] = {}
    for entry in cache.values():
        c = entry.get("capability") or "unknown"
        by_cap[c] = by_cap.get(c, 0) + 1
    summary = "  ".join(f"{c}={n}" for c, n in sorted(by_cap.items()))
    print(
        f"  done: {fresh_probes} probe(s); "
        f"{fully_cached} band(s) fully cached; "
        f"{skipped_missing} not on disk; "
        f"{skipped_arch} skipped as cached unsupported_arch",
        file=sys.stderr,
    )
    print(f"  capability counts: {summary}", file=sys.stderr)
