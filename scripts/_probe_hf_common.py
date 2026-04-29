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
)


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_CATALOG = REPO_ROOT / "deploy" / "models.yaml"
DEFAULT_MODELS_DIR = os.environ.get(
    "VLLM_MODELS_DIR", "/var/cache/devai/ollama/models/vllm"
)
DEFAULT_PROMPT = "Answer with only the final number: What is 17 + 25?"
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
    # build_args(model_name, max_ctx, host_frac) → CMD list (excluding entrypoint).
    build_args: Callable[[str, int, float], list[str]] = field(repr=False)
    schema_version: int = 1


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
) -> None:
    """Launch a probe container detached on localhost loopback.

    `--entrypoint` is mandatory: the upstream vllm and sglang images
    ship with their own ENTRYPOINTs that swallow our CMD args (the
    "vllm: error: unrecognized arguments" we hit on the first probe
    run). Replace it explicitly to match the router's libpod spec.
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

def classify_chat_response(resp: dict) -> tuple[str, dict]:
    """Map /v1/chat/completions response to (capability, evidence).

    v1: looks for inline `<think>` markers only. Structured-reasoning
    detection (vLLM's `reasoning_content` field, requires
    `--reasoning-parser` set at startup) is deferred until tool_parser
    detection lands.
    """
    if "error" in resp:
        return "error", {"error": resp["error"]}
    choices = resp.get("choices") or []
    if not choices:
        return "unsupported", {"reason": "no choices in response"}
    msg = (choices[0].get("message") or {})
    content = msg.get("content") or ""
    if has_inline_think_markers(content):
        return "inline", {"content_preview": content[:200]}
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
) -> dict:
    """Launch the backend, measure one (vram, ctx) cell, return a record.

    Always tears down the container before returning. The record's
    `fits` field is the single source of truth downstream.
    """
    started = time.time()
    host_frac = host_scaled_fraction(
        model_size_gb, band_gb, host_vram_gb, spec.reserve_gb,
    )
    cmd_args = spec.build_args(model_name, requested_ctx, host_frac)

    env_vars = {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"}
    container_remove(runtime, container_name)
    try:
        container_run_detached(
            runtime, container_name, image, probe_port, models_dir, env_vars,
            spec.entrypoint, cmd_args,
        )
    except RuntimeError as e:
        return _failure_record(
            ctx=requested_ctx, vram_gb=band_gb, started=started,
            evidence={"kind": "infra", "error": str(e)},
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
        container_remove(runtime, container_name)
        return _failure_record(
            ctx=requested_ctx, vram_gb=band_gb, started=started,
            evidence=evidence, startup_seconds=startup_seconds,
        )

    actual_max = 0
    try:
        models_resp = http_get(f"{base_url}/v1/models", timeout=10.0)
        for entry in (models_resp.get("data") or []):
            mm = entry.get("max_model_len")
            if isinstance(mm, int) and mm > actual_max:
                actual_max = mm
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        actual_max = 0

    if actual_max and actual_max < requested_ctx:
        container_remove(runtime, container_name)
        return _failure_record(
            ctx=requested_ctx, vram_gb=band_gb, started=started,
            startup_seconds=startup_seconds,
            evidence={
                "kind": "clamped_ctx",
                "actual_context": actual_max,
                "requested_context": requested_ctx,
            },
        )

    chat_body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 64,
        "stream": False,
    }
    chat_resp: dict
    try:
        chat_resp = http_post(
            f"{base_url}/v1/chat/completions", chat_body, timeout=CHAT_TIMEOUT,
        )
    except urllib.error.HTTPError as e:
        body = e.read() if hasattr(e, "read") else b""
        chat_resp = {
            "error": f"HTTP {e.code}: {e.reason}",
            "body": body[:200].decode(errors='replace'),
        }
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        chat_resp = {"error": f"{type(e).__name__}: {e}"}

    used_mb = gpu_memory_used_mb()
    actual_vram_gb = round(used_mb / 1024, 2)

    container_remove(runtime, container_name)

    capability, cap_evidence = classify_chat_response(chat_resp)
    if capability == "error":
        return _failure_record(
            ctx=requested_ctx, vram_gb=band_gb, started=started,
            startup_seconds=startup_seconds, actual_vram_gb=actual_vram_gb,
            actual_context=actual_max,
            evidence={"kind": "oom_chat", **cap_evidence},
        )

    return {
        "ctx": requested_ctx,
        "vram_gb": band_gb,
        "fits": True,
        "actual_vram_gb": actual_vram_gb,
        "actual_context": actual_max or requested_ctx,
        "capability": capability,
        "startup_seconds": startup_seconds,
        "probe_seconds": round(time.time() - started, 2),
        "probed_at": now_iso(),
        "evidence": cap_evidence,
    }


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
            "tool_parser": None,
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
    return entry


def reflect_first_cell_to_top_level(entry: dict, rec: dict) -> None:
    """Promote the first probe cell's outcome to entry's top level.

    Only fires when capability is unknown — preserves terminal states
    like `unsupported_arch` across re-probes.
    """
    if entry.get("capability") and entry["capability"] != "unknown":
        return
    if rec.get("fits"):
        entry["capability"] = rec.get("capability") or "unknown"
        actual_ctx = int(rec.get("actual_context") or 0)
        if actual_ctx and (entry.get("max_context") or 0) < actual_ctx:
            entry["max_context"] = actual_ctx
        entry["evidence"] = rec.get("evidence") or {}
        return
    ev = rec.get("evidence") or {}
    if ev.get("kind") == "arch":
        entry["capability"] = "unsupported_arch"
    else:
        entry["capability"] = "error"
    entry["evidence"] = ev


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

    vrams = parse_vram_list(args.vram) if args.vram else standard_vram_budgets()
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

            print(f"  {name} @ {vram_label(vram_gb)}: probing "
                  f"{','.join(context_label(c) for c in missing)} ...",
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
                )
                band[str(ctx)] = rec
                entry.setdefault("first_probed_at", rec["probed_at"])
                entry["last_probed_at"] = rec["probed_at"]
                if first_seen_record is None:
                    first_seen_record = rec
                    reflect_first_cell_to_top_level(entry, rec)
                fresh_probes += 1
                if not args.no_cache_write:
                    save_cache(args.cache, cache)

                cap_marker = rec.get("capability", "?")
                if rec.get("fits"):
                    print(f"    {context_label(ctx):>4s} fits  "
                          f"vram={rec.get('actual_vram_gb', '?')} "
                          f"cap={cap_marker} "
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

        # capability is set by reflect_first_cell_to_top_level on the first
        # observed cell and is NOT re-derived from probe records, so terminal
        # states like `unsupported_arch` survive cache-write boundaries.
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
