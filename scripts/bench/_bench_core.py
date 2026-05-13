"""Shared helpers for the bench harness.

Mirrors the layout of scripts/_probe_core.py — cache I/O, time, and
HTTP helpers — but adds streaming-HTTP support that the probe scaffold
deliberately doesn't carry. The probe runs `stream:false`; the bench
needs SSE so it can record time-to-first-token.

Imports stdlib-only so the harness inherits the same "runs anywhere"
constraint as the probers.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CACHE_PATH = REPO_ROOT / "deploy" / ".bench-cache.json"


# ── Streaming HTTP ──────────────────────────────────────────────────────────

def http_post_stream(
    url: str, body: dict, timeout: float = 600.0
) -> Iterator[tuple[float, str]]:
    """POST a streaming chat-completion request, yield ``(t_event, raw_line)``.

    Each yielded tuple is the wall-clock time the event arrived and a
    raw SSE ``data:`` payload (decoded text, no parsing). The caller
    decides whether to parse the JSON or treat the line as `[DONE]`.

    This keeps the streaming primitive minimal — one function, no
    class hierarchy. Callers that want JSON-parsed deltas can wrap
    this with their own ``json.loads`` + a typed accumulator.
    """
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        # urllib's response is a file-like; iterate line-by-line.
        # SSE frames: `data: <json>\n\n`. Skip blank lines; strip
        # the `data: ` prefix; surface whatever's left to the caller.
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            if line.startswith("data: "):
                payload = line[len("data: "):]
            elif line.startswith("data:"):
                payload = line[len("data:"):]
            else:
                continue
            yield (time.time(), payload)


def stream_chat_completion(
    base_url: str, body: dict, timeout: float = 600.0
) -> dict:
    """High-level streaming wrapper: open ``/v1/chat/completions`` with
    ``stream:true``, return aggregated metrics + concatenated content.

    Returns ``{"content": str, "completion_tokens": int, "t_open": float,
    "t_first_token": float|None, "t_done": float, "finish_reason": str|None}``.

    ``t_first_token`` is the timestamp of the first SSE event that
    carried a non-empty ``choices[0].delta.content``. ``None`` when the
    response was tool-calls-only (no text deltas). For TTFT
    measurement on plain text answers, ``t_first_token`` is the
    relevant event.
    """
    streamed_body = {**body, "stream": True}
    url = base_url.rstrip("/") + "/v1/chat/completions"
    t_open = time.time()
    t_first_token: float | None = None
    pieces: list[str] = []
    reasoning_pieces: list[str] = []
    completion_tokens = 0
    finish_reason: str | None = None
    t_done = t_open
    for t_event, payload in http_post_stream(url, streamed_body, timeout):
        if payload == "[DONE]":
            t_done = t_event
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        # TTFT counts ANY token from the model — content, reasoning,
        # or tool-call deltas. Reasoning models emit `<think>` blocks
        # first and the parser may put them in `delta.reasoning` or
        # `delta.reasoning_content` (vLLM/SGLang ≥0.19 vs older). For
        # the bench's purposes, that's still the first token the user
        # would see arriving over the wire — what matters is "model
        # started producing", not "model finished thinking".
        content_piece = delta.get("content")
        reasoning_piece = (
            delta.get("reasoning_content")
            or delta.get("reasoning")
        )
        if t_first_token is None and (content_piece or reasoning_piece):
            t_first_token = t_event
        if content_piece:
            pieces.append(content_piece)
        if reasoning_piece:
            reasoning_pieces.append(reasoning_piece)
        if choices[0].get("finish_reason"):
            finish_reason = choices[0]["finish_reason"]
        usage = obj.get("usage") or {}
        if "completion_tokens" in usage:
            completion_tokens = int(usage["completion_tokens"])
        t_done = t_event
    content = "".join(pieces)
    reasoning_content = "".join(reasoning_pieces)
    # Token-count reconciliation: vLLM with --reasoning-parser qwen3
    # populates usage.completion_tokens with ONLY content tokens —
    # reasoning_content tokens (which can dominate the stream for
    # thinking-heavy prompts) are excluded. The deepseek_r1 and
    # harmony parsers include reasoning. To get a parser-agnostic
    # decode-rate metric, fall back to a character-based estimate
    # (≈ 4 chars/token, the standard rough heuristic) over BOTH
    # content and reasoning_content streams. Take the max so that
    # accurate parsers aren't penalised by the heuristic's noise on
    # short outputs.
    char_based_tokens = (len(content) + len(reasoning_content)) // 4
    effective_tokens = max(completion_tokens, char_based_tokens)
    return {
        "content": content,
        "reasoning_content": reasoning_content,
        "completion_tokens": completion_tokens,
        "effective_tokens": effective_tokens,
        "t_open": t_open,
        "t_first_token": t_first_token,
        "t_done": t_done,
        "finish_reason": finish_reason,
    }


# ── Percentile helpers ──────────────────────────────────────────────────────

def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile. ``p`` in [0, 100]. Empty → 0.0."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_v) - 1)
    frac = k - lo
    return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * frac


def p50(values: list[float]) -> float:
    return percentile(values, 50)


def p95(values: list[float]) -> float:
    return percentile(values, 95)


# ── Leak regex compiler ─────────────────────────────────────────────────────

def load_leak_patterns(
    path: Path | None = None,
) -> list[tuple[str, "re.Pattern[str]"]]:
    """Read newline-separated regex patterns from ``data/leak_markers.txt``.

    Returns ``[(label, compiled_pattern), ...]``. Lines starting with
    ``#`` are comments. Each pattern is a Python regex (the file uses
    backslash-escaped pipes so ``<\\|im_end\\|>`` matches the literal
    token). The label is the raw line as the user wrote it — preserved
    so the cache reports human-readable marker names like
    ``"<|im_end|>"`` rather than the escaped form.
    """
    p = path or (DATA_DIR / "leak_markers.txt")
    out: list[tuple[str, re.Pattern[str]]] = []
    if not p.is_file():
        return out
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        label = line.replace("\\|", "|")
        try:
            out.append((label, re.compile(line)))
        except re.error:
            # Skip malformed regex but don't crash the whole sweep.
            continue
    return out


def sweep_for_leaks(
    text: str, patterns: list[tuple[str, "re.Pattern[str]"]]
) -> dict[str, int]:
    """Count hits per pattern in ``text``. Result is a flat dict keyed by
    label. Patterns that never match are still present with count 0 so
    the cache row is comparable across models.
    """
    return {label: len(rx.findall(text)) for label, rx in patterns}


# ── Cache row builder ───────────────────────────────────────────────────────

# Top-level cache keys reserved for harness metadata (schema v2+). Anything
# else at the top level is a bench row keyed by ``<repo>@<sha>::<backend>`` or
# ``<digest>::<backend>``. Consumers iterating cache.items() must skip these.
META_KEYS: frozenset[str] = frozenset({"_meta"})

# Top-level bench-cache schema version. Bumped when readers must change
# behaviour to interpret a row correctly; row-level ``schema_version``
# continues to track per-row evolution. ``assert_cache_schema_compatible``
# is the gate that prevents a future v3 cache from being silently misread
# by a v2 consumer.
BENCH_CACHE_SCHEMA_VERSION: int = 2


def is_row_key(key: str) -> bool:
    """True when ``key`` names a bench row (not a meta block).

    Centralised so report/picker/migration code stays in sync if more
    meta keys are added later.
    """
    return not (key in META_KEYS or key.startswith("_"))


def assert_cache_schema_compatible(cache: dict) -> None:
    """Refuse to read a bench cache produced by a newer writer.

    A cache with ``_meta.schema_version > BENCH_CACHE_SCHEMA_VERSION``
    means a future writer added row fields or restructured keys; current
    readers would silently mis-render the leaderboard or, worse, stamp
    new rows into a layout the future writer doesn't expect. Older or
    missing top-level versions are treated as v1 == compatible (no
    structural changes required of readers).
    """
    meta = cache.get("_meta") or {}
    version = meta.get("schema_version")
    if isinstance(version, int) and version > BENCH_CACHE_SCHEMA_VERSION:
        raise RuntimeError(
            f"bench cache has _meta.schema_version={version}, "
            f"this binary supports up to {BENCH_CACHE_SCHEMA_VERSION}. "
            f"Upgrade the harness or delete the cache to start fresh."
        )


def stamp_cache_schema(cache: dict) -> None:
    """Write the current writer's schema version into ``cache["_meta"]``.

    Called from update_row so any touched cache gets the marker even if
    it predates this field. Idempotent.
    """
    meta = cache.setdefault("_meta", {})
    meta["schema_version"] = BENCH_CACHE_SCHEMA_VERSION


def update_row(
    cache: dict,
    key: str,
    *,
    model: str,
    backend: str,
    router_endpoint: str,
    task_results: dict[str, dict] | None = None,
    metrics: dict | None = None,
    host_env_id: str | None = None,
) -> dict:
    """Merge a single bench result into the cache.

    Reuses any existing row at ``key`` so partial re-runs (`--tasks
    leak` after a full bench, etc.) don't lose unrelated task data.
    Updates ``last_benched_at`` always; sets ``first_benched_at`` only
    when the row is fresh.

    ``host_env_id`` (when supplied) is stamped on the row so the
    leaderboard can join back to ``cache["_meta"]["host_env_history"]``
    and prove which kernel + driver + GPU produced these numbers. Each
    re-bench against a different host environment yields a distinct id
    so consumers can spot mixed-provenance rows.
    """
    now = _now_iso()
    row = cache.get(key) or {
        "schema_version": 2,
        "model": model,
        "backend": backend,
        "router_endpoint": router_endpoint,
        "tasks": {},
        "metrics": {},
        "first_benched_at": now,
    }
    row["model"] = model
    row["backend"] = backend
    row["router_endpoint"] = router_endpoint
    row.setdefault("tasks", {})
    row.setdefault("metrics", {})
    # Schema bump on touch so older rows acquire the field even if no
    # other fields changed in this update.
    row["schema_version"] = max(int(row.get("schema_version", 1)), 2)
    stamp_cache_schema(cache)
    if task_results:
        for tname, tresult in task_results.items():
            row["tasks"][tname] = tresult
    if metrics:
        row["metrics"].update(metrics)
    if host_env_id is not None:
        row["host_env_id"] = host_env_id
    row["last_benched_at"] = now
    cache[key] = row
    return row


def reset_row_for_force(cache: dict, key: str) -> None:
    """Clear ``tasks`` and ``metrics`` for a force re-bench, preserving
    provenance (``first_benched_at``, ``model``, ``backend``,
    ``router_endpoint``). No-op when the row doesn't exist yet.

    Without this, ``--force`` re-runs the tasks but leaves stale
    metric fields (e.g. an old ``ttft_ms_first`` from a prior driver
    version) sitting next to the new run -- and the leaderboard can't
    tell which is which.
    """
    row = cache.get(key)
    if not isinstance(row, dict):
        return
    row["tasks"] = {}
    row["metrics"] = {}


# ── Host environment capture ────────────────────────────────────────────────

def capture_host_env() -> dict:
    """Snapshot the running host's kernel, GPU driver, and GPU model.

    Best-effort. Each piece is wrapped so a missing tool (no
    ``nvidia-smi`` on a CPU-only host, ``uname`` unavailable on
    Windows-via-WSL surfaces, etc.) leaves a sparse dict rather than
    crashing the bench. The dict is fed to ``host_env_id`` for hashing
    and to ``cache["_meta"]["host_env_history"]`` for human reading.
    """
    import datetime as dt
    import platform
    import subprocess

    out: dict = {
        "kernel": platform.release(),
        "captured_at": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    }
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            line = r.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                out["driver_version"] = parts[0]
                out["gpu_name"] = parts[1]
                # memory.total looks like "24576 MiB" -- parse the integer.
                mt_raw = parts[2].split()[0] if parts[2] else ""
                try:
                    out["gpu_memory_gb"] = round(int(mt_raw) / 1024.0, 2)
                except ValueError:
                    pass
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    # CUDA runtime: nvidia-smi prints "CUDA Version: 13.0" in the
    # header summary. The --query surface doesn't expose it, so we
    # parse the human-readable output as a fallback.
    try:
        r = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in (r.stdout or "").splitlines():
            if "CUDA Version" in line and ":" in line:
                tail = line.split("CUDA Version", 1)[1]
                _, _, val = tail.partition(":")
                cuda_ver = val.split("|")[0].strip()
                if cuda_ver:
                    out["cuda_version"] = cuda_ver
                break
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return out


def host_env_id(host_env: dict) -> str:
    """Stable 12-char hash of the env, ignoring ``captured_at``.

    Same kernel + driver + GPU model on different days -> same id, so
    ``cache["_meta"]["host_env_history"]`` accumulates one entry per
    distinct environment instead of one per bench run.
    """
    import hashlib

    sub = {k: v for k, v in host_env.items() if k != "captured_at"}
    raw = json.dumps(sub, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def stamp_host_env(cache: dict, env: dict) -> str:
    """Record ``env`` under ``cache["_meta"]["host_env_history"][<id>]``
    and return the id. Idempotent for the same env contents.

    The pointer ``cache["_meta"]["current_host_env_id"]`` is updated
    on every call so consumers can answer "which env produced the
    most recent bench in this cache?" without scanning rows.
    """
    env_id = host_env_id(env)
    meta = cache.setdefault("_meta", {})
    history = meta.setdefault("host_env_history", {})
    if env_id not in history:
        history[env_id] = env
    meta["current_host_env_id"] = env_id
    return env_id


def _now_iso() -> str:
    """Local copy of _probe_core.now_iso so this module stays
    importable when scripts/ isn't on sys.path. Same UTC ISO-8601
    second-resolution shape."""
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


# ── Router endpoint resolver ────────────────────────────────────────────────

ROUTER_HOST = os.environ.get("DEVAI_ROUTER_HOST", "devai-router")
BACKEND_PORTS = {
    "ollama": 11434,
    "vllm": 11435,
    "sglang": 11436,
}


def router_url_for(backend: str) -> str:
    port = BACKEND_PORTS.get(backend)
    if port is None:
        raise ValueError(f"unknown backend: {backend!r}")
    return f"http://{ROUTER_HOST}:{port}"


# ── Catalog helpers ─────────────────────────────────────────────────────────

def cache_key_for_entry(entry: dict, backend: str) -> str | None:
    """Resolve the top-level key for the **bench** cache.

    The same model can be served by more than one backend (the only
    overlap on this hardware is ``gpt-oss-20b`` and
    ``DeepSeek-R1-Distill-Qwen-7B``, which both fit on vLLM and
    SGLang). Bench scores, cold-start times, peak VRAM and TPS all
    depend on the backend, so each (model, backend) pair gets its
    own row. The key shape is:

    - HF backends: ``<repo>@<sha>::<backend>`` (e.g.
      ``openai/gpt-oss-20b@6cee5e81ee83::vllm``).
    - Ollama: ``<digest>::ollama``.

    Returns None when the underlying probe cache entry is malformed
    (no digest for Ollama, no repo+sha for HF).

    Note: the **probe** caches are still keyed by the bare identifier
    (digest, or ``<repo>@<sha>``); only the bench cache uses the
    composite form. ``run_for_target`` joins by reading the probe-
    cache entry first, then minting the bench-cache key from it.
    """
    base: str | None = None
    if backend == "ollama":
        digest = entry.get("digest")
        if isinstance(digest, str) and digest:
            base = digest
    else:
        repo = entry.get("repo")
        sha = entry.get("sha")
        if isinstance(repo, str) and isinstance(sha, str) and repo and sha:
            base = f"{repo}@{sha}"
    if base is None:
        return None
    return f"{base}::{backend}"


def migrate_bench_cache_keys(cache: dict) -> int:
    """Idempotent migration from pre-2026-05-02 cache keys (no
    ``::<backend>`` suffix) to the composite form. Walks the cache
    once and renames in place. Returns the number of keys renamed.

    Old rows always carry a ``backend`` field, so we use that to
    decide the suffix. Calling this on an already-migrated cache is
    a no-op. Top-level meta keys (``_meta`` etc.) are skipped via
    ``is_row_key``.
    """
    renames: list[tuple[str, str]] = []
    for k in list(cache.keys()):
        if not is_row_key(k):
            continue
        v = cache.get(k)
        if not isinstance(v, dict):
            continue
        if "::" in k:
            continue
        backend = v.get("backend")
        if not isinstance(backend, str) or not backend:
            continue
        renames.append((k, f"{k}::{backend}"))
    for old, new in renames:
        if new in cache:
            # Defensive: if a composite-key row already exists, leave
            # the old one alone rather than overwriting. Caller should
            # investigate manually.
            continue
        cache[new] = cache.pop(old)
    return len(renames)


def serving_alias(entry: dict) -> str | None:
    """Pick a single name to send as the OpenAI ``model`` field.

    Probes record the alias list from the catalog or from `/api/show`.
    For HF the canonical alias is the model directory name (matches
    the router's `--served-model-name`). For Ollama any alias works
    since the daemon resolves to digest.
    """
    aliases = entry.get("aliases") or []
    if isinstance(aliases, list) and aliases:
        return aliases[0]
    return None


def serving_alias_with_ctx(alias: str, ctx: int, backend: str) -> str:
    """Build the picker-style ``<name>@<ctx>`` suffix for HF backends.

    Ollama doesn't honour the suffix on the OpenAI-compat path, so it
    sends the bare alias. HF backends need the suffix so the router
    recreates with the right ``--max-model-len``.
    """
    if backend == "ollama" or ctx <= 0:
        return alias
    return f"{alias}@{ctx}"
