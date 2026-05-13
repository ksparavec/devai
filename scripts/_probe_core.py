"""Backend-agnostic helpers for the per-(VRAM, ctx) probe pipeline.

Used by scripts/probe-ollama-reasoning.py today. Will be shared with
scripts/probe-vllm-reasoning.py and scripts/probe-sglang-reasoning.py
when those land. Anything specific to a wire protocol (Ollama
/api/chat, OpenAI /v1/chat/completions, SGLang /generate) lives in
the per-backend prober — only logic that operates on the cache shape
or on raw response text belongs here.

Cache shape (shared across backends):

    <top-key>: {
      "schema_version": <int>,
      "aliases": [...],
      "max_context": <int>,
      "capability": "structured|inline|unsupported|error|unknown",
      "probes": {
        "<vram_gb>": {
          "<ctx>": {
            "ctx", "vram_gb",
            "capability",
            "probed_at", "probe_seconds",
            ...backend-specific fields
          }
        }
      },
      "first_probed_at", "last_probed_at"
    }

Top-level keys differ per backend (Ollama: digest; vLLM/SGLang: repo@sha).
The probes-dict shape is identical so smallest_clean_probe and
update_canonical_capability work for any backend.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Callable

from _capability import Capability


# ── Time ─────────────────────────────────────────────────────────────────────

def now_iso() -> str:
    """UTC timestamp at second resolution, ISO 8601, no microseconds."""
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


# ── HTTP ─────────────────────────────────────────────────────────────────────

def http_post(url: str, body: dict, timeout: float) -> dict:
    """POST JSON, return parsed JSON. Raises on transport / decode errors."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_get(url: str, timeout: float) -> dict:
    """GET, return parsed JSON. Raises on transport / decode errors."""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# ── Cache I/O ────────────────────────────────────────────────────────────────

def load_cache(path: Path) -> dict:
    """Return parsed cache or empty dict on missing.

    A missing file is normal (first probe run). A corrupt file is NOT
    normal -- it means a previous writer was killed mid-write or the
    disk lost data. Treating corruption as "empty" silently discards
    hundreds of probe-hours, so we log loudly and re-raise so the
    caller can decide whether to abort or rebuild.
    """
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(
            f"[probe-core] CORRUPT cache at {path}: {exc}. "
            f"Refusing to silently treat as empty. "
            f"Delete the file to force a fresh probe run.",
            file=sys.stderr,
        )
        raise


def save_cache(path: Path, cache: dict) -> None:
    """Write cache atomically (tmp file + os.replace), sorted keys, trailing newline.

    POSIX guarantees os.replace is atomic on the same filesystem, so a
    crash mid-write leaves either the old file intact or the new file
    fully written -- never a partial JSON document that load_cache
    would later refuse. Sorted keys keep diffs stable across runs and
    across machines -- important for catching real probe deltas in
    code review.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(cache, indent=2, sort_keys=True) + "\n"
    tmp.write_text(payload)
    os.replace(tmp, path)


# ── Alias selection ──────────────────────────────────────────────────────────

def canonical_alias(aliases: list[str]) -> str:
    """Pick the canonical name from an entry's alias list.

    Preference: explicit tag over `:latest`, longer (more descriptive)
    over shorter. Aliases share semantics within a backend's grouping
    key (Ollama: digest, vLLM: repo@sha), so the choice only affects
    log readability and downstream display.
    """
    if not aliases:
        return ""
    return sorted(aliases, key=lambda n: (n.endswith(":latest"), -len(n)))[0]


# ── Reasoning capability helpers ─────────────────────────────────────────────

def has_inline_think_markers(text: str) -> bool:
    """Detect `<think>` block markers in plain text content.

    Both opening and closing tags trigger — some models emit only one
    when the response is truncated by num_predict. Backend-agnostic
    because it operates on already-extracted content text.
    """
    if not text:
        return False
    return "<think>" in text or "</think>" in text


# ── Probes-dict navigation ───────────────────────────────────────────────────

def smallest_clean_probe(entry: dict) -> dict | None:
    """Smallest-ctx probe whose capability is a non-error reasoning value.

    Handles both schema layouts:
      nested:  probes[<vram>][<ctx>]  → walk both levels.
      flat:    probes[<ctx>]          → walk one level. Encountered
               transiently while migrating older schemas.

    Distinguishes the two at each value by checking for the `capability`
    key — present on a probe record, absent on a vram bucket (which
    contains probe records as its values). Ties broken by smallest vram
    so the "tightest" GPU that still classified cleanly wins.
    """
    probes = entry.get("probes") or {}
    candidates: list[dict] = []
    for value in probes.values():
        if not isinstance(value, dict):
            continue
        if "capability" in value:
            # Flat (single-dimension) probe record.
            if value.get("capability") not in (None, Capability.ERROR):
                candidates.append(value)
        else:
            # vram bucket (nested layout).
            for p in value.values():
                if isinstance(p, dict) and p.get("capability") not in (None, Capability.ERROR):
                    candidates.append(p)
    if not candidates:
        return None
    candidates.sort(
        key=lambda p: (int(p.get("ctx") or 0), int(p.get("vram_gb") or 0)),
    )
    return candidates[0]


def update_canonical_capability(entry: dict) -> None:
    """Refresh top-level `capability` from the smallest clean probe.

    The smallest fitting tier is the most-trustworthy capability signal
    (highest chance of fitting on GPU, no spill). When every probe
    errored we fall back to "error"; with no probes at all, "unknown".
    """
    smallest = smallest_clean_probe(entry)
    if smallest:
        entry["capability"] = smallest.get("capability") or Capability.UNKNOWN
        return
    if entry.get("probes"):
        entry["capability"] = Capability.ERROR
    else:
        entry.setdefault("capability", Capability.UNKNOWN)


# ── Implied-fail propagation ─────────────────────────────────────────────────

def propagate_implied_fail(
    *,
    vram_band: dict[str, dict],
    targets: list[int],
    failed_ctx: int,
    force_set: set[int],
    build_implied_record: Callable[[int], dict],
) -> list[int]:
    """Fill larger ctx slots with synthetic "implied fail" records.

    KV memory grows with ctx and weights are constant — so if (vram,
    ctx_low) failed to fit, every larger ctx at the same vram will too.
    Skip the actual probes and write deterministic placeholders. This
    saves wall time and avoids stressing the daemon with enormous
    CPU/RAM offload loads (the original cause of the qwen3.5:9b @ 16G
    crash sequence in the Ollama prober).

    `build_implied_record(larger_ctx)` is supplied per-backend so the
    record shape matches what that backend stores for real probes
    (Ollama: `fully_on_gpu: False`, evidence with `implied_from_ctx`;
    vLLM/SGLang: `fits: False`, `evidence.kind: "implied_spill"`).

    Cells already in vram_band are preserved unless the operator forced
    them — same rule as the live probe loop.

    Returns ascending list of ctx values that were filled.
    """
    implied: list[int] = []
    for larger in (t for t in targets if t > failed_ctx):
        larger_key = str(larger)
        if larger_key in vram_band and larger not in force_set:
            continue
        vram_band[larger_key] = build_implied_record(larger)
        implied.append(larger)
    return implied
