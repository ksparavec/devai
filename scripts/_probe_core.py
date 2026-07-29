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
import subprocess
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


# ── Image-digest drift (Phase C) ─────────────────────────────────────────────

def image_digest_via_cli(runtime: str, image_ref: str) -> str | None:
    """Return the sha256 digest of a local image via `<runtime> image inspect`.

    Tries `.Digest` (manifest digest) first, falling back to the first entry
    of `.RepoDigests` (`repo@sha256:...`, from which we keep only the digest).
    Returns None on any error (image absent, runtime missing) -- image-drift
    detection fails open.
    """
    for fmt in ("{{.Digest}}", "{{index .RepoDigests 0}}"):
        try:
            out = subprocess.run(
                [runtime, "image", "inspect", "--format", fmt, image_ref],
                capture_output=True, text=True, check=False, timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        val = (out.stdout or "").strip()
        if out.returncode == 0 and "sha256:" in val:
            return val.split("@")[-1] if "@" in val else val
    return None


def stamp_image_digest(cache: dict, *, digest: str, image_ref: str) -> None:
    """Record, in a top-level `_meta` block, the backend image digest the cache
    was probed against. Mirrors the bench cache's host-env stamping:
    `_meta.image_history[<digest>]` accumulates every image seen and
    `_meta.current_image_digest` points at the one this run used. The router
    compares current_image_digest against the running image to detect drift (a
    moved tag that silently invalidates the probe data). No-op on a falsy digest
    so a failed inspect never corrupts the block.
    """
    if not digest:
        return
    meta = cache.setdefault("_meta", {})
    history = meta.setdefault("image_history", {})
    if digest not in history:
        history[digest] = {"image_ref": image_ref, "first_seen": now_iso()}
    meta["current_image_digest"] = digest
    meta["current_image_ref"] = image_ref


# Per-row engine-image stamp. Same field name the bench cache uses for the
# same purpose (scripts/bench/_bench_core.py), so an operator reading either
# cache is answering the question the same way.
ROW_IMAGE_FIELD = "backend_image_digest"


def stamp_row_image(entry: dict, digest: str | None) -> None:
    """Record which engine image THIS row's verdicts were measured under.

    `_meta.current_image_digest` is cache-WIDE and therefore cannot answer
    the per-row question. Only models whose weights are still on disk can be
    re-probed, so a partial re-probe is the normal case: it moves the
    `_meta` pointer to the new image and leaves every un-reprobed row
    untouched. Without this stamp those rows are indistinguishable from
    freshly measured ones and `make probe-check` reports no drift.

    No-op on a falsy digest: an absent stamp honestly reads as "unknown",
    an empty one would read as a measurement that never happened.
    """
    if digest:
        entry[ROW_IMAGE_FIELD] = digest


def backfill_row_images(cache: dict) -> int:
    """Attribute unstamped rows to the cache's CURRENT `_meta` digest.

    A row carrying no stamp was measured under whatever
    `_meta.current_image_digest` pointed at when it was written -- and that
    is still what it points at, right up until the next run overwrites it.
    Calling this immediately BEFORE stamp_image_digest therefore recovers
    the correct attribution for every legacy row exactly once, instead of
    stranding them as permanently "unknown".

    Idempotent: rows that already carry a stamp are left alone, so this is
    a no-op on every subsequent run. Returns the number of rows backfilled.
    """
    meta = cache.get("_meta")
    if not isinstance(meta, dict):
        return 0
    digest = meta.get("current_image_digest")
    if not digest:
        return 0
    n = 0
    for name, entry in cache.items():
        if name.startswith("_") or not isinstance(entry, dict):
            continue
        if not entry.get(ROW_IMAGE_FIELD):
            entry[ROW_IMAGE_FIELD] = digest
            n += 1
    return n


def row_image_is_stale(
    entry: dict, run_digest: str | None, *, writable: bool = True
) -> bool:
    """True when this row's verdicts were measured on a DIFFERENT engine.

    This is an auto-invalidation trigger for the probers, unlike
    launch-fingerprint drift which is deliberately report-only. The
    difference is precision: a fingerprint is per-(model, launch shape)
    and the prober would have to rebuild each row's full argv to know
    what today's launch hashes to, so a near-miss turns a routine probe
    into an unrequested fleet-wide re-probe. The image digest is ONE
    value, known exactly for the current run, so the comparison is exact.

    Returns False when either side is unknown (nothing to compare) or
    when the caller cannot persist a result anyway (`writable=False`,
    i.e. --no-cache-write): invalidating a cell we cannot rewrite would
    just re-probe it on every dry run.
    """
    if not writable:
        return False
    stamp = entry.get(ROW_IMAGE_FIELD)
    if not stamp or not run_digest:
        return False
    return stamp != run_digest


def image_stamp_survey(
    cache: dict, current_digest: str | None
) -> dict[str, list[str]]:
    """Group model rows by the engine image their verdicts were measured under.

    Buckets: "current" (stamp matches `current_digest`), "stale" (stamp
    differs), "unstamped" (no stamp -- predates stamping, or the image
    inspect failed).

    With no `current_digest` every row lands in "unstamped": without a
    baseline nothing can be called stale. Guessing would either force a
    needless full re-probe or hide real drift, and bench-sync's classify()
    already makes exactly this call for the same reason.
    """
    out: dict[str, list[str]] = {"current": [], "stale": [], "unstamped": []}
    for name, entry in cache.items():
        if name.startswith("_") or not isinstance(entry, dict):
            continue
        stamp = entry.get(ROW_IMAGE_FIELD)
        if not current_digest or not stamp:
            out["unstamped"].append(name)
        elif stamp == current_digest:
            out["current"].append(name)
        else:
            out["stale"].append(name)
    return out


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
