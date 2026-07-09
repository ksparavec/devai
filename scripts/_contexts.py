"""Shared dimension constants and parsers for the model-fit pipeline.

Two dimensions parameterise probing today: GPU VRAM size and request
context size. Both expand to more values in the future. Tier arrays and
the format helpers live here so the prober, picker, selector, and router
cannot drift on what counts as a "standard" tier.

User-facing format is human-readable: "16G" for VRAM in gigabytes, "32K"
for context length in 1024-token chunks. We preserve those tokens at the
input boundary and parse them once into ints for arithmetic.

Errors propagate verbatim; nothing in this module talks to the network.
"""

from __future__ import annotations

import os
from collections.abc import Callable

# Default tier arrays. Drive every loop in the probe and diagnostics.
# Adding a new tier is a one-line edit here — no Python flow changes.
STANDARD_CONTEXTS_K: tuple[int, ...] = (32, 64, 128, 256)
STANDARD_VRAM_GB: tuple[int, ...] = (16, 24)

# Back-compat: the old-name constant resolves to context tokens for any
# caller that expected raw token counts.
STANDARD_CONTEXTS: tuple[int, ...] = tuple(k * 1024 for k in STANDARD_CONTEXTS_K)

# Binary-search grid: every multiple of 32K up to 256K (256/8 = 32K step).
# The probe keeps ONE ctx per (model, backend) -- the largest that serves --
# and finds it on this finer grid so the result is precise (160K, not just
# "128K or 256K"). Every result is one of these 8 tiers, hence a multiple of
# 32K. Widen/narrow by editing this one line.
BINARY_SEARCH_CONTEXTS_K: tuple[int, ...] = (32, 64, 96, 128, 160, 192, 224, 256)
BINARY_SEARCH_CONTEXTS: tuple[int, ...] = tuple(
    k * 1024 for k in BINARY_SEARCH_CONTEXTS_K
)


def binary_search_max_ctx(
    works: Callable[[int], bool],
    *,
    position_limit: int | None = None,
    grid: tuple[int, ...] = BINARY_SEARCH_CONTEXTS,
) -> int | None:
    """Return the largest ``grid`` context where ``works(ctx)`` is True.

    Implements the "test the top, then bisect" tree: probe the highest tier
    first (the common case is a model that fits its full ceiling, resolved in
    one probe); on failure, binary-search the remainder. Over the 8-tier grid
    that is at most 4 ``works`` calls (1 top probe + <=3 bisections).

    ``position_limit`` (a model's as-delivered max context) short-circuits any
    tier above it to False **without** calling ``works`` -- the engine would
    device-assert past its trained ceiling, so there is nothing to launch.

    Monotonic-fit assumption: if a model serves ctx X it serves every smaller
    ctx (KV memory only shrinks). Returns None when even the smallest tier
    fails (the model fits nowhere on this grid -> caller should exclude it).
    """
    if not grid:
        return None
    ordered = sorted(set(grid))

    def ok(ctx: int) -> bool:
        if position_limit is not None and ctx > position_limit:
            return False  # past the model's ceiling; instant fail, no launch
        return works(ctx)

    hi = len(ordered) - 1
    # Fast path: try the ceiling first.
    if ok(ordered[hi]):
        return ordered[hi]

    # Ceiling failed -- binary-search [0, hi-1] for the largest working tier.
    lo, hi = 0, hi - 1
    best: int | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if ok(ordered[mid]):
            best = ordered[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def parse_context_token(raw: str) -> int:
    """Parse "32K" → 32768. Strict: must end in 'K' (case-insensitive)."""
    if raw is None:
        raise ValueError("expected context token like '32K', got None")
    text = raw.strip().upper()
    if not text.endswith("K"):
        raise ValueError(f"context token must end in 'K', got: {raw!r}")
    body = text[:-1].strip()
    if not body:
        raise ValueError(f"empty context value: {raw!r}")
    value = int(body) * 1024
    if value <= 0:
        raise ValueError(f"context must be positive: {raw!r}")
    return value


def parse_context_list(raw: str) -> list[int]:
    """Parse "32K,64K,128K,256K" or "32768,65536" into a list[int].

    Returns [] for empty / disabled inputs ("", "none", "off", "0").
    Both K-suffix and bare-int forms are accepted (back-compat).
    """
    if raw is None:
        return []
    text = raw.strip().lower()
    if text in ("", "none", "off", "0"):
        return []
    out: list[int] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if item.endswith("k"):
            value = int(item[:-1]) * 1024
        else:
            value = int(item)
        if value > 0 and value not in out:
            out.append(value)
    return out


def parse_context_value(raw: str) -> int:
    """Parse a single context value. Raises if more than one given."""
    values = parse_context_list(raw)
    if len(values) != 1:
        raise ValueError(f"expected a single context value, got: {raw!r}")
    return values[0]


def parse_vram_token(raw: str) -> int:
    """Parse "16G" → 16. Strict: must end in 'G' (case-insensitive)."""
    if raw is None:
        raise ValueError("expected vram token like '16G', got None")
    text = raw.strip().upper()
    if not text.endswith("G"):
        raise ValueError(f"vram token must end in 'G', got: {raw!r}")
    body = text[:-1].strip()
    if not body:
        raise ValueError(f"empty vram value: {raw!r}")
    value = int(body)
    if value <= 0:
        raise ValueError(f"vram must be positive: {raw!r}")
    return value


def parse_vram_list(raw: str) -> list[int]:
    """Parse "16G,24G" into [16, 24] (gigabytes as ints).

    Returns [] for empty inputs. Both G-suffix and bare-int forms work.
    """
    if raw is None:
        return []
    text = raw.strip()
    if text.lower() in ("", "none", "off", "0"):
        return []
    out: list[int] = []
    for part in text.split(","):
        item = part.strip().upper()
        if not item:
            continue
        if item.endswith("G"):
            value = int(item[:-1])
        else:
            value = int(item)
        if value > 0 and value not in out:
            out.append(value)
    return out


def standard_contexts(env_override: str | None = None) -> list[int]:
    """Return the active context-tier list — env override or the default."""
    raw = (
        env_override
        if env_override is not None
        else os.environ.get("PROBE_CONTEXTS")
    )
    if raw:
        parsed = parse_context_list(raw)
        if parsed:
            return parsed
    return list(STANDARD_CONTEXTS)


def standard_vram_budgets(env_override: str | None = None) -> list[int]:
    """Return the active VRAM-band list in GB — env override or default."""
    raw = (
        env_override
        if env_override is not None
        else os.environ.get("PROBE_VRAMS")
    )
    if raw:
        parsed = parse_vram_list(raw)
        if parsed:
            return parsed
    return list(STANDARD_VRAM_GB)


def effective_targets(tiers: list[int], max_context: int) -> list[int]:
    """Per-model effective probe targets — no clamps.

    Returns the requested tiers, deduped and ascending. The model's
    declared max_context is ignored: a model card claiming 131K can
    still be probed at 256K (rope-extrapolation territory) so the
    operator sees the actual engine outcome, not a pre-emptive skip.
    Used to clamp at `max_context`; that hid valid probes whenever
    the engine accepted ctx beyond the model's nominal ceiling.
    """
    return sorted({t for t in tiers if t > 0})


def context_label(context: int) -> str:
    """Render an int context as the canonical token: 32768 → '32K'."""
    return f"{context // 1024}K" if context >= 1024 else str(context)


def vram_label(vram_gb: int) -> str:
    """Render an int VRAM value as the canonical token: 16 → '16G'."""
    return f"{vram_gb}G"


def vram_overhead_bytes(host_gb: int, target_gb: int) -> int:
    """Bytes for OLLAMA_GPU_OVERHEAD so the daemon believes it has
    target_gb of usable VRAM on a host_gb card.

    Returns 0 when target >= host (no constraint needed; the probe runs
    against the full card). Raises when target > host (we cannot simulate
    a larger card on a smaller one).
    """
    if host_gb <= 0:
        raise ValueError(f"host VRAM must be positive: {host_gb}")
    if target_gb <= 0:
        raise ValueError(f"target VRAM must be positive: {target_gb}")
    if target_gb > host_gb:
        raise ValueError(
            f"target VRAM {target_gb}G exceeds host VRAM {host_gb}G — "
            "cannot simulate a larger card on a smaller one"
        )
    if target_gb == host_gb:
        return 0
    return (host_gb - target_gb) * 1024 * 1024 * 1024
