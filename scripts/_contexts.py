"""Shared context-tier constants used by the probe driver, the picker, and
select-models. Keeping them here stops the three callers from drifting.

Tiers are powers-of-two-ish boundaries that match how ollama and friends
expose context budgets. The defaults are also what the JupyterLab picker
displays as user-facing rows.

Errors propagate verbatim; nothing in this module talks to the network.
"""

from __future__ import annotations

import os

STANDARD_CONTEXTS: tuple[int, ...] = (32768, 65536, 131072, 262144)


def parse_context_list(raw: str) -> list[int]:
    """Parse "32K,64K,128K" or "32768,65536,131072" into a list[int].

    Returns [] for empty / disabled inputs ("", "none", "off", "0").
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
    """Parse a single context value. Raises ValueError if more than one."""
    values = parse_context_list(raw)
    if len(values) != 1:
        raise ValueError(f"expected a single context value, got: {raw!r}")
    return values[0]


def standard_contexts(env_override: str | None = None) -> list[int]:
    """Return the active tier list — env override or the default."""
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


def effective_targets(tiers: list[int], max_context: int) -> list[int]:
    """Per-model effective probe targets.

    Cap each tier at the model's design ceiling and dedup. Return ascending.
    A model with max_context=98304 asked for tiers [32K, 64K, 128K, 256K]
    yields [32768, 65536, 98304] — never above the ceiling.
    """
    if max_context <= 0:
        return sorted({t for t in tiers if t > 0})
    capped = {min(t, max_context) for t in tiers if t > 0}
    return sorted(capped)


def context_label(context: int) -> str:
    return f"{context // 1024}K" if context >= 1024 else str(context)
