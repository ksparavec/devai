#!/usr/bin/env python3
"""Render ``deploy/.bench-cache.json`` as a Markdown leaderboard.

Pure read-only. Joins each row's task scores and metrics into a
single line per (model, backend) pair, sorted by aggregate score
(descending). Safe to run any time; doesn't touch the cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _probe_core import load_cache  # noqa: E402
from bench._bench_core import DEFAULT_CACHE_PATH  # noqa: E402


def _pick_score(tasks: dict, prefix: str, key: str) -> float | None:
    """Return ``tasks[<prefix>_*][<key>]`` for the first matching
    subset-keyed entry. Different runs may use different ``n``, so we
    look for any task whose name starts with ``prefix``."""
    for tname, tdata in tasks.items():
        if tname.startswith(prefix) and isinstance(tdata, dict):
            v = tdata.get(key)
            if v is not None:
                return float(v)
    return None


def _aggregate(row: dict) -> float | None:
    """Composite score = unweighted mean of available correctness
    scores. None when a row has no scored tasks (latency-only run).
    """
    tasks = row.get("tasks") or {}
    parts: list[float] = []
    for prefix, key in (
        ("gsm8k_", "score"),
        ("humaneval_", "pass@1"),
        ("tools_use", "score"),
    ):
        v = _pick_score(tasks, prefix, key)
        if v is not None:
            parts.append(v)
    if not parts:
        return None
    return sum(parts) / len(parts)


def _fmt(v: object, suffix: str = "") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}{suffix}".rstrip()
    return f"{v}{suffix}"


def render(cache: dict) -> str:
    rows: list[dict] = []
    for key, row in cache.items():
        if not isinstance(row, dict):
            continue
        agg = _aggregate(row)
        rows.append({
            "key": key,
            "model": row.get("model", key),
            "backend": row.get("backend", "?"),
            "agg": agg,
            "row": row,
        })
    rows.sort(key=lambda r: (r["agg"] is None, -(r["agg"] or 0.0)))

    lines: list[str] = []
    lines.append(
        "| Model | Backend | Agg | GSM8K | HumanEval | Tools | Leak rate | "
        "TTFT first | TTFT p50 | TPS | Peak VRAM |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for r in rows:
        row = r["row"]
        tasks = row.get("tasks") or {}
        metrics = row.get("metrics") or {}
        gsm = _pick_score(tasks, "gsm8k_", "score")
        he = _pick_score(tasks, "humaneval_", "pass@1")
        tools = _pick_score(tasks, "tools_use", "score")
        leak = (tasks.get("leak_probe") or {}).get("leak_rate")
        ttft_first = metrics.get("ttft_ms_first")
        ttft_p50 = metrics.get("ttft_ms_steady_p50")
        tps = metrics.get("tps_sustained_p50")
        peak = metrics.get("peak_vram_gb")
        lines.append(
            f"| {r['model']} | {r['backend']} | {_fmt(r['agg'])} | "
            f"{_fmt(gsm)} | {_fmt(he)} | {_fmt(tools)} | {_fmt(leak)} | "
            f"{_fmt(ttft_first, ' ms')} | {_fmt(ttft_p50, ' ms')} | "
            f"{_fmt(tps, ' tok/s')} | {_fmt(peak, ' GB')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    args = ap.parse_args()
    cache = load_cache(args.cache)
    if not cache:
        print(f"# Bench leaderboard\n\n_no data — bench cache at {args.cache} is empty_")
        return
    print("# Bench leaderboard\n")
    print(render(cache))


if __name__ == "__main__":
    main()
