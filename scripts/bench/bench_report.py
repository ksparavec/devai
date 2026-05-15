#!/usr/bin/env python3
"""Render ``deploy/.bench-cache.json`` as a Markdown leaderboard.

Pure read-only. Joins each row's task scores and metrics into a
single line per (model, backend) pair, sorted by aggregate score
(descending). Safe to run any time; doesn't touch the cache.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _probe_core import load_cache  # noqa: E402
from bench._bench_core import (  # noqa: E402
    DEFAULT_CACHE_PATH,
    assert_cache_schema_compatible,
    is_row_key,
    migrate_bench_cache_keys,
)

# Host VRAM cap. Defaults to 24 GB to match the project's reference card
# (RTX 4000 PRO Blackwell). Override with GPU_MEMORY_GB at run-time when
# rendering against a cache produced on different hardware.
DEFAULT_HOST_VRAM_GB = float(os.environ.get("GPU_MEMORY_GB", "24"))


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
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}{suffix}".rstrip()
    return f"{v}{suffix}"


def _kv_pressure_pct(peak_vram_gb: float | None, host_vram_gb: float) -> float | None:
    """``peak_vram_gb / host_vram_gb`` as a percentage, or None if peak
    is missing. The bench's "KV-pressure observations" section in
    ``docs/bench-results.md`` calls 95 % the threshold where KV paging
    starts to bite -- this column makes that visible per row.
    """
    if peak_vram_gb is None or host_vram_gb <= 0:
        return None
    return float(peak_vram_gb) / float(host_vram_gb) * 100.0


def _ctx_label(ctx: int) -> str:
    """Render a context size as ``32K``, ``128K``, etc. when the value
    is a clean multiple of 1024; otherwise fall back to the raw int.
    ``0`` and unknown ctxs render as ``-`` so the leaderboard column
    stays narrow.
    """
    if not ctx or ctx <= 0:
        return "-"
    if ctx % 1024 == 0:
        k = ctx // 1024
        return f"{k}K"
    return str(ctx)


def render(cache: dict, host_vram_gb: float = DEFAULT_HOST_VRAM_GB) -> str:
    rows: list[dict] = []
    for key, row in cache.items():
        if not is_row_key(key):
            # Meta blocks (_meta etc.) live alongside row keys at the top
            # level; the leaderboard only renders bench rows.
            continue
        if not isinstance(row, dict):
            continue
        agg = _aggregate(row)
        ctx_raw = row.get("context")
        try:
            ctx = int(ctx_raw) if ctx_raw is not None else 0
        except (TypeError, ValueError):
            ctx = 0
        rows.append({
            "key": key,
            "model": row.get("model", key),
            "backend": row.get("backend", "?"),
            "ctx": ctx,
            "agg": agg,
            "row": row,
        })
    # Group by (model, ctx) so multi-ctx benches cluster together; ties
    # broken by aggregate score so the strongest row floats up within
    # each cluster.
    rows.sort(
        key=lambda r: (r["model"], r["ctx"], r["agg"] is None, -(r["agg"] or 0.0))
    )

    meta = cache.get("_meta") or {}
    history = meta.get("host_env_history") or {}
    current_id = meta.get("current_host_env_id")

    lines: list[str] = []
    if current_id and current_id in history:
        env = history[current_id]
        lines.append(
            f"_Host env_ `{current_id}`: kernel `{env.get('kernel', '?')}`, "
            f"driver `{env.get('driver_version', '?')}`, "
            f"GPU `{env.get('gpu_name', '?')}` "
            f"({env.get('gpu_memory_gb', '?')} GB), "
            f"CUDA `{env.get('cuda_version', '?')}`, "
            f"captured `{env.get('captured_at', '?')}`."
        )
        lines.append("")
    lines.append(
        "| Model | Backend | CTX | Env | Agg | GSM8K | HumanEval | Tools | "
        "Leak rate | TTFT first | TTFT p50 | TPS | Peak VRAM | KV % |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
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
        kv_pct = _kv_pressure_pct(peak, host_vram_gb)
        # Round KV % to one decimal so the column stays narrow.
        kv_str = "-" if kv_pct is None else f"{kv_pct:.1f}%"
        env_id = row.get("host_env_id") or "-"
        lines.append(
            f"| {r['model']} | {r['backend']} | {_ctx_label(r['ctx'])} | "
            f"{env_id} | {_fmt(r['agg'])} | "
            f"{_fmt(gsm)} | {_fmt(he)} | {_fmt(tools)} | {_fmt(leak)} | "
            f"{_fmt(ttft_first, ' ms')} | {_fmt(ttft_p50, ' ms')} | "
            f"{_fmt(tps, ' tok/s')} | {_fmt(peak, ' GB')} | {kv_str} |"
        )
    lines.append("")
    lines.append(
        "_Schema v3: each row reflects one (model, backend, ctx) cell. "
        "Rows are grouped by (model, ctx); `-` means no bench data at "
        "that cell. Re-run `make bench --ctx <N>` to fill missing tiers._"
    )
    lines.append("")
    lines.append(
        f"_KV % = `peak_vram_gb / {host_vram_gb:g}` (host VRAM cap, "
        f"override via `GPU_MEMORY_GB`). 95 % is the rule-of-thumb "
        f"threshold where KV paging starts to bite -- see "
        f"`docs/bench-results.md` > 'KV-pressure observations'._"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    ap.add_argument(
        "--host-vram-gb",
        type=float,
        default=DEFAULT_HOST_VRAM_GB,
        help="host VRAM cap used to compute the KV %% column",
    )
    args = ap.parse_args()
    cache = load_cache(args.cache)
    assert_cache_schema_compatible(cache)
    # In-memory only -- bench_report is read-only against the on-disk cache.
    # The runner is the writer; it persists the migrated form on next save.
    migrate_bench_cache_keys(cache)
    if not cache:
        print(
            f"# Bench leaderboard\n\n"
            f"_no data -- bench cache at {args.cache} is empty_"
        )
        return
    print("# Bench leaderboard\n")
    print(render(cache, host_vram_gb=args.host_vram_gb))


if __name__ == "__main__":
    main()
