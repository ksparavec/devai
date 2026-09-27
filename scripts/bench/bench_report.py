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


def _sampling_footnote(rows: dict) -> list[str]:
    """Lines naming every row NOT benched greedily.

    A leaderboard that mixes greedy and non-greedy rows without saying so
    invites exactly the wrong comparison. Rows predating the stamp are
    listed separately as unknown rather than assumed greedy.
    """
    non_greedy: list[str] = []
    unknown = 0
    for key, row in rows.items():
        if key.startswith("_") or not isinstance(row, dict):
            continue
        samp = (row.get("metrics") or {}).get("sampling")
        if not isinstance(samp, dict):
            unknown += 1
            continue
        if samp.get("source") != "greedy_default":
            non_greedy.append(
                f"  - `{row.get('model')}` [{row.get('backend')}] @ "
                f"{row.get('context')}: temperature={samp.get('temperature')}, "
                f"top_p={samp.get('top_p')} -- NOT directly comparable with the "
                f"greedy rows (see deploy/bench-sampling.json for why)")
    out: list[str] = []
    if non_greedy or unknown:
        out.append("")
        out.append("### Sampling")
        out.append("")
        out.append("Scored tasks run greedily (temperature=0, top_p=1.0) so rows "
                   "are comparable across backends, which default differently.")
        if non_greedy:
            out.append("")
            out.append("Exceptions, benched at their override:")
            out.extend(non_greedy)
        if unknown:
            out.append("")
            out.append(f"{unknown} row(s) predate per-row sampling stamps and "
                       f"carry UNKNOWN sampling. Re-bench to make them "
                       f"comparable.")
    return out


def _pick_score(tasks: dict, prefix: str, key: str) -> float | None:
    """Return ``tasks[<prefix>_*][<key>]`` for the first matching
    subset-keyed entry. Different runs may use different ``n``, so we
    look for any task whose name starts with ``prefix``.

    Prefixes must be specific enough to name exactly one task family.
    In particular HumanEval and HumanEval+ are separate benchmarks
    written to the same row (``humaneval_subset_<n>`` and
    ``humaneval_plus_subset_<n>``), so a bare ``humaneval_`` prefix
    matches BOTH and publishes whichever happens to come first. Always
    use ``humaneval_subset_`` / ``humaneval_plus_subset_`` -- never
    bare ``humaneval_``. scripts/model-picker.py keys the same way.
    """
    for tname, tdata in tasks.items():
        if tname.startswith(prefix) and isinstance(tdata, dict):
            v = tdata.get(key)
            if v is not None:
                return float(v)
    return None


def _pick_timeouts(tasks: dict, prefix: str) -> int:
    """``n_timeouts`` of the entry ``_pick_score`` reads (0 when absent:
    rows benched before 2026-09-27 did not record time-outs)."""
    for tname, tdata in tasks.items():
        if tname.startswith(prefix) and isinstance(tdata, dict):
            return int(tdata.get("n_timeouts") or 0)
    return 0


def _fmt_score(tasks: dict, prefix: str, key: str) -> str:
    """Score cell; a task with time-outs shows them, since a time-out is
    scored as a wrong answer and the score alone cannot tell them apart."""
    cell = _fmt(_pick_score(tasks, prefix, key))
    t = _pick_timeouts(tasks, prefix)
    return f"{cell} (t={t})" if t else cell


def _aggregate(row: dict) -> float | None:
    """Composite score = unweighted mean of available correctness
    scores. None when a row has no scored tasks (latency-only run).
    """
    tasks = row.get("tasks") or {}
    parts: list[float] = []
    for prefix, key in (
        ("gsm8k_", "score"),
        ("humaneval_subset_", "pass@1"),
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
    is missing. How full the card got -- NOT a measure of KV pressure:
    vLLM and SGLang preallocate their pool at launch, so this mostly
    reflects --gpu-memory-utilization / --mem-fraction-static. The "95 %
    threshold where KV paging starts to bite" this column used to be
    read against had no data behind it and is withdrawn
    (docs/bench-results.md "KV-pressure observations").
    """
    if peak_vram_gb is None or host_vram_gb <= 0:
        return None
    return float(peak_vram_gb) / float(host_vram_gb) * 100.0


def _env_label(row: dict) -> str:
    """Render the host environment(s) that produced this row.

    ``host_env_id`` is stamped per task (each task's own provenance)
    and on the row itself (the run that produced ``metrics``). A row
    whose tasks were benched across several host environments -- e.g.
    a forced re-run of the default tasks after a driver upgrade, which
    leaves the sharper benches untouched -- renders every distinct id
    so the mixed provenance is visible instead of being flattened to
    the most recent one. Legacy rows carry only the row-level id.
    """
    ids: list[str] = []
    row_id = row.get("host_env_id")
    if isinstance(row_id, str) and row_id:
        ids.append(row_id)
    for tdata in (row.get("tasks") or {}).values():
        if not isinstance(tdata, dict):
            continue
        tid = tdata.get("host_env_id")
        if isinstance(tid, str) and tid and tid not in ids:
            ids.append(tid)
    if not ids:
        return "-"
    return ", ".join(sorted(ids))


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
        "| Model | Backend | CTX | Env | Agg | GSM8K | HumanEval | "
        "HumanEval+ | Tools | "
        "Leak rate | TTFT first | TTFT p50 | TPS | Peak VRAM | VRAM % |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for r in rows:
        row = r["row"]
        tasks = row.get("tasks") or {}
        metrics = row.get("metrics") or {}
        gsm = _fmt_score(tasks, "gsm8k_", "score")
        he = _fmt_score(tasks, "humaneval_subset_", "pass@1")
        hep = _fmt_score(tasks, "humaneval_plus_subset_", "pass@1")
        tools = _fmt_score(tasks, "tools_use", "score")
        leak = (tasks.get("leak_probe") or {}).get("leak_rate")
        ttft_first = metrics.get("ttft_ms_first")
        ttft_p50 = metrics.get("ttft_ms_steady_p50")
        tps = metrics.get("tps_sustained_p50")
        peak = metrics.get("peak_vram_gb")
        kv_pct = _kv_pressure_pct(peak, host_vram_gb)
        # Round VRAM % to one decimal so the column stays narrow.
        kv_str = "-" if kv_pct is None else f"{kv_pct:.1f}%"
        env_id = _env_label(row)
        lines.append(
            f"| {r['model']} | {r['backend']} | {_ctx_label(r['ctx'])} | "
            f"{env_id} | {_fmt(r['agg'])} | "
            f"{gsm} | {he} | {hep} | "
            f"{tools} | {_fmt(leak)} | "
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
        "_`(t=N)`: N samples hit the per-sample working-time limit and are "
        "scored as wrong answers; the score with them counted right is an "
        "upper bound. Rows benched before 2026-09-27 did not record "
        "time-outs (and their limit also counted queueing)._"
    )
    lines.append("")
    lines.append(
        f"_VRAM % = `peak_vram_gb / {host_vram_gb:g}` (host VRAM cap, "
        f"override via `GPU_MEMORY_GB`): how full the card got, not KV "
        f"pressure -- vLLM and SGLang preallocate their pool at launch. No "
        f"threshold is implied; the '95 % where KV paging starts to bite' "
        f"once quoted here had no data behind it and is withdrawn (see "
        f"`docs/bench-results.md` > 'KV-pressure observations')._"
    )
    lines.extend(_sampling_footnote(cache))
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    ap.add_argument(
        "--host-vram-gb",
        type=float,
        default=DEFAULT_HOST_VRAM_GB,
        help="host VRAM cap used to compute the VRAM %% column",
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
