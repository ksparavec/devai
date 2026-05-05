#!/usr/bin/env python3
"""Pearls bench runner -- runs the Programming Pearls task against
every coding-tier model the probe cache reports as fitting on this
host.

Mirrors scripts/bench/bench_runner.py but with a tighter scope:

  - one task (``pearls``), no gsm8k/humaneval/tools/leak/longctx;
  - default ``--repo`` regex restricts the sweep to coding-tier
    families (``qwen3-coder``, ``gpt-oss``, ``deepseek``, ``nemotron``,
    Qwen3 8B/14B/32B). Any explicit ``--repo`` overrides this.
  - writes ``deploy/.bench-pearls-cache.json`` (separate from
    ``.bench-cache.json``) so v1 can be torn down without affecting
    the main leaderboard.

Heavy machinery -- model discovery, inspect_ai dispatch, VRAM
sampling, Prometheus snapshot -- is imported from the sibling runner
so the pearls runner stays a thin orchestration wrapper. The plug-in
path later: move ``tasks/pearls.py`` into ``scripts/bench/tasks/``,
add a branch in ``scripts/bench/bench_runner.py``, retire this file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Sibling-runner helpers: model discovery, inspect_ai dispatch, router
# health check, Prometheus snapshot. The pearls runner wraps these
# rather than re-implementing them.
from _probe_core import http_get, load_cache, save_cache  # noqa: E402
from bench.bench_runner import (  # noqa: E402
    DEFAULT_HOST_VRAM_GB,
    DEFAULT_INSPECT_LOG_DIR,
    PROBE_CACHE_BY_BACKEND,
    _aggregate_score,
    _check_router,
    _fetch_backend_metrics,
    _invoke_inspect_task,
    _now_iso,
    discover_models,
)
from bench.bench_vram_snapshot import VramSampler  # noqa: E402
from bench_pearls._bench_core import (  # noqa: E402
    DEFAULT_CACHE_PATH,
    migrate_bench_cache_keys,
    router_url_for,
    serving_alias_with_ctx,
    update_row,
)

# ─────────────────────────────────────────────────────────────────────────────
# Coding-tier filter
# ─────────────────────────────────────────────────────────────────────────────

# Default ``--repo`` regex when the user doesn't supply one. Matches
# the probe-cache top-level key (HF: ``<repo>@<sha>``; Ollama: digest
# with name aliases). Conservative: families known to do well at code
# generation. Override via ``--repo`` for ad-hoc sweeps.
#
# Why a hardcoded regex instead of a families-file lookup: the picker
# uses the same approach (a constant set of family roots) and we want
# the pearls runner to be self-contained until it's folded into the
# main bench, at which point the policy decision moves with it.
_DEFAULT_CODING_TIER_RX = (
    r"(?:qwen3-coder|gpt-oss|deepseek|nemotron|"
    r"qwen3[._-]?(?:8b|14b|32b)|"
    r"qwen2\.5[._-]?coder|"
    r"codellama|granite[._-]?code)"
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-target runner
# ─────────────────────────────────────────────────────────────────────────────

def run_for_target(
    target: dict,
    *,
    backend: str,
    router_url: str,
    n_pearls: int,
    log_dir: Path,
    cache: dict,
    cache_path: Path,
    force: bool,
) -> None:
    """Run the pearls task against one model and persist results.

    Single task (no per-task dispatch loop) keeps the function short.
    Skips if the cache row already has a ``pearls_*`` entry, unless
    ``--force``. Errors print and continue -- one bad model shouldn't
    abort a sweep.
    """
    served = serving_alias_with_ctx(target["alias"], target["ctx"], backend)
    key = target["key"]
    existing = cache.get(key) or {}
    existing_tasks = existing.get("tasks") or {}

    if not force and any(t.startswith("pearls") for t in existing_tasks):
        print(
            f"  skip {target['alias']} ({backend}): "
            f"already benched (use --force to re-run)",
            file=sys.stderr,
        )
        return

    print(
        f"\n=== {target['alias']} (backend={backend}, ctx={target['ctx']}) ===",
        file=sys.stderr,
    )
    print(f"  served as: {served}", file=sys.stderr)
    print(f"  router:    {router_url}", file=sys.stderr)

    sampler = VramSampler(interval=1.0)
    sampler.start()
    started = time.time()
    task_results: dict[str, dict] = {}

    try:
        from bench_pearls.tasks.pearls import pearls_task

        print(f"  [pearls]  running n={n_pearls} ...", file=sys.stderr)
        try:
            eval_log = _invoke_inspect_task(
                task_obj=pearls_task(n=n_pearls),
                served_model=served,
                router_url=router_url,
                log_dir=log_dir,
                # Bigger budget than humaneval: pearls problems include
                # quickselect-on-500-elements and intset-replay-of-2000-
                # ops, so the in-test work itself is heavier than
                # HumanEval's micro-fns. 900 s also gives cold-start
                # vLLM headroom on the first sample.
                timeout_s=900.0,
            )
            score, n = _aggregate_score(eval_log)
            by_difficulty = _by_difficulty_breakdown(eval_log)
            by_column = _by_column_breakdown(eval_log)
            task_results[f"pearls_subset_{n}"] = {
                "pass@1": round(score, 4),
                "n": n,
                "by_difficulty": by_difficulty,
                "by_column": by_column,
                "ran_at": _now_iso(),
                "inspect_log_dir": str(log_dir),
            }
            print(f"    pass@1: {score:.4f} (n={n})", file=sys.stderr)
            if by_difficulty:
                print(f"    by_difficulty: {by_difficulty}", file=sys.stderr)
            if by_column:
                print(f"    by_column: {by_column}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 — inspect_ai surfaces many error shapes
            print(f"    !! pearls failed: {e}", file=sys.stderr)

    finally:
        vram = sampler.stop()
        elapsed = time.time() - started
        print(
            f"  done in {elapsed:.1f}s; peak VRAM {vram['peak_vram_gb']} GB",
            file=sys.stderr,
        )

    metrics = {
        "peak_vram_gb": vram["peak_vram_gb"],
        "mean_vram_gb": vram["mean_vram_gb"],
        "vram_samples": vram["n_samples"],
    }
    backend_metrics = _fetch_backend_metrics(backend)
    if backend_metrics:
        metrics.update(backend_metrics)
    update_row(
        cache,
        key,
        model=target["alias"],
        backend=backend,
        router_endpoint=router_url,
        task_results=task_results,
        metrics=metrics,
    )
    save_cache(cache_path, cache)


def _by_difficulty_breakdown(eval_log) -> dict[str, float]:
    """Group per-sample scores by ``metadata.difficulty`` and return
    per-bucket pass@1. Lets the leaderboard report show whether a
    model whiffs only on hard problems vs. across the board.
    """
    return _bucketize(eval_log, "difficulty")


def _by_column_breakdown(eval_log) -> dict[str, float]:
    """Group per-sample scores by Bentley column number. Useful for
    spotting algorithmic-area gaps -- e.g. a model that aces Column 1
    bitset work but flunks Column 14 heaps.
    """
    return _bucketize(eval_log, "column")


def _bucketize(eval_log, metadata_key: str) -> dict[str, float]:
    """Generic per-metadata-key pass@1 grouping. Returns ``{key: rate}``
    rounded to 4 dp. Mirrors ``bench_runner._by_subcase_breakdown`` --
    different field, same shape.
    """
    samples = getattr(eval_log, "samples", None) or []
    buckets: dict[str, list[float]] = {}
    for s in samples:
        meta = getattr(s, "metadata", None) or {}
        bucket = meta.get(metadata_key)
        if bucket is None:
            continue
        scores = getattr(s, "scores", None) or {}
        score_obj = next(
            (cand for cand in scores.values() if cand is not None), None
        )
        if score_obj is None:
            continue
        try:
            v = float(getattr(score_obj, "value", 0.0))
        except (TypeError, ValueError):
            v = 0.0
        buckets.setdefault(str(bucket), []).append(v)
    return {k: round(sum(v) / len(v), 4) for k, v in buckets.items() if v}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", required=True, choices=["ollama", "vllm", "sglang"])
    ap.add_argument(
        "--repo", default="",
        help=(
            "regex filter on probe-cache top-level key. Default: "
            "coding-tier families only "
            "(qwen3-coder, gpt-oss, deepseek, nemotron, Qwen3 8B/14B/32B, "
            "qwen2.5-coder, codellama, granite-code). Pass ``.`` to "
            "match every probed model regardless of family."
        ),
    )
    ap.add_argument("--force", action="store_true",
                    help="re-run the pearls task even when a row exists")
    ap.add_argument("--host-vram-gb", type=int, default=DEFAULT_HOST_VRAM_GB)
    ap.add_argument(
        "--n-pearls", type=int,
        default=int(os.environ.get("BENCH_N_PEARLS", "12")),
        help="number of pearls problems per model (default 12 = full set)",
    )
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_INSPECT_LOG_DIR)
    args = ap.parse_args()

    router_url = router_url_for(args.backend)
    _check_router(router_url)

    repo_filter = args.repo if args.repo else _DEFAULT_CODING_TIER_RX
    targets = discover_models(
        args.backend,
        host_vram_gb=args.host_vram_gb,
        repo_filter=repo_filter,
    )
    if not targets:
        sys.exit(
            f"no fitting {args.backend} models match {repo_filter!r} in "
            f"probe cache at {PROBE_CACHE_BY_BACKEND[args.backend]} "
            f"(host_vram_gb={args.host_vram_gb}). Pass --repo '.' to "
            f"include every model, or adjust the regex."
        )

    print(
        f"bench-pearls: backend={args.backend}, host_vram="
        f"{args.host_vram_gb}G, router={router_url}, "
        f"n_targets={len(targets)}, repo_filter={repo_filter!r}",
        file=sys.stderr,
    )

    cache = load_cache(args.cache)
    n_migrated = migrate_bench_cache_keys(cache)
    if n_migrated:
        print(
            f"bench-pearls: migrated {n_migrated} pre-2026-05-02 cache keys",
            file=sys.stderr,
        )
    for tgt in targets:
        run_for_target(
            tgt,
            backend=args.backend,
            router_url=router_url,
            n_pearls=args.n_pearls,
            log_dir=args.log_dir,
            cache=cache,
            cache_path=args.cache,
            force=args.force,
        )

    print(f"\nbench-pearls: wrote {args.cache}", file=sys.stderr)


if __name__ == "__main__":
    main()
