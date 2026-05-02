#!/usr/bin/env python3
"""Bench runner — drives inspect_ai + the latency/leak sidecar against
every model the probe cache reports as fitting on this host.

Reads the per-backend probe caches (``deploy/.<backend>-reasoning-cache.json``)
to discover which models to bench, runs the requested tasks for each
one through ``http://devai-router:<port>/v1``, and writes results to
``deploy/.bench-cache.json`` keyed by ``<repo@sha>`` (HF) or
``<digest>`` (Ollama) — the same shape the probe caches use, so a
downstream consumer can join rows by key.

Per-model lifecycle:
  1. Start the VRAM sampler (background thread, nvidia-smi every 1s).
  2. For each requested task: invoke inspect_ai (gsm8k/humaneval/tools)
     or call the latency-leak sidecar.
  3. Stop the sampler, capture peak/mean VRAM.
  4. Merge results into the cache row, save.

Skips re-running tasks already present in the cache row unless
``--force`` is set. Per-task ``--n-*`` knobs scale subset sizes.

Errors surface verbatim — no swallowing — so a failing model leaves
a clear trace and doesn't silently pollute the leaderboard.
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

from _probe_core import http_get, load_cache, save_cache  # noqa: E402
from bench import bench_latency_leak  # noqa: E402
from bench._bench_core import (  # noqa: E402
    DEFAULT_CACHE_PATH,
    cache_key_for_entry,
    migrate_bench_cache_keys,
    router_url_for,
    serving_alias,
    serving_alias_with_ctx,
    update_row,
)
from bench.bench_vram_snapshot import VramSampler  # noqa: E402

# --- Constants ---

PROBE_CACHE_BY_BACKEND = {
    "ollama": REPO_ROOT / "deploy" / ".ollama-reasoning-cache.json",
    "vllm": REPO_ROOT / "deploy" / ".vllm-reasoning-cache.json",
    "sglang": REPO_ROOT / "deploy" / ".sglang-reasoning-cache.json",
}
DEFAULT_HOST_VRAM_GB = int(os.environ.get("GPU_MEMORY_GB", "24"))
DEFAULT_INSPECT_LOG_DIR = Path(
    os.environ.get("BENCH_INSPECT_LOG_DIR", "/var/cache/devai/bench/inspect-logs")
)


# --- Model discovery ---

def _has_fitting_cell(entry: dict, host_vram_gb: int) -> tuple[bool, int]:
    """Return ``(any_fits, best_ctx)`` for the host's VRAM band. The
    "fits" verdict varies per backend: Ollama uses ``fully_on_gpu``,
    HF uses ``fits``. We accept either.
    """
    band = (entry.get("probes") or {}).get(str(host_vram_gb)) or {}
    if not isinstance(band, dict):
        return (False, 0)
    best_ctx = 0
    for ctx_str, cell in band.items():
        if not isinstance(cell, dict):
            continue
        ok = bool(cell.get("fully_on_gpu") or cell.get("fits"))
        if not ok:
            continue
        try:
            ctx = int(ctx_str)
        except ValueError:
            continue
        if ctx > best_ctx:
            best_ctx = ctx
    return (best_ctx > 0, best_ctx)


def discover_models(
    backend: str, *, host_vram_gb: int, repo_filter: str | None
) -> list[dict]:
    """Return a list of bench targets from the probe cache.

    Each target is ``{"key": <top-level-key>, "alias": <model-name>,
    "ctx": <best-fitting-ctx>}``. ``alias`` is what gets sent to the
    router as the OpenAI ``model`` field; for HF backends the
    runner appends ``@<ctx>`` so the router recreates with the right
    ``--max-model-len``.
    """
    import re

    rx = re.compile(repo_filter) if repo_filter else None
    cache_path = PROBE_CACHE_BY_BACKEND[backend]
    cache = load_cache(cache_path)
    out: list[dict] = []
    for key, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        cap = entry.get("capability")
        if cap in ("unsupported_arch",):
            continue
        if rx is not None and not rx.search(key):
            continue
        ok, best_ctx = _has_fitting_cell(entry, host_vram_gb)
        if not ok:
            continue
        alias = serving_alias(entry)
        cache_key = cache_key_for_entry(entry, backend) or key
        if not alias:
            continue
        out.append({"key": cache_key, "alias": alias, "ctx": best_ctx, "entry": entry})
    return out


# --- inspect_ai task dispatch ---

def _invoke_inspect_task(
    *,
    task_obj,
    served_model: str,
    router_url: str,
    log_dir: Path,
    timeout_s: float,
    fail_on_error: bool | None = None,
):
    """Run an inspect_ai Task against the router. Returns the EvalLog.

    ``fail_on_error=False`` lets a single failing sample (e.g. a router
    400 against a forced-mode model on a multi-tool request) be recorded
    as a failed sample rather than aborting the whole task. Default
    None preserves inspect_ai's stricter built-in behaviour for tasks
    where any error indicates a real bug.
    """
    from inspect_ai import eval as inspect_eval

    log_dir.mkdir(parents=True, exist_ok=True)
    # The router's vLLM/SGLang ports speak vanilla OpenAI; Ollama too
    # via /v1. Auth doesn't matter — router is internal — but the SDK
    # complains if API key is empty, so set a placeholder.
    os.environ.setdefault("OPENAI_API_KEY", "devai-router-no-auth")
    eval_kwargs = dict(
        model=f"openai/{served_model}",
        model_base_url=router_url + "/v1",
        log_dir=str(log_dir),
        # message_limit caps the assistant <-> tool turn-loop length
        # (relevant for tools_use; conservative cap keeps a misbehaving
        # model from running forever).
        message_limit=20,
        # time_limit is per-sample wall clock. Generous because cold-
        # start vLLM can need 90+ seconds on first request and the
        # sample-level timeout fires AFTER the model is loaded.
        time_limit=int(timeout_s),
    )
    if fail_on_error is not None:
        eval_kwargs["fail_on_error"] = fail_on_error
    logs = inspect_eval(task_obj, **eval_kwargs)
    return logs[0] if isinstance(logs, list) else logs


def _aggregate_score(eval_log) -> tuple[float, int]:
    """Pull the headline accuracy from an EvalLog and the sample count.
    inspect_ai stores results under ``log.results.scores[*].metrics``.
    """
    results = getattr(eval_log, "results", None)
    n = 0
    samples = getattr(eval_log, "samples", None) or []
    n = len(samples)
    if results is None or not getattr(results, "scores", None):
        return (0.0, n)
    s = results.scores[0]
    metrics = getattr(s, "metrics", None) or {}
    acc = metrics.get("accuracy")
    if acc is None:
        return (0.0, n)
    val = getattr(acc, "value", acc)
    try:
        return (float(val), n)
    except (TypeError, ValueError):
        return (0.0, n)


def _by_subcase_breakdown(eval_log) -> dict[str, float]:
    """For ``tools_use`` only. Walk per-sample scores, group by
    metadata.subcase, return per-subcase accuracy.

    Reads ``sample.scores`` (plural dict, the current API). Each
    sample may carry multiple named scorers; we take the first
    Score whose value is numeric. The deprecated ``sample.score``
    singular field is intentionally not consulted.
    """
    samples = getattr(eval_log, "samples", None) or []
    buckets: dict[str, list[float]] = {}
    for s in samples:
        meta = getattr(s, "metadata", None) or {}
        subcase = meta.get("subcase")
        if not subcase:
            continue
        scores = getattr(s, "scores", None) or {}
        score_obj = None
        for cand in scores.values():
            if cand is not None:
                score_obj = cand
                break
        if score_obj is None:
            continue
        try:
            v = float(getattr(score_obj, "value", 0.0))
        except (TypeError, ValueError):
            v = 0.0
        buckets.setdefault(subcase, []).append(v)
    return {k: round(sum(v) / len(v), 4) for k, v in buckets.items() if v}


# --- Main loop ---

def run_for_target(
    target: dict,
    *,
    backend: str,
    router_url: str,
    tasks: list[str],
    n_gsm8k: int,
    n_humaneval: int,
    n_tools: int,
    n_leak_prompts: int,
    log_dir: Path,
    cache: dict,
    cache_path: Path,
    force: bool,
) -> None:
    """Run all requested tasks against one model and persist results."""
    served = serving_alias_with_ctx(target["alias"], target["ctx"], backend)
    key = target["key"]
    existing = cache.get(key) or {}
    existing_tasks = (existing.get("tasks") or {})

    print(f"\n=== {target['alias']} (backend={backend}, ctx={target['ctx']}) ===",
          file=sys.stderr)
    print(f"  served as: {served}", file=sys.stderr)
    print(f"  router:    {router_url}", file=sys.stderr)

    sampler = VramSampler(interval=1.0)
    sampler.start()
    started = time.time()
    task_results: dict[str, dict] = {}

    try:
        if "leak" in tasks and (force or "leak_probe" not in existing_tasks):
            print(f"  [leak]    streaming {n_leak_prompts} prompts...", file=sys.stderr)
            try:
                latency = bench_latency_leak.run(
                    model=served,
                    router_url=router_url,
                    n=n_leak_prompts,
                )
                task_results["leak_probe"] = {
                    "leak_rate": latency["leak_rate"],
                    "leaked_markers": latency["leaked_markers"],
                    "n_prompts": latency["n_samples"] + latency["n_errors"],
                    "n_errors": latency["n_errors"],
                    "ran_at": _now_iso(),
                }
                # Latency metrics live under "metrics", not "tasks".
                _latency_metrics_into_row(cache, key, latency, target, backend, router_url)
                _print_latency_summary(latency)
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
                print(f"    !! leak/latency failed: {e}", file=sys.stderr)

        if "gsm8k" in tasks and (force or "gsm8k" not in [_strip_subset(t) for t in existing_tasks]):
            from bench.tasks.gsm8k import gsm8k_task
            print(f"  [gsm8k]   running n={n_gsm8k} ...", file=sys.stderr)
            try:
                eval_log = _invoke_inspect_task(
                    task_obj=gsm8k_task(n=n_gsm8k),
                    served_model=served,
                    router_url=router_url,
                    log_dir=log_dir,
                    timeout_s=600.0,
                )
                score, n = _aggregate_score(eval_log)
                task_results[f"gsm8k_subset_{n}"] = {
                    "score": round(score, 4),
                    "n": n,
                    "ran_at": _now_iso(),
                    "inspect_log_dir": str(log_dir),
                }
                print(f"    score: {score:.4f} (n={n})", file=sys.stderr)
            except Exception as e:  # noqa: BLE001 — inspect_ai surfaces many error shapes
                print(f"    !! gsm8k failed: {e}", file=sys.stderr)

        if "humaneval" in tasks and (force or "humaneval" not in [_strip_subset(t) for t in existing_tasks]):
            from bench.tasks.humaneval import humaneval_task
            print(f"  [humaneval] running n={n_humaneval} ...", file=sys.stderr)
            try:
                eval_log = _invoke_inspect_task(
                    task_obj=humaneval_task(n=n_humaneval),
                    served_model=served,
                    router_url=router_url,
                    log_dir=log_dir,
                    timeout_s=900.0,
                )
                score, n = _aggregate_score(eval_log)
                task_results[f"humaneval_subset_{n}"] = {
                    "pass@1": round(score, 4),
                    "n": n,
                    "ran_at": _now_iso(),
                    "inspect_log_dir": str(log_dir),
                }
                print(f"    pass@1: {score:.4f} (n={n})", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"    !! humaneval failed: {e}", file=sys.stderr)

        if "tools" in tasks and (force or "tools" not in [_strip_subset(t) for t in existing_tasks]):
            from bench.tasks.tools_use import tools_use_task
            print(f"  [tools]   running n={n_tools} ...", file=sys.stderr)
            try:
                eval_log = _invoke_inspect_task(
                    task_obj=tools_use_task(n=n_tools),
                    served_model=served,
                    router_url=router_url,
                    log_dir=log_dir,
                    timeout_s=600.0,
                    # Forced-mode models historically tripped the router's
                    # tool_choice_pinning_required check on individual
                    # samples; per-sample pinning fixes the request shape,
                    # but keep fail_on_error=False as belt-and-suspenders
                    # so a single anomalous sample doesn't void the run.
                    fail_on_error=False,
                )
                score, n = _aggregate_score(eval_log)
                by_sub = _by_subcase_breakdown(eval_log)
                task_results[f"tools_use_{n}"] = {
                    "score": round(score, 4),
                    "n": n,
                    "by_subcase": by_sub,
                    "ran_at": _now_iso(),
                    "inspect_log_dir": str(log_dir),
                }
                print(f"    score: {score:.4f} (n={n})", file=sys.stderr)
                if by_sub:
                    print(f"    by_subcase: {by_sub}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"    !! tools_use failed: {e}", file=sys.stderr)

    finally:
        vram = sampler.stop()
        elapsed = time.time() - started
        print(f"  done in {elapsed:.1f}s; peak VRAM {vram['peak_vram_gb']} GB",
              file=sys.stderr)

    metrics = {
        "peak_vram_gb": vram["peak_vram_gb"],
        "mean_vram_gb": vram["mean_vram_gb"],
        "vram_samples": vram["n_samples"],
    }
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


def _latency_metrics_into_row(
    cache: dict, key: str, latency: dict,
    target: dict, backend: str, router_url: str,
) -> None:
    """Latency metrics belong on row.metrics, not row.tasks. Pulled
    out so the leak-task branch can write both shapes from one
    sidecar invocation.
    """
    update_row(
        cache, key,
        model=target["alias"], backend=backend, router_endpoint=router_url,
        metrics={
            "ttft_ms_first": latency.get("ttft_ms_first"),
            "ttft_ms_steady_p50": latency.get("ttft_ms_steady_p50"),
            "ttft_ms_steady_p95": latency.get("ttft_ms_steady_p95"),
            "tps_sustained_p50": latency.get("tps_sustained_p50"),
            "n_latency_samples": latency.get("n_samples"),
        },
    )


def _print_latency_summary(latency: dict) -> None:
    f = latency.get("ttft_ms_first")
    p50 = latency.get("ttft_ms_steady_p50")
    p95 = latency.get("ttft_ms_steady_p95")
    tps = latency.get("tps_sustained_p50")
    nleak = sum(latency.get("leaked_markers", {}).values())
    print(
        f"    ttft_first={f}ms  steady_p50={p50}ms  steady_p95={p95}ms  "
        f"tps={tps}/s  leaks={nleak}",
        file=sys.stderr,
    )


def _strip_subset(task_name: str) -> str:
    """Map cache task keys back to user-visible task names.

    ``gsm8k_subset_100`` → ``gsm8k``; ``humaneval_subset_50`` →
    ``humaneval``; ``tools_use_20`` → ``tools``; ``leak_probe`` →
    ``leak``. Used to skip already-cached tasks without forcing the
    same n.
    """
    if task_name.startswith("gsm8k_"):
        return "gsm8k"
    if task_name.startswith("humaneval_"):
        return "humaneval"
    if task_name.startswith("tools_use"):
        return "tools"
    if task_name == "leak_probe":
        return "leak"
    return task_name


def _now_iso() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


# --- CLI ---

def _check_router(router_url: str) -> None:
    """Hit ``/health`` so a bad route fails before the first model
    runs (saves the cold-start startup time on a bogus endpoint).
    """
    try:
        http_get(router_url + "/health", timeout=5.0)
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        # /health may return non-JSON ("OK"); we just want to confirm
        # the port answers. urlopen raising URLError is the real
        # failure case.
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", required=True, choices=["ollama", "vllm", "sglang"])
    ap.add_argument(
        "--tasks", default="gsm8k,humaneval,tools,leak",
        help="comma-separated subset",
    )
    ap.add_argument("--repo", default="", help="regex filter on probe-cache top-level key")
    ap.add_argument("--force", action="store_true", help="re-run tasks even if cached")
    ap.add_argument("--host-vram-gb", type=int, default=DEFAULT_HOST_VRAM_GB)
    ap.add_argument("--n-gsm8k", type=int, default=int(os.environ.get("BENCH_N_GSM8K", "100")))
    ap.add_argument("--n-humaneval", type=int, default=int(os.environ.get("BENCH_N_HUMANEVAL", "50")))
    ap.add_argument("--n-tools", type=int, default=int(os.environ.get("BENCH_N_TOOLS", "20")))
    ap.add_argument("--n-leak-prompts", type=int,
                    default=int(os.environ.get("BENCH_N_LEAK_PROMPTS", "40")))
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_INSPECT_LOG_DIR)
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    invalid = [t for t in tasks if t not in {"gsm8k", "humaneval", "tools", "leak"}]
    if invalid:
        sys.exit(f"unknown task(s): {invalid}")

    router_url = router_url_for(args.backend)
    _check_router(router_url)

    repo_filter = args.repo or None
    targets = discover_models(
        args.backend,
        host_vram_gb=args.host_vram_gb,
        repo_filter=repo_filter,
    )
    if not targets:
        sys.exit(
            f"no fitting {args.backend} models in probe cache at "
            f"{PROBE_CACHE_BY_BACKEND[args.backend]} (host_vram_gb="
            f"{args.host_vram_gb}, repo={repo_filter!r})"
        )

    print(
        f"bench: backend={args.backend}, host_vram={args.host_vram_gb}G, "
        f"router={router_url}, tasks={tasks}, n_targets={len(targets)}",
        file=sys.stderr,
    )

    cache = load_cache(args.cache)
    n_migrated = migrate_bench_cache_keys(cache)
    if n_migrated:
        print(
            f"bench: migrated {n_migrated} pre-2026-05-02 cache keys to "
            f"<repo>@<sha>::<backend> form",
            file=sys.stderr,
        )
    for tgt in targets:
        run_for_target(
            tgt,
            backend=args.backend,
            router_url=router_url,
            tasks=tasks,
            n_gsm8k=args.n_gsm8k,
            n_humaneval=args.n_humaneval,
            n_tools=args.n_tools,
            n_leak_prompts=args.n_leak_prompts,
            log_dir=args.log_dir,
            cache=cache,
            cache_path=args.cache,
            force=args.force,
        )

    print(f"\nbench: wrote {args.cache}", file=sys.stderr)


if __name__ == "__main__":
    main()
