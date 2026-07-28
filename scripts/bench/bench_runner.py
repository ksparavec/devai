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
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _capability import Capability  # noqa: E402
from _probe_core import http_get, load_cache, save_cache  # noqa: E402
from bench import bench_latency_leak  # noqa: E402
from bench._bench_core import (  # noqa: E402
    DEFAULT_CACHE_PATH,
    assert_cache_schema_compatible,
    cache_key_for_entry,
    capture_host_env,
    migrate_bench_cache_keys,
    router_url_for,
    serving_alias,
    serving_alias_with_ctx,
    stamp_host_env,
    update_row,
)
from bench.bench_vram_snapshot import VramSampler  # noqa: E402

# --- Constants ---

PROBE_CACHE_BY_BACKEND = {
    "ollama": REPO_ROOT / "deploy" / ".ollama-reasoning-cache.json",
    "vllm": REPO_ROOT / "deploy" / ".vllm-reasoning-cache.json",
    "sglang": REPO_ROOT / "deploy" / ".sglang-reasoning-cache.json",
}

def probe_image_digest(backend: str) -> str | None:
    """The backend engine image digest the probe cache was measured under.

    Read from the probe cache's `_meta.current_image_digest`, which the
    prober stamps (see `_probe_core.stamp_image_digest`). `make bench`
    runs inside a container with no access to the host container runtime,
    so inspecting the image live is not an option -- and the probe cache
    is the better source regardless: if the engine image moved, the fit
    data that selected these bench targets is equally stale.

    Returns None for Ollama (its cache carries no image block) and on any
    read error. Callers stamp nothing in that case, and bench-sync
    classifies an unstamped row `unknown` rather than guessing.
    """
    path = PROBE_CACHE_BY_BACKEND.get(backend)
    if path is None:
        return None
    try:
        cache = load_cache(path)
    except Exception:
        return None
    meta = cache.get("_meta")
    if not isinstance(meta, dict):
        return None
    digest = meta.get("current_image_digest")
    return digest if isinstance(digest, str) and digest else None


# Weight stores for the HF backends, checked before a model is benched.
# Ollama is absent on purpose: its weights live in a blob store keyed by
# digest, not a directory named after the model, so there is no
# equivalent path to stat.
#
# The probe cache advertising a model is NOT evidence the weights are
# present. The two SGLang/vLLM stores are separate volumes, and
# `make model-pull` only ever writes the vLLM one, so an SGLang-probed
# model routinely has a fits=true cell and no weights at all. Benching
# such a model cannot work: the router has nothing to launch, and the
# whole task sweep records tps=0.0 / ttft=None. Observed 2026-07-25 --
# `make bench-sglang` started working through six absent models before it
# was stopped by hand. select-models.py has carried this same check as
# `sglang_weight_gaps` all along; the bench runner simply never used it.
HF_WEIGHT_STORE_BY_BACKEND = {
    "vllm": Path(os.environ.get("VLLM_MODELS_DIR", "/var/cache/devai/vllm")),
    "sglang": Path(os.environ.get("SGLANG_MODELS_DIR", "/var/cache/devai/sglang")),
}


def weights_present(backend: str, alias: str) -> bool:
    """True when `alias` has loadable weights in `backend`'s store.

    Fails OPEN for Ollama (no directory to check) and when the store
    itself is not visible -- the mount is `wildcard`-guarded in the
    Makefile, and a missing mount must not silently skip every model.
    """
    store = HF_WEIGHT_STORE_BY_BACKEND.get(backend)
    if store is None or not store.is_dir():
        return True
    return (store / alias / "config.json").is_file()
DEFAULT_HOST_VRAM_GB = int(os.environ.get("GPU_MEMORY_GB", "24"))
DEFAULT_INSPECT_LOG_DIR = Path(
    os.environ.get("BENCH_INSPECT_LOG_DIR", "/var/cache/devai/bench/inspect-logs")
)
# Tasks a plain `make bench` runs. Must cover every benchmark column the
# picker renders: gpqa is its DEFAULT SORT column, and humaneval_plus /
# mmlu_pro / gpqa used to be opt-in, so a default run left four of the
# seven columns permanently blank. longctx stays out -- it is a
# per-context probe, not a leaderboard score.
DEFAULT_TASKS = "gsm8k,humaneval,humaneval_plus,mmlu_pro,gpqa,tools,leak"

# Backend Prometheus /metrics endpoints reachable from the bench
# runner's network (devai-net). vLLM exposes a Prometheus exporter on
# its internal serving port; SGLang too. Ollama does not expose
# Prometheus metrics (None -> skip).
BACKEND_METRICS_URL = {
    "vllm": os.environ.get(
        "BENCH_VLLM_METRICS_URL", "http://devai-vllm:11434/metrics"
    ),
    "sglang": os.environ.get(
        "BENCH_SGLANG_METRICS_URL", "http://devai-sglang:11434/metrics"
    ),
    "ollama": None,
}

# vLLM Prometheus metrics we capture per run. The names are vLLM-side
# canonical (prefix ``vllm:``); we strip the prefix and store under
# ``vllm_<short>`` keys in the bench cache row's metrics block. SGLang's
# Prometheus exporter uses different metric names; we also try a small
# matching set there. Backends that don't expose either are skipped.
_VLLM_METRIC_PATTERNS = (
    # gauge in [0.0, 1.0]: KV-cache utilization. End-of-run value is
    # not a peak -- after the last sample drains the queue, usage falls.
    # We document the limitation in docs/bench-results.md and leave
    # max-during-run for a follow-up. Note: the current vLLM image
    # exposes this as ``vllm:kv_cache_usage_perc``; older docs sometimes
    # call it ``gpu_cache_usage_perc`` (renamed upstream).
    "vllm:kv_cache_usage_perc",
    # counter: cumulative preemptions over the run. Container is
    # recreated per (model, ctx) for HF backends, so this resets per
    # row -- absolute end-of-run value is the run total.
    "vllm:num_preemptions_total",
)
_SGLANG_METRIC_PATTERNS = (
    # SGLang exposes radix-cache hit rate and request queue depth on
    # similar metric names. Kept here so the same code path covers
    # both backends; if the exporter is silent the metric just doesn't
    # appear and the row's metrics block is unaffected.
    "sglang:cache_hit_rate",
    "sglang:num_running_reqs",
    "sglang:num_used_tokens",
)


def _parse_prometheus_max(text: str, metric_name: str) -> float | None:
    """Parse Prometheus text for the maximum value of ``metric_name``.

    A metric can have several rows with different label sets, e.g.::

        vllm:gpu_cache_usage_perc{model_name="..."} 0.42
        vllm:gpu_cache_usage_perc{model_name="other"} 0.01

    We take the max so a row with multiple label combinations doesn't
    silently lose the meaningful value. Lines starting with ``#`` are
    HELP / TYPE comments and are skipped.
    """
    best: float | None = None
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if not line.startswith(metric_name):
            continue
        # The metric name must end here; a trailing brace or space.
        nxt = line[len(metric_name) : len(metric_name) + 1]
        if nxt not in ("{", " "):
            continue
        try:
            v = float(line.rsplit(" ", 1)[-1])
        except (ValueError, IndexError):
            continue
        if best is None or v > best:
            best = v
    return best


def _fetch_backend_metrics(backend: str) -> dict[str, float]:
    """End-of-run snapshot of the backend's Prometheus /metrics endpoint.

    Returns a flat dict suitable for merging into a row's
    ``metrics`` block. Best-effort: any transport, decode, or parse
    failure returns an empty dict (the bench result is still valid
    without these). Keys are flattened by replacing ``:`` with ``_``
    so downstream JSON consumers don't trip on the colon.
    """
    url = BACKEND_METRICS_URL.get(backend)
    if not url:
        return {}
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError):
        return {}
    patterns = _VLLM_METRIC_PATTERNS if backend == "vllm" else _SGLANG_METRIC_PATTERNS
    out: dict[str, float] = {}
    for full_name in patterns:
        v = _parse_prometheus_max(text, full_name)
        if v is not None:
            out[full_name.replace(":", "_")] = v
    return out


# --- Model discovery ---

def _fitting_ctxs(entry: dict, host_vram_gb: int) -> list[int]:
    """Return the sorted list of ctx tiers (ints) where ``entry`` fits at
    ``host_vram_gb``. The "fits" verdict varies per backend: Ollama uses
    ``fully_on_gpu``, HF uses ``fits``. We accept either.

    Empty list when nothing fits. Sorted ascending; callers that want
    "largest fitting" take ``[-1]``.
    """
    band = (entry.get("probes") or {}).get(str(host_vram_gb)) or {}
    if not isinstance(band, dict):
        return []
    out: list[int] = []
    for ctx_str, cell in band.items():
        if not isinstance(cell, dict):
            continue
        # serving_ok=false (set by `make probe-load-*`) means the cell
        # loaded but OOMed under a near-full-context request -- benching
        # there would just reproduce the OOM. Skip it; serving_ok absent
        # (load probe never ran) keeps the fit-only verdict.
        ok = (bool(cell.get("fully_on_gpu") or cell.get("fits"))
              and cell.get("serving_ok") is not False)
        if not ok:
            continue
        try:
            ctx = int(ctx_str)
        except ValueError:
            continue
        if ctx > 0:
            out.append(ctx)
    return sorted(out)


def discover_models(
    backend: str,
    *,
    host_vram_gb: int,
    repo_filter: str | None,
    ctx_filter: list[int] | None = None,
) -> list[dict]:
    """Return a list of bench targets from the probe cache.

    Each target is ``{"key": <top-level-key>, "alias": <model-name>,
    "ctx": <ctx-int>, "entry": <probe-row>}``. ``alias`` is what gets
    sent to the router as the OpenAI ``model`` field; for HF backends
    the runner appends ``@<ctx>`` so the router recreates with the
    right ``--max-model-len``.

    ``ctx_filter`` semantics (per docs/plans/bench-rewrite.md):

    - ``None`` (default): one target per (model, largest-fitting-ctx).
      Same wall-clock cost as the pre-v3 runner.
    - explicit list (e.g. ``[32768, 131072]``): one target per
      (model, ctx) for each ctx that's both in the filter AND a
      fits=true cell. Missing-cell pairs are skipped with a stderr
      note.
    - explicit list spelled ``[-1]``: shorthand for "all fitting ctxs"
      -- emits one target per (model, ctx) for every fits=true cell.

    The bench-cache key is minted with ``cache_key_for_entry(entry,
    backend, ctx)`` so each ctx tier writes to its own row.
    """
    import re

    rx = re.compile(repo_filter) if repo_filter else None
    cache_path = PROBE_CACHE_BY_BACKEND[backend]
    cache = load_cache(cache_path)
    out: list[dict] = []
    all_ctxs = ctx_filter == [-1]
    for key, entry in cache.items():
        if not isinstance(entry, dict):
            continue
        cap = entry.get("capability")
        if cap == Capability.UNSUPPORTED_ARCH:
            continue
        if rx is not None and not rx.search(key):
            continue
        fitting = _fitting_ctxs(entry, host_vram_gb)
        if not fitting:
            continue
        if ctx_filter is None:
            chosen = [fitting[-1]]
        elif all_ctxs:
            chosen = list(fitting)
        else:
            chosen = [c for c in ctx_filter if c in fitting]
            missing = [c for c in ctx_filter if c not in fitting]
            if missing:
                print(
                    f"  note: {key}: skipping ctx(s) {missing} -- no "
                    f"fits=true cell at host_vram={host_vram_gb}G "
                    f"(fitting tiers: {fitting})",
                    file=sys.stderr,
                )
            if not chosen:
                continue
        alias = serving_alias(entry)
        if not alias:
            continue
        if not weights_present(backend, alias):
            print(
                f"  note: {alias}: skipping -- probe cache says it fits on "
                f"{backend} but no weights under "
                f"{HF_WEIGHT_STORE_BY_BACKEND[backend]}/{alias}. "
                f"Download with: python3 scripts/select-models.py --name "
                f"{alias} --download --hf-store {backend}",
                file=sys.stderr,
            )
            continue
        for ctx in chosen:
            cache_key = cache_key_for_entry(entry, backend, ctx) or key
            out.append(
                {
                    "key": cache_key,
                    "alias": alias,
                    "ctx": int(ctx),
                    "entry": entry,
                }
            )
    return out


# --- CLI ctx parsing helpers ---

_CTX_TIER_ALIASES: dict[str, int] = {
    "32K": 32768,
    "64K": 65536,
    "128K": 131072,
    "256K": 262144,
}


def _parse_ctx_token(token: str) -> int:
    """``32K`` -> 32768; ``128k`` -> 131072; integer string -> int.

    Raises ValueError on malformed input so the CLI parser surfaces
    a clear error instead of silently dropping the tier.
    """
    s = token.strip().upper()
    if not s:
        raise ValueError("empty ctx token")
    if s in _CTX_TIER_ALIASES:
        return _CTX_TIER_ALIASES[s]
    if s.endswith("K"):
        head = s[:-1]
        return int(head) * 1024
    return int(s)


def parse_ctx_list(raw: str) -> list[int]:
    """Comma-separated list of ctx tiers. Empty input -> empty list."""
    if not raw:
        return []
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(_parse_ctx_token(part))
    return out


# --- inspect_ai task dispatch ---

# Sampling for the scored inspect_ai tasks (gsm8k, humaneval,
# humaneval_plus, mmlu_pro, gpqa, tools).
#
# These used to pass NO sampling parameters at all, which meant every
# scored task ran at whatever the serving backend happened to default to
# -- and vLLM, SGLang and Ollama do not default the same. Two models
# benched on different backends were therefore not being compared under
# the same conditions, which is the one thing a leaderboard has to get
# right.
#
# Zero is not a new policy here: the two sidecar tasks that talk HTTP
# directly already pin it (bench_longctx.py, bench_latency_leak.py), and
# docs/sampling-strategies.md prescribes greedy decoding for
# deterministic eval. This makes the inspect_ai path consistent with the
# rest of the harness rather than introducing a rule.
#
# Override with BENCH_TEMPERATURE / BENCH_TOP_P when deliberately
# measuring a model at its card-recommended sampling (some reasoning
# models recommend temperature > 0 and can degrade into repetition at
# greedy). Doing so makes those rows non-comparable with the rest of the
# table, so record why.
#
# NOTE: rows benched BEFORE this change carry unknown sampling. They are
# not comparable with rows benched after it; re-bench to compare.
BENCH_TEMPERATURE = float(os.environ.get("BENCH_TEMPERATURE", "0"))
BENCH_TOP_P = float(os.environ.get("BENCH_TOP_P", "1.0"))


BENCH_SAMPLING_PATH = REPO_ROOT / "deploy" / "bench-sampling.json"


def load_sampling_overrides(path: Path | None = None) -> dict:
    """Per-model sampling overrides. Missing/malformed file -> {}."""
    p = path or BENCH_SAMPLING_PATH
    try:
        with open(p) as f:
            return (json.load(f) or {}).get("models") or {}
    except (OSError, ValueError):
        return {}


def sampling_for(alias: str, overrides: dict | None = None) -> tuple[float, float]:
    """(temperature, top_p) for `alias`.

    Greedy by default. A handful of models cannot be benched greedily --
    NVIDIA-Nemotron-Nano-9B-v2 loops on its own <think> trace at
    temperature 0 and never emits an answer, which scores ~0 on every
    task and reads as a capability failure rather than the sampling
    artifact it is. deploy/bench-sampling.json carries those exceptions.
    """
    ov = load_sampling_overrides() if overrides is None else overrides
    for name, cfg in ov.items():
        if name in alias:
            return (
                float(cfg.get("temperature", BENCH_TEMPERATURE)),
                float(cfg.get("top_p", BENCH_TOP_P)),
            )
    return BENCH_TEMPERATURE, BENCH_TOP_P


def sampling_record(alias: str, overrides: dict | None = None) -> dict:
    """What sampling this row was ACTUALLY benched at, for the cache.

    Zero of 21 existing rows record this, which makes a mixed cache
    unauditable: greedy rows and override rows sit side by side in the
    leaderboard with nothing to say they are not comparable. At least one
    model already has an override (NVIDIA-Nemotron-Nano-9B-v2 loops on its
    own <think> trace at temperature 0), so "mixed" is the real state, not
    a hypothetical.

    `source` is "override" when deploy/bench-sampling.json matched this
    alias, else "greedy_default".
    """
    ov = load_sampling_overrides() if overrides is None else overrides
    temperature, top_p = sampling_for(alias, ov)
    matched = any(name in alias for name in ov)
    return {
        "temperature": temperature,
        "top_p": top_p,
        "source": "override" if matched else "greedy_default",
        "comparable": not matched,
    }


def _sampling_config(alias: str = ""):
    """GenerateConfig pinning sampling for the scored tasks.

    Imported lazily, like inspect_eval itself, so the module stays
    importable without inspect_ai installed (the unit tests rely on
    that).
    """
    from inspect_ai.model import GenerateConfig

    temperature, top_p = sampling_for(alias)
    return GenerateConfig(temperature=temperature, top_p=top_p)


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
        # Explicit sampling. Without this the backend's own default
        # applies and differs per engine -- see BENCH_TEMPERATURE above.
        # Keyed on the served model so deploy/bench-sampling.json can
        # exempt the models that cannot be benched greedily.
        config=_sampling_config(served_model),
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

# Metrics that gate an early drop. Tools is intentionally excluded: it is a
# saturated 20-sample microbench and a low/zero score there usually reflects a
# parser-curation gap (a missing tool-call parser), not model quality.
_DROP_SCORE_METRICS = (
    ("gsm8k_subset_", "score", "gsm8k"),
    ("humaneval_subset_", "pass@1", "humaneval"),
)


def _evaluate_drop_trigger(task_results: dict, threshold: float) -> dict | None:
    """Return a drop-recommendation dict when the results so far disqualify
    the model, else None.

    Triggers: any leak (leak_rate > 0), or gsm8k / humaneval below
    ``threshold``. Safe to call incrementally after each task -- it only
    inspects the metrics currently present in ``task_results``.
    """
    leak = task_results.get("leak_probe")
    if leak is not None:
        rate = leak.get("leak_rate") or 0
        if rate > 0:
            return {
                "reason": "leak",
                "metric": "leak_rate",
                "value": rate,
                "threshold": 0,
                "detail": f"leak_rate={rate} > 0",
            }
    for prefix, field, label in _DROP_SCORE_METRICS:
        for tname, tresult in task_results.items():
            if tname.startswith(prefix) and isinstance(tresult, dict) and field in tresult:
                val = tresult[field]
                if isinstance(val, (int, float)) and val < threshold:
                    return {
                        "reason": "low_score",
                        "metric": label,
                        "value": val,
                        "threshold": threshold,
                        "detail": f"{label}={val} < {threshold}",
                    }
    return None


def run_for_target(
    target: dict,
    *,
    backend: str,
    router_url: str,
    tasks: list[str],
    n_gsm8k: int,
    n_humaneval: int,
    n_tools: int,
    n_mmlu_pro: int,
    n_gpqa: int,
    n_leak_prompts: int,
    n_longctx_fraction: float,
    n_longctx_max_tokens: int,
    log_dir: Path,
    cache: dict,
    cache_path: Path,
    force: bool,
    host_env_id: str | None = None,
    backend_image_digest: str | None = None,
    drop_threshold: float = 0.70,
    early_drop: bool = True,
) -> None:
    """Run all requested tasks against one model and persist results.

    Cache data is immutable: a task result, once written, is never deleted
    or modified -- with exactly one exception, ``--force``, which re-runs a
    task and overwrites its entry ONLY when the new run succeeds. A failed
    task writes nothing (the prior value, if any, stands). Without --force,
    a task that already has a result is skipped entirely. So `--force` never
    wipes the row upfront: unrelated tasks (e.g. the sharper benches) survive
    a forced re-run of the default tasks.
    """
    served = serving_alias_with_ctx(target["alias"], target["ctx"])
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
    # Early-drop: once a disqualifier trips (leak, or gsm8k/humaneval below
    # drop_threshold), skip this model's remaining tasks and flag it for drop.
    # The flag is recorded on the row; it never deletes weights or edits the
    # exclusion ledger -- that stays an explicit operator action.
    drop_flag: dict | None = None

    def _check_drop() -> None:
        nonlocal drop_flag
        if not early_drop or drop_flag is not None:
            return
        flag = _evaluate_drop_trigger(task_results, drop_threshold)
        if flag:
            flag["ran_at"] = _now_iso()
            drop_flag = flag
            print(
                f"  !! DROP-FLAG: {flag['detail']} -- skipping remaining "
                f"tasks for {target['alias']}",
                file=sys.stderr,
            )

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
                _latency_metrics_into_row(
                    cache, key, latency, target, backend, router_url,
                    host_env_id=host_env_id,
                    backend_image_digest=backend_image_digest,
                )
                _print_latency_summary(latency)
            except Exception as e:  # noqa: BLE001
                print(f"    !! leak/latency failed: {e}", file=sys.stderr)
        _check_drop()

        if drop_flag is None and "gsm8k" in tasks and (force or "gsm8k" not in [_strip_subset(t) for t in existing_tasks]):
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
                # Immutable-on-failure: write nothing. A failed task leaves the
                # cache untouched (prior value, if any, stands; absent stays
                # absent so a later run retries it).
        _check_drop()

        if drop_flag is None and "humaneval" in tasks and (force or "humaneval" not in [_strip_subset(t) for t in existing_tasks]):
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
        _check_drop()

        if drop_flag is None and "humaneval_plus" in tasks and (force or "humaneval_plus" not in [_strip_subset(t) for t in existing_tasks]):
            from bench.tasks.humaneval_plus import humaneval_plus_task
            print(f"  [humaneval+] running n={n_humaneval} ...", file=sys.stderr)
            try:
                eval_log = _invoke_inspect_task(
                    task_obj=humaneval_plus_task(n=n_humaneval),
                    served_model=served,
                    router_url=router_url,
                    log_dir=log_dir,
                    timeout_s=900.0,
                )
                score, n = _aggregate_score(eval_log)
                task_results[f"humaneval_plus_subset_{n}"] = {
                    "pass@1": round(score, 4),
                    "n": n,
                    "ran_at": _now_iso(),
                    "inspect_log_dir": str(log_dir),
                }
                print(f"    pass@1: {score:.4f} (n={n})", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"    !! humaneval_plus failed: {e}", file=sys.stderr)

        if drop_flag is None and "mmlu_pro" in tasks and (force or "mmlu_pro" not in [_strip_subset(t) for t in existing_tasks]):
            from bench.tasks.mmlu_pro import mmlu_pro_task
            print(f"  [mmlu_pro] running n={n_mmlu_pro} ...", file=sys.stderr)
            try:
                eval_log = _invoke_inspect_task(
                    task_obj=mmlu_pro_task(n=n_mmlu_pro),
                    served_model=served,
                    router_url=router_url,
                    log_dir=log_dir,
                    timeout_s=900.0,
                )
                score, n = _aggregate_score(eval_log)
                task_results[f"mmlu_pro_subset_{n}"] = {
                    "score": round(score, 4),
                    "n": n,
                    "ran_at": _now_iso(),
                    "inspect_log_dir": str(log_dir),
                }
                print(f"    score: {score:.4f} (n={n})", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"    !! mmlu_pro failed: {e}", file=sys.stderr)

        if drop_flag is None and "gpqa" in tasks and (force or "gpqa" not in [_strip_subset(t) for t in existing_tasks]):
            from bench.tasks.gpqa import gpqa_task
            print(f"  [gpqa]    running n={n_gpqa} ...", file=sys.stderr)
            try:
                eval_log = _invoke_inspect_task(
                    task_obj=gpqa_task(n=n_gpqa),
                    served_model=served,
                    router_url=router_url,
                    log_dir=log_dir,
                    timeout_s=900.0,
                )
                score, n = _aggregate_score(eval_log)
                task_results[f"gpqa_subset_{n}"] = {
                    "score": round(score, 4),
                    "n": n,
                    "ran_at": _now_iso(),
                    "inspect_log_dir": str(log_dir),
                }
                print(f"    score: {score:.4f} (n={n})", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"    !! gpqa failed: {e}", file=sys.stderr)

        _tool_parser = (target.get("entry") or {}).get("tool_parser")
        _tools_wanted = (drop_flag is None and "tools" in tasks
                         and (force or "tools" not in
                              [_strip_subset(t) for t in existing_tasks]))
        if _tools_wanted and backend != "ollama" and not _tool_parser:
            # A vLLM/SGLang row with no probe-verified tool parser CANNOT
            # call tools: the router strips `tools`/`tool_choice` from the
            # request (maybeStripTools) precisely so the engine does not
            # 400. Running the task anyway measured the router's own
            # stripping and recorded 0.0 -- indistinguishable in the
            # leaderboard and the picker from a model that was asked and
            # got it wrong. Record an explicit sentinel instead.
            #
            # Ollama is exempt: it negotiates tools natively over /api/chat
            # and has no probed tool_parser by design.
            print(f"  [tools]   SKIPPED -- {backend}/{served} has no "
                  f"probe-verified tool parser, so the router strips tools; "
                  f"a score here would measure nothing.", file=sys.stderr)
            task_results[f"tools_use_{n_tools}"] = {
                "score": None,
                "n": 0,
                "skipped": "no_tool_parser",
                "ran_at": _now_iso(),
            }
        elif _tools_wanted:
            from bench.tasks.tools_use import tools_use_task
            print(f"  [tools]   running n={n_tools} ...", file=sys.stderr)
            try:
                # Pass the model's probed tool_mode so the task drives
                # auto-mode models with tool_choice="auto" (pinning breaks
                # non-standard formats like Nemotron's <TOOLCALL>) and keeps
                # pinning forced-mode models. vLLM/SGLang: probed tool_mode
                # (auto|forced). Ollama has no probed tool_mode -- it
                # negotiates tools natively via /api/chat, which is an auto
                # flow -- so default Ollama to "auto"; only vLLM/SGLang rows
                # without a probed mode fall back to "forced".
                _tool_mode = (target.get("entry") or {}).get("tool_mode") or (
                    "auto" if backend == "ollama" else "forced")
                eval_log = _invoke_inspect_task(
                    task_obj=tools_use_task(n=n_tools, tool_mode=_tool_mode),
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
                    # The mode the score was MEASURED in. A forced-mode
                    # number and an auto-mode number are not comparable --
                    # forced pins the tool for the model, auto asks it to
                    # choose -- so the leaderboard must be able to say which.
                    "tool_mode": _tool_mode,
                    "tool_parser": _tool_parser,
                    "ran_at": _now_iso(),
                    "inspect_log_dir": str(log_dir),
                }
                print(f"    score: {score:.4f} (n={n})", file=sys.stderr)
                if by_sub:
                    print(f"    by_subcase: {by_sub}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"    !! tools_use failed: {e}", file=sys.stderr)

        if drop_flag is None and "longctx" in tasks and (force or "longctx_probe" not in existing_tasks):
            from bench import bench_longctx
            print(
                f"  [longctx] one prompt at {n_longctx_fraction:g}x ctx="
                f"{target['ctx']} ...",
                file=sys.stderr,
            )
            try:
                lc = bench_longctx.run(
                    model=served,
                    router_url=router_url,
                    ctx_target=int(target["ctx"]),
                    fraction=float(n_longctx_fraction),
                    max_output_tokens=int(n_longctx_max_tokens),
                    timeout_s=900.0,
                    fetch_metrics=lambda: _fetch_backend_metrics(backend),
                )
                task_results["longctx_probe"] = {**lc, "ran_at": _now_iso()}
                if lc.get("error"):
                    print(f"    !! longctx errored: {lc['error']}", file=sys.stderr)
                else:
                    print(
                        f"    target_in={lc['input_tokens_target']}  "
                        f"ttft={lc['ttft_ms']}ms  "
                        f"tps_decode={lc['tps_during_decode']}/s  "
                        f"out={lc['output_tokens']}  "
                        f"finish={lc['finish_reason']}  "
                        f"kv={lc['peak_kv_cache_perc']}  "
                        f"preempt={lc['preemptions']}",
                        file=sys.stderr,
                    )
            except Exception as e:  # noqa: BLE001
                print(f"    !! longctx failed: {e}", file=sys.stderr)

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
    # Best-effort end-of-run /metrics snapshot from the model server.
    # The container is still alive at this point (it stops on idle
    # timeout, well after we finish here). Returns {} for backends
    # without a Prometheus exporter, or on transport/parse failure.
    backend_metrics = _fetch_backend_metrics(backend)
    if backend_metrics:
        metrics.update(backend_metrics)
        # One-line summary so the operator sees the snapshot landed.
        kv = backend_metrics.get("vllm_kv_cache_usage_perc")
        preempt = backend_metrics.get("vllm_num_preemptions_total")
        if kv is not None or preempt is not None:
            print(
                f"  /metrics: kv_cache={kv}  preemptions={preempt}",
                file=sys.stderr,
            )
    # Record the sampling this row was measured at, so a cache holding
    # both greedy and override rows stays auditable.
    metrics = dict(metrics or {})
    metrics["sampling"] = sampling_record(target["alias"])
    update_row(
        cache,
        key,
        model=target["alias"],
        backend=backend,
        router_endpoint=router_url,
        context=int(target["ctx"]),
        task_results=task_results,
        metrics=metrics,
        host_env_id=host_env_id,
        backend_image_digest=backend_image_digest,
        drop_recommendation=drop_flag,
        clear_drop_recommendation=_should_clear_drop(
            existing, task_results, drop_flag),
    )
    save_cache(cache_path, cache)



def _should_clear_drop(existing: dict, task_results: dict,
                       drop_flag: dict | None) -> bool:
    """True when this run re-measured the flagged metric and it passed.

    A drop flag records the last MEASUREMENT of one metric. Left in
    place after the cause is fixed it becomes permanent: bench-sync
    classifies the row `dropped` and never re-benches it, so nothing can
    ever clear it. Nemotron-Nano-9B-v2 hit exactly that -- HumanEval 0.04
    on SGLang purely because no reasoning parser was wired, and once the
    parser was fixed the stale flag would have kept the row excluded from
    every future run.

    Deliberately narrow. Clearing on ANY clean run would let a partial
    re-run (`--tasks leak`) erase a humaneval verdict it never re-tested,
    which is how a genuinely bad model sneaks back onto the leaderboard.
    So the flagged metric must appear in THIS run's results.
    """
    if drop_flag is not None:
        return False                      # this run tripped its own flag
    prev = existing.get("drop_recommendation")
    if not isinstance(prev, dict):
        return False
    metric = prev.get("metric")
    if not metric:
        return False                      # unattributable -- leave it alone
    return any(_strip_subset(name) == metric for name in task_results)

def _latency_metrics_into_row(
    cache: dict, key: str, latency: dict,
    target: dict, backend: str, router_url: str,
    host_env_id: str | None = None,
    backend_image_digest: str | None = None,
) -> None:
    """Latency metrics belong on row.metrics, not row.tasks. Pulled
    out so the leak-task branch can write both shapes from one
    sidecar invocation.
    """
    update_row(
        cache, key,
        model=target["alias"], backend=backend, router_endpoint=router_url,
        context=int(target.get("ctx") or 0),
        metrics={
            "ttft_ms_first": latency.get("ttft_ms_first"),
            "ttft_ms_steady_p50": latency.get("ttft_ms_steady_p50"),
            "ttft_ms_steady_p95": latency.get("ttft_ms_steady_p95"),
            "tps_sustained_p50": latency.get("tps_sustained_p50"),
            "n_latency_samples": latency.get("n_samples"),
        },
        host_env_id=host_env_id,
        backend_image_digest=backend_image_digest,
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

    ``gsm8k_subset_100`` -> ``gsm8k``; ``humaneval_subset_50`` ->
    ``humaneval``; ``tools_use_20`` -> ``tools``; ``leak_probe`` ->
    ``leak``; ``longctx_probe`` -> ``longctx``. Used to skip
    already-cached tasks without forcing the same n.
    """
    if task_name.startswith("gsm8k_"):
        return "gsm8k"
    # humaneval_plus MUST be checked before humaneval -- both share the
    # "humaneval" prefix and a naive check would mis-bucket the plus rows.
    if task_name.startswith("humaneval_plus_"):
        return "humaneval_plus"
    if task_name.startswith("humaneval_"):
        return "humaneval"
    if task_name.startswith("mmlu_pro_"):
        return "mmlu_pro"
    if task_name.startswith("gpqa_"):
        return "gpqa"
    if task_name.startswith("tools_use"):
        return "tools"
    if task_name == "leak_probe":
        return "leak"
    if task_name == "longctx_probe":
        return "longctx"
    return task_name


def _now_iso() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


# --- CLI ---

def _check_router(router_url: str) -> None:
    """Hit ``/health`` so a bad route fails before the first model
    runs (saves the cold-start startup time on a bogus endpoint).

    Suppresses only the JSON-decode case: /health legitimately returns
    non-JSON ("OK"), which http_get's json.loads chokes on, but that
    means the port DID answer. Transport failures (URLError, OSError)
    propagate so the operator sees "router not reachable" immediately
    instead of chasing a 10-minute opaque streaming timeout on the
    first model.
    """
    try:
        http_get(router_url + "/health", timeout=5.0)
    except json.JSONDecodeError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", required=True, choices=["ollama", "vllm", "sglang"])
    ap.add_argument(
        "--tasks",
        default=DEFAULT_TASKS,
        help="comma-separated subset. The default covers every benchmark "
             "column the picker renders (see DEFAULT_TASKS) and therefore "
             "takes materially longer than the pre-2026-07 default; pass a "
             "narrower list to trade coverage for wall time.",
    )
    ap.add_argument("--repo", default="", help="regex filter on probe-cache top-level key")
    ap.add_argument("--force", action="store_true", help="re-run tasks even if cached")
    ap.add_argument("--host-vram-gb", type=int, default=DEFAULT_HOST_VRAM_GB)
    ap.add_argument("--n-gsm8k", type=int, default=int(os.environ.get("BENCH_N_GSM8K", "100")))
    ap.add_argument("--n-humaneval", type=int, default=int(os.environ.get("BENCH_N_HUMANEVAL", "50")))
    ap.add_argument("--n-tools", type=int, default=int(os.environ.get("BENCH_N_TOOLS", "20")))
    ap.add_argument("--n-mmlu-pro", type=int, default=int(os.environ.get("BENCH_N_MMLU_PRO", "100")))
    ap.add_argument("--n-gpqa", type=int, default=int(os.environ.get("BENCH_N_GPQA", "100")))
    ap.add_argument(
        "--drop-threshold", type=float,
        default=float(os.environ.get("BENCH_DROP_THRESHOLD", "0.70")),
        help="early-drop floor: a model scoring below this on gsm8k or "
             "humaneval (or with any leak) skips its remaining tasks and is "
             "flagged for drop. Tools is excluded. Default 0.70.",
    )
    ap.add_argument(
        "--no-early-drop", action="store_true",
        help="disable early-drop -- run every task even when a model trips a "
             "disqualifier (still records scores, no drop flag).",
    )
    ap.add_argument("--n-leak-prompts", type=int,
                    default=int(os.environ.get("BENCH_N_LEAK_PROMPTS", "40")))
    ap.add_argument(
        "--n-longctx-fraction", type=float,
        default=float(os.environ.get("BENCH_N_LONGCTX_FRACTION", "0.8")),
        help="prompt size as a fraction of the model's max ctx (default 0.8)",
    )
    ap.add_argument(
        "--n-longctx-max-tokens", type=int,
        default=int(os.environ.get("BENCH_N_LONGCTX_MAX_TOKENS", "64")),
        help="response token cap for the long-context probe; small by "
             "design -- the probe measures prefill TTFT and steady-state "
             "decode rate, not response length",
    )
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_INSPECT_LOG_DIR)
    ap.add_argument(
        "--ctx",
        default="",
        help="comma-separated ctx tiers (e.g. '32K,128K' or '32768,131072'). "
             "Default: largest fitting ctx per model.",
    )
    ap.add_argument(
        "--all-ctx",
        action="store_true",
        help="iterate every fits=true ctx per model (overrides --ctx).",
    )
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    invalid = [t for t in tasks if t not in {
        "gsm8k", "humaneval", "humaneval_plus", "mmlu_pro", "gpqa", "tools", "leak", "longctx"}]
    if invalid:
        sys.exit(f"unknown task(s): {invalid}")

    router_url = router_url_for(args.backend)
    _check_router(router_url)

    repo_filter = args.repo or None
    if args.all_ctx:
        ctx_filter: list[int] | None = [-1]
    else:
        try:
            parsed = parse_ctx_list(args.ctx)
        except ValueError as e:
            sys.exit(f"--ctx parse error: {e}")
        ctx_filter = parsed or None
    targets = discover_models(
        args.backend,
        host_vram_gb=args.host_vram_gb,
        repo_filter=repo_filter,
        ctx_filter=ctx_filter,
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
    assert_cache_schema_compatible(cache)
    n_migrated = migrate_bench_cache_keys(cache)
    if n_migrated:
        print(
            f"bench: migrated {n_migrated} pre-2026-05-02 cache keys to "
            f"<repo>@<sha>::<backend> form",
            file=sys.stderr,
        )
    # Snapshot the host environment once per run. Stamped on every row
    # touched below so a re-bench against a different driver/kernel
    # doesn't silently mix numbers with the previous run.
    env = capture_host_env()
    env_id = stamp_host_env(cache, env)
    print(
        f"bench: host_env_id={env_id} "
        f"kernel={env.get('kernel')!r} "
        f"driver={env.get('driver_version')!r} "
        f"gpu={env.get('gpu_name')!r}",
        file=sys.stderr,
    )
    image_digest = probe_image_digest(args.backend)
    if image_digest:
        print(f"bench: backend_image_digest={image_digest}", file=sys.stderr)
    else:
        print(
            f"bench: no image digest available for {args.backend}; rows will be "
            f"unstamped and bench-sync will classify them 'unknown', not stale",
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
            n_mmlu_pro=args.n_mmlu_pro,
            n_gpqa=args.n_gpqa,
            drop_threshold=args.drop_threshold,
            early_drop=not args.no_early_drop,
            n_leak_prompts=args.n_leak_prompts,
            n_longctx_fraction=args.n_longctx_fraction,
            n_longctx_max_tokens=args.n_longctx_max_tokens,
            log_dir=args.log_dir,
            cache=cache,
            cache_path=args.cache,
            force=args.force,
            host_env_id=env_id,
            backend_image_digest=image_digest,
        )

    print(f"\nbench: wrote {args.cache}", file=sys.stderr)


if __name__ == "__main__":
    main()
