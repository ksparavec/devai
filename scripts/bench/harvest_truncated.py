#!/usr/bin/env python3
"""Score a bench task that was stopped at its wall-clock deadline.

bench-sync stops every task after its deadline (owner's rule, 2026-09-28:
30 min per task, then SIGINT so inspect can write a `cancelled` log, then
SIGKILL). The task never returned, so nothing was written to the cache.
This reads the partial inspect log and writes the task entry.

Scored on the UNBROKEN PREFIX only (owner's choice): the questions run in a
fixed order -- the benchmark's seeded random order, or its first n -- and
the entry covers questions 1..k, where k+1 is the first question without a
result. Questions still running at the cut are disproportionately the long,
hard ones; keeping "whatever finished" would drop them and inflate the score
of a slow model. A prefix of a random order is still an unbiased random
sample, only a smaller one. A question stopped by the per-question working
limit HAS a result (scored wrong, counted in n_timeouts); a question that
raised has none and ends the prefix.

Usage (inside the lab image -- inspect-ai 0.3.271 logs are zstd, which only
Python >= 3.14 reads; `make bench-harvest HARVEST_ARGS='...'`):

  harvest_truncated.py --log <eval> --key <bench-cache key> --model <alias>
      --backend <b> --ctx <n> --task <task> --planned-n <n> --deadline-s <s>
      [--cache /deploy/.bench-cache.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _probe_core import load_cache, save_cache  # noqa: E402
from bench._bench_core import DEFAULT_CACHE_PATH, _now_iso, router_url_for, update_row  # noqa: E402

TASK_KEY_PREFIX = {"gsm8k": "gsm8k_subset_", "humaneval": "humaneval_subset_",
                   "humaneval_plus": "humaneval_plus_subset_", "mmlu_pro": "mmlu_pro_subset_",
                   "gpqa": "gpqa_subset_", "tools": "tools_use_"}
TIMEOUT_TYPES = ("time", "working")


def read_samples(log: Path) -> tuple[list[dict], str | None]:
    """Samples of a (possibly partial, header-less) .eval log, and its status."""
    z = zipfile.ZipFile(log)
    names = z.namelist()
    status = json.loads(z.read("header.json")).get("status") if "header.json" in names else None
    out = []
    for nm in names:
        if not nm.startswith("samples/"):
            continue
        s = json.loads(z.read(nm))
        sc = next((v for v in (s.get("scores") or {}).values() if v is not None), None)
        limit = s.get("limit") or {}
        out.append({"id": int(s["id"]),
                    "has_result": sc is not None and not s.get("error"),
                    "correct": sc is not None and sc.get("value") in ("C", 1, 1.0, True),
                    "timeout": isinstance(limit, dict) and limit.get("type") in TIMEOUT_TYPES,
                    "limited": bool(limit) and not (isinstance(limit, dict) and limit.get("type") in TIMEOUT_TYPES),
                    "subcase": (s.get("metadata") or {}).get("subcase")})
    return out, status


def prefix(samples: list[dict]) -> list[dict]:
    """Samples 1..k with k maximal such that every one of them has a result."""
    by_id = {s["id"]: s for s in samples}
    out, i = [], 1
    while i in by_id and by_id[i]["has_result"]:
        out.append(by_id[i])
        i += 1
    return out


def entry(task: str, samples: list[dict], status: str | None, *, planned_n: int,
          deadline_s: int, log_name: str) -> dict:
    pre = prefix(samples)
    k = len(pre)
    x = sum(s["correct"] for s in pre)
    e = {"pass@1" if task.startswith("humaneval") else "score": round(x / k, 4) if k else None,
         "n": k, "n_planned": planned_n, "n_in_log": len(samples),
         "truncated": {"reason": "task wall-clock deadline", "deadline_s": deadline_s,
                       "prefix": k, "log_status": status},
         "n_timeouts": sum(s["timeout"] for s in pre), "n_limited": sum(s["limited"] for s in pre),
         "n_errors": sum(not s["has_result"] for s in samples),
         "inspect_log": log_name, "ran_at": _now_iso()}
    if task == "tools" and k:
        subs: dict[str, list[int]] = {}
        for s in pre:
            if s["subcase"]:
                subs.setdefault(s["subcase"], []).append(int(s["correct"]))
        e["by_subcase"] = {k2: round(sum(v) / len(v), 4) for k2, v in subs.items()}
    return e


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--task", required=True, choices=sorted(TASK_KEY_PREFIX))
    ap.add_argument("--planned-n", type=int, required=True)
    ap.add_argument("--deadline-s", type=int, required=True)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    a = ap.parse_args(argv)

    samples, status = read_samples(a.log)
    e = entry(a.task, samples, status, planned_n=a.planned_n, deadline_s=a.deadline_s,
              log_name=a.log.name)
    cache = load_cache(a.cache)
    host_env_id = (cache.get("_meta") or {}).get("current_host_env_id")
    update_row(cache, a.key, model=a.model, backend=a.backend,
               router_endpoint=router_url_for(a.backend), context=a.ctx,
               task_results={f"{TASK_KEY_PREFIX[a.task]}{a.planned_n}": e},
               host_env_id=host_env_id)
    save_cache(a.cache, cache)
    print(f"harvested {a.task} for {a.model} [{a.backend}]: prefix {e['n']}/{a.planned_n} "
          f"({e['n_in_log']} in log, status {status}), "
          f"score {e.get('score', e.get('pass@1'))}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
