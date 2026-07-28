#!/usr/bin/env python3
"""Closed-loop bench planner and driver -- `make bench-plan` / `make bench-sync`.

Populating the leaderboard is a state transition an operator currently
does by hand and by memory: work out which probed models have no bench
row, which rows predate the current driver, which predate the current
engine image, then sequence probe -> cache-up -> bench -> report without
leaving the stack down. `plan_bench()` makes that diff explicit and
`execute()` runs it as one bounded, resumable job.

Resumability is inherited rather than invented: `update_row` in
_bench_core is a pure merge and the runner skips tasks already present
unless `--force`, so re-running after an interrupt continues where it
stopped. This script adds no state of its own.

Design constraints worth stating, because each one is a decision:

- **The target set is not re-derived.** `bench_runner.discover_models()`
  already diffs the probe cache, honours `serving_ok is not False`, and
  checks that weights are actually on disk. A second implementation here
  would drift from it, and a target set that disagrees with the runner's
  is worse than none.

- **The ledger is read-only by default.** `bench_runner` documents that a
  drop flag "never deletes weights or edits the exclusion ledger -- that
  stays an explicit operator action". `--record-drops` is the explicit
  action; without it a drop is reported and nothing is written.

- **The stack always comes back up.** The probe phase needs the GPU
  exclusively, so the serving backends go down. A `finally` guarantees
  `cache-up`, because leaving the whole inference stack offline after a
  failed probe is a far worse outcome than a failed bench.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _model_status as MS  # noqa: E402

BENCH_CACHE = REPO_ROOT / "deploy" / ".bench-cache.json"

# Tasks a row must carry to count as complete. Mirrors the runner's
# default BENCH_TASKS; an operator narrowing BENCH_TASKS narrows this too,
# via --tasks, so a deliberately partial sweep is not reported as
# permanently incomplete.
DEFAULT_TASKS = ("gsm8k", "humaneval", "humaneval_plus", "mmlu_pro",
                 "gpqa", "tools", "leak")

BACKENDS = ("vllm", "sglang", "ollama")

# Classification buckets, most-actionable first. Order matters: it is the
# print order and the order execute() drains.
CLASSES = ("new", "incomplete", "stale_env", "stale_image",
           "dropped", "excluded", "current")


def _load_bench_runner():
    """Import the runner as a module (its filename is not importable as-is).

    Imported lazily so `--help` and the unit tests do not pay for the
    runner's own imports.
    """
    spec = importlib.util.spec_from_file_location(
        "bench_runner", REPO_ROOT / "scripts" / "bench" / "bench_runner.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_runner"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_bench_cache(path: Path = BENCH_CACHE) -> dict:
    """Read the bench cache; missing or malformed -> empty (fail open)."""
    try:
        data = json.loads(Path(path).read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _row_tasks(row: dict, runner) -> set[str]:
    """Task names present on a row, as the user spells them.

    Rows record `humaneval_subset_50`, `tools_use_20`, `leak_probe`; the
    caller asks about `humaneval`, `tools`, `leak`. The runner's
    `_strip_subset` is the canonical mapping and the runner itself uses it
    to decide what to skip, so this defers to it rather than
    reimplementing.

    Reimplementing it is not hypothetical: the first version of this
    function split on `_subset_`/`_probe` and reported `tools` missing on
    9 of 10 rows that had in fact benched it -- which would have re-run
    the whole leaderboard to reproduce data it already had.
    """
    tasks = row.get("tasks")
    if not isinstance(tasks, dict):
        return set()
    return {runner._strip_subset(name) for name in tasks}


def classify_target(
    target: dict,
    *,
    backend: str,
    bench_cache: dict,
    ledger: dict,
    required_tasks: tuple[str, ...],
    current_env_id: str | None,
    current_image_digest: str | None,
    runner,
) -> tuple[str, str]:
    """Classify one bench target. Returns (class, human-readable reason).

    Exactly one class per target, evaluated in priority order: an
    excluded model is not also 'new', and a dropped row is not also
    'stale'.
    """
    key = target["key"]
    alias = target.get("alias") or key
    ctx = target.get("ctx")
    sha = (target.get("entry") or {}).get("sha")

    if MS.is_bench_excluded(ledger, alias, backend, ctx=ctx, sha=sha):
        reason = MS.exclusion_reason(ledger, alias, backend)
        return "excluded", f"ledger: {reason}"

    row = bench_cache.get(key)
    if not isinstance(row, dict):
        return "new", "no bench row"

    if row.get("drop_recommendation"):
        drop = row["drop_recommendation"]
        why = drop.get("reason") if isinstance(drop, dict) else drop
        return "dropped", f"drop flag: {why}"

    missing = [t for t in required_tasks if t not in _row_tasks(row, runner)]
    if missing:
        return "incomplete", f"missing tasks: {', '.join(missing)}"

    # Staleness is only meaningful when we know what "current" is AND the
    # row was stamped. An unstamped row is `unknown`, never stale --
    # guessing would either force a needless full re-bench or hide a real
    # drift, and both are worse than saying so.
    row_env = row.get("host_env_id")
    if current_env_id and row_env and row_env != current_env_id:
        return "stale_env", f"host env {row_env} != {current_env_id}"

    row_image = row.get("backend_image_digest")
    if current_image_digest and row_image and row_image != current_image_digest:
        return "stale_image", f"image {row_image[:19]} != {current_image_digest[:19]}"

    if current_env_id and not row_env:
        return "current", "complete (unstamped host env -- cannot judge staleness)"
    if current_image_digest and not row_image:
        return "current", "complete (unstamped image -- cannot judge staleness)"
    return "current", "complete"


def plan_bench(
    backends: tuple[str, ...] = BACKENDS,
    *,
    host_vram_gb: int,
    repo_filter: str | None = None,
    ctx_filter: list[int] | None = None,
    required_tasks: tuple[str, ...] = DEFAULT_TASKS,
    bench_cache: dict | None = None,
    ledger: dict | None = None,
    runner=None,
) -> dict:
    """Diff probe caches against the bench cache. Read-only.

    Returns {class: [ {backend, key, alias, ctx, reason}, ... ]} for every
    class in CLASSES.
    """
    runner = runner or _load_bench_runner()
    bench_cache = load_bench_cache() if bench_cache is None else bench_cache
    ledger = MS.load_ledger() if ledger is None else ledger

    meta = bench_cache.get("_meta") if isinstance(bench_cache.get("_meta"), dict) else {}
    current_env_id = meta.get("current_host_env_id")

    plan: dict[str, list[dict]] = {c: [] for c in CLASSES}
    for backend in backends:
        try:
            targets = runner.discover_models(
                backend, host_vram_gb=host_vram_gb,
                repo_filter=repo_filter, ctx_filter=ctx_filter)
        except Exception as e:  # noqa: BLE001
            print(f"  note: {backend}: cannot enumerate targets ({e})",
                  file=sys.stderr)
            continue
        digest = runner.probe_image_digest(backend)
        for t in targets:
            cls, why = classify_target(
                t, backend=backend, bench_cache=bench_cache, ledger=ledger,
                required_tasks=required_tasks, current_env_id=current_env_id,
                current_image_digest=digest, runner=runner)
            plan[cls].append({
                "backend": backend,
                "key": t["key"],
                "alias": t.get("alias") or t["key"],
                "ctx": t.get("ctx"),
                "reason": why,
                # Carried so execute() can tell a STALE row (which must be
                # re-benched with --force, because update_row merges) from a
                # new/incomplete one (which must not be, or resumability is
                # lost). needs_bench() flattens the classes, so without this
                # the distinction is gone by the time it is needed.
                "class": cls,
            })
    return plan


def needs_bench(plan: dict) -> list[dict]:
    """The classes execute() will actually re-run, in priority order.

    `dropped` is deliberately absent: a drop is a verdict, and re-running
    it burns an hour to reproduce a number the operator already has.
    """
    out: list[dict] = []
    for cls in ("new", "incomplete", "stale_env", "stale_image"):
        out.extend(plan.get(cls, []))
    return out


def print_plan(plan: dict, *, host_vram_gb: int, max_targets: int) -> None:
    total = sum(len(v) for v in plan.values())
    print(f"bench-plan: {total} target(s) at host_vram={host_vram_gb}G\n")
    for cls in CLASSES:
        rows = plan.get(cls) or []
        if not rows:
            continue
        print(f"  {cls} ({len(rows)}):")
        for r in rows:
            print(f"    - {r['alias']} [{r['backend']}] @ {r['ctx']}  -- {r['reason']}")
        print()
    queue = needs_bench(plan)
    if not queue:
        print("  nothing to bench: every target is current, dropped, or excluded.")
        return
    shown = queue[:max_targets] if max_targets > 0 else queue
    print(f"  would bench {len(shown)} of {len(queue)} (BENCH_MAX_TARGETS={max_targets}):")
    for r in shown:
        print(f"    - {r['alias']} [{r['backend']}] @ {r['ctx']}")
    if len(shown) < len(queue):
        # Never let a budget silently look like full coverage.
        print(f"  NOTE: {len(queue) - len(shown)} target(s) left unbenched by "
              f"the BENCH_MAX_TARGETS budget; re-run to continue.")


def _run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def execute(plan: dict, *, max_targets: int, tasks: tuple[str, ...],
            record_drops: bool = False) -> int:
    """Bench the queued targets, grouped by backend.

    Grouping matters: each backend switch costs a full model load, so a
    mixed queue interleaved by model would thrash the GPU. One `make
    bench-<backend>` per backend, filtered to the queued rows, keeps that
    to one switch per backend.
    """
    queue = needs_bench(plan)
    if not queue:
        print("\nnothing to bench.")
        return 0
    if max_targets > 0 and len(queue) > max_targets:
        dropped = len(queue) - max_targets
        queue = queue[:max_targets]
        print(f"\nbudget: benching {max_targets}, leaving {dropped} for a "
              f"later run (BENCH_MAX_TARGETS)")

    # Group by (backend, force). Backend grouping keeps GPU switches to
    # one per backend. The force split matters for correctness: update_row
    # is a pure MERGE, so re-benching a stale row without --force
    # overwrites only the tasks that actually re-ran and leaves the rest
    # of the row's metrics behind -- producing a row that is half old
    # host-env/image and half new, with a single fresh stamp claiming all
    # of it. `new` and `incomplete` rows have nothing stale to merge into
    # and must NOT be forced, or every partially-benched row restarts from
    # scratch and the resumability the planner exists for is lost.
    _FORCE_CLASSES = ("stale_env", "stale_image")
    by_group: dict[tuple[str, bool], list[dict]] = {}
    for r in queue:
        force = r.get("class") in _FORCE_CLASSES
        by_group.setdefault((r["backend"], force), []).append(r)

    rc = 0
    for (backend, force), rows in sorted(by_group.items()):
        # BENCH_REPO is a regex over the PROBE-cache key, so the bench
        # key's ::<backend>::<ctx> suffix has to come off first -- see
        # probe_key(). An alternation of the de-suffixed keys reproduces
        # this selection inside the runner. Duplicates collapse because
        # two ctx tiers of one model share a probe key.
        pattern = "|".join(sorted({_escape(probe_key(r["key"])) for r in rows}))
        cmd = ["make", f"bench-{backend}",
               f"BENCH_REPO={pattern}",
               f"BENCH_TASKS={','.join(tasks)}"]
        if force:
            cmd.append("BENCH_FORCE=1")
            print(f"  ({backend}: {len(rows)} stale row(s) -> BENCH_FORCE=1, "
                  f"so the whole row is re-measured rather than merged)")
        step = _run(cmd)
        if step != 0:
            print(f"\nstep failed (rc={step}): {' '.join(cmd)}", file=sys.stderr)
            rc = step

    if record_drops:
        rc_drop = _record_drops(plan)
        if rc_drop != 0:
            rc = rc_drop
    else:
        n = len(plan.get("dropped") or [])
        if n:
            print(f"\n{n} target(s) carry a drop flag. The ledger was NOT "
                  f"modified -- re-run with --record-drops to record them.")

    report = _run(["make", "bench-report"])
    return rc or report


def probe_key(bench_key: str) -> str:
    """The probe-cache key underlying a bench-cache key.

    `discover_models` returns the BENCH-cache key
    (`<repo>@<sha>::<backend>::<ctx>`, or `<digest>::ollama::<ctx>`), but
    `--repo` is matched with `re.search` against PROBE-cache keys, which
    carry no backend or ctx suffix. Handing the bench key straight to
    `--repo` therefore matches nothing at all, and the runner exits with
    "no fitting <backend> models in probe cache" -- which is what the
    first version of execute() did on every single run.

    Splitting on the first `::` is safe for both key shapes: HF repo
    names and Ollama manifest digests never contain a colon pair.
    """
    return bench_key.split("::", 1)[0]


def _escape(s: str) -> str:
    """Escape a probe-cache key for use inside a BENCH_REPO regex."""
    import re
    return re.escape(s)


def _record_drops(plan: dict) -> int:
    rows = plan.get("dropped") or []
    if not rows:
        return 0
    ledger = MS.load_ledger()
    for r in rows:
        entry = MS.record_bench_verdict(
            ledger, r["alias"], r["backend"], "bench_dropped",
            detail=r["reason"], ctx=r["ctx"])
        print(f"  recorded bench_dropped: {r['alias']} [{r['backend']}] "
              f"@ {r['ctx']} (attempt {entry['attempts']})")
    MS.save_ledger(ledger)
    print(f"\nwrote {len(rows)} verdict(s) to the exclusion ledger "
          f"(clear with `make model-status CLEAR=<name>::<backend>`)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Plan and run the bench closed loop.")
    ap.add_argument("--backend", action="append", choices=BACKENDS,
                    help="Restrict to one backend (repeatable). Default: all.")
    ap.add_argument("--vram", type=int,
                    default=int(float(os.environ.get("GPU_MEMORY_GB", 24))),
                    help="Host VRAM band to plan against.")
    ap.add_argument("--repo", default=os.environ.get("BENCH_REPO") or None,
                    help="Regex filter over probe-cache keys.")
    ap.add_argument("--ctx", default=os.environ.get("BENCH_CTX") or None,
                    help="Comma-separated ctx list, or 'all'.")
    ap.add_argument("--tasks",
                    default=os.environ.get("BENCH_TASKS") or ",".join(DEFAULT_TASKS),
                    help="Comma-separated task list; also defines 'complete'.")
    ap.add_argument("--max-targets", type=int,
                    default=int(os.environ.get("BENCH_MAX_TARGETS", 0)),
                    help="Cap targets benched this run. 0 = unlimited.")
    ap.add_argument("--record-drops", action="store_true",
                    help="Write bench_dropped verdicts to the exclusion "
                         "ledger. Off by default -- a drop is an operator "
                         "decision, not an automatic one.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan only. Touches no container and no file.")
    args = ap.parse_args(argv)

    ctx_filter = None
    if args.ctx:
        ctx_filter = [-1] if args.ctx.strip().lower() == "all" else [
            int(c) for c in args.ctx.split(",") if c.strip()]
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    backends = tuple(args.backend) if args.backend else BACKENDS

    plan = plan_bench(backends, host_vram_gb=args.vram, repo_filter=args.repo,
                      ctx_filter=ctx_filter, required_tasks=tasks)
    print_plan(plan, host_vram_gb=args.vram, max_targets=args.max_targets)

    if args.dry_run:
        print("\ndry run: nothing executed.")
        return 0
    return execute(plan, max_targets=args.max_targets, tasks=tasks,
                   record_drops=args.record_drops)


if __name__ == "__main__":
    sys.exit(main())
