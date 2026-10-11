#!/usr/bin/env python3
"""vLLM 0.30.0 prefix-cache retention rule (platform-rebuild Phase 4, step 4v).

Read-only on every input, standard library + statlib.py only. One
subcommand:

  retention  the prefix-cache retention rule (a)-(b): `None` against the
             default, over the per-session bench_concurrency reports.

  vllm_upgrade_accept.py retention \\
      --default <session.json>... --none <session.json>... [--out report.json]

The `accept` subcommand (0.28.0 against 0.30.0: quality non-inferiority,
speed ratios, MTP acceptance length) was removed on 2026-10-07 with the
0.28.0 arm itself: the operator accepts 0.30.0 on its own probes (plan E7).
Git history keeps it.

Retention (plan 4v, "Prefix-cache retention"; same image in both arms, so no
id check). `--default` / `--none`: the per-session JSON reports
`bench_concurrency.py --mode multiturn` printed, one file per session. The
sessions pair by `salt` (session i ran with the same salt in both arms); the
input is refused when the salts do not pair, a turn did not succeed, or a
session's counter read-out is incomplete.
  - (a) in every session, the None arm's cached prompt tokens on turns 2..N
    (vllm:prefix_cache_hits_total, per-turn deltas) exceed the default's;
    deterministic (docs/statistics-primer.md Sec. 12).
  - (b) per session, the median TTFT of turns 2..N; the ratio of the arms'
    medians of those (None / default) has a 95% upper bound not above 1.0,
    by statlib.boot_ci_two_sample over sessions (10,000 resamples), labelled
    approximate (a percentile bootstrap over a handful of sessions).
  - adopt `None` only if (a) and (b) hold; (c), the exact-context probes with
    the flag, is not evaluated here.

Exit codes: 0 a verdict was reached (adopt or keep_default); 2 bad input.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]

import statlib as L  # noqa: E402

HITS = "vllm:prefix_cache_hits_total"
BOOT_REPS = 10000

EXIT_OK, EXIT_BAD_INPUT = 0, 2


class BadInput(Exception):
    """An input the rule cannot be applied to: exit 2, no verdict."""


def _load_session(path: Path) -> dict:
    try:
        s = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise BadInput(f"{path}: {e}") from e
    if not s.get("salt") or not isinstance(s.get("turns"), list):
        raise BadInput(f"{path}: not a bench_concurrency multiturn report (no salt or turns)")
    if s.get("metrics_complete") is not True:
        raise BadInput(f"{path}: the counter read-out is incomplete (metrics_complete is not true)")
    later = [t for t in s["turns"] if t.get("turn", 0) >= 2]
    if not later:
        raise BadInput(f"{path}: no turn 2 or later")
    for t in s["turns"]:
        if not t.get("ok") or t.get("ttft_ms") is None:
            raise BadInput(f"{path}: turn {t.get('turn')} did not succeed")
    for t in later:
        if HITS not in (t.get("counters_delta") or {}):
            raise BadInput(f"{path}: turn {t['turn']} has no {HITS} delta")
    return s


def _arm(paths: list[Path], label: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in paths:
        s = _load_session(p)
        if s["salt"] in out:
            raise BadInput(f"{label}: salt {s['salt']} appears in more than one session")
        out[s["salt"]] = s
    return out


def _session_facts(s: dict) -> dict:
    later = [t for t in s["turns"] if t["turn"] >= 2]
    tokens = [t.get("completion_tokens") for t in s["turns"]]
    return {"cached_tokens_turn2plus": math.fsum(t["counters_delta"][HITS] for t in later),
            "ttft_median_turn2plus_ms": statistics.median(t["ttft_ms"] for t in later),
            "turns": len(s["turns"]),
            "completion_tokens_min": min((x for x in tokens if x is not None), default=None),
            "min_tokens": s.get("min_tokens")}


def _median_ratio(xs: list[float], ys: list[float]) -> float:
    return statistics.median(xs) / statistics.median(ys)


def retention(args: argparse.Namespace) -> tuple[dict, int]:
    default, none = _arm(args.default, "default"), _arm(args.none, "None")
    if set(default) != set(none):
        raise BadInput("the arms' salts do not pair: default only "
                       f"{sorted(set(default) - set(none))}, None only {sorted(set(none) - set(default))}")
    sessions = {}
    for salt in sorted(default):
        d, n = _session_facts(default[salt]), _session_facts(none[salt])
        sessions[salt] = {"default": d, "none": n,
                          "none_caches_more": n["cached_tokens_turn2plus"] > d["cached_tokens_turn2plus"]}
    rule_a = all(s["none_caches_more"] for s in sessions.values())
    xs = [s["none"]["ttft_median_turn2plus_ms"] for s in sessions.values()]
    ys = [s["default"]["ttft_median_turn2plus_ms"] for s in sessions.values()]
    lo, hi = L.boot_ci_two_sample(xs, ys, _median_ratio, reps=BOOT_REPS)
    rule_b = hi is not None and hi <= 1.0
    report = {
        "kind": "vllm_prefix_cache_retention", "n_sessions": len(sessions), "sessions": sessions,
        "rule_a": {"holds": rule_a, "failing_sessions":
                   [k for k, s in sessions.items() if not s["none_caches_more"]]},
        "rule_b": {"holds": rule_b, "ratio_none_over_default": _median_ratio(xs, ys),
                   "boot95": [lo, hi], "upper_bound_approximate": True,
                   "note": f"percentile bootstrap over {len(sessions)} sessions per arm"},
        "rule_c": "not evaluated here: the exact-context probes with the flag",
        "verdict": "adopt" if rule_a and rule_b else "keep_default",
    }
    return report, EXIT_OK


# ---------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retention", help="prefix-cache retention rule (a)-(b)")
    r.add_argument("--default", type=Path, nargs="+", required=True)
    r.add_argument("--none", type=Path, nargs="+", required=True)
    r.add_argument("--out", type=Path, default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report, code = retention(args)
    except BadInput as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_BAD_INPUT
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    else:
        print(text)
    print(f"verdict: {report['verdict']}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
