#!/usr/bin/env python3
"""vLLM 0.30.0 acceptance rule (platform-rebuild Phase 4, step 4v): statistics and verdicts.

Read-only on every input, standard library + statlib.py only. Two subcommands:

  accept     quality (non-inferiority, deciding), speed and MTP acceptance
             length (reported), for the old build (0.28.0) against the new
             one (0.30.0), on the two arms' bench logs.
  retention  the prefix-cache retention rule (a)-(b): `None` against the
             default, over the per-session bench_concurrency reports.

Usage:

  vllm_upgrade_accept.py accept \\
      --old-logs <dir-or-.eval>... --new-logs <dir-or-.eval>... \\
      --old-tag-id <id> --old-served-id <id> --new-tag-id <id> --new-served-id <id> \\
      [--old-speed <manifest.json> --new-speed <manifest.json>] [--out report.json]

  vllm_upgrade_accept.py retention \\
      --default <session.json>... --none <session.json>... [--out report.json]

`--old-logs` / `--new-logs`: each build's inspect_ai `.eval` logs (files, or
directories searched for `*.eval`). Exactly one gsm8k and one humaneval log
per build (by the log's own task name); anything else is refused. inspect-ai
0.3.271 writes its archives with zstd, which Python's zipfile reads only from
3.14 on: run this inside the lab image for real logs (scripts/stats/README.md).

Image ids (plan 4v, "Image selection reaches every consumer"): the tag id is
`podman image inspect --format '{{.Id}}' <tag>`, recorded before the arm; the
served id is `podman inspect --format '{{.Image}}' devai-vllm-devai`, recorded
after it. The verdict is REFUSED -- exit 3, and the report carries no quality
verdict at all -- when the two arms' tag ids are equal (both arms ran one
build), or when an arm's served id differs from its tag id (the router served
another image). Ids compare without a `sha256:` prefix, case-insensitive; a
12-character or longer prefix of the other id counts as the same id.

Quality (deciding). Per task, items are paired by sample id across builds.
  - The bound: a time-out on the new build counts wrong there, a time-out on
    the old build counts right there, and the other build's observed outcome
    stands (an item timed out on both is right on old, wrong on new): the
    conservative assignment (docs/statistics-primer.md Sec. 7). The
    difference is p(old) - p(new); its upper bound is the upper end of the
    two-sided 95% Newcombe interval (statlib.newcombe_paired), a one-sided
    97.5% bound (Bonferroni over the two tasks). It must lie below the
    task's margin (gsm8k 3 points, humaneval 5).
  - Fail: an exact McNemar test (statlib.mcnemar_exact; time-outs wrong, as
    the harness scores them), Holm over the two tasks, shows a loss on the
    new build (more items right only on old) with an adjusted p < 0.05.
  - Verdict: fail if any task meets Fail (Fail takes precedence over the
    bound); pass if both bounds lie below their margins and no task meets
    Fail; anything else inconclusive. An item present on one build only, or
    with no score (an errored sample), is reported and makes Pass impossible:
    the rule is stated on per-item pairs.
  - Each task's expected half-width at 10% discordance
    (statlib.mde_paired_halfwidth) is reported beside its interval.

Speed (reported, not deciding). `--old-speed` / `--new-speed` are JSON
manifests, one per build, paths relative to the manifest's directory:

  {"rows": {"<row>": {"latency": "logs/latency-<alias>.jsonl",
                      "metrics_before": "mtp/<row>-before.prom",   # optional
                      "metrics_after": "mtp/<row>-after.prom",     # optional
                      "started_before": "<StartedAt>",             # optional
                      "started_after": "<StartedAt>"}}}            # optional

  Rows pair by key across the two manifests. The latency file holds one
  record per prompt (bench_runner: i, id, ttft_ms, effective_tokens,
  token_source, stream_seconds, finish_reason, error). Prompts 1-39 only
  (prompt 0 is the cold start), paired by id. TTFT ratio new/old per
  prompt; decode-rate ratio (effective_tokens / stream_seconds) only where
  token_source is "usage" on both builds and at least 64 tokens were
  generated on each. Each ratio is the geometric mean of the per-prompt
  ratios, with a 95% percentile bootstrap interval over prompts
  (statlib.boot_ci). MTP mean acceptance length = 1 + delta accepted draft
  tokens / delta drafts, from the two saved /metrics texts; reported missing
  when the start time changed (the engine was relaunched during the run), a
  counter is absent, or no draft was made.

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

Exit codes: 0 a verdict was reached (pass, fail or inconclusive; adopt or
keep_default); 2 bad input; 3 refused (image ids).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import zipfile
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent)]

import statlib as L  # noqa: E402
from bench._bench_core import parse_prometheus_counters  # noqa: E402

OLD_LABEL, NEW_LABEL = "0.28.0", "0.30.0"
TASKS = ("gsm8k", "humaneval")
TASK_NAMES = {"gsm8k_task": "gsm8k", "gsm8k": "gsm8k",
              "humaneval_task": "humaneval", "humaneval": "humaneval"}
# Non-inferiority margins, fixed before the runs (plan 4v, operator 2026-10-06).
MARGINS = {"gsm8k": 0.03, "humaneval": 0.05}
FAIL_ALPHA = 0.05
DISCORDANCE_FOR_HALFWIDTH = 0.10
TIMEOUT_TYPES = ("time", "working")
CORRECT_VALUES = ("C", 1, 1.0, True)

# Speed (plan 4v): prompts 1-39, paired by id; decode only on engine counts
# of at least 64 tokens on both builds.
SPEED_FIRST_I, SPEED_LAST_I = 1, 39
MIN_DECODE_TOKENS = 64
ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
DRAFTS = "vllm:spec_decode_num_drafts_total"

HITS = "vllm:prefix_cache_hits_total"
BOOT_REPS = 10000
ID_PREFIX_MIN = 12

EXIT_OK, EXIT_BAD_INPUT, EXIT_REFUSED = 0, 2, 3


class BadInput(Exception):
    """Input the rule cannot be computed on; exit 2 with the message."""


# ---------------------------------------------------------------- inspect logs

def _task_of(z: zipfile.ZipFile, names: list[str]) -> str | None:
    for member in ("header.json", "_journal/start.json"):
        if member in names:
            raw = (json.loads(z.read(member)).get("eval") or {}).get("task") or ""
            return TASK_NAMES.get(raw.replace("@", "/").rsplit("/", 1)[-1])
    return None


def _outcome(sample: dict) -> str:
    """right, wrong, timeout (scored wrong by the harness), or error (no result)."""
    limit = sample.get("limit")
    if isinstance(limit, dict) and limit.get("type") in TIMEOUT_TYPES:
        return "timeout"
    score = next((v for v in (sample.get("scores") or {}).values() if v is not None), None)
    if sample.get("error") or score is None:
        return "error"
    return "right" if score.get("value") in CORRECT_VALUES else "wrong"


def read_eval_log(path: Path) -> tuple[str | None, dict[str, str]]:
    """(task, {sample id: outcome}) of one .eval log; header-less logs included."""
    try:
        z = zipfile.ZipFile(path)
        names = z.namelist()
        task = _task_of(z, names)
        outcomes: dict[str, str] = {}
        for nm in names:
            if not nm.startswith("samples/"):
                continue
            s = json.loads(z.read(nm))
            sid = str(s["id"])
            if sid in outcomes:
                raise BadInput(f"{path}: sample {sid} appears twice (more than one epoch?)")
            outcomes[sid] = _outcome(s)
    except NotImplementedError as e:
        raise BadInput(f"{path}: {e} -- inspect-ai 0.3.271 logs are zstd; run this with "
                       "Python >= 3.14 (the lab image, scripts/stats/README.md)") from e
    except (zipfile.BadZipFile, KeyError, ValueError) as e:
        raise BadInput(f"{path}: not a readable inspect .eval log ({e})") from e
    return task, outcomes


def _eval_files(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.glob("*.eval")))
        elif p.is_file():
            out.append(p)
        else:
            raise BadInput(f"{p}: no such file or directory")
    return out


def load_build_logs(paths: list[Path], label: str) -> dict[str, dict[str, str]]:
    """{task: outcomes} for one build; exactly one log per task in TASKS."""
    found: dict[str, list[tuple[Path, dict[str, str]]]] = {t: [] for t in TASKS}
    for f in _eval_files(paths):
        task, outcomes = read_eval_log(f)
        if task in found:
            found[task].append((f, outcomes))
    out = {}
    for t in TASKS:
        logs = found[t]
        if len(logs) != 1:
            names = ", ".join(str(f) for f, _ in logs) or "none"
            raise BadInput(f"{label}: expected exactly one {t} log, found {len(logs)} ({names})")
        out[t] = logs[0][1]
    return out


# ---------------------------------------------------------------- quality

def _table(pairs: list[tuple[bool, bool]]) -> dict[str, int]:
    """2x2 over (old right, new right): a both, b old only, c new only, d neither."""
    return {"a": sum(o and n for o, n in pairs), "b": sum(o and not n for o, n in pairs),
            "c": sum(n and not o for o, n in pairs), "d": sum(not o and not n for o, n in pairs)}


def compare_task(task: str, old: dict[str, str], new: dict[str, str]) -> dict:
    """Bound, McNemar inputs and counts for one task (Holm is applied across tasks)."""
    ids = sorted(set(old) | set(new), key=lambda s: (len(s), s))
    paired = [i for i in ids if old.get(i) in ("right", "wrong", "timeout")
              and new.get(i) in ("right", "wrong", "timeout")]
    unpaired = [i for i in ids if i not in paired]
    # The bound: time-out right on old, wrong on new; the observed outcome otherwise.
    bound = _table([(old[i] in ("right", "timeout"), new[i] == "right") for i in paired])
    # McNemar: time-outs wrong on both builds, as the harness scores them.
    harness = _table([(old[i] == "right", new[i] == "right") for i in paired])
    out = {
        "n_pairs": len(paired),
        "unpaired_or_unscored": [{"id": i, OLD_LABEL: old.get(i, "absent"),
                                  NEW_LABEL: new.get(i, "absent")} for i in unpaired],
        "timeouts": {OLD_LABEL: sum(v == "timeout" for v in old.values()),
                     NEW_LABEL: sum(v == "timeout" for v in new.values())},
        "margin": MARGINS[task],
        "bound_table": bound,
        "harness_table": harness,
        "expected_halfwidth_at_10pct_discordance": (
            L.mde_paired_halfwidth(len(paired), DISCORDANCE_FOR_HALFWIDTH) if paired else None),
    }
    if not paired:
        return {**out, "difference_old_minus_new": None, "newcombe95": None,
                "upper_bound": None, "bound_below_margin": False,
                "accuracy": None, "mcnemar_p": None}
    diff, lo, hi = L.newcombe_paired(bound["a"], bound["b"], bound["c"], bound["d"])
    n = len(paired)
    return {**out, "difference_old_minus_new": diff, "newcombe95": [lo, hi],
            "upper_bound": hi, "bound_below_margin": hi < MARGINS[task],
            "accuracy": {"harness": {OLD_LABEL: (harness["a"] + harness["b"]) / n,
                                     NEW_LABEL: (harness["a"] + harness["c"]) / n},
                         "bound": {OLD_LABEL: (bound["a"] + bound["b"]) / n,
                                   NEW_LABEL: (bound["a"] + bound["c"]) / n}},
            "mcnemar_p": L.mcnemar_exact(harness["b"], harness["c"])}


def quality(old_logs: dict[str, dict[str, str]], new_logs: dict[str, dict[str, str]]) -> dict:
    tasks = {t: compare_task(t, old_logs[t], new_logs[t]) for t in TASKS}
    holm = L.holm({t: (r["mcnemar_p"] if r["mcnemar_p"] is not None else 1.0)
                   for t, r in tasks.items()})
    reasons = []
    for t, r in tasks.items():
        r["mcnemar_p_holm"] = holm[t]
        h = r["harness_table"]
        r["fail"] = h["b"] > h["c"] and holm[t] < FAIL_ALPHA
        if r["fail"]:
            reasons.append(f"{t}: significant loss on {NEW_LABEL} (Holm-adjusted p = {holm[t]:.4g})")
    if any(r["fail"] for r in tasks.values()):
        verdict = "fail"
    else:
        for t, r in tasks.items():
            if not r["bound_below_margin"]:
                reasons.append(f"{t}: upper bound {r['upper_bound']} not below the margin {r['margin']}")
            if r["unpaired_or_unscored"]:
                reasons.append(f"{t}: {len(r['unpaired_or_unscored'])} item(s) without a pair")
        verdict = "inconclusive" if reasons else "pass"
    return {"margins": MARGINS, "fail_alpha": FAIL_ALPHA, "tasks": tasks,
            "verdict": verdict, "verdict_reasons": reasons}


# ---------------------------------------------------------------- image ids

def _norm_id(s: str) -> str:
    s = (s or "").strip().lower()
    return s[len("sha256:"):] if s.startswith("sha256:") else s


def same_image_id(a: str, b: str) -> bool:
    a, b = _norm_id(a), _norm_id(b)
    if a == b:
        return True
    return min(len(a), len(b)) >= ID_PREFIX_MIN and (a.startswith(b) or b.startswith(a))


def image_refusals(ids: dict[str, dict[str, str]]) -> list[str]:
    for arm, d in ids.items():
        for k, v in d.items():
            if not _norm_id(v):
                raise BadInput(f"the {arm} arm's {k} is empty")
    out = []
    if same_image_id(ids[OLD_LABEL]["tag_id"], ids[NEW_LABEL]["tag_id"]):
        out.append("both arms' tags name the same image id: one build ran twice")
    for arm, d in ids.items():
        if not same_image_id(d["tag_id"], d["served_id"]):
            out.append(f"the {arm} arm was served by image {d['served_id']}, "
                       f"not its tag's {d['tag_id']}")
    return out


# ---------------------------------------------------------------- speed

def _geomean(ratios: list[float]) -> float:
    return math.exp(math.fsum(math.log(r) for r in ratios) / len(ratios))


def read_latency(path: Path) -> dict[str, dict]:
    """{id: record} for prompts 1-39 that succeeded."""
    out: dict[str, dict] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError as e:
        raise BadInput(f"{path}: {e}") from e
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise BadInput(f"{path}:{n}: {e}") from e
        i = rec.get("i")
        if not isinstance(i, int) or not SPEED_FIRST_I <= i <= SPEED_LAST_I or rec.get("error"):
            continue
        rid = str(rec.get("id"))
        if rid in out:
            raise BadInput(f"{path}: prompt id {rid} appears twice")
        out[rid] = rec
    return out


def _decode_rate(rec: dict) -> float | None:
    tok, sec = rec.get("effective_tokens"), rec.get("stream_seconds")
    if rec.get("token_source") != "usage" or not tok or tok < MIN_DECODE_TOKENS or not sec or sec <= 0:
        return None
    return tok / sec


def _ratio_summary(ratios: list[float]) -> dict:
    if not ratios:
        return {"n": 0, "geomean_ratio": None, "boot95": None}
    lo, hi = L.boot_ci(ratios, _geomean, reps=BOOT_REPS)
    return {"n": len(ratios), "geomean_ratio": _geomean(ratios), "boot95": [lo, hi]}


def speed_ratios(old: dict[str, dict], new: dict[str, dict]) -> dict:
    """new/old ratios per prompt, paired by id."""
    ids = sorted(set(old) & set(new))
    ttft = [new[i]["ttft_ms"] / old[i]["ttft_ms"] for i in ids
            if (old[i].get("ttft_ms") or 0) > 0 and (new[i].get("ttft_ms") or 0) > 0]
    decode = []
    for i in ids:
        ro, rn = _decode_rate(old[i]), _decode_rate(new[i])
        if ro and rn:
            decode.append(rn / ro)
    return {"prompts_paired": len(ids), "ttft": _ratio_summary(ttft),
            "decode_rate": _ratio_summary(decode)}


def mtp_acceptance(row: dict, base: Path) -> dict:
    keys = ("metrics_before", "metrics_after", "started_before", "started_after")
    if not all(row.get(k) for k in keys):
        return {"value": None, "missing": "no /metrics read-out given"}
    if row["started_before"] != row["started_after"]:
        return {"value": None, "missing": "the engine was relaunched during the run"}
    try:
        before = parse_prometheus_counters((base / row["metrics_before"]).read_text())
        after = parse_prometheus_counters((base / row["metrics_after"]).read_text())
    except OSError as e:
        raise BadInput(str(e)) from e
    if not all(k in d for k in (ACCEPTED, DRAFTS) for d in (before, after)):
        return {"value": None, "missing": "a speculative-decoding counter is absent"}
    drafts = after[DRAFTS] - before[DRAFTS]
    if drafts <= 0:
        return {"value": None, "missing": "no draft was made during the run"}
    accepted = after[ACCEPTED] - before[ACCEPTED]
    return {"value": 1.0 + accepted / drafts, "drafts": drafts, "accepted_tokens": accepted}


def _load_manifest(path: Path) -> tuple[dict, Path]:
    try:
        m = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise BadInput(f"{path}: {e}") from e
    if not isinstance(m.get("rows"), dict):
        raise BadInput(f"{path}: no 'rows' object")
    return m["rows"], path.parent


def speed(old_manifest: Path, new_manifest: Path) -> dict:
    (old_rows, old_base), (new_rows, new_base) = _load_manifest(old_manifest), _load_manifest(new_manifest)
    rows = {}
    for row in sorted(set(old_rows) & set(new_rows)):
        o, n = old_rows[row], new_rows[row]
        if not o.get("latency") or not n.get("latency"):
            raise BadInput(f"row {row}: a 'latency' file is required in both manifests")
        rows[row] = {**speed_ratios(read_latency(old_base / o["latency"]),
                                    read_latency(new_base / n["latency"])),
                     "mtp_acceptance_length": {OLD_LABEL: mtp_acceptance(o, old_base),
                                               NEW_LABEL: mtp_acceptance(n, new_base)}}
    return {"rows": rows, "unpaired_rows": sorted(set(old_rows) ^ set(new_rows)),
            "prompts": [SPEED_FIRST_I, SPEED_LAST_I], "min_decode_tokens": MIN_DECODE_TOKENS}


def accept(args: argparse.Namespace) -> tuple[dict, int]:
    ids = {OLD_LABEL: {"tag_id": args.old_tag_id, "served_id": args.old_served_id},
           NEW_LABEL: {"tag_id": args.new_tag_id, "served_id": args.new_served_id}}
    report = {"kind": "vllm_upgrade_accept", "builds": {"old": OLD_LABEL, "new": NEW_LABEL},
              "images": ids}
    refusals = image_refusals(ids)
    if refusals:
        return {**report, "verdict": "refused", "refusals": refusals}, EXIT_REFUSED
    report["quality"] = quality(load_build_logs(args.old_logs, OLD_LABEL),
                                load_build_logs(args.new_logs, NEW_LABEL))
    report["verdict"] = report["quality"]["verdict"]
    if args.old_speed or args.new_speed:
        if not (args.old_speed and args.new_speed):
            raise BadInput("--old-speed and --new-speed go together")
        report["speed"] = speed(args.old_speed, args.new_speed)
    return report, EXIT_OK


# ---------------------------------------------------------------- retention

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
    a = sub.add_parser("accept", help="quality, speed and MTP acceptance: 0.28.0 against 0.30.0")
    a.add_argument("--old-logs", type=Path, nargs="+", required=True)
    a.add_argument("--new-logs", type=Path, nargs="+", required=True)
    for arm in ("old", "new"):
        a.add_argument(f"--{arm}-tag-id", required=True)
        a.add_argument(f"--{arm}-served-id", required=True)
        a.add_argument(f"--{arm}-speed", type=Path, default=None)
    r = sub.add_parser("retention", help="prefix-cache retention rule (a)-(b)")
    r.add_argument("--default", type=Path, nargs="+", required=True)
    r.add_argument("--none", type=Path, nargs="+", required=True)
    for p in (a, r):
        p.add_argument("--out", type=Path, default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report, code = (accept if args.cmd == "accept" else retention)(args)
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
