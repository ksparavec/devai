#!/usr/bin/env python3
"""Scan every v2-scored HumanEval / HumanEval+ log for failures caused by
the v2 fence regex swallowing the first line's indentation.

Usage: python3 bench_scorer_scan.py <logs_extracted.json> <inspect-log-dir> <out.json>

Parse-only (compile()); no model-generated code is executed. A failure is an
"indentation-strip candidate" when (1) the scored v2 program does not parse,
and (2) the same last fenced block extracted WITHOUT consuming the leading
indentation does parse. Such an item might have passed with a correct
extractor; whether it would pass the tests is unknown without execution.
"""
import json
import re
import sys
from pathlib import Path

import bench_stats as A

FIXED_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def fixed_clean(text, entry_point=""):
    if not text:
        return ""
    cleaned = A.V2_THINK.sub("", text).strip()
    fences = FIXED_FENCE.findall(cleaned)
    if fences:
        return fences[-1].rstrip("\n")
    return A.v2_clean(text, entry_point)


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)  # usage
    runs = A.load_runs(Path(sys.argv[1]))
    logdir = Path(sys.argv[2])
    out = []
    for r in runs:
        if r["task"] not in ("humaneval", "humaneval_plus") or r["created"] < "2026-05-02T19:30":
            continue
        smp = A.read_samples(logdir, r["file"])
        cand = []
        n_fail = 0
        for s in smp:
            if s["value"] == 1.0:
                continue
            n_fail += 1
            c2 = A.v2_clean(s["text"], s["entry_point"])
            if A.compiles(A.program(s, c2)):
                continue
            cf = fixed_clean(s["text"], s["entry_point"])
            if cf != c2 and A.compiles(A.program(s, cf)):
                cand.append(s["id"])
        out.append({"run": A.run_id(r), "task": r["task"], "model": r["model"], "backend": r["backend"],
                    "created": r["created"], "x": int(r["x"]), "n": r["n"], "failures": n_fail,
                    "indent_strip_candidates": len(cand), "candidate_ids": cand})
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
    tot_f = sum(o["failures"] for o in out)
    tot_c = sum(o["indent_strip_candidates"] for o in out)
    print(f"{len(out)} v2-scored runs; failures {tot_f}; indentation-strip candidates {tot_c}")
    for o in out:
        if o["indent_strip_candidates"]:
            print(f"  {o['created'][:16]} {o['model'][:45]:45} {o['backend']:10} {o['task']:15} {o['x']}/{o['n']} cand={o['indent_strip_candidates']}")


if __name__ == "__main__":
    main()
