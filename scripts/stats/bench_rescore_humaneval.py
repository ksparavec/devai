#!/usr/bin/env python3
"""Re-execute every logged v2-scored HumanEval / HumanEval+ answer with the
FIXED completion extractor (scripts/bench/tasks/humaneval.py, 2026-09-27)
and report which scores the v2 extractor got wrong.

Usage (inside the lab image: this EXECUTES model-generated code, in the
bench scorer's own subprocess sandbox, and imports inspect_ai):

  podman run --rm --network=none --entrypoint python3 \\
    -v $PWD:/repo:ro -v /var/cache/devai/bench/inspect-logs:/logs:ro \\
    -v $O:/o localhost/devai-lab-gpu:latest \\
    /repo/scripts/stats/bench_rescore_humaneval.py /o/logs_extracted.json /logs /o/humaneval_rescore.json

Only answers whose extraction CHANGES are executed; every other item keeps
its logged score. Reports, per run, fail->pass and pass->fail flips, the
cause of each flip (unfenced body dedented by str.strip(), or a fenced
body dedented by the fence regex), and the port-11435 vs port-11437
AutoRound HumanEval pair (docs/router.md) before and after.
"""
import json
import re
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent)]

import bench_stats as A  # noqa: E402
import statlib as L  # noqa: E402
from bench.tasks import humaneval as H  # noqa: E402


def fixed_outcomes(logdir: Path, run: dict) -> tuple[dict, list[dict]]:
    """Per-item pass/fail under the fixed extractor, plus the flips."""
    out, flips = {}, []
    for s in A.read_samples(logdir, run["file"]):
        old = s["value"] == 1.0
        c_old = A.v2_clean(s["text"], s["entry_point"])
        c_new = H._clean_completion(s["text"], s["entry_point"])
        new = old if c_new == c_old else H._run_check_in_subprocess(A.program(s, c_new))[0]
        out[s["id"]] = new
        if new != old:
            fence = re.search(r"```([^\n]*)\n", A.V2_THINK.sub("", s["text"]))
            flips.append({"id": s["id"], "to": "pass" if new else "fail",
                          "cause": "unfenced body" if fence is None else "fenced body",
                          "v2_program_compiles": A.compiles(A.program(s, c_old))})
    return out, flips


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit(__doc__)  # usage
    runs = [r for r in A.load_runs(Path(sys.argv[1]))
            if r["task"] in ("humaneval", "humaneval_plus") and r["created"] >= "2026-05-02T19:30"]
    logdir = Path(sys.argv[2])
    report, outcomes = [], {}
    for r in runs:
        o, flips = fixed_outcomes(logdir, r)
        outcomes[r["file"]] = o
        report.append({"run": A.run_id(r), "task": r["task"], "model": r["model"],
                       "backend": r["backend"], "created": r["created"], "n": r["n"],
                       "x_v2": int(r["x"]), "x_fixed": sum(o.values()), "flips": flips})
    pair = {}
    ar = [r for r in runs if "AutoRound" in r["model"] and r["task"] == "humaneval"
          and r["created"].startswith("2026-09-22")]
    v = next((r for r in ar if r["backend"] == "vllm"), None)
    d = next((r for r in ar if r["backend"] == "vllm-devai"), None)  # the first, pre-fix run
    if v and d:
        for label, a, b in (("v2", {s["id"]: s["value"] == 1.0 for s in A.read_samples(logdir, v["file"])},
                                   {s["id"]: s["value"] == 1.0 for s in A.read_samples(logdir, d["file"])}),
                            ("fixed", outcomes[v["file"]], outcomes[d["file"]])):
            ids = sorted(set(a) & set(b), key=int)
            bb = sum(a[i] and not b[i] for i in ids)
            cc = sum(b[i] and not a[i] for i in ids)
            pair[label] = {"shared": len(ids), "vllm": sum(a[i] for i in ids),
                           "vllm_devai": sum(b[i] for i in ids), "b": bb, "c": cc,
                           "p_mcnemar": L.mcnemar_exact(bb, cc)}
    Path(sys.argv[3]).write_text(json.dumps({"runs": report, "autoround_pair": pair}, indent=1))
    flips = [f for rr in report for f in rr["flips"]]
    f2p = [f for f in flips if f["to"] == "pass"]
    print(f"{len(report)} runs; {sum(bool(rr['flips']) for rr in report)} with flips; "
          f"fail->pass {len(f2p)} (unfenced body {sum(f['cause'] == 'unfenced body' for f in f2p)}, "
          f"fenced body {sum(f['cause'] == 'fenced body' for f in f2p)}; "
          f"v2 program compiled {sum(f['v2_program_compiles'] for f in f2p)}); "
          f"pass->fail {len(flips) - len(f2p)}")
    for rr in report:
        if rr["flips"]:
            print(f"  {rr['created'][:16]} {rr['model'][:45]:45} {rr['backend']:10} {rr['task']:15} "
                  f"{rr['x_v2']}/{rr['n']} -> {rr['x_fixed']}/{rr['n']}")
    for k, p in pair.items():
        print(f"AutoRound HumanEval, port 11435 vs 11437 ({k}): {p}")


if __name__ == "__main__":
    main()
