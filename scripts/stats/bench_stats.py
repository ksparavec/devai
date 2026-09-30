#!/usr/bin/env python3
"""Statistical re-analysis of the bench-leaderboard claims (devai repo).

Usage:
  python3 bench_stats.py <logs_extracted.json> <inspect-log-dir> <bench-cache.json> <out-dir>

Inputs are read-only. Writes <out-dir>/results.json (every computed
number) and <out-dir>/results.md (tables). Stdlib only; deterministic
(all resampling uses fixed seeds). Depends on statlib.py (same dir) and
on logs_extracted.json produced by bench_extract_logs.py.
"""
from __future__ import annotations

import json
import math
import re
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import statlib as S

SEED = 20260927
B = 10000

PORT_BACKEND = {"11434": "ollama", "11435": "vllm", "11436": "sglang",
                "11437": "vllm-devai", "11438": "laya"}
TASK_SHORT = {"gsm8k_task": "gsm8k", "humaneval_task": "humaneval",
              "humaneval_plus_task": "humaneval_plus", "mmlu_pro_task": "mmlu_pro",
              "gpqa_task": "gpqa", "tools_use_task": "tools"}
# Population sizes of the full benchmark split the subset is drawn from.
# gsm8k 1319 and humaneval 164 are printed by the 2026-05-05 run log
# ("Generating test split ... 1319/1319", "164/164"); gpqa-diamond 198 and
# MMLU-Pro test 12032 are the published split sizes (not re-verified
# offline); tools = the 20 hand-written prompts in tools_prompts.jsonl.
POP = {"gsm8k": 1319, "humaneval": 164, "humaneval_plus": 164,
       "mmlu_pro": 12032, "gpqa": 198, "tools": 20}
SELECTION = {
    "gsm8k": "first n of the 1319-item test split (shuffle=False): fixed, non-random items",
    "humaneval": "first n of 164 problems (shuffle=False): fixed, non-random items",
    "humaneval_plus": "first n of 164 problems (shuffle=False): fixed, non-random items",
    "mmlu_pro": "HF shuffle(seed=42) then first n: one simple random sample w/o replacement, reused for every model",
    "gpqa": "HF shuffle(seed=42) then first n of 198: one simple random sample w/o replacement, reused for every model",
    "tools": "first n of 20 hand-written prompts (n=20 is the whole set)",
}


def ts(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def r4(x):
    return None if x is None else round(x, 4)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def served_model_name(logged: str) -> str:
    """The router alias from an inspect log's model: `openai/<alias>` (the
    harness until 2026-09-27) or `openai-api/<service>/<alias>` (after)."""
    return re.sub(r"^(?:openai-api/[^/]+|openai)/", "", logged)


def load_runs(path: Path) -> list[dict]:
    runs = []
    for r in json.loads(path.read_text()):
        if r.get("status") != "success":
            continue
        task = TASK_SHORT.get(r.get("task"))
        m = re.search(r":(\d+)/", r.get("model_base_url") or "")
        items = sorted(r.get("items") or [], key=lambda i: int(i["id"]))
        vals = [i["value"] for i in items]
        if not items or all(v is None for v in vals):
            continue  # an aborted run (every sample errored) is not a measurement
        runs.append({
            "file": r["file"], "created": r["created"], "completed": r.get("completed_at"),
            "task": task, "model": served_model_name(r.get("model") or ""),
            "backend": PORT_BACKEND.get(m.group(1)) if m else None,
            "n": len(items), "x": sum(v for v in vals if v is not None),
            "n_unscored": sum(v is None for v in vals),
            "vals": [0.0 if v is None else v for v in vals],
            "keys": [i.get("item_key") for i in items],
            "subcases": [i.get("subcase") for i in items],
            # inspect_ai sample limit hit (time_limit / message_limit): the
            # sample was cut off and scored 0 -- a censored outcome.
            "limits": [bool(i.get("limit")) for i in items],
            "task_args": r.get("task_args") or {},
            "accuracy_inspect": r.get("accuracy"),
        })
    runs.sort(key=lambda r: r["created"])
    return runs


def find(runs, task, model, backend=None, n=None, created=None, created_before=None,
         created_after=None):
    out = []
    for r in runs:
        if r["task"] != task or r["model"] != model:
            continue
        if backend and r["backend"] != backend:
            continue
        if n and r["n"] != n:
            continue
        if created and not r["created"].startswith(created):
            continue
        if created_before and r["created"] >= created_before:
            continue
        if created_after and r["created"] < created_after:
            continue
        out.append(r)
    return out


def one(runs, *a, **k):
    got = find(runs, *a, **k)
    if len(got) != 1:
        raise RuntimeError(f"expected exactly one run for {a} {k}, got {len(got)}: "
                           f"{[g['file'] for g in got]}")
    return got[0]


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------

def prop_summary(x: int, n: int, task: str, t: int | None = None) -> dict:
    cp = S.clopper_pearson(int(x), n)
    wl = S.wilson(int(x), n)
    out = {"x": int(x), "n": n, "estimate": r4(x / n), "cp95": [r4(cp[0]), r4(cp[1])],
           "wilson95": [r4(wl[0]), r4(wl[1])], "selection": SELECTION.get(task, "")}
    if t is not None:
        out["time_limit_hits"] = int(t)
        if t:
            done = n - t
            cpd = S.clopper_pearson(int(x), done) if done else (0.0, 1.0)
            out["accuracy_among_completed"] = r4(x / done) if done else None
            out["accuracy_among_completed_cp95"] = [r4(cpd[0]), r4(cpd[1])]
            out["range_if_censored_items_all_wrong_or_all_right"] = [r4(x / n), r4((x + t) / n)]
    N = POP.get(task)
    if task in ("gpqa", "mmlu_pro") and N and n < N:
        h = S.hypergeometric_interval(int(x), n, N) if N <= 2000 else None
        out["fpc_factor"] = r4(math.sqrt((N - n) / (N - 1)))
        if h:
            out["fpc_exact95_deterministic_only"] = [r4(h[0]), r4(h[1])]
        out["population_N"] = N
    return out


def paired_table(va: list[float], vb: list[float]) -> dict:
    """2x2 for paired binary outcomes (A vs B on the same items)."""
    a = b = c = d = 0
    for x, y in zip(va, vb):
        x, y = int(round(x)), int(round(y))
        if x and y:
            a += 1
        elif x and not y:
            b += 1
        elif (not x) and y:
            c += 1
        else:
            d += 1
    n = a + b + c + d
    diff, lo, hi = S.newcombe_paired(a, b, c, d)
    return {"n": n, "a_both": a, "b_only_A": b, "c_only_B": c, "d_neither": d,
            "pA": r4((a + b) / n), "pB": r4((a + c) / n),
            "diff_A_minus_B": r4(diff), "newcombe10_95": [r4(lo), r4(hi)],
            "mcnemar_exact_p": r4(S.mcnemar_exact(b, c)),
            "discordant_fraction": r4((b + c) / n)}


def compare_runs(ra: dict, rb: dict, label: str, note: str = "") -> dict:
    """Paired comparison on the common prefix of items (verified identical)."""
    k = min(ra["n"], rb["n"])
    if ra["keys"][:k] != rb["keys"][:k]:
        raise RuntimeError(f"item mismatch for {label}")
    out = paired_table(ra["vals"][:k], rb["vals"][:k])
    out["time_limit_hits_A_B"] = [sum(ra["limits"][:k]), sum(rb["limits"][:k])]
    out.update({"label": label, "A": run_id(ra), "B": run_id(rb),
                "A_full": f"{int(ra['x'])}/{ra['n']}", "B_full": f"{int(rb['x'])}/{rb['n']}",
                "paired_on_first": k, "note": note})
    return out


def run_id(r: dict) -> str:
    return f"{r['model']} [{r['backend']}] {r['task']} n={r['n']} {r['created'][:16]} ({r['file'][:60]})"


# --------------------------------------------------------------------------
# Section A-C: the 2026-05-05 leaderboard (docs/bench-results.md)
# --------------------------------------------------------------------------

SWEEP_0505 = [  # doc order (sorted by aggregate), served names
    "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4@131072",
    "Qwen3-14B-NVFP4@65536",
    "Qwen3-8B-NVFP4@131072",
    "gpt-oss-20b@262144",
    "Qwen3.5-9B-NVFP4@131072",
    "DeepSeek-R1-Distill-Qwen-7B@65536",
    "Llama-3.1-8B-Instruct-NVFP4@131072",
    "DeepSeek-R1-Distill-Llama-8B@32768",
    "NVIDIA-Nemotron-Nano-9B-v2-NVFP4@65536",
]
# (gsm8k, humaneval, tools, subcases E/S/M/F, aggregate) as printed in the doc.
DOC_0505 = {
    "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4@131072": (0.99, 0.98, 1.00, (1.0, 1.0, 1.0, 1.0), 0.99),
    "Qwen3-14B-NVFP4@65536": (0.98, 0.94, 1.00, (1.0, 1.0, 1.0, 1.0), 0.97),
    "Qwen3-8B-NVFP4@131072": (0.97, 0.88, 0.95, (1.0, 1.0, 1.0, 0.8), 0.93),
    "gpt-oss-20b@262144": (0.97, 0.98, 0.85, (0.4, 1.0, 1.0, 1.0), 0.93),
    "Qwen3.5-9B-NVFP4@131072": (0.98, 0.78, 0.95, (0.8, 1.0, 1.0, 1.0), 0.90),
    "DeepSeek-R1-Distill-Qwen-7B@65536": (0.87, 0.94, 0.60, (1.0, 0.4, 0.6, 0.4), 0.80),
    "Llama-3.1-8B-Instruct-NVFP4@131072": (0.78, 0.80, 0.70, (1.0, 0.4, 1.0, 0.4), 0.76),
    "DeepSeek-R1-Distill-Llama-8B@32768": (0.63, 0.00, 0.80, (1.0, 0.6, 1.0, 0.6), 0.48),
    "NVIDIA-Nemotron-Nano-9B-v2-NVFP4@65536": (0.74, 0.06, 0.00, (0.0, 0.0, 0.0, 0.0), 0.27),
}
SUBCASES = ["empty_schema", "single_arg", "multi_tool_pick", "result_followup"]


def sweep_runs(runs):
    """The 2026-05-05 14:03Z invocation (run log bench-vllm-run-20260505T140328Z.log)."""
    sel = {}
    for m in SWEEP_0505:
        sel[m] = {t: one(runs, t, m, backend="vllm", created_after="2026-05-05T14:00",
                         created_before="2026-05-05T17:30")
                  for t in ("gsm8k", "humaneval", "tools")}
    return sel


def section_leaderboard(runs):
    sel = sweep_runs(runs)
    rows = []
    discrepancies = []
    for m in SWEEP_0505:
        g, h, t = sel[m]["gsm8k"], sel[m]["humaneval"], sel[m]["tools"]
        est, lo, hi = S.stratified_bootstrap_mean_of_means([g["vals"], h["vals"], t["vals"]], reps=B, seed=SEED)
        # Analytic SE of the unweighted mean of three independent proportions.
        se = math.sqrt(sum(r["x"] / r["n"] * (1 - r["x"] / r["n"]) / r["n"] for r in (g, h, t))) / 3
        subs = {}
        for sc in SUBCASES:
            v = [val for val, s in zip(t["vals"], t["subcases"]) if s == sc]
            subs[sc] = prop_summary(sum(v), len(v), "tools")
        row = {"model": m,
               "gsm8k": prop_summary(g["x"], g["n"], "gsm8k", sum(g["limits"])),
               "humaneval": prop_summary(h["x"], h["n"], "humaneval", sum(h["limits"])),
               "tools": prop_summary(t["x"], t["n"], "tools", sum(t["limits"])),
               "tools_subcases": subs,
               "aggregate": {"estimate": r4(est), "bootstrap95": [r4(lo), r4(hi)],
                             "analytic_se": r4(se),
                             "wald95": [r4(est - 1.96 * se), r4(min(1.0, est + 1.96 * se))],
                             "definition": "unweighted mean of three proportions with n=100, 50, 20"},
               "logs": {k: v["file"] for k, v in sel[m].items()}}
        rows.append(row)
        doc = DOC_0505[m]
        got = (g["x"] / g["n"], h["x"] / h["n"], t["x"] / t["n"])
        for name, dv, gv in zip(("gsm8k", "humaneval", "tools"), doc[:3], got):
            if abs(dv - gv) > 0.005:
                discrepancies.append({"model": m, "field": name, "doc": dv, "log": gv})
        for sc, dv in zip(SUBCASES, doc[3]):
            if abs(dv - subs[sc]["estimate"]) > 0.005:
                discrepancies.append({"model": m, "field": sc, "doc": dv, "log": subs[sc]["estimate"]})
        if abs(doc[4] - est) > 0.0051:
            discrepancies.append({"model": m, "field": "aggregate", "doc": doc[4], "log": est})
    return sel, rows, discrepancies


def section_ranking(sel):
    """Every pairwise comparison between the 9 rows on the aggregate (paired
    sign-flip randomization test, B=10000) and per task (exact McNemar)."""
    names = SWEEP_0505
    agg_pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, Bm = sel[names[i]], sel[names[j]]
            diffs = [[a - b for a, b in zip(A[t]["vals"], Bm[t]["vals"])]
                     for t in ("gsm8k", "humaneval", "tools")]
            obs, p = S.paired_signflip_test(diffs, reps=B, seed=SEED)
            est, lo, hi = S.paired_bootstrap_diff([A[t]["vals"] for t in ("gsm8k", "humaneval", "tools")],
                                                  [Bm[t]["vals"] for t in ("gsm8k", "humaneval", "tools")],
                                                  reps=B, seed=SEED)
            agg_pairs.append({"A": names[i], "B": names[j], "rank_gap": j - i,
                              "diff": r4(obs), "bootstrap95": [r4(lo), r4(hi)], "p_perm": r4(p)})
    adj = S.holm_list([p["p_perm"] for p in agg_pairs])
    for p, a in zip(agg_pairs, adj):
        p["p_holm_all36"] = r4(a)
    adjacent = [p for p in agg_pairs if p["rank_gap"] == 1]
    adj2 = S.holm_list([p["p_perm"] for p in adjacent])
    for p, a in zip(adjacent, adj2):
        p["p_holm_adjacent8"] = r4(a)
    # Per-task: leader vs every other row.
    leader = names[0]
    per_task = []
    for t in ("gsm8k", "humaneval", "tools"):
        fam = []
        for other in names[1:]:
            c = paired_table(sel[leader][t]["vals"], sel[other][t]["vals"])
            c.update({"task": t, "A": leader, "B": other})
            fam.append(c)
        per_task.extend(fam)
    adj3 = S.holm_list([c["mcnemar_exact_p"] for c in per_task])
    for c, a in zip(per_task, adj3):
        c["p_holm_24"] = r4(a)
    return {"aggregate_pairs": agg_pairs, "adjacent": adjacent, "leader_vs_others_per_task": per_task}


THRESH = {"gsm8k": 0.9, "humaneval": 0.7, "tools": 0.9}


def section_thresholds(rows):
    out = []
    for row in rows:
        rec = {"model": row["model"]}
        for t, thr in THRESH.items():
            lo, hi = row[t]["cp95"]
            est = row[t]["estimate"]
            if lo >= thr:
                verdict = "CI entirely >= threshold"
            elif hi < thr:
                verdict = "CI entirely < threshold"
            else:
                verdict = "CI straddles threshold (classification not determined by the data)"
            rec[t] = {"estimate": est, "cp95": [lo, hi], "threshold": thr,
                      "point_meets": est >= thr, "verdict": verdict}
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# Completions (static scorer analysis; no model code is executed)
# --------------------------------------------------------------------------

V1_FENCE = re.compile(r"^```(?:python)?\s*\n(.*?)\n```\s*$", re.DOTALL)
V2_FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)
V2_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def v1_clean(text: str) -> str:
    if not text:
        return ""
    m = V1_FENCE.search(text.strip())
    return m.group(1) if m else text


def v2_clean(text: str, entry_point: str = "") -> str:
    if not text:
        return ""
    cleaned = V2_THINK.sub("", text).strip()
    fences = V2_FENCE.findall(cleaned)
    if fences:
        return fences[-1].rstrip("\n")
    if entry_point:
        m = re.search(rf"^def\s+{re.escape(entry_point)}\s*\(", cleaned, flags=re.MULTILINE)
        if m:
            return cleaned[m.start():]
    return cleaned


def read_samples(logdir: Path, fname: str) -> list[dict]:
    z = zipfile.ZipFile(logdir / fname)
    out = []
    for nm in z.namelist():
        if not nm.startswith("samples/"):
            continue
        s = json.loads(z.read(nm))
        att = s.get("attachments") or {}

        def res(v):
            if isinstance(v, str) and v.startswith("attachment://"):
                return att.get(v[len("attachment://"):], "")
            return v if isinstance(v, str) else ""

        msg = ((s.get("output") or {}).get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content")
        text, reasoning = "", ""
        if isinstance(content, str):
            text = res(content)
        elif isinstance(content, list):
            text = "".join(res(p.get("text", "")) for p in content if p.get("type") == "text")
            reasoning = "".join(res(p.get("reasoning", "")) for p in content if p.get("type") == "reasoning")
        sc = next(iter((s.get("scores") or {}).values()), {}) or {}
        meta = s.get("metadata") or {}
        out.append({"id": int(s["id"]), "text": text, "reasoning": reasoning,
                    "value": sc.get("value"), "explanation": sc.get("explanation") or "",
                    "entry_point": meta.get("entry_point", ""),
                    "prompt": res(s.get("input")), "test": res(meta.get("test", ""))})
    out.sort(key=lambda d: d["id"])
    return out


def compiles(src: str) -> bool:
    """Parse-only check (compile() never executes the code)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            compile(src, "<humaneval-static>", "exec")
            return True
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            return False


def program(sample: dict, completion: str) -> str:
    # Same concatenation as both scorer versions.
    return (sample["prompt"] + "\n" + completion + "\n\n" + sample["test"]
            + f"\n\ncheck({sample['entry_point']})\n")


def classify_failure(expl: str) -> str:
    e = expl or ""
    if "SyntaxError" in e or "IndentationError" in e or "invalid character" in e:
        return "syntax/indentation error"
    if "AssertionError" in e:
        return "assertion (wrong answer)"
    if "timeout" in e.lower() or "TimeoutError" in e:
        return "timeout"
    if "NameError" in e:
        return "NameError"
    if not e.strip():
        return "(no explanation)"
    return "other runtime error"


def section_scorer_static(runs, logdir: Path):
    """Test the doc's attribution of R1-Qwen-7B's 0.26 -> 0.94 to the scorer fix,
    and the 'no-op for already-passing models' claim, by applying the v1 and
    v2 completion cleaners (text processing only) to the retained completions."""
    res = {}
    # v1-scored HumanEval runs (2026-05-02, before commit ad597ca at 20:09Z).
    v1_runs = [r for r in runs if r["task"] == "humaneval" and r["n"] == 50
               and r["created"] < "2026-05-02T19:30"]
    per_run = []
    for r in v1_runs:
        smp = read_samples(logdir, r["file"])
        n_pass = sum(1 for s in smp if s["value"] == 1.0)
        fails = [s for s in smp if s["value"] != 1.0]
        cls = defaultdict(int)
        changed_fail = changed_pass = 0
        pass_v2_breaks = 0          # v1 passed, v2 program no longer parses
        pass_v2_diff_nonws = 0      # v1 passed, v2 program differs beyond trailing whitespace
        fail_syntax_v2_parses = 0   # v1 failed with a syntax error, v2 program parses
        fail_syntax = 0
        for s in smp:
            c1 = v1_clean(s["text"])
            c2 = v2_clean(s["text"], s["entry_point"])
            ch = c1 != c2
            p1, p2 = program(s, c1), program(s, c2)
            if s["value"] == 1.0:
                changed_pass += ch
                if not compiles(p2):
                    pass_v2_breaks += 1
                if c1.rstrip() != c2.rstrip():
                    pass_v2_diff_nonws += 1
            else:
                changed_fail += ch
                k = classify_failure(s["explanation"])
                cls[k] += 1
                if k == "syntax/indentation error":
                    fail_syntax += 1
                    fail_syntax_v2_parses += compiles(p2)
        per_run.append({"run": run_id(r), "pass": n_pass, "n": len(smp),
                        "fail_classes": dict(cls),
                        "failed_items_where_v2_changes_program": changed_fail,
                        "passed_items_where_v2_changes_program": changed_pass,
                        "passed_items_v2_differs_beyond_trailing_ws": pass_v2_diff_nonws,
                        "passed_items_where_v2_program_fails_to_parse": pass_v2_breaks,
                        "syntax_failures": fail_syntax,
                        "syntax_failures_where_v2_program_parses": fail_syntax_v2_parses})
    res["v1_scored_runs"] = per_run
    # Cross-tab for R1-Qwen-7B: 2026-05-02 (v1) vs 2026-05-05 (v2).
    a = one(runs, "humaneval", "DeepSeek-R1-Distill-Qwen-7B@65536", created="2026-05-02")
    b = one(runs, "humaneval", "DeepSeek-R1-Distill-Qwen-7B@65536", created="2026-05-05")
    sa = read_samples(logdir, a["file"])
    sb = read_samples(logdir, b["file"])
    xt = defaultdict(int)
    for s1, s2 in zip(sa, sb):
        fixable = (s1["value"] != 1.0) and v1_clean(s1["text"]) != v2_clean(s1["text"], s1["entry_point"])
        k = ("0502 pass" if s1["value"] == 1.0 else ("0502 fail, v2 would change program" if fixable
                                                     else "0502 fail, v2 same program"))
        xt[f"{k} | 0505 {'pass' if s2['value'] == 1.0 else 'fail'}"] += 1
    res["r1_qwen_crosstab"] = dict(xt)
    res["r1_qwen_mcnemar"] = compare_runs(b, a, "R1-Qwen-7B HumanEval 05-05 (v2) vs 05-02 (v1)")
    # Byte-level BPE markers (U+0120 space marker, U+010A newline marker).
    bpe = []
    for model in ("DeepSeek-R1-Distill-Llama-8B@32768", "DeepSeek-R1-Distill-Qwen-7B@65536"):
        for r in find(runs, "humaneval", model, n=50):
            smp = read_samples(logdir, r["file"])
            has = sum(1 for s in smp if ("\u0120" in s["text"] or "\u010a" in s["text"]))
            has_r = sum(1 for s in smp if ("\u0120" in s["reasoning"] or "\u010a" in s["reasoning"]))
            cls = defaultdict(int)
            for s in smp:
                if s["value"] != 1.0:
                    cls[classify_failure(s["explanation"])] += 1
            bpe.append({"run": run_id(r), "pass": sum(1 for s in smp if s["value"] == 1.0),
                        "n": len(smp), "text_with_bpe_markers": has,
                        "reasoning_with_bpe_markers": has_r, "fail_classes": dict(cls)})
    res["bpe_markers"] = bpe
    return res


# --------------------------------------------------------------------------
# Incidental replications and change claims
# --------------------------------------------------------------------------

# Documented changes (UTC) that break a "replicate" between two runs. Sources:
# git log (commit times converted to UTC) and the bench-results / backends docs.
KNOWN_CHANGES = [
    ("2026-05-02T18:45", "tools", None, None, "tools_use solver: tool_choice=auto via generate() -> pinned tool_loop_with_pin"),
    ("2026-05-02T19:30", "humaneval", None, None, "HumanEval completion cleaner v1 -> v2 (ad597ca)"),
    ("2026-05-05T09:36", None, "NVIDIA-Nemotron-3-Nano", None, "b730985: --enforce-eager dropped, --max-num-seqs 8"),
    ("2026-07-18T10:47", None, None, "ollama", "8148bb1: router recreates Ollama per model at probed ctx, no CPU spill"),
    ("2026-07-18T14:15", "tools", "Gemma-4-26B", "vllm", "3d14333: gemma4 vLLM tool parser curated"),
    ("2026-07-19T00:40", "humaneval_plus", None, None, "HumanEval+ sandbox fixes (bc31964, f6dd31a)"),
    ("2026-07-19T12:02", "tools", "Gemma-4-26B", "vllm", "80d33fa: gemma4 tool parser verified"),
    ("2026-07-20T13:30", None, "Nemotron-Nano-9B-v2", "vllm", "d09887c/09af08b (committed 13:56Z/14:26Z; in use from the 13:37Z rerun): reasoning + tool parser wired, vLLM v0.25.1 pin"),
    ("2026-07-21T09:00", None, "qwen3.6:35b-a3b-mtp", "ollama", "served ctx 65536/f16 -> 131072/q8_0 (different cache row; 8325255 per-tier KV)"),
    ("2026-07-21T15:00", None, "gemma4:26b-a4b-it", "ollama", "served ctx 131072 -> 262144 (different cache row)"),
    ("2026-07-22T07:00", None, "gpt-oss-20b", "vllm", "vLLM launch without --kv-cache-dtype from 07-22 07:02Z (fp8 before; devai-vllm.log)"),
    ("2026-07-26T22:03", None, "Nemotron-Nano-9B-v2", "sglang", "dcd3b45: SGLang reasoning parser wired"),
    ("2026-07-27T19:00", None, None, "sglang", "c9ad978: router SGLang reasoning request path rewritten"),
    ("2026-07-29T05:00", None, None, "sglang", "SGLang image v0.5.10.post1 -> v0.5.16 (committed 5aaf6e5, 14:50Z)"),
    ("2026-09-22T12:00", None, "devai", "vllm-devai", "parser + reasoning-policy fixes for vllm-devai (engine-keyed lookup)"),
]


def changes_between(ra, rb):
    out = []
    for when, task, model_sub, backend, desc in KNOWN_CHANGES:
        if not (ra["created"] < when <= rb["created"]):
            continue
        if task and not (ra["task"] == task or (task == "humaneval" and ra["task"] == "humaneval_plus")):
            continue
        if model_sub and model_sub not in ra["model"]:
            continue
        if backend and ra["backend"] != backend:
            continue
        out.append(desc)
    return out


def section_replications(runs):
    groups = defaultdict(list)
    for r in runs:
        if r["n"] < 20:
            continue  # smoke tests
        groups[(r["model"], r["backend"], r["task"], r["n"], json.dumps(r["task_args"], sort_keys=True))].append(r)
    out = []
    for key, rs in groups.items():
        rs.sort(key=lambda r: r["created"])
        for ra, rb in zip(rs, rs[1:]):
            c = compare_runs(ra, rb, f"{key[0]} [{key[1]}] {key[2]} n={key[3]}")
            c["documented_changes_between"] = changes_between(ra, rb)
            c["clean_replicate"] = not c["documented_changes_between"]
            pbar = (c["pA"] + c["pB"]) / 2
            c["hours_apart"] = r4((ts(rb["created"]) - ts(ra["created"])).total_seconds() / 3600)
            # Under iid replicate runs with per-item success prob p_i:
            # E[discordance] = 2 * mean p_i(1-p_i). Share of the binomial
            # variance pbar(1-pbar) that is decoding (within-item) variance:
            c["decoding_var_est_mean_p_1mp"] = r4(c["discordant_fraction"] / 2)
            c["decoding_share_of_binomial_var"] = (r4(min(1.0, (c["discordant_fraction"] / 2) / (pbar * (1 - pbar))))
                                                   if 0 < pbar < 1 else None)
            out.append(c)
    return out


def section_change_claims(runs, logdir):
    R = {}
    # bench-results.md: gpt-oss tools 0.90 (05-02) -> 0.85 (05-05).
    a = one(runs, "tools", "gpt-oss-20b@262144", created="2026-05-02")
    b = one(runs, "tools", "gpt-oss-20b@262144", created="2026-05-05")
    c = compare_runs(b, a, "gpt-oss-20b tools_use 05-05 vs 05-02",
                     "PROTOCOL CHANGED: 05-02 sent tool_choice=auto via inspect's generate() loop; "
                     "05-05 pinned tool_choice to the expected function (tool_loop_with_pin).")
    es_a = [v for v, s in zip(a["vals"], a["subcases"]) if s == "empty_schema"]
    es_b = [v for v, s in zip(b["vals"], b["subcases"]) if s == "empty_schema"]
    c["empty_schema_0502"] = f"{int(sum(es_a))}/5"
    c["empty_schema_0505"] = f"{int(sum(es_b))}/5"
    R["gptoss_tools_0502_0505"] = c
    # R1-Qwen HumanEval 0.26 -> 0.94.
    R["r1qwen_he_0502_0505"] = compare_runs(
        one(runs, "humaneval", "DeepSeek-R1-Distill-Qwen-7B@65536", created="2026-05-05"),
        one(runs, "humaneval", "DeepSeek-R1-Distill-Qwen-7B@65536", created="2026-05-02"),
        "R1-Qwen-7B HumanEval 05-05 vs 05-02", "scorer v1 (05-02) vs v2 (05-05)")
    # Nemotron-Nano-9B-v2 HumanEval 0/50 (v1, 14:20Z) -> 3/50 (v2 rerun, 19:44Z).
    R["nemotron9b_he_scorerfix"] = compare_runs(
        one(runs, "humaneval", "NVIDIA-Nemotron-Nano-9B-v2-NVFP4@65536", created="2026-05-02T19"),
        one(runs, "humaneval", "NVIDIA-Nemotron-Nano-9B-v2-NVFP4@65536", created="2026-05-02T14"),
        "Nemotron-Nano-9B-v2 HumanEval v2 rerun vs v1 run (2026-05-02)",
        "different completions: the rerun re-generated every answer")
    # Llama-3.1 HumanEval 0.72 (05-02, cited in Issue #1) vs 0.80 (05-05 table).
    R["llama_he_072_080"] = compare_runs(
        one(runs, "humaneval", "Llama-3.1-8B-Instruct-NVFP4@131072", created="2026-05-05"),
        one(runs, "humaneval", "Llama-3.1-8B-Instruct-NVFP4@131072", created="2026-05-02"),
        "Llama-3.1-8B HumanEval 05-05 vs 05-02", "scorer v1 (05-02) vs v2 (05-05)")
    # Nemotron-3-Nano: 09:17Z (--enforce-eager, before b730985) vs 15:18Z (CUDA graphs, max-num-seqs 8).
    for t in ("gsm8k", "humaneval", "tools"):
        R[f"nemotron3_{t}_eager_vs_graphs"] = compare_runs(
            one(runs, t, "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4@131072", created="2026-05-05T15"),
            one(runs, t, "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4@131072", created="2026-05-05T09"),
            f"Nemotron-3-Nano {t}: 15:18Z (graphs) vs 09:17Z (--enforce-eager)",
            "commit b730985 (2026-05-05 09:36Z) between the runs; same items, same harness")
    # Nemotron-Nano-9B-v2: 'genuine coding weakness' vs later measurement.
    R["nemotron9b_he_may_vs_jul"] = compare_runs(
        one(runs, "humaneval", "NVIDIA-Nemotron-Nano-9B-v2-NVFP4@131072", created="2026-07-20T13:37"),
        one(runs, "humaneval", "NVIDIA-Nemotron-Nano-9B-v2-NVFP4@65536", created="2026-05-05"),
        "Nemotron-Nano-9B-v2 HumanEval 2026-07-20 (vLLM, 131072) vs 2026-05-05 (vLLM, 65536)",
        "configuration changed in between (parsers / sampling override / ctx); not a replicate")
    return R


# --------------------------------------------------------------------------
# docs/backends.md and docs/router.md bench claims
# --------------------------------------------------------------------------

def cache_task_run(runs, cache_row, task_key, backend, tol_s=300):
    """Find the log that produced a cache task entry: same backend, task,
    n, model prefix, completion time within tol_s of ran_at."""
    t = cache_row["tasks"][task_key]
    task = ("humaneval_plus" if task_key.startswith("humaneval_plus") else
            "humaneval" if task_key.startswith("humaneval") else
            "gsm8k" if task_key.startswith("gsm8k") else
            "mmlu_pro" if task_key.startswith("mmlu_pro") else
            "gpqa" if task_key.startswith("gpqa") else
            "tools" if task_key.startswith("tools") else None)
    if task is None:
        return None
    ran = ts(t.get("ran_at"))
    alias = cache_row["model"]
    best = None
    for r in runs:
        if r["task"] != task or r["backend"] != backend or r["n"] != t.get("n"):
            continue
        if not (r["model"] == alias or r["model"].startswith(alias + "@")):
            continue
        dt = abs((ts(r["completed"]) - ran).total_seconds()) if r["completed"] and ran else 1e9
        if dt <= tol_s and (best is None or dt < best[0]):
            best = (dt, r)
    return best[1] if best else None


def section_cache_vs_logs(runs, cache):
    rows = []
    for key, row in cache.items():
        if key.startswith("_"):
            continue
        for tk, tv in (row.get("tasks") or {}).items():
            if tk in ("leak_probe", "longctx_probe") or tv.get("score") is None and tv.get("pass@1") is None:
                continue
            r = cache_task_run(runs, row, tk, row.get("backend"))
            val = tv.get("pass@1", tv.get("score"))
            rec = {"row": key, "task": tk, "cache_value": val, "cache_n": tv.get("n"),
                   "ran_at": tv.get("ran_at")}
            if r is None:
                rec["log"] = None
                rec["status"] = "NO MATCHING LOG"
            else:
                rec["log"] = r["file"]
                rec["log_x"] = int(r["x"])
                rec["log_n"] = r["n"]
                rec["log_value"] = r4(r["x"] / r["n"])
                rec["status"] = "match" if abs(r["x"] / r["n"] - val) < 5e-4 else "MISMATCH"
                if tk.startswith("tools") and tv.get("by_subcase"):
                    for sc, sv in tv["by_subcase"].items():
                        v = [x for x, s in zip(r["vals"], r["subcases"]) if s == sc]
                        if abs(sum(v) / len(v) - sv) > 5e-4:
                            rec["status"] = "MISMATCH (subcase)"
            rows.append(rec)
    return rows


def section_backends_vllm_vs_sglang(runs, cache):
    """Reconstruct the '18 paired McNemar tests' of docs/backends.md."""
    pairs = []
    spec = [
        ("Ornith-1.0-9B-NVFP4", "ksparavec/Ornith-1.0-9B-NVFP4@e6b618246f78::vllm::262144",
         "ksparavec/Ornith-1.0-9B-NVFP4@e6b618246f78::sglang::196608"),
        ("Qwen3.5-9B-NVFP4", "ykarout/Qwen3.5-9B-NVFP4@bd8c8f493d2e::vllm::262144",
         "ykarout/Qwen3.5-9B-NVFP4@bd8c8f493d2e::sglang::196608"),
        ("gpt-oss-20b", "openai/gpt-oss-20b@6cee5e81ee83::vllm::131072",
         "openai/gpt-oss-20b@6cee5e81ee83::sglang::131072"),
    ]
    for model, kv, ks in spec:
        rv, rs = cache[kv], cache[ks]
        for task in ("gsm8k", "humaneval", "humaneval_plus", "mmlu_pro", "gpqa", "tools"):
            tv = next(k for k in rv["tasks"] if k.startswith(task + "_") and not
                      (task == "humaneval" and k.startswith("humaneval_plus")))
            tsg = next(k for k in rs["tasks"] if k.startswith(task + "_") and not
                       (task == "humaneval" and k.startswith("humaneval_plus")))
            lv = cache_task_run(runs, rv, tv, "vllm")
            ls = cache_task_run(runs, rs, tsg, "sglang")
            note = []
            if lv["n"] != ls["n"]:
                note.append(f"n differs ({lv['n']} vs {ls['n']}): paired on first {min(lv['n'], ls['n'])}")
            if task == "tools":
                mv = lv["task_args"].get("tool_mode", "forced(default)")
                ms = ls["task_args"].get("tool_mode", "forced(default)")
                if mv != ms:
                    note.append(f"tool_mode differs (vLLM {mv} vs SGLang {ms}): not the same protocol")
            if rv.get("context") != rs.get("context"):
                note.append(f"ctx differs ({rv.get('context')} vs {rs.get('context')})")
            c = compare_runs(ls, lv, f"{model} {task}: SGLang vs vLLM", "; ".join(note))
            c["model"] = model
            c["task"] = task
            pairs.append(c)
    adj = S.holm_list([p["mcnemar_exact_p"] for p in pairs])
    for p, a in zip(pairs, adj):
        p["p_holm_18"] = r4(a)
        p["p_bonferroni_18"] = r4(min(1.0, 18 * p["mcnemar_exact_p"]))
    # Variant B: same family, but the two tools_use tests re-paired with the
    # vLLM run measured in the SAME protocol as SGLang (pinned/forced), which
    # is what a like-for-like comparison needs.
    alt = []
    for p in pairs:
        if p["task"] == "tools" and p["model"] in ("Ornith-1.0-9B-NVFP4", "Qwen3.5-9B-NVFP4"):
            served_v = {"Ornith-1.0-9B-NVFP4": ("Ornith-1.0-9B-NVFP4@262144", "2026-07-16"),
                        "Qwen3.5-9B-NVFP4": ("Qwen3.5-9B-NVFP4@262144", "2026-07-17")}[p["model"]]
            served_s = {"Ornith-1.0-9B-NVFP4": "Ornith-1.0-9B-NVFP4@196608",
                        "Qwen3.5-9B-NVFP4": "Qwen3.5-9B-NVFP4@196608"}[p["model"]]
            lv = one(runs, "tools", served_v[0], backend="vllm", created=served_v[1])
            ls = one(runs, "tools", served_s, backend="sglang", created="2026-07-29")
            c = compare_runs(ls, lv, f"{p['model']} tools: SGLang vs vLLM (both pinned/forced)",
                             "protocol-matched; ctx differs; vLLM run is from an earlier date")
            c["model"], c["task"] = p["model"], "tools"
            alt.append(c)
        else:
            alt.append(dict(p))
    adj = S.holm_list([p["mcnemar_exact_p"] for p in alt])
    for p, a in zip(alt, adj):
        p["p_holm_18"] = r4(a)
    return {"as_cached": pairs, "protocol_matched_tools": alt}


def section_backends_other(runs):
    R = {}
    # 'the SAME model on the SAME backend moved 0.86 -> 0.74 across the engine bump alone'
    R["qwen35_sglang_he_engine_bump"] = compare_runs(
        one(runs, "humaneval", "Qwen3.5-9B-NVFP4@196608", backend="sglang", created="2026-07-29"),
        one(runs, "humaneval", "Qwen3.5-9B-NVFP4@196608", backend="sglang", created="2026-07-26"),
        "Qwen3.5-9B SGLang HumanEval 07-29 (v0.5.16) vs 07-26 (v0.5.10.post1)",
        "engine image changed between runs AND decoding is stochastic; the two causes are confounded")
    # Minimum resolvable difference at n=50 (95% half-widths).
    R["mde_n50"] = {
        "unpaired_halfwidth_at_p": {str(p): r4(S.mde_unpaired_halfwidth(50, p)) for p in (0.7, 0.8, 0.9, 0.95)},
        "paired_halfwidth_at_discordance": {str(d): r4(S.mde_paired_halfwidth(50, d)) for d in (0.06, 0.1, 0.2, 0.3)},
    }
    # Prepared checkpoints vs the as-delivered NVFP4 (164 problems).
    asd_he = one(runs, "humaneval", "Qwen3.8-27B-MTP-NVFP4@32768", n=164)
    asd_hp = one(runs, "humaneval_plus", "Qwen3.8-27B-MTP-NVFP4@32768", n=164)
    prep_he = one(runs, "humaneval", "Qwen3.8-27B-MTP-NVFP4@98304", n=164)
    prep_hp = one(runs, "humaneval_plus", "Qwen3.8-27B-MTP-NVFP4@98304", n=164)
    ar_he = one(runs, "humaneval", "Qwen3.8-27B-W4A16-AutoRound@131072", n=164)
    ar_hp = one(runs, "humaneval_plus", "Qwen3.8-27B-W4A16-AutoRound@131072", n=164)
    fam = [
        compare_runs(prep_he, asd_he, "HumanEval: prepared NVFP4 @98304 vs as-delivered NVFP4 @32768", "ctx differs"),
        compare_runs(prep_hp, asd_hp, "HumanEval+: prepared NVFP4 @98304 vs as-delivered NVFP4 @32768", "ctx differs"),
        compare_runs(ar_he, asd_he, "HumanEval: prepared AutoRound @131072 vs as-delivered NVFP4 @32768", "different quantization and ctx"),
        compare_runs(ar_hp, asd_hp, "HumanEval+: prepared AutoRound @131072 vs as-delivered NVFP4 @32768", "different quantization and ctx"),
    ]
    adj = S.holm_list([c["mcnemar_exact_p"] for c in fam])
    for c, a in zip(fam, adj):
        c["p_holm_4"] = r4(a)
    R["prepared_vs_asdelivered_164"] = fam
    R["prepared_164_intervals"] = {
        run_id(r): prop_summary(r["x"], r["n"], r["task"]) for r in (asd_he, asd_hp, prep_he, prep_hp, ar_he, ar_hp)}
    # router.md: 97.6 % via 11435 vs 46 % via 11437 (same weights), on the common first 50.
    dev_bad = one(runs, "humaneval", "Qwen3.8-27B-W4A16-devai-AutoRound@131072", created="2026-09-22T11")
    dev_fix = one(runs, "humaneval", "Qwen3.8-27B-W4A16-devai-AutoRound@131072", created="2026-09-22T12")
    R["router_976_vs_46"] = [
        compare_runs(ar_he, dev_bad, "AutoRound HumanEval: port 11435 (164-run, first 50) vs port 11437 before policy fix",
                     "n differs (164 vs 50): compared on the first 50 problems; the 164-run scores 97.6% overall"),
        compare_runs(dev_fix, dev_bad, "AutoRound-devai HumanEval: port 11437 after vs before the policy fix", ""),
    ]
    mtp_bad = one(runs, "humaneval", "Qwen3.8-27B-MTP-devai-NVFP4@118784", created="2026-09-22T11")
    mtp_fix = one(runs, "humaneval", "Qwen3.8-27B-MTP-devai-NVFP4@118784", created="2026-09-22T12")
    R["router_976_vs_46"].append(compare_runs(mtp_fix, mtp_bad, "MTP-devai-NVFP4 HumanEval: port 11437 after vs before the policy fix", ""))
    return R


def section_variance(reps):
    """Decoding (within-item) vs item-selection variance, from clean replicates.

    Model: item i has success probability p_i; a run draws one Bernoulli per
    item. For two iid runs on the same items, E[discordant fraction] =
    2 * mean_i p_i(1-p_i). With pbar the mean success rate,
    pbar(1-pbar) = S2_between + mean_i p_i(1-p_i) (population variances).
    For a SRSWOR of n from N: Var(mean) = (N-n)/(N-1) * S2_between/n +
    mean_i p_i(1-p_i)/n, versus the binomial pbar(1-pbar)/n that
    Clopper-Pearson assumes. Reported per clean replicate pair.
    """
    out = []
    pooled = defaultdict(lambda: [0, 0])
    for c in reps:
        if not c["clean_replicate"]:
            continue
        task = c["label"].split("] ")[1].split(" n=")[0]
        n = c["n"]
        pooled[task][0] += c["b_only_A"] + c["c_only_B"]
        pooled[task][1] += n
        pbar = (c["pA"] + c["pB"]) / 2
        if not (0 < pbar < 1):
            continue
        dec = c["discordant_fraction"] / 2
        tot = pbar * (1 - pbar)
        s2b = max(0.0, tot - dec)
        rec = {"label": c["label"], "A": c["A_full"], "B": c["B_full"], "hours_apart": c["hours_apart"],
               "discordant": c["discordant_fraction"], "decoding_var": r4(dec), "binomial_var": r4(tot),
               "between_item_var": r4(s2b), "decoding_share": r4(min(1.0, dec / tot))}
        N = POP.get(task)
        if task in ("gpqa", "mmlu_pro") and N:
            f = (N - n) / (N - 1)
            rec["fpc"] = r4(f)
            rec["se_ratio_true_vs_binomial"] = r4(math.sqrt((f * s2b + dec) / tot))
            rec["se_ratio_fullFPC_vs_binomial"] = r4(math.sqrt(f))
        out.append(rec)
    pooled_out = {t: {"discordant_items": d, "item_pairs": m, "pooled_discordance": r4(d / m),
                      "implied_mean_p_1mp": r4(d / m / 2)} for t, (d, m) in pooled.items()}
    return {"pairs": out, "pooled_by_task": pooled_out}


# --------------------------------------------------------------------------
# Leak probe
# --------------------------------------------------------------------------

def section_leak():
    out = {}
    lo, hi = S.clopper_pearson(0, 40)
    out["zero_of_40"] = {"statement": "0 marker hits in 40 prompts",
                         "cp95_upper_per_prompt_probability": r4(hi),
                         "note": "only meaningful for a hypothetical population of similar prompts; the 40 are a fixed hand-written set run at temperature 0"}
    for k, label in ((3, "Nemotron-Nano-9B-v2 2026-05-05: 3 '</think>' hits"),
                     (18, "Qwen3.8-27B-MTP-devai-NVFP4 before parser fix: 18 '</think>' hits"),
                     (22, "Qwen3.8-27B-W4A16-devai-AutoRound before parser fix: 22 '</think>' hits")):
        plo, phi = S.poisson_exact_ci(k)
        out[label] = {"hits": k, "prompts": 40, "leak_rate_hits_per_prompt": r4(k / 40),
                      "garwood95_hits_per_prompt": [r4(plo / 40), r4(phi / 40)],
                      "prompts_with_a_hit": f"between 1 and {min(k, 40)} (not recorded)"}
    return out


# --------------------------------------------------------------------------
# Current cache: standard table
# --------------------------------------------------------------------------

def section_current_table(runs, cache):
    rows = []
    for key, row in sorted(cache.items()):
        if key.startswith("_"):
            continue
        for tk, tv in sorted((row.get("tasks") or {}).items()):
            if tk in ("leak_probe", "longctx_probe"):
                continue
            val = tv.get("pass@1", tv.get("score"))
            if val is None:
                continue
            n = tv["n"]
            task = ("humaneval_plus" if tk.startswith("humaneval_plus") else
                    "humaneval" if tk.startswith("humaneval") else tk.split("_subset")[0].replace("tools_use_20", "tools"))
            if tk.startswith("tools"):
                task = "tools"
            x = round(val * n)
            lr = cache_task_run(runs, row, tk, row.get("backend"))
            tcount = sum(lr["limits"]) if lr else None
            rec = {"row": key, "task": tk, **prop_summary(x, n, task, tcount),
                   "log": lr["file"] if lr else None}
            if tk.startswith("tools"):
                rec["tool_mode"] = tv.get("tool_mode", "not recorded (pre-2026-07-20 rows: pinned/forced)")
            rows.append(rec)
    return rows


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

def fmt_ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def md(res) -> str:
    L = []
    L.append("# Bench re-analysis -- computed tables\n")
    L.append("All intervals 95 %, two-sided. CP = Clopper-Pearson exact; Wilson = Wilson score. "
             "Paired differences: Newcombe method 10; tests: exact McNemar. Aggregate = unweighted mean "
             "of GSM8K (n=100), HumanEval (n=50), tools_use (n=20); its CI is a stratified percentile "
             f"bootstrap (B={B}, seed={SEED}).\n")
    L.append("## Self-tests\n\nSee tests/python/test_stats_statlib.py.\n")
    for c in res["selftest"]:
        L.append(f"- {'PASS' if c['ok'] else 'FAIL'} {c['check']}: got {c['got']} want {c['want']}")
    L.append("\n## 2026-05-05 leaderboard (docs/bench-results.md full table)\n")
    L.append("| Model | GSM8K x/n | CP95 | HumanEval x/n | CP95 | tools x/n | CP95 | Aggregate | bootstrap95 | time-limit hits G/H/T |")
    L.append("|---|---:|---|---:|---|---:|---|---:|---|---|")
    for r in res["leaderboard"]:
        g, h, t, a = r["gsm8k"], r["humaneval"], r["tools"], r["aggregate"]
        L.append(f"| {r['model']} | {g['x']}/{g['n']} | {fmt_ci(g['cp95'])} | {h['x']}/{h['n']} | {fmt_ci(h['cp95'])} "
                 f"| {t['x']}/{t['n']} | {fmt_ci(t['cp95'])} | {a['estimate']:.3f} | {fmt_ci(a['bootstrap95'])} "
                 f"| {g['time_limit_hits']}/{h['time_limit_hits']}/{t['time_limit_hits']} |")
    L.append("\nTools subcases (x/5, CP95):\n")
    L.append("| Model | empty_schema | single_arg | multi_tool_pick | result_followup |")
    L.append("|---|---|---|---|---|")
    for r in res["leaderboard"]:
        cells = [f"{r['tools_subcases'][s]['x']}/5 {fmt_ci(r['tools_subcases'][s]['cp95'])}" for s in SUBCASES]
        L.append(f"| {r['model']} | " + " | ".join(cells) + " |")
    L.append(f"\nDoc-vs-log discrepancies in the 05-05 table: {res['leaderboard_discrepancies'] or 'none'}\n")
    L.append("## Adjacent-rank comparisons on the aggregate (family of 8, Holm)\n")
    L.append("| Higher | Lower | diff | bootstrap95 | p (sign-flip) | p Holm(8) | p Holm(36) |")
    L.append("|---|---|---:|---|---:|---:|---:|")
    for p in res["ranking"]["adjacent"]:
        L.append(f"| {p['A']} | {p['B']} | {p['diff']:+.3f} | {fmt_ci(p['bootstrap95'])} | {p['p_perm']:.4f} "
                 f"| {p['p_holm_adjacent8']:.4f} | {p['p_holm_all36']:.4f} |")
    L.append("\n## Leader (Nemotron-3-Nano) vs each other row, per task (family of 24, Holm)\n")
    L.append("| Task | Other | leader x | other x | b | c | diff | Newcombe95 | p | p Holm(24) |")
    L.append("|---|---|---:|---:|---:|---:|---:|---|---:|---:|")
    for c in res["ranking"]["leader_vs_others_per_task"]:
        L.append(f"| {c['task']} | {c['B']} | {c['a_both'] + c['b_only_A']} | {c['a_both'] + c['c_only_B']} | "
                 f"{c['b_only_A']} | {c['c_only_B']} | {c['diff_A_minus_B']:+.3f} | {fmt_ci(c['newcombe10_95'])} | "
                 f"{c['mcnemar_exact_p']:.4f} | {c['p_holm_24']:.4f} |")
    L.append("\n## Threshold robustness (PRODUCTION_AGENTIC quality thresholds)\n")
    L.append("| Model | GSM8K>=0.9 | HumanEval>=0.7 | tools>=0.9 |")
    L.append("|---|---|---|---|")
    for r in res["thresholds"]:
        cell = []
        for t in ("gsm8k", "humaneval", "tools"):
            v = r[t]
            short = {"CI entirely >= threshold": "meets (CI above)",
                     "CI entirely < threshold": "fails (CI below)"}.get(v["verdict"], "UNDETERMINED (CI straddles)")
            cell.append(f"{v['estimate']:.2f} {fmt_ci(v['cp95'])} {short}")
        L.append(f"| {r['model']} | " + " | ".join(cell) + " |")
    L.append("\n## Change / comparison claims (paired, exact McNemar)\n")
    L.append("| Comparison | A | B | b | c | diff A-B | Newcombe95 | p | note |")
    L.append("|---|---|---|---:|---:|---:|---|---:|---|")
    for k, c in res["change_claims"].items():
        L.append(f"| {c['label']} | {c['A_full']} | {c['B_full']} | {c['b_only_A']} | {c['c_only_B']} | "
                 f"{c['diff_A_minus_B']:+.3f} | {fmt_ci(c['newcombe10_95'])} | {c['mcnemar_exact_p']:.4f} | {c['note']} |")
    L.append("\n## Static scorer analysis (no model code executed)\n")
    for r in res["scorer_static"]["v1_scored_runs"]:
        L.append(f"- {r['run']}: pass {r['pass']}/{r['n']}; fail classes {r['fail_classes']}; "
                 f"syntax failures whose v2 program parses: {r['syntax_failures_where_v2_program_parses']}/{r['syntax_failures']}; "
                 f"passed items where v2 output differs beyond trailing whitespace: {r['passed_items_v2_differs_beyond_trailing_ws']}; "
                 f"passed items whose v2 program no longer parses: {r['passed_items_where_v2_program_fails_to_parse']}")
    L.append(f"- R1-Qwen-7B cross-tab (05-02 v1 status | 05-05 v2 outcome): {res['scorer_static']['r1_qwen_crosstab']}")
    for r in res["scorer_static"]["bpe_markers"]:
        L.append(f"- BPE markers: {r['run']}: pass {r['pass']}/{r['n']}, text with U+0120/U+010A: "
                 f"{r['text_with_bpe_markers']}, reasoning with markers: {r['reasoning_with_bpe_markers']}, "
                 f"fail classes {r['fail_classes']}")
    L.append("\n## docs/backends.md: vLLM vs SGLang, 18 paired tests (family of 18)\n")
    L.append("| Model | Task | SGLang | vLLM | paired n | b (SGL only) | c (vLLM only) | diff | Newcombe95 | p | p Holm | note |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|")
    for c in res["backends_vllm_vs_sglang"]["as_cached"] + [
            x for x in res["backends_vllm_vs_sglang"]["protocol_matched_tools"] if "both pinned" in x["label"]]:
        L.append(f"| {c['model']} | {c['task']} | {c['A_full']} | {c['B_full']} | {c['paired_on_first']} | "
                 f"{c['b_only_A']} | {c['c_only_B']} | {c['diff_A_minus_B']:+.3f} | {fmt_ci(c['newcombe10_95'])} | "
                 f"{c['mcnemar_exact_p']:.4f} | {c['p_holm_18']:.4f} | {c['note']} |")
    bo = res["backends_other"]
    L.append("\n## docs/backends.md / docs/router.md other claims\n")
    c = bo["qwen35_sglang_he_engine_bump"]
    L.append(f"- {c['label']}: {c['A_full']} vs {c['B_full']}, b={c['b_only_A']} c={c['c_only_B']}, "
             f"diff {c['diff_A_minus_B']:+.3f} {fmt_ci(c['newcombe10_95'])}, p={c['mcnemar_exact_p']}")
    L.append(f"- n=50 95% half-widths: {bo['mde_n50']}")
    for c in bo["prepared_vs_asdelivered_164"]:
        L.append(f"- {c['label']}: {c['A_full']} vs {c['B_full']}, b={c['b_only_A']} c={c['c_only_B']}, "
                 f"diff {c['diff_A_minus_B']:+.4f} {fmt_ci(c['newcombe10_95'])}, p={c['mcnemar_exact_p']}, Holm(4)={c['p_holm_4']}")
    for c in bo["router_976_vs_46"]:
        L.append(f"- {c['label']}: {c['A_full']} vs {c['B_full']} (paired on first {c['paired_on_first']}), "
                 f"b={c['b_only_A']} c={c['c_only_B']}, diff {c['diff_A_minus_B']:+.3f} {fmt_ci(c['newcombe10_95'])}, "
                 f"p={c['mcnemar_exact_p']}")
    L.append("\n## Incidental replications (same served name, backend, task, n, task args)\n")
    L.append("| Label | A (earlier) | B (later) | hours | b | c | discordant | p | time-limit hits A/B | documented change between |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for c in res["replications"]:
        L.append(f"| {c['label']} | {c['A_full']} {c['A'].split('n=')[1].split(' ')[1]} | "
                 f"{c['B_full']} {c['B'].split('n=')[1].split(' ')[1]} | {c['hours_apart']} | {c['b_only_A']} | {c['c_only_B']} | "
                 f"{c['discordant_fraction']:.3f} | {c['mcnemar_exact_p']:.4f} | {c['time_limit_hits_A_B']} | "
                 f"{'; '.join(c['documented_changes_between']) or 'none documented'} |")
    L.append("\n## Decoding vs item-selection variance (clean replicate pairs only)\n")
    L.append("| Pair | A | B | hours | discordant | mean p(1-p) | pbar(1-pbar) | decoding share | SE ratio (true/binomial) | SE ratio (full FPC/binomial) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for v in res["variance"]["pairs"]:
        L.append(f"| {v['label']} | {v['A']} | {v['B']} | {v['hours_apart']} | {v['discordant']:.3f} | {v['decoding_var']:.4f} | "
                 f"{v['binomial_var']:.4f} | {v['decoding_share']:.2f} | {v.get('se_ratio_true_vs_binomial', '-')} | {v.get('se_ratio_fullFPC_vs_binomial', '-')} |")
    L.append(f"\nPooled discordance by task (clean pairs): {res['variance']['pooled_by_task']}\n")
    L.append("\n## Leak probe\n")
    for k, v in res["leak"].items():
        L.append(f"- {k}: {v}")
    L.append("\n## Cache vs log reconciliation\n")
    bad = [r for r in res["cache_vs_logs"] if r["status"] != "match"]
    L.append(f"{len(res['cache_vs_logs'])} cache task entries checked; non-matching: {len(bad)}\n")
    for r in bad:
        L.append(f"- {r}")
    L.append("\n## Current cache: standard table\n")
    L.append("| Row | Task | n | x | estimate | CP95 | Wilson95 | FPC exact95 (deterministic only) | time-limit hits | range if censored items were right |")
    L.append("|---|---|---:|---:|---:|---|---|---|---:|---|")
    for r in res["current_table"]:
        f = r.get("fpc_exact95_deterministic_only")
        rg = r.get("range_if_censored_items_all_wrong_or_all_right")
        L.append(f"| {r['row']} | {r['task']} | {r['n']} | {r['x']} | {r['estimate']:.3f} | {fmt_ci(r['cp95'])} | "
                 f"{fmt_ci(r['wilson95'])} | {fmt_ci(f) if f else '-'} | {r.get('time_limit_hits')} | {fmt_ci(rg) if rg else '-'} |")
    return "\n".join(L) + "\n"


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)  # usage
    logs_json, logdir, cache_path, outdir = (Path(a) for a in sys.argv[1:5])
    outdir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(logs_json)
    cache = json.loads(cache_path.read_text())
    res = {"meta": {"seed": SEED, "bootstrap_B": B, "n_runs_loaded": len(runs),
                    "inputs": [str(logs_json), str(logdir), str(cache_path)]}}
    res["selftest"] = []
    sel, rows, disc = section_leaderboard(runs)
    res["leaderboard"] = rows
    res["leaderboard_discrepancies"] = disc
    res["ranking"] = section_ranking(sel)
    res["thresholds"] = section_thresholds(rows)
    res["change_claims"] = section_change_claims(runs, logdir)
    res["scorer_static"] = section_scorer_static(runs, logdir)
    res["backends_vllm_vs_sglang"] = section_backends_vllm_vs_sglang(runs, cache)
    res["backends_other"] = section_backends_other(runs)
    res["replications"] = section_replications(runs)
    res["variance"] = section_variance(res["replications"])
    res["leak"] = section_leak()
    res["cache_vs_logs"] = section_cache_vs_logs(runs, cache)
    res["current_table"] = section_current_table(runs, cache)
    (outdir / "results.json").write_text(json.dumps(res, indent=1))
    (outdir / "results.md").write_text(md(res))
    print(f"wrote {outdir / 'results.json'} and {outdir / 'results.md'}")
    bad = [c for c in res["selftest"] if not c["ok"]]
    if bad:
        print(f"SELFTEST FAILURES: {bad}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
