#!/usr/bin/env python3
"""Use-case score table: benchmark items regrouped by what they measure.

Usage: python3 usecase_scores.py <logs_extracted.json> <bench-cache.json> <out_dir>

  logs_extracted.json  from bench_extract_logs.py (per-item outcomes with the
                       MMLU-Pro `category` / GPQA `subdomain` tags; inspect-ai
                       0.3.271 logs are zstd, so extract inside the lab image)
  bench-cache.json     the bench cache whose rows are scored (every task entry
                       is matched to the log that produced it)

Mapping (owner's choice, "Scheme A", 2026-09-27). No item counts twice, and
every MMLU-Pro category is used exactly once:

  coding             HumanEval+ and HumanEval (the same first 50 problems;
                     each problem is ONE cluster of two outcomes), MMLU-Pro
                     computer science
  general_reasoning  MMLU-Pro law, history, philosophy, psychology,
                     economics, business, health, other
  problem_analysis   GSM8K, MMLU-Pro math, physics, chemistry, engineering
  complex_systems    GPQA-Diamond (all subdomains), MMLU-Pro biology

Score = share of the use case's items answered correctly (items pooled, one
weight each). A time-out is scored wrong, as the harness scores it, and is
counted beside the score; (x + t)/N is the upper bound.

  Interval   Clopper-Pearson when every cluster is one item; otherwise a
             percentile bootstrap over clusters (coding), B = 10000, fixed seed.
  Compare    rows paired on identical items (content key); difference of
             pooled shares with a cluster bootstrap interval and a paired
             sign-flip test on cluster-level differences; Holm over the four
             winner-vs-runner-up tests.
  Rank       by the score's point estimate. Speed is shown and never
             overrides a quality lead (the owner: 10 % better results are
             worth 20 % less speed). Winner = best row; runner-up = best row
             of a DIFFERENT base model (owner's choice). A row that lacks any
             of the use case's components is not ranked.

Read-only on its inputs; stdlib only.
"""
import json
import random
import re
import sys
from pathlib import Path

import bench_stats as A
import statlib as S

SEED = 20260927
B = 10000

USE_CASES = {
    "coding": [("humaneval_plus", None), ("humaneval", None),
               ("mmlu_pro", {"computer science"})],
    "general_reasoning": [("mmlu_pro", {"law", "history", "philosophy", "psychology",
                                        "economics", "business", "health", "other"})],
    "problem_analysis": [("gsm8k", None),
                         ("mmlu_pro", {"math", "physics", "chemistry", "engineering"})],
    "complex_systems": [("gpqa", None), ("mmlu_pro", {"biology"})],
}
TITLES = {"coding": "Coding", "general_reasoning": "General reasoning",
          "problem_analysis": "Problem analysis", "complex_systems": "Complex systems"}
AGENT_CTX_FLOOR = 131072  # memory: a fresh agent session needs ~47K; 128K is the floor
TIMEOUT_TYPES = ("time", "working")


def base_model(alias: str) -> str:
    """Group quantisations/backends of one model; the runner-up must differ."""
    a = alias.lower()
    if "qwen3.8" in a and "27b" in a:
        return "Qwen3.8-27B"
    if "qwen3.6" in a and "35b-a3b" in a:
        return "Qwen3.6-35B-A3B"
    if "gemma-4-26b-a4b" in a or "gemma4:26b" in a:
        return "Gemma-4-26B-A4B"
    if "gpt-oss-20b" in a:
        return "gpt-oss-20b"
    return alias.split("@")[0]


def is_timeout(limit) -> bool:
    if isinstance(limit, dict):
        return limit.get("type") in TIMEOUT_TYPES
    return isinstance(limit, str) and limit in TIMEOUT_TYPES


def load_runs(path: Path) -> list[dict]:
    """Runs with their items, in the shape bench_stats.cache_task_run expects."""
    runs = []
    for r in json.loads(path.read_text()):
        # Every log with items, whatever its status: a task stopped at its
        # deadline has a `cancelled` (or, after SIGKILL, header-less) log,
        # and its cache entry names that log explicitly.
        if not r.get("items"):
            continue
        m = re.search(r":(\d+)/", r.get("model_base_url") or "")
        runs.append({"file": r["file"], "created": r["created"], "completed": r.get("completed_at"),
                     "task": A.TASK_SHORT.get(r.get("task")),
                     "model": A.served_model_name(r.get("model") or ""),
                     "backend": A.PORT_BACKEND.get(m.group(1)) if m else None,
                     "n": len(r["items"]), "items": r["items"], "status": r.get("status")})
    return runs


def items_for(task: str, run: dict, cats: set | None, max_id: int | None = None) -> list[dict]:
    """Scorable items of one run for one component: (cluster, key, y, timeout).

    max_id: for a task stopped at its deadline, the unbroken prefix its cache
    entry was scored on (harvest_truncated.py) -- questions 1..max_id only.
    """
    out = []
    for it in run["items"]:
        if cats is not None and (it.get("category") or "") not in cats:
            continue
        if max_id is not None and int(it.get("id") or 0) > max_id:
            continue
        y = 1 if it.get("value") == 1.0 else 0
        if task in ("humaneval", "humaneval_plus"):
            cluster = "he:" + str(it.get("task_id") or it.get("id"))
        else:
            cluster = f"{task}:{it.get('item_key') or it.get('id')}"
        out.append({"cluster": cluster, "key": f"{task}:{it.get('item_key') or it.get('id')}",
                    "y": y, "timeout": is_timeout(it.get("limit"))})
    return out


def row_items(runs: list[dict], row: dict, use_case: str) -> tuple[list[dict] | None, dict]:
    """All items of a cache row for a use case, or None if a component is missing."""
    items, comps = [], {}
    for task, cats in USE_CASES[use_case]:
        key = next((k for k in row.get("tasks", {}) if k.startswith(
            {"humaneval": "humaneval_subset_", "humaneval_plus": "humaneval_plus_subset_",
             "gsm8k": "gsm8k_subset_", "mmlu_pro": "mmlu_pro_subset_", "gpqa": "gpqa_subset_"}[task])), None)
        entry = row["tasks"][key] if key else {}
        run = None
        if entry.get("inspect_log"):
            run = next((r for r in runs if r["file"] == entry["inspect_log"]), None)
        elif key:
            run = A.cache_task_run([r for r in runs if r.get("status") == "success"],
                                   row, key, row.get("backend"))
        if run is None:
            return None, comps
        got = items_for(task, run, cats, max_id=(entry.get("truncated") or {}).get("prefix"))
        if not got:
            return None, comps
        name = task if cats is None else f"{task}[{','.join(sorted(cats))}]"
        comps[name] = [sum(i["y"] for i in got), len(got)]
        items += got
    return items, comps


def clusters_of(items: list[dict]) -> list[tuple[int, int]]:
    agg: dict[str, list[int]] = {}
    for i in items:
        c = agg.setdefault(i["cluster"], [0, 0])
        c[0] += i["y"]
        c[1] += 1
    return [tuple(v) for v in agg.values()]


def boot_ratio(units: list[tuple[float, int]], reps: int = B, seed: int = SEED) -> tuple[float, float]:
    """Percentile interval of sum(num)/sum(size), resampling units."""
    rng = random.Random(seed)
    n = len(units)
    stats = []
    for _ in range(reps):
        num = size = 0.0
        for _ in range(n):
            a, s = units[rng.randrange(n)]
            num += a
            size += s
        stats.append(num / size)
    stats.sort()
    return S.quantile_type7(stats, 0.025), S.quantile_type7(stats, 0.975)


def score(items: list[dict]) -> dict:
    x, n = sum(i["y"] for i in items), len(items)
    t = sum(i["timeout"] for i in items)
    cl = clusters_of(items)
    if all(s == 1 for _, s in cl):
        lo, hi = S.clopper_pearson(x, n)
        method = "Clopper-Pearson"
    else:
        lo, hi = boot_ratio(cl)
        method = f"cluster bootstrap ({len(cl)} clusters)"
    return {"x": x, "n": n, "score": x / n, "ci": [lo, hi], "ci_method": method,
            "timeouts": t, "upper_if_timeouts_right": (x + t) / n}


def compare(ia: list[dict], ib: list[dict]) -> dict:
    """a - b on the items both rows answered (identical content keys)."""
    by_a = {i["key"]: i for i in ia}
    by_b = {i["key"]: i for i in ib}
    shared = sorted(set(by_a) & set(by_b))
    if not shared:
        return {"shared": 0}
    agg: dict[str, list[int]] = {}
    for k in shared:
        c = agg.setdefault(by_a[k]["cluster"], [0, 0])
        c[0] += by_a[k]["y"] - by_b[k]["y"]
        c[1] += 1
    units = list(agg.values())
    diff = sum(d for d, _ in units) / len(shared)
    lo, hi = boot_ratio([tuple(u) for u in units])
    _, p = S.paired_signflip_test([[float(d) for d, _ in units]], reps=B, seed=SEED)
    return {"shared": len(shared), "diff": diff, "ci": [lo, hi], "p_signflip": p,
            "a_better": sum(1 for d, _ in units if d > 0), "b_better": sum(1 for d, _ in units if d < 0)}


def rank_use_case(runs: list[dict], cache: dict, use_case: str) -> dict:
    rows, unranked = [], []
    for key, row in cache.items():
        if not A_is_row(key, row):
            continue
        items, comps = row_items(runs, row, use_case)
        label = {"model": row.get("model"), "backend": row.get("backend"), "ctx": row.get("context"),
                 "base": base_model(row.get("model", ""))}
        if items is None:
            unranked.append({**label, "components_found": comps})
            continue
        m = row.get("metrics") or {}
        rows.append({**label, **score(items), "components": comps,
                     "tps": m.get("tps_sustained_p50"), "tps_token_sources": m.get("tps_token_sources"),
                     "below_agent_ctx_floor": (row.get("context") or 0) < AGENT_CTX_FLOOR,
                     "_items": items})
    rows.sort(key=lambda r: (-r["score"], -(r["tps"] or 0)))
    out = {"use_case": use_case, "rows": rows, "unranked": unranked}
    if rows:
        win = rows[0]
        runner = next((r for r in rows[1:] if r["base"] != win["base"]), None)
        out["winner"] = win
        out["runner_up"] = runner
        if runner:
            out["winner_vs_runner_up"] = compare(win["_items"], runner["_items"])
        out["winner_vs_same_base"] = [
            {"model": r["model"], "backend": r["backend"], **compare(win["_items"], r["_items"])}
            for r in rows[1:] if r["base"] == win["base"]]
    return out


def A_is_row(key: str, row) -> bool:
    return not key.startswith("_") and isinstance(row, dict) and "tasks" in row


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def md(results: list[dict]) -> str:
    L = ["# Use-case scores", "",
         "Share of each use case's benchmark items answered correctly, one run per row at",
         "greedy decoding (post-fix harness). Interval: 95 %; `t` = time-outs (scored wrong).",
         "Winner = best row; runner-up = best row of a different base model. Ranked by",
         "quality; speed (tok/s, single stream, engine-counted, MTP off on vLLM rows) is shown",
         "and never overrides a quality lead.", ""]
    for res in results:
        L += [f"## {TITLES[res['use_case']]}", "",
              "| # | Model | Backend | Ctx | Score % | 95 % CI | x/N | t | tok/s | Components |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(res["rows"], 1):
            comps = "; ".join(f"{k} {a}/{b}" for k, (a, b) in r["components"].items())
            flag = " (ctx < 128K)" if r["below_agent_ctx_floor"] and res["use_case"] == "coding" else ""
            L.append(f"| {i} | {r['model']}{flag} | {r['backend']} | {r['ctx']} | {pct(r['score'])} | "
                     f"{pct(r['ci'][0])}-{pct(r['ci'][1])} | {r['x']}/{r['n']} | {r['timeouts']} | "
                     f"{r['tps'] if r['tps'] is not None else '-'} | {comps} |")
        for u in res["unranked"]:
            L.append(f"| - | {u['model']} | {u['backend']} | {u['ctx']} | not ranked: missing a component | | | | | |")
        L.append("")
        w, ru, c = res.get("winner"), res.get("runner_up"), res.get("winner_vs_runner_up")
        if w:
            L.append(f"- **Winner:** {w['model']} [{w['backend']}], {pct(w['score'])} %.")
        if ru and c and c.get("shared"):
            est = "established" if c.get("p_holm", c["p_signflip"]) < 0.05 and c["ci"][0] > 0 else "not established"
            L.append(f"- **Runner-up:** {ru['model']} [{ru['backend']}], {pct(ru['score'])} %. "
                     f"Winner minus runner-up on {c['shared']} shared items: {100 * c['diff']:+.1f} points "
                     f"(95 % CI {100 * c['ci'][0]:+.1f} to {100 * c['ci'][1]:+.1f}), sign-flip p = "
                     f"{c['p_signflip']:.3f}, Holm p = {c.get('p_holm', float('nan')):.3f}: lead {est}.")
        for s in res.get("winner_vs_same_base", []):
            if s.get("shared"):
                L.append(f"- Same base model, {s['model']} [{s['backend']}]: winner minus it "
                         f"{100 * s['diff']:+.1f} points (95 % CI {100 * s['ci'][0]:+.1f} to "
                         f"{100 * s['ci'][1]:+.1f}), p = {s['p_signflip']:.3f} (raw).")
        L.append("")
    return "\n".join(L) + "\n"


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)  # usage
    runs = load_runs(Path(sys.argv[1]))
    cache = json.loads(Path(sys.argv[2]).read_text())
    out = Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    results = [rank_use_case(runs, cache, uc) for uc in USE_CASES]
    tests = {r["use_case"]: r["winner_vs_runner_up"]["p_signflip"] for r in results
             if r.get("winner_vs_runner_up", {}).get("shared")}
    for uc, p in S.holm(tests).items():
        next(r for r in results if r["use_case"] == uc)["winner_vs_runner_up"]["p_holm"] = p
    for r in results:
        for row in r["rows"]:
            row.pop("_items", None)
        for k in ("winner", "runner_up"):
            if r.get(k):
                r[k] = {kk: v for kk, v in r[k].items() if kk != "_items"}
    (out / "usecase_scores.json").write_text(json.dumps(results, indent=1))
    (out / "usecase_scores.md").write_text(md(results))
    print(md(results))


if __name__ == "__main__":
    main()
