"""Statistical re-analysis of the laya trainer claims (read-only on every input).

    python3 laya_stats.py [--runs DIR] [--datasets DIR] [--phase4 DIR] [--shadow FILE]
                          [--fresh-meta FILE] [--zeroshot FILE] [--trainer-log FILE]
                          [--router-log FILE] [--corpus-meta FILE] [--out DIR]

Defaults point at this host's retained data. Writes <out>/laya_stats.json and
<out>/laya_stats.md. Standard library only; deterministic (seeded bootstrap).
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import statlib as S  # noqa: E402

HOME = Path.home()
D = dict(
    runs="/var/cache/devai/laya/runs",
    datasets="/var/cache/devai/laya/datasets",
    phase4=str(HOME / "devai-home/phase4"),
    shadow=str(HOME / "devai-home/.local/share/aiagent/artifacts/system1/skills/polarity/classify/shadow.jsonl"),
    fresh_meta=str(HOME / "laya-pilot/fresh500.meta.jsonl"),
    corpus_meta=str(HOME / "laya-pilot/corpus.meta.jsonl"),
    zeroshot=str(HOME / "git/aiagent/docs/design/laya-system1/handson/results_quality.json"),
    trainer_log="/var/cache/devai/logs/devai-laya-trainer.log",
    router_log="/var/cache/devai/logs/devai-router.log",
    out=str(Path.home() / ".cache/devai/stats/laya"),
)
CAMPAIGN = {"0": "ftjob-6af5edfd74fc76e4556837b9", "1": "ftjob-d0e411c9b85d1168ff7ea1fa",
            "2": "ftjob-a40f596f637ba6c331490aea"}
FAST = "ftjob-220e2e9aefbd968f47ab66d4"
REFERENCE = "ftjob-f91b6d747c913af35c209ad9"


def prop(x: int, n: int) -> dict:
    lo, hi = S.clopper_pearson(x, n)
    wl, wh = S.wilson(x, n)
    return {"x": x, "n": n, "p": x / n if n else None, "cp95": [lo, hi], "wilson95": [wl, wh],
            "cp_lower_one_sided_95": S.clopper_pearson_lower_one_sided(x, n, 0.05)}


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ------------------------------------------------------------------ A: runs
def runs_section(runs: Path, trainer_log: Path) -> dict:
    started, finished = {}, {}
    pat = re.compile(r" (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) INFO laya_trainer: job (ftjob-\w+) (started|finished)")
    for line in trainer_log.read_text(errors="replace").splitlines():
        m = pat.search(line)
        if m:
            t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
            (started if m.group(3) == "started" else finished)[m.group(2)] = t
    out = {}
    for d in sorted(runs.iterdir()):
        m = json.loads((d / "manifest.json").read_text())
        j = json.loads((d / "job.json").read_text())
        jid = j["id"]
        hold_log = ((finished[jid] - started[jid]).total_seconds()
                    if jid in started and jid in finished else None)
        out[jid] = {
            "dataset": m["dataset"]["id"], "counts": m["dataset"]["counts"],
            "memory_mode": m["trainer"]["hyperparameters"]["memory_mode"],
            "seed": m["trainer"]["seed"], "metadata": j.get("metadata"),
            "hold_s_jobjson_int": j["finished_at"] - j["devai"]["started_at"],
            "hold_s_trainer_log": hold_log,
            "phase_timings_s": m["trainer"]["timings"],
            "peak_vram_gib_torch": m["trainer"]["peak_vram_gib"],
            "parity": m["parity"], "golden": m["golden"],
            "calibration": {k: m["calibration"][k] for k in ("n", "per_type_n", "temperature_raw",
                                                              "temperature_applied", "clamp", "min_samples")},
            "loss_per_epoch": m["metrics"]["loss_per_epoch"],
            "trained_tokens": m["metrics"]["trained_tokens"],
        }
    # jobs whose run directories were deleted by a passing make laya-check
    extra = {jid: (finished[jid] - started[jid]).total_seconds()
             for jid in started if jid in finished and jid not in out}
    return {"jobs": out, "log_only_jobs_hold_s": extra}


def hold_model(jobs: dict) -> dict:
    """Per-phase rates and a two-point linear extrapolation of the hold vs dataset scale."""
    rates = {}
    for r, jid in CAMPAIGN.items():
        j = jobs[jid]
        c, t = j["counts"], j["phase_timings_s"]
        rates[r] = {"train_rows": c["train"],
                    "training_ms_per_train_row_4ep": 1000 * t["training"] / c["train"],
                    "parity_s_per_heldout_row": t["checking_parity"] / c["heldout"],
                    "calib_s_per_calib_row": t["calibrating"] / c["calib"],
                    "exporting_s": t["exporting"], "validating_s": t["validating"],
                    "hold_s": j["hold_s_trainer_log"]}
    ref, r0 = jobs[REFERENCE], jobs[CAMPAIGN["0"]]
    size = lambda j: j["counts"]["train"] + j["counts"]["calib"] + j["counts"]["heldout"]  # noqa: E731
    x1, y1 = size(ref) / size(r0), ref["hold_s_trainer_log"]
    y2 = r0["hold_s_trainer_log"]
    v = (y2 - y1) / (1 - x1)
    f = y2 - v
    return {"per_round_rates": rates,
            "two_point_linear": {"fixed_s": f, "variable_s_at_round0_scale": v,
                                 "scale_at_900s": (900 - f) / v,
                                 "note": "hold = fixed + variable * k, k = labeled rows / round 0's 2059; "
                                         "fitted through the reference job (72 labeled rows) and round 0; "
                                         "same text lengths, 4 epochs, lean, this host"},
             "cap_over_max_hold": 900 / max(jobs[j]["hold_s_trainer_log"] for j in CAMPAIGN.values()),
             "cap_over_min_hold": 900 / min(jobs[j]["hold_s_trainer_log"] for j in CAMPAIGN.values())}


def teacher_cold_starts(router_log: Path) -> list[dict]:
    pat = re.compile(r"^\S+ (\d{4}/\d\d/\d\d \d\d:\d\d:\d\d) (waiting for vllm-devai at|vllm-devai ready|"
                     r"starting vllm-devai with model (\S+) \(ctx=(\d+), mtp=([^)]*)\))")
    out, w, spec = [], None, None
    for line in router_log.read_text(errors="replace").splitlines():
        m = pat.match(line)
        if not m:
            continue
        t = dt.datetime.strptime(m.group(1), "%Y/%m/%d %H:%M:%S")
        if m.group(2).startswith("starting"):
            spec = f"{m.group(3)}@{m.group(4)} mtp={m.group(5)}"
        elif m.group(2).startswith("waiting"):
            w = t
        elif w is not None:
            out.append({"waiting_at_utc": w.isoformat(), "seconds": (t - w).total_seconds(), "spec": spec})
            w = None
    return out


# ------------------------------------------------------------------ C: gate
def gate_section(phase4: Path, datasets: Path) -> dict:
    ev = {r: json.loads((phase4 / f"eval-r{r}.json").read_text()) for r in ("0", "1", "2")}
    ev["1@0.90"] = json.loads((phase4 / "eval-r1-p090.json").read_text())
    held = jl(datasets / "ds-e9443d334cb2" / "heldout.jsonl")
    lab = collections.Counter(r["gold"]["polarity"]["label"] for r in held)
    unanimous = sum(1 for r in held if max(r["gold"]["polarity"]["probabilities"].values()) > 0.99)
    maj_label, maj_n = lab.most_common(1)[0]
    n_h = len(held)
    rounds = {}
    for r, e in ev.items():
        q = e["questions"][0]
        acc_x = round(q["accuracy"] * q["n_heldout"])
        assert abs(acc_x / q["n_heldout"] - q["accuracy"]) < 1e-12
        rounds[r] = {
            "run": e["run"], "target": e["targets"]["precision"], "verdict": e["verdict"],
            "tau": q["tau"], "accepted": q["accepted"], "correct": q["correct"],
            "precision": prop(q["correct"], q["accepted"]),
            "precision_lower_reported_by_aiagent": q["precision_lower"],
            "coverage": prop(q["accepted"], q["n_heldout"]),
            "accuracy": prop(acc_x, q["n_heldout"]),
            "accuracy_vs_majority_class": {
                "majority_label": maj_label, "p0": maj_n / n_h,
                "exact_binomial_p_greater": S.binom_test_greater(acc_x, n_h, maj_n / n_h)},
            "ece_15bin": q["ece"], "brier": q["brier"],
            "coverage_at_precision_on_heldout_descriptive": q["coverage_at_precision"],
        }
    # round-to-round accuracy (paired per-item data NOT retained)
    comp = {}
    for a, b in (("0", "1"), ("1", "2"), ("0", "2")):
        xa, xb = rounds[a]["accuracy"]["x"], rounds[b]["accuracy"]["x"]
        net = xb - xa
        comp[f"r{a}->r{b}"] = {
            "accuracy": [xa / n_h, xb / n_h], "net_items": net,
            "newcombe_unpaired_95": S.newcombe_unpaired(xb, n_h, xa, n_h),
            "mcnemar_smallest_possible_p": S.mcnemar_exact(abs(net), 0) if net else 1.0,
            "note": "items are paired but per-item outcomes are not retained; the unpaired interval "
                    "ignores the pairing; the exact McNemar p cannot be smaller than the value shown "
                    "(reached only if every discordant item moved in one direction)"}
    x, n = rounds["1"]["correct"], rounds["1"]["accepted"]
    sel = {"r1_81_of_84": {
        "one_sided_lower_alpha_0.05": S.clopper_pearson_lower_one_sided(x, n, 0.05),
        "two_sided_95_lower": S.clopper_pearson(x, n)[0],
        "bonferroni_3_rounds_one_sided_lower_alpha_0.05/3": S.clopper_pearson_lower_one_sided(x, n, 0.05 / 3),
        "note": "round and target were chosen after the held-out results were seen"}}
    return {"heldout": {"n": n_h, "labels": dict(lab), "teacher_unanimous_k3": unanimous,
                        "majority_class_rate": maj_n / n_h},
            "rounds": rounds, "round_comparisons": comp, "selection_sensitivity": sel,
            "min_certifiable_0.95_0.05": S.clopper_pearson_lower_one_sided(59, 59, 0.05)}


# ------------------------------------------------------------------ D/E: shadow runs
def shadow_sections(shadow: Path, phase4: Path, datasets: Path, fresh_meta: Path) -> dict:
    lines = jl(shadow)
    s100 = lines[:100]
    assert all(r["artifact_id"].startswith("a866e0a4734f") for r in s100)
    acc = [r for r in s100 if r["would_accept"]]
    # sampling frame of the 100 texts
    texts = [t.decode() for t in (phase4 / "shadow-texts.bin").read_bytes().split(b"\0") if t]
    where = {}
    for sp in ("train", "calib", "heldout", "pool"):
        for r in jl(datasets / "ds-57c9c09510f6" / f"{sp}.jsonl"):
            st = r["state"]
            where[st["text"] if isinstance(st, dict) else st] = sp
    frame = collections.Counter(where.get(t, "none") for t in texts)
    shadow100 = {"n": len(s100), "agreement_all": prop(sum(r["agree"] for r in s100), len(s100)),
                 "would_accept": prop(len(acc), len(s100)),
                 "agreement_accepted": prop(sum(r["agree"] for r in acc), len(acc)),
                 "sampling_frame_in_round1_dataset": dict(frame),
                 "note": "pool rows left after round 1's repair = the rows the round-0 student was MOST "
                         "sure of (repair labels the least-sure ones); held-out rows were already used "
                         "for certification. Not a random sample."}
    fl = jl(phase4 / "fresh-shadow-log.jsonl")
    meta = {m["doc_id"].split(":", 1)[1]: m for m in jl(fresh_meta)}
    assert len(fl) == 500 and all(r["input_sha256"] in meta for r in fl)
    def summ(rows):
        a = [r for r in rows if r["would_accept"]]
        return {"n": len(rows), "agreement_all": prop(sum(r["agree"] for r in rows), len(rows)),
                "would_accept": prop(len(a), len(rows)),
                "agreement_accepted": prop(sum(r["agree"] for r in a), len(a))}
    nonwiki = [r for r in fl if not meta[r["input_sha256"]]["source"].startswith("wikipedia")]
    by_src = {}
    for src in sorted({m["source"] for m in meta.values()}):
        rows = [r for r in fl if meta[r["input_sha256"]]["source"] == src]
        a = [r for r in rows if r["would_accept"]]
        by_src[src] = {"n": len(rows), "accepted": len(a), "agree_accepted": sum(r["agree"] for r in a)}
    ms = sorted(r["student_ms"] for r in fl[1:])
    return {"shadow100": shadow100,
            "fresh500": {"all": summ(fl), "excluding_wikipedia": summ(nonwiki), "by_source": by_src,
                         "student_ms_excluding_first": {"n": len(ms), "p50": ms[len(ms) // 2],
                                                        "p95": ms[int(0.95 * (len(ms) - 1))],
                                                        "mean": sum(ms) / len(ms), "max": ms[-1]}}}


# ------------------------------------------------------------------ F: zero-shot
def zeroshot_section(path: Path) -> dict:
    R = json.loads(path.read_text())["routed"]
    tasks = {
        "sentiment_5level": [(r["gold"], r["pred"], r["conf"], r["ok"], r["id"]) for r in R["sentiment"]["rows"]],
        "sarcastic_or_mixed_noul": [(r["sm_gold"], r["sm_p"] >= 0.5, max(r["sm_p"], 1 - r["sm_p"]), r["sm_ok"], r["id"])
                                    for r in R["sentiment"]["rows"]],
        "skill_routing": [(r["gold"], r["pred"], r["conf"], r["ok"], r["id"]) for r in R["routing"]["rows"]],
        "purchase_noul": [(r["pur_gold"] if "pur_gold" in r else None, None, r["pur_conf"], r["pur_ok"], r["id"])
                          for r in R["extraction"]["rows"]],
        "merchant_noul": [(r["mer_gold"] if "mer_gold" in r else None, None, r["mer_conf"], r["mer_ok"], r["id"])
                          for r in R["extraction"]["rows"]],
    }
    # gold for the extraction rows comes from gold.py (the results rows carry ok flags)
    import importlib.util
    spec = importlib.util.spec_from_file_location("gold", path.parent / "gold.py")
    gold = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gold)
    gold_ext = {e[0]: (e[2], e[3]) for e in gold.EXTRACTION}
    tasks["purchase_noul"] = [(gold_ext[i][0], None, c, ok, i) for (_, _, c, ok, i) in tasks["purchase_noul"]]
    tasks["merchant_noul"] = [(gold_ext[i][1], None, c, ok, i) for (_, _, c, ok, i) in tasks["merchant_noul"]]
    res, pvals = {}, {}
    for name, rows in tasks.items():
        n = len(rows)
        x = sum(1 for r in rows if r[3])
        cnt = collections.Counter(r[0] for r in rows)
        top = max(cnt.values())
        majority_labels = sorted((k for k, v in cnt.items() if v == top), key=str)
        mc = {}
        for ml in majority_labels:  # paired McNemar against each tied majority rule
            b = sum(1 for r in rows if r[3] and r[0] != ml)      # laya right, rule wrong
            c = sum(1 for r in rows if not r[3] and r[0] == ml)  # laya wrong, rule right
            a = sum(1 for r in rows if r[3] and r[0] == ml)
            d = n - a - b - c
            mc[str(ml)] = {"a": a, "b": b, "c": c, "d": d, "p_exact": S.mcnemar_exact(b, c),
                           "newcombe_paired_95": S.newcombe_paired(a, b, c, d)}
        worst = max(v["p_exact"] for v in mc.values())
        pvals[name] = worst
        res[name] = {"laya": prop(x, n), "majority_rate": top / n, "majority_labels": [str(m) for m in majority_labels],
                     "exact_binomial_vs_fixed_majority_rate_p_greater": S.binom_test_greater(x, n, top / n),
                     "mcnemar_vs_majority_rule": mc}
    holm = S.holm(pvals)
    for k in res:
        res[k]["mcnemar_p_used_for_family"] = pvals[k]
        res[k]["holm_adjusted_p"] = holm[k]
    # AUROC, pooled as in gating.py: positive class = correct decision, score = answer_confidence
    pairs, clusters = [], collections.OrderedDict()
    for name, rows in tasks.items():
        for (_, _, conf, ok, iid) in rows:
            pairs.append((conf, bool(ok), name))
            clusters.setdefault(iid, []).append((conf, bool(ok)))
    pos = [c for c, o, _ in pairs if o]
    neg = [c for c, o, _ in pairs if not o]
    a = S.auroc(pos, neg)
    se_hm, lo_hm, hi_hm = S.hanley_mcneil(a, len(pos), len(neg))
    _, se_dl, lo_dl, hi_dl = S.delong(pos, neg)
    lo_b, hi_b, sk_b = S.bootstrap_auroc([[(c, o)] for c, o, _ in pairs], 10000, "laya-auroc-items-v1")
    lo_c, hi_c, sk_c = S.bootstrap_auroc(list(clusters.values()), 10000, "laya-auroc-clusters-v1")
    per_task = {}
    for name in tasks:
        p = [c for c, o, t in pairs if t == name and o]
        q = [c for c, o, t in pairs if t == name and not o]
        per_task[name] = {"n_pos": len(p), "n_neg": len(q), "auroc": S.auroc(p, q) if p and q else None}
    auc = {"positive_class": "decision correct (matches the hand label)", "score": "answer_confidence "
           "(noul: max(P, 1-P))", "n": len(pairs), "n_pos": len(pos), "n_neg": len(neg), "auroc": a,
           "hanley_mcneil_95": [lo_hm, hi_hm], "hanley_mcneil_se": se_hm,
           "delong_95": [lo_dl, hi_dl], "delong_se": se_dl,
           "bootstrap_items_95": [lo_b, hi_b], "bootstrap_items_skipped": sk_b,
           "bootstrap_clusters_95": [lo_c, hi_c], "n_clusters": len(clusters), "bootstrap_clusters_skipped": sk_c,
           "per_task": per_task,
           "caveat": "63 decisions from 38 texts (sentiment and extraction texts contribute 2 decisions each), "
                     "5 heterogeneous tasks pooled; the cluster bootstrap respects the text grouping"}
    return {"tasks": res, "family": "5 zero-shot tasks, laya routed vs the in-sample majority-class rule, "
                                    "exact McNemar (worst case over tied majority labels), Holm",
            "auroc": auc}


# ------------------------------------------------------------------ report
def fmt(p, d=3):
    return "-" if p is None else f"{p:.{d}f}"


def ci(v, d=3):
    return f"[{v[0]:.{d}f}, {v[1]:.{d}f}]"


def markdown(res: dict) -> str:
    L = ["# laya re-analysis tables (generated by laya_stats.py)", ""]
    L += ["## A. Deterministic checks per GPU job (retained run directories)", "",
          "| job | dataset | mode | parity max abs prob diff | argmax | golden n | tol | T applied (choice) | calib n |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for jid, j in res["runs"]["jobs"].items():
        L.append(f"| {jid[:12]} | {j['dataset']} | {j['memory_mode']} | {j['parity']['max_abs_prob_diff']:.2e} | "
                 f"{j['parity']['argmax_agreement']} | {j['golden']['n']} | {j['golden']['tolerance']} | "
                 f"{j['calibration']['temperature_applied'][0]:.3f} | {j['calibration']['n']} |")
    L += ["", "## B. Timings (single observations; all values listed)", "",
          "| job | round | train rows | hold s (log) | hold s (job.json, int) | training s | parity s | calib s | peak VRAM GiB (torch) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for jid, j in res["runs"]["jobs"].items():
        t = j["phase_timings_s"]
        rnd = (j["metadata"] or {}).get("round", "-")
        L.append(f"| {jid[:12]} | {rnd} | {j['counts']['train']} | {fmt(j['hold_s_trainer_log'], 1)} | "
                 f"{j['hold_s_jobjson_int']} | {t['training']} | {t['checking_parity']} | {t['calibrating']} | "
                 f"{j['peak_vram_gib_torch']} |")
    for jid, h in res["runs"]["log_only_jobs_hold_s"].items():
        L.append(f"| {jid[:12]} (dir deleted) | ref | 50 | {h:.1f} | - | - | - | - | - |")
    hm = res["hold_model"]
    L += ["", f"Cap / longest campaign hold = {hm['cap_over_max_hold']:.2f}; cap / shortest = {hm['cap_over_min_hold']:.2f}. "
          f"Two-point linear model: fixed {hm['two_point_linear']['fixed_s']:.1f} s + "
          f"{hm['two_point_linear']['variable_s_at_round0_scale']:.1f} s per round-0-sized dataset; "
          f"900 s at about {hm['two_point_linear']['scale_at_900s']:.1f} x round 0's labeled rows (extrapolation).", ""]
    cs = [c for c in res["teacher_cold_starts"] if c["waiting_at_utc"] >= "2026-09-25"]
    L += ["Teacher cold starts on 2026-09-25 (router waiting -> ready, 1 s resolution): "
          + ", ".join(f"{int(c['seconds'])}" for c in cs) + " s", ""]
    g = res["gate"]
    L += ["## C. Ship gate on held-out (aiagent eval reports; estimand = agreement with the teacher's label)", "",
          f"Held-out n = {g['heldout']['n']}; teacher unanimous (3 of 3) on {g['heldout']['teacher_unanimous_k3']}; "
          f"majority-class rate {g['heldout']['majority_class_rate']:.3f}.", "",
          "| round | target | tau | precision x/n | precision | CP 95% two-sided | CP lower one-sided 95% (aiagent) | coverage x/n | coverage CP 95% | accuracy x/n | accuracy CP 95% | verdict |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r, v in g["rounds"].items():
        p, c, a = v["precision"], v["coverage"], v["accuracy"]
        L.append(f"| {r} | {v['target']} | {v['tau']:.4f} | {p['x']}/{p['n']} | {p['p']:.4f} | {ci(p['cp95'], 4)} | "
                 f"{p['cp_lower_one_sided_95']:.4f} ({v['precision_lower_reported_by_aiagent']:.4f}) | "
                 f"{c['x']}/{c['n']} | {ci(c['cp95'])} | {a['x']}/{a['n']} | {ci(a['cp95'])} | {v['verdict']} |")
    L += ["", "| comparison | accuracy | net items | Newcombe unpaired 95% | smallest possible exact McNemar p |",
          "| --- | --- | --- | --- | --- |"]
    for k, v in g["round_comparisons"].items():
        d = v["newcombe_unpaired_95"]
        L.append(f"| {k} | {v['accuracy'][0]:.4f} -> {v['accuracy'][1]:.4f} | {v['net_items']:+d} | "
                 f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] | {v['mcnemar_smallest_possible_p']:.4f} |")
    s = g["selection_sensitivity"]["r1_81_of_84"]
    L += ["", f"81/84 lower bounds: one-sided 95% {s['one_sided_lower_alpha_0.05']:.4f}; two-sided 95% "
          f"{s['two_sided_95_lower']:.4f}; Bonferroni over 3 rounds (one-sided alpha 0.0167) "
          f"{s['bonferroni_3_rounds_one_sided_lower_alpha_0.05/3']:.4f}.", ""]
    sh = res["shadow"]
    L += ["## D/E. Shadow runs of the installed round-1 student", "",
          "| set | n | agreement all | would_accept | agreement among accepted | CP 95% (accepted) | CP lower one-sided 95% |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, v in (("shadow100 (biased frame)", sh["shadow100"]), ("fresh500", sh["fresh500"]["all"]),
                    ("fresh500 excluding Wikipedia", sh["fresh500"]["excluding_wikipedia"])):
        a, w, aa = v["agreement_all"], v["would_accept"], v["agreement_accepted"]
        L.append(f"| {name} | {v['n']} | {a['x']}/{a['n']} = {a['p']:.3f} | {w['x']}/{w['n']} = {w['p']:.3f} | "
                 f"{aa['x']}/{aa['n']} = {aa['p']:.3f} | {ci(aa['cp95'])} | {aa['cp_lower_one_sided_95']:.3f} |")
    L += ["", f"shadow100 sampling frame (round-1 dataset split of each text): {sh['shadow100']['sampling_frame_in_round1_dataset']}",
          "", "## F. Zero-shot (hand-labelled, routed laya vs majority-class rule)", "",
          "| task | laya x/n | CP 95% | majority rate | exact binomial p (H1: > majority rate) | McNemar b,c (worst tie) | exact McNemar p | Holm p |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, v in res["zeroshot"]["tasks"].items():
        worst = max(v["mcnemar_vs_majority_rule"].values(), key=lambda m: m["p_exact"])
        L.append(f"| {k} | {v['laya']['x']}/{v['laya']['n']} | {ci(v['laya']['cp95'])} | {v['majority_rate']:.3f} | "
                 f"{v['exact_binomial_vs_fixed_majority_rate_p_greater']:.3f} | {worst['b']},{worst['c']} | "
                 f"{worst['p_exact']:.3f} | {v['holm_adjusted_p']:.3f} |")
    au = res["zeroshot"]["auroc"]
    L += ["", f"AUROC (positive = correct decision, score = answer_confidence): {au['auroc']:.3f}, n_pos {au['n_pos']}, "
          f"n_neg {au['n_neg']}; Hanley-McNeil 95% {ci(au['hanley_mcneil_95'])}; DeLong 95% {ci(au['delong_95'])}; "
          f"bootstrap (decisions, 10,000) {ci(au['bootstrap_items_95'])}; cluster bootstrap ({au['n_clusters']} texts) "
          f"{ci(au['bootstrap_clusters_95'])}.", "",
          "Per task: " + "; ".join(f"{k} {fmt(v['auroc'])} ({v['n_pos']}+/{v['n_neg']}-)" for k, v in au["per_task"].items()),
          "", "## G. Dynamic int8 vs fp32 (aiagent research, base checkpoint)", "",
          f"argmax agreement 97/133 = {97/133:.3f}; CP 95% {ci(res['int8']['cp95'])} (descriptive: a fixed probe set, "
          "not a random sample)."]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    for k, v in D.items():
        ap.add_argument("--" + k.replace("_", "-"), default=v)
    a = ap.parse_args()
    res = {"self_tests": "see tests/python/test_stats_statlib.py"}
    res["runs"] = runs_section(Path(a.runs), Path(a.trainer_log))
    res["hold_model"] = hold_model(res["runs"]["jobs"])
    res["teacher_cold_starts"] = teacher_cold_starts(Path(a.router_log))
    res["gate"] = gate_section(Path(a.phase4), Path(a.datasets))
    res["shadow"] = shadow_sections(Path(a.shadow), Path(a.phase4), Path(a.datasets), Path(a.fresh_meta))
    res["zeroshot"] = zeroshot_section(Path(a.zeroshot))
    res["int8"] = prop(97, 133)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "laya_stats.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    (out / "laya_stats.md").write_text(markdown(res))
    print(markdown(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
