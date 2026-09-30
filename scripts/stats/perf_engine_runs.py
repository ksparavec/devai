"""Per-request re-analysis of the Qwen3.8-27B MTP / throughput measurements
from the engine's own per-request log lines.

Read-only. Stdlib only. Deterministic (bootstrap seed in statlib).

    python3 perf_engine_runs.py /var/cache/devai/logs/devai-vllm.log \
        /var/cache/devai/logs/devai-vllm-devai.log \
        /home/sparavec/git/devai/scripts/bench/data/latency_prompts.jsonl out/

Source of the per-request data
------------------------------
The home-built vLLM 0.28 images carry HyperQwen's engine-completion-log
patch: one ``Request finished`` line per request with engine-counted
``generated_tokens`` and ``elapsed_s`` (engine wall time from arrival to
finish; it INCLUDES queueing and prefill, i.e. it is not a pure decode time).
Stock vLLM (port 11435 before 2026-09-21) has no such line, so this analysis
covers only runs made on the home-built images (2026-09-21 onward).

Per-request "engine rate" below = generated_tokens / elapsed_s. For the
TTFT-adjusted variant the run's harness steady TTFT p50 is subtracted from
elapsed_s (an approximation: per-request TTFT was not logged).

Windows (UTC) are the documented runs; each is identified by time window
and by the prompt-token / max_tokens pattern of the requests it contains.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_enginelog import iter_records, utc  # noqa: E402
from statlib import (boot_ci, boot_ci_two_sample, describe, median_ci,  # noqa: E402
                        quantile, rnd)

# Harness (client-side) values reported for the same runs, copied from the
# retained client logs (see REPORT.md provenance table).
HARNESS = {
    "AR_mtp_on_leak": {"tps_sustained_p50": 95.13, "ttft_ms_steady_p50": 70.7,
                       "src": "~/.cache/devai/autoround-tests.log (2026-09-21 14:20 local)"},
    "NV_mtp_on_leak": {"tps_sustained_p50": 90.43, "ttft_ms_steady_p50": 72.3,
                       "src": "~/.cache/devai/nvfp4-prepared-tests.log (2026-09-21 23:01 local)"},
    "NV_mtp_off_leak": {"tps_sustained_p50": 36.76, "ttft_ms_steady_p50": 84.5,
                        "src": "~/.cache/devai/bench-derived.log line 20 = bench cache row"},
    "AR_mtp_off_leak": {"tps_sustained_p50": 38.77, "ttft_ms_steady_p50": 60.1,
                        "src": "~/.cache/devai/bench-derived.log line 90 = bench cache row"},
}

LEAK_RUNS = {
    # name: (log key, start, end, description)
    "AR_mtp_on_leak": ("vllm", "2026-09-21T12:23:20", "2026-09-21T12:24:31",
                       "Qwen3.8-27B-W4A16-AutoRound (prepared in place) ::mtp@131072, port 11435, image localhost/devai-vllm"),
    "NV_mtp_on_leak": ("vllm", "2026-09-21T21:06:00", "2026-09-21T21:07:17",
                       "Qwen3.8-27B-MTP-NVFP4 (prepared in place) ::mtp@98304, port 11435, image localhost/devai-vllm"),
    "NV_mtp_off_leak": ("devai", "2026-09-22T12:19:40", "2026-09-22T12:22:31",
                        "Qwen3.8-27B-MTP-devai-NVFP4 @118784, MTP off, port 11437 (bench-vllm-devai attempt 3)"),
    "AR_mtp_off_leak": ("devai", "2026-09-22T12:32:30", "2026-09-22T12:35:30",
                        "Qwen3.8-27B-W4A16-devai-AutoRound @131072, MTP off, port 11437 (bench-vllm-devai attempt 3)"),
}

CODE_RUNS = {
    # 6 x 512-token completions of ONE code prompt after one warm-up request.
    "NV_mtp_on_code_nothink": ("devai", "2026-09-22T13:16:50", "2026-09-22T13:17:24",
                               "::nothink::mtp, 2026-09-22 15:17 local, the '95.1 tok/s' figure"),
    "NV_mtp_on_code_policybug_1": ("devai", "2026-09-22T10:14:10", "2026-09-22T10:14:52",
                                   "::nothink silently ignored (router policy bug): thinking ON, the '74.8 tok/s' figure"),
    "NV_mtp_on_code_policybug_2": ("devai", "2026-09-22T10:15:20", "2026-09-22T10:16:05", "repeat"),
    "NV_mtp_on_code_policybug_3": ("devai", "2026-09-22T10:17:50", "2026-09-22T10:18:34", "repeat"),
}

CONC_RUNS = {
    # conc_decode.py: warm-up + 1 + 2 + 4 non-streaming requests, max_tokens 700,
    # default sampling (temperature not set), one batch per level.
    "AR_conc": ("vllm", "2026-09-21T12:28:05", "2026-09-21T12:28:41", "prepared AutoRound ::mtp@131072"),
    "NV_conc": ("vllm", "2026-09-21T21:07:52", "2026-09-21T21:08:26", "prepared NVFP4 ::mtp@98304"),
}

DEPTH_RUNS = {
    "AR_depth": ("vllm", "2026-09-21T12:24:31", "2026-09-21T12:25:35"),
    "NV_depth": ("vllm", "2026-09-21T21:07:17", "2026-09-21T21:07:52"),
}
# client-side values from the retained logs (DEPTH lines)
DEPTH_CLIENT = {
    "AR_depth": [(3932, 1840.8, 103.58), (10485, 3955.7, 105.19), (43253, 18655.1, 98.34), (91750, 34357.6, 88.46)],
    "NV_depth": [(3932, 713.0, 103.77), (10485, 1498.7, 102.4), (43253, 8506.9, 96.42), (91750, 19288.1, 90.53)],
}


def load_max_tokens(path, n=40):
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append(int(o.get("max_tokens", 256)))
    return out[:n]


def reqs(logs, key, a, b):
    return [r for r in iter_records(logs[key], utc(a), utc(b)) if r["type"] == "req"]


def spec(logs, key, a, b):
    return [r for r in iter_records(logs[key], utc(a), utc(b)) if r["type"] == "spec"]


def align_leak(records, max_tokens):
    """Assign engine records to prompt indices 0..39 in send order.

    A record matches prompt k when generated <= max_tokens[k] and, if the
    request stopped on length, generated == max_tokens[k]. Records that do not
    match the next expected prompt are treated as other clients' requests and
    skipped (reported)."""
    out, skipped, k = {}, [], 0
    for r in records:
        if k >= len(max_tokens):
            skipped.append(r)
            continue
        mt = max_tokens[k]
        ok = r["generated_tokens"] <= mt and (r["finish"] != "length" or r["generated_tokens"] == mt)
        if ok:
            out[k] = r
            k += 1
        else:
            skipped.append(r)
    return out, skipped


def rate(r, ttft_s=0.0):
    e = r["elapsed_s"] - ttft_s
    return r["generated_tokens"] / e if e > 0 else None


def gmean(xs):
    return math.exp(statistics.fmean(math.log(x) for x in xs))


def summarise_leak(name, recs, ttft_ms):
    # exclude prompt 0: its engine elapsed includes the first-request warm-up
    ks = sorted(k for k in recs if k != 0)
    raw = [rate(recs[k]) for k in ks]
    adj = [rate(recs[k], ttft_ms / 1000.0) for k in ks]
    long_ = [rate(recs[k], ttft_ms / 1000.0) for k in ks if recs[k]["generated_tokens"] >= 128]
    return {
        "n_prompts_aligned": len(recs), "n_used": len(ks),
        "prompt0_elapsed_s": recs.get(0, {}).get("elapsed_s"),
        "engine_rate_raw": describe(raw),
        "engine_rate_ttft_adjusted": describe(adj),
        "engine_rate_ttft_adjusted_generated_ge_128": describe(long_),
        "tokens_total": sum(recs[k]["generated_tokens"] for k in ks),
        "elapsed_total_s": sum(recs[k]["elapsed_s"] for k in ks),
        "ratio_of_sums_tok_per_s": sum(recs[k]["generated_tokens"] for k in ks) / sum(recs[k]["elapsed_s"] for k in ks),
    }


def paired_mtp(on, off, ttft_on_ms, ttft_off_ms, min_tokens=64):
    ks = sorted(k for k in on if k in off and k != 0
                and on[k]["generated_tokens"] >= min_tokens and off[k]["generated_tokens"] >= min_tokens)
    pairs = [(rate(on[k], ttft_on_ms / 1000), rate(off[k], ttft_off_ms / 1000)) for k in ks]
    ratios = [a / b for a, b in pairs]
    same_len = sum(1 for k in ks if on[k]["generated_tokens"] == off[k]["generated_tokens"])
    res = {
        "design": "paired by prompt index (same 40-prompt latency set, temperature 0); "
                  "MTP-on and MTP-off runs on different days, images and (NVFP4) contexts",
        "n_pairs": len(ks), "min_generated_tokens": min_tokens,
        "pairs_with_identical_generated_token_count": same_len,
        "ratio_on_over_off": describe(ratios),
        "geometric_mean_ratio": gmean(ratios),
        "geometric_mean_ratio_boot_ci95": list(boot_ci(ratios, gmean)),
        "median_ratio_ci95_order_stat": list(median_ci(ratios)[:2]),
        "on_rate": describe([a for a, _ in pairs]),
        "off_rate": describe([b for _, b in pairs]),
    }
    return res


def conc_table(records):
    """Assign warm-up/L1/L2/L4 by order of arrival (finish - elapsed)."""
    rs = sorted(records, key=lambda r: (r["t"].timestamp() - r["elapsed_s"]))
    # conc_decode.py prompts are 29-40 tokens with max_tokens 700; drop the
    # 8-token warm-up of the bench_concurrency sweep that follows it.
    rs = [r for r in rs if r["prompt_tokens"] < 100 and r["generated_tokens"] > 8]
    groups = {"warmup": rs[0:1], "L1": rs[1:2], "L2": rs[2:4], "L4": rs[4:8]}
    out = {}
    for g, lst in groups.items():
        toks = sum(r["generated_tokens"] for r in lst)
        mx = max(r["elapsed_s"] for r in lst)
        out[g] = {
            "n_requests": len(lst),
            "per_request": [{"prompt_tokens": r["prompt_tokens"], "generated": r["generated_tokens"],
                             "elapsed_s": r["elapsed_s"], "rate": round(rate(r), 1)} for r in lst],
            "aggregate_engine_tok_per_s": toks / mx,
            "per_stream_rate_min_max": [min(rate(r) for r in lst), max(rate(r) for r in lst)],
        }
    return out


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)  # usage
    vllm_log, devai_log, prompts, outdir = sys.argv[1:5]
    os.makedirs(outdir, exist_ok=True)
    logs = {"vllm": vllm_log, "devai": devai_log}
    mt = load_max_tokens(prompts)
    result = {"max_tokens_first_40": mt}

    # ---- leak runs (40 latency prompts) ----
    leak = {}
    aligned = {}
    for name, (key, a, b, desc) in LEAK_RUNS.items():
        recs = reqs(logs, key, a, b)
        al, skipped = align_leak(recs, mt)
        aligned[name] = al
        s = summarise_leak(name, al, HARNESS[name]["ttft_ms_steady_p50"])
        s["description"] = desc
        s["skipped_records"] = [{"prompt_tokens": r["prompt_tokens"], "generated": r["generated_tokens"],
                                 "elapsed_s": r["elapsed_s"]} for r in skipped]
        h = HARNESS[name]
        s["harness_tps_sustained_p50_chars_div_4"] = h["tps_sustained_p50"]
        s["harness_over_engine_median_adjusted"] = h["tps_sustained_p50"] / s["engine_rate_ttft_adjusted"]["median"]
        s["harness_src"] = h["src"]
        sp = spec(logs, key, a, b)
        acc = sum(x["accepted"] for x in sp)
        dra = sum(x["drafted"] for x in sp)
        if dra:
            s["spec_decode"] = {"windows": len(sp), "accepted": acc, "drafted": dra,
                                "draft_acceptance_rate": acc / dra,
                                "window_mean_acceptance_length": describe([x["mean_accept_len"] for x in sp])}
        leak[name] = s
    result["leak_runs"] = leak

    # ---- paired MTP on/off on the same prompts ----
    result["mtp_paired"] = {
        "AutoRound": paired_mtp(aligned["AR_mtp_on_leak"], aligned["AR_mtp_off_leak"],
                                HARNESS["AR_mtp_on_leak"]["ttft_ms_steady_p50"],
                                HARNESS["AR_mtp_off_leak"]["ttft_ms_steady_p50"]),
        "NVFP4": paired_mtp(aligned["NV_mtp_on_leak"], aligned["NV_mtp_off_leak"],
                            HARNESS["NV_mtp_on_leak"]["ttft_ms_steady_p50"],
                            HARNESS["NV_mtp_off_leak"]["ttft_ms_steady_p50"]),
    }
    # all prompts (>=16 tokens) sensitivity
    result["mtp_paired_all_lengths"] = {
        "AutoRound": paired_mtp(aligned["AR_mtp_on_leak"], aligned["AR_mtp_off_leak"], 70.7, 60.1, min_tokens=1),
        "NVFP4": paired_mtp(aligned["NV_mtp_on_leak"], aligned["NV_mtp_off_leak"], 72.3, 84.5, min_tokens=1),
    }
    # build comparison within MTP state (same prompts)
    result["build_paired_AR_over_NV"] = {
        "mtp_on": paired_mtp(aligned["AR_mtp_on_leak"], aligned["NV_mtp_on_leak"], 70.7, 72.3),
        "mtp_off": paired_mtp(aligned["AR_mtp_off_leak"], aligned["NV_mtp_off_leak"], 60.1, 84.5),
    }

    # ---- code-prompt runs (6 x 512) ----
    code = {}
    for name, (key, a, b, desc) in CODE_RUNS.items():
        recs = reqs(logs, key, a, b)
        warm, main_ = recs[0], recs[1:]
        main_ = [r for r in main_ if r["prompt_tokens"] == warm["prompt_tokens"]]
        rates = [rate(r) for r in main_]
        code[name] = {
            "description": desc, "n": len(main_),
            "per_request": [(r["generated_tokens"], r["elapsed_s"], round(rate(r), 1)) for r in main_],
            "ratio_of_sums": sum(r["generated_tokens"] for r in main_) / sum(r["elapsed_s"] for r in main_),
            "mean_of_rates": statistics.fmean(rates), "median_of_rates": quantile(rates, 0.5),
            "min": min(rates), "max": max(rates),
            "warmup_elapsed_s": warm["elapsed_s"],
        }
    result["code_prompt_runs"] = code

    # ---- concurrency runs ----
    result["conc_runs"] = {name: {"description": d, **conc_table(reqs(logs, key, a, b))}
                           for name, (key, a, b, d) in CONC_RUNS.items()}

    # ---- depth runs: actual prompt tokens vs the documented targets ----
    depth = {}
    for name, (key, a, b) in DEPTH_RUNS.items():
        recs = [r for r in reqs(logs, key, a, b) if r["prompt_tokens"] > 1000]
        rows = []
        for r, (target, ttft_ms, tps_client) in zip(recs, DEPTH_CLIENT[name]):
            rows.append({"target_tokens_doc": target, "actual_prompt_tokens": r["prompt_tokens"],
                         "actual_over_target": r["prompt_tokens"] / target,
                         "client_ttft_ms": ttft_ms,
                         "prefill_tok_per_s_actual": r["prompt_tokens"] / (ttft_ms / 1000),
                         "prefill_tok_per_s_target_based": target / (ttft_ms / 1000),
                         "engine_generated": r["generated_tokens"], "client_tps_during_decode": tps_client})
        depth[name] = rows
    ratios = [ar["client_ttft_ms"] / nv["client_ttft_ms"] for ar, nv in zip(depth["AR_depth"], depth["NV_depth"])]
    depth["ttft_ratio_AR_over_NV_per_depth"] = ratios
    depth["ttft_ratio_geometric_mean"] = gmean(ratios)
    result["depth_runs"] = depth

    # ---- figures cited by the docs that are derived from the above (added
    # 2026-09-27; every earlier key is unchanged) ----
    figs = {}
    pos = {}
    pooled = []
    for name in ("AR_mtp_on_leak", "NV_mtp_on_leak"):
        key, a, b, _ = LEAK_RUNS[name]
        wins = [x["per_position"] for x in spec(logs, key, a, b) if x.get("per_position")]
        pooled += wins
        pos[name] = {"windows": len(wins), "per_window": wins,
                     "median_per_position": [quantile([w[i] for w in wins], 0.5) for i in range(len(wins[0]))]}
    pos["pooled"] = {"windows": len(pooled),
                     "median_per_position": [quantile([w[i] for w in pooled], 0.5) for i in range(len(pooled[0]))],
                     "min_per_position": [min(w[i] for w in pooled) for i in range(len(pooled[0]))],
                     "max_per_position": [max(w[i] for w in pooled) for i in range(len(pooled[0]))]}
    figs["mtp_per_position_acceptance_10s_windows"] = pos
    figs["harness_over_engine_median_gen_ge_128"] = {
        name: s_["harness_tps_sustained_p50_chars_div_4"] / s_["engine_rate_ttft_adjusted_generated_ge_128"]["median"]
        for name, s_ in leak.items()}
    figs["depth_ms_per_prompt_token"] = {
        name: [1000.0 / r_["prefill_tok_per_s_actual"] for r_ in depth[name]] for name in ("AR_depth", "NV_depth")}
    off_pairs = result["mtp_paired"]["NVFP4"]["off_rate"]["median"]
    off_128 = leak["NV_mtp_off_leak"]["engine_rate_ttft_adjusted_generated_ge_128"]["median"]
    figs["code_policybug_median_over_nv_mtp_off_rate"] = {
        "assumption": "the MTP-off rate of NVFP4-devai is the same for the code prompt as for the latency prompts "
                      "(no MTP-off run of the code prompt exists)",
        "mtp_off_rate_paired_median": off_pairs, "mtp_off_rate_gen_ge_128_median": off_128,
        "ratios_vs_paired_median": {n: code[n]["median_of_rates"] / off_pairs
                                    for n in code if n.startswith("NV_mtp_on_code_policybug")},
        "ratios_vs_gen_ge_128_median": {n: code[n]["median_of_rates"] / off_128
                                        for n in code if n.startswith("NV_mtp_on_code_policybug")},
    }
    # MTP acceptance pooled over both MTP-on leak runs: mean tokens per
    # verification step = 1 + K x accepted / drafted (K = 3)
    sd_ar, sd_nv = leak["AR_mtp_on_leak"]["spec_decode"], leak["NV_mtp_on_leak"]["spec_decode"]
    pooled_rate = (sd_ar["accepted"] + sd_nv["accepted"]) / (sd_ar["drafted"] + sd_nv["drafted"])
    figs["mtp_pooled_acceptance"] = {"accepted_over_drafted": pooled_rate, "k": 3,
                                     "mean_tokens_per_step": 1 + 3 * pooled_rate}
    ext = (0.87, 0.72, 0.61)
    figs["external_qwen36_acceptance_example"] = {
        "rates": ext, "sum_if_unconditional": 1 + sum(ext),
        "product_if_conditional": 1 + ext[0] + ext[0] * ext[1] + ext[0] * ext[1] * ext[2]}
    # Output divergence MTP on vs off: only pairs where not both arms stopped
    # on length can show a length difference; prompt-token agreement checks
    # the reconstructed pairing.
    div = {}
    for b_, on_n, off_n in (("AutoRound", "AR_mtp_on_leak", "AR_mtp_off_leak"),
                            ("NVFP4", "NV_mtp_on_leak", "NV_mtp_off_leak")):
        on, off = aligned[on_n], aligned[off_n]
        ks = sorted(k for k in on if k in off and k != 0
                    and on[k]["generated_tokens"] >= 64 and off[k]["generated_tokens"] >= 64)
        both_len = [k for k in ks if on[k]["finish"] == "length" and off[k]["finish"] == "length"]
        inform = [k for k in ks if k not in both_len]
        div[b_] = {"pairs": len(ks), "prompt_tokens_equal": sum(1 for k in ks
                                                                 if on[k]["prompt_tokens"] == off[k]["prompt_tokens"]),
                   "both_stopped_on_length": len(both_len), "informative_pairs": len(inform),
                   "informative_pairs_with_different_length": sum(
                       1 for k in inform if on[k]["generated_tokens"] != off[k]["generated_tokens"])}
    figs["mtp_output_length_divergence"] = div
    # Depth runs: each prompt extends the previous one, so vLLM's prefix cache
    # serves part of it. Cumulative hit rate from the 10-s windows; hits implied
    # by it, assuming the rate covers every request since launch (queries =
    # prompt tokens of admitted requests; admitted = arrival <= window time).
    dp = {}
    for dname, lname in (("AR_depth", "AR_mtp_on_leak"), ("NV_depth", "NV_mtp_on_leak")):
        key, a, b = DEPTH_RUNS[dname]
        drecs = [r for r in reqs(logs, key, a, b) if r["prompt_tokens"] > 1000]
        wins = [w for w in iter_records(logs[key], utc(a), utc(b)) if w["type"] == "win"]
        q_leak = sum(r["prompt_tokens"] for r in aligned[lname].values())
        rows_ = []
        for w in wins:
            if not w.get("prefix_hit_rate_pct"):
                continue
            adm = [r for r in drecs if r["t"].timestamp() - r["elapsed_s"] <= w["t"].timestamp()]
            q = q_leak + sum(r["prompt_tokens"] for r in adm)
            rows_.append({"t_utc": w["t"].isoformat()[:19], "hit_rate_pct": w["prefix_hit_rate_pct"],
                          "admitted_depth_prompts": [r["prompt_tokens"] for r in adm],
                          "queries": q, "implied_hits": w["prefix_hit_rate_pct"] / 100 * q})
        ttft = [x[1] / 1000 for x in DEPTH_CLIENT[dname]]
        pt = [r["prompt_tokens"] for r in drecs]
        # Implied per-group rates: the first long prompt has nothing to reuse
        # (assumption: 0 hits, it shares less than one KV block with the short
        # prompts); the windows then separate {2nd, 3rd} from the 4th prompt.
        steps = []
        prev_n, prev_h = 1, 0.0
        for r_ in rows_:
            n_adm = len(r_["admitted_depth_prompts"])
            if n_adm > prev_n:
                idx = list(range(prev_n, n_adm))
                new_tok = sum(pt_ for pt_ in r_["admitted_depth_prompts"][prev_n:n_adm]) - (r_["implied_hits"] - prev_h)
                steps.append({"prompts": [p_ for p_ in r_["admitted_depth_prompts"][prev_n:n_adm]],
                              "implied_hits": r_["implied_hits"] - prev_h,
                              "implied_new_tokens": new_tok,
                              "implied_prefill_rate_new_tokens": new_tok / sum(
                                  DEPTH_CLIENT[dname][i][1] / 1000 for i in idx)})
                prev_n, prev_h = n_adm, r_["implied_hits"]
        dp[dname] = {
            "implied_by_hit_rate": steps,
            "windows": rows_,
            "prefill_rate_if_no_reuse": [p_ / t_ for p_, t_ in zip(pt, ttft)],
            "prefill_rate_if_previous_prompt_fully_reused": [pt[0] / ttft[0]] + [
                (pt[i] - pt[i - 1]) / ttft[i] for i in range(1, len(pt))],
        }
    figs["depth_prefix_cache"] = dp
    result["doc_figures"] = figs

    with open(os.path.join(outdir, "engine_runs.json"), "w") as fh:
        json.dump(rnd(result, 3), fh, indent=1, default=str)
    L = ["## Leak-prompt runs (engine per-request rates, prompts 1..39)", "",
         "| run | n | median raw | median TTFT-adj (95% CI) | median TTFT-adj, gen>=128 (95% CI, n) | harness p50 (chars/4) | harness / engine-adj |",
         "|---|---:|---:|---|---|---:|---:|"]
    for name, s_ in leak.items():
        a_, b_ = s_["engine_rate_ttft_adjusted"], s_["engine_rate_ttft_adjusted_generated_ge_128"]
        L.append(f"| {name} | {s_['n_used']} | {s_['engine_rate_raw']['median']:.1f} | {a_['median']:.1f} "
                 f"[{a_['median_ci95'][0]:.1f}, {a_['median_ci95'][1]:.1f}] | {b_['median']:.1f} "
                 f"[{b_['median_ci95'][0]:.1f}, {b_['median_ci95'][1]:.1f}], n={b_['n']} | "
                 f"{s_['harness_tps_sustained_p50_chars_div_4']} | {s_['harness_over_engine_median_adjusted']:.3f} |")
    L += ["", "## MTP on / off, paired by prompt (gen >= 64 tokens in both)", "",
          "| build | n pairs | same length | geo-mean ratio (95% boot CI) | median ratio (95% order-stat CI) | min..max |",
          "|---|---:|---:|---|---|---|"]
    for b, m in result["mtp_paired"].items():
        r_ = m["ratio_on_over_off"]
        L.append(f"| {b} | {m['n_pairs']} | {m['pairs_with_identical_generated_token_count']} | "
                 f"{m['geometric_mean_ratio']:.2f} [{m['geometric_mean_ratio_boot_ci95'][0]:.2f}, {m['geometric_mean_ratio_boot_ci95'][1]:.2f}] | "
                 f"{r_['median']:.2f} [{r_['median_ci95'][0]:.2f}, {r_['median_ci95'][1]:.2f}] | {r_['min']:.2f}..{r_['max']:.2f} |")
    L += ["", "## Code-prompt runs (one prompt, 5-6 replicate requests)", "",
          "| run | n | median | ratio of sums | min..max |", "|---|---:|---:|---:|---|"]
    for name, c in code.items():
        L.append(f"| {name} | {c['n']} | {c['median_of_rates']:.1f} | {c['ratio_of_sums']:.1f} | {c['min']:.1f}..{c['max']:.1f} |")
    L += ["", "## Concurrency (one batch per level, max_tokens 700)", "",
          "| run | level | n | aggregate tok/s (engine) | per-stream min..max |", "|---|---|---:|---:|---|"]
    for name, c in result["conc_runs"].items():
        for lvl in ("warmup", "L1", "L2", "L4"):
            g = c[lvl]
            L.append(f"| {name} | {lvl} | {g['n_requests']} | {g['aggregate_engine_tok_per_s']:.1f} | "
                     f"{g['per_stream_rate_min_max'][0]:.1f}..{g['per_stream_rate_min_max'][1]:.1f} |")
    L += ["", "## Depth runs (one request per depth)", "",
          "| run | doc target tokens | actual prompt tokens | TTFT ms | prefill tok/s (actual) | decode tok/s (client) |",
          "|---|---:|---:|---:|---:|---:|"]
    for name in ("AR_depth", "NV_depth"):
        for r_ in depth[name]:
            L.append(f"| {name} | {r_['target_tokens_doc']} | {r_['actual_prompt_tokens']} | {r_['client_ttft_ms']} | "
                     f"{r_['prefill_tok_per_s_actual']:.0f} | {r_['client_tps_during_decode']} |")
    L.append(f"\nTTFT ratio AR/NV per depth: {[round(x, 2) for x in ratios]}, geometric mean {gmean(ratios):.2f}")
    pp = figs["mtp_per_position_acceptance_10s_windows"]["pooled"]
    L += ["", "## Figures cited by the docs (derived from the tables above)", "",
          f"- MTP per-position acceptance, median over {pp['windows']} 10-s windows of the two MTP-on leak runs: "
          f"{[round(x, 3) for x in pp['median_per_position']]} (window range "
          f"{[round(x, 3) for x in pp['min_per_position']]} .. {[round(x, 3) for x in pp['max_per_position']]})",
          "- harness p50 (chars/4) / engine median (gen >= 128, TTFT-adjusted): "
          + ", ".join(f"{n} {v:.3f}" for n, v in figs["harness_over_engine_median_gen_ge_128"].items()),
          "- ms per prompt token (client TTFT / engine prompt tokens): "
          + "; ".join(f"{n} {[round(x, 3) for x in v]}" for n, v in figs["depth_ms_per_prompt_token"].items()),
          "- code-prompt policy-bug medians / NVFP4 MTP-off rate (assumes a prompt-independent MTP-off rate): "
          + ", ".join(f"{n} {v:.2f}" for n, v in
                      figs["code_policybug_median_over_nv_mtp_off_rate"]["ratios_vs_paired_median"].items())]
    pa = figs["mtp_pooled_acceptance"]
    L += [f"- MTP pooled accepted/drafted {pa['accepted_over_drafted']:.3f} -> "
          f"{pa['mean_tokens_per_step']:.2f} tokens per verification step (K=3)",
          "- output-length divergence MTP on vs off: " + "; ".join(
              f"{b_} {d['informative_pairs_with_different_length']}/{d['informative_pairs']} informative pairs "
              f"({d['both_stopped_on_length']} of {d['pairs']} pairs both stopped on length; prompt tokens equal "
              f"in {d['prompt_tokens_equal']})" for b_, d in figs["mtp_output_length_divergence"].items()),
          "- depth runs, prefill rate bounds (tok/s): " + "; ".join(
              f"{n} no reuse {[round(x) for x in v['prefill_rate_if_no_reuse']]}, previous prompt fully reused "
              f"{[round(x) for x in v['prefill_rate_if_previous_prompt_fully_reused']]}, implied by hit rate "
              f"{[round(x['implied_prefill_rate_new_tokens']) for x in v['implied_by_hit_rate']]}"
              for n, v in figs["depth_prefix_cache"].items())]
    with open(os.path.join(outdir, "engine_runs.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(json.dumps(rnd({k: v for k, v in result.items() if k in ("mtp_paired", "code_prompt_runs")}, 3), indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
