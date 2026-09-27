"""Re-analysis of the 2026-09-26 t/S measurement (aiagent sentiment PR 1).

Read-only. Stdlib only. Deterministic.

    python3 perf_ts.py /var/cache/devai/logs/devai-vllm-devai.log \
        <scratchpad>/measure.json out/

Client side (measure.json, produced by ~/laya-pilot/measure_ts.py): summary
only -- n=24, t_median_s, t_p90_s (index int(0.9*(n-1)) of the sorted
sequential times, i.e. the 21st of 24 order statistics), sequential_total_s,
parallel_total_s (one wall-clock realisation), parallel_call_median_s, S
(= sequential_total / parallel_total), failures. Per-call client times were
not saved.

Engine side: the vllm-devai engine logs one ``Request finished`` line per
request (engine-counted generated tokens, engine elapsed from arrival to
finish, 10 ms resolution). The run in the window 2026-09-26 09:39:03-09:40:12
UTC consists of 49 requests: 1 warm-up, 24 sequential, 24 at 4 in flight.
Engine elapsed excludes the client/DSPy/HTTP/router overhead, so it is a
lower bound on the client-side call time; the per-call overhead is estimated
as (client sequential total - engine sequential total) / 24.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_enginelog import iter_records, utc  # noqa: E402
from statlib import boot_ci, boot_ci_two_sample, describe, quantile, quantile_ci, rnd  # noqa: E402

START, END = "2026-09-26T09:39:03", "2026-09-26T09:40:15"


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)  # usage
    log, measure_json, outdir = sys.argv[1:4]
    os.makedirs(outdir, exist_ok=True)
    client = json.load(open(measure_json))
    recs = [r for r in iter_records(log, utc(START), utc(END)) if r["type"] == "req"]
    wins = [r for r in iter_records(log, utc(START), utc(END)) if r["type"] == "win"]
    specs = [r for r in iter_records(log, utc(START), utc(END)) if r["type"] == "spec"]
    n = client["n"]
    assert len(recs) == 1 + 2 * n, f"expected {1 + 2 * n} requests, found {len(recs)}"
    # Order by engine arrival time (finish - elapsed; finish has 1-s resolution).
    recs.sort(key=lambda r: (r["t"].timestamp() - r["elapsed_s"], r["t"].timestamp()))
    # The sequential phase finishes strictly one after another; the parallel
    # phase is the last 24 by arrival. Log order is finish order, so use it
    # for the sequential split (arrival estimates are +-1 s).
    by_finish = sorted(recs, key=lambda r: r["t"])
    warm = by_finish[0]
    seq = by_finish[1:1 + n]
    par = by_finish[1 + n:]
    seq_e = [r["elapsed_s"] for r in seq]
    par_e = [r["elapsed_s"] for r in par]
    seq_tok = [r["generated_tokens"] for r in seq]
    par_tok = [r["generated_tokens"] for r in par]
    seq_prompt = [r["prompt_tokens"] for r in seq]
    par_prompt = [r["prompt_tokens"] for r in par]

    def p90_index(xs):
        return sorted(xs)[int(0.9 * (len(xs) - 1))]

    eng_seq_total = sum(seq_e)
    overhead = (client["sequential_total_s"] - eng_seq_total) / n
    # achieved concurrency during the parallel phase: engine busy-seconds per
    # client wall-second (upper-bounded by max_num_seqs = 4)
    c_eff = sum(par_e) / client["parallel_total_s"]
    # Decomposition: S = (mean_seq_client / mean_par_client) * c_eff_client.
    # With engine times: S_eng = mean(seq_e) / mean(par_e) * c_eff (identity
    # when the parallel wall time is the client's).
    slow = statistics.fmean(par_e) / statistics.fmean(seq_e)
    tok_seq_rate = sum(seq_tok) / client["sequential_total_s"]
    tok_par_rate = sum(par_tok) / client["parallel_total_s"]

    res = {
        "window_utc": [START, END],
        "client_summary": client,
        "warmup": {"generated": warm["generated_tokens"], "elapsed_s": warm["elapsed_s"]},
        "engine_sequential_elapsed_s": describe(seq_e),
        "engine_sequential_elapsed_p90_index_def": p90_index(seq_e),
        "engine_sequential_elapsed_p90_type7": quantile(seq_e, 0.9),
        "p90_distribution_free_ci95": quantile_ci(seq_e, 0.9),
        "p90_index_def_is_order_statistic": f"{int(0.9 * (n - 1)) + 1} of {n} (empirical CDF {(int(0.9 * (n - 1)) + 1) / n:.3f})",
        "engine_sequential_total_s": eng_seq_total,
        "client_overhead_per_call_s_est": overhead,
        "client_mean_call_s": client["sequential_total_s"] / n,
        "engine_mean_call_plus_overhead_s": statistics.fmean(seq_e) + overhead,
        "mean_boot_ci95_client_scale_s": [x + overhead for x in describe(seq_e)["mean_boot_ci95"]],
        "median_ci95_client_scale_s": [x + overhead for x in describe(seq_e)["median_ci95"]],
        "engine_parallel_elapsed_s": describe(par_e),
        "sequential_generated_tokens": describe(seq_tok),
        "parallel_generated_tokens": describe(par_tok),
        "sequential_prompt_tokens": describe(seq_prompt),
        "parallel_prompt_tokens": describe(par_prompt),
        "tokens_parallel_over_sequential": sum(par_tok) / sum(seq_tok),
        "engine_rate_sequential_tok_per_s": describe([t / e for t, e in zip(seq_tok, seq_e)]),
        "engine_rate_parallel_tok_per_s": describe([t / e for t, e in zip(par_tok, par_e)]),
        "achieved_concurrency_c_eff": c_eff,
        "per_call_slowdown_par_over_seq_engine": slow,
        "per_call_slowdown_boot_ci95_unpaired": list(boot_ci_two_sample(
            seq_e, par_e, lambda a, b: statistics.fmean(b) / statistics.fmean(a))),
        "S_client": client["S"],
        "S_decomposed_engine": c_eff / slow,
        "S_token_normalised": tok_par_rate / tok_seq_rate,
        "S_bound_from_max_num_seqs": 4,
        "engine_windows_during_run": [{"t": w["t"].strftime("%H:%M:%S"), "gen_tps": w["gen_tps"],
                                       "running": w["running"], "waiting": w["waiting"]} for w in wins],
        "spec_decode": {"accepted": sum(s["accepted"] for s in specs),
                        "drafted": sum(s["drafted"] for s in specs),
                        "acceptance_rate": (sum(s["accepted"] for s in specs) /
                                            max(1, sum(s["drafted"] for s in specs))),
                        "window_mean_acceptance_length": [s["mean_accept_len"] for s in specs]},
        "per_request": {"sequential": [(r["prompt_tokens"], r["generated_tokens"], r["elapsed_s"]) for r in seq],
                        "parallel": [(r["prompt_tokens"], r["generated_tokens"], r["elapsed_s"]) for r in par]},
    }
    # Bootstrap CI for S that reflects sampling of calls (not of the single
    # parallel wall-clock realisation, which is held fixed through c_eff).
    res["S_decomposed_boot_ci95_calls_only"] = [c_eff / x for x in reversed(
        res["per_call_slowdown_boot_ci95_unpaired"])]
    with open(os.path.join(outdir, "ts.json"), "w") as fh:
        json.dump(rnd(res, 4), fh, indent=1)
    e = res["engine_sequential_elapsed_s"]
    md = ["| quantity | value |", "|---|---|",
          f"| client median t (s), as reported | {client['t_median_s']} |",
          f"| engine median t (s) + {overhead:.3f} s overhead, 95% order-stat CI | {e['median'] + overhead:.3f} [{res['median_ci95_client_scale_s'][0]:.3f}, {res['median_ci95_client_scale_s'][1]:.3f}] |",
          f"| client mean t (s) = total/24 | {client['sequential_total_s'] / n:.3f} |",
          f"| mean t, 95% bootstrap CI (s) | [{res['mean_boot_ci95_client_scale_s'][0]:.3f}, {res['mean_boot_ci95_client_scale_s'][1]:.3f}] |",
          f"| client p90 (index int(0.9*(n-1)), 21st of 24) | {client['t_p90_s']} |",
          f"| engine p90 type 7 + overhead | {res['engine_sequential_elapsed_p90_type7'] + overhead:.3f} |",
          f"| distribution-free 95% CI for p90 at n=24 | not attainable (max coverage {res['p90_distribution_free_ci95'].get('max_coverage_with_min_max', 0):.3f}) |",
          f"| S client | {client['S']} |",
          f"| achieved concurrency c_eff | {c_eff:.2f} |",
          f"| per-call slowdown under 4 in flight (95% CI) | {slow:.3f} [{res['per_call_slowdown_boot_ci95_unpaired'][0]:.3f}, {res['per_call_slowdown_boot_ci95_unpaired'][1]:.3f}] |",
          f"| S = c_eff / slowdown (95% CI, call sampling only) | {c_eff / slow:.2f} [{res['S_decomposed_boot_ci95_calls_only'][0]:.2f}, {res['S_decomposed_boot_ci95_calls_only'][1]:.2f}] |",
          f"| S token-normalised | {res['S_token_normalised']:.2f} |",
          f"| draft acceptance rate (MTP) | {res['spec_decode']['acceptance_rate']:.3f} |"]
    with open(os.path.join(outdir, "ts.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    show = {k: v for k, v in res.items() if k not in ("per_request", "engine_windows_during_run")}
    print(json.dumps(rnd(show, 3), indent=1))
    print("windows:", [(w["t"], w["gen_tps"], w["running"], w["waiting"]) for w in res["engine_windows_during_run"]])


if __name__ == "__main__":
    main()
