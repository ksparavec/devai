"""Latency / throughput / VRAM metrics in the bench cache: provenance flags,
what each statistic is, and what repeated runs say about run-to-run spread.

Read-only. Stdlib only. Deterministic.

    python3 perf_bench_cache.py /home/sparavec/git/devai/deploy/.bench-cache.json out/

Facts established from the code (cited in REPORT.md):
- tps_sustained_p50: per request, tokens / (t_done - t_first_token), where
  tokens = max(usage.completion_tokens, (len(content)+len(reasoning))//4);
  p50 = type-7 linear-interpolation median over the successful requests of
  ONE run of 40 fixed prompts (first prompt included). usage is never sent by
  vLLM / SGLang / Ollama for this request shape (no stream_options), so the
  token count is the chars/4 estimate for every row, before and after the
  2026-09-22 harness change.
- ttft_ms_first: TTFT of the run's first prompt. It is a COLD start only when
  the router had to launch the backend for that request.
- ttft_ms_steady_p50/p95: type-7 quantiles over prompts 2..40 (n=39).
- peak/mean_vram_gb: max / mean of device-wide nvidia-smi memory.used (MiB,
  divided by 1024, so GiB), sampled at 1 Hz from BEFORE the first request
  (cold start included) to the end of the run.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from statlib import boot_ci, describe, rnd  # noqa: E402

# Repeated observations of the same (model, backend) quoted in docs/bench-results.md
# (2026-05-02 TPS-fix validation table = "after fix"; 2026-05-05 full table)
# plus the current cache rows. Context and image may differ between runs.
REPEATS = {
    "Qwen3-14B-NVFP4 (vllm)": [("2026-05-02", 62.02), ("2026-05-05 @65536", 61.94), ("2026-07-19 @32768", 67.14)],
    "gpt-oss-20b (vllm)": [("2026-05-02", 136.37), ("2026-05-05 @262144", 139.2), ("2026-07-22 @131072", 135.01)],
    "Qwen3-8B-NVFP4 (vllm)": [("2026-05-02", 98.31), ("2026-05-05 @131072", 102.09), ("2026-07-19 @32768", 110.87)],
    "Qwen3.5-9B-NVFP4 (vllm)": [("2026-05-02", 55.73), ("2026-05-05 @131072", 55.27), ("2026-07-20 @262144", 57.55)],
    "DeepSeek-R1-Distill-Qwen-7B (vllm)": [("2026-05-02", 45.18), ("2026-05-05 @65536", 44.47)],
    "DeepSeek-R1-Distill-Llama-8B (vllm)": [("2026-05-02", 42.51), ("2026-05-05 @32768", 42.48)],
    "Nemotron-Nano-9B-v2-NVFP4 (vllm)": [("2026-05-02 before fix", 82.0), ("2026-05-02 after fix", 81.08),
                                          ("2026-05-05 @65536", 80.38), ("2026-07-20 @131072", 85.86)],
    "Llama-3.1-8B-Instruct-NVFP4 (vllm)": [("2026-05-02 before fix", 95.53), ("2026-05-02 after fix", 95.88),
                                            ("2026-05-05 @131072", 95.43)],
    "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 (vllm)": [("2026-05-05 @131072", 143.8), ("followup #11", 144.84),
                                                     ("2026-07-19 @163840", 40.25)],
}


def _chi2_cdf(x: float, k: int) -> float:
    """Regularised lower incomplete gamma P(k/2, x/2) by its power series."""
    if x <= 0:
        return 0.0
    a, z = k / 2.0, x / 2.0
    term = total = 1.0 / a
    n = 0
    while term > 1e-15 * total:
        n += 1
        term *= z / (a + n)
        total += term
    return math.exp(-z + a * math.log(z) - math.lgamma(a)) * total


def _chi2_quantile(p: float, k: int) -> float:
    lo, hi = 0.0, 1.0
    while _chi2_cdf(hi, k) < p:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if _chi2_cdf(mid, k) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)  # usage
    path, outdir = sys.argv[1:3]
    os.makedirs(outdir, exist_ok=True)
    cache = json.load(open(path))
    rows = []
    for k, r in cache.items():
        if k.startswith("_"):
            continue
        m = r.get("metrics", {})
        lp = r.get("tasks", {}).get("leak_probe", {})
        first = m.get("ttft_ms_first")
        rows.append({
            "key": k, "backend": r.get("backend"), "ctx": r.get("context"),
            "leak_task_ran_at": lp.get("ran_at"), "last_benched_at": r.get("last_benched_at"),
            "host_env_id": r.get("host_env_id"), "image_digest": (r.get("backend_image_digest") or "")[:19],
            "tps_sustained_p50": m.get("tps_sustained_p50"), "n_latency_samples": m.get("n_latency_samples"),
            "ttft_ms_first": first, "ttft_first_is_cold": None if first is None else first > 5000,
            "ttft_ms_steady_p50": m.get("ttft_ms_steady_p50"), "ttft_ms_steady_p95": m.get("ttft_ms_steady_p95"),
            "p95_over_p50": (m["ttft_ms_steady_p95"] / m["ttft_ms_steady_p50"]) if m.get("ttft_ms_steady_p50") else None,
            "peak_vram_gb": m.get("peak_vram_gb"), "mean_vram_gb": m.get("mean_vram_gb"),
            "vram_samples": m.get("vram_samples"),
            "mean_over_peak": (m["mean_vram_gb"] / m["peak_vram_gb"]) if m.get("peak_vram_gb") else None,
            "token_count_method": "chars/4 estimate (usage block never requested)",
        })

    # VRAM: does mean/peak depend on run length (cold-start share)?
    pairs = [(r["vram_samples"], r["mean_over_peak"]) for r in rows if r["vram_samples"] and r["mean_over_peak"]]

    def spearman(ps):
        def ranks(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            rk = [0.0] * len(v)
            i = 0
            while i < len(v):
                j = i
                while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                    j += 1
                for t in range(i, j + 1):
                    rk[order[t]] = (i + j) / 2 + 1
                i = j + 1
            return rk
        x = ranks([a for a, _ in ps])
        y = ranks([b for _, b in ps])
        mx, my = statistics.fmean(x), statistics.fmean(y)
        num = sum((a - mx) * (b - my) for a, b in zip(x, y))
        den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
        return num / den
    rho = spearman(pairs)
    rho_ci = boot_ci(pairs, spearman)
    short = [p for s, p in pairs if s < 200]
    long_ = [p for s, p in pairs if s >= 1000]

    # Run-to-run spread from repeated observations
    rep = {}
    logratios_0502_0505 = []
    for name, obs in REPEATS.items():
        vals = [v for _, v in obs]
        rep[name] = {"obs": obs, "max_over_min": max(vals) / min(vals),
                     "cv_percent": 100 * statistics.stdev(vals) / statistics.fmean(vals)}
        d = dict(obs)
        a = next((v for k, v in obs if k.startswith("2026-05-02") and "before" not in k), None)
        b = next((v for k, v in obs if k.startswith("2026-05-05")), None)
        if a and b:
            logratios_0502_0505.append(math.log(b / a))
    sd_lr = statistics.stdev(logratios_0502_0505)
    out = {
        "rows": rows,
        "vram_mean_over_peak_vs_samples": {
            "n_rows": len(pairs), "spearman_rho": rho, "spearman_boot_ci95": list(rho_ci),
            "mean_over_peak_rows_lt_200_samples": describe(short) if len(short) > 1 else short,
            "mean_over_peak_rows_ge_1000_samples": describe(long_) if len(long_) > 1 else long_,
        },
        "repeats": rep,
        "run_to_run_0502_vs_0505": {
            "n_models": len(logratios_0502_0505),
            "log_ratios": logratios_0502_0505,
            "sd_of_log_ratio": sd_lr,
            "implied_single_run_cv_percent": 100 * sd_lr / math.sqrt(2),
            "note": "each pair = two single runs of the same model/backend 3 days apart; context may differ",
        },
        # added 2026-09-27: chi-square 95 % interval for the implied
        # single-run CV (normal log ratios, df = n - 1), and the range of
        # the between-run differences, as cited in the docs
        "run_to_run_0502_vs_0505_interval": {
            "df": len(logratios_0502_0505) - 1,
            "implied_single_run_cv_percent_ci95": [
                100 * sd_lr / math.sqrt(2) * math.sqrt((len(logratios_0502_0505) - 1)
                                                       / _chi2_quantile(0.975, len(logratios_0502_0505) - 1)),
                100 * sd_lr / math.sqrt(2) * math.sqrt((len(logratios_0502_0505) - 1)
                                                       / _chi2_quantile(0.025, len(logratios_0502_0505) - 1))],
            "difference_percent_min_max": [100 * (math.exp(min(logratios_0502_0505)) - 1),
                                           100 * (math.exp(max(logratios_0502_0505)) - 1)],
            "note": "the 2026-05-02 values are quoted from bench-results.md, not retained; contexts differed",
        },
        "ttft_first_warm_rows": [r["key"] for r in rows if r["ttft_first_is_cold"] is False],
        "sglang_steady_ttft_p50": [(r["key"], r["ttft_ms_steady_p50"], r["ttft_ms_steady_p95"]) for r in rows if r["backend"] == "sglang"],
    }
    with open(os.path.join(outdir, "bench_cache.json"), "w") as fh:
        json.dump(rnd(out, 4), fh, indent=1)
    lines = ["| row | backend | ctx | leak task ran | TPS p50 | TTFT first ms (cold?) | TTFT p50/p95 ms | peak/mean VRAM GiB | samples |",
             "|---|---|---:|---|---:|---|---|---|---:|"]
    for r in rows:
        lines.append(f"| {r['key'][:48]} | {r['backend']} | {r['ctx']} | {str(r['leak_task_ran_at'])[:10]} | {r['tps_sustained_p50']} | {r['ttft_ms_first']} ({'cold' if r['ttft_first_is_cold'] else 'WARM'}) | {r['ttft_ms_steady_p50']}/{r['ttft_ms_steady_p95']} | {r['peak_vram_gb']}/{r['mean_vram_gb']} | {r['vram_samples']} |")
    with open(os.path.join(outdir, "bench_cache.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(rnd({k: v for k, v in out.items() if k != "rows"}, 4), indent=1))


if __name__ == "__main__":
    main()
