"""Bandwidth-bound decode ceiling: the docs' model, re-derived with explicit
assumptions and propagated input uncertainty.

Read-only. Stdlib only. Deterministic.

    python3 perf_ceiling.py /var/cache/devai/vllm out/engine_runs.json out/

Model (batch 1, MTP off, short context):
    tokens/s <= B / bytes_per_token
    bytes_per_token = bytes of every weight tensor READ for one decode step
                    = all transformer-layer weights + lm_head (+ norms)
    NOT included: the input embedding table (one row is gathered per token,
    not the whole table), vision tower, MTP head (MTP off), KV / recurrent
    state (short prompts: < 1 % of weight bytes, computed below).
Utilisation u = measured_tokens_per_s * bytes_per_token / B. u is an upper
bound on the fraction of peak bandwidth used for weights only if the token
count is exact; the bench's token count is an estimate (chars/4).

Per-token bytes come from the safetensors headers (tensor byte ranges) of the
checkpoints still on disk; for checkpoints no longer on disk they are
computed from public parameter counts (stated below).
"""
from __future__ import annotations

import glob
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from statlib import rnd  # noqa: E402

BANDWIDTHS = {
    "640 GB/s (docs; source given only as bench-results.md, no primary source)": 640e9,
    "672 GB/s (192-bit x 28 Gbps GDDR7 -- a commonly published figure; NOT verified here)": 672e9,
}


def header_bytes(ckpt_dir):
    """{category: bytes} from all *.safetensors headers in a directory."""
    cats = {}
    for f in sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            if k.startswith("lm_head"):
                c = "lm_head (read every step)"
            elif "embed_tokens" in k:
                c = "embed_tokens (one row gathered per step; NOT counted)"
            elif k.startswith("model.visual") or ".visual." in k:
                c = "vision tower (NOT counted, text-only serving)"
            elif k.startswith("mtp."):
                c = "MTP head / draft head (NOT counted, MTP off)"
            else:
                c = "language-model layers + norms (read every step)"
            cats[c] = cats.get(c, 0) + (b - a)
    return cats


def per_token(cats):
    return sum(v for k, v in cats.items() if "read every step" in k)


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)  # usage
    store, engine_json, outdir = sys.argv[1:4]
    os.makedirs(outdir, exist_ok=True)
    eng = json.load(open(engine_json))
    rows = []

    # ---- checkpoints on disk ----
    on_disk = {
        "Qwen3.8-27B-W4A16-devai-AutoRound": {
            "measured": [("engine-counted, 2026-09-22, MTP off, >=128-token outputs among 39 latency prompts, median of per-request rates (TTFT-adjusted)",
                          eng["leak_runs"]["AR_mtp_off_leak"]["engine_rate_ttft_adjusted_generated_ge_128"]["median"],
                          eng["leak_runs"]["AR_mtp_off_leak"]["engine_rate_ttft_adjusted_generated_ge_128"]["median_ci95"])]},
        "Qwen3.8-27B-MTP-devai-NVFP4": {
            "measured": [("engine-counted, 2026-09-22, MTP off, >=128-token outputs among 39 latency prompts, median of per-request rates (TTFT-adjusted)",
                          eng["leak_runs"]["NV_mtp_off_leak"]["engine_rate_ttft_adjusted_generated_ge_128"]["median"],
                          eng["leak_runs"]["NV_mtp_off_leak"]["engine_rate_ttft_adjusted_generated_ge_128"]["median_ci95"])]},
        "Qwen3.5-9B-NVFP4": {
            "measured": [("bench tps_sustained_p50 (chars/4), 2026-05-02 TPS-fix table", 55.73, None),
                         ("bench tps_sustained_p50 (chars/4), 2026-05-05 sweep @131072", 55.27, None),
                         ("bench tps_sustained_p50 (chars/4), 2026-07-20 row @262144", 57.55, None)]},
    }
    for name, spec in on_disk.items():
        d = os.path.join(store, name)
        cats = header_bytes(d)
        bpt = per_token(cats)
        for label, val, ci in spec["measured"]:
            for bw_label, bw in BANDWIDTHS.items():
                rows.append({"model": name, "bytes_per_token_GB": bpt / 1e9, "bytes_source": "safetensors headers",
                             "measurement": label, "tokens_per_s": val, "tokens_per_s_ci95": ci,
                             "bandwidth": bw_label, "ceiling_tok_s": bw / bpt,
                             "utilisation": val * bpt / bw,
                             "utilisation_ci95": None if not ci else [ci[0] * bpt / bw, ci[1] * bpt / bw],
                             "components_GB": {k: v / 1e9 for k, v in cats.items()}})

    # ---- checkpoints no longer on disk: public parameter counts ----
    # Qwen3-8B: 8.2 B total, 6.95 B non-embedding (Qwen3 model card);
    # vocab 151936 x hidden 4096 = 0.622 B per matrix. NVFP4 = 4 bit + one
    # FP8 scale per 16 values = 4.5 bit = 0.5625 B/param; lm_head BF16
    # (quantization_config.ignore). The catalog's on-disk size "5.96 GB" equals
    # 6.40e9 bytes = body 3.91 GB + TWO BF16 vocab matrices 2 x 1.244 GB, i.e.
    # it is consistent with UNTIED embed_tokens / lm_head (docs say tied).
    # Per-token bytes are ~5.15 GB either way (the embedding table is gathered,
    # not streamed).
    q3_body = 6.95e9 * 0.5625
    q3_lm = 151936 * 4096 * 2
    q3 = q3_body + q3_lm
    # R1-Distill-Llama-8B: Llama-3.1-8B shape, 8.03 B params incl. untied
    # embed (128256 x 4096) and lm_head, BF16.
    ll_total = 8.03e9 * 2
    ll_embed = 128256 * 4096 * 2
    ll = ll_total - ll_embed
    offdisk = [
        ("Qwen3-8B-NVFP4", q3, "public parameter counts (see comment)",
         [("docs: 98.3 (2026-05-02, bench, chars/4)", 98.31), ("2026-05-05 sweep @131072 (bench, chars/4)", 102.09),
          ("2026-07-17/19 cache row @32768 (bench, chars/4)", 110.87)]),
        ("DeepSeek-R1-Distill-Llama-8B (BF16)", ll, "public parameter counts; docs use 16 GB (total incl. embedding table)",
         [("2026-05-02 TPS-fix table (bench, chars/4)", 42.51), ("2026-05-05 sweep (bench, chars/4)", 42.48)]),
    ]
    for name, bpt, src, meas in offdisk:
        for label, val in meas:
            for bw_label, bw in BANDWIDTHS.items():
                rows.append({"model": name, "bytes_per_token_GB": bpt / 1e9, "bytes_source": src,
                             "measurement": label, "tokens_per_s": val, "tokens_per_s_ci95": None,
                             "bandwidth": bw_label, "ceiling_tok_s": bw / bpt,
                             "utilisation": val * bpt / bw, "utilisation_ci95": None})

    # doc's own numbers for comparison
    doc = {
        "Qwen3-8B doc": {"bytes_GB": 5.1, "ceiling": 640 / 5.1, "measured": 98.3, "u": 98.3 * 5.1 / 640},
        "BF16 doc": {"bytes_GB": 16.0, "ceiling": 640 / 16.0, "measured": 42.5, "u": 42.5 * 16 / 640},
        "Qwen3.5-9B doc": {"ceiling_stated": 110, "implied_bytes_GB": 640 / 110, "measured": 55.7},
    }

    # KV / state term at short context for Qwen3-8B (docs' 72 KiB/token FP8)
    kv_short = {"tokens_in_context_max": 50 + 256, "kv_bytes_per_token": 73728,
                "kv_bytes_read_last_step_GB": (50 + 256) * 73728 / 1e9,
                "fraction_of_weight_bytes": (50 + 256) * 73728 / q3}

    # Two-build test of the model: rates should scale as 1/bytes.
    ar = [r for r in rows if r["model"] == "Qwen3.8-27B-W4A16-devai-AutoRound"][0]
    nv = [r for r in rows if r["model"] == "Qwen3.8-27B-MTP-devai-NVFP4"][0]
    pred_ratio = nv["bytes_per_token_GB"] / ar["bytes_per_token_GB"]
    obs = eng["build_paired_AR_over_NV"]["mtp_off"]
    two_build = {"predicted_rate_ratio_AR_over_NV_from_bytes": pred_ratio,
                 "observed_paired_geometric_mean_ratio": obs["geometric_mean_ratio"],
                 "observed_ci95": obs["geometric_mean_ratio_boot_ci95"],
                 "note": "paired over 34 latency prompts, same day, MTP off, engine-counted tokens"}

    out = {"rows": rows, "doc_numbers": doc, "kv_short_context": kv_short, "two_build_test": two_build}
    # added 2026-09-27: the cross-model NVFP4 8B / BF16 8B figure cited in
    # docs/llm-tokens-and-speed.md (single runs, different models)
    q3_meas = [v for _, v in offdisk[0][3]]
    ll_meas = [v for _, v in offdisk[1][3]]
    ratios = [a / b for a in q3_meas for b in ll_meas]
    # two-build test: the same ratio on each subset, and the prediction if
    # only the utilised fraction of a step scales with bytes
    ar128 = eng["leak_runs"]["AR_mtp_off_leak"]["engine_rate_ttft_adjusted_generated_ge_128"]["median"]
    nv128 = eng["leak_runs"]["NV_mtp_off_leak"]["engine_rate_ttft_adjusted_generated_ge_128"]["median"]
    ar64 = eng["mtp_paired"]["AutoRound"]["off_rate"]["median"]
    nv64 = eng["mtp_paired"]["NVFP4"]["off_rate"]["median"]
    u_mean = (ar["utilisation"] + nv["utilisation"]) / 2
    out["two_build_subsets_and_adjusted_prediction"] = {
        "ratio_of_medians_gen_ge_128": ar128 / nv128,
        "ratio_of_medians_34_pairs_gen_ge_64": ar64 / nv64,
        "paired_geometric_mean_34_pairs": obs["geometric_mean_ratio"],
        "mean_utilisation_at_640": u_mean,
        "utilisation_34_pair_medians_at_640": {"AutoRound": ar64 * ar["bytes_per_token_GB"] * 1e9 / 640e9,
                                               "NVFP4": nv64 * nv["bytes_per_token_GB"] * 1e9 / 640e9},
        "prediction_pure_bytes": pred_ratio,
        "prediction_utilisation_adjusted": 1 + (pred_ratio - 1) * u_mean,
        # same prediction with the utilisations of the 34-pair subset the
        # docs' two-build table uses (one subset throughout)
        "prediction_utilisation_adjusted_34_pairs": 1 + (pred_ratio - 1) * (
            ar64 * ar["bytes_per_token_GB"] * 1e9 / 640e9 + nv64 * nv["bytes_per_token_GB"] * 1e9 / 640e9) / 2,
        "note": "the CI covers prompts within one run per build; run-to-run noise (about 1-2 % per run) "
                "exceeds the 1 % shortfall"}
    out["cross_model_qwen3_8b_nvfp4_over_r1_llama_8b_bf16"] = {
        "measured_ratio_min": min(ratios), "measured_ratio_max": max(ratios),
        "byte_ratio_predicted": ll / q3}
    with open(os.path.join(outdir, "ceiling.json"), "w") as fh:
        json.dump(rnd(out, 4), fh, indent=1)
    lines = ["| model | per-token GB (source) | measurement | tok/s | bandwidth | ceiling tok/s | utilisation |",
             "|---|---|---|---:|---|---:|---:|"]
    for r in rows:
        ci = "" if not r["utilisation_ci95"] else f" [{r['utilisation_ci95'][0]:.3f}, {r['utilisation_ci95'][1]:.3f}]"
        lines.append(f"| {r['model']} | {r['bytes_per_token_GB']:.3f} ({r['bytes_source'][:28]}) | {r['measurement'][:60]} | {r['tokens_per_s']:.2f} | {r['bandwidth'][:9]} | {r['ceiling_tok_s']:.1f} | {r['utilisation']:.3f}{ci} |")
    tb = out["two_build_subsets_and_adjusted_prediction"]
    lines += ["", f"Two-build: ratio of medians gen>=128 {tb['ratio_of_medians_gen_ge_128']:.3f}, 34 pairs "
                  f"{tb['ratio_of_medians_34_pairs_gen_ge_64']:.3f}, paired geo-mean {tb['paired_geometric_mean_34_pairs']:.3f}; "
                  f"predicted {tb['prediction_pure_bytes']:.3f} (pure bytes), "
                  f"{tb['prediction_utilisation_adjusted']:.3f} (utilisation-adjusted, u={tb['mean_utilisation_at_640']:.3f}); "
                  f"{tb['prediction_utilisation_adjusted_34_pairs']:.3f} with the 34-pair utilisations"]
    cm = out["cross_model_qwen3_8b_nvfp4_over_r1_llama_8b_bf16"]
    lines += ["", f"Cross-model (single runs, different models): Qwen3-8B-NVFP4 / R1-Distill-Llama-8B tok/s "
                  f"{cm['measured_ratio_min']:.2f}-{cm['measured_ratio_max']:.2f}; byte ratio {cm['byte_ratio_predicted']:.2f}"]
    with open(os.path.join(outdir, "ceiling.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(rnd({"doc": doc, "kv_short": kv_short, "two_build": two_build}, 4), indent=1))


if __name__ == "__main__":
    main()
