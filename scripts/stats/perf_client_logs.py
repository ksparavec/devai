"""Client-side logs of the 2026-09-21 agent-shape experiments
(~/.cache/devai/*.log): bench_concurrency sweep / multiturn JSON and the
DEPTH lines. Read-only. Stdlib only. Deterministic.

    python3 perf_client_logs.py ~/.cache/devai /var/cache/devai/logs/devai-vllm.log out/

Adds the ACTUAL prompt-token counts of the "20K" shared/disjoint prefixes
(the harness sizes prompts at 3.5 chars/token; the engine counts them).
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_enginelog import iter_records, utc  # noqa: E402
from statlib import rnd  # noqa: E402

LOGS = {
    "AR_prepared_vllm (autoround-tests.log)": "autoround-tests.log",
    "NV_prepared_vllm (nvfp4-prepared-tests.log)": "nvfp4-prepared-tests.log",
    "NV_asdelivered_vllm_32K_MTP_off (agent-shape-vllm.log)": "agent-shape-vllm.log",
    "Ollama_qwen3.8_mtp_q4 (agent-shape-ollama.log)": "agent-shape-ollama.log",
    "AR_prepared_decode_conc_sweep_1000tok (autoround-decode-conc.log)": "autoround-decode-conc.log",
}


def json_docs(text):
    docs, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    docs.append(json.loads(text[start:i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
    return docs


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)  # usage
    d, vllm_log, outdir = sys.argv[1:4]
    os.makedirs(outdir, exist_ok=True)
    out = {}
    for label, fn in LOGS.items():
        text = open(os.path.join(d, fn), errors="replace").read()
        docs = json_docs(text)
        sweeps = [x for x in docs if "cells" in x]
        multi = [x for x in docs if "turns" in x]
        depth = [json.loads(m) for m in re.findall(r"^DEPTH (\{.*\})$", text, re.M)]
        out[label] = {
            "sweeps": [{"prefix_tokens_target": s["prefix_tokens"], "max_tokens": s["max_tokens"],
                        "cells": [{k: c[k] for k in ("concurrency", "prefix", "n_ok", "wall_s", "aggregate_tps",
                                                     "ttft_p50_ms", "ttft_p95_ms", "completion_tokens")}
                                  for c in s["cells"]],
                        "prefix_gain_by_level": s["prefix_gain_by_level"],
                        "batch_scaling_by_level": s["batch_scaling_by_level"]} for s in sweeps],
            "multiturn": [{"ttft_slope_ms_per_1k_tokens": m["ttft_slope_ms_per_1k_tokens"],
                           "turns": [(t["history_tokens"], t["ttft_ms"]) for t in m["turns"]]} for m in multi],
            "depth": depth,
        }
    # actual token count of the "20000-token" prefix on the Qwen3.8 tokenizer:
    recs = [r for r in iter_records(vllm_log, utc("2026-09-21T12:25:35"), utc("2026-09-21T12:27:00"))
            if r["type"] == "req" and r["prompt_tokens"] > 1000]
    out["actual_prompt_tokens_for_20000_target_qwen38"] = sorted({r["prompt_tokens"] for r in recs})
    recs = [r for r in iter_records(vllm_log, utc("2026-09-21T12:27:25"), utc("2026-09-21T12:27:45"))
            if r["type"] == "req" and r["prompt_tokens"] > 100]
    out["actual_prompt_tokens_for_1000_target_qwen38"] = sorted({r["prompt_tokens"] for r in recs})
    # added 2026-09-27: bench_concurrency builds the prefix as exactly
    # int(target x 3.5) characters (before a ~20-character turn suffix)
    t20 = out["actual_prompt_tokens_for_20000_target_qwen38"]
    out["filler_chars_per_token_20000_target_qwen38"] = int(20000 * 3.5) / (sum(t20) / len(t20))
    with open(os.path.join(outdir, "client_logs.json"), "w") as fh:
        json.dump(rnd(out, 3), fh, indent=1)
    for label, v in out.items():
        if not isinstance(v, dict):
            print(label, v)
            continue
        print("==", label)
        for s in v["sweeps"]:
            print("  sweep target prefix", s["prefix_tokens_target"], "max_tokens", s["max_tokens"])
            for c in s["cells"]:
                print("   ", c)
        for m in v["multiturn"]:
            print("  multiturn slope", m["ttft_slope_ms_per_1k_tokens"], m["turns"])
        for x in v["depth"]:
            print("  depth", x)


if __name__ == "__main__":
    main()
