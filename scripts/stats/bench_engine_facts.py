#!/usr/bin/env python3
"""Read-only facts from the engine container logs (stdlib only).

Usage: python3 bench_engine_facts.py /var/cache/devai/logs <out.json>

1. Per model, the default sampling parameters the engine applied when a
   request carried none (vLLM: "Default vLLM sampling parameters have been
   overridden by the model's generation_config.json"; SGLang: "Using default
   chat sampling params from model generation config"). A model with no such
   line fell back to the engine's own default (temperature 1.0, top_p 1.0).
2. kv_cache_dtype / enforce_eager of every Nemotron-3-Nano vLLM launch.
3. Date range each log covers.
"""
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path


def first_date(p):
    with open(p, errors="replace") as f:
        for line in f:
            m = re.match(r"(\d{4}-\d{2}-\d{2})T", line)
            if m:
                return m.group(1)
    return None


def scan(path, model_re, samp_re, launch_marker):
    cur = None
    seen = OrderedDict()
    launches = OrderedDict()
    with open(path, errors="replace") as f:
        for line in f:
            if launch_marker in line:
                m = model_re.search(line)
                if m:
                    cur = m.group(1).split("/")[-1]
                    launches.setdefault(cur, []).append(line[:25])
            s = samp_re.search(line)
            if s and cur:
                seen.setdefault(cur, {}).setdefault(s.group(1), []).append(line[:10])
    return seen, launches


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)  # usage
    d = Path(sys.argv[1])
    out = {}
    vre = re.compile(r"'model': '(/models/[^']+)'")
    vs = re.compile(r"overridden by the model's `generation_config.json`: `(\{[^`]*\})`")
    for b in ("vllm", "vllm-devai"):
        p = d / f"devai-{b}.log"
        seen, launches = scan(p, vre, vs, "non-default args")
        out[b] = {"log_starts": first_date(p),
                  "launched_models": {k: len(v) for k, v in launches.items()},
                  "default_sampling_from_generation_config": {
                      k: {s: {"first": v[0], "last": v[-1], "n": len(v)} for s, v in sv.items()}
                      for k, sv in seen.items()},
                  "models_without_override_line (engine default T=1.0, top_p=1.0)":
                      sorted(set(launches) - set(seen))}
    p = d / "devai-sglang.log"
    seen, launches = scan(p, re.compile(r"model_path='(/models/[^']+)'"),
                          re.compile(r"Using default chat sampling params from model generation config: (\{.*?\})"),
                          "server_args=")
    out["sglang"] = {"log_starts": first_date(p),
                     "launched_models": {k: len(v) for k, v in launches.items()},
                     "default_sampling_from_generation_config": {
                         k: {s: {"first": v[0], "last": v[-1], "n": len(v)} for s, v in sv.items()}
                         for k, sv in seen.items()},
                     "models_without_override_line (engine default T=1.0, top_p=1.0)":
                         sorted(set(launches) - set(seen))}
    nem = []
    with open(d / "devai-vllm.log", errors="replace") as f:
        for line in f:
            if "non-default args" in line and "Nemotron-3-Nano" in line:
                kv = re.search(r"'kv_cache_dtype': '([^']+)'", line)
                ee = re.search(r"'enforce_eager': (\w+)", line)
                nem.append({"when": line[:25], "kv_cache_dtype": kv.group(1) if kv else "(not passed)",
                            "enforce_eager": ee.group(1) if ee else "(not passed)"})
    out["nemotron3_vllm_launches"] = nem
    r = []
    with open(d / "devai-router.log", errors="replace") as f:
        for line in f:
            if line.startswith("2026-05-05") and "Nemotron-3-Nano" in line and "starting vllm" in line:
                r.append(line.strip()[:160])
    out["router_log_starts"] = first_date(d / "devai-router.log")
    out["router_nemotron3_starts_2026_05_05"] = r
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
