"""Engine start-up phases, engine memory facts and router joins, from the
persisted vLLM and router logs; plus the 2026-05-05 bench-run summary lines.

Read-only. Stdlib only. Deterministic.

    python3 perf_phases.py /var/cache/devai/logs/devai-router.log \
        /var/cache/devai/logs/devai-vllm.log \
        /var/cache/devai/logs/devai-vllm-devai.log out/ \
        [/var/cache/devai/bench/bench-vllm-run-20260505T140328Z.log]

Engine launches
---------------
Each ``non-default args`` line starts a launch (a line the logger captured
twice, same timestamp and args, is ignored). The lines that follow, up to the
next launch, are attributed to it: vLLM version, checkpoint size, weight and
model loading, torch.compile per range, the device memory accounting (vLLM
0.28: consumed memory and peak activation), the KV pool (GiB and tokens), the
effective ``--gpu-memory-utilization`` (vLLM 0.22 logs it on a separate line
when it equals the default and is missing from the args), CUDA-graph capture
(seconds and GiB) and the ``init engine`` total with its compilation share.

Router join
-----------
Router launches come from ``perf_coldstart.router_launches`` (``starting`` ->
``ready``). Each is joined to the engine launch of the same model whose args
line falls inside [starting, ready]. Phases:

- router start -> engine args line: container start + Python imports;
- engine args -> init done: engine process, config, model loading, init;
- init done -> router ready: last /health poll (2 s interval).

All timestamps are the logger's (1-s resolution); durations inside the
engine (weights, model load, init) are vLLM's own and finer.

Groups are the ones cited in docs/nvfp4-coldstart.md,
docs/llm-tokens-and-speed.md and docs/multi-token-prediction.md.
"""
from __future__ import annotations

import ast
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_coldstart import router_launches_detailed  # noqa: E402
from perf_enginelog import parse_ts  # noqa: E402
from statlib import describe, rnd  # noqa: E402

KEYWORDS = ("non-default args", "Initializing a V1 LLM engine", "Checkpoint size",
            "Loading weights took", "Model loading took", "Compiling a graph for compile range",
            "Available KV cache memory", "GPU KV cache size", "Graph capturing finished",
            "CUDA graph pool memory", "init engine", "peak activation",
            "--gpu-memory-utilization=", "Profiling CUDA graph memory")
RX = {
    "version": re.compile(r"Initializing a V1 LLM engine \((v[^)]+)\)"),
    "checkpoint_gib": re.compile(r"Checkpoint size: ([\d.]+) GiB"),
    "weights_s": re.compile(r"Loading weights took ([\d.]+) seconds"),
    "model_load": re.compile(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds"),
    "compile_range_s": re.compile(r"Compiling a graph for compile range .* takes ([\d.]+) s"),
    "kv_avail_gib": re.compile(r"Available KV cache memory: ([\d.]+) GiB"),
    "kv_tokens": re.compile(r"GPU KV cache size: ([\d,]+) tokens"),
    "graph": re.compile(r"Graph capturing finished in ([\d.]+) secs, took ([\d.]+) GiB"),
    "graph_pool": re.compile(r"CUDA graph pool memory: ([\d.]+) GiB \(actual\), ([\d.]+) GiB \(estimated\)"),
    "init": re.compile(r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s"
                       r"(?: \(compilation: ([\d.]+) s\))?"),
    "memory": re.compile(r"Free memory on device \(([\d.]+)/([\d.]+) GiB\) on startup.*?"
                         r"Actual usage is ([\d.]+) GiB for consumed memory \(weights \+ non-torch\), "
                         r"([\d.]+) GiB for peak activation"),
    "gpu_util_line": re.compile(r"current --gpu-memory-utilization=([\d.]+) is equivalent"),
    "args": re.compile(r"non-default args: (\{.*\})"),
    "capture_sizes": re.compile(r"Profiling CUDA graph memory: PIECEWISE=(\d+) \(largest=(\d+)\), "
                                r"FULL=(\d+) \(largest=(\d+)\)"),
}


def engine_launches(path, log_label):
    launches = []
    cur = None
    seen = set()
    with open(path, errors="replace") as fh:
        for line in fh:
            if not any(k in line for k in KEYWORDS):
                continue
            # the persisted logs repeat whole blocks verbatim (logger replay):
            # count every line once, timestamp included
            if line in seen:
                continue
            seen.add(line)
            t = parse_ts(line)
            if t is None:
                continue
            m = RX["args"].search(line)
            if m:
                raw = m.group(1)
                if launches and launches[-1]["t"] == t and launches[-1]["raw"] == raw:
                    cur = {"t": t, "raw": raw, "dup": True}  # duplicated capture; discard lines
                    continue
                try:
                    args = ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    args = {}
                cur = {"log": log_label, "t": t, "raw": raw,
                       "model": os.path.basename(str(args.get("model", "?"))),
                       "max_model_len": args.get("max_model_len"),
                       "spec": "mtp" if args.get("speculative_config") else "off",
                       "max_num_seqs": args.get("max_num_seqs"),
                       "enforce_eager": bool(args.get("enforce_eager", False)),
                       "gpu_memory_utilization_arg": args.get("gpu_memory_utilization"),
                       "weights_s": 0.0, "compile_ranges_s": []}
                launches.append(cur)
                continue
            if cur is None or cur.get("dup"):
                continue
            for key in ("version", "checkpoint_gib", "gpu_util_line"):
                m = RX[key].search(line)
                if m:
                    cur[key] = m.group(1)
                    if key == "version":
                        cur["t_engine_core"] = t
            m = RX["capture_sizes"].search(line)
            if m:
                cur["cudagraph_sizes"] = {"piecewise_n": int(m.group(1)), "piecewise_largest": int(m.group(2)),
                                          "full_n": int(m.group(3)), "full_largest": int(m.group(4))}
            m = RX["weights_s"].search(line)
            if m:
                cur["weights_s"] += float(m.group(1))
            m = RX["model_load"].search(line)
            if m:
                cur["model_load_gib"], cur["model_load_s"] = float(m.group(1)), float(m.group(2))
                cur["t_model_loaded"] = t
            m = RX["compile_range_s"].search(line)
            if m:
                cur["compile_ranges_s"].append(float(m.group(1)))
            m = RX["kv_avail_gib"].search(line)
            if m:
                cur["kv_avail_gib"] = float(m.group(1))
            m = RX["kv_tokens"].search(line)
            if m:
                cur["kv_tokens"] = int(m.group(1).replace(",", ""))
            m = RX["graph"].search(line)
            if m:
                cur["graph_capture_s"], cur["graph_gib"] = float(m.group(1)), float(m.group(2))
            m = RX["graph_pool"].search(line)
            if m:
                cur["graph_pool_actual_gib"], cur["graph_pool_estimated_gib"] = float(m.group(1)), float(m.group(2))
            m = RX["memory"].search(line)
            if m:
                cur["device_free_gib"], cur["device_total_gib"] = float(m.group(1)), float(m.group(2))
                cur["consumed_gib"], cur["peak_activation_gib"] = float(m.group(3)), float(m.group(4))
            m = RX["init"].search(line)
            if m:
                cur["init_engine_s"] = float(m.group(1))
                cur["init_compilation_s"] = float(m.group(2)) if m.group(2) else None
                cur["t_init_done"] = t
    out = []
    for L in launches:
        if L.get("dup"):
            continue
        L = dict(L)
        L.pop("raw")
        util = L.get("gpu_memory_utilization_arg")
        if util is None and L.get("gpu_util_line"):
            util = float(L["gpu_util_line"])
        L["gpu_memory_utilization"] = util
        L.pop("gpu_util_line", None)
        if "t_init_done" in L:
            L["args_to_init_done_s"] = (L["t_init_done"] - L["t"]).total_seconds()
        if "t_engine_core" in L:
            # API server start, config/tokenizer, engine-core process spawn
            L["args_to_engine_core_s"] = (L["t_engine_core"] - L["t"]).total_seconds()
            if "t_model_loaded" in L:
                # CUDA context init + weight loading + model setup (1-s log resolution)
                L["engine_core_to_model_loaded_s"] = (L["t_model_loaded"] - L["t_engine_core"]).total_seconds()
        if L.get("kv_avail_gib") and L.get("kv_tokens"):
            L["kv_bytes_per_token"] = L["kv_avail_gib"] * 2 ** 30 / L["kv_tokens"]
        out.append(L)
    return out


def join(router, engines):
    """Router launch -> engine launch of the same model inside [t0, t_ready]."""
    rows = []
    for r in router:
        cands = [e for e in engines if e["model"] == r["model"] and r["t0"] <= e["t"] <= r["t_ready"]]
        if not cands:
            continue
        e = cands[0]
        row = {"t0_utc": r["t0"].isoformat(), "backend": r["backend"], "model": r["model"],
               "ctx": r["ctx"], "mtp": r["mtp"], "launch_to_ready_s": r["seconds"],
               "router_to_args_s": (e["t"] - r["t0"]).total_seconds(), "engine": e}
        if "t_init_done" in e:
            row["init_done_to_ready_s"] = (r["t_ready"] - e["t_init_done"]).total_seconds()
        rows.append(row)
    return rows


FIELDS = [("launch_to_ready_s", None), ("router_to_args_s", None), ("init_done_to_ready_s", None),
          ("args_to_init_done_s", "engine"), ("weights_s", "engine"), ("model_load_s", "engine"),
          ("model_load_gib", "engine"), ("init_engine_s", "engine"), ("init_compilation_s", "engine"),
          ("graph_capture_s", "engine"), ("graph_gib", "engine"), ("kv_avail_gib", "engine"),
          ("kv_tokens", "engine"), ("peak_activation_gib", "engine"), ("gpu_memory_utilization", "engine"),
          ("args_to_engine_core_s", "engine"), ("engine_core_to_model_loaded_s", "engine"),
          ("kv_bytes_per_token", "engine")]


def summarise(rows):
    out = {"n": len(rows), "launches_utc": [r["t0_utc"][:19] for r in rows]}
    for f, where in FIELDS:
        vals = [(r["engine"] if where else r).get(f) for r in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        d = describe(vals) if len(vals) > 1 else {"n": 1, "min": vals[0], "median": vals[0], "max": vals[0]}
        out[f] = rnd(d, 3)
    compile_second = [e["engine"]["compile_ranges_s"][1] for e in rows if len(e["engine"]["compile_ranges_s"]) > 1]
    if compile_second:
        out["second_compile_range_s"] = [min(compile_second), max(compile_second)]
    out["versions"] = sorted({str(r["engine"].get("version")) for r in rows})
    out["max_num_seqs"] = sorted({str(r["engine"].get("max_num_seqs")) for r in rows})
    return out


BENCH_HDR = re.compile(r"^=== (\S+) \(backend=(\w+), ctx=(\d+)\) ===")
BENCH_LAT = re.compile(r"ttft_first=([\d.]+)ms\s+steady_p50=([\d.]+)ms\s+steady_p95=([\d.]+)ms\s+tps=([\d.]+)/s")


def bench_run_summaries(path):
    rows, cur = [], None
    for line in open(path, errors="replace"):
        m = BENCH_HDR.search(line)
        if m:
            cur = {"model": m.group(1), "backend": m.group(2), "ctx": int(m.group(3))}
            continue
        m = BENCH_LAT.search(line)
        if m and cur:
            rows.append({**cur, "ttft_first_ms": float(m.group(1)), "ttft_steady_p50_ms": float(m.group(2)),
                         "ttft_steady_p95_ms": float(m.group(3)), "tps_sustained_p50": float(m.group(4))})
    return rows


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)  # usage
    router_log, vllm_log, devai_log, outdir = sys.argv[1:5]
    bench_log = sys.argv[5] if len(sys.argv) > 5 else None
    os.makedirs(outdir, exist_ok=True)
    router, dropped, drop_details = router_launches_detailed(router_log)
    # distinct "did not become ready" lines (the log replays blocks, so count
    # each line once), and the launches they belong to
    never_ready = collections.Counter()
    never_lines = set()
    for line in open(router_log, errors="replace"):
        m = re.search(r"error.*?: (\S+) did not become ready within (\S+)", line)
        if m and line not in never_lines:
            never_lines.add(line)
            never_ready[f"{m.group(1)} within {m.group(2)}"] += 1
    never_launches = [d for d in drop_details if d["reason"] == "never_ready"]
    restart_drops = [d for d in drop_details if d["reason"] == "router_restart"]

    eng_vllm = engine_launches(vllm_log, "devai-vllm.log")
    eng_devai = engine_launches(devai_log, "devai-vllm-devai.log")
    j_vllm = join([r for r in router if r["backend"] == "vllm"], eng_vllm)
    j_devai = join([r for r in router if r["backend"] == "vllm-devai"], eng_devai)

    nv = "Qwen3.8-27B-MTP-devai-NVFP4"
    groups = {
        "qwen3_8b_nvfp4_2026_07": [r for r in j_vllm if r["model"] == "Qwen3-8B-NVFP4"
                                   and "2026-07-16" <= r["t0_utc"] < "2026-07-20"],
        "nvfp4_devai_mtp_on_no_cache_volumes_2026_09_22": [
            r for r in j_devai if r["model"] == nv and r["mtp"] != "off" and r["t0_utc"] < "2026-09-22T16:47"],
        "nvfp4_devai_mtp_off_2026_09_22": [
            r for r in j_devai if r["model"] == nv and r["mtp"] == "off" and r["t0_utc"] < "2026-09-23"],
        "autoround_devai_mtp_off_2026_09_22": [
            r for r in j_devai if r["model"] == "Qwen3.8-27B-W4A16-devai-AutoRound" and r["mtp"] == "off"],
        # torch.compile cache persisted: vLLM loaded the compiled graphs (compilation < 5 s)
        "nvfp4_devai_mtp_on_torch_compile_cached_2026_09_22": [
            r for r in j_devai if r["model"] == nv and r["mtp"] != "off" and r["t0_utc"] < "2026-09-23"
            and (r["engine"].get("init_compilation_s") or 99) < 5],
        "nvfp4_devai_mtp_on_flashinfer_warm_compile_cold_from_2026_09_23": [
            r for r in j_devai if r["model"] == nv and r["mtp"] != "off" and r["t0_utc"] >= "2026-09-23"],
    }
    summaries = {k: summarise(v) for k, v in groups.items()}

    # Engine-only facts cited in the docs
    q35 = [{"t_utc": e["t"].isoformat()[:19], "kv_tokens": e.get("kv_tokens"),
            "init_compilation_s": e.get("init_compilation_s")}
           for e in eng_vllm if e["model"] == "Qwen3.5-9B-NVFP4" and e["t"].isoformat() >= "2026-09-22T22"
           and e["t"].isoformat() < "2026-09-24"]
    nemo = [{"t_utc": e["t"].isoformat()[:19], "max_model_len": e["max_model_len"],
             "enforce_eager": e["enforce_eager"], "max_num_seqs": e["max_num_seqs"], "version": e.get("version")}
            for e in eng_vllm if e["model"] == "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"]
    q38_act = collections.defaultdict(list)
    for e in eng_devai:
        if e["model"] == nv and "peak_activation_gib" in e:
            comp = e.get("init_compilation_s")
            comp = 99 if comp is None else comp
            state = ("all_graphs_loaded_from_cache_compilation_lt_1s" if comp < 1 else
                     "partly_cached_compilation_1_to_5s" if comp < 5 else "compiled_at_launch")
            q38_act[state].append((e["t"].isoformat()[:19], e["peak_activation_gib"], e.get("kv_tokens"),
                                   e["spec"], comp))
    devai_utils = sorted({e.get("gpu_memory_utilization") for e in eng_devai
                          if e.get("gpu_memory_utilization") is not None})
    device_total = sorted({e.get("device_total_gib") for e in eng_devai if e.get("device_total_gib")})
    all_weights = [e["weights_s"] for e in eng_devai if e["weights_s"]]
    all_loads = [e["model_load_s"] for e in eng_devai if "model_load_s" in e]
    graph_gib_devai = sorted({e["graph_gib"] for e in eng_devai if "graph_gib" in e})

    out = {
        "router_paired_launches": len(router),
        "router_max_launch_to_ready_s": max(r["seconds"] for r in router),
        "router_never_ready": dict(never_ready),
        "router_never_ready_total": sum(never_ready.values()),
        "groups": summaries,
        "qwen3_8b_2026_07_engine": [{k: v for k, v in r["engine"].items() if k not in ("t", "t_init_done", "log")}
                                    for r in groups["qwen3_8b_nvfp4_2026_07"]],
        "qwen35_9b_kv_pool_2026_09_22_23": q35,
        "nemotron3_nano_launch_args": nemo,
        "qwen38_nvfp4_devai_peak_activation": dict(q38_act),
        "vllm_devai_gpu_memory_utilization_values": devai_utils,
        "vllm_devai_device_total_gib": device_total,
        "vllm_devai_all_launches_weights_s_range": [min(all_weights), max(all_weights)] if all_weights else None,
        "vllm_devai_all_launches_model_load_s_range": [min(all_loads), max(all_loads)] if all_loads else None,
        "vllm_devai_graph_pool_gib_values": graph_gib_devai,
        "engine_launches_parsed": {"devai-vllm.log": len(eng_vllm), "devai-vllm-devai.log": len(eng_devai)},
    }
    # ---- added 2026-09-27 ----
    real_never = [d for d in never_launches if not d["model"].startswith("claude-")]
    out["router_never_ready_launches"] = {
        "distinct_error_lines": len(never_lines),
        "launches": [{"model": d["model"], "t0_utc": d["t0_utc"][:19], "seconds": d["seconds_until_drop"],
                      "limit": (re.search(r"within (\S+)", d["line"]) or [None, None])[1]} for d in never_launches],
        "n_launches": len(never_launches),
        "n_model_launches_excluding_client_model_names": len(real_never),
        "model_launches_by_limit": dict(collections.Counter(
            re.search(r"within (\S+)", d["line"]).group(1) for d in real_never)),
        "note": "a launch 'of' a Claude model name is the router trying to start a name that is not a served model",
    }
    out["router_restart_abandoned_launches"] = {
        "n": len(restart_drops),
        "seconds_until_restart": sorted(d["seconds_until_drop"] for d in restart_drops),
        "n_longer_than_max_observed": sum(1 for d in restart_drops
                                          if d["seconds_until_drop"] > out["router_max_launch_to_ready_s"]),
    }
    mtp_on = [e for e in eng_devai if e["model"] == nv and e["spec"] == "mtp" and e["max_model_len"] == 118784
              and e.get("gpu_memory_utilization") == 0.96 and e["t"].isoformat() < "2026-09-22T16:47"]
    mtp_off = [e for e in eng_devai if e["model"] == nv and e["spec"] == "off" and e["max_model_len"] == 118784
               and e.get("gpu_memory_utilization") == 0.96 and e["t"].isoformat() < "2026-09-23"]

    def uniq(rows, k):
        return sorted({r.get(k) for r in rows if r.get(k) is not None})
    if mtp_on and mtp_off:
        on_w, off_w = uniq(mtp_on, "model_load_gib"), uniq(mtp_off, "model_load_gib")
        on_kv, off_kv = uniq(mtp_on, "kv_avail_gib"), uniq(mtp_off, "kv_avail_gib")
        on_t, off_t = uniq(mtp_on, "kv_tokens"), uniq(mtp_off, "kv_tokens")
        bpt_on = on_kv[0] * 2 ** 30 / on_t[0]
        bpt_off = off_kv[0] * 2 ** 30 / off_t[0]
        out["mtp_memory_cost_nvfp4_devai_2026_09_22"] = {
            "conditions": "Qwen3.8-27B-MTP-devai-NVFP4, vLLM 0.28.0, gpu util 0.96, max_model_len 118784, "
                          "launches before the cache volumes (2026-09-22)",
            "n_launches": {"mtp_on": sum(1 for e in mtp_on if e.get("kv_tokens")),
                           "mtp_off": sum(1 for e in mtp_off if e.get("kv_tokens"))},
            "n_launches_aborted_before_kv_sizing": {"mtp_on": sum(1 for e in mtp_on if not e.get("kv_tokens")),
                                                    "mtp_off": sum(1 for e in mtp_off if not e.get("kv_tokens"))},
            "model_load_gib": {"mtp_on": on_w, "mtp_off": off_w},
            "kv_avail_gib": {"mtp_on": on_kv, "mtp_off": off_kv},
            "kv_tokens": {"mtp_on": on_t, "mtp_off": off_t},
            "weights_delta_gib": on_w[0] - off_w[0],
            "kv_bytes_per_token": {"mtp_on": bpt_on, "mtp_off": bpt_off, "ratio_on_over_off": bpt_on / bpt_off},
            "kv_pool_tokens_ratio_on_over_off": on_t[0] / off_t[0],
            "note": "the MTP-on pool of 118,784 tokens equals max_model_len: the context was chosen as the "
                    "largest that fits with MTP (exact-context probe)",
        }
    out["cudagraph_capture_sizes"] = {
        "qwen3_8b_nvfp4_2026_07": [r["engine"].get("cudagraph_sizes") for r in groups["qwen3_8b_nvfp4_2026_07"]],
        "qwen38_devai": sorted({json.dumps(e.get("cudagraph_sizes"), sort_keys=True) for e in eng_devai
                                if e.get("cudagraph_sizes")}),
    }
    if bench_log:
        out["bench_run_summaries"] = {"source": bench_log, "rows": bench_run_summaries(bench_log)}
    with open(os.path.join(outdir, "phases.json"), "w") as fh:
        json.dump(rnd(out, 4), fh, indent=1, default=str)

    def rng(d, f, k=1):
        if f not in d:
            return "-"
        x = d[f]
        if x.get("n", 1) == 1:
            return f"{x['median']:.{k}f}"
        return f"{x['min']:.{k}f}-{x['max']:.{k}f} (median {x['median']:.{k}f})"
    lines = ["# Launch phases (router + vLLM log)", "",
             "| group | n | router->args s | model load s | weights s | init engine s | compilation s | graph capture s | launch->ready s |",
             "|---|---:|---|---|---|---|---|---|---|"]
    for k, s in summaries.items():
        lines.append(f"| {k} | {s['n']} | {rng(s, 'router_to_args_s', 0)} | {rng(s, 'model_load_s', 2)} | "
                     f"{rng(s, 'weights_s', 2)} | {rng(s, 'init_engine_s', 1)} | {rng(s, 'init_compilation_s', 2)} | "
                     f"{rng(s, 'graph_capture_s', 0)} | {rng(s, 'launch_to_ready_s', 0)} |")
    lines += ["", f"Router: {len(router)} paired launches, max {out['router_max_launch_to_ready_s']:.0f} s; "
                  f"never ready: {dict(never_ready)}", ""]
    nr = out["router_never_ready_launches"]
    lines += [f"Never-ready: {nr['distinct_error_lines']} distinct error lines, {nr['n_launches']} launches, "
              f"{nr['n_model_launches_excluding_client_model_names']} of real models "
              f"(by limit {nr['model_launches_by_limit']}). Restart-abandoned: "
              f"{out['router_restart_abandoned_launches']['n']} launches, seconds "
              f"{out['router_restart_abandoned_launches']['seconds_until_restart']}", ""]
    if "mtp_memory_cost_nvfp4_devai_2026_09_22" in out:
        mc = out["mtp_memory_cost_nvfp4_devai_2026_09_22"]
        lines += [f"MTP memory cost (NVFP4-devai, 2026-09-22): weights {mc['model_load_gib']}, "
                  f"KV GiB {mc['kv_avail_gib']}, KV tokens {mc['kv_tokens']}, weights delta "
                  f"{mc['weights_delta_gib']:.2f} GiB, KV bytes/token ratio "
                  f"{mc['kv_bytes_per_token']['ratio_on_over_off']:.3f}, pool ratio "
                  f"{mc['kv_pool_tokens_ratio_on_over_off']:.3f}", ""]
    lines += ["## Qwen3-8B-NVFP4, 2026-07 launches (engine facts)", "",
              "| launch | version | gpu util | max_num_seqs | checkpoint GiB | model load GiB / s | weights s | KV GiB / tokens | graph s / GiB | init s (compilation) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in groups["qwen3_8b_nvfp4_2026_07"]:
        e = r["engine"]
        lines.append(f"| {r['t0_utc'][:19]} | {e.get('version')} | {e.get('gpu_memory_utilization')} | {e.get('max_num_seqs')} | "
                     f"{e.get('checkpoint_gib')} | {e.get('model_load_gib')} / {e.get('model_load_s')} | {e['weights_s']:.2f} | "
                     f"{e.get('kv_avail_gib')} / {e.get('kv_tokens')} | {e.get('graph_capture_s')} / {e.get('graph_gib')} | "
                     f"{e.get('init_engine_s')} ({e.get('init_compilation_s')}) |")
    lines += ["", "## Qwen3.8 NVFP4-devai peak activation by compile state", ""]
    for k, v in q38_act.items():
        acts = sorted({x[1] for x in v})
        kvs = sorted({x[2] for x in v if x[2]})
        lines.append(f"- {k}: n={len(v)}, peak activation GiB {acts}, KV tokens {kvs}")
    lines += ["", "## Qwen3.5-9B-NVFP4 KV pool, 2026-09-22/23 (stock vLLM)", ""]
    for x in q35:
        lines.append(f"- {x['t_utc']}: KV {x['kv_tokens']} tokens, compilation {x['init_compilation_s']} s")
    lines += ["", "## Nemotron-3-Nano-30B-A3B launch args", ""]
    for x in nemo:
        lines.append(f"- {x['t_utc']}: max_model_len {x['max_model_len']}, enforce_eager {x['enforce_eager']}, "
                     f"max_num_seqs {x['max_num_seqs']}, {x['version']}")
    if bench_log:
        lines += ["", "## Bench run summary lines", "", "| model | ctx | TTFT first ms | steady p50 / p95 ms | TPS p50 |",
                  "|---|---:|---:|---|---:|"]
        for x in out["bench_run_summaries"]["rows"]:
            lines.append(f"| {x['model']} | {x['ctx']} | {x['ttft_first_ms']} | {x['ttft_steady_p50_ms']} / "
                         f"{x['ttft_steady_p95_ms']} | {x['tps_sustained_p50']} |")
    with open(os.path.join(outdir, "phases.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
