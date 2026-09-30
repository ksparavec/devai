"""Cold-start (launch -> ready) durations from the persisted router log, plus
per-launch engine phase timings from the persisted vllm-devai engine log.

Read-only. Stdlib only. Deterministic.

    python3 perf_coldstart.py /var/cache/devai/logs/devai-router.log \
        /var/cache/devai/logs/devai-vllm-devai.log  out/

What is measured
----------------
Router: ``starting <backend> with model <M> (ctx=<N>[, mtp=<x>])...`` ->
next ``<backend> ready`` line for the same backend. The interval is the
router-side launch window: stop placeholder, create + start container,
engine start-up, /health polling (2 s poll interval, gpu-arbiter/main.go waitForHealthy, so the
ready time is quantised to the poll). Timestamps have 1-s resolution.
A launch is DROPPED when an error line for the backend, another
``starting`` for the backend, or a router restart (``probe cache:`` load
line) intervenes, or when no ready line follows within 900 s.

For Ollama ``ready`` means the Ollama server answers; the model is loaded
lazily on the first request, so Ollama intervals are NOT model cold starts.

Engine (vllm-devai only): each ``non-default args`` line starts a launch;
the phase lines that follow are attributed to it.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_enginelog import iter_records, parse_ts  # noqa: E402
from statlib import describe, rnd  # noqa: E402

START_RE = re.compile(r"starting (\S+) with model (\S+) \(ctx=(\d+)(?:, mtp=([^)]*))?\)")
READY_RE = re.compile(r"\d\d:\d\d:\d\d (\S+) ready$")
ERR_RE = re.compile(r"error[^:]*: (\S+) ")
RESTART_RE = re.compile(r"probe cache: .* loaded")


def router_launches_detailed(path):
    """Paired launches, drop counts, and one record per dropped launch.

    The persisted log repeats whole blocks verbatim (the logger sidecar
    replays a container's history when it restarts), so every line is
    counted once: a line identical to an earlier one, timestamp included,
    is skipped."""
    pending = {}
    out = []
    dropped = collections.Counter()
    details = []
    seen = set()

    def drop(p, t, why, text=""):
        details.append({"backend": p["backend"], "model": p["model"], "ctx": p["ctx"],
                        "t0_utc": p["t0"].isoformat(), "reason": why,
                        "seconds_until_drop": (t - p["t0"]).total_seconds(), "line": text[:160]})

    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line in seen:
                dropped["duplicate_log_lines_skipped"] += 1
                continue
            seen.add(line)
            t = parse_ts(line)
            if t is None:
                continue
            if RESTART_RE.search(line):
                for b in list(pending):
                    dropped["router_restart"] += 1
                    drop(pending.pop(b), t, "router_restart")
                continue
            m = START_RE.search(line)
            if m:
                b = m.group(1)
                if b in pending:
                    dropped["superseded_by_new_start"] += 1
                    drop(pending[b], t, "superseded_by_new_start")
                pending[b] = {"backend": b, "model": m.group(2), "ctx": int(m.group(3)),
                              "mtp": m.group(4) or "n/a", "t0": t}
                continue
            m = READY_RE.search(line)
            if m:
                b = m.group(1)
                if b in pending:
                    p = pending.pop(b)
                    d = (t - p["t0"]).total_seconds()
                    if d <= 900:
                        p["t_ready"] = t
                        p["seconds"] = d
                        out.append(p)
                    else:
                        dropped["over_900s"] += 1
                        drop(p, t, "over_900s")
                continue
            m = ERR_RE.search(line)
            if m and m.group(1) in pending:
                p = pending.pop(m.group(1))
                dropped["error"] += 1
                drop(p, t, "never_ready" if "did not become ready" in line else "error", line)
    return out, dict(dropped), details


def router_launches(path):
    out, dropped, _ = router_launches_detailed(path)
    return out, dropped


def engine_launches(path):
    launches = []
    cur = None
    for r in iter_records(path):
        if r["type"] == "args":
            a = r["args"]
            if launches and launches[-1]["t"] == r["t"] and launches[-1]["raw"] == a:
                # the logger captured this launch twice; route the duplicate
                # phase lines to a throw-away record
                cur = {"t": r["t"], "phases": collections.defaultdict(list), "raw": a}
                continue
            model = re.search(r"'model': '([^']+)'", a)
            mml = re.search(r"'max_model_len': (\d+)", a)
            spec = "mtp" if "speculative_config" in a else "off"
            cur = {"t": r["t"], "model": os.path.basename(model.group(1)) if model else "?",
                   "max_model_len": int(mml.group(1)) if mml else None, "spec": spec,
                   "phases": collections.defaultdict(list), "raw": a}
            launches.append(cur)
        elif r["type"] == "phase" and cur is not None:
            cur["phases"][r["name"]].append(r["vals"])
            if r["name"] == "init_engine":
                cur["t_init_done"] = r["t"]
    rows = []
    for L in launches:
        ph = L["phases"]
        if "init_engine" not in ph:
            continue
        init = ph["init_engine"][0]
        row = {
            "t_utc": L["t"].isoformat(), "model": L["model"], "max_model_len": L["max_model_len"],
            "spec": L["spec"],
            "args_to_init_done_s": (L["t_init_done"] - L["t"]).total_seconds(),
            "weights_s": sum(float(v[0]) for v in ph.get("weights_s", [])),
            "model_load_s": float(ph["model_load"][0][1]) if ph.get("model_load") else None,
            "model_load_gib": float(ph["model_load"][0][0]) if ph.get("model_load") else None,
            "compile_ranges_s": [float(v[0]) for v in ph.get("compile_range_s", [])],
            "kv_tokens": int(ph["kv_tokens"][0][0].replace(",", "")) if ph.get("kv_tokens") else None,
            "graph_capture_s": float(ph["graph_capture_s"][0][0]) if ph.get("graph_capture_s") else None,
            "init_engine_s": float(init[0]),
            "init_compilation_s": float(init[1]) if len(init) > 1 else None,
        }
        rows.append(row)
    return rows


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)  # usage
    router_log, engine_log, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    launches, dropped, drop_details = router_launches_detailed(router_log)

    groups = collections.defaultdict(list)
    for L in launches:
        groups[(L["backend"], L["model"], L["mtp"])].append(L)
    by_group = []
    for (b, m, mtp), Ls in sorted(groups.items()):
        secs = [x["seconds"] for x in Ls]
        d = describe(secs) if len(secs) > 1 else {"n": 1, "median": secs[0], "min": secs[0], "max": secs[0]}
        d.update({"backend": b, "model": m, "mtp": mtp,
                  "first": Ls[0]["t0"].date().isoformat(), "last": Ls[-1]["t0"].date().isoformat(),
                  "ctxs": sorted({x["ctx"] for x in Ls})})
        by_group.append(rnd(d, 1))

    by_backend = {}
    for b in sorted({x["backend"] for x in launches}):
        secs = [x["seconds"] for x in launches if x["backend"] == b]
        by_backend[b] = rnd(describe(secs), 1)

    # vllm-devai launches in time order (the claims in docs concern them)
    devai = [{"t0_utc": x["t0"].isoformat(), "model": x["model"], "ctx": x["ctx"], "mtp": x["mtp"],
              "seconds": x["seconds"]} for x in launches if x["backend"] == "vllm-devai"]

    eng = engine_launches(engine_log)

    # Specific claims
    def find(backend, t_iso_prefix):
        return [x for x in devai if x["t0_utc"].startswith(t_iso_prefix)]

    claims = {
        "275s_empty_volumes (2026-09-22 16:45 UTC)": find("vllm-devai", "2026-09-22T16:45"),
        "51s_warm_caches (2026-09-22 16:50 UTC)": find("vllm-devai", "2026-09-22T16:50"),
        "all vllm-devai launches 2026-09-23": [x for x in devai if x["t0_utc"].startswith("2026-09-23")],
    }
    # laya plan: '119-124 s typical' -- launches after torch.compile cache
    # stopped being persisted (2026-09-23) up to the end of the log.
    after = [x for x in devai if x["t0_utc"] >= "2026-09-23T00:00"]
    claims["vllm-devai launches from 2026-09-23T00:00, FlashInfer cache warm, torch.compile cache NOT persisted (for the '119-124 s typical' claim)"] = {
        "launches": after,
        "summary_by_model_mtp": {
            f"{m}|{mtp}": rnd(describe([x["seconds"] for x in after if x["model"] == m and x["mtp"] == mtp]), 1)
            for (m, mtp) in sorted({(x["model"], x["mtp"]) for x in after})
            if len([x for x in after if x["model"] == m and x["mtp"] == mtp]) > 1
        },
    }

    def eng_sum(sel):
        rows = [e for e in eng if sel(e)]
        if len(rows) < 2:
            return rows
        return {"n": len(rows),
                "init_engine_s": rnd(describe([e["init_engine_s"] for e in rows]), 2),
                "args_to_init_done_s": rnd(describe([e["args_to_init_done_s"] for e in rows]), 1),
                "weights_s": rnd(describe([e["weights_s"] for e in rows]), 2),
                "model_load_s": rnd(describe([e["model_load_s"] for e in rows]), 2)}
    claims["engine phases, NVFP4-devai MTP on, from 2026-09-23T00:00"] = eng_sum(
        lambda e: e["t_utc"] >= "2026-09-23T00:00" and e["model"] == "Qwen3.8-27B-MTP-devai-NVFP4" and e["spec"] == "mtp")
    claims["engine phases, NVFP4-devai MTP on, 2026-09-22 before 16:50 (no cache volumes)"] = eng_sum(
        lambda e: e["t_utc"] < "2026-09-22T16:50" and e["model"] == "Qwen3.8-27B-MTP-devai-NVFP4" and e["spec"] == "mtp")
    claims["engine phases, all vllm-devai launches (weight-load phase)"] = eng_sum(lambda e: True)
    pre = [x for x in devai if x["t0_utc"] < "2026-09-22T16:50"]
    claims["router launch->ready, vllm-devai before 2026-09-22T16:50 (no cache volumes)"] = {
        f"{m}|{mtp}": rnd(describe([x["seconds"] for x in pre if x["model"] == m and x["mtp"] == mtp]), 1)
        for (m, mtp) in sorted({(x["model"], x["mtp"]) for x in pre})}

    # added 2026-09-27: every dropped launch, and the censoring summary
    never = [d for d in drop_details if d["reason"] == "never_ready"]
    claims["dropped_launches"] = {
        "details": drop_details,
        "never_ready": {"n": len(never),
                        "by_limit": dict(collections.Counter(
                            re.search(r"within (\S+)", d["line"]).group(1) for d in never)),
                        "models": dict(collections.Counter(d["model"] for d in never))},
        "router_restart_seconds_until_drop": sorted(
            d["seconds_until_drop"] for d in drop_details if d["reason"] == "router_restart"),
        "note": "launches with no ready line are right-censored: their duration exceeds the time shown",
    }
    # launch starts in the log, with and without the replayed copies
    raw_starts, distinct_starts = 0, set()
    with open(router_log, errors="replace") as fh:
        for line in fh:
            if START_RE.search(line):
                raw_starts += 1
                distinct_starts.add(line.rstrip("\n"))
    claims["launch_starts"] = {
        "raw_start_lines": raw_starts, "distinct_start_lines": len(distinct_starts),
        "distinct_accounted": len(launches) + sum(v for k, v in dropped.items()
                                                  if k != "duplicate_log_lines_skipped"),
        "note": "distinct starts = paired launches + launches set aside (superseded, error incl. never "
                "ready, router restart, over 900 s)"}
    out = {"router_launch_to_ready": {"n_paired": len(launches), "dropped": dropped,
                                      "by_backend": by_backend, "by_backend_model_mtp": by_group},
           "vllm_devai_launches": devai,
           "vllm_devai_engine_phases": eng,
           "claims": claims}
    with open(os.path.join(outdir, "coldstart.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)

    lines = ["# Cold start (router launch -> ready), persisted router log", "",
             f"Paired launches: {len(launches)}; dropped: {dropped}", "",
             "## By backend (all models pooled -- descriptive only)", "",
             "| backend | n | min | q25 | median | median 95% CI (order stat.) | q75 | q90 | max |",
             "|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    for b, d in by_backend.items():
        lines.append(f"| {b} | {d['n']} | {d['min']} | {d['q25']} | {d['median']} | {d.get('median_ci95')} | {d['q75']} | {d['q90']} | {d['max']} |")
    lines += ["", "## By backend x model x MTP (n >= 3 shown)", "",
              "| backend | model | mtp | n | dates | median | median 95% CI | min | max |",
              "|---|---|---|---:|---|---:|---|---:|---:|"]
    for d in by_group:
        if d["n"] >= 3:
            lines.append(f"| {d['backend']} | {d['model']} | {d['mtp']} | {d['n']} | {d['first']}..{d['last']} | {d['median']} | {d.get('median_ci95')} | {d['min']} | {d['max']} |")
    lines += ["", "## vllm-devai engine phases per launch (engine log)", "",
              "| t (UTC) | model | max_model_len | spec | args->init done s | weights s | model load s | compile ranges s | graph capture s | init engine s (compilation s) | KV tokens |",
              "|---|---|---:|---|---:|---:|---:|---|---:|---|---:|"]
    for e in eng:
        lines.append(f"| {e['t_utc'][:19]} | {e['model']} | {e['max_model_len']} | {e['spec']} | {e['args_to_init_done_s']:.0f} | {e['weights_s']:.2f} | {e['model_load_s']} | {e['compile_ranges_s']} | {e['graph_capture_s']} | {e['init_engine_s']} ({e['init_compilation_s']}) | {e['kv_tokens']} |")
    lines += ["", "## vllm-devai router launches", "", "| t0 (UTC) | model | ctx | mtp | launch->ready s |", "|---|---|---:|---|---:|"]
    for x in devai:
        lines.append(f"| {x['t0_utc'][:19]} | {x['model']} | {x['ctx']} | {x['mtp']} | {x['seconds']:.0f} |")
    with open(os.path.join(outdir, "coldstart.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(json.dumps({"n": len(launches), "dropped": dropped, "by_backend": by_backend}, indent=1))


if __name__ == "__main__":
    main()
