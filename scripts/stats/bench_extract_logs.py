#!/usr/bin/env python3
"""Extract per-item outcomes from inspect_ai .eval logs (zip + JSON).

Usage: python3 bench_extract_logs.py <inspect-log-dir> <out.json>

Stdlib only. Read-only on the input directory. For every log records:
header facts (created, task, task_args, model, generate config actually
sent), dataset sample ids and shuffle flag, the headline accuracy
inspect_ai computed, and the per-sample outcomes (id, value, error),
plus a scan of every model event's HTTP request for sampling keys
(temperature/top_p/seed) to verify what decoding the backend was asked for.
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

SAMPLING_KEYS = ("temperature", "top_p", "top_k", "seed", "min_p")


def to_num(v):
    if isinstance(v, (int, float)):
        return float(v)
    if v == "C":
        return 1.0
    if v == "I":
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def scan_requests(z, names):
    """Count model events and which sampling keys their requests carried."""
    n_events = 0
    keys_seen = {}
    configs_seen = {}
    for nm in names:
        if not nm.startswith("samples/"):
            continue
        s = json.loads(z.read(nm))
        for e in s.get("events") or []:
            if e.get("event") != "model":
                continue
            n_events += 1
            cfg = e.get("config") or {}
            ck = json.dumps(cfg, sort_keys=True)
            configs_seen[ck] = configs_seen.get(ck, 0) + 1
            req = ((e.get("call") or {}).get("request")) or {}
            tc = req.get("tool_choice")
            if tc is not None or req.get("tools"):
                tck = "tool_choice=" + json.dumps(tc, sort_keys=True)[:60]
                keys_seen[tck] = keys_seen.get(tck, 0) + 1
            for k in SAMPLING_KEYS:
                if k in req:
                    val = json.dumps(req[k])
                    keys_seen[f"{k}={val}"] = keys_seen.get(f"{k}={val}", 0) + 1
    return n_events, keys_seen, configs_seen


def sample_meta(z, names):
    """Per-sample metadata we need (tools subcase, message count, errors)."""
    out = {}
    for nm in names:
        if not nm.startswith("samples/"):
            continue
        s = json.loads(z.read(nm))
        out[str(s.get("id"))] = {
            "error": (s.get("error") or {}).get("message") if isinstance(s.get("error"), dict) else s.get("error"),
            "limit": s.get("limit"),
            "subcase": (s.get("metadata") or {}).get("subcase"),
            "task_id": (s.get("metadata") or {}).get("task_id"),
            "question_id": (s.get("metadata") or {}).get("question_id"),
            # Per-question tags the use-case scores group by
            # (usecase_scores.py): MMLU-Pro category, GPQA subdomain.
            "category": (s.get("metadata") or {}).get("category"),
            "subdomain": (s.get("metadata") or {}).get("subdomain"),
            "input_head": (s.get("input") if isinstance(s.get("input"), str) else "")[:80],
            # Content key: sample ids are positional (1..n) and do NOT
            # identify an item across runs when the dataset was shuffled.
            "item_key": hashlib.sha1(json.dumps(s.get("input"), sort_keys=True).encode()).hexdigest()[:16],
            "target": s.get("target") if isinstance(s.get("target"), str) else json.dumps(s.get("target")),
        }
    return out


def extract(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    rec = {"file": path.name}
    if "header.json" not in names:
        rec["status"] = "no_header"
        start = json.loads(z.read("_journal/start.json")) if "_journal/start.json" in names else {}
        ev = start.get("eval") or {}
        rec.update({"task": ev.get("task"), "model": ev.get("model"), "created": ev.get("created")})
        return rec
    h = json.loads(z.read("header.json"))
    ev = h.get("eval") or {}
    ds = ev.get("dataset") or {}
    res = h.get("results") or {}
    scores = res.get("scores") or []
    acc = None
    scored = unscored = None
    if scores:
        m = scores[0].get("metrics") or {}
        acc = (m.get("accuracy") or {}).get("value")
        scored = scores[0].get("scored_samples")
        unscored = scores[0].get("unscored_samples")
    rec.update({
        "status": h.get("status"),
        "created": ev.get("created"),
        "completed_at": (h.get("stats") or {}).get("completed_at"),
        "task": ev.get("task"),
        "task_args": ev.get("task_args"),
        "model": ev.get("model"),
        "model_base_url": ev.get("model_base_url"),
        "model_generate_config": ev.get("model_generate_config"),
        "plan_config": (h.get("plan") or {}).get("config"),
        "eval_config_keys": sorted((ev.get("config") or {}).keys()),
        "fail_on_error": (ev.get("config") or {}).get("fail_on_error"),
        "inspect_version": (ev.get("packages") or {}).get("inspect_ai"),
        "dataset_name": ds.get("name"),
        "dataset_samples": ds.get("samples"),
        "sample_ids": ds.get("sample_ids"),
        "shuffled": ds.get("shuffled"),
        "total_samples": res.get("total_samples"),
        "completed_samples": res.get("completed_samples"),
        "accuracy": acc,
        "scored_samples": scored,
        "unscored_samples": unscored,
        "error": (h.get("error") or {}).get("message") if isinstance(h.get("error"), dict) else None,
    })
    summ = json.loads(z.read("summaries.json")) if "summaries.json" in names else []
    items = []
    for s in summ:
        sc = s.get("scores") or {}
        val = None
        raw = None
        for v in sc.values():
            if v is not None:
                raw = v.get("value")
                val = to_num(raw)
                break
        items.append({"id": str(s.get("id")), "value": val, "raw": raw,
                      "error": s.get("error"), "limit": s.get("limit")})
    rec["items"] = items
    meta = sample_meta(z, names)
    for it in items:
        m = meta.get(it["id"]) or {}
        it.update({k: m.get(k) for k in ("subcase", "task_id", "question_id", "category", "subdomain", "input_head", "item_key", "target")})
        if m.get("error") and not it.get("error"):
            it["error"] = m.get("error")
    n_ev, keys_seen, cfgs = scan_requests(z, names)
    rec["n_model_events"] = n_ev
    rec["request_sampling_keys"] = keys_seen
    rec["event_configs"] = cfgs
    return rec


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)  # usage
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    recs = []
    for p in sorted(src.glob("*.eval")):
        try:
            recs.append(extract(p))
        except Exception as e:  # report, never silently drop
            recs.append({"file": p.name, "status": f"extract_error: {e!r}"})
    out.write_text(json.dumps(recs, indent=1))
    print(f"wrote {len(recs)} records to {out}")


if __name__ == "__main__":
    main()
