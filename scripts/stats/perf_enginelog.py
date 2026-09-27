"""Read-only parsers for the persisted vLLM / router logs (stdlib only).

Persisted log lines start with the logger's host timestamp, local time with
offset and 1-second resolution, e.g. ``2026-09-26T11:39:06+02:00``. All times
returned here are converted to UTC ``datetime`` objects.

Record types
------------
req   one per ``Request finished`` line (HyperQwen engine-completion-log patch,
      present only in the home-built vLLM 0.28 images): engine request id,
      finish reason, prompt_tokens, generated_tokens (engine-counted),
      elapsed_s (engine wall time since arrival, 10 ms resolution).
win   one per vLLM ``loggers.py`` 10-s window: avg prompt / generation
      throughput (engine-counted), running and waiting request counts.
spec  one per ``SpecDecoding metrics`` 10-s window: mean acceptance length,
      accepted and drafted token counts.
args  one per ``non-default args`` line (marks an engine launch).
phase engine start-up phase timings (weights, model loading, compile, KV
      pool, graph capture, init engine).
"""
from __future__ import annotations

import datetime as dt
import re

TS_RE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)([+-]\d\d:\d\d)")
REQ_RE = re.compile(
    r"Request finished: req (\S+) finish_reason=(\S+) prompt_tokens=(\d+) "
    r"generated_tokens=(\d+) elapsed_s=([\d.]+)")
WIN_RE = re.compile(
    r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: "
    r"([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs")
SPEC_RE = re.compile(
    r"Mean acceptance length: ([\d.]+), .*Accepted: (\d+) tokens, "
    r"Drafted: (\d+) tokens")
ARGS_RE = re.compile(r"non-default args: (\{.*\})")
PREFIX_RE = re.compile(r"Prefix cache hit rate: ([\d.]+)%")
POS_RE = re.compile(r"Per-position acceptance rate: ([\d., ]+?), Avg")
PHASE_RES = {
    "weights_s": re.compile(r"Loading weights took ([\d.]+) seconds"),
    "model_load": re.compile(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds"),
    "compile_range_s": re.compile(r"Compiling a graph for compile range .* takes ([\d.]+) s"),
    "kv_avail_gib": re.compile(r"Available KV cache memory: ([\d.]+) GiB"),
    "kv_tokens": re.compile(r"GPU KV cache size: ([\d,]+) tokens"),
    "graph_capture_s": re.compile(r"Graph capturing finished in ([\d.]+) secs"),
    "init_engine": re.compile(
        r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s"
        r"(?: \(compilation: ([\d.]+) s\))?"),
}


def parse_ts(line: str):
    m = TS_RE.match(line)
    if not m:
        return None
    base, off = m.group(1), m.group(2)
    t = dt.datetime.fromisoformat(base + off)
    return t.astimezone(dt.timezone.utc)


def utc(s: str) -> dt.datetime:
    """'2026-09-26T09:39:03' (UTC, no offset) -> aware datetime."""
    return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)


def iter_records(path: str, start: dt.datetime | None = None,
                 end: dt.datetime | None = None):
    seen = set()
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if ("Request finished" not in line and "throughput" not in line
                    and "SpecDecoding" not in line and "non-default args" not in line
                    and "took" not in line and "takes" not in line
                    and "KV cache" not in line and "Graph capturing" not in line):
                continue
            # The persisted logs repeat whole blocks verbatim (the logger
            # sidecar replays a container's history when it restarts): count
            # every line once, timestamp included.
            if line in seen:
                continue
            seen.add(line)
            t = parse_ts(line)
            if t is None:
                continue
            if start and t < start:
                continue
            if end and t > end:
                break
            m = REQ_RE.search(line)
            if m:
                yield {"type": "req", "t": t, "id": m.group(1), "finish": m.group(2),
                       "prompt_tokens": int(m.group(3)),
                       "generated_tokens": int(m.group(4)),
                       "elapsed_s": float(m.group(5))}
                continue
            m = WIN_RE.search(line)
            if m:
                ph = PREFIX_RE.search(line)
                yield {"type": "win", "t": t, "prompt_tps": float(m.group(1)),
                       "gen_tps": float(m.group(2)), "running": int(m.group(3)),
                       "waiting": int(m.group(4)),
                       "prefix_hit_rate_pct": float(ph.group(1)) if ph else None}
                continue
            m = SPEC_RE.search(line)
            if m:
                pm = POS_RE.search(line)
                yield {"type": "spec", "t": t, "mean_accept_len": float(m.group(1)),
                       "accepted": int(m.group(2)), "drafted": int(m.group(3)),
                       "per_position": ([float(x) for x in pm.group(1).split(",")] if pm else None)}
                continue
            m = ARGS_RE.search(line)
            if m:
                yield {"type": "args", "t": t, "args": m.group(1)}
                continue
            for name, rx in PHASE_RES.items():
                m = rx.search(line)
                if m:
                    yield {"type": "phase", "t": t, "name": name,
                           "vals": [g for g in m.groups() if g is not None]}
                    break
