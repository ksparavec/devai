"""Concurrency + prefix-reuse benchmark.

Every other number in this harness is SINGLE-STREAM: `tps_sustained_p50`
is what one user experiences, which is the right headline for a picker
column. But it makes one whole class of engine behaviour invisible.

Decode is bound by *weight* bandwidth -- each step reads all the weights
once -- so that single read produces one token for EVERY sequence in
flight. Aggregate throughput therefore scales with concurrency until
something else binds, and a single-stream benchmark cannot see any of it.
Worse, it cannot see prefix reuse at all: SGLang's RadixAttention and
vLLM's `--enable-prefix-caching` both exist to skip recomputing a shared
prompt prefix, and with one request at a time there is nothing to share.

That gap is not academic here. The lab's real workload is coding agents
(Claude Code, aider, opencode), whose requests carry a large STABLE
prefix -- system prompt, tool schemas, file context -- followed by a
short varying turn. Choosing between backends on single-turn numbers
alone measures the one axis where they are equivalent.

This task sweeps two axes, because the vendor claims confound them:

    concurrency:  1, 2, 4, 8, 16, 32   (batching)
    prefix:       shared | disjoint    (prefix cache)

`shared` sends N concurrent requests carrying an IDENTICAL long prefix;
`disjoint` gives each request its own. Batching lifts both arms; only
prefix caching lifts `shared` above `disjoint`. The gap between the two
arms is the prefix-cache effect, isolated.

Cell hygiene, so the cells cannot contaminate each other:

  - Each CELL gets a unique prefix, so a warm radix tree from an earlier
    cell cannot flatter a later one. Sharing is measured strictly WITHIN
    a batch.
  - A warmup request runs first, outside all measurement, so the ~60s
    cold start (see bench_latency_leak's ttft_ms_first) lands nowhere
    near a reported number.
  - HTTP 429 is counted SEPARATELY and never treated as a slow request.
    The router caps in-flight requests per backend at
    MAX_CONCURRENT_REQUESTS (default 32) and refuses beyond it; without
    this accounting a CAPPED run and a SLOW run produce the same
    throughput number, which is precisely the kind of silent
    misattribution this repo keeps rediscovering. Any cell that saw a
    429 is flagged `capped: true` -- its numbers describe the router's
    admission gate, not the engine.

Temperature is pinned to 0, matching the rest of the harness and
docs/sampling-strategies.md -- throughput must not vary with sampling
luck.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench._bench_core import (  # noqa: E402
    p50,
    p95,
    stream_chat_completion,
)

# Concurrency levels. Stops at 32 because that is the router's default
# MAX_CONCURRENT_REQUESTS and the engines' matching --max-num-seqs /
# --max-running-requests; past it we would measure our own gate.
DEFAULT_LEVELS = (1, 2, 4, 8, 16, 32)

# Size of the stable prefix, in approximate tokens. Chosen to resemble a
# real agent turn (system prompt + tool schemas + a file or two) rather
# than a synthetic best case: big enough that recomputing it is
# expensive, small enough that 32 concurrent copies still fit a 24 GiB
# card's KV pool at moderate context.
DEFAULT_PREFIX_TOKENS = 4096

# Short completions: this measures PREFILL reuse and batch decode
# throughput, not long-form generation. Long outputs would swamp the
# TTFT signal that prefix caching actually moves.
DEFAULT_MAX_TOKENS = 64

# ~3.5 chars/token for English prose, the same seed estimate the load
# probe uses. Exactness does not matter -- both arms use the identical
# construction, so any bias cancels in the shared-vs-disjoint comparison.
_CHARS_PER_TOKEN = 3.5

_FILLER = (
    "The deployment pipeline validates each artifact before promotion. "
    "Reviewers annotate findings inline and the bot collates them nightly. "
)


def _prefix(tokens: int, salt: str) -> str:
    """Deterministic filler of roughly `tokens` tokens, unique per salt.

    The salt leads so that two prefixes diverge at token ~0: a radix tree
    keys on the longest COMMON prefix, so a trailing salt would leave the
    entire body shared and silently turn the `disjoint` arm into a second
    `shared` arm -- measuring nothing.
    """
    head = f"# session {salt}\n"
    want = max(1, int(tokens * _CHARS_PER_TOKEN) - len(head))
    body = (_FILLER * (want // len(_FILLER) + 1))[:want]
    return head + body


def _one_request(router_url: str, model: str, prompt: str,
                 max_tokens: int, timeout_s: float) -> dict:
    """Single streamed completion. Classifies 429 apart from other errors."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    try:
        res = stream_chat_completion(router_url, body, timeout=timeout_s)
    except urllib.error.HTTPError as e:
        # 429 is the router's admission gate, not engine slowness. Keep
        # it distinguishable -- see the module docstring.
        return {"ok": False, "capped": e.code == 429, "error": f"HTTP {e.code}"}
    except Exception as e:                                     # noqa: BLE001
        return {"ok": False, "capped": False, "error": f"{type(e).__name__}: {e}"}

    t_first = res.get("t_first_token")
    return {
        "ok": True,
        "capped": False,
        "ttft_ms": ((t_first - res["t_open"]) * 1000) if t_first else None,
        # effective_tokens, NOT completion_tokens: the latter is populated
        # only from a `usage` block, which the engine omits unless the
        # request sets stream_options.include_usage -- so on a reasoning
        # model it reads 0 and aggregate throughput comes out as 0.00 for
        # every cell. _bench_core already computes effective_tokens as
        # max(usage, char-based estimate) for exactly this reason; the
        # leak task uses it and this one must too.
        "tokens": res.get("effective_tokens") or res.get("completion_tokens") or 0,
        "wall_s": res["t_done"] - res["t_open"],
    }


def _run_cell(router_url: str, model: str, concurrency: int, shared: bool,
              prefix_tokens: int, max_tokens: int, timeout_s: float,
              cell_id: str) -> dict:
    """One (concurrency, prefix-mode) cell.

    Aggregate tok/s is total completion tokens over the batch's WALL
    time, not the sum of per-request rates: the point of batching is that
    requests overlap, so summing per-request rates would double-count the
    overlap and inflate the number.
    """
    if shared:
        prompts = [_prefix(prefix_tokens, cell_id) + f"\n\nTurn {i}: reply OK."
                   for i in range(concurrency)]
    else:
        prompts = [_prefix(prefix_tokens, f"{cell_id}-{i}") + f"\n\nTurn {i}: reply OK."
                   for i in range(concurrency)]

    t0 = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_one_request, router_url, model, p, max_tokens, timeout_s)
            for p in prompts
        ]
        for f in as_completed(futures):
            results.append(f.result())
    wall = time.time() - t0

    ok = [r for r in results if r["ok"]]
    n_capped = sum(1 for r in results if r.get("capped"))
    n_error = sum(1 for r in results if not r["ok"] and not r.get("capped"))
    ttfts = [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None]
    tokens = sum(r["tokens"] for r in ok)

    return {
        "concurrency": concurrency,
        "prefix": "shared" if shared else "disjoint",
        "n_ok": len(ok),
        "n_429": n_capped,
        "n_error": n_error,
        # A cell that hit the admission gate is describing the ROUTER,
        # not the engine. Flagged so a reader cannot mistake it for a
        # throughput measurement.
        "capped": n_capped > 0,
        "wall_s": round(wall, 2),
        "aggregate_tps": round(tokens / wall, 2) if wall > 0 and tokens else 0.0,
        "ttft_p50_ms": round(p50(ttfts), 1) if ttfts else None,
        "ttft_p95_ms": round(p95(ttfts), 1) if ttfts else None,
        "completion_tokens": tokens,
        "first_error": next(
            (r["error"] for r in results if not r["ok"]), None),
    }


def run(
    model: str,
    router_url: str,
    *,
    levels: tuple[int, ...] = DEFAULT_LEVELS,
    prefix_tokens: int = DEFAULT_PREFIX_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_s: float = 900.0,
    modes: tuple[str, ...] = ("shared", "disjoint"),
) -> dict:
    """Sweep concurrency x prefix-mode. Returns cells plus a summary.

    `prefix_gain_*` is the headline: aggregate tok/s in the `shared` arm
    divided by the same level's `disjoint` arm. >1 means the engine
    genuinely reused the shared prefix. ~1 means prefix caching bought
    nothing at that level, whatever the vendor claims.
    """
    # Warm the model OUTSIDE measurement: the first request after a
    # container recreate pays a ~60s cold start that would otherwise be
    # attributed to whichever cell ran first.
    _one_request(router_url, model, "Reply with OK.", 8, timeout_s)

    cells: list[dict] = []
    for level in levels:
        for mode in modes:
            cell_id = f"c{level}-{mode}"
            cell = _run_cell(
                router_url, model, level, mode == "shared",
                prefix_tokens, max_tokens, timeout_s, cell_id)
            cells.append(cell)
            flag = "  [CAPPED: hit router 429]" if cell["capped"] else ""
            print(
                f"    c={level:<3} {mode:<8} "
                f"agg_tps={cell['aggregate_tps']:>8.2f} "
                f"ttft_p50={str(cell['ttft_p50_ms']):>8}ms "
                f"ok={cell['n_ok']}/{level}{flag}",
                file=sys.stderr)

    by = {(c["concurrency"], c["prefix"]): c for c in cells}
    gains = {}
    for level in levels:
        s, d = by.get((level, "shared")), by.get((level, "disjoint"))
        if s and d and d["aggregate_tps"] > 0 and not (s["capped"] or d["capped"]):
            gains[str(level)] = round(s["aggregate_tps"] / d["aggregate_tps"], 3)

    scaling = {}
    base = by.get((levels[0], "disjoint"))
    for level in levels:
        d = by.get((level, "disjoint"))
        if base and d and base["aggregate_tps"] > 0 and not d["capped"]:
            scaling[str(level)] = round(d["aggregate_tps"] / base["aggregate_tps"], 3)

    return {
        "cells": cells,
        "prefix_tokens": prefix_tokens,
        "max_tokens": max_tokens,
        "levels": list(levels),
        # Prefix-cache effect, per concurrency level.
        "prefix_gain_by_level": gains,
        # Batching effect: disjoint throughput relative to c=1. Isolates
        # scaling from prefix reuse.
        "batch_scaling_by_level": scaling,
        "any_capped": any(c["capped"] for c in cells),
    }


def _turn(router_url: str, model: str, messages: list[dict],
          max_tokens: int, timeout_s: float) -> dict:
    """One conversational turn carrying the whole history so far."""
    body = {"model": model, "messages": messages,
            "max_tokens": max_tokens, "temperature": 0}
    try:
        res = stream_chat_completion(router_url, body, timeout=timeout_s)
    except urllib.error.HTTPError as e:
        return {"ok": False, "capped": e.code == 429, "error": f"HTTP {e.code}"}
    except Exception as e:                                     # noqa: BLE001
        return {"ok": False, "capped": False, "error": f"{type(e).__name__}: {e}"}
    t_first = res.get("t_first_token")
    return {
        "ok": True, "capped": False,
        "ttft_ms": ((t_first - res["t_open"]) * 1000) if t_first else None,
        "content": res.get("content") or "",
        "tokens": res.get("effective_tokens") or res.get("completion_tokens") or 0,
    }


def _slope_ms_per_1k(points: list[tuple[float, float]]) -> float | None:
    """Least-squares slope of ttft_ms vs history tokens, scaled per 1K.

    This is the whole measurement. A prefix cache that works keeps
    time-to-first-token roughly CONSTANT as the conversation grows,
    because only the newly-appended tokens need prefilling -- slope ~0.
    Without reuse the engine re-prefills the entire history every turn and
    TTFT climbs linearly with it -- slope > 0. Comparing slopes therefore
    compares prefix-cache effectiveness directly, and unlike a raw TTFT it
    is insensitive to constant per-request overhead, which cancels.
    """
    n = len(points)
    if n < 2:
        return None
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    den = sum((x - mx) ** 2 for x, _ in points)
    if den == 0:
        return None
    num = sum((x - mx) * (y - my) for x, y in points)
    return round((num / den) * 1000.0, 3)


def run_multiturn(
    model: str,
    router_url: str,
    *,
    turns: int = 8,
    prefix_tokens: int = DEFAULT_PREFIX_TOKENS,
    growth_tokens: int = 4096,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_s: float = 900.0,
    salt: str = "mt",
) -> dict:
    """Sequential agent-style session: N turns over a GROWING history.

    This is the arm that matches how this lab is actually used. The
    concurrency sweep above measures prefix sharing ACROSS simultaneous
    requests -- the vendor's benchmark -- but the router serves one model
    to one user, so an agent session is sequential turns, and reuse there
    happens across TIME, not across concurrent sequences.

    Each turn appends a simulated tool result (a file dump), which is what
    actually inflates an agent conversation, then re-sends the entire
    history. Turn 1 pays full prefill; every later turn re-sends
    everything it already sent, so a working prefix cache should make
    turns 2..N cost only their new tokens.
    """
    # Warm the model OUTSIDE the measured turns. Without this, turn 1
    # carries the container cold start -- measured at 76,332 ms on
    # gpt-oss-20b, i.e. ~130x a warm turn -- which dominates the
    # least-squares fit and drove the slope NEGATIVE, reporting
    # "TTFT falls as history grows". A deliberately tiny prompt with a
    # different salt, so it cannot seed the prefix being measured.
    _one_request(router_url, model, f"# warmup {salt}\nReply with OK.",
                 8, timeout_s)

    system = _prefix(prefix_tokens, salt)
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Reply with OK."},
    ]
    records: list[dict] = []
    points: list[tuple[float, float]] = []
    for i in range(turns):
        hist_chars = sum(len(m["content"]) for m in messages)
        hist_tokens = hist_chars / _CHARS_PER_TOKEN
        r = _turn(router_url, model, messages, max_tokens, timeout_s)
        rec = {
            "turn": i + 1,
            "history_tokens": int(hist_tokens),
            "ttft_ms": round(r["ttft_ms"], 1) if r.get("ttft_ms") else None,
            "ok": r["ok"],
            "capped": r.get("capped", False),
            "error": r.get("error"),
        }
        records.append(rec)
        print(f"    turn {i+1:<2} history={int(hist_tokens):>7} tok  "
              f"ttft={str(rec['ttft_ms']):>9} ms"
              f"{'  [429]' if rec['capped'] else ''}"
              f"{'  ' + str(r.get('error')) if not r['ok'] else ''}",
              file=sys.stderr)
        if not r["ok"]:
            break
        if rec["ttft_ms"] is not None:
            points.append((hist_tokens, rec["ttft_ms"]))
        # Grow the conversation the way an agent does: the assistant's
        # reply plus a chunk of tool output (file contents).
        messages.append({"role": "assistant", "content": r["content"] or "OK"})
        messages.append({
            "role": "user",
            "content": (_prefix(growth_tokens, f"{salt}-t{i}")
                        + f"\n\nGiven the file above, reply with OK ({i}).")})

    ok = [r for r in records if r["ok"] and r["ttft_ms"] is not None]
    return {
        "turns": records,
        "n_ok": len(ok),
        # THE headline: ms of TTFT added per 1K tokens of history.
        # ~0 => prefix reuse working. Large => re-prefilling every turn.
        "ttft_slope_ms_per_1k_tokens": _slope_ms_per_1k(points),
        "first_turn_ttft_ms": ok[0]["ttft_ms"] if ok else None,
        "last_turn_ttft_ms": ok[-1]["ttft_ms"] if ok else None,
        "final_history_tokens": ok[-1]["history_tokens"] if ok else None,
        "prefix_tokens": prefix_tokens,
        "growth_tokens": growth_tokens,
        "any_capped": any(r["capped"] for r in records),
    }


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--router-url", required=True,
                    help="e.g. http://devai-router:11436")
    ap.add_argument("--levels", default=",".join(str(x) for x in DEFAULT_LEVELS))
    ap.add_argument("--prefix-tokens", type=int, default=DEFAULT_PREFIX_TOKENS)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--mode", choices=("sweep", "multiturn"), default="sweep",
                    help="sweep = concurrency x prefix-sharing; "
                         "multiturn = sequential growing-history session")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--growth-tokens", type=int, default=4096)
    args = ap.parse_args()
    if args.mode == "multiturn":
        print(json.dumps(run_multiturn(
            model=args.model, router_url=args.router_url, turns=args.turns,
            prefix_tokens=args.prefix_tokens, growth_tokens=args.growth_tokens,
            max_tokens=args.max_tokens, timeout_s=args.timeout), indent=2))
        return
    out = run(
        model=args.model,
        router_url=args.router_url,
        levels=tuple(int(x) for x in args.levels.split(",") if x.strip()),
        prefix_tokens=args.prefix_tokens,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    _cli()
