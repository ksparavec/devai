"""Streaming latency + template-leak probe.

Sends a fixed prompt set to ``<router>/v1/chat/completions`` with
``stream:true`` and records:

  - **Time to first token** for each prompt. The very first prompt of
    a fresh model (``i==0``) is reported separately as
    ``ttft_ms_first`` — that's the cold-start metric the user wants
    (container recreate + weight load + KV alloc + prefill + first
    token). Remaining prompts give steady-state p50 / p95.
  - **Sustained tokens-per-second** during the streaming body of each
    request, p50 across prompts.
  - **Leak markers** in concatenated response content. The compiled
    regex set lives in ``data/leak_markers.txt``.

Mirrors the runtime style of scripts/probe-* — stdlib only, no third-
party HTTP. Can be imported (``run(...)``) or invoked as a module
directly for ad-hoc runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bench._bench_core import (  # noqa: E402  — local import after sys.path fix
    DATA_DIR,
    load_leak_patterns,
    p50,
    p95,
    stream_chat_completion,
    sweep_for_leaks,
)


def load_prompts(path: Path | None = None) -> list[dict]:
    p = path or (DATA_DIR / "latency_prompts.jsonl")
    out: list[dict] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def run(
    *,
    model: str,
    router_url: str,
    n: int = 40,
    timeout_s: float = 600.0,
    prompts_path: Path | None = None,
) -> dict:
    """Stream ``n`` prompts against the router; return aggregated metrics
    plus a leak-marker histogram.

    The first prompt's TTFT is recorded as ``ttft_ms_first`` (cold
    start). Remaining prompts feed ``ttft_ms_steady_p50`` and
    ``ttft_ms_steady_p95`` so the cold-start outlier doesn't pollute
    the warm-state percentiles.

    ``content_blob`` carries every response body concatenated with
    blank-line separators — that's what the leak sweep runs against.
    Returned ``leaked_markers`` is a flat dict keyed by the
    human-readable marker label (e.g. ``"<|im_end|>"``); zero-count
    markers are present so cross-model comparison is straightforward.
    """
    prompts = load_prompts(prompts_path)[:n]
    if not prompts:
        raise RuntimeError("no latency prompts loaded")

    ttft_first_ms: float | None = None
    ttft_steady_ms: list[float] = []
    tps_list: list[float] = []
    content_blobs: list[str] = []
    errors: list[dict] = []

    for i, item in enumerate(prompts):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": item["prompt"]}],
            "max_tokens": int(item.get("max_tokens", 256)),
            "temperature": 0,
        }
        try:
            res = stream_chat_completion(router_url, body, timeout=timeout_s)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            errors.append({"id": item.get("id", str(i)), "error": str(e)})
            continue

        if res["t_first_token"] is not None:
            ttft_ms = (res["t_first_token"] - res["t_open"]) * 1000
            if i == 0:
                ttft_first_ms = ttft_ms
            else:
                ttft_steady_ms.append(ttft_ms)
            stream_seconds = res["t_done"] - res["t_first_token"]
            # Use effective_tokens (max of usage.completion_tokens and
            # char-based estimate) so reasoning-heavy streams under the
            # qwen3 parser don't report 1/10th the real decode rate.
            tokens = res.get("effective_tokens") or res["completion_tokens"]
            if stream_seconds > 0 and tokens > 0:
                tps_list.append(tokens / stream_seconds)

        # Sweep both content AND reasoning_content for leaks — template
        # markers can leak into either field on misconfigured parsers.
        content_blobs.append(res["content"] or "")
        if res.get("reasoning_content"):
            content_blobs.append(res["reasoning_content"])

    blob = "\n\n".join(content_blobs)
    patterns = load_leak_patterns()
    leaks = sweep_for_leaks(blob, patterns)
    n_leak_hits = sum(leaks.values())

    return {
        "ttft_ms_first": (
            round(ttft_first_ms, 1) if ttft_first_ms is not None else None
        ),
        "ttft_ms_steady_p50": round(p50(ttft_steady_ms), 1),
        "ttft_ms_steady_p95": round(p95(ttft_steady_ms), 1),
        "tps_sustained_p50": round(p50(tps_list), 2),
        "n_samples": len(prompts) - len(errors),
        "n_errors": len(errors),
        "errors": errors[:5],  # cap so cache rows stay small
        "leak_rate": (
            round(n_leak_hits / max(len(prompts), 1), 4)
            if patterns else 0.0
        ),
        "leaked_markers": leaks,
    }


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="model name to send as `model:` field")
    ap.add_argument("--router-url", required=True, help="e.g. http://devai-router:11435")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    args = ap.parse_args()
    out = run(
        model=args.model,
        router_url=args.router_url,
        n=args.n,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    _cli()
