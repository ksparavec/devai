#!/usr/bin/env python3
"""Long-context probe: send ONE request at ~80% of the model's max
context tier and capture how the backend behaves under KV pressure.

The existing ``leak`` probe (40 short prompts, sequential) catches
parser leaks and short-prompt latency but never builds queue pressure
or fills the KV pool. The long-context probe complements it by
exercising the failure modes that only surface near the context
ceiling:

  - prefill latency at large input sizes
  - KV-cache utilization at near-saturation
  - paged-attention preemption (request gets swapped/recomputed)
  - refusal-by-context-overflow (if the request silently truncates or
    errors at exactly the boundary the probe never thought about)

What we capture per run:

  - ``input_tokens``         -- request body's prompt token count, as
                               reported by vLLM in usage.prompt_tokens
                               (or `None` if the response didn't carry
                               it)
  - ``output_tokens``        -- usage.completion_tokens
  - ``ttft_ms``              -- ``t_first_token - t_open`` in ms; the
                               prefill cost we wanted to measure
  - ``tps_during_decode``    -- output_tokens / (t_done - t_first_token)
                               for the decode phase only
  - ``peak_kv_cache_perc``   -- vllm:kv_cache_usage_perc snapshot taken
                               immediately after the request returns
                               (best-effort; null if /metrics is
                               unreachable)
  - ``preemptions``          -- vllm:num_preemptions_total snapshot at
                               request end (the counter resets per
                               container recreate, so this is the
                               run-local total)
  - ``finish_reason``        -- ``stop`` / ``length`` / ``error``
  - ``error``                -- non-null when the backend rejected
                               the request (e.g. context overflow);
                               carries the upstream message verbatim

Caller is the same path as the leak probe -- bench_runner injects
the optional ``fetch_metrics`` callable so this module doesn't need
to know how vLLM /metrics endpoints are addressed.
"""

from __future__ import annotations

import time
from typing import Callable

from bench._bench_core import stream_chat_completion

# Average chars per token for English text. Used to size the filler
# string so vLLM's tokenizer ends up with roughly the requested input
# length. Slightly conservative (under-shoots) to avoid tripping
# max_model_len on the upper end. Tuned empirically across the bench's
# vLLM tokenizers (Llama-3, Qwen-2/3, DeepSeek, Nemotron) -- they all
# land within 3.0-3.8 chars/token on prose, so 3.5 is a safe middle.
_CHARS_PER_TOKEN = 3.5

# Public-domain prose chunk to repeat as filler. Picked deliberately:
# a) it's recognisable, so the question at the tail can reference it
# without ambiguity; b) it's English-only, so byte-level BPE families
# (Llama-3) tokenize predictably; c) it has no special tokens or chat
# markers that could confuse the chat template. About 1.4 KB; we
# repeat as needed to hit the target.
_FILLER_CHUNK = (
    "Call me Ishmael. Some years ago -- never mind how long precisely "
    "-- having little or no money in my purse, and nothing particular "
    "to interest me on shore, I thought I would sail about a little "
    "and see the watery part of the world. It is a way I have of "
    "driving off the spleen, and regulating the circulation. Whenever "
    "I find myself growing grim about the mouth; whenever it is a "
    "damp, drizzly November in my soul; whenever I find myself "
    "involuntarily pausing before coffin warehouses, and bringing up "
    "the rear of every funeral I meet; and especially whenever my "
    "hypos get such an upper hand of me, that it requires a strong "
    "moral principle to prevent me from deliberately stepping into "
    "the street, and methodically knocking people's hats off -- then, "
    "I account it high time to get to sea as soon as I can. This is "
    "my substitute for pistol and ball. With a philosophical flourish "
    "Cato throws himself upon his sword; I quietly take to the ship. "
    "There is nothing surprising in this. If they but knew it, almost "
    "all men in their degree, some time or other, cherish very nearly "
    "the same feelings towards the ocean with me. "
)

_TAIL_INSTRUCTION = (
    "\n\nBased on the long passage above, answer in EXACTLY one short "
    "sentence: who is the narrator and what does he plan to do? Do "
    "not quote the passage. Do not explain. One sentence."
)


def _build_long_prompt(target_input_tokens: int) -> str:
    """Build a prompt of approximately ``target_input_tokens`` tokens.

    Repeats ``_FILLER_CHUNK`` enough times to fill (target * chars-per-
    token) characters, then appends the tail instruction. Slight
    under-estimate is preferred over an overflow; vLLM rejects requests
    that exceed ``--max-model-len`` and the run is wasted.
    """
    target_chars = max(0, target_input_tokens) * _CHARS_PER_TOKEN
    target_chars -= len(_TAIL_INSTRUCTION)
    if target_chars <= 0:
        return _TAIL_INSTRUCTION.lstrip()
    repeats = max(1, int(target_chars // len(_FILLER_CHUNK)) + 1)
    body = (_FILLER_CHUNK * repeats)[: int(target_chars)]
    return body + _TAIL_INSTRUCTION


def run(
    *,
    model: str,
    router_url: str,
    ctx_target: int,
    fraction: float = 0.8,
    max_output_tokens: int = 64,
    timeout_s: float = 600.0,
    fetch_metrics: Callable[[], dict[str, float]] | None = None,
) -> dict:
    """Run one long-context probe and return a result dict.

    ``ctx_target`` is the model's effective max context (the ``@<ctx>``
    suffix the picker emits). The probe sends a prompt sized to
    ``fraction * ctx_target`` tokens.

    ``fetch_metrics`` is invoked once after the request returns to
    capture vLLM's /metrics snapshot. The caller (bench_runner) passes
    in its existing ``_fetch_backend_metrics(backend)`` partial.
    Returns ``{}`` on transport failure -- result fields just stay
    ``None``.

    Errors from the backend (HTTP 4xx/5xx, timeout, context overflow)
    are caught and returned as ``error`` in the result rather than
    raised. The bench harness treats any longctx failure as soft so a
    single bad row doesn't void the run.
    """
    target_in = int(ctx_target * fraction)
    prompt = _build_long_prompt(target_in)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_output_tokens),
        "temperature": 0.0,
    }

    started = time.time()
    error: str | None = None
    res: dict | None = None
    try:
        res = stream_chat_completion(router_url, body, timeout=timeout_s)
    except Exception as e:  # noqa: BLE001 -- backend errors come in many shapes
        error = f"{type(e).__name__}: {e}"
    elapsed = time.time() - started

    # End-of-request /metrics snapshot. Best-effort: bench result
    # remains valid without it.
    metrics_snap: dict[str, float] = {}
    if fetch_metrics is not None:
        try:
            metrics_snap = fetch_metrics() or {}
        except Exception:  # noqa: BLE001
            metrics_snap = {}

    if res is None:
        return {
            "input_tokens_target": target_in,
            "input_tokens_chars": len(prompt),
            "ttft_ms": None,
            "tps_during_decode": None,
            "output_tokens": None,
            "finish_reason": None,
            "peak_kv_cache_perc": metrics_snap.get("vllm_kv_cache_usage_perc"),
            "preemptions": metrics_snap.get("vllm_num_preemptions_total"),
            "elapsed_s": round(elapsed, 1),
            "error": error,
        }

    ttft_ms = None
    if res.get("t_first_token") is not None:
        ttft_ms = round((res["t_first_token"] - res["t_open"]) * 1000.0, 1)

    tps_decode = None
    if (
        res.get("t_first_token") is not None
        and res.get("output_tokens") is None
    ):
        # Use effective_tokens: parser-agnostic, accounts for reasoning
        # streams the same way the leak probe's TPS does.
        eff = res.get("effective_tokens") or 0
        decode_dt = res["t_done"] - res["t_first_token"]
        if decode_dt > 0 and eff > 0:
            tps_decode = round(eff / decode_dt, 2)
    elif res.get("t_first_token") is not None:
        eff = res.get("effective_tokens") or 0
        decode_dt = res["t_done"] - res["t_first_token"]
        if decode_dt > 0 and eff > 0:
            tps_decode = round(eff / decode_dt, 2)

    return {
        "input_tokens_target": target_in,
        "input_tokens_chars": len(prompt),
        "ttft_ms": ttft_ms,
        "tps_during_decode": tps_decode,
        "output_tokens": res.get("completion_tokens") or res.get("effective_tokens"),
        "finish_reason": res.get("finish_reason"),
        "peak_kv_cache_perc": metrics_snap.get("vllm_kv_cache_usage_perc"),
        "preemptions": metrics_snap.get("vllm_num_preemptions_total"),
        "elapsed_s": round(elapsed, 1),
        "error": None,
    }
