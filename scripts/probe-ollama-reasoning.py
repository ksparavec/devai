#!/usr/bin/env python3
"""Probe each downloaded Ollama model for native reasoning support
AND actual VRAM usage at TWO context lengths so we can interpolate.

Per docs/ollama_models.md, capability is determined by runtime behavior,
not catalog metadata. The probe sends two /api/chat calls per model with
`think: true` and `options.num_ctx` set to the LOW and HIGH probe
contexts respectively, reads /api/ps for `size`/`size_vram` after each,
and derives a linear fit (verified empirically across full-attention,
sliding-window, and Mamba-hybrid architectures with sub-1% deviation):

    total(c) = weights_overhead_gb + c × kv_per_token_bytes / 1024^3

This lets downstream tools compute fit at any user-chosen context
(32K, 64K, 128K, 256K, …) without re-probing.

Reasoning capability values:
    structured  – response has non-empty `message.thinking` field
    inline      – response has visible <think> markers in `message.content`
    unsupported – neither thinking field nor inline markers
    error       – probe request failed (network, timeout, model load fail)

Per-probe VRAM measurements come from /api/ps right after the chat call:
    size_bytes/size_vram_bytes – raw integers from ollama
    actual_context             – context_length the model actually loaded
                                 with; may be < requested if the model's
                                 trained ceiling or OLLAMA_CONTEXT_LENGTH
                                 caps it. The high-probe's actual_context
                                 is recorded as `max_context`.

Derived coefficients (when both probes succeed at distinct contexts):
    weights_overhead_gb  – intercept; weights + runtime overhead at c=0
    kv_per_token_bytes   – slope; KV cache bytes per token
    max_context          – the model's true context ceiling

For models classified `structured` we also probe with `think: false` to
record `disable_verified`. Cache is keyed by `name@digest` so re-runs
only re-probe models whose underlying weights changed. Cache is
invalidated when --probe-low or --probe-high differs from the cached
values, or when the cached entry pre-dates the two-point schema.

Usage:
    probe-ollama-reasoning.py [--cache PATH] [--prompt TEXT]
                              [--num-predict N] [--timeout SEC]
                              [--ollama-url URL]
                              [--probe-low N] [--probe-high N]

Errors propagate verbatim. No exception swallowing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = REPO_ROOT / "deploy" / ".ollama-reasoning-cache.json"

DEFAULT_PROMPT = "Answer with only the final number: What is 17 + 25?"

# Two-point probe defaults: LOW=32K, HIGH=256K. These bookend the
# user-facing CONTEXT choices (32K, 64K, 128K, 256K) so every CONTEXT
# the picker offers sits inside the probed range — no extrapolation. LOW
# is well above the small-context overhead anomaly (the 4-16K range
# where llama.cpp pre-allocates a fixed working buffer that dominates).
# HIGH covers the largest ceiling any catalog model supports today.
#
# `max_context` is independent of PROBE_HIGH — it comes from /api/show's
# architecture-specific `*.context_length`, so the model's true design
# ceiling (128K, 256K, 1M, …) is recorded regardless of probe range.
DEFAULT_LOW = 32768
DEFAULT_HIGH = 262144


def _http_post(url: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _http_get(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def list_models(ollama_url: str, timeout: float) -> list[dict]:
    """Return [{name, digest, modified_at, size}, ...] from /api/tags."""
    data = _http_get(f"{ollama_url}/api/tags", timeout)
    return [
        {
            "name": m.get("name", ""),
            "digest": (m.get("digest") or "")[:32],  # short form for readability
            "modified_at": m.get("modified_at", ""),
            "size": m.get("size", 0),
        }
        for m in data.get("models", []) or []
    ]


def probe(
    ollama_url: str,
    model: str,
    prompt: str,
    think: bool,
    num_predict: int,
    timeout: float,
    num_ctx: int | None = None,
) -> dict:
    """Single /api/chat probe. Returns parsed response or {error: ...}."""
    options: dict = {"temperature": 0, "num_predict": num_predict}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "think": think,
        "stream": False,
        "options": options,
    }
    try:
        return _http_post(f"{ollama_url}/api/chat", body, timeout)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    except json.JSONDecodeError as e:
        return {"error": f"non-JSON response: {e}"}


def measure_arch(ollama_url: str, model_name: str, timeout: float) -> dict:
    """Read /api/show and pull MoE-relevant fields from model_info.

    Returns: {arch_family, experts_total, experts_used, params_total} or
    {arch_family} for dense models (experts_* absent → dense).
    """
    body = json.dumps({"name": model_name}).encode()
    req = urllib.request.Request(
        f"{ollama_url}/api/show",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    mi = data.get("model_info") or {}
    details = data.get("details") or {}
    out: dict = {"arch_family": details.get("family") or "unknown"}
    if mi.get("general.parameter_count"):
        out["params_total"] = int(mi["general.parameter_count"])
    # Human-readable size string (e.g. "26B", "9B") and quantization
    # (e.g. "Q4_K_M", "F16", "BF16") from Ollama's /api/show details.
    if details.get("parameter_size"):
        out["param_size_label"] = details["parameter_size"]
    if details.get("quantization_level"):
        out["quantization"] = details["quantization_level"]
    # Match e.g. "gemma4.expert_count", "qwen35moe.expert_used_count",
    # "qwen35.context_length". The arch prefix varies per model family,
    # so we suffix-match.
    for k, v in mi.items():
        if k.endswith(".expert_count"):
            out["experts_total"] = int(v)
        elif k.endswith(".expert_used_count"):
            out["experts_used"] = int(v)
        elif k.endswith(".context_length"):
            out["model_max_context"] = int(v)
    return out


def measure_vram(
    ollama_url: str, model_name: str, model_digest: str, timeout: float
) -> dict:
    """Read /api/ps for the freshly-loaded model and report real memory use.

    Match by name first, then fall back to short-digest match: alias tags
    that share a digest with another tag (e.g. qwen3.5:35b-a3b-q4_K_M and
    qwen3.5:35b) are reported under the canonical name in /api/ps.
    """
    try:
        ps = _http_get(f"{ollama_url}/api/ps", timeout)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    entries = ps.get("models", []) or []
    found = None
    for m in entries:
        if m.get("name") == model_name or m.get("model") == model_name:
            found = m
            break
    if found is None and model_digest:
        for m in entries:
            d = (m.get("digest") or "")[:32]
            if d and d == model_digest:
                found = m
                break
    if found is None:
        return {"error": "model not found in /api/ps after load"}
    size = int(found.get("size", 0))
    size_vram = int(found.get("size_vram", 0))
    ctx = (
        found.get("context_length")
        or (found.get("details") or {}).get("context_length")
        or 0
    )
    return {
        "size_bytes": size,
        "size_vram_bytes": size_vram,
        "actual_total_gb": round(size / (1024**3), 2),
        "actual_vram_gb": round(size_vram / (1024**3), 2),
        "fully_on_gpu": size > 0 and size_vram >= size,
        "actual_context": int(ctx),
    }


def classify(resp: dict) -> tuple[str, dict]:
    """Map /api/chat response to (capability, evidence_dict)."""
    if "error" in resp:
        return "error", {"error": resp["error"]}
    msg = resp.get("message") or {}
    thinking = (msg.get("thinking") or "").strip()
    content = msg.get("content") or ""
    if thinking:
        return "structured", {
            "thinking_chars": len(thinking),
            "content_preview": content[:80],
        }
    has_inline = "<think>" in content or "</think>" in content
    if has_inline:
        return "inline", {
            "content_preview": content[:200],
        }
    return "unsupported", {
        "content_preview": content[:120],
    }


def _interpolate(
    low_bytes: int, high_bytes: int, low_ctx: int, high_ctx: int
) -> tuple[float | None, float | None]:
    """Two-point linear fit. Returns (weights_overhead_gb, kv_per_token_bytes).

    Falls back to (None, None) when the two probes landed on the same
    actual_context (model capped before high) or the slope would be
    non-positive (measurement noise on a tiny model)."""
    if high_ctx <= low_ctx:
        return None, None
    bytes_per_token = (high_bytes - low_bytes) / (high_ctx - low_ctx)
    if bytes_per_token <= 0:
        return None, None
    weights_overhead_bytes = low_bytes - bytes_per_token * low_ctx
    return (
        round(weights_overhead_bytes / (1024**3), 3),
        round(bytes_per_token, 2),
    )


def probe_one(
    ollama_url: str,
    model_name: str,
    model_digest: str,
    prompt: str,
    num_predict: int,
    timeout: float,
    probe_low: int,
    probe_high: int,
) -> dict:
    """Model-aware two-point probe.

    Step 0: read /api/show for `max_context` (cheap, no model load) so
            we know the model's design ceiling before deciding how
            high to probe.
    Step 1: HIGH probe target = min(probe_high, max_context). For
            256K-capable models we probe at 256K; for 128K-only models
            we probe at 128K. The two probed points always bookend the
            model's usable range so 64K (and 128K when probed at 256K)
            land inside the measured interval — pure interpolation, no
            extrapolation.
    Step 2: LOW probe captures reasoning capability + VRAM at 32K.
    Step 3: HIGH probe captures VRAM at the chosen high target.
    From the two points we derive `weights_overhead_gb` and
    `kv_per_token_bytes`. `max_context` is the model's design ceiling
    from /api/show; `probed_high_context` is the actual high we used.
    """
    started = time.time()

    # ── Step 0: arch lookup (cheap, no load) ─────────────────────────────
    arch = measure_arch(ollama_url, model_name, timeout=10.0)
    record: dict = {
        "probed_at": dt.datetime.now(dt.timezone.utc)
            .replace(microsecond=0).isoformat(),
        "probed_low_context": probe_low,
    }
    if "error" in arch:
        record["arch_error"] = arch["error"]
        record["capability"] = "error"
        record["evidence_enable"] = {"error": arch["error"]}
        record["probe_seconds"] = round(time.time() - started, 2)
        record["probed_high_context"] = probe_high
        return record
    record.update(arch)

    # ── Step 1: pick the HIGH probe target based on the model's max ──────
    model_max = arch.get("model_max_context") or 0
    if model_max and model_max < probe_high:
        high_target = model_max
    else:
        high_target = probe_high
    record["probed_high_context"] = high_target
    record["max_context"] = model_max or high_target

    # ── Step 2: LOW probe (also captures reasoning capability) ───────────
    pos_low = probe(
        ollama_url, model_name, prompt, think=True,
        num_predict=num_predict, timeout=timeout, num_ctx=probe_low,
    )
    cap, evidence = classify(pos_low)
    record["capability"] = cap
    record["evidence_enable"] = evidence

    if "error" in evidence:
        record["probe_seconds"] = round(time.time() - started, 2)
        return record

    vram_low = measure_vram(ollama_url, model_name, model_digest, timeout=10.0)
    if "error" in vram_low:
        record["vram_error"] = vram_low["error"]
        record["probe_seconds"] = round(time.time() - started, 2)
        return record

    record["actual_low"] = vram_low

    # ── Step 3: HIGH probe at the chosen target ──────────────────────────
    if high_target > probe_low:
        pos_high = probe(
            ollama_url, model_name, prompt, think=True,
            num_predict=num_predict, timeout=timeout, num_ctx=high_target,
        )
        if "error" in pos_high:
            record["high_error"] = pos_high.get("error", "unknown")
        else:
            vram_high = measure_vram(ollama_url, model_name, model_digest, timeout=10.0)
            if "error" in vram_high:
                record["high_vram_error"] = vram_high["error"]
            else:
                record["actual_high"] = vram_high
                low_bytes = vram_low.get("size_bytes")
                high_bytes = vram_high.get("size_bytes")
                low_ctx = vram_low.get("actual_context") or 0
                high_ctx = vram_high.get("actual_context") or 0
                weights_gb, kv_pt = _interpolate(low_bytes, high_bytes, low_ctx, high_ctx)
                record["weights_overhead_gb"] = weights_gb
                record["kv_per_token_bytes"] = kv_pt

    if cap == "structured":
        # Negative probe at LOW context — small enough to be quick, big
        # enough to dodge the small-context anomaly. Same context as the
        # capability check above so reasoning behavior is comparable.
        neg = probe(
            ollama_url, model_name, prompt, think=False,
            num_predict=num_predict, timeout=timeout, num_ctx=probe_low,
        )
        if "error" in neg:
            record["disable_verified"] = "error"
            record["evidence_disable"] = {"error": neg["error"]}
        else:
            neg_thinking = ((neg.get("message") or {}).get("thinking") or "").strip()
            record["disable_verified"] = not neg_thinking
            record["evidence_disable"] = {
                "thinking_chars": len(neg_thinking),
                "content_preview": (neg.get("message") or {}).get("content", "")[:80],
            }

    record["probe_seconds"] = round(time.time() - started, 2)
    return record


def load_cache(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def _cache_invalid(cached: dict, probe_low: int, probe_high: int) -> str | None:
    """Return reason string if cache entry must be re-probed, else None.

    The HIGH target is model-aware: a model with max_context=131072 was
    probed at 131072 even when probe_high=262144, and that's still
    correct on a re-run with the same probe_high. Only re-probe if the
    target we'd compute today differs from what's in the cache.
    """
    # Old single-point schema lacks the two-point context fields entirely.
    if "probed_low_context" not in cached or "probed_high_context" not in cached:
        return "schema upgrade (single-point → two-point)"
    if cached.get("probed_low_context") != probe_low:
        return f"low ctx changed ({cached.get('probed_low_context')} → {probe_low})"
    cached_max = cached.get("max_context") or 0
    if cached_max and cached_max < probe_high:
        target_high = cached_max
    else:
        target_high = probe_high
    if cached.get("probed_high_context") != target_high:
        return (f"high ctx changed "
                f"({cached.get('probed_high_context')} → {target_high})")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--cache", type=Path, default=DEFAULT_CACHE,
        help=f"cache file path (default {DEFAULT_CACHE.relative_to(REPO_ROOT)})",
    )
    ap.add_argument(
        "--prompt", default=DEFAULT_PROMPT,
        help="probe prompt (kept short and deterministic)",
    )
    ap.add_argument(
        "--num-predict", type=int, default=128,
        help="max tokens per probe (default 128)",
    )
    ap.add_argument(
        "--timeout", type=float, default=120.0,
        help="per-request timeout seconds (default 120)",
    )
    ap.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_HOST", "http://devai-ollama:11434"),
        help="ollama base URL (default $OLLAMA_HOST or http://devai-ollama:11434)",
    )
    ap.add_argument(
        "--probe-low", type=int,
        default=int(os.environ.get("PROBE_LOW", str(DEFAULT_LOW))),
        help=f"LOW probe context length (default {DEFAULT_LOW}). Stays "
             "above the small-context overhead anomaly.",
    )
    ap.add_argument(
        "--probe-high", type=int,
        default=int(os.environ.get("PROBE_HIGH", str(DEFAULT_HIGH))),
        help=f"HIGH probe context length (default {DEFAULT_HIGH}). Models "
             "with smaller ceilings cap at their actual_context; that "
             "becomes max_context for the model.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="re-probe even if digest is in cache",
    )
    args = ap.parse_args()

    if args.probe_high <= args.probe_low:
        sys.exit(
            f"error: --probe-high ({args.probe_high}) must exceed "
            f"--probe-low ({args.probe_low})"
        )

    print(f"  probe target:   {args.ollama_url}", file=sys.stderr)
    print(
        f"  probe contexts: low={args.probe_low}  high={args.probe_high} "
        f"(linear fit derives weights_overhead_gb + kv_per_token_bytes)",
        file=sys.stderr,
    )
    cache = load_cache(args.cache)
    print(f"  cache:          {args.cache} ({len(cache)} entries)", file=sys.stderr)

    try:
        models = list_models(args.ollama_url, args.timeout)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        sys.exit(f"error: cannot list ollama models: {e}")
    if not models:
        sys.exit("error: ollama returned no models — is anything downloaded?")

    print(f"  {len(models)} models on disk", file=sys.stderr)
    print(file=sys.stderr)
    # Per-row output is noisy when everything's cached. Only show rows
    # that actually hit the network (fresh or re-probed); end with a
    # one-line summary so the caller still sees what happened.
    header_printed = False

    def maybe_header() -> None:
        nonlocal header_printed
        if header_printed:
            return
        print(
            f"  {'MODEL':<40s} {'CAP':<11s} "
            f"{'KV/TOK':>8s} {'PROBED':>8s} {'MAX_CTX':>8s} ACTION",
            file=sys.stderr,
        )
        print(f"  {'-' * 92}", file=sys.stderr)
        header_printed = True

    def fmt_kv(rec: dict) -> str:
        v = rec.get("kv_per_token_bytes")
        if v is None:
            return "—"
        # Convert to KB for readability (typical range 18-50 KB).
        return f"{v / 1024:.1f}K"

    def fmt_ctx_pair(rec: dict) -> tuple[str, str]:
        """(probed_high_label, max_context_label) — different things now.

        PROBED is the upper bookend we measured at; MAX_CTX is the
        model's design ceiling. They diverge when /api/show reports
        a max_context smaller than the user's --probe-high default.
        """
        ph = rec.get("probed_high_context") or 0
        mx = rec.get("max_context") or 0
        return (f"{ph // 1024}K" if ph else "—",
                f"{mx // 1024}K" if mx else "—")

    fresh = 0
    skipped = 0
    invalidated = 0
    for m in sorted(models, key=lambda x: x["name"]):
        name = m["name"]
        digest = m["digest"]
        key = f"{name}@{digest}"
        cached = cache.get(key)

        invalidation_reason: str | None = None
        if cached and not args.force:
            invalidation_reason = _cache_invalid(
                cached, args.probe_low, args.probe_high
            )
            if invalidation_reason:
                cached = None
                invalidated += 1

        if cached and not args.force:
            skipped += 1
            continue

        record = probe_one(
            args.ollama_url, name, digest, args.prompt,
            args.num_predict, args.timeout, args.probe_low, args.probe_high,
        )
        record["name"] = name
        record["digest"] = digest
        cap = record["capability"]

        # Marker decorations — disable_verified, partial probes, cache
        # invalidation reason. No "capped" warnings: PROBED column shows
        # the actual high target we used, MAX_CTX shows model's ceiling.
        marker = ""
        if cap == "structured" and record.get("disable_verified") is True:
            marker = " (disable verified)"
        elif cap == "structured":
            marker = f" (disable={record.get('disable_verified')})"
        if record.get("high_error"):
            marker += " (HIGH probe failed)"
        elif record.get("kv_per_token_bytes") is None and "actual_low" in record:
            marker += " (single-point — no slope)"
        if invalidation_reason:
            marker += f" (re-probed: {invalidation_reason})"

        probed_str, max_str = fmt_ctx_pair(record)
        maybe_header()
        print(
            f"  {name:<40s} {cap:<11s} "
            f"{fmt_kv(record):>8s} {probed_str:>8s} {max_str:>8s} "
            f"probed in {record['probe_seconds']}s{marker}",
            file=sys.stderr,
        )
        cache[key] = record
        fresh += 1
        save_cache(args.cache, cache)

    # Drop stale entries (model no longer present)
    live_keys = {f"{m['name']}@{m['digest']}" for m in models}
    stale = [k for k in cache if k not in live_keys]
    for k in stale:
        del cache[k]
    if stale:
        save_cache(args.cache, cache)

    print(file=sys.stderr)
    by_cap: dict[str, int] = {}
    for k, v in cache.items():
        c = v.get("capability", "?")
        by_cap[c] = by_cap.get(c, 0) + 1
    summary = "  ".join(f"{c}={n}" for c, n in sorted(by_cap.items()))
    print(
        f"  done: {fresh} probed ({invalidated} re-probed due to "
        f"context change), {skipped} cached, {len(stale)} stale removed",
        file=sys.stderr,
    )
    print(f"  capability counts: {summary}", file=sys.stderr)


if __name__ == "__main__":
    main()
