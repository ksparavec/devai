#!/usr/bin/env python3
"""Load-probe: serving-time VRAM measurement under near-full-context fills.

The fit prober (probe-{vllm,sglang}-reasoning) launches each model and
snapshots GPU memory right after /health plus one short chat. That
captures the *load ceiling* -- weights + the reserved KV pool -- but NOT
the *serving transient* vLLM/SGLang allocate once a real, near-full
context request arrives: the softcap-logits buffer
(``max_num_batched_tokens x vocab x 4 bytes``) plus attention and
activation workspace. DiffusionGemma fit at 256K in the fit cache yet
crash-looped on the first >512-token prompt because that transient did
not fit the free VRAM the fit probe left unmeasured.

This pass *layers onto* the existing fit cache -- it never rewrites a
fit cell, only augments it. For each downloaded model, at the host VRAM
band, it walks the context tiers ASCENDING (32K -> 64K -> 128K -> 256K),
and for every tier the fit prober already marked ``fits=true`` it:

  1. relaunches the backend at that ``--max-model-len`` with the SAME
     verified parsers + recovery flags the router serves with,
  2. records a baseline VRAM reading once /health passes (model loaded,
     idle, before the big request),
  3. builds a haystack prompt filled to ``ctx - 2048`` tokens from
     vendored public-domain books, with a unique needle at mid-depth,
  4. sends ONE ~2048-token completion while a 0.1s VramSampler runs,
  5. captures the peak, scores needle retrieval, and classifies OOM
     (HTTP/transport error / container exit / OOM markers in logs),
  6. augments the existing cell with ``serving_ok`` / ``serving_peak_gb``
     / ``transient_gb`` / ``needle_score`` / ``predicted_logits_gb``,
  7. STOPS ascending at the first OOM -- higher tiers can only be worse,
     and marks them ``serving_ok=false`` (implied) without a launch.

Additive cache fields only; no schema bump. Existing readers ignore the
new keys; the router + picker gate on ``serving_ok`` when present.

Hard pre-condition (same as the fit prober): devai-router / devai-vllm /
devai-sglang stopped. Run ``make cache-down`` first. Errors propagate;
no silent swallowing beyond the request-level error that IS the OOM
signal being measured.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from _contexts import (
    binary_search_max_ctx,
    context_label,
    vram_label,
)
from _probe_core import (
    image_digest_via_cli,
    load_cache,
    now_iso,
    save_cache,
    stamp_image_digest,
)
from _probe_hf_common import (
    CHAT_TIMEOUT,
    HEALTH_POLL_INTERVAL,
    STARTUP_TIMEOUT,
    BackendSpec,
    _ledger_clear,
    _ledger_record,
    _ledger_reason,
    _load_ledger,
    _post_chat,
    _resolve_plugins,
    _save_ledger,
    assert_no_active_backends,
    classify_failure_logs,
    container_logs,
    container_remove,
    container_run_detached,
    container_state,
    effective_position_limit,
    gpu_memory_used_mb,
    host_scaled_fraction,
    http_get,
    http_post,
    is_downloaded,
    load_catalog_hf_rows,
    model_size_gb_from_row,
    recovery_image,
    recovery_overrides,
)

# bench/ is a package; VramSampler polls nvidia-smi on a fixed cadence.
# 0.1s catches the prefill transient peak that a single-shot snapshot
# (what the fit prober uses) would miss between decode steps.
from bench.bench_vram_snapshot import VramSampler

# ── Corpus + needle ──────────────────────────────────────────────────────────

# The haystack corpus is public-domain text -- too large (~4.4 MB) and too
# freely-downloadable to vendor in git. It is fetched on first use into the
# user cache (NOT /var/cache/devai -- that path holds only empty folders
# used as external-volume mount points; see CLAUDE.md). Override the
# location with DEVAI_PROBE_CORPUS_DIR (e.g. on an air-gapped host:
# pre-populate it and the probe never reaches the network).
_XDG_CACHE = os.environ.get("XDG_CACHE_HOME") or os.path.join(
    os.path.expanduser("~"), ".cache")
_CORPUS_DIR = Path(os.environ.get(
    "DEVAI_PROBE_CORPUS_DIR",
    os.path.join(_XDG_CACHE, "devai", "probe-corpus")))

# cache filename -> Project Gutenberg mirror URLs (tried in order). The
# /cache/epub/ form is the modern canonical path; /files/ is the fallback.
_CORPUS_SOURCES: dict[str, list[str]] = {
    "moby-dick.txt": [
        "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
        "https://www.gutenberg.org/files/2701/2701-0.txt",
    ],
    "war-and-peace.txt": [
        "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
        "https://www.gutenberg.org/files/2600/2600-0.txt",
    ],
}

# A fetched book must clear this many chars after boilerplate stripping or
# we treat the download as junk (truncated / error page) and try the next
# mirror. Both target books are > 1 MB.
_CORPUS_MIN_CHARS = 200_000

# Seed estimate only -- the real prompt size is tokenizer-verified against
# the backend's /tokenize endpoint (see build_full_window_prompt). English
# prose is ~4 chars/token under BPE; 3.5 deliberately undershoots so the
# first pass is below target and we grow up to it rather than overflow.
_CHARS_PER_TOKEN = 3.5

# Tokens reserved below --max-model-len for the completion, and the max
# completion length. Reasoning models (capability structured/inline) emit a
# long <think> trace before the final answer; a 256-token budget gets
# consumed by the trace and the needle-recall answer never lands, scoring a
# FALSE needle=0.0 (observed on Qwen3.6-27B-MTP-pi-tune-NVFP4: 0.0 at 256,
# 1.0 at 2048 -- reasoning recall was fine all along). 2048 gives think room.
# Trade-off: the larger headroom fills the KV pool slightly less at prefill
# (~93% at 32K vs ~98% before), so the pool-ceiling OOM signal is marginally
# weaker at small ctx. Accepted: the dominant serving OOM on this fleet is the
# max_num_batched_tokens softcap-logits / attention-workspace transient
# (independent of this budget), not the last few % of KV. max_tokens stays
# below headroom so prompt+output < ctx (no length-reject).
_OUTPUT_HEADROOM_TOKENS = 2304
_MAX_OUTPUT_TOKENS = 2048

# Tokenizer-verify loop bounds: converge the prompt to within _TOKENIZE_TOL
# tokens of (ctx - _OUTPUT_HEADROOM_TOKENS), at most _TOKENIZE_MAX_ITERS
# /tokenize round-trips. Land AT or just UNDER target (never over, so the
# request can't be length-rejected).
_TOKENIZE_TOL = 96
_TOKENIZE_MAX_ITERS = 6

# Chars of corpus sent in the calibration chat when /tokenize is absent
# (e.g. SGLang) -- enough to measure chars/token accurately, small enough
# to prefill in a couple seconds.
_CALIBRATION_SAMPLE_CHARS = 40_000

# A novel string that cannot be in any training set, so a correct answer
# proves retrieval from THIS context rather than memorisation.
_NEEDLE_CODE = "RHINO-7741-DELTA-VAULT"
_NEEDLE_SENTENCE = (
    "\n\n=== OPERATOR NOTE (memorise this) ===\n"
    f"The one-time vault access code for this session is {_NEEDLE_CODE}. "
    "You will be asked to recall it verbatim at the very end.\n"
    "=== END OPERATOR NOTE ===\n\n"
)
_QUESTION = (
    "\n\nIgnore the literary content above. Question: what is the exact "
    "vault access code stated in the OPERATOR NOTE somewhere in the text "
    "above? Reply with ONLY the code, nothing else."
)

# Substrings that mean the engine OOMed / died mid-request even when the
# container is still nominally 'running' (vLLM may return a 500 but keep
# the process up; SGLang tends to exit). Case-insensitive scan.
_OOM_MARKERS = (
    "out of memory",
    "outofmemory",
    "cuda error",
    "enginedeaderror",
    "enginecore",
    "no available memory for the cache blocks",
    "died with",
)

# Chars of container-log tail stored on a failure. Engine-init crashes log
# the real root cause ABOVE the outer "Engine core initialization failed"
# wrapper, so a too-short tail captures only the wrapper -- 8K covers the
# full subprocess traceback for the failures seen on this fleet.
_LOG_EXCERPT_CHARS = 8000


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Drop the Project Gutenberg license header/footer, keeping the work
    itself. Marks: '*** START OF TH... ***' / '*** END OF TH...'."""
    start = text.find("*** START OF TH")
    if start != -1:
        nl = text.find("\n", start)
        if nl != -1:
            text = text[nl + 1:]
    end = text.find("*** END OF TH")
    if end != -1:
        text = text[:end]
    return "\n".join(ln.rstrip() for ln in text.splitlines()).strip()


def _download_text(url: str, timeout: float = 60.0) -> str:
    """GET a URL as UTF-8 text. A real User-Agent avoids Gutenberg's
    default-urllib block. Honours http(s)_proxy env via urllib."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "devai-load-probe/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — fixed gutenberg.org URLs
        return resp.read().decode("utf-8", errors="replace")


def _ensure_corpus_file(name: str, urls: list[str], cache_dir: Path) -> Path:
    """Return the cached book path, downloading + boilerplate-stripping
    from the first working mirror when absent. Raises with an actionable
    message if every mirror fails and nothing is cached."""
    dest = cache_dir / name
    if dest.is_file() and dest.stat().st_size >= _CORPUS_MIN_CHARS:
        return dest
    cache_dir.mkdir(parents=True, exist_ok=True)
    last_err = "no mirrors tried"
    for url in urls:
        try:
            stripped = _strip_gutenberg_boilerplate(_download_text(url))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = f"{url}: {type(e).__name__}: {e}"
            continue
        if len(stripped) < _CORPUS_MIN_CHARS:
            last_err = f"{url}: stripped text too short ({len(stripped)} chars)"
            continue
        dest.write_text(stripped, encoding="utf-8")
        print(f"    [corpus] fetched {name} <- {url} "
              f"({len(stripped) // 1024}K chars) -> {dest}", file=sys.stderr)
        return dest
    raise RuntimeError(
        f"could not obtain load-probe corpus '{name}': not cached at {dest} "
        f"and all mirrors failed (last: {last_err}). On an air-gapped host, "
        f"set DEVAI_PROBE_CORPUS_DIR to a dir holding {name}.")


def load_corpus() -> str:
    """Concatenate the public-domain books into one haystack, fetching any
    that aren't cached (see _CORPUS_DIR / _CORPUS_SOURCES)."""
    parts = [
        _ensure_corpus_file(name, urls, _CORPUS_DIR).read_text(
            encoding="utf-8", errors="replace")
        for name, urls in _CORPUS_SOURCES.items()
    ]
    return "\n\n".join(parts)


def _assemble_prompt(corpus: str, body_chars: int, depth: float) -> str:
    """Build a haystack of ``body_chars`` chars from the (repeated) corpus,
    insert the needle at ``depth`` (0.0=top, 1.0=bottom) snapped to a
    paragraph boundary, and append the recall question."""
    body_chars = max(1, body_chars)
    haystack = corpus
    while len(haystack) < body_chars:
        haystack += "\n\n" + corpus
    haystack = haystack[:body_chars]
    cut = max(0, min(len(haystack), int(len(haystack) * depth)))
    nl = haystack.find("\n", cut)
    if nl != -1:
        cut = nl
    return haystack[:cut] + _NEEDLE_SENTENCE + haystack[cut:] + _QUESTION


def build_haystack_prompt(corpus: str, ctx_target: int, depth: float = 0.5) -> str:
    """Char-estimate fill to ``ctx_target - _OUTPUT_HEADROOM_TOKENS`` tokens.
    Used as the seed/fallback when no tokenizer is reachable; the actual
    full-window fill is tokenizer-verified by build_full_window_prompt."""
    body_tokens = max(256, ctx_target - _OUTPUT_HEADROOM_TOKENS)
    return _assemble_prompt(corpus, int(body_tokens * _CHARS_PER_TOKEN), depth)


def _count_tokens(base_url: str, model_name: str, prompt: str) -> int | None:
    """Token count of ``prompt`` AS the backend will see it in a chat
    request -- i.e. with the chat template applied (add_generation_prompt)
    so template overhead is included. Uses the OpenAI-compatible /tokenize
    endpoint (vLLM and SGLang both expose it). None when unavailable, so
    the caller falls back to the char estimate.
    """
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "add_generation_prompt": True,
    }
    try:
        resp = http_post(f"{base_url}/tokenize", body, timeout=30.0)
    except (urllib.error.URLError, TimeoutError, OSError,
            json.JSONDecodeError):
        return None
    if not isinstance(resp, dict) or resp.get("error"):
        return None
    if isinstance(resp.get("count"), int):
        return resp["count"]
    toks = resp.get("tokens")
    return len(toks) if isinstance(toks, list) else None


def _calibrate_chars_per_token(
    base_url: str, model_name: str, corpus: str,
) -> float | None:
    """Measure chars/token for this model+corpus from a short chat's
    usage.prompt_tokens -- a backend-agnostic substitute for /tokenize on
    engines that don't expose it (e.g. SGLang). Returns None when the
    calibration chat fails (caller then keeps the static char estimate).
    """
    sample = corpus[:_CALIBRATION_SAMPLE_CHARS]
    resp = _post_chat(
        base_url,
        {"model": model_name,
         "messages": [{"role": "user", "content": sample}],
         "max_tokens": 1, "temperature": 0.0, "stream": False},
        timeout=180.0,
    )
    if isinstance(resp, dict) and not resp.get("error"):
        pt = (resp.get("usage") or {}).get("prompt_tokens")
        if isinstance(pt, int) and pt > 0:
            return len(sample) / pt
    return None


def build_full_window_prompt(
    base_url: str, model_name: str, corpus: str, ctx: int, depth: float,
) -> tuple[str, int | None, str]:
    """Build a prompt that fills the KV pool to ~99% of ``ctx``: target
    ``ctx - _OUTPUT_HEADROOM_TOKENS`` ACTUAL tokens, verified via /tokenize
    and grown/trimmed to land at-or-just-under target. This is what makes
    the load probe exercise the pool near its true ceiling instead of the
    ~88% a static char estimate reaches.

    Returns ``(prompt, actual_tokens|None, method)`` where method is
    ``"tokenized"`` (verified) or ``"char-estimate"`` (no /tokenize -- the
    fill may undershoot; the caller records the real fill from usage).
    """
    target = max(256, ctx - _OUTPUT_HEADROOM_TOKENS)
    body_chars = int(target * _CHARS_PER_TOKEN)
    prompt = _assemble_prompt(corpus, body_chars, depth)
    count = _count_tokens(base_url, model_name, prompt)
    if count is None:
        # /tokenize unavailable (e.g. SGLang): calibrate chars/token from a
        # short chat's usage.prompt_tokens and size from the measured ratio
        # -- far closer to the window than the static 3.5 estimate, with no
        # /tokenize dependency. Aims just under target so the request can't
        # be length-rejected; the HTTP-400 retry is the final backstop.
        cpt = _calibrate_chars_per_token(base_url, model_name, corpus)
        if cpt is None:
            return prompt, None, "char-estimate"
        safe_target = max(256, target - _TOKENIZE_TOL)
        prompt = _assemble_prompt(corpus, int(safe_target * cpt), depth)
        return prompt, None, "calibrated"

    for _ in range(_TOKENIZE_MAX_ITERS):
        # Done once we are within tolerance AND not over target (staying
        # under guarantees the request can't be length-rejected).
        if target - _TOKENIZE_TOL <= count <= target:
            break
        # Re-estimate chars/token from the live measurement and aim a
        # half-tolerance under target so we converge from below.
        measured_cpt = len(prompt) / max(1, count)
        desired = target - _TOKENIZE_TOL // 2
        body_chars = max(1, body_chars + int((desired - count) * measured_cpt))
        prompt = _assemble_prompt(corpus, body_chars, depth)
        new_count = _count_tokens(base_url, model_name, prompt)
        if new_count is None:
            break
        count = new_count

    # Final guard: if we still overshot target, shrink until <= target so
    # prompt + max_tokens stays under ctx.
    while count is not None and count > target and body_chars > 1:
        body_chars = int(body_chars * 0.97)
        prompt = _assemble_prompt(corpus, body_chars, depth)
        count = _count_tokens(base_url, model_name, prompt)

    return prompt, count, "tokenized"


def score_needle(response_text: str) -> float:
    """1.0 when the model echoed the exact needle code, else 0.0."""
    return 1.0 if _NEEDLE_CODE in (response_text or "") else 0.0


# ── Diagnostic: predicted logits buffer ──────────────────────────────────────

def _read_vocab_size(models_dir: str, model_name: str) -> int | None:
    """vocab_size from the on-disk config.json (top-level or text_config)."""
    cfg = Path(models_dir) / model_name / "config.json"
    if not cfg.is_file():
        return None
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if isinstance(data.get("vocab_size"), int):
        return data["vocab_size"]
    tc = data.get("text_config")
    if isinstance(tc, dict) and isinstance(tc.get("vocab_size"), int):
        return tc["vocab_size"]
    return None


def _flag_value(args_list: list[str], flag: str) -> str | None:
    for i, a in enumerate(args_list):
        if a == flag and i + 1 < len(args_list):
            return args_list[i + 1]
    return None


def predicted_logits_gb(
    models_dir: str, model_name: str, max_num_batched_tokens: int | None
) -> float | None:
    """Diagnostic estimate of the per-step softcap-logits buffer:
    ``max_num_batched_tokens x vocab x 4 bytes`` (fp32 logits). This is
    the allocation that OOMed DiffusionGemma on its first full prefill
    chunk. Returns None when vocab or the flag value is unknown rather
    than guessing.
    """
    vocab = _read_vocab_size(models_dir, model_name)
    if not vocab or not max_num_batched_tokens:
        return None
    return round(max_num_batched_tokens * vocab * 4 / 1e9, 3)


# ── Serving-failure classification ───────────────────────────────────────────

def _logs_show_oom(logs: str) -> str | None:
    low = (logs or "").lower()
    for marker in _OOM_MARKERS:
        if marker in low:
            return marker
    return None


def _detect_serving_failure(
    runtime: str,
    container: str,
    chat_resp: dict | None,
    request_error: str | None,
    logs: str,
) -> tuple[bool, str]:
    """Decide whether the load request crashed the engine. Returns
    ``(failed, reason)``. Order: transport error -> API error body ->
    container died -> OOM marker in logs.
    """
    if request_error:
        return True, f"request_error: {request_error}"
    if isinstance(chat_resp, dict) and chat_resp.get("error"):
        return True, f"api_error: {str(chat_resp.get('error'))[:200]}"
    state = container_state(runtime, container)
    if state not in ("running",):
        return True, f"container_state={state}"
    marker = _logs_show_oom(logs)
    if marker is not None and not (isinstance(chat_resp, dict) and chat_resp.get("choices")):
        return True, f"oom_marker: {marker}"
    return False, ""


# ── VRAM settle between cells ─────────────────────────────────────────────────

# GPU is considered idle below this; the probe host runs nothing else on
# the GPU during a sweep, so a clean teardown drops to ~0-200 MB.
_VRAM_IDLE_MB = 2048
_VRAM_SETTLE_TIMEOUT_S = 90.0
_VRAM_SETTLE_POLL_S = 2.0


def _wait_for_vram_settle(
    threshold_mb: int = _VRAM_IDLE_MB,
    timeout_s: float = _VRAM_SETTLE_TIMEOUT_S,
    poll_s: float = _VRAM_SETTLE_POLL_S,
) -> int:
    """Block until GPU memory drops below `threshold_mb` (prior container's
    CUDA context released) or `timeout_s` elapses. Returns the last
    reading. Prevents a heavy cell's lingering VRAM from OOMing the next
    cell's startup. No-op effect when the GPU is already idle.
    """
    deadline = time.time() + timeout_s
    used = gpu_memory_used_mb()
    while used >= threshold_mb and time.time() < deadline:
        time.sleep(poll_s)
        used = gpu_memory_used_mb()
    return used


# ── Per-cell load probe ──────────────────────────────────────────────────────

def load_probe_one_cell(
    spec: BackendSpec,
    *,
    runtime: str,
    image: str,
    container_name: str,
    probe_port: int,
    models_dir: str,
    model_name: str,
    requested_ctx: int,
    band_gb: int,
    host_vram_gb: float,
    model_size_gb: float,
    reasoning_parser: str | None,
    tool_parser: str | None,
    corpus: str,
    needle_depth: float,
) -> dict:
    """Launch the backend at ``requested_ctx`` exactly as the router
    would (same parsers, recovery flags, image override), send ONE
    near-full-context completion under a 0.1s VRAM sampler, and return
    the serving-* augmentation dict for the cell. Always tears the
    container down before returning.
    """
    started = time.time()
    host_frac = host_scaled_fraction(
        model_size_gb, band_gb, host_vram_gb, spec.reserve_gb,
    )
    plugin_volume, tool_plugin_path, reasoning_plugin_path = _resolve_plugins(
        spec, reasoning_parser, tool_parser,
    )
    cmd_args = spec.build_args(
        model_name, requested_ctx, host_frac,
        reasoning_parser=reasoning_parser, tool_parser=tool_parser,
        reasoning_parser_plugin=reasoning_plugin_path,
        tool_parser_plugin=tool_plugin_path,
        speculative_config=None,
    )
    env_vars = {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"}
    extra_flags, extra_env = recovery_overrides(model_name)
    if extra_flags:
        cmd_args = list(cmd_args) + extra_flags
    if extra_env:
        env_vars = {**env_vars, **extra_env}
    image_override = recovery_image(model_name)
    if image_override and image_override != image:
        print(f"    [recovery] {model_name}: load-probing on pinned image "
              f"{image_override}", file=sys.stderr)
        image = image_override
    extra_volumes = [plugin_volume] if plugin_volume is not None else []

    mnbt = _flag_value(cmd_args, "--max-num-batched-tokens")
    mnbt_int = int(mnbt) if mnbt and mnbt.isdigit() else None
    pred_logits = predicted_logits_gb(models_dir, model_name, mnbt_int)

    container_remove(runtime, container_name)
    # Wait for the prior cell's CUDA context to fully release before this
    # launch. A heavy full-context request leaves more VRAM to reclaim
    # than the fit probe's tiny chats, and `podman rm` returns before the
    # GPU driver frees it -- so back-to-back cells can OOM the next launch
    # at load time ("container exited before /health"). Block until the
    # GPU drops near-idle (or a short timeout) so each launch starts clean.
    _wait_for_vram_settle()
    try:
        container_run_detached(
            runtime, container_name, image, probe_port, models_dir, env_vars,
            spec.entrypoint, cmd_args, extra_volumes=extra_volumes,
        )
    except RuntimeError as e:
        return {
            "serving_ok": False,
            "serving_error": f"launch_failed: {e}",
            "serving_kind": "infra",
            "predicted_logits_gb": pred_logits,
            "serving_probed_at": now_iso(),
        }

    base_url = f"http://127.0.0.1:{probe_port}"
    health_deadline = time.time() + STARTUP_TIMEOUT
    healthy = False
    last_err: str | None = None
    while time.time() < health_deadline:
        try:
            http_get(f"{base_url}/health", timeout=5.0)
            healthy = True
            break
        except urllib.error.URLError as e:
            last_err = f"URLError: {e}"
        except (TimeoutError, OSError) as e:
            last_err = f"{type(e).__name__}: {e}"
        except json.JSONDecodeError:
            healthy = True
            break
        state = container_state(runtime, container_name)
        if state in ("exited", "stopped", "absent"):
            last_err = f"container {state} before /health"
            break
        time.sleep(HEALTH_POLL_INTERVAL)

    if not healthy:
        logs = container_logs(runtime, container_name)
        ev = classify_failure_logs(logs)
        container_remove(runtime, container_name)
        rec = {
            "serving_ok": False,
            "serving_error": last_err or "no /health",
            "serving_kind": ev.get("kind", "infra"),
            "predicted_logits_gb": pred_logits,
            "serving_probed_at": now_iso(),
            "serving_seconds": round(time.time() - started, 1),
        }
        # A container that loaded fine at a lower tier but died at a
        # higher one is usually leftover VRAM from the prior cell OOMing
        # this launch (see the _wait_for_vram_settle guard) -- capture the
        # log tail so the re-probe can confirm OOM vs a genuine load bug.
        if logs:
            rec["serving_log_excerpt"] = logs[-_LOG_EXCERPT_CHARS:]
        return rec

    # Baseline: model loaded + KV pool reserved, idle, before the big
    # request. transient = peak - baseline isolates the per-step buffers.
    baseline_gb = round(gpu_memory_used_mb() / 1024, 2)

    # Tokenizer-verified fill to ~99% of the window so the KV pool is
    # exercised near its true ceiling (where real OOMs happen), not the
    # ~88% a static char estimate reaches. max_tokens stays well under the
    # headroom so prompt+output can't exceed --max-model-len.
    prompt, fill_tokens, fill_method = build_full_window_prompt(
        base_url, model_name, corpus, requested_ctx, needle_depth)
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": _MAX_OUTPUT_TOKENS,
        "stream": False,
    }

    # Near-full-context prefill needs far longer than the fit probe's
    # 60s CHAT_TIMEOUT: a 254K-token prefill on a 26B model is minutes,
    # not seconds. Scale with ctx so small tiers still fail fast on a
    # genuine hang while large tiers get the time a real prefill needs
    # (~400 tok/s conservative prefill floor + 120s slack). Without this,
    # every 256K cell mis-classifies a still-progressing prefill as a
    # serving OOM.
    chat_timeout = max(CHAT_TIMEOUT, 120.0 + requested_ctx / 400.0)

    sampler = VramSampler(interval=0.1)
    request_error: str | None = None
    chat_resp: dict | None = None
    sampler.start()
    try:
        chat_resp = _post_chat(base_url, body, timeout=chat_timeout)
        # HTTP 400 = the char-estimated prompt overshot THIS tokenizer's
        # max-model-len budget (chars/token varies per tokenizer). That's
        # a prompt-sizing artifact, not a serving verdict -- trim to ~85%
        # of ctx and retry once. Still a near-full-context stress test.
        if (isinstance(chat_resp, dict)
                and "HTTP 400" in str(chat_resp.get("error", ""))):
            body["messages"][0]["content"] = build_haystack_prompt(
                corpus, int(requested_ctx * 0.85), depth=needle_depth)
            chat_resp = _post_chat(base_url, body, timeout=chat_timeout)
    except (urllib.error.URLError, TimeoutError, OSError,
            json.JSONDecodeError) as e:
        # The request error IS the OOM signal we are here to measure.
        request_error = f"{type(e).__name__}: {e}"
    finally:
        snap = sampler.stop()

    peak_gb = round(float(snap.get("peak_vram_gb", 0.0) or 0.0), 2)
    logs_tail = container_logs(runtime, container_name)
    failed, reason = _detect_serving_failure(
        runtime, container_name, chat_resp, request_error, logs_tail,
    )

    content = ""
    reasoning_text = ""
    input_tokens = output_tokens = 0
    if isinstance(chat_resp, dict):
        try:
            msg = ((chat_resp.get("choices") or [{}])[0].get("message") or {})
            content = msg.get("content") or ""
            # Reasoning models (R1-Distill, Qwen3 thinking) may emit the
            # answer in reasoning_content, or burn the output budget on
            # the <think> trace before reaching content. Score both so a
            # genuine recall in the reasoning isn't logged as needle=0.
            reasoning_text = msg.get("reasoning_content") or ""
        except (AttributeError, IndexError, TypeError):
            content = ""
        usage = chat_resp.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)

    needle = 0.0 if failed else max(
        score_needle(content), score_needle(reasoning_text))
    transient = round(max(0.0, peak_gb - baseline_gb), 2) if peak_gb else 0.0

    container_remove(runtime, container_name)

    rec: dict = {
        "serving_ok": not failed,
        "serving_peak_gb": peak_gb,
        "serving_baseline_gb": baseline_gb,
        "transient_gb": transient,
        "needle_score": needle,
        "serving_input_tokens": input_tokens,
        "serving_output_tokens": output_tokens,
        # How full the KV pool actually got: input_tokens / ctx, from the
        # engine's own usage count. ~0.99 means the window was genuinely
        # exercised; a low value (char-estimate fallback) means the OOM
        # ceiling at full fill was NOT tested -- surfaced for auditing.
        "serving_fill_ratio": (round(input_tokens / requested_ctx, 3)
                               if input_tokens else None),
        "serving_fill_method": fill_method,
        "predicted_logits_gb": pred_logits,
        "serving_chat_timeout_s": round(chat_timeout, 1),
        "serving_probed_at": now_iso(),
        "serving_seconds": round(time.time() - started, 1),
    }
    if failed:
        rec["serving_error"] = reason
        # Capture diagnostics so a failure is explainable without
        # relaunching: vLLM/SGLang put the real exception in the HTTP
        # error body, and the container-log tail shows OOM vs a transient
        # 500 vs a real engine bug. A single bare "HTTP 500" is too weak
        # to condemn a model's serveable ctx -- this is the evidence.
        if isinstance(chat_resp, dict) and chat_resp.get("body"):
            rec["serving_error_body"] = str(chat_resp["body"])[:500]
        if logs_tail:
            rec["serving_log_excerpt"] = logs_tail[-_LOG_EXCERPT_CHARS:]
    return rec


# ── Pass driver ──────────────────────────────────────────────────────────────

def run_load_probe_pass(spec: BackendSpec, args: argparse.Namespace) -> None:
    """Binary-search the max SERVING ctx per model and collapse to ONE cell.

    The FIT pass (run_probe_pass) left a single cell at the largest FITTING
    ctx. Here we binary-search the largest ctx that actually SERVES a
    near-full-context request (serving_ok), capped at that fit ctx and the
    model's position_limit, then keep exactly one cell at the winning ctx:

      - serves at the fit ctx  -> augment that cell in place with serving data.
      - serves only lower       -> MOVE the single cell down to the serving ctx.
      - serves nowhere          -> keep the fit cell, mark serving_ok=False, and
                                    record an `oom` ledger exclusion.

    Backend-agnostic: identical for vLLM and SGLang via their BackendSpec.
    """
    assert_no_active_backends(args.runtime)

    catalog_rows = load_catalog_hf_rows(args.catalog, spec.name)
    if args.repo:
        rx = re.compile(args.repo)
        catalog_rows = [r for r in catalog_rows if rx.search(r.get("repo") or "")]
    if not catalog_rows:
        sys.exit("error: no HF rows match — check --repo filter or models.yaml")

    models_dir = Path(args.models_dir)
    if not models_dir.is_dir():
        sys.exit(f"error: models dir not found: {models_dir}")

    cache = load_cache(args.cache)
    # Phase C: idempotent re-stamp of the image digest (the load pass may run
    # without a fresh fit pass; keep _meta current for the router's drift check).
    stamp_image_digest(
        cache, digest=image_digest_via_cli(args.runtime, args.image),
        image_ref=args.image)
    if not args.no_cache_write:
        save_cache(args.cache, cache)
    ledger = _load_ledger()
    corpus = load_corpus()
    band_gb = int(args.host_vram_gb)
    needle_depth = float(getattr(args, "needle_depth", 0.5))

    print(f"  load-prober:    probe-{spec.name} --load", file=sys.stderr)
    print(f"  cache:          {args.cache} ({len(cache)} entries)", file=sys.stderr)
    print(f"  vram band:      {vram_label(band_gb)} (host)", file=sys.stderr)
    print(f"  search:         binary max serving ctx on the 32K grid, "
          f"capped at the fit ctx", file=sys.stderr)
    print(f"  needle depth:   {needle_depth:.2f}", file=sys.stderr)
    print(f"  {spec.name} image:     {args.image}", file=sys.stderr)
    print(file=sys.stderr)

    probed = 0
    skipped = 0

    for row in catalog_rows:
        name = row.get("name") or ""
        repo = row.get("repo") or ""
        sha = (row.get("sha") or "").strip()
        if not (name and repo and sha):
            continue
        if not is_downloaded(name, models_dir):
            skipped += 1
            continue

        key = f"{repo}@{sha}"
        entry = cache.get(key)
        if not entry:
            print(f"  [skip] {name}: no fit-cache entry — run "
                  f"`make probe-{spec.name}` first", file=sys.stderr)
            skipped += 1
            continue
        band = (entry.get("probes") or {}).get(str(band_gb))
        if not band:
            print(f"  [skip] {name}: no fit cells at {vram_label(band_gb)} band",
                  file=sys.stderr)
            skipped += 1
            continue

        # The FIT pass leaves ONE fitting cell (the largest fitting ctx).
        fitting = {int(k): v for k, v in band.items()
                   if isinstance(v, dict) and v.get("fits")}
        if not fitting:
            print(f"  [skip] {name}: no fitting cell to serve-probe",
                  file=sys.stderr)
            skipped += 1
            continue
        max_fit_ctx = max(fitting)
        fit_cell = fitting[max_fit_ctx]

        # Cached: the single cell already carries a serving verdict.
        if fit_cell.get("serving_ok") is not None and not args.force:
            print(f"    {context_label(max_fit_ctx):>4s}  cached "
                  f"serving_ok={fit_cell.get('serving_ok')}", file=sys.stderr)
            skipped += 1
            continue

        row_parsers = (row.get("parsers") or {}).get(spec.name) or {}
        # Serve with what the router will serve with: the entry's verified
        # parsers (set by the fit prober), catalog hint as fallback.
        reasoning_parser = (entry.get("reasoning_parser")
                            or row_parsers.get("reasoning") or None)
        tool_parser = (entry.get("tool_parser")
                       or row_parsers.get("tool") or None)
        size_gb = model_size_gb_from_row(row)

        # Can't serve above what fits, nor above the model's as-delivered
        # ceiling. binary_search_max_ctx instant-fails ctxs above this cap.
        pos_limit = entry.get("position_limit") or effective_position_limit(
            name, models_dir)
        serving_cap = max_fit_ctx
        if isinstance(pos_limit, int) and pos_limit > 0:
            serving_cap = min(serving_cap, pos_limit)

        probed_cells: dict[int, dict] = {}

        def _serving_works(ctx: int) -> bool:
            print(f"  {name} @ {vram_label(band_gb)} "
                  f"ctx={context_label(ctx)}: load-probing ...", file=sys.stderr)
            rec = load_probe_one_cell(
                spec,
                runtime=args.runtime,
                image=args.image,
                container_name=args.container_name,
                probe_port=args.probe_port,
                models_dir=str(models_dir),
                model_name=name,
                requested_ctx=ctx,
                band_gb=band_gb,
                host_vram_gb=args.host_vram_gb,
                model_size_gb=size_gb,
                reasoning_parser=reasoning_parser,
                tool_parser=tool_parser,
                corpus=corpus,
                needle_depth=needle_depth,
            )
            probed_cells[ctx] = rec
            ok = rec.get("serving_ok")
            print(f"    {context_label(ctx):>4s}  serving_ok={ok} "
                  f"peak={rec.get('serving_peak_gb')}G "
                  f"needle={rec.get('needle_score')} "
                  f"{('[' + rec.get('serving_error', '') + ']') if not ok else ''}",
                  file=sys.stderr)
            return bool(ok)

        max_serving = binary_search_max_ctx(
            _serving_works, position_limit=serving_cap)

        if max_serving is not None:
            load_rec = probed_cells[max_serving]
            for stale in ("serving_error", "serving_kind"):
                fit_cell.pop(stale, None)
            if max_serving == max_fit_ctx:
                # Serves at the fit ctx: augment the existing cell in place.
                fit_cell.update(load_rec)
            else:
                # Serves only below the fit ctx: MOVE the single cell down,
                # carrying the fit metadata (capability/parsers) + serving data.
                moved = dict(fit_cell)
                moved["ctx"] = max_serving
                moved["actual_context"] = max_serving
                base = load_rec.get("serving_baseline_gb")
                if base:
                    moved["actual_vram_gb"] = base
                moved.update(load_rec)
                band.clear()
                band[str(max_serving)] = moved
                # The load pass owns max_context maintenance directly (it does
                # not call refresh_top_level_from_cells); set the shrunk ceiling
                # to the serving winner so the router advertises exactly what
                # served.
                entry["max_context"] = max_serving
            entry["last_load_probed_at"] = load_rec.get("serving_probed_at")
            # A model that now serves clears a stale prober-owned oom exclusion.
            if _ledger_reason(ledger, name, spec.name) == "oom":
                _ledger_clear(ledger, name, spec.name)
            probed += 1
        elif probed_cells:
            # Fits but serves nowhere (OOM at every probed ctx). Keep the fit
            # cell, stamp the failing serving verdict, exclude with `oom`.
            lo = min(probed_cells)
            fit_cell.update(probed_cells[lo])
            fit_cell["serving_ok"] = False
            entry["last_load_probed_at"] = now_iso()
            _ledger_record(
                ledger, name, spec.name, "oom",
                detail="serving OOM at every probed ctx (fits but cannot serve)",
                repo=repo, host_vram=args.host_vram_gb, sha=sha)
            probed += 1

        if not args.no_cache_write:
            save_cache(args.cache, cache)

    if not args.no_cache_write:
        _save_ledger(ledger, host_vram_gb=args.host_vram_gb)

    print(file=sys.stderr)
    print(f"  load-probe done: {probed} model(s) probed, {skipped} skipped",
          file=sys.stderr)
