#!/usr/bin/env python3
"""Probe downloaded Ollama models for reasoning support and per-tier VRAM use.

Capability is determined by runtime behavior, not catalog metadata. The
probe sends /api/chat with `think: true` and `options.num_ctx` set to
each requested tier and reads /api/ps for `size`/`size_vram`. Results
are stored per (digest, context) — there is no interpolation. If a
(model, context) pair isn't in the cache, downstream tools will treat
it as ineligible at that context.

Per-probe capability values:
    structured  – response has non-empty `message.thinking` field
    inline      – response has visible <think> markers in `message.content`
    unsupported – neither thinking field nor inline markers
    error       – probe request failed OR the load spilled to CPU/RAM

Cache schema (v2, JSON, digest-keyed):

    {
      "<digest>": {
        "schema_version": 2,
        "digest": "<short>",
        "aliases": ["name1", "name2", ...],
        "max_context": <int>,                      # /api/show ceiling
        "arch_family": "<str>",
        "param_size_label": "<str>",               # optional
        "quantization": "<str>",                   # optional
        "experts_total": <int>,                    # optional (MoE)
        "experts_used": <int>,                     # optional (MoE)
        "params_total": <int>,                     # optional
        "capability": "structured|inline|unsupported|error|unknown",
        "disable_verified": true|false|"error",    # only for `structured`
        "evidence_disable": {...},
        "probes": {
          "<ctx>": {
            "ctx": <int>,
            "size_bytes": <int>,
            "size_vram_bytes": <int>,
            "actual_total_gb": <float>,
            "actual_vram_gb": <float>,
            "fully_on_gpu": <bool>,
            "actual_context": <int>,
            "capability": "...",
            "evidence": {...},
            "probed_at": "<iso>",
            "probe_seconds": <float>
          },
          ...
        },
        "first_probed_at": "<iso>",
        "last_probed_at": "<iso>"
      },
      ...
    }

Probing is incremental: a probe entry is NEVER overwritten unless --force
or --force-ctx asks for it. Adding a new tier next run only fills gaps.
Aliases sharing a digest share their probes — we probe each digest once
per tier. Migration of v1 cache (name@digest keys, two-point
interpolation coefficients) is automatic on first run.

Usage:
    probe-ollama-reasoning.py [--cache PATH] [--prompt TEXT]
                              [--num-predict N] [--timeout SEC]
                              [--ollama-url URL]
                              [--probe-contexts 32K,64K,128K,256K]
                              [--force] [--force-ctx 64K,128K]
                              [MODEL ...]

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
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from _contexts import (  # noqa: E402  — local import after sys.path fix
    STANDARD_CONTEXTS,
    context_label,
    effective_targets,
    parse_context_list,
)

DEFAULT_CACHE = REPO_ROOT / "deploy" / ".ollama-reasoning-cache.json"
DEFAULT_PROMPT = "Answer with only the final number: What is 17 + 25?"
SCHEMA_VERSION = 2


# ── HTTP helpers ─────────────────────────────────────────────────────────────

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


# ── Live model lookups (/api/tags, /api/show, /api/ps) ───────────────────────

def list_models(ollama_url: str, timeout: float) -> list[dict]:
    """Return [{name, digest, modified_at, size}, ...] from /api/tags."""
    data = _http_get(f"{ollama_url}/api/tags", timeout)
    return [
        {
            "name": m.get("name", ""),
            "digest": (m.get("digest") or "")[:32],
            "modified_at": m.get("modified_at", ""),
            "size": m.get("size", 0),
        }
        for m in data.get("models", []) or []
    ]


def measure_arch(ollama_url: str, model_name: str, timeout: float) -> dict:
    """Read /api/show. Returns architecture / metadata fields or {error}."""
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
    if details.get("parameter_size"):
        out["param_size_label"] = details["parameter_size"]
    if details.get("quantization_level"):
        out["quantization"] = details["quantization_level"]
    # Suffix-match family-prefixed keys (gemma4.context_length,
    # qwen35moe.expert_count, …). The arch prefix varies per family.
    for k, v in mi.items():
        if k.endswith(".expert_count"):
            out["experts_total"] = int(v)
        elif k.endswith(".expert_used_count"):
            out["experts_used"] = int(v)
        elif k.endswith(".context_length"):
            out["max_context"] = int(v)
    return out


def measure_vram(
    ollama_url: str, model_name: str, model_digest: str, timeout: float
) -> dict:
    """Read /api/ps for the freshly-loaded model and report real memory use.

    Aliases sharing a digest may show under the canonical name in /api/ps,
    so we fall back to a short-digest match.
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


def chat_probe(
    ollama_url: str,
    model: str,
    prompt: str,
    think: bool,
    num_predict: int,
    timeout: float,
    num_ctx: int,
) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "think": think,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }
    try:
        return _http_post(f"{ollama_url}/api/chat", body, timeout)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    except json.JSONDecodeError as e:
        return {"error": f"non-JSON response: {e}"}


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
        return "inline", {"content_preview": content[:200]}
    return "unsupported", {"content_preview": content[:120]}


# ── Probe orchestration ──────────────────────────────────────────────────────

def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def canonical_alias(aliases: list[str]) -> str:
    """Pick the canonical name from a digest's alias list.

    Preference: explicit tag over `:latest`, longer (more descriptive)
    over shorter. Ollama treats aliases of one digest interchangeably,
    so the choice only affects log readability and downstream display.
    """
    if not aliases:
        return ""
    return sorted(aliases, key=lambda n: (n.endswith(":latest"), -len(n)))[0]


def _is_think_param_rejection(evidence: dict) -> bool:
    """Heuristic: HTTP 400 from /api/chat with `think:true` is Ollama's
    way of saying the model doesn't expose a thinking field. Other 4xx
    errors (e.g. 404 for an unknown model name) are not retried.
    """
    err = str((evidence or {}).get("error") or "")
    return err.startswith("HTTP 400")


def probe_one_context(
    ollama_url: str,
    model_name: str,
    digest: str,
    prompt: str,
    num_predict: int,
    timeout: float,
    ctx: int,
) -> dict:
    """Single-context probe (think=true). Records measurement + per-ctx capability.

    Falls back to `think=false` when the first call returns HTTP 400 —
    non-thinking models on recent Ollama versions reject the think flag
    instead of silently ignoring it. The fallback is recorded as
    `unsupported` (no thinking field by definition) and the original 400
    note is preserved under `think_param_rejected` so the rejection isn't
    silently lost.
    """
    started = time.time()
    resp = chat_probe(
        ollama_url, model_name, prompt, think=True,
        num_predict=num_predict, timeout=timeout, num_ctx=ctx,
    )
    cap, evidence = classify(resp)
    think_rejected_note: str | None = None
    if cap == "error" and _is_think_param_rejection(evidence):
        think_rejected_note = str(evidence.get("error") or "HTTP 400")
        resp = chat_probe(
            ollama_url, model_name, prompt, think=False,
            num_predict=num_predict, timeout=timeout, num_ctx=ctx,
        )
        cap, evidence = classify(resp)
        if cap != "error":
            evidence = {**evidence, "think_param_rejected": think_rejected_note}
    record: dict = {
        "ctx": ctx,
        "capability": cap,
        "evidence": evidence,
        "probed_at": now_iso(),
    }
    if think_rejected_note and cap != "error":
        record["think_param_rejected"] = True
    if cap != "error":
        vram = measure_vram(ollama_url, model_name, digest, timeout=10.0)
        if "error" in vram:
            record["capability"] = "error"
            record["evidence"] = {"error": f"vram: {vram['error']}"}
        else:
            record.update({
                "size_bytes": vram["size_bytes"],
                "size_vram_bytes": vram["size_vram_bytes"],
                "actual_total_gb": vram["actual_total_gb"],
                "actual_vram_gb": vram["actual_vram_gb"],
                "fully_on_gpu": vram["fully_on_gpu"],
                "actual_context": vram["actual_context"],
            })
            if not vram["fully_on_gpu"]:
                # Spill into CPU/RAM at this context — model is unusable here
                # even if reasoning behaviour was clean.
                record["evidence"] = {
                    "error": f"CPU/RAM spill at {ctx} context",
                    "actual_total_gb": vram["actual_total_gb"],
                    "actual_vram_gb": vram["actual_vram_gb"],
                    "original_capability": cap,
                }
                record["capability"] = "error"
    record["probe_seconds"] = round(time.time() - started, 2)
    return record


def smallest_clean_probe(entry: dict) -> dict | None:
    """Smallest-ctx probe whose capability is a non-error reasoning value."""
    probes = entry.get("probes") or {}
    candidates = sorted(
        (
            p for p in probes.values()
            if isinstance(p, dict)
            and p.get("capability") not in (None, "error")
        ),
        key=lambda p: int(p.get("ctx") or 0),
    )
    return candidates[0] if candidates else None


def update_canonical_capability(entry: dict) -> None:
    """Refresh top-level `capability` from the smallest clean probe.

    The smallest fitting tier is the most-trustworthy capability signal
    (highest chance of fitting on GPU, no spill). When every probe
    errored we fall back to "error"; with no probes at all, "unknown".
    """
    smallest = smallest_clean_probe(entry)
    if smallest:
        entry["capability"] = smallest.get("capability") or "unknown"
        return
    if entry.get("probes"):
        entry["capability"] = "error"
    else:
        entry.setdefault("capability", "unknown")


def maybe_probe_disable(
    ollama_url: str,
    canonical_name: str,
    entry: dict,
    prompt: str,
    num_predict: int,
    timeout: float,
) -> bool:
    """Run the negative think=false probe at the smallest clean tier.

    Runs only when capability is `structured` and disable_verified isn't
    already recorded. Returns True iff a probe was issued.
    """
    if entry.get("capability") != "structured":
        entry.pop("disable_verified", None)
        entry.pop("evidence_disable", None)
        return False
    if "disable_verified" in entry:
        return False
    smallest = smallest_clean_probe(entry)
    if not smallest:
        return False
    ctx = int(smallest.get("ctx") or 0)
    if ctx <= 0:
        return False
    resp = chat_probe(
        ollama_url, canonical_name, prompt, think=False,
        num_predict=num_predict, timeout=timeout, num_ctx=ctx,
    )
    if "error" in resp:
        entry["disable_verified"] = "error"
        entry["evidence_disable"] = {"error": resp["error"], "probed_ctx": ctx}
        return True
    msg = resp.get("message") or {}
    thinking = (msg.get("thinking") or "").strip()
    entry["disable_verified"] = not thinking
    entry["evidence_disable"] = {
        "thinking_chars": len(thinking),
        "content_preview": (msg.get("content") or "")[:80],
        "probed_ctx": ctx,
    }
    return True


# ── Cache I/O + migration ────────────────────────────────────────────────────

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


def is_v2_entry(entry: dict) -> bool:
    return isinstance(entry, dict) and entry.get("schema_version") == SCHEMA_VERSION


def _migrate_one(entry: dict, old: dict) -> None:
    """Merge one v1 record into a v2 entry without overriding existing fields."""
    for field in (
        "arch_family", "param_size_label", "quantization",
        "experts_total", "experts_used", "params_total", "max_context",
    ):
        if field not in entry and old.get(field) is not None:
            entry[field] = old[field]
    if "max_context" not in entry and old.get("model_max_context"):
        entry["max_context"] = old["model_max_context"]

    v1_cap = old.get("capability") or "unknown"
    original_cap = (
        (old.get("evidence_enable") or {}).get("original_capability") or v1_cap
    )
    if "capability" not in entry:
        # If v1 flipped to "error" because of spill, recover the pre-spill cap.
        entry["capability"] = original_cap if v1_cap == "error" else v1_cap
    if "disable_verified" not in entry and "disable_verified" in old:
        entry["disable_verified"] = old["disable_verified"]
    if "evidence_disable" not in entry and "evidence_disable" in old:
        entry["evidence_disable"] = old["evidence_disable"]

    for slot in ("actual_low", "actual_high"):
        point = old.get(slot)
        if not isinstance(point, dict):
            continue
        ctx = int(point.get("actual_context") or 0)
        if ctx <= 0:
            continue
        ctx_key = str(ctx)
        if ctx_key in entry["probes"]:
            continue
        fully_on_gpu = bool(point.get("fully_on_gpu", False))
        if not fully_on_gpu:
            cap_for_probe = "error"
            evidence: dict = {
                "error": f"CPU/RAM spill at {ctx} context (migrated)",
                "actual_total_gb": point.get("actual_total_gb"),
                "actual_vram_gb": point.get("actual_vram_gb"),
                "original_capability": original_cap,
            }
        elif slot == "actual_low":
            cap_for_probe = original_cap
            evidence = {}
        else:
            cap_for_probe = v1_cap if v1_cap != "error" else original_cap
            evidence = {}
        entry["probes"][ctx_key] = {
            "ctx": ctx,
            "size_bytes": int(point.get("size_bytes") or 0),
            "size_vram_bytes": int(point.get("size_vram_bytes") or 0),
            "actual_total_gb": float(point.get("actual_total_gb") or 0),
            "actual_vram_gb": float(point.get("actual_vram_gb") or 0),
            "fully_on_gpu": fully_on_gpu,
            "actual_context": ctx,
            "capability": cap_for_probe,
            "evidence": evidence,
            "probed_at": old.get("probed_at") or now_iso(),
            "migrated_from_v1": True,
        }

    if "first_probed_at" not in entry and old.get("probed_at"):
        entry["first_probed_at"] = old["probed_at"]
    if old.get("probed_at"):
        prior = entry.get("last_probed_at") or ""
        if old["probed_at"] > prior:
            entry["last_probed_at"] = old["probed_at"]


def migrate_v1_to_v2(old_cache: dict) -> dict:
    """Convert legacy `name@digest` entries into digest-keyed v2 records.

    Two passes: copy v2 entries verbatim first (they're authoritative),
    then merge v1 records into the same digest entries without
    overriding any v2 field. Probe entries are keyed by actual_context.
    Interpolation coefficients are dropped.
    """
    new_cache: dict = {}
    for key, old in old_cache.items():
        if is_v2_entry(old):
            digest = old.get("digest") or key
            new_cache[digest] = old

    for key, old in old_cache.items():
        if not isinstance(old, dict) or is_v2_entry(old):
            continue
        digest = old.get("digest") or (key.split("@", 1)[1] if "@" in key else "")
        if not digest:
            continue
        name = old.get("name") or (key.split("@", 1)[0] if "@" in key else key)
        entry = new_cache.setdefault(digest, {
            "schema_version": SCHEMA_VERSION,
            "digest": digest,
            "aliases": [],
            "probes": {},
        })
        if name and name not in entry["aliases"]:
            entry["aliases"].append(name)
        _migrate_one(entry, old)

    for entry in new_cache.values():
        entry.setdefault("probes", {})
        entry.setdefault("aliases", [])
        update_canonical_capability(entry)
    return new_cache


def ensure_v2(cache: dict) -> tuple[dict, int]:
    """Return (v2_cache, migrated_count). Idempotent on already-v2 caches."""
    if not cache:
        return {}, 0
    if all(is_v2_entry(v) for v in cache.values()):
        return cache, 0
    migrated_legacy = sum(1 for v in cache.values() if not is_v2_entry(v))
    return migrate_v1_to_v2(cache), migrated_legacy


# ── Main loop ────────────────────────────────────────────────────────────────

def group_by_digest(models: list[dict]) -> dict[str, list[dict]]:
    by_digest: dict[str, list[dict]] = {}
    for m in models:
        d = m.get("digest")
        if not d:
            continue
        by_digest.setdefault(d, []).append(m)
    return by_digest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--cache", type=Path, default=DEFAULT_CACHE,
        help=f"cache file (default {DEFAULT_CACHE.relative_to(REPO_ROOT)})",
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
        help="ollama base URL",
    )
    ap.add_argument(
        "--probe-contexts",
        default=os.environ.get(
            "PROBE_CONTEXTS", ",".join(str(c) for c in STANDARD_CONTEXTS)
        ),
        help=(
            "comma-separated context tiers to probe per model "
            f"(default {','.join(str(c) for c in STANDARD_CONTEXTS)})"
        ),
    )
    ap.add_argument(
        "--force", action="store_true",
        help="re-probe every requested tier (ignores cached probes)",
    )
    ap.add_argument(
        "--force-ctx", default="",
        help="comma-separated tier(s) to re-probe; other tiers stay cached",
    )
    ap.add_argument(
        "models", nargs="*",
        help="optional model names to probe; defaults to every downloaded model",
    )
    args = ap.parse_args()

    tiers = parse_context_list(args.probe_contexts)
    if not tiers:
        sys.exit("error: --probe-contexts produced an empty list")
    force_raw: list[int] = []
    if args.force:
        force_raw = list(tiers)
    elif args.force_ctx:
        force_raw = parse_context_list(args.force_ctx)

    print(f"  probe target:   {args.ollama_url}", file=sys.stderr)
    print(
        f"  probe tiers:    {','.join(context_label(t) for t in tiers)}",
        file=sys.stderr,
    )
    if force_raw:
        print(
            f"  force re-probe: {','.join(context_label(t) for t in sorted(set(force_raw)))}",
            file=sys.stderr,
        )

    raw_cache = load_cache(args.cache)
    cache, migrated = ensure_v2(raw_cache)
    note = f"migrated {migrated} from v1" if migrated else ""
    print(
        f"  cache:          {args.cache} ({len(cache)} digest entries"
        + (f", {note}" if note else "")
        + ")",
        file=sys.stderr,
    )
    if migrated:
        save_cache(args.cache, cache)

    try:
        all_models = list_models(args.ollama_url, args.timeout)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        sys.exit(f"error: cannot list ollama models: {e}")
    if not all_models:
        sys.exit("error: ollama returned no models — is anything downloaded?")

    target_models = all_models
    if args.models:
        requested = set(args.models)
        target_models = [m for m in all_models if m["name"] in requested]
        found = {m["name"] for m in target_models}
        missing = sorted(requested - found)
        if missing:
            sys.exit(
                "error: requested model(s) not found on disk: "
                + ", ".join(missing)
            )

    target_by_digest = group_by_digest(target_models)
    all_by_digest = group_by_digest(all_models)

    print(
        f"  {len(all_models)} tags on disk, {len(all_by_digest)} unique digests"
        + (
            f"; targeting {len(target_models)} tag(s) "
            f"({len(target_by_digest)} digest(s))"
            if args.models else ""
        ),
        file=sys.stderr,
    )
    print(file=sys.stderr)

    # Reconcile aliases for every known digest, not only target ones —
    # otherwise an alias that disappeared from disk would linger forever
    # in entries we don't touch this run.
    cleaned_aliases = 0
    for digest, tags in all_by_digest.items():
        entry = cache.get(digest)
        if entry is None:
            continue
        live_aliases = sorted({t["name"] for t in tags})
        prev = sorted(entry.get("aliases") or [])
        if prev != live_aliases:
            entry["aliases"] = live_aliases
            cleaned_aliases += 1

    header_printed = False

    def maybe_header() -> None:
        nonlocal header_printed
        if header_printed:
            return
        print(
            f"  {'CANONICAL':<40s} {'CAP':<11s} "
            f"{'TIERS':<28s} ACTION",
            file=sys.stderr,
        )
        print(f"  {'-' * 100}", file=sys.stderr)
        header_printed = True

    fresh_probes = 0
    fully_cached = 0
    arch_failures = 0

    for digest, tags in sorted(target_by_digest.items()):
        live_aliases = sorted({t["name"] for t in tags})
        entry = cache.get(digest)
        if entry is None:
            entry = {
                "schema_version": SCHEMA_VERSION,
                "digest": digest,
                "aliases": live_aliases,
                "probes": {},
            }
            cache[digest] = entry
        else:
            existing = set(entry.get("aliases") or [])
            existing.update(live_aliases)
            entry["aliases"] = sorted(existing)

        canonical = canonical_alias(live_aliases)

        # Step 0: cheap arch lookup. Skipped when max_context already known.
        if not entry.get("max_context") or "arch_family" not in entry:
            arch = measure_arch(args.ollama_url, canonical, timeout=10.0)
            if "error" in arch:
                maybe_header()
                print(
                    f"  {canonical:<40s} {'error':<11s} "
                    f"{'-':<28s} arch lookup failed: {arch['error']}",
                    file=sys.stderr,
                )
                entry["arch_error"] = arch["error"]
                arch_failures += 1
                save_cache(args.cache, cache)
                continue
            for k, v in arch.items():
                if v is not None:
                    entry[k] = v

        max_ctx = int(entry.get("max_context") or 0)
        if max_ctx <= 0:
            maybe_header()
            print(
                f"  {canonical:<40s} {'error':<11s} "
                f"{'-':<28s} no max_context from /api/show",
                file=sys.stderr,
            )
            arch_failures += 1
            continue

        targets = effective_targets(tiers, max_ctx)
        if not targets:
            continue

        force_set = set(effective_targets(force_raw, max_ctx))
        existing_probes = entry.get("probes") or {}
        # Auto-retry cached HTTP 400 errors — they're the legacy result of
        # probing non-thinking models with `think:true` before the fallback
        # path existed. Other errors (timeouts, 5xx) stay cached so transient
        # backend issues don't burn probe budget on every run.
        missing = [
            t for t in targets
            if str(t) not in existing_probes
            or t in force_set
            or _is_think_param_rejection(
                (existing_probes.get(str(t)) or {}).get("evidence") or {}
            )
            and (existing_probes.get(str(t)) or {}).get("capability") == "error"
        ]
        if not missing:
            fully_cached += 1
            continue

        ctxs_done: list[int] = []
        for ctx in missing:
            rec = probe_one_context(
                args.ollama_url, canonical, digest, args.prompt,
                args.num_predict, args.timeout, ctx,
            )
            entry.setdefault("probes", {})[str(ctx)] = rec
            entry.setdefault("first_probed_at", rec["probed_at"])
            entry["last_probed_at"] = rec["probed_at"]
            ctxs_done.append(ctx)
            fresh_probes += 1
            save_cache(args.cache, cache)

        update_canonical_capability(entry)
        # disable_verified: clear stale value on force, then re-probe.
        if force_set:
            entry.pop("disable_verified", None)
            entry.pop("evidence_disable", None)
        ran_disable = maybe_probe_disable(
            args.ollama_url, canonical, entry,
            args.prompt, args.num_predict, args.timeout,
        )

        cap = entry.get("capability") or "unknown"
        marker = ""
        if cap == "structured":
            dv = entry.get("disable_verified")
            if dv is True:
                marker = " (disable verified)"
            elif dv is False:
                marker = " (disable not honored)"
            elif dv == "error":
                marker = " (disable probe failed)"
        if any(t > max_ctx for t in tiers):
            marker += f" (capped at {context_label(max_ctx)})"
        if ran_disable:
            marker += " +disable"
        ctx_summary = ",".join(context_label(c) for c in ctxs_done) or "-"
        maybe_header()
        print(
            f"  {canonical:<40s} {cap:<11s} "
            f"{ctx_summary:<28s} probed{marker}",
            file=sys.stderr,
        )
        save_cache(args.cache, cache)

    # Drop digest entries whose tags are all gone from disk.
    live_digests = set(all_by_digest)
    stale = [d for d in cache if d not in live_digests]
    for d in stale:
        del cache[d]
    if stale or cleaned_aliases:
        save_cache(args.cache, cache)

    print(file=sys.stderr)
    by_cap: dict[str, int] = {}
    for entry in cache.values():
        c = entry.get("capability") or "unknown"
        by_cap[c] = by_cap.get(c, 0) + 1
    summary = "  ".join(f"{c}={n}" for c, n in sorted(by_cap.items()))
    print(
        f"  done: {fresh_probes} probe(s) across {len(target_by_digest)} digest(s); "
        f"{fully_cached} digest(s) fully cached; "
        f"{cleaned_aliases} alias list(s) reconciled; "
        f"{arch_failures} arch lookup failure(s); "
        f"{len(stale)} stale digest(s) removed",
        file=sys.stderr,
    )
    print(f"  capability counts: {summary}", file=sys.stderr)


if __name__ == "__main__":
    main()
