#!/usr/bin/env python3
"""Host-local model exclusion ledger (deploy/.model-status.json).

Records every "do not bother with this model on this host" verdict so unfit
models are not re-downloaded, re-probed, or re-listed as "not on disk". The
catalog (deploy/models.yaml) stays the host-agnostic superset; this ledger
is the host-LOCAL negative overlay. Gitignored, like the probe caches.

Keyed by catalog `name` + `backend` -- sha-stable, so a re-quant of the same
model keeps its verdict, unlike the repo@sha probe-cache key. Reasons:
  too_big / too_small  -- outside the host VRAM window (vram-dependent)
  unsupported_arch     -- the engine can't load the architecture (terminal,
                          vram- and sha-independent)
  oom                  -- ran out of memory at serve time (weight-specific;
                          re-checked on a new sha)
  retired              -- fit fine, but lost the bake-off: superseded by a
                          better model, or dropped on context/speed grounds.
                          Carries `superseded_by`. Vram-DEPENDENT: a bigger
                          GPU genuinely changes the comparison, so the
                          verdict is re-derived on a VRAM change.
  manual               -- operator-pinned

`retired` exists because every other reason answers "this model cannot run
here", and none of them could express "it ran, we just chose something
else". That gap had a measurable cost: of 11 vLLM models probed-fitting
whose weights were later removed, only the 5 that had been benched carried
any ledger entry. The 6 removed before they were benched left no trace at
all, so nothing recorded whether they were rejected or merely forgotten.
See `record_retirement` below and select-models.py's `delete()`.

Sha-stability and vram-stability follow plan decision 2: only too_big /
too_small / unsupported_arch / retired / manual survive a re-quant; only
unsupported_arch / manual survive a GPU-VRAM change.

See docs/plans/model-lifecycle-ledger.md (Phase 3). Read-only inspection /
manual clear:  python3 scripts/_model_status.py [--clear NAME[::BACKEND]]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = REPO_ROOT / "deploy" / ".model-status.json"
SCHEMA_VERSION = 1

VALID_REASONS = ("too_big", "too_small", "unsupported_arch", "oom",
                 "retired", "manual", "bench_dropped", "bench_failed")
# Bench verdicts. These say how a model PERFORMED, not whether it loads,
# so they must never gate download or probe: a model dropped for leaking
# its system prompt is still perfectly downloadable and probeable, and
# re-probing it is exactly what an operator does after a re-quant.
#
# They are therefore queried through is_bench_excluded(), NEVER through
# is_excluded(). Keeping them out of both reason tuples below makes
# is_excluded() fall through to its unknown-reason branch and fail open,
# which is the required behaviour.
#
# This has to be explicit. docs/plans/card-derived-hints-and-bench-sync.md
# reasoned that leaving is_excluded()'s allowlist untouched would make the
# new reasons fail open "by construction" -- true when that allowlist was
# a hand-written literal, but it is now DERIVED from VALID_REASONS (the
# fix for `retired` silently failing open). Under the derived form, simply
# adding a reason opts it INTO gating. The subtraction below is what
# actually delivers the intended behaviour.
_BENCH_REASONS = ("bench_dropped", "bench_failed")
# Reasons that do NOT depend on the exact weights (survive a re-quant).
# `retired` is here because a re-quant of a model already rejected on
# context or speed grounds does not change that judgement.
_SHA_STABLE_REASONS = ("too_big", "too_small", "unsupported_arch",
                       "retired", "manual")
# Reasons that do NOT depend on the GPU VRAM (survive a GPU change).
# `retired` is deliberately NOT here: it is a judgement relative to the
# rest of the fleet at a given VRAM budget, and most retirements on this
# host cite a context ceiling. A bigger GPU genuinely reopens that.
_VRAM_INDEPENDENT_REASONS = ("unsupported_arch", "manual")

# Everything else is vram-dependent: the verdict is only trusted when it
# was judged at this GPU's VRAM. DERIVED rather than hand-listed -- when
# `retired` was added, is_excluded still carried a literal
# ("too_big", "too_small", "oom") tuple, so retirements fell through to
# the unknown-reason branch and silently failed open. Deriving it means a
# new reason is vram-dependent by default, which is the safe direction.
_VRAM_DEPENDENT_REASONS = tuple(
    r for r in VALID_REASONS
    if r not in _VRAM_INDEPENDENT_REASONS and r not in _BENCH_REASONS
)

# How many recorded failures before a `bench_failed` verdict is believed.
# One failure is usually a transient -- a cold-start timeout, a router
# recreate landing mid-request. Excluding on the first would quietly
# shrink the leaderboard on infrastructure noise, which is the failure
# mode this whole ledger exists to make visible rather than silent.
BENCH_FAILED_ATTEMPTS_BEFORE_EXCLUDE = 2


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _empty() -> dict:
    return {"_meta": {"schema_version": SCHEMA_VERSION}, "models": {}}


def load_ledger(path: Path = LEDGER_PATH) -> dict:
    """Read the ledger; missing or malformed -> empty (fail open)."""
    p = Path(path)
    if not p.is_file():
        return _empty()
    try:
        data = json.loads(p.read_text())
    except Exception:
        return _empty()
    if not isinstance(data, dict):
        return _empty()
    data.setdefault("_meta", {"schema_version": SCHEMA_VERSION})
    data.setdefault("models", {})
    return data


def save_ledger(ledger: dict, path: Path = LEDGER_PATH, *,
                host_vram_gb: float | None = None) -> None:
    """Write the ledger atomically (tmp file + os.replace).

    Same idiom as the probe caches (_probe_core.save_cache): POSIX
    guarantees os.replace is atomic on the same filesystem, so a crash
    mid-write leaves either the old ledger intact or the new one fully
    written -- never a truncated JSON document that load_ledger would
    silently degrade to "nothing is excluded", re-opening every model
    this host already rejected.
    """
    meta = ledger.setdefault("_meta", {})
    meta["schema_version"] = SCHEMA_VERSION
    if host_vram_gb is not None:
        meta["host_vram_gb"] = float(host_vram_gb)
    meta["updated_at"] = _now()
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True))
    os.replace(tmp, p)


def record_exclusion(ledger: dict, name: str, backend: str, reason: str, *,
                     detail: str = "", repo: str | None = None,
                     host_vram: float | None = None, ctx: int | None = None,
                     sha: str | None = None,
                     superseded_by: str | None = None) -> None:
    """Mark (name, backend) excluded for `reason`. Idempotent (overwrites).

    `superseded_by` names the model that displaced this one. It is only
    meaningful for `retired` and is stored as a structured field rather
    than buried in `detail`, so "what replaced X" and "what did Y
    displace" are both answerable without parsing free text.
    """
    if reason not in VALID_REASONS:
        raise ValueError(f"invalid reason {reason!r}; expected {VALID_REASONS}")
    m = ledger.setdefault("models", {}).setdefault(name, {})
    if repo:
        m["repo"] = repo
    entry = {
        "status": "excluded",
        "reason": reason,
        "detail": detail,
        "judged_at": {
            "host_vram_gb": float(host_vram) if host_vram is not None else None,
            "ctx": int(ctx) if ctx is not None else None,
        },
        "last_sha": sha,
        "sha_stable": reason in _SHA_STABLE_REASONS,
        "updated_at": _now(),
    }
    if superseded_by:
        entry["superseded_by"] = superseded_by
    m.setdefault("backends", {})[backend] = entry


def record_retirement(ledger: dict, name: str, backend: str, *,
                      detail: str = "", superseded_by: str | None = None,
                      repo: str | None = None, host_vram: float | None = None,
                      sha: str | None = None) -> None:
    """Record that (name, backend) ran fine but was deliberately dropped.

    Called on the weight-removal path so a model cannot leave the fleet
    without leaving a reason behind. Never overwrites a stronger verdict:
    if the model is already excluded for a hard reason (oom,
    unsupported_arch, too_big, too_small) that verdict is the real story
    and retirement would obscure it.
    """
    existing = _backend_entry(ledger, name, backend)
    if existing is not None and existing.get("reason") in (
            "oom", "unsupported_arch", "too_big", "too_small"):
        return
    record_exclusion(ledger, name, backend, "retired", detail=detail,
                     repo=repo, host_vram=host_vram, sha=sha,
                     superseded_by=superseded_by)


def record_bench_verdict(ledger: dict, name: str, backend: str, reason: str, *,
                         detail: str = "", ctx: int | None = None,
                         repo: str | None = None,
                         sha: str | None = None) -> dict:
    """Record a bench-time verdict for (name, backend).

    Separate from record_exclusion because the two answer different
    questions and must not be confused at the call site: this one says
    "benched badly", not "will not load".

    `bench_failed` accumulates an attempt counter rather than excluding
    outright -- see BENCH_FAILED_ATTEMPTS_BEFORE_EXCLUDE. `bench_dropped`
    is a quality judgement and applies immediately.

    Returns the stored entry so a caller can report the attempt count.
    """
    if reason not in _BENCH_REASONS:
        raise ValueError(
            f"invalid bench reason {reason!r}; expected {_BENCH_REASONS}")

    prev = _backend_entry(ledger, name, backend)
    attempts = 1
    if prev is not None and prev.get("reason") == reason:
        try:
            attempts = int(prev.get("attempts", 1)) + 1
        except (TypeError, ValueError):
            attempts = 1

    m = ledger.setdefault("models", {}).setdefault(name, {})
    if repo:
        m["repo"] = repo
    entry = {
        "status": "excluded",
        "reason": reason,
        "detail": detail,
        "judged_at": {
            # Deliberately no host_vram_gb: a bench verdict is about model
            # quality, which a different GPU does not change. Recording one
            # would invite a reader to re-derive it on a GPU swap.
            "host_vram_gb": None,
            "ctx": int(ctx) if ctx is not None else None,
        },
        "last_sha": sha,
        "sha_stable": False,   # a re-quant genuinely re-opens the question
        "attempts": attempts,
        "updated_at": _now(),
    }
    m.setdefault("backends", {})[backend] = entry
    return entry


def is_bench_excluded(ledger: dict, name: str, backend: str, *,
                      ctx: int | None = None, sha: str | None = None) -> bool:
    """True iff a BENCH verdict currently disqualifies (name, backend).

    Deliberately a separate query from is_excluded(): bench verdicts must
    not stop a model being downloaded or probed.

    Stability rules:
      - VRAM-independent. A leak or a failing score is a property of the
        model, not of the card it ran on.
      - sha-dependent. A re-quant is a different artefact and re-opens the
        question.
      - ctx-scoped, judged-ctx-and-above. Long-context behaviour is where
        these failures concentrate (a model that leaks at 128K may be
        clean at 32K), so a verdict at 131072 says nothing about 32768.
        A verdict with no recorded ctx applies everywhere -- it is the
        only safe reading of "we do not know where this was judged".
      - `bench_failed` needs repetition. A single failure is usually
        infrastructure noise.

    Fails OPEN on any malformed row, like is_excluded().
    """
    e = _backend_entry(ledger, name, backend)
    if e is None:
        return False
    reason = e.get("reason")
    if reason not in _BENCH_REASONS:
        return False

    last = e.get("last_sha")
    if sha and last and sha != last:
        return False                       # re-quant -> re-check

    if reason == "bench_failed":
        try:
            attempts = int(e.get("attempts", 1))
        except (TypeError, ValueError):
            attempts = 1
        if attempts < BENCH_FAILED_ATTEMPTS_BEFORE_EXCLUDE:
            return False

    ja = e.get("judged_at")
    judged_ctx = ja.get("ctx") if isinstance(ja, dict) else None
    if judged_ctx is None or ctx is None:
        return True                        # unscoped verdict applies everywhere
    try:
        return int(ctx) >= int(judged_ctx)
    except (TypeError, ValueError):
        return True


def _backend_entry(ledger: dict, name: str, backend: str) -> dict | None:
    """The excluded backend entry, or None. Fails open on any malformed
    (JSON-valid but wrong-shape) row -- a corrupt ledger excludes nothing."""
    models = ledger.get("models")
    m = models.get(name) if isinstance(models, dict) else None
    if not isinstance(m, dict):
        return None
    backends = m.get("backends")
    e = backends.get(backend) if isinstance(backends, dict) else None
    return e if isinstance(e, dict) and e.get("status") == "excluded" else None


# Reasons whose basis is VRAM arithmetic rather than an engine-specific
# capability or an operator judgement. Only these can carry from one
# backend to another.
_VRAM_DERIVED_REASONS = ("too_big", "too_small", "oom")

# backend -> backends whose VRAM-derived verdicts also apply here.
#
# ONLY vllm -> sglang, and only in that direction. SGLang needs strictly
# MORE VRAM than vLLM for the same (model, ctx) on this fleet:
#
#   * reserve:  SGLANG_RESERVE_GB = 3.0  vs  VLLM_RESERVE_GB = 2.0
#   * KV dtype: the vLLM prober launches --kv-cache-dtype fp8 (1 byte per
#     element); the SGLang prober emits no such flag, so it runs the
#     engine default -- unquantized, 2 bytes. Twice the KV.
#
# So "vLLM could not fit this" implies "SGLang cannot fit it either", and
# probing it on SGLang burns a cold start to rediscover a known answer.
# The converse is NOT true and must never be added: SGLang failing says
# nothing about vLLM, which has more room.
#
# Deliberately excluded from this mechanism:
#
#   unsupported_arch -- engine-specific by definition. vLLM and SGLang do
#     not support the same architecture set, which is the entire reason
#     the ledger is keyed per backend.
#   manual -- an operator verdict with no recorded physics. Propagating it
#     would be actively wrong here: Qwen3-8B-NVFP4 and Qwen3-14B-NVFP4
#     both carry a vLLM `manual` exclusion AND fit on SGLang, where they
#     are currently served.
#   retired / bench_* -- not probe gates.
_VRAM_IMPLIED_BY: dict[str, tuple[str, ...]] = {"sglang": ("vllm",)}


def implied_vram_exclusion(ledger: dict, name: str, backend: str, *,
                           host_vram: float | None,
                           sha: str | None = None) -> tuple[str, str] | None:
    """A VRAM-derived exclusion on a roomier backend that applies here.

    Returns ``(source_backend, reason)`` or None. Used to skip a probe
    that would only rediscover a VRAM verdict already measured on a
    backend with strictly more headroom -- see _VRAM_IMPLIED_BY for why
    the implication is one-way.

    Reuses `is_excluded` for the source backend so the stability rules
    (re-derive on a VRAM change, re-check `oom` on a new sha) apply
    identically; an implied exclusion is never more durable than the
    verdict it is derived from.
    """
    for source in _VRAM_IMPLIED_BY.get(backend, ()):
        e = _backend_entry(ledger, name, source)
        if e is None:
            continue
        reason = e.get("reason")
        if reason not in _VRAM_DERIVED_REASONS:
            continue
        if is_excluded(ledger, name, source, host_vram=host_vram, sha=sha):
            return source, str(reason)
    return None


def is_excluded(ledger: dict, name: str, backend: str, *,
                host_vram: float | None, sha: str | None = None) -> bool:
    """True iff (name, backend) is excluded AND that verdict still applies at
    the current host VRAM / model sha (stability rules per decision 2).

    Fails OPEN: any malformed row degrades to 'not excluded' rather than
    raising, so a hand-edited / corrupt ledger never aborts a probe run."""
    e = _backend_entry(ledger, name, backend)
    if e is None:
        return False
    reason = e.get("reason")
    if reason in _VRAM_INDEPENDENT_REASONS:        # unsupported_arch, manual
        return True
    if reason not in _VRAM_DEPENDENT_REASONS:
        return False                              # unknown reason -> fail open
    # vram-dependent: trust the verdict ONLY when judged at this GPU's VRAM.
    # A missing/corrupt/different judged_vram means we cannot confirm it still
    # applies -> re-derive (fail open) rather than exclude.
    if host_vram is not None:
        ja = e.get("judged_at")
        judged_vram = ja.get("host_vram_gb") if isinstance(ja, dict) else None
        try:
            if judged_vram is None or float(judged_vram) != float(host_vram):
                return False
        except (TypeError, ValueError):
            return False
    if reason == "oom":
        last = e.get("last_sha")
        if sha and last and sha != last:
            return False  # re-quant -> re-check
    return True


def exclusion_reason(ledger: dict, name: str, backend: str) -> str | None:
    e = _backend_entry(ledger, name, backend)
    return e.get("reason") if e else None


def clear(ledger: dict, name: str, backend: str | None = None) -> bool:
    """Remove an exclusion. backend=None clears all backends for the name."""
    models = ledger.get("models") or {}
    if name not in models:
        return False
    if backend is None:
        del models[name]
        return True
    backends = models[name].get("backends", {})
    if backend in backends:
        del backends[backend]
        if not backends:
            del models[name]
        return True
    return False


def prune_to_catalog(ledger: dict, catalog_names: set[str]) -> int:
    """Drop ledger rows for models no longer in the catalog. Returns count."""
    models = ledger.get("models") or {}
    drop = [n for n in models if n not in catalog_names]
    for n in drop:
        del models[n]
    return len(drop)


def iter_exclusions(ledger: dict):
    """Yield (name, backend, entry) for every recorded exclusion. Skips
    malformed rows so `make model-status` never crashes on a corrupt file."""
    models = ledger.get("models")
    for name, m in (models.items() if isinstance(models, dict) else []):
        backends = m.get("backends") if isinstance(m, dict) else None
        for backend, e in (backends.items() if isinstance(backends, dict) else []):
            if isinstance(e, dict) and e.get("status") == "excluded":
                yield name, backend, e


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect / edit the model "
                                             "exclusion ledger (read-only by "
                                             "default).")
    ap.add_argument("--clear", metavar="NAME[::BACKEND]",
                    help="Remove an exclusion (all backends if ::BACKEND "
                         "omitted), then exit.")
    ap.add_argument("--retire", metavar="NAME::BACKEND",
                    help="Record that a model ran fine but was dropped "
                         "(superseded, or rejected on context/speed "
                         "grounds), then exit. Requires --reason.")
    ap.add_argument("--reason", default="",
                    help="Why it was retired. Required with --retire.")
    ap.add_argument("--superseded-by", metavar="MODEL", default=None,
                    help="Optional: the model that displaced this one.")
    ap.add_argument("--ledger", default=str(LEDGER_PATH))
    args = ap.parse_args(argv)
    path = Path(args.ledger)
    ledger = load_ledger(path)
    if args.retire:
        name, sep, backend = args.retire.partition("::")
        if not sep or not backend:
            print("--retire needs NAME::BACKEND (the verdict is per-backend: "
                  "a model can be retired from SGLang and kept on vLLM)",
                  file=sys.stderr)
            return 1
        if not args.reason:
            print("--retire requires --reason; recording a retirement with "
                  "no reason recreates the gap this exists to close",
                  file=sys.stderr)
            return 1
        record_retirement(ledger, name, backend, detail=args.reason,
                          superseded_by=args.superseded_by)
        save_ledger(ledger, path)
        sup = f" (superseded by {args.superseded_by})" if args.superseded_by else ""
        print(f"retired {name} [{backend}]: {args.reason}{sup}")
        return 0
    if args.clear:
        name, _, backend = args.clear.partition("::")
        ok = clear(ledger, name, backend or None)
        if ok:
            save_ledger(ledger, path)
            print(f"cleared {args.clear}")
            return 0
        print(f"no such exclusion: {args.clear}", file=sys.stderr)
        return 1
    meta = ledger.get("_meta", {})
    rows = list(iter_exclusions(ledger))
    print(f"# model exclusion ledger ({path})")
    print(f"# host_vram_gb={meta.get('host_vram_gb', '?')} "
          f"updated={meta.get('updated_at', '?')}  {len(rows)} exclusion(s)")
    for name, backend, e in sorted(rows):
        print(f"  {name:<42} {backend:<7} {e.get('reason'):<16} "
              f"{e.get('detail', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
