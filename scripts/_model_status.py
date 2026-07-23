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
  manual               -- operator-pinned

Sha-stability and vram-stability follow plan decision 2: only too_big /
too_small / unsupported_arch / manual survive a re-quant; only
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

VALID_REASONS = ("too_big", "too_small", "unsupported_arch", "oom", "manual")
# Reasons that do NOT depend on the exact weights (survive a re-quant).
_SHA_STABLE_REASONS = ("too_big", "too_small", "unsupported_arch", "manual")
# Reasons that do NOT depend on the GPU VRAM (survive a GPU change).
_VRAM_INDEPENDENT_REASONS = ("unsupported_arch", "manual")


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
                     sha: str | None = None) -> None:
    """Mark (name, backend) excluded for `reason`. Idempotent (overwrites)."""
    if reason not in VALID_REASONS:
        raise ValueError(f"invalid reason {reason!r}; expected {VALID_REASONS}")
    m = ledger.setdefault("models", {}).setdefault(name, {})
    if repo:
        m["repo"] = repo
    m.setdefault("backends", {})[backend] = {
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
    if reason not in ("too_big", "too_small", "oom"):
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
    ap.add_argument("--ledger", default=str(LEDGER_PATH))
    args = ap.parse_args(argv)
    path = Path(args.ledger)
    ledger = load_ledger(path)
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
