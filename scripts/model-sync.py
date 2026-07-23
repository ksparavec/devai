#!/usr/bin/env python3
"""Closed-loop model onboarding: catalog -> download -> probe -> record.

Diffs the catalog (deploy/models.yaml) against what the host has already
evaluated (probe caches + exclusion ledger) and, for each GENUINELY NEW row,
runs the existing ledger-aware tools to download + probe it and record the
outcome. So a freshly discovered/added catalog row becomes either `serving`
(fits + loads) or `excluded` (too_big/too_small/unsupported_arch) with no
manual bookkeeping.

Classification (pure, unit-tested):
  excluded   -- every advertised backend is excluded in the ledger.
  evaluated  -- already has a probe-cache entry (serving or rejected).
  new        -- neither: needs onboarding.

Ledger hygiene: on a non-dry run the ledger is first pruned of models the
catalog no longer carries (guarded -- see `prune_ledger`).

Execution (composition of existing targets; needs a GPU + podman):
  make model-pull DOWNLOAD_LIMIT=<budget>   # pulls new best-fit, records
                                            # too_big/too_small to the ledger
  make cache-down                           # GPU exclusivity for HF probes
  make probe-vllm / probe-sglang / probe    # probes new, records
                                            # unsupported_arch, skips excluded
  make cache-up                             # restore serving backends

`--dry-run` prints the plan and does nothing. `--max-downloads` caps the
unattended download budget. See docs/plans/model-lifecycle-ledger.md Phase 4.

    python3 scripts/model-sync.py --dry-run
    python3 scripts/model-sync.py --max-downloads 3 --family qwen3.5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import _model_status as MS  # noqa: E402

CATALOG = REPO_ROOT / "deploy" / "models.yaml"
OLLAMA_CACHE = REPO_ROOT / "deploy" / ".ollama-reasoning-cache.json"
VLLM_CACHE = REPO_ROOT / "deploy" / ".vllm-reasoning-cache.json"
SGLANG_CACHE = REPO_ROOT / "deploy" / ".sglang-reasoning-cache.json"


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_catalog(path: Path = CATALOG) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text()) if Path(path).is_file() else {}
    return (data or {}).get("models", []) or []


def is_probed(row: dict, ollama_cache: dict, vllm_cache: dict,
              sglang_cache: dict) -> bool:
    """True iff the host already has a probe-cache entry for this row.

    HF rows are keyed `repo@sha`: that exact key is AUTHORITATIVE -- we do NOT
    fall back to an alias match, because a stranded `repo@OLDsha` entry of the
    same repo (re-quant) carries the same name in its aliases and would shadow
    the new sha, leaving a re-quanted model classified 'evaluated' forever (the
    gemma double-key bug). Ollama rows are digest-keyed (no repo/sha), so for
    those we match the name against the ollama cache's aliases.
    """
    repo = row.get("repo")
    sha = (row.get("sha") or "").strip()
    name = row.get("name") or ""
    if repo and sha:
        key = f"{repo}@{sha}"
        return key in vllm_cache or key in sglang_cache
    # Ollama (or any repo/sha-less) row: match the name against cached aliases.
    for cache in (ollama_cache, vllm_cache, sglang_cache):
        for k, e in cache.items():
            if k.startswith("_") or not isinstance(e, dict):
                continue
            if name and name in (e.get("aliases") or []):
                return True
    return False


def _row_backends(row: dict) -> list[str]:
    return [b for b in (row.get("backend") or [])
            if b in ("ollama", "vllm", "sglang")]


def plan_sync(catalog_rows: list[dict], ollama_cache: dict, vllm_cache: dict,
              sglang_cache: dict, ledger: dict, *, host_vram: float) -> dict:
    """Classify catalog rows into new / evaluated / excluded. Pure."""
    new, evaluated, excluded = [], [], []
    for row in catalog_rows:
        name = row.get("name") or ""
        sha = (row.get("sha") or "").strip() or None
        backends = _row_backends(row)
        if backends and all(
                MS.is_excluded(ledger, name, b, host_vram=host_vram, sha=sha)
                for b in backends):
            excluded.append(row)
        elif is_probed(row, ollama_cache, vllm_cache, sglang_cache):
            evaluated.append(row)
        else:
            new.append(row)
    return {"new": new, "evaluated": evaluated, "excluded": excluded}


def _print_plan(plan: dict, host_vram: float, max_downloads: int,
                family: str | None) -> None:
    scope = f" family~{family}" if family else ""
    print(f"# model-sync plan (host_vram={host_vram:g} GB,"
          f" max_downloads={max_downloads}{scope})")
    print(f"#   {len(plan['evaluated'])} already evaluated, "
          f"{len(plan['excluded'])} excluded, {len(plan['new'])} new")
    for row in plan["new"][:max_downloads]:
        print(f"  + onboard  {row.get('name')}  "
              f"[{','.join(_row_backends(row))}]")
    extra = len(plan["new"]) - max_downloads
    if extra > 0:
        print(f"  ... and {extra} more new row(s) beyond the "
              f"max_downloads={max_downloads} budget this run")


# A prune that drops more of the ledger than this looks like a truncated
# catalog rather than a real removal -- refuse and let the operator decide.
_PRUNE_MAX_FRACTION = 0.5


def prune_ledger(catalog_rows: list[dict], ledger: dict, *,
                 path: Path = MS.LEDGER_PATH) -> int:
    """Drop ledger rows for models the catalog no longer carries.

    model-sync is the right owner: it is the only tool that loads the FULL
    catalog fresh from disk purely to reconcile it against host state (and
    with REGEN=1 the catalog was just regenerated, or the target aborted).
    Callers must pass the UNFILTERED catalog -- pruning against a
    `--family`-scoped subset would drop every other family's verdicts.

    A stale ledger row is harmless; a wrongly-pruned one silently
    re-downloads and re-probes a model this host already rejected. So two
    guards, either of which makes this a no-op:
      - the catalog must have parsed at least one named row (a missing or
        unreadable models.yaml loads as [] and would prune everything);
      - never drop more than half the ledger in one run -- a truncated
        catalog is indistinguishable from a mass removal.
    Returns the number of rows dropped (0 when a guard fires).
    """
    catalog_names = {r.get("name") for r in catalog_rows if r.get("name")}
    if not catalog_names:
        print("  [warn] catalog has no named rows -- skipping ledger prune",
              file=sys.stderr)
        return 0
    models = ledger.get("models") or {}
    stale = sorted(n for n in models if n not in catalog_names)
    if not stale:
        return 0
    if len(stale) > len(models) * _PRUNE_MAX_FRACTION:
        print(f"  [warn] ledger prune would drop {len(stale)}/{len(models)} "
              f"row(s) -- refusing (a truncated catalog looks exactly like "
              f"this). Clear by hand: make model-status CLEAR=<name>",
              file=sys.stderr)
        return 0
    n = MS.prune_to_catalog(ledger, catalog_names)
    MS.save_ledger(ledger, path)
    print(f"  pruned {n} stale ledger row(s): {', '.join(stale)}")
    return n


def _run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def execute(plan: dict, *, max_downloads: int) -> int:
    """Run the existing ledger-aware targets to onboard the new rows.

    Downloads exactly the budgeted new rows BY NAME (`make model-pull
    NAME=<name>`), so max_downloads is a true RUN-TOTAL cap -- not the
    per-(family,backend,ctx)-cell limit that `DOWNLOAD_LIMIT` would impose.
    Then probes once (incremental + ledger-gated) under GPU exclusivity. A
    download failure for one row is logged and skipped, not fatal.
    """
    queue = plan["new"][:max_downloads]
    if not queue:
        print("\nnothing new to onboard.")
        return 0
    pulled = 0
    for row in queue:
        rc = _run(["make", "model-pull", f"NAME={row['name']}"])
        if rc != 0:
            print(f"  ! download failed (rc={rc}): {row['name']}", file=sys.stderr)
        else:
            pulled += 1
    if pulled == 0:
        print("\nno rows downloaded; skipping probe phase.")
        return 1
    # The HF probers need GPU exclusivity, so the serving backends go DOWN
    # first. Whatever happens in between -- a non-zero probe rc, a raised
    # exception, Ctrl-C -- `make cache-up` MUST run in the finally, or a
    # failed probe leaves the ENTIRE inference stack offline. The original
    # failure (return code or exception) is preserved: the finally neither
    # returns nor raises on the success path.
    rc = 0
    try:
        for cmd in (["make", "cache-down"], ["make", "probe-vllm"],
                    ["make", "probe-sglang"]):
            rc = _run(cmd)
            if rc != 0:
                print(f"\nstep failed (rc={rc}): {' '.join(cmd)}",
                      file=sys.stderr)
                break
    finally:
        up_rc = _run(["make", "cache-up"])
        if up_rc != 0:
            print(f"\nstep failed (rc={up_rc}): make cache-up",
                  file=sys.stderr)
    if rc != 0:
        return rc
    if up_rc != 0:
        return up_rc
    rc = _run(["make", "probe"])
    if rc != 0:
        print(f"\nstep failed (rc={rc}): make probe", file=sys.stderr)
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Closed-loop catalog -> download -> probe -> record.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and exit; change nothing.")
    ap.add_argument("--max-downloads", type=int,
                    default=int(os.environ.get("SYNC_MAX_DOWNLOADS", "3")),
                    help="Cap new downloads this run (default 3 / "
                         "$SYNC_MAX_DOWNLOADS).")
    ap.add_argument("--family", default=None,
                    help="Scope to an EXACT catalog family name (e.g. qwen3.5)"
                         " -- matches `make model-pull FAMILY=` semantics.")
    ap.add_argument("--vram", type=float,
                    default=float(os.environ.get("GPU_MEMORY_GB", "24")),
                    help="Host VRAM budget in GB (default $GPU_MEMORY_GB or 24).")
    args = ap.parse_args(argv)

    catalog = load_catalog()
    ledger = MS.load_ledger()
    if not args.dry_run:
        # Against the UNFILTERED catalog, before the --family scoping below.
        prune_ledger(catalog, ledger)
    if args.family:
        catalog = [r for r in catalog if (r.get("family") or "") == args.family]
    plan = plan_sync(catalog, _read_json(OLLAMA_CACHE), _read_json(VLLM_CACHE),
                     _read_json(SGLANG_CACHE), ledger,
                     host_vram=args.vram)
    _print_plan(plan, args.vram, args.max_downloads, args.family)
    if args.dry_run:
        return 0
    return execute(plan, max_downloads=args.max_downloads)


if __name__ == "__main__":
    sys.exit(main())
