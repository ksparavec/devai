#!/usr/bin/env python3
"""Report backend image drift against the probe caches (Phase C).

For each HF backend (vLLM, SGLang) compare the digest of the locally
available container image against the digest its probe cache was captured
with (`_meta.current_image_digest`, stamped by scripts/_probe_core.
stamp_image_digest). A mismatch means the cache's fit / serving_ok / parser
data was measured on a different image and should be refreshed:

    make cache-down && make probe-vllm && make probe-load-vllm   # or sglang

This is the operator-facing companion to the router's startup drift check
(gpu-arbiter reads the same `_meta` and flags stale backends with a loud log
+ X-DevAI-Warning). Read-only; writes nothing.

Exit status: 0 when no backend has drifted (fresh, or no baseline to compare);
1 when at least one backend is stale -- so it can gate a probe-refresh step.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _probe_core import image_digest_via_cli  # noqa: E402

# (backend, default cache path, image-ref env var, image default). Defaults
# mirror deploy/docker-compose.yaml so `make probe-check` matches what the
# router and probers actually launch.
BACKENDS = (
    (
        "vllm",
        "deploy/.vllm-reasoning-cache.json",
        "VLLM_IMAGE",
        "docker.io/vllm/vllm-openai:v0.22.1-x86_64-cu129-ubuntu2404",
    ),
    (
        "sglang",
        "deploy/.sglang-reasoning-cache.json",
        "SGLANG_IMAGE",
        "docker.io/lmsysorg/sglang:v0.5.10.post1-cu130",
    ),
)


def _probed_digest(cache_path: Path) -> str | None:
    """current_image_digest from a cache's _meta block, or None."""
    try:
        data = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    meta = data.get("_meta")
    if not isinstance(meta, dict):
        return None
    return meta.get("current_image_digest")


def _report_fingerprints() -> bool:
    """Report launch-argv fingerprint spread per backend cache.

    A cell's `launch_fingerprint` records the SHAPE of the argv it was
    measured under (see _probe_hf_common.launch_fingerprint). Cells within
    one backend legitimately differ -- parser names and per-model recovery
    flags are part of the shape -- so this does NOT flag a mismatch as an
    error. What it surfaces is UNSTAMPED cells: those predate the
    fingerprint entirely and cannot be checked against anything, which is
    the condition that let seven SGLang cells keep a verdict measured
    under an argv the lab no longer emits.

    Returns True when any cache still holds unstamped cells.
    """
    print("Launch-argv fingerprints")
    print("=" * 72)
    any_unstamped = False
    repo_root = Path(__file__).resolve().parent.parent
    for backend, rel_path, _env, _img in BACKENDS:
        cache_path = repo_root / rel_path
        if not cache_path.exists():
            continue
        try:
            data = json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        stamped = unstamped = 0
        prints: set[str] = set()
        for key, entry in data.items():
            if key.startswith("_") or not isinstance(entry, dict):
                continue
            for band in (entry.get("probes") or {}).values():
                if not isinstance(band, dict):
                    continue
                for cell in band.values():
                    if not isinstance(cell, dict) or not cell.get("fits"):
                        continue
                    fp = cell.get("launch_fingerprint")
                    if fp:
                        stamped += 1
                        prints.add(str(fp))
                    else:
                        unstamped += 1
        total = stamped + unstamped
        if not total:
            continue
        if unstamped:
            any_unstamped = True
            print(f"  {backend:<7} {unstamped}/{total} fitting cell(s) UNSTAMPED "
                  f"-- measured before launch fingerprints existed, so the argv "
                  f"they were measured under is unknown")
        else:
            print(f"  {backend:<7} all {total} fitting cell(s) stamped")
        if prints:
            print(f"          shapes  : {', '.join(sorted(prints))}")
    print()
    if any_unstamped:
        print("Unstamped cells are not proof of staleness, only of age. "
              "Re-probe with PROBE_FORCE=1 to stamp them.")
        print()
    return False  # advisory only -- never fails the gate on age alone


def main() -> int:
    repo_root = Path(
        os.environ.get("DEVAI_REPO_ROOT", Path(__file__).resolve().parents[1]))
    runtime = os.environ.get("CONTAINER_RUNTIME", "podman")

    stale_any = False
    print(f"probe-check: runtime={runtime}\n")
    for backend, rel_cache, img_env, img_default in BACKENDS:
        cache_path = repo_root / (os.environ.get(
            f"{backend.upper()}_PROBE_CACHE_PATH") or rel_cache)
        # `or img_default` (not a get() default): the Makefile may export the
        # var as an empty string when it's unset in .env, which would
        # otherwise override the fallback and report a false IMAGE-ABSENT.
        image_ref = os.environ.get(img_env) or img_default

        probed = _probed_digest(cache_path)
        running = image_digest_via_cli(runtime, image_ref)

        if probed is None:
            status = "NO-BASELINE (cache has no _meta; re-probe to stamp)"
        elif running is None:
            status = f"IMAGE-ABSENT ({image_ref} not pulled locally)"
        elif probed == running:
            status = "OK"
        else:
            status = "DRIFT -- re-probe needed"
            stale_any = True

        print(f"  {backend:<7} {status}")
        print(f"          image   : {image_ref}")
        print(f"          probed  : {probed or '(none)'}")
        print(f"          running : {running or '(none)'}\n")

    stale_any = _report_fingerprints() or stale_any

    if stale_any:
        print("At least one backend has drifted. Refresh with:")
        print("  make cache-down && make probe-vllm && make probe-load-vllm")
        print("  make cache-down && make probe-sglang && make probe-load-sglang")
        return 1
    print("No drift detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
