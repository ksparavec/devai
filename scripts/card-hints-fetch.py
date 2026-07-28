#!/usr/bin/env python3
"""Stage METADATA ONLY for curated families, for out-of-sample validation.

`make card-hints` can only compare predictions against checkpoints that
are downloaded, and the discriminator rules were written by reading
exactly those checkpoints -- so that comparison is in-sample and proves
very little. This fetches the three metadata files (a few hundred KB per
repo, no weights, no GPU) for families the rules were NOT fitted on, so
the prediction can be tested where it might actually be wrong.

Staged under the USER cache, not /var/cache/devai: per CLAUDE.md's
mount-point convention, a new top-level directory there is not
volume-backed and would silently land on the root filesystem.

    make card-hints-fetch          # all curated families
    make card-hints-fetch FAMILY=llama3.1
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST = Path(
    os.environ.get("DEVAI_CARD_HINTS_DIR")
    or (Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
        / "devai" / "card-hints")
)
# Exactly what the discriminators read. Nothing else is fetched, so this
# cannot pull weights even by accident.
INCLUDE = ("chat_template.jinja", "tokenizer_config.json",
           "generation_config.json")


def families() -> list[dict]:
    import yaml
    data = yaml.safe_load((REPO_ROOT / "scripts" / "model-families.yaml").read_text())
    return (data or {}).get("families") or []


def main() -> int:
    only = os.environ.get("FAMILY") or ""
    targets: list[tuple[str, str]] = []
    for fam in families():
        name = fam.get("name") or ""
        if only and name != only:
            continue
        if not (fam.get("parsers") or {}):
            continue  # nothing curated to compare against
        repos = fam.get("hf_repos") or []
        if not repos:
            continue
        repo = repos[0] if isinstance(repos[0], str) else None
        if repo:
            targets.append((name, repo))

    if not targets:
        print(f"no curated families matched (FAMILY={only!r})")
        return 0

    DEST.mkdir(parents=True, exist_ok=True)
    print(f"staging metadata for {len(targets)} family/families -> {DEST}\n")

    failures = 0
    for name, repo in targets:
        out = DEST / repo.split("/")[-1]
        cmd = ["hf", "download", repo, "--local-dir", str(out)]
        for pat in INCLUDE:
            cmd += ["--include", pat]
        print(f"  {name:22} {repo}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            failures += 1
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            print(f"    FAILED: {tail[-1] if tail else 'unknown error'}")
            continue
        got = sorted(p.name for p in out.iterdir()) if out.is_dir() else []
        got = [g for g in got if not g.startswith(".")]
        print(f"    ok: {', '.join(got) or '(nothing matched the include list)'}")

    print(f"\ndone. {len(targets) - failures}/{len(targets)} staged.")
    print("Now run `make card-hints` to compare predictions against curated values.")
    # A gated/unavailable repo is not a build failure -- the fixtures are
    # committed and the tests run offline.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
