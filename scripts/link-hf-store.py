#!/usr/bin/env python3
"""Hard-link HF model weights from one backend store into another.

vLLM and SGLang read separate directories but (since the 2026-07 storage
change) share one filesystem, so a model staged for both costs one copy,
not two -- if the second is hard-linked rather than downloaded again.

Hard links, not symlinks: each container bind-mounts only its OWN store,
so a symlink into the other store would dangle inside the container. A
hard link is just a second name for the same inode and resolves fine.

Idempotent, and safe to interrupt: the new tree is built beside the old
one and swapped in atomically, so a failure leaves the existing store
untouched.

  python3 scripts/link-hf-store.py --from vllm --to sglang [--name M] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

STORES = {
    "vllm": Path(os.environ.get("VLLM_MODELS_DIR", "/var/cache/devai/vllm")),
    "sglang": Path(os.environ.get("SGLANG_MODELS_DIR", "/var/cache/devai/sglang")),
}

# Alternative-format subtrees neither engine loads -- same list
# select-models.py excludes at download time. Skipped so the linked tree
# stays clean; linking them would cost no space but would misrepresent
# what the store holds.
EXCLUDE_DIRS = {"original", "metal", ".cache"}
EXCLUDE_SUFFIXES = (".gguf",)


def _already_shared(src_file: Path, existing: Path) -> bool:
    """True when `existing` is already the same inode as `src_file`."""
    try:
        return existing.exists() and existing.samefile(src_file)
    except OSError:
        return False


def link_tree(src: Path, staging: Path, current: Path,
              dry_run: bool) -> tuple[int, int, int]:
    """Hard-link every wanted file under `src` into `staging`.

    `current` is the destination tree as it stands right now (which may
    not exist). It is consulted ONLY to tell apart bytes this run will
    actually free from bytes that are already shared -- re-running on a
    linked store must report 0 reclaimed, not the full model size. That
    distinction is the whole point of the dry run.

    Returns (files_linked, bytes_reclaimed, files_already_shared).
    """
    files = reclaimed = shared = 0
    for root, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        rel = Path(root).relative_to(src)
        target_dir = staging / rel
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
        for fn in filenames:
            if fn.endswith(EXCLUDE_SUFFIXES):
                continue
            s = Path(root) / fn
            if s.is_symlink():        # never follow; the target may be excluded
                continue
            files += 1
            if _already_shared(s, current / rel / fn):
                shared += 1
            else:
                reclaimed += s.stat().st_size
            if not dry_run:
                d = target_dir / fn
                if d.exists():
                    d.unlink()
                os.link(s, d)
    return files, reclaimed, shared


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", choices=sorted(STORES), default="vllm")
    ap.add_argument("--to", dest="dst", choices=sorted(STORES), default="sglang")
    ap.add_argument("--name", help="Only this model (default: every model in --from).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.src == args.dst:
        sys.exit("error: --from and --to must differ")
    src_root, dst_root = STORES[args.src], STORES[args.dst]
    if not src_root.is_dir():
        sys.exit(f"error: source store {src_root} does not exist")

    if src_root.stat().st_dev != dst_root.stat().st_dev:
        sys.exit(
            f"error: {src_root} and {dst_root} are on different filesystems "
            f"(dev {src_root.stat().st_dev} vs {dst_root.stat().st_dev}); hard "
            f"links cannot cross that boundary. Download into {args.dst} "
            f"instead: scripts/select-models.py --name <M> --download "
            f"--hf-store {args.dst}")

    names = [args.name] if args.name else sorted(
        p.name for p in src_root.iterdir()
        if p.is_dir() and (p / "config.json").is_file())
    if not names:
        print(f"  nothing to link: no models under {src_root}", file=sys.stderr)
        return 0

    grand_files = grand_bytes = grand_shared = 0
    for name in names:
        src = src_root / name
        if not (src / "config.json").is_file():
            print(f"  [skip] {name}: not a model dir under {src_root}", file=sys.stderr)
            continue
        dst = dst_root / name
        staging = dst_root / f".{name}.linking"

        files, nbytes, shared = link_tree(src, staging, dst, args.dry_run)
        grand_files += files
        grand_bytes += nbytes
        grand_shared += shared
        gib = nbytes / (1024 ** 3)
        if shared == files and files:
            note = "already fully shared, nothing to reclaim"
        elif shared:
            note = f"{gib:.1f} GiB ({shared}/{files} files already shared)"
        else:
            note = f"{gib:.1f} GiB"
        if args.dry_run:
            print(f"  would link {name}: {files} files, {note}")
            continue
        # Atomic-ish swap: move the old tree aside, put the new one in
        # place, then delete. An interrupt leaves either the old or the
        # new tree at `dst`, never a half-built one.
        old = dst_root / f".{name}.old"
        if dst.exists():
            os.replace(dst, old)
        os.replace(staging, dst)
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
        print(f"  linked {name}: {files} files, {note}")

    verb = "would reclaim" if args.dry_run else "reclaimed"
    tail = (f" ({grand_shared} of {grand_files} were already shared)"
            if grand_shared else "")
    print(f"  {verb} ~{grand_bytes / (1024 ** 3):.1f} GiB "
          f"across {grand_files - grand_shared} file(s){tail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
