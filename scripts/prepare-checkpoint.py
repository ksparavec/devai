#!/usr/bin/env python3
"""Prepare a downloaded checkpoint for MTP on 24 GB, repeatably, with a record.

`make model-prepare NAME=<catalog row>`

What it does, in place, to <store>/<name>/:
  1. HyperQwen's CPU scripts (third_party/hyperqwen/prepare/, vendored
     unmodified, Apache-2.0) convert embed_tokens, lm_head and the mtp.*
     module from bf16 to int8 group-128 and add a 40,960-token draft head.
     Which scripts run depends on the shard layout, exactly as upstream's
     README prescribes:
       sharded checkpoint -> quant_lm_head.py, quant_embed.py, quant_mtp.py,
                             build_draft_vocab.py
       single shard       -> quant_heads_stream.py, build_draft_vocab.py
  2. When the body is not int-quantized (an NVFP4 body), scripts/
     mixed_quant_groups.py rewrites the three new config groups so vLLM
     resolves them as weight-only int8 -- upstream's scripts clone the body
     group, which is wrong there.
  3. PREPARED.json is written into the directory: the vendored-script commit,
     every step with its arguments and exit code, and sha256 + size of every
     rewritten or added file BEFORE and AFTER. The upstream scripts keep the
     originals next to the files (.bak, .bak_embed, .bak-mtp, .bak-orig,
     .bak-draft); the manifest names each one.

Why this exists: two checkpoints were prepared by hand on 2026-09-21 and the
commands lived only in that session. The repo's rule is that the model
stores are changed only by scripts that enforce the paths and record what
they did; this is that script for preparation. `--reconstruct` writes the
manifest for a directory that was prepared by hand, from its backups,
without running anything.

Every step runs inside the home-built devai-vllm image (it has torch,
safetensors and compressed_tensors, and it is the only image that can serve
the result: stock vLLM cannot load a quantized embedding for this
architecture). A failing step stops the run and writes no manifest.

Afterwards: re-probe the row (`make probe-vllm PROBE_REPO=<repo> ...`, or
PROBE_CTX_EXACT=<ctx> to confirm a computed ceiling), add the `image` and
flags to deploy/recovery-flags.json if not already there, and restart
devai-router -- it does not reload the probe cache on its own.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from mixed_quant_groups import normalise_int_groups  # noqa: E402

# The model stores. Written out, not derived, per the storage-layout rule in
# scripts/select-models.py.
STORES = {
    "vllm": Path("/var/cache/devai/vllm"),
    "sglang": Path("/var/cache/devai/sglang"),
}
IMAGE = "docker.io/devai/vllm-devai:latest"   # local image store only, never pulled
VENDOR_DIR = REPO_ROOT / "third_party" / "hyperqwen"
VENDOR_URL = "https://github.com/syv-ai/HyperQwen"
VENDOR_LICENSE = "Apache-2.0"
# Must equal HYPERQWEN_COMMIT in scripts/build-vllm.sh: the vLLM patches and
# these scripts are one series and are validated together upstream.
VENDOR_COMMIT = "c0c81bbbbf91f11b54af7b95f49bd6d1570c2ba0"
MANIFEST = "PREPARED.json"
DRAFT_IDS = "prepare/draft_vocab_ids.json"

# Backups the upstream scripts leave, ordered so that for any file the FIRST
# match is the copy made earliest in the pipeline, i.e. the original:
#   .bak-orig  quant_heads_stream.py renames every shard it rewrites
#   .bak-quant quant_lm_head.py / quant_heads_stream.py copy config + index first
#   .bak       quant_lm_head.py, the lm_head shard
#   .bak_embed quant_embed.py, the embed_tokens shard
#   .bak-mtp   quant_mtp.py: the extras shard, and config + index again
#   .bak-draft build_draft_vocab.py, the extras shard again
#   .bak-mixed scripts/mixed_quant_groups.py, config again
_BACKUP_SUFFIXES = (".bak-orig", ".bak-quant", ".bak", ".bak_embed", ".bak-mtp", ".bak-draft", ".bak-mixed")
# Files the preparation CREATES (no original exists).
_CREATED = ("model_extra_tensors.safetensors", "mtp_draft_vocab_ids.pt")

Runner = Callable[[list[str]], int]
CATALOG = REPO_ROOT / "deploy" / "models.yaml"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_backup(name: str) -> bool:
    return any(name.endswith(s) for s in _BACKUP_SUFFIXES)


def _snapshot(model_dir: Path) -> dict[str, tuple[str, int]]:
    """sha256 + size of every non-backup file (top level only)."""
    out = {}
    for p in sorted(model_dir.iterdir()):
        if p.is_file() and not _is_backup(p.name) and p.name != MANIFEST:
            out[p.name] = (sha256_of(p), p.stat().st_size)
    return out


def _backup_for(model_dir: Path, name: str) -> str | None:
    """The upstream backup holding `name`'s ORIGINAL bytes, when one exists.
    Prefers the earliest in the pipeline: .bak-orig (stream), then the
    per-script backups, then .bak-quant/.bak-mixed for the metadata files."""
    for suffix in _BACKUP_SUFFIXES:
        if (model_dir / (name + suffix)).exists():
            return name + suffix
    return None


def choose_method(model_dir: Path) -> str:
    """'sharded' for a multi-shard checkpoint (upstream's four-script recipe),
    'stream' for a single shard (quant_heads_stream.py)."""
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shards = {f for f in index["weight_map"].values()
              if f.startswith("model-") and f.endswith(".safetensors")}
    return "sharded" if len(shards) > 1 else "stream"


def steps_for(method: str) -> list[list[str]]:
    draft = ["prepare/build_draft_vocab.py", "/model", "--ids", DRAFT_IDS]
    if method == "sharded":
        return [["prepare/quant_lm_head.py", "/model"],
                ["prepare/quant_embed.py", "/model"],
                ["prepare/quant_mtp.py", "/model"],
                draft]
    return [["prepare/quant_heads_stream.py", "/model"], draft]


def body_is_int(config: dict) -> bool:
    g0 = config["quantization_config"]["config_groups"]["group_0"]
    return (g0.get("weights") or {}).get("type") == "int"


def container_argv(model_dir: Path, step: list[str]) -> list[str]:
    return ["podman", "run", "--rm", "--security-opt=label=disable",
            "-v", f"{model_dir}:/model:rw", "-v", f"{VENDOR_DIR}:/hq:ro",
            "-w", "/hq", "--entrypoint", "python3", IMAGE, *step]


def _podman_image_exists(image: str) -> bool:
    return subprocess.run(["podman", "image", "exists", image]).returncode == 0


def preflight(model_dir: Path, image_exists: Callable[[str], bool]) -> str | None:
    if not (model_dir / "config.json").is_file() or not (model_dir / "model.safetensors.index.json").is_file():
        return f"{model_dir}: not a downloaded checkpoint (no config.json + index)"
    if (model_dir / MANIFEST).exists():
        return f"{model_dir}: already prepared ({MANIFEST} present)"
    if any(_is_backup(p.name) for p in model_dir.iterdir()):
        return (f"{model_dir}: holds .bak* files from an earlier (hand-run or "
                f"interrupted) preparation but no {MANIFEST}; running the scripts "
                "again would double-quantize. Use --reconstruct if it is complete.")
    config = json.loads((model_dir / "config.json").read_text())
    qc = config.get("quantization_config") or {}
    if qc.get("quant_method") != "compressed-tensors" or "config_groups" not in qc:
        return (f"{model_dir}: quant_method={qc.get('quant_method')!r}; the vendored "
                "scripts write compressed-tensors pack-quantized tensors and nothing else")
    if not image_exists(IMAGE):
        return f"image {IMAGE} not found -- run 'make build-vllm' first"
    return None


def _manifest_files(before: dict, after: dict, model_dir: Path,
                    source_dir: Path | None = None) -> dict:
    files = {}
    for name, (sha, size) in after.items():
        prev = before.get(name)
        if prev is not None and prev[0] == sha:
            continue                      # untouched
        entry = {
            "sha256_before": prev[0] if prev else None,
            "size_before": prev[1] if prev else None,
            "sha256_after": sha,
            "size_after": size,
            "backup": None if source_dir else _backup_for(model_dir, name),
        }
        if source_dir:
            entry["original_in"] = str(source_dir) if prev else None
        files[name] = entry
    return files


def derived_name_for(source: str, catalog: Path = CATALOG) -> str | None:
    """The catalog row declared `derived_from: <source>`, or None."""
    import yaml
    try:
        rows = (yaml.safe_load(catalog.read_text()) or {}).get("models") or []
    except (OSError, yaml.YAMLError):
        return None
    for r in rows:
        if r.get("source") == "derived" and r.get("derived_from") == source:
            return r.get("name")
    return None


def _copy_on_write(src: Path, dst: Path) -> None:
    """dst = a copy of src's top-level files, reflinked where the filesystem
    allows (XFS with reflink: zero extra space until a file is rewritten;
    the upstream scripts rename-then-write, so the source is never touched
    through a shared link). `.cache/` -- Hugging Face download metadata --
    is deliberately not copied: the derived directory is not a download."""
    dst.mkdir()
    for f in sorted(src.iterdir()):
        if not f.is_file():
            continue
        subprocess.run(["cp", "--reflink=auto", "-p", str(f), str(dst / f.name)], check=True)


def _write_manifest(model_dir: Path, payload: dict) -> None:
    (model_dir / MANIFEST).write_text(json.dumps(payload, indent=2) + "\n")


def prepare(model_dir: Path, name: str, *, runner: Runner,
            derived_from: str | None = None, source_dir: Path | None = None) -> int:
    config = json.loads((model_dir / "config.json").read_text())
    method = choose_method(model_dir)
    before = _snapshot(model_dir)
    print(f"==> preparing {name} ({method}; body "
          f"{'int' if body_is_int(config) else 'float'}-quantized)")
    steps_run = []
    for step in steps_for(method):
        argv = container_argv(model_dir, step)
        print(f"    {' '.join(step)}")
        t0 = time.time()
        rc = runner(argv)
        steps_run.append({"script": step[0], "args": step[1:], "rc": rc,
                          "seconds": round(time.time() - t0, 1)})
        if rc != 0:
            print(f"error: {step[0]} exited {rc}; stopping, no manifest written. "
                  f"The upstream scripts keep originals as .bak* files -- inspect "
                  f"{model_dir} before retrying.", file=sys.stderr)
            return 1
    mixed = False
    config = json.loads((model_dir / "config.json").read_text())
    if not body_is_int(config):
        print("    scripts/mixed_quant_groups.py (float body: int8 groups need their own format)")
        fixed = normalise_int_groups(config["quantization_config"])
        cfg_path = model_dir / "config.json"
        (model_dir / "config.json.bak-mixed").write_bytes(cfg_path.read_bytes())
        cfg_path.write_text(json.dumps(dict(config, quantization_config=fixed), indent=2))
        mixed = True
    if source_dir is not None:
        # A derived directory keeps no backups: the source holds every
        # original, and the manifest pairs each rewritten file with it.
        for bak in [q for q in model_dir.iterdir() if _is_backup(q.name)]:
            bak.unlink()
    after = _snapshot(model_dir)
    _write_manifest(model_dir, {
        "name": name,
        "prepared_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "method": method,
        "derived_from": derived_from,
        "source_dir": str(source_dir) if source_dir else None,
        "hyperqwen": {"url": VENDOR_URL, "commit": VENDOR_COMMIT, "license": VENDOR_LICENSE,
                      "vendored_at": str(VENDOR_DIR.relative_to(REPO_ROOT))},
        "image": IMAGE,
        "steps": steps_run,
        "mixed_quant_groups": mixed,
        "files": _manifest_files(before, after, model_dir, source_dir),
        "reconstructed": False,
    })
    total = sum(v[1] for v in after.values()) / 2**30
    print(f"==> done: {len(after)} live files, {total:.2f} GiB; manifest {model_dir / MANIFEST}")
    print("    next: make cache-down && make probe-vllm PROBE_REPO=<repo> [PROBE_CTX_EXACT=<ctx>]; "
          "make cache-up; podman restart devai-router")
    return 0


def derive(source_dir: Path, source_name: str, derived_dir: Path, derived_name: str,
           *, runner: Runner, image_exists: Callable[[str], bool]) -> int:
    """Make <derived_dir> from an as-downloaded <source_dir> and prepare it
    there. The source is never modified."""
    problem = preflight(source_dir, image_exists)
    if problem:
        print(f"error: source {problem}", file=sys.stderr)
        return 1
    if derived_dir.exists():
        print(f"error: {derived_dir} already exists; remove it to derive again", file=sys.stderr)
        return 1
    print(f"==> deriving {derived_name} from {source_name} (copy-on-write)")
    _copy_on_write(source_dir, derived_dir)
    return prepare(derived_dir, derived_name, runner=runner,
                   derived_from=source_name, source_dir=source_dir)


def reconstruct(model_dir: Path, name: str, method: str,
                source_dir: Path | None = None, derived_from: str | None = None) -> int:
    """Manifest for a directory prepared by hand: 'before' comes from the
    upstream backups (or, for a derived directory, from the source's copy
    of each file), 'after' from the live files. Runs nothing."""
    if (model_dir / MANIFEST).exists():
        print(f"error: {model_dir} already has {MANIFEST}", file=sys.stderr)
        return 1
    if source_dir is None and not any(_is_backup(p.name) for p in model_dir.iterdir()):
        print(f"error: {model_dir} has no .bak* files; it was not prepared", file=sys.stderr)
        return 1
    live = _snapshot(model_dir)
    after, before = {}, {}
    if source_dir is not None:
        for fname, (sha, size) in live.items():
            src = source_dir / fname
            if src.is_file():
                ssha = sha256_of(src)
                if ssha == sha:
                    continue              # identical to the source: untouched
                before[fname] = (ssha, src.stat().st_size)
                after[fname] = (sha, size)
            elif fname in _CREATED:
                after[fname] = (sha, size)
    else:
        # Only files the preparation touched: those with an upstream
        # backup (rewritten) and the ones it creates.
        for fname, (sha, size) in live.items():
            b = _backup_for(model_dir, fname)
            if b:
                before[fname] = (sha256_of(model_dir / b), (model_dir / b).stat().st_size)
                after[fname] = (sha, size)
            elif fname in _CREATED:
                after[fname] = (sha, size)
    config = json.loads((model_dir / "config.json").read_text())
    _write_manifest(model_dir, {
        "name": name,
        "prepared_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "method": method,
        "derived_from": derived_from,
        "source_dir": str(source_dir) if source_dir else None,
        "hyperqwen": {"url": VENDOR_URL, "commit": VENDOR_COMMIT, "license": VENDOR_LICENSE,
                      "vendored_at": str(VENDOR_DIR.relative_to(REPO_ROOT))},
        "image": IMAGE,
        "steps": [{"script": s[0], "args": s[1:], "rc": 0, "seconds": None} for s in steps_for(method)],
        "mixed_quant_groups": (model_dir / "config.json.bak-mixed").exists() or not body_is_int(config),
        "files": _manifest_files(before, after, model_dir, source_dir),
        "reconstructed": True,
        "reconstructed_note": "prepared by hand on 2026-09-21 with the same scripts and "
                              "arguments; 'before' hashes are taken from the upstream "
                              "backups, so files with no backup have none",
    })
    print(f"==> manifest reconstructed: {model_dir / MANIFEST}")
    return 0


def main(argv: list[str], *, runner: Runner | None = None,
         image_exists: Callable[[str], bool] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="catalog row name (= directory name in the store)")
    ap.add_argument("--store", choices=sorted(STORES), default="vllm")
    ap.add_argument("--dir", help="explicit checkpoint directory (tests; overrides --store)")
    ap.add_argument("--reconstruct", action="store_true",
                    help="write the manifest for a directory prepared by hand; runs nothing")
    ap.add_argument("--method", choices=("sharded", "stream"),
                    help="with --reconstruct: which recipe was used")
    ap.add_argument("--derive", action="store_true",
                    help="prepare INTO a new directory named by the catalog's derived row "
                         "(derived_from == NAME); the source stays as downloaded")
    ap.add_argument("--to", help="with --derive: the derived name (default: from the catalog)")
    ap.add_argument("--catalog", default=str(CATALOG), help=argparse.SUPPRESS)
    ap.add_argument("--source-dir", help="with --reconstruct on a derived directory: its source")
    args = ap.parse_args(argv)
    runner = runner or (lambda a: subprocess.call(a))
    image_exists = image_exists or _podman_image_exists
    model_dir = Path(args.dir) if args.dir else STORES[args.store] / args.name
    if not model_dir.is_dir():
        print(f"error: {model_dir}: no such directory (download it first: "
              f"make model-pull NAME={args.name})", file=sys.stderr)
        return 1
    if args.reconstruct:
        if not args.method:
            print("error: --reconstruct needs --method sharded|stream", file=sys.stderr)
            return 2
        src = Path(args.source_dir) if args.source_dir else None
        return reconstruct(model_dir, args.name, args.method, source_dir=src,
                           derived_from=src.name if src else None)
    if args.derive:
        derived = args.to or derived_name_for(args.name, Path(args.catalog))
        if not derived:
            print(f"error: no derived row for {args.name} in {args.catalog}; declare it under "
                  f"the family's `derived:` in scripts/model-families.yaml and run "
                  f"`make catalog-regen`, or pass --to <name>", file=sys.stderr)
            return 1
        return derive(model_dir, args.name, model_dir.parent / derived, derived,
                      runner=runner, image_exists=image_exists)
    problem = preflight(model_dir, image_exists)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 1
    return prepare(model_dir, args.name, runner=runner)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
