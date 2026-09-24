"""What the trainer promises its callers: exit codes, hashes, file writing.

Standard library only, so the controller and the tests of everything that is
not training can run without torch.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from enum import IntEnum
from pathlib import Path
from typing import Any

# Dataset schema versions this trainer accepts (devai owns the schema; see
# docs/plans/laya-trainer.md and aiagent's design doc section 8.1).
DATASET_SCHEMA_VERSIONS = frozenset({1})
# Artifact format this trainer writes (devai owns it; aiagent section 8.3).
ARTIFACT_FORMAT_VERSION = 1

SPLITS = ("train", "calib", "heldout", "pool")
REQUIRED_SPLITS = ("train", "calib", "heldout")

# laya is pinned by hash in requirements.lock; this is the upstream commit of
# that release, recorded in every artifact manifest.
LAYA_VERSION = "0.3.20"
LAYA_COMMIT = "23a1752"


class ExitCode(IntEnum):
    OK = 0
    UNEXPECTED = 1
    DATASET = 3
    BASE = 4
    GPU = 5
    DIVERGED = 6
    EXPORT = 7
    TIMEOUT = 124


# The `error.code` string a failed job carries for each exit code.
ERROR_CODES = {
    ExitCode.UNEXPECTED: "unexpected_error",
    ExitCode.DATASET: "dataset_contract_violation",
    ExitCode.BASE: "base_checkpoint_mismatch",
    ExitCode.GPU: "gpu_unavailable_or_oom",
    ExitCode.DIVERGED: "training_diverged",
    ExitCode.EXPORT: "export_or_parity_failure",
    ExitCode.TIMEOUT: "timeout",
}


def error_code(exit_code: int) -> str:
    try:
        return ERROR_CODES[ExitCode(exit_code)]
    except ValueError:
        return f"exit_{exit_code}"


class JobError(Exception):
    """A job failure with the exit code the job process ends with."""

    def __init__(self, code: ExitCode, message: str) -> None:
        super().__init__(message)
        self.code = ExitCode(code)
        self.message = message


def canonical_hash(obj: Any) -> str:
    """H(x) from the contract: sha256 over compact JSON, key order as given.

    Order is part of the value on purpose (option order sets marker positions),
    so keys are NOT sorted.
    """
    data = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write JSON so a concurrent reader sees the old file or the new one."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def copy_dir_files(src: Path, dst: Path) -> None:
    """Copy the regular files of `src` into a new directory `dst`, byte for byte.

    Deliberately NOT shutil.copytree: that copies modes too, and the base
    checkpoints are sealed (0444 files in 0555 directories), so the copy would
    be a directory nobody can later replace or remove without root. Used for
    the tokenizer, which must stay byte-identical to the base (a tokenizer
    re-saved by transformers can persist state such as truncation).
    """
    dst = Path(dst)
    dst.mkdir(parents=True)
    for p in sorted(Path(src).iterdir()):
        if p.is_file() and not p.is_symlink():
            shutil.copyfile(p, dst / p.name)
            os.chmod(dst / p.name, 0o644)


def read_sha256sums(path: Path) -> dict[str, str]:
    """Parse `sha256sum` output: `<hex>  <name>` (or `<hex> *<name>`) per line."""
    out: dict[str, str] = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        digest, sep, name = line.partition(" ")
        name = name[1:] if name.startswith((" ", "*")) else name
        if not sep or len(digest) != 64 or not name:
            raise ValueError(f"{path.name}:{n}: not a sha256sum line: {line!r}")
        out[name] = digest.lower()
    return out


def write_sha256sums(directory: Path, names: list[str]) -> None:
    lines = [f"{sha256_file(Path(directory) / n)}  {n}" for n in sorted(names)]
    (Path(directory) / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
