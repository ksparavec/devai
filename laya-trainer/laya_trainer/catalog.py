"""The laya catalog (deploy/laya-models.yaml) and base checkpoint verification.

One implementation, two users: scripts/select-models.py imports this on the
host to download a base checkpoint, and the trainer image uses it to refuse a
base that does not match the catalog (exit 4). Standard library + PyYAML only.

A checkpoint lives in `<store>/base/<name>@<revision[:12]>/` with the repo
subfolder stripped, so the same directory name always means the same bytes.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

SCHEMA_VERSION = 1

# What laya needs to load a checkpoint without network access: the weights,
# the decision-head config, the tokenizer, and the encoder config (without
# encoder/config.json, build_model falls back to downloading the base encoder
# from the Hub, which the offline trainer cannot do).
REQUIRED_FILES = (
    "model.safetensors",
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHUNK = 1 << 20


class CatalogError(ValueError):
    """The catalog file is malformed. The message names the row and the field."""


def _safe_relpath(value: Any) -> bool:
    """A relative POSIX path that cannot leave the directory it is joined to."""
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    p = PurePosixPath(value)
    return not p.is_absolute() and ".." not in p.parts and "." not in p.parts


def _check_row(i: int, row: Any) -> dict:
    where = f"row {i}"
    if not isinstance(row, dict):
        raise CatalogError(f"{where}: must be a mapping")
    name = row.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise CatalogError(f"{where}: name {name!r} must match {_NAME_RE.pattern}")
    where = f"row {name!r}"
    repo = row.get("repo")
    if not isinstance(repo, str) or not _REPO_RE.match(repo):
        raise CatalogError(f"{where}: repo {repo!r} must look like org/name")
    rev = row.get("revision")
    if not isinstance(rev, str) or not _REVISION_RE.match(rev):
        raise CatalogError(
            f"{where}: revision {rev!r} must be a full 40-hex commit; a branch or tag "
            f"can move, and a moved base silently changes every student trained on it")
    sub = row.get("subfolder", "")
    if not isinstance(sub, str) or (sub and not _safe_relpath(sub)):
        raise CatalogError(f"{where}: subfolder {sub!r} must be empty or a relative path")
    files = row.get("files")
    if not isinstance(files, dict) or not files:
        raise CatalogError(f"{where}: files must be a non-empty mapping")
    for rel, meta in files.items():
        if not _safe_relpath(rel):
            raise CatalogError(f"{where}: file path {rel!r} must be relative and stay inside")
        if not isinstance(meta, dict):
            raise CatalogError(f"{where}: {rel}: must map to sha256 and size")
        if not isinstance(meta.get("sha256"), str) or not _SHA256_RE.match(meta["sha256"]):
            raise CatalogError(f"{where}: {rel}: sha256 must be 64 lowercase hex characters")
        size = meta.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise CatalogError(f"{where}: {rel}: size must be a positive integer")
    missing = [f for f in REQUIRED_FILES if f not in files]
    if missing:
        raise CatalogError(f"{where}: files is missing {missing}")
    if not isinstance(row.get("default", False), bool):
        raise CatalogError(f"{where}: default must be true or false")
    return {**row, "subfolder": sub, "default": row.get("default", False)}


def parse_catalog(doc: Any) -> list[dict]:
    """Validate a loaded catalog document and return its rows."""
    if not isinstance(doc, dict):
        raise CatalogError("catalog must be a mapping")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise CatalogError(
            f"schema_version {doc.get('schema_version')!r} is not {SCHEMA_VERSION}")
    rows = doc.get("models")
    if not isinstance(rows, list) or not rows:
        raise CatalogError("models must be a non-empty list")
    checked = [_check_row(i, r) for i, r in enumerate(rows)]
    names = [r["name"] for r in checked]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise CatalogError(f"duplicate names: {dupes}")
    defaults = [r["name"] for r in checked if r["default"]]
    if len(defaults) != 1:
        raise CatalogError(f"exactly one row must be the default, found {defaults}")
    return checked


def load_catalog(path: Path) -> list[dict]:
    """Read and validate the catalog file. Raises CatalogError or OSError."""
    return parse_catalog(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def find(rows: list[dict], name: str) -> dict | None:
    return next((r for r in rows if r["name"] == name), None)


def default_row(rows: list[dict]) -> dict:
    return next(r for r in rows if r["default"])


def checkpoint_dirname(row: dict) -> str:
    """`<name>@<revision[:12]>`: the directory name pins the bytes."""
    return f"{row['name']}@{row['revision'][:12]}"


def repo_paths(row: dict) -> list[str]:
    """The files to download, as paths inside the repo (subfolder included)."""
    sub = row["subfolder"]
    return [f"{sub}/{rel}" if sub else rel for rel in row["files"]]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_dir(directory: Path, row: dict) -> list[str]:
    """Every problem that stops `directory` from being this row's checkpoint.

    An empty list means every catalog file is present with the pinned size and
    sha256. Size is compared first, so a truncated download fails without
    hashing hundreds of megabytes.
    """
    directory = Path(directory)
    problems = []
    for rel, meta in row["files"].items():
        p = directory / rel
        if not p.is_file():
            problems.append(f"{rel}: missing")
            continue
        size = p.stat().st_size
        if size != meta["size"]:
            problems.append(f"{rel}: size {size} != {meta['size']}")
            continue
        digest = file_sha256(p)
        if digest != meta["sha256"]:
            problems.append(f"{rel}: sha256 {digest} != {meta['sha256']}")
    return problems


def make_read_only(directory: Path) -> None:
    """Files 0444, directories 0555, deepest first."""
    for root, dirs, files in os.walk(directory, topdown=False):
        for f in files:
            os.chmod(os.path.join(root, f), 0o444)
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o555)
    os.chmod(directory, 0o555)


def make_writable(directory: Path) -> None:
    """Undo make_read_only (owner write bits only), so a tree can be removed."""
    os.chmod(directory, os.stat(directory).st_mode | stat.S_IWUSR)
    for root, dirs, files in os.walk(directory):
        for name in dirs + files:
            p = os.path.join(root, name)
            if not os.path.islink(p):
                os.chmod(p, os.stat(p).st_mode | stat.S_IWUSR)
