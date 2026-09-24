"""Dataset import and the dataset contract (aiagent -> devai), schema_version 1.

A dataset is a directory `inbox/ds-<12 hex>/` written by aiagent, where the id
is "ds-" + the first 12 hex of sha256(manifest.json bytes):

    manifest.json   schema_version, base_checkpoint, max_len, head_max_len,
                    splits {method, counts}, files {name: {sha256, rows}},
                    producer (opaque to devai, except producer.binds)
    train.jsonl  calib.jsonl  heldout.jsonl  [pool.jsonl]
    SHA256SUMS      `sha256sum` lines for exactly the other files

Import copies it to `datasets/<id>/` (read-only afterwards). Validation then
checks every row and RE-TOKENIZES it with laya's own `build_sequence`: the
token ids the student is trained on must be exactly the ids aiagent's runtime
will feed it, so each labeled row carries `student_tokens.<question>.ids_sha256
= H(ids)` and a mismatch stops the job (exit 3) before any GPU time is spent.
Pool rows (no gold, no teacher) are counted and never trained on.

The rules mirror aiagent's own validator (its spec section 3.3). Import and
manifest checks are standard library only; encoding imports laya.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

from . import catalog as cat
from .contract import (DATASET_SCHEMA_VERSIONS, REQUIRED_SPLITS, SPLITS, ExitCode,
                       JobError, canonical_hash, read_sha256sums, sha256_file)
from .jobs import DATASET_ID_RE

ALLOWED_FILES = frozenset({"manifest.json"} | {f"{s}.jsonl" for s in SPLITS})
REQUIRED_FILES = frozenset({"manifest.json"} | {f"{s}.jsonl" for s in REQUIRED_SPLITS})
PROB_TOLERANCE = 1e-6


def _violation(message: str) -> JobError:
    return JobError(ExitCode.DATASET, message)


def verify_checksums(directory: Path) -> dict[str, str]:
    """Check SHA256SUMS covers exactly the dataset files and that they match."""
    directory = Path(directory)
    if directory.is_symlink():
        raise _violation(f"{directory.name}: a dataset directory must not be a symlink")
    sums_path = directory / "SHA256SUMS"
    if not sums_path.is_file() or sums_path.is_symlink():
        raise _violation(f"{directory.name}: SHA256SUMS missing")
    try:
        sums = read_sha256sums(sums_path)
    except ValueError as e:
        raise _violation(str(e)) from None
    entries = [p for p in directory.iterdir() if p.name != "SHA256SUMS"]
    # A symlinked file would import whatever it points at into the store.
    odd = sorted(p.name for p in entries if p.is_symlink() or not p.is_file())
    if odd:
        raise _violation(f"{directory.name}: {odd} are not regular files")
    present = {p.name for p in entries}
    unknown = sorted(present - ALLOWED_FILES)
    if unknown:
        raise _violation(f"{directory.name}: unexpected files {unknown}; allowed: {sorted(ALLOWED_FILES)}")
    missing = sorted(REQUIRED_FILES - present)
    if missing:
        raise _violation(f"{directory.name}: missing {missing}")
    if set(sums) != present:
        raise _violation(f"{directory.name}: SHA256SUMS lists {sorted(sums)} but the directory "
                         f"holds {sorted(present)}")
    for name, digest in sums.items():
        actual = sha256_file(directory / name)
        if actual != digest:
            raise _violation(f"{directory.name}/{name}: sha256 {actual} != SHA256SUMS {digest}")
    return sums


def _check_id(ds_id: str, sums: dict[str, str]) -> None:
    want = "ds-" + sums["manifest.json"][:12]
    if ds_id != want:
        raise _violation(f"dataset id {ds_id} does not name its manifest (expected {want}: "
                         f"'ds-' + the first 12 hex of sha256(manifest.json))")


def import_dataset(store: Path, ds_id: str) -> Path:
    """Copy inbox/<id> to datasets/<id> (verified, read-only) and return it."""
    if not isinstance(ds_id, str) or not DATASET_ID_RE.match(ds_id):
        raise _violation(f"training_file {ds_id!r} is not a dataset id (ds-<12 hex>)")
    store = Path(store)
    src, dst = store / "inbox" / ds_id, store / "datasets" / ds_id
    if dst.is_dir():
        existing = verify_checksums(dst)
        _check_id(ds_id, existing)
        if src.is_dir() and read_sha256sums(src / "SHA256SUMS") != existing:
            raise _violation(f"datasets/{ds_id} already exists with different contents; "
                             f"a dataset id names one immutable dataset")
        return dst
    if not src.is_dir():
        raise _violation(f"inbox/{ds_id} not found")
    sums = verify_checksums(src)
    _check_id(ds_id, sums)
    for stale in (store / "datasets").glob(f".importing-{ds_id}-*"):
        cat.make_writable(stale)
        shutil.rmtree(stale)
    tmp = store / "datasets" / f".importing-{ds_id}-{os.getpid()}"
    tmp.mkdir(parents=True)
    for name in [*sums, "SHA256SUMS"]:
        shutil.copyfile(src / name, tmp / name)
    verify_checksums(tmp)  # the copy, not just the source
    cat.make_read_only(tmp)
    os.rename(tmp, dst)
    return dst


def _int_in(value: Any, lo: int, hi: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and lo <= value <= hi


def load_manifest(ds_dir: Path) -> dict:
    """Parse and check manifest.json against the directory (not the rows)."""
    ds_dir = Path(ds_dir)
    try:
        m = json.loads((ds_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise _violation(f"manifest.json: {e}") from None
    if not isinstance(m, dict):
        raise _violation("manifest.json must be an object")
    if m.get("schema_version") not in DATASET_SCHEMA_VERSIONS:
        raise _violation(f"schema_version {m.get('schema_version')!r} is not supported "
                         f"(supported: {sorted(DATASET_SCHEMA_VERSIONS)})")
    base = m.get("base_checkpoint")
    fields = ("name", "revision", "weights_sha256", "tokenizer_sha256")
    if not isinstance(base, dict) or not all(isinstance(base.get(f), str) for f in fields):
        raise _violation(f"base_checkpoint must be an object with string {list(fields)}")
    if not _int_in(m.get("max_len"), 32, 8192):
        raise _violation("max_len must be an integer in [32, 8192]")
    if not _int_in(m.get("head_max_len"), 16, m["max_len"] - 16):
        raise _violation("head_max_len must be an integer in [16, max_len - 16]")
    splits = m.get("splits")
    if not isinstance(splits, dict) or not isinstance(splits.get("method"), str):
        raise _violation("splits must be an object with a string method")
    counts = splits.get("counts")
    if not isinstance(counts, dict) or set(counts) - set(SPLITS) or \
            not all(_int_in(v, 0, 10**9) for v in counts.values()):
        raise _violation(f"splits.counts must map a subset of {list(SPLITS)} to row counts")
    empty = [s for s in REQUIRED_SPLITS if not counts.get(s)]
    if empty:
        raise _violation(f"splits {empty} must not be empty")
    jsonl = sorted(p.name for p in ds_dir.glob("*.jsonl"))
    files = m.get("files")
    if not isinstance(files, dict) or sorted(files) != jsonl:
        raise _violation(f"files must describe exactly the row files {jsonl}")
    sums = read_sha256sums(ds_dir / "SHA256SUMS")
    for name, meta in files.items():
        if not isinstance(meta, dict) or meta.get("sha256") != sums.get(name) or \
                not _int_in(meta.get("rows"), 0, 10**9):
            raise _violation(f"files[{name!r}] must carry the file's sha256 and its row count")
    producer = m.get("producer", {})
    if not isinstance(producer, dict) or not isinstance(producer.get("binds", {}), dict):
        raise _violation("producer must be an object, and producer.binds an object")
    return m


def check_base(manifest: dict, row: dict, base_dir: Path) -> None:
    """The dataset was built for this catalog row, and its files are intact."""
    want = manifest["base_checkpoint"]
    have = {"name": row["name"], "revision": row["revision"],
            "weights_sha256": row["files"]["model.safetensors"]["sha256"],
            "tokenizer_sha256": row["files"]["tokenizer/tokenizer.json"]["sha256"]}
    diff = [f"{k}: dataset {want[k]!r} != catalog {have[k]!r}" for k in have if want[k] != have[k]]
    if diff:
        raise JobError(ExitCode.BASE, "dataset was built for another base checkpoint: " + "; ".join(diff))
    if not Path(base_dir).is_dir():
        raise JobError(ExitCode.BASE, f"base checkpoint {base_dir} missing; run "
                                      f"`make model-pull NAME={row['name']}` on the host")
    problems = cat.verify_dir(base_dir, row)
    if problems:
        raise JobError(ExitCode.BASE, f"base checkpoint {base_dir} does not match the catalog: "
                                      + "; ".join(problems))


def option_keys(q: dict) -> list[str]:
    """Probability keys in label-index order, for a laya internal question."""
    if q["t"] == "choice":
        return [str(k) for k in q["crit"]]
    if q["t"] == "score":
        return [str(i) for i in range(len(q["crit"]))]
    return ["false", "true"]


def _null_path(value: Any, path: str) -> str | None:
    if value is None:
        return path
    if isinstance(value, dict):
        for k, v in value.items():
            if (found := _null_path(v, f"{path}.{k}")) is not None:
                return found
    elif isinstance(value, list):
        for i, v in enumerate(value):
            if (found := _null_path(v, f"{path}[{i}]")) is not None:
                return found
    return None


def _check_row_shape(row: Any, split: str, where: str) -> None:
    if not isinstance(row, dict):
        raise _violation(f"{where}: a row must be an object")
    # Null is never a value in a row (aiagent drops None fields). `questions`
    # is exempt: it is laya's question object verbatim, where a null
    # criterion description is legitimate.
    for key, value in row.items():
        if key != "questions" and (found := _null_path(value, key)) is not None:
            raise _violation(f"{where}: null at {found}")
    labeled = split != "pool"
    for key in ("gold", "teacher"):
        if labeled and not isinstance(row.get(key), dict):
            raise _violation(f"{where}: a {split} row needs a {key} object")
        if not labeled and key in row:
            raise _violation(f"{where}: a pool row carries no {key}")
    for key, kind in (("id", str), ("group_id", str), ("questions", dict), ("student_tokens", dict)):
        if not isinstance(row.get(key), kind) or not row[key]:
            raise _violation(f"{where}: {key} must be a non-empty {kind.__name__}")
    if row.get("split") != split:
        raise _violation(f"{where}: split {row.get('split')!r} does not match the file")
    if not isinstance(row.get("synthetic"), bool):
        raise _violation(f"{where}: synthetic must be true or false")
    if row["synthetic"] and split in ("calib", "heldout"):
        raise _violation(f"{where}: synthetic rows are not allowed in {split}")
    if not isinstance(row.get("state"), (str, dict, list)):
        raise _violation(f"{where}: state must be a string, object or list")
    qids = list(row["questions"])
    keys = ("gold", "student_tokens") if labeled else ("student_tokens",)
    for key in keys:
        if sorted(row[key]) != sorted(qids):
            raise _violation(f"{where}: {key} keys {sorted(row[key])} != questions {sorted(qids)}")


def _target(gold: Any, keys: list[str], where: str) -> tuple[list[float], int]:
    if not isinstance(gold, dict) or not isinstance(gold.get("probabilities"), dict):
        raise _violation(f"{where}: gold needs a probabilities object")
    probs = gold["probabilities"]
    if list(probs) != keys:
        raise _violation(f"{where}: probability keys {list(probs)} != options {keys} (in order)")
    target = []
    for k in keys:
        v = probs[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise _violation(f"{where}: probability {k!r}={v!r} is not a finite number >= 0")
        target.append(float(v))
    if abs(sum(target) - 1.0) > PROB_TOLERANCE:
        raise _violation(f"{where}: probabilities sum to {sum(target)!r}, not 1")
    label = gold.get("label")
    first_argmax = keys[target.index(max(target))]
    if label != first_argmax:
        raise _violation(f"{where}: label {label!r} is not the first argmax {first_argmax!r}")
    return target, keys.index(label)


def load_tokenizer(base_dir: Path, cfg: dict):
    """laya's own tokenizer loader, so tokenization matches laya's runtime."""
    from laya.agent import _load_tokenizer
    return _load_tokenizer(str(Path(base_dir) / "tokenizer"), cfg)


def _encode_row(row: dict, split: str, where: str, tok, max_len: int, head_max_len: int) -> list[dict]:
    from laya.agent import Agent
    from laya.common import QTYPES, build_sequence, render_options, serialize_state

    state = row["state"]
    truncate_left = isinstance(state, list)
    state_ids = tok(serialize_state(state).replace(tok.mask_token, " "),
                    add_special_tokens=False)["input_ids"]
    items = []
    for qid, qdef in row["questions"].items():
        qwhere = f"{where} question {qid!r}"
        try:
            Agent._check_question(qid, qdef)
        except ValueError as e:
            raise _violation(f"{where}: {e}") from None
        q = Agent._to_internal(qdef)
        keys = option_keys(q)
        ids, markers = build_sequence(tok, state, q, max_len, head_max_len,
                                      truncate_left=truncate_left, state_ids=state_ids)
        if len(markers) != len(render_options(q)):
            raise _violation(f"{qwhere}: options exceed head_max_len={head_max_len}")
        stateless, _ = build_sequence(tok, state, q, max_len, head_max_len,
                                      truncate_left=truncate_left, state_ids=[])
        if len(state_ids) > max_len - len(stateless):
            raise _violation(f"{qwhere}: the state ({len(state_ids)} tokens) does not fit "
                             f"max_len={max_len}; segments must never be truncated")
        st = row["student_tokens"][qid]
        if not isinstance(st, dict) or st.get("n") != len(ids) or \
                st.get("ids_sha256") != canonical_hash(ids):
            raise _violation(
                f"{qwhere}: student_tokens do not match laya's tokenization "
                f"(n={len(ids)}, ids_sha256={canonical_hash(ids)}); the dataset was "
                f"tokenized differently from the student")
        if split == "pool":
            continue
        target, label = _target(row["gold"][qid], keys, qwhere)
        items.append({
            "ids": ids, "markers": markers, "qtype": QTYPES[q["t"]],
            "target": target, "label": label, "keys": keys,
            "row_id": row["id"], "qid": qid, "state": state, "question": qdef,
        })
    return items


def encode_dataset(ds_dir: Path, manifest: dict, tok) -> dict[str, list[dict]]:
    """Validate every row and turn each labeled (row, question) into a training item.

    Returns items per split; pool rows are validated and counted but yield no
    items (the trainer never trains on them).
    """
    max_len, head_max_len = manifest["max_len"], manifest["head_max_len"]
    out: dict[str, list[dict]] = {s: [] for s in SPLITS}
    counts: dict[str, int] = {s: 0 for s in SPLITS}
    groups: dict[str, set[str]] = {s: set() for s in SPLITS}
    seen: set[str] = set()
    for split in SPLITS:
        path = Path(ds_dir) / f"{split}.jsonl"
        if not path.exists():
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            where = f"{split}.jsonl:{n}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise _violation(f"{where}: {e}") from None
            _check_row_shape(row, split, where)
            if row["id"] in seen:
                raise _violation(f"{where}: duplicate row id {row['id']!r}")
            seen.add(row["id"])
            counts[split] += 1
            groups[split].add(row["group_id"])
            out[split] += _encode_row(row, split, where, tok, max_len, head_max_len)
    declared = manifest["splits"]["counts"]
    wrong = [f"{s}: manifest {declared.get(s, 0)} != rows {counts[s]}"
             for s in SPLITS if declared.get(s, 0) != counts[s]]
    wrong += [f"files[{s}.jsonl].rows {manifest['files'][f'{s}.jsonl']['rows']} != rows {counts[s]}"
              for s in SPLITS if f"{s}.jsonl" in manifest["files"]
              and manifest["files"][f"{s}.jsonl"]["rows"] != counts[s]]
    if wrong:
        raise _violation("row counts do not match the manifest: " + "; ".join(wrong))
    # Calib and held-out must not share a document with anything the student
    # trains on (train and pool may share groups after a repair round).
    sides = {"calib": groups["calib"], "heldout": groups["heldout"],
             "train+pool": groups["train"] | groups["pool"]}
    names = list(sides)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sides[a] & sides[b]
            if shared:
                raise _violation(f"{a} and {b} share {len(shared)} group_id(s), e.g. {sorted(shared)[0]}")
    return out
