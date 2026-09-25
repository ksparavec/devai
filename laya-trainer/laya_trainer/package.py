"""The artifact (devai -> aiagent): runs/<job>/, format_version 1.

The artifact is manifest.json plus every key of manifest.files:

    model.onnx [+ model.onnx.data]   the graph aiagent runs
    tokenizer/                       the base checkpoint's tokenizer, byte for byte
    rl_agent_config.json             base config, dataset token limits, applied temperatures
    golden.jsonl                     answers aiagent's runtime must reproduce
    NOTICE

SHA256SUMS lists exactly those plus the checkpoint/ files (torch-loadable fp32
weights; on the volume, never part of what aiagent installs). job.json,
events.jsonl and outcome.json are job bookkeeping in the same directory and are
listed nowhere. devai reports loss and parity only; whether the student ships
is aiagent's decision, made through its own ONNX runtime.
"""

from __future__ import annotations

import os
import platform
import shutil
import time
from pathlib import Path

from .contract import (ARTIFACT_FORMAT_VERSION, LAYA_COMMIT, LAYA_VERSION, canonical_hash,
                       copy_dir_files, sha256_file, write_json_atomic, write_sha256sums)

CHECKPOINT_DIR = "checkpoint"
# Job bookkeeping, and the files that describe the artifact rather than being it.
NOT_IN_MANIFEST = frozenset({"job.json", "events.jsonl", "outcome.json", "manifest.json",
                             "SHA256SUMS"})


def artifact_config(cfg: dict, calibration: dict, *, fine_tuned_model: str,
                    training: dict) -> dict:
    """The run's rl_agent_config.json.

    `cfg` already carries the dataset's max_len / head_max_len (the only source
    of them). The per-type temperatures are the calibrated, clamped ones, and
    `temperature_by_options` is emptied: a per-type fit would be hidden by the
    base's per-bucket values.
    """
    out = dict(cfg)
    out["temperature"] = calibration["temperature_applied"]
    out["temperature_by_options"] = {}
    out["fine_tuned"] = True
    out["model_name"] = fine_tuned_model
    out["training"] = training
    return out


def write_configs(run_dir: Path, cfg: dict) -> None:
    """The same config at the artifact root and in checkpoint/, so laya's Agent
    on the checkpoint answers exactly like the artifact (golden answers)."""
    write_json_atomic(Path(run_dir) / "rl_agent_config.json", cfg)
    write_json_atomic(Path(run_dir) / CHECKPOINT_DIR / "rl_agent_config.json", cfg)


def notice_text(row: dict, fine_tuned_model: str) -> str:
    return (
        f"{fine_tuned_model}\n\n"
        f"A fine-tuned derivative of the laya checkpoint {row['name']} "
        f"(https://huggingface.co/{row['repo']}, revision {row['revision']}"
        f"{', subfolder ' + row['subfolder'] if row['subfolder'] else ''}), "
        f"licensed {row.get('license', 'as stated upstream')}. "
        f"Its encoder was initialised from {row.get('encoder', 'the upstream encoder')}.\n\n"
        f"laya: https://github.com/NandhaKishorM/laya (Apache-2.0), version {LAYA_VERSION}, "
        f"commit {LAYA_COMMIT}.\n"
        f"Fine-tuned by devai's laya trainer on labels produced by a teacher model; see "
        f"manifest.json for the dataset binding.\n"
    )


def _versions() -> dict:
    import importlib.metadata as md
    out = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "onnx", "onnxscript", "onnxruntime", "laya",
                "tokenizers", "safetensors", "numpy"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def _tree(run_dir: Path) -> dict[str, dict]:
    """sha256 + size of every regular file under run_dir, by relative path."""
    table = {}
    for p in sorted(run_dir.rglob("*")):
        rel = p.relative_to(run_dir).as_posix()
        if p.is_file() and rel not in NOT_IN_MANIFEST:
            table[rel] = {"sha256": sha256_file(p), "size": p.stat().st_size}
    return table


def write_artifact(run_dir: Path, *, job: dict, row: dict, base_dir: Path,
                   ds_id: str, ds_manifest: dict, ds_manifest_sha256: str, calibration: dict,
                   export: dict, parity: dict, golden: dict, train_result, device: dict,
                   timings: dict, fine_tuned_model: str) -> dict:
    """Write tokenizer, NOTICE, manifest.json and SHA256SUMS; return the manifest.

    model.onnx, golden.jsonl, the configs and checkpoint/ must already be there.
    """
    run_dir = Path(run_dir)
    tok_dir = run_dir / "tokenizer"
    if tok_dir.exists():
        shutil.rmtree(tok_dir)
    copy_dir_files(Path(base_dir) / "tokenizer", tok_dir)
    (run_dir / "NOTICE").write_text(notice_text(row, fine_tuned_model), encoding="utf-8")

    producer = ds_manifest.get("producer") or {}
    binds = dict(producer.get("binds") or {})  # verbatim, plus the dataset binding
    binds["dataset_manifest_sha256"] = ds_manifest_sha256
    base = {"name": row["name"], "revision": row["revision"],
            "weights_sha256": row["files"]["model.safetensors"]["sha256"],
            "tokenizer_sha256": row["files"]["tokenizer/tokenizer.json"]["sha256"],
            "dir": f"{row['name']}@{row['revision'][:12]}"}
    trainer_config = {"hyperparameters": job["hyperparameters"], "seed": job["seed"],
                      "laya": LAYA_VERSION, "max_len": ds_manifest["max_len"],
                      "head_max_len": ds_manifest["head_max_len"]}
    tree = _tree(run_dir)
    files = {rel: meta for rel, meta in tree.items()
             if not rel.startswith(CHECKPOINT_DIR + "/")}
    artifact_id = canonical_hash({
        "binds": binds, "base": base, "trainer_config_sha256": canonical_hash(trainer_config),
        "files": {rel: meta["sha256"] for rel, meta in files.items()},
    })
    manifest = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "artifact_id": artifact_id,
        "job_id": job["id"],
        "fine_tuned_model": fine_tuned_model,
        "created_at": int(time.time()),
        "base_checkpoint": base,
        "dataset": {"id": ds_id, "manifest_sha256": ds_manifest_sha256,
                    "counts": ds_manifest["splits"]["counts"]},
        "binds": binds,
        "laya": {"version": LAYA_VERSION, "commit": LAYA_COMMIT},
        "trainer": {
            "build": os.environ.get("LAYA_TRAINER_BUILD"),
            "image": os.environ.get("LAYA_TRAINER_IMAGE"),
            "versions": _versions(),
            "device": device,
            "hyperparameters": job["hyperparameters"],
            "seed": job["seed"],
            "config_sha256": canonical_hash(trainer_config),
            "timings": timings,
            "peak_vram_gib": train_result.peak_vram_gib,
        },
        "calibration": calibration,
        "export": export,
        "parity": parity,
        "golden": golden,
        "metrics": {"loss_per_epoch": [e["loss"] for e in train_result.epochs],
                    "reward_per_epoch": [e["reward"] for e in train_result.epochs],
                    "trained_tokens": train_result.trained_tokens},
        "files": files,
    }
    write_json_atomic(run_dir / "manifest.json", manifest)
    write_sha256sums(run_dir, ["manifest.json", *tree])
    return manifest
