"""One job, in its own process: import -> validate -> train -> package.

    python -m laya_trainer.run_job --store /laya --catalog <yaml> --job <id> [--device cuda]

The controller starts this and waits. Running it as a separate process means a
crash or an out-of-memory error cannot take the controller (and with it the
router's view of the trainer) down, and all GPU memory is released when it
exits. It writes the job's phase to job.json as it goes and, at the end,
outcome.json; its exit code is the contract's (0 ok, 3 dataset, 4 base,
5 GPU, 6 diverged, 7 export/parity, 1 anything else).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from . import catalog as cat
from .contract import ExitCode, JobError, sha256_file, write_json_atomic
from .jobs import JobStore


def fine_tuned_model_id(job: dict) -> str:
    return f"ft:{job['model']}:devai:{job.get('suffix') or 'laya'}:{job['id'][6:18]}"


class Phases:
    """Phase bookkeeping: job.json's devai.phase, an event, and timings."""

    def __init__(self, store: JobStore, job_id: str) -> None:
        self.store, self.job_id = store, job_id
        self.timings: dict[str, float] = {}
        self._name: str | None = None
        self._t0 = self._start = time.monotonic()

    def enter(self, name: str, *, status: str | None = None) -> None:
        self._close()
        self._name, self._t0 = name, time.monotonic()
        fields = {"status": status} if status else {}
        self.store.update(self.job_id, devai={"phase": name}, **fields)
        self.store.add_event(self.job_id, f"Phase: {name}")

    def _close(self) -> None:
        if self._name:
            self.timings[self._name] = round(time.monotonic() - self._t0, 2)

    def finish(self) -> dict:
        self._close()
        self._name = None
        self.timings["total"] = round(time.monotonic() - self._start, 2)
        return self.timings


def run(store_dir: Path, catalog_path: Path, job_id: str, device_name: str) -> dict:
    """Run the job; return outcome fields. Raises JobError on a contract failure."""
    import torch

    from . import calibrate, dataset, export, golden, package, parity, train

    store = JobStore(store_dir)
    job = store.read(job_id)
    if job is None:
        raise JobError(ExitCode.UNEXPECTED, f"job {job_id} not found")
    run_dir = store.job_dir(job_id)
    phases = Phases(store, job_id)

    try:
        rows = cat.load_catalog(catalog_path)
    except (OSError, cat.CatalogError) as e:
        raise JobError(ExitCode.BASE, f"laya catalog unreadable: {e}") from None
    row = cat.find(rows, job["model"])
    if row is None:
        raise JobError(ExitCode.BASE, f"model {job['model']!r} is not in the laya catalog")
    base_dir = Path(store_dir) / "base" / cat.checkpoint_dirname(row)

    phases.enter("importing")
    ds_dir = dataset.import_dataset(Path(store_dir), job["training_file"])

    phases.enter("validating")
    manifest = dataset.load_manifest(ds_dir)
    dataset.check_base(manifest, row, base_dir)
    base_cfg = json.loads((base_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    tok = dataset.load_tokenizer(base_dir, base_cfg)
    splits = dataset.encode_dataset(ds_dir, manifest, tok)
    counts = {s: len(v) for s, v in splits.items()}
    store.add_event(job_id, "Dataset validated", data={"items": counts})

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise JobError(ExitCode.GPU, "CUDA is not available in the trainer container")
    device_info = ({"type": "cuda", "name": torch.cuda.get_device_name(device),
                    "capability": list(torch.cuda.get_device_capability(device))}
                   if device.type == "cuda" else {"type": device.type})
    cfg = train.checkpoint_cfg(base_cfg, manifest)
    model = train.load_model(base_dir, base_cfg)
    pad_id = tok.pad_token_id

    def on_epoch(ep: dict) -> None:
        store.add_event(job_id, f"Epoch {ep['epoch']}/{ep['n_epochs']}: loss {ep['loss']:.4f}",
                        data=ep)

    def on_checkpoint(epoch: int) -> None:
        train.save_checkpoint(model, run_dir / "checkpoint", base_dir=base_dir, cfg=cfg,
                              meta={"epoch": epoch, "job_id": job_id})

    phases.enter("training", status="running")
    result = train.train(model, splits["train"], job["hyperparameters"], device=device,
                         seed=job["seed"], pad_id=pad_id, on_epoch=on_epoch,
                         on_checkpoint=on_checkpoint)
    model = model.cpu().float().eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    phases.enter("exporting")
    try:
        export_info = export.export_onnx(model, run_dir)
        session = export.ort_session(run_dir / export.ONNX_NAME)
    except Exception as e:  # any exporter or ORT load failure is an export failure
        raise JobError(ExitCode.EXPORT, f"ONNX export failed: {type(e).__name__}: {e}") from None

    phases.enter("calibrating")
    calib_z = export.ort_logits(session, splits["calib"], pad_id)
    calibration = calibrate.calibrate(splits["calib"], calib_z)

    phases.enter("checking_parity")
    heldout = splits["heldout"]
    onnx_z = export.ort_logits(session, heldout, pad_id)
    parity_info = parity.check_parity(heldout, export.torch_logits(model, heldout, pad_id), onnx_z)

    phases.enter("packaging")
    ft_model = fine_tuned_model_id(job)
    training = {"job_id": job_id, "base": cat.checkpoint_dirname(row),
                "dataset": job["training_file"], "epochs": len(result.epochs),
                "updates": result.updates, "seconds": result.seconds}
    # The calibrated config goes into checkpoint/ too, so laya's own Agent on
    # the checkpoint -- the golden reference -- answers exactly as the artifact.
    package.write_configs(run_dir, package.artifact_config(
        cfg, calibration, fine_tuned_model=ft_model, training=training))
    rows_g = golden.golden_rows(run_dir / package.CHECKPOINT_DIR, heldout)
    golden.write_golden(run_dir / "golden.jsonl", rows_g)
    timings = phases.finish()
    package.write_artifact(
        run_dir, job=job, row=row, base_dir=base_dir, ds_id=job["training_file"],
        ds_manifest=manifest, ds_manifest_sha256=sha256_file(ds_dir / "manifest.json"),
        calibration=calibration, export=export_info, parity=parity_info,
        golden={"n": len(rows_g), "tolerance": golden.TOLERANCE, "file": "golden.jsonl"},
        train_result=result, device=device_info, timings=timings, fine_tuned_model=ft_model)
    return {"fine_tuned_model": ft_model, "trained_tokens": result.trained_tokens,
            "result_files": [f"runs/{job_id}/manifest.json"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    store = JobStore(args.store)
    outcome_path = store.job_dir(args.job) / "outcome.json"
    try:
        outcome = run(args.store, args.catalog, args.job, args.device)
        outcome["exit_code"] = int(ExitCode.OK)
    except JobError as e:
        outcome = {"exit_code": int(e.code), "message": e.message}
    except Exception as e:  # noqa: BLE001 -- the job's last word must reach job.json
        traceback.print_exc()
        outcome = {"exit_code": int(ExitCode.UNEXPECTED), "message": f"{type(e).__name__}: {e}"}
    write_json_atomic(outcome_path, outcome)
    if outcome["exit_code"]:
        print(f"laya job {args.job} failed (exit {outcome['exit_code']}): {outcome['message']}",
              file=sys.stderr, flush=True)
    return outcome["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
