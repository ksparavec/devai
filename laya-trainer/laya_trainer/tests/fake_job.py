"""Stands in for `python -m laya_trainer.run_job` in the controller tests.

Takes the same arguments and behaves as $FAKE_JOB_MODE says: succeed, fail3
(a dataset violation with outcome.json), crash (exit 9, no outcome), sleep,
finish_on_term (completes successfully when stopped).
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

from ..contract import write_json_atomic
from ..jobs import JobStore


def main() -> int:
    ap = argparse.ArgumentParser()
    for flag in ("--store", "--catalog", "--job", "--device"):
        ap.add_argument(flag)
    args = ap.parse_args()
    store = JobStore(Path(args.store))
    run_dir = store.job_dir(args.job)
    store.update(args.job, status="running", devai={"phase": "training"})
    mode = os.environ.get("FAKE_JOB_MODE", "succeed")
    if mode == "finish_on_term":
        # A job that completes at the very moment the trainer is stopped.
        def done(*_):
            _succeed(store, run_dir, args.job)
            sys.exit(0)
        signal.signal(signal.SIGTERM, done)
        time.sleep(120)
        return 1
    if mode == "sleep":
        time.sleep(120)
        return 0
    if mode == "crash":
        return 9
    if mode == "fail3":
        write_json_atomic(run_dir / "outcome.json", {"exit_code": 3, "message": "bad dataset"})
        return 3
    _succeed(store, run_dir, args.job)
    return 0


def _succeed(store: JobStore, run_dir: Path, job_id: str) -> None:
    (run_dir / "model.onnx").write_bytes(b"onnx")
    write_json_atomic(run_dir / "outcome.json", {
        "exit_code": 0, "fine_tuned_model": f"ft:laya-fake:devai:x:{job_id[6:18]}",
        "trained_tokens": 10, "result_files": [f"runs/{job_id}/manifest.json"]})


if __name__ == "__main__":
    sys.exit(main())
