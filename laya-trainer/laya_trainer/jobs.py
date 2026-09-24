"""Job state on the volume: runs/<job>/job.json and events.jsonl.

The volume is the source of truth. aiagent reads these files directly when the
trainer is not resident (the router never launches the trainer just to answer a
status read), so everything the API returns is exactly what is on disk.

Writers: the controller writes job.json when a job is created and after its
process has exited; the job process writes it (phase, status) while it runs.
The two never overlap in time. Events are appended by both, one short JSON
line per write (O_APPEND), newest last.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any

from .contract import write_json_atomic

JOB_ID_RE = re.compile(r"^ftjob-[0-9a-f]{24}$")
DATASET_ID_RE = re.compile(r"^ds-[0-9a-f]{12}$")
SUFFIX_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")

TERMINAL = frozenset({"succeeded", "failed", "cancelled"})

# OpenAI's `batch_size` is examples per optimizer update; devai adds the
# forward-pass size (micro_batch_size) and the memory mode.
HYPERPARAMETER_DEFAULTS: dict[str, Any] = {
    "n_epochs": 4,
    "batch_size": 32,
    "learning_rate_multiplier": 1.0,
    "micro_batch_size": 8,
    "memory_mode": "lean",
}
_INT_RANGES = {"n_epochs": (1, 50), "batch_size": (1, 512), "micro_batch_size": (1, 64)}
MEMORY_MODES = ("lean", "fast")
DEFAULT_SEED = 42


def resolve_hyperparameters(raw: Any) -> dict[str, Any]:
    """Fill defaults (OpenAI's "auto" means the default) and check ranges.

    Raises ValueError naming the offending key.
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("hyperparameters must be an object")
    unknown = sorted(set(raw) - set(HYPERPARAMETER_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown hyperparameters {unknown}; known: {sorted(HYPERPARAMETER_DEFAULTS)}")
    out = dict(HYPERPARAMETER_DEFAULTS)
    for key, value in raw.items():
        if value == "auto":
            continue
        if key in _INT_RANGES:
            lo, hi = _INT_RANGES[key]
            if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
                raise ValueError(f"{key} must be an integer in [{lo}, {hi}] or \"auto\"")
        elif key == "learning_rate_multiplier":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 10:
                raise ValueError("learning_rate_multiplier must be a number in (0, 10] or \"auto\"")
            value = float(value)
        elif key == "memory_mode" and value not in MEMORY_MODES:
            raise ValueError(f"memory_mode must be one of {list(MEMORY_MODES)}")
        out[key] = value
    return out


def new_job_id() -> str:
    return "ftjob-" + secrets.token_hex(12)


def _event(message: str, level: str, data: dict | None) -> dict:
    return {
        "object": "fine_tuning.job.event",
        "id": "ftevent-" + secrets.token_hex(12),
        "created_at": int(time.time()),
        "level": level,
        "message": message,
        "type": "metrics" if data else "message",
        "data": data or {},
    }


class JobStore:
    """runs/<job>/ under the laya store."""

    def __init__(self, store: Path) -> None:
        self.runs = Path(store) / "runs"

    def job_dir(self, job_id: str) -> Path:
        if not JOB_ID_RE.match(job_id):
            raise ValueError(f"not a job id: {job_id!r}")
        return self.runs / job_id

    def create(self, *, model: str, training_file: str, hyperparameters: dict,
               seed: int, suffix: str | None, metadata: dict | None,
               base_dirname: str) -> dict:
        job_id = new_job_id()
        d = self.job_dir(job_id)
        d.mkdir(parents=True)
        now_ns = time.time_ns()
        now = now_ns // 1_000_000_000
        job = {
            "object": "fine_tuning.job",
            "id": job_id,
            "model": model,
            "created_at": now,
            "finished_at": None,
            "fine_tuned_model": None,
            "organization_id": "devai",
            "result_files": [],
            "status": "validating_files",
            "validation_file": None,
            "training_file": training_file,
            "hyperparameters": hyperparameters,
            "trained_tokens": None,
            "error": None,
            "seed": seed,
            "suffix": suffix,
            "metadata": metadata,
            "estimated_finish": None,
            "integrations": [],
            "devai": {
                "phase": "queued",
                "started_at": now,
                # created_at has one-second resolution; this orders the listing.
                "created_ns": now_ns,
                "exit_code": None,
                "run_dir": f"runs/{job_id}",
                "base_checkpoint": base_dirname,
            },
        }
        self.write(job)
        self.add_event(job_id, "Job created")
        return job

    def read(self, job_id: str) -> dict | None:
        try:
            p = self.job_dir(job_id) / "job.json"
        except ValueError:
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def write(self, job: dict) -> None:
        write_json_atomic(self.job_dir(job["id"]) / "job.json", job)

    def update(self, job_id: str, *, devai: dict | None = None, **fields: Any) -> dict:
        """Read-modify-write job.json. `devai` keys merge into job["devai"]."""
        job = self.read(job_id)
        if job is None:
            raise FileNotFoundError(job_id)
        job.update(fields)
        if devai:
            job["devai"] = {**job.get("devai", {}), **devai}
        self.write(job)
        return job

    def add_event(self, job_id: str, message: str, *, level: str = "info",
                  data: dict | None = None) -> dict:
        ev = _event(message, level, data)
        line = json.dumps(ev, ensure_ascii=False, separators=(",", ":")) + "\n"
        fd = os.open(self.job_dir(job_id) / "events.jsonl",
                     os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
        return ev

    def events(self, job_id: str) -> list[dict]:
        """Newest first, as the OpenAI API lists them."""
        try:
            text = (self.job_dir(job_id) / "events.jsonl").read_text(encoding="utf-8")
        except (FileNotFoundError, ValueError):
            return []
        out = []
        for line in text.splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line torn by a crash mid-write; the rest still count
        return out[::-1]

    def list(self) -> list[dict]:
        """Every job, newest first."""
        if not self.runs.is_dir():
            return []
        jobs = [self.read(p.name) for p in self.runs.iterdir() if JOB_ID_RE.match(p.name)]
        return sorted((j for j in jobs if j), reverse=True, key=lambda j: (
            (j.get("devai") or {}).get("created_ns", j["created_at"] * 1_000_000_000), j["id"]))

    def fail(self, job_id: str, code: str, message: str, exit_code: int | None) -> dict:
        job = self.update(job_id, status="failed", finished_at=int(time.time()),
                          error={"code": code, "message": message, "param": None},
                          devai={"exit_code": exit_code})
        self.add_event(job_id, message, level="error")
        return job

    def reconcile(self) -> list[str]:
        """Mark every job left non-terminal by a stopped trainer as failed.

        A router eviction at the hold cap, `make cache-down` or a crash stops
        the container mid-job; nothing else would ever finish those records.
        """
        stale = [j["id"] for j in self.list() if j["status"] not in TERMINAL]
        for job_id in stale:
            self.fail(job_id, "trainer_stopped",
                      "The trainer stopped during this job (router hold cap, cache-down or a "
                      "crash). Epoch checkpoints already written are in checkpoint/.", None)
        return stale
