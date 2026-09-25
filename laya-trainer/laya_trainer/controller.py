"""The trainer's HTTP API: a subset of OpenAI's fine-tuning jobs API.

    POST /v1/fine_tuning/jobs                 start a job (one at a time; 409 otherwise)
    GET  /v1/fine_tuning/jobs[?limit&after]   list jobs, newest first
    GET  /v1/fine_tuning/jobs/{id}            one job
    GET  /v1/fine_tuning/jobs/{id}/events     its events, newest first
    POST /v1/fine_tuning/jobs/{id}/cancel     stop it (body optional)
    GET  /v1/models                           base checkpoints + finished runs
    GET  /health                              {"status": "ok" | "busy", "job", "phase", "started_at"}

`training_file` is a dataset id in the store's inbox/ (OpenAI file ids are
opaque strings, so no /v1/files upload exists). The router reads /health: while
it says "busy" the router holds the GPU for this container until
started_at + LAYA_MAX_HOLD_S (docs/plans/laya-trainer.md, Phase 3), so "busy"
lasts from the moment a job is accepted until its process has exited and its
record is final.

Standard library only (http.server): six endpoints need no framework, and every
package kept out of the image is one less entry in the hash lock.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import catalog as cat
from .contract import ExitCode, error_code
from .jobs import (DATASET_ID_RE, DEFAULT_SEED, JOB_ID_RE, SUFFIX_RE, TERMINAL, JobStore,
                   resolve_hyperparameters)

log = logging.getLogger("laya_trainer")

_STOPPED_MESSAGE = ("The trainer was stopped during this job (router hold cap, cache-down or "
                    "shutdown). Epoch checkpoints already written are in checkpoint/.")
MAX_BODY_BYTES = 1 << 20
# podman stop allows 10 s: SIGTERM -> SIGKILL for the job process, then a
# short wait for the watcher to write the final record.
STOP_GRACE_S = 4.0
FINALIZE_WAIT_S = 3.0
CANCEL_WAIT_S = 15.0        # how long a cancel call waits for the job record to be final


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str, *, code: str | None = None,
                 param: str | None = None, type_: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.status, self.message, self.code, self.param, self.type = status, message, code, param, type_

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.type,
                          "param": self.param, "code": self.code}}


class JobManager:
    """Owns the one running job process and every job record's final state."""

    def __init__(self, store_dir: Path, catalog_path: Path, *, device: str = "cuda",
                 timeout_s: float = 0, job_cmd: list[str] | None = None) -> None:
        self.store_dir = Path(store_dir)
        self.catalog_path = Path(catalog_path)
        self.store = JobStore(self.store_dir)
        self.device = device
        self.timeout_s = timeout_s
        self.job_cmd = job_cmd or [sys.executable, "-m", "laya_trainer.run_job"]
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._job_id: str | None = None
        self._started_at: int | None = None
        self._cancelled = False
        self._timed_out = False
        self._stopping = False
        # The job whose final record has been claimed. Exactly one writer
        # (the watcher's _finalize, or shutdown's fallback) may claim a job,
        # or a job finishing during a stop could be recorded twice -- and the
        # second write would land in a run directory already sealed.
        self._finalized: str | None = None
        self._done = threading.Event()
        self._done.set()

    # -- reads ---------------------------------------------------------------

    def catalog(self) -> list[dict]:
        try:
            return cat.load_catalog(self.catalog_path)
        except (OSError, cat.CatalogError) as e:
            raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, f"laya catalog unreadable: {e}",
                           type_="server_error") from None

    def health(self) -> dict:
        with self._lock:
            job_id, started_at = self._job_id, self._started_at
        if job_id is None:
            return {"status": "ok", "job": None, "phase": None, "started_at": None}
        job = self.store.read(job_id) or {}
        return {"status": "busy", "job": job_id,
                "phase": (job.get("devai") or {}).get("phase"), "started_at": started_at}

    def models(self) -> dict:
        data = [{"id": r["name"], "object": "model", "created": 0, "owned_by": "devai-laya",
                 "devai": {"kind": "base", "dir": cat.checkpoint_dirname(r)}}
                for r in self.catalog()]
        data += [{"id": j["fine_tuned_model"], "object": "model", "created": j["finished_at"],
                  "owned_by": "devai-laya", "devai": {"kind": "fine_tuned", "job": j["id"]}}
                 for j in self.store.list() if j["status"] == "succeeded" and j["fine_tuned_model"]]
        return {"object": "list", "data": data}

    def get(self, job_id: str) -> dict:
        job = self.store.read(job_id) if JOB_ID_RE.match(job_id) else None
        if job is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"no fine-tuning job {job_id!r}",
                           code="not_found", param="fine_tuning_job_id")
        return job

    # -- writes --------------------------------------------------------------

    def create(self, body: dict) -> dict:
        req = self._check_create(body)
        with self._lock:
            if self._job_id is not None:
                raise ApiError(HTTPStatus.CONFLICT,
                               f"job {self._job_id} is still running; one job at a time",
                               code="job_in_progress")
            job = self.store.create(**req)
            self._job_id, self._started_at = job["id"], job["devai"]["started_at"]
            self._cancelled = self._timed_out = False
            self._done.clear()
            try:
                self._proc = subprocess.Popen(
                    [*self.job_cmd, "--store", str(self.store_dir), "--catalog",
                     str(self.catalog_path), "--job", job["id"], "--device", self.device])
            except Exception as e:  # noqa: BLE001 -- never leave /health "busy" with no process
                self._job_id = self._started_at = None
                self._done.set()
                return self.store.fail(job["id"], error_code(ExitCode.UNEXPECTED),
                                       f"could not start the job process: {e}", int(ExitCode.UNEXPECTED))
            proc, job_id = self._proc, job["id"]
        log.info("job %s started (model %s, dataset %s)", job_id, req["model"], req["training_file"])
        threading.Thread(target=self._watch, args=(job_id, proc), daemon=True,
                         name=f"watch-{job_id}").start()
        return job

    def _check_create(self, body: dict) -> dict:
        rows = self.catalog()
        model = body.get("model")
        row = cat.find(rows, model) if isinstance(model, str) else None
        if row is None:
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           f"model {model!r} is not a laya base checkpoint; known: "
                           f"{[r['name'] for r in rows]}", param="model", code="model_not_found")
        base = self.store_dir / "base" / cat.checkpoint_dirname(row)
        if not base.is_dir():
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           f"base checkpoint {base.name} is not on disk; run "
                           f"`make model-pull NAME={row['name']}` on the host", param="model",
                           code="base_checkpoint_missing")
        ds = body.get("training_file")
        if not isinstance(ds, str) or not DATASET_ID_RE.match(ds):
            raise ApiError(HTTPStatus.BAD_REQUEST, "training_file must be a dataset id (ds-<12 hex>)",
                           param="training_file")
        if not (self.store_dir / "inbox" / ds).is_dir() and not (self.store_dir / "datasets" / ds).is_dir():
            raise ApiError(HTTPStatus.BAD_REQUEST, f"dataset {ds} is in neither inbox/ nor datasets/",
                           param="training_file", code="dataset_not_found")
        if body.get("validation_file") is not None:
            raise ApiError(HTTPStatus.BAD_REQUEST, "validation_file is not supported; the dataset "
                                                   "carries its own calib and heldout splits",
                           param="validation_file")
        try:
            hp = resolve_hyperparameters(body.get("hyperparameters"))
        except ValueError as e:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(e), param="hyperparameters") from None
        seed = body.get("seed", DEFAULT_SEED)
        if seed is None:
            seed = DEFAULT_SEED
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 31:
            raise ApiError(HTTPStatus.BAD_REQUEST, "seed must be an integer in [0, 2**31)", param="seed")
        suffix = body.get("suffix")
        if suffix is not None and (not isinstance(suffix, str) or not SUFFIX_RE.match(suffix)):
            raise ApiError(HTTPStatus.BAD_REQUEST, f"suffix must match {SUFFIX_RE.pattern}", param="suffix")
        meta = body.get("metadata")
        if meta is not None and (not isinstance(meta, dict) or len(meta) > 16 or not all(
                isinstance(k, str) and isinstance(v, str) and len(k) <= 64 and len(v) <= 512
                for k, v in meta.items())):
            raise ApiError(HTTPStatus.BAD_REQUEST, "metadata must be at most 16 string pairs",
                           param="metadata")
        return {"model": row["name"], "training_file": ds, "hyperparameters": hp, "seed": seed,
                "suffix": suffix, "metadata": meta, "base_dirname": cat.checkpoint_dirname(row)}

    def cancel(self, job_id: str) -> dict:
        job = self.get(job_id)
        with self._lock:
            running = self._job_id == job_id and self._proc is not None
            if running:
                self._cancelled = True
                self._proc.terminate()
        if not running:
            if job["status"] in TERMINAL:
                raise ApiError(HTTPStatus.CONFLICT, f"job {job_id} is already {job['status']}",
                               code="job_not_cancellable")
            raise ApiError(HTTPStatus.CONFLICT, f"job {job_id} is not running here",
                           code="job_not_cancellable")
        self._done.wait(CANCEL_WAIT_S)
        return self.get(job_id)

    def _watch(self, job_id: str, proc: subprocess.Popen) -> None:
        try:
            proc.wait(timeout=self.timeout_s or None)
        except subprocess.TimeoutExpired:
            with self._lock:
                self._timed_out = True
            self._stop(proc)
        self._finalize(job_id, proc.returncode)

    @staticmethod
    def _stop(proc: subprocess.Popen) -> None:
        proc.terminate()
        try:
            proc.wait(timeout=STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def _finalize(self, job_id: str, returncode: int) -> None:
        with self._lock:
            claimed = self._finalized != job_id
            self._finalized = job_id
            cancelled, timed_out, stopping = self._cancelled, self._timed_out, self._stopping
        try:
            if claimed:
                self._write_final(job_id, returncode, cancelled, timed_out, stopping)
        except Exception:  # noqa: BLE001 -- /health must never stay "busy" over a failed write
            log.exception("could not write the final record of job %s", job_id)
        finally:
            log.info("job %s finished (exit %s)", job_id, returncode)
            with self._lock:
                self._proc = self._job_id = self._started_at = None
            self._done.set()

    def _write_final(self, job_id: str, returncode: int, cancelled: bool, timed_out: bool,
                     stopping: bool) -> None:
        try:
            outcome = json.loads((self.store.job_dir(job_id) / "outcome.json").read_text())
        except (OSError, json.JSONDecodeError):
            outcome = None
        now = int(time.time())
        # A job that completed is recorded as completed, even if a cancel, a
        # timeout or a stop arrived in the same moment.
        if returncode == 0 and outcome and outcome.get("exit_code") == 0:
            self.store.update(job_id, status="succeeded", finished_at=now,
                              fine_tuned_model=outcome["fine_tuned_model"],
                              trained_tokens=outcome["trained_tokens"],
                              result_files=outcome["result_files"],
                              devai={"phase": "done", "exit_code": 0})
            self.store.add_event(job_id, "The job has successfully completed")
            cat.make_read_only(self.store.job_dir(job_id))
        elif cancelled:
            self.store.update(job_id, status="cancelled", finished_at=now,
                              devai={"phase": "cancelled", "exit_code": returncode})
            self.store.add_event(job_id, "Job cancelled; epoch checkpoints already written are in "
                                         "checkpoint/")
        elif timed_out:
            self.store.fail(job_id, error_code(ExitCode.TIMEOUT),
                            f"the job exceeded LAYA_JOB_TIMEOUT_S={self.timeout_s:g}", int(ExitCode.TIMEOUT))
        elif stopping:
            self.store.fail(job_id, "trainer_stopped", _STOPPED_MESSAGE, returncode)
        else:
            code = (outcome or {}).get("exit_code") or (returncode if returncode and returncode > 0
                                                         else int(ExitCode.UNEXPECTED))
            message = (outcome or {}).get("message") or f"the job process ended with {returncode}"
            self.store.fail(job_id, error_code(code), message, code)

    def shutdown(self) -> None:
        """SIGTERM from podman stop (router eviction, cache-down): stop the job
        and leave its record final, within podman's 10 s."""
        with self._lock:
            proc, job_id = self._proc, self._job_id
            self._stopping = True
        if proc is None:
            return
        log.warning("stopping job %s: the trainer is being stopped", job_id)
        self._stop(proc)
        if self._done.wait(FINALIZE_WAIT_S):
            return
        # The watcher did not get there in time. Write the record here --
        # unless the watcher has claimed it meanwhile.
        with self._lock:
            claimed = self._finalized != job_id
            self._finalized = job_id
        if claimed:
            try:
                self.store.fail(job_id, "trainer_stopped", _STOPPED_MESSAGE, None)
            except OSError:
                log.exception("could not record job %s as stopped", job_id)


def make_handler(manager: JobManager) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "devai-laya-trainer"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            log.debug("%s %s", self.address_string(), fmt % args)

        def _send(self, status: int, obj: dict) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body too large")
            raw = self.rfile.read(length) if length else b""
            if not raw.strip():
                return {}
            try:
                body = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid JSON body: {e}") from None
            if not isinstance(body, dict):
                raise ApiError(HTTPStatus.BAD_REQUEST, "the request body must be a JSON object")
            return body

        def _dispatch(self, method: str) -> None:
            url = urlsplit(self.path)
            parts = [p for p in url.path.split("/") if p]
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            try:
                body = self._body() if method == "POST" else {}
                status, obj = self._route(method, parts, query, body)
            except ApiError as e:
                status, obj = e.status, e.body()
            except Exception as e:  # noqa: BLE001 -- answer, never drop the connection
                log.exception("unhandled error on %s %s", method, self.path)
                status, obj = HTTPStatus.INTERNAL_SERVER_ERROR, ApiError(
                    HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(e).__name__}: {e}",
                    type_="server_error").body()
            self._send(status, obj)

        def _route(self, method: str, parts: list[str], query: dict, body: dict):
            if method == "GET" and parts == ["health"]:
                return HTTPStatus.OK, manager.health()
            if method == "GET" and parts == ["v1", "models"]:
                return HTTPStatus.OK, manager.models()
            if parts[:3] == ["v1", "fine_tuning", "jobs"]:
                rest = parts[3:]
                if method == "POST" and not rest:
                    return HTTPStatus.OK, manager.create(body)
                if method == "GET" and not rest:
                    return HTTPStatus.OK, _page(manager.store.list(), query)
                if method == "GET" and len(rest) == 1:
                    return HTTPStatus.OK, manager.get(rest[0])
                if method == "GET" and len(rest) == 2 and rest[1] == "events":
                    manager.get(rest[0])
                    return HTTPStatus.OK, _page(manager.store.events(rest[0]), query)
                if method == "POST" and len(rest) == 2 and rest[1] == "cancel":
                    return HTTPStatus.OK, manager.cancel(rest[0])
            raise ApiError(HTTPStatus.NOT_FOUND, f"no route for {method} {self.path}", code="not_found")

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

    return Handler


def _page(items: list[dict], query: dict) -> dict:
    """OpenAI list pagination: newest first, `after` = the last id seen."""
    after = query.get("after")
    if after:
        ids = [i["id"] for i in items]
        items = items[ids.index(after) + 1:] if after in ids else []
    try:
        limit = max(1, min(int(query.get("limit", 20)), 100))
    except ValueError:
        raise ApiError(HTTPStatus.BAD_REQUEST, "limit must be an integer", param="limit") from None
    return {"object": "list", "data": items[:limit], "has_more": len(items) > limit}


def serve(manager: JobManager, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(manager))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="http").start()
    return server


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="devai laya trainer controller")
    ap.add_argument("--store", type=Path, default=Path("/laya"))
    ap.add_argument("--catalog", type=Path, default=Path("/etc/devai/laya-models.yaml"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=11434)
    ap.add_argument("--device", default=os.environ.get("LAYA_TRAINER_DEVICE", "cuda"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr)
    manager = JobManager(args.store, args.catalog, device=args.device,
                         timeout_s=float(os.environ.get("LAYA_JOB_TIMEOUT_S", "0") or 0))
    for sub in ("inbox", "datasets", "runs"):
        (args.store / sub).mkdir(parents=True, exist_ok=True)
    stale = manager.store.reconcile()
    if stale:
        log.warning("marked %d job(s) left unfinished by a stopped trainer as failed: %s",
                    len(stale), ", ".join(stale))
    server = serve(manager, args.host, args.port)
    log.info("laya trainer listening on %s:%d (store %s, device %s)",
             args.host, args.port, args.store, args.device)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    while not stop.wait(1.0):
        pass
    manager.shutdown()
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
