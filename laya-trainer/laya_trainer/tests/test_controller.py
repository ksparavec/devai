"""The HTTP API and the job lifecycle, with a stand-in job process (no torch).

What the router relies on is pinned here: /health says "busy" from the moment
a job is accepted until its record is final, and "ok" otherwise.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from .. import catalog as cat
from ..controller import JobManager, serve
from .util import tmpdir, write_catalog

DS = "ds-0123456789ab"
ROW = {"name": "laya-fake", "default": True, "repo": "devai/laya-fake", "revision": "1" * 40,
       "subfolder": "", "files": {rel: {"sha256": "0" * 64, "size": 1} for rel in cat.REQUIRED_FILES}}


class ControllerTest(unittest.TestCase):

    def setUp(self) -> None:
        root = tmpdir(self)
        self.store = root / "laya"
        for sub in ("inbox", "datasets", "runs"):
            (self.store / sub).mkdir(parents=True)
        (self.store / "base" / cat.checkpoint_dirname(ROW)).mkdir(parents=True)
        (self.store / "inbox" / DS).mkdir()
        self.catalog = write_catalog(root / "laya-models.yaml", ROW)
        self.mode("succeed")
        self.start()

    def start(self, timeout_s: float = 0) -> None:
        self.manager = JobManager(self.store, self.catalog, device="cpu", timeout_s=timeout_s,
                                  job_cmd=[sys.executable, "-m", "laya_trainer.tests.fake_job"])
        self.server = serve(self.manager, "127.0.0.1", 0)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.manager.shutdown)

    def mode(self, mode: str) -> None:
        old = os.environ.get("FAKE_JOB_MODE")
        os.environ["FAKE_JOB_MODE"] = mode
        self.addCleanup(lambda: os.environ.pop("FAKE_JOB_MODE", None) if old is None
                        else os.environ.__setitem__("FAKE_JOB_MODE", old))

    def call(self, method: str, path: str, body=None, raw: bytes | None = None) -> tuple[int, dict]:
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def create(self, **over) -> tuple[int, dict]:
        return self.call("POST", "/v1/fine_tuning/jobs",
                         {"model": "laya-fake", "training_file": DS, "suffix": "polarity",
                          "hyperparameters": {"n_epochs": 1}, **over})

    def wait_idle(self, timeout: float = 30) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.call("GET", "/health")[1]["status"] == "ok":
                return
            time.sleep(0.05)
        self.fail("the trainer stayed busy")

    # -- the happy path ----------------------------------------------------------

    def test_idle_health(self) -> None:
        self.assertEqual(self.call("GET", "/health"),
                         (200, {"status": "ok", "job": None, "phase": None, "started_at": None}))

    def test_a_job_runs_to_success(self) -> None:
        status, job = self.create()
        self.assertEqual(status, 200)
        self.assertEqual(job["status"], "validating_files")
        self.assertEqual(job["hyperparameters"]["n_epochs"], 1)
        self.wait_idle()
        _, done = self.call("GET", f"/v1/fine_tuning/jobs/{job['id']}")
        self.assertEqual(done["status"], "succeeded")
        self.assertEqual(done["fine_tuned_model"], f"ft:laya-fake:devai:x:{job['id'][6:18]}")
        self.assertIsNotNone(done["finished_at"])
        run_dir = self.store / "runs" / job["id"]
        self.assertEqual(stat.S_IMODE(run_dir.stat().st_mode) & 0o222, 0, "succeeded run is sealed")
        ids = [m["id"] for m in self.call("GET", "/v1/models")[1]["data"]]
        self.assertEqual(ids, ["laya-fake", done["fine_tuned_model"]])

    def test_health_is_busy_for_the_whole_job(self) -> None:
        self.mode("sleep")
        _, job = self.create()
        _, h = self.call("GET", "/health")
        self.assertEqual((h["status"], h["job"]), ("busy", job["id"]))
        self.assertEqual(h["started_at"], job["devai"]["started_at"])

    def test_one_job_at_a_time(self) -> None:
        self.mode("sleep")
        self.create()
        status, err = self.create()
        self.assertEqual(status, 409)
        self.assertEqual(err["error"]["code"], "job_in_progress")

    def test_cancel_with_an_empty_body(self) -> None:
        self.mode("sleep")
        _, job = self.create()
        status, out = self.call("POST", f"/v1/fine_tuning/jobs/{job['id']}/cancel", raw=b"")
        self.assertEqual(status, 200)
        self.assertEqual(out["status"], "cancelled")
        self.assertEqual(self.call("GET", "/health")[1]["status"], "ok")
        status, err = self.call("POST", f"/v1/fine_tuning/jobs/{job['id']}/cancel")
        self.assertEqual((status, err["error"]["code"]), (409, "job_not_cancellable"))

    def test_events_and_listing(self) -> None:
        _, first = self.create()
        self.wait_idle()
        _, second = self.create()
        self.wait_idle()
        _, page = self.call("GET", "/v1/fine_tuning/jobs?limit=1")
        self.assertEqual([j["id"] for j in page["data"]], [second["id"]])
        self.assertTrue(page["has_more"])
        _, page = self.call("GET", f"/v1/fine_tuning/jobs?after={second['id']}")
        self.assertEqual([j["id"] for j in page["data"]], [first["id"]])
        _, events = self.call("GET", f"/v1/fine_tuning/jobs/{first['id']}/events")
        self.assertEqual(events["data"][0]["message"], "The job has successfully completed")
        self.assertEqual(events["data"][-1]["message"], "Job created")

    # -- failures ----------------------------------------------------------------

    def test_a_contract_failure_carries_its_code(self) -> None:
        self.mode("fail3")
        _, job = self.create()
        self.wait_idle()
        _, done = self.call("GET", f"/v1/fine_tuning/jobs/{job['id']}")
        self.assertEqual(done["status"], "failed")
        self.assertEqual(done["error"]["code"], "dataset_contract_violation")
        self.assertEqual(done["error"]["message"], "bad dataset")
        self.assertEqual(done["devai"]["exit_code"], 3)

    def test_a_crash_without_outcome_is_still_final(self) -> None:
        self.mode("crash")
        _, job = self.create()
        self.wait_idle()
        _, done = self.call("GET", f"/v1/fine_tuning/jobs/{job['id']}")
        self.assertEqual((done["status"], done["error"]["code"]), ("failed", "exit_9"))

    def test_a_timeout_is_exit_124(self) -> None:
        self.server.shutdown()
        self.start(timeout_s=0.5)
        self.mode("sleep")
        _, job = self.create()
        self.wait_idle()
        _, done = self.call("GET", f"/v1/fine_tuning/jobs/{job['id']}")
        self.assertEqual((done["error"]["code"], done["devai"]["exit_code"]), ("timeout", 124))

    def test_shutdown_leaves_the_record_final(self) -> None:
        self.mode("sleep")
        _, job = self.create()
        self.manager.shutdown()
        done = json.loads((self.store / "runs" / job["id"] / "job.json").read_text())
        self.assertEqual((done["status"], done["error"]["code"]), ("failed", "trainer_stopped"))

    def test_a_job_finishing_during_a_stop_is_recorded_as_succeeded(self) -> None:
        """Stop and completion race: the record must say what happened, once."""
        self.mode("finish_on_term")
        _, job = self.create()
        time.sleep(0.5)  # let the fake job install its handler
        self.manager.shutdown()
        done = json.loads((self.store / "runs" / job["id"] / "job.json").read_text())
        self.assertEqual(done["status"], "succeeded")
        self.assertIsNone(done["error"])

    # -- request validation ------------------------------------------------------

    def test_bad_requests(self) -> None:
        cases = [
            ({"model": "gpt-4o"}, "model"),
            ({"training_file": "file-abc"}, "training_file"),
            ({"training_file": "ds-ffffffffffff"}, "training_file"),
            ({"validation_file": DS}, "validation_file"),
            ({"hyperparameters": {"n_epochs": 0}}, "hyperparameters"),
            ({"suffix": "Bad Suffix!"}, "suffix"),
            ({"seed": -1}, "seed"),
            ({"metadata": {"k": 1}}, "metadata"),
        ]
        for over, param in cases:
            status, err = self.create(**over)
            self.assertEqual((status, err["error"]["param"]), (400, param), over)
        self.assertEqual(self.call("GET", "/health")[1]["status"], "ok")

    def test_a_missing_base_names_the_pull_command(self) -> None:
        (self.store / "base" / cat.checkpoint_dirname(ROW)).rmdir()
        status, err = self.create()
        self.assertEqual((status, err["error"]["code"]), (400, "base_checkpoint_missing"))
        self.assertIn("make model-pull NAME=laya-fake", err["error"]["message"])

    def test_invalid_json_and_unknown_routes(self) -> None:
        self.assertEqual(self.call("POST", "/v1/fine_tuning/jobs", raw=b"{nope")[0], 400)
        self.assertEqual(self.call("GET", "/v1/fine_tuning/jobs/ftjob-" + "0" * 24)[0], 404)
        self.assertEqual(self.call("GET", "/v1/fine_tuning/jobs/../../x")[0], 404)
        self.assertEqual(self.call("GET", "/v1/chat/completions")[0], 404)


if __name__ == "__main__":
    unittest.main()
