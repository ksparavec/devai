"""`make laya-check`: the reference run of teacher/trainer coordination.

scripts/laya-check.py drives one known laya job (ds-a6c9c8248242, kept in the
laya store since 2026-09-25) through the router's trainer port and checks every
hand-off between the teacher and the trainer. What is pinned here:

- the verdicts: each failure mode (no swap, no hold -- a 200 while the job
  runs means the router evicted it --, a failed or timed-out job, a changed
  data path, a parity miss, aiagent's rejection or a golden mismatch, a
  teacher that did not come back as it was) fails exactly its own check;
- the job driver records what the router answered and decides nothing: no
  teacher string means no job request at all (nothing is evicted), the hold
  probe goes out the first time the job is seen running, a job past its
  timeout is cancelled, and the teacher is always warmed back;
- the teacher string rebuilt from /health relaunches what was loaded;
- the dataset is rewritten only when it is gone, and a rewrite under another
  id stops the run before the GPU is touched;
- a passing run deletes ONLY its own run directory; a failing run, or
  --keep-run, keeps it; nothing outside runs/<its job id> can be removed;
- the containers: aiagent's steps without network or GPU and with /laya
  read-only (inbox/ writable only for the writer); the driver on devai-net;
- the Makefile target and its knobs.

Stdlib unittest. No container is started: podman is faked, and so is HTTP.
The live run is `make laya-check` (docs/laya-trainer.md).
"""

from __future__ import annotations

import ast
import importlib.util
import io
import itertools
import json
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lc = _load("laya_check_host", REPO_ROOT / "scripts" / "laya-check.py")
dj = _load("laya_check_drive", REPO_ROOT / "scripts" / "laya_check" / "drive_job.py")

MAKEFILE = (REPO_ROOT / "Makefile").read_text()
JOB = "ftjob-" + "ab" * 12
TEACHER = "Qwen3.8-27B-MTP-devai-NVFP4::nothink::mtp@118784"


def passing_drive() -> dict:
    return {
        "teacher": TEACHER,
        "teacher_health_before": {"running": True, "current_model": "Qwen3.8-27B-MTP-devai-NVFP4",
                                  "current_context": 118784, "current_spec": "mtp/k=3"},
        "post": {"status": 200, "seconds": 13.6},
        "job_id": JOB,
        "hold": {"status": 503, "retry_after": "30", "code": "gpu_held_by_job"},
        "job": {"status": "succeeded", "devai": {"exit_code": 0}, "error": None},
        "warm": {"status": 200, "seconds": 126.6, "attempts": 1},
        "teacher_health_after": {"running": True, "current_model": "Qwen3.8-27B-MTP-devai-NVFP4",
                                 "current_context": 118784, "current_spec": "mtp/k=3"},
    }


def passing_manifest() -> dict:
    return {
        "dataset": {"id": "ds-a6c9c8248242",
                    "counts": {"train": 50, "calib": 10, "heldout": 12, "pool": 8}},
        "metrics": {"trained_tokens": 8172},
        "parity": {"n": 12, "max_abs_prob_diff": 5.7e-07, "argmax_agreement": "12/12",
                   "tolerance": 0.001},
        "golden": {"n": 12},
        "trainer": {"peak_vram_gib": 3.17, "timings": {"training": 4.1}},
    }


def passing_verify() -> dict:
    return {"accepted": True, "artifact_id": "0" * 64, "golden": 12, "error": None}


def failed(checks: list[dict]) -> set[str]:
    return {c["name"] for c in checks if not c["ok"]}


class VerdictTest(unittest.TestCase):
    """Each failure mode fails exactly its own check."""

    def evaluate(self, drive=None, manifest=None, verify=None):
        return lc.evaluate(drive or passing_drive(), manifest or passing_manifest(),
                           verify or passing_verify())

    def test_the_reference_run_passes(self):
        checks = self.evaluate()
        self.assertEqual(failed(checks), set())
        self.assertEqual([c["name"] for c in checks],
                         ["swap", "hold", "job", "data", "parity", "aiagent", "teacher"])

    def test_a_refused_job_request_fails_the_swap(self):
        d = passing_drive()
        d["post"] = {"status": 409, "seconds": 0.1}
        self.assertEqual(failed(self.evaluate(drive=d)), {"swap"})

    def test_a_swap_with_no_resident_teacher_evicted_nothing(self):
        # The job request then evicts nothing: the path is not exercised.
        for before in ({"running": False, "current_model": ""}, {}):
            d = passing_drive()
            d["teacher_health_before"] = before
            self.assertEqual(failed(self.evaluate(drive=d)), {"swap"}, before)

    def test_a_teacher_adopted_with_its_model_unknown_is_resident(self):
        # A restarted router adopts the running engine without knowing its
        # model; the job request evicts it all the same (seen live 2026-09-25).
        d = passing_drive()
        d["teacher_health_before"] = {"running": True, "current_model": "",
                                      "current_context": 0, "current_spec": "off"}
        self.assertEqual(failed(self.evaluate(drive=d)), set())

    def test_a_teacher_served_during_the_job_fails_the_hold(self):
        # A 200 here means the router evicted a running training job.
        d = passing_drive()
        d["hold"] = {"status": 200, "retry_after": None, "code": None}
        self.assertEqual(failed(self.evaluate(drive=d)), {"hold"})

    def test_a_503_without_the_job_code_or_retry_after_fails_the_hold(self):
        for bad in ({"code": "backend_unavailable"}, {"retry_after": None}):
            d = passing_drive()
            d["hold"].update(bad)
            self.assertEqual(failed(self.evaluate(drive=d)), {"hold"}, bad)

    def test_a_job_never_seen_running_fails_the_hold(self):
        d = passing_drive()
        d["hold"] = None
        self.assertEqual(failed(self.evaluate(drive=d)), {"hold"})

    def test_a_failed_or_timed_out_job_fails_the_job(self):
        d = passing_drive()
        d["job"] = {"status": "failed", "devai": {"exit_code": 5}, "error": {"code": "oom"}}
        self.assertEqual(failed(self.evaluate(drive=d)), {"job"})
        d = passing_drive()
        d["timed_out"] = True
        self.assertEqual(failed(self.evaluate(drive=d)), {"job"})

    def test_a_driver_exception_or_poll_error_is_reported_with_the_job(self):
        d = passing_drive()
        d["job"] = {"status": "running", "devai": {}}
        d["exception"] = "AttributeError: boom"
        d["poll_error"] = {"status": 503, "body": "not running"}
        checks = {c["name"]: c for c in self.evaluate(drive=d)}
        self.assertFalse(checks["job"]["ok"])
        self.assertIn("AttributeError: boom", checks["job"]["detail"])
        self.assertIn("poll_error", checks["job"]["detail"])

    def test_a_changed_data_path_fails_the_data_check(self):
        for path, value in ((("metrics", "trained_tokens"), 8000),
                            (("dataset", "id"), "ds-000000000000"),
                            (("dataset", "counts"), {"train": 49})):
            m = passing_manifest()
            m[path[0]][path[1]] = value
            self.assertEqual(failed(self.evaluate(manifest=m)), {"data"}, path)

    def test_a_parity_miss_fails_parity(self):
        m = passing_manifest()
        m["parity"]["max_abs_prob_diff"] = 0.002
        self.assertEqual(failed(self.evaluate(manifest=m)), {"parity"})
        m = passing_manifest()
        m["parity"]["argmax_agreement"] = "11/12"
        self.assertEqual(failed(self.evaluate(manifest=m)), {"parity"})

    def test_aiagents_rejection_or_a_golden_mismatch_fails_aiagent(self):
        self.assertEqual(failed(self.evaluate(verify={"accepted": False, "error": "binds"})),
                         {"aiagent"})
        self.assertEqual(failed(self.evaluate(verify={**passing_verify(), "golden": 11})),
                         {"aiagent"})

    def test_a_teacher_not_back_as_it_was_fails_the_teacher_check(self):
        for key, value in (("current_model", "other"), ("current_context", 65536),
                           ("current_spec", "off")):
            d = passing_drive()
            d["teacher_health_after"][key] = value
            self.assertEqual(failed(self.evaluate(drive=d)), {"teacher"}, key)
        d = passing_drive()
        d["warm"] = {"status": 503, "seconds": 1.0}
        self.assertEqual(failed(self.evaluate(drive=d)), {"teacher"})

    def test_a_run_that_stopped_early_fails_everything_after_it(self):
        d = {"teacher": None, "error": "no teacher string"}
        checks = lc.evaluate(d, None, None)
        self.assertEqual(failed(checks), {c["name"] for c in checks})


class TeacherStringTest(unittest.TestCase):

    def test_parse_teacher(self):
        cases = {
            TEACHER: ("Qwen3.8-27B-MTP-devai-NVFP4", 118784, True),
            "M@32768": ("M", 32768, False),
            "M": ("M", None, False),
            "M::nomtp@1024": ("M", 1024, False),
            "M::mtp": ("M", None, True),
            "qwen3.5:9b-q8_0": ("qwen3.5:9b-q8_0", None, False),
            # aiagent/litellm append their reasoning suffix after @<ctx>
            "M@1024::nothink": ("M", 1024, False),
            "M::mtp@1024::nothink": ("M", 1024, True),
            "M@118784::nothink::mtp": ("M", 118784, True),
            "M::mtp::nothink@118784": ("M", 118784, True),
            # the leftmost of a repeated kind wins, as in peelControlSuffixes
            "M::nomtp::mtp": ("M", None, False),
            # not suffixes the router peels
            "M@0": ("M@0", None, False),
            "M::custom": ("M::custom", None, False),
        }
        for name, want in cases.items():
            self.assertEqual(lc.parse_teacher(name), want, name)

    def test_the_string_rebuilt_from_health_relaunches_what_was_loaded(self):
        h = {"running": True, "current_model": "M", "current_context": 118784,
             "current_spec": "mtp/k=3"}
        self.assertEqual(dj.teacher_from_health(h), "M::mtp@118784")
        self.assertEqual(lc.parse_teacher(dj.teacher_from_health(h)), ("M", 118784, True))
        self.assertEqual(dj.teacher_from_health({**h, "current_spec": "off"}), "M@118784")
        self.assertEqual(dj.teacher_from_health({**h, "current_context": 0}), "M::mtp")

    def test_nothing_loaded_gives_no_string(self):
        # The router adopts a running engine at boot with the model unknown.
        self.assertIsNone(dj.teacher_from_health({"running": True, "current_model": ""}))
        self.assertIsNone(dj.teacher_from_health({"running": False, "current_model": "M"}))
        self.assertIsNone(dj.teacher_from_health({}))


HOLD = (503, {"Retry-After": "30"}, {"error": {"code": "gpu_held_by_job"}})
OK = (200, {}, {"choices": []})


class FakeRouter:
    """Stands in for drive_job.call: scripted answers, recorded requests.

    `states`: what successive job GETs report (the last one repeats); an int
    instead of a status name answers that HTTP status instead of the job.
    `chats`: successive answers to teacher chat requests (the last repeats).
    """

    def __init__(self, states, *, post_status=200, health=None, chats=(HOLD, OK),
                 runner_status="ok"):
        self.states = list(states)
        self.post_status = post_status
        self.health = health or {"running": True, "current_model": "M",
                                 "current_context": 1024, "current_spec": "off"}
        self.chat_answers = list(chats)
        self.runner_status = runner_status
        self.calls = []

    def __call__(self, method, url, body=None, timeout=60):
        self.calls.append((method, url, body))
        if url.endswith(":11438/health"):
            return 200, {}, {"job_runner": {"status": self.runner_status, "job": "ftjob-other"}}
        if url.endswith("/health"):
            return 200, {}, self.health
        if method == "POST" and url.endswith("/v1/fine_tuning/jobs"):
            return self.post_status, {}, {"id": JOB, "status": "validating_files"}
        if url.endswith(f"/jobs/{JOB}"):
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            if isinstance(state, int):
                return state, {}, {"error": "laya-trainer is not running"}
            return 200, {}, {"id": JOB, "status": state, "devai": {"phase": state}}
        if "/events" in url:
            return 200, {}, {"data": [{"created_at": 2, "message": "done"},
                                      {"created_at": 1, "message": "Job created"}]}
        if url.endswith("/cancel"):
            return 200, {}, {}
        if url.endswith("/chat/completions"):
            ans = self.chat_answers.pop(0) if len(self.chat_answers) > 1 else self.chat_answers[0]
            return ans
        raise AssertionError(f"unexpected {method} {url}")

    def sent(self, method, suffix):
        return [c for c in self.calls if c[0] == method and c[1].endswith(suffix)]

    @property
    def chats(self):
        return len(self.sent("POST", "/chat/completions"))


CFG = {"router": "http://r", "trainer_port": 11438, "teacher_port": 11437, "teacher": "",
       "base": "laya-multilingual", "dataset": "ds-a6c9c8248242", "suffix": "laya-check",
       "hyperparameters": {}, "seed": 42, "job_timeout_s": 60, "launch_timeout_s": 5}


class DriverTest(unittest.TestCase):

    def drive(self, router, **cfg):
        with mock.patch.object(dj, "call", router), mock.patch.object(dj.time, "sleep"), \
                mock.patch.object(dj, "log"):
            return dj.main({**CFG, **cfg})

    def test_the_reference_sequence(self):
        r = FakeRouter(["queued", "running", "running", "succeeded"])
        out = self.drive(r)
        self.assertEqual(out["teacher"], "M@1024")
        self.assertEqual(out["hold"], {"status": 503, "retry_after": "30",
                                       "code": "gpu_held_by_job"})
        self.assertEqual(out["job"]["status"], "succeeded")
        self.assertEqual(out["warm"], {"status": 200, "seconds": out["warm"]["seconds"],
                                       "attempts": 1})
        self.assertEqual(out["events"], [[1, "Job created"], [2, "done"]])
        self.assertEqual(r.chats, 2, "one hold probe, one warm-up")
        post = r.sent("POST", "/v1/fine_tuning/jobs")[0][2]
        self.assertEqual((post["model"], post["training_file"], post["seed"]),
                         ("laya-multilingual", "ds-a6c9c8248242", 42))

    def test_the_hold_probe_follows_the_first_running_poll(self):
        r = FakeRouter(["queued", "running", "running", "running", "succeeded"])
        self.drive(r)
        kinds = ["chat" if c[1].endswith("/chat/completions") else
                 "poll" if c[1].endswith(f"/jobs/{JOB}") else "other" for c in r.calls]
        first_chat = kinds.index("chat")
        self.assertEqual(kinds[:first_chat].count("poll"), 2, "queued, then running -> probe")

    def test_the_warm_up_waits_out_the_routers_cached_hold(self):
        # The router keeps a hold verdict for Retry-After seconds after the job
        # has ended; a single warm-up there would leave the teacher unloaded.
        r = FakeRouter(["running", "succeeded"], chats=(HOLD, HOLD, HOLD, OK))
        with mock.patch.object(dj, "call", r), mock.patch.object(dj, "log"), \
                mock.patch.object(dj.time, "sleep") as sleep:
            out = dj.main({**CFG, "launch_timeout_s": 900})
        self.assertEqual((out["warm"]["status"], out["warm"]["attempts"]), (200, 3))
        self.assertEqual([c.args[0] for c in sleep.call_args_list if c.args[0] == 31], [31, 31])

    def test_the_warm_up_gives_up_after_its_budget(self):
        r = FakeRouter(["running", "succeeded"], chats=(HOLD,))
        with mock.patch.object(dj, "call", r), mock.patch.object(dj, "log"), \
                mock.patch.object(dj.time, "sleep"), \
                mock.patch.object(dj.time, "monotonic", side_effect=itertools.count(0, 10)):
            out = dj.main({**CFG, "launch_timeout_s": 100})
        self.assertEqual(out["warm"]["status"], 503)
        self.assertLess(out["warm"]["attempts"], 20)

    def test_no_teacher_means_no_job_request(self):
        r = FakeRouter(["succeeded"], health={"running": True, "current_model": ""})
        out = self.drive(r)
        self.assertIn("no teacher string", out["error"])
        self.assertEqual(r.sent("POST", "/v1/fine_tuning/jobs"), [], "nothing may be evicted")

    def test_a_busy_trainer_means_no_job_request(self):
        r = FakeRouter(["succeeded"], runner_status="busy")
        out = self.drive(r)
        self.assertIn("already running", out["error"])
        self.assertEqual(r.sent("POST", "/v1/fine_tuning/jobs"), [])
        self.assertEqual(r.chats, 0)

    def test_a_given_teacher_wins_over_health(self):
        out = self.drive(FakeRouter(["running", "succeeded"]), teacher=TEACHER)
        self.assertEqual(out["teacher"], TEACHER)

    def test_a_409_job_request_leaves_the_teacher_alone(self):
        # Another job holds the trainer; nothing of ours evicted the teacher.
        r = FakeRouter(["running"], post_status=409)
        out = self.drive(r)
        self.assertEqual(out["post"]["status"], 409)
        self.assertEqual(r.chats, 0)
        self.assertNotIn("warm", out)

    def test_a_job_request_the_trainer_refuses_still_restores_the_teacher(self):
        # The router evicted the teacher before the controller said no.
        r = FakeRouter(["running"], post_status=400, chats=(OK,))
        out = self.drive(r)
        self.assertEqual((out["post"]["status"], out["warm"]["status"]), (400, 200))

    def test_a_job_past_its_timeout_is_cancelled_and_the_teacher_still_warmed(self):
        r = FakeRouter(["running"])
        out = self.drive(r, job_timeout_s=-1)
        self.assertTrue(out["timed_out"])
        self.assertEqual(len(r.sent("POST", "/cancel")), 1)
        self.assertEqual(out["warm"]["status"], 200)

    def test_the_hold_probe_goes_out_once(self):
        r = FakeRouter(["running"] * 5 + ["succeeded"])
        out = self.drive(r)
        self.assertEqual(r.chats, 2)
        self.assertEqual(out["hold"]["status"], 503)

    def test_a_teacher_served_during_the_job_stops_polling(self):
        # A 200 means the router evicted the running job: nothing to wait for.
        r = FakeRouter(["running"], chats=(OK,))
        out = self.drive(r, job_timeout_s=10 ** 6)
        self.assertEqual(out["hold"]["status"], 200)
        self.assertIn("evicted", out["poll_error"]["body"])
        self.assertEqual(out["warm"]["status"], 200)

    def test_a_trainer_that_stops_answering_ends_the_polling(self):
        r = FakeRouter(["running", 503])
        out = self.drive(r, job_timeout_s=10 ** 6)
        self.assertEqual(out["poll_error"]["status"], 503)
        self.assertEqual(len(r.sent("GET", f"/jobs/{JOB}")), 1 + dj.MAX_POLL_FAILURES)
        self.assertNotIn("timed_out", out)

    def test_a_string_error_body_does_not_crash_the_driver(self):
        # The router writes some refusals as {"error": "<text>"}.
        bad = (404, {}, {"error": "unknown model \"X\" for vllm-devai"})
        r = FakeRouter(["running", "succeeded"], chats=(bad,))
        out = self.drive(r)
        self.assertEqual(out["hold"], {"status": 404, "retry_after": None, "code": None})
        self.assertEqual(out["warm"]["status"], 404)
        self.assertEqual(out["job_id"], JOB)

    def test_an_exception_after_the_job_request_still_restores_the_teacher(self):
        r = FakeRouter(["running", "succeeded"], chats=(OK,))

        def flaky(method, url, body=None, timeout=60):
            if url.endswith(f"/jobs/{JOB}"):
                raise RuntimeError("boom")
            return r(method, url, body, timeout)
        out = self.drive(flaky)
        self.assertEqual(out["exception"], "RuntimeError: boom")
        self.assertEqual((out["job_id"], out["warm"]["status"]), (JOB, 200))
        self.assertIn("teacher_health_after", out)

    def test_error_code_reads_only_the_object_shape(self):
        self.assertEqual(dj.error_code({"error": {"code": "gpu_held_by_job"}}), "gpu_held_by_job")
        for body in ({"error": "text"}, {"error": None}, {}, [], "x", None):
            self.assertIsNone(dj.error_code(body), body)

    def test_the_driver_is_stdlib_only(self):
        # It runs in whatever image has a python3 on devai-net.
        tree = ast.parse((REPO_ROOT / "scripts/laya_check/drive_job.py").read_text())
        mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)
                for a in n.names}
        mods |= {n.module.split(".")[0] for n in ast.walk(tree)
                 if isinstance(n, ast.ImportFrom) and n.module}
        self.assertLessEqual(mods, {"json", "sys", "time", "urllib"})


def _read_only_run(store: Path, job: str) -> Path:
    d = store / "runs" / job
    (d / "checkpoint").mkdir(parents=True)
    for f in (d / "manifest.json", d / "checkpoint" / "model.safetensors"):
        f.write_text("x")
        f.chmod(0o444)
    for p in (d / "checkpoint", d):
        p.chmod(0o555)
    return d


class RemoveRunTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Path(self.tmp.name)

    def tearDown(self):
        for p in self.store.rglob("*"):
            if p.is_dir():
                p.chmod(stat.S_IRWXU)
        self.tmp.cleanup()

    def test_only_this_runs_directory_goes(self):
        reference = _read_only_run(self.store, "ftjob-f91b6d747c913af35c209ad9")
        mine = _read_only_run(self.store, JOB)
        lc.remove_run(self.store, JOB)
        self.assertFalse(mine.exists())
        self.assertTrue((reference / "manifest.json").is_file())

    def test_anything_but_a_job_id_is_refused(self):
        _read_only_run(self.store, JOB)
        for bad in ("", "..", "../runs", f"{JOB}/..", "ftjob-XYZ", "datasets"):
            with self.assertRaises(ValueError, msg=bad):
                lc.remove_run(self.store, bad)
        self.assertTrue((self.store / "runs" / JOB).is_dir())

    def test_a_symlinked_run_is_not_followed(self):
        # runs/<job> pointing at the kept reference run must not delete it.
        reference = _read_only_run(self.store, "ftjob-f91b6d747c913af35c209ad9")
        (self.store / "runs" / JOB).symlink_to(reference)
        with self.assertRaises(ValueError):
            lc.remove_run(self.store, JOB)
        self.assertTrue((reference / "manifest.json").is_file())


BASE = "laya-multilingual@55cf4c4ebb4e"
DATASET_MANIFEST = {"base_checkpoint": {"name": "laya-multilingual",
                                        "revision": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"}}


def _stage_dataset(store: Path, sub: str = "datasets", ds: str = "ds-a6c9c8248242") -> None:
    d = store / sub / ds
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(DATASET_MANIFEST))


class FakePodman:
    """Answers the host script's container calls from scripted results."""

    def __init__(self, store: Path, *, drive=None, verify=None, writer_id=None,
                 missing_image=None, job_json=None):
        self.store = store
        self.drive = passing_drive() if drive is None else drive
        self.verify = passing_verify() if verify is None else verify
        self.writer_id = writer_id
        self.missing_image = missing_image
        self.job_json = job_json
        self.calls = []

    def kind(self, argv):
        if argv[:3] == ["podman", "image", "exists"]:
            return "exists"
        if "--network" in argv and argv[argv.index("--network") + 1] == lc.NETWORK:
            return "drive"
        return "writer" if any(a.endswith(":/laya/inbox") for a in argv) else "verify"

    def __call__(self, argv, stdin):
        k = self.kind(argv)
        self.calls.append((k, argv, stdin))
        if k == "exists":
            return (1 if argv[3] == self.missing_image else 0), ""
        if k == "writer":
            _stage_dataset(self.store, "inbox", self.writer_id)
            return 0, json.dumps({"dataset": self.writer_id, "counts": {}}) + "\n"
        if k == "drive":
            if self.drive.get("job_id"):
                d = self.store / "runs" / self.drive["job_id"]
                d.mkdir(parents=True)
                (d / "manifest.json").write_text(json.dumps(passing_manifest()))
                if self.job_json is not None:
                    (d / "job.json").write_text(json.dumps(self.job_json))
            return 0, "progress\n" + json.dumps(self.drive) + "\n"
        return 0, json.dumps(self.verify) + "\n"

    def kinds(self):
        return [c[0] for c in self.calls]


class MainTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Path(self.tmp.name)
        _stage_dataset(self.store)
        (self.store / "base" / BASE).mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, podman, *args):
        with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            return lc.main(list(args), runner=podman, store=self.store)

    def test_a_passing_run_deletes_its_run_directory(self):
        p = FakePodman(self.store)
        self.assertEqual(self.run_main(p), 0)
        self.assertEqual(p.kinds(), ["exists", "exists", "drive", "verify"])
        self.assertFalse((self.store / "runs" / JOB).exists())

    def test_keep_run_keeps_it(self):
        p = FakePodman(self.store)
        self.assertEqual(self.run_main(p, "--keep-run"), 0)
        self.assertTrue((self.store / "runs" / JOB / "manifest.json").is_file())

    def test_a_failing_run_keeps_it(self):
        drive = passing_drive()
        drive["hold"] = {"status": 200, "retry_after": None, "code": None}
        p = FakePodman(self.store, drive=drive)
        self.assertEqual(self.run_main(p), 1)
        self.assertTrue((self.store / "runs" / JOB).is_dir())

    def test_a_missing_image_stops_before_anything_runs(self):
        p = FakePodman(self.store, missing_image="devai-lab-gpu")
        self.assertEqual(self.run_main(p), 2)
        self.assertNotIn("drive", p.kinds())

    def test_a_present_dataset_is_not_rewritten(self):
        p = FakePodman(self.store)
        self.run_main(p)
        self.assertNotIn("writer", p.kinds())

    def test_a_dataset_only_in_the_inbox_is_not_rewritten(self):
        shutil.rmtree(self.store / "datasets")
        _stage_dataset(self.store, "inbox")
        p = FakePodman(self.store)
        self.assertEqual(self.run_main(p), 0)
        self.assertNotIn("writer", p.kinds())

    def test_a_missing_base_stops_before_the_gpu(self):
        # The trainer would refuse the job only after the teacher was evicted.
        (self.store / "base" / BASE).rmdir()
        p = FakePodman(self.store)
        self.assertEqual(self.run_main(p), 1)
        self.assertNotIn("drive", p.kinds())

    def test_a_job_polling_lost_is_read_from_the_volume(self):
        drive = passing_drive()
        drive["job"] = {"status": "running", "devai": {}}
        drive["poll_error"] = {"status": 503, "body": "not running"}
        final = {"status": "failed", "devai": {"exit_code": None},
                 "error": {"code": "trainer_stopped"}}
        p = FakePodman(self.store, drive=drive, job_json=final)
        out = io.StringIO()
        with mock.patch("sys.stdout", out), mock.patch("sys.stderr"):
            rc = lc.main([], runner=p, store=self.store)
        self.assertEqual(rc, 1)
        self.assertIn("status=failed", out.getvalue())
        self.assertIn("read from the volume", out.getvalue())
        self.assertTrue((self.store / "runs" / JOB).is_dir(), "a failing run is kept")

    def test_a_gone_dataset_is_rewritten_by_aiagents_writer(self):
        shutil.rmtree(self.store / "datasets")
        p = FakePodman(self.store, writer_id=lc.REFERENCE["dataset"])
        self.assertEqual(self.run_main(p), 0)
        self.assertEqual(p.kinds(), ["exists", "exists", "writer", "drive", "verify"])

    def test_a_rewrite_under_another_id_stops_before_the_gpu(self):
        shutil.rmtree(self.store / "datasets")
        p = FakePodman(self.store, writer_id="ds-000000000000")
        self.assertEqual(self.run_main(p), 1)
        self.assertNotIn("drive", p.kinds(), "the teacher must not be evicted")

    def test_the_driver_gets_the_reference_job(self):
        p = FakePodman(self.store)
        self.run_main(p, "--teacher", TEACHER, "--teacher-port", "11435")
        argv = next(c[1] for c in p.calls if c[0] == "drive")
        cfg = json.loads(argv[-1])
        self.assertEqual((cfg["teacher"], cfg["teacher_port"], cfg["trainer_port"]),
                         (TEACHER, 11435, 11438))
        self.assertEqual((cfg["dataset"], cfg["base"], cfg["seed"]),
                         ("ds-a6c9c8248242", "laya-multilingual", 42))
        self.assertEqual(cfg["hyperparameters"], lc.HYPERPARAMETERS)

    def test_the_verifier_reads_this_run_and_the_reference_dataset(self):
        p = FakePodman(self.store)
        self.run_main(p)
        argv = next(c[1] for c in p.calls if c[0] == "verify")
        self.assertEqual(argv[-2:], [f"/laya/runs/{JOB}", "/laya/datasets/ds-a6c9c8248242"])

    def test_a_driver_crash_is_a_failure_not_a_pass(self):
        p = FakePodman(self.store)

        def runner(argv, stdin):
            return (1, "") if p.kind(argv) == "drive" else p(argv, stdin)
        self.assertEqual(self.run_main(runner), 1)


class ContainerTest(unittest.TestCase):

    def test_aiagent_runs_without_network_or_gpu_and_reads_laya_only(self):
        argv = lc.lab_argv("devai-lab-gpu", Path("/s"), ["x"], write_inbox=False)
        self.assertEqual(argv[argv.index("--network") + 1], "none")
        self.assertIn("--userns=keep-id", argv)
        self.assertNotIn("--device", argv)
        mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
        self.assertEqual(mounts, ["/s:/laya:ro"])

    def test_only_the_writer_gets_the_inbox(self):
        argv = lc.lab_argv("devai-lab-gpu", Path("/s"), ["/laya"], write_inbox=True)
        mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
        self.assertEqual(mounts, ["/s:/laya:ro", "/s/inbox:/laya/inbox"])

    def test_aiagents_python_is_the_one_its_launcher_names(self):
        argv = lc.lab_argv("img", Path("/s"), ["a", "b"], write_inbox=False)
        self.assertEqual(argv[-5:], ["-c", lc.AIAGENT_PYTHON, "sh", "a", "b"])
        self.assertIn("/opt/aiagent/bin/aiagent", lc.AIAGENT_PYTHON)
        self.assertIn(' -I - "$@"', lc.AIAGENT_PYTHON)

    def test_the_driver_runs_on_devai_net(self):
        argv = lc.driver_argv("trainer:img", {"a": 1})
        self.assertEqual(argv[argv.index("--network") + 1], "devai-net")
        self.assertEqual(argv[-4:], ["trainer:img", "-I", "-", '{"a": 1}'])

    def test_the_in_container_scripts_compile(self):
        for name in ("drive_job.py", "write_dataset.py", "verify_artifact.py"):
            compile((lc.HERE / name).read_text(), name, "exec")


class MakefileTest(unittest.TestCase):

    def test_target_and_knobs(self):
        self.assertIn("laya-check", MAKEFILE.split(".PHONY: build-laya-trainer")[1].split("\n")[0])
        recipe = MAKEFILE.split("\nlaya-check:")[1].split("\n\n")[0]
        self.assertIn("evicts the teacher", recipe.splitlines()[0])
        for piece in ("scripts/laya-check.py", "--lab-image $(IMAGE_NAME_GPU)",
                      "--trainer-image $(LAYA_TRAINER_IMAGE)", "--teacher '$(TEACHER)'",
                      "--teacher-port $(TEACHER_PORT)", "--keep-run"):
            self.assertIn(piece, recipe)

    def test_the_reference_is_documented(self):
        doc = (REPO_ROOT / "docs" / "laya-trainer.md").read_text()
        self.assertIn("make laya-check", doc)
        self.assertIn(lc.REFERENCE["dataset"], doc)


if __name__ == "__main__":
    unittest.main()
