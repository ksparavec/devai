"""Per-model clean slate, per-task deadline, prefix-scored partial results.

Owner's rules (2026-09-28): before every model the GPU is wiped; every task
is stopped after 30 minutes whatever it has done, and its partial log is
scored on the unbroken prefix of questions. Also here: an eval that ended in
error is not a score (three models were scored 0/11..0/29 and drop-flagged
on 2026-09-28 when their evals aborted on request errors).

Stdlib unittest; inspect_ai, make, podman and signals are faked.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bench import bench_runner  # noqa: E402
from bench import harvest_truncated as H  # noqa: E402


def _load_bench_sync():
    spec = importlib.util.spec_from_file_location("bench_sync_deadline", REPO_ROOT / "scripts" / "bench-sync.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_sync_deadline"] = mod
    spec.loader.exec_module(mod)
    return mod


BS = _load_bench_sync()


class ErroredEvalTest(unittest.TestCase):
    def _invoke(self, log):
        fake = types.ModuleType("inspect_ai")
        fake.eval = lambda task_obj, **kw: [log]
        with mock.patch.dict(sys.modules, {"inspect_ai": fake}), tempfile.TemporaryDirectory() as tmp:
            return bench_runner._invoke_inspect_task(
                task_obj=object(), served_model="m", backend="vllm",
                router_url="http://r", log_dir=Path(tmp), timeout_s=600.0)

    def test_an_errored_eval_raises_and_is_not_scored(self) -> None:
        log = types.SimpleNamespace(status="error", error=types.SimpleNamespace(
            message="\nModelGenerateError('Request...')"))
        with self.assertRaises(RuntimeError) as ctx:
            self._invoke(log)
        self.assertIn("ModelGenerateError", str(ctx.exception))

    def test_an_interrupted_eval_with_no_log_says_so(self) -> None:
        fake = types.ModuleType("inspect_ai")
        fake.eval = lambda task_obj, **kw: []
        with mock.patch.dict(sys.modules, {"inspect_ai": fake}), tempfile.TemporaryDirectory() as tmp, \
                self.assertRaisesRegex(RuntimeError, "interrupted"):
            bench_runner._invoke_inspect_task(task_obj=object(), served_model="m", backend="vllm",
                                              router_url="http://r", log_dir=Path(tmp), timeout_s=60.0)

    def test_success_and_cancelled_are_distinguished(self) -> None:
        ok = types.SimpleNamespace(status="success", error=None)
        self.assertIs(self._invoke(ok), ok)
        with self.assertRaises(RuntimeError):
            self._invoke(types.SimpleNamespace(status="cancelled", error=None))


class RunnerLinksTest(unittest.TestCase):
    def test_every_task_entry_names_its_log(self) -> None:
        src = (REPO_ROOT / "scripts" / "bench" / "bench_runner.py").read_text()
        self.assertEqual(src.count('"inspect_log": _log_name(eval_log),'), 6)
        log = types.SimpleNamespace(location="/var/cache/devai/bench/inspect-logs/x_gpqa-task_A.eval")
        self.assertEqual(bench_runner._log_name(log), "x_gpqa-task_A.eval")
        self.assertIsNone(bench_runner._log_name(types.SimpleNamespace(location=None)))

    def test_planned_n_follows_the_env_like_the_cli_defaults(self) -> None:
        with mock.patch.dict("os.environ", {"BENCH_N_GPQA": "60"}):
            self.assertEqual(bench_runner.planned_n("gpqa"), 60)
        self.assertEqual(bench_runner.planned_n("humaneval_plus"), bench_runner.planned_n("humaneval"))


def _eval_zip(path: Path, samples: list[dict], status: str | None) -> None:
    with zipfile.ZipFile(path, "w") as z:
        if status is not None:
            z.writestr("header.json", json.dumps({"status": status}))
        for s in samples:
            z.writestr(f"samples/{s['id']}_epoch_1.json", json.dumps(s))


def _s(i, value="C", limit=None, error=None, subcase=None):
    return {"id": i, "scores": {"x": {"value": value}}, "limit": limit, "error": error,
            "metadata": {"subcase": subcase} if subcase else {}}


class HarvestTest(unittest.TestCase):
    def test_prefix_stops_at_the_first_question_without_a_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "2026_gpqa-task_X.eval"
            # 1,2 right, 3 timed out (a result: wrong), 4 missing, 5,6 finished late
            _eval_zip(log, [_s(1), _s(2), _s(3, "I", limit={"type": "working", "limit": 900}),
                            _s(5), _s(6)], status="cancelled")
            samples, status = H.read_samples(log)
            e = H.entry("gpqa", samples, status, planned_n=100, deadline_s=1800, log_name=log.name)
        self.assertEqual((e["n"], e["score"], e["n_timeouts"], e["n_in_log"]), (3, round(2 / 3, 4), 1, 5))
        self.assertEqual(e["truncated"], {"reason": "task wall-clock deadline", "deadline_s": 1800,
                                          "prefix": 3, "log_status": "cancelled"})
        self.assertEqual(e["inspect_log"], log.name)

    def test_an_errored_question_ends_the_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "g.eval"
            _eval_zip(log, [_s(1), _s(2, error={"message": "boom"}), _s(3)], status=None)
            samples, status = H.read_samples(log)
            e = H.entry("humaneval", samples, status, planned_n=50, deadline_s=1800, log_name="g.eval")
        self.assertEqual((e["n"], e["pass@1"], e["n_errors"]), (1, 1.0, 1))
        self.assertIsNone(e["truncated"]["log_status"])

    def test_nothing_finished_is_recorded_as_such(self) -> None:
        e = H.entry("mmlu_pro", [], None, planned_n=100, deadline_s=1800, log_name="m.eval")
        self.assertEqual((e["n"], e["score"]), (0, None))

    def test_tools_keeps_its_subcase_breakdown(self) -> None:
        samples = [{"id": 1, "has_result": True, "correct": True, "timeout": False, "limited": False, "subcase": "single_arg"},
                   {"id": 2, "has_result": True, "correct": False, "timeout": False, "limited": False, "subcase": "single_arg"}]
        e = H.entry("tools", samples, "cancelled", planned_n=20, deadline_s=1800, log_name="t.eval")
        self.assertEqual(e["by_subcase"], {"single_arg": 0.5})


class FakeCleanSlate:
    class CleanSlateError(RuntimeError):
        pass

    def __init__(self):
        self.calls = []

    def clean_slate(self):
        self.calls.append("clean_slate")

    def stop_bench_runners(self, grace_s=30.0):
        self.calls.append("stop_bench_runners")
        return []


class FakeProc:
    def __init__(self, timeout: bool):
        self.timeout = timeout
        self.pid = 999999

    def wait(self, timeout=None):
        if self.timeout and timeout is not None:
            raise subprocess.TimeoutExpired("make", timeout)
        return 0


class ExecuteTest(unittest.TestCase):
    def _execute(self, targets, slow_tasks=(), drop_after=None):
        cs = FakeCleanSlate()
        runner = types.SimpleNamespace(planned_n=lambda t: 100, _strip_subset=bench_runner._strip_subset)
        launched, harvested = [], []

        def fake_popen(cmd, cwd=None, start_new_session=False):
            task = next(a.split("=", 1)[1] for a in cmd if a.startswith("BENCH_TASKS="))
            launched.append((cmd[1], task))
            return FakeProc(timeout=task in slow_tasks)

        def fake_run(cmd):
            if cmd[:2] == ["make", "bench-harvest"]:
                harvested.append(cmd[2])
            return 0

        def fake_row(key):
            done = [t for _, t in launched]
            return {"drop_recommendation": {"x": 1}} if drop_after and drop_after in done else {}

        plan = {"new": targets}
        with mock.patch.object(BS.subprocess, "Popen", side_effect=fake_popen), \
                mock.patch.object(BS, "_run", side_effect=fake_run), \
                mock.patch.object(BS, "_cache_row", side_effect=fake_row), \
                mock.patch.object(BS, "partial_log", return_value=Path("/l/x_gpqa-task_A.eval")), \
                mock.patch.object(BS.os, "killpg"), \
                mock.patch("sys.stdout"):
            rc = BS.execute(plan, max_targets=0, tasks=("leak", "gsm8k", "gpqa"), deadline_s=1800,
                            runner=runner, cs=cs)
        return rc, cs.calls, launched, harvested

    T1 = {"backend": "vllm", "key": "org/a@1::vllm::131072", "alias": "a", "ctx": 131072, "class": "new"}
    T2 = {"backend": "ollama", "key": "d1::ollama::131072", "alias": "b", "ctx": 131072, "class": "new"}

    def test_a_clean_slate_before_every_model_and_one_run_per_task(self) -> None:
        rc, calls, launched, _ = self._execute([self.T2, self.T1])
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ["clean_slate", "clean_slate"])
        self.assertEqual(launched, [("bench-vllm", "leak"), ("bench-vllm", "gsm8k"), ("bench-vllm", "gpqa"),
                                    ("bench-ollama", "leak"), ("bench-ollama", "gsm8k"), ("bench-ollama", "gpqa")])

    def test_a_task_past_its_deadline_is_stopped_and_its_log_scored(self) -> None:
        _, calls, launched, harvested = self._execute([self.T1], slow_tasks=("gpqa",))
        self.assertEqual(calls, ["clean_slate", "stop_bench_runners"])
        self.assertEqual(len(harvested), 1)
        self.assertIn("--task gpqa", harvested[0])
        self.assertIn("--planned-n 100", harvested[0])
        self.assertIn("--deadline-s 1800", harvested[0])
        self.assertIn("--key org/a@1::vllm::131072", harvested[0])

    def test_a_drop_flag_ends_the_models_remaining_tasks(self) -> None:
        _, _, launched, _ = self._execute([self.T1], drop_after="gsm8k")
        self.assertEqual([t for _, t in launched], ["leak", "gsm8k"])

    def test_no_clean_slate_no_bench(self) -> None:
        cs = FakeCleanSlate()
        cs.clean_slate = mock.Mock(side_effect=cs.CleanSlateError("gpu busy"))
        runner = types.SimpleNamespace(planned_n=lambda t: 100, _strip_subset=bench_runner._strip_subset)
        with mock.patch.object(BS.subprocess, "Popen") as popen, \
                mock.patch.object(BS, "_cache_row", return_value={}), \
                mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            rc = BS.execute({"new": [self.T1]}, max_targets=0, tasks=("gsm8k",), runner=runner, cs=cs)
        self.assertEqual(rc, 4)
        popen.assert_not_called()

    def test_the_deadline_default_is_thirty_minutes(self) -> None:
        self.assertEqual(BS.TASK_DEADLINE_S, 1800)


if __name__ == "__main__":
    unittest.main()
