"""The bench must not queue more requests than the backend serves at once.

inspect_ai fires 10 requests concurrently by default, and its per-sample
`time_limit` clock starts when the sample starts -- NOT when the backend
begins answering it. Ollama here serves ONE request at a time
(`OLLAMA_NUM_PARALLEL:1`, `n_seq_max = 1` in its log; it is not set anywhere
in this repo, that is Ollama's own choice for a model this size), so nine of
every ten samples spent their clock waiting in a queue.

Measured 2026-09-19 on qwen3.8:27b-ud-q4_k_xl (dense 27B, 30 tok/s, thinking)
during MMLU-Pro: samples took 300-600 s of wall clock for ~1000 output tokens
(~35 s of real generation), and 5 of the first 40 hit the 900 s limit and were
scored INCORRECT -- ten `context canceled` lines in the router log, all in
that one stage. The model never got a fair attempt at them. Fast MoE models
(~175 tok/s) drain the queue before the clock matters, so the defect
penalised exactly the slow dense models and nobody had seen it.

With the cap, the clock measures the model. Total wall time is unchanged: the
GPU was already busy the whole time, serially.

A second defect found the same evening: results were written to the bench
cache only when a model's LAST task finished. Stopping a 3.5-hour run at hour
two lost every completed stage, although bench-sync documents itself as
resumable. Each task's result is now persisted as soon as it exists.

Stdlib unittest only; inspect_ai is faked, nothing is served.
"""

from __future__ import annotations

import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bench import bench_runner  # noqa: E402

SOURCE = (REPO_ROOT / "scripts" / "bench" / "bench_runner.py").read_text()


class MaxConnectionsTest(unittest.TestCase):
    def test_ollama_is_benched_one_request_at_a_time(self) -> None:
        self.assertEqual(bench_runner.max_connections_for("ollama"), 1)

    def test_batching_backends_keep_inspects_own_default(self) -> None:
        # vLLM and SGLang batch continuously; nothing measured says their
        # queue distorts the clock, so nothing is changed for them.
        for backend in ("vllm", "sglang"):
            self.assertIsNone(bench_runner.max_connections_for(backend), backend)


class InvokeWiringTest(unittest.TestCase):
    def _invoke(self, backend: str) -> dict:
        captured: dict = {}

        def fake_eval(task_obj, **kwargs):
            captured.update(kwargs)
            return [object()]

        fake = types.ModuleType("inspect_ai")
        fake.eval = fake_eval
        fake_model = types.ModuleType("inspect_ai.model")
        fake_model.GenerateConfig = lambda **kw: kw
        with mock.patch.dict(sys.modules, {"inspect_ai": fake,
                                           "inspect_ai.model": fake_model}), \
                tempfile.TemporaryDirectory() as tmp:
            bench_runner._invoke_inspect_task(
                task_obj=object(), served_model="m", backend=backend,
                router_url="http://r", log_dir=Path(tmp), timeout_s=900.0)
        return captured

    def test_ollama_eval_is_capped_at_one_connection(self) -> None:
        self.assertEqual(self._invoke("ollama").get("max_connections"), 1)

    def test_vllm_eval_is_left_alone(self) -> None:
        self.assertNotIn("max_connections", self._invoke("vllm"))

    def test_every_call_site_says_which_backend_it_is_benching(self) -> None:
        # The calls sit inside `except Exception`, so a call site that forgot
        # `backend=` would not crash -- it would record the task as failed.
        calls = re.findall(r"= _invoke_inspect_task\((.*?)\n\s*\)", SOURCE, re.S)
        self.assertGreaterEqual(len(calls), 6)
        for body in calls:
            self.assertIn("backend=backend", body, body)


class PersistAfterEveryTaskTest(unittest.TestCase):
    """A finished task must survive the run being stopped."""

    def test_cache_is_saved_after_the_first_task_not_only_at_the_end(self) -> None:
        saves: list[dict] = []

        def fake_save(path, cache):
            row = cache.get("k") or {}
            saves.append(dict(row.get("tasks") or {}))

        fake_latency = {"leak_rate": 0.0, "leaked_markers": [], "n_samples": 40,
                        "n_errors": 0, "tps_sustained_p50": 30.0}
        target = {"alias": "fam:27b", "ctx": 131072, "key": "k"}
        with mock.patch.object(bench_runner, "save_cache", fake_save), \
                mock.patch.object(bench_runner.bench_latency_leak, "run",
                                  return_value=fake_latency), \
                mock.patch.object(bench_runner, "_print_latency_summary"), \
                mock.patch.object(bench_runner, "_fetch_backend_metrics",
                                  return_value={}), \
                mock.patch.object(bench_runner, "VramSampler") as sampler:
            sampler.return_value.stop.return_value = {
                "peak_vram_gb": 1, "mean_vram_gb": 1, "n_samples": 1}
            # One task only: the save that follows it must not be the
            # end-of-model save, or a stopped run would still lose it.
            bench_runner.run_for_target(
                target, backend="ollama", router_url="http://r", tasks=["leak"],
                n_gsm8k=1, n_humaneval=1, n_tools=1, n_mmlu_pro=1, n_gpqa=1,
                n_leak_prompts=40, n_longctx_fraction=0.8,
                n_longctx_max_tokens=64, log_dir=Path("/nonexistent"),
                cache={}, cache_path=Path("/nonexistent/c.json"), force=False)
        self.assertGreaterEqual(len(saves), 2,
                                "expected a save right after the task AND at the end")
        self.assertIn("leak_probe", saves[0],
                      "the first save must already carry the finished task")


if __name__ == "__main__":
    unittest.main()
