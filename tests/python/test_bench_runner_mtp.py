"""bench_runner's 4v switches: --mtp and --max-connections (platform-rebuild Phase 4).

4v judges vLLM 0.30.0 against 0.28.0 on the teacher AS THE PICKER SERVES
IT, which is with its MTP drafter (`<name>::mtp@<ctx>`; CLAUDE.md, "MTP is
unconditional"). The harness never sent `::mtp`, so every bench number was
MTP-off. `--mtp` sends it -- but only for a row whose probe cell records
`mtp_fits: true`; any other selected row refuses the whole run before
anything is benched, because an MTP-off measurement labelled MTP-on, or a
launch failure, is worse than no measurement.

`--max-connections` overrides the per-backend cap for one run: the 4v
quality runs use 4, the teacher's --max-num-seqs, so no sample waits
inside the engine on its working-time clock. The value used is recorded,
on every scored task entry and on the row.

The latency task's per-prompt records go to `<log-dir>/latency-<alias>.jsonl`
(alias as sent), never into the cache row.

Stdlib unittest; inspect_ai and the router are faked, nothing is served.
"""

from __future__ import annotations

import json
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


def _entry(*, mtp_fits: bool | None, ctx: int = 118784, vram: str = "24") -> dict:
    cell = {"fits": True}
    if mtp_fits is not None:
        cell["mtp_fits"] = mtp_fits
    return {"aliases": ["Q-devai"], "probes": {vram: {str(ctx): cell}}}


def _target(alias: str, *, mtp_fits: bool | None, ctx: int = 118784) -> dict:
    return {"key": f"{alias}::vllm-devai::{ctx}", "alias": alias, "ctx": ctx,
            "entry": _entry(mtp_fits=mtp_fits, ctx=ctx)}


class MtpTargetsTest(unittest.TestCase):
    def test_mtp_goes_before_the_context_pin(self) -> None:
        [t] = bench_runner.with_mtp([_target("Qwen3.8-27B-MTP-devai-NVFP4", mtp_fits=True)], 24)
        self.assertTrue(t["mtp"])
        self.assertEqual(bench_runner.served_name(t),
                         "Qwen3.8-27B-MTP-devai-NVFP4::mtp@118784")

    def test_without_mtp_the_alias_is_unchanged(self) -> None:
        t = _target("Q", mtp_fits=True)
        self.assertEqual(bench_runner.served_name(t), "Q@118784")

    def test_the_row_key_does_not_change(self) -> None:
        t = _target("Q", mtp_fits=True)
        [m] = bench_runner.with_mtp([t], 24)
        self.assertEqual(m["key"], t["key"])

    def test_a_row_without_mtp_fits_true_refuses_the_whole_run(self) -> None:
        good = _target("Good", mtp_fits=True)
        for bad in (_target("NoFlag", mtp_fits=None), _target("Drafter-OOM", mtp_fits=False)):
            with self.assertRaises(ValueError) as cm:
                bench_runner.with_mtp([good, bad], 24)
            self.assertIn(f"{bad['alias']}@118784", str(cm.exception))
            self.assertNotIn("Good@", str(cm.exception))

    def test_the_cell_is_read_from_the_host_band(self) -> None:
        t = {"key": "k", "alias": "Q", "ctx": 32768,
             "entry": {"probes": {"16": {"32768": {"fits": True, "mtp_fits": True}},
                                  "24": {"32768": {"fits": True}}}}}
        with self.assertRaises(ValueError):
            bench_runner.with_mtp([t], 24)
        self.assertTrue(bench_runner.with_mtp([t], 16)[0]["mtp"])


class MainRefusesBeforeBenchingTest(unittest.TestCase):
    """--mtp on a mixed selection exits non-zero before any model runs."""

    def _main(self, argv: list[str], targets: list[dict]) -> mock.MagicMock:
        with mock.patch.object(sys, "argv", ["bench_runner.py", *argv]), \
                mock.patch.object(bench_runner, "_check_router"), \
                mock.patch.object(bench_runner, "discover_models", return_value=targets), \
                mock.patch.object(bench_runner, "load_cache", return_value={}), \
                mock.patch.object(bench_runner, "capture_host_env", return_value={}), \
                mock.patch.object(bench_runner, "stamp_host_env", return_value="env"), \
                mock.patch.object(bench_runner, "probe_image_digest", return_value=None), \
                mock.patch.object(bench_runner, "run_for_target") as rft:
            bench_runner.main()
        return rft

    def test_refused_run_benches_nothing(self) -> None:
        targets = [_target("Good", mtp_fits=True), _target("Bad", mtp_fits=False)]
        with mock.patch.object(bench_runner, "load_cache") as lc, \
                mock.patch.object(bench_runner, "run_for_target") as rft, \
                mock.patch.object(sys, "argv", ["bench_runner.py", "--backend",
                                                "vllm-devai", "--mtp"]), \
                mock.patch.object(bench_runner, "_check_router"), \
                mock.patch.object(bench_runner, "discover_models", return_value=targets), \
                mock.patch("sys.stderr"), self.assertRaises(SystemExit) as cm:
            bench_runner.main()
        rft.assert_not_called()
        lc.assert_not_called()          # refused before the cache is even read
        self.assertIn("Bad@118784", str(cm.exception.code))

    def test_accepted_run_passes_mtp_targets_and_the_connection_cap(self) -> None:
        targets = [_target("Good", mtp_fits=True)]
        with mock.patch("sys.stderr"):
            rft = self._main(["--backend", "vllm-devai", "--mtp",
                              "--max-connections", "4"], targets)
        rft.assert_called_once()
        self.assertTrue(rft.call_args.args[0]["mtp"])
        self.assertEqual(rft.call_args.kwargs["max_connections"], 4)

    def test_without_flags_the_targets_and_cap_are_as_before(self) -> None:
        targets = [_target("Good", mtp_fits=None)]
        with mock.patch("sys.stderr"):
            rft = self._main(["--backend", "vllm-devai"], targets)
        self.assertNotIn("mtp", rft.call_args.args[0])
        self.assertIsNone(rft.call_args.kwargs["max_connections"])

    def test_a_cap_below_one_is_refused(self) -> None:
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as cm:
            self._main(["--backend", "vllm-devai", "--max-connections", "0"], [])
        self.assertNotEqual(cm.exception.code, 0)


class MaxConnectionsOverrideTest(unittest.TestCase):
    def _invoke(self, backend: str, **kw) -> dict:
        captured: dict = {}

        def fake_eval(task_obj, **kwargs):
            captured.update(kwargs)
            return [object()]

        fake = types.ModuleType("inspect_ai")
        fake.eval = fake_eval
        with mock.patch.dict(sys.modules, {"inspect_ai": fake}), \
                tempfile.TemporaryDirectory() as tmp:
            bench_runner._invoke_inspect_task(
                task_obj=object(), served_model="m", backend=backend,
                router_url="http://r", log_dir=Path(tmp), timeout_s=900.0, **kw)
        return captured

    def test_the_override_reaches_inspect(self) -> None:
        self.assertEqual(self._invoke("vllm-devai", max_connections=4)["max_connections"], 4)
        self.assertEqual(self._invoke("ollama", max_connections=2)["max_connections"], 2)

    def test_the_default_is_unchanged_without_it(self) -> None:
        self.assertEqual(self._invoke("vllm-devai")["max_connections"], 10)
        self.assertEqual(self._invoke("ollama")["max_connections"], 1)
        self.assertEqual(self._invoke("vllm-devai", max_connections=None)["max_connections"], 10)

    def test_every_scored_call_site_passes_the_runs_value(self) -> None:
        import re
        calls = re.findall(r"= _invoke_inspect_task\((.*?)\n\s*\)", SOURCE, re.S)
        self.assertGreaterEqual(len(calls), 6)
        for body in calls:
            self.assertIn("max_connections=used_connections", body, body)


class RecordedOnTheRowTest(unittest.TestCase):
    """The value used, and MTP, land on every task entry and on the row."""

    def _run(self, target: dict, *, max_connections: int | None,
             log_dir: Path, latency: dict | None = None) -> dict:
        cache: dict = {}
        log = types.SimpleNamespace(samples=[], status="success")
        # inspect_ai is not installed on the host; the task module only
        # has to hand run_for_target a task object.
        fake_gsm8k = types.ModuleType("bench.tasks.gsm8k")
        fake_gsm8k.gsm8k_task = lambda n: object()
        latency = latency or {"leak_rate": 0.0, "leaked_markers": {}, "n_samples": 1,
                              "n_errors": 0, "tps_sustained_p50": 30.0}
        with mock.patch.dict(sys.modules, {"bench.tasks.gsm8k": fake_gsm8k}), \
                mock.patch.object(bench_runner, "save_cache"), \
                mock.patch.object(bench_runner.bench_latency_leak, "run",
                                  return_value=latency) as lat, \
                mock.patch.object(bench_runner, "_print_latency_summary"), \
                mock.patch.object(bench_runner, "_fetch_backend_metrics", return_value={}), \
                mock.patch.object(bench_runner, "_invoke_inspect_task",
                                  return_value=log) as inv, \
                mock.patch.object(bench_runner, "_aggregate_score", return_value=(0.9, 5)), \
                mock.patch.object(bench_runner, "_log_name", return_value="x.eval"), \
                mock.patch.object(bench_runner, "VramSampler") as sampler, \
                mock.patch("sys.stderr"):
            sampler.return_value.stop.return_value = {
                "peak_vram_gb": 1, "mean_vram_gb": 1, "n_samples": 1}
            bench_runner.run_for_target(
                target, backend="vllm-devai", router_url="http://r",
                tasks=["leak", "gsm8k"], n_gsm8k=5, n_humaneval=1, n_tools=1,
                n_mmlu_pro=1, n_gpqa=1, n_leak_prompts=1, n_longctx_fraction=0.8,
                n_longctx_max_tokens=64, log_dir=log_dir, cache=cache,
                cache_path=log_dir / "c.json", force=False, early_drop=False,
                max_connections=max_connections)
        self.inv, self.lat = inv, lat
        return cache[target["key"]]

    def test_mtp_run_records_mtp_and_the_cap(self) -> None:
        [t] = bench_runner.with_mtp([_target("Q", mtp_fits=True)], 24)
        with tempfile.TemporaryDirectory() as tmp:
            row = self._run(t, max_connections=4, log_dir=Path(tmp))
        self.assertEqual(self.inv.call_args.kwargs["served_model"], "Q::mtp@118784")
        self.assertEqual(self.inv.call_args.kwargs["max_connections"], 4)
        self.assertEqual(self.lat.call_args.kwargs["model"], "Q::mtp@118784")
        gsm = row["tasks"]["gsm8k_subset_5"]
        self.assertEqual(gsm["max_connections"], 4)
        self.assertTrue(gsm["mtp"])
        self.assertTrue(row["tasks"]["leak_probe"]["mtp"])
        # The latency sidecar is sequential: no connection cap applies to it.
        self.assertNotIn("max_connections", row["tasks"]["leak_probe"])
        self.assertIs(row["metrics"]["mtp"], True)
        self.assertEqual(row["metrics"]["max_connections"], 4)

    def test_default_run_records_the_backend_default_and_no_mtp(self) -> None:
        t = _target("Q", mtp_fits=None)
        with tempfile.TemporaryDirectory() as tmp:
            row = self._run(t, max_connections=None, log_dir=Path(tmp))
        self.assertEqual(self.inv.call_args.kwargs["served_model"], "Q@118784")
        gsm = row["tasks"]["gsm8k_subset_5"]
        self.assertEqual(gsm["max_connections"], bench_runner.max_connections_for("vllm-devai"))
        self.assertNotIn("mtp", gsm)
        self.assertIs(row["metrics"]["mtp"], False)

    def test_latency_records_go_to_the_log_dir_not_the_row(self) -> None:
        [t] = bench_runner.with_mtp([_target("Q", mtp_fits=True)], 24)
        records = [
            {"i": 0, "id": "warmup-fact", "ttft_ms": 900.0, "effective_tokens": 3,
             "token_source": "usage", "stream_seconds": 0.1, "finish_reason": "stop",
             "error": None},
            {"i": 1, "id": "warmup-arith", "ttft_ms": None, "effective_tokens": None,
             "token_source": None, "stream_seconds": None, "finish_reason": None,
             "error": "HTTP Error 503"},
        ]
        latency = {"leak_rate": 0.0, "leaked_markers": {}, "n_samples": 1,
                   "n_errors": 1, "tps_sustained_p50": 30.0, "records": records}
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "4v" / "logs"     # created on demand
            row = self._run(t, max_connections=4, log_dir=logs, latency=latency)
            path = logs / "latency-Q::mtp@118784.jsonl"
            self.assertTrue(path.is_file())
            lines = [json.loads(ln) for ln in path.read_text().splitlines()]
        self.assertEqual(lines, records)
        self.assertNotIn("records", json.dumps(row))

    def test_records_file_replaces_an_earlier_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            bench_runner.write_latency_records(d, "Q@1", [{"i": 0}, {"i": 1}])
            p = bench_runner.write_latency_records(d, "Q@1", [{"i": 7}])
            self.assertEqual(p.read_text(), '{"i": 7}\n')
            self.assertEqual(sorted(x.name for x in d.iterdir()), ["latency-Q@1.jsonl"])

    def test_a_slash_in_the_name_stays_in_the_log_dir(self) -> None:
        p = bench_runner.latency_records_path(Path("/logs"), "hf.co/x/y:q4@32768")
        self.assertEqual(p.parent, Path("/logs"))


class ScratchCacheAndLogDirTest(unittest.TestCase):
    """4v points --cache and --log-dir at scratch paths per build."""

    def test_flags_parse_to_the_given_paths(self) -> None:
        with mock.patch.object(sys, "argv", [
                "bench_runner.py", "--backend", "vllm-devai",
                "--cache", "/var/cache/devai/bench/4v/tag/cache.json",
                "--log-dir", "/var/cache/devai/bench/4v/tag/logs"]), \
                mock.patch.object(bench_runner, "_check_router"), \
                mock.patch.object(bench_runner, "discover_models",
                                  return_value=[_target("Q", mtp_fits=True)]), \
                mock.patch.object(bench_runner, "load_cache", return_value={}) as lc, \
                mock.patch.object(bench_runner, "capture_host_env", return_value={}), \
                mock.patch.object(bench_runner, "stamp_host_env", return_value="env"), \
                mock.patch.object(bench_runner, "probe_image_digest", return_value=None), \
                mock.patch.object(bench_runner, "run_for_target") as rft, \
                mock.patch("sys.stderr"):
            bench_runner.main()
        lc.assert_called_once_with(Path("/var/cache/devai/bench/4v/tag/cache.json"))
        self.assertEqual(rft.call_args.kwargs["log_dir"],
                         Path("/var/cache/devai/bench/4v/tag/logs"))
        self.assertEqual(rft.call_args.kwargs["cache_path"],
                         Path("/var/cache/devai/bench/4v/tag/cache.json"))


if __name__ == "__main__":
    unittest.main()
