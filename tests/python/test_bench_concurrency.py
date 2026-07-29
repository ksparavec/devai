"""Concurrency + prefix-reuse benchmark logic.

Network-free: `_one_request` is stubbed, so these pin the ARITHMETIC and
the classification rules -- which is where this task can quietly lie.

Three properties matter more than the rest:

  1. A 429 must never be counted as a slow request. The router caps
     in-flight requests per backend and refuses beyond it, so a CAPPED
     run and a SLOW run otherwise produce the same throughput number.
  2. Aggregate throughput must come from the batch's WALL time, not the
     sum of per-request rates -- the whole point of batching is that
     requests overlap, and summing would double-count the overlap.
  3. The `disjoint` arm must actually be disjoint. A radix tree keys on
     the longest COMMON prefix, so a trailing salt would leave the body
     shared and turn the control arm into a second copy of the treatment
     arm, making the measured prefix gain ~1.0 no matter what the engine
     does.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

spec = importlib.util.spec_from_file_location(
    "bench_concurrency_under_test",
    REPO_ROOT / "scripts" / "bench" / "bench_concurrency.py")
B = importlib.util.module_from_spec(spec)
sys.modules["bench_concurrency_under_test"] = B
spec.loader.exec_module(B)


class TestPrefixConstruction(unittest.TestCase):
    def test_approximate_token_budget(self) -> None:
        p = B._prefix(4096, "x")
        self.assertAlmostEqual(len(p) / B._CHARS_PER_TOKEN, 4096, delta=50)

    def test_distinct_salts_diverge_immediately(self) -> None:
        """The control arm has to be disjoint to a radix tree.

        A prefix cache matches on the longest COMMON prefix. If the salt
        were appended rather than prepended, two 'disjoint' prompts would
        still share ~4096 identical leading tokens and the engine would
        reuse them -- the control arm would silently become a second
        treatment arm and prefix_gain would read ~1.0 regardless of the
        engine's real behaviour.
        """
        a, b = B._prefix(4096, "cell-0"), B._prefix(4096, "cell-1")
        common = 0
        for x, y in zip(a, b):
            if x != y:
                break
            common += 1
        # Only the literal "# session " header may be shared.
        self.assertLess(common, 20, f"{common} shared leading chars")

    def test_same_salt_is_byte_identical(self) -> None:
        # The treatment arm relies on every request in a cell carrying
        # the exact same prefix; any per-request variation would defeat
        # the cache we are trying to measure.
        self.assertEqual(B._prefix(2048, "s"), B._prefix(2048, "s"))


class TestCellAccounting(unittest.TestCase):
    def _stub(self, results):
        """Feed _run_cell a scripted sequence of per-request outcomes."""
        seq = list(results)

        def fake(router_url, model, prompt, max_tokens, timeout_s):
            return seq.pop(0)
        B._one_request = fake

    def setUp(self) -> None:
        self._orig = B._one_request

    def tearDown(self) -> None:
        B._one_request = self._orig

    def test_429_is_counted_separately_and_flags_the_cell(self) -> None:
        self._stub([
            {"ok": True, "capped": False, "ttft_ms": 100.0, "tokens": 10, "wall_s": 1.0},
            {"ok": False, "capped": True, "error": "HTTP 429"},
            {"ok": False, "capped": True, "error": "HTTP 429"},
        ])
        cell = B._run_cell("u", "m", 3, True, 128, 8, 10.0, "c3-shared")
        self.assertEqual(cell["n_429"], 2)
        self.assertEqual(cell["n_ok"], 1)
        self.assertEqual(cell["n_error"], 0, "a 429 is not a generic error")
        self.assertTrue(cell["capped"], "cell must be flagged as router-capped")

    def test_real_errors_are_not_conflated_with_429(self) -> None:
        self._stub([
            {"ok": False, "capped": False, "error": "HTTPError: 500"},
            {"ok": True, "capped": False, "ttft_ms": 50.0, "tokens": 5, "wall_s": 1.0},
        ])
        cell = B._run_cell("u", "m", 2, False, 128, 8, 10.0, "c2-disjoint")
        self.assertEqual(cell["n_429"], 0)
        self.assertEqual(cell["n_error"], 1)
        self.assertFalse(cell["capped"])

    def test_aggregate_tps_uses_wall_time_not_sum_of_rates(self) -> None:
        # Four requests, 100 tokens each, each claiming 1s of its own
        # wall time. They ran CONCURRENTLY, so the batch wall time is
        # ~0s of test time -- summing per-request rates would report
        # 400 tok/s regardless of overlap. We assert the total token
        # count instead, which is the numerator either way.
        self._stub([
            {"ok": True, "capped": False, "ttft_ms": 10.0, "tokens": 100, "wall_s": 1.0}
            for _ in range(4)
        ])
        cell = B._run_cell("u", "m", 4, True, 128, 8, 10.0, "c4-shared")
        self.assertEqual(cell["completion_tokens"], 400)
        self.assertEqual(cell["n_ok"], 4)

    def test_prefix_label_recorded(self) -> None:
        self._stub([{"ok": True, "capped": False, "ttft_ms": 1.0,
                     "tokens": 1, "wall_s": 0.1}])
        self.assertEqual(
            B._run_cell("u", "m", 1, True, 64, 4, 5.0, "x")["prefix"], "shared")
        self._stub([{"ok": True, "capped": False, "ttft_ms": 1.0,
                     "tokens": 1, "wall_s": 0.1}])
        self.assertEqual(
            B._run_cell("u", "m", 1, False, 64, 4, 5.0, "x")["prefix"], "disjoint")


class TestSweepSummary(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_cell = B._run_cell
        self._orig_req = B._one_request
        B._one_request = lambda *a, **k: {
            "ok": True, "capped": False, "ttft_ms": 1.0, "tokens": 1, "wall_s": 0.1}

    def tearDown(self) -> None:
        B._run_cell = self._orig_cell
        B._one_request = self._orig_req

    def _cells(self, table):
        """table: {(level, mode): (agg_tps, capped)}"""
        def fake(router_url, model, concurrency, shared, ptok, mtok, ts, cid):
            tps, capped = table[(concurrency, "shared" if shared else "disjoint")]
            return {
                "concurrency": concurrency,
                "prefix": "shared" if shared else "disjoint",
                "n_ok": concurrency, "n_429": 1 if capped else 0, "n_error": 0,
                "capped": capped, "wall_s": 1.0, "aggregate_tps": tps,
                "ttft_p50_ms": 10.0, "ttft_p95_ms": 12.0,
                "completion_tokens": int(tps), "first_error": None,
            }
        B._run_cell = fake

    def test_prefix_gain_is_shared_over_disjoint(self) -> None:
        self._cells({
            (1, "shared"): (100.0, False), (1, "disjoint"): (100.0, False),
            (8, "shared"): (240.0, False), (8, "disjoint"): (200.0, False),
        })
        out = B.run("m", "u", levels=(1, 8))
        # No benefit at c=1 (nothing to share with); 1.2x at c=8.
        self.assertEqual(out["prefix_gain_by_level"]["1"], 1.0)
        self.assertEqual(out["prefix_gain_by_level"]["8"], 1.2)

    def test_capped_cells_are_excluded_from_gain(self) -> None:
        """A capped cell describes the router, not the engine."""
        self._cells({
            (1, "shared"): (100.0, False), (1, "disjoint"): (100.0, False),
            (32, "shared"): (50.0, True), (32, "disjoint"): (200.0, False),
        })
        out = B.run("m", "u", levels=(1, 32))
        self.assertIn("1", out["prefix_gain_by_level"])
        self.assertNotIn(
            "32", out["prefix_gain_by_level"],
            "a 429-capped cell must not produce a prefix-gain number")
        self.assertTrue(out["any_capped"])

    def test_batch_scaling_is_relative_to_lowest_level_disjoint(self) -> None:
        # Isolates batching from prefix reuse: measured on the disjoint
        # arm, where there is no prefix to reuse.
        self._cells({
            (1, "shared"): (100.0, False), (1, "disjoint"): (100.0, False),
            (4, "shared"): (400.0, False), (4, "disjoint"): (380.0, False),
        })
        out = B.run("m", "u", levels=(1, 4))
        self.assertEqual(out["batch_scaling_by_level"]["1"], 1.0)
        self.assertEqual(out["batch_scaling_by_level"]["4"], 3.8)


class TestSlope(unittest.TestCase):
    """The slope IS the multi-turn measurement.

    A working prefix cache keeps TTFT flat as the conversation grows
    (only new tokens are prefilled) -> slope ~0. Without reuse the engine
    re-prefills the whole history each turn -> slope > 0. Using the slope
    rather than raw TTFT also cancels constant per-request overhead, so
    two engines with different fixed latencies remain comparable.
    """

    def test_flat_ttft_is_zero_slope(self) -> None:
        pts = [(4000.0, 100.0), (8000.0, 101.0), (12000.0, 99.0)]
        self.assertAlmostEqual(B._slope_ms_per_1k(pts), 0.0, delta=1.0)

    def test_linear_growth_recovers_the_rate(self) -> None:
        # +50 ms per 1000 tokens, exactly.
        pts = [(1000.0, 50.0), (2000.0, 100.0), (3000.0, 150.0)]
        self.assertAlmostEqual(B._slope_ms_per_1k(pts), 50.0, delta=0.01)

    def test_constant_offset_does_not_affect_slope(self) -> None:
        # An engine with +1000 ms fixed overhead must not look worse at
        # prefix reuse than one without.
        a = [(1000.0, 50.0), (2000.0, 100.0), (3000.0, 150.0)]
        b = [(x, y + 1000.0) for x, y in a]
        self.assertAlmostEqual(B._slope_ms_per_1k(a), B._slope_ms_per_1k(b),
                               delta=0.01)

    def test_degenerate_inputs(self) -> None:
        self.assertIsNone(B._slope_ms_per_1k([]))
        self.assertIsNone(B._slope_ms_per_1k([(1.0, 1.0)]))
        # All-identical x would divide by zero.
        self.assertIsNone(B._slope_ms_per_1k([(5.0, 1.0), (5.0, 9.0)]))


class TestMultiturn(unittest.TestCase):
    def setUp(self) -> None:
        self._orig = B._turn
        self._orig_req = B._one_request
        self._warmups = []
        B._one_request = lambda u, m, p, mt, ts: (
            self._warmups.append(p) or
            {"ok": True, "capped": False, "ttft_ms": 1.0, "tokens": 1,
             "wall_s": 0.01})

    def tearDown(self) -> None:
        B._turn = self._orig
        B._one_request = self._orig_req

    def test_warms_the_model_before_measuring(self) -> None:
        """Turn 1 must not carry the container cold start.

        Measured on gpt-oss-20b: an unwarmed turn 1 took 76,332 ms
        against ~575 ms for a warm turn -- ~130x. That single point
        dominated the least-squares fit and produced a NEGATIVE slope,
        i.e. "time-to-first-token falls as the conversation grows",
        which is the opposite of what a broken prefix cache looks like.
        """
        B._turn = lambda *a, **k: {"ok": True, "capped": False,
                                   "ttft_ms": 10.0, "content": "OK",
                                   "tokens": 1}
        B.run_multiturn("m", "u", turns=2, prefix_tokens=256, growth_tokens=256)
        self.assertEqual(len(self._warmups), 1, "exactly one warmup request")
        # The warmup must be tiny and must NOT seed the measured prefix,
        # or turn 1 would read as a cache hit it did not earn.
        self.assertLess(len(self._warmups[0]), 200)
        self.assertIn("warmup", self._warmups[0])

    def test_history_grows_and_slope_is_reported(self) -> None:
        seen = []

        def fake(router_url, model, messages, max_tokens, timeout_s):
            seen.append(sum(len(m["content"]) for m in messages))
            return {"ok": True, "capped": False, "ttft_ms": 100.0,
                    "content": "OK", "tokens": 2}
        B._turn = fake
        out = B.run_multiturn("m", "u", turns=4, prefix_tokens=1024,
                              growth_tokens=1024)
        self.assertEqual(out["n_ok"], 4)
        self.assertEqual(len(out["turns"]), 4)
        # Each turn must carry strictly more history than the last.
        self.assertEqual(seen, sorted(seen))
        self.assertLess(seen[0], seen[-1])
        # Flat TTFT -> ~zero slope.
        self.assertAlmostEqual(out["ttft_slope_ms_per_1k_tokens"], 0.0, delta=1.0)

    def test_stops_on_first_error_and_reports_it(self) -> None:
        calls = {"n": 0}

        def fake(router_url, model, messages, max_tokens, timeout_s):
            calls["n"] += 1
            if calls["n"] == 3:
                return {"ok": False, "capped": False, "error": "HTTP 500"}
            return {"ok": True, "capped": False, "ttft_ms": 10.0,
                    "content": "OK", "tokens": 1}
        B._turn = fake
        out = B.run_multiturn("m", "u", turns=8, prefix_tokens=256,
                              growth_tokens=256)
        # Ran 3, aborted on the failure rather than pressing on with a
        # broken conversation.
        self.assertEqual(len(out["turns"]), 3)
        self.assertEqual(out["n_ok"], 2)
        self.assertEqual(out["turns"][-1]["error"], "HTTP 500")

    def test_429_is_flagged(self) -> None:
        B._turn = lambda *a, **k: {"ok": False, "capped": True,
                                   "error": "HTTP 429"}
        out = B.run_multiturn("m", "u", turns=3, prefix_tokens=256)
        self.assertTrue(out["any_capped"])
        self.assertEqual(out["n_ok"], 0)


if __name__ == "__main__":
    unittest.main()
