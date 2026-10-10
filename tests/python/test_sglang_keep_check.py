"""Phase 15's keep-or-freeze check (scripts/bench/sglang_keep_check.py, I16).

A fake engine stands in for all three configurations at the HTTP layer
(bench_core's streamed completion, /metrics), so the whole measure() path
runs: the dry run, the steady prompts, the sessions, the failure rule and
the decision. Pinned:

  * a failed turn in an SGLang arm (radix on or off) ends the run as a
    freeze, one in the vLLM arm voids it, and the engine is stopped;
  * a dry run with a zero hit rate measures nothing;
  * I16's four conditions and the keep/freeze outcome.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "bench"))

_spec = importlib.util.spec_from_file_location(
    "sglang_keep_check", REPO_ROOT / "scripts" / "bench" / "sglang_keep_check.py")
KC = importlib.util.module_from_spec(_spec)
sys.modules["sglang_keep_check"] = KC
_spec.loader.exec_module(KC)
BC = KC.BC

PROMPTS = [{"id": f"p{i}", "prompt": f"prompt {i}", "max_tokens": 8} for i in range(31)]
SESSIONS = 4


class FakeEngines:
    """Three configurations behind one fake HTTP layer.

    TTFT per config: sglang 100 ms steady / 300 ms on turns, noradix 900 ms
    on turns, vllm 120 ms steady / 600 ms on turns (keep, unless a test
    changes them). SGLang radix-on finds 800 of every 1000 prompt tokens
    cached on turns 2-5.
    """

    def __init__(self, fail: tuple[str, int] | None = None, cached: int = 800):
        self.fail = fail                  # (config, n-th streamed request on it)
        self.cached = cached
        self.calls: dict[str, int] = {}
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.ttft = {"sglang": (0.1, 0.3), "sglang-noradix": (0.1, 0.9), "vllm": (0.12, 0.6)}

    def start(self, config: str) -> KC.Engine:
        self.started.append(config)
        return KC.Engine(config, f"http://fake/{config}", {"argv": [config]})

    def stop(self, engine: KC.Engine) -> None:
        self.stopped.append(engine.config)

    def _config(self, url: str) -> str:
        return url.rstrip("/").split("/")[-1].removesuffix("/metrics")

    def stream(self, url: str, body: dict, timeout: float = 600.0) -> dict:
        config = self._config(url)
        n = self.calls[config] = self.calls.get(config, 0) + 1
        if self.fail == (config, n):
            raise OSError("connection reset by peer")
        turn = len(body["messages"]) > 1      # a session turn carries the system prefix
        steady, turns = self.ttft[config]
        return {"content": "ok", "t_open": 0.0, "t_first_token": turns if turn else steady,
                "t_done": 1.0, "effective_tokens": 8, "completion_tokens": 8, "token_source": "usage"}

    def metrics(self, url: str, timeout: float) -> str:
        config = url.split("/")[-2]
        n = self.calls.get(config, 0)
        session_turns = [c for c in range(1, n + 1)]
        cached = 0
        if config == "sglang":
            # Only the session turns count (steady prompts carry no history):
            # every turn but each session's first finds `cached` tokens.
            cached = self.cached * sum(1 for c in session_turns if self._turn_index(config, c) > 1)
        return (f'sglang:prompt_tokens_total{{m="x"}} {1000 * n}\n'
                f'sglang:cached_tokens_total{{m="x",cache_source="device"}} {cached}\n')

    def _turn_index(self, config: str, c: int) -> int:
        return self.turn_index.get((config, c), 0)


class MeasureTest(unittest.TestCase):
    def setUp(self) -> None:
        self._orig = (BC.stream_chat_completion, KC.stream_chat_completion, BC._fetch_text,
                      BC._one_request, BC.SETTLE_PAUSE_S)
        BC.SETTLE_PAUSE_S = 0.0
        BC._one_request = lambda u, m, p, mt, ts, body_fields=None: {
            "ok": True, "capped": False, "ttft_ms": 1.0, "tokens": 1, "wall_s": 0.01}

    def tearDown(self) -> None:
        (BC.stream_chat_completion, KC.stream_chat_completion, BC._fetch_text,
         BC._one_request, BC.SETTLE_PAUSE_S) = self._orig

    def _run(self, fake: FakeEngines) -> dict:
        fake.turn_index = {}

        def stream(url, body, timeout=600.0):
            res = fake.stream(url, body, timeout)
            config = fake._config(url)
            if len(body["messages"]) > 1:
                # Turn number within its session: 1 for the first message pair.
                fake.turn_index[(config, fake.calls[config])] = (len(body["messages"]) - 2) // 2 + 1
            return res
        BC.stream_chat_completion = stream
        KC.stream_chat_completion = stream
        BC._fetch_text = fake.metrics
        return KC.measure(model="m", salt="s", prompts=PROMPTS, start=fake.start, stop=fake.stop,
                          sessions=SESSIONS, dry_sessions=2, log=lambda m: None)

    def test_a_complete_run_decides_keep(self) -> None:
        fake = FakeEngines()
        r = self._run(fake)
        self.assertEqual(fake.started, ["sglang", "sglang-noradix", "vllm"])
        self.assertEqual(fake.stopped, fake.started)
        self.assertEqual(r["outcome"], "keep", r.get("conditions"))
        self.assertAlmostEqual(r["steady_ratio_sglang_over_vllm"], 100 / 120, places=3)
        self.assertEqual(r["turn2plus_sglang_over_vllm"]["ratio"], 0.5)
        self.assertEqual(r["turn2plus_radix_on_over_off"]["ratio"], round(300 / 900, 4))
        self.assertEqual(r["pooled_hit_rate_radix_on"], 0.8)
        self.assertGreater(r["dry_run"]["sglang_hit_rate_pooled"], 0)
        self.assertEqual([s["salt"] for s in r["dry_run"]["sessions"]], ["s-dry-sglang-0", "s-dry-sglang-1"])
        self.assertEqual(r["results"]["vllm"]["sessions"]["sessions"][0]["salt"], "s-vllm-0")

    def test_slower_sglang_turns_freeze(self) -> None:
        fake = FakeEngines()
        fake.ttft["sglang"] = (0.1, 0.5)          # 500/600 = 0.83 > 0.75
        r = self._run(fake)
        self.assertEqual(r["outcome"], "freeze")
        self.assertFalse(r["conditions"]["b_turn2plus_ratio_at_most_0.75_interval_below_1"])

    def test_a_failed_turn_in_each_sglang_arm_is_a_freeze(self) -> None:
        # Request 4 on radix-on is in the dry run; on radix-off it is a session turn.
        for config in ("sglang", "sglang-noradix"):
            fake = FakeEngines(fail=(config, 4))
            r = self._run(fake)
            self.assertEqual(r["outcome"], "freeze", config)
            self.assertEqual(r["failure"]["config"], config)
            self.assertIn("connection reset", r["failure"]["what"])
            self.assertEqual(fake.stopped[-1], config, "the failed engine is stopped")
            self.assertNotIn("conditions", r)

    def test_a_failed_turn_in_the_vllm_arm_voids_the_run(self) -> None:
        fake = FakeEngines(fail=("vllm", 33))     # past its 31 steady prompts: a session turn
        r = self._run(fake)
        self.assertEqual(r["outcome"], "void")
        self.assertEqual(r["failure"]["config"], "vllm")
        self.assertIn("session s-vllm-0", r["failure"]["what"])

    def test_a_failed_steady_prompt_counts_like_a_failed_turn(self) -> None:
        fake = FakeEngines(fail=("vllm", 5))
        r = self._run(fake)
        self.assertEqual(r["outcome"], "void")
        self.assertIn("steady prompt 4", r["failure"]["what"])

    def test_a_dry_run_without_hits_measures_nothing(self) -> None:
        fake = FakeEngines(cached=0)
        r = self._run(fake)
        self.assertEqual(r["outcome"], "refused")
        self.assertIn("zero SGLang hit rate", r["failure"]["what"])
        self.assertEqual(fake.started, ["sglang"])
        self.assertEqual(fake.stopped, ["sglang"])
        self.assertNotIn("sessions", r["results"]["sglang"])

    def test_a_dry_run_without_counters_says_so(self) -> None:
        fake = FakeEngines()
        fake.metrics = lambda url, timeout: "# no sglang counters here\n"
        r = self._run(fake)
        self.assertEqual(r["outcome"], "refused")
        self.assertIn("no SGLang cached/prompt token counters", r["failure"]["what"])

    def test_a_late_refusal_voids_the_run_and_keeps_the_measured_arms(self) -> None:
        fake = FakeEngines()
        start = fake.start

        def busy_card(config):
            if config == "vllm":
                raise KC.CheckRefused("before launching vllm: memory.used 900 MiB > 512")
            return start(config)
        fake.start = busy_card
        r = self._run(fake)
        self.assertEqual(r["outcome"], "void")
        self.assertEqual(r["failure"]["config"], "vllm")
        self.assertEqual(set(r["results"]), {"sglang", "sglang-noradix"})
        self.assertEqual(len(r["results"]["sglang"]["sessions"]["sessions"]), SESSIONS)

    def test_an_unexpected_error_voids_the_run_and_keeps_the_launch(self) -> None:
        fake = FakeEngines()
        orig = KC.steady_ttft

        def broken(*a, **k):
            raise KeyError("t_open")
        KC.steady_ttft = broken
        try:
            r = self._run(fake)
        finally:
            KC.steady_ttft = orig
        self.assertEqual(r["outcome"], "void")
        self.assertIn("unexpected KeyError", r["failure"]["what"])
        self.assertEqual(r["results"]["sglang"]["launch"], {"argv": ["sglang"]})
        self.assertEqual(fake.stopped, ["sglang"])

    def test_a_counter_first_exported_after_turn_1_counts_from_zero(self) -> None:
        recs = [{"turn": i, "ok": True, "ttft_ms": 1.0, "completion_tokens": 8, "metrics_error": None,
                 "counters_before": b, "counters_after": a}
                for i, (b, a) in enumerate([
                    ({}, {"sglang:prompt_tokens_total": 1000.0}),
                    ({"sglang:prompt_tokens_total": 1000.0},
                     {"sglang:prompt_tokens_total": 2000.0, "sglang:cached_tokens_total": 800.0})], 1)]
        s = BC._session_summary({"salt": "x", "turns": recs}, 2)
        self.assertEqual(s["sglang_hit_rate_turn2plus"], 0.8)


class DecideTest(unittest.TestCase):
    def _results(self, sg: list[float], nr: list[float], vl: list[float],
                 steady=(100.0, 100.0), hit=0.5) -> dict:
        sess = lambda vals: {"sessions": [{"turn2plus_median_ms": v} for v in vals],  # noqa: E731
                             "sglang_hit_rate_pooled": hit}
        return {"sglang": {"steady": {"median_ms": steady[0]}, "sessions": sess(sg)},
                "sglang-noradix": {"sessions": sess(nr)},
                "vllm": {"steady": {"median_ms": steady[1]}, "sessions": sess(vl)}}

    def test_each_condition_alone_decides_a_freeze(self) -> None:
        good = dict(sg=[300, 310, 290, 305], nr=[900, 880, 910, 905], vl=[600, 620, 590, 610])
        self.assertEqual(KC.decide(self._results(**good), reps=2000)["outcome"], "keep")
        cases = {
            "a_steady_ratio_at_most_3": dict(good, steady=(400.0, 100.0)),
            "b_turn2plus_ratio_at_most_0.75_interval_below_1": dict(good, sg=[500, 480, 520, 505]),
            "c_radix_on_over_off_interval_below_1": dict(good, nr=[300, 310, 290, 305]),
            "d_pooled_hit_rate_above_0": dict(good, hit=0.0),
        }
        for name, kw in cases.items():
            d = KC.decide(self._results(**kw), reps=2000)
            self.assertEqual(d["outcome"], "freeze", name)
            self.assertFalse(d["conditions"][name], name)


if __name__ == "__main__":
    unittest.main()
