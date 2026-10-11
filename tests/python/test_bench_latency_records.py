"""Per-prompt latency records, and the /metrics counter parser (4v).

bench_latency_leak.run used to return aggregates only (p50/p95 TTFT, p50
tok/s). 4v compares two vLLM builds prompt by prompt -- the same 40
latency prompts, paired by id, prompts 1-39 -- which aggregates cannot do,
so `run` also returns one record per prompt sent. A failed request is a
record too, with its error and no numbers, so a reader can tell "not
measured" from "measured as zero".

parse_prometheus_counters is the one reader of the engines' /metrics that
bench_concurrency (prefix-cache hits per turn) and the 4v statistics
(MTP acceptance length) share. Counters are summed over label sets, and
a counter the text does not carry is ABSENT, never 0.

Stdlib unittest; the stream is stubbed, nothing is served.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bench import _bench_core, bench_latency_leak  # noqa: E402

_KEYS = {"i", "id", "ttft_ms", "effective_tokens", "token_source",
         "stream_seconds", "finish_reason", "error"}


def _res(*, t_first: float | None, t_done: float, tokens: int, source: str = "usage",
         finish: str | None = "stop") -> dict:
    return {"content": "ok", "reasoning_content": "", "completion_tokens": tokens,
            "prompt_tokens": 10, "effective_tokens": tokens, "token_source": source,
            "t_open": 100.0, "t_first_token": t_first, "t_done": t_done,
            "finish_reason": finish}


class PerPromptRecordsTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.prompts = Path(tmp.name) / "p.jsonl"
        self.prompts.write_text(
            '{"id": "warmup-fact", "prompt": "a", "max_tokens": 8}\n'
            '# a comment line\n'
            '{"id": "math-1", "prompt": "b"}\n'
            '{"prompt": "c"}\n'
            '{"id": "math-3", "prompt": "d"}\n')

    def _run(self, responses: list) -> dict:
        it = iter(responses)

        def fake(url, body, timeout=600.0):
            r = next(it)
            if isinstance(r, Exception):
                raise r
            return r
        with mock.patch.object(bench_latency_leak, "stream_chat_completion", fake), \
                mock.patch.object(bench_latency_leak, "load_leak_patterns", return_value=[]):
            return bench_latency_leak.run(model="Q::mtp@118784", router_url="http://r",
                                          n=40, prompts_path=self.prompts)

    def test_one_record_per_prompt_in_send_order(self) -> None:
        out = self._run([
            _res(t_first=100.5, t_done=102.5, tokens=200),
            _res(t_first=100.25, t_done=101.25, tokens=64, finish="length"),
            urllib.error.URLError("refused"),
            _res(t_first=None, t_done=100.75, tokens=0, source="chars/4", finish="tool_calls"),
        ])
        recs = out["records"]
        self.assertEqual([r["i"] for r in recs], [0, 1, 2, 3])
        self.assertEqual([r["id"] for r in recs], ["warmup-fact", "math-1", "2", "math-3"])
        for r in recs:
            self.assertEqual(set(r), _KEYS)
        first, second, failed, toolonly = recs
        self.assertEqual(first["ttft_ms"], 500.0)
        self.assertEqual(first["stream_seconds"], 2.0)
        self.assertEqual(first["effective_tokens"], 200)
        self.assertEqual(first["token_source"], "usage")
        self.assertEqual(first["finish_reason"], "stop")
        self.assertIsNone(first["error"])
        self.assertEqual(second["finish_reason"], "length")
        self.assertEqual(second["stream_seconds"], 1.0)

    def test_a_failed_request_is_recorded_with_its_error_and_no_numbers(self) -> None:
        out = self._run([
            _res(t_first=100.5, t_done=102.5, tokens=200),
            urllib.error.HTTPError("http://r", 503, "Service Unavailable", {}, None),
            _res(t_first=100.5, t_done=101.0, tokens=5),
            _res(t_first=100.5, t_done=101.0, tokens=5),
        ])
        failed = out["records"][1]
        self.assertIn("503", failed["error"])
        for k in ("ttft_ms", "effective_tokens", "token_source", "stream_seconds",
                  "finish_reason"):
            self.assertIsNone(failed[k], k)
        # The aggregates still count it as an error, as before.
        self.assertEqual(out["n_errors"], 1)

    def test_no_token_means_no_ttft_and_no_stream_time(self) -> None:
        out = self._run([_res(t_first=None, t_done=100.75, tokens=0, source="chars/4",
                              finish="tool_calls")] * 4)
        rec = out["records"][0]
        self.assertIsNone(rec["ttft_ms"])
        self.assertIsNone(rec["stream_seconds"])
        self.assertEqual(rec["token_source"], "chars/4")
        self.assertIsNone(rec["error"])

    def test_aggregates_are_unchanged(self) -> None:
        out = self._run([_res(t_first=100.5, t_done=102.5, tokens=200)] * 4)
        for k in ("ttft_ms_first", "ttft_ms_steady_p50", "ttft_ms_steady_p95",
                  "tps_sustained_p50", "tps_token_sources", "n_samples", "n_errors",
                  "leak_rate", "leaked_markers"):
            self.assertIn(k, out)
        self.assertEqual(out["ttft_ms_first"], 500.0)
        self.assertEqual(out["tps_sustained_p50"], 100.0)


class PrometheusCountersTest(unittest.TestCase):
    _TEXT = (
        "# HELP vllm:prefix_cache_hits Prefix cache hits, in terms of number of cached tokens.\n"
        "# TYPE vllm:prefix_cache_hits counter\n"
        'vllm:prefix_cache_hits_total{engine="0",model_name="Q"} 432.0\n'
        'vllm:prefix_cache_hits_total{engine="1",model_name="Q"} 8.0\n'
        'vllm:prefix_cache_hits_created{engine="0",model_name="Q"} 1.7e+09\n'
        'vllm:spec_decode_num_drafts_total{engine="0",model_name="Q"} 120\n'
        'vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="Q"} 210\n'
        'sglang:cached_tokens_total{cache_source="device"} 10\n'
        'sglang:cached_tokens_total{cache_source="host"} 5\n'
        "process_open_fds 12\n"
        "garbage line without a value\n"
    )

    def test_samples_are_summed_over_label_sets(self) -> None:
        got = _bench_core.parse_prometheus_counters(self._TEXT, _bench_core.ENGINE_COUNTERS)
        self.assertEqual(got["vllm:prefix_cache_hits_total"], 440.0)
        self.assertEqual(got["sglang:cached_tokens_total"], 15.0)
        self.assertEqual(got["vllm:spec_decode_num_drafts_total"], 120.0)

    def test_an_absent_counter_is_absent_not_zero(self) -> None:
        got = _bench_core.parse_prometheus_counters(self._TEXT, _bench_core.ENGINE_COUNTERS)
        self.assertNotIn("vllm:prefix_cache_queries_total", got)
        self.assertNotIn("sglang:prompt_tokens_total", got)

    def test_created_samples_and_comments_are_not_counters(self) -> None:
        got = _bench_core.parse_prometheus_counters(self._TEXT, _bench_core.ENGINE_COUNTERS)
        self.assertNotIn("vllm:prefix_cache_hits_created", got)
        self.assertEqual(set(got) - set(_bench_core.ENGINE_COUNTERS), set())

    def test_without_a_filter_every_sample_is_returned(self) -> None:
        got = _bench_core.parse_prometheus_counters(self._TEXT)
        self.assertEqual(got["process_open_fds"], 12.0)
        self.assertIn("vllm:prefix_cache_hits_created", got)
        self.assertNotIn("garbage", got)

    def test_the_counter_set_names_both_engines(self) -> None:
        for name in ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
                     "vllm:spec_decode_num_drafts_total",
                     "vllm:spec_decode_num_accepted_tokens_total",
                     "sglang:cached_tokens_total", "sglang:prompt_tokens_total"):
            self.assertIn(name, _bench_core.ENGINE_COUNTERS)


if __name__ == "__main__":
    unittest.main()
