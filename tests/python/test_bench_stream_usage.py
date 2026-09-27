"""The streaming helper must read the trailing usage-only SSE chunk.

With `stream_options.include_usage`, an OpenAI-compatible engine ends the
stream with ONE more chunk that carries `usage` and an EMPTY `choices` list
(that is the spec: no delta, just the totals). vLLM 0.22 / 0.28 and SGLang
send exactly that. `stream_chat_completion` skipped every chunk without
choices BEFORE it looked at `usage`, so `completion_tokens` stayed 0 on those
backends and every TPS number fell back to the chars/4 estimate -- which
under-counts code (about 3 chars per token) by 10-25 %. Measured 2026-09-22
on Qwen3.8-27B-MTP-devai-NVFP4 through the router: the engine's own counter
said 78-84 tok/s while the harness reported 72.8. Ollama puts `usage` on the
same chunk as the final delta, so its rows were never affected -- the bias
was one-sided, against vLLM and SGLang.

Stdlib unittest only; the HTTP primitive is faked, nothing is served.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bench import _bench_core  # noqa: E402


def _chunk(delta: dict | None, finish: str | None = None, usage: dict | None = None) -> str:
    obj: dict = {"id": "x", "object": "chat.completion.chunk", "choices": []}
    if delta is not None:
        obj["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
    if usage is not None:
        obj["usage"] = usage
    return json.dumps(obj)


def _fake_stream(payloads: list[str]):
    def fake(url: str, body: dict, timeout: float = 600.0):
        for i, p in enumerate(payloads):
            yield (100.0 + i, p)
    return fake


class TrailingUsageChunkTest(unittest.TestCase):
    def test_usage_only_final_chunk_sets_completion_tokens(self):
        # Arrange: vLLM/SGLang shape -- usage arrives AFTER the last delta,
        # on a chunk whose choices list is empty.
        payloads = [
            _chunk({"role": "assistant", "content": ""}),
            _chunk({"content": "def f():"}),
            _chunk({"content": " pass"}, finish="length"),
            _chunk(None, usage={"prompt_tokens": 55, "completion_tokens": 12,
                                "total_tokens": 67}),
            "[DONE]",
        ]
        with mock.patch.object(_bench_core, "http_post_stream", _fake_stream(payloads)):
            # Act
            res = _bench_core.stream_chat_completion("http://router", {"model": "m"})
        # Assert: the engine's count wins over the chars/4 fallback (13 chars -> 3).
        self.assertEqual(res["completion_tokens"], 12)
        self.assertEqual(res["effective_tokens"], 12)
        self.assertEqual(res["content"], "def f(): pass")
        self.assertEqual(res["finish_reason"], "length")

    def test_usage_on_final_delta_chunk_still_read(self):
        # Ollama shape -- usage shares the chunk with the final delta.
        payloads = [
            _chunk({"content": "hello"}, finish="stop",
                   usage={"prompt_tokens": 5, "completion_tokens": 9}),
            "[DONE]",
        ]
        with mock.patch.object(_bench_core, "http_post_stream", _fake_stream(payloads)):
            res = _bench_core.stream_chat_completion("http://router", {"model": "m"})
        self.assertEqual(res["completion_tokens"], 9)
        self.assertEqual(res["effective_tokens"], 9)

    def test_no_usage_anywhere_falls_back_to_char_estimate(self):
        payloads = [
            _chunk({"content": "x" * 40}, finish="stop"),
            "[DONE]",
        ]
        with mock.patch.object(_bench_core, "http_post_stream", _fake_stream(payloads)):
            res = _bench_core.stream_chat_completion("http://router", {"model": "m"})
        self.assertEqual(res["completion_tokens"], 0)
        self.assertEqual(res["effective_tokens"], 10)

    def test_usage_only_chunk_does_not_move_first_token_time(self):
        # TTFT is the first DELTA; a usage chunk carries no token.
        payloads = [
            _chunk({"content": "a"}, finish="stop"),
            _chunk(None, usage={"completion_tokens": 1}),
            "[DONE]",
        ]
        with mock.patch.object(_bench_core, "http_post_stream", _fake_stream(payloads)):
            res = _bench_core.stream_chat_completion("http://router", {"model": "m"})
        self.assertEqual(res["t_first_token"], 100.0)
        self.assertEqual(res["t_done"], 102.0)


if __name__ == "__main__":
    unittest.main()


class IncludeUsageRequestedTest(unittest.TestCase):
    """The bench asks for usage; without it vLLM and Ollama send none."""

    def _capture(self, body: dict, payloads: list[str]) -> tuple[dict, dict]:
        seen: dict = {}

        def fake(url: str, sent: dict, timeout: float = 600.0):
            seen.update(sent)
            yield from _fake_stream(payloads)(url, sent, timeout)

        with mock.patch.object(_bench_core, "http_post_stream", fake):
            res = _bench_core.stream_chat_completion("http://r", body)
        return seen, res

    def test_request_sets_include_usage(self):
        seen, _ = self._capture({"model": "m", "messages": []}, ["[DONE]"])
        self.assertTrue(seen["stream"])
        self.assertEqual(seen["stream_options"], {"include_usage": True})

    def test_callers_stream_options_are_kept(self):
        seen, _ = self._capture(
            {"model": "m", "stream_options": {"continuous_usage_stats": True}}, ["[DONE]"])
        self.assertEqual(seen["stream_options"],
                         {"continuous_usage_stats": True, "include_usage": True})

    def test_engine_count_wins_over_a_larger_char_estimate(self):
        # 60 characters -> chars/4 = 15, but the engine generated 10 tokens.
        payloads = [_chunk({"content": "x" * 60}, finish="stop"),
                    _chunk(None, usage={"completion_tokens": 10}), "[DONE]"]
        _, res = self._capture({"model": "m"}, payloads)
        self.assertEqual(res["effective_tokens"], 10)
        self.assertEqual(res["token_source"], "usage")

    def test_fallback_is_labelled(self):
        _, res = self._capture({"model": "m"}, [_chunk({"content": "x" * 40}, finish="stop"), "[DONE]"])
        self.assertEqual(res["effective_tokens"], 10)
        self.assertEqual(res["token_source"], "chars/4")


class TokenSourceReachesTheRowTest(unittest.TestCase):
    def test_latency_metrics_keep_the_token_sources(self) -> None:
        from bench import bench_runner
        cache: dict = {}
        latency = {"ttft_ms_first": 1.0, "ttft_ms_steady_p50": 2.0, "ttft_ms_steady_p95": 3.0,
                   "tps_sustained_p50": 40.0, "n_samples": 5, "tps_token_sources": {"usage": 5}}
        bench_runner._latency_metrics_into_row(
            cache, "k", latency, {"alias": "m", "ctx": 32768}, "vllm", "http://r")
        self.assertEqual(cache["k"]["metrics"]["tps_token_sources"], {"usage": 5})
