"""The long-context probe's prompts share no prefix across requests.

vLLM and SGLang cache KV by prefix, hashed in blocks chained from the start
of the prompt. `_build_long_prompt` used to open every prompt with the same
filler, so a re-run on a warm engine -- or a series of depths, each prompt
extending the last, as in the 2026-09-21 depth runs -- was served partly
from the cache (18.7-34.6 % of prompt tokens there) and TTFT stopped
measuring prefill. Each request now opens with a unique salt.

Stdlib unittest only; the HTTP helper is faked.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bench import bench_longctx  # noqa: E402


def _run_capturing() -> tuple[str, dict]:
    sent: dict = {}

    def fake_stream(url, body, timeout=600.0):
        sent.update(body)
        return {"content": "x", "reasoning_content": "", "completion_tokens": 5,
                "prompt_tokens": 1234, "effective_tokens": 5, "token_source": "usage",
                "t_open": 0.0, "t_first_token": 1.0, "t_done": 2.0, "finish_reason": "stop"}

    with mock.patch.object(bench_longctx, "stream_chat_completion", fake_stream):
        res = bench_longctx.run(model="m", router_url="http://r", ctx_target=8192)
    return sent["messages"][0]["content"], res


class SaltTest(unittest.TestCase):
    def test_two_requests_differ_from_the_first_characters(self) -> None:
        a, _ = _run_capturing()
        b, _ = _run_capturing()
        self.assertNotEqual(a[:48], b[:48])

    def test_salt_opens_the_prompt(self) -> None:
        p = bench_longctx._build_long_prompt(4000, salt="abc")
        self.assertTrue(p.startswith("[run abc]\n"))
        self.assertTrue(p.endswith(bench_longctx._TAIL_INSTRUCTION))

    def test_salt_does_not_grow_the_prompt(self) -> None:
        # The salt comes out of the character budget, not on top of it.
        plain = bench_longctx._build_long_prompt(4000)
        salted = bench_longctx._build_long_prompt(4000, salt="0" * 32)
        self.assertEqual(len(plain), len(salted))

    def test_unsalted_builder_is_unchanged(self) -> None:
        self.assertTrue(bench_longctx._build_long_prompt(4000).startswith("Call me Ishmael."))


class EngineCountTest(unittest.TestCase):
    def test_engine_prompt_token_count_is_recorded(self) -> None:
        _, res = _run_capturing()
        self.assertEqual(res["input_tokens"], 1234)
        self.assertEqual(res["input_tokens_target"], int(8192 * 0.8))


if __name__ == "__main__":
    unittest.main()
