"""scripts/stats/vllm_upgrade_accept.py: the 4v prefix-cache retention rule.

Plan platform-rebuild, Phase 4 step 4v. Every case runs on fixed fake
bench_concurrency multiturn reports in their real on-disk shape. (The
0.28.0-against-0.30.0 `accept` cases went with that subcommand, plan E7.)
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "stats"))

import vllm_upgrade_accept as V  # noqa: E402


class _Case(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _run(self, argv: list[str]) -> tuple[int, dict | None, str]:
        out = self.tmp / "report.json"
        err = io.StringIO()
        with redirect_stderr(err):
            code = V.main(argv + ["--out", str(out)])
        report = json.loads(out.read_text()) if out.exists() else None
        if out.exists():
            out.unlink()
        return code, report, err.getvalue()


def _session(salt: str, ttfts: list[float], hits: list[float], *, ok: bool = True,
             complete: bool = True) -> dict:
    turns = [{"turn": t, "ttft_ms": ttft, "ok": ok if t == len(ttfts) else True,
              "completion_tokens": 1024, "token_source": "usage",
              "counters_delta": {V.HITS: h, "vllm:prefix_cache_queries_total": 5000.0}}
             for t, (ttft, h) in enumerate(zip(ttfts, hits), 1)]
    return {"mode": "multiturn", "salt": salt, "min_tokens": 1024, "max_tokens": 1024,
            "metrics_url": "http://devai-vllm-devai:11434/metrics", "metrics_complete": complete,
            "turns": turns}


class RetentionTest(_Case):
    def _arms(self, *, equal_in: str | None = None, none_ttft: float = 800.0) -> tuple[list[str], list[str]]:
        argv = {"default": [], "none": []}
        for i in range(8):
            salt = f"nonce-{i}"
            d = _session(salt, [3000, 1000 + i, 1010 + i, 1020 + i, 1030 + i], [0, 4000, 8000, 12000, 16000])
            nh = [0, 4000, 8000, 12000, 16000] if salt == equal_in else [0, 4864, 8864, 12864, 16864]
            n = _session(salt, [3000] + [none_ttft + i + k for k in (0, 10, 20, 30)], nh)
            for arm, body in (("default", d), ("none", n)):
                p = self.tmp / f"{arm}-{i}.json"
                p.write_text(json.dumps(body))
                argv[arm].append(str(p))
        return argv["default"], argv["none"]

    def _retention(self, default: list[str], none: list[str]):
        return self._run(["retention", "--default", *default, "--none", *none])

    def test_adopted_when_a_and_b_hold(self) -> None:
        code, r, _ = self._retention(*self._arms())
        self.assertEqual(code, 0)
        self.assertTrue(r["rule_a"]["holds"])
        self.assertTrue(r["rule_b"]["holds"])
        self.assertLessEqual(r["rule_b"]["boot95"][1], 1.0)
        self.assertTrue(r["rule_b"]["upper_bound_approximate"])
        s = r["sessions"]["nonce-0"]
        self.assertEqual(s["default"]["cached_tokens_turn2plus"], 40000)
        self.assertEqual(s["none"]["cached_tokens_turn2plus"], 43456)
        self.assertEqual(s["default"]["ttft_median_turn2plus_ms"], 1015)   # turn 1 excluded
        self.assertEqual(r["verdict"], "adopt")
        self.assertIn("not evaluated", r["rule_c"])

    def test_rule_a_failing_in_one_session_keeps_the_default(self) -> None:
        code, r, _ = self._retention(*self._arms(equal_in="nonce-5"))
        self.assertEqual(code, 0)
        self.assertFalse(r["rule_a"]["holds"])
        self.assertEqual(r["rule_a"]["failing_sessions"], ["nonce-5"])
        self.assertTrue(r["rule_b"]["holds"])
        self.assertEqual(r["verdict"], "keep_default")

    def test_rule_b_failing_keeps_the_default(self) -> None:
        code, r, _ = self._retention(*self._arms(none_ttft=1100.0))
        self.assertFalse(r["rule_b"]["holds"])
        self.assertEqual(r["verdict"], "keep_default")

    def test_sessions_that_do_not_pair_are_refused(self) -> None:
        default, none = self._arms()
        code, r, err = self._retention(default[:-1], none)
        self.assertEqual(code, 2)
        self.assertIn("do not pair", err)

    def test_a_failed_turn_or_an_incomplete_readout_is_refused(self) -> None:
        default, none = self._arms()
        bad = self.tmp / "bad.json"
        for body, msg in ((_session("nonce-0", [1, 2], [0, 1], ok=False), "did not succeed"),
                          (_session("nonce-0", [1, 2], [0, 1], complete=False), "incomplete")):
            bad.write_text(json.dumps(body))
            code, _, err = self._retention([str(bad)] + default[1:], none)
            self.assertEqual(code, 2)
            self.assertIn(msg, err)


if __name__ == "__main__":
    unittest.main()
