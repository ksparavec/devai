"""scripts/stats/vllm_upgrade_accept.py: the 4v acceptance rule's verdicts.

Plan platform-rebuild, Phase 4 step 4v. Every case the deliverable row names
runs on fixed fake inputs in the real on-disk shapes: inspect_ai .eval zips
(header.json + samples/<id>_epoch_1.json, a time-out marked by the sample's
`limit` of type `working`), the bench harness's latency-<alias>.jsonl, and
bench_concurrency's multiturn reports. Numbers are checked against statlib
and, where small enough, by hand.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "stats"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import statlib as L  # noqa: E402
import vllm_upgrade_accept as V  # noqa: E402

OLD, NEW = "0.28.0", "0.30.0"
TAG_OLD = "sha256:" + "a" * 64
TAG_NEW = "b" * 64


def _sample(i: int, outcome: str) -> dict:
    timeout = outcome == "timeout"
    value = "C" if outcome == "right" else "I"   # a time-out is scored wrong
    return {"id": i, "scores": {"match": {"value": value}},
            "limit": {"type": "working", "limit": 900} if timeout else None, "error": None}


def _write_log(path: Path, task: str, outcomes: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("header.json", json.dumps({"status": "success", "eval": {"task": task}}))
        for i, o in enumerate(outcomes, 1):
            z.writestr(f"samples/{i}_epoch_1.json", json.dumps(_sample(i, o)))


def _pairs(a: int, b: int, c: int, d: int) -> list[tuple[str, str]]:
    """a both right, b right on old only, c right on new only, d neither."""
    return ([("right", "right")] * a + [("right", "wrong")] * b
            + [("wrong", "right")] * c + [("wrong", "wrong")] * d)


class _Case(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _builds(self, gsm8k: list[tuple[str, str]], humaneval: list[tuple[str, str]]) -> tuple[Path, Path]:
        old, new = self.tmp / "old", self.tmp / "new"
        for d in (old, new):
            d.mkdir(exist_ok=True)
        for task, pairs in (("gsm8k_task", gsm8k), ("humaneval_task", humaneval)):
            _write_log(old / f"2026-10-08_{task}_x.eval", task, [p[0] for p in pairs])
            _write_log(new / f"2026-10-08_{task}_y.eval", task, [p[1] for p in pairs])
        return old, new

    def _run(self, argv: list[str]) -> tuple[int, dict | None, str]:
        out = self.tmp / "report.json"
        err = io.StringIO()
        with redirect_stderr(err):
            code = V.main(argv + ["--out", str(out)])
        report = json.loads(out.read_text()) if out.exists() else None
        if out.exists():
            out.unlink()
        return code, report, err.getvalue()

    def _accept(self, old: Path, new: Path, *, tag_old: str = TAG_OLD, served_old: str = TAG_OLD,
                tag_new: str = TAG_NEW, served_new: str = TAG_NEW, extra: list[str] = ()):
        return self._run(["accept", "--old-logs", str(old), "--new-logs", str(new),
                          "--old-tag-id", tag_old, "--old-served-id", served_old,
                          "--new-tag-id", tag_new, "--new-served-id", served_new, *extra])


GSM_EVEN = _pairs(1100, 10, 10, 199)     # n = 1319, no difference
HE_EVEN = _pairs(140, 2, 2, 20)          # n = 164, no difference


class VerdictTest(_Case):
    def test_pass(self) -> None:
        code, r, _ = self._accept(*self._builds(GSM_EVEN, HE_EVEN))
        self.assertEqual(code, 0)
        self.assertEqual(r["verdict"], "pass", r["quality"]["verdict_reasons"])
        g = r["quality"]["tasks"]["gsm8k"]
        self.assertEqual(g["n_pairs"], 1319)
        self.assertEqual(g["bound_table"], {"a": 1100, "b": 10, "c": 10, "d": 199})
        self.assertEqual(g["upper_bound"], L.newcombe_paired(1100, 10, 10, 199)[2])
        self.assertLess(g["upper_bound"], 0.03)
        # The plan's expected half-widths: about 1.7 and 4.8 points.
        self.assertAlmostEqual(g["expected_halfwidth_at_10pct_discordance"], 0.0171, places=4)
        self.assertAlmostEqual(r["quality"]["tasks"]["humaneval"]["expected_halfwidth_at_10pct_discordance"],
                               0.0484, places=4)
        self.assertEqual(g["mcnemar_p"], 1.0)

    def test_fail(self) -> None:
        # humaneval: 20 items right only on 0.28.0, 2 only on 0.30.0.
        # McNemar p = 2 * (1 + 22 + 231) / 2^22; Holm doubles the smaller p.
        code, r, _ = self._accept(*self._builds(GSM_EVEN, _pairs(120, 20, 2, 22)))
        self.assertEqual(code, 0)
        h = r["quality"]["tasks"]["humaneval"]
        self.assertAlmostEqual(h["mcnemar_p"], 508 / 2 ** 22, places=12)
        self.assertAlmostEqual(h["mcnemar_p_holm"], 2 * 508 / 2 ** 22, places=12)
        self.assertTrue(h["fail"])
        self.assertEqual(r["verdict"], "fail")

    def test_a_gain_on_the_new_build_is_never_a_failure(self) -> None:
        code, r, _ = self._accept(*self._builds(GSM_EVEN, _pairs(120, 2, 20, 22)))
        self.assertFalse(r["quality"]["tasks"]["humaneval"]["fail"])
        self.assertEqual(r["verdict"], "pass")

    def test_inconclusive(self) -> None:
        # 6 vs 3 discordant on humaneval: no significant loss (p = 260/512),
        # but the bound reaches past the 5-point margin.
        code, r, _ = self._accept(*self._builds(GSM_EVEN, _pairs(130, 6, 3, 25)))
        self.assertEqual(code, 0)
        h = r["quality"]["tasks"]["humaneval"]
        self.assertAlmostEqual(h["mcnemar_p"], 260 / 512, places=12)
        self.assertFalse(h["fail"])
        self.assertGreaterEqual(h["upper_bound"], 0.05)
        self.assertEqual(r["verdict"], "inconclusive")

    def test_a_task_meeting_both_the_bound_and_the_fail_test_fails(self) -> None:
        # gsm8k: 25 vs 5 discordant. Difference 20/1319 = 1.5 points, upper
        # bound inside the 3-point margin -- and a significant loss.
        code, r, _ = self._accept(*self._builds(_pairs(1100, 25, 5, 189), HE_EVEN))
        g = r["quality"]["tasks"]["gsm8k"]
        self.assertTrue(g["bound_below_margin"])
        self.assertTrue(g["fail"])
        self.assertEqual(r["verdict"], "fail")

    def test_an_unpaired_item_makes_pass_impossible(self) -> None:
        old, new = self._builds(GSM_EVEN, HE_EVEN)
        _write_log(next(new.glob("*humaneval*")), "humaneval_task", [p[1] for p in HE_EVEN][:-1])
        code, r, _ = self._accept(old, new)
        self.assertEqual(r["quality"]["tasks"]["humaneval"]["unpaired_or_unscored"],
                         [{"id": "164", OLD: "wrong", NEW: "absent"}])
        self.assertEqual(r["verdict"], "inconclusive")

    def test_a_missing_task_log_is_bad_input(self) -> None:
        old, new = self._builds(GSM_EVEN, HE_EVEN)
        next(new.glob("*humaneval*")).unlink()
        code, r, err = self._accept(old, new)
        self.assertEqual(code, 2)
        self.assertIsNone(r)
        self.assertIn("expected exactly one humaneval log", err)


class TimeoutTest(_Case):
    """A time-out: wrong on 0.30.0, right on 0.28.0 for the bound; wrong for McNemar."""

    def _humaneval(self, extra: list[tuple[str, str]]) -> dict:
        _, r, _ = self._accept(*self._builds(GSM_EVEN, _pairs(140, 0, 0, 20) + extra))
        return r["quality"]["tasks"]["humaneval"]

    def test_timeout_on_the_new_build_only(self) -> None:
        h = self._humaneval([("right", "timeout"), ("wrong", "timeout")])
        # Observed right on 0.28.0 stands: b; observed wrong stands: d.
        self.assertEqual(h["bound_table"], {"a": 140, "b": 1, "c": 0, "d": 21})
        self.assertEqual(h["harness_table"], {"a": 140, "b": 1, "c": 0, "d": 21})
        self.assertEqual(h["timeouts"], {OLD: 0, NEW: 2})

    def test_timeout_on_the_old_build_only(self) -> None:
        h = self._humaneval([("timeout", "right"), ("timeout", "wrong")])
        # The bound: right on 0.28.0, with 0.30.0's observed outcome: a and b.
        self.assertEqual(h["bound_table"], {"a": 141, "b": 1, "c": 0, "d": 20})
        # McNemar: wrong on 0.28.0: c and d.
        self.assertEqual(h["harness_table"], {"a": 140, "b": 0, "c": 1, "d": 21})
        self.assertEqual(h["timeouts"], {OLD: 2, NEW: 0})

    def test_timeout_on_both_builds(self) -> None:
        h = self._humaneval([("timeout", "timeout")])
        self.assertEqual(h["bound_table"], {"a": 140, "b": 1, "c": 0, "d": 20})
        self.assertEqual(h["harness_table"], {"a": 140, "b": 0, "c": 0, "d": 21})
        self.assertEqual(h["upper_bound"], L.newcombe_paired(140, 1, 0, 20)[2])
        self.assertEqual(h["mcnemar_p"], L.mcnemar_exact(0, 0))


class ImageIdTest(_Case):
    def test_equal_tag_ids_are_refused(self) -> None:
        old, new = self._builds(GSM_EVEN, HE_EVEN)
        code, r, err = self._accept(old, new, tag_new=TAG_OLD, served_new="A" * 64)
        self.assertEqual(code, 3)
        self.assertEqual(r["verdict"], "refused")
        self.assertNotIn("quality", r)
        self.assertNotIn("pass", json.dumps(r).lower())
        self.assertNotIn("pass", err.lower())

    def test_a_served_id_other_than_the_tags_is_refused(self) -> None:
        old, new = self._builds(GSM_EVEN, HE_EVEN)
        code, r, _ = self._accept(old, new, served_new="c" * 64)
        self.assertEqual(code, 3)
        self.assertEqual(len(r["refusals"]), 1)
        self.assertIn(NEW, r["refusals"][0])
        self.assertNotIn("pass", json.dumps(r).lower())

    def test_ids_compare_without_prefix_and_case(self) -> None:
        self.assertTrue(V.same_image_id("sha256:" + "AB" * 32, "ab" * 32))
        self.assertTrue(V.same_image_id("ab" * 32, "abababababab"))
        self.assertFalse(V.same_image_id("ab" * 32, "abab"))
        self.assertFalse(V.same_image_id("ab" * 32, "cd" * 32))


def _latency(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _rec(i: int, ttft: float, tokens: int, secs: float, source: str = "usage", error=None) -> dict:
    return {"i": i, "id": f"p{i}", "ttft_ms": ttft, "effective_tokens": tokens, "token_source": source,
            "stream_seconds": secs, "finish_reason": "stop", "error": error}


class SpeedTest(_Case):
    def _manifest(self, name: str, latency: list[dict], **mtp) -> Path:
        d = self.tmp / name
        d.mkdir()
        _latency(d / "latency-row.jsonl", latency)
        row = {"latency": "latency-row.jsonl"}
        for k, v in mtp.items():
            if k.startswith("metrics_"):
                (d / f"{k}.prom").write_text(v)
                row[k] = f"{k}.prom"
            else:
                row[k] = v
        (d / "speed.json").write_text(json.dumps({"rows": {"teacher": row}}))
        return d / "speed.json"

    def test_pairing_keeps_prompts_1_to_39_and_the_decode_filter(self) -> None:
        old = [_rec(0, 9000, 100, 1.0), _rec(1, 100, 100, 1.0), _rec(2, 100, 100, 1.0),
               _rec(3, 100, 100, 1.0), _rec(4, 100, 100, 1.0), _rec(5, 100, 63, 1.0),
               _rec(39, 100, 100, 1.0), _rec(40, 100, 100, 1.0),
               _rec(6, None, None, None, None, error="HTTP 500")]
        new = [_rec(0, 10, 100, 1.0),            # prompt 0: the cold start, excluded
               _rec(1, 200, 125, 1.0),           # ttft x2, decode x1.25
               _rec(2, 50, 100, 1.0, "chars/4"),  # ttft x0.5, decode not engine-counted
               _rec(3, 100, 100, 2.0),           # ttft x1, decode x0.5
               _rec(4, 100, 200, 1.0),           # ttft x1, decode x2
               _rec(5, 100, 100, 1.0),           # ttft x1, decode: 63 tokens on 0.28.0
               _rec(39, 100, 100, 1.0),          # ttft x1, decode x1
               _rec(40, 1, 1000, 1.0),           # beyond prompt 39, excluded
               _rec(6, 100, 100, 1.0)]           # failed on 0.28.0: no pair
        s = V.speed(self._manifest("o", old), self._manifest("n", new))["rows"]["teacher"]
        self.assertEqual(s["prompts_paired"], 6)
        self.assertEqual(s["ttft"]["n"], 6)
        self.assertAlmostEqual(s["ttft"]["geomean_ratio"], 1.0, places=12)   # 2 * 0.5 * 1 * 1 * 1 * 1
        self.assertEqual(s["decode_rate"]["n"], 4)
        self.assertAlmostEqual(s["decode_rate"]["geomean_ratio"], (1.25 * 0.5 * 2 * 1) ** 0.25, places=12)
        lo, hi = s["decode_rate"]["boot95"]
        self.assertLessEqual(lo, s["decode_rate"]["geomean_ratio"])
        self.assertGreaterEqual(hi, s["decode_rate"]["geomean_ratio"])

    def test_mtp_acceptance_length_and_its_missing_cases(self) -> None:
        before = ('vllm:spec_decode_num_drafts_total{engine="0"} 100\n'
                  'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 150\n')
        after = ('vllm:spec_decode_num_drafts_total{engine="0"} 200\n'
                 'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 350\n')
        lat = [_rec(1, 100, 100, 1.0)]
        old = self._manifest("o", lat, metrics_before=before, metrics_after=after,
                             started_before="t0", started_after="t0")
        new = self._manifest("n", lat, metrics_before=before, metrics_after=after,
                             started_before="t0", started_after="t1")
        m = V.speed(old, new)["rows"]["teacher"]["mtp_acceptance_length"]
        self.assertEqual(m[OLD]["value"], 3.0)                 # 1 + 200 / 100
        self.assertIsNone(m[NEW]["value"])
        self.assertIn("relaunched", m[NEW]["missing"])
        self.assertIn("no /metrics", V.mtp_acceptance({"latency": "x"}, self.tmp)["missing"])


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
