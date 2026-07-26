"""bench-sync: classification, ledger separation, and budget honesty.

Populating the leaderboard was a state transition done by hand and by
memory. `plan_bench()` makes the diff explicit; these tests pin the parts
that are easy to get subtly wrong and expensive to notice.

Three groups:

1. **Classification.** Every target lands in exactly one bucket, and the
   priority order is deliberate -- an excluded model is not also 'new', a
   dropped row is not also 'stale'. Also pinned: an UNSTAMPED row is
   never 'stale'. Guessing there would either force a needless full
   re-bench (hours of GPU) or hide a real drift.

2. **Ledger separation.** Bench verdicts say how a model PERFORMED, not
   whether it loads, so they must never gate download or probe. This is
   the group that catches the real trap: `_VRAM_DEPENDENT_REASONS` is
   DERIVED from `VALID_REASONS`, so merely adding a reason opts it INTO
   is_excluded()'s gating -- the opposite of what is wanted. The plan's
   own note (that the new reasons would fail open "by construction") was
   written against an older hand-written allowlist and no longer holds.

3. **Budget honesty.** A capped run must say what it left undone. Silent
   truncation reads as full coverage.

Stdlib unittest only.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load(name: str, rel: str):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bs = _load("bench_sync", "scripts/bench-sync.py")
MS = _load("model_status_bs", "scripts/_model_status.py")
runner = _load("bench_runner_bs", "scripts/bench/bench_runner.py")

TASKS = ("gsm8k", "humaneval", "tools", "leak")
ENV_NOW = "env-current"
IMG_NOW = "sha256:aaaa"


def _target(key="repo/M@abc", alias="M", ctx=131072, sha="abc"):
    return {"key": key, "alias": alias, "ctx": ctx,
            "entry": {"sha": sha, "aliases": [alias]}}


def _complete_row(**over):
    row = {
        # Real cache spellings, not the user-facing names. Getting this
        # wrong is the bug this fixture exists to catch.
        "tasks": {"gsm8k_subset_100": {}, "humaneval_subset_50": {},
                  "tools_use_20": {}, "leak_probe": {}},
        "host_env_id": ENV_NOW,
        "backend_image_digest": IMG_NOW,
    }
    row.update(over)
    return row


def _classify(target, *, row=None, ledger=None, tasks=TASKS,
              env=ENV_NOW, image=IMG_NOW):
    cache = {target["key"]: row} if row is not None else {}
    return bs.classify_target(
        target, backend="vllm", bench_cache=cache,
        ledger=ledger if ledger is not None else MS._empty(),
        required_tasks=tasks, current_env_id=env,
        current_image_digest=image, runner=runner)


class ClassificationTest(unittest.TestCase):
    def test_no_row_is_new(self):
        cls, _ = _classify(_target())
        self.assertEqual(cls, "new")

    def test_complete_and_stamped_is_current(self):
        cls, _ = _classify(_target(), row=_complete_row())
        self.assertEqual(cls, "current")

    def test_cache_task_spellings_count_as_complete(self):
        """Rows store `tools_use_20` / `leak_probe`; callers ask for
        `tools` / `leak`. A stripper that does not know the mapping
        reported `tools` missing on 9 of 10 already-benched rows, which
        would have re-run the entire leaderboard."""
        got = bs._row_tasks(_complete_row(), runner)
        self.assertEqual(got, {"gsm8k", "humaneval", "tools", "leak"})

    def test_humaneval_plus_is_not_bucketed_as_humaneval(self):
        row = _complete_row(tasks={"humaneval_plus_subset_50": {}})
        self.assertEqual(bs._row_tasks(row, runner), {"humaneval_plus"})

    def test_missing_task_is_incomplete(self):
        row = _complete_row(tasks={"gsm8k_subset_100": {}})
        cls, why = _classify(_target(), row=row)
        self.assertEqual(cls, "incomplete")
        self.assertIn("humaneval", why)

    def test_changed_host_env_is_stale(self):
        row = _complete_row(host_env_id="env-old")
        cls, _ = _classify(_target(), row=row)
        self.assertEqual(cls, "stale_env")

    def test_changed_image_is_stale(self):
        row = _complete_row(backend_image_digest="sha256:bbbb")
        cls, _ = _classify(_target(), row=row)
        self.assertEqual(cls, "stale_image")

    def test_unstamped_row_is_never_stale(self):
        """A row written before the field existed carries no evidence
        either way. Calling it stale forces hours of needless GPU time;
        calling it fresh hides a real drift. It is reported as current
        with the reason spelled out."""
        for field in ("host_env_id", "backend_image_digest"):
            with self.subTest(missing=field):
                row = _complete_row()
                del row[field]
                cls, why = _classify(_target(), row=row)
                self.assertEqual(cls, "current")
                self.assertIn("unstamped", why)

    def test_drop_flag_outranks_staleness(self):
        """Re-benching a dropped row burns an hour reproducing a verdict
        the operator already has."""
        row = _complete_row(host_env_id="env-old",
                            drop_recommendation={"reason": "leak"})
        cls, why = _classify(_target(), row=row)
        self.assertEqual(cls, "dropped")
        self.assertIn("leak", why)

    def test_ledger_exclusion_outranks_everything(self):
        ledger = MS._empty()
        MS.record_bench_verdict(ledger, "M", "vllm", "bench_dropped",
                                ctx=131072, sha="abc")
        cls, why = _classify(_target(), ledger=ledger)
        self.assertEqual(cls, "excluded")
        self.assertIn("bench_dropped", why)

    def test_narrowed_task_list_does_not_report_permanent_incompleteness(self):
        row = _complete_row(tasks={"gsm8k_subset_100": {}})
        cls, _ = _classify(_target(), row=row, tasks=("gsm8k",))
        self.assertEqual(cls, "current")


class LedgerSeparationTest(unittest.TestCase):
    """Bench verdicts must not gate download or probe."""

    def setUp(self):
        self.ledger = MS._empty()

    def test_bench_dropped_does_not_gate_probe_or_download(self):
        MS.record_bench_verdict(self.ledger, "M", "vllm", "bench_dropped",
                                ctx=131072, sha="abc")
        self.assertFalse(
            MS.is_excluded(self.ledger, "M", "vllm", host_vram=24, sha="abc"),
            "a bench verdict must never stop a model being probed or "
            "downloaded -- only is_bench_excluded() may act on it")
        self.assertTrue(
            MS.is_bench_excluded(self.ledger, "M", "vllm", ctx=131072, sha="abc"))

    def test_bench_reasons_are_excluded_from_the_derived_gating_tuple(self):
        """The trap. _VRAM_DEPENDENT_REASONS is derived from
        VALID_REASONS, so adding a reason opts it INTO is_excluded()'s
        gating unless it is explicitly subtracted."""
        for r in MS._BENCH_REASONS:
            with self.subTest(reason=r):
                self.assertNotIn(r, MS._VRAM_DEPENDENT_REASONS)
                self.assertNotIn(r, MS._VRAM_INDEPENDENT_REASONS)

    def test_verdict_is_vram_independent(self):
        """A leak is a property of the model, not the card."""
        MS.record_bench_verdict(self.ledger, "M", "vllm", "bench_dropped",
                                ctx=131072)
        for vram in (24, 48, 80):
            self.assertTrue(MS.is_bench_excluded(
                self.ledger, "M", "vllm", ctx=131072))

    def test_verdict_applies_at_judged_ctx_and_above_only(self):
        MS.record_bench_verdict(self.ledger, "M", "vllm", "bench_dropped",
                                ctx=131072)
        self.assertFalse(MS.is_bench_excluded(self.ledger, "M", "vllm", ctx=32768))
        self.assertTrue(MS.is_bench_excluded(self.ledger, "M", "vllm", ctx=131072))
        self.assertTrue(MS.is_bench_excluded(self.ledger, "M", "vllm", ctx=262144))

    def test_unscoped_verdict_applies_everywhere(self):
        MS.record_bench_verdict(self.ledger, "M", "vllm", "bench_dropped")
        self.assertTrue(MS.is_bench_excluded(self.ledger, "M", "vllm", ctx=32768))

    def test_requant_reopens_the_question(self):
        MS.record_bench_verdict(self.ledger, "M", "vllm", "bench_dropped",
                                ctx=131072, sha="abc")
        self.assertFalse(MS.is_bench_excluded(
            self.ledger, "M", "vllm", ctx=131072, sha="def"))

    def test_bench_failed_needs_repetition(self):
        """One failure is usually a cold-start timeout or a recreate
        landing mid-request. Excluding on the first would shrink the
        leaderboard on infrastructure noise."""
        MS.record_bench_verdict(self.ledger, "M", "vllm", "bench_failed", ctx=32768)
        self.assertFalse(MS.is_bench_excluded(self.ledger, "M", "vllm", ctx=32768))
        e = MS.record_bench_verdict(self.ledger, "M", "vllm", "bench_failed",
                                    ctx=32768)
        self.assertEqual(e["attempts"], 2)
        self.assertTrue(MS.is_bench_excluded(self.ledger, "M", "vllm", ctx=32768))

    def test_rejects_a_non_bench_reason(self):
        with self.assertRaises(ValueError):
            MS.record_bench_verdict(self.ledger, "M", "vllm", "too_big")

    def test_malformed_ledger_fails_open(self):
        self.assertFalse(MS.is_bench_excluded({}, "M", "vllm", ctx=1))
        self.assertFalse(MS.is_bench_excluded(
            {"models": "not-a-dict"}, "M", "vllm", ctx=1))


class QueueAndBudgetTest(unittest.TestCase):
    def _plan(self):
        return {
            "new": [{"backend": "vllm", "key": "k1", "alias": "A", "ctx": 1,
                     "reason": ""}],
            "incomplete": [{"backend": "vllm", "key": "k2", "alias": "B",
                            "ctx": 1, "reason": ""}],
            "stale_env": [{"backend": "sglang", "key": "k3", "alias": "C",
                           "ctx": 1, "reason": ""}],
            "stale_image": [],
            "dropped": [{"backend": "vllm", "key": "k4", "alias": "D",
                         "ctx": 1, "reason": ""}],
            "excluded": [{"backend": "vllm", "key": "k5", "alias": "E",
                          "ctx": 1, "reason": ""}],
            "current": [{"backend": "vllm", "key": "k6", "alias": "F",
                         "ctx": 1, "reason": ""}],
        }

    def test_queue_excludes_dropped_current_and_excluded(self):
        got = [r["alias"] for r in bs.needs_bench(self._plan())]
        self.assertEqual(got, ["A", "B", "C"])

    def test_budget_truncation_is_announced(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            bs.print_plan(self._plan(), host_vram_gb=24, max_targets=1)
        out = buf.getvalue()
        self.assertIn("would bench 1 of 3", out)
        self.assertIn("left unbenched", out,
                      "a silent cap reads as full coverage")

    def test_no_note_when_the_budget_covers_everything(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            bs.print_plan(self._plan(), host_vram_gb=24, max_targets=0)
        self.assertNotIn("left unbenched", buf.getvalue())


class ImageStampTest(unittest.TestCase):
    """`stale(image)` is only possible if rows carry the digest."""

    def test_update_row_stamps_the_digest(self):
        core = _load("bench_core_bs", "scripts/bench/_bench_core.py")
        cache: dict = {}
        core.update_row(cache, "k", model="M", backend="vllm",
                        router_endpoint="http://r", context=32768,
                        backend_image_digest="sha256:dead")
        self.assertEqual(cache["k"]["backend_image_digest"], "sha256:dead")

    def test_absent_digest_leaves_the_field_off_rather_than_writing_null(self):
        """A null would be indistinguishable from a real value to a
        careless reader; absence is what classify_target checks for."""
        core = _load("bench_core_bs2", "scripts/bench/_bench_core.py")
        cache: dict = {}
        core.update_row(cache, "k", model="M", backend="vllm",
                        router_endpoint="http://r", context=32768,
                        backend_image_digest=None)
        self.assertNotIn("backend_image_digest", cache["k"])

    def test_probe_image_digest_reads_the_probe_cache_meta(self):
        d = runner.probe_image_digest("vllm")
        self.assertTrue(d is None or isinstance(d, str))

    def test_probe_image_digest_is_none_for_an_unknown_backend(self):
        self.assertIsNone(runner.probe_image_digest("nosuchbackend"))


class MakefileWiringTest(unittest.TestCase):
    def test_targets_exist(self):
        mk = (REPO_ROOT / "Makefile").read_text()
        for target in ("bench-plan:", "bench-sync:"):
            self.assertIn(target, mk)
        self.assertIn("BENCH_MAX_TARGETS", mk)

    def test_bench_sync_warns_it_is_gpu_exclusive(self):
        mk = (REPO_ROOT / "Makefile").read_text()
        block = mk.split("bench-sync:", 1)[1].split("\n\n", 1)[0]
        self.assertIn("GPU-exclusive", block,
                      "the target must announce that it cannot overlap an "
                      "interactive session")


class BenchRepoPatternTest(unittest.TestCase):
    """execute() filters the runner with BENCH_REPO. That regex is matched
    against PROBE-cache keys, but discover_models hands back BENCH-cache
    keys -- which carry a ::<backend>::<ctx> suffix the probe keys do not.

    The first implementation passed the bench key through unchanged, so
    the alternation matched nothing and every single run died with "no
    fitting <backend> models in probe cache". The plan looked perfect and
    zero work happened. These tests exist so that cannot recur silently.
    """

    def test_probe_key_strips_the_backend_and_ctx_suffix(self):
        self.assertEqual(
            bs.probe_key("openai/gpt-oss-20b@6cee5e81ee83::sglang::131072"),
            "openai/gpt-oss-20b@6cee5e81ee83")

    def test_probe_key_handles_ollama_digest_keys(self):
        self.assertEqual(
            bs.probe_key("5571076f3d70050487b26b341705799e::ollama::262144"),
            "5571076f3d70050487b26b341705799e")

    def test_probe_key_is_idempotent_on_an_already_bare_key(self):
        self.assertEqual(bs.probe_key("org/M@abc"), "org/M@abc")

    def test_built_pattern_matches_the_bare_probe_key(self):
        """The end-to-end property: whatever execute() builds must
        actually select the model inside the runner."""
        import re
        bench_key = "ykarout/Qwen3.5-9B-NVFP4@bd8c8f493d2e::sglang::196608"
        probe = "ykarout/Qwen3.5-9B-NVFP4@bd8c8f493d2e"
        rx = re.compile(bs._escape(bs.probe_key(bench_key)))
        self.assertTrue(rx.search(probe),
                        "BENCH_REPO pattern does not match the probe key it "
                        "is supposed to select")

    def test_regex_metacharacters_in_a_repo_name_are_escaped(self):
        """Qwen3.5's dot would otherwise match any character, and could
        select a different model."""
        import re
        rx = re.compile(bs._escape(bs.probe_key("org/Qwen3.5-9B@abc::vllm::32768")))
        self.assertTrue(rx.search("org/Qwen3.5-9B@abc"))
        self.assertFalse(rx.search("org/Qwen3X5-9B@abc"))

    def test_two_ctx_tiers_of_one_model_collapse_to_one_alternative(self):
        """Both tiers share a probe key; emitting it twice would be
        harmless but noisy, and hides the real cardinality."""
        keys = ["org/M@abc::vllm::32768", "org/M@abc::vllm::131072"]
        self.assertEqual(len({bs.probe_key(k) for k in keys}), 1)


if __name__ == "__main__":
    unittest.main()
