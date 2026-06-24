"""Unit tests for the probe failure classifier (Phase 1).

Covers scripts/_probe_hf_common.py:_failure_excerpt + classify_failure_logs:
the saved excerpt must capture the ROOT cause (near the top of a long vLLM
traceback), not just the generic "Engine core initialization failed" tail
that buried the gemma-4 cause. Pattern matching runs against the full log.

Stdlib unittest only. Run with:
    python3 -m unittest tests.python.test_probe_classify
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _probe_hf_common as P  # noqa: E402


def _long_log(root_cause: str) -> str:
    """A realistic failed-launch log: startup noise, the real cause, a long
    traceback, then the generic wrapper at the very end."""
    return (
        "INFO startup\n" + "INFO loading weights\n" * 8
        + root_cause + "\n"
        + "\n".join(f"  File \"x.py\", line {i}, in f" for i in range(100))
        + "\nRuntimeError: Engine core initialization failed. "
          "See root cause above. Failed core proc(s): {}\n"
    )


class TestFailureExcerpt(unittest.TestCase):
    def test_captures_root_cause_not_just_tail(self) -> None:
        logs = _long_log("ValueError: Model architectures ['FooForCausalLM'] "
                         "are not supported for now")
        excerpt = P._failure_excerpt(logs)
        self.assertIn("Model architectures", excerpt)   # root cause preserved
        # bounded
        self.assertLessEqual(len(excerpt.splitlines()), 121)

    def test_short_log_kept_whole(self) -> None:
        short = "line1\nValueError: boom\nline3"
        self.assertEqual(P._failure_excerpt(short), short)

    def test_no_anchor_falls_back_to_tail(self) -> None:
        logs = "\n".join(f"info line {i}" for i in range(300))
        excerpt = P._failure_excerpt(logs)
        self.assertIn("info line 299", excerpt)          # tail present
        self.assertNotIn("info line 0", excerpt)         # head dropped


class TestClassifyFailureLogs(unittest.TestCase):
    def test_arch_match_through_full_log(self) -> None:
        # The arch line is near the TOP, buried under 100 traceback lines;
        # it must still classify arch (full-log match) and land in excerpt.
        logs = _long_log("ValueError: Model architectures ['X'] are not "
                         "supported for now")
        rec = P.classify_failure_logs(logs)
        self.assertEqual(rec["kind"], "arch")
        self.assertEqual(rec["matched_pattern"], "Model architectures")
        self.assertIn("Model architectures", rec["log_excerpt"])

    def test_oom_classified(self) -> None:
        rec = P.classify_failure_logs(_long_log("torch.cuda.OutOfMemoryError: "
                                                "CUDA out of memory"))
        self.assertEqual(rec["kind"], "oom_startup")

    def test_unknown_is_infra(self) -> None:
        # A genuine infra failure (no arch/quant/oom marker) stays infra --
        # NOT forced terminal. (gemma-4's MM-config failure is one of these:
        # it is fixed with a recovery flag, not by excluding the model.)
        rec = P.classify_failure_logs(_long_log(
            "Chunked MM input disabled but max_tokens_per_mm_item (2496) is "
            "larger than max_num_batched_tokens (2048)"))
        self.assertEqual(rec["kind"], "infra")


class TestShaStability(unittest.TestCase):
    """Phase 2: terminal verdicts survive a re-quant; orphans pruned."""

    def _entry(self, repo, sha, capability):
        return {"schema_version": 2, "repo": repo, "sha": sha,
                "capability": capability, "evidence": {"kind": "arch"},
                "probes": {}, "aliases": []}

    def test_carry_forward_unsupported_arch(self) -> None:
        cache = {"r/M@oldsha": self._entry("r/M", "oldsha",
                                           P.Capability.UNSUPPORTED_ARCH)}
        new = P.ensure_entry(cache, "r/M@newsha", "r/M", "newsha", "M", 2,
                             "hf", 10.0)
        self.assertEqual(new["capability"], P.Capability.UNSUPPORTED_ARCH)
        self.assertEqual(new.get("carried_from_sha"), "oldsha")

    def test_oom_not_carried_forward(self) -> None:
        cache = {"r/M@oldsha": self._entry("r/M", "oldsha", P.Capability.ERROR)}
        new = P.ensure_entry(cache, "r/M@newsha", "r/M", "newsha", "M", 2,
                             "hf", 10.0)
        self.assertEqual(new["capability"], P.Capability.UNKNOWN)  # re-checked
        self.assertNotIn("carried_from_sha", new)

    def test_prune_drops_orphan_keeps_current(self) -> None:
        cache = {
            "r/M@old": self._entry("r/M", "old", P.Capability.STRUCTURED),
            "r/M@cur": self._entry("r/M", "cur", P.Capability.STRUCTURED),
        }
        catalog = [{"repo": "r/M", "sha": "cur"}]
        n = P.prune_orphaned_shas(cache, catalog)
        self.assertEqual(n, 1)
        self.assertIn("r/M@cur", cache)
        self.assertNotIn("r/M@old", cache)

    def test_prune_keeps_last_entry_when_no_current(self) -> None:
        # Current sha never probed (model not on disk) -> keep the only data.
        cache = {"r/M@old": self._entry("r/M", "old", P.Capability.UNSUPPORTED_ARCH)}
        catalog = [{"repo": "r/M", "sha": "cur"}]
        n = P.prune_orphaned_shas(cache, catalog)
        self.assertEqual(n, 0)
        self.assertIn("r/M@old", cache)


class TestEntryFitsAnywhere(unittest.TestCase):
    """Drives the un-exclude-on-recovery path: a model that loaded anywhere."""

    def test_true_when_any_cell_fits(self) -> None:
        entry = {"probes": {"24": {"32768": {"fits": False},
                                   "65536": {"fits": True}}}}
        self.assertTrue(P._entry_fits_anywhere(entry))

    def test_false_when_no_cell_fits(self) -> None:
        entry = {"probes": {"24": {"32768": {"fits": False}}}}
        self.assertFalse(P._entry_fits_anywhere(entry))

    def test_handles_missing_or_malformed_probes(self) -> None:
        self.assertFalse(P._entry_fits_anywhere({}))
        self.assertFalse(P._entry_fits_anywhere({"probes": {"24": "bad"}}))


class TestOomEverywhere(unittest.TestCase):
    """Drives the ledger `oom` exclusion: fits nowhere + OOM failures."""

    def test_oom_at_all_tiers(self) -> None:
        entry = {"probes": {"24": {
            "32768": {"fits": False, "evidence": {"kind": "oom_startup"}},
            "65536": {"fits": False, "evidence": {"kind": "implied_spill"}}}}}
        self.assertTrue(P._entry_oom_everywhere(entry))

    def test_fits_somewhere_is_not_oom(self) -> None:
        entry = {"probes": {"24": {
            "32768": {"fits": True},
            "65536": {"fits": False, "evidence": {"kind": "oom_startup"}}}}}
        self.assertFalse(P._entry_oom_everywhere(entry))

    def test_arch_failure_is_not_oom(self) -> None:
        # A terminal arch failure is owned by unsupported_arch, not oom.
        entry = {"probes": {"24": {
            "32768": {"fits": False, "evidence": {"kind": "arch"}}}}}
        self.assertFalse(P._entry_oom_everywhere(entry))

    def test_infra_failure_is_not_oom(self) -> None:
        # Genuine infra (retryable) must NOT be recorded as a durable oom.
        entry = {"probes": {"24": {
            "32768": {"fits": False, "evidence": {"kind": "infra"}}}}}
        self.assertFalse(P._entry_oom_everywhere(entry))


if __name__ == "__main__":
    unittest.main()
