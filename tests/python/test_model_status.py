"""Unit tests for the model exclusion ledger (Phase 3).

Covers scripts/_model_status.py: record/load/save round-trip, the
sha-stability and vram-stability rules of is_excluded (decision 2), clear,
prune_to_catalog, and the CLI --clear. Stdlib unittest only.

    python3 -m unittest tests.python.test_model_status
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _model_status as MS  # noqa: E402


class TestRoundTrip(unittest.TestCase):
    def test_record_load_save(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ledger.json"
            led = MS.load_ledger(p)
            MS.record_exclusion(led, "gemma-4-31b-it", "vllm", "too_big",
                                detail="58 GB > 24 GB", repo="google/gemma-4-31b-it",
                                host_vram=24, ctx=32768)
            MS.save_ledger(led, p, host_vram_gb=24)
            led2 = MS.load_ledger(p)
            self.assertEqual(MS.exclusion_reason(led2, "gemma-4-31b-it", "vllm"),
                             "too_big")
            self.assertEqual(led2["_meta"]["host_vram_gb"], 24.0)

    def test_missing_file_is_empty(self) -> None:
        led = MS.load_ledger(Path("/nonexistent/x.json"))
        self.assertEqual(led["models"], {})

    def test_invalid_reason_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MS.record_exclusion({}, "m", "vllm", "bogus")


class TestStabilityRules(unittest.TestCase):
    def _led(self, reason, *, host_vram=24, sha="abc"):
        led = {"models": {}}
        MS.record_exclusion(led, "m", "vllm", reason, host_vram=host_vram, sha=sha)
        return led

    def test_unsupported_arch_vram_and_sha_independent(self) -> None:
        led = self._led("unsupported_arch")
        self.assertTrue(MS.is_excluded(led, "m", "vllm", host_vram=24, sha="abc"))
        self.assertTrue(MS.is_excluded(led, "m", "vllm", host_vram=80, sha="zzz"))

    def test_too_big_survives_sha_but_not_vram_change(self) -> None:
        led = self._led("too_big")
        self.assertTrue(MS.is_excluded(led, "m", "vllm", host_vram=24, sha="new"))
        # GPU upgrade -> re-derive
        self.assertFalse(MS.is_excluded(led, "m", "vllm", host_vram=80, sha="abc"))

    def test_oom_rechecked_on_new_sha(self) -> None:
        led = self._led("oom", sha="oldsha")
        self.assertTrue(MS.is_excluded(led, "m", "vllm", host_vram=24, sha="oldsha"))
        self.assertFalse(MS.is_excluded(led, "m", "vllm", host_vram=24, sha="newsha"))

    def test_wrong_backend_not_excluded(self) -> None:
        led = self._led("unsupported_arch")
        self.assertFalse(MS.is_excluded(led, "m", "sglang", host_vram=24))

    def test_unknown_model_not_excluded(self) -> None:
        self.assertFalse(MS.is_excluded({"models": {}}, "m", "vllm", host_vram=24))


class TestFailOpenOnCorruptRows(unittest.TestCase):
    """A JSON-valid but malformed ledger must exclude nothing, never raise."""

    def test_model_value_not_dict(self) -> None:
        led = {"models": {"m": "corrupt"}}
        self.assertFalse(MS.is_excluded(led, "m", "vllm", host_vram=24))
        list(MS.iter_exclusions(led))  # must not raise

    def test_judged_at_not_dict(self) -> None:
        led = {"models": {"m": {"backends": {"vllm": {
            "status": "excluded", "reason": "too_big", "judged_at": "CORRUPT"}}}}}
        self.assertFalse(MS.is_excluded(led, "m", "vllm", host_vram=24))

    def test_non_numeric_judged_vram(self) -> None:
        led = {"models": {"m": {"backends": {"vllm": {
            "status": "excluded", "reason": "too_big",
            "judged_at": {"host_vram_gb": "24gb"}}}}}}
        self.assertFalse(MS.is_excluded(led, "m", "vllm", host_vram=24))

    def test_backends_not_dict(self) -> None:
        led = {"models": {"m": {"backends": "nope"}}}
        self.assertFalse(MS.is_excluded(led, "m", "vllm", host_vram=24))
        list(MS.iter_exclusions(led))  # must not raise


class TestClearAndPrune(unittest.TestCase):
    def test_clear_one_backend_then_name(self) -> None:
        led = {"models": {}}
        MS.record_exclusion(led, "m", "vllm", "too_big", host_vram=24)
        MS.record_exclusion(led, "m", "sglang", "too_big", host_vram=24)
        self.assertTrue(MS.clear(led, "m", "vllm"))
        self.assertIn("m", led["models"])           # sglang remains
        self.assertTrue(MS.clear(led, "m", "sglang"))
        self.assertNotIn("m", led["models"])         # name gone when empty

    def test_prune_to_catalog(self) -> None:
        led = {"models": {}}
        MS.record_exclusion(led, "keep", "vllm", "too_big", host_vram=24)
        MS.record_exclusion(led, "gone", "vllm", "too_big", host_vram=24)
        n = MS.prune_to_catalog(led, {"keep"})
        self.assertEqual(n, 1)
        self.assertIn("keep", led["models"])
        self.assertNotIn("gone", led["models"])


class TestCli(unittest.TestCase):
    def test_clear_via_cli(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ledger.json"
            led = MS.load_ledger(p)
            MS.record_exclusion(led, "m", "vllm", "too_big", host_vram=24)
            MS.save_ledger(led, p)
            rc = MS._main(["--ledger", str(p), "--clear", "m::vllm"])
            self.assertEqual(rc, 0)
            self.assertFalse(MS.is_excluded(MS.load_ledger(p), "m", "vllm",
                                            host_vram=24))

    def test_list_runs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ledger.json"
            self.assertEqual(MS._main(["--ledger", str(p)]), 0)


if __name__ == "__main__":
    unittest.main()
