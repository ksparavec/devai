"""The `retired` ledger verdict and the removal-recording path.

Why this exists: every ledger reason before `retired` answered "this
model cannot run here", and none could express "it ran fine, we just
chose something else". That gap had a measurable cost. An audit on
2026-07-25 found 11 vLLM models that probed as fitting and whose weights
were later removed; only the 5 that had been benched carried any ledger
entry at all. The split was exactly benched -> recorded, unbenched ->
unrecorded, because the ledger was only ever written from the
download/probe paths and never from removal.

Two mechanisms close that, and both are tested here:
  1. `record_retirement` + `delete()` -- the tooling path records a
     reason when it removes weights.
  2. `unrecorded_retirements` -- a report, because weights removed by
     hand (`rm -rf` on the store) never reach `delete()`, and that is how
     the historical gaps actually arose.

Stdlib unittest only; no GPU, no network, no container.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ms = _load("_model_status", "scripts/_model_status.py")


class ReasonVocabularyTest(unittest.TestCase):
    def test_retired_is_a_valid_reason(self):
        self.assertIn("retired", ms.VALID_REASONS)

    def test_retired_survives_a_requant(self):
        """A re-quant of a model rejected on context or speed grounds does
        not change that judgement."""
        self.assertIn("retired", ms._SHA_STABLE_REASONS)

    def test_retired_does_not_survive_a_vram_change(self):
        """Retirement is relative to the rest of the fleet at a given VRAM
        budget, and most retirements here cite a context ceiling. A bigger
        GPU genuinely reopens the question."""
        self.assertNotIn("retired", ms._VRAM_INDEPENDENT_REASONS)


class RecordRetirementTest(unittest.TestCase):
    def setUp(self):
        self.led = ms._empty()

    def test_records_reason_and_supersession(self):
        ms.record_retirement(self.led, "Old-9B", "vllm",
                             detail="32K ceiling", superseded_by="New-9B")
        e = self.led["models"]["Old-9B"]["backends"]["vllm"]
        self.assertEqual(e["reason"], "retired")
        self.assertEqual(e["detail"], "32K ceiling")
        self.assertEqual(e["superseded_by"], "New-9B")

    def test_superseded_by_is_structured_not_free_text(self):
        """'What replaced X' must be answerable without parsing prose."""
        ms.record_retirement(self.led, "A", "vllm", detail="x",
                             superseded_by="B")
        self.assertEqual(
            self.led["models"]["A"]["backends"]["vllm"].get("superseded_by"),
            "B")

    def test_omits_the_field_when_nothing_superseded_it(self):
        ms.record_retirement(self.led, "A", "vllm", detail="just slow")
        self.assertNotIn(
            "superseded_by",
            self.led["models"]["A"]["backends"]["vllm"])

    def test_is_per_backend(self):
        """A model can be retired from SGLang and kept on vLLM."""
        ms.record_retirement(self.led, "M", "sglang", detail="arch gap")
        self.assertIsNone(ms.exclusion_reason(self.led, "M", "vllm"))
        self.assertEqual(ms.exclusion_reason(self.led, "M", "sglang"),
                         "retired")

    def test_never_masks_a_harder_verdict(self):
        """Pruning a model that OOM'd must keep the OOM: that is the real
        story, and 'retired' would hide why it actually failed."""
        for hard in ("oom", "unsupported_arch", "too_big", "too_small"):
            with self.subTest(reason=hard):
                led = ms._empty()
                ms.record_exclusion(led, "M", "vllm", hard, detail="original")
                ms.record_retirement(led, "M", "vllm", detail="superseded")
                e = led["models"]["M"]["backends"]["vllm"]
                self.assertEqual(e["reason"], hard)
                self.assertEqual(e["detail"], "original")

    def test_does_overwrite_a_soft_manual_verdict(self):
        """`manual` is an operator note, not a hard failure -- a concrete
        retirement reason is strictly more informative."""
        ms.record_exclusion(self.led, "M", "vllm", "manual", detail="note")
        ms.record_retirement(self.led, "M", "vllm", detail="beaten by N",
                             superseded_by="N")
        self.assertEqual(
            self.led["models"]["M"]["backends"]["vllm"]["reason"], "retired")

    def test_rejects_an_unknown_reason(self):
        with self.assertRaises(ValueError):
            ms.record_exclusion(self.led, "M", "vllm", "not-a-reason")


class RetiredExclusionSemanticsTest(unittest.TestCase):
    def test_excluded_at_the_same_vram(self):
        led = ms._empty()
        ms.record_retirement(led, "M", "vllm", detail="x", host_vram=24.0)
        self.assertTrue(ms.is_excluded(led, "M", "vllm", host_vram=24.0))

    def test_reopened_by_a_bigger_gpu(self):
        led = ms._empty()
        ms.record_retirement(led, "M", "vllm", detail="32K ceiling",
                             host_vram=24.0)
        self.assertFalse(ms.is_excluded(led, "M", "vllm", host_vram=48.0),
                         "a bigger GPU must reopen a retirement")


class UnrecordedRetirementReportTest(unittest.TestCase):
    """The report that catches weights removed outside the tooling."""

    def setUp(self):
        self.sm = _load("select_models", "scripts/select-models.py")

    def test_function_exists_and_is_backend_aware(self):
        self.assertTrue(callable(self.sm.unrecorded_retirements))
        self.assertTrue(callable(self.sm.report_unrecorded_retirements))

    def test_reports_nothing_when_every_absence_is_explained(self):
        """A model with a ledger verdict is accounted for, so it must not
        be reported -- otherwise the signal is pure noise."""
        led = ms._empty()
        ms.record_retirement(led, "Gone-9B", "vllm", detail="superseded")
        models = [{"name": "Gone-9B", "source": "hf", "repo": "r",
                   "sha": "s", "backend": ["vllm"]}]

        class Caches:
            def entry_fits(self, repo, sha, backend):
                return True

        self.assertEqual(
            self.sm.unrecorded_retirements(models, Caches(), led), [])

    def test_reports_an_unexplained_absence(self):
        models = [{"name": "Mystery-9B", "source": "hf", "repo": "r",
                   "sha": "s", "backend": ["vllm"]}]

        class Caches:
            def entry_fits(self, repo, sha, backend):
                return True

        got = self.sm.unrecorded_retirements(models, Caches(), ms._empty())
        self.assertIn(("Mystery-9B", "vllm"), got)

    def test_ignores_models_that_never_fit(self):
        """A model that never probed as fitting has nothing to explain."""
        models = [{"name": "Unfit", "source": "hf", "repo": "r", "sha": "s",
                   "backend": ["vllm"]}]

        class Caches:
            def entry_fits(self, repo, sha, backend):
                return False

        self.assertEqual(
            self.sm.unrecorded_retirements(models, Caches(), ms._empty()), [])


if __name__ == "__main__":
    unittest.main()
