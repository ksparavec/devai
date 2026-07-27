"""The picker must feed the bench gate a real artefact identity, and must
degrade rather than crash when the ledger helper is older than itself.

Two defects, both in the `is_bench_excluded` call added by 55fd0b7.

1. IDENTITY NEVER ARRIVED. The call passed `sha=decorated.get("sha")`,
   but a picker row has no top-level `sha` -- the identity lives inside
   `probe`, as `sha` for the repo+sha-keyed HF caches and as `digest` for
   the digest-keyed Ollama cache. So `sha` was None for EVERY backend,
   not just Ollama, which silently disabled the ledger's documented
   "a re-quant genuinely re-opens the question" rule everywhere: once a
   verdict was recorded it stuck forever, even after the weights were
   replaced, and only `make model-status CLEAR=` could lift it.

   This mattered in practice, not just in theory. `bench-sync` records
   verdicts with no sha (so the rule is inert for those rows either way),
   but a hand-recorded verdict DOES carry `last_sha` -- the one real
   entry on this fleet does -- and those are exactly the rows the rule
   exists for.

2. THE GUARD CHECKED THE MODULE, NOT THE FUNCTION. `_MS is not None` is
   satisfied by any `_model_status` that imports, including one predating
   `is_bench_excluded`. fcd0a54 made the IMPORT optional precisely
   because devai-agent bind-mounts a fresh picker into an older image,
   but the call site then raised AttributeError on that same skew -- the
   partial-install crash it set out to prevent, and contrary to its own
   rationale that "losing a filter shows extra rows while losing the
   picker shows none".

Stdlib unittest. The identity tests need models on disk and skip without
them; the degradation tests are data-independent.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load(name, rel):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mp = _load("model_picker_verdict_identity", "scripts/model-picker.py")


def _ms():
    return _load("ms_verdict_identity", "scripts/_model_status.py")


class LedgerFixture(unittest.TestCase):
    """Shared setUp: real rows from disk, ledger swapped for a scratch one."""

    def setUp(self):
        self.MS = _ms()
        self.addCleanup(setattr, mp, "_BENCH_LEDGER", mp._BENCH_LEDGER)
        mp._BENCH_LEDGER = self.MS._empty()
        self.models = mp._discover_models()
        rows, _ = mp._build_candidates(self.models, {})
        if not rows:
            self.skipTest("no models on disk to exercise the gate")
        self.rows = rows

    def _with(self, ledger):
        mp._BENCH_LEDGER = ledger
        rows, hidden = mp._build_candidates(self.models, {})
        return [(r["name"], r["backend"]) for r in rows], hidden


class ArtefactIdentityTest(LedgerFixture):
    def test_every_row_yields_an_identity(self):
        """No backend may fall through to None -- that is defect 1."""
        for r in self.rows:
            with self.subTest(name=r["name"], backend=r["backend"]):
                self.assertTrue(
                    mp._artefact_sha(r),
                    f"{r['name']} [{r['backend']}] has no artefact identity; "
                    "the ledger's re-quant rule is inert for it")

    def test_identity_comes_from_the_right_probe_field(self):
        """HF caches key on sha, the Ollama cache on digest. A rename in
        either would silently re-disable the rule, so pin the source."""
        for r in self.rows:
            probe = r.get("probe") or {}
            key = "digest" if r["backend"] == "ollama" else "sha"
            with self.subTest(name=r["name"], backend=r["backend"]):
                self.assertIn(key, probe)
                self.assertEqual(mp._artefact_sha(r), str(probe[key]))

    def test_verdict_matching_the_current_artefact_still_hides(self):
        r = self.rows[0]
        led = self.MS._empty()
        self.MS.record_bench_verdict(
            led, r["name"], r["backend"], "bench_dropped",
            sha=mp._artefact_sha(r))
        rows, hidden = self._with(led)
        self.assertNotIn((r["name"], r["backend"]), rows)
        self.assertEqual(hidden["bench_excluded"], 1)

    def test_verdict_against_a_superseded_artefact_reopens(self):
        """The regression: a re-quant must put the row back on offer."""
        r = self.rows[0]
        led = self.MS._empty()
        self.MS.record_bench_verdict(
            led, r["name"], r["backend"], "bench_dropped",
            sha="0000deadbeef")           # not this checkpoint any more
        rows, hidden = self._with(led)
        self.assertIn((r["name"], r["backend"]), rows,
                      "a verdict on a superseded artefact must not hide the row")
        self.assertEqual(hidden["bench_excluded"], 0)

    def test_verdict_with_no_recorded_sha_still_hides(self):
        """bench-sync records without a sha; those verdicts must keep
        applying, or automated drops would become no-ops."""
        r = self.rows[0]
        led = self.MS._empty()
        self.MS.record_bench_verdict(led, r["name"], r["backend"],
                                     "bench_dropped")
        rows, _ = self._with(led)
        self.assertNotIn((r["name"], r["backend"]), rows)


class MissingHelperDegradesTest(LedgerFixture):
    def test_module_without_the_gate_function_does_not_crash(self):
        """Defect 2: an image-skew _model_status that imports but lacks
        is_bench_excluded must lose the filter, not the picker."""
        class Stale:
            """Predates is_bench_excluded."""

        with mock.patch.object(mp, "_MS", Stale()):
            rows, hidden = mp._build_candidates(self.models, {})
        self.assertEqual(len(rows), len(self.rows))
        self.assertEqual(hidden["bench_excluded"], 0)

    def test_absent_module_does_not_crash(self):
        with mock.patch.object(mp, "_MS", None):
            rows, hidden = mp._build_candidates(self.models, {})
        self.assertEqual(len(rows), len(self.rows))
        self.assertEqual(hidden["bench_excluded"], 0)


class ArtefactShaUnitTest(unittest.TestCase):
    """_artefact_sha in isolation -- no models on disk required."""

    def test_prefers_hf_sha(self):
        self.assertEqual(
            mp._artefact_sha({"probe": {"sha": "abc123", "digest": "zzz"}}),
            "abc123")

    def test_falls_back_to_ollama_digest(self):
        self.assertEqual(
            mp._artefact_sha({"probe": {"digest": "def456"}}), "def456")

    def test_none_when_probe_carries_neither(self):
        self.assertIsNone(mp._artefact_sha({"probe": {"repo": "x/y"}}))

    def test_survives_a_missing_or_malformed_probe(self):
        for row in ({}, {"probe": None}, {"probe": "not-a-dict"},
                    {"probe": []}):
            with self.subTest(row=row):
                self.assertIsNone(mp._artefact_sha(row))

    def test_empty_values_are_not_treated_as_an_identity(self):
        # An empty string would satisfy `if sha` on the ledger side and
        # skew the comparison; treat it as absent.
        self.assertIsNone(mp._artefact_sha({"probe": {"sha": ""}}))
        self.assertEqual(
            mp._artefact_sha({"probe": {"sha": "", "digest": "d1"}}), "d1")

    def test_coerces_to_str(self):
        self.assertEqual(mp._artefact_sha({"probe": {"digest": 12345}}),
                         "12345")


if __name__ == "__main__":
    unittest.main()
