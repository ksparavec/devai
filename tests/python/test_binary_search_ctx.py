"""Unit tests for binary_search_max_ctx (max-serving-ctx finder).

Verifies the exact "test-the-top-then-bisect" decision tree, the <=4-probe
budget, the position_limit instant-fail (never launch above a model's
as-delivered ceiling), and the full result mapping. Stdlib unittest only;
pure logic, no disk or network.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _contexts import (  # noqa: E402
    BINARY_SEARCH_CONTEXTS,
    binary_search_max_ctx,
)

K = 1024


def _threshold(max_ok_ctx: int):
    """Monotonic predicate: works for any ctx <= max_ok_ctx. Records probes."""
    probes: list[int] = []

    def works(ctx: int) -> bool:
        probes.append(ctx)
        return ctx <= max_ok_ctx

    return works, probes


class TestResultMapping(unittest.TestCase):
    def test_every_tier_is_found_exactly(self) -> None:
        for true_max in BINARY_SEARCH_CONTEXTS:
            works, _ = _threshold(true_max)
            self.assertEqual(
                binary_search_max_ctx(works), true_max, f"true_max={true_max}"
            )

    def test_between_tiers_rounds_down(self) -> None:
        # A real ceiling of 200K sits between 192K and 224K -> pick 192K.
        works, _ = _threshold(200 * K)
        self.assertEqual(binary_search_max_ctx(works), 192 * K)

    def test_top_works_is_a_single_probe(self) -> None:
        works, probes = _threshold(256 * K)
        self.assertEqual(binary_search_max_ctx(works), 256 * K)
        self.assertEqual(probes, [256 * K])  # fast path

    def test_nothing_fits_returns_none(self) -> None:
        works, _ = _threshold(0)
        self.assertIsNone(binary_search_max_ctx(works))

    def test_probe_budget_at_most_four(self) -> None:
        for true_max in list(BINARY_SEARCH_CONTEXTS) + [0, 200 * K]:
            works, probes = _threshold(true_max)
            binary_search_max_ctx(works)
            self.assertLessEqual(len(probes), 4, f"true_max={true_max} {probes}")


class TestExactDecisionTree(unittest.TestCase):
    """The precise probe sequences from the specification."""

    def _probe_seq_for(self, true_max: int) -> tuple[int | None, list[int]]:
        works, probes = _threshold(true_max)
        result = binary_search_max_ctx(works)
        return result, [p // K for p in probes]  # probes as K for readability

    def test_224_path(self) -> None:  # 256 fail -> 128 -> 192 -> 224
        result, seq = self._probe_seq_for(224 * K)
        self.assertEqual(result, 224 * K)
        self.assertEqual(seq, [256, 128, 192, 224])

    def test_192_path(self) -> None:  # 224 fails -> keep 192
        result, seq = self._probe_seq_for(192 * K)
        self.assertEqual(result, 192 * K)
        self.assertEqual(seq, [256, 128, 192, 224])

    def test_160_path(self) -> None:  # 192 fail -> 160
        result, seq = self._probe_seq_for(160 * K)
        self.assertEqual(result, 160 * K)
        self.assertEqual(seq, [256, 128, 192, 160])

    def test_128_path(self) -> None:  # 160 fail -> keep 128
        result, seq = self._probe_seq_for(128 * K)
        self.assertEqual(result, 128 * K)
        self.assertEqual(seq, [256, 128, 192, 160])

    def test_96_path(self) -> None:  # 128 fail -> 64 -> 96
        result, seq = self._probe_seq_for(96 * K)
        self.assertEqual(result, 96 * K)
        self.assertEqual(seq, [256, 128, 64, 96])

    def test_64_path(self) -> None:  # 96 fail -> keep 64
        result, seq = self._probe_seq_for(64 * K)
        self.assertEqual(result, 64 * K)
        self.assertEqual(seq, [256, 128, 64, 96])

    def test_32_path(self) -> None:  # 64 fail -> 32
        result, seq = self._probe_seq_for(32 * K)
        self.assertEqual(result, 32 * K)
        self.assertEqual(seq, [256, 128, 64, 32])

    def test_none_path(self) -> None:  # 32 fail -> exclude
        result, seq = self._probe_seq_for(0)
        self.assertIsNone(result)
        self.assertEqual(seq, [256, 128, 64, 32])


class TestPositionLimit(unittest.TestCase):
    def test_ceiling_capped_and_never_launched_above_limit(self) -> None:
        # Model can serve anything it can launch, but its trained ceiling is
        # 128K. Tiers above 128K must be instant-failed WITHOUT calling works.
        launched: list[int] = []

        def works(ctx: int) -> bool:
            launched.append(ctx)
            return True  # anything that actually launches serves

        result = binary_search_max_ctx(works, position_limit=128 * K)
        self.assertEqual(result, 128 * K)
        self.assertTrue(all(c <= 128 * K for c in launched), launched)
        self.assertNotIn(256 * K, launched)
        self.assertNotIn(192 * K, launched)

    def test_40k_model_lands_on_32k(self) -> None:
        # A 40K-ceiling model (Qwen3-8B-class): only 32K is <= limit.
        launched: list[int] = []

        def works(ctx: int) -> bool:
            launched.append(ctx)
            return True

        result = binary_search_max_ctx(works, position_limit=40960)
        self.assertEqual(result, 32 * K)
        self.assertEqual(launched, [32 * K])  # only the one loadable tier ran


if __name__ == "__main__":
    unittest.main()
