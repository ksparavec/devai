"""scripts/stats/statlib.py must reproduce published reference values.

Every interval, test and power figure in the experiment docs is computed by
this module (docs/statistics-primer.md), so it is checked here against values
printed in the method papers, against closed forms, and against the defining
equations evaluated in exact rational arithmetic. Two independent
implementations written for the 2026-09-27 re-analysis agreed to 2e-16 on the
Newcombe intervals; these tests pin the merged version.

Stdlib unittest only.
"""

from __future__ import annotations

import math
import sys
import unittest
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "stats"))

import statlib as S  # noqa: E402


class OneProportionTest(unittest.TestCase):
    def test_clopper_pearson_and_wilson_match_newcombe_1998_table_i(self):
        # Newcombe (1998), Stat Med 17:857, Table I (4 decimals).
        cases = [((81, 263), (0.2527, 0.3676), (0.2553, 0.3662)),
                 ((15, 148), (0.0578, 0.1617), (0.0624, 0.1605)),
                 ((0, 20), (0.0, 0.1684), (0.0, 0.1611)),
                 ((1, 29), (0.0009, 0.1776), (0.0061, 0.1718))]
        for (x, n), cp, wi in cases:
            lo, hi = S.clopper_pearson(x, n)
            self.assertAlmostEqual(lo, cp[0], delta=6e-5)
            self.assertAlmostEqual(hi, cp[1], delta=6e-5)
            lo, hi = S.wilson(x, n)
            self.assertAlmostEqual(lo, wi[0], delta=6e-5)
            self.assertAlmostEqual(hi, wi[1], delta=6e-5)

    def test_clopper_pearson_closed_forms(self):
        self.assertAlmostEqual(S.clopper_pearson(0, 10)[1], 1 - 0.025 ** (1 / 10), places=9)
        self.assertAlmostEqual(S.clopper_pearson(10, 10)[0], 0.025 ** (1 / 10), places=9)

    def test_clopper_pearson_solves_its_defining_equations_exactly(self):
        def sf_exact(k, n, p):
            p = Fraction(p)
            return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))
        for x, n in [(15, 148), (52, 60), (45, 60), (47, 60)]:
            lo, hi = S.clopper_pearson(x, n)
            self.assertAlmostEqual(sf_exact(x, n, lo), 0.025, delta=1e-9)
            self.assertAlmostEqual(1 - sf_exact(x + 1, n, hi), 0.025, delta=1e-9)

    def test_clopper_pearson_coverage_is_at_least_nominal(self):
        n = 60
        cis = [S.clopper_pearson(x, n) for x in range(n + 1)]
        for p in (0.05, 0.2, 0.5, 0.75, 0.85, 0.95):
            cov = sum(S.binom_pmf(x, n, p) for x in range(n + 1) if cis[x][0] <= p <= cis[x][1])
            self.assertGreaterEqual(cov, 0.95 - 1e-9)

    def test_one_sided_lower_bound(self):
        # n/n successes: bound = alpha^(1/n); 59/59 is the first to clear 0.95.
        self.assertAlmostEqual(S.clopper_pearson_lower_one_sided(59, 59), 0.05 ** (1 / 59), places=9)
        self.assertGreater(S.clopper_pearson_lower_one_sided(59, 59), 0.95)
        self.assertLess(S.clopper_pearson_lower_one_sided(58, 58), 0.95)
        # equals the lower end of the two-sided 90% interval
        self.assertAlmostEqual(S.clopper_pearson_lower_one_sided(81, 84),
                               S.clopper_pearson(81, 84, alpha=0.10)[0], places=12)

    def test_hypergeometric_interval_nests_inside_clopper_pearson(self):
        for x in (45, 52):
            h = S.hypergeometric_interval(x, 60, 198)
            c = S.clopper_pearson(x, 60)
            self.assertLessEqual(c[0], h[0] + 1e-9)
            self.assertLessEqual(h[1], c[1] + 1e-9)
            self.assertTrue(h[0] <= x / 60 <= h[1])

    def test_binomial_tests(self):
        self.assertAlmostEqual(S.binom_test_two_sided(7, 14, 0.5), 1.0, places=12)
        self.assertAlmostEqual(S.binom_test_greater(6, 14, 4 / 14), 0.1848, delta=5e-5)


class TwoProportionTest(unittest.TestCase):
    def test_mcnemar_exact(self):
        self.assertAlmostEqual(S.mcnemar_exact(0, 5), 0.0625, places=12)
        self.assertAlmostEqual(S.mcnemar_exact(1, 9), 2 * 11 / 1024, places=12)
        self.assertAlmostEqual(S.mcnemar_exact(8, 2), 0.109375, places=12)
        self.assertEqual(S.mcnemar_exact(3, 3), 1.0)
        self.assertEqual(S.mcnemar_exact(0, 0), 1.0)

    def test_newcombe_unpaired_matches_newcombe_1998a(self):
        # Newcombe (1998a), Stat Med 17:873, method 10 worked examples.
        for (x1, n1, x2, n2), ref in [((56, 70, 48, 80), (0.0524, 0.3339)),
                                      ((9, 10, 3, 10), (0.1705, 0.8090))]:
            _, lo, hi = S.newcombe_unpaired(x1, n1, x2, n2)
            self.assertAlmostEqual(lo, ref[0], delta=1e-4)
            self.assertAlmostEqual(hi, ref[1], delta=1e-4)

    def test_newcombe_paired_reduces_to_unpaired_when_phi_is_zero(self):
        _, lo, hi = S.newcombe_paired(12, 12, 8, 8)  # a*d == b*c
        _, lo2, hi2 = S.newcombe_unpaired(24, 40, 20, 40)
        self.assertAlmostEqual(lo, lo2, places=12)
        self.assertAlmostEqual(hi, hi2, places=12)

    def test_newcombe_paired_is_antisymmetric_in_b_and_c(self):
        _, lo, hi = S.newcombe_paired(10, 7, 2, 11)
        _, lo2, hi2 = S.newcombe_paired(10, 2, 7, 11)
        self.assertAlmostEqual(lo, -hi2, places=12)
        self.assertAlmostEqual(hi, -lo2, places=12)

    def test_newcombe_paired_exact_coverage_is_near_nominal(self):
        n = 30
        for probs in [(0.70, 0.08, 0.08, 0.14), (0.75, 0.03, 0.12, 0.10)]:
            truth = probs[1] - probs[2]
            cov = 0.0
            for a in range(n + 1):
                for b in range(n + 1 - a):
                    for c in range(n + 1 - a - b):
                        d = n - a - b - c
                        lp = (math.lgamma(n + 1) - math.lgamma(a + 1) - math.lgamma(b + 1)
                              - math.lgamma(c + 1) - math.lgamma(d + 1))
                        lp += sum(k * math.log(pr) for k, pr in zip((a, b, c, d), probs))
                        _, lo, hi = S.newcombe_paired(a, b, c, d)
                        if lo - 1e-12 <= truth <= hi + 1e-12:
                            cov += math.exp(lp)
            self.assertTrue(0.90 <= cov <= 0.995, (probs, cov))

    def test_holm(self):
        adj = S.holm({"a": 0.004, "b": 0.020, "c": 0.030, "d": 0.40})
        self.assertAlmostEqual(adj["a"], 0.016, places=12)
        self.assertAlmostEqual(adj["b"], 0.060, places=12)
        self.assertAlmostEqual(adj["c"], 0.060, places=12)
        self.assertAlmostEqual(adj["d"], 0.40, places=12)

    def test_sign_flip(self):
        self.assertAlmostEqual(S.sign_flip_exact([2, 2, 2, 2, 0]), 2 / 16, places=12)
        self.assertEqual(S.sign_flip_exact([1, -1]), 1.0)


class PowerTest(unittest.TestCase):
    def test_exact_mcnemar_size_is_at_most_alpha(self):
        self.assertLessEqual(S.mcnemar_power(60, 0.075, 0.075), S.ALPHA + 1e-12)

    def test_two_proportion_power(self):
        self.assertAlmostEqual(S.power_two_proportions(0.8, 0.7, 60), 0.242, delta=5e-4)
        self.assertEqual(S.n_two_proportions(0.8, 0.7), 294)


class QuantileTest(unittest.TestCase):
    def test_median_interval_ranks(self):
        self.assertEqual(S.median_ci_ranks(24)[:2], (7, 18))
        self.assertAlmostEqual(S.median_ci_ranks(24)[2], 0.9773, delta=5e-5)
        self.assertEqual(S.median_ci_ranks(100)[:2], (40, 61))

    def test_quantile_definitions_differ_at_small_n(self):
        xs = [float(i) for i in range(1, 25)]
        self.assertEqual(S.quantile_nearest_rank(xs, 0.9), 22.0)
        self.assertAlmostEqual(S.quantile_type7(xs, 0.9), 21.7, places=12)
        self.assertEqual(S.quantile_floor_index(xs, 0.9), 21.0)


class AurocTest(unittest.TestCase):
    def test_auroc_basics(self):
        self.assertEqual(S.auroc([1, 2], [0]), 1.0)
        self.assertEqual(S.auroc([0], [1]), 0.0)
        self.assertEqual(S.auroc([1], [1]), 0.5)
        a, se, _, _ = S.delong([0.9, 0.8, 0.7], [0.1, 0.2])
        self.assertEqual((a, se), (1.0, 0.0))

    def test_hanley_mcneil_closed_form_at_half(self):
        se, _, _ = S.hanley_mcneil(0.5, 10, 10)
        self.assertAlmostEqual(se, math.sqrt((0.25 + 9 / 12 + 9 / 12) / 100), places=12)

    def test_bootstrap_is_deterministic(self):
        units = [[(0.9, True)], [(0.2, False)], [(0.7, True)], [(0.4, False)], [(0.6, False)]]
        self.assertEqual(S.bootstrap_auroc(units, 500, "x"), S.bootstrap_auroc(units, 500, "x"))



class SampleSizeTest(unittest.TestCase):
    def test_connor_paired_sample_sizes(self):
        # (1.96 sqrt(.10) + 0.8416 sqrt(.10 - .01))^2 / .01 = 76.09 -> 77 pairs
        self.assertEqual(S.n_paired_superiority(0.10, 0.10), 77)
        # (1.96 + 0.8416)^2 * .10 / .02^2 = 1962.3 -> 1963 item pairs
        self.assertEqual(S.n_paired_noninferiority(0.10, 0.02), 1963)

    def test_mcnemar_mde_is_the_smallest_delta_reaching_the_target(self):
        n, psi = 60, 0.22
        mde = S.mcnemar_mde(n, psi)
        self.assertIsNotNone(mde)
        self.assertGreaterEqual(S.mcnemar_power(n, (psi + mde) / 2, (psi - mde) / 2), 0.80)
        below = mde - 0.0025
        self.assertLess(S.mcnemar_power(n, (psi + below) / 2, (psi - below) / 2), 0.80)

    def test_wald_halfwidths(self):
        self.assertAlmostEqual(S.mde_paired_halfwidth(50, 0.10), S.Z975 * (0.10 / 50) ** 0.5, places=12)
        self.assertAlmostEqual(S.mde_unpaired_halfwidth(50, 0.8), S.Z975 * (2 * 0.8 * 0.2 / 50) ** 0.5, places=12)


class DescriptiveTest(unittest.TestCase):
    def test_quantile_edges(self):
        self.assertIsNone(S.quantile([], 0.5))
        self.assertEqual(S.quantile([3.0], 0.9), 3.0)
        self.assertEqual(S.quantile_type7([3.0], 0.9), 3.0)
        self.assertEqual(S.quantile([4.0, 1.0, 3.0, 2.0], 0.5), 2.5)  # unsorted input

    def test_median_interval_needs_six_observations(self):
        with self.assertRaises(ValueError):
            S.median_ci_ranks(5)
        self.assertIsNone(S.median_ci([1.0, 2.0, 3.0, 4.0, 5.0]))
        lo, hi, cov, ranks = S.median_ci([float(i) for i in range(1, 25)])
        self.assertEqual((lo, hi, ranks), (7.0, 18.0, (7, 18)))

    def test_quantile_interval_minimum_n(self):
        # [x(1), x(n)] covers the 0.9-quantile with prob 1 - 0.9^n - 0.1^n: needs n >= 29
        self.assertFalse(S.quantile_ci([float(i) for i in range(28)], 0.9)["available"])
        self.assertTrue(S.quantile_ci([float(i) for i in range(29)], 0.9)["available"])

    def test_describe_handles_empty_and_missing_values(self):
        self.assertEqual(S.describe([]), {"n": 0})
        self.assertEqual(S.describe([None]), {"n": 0, "n_none_dropped": 1})
        d = S.describe([1.0, None, 3.0, 2.0])
        self.assertEqual((d["n"], d["n_none_dropped"], d["min"], d["max"], d["median"]), (3, 1, 1.0, 3.0, 2.0))
        self.assertNotIn("n_none_dropped", S.describe([1.0, 2.0]))

    def test_rnd_is_recursive(self):
        self.assertEqual(S.rnd({"a": [1.23456, {"b": 2.34567}], "c": "x"}, 2), {"a": [1.23, {"b": 2.35}], "c": "x"})


class ResamplingTest(unittest.TestCase):
    def test_bootstrap_of_a_constant_is_the_constant(self):
        self.assertEqual(S.boot_ci([2.0] * 10, lambda v: sum(v) / len(v), reps=200), (2.0, 2.0))
        lo, hi = S.boot_ci_two_sample([3.0] * 5, [1.0] * 7, lambda x, y: sum(x) / len(x) - sum(y) / len(y), reps=200)
        self.assertEqual((lo, hi), (2.0, 2.0))

    def test_bootstrap_is_deterministic_and_brackets_the_estimate(self):
        data = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        mean = lambda v: sum(v) / len(v)  # noqa: E731
        first = S.boot_ci(data, mean, reps=2000, seed=7)
        self.assertEqual(first, S.boot_ci(data, mean, reps=2000, seed=7))
        self.assertTrue(first[0] <= 0.7 <= first[1])

    def test_stratified_bootstrap_estimate_is_the_mean_of_stratum_means(self):
        est, lo, hi = S.stratified_bootstrap_mean_of_means([[1.0, 0.0], [1.0, 1.0, 1.0, 1.0]], reps=500)
        self.assertAlmostEqual(est, 0.75, places=12)
        self.assertTrue(lo <= est <= hi)
        self.assertEqual(S.stratified_bootstrap_mean_of_means([[1.0] * 3, [0.0] * 4], reps=100)[1:], (0.5, 0.5))

    def test_paired_signflip(self):
        _, p = S.paired_signflip_test([[0.0] * 10], reps=200)
        self.assertEqual(p, 1.0)
        _, p = S.paired_signflip_test([[1.0] * 30], reps=2000)
        self.assertLess(p, 0.01)

    def test_paired_bootstrap_requires_pairing(self):
        with self.assertRaises(ValueError):
            S.paired_bootstrap_diff([[1.0, 0.0]], [[1.0]])
        self.assertEqual(S.paired_bootstrap_diff([[1.0, 0.0, 1.0]], [[1.0, 0.0, 1.0]], reps=200), (0.0, 0.0, 0.0))

    def test_bootstrap_auroc_refuses_single_class_units(self):
        with self.assertRaises(ValueError):
            S.bootstrap_auroc([[(0.9, True)], [(0.8, True)]], reps=50)


class PoissonTest(unittest.TestCase):
    def test_garwood_matches_published_values(self):
        # Garwood exact 95% limits: k=0 -> [0, 3.689]; k=3 -> [0.619, 8.767]; k=10 -> [4.795, 18.39]
        for k, (lo, hi) in [(0, (0.0, 3.689)), (3, (0.619, 8.767)), (10, (4.795, 18.39))]:
            got = S.poisson_exact_ci(k)
            self.assertAlmostEqual(got[0], lo, delta=1e-3)
            self.assertAlmostEqual(got[1], hi, delta=1e-2)

    def test_garwood_bracket_widens_for_small_alpha(self):
        lo, hi = S.poisson_exact_ci(0, alpha=1e-12)
        self.assertAlmostEqual(hi, -math.log(0.5e-12), delta=1e-3)  # k = 0: upper = -ln(alpha/2)


if __name__ == "__main__":
    unittest.main()
