"""Every worked-example number in docs/statistics-primer.md, recomputed.

    python3 scripts/stats/primer_examples.py

prints each example with the string the primer shows. The test
tests/python/test_stats_primer_examples.py asserts that every string appears in
the primer, so a changed number cannot drift silently. Standard library only.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import statlib as S  # noqa: E402


def _iv(lo: float, hi: float) -> str:
    return f"[{lo:.3f}, {hi:.3f}]"


def examples() -> list[tuple[str, str]]:
    """(description, text as printed in the primer)."""
    out: list[tuple[str, str]] = []
    out.append(("47/60 Clopper-Pearson", f"Clopper-Pearson  {_iv(*S.clopper_pearson(47, 60))}"))
    out.append(("47/60 Wilson", f"Wilson           {_iv(*S.wilson(47, 60))}"))
    for n in (20, 50, 60, 100, 198, 1000):
        x = round(0.8 * n)
        lo, hi = S.clopper_pearson(x, n)
        out.append((f"width table n={n}", f"| {n} | {x} | {_iv(lo, hi)} | {hi - lo:.2f} |"))
    out.append(("60/60", f"60 out of 60 gives {_iv(*S.clopper_pearson(60, 60))}"))
    out.append(("0/40", f"0 out of 40 gives {_iv(*S.clopper_pearson(0, 40))}"))
    fpc = ((198 - 60) / (198 - 1)) ** 0.5
    out.append(("FPC factor", f"sqrt((198 - 60) / (198 - 1)) = {fpc:.3f}"))
    out.append(("one-sided 81/84", f"one-sided 95% lower bound\nof {S.clopper_pearson_lower_one_sided(81, 84):.3f}"))
    out.append(("two-sided 81/84", f"two-sided 95% interval of {_iv(*S.clopper_pearson(81, 84))}"))
    out.append(("McNemar 8/2", f"m = 10,  p = 2 x P(X <= 2 | Binomial(10, 0.5)) = {S.mcnemar_exact(8, 2):.3f}"))
    out.append(("McNemar 7/0", f"p = {S.mcnemar_exact(7, 0):.3f}"))
    out.append(("power n=60", f"n = 60 per arm      power ~ {S.power_two_proportions(0.8, 0.7, 60):.2f}"))
    out.append(("power n=300", f"n = 300 per arm     power ~ {S.power_two_proportions(0.8, 0.7, 300):.2f}"))
    out.append(("n for 80%", f"n = {S.n_two_proportions(0.8, 0.7)} per arm     is needed for 80% power"))
    adj = S.holm({"a": 0.004, "b": 0.020, "c": 0.030, "d": 0.40})
    out.append(("Holm row 1", f"| 0.004 | {adj['a']:.3f} | yes |"))
    out.append(("Holm row 2", f"| 0.020 | {adj['b']:.3f} | no |"))
    for n in (24, 60, 100):
        r, s, cov = S.median_ci_ranks(n)
        out.append((f"median CI n={n}", f"| {n} | [x({r}), x({s})] | {100 * cov:.1f}% |"))
    for npos, label in ((20, "20 / 20"), (50, "50 / 50"), (200, "200 / 200")):
        _, lo, hi = S.hanley_mcneil(0.64, npos, npos)
        out.append((f"AUROC {label}", f"| {label} | [{lo:.2f}, {hi:.2f}]"))
    out.append(("6/14 vs 4/14", f"P(X >= 6 | n = 14, p = 0.286) is {S.binom_test_greater(6, 14, 4 / 14):.3f}"))
    out.append(("6/14 CI", f"Clopper-Pearson interval {_iv(*S.clopper_pearson(6, 14))}"))
    out.append(("27/30", f"Clopper-Pearson interval {_iv(*S.clopper_pearson(27, 30))} (27 of 30)"))
    out.append(("270/300", f"{_iv(*S.clopper_pearson(270, 300))} (270 of 300)"))
    lo, hi = S.poisson_exact_ci(3)
    out.append(("Garwood 3 matches", f"for 3 matches, {_iv(lo, hi)} matches, or\n{_iv(lo / 40, hi / 40)} per prompt"))
    out.append(("time-out bounds", f"between\n{74 / 100:.2f} and {98 / 100:.2f}"))
    out.append(("among completed", f"(74/76 = {74 / 76:.3f}, 95%\ninterval {_iv(*S.clopper_pearson(74, 76))})"))
    n90 = math.ceil(math.log(0.05) / math.log(0.9))
    n95 = math.ceil(math.log(0.05) / math.log(0.95))
    out.append(("quantile minimum n", f"n >= {n90} for a p90 and n >= {n95} for a p95"))
    lo, hi = S.clopper_pearson(17, 20)
    out.append(("17/20", f"17/20 has the 95% interval\n[{lo:.2f}, {hi:.2f}]"))
    lo, hi = S.clopper_pearson(20, 20)
    out.append(("20/20", f"20/20 has [{lo:.2f}, {hi:.2f}]"))
    out.append(("S decomposition", f"gives about {3.74 / 1.19:.2f}"))
    lo = S.clopper_pearson(74, 100)[0]
    hi = S.clopper_pearson(98, 100)[1]
    out.append(("time-out interval", f"For the\nexample that gives {_iv(lo, hi)}"))
    top = [_softmax_top(z, T) for T in (1, 5) for z in ((2, 0, 0, 0), (3, 2.5, -10, -10))]
    out.append(("temperature reorder", f"top-option confidences of {top[0]:.3f} and {top[1]:.3f} at T = 1, but {top[2]:.3f} and {top[3]:.3f}\nat T = 5"))
    _, plo, phi = S.newcombe_paired(42, 4, 10, 4)
    _, ulo, uhi = S.newcombe_unpaired(46, 60, 52, 60)
    out.append(("pairing gain", f"[{plo:.3f},\n+{phi:.3f}] against [{ulo:.3f}, +{uhi:.3f}] unpaired"))
    return out


def _softmax_top(z: tuple[float, ...], temperature: float) -> float:
    e = [math.exp(v / temperature) for v in z]
    return max(e) / sum(e)


if __name__ == "__main__":
    for desc, text in examples():
        print(f"{desc:22s} {text!r}")
