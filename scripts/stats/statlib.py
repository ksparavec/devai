"""Statistics primitives for the experiment re-analyses (docs/statistics-primer.md).

Standard library only, deterministic, exact or closed-form wherever possible, so
every number the docs report can be reproduced without scipy. Each function names
the method and its reference; tests/python/test_stats_statlib.py checks them
against published values.

Conventions:
- two-sided 95% unless stated; ALPHA = 0.05;
- proportions: Clopper-Pearson (exact) primary, Wilson secondary;
- paired comparisons: exact McNemar + Newcombe (1998b) method 10;
- unpaired: Newcombe (1998a) method 10;
- families: Holm step-down.
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist

ALPHA = 0.05
Z975 = NormalDist().inv_cdf(0.975)  # 1.959963984540054
Z80 = NormalDist().inv_cdf(0.80)    # 0.8416212335729143
_BISECT_ITERS = 200


# ---------------------------------------------------------------- binomial

def _log_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def binom_pmf(k: int, n: int, p: float) -> float:
    return math.exp(_log_pmf(k, n, p))


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k), X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return min(1.0, math.fsum(binom_pmf(i, n, p) for i in range(k + 1)))


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k), X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return min(1.0, math.fsum(binom_pmf(i, n, p) for i in range(k, n + 1)))


def _bisect_increasing(f, target: float) -> float:
    """p in [0, 1] with f(p) = target, for f increasing in p."""
    lo, hi = 0.0, 1.0
    for _ in range(_BISECT_ITERS):
        mid = (lo + hi) / 2.0
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ---------------------------------------------------------------- one proportion

def clopper_pearson(x: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Two-sided (1 - alpha) Clopper-Pearson (1934) exact interval.

    Inverts two one-sided exact binomial tests: the lower bound solves
    P(X >= x | p) = alpha/2, the upper bound P(X <= x | p) = alpha/2.
    Coverage >= 1 - alpha for every p.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    a2 = alpha / 2.0
    lower = 0.0 if x == 0 else _bisect_increasing(lambda p: binom_sf(x, n, p), a2)
    upper = 1.0 if x == n else _bisect_increasing(lambda p: 1.0 - binom_cdf(x, n, p), 1.0 - a2)
    return lower, upper


def clopper_pearson_lower_one_sided(x: int, n: int, alpha: float = ALPHA) -> float:
    """One-sided (1 - alpha) lower bound = lower end of the two-sided (1 - 2 alpha) interval."""
    if x == 0:
        return 0.0
    return clopper_pearson(x, n, 2.0 * alpha)[0]


def wilson(x: int, n: int, z: float = Z975) -> tuple[float, float]:
    """Wilson (1927) score interval."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = x / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def _hyper_pmf(k: int, N: int, K: int, n: int) -> float:
    if k < max(0, n - (N - K)) or k > min(n, K):
        return 0.0
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)


def hypergeometric_interval(x: int, n: int, N: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Exact interval for the population proportion K/N, n of N drawn without replacement.

    Valid ONLY when each item's outcome is fixed (deterministic decoding): the
    finite-population correction does not remove decoding randomness.
    Cost is O(N * n) per call (about 1 s at N = 2000, n = 100).
    """
    a2 = alpha / 2.0
    ks = range(N + 1)
    lower = min(K for K in ks if math.fsum(_hyper_pmf(k, N, K, n) for k in range(x, n + 1)) > a2)
    upper = max(K for K in ks if math.fsum(_hyper_pmf(k, N, K, n) for k in range(x + 1)) > a2)
    return lower / N, upper / N


def binom_test_greater(x: int, n: int, p0: float) -> float:
    """Exact one-sided p-value for H1: p > p0."""
    return binom_sf(x, n, p0)


def binom_test_two_sided(x: int, n: int, p0: float) -> float:
    """Exact two-sided p-value, 'minlike' method (as R's binom.test)."""
    d = binom_pmf(x, n, p0) * (1 + 1e-7)
    return min(1.0, math.fsum(q for q in (binom_pmf(i, n, p0) for i in range(n + 1)) if q <= d))


# ---------------------------------------------------------------- two proportions

def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar (1947) test: 2 P(X <= min(b, c)), X ~ Bin(b + c, 1/2), capped at 1."""
    m = b + c
    if m == 0:
        return 1.0
    return min(1.0, 2.0 * binom_cdf(min(b, c), m, 0.5))


def newcombe_unpaired(x1: int, n1: int, x2: int, n2: int, z: float = Z975) -> tuple[float, float, float]:
    """Newcombe (1998a) method 10 hybrid score interval for p1 - p2, independent samples.

    Returns (difference, lower, upper).
    """
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1, z)
    l2, u2 = wilson(x2, n2, z)
    d = p1 - p2
    return (d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def newcombe_paired(a: int, b: int, c: int, d: int, z: float = Z975) -> tuple[float, float, float]:
    """Newcombe (1998b) method 10 interval for p1 - p2 on paired data, no continuity correction.

    Table: a = both succeed, b = only configuration 1 succeeds, c = only
    configuration 2 succeeds, d = both fail. p1 = (a+b)/n, p2 = (a+c)/n,
    phi = (ad - bc)/sqrt(product of margins), 0 when a margin is 0.
    Returns (difference, lower, upper).
    """
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(den) if den > 0 else 0.0
    dl = (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2
    du = (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2
    diff = p1 - p2
    return diff, max(-1.0, diff - math.sqrt(max(0.0, dl))), min(1.0, diff + math.sqrt(max(0.0, du)))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm (1979) step-down adjusted p-values (monotone)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[key] = running
    return out


def sign_flip_exact(diffs: list[int]) -> float:
    """Exact two-sided paired sign-flip (permutation) test on integer differences.

    H0: each non-zero difference is symmetric about 0.
    """
    nonzero = [abs(v) for v in diffs if v != 0]
    observed = abs(sum(diffs))
    dist = {0: 1}
    for v in nonzero:
        nxt: dict[int, int] = {}
        for s, w in dist.items():
            nxt[s + v] = nxt.get(s + v, 0) + w
            nxt[s - v] = nxt.get(s - v, 0) + w
        dist = nxt
    return sum(w for s, w in dist.items() if abs(s) >= observed) / 2 ** len(nonzero)


# ---------------------------------------------------------------- power, sample size

def mcnemar_power(n: int, pb: float, pc: float, alpha: float = ALPHA) -> float:
    """Exact power of the two-sided exact McNemar test, by enumeration.

    pb, pc: probabilities of the two discordant cells.
    """
    psi = pb + pc
    if psi <= 0:
        return 0.0
    r = pb / psi
    power = 0.0
    for m in range(n + 1):
        pm = binom_pmf(m, n, psi)
        if pm < 1e-15:
            continue
        power += pm * math.fsum(binom_pmf(bb, m, r) for bb in range(m + 1)
                                if mcnemar_exact(bb, m - bb) <= alpha)
    return power


def mcnemar_mde(n: int, psi: float, target: float = 0.80, step: float = 0.0025) -> float | None:
    """Smallest |pb - pc| with exact McNemar power >= target at discordance psi."""
    delta = step
    while delta < psi:
        if mcnemar_power(n, (psi + delta) / 2, (psi - delta) / 2) >= target:
            return round(delta, 4)
        delta += step
    return None


def n_paired_superiority(psi: float, delta: float) -> int:
    """Pairs for a paired difference delta, two-sided 5%, power 80% (Connor 1987, normal approx.)."""
    return math.ceil((Z975 * math.sqrt(psi) + Z80 * math.sqrt(max(psi - delta * delta, 1e-12))) ** 2
                     / delta ** 2)


def n_paired_noninferiority(psi: float, margin: float) -> int:
    """Pairs to show a loss below margin (true difference 0, one-sided 2.5%, power 80%)."""
    return math.ceil((Z975 + Z80) ** 2 * psi / margin ** 2)


def power_two_proportions(p1: float, p2: float, n: int, alpha: float = ALPHA) -> float:
    """Power of the two-sided unpaired z-test for p1 vs p2, n per arm (normal approx.)."""
    z = NormalDist().inv_cdf(1 - alpha / 2)
    pbar = (p1 + p2) / 2
    se0 = math.sqrt(2 * pbar * (1 - pbar) / n)
    se1 = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    return NormalDist().cdf((abs(p1 - p2) - z * se0) / se1)


def n_two_proportions(p1: float, p2: float, target: float = 0.80, alpha: float = ALPHA) -> int:
    """Smallest n per arm with power_two_proportions >= target."""
    n = 2
    while power_two_proportions(p1, p2, n, alpha) < target:
        n += 1
    return n


# ---------------------------------------------------------------- quantiles

def median_ci_ranks(n: int, alpha: float = ALPHA) -> tuple[int, int, float]:
    """Distribution-free interval for the median: 1-based ranks (r, n - r + 1) and exact coverage.

    r is the largest rank with P(Bin(n, 1/2) <= r - 1) <= alpha/2.
    """
    candidates = [k for k in range(1, n // 2 + 1) if binom_cdf(k - 1, n, 0.5) <= alpha / 2]
    if not candidates:
        raise ValueError(f"n={n} is too small for a {1 - alpha:.0%} median interval")
    r = max(candidates)
    return r, n - r + 1, 1.0 - 2.0 * binom_cdf(r - 1, n, 0.5)


def quantile_nearest_rank(sorted_xs: list[float], q: float) -> float:
    """Nearest-rank definition: x(ceil(q n)), 1-based."""
    n = len(sorted_xs)
    return sorted_xs[max(1, math.ceil(q * n)) - 1]


def quantile_type7(sorted_xs: list[float], q: float) -> float:
    """Hyndman-Fan type 7 (linear interpolation; NumPy / R default)."""
    n = len(sorted_xs)
    h = (n - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, n - 1)
    return sorted_xs[lo] + (h - lo) * (sorted_xs[hi] - sorted_xs[lo])


def quantile_floor_index(sorted_xs: list[float], q: float) -> float:
    """The definition sorted(x)[int(q (n - 1))] (0-based), used by some ad-hoc scripts."""
    return sorted_xs[int(q * (len(sorted_xs) - 1))]


# ---------------------------------------------------------------- AUROC

def auroc(pos: list[float], neg: list[float]) -> float:
    """P(score of a random positive > score of a random negative), ties count 1/2."""
    s = 0.0
    for p in pos:
        for q in neg:
            s += 1.0 if p > q else (0.5 if p == q else 0.0)
    return s / (len(pos) * len(neg))


def hanley_mcneil(a: float, n_pos: int, n_neg: int, z: float = Z975) -> tuple[float, float, float]:
    """Hanley-McNeil (1982) standard error and Wald interval for an AUROC.

    Returns (se, lower, upper).
    """
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    se = math.sqrt(max(0.0, var))
    return se, max(0.0, a - z * se), min(1.0, a + z * se)


def delong(pos: list[float], neg: list[float], z: float = Z975) -> tuple[float, float, float, float]:
    """DeLong et al. (1988) variance of the empirical AUROC; Wald interval.

    Returns (auroc, se, lower, upper).
    """
    m, n = len(pos), len(neg)

    def psi(x: float, y: float) -> float:
        return 1.0 if x > y else (0.5 if x == y else 0.0)

    v10 = [sum(psi(x, y) for y in neg) / n for x in pos]
    v01 = [sum(psi(x, y) for x in pos) / m for y in neg]
    a = sum(v10) / m
    s10 = sum((v - a) ** 2 for v in v10) / (m - 1) if m > 1 else 0.0
    s01 = sum((v - a) ** 2 for v in v01) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(s10 / m + s01 / n)
    return a, se, max(0.0, a - z * se), min(1.0, a + z * se)


def bootstrap_auroc(units: list[list[tuple[float, bool]]], reps: int = 10000,
                    seed: str = "devai") -> tuple[float, float, int]:
    """Percentile bootstrap interval of a pooled AUROC, resampling whole units.

    A unit is a list of (score, is_positive) pairs: one pair per unit gives the
    ordinary bootstrap, several (decisions sharing one text) a cluster bootstrap.
    Resamples lacking a class are skipped and counted. Returns (lower, upper, skipped).
    """
    rng = random.Random(seed)
    k = len(units)
    values: list[float] = []
    skipped = 0
    for _ in range(reps):
        pairs = [pr for _ in range(k) for pr in units[rng.randrange(k)]]
        pos = [s for s, o in pairs if o]
        neg = [s for s, o in pairs if not o]
        if not pos or not neg:
            skipped += 1
            continue
        values.append(auroc(pos, neg))
    if not values:
        raise ValueError("every bootstrap resample lacked a class; check the units for class imbalance")
    values.sort()
    lo = values[math.floor(0.025 * (len(values) - 1))]
    hi = values[math.ceil(0.975 * (len(values) - 1))]
    return lo, hi, skipped



# ---------------------------------------------------------------- rates

def poisson_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    if lam <= 0:
        return 1.0
    return min(1.0, math.fsum(math.exp(-lam + i * math.log(lam) - math.lgamma(i + 1))
                              for i in range(k + 1)))


def poisson_exact_ci(k: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Garwood (1936) exact interval for a Poisson mean given an observed count k."""
    a2 = alpha / 2.0

    def solve(f, target):  # f increasing in the mean
        lo, hi = 1e-12, float(10 * k + 50)
        while f(hi) < target:  # widen until the bracket holds the root, whatever alpha is
            lo, hi = hi, 2.0 * hi
        for _ in range(_BISECT_ITERS):
            mid = (lo + hi) / 2.0
            if f(mid) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    lower = 0.0 if k == 0 else solve(lambda m: 1.0 - poisson_cdf(k - 1, m), a2)
    upper = solve(lambda m: 1.0 - poisson_cdf(k, m), 1.0 - a2)
    return lower, upper


def holm_list(pvals: list[float]) -> list[float]:
    """Holm (1979) adjusted p-values for a list, in the input order."""
    adj = holm({str(i): p for i, p in enumerate(pvals)})
    return [adj[str(i)] for i in range(len(pvals))]


# ---------------------------------------------------------------- resampling, descriptive

def quantile(xs: list[float], q: float) -> float | None:
    """Hyndman-Fan type 7 quantile of unsorted data (None for an empty list)."""
    s = sorted(xs)
    if not s:
        return None
    return quantile_type7(s, q)


def median_ci(xs: list[float], level: float = 0.95):
    """Distribution-free median interval from data: (lo, hi, coverage, (r, n - r + 1)), or None."""
    s = sorted(xs)
    try:
        r, u, cov = median_ci_ranks(len(s), 1.0 - level)
    except ValueError:
        return None
    return s[r - 1], s[u - 1], cov, (r, u)


def quantile_ci(xs: list[float], q: float, level: float = 0.95) -> dict:
    """Shortest distribution-free rank interval [x(l), x(u)] for the q-quantile.

    Coverage P(l <= B <= u - 1), B ~ Bin(n, q). {'available': False, ...} when
    even [x(1), x(n)] does not reach the level.
    """
    s = sorted(xs)
    n = len(s)
    cdf: list[float] = []
    acc = 0.0
    for i in range(n + 1):
        acc += math.comb(n, i) * q ** i * (1 - q) ** (n - i)
        cdf.append(acc)

    def cov(lo: int, hi: int) -> float:
        return cdf[hi - 1] - (cdf[lo - 1] if lo >= 1 else 0.0)

    best = None
    for lo in range(1, n + 1):
        for hi in range(lo + 1, n + 1):
            c = cov(lo, hi)
            if c >= level:
                if best is None or hi - lo < best[1] - best[0]:
                    best = (lo, hi, c)
                break
    if best is None:
        return {"available": False, "max_coverage_with_min_max": cov(1, n)}
    lo, hi, c = best
    return {"available": True, "lo": s[lo - 1], "hi": s[hi - 1], "ranks": [lo, hi], "coverage": c}


def boot_ci(data: list, stat, level: float = 0.95, reps: int = 10000,
            seed: int | str = 20260927) -> tuple[float | None, float | None]:
    """Percentile bootstrap of stat(sample), resampling data (tuples for paired data)."""
    rng = random.Random(seed)
    n = len(data)
    values = []
    for _ in range(reps):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        try:
            v = stat(sample)
        except ZeroDivisionError:
            continue
        if v is not None:
            values.append(v)
    a = (1 - level) / 2
    return quantile(values, a), quantile(values, 1 - a)


def boot_ci_two_sample(x: list, y: list, stat, level: float = 0.95, reps: int = 10000,
                       seed: int | str = 20260927) -> tuple[float | None, float | None]:
    """Unpaired percentile bootstrap: resample x and y independently; stat(xs, ys)."""
    rng = random.Random(seed)
    values = []
    for _ in range(reps):
        xs = [x[rng.randrange(len(x))] for _ in range(len(x))]
        ys = [y[rng.randrange(len(y))] for _ in range(len(y))]
        values.append(stat(xs, ys))
    a = (1 - level) / 2
    return quantile(values, a), quantile(values, 1 - a)


def describe(xs: list[float]) -> dict:
    """n, range, mean, sd, type-7 quantiles, median interval, bootstrap interval of the mean.

    None values are dropped and counted (n_none_dropped); an empty input gives {"n": 0}.
    """
    import statistics
    s = sorted(v for v in xs if v is not None)
    n = len(s)
    dropped = len(xs) - n
    if n == 0:
        return {"n": 0, **({"n_none_dropped": dropped} if dropped else {})}
    out = {"n": n, "min": s[0], "max": s[-1], "mean": statistics.fmean(s),
           "sd": statistics.stdev(s) if n > 1 else None,
           "q10": quantile(s, 0.10), "q25": quantile(s, 0.25), "median": quantile(s, 0.5),
           "q75": quantile(s, 0.75), "q90": quantile(s, 0.90), "q95": quantile(s, 0.95)}
    ci = median_ci(s)
    out["median_ci95"] = None if ci is None else [ci[0], ci[1]]
    out["median_ci95_coverage"] = None if ci is None else round(ci[2], 4)
    out["median_ci95_ranks"] = None if ci is None else list(ci[3])
    if n > 1:
        out["mean_boot_ci95"] = list(boot_ci(s, statistics.fmean))
    if dropped:
        out["n_none_dropped"] = dropped  # values that could not be computed (e.g. a rate with no decode time)
    return out


def rnd(x, k: int = 3):
    """Round floats recursively (for JSON output)."""
    if isinstance(x, float):
        return round(x, k)
    if isinstance(x, list):
        return [rnd(v, k) for v in x]
    if isinstance(x, dict):
        return {a: rnd(b, k) for a, b in x.items()}
    return x


def stratified_bootstrap_mean_of_means(strata: list[list[float]], reps: int = 10000,
                                       seed: int = 20260927) -> tuple[float, float, float]:
    """Unweighted mean of per-stratum means with a percentile bootstrap that
    resamples items within each stratum. Returns (estimate, lower, upper)."""
    rng = random.Random(seed)
    est = sum(sum(s) / len(s) for s in strata) / len(strata)
    stats = []
    for _ in range(reps):
        tot = 0.0
        for s in strata:
            n = len(s)
            tot += sum(s[rng.randrange(n)] for _ in range(n)) / n
        stats.append(tot / len(strata))
    stats.sort()
    return est, quantile_type7(stats, 0.025), quantile_type7(stats, 0.975)


def paired_signflip_test(strata_diffs: list[list[float]], reps: int = 10000,
                         seed: int = 20260927) -> tuple[float, float]:
    """Monte Carlo paired randomization test for a mean-of-stratum-means difference.

    strata_diffs[s][i] = y_A(i) - y_B(i). Under H0 each difference's sign is
    exchangeable. Two-sided p with the +1 correction. Returns (statistic, p).
    """
    rng = random.Random(seed)

    def stat(signs=None) -> float:
        tot = 0.0
        for s, ds in enumerate(strata_diffs):
            if signs is None:
                tot += sum(ds) / len(ds)
            else:
                tot += sum(d * sg for d, sg in zip(ds, signs[s])) / len(ds)
        return tot / len(strata_diffs)

    obs = stat()
    extreme = 0
    for _ in range(reps):
        signs = [[1 if rng.random() < 0.5 else -1 for _ in ds] for ds in strata_diffs]
        if abs(stat(signs)) >= abs(obs) - 1e-12:
            extreme += 1
    return obs, (extreme + 1) / (reps + 1)


def paired_bootstrap_diff(strata_a: list[list[float]], strata_b: list[list[float]],
                          reps: int = 10000, seed: int = 20260927) -> tuple[float, float, float]:
    """Percentile bootstrap interval for a difference of mean-of-stratum-means,
    resampling paired items within each stratum. Returns (estimate, lower, upper)."""
    if len(strata_a) != len(strata_b) or any(len(a) != len(b) for a, b in zip(strata_a, strata_b)):
        raise ValueError("strata_a and strata_b must be paired: same strata, same length per stratum")
    rng = random.Random(seed)

    def mm(strata, idx=None) -> float:
        tot = 0.0
        for s, vals in enumerate(strata):
            tot += (sum(vals) if idx is None else sum(vals[i] for i in idx[s])) / len(vals)
        return tot / len(strata)

    est = mm(strata_a) - mm(strata_b)
    stats = []
    for _ in range(reps):
        idx = [[rng.randrange(len(s)) for _ in range(len(s))] for s in strata_a]
        stats.append(mm(strata_a, idx) - mm(strata_b, idx))
    stats.sort()
    return est, quantile_type7(stats, 0.025), quantile_type7(stats, 0.975)


def mde_paired_halfwidth(n: int, discordance: float, z: float = Z975) -> float:
    """Approximate half-width of a paired difference of proportions (b ~ c, Wald)."""
    return z * math.sqrt(discordance / n)


def mde_unpaired_halfwidth(n: int, p: float, z: float = Z975) -> float:
    """Approximate half-width of an unpaired difference of proportions at a common p."""
    return z * math.sqrt(2 * p * (1 - p) / n)
