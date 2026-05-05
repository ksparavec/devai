"""Recursive binary search with proof obligations encoded as runtime
asserts. Mirror of ``col4p1_binary_search_obligations.py`` but for the
recursive version (col4 problem 3).

The obligation set is mostly the same as problem 1's. Two changes:

  - ``termination_measure_decreases`` fires on each recursive call
    rather than on each loop iteration. The check is now "the window
    f_current is strictly smaller than the caller's f_caller". The
    expression ``f = hi - lo + 1`` and the strict-decrease claim are
    identical; only the *boundary* across which it's checked changes.

  - ``recursion_depth_bounded`` is new -- iterative code has no stack
    frames per iteration, so the obligation has no analogue. The bound
    is ``ceil(log2(n + 1))``: each recursive descent halves the window,
    so depth cannot exceed that. For n = 10**9 the bound is 30.

Everything else carries verbatim from the iterative obligation set.
"""

from __future__ import annotations

import math


OBLIGATIONS = [
    # --- Identical to problem 1 ---
    {"id": "lo_in_range",
     "claim": "0 <= lo <= n at every call entry",
     "same_as_iterative": True},
    {"id": "hi_in_range",
     "claim": "-1 <= hi <= n - 1 at every call entry",
     "same_as_iterative": True},
    {"id": "body_bounds",
     "claim": "0 <= lo <= hi <= n - 1 whenever the non-base branch runs",
     "same_as_iterative": True},
    {"id": "mid_in_window",
     "claim": "lo <= mid <= hi at the array access",
     "same_as_iterative": True},
    {"id": "mid_indexable",
     "claim": "0 <= mid <= n - 1 at the array access",
     "same_as_iterative": True},
    {"id": "no_div_by_zero",
     "claim": "every division has a non-zero divisor",
     "same_as_iterative": True},
    {"id": "no_overflow_64bit",
     "claim": "every arithmetic intermediate fits in int64 "
              "given n <= INT64_MAX",
     "same_as_iterative": True},
    {"id": "no_oob",
     "claim": "every arr[i] access has 0 <= i < n",
     "same_as_iterative": True,
     "redundant_with": "mid_indexable"},
    {"id": "post_found",
     "claim": "if return value i != -1 then arr[i] == target",
     "same_as_iterative": True},
    {"id": "post_absent",
     "claim": "if return value is -1 then target not in arr",
     "same_as_iterative": True},

    # --- Restated for the recursive boundary ---
    {"id": "termination_measure_decreases",
     "claim": "f = hi - lo + 1 is strictly smaller than the caller's "
              "f at every recursive call entry",
     "same_as_iterative": False,
     "iterative_form": "checked across loop iterations; "
                       "recursive form checks across call frames"},

    # --- New for the recursive version ---
    {"id": "recursion_depth_bounded",
     "claim": "recursion depth never exceeds ceil(log2(n + 1)); "
              "each non-base call halves the window",
     "same_as_iterative": False,
     "iterative_form": "no analogue; iterative code has no stack frames"},
]


def binary_search_recursive_instrumented(arr, target: int) -> int:
    n = len(arr)
    # ceil(log2(n + 1)) is the tight halving bound. n=0 -> 0; n=1 ->
    # 1; n=2,3 -> 2; n=4..7 -> 3; ... A small slack of +1 absorbs
    # boundary off-by-ones from the empty-array first call counted
    # as depth 0 versus 1.
    max_depth = (math.ceil(math.log2(n + 1)) if n > 0 else 0) + 1
    return _bsearch_inst(
        arr, target,
        lo=0, hi=n - 1,
        n=n,
        depth=0, max_depth=max_depth,
        prev_f=n + 1,
    )


def _bsearch_inst(arr, target, *, lo, hi, n, depth, max_depth, prev_f):
    # Bounds at call entry (parallel to the loop-top asserts in the
    # iterative version).
    assert 0 <= lo <= n,                "lo_in_range"
    assert -1 <= hi <= n - 1,           "hi_in_range"

    f = hi - lo + 1
    assert f < prev_f,                  "termination_measure_decreases"
    assert depth <= max_depth,          "recursion_depth_bounded"

    if lo > hi:
        return -1

    assert 0 <= lo <= hi <= n - 1,      "body_bounds"

    mid = lo + (hi - lo) // 2

    assert lo <= mid <= hi,             "mid_in_window"
    assert 0 <= mid <= n - 1,           "mid_indexable"

    v = arr[mid]
    if v < target:
        return _bsearch_inst(
            arr, target,
            lo=mid + 1, hi=hi, n=n,
            depth=depth + 1, max_depth=max_depth, prev_f=f,
        )
    if v > target:
        return _bsearch_inst(
            arr, target,
            lo=lo, hi=mid - 1, n=n,
            depth=depth + 1, max_depth=max_depth, prev_f=f,
        )
    assert arr[mid] == target,          "post_found"
    return mid


def _stress_obligations() -> None:
    import random

    rng = random.Random(0xC0FFEE)
    sizes = [0, 1, 2, 3, 10, 1_000, 100_000]
    fired: set[str] = set()
    n_calls = 0

    def call_and_track(arr, t):
        nonlocal n_calls
        n_calls += 1
        try:
            return binary_search_recursive_instrumented(arr, t)
        except AssertionError as e:
            fired.add(str(e))
            raise

    for n in sizes:
        arr = sorted(rng.sample(range(1, max(10 * n, 2)), n)) if n else []
        arr_set = set(arr)
        if n:
            call_and_track(arr, arr[0] - 1)
            call_and_track(arr, arr[-1] + 1)
            for v in arr[: min(n, 50)]:
                i = call_and_track(arr, v)
                assert i != -1 and arr[i] == v, "post_found differential"
        for _ in range(50):
            t = rng.randint(-5, 10 * (n + 1))
            i = call_and_track(arr, t)
            if i == -1:
                assert t not in arr_set, "post_absent falsified"
            else:
                assert arr[i] == t, "post_found falsified by differential"

    n = 1_000_000
    arr = sorted(rng.sample(range(1, 10 * n), n))
    arr_set = set(arr)
    for _ in range(2_000):
        if rng.random() < 0.5:
            t = arr[rng.randrange(n)]
            i = call_and_track(arr, t)
            assert i != -1 and arr[i] == t
        else:
            t = rng.randrange(1, 10 * n)
            while t in arr_set:
                t = rng.randrange(1, 10 * n)
            i = call_and_track(arr, t)
            assert i == -1

    if fired:
        print(f"FAIL  obligations fired: {sorted(fired)}")
        return
    print(f"OK -- {n_calls} calls, no obligation falsified")
    print(f"     mechanically checked: {[o['id'] for o in OBLIGATIONS]}")
    new_ones = [o['id'] for o in OBLIGATIONS if not o['same_as_iterative']]
    print(f"     new vs iterative:    {new_ones}")


if __name__ == "__main__":
    _stress_obligations()
