"""Programming Pearls Column 4, Problem 10.

Introduce mutations into the binary search function from col4p1, run
each through Track 1 (functional correctness via oracle compare) and
Track 2 (internal proof obligations as runtime asserts), and report
what each track catches.

This is a meta-experiment on the bench grader. The result table tells
us the complementarity profile of Tracks 1 and 2: which bug classes
each one catches alone, which they overlap on, and which slip through
both. The "slip through both" cell is the scary one and is what we
want a clear honest picture of.

Each mutation has two implementations:

  - ``<id>_plain``: the buggy code, no asserts. Run by the Track 1
    grader, which compares the function's return to an oracle.
  - ``<id>_inst``:  the same buggy code with the SAME obligation
    asserts as the correct baseline. Run by the Track 2 grader,
    which watches for AssertionError firing during execution.

Both runners share a hard iteration cap (_MAX_ITER) so mutations
that loop forever surface as TimeoutError instead of hanging.
"""

from __future__ import annotations

import random
from typing import Callable

# Cap on iterations of the loop body. The correct algorithm needs
# at most ceil(log2(n+1)) iterations; for n <= 10**4, that's 14.
# 100 is generous enough that any genuine algorithm finishes and
# any infinite-loop mutation surfaces quickly.
_MAX_ITER = 100


# === Test battery ============================================================

def _battery():
    """Yield (arr, target) tuples covering edge cases plus a 1k-element
    random stress segment with mixed present/absent probes.
    """
    yield ([], 5)
    yield ([7], 7)
    yield ([7], 6)
    yield ([7], 8)
    yield ([3, 5, 7, 9], 1)
    yield ([3, 5, 7, 9], 100)
    arr = [1, 3, 5, 7, 9, 11, 13, 15]
    for v in arr:
        yield (arr, v)
    yield ([1, 2, 2, 2, 2, 3], 2)

    rng = random.Random(0xC0FFEE)
    n = 1000
    a = sorted(rng.sample(range(1, 10 * n), n))
    aset = set(a)
    for _ in range(300):
        if rng.random() < 0.5:
            yield (a, a[rng.randrange(n)])
        else:
            t = rng.randrange(1, 10 * n)
            while t in aset:
                t = rng.randrange(1, 10 * n)
            yield (a, t)


def _is_valid(arr, target, returned) -> bool:
    """Track 1's oracle: returned must be ``-1`` (only when target
    absent) or a valid index where ``arr[i] == target``.
    """
    if returned == -1:
        return target not in arr
    if not isinstance(returned, int):
        return False
    return 0 <= returned < len(arr) and arr[returned] == target


# === Baseline ================================================================

def baseline_plain(arr, target):
    lo, hi = 0, len(arr) - 1
    for _ in range(_MAX_ITER):
        if lo > hi:
            return -1
        mid = lo + (hi - lo) // 2
        v = arr[mid]
        if v < target:
            lo = mid + 1
        elif v > target:
            hi = mid - 1
        else:
            return mid
    raise TimeoutError("max_iter exceeded")


def baseline_inst(arr, target):
    n = len(arr)
    lo, hi = 0, n - 1
    prev_f = n + 1
    for _ in range(_MAX_ITER):
        assert 0 <= lo <= n,            "lo_in_range"
        assert -1 <= hi <= n - 1,       "hi_in_range"
        if lo > hi:
            return -1
        f = hi - lo + 1
        assert f < prev_f,              "termination_measure_decreases"
        prev_f = f
        assert 0 <= lo <= hi <= n - 1,  "body_bounds"
        mid = lo + (hi - lo) // 2
        assert lo <= mid <= hi,         "mid_in_window"
        assert 0 <= mid <= n - 1,       "mid_indexable"
        v = arr[mid]
        if v < target:
            lo = mid + 1
        elif v > target:
            hi = mid - 1
        else:
            return mid
    raise TimeoutError("max_iter exceeded")


# === Mutations ===============================================================

# --- M1: hi initialized off-by-one high (hi = len(arr)) ---
def m1_plain(arr, target):
    lo, hi = 0, len(arr)  # BUG
    for _ in range(_MAX_ITER):
        if lo > hi:
            return -1
        mid = lo + (hi - lo) // 2
        v = arr[mid]
        if v < target:
            lo = mid + 1
        elif v > target:
            hi = mid - 1
        else:
            return mid
    raise TimeoutError("max_iter exceeded")


def m1_inst(arr, target):
    n = len(arr)
    lo, hi = 0, n  # BUG
    prev_f = n + 2
    for _ in range(_MAX_ITER):
        assert 0 <= lo <= n,            "lo_in_range"
        assert -1 <= hi <= n - 1,       "hi_in_range"
        if lo > hi:
            return -1
        f = hi - lo + 1
        assert f < prev_f,              "termination_measure_decreases"
        prev_f = f
        assert 0 <= lo <= hi <= n - 1,  "body_bounds"
        mid = lo + (hi - lo) // 2
        assert lo <= mid <= hi,         "mid_in_window"
        assert 0 <= mid <= n - 1,       "mid_indexable"
        v = arr[mid]
        if v < target:
            lo = mid + 1
        elif v > target:
            hi = mid - 1
        else:
            return mid
    raise TimeoutError("max_iter exceeded")


# --- M2: lo update doesn't advance (lo = mid) ---
def m2_plain(arr, target):
    lo, hi = 0, len(arr) - 1
    for _ in range(_MAX_ITER):
        if lo > hi:
            return -1
        mid = lo + (hi - lo) // 2
        v = arr[mid]
        if v < target:
            lo = mid  # BUG
        elif v > target:
            hi = mid - 1
        else:
            return mid
    raise TimeoutError("max_iter exceeded")


def m2_inst(arr, target):
    n = len(arr)
    lo, hi = 0, n - 1
    prev_f = n + 1
    for _ in range(_MAX_ITER):
        assert 0 <= lo <= n,            "lo_in_range"
        assert -1 <= hi <= n - 1,       "hi_in_range"
        if lo > hi:
            return -1
        f = hi - lo + 1
        assert f < prev_f,              "termination_measure_decreases"
        prev_f = f
        assert 0 <= lo <= hi <= n - 1,  "body_bounds"
        mid = lo + (hi - lo) // 2
        assert lo <= mid <= hi,         "mid_in_window"
        assert 0 <= mid <= n - 1,       "mid_indexable"
        v = arr[mid]
        if v < target:
            lo = mid  # BUG
        elif v > target:
            hi = mid - 1
        else:
            return mid
    raise TimeoutError("max_iter exceeded")


# --- M3: strict loop guard (if lo >= hi: return -1) ---
def m3_plain(arr, target):
    lo, hi = 0, len(arr) - 1
    for _ in range(_MAX_ITER):
        if lo >= hi:  # BUG: should be lo > hi
            return -1
        mid = lo + (hi - lo) // 2
        v = arr[mid]
        if v < target:
            lo = mid + 1
        elif v > target:
            hi = mid - 1
        else:
            return mid
    raise TimeoutError("max_iter exceeded")


def m3_inst(arr, target):
    n = len(arr)
    lo, hi = 0, n - 1
    prev_f = n + 1
    for _ in range(_MAX_ITER):
        assert 0 <= lo <= n,            "lo_in_range"
        assert -1 <= hi <= n - 1,       "hi_in_range"
        if lo >= hi:  # BUG
            return -1
        f = hi - lo + 1
        assert f < prev_f,              "termination_measure_decreases"
        prev_f = f
        assert 0 <= lo <= hi <= n - 1,  "body_bounds"
        mid = lo + (hi - lo) // 2
        assert lo <= mid <= hi,         "mid_in_window"
        assert 0 <= mid <= n - 1,       "mid_indexable"
        v = arr[mid]
        if v < target:
            lo = mid + 1
        elif v > target:
            hi = mid - 1
        else:
            return mid
    raise TimeoutError("max_iter exceeded")


# --- M4: naive midpoint mid = (lo + hi) // 2 (overflow only in fixed-width) ---
def m4_plain(arr, target):
    lo, hi = 0, len(arr) - 1
    for _ in range(_MAX_ITER):
        if lo > hi:
            return -1
        mid = (lo + hi) // 2  # BUG (only manifests in fixed-width int)
        v = arr[mid]
        if v < target:
            lo = mid + 1
        elif v > target:
            hi = mid - 1
        else:
            return mid
    raise TimeoutError("max_iter exceeded")


def m4_inst(arr, target):
    n = len(arr)
    lo, hi = 0, n - 1
    prev_f = n + 1
    for _ in range(_MAX_ITER):
        assert 0 <= lo <= n,            "lo_in_range"
        assert -1 <= hi <= n - 1,       "hi_in_range"
        if lo > hi:
            return -1
        f = hi - lo + 1
        assert f < prev_f,              "termination_measure_decreases"
        prev_f = f
        assert 0 <= lo <= hi <= n - 1,  "body_bounds"
        mid = (lo + hi) // 2  # BUG
        assert lo <= mid <= hi,         "mid_in_window"
        assert 0 <= mid <= n - 1,       "mid_indexable"
        v = arr[mid]
        if v < target:
            lo = mid + 1
        elif v > target:
            hi = mid - 1
        else:
            return mid
    raise TimeoutError("max_iter exceeded")


# --- M5: off-by-one updates (lo = mid - 1, hi = mid + 1) ---
def m5_plain(arr, target):
    lo, hi = 0, len(arr) - 1
    for _ in range(_MAX_ITER):
        if lo > hi:
            return -1
        mid = lo + (hi - lo) // 2
        v = arr[mid]
        if v < target:
            lo = mid - 1  # BUG
        elif v > target:
            hi = mid + 1  # BUG
        else:
            return mid
    raise TimeoutError("max_iter exceeded")


def m5_inst(arr, target):
    n = len(arr)
    lo, hi = 0, n - 1
    prev_f = n + 1
    for _ in range(_MAX_ITER):
        assert 0 <= lo <= n,            "lo_in_range"
        assert -1 <= hi <= n - 1,       "hi_in_range"
        if lo > hi:
            return -1
        f = hi - lo + 1
        assert f < prev_f,              "termination_measure_decreases"
        prev_f = f
        assert 0 <= lo <= hi <= n - 1,  "body_bounds"
        mid = lo + (hi - lo) // 2
        assert lo <= mid <= hi,         "mid_in_window"
        assert 0 <= mid <= n - 1,       "mid_indexable"
        v = arr[mid]
        if v < target:
            lo = mid - 1  # BUG
        elif v > target:
            hi = mid + 1  # BUG
        else:
            return mid
    raise TimeoutError("max_iter exceeded")


# --- M6: return mid + 1 on match ---
def m6_plain(arr, target):
    lo, hi = 0, len(arr) - 1
    for _ in range(_MAX_ITER):
        if lo > hi:
            return -1
        mid = lo + (hi - lo) // 2
        v = arr[mid]
        if v < target:
            lo = mid + 1
        elif v > target:
            hi = mid - 1
        else:
            return mid + 1  # BUG
    raise TimeoutError("max_iter exceeded")


def m6_inst(arr, target):
    n = len(arr)
    lo, hi = 0, n - 1
    prev_f = n + 1
    for _ in range(_MAX_ITER):
        assert 0 <= lo <= n,            "lo_in_range"
        assert -1 <= hi <= n - 1,       "hi_in_range"
        if lo > hi:
            return -1
        f = hi - lo + 1
        assert f < prev_f,              "termination_measure_decreases"
        prev_f = f
        assert 0 <= lo <= hi <= n - 1,  "body_bounds"
        mid = lo + (hi - lo) // 2
        assert lo <= mid <= hi,         "mid_in_window"
        assert 0 <= mid <= n - 1,       "mid_indexable"
        v = arr[mid]
        if v < target:
            lo = mid + 1
        elif v > target:
            hi = mid - 1
        else:
            return mid + 1  # BUG
    raise TimeoutError("max_iter exceeded")


# --- M7: probe at arr[lo] instead of arr[mid] ---
def m7_plain(arr, target):
    lo, hi = 0, len(arr) - 1
    for _ in range(_MAX_ITER):
        if lo > hi:
            return -1
        mid = lo + (hi - lo) // 2
        v = arr[lo]  # BUG
        if v < target:
            lo = mid + 1
        elif v > target:
            hi = mid - 1
        else:
            return mid
    raise TimeoutError("max_iter exceeded")


def m7_inst(arr, target):
    n = len(arr)
    lo, hi = 0, n - 1
    prev_f = n + 1
    for _ in range(_MAX_ITER):
        assert 0 <= lo <= n,            "lo_in_range"
        assert -1 <= hi <= n - 1,       "hi_in_range"
        if lo > hi:
            return -1
        f = hi - lo + 1
        assert f < prev_f,              "termination_measure_decreases"
        prev_f = f
        assert 0 <= lo <= hi <= n - 1,  "body_bounds"
        mid = lo + (hi - lo) // 2
        assert lo <= mid <= hi,         "mid_in_window"
        assert 0 <= mid <= n - 1,       "mid_indexable"
        v = arr[lo]  # BUG
        if v < target:
            lo = mid + 1
        elif v > target:
            hi = mid - 1
        else:
            return mid
    raise TimeoutError("max_iter exceeded")


# --- M8: lo initialized to 1 (skips arr[0]) ---
def m8_plain(arr, target):
    lo, hi = 1, len(arr) - 1  # BUG
    for _ in range(_MAX_ITER):
        if lo > hi:
            return -1
        mid = lo + (hi - lo) // 2
        v = arr[mid]
        if v < target:
            lo = mid + 1
        elif v > target:
            hi = mid - 1
        else:
            return mid
    raise TimeoutError("max_iter exceeded")


def m8_inst(arr, target):
    n = len(arr)
    lo, hi = 1, n - 1  # BUG
    prev_f = n + 1
    for _ in range(_MAX_ITER):
        assert 0 <= lo <= n,            "lo_in_range"
        assert -1 <= hi <= n - 1,       "hi_in_range"
        if lo > hi:
            return -1
        f = hi - lo + 1
        assert f < prev_f,              "termination_measure_decreases"
        prev_f = f
        assert 0 <= lo <= hi <= n - 1,  "body_bounds"
        mid = lo + (hi - lo) // 2
        assert lo <= mid <= hi,         "mid_in_window"
        assert 0 <= mid <= n - 1,       "mid_indexable"
        v = arr[mid]
        if v < target:
            lo = mid + 1
        elif v > target:
            hi = mid - 1
        else:
            return mid
    raise TimeoutError("max_iter exceeded")


# --- M9: missing return -1 (falls through to None) ---
def m9_plain(arr, target):
    lo, hi = 0, len(arr) - 1
    for _ in range(_MAX_ITER):
        if lo > hi:
            return None  # BUG: should be -1
        mid = lo + (hi - lo) // 2
        v = arr[mid]
        if v < target:
            lo = mid + 1
        elif v > target:
            hi = mid - 1
        else:
            return mid
    raise TimeoutError("max_iter exceeded")


def m9_inst(arr, target):
    n = len(arr)
    lo, hi = 0, n - 1
    prev_f = n + 1
    for _ in range(_MAX_ITER):
        assert 0 <= lo <= n,            "lo_in_range"
        assert -1 <= hi <= n - 1,       "hi_in_range"
        if lo > hi:
            return None  # BUG
        f = hi - lo + 1
        assert f < prev_f,              "termination_measure_decreases"
        prev_f = f
        assert 0 <= lo <= hi <= n - 1,  "body_bounds"
        mid = lo + (hi - lo) // 2
        assert lo <= mid <= hi,         "mid_in_window"
        assert 0 <= mid <= n - 1,       "mid_indexable"
        v = arr[mid]
        if v < target:
            lo = mid + 1
        elif v > target:
            hi = mid - 1
        else:
            return mid
    raise TimeoutError("max_iter exceeded")


# === Track runners ===========================================================

def track1(fn: Callable) -> tuple[bool, str]:
    """Functional correctness via oracle compare. Returns
    ``(passed, first_failure_summary)``. ``passed=True`` means every
    test input got a valid answer per ``_is_valid``.
    """
    for arr, target in _battery():
        try:
            ans = fn(arr, target)
        except TimeoutError:
            return False, "TimeoutError (loop didn't terminate)"
        except IndexError as e:
            return False, f"IndexError: {e}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
        if not _is_valid(arr, target, ans):
            arr_repr = repr(arr) if len(arr) < 10 else f"<len-{len(arr)}>"
            return False, f"wrong answer: bs({arr_repr}, {target}) = {ans!r}"
    return True, ""


def track2(fn_inst: Callable) -> tuple[bool, str]:
    """Internal obligation asserts. Returns
    ``(passed, first_assertion_or_runtime_error)``.
    """
    for arr, target in _battery():
        try:
            fn_inst(arr, target)
        except AssertionError as e:
            return False, f"obligation fired: {e}"
        except TimeoutError:
            return False, "TimeoutError (loop didn't terminate)"
        except Exception as e:
            # Runtime errors (IndexError, etc.) count as Track 2 failures
            # too -- the program failed to complete cleanly. Keep them
            # distinct from named obligations in the report.
            return False, f"runtime error: {type(e).__name__}: {e}"
    return True, ""


# === Driver ==================================================================

MUTATIONS = [
    ("base", "(correct)",                       baseline_plain, baseline_inst),
    ("M1",   "hi = len(arr)",                   m1_plain,       m1_inst),
    ("M2",   "lo = mid (no advance)",           m2_plain,       m2_inst),
    ("M3",   "if lo >= hi (strict guard)",      m3_plain,       m3_inst),
    ("M4",   "mid = (lo + hi) // 2",            m4_plain,       m4_inst),
    ("M5",   "lo=mid-1, hi=mid+1",              m5_plain,       m5_inst),
    ("M6",   "return mid + 1 on match",         m6_plain,       m6_inst),
    ("M7",   "v = arr[lo] (wrong probe)",       m7_plain,       m7_inst),
    ("M8",   "lo = 1 (skip arr[0])",            m8_plain,       m8_inst),
    ("M9",   "return None on miss",             m9_plain,       m9_inst),
]


def main() -> None:
    print(f"\n{'ID':<6}{'Mutation':<32}{'T1':<8}{'T2':<8}{'First failure':<60}")
    print("-" * 114)
    summary = {"T1_only": 0, "T2_only": 0, "both": 0, "neither": 0}

    for mid, desc, plain, inst in MUTATIONS:
        t1_pass, t1_msg = track1(plain)
        t2_pass, t2_msg = track2(inst)

        t1_str = "pass" if t1_pass else "CATCH"
        t2_str = "pass" if t2_pass else "CATCH"

        if mid != "base":
            if not t1_pass and not t2_pass:
                summary["both"] += 1
            elif not t1_pass:
                summary["T1_only"] += 1
            elif not t2_pass:
                summary["T2_only"] += 1
            else:
                summary["neither"] += 1

        first = t1_msg if not t1_pass else (t2_msg if not t2_pass else "-")
        # Cap the message width.
        if len(first) > 56:
            first = first[:55] + "..."
        print(f"{mid:<6}{desc:<32}{t1_str:<8}{t2_str:<8}{first:<60}")

    print("-" * 114)
    n_mut = len(MUTATIONS) - 1
    print(f"Across {n_mut} non-baseline mutations:")
    print(f"  caught by both tracks:  {summary['both']:>2}")
    print(f"  caught by Track 1 only: {summary['T1_only']:>2}")
    print(f"  caught by Track 2 only: {summary['T2_only']:>2}")
    print(f"  CAUGHT BY NEITHER:      {summary['neither']:>2}")


if __name__ == "__main__":
    main()
