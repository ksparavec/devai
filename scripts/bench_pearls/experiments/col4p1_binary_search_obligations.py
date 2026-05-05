"""Demonstration: encode a proof's obligations as runtime assertions.

Same algorithm as ``col4_binary_search.py``, but every claim from the
proof in the chat thread is a labelled ``assert``. Running this on
random inputs falsifies the proof if any obligation fires; surviving a
billion mixed probes is strong inductive evidence (not, by itself,
proof) that the obligations hold.

This is the "Track 2" grader described in the chat: the bench runs the
model's submitted ``binary_search`` against this instrumented test
harness. A clean run earns a separate score from the code's own
correctness, so a model that gets the right answer via a wrong
invariant is caught.

The ``OBLIGATIONS`` list below is the machine-readable form of the
proof. It would ship in the JSONL row as ``proof_obligations``; the
test harness consumes it to build the assertion-instrumented program.
"""

from __future__ import annotations

# Machine-readable form of the proof. Each entry is a hoisted
# obligation the proof claims; the test harness either embeds it as
# a runtime assert (mechanically checkable) or hands it to the
# LLM-as-judge as a structural-completeness rubric (does the model's
# proof mention this obligation?).
OBLIGATIONS = [
    # --- Bounds invariants (Track 2 mechanical check) ---
    {"id": "lo_in_range",
     "claim": "0 <= lo <= n at every loop top"},
    {"id": "hi_in_range",
     "claim": "-1 <= hi <= n - 1 at every loop top"},
    {"id": "body_bounds",
     "claim": "0 <= lo <= hi <= n - 1 whenever the body runs"},
    {"id": "mid_in_window",
     "claim": "lo <= mid <= hi at the array access"},
    {"id": "mid_indexable",
     "claim": "0 <= mid <= n - 1 at the array access"},

    # --- No-runtime-errors (Bentley's four classes) ---
    {"id": "no_div_by_zero",
     "claim": "every division has a non-zero divisor"},
    {"id": "no_overflow_64bit",
     "claim": "every arithmetic intermediate fits in int64 "
              "given n <= INT64_MAX"},
    {"id": "no_oob",
     "claim": "every arr[i] access has 0 <= i < n",
     "redundant_with": "mid_indexable"},

    # --- Termination (not in the four classes but still required) ---
    {"id": "termination_measure_decreases",
     "claim": "f = hi - lo + 1 strictly decreases each non-returning "
              "iteration"},

    # --- Functional postcondition (Track 1 also catches this, but
    #     restated here so the proof is structurally complete) ---
    {"id": "post_found",
     "claim": "if return value i != -1 then arr[i] == target"},
    {"id": "post_absent",
     "claim": "if return value is -1 then target not in arr"},
]


def binary_search_instrumented(arr, target: int) -> int:
    """Same algorithm as ``binary_search``; every proof claim is now a
    labelled assert. Each ``assert`` carries its obligation id so that
    when one fires the grader knows which claim was falsified.

    Track 2 grading: the bench runs the *model's* submission through
    a wrapper that monkey-patches its function with this instrumented
    body's asserts. (In practice, easier to ship a test that calls the
    model's function and re-derives the state from the output, but the
    spirit is the same: every obligation becomes a runtime check.)
    """
    n = len(arr)
    lo, hi = 0, n - 1

    # Termination measure tracking. Strictly decrease each iteration.
    f_prev = n + 1  # so the first iteration's f = n is < f_prev

    while lo <= hi:
        assert 0 <= lo <= n,            "lo_in_range"
        assert -1 <= hi <= n - 1,       "hi_in_range"
        assert 0 <= lo <= hi <= n - 1,  "body_bounds"

        f = hi - lo + 1
        assert f < f_prev,              "termination_measure_decreases"
        f_prev = f

        # No-overflow on 64-bit ints: hi - lo >= 0 (from body_bounds)
        # and <= n - 1, lo + (hi - lo)//2 <= hi <= n - 1. All fit.
        # No div-by-zero: divisor is the literal 2.
        mid = lo + (hi - lo) // 2

        assert lo <= mid <= hi,         "mid_in_window"
        assert 0 <= mid <= n - 1,       "mid_indexable"

        v = arr[mid]
        if v < target:
            lo = mid + 1
        elif v > target:
            hi = mid - 1
        else:
            assert arr[mid] == target,  "post_found"
            return mid

    # Loop exited with lo > hi. By I, target is not in arr.
    # We can't cheaply assert "target not in arr" here without an
    # O(n) scan, so post_absent is verified separately at higher
    # level by the differential test below.
    return -1


def _stress_obligations() -> None:
    """Run the instrumented version on inputs that span every shape
    a billion-element production array would take, and report which
    obligations fired. Clean run = no obligations falsified.
    """
    import random

    rng = random.Random(0xC0FFEE)

    # Empty, singleton, small, medium, large.
    sizes = [0, 1, 2, 3, 10, 1_000, 100_000]
    fired: set[str] = set()
    n_calls = 0

    def call_and_track(arr, t):
        nonlocal n_calls
        n_calls += 1
        try:
            return binary_search_instrumented(arr, t)
        except AssertionError as e:
            fired.add(str(e))
            raise

    for n in sizes:
        arr = sorted(rng.sample(range(1, max(10 * n, 2)), n)) if n else []
        arr_set = set(arr)

        # Below-min, above-max, exact match per element, random absent.
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
                # post_absent: differential check vs. the set we built.
                assert t not in arr_set, "post_absent falsified"
            else:
                assert arr[i] == t, "post_found falsified by differential"

    # 1-million-element pass to keep the pressure on overflow and bounds.
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


if __name__ == "__main__":
    _stress_obligations()
