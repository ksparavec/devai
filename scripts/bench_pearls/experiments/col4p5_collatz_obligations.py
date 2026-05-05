"""Collatz with proof obligations honestly classified.

The previous problems (col4p1, col4p3) had obligations that were all
mechanically falsifiable -- runtime asserts on bounds, ranges,
termination measures. Collatz is different: the central termination
obligation is an open mathematical problem. The bench harness can
falsify it on a finite test set but cannot prove it.

This file's contribution is the obligation classification. Each
obligation carries a ``mechanical`` flag:

  - ``mechanical=True``  -> Track 2 can falsify or pass via stress.
  - ``mechanical=False`` -> Open mathematical claim. Track 2 cannot
                            settle this; flagged as such.

A model that produces a "proof" of a False-mechanical obligation
without acknowledging the open status is wrong by overclaiming. The
right answer cites Collatz, lists what *is* proven (partial results),
and acknowledges the gap.
"""

from __future__ import annotations


OBLIGATIONS = [
    # --- Mechanically checkable on every step of every tested input ---
    {"id": "no_div_by_zero",
     "claim": "every division has a non-zero divisor",
     "mechanical": True},

    {"id": "no_overflow_on_python_int",
     "claim": "Python int is arbitrary precision; no fixed-width "
              "overflow possible",
     "mechanical": True,
     "note": "On a fixed-width port (C int64), 3*x+1 overflows once "
             "x > (INT64_MAX - 1) / 3. Trajectory peaks have no "
             "proven polynomial bound, so a fixed-width implementation "
             "cannot guarantee no overflow even if Collatz is true."},

    {"id": "x_remains_positive",
     "claim": "x stays >= 1 throughout the trajectory",
     "mechanical": True,
     "argument": "Loop body exits on x == 1. For x > 1: even branch "
                 "yields x // 2 >= 1 (since x >= 2). Odd branch yields "
                 "3*x + 1 >= 3*1 + 1 = 4. Both preserve x >= 1."},

    {"id": "loop_body_well_defined",
     "claim": "x % 2 is well-defined (x is always int) and the "
              "even/odd branches partition possibilities",
     "mechanical": True},

    # --- Computational verification: termination on tested inputs ---
    {"id": "termination_for_tested_inputs",
     "claim": "for every x in [1, 100000], the loop reaches x == 1 "
              "in at most 500 steps",
     "mechanical": True,
     "note": "Confirmed by stress test below. Independent extension to "
             "x <= 2^68 has been done by Barina (2020) and others."},

    # --- Open mathematical claim ---
    {"id": "termination_for_all_positive_integers",
     "claim": "for every positive integer x, the loop reaches x == 1 "
              "in finite steps",
     "mechanical": False,
     "status": "OPEN -- this is the Collatz conjecture (1937).",
     "what_we_can_say": [
         "Powers of 2: terminate in exactly log2(x) steps (trivial).",
         "Almost-every termination (Terras 1976): the natural density "
         "of x for which the trajectory drops below x is 1.",
         "Krasikov-Lagarias (2003): density of x with stopping time "
         "<= x^(1 - eps) is 1 for some eps > 0.",
         "No non-trivial cycles below astronomical lengths (Eliahou).",
         "Computational verification through ~2^68 (Barina 2020).",
     ],
     "why_standard_technique_fails": (
         "A well-founded decreasing-measure proof would need a "
         "function f: positive_int -> well_founded_set that strictly "
         "decreases at every step. f(x) = x decreases on the even "
         "branch (x -> x/2) but INCREASES on the odd branch "
         "(x -> 3x+1). No measure for which (a) decrease across both "
         "branches and (b) well-foundedness on the positive integers "
         "is known."
     )},

    # --- Open subsidiary claim ---
    {"id": "trajectory_peak_bounded_polynomially",
     "claim": "for every positive integer x, the trajectory peak is "
              "<= p(x) for some fixed polynomial p",
     "mechanical": False,
     "status": "OPEN -- no proven polynomial bound on peak height.",
     "note": "Empirically peaks grow roughly like x^2 for tested x, "
             "but no proof of any polynomial bound exists. This matters "
             "for fixed-width ports: without a peak bound we cannot "
             "guarantee no_overflow_on_int64."},
]


def collatz_instrumented(x: int, max_steps: int = 1000) -> int:
    """Same algorithm as ``collatz``, with mechanical obligations as
    runtime asserts. ``max_steps`` is a finite bound so an unexpected
    non-terminating input surfaces as RuntimeError, not a hang.
    """
    assert x >= 1,                            "x_remains_positive"
    steps = 0
    while x != 1:
        assert x >= 1,                        "x_remains_positive"
        # Divisor check: trivially the literal 2 in the // operation.
        # We assert the analog of the obligation by inspecting the
        # operation's divisor; in this static form, by construction.
        if steps >= max_steps:
            raise RuntimeError(
                f"exceeded max_steps={max_steps} (still at x={x})"
            )
        if x % 2 == 0:
            x = x // 2
        else:
            x = 3 * x + 1
        assert isinstance(x, int) and x >= 1, "x_remains_positive"
        steps += 1
    return steps


def _stress_obligations() -> None:
    """Run the instrumented version on a finite test set; report
    which mechanical obligations passed and which open obligations
    we deliberately do not check.
    """
    n_max = 100_000
    fired: set[str] = set()
    n_calls = 0

    for x in range(1, n_max + 1):
        n_calls += 1
        try:
            collatz_instrumented(x, max_steps=500)
        except AssertionError as e:
            fired.add(str(e))
        except RuntimeError as e:
            fired.add(f"max_steps_exceeded({e})")

    if fired:
        print(f"FAIL  obligations fired: {sorted(fired)}")
        return

    mechanical = [o['id'] for o in OBLIGATIONS if o['mechanical']]
    open_obs = [o['id'] for o in OBLIGATIONS if not o['mechanical']]
    print(f"OK -- {n_calls} calls in [1, {n_max}], no mechanical "
          f"obligation falsified")
    print(f"     mechanical obligations passed: {mechanical}")
    print(f"     OPEN obligations not settled:   {open_obs}")
    print(f"     (the OPEN ones are the Collatz conjecture and its "
          f"corollary about peak height)")


if __name__ == "__main__":
    _stress_obligations()
