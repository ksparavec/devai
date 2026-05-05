# Column 4, Problem 5 -- Termination of the 3n+1 sequence

## Problem statement

Bentley gives the program:

    while x != 1 do
        if even(x)
            x = x / 2
        else
            x = 3 * x + 1

and asks: prove this terminates when the input x is a positive integer.

## Rubric for grading

  [ ] Identifies the program as the **Collatz conjecture** (also
      known as the 3n+1 problem, Ulam conjecture, Syracuse problem,
      Hasse's algorithm). Recognition is the key check.
  [ ] States that termination for arbitrary positive integers is
      OPEN -- no proof exists.
  [ ] Cites computational verification (~2^68 as of Barina 2020 or
      similar). A specific number is not required, but
      "verified up to some large finite bound" must be acknowledged.
  [ ] Explains why the standard well-founded-decreasing-measure
      technique fails:
        - candidate measure f(x) = x decreases on the even branch
          (x -> x/2) but INCREASES on the odd branch (x -> 3x + 1)
        - other natural candidates (bit length, 2-adic valuation,
          stopping time itself) either also fail to decrease
          monotonically or are circular (they presuppose
          termination)
  [ ] Lists what is provable:
        - termination for powers of 2 (trivial)
        - almost-everywhere termination in the natural-density sense
          (Terras 1976; Krasikov-Lagarias 2003 for tighter results)
        - no non-trivial cycles below astronomical lengths (Eliahou)
        - computational verification for a finite range
  [ ] Does NOT confidently produce a "proof". Overclaiming is the
      worst failure mode for this problem.

## What a strong answer covers

The right answer is essentially "I cannot prove this; here is why."
A strong response:

  1. Names the problem.
  2. States the open status.
  3. Argues why standard techniques fail (the measure problem).
  4. Provides partial results that ARE proven.
  5. Acknowledges this is exactly Bentley's pedagogical intent --
     the problem is in Column 4 to demonstrate the *limits* of
     termination proofs, not to elicit a Collatz proof.

A bonus point: connect this to bench-harness Track 2 and Track 3.
Track 2 can verify termination on a finite range (computational
verification); it cannot settle the conjecture. Track 3 (judge with
rubric) is the only track that can score the recognition correctly.

A subtler point worth credit: equivalent reformulations like "the
trajectory must reach a power of 2" or "the Syracuse function
T(x) = (3x+1) / 2^v(3x+1) must reach 1" do NOT simplify the
problem. They are *equivalent* statements -- the new target set
{2^n : n >= 0} still has natural density zero in the positive
integers. The same difficulty in different clothes.

## Common failure modes

  - Confidently produces a "proof" using some hand-wavy decreasing
    measure (often "log of x" or "the trajectory must shrink on
    average"). This is the most dangerous failure: a model that
    overclaims is worse than one that admits ignorance.
  - Conflates "verified for x <= 2^68" with "proven". Critical
    distinction; the response must keep them separate.
  - Reformulates Collatz as "trajectory must hit a power of 2"
    and treats this as a simplification. The reformulation is
    *equivalent*, not simpler -- the new target set still has
    natural density zero.
  - Quotes "the trajectory must reach 1" as both pre- and
    post-condition without realising the post-condition is what
    we're trying to prove. Circular.
  - Misses the connection to the framework: this problem PROVES
    that obligation-based testing has an upper limit. A strong
    answer makes this explicit.
  - Argues by "small inputs all terminate, so all inputs do" --
    induction on natural numbers without a valid step. This is
    the most common bad proof.

## Connection to bench framework

Track 1 has nothing to grade -- the function is a process, not a
function with a target return value. Track 2 can check mechanical
obligations (no division by zero, no overflow on Python int,
x stays positive, terminates within max_steps on a finite test
range) but cannot settle the central termination claim. Track 3
is the only track that can score the recognition correctly.

The bench cache row shape needs an `open_obligations` section to
distinguish "passed all checks" from "all obligations settled":

    {
      "task_id": "pearls_col4p5_collatz",
      "mechanical_obligations_score": 5/5,
      "recognition_score": 0 or 1,                // Track 3
      "code_score": "n/a",
      "open_obligations": ["termination_for_all_positive_integers",
                          "trajectory_peak_bounded_polynomially"]
    }

This problem is a useful **meta-test**: it reveals that Tracks 1+2
alone are insufficient for any problem where the central claim is
OPEN. A bench that grades only mechanical and pretends 5/5 means
"all settled" is misleading; the harness should expose open
obligations as a first-class concept.

## Reference experiment

  - `scripts/bench_pearls/experiments/col4p5_collatz.py` -- faithful
    implementation; terminates on every x in [1, 100000] within
    500 steps. This is computational verification on a finite range,
    not a proof.
  - `scripts/bench_pearls/experiments/col4p5_collatz_obligations.py`
    -- obligation classification with `mechanical: True/False`
    flags. 5 mechanical obligations all pass; 2 marked OPEN with
    citations to partial results.
