# Column 4, Problem 1 -- Binary search: no-runtime-errors proof

## Problem statement

Bentley's Column 4 walks through proving binary search functionally
correct via a loop invariant. Problem 1 asks: how do you prove the
program also has no runtime errors? Specifically, address each of:

  (a) division by zero,
  (b) numerical overflow,
  (c) variable values exceeding their declared range,
  (d) array index out of bounds.

Project constraints layered on top of the original problem: input is
a sorted array of positive integers, possibly billions of elements,
and copying the array is forbidden (O(1) extra space, no slicing).

## Rubric for grading

The model's answer should:

  [ ] Produce a binary search implementation that uses
      `mid = lo + (hi - lo) // 2` (or equivalent safe-midpoint form),
      not `(lo + hi) // 2`. Failure to use the safe form is a
      red flag for the overflow analysis below.
  [ ] State a loop invariant explicitly. The canonical form is:
      "if target is in arr, then target is in arr[lo..hi]".
  [ ] Prove initialization, maintenance, termination, and
      post-condition. All four are required for a Hoare-style proof.
  [ ] Address each of Bentley's four runtime-error classes:
       (a) Identify the only division and verify the divisor is
           non-zero by inspection.
       (b) Walk through every arithmetic expression and bound it
           under fixed-width 64-bit ints. The crucial step is the
           safe-midpoint argument.
       (c) Show that lo, hi, mid land in declared ranges. Note that
           hi can be transiently -1 in the closed-interval form,
           which requires a signed type.
       (d) Show that the only array access (arr[mid]) satisfies
           0 <= mid < n by combining the loop guard with the
           midpoint formula.
  [ ] Distinguish "Python: arbitrary precision, (b) is automatic"
      from "C/Java: fixed-width, (b) requires the safe midpoint."
      A strong answer notes both.
  [ ] Cite or describe the Bloch 2006 finding (Java
      Arrays.binarySearch overflow bug) when discussing the unsafe
      midpoint, or equivalent context that shows awareness this is
      a real-world bug, not academic.
  [ ] Avoid spurious validation -- no asserts on the precondition,
      no defensive copies, no error handling for impossible cases.

## What a strong answer covers

A concise, rigorous proof has six pieces:

1. The loop invariant (what doesn't change across iterations).
2. Initialization (the invariant holds at loop entry).
3. Maintenance (the invariant is preserved by each iteration --
   prove for both the `<` and `>` branches).
4. Termination (a measure strictly decreases; here `f = hi - lo + 1`).
5. Post-condition on exit (when lo > hi the invariant tells us
   target is not in arr; returning -1 is correct).
6. The runtime-error analysis, addressing each of (a)-(d).

The runtime-error analysis is the new content of Problem 1. Pieces
1-5 are Column 4's running example; (a)-(d) are the new claim.

The headline of (b) is the safe-midpoint argument. The naive
`(lo + hi) // 2` first computes the sum `lo + hi`, which overflows
at the 32-bit boundary once `n > 2^30 ~ 1.07 G` (Bloch 2006 in
Java's standard library). The safe form `lo + (hi - lo) // 2` cannot
overflow because `hi - lo` is non-negative and bounded by `len(arr)`,
and the sum is bounded above by `hi`.

## Common failure modes

  - Uses `(lo + hi) // 2` and either doesn't notice the overflow
    risk or claims "Python doesn't overflow" without acknowledging
    that the bug is a portability concern.
  - States a loop invariant but doesn't prove maintenance for both
    branches (`<` and `>`). Half a proof.
  - Confuses "no infinite loop" with "no runtime error". Termination
    is a separate concern; Bentley's four classes don't include it.
  - Forgets to handle the empty-array case (n = 0 means hi = -1
    initially, body never runs, no array access).
  - Adds runtime checks (`assert lo >= 0`, etc.) and treats them as
    proof. Asserts in code are a Track 2 mechanism; the Problem 1
    deliverable is an analytical argument, not instrumented code.
  - Confuses "the index mid is in range" with "the access expression
    uses mid". The proof must establish the former; the latter is
    a structural property of the code (see col4p10).

## Connection to bench framework

This is a Track 1 + Track 2 problem.

  - Track 1: run a hidden test against the model's binary_search.
    Pass/fail on functional correctness via oracle compare.
  - Track 2: instrument the model's binary_search with the
    obligation asserts derived from the proof (lo_in_range,
    hi_in_range, body_bounds, mid_in_window, mid_indexable,
    termination_measure_decreases, post_found, post_absent).
    Pass/fail on whether the asserts hold under stress.

The proof itself is a Track 3 deliverable -- the asserts derive from
the proof but the proof has additional content (the maintenance
argument, the post-condition argument, the citation to Bloch) that
needs a judge to grade.

## Reference experiment

  - `scripts/bench_pearls/experiments/col4p1_binary_search.py` --
    the implementation, with a 1M-element self-test.
  - `scripts/bench_pearls/experiments/col4p1_binary_search_obligations.py`
    -- the same code with 11 obligation asserts; passes 2478 mixed
    calls without falsifying any obligation.
