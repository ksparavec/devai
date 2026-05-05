# Column 4, Problem 10 -- Mutation testing of binary search

## Problem statement

Introduce errors into the binary search function and observe whether
(and how) these errors are caught when verifying the code.

This is a meta-problem: the deliverable is a study of the **grader**,
not a new algorithm. It probes the limits of the bench framework
itself.

## Rubric for grading

  [ ] Implements multiple distinct mutations (>=5), each illustrating
      a different bug class:
        - bounds (off-by-one in init or update of lo/hi)
        - termination (no advance, wrong direction)
        - control flow (loop guard wrong)
        - return value (off-by-one or missing)
        - access expression (wrong variable read)
        - overflow / midpoint (the Bloch bug)
  [ ] Tests each mutation under at least two distinct grading
      regimes: functional correctness (Track 1) vs. internal
      assertions (Track 2).
  [ ] Reports the **four-cell complementarity matrix**:
        - caught by both
        - caught by Track 1 only
        - caught by Track 2 only
        - caught by neither
  [ ] Identifies and discusses the "neither" cell. The naive midpoint
      `(lo + hi) // 2` is the canonical inhabitant: in Python (or any
      arbitrary-precision-int language) the bug is invisible because
      no overflow occurs. Static analysis or fixed-width execution is
      required to catch it.
  [ ] Notes that empirically the "Track 2 only" cell is hard to
      populate from code mutations. Mutations naturally produce
      wrong outputs (Track 1 catches) or violate internals (Track 2
      catches alongside Track 1). A "right output via wrong
      invariant" mutation is contrived; this cell's real value is
      for grading PROOFS, not code.
  [ ] Identifies blind spots in the obligation set:
        - early-return short-circuits asserts (e.g., `if lo >= hi:
          return -1` followed by no further checks before return)
        - the asserts speak to state, not control flow (a bug in
          the access expression like `arr[lo]` instead of `arr[mid]`
          slips through if lo happens to be in range)
        - asserts at the top of the loop don't speak to what's
          returned (e.g., `return mid + 1` is invisible to internal
          checks)
  [ ] Acknowledges sensitivity to the test battery. Track 2 can
      catch a bug *earlier* than Track 1 if the relevant input is
      tested first; reordering the battery changes apparent
      "first failure" without changing the underlying detection.

## What a strong answer covers

The honest finding is that Tracks 1 and 2 are **complementary but
unequal**:

  - Track 1 is the stronger functional filter for code grading.
    Most code mutations produce wrong outputs detectable by oracle
    compare.
  - Track 2's separate value is in **proof grading**. When the model
    submits both code and a proof, Track 2 checks whether the
    proof's claimed invariants actually hold during execution -- a
    check Track 1 cannot do.
  - The "neither" cell exists. The bench should not pretend
    9/9 Track 1 + 9/9 Track 2 means "all bugs caught" -- the
    overflow class is a documented blind spot.

A strong answer reports the four-cell matrix, explains which cell
each mutation lands in and why, and uses the "neither" cell to
motivate the addition of a Track 4 (static analysis / linter) for
patterns like `(a + b) / 2` that runtime testing cannot expose.

## Common failure modes

  - Tests only one mutation type (e.g. only off-by-one) and reports
    "Track 2 catches everything". Single-class testing produces
    misleading dominance.
  - Conflates Track 1 (oracle compare) with Track 2 (internal
    asserts). They are different filters; reporting one combined
    pass/fail loses the diagnostic value.
  - Misses the overflow-in-Python blind spot. A strong answer
    explicitly tests the naive midpoint and notes that Python's
    arbitrary precision masks the bug.
  - Reports counts without interpreting them. The four-cell
    decomposition is the answer, not the totals.
  - Constructs mutations that Track 2 catches and Track 1 misses
    via test-battery gerrymandering (e.g. test only inputs where
    the bug doesn't manifest functionally). This is misleading
    -- a fair test battery would catch via Track 1 too.

## Connection to bench framework

This problem IS the connection. It tells the bench designer:

  1. **Run Tracks 1 and 2 separately, report separately.** Combined
     scores destroy the diagnostic value.
  2. **Add a Track 4** (static analysis / linter) for the overflow
     class. Patterns like `(a + b) / 2` can be flagged without
     execution. This is mechanical and cheap.
  3. **Track 2's value emerges most clearly in proof grading**, not
     code grading. Mutation testing under-represents Track 2's
     contribution because mutations are derived from buggy code,
     and buggy code usually produces wrong outputs.
  4. The obligation set is incomplete by design. Bugs in early
     control flow, return values, and access expressions slip
     through. Track 1 closes most of those gaps; Track 4 closes
     the overflow gap. Trying to make Track 2 complete via more
     elaborate asserts hits diminishing returns quickly.

## Reference experiment

  - `scripts/bench_pearls/experiments/col4p10_mutation_harness.py`
    -- 9 mutations + baseline, both Track 1 and Track 2 runners,
    automatic four-cell decomposition.

Empirical result on this codebase (Intel Core Ultra 9 285, Python
3.13, numpy 2.2.4):

      caught by both tracks:   4    (M1, M2, M5, M8)
      caught by Track 1 only:  4    (M3, M6, M7, M9)
      caught by Track 2 only:  0
      CAUGHT BY NEITHER:       1    (M4 -- naive midpoint)

The "Track 2 only" empty cell is the most informative result. It
confirms that mutation testing on code under-represents Track 2's
value, and shifts the case for Track 2 toward proof grading.
