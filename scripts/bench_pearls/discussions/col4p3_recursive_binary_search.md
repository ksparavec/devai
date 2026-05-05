# Column 4, Problem 3 -- Recursive binary search; same/different vs iterative

## Problem statement

Write and verify a recursive binary search. Identify which parts of
the code and proof are the same as the iterative version (Problem 1),
and which have changed.

## Rubric for grading

  [ ] Recursive implementation with the same public contract as
      Problem 1's iterative version (same signature, same pre/post
      conditions, same return convention).
  [ ] Helper function carries lo and hi as parameters; no slicing
      of arr. Storage must remain O(1) extra (recursion stack
      excluded; bounded at log n frames).
  [ ] Uses the same safe midpoint as the iterative version
      (`lo + (hi - lo) // 2`).
  [ ] Acknowledges Python recursion limit context: depth is O(log n);
      ~30 frames at billion-element scale, well below the default
      sys.recursionlimit of 1000. So no setrecursionlimit nudge is
      needed, but only because the halving bound is tight.
  [ ] Reuses the iterative proof's correctness machinery, recast as
      induction:
        - loop invariant becomes inductive hypothesis on the window
        - maintenance step becomes precondition discharge for the
          recursive call
        - termination measure remains `f = hi - lo + 1`, now bounding
          recursion depth instead of loop iterations
  [ ] Notes that Bentley's four runtime-error arguments
      ((a) div-by-zero, (b) overflow, (c) declared range, (d) OOB)
      carry verbatim. Same arithmetic, same locations, same bounds.
  [ ] Identifies the new obligation specific to recursion:
      `recursion_depth_bounded`, with the bound `ceil(log2(n+1))`
      from the halving argument. This obligation has no iterative
      analogue.
  [ ] Frames the comparison precisely: NOT "everything different",
      NOT "everything the same", but a small list of specific
      changes against a backdrop of structural identity.

## What a strong answer covers

The headline insight is that the proof structure is mostly invariant
under iterative -> recursive translation. The translation is a
reframing, not a rewrite.

Same:
  - The arithmetic (mid formula, safe midpoint).
  - All four Bentley runtime-error arguments verbatim.
  - The window invariant (target in arr -> target in arr[lo..hi]).
  - The termination measure expression.
  - Storage agnosticism.
  - The three-way branch on `arr[mid]`.

Changed:
  - Loop invariant becomes inductive hypothesis (same proposition,
    different framing -- iterative proves it for all iterations
    via init+maintenance; recursive assumes it for smaller windows
    and proves it for the call).
  - Maintenance becomes precondition discharge for the recursive
    call (same algebra, different audience).
  - Termination measure now bounds recursion depth instead of loop
    iterations. This matters for safety because deep recursion
    eventually overflows the stack -- a runtime error not in
    Bentley's original four classes but worth flagging.
  - New obligation `recursion_depth_bounded` with no iterative
    analogue. Iterative code has no stack frames per iteration.
  - Loop guard `while lo <= hi` becomes base case
    `if lo > hi: return -1`.

## Common failure modes

  - Slices arr at recursive calls (`return _bsearch(arr[mid+1:], ...)`).
    Violates the no-copying constraint and silently changes the
    complexity from O(log n) to O(n log n). At billion-element
    scale this is a fatal bug.
  - Re-derives the entire proof from scratch instead of pointing
    to which pieces transfer. The point of Problem 3 is to surface
    the structure-preserving translation, not to repeat the work.
  - Misses recursion depth as a new safety obligation. Defaults to
    "Python handles it" without naming the implicit log-n bound.
  - Confuses "tail recursion" with "iteration". Python doesn't
    optimize tail calls; the recursive form pays a stack frame per
    level. At billion-element scale this is still ~30 frames, not
    a problem -- but only because the halving bound is tight.
  - Lists more changes than there really are. A model that
    over-reports differences is signaling that it doesn't see the
    underlying invariance.

## Connection to bench framework

Same as col4p1: Track 1 + Track 2 + (optional) Track 3 for the
prose comparison. The obligation set adds one entry
(`recursion_depth_bounded`) and restates one
(`termination_measure_decreases` now checked across call frames
instead of loop iterations). The other obligations carry verbatim.

This problem is also a useful **Track 2 robustness check** -- if a
model's iterative obligation set works on the recursive translation
with only the two changes named above, that's evidence the
obligation set is the right level of abstraction.

## Reference experiment

  - `scripts/bench_pearls/experiments/col4p3_binary_search_recursive.py`
    -- the implementation, with 1M-element self-test plus a
    differential check against the iterative col4p1 version.
  - `scripts/bench_pearls/experiments/col4p3_binary_search_recursive_obligations.py`
    -- 12 obligation asserts (10 unchanged from iterative,
    1 restated for recursion, 1 new for depth). Passes 2478 mixed
    calls without falsifying any obligation.
