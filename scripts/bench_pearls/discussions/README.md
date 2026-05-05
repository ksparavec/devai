# Pearls discussions

One markdown document per Programming Pearls problem we have worked
through, used as the source-of-truth rubric for grading model
answers in the future bench harness (Track 3 / judge layer).

Each document follows the same template:

  - **Problem statement** -- Bentley's question, with any project-
    specific constraints we layered on top.
  - **Rubric for grading** -- checkable items the model's answer
    should hit.
  - **What a strong answer covers** -- the high-level shape of a
    correct response, including the key insight.
  - **Common failure modes** -- specific things to watch for in
    weak answers.
  - **Connection to bench framework** -- which track(s) apply, and
    what the problem teaches about the bench design.
  - **Reference experiment** -- pointer to the code under
    `scripts/bench_pearls/experiments/`.

## Index

| Problem | Topic | Tracks | Reference experiment |
|---|---|---|---|
| [col4p1](col4p1_binary_search_no_runtime_errors.md) | iterative binary search; no-runtime-errors proof | T1 + T2 + T3 | `experiments/col4p1_*.py` |
| [col4p3](col4p3_recursive_binary_search.md) | recursive binary search; same/diff vs iterative | T1 + T2 + T3 | `experiments/col4p3_*.py` |
| [col4p5](col4p5_collatz_termination.md) | 3n+1 termination (Collatz, OPEN) | T2 (partial) + T3 | `experiments/col4p5_*.py` |
| [col4p10](col4p10_mutation_testing.md) | mutation testing the binary-search grader | meta | `experiments/col4p10_*.py` |
| [col6p3](col6p3_float_vs_double.md) | float vs double speedup measurement | T4 (methodology) + T3 | `experiments/col6p3_*.py` |
| [col6p6](col6p6_efficiency_proverb.md) | the efficiency-vs-correctness proverb | T3 only | `experiments/col6p6_*.py` |

## Track legend

The four-track shape was discovered iteratively while working
through the problems above. The bench harness will need to route
each problem to the right track(s) when it goes live.

  - **T1 (functional correctness)** -- run the model's code against
    a hidden test, compare the return to an oracle. Pass/fail.
    Mechanical. Always trustworthy when applicable.
  - **T2 (proof obligations as runtime asserts)** -- instrument the
    model's code with the obligation asserts derived from the
    problem's rubric, run under stress, watch for `AssertionError`.
    Pass/fail. Mechanical. Catches "right answer via wrong
    invariant" but only when invariants can be expressed as
    runtime predicates.
  - **T3 (judge with rubric)** -- LLM-as-judge over the prose
    answer using the rubric in the problem's discussion document.
    Fuzzy. Necessary for arguments, recognitions, and proofs whose
    structure cannot be reduced to runtime asserts.
  - **T4 (methodology score)** -- mechanical checks on a measurement
    deliverable: did the submission warm up, repeat, report
    variance, span multiple workloads, name the system context?
    Counts checked items against a methodological rubric; no
    oracle on the numbers themselves.

## Why this lives next to the experiments

The discussion documents and the experiment scripts are
complementary, not redundant:

  - The **experiment** is the canonical implementation -- a
    self-test that the bench harness can run to verify Tracks 1
    and 2 mechanically.
  - The **discussion** is the rubric for everything the experiment
    cannot grade -- the prose argument, the recognition of an open
    problem, the methodology of a measurement, the awareness of
    counterexamples.

When the bench eventually goes live, the harness will:

  1. Read the discussion to determine which tracks apply.
  2. Run the experiment to grade T1 and T2 (where applicable).
  3. Hand the prose to the judge with the rubric for T3.
  4. Mechanically check T4 against the methodology rubric (where
     applicable).

The discussions are intentionally written to be both human-
readable (so a person can review them) and rubric-shaped (so a
judge can apply them deterministically).

## Adding new problems

When you pick a new Bentley problem, the workflow is:

  1. Write the implementation + obligations files under
     `experiments/colXpY_*.py`.
  2. Write the discussion under `discussions/colXpY_*.md` using the
     template above.
  3. Add a row to the index table.

The discussion document should be written *after* the experiment
files, because the rubric is sharper once you have working code
that lets you see what the model actually has to produce.
