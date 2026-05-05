# Column 6, Problem 6 -- The efficiency-vs-correctness proverb

## Problem statement

The proverb: "Efficiency always comes after correctness -- if the
program's output is wrong, speed is useless." Is this statement
correct?

The right answer pushes back. Bentley places this in Column 6
("Perspective on Performance"), which argues that performance is a
first-class design concern, not an afterthought. The problem is
intended to elicit critical engagement with the proverb, not
agreement.

## Rubric for grading

  [ ] **Distinguishes the two readings of the proverb**:
        - work-ordering advice (mostly correct)
        - absolute claim about value (mostly wrong / misleading)
  [ ] Provides counterexamples in **at least two of the three
      classes**:
        - specifications that include timing (real-time systems,
          network protocols, UI responsiveness, trading)
        - problems too large for exact answers (numerical PDEs,
          Monte Carlo, ML inference, web search ranking)
        - performance bugs are correctness bugs (asymptotic
          blowup, memory leaks, garbage collector pauses, denial
          of service)
  [ ] Articulates the **reformulation**: "correctness" properly
      read includes the full functional spec (latency, throughput,
      accuracy, behaviour at scale). Once you broaden the term,
      the proverb becomes tautological and uninformative.
  [ ] Avoids the trap of either:
        - blindly agreeing (correctness first, period)
        - blindly disagreeing (the proverb is wrong, period)
      The right answer holds both halves and names the conditions
      under which each holds.
  [ ] Bonus: provides a **concrete demo** showing the gap. The
      canonical demo is two correct implementations of the same
      function (e.g. fib(n)) where one is exponential and one is
      linear -- both produce the right value but the exponential
      one is unusable for any realistic input. Demonstrates that
      "correct output" is not "correct".

## What a strong answer covers

The proverb is **right** when:
  - correctness is a clean binary spec
  - the latency / throughput / size budget is loose
  - speed can be bought elsewhere (more hardware, more time)

The proverb is **wrong / misleading** when:
  - the spec includes timing
  - the problem has no exact solution within budget
  - asymptotic behaviour is part of the contract

The deeper truth: **performance is a correctness property of any
system whose spec includes timing, scale, or approximation
tolerance.** For those systems -- which is most production systems
-- there is no "first correctness, then speed". They are the same
axis. The proverb's framing assumes a clean correct/wrong binary on
*outputs* that often does not exist.

A concrete demo: two correct implementations of `fib(n)`, one O(n)
iterative and one O(2^n) recursive. Both return the same value on
every input that returns. The recursive version is unusable for
n >= ~40. By the proverb's reading -- "is the output right?" -- it
is correct. By the spec's reading -- "does the function deliver an
answer in usable time on realistic inputs?" -- it is not.

## Common failure modes

  - **Agrees with the proverb without engaging.** Misses Bentley's
    pedagogical intent. Cites Knuth's "premature optimization is
    the root of all evil" as if it settles the question. Knuth's
    quote is the strongest form of the work-ordering advice; it
    doesn't address the absolute claim.
  - **Disagrees with the proverb without giving counterexamples.**
    The rubric requires concrete cases, not handwaving.
  - **Misses the "performance bug = correctness bug" angle.** This
    is the most modern of the three counterexample classes and the
    most relevant for production systems. A strong answer leads
    with this.
  - **Treats the proverb as an empirical claim** ("most of the
    time speed doesn't matter"). It's a definitional / conceptual
    claim about the relationship between correctness and speed.
    Empirical frequency is irrelevant.
  - **Lists examples without naming the underlying class.** A list
    of 10 cases without the abstraction "specs that include timing"
    is less useful than 3 cases per class with the class named.

## Connection to bench framework

Track 1 doesn't apply (no function to write). Track 2 doesn't apply
(no obligations to assert). This is purely **Track 3 territory**: a
judge with a rubric that scores the submission's reasoning quality.

The bench design implication: the framework needs Track 3 not just
as an optional add-on for proof grading, but as a **primary grader
for essay / argument problems**. Bentley's columns include several
of these:

  - col1: "what data structure would you choose for X" (design)
  - col2: "what algorithmic strategy applies to Y" (analysis)
  - col6: this problem (criticism of received wisdom)

A bench that grades only function outputs and obligations cannot
score these problems at all. The bench framework should expose
"argument" as a first-class problem category alongside "function
with right answer", "function with proof", and "measurement
deliverable".

## Reference experiment

  - `scripts/bench_pearls/experiments/col6p6_proverb_demo.py` --
    two `fib(n)` implementations, one O(n) and one O(2^n). Both
    correct on returning inputs; the recursive one fails to return
    for n >= ~40 in any practical time. 30 lines, makes the gap
    concrete.

Sample output (Intel Core Ultra 9 285, Python 3.13):

       n   fib_iter (s)    fib_rec (s)   same?  value
      10       0.000002        0.000005    yes  55
      20       0.000001        0.000498    yes  6765
      30       0.000001        0.055913    yes  832040
      35       0.000002        0.506306    yes  9227465

By n=35 the recursive form is 250000x slower. Output: identical.
Behaviour for any realistic n: useless. The proverb has no
vocabulary for this case.
