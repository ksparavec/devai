# Column 6, Problem 3 -- Float vs double speedup measurement

## Problem statement

Appel discovered that switching from double-precision to single-
precision arithmetic doubled the speed of his programs. Choose a
suitable test to measure this speedup on your computer system.

The deliverable is a **measurement**, not an algorithm. Grading is
by methodology and honesty, not by an oracle on outputs.

## Rubric for grading

  [ ] Tests **multiple workload classes**, not one number. At
      minimum:
        - memory-bound (vector add, reduction sum)
        - compute-bound (BLAS gemm / matmul)
        - mixed (dot product)
  [ ] Tests **multiple sizes per workload**, spanning cache
      regimes (in-L1, in-LLC, out-of-cache).
  [ ] Uses **identical inputs across dtypes**. Cast f64 base to
      f32 to eliminate input-distribution noise.
  [ ] Has a **warmup phase** (>=2 runs discarded). First-run
      effects (cold cache, BLAS thread pool init, page faults) can
      dominate small-N measurements.
  [ ] Reports more than just one timing per cell -- median +
      variance, or median + min + max. CV/stdev signals whether
      the result is trustworthy.
  [ ] Uses `time.perf_counter` or equivalent monotonic high-
      resolution timer. Not `time.time` for sub-second intervals.
  [ ] Reports system context: CPU model, BLAS variant if available,
      thread count, OMP_NUM_THREADS setting.
  [ ] **Acknowledges the precision tradeoff.** f32 has ~7 decimal
      digits; f64 has ~15. Speed alone is not a reason to switch.
  [ ] **Acknowledges hardware dependence.** Speedup varies with
      vector width (AVX2 vs AVX-512), memory bandwidth, BLAS choice.
      The number on this machine is not the number on your machine.
  [ ] Reports a number with context, not as an absolute fact.
      "On this machine, geomean 1.50x across this workload set,
      0.93x to 2.96x range" beats "f32 is 2x faster".

## What a strong answer covers

Appel's 2x figure is roughly right *for the workload class he wrote*
(numerical code, large memory footprint, vectorisable inner loops)
and is wrong as a universal claim. A suitable test exposes the
spread:

  - **Memory-bound at large sizes -> ~2x** (the headline result).
    Halving the bytes halves the time. This is what Appel was
    seeing.
  - **In-cache compute -> closer to 1.0x.** FPU throughput is
    similar for f32 and f64 on modern CPUs once data is in L1.
  - **BLAS-tuned matmul -> sub-2x; small matrices may even be
    slower.** Both DGEMM and SGEMM are aggressively optimized;
    halving the dtype doesn't halve the time.
  - **Reductions on large arrays -> often >2x.** Zero arithmetic
    intensity, pure bandwidth.

A strong answer reports the spread, names the regimes, and
concludes "the proverb of 2x is workload-dependent" rather than
confirming or denying it as a universal claim.

## Common failure modes

  - **Single workload** (e.g. matmul only) -> wrong conclusion.
    Matmul is the worst case for showing the speedup because BLAS
    gemm is aggressively tuned for both dtypes. A "matmul shows
    only 1.4x" report misleads about the proverb's general truth.
  - **Single size** -> hides cache effects. The speedup transitions
    sharply between in-cache and out-of-cache regimes.
  - **Pure-Python loops** -> sees no speedup at all because
    CPython's `PyFloat` is unconditionally a 64-bit double. Reports
    1.0x and concludes the proverb is wrong. Misses that the
    speedup lives at the C / BLAS / SIMD level.
  - **No warmup** -> first-run measurements dominated by cold
    cache, BLAS thread pool init, page faults. Ratios are noise.
  - **No variance reporting** -> a 2.96x result on a noisy 10K-
    element dot product looks identical to a real 2.0x on 10M
    elements. CV is the difference.
  - **Confuses speed with accuracy.** Single precision is ~7
    digits; double is ~15. Reductions accumulate roundoff visibly
    faster in f32. A "suitable test" mentions the tradeoff even if
    measuring only speed.
  - **Reports one number** ("2x") without the system context that
    determines reproducibility on other hardware. The geomean is
    not the speedup; it's the geomean across this workload set on
    this machine.

## Connection to bench framework

Track 1 (oracle compare) doesn't apply -- there is no right return
value, only a measurement. Track 2 (obligations) doesn't apply
either.

This problem requires a fourth track:

    Track 4 (methodology score): mechanical checks on the submitted
    benchmark for warmup, repeats, variance reporting, multiple
    workloads, multiple sizes, system context. Counts checked items
    against the rubric above; no oracle on the numbers themselves.

Plus optional Track 3 (judge): grade the submission's prose
interpretation -- did it acknowledge the precision tradeoff and
hardware dependence? Did it report the spread rather than a single
number?

This problem opens the door for an entire class of "measurement
deliverable" problems where the work is methodological and the
grader cannot rely on a return-value oracle. Bentley's columns
contain several others (any time he asks "how fast is X on your
machine"), and the bench design must accommodate them.

## Reference experiment

  - `scripts/bench_pearls/experiments/col6p3_float_vs_double.py` --
    4 workloads (vector_add, dot, matmul, sum), 3 sizes each, on
    Intel Core Ultra 9 285 with numpy 2.2.4.

Empirical result:

      geomean speedup:   1.50x
      min / max:         0.93x / 2.96x
      large-input geom:  1.51x

The 0.93x lower bound is `matmul` at n=128 (BLAS dispatch overhead).
The 2.96x upper bound is `dot` at n=10K (noise on a microsecond
measurement, CV 24.8%; should be discarded). The realistic
"Appel" cell is `vector_add` at n=10M which lands at 2.11x -- the
clean confirmation of the proverb on a memory-bound workload.
