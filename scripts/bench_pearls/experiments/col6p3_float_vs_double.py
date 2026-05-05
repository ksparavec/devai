"""Programming Pearls Column 6, Problem 3.

Appel observed (1980s-era) that switching from double to single
precision doubled his programs' speed. Design a "suitable test" to
measure this on the current system.

A *suitable test* is workload-aware. Appel's claim is not a universal
law -- the speedup depends on whether the workload is memory-bound,
compute-bound, or vectorizable. A single-number benchmark obscures
this. So this harness sweeps four workload classes at three sizes
each and reports the f64 -> f32 speedup per cell:

  - ``vector_add``   element-wise add: memory-bound at large sizes
  - ``dot``          dot product:       memory-bound at large sizes
  - ``matmul``       BLAS gemm:         compute-bound, BLAS uses SIMD
  - ``sum``          reduction:         memory-bound at large sizes

For each (workload, size, dtype) cell the test:
  1. Builds inputs in the requested dtype (float32 or float64).
  2. Runs the operation `warmup` times to populate cache / JIT BLAS
     thread pools.
  3. Times `repeats` measured runs and reports the median.
  4. Repeats for the other dtype on identical inputs.
  5. Computes speedup = median(f64) / median(f32).

The harness uses numpy (BLAS-backed) so the dtypes drive real C/Fortran
code paths. Pure-Python loops over Python floats see no float/double
distinction at all -- CPython boxes both as ``PyFloat_Type``, which is
itself a 64-bit double. Appel's observation lives at the C level.

Honest caveats reported alongside results:
  - Single precision gives DIFFERENT numerical results. Do not pick
    f32 just for speed if your numerics need 15 significant digits.
  - Speedup is system-dependent: CPU vector width, BLAS variant,
    memory bandwidth, cache size. The numbers below are this machine.
  - BLAS is multithreaded by default. Thread count affects matmul
    most; we report it but don't pin it.
"""

from __future__ import annotations

import os
import platform
import statistics
import time
from typing import Callable

import numpy as np


def time_call(fn: Callable, *, repeats: int = 7, warmup: int = 2) -> dict:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return {
        "min":    min(samples),
        "median": statistics.median(samples),
        "max":    max(samples),
        "stdev":  statistics.stdev(samples) if repeats > 1 else 0.0,
        "n":      repeats,
    }


# === Workloads ===============================================================

def w_vector_add(n: int, dtype) -> Callable:
    rng = np.random.default_rng(0)
    a = rng.random(n, dtype=np.float64).astype(dtype)
    b = rng.random(n, dtype=np.float64).astype(dtype)
    c = np.empty(n, dtype=dtype)

    def run():
        np.add(a, b, out=c)
    return run


def w_dot(n: int, dtype) -> Callable:
    rng = np.random.default_rng(1)
    a = rng.random(n, dtype=np.float64).astype(dtype)
    b = rng.random(n, dtype=np.float64).astype(dtype)

    def run():
        np.dot(a, b)
    return run


def w_matmul(n: int, dtype) -> Callable:
    rng = np.random.default_rng(2)
    a = rng.random((n, n), dtype=np.float64).astype(dtype)
    b = rng.random((n, n), dtype=np.float64).astype(dtype)

    def run():
        a @ b
    return run


def w_sum(n: int, dtype) -> Callable:
    rng = np.random.default_rng(3)
    a = rng.random(n, dtype=np.float64).astype(dtype)

    def run():
        a.sum()
    return run


WORKLOADS = [
    ("vector_add", w_vector_add, [10_000, 1_000_000, 10_000_000]),
    ("dot",        w_dot,        [10_000, 1_000_000, 10_000_000]),
    ("matmul",     w_matmul,     [128, 512, 1024]),
    ("sum",        w_sum,        [10_000, 1_000_000, 10_000_000]),
]


# === Driver ==================================================================

def _system_info() -> dict:
    cpu = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    threads = os.cpu_count() or 0
    omp = os.environ.get("OMP_NUM_THREADS", "<unset; BLAS picks>")
    return {
        "cpu": cpu or "unknown",
        "platform": platform.platform(),
        "numpy": np.__version__,
        "threads_available": threads,
        "omp_num_threads": omp,
    }


def main() -> None:
    info = _system_info()
    print("System under test:")
    for k, v in info.items():
        print(f"  {k:<22} {v}")

    print("\nMethodology: 2 warmup runs, 7 measured runs, report median;")
    print("speedup = median(f64) / median(f32). Inputs identical between "
          "dtypes (cast from the same f64 base).\n")

    fmt = (
        "{wl:<12} {n:>12}  "
        "{f64_med:>10}  {f64_stdev:>8}  "
        "{f32_med:>10}  {f32_stdev:>8}  "
        "{speedup:>9}"
    )
    print(fmt.format(
        wl="workload", n="n",
        f64_med="f64 ms", f64_stdev="+/-  %",
        f32_med="f32 ms", f32_stdev="+/-  %",
        speedup="speedup",
    ))
    print("-" * 88)

    speedups: list[tuple[str, int, float]] = []

    for name, builder, sizes in WORKLOADS:
        for n in sizes:
            r64 = time_call(builder(n, np.float64))
            r32 = time_call(builder(n, np.float32))
            sp = r64["median"] / r32["median"]
            speedups.append((name, n, sp))
            cv64 = (r64["stdev"] / r64["median"] * 100) if r64["median"] else 0
            cv32 = (r32["stdev"] / r32["median"] * 100) if r32["median"] else 0
            print(fmt.format(
                wl=name, n=f"{n:,}",
                f64_med=f"{r64['median']*1000:.3f}",
                f64_stdev=f"{cv64:5.1f}",
                f32_med=f"{r32['median']*1000:.3f}",
                f32_stdev=f"{cv32:5.1f}",
                speedup=f"{sp:.2f}x",
            ))
        print()

    print("Summary:")
    if speedups:
        ratios = [s for _, _, s in speedups]
        print(f"  geomean speedup:   {statistics.geometric_mean(ratios):.2f}x")
        print(f"  min / max:         {min(ratios):.2f}x / {max(ratios):.2f}x")
        big = [s for w, n, s in speedups if n >= 1_000_000 or
               (w == 'matmul' and n >= 512)]
        if big:
            print(f"  large-input geom:  {statistics.geometric_mean(big):.2f}x  "
                  f"(memory- or compute-pressure regime)")

    print("\nCaveats:")
    print("  - Speedup varies sharply across (workload, size). 2x is the")
    print("    rule of thumb for memory-bound and SIMD-vectorizable code,")
    print("    not a universal constant. Small in-cache workloads benefit")
    print("    less; some BLAS gemm paths are similarly tuned for both")
    print("    dtypes and show <2x.")
    print("  - Numerical accuracy differs. f32 has ~7 decimal digits; f64")
    print("    has ~15. Reductions over large arrays accumulate roundoff")
    print("    much faster in f32. Speed alone is not a reason to switch.")
    print("  - Pure-Python loops do NOT see this distinction -- CPython's")
    print("    PyFloat is a 64-bit double regardless. The speedup lives at")
    print("    the BLAS / C / SIMD level.")


if __name__ == "__main__":
    main()
