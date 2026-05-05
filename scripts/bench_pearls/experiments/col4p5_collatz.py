"""Programming Pearls Column 4, Problem 5.

The pseudocode given is::

    while x != 1 do
        if even(x)
            x = x / 2
        else
            x = 3 * x + 1

This is the **Collatz conjecture** (also: 3n+1 problem, Ulam conjecture,
Syracuse problem, Hasse's algorithm), posed by Lothar Collatz in 1937.

**Termination for arbitrary positive-integer inputs is OPEN.** No
proof exists. Computational verification has confirmed every positive
integer up to ~2^68 ~ 2.95 * 10^20 reaches 1 (Barina 2020 and later
work), but extrapolation from sample to population is not a proof.

The implementation below faithfully runs the algorithm. A bench-side
safety cap ``max_steps`` lets the test harness bound wall-clock time
in case some input would not terminate -- this is a property of the
test harness, not the algorithm. The original Bentley pseudocode has
no such cap.

Storage: O(1). Python ``int`` is arbitrary precision so trajectory
peaks (which can grow far above the input -- collatz(27) peaks at
9232) cannot overflow. No proven polynomial upper bound on peak
height as a function of input is known.
"""

from __future__ import annotations


def collatz(x: int, max_steps: int | None = None) -> int:
    """Run the Collatz iteration on ``x`` until it reaches 1; return
    the number of steps taken.

    Pre:
        - ``x`` is a positive integer (x >= 1).
        - ``max_steps`` is None (run until x = 1) or a positive int
          upper bound for safety in test harnesses.

    Post:
        - if the loop exits naturally, returns the step count.
        - if ``max_steps`` is hit first, raises RuntimeError.

    Whether the natural exit happens for every positive integer is
    the unproven Collatz conjecture. For the inputs the bench tests
    (computational verification on a finite range), termination has
    been observed; we make no claim beyond that range.
    """
    if x < 1:
        raise ValueError(f"x must be a positive integer, got {x}")
    steps = 0
    while x != 1:
        if max_steps is not None and steps >= max_steps:
            raise RuntimeError(
                f"exceeded max_steps={max_steps} (still at x={x} "
                f"after {steps} steps)"
            )
        if x % 2 == 0:
            x = x // 2
        else:
            x = 3 * x + 1
        steps += 1
    return steps


def _selftest() -> None:
    # Powers of two: trivially terminate in exactly k steps.
    for k in range(20):
        assert collatz(2 ** k) == k, k

    # Hand-verifiable short trajectories.
    # 3 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1
    assert collatz(3) == 7
    # 6 -> 3 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1
    assert collatz(6) == 8
    # 27 is the famous stubborn small input: 111 steps, peaks at 9232.
    assert collatz(27) == 111

    # Computational verification on a finite range. This is NOT a
    # proof of Collatz; it is evidence consistent with Collatz.
    # Max stopping time for x < 10^5 is 350 (achieved at x = 77031);
    # max_steps=500 leaves headroom and would surface any anomaly
    # quickly.
    n_max = 100_000
    for x in range(1, n_max + 1):
        steps = collatz(x, max_steps=500)
        assert steps >= 0

    print(f"OK -- all x in [1, {n_max}] terminate within 500 steps")
    print("     This is computational verification, NOT a proof.")
    print("     Termination for all positive integers is the Collatz")
    print("     conjecture (1937), an OPEN problem in mathematics.")


if __name__ == "__main__":
    _selftest()
