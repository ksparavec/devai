"""Bentley Column 6 Problem 6 -- the proverb under scrutiny.

Two correct implementations of fib(n). Both return the same value. One
is O(n); the other is O(2^n). The recursive version's output is *right
on every input it returns on*, but for n >= ~40 it fails to return in
any practical time. Is it "correct"?

The proverb says: "if the program's output is wrong, speed is useless."
The recursive fib's output isn't wrong -- it's the *behaviour* that's
wrong. Calling code can't tell the difference.

Demonstration of the chat-thread argument that the proverb is right as
work-ordering advice but wrong as an absolute claim. Performance bugs
are correctness bugs once the spec includes "must return within a
reasonable time on realistic inputs."
"""

from __future__ import annotations

import time


def fib_iter(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def fib_rec(n: int) -> int:
    if n < 2:
        return n
    return fib_rec(n - 1) + fib_rec(n - 2)


def _demo() -> None:
    print(f"{'n':>4}  {'fib_iter (s)':>14}  {'fib_rec (s)':>14}  same?  value")
    for n in [10, 20, 30, 35]:
        t = time.perf_counter()
        a = fib_iter(n)
        ti = time.perf_counter() - t

        t = time.perf_counter()
        b = fib_rec(n)
        tr = time.perf_counter() - t

        ok = "yes" if a == b else "NO"
        print(f"{n:>4}  {ti:>14.6f}  {tr:>14.6f}  {ok:>5}  {a}")

    print()
    print("Both implementations satisfy 'correct output for inputs that")
    print("return'. fib_rec at n=40 needs ~30 s; n=50 needs hours; n=70")
    print("outlives the universe. The output is never 'wrong'; the")
    print("function simply fails to deliver one in any usable time.")
    print()
    print("Is fib_rec correct? The proverb's binary -- 'output right or")
    print("wrong' -- has no answer for this. The system-level binary --")
    print("'meets spec or doesn't' -- says clearly: no.")


if __name__ == "__main__":
    _demo()
