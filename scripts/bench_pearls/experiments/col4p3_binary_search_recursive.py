"""Programming Pearls Column 4, Problem 3.

Recursive binary search. Same contract as the iterative version in
``col4p1_binary_search.py``: sorted positive integers, billions of
elements, no slicing or copying. Storage-agnostic -- only ``len(arr)``
and ``arr[i]`` are used.

The public entry point ``binary_search_recursive(arr, target)`` opens a
private helper ``_bsearch(arr, target, lo, hi)`` that recurses on the
window ``[lo, hi]``. The helper is what carries the inductive
hypothesis; keeping it private means the public contract stays
identical to the iterative function (callers don't pass indices).

For an n-element array, recursion depth is at most ceil(log2(n + 1))
because each non-base call halves the window. n = 10**9 -> depth 30,
well below CPython's default recursion limit of 1000. So no
sys.setrecursionlimit nudge is needed for any array size that fits in
64-bit indices.
"""

from __future__ import annotations


def binary_search_recursive(arr, target: int) -> int:
    """Return the index of ``target`` in sorted ``arr``, or ``-1``.

    Identical pre/post conditions to the iterative ``binary_search``:

    Pre:
        - ``arr`` is non-decreasing.
        - ``arr[i]`` and ``len(arr)`` are O(1).
        - ``target`` is comparable with elements of ``arr``.

    Post:
        - returns ``i`` with ``0 <= i < len(arr)`` and
          ``arr[i] == target``, OR
        - returns ``-1`` and ``target`` is not present in ``arr``.

    Duplicates: any matching index is legal. Space: O(log n) for the
    recursion stack -- ~30 frames at billion-element scale, ~6-15 KiB
    of CPython stack memory; effectively O(1).
    """
    return _bsearch(arr, target, 0, len(arr) - 1)


def _bsearch(arr, target: int, lo: int, hi: int) -> int:
    """Recurse on the closed window ``arr[lo..hi]``.

    Inductive hypothesis (the recursive analogue of the iterative
    loop invariant): if ``target`` appears in ``arr``, then it appears
    at some index in ``[lo, hi]``. Established by the caller; the
    function preserves it across each recursive descent.
    """
    if lo > hi:
        return -1
    # Same safe midpoint as the iterative version. Same overflow
    # analysis carries verbatim.
    mid = lo + (hi - lo) // 2
    v = arr[mid]
    if v < target:
        return _bsearch(arr, target, mid + 1, hi)
    elif v > target:
        return _bsearch(arr, target, lo, mid - 1)
    else:
        return mid


def _selftest() -> None:
    import random

    assert binary_search_recursive([], 5) == -1

    assert binary_search_recursive([7], 7) == 0
    assert binary_search_recursive([7], 6) == -1
    assert binary_search_recursive([7], 8) == -1

    assert binary_search_recursive([3, 5, 7, 9], 1) == -1
    assert binary_search_recursive([3, 5, 7, 9], 100) == -1

    arr = [1, 3, 5, 7, 9, 11, 13, 15]
    for i, v in enumerate(arr):
        assert binary_search_recursive(arr, v) == i

    arr = [1, 2, 2, 2, 2, 3]
    i = binary_search_recursive(arr, 2)
    assert 1 <= i <= 4 and arr[i] == 2

    assert binary_search_recursive([1, 2, 3, 4, 5], 0) == -1
    assert binary_search_recursive([1, 2, 3, 4, 5], -7) == -1

    # Differential against the iterative version.
    from col4p1_binary_search import binary_search as bsearch_iter

    rng = random.Random(1)
    n = 1_000_000
    arr = sorted(rng.sample(range(1, 10 * n), n))
    arr_set = set(arr)
    for _ in range(2_000):
        if rng.random() < 0.5:
            t = arr[rng.randrange(n)]
        else:
            t = rng.randrange(1, 10 * n)
            while t in arr_set:
                t = rng.randrange(1, 10 * n)
        a = binary_search_recursive(arr, t)
        b = bsearch_iter(arr, t)
        # Both must return -1 or a valid matching index. They need
        # not return the SAME index when target is duplicated, but
        # for unique-element inputs they should agree exactly.
        if a == -1:
            assert b == -1, t
        else:
            assert b != -1 and arr[a] == arr[b] == t, t

    print(f"OK -- {n}-element stress + edge cases all pass; "
          f"recursive matches iterative")


if __name__ == "__main__":
    _selftest()
