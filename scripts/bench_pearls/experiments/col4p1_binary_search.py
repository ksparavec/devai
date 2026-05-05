"""Programming Pearls Column 4, Problem 1.

Two deliverables:

  1. ``binary_search(arr, target)`` -- a correct implementation.
  2. A proof, in the chat response that ships with this file, that the
     program (a) always returns the right answer and (b) has no runtime
     errors in any of Bentley's four classes (division by zero,
     numerical overflow, variable values exceeding their declared
     range, array index out of bounds).

Storage-agnostic by design. ``arr`` need only support ``len()`` and
``arr[i]``, so it works equally on a Python ``list``, an
``array.array('q', ...)`` for 8-bytes-per-element packing, a numpy
``ndarray``, or an ``mmap``-backed buffer wrapped in a sequence
adapter. At billion-element scale, mmap over a sorted 8-byte int file
gives O(1) memory regardless of array size -- see the bottom of this
file for a 5-line example.
"""

from __future__ import annotations


def binary_search(arr, target: int) -> int:
    """Return the index of ``target`` in sorted ``arr``, or ``-1``.

    Pre:
        - ``arr`` is non-decreasing.
        - ``arr[i]`` and ``len(arr)`` are O(1).
        - ``target`` is comparable with elements of ``arr``.

    Post:
        - returns ``i`` with ``0 <= i < len(arr)`` and
          ``arr[i] == target``, OR
        - returns ``-1`` and ``target`` is not present in ``arr``.

    Duplicates: any matching index is legal (no leftmost/rightmost
    promise). Space: O(1). No slicing or copying.
    """
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        # Safe midpoint. ``(lo + hi) // 2`` overflows in fixed-width
        # signed-int languages once ``lo + hi`` exceeds INT64_MAX --
        # the bug Joshua Bloch found in Java's Arrays.binarySearch.
        # ``lo + (hi - lo) // 2`` cannot overflow: ``hi - lo`` is
        # non-negative and bounded by ``len(arr) - 1``, and the sum
        # is bounded above by ``hi`` itself.
        mid = lo + (hi - lo) // 2
        v = arr[mid]
        if v < target:
            lo = mid + 1
        elif v > target:
            hi = mid - 1
        else:
            return mid
    return -1


def _selftest() -> None:
    import random

    assert binary_search([], 5) == -1

    assert binary_search([7], 7) == 0
    assert binary_search([7], 6) == -1
    assert binary_search([7], 8) == -1

    assert binary_search([3, 5, 7, 9], 1) == -1
    assert binary_search([3, 5, 7, 9], 100) == -1

    arr = [1, 3, 5, 7, 9, 11, 13, 15]
    for i, v in enumerate(arr):
        assert binary_search(arr, v) == i

    # Duplicates: any matching index is legal.
    arr = [1, 2, 2, 2, 2, 3]
    i = binary_search(arr, 2)
    assert 1 <= i <= 4 and arr[i] == 2

    # Positive-only array; non-positive target should return -1.
    assert binary_search([1, 2, 3, 4, 5], 0) == -1
    assert binary_search([1, 2, 3, 4, 5], -7) == -1

    # 1 million element stress + 2000 mixed probes.
    rng = random.Random(1)
    n = 1_000_000
    arr = sorted(rng.sample(range(1, 10 * n), n))
    arr_set = set(arr)
    for _ in range(2_000):
        if rng.random() < 0.5:
            t = arr[rng.randrange(n)]
            i = binary_search(arr, t)
            assert i != -1 and arr[i] == t, t
        else:
            t = rng.randrange(1, 10 * n)
            while t in arr_set:
                t = rng.randrange(1, 10 * n)
            assert binary_search(arr, t) == -1, t

    print(f"OK -- {n}-element stress + edge cases all pass")


# ─── mmap example for billion-element arrays ────────────────────────────────
# A binary file of int64 values, sorted ascending, can back the search
# without ever loading more than one element at a time:
#
#   import mmap, struct
#   class Int64File:
#       def __init__(self, path):
#           self._f = open(path, "rb")
#           self._mm = mmap.mmap(self._f.fileno(), 0, prot=mmap.PROT_READ)
#       def __len__(self):
#           return len(self._mm) // 8
#       def __getitem__(self, i):
#           return struct.unpack_from("<q", self._mm, i * 8)[0]
#
#   binary_search(Int64File("sorted_ints.bin"), target)
#
# OS demand-paging brings in only the pages binary_search actually
# probes -- O(log n) pages, each 4 KiB. A 16 GB sorted file binary-
# searched this way touches ~30 pages, ~120 KiB of resident memory.


if __name__ == "__main__":
    _selftest()
