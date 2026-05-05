#!/usr/bin/env python3
"""Source-of-truth for the Programming Pearls problem set.

Edit the ``PROBLEMS`` list below, then run this script to refresh
``pearls_problems.jsonl``. The JSONL is the runtime artifact loaded by
``tasks/pearls.py``; this Python module is what you edit by hand.

Why two files: the bench loader (mirroring ``tools_use.py``) reads JSONL
because that's the existing convention. JSONL forces escaping of
multi-line code strings, which is unreadable for problems whose prompts
are entire docstrings. So we keep the human-edit form here and produce
the JSONL mechanically.

Each problem dict has the same shape as a HumanEval row plus two extras
for traceability:

    task_id          -- unique identifier (used as cache key suffix)
    column           -- source column from Bentley's Programming Pearls
    difficulty       -- "easy" | "medium" | "hard" -- rough estimate
    prompt           -- function signature + docstring (what the model sees)
    entry_point      -- name of the function the model must define
    canonical_solution -- a known-good implementation; not sent to the
                          model, just used by the dry-run verifier in
                          tests/test_pearls_problems.py to confirm the
                          test asserts are satisfiable in <10s under the
                          subprocess sandbox
    test             -- code that defines ``check(candidate)``; the
                        scorer concatenates ``prompt + completion + test
                        + check(<entry_point>)`` and runs it

Citations refer to *Programming Pearls* (Bentley, 2nd ed., 2000). The
column numbers below are page-stable across the 1st and 2nd editions.
"""

from __future__ import annotations

import json
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Problem set
# ─────────────────────────────────────────────────────────────────────────────

PROBLEMS: list[dict] = []

# ─── Column 1 ─ Bitsort ──────────────────────────────────────────────────────
PROBLEMS.append({
    "task_id": "pearls_col1_bitsort",
    "column": 1,
    "difficulty": "easy",
    "entry_point": "bitsort",
    "prompt": '''from typing import List


def bitsort(values: List[int], universe: int) -> List[int]:
    """Sort ``values`` ascending using a bitset over ``[0, universe)``.

    Programming Pearls Column 1 in miniature: instead of an in-memory
    comparison sort, set bit ``v`` for each ``v`` in ``values``, then
    read the bits out in ascending order. All elements are unique and
    in ``[0, universe)``.

    Args:
        values: list of unique non-negative ints, each < universe.
        universe: exclusive upper bound on element values; >= 0.

    Returns:
        A new list with the same elements in ascending order. Empty
        input returns the empty list.
    """
''',
    "canonical_solution": '''    if universe <= 0 or not values:
        return []
    bitmap = 0
    for v in values:
        bitmap |= 1 << v
    out: List[int] = []
    i = 0
    while bitmap:
        if bitmap & 1:
            out.append(i)
        bitmap >>= 1
        i += 1
    return out
''',
    "test": '''def check(candidate):
    import random
    assert candidate([], 100) == []
    assert candidate([0], 1) == [0]
    assert candidate([5, 2, 7, 1, 9], 10) == [1, 2, 5, 7, 9]
    rng = random.Random(0xBEE)
    sample = rng.sample(range(1000), 200)
    assert candidate(sample, 1000) == sorted(sample)
    sample = rng.sample(range(10000), 1000)
    assert candidate(sample, 10000) == sorted(sample)
''',
})

# ─── Column 2 ─ Vector rotation (Aha! Algorithms) ────────────────────────────
PROBLEMS.append({
    "task_id": "pearls_col2_rotate",
    "column": 2,
    "difficulty": "easy",
    "entry_point": "rotate_left",
    "prompt": '''from typing import List


def rotate_left(items: List[int], i: int) -> List[int]:
    """Rotate ``items`` left by ``i`` positions and return a new list.

    Programming Pearls Column 2 ('Aha! Algorithms') uses vector
    rotation as one of its three motivating problems. After rotation,
    ``new[0]`` is the element originally at index ``i mod len(items)``.

    Args:
        items: list to rotate; the input is not mutated.
        i: shift amount; can be negative or larger than ``len(items)``.

    Returns:
        A new list where ``new[k] == items[(k + i) mod len(items)]``.
        Empty input returns the empty list.
    """
''',
    "canonical_solution": '''    n = len(items)
    if n == 0:
        return []
    s = i % n
    return items[s:] + items[:s]
''',
    "test": '''def check(candidate):
    assert candidate([], 0) == []
    assert candidate([], 5) == []
    assert candidate([1, 2, 3, 4, 5], 0) == [1, 2, 3, 4, 5]
    assert candidate([1, 2, 3, 4, 5], 1) == [2, 3, 4, 5, 1]
    assert candidate([1, 2, 3, 4, 5], 2) == [3, 4, 5, 1, 2]
    assert candidate([1, 2, 3, 4, 5], 5) == [1, 2, 3, 4, 5]
    assert candidate([1, 2, 3, 4, 5], 7) == [3, 4, 5, 1, 2]
    assert candidate([1, 2, 3, 4, 5], -1) == [5, 1, 2, 3, 4]
    assert candidate([1, 2, 3, 4, 5], -7) == [4, 5, 1, 2, 3]
    big = list(range(100))
    assert candidate(big, 33) == big[33:] + big[:33]
    assert candidate(big, 0) == big
    assert candidate(big, 100) == big
''',
})

# ─── Column 2 ─ Anagram grouping ─────────────────────────────────────────────
PROBLEMS.append({
    "task_id": "pearls_col2_anagram_groups",
    "column": 2,
    "difficulty": "easy",
    "entry_point": "anagram_groups",
    "prompt": '''from typing import List


def anagram_groups(words: List[str]) -> List[List[str]]:
    """Group anagrams from ``words`` together.

    Programming Pearls Column 2 sketches the idea: each word's
    "signature" is its sorted-letter string; words sharing a signature
    are mutual anagrams.

    Args:
        words: list of lowercase ASCII words; may contain duplicates.

    Returns:
        A list of groups. Each group is a list of words that are mutual
        anagrams, sorted ascending within the group. The outer list is
        sorted by the group's first element. Single-member groups are
        included.
    """
''',
    "canonical_solution": '''    buckets: dict[str, list[str]] = {}
    for w in words:
        sig = "".join(sorted(w))
        buckets.setdefault(sig, []).append(w)
    out = [sorted(g) for g in buckets.values()]
    out.sort(key=lambda g: g[0])
    return out
''',
    "test": '''def check(candidate):
    assert candidate([]) == []
    assert candidate(["abc"]) == [["abc"]]
    out = candidate(["bat", "tab", "cat", "act", "tac", "dog"])
    assert out == [["act", "cat", "tac"], ["bat", "tab"], ["dog"]]
    out = candidate(["a", "b", "a"])
    assert out == [["a", "a"], ["b"]]
    out = candidate(["listen", "silent", "enlist", "google", "gooegl"])
    assert out == [["enlist", "listen", "silent"], ["gooegl", "google"]]
    # equal-length non-anagram words remain in separate groups
    out = candidate(["cat", "dog"])
    assert out == [["cat"], ["dog"]]
''',
})

# ─── Column 4 ─ Verified binary search ───────────────────────────────────────
PROBLEMS.append({
    "task_id": "pearls_col4_binary_search_leftmost",
    "column": 4,
    "difficulty": "medium",
    "entry_point": "binary_search_leftmost",
    "prompt": '''from typing import List


def binary_search_leftmost(sorted_items: List[int], target: int) -> int:
    """Return the leftmost index where ``target`` could be inserted into
    ``sorted_items`` to keep it sorted.

    Equivalent to ``bisect.bisect_left``. Programming Pearls Column 4
    uses this exact problem to walk through proving an iterative loop
    correct via invariants. Implement iteratively; do not call the
    ``bisect`` module.

    Args:
        sorted_items: ascending list (may contain duplicates).
        target: value to locate.

    Returns:
        Index ``i`` such that ``sorted_items[:i]`` are all strictly
        less than ``target`` and ``sorted_items[i:]`` are all greater
        than or equal to ``target``. Empty input returns 0.
    """
''',
    "canonical_solution": '''    lo, hi = 0, len(sorted_items)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_items[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo
''',
    "test": '''def check(candidate):
    import bisect, random
    assert candidate([], 5) == 0
    assert candidate([1, 2, 3, 4, 5], 0) == 0
    assert candidate([1, 2, 3, 4, 5], 6) == 5
    assert candidate([1, 2, 3, 4, 5], 3) == 2
    # leftmost on duplicates
    assert candidate([1, 2, 2, 2, 3], 2) == 1
    assert candidate([2, 2, 2, 2], 2) == 0
    assert candidate([2, 2, 2, 2], 3) == 4
    rng = random.Random(0xC0FFEE)
    arr = sorted(rng.randint(0, 1000) for _ in range(500))
    for _ in range(200):
        t = rng.randint(-5, 1005)
        assert candidate(arr, t) == bisect.bisect_left(arr, t), (arr, t)
''',
})

# ─── Column 8 ─ Maximum-sum subarray (Kadane) ────────────────────────────────
PROBLEMS.append({
    "task_id": "pearls_col8_max_subarray",
    "column": 8,
    "difficulty": "medium",
    "entry_point": "max_subarray_sum",
    "prompt": '''from typing import List


def max_subarray_sum(values: List[int]) -> int:
    """Return the maximum sum over all contiguous subarrays of ``values``.

    Programming Pearls Column 8 ('Algorithm Design Techniques') walks
    this problem from the O(n^3) brute force through O(n^2), O(n log n),
    and finally Kadane's O(n) one-pass algorithm. Any correct algorithm
    is acceptable; the column's interest is the design progression.

    Args:
        values: list of ints; can be negative, positive, or zero.

    Returns:
        The maximum subarray sum. The empty subarray has sum 0, so for
        an all-negative list the answer is 0; for an empty list the
        answer is 0; for ``[5]`` the answer is 5.
    """
''',
    "canonical_solution": '''    best = 0
    cur = 0
    for v in values:
        cur = max(0, cur + v)
        if cur > best:
            best = cur
    return best
''',
    "test": '''def check(candidate):
    import random
    assert candidate([]) == 0
    assert candidate([5]) == 5
    assert candidate([-3]) == 0
    assert candidate([1, 2, 3, 4]) == 10
    assert candidate([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
    assert candidate([-1, -2, -3]) == 0
    assert candidate([5, -2, 3, -1, 5]) == 10
    rng = random.Random(2024)
    arr = [rng.randint(-50, 50) for _ in range(200)]
    # O(n^2) reference
    best = 0
    for i in range(len(arr)):
        s = 0
        for j in range(i, len(arr)):
            s += arr[j]
            if s > best:
                best = s
    assert candidate(arr) == best
''',
})

# ─── Column 9 ─ Code tuning: binomial coefficient without overflow ───────────
PROBLEMS.append({
    "task_id": "pearls_col9_binomial",
    "column": 9,
    "difficulty": "medium",
    "entry_point": "n_choose_k",
    "prompt": '''def n_choose_k(n: int, k: int) -> int:
    """Return the binomial coefficient C(n, k) as an arbitrary-precision int.

    Programming Pearls Column 9 ('Code Tuning') uses the binomial
    coefficient as an example of avoiding the trap in
    ``factorial(n) // (factorial(k) * factorial(n - k))`` -- huge
    intermediate values that take excessive memory and time. The
    tuned form multiplies and divides in lockstep so the running
    value stays small.

    Args:
        n: non-negative int.
        k: non-negative int. Result is 0 when ``k > n``.

    Returns:
        ``C(n, k)`` as a Python int. ``n_choose_k(0, 0) == 1``.
        Negative inputs are out of spec; tests do not exercise them.
    """
''',
    "canonical_solution": '''    if k < 0 or k > n:
        return 0
    if k > n - k:
        k = n - k
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result
''',
    "test": '''def check(candidate):
    assert candidate(0, 0) == 1
    assert candidate(5, 0) == 1
    assert candidate(5, 5) == 1
    assert candidate(5, 6) == 0
    assert candidate(10, 3) == 120
    assert candidate(20, 10) == 184756
    # row sum = 2^n
    n = 30
    assert sum(candidate(n, k) for k in range(n + 1)) == 2 ** n
    # symmetry
    for n in range(1, 25):
        for k in range(0, n + 1):
            assert candidate(n, k) == candidate(n, n - k)
    # no overflow / no factorial blowup
    assert candidate(100, 50) == 100891344545564193334812497256
''',
})

# ─── Column 11 ─ Quickselect ─────────────────────────────────────────────────
PROBLEMS.append({
    "task_id": "pearls_col11_quickselect",
    "column": 11,
    "difficulty": "medium",
    "entry_point": "kth_smallest",
    "prompt": '''from typing import List


def kth_smallest(values: List[int], k: int) -> int:
    """Return the ``k``-th smallest element of ``values`` (0-indexed).

    Programming Pearls Column 11 introduces partition-based selection
    (quickselect) using the same partition Hoare invented for
    quicksort. Any correct selection algorithm is acceptable.

    Args:
        values: non-empty list of ints (can have duplicates).
        k: 0 <= k < len(values).

    Returns:
        The element that would land at index ``k`` if ``values`` were
        sorted ascending. For ``[3, 1, 2]`` and ``k=1`` returns 2.
    """
''',
    "canonical_solution": '''    arr = list(values)
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        pivot = arr[(lo + hi) // 2]
        i, j = lo, hi
        while i <= j:
            while arr[i] < pivot:
                i += 1
            while arr[j] > pivot:
                j -= 1
            if i <= j:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
                j -= 1
        if k <= j:
            hi = j
        elif k >= i:
            lo = i
        else:
            return arr[k]
    return arr[k]
''',
    "test": '''def check(candidate):
    import random
    assert candidate([5], 0) == 5
    assert candidate([3, 1, 2], 0) == 1
    assert candidate([3, 1, 2], 1) == 2
    assert candidate([3, 1, 2], 2) == 3
    # duplicates
    assert candidate([5, 5, 5], 0) == 5
    assert candidate([5, 5, 5], 2) == 5
    assert candidate([1, 1, 2, 2, 3, 3], 3) == 2
    rng = random.Random(7)
    arr = [rng.randint(-1000, 1000) for _ in range(500)]
    s = sorted(arr)
    for k in (0, 1, 100, 250, 499):
        assert candidate(list(arr), k) == s[k]
''',
})

# ─── Column 12 ─ Sample k of n (Knuth's Algorithm S) ─────────────────────────
PROBLEMS.append({
    "task_id": "pearls_col12_sample_k_of_n",
    "column": 12,
    "difficulty": "medium",
    "entry_point": "sample_k_of_n",
    "prompt": '''from typing import List
import random


def sample_k_of_n(rng: random.Random, n: int, k: int) -> List[int]:
    """Return ``k`` distinct integers chosen uniformly from ``range(n)``,
    in ascending order.

    Programming Pearls Column 12 ('A Sample Problem') is the classic
    setting for Knuth's Algorithm S. The supplied ``rng`` is the only
    permitted source of randomness; the function must not call the
    module-level ``random`` API, so tests can pin the seed and get
    reproducible output.

    Args:
        rng: a seeded ``random.Random`` instance.
        n: population size; n >= 0.
        k: sample size; 0 <= k <= n.

    Returns:
        A new list of ``k`` distinct ints in ``range(n)``, ascending.
        ``k == 0`` returns the empty list.
    """
''',
    "canonical_solution": '''    selected: List[int] = []
    seen = 0
    remaining = k
    for i in range(n):
        if remaining == 0:
            break
        # P(select i) = remaining / (n - seen)
        if rng.random() < remaining / (n - seen):
            selected.append(i)
            remaining -= 1
        seen += 1
    return selected
''',
    "test": '''def check(candidate):
    import random
    rng = random.Random(42)
    assert candidate(rng, 0, 0) == []
    assert candidate(rng, 10, 0) == []
    out = candidate(rng, 10, 10)
    assert out == list(range(10))
    out = candidate(rng, 100, 5)
    assert len(out) == 5
    assert len(set(out)) == 5
    assert all(0 <= v < 100 for v in out)
    assert out == sorted(out)
    # determinism: same seed -> same output
    a = candidate(random.Random(1), 1000, 50)
    b = candidate(random.Random(1), 1000, 50)
    assert a == b
    assert len(set(a)) == 50
    assert all(0 <= v < 1000 for v in a)
    # large k of n
    big = candidate(random.Random(2), 10000, 9999)
    assert len(big) == 9999
    assert len(set(big)) == 9999
    assert big == sorted(big)
''',
})

# ─── Column 13 ─ Set with replay (bitmap representation hint) ────────────────
PROBLEMS.append({
    "task_id": "pearls_col13_intset_replay",
    "column": 13,
    "difficulty": "medium",
    "entry_point": "intset_replay",
    "prompt": '''from typing import List, Tuple, Union


def intset_replay(
    universe: int, ops: List[Tuple[str, int]]
) -> List[Union[bool, int]]:
    """Replay a sequence of ``(op, arg)`` operations against an integer
    set over ``range(universe)`` and return each operation's result.

    Programming Pearls Column 13 ('Searching') discusses set
    representations; the bitmap variant fits when the universe is
    dense and bounded -- a good choice given the API below.

    Operations:
      - ``("add", v)``    -> True if newly inserted, False if already in.
      - ``("has", v)``    -> True/False membership test.
      - ``("remove", v)`` -> True if it was present (and is now gone),
                             False if it wasn't in the set.
      - ``("size", _)``   -> current cardinality (an int). The arg is
                             ignored.

    Args:
        universe: exclusive upper bound on values; ops only reference
            ``v`` in ``range(universe)``.
        ops: ordered list of operation tuples.

    Returns:
        A list whose i-th entry is the result of ``ops[i]``. An empty
        ``ops`` list returns ``[]``.
    """
''',
    "canonical_solution": '''    present = bytearray(universe)
    count = 0
    out: List[Union[bool, int]] = []
    for op, v in ops:
        if op == "add":
            if present[v]:
                out.append(False)
            else:
                present[v] = 1
                count += 1
                out.append(True)
        elif op == "has":
            out.append(bool(present[v]))
        elif op == "remove":
            if present[v]:
                present[v] = 0
                count -= 1
                out.append(True)
            else:
                out.append(False)
        elif op == "size":
            out.append(count)
        else:
            raise ValueError(f"unknown op {op!r}")
    return out
''',
    "test": '''def check(candidate):
    import random
    assert candidate(10, []) == []
    out = candidate(10, [
        ("has", 3), ("add", 3), ("has", 3), ("add", 3),
        ("size", 0), ("remove", 3), ("has", 3),
        ("remove", 3), ("size", 0),
    ])
    assert out == [False, True, True, False, 1, True, False, False, 0]
    # stress against a built-in set oracle
    rng = random.Random(123)
    universe = 500
    oracle = set()
    ops = []
    expected = []
    for _ in range(2000):
        op = rng.choice(["add", "has", "remove", "size"])
        v = rng.randrange(universe) if op != "size" else 0
        ops.append((op, v))
        if op == "add":
            expected.append(v not in oracle)
            oracle.add(v)
        elif op == "has":
            expected.append(v in oracle)
        elif op == "remove":
            expected.append(v in oracle)
            oracle.discard(v)
        else:
            expected.append(len(oracle))
    assert candidate(universe, ops) == expected
''',
})

# ─── Column 14 ─ Top-k via heap ──────────────────────────────────────────────
PROBLEMS.append({
    "task_id": "pearls_col14_top_k_largest",
    "column": 14,
    "difficulty": "easy",
    "entry_point": "top_k_largest",
    "prompt": '''from typing import List


def top_k_largest(values: List[int], k: int) -> List[int]:
    """Return the ``k`` largest elements of ``values`` in descending order.

    Programming Pearls Column 14 ('Heaps') gives the priority queue as
    the data structure for this: maintain a size-``k`` min-heap, push
    each incoming value, pop when size exceeds ``k``. Result is the
    heap's contents, sorted descending. The Python ``heapq`` module is
    fair game; the column's interest is the O(n log k) complexity.

    Args:
        values: list of ints; len(values) >= k.
        k: 0 <= k <= len(values).

    Returns:
        New list of length ``k`` with the k largest elements in
        descending order. ``k == 0`` returns []. Ties: when several
        values share the cutoff, any of them is acceptable; tests
        check sorted-descending output and length.
    """
''',
    "canonical_solution": '''    import heapq
    if k <= 0:
        return []
    heap: List[int] = []
    for v in values:
        if len(heap) < k:
            heapq.heappush(heap, v)
        elif v > heap[0]:
            heapq.heapreplace(heap, v)
    return sorted(heap, reverse=True)
''',
    "test": '''def check(candidate):
    import random
    assert candidate([], 0) == []
    assert candidate([5], 1) == [5]
    assert candidate([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], 3) == [9, 6, 5]
    assert candidate([10, 20, 30, 40, 50], 5) == [50, 40, 30, 20, 10]
    out = candidate([5, 5, 5, 5, 5, 1, 2, 3], 4)
    assert len(out) == 4
    assert out == sorted(out, reverse=True)
    assert all(v == 5 for v in out)
    rng = random.Random(1)
    arr = [rng.randint(-1000, 1000) for _ in range(2000)]
    assert candidate(arr, 50) == sorted(arr, reverse=True)[:50]
''',
})

# ─── Column 15 ─ Longest repeated substring ──────────────────────────────────
PROBLEMS.append({
    "task_id": "pearls_col15_longest_repeated_substring",
    "column": 15,
    "difficulty": "hard",
    "entry_point": "longest_repeated_substring",
    "prompt": '''def longest_repeated_substring(s: str) -> str:
    """Return the longest substring of ``s`` that occurs at least twice.

    Programming Pearls Column 15 ('Strings of Pearls') solves this with
    a suffix array: sort all n suffixes, then the longest repeated
    substring is the longest common prefix between any two adjacent
    suffixes in the sorted list. Naive O(n^2) also works for the test
    sizes here. Overlapping occurrences count.

    Args:
        s: input string.

    Returns:
        Any of the longest repeated substrings (tied winners are all
        acceptable; tests check length and 'occurs >= 2 times' but
        not identity). Empty string when no character repeats.
    """
''',
    "canonical_solution": '''    n = len(s)
    if n < 2:
        return ""
    suffixes = sorted(range(n), key=lambda i: s[i:])
    best = ""
    for a, b in zip(suffixes, suffixes[1:]):
        k = 0
        while a + k < n and b + k < n and s[a + k] == s[b + k]:
            k += 1
        if k > len(best):
            best = s[a:a + k]
    return best
''',
    "test": '''def check(candidate):
    def brute(s):
        n = len(s)
        for L in range(n - 1, 0, -1):
            seen = {}
            for i in range(n - L + 1):
                sub = s[i:i + L]
                if sub in seen:
                    return sub
                seen[sub] = i
        return ""

    def occurs_at_least_twice(text, sub):
        # str.count is non-overlapping; do an overlapping count.
        if not sub:
            return False
        c = 0
        i = 0
        while True:
            j = text.find(sub, i)
            if j == -1:
                return c >= 2
            c += 1
            if c >= 2:
                return True
            i = j + 1

    assert candidate("") == ""
    assert candidate("abc") == ""
    assert candidate("aa") == "a"
    assert candidate("abcdef") == ""
    samples = (
        "banana",
        "mississippi",
        "ababab",
        "abcabcabc",
        "abracadabra",
        "ababcabcabab",
        "the quick brown fox jumps over the lazy dog the fox",
    )
    for sample in samples:
        out = candidate(sample)
        ref = brute(sample)
        assert len(out) == len(ref), (sample, out, ref)
        if out:
            assert out in sample, (sample, out)
            assert occurs_at_least_twice(sample, out), (sample, out)
''',
})

# ─── Column 15 ─ Word frequency concordance ──────────────────────────────────
PROBLEMS.append({
    "task_id": "pearls_col15_word_frequencies",
    "column": 15,
    "difficulty": "easy",
    "entry_point": "word_frequencies",
    "prompt": '''from typing import List, Tuple


def word_frequencies(text: str) -> List[Tuple[str, int]]:
    """Return word counts in ``text`` as ``(word, count)`` pairs sorted
    by descending count, then ascending word for ties.

    Programming Pearls Column 15 uses this as the simplest concordance.
    Words are maximal runs of non-whitespace characters; tokens are
    NOT lowercased and punctuation is NOT stripped, so ``"Foo"`` and
    ``"foo"`` count separately, as do ``"foo"`` and ``"foo,"``.

    Args:
        text: input text; can be empty or contain any whitespace.

    Returns:
        List of ``(word, count)`` pairs ordered by ``count`` descending;
        ties broken by ``word`` ascending. Empty text returns ``[]``.
    """
''',
    "canonical_solution": '''    counts: dict[str, int] = {}
    for w in text.split():
        counts[w] = counts.get(w, 0) + 1
    return sorted(counts.items(), key=lambda p: (-p[1], p[0]))
''',
    "test": '''def check(candidate):
    import random
    assert candidate("") == []
    assert candidate("a") == [("a", 1)]
    assert candidate("a a a") == [("a", 3)]
    assert candidate("a b a b c") == [("a", 2), ("b", 2), ("c", 1)]
    # case and punctuation are preserved
    assert candidate("Foo foo Foo") == [("Foo", 2), ("foo", 1)]
    assert candidate("foo, foo bar") == [("bar", 1), ("foo", 1), ("foo,", 1)]
    # mixed whitespace handling
    text = "x\\n\\ty  z\\nx x\\ty"
    assert candidate(text) == [("x", 3), ("y", 2), ("z", 1)]
    # tie-break by ascending word
    assert candidate("zebra apple zebra apple banana") == [
        ("apple", 2), ("zebra", 2), ("banana", 1),
    ]
    # large input against a counted reference
    rng = random.Random(7)
    words = [rng.choice(["foo", "bar", "baz", "qux"]) for _ in range(5000)]
    out = candidate(" ".join(words))
    expected_counts = {w: words.count(w) for w in set(words)}
    expected = sorted(expected_counts.items(), key=lambda p: (-p[1], p[0]))
    assert out == expected
''',
})


# ─────────────────────────────────────────────────────────────────────────────
# JSONL writer
# ─────────────────────────────────────────────────────────────────────────────

def _emit_jsonl(out_path: Path) -> int:
    out_path.write_text(
        "\n".join(
            json.dumps(p, ensure_ascii=True, sort_keys=True) for p in PROBLEMS
        )
        + "\n"
    )
    return len(PROBLEMS)


def main() -> None:
    here = Path(__file__).resolve().parent
    out = here / "pearls_problems.jsonl"
    n = _emit_jsonl(out)
    print(f"wrote {n} problems to {out}")


if __name__ == "__main__":
    main()
