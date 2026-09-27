"""The primer's worked examples must equal what scripts/stats computes.

docs/statistics-primer.md quotes about thirty computed numbers (intervals,
p-values, power, median-interval ranks). scripts/stats/primer_examples.py
recomputes each one and yields the exact text the primer shows; a number
edited by hand, or a change to statlib, fails here instead of leaving the
appendix silently wrong. Whitespace is normalised so line wrapping is free.

Stdlib unittest only.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "stats"))

import primer_examples  # noqa: E402

PRIMER = REPO_ROOT / "docs" / "statistics-primer.md"


def _norm(text: str) -> str:
    return " ".join(text.split())


class PrimerExamplesTest(unittest.TestCase):
    def test_every_computed_example_appears_in_the_primer(self):
        text = _norm(PRIMER.read_text(encoding="utf-8"))
        for desc, shown in primer_examples.examples():
            with self.subTest(example=desc):
                self.assertIn(_norm(shown), text)

    def test_primer_is_ascii_only(self):
        # Repo rule for markdown (CLAUDE.md, "Documentation conventions").
        data = PRIMER.read_bytes()
        self.assertTrue(all(b < 0x80 for b in data), "non-ASCII byte in the primer")


if __name__ == "__main__":
    unittest.main()
