"""The picker's "hidden:" footer must render every bucket it counts.

Regression pin for a NameError that made `devai-agent` unusable on any
host whose exclusion ledger carried a bench verdict.

55fd0b7 added the `bench_excluded` bucket to `_build_candidates` and a
matching footer line to `_build_menu`, but the footer branch appended to
`notes` where every sibling branch in the same block appends to `bits`.
`notes` is not a local of `_build_menu` -- it only exists in a different
function -- so the branch raised

    NameError: name 'notes' is not defined

The branch is guarded by `if hidden.get("bench_excluded")`, so it is dead
code until an operator actually records a bench verdict. That is why it
shipped green: on a fleet with an empty ledger the footer never reaches
the typo. The moment one model was dropped the picker aborted during
menu construction -- before fzf was ever spawned, so the user saw a bare
traceback instead of a model list.

These tests exercise `_build_menu` with `_candidates=[]` so only the
footer runs, and assert each bucket both renders when non-zero and stays
absent when zero (a footer that over-claims is its own bug).

Stdlib unittest only. No probe/bench cache required.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "model_picker_hidden_footer", REPO_ROOT / "scripts" / "model-picker.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_picker_hidden_footer"] = mod
    spec.loader.exec_module(mod)
    return mod


mp = _load()

ALL_ZERO = {"missing_capability": 0, "no_fitting_ctx": 0, "bench_excluded": 0}


def _menu(**counts):
    """Render a menu with no candidates so only the footer executes."""
    hidden = dict(ALL_ZERO)
    hidden.update(counts)
    lines, _selectable, _items = mp._build_menu(
        [], None, _candidates=[], _hidden=hidden)
    return "\n".join(lines)


def _footer(text: str) -> str:
    for line in text.splitlines():
        if "hidden:" in line:
            return line
    return ""


class HiddenFooterTest(unittest.TestCase):
    def test_bench_excluded_alone_does_not_raise(self):
        """The exact crash: one bench verdict, nothing else hidden."""
        self.assertIn("dropped on bench verdict", _menu(bench_excluded=1))

    def test_bench_excluded_reports_its_count(self):
        self.assertIn("3 dropped on bench verdict", _menu(bench_excluded=3))

    def test_bench_excluded_names_the_restore_path(self):
        # An operator who cannot see the model needs to know how to get
        # it back; the count on its own is a dead end.
        footer = _footer(_menu(bench_excluded=1))
        self.assertIn("make model-status", footer)
        self.assertIn("CLEAR=", footer)

    def test_every_bucket_renders_together(self):
        footer = _footer(_menu(
            bench_excluded=1, no_fitting_ctx=2, missing_capability=4))
        self.assertIn("dropped on bench verdict", footer)
        self.assertIn("2 no context tier fits", footer)
        self.assertIn("4 not probed/probe failed", footer)

    def test_each_bucket_survives_alone(self):
        # Each branch appends to the same list; a typo in any one of them
        # only shows when that bucket is the non-zero one.
        for bucket, needle in (
            ("bench_excluded", "dropped on bench verdict"),
            ("no_fitting_ctx", "no context tier fits"),
            ("missing_capability", "not probed/probe failed"),
        ):
            with self.subTest(bucket=bucket):
                self.assertIn(needle, _menu(**{bucket: 1}))

    def test_no_footer_when_nothing_hidden(self):
        self.assertEqual(_footer(_menu()), "")

    def test_zero_bucket_is_not_claimed(self):
        footer = _footer(_menu(no_fitting_ctx=1))
        self.assertNotIn("dropped on bench verdict", footer)
        self.assertNotIn("not probed/probe failed", footer)


class HiddenFooterSourceTest(unittest.TestCase):
    """`notes` must not reappear in `_build_menu`.

    The bug was a single wrong identifier that static syntax checks and an
    empty-ledger test run both pass. Pin the function body directly so a
    future edit cannot reintroduce it in a branch no test data reaches.
    """

    def test_build_menu_has_no_notes_identifier(self):
        import inspect
        offenders = [
            f"{n}: {line.strip()}"
            for n, line in enumerate(
                inspect.getsource(mp._build_menu).splitlines(), start=1)
            if re.search(r"\bnotes\b", line)
        ]
        # Report just the offending lines -- asserting on the whole function
        # body dumps 100+ lines into the failure output and buries the cause.
        self.assertEqual(offenders, [], f"`notes` in _build_menu: {offenders}")


if __name__ == "__main__":
    unittest.main()
