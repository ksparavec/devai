"""Every picker modal remembers where the cursor was.

Backing out of a sub-modal used to drop you at the top of the model
list, so choosing the 9th model, pressing Esc in the agent modal, and
coming back meant scrolling to it again. The fix is a per-modal cursor
memory that `_fzf` restores via fzf's `start:pos(N)` action.

Two halves, tested at the right level for each:

  - fzf's own semantics were established directly against fzf 0.60,
    outside Python: with `--header-lines=3 --sync`,
    `start:pos(1)+accept` returns the first ITEM (not the first line),
    pos(2) the second, pos(4) the fourth. Crucially `--sync` is REQUIRED
    -- without it the binding can run before the item list is loaded and
    the position is silently dropped.

  - what this file covers: that `_fzf` emits those flags correctly, that
    the lines-index <-> item-position conversion is right (they differ
    by header_lines), that the position is saved on selection, and that
    a stale position from a longer list is clamped rather than passed
    through to fzf.

Argument construction is asserted by intercepting subprocess.run, which
is deterministic and needs no TTY -- driving a full-screen TUI from a
test would prove less and hang more.

Stdlib unittest only.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "model_picker_history", REPO_ROOT / "scripts" / "model-picker.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_picker_history"] = mod
    spec.loader.exec_module(mod)
    return mod


mp = _load()

LINES = ["HDR1", "HDR2", "HDR3", "alpha", "bravo", "charlie", "delta"]
SEL = [False, False, False, True, True, True, True]


class _Capture:
    """Stand in for subprocess.run, recording argv and returning a pick."""

    def __init__(self, picked_tag: str):
        self.argv: list[str] = []
        self._tag = picked_tag

    def __call__(self, args, **kwargs):
        self.argv = args
        return subprocess.CompletedProcess(
            args, 0, stdout=f"{self._tag}\trow\n", stderr="")


class FzfPositionArgsTest(unittest.TestCase):
    def setUp(self):
        mp._LAST_POS.clear()
        self._real = subprocess.run
        self.addCleanup(setattr, subprocess, "run", self._real)

    def _run(self, tag, **kw):
        cap = _Capture(tag)
        subprocess.run = cap
        idx = mp._fzf(LINES, "hdr", selectable=SEL, header_lines=3, **kw)
        return idx, cap.argv

    def test_first_visit_emits_no_position_and_no_sync(self):
        """A modal with no history must keep fzf's default startup."""
        _, argv = self._run("3", memory_key="model")
        self.assertNotIn("--sync", argv)
        self.assertFalse([a for a in argv if a.startswith("start:pos(")])

    def test_selection_is_remembered_as_an_item_position(self):
        """lines index 5 ('charlie') sits at item position 3, because the
        three header lines are not items."""
        idx, _ = self._run("5", memory_key="model")
        self.assertEqual(idx, 5)
        self.assertEqual(mp._LAST_POS["model"], 3)

    def test_second_visit_restores_that_position(self):
        self._run("5", memory_key="model")
        _, argv = self._run("3", memory_key="model")
        self.assertIn("--sync", argv,
                      "start:pos() is unreliable without --sync")
        self.assertIn("start:pos(3)", argv)

    def test_position_round_trips_to_the_same_row(self):
        """The conversion must be reversible: whatever row was picked is
        the row the restored position points at."""
        for lines_idx in (3, 4, 5, 6):
            with self.subTest(lines_idx=lines_idx):
                mp._LAST_POS.clear()
                self._run(str(lines_idx), memory_key="model")
                pos = mp._LAST_POS["model"]
                self.assertEqual(LINES[pos - 1 + 3], LINES[lines_idx])

    def test_stale_position_is_clamped_to_the_shorter_list(self):
        """A model disappearing (unprobed, excluded) shortens the list.
        Passing a position past the end would be a bad argument to fzf."""
        mp._LAST_POS["model"] = 99
        _, argv = self._run("3", memory_key="model")
        self.assertIn("start:pos(4)", argv,
                      "4 items follow the 3 header lines")

    def test_zero_and_negative_are_floored_to_one(self):
        mp._LAST_POS["model"] = 0
        _, argv = self._run("3", memory_key="model")
        self.assertNotIn("--sync", argv, "0 means 'nothing saved'")
        mp._LAST_POS["model"] = -5
        _, argv = self._run("3", memory_key="model")
        self.assertIn("start:pos(1)", argv)

    def test_no_memory_key_never_remembers(self):
        """Callers that opt out must not leak into the shared dict."""
        self._run("5")
        self.assertEqual(mp._LAST_POS, {})

    def test_modals_do_not_share_a_position(self):
        self._run("5", memory_key="model")
        self._run("3", memory_key="agent")
        self.assertEqual(mp._LAST_POS["model"], 3)
        self.assertEqual(mp._LAST_POS["agent"], 1)

    def test_headerless_modal_maps_index_to_position_directly(self):
        """Sub-modals (agent, reasoning, ctx tier) pass no header_lines,
        so index 2 is position 3."""
        cap = _Capture("2")
        subprocess.run = cap
        mp._fzf(["a", "b", "c"], "hdr", memory_key="agent")
        self.assertEqual(mp._LAST_POS["agent"], 3)


class AllModalsOptInTest(unittest.TestCase):
    """The ask was history on ALL levels, not just the model list."""

    def test_every_fzf_call_site_passes_a_memory_key(self):
        src = (REPO_ROOT / "scripts" / "model-picker.py").read_text()
        calls = [ln for ln in src.split("\n")
                 if "_fzf(" in ln and "def _fzf" not in ln]
        # The model list spans several lines; its memory_key is asserted
        # separately below.
        single_line = [ln for ln in calls if ln.rstrip().endswith(")")]
        for ln in single_line:
            with self.subTest(call=ln.strip()):
                self.assertIn("memory_key=", ln)

    def test_expected_modal_keys_are_all_present(self):
        src = (REPO_ROOT / "scripts" / "model-picker.py").read_text()
        for key in ("model", "agent", "kv_tier", "reasoning", "mtp",
                    "aiagent_gpu"):
            with self.subTest(key=key):
                self.assertIn(f'memory_key="{key}"', src)


if __name__ == "__main__":
    unittest.main()
