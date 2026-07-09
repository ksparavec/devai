"""max_context derivation in refresh_top_level_from_cells.

max_context must always equal the largest clean probed actual_context,
capped at the model's position_limit. It GROWS when a higher tier is
verified, SHRINKS when the position limit caps an over-promise, and --
the case the single-cell binary search introduced -- SHRINKS when a full
re-probe replaces a multi-cell entry whose stale max exceeded the new lone
winner. A stale max_context that points at a ctx with no backing cell would
make the router advertise/launch a context that was never serving-verified.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _probe_hf_common import refresh_top_level_from_cells  # noqa: E402


def _cell(ctx: int) -> dict:
    return {
        "ctx": ctx,
        "vram_gb": 24,
        "fits": True,
        "capability": "structured",
        "actual_context": ctx,
        "probed_at": "2026-07-09T00:00:00+00:00",
        "reasoning_parser": None,
        "tool_parser": None,
    }


def _entry(cells_by_ctx: dict[int, dict], *, max_context: int, position_limit=None) -> dict:
    e = {
        "capability": "structured",
        "max_context": max_context,
        "probes": {"24": {str(c): cell for c, cell in cells_by_ctx.items()}},
    }
    if position_limit is not None:
        e["position_limit"] = position_limit
    return e


class TestMaxContextSync(unittest.TestCase):
    def test_grows_when_higher_tier_verified(self) -> None:
        e = _entry({32768: _cell(32768), 131072: _cell(131072)}, max_context=32768)
        refresh_top_level_from_cells(e)
        self.assertEqual(e["max_context"], 131072)

    def test_position_limit_caps_over_promise(self) -> None:
        # A cell fit at 131072 (VLLM_ALLOW_LONG_MAX_MODEL_LEN) but the model
        # asserts past 40960 at serve time -> max_context must cap at 40960.
        e = _entry({131072: _cell(131072)}, max_context=131072, position_limit=40960)
        refresh_top_level_from_cells(e)
        self.assertEqual(e["max_context"], 40960)

    def test_single_cell_shrink_below_stale_max(self) -> None:
        # THE FIX: a full re-probe left one winner at 32768, but a stale
        # multi-cell run had recorded max_context=40960. 40960 is not itself
        # the position limit being newly applied, so the old grow/cap-only
        # logic left it untouched -> stale over-promise. It must sync to 32768.
        e = _entry({32768: _cell(32768)}, max_context=40960, position_limit=40960)
        refresh_top_level_from_cells(e)
        self.assertEqual(e["max_context"], 32768)

    def test_noop_when_already_correct(self) -> None:
        e = _entry({32768: _cell(32768)}, max_context=32768, position_limit=40960)
        refresh_top_level_from_cells(e)
        self.assertEqual(e["max_context"], 32768)


if __name__ == "__main__":
    unittest.main()
