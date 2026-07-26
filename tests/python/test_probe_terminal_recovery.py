"""A stale terminal probe verdict must yield to a clean re-probe.

Observed 2026-07-25. ykarout/Qwen3.5-9B-NVFP4 was re-probed under SGLang
with PROBE_FORCE=1 and came back clean: it fit at 256K, classified
`structured`, and resolved rp=qwen3 / tp=qwen. The cell recorded all of
that correctly. The top-level row did not -- it stayed at `error` with
`evidence.kind=quant` and both parser fields None, carried over from a
July run.

The cause was an early return in refresh_top_level_from_cells: an `error`
with arch/quant evidence is treated as tier-independent and therefore
sticky, and the guard ran BEFORE the clean-cell lookup, so no amount of
re-probing could clear it.

Why it mattered: the router injects `--reasoning-parser` and
`--tool-call-parser` from those top-level fields at container launch. The
model would have been served with neither -- silently losing tool calling
and reasoning on a model that probes perfectly -- and the picker would
have shown it as `error`.

The rule now: stickiness holds only while there is no clean cell. A
tier-independent load failure and a probe that loaded and produced
structured output cannot both be true of the same (repo, sha, backend).

Stdlib unittest only.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "_probe_hf_common", REPO_ROOT / "scripts" / "_probe_hf_common.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_probe_hf_common"] = mod
    spec.loader.exec_module(mod)
    return mod


ph = _load()


def _clean_cell(ctx: int = 262144) -> dict:
    return {
        "fits": True,
        "ctx": ctx,
        "actual_context": ctx,
        "capability": "structured",
        "reasoning_parser": "qwen3",
        "tool_parser": "qwen",
        "probed_at": "2026-07-25T14:14:00+00:00",
        "evidence": {"tool": {"mode": "auto", "verified": True}},
    }


class TerminalYieldsToCleanCellTest(unittest.TestCase):
    def test_quant_error_cleared_by_a_clean_cell(self):
        entry = {
            "capability": "error",
            "evidence": {"kind": "quant", "startup_error": "old failure"},
            "reasoning_parser": None,
            "tool_parser": None,
            "probes": {"24": {"262144": _clean_cell()}},
        }
        ph.refresh_top_level_from_cells(entry)
        self.assertEqual(entry["capability"], "structured")

    def test_parsers_recovered_so_the_router_can_launch_correctly(self):
        """The router reads these two fields to build its engine flags."""
        entry = {
            "capability": "error",
            "evidence": {"kind": "quant"},
            "reasoning_parser": None,
            "tool_parser": None,
            "probes": {"24": {"262144": _clean_cell()}},
        }
        ph.refresh_top_level_from_cells(entry)
        self.assertEqual(entry["reasoning_parser"], "qwen3")
        self.assertEqual(entry["tool_parser"], "qwen")

    def test_arch_error_cleared_by_a_clean_cell(self):
        entry = {
            "capability": "error",
            "evidence": {"kind": "arch"},
            "probes": {"24": {"262144": _clean_cell()}},
        }
        ph.refresh_top_level_from_cells(entry)
        self.assertEqual(entry["capability"], "structured")

    def test_unsupported_arch_cleared_by_a_clean_cell(self):
        """If the engine loaded it and answered, it supports the arch."""
        entry = {
            "capability": "unsupported_arch",
            "evidence": {"kind": "arch"},
            "probes": {"24": {"262144": _clean_cell()}},
        }
        ph.refresh_top_level_from_cells(entry)
        self.assertEqual(entry["capability"], "structured")


class TerminalStaysStickyWithoutEvidenceTest(unittest.TestCase):
    """The stickiness itself must survive -- this is the behaviour the
    early return was protecting, and it is still correct when nothing
    contradicts it."""

    def test_quant_error_with_no_clean_cell_stays(self):
        entry = {
            "capability": "error",
            "evidence": {"kind": "quant"},
            "probes": {"24": {"32768": {
                "fits": False, "capability": "error",
                "evidence": {"kind": "quant"},
            }}},
        }
        ph.refresh_top_level_from_cells(entry)
        self.assertEqual(entry["capability"], "error")

    def test_unsupported_arch_with_no_cells_stays(self):
        entry = {"capability": "unsupported_arch",
                 "evidence": {"kind": "arch"}, "probes": {}}
        ph.refresh_top_level_from_cells(entry)
        self.assertEqual(entry["capability"], "unsupported_arch")

    def test_oom_error_still_yields_to_a_higher_tier_success(self):
        """Pre-existing behaviour: a 16G spill must not invalidate a 24G
        fit. oom is tier-specific, never terminal."""
        entry = {
            "capability": "error",
            "evidence": {"kind": "oom"},
            "probes": {
                "16": {"32768": {"fits": False, "capability": "error",
                                 "evidence": {"kind": "oom"}}},
                "24": {"262144": _clean_cell()},
            },
        }
        ph.refresh_top_level_from_cells(entry)
        self.assertEqual(entry["capability"], "structured")


if __name__ == "__main__":
    unittest.main()
