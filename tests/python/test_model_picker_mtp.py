"""Picker MTP-offerability gate.

The picker must not surface the ::mtp toggle (nor a 'Yes' MTP column) for a row
whose fit probe recorded mtp_fits=false -- the qwen3_5_mtp draft lm_head OOMs at
load, so serving it would 503. Mirrors the router's modelMTPUnfit suppression.

_has_mtp(m) is now: catalog declares an `mtp:` block AND the probe didn't record
mtp_fits=false. _mtp_probe_unfit(m) reads m['probe']['probes'][vram][ctx].
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_picker():
    spec = importlib.util.spec_from_file_location(
        "_picker_under_test_mtp",
        str(REPO_ROOT / "scripts" / "model-picker.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


PICKER = _load_picker()


def _row(has_block: bool = True, mtp_fits_values=()) -> dict:
    """Build a picker row. mtp_fits_values: per-cell mtp_fits (True/False/None)."""
    band = {}
    for i, v in enumerate(mtp_fits_values):
        cell = {"ctx": 32768 + i, "fits": True}
        if v is not None:
            cell["mtp_fits"] = v
        band[str(32768 + i)] = cell
    return {
        "catalog_meta": (
            {"mtp": {"method": "qwen3_5_mtp", "num_speculative_tokens": 3}}
            if has_block else {}
        ),
        "probe": {"probes": {"24": band}},
    }


class TestMTPOfferability(unittest.TestCase):
    def test_probe_false_suppresses(self) -> None:
        m = _row(True, [False])
        self.assertTrue(PICKER._mtp_probe_unfit(m))
        self.assertFalse(PICKER._has_mtp(m))  # not offered -> column 'No', no sub-modal

    def test_probe_true_offers(self) -> None:
        m = _row(True, [True])
        self.assertFalse(PICKER._mtp_probe_unfit(m))
        self.assertTrue(PICKER._has_mtp(m))

    def test_probe_absent_offers(self) -> None:
        # un-probed for MTP (no mtp_fits key) -> preserve prior behaviour
        m = _row(True, [None])
        self.assertFalse(PICKER._mtp_probe_unfit(m))
        self.assertTrue(PICKER._has_mtp(m))

    def test_no_cells_offers(self) -> None:
        m = _row(True, [])
        self.assertFalse(PICKER._mtp_probe_unfit(m))
        self.assertTrue(PICKER._has_mtp(m))

    def test_any_true_wins(self) -> None:
        m = _row(True, [False, True])
        self.assertFalse(PICKER._mtp_probe_unfit(m))
        self.assertTrue(PICKER._has_mtp(m))

    def test_no_catalog_block_never_offers(self) -> None:
        # even if the probe said MTP fits, no catalog mtp: block -> not offered
        m = _row(False, [True])
        self.assertFalse(PICKER._has_mtp(m))


if __name__ == "__main__":
    unittest.main()
