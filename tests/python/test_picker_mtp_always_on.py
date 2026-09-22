"""The picker's MTP UI is unconditional: no preview flag.

`DEVAI_MTP_PREVIEW` gated the MTP column, the ON/OFF sub-modal and the
`::mtp` suffix behind an env var that defaulted to off and that nothing in
the repo (launcher, compose, Makefile) ever set. It was meant to be flipped
"alongside the router's parseMTPOverride wiring in Phase 5"; that wiring
shipped in 2026-05 and the flag stayed. Net effect on 2026-09-22: a row
with a catalog `mtp:` block and a probe cell `mtp_fits=true` was launched
MTP-off from the picker with no indication that the mode existed (37 vs
95 tok/s measured on Qwen3.8-27B-MTP-devai-NVFP4). Operator decision: the
flag is gone; an MTP-capable row always gets the column and the sub-modal.

Stdlib unittest only; no fzf, nothing served.
"""

from __future__ import annotations

import importlib.util
import os
import re
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
PICKER = REPO_ROOT / "scripts" / "model-picker.py"


def _load_picker():
    spec = importlib.util.spec_from_file_location("model_picker_mtp_always_on", PICKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mtp_row() -> dict:
    return {
        "name": "X-devai-NVFP4", "backend": "vllm-devai", "capability": "structured",
        "vram": {}, "probe": {"probes": {"24": {"118784": {"mtp_fits": True}}}},
        "catalog_meta": {"mtp": {"method": "mtp", "num_speculative_tokens": 3}},
        "details": {"quantization": "NVFP4"},
    }


class NoPreviewFlagTest(unittest.TestCase):
    def test_flag_is_gone_from_the_source(self) -> None:
        src = PICKER.read_text()
        # The identifier and the env read must be gone; the historical note
        # that names DEVAI_MTP_PREVIEW in a comment may stay.
        self.assertIsNone(re.search(r"(?<![A-Z])_MTP_PREVIEW\b", src))
        self.assertNotIn('environ.get("DEVAI_MTP_PREVIEW"', src)

    def test_mtp_column_renders_without_any_env(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEVAI_MTP_PREVIEW", None)
            mp = _load_picker()
        self.assertTrue(mp._has_mtp(_mtp_row()))
        row = mp._format_model_row(_mtp_row(), 1)
        self.assertRegex(row, r"\bYes\b", "MTP column must show Yes for an MTP-capable row")
        lines, _, _ = mp._build_menu([_mtp_row()])
        self.assertTrue(any(re.search(r"\bMTP\b", ln) for ln in lines[:3]), "menu header must carry the MTP column")

    def test_suffix_emission_depends_only_on_the_sub_modal(self) -> None:
        src = PICKER.read_text()
        self.assertIn('mtp_suffix = "::mtp" if mtp_mode == "on" else ""', src)
        self.assertIn("if _has_mtp(model):", src)


if __name__ == "__main__":
    unittest.main()
