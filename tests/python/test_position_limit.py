"""Tests for the position-limit catalog cap.

effective_position_limit reads a model's HARD context ceiling from
config.json (max_position_embeddings, rope-extended), and
refresh_top_level_from_cells caps/ shrinks max_context at it so the fit
probe stops advertising a context the model asserts on at serve time
(the Qwen3-8B/14B-NVFP4 40960 / gpt-oss-20b 131072 case).

Run: python3 -m unittest tests.python.test_position_limit
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _probe_hf_common import (  # noqa: E402
    effective_position_limit,
    refresh_top_level_from_cells,
)


def _write_cfg(root: Path, name: str, payload: dict) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(payload))


class TestEffectivePositionLimit(unittest.TestCase):
    def test_no_rope_returns_mpe(self) -> None:
        # Qwen3-8B/14B-NVFP4 shape: 40960, no YaRN.
        with tempfile.TemporaryDirectory() as td:
            _write_cfg(Path(td), "m", {"max_position_embeddings": 40960,
                                       "rope_scaling": None})
            self.assertEqual(effective_position_limit("m", Path(td)), 40960)

    def test_yarn_widens_to_factor_times_original(self) -> None:
        # gpt-oss-20b shape: mpe already 131072, YaRN factor 32 x 4096.
        with tempfile.TemporaryDirectory() as td:
            _write_cfg(Path(td), "m", {
                "max_position_embeddings": 131072,
                "rope_scaling": {"rope_type": "yarn", "factor": 32.0,
                                 "original_max_position_embeddings": 4096},
            })
            self.assertEqual(effective_position_limit("m", Path(td)), 131072)

    def test_yarn_extends_beyond_mpe(self) -> None:
        # Convention where mpe is the BASE and factor*original is larger.
        with tempfile.TemporaryDirectory() as td:
            _write_cfg(Path(td), "m", {
                "max_position_embeddings": 8192,
                "rope_scaling": {"factor": 4.0,
                                 "original_max_position_embeddings": 32768},
            })
            # max(8192, 4*32768=131072) = 131072
            self.assertEqual(effective_position_limit("m", Path(td)), 131072)

    def test_text_config_scope(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _write_cfg(Path(td), "m", {"text_config": {"max_position_embeddings": 262144}})
            self.assertEqual(effective_position_limit("m", Path(td)), 262144)

    def test_takes_larger_across_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _write_cfg(Path(td), "m", {
                "max_position_embeddings": 4096,
                "text_config": {"max_position_embeddings": 131072},
            })
            self.assertEqual(effective_position_limit("m", Path(td)), 131072)

    def test_none_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _write_cfg(Path(td), "m", {"hidden_size": 4096})
            self.assertIsNone(effective_position_limit("m", Path(td)))

    def test_none_when_config_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(effective_position_limit("absent", Path(td)))


class TestMaxContextCap(unittest.TestCase):
    def _entry(self, position_limit, largest_ctx, max_context=0):
        # Minimal entry: one clean cell at largest_ctx so
        # refresh_top_level_from_cells derives max_context from it.
        return {
            "schema_version": 2,
            "capability": "inline",
            "probes": {
                "24": {
                    str(largest_ctx): {
                        "ctx": largest_ctx,
                        "capability": "inline",
                        "fits": True,
                        "actual_context": largest_ctx,
                    }
                }
            },
            "position_limit": position_limit,
            "max_context": max_context,
        }

    def test_shrinks_over_promise_to_limit(self) -> None:
        # Qwen3-8B: a prior run recorded a 131072 cell + max_context, but
        # the real limit is 40960 -> must shrink.
        e = self._entry(position_limit=40960, largest_ctx=131072, max_context=131072)
        refresh_top_level_from_cells(e)
        self.assertEqual(e["max_context"], 40960)

    def test_no_cap_when_within_limit(self) -> None:
        # Llama-3.1-8B: 131072 cell, limit 131072 -> unchanged.
        e = self._entry(position_limit=131072, largest_ctx=131072, max_context=0)
        refresh_top_level_from_cells(e)
        self.assertEqual(e["max_context"], 131072)

    def test_no_position_limit_is_legacy(self) -> None:
        # No position_limit stamped -> behaves as before (grows to cell).
        e = self._entry(position_limit=None, largest_ctx=131072, max_context=0)
        e.pop("position_limit")
        refresh_top_level_from_cells(e)
        self.assertEqual(e["max_context"], 131072)


if __name__ == "__main__":
    unittest.main()
