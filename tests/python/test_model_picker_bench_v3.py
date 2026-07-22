"""Picker-side schema-v3 bench-record tests.

Covers:
  - _load_bench_records: returns dict keyed by (model, backend, ctx),
    indexes pre-v3 rows at ctx=0, emits a one-line stderr warning.
  - _build_candidates: lookup uses (name, backend, _picker_context),
    annotates _picker_bench_other_ctxs.
  - _format_model_properties: emits the "Bench: not available at ctx=N
    (have ...)" line when bench data exists at other ctxs.

The picker filename has a hyphen, so it isn't a valid Python module
name. We load it via importlib.util.spec_from_file_location.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _load_picker():
    spec = importlib.util.spec_from_file_location(
        "_picker_under_test",
        str(REPO_ROOT / "scripts" / "model-picker.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


PICKER = _load_picker()


def _write_bench_cache(tmp: Path, payload: dict) -> str:
    p = tmp / ".bench-cache.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


class TestLoadBenchRecordsV3(unittest.TestCase):
    def test_v3_rows_keyed_by_triple(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cache = {
                "_meta": {"schema_version": 3},
                "openai/gpt-oss-20b@x::vllm::262144": {
                    "model": "gpt-oss-20b",
                    "backend": "vllm",
                    "context": 262144,
                    "metrics": {"tps_sustained_p50": 139.2},
                },
                "openai/gpt-oss-20b@x::vllm::32768": {
                    "model": "gpt-oss-20b",
                    "backend": "vllm",
                    "context": 32768,
                    "metrics": {"tps_sustained_p50": 150.0},
                },
            }
            path = _write_bench_cache(tmp, cache)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                rec = PICKER._load_bench_records([path])
            self.assertIn(("gpt-oss-20b", "vllm", 262144), rec)
            self.assertIn(("gpt-oss-20b", "vllm", 32768), rec)
            self.assertEqual(
                rec[("gpt-oss-20b", "vllm", 262144)]["metrics"][
                    "tps_sustained_p50"
                ],
                139.2,
            )
            # No legacy rows -> no warning.
            self.assertNotIn("pre-v3", stderr.getvalue())

    def test_pre_v3_rows_indexed_at_zero_with_warning(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cache = {
                "openai/gpt-oss-20b@x::vllm": {
                    # No context field -> ctx=0
                    "model": "gpt-oss-20b",
                    "backend": "vllm",
                },
            }
            path = _write_bench_cache(tmp, cache)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                rec = PICKER._load_bench_records([path])
            self.assertIn(("gpt-oss-20b", "vllm", 0), rec)
            self.assertIn("pre-v3", stderr.getvalue())

    def test_empty_when_no_paths(self) -> None:
        rec = PICKER._load_bench_records(["/nonexistent/path"])
        self.assertEqual(rec, {})


class TestBuildCandidatesUsesCtxKey(unittest.TestCase):
    def _model(
        self,
        name: str,
        backend: str = "vllm",
        ctx: int = 131072,
    ) -> dict:
        return {
            "name": name,
            "backend": backend,
            "capability": "structured",
            "tool_parser": "hermes",
            "details": {"quantization": "NVFP4", "param_size": "8B"},
            "_max_fitting_ctx_info_override": {
                "_picker_ctx": ctx,
                "total_gb": 22.0,
            },
        }

    def setUp(self) -> None:
        # _max_fitting_ctx_info(m) is the source of _picker_context.
        # We monkey-patch it so the test can pin a ctx without setting
        # up real probe-cache fixtures.
        self._orig = PICKER._max_fitting_ctx_info

        def fake(m):
            return m.get("_max_fitting_ctx_info_override")

        PICKER._max_fitting_ctx_info = fake  # type: ignore[assignment]

    def tearDown(self) -> None:
        PICKER._max_fitting_ctx_info = self._orig  # type: ignore[assignment]

    def test_lookup_by_triple_hits_correct_ctx(self) -> None:
        bench_records = {
            ("Qwen3-8B-NVFP4", "vllm", 131072): {
                "model": "Qwen3-8B-NVFP4",
                "backend": "vllm",
                "context": 131072,
                "tasks": {
                    "gsm8k_subset_100": {
                        "score": 0.85,
                        "ran_at": "2026-05-05T10:00:00+00:00",
                    },
                    "humaneval_subset_50": {
                        "pass@1": 0.7,
                        "ran_at": "2026-05-05T10:00:00+00:00",
                    },
                    "tools_use_20": {
                        "score": 0.9,
                        "ran_at": "2026-05-05T10:00:00+00:00",
                    },
                },
                "metrics": {"tps_sustained_p50": 80.0},
            },
            ("Qwen3-8B-NVFP4", "vllm", 32768): {
                "model": "Qwen3-8B-NVFP4",
                "backend": "vllm",
                "context": 32768,
                "tasks": {
                    "gsm8k_subset_100": {
                        "score": 0.99,
                        "ran_at": "2026-05-05T10:00:00+00:00",
                    },
                },
                "metrics": {"tps_sustained_p50": 95.0},
            },
        }
        models = [self._model("Qwen3-8B-NVFP4", ctx=131072)]
        candidates, _ = PICKER._build_candidates(models, bench_records)
        self.assertEqual(len(candidates), 1)
        c = candidates[0]
        self.assertEqual(c["_picker_context"], 131072)
        # Must hit the 128K row, not the 32K row.
        self.assertEqual(
            c["_picker_bench_row"]["metrics"]["tps_sustained_p50"], 80.0
        )
        # The 32K row is still recorded as "available at other ctx".
        self.assertEqual(c["_picker_bench_other_ctxs"], [32768])

    def test_no_bench_at_chosen_ctx_records_other_tiers(self) -> None:
        bench_records = {
            ("Qwen3-8B-NVFP4", "vllm", 32768): {
                "model": "Qwen3-8B-NVFP4",
                "backend": "vllm",
                "context": 32768,
                "tasks": {},
                "metrics": {"tps_sustained_p50": 95.0},
            },
        }
        models = [self._model("Qwen3-8B-NVFP4", ctx=131072)]
        candidates, _ = PICKER._build_candidates(models, bench_records)
        c = candidates[0]
        self.assertIsNone(c["_picker_bench_row"])
        self.assertEqual(c["_picker_bench_other_ctxs"], [32768])

    def test_legacy_ctx_zero_row_does_not_match_real_picker_ctx(self) -> None:
        bench_records = {
            ("Qwen3-8B-NVFP4", "vllm", 0): {
                "model": "Qwen3-8B-NVFP4",
                "backend": "vllm",
                "tasks": {},
                "metrics": {"tps_sustained_p50": 80.0},
            },
        }
        models = [self._model("Qwen3-8B-NVFP4", ctx=131072)]
        candidates, _ = PICKER._build_candidates(models, bench_records)
        c = candidates[0]
        # ctx=0 row must not silently substitute for a real ctx request.
        self.assertIsNone(c["_picker_bench_row"])
        # ctx=0 is excluded from "other ctxs" (it's not a real tier).
        self.assertEqual(c["_picker_bench_other_ctxs"], [])


class TestFormatModelPropertiesNotAvailable(unittest.TestCase):
    def test_emits_not_available_line_when_other_ctxs_exist(self) -> None:
        m = {
            "_picker_context": 131072,
            "_picker_scores": {"total": None},
            "_picker_bench_row": None,
            "_picker_bench_other_ctxs": [32768, 65536],
        }
        out = PICKER._format_model_properties(m, comparison=None)
        self.assertIn("not available at ctx=128K", out)
        self.assertIn("32K", out)
        self.assertIn("64K", out)
        self.assertIn("make bench --ctx 128K", out)


class UseCaseRecommenderTests(unittest.TestCase):
    """The per-model use-case recommender (_use_case_ratings / tiers /
    _format_use_case_recommendations) and the auxiliary gsm/leak metrics."""

    def _model(self, *, ctx, gpqa, mmlu, hevp, code=0.8, gsm=0.95,
               tps=80.0, leak=0.0):
        return {
            "name": "m",
            "_picker_context": ctx,
            "_picker_scores": {
                "tps": tps, "code": code, "hevp": hevp, "mmlu": mmlu,
                "gpqa": gpqa, "tools": 1.0, "gsm": gsm, "leak": leak,
            },
        }

    def test_picker_scores_includes_gsm_and_leak(self):
        row = {
            "tasks": {
                "gsm8k_subset_100": {"score": 0.9},
                "leak_probe": {"leak_rate": 0.05},
            },
            "metrics": {},
        }
        s = PICKER._picker_scores(row)
        self.assertEqual(s["gsm"], 0.9)
        self.assertEqual(s["leak"], 0.05)
        # None-row still carries the keys (so consumers never KeyError).
        self.assertIsNone(PICKER._picker_scores(None)["gsm"])

    def test_unbenched_returns_none(self):
        m = {"_picker_context": 262144, "_picker_scores": {"gpqa": None,
             "mmlu": None, "hevp": None}}
        self.assertIsNone(PICKER._use_case_ratings(m))

    def test_tiers_boundaries(self):
        self.assertEqual(PICKER._use_case_tier(80), "Excellent")
        self.assertEqual(PICKER._use_case_tier(79.9), "Strong")
        self.assertEqual(PICKER._use_case_tier(65), "Strong")
        self.assertEqual(PICKER._use_case_tier(50), "Good")
        self.assertEqual(PICKER._use_case_tier(35), "Fair")
        self.assertEqual(PICKER._use_case_tier(34.9), "Weak")

    def test_long_ctx_reasoner_tops_doc_tasks(self):
        # 256K + strong MMLU/GPQA, mediocre coder -> doc_qa/summary rank first.
        m = self._model(ctx=262144, gpqa=0.78, mmlu=0.83, hevp=0.66, code=0.68)
        ranked = PICKER._use_case_ratings(m)
        top2 = {k for k, _ in ranked[:2]}
        self.assertEqual(top2, {"doc_qa", "summary"})

    def test_short_ctx_coder_tops_coding_not_docs(self):
        # 64K + strong coding -> coding first; short ctx pushes docs down.
        m = self._model(ctx=65536, gpqa=0.87, mmlu=0.82, hevp=0.98, code=0.96)
        ranked = PICKER._use_case_ratings(m)
        self.assertEqual(ranked[0][0], "coding")
        order = [k for k, _ in ranked]
        self.assertLess(order.index("coding"), order.index("summary"))
        self.assertLess(order.index("coding"), order.index("doc_qa"))

    def test_format_lists_all_five_use_cases(self):
        m = self._model(ctx=262144, gpqa=0.78, mmlu=0.83, hevp=0.66)
        out = PICKER._format_use_case_recommendations(m)
        self.assertIn("Recommended for:", out)
        for label in ("Coding", "Math / analysis", "Gen. reasoning",
                      "Doc summary", "Doc Q&A"):
            self.assertIn(label, out)
        # honesty note about the proxied doc scores
        self.assertIn("estimate", out.lower())

    def test_format_empty_when_unbenched(self):
        m = {"_picker_context": 0, "_picker_scores": {"gpqa": None,
             "mmlu": None, "hevp": None}}
        self.assertEqual(PICKER._format_use_case_recommendations(m), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
