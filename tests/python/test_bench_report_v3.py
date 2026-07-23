"""Tests for bench_report.render() rendering schema v3 caches.

Verifies:
  - CTX column is present in the header.
  - Multi-ctx benches of the same model produce multiple rows,
    grouped (model, ctx).
  - Missing ctx renders as `-`.
  - _ctx_label round-trips standard tiers.
  - schema-v3 footer note is emitted.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bench import bench_report  # noqa: E402


class TestCtxLabel(unittest.TestCase):
    def test_zero_renders_dash(self) -> None:
        self.assertEqual(bench_report._ctx_label(0), "-")

    def test_negative_renders_dash(self) -> None:
        self.assertEqual(bench_report._ctx_label(-1), "-")

    def test_kilo_multiples(self) -> None:
        self.assertEqual(bench_report._ctx_label(32768), "32K")
        self.assertEqual(bench_report._ctx_label(65536), "64K")
        self.assertEqual(bench_report._ctx_label(131072), "128K")
        self.assertEqual(bench_report._ctx_label(262144), "256K")

    def test_non_kilo_falls_back_to_int(self) -> None:
        self.assertEqual(bench_report._ctx_label(40000), "40000")


def _row(
    model: str,
    backend: str,
    ctx: int,
    *,
    tps: float | None = None,
    peak: float | None = None,
    gsm: float | None = None,
    he: float | None = None,
    tools: float | None = None,
    hep: float | None = None,
) -> dict:
    tasks: dict = {}
    if gsm is not None:
        tasks["gsm8k_subset_100"] = {
            "score": gsm,
            "ran_at": "2026-05-05T10:00:00+00:00",
        }
    if hep is not None:
        # Deliberately inserted BEFORE plain HumanEval: a bare
        # "humaneval_" prefix match would return this one first.
        tasks["humaneval_plus_subset_50"] = {
            "pass@1": hep,
            "ran_at": "2026-05-06T10:00:00+00:00",
        }
    if he is not None:
        tasks["humaneval_subset_50"] = {
            "pass@1": he,
            "ran_at": "2026-05-05T10:00:00+00:00",
        }
    if tools is not None:
        tasks["tools_use_20"] = {
            "score": tools,
            "ran_at": "2026-05-05T10:00:00+00:00",
        }
    metrics: dict = {}
    if tps is not None:
        metrics["tps_sustained_p50"] = tps
    if peak is not None:
        metrics["peak_vram_gb"] = peak
    return {
        "model": model,
        "backend": backend,
        "context": ctx,
        "tasks": tasks,
        "metrics": metrics,
        "host_env_id": "abc12345",
        "schema_version": 3,
    }


class TestRenderV3(unittest.TestCase):
    def test_ctx_column_present_in_header(self) -> None:
        cache = {
            "openai/gpt-oss-20b@x::vllm::262144": _row(
                "gpt-oss-20b", "vllm", 262144, tps=139.2
            ),
        }
        out = bench_report.render(cache, host_vram_gb=24.0)
        self.assertIn("| CTX |", out)

    def test_multi_ctx_rows_for_same_model(self) -> None:
        cache = {
            "openai/gpt-oss-20b@x::vllm::32768": _row(
                "gpt-oss-20b", "vllm", 32768, tps=150.0, gsm=0.99
            ),
            "openai/gpt-oss-20b@x::vllm::262144": _row(
                "gpt-oss-20b", "vllm", 262144, tps=139.2, gsm=0.85
            ),
        }
        out = bench_report.render(cache, host_vram_gb=24.0)
        # Both rows present.
        self.assertIn("| 32K |", out)
        self.assertIn("| 256K |", out)
        # Rows for the same model cluster together (32K appears before
        # 256K since they share the model name and sort by ctx).
        idx32 = out.index("| 32K |")
        idx256 = out.index("| 256K |")
        self.assertLess(idx32, idx256)

    def test_missing_ctx_renders_as_dash(self) -> None:
        # Pre-v3 row that the migrator left at ctx=0.
        cache = {
            "legacy::vllm::0": {
                "model": "legacy",
                "backend": "vllm",
                "context": 0,
                "tasks": {},
                "metrics": {"tps_sustained_p50": 80.0},
            }
        }
        out = bench_report.render(cache, host_vram_gb=24.0)
        self.assertIn("| - |", out)

    def test_missing_context_field_renders_as_dash(self) -> None:
        cache = {
            "legacy::vllm": {
                "model": "legacy",
                "backend": "vllm",
                "tasks": {},
                "metrics": {},
            }
        }
        out = bench_report.render(cache, host_vram_gb=24.0)
        self.assertIn("| - |", out)

    def test_meta_rows_skipped(self) -> None:
        cache = {
            "_meta": {"schema_version": 3, "current_host_env_id": "abc"},
            "openai/gpt-oss-20b@x::vllm::262144": _row(
                "gpt-oss-20b", "vllm", 262144, tps=139.2
            ),
        }
        out = bench_report.render(cache, host_vram_gb=24.0)
        self.assertNotIn("schema_version", out)
        # Header row + 1 data row + (no env header, _meta has no
        # host_env_history).
        self.assertEqual(out.count("| gpt-oss-20b |"), 1)

    def test_humaneval_and_plus_are_separate_columns(self) -> None:
        # Regression: a bare "humaneval_" prefix matched BOTH tasks and
        # published the HumanEval+ score in the HumanEval column.
        cache = {
            "nvidia/Nemotron@x::vllm::163840": _row(
                "Nemotron", "vllm", 163840, he=1.0, hep=0.84
            ),
        }
        out = bench_report.render(cache, host_vram_gb=24.0)
        self.assertIn("| HumanEval | HumanEval+ |", out)
        data = [ln for ln in out.splitlines() if ln.startswith("| Nemotron |")]
        self.assertEqual(len(data), 1)
        cells = [c.strip() for c in data[0].split("|")]
        # ... | Agg | GSM8K | HumanEval | HumanEval+ | ...
        self.assertIn("1.000", cells)
        self.assertIn("0.840", cells)
        self.assertLess(cells.index("1.000"), cells.index("0.840"))

    def test_aggregate_uses_plain_humaneval_only(self) -> None:
        cache = {
            "nvidia/Nemotron@x::vllm::163840": _row(
                "Nemotron", "vllm", 163840,
                gsm=1.0, he=1.0, hep=0.0, tools=1.0,
            ),
        }
        row = cache["nvidia/Nemotron@x::vllm::163840"]
        # gsm 1.0 + humaneval 1.0 + tools 1.0 -> 1.0; the 0.0 HumanEval+
        # must not drag the composite down.
        self.assertEqual(bench_report._aggregate(row), 1.0)

    def test_env_column_lists_mixed_task_provenance(self) -> None:
        row = _row("Nemotron", "vllm", 163840, he=1.0)
        row["tasks"]["humaneval_subset_50"]["host_env_id"] = "abc12345"
        row["tasks"]["gpqa_subset_100"] = {
            "score": 0.7,
            "host_env_id": "zz9999999",
        }
        out = bench_report.render(
            {"nvidia/Nemotron@x::vllm::163840": row}, host_vram_gb=24.0
        )
        self.assertIn("abc12345, zz9999999", out)

    def test_env_column_single_id_when_uniform(self) -> None:
        row = _row("Nemotron", "vllm", 163840, he=1.0)
        row["tasks"]["humaneval_subset_50"]["host_env_id"] = "abc12345"
        out = bench_report.render(
            {"nvidia/Nemotron@x::vllm::163840": row}, host_vram_gb=24.0
        )
        self.assertIn("| abc12345 |", out)
        self.assertNotIn("abc12345,", out)

    def test_v3_footer_note_present(self) -> None:
        cache = {
            "openai/gpt-oss-20b@x::vllm::262144": _row(
                "gpt-oss-20b", "vllm", 262144, tps=139.2
            ),
        }
        out = bench_report.render(cache, host_vram_gb=24.0)
        self.assertIn("Schema v3", out)
        self.assertIn("make bench --ctx", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
