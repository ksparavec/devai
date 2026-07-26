"""Unit tests for bench_runner ctx-tier handling.

Covers:
  - _parse_ctx_token / parse_ctx_list (CLI flag parser)
  - _fitting_ctxs (probe-cache fit projection)
  - discover_models (against an injected fake probe cache)

Stdlib unittest only. discover_models calls load_cache() which reads
JSON off disk; the tests monkey-patch the loader to return a synthetic
probe-cache dict so nothing on disk is touched.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bench import bench_runner  # noqa: E402


class TestParseCtxToken(unittest.TestCase):
    def test_kilo_suffix(self) -> None:
        self.assertEqual(bench_runner._parse_ctx_token("32K"), 32768)
        self.assertEqual(bench_runner._parse_ctx_token("128k"), 131072)
        self.assertEqual(bench_runner._parse_ctx_token("256K"), 262144)

    def test_raw_int(self) -> None:
        self.assertEqual(bench_runner._parse_ctx_token("65536"), 65536)
        self.assertEqual(bench_runner._parse_ctx_token(" 8192 "), 8192)

    def test_arbitrary_kilo(self) -> None:
        # Catches the case where someone passes an unfamiliar tier.
        self.assertEqual(bench_runner._parse_ctx_token("17K"), 17 * 1024)

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            bench_runner._parse_ctx_token("")

    def test_garbage_raises(self) -> None:
        with self.assertRaises(ValueError):
            bench_runner._parse_ctx_token("not-a-number")


class TestParseCtxList(unittest.TestCase):
    def test_empty_string(self) -> None:
        self.assertEqual(bench_runner.parse_ctx_list(""), [])

    def test_single(self) -> None:
        self.assertEqual(bench_runner.parse_ctx_list("32K"), [32768])

    def test_multiple_tiers(self) -> None:
        self.assertEqual(
            bench_runner.parse_ctx_list("32K,128K,256K"),
            [32768, 131072, 262144],
        )

    def test_mixed_int_and_kilo(self) -> None:
        self.assertEqual(
            bench_runner.parse_ctx_list("32768,128K"), [32768, 131072]
        )

    def test_extra_commas_ignored(self) -> None:
        self.assertEqual(
            bench_runner.parse_ctx_list(",32K,,64K,"), [32768, 65536]
        )


class TestFittingCtxs(unittest.TestCase):
    def test_no_band_at_vram(self) -> None:
        entry = {"probes": {"24": {}}}
        self.assertEqual(bench_runner._fitting_ctxs(entry, 16), [])

    def test_only_failing_cells(self) -> None:
        entry = {
            "probes": {
                "24": {
                    "32768": {"fits": False},
                    "65536": {"fits": False},
                }
            }
        }
        self.assertEqual(bench_runner._fitting_ctxs(entry, 24), [])

    def test_hf_fits_field(self) -> None:
        entry = {
            "probes": {
                "24": {
                    "32768": {"fits": True},
                    "65536": {"fits": False},
                    "131072": {"fits": True},
                }
            }
        }
        self.assertEqual(bench_runner._fitting_ctxs(entry, 24), [32768, 131072])

    def test_ollama_fully_on_gpu(self) -> None:
        entry = {
            "probes": {
                "24": {
                    "32768": {"fully_on_gpu": True},
                    "131072": {"fully_on_gpu": False},
                }
            }
        }
        self.assertEqual(bench_runner._fitting_ctxs(entry, 24), [32768])

    def test_invalid_ctx_keys_skipped(self) -> None:
        entry = {
            "probes": {
                "24": {
                    "junk": {"fits": True},
                    "32768": {"fits": True},
                }
            }
        }
        self.assertEqual(bench_runner._fitting_ctxs(entry, 24), [32768])


_PROBE_FIXTURE = {
    "openai/gpt-oss-20b@deadbeef": {
        "repo": "openai/gpt-oss-20b",
        "sha": "deadbeef",
        "aliases": ["gpt-oss-20b"],
        "capability": "structured",
        "probes": {
            "24": {
                "32768":  {"fits": True},
                "65536":  {"fits": True},
                "131072": {"fits": True},
                "262144": {"fits": True},
            }
        },
    },
    "nvidia/Qwen3-8B-NVFP4@cafebabe": {
        "repo": "nvidia/Qwen3-8B-NVFP4",
        "sha": "cafebabe",
        "aliases": ["Qwen3-8B-NVFP4"],
        "capability": "inline",
        "probes": {
            "24": {
                "32768":  {"fits": True},
                "131072": {"fits": True},
                "262144": {"fits": False},
            }
        },
    },
    "nvidia/Llama-3.1-8B-Instruct-NVFP4@feedface": {
        "repo": "nvidia/Llama-3.1-8B-Instruct-NVFP4",
        "sha": "feedface",
        "aliases": ["Llama-3.1-8B-Instruct-NVFP4"],
        "capability": "structured",
        "probes": {
            "24": {
                "131072": {"fits": True},
            }
        },
    },
    # Architecture-rejected entry; should be filtered.
    "broken/UnsupportedArch@x": {
        "repo": "broken/UnsupportedArch",
        "sha": "x",
        "aliases": ["broken"],
        "capability": "unsupported_arch",
        "probes": {
            "24": {
                "32768": {"fits": True},
            }
        },
    },
}


def _patched_load_cache(_path):  # noqa: ANN001 -- match upstream signature
    return _PROBE_FIXTURE


class TestDiscoverModels(unittest.TestCase):
    def setUp(self) -> None:
        # Patch load_cache so nothing touches the on-disk probe cache.
        self.lc = mock.patch.object(
            bench_runner, "load_cache", _patched_load_cache
        )
        self.lc.start()
        # These tests are about ctx SELECTION, not weight presence, and
        # their fixture models are invented -- so they must not be graded
        # against the real weight stores. Pointing the stores at a path
        # that does not exist makes weights_present fail open, which is
        # the documented behaviour when a store is not mounted.
        # See tests/python/test_bench_store_gap.py for the check itself.
        self.stores = mock.patch.dict(
            bench_runner.HF_WEIGHT_STORE_BY_BACKEND,
            {"vllm": Path("/nonexistent/vllm-store"),
             "sglang": Path("/nonexistent/sglang-store")},
        )
        self.stores.start()

    def tearDown(self) -> None:
        self.stores.stop()
        self.lc.stop()

    def test_default_picks_largest_fitting_per_model(self) -> None:
        targets = bench_runner.discover_models(
            "vllm",
            host_vram_gb=24,
            repo_filter=None,
        )
        # 3 fitting models (UnsupportedArch dropped). Default picks the
        # largest fits=true ctx per model.
        ctxs_by_alias = {t["alias"]: t["ctx"] for t in targets}
        self.assertEqual(ctxs_by_alias["gpt-oss-20b"], 262144)
        self.assertEqual(ctxs_by_alias["Qwen3-8B-NVFP4"], 131072)
        self.assertEqual(
            ctxs_by_alias["Llama-3.1-8B-Instruct-NVFP4"], 131072
        )
        self.assertNotIn("broken", ctxs_by_alias)
        # Each target must carry a v3-style cache key.
        for t in targets:
            self.assertIn("::vllm::", t["key"])

    def test_explicit_ctx_filter_intersects_with_fitting(self) -> None:
        targets = bench_runner.discover_models(
            "vllm",
            host_vram_gb=24,
            repo_filter=None,
            ctx_filter=[32768, 131072],
        )
        # gpt-oss-20b: fits at both -> 2 targets
        # Qwen3-8B-NVFP4: fits at both -> 2 targets
        # Llama-3.1-8B-Instruct-NVFP4: only fits at 128K -> 1 target
        rows = [(t["alias"], t["ctx"]) for t in targets]
        self.assertIn(("gpt-oss-20b", 32768), rows)
        self.assertIn(("gpt-oss-20b", 131072), rows)
        self.assertIn(("Qwen3-8B-NVFP4", 32768), rows)
        self.assertIn(("Qwen3-8B-NVFP4", 131072), rows)
        self.assertIn(("Llama-3.1-8B-Instruct-NVFP4", 131072), rows)
        self.assertNotIn(
            ("Llama-3.1-8B-Instruct-NVFP4", 32768), rows
        )
        self.assertEqual(len(rows), 5)

    def test_all_ctx_iterates_every_fitting_cell(self) -> None:
        targets = bench_runner.discover_models(
            "vllm",
            host_vram_gb=24,
            repo_filter=None,
            ctx_filter=[-1],
        )
        rows = sorted((t["alias"], t["ctx"]) for t in targets)
        # gpt-oss-20b: 4 ctxs; Qwen3-8B-NVFP4: 2 ctxs;
        # Llama-3.1-8B-Instruct-NVFP4: 1 ctx; broken excluded.
        self.assertEqual(len(rows), 4 + 2 + 1)
        self.assertIn(("gpt-oss-20b", 32768), rows)
        self.assertIn(("gpt-oss-20b", 262144), rows)

    def test_repo_filter_applied(self) -> None:
        targets = bench_runner.discover_models(
            "vllm",
            host_vram_gb=24,
            repo_filter="Qwen3-8B-NVFP4",
        )
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["alias"], "Qwen3-8B-NVFP4")

    def test_cache_key_carries_ctx(self) -> None:
        targets = bench_runner.discover_models(
            "vllm",
            host_vram_gb=24,
            repo_filter="gpt-oss-20b",
            ctx_filter=[131072, 262144],
        )
        keys = sorted(t["key"] for t in targets)
        self.assertEqual(
            keys,
            [
                "openai/gpt-oss-20b@deadbeef::vllm::131072",
                "openai/gpt-oss-20b@deadbeef::vllm::262144",
            ],
        )


class TestDefaultTasks(unittest.TestCase):
    """A plain `make bench` must populate every benchmark column the
    picker renders -- GPQA is its default sort column."""

    def test_default_task_set_covers_picker_columns(self) -> None:
        tasks = bench_runner.DEFAULT_TASKS.split(",")
        for name in (
            "gsm8k", "humaneval", "humaneval_plus",
            "mmlu_pro", "gpqa", "tools", "leak",
        ):
            self.assertIn(name, tasks)

    def test_longctx_stays_opt_in(self) -> None:
        # longctx is a per-context probe, not a leaderboard score.
        self.assertNotIn("longctx", bench_runner.DEFAULT_TASKS.split(","))


class TestEvaluateDropTrigger(unittest.TestCase):
    """Early-drop disqualifier logic: leak or low gsm8k/humaneval trips a
    drop; tools is excluded; passing results and partial results don't."""

    def test_leak_triggers(self) -> None:
        flag = bench_runner._evaluate_drop_trigger(
            {"leak_probe": {"leak_rate": 0.1}}, 0.70)
        self.assertIsNotNone(flag)
        self.assertEqual(flag["reason"], "leak")

    def test_low_gsm8k_triggers(self) -> None:
        flag = bench_runner._evaluate_drop_trigger(
            {"gsm8k_subset_100": {"score": 0.60}}, 0.70)
        self.assertIsNotNone(flag)
        self.assertEqual(flag["metric"], "gsm8k")
        self.assertEqual(flag["value"], 0.60)

    def test_low_humaneval_triggers(self) -> None:
        flag = bench_runner._evaluate_drop_trigger(
            {"humaneval_subset_50": {"pass@1": 0.56}}, 0.70)
        self.assertIsNotNone(flag)
        self.assertEqual(flag["metric"], "humaneval")

    def test_passing_scores_do_not_trigger(self) -> None:
        flag = bench_runner._evaluate_drop_trigger(
            {
                "leak_probe": {"leak_rate": 0.0},
                "gsm8k_subset_100": {"score": 0.96},
                "humaneval_subset_50": {"pass@1": 1.0},
            },
            0.70,
        )
        self.assertIsNone(flag)

    def test_low_tools_is_not_a_trigger(self) -> None:
        # tools is a saturated microbench + parser-artifact prone; a 0.00
        # there must not disqualify a model.
        flag = bench_runner._evaluate_drop_trigger(
            {"tools_use_20": {"score": 0.0}}, 0.70)
        self.assertIsNone(flag)

    def test_at_threshold_does_not_trigger(self) -> None:
        # Strictly-below only: exactly 0.70 is a keep.
        flag = bench_runner._evaluate_drop_trigger(
            {"gsm8k_subset_100": {"score": 0.70}}, 0.70)
        self.assertIsNone(flag)

    def test_empty_results_do_not_trigger(self) -> None:
        self.assertIsNone(bench_runner._evaluate_drop_trigger({}, 0.70))


if __name__ == "__main__":
    unittest.main(verbosity=2)
