"""Schema v2 -> v3 migration unit tests for the bench cache.

Stdlib unittest only -- the project doesn't pull pytest, and the
bench harness is itself stdlib-only. Each test isolates against a
synthetic in-memory cache; nothing touches deploy/.bench-cache.json.

Run with: python3 -m unittest tests.python.test_bench_core_v3
or:        python3 tests/python/test_bench_core_v3.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bench._bench_core import (  # noqa: E402
    BENCH_CACHE_SCHEMA_VERSION,
    RECOVERED_CTX_MAP,
    _key_suffix_count,
    cache_key_for_entry,
    is_row_key,
    migrate_bench_cache_keys,
    update_row,
)


class TestSchemaVersionConstants(unittest.TestCase):
    def test_schema_version_is_v3(self) -> None:
        self.assertEqual(BENCH_CACHE_SCHEMA_VERSION, 3)

    def test_recovered_ctx_map_is_frozen_at_nine_entries(self) -> None:
        # Per docs/plans/bench-rewrite.md: the map captures historical
        # fact for the 9 v2 rows. Adding a 10th here would shadow the
        # "re-bench is the right answer" policy with stale ctxs.
        self.assertEqual(len(RECOVERED_CTX_MAP), 9)
        self.assertEqual(RECOVERED_CTX_MAP["gpt-oss-20b"], 262144)
        self.assertEqual(RECOVERED_CTX_MAP["Qwen3-8B-NVFP4"], 131072)
        self.assertEqual(
            RECOVERED_CTX_MAP["DeepSeek-R1-Distill-Llama-8B"], 32768
        )


class TestKeySuffixCount(unittest.TestCase):
    def test_bare_base(self) -> None:
        self.assertEqual(_key_suffix_count("openai/gpt-oss-20b@deadbeef"), 0)

    def test_v2_base_backend(self) -> None:
        self.assertEqual(
            _key_suffix_count("openai/gpt-oss-20b@deadbeef::vllm"), 1
        )

    def test_v3_base_backend_ctx(self) -> None:
        self.assertEqual(
            _key_suffix_count("openai/gpt-oss-20b@deadbeef::vllm::262144"),
            2,
        )

    def test_ollama_digest_v3(self) -> None:
        self.assertEqual(
            _key_suffix_count("sha256abc123::ollama::32768"), 2
        )


class TestCacheKeyForEntry(unittest.TestCase):
    def test_hf_v3_with_ctx(self) -> None:
        entry = {"repo": "openai/gpt-oss-20b", "sha": "6cee5e81ee83"}
        self.assertEqual(
            cache_key_for_entry(entry, "vllm", 262144),
            "openai/gpt-oss-20b@6cee5e81ee83::vllm::262144",
        )

    def test_hf_legacy_no_ctx(self) -> None:
        entry = {"repo": "openai/gpt-oss-20b", "sha": "6cee5e81ee83"}
        self.assertEqual(
            cache_key_for_entry(entry, "vllm", None),
            "openai/gpt-oss-20b@6cee5e81ee83::vllm",
        )

    def test_ollama_v3_with_ctx(self) -> None:
        entry = {"digest": "sha256abc"}
        self.assertEqual(
            cache_key_for_entry(entry, "ollama", 32768),
            "sha256abc::ollama::32768",
        )

    def test_malformed_returns_none(self) -> None:
        self.assertIsNone(cache_key_for_entry({}, "vllm", 1024))
        self.assertIsNone(cache_key_for_entry({"repo": "x"}, "vllm", 1024))


class TestIsRowKey(unittest.TestCase):
    def test_meta_excluded(self) -> None:
        self.assertFalse(is_row_key("_meta"))
        self.assertFalse(is_row_key("_anything_underscore_prefixed"))

    def test_v2_row_accepted(self) -> None:
        self.assertTrue(is_row_key("openai/gpt-oss-20b@abc::vllm"))

    def test_v3_row_accepted(self) -> None:
        self.assertTrue(is_row_key("openai/gpt-oss-20b@abc::vllm::262144"))


def _make_v2_row(model: str, backend: str = "vllm") -> dict:
    """Build a synthetic v2-shaped row."""
    return {
        "schema_version": 2,
        "model": model,
        "backend": backend,
        "router_endpoint": "http://devai-router:11435",
        "tasks": {"gsm8k_subset_100": {"score": 0.85, "n": 100}},
        "metrics": {"peak_vram_gb": 22.4, "tps_sustained_p50": 80.5},
        "first_benched_at": "2026-05-05T16:46:36+00:00",
        "last_benched_at": "2026-05-05T16:48:43+00:00",
    }


class TestMigrateV2ToV3(unittest.TestCase):
    def test_migrates_known_model_with_recovered_ctx(self) -> None:
        cache = {
            "openai/gpt-oss-20b@6cee5e81ee83::vllm": _make_v2_row(
                "gpt-oss-20b"
            ),
        }
        n = migrate_bench_cache_keys(cache)
        self.assertEqual(n, 1)
        new_key = "openai/gpt-oss-20b@6cee5e81ee83::vllm::262144"
        self.assertIn(new_key, cache)
        self.assertNotIn("openai/gpt-oss-20b@6cee5e81ee83::vllm", cache)
        self.assertEqual(cache[new_key]["context"], 262144)
        self.assertEqual(cache[new_key]["schema_version"], 3)
        self.assertNotIn("_migration_warning", cache[new_key])

    def test_unknown_model_lands_with_ctx_zero(self) -> None:
        cache = {
            "some/never-seen-model@abc123::vllm": _make_v2_row(
                "some-unrecognised-model"
            ),
        }
        n = migrate_bench_cache_keys(cache)
        self.assertEqual(n, 1)
        new_key = "some/never-seen-model@abc123::vllm::0"
        self.assertIn(new_key, cache)
        self.assertEqual(cache[new_key]["context"], 0)
        self.assertEqual(
            cache[new_key]["_migration_warning"],
            "ctx not recovered; re-bench to populate",
        )

    def test_migration_is_idempotent(self) -> None:
        cache = {
            "openai/gpt-oss-20b@6cee5e81ee83::vllm": _make_v2_row(
                "gpt-oss-20b"
            ),
            "nvidia/Qwen3-8B-NVFP4@abc::vllm": _make_v2_row("Qwen3-8B-NVFP4"),
        }
        n_first = migrate_bench_cache_keys(cache)
        n_second = migrate_bench_cache_keys(cache)
        self.assertEqual(n_first, 2)
        self.assertEqual(n_second, 0)
        self.assertEqual(
            cache["openai/gpt-oss-20b@6cee5e81ee83::vllm::262144"]["context"],
            262144,
        )

    def test_pre_v2_bare_base_promotes_through_both_layers(self) -> None:
        # A bare key with no ::backend suffix should land at the v3 form
        # (base::backend::ctx) in a single migrate call.
        row = _make_v2_row("Qwen3-8B-NVFP4")
        cache = {"nvidia/Qwen3-8B-NVFP4@abcd": row}
        n = migrate_bench_cache_keys(cache)
        # 1 v1->v2 rename + 1 v2->v3 rename
        self.assertEqual(n, 2)
        self.assertIn("nvidia/Qwen3-8B-NVFP4@abcd::vllm::131072", cache)

    def test_meta_block_skipped(self) -> None:
        cache = {
            "_meta": {"schema_version": 2, "current_host_env_id": "abc"},
            "openai/gpt-oss-20b@6cee5e81ee83::vllm": _make_v2_row(
                "gpt-oss-20b"
            ),
        }
        migrate_bench_cache_keys(cache)
        self.assertIn("_meta", cache)
        self.assertEqual(cache["_meta"]["current_host_env_id"], "abc")
        self.assertEqual(cache["_meta"]["schema_version"], 3)

    def test_v3_row_not_remigrated(self) -> None:
        cache = {
            "openai/gpt-oss-20b@x::vllm::262144": {
                "schema_version": 3,
                "model": "gpt-oss-20b",
                "backend": "vllm",
                "context": 262144,
                "router_endpoint": "http://devai-router:11435",
                "tasks": {},
                "metrics": {},
            },
        }
        n = migrate_bench_cache_keys(cache)
        self.assertEqual(n, 0)
        self.assertIn("openai/gpt-oss-20b@x::vllm::262144", cache)

    def test_explicit_context_field_overrides_recovered_map(self) -> None:
        # Row already carries context=32768 even though the model is in
        # the recovered map at 262144 -- prefer the explicit field.
        row = _make_v2_row("gpt-oss-20b")
        row["context"] = 32768
        cache = {"openai/gpt-oss-20b@x::vllm": row}
        migrate_bench_cache_keys(cache)
        self.assertIn("openai/gpt-oss-20b@x::vllm::32768", cache)
        self.assertNotIn("openai/gpt-oss-20b@x::vllm::262144", cache)


class TestUpdateRowStampsContext(unittest.TestCase):
    def test_new_row_carries_context_and_v3_schema(self) -> None:
        cache: dict = {}
        update_row(
            cache,
            "openai/gpt-oss-20b@abc::vllm::262144",
            model="gpt-oss-20b",
            backend="vllm",
            router_endpoint="http://devai-router:11435",
            context=262144,
            metrics={"peak_vram_gb": 22.4},
        )
        row = cache["openai/gpt-oss-20b@abc::vllm::262144"]
        self.assertEqual(row["context"], 262144)
        self.assertEqual(row["schema_version"], 3)
        self.assertEqual(row["metrics"]["peak_vram_gb"], 22.4)
        self.assertEqual(cache["_meta"]["schema_version"], 3)

    def test_update_preserves_existing_context(self) -> None:
        cache: dict = {}
        update_row(
            cache,
            "openai/gpt-oss-20b@abc::vllm::262144",
            model="gpt-oss-20b",
            backend="vllm",
            router_endpoint="http://devai-router:11435",
            context=262144,
            metrics={"peak_vram_gb": 22.4},
        )
        # Re-touch with no context arg -- should not zero out the field.
        update_row(
            cache,
            "openai/gpt-oss-20b@abc::vllm::262144",
            model="gpt-oss-20b",
            backend="vllm",
            router_endpoint="http://devai-router:11435",
            metrics={"tps_sustained_p50": 139.2},
        )
        row = cache["openai/gpt-oss-20b@abc::vllm::262144"]
        self.assertEqual(row["context"], 262144)
        self.assertEqual(row["metrics"]["peak_vram_gb"], 22.4)
        self.assertEqual(row["metrics"]["tps_sustained_p50"], 139.2)

    def test_update_is_immutable_merge_never_wipes_other_tasks(self) -> None:
        # Immutability foundation: writing one task must never drop the others.
        # This is what a --force re-bench of the default tasks relies on to
        # leave the sharper benches (mmlu_pro/gpqa) untouched.
        key = "bg/Gemma-4-26B-A4B-it-NVFP4@a1::vllm::262144"
        cache: dict = {}
        update_row(cache, key, model="Gemma-4-26B-A4B-it-NVFP4", backend="vllm",
                   router_endpoint="http://r", context=262144,
                   task_results={"mmlu_pro_subset_100": {"score": 0.82},
                                 "gpqa_subset_100": {"score": 0.70}})
        # A later (forced) run overwrites only gsm8k -- the sharper benches stay.
        update_row(cache, key, model="Gemma-4-26B-A4B-it-NVFP4", backend="vllm",
                   router_endpoint="http://r", context=262144,
                   task_results={"gsm8k_subset_100": {"score": 0.96}})
        tasks = cache[key]["tasks"]
        self.assertEqual(tasks["mmlu_pro_subset_100"]["score"], 0.82)
        self.assertEqual(tasks["gpqa_subset_100"]["score"], 0.70)
        self.assertEqual(tasks["gsm8k_subset_100"]["score"], 0.96)

    def test_force_success_overwrites_same_task(self) -> None:
        # The one allowed mutation: a successful re-run overwrites its own entry.
        key = "m@a1::vllm::32768"
        cache: dict = {}
        update_row(cache, key, model="m", backend="vllm", router_endpoint="http://r",
                   context=32768, task_results={"gsm8k_subset_100": {"score": 0.60}})
        update_row(cache, key, model="m", backend="vllm", router_endpoint="http://r",
                   context=32768, task_results={"gsm8k_subset_100": {"score": 0.97}})
        self.assertEqual(cache[key]["tasks"]["gsm8k_subset_100"]["score"], 0.97)


if __name__ == "__main__":
    unittest.main(verbosity=2)
