"""scripts/stats/usecase_scores.py: benchmark items regrouped into use cases.

Synthetic runs and cache rows only; stdlib unittest.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "stats"))

import usecase_scores as U  # noqa: E402

MMLU_CATEGORIES = {"math", "physics", "chemistry", "law", "engineering", "other", "economics",
                   "health", "psychology", "business", "biology", "philosophy",
                   "computer science", "history"}


def _run(task, model, backend, items, completed="2026-09-28T10:00:00+00:00"):
    return {"file": f"{task}-{model}-{backend}.eval", "created": completed, "completed": completed,
            "task": task, "model": model, "backend": backend, "n": len(items), "items": items,
            "status": "success"}


def _item(i, y, **kw):
    return {"id": str(i), "value": 1.0 if y else 0.0, "item_key": f"k{i}", **kw}


def _row(model, backend, tasks, ctx=131072, tps=50.0):
    return {"model": model, "backend": backend, "context": ctx,
            "tasks": {k: {"n": n, "ran_at": "2026-09-28T10:00:00+00:00"} for k, n in tasks.items()},
            "metrics": {"tps_sustained_p50": tps}}


class MappingTest(unittest.TestCase):
    def test_every_mmlu_pro_category_is_used_exactly_once(self) -> None:
        used = [c for comps in U.USE_CASES.values() for task, cats in comps
                if task == "mmlu_pro" for c in cats]
        self.assertEqual(sorted(used), sorted(MMLU_CATEGORIES))

    def test_no_whole_benchmark_is_in_two_use_cases(self) -> None:
        whole = [task for comps in U.USE_CASES.values() for task, cats in comps if cats is None]
        self.assertEqual(len(whole), len(set(whole)))

    def test_base_model_groups_quantisations_and_backends(self) -> None:
        for a in ("qwen3.8:27b-mtp-q4_K_M", "Qwen3.8-27B-MTP-devai-NVFP4",
                  "Qwen3.8-27B-W4A16-devai-AutoRound", "Qwen3.8-27B-MTP-NVFP4"):
            self.assertEqual(U.base_model(a), "Qwen3.8-27B", a)
        self.assertEqual(U.base_model("gemma4:26b-a4b-it-q4_K_M"), "Gemma-4-26B-A4B")
        self.assertEqual(U.base_model("Gemma-4-26B-A4B-it-NVFP4"), "Gemma-4-26B-A4B")
        self.assertNotEqual(U.base_model("qwen3.6:35b-a3b-mtp-q4_K_M"), "Qwen3.8-27B")


class ScoreTest(unittest.TestCase):
    def test_humaneval_and_plus_share_a_cluster_per_problem(self) -> None:
        he = U.items_for("humaneval", _run("humaneval", "m", "vllm",
                         [_item(1, 1, task_id="HumanEval/0")]), None)
        hp = U.items_for("humaneval_plus", _run("humaneval_plus", "m", "vllm",
                         [_item(1, 0, task_id="HumanEval/0")]), None)
        self.assertEqual(he[0]["cluster"], hp[0]["cluster"])
        self.assertNotEqual(he[0]["key"], hp[0]["key"])

    def test_pooled_score_and_clopper_pearson_without_clusters(self) -> None:
        items = [{"cluster": f"c{i}", "key": f"k{i}", "y": int(i < 8), "timeout": i == 9}
                 for i in range(10)]
        s = U.score(items)
        self.assertEqual((s["x"], s["n"], s["timeouts"]), (8, 10, 1))
        self.assertAlmostEqual(s["upper_if_timeouts_right"], 0.9)
        self.assertEqual(s["ci_method"], "Clopper-Pearson")

    def test_clustered_items_use_the_cluster_bootstrap(self) -> None:
        items = [{"cluster": f"c{i // 2}", "key": f"k{i}", "y": i % 3 != 0, "timeout": False}
                 for i in range(20)]
        s = U.score(items)
        self.assertTrue(s["ci_method"].startswith("cluster bootstrap"))
        self.assertLessEqual(s["ci"][0], s["score"])
        self.assertLessEqual(s["score"], s["ci"][1])

    def test_mmlu_category_filter(self) -> None:
        run = _run("mmlu_pro", "m", "vllm", [_item(1, 1, category="law"), _item(2, 1, category="math")])
        got = U.items_for("mmlu_pro", run, {"law"})
        self.assertEqual([g["key"] for g in got], ["mmlu_pro:k1"])

    def test_time_outs_are_recognised(self) -> None:
        self.assertTrue(U.is_timeout({"type": "working", "limit": 900}))
        self.assertFalse(U.is_timeout({"type": "message", "limit": 20}))
        self.assertFalse(U.is_timeout(None))


class RankTest(unittest.TestCase):
    def _fixture(self):
        def ga(model, backend, correct):
            return _run("gpqa", model, backend, [_item(i, i < correct) for i in range(10)])

        def bio(model, backend, correct):
            return _run("mmlu_pro", model, backend,
                        [_item(100 + i, i < correct, category="biology") for i in range(4)])
        runs = [ga("qwen3.8:27b-mtp-q4_K_M@131072", "ollama", 9), bio("qwen3.8:27b-mtp-q4_K_M@131072", "ollama", 4),
                ga("Qwen3.8-27B-MTP-devai-NVFP4@118784", "vllm-devai", 8), bio("Qwen3.8-27B-MTP-devai-NVFP4@118784", "vllm-devai", 4),
                ga("gpt-oss-20b@131072", "vllm", 6), bio("gpt-oss-20b@131072", "vllm", 3),
                ga("gemma-missing-bio", "vllm", 10)]
        cache = {"a": _row("qwen3.8:27b-mtp-q4_K_M", "ollama", {"gpqa_subset_10": 10, "mmlu_pro_subset_4": 4}),
                 "b": _row("Qwen3.8-27B-MTP-devai-NVFP4", "vllm-devai", {"gpqa_subset_10": 10, "mmlu_pro_subset_4": 4}, ctx=118784),
                 "c": _row("gpt-oss-20b", "vllm", {"gpqa_subset_10": 10, "mmlu_pro_subset_4": 4}),
                 "d": _row("gemma-missing-bio", "vllm", {"gpqa_subset_10": 10}),
                 "_meta": {}}
        return runs, cache

    def test_runner_up_is_the_best_row_of_another_base_model(self) -> None:
        runs, cache = self._fixture()
        res = U.rank_use_case(runs, cache, "complex_systems")
        self.assertEqual(res["winner"]["model"], "qwen3.8:27b-mtp-q4_K_M")
        self.assertEqual(res["runner_up"]["model"], "gpt-oss-20b")
        self.assertEqual([s["model"] for s in res["winner_vs_same_base"]], ["Qwen3.8-27B-MTP-devai-NVFP4"])

    def test_a_row_missing_a_component_is_not_ranked(self) -> None:
        runs, cache = self._fixture()
        res = U.rank_use_case(runs, cache, "complex_systems")
        self.assertIn("gemma-missing-bio", [u["model"] for u in res["unranked"]])
        self.assertNotIn("gemma-missing-bio", [r["model"] for r in res["rows"]])

    def test_comparison_is_paired_on_shared_items(self) -> None:
        runs, cache = self._fixture()
        res = U.rank_use_case(runs, cache, "complex_systems")
        c = res["winner_vs_runner_up"]
        self.assertEqual(c["shared"], 14)
        self.assertAlmostEqual(c["diff"], (13 - 9) / 14)

    def test_an_explicit_log_link_wins_and_a_truncated_task_counts_its_prefix_only(self) -> None:
        # A task stopped at its deadline: cancelled log, 10 questions in it,
        # the cache entry scored on the unbroken prefix 1..6.
        runs = [dict(_run("gpqa", "m@131072", "vllm", [_item(i, i % 2) for i in range(1, 11)]),
                     file="cut.eval", status="cancelled"),
                _run("mmlu_pro", "m@131072", "vllm", [_item(100, 1, category="biology")])]
        row = _row("m", "vllm", {"gpqa_subset_100": 6, "mmlu_pro_subset_1": 1})
        row["tasks"]["gpqa_subset_100"].update({"inspect_log": "cut.eval", "truncated": {"prefix": 6}})
        items, comps = U.row_items(runs, row, "complex_systems")
        self.assertEqual(comps["gpqa"], [3, 6])          # questions 1..6: 1, 3, 5 right
        self.assertEqual(len(items), 7)

    def test_markdown_renders(self) -> None:
        runs, cache = self._fixture()
        text = U.md([U.rank_use_case(runs, cache, uc) for uc in U.USE_CASES])
        self.assertIn("## Complex systems", text)
        self.assertIn("**Winner:** qwen3.8:27b-mtp-q4_K_M", text)
        self.assertEqual(sum(ord(ch) > 127 for ch in text), 0)


if __name__ == "__main__":
    unittest.main()
