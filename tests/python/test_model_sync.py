"""Unit tests for the model-sync diff logic (Phase 4).

Covers scripts/model-sync.py:plan_sync + is_probed -- classifying catalog
rows into new / evaluated / excluded. The execution path (download + probe
via subprocess) needs a GPU and is exercised by `make model-sync`, not here.

    python3 -m unittest tests.python.test_model_sync
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import _model_status as MS  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "model_sync", REPO_ROOT / "scripts" / "model-sync.py")
ms = importlib.util.module_from_spec(_spec)
sys.modules["model_sync"] = ms
_spec.loader.exec_module(ms)


CATALOG = [
    {"name": "Qwen3-8B-NVFP4", "repo": "nvidia/Qwen3-8B-NVFP4", "sha": "aaa",
     "backend": ["vllm", "sglang"], "family": "qwen3"},
    {"name": "Qwen3-32B-NVFP4", "repo": "nvidia/Qwen3-32B-NVFP4", "sha": "bbb",
     "backend": ["vllm", "sglang"], "family": "qwen3"},
    {"name": "gemma-4-31b-it", "repo": "google/gemma-4-31b-it", "sha": "ccc",
     "backend": ["vllm", "sglang"], "family": "gemma4"},
    {"name": "qwen3:8b-q4", "backend": ["ollama"], "family": "qwen3"},
]


class TestIsProbed(unittest.TestCase):
    def test_hf_probed_by_repo_sha(self) -> None:
        vllm = {"nvidia/Qwen3-8B-NVFP4@aaa": {"repo": "nvidia/Qwen3-8B-NVFP4"}}
        self.assertTrue(ms.is_probed(CATALOG[0], {}, vllm, {}))
        self.assertFalse(ms.is_probed(CATALOG[1], {}, vllm, {}))  # bbb not cached

    def test_hf_new_sha_not_probed_despite_stale_alias(self) -> None:
        # Realistic: the stale OLDsha entry carries the catalog name in its
        # aliases (as real caches do). The new sha must NOT be "probed" -- a
        # re-quant has to be re-onboarded (the gemma double-key bug).
        vllm = {"nvidia/Qwen3-8B-NVFP4@OLDsha": {
            "repo": "nvidia/Qwen3-8B-NVFP4", "aliases": ["Qwen3-8B-NVFP4"]}}
        self.assertFalse(ms.is_probed(CATALOG[0], {}, vllm, {}))

    def test_ollama_probed_by_alias(self) -> None:
        oll = {"digest123": {"aliases": ["qwen3:8b-q4"]}}
        self.assertTrue(ms.is_probed(CATALOG[3], oll, {}, {}))
        self.assertFalse(ms.is_probed(CATALOG[3], {"d": {"aliases": ["other"]}},
                                      {}, {}))


class TestPlanSync(unittest.TestCase):
    def test_three_way_split(self) -> None:
        # 8B: probed (evaluated). 32B: excluded both backends. gemma: new. ollama: new.
        vllm = {"nvidia/Qwen3-8B-NVFP4@aaa": {"repo": "nvidia/Qwen3-8B-NVFP4"}}
        ledger = {"models": {}}
        for b in ("vllm", "sglang"):
            MS.record_exclusion(ledger, "Qwen3-32B-NVFP4", b, "too_big",
                                host_vram=24, sha="bbb")
        plan = ms.plan_sync(CATALOG, {}, vllm, {}, ledger, host_vram=24)
        names = lambda key: {r["name"] for r in plan[key]}
        self.assertEqual(names("evaluated"), {"Qwen3-8B-NVFP4"})
        self.assertEqual(names("excluded"), {"Qwen3-32B-NVFP4"})
        self.assertEqual(names("new"), {"gemma-4-31b-it", "qwen3:8b-q4"})

    def test_partial_backend_exclusion_is_not_excluded(self) -> None:
        # Excluded on sglang only -> still a candidate (vllm may serve it).
        ledger = {"models": {}}
        MS.record_exclusion(ledger, "Qwen3-32B-NVFP4", "sglang",
                            "unsupported_arch", host_vram=24)
        plan = ms.plan_sync([CATALOG[1]], {}, {}, {}, ledger, host_vram=24)
        self.assertEqual({r["name"] for r in plan["new"]}, {"Qwen3-32B-NVFP4"})
        self.assertEqual(plan["excluded"], [])

    def test_vram_change_reopens_too_big(self) -> None:
        ledger = {"models": {}}
        for b in ("vllm", "sglang"):
            MS.record_exclusion(ledger, "Qwen3-32B-NVFP4", b, "too_big",
                                host_vram=24, sha="bbb")
        # On an 80 GB host the too_big verdict no longer applies -> new again.
        plan = ms.plan_sync([CATALOG[1]], {}, {}, {}, ledger, host_vram=80)
        self.assertEqual({r["name"] for r in plan["new"]}, {"Qwen3-32B-NVFP4"})


class TestExecuteBudget(unittest.TestCase):
    def test_max_downloads_is_a_run_total(self) -> None:
        # 5 new rows, budget 2 -> exactly 2 `model-pull NAME=` calls, not a
        # per-cell DOWNLOAD_LIMIT that could pull dozens.
        calls = []
        orig = ms._run
        ms._run = lambda cmd: (calls.append(cmd) or 0)
        try:
            plan = {"new": [{"name": f"M{i}", "backend": ["vllm"]}
                            for i in range(5)]}
            rc = ms.execute(plan, max_downloads=2)
        finally:
            ms._run = orig
        self.assertEqual(rc, 0)
        pulls = [c for c in calls if any(x.startswith("NAME=") for x in c)]
        self.assertEqual(len(pulls), 2)
        self.assertEqual({c[-1] for c in pulls}, {"NAME=M0", "NAME=M1"})


class TestDryRunMain(unittest.TestCase):
    def test_dry_run_changes_nothing(self) -> None:
        # --dry-run must never invoke execute(); patch it to detect a call.
        called = []
        orig = ms.execute
        ms.execute = lambda *a, **k: called.append(True) or 0
        try:
            rc = ms.main(["--dry-run", "--family", "nonexistent-family"])
        finally:
            ms.execute = orig
        self.assertEqual(rc, 0)
        self.assertEqual(called, [])  # execute never ran


if __name__ == "__main__":
    unittest.main()
