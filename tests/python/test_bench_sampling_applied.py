"""The bench's sampling settings reach the backend (defect D1).

Until 2026-09-27 the harness passed `config=GenerateConfig(...)` to
inspect_ai.eval(), which has no `config` parameter: inspect 0.3.158 folded
it into a GenerateConfig as an unknown field and pydantic dropped it, so
every scored row ran at the engine's default sampler; 0.3.271 rejects it
with a ValidationError. Sampling now goes in as eval() keywords.

And through the GENERIC OpenAI-compatible provider: inspect 0.3.271's
`openai` provider takes names like Qwen3.5-9B-NVFP4 for GPT-5 models, moves
them to /v1/responses and drops temperature/top_p (checked on the wire in
the lab image against a fake server; commit message has the details).

Stdlib unittest only; inspect_ai is faked.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "stats"))

from bench import bench_runner  # noqa: E402
import bench_stats  # noqa: E402

SOURCE = (REPO_ROOT / "scripts" / "bench" / "bench_runner.py").read_text()


def _invoke(served_model: str) -> dict:
    captured: dict = {}

    def fake_eval(task_obj, **kwargs):
        captured.update(kwargs)
        return [object()]

    fake = types.ModuleType("inspect_ai")
    fake.eval = fake_eval
    with mock.patch.dict(sys.modules, {"inspect_ai": fake}), \
            mock.patch.dict(os.environ, {}, clear=False), \
            tempfile.TemporaryDirectory() as tmp:
        bench_runner._invoke_inspect_task(
            task_obj=object(), served_model=served_model, backend="vllm",
            router_url="http://r", log_dir=Path(tmp), timeout_s=900.0)
        captured["_env_key"] = os.environ.get("DEVAI_API_KEY")
    return captured


class SamplingKeywordsTest(unittest.TestCase):
    def test_greedy_default_goes_in_as_eval_keywords(self) -> None:
        kw = _invoke("Qwen3.5-9B-NVFP4@131072")
        self.assertEqual(kw.get("temperature"), 0.0)
        self.assertEqual(kw.get("top_p"), 1.0)
        self.assertNotIn("config", kw)

    def test_override_models_get_their_sampling(self) -> None:
        kw = _invoke("NVIDIA-Nemotron-Nano-9B-v2-NVFP4@131072")
        self.assertEqual((kw.get("temperature"), kw.get("top_p")), (0.6, 0.95))

    def test_the_stamp_matches_what_is_sent(self) -> None:
        for alias in ("Qwen3.5-9B-NVFP4@131072", "NVIDIA-Nemotron-Nano-9B-v2-NVFP4@131072"):
            kw = _invoke(alias)
            rec = bench_runner.sampling_record(alias)
            self.assertEqual((rec["temperature"], rec["top_p"]),
                             (kw["temperature"], kw["top_p"]), alias)

    def test_no_config_keyword_anywhere_in_the_invocation(self) -> None:
        body = SOURCE[SOURCE.index("def _invoke_inspect_task("):SOURCE.index("def _aggregate_score(")]
        self.assertNotRegex(body, r"(?m)^\s*config\s*=")  # code, not the comment


class ProviderTest(unittest.TestCase):
    def test_generic_openai_compatible_provider(self) -> None:
        kw = _invoke("qwen3.8:27b-ud-q4_k_xl")
        self.assertEqual(kw["model"], "openai-api/devai/qwen3.8:27b-ud-q4_k_xl")
        self.assertEqual(kw["model_base_url"], "http://r/v1")

    def test_tools_are_not_sent_strict(self) -> None:
        # strict:true would turn on vLLM 0.28's grammar-constrained
        # tool-call decoding for tool_choice="auto" (vllm-devai).
        self.assertEqual(_invoke("m")["model_args"], {"strict_tools": False})

    def test_the_provider_finds_its_api_key(self) -> None:
        self.assertTrue(_invoke("m")["_env_key"])


class StatsReaderTest(unittest.TestCase):
    def test_served_name_is_recovered_from_both_provider_spellings(self) -> None:
        for logged in ("openai/Qwen3.5-9B-NVFP4@131072",
                       "openai-api/devai/Qwen3.5-9B-NVFP4@131072"):
            self.assertEqual(bench_stats.served_model_name(logged), "Qwen3.5-9B-NVFP4@131072")
        self.assertEqual(bench_stats.served_model_name("openai-api/devai/qwen3.8:27b"),
                         "qwen3.8:27b")


if __name__ == "__main__":
    unittest.main()
