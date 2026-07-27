"""sglang-backend-remediation Phase 2: degeneracy predicate + launch_args kind.

Two defects, both of which let a broken engine be certified as healthy:

1. `_probe_load.py` computed ``serving_ok = not failed``, consulting
   nothing about what the engine actually returned. Ornith-1.0-9B-NVFP4 on
   SGLang raised a CUDA device-side assert, killed its scheduler, kept
   answering 200s whose content was empty and whose reasoning_content was
   thousands of '!', and was written to the cache as
   ``serving_ok: true, fits: true, capability: structured`` at 131072.

2. `classify_failure_logs` matched the bare token "GPTQ" against the WHOLE
   log, so an argparse usage dump -- which enumerates every --quantization
   choice, including gptq and gptq_marlin -- was filed as ``kind: quant``.
   Seven SGLang cells recorded a permanent per-model quantisation verdict
   for what was really this lab sending SGLang a vLLM-only recovery flag.

Stdlib unittest only. Run with:
    python3 -m unittest tests.python.test_probe_degenerate_and_launch_args
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _probe_hf_common as P  # noqa: E402

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "sglang"


class TestIsDegenerateGeneration(unittest.TestCase):
    """The predicate must catch the real corpse and spare the real models."""

    def test_ornith_shape_is_degenerate(self):
        # The actual recorded failure: empty content, reasoning_content a
        # long run of '!', finish_reason "length", output at the 2048 cap.
        ok, reason = P.is_degenerate_generation(
            content="",
            reasoning="!" * 4000,
            finish_reason="length",
            output_tokens=2048,
            output_cap=2048,
        )
        self.assertTrue(ok, "the recorded Ornith response must be degenerate")
        self.assertTrue(reason, "a degenerate verdict must explain itself")

    def test_healthy_reasoning_model_is_not_degenerate(self):
        # Qwen3.5-9B: 477 coherent output tokens, well under the cap. This
        # is the case the predicate must NOT reject -- gating on
        # needle_score would have, at a 100% false-positive rate on this
        # fleet's reasoning models.
        ok, _ = P.is_degenerate_generation(
            content="The needle is 8675309. I checked the haystack carefully.",
            reasoning="Let me search the document for the number..." * 20,
            finish_reason="stop",
            output_tokens=477,
            output_cap=2048,
        )
        self.assertFalse(ok)

    def test_empty_content_below_cap_is_not_degenerate(self):
        # A reasoning model that answered inside <think> and stopped early
        # is legitimate. BOTH conditions are required.
        ok, _ = P.is_degenerate_generation(
            content="",
            reasoning="thinking about it, the answer is 42",
            finish_reason="stop",
            output_tokens=300,
            output_cap=2048,
        )
        self.assertFalse(ok, "empty content alone must not condemn a cell")

    def test_full_budget_with_real_content_is_not_degenerate(self):
        # Hitting the cap is normal for a verbose model. Content is present,
        # so it is generating.
        ok, _ = P.is_degenerate_generation(
            content="A long but genuine answer. " * 200,
            reasoning="",
            finish_reason="length",
            output_tokens=2048,
            output_cap=2048,
        )
        self.assertFalse(ok)

    def test_repeated_character_run_caught_below_cap(self):
        # The same corpse, terminating before the cap.
        ok, reason = P.is_degenerate_generation(
            content="!" * 500,
            reasoning="",
            finish_reason="stop",
            output_tokens=120,
            output_cap=2048,
        )
        self.assertTrue(ok)
        self.assertIn("repeated", reason)

    def test_short_punctuation_answer_is_not_degenerate(self):
        # "!!!" is a legitimate (if unhelpful) short answer. Below the
        # minimum-chars floor, a run proves nothing.
        ok, _ = P.is_degenerate_generation(
            content="!!!",
            reasoning="",
            finish_reason="stop",
            output_tokens=3,
            output_cap=2048,
        )
        self.assertFalse(ok)

    def test_missing_values_do_not_crash(self):
        ok, _ = P.is_degenerate_generation(None, None, None, None, None)
        self.assertFalse(ok)


class TestLaunchArgsClassification(unittest.TestCase):
    """An argparse rejection is our defect, not a verdict about the model."""

    def _fixture(self) -> str:
        return (FIXTURES / "argparse-rejection.log").read_text()

    def test_real_argparse_dump_is_launch_args_not_quant(self):
        # The exact regression: this fixture is the recorded SGLang usage
        # dump for Qwen3-Coder-30B-A3B-Instruct-FP4, with the real
        # --quantization choices line from the pinned image restored (the
        # cache's 120-line excerpt had truncated it, which is why the cell
        # recorded matched_pattern "GPTQ" with no GPTQ visible in it).
        logs = self._fixture()
        self.assertIn("gptq", logs.lower(), "fixture must reproduce the trap")
        res = P.classify_failure_logs(logs)
        self.assertEqual(res["kind"], "launch_args",
                         f"argparse rejection misfiled as {res['kind']!r}")
        self.assertNotEqual(res["kind"], "quant")

    def test_gptq_in_help_text_alone_does_not_match_quant(self):
        # Negative control the plan asks for explicitly: the token appearing
        # in help text must not be a quantisation verdict.
        logs = ("usage: sglang serve [-h]\n"
                "  [--quantization {awq,fp8,gptq,gptq_marlin,awq_marlin}]\n"
                "INFO: server ready\n")
        res = P.classify_failure_logs(logs)
        self.assertNotEqual(res["kind"], "quant")

    def test_genuine_quant_failure_still_classifies_as_quant(self):
        # The tightened patterns must not blind the classifier to a real one.
        logs = "ValueError: GPTQ quantization is not supported on this GPU\n"
        res = P.classify_failure_logs(logs)
        self.assertEqual(res["kind"], "quant")

    def test_launch_args_wins_over_arch_patterns(self):
        # A usage dump names architectures too. launch_args is checked
        # first precisely so it cannot be outvoted by that vocabulary.
        logs = ("usage: sglang serve\n"
                "Model architectures ['FooForCausalLM'] are not supported\n"
                "sglang serve: error: unrecognized arguments: --enforce-eager\n")
        self.assertEqual(P.classify_failure_logs(logs)["kind"], "launch_args")

    def test_launch_args_has_severity_zero(self):
        # The load-bearing property: severity 0 means refresh_* never
        # promotes it to Capability.ERROR, which the router treats as
        # terminal. A positive severity here writes a model off permanently
        # for a flag WE chose to send.
        entry = {
            "probes": {"24": {"131072": {
                "capability": None,
                "evidence": {"kind": "launch_args", "matched_pattern":
                             "unrecognized arguments"},
            }}},
        }
        P.refresh_top_level_from_cells(entry)
        self.assertNotIn(entry.get("capability"),
                         (P.Capability.ERROR, P.Capability.UNSUPPORTED_ARCH),
                         "launch_args must never write a terminal capability")


if __name__ == "__main__":
    unittest.main()
