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


class TestKVDtypeValidation(unittest.TestCase):
    """PROBE_KV_CACHE_TYPE is one knob shared by two engines that accept
    different spellings. vLLM takes a bare `fp8`; SGLang does not.
    """

    def _spec(self, name: str, allowed: tuple[str, ...]):
        return P.BackendSpec(
            name=name, image="i", container_name="c", probe_port=1,
            cache_path=Path("/tmp/unused"), reserve_gb=3.0,
            entrypoint="python3", build_args=lambda *a, **k: [],
            allowed_kv_dtypes=allowed,
        )

    def test_sglang_rejects_bare_fp8_with_a_usable_hint(self):
        spec = self._spec("sglang", ("auto", "fp8_e5m2", "fp8_e4m3", "bf16",
                                     "bfloat16", "fp4_e2m1"))
        with self.assertRaises(SystemExit) as ctx:
            P.validate_kv_dtype(spec, "fp8")
        msg = str(ctx.exception)
        self.assertIn("fp8_e4m3", msg, "must name the correct spelling")
        self.assertIn("sglang", msg, "must name the backend")

    def test_vllm_accepts_bare_fp8(self):
        spec = self._spec("vllm", ("auto", "fp8", "fp8_e4m3", "fp8_e5m2"))
        P.validate_kv_dtype(spec, "fp8")  # must not raise

    def test_empty_and_none_are_the_engine_default(self):
        spec = self._spec("sglang", ("auto", "fp8_e4m3"))
        P.validate_kv_dtype(spec, None)
        P.validate_kv_dtype(spec, "")

    def test_no_allowed_set_disables_validation(self):
        # Backward compatible: a spec that has not declared its set yet
        # must not start rejecting dtypes that used to work.
        spec = self._spec("legacy", ())
        P.validate_kv_dtype(spec, "anything")


class TestLaunchFingerprint(unittest.TestCase):
    """The fingerprint must change when the LAUNCH SHAPE changes, and not
    when the model, context or memory fraction does -- otherwise it is
    noise and nobody will act on it.
    """

    _BASE = ["-m", "sglang.launch_server",
             "--model-path", "/models/A", "--served-model-name", "A",
             "--tp-size", "1", "--mem-fraction-static", "0.8750",
             "--context-length", "131072", "--trust-remote-code"]

    def test_per_cell_values_are_elided(self):
        other = ["-m", "sglang.launch_server",
                 "--model-path", "/models/B", "--served-model-name", "B",
                 "--tp-size", "1", "--mem-fraction-static", "0.9100",
                 "--context-length", "32768", "--trust-remote-code"]
        self.assertEqual(P.launch_fingerprint(self._BASE),
                         P.launch_fingerprint(other),
                         "model, ctx and mem-fraction must not affect the hash")

    def test_flag_rename_changes_the_fingerprint(self):
        # The exact change this phase made: --tp -> --tp-size.
        renamed = [t.replace("--tp-size", "--tp") for t in self._BASE]
        self.assertNotEqual(P.launch_fingerprint(self._BASE),
                            P.launch_fingerprint(renamed))

    def test_added_flag_changes_the_fingerprint(self):
        added = self._BASE + ["--disable-piecewise-cuda-graph"]
        self.assertNotEqual(P.launch_fingerprint(self._BASE),
                            P.launch_fingerprint(added))

    def test_parser_choice_changes_the_fingerprint(self):
        a = self._BASE + ["--reasoning-parser", "qwen3"]
        b = self._BASE + ["--reasoning-parser", "gpt-oss"]
        self.assertNotEqual(P.launch_fingerprint(a), P.launch_fingerprint(b),
                            "a different parser IS a different launch")

    def test_is_stable_and_short(self):
        fp = P.launch_fingerprint(self._BASE)
        self.assertEqual(fp, P.launch_fingerprint(list(self._BASE)))
        self.assertEqual(len(fp), 12)


class TestImpliedVramExclusion(unittest.TestCase):
    """A VRAM verdict measured on a roomier backend applies to a tighter
    one, so SGLang need not rediscover it. One-way and reason-scoped.
    """

    def setUp(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import _model_status
        self.MS = _model_status

    def _ledger(self, backend: str, reason: str, vram=24.0):
        led: dict = {}
        self.MS.record_exclusion(led, "M", backend, reason,
                                 detail="t", host_vram=vram, sha="s1")
        return led

    def test_vllm_oom_implies_sglang(self):
        led = self._ledger("vllm", "oom")
        self.assertEqual(
            self.MS.implied_vram_exclusion(led, "M", "sglang", host_vram=24.0,
                                           sha="s1"),
            ("vllm", "oom"))

    def test_vllm_too_big_implies_sglang(self):
        led = self._ledger("vllm", "too_big")
        got = self.MS.implied_vram_exclusion(led, "M", "sglang", host_vram=24.0)
        self.assertEqual(got, ("vllm", "too_big"))

    def test_manual_does_not_propagate(self):
        # Load-bearing: Qwen3-8B-NVFP4 and Qwen3-14B-NVFP4 both carry a
        # vLLM `manual` exclusion AND fit on SGLang, where they serve.
        # Propagating `manual` would delete working rows.
        led = self._ledger("vllm", "manual")
        self.assertIsNone(
            self.MS.implied_vram_exclusion(led, "M", "sglang", host_vram=24.0))

    def test_unsupported_arch_does_not_propagate(self):
        # Engine-specific by definition -- the two engines do not support
        # the same architecture set, which is why the ledger is keyed per
        # backend in the first place.
        led = self._ledger("vllm", "unsupported_arch")
        self.assertIsNone(
            self.MS.implied_vram_exclusion(led, "M", "sglang", host_vram=24.0))

    def test_direction_is_one_way(self):
        # SGLang failing says NOTHING about vLLM, which has more room.
        led = self._ledger("sglang", "oom")
        self.assertIsNone(
            self.MS.implied_vram_exclusion(led, "M", "vllm", host_vram=24.0))

    def test_inherits_the_source_stability_rules(self):
        # An implied exclusion must never outlive the verdict it derives
        # from: a different host VRAM re-derives, and a new sha re-checks.
        led = self._ledger("vllm", "oom", vram=24.0)
        self.assertIsNone(
            self.MS.implied_vram_exclusion(led, "M", "sglang", host_vram=48.0),
            "a VRAM change must re-derive, not inherit")
        self.assertIsNone(
            self.MS.implied_vram_exclusion(led, "M", "sglang", host_vram=24.0,
                                           sha="s2"),
            "a re-quant must re-check an oom verdict")

    def test_absent_source_entry_is_not_an_exclusion(self):
        self.assertIsNone(
            self.MS.implied_vram_exclusion({}, "M", "sglang", host_vram=24.0))
