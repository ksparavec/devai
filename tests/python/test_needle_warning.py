"""needle_score: repaired, surfaced as a warning, never a gate.

The LOAD probe buries one needle at mid-depth in a nearly-full context
window and checks whether the model repeats it back. Nothing read the
result, and an earlier note in docs/plans/README.md proposed gating on
it. Gating would have been actively harmful.

Measured on this fleet 2026-07-26: of 23 load-probed cells, 4 scored 0.0,
and 3 of those terminated at exactly serving_output_tokens == 2048 -- the
output ceiling -- while none of the 19 cells scoring 1.0 came within 4x
of it. All four zeros were reasoning models that spent the answer budget
on a <think> trace. A gate would have hidden both DeepSeek-R1-Distill
models on a 100% false-positive rate.

The field was also confounded at source: `needle = 0.0 if failed else
...` meant every serving failure manufactured a recall verdict
indistinguishable from a real one.

So: repair the measurement (None on failure, plus a needle_valid flag
that rejects the two known artefacts), surface it as a WARNING in the
picker preview, and gate on nothing.

Stdlib unittest only.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load(name: str, rel: str):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mp = _load("model_picker", "scripts/model-picker.py")
BAND = str(int(mp._VRAM_BUDGET))
CTX = 262144


def _model(**cell) -> dict:
    return {"vram": {"probes": {BAND: {str(CTX): cell}}}}


class NeedleWarningGateTest(unittest.TestCase):
    def test_warns_on_a_trustworthy_failure(self):
        self.assertTrue(mp._needle_failed_at(
            _model(needle_valid=True, needle_score=0.0), CTX))

    def test_silent_when_recall_succeeded(self):
        self.assertFalse(mp._needle_failed_at(
            _model(needle_valid=True, needle_score=1.0), CTX))

    def test_silent_on_the_output_truncation_artefact(self):
        """finish_reason == length: the model never reached the answer, so
        a missing needle says nothing about recall. 3 of this fleet's 4
        zeros are this case."""
        self.assertFalse(mp._needle_failed_at(
            _model(needle_valid=False, needle_score=0.0,
                   serving_finish_reason="length"), CTX))

    def test_silent_on_an_underfilled_window(self):
        """SGLang has no /tokenize and calibrates chars-per-token, landing
        as low as 0.75 fill. A needle at depth 0.5 of that sits near 37%
        of the ADVERTISED context -- it never tested the long window."""
        self.assertFalse(mp._needle_failed_at(
            _model(needle_valid=False, needle_score=0.0,
                   serving_fill_ratio=0.748), CTX))

    def test_silent_on_legacy_cells_without_the_flag(self):
        """Cells written before needle_valid existed carry no vouching, so
        they must not produce warnings retroactively."""
        self.assertFalse(mp._needle_failed_at(_model(needle_score=0.0), CTX))

    def test_silent_when_the_serve_itself_failed(self):
        """A failed serve produces no answer to score. needle is None, not
        0.0 -- that confound is the whole reason this field was unreadable."""
        self.assertFalse(mp._needle_failed_at(
            _model(needle_valid=False, needle_score=None), CTX))

    def test_scoped_to_the_offered_context(self):
        m = _model(needle_valid=True, needle_score=0.0)
        self.assertFalse(mp._needle_failed_at(m, 131072))

    def test_tolerates_a_missing_probe_block(self):
        self.assertFalse(mp._needle_failed_at({}, CTX))
        self.assertFalse(mp._needle_failed_at({"vram": {}}, CTX))


class NeedleIsNeverAGateTest(unittest.TestCase):
    """The warning must not become an eligibility filter. If any of these
    start reading needle_score, a one-sample measurement at the hardest
    depth begins hiding models from the operator."""

    def test_no_consumer_gates_on_needle_score(self):
        gating_sites = [
            REPO_ROOT / "scripts" / "select-models.py",
            REPO_ROOT / "scripts" / "bench" / "bench_runner.py",
            REPO_ROOT / "gpu-arbiter" / "main.go",
        ]
        for path in gating_sites:
            with self.subTest(file=path.name):
                self.assertNotIn(
                    "needle_score", path.read_text(),
                    f"{path.name} reads needle_score -- it must not gate on a "
                    f"single-sample recall probe")


class ProbeServeConcurrencyParityTest(unittest.TestCase):
    """The probers must launch with the same concurrent-sequence cap the
    router serves with.

    Both engines size their CUDA-graph capture set (and vLLM its
    memory-profiling dummy forward) off this value, so a probe that omits
    it measures a different VRAM reservation than the one that actually
    serves -- while both arg builders' docstrings claimed they mirrored
    the router's entrypoint. Found 2026-07-26.
    """

    def setUp(self):
        self.v = _load("probe_vllm", "scripts/probe-vllm-reasoning.py")
        self.s = _load("probe_sglang", "scripts/probe-sglang-reasoning.py")

    def test_vllm_prober_emits_max_num_seqs(self):
        args = self.v.vllm_command_args("M", 131072, 0.88)
        self.assertIn("--max-num-seqs", args)

    def test_sglang_prober_emits_max_running_requests(self):
        args = self.s.sglang_command_args("M", 131072, 0.88)
        self.assertIn("--max-running-requests", args)

    def test_both_default_to_the_router_admission_limit(self):
        """MAX_CONCURRENT_REQUESTS defaults to 32 in gpu-arbiter."""
        self.assertEqual(self.v.MAX_NUM_SEQS, 32)
        self.assertEqual(self.s.MAX_RUNNING_REQUESTS, 32)

    def test_flag_precedes_parser_flags(self):
        """Recovery flags are appended after, so a per-model override must
        still win last -- same ordering the router uses."""
        args = self.v.vllm_command_args(
            "M", 131072, 0.88, reasoning_parser="qwen3", tool_parser="qwen3_xml")
        self.assertLess(args.index("--max-num-seqs"),
                        args.index("--reasoning-parser"))

    def test_zero_omits_the_flag(self):
        """Mirrors the router's `if lc.MaxNumSeqs > 0` guard, so an
        operator can fall back to the engine default."""
        orig = self.v.MAX_NUM_SEQS
        self.v.MAX_NUM_SEQS = 0
        self.addCleanup(setattr, self.v, "MAX_NUM_SEQS", orig)
        self.assertNotIn("--max-num-seqs",
                         self.v.vllm_command_args("M", 131072, 0.88))


if __name__ == "__main__":
    unittest.main()
