"""SGLang bench candidacy is gated on vLLM having benched it first.

SGLang needs strictly MORE VRAM than vLLM for the same (model, ctx) on
this fleet -- reserve 3.0 GB vs 2.0, and an unstamped cell runs the
engine default (unquantized KV, 2 bytes) where vLLM launches fp8 (1
byte). So a model vLLM could not serve cannot pass on SGLang, and
benching it there spends a cold start plus a full task sweep to
rediscover a known answer.

The gate is CODED here rather than documented, per the operator
decision: unprobed, unbenched and previously-dropped models must never
become SGLang bench targets.

Stdlib unittest only; no network, no container, no GPU.
"""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

runner = importlib.import_module("bench.bench_runner")
import _model_status as MS  # noqa: E402


def _bench_cache(rows):
    """rows: list of (model, backend, ctx) -> a minimal v3-shaped cache."""
    out = {"_meta": {}}
    for model, backend, ctx in rows:
        out[f"{model}@sha::{backend}::{ctx}"] = {
            "model": model, "backend": backend, "context": ctx,
        }
    return out


class TestVllmBenchCeiling(unittest.TestCase):
    def test_returns_largest_benched_ctx(self):
        c = _bench_cache([("M", "vllm", 32768), ("M", "vllm", 262144)])
        self.assertEqual(runner.vllm_bench_ceiling("M", c), 262144)

    def test_ignores_other_backends(self):
        c = _bench_cache([("M", "sglang", 262144), ("M", "ollama", 131072)])
        self.assertIsNone(runner.vllm_bench_ceiling("M", c),
                          "only vLLM rows may vouch for SGLang")

    def test_unknown_model_is_none(self):
        self.assertIsNone(runner.vllm_bench_ceiling("Nope", _bench_cache([])))

    def test_skips_meta_block(self):
        c = _bench_cache([("M", "vllm", 32768)])
        c["_meta"] = {"host_env_history": {"x": {}}}
        self.assertEqual(runner.vllm_bench_ceiling("M", c), 32768)


class TestGateReason(unittest.TestCase):
    def test_vllm_and_ollama_are_ungated(self):
        for backend in ("vllm", "ollama"):
            self.assertIsNone(
                runner.gate_reason(backend, "M", 131072,
                                   bench_cache=_bench_cache([]), ledger={}),
                f"{backend} must not be gated on any other backend")

    def test_sglang_blocked_when_vllm_never_benched(self):
        why = runner.gate_reason("sglang", "M", 131072,
                                 bench_cache=_bench_cache([]), ledger={})
        self.assertIsNotNone(why)
        self.assertIn("never benched", why)

    def test_sglang_allowed_when_vllm_benched_at_or_above(self):
        c = _bench_cache([("M", "vllm", 262144)])
        # equal
        self.assertIsNone(runner.gate_reason("sglang", "M", 262144,
                                             bench_cache=c, ledger={}))
        # above -- the real Qwen3.5-9B case: vLLM 256K vouches for SGLang 192K
        self.assertIsNone(runner.gate_reason("sglang", "M", 196608,
                                             bench_cache=c, ledger={}))

    def test_sglang_blocked_when_vllm_ceiling_is_lower(self):
        # The inherited claim is context-dependent: a vLLM pass at 32K says
        # nothing about SGLang at 128K.
        c = _bench_cache([("M", "vllm", 32768)])
        why = runner.gate_reason("sglang", "M", 131072, bench_cache=c,
                                 ledger={})
        self.assertIsNotNone(why)
        self.assertIn("32768", why)

    def test_sglang_blocked_by_a_vllm_bench_verdict(self):
        c = _bench_cache([("M", "vllm", 262144)])
        led: dict = {}
        MS.record_bench_verdict(led, "M", "vllm", "bench_dropped",
                                detail="leak", ctx=131072)
        why = runner.gate_reason("sglang", "M", 131072, bench_cache=c,
                                 ledger=led)
        self.assertIsNotNone(why, "a vLLM drop must disqualify SGLang")

    def test_sglang_blocked_by_its_own_prior_verdict(self):
        # bench-sync classifies these as `excluded`, but a direct
        # `make bench-sglang` goes straight through discover_models.
        c = _bench_cache([("M", "vllm", 262144)])
        led: dict = {}
        MS.record_bench_verdict(led, "M", "sglang", "bench_dropped",
                                detail="leak", ctx=131072)
        why = runner.gate_reason("sglang", "M", 131072, bench_cache=c,
                                 ledger=led)
        self.assertIsNotNone(why)
        self.assertIn("previous sglang bench session", why)

    def test_gate_is_one_way(self):
        # An SGLang bench row must never vouch for a vLLM target, and a
        # missing vLLM row must never block vLLM itself.
        c = _bench_cache([("M", "sglang", 262144)])
        self.assertIsNone(runner.gate_reason("vllm", "M", 262144,
                                             bench_cache=c, ledger={}))

    def test_reason_is_human_readable(self):
        why = runner.gate_reason("sglang", "M", 131072,
                                 bench_cache=_bench_cache([]), ledger={})
        # It gets printed to stderr as the skip note; a bare boolean would
        # make the skip look like coverage.
        self.assertIn("vllm", why)
        self.assertGreater(len(why), 40)


if __name__ == "__main__":
    unittest.main()
