"""The picker must surface SGLang rows, and must not swallow them.

Two separate gates had to open. Both were invisible while SGLang was
hidden, and the second only revealed itself once the first was lifted --
which is why this file pins both rather than just the one that was
edited.

1. `_PICKER_BACKENDS` deliberately excluded "sglang" from 2026-05-01
   (87aa382) on agent-compatibility grounds. Re-enabled 2026-07-27 by
   operator decision after SGLang was probed and benched on this fleet.

2. `_dedup_hf_by_name` keyed on the model NAME alone and kept only the
   highest-priority backend (vLLM > SGLang). With SGLang filtered out
   every name had at most one HF row, so the function was a no-op and
   the bug could not be seen. The moment SGLang was re-enabled it
   silently dropped all four SGLang rows, because every SGLang model on
   this fleet is also probed on vLLM -- so lifting gate 1 alone still
   produced a picker with zero SGLang models.

   That also contradicted the documented contract in CLAUDE.md: "the
   picker shows one row per (model, backend) pair".

Stdlib unittest only. No probe/bench cache required -- these test the
selection logic, not this host's data.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "model_picker_sglang", REPO_ROOT / "scripts" / "model-picker.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_picker_sglang"] = mod
    spec.loader.exec_module(mod)
    return mod


mp = _load()


def _row(name, backend):
    return {"name": name, "backend": backend}


class PickerBackendsTest(unittest.TestCase):
    def test_sglang_is_surfaced(self):
        self.assertIn("sglang", mp._PICKER_BACKENDS)

    def test_hf_backends_derive_from_picker_backends(self):
        self.assertEqual(
            set(mp._PICKER_HF_BACKENDS),
            set(mp._PICKER_BACKENDS) - {"ollama"})

    def test_every_surfaced_backend_has_a_port(self):
        """A backend in the menu with no _BACKENDS entry would raise
        KeyError inside _build, at the moment the user hits enter."""
        for b in mp._PICKER_BACKENDS:
            with self.subTest(backend=b):
                self.assertIn(b, mp._BACKENDS)


class DedupKeepsBothBackendsTest(unittest.TestCase):
    """The regression that made lifting gate 1 useless."""

    def test_a_model_on_both_backends_keeps_both_rows(self):
        rows = mp._dedup_hf_rows([
            _row("Qwen3.5-9B-NVFP4", "vllm"),
            _row("Qwen3.5-9B-NVFP4", "sglang"),
        ])
        self.assertEqual(
            [(r["name"], r["backend"]) for r in rows],
            [("Qwen3.5-9B-NVFP4", "vllm"), ("Qwen3.5-9B-NVFP4", "sglang")],
            "an SGLang row must survive alongside its vLLM twin")

    def test_vllm_leads_within_a_name(self):
        rows = mp._dedup_hf_rows([
            _row("M", "sglang"),
            _row("M", "vllm"),
        ])
        self.assertEqual(rows[0]["backend"], "vllm")

    def test_true_duplicates_still_collapse(self):
        rows = mp._dedup_hf_rows([_row("M", "vllm"), _row("M", "vllm")])
        self.assertEqual(len(rows), 1)

    def test_ollama_rows_pass_through(self):
        rows = mp._dedup_hf_rows([
            _row("gemma4:26b-a4b-it-q4_K_M", "ollama"),
            _row("M", "vllm"),
        ])
        self.assertEqual(len(rows), 2)

    def test_distinct_names_are_untouched(self):
        rows = mp._dedup_hf_rows([_row("A", "vllm"), _row("B", "sglang")])
        self.assertEqual(len(rows), 2)

    def test_empty_input(self):
        self.assertEqual(mp._dedup_hf_rows([]), [])


class BackendRoutingTest(unittest.TestCase):
    """Picking an SGLang row must actually address SGLang. The port is
    carried in the environment, not in argv, so a wrong port would look
    fine in the emitted command and silently serve from vLLM."""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       ("ANTHROPIC_BASE_URL", "AIAGENT_API_BASE",
                        "ANTHROPIC_AUTH_TOKEN", "AIAGENT_MODEL")}
        self.addCleanup(self._restore)

    def _restore(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_sglang_routes_to_11436(self):
        os.environ.pop("ANTHROPIC_BASE_URL", None)
        mp._build("claude", "Ornith-1.0-9B-NVFP4", "sglang")
        self.assertTrue(
            os.environ["ANTHROPIC_BASE_URL"].endswith(":11436"),
            f"got {os.environ['ANTHROPIC_BASE_URL']}")

    def test_vllm_routes_to_11435(self):
        os.environ.pop("ANTHROPIC_BASE_URL", None)
        mp._build("claude", "Ornith-1.0-9B-NVFP4", "vllm")
        self.assertTrue(
            os.environ["ANTHROPIC_BASE_URL"].endswith(":11435"),
            f"got {os.environ['ANTHROPIC_BASE_URL']}")

    def test_ports_are_distinct_per_backend(self):
        ports = {b: mp._BACKENDS[b][2] for b in mp._PICKER_BACKENDS}
        self.assertEqual(len(set(ports.values())), len(ports),
                         f"backend ports collide: {ports}")


class DocumentedContractTest(unittest.TestCase):
    def test_claude_md_states_one_row_per_model_backend(self):
        text = (REPO_ROOT / "CLAUDE.md").read_text()
        self.assertIn("one row per `(model, backend)` pair", text,
                      "the dedup behaviour and the doc must agree")


if __name__ == "__main__":
    unittest.main()
