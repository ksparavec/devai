"""Picker <-> Ollama joins: catalog case, backend resolution, mixed-KV pin.

1. Ollama canonicalizes a k-quant tag's case on `ollama create` (a catalog
   `ornith:9b-q4_k_m` is stored on disk as `ornith:9b-q4_K_M`), while
   generate-catalog lowercases every gguf-derived tag. Ollama tags are
   case-insensitive, so `_discover_models` must still attach the catalog
   family/purpose metadata to the on-disk (canonicalized-case) row --
   otherwise gguf_repos k-quant rows show a blank family column.

2. `_backend_for_model_name` must resolve the backend from the probe
   caches, not from the name's shape: a mixed-KV Ollama row is emitted
   with a pinned `@<ctx>` too, so "`@` means vLLM" routes it to the wrong
   router port. Cache-key PRESENCE alone is not enough on the HF side --
   a failed probe writes a row too, and only picker-exposed backends
   (`_PICKER_HF_BACKENDS`) with a serveable cell may be routed to.

3. The mixed-KV context-tier pin (`_kv_cells` / `_kv_mixed` /
   `_kv_for_ctx` / `_resolve_kv_tier` / `_serving_name`) is what makes the
   router reproduce the tier's PROBED KV dtype. Losing the pin silently
   serves a quantized-KV tier the user did not choose (q8_0 measures
   ~-10 GPQA points on long chains).

Stdlib unittest only. Run with:
    python3 -m unittest tests.python.test_model_picker_ollama_case
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_picker():
    spec = importlib.util.spec_from_file_location(
        "_picker_under_test_ollama_case",
        str(REPO_ROOT / "scripts" / "model-picker.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


PICKER = _load_picker()


class TestOllamaCaseInsensitiveCatalogJoin(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)

        # On-disk manifest tag: Ollama's canonicalized (mixed) case.
        manif = root / "manifests" / "ornith"
        manif.mkdir(parents=True)
        (manif / "9b-q4_K_M").write_text("{}")

        # Catalog: lowercase name, exactly as generate-catalog emits.
        cat = root / "models.yaml"
        cat.write_text(
            'models:\n'
            '  - name: "ornith:9b-q4_k_m"\n'
            '    family: ornith\n'
            '    source: gguf\n'
            '    size: "5.24 GB"\n'
            '    purpose: "ornith gguf row"\n'
        )

        # Probe cache: alias mirrors the on-disk (mixed-case) tag.
        cache = root / ".ollama-reasoning-cache.json"
        cache.write_text(json.dumps({
            "9f73f59457a2f1e1": {
                "schema_version": 3,
                "aliases": ["ornith:9b-q4_K_M"],
                "capability": "structured",
                "probes": {"24": {"32768": {"fits": True, "fully_on_gpu": True}}},
            }
        }))

        self._saved = {}
        patches = {
            "_OLLAMA_MANIFESTS": str(root / "manifests"),
            "_CATALOG_PATHS": [str(cat)],
            "_PROBE_CACHE_PATHS": [str(cache)],
            "_VLLM_PROBE_CACHE_PATHS": [str(root / "no-vllm")],
            "_SGLANG_PROBE_CACHE_PATHS": [str(root / "no-sglang")],
            "_VLLM_DIR": str(root / "no-vllm-dir"),
        }
        for attr, val in patches.items():
            self._saved[attr] = getattr(PICKER, attr)
            setattr(PICKER, attr, val)
        # Avoid parsing the stub manifest for its byte size.
        self._saved["_ollama_disk_size_gb"] = PICKER._ollama_disk_size_gb
        PICKER._ollama_disk_size_gb = lambda lib, tag: 5.24

    def tearDown(self) -> None:
        for attr, val in self._saved.items():
            setattr(PICKER, attr, val)
        self.tmp.cleanup()

    def test_lowercase_catalog_joins_canonicalized_ondisk_tag(self) -> None:
        # Sanity: the exact-case catalog lookup misses (this was the bug).
        catalog = PICKER._load_catalog()
        self.assertIsNone(catalog.get("ornith:9b-q4_K_M"))

        rows = PICKER._discover_models()
        ornith = [r for r in rows if r["name"].lower().startswith("ornith:")]
        self.assertEqual(len(ornith), 1)
        row = ornith[0]
        self.assertEqual(row["name"], "ornith:9b-q4_K_M")   # on-disk tag
        # The fix: catalog family/purpose still attach despite the case diff.
        self.assertEqual(row["family"], "ornith")
        self.assertEqual(row["purpose"], "ornith gguf row")


class TestPeelControlSuffixes(unittest.TestCase):
    def test_peels_ctx_and_markers_in_any_order(self) -> None:
        peel = PICKER._peel_control_suffixes
        self.assertEqual(peel("qwen3.6:35b-a3b-mtp"), "qwen3.6:35b-a3b-mtp")
        self.assertEqual(peel("qwen3.6:35b-a3b-mtp@131072"),
                         "qwen3.6:35b-a3b-mtp")
        self.assertEqual(peel("Qwen3-8B-NVFP4::nothink@32768"),
                         "Qwen3-8B-NVFP4")
        # aiagent/litellm appends ::<reasoning> AFTER the @<ctx>.
        self.assertEqual(peel("Qwen3-8B-NVFP4@32768::nothink"),
                         "Qwen3-8B-NVFP4")
        self.assertEqual(peel("Qwen3-8B-NVFP4::nothink::mtp@32768"),
                         "Qwen3-8B-NVFP4")

    def test_non_numeric_at_tail_is_not_a_ctx(self) -> None:
        # HF repo@sha must survive intact.
        self.assertEqual(
            PICKER._peel_control_suffixes("nvidia/Qwen3-8B-NVFP4@abc123"),
            "nvidia/Qwen3-8B-NVFP4@abc123",
        )

    def test_unrecognised_marker_token_survives(self) -> None:
        # The arbiter's sub-parsers strip only tokens they recognise, so
        # a name that legitimately contains `::` must survive untouched.
        peel = PICKER._peel_control_suffixes
        self.assertEqual(peel("weird::name"), "weird::name")
        self.assertEqual(peel("weird::name@32768"), "weird::name")
        self.assertEqual(peel("weird::name::nothink"), "weird::name")

    def test_non_positive_ctx_is_not_a_ctx(self) -> None:
        # parseCtxOverride requires n > 0.
        peel = PICKER._peel_control_suffixes
        self.assertEqual(peel("foo@0"), "foo@0")
        self.assertEqual(peel("foo@-1"), "foo@-1")

    def test_marker_case_and_whitespace_tolerated(self) -> None:
        # Sub-parsers lowercase + TrimSpace the token before matching.
        peel = PICKER._peel_control_suffixes
        self.assertEqual(peel("Qwen3-8B-NVFP4::NoThink"), "Qwen3-8B-NVFP4")
        self.assertEqual(peel("Qwen3-8B-NVFP4@ 32768 "), "Qwen3-8B-NVFP4")


class TestBackendResolutionFromProbeCaches(unittest.TestCase):
    """`@<ctx>` alone must not imply vLLM -- mixed-KV Ollama rows pin it."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)

        ollama = root / ".ollama-reasoning-cache.json"
        ollama.write_text(json.dumps({
            "aaaabbbbccccdddd": {
                "schema_version": 3,
                "aliases": ["qwen3.6:35b-a3b-mtp"],
                "capability": "inline",
                "probes": {"24": {"131072": {"fits": True,
                                             "fully_on_gpu": True}}},
            }
        }))
        vllm = root / ".vllm-reasoning-cache.json"
        vllm.write_text(json.dumps({
            "nvidia/Qwen3-8B-NVFP4@sha": {
                "aliases": ["Qwen3-8B-NVFP4"],
                "capability": "structured",
                "probes": {"24": {"32768": {"fits": True}}},
            },
            # Loaded, but OOMed under a near-full-context request.
            "nvidia/Serves-Not-NVFP4@sha": {
                "aliases": ["Serves-Not-NVFP4"],
                "capability": "structured",
                "probes": {"24": {"32768": {"fits": True,
                                            "serving_ok": False}}},
            },
        }))
        sglang = root / ".sglang-reasoning-cache.json"
        sglang.write_text(json.dumps({
            # Fitting on SGLang -- but SGLang is filtered out of the
            # picker (_PICKER_HF_BACKENDS), so it must not be routed to.
            "meta/Llama-3.1-8B-Instruct-NVFP4@sha": {
                "aliases": ["Llama-3.1-8B-Instruct-NVFP4"],
                "capability": "structured",
                "probes": {"24": {"131072": {"fits": True}}},
            },
            # THE REGRESSION: a failed SGLang probe row is still a row.
            "nvidia/Gemma-4-31B-IT-NVFP4@sha": {
                "aliases": ["Gemma-4-31B-IT-NVFP4"],
                "capability": "error",
                "probes": {"24": {"262144": {"fits": False,
                                             "capability": "error"}}},
            },
        }))

        self._saved = {}
        patches = {
            "_PROBE_CACHE_PATHS": [str(ollama)],
            "_VLLM_PROBE_CACHE_PATHS": [str(vllm)],
            "_SGLANG_PROBE_CACHE_PATHS": [str(sglang)],
            "_PICKER_HF_BACKENDS": ("vllm",),
        }
        for attr, val in patches.items():
            self._saved[attr] = getattr(PICKER, attr)
            setattr(PICKER, attr, val)

    def tearDown(self) -> None:
        for attr, val in self._saved.items():
            setattr(PICKER, attr, val)
        self.tmp.cleanup()

    def test_pinned_ollama_row_resolves_to_ollama(self) -> None:
        # THE REGRESSION: shape-based detection called this vLLM.
        self.assertEqual(
            PICKER._backend_for_model_name("qwen3.6:35b-a3b-mtp@131072"),
            "ollama",
        )

    def test_bare_ollama_row_resolves_to_ollama(self) -> None:
        self.assertEqual(
            PICKER._backend_for_model_name("qwen3.6:35b-a3b-mtp"), "ollama")

    def test_vllm_row_resolves_to_vllm(self) -> None:
        self.assertEqual(
            PICKER._backend_for_model_name("Qwen3-8B-NVFP4@32768"), "vllm")

    def test_sglang_only_row_is_not_routed_while_sglang_is_filtered(
            self) -> None:
        # SGLang is not in _PICKER_HF_BACKENDS, so a row that exists only
        # in the SGLang cache must fall back to the shape heuristic
        # (vLLM port) rather than being dispatched to a backend the
        # picker never offers the model on.
        self.assertEqual(
            PICKER._backend_for_model_name(
                "Llama-3.1-8B-Instruct-NVFP4@131072"),
            "vllm",
        )

    def test_sglang_row_resolves_to_sglang_when_exposed(self) -> None:
        PICKER._PICKER_HF_BACKENDS = ("vllm", "sglang")
        self.assertEqual(
            PICKER._backend_for_model_name(
                "Llama-3.1-8B-Instruct-NVFP4@131072"),
            "sglang",
        )

    def test_failed_sglang_probe_row_is_not_a_backend_match(self) -> None:
        # THE REGRESSION: cache-key presence alone routed this to 11436
        # even with capability=error / fits=false, and even with SGLang
        # exposed there is no serveable cell to route to.
        for exposed in (("vllm",), ("vllm", "sglang")):
            PICKER._PICKER_HF_BACKENDS = exposed
            self.assertEqual(
                PICKER._backend_for_model_name("Gemma-4-31B-IT-NVFP4@262144"),
                "vllm",
            )

    def test_serving_ok_false_cell_is_not_a_backend_match(self) -> None:
        # fits=true but the LOAD probe says it OOMs when actually served.
        self.assertFalse(
            PICKER._hf_entry_serveable(
                PICKER._load_hf_probe_records(
                    PICKER._VLLM_PROBE_CACHE_PATHS)["Serves-Not-NVFP4"]))

    def test_suffix_order_does_not_matter(self) -> None:
        for name in ("qwen3.6:35b-a3b-mtp::nothink@131072",
                     "qwen3.6:35b-a3b-mtp@131072::nothink"):
            self.assertEqual(PICKER._backend_for_model_name(name), "ollama")

    def test_unknown_name_falls_back_to_shape_heuristic(self) -> None:
        self.assertEqual(
            PICKER._backend_for_model_name("never-probed@32768"), "vllm")
        self.assertEqual(
            PICKER._backend_for_model_name("never-probed"), "ollama")


class TestMixedKVContextPin(unittest.TestCase):
    """The mixed-KV tier sub-modal + `@<ctx>` pin (quality-critical)."""

    def setUp(self) -> None:
        self._saved_budget = PICKER._VRAM_BUDGET
        PICKER._VRAM_BUDGET = 24.0
        self._saved_fzf = PICKER._fzf

    def tearDown(self) -> None:
        PICKER._VRAM_BUDGET = self._saved_budget
        PICKER._fzf = self._saved_fzf

    def _row(self, cells: dict[str, dict], backend: str = "ollama") -> dict:
        return {
            "name": "qwen3.6:35b-a3b-mtp",
            "backend": backend,
            "vram": {"probes": {"24": cells}},
            "_picker_context": max(int(k) for k in cells),
        }

    def test_kv_cells_reads_stamped_dtype_and_skips_spilled(self) -> None:
        m = self._row({
            "65536": {"fully_on_gpu": True, "kv_cache_type": "f16"},
            "131072": {"fully_on_gpu": True, "kv_cache_type": "q8_0"},
            "262144": {"fully_on_gpu": False, "kv_cache_type": "q8_0"},
        })
        self.assertEqual(PICKER._kv_cells(m), {65536: "f16", 131072: "q8_0"})

    def test_unstamped_ollama_cell_decodes_to_f16(self) -> None:
        m = self._row({"32768": {"fully_on_gpu": True}})
        self.assertEqual(PICKER._kv_cells(m), {32768: "f16"})

    def test_unstamped_vllm_cell_decodes_to_fp8(self) -> None:
        m = self._row({"32768": {"fully_on_gpu": True}}, backend="vllm")
        self.assertEqual(PICKER._kv_cells(m), {32768: "fp8"})

    def test_mixed_detection(self) -> None:
        mixed = self._row({
            "65536": {"fully_on_gpu": True, "kv_cache_type": "f16"},
            "131072": {"fully_on_gpu": True, "kv_cache_type": "q8_0"},
        })
        uniform = self._row({
            "65536": {"fully_on_gpu": True, "kv_cache_type": "f16"},
            "131072": {"fully_on_gpu": True, "kv_cache_type": "auto"},
        })
        self.assertTrue(PICKER._kv_mixed(mixed))
        # f16 / auto / "" all mean unquantized -> one kind, not mixed.
        self.assertFalse(PICKER._kv_mixed(uniform))

    def test_kv_for_ctx_picks_smallest_covering_tier(self) -> None:
        m = self._row({
            "65536": {"fully_on_gpu": True, "kv_cache_type": "f16"},
            "131072": {"fully_on_gpu": True, "kv_cache_type": "q8_0"},
        })
        self.assertEqual(PICKER._kv_for_ctx(m, 32768), "f16")
        self.assertEqual(PICKER._kv_for_ctx(m, 65536), "f16")
        self.assertEqual(PICKER._kv_for_ctx(m, 131072), "q8_0")

    def test_resolve_kv_tier_skips_modal_for_uniform_kv(self) -> None:
        m = self._row({
            "65536": {"fully_on_gpu": True, "kv_cache_type": "f16"},
            "131072": {"fully_on_gpu": True, "kv_cache_type": "f16"},
        })
        called = []
        PICKER._fzf = lambda *a, **k: called.append(a) or 0
        self.assertEqual(PICKER._resolve_kv_tier(m), (131072, False))
        self.assertEqual(called, [])   # no sub-modal for uniform KV

    def test_resolve_kv_tier_pins_chosen_tier_and_warns(self) -> None:
        m = self._row({
            "65536": {"fully_on_gpu": True, "kv_cache_type": "f16"},
            "131072": {"fully_on_gpu": True, "kv_cache_type": "q8_0"},
        })
        seen: dict = {}

        def fake_fzf(lines, header, **kwargs):
            seen["lines"] = lines
            return 0            # tiers are sorted descending -> 128K

        PICKER._fzf = fake_fzf
        self.assertEqual(PICKER._resolve_kv_tier(m), (131072, True))
        joined = "\n".join(seen["lines"])
        self.assertIn("q8_0", joined)
        self.assertIn("weaker long-form reasoning", joined)
        self.assertIn("GPQA", joined)
        self.assertIn("full quality", joined)

    def test_resolve_kv_tier_second_row_selects_f16_tier(self) -> None:
        m = self._row({
            "65536": {"fully_on_gpu": True, "kv_cache_type": "f16"},
            "131072": {"fully_on_gpu": True, "kv_cache_type": "q8_0"},
        })
        PICKER._fzf = lambda lines, header, **kw: 1
        self.assertEqual(PICKER._resolve_kv_tier(m), (65536, True))

    def test_resolve_kv_tier_esc_returns_none(self) -> None:
        m = self._row({
            "65536": {"fully_on_gpu": True, "kv_cache_type": "f16"},
            "131072": {"fully_on_gpu": True, "kv_cache_type": "q8_0"},
        })
        PICKER._fzf = lambda lines, header, **kw: None
        self.assertIsNone(PICKER._resolve_kv_tier(m))

    def test_serving_name_pins_ctx_only_when_pinned_for_ollama(self) -> None:
        self.assertEqual(
            PICKER._serving_name("qwen3.6:35b-a3b-mtp", "ollama", "", "",
                                 131072, True),
            "qwen3.6:35b-a3b-mtp@131072",
        )
        self.assertEqual(
            PICKER._serving_name("qwen3.6:35b-a3b-mtp", "ollama", "", "",
                                 131072, False),
            "qwen3.6:35b-a3b-mtp",
        )

    def test_serving_name_always_pins_ctx_for_hf_backends(self) -> None:
        for backend in ("vllm", "sglang"):
            self.assertEqual(
                PICKER._serving_name("Qwen3-8B-NVFP4", backend, "", "",
                                     32768, False),
                "Qwen3-8B-NVFP4@32768",
            )

    def test_serving_name_canonical_suffix_order(self) -> None:
        self.assertEqual(
            PICKER._serving_name("Qwen3-8B-NVFP4", "vllm", "::nothink",
                                 "::mtp", 32768, False),
            "Qwen3-8B-NVFP4::nothink::mtp@32768",
        )
        self.assertEqual(
            PICKER._serving_name("qwen3.6:35b-a3b-mtp", "ollama", "::nothink",
                                 "", 131072, True),
            "qwen3.6:35b-a3b-mtp::nothink@131072",
        )

    def test_pinned_name_round_trips_back_to_ollama_backend(self) -> None:
        # End-to-end of the two fixes: what the picker emits for a
        # mixed-KV Ollama row must not be read back as vLLM by shape.
        name = PICKER._serving_name("qwen3.6:35b-a3b-mtp", "ollama", "", "",
                                    131072, True)
        self.assertEqual(PICKER._peel_control_suffixes(name),
                         "qwen3.6:35b-a3b-mtp")


if __name__ == "__main__":
    unittest.main()
