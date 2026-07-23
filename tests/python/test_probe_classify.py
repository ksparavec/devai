"""Unit tests for the probe failure classifier (Phase 1).

Covers scripts/_probe_hf_common.py:_failure_excerpt + classify_failure_logs:
the saved excerpt must capture the ROOT cause (near the top of a long vLLM
traceback), not just the generic "Engine core initialization failed" tail
that buried the gemma-4 cause. Pattern matching runs against the full log.

Stdlib unittest only. Run with:
    python3 -m unittest tests.python.test_probe_classify
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _probe_hf_common as P  # noqa: E402


def _load_script(mod_name: str, filename: str):
    """Import a hyphenated script from scripts/ under `mod_name`."""
    spec = importlib.util.spec_from_file_location(
        mod_name, str(REPO_ROOT / "scripts" / filename))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _long_log(root_cause: str) -> str:
    """A realistic failed-launch log: startup noise, the real cause, a long
    traceback, then the generic wrapper at the very end."""
    return (
        "INFO startup\n" + "INFO loading weights\n" * 8
        + root_cause + "\n"
        + "\n".join(f"  File \"x.py\", line {i}, in f" for i in range(100))
        + "\nRuntimeError: Engine core initialization failed. "
          "See root cause above. Failed core proc(s): {}\n"
    )


class TestFailureExcerpt(unittest.TestCase):
    def test_captures_root_cause_not_just_tail(self) -> None:
        logs = _long_log("ValueError: Model architectures ['FooForCausalLM'] "
                         "are not supported for now")
        excerpt = P._failure_excerpt(logs)
        self.assertIn("Model architectures", excerpt)   # root cause preserved
        # bounded
        self.assertLessEqual(len(excerpt.splitlines()), 121)

    def test_short_log_kept_whole(self) -> None:
        short = "line1\nValueError: boom\nline3"
        self.assertEqual(P._failure_excerpt(short), short)

    def test_no_anchor_falls_back_to_tail(self) -> None:
        logs = "\n".join(f"info line {i}" for i in range(300))
        excerpt = P._failure_excerpt(logs)
        self.assertIn("info line 299", excerpt)          # tail present
        self.assertNotIn("info line 0", excerpt)         # head dropped


class TestClassifyFailureLogs(unittest.TestCase):
    def test_arch_match_through_full_log(self) -> None:
        # The arch line is near the TOP, buried under 100 traceback lines;
        # it must still classify arch (full-log match) and land in excerpt.
        logs = _long_log("ValueError: Model architectures ['X'] are not "
                         "supported for now")
        rec = P.classify_failure_logs(logs)
        self.assertEqual(rec["kind"], "arch")
        self.assertEqual(rec["matched_pattern"], "Model architectures")
        self.assertIn("Model architectures", rec["log_excerpt"])

    def test_oom_classified(self) -> None:
        rec = P.classify_failure_logs(_long_log("torch.cuda.OutOfMemoryError: "
                                                "CUDA out of memory"))
        self.assertEqual(rec["kind"], "oom_startup")

    def test_unknown_is_infra(self) -> None:
        # A genuine infra failure (no arch/quant/oom marker) stays infra --
        # NOT forced terminal. (gemma-4's MM-config failure is one of these:
        # it is fixed with a recovery flag, not by excluding the model.)
        rec = P.classify_failure_logs(_long_log(
            "Chunked MM input disabled but max_tokens_per_mm_item (2496) is "
            "larger than max_num_batched_tokens (2048)"))
        self.assertEqual(rec["kind"], "infra")


class TestShaStability(unittest.TestCase):
    """Phase 2: terminal verdicts survive a re-quant; orphans pruned."""

    def _entry(self, repo, sha, capability):
        return {"schema_version": 2, "repo": repo, "sha": sha,
                "capability": capability, "evidence": {"kind": "arch"},
                "probes": {}, "aliases": []}

    def test_carry_forward_unsupported_arch(self) -> None:
        cache = {"r/M@oldsha": self._entry("r/M", "oldsha",
                                           P.Capability.UNSUPPORTED_ARCH)}
        new = P.ensure_entry(cache, "r/M@newsha", "r/M", "newsha", "M", 2,
                             "hf", 10.0)
        self.assertEqual(new["capability"], P.Capability.UNSUPPORTED_ARCH)
        self.assertEqual(new.get("carried_from_sha"), "oldsha")

    def test_oom_not_carried_forward(self) -> None:
        cache = {"r/M@oldsha": self._entry("r/M", "oldsha", P.Capability.ERROR)}
        new = P.ensure_entry(cache, "r/M@newsha", "r/M", "newsha", "M", 2,
                             "hf", 10.0)
        self.assertEqual(new["capability"], P.Capability.UNKNOWN)  # re-checked
        self.assertNotIn("carried_from_sha", new)

    def test_prune_drops_orphan_keeps_current(self) -> None:
        cache = {
            "r/M@old": self._entry("r/M", "old", P.Capability.STRUCTURED),
            "r/M@cur": self._entry("r/M", "cur", P.Capability.STRUCTURED),
        }
        catalog = [{"repo": "r/M", "sha": "cur"}]
        n = P.prune_orphaned_shas(cache, catalog)
        self.assertEqual(n, 1)
        self.assertIn("r/M@cur", cache)
        self.assertNotIn("r/M@old", cache)

    def test_prune_keeps_last_entry_when_no_current(self) -> None:
        # Current sha never probed (model not on disk) -> keep the only data.
        cache = {"r/M@old": self._entry("r/M", "old", P.Capability.UNSUPPORTED_ARCH)}
        catalog = [{"repo": "r/M", "sha": "cur"}]
        n = P.prune_orphaned_shas(cache, catalog)
        self.assertEqual(n, 0)
        self.assertIn("r/M@old", cache)


class TestEntryFitsAnywhere(unittest.TestCase):
    """Drives the un-exclude-on-recovery path: a model that loaded anywhere."""

    def test_true_when_any_cell_fits(self) -> None:
        entry = {"probes": {"24": {"32768": {"fits": False},
                                   "65536": {"fits": True}}}}
        self.assertTrue(P._entry_fits_anywhere(entry))

    def test_false_when_no_cell_fits(self) -> None:
        entry = {"probes": {"24": {"32768": {"fits": False}}}}
        self.assertFalse(P._entry_fits_anywhere(entry))

    def test_handles_missing_or_malformed_probes(self) -> None:
        self.assertFalse(P._entry_fits_anywhere({}))
        self.assertFalse(P._entry_fits_anywhere({"probes": {"24": "bad"}}))


class TestOomEverywhere(unittest.TestCase):
    """Drives the ledger `oom` exclusion: fits nowhere + OOM failures."""

    def test_oom_at_all_tiers(self) -> None:
        entry = {"probes": {"24": {
            "32768": {"fits": False, "evidence": {"kind": "oom_startup"}},
            "65536": {"fits": False, "evidence": {"kind": "implied_spill"}}}}}
        self.assertTrue(P._entry_oom_everywhere(entry))

    def test_fits_somewhere_is_not_oom(self) -> None:
        entry = {"probes": {"24": {
            "32768": {"fits": True},
            "65536": {"fits": False, "evidence": {"kind": "oom_startup"}}}}}
        self.assertFalse(P._entry_oom_everywhere(entry))

    def test_arch_failure_is_not_oom(self) -> None:
        # A terminal arch failure is owned by unsupported_arch, not oom.
        entry = {"probes": {"24": {
            "32768": {"fits": False, "evidence": {"kind": "arch"}}}}}
        self.assertFalse(P._entry_oom_everywhere(entry))

    def test_infra_failure_is_not_oom(self) -> None:
        # Genuine infra (retryable) must NOT be recorded as a durable oom.
        entry = {"probes": {"24": {
            "32768": {"fits": False, "evidence": {"kind": "infra"}}}}}
        self.assertFalse(P._entry_oom_everywhere(entry))


class TestRecoveryRegistryBackendFilter(unittest.TestCase):
    """Cross-unit contract C2: a recovery entry MAY declare `backends`.
    Absent = applies to every backend (backward compatible); present =
    only the listed backends. vLLM-only flags like --language-model-only
    are not valid SGLang arguments and would fail the launch outright."""

    def setUp(self) -> None:
        self._saved = P._RECOVERY_REGISTRY
        P._RECOVERY_REGISTRY = {
            "AllBackends": {"engine_flags": ["--enforce-eager"],
                            "engine_env": {"A": "1"}},
            "VllmOnly": {"backends": ["vllm"],
                         "engine_flags": ["--language-model-only"],
                         "engine_env": {"B": "2"},
                         "image": "custom/vllm:gemma"},
            "Neither": {"backends": [], "engine_flags": ["--nope"]},
        }

    def tearDown(self) -> None:
        P._RECOVERY_REGISTRY = self._saved

    def test_absent_backends_applies_everywhere(self) -> None:
        for backend in ("vllm", "sglang"):
            flags, env = P.recovery_overrides("AllBackends", backend)
            self.assertEqual(flags, ["--enforce-eager"])
            self.assertEqual(env, {"A": "1"})

    def test_listed_backend_gets_the_entry(self) -> None:
        flags, env = P.recovery_overrides("VllmOnly", "vllm")
        self.assertEqual(flags, ["--language-model-only"])
        self.assertEqual(env, {"B": "2"})
        self.assertEqual(P.recovery_image("VllmOnly", "vllm"),
                         "custom/vllm:gemma")

    def test_unlisted_backend_is_skipped(self) -> None:
        flags, env = P.recovery_overrides("VllmOnly", "sglang")
        self.assertEqual(flags, [])
        self.assertEqual(env, {})
        self.assertIsNone(P.recovery_image("VllmOnly", "sglang"))

    def test_empty_backends_list_applies_to_nothing(self) -> None:
        for backend in ("vllm", "sglang"):
            self.assertEqual(P.recovery_overrides("Neither", backend), ([], {}))

    def test_unknown_model_is_empty(self) -> None:
        self.assertEqual(P.recovery_overrides("Absent", "vllm"), ([], {}))
        self.assertIsNone(P.recovery_image("Absent", "vllm"))

    def test_predicate_shape(self) -> None:
        self.assertTrue(P._entry_applies_to_backend({}, "vllm"))
        with contextlib.redirect_stderr(io.StringIO()):  # warns; see test below
            self.assertTrue(
                P._entry_applies_to_backend({"backends": "vllm"}, "vllm"))  # not a list
        self.assertFalse(
            P._entry_applies_to_backend({"backends": ["sglang"]}, "vllm"))

    def test_malformed_backends_warns_naming_the_model(self) -> None:
        # Contract: a non-list `backends` is treated as ABSENT (never as a
        # silent drop of the entry's recovery flags -- that is the OOM class
        # the registry exists to prevent) and must warn, naming the model so
        # the operator can find the typo.
        P._WARNED_BAD_BACKENDS.discard("TypoModel")
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            applies = P._entry_applies_to_backend(
                {"backends": "vllm"}, "sglang", "TypoModel")
        self.assertTrue(applies)
        err = buf.getvalue()
        self.assertIn("TypoModel", err)
        self.assertIn("backends", err)


class TestSchemaVersionNotBumpedWithoutReprobe(unittest.TestCase):
    """ensure_entry must NOT declare a legacy v1 entry as v2 on the basis
    of backfilled non-verified defaults -- that defeats the router's
    deliberate v1 refusal (which exists because a v1 entry has no probed
    tool_parser / disable_verified)."""

    def test_v1_entry_keeps_its_version(self) -> None:
        cache = {"r/M@s": {"schema_version": 1, "repo": "r/M", "sha": "s",
                           "aliases": ["M"], "probes": {}}}
        entry = P.ensure_entry(cache, "r/M@s", "r/M", "s", "M", 2, "hf", 10.0)
        self.assertEqual(entry["schema_version"], 1)
        # The v2 fields are still backfilled (non-verified defaults) so the
        # rest of the prober can read them.
        self.assertIsNone(entry["tool_parser"])
        self.assertIs(entry["disable_verified"], False)

    def test_new_entry_is_created_at_target_version(self) -> None:
        cache: dict = {}
        entry = P.ensure_entry(cache, "r/N@s", "r/N", "s", "N", 2, "hf", 10.0)
        self.assertEqual(entry["schema_version"], 2)


class TestStaleSchemaForcesReprobe(unittest.TestCase):
    """The companion to the class above: because ensure_entry refuses to
    bump, and the bump at the end of run_probe_pass is gated on freshly
    probed cells, a v1 entry that ALREADY has cached cells would short-
    circuit forever -- making gpu-arbiter's own "Re-probe with make
    probe-vllm / make probe-sglang to upgrade" instruction unachievable
    without PROBE_FORCE=1. run_probe_pass must therefore re-probe an
    out-of-date entry. The container-launching probe_one_cell is replaced
    with a scripted fake, so this needs no GPU.
    """

    def _run(self, *, schema_version, cells, no_cache_write=False):
        """Drive run_probe_pass over one synthetic catalog row.
        Returns (entry, probe_calls)."""
        calls: list[int] = []
        cache = {
            "vendor/m@sha1": {
                "schema_version": schema_version,
                "repo": "vendor/m",
                "sha": "sha1",
                "aliases": ["m"],
                "capability": "inline",
                "probes": {"24": dict(cells)},
            }
        }
        saved = {name: getattr(P, name) for name in (
            "assert_no_active_backends", "install_probe_cleanup",
            "load_catalog_hf_rows", "is_downloaded", "model_kind_from_disk",
            "model_size_gb_from_row", "load_cache", "save_cache",
            "image_digest_via_cli", "effective_position_limit",
            "probe_one_cell", "_load_ledger", "_save_ledger",
            "_ledger_reason", "_ledger_record", "_ledger_clear",
            "_ledger_is_excluded",
        )}
        P.assert_no_active_backends = lambda runtime: None
        P.install_probe_cleanup = lambda: None
        P.load_catalog_hf_rows = lambda catalog, name: [
            {"name": "m", "repo": "vendor/m", "sha": "sha1", "parsers": {}}
        ]
        P.is_downloaded = lambda name, models_dir: True
        P.model_kind_from_disk = lambda name, models_dir: "hf"
        P.model_size_gb_from_row = lambda row: 5.0
        P.load_cache = lambda path: cache
        P.save_cache = lambda path, c: None
        P.image_digest_via_cli = lambda runtime, image: "sha256:deadbeef"
        P.effective_position_limit = lambda name, models_dir: None
        P._load_ledger = lambda: {}
        P._save_ledger = lambda ledger, host_vram_gb=None: None
        P._ledger_reason = lambda ledger, name, backend: None
        P._ledger_record = lambda ledger, name, backend, reason, **kw: None
        P._ledger_clear = lambda ledger, name, backend: None
        P._ledger_is_excluded = lambda ledger, name, backend, **kw: False

        def fake_probe(spec, *, requested_ctx, **kw):
            calls.append(requested_ctx)
            return {
                "ctx": requested_ctx, "vram_gb": 24, "fits": True,
                "actual_context": requested_ctx, "actual_vram_gb": 20.0,
                "capability": "inline", "evidence": {},
                "reasoning_parser": "qwen3", "tool_parser": "hermes",
                "disable_verified": True, "probed_at": "2026-07-23T00:00:00Z",
                "startup_seconds": 1.0,
            }

        P.probe_one_cell = fake_probe

        with tempfile.TemporaryDirectory() as td:
            args = argparse.Namespace(
                runtime="podman", repo="", catalog=Path(td) / "models.yaml",
                cache=Path(td) / "cache.json", models_dir=td,
                vram="", ctx="32K", host_vram_gb=24, image="img",
                container_name="c", probe_port=18000, prompt="hi",
                force=False, force_arch=False, no_mtp=True,
                no_cache_write=no_cache_write,
            )
            spec = types.SimpleNamespace(name="vllm", schema_version=2)
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    P.run_probe_pass(spec, args)
                finally:
                    for k, v in saved.items():
                        setattr(P, k, v)
        return cache["vendor/m@sha1"], calls

    _CACHED_CELL = {"32768": {"ctx": 32768, "fits": True, "capability": "inline",
                              "actual_context": 32768,
                              "probed_at": "2026-01-01T00:00:00Z"}}

    def test_v1_entry_with_cached_cells_is_reprobed_and_bumped(self) -> None:
        entry, calls = self._run(schema_version=1, cells=self._CACHED_CELL)
        self.assertEqual(calls, [32768])            # actually re-probed
        self.assertEqual(entry["schema_version"], 2)

    def test_current_version_entry_stays_cached(self) -> None:
        entry, calls = self._run(schema_version=2, cells=self._CACHED_CELL)
        self.assertEqual(calls, [])                 # no GPU minutes burned
        self.assertEqual(entry["schema_version"], 2)

    def test_no_cache_write_does_not_reprobe(self) -> None:
        # Mirrors --force: re-probing when the result cannot be saved would
        # burn GPU time for nothing.
        entry, calls = self._run(schema_version=1, cells=self._CACHED_CELL,
                                 no_cache_write=True)
        self.assertEqual(calls, [])
        self.assertEqual(entry["schema_version"], 1)


class TestContainerRegistryTeardown(unittest.TestCase):
    """A probe container must never outlive the process: container_remove
    is bounded + idempotent, and the registry drives the signal/atexit
    sweep that keeps an orphan from silently holding the GPU."""

    def setUp(self) -> None:
        self._saved = set(P._ACTIVE_CONTAINERS)
        P._ACTIVE_CONTAINERS.clear()

    def tearDown(self) -> None:
        P._ACTIVE_CONTAINERS.clear()
        P._ACTIVE_CONTAINERS.update(self._saved)

    def test_remove_is_bounded_and_never_raises(self) -> None:
        # No such runtime binary -> FileNotFoundError must be swallowed.
        P._ACTIVE_CONTAINERS.add(("devai-no-such-runtime", "c1"))
        P.container_remove("devai-no-such-runtime", "c1")
        self.assertNotIn(("devai-no-such-runtime", "c1"), P._ACTIVE_CONTAINERS)

    def test_teardown_drains_registry_and_is_idempotent(self) -> None:
        P._ACTIVE_CONTAINERS.update({
            ("devai-no-such-runtime", "a"),
            ("devai-no-such-runtime", "b"),
        })
        P._teardown_active_containers()
        self.assertEqual(P._ACTIVE_CONTAINERS, set())
        P._teardown_active_containers()   # second call is a no-op
        self.assertEqual(P._ACTIVE_CONTAINERS, set())


class TestVerifyBackendFlagsExactMatch(unittest.TestCase):
    """A REMOVED flag whose name is a prefix of a surviving one must FAIL
    the gate -- a plain substring test passed it silently."""

    def setUp(self) -> None:
        self.mod = _load_script("_verify_backend_flags_under_test",
                                "verify-backend-flags.py")

    def test_prefix_of_another_flag_is_not_a_match(self) -> None:
        help_text = "  --kv-cache-dtype-foo {a,b}   some other flag\n"
        self.assertFalse(self.mod._flag_present("--kv-cache-dtype", help_text))
        self.assertFalse(self.mod._flag_present("--model", "  --model-path P\n"))

    def test_real_forms_still_match(self) -> None:
        for text in ("  --kv-cache-dtype {auto,fp8}\n",
                     "  --kv-cache-dtype=DTYPE\n",
                     "  --enable-prefix-caching, --no-enable-prefix-caching\n"):
            flag = ("--enable-prefix-caching" if "prefix" in text
                    else "--kv-cache-dtype")
            self.assertTrue(self.mod._flag_present(flag, text), text)

    def test_check_flags_reports_the_missing_key_with_the_real_spelling(self) -> None:
        missing = self.mod._check_flags(
            "vllm", {"kv_cache_dtype": "--kv-cache-dtype"},
            "  --kv-cache-dtype-foo X\n")
        self.assertEqual(len(missing), 1)
        self.assertTrue(missing[0].startswith("kv_cache_dtype=--kv-cache-dtype"))
        self.assertIn("--kv-cache-dtype-foo", missing[0])

    def test_no_hint_when_the_flag_is_simply_gone(self) -> None:
        missing = self.mod._check_flags(
            "vllm", {"gone": "--gone"}, "  --something-else X\n")
        self.assertEqual(missing, ["gone=--gone"])

    def test_pinned_yaml_carries_kv_cache_dtype_for_both(self) -> None:
        import yaml
        cfg = yaml.safe_load(
            (REPO_ROOT / "deploy" / "backend-flags.yaml").read_text())
        for backend in ("vllm", "sglang"):
            self.assertEqual(cfg[backend].get("kv_cache_dtype"),
                             "--kv-cache-dtype", backend)


class TestOllamaDisableProbeError(unittest.TestCase):
    """probe-ollama-reasoning must never write a STRING into
    disable_verified: the router decodes it as *bool and unmarshals the
    whole cache file in one call, so one such entry registers ZERO Ollama
    models -- and the `already recorded` guard made it stick forever."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_script("_probe_ollama_under_test",
                               "probe-ollama-reasoning.py")

    def _entry(self) -> dict:
        return {
            "capability": self.mod.Capability.STRUCTURED,
            "probes": {"24": {"32768": {"ctx": 32768, "capability": "structured",
                                        "fully_on_gpu": True}}},
        }

    def _run(self, resp: dict, entry: dict) -> bool:
        orig = self.mod.chat_probe
        self.mod.chat_probe = lambda *a, **kw: resp
        try:
            return self.mod.maybe_probe_disable(
                "http://x", "m", entry, "p", 32, 5.0)
        finally:
            self.mod.chat_probe = orig

    def test_error_leaves_field_absent_and_records_evidence(self) -> None:
        entry = self._entry()
        self.assertTrue(self._run({"error": "HTTP 500"}, entry))
        self.assertNotIn("disable_verified", entry)
        self.assertEqual(entry["evidence_disable"]["error"], "HTTP 500")
        # ...and the whole cache still round-trips as JSON the router can
        # decode into a *bool (no string sentinel anywhere).
        json.loads(json.dumps(entry))

    def test_error_is_retried_on_the_next_pass(self) -> None:
        entry = self._entry()
        self._run({"error": "HTTP 500"}, entry)
        # Second pass: the probe must run again (not short-circuit).
        self.assertTrue(self._run({"message": {"thinking": "", "content": "4"}},
                                  entry))
        self.assertIs(entry["disable_verified"], True)

    def test_legacy_string_sentinel_self_heals(self) -> None:
        entry = self._entry()
        entry["disable_verified"] = "error"   # pre-fix cache shape
        self.assertTrue(self._run({"message": {"thinking": "", "content": "4"}},
                                  entry))
        self.assertIs(entry["disable_verified"], True)

    def test_recorded_bool_is_not_reprobed(self) -> None:
        entry = self._entry()
        entry["disable_verified"] = False
        self.assertFalse(self._run({"message": {"thinking": ""}}, entry))

    def test_flash_attention_is_stamped_on_the_cell(self) -> None:
        import os
        mod = self.mod
        orig_chat, orig_vram = mod.chat_probe, mod.measure_vram
        mod.chat_probe = lambda *a, **kw: {"message": {"thinking": "t",
                                                       "content": "c"}}
        mod.measure_vram = lambda *a, **kw: {
            "size_bytes": 1, "size_vram_bytes": 1, "actual_total_gb": 1.0,
            "actual_vram_gb": 1.0, "fully_on_gpu": True, "actual_context": 32768}
        prev = os.environ.get("OLLAMA_FLASH_ATTENTION")
        try:
            os.environ["OLLAMA_FLASH_ATTENTION"] = "1"
            on = mod.probe_one_context("http://x", "m", "d", "p", 32, 5.0, 32768)
            os.environ["OLLAMA_FLASH_ATTENTION"] = "0"
            off = mod.probe_one_context("http://x", "m", "d", "p", 32, 5.0, 32768)
        finally:
            mod.chat_probe, mod.measure_vram = orig_chat, orig_vram
            if prev is None:
                os.environ.pop("OLLAMA_FLASH_ATTENTION", None)
            else:
                os.environ["OLLAMA_FLASH_ATTENTION"] = prev
        self.assertIs(on["flash_attention"], True)
        self.assertIs(off["flash_attention"], False)


class TestKvCacheDtypeThreading(unittest.TestCase):
    """build_args must honour an explicit per-launch KV dtype (what the
    load probe passes from the cell's stamp) and fall back to the pass
    default when None."""

    def setUp(self) -> None:
        self.vllm = _load_script("_probe_vllm_under_test",
                                 "probe-vllm-reasoning.py")
        self.sglang = _load_script("_probe_sglang_under_test",
                                   "probe-sglang-reasoning.py")

    def _kv(self, args: list[str]) -> str | None:
        for i, a in enumerate(args):
            if a == "--kv-cache-dtype" and i + 1 < len(args):
                return args[i + 1]
        return None

    def test_vllm_explicit_dtype_wins(self) -> None:
        args = self.vllm.vllm_command_args("m", 32768, 0.9, kv_cache_dtype="auto")
        self.assertEqual(self._kv(args), "auto")

    def test_vllm_none_uses_pass_default(self) -> None:
        args = self.vllm.vllm_command_args("m", 32768, 0.9)
        self.assertEqual(self._kv(args), self.vllm.KV_CACHE_DTYPE)

    def test_sglang_empty_dtype_emits_no_flag(self) -> None:
        args = self.sglang.sglang_command_args("m", 32768, 0.9, kv_cache_dtype="")
        self.assertIsNone(self._kv(args))

    def test_sglang_explicit_dtype_emits_flag(self) -> None:
        args = self.sglang.sglang_command_args(
            "m", 32768, 0.9, kv_cache_dtype="fp8_e5m2")
        self.assertEqual(self._kv(args), "fp8_e5m2")


if __name__ == "__main__":
    unittest.main()
