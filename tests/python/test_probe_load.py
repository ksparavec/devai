"""Unit tests for the serving-time LOAD probe pure logic.

Covers the deterministic, GPU-free surface of scripts/_probe_load.py:
prompt construction, needle scoring, the predicted-logits diagnostic,
flag extraction, OOM-marker detection, and serving-failure
classification. The container-launch paths (load_probe_one_cell,
run_load_probe_pass) need a real GPU + podman and are exercised by
`make probe-load-vllm`, not here.

Stdlib unittest only. Run with:
    python3 -m unittest tests.python.test_probe_load
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _probe_load as L  # noqa: E402


class TestCorpus(unittest.TestCase):
    """The corpus is fetched on demand into the user cache, not vendored.
    These tests cover the boilerplate strip and the cache-hit path
    without touching the network."""

    def test_sources_declared(self) -> None:
        self.assertIn("moby-dick.txt", L._CORPUS_SOURCES)
        self.assertIn("war-and-peace.txt", L._CORPUS_SOURCES)
        for urls in L._CORPUS_SOURCES.values():
            self.assertTrue(all(u.startswith("https://") for u in urls))

    def test_cache_dir_not_under_var_cache_devai(self) -> None:
        # /var/cache/devai holds only external-volume mount points; the
        # corpus must default elsewhere (user cache). See CLAUDE.md.
        self.assertNotIn("/var/cache/devai", str(L._CORPUS_DIR))

    def test_strip_gutenberg_boilerplate(self) -> None:
        raw = ("license preamble junk\n"
               "*** START OF THE PROJECT GUTENBERG EBOOK MOBY DICK ***\n"
               "Call me Ishmael.\nWhale.\n"
               "*** END OF THE PROJECT GUTENBERG EBOOK MOBY DICK ***\n"
               "trailing license junk")
        out = L._strip_gutenberg_boilerplate(raw)
        self.assertIn("Call me Ishmael.", out)
        self.assertNotIn("preamble junk", out)
        self.assertNotIn("trailing license junk", out)
        self.assertNotIn("GUTENBERG", out)

    def test_load_corpus_reads_cache_without_download(self) -> None:
        # Pre-populate a temp cache dir so no mirror is contacted.
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for name in L._CORPUS_SOURCES:
                (d / name).write_text("Lorem ipsum dolor. " * 30_000)
            orig = L._CORPUS_DIR
            L._CORPUS_DIR = d
            try:
                text = L.load_corpus()
            finally:
                L._CORPUS_DIR = orig
            self.assertGreater(len(text), 1_000_000)

    def test_ensure_corpus_file_uses_cache_hit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "moby-dick.txt").write_text("x" * (L._CORPUS_MIN_CHARS + 1))
            # URLs are bogus; a cache hit must avoid contacting them.
            p = L._ensure_corpus_file(
                "moby-dick.txt", ["https://invalid.invalid/x"], d)
            self.assertEqual(p, d / "moby-dick.txt")


class TestBuildHaystackPrompt(unittest.TestCase):
    def setUp(self) -> None:
        # Deterministic synthetic corpus avoids a multi-MB disk read per
        # test while still exercising the fill/insert/repeat logic.
        self.corpus = ("The quick brown fox jumps over the lazy dog. " * 50)

    def test_needle_and_question_present(self) -> None:
        p = L.build_haystack_prompt(self.corpus, 4096, depth=0.5)
        self.assertIn(L._NEEDLE_CODE, p)
        self.assertIn("OPERATOR NOTE", p)
        self.assertTrue(p.rstrip().endswith("nothing else."))

    def test_larger_ctx_yields_longer_prompt(self) -> None:
        small = L.build_haystack_prompt(self.corpus, 4096)
        large = L.build_haystack_prompt(self.corpus, 32768)
        self.assertGreater(len(large), len(small))

    def test_fill_scales_with_chars_per_token(self) -> None:
        ctx = 16384
        p = L.build_haystack_prompt(self.corpus, ctx, depth=0.5)
        body_tokens = ctx - L._OUTPUT_HEADROOM_TOKENS
        expected_body_chars = int(body_tokens * L._CHARS_PER_TOKEN)
        # Prompt = body + needle sentence + question; should exceed the
        # body estimate but stay within a small constant overhead of it.
        self.assertGreaterEqual(len(p), expected_body_chars)
        overhead = len(L._NEEDLE_SENTENCE) + len(L._QUESTION) + 64
        self.assertLessEqual(len(p), expected_body_chars + overhead)

    def test_depth_moves_needle_position(self) -> None:
        top = L.build_haystack_prompt(self.corpus, 8192, depth=0.1)
        bottom = L.build_haystack_prompt(self.corpus, 8192, depth=0.9)
        self.assertLess(top.index(L._NEEDLE_CODE) / len(top), 0.5)
        self.assertGreater(bottom.index(L._NEEDLE_CODE) / len(bottom), 0.5)


class TestFullWindowFill(unittest.TestCase):
    """Tokenizer-verified sizing fills the KV pool to ~99% of ctx so the
    load probe exercises the true ceiling, not the ~88% a char estimate
    reaches. No real network -- _count_tokens / http_post are stubbed."""

    def setUp(self) -> None:
        self.corpus = "Word " * 500_000

    def test_falls_back_to_char_estimate_without_tokenizer(self) -> None:
        orig = L._count_tokens
        L._count_tokens = lambda b, m, p: None
        try:
            prompt, count, method = L.build_full_window_prompt(
                "http://x", "m", self.corpus, 8192, 0.5)
        finally:
            L._count_tokens = orig
        self.assertEqual(method, "char-estimate")
        self.assertIsNone(count)
        self.assertIn(L._NEEDLE_CODE, prompt)

    def test_converges_to_target_and_never_overshoots(self) -> None:
        # Simulate a tokenizer at ~4.0 chars/token -- denser than the 3.5
        # seed, so the first pass undershoots and the loop must grow it.
        calls = {"n": 0}

        def fake_count(base_url, model_name, prompt):
            calls["n"] += 1
            return len(prompt) // 4

        orig = L._count_tokens
        L._count_tokens = fake_count
        try:
            ctx = 65536
            prompt, count, method = L.build_full_window_prompt(
                "http://x", "m", self.corpus, ctx, 0.5)
        finally:
            L._count_tokens = orig
        target = ctx - L._OUTPUT_HEADROOM_TOKENS
        self.assertEqual(method, "tokenized")
        self.assertLessEqual(count, target)               # never over ctx
        self.assertGreaterEqual(count, target - L._TOKENIZE_TOL)  # near target
        # Fill reaches within the tokenize tolerance of the target window
        # (ctx - _OUTPUT_HEADROOM_TOKENS). Expressed as a fraction so it tracks
        # the headroom instead of hardcoding a fill % (headroom was raised to
        # 2304 for reasoning-model needle recall).
        self.assertGreaterEqual(count / ctx, (target - L._TOKENIZE_TOL) / ctx)
        self.assertGreater(calls["n"], 1)                 # iterated

    def test_calibration_fallback_when_no_tokenize(self) -> None:
        # /tokenize absent (SGLang) but the calibration chat works -> size
        # from the measured chars/token, method="calibrated".
        orig_count, orig_cal = L._count_tokens, L._calibrate_chars_per_token
        L._count_tokens = lambda b, m, p: None
        L._calibrate_chars_per_token = lambda b, m, c: 4.0
        try:
            ctx = 65536
            prompt, count, method = L.build_full_window_prompt(
                "http://x", "m", self.corpus, ctx, 0.5)
        finally:
            L._count_tokens, L._calibrate_chars_per_token = orig_count, orig_cal
        self.assertEqual(method, "calibrated")
        self.assertIsNone(count)
        # Sized at ~4 chars/token to just under the token target -> the
        # char length should be near (ctx - headroom - tol) * 4.
        approx = (ctx - L._OUTPUT_HEADROOM_TOKENS - L._TOKENIZE_TOL) * 4.0
        self.assertGreater(len(prompt), approx * 0.9)

    def test_char_estimate_when_calibration_also_fails(self) -> None:
        orig_count, orig_cal = L._count_tokens, L._calibrate_chars_per_token
        L._count_tokens = lambda b, m, p: None
        L._calibrate_chars_per_token = lambda b, m, c: None
        try:
            _, count, method = L.build_full_window_prompt(
                "http://x", "m", self.corpus, 8192, 0.5)
        finally:
            L._count_tokens, L._calibrate_chars_per_token = orig_count, orig_cal
        self.assertEqual(method, "char-estimate")
        self.assertIsNone(count)

    def test_count_tokens_none_on_transport_error(self) -> None:
        orig = L.http_post

        def boom(url, body, timeout):
            raise OSError("connection refused")

        L.http_post = boom
        try:
            self.assertIsNone(L._count_tokens("http://x", "m", "hi"))
        finally:
            L.http_post = orig

    def test_count_tokens_parses_count_then_tokens(self) -> None:
        orig = L.http_post
        L.http_post = lambda url, body, timeout: {"count": 42}
        try:
            self.assertEqual(L._count_tokens("http://x", "m", "hi"), 42)
            L.http_post = lambda url, body, timeout: {"tokens": [1, 2, 3]}
            self.assertEqual(L._count_tokens("http://x", "m", "hi"), 3)
            L.http_post = lambda url, body, timeout: {"error": "no"}
            self.assertIsNone(L._count_tokens("http://x", "m", "hi"))
        finally:
            L.http_post = orig


class TestScoreNeedle(unittest.TestCase):
    def test_hit(self) -> None:
        self.assertEqual(
            L.score_needle(f"The code is {L._NEEDLE_CODE}."), 1.0)

    def test_miss(self) -> None:
        self.assertEqual(L.score_needle("I could not find a code."), 0.0)

    def test_empty(self) -> None:
        self.assertEqual(L.score_needle(""), 0.0)
        self.assertEqual(L.score_needle(None), 0.0)


class TestPredictedLogits(unittest.TestCase):
    def _write_config(self, root: Path, name: str, payload: dict) -> None:
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(payload))

    def test_top_level_vocab(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_config(Path(td), "m", {"vocab_size": 152064})
            gb = L.predicted_logits_gb(td, "m", 512)
            self.assertEqual(gb, round(512 * 152064 * 4 / 1e9, 3))

    def test_nested_text_config_vocab(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_config(
                Path(td), "m", {"text_config": {"vocab_size": 256000}})
            self.assertEqual(L._read_vocab_size(td, "m"), 256000)

    def test_none_when_vocab_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_config(Path(td), "m", {"hidden_size": 4096})
            self.assertIsNone(L.predicted_logits_gb(td, "m", 512))

    def test_none_when_mnbt_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_config(Path(td), "m", {"vocab_size": 152064})
            self.assertIsNone(L.predicted_logits_gb(td, "m", None))

    def test_none_when_config_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(L._read_vocab_size(td, "absent"))


class TestFlagValue(unittest.TestCase):
    def test_found(self) -> None:
        args = ["--model", "/m", "--max-num-batched-tokens", "512", "--x"]
        self.assertEqual(L._flag_value(args, "--max-num-batched-tokens"), "512")

    def test_absent(self) -> None:
        self.assertIsNone(L._flag_value(["--x", "1"], "--max-num-batched-tokens"))

    def test_flag_at_end_no_value(self) -> None:
        self.assertIsNone(L._flag_value(["--a", "--max-num-batched-tokens"],
                                        "--max-num-batched-tokens"))


class TestOomMarkers(unittest.TestCase):
    def test_case_insensitive_cuda_oom(self) -> None:
        self.assertEqual(
            L._logs_show_oom("blah CUDA out of memory blah"), "out of memory")

    def test_engine_dead(self) -> None:
        self.assertEqual(
            L._logs_show_oom("ERROR EngineDeadError: ..."), "enginedeaderror")

    def test_clean_logs(self) -> None:
        self.assertIsNone(L._logs_show_oom("INFO server ready on :11434"))


class TestDetectServingFailure(unittest.TestCase):
    def test_request_error_short_circuits(self) -> None:
        # request_error is checked before container_state, so this needs
        # no podman.
        failed, reason = L._detect_serving_failure(
            "podman", "c", None, "URLError: connection refused", "")
        self.assertTrue(failed)
        self.assertIn("request_error", reason)

    def test_api_error_body(self) -> None:
        failed, reason = L._detect_serving_failure(
            "podman", "c", {"error": "out of memory"}, None, "")
        self.assertTrue(failed)
        self.assertIn("api_error", reason)

    def test_ok_path_when_container_running(self) -> None:
        # Monkeypatch container_state so no podman is required.
        orig = L.container_state
        L.container_state = lambda runtime, name: "running"
        try:
            good = {"choices": [{"message": {"content": L._NEEDLE_CODE}}]}
            failed, reason = L._detect_serving_failure(
                "podman", "c", good, None, "INFO ready")
        finally:
            L.container_state = orig
        self.assertFalse(failed)
        self.assertEqual(reason, "")

    def test_container_died_is_failure(self) -> None:
        orig = L.container_state
        L.container_state = lambda runtime, name: "exited"
        try:
            failed, reason = L._detect_serving_failure(
                "podman", "c", {"choices": []}, None, "")
        finally:
            L.container_state = orig
        self.assertTrue(failed)
        self.assertIn("container_state=exited", reason)

    def test_oom_marker_with_no_choices(self) -> None:
        orig = L.container_state
        L.container_state = lambda runtime, name: "running"
        try:
            # Engine returned 200 but with no choices AND the log shows an
            # OOM -> treat as failure.
            failed, reason = L._detect_serving_failure(
                "podman", "c", {"choices": []}, None,
                "torch.OutOfMemoryError: CUDA out of memory")
        finally:
            L.container_state = orig
        self.assertTrue(failed)
        self.assertIn("oom_marker", reason)


class TestCellKvCacheDtype(unittest.TestCase):
    """The load probe must relaunch under the dtype the fit cell was
    MEASURED with, not this pass's PROBE_KV_CACHE_TYPE default -- the
    router serves under the STAMPED dtype (resolveKVCacheType), so serving
    numbers taken under a different one describe nothing real."""

    def test_stamp_wins_over_pass_default(self) -> None:
        self.assertEqual(
            L._cell_kv_cache_dtype({"kv_cache_type": "auto"}, "vllm"), "auto")
        self.assertEqual(
            L._cell_kv_cache_dtype({"kv_cache_type": "fp8_e5m2"}, "sglang"),
            "fp8_e5m2")

    def test_unstamped_legacy_cell_matches_the_router(self) -> None:
        # gpu-arbiter synthesizeHFFromCache: "" decodes to fp8 on vLLM (the
        # prober's historical hardcode) and to the engine default on SGLang.
        self.assertEqual(L._cell_kv_cache_dtype({}, "vllm"), "fp8")
        self.assertEqual(L._cell_kv_cache_dtype({}, "sglang"), "")
        self.assertEqual(L._cell_kv_cache_dtype({"kv_cache_type": ""}, "vllm"),
                         "fp8")

    def test_non_string_stamp_falls_back(self) -> None:
        self.assertEqual(L._cell_kv_cache_dtype({"kv_cache_type": 8}, "vllm"),
                         "fp8")


class TestRunLoadProbePassWriteLogic(unittest.TestCase):
    """GPU-free coverage of run_load_probe_pass's single-cell binary-search
    write logic: augment-in-place, MOVE-down when serving caps below the fit
    ctx, serves-nowhere -> serving_ok=False + `oom` ledger, and stale-error
    clearing. The container-launching load_probe_one_cell is replaced with a
    scripted fake.
    """

    def _run(self, *, cells, scripted, force=False, ctx=None):
        """Drive run_load_probe_pass against a synthetic cache. `cells` is the
        initial probes['24'] map; `scripted` maps ctx-int -> serving rec.
        Returns (cache, calls, ledger_records).
        """
        import argparse

        calls: list[int] = []
        self.kv_seen: list[str | None] = []
        ledger_records: list[tuple] = []
        cache = {
            "vendor/m@sha1": {
                "schema_version": 2,
                "repo": "vendor/m",
                "sha": "sha1",
                "aliases": ["m"],
                "max_context": 262144,
                "capability": "inline",
                "probes": {"24": cells},
            }
        }

        saved = {
            "assert_no_active_backends": L.assert_no_active_backends,
            "load_catalog_hf_rows": L.load_catalog_hf_rows,
            "load_corpus": L.load_corpus,
            "is_downloaded": L.is_downloaded,
            "model_size_gb_from_row": L.model_size_gb_from_row,
            "load_cache": L.load_cache,
            "save_cache": L.save_cache,
            "load_probe_one_cell": L.load_probe_one_cell,
            "_load_ledger": L._load_ledger,
            "_save_ledger": L._save_ledger,
            "_ledger_reason": L._ledger_reason,
            "_ledger_record": L._ledger_record,
            "_ledger_clear": L._ledger_clear,
            "effective_position_limit": L.effective_position_limit,
        }
        L.assert_no_active_backends = lambda runtime: None
        L.load_catalog_hf_rows = lambda catalog, name: [
            {"name": "m", "repo": "vendor/m", "sha": "sha1", "parsers": {}}
        ]
        L.load_corpus = lambda: "filler text. " * 100
        L.is_downloaded = lambda name, models_dir: True
        L.model_size_gb_from_row = lambda row: 5.0
        L.load_cache = lambda path: cache
        L.save_cache = lambda path, c: None
        L._load_ledger = lambda: {}
        L._save_ledger = lambda ledger, host_vram_gb=None: None
        L._ledger_reason = lambda ledger, name, backend: None
        L._ledger_record = lambda ledger, name, backend, reason, **kw: (
            ledger_records.append((name, backend, reason)))
        L._ledger_clear = lambda ledger, name, backend: None
        # No config.json in the temp models_dir -> no position_limit cap; the
        # serving search is bounded only by the fit ctx.
        L.effective_position_limit = lambda name, models_dir: None

        def fake_probe(spec, *, requested_ctx, **kw):
            calls.append(requested_ctx)
            self.kv_seen.append(kw.get("kv_cache_dtype"))
            return dict(scripted[requested_ctx])

        L.load_probe_one_cell = fake_probe

        with tempfile.TemporaryDirectory() as td:
            args = argparse.Namespace(
                runtime="podman", repo="", catalog=Path(td) / "models.yaml",
                cache=Path(td) / "cache.json",
                models_dir=td, host_vram_gb=24, ctx=ctx,
                image="img", container_name="c", probe_port=18000,
                force=force, no_cache_write=False, needle_depth=0.5,
            )
            import types
            spec = types.SimpleNamespace(name="vllm")
            try:
                L.run_load_probe_pass(spec, args)
            finally:
                for k, v in saved.items():
                    setattr(L, k, v)
        return cache, calls, ledger_records

    def test_additive_augment_keeps_fit_keys(self) -> None:
        # Fits and serves at the same ctx: augment the one cell in place.
        cells = {
            "32768": {"ctx": 32768, "fits": True, "actual_vram_gb": 20.0,
                      "capability": "inline"},
        }
        scripted = {32768: {"serving_ok": True, "serving_peak_gb": 21.0,
                            "transient_gb": 1.0, "needle_score": 1.0}}
        cache, calls, _ = self._run(cells=cells, scripted=scripted)
        cell = cache["vendor/m@sha1"]["probes"]["24"]["32768"]
        self.assertTrue(cell["fits"])
        self.assertEqual(cell["actual_vram_gb"], 20.0)
        self.assertEqual(cell["capability"], "inline")
        self.assertTrue(cell["serving_ok"])
        self.assertEqual(cell["transient_gb"], 1.0)
        self.assertEqual(calls, [32768])

    def test_serving_below_fit_moves_single_cell_down(self) -> None:
        # Fits at 128K but serves only at 64K: the one cell MOVES to 64K,
        # carries the fit capability, and max_context shrinks.
        cells = {"131072": {"ctx": 131072, "fits": True, "capability": "inline",
                            "actual_vram_gb": 22.0}}
        scripted = {
            131072: {"serving_ok": False, "serving_error": "oom_marker: CUDA out of memory"},
            65536:  {"serving_ok": True, "serving_peak_gb": 21.0,
                     "serving_baseline_gb": 19.0, "needle_score": 1.0},
            98304:  {"serving_ok": False, "serving_error": "oom_marker: CUDA out of memory"},
        }
        cache, calls, _ = self._run(cells=cells, scripted=scripted)
        entry = cache["vendor/m@sha1"]
        band = entry["probes"]["24"]
        self.assertEqual(set(band), {"65536"})            # exactly one cell, moved
        self.assertEqual(band["65536"]["ctx"], 65536)
        self.assertTrue(band["65536"]["serving_ok"])
        self.assertTrue(band["65536"]["fits"])             # carried from fit cell
        self.assertEqual(band["65536"]["capability"], "inline")
        self.assertEqual(band["65536"]["actual_vram_gb"], 19.0)  # served baseline
        self.assertEqual(entry["max_context"], 65536)      # ceiling shrunk
        self.assertEqual(calls, [131072, 65536, 98304])    # top-then-bisect

    def test_serving_nowhere_marks_fail_and_excludes(self) -> None:
        # Fits at 32K but OOMs when actually serving -> serving_ok=False + oom.
        cells = {"32768": {"ctx": 32768, "fits": True, "capability": "inline"}}
        scripted = {32768: {"serving_ok": False, "serving_error": "oom_marker: CUDA out of memory"}}
        cache, calls, ledger_records = self._run(cells=cells, scripted=scripted)
        cell = cache["vendor/m@sha1"]["probes"]["24"]["32768"]
        self.assertTrue(cell["fits"])                      # still fits...
        self.assertFalse(cell["serving_ok"])               # ...but cannot serve
        self.assertEqual(calls, [32768])
        self.assertIn(("m", "vllm", "oom"), ledger_records)

    def test_infra_serving_failure_writes_no_exclusion(self) -> None:
        # An `oom` exclusion is sha-stable: it survives until the model's
        # weights change. Recording one for a transient 500, a queue-full
        # 503 or an engine bug therefore hides a model that fits perfectly
        # well, indefinitely. _detect_serving_failure turns ANY error body
        # into a failure, so this path was reachable from a single blip.
        cells = {"32768": {"ctx": 32768, "fits": True, "capability": "inline"}}
        scripted = {32768: {"serving_ok": False,
                            "serving_error": "api_error: internal server error"}}
        cache, _, ledger_records = self._run(cells=cells, scripted=scripted)
        cell = cache["vendor/m@sha1"]["probes"]["24"]["32768"]
        self.assertFalse(cell["serving_ok"], "the cell is still unserveable")
        self.assertEqual(
            [r for r in ledger_records if r[2] == "oom"], [],
            "a non-OOM serving failure must not write an oom exclusion")

    def test_degenerate_serving_excludes_as_manual_not_oom(self) -> None:
        # The Ornith case: the engine served without OOMing and produced
        # garbage. `oom` would be a false statement about the cause AND is
        # re-checked only on a new sha; `manual` is sha-stable and carries
        # the evidence.
        cells = {"32768": {"ctx": 32768, "fits": True, "capability": "inline"}}
        scripted = {32768: {"serving_ok": False,
                            "serving_degenerate": True,
                            "serving_degenerate_reason": "empty content at cap",
                            "serving_error": "degenerate"}}
        _, _, ledger_records = self._run(cells=cells, scripted=scripted)
        reasons = [r[2] for r in ledger_records]
        self.assertIn("manual", reasons)
        self.assertNotIn("oom", reasons)

    def test_relaunches_under_the_cells_stamped_kv_dtype(self) -> None:
        # Cell fit-probed with PROBE_KV_CACHE_TYPE=auto: a fleet-wide
        # `make probe-load-vllm` (pass default fp8) must NOT silently
        # relaunch it under fp8.
        cells = {"32768": {"ctx": 32768, "fits": True, "kv_cache_type": "auto"}}
        scripted = {32768: {"serving_ok": True, "serving_peak_gb": 20.0}}
        self._run(cells=cells, scripted=scripted)
        self.assertEqual(self.kv_seen, ["auto"])

    def test_unstamped_cell_uses_the_router_fallback(self) -> None:
        cells = {"32768": {"ctx": 32768, "fits": True}}
        scripted = {32768: {"serving_ok": True, "serving_peak_gb": 20.0}}
        self._run(cells=cells, scripted=scripted)
        self.assertEqual(self.kv_seen, ["fp8"])   # spec.name == "vllm"

    def test_ctx_caps_the_search_grid(self) -> None:
        # --ctx / PROBE_CONTEXTS must CAP the serving search exactly as it
        # caps the fit search. Before this was wired, the load prober parsed
        # --ctx and discarded it, so `--ctx 32K` still launched (and
        # OOM-killed) 256K/128K/... containers -- real GPU minutes at tiers
        # the operator explicitly excluded.
        cells = {"262144": {"ctx": 262144, "fits": True, "capability": "inline"}}
        scripted = {32768: {"serving_ok": True, "serving_peak_gb": 20.0,
                            "needle_score": 1.0}}
        _, calls, _ = self._run(cells=cells, scripted=scripted, ctx="32K")
        self.assertEqual(calls, [32768])

    def test_multi_tier_ctx_grid_keeps_the_tiers_below_the_ceiling(self) -> None:
        # `--ctx 32K,64K` -> ceiling 64K -> grid {32K, 64K}; the top tier is
        # tried first (fast path), so a serving 64K resolves in one launch.
        cells = {"262144": {"ctx": 262144, "fits": True, "capability": "inline"}}
        scripted = {65536: {"serving_ok": True, "serving_peak_gb": 21.0},
                    32768: {"serving_ok": True, "serving_peak_gb": 20.0}}
        _, calls, _ = self._run(cells=cells, scripted=scripted, ctx="32K,64K")
        self.assertEqual(calls, [65536])

    def test_off_grid_ctx_tier_is_still_probed(self) -> None:
        # An explicit non-grid tier is unioned into the grid (same as the
        # fit pass) instead of being silently rounded away.
        cells = {"262144": {"ctx": 262144, "fits": True, "capability": "inline"}}
        scripted = {40960: {"serving_ok": True, "serving_peak_gb": 20.0}}
        _, calls, _ = self._run(cells=cells, scripted=scripted, ctx="40K")
        self.assertEqual(calls, [40960])

    def test_force_clears_stale_serving_error(self) -> None:
        cells = {
            "32768": {"ctx": 32768, "fits": True, "serving_ok": False,
                      "serving_error": "old: container_state=exited"},
        }
        scripted = {32768: {"serving_ok": True, "serving_peak_gb": 20.0}}
        cache, calls, _ = self._run(cells=cells, scripted=scripted, force=True)
        cell = cache["vendor/m@sha1"]["probes"]["24"]["32768"]
        self.assertTrue(cell["serving_ok"])
        self.assertNotIn("serving_error", cell)  # stale error cleared


if __name__ == "__main__":
    unittest.main()


class ServingSearchStepsDownTest(unittest.TestCase):
    """A model that FITS at its ceiling but only SERVES lower must be
    recorded at the serving ctx, not discarded.

    Ornith-1.0-9B-NVFP4 fits at 256K on SGLang and degenerates there
    (emits `!` until the request times out), but serves cleanly at 128K
    -- 21.95 GB peak, 119,555 input tokens. If the search stopped at a
    failing ceiling the model would vanish from the picker entirely on
    the strength of one context it happens to fail.

    Pinned because the consequence is silent: a lost model looks
    identical to a model that was never probed.
    """

    def setUp(self):
        import importlib.util as _il
        spec = _il.spec_from_file_location(
            "contexts_stepdown", REPO_ROOT / "scripts" / "_contexts.py")
        self.cx = _il.module_from_spec(spec)
        spec.loader.exec_module(self.cx)

    def test_ceiling_failure_searches_downward(self):
        probed = []

        def works(ctx):
            probed.append(ctx)
            return ctx <= 131072

        got = self.cx.binary_search_max_ctx(
            works, position_limit=262144,
            grid=self.cx.BINARY_SEARCH_CONTEXTS)
        self.assertEqual(got, 131072)
        self.assertGreater(len(probed), 1,
                           "a failing ceiling must not end the search")
        self.assertIn(262144, probed, "the ceiling is probed first")

    def test_serving_everywhere_resolves_in_one_probe(self):
        probed = []

        def works(ctx):
            probed.append(ctx)
            return True

        got = self.cx.binary_search_max_ctx(
            works, grid=self.cx.BINARY_SEARCH_CONTEXTS)
        self.assertEqual(got, max(self.cx.BINARY_SEARCH_CONTEXTS))
        self.assertEqual(len(probed), 1, "the common case costs one launch")

    def test_serving_nowhere_returns_none(self):
        got = self.cx.binary_search_max_ctx(
            lambda ctx: False, grid=self.cx.BINARY_SEARCH_CONTEXTS)
        self.assertIsNone(got)

    def test_single_tier_grid_cannot_step_down(self):
        """Worth stating: with a one-element grid a failure yields None
        and there is nowhere to step. An operator narrowing the grid to a
        single ctx is asking for a pass/fail on that ctx, not a search."""
        probed = []

        def works(ctx):
            probed.append(ctx)
            return False

        self.assertIsNone(
            self.cx.binary_search_max_ctx(works, grid=(262144,)))
        self.assertEqual(probed, [262144])
