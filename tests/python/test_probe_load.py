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
        # ~99% of the window -> the pool is genuinely exercised.
        self.assertGreater(count / ctx, 0.98)
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


class TestRunLoadProbePassWriteLogic(unittest.TestCase):
    """GPU-free coverage of run_load_probe_pass's cache-write logic:
    additive augmentation, ascending stop-at-OOM with implied marking,
    and stale-error clearing. The container-launching load_probe_one_cell
    is replaced with a scripted fake.
    """

    def _run(self, *, cells, scripted, force=False):
        """Drive run_load_probe_pass against a synthetic cache. `cells`
        is the initial probes['24'] map; `scripted` maps ctx-int ->
        serving rec returned by the fake probe. Returns (cache, calls).
        """
        import argparse

        calls: list[int] = []
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

        def fake_probe(spec, *, requested_ctx, **kw):
            calls.append(requested_ctx)
            return dict(scripted[requested_ctx])

        L.load_probe_one_cell = fake_probe

        with tempfile.TemporaryDirectory() as td:
            args = argparse.Namespace(
                runtime="podman", repo="", catalog=Path(td) / "models.yaml",
                cache=Path(td) / "cache.json",
                models_dir=td, host_vram_gb=24, ctx="32K,64K,128K",
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
        return cache, calls

    def test_additive_augment_keeps_fit_keys(self) -> None:
        cells = {
            "32768": {"ctx": 32768, "fits": True, "actual_vram_gb": 20.0,
                      "capability": "inline"},
        }
        scripted = {32768: {"serving_ok": True, "serving_peak_gb": 21.0,
                            "transient_gb": 1.0, "needle_score": 1.0}}
        cache, calls = self._run(cells=cells, scripted=scripted)
        cell = cache["vendor/m@sha1"]["probes"]["24"]["32768"]
        # Original fit fields survive...
        self.assertTrue(cell["fits"])
        self.assertEqual(cell["actual_vram_gb"], 20.0)
        self.assertEqual(cell["capability"], "inline")
        # ...and the serving fields are added.
        self.assertTrue(cell["serving_ok"])
        self.assertEqual(cell["transient_gb"], 1.0)
        self.assertEqual(calls, [32768])

    def test_ascending_stop_marks_higher_tiers_implied(self) -> None:
        cells = {
            "32768":  {"ctx": 32768, "fits": True},
            "65536":  {"ctx": 65536, "fits": True},
            "131072": {"ctx": 131072, "fits": True},
        }
        scripted = {
            32768: {"serving_ok": True, "serving_peak_gb": 20.0},
            65536: {"serving_ok": False, "serving_error": "oom_marker: x"},
        }
        cache, calls = self._run(cells=cells, scripted=scripted)
        band = cache["vendor/m@sha1"]["probes"]["24"]
        self.assertTrue(band["32768"]["serving_ok"])
        self.assertFalse(band["65536"]["serving_ok"])
        # 131072 must be marked implied WITHOUT launching a probe for it.
        self.assertFalse(band["131072"]["serving_ok"])
        self.assertEqual(band["131072"]["serving_error"],
                         "implied_by_lower_tier_oom")
        self.assertEqual(calls, [32768, 65536])  # 131072 never probed

    def test_force_clears_stale_serving_error(self) -> None:
        cells = {
            "32768": {"ctx": 32768, "fits": True, "serving_ok": False,
                      "serving_error": "old: container_state=exited"},
        }
        scripted = {32768: {"serving_ok": True, "serving_peak_gb": 20.0}}
        cache, calls = self._run(cells=cells, scripted=scripted, force=True)
        cell = cache["vendor/m@sha1"]["probes"]["24"]["32768"]
        self.assertTrue(cell["serving_ok"])
        self.assertNotIn("serving_error", cell)  # stale error cleared


if __name__ == "__main__":
    unittest.main()
