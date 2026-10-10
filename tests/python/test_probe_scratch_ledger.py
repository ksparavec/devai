"""`--ledger PATH` (Make PROBE_LEDGER): a scratch exclusion ledger for the probers.

Plan Phase 4 step 4v probes the two prepared Qwen3.8 rows on two image tags
(0.28.0 and 0.30.0), each with its own scratch copy of the probe cache AND of
deploy/.model-status.json. Two things must hold, and both are pinned here:

  * a fit pass (here: the exact-context probe) and a load pass given --ledger
    read and write that file only -- deploy/.model-status.json stays
    byte-identical, so no 4v verdict reaches the real ledger or survives a
    rollback;
  * a load-probe OOM is a sha-stable `oom` verdict, and the fit pass skips a
    row its ledger excludes. Recorded in one tag's scratch ledger it must not
    stop the OTHER tag's exact-context probe (its own scratch ledger) from
    probing the row. The control shows the same probe IS stopped when it
    shares the first tag's ledger, so the test can fail.

I24's switch (PROBE_IGNORE_CURATED_PARSERS) writes its verdicts to the scratch
ledger too. The real ledger functions run on temp files; the launches are
scripted fakes. A ledger call without a path -- the real file -- is recorded
and never forwarded, so even a failing test cannot write the operator's
ledger. Stdlib unittest; no container, no GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _model_status as MS  # noqa: E402
import _probe_hf_common as hf  # noqa: E402
import _probe_load as L  # noqa: E402

REAL_LEDGER = REPO_ROOT / "deploy" / ".model-status.json"
ROW = {"name": "m", "repo": "vendor/m", "sha": "sha1", "parsers": {}}
KEY = "vendor/m@sha1"


def _real_ledger_bytes() -> bytes | None:
    return REAL_LEDGER.read_bytes() if REAL_LEDGER.exists() else None


class _GuardedLedger:
    """The real load/save on explicit paths; a pathless call is recorded."""

    def __init__(self) -> None:
        self.pathless_loads = 0
        self.pathless_saves = 0
        self.saved_to: list[Path] = []

    def load(self, *path: Path) -> dict:
        if not path:
            self.pathless_loads += 1
            return {}
        return MS.load_ledger(*path)

    def save(self, ledger: dict, *path: Path, host_vram_gb=None) -> None:
        if not path:
            self.pathless_saves += 1
            return
        self.saved_to.append(Path(path[0]))
        MS.save_ledger(ledger, path[0], host_vram_gb=host_vram_gb)


def _fit_cell(ctx: int) -> dict:
    return {"ctx": ctx, "fits": True, "capability": "structured",
            "actual_context": ctx, "actual_vram_gb": 21.0, "kv_cache_type": "fp8"}


def run_exact_probe(ledger: Path | None, guard: _GuardedLedger, cache: dict,
                    *, ignore_curated: bool = False,
                    scratch_cache: Path | None = None) -> list[int]:
    """hf.run_probe_pass at --ctx-exact 118784; returns the launched contexts."""
    calls: list[int] = []
    names = ("assert_no_active_backends", "install_probe_cleanup", "load_catalog_hf_rows",
             "is_downloaded", "model_kind_from_disk", "model_size_gb_from_row", "load_cache",
             "save_cache", "image_digest_via_cli", "effective_position_limit", "probe_one_cell",
             "_load_ledger", "_save_ledger", "derive_parser_for")
    saved = {n: getattr(hf, n) for n in names}
    hf.assert_no_active_backends = lambda runtime: None
    hf.install_probe_cleanup = lambda: None
    hf.load_catalog_hf_rows = lambda catalog, name: [dict(ROW)]
    hf.is_downloaded = lambda name, models_dir: True
    hf.model_kind_from_disk = lambda name, models_dir: "hf"
    hf.model_size_gb_from_row = lambda r: 16.0
    hf.load_cache = lambda path: cache
    hf.save_cache = lambda path, c: None
    hf.image_digest_via_cli = lambda runtime, image: "sha256:deadbeef"
    hf.effective_position_limit = lambda name, models_dir: 262144
    hf._load_ledger = guard.load
    hf._save_ledger = guard.save
    hf.derive_parser_for = lambda name, models_dir, backend, kind: None

    def fake_probe(spec, *, requested_ctx, **kw):
        calls.append(requested_ctx)
        return {**_fit_cell(requested_ctx), "vram_gb": 24, "reasoning_parser": None,
                "tool_parser": None, "disable_verified": False, "probed_at": "t",
                "startup_seconds": 1.0, "evidence": {}}
    hf.probe_one_cell = fake_probe
    with tempfile.TemporaryDirectory() as td:
        args = argparse.Namespace(
            runtime="podman", repo="vendor/m", catalog=Path(td) / "models.yaml",
            cache=scratch_cache or Path(td) / "cache.json", models_dir=td, vram="", ctx="",
            host_vram_gb=24, image="img", container_name="c", probe_port=18000,
            prompt="hi", force=False, force_arch=False, no_mtp=True,
            no_cache_write=False, ctx_exact=118784, ledger=ledger,
            ignore_curated_parsers=ignore_curated)
        spec = types.SimpleNamespace(name="vllm-devai", schema_version=2,
                                     cache_path=Path(td) / "the-backends-own-cache.json")
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                hf.run_probe_pass(spec, args)
        finally:
            for k, v in saved.items():
                setattr(hf, k, v)
    return calls


def run_load_probe(ledger: Path | None, guard: _GuardedLedger, cache: dict,
                   serving: dict) -> list[int]:
    """L.run_load_probe_pass, every launch answering `serving`; launched ctxs."""
    calls: list[int] = []
    names = ("assert_no_active_backends", "install_probe_cleanup", "load_catalog_hf_rows",
             "load_corpus", "is_downloaded", "model_size_gb_from_row", "load_cache",
             "save_cache", "image_digest_via_cli", "load_probe_one_cell",
             "_load_ledger", "_save_ledger", "effective_position_limit")
    saved = {n: getattr(L, n) for n in names}
    L.assert_no_active_backends = lambda runtime: None
    L.install_probe_cleanup = lambda: None
    L.load_catalog_hf_rows = lambda catalog, name: [dict(ROW)]
    L.load_corpus = lambda: "filler text. " * 100
    L.is_downloaded = lambda name, models_dir: True
    L.model_size_gb_from_row = lambda row: 16.0
    L.load_cache = lambda path: cache
    L.save_cache = lambda path, c: None
    L.image_digest_via_cli = lambda runtime, image: "sha256:deadbeef"
    L._load_ledger = guard.load
    L._save_ledger = guard.save
    L.effective_position_limit = lambda name, models_dir: None

    def fake_probe(spec, *, requested_ctx, **kw):
        calls.append(requested_ctx)
        return dict(serving)
    L.load_probe_one_cell = fake_probe
    with tempfile.TemporaryDirectory() as td:
        args = argparse.Namespace(
            runtime="podman", repo="", catalog=Path(td) / "models.yaml",
            cache=Path(td) / "cache.json", models_dir=td, host_vram_gb=24, ctx="128K",
            image="img", container_name="c", probe_port=18000, force=False,
            no_cache_write=False, needle_depth=0.5, ledger=ledger)
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                L.run_load_probe_pass(types.SimpleNamespace(name="vllm-devai"), args)
        finally:
            for k, v in saved.items():
                setattr(L, k, v)
    return calls


def _cache_with_fit_cell(ctx: int) -> dict:
    return {KEY: {"schema_version": 2, "repo": "vendor/m", "sha": "sha1", "aliases": ["m"],
                  "capability": "structured", "probes": {"24": {str(ctx): _fit_cell(ctx)}}}}


_OOM = {"serving_ok": False, "serving_error": "oom_marker: CUDA out of memory"}


class ScratchLedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = Path(self._td.name)
        self.real_before = _real_ledger_bytes()

    def tearDown(self) -> None:
        self.assertEqual(_real_ledger_bytes(), self.real_before,
                         "deploy/.model-status.json changed")

    def test_exact_context_probe_with_ledger_writes_only_the_scratch_file(self) -> None:
        guard, scratch = _GuardedLedger(), self.tmp / "tag-0.30.0.ledger.json"
        calls = run_exact_probe(scratch, guard, _cache_with_fit_cell(98304))
        self.assertEqual(calls, [118784])
        self.assertEqual((guard.pathless_loads, guard.pathless_saves), (0, 0))
        self.assertEqual(guard.saved_to, [scratch])
        self.assertTrue(scratch.is_file())

    def test_load_probe_with_ledger_writes_only_the_scratch_file(self) -> None:
        guard, scratch = _GuardedLedger(), self.tmp / "tag-0.28.0.ledger.json"
        calls = run_load_probe(scratch, guard, _cache_with_fit_cell(131072), _OOM)
        self.assertEqual(calls[0], 131072)
        self.assertEqual((guard.pathless_loads, guard.pathless_saves), (0, 0))
        self.assertEqual(guard.saved_to, [scratch])
        self.assertEqual(MS.exclusion_reason(MS.load_ledger(scratch), "m", "vllm-devai"), "oom")

    def test_without_ledger_both_passes_use_the_real_ledger_as_before(self) -> None:
        # The default is unchanged: no path is passed, so _model_status's
        # LEDGER_PATH is used (recorded here, never forwarded).
        guard = _GuardedLedger()
        run_exact_probe(None, guard, _cache_with_fit_cell(98304))
        run_load_probe(None, guard, _cache_with_fit_cell(131072), _OOM)
        self.assertEqual((guard.pathless_loads, guard.pathless_saves), (2, 2))
        self.assertEqual(guard.saved_to, [])

    def test_an_oom_in_one_tags_ledger_does_not_stop_the_other_tags_exact_probe(self) -> None:
        old_tag, new_tag = self.tmp / "0.28.0.ledger.json", self.tmp / "0.30.0.ledger.json"
        guard = _GuardedLedger()
        # 0.28.0: the load probe OOMs at the row's context -> `oom` there.
        run_load_probe(old_tag, guard, _cache_with_fit_cell(131072), _OOM)
        self.assertEqual(MS.exclusion_reason(MS.load_ledger(old_tag), "m", "vllm-devai"), "oom")
        # 0.30.0: its own scratch ledger -> the exact-context probe launches.
        self.assertEqual(run_exact_probe(new_tag, guard, _cache_with_fit_cell(98304)), [118784])
        self.assertIsNone(MS.exclusion_reason(MS.load_ledger(new_tag), "m", "vllm-devai"))
        # Control: sharing the 0.28.0 ledger, the same probe skips the row.
        self.assertEqual(run_exact_probe(old_tag, guard, _cache_with_fit_cell(98304)), [])
        self.assertEqual((guard.pathless_loads, guard.pathless_saves), (0, 0))

    def test_the_i24_switch_writes_its_verdicts_to_the_scratch_ledger(self) -> None:
        guard, scratch = _GuardedLedger(), self.tmp / "i24.ledger.json"
        calls = run_exact_probe(scratch, guard, _cache_with_fit_cell(98304),
                                ignore_curated=True, scratch_cache=self.tmp / "i24.cache.json")
        self.assertEqual(calls, [118784])
        self.assertEqual(guard.saved_to, [scratch])
        self.assertEqual((guard.pathless_loads, guard.pathless_saves), (0, 0))

    def test_the_i24_switch_refuses_the_real_ledger(self) -> None:
        guard = _GuardedLedger()
        with self.assertRaises(SystemExit):
            run_exact_probe(REAL_LEDGER, guard, _cache_with_fit_cell(98304),
                            ignore_curated=True, scratch_cache=self.tmp / "i24.cache.json")
        self.assertEqual(guard.saved_to, [])

    def test_a_missing_scratch_ledger_reads_as_empty(self) -> None:
        self.assertEqual(MS.load_ledger(self.tmp / "absent.json")["models"], {})


class LedgerFlagTest(unittest.TestCase):
    def test_every_prober_accepts_ledger_and_mtp(self) -> None:
        import importlib.util
        # probe-sglang-reasoning.py went to attic/sglang/ (Phase 15, M16).
        for script in ("probe-vllm-reasoning.py", "probe-vllm-devai-reasoning.py"):
            with self.subTest(script=script):
                spec = importlib.util.spec_from_file_location(
                    f"ledger_flag_{script.replace('-', '_')}", REPO_ROOT / "scripts" / script)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                ap = hf.build_argparser(mod.SPEC, "doc")
                a = ap.parse_args(["--load", "--mtp", "--ledger", "/s/l.json",
                                   "--cache", "/s/c.json", "--force"])
                self.assertEqual((a.load, a.mtp, a.ledger, a.cache, a.force),
                                 (True, True, Path("/s/l.json"), Path("/s/c.json"), True))
                d = ap.parse_args([])
                self.assertEqual((d.mtp, d.ledger, d.cache), (False, None, mod.SPEC.cache_path))

    def test_ledger_file_args(self) -> None:
        self.assertEqual(hf.ledger_file_args(argparse.Namespace()), ())
        self.assertEqual(hf.ledger_file_args(argparse.Namespace(ledger=None)), ())
        self.assertEqual(hf.ledger_file_args(argparse.Namespace(ledger="/x.json")),
                         (Path("/x.json"),))

    def test_the_fit_pass_refuses_mtp(self) -> None:
        # --mtp is a load-pass flag; the fit pass probes the drafter itself.
        with self.assertRaises(SystemExit) as cm:
            hf.run_probe_pass(types.SimpleNamespace(name="vllm"),
                              argparse.Namespace(mtp=True, ignore_curated_parsers=False))
        self.assertIn("--load", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
