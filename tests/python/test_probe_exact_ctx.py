"""`make probe-vllm PROBE_CTX_EXACT=<ctx>`: confirm ONE predicted context, no search.

The fit probe binary-searches a 32K grid. That is right when nothing is known,
and wrong when the operator has already computed the ceiling from the engine's
own memory accounting (weights, activation peak, bytes per KV token) and only
wants it confirmed -- e.g. a 4.18 GiB pool at 38.6 KiB/token is ~113K tokens,
and a 32K grid can only answer "96K". Two things make an exact probe different
from a one-tier search:

  * a MISS must not be recorded. On the search path a model that fits nowhere
    gets an `oom` ledger verdict and its band is replaced by the failing cell;
    for an exact probe a miss just means the number was a little high, so the
    previous cell stays and the engine's own estimate ("the estimated maximum
    model length is N") is reported for the retry.
  * a value above the model's position limit is refused before launching:
    the search treats that as unsupported_arch, which is a verdict about the
    checkpoint, not about a typo.

Stdlib unittest; no container, no GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _probe_hf_common as hf  # noqa: E402

MAKEFILE = (REPO_ROOT / "Makefile").read_text()


def _vllm_spec():
    spec = importlib.util.spec_from_file_location(
        "probe_vllm_for_exact_test", REPO_ROOT / "scripts" / "probe-vllm-reasoning.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["probe_vllm_for_exact_test"] = mod
    spec.loader.exec_module(mod)
    return mod.SPEC


class EstimateParsingTest(unittest.TestCase):
    _EXCERPT = ("ValueError: To serve at least one request with the model's max seq "
                "len (163840), (5.72 GiB KV cache is needed, which is larger than the "
                "available KV cache memory (4.69 GiB). Based on the available memory, "
                "the estimated maximum model length is 131456. Try increasing "
                "`gpu_memory_utilization` ...")

    def test_reads_the_engines_estimate_from_the_failure_evidence(self) -> None:
        self.assertEqual(hf.engine_max_len_estimate({"log_excerpt": self._EXCERPT}), 131456)

    def test_no_estimate_when_the_failure_was_something_else(self) -> None:
        self.assertIsNone(hf.engine_max_len_estimate(
            {"log_excerpt": "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.37 GiB"}))
        self.assertIsNone(hf.engine_max_len_estimate({}))
        self.assertIsNone(hf.engine_max_len_estimate(None))


class ExactGridTest(unittest.TestCase):
    def test_exact_grid_is_the_one_requested_value(self) -> None:
        self.assertEqual(hf.exact_ctx_grid(112640, position_limit=262144), (112640,))
        self.assertEqual(hf.exact_ctx_grid(112640, position_limit=None), (112640,))

    def test_a_value_past_the_models_ceiling_is_refused_before_any_launch(self) -> None:
        with self.assertRaises(ValueError) as cm:
            hf.exact_ctx_grid(150000, position_limit=131072)
        self.assertIn("131072", str(cm.exception))

    def test_off_grid_values_are_allowed_that_is_the_point(self) -> None:
        # 110K in this repo's K=1024 convention; not a 32K multiple.
        self.assertEqual(hf.exact_ctx_grid(112640, position_limit=262144), (112640,))


class ArgparserTest(unittest.TestCase):
    def test_flag_exists_and_parses_the_repos_k_notation(self) -> None:
        ap = hf.build_argparser(_vllm_spec(), "doc")
        args = ap.parse_args(["--ctx-exact", "110K", "--repo", "x"])
        self.assertEqual(args.ctx_exact, 110 * 1024)
        args = ap.parse_args(["--ctx-exact", "112640", "--repo", "x"])
        self.assertEqual(args.ctx_exact, 112640)

    def test_default_is_off(self) -> None:
        self.assertIsNone(hf.build_argparser(_vllm_spec(), "doc").parse_args([]).ctx_exact)


class WiringTest(unittest.TestCase):
    def test_make_probe_targets_forward_the_knob(self) -> None:
        for target in ("probe-vllm", "probe-sglang"):
            recipe = MAKEFILE.split(f"\n{target}:", 1)[1].split("\n\n", 1)[0]
            self.assertIn("$(if $(PROBE_CTX_EXACT),--ctx-exact $(PROBE_CTX_EXACT),)", recipe, target)


if __name__ == "__main__":
    unittest.main()


class ExactModeEndToEndTest(unittest.TestCase):
    """run_probe_pass in exact mode, with the container launcher replaced by a
    scripted fake (the pattern test_probe_classify.py uses). The interesting
    case is a row that DECLARES MTP: the fit pass hits, the MTP pass misses.
    Measured for real on 2026-09-22: 120K fit without MTP, MTP missed, and the
    first version of exact mode recorded a 120K cell with mtp_fits=false --
    which made the router refuse `::mtp` for a row whose 115K cell had served
    MTP fine. A hit has to mean the declared configuration works there.
    """

    _PRIOR = {"117760": {"ctx": 117760, "fits": True, "capability": "structured",
                         "actual_vram_gb": 21.96, "mtp_fits": True, "probed_at": "x",
                         "evidence": {}}}

    def _run(self, *, ctx_exact, base_fits, mtp_fits, mtp_declared=True, pos_limit=262144):
        P = hf
        calls: list[tuple[int, bool]] = []
        cache = {"vendor/m@sha1": {"schema_version": 2, "repo": "vendor/m", "sha": "sha1",
                                   "aliases": ["m"], "capability": "structured",
                                   "probes": {"24": dict(self._PRIOR)}}}
        row = {"name": "m", "repo": "vendor/m", "sha": "sha1", "parsers": {}}
        if mtp_declared:
            row["mtp"] = {"method": "mtp", "num_speculative_tokens": 3}
        names = ("assert_no_active_backends", "install_probe_cleanup", "load_catalog_hf_rows",
                 "is_downloaded", "model_kind_from_disk", "model_size_gb_from_row", "load_cache",
                 "save_cache", "image_digest_via_cli", "effective_position_limit", "probe_one_cell",
                 "_load_ledger", "_save_ledger", "_ledger_reason", "_ledger_record",
                 "_ledger_clear", "_ledger_is_excluded")
        saved = {n: getattr(P, n) for n in names}
        P.assert_no_active_backends = lambda runtime: None
        P.install_probe_cleanup = lambda: None
        P.load_catalog_hf_rows = lambda catalog, name: [row]
        P.is_downloaded = lambda name, models_dir: True
        P.model_kind_from_disk = lambda name, models_dir: "hf"
        P.model_size_gb_from_row = lambda r: 16.0
        P.load_cache = lambda path: cache
        P.save_cache = lambda path, c: None
        P.image_digest_via_cli = lambda runtime, image: "sha256:deadbeef"
        P.effective_position_limit = lambda name, models_dir: pos_limit
        P._load_ledger = lambda: {}
        P._save_ledger = lambda ledger, host_vram_gb=None: None
        P._ledger_reason = lambda ledger, name, backend: None
        P._ledger_record = lambda ledger, name, backend, reason, **kw: None
        P._ledger_clear = lambda ledger, name, backend: None
        P._ledger_is_excluded = lambda ledger, name, backend, **kw: False

        def fake_probe(spec, *, requested_ctx, mtp_method=None, **kw):
            is_mtp = mtp_method is not None
            calls.append((requested_ctx, is_mtp))
            ok = mtp_fits if is_mtp else base_fits
            rec = {"ctx": requested_ctx, "vram_gb": 24, "fits": ok, "capability": "structured",
                   "actual_context": requested_ctx, "actual_vram_gb": 21.96 if ok else None,
                   "reasoning_parser": "qwen3", "tool_parser": "qwen3_xml",
                   "disable_verified": False, "probed_at": "2026-09-22T00:00:00Z",
                   "startup_seconds": 1.0, "evidence": {}}
            if not ok:
                rec["evidence"] = {"kind": "oom_startup", "log_excerpt":
                                   "the estimated maximum model length is 118976."}
            return rec
        P.probe_one_cell = fake_probe
        with tempfile.TemporaryDirectory() as td:
            args = argparse.Namespace(
                runtime="podman", repo="vendor/m", catalog=Path(td) / "models.yaml",
                cache=Path(td) / "cache.json", models_dir=td, vram="", ctx="",
                host_vram_gb=24, image="img", container_name="c", probe_port=18000,
                prompt="hi", force=False, force_arch=False, no_mtp=False,
                no_cache_write=False, ctx_exact=ctx_exact)
            spec = types.SimpleNamespace(name="vllm", schema_version=2)
            err = io.StringIO()
            try:
                with contextlib.redirect_stderr(err):
                    P.run_probe_pass(spec, args)
                rc = 0
            except SystemExit as e:
                rc = e.code
            finally:
                for k, v in saved.items():
                    setattr(P, k, v)
        return cache["vendor/m@sha1"]["probes"]["24"], calls, rc, err.getvalue()

    def test_hit_on_both_passes_replaces_the_cell(self) -> None:
        band, calls, rc, _ = self._run(ctx_exact=118784, base_fits=True, mtp_fits=True)
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [(118784, False), (118784, True)])
        self.assertEqual(list(band), ["118784"])
        self.assertTrue(band["118784"]["mtp_fits"])

    def test_fit_hit_but_mtp_miss_keeps_the_previous_cell_and_reports_the_estimate(self) -> None:
        band, calls, rc, err = self._run(ctx_exact=122880, base_fits=True, mtp_fits=False)
        self.assertNotEqual(rc, 0)
        self.assertEqual(calls, [(122880, False), (122880, True)])
        self.assertEqual(band, self._PRIOR, "the 115K cell that served MTP must survive")
        self.assertIn("118976", err)
        self.assertIn("miss", err)

    def test_fit_miss_keeps_the_previous_cell_and_reports_the_estimate(self) -> None:
        band, calls, rc, err = self._run(ctx_exact=122880, base_fits=False, mtp_fits=False)
        self.assertNotEqual(rc, 0)
        self.assertEqual(calls, [(122880, False)])
        self.assertEqual(band, self._PRIOR)
        self.assertIn("118976", err)

    def test_row_without_mtp_needs_only_the_fit_pass(self) -> None:
        band, calls, rc, _ = self._run(ctx_exact=122880, base_fits=True, mtp_fits=False,
                                       mtp_declared=False)
        self.assertEqual(rc, 0)
        self.assertEqual(calls, [(122880, False)])
        self.assertEqual(list(band), ["122880"])

    def test_value_above_position_limit_is_refused_without_launching(self) -> None:
        band, calls, rc, err = self._run(ctx_exact=122880, base_fits=True, mtp_fits=True,
                                         pos_limit=100000)
        self.assertNotEqual(rc, 0)
        self.assertEqual(calls, [])
        self.assertEqual(band, self._PRIOR)
