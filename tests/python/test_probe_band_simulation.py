"""`make probe` must really simulate a smaller GPU -- and refuse to lie if not.

The Ollama probe measures each model on several VRAM "bands" (16G, 24G, ...)
using ONE physical card. The probe loads every model with `num_gpu: 999`
("all layers on the GPU, or fail"), mirroring the router's warm-load, so the
question each cell answers is: does a forced-full load fit on a card this big?

History of getting that wrong, all on 2026-09-19 with Ollama 0.34.2:

  1. OLLAMA_GPU_OVERHEAD, the original knob, no longer governs placement --
     Ollama moved it to llama.cpp's automatic fit. The 16G band came out
     byte-identical to the 24G band: 21.85 GiB "fully on GPU" on a 16 GB card.
     The cache already held the same impossibility from July (qwen3.6:35b-a3b
     -mtp, 20.64 GiB at 16G); nobody had noticed.
  2. LLAMA_ARG_FIT_TARGET, llama.cpp's own knob, looked like the fix and
     worked in a hand test (13.86 GiB, spilled). It does NOTHING for the
     probe: `num_gpu: 999` reaches the runner as an explicit `-ngl 999`, and
     llama.cpp's fit only adjusts parameters that are UNSET. The model loaded
     at 17.65 GiB regardless.

No environment variable can make a forced-full load fail while the memory is
physically there. So the simulation is PHYSICAL: scripts/vram-ballast.py
holds (host - band) of VRAM through the driver API while the band is probed,
and the card genuinely has only the band's worth free. Measured: q4_k_xl at
32K (17.65 GiB) then fails with a real `cudaMalloc failed: out of memory`,
while q3_k_xl (13.69 GiB) loads fully -- exactly as on a 16 GB card. It
depends on no engine setting, so an upstream release cannot quietly break it.

And independently of HOW the band is simulated: a cell whose VRAM exceeds its
own band is impossible, and the prober must stop rather than record it. That
guard caught defect 2 above the first time it ran.

Stdlib unittest only; no container, no GPU, no network.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import _contexts  # noqa: E402


def _load(filename: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, _SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


prober = _load("probe-ollama-reasoning.py", "probe_ollama_band")

MAKEFILE = (REPO_ROOT / "Makefile").read_text()
COMPOSE = (REPO_ROOT / "deploy" / "docker-compose.yaml").read_text()
BALLAST = _SCRIPTS / "vram-ballast.py"


class BallastMathTest(unittest.TestCase):
    def test_host_sized_band_needs_no_ballast(self) -> None:
        # The 24G band on a 24G card runs under production conditions.
        self.assertEqual(_contexts.ballast_mib(24, 24), 0)

    def test_smaller_band_holds_exactly_the_difference(self) -> None:
        self.assertEqual(_contexts.ballast_mib(24, 16), 8 * 1024)
        self.assertEqual(_contexts.ballast_mib(48, 24), 24 * 1024)

    def test_cannot_simulate_a_larger_card(self) -> None:
        with self.assertRaises(ValueError):
            _contexts.ballast_mib(16, 24)


class BandGuardTest(unittest.TestCase):
    """A cell using more VRAM than its band is not a measurement, it is proof
    the simulation is off."""

    def test_the_cell_that_was_actually_recorded_is_refused(self) -> None:
        msg = prober.band_violation(
            {"fully_on_gpu": True, "actual_vram_gb": 21.85}, 16)
        self.assertIsNotNone(msg)
        self.assertIn("21.85", msg)
        self.assertIn("16", msg)

    def test_spilled_cell_over_the_band_is_refused_too(self) -> None:
        self.assertIsNotNone(prober.band_violation(
            {"fully_on_gpu": False, "actual_vram_gb": 18.29}, 16))

    def test_honest_cells_pass(self) -> None:
        for rec, band in (({"fully_on_gpu": True, "actual_vram_gb": 13.69}, 16),
                          ({"fully_on_gpu": True, "actual_vram_gb": 15.75}, 16),
                          ({"fully_on_gpu": True, "actual_vram_gb": 21.85}, 24)):
            self.assertIsNone(prober.band_violation(rec, band), (rec, band))

    def test_cell_without_a_measurement_is_not_a_violation(self) -> None:
        # OOM cells carry no VRAM figure -- that is the honest outcome.
        self.assertIsNone(prober.band_violation({"capability": "error"}, 16))
        self.assertIsNone(prober.band_violation({"actual_vram_gb": None}, 16))


class BallastScriptTest(unittest.TestCase):
    """The GPU cannot be exercised here; what can be is that every way of
    getting it wrong is an ERROR, never a quiet no-op -- a ballast that
    silently holds nothing is the original bug all over again."""

    def _run(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(BALLAST), *args],
                              env=dict(os.environ, **env),
                              capture_output=True, text=True, timeout=30)

    def test_missing_driver_library_is_a_clear_error(self) -> None:
        r = self._run("1024", VRAM_BALLAST_LIBCUDA="/nonexistent/libcuda.so.1")
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("ready", r.stdout)
        self.assertIn("libcuda", r.stderr)

    def test_size_must_be_a_positive_integer(self) -> None:
        for bad in ("0", "-5", "lots"):
            r = self._run(bad, VRAM_BALLAST_LIBCUDA="/nonexistent/libcuda.so.1")
            self.assertNotEqual(r.returncode, 0, bad)
            self.assertNotIn("ready", r.stdout)


class WiringTest(unittest.TestCase):
    def _probe_recipe(self) -> str:
        return MAKEFILE.split("\nprobe:", 1)[1].split("\nprobe-vllm:", 1)[0]

    def _executed(self, text: str) -> str:
        return "\n".join(ln for ln in text.splitlines()
                         if not ln.lstrip().startswith(("@#", "#")))

    def test_probe_holds_a_ballast_sized_by_the_shared_helper(self) -> None:
        recipe = self._executed(self._probe_recipe())
        self.assertIn("ballast_mib", recipe)
        self.assertIn("scripts/vram-ballast.py", recipe)

    def test_probe_checks_the_card_really_shrank(self) -> None:
        # Physical verification, not trust: free VRAM must be within the band
        # before a single cell is probed.
        recipe = self._executed(self._probe_recipe())
        self.assertIn("memory.free", recipe)

    def test_ballast_is_released_on_every_exit_path(self) -> None:
        # A ballast left behind would silently take 8 GiB off the production
        # GPU. The EXIT trap is what runs on success, failure and Ctrl-C.
        restore = self._probe_recipe().split("restore_ollama()", 1)[1].split("};", 1)[0]
        self.assertIn("release_ballast", restore)

    def test_no_engine_knob_is_relied_on_any_more(self) -> None:
        recipe = self._executed(self._probe_recipe())
        self.assertNotIn("LLAMA_ARG_FIT_TARGET", recipe)
        self.assertNotIn("vram_overhead_bytes", recipe)
        self.assertFalse((REPO_ROOT / "deploy" / "docker-compose.probe-band.yaml").exists(),
                         "the fit-target overlay is inert for forced-full loads")

    def test_prober_is_told_every_condition_it_stamps(self) -> None:
        # The prober records kv_cache_type and flash_attention on each cell
        # FROM ITS OWN ENVIRONMENT, while the daemon gets them through
        # compose. The recipe forwarded only the KV dtype, so `flash_attention`
        # was stamped False on every cell ever written -- including q8_0
        # tiers, which Ollama only honours WITH flash attention on. A cell
        # must not misreport the conditions it was measured under.
        recipe = self._executed(self._probe_recipe())
        prober_run = recipe.split("probe-ollama-reasoning.py", 1)[0].rsplit("run --rm", 1)[1]
        for var in ("OLLAMA_KV_CACHE_TYPE", "OLLAMA_FLASH_ATTENTION"):
            self.assertIn(f"-e {var}=", prober_run, f"{var} not forwarded to the prober")

    def test_default_probe_band_is_the_host_card_and_nothing_else(self) -> None:
        # The operator probes the hardware this machine HAS. Simulating a
        # smaller card stays available on request (PROBE_VRAMS=16G,24G) but is
        # not something `make probe` does uninvited: it costs a ballast and
        # GPU time, and writes cells nobody on this host consumes. Tied to
        # GPU_MEMORY_GB rather than a literal so it follows the hardware.
        self.assertRegex(MAKEFILE, r"(?m)^PROBE_VRAMS\s*\?=\s*\$\(GPU_MEMORY_GB\)G\s*$")
        self.assertNotRegex(MAKEFILE, r"(?m)^PROBE_VRAMS\s*\?=.*16G")

    def test_serving_compose_never_sets_a_fit_target(self) -> None:
        self.assertNotIn("LLAMA_ARG_FIT_TARGET", self._executed(COMPOSE))


if __name__ == "__main__":
    unittest.main()
