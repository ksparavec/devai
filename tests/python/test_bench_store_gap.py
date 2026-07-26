"""The bench runner must not bench a model whose weights are absent.

Observed 2026-07-25. `make bench-sglang` began working through the SGLang
probe cache and started on DeepSeek-R1-Distill-Llama-8B, which has no
weights in the SGLang store. The router had nothing to launch, so the
leak task recorded tps=0.0 / ttft=None, and the run would have burned a
full 7-task sweep on each of SIX absent models before reaching a real
one.

The cause is that a fits=true probe cell is not evidence the weights are
present. vLLM and SGLang use SEPARATE volumes, and `make model-pull`
writes only the vLLM one, so an SGLang-probed model routinely has fit
data and no weights at all. select-models.py has carried exactly this
check as `sglang_weight_gaps` since it was written; the bench runner
never used it.

Two halves are tested here: the predicate, and that discover_models
actually applies it. The Makefile half -- mounting the two stores
read-only into the bench container, without which the runner cannot see
them at all -- is asserted in test_bench_store_mounts below.

Stdlib unittest only.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "bench_runner", REPO_ROOT / "scripts" / "bench" / "bench_runner.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_runner"] = mod
    spec.loader.exec_module(mod)
    return mod


br = _load()


class WeightsPresentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self._orig = dict(br.HF_WEIGHT_STORE_BY_BACKEND)
        br.HF_WEIGHT_STORE_BY_BACKEND["sglang"] = self.store
        self.addCleanup(
            br.HF_WEIGHT_STORE_BY_BACKEND.update, self._orig)

    def test_present_when_config_json_exists(self):
        (self.store / "Good-9B").mkdir()
        (self.store / "Good-9B" / "config.json").write_text("{}")
        self.assertTrue(br.weights_present("sglang", "Good-9B"))

    def test_absent_when_directory_missing(self):
        self.assertFalse(br.weights_present("sglang", "Missing-9B"))

    def test_absent_when_directory_exists_but_has_no_config(self):
        """A half-finished download leaves the directory behind. It is not
        loadable, so it must not be benched."""
        (self.store / "Partial-9B").mkdir()
        self.assertFalse(br.weights_present("sglang", "Partial-9B"))

    def test_ollama_fails_open(self):
        """Ollama keeps weights in a digest-keyed blob store, not a
        directory named after the model -- there is nothing to stat."""
        self.assertTrue(br.weights_present("ollama", "qwen3.6:35b-a3b-mtp"))

    def test_missing_store_fails_open(self):
        """The Makefile mount is wildcard-guarded. If the volume is not
        mounted the runner must bench everything rather than silently
        skip every model."""
        br.HF_WEIGHT_STORE_BY_BACKEND["sglang"] = Path("/nonexistent/store")
        self.assertTrue(br.weights_present("sglang", "Anything"))


class DiscoverModelsAppliesTheCheckTest(unittest.TestCase):
    """The predicate is useless if discover_models does not consult it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self._orig = dict(br.HF_WEIGHT_STORE_BY_BACKEND)
        br.HF_WEIGHT_STORE_BY_BACKEND["sglang"] = self.store
        self.addCleanup(br.HF_WEIGHT_STORE_BY_BACKEND.update, self._orig)

        (self.store / "OnDisk-9B").mkdir()
        (self.store / "OnDisk-9B" / "config.json").write_text("{}")

        self.cache = {
            "org/OnDisk-9B@aaa": {
                "aliases": ["OnDisk-9B"], "repo": "org/OnDisk-9B", "sha": "aaa",
                "capability": "structured",
                "probes": {"24": {"32768": {"fits": True, "ctx": 32768}}},
            },
            "org/Absent-9B@bbb": {
                "aliases": ["Absent-9B"], "repo": "org/Absent-9B", "sha": "bbb",
                "capability": "structured",
                "probes": {"24": {"32768": {"fits": True, "ctx": 32768}}},
            },
        }

    def test_absent_model_is_not_a_target(self):
        orig = br.load_cache
        br.load_cache = lambda _p: self.cache
        self.addCleanup(setattr, br, "load_cache", orig)

        targets = br.discover_models("sglang", host_vram_gb=24,
                                     repo_filter=None)
        aliases = {t["alias"] for t in targets}
        self.assertIn("OnDisk-9B", aliases)
        self.assertNotIn("Absent-9B", aliases,
                         "a model with no weights must not be benched")


class BenchStoreMountsTest(unittest.TestCase):
    """The runner cannot check a store the container cannot see."""

    def test_makefile_mounts_both_hf_stores_read_only(self):
        mk = (REPO_ROOT / "Makefile").read_text()
        block = mk.split("BENCH_CACHE_MOUNTS =", 1)[1].split("\n\n", 1)[0]
        for var in ("VLLM_MODELS_DIR", "SGLANG_MODELS_DIR"):
            self.assertIn(var, block,
                          f"{var} not mounted into the bench container")
        self.assertIn(":ro", block, "weight stores must be read-only")


class SamplingOverrideTest(unittest.TestCase):
    """Greedy decoding is right for eval comparability, but a few models
    cannot be benched greedily at all.

    NVIDIA-Nemotron-Nano-9B-v2 loops on its own <think> trace at
    temperature 0 and never emits an answer (documented against the
    family in scripts/model-families.yaml). Every scored task then reads
    ~0, which looks like a capability failure and is really a sampling
    artifact -- the exact kind of silently-wrong number this project has
    been cleaning up. deploy/bench-sampling.json carries the exceptions.
    """

    def test_default_is_greedy(self):
        t, p = br.sampling_for("some-ordinary-model@131072")
        self.assertEqual(t, br.BENCH_TEMPERATURE)
        self.assertEqual(p, br.BENCH_TOP_P)

    def test_nemotron_is_exempted(self):
        t, _ = br.sampling_for("NVIDIA-Nemotron-Nano-9B-v2-NVFP4@131072")
        self.assertGreaterEqual(
            t, 0.6, "Nemotron must not be benched greedily -- it loops")

    def test_override_matches_on_substring_so_the_ctx_suffix_is_ignored(self):
        bare, _ = br.sampling_for("NVIDIA-Nemotron-Nano-9B-v2-NVFP4")
        suffixed, _ = br.sampling_for("NVIDIA-Nemotron-Nano-9B-v2-NVFP4@131072")
        self.assertEqual(bare, suffixed)

    def test_shipped_override_file_parses_and_is_documented(self):
        ov = br.load_sampling_overrides()
        self.assertIn("NVIDIA-Nemotron-Nano-9B-v2-NVFP4", ov)
        for name, cfg in ov.items():
            with self.subTest(model=name):
                self.assertTrue(
                    cfg.get("reason"),
                    f"{name}: an override without a reason will be 'fixed' "
                    f"back to greedy by the next reader")

    def test_missing_file_falls_back_to_defaults(self):
        self.assertEqual(br.load_sampling_overrides(Path("/nonexistent.json")), {})


class CacheServicesInSyncTest(unittest.TestCase):
    """`make cache-up` names its services explicitly, so that list must
    not drift from the compose file.

    It names them one at a time because the router replaces each backend
    placeholder with a container created through the libpod API, carrying
    none of compose's project labels. Compose does not recognise those as
    its own, tries to create a fresh container, and dies on the name
    collision -- which took down the entire cache-up target and with it
    every test and bench target that depends on it.

    If a service is added to compose and not to CACHE_SERVICES, cache-up
    silently stops starting it.
    """

    def _makefile_var(self, name: str) -> set[str]:
        mk = (REPO_ROOT / "Makefile").read_text()
        line = mk.split(f"{name} =", 1)[1]
        # Consume backslash continuations.
        out, buf = [], ""
        for raw in line.splitlines():
            buf = raw.rstrip()
            cont = buf.endswith("\\")
            out.append(buf.rstrip("\\").strip())
            if not cont:
                break
        return set(" ".join(out).split())

    def test_cache_services_matches_compose(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
        compose = yaml.safe_load(
            (REPO_ROOT / "deploy" / "docker-compose.yaml").read_text())
        # Profile-gated services are opt-in and not part of cache-up.
        declared = {
            name for name, svc in (compose.get("services") or {}).items()
            if not (svc or {}).get("profiles")
        }
        listed = self._makefile_var("CACHE_SERVICES")
        self.assertEqual(
            listed, declared,
            f"CACHE_SERVICES drifted from docker-compose.yaml: "
            f"only-in-Makefile={listed - declared}, "
            f"only-in-compose={declared - listed}")

    def test_backend_services_are_a_subset(self):
        backends = self._makefile_var("CACHE_BACKEND_SERVICES")
        self.assertTrue(backends <= self._makefile_var("CACHE_SERVICES"))
        self.assertEqual(backends, {"ollama", "vllm", "sglang"})


if __name__ == "__main__":
    unittest.main()
