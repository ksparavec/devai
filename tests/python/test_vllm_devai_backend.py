"""`vllm-devai`: the fourth backend, wired everywhere the other three are.

2026-09-22, operator decision: the home-built, HyperQwen-patched vLLM 0.28.0
image is a backend of its own -- port 11437, container devai-vllm-devai,
probe cache deploy/.vllm-devai-reasoning-cache.json, bench/picker backend id
`vllm-devai` -- so the stock vLLM 0.22.1 and the custom build can be
addressed through the router, probed, benched and compared side by side.
The image carries a docker.io name (`docker.io/devai/vllm-devai:<tag>`) but
lives only in the local image store: nothing is pushed, and `make pull-images`
must never try to fetch it.

A backend that is registered in the router but missing from one of the
Python tools fails quietly (no probe cells -> no picker rows -> "the model
is not there"), so every registration point is pinned here, plus the two
rules the router enforces: same engine as vllm, same model store as vllm.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

MAKEFILE = (REPO_ROOT / "Makefile").read_text()
COMPOSE = (REPO_ROOT / "deploy" / "docker-compose.yaml").read_text()
ROUTER = (REPO_ROOT / "gpu-arbiter" / "main.go").read_text()
IMAGE_TAG = "v0.28.0-cu131-debian13-hq.c0c81bb"
IMAGE = f"docker.io/devai/vllm-devai:{IMAGE_TAG}"
IMAGE_LATEST = "docker.io/devai/vllm-devai:latest"


def _load(path: Path, modname: str):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


class ImageNameTest(unittest.TestCase):
    def test_one_image_name_everywhere(self) -> None:
        # compose service, router env in compose, router default, Makefile var
        self.assertIn(f"${{VLLM_DEVAI_IMAGE:-{IMAGE_LATEST}}}", COMPOSE)
        self.assertIn(f'env("VLLM_DEVAI_IMAGE", "{IMAGE_LATEST}")', ROUTER)
        self.assertRegex(MAKEFILE, rf"(?m)^VLLM_DEVAI_IMAGE \?= {re.escape(IMAGE_LATEST)}$")

    def test_build_tags_the_versioned_and_the_latest_name(self) -> None:
        recipe = MAKEFILE.split("\nbuild-vllm-image:", 1)[1].split("\n\n", 1)[0]
        self.assertIn("-t $(VLLM_DEVAI_IMAGE_TAG) -t $(VLLM_DEVAI_IMAGE)", recipe)
        self.assertRegex(MAKEFILE, rf"(?m)^VLLM_DEVAI_IMAGE_TAG \?= {re.escape(IMAGE)}$")
        self.assertNotIn("-t devai-vllm ", recipe + " ")

    def test_tag_quotes_the_real_versions(self) -> None:
        build = (REPO_ROOT / "scripts" / "build-vllm.sh").read_text()
        self.assertIn('VLLM_VERSION="${VLLM_VERSION:-0.28.0}"', build)
        self.assertIn('CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}"', build)
        self.assertIn("HYPERQWEN_COMMIT:-c0c81bb", build)
        self.assertIn("debian:trixie-slim", (REPO_ROOT / "deploy" / "Dockerfile.vllm").read_text())

    def test_pull_images_can_never_pull_it(self) -> None:
        # The pull list is `compose config --images | grep -vE ...`; the
        # filter must match this image's registry NAMESPACE (`devai/`): its
        # tag part `vllm-devai` does not contain `devai-`, and an unmatched
        # image would be pulled from docker.io, fail 3 times and abort.
        self.assertIn("grep -vE 'devai-|devai/'", MAKEFILE)
        self.assertRegex(IMAGE_LATEST, r"devai-|devai/")


class ComposeTest(unittest.TestCase):
    def test_placeholder_service_mirrors_vllm(self) -> None:
        svc = COMPOSE.split("\n  vllm-devai:", 1)[1].split("\n  sglang:", 1)[0]
        self.assertIn("container_name: devai-vllm-devai", svc)
        self.assertIn("hostname: vllm-devai", svc)
        self.assertIn("/var/cache/devai/vllm:/models:ro", svc, "same store as vllm")
        self.assertIn('entrypoint: ["sleep", "infinity"]', svc)
        self.assertIn("pull_policy: never", svc, "local-only image: compose must not try the registry")

    def test_router_gets_url_port_container_image_and_cache(self) -> None:
        for line in ("VLLM_DEVAI_URL=http://vllm-devai:11434", "VLLM_DEVAI_PORT=11437",
                     "VLLM_DEVAI_CONTAINER=devai-vllm-devai",
                     "./.vllm-devai-reasoning-cache.json:/etc/devai/.vllm-devai-reasoning-cache.json:ro"):
            self.assertIn(line, COMPOSE, line)

    def test_cache_file_exists_so_the_bind_mount_is_a_file_not_a_directory(self) -> None:
        p = REPO_ROOT / "deploy" / ".vllm-devai-reasoning-cache.json"
        self.assertTrue(p.is_file())
        json.loads(p.read_text())


class ProberTest(unittest.TestCase):
    def test_prober_script_is_the_vllm_spec_with_its_own_identity(self) -> None:
        mod = _load(REPO_ROOT / "scripts" / "probe-vllm-devai-reasoning.py", "probe_vllm_devai")
        spec = mod.SPEC
        self.assertEqual(spec.name, "vllm-devai")
        self.assertEqual(spec.engine, "vllm")
        self.assertEqual(spec.container_name, "devai-vllm-devai-probe")
        self.assertEqual(spec.cache_path.name, ".vllm-devai-reasoning-cache.json")
        self.assertEqual(spec.image, IMAGE_LATEST)
        vllm = _load(REPO_ROOT / "scripts" / "probe-vllm-reasoning.py", "probe_vllm_for_devai_test").SPEC
        self.assertEqual(spec.build_args.__name__, vllm.build_args.__name__,
                         "same launch argv builder as vllm")
        self.assertEqual(spec.allowed_kv_dtypes, vllm.allowed_kv_dtypes)

    def test_engine_gates_mtp_pass_and_recovery_scoping(self) -> None:
        import _probe_hf_common as hf
        # recovery entries scoped to the engine apply; to the name apply; sglang does not
        self.assertTrue(hf._entry_applies_to_backend({"backends": ["vllm"]}, "vllm-devai"))
        self.assertTrue(hf._entry_applies_to_backend({"backends": ["vllm-devai"]}, "vllm-devai"))
        self.assertFalse(hf._entry_applies_to_backend({"backends": ["vllm-devai"]}, "vllm"))
        self.assertFalse(hf._entry_applies_to_backend({"backends": ["sglang"]}, "vllm-devai"))
        self.assertEqual(hf.engine_of("vllm-devai"), "vllm")
        self.assertEqual(hf.engine_of("sglang"), "sglang")

    def test_make_targets(self) -> None:
        recipe = MAKEFILE.split("\nprobe-vllm-devai:", 1)[1].split("\n\n", 1)[0]
        self.assertIn("scripts/probe-vllm-devai-reasoning.py", recipe)
        self.assertIn("--models-dir $(VLLM_MODELS_DIR)", recipe)
        self.assertIn("$(if $(PROBE_CTX_EXACT),--ctx-exact $(PROBE_CTX_EXACT),)", recipe)
        bench = MAKEFILE.split("\nbench-vllm-devai:", 1)[1].split("\n\n", 1)[0]
        self.assertIn("--backend vllm-devai", bench)


class ToolingTest(unittest.TestCase):
    def test_bench_runner_knows_the_backend(self) -> None:
        from bench import bench_runner, _bench_core
        self.assertEqual(bench_runner.PROBE_CACHE_BY_BACKEND["vllm-devai"].name, ".vllm-devai-reasoning-cache.json")
        self.assertEqual(bench_runner.HF_WEIGHT_STORE_BY_BACKEND["vllm-devai"],
                         bench_runner.HF_WEIGHT_STORE_BY_BACKEND["vllm"])
        self.assertIn("vllm-devai", bench_runner.BACKEND_METRICS_URL)
        self.assertEqual(_bench_core.ROUTER_PORT_BY_BACKEND["vllm-devai"], 11437)
        self.assertIsNone(bench_runner.max_connections_for("vllm-devai"))

    def test_bench_sync_iterates_it(self) -> None:
        bs = _load(REPO_ROOT / "scripts" / "bench-sync.py", "bench_sync_for_devai_test")
        self.assertIn("vllm-devai", bs.BACKENDS)

    def test_select_models_maps_it_to_the_vllm_store(self) -> None:
        sm = _load(REPO_ROOT / "scripts" / "select-models.py", "select_models_for_devai_test")
        self.assertEqual(sm.HF_STORES["vllm-devai"], sm.VLLM_STORE)
        self.assertEqual(sm.VLLM_DEVAI_PROBE_CACHE.name, ".vllm-devai-reasoning-cache.json")

    def test_picker_lists_it_after_vllm(self) -> None:
        src = (REPO_ROOT / "scripts" / "model-picker.py").read_text()
        self.assertIn('_PICKER_BACKENDS: tuple[str, ...] = ("ollama", "vllm", "vllm-devai", "sglang")', src)
        self.assertIn('"vllm-devai": ("vLLM devai"', src)
        self.assertIn("11437", src)
        self.assertRegex(src, r'hf_priority = \{"vllm": 3, "vllm-devai": 2, "sglang": 1\}')

    def test_catalog_generator_offers_hf_rows_to_it(self) -> None:
        gc = _load(REPO_ROOT / "scripts" / "generate-catalog.py", "generate_catalog_for_devai_test")
        self.assertEqual(gc.HF_BACKENDS, ["vllm", "vllm-devai", "sglang"])

    def test_makefile_mounts_and_links_the_cache_wherever_the_other_three_are(self) -> None:
        # bench container mount, `install` placeholder + symlink, `uninstall`
        self.assertEqual(MAKEFILE.count(".vllm-devai-reasoning-cache.json"),
                         MAKEFILE.count(".sglang-reasoning-cache.json"),
                         "every Makefile site that names the sglang cache must name this one")

    def test_launcher_mounts_its_cache_and_status_server_bakes_it(self) -> None:
        self.assertIn('"vllm-devai": CONFIG_DIR / ".vllm-devai-reasoning-cache.json"',
                      (REPO_ROOT / "bin" / "devai-agent").read_text())
        self.assertIn(".vllm-devai-reasoning-cache.json",
                      (REPO_ROOT / "deploy" / "Dockerfile.mcp-modelstatus").read_text())
        go = (REPO_ROOT / "devai-tools" / "internal" / "modelcache" / "probecache.go").read_text()
        self.assertIn('"vllm-devai"', go)


if __name__ == "__main__":
    unittest.main()
