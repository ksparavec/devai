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
import subprocess
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
        self.assertNotIn("-t devai-vllm ", recipe + " ")

    def _rendered_tags(self, *args: str) -> str:
        # `make -n` prints the build command without running it; the
        # recipe's tests and podman are not executed.
        r = subprocess.run(["make", "-n", "build-vllm-image", *args], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_the_versioned_tag_follows_vllm_version(self) -> None:
        # v<version>-cu131-debian13-hq.<HyperQwen commit, 7 hex> from the
        # record (4v); 0.28.0's is the tag the kept rollback image carries.
        self.assertIn(f"-t {IMAGE} -t {IMAGE_LATEST}", self._rendered_tags("VLLM_VERSION=0.28.0"))
        self.assertIn("-t docker.io/devai/vllm-devai:v0.30.0-cu131-debian13-hq.7af097b",
                      self._rendered_tags())
        # 4v builds with VLLM_DEVAI_IMAGE set to the new tag, so :latest
        # stays on 0.28.0 until the acceptance rule passes.
        new = "docker.io/devai/vllm-devai:v0.30.0-cu131-debian13-hq.7af097b"
        out = self._rendered_tags("VLLM_VERSION=0.30.0", f"VLLM_DEVAI_IMAGE={new}")
        self.assertIn(f"-t {new} -t {new} ", out)
        self.assertNotIn(":latest", out)
        self.assertIn("vllm-build/0.30.0/dist:/var/cache/vllm-dist:ro", out)

    def test_a_record_without_hyperqwen_gets_a_stock_tag(self) -> None:
        self.assertRegex(MAKEFILE, r"(?m)^VLLM_DEVAI_IMAGE_TAG \?= docker\.io/devai/vllm-devai:"
                                   r"v\$\(VLLM_VERSION\)-cu131-debian13-\$\(if \$\(VLLM_HYPERQWEN_SHORT\),"
                                   r"hq\.\$\(VLLM_HYPERQWEN_SHORT\),stock\)$")

    def test_image_selection_reaches_the_probers_and_compose(self) -> None:
        # Without the export a .env or command-line value never reached the
        # probers or compose, which fell back to :latest -- both arms of 4v's
        # comparison could silently run one build.
        self.assertRegex(MAKEFILE, r"(?m)^export VLLM_DEVAI_IMAGE$")
        r = subprocess.run(["make", "-s", "--no-print-directory", "-f", "Makefile", "-f", "-",
                            "_devai_env", "VLLM_DEVAI_IMAGE=docker.io/devai/vllm-devai:vX"],
                           input="_devai_env:\n\t@printenv VLLM_DEVAI_IMAGE\n", cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "docker.io/devai/vllm-devai:vX")

    def test_tag_quotes_the_real_versions(self) -> None:
        build = (REPO_ROOT / "scripts" / "build-vllm.sh").read_text()
        self.assertIn('VLLM_VERSION="${VLLM_VERSION:-0.30.0}"', build)
        self.assertIn('CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}"', build)
        records = REPO_ROOT / "deploy" / "vllm-builds"
        self.assertIn("HYPERQWEN_COMMIT=c0c81bb", (records / "0.28.0.sh").read_text())
        self.assertIn("HYPERQWEN_COMMIT=7af097b", (records / "0.30.0.sh").read_text())
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
        # vLLM's builder, then the 0.30.0 backend's engine args (plan 4v,
        # prefix-cache retention adopted 2026-10-07); recovery flags follow
        # in _probe_hf_common, as in the router's vllmEntrypoint.
        base = vllm.build_args("m", 32768, 0.9, reasoning_parser="qwen3",
                               speculative_config='{"method": "mtp"}')
        self.assertEqual(spec.build_args("m", 32768, 0.9, reasoning_parser="qwen3",
                                         speculative_config='{"method": "mtp"}'),
                         base + ["--prefix-cache-retention-interval", "None"])
        self.assertNotIn("--prefix-cache-retention-interval", base,
                         "stock vLLM 0.22.1 has no such flag")
        self.assertEqual(spec.allowed_kv_dtypes, vllm.allowed_kv_dtypes)

    def test_retention_args_equal_the_routers(self) -> None:
        mod = _load(REPO_ROOT / "scripts" / "probe-vllm-devai-reasoning.py", "probe_vllm_devai_parity")
        go = (REPO_ROOT / "gpu-arbiter" / "engine_cache.go").read_text()
        m = re.search(r'prefixCacheRetentionArgs = \[\]string\{([^}]*)\}', go)
        self.assertIsNotNone(m, "gpu-arbiter/engine_cache.go lost prefixCacheRetentionArgs")
        self.assertEqual(mod.PREFIX_CACHE_RETENTION_ARGS, re.findall(r'"([^"]*)"', m.group(1)))

    def test_engine_gates_mtp_pass_and_recovery_scoping(self) -> None:
        import _probe_hf_common as hf
        # recovery entries scoped to the engine apply; to the name apply; sglang does not
        self.assertTrue(hf._entry_applies_to_backend({"backends": ["vllm"]}, "vllm-devai"))
        self.assertTrue(hf._entry_applies_to_backend({"backends": ["vllm-devai"]}, "vllm-devai"))
        self.assertFalse(hf._entry_applies_to_backend({"backends": ["vllm-devai"]}, "vllm"))
        self.assertFalse(hf._entry_applies_to_backend({"backends": ["sglang"]}, "vllm-devai"))
        self.assertEqual(hf.engine_of("vllm-devai"), "vllm")
        self.assertEqual(hf.engine_of("sglang"), "sglang")

    def test_vllm_devai_launches_fetch_no_flashinfer_cubins_at_run_time(self) -> None:
        # Plan 4v: every vllm-devai launch runs with run-time cubin downloads
        # off, as 4d's launcher will serve; vllm and sglang launches do not
        # carry it. Pinned to the router's buildContainerSpec on the Go side.
        import _probe_hf_common as hf
        want = "FLASHINFER_CUBINS_REPOSITORY=file:///nonexistent/"

        def argv(backend, env=None):
            return hf.container_run_args(
                "podman", "c", "img", 18000, "/m", env or {"A": "1"},
                "python3", ["-m", "x"], backend=backend)

        def envs(a):
            return [a[i + 1] for i, t in enumerate(a) if t == "--env"]
        self.assertIn(want, envs(argv("vllm-devai")))
        self.assertIn("A=1", envs(argv("vllm-devai")))
        for other in ("vllm", "sglang", None):
            self.assertFalse(any(e.startswith("FLASHINFER_CUBINS_REPOSITORY=")
                                 for e in envs(argv(other))), other)
        # A per-model environment cannot turn the downloads back on.
        a = argv("vllm-devai", {"FLASHINFER_CUBINS_REPOSITORY": "https://example"})
        self.assertEqual([e for e in envs(a) if e.startswith("FLASHINFER_")], [want])
        # Both launch sites pass the backend through.
        import inspect
        import _probe_load as lp
        self.assertIn("backend=spec.name", inspect.getsource(hf.probe_one_cell))
        self.assertIn("backend=spec.name", inspect.getsource(lp.load_probe_one_cell))

    def test_make_targets(self) -> None:
        recipe = MAKEFILE.split("\nprobe-vllm-devai:", 1)[1].split("\n\n", 1)[0]
        self.assertIn("scripts/probe-vllm-devai-reasoning.py", recipe)
        self.assertIn("--models-dir $(VLLM_MODELS_DIR)", recipe)
        self.assertIn("$(if $(PROBE_CTX_EXACT),--ctx-exact $(PROBE_CTX_EXACT),)", recipe)
        self.assertIn("$(if $(PROBE_LEDGER),--ledger '$(PROBE_LEDGER)',)", recipe)
        bench = MAKEFILE.split("\nbench-vllm-devai:", 1)[1].split("\n\n", 1)[0]
        self.assertIn("--backend vllm-devai", bench)


class Phase4vMakefileWiringTest(unittest.TestCase):
    """The knobs 4v's runs use, from the Makefile to the scripts' flags."""

    @staticmethod
    def _recipe(target: str) -> str:
        return MAKEFILE.split(f"\n{target}:", 1)[1].split("\n\n", 1)[0]

    def test_load_probes_take_the_drafter_a_scratch_cache_and_a_scratch_ledger(self) -> None:
        self.assertIn("$(if $(PROBE_MTP),--mtp,)", self._recipe("probe-load-vllm"))
        devai = self._recipe("probe-load-vllm-devai")
        self.assertRegex(devai.split("\n", 1)[0], r"^ ## \S", "documented target")
        self.assertIn("python3 scripts/probe-vllm-devai-reasoning.py --load", devai)
        for flag in ("$(if $(PROBE_MTP),--mtp,)", "$(if $(PROBE_CACHE),--cache '$(PROBE_CACHE)',)",
                     "$(if $(PROBE_LEDGER),--ledger '$(PROBE_LEDGER)',)", "$(if $(PROBE_FORCE),--force,)",
                     "$(if $(PROBE_REPO),--repo '$(PROBE_REPO)',)"):
            self.assertIn(flag, devai)
        self.assertRegex(MAKEFILE, r"(?m)^\.PHONY:.*\bprobe-load-vllm-devai\b")

    def test_bench_runs_take_a_scratch_cache_logs_mtp_and_connections(self) -> None:
        flags = MAKEFILE.split("\nBENCH_RUN_FLAGS = \\\n", 1)[1].split("\n\n", 1)[0]
        for flag in ("$(if $(BENCH_CACHE),--cache '$(BENCH_CACHE)',)",
                     "$(if $(BENCH_LOG_DIR),--log-dir '$(BENCH_LOG_DIR)',)",
                     "$(if $(BENCH_MTP),--mtp,)",
                     "$(if $(BENCH_MAX_CONNECTIONS),--max-connections $(BENCH_MAX_CONNECTIONS),)"):
            self.assertIn(flag, flags)
        self.assertIn("$(BENCH_RUN_FLAGS)", self._recipe("bench-vllm-devai"))

    def test_scratch_bench_paths_must_lie_under_the_bench_mount(self) -> None:
        # The bench container's only read-write mount besides deploy/ is
        # $(CACHE_DIR)/bench; a scratch path elsewhere would vanish with the
        # container. Checked at parse time, so `make help` shows it.
        for var in ("BENCH_CACHE", "BENCH_LOG_DIR"):
            r = subprocess.run(["make", "-n", "help", f"{var}=/tmp/x/cache.json"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=60)
            self.assertNotEqual(r.returncode, 0, var)
            self.assertIn(f"{var}=/tmp/x/cache.json must lie under /var/cache/devai/bench/", r.stderr)
            ok = subprocess.run(["make", "-n", "help", f"{var}=/var/cache/devai/bench/4v/t/x"],
                                cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(ok.returncode, 0, ok.stderr)

    def test_bench_concurrency_passes_salt_min_tokens_and_metrics_url(self) -> None:
        recipe = self._recipe("bench-concurrency")
        for flag in ("$(if $(SALT),--salt '$(SALT)',)", "$(if $(MIN_TOKENS),--min-tokens $(MIN_TOKENS),)",
                     "$(if $(METRICS_URL),--metrics-url '$(METRICS_URL)',)"):
            self.assertIn(flag, recipe)
        # The report is captured as `make bench-concurrency ... > session.json`:
        # the container command must not be echoed into it.
        self.assertIn("\t@$(CONTAINER_RUNTIME) run --rm", recipe)


class ToolingTest(unittest.TestCase):
    def test_bench_runner_knows_the_backend(self) -> None:
        from bench import bench_runner, _bench_core
        self.assertEqual(bench_runner.PROBE_CACHE_BY_BACKEND["vllm-devai"].name, ".vllm-devai-reasoning-cache.json")
        self.assertEqual(bench_runner.HF_WEIGHT_STORE_BY_BACKEND["vllm-devai"],
                         bench_runner.HF_WEIGHT_STORE_BY_BACKEND["vllm"])
        self.assertIn("vllm-devai", bench_runner.BACKEND_METRICS_URL)
        self.assertEqual(_bench_core.ROUTER_PORT_BY_BACKEND["vllm-devai"], 11437)
        # A batching engine: the same static cap as vllm (not Ollama's 1).
        self.assertEqual(bench_runner.max_connections_for("vllm-devai"),
                         bench_runner.max_connections_for("vllm"))

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

    def test_launcher_mounts_its_cache(self) -> None:
        # The MCP status server that baked this cache too was removed on
        # 2026-10-07 (platform-rebuild Phase 1, M10).
        self.assertIn('"vllm-devai": CONFIG_DIR / ".vllm-devai-reasoning-cache.json"',
                      (REPO_ROOT / "bin" / "devai-agent").read_text())


if __name__ == "__main__":
    unittest.main()


class DerivedRowsAcrossToolsTest(unittest.TestCase):
    """A derived catalog row is probed like an HF row, never downloaded."""

    _ROW = {"name": "X-devai-NVFP4", "source": "derived", "derived_from": "X-NVFP4",
            "repo": "devai/X-devai-NVFP4", "sha": "abc", "backend": ["vllm-devai"]}

    def test_prober_loads_derived_rows_for_the_backend_that_serves_them(self) -> None:
        import tempfile, yaml
        import _probe_hf_common as hf
        with tempfile.TemporaryDirectory() as td:
            cat = Path(td) / "models.yaml"
            cat.write_text(yaml.safe_dump({"models": [self._ROW]}))
            self.assertEqual([r["name"] for r in hf.load_catalog_hf_rows(cat, "vllm-devai")], ["X-devai-NVFP4"])
            self.assertEqual(hf.load_catalog_hf_rows(cat, "vllm"), [])

    def test_select_models_refuses_to_download_a_derived_row(self) -> None:
        sm = _load(REPO_ROOT / "scripts" / "select-models.py", "select_models_for_derived_test")
        with self.assertRaises(SystemExit) as cm:
            sm.pull(dict(self._ROW))
        self.assertIn("make model-prepare NAME=X-NVFP4", str(cm.exception))

    def test_model_sync_never_queues_a_derived_row(self) -> None:
        ms = _load(REPO_ROOT / "scripts" / "model-sync.py", "model_sync_for_derived_test")
        plan = ms.plan_sync([dict(self._ROW)], {}, {}, {}, {}, host_vram=24)
        self.assertEqual(plan["new"], [])
        self.assertEqual([r["name"] for r in plan["evaluated"]], ["X-devai-NVFP4"])


class ParserHintsByEngineTest(unittest.TestCase):
    """Curated `parsers:` blocks and card-hint tables are keyed by ENGINE.

    Found 2026-09-22 by the first bench of the two derived rows: the prober
    looked the catalog's `parsers` block up by backend NAME, `vllm-devai` has
    no block, so both rows were probed with no parsers and the cache recorded
    reasoning_parser / tool_parser None. The router then launched them without
    `--reasoning-parser qwen3` / `--tool-call-parser qwen3_xml`, reasoning
    bled into content (leak 45 % and 55 %) and the bench early-dropped both --
    a verdict about the launch flags, not the checkpoints. The picker's
    TOOLS fallback keyed the same way.
    """

    _PARSERS = {"vllm": {"reasoning": "qwen3", "tool": "qwen3_xml"},
                "sglang": {"reasoning": "qwen3", "tool": "qwen"}}

    def test_prober_reads_the_engine_block_for_vllm_devai(self) -> None:
        import _probe_hf_common as hf
        row = {"name": "X", "parsers": dict(self._PARSERS)}
        self.assertEqual(hf.curated_parsers(row, "vllm-devai"), ("qwen3", "qwen3_xml"))
        self.assertEqual(hf.curated_parsers(row, "vllm"), ("qwen3", "qwen3_xml"))
        self.assertEqual(hf.curated_parsers(row, "sglang"), ("qwen3", "qwen"))
        self.assertEqual(hf.curated_parsers({"name": "X"}, "vllm-devai"), (None, None))

    def test_prober_lets_a_name_specific_block_override_per_key(self) -> None:
        import _probe_hf_common as hf
        row = {"parsers": {**self._PARSERS, "vllm-devai": {"tool": "hermes"}}}
        self.assertEqual(hf.curated_parsers(row, "vllm-devai"), ("qwen3", "hermes"))

    def test_prober_derives_card_hints_for_the_engine(self) -> None:
        from unittest import mock
        import _probe_hf_common as hf
        with mock.patch.object(hf._card_hints, "derive_parser", return_value="qwen3") as dp:
            self.assertEqual(hf.derive_parser_for("X", Path("/m"), "vllm-devai", "reasoning"), "qwen3")
        dp.assert_called_once_with("X", Path("/m"), "vllm", "reasoning")

    def test_run_probe_pass_uses_the_engine_keyed_helpers(self) -> None:
        import inspect
        import _probe_hf_common as hf
        src = inspect.getsource(hf.run_probe_pass)
        self.assertIn("curated_parsers(row, spec.name)", src)
        # The parser order (Phase 17a) lives in resolve_parsers, which
        # run_probe_pass calls with the backend name.
        self.assertIn("resolve_parsers(", src)
        self.assertIn("name, models_dir, spec.name, curated_reasoning, curated_tool", src)
        self.assertIn("derive_parser_for(", inspect.getsource(hf.resolve_parsers))
        self.assertNotIn('("parsers") or {}).get(spec.name)', src)
        self.assertNotIn("derive_parser(\n                    name, models_dir, spec.name", src)

    def test_picker_tools_fallback_reads_the_engine_block(self) -> None:
        mp = _load(REPO_ROOT / "scripts" / "model-picker.py", "model_picker_for_parser_test")
        self.assertEqual(mp._resolve_tool_parser({}, self._PARSERS, "vllm-devai"), "qwen3_xml")
        self.assertEqual(mp._resolve_tool_parser({}, self._PARSERS, "sglang"), "qwen")
        self.assertEqual(mp._resolve_tool_parser({"tool_parser": "probed"}, self._PARSERS, "vllm-devai"), "probed")
        self.assertEqual(mp._resolve_tool_parser({}, {**self._PARSERS, "vllm-devai": {"tool": "hermes"}}, "vllm-devai"), "hermes")
        self.assertEqual(mp._resolve_tool_parser({}, {}, "vllm-devai"), "N/A")

    def test_load_probe_uses_the_engine_keyed_helper(self) -> None:
        import inspect
        import _probe_load as lp
        src = inspect.getsource(lp.run_load_probe_pass)
        self.assertIn("curated_parsers(row, spec.name)", src)
        self.assertNotIn('("parsers") or {}).get(spec.name)', src)


class CacheDownRemovesTheRecreatedContainerTest(unittest.TestCase):
    """`make cache-down` must force-remove devai-vllm-devai by name.

    The router recreates every HF backend container via libpod when a
    request arrives, and the recreated container carries no compose labels,
    so `compose down --remove-orphans` leaves it behind. cache-down therefore
    removes the recreated containers by NAME -- and on 2026-09-22 the list
    still read vllm/sglang/ollama: after the first vllm-devai bench the
    router-built `devai-vllm-devai` survived `cache-down` holding 21.8 GiB,
    every probe launch failed `kind=infra`, and the next `cache-up` died on
    the name collision while reporting rc=0.
    """

    _MAKEFILE = (REPO_ROOT / "Makefile").read_text()

    def _recipe(self, target: str) -> str:
        m = re.search(rf"^{re.escape(target)}:.*?(?=^\S)", self._MAKEFILE, re.M | re.S)
        self.assertIsNotNone(m, f"no {target} recipe")
        return m.group(0)

    def test_cache_down_names_every_router_recreated_container(self) -> None:
        loop = re.search(r"for name in ([^;]+); do", self._recipe("cache-down"))
        self.assertIsNotNone(loop)
        names = loop.group(1).split()
        for n in ("devai-vllm", "devai-sglang", "devai-ollama", "devai-vllm-devai"):
            self.assertIn(n, names)

    def test_cache_up_leaves_the_router_built_container_alone(self) -> None:
        # cache-up skips CACHE_BACKEND_SERVICES that already exist. With
        # vllm-devai missing from that list, compose tried to create
        # devai-vllm-devai itself and the whole target aborted on the name
        # collision -- before recreating the router it was run for
        # (2026-09-22).
        m = re.search(r"^CACHE_BACKEND_SERVICES = ([^\n]+)", self._MAKEFILE, re.M)
        self.assertIsNotNone(m)
        self.assertIn("vllm-devai", m.group(1).split())

    def test_test_agents_cleanup_names_vllm_devai_too(self) -> None:
        rm = re.search(r"rm -f ([^\n]+?) 2>/dev/null", self._recipe("test-agents"))
        self.assertIsNotNone(rm)
        self.assertIn("devai-vllm-devai", rm.group(1).split())

    def test_install_stages_the_vllm_devai_cache_symlink(self) -> None:
        loop = re.search(r"for cache in ([^;]+); do", self._recipe("install"))
        self.assertIsNotNone(loop)
        self.assertIn("vllm-devai", loop.group(1).split())


class EngineKeyedProbeShapesTest(unittest.TestCase):
    """The probe's request shapes and KV default are per ENGINE as well.

    Third member of the same family found on 2026-09-22: the disable / enable
    thinking bodies were built behind `if backend == "vllm" ... elif "sglang"`,
    so for `vllm-devai` the disable probe sent the base body UNCHANGED, the
    model kept thinking, and both derived rows were recorded
    disable_verified=false -- which made `::nothink` a no-op on port 11437
    although the engine honours the router's disable shape (measured:
    reasoning_tokens 0). The load probe's legacy KV default had the same
    `== "vllm"` test.
    """

    _BASE = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    def test_disable_and_enable_bodies_match_vllm_for_vllm_devai(self) -> None:
        import _probe_hf_common as hf
        for build in (hf.build_disable_thinking_body, hf.build_enable_thinking_body):
            devai = build("vllm-devai", dict(self._BASE))
            self.assertEqual(devai, build("vllm", dict(self._BASE)), build.__name__)
            self.assertNotEqual(devai, self._BASE, f"{build.__name__} left the body unchanged")
        # The disable shape is now the same top-level one on both engines
        # (vLLM never read the old `extra_body` spelling); the keying is
        # still by engine, which the enable body still shows (SGLang adds
        # separate_reasoning) and an unknown engine, which gets nothing.
        self.assertNotEqual(hf.build_enable_thinking_body("sglang", dict(self._BASE)),
                            hf.build_enable_thinking_body("vllm", dict(self._BASE)))
        self.assertEqual(hf.build_disable_thinking_body("ollama", dict(self._BASE)), self._BASE)

    def test_load_probe_legacy_kv_default_is_fp8_for_vllm_devai(self) -> None:
        import _probe_load as lp
        self.assertEqual(lp._cell_kv_cache_dtype({}, "vllm-devai"), "fp8")
        self.assertEqual(lp._cell_kv_cache_dtype({}, "vllm"), "fp8")
        self.assertEqual(lp._cell_kv_cache_dtype({}, "sglang"), "")
        self.assertEqual(lp._cell_kv_cache_dtype({"kv_cache_type": "auto"}, "vllm-devai"), "auto")

    def test_no_backend_name_comparison_survives_in_the_probers(self) -> None:
        # Every engine-dependent branch must go through engine_of().
        pat = re.compile(r'\b(backend|spec\.name)\s*(==|!=)\s*["\'](vllm|sglang)["\']')
        for f in ("_probe_hf_common.py", "_probe_load.py"):
            src = (REPO_ROOT / "scripts" / f).read_text()
            hits = [ln for ln in src.splitlines() if pat.search(ln)]
            self.assertEqual(hits, [], f"{f}: name-keyed engine branches: {hits}")


class ProbeCheckCoversVLLMDevaiTest(unittest.TestCase):
    """`make probe-check` must report image drift for the vllm-devai cache too."""

    def test_probe_check_table_has_the_vllm_devai_backend(self) -> None:
        pc = _load(REPO_ROOT / "scripts" / "probe-check.py", "probe_check_for_devai_test")
        rows = {b[0]: b for b in pc.BACKENDS}
        self.assertIn("vllm-devai", rows)
        name, cache, env, image = rows["vllm-devai"]
        self.assertEqual(cache, "deploy/.vllm-devai-reasoning-cache.json")
        self.assertEqual(env, "VLLM_DEVAI_IMAGE")
        self.assertEqual(image, "docker.io/devai/vllm-devai:latest")


class ProbeDisableShapeMatchesRouterTest(unittest.TestCase):
    """The probe's vLLM disable body must be the router's, field for field.

    applyVLLMPolicy disables with `reasoning_effort: "none"` PLUS
    `chat_template_kwargs.enable_thinking: false`; the prober once sent
    only the kwarg, so the probe kept measuring a thinking model and
    recorded disable_verified=false for rows the router can in fact
    silence (2026-09-22, both derived rows, re-probed twice). Later the
    same day the kwarg moved from `extra_body` -- which vLLM never reads --
    to the top level, on both the router and here; `extra_body` must not
    reappear, or the probe is back to measuring a fiction.
    """

    def test_vllm_disable_body_carries_both_router_fields(self) -> None:
        import _probe_hf_common as hf
        base = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0}
        for backend in ("vllm", "vllm-devai", "sglang"):
            body = hf.build_disable_thinking_body(backend, dict(base))
            self.assertEqual(body.get("reasoning_effort"), "none", backend)
            self.assertIs(body["chat_template_kwargs"]["enable_thinking"], False, backend)
            self.assertNotIn("extra_body", body, backend)
            self.assertEqual(body["temperature"], 0)

    def test_enable_body_is_top_level_on_every_engine(self) -> None:
        import _probe_hf_common as hf
        base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        for backend in ("vllm", "vllm-devai", "sglang"):
            body = hf.build_enable_thinking_body(backend, dict(base))
            self.assertIs(body["chat_template_kwargs"]["enable_thinking"], True, backend)
            self.assertNotIn("extra_body", body, backend)
