"""vLLM launch recipes (plan Phase 17a, from odysseus CB-3).

  scripts/fetch-vllm-recipes.py  github.com/vllm-project/recipes -> one seed
                                 per catalog row in deploy/recipe-seeds.yaml;
                                 offline-first, never a failure
  scripts/recipe-apply.py        one row's reviewed flags -> one `recipe`
                                 group in deploy/recovery-flags.generated.json,
                                 only where min_vllm_version admits the vLLM
                                 build-vllm.sh builds, and only flags the
                                 argument builders never emit
  scripts/_probe_hf_common.py    the parser order: curated, card-derived,
                                 recipe seed, name heuristic, none; and I24's
                                 PROBE_IGNORE_CURATED_PARSERS switch

Network-free: every fetch goes through a fake. Stdlib unittest only.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _probe_hf_common as P  # noqa: E402
import _vllm_recipes as R  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FETCH = _load(REPO_ROOT / "scripts" / "fetch-vllm-recipes.py", "fetch_vllm_recipes")
APPLY = _load(REPO_ROOT / "scripts" / "recipe-apply.py", "recipe_apply")

COMMIT = "a" * 40

QWEN_RECIPE = """\
model:
  model_id: "Qwen/Qwen3.5-9B"
  min_vllm_version: "0.17.0"
  base_args: ["--trust-remote-code"]
  base_env: {}
features:
  tool_calling:
    args: ["--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder"]
  reasoning:
    args: ["--reasoning-parser", "qwen3"]
  text_only:
    args: ["--language-model-only"]
  long_context:
    args: ["--max-model-len", "1010000"]
    env: {VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"}
opt_in_features: [text_only, long_context]
variants:
  default: {precision: bf16, vram_minimum_gb: 22}
  fp8:
    model_id: "RedHatAI/Qwen3.5-9B-FP8-dynamic"
    precision: fp8
    vram_minimum_gb: 11
    extra_args: ["--kv-cache-dtype", "fp8"]
  nvfp4:
    model_id: "nvidia/Qwen3.5-9B-NVFP4"
    precision: nvfp4
    vram_minimum_gb: 7
    extra_args: ["--async-scheduling"]
    extra_env: {VAR: "1"}
  unrelated:
    model_id: "Odd/Unrelated-Name"
    precision: int4
    vram_minimum_gb: 5
"""

NEMOTRON_RECIPE = """\
model:
  model_id: "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
  min_vllm_version: "0.10.1"
  base_args: ["--trust-remote-code"]
features:
  tool_calling:
    args: ["--enable-auto-tool-choice", "--tool-parser-plugin", "x.py", "--tool-call-parser", "nemotron_json"]
variants:
  default: {precision: bf16, vram_minimum_gb: 22}
"""

PATHS = ["models/Qwen/Qwen3.5-9B.yaml", "models/nvidia/NVIDIA-Nemotron-Nano-9B-v2.yaml",
         "models/Qwen/Qwen3-8B.yaml", "models/Other/Qwen3-8B.yaml",
         "models/../../etc/passwd.yaml"]

CATALOG = {"models": [
    {"name": "Qwen3.5-9B-NVFP4", "source": "hf", "repo": "ykarout/Qwen3.5-9B-NVFP4"},
    {"name": "Qwen3.5-9B-NV", "source": "hf", "repo": "nvidia/Qwen3.5-9B-NVFP4"},
    {"name": "Qwen3.5-9B", "source": "hf", "repo": "Qwen/Qwen3.5-9B"},
    {"name": "Qwen3.5-9B-devai-NVFP4", "source": "derived", "repo": "devai/Qwen3.5-9B-devai-NVFP4",
     "derived_from": "Qwen3.5-9B-NVFP4"},
    {"name": "Nano", "source": "hf", "repo": "nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4"},
    {"name": "Qwen3-8B-NVFP4", "source": "hf", "repo": "nvidia/Qwen3-8B-NVFP4"},
    # Named unlike its recipe; found as a variant of the recipe loaded above.
    {"name": "Odd-Name", "source": "hf", "repo": "Odd/Unrelated-Name"},
    {"name": "qwen3.5:9b", "source": "ollama"},
]}


class FakeGitHub:
    """Serves the recipe repository; counts calls; can be switched off."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.down = False
        self.files = {"models/Qwen/Qwen3.5-9B.yaml": QWEN_RECIPE,
                      "models/nvidia/NVIDIA-Nemotron-Nano-9B-v2.yaml": NEMOTRON_RECIPE}

    def __call__(self, url: str, accept: str | None) -> bytes:
        self.calls.append(url)
        if self.down:
            raise OSError("network unreachable")
        if url.endswith("/commits/main"):
            return COMMIT.encode()
        if "/git/trees/" in url:
            return json.dumps({"tree": [{"type": "blob", "path": p} for p in PATHS]}).encode()
        for path, body in self.files.items():
            if url.endswith(f"/{COMMIT}/{path}"):
                return body.encode()
        raise OSError(f"404 {url}")


class FetcherRun(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.cache = self.tmp / "cache"
        self.out = self.tmp / "recipe-seeds.yaml"
        self.catalog = self.tmp / "models.yaml"
        self.catalog.write_text(yaml.safe_dump(CATALOG))
        self.gh = FakeGitHub()

    def run_fetch(self, *extra: str) -> tuple[int, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = FETCH.main(["--catalog", str(self.catalog), "--out", str(self.out),
                             "--cache-dir", str(self.cache), *extra], fetch=self.gh)
        return rc, out.getvalue() + err.getvalue()


class MatchingTest(unittest.TestCase):
    def test_base_name_strips_packaging_suffixes(self) -> None:
        for repo, base in (("nvidia/Qwen3.8-27B-MTP-NVFP4", "qwen3.8-27b"),
                           ("dbirks/Qwen3.8-27B-W4A16-AutoRound", "qwen3.8-27b"),
                           ("devai/Qwen3.8-27B-MTP-devai-NVFP4", "qwen3.8-27b"),
                           ("x/Qwen3.6-27B-Text-NVFP4-MTP", "qwen3.6-27b"),
                           ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nvidia-nemotron-3-nano-30b-a3b"),
                           ("google/gemma-4-31B-it", "gemma-4-31b-it")):
            self.assertEqual(FETCH.base_name(repo), base, repo)
        # A fine-tune's own name is kept, so it matches no base recipe.
        self.assertEqual(FETCH.base_name("x/Qwen3.6-27B-MTP-pi-tune-NVFP4"), "qwen3.6-27b-pi-tune")

    def test_match_exact_then_base_name_and_ambiguity(self) -> None:
        by_id, by_base = FETCH.build_lookup(PATHS[:4])
        self.assertEqual(FETCH.match_repo("qwen/qwen3.5-9b", by_id, by_base),
                         ("models/Qwen/Qwen3.5-9B.yaml", "exact"))
        self.assertEqual(FETCH.match_repo("ykarout/Qwen3.5-9B-NVFP4", by_id, by_base),
                         ("models/Qwen/Qwen3.5-9B.yaml", "base-name"))
        path, why = FETCH.match_repo("nvidia/Qwen3-8B-NVFP4", by_id, by_base)
        self.assertIsNone(path)
        self.assertTrue(why.startswith("ambiguous"))
        self.assertEqual(FETCH.match_repo("x/Unknown-1B", by_id, by_base), (None, None))

    def test_normalise_reads_parsers_from_default_on_features_first(self) -> None:
        norm = FETCH.normalise_recipe(yaml.safe_load(QWEN_RECIPE), "models/Qwen/Qwen3.5-9B.yaml")
        self.assertEqual(norm["parsers"], {"tool": "qwen3_coder", "reasoning": "qwen3"})
        self.assertEqual(norm["base_args"], ["--trust-remote-code"])
        self.assertEqual(norm["min_vllm_version"], "0.17.0")
        self.assertEqual(norm["variants"]["nvfp4"]["extra_args"], ["--async-scheduling"])
        self.assertEqual(norm["variants"]["fp8"]["vram_minimum_gb"], 11)

    def test_a_parser_that_needs_a_plugin_file_is_not_seeded(self) -> None:
        norm = FETCH.normalise_recipe(yaml.safe_load(NEMOTRON_RECIPE), "p")
        self.assertEqual(norm["parsers"], {})
        self.assertIn("plugin", " ".join(norm["notes"]))


class FetchAndSeedTest(FetcherRun):
    def seeds(self) -> dict:
        return R.load_seeds(self.out)

    def test_live_fetch_writes_one_seed_per_matched_row(self) -> None:
        rc, log = self.run_fetch()
        self.assertEqual(rc, 0, log)
        seeds = self.seeds()
        self.assertEqual(list(seeds), ["Qwen3.5-9B-NVFP4", "Qwen3.5-9B-NV", "Qwen3.5-9B",
                                       "Qwen3.5-9B-devai-NVFP4", "Nano", "Odd-Name"])
        odd = seeds["Odd-Name"]
        self.assertEqual((odd["match"], odd["variant"], odd["recipe"], odd["vram_minimum_gb"]),
                         ("variant", "unrelated", "models/Qwen/Qwen3.5-9B.yaml", 5))
        self.assertEqual(FETCH.base_name("Odd/Unrelated-Name"), "unrelated-name")
        doc = yaml.safe_load(self.out.read_text())
        self.assertEqual((doc["version"], doc["commit"]), (1, COMMIT))
        # The traversal path from the tree was never fetched.
        self.assertFalse([u for u in self.gh.calls if "passwd" in u])
        exact = seeds["Qwen3.5-9B"]
        self.assertEqual((exact["match"], exact["variant"], exact["vram_minimum_gb"]),
                         ("exact", "default", 22))
        self.assertEqual(exact["commit"], COMMIT)
        self.assertEqual(exact["parsers"], {"tool": "qwen3_coder", "reasoning": "qwen3"})
        # The row IS a variant's checkpoint: its extras come along.
        variant = seeds["Qwen3.5-9B-NV"]
        self.assertEqual((variant["match"], variant["variant"]), ("variant", "nvfp4"))
        self.assertEqual(variant["engine_flags"], ["--trust-remote-code", "--async-scheduling"])
        self.assertEqual(variant["engine_env"], {"VAR": "1"})
        # Another publisher's NVFP4: base args only, VRAM for its precision.
        base = seeds["Qwen3.5-9B-NVFP4"]
        self.assertEqual((base["match"], base["variant"], base["precision"], base["vram_minimum_gb"]),
                         ("base-name", None, "nvfp4", 7))
        self.assertEqual(base["engine_flags"], ["--trust-remote-code"])
        self.assertEqual(seeds["Qwen3.5-9B-devai-NVFP4"]["recipe"], "models/Qwen/Qwen3.5-9B.yaml")
        self.assertEqual(seeds["Nano"]["min_vllm_version"], "0.10.1")
        self.assertTrue((self.cache / "index.json").is_file())
        self.assertTrue((self.cache / "files" / COMMIT / "models/Qwen/Qwen3.5-9B.yaml").is_file())

    def test_a_fresh_cache_needs_no_network(self) -> None:
        self.run_fetch()
        first = self.out.read_text()
        self.gh.calls.clear()
        self.gh.down = True
        rc, _ = self.run_fetch()
        self.assertEqual(rc, 0)
        self.assertEqual(self.gh.calls, [])
        self.assertEqual(self.out.read_text(), first)

    def test_a_stale_cache_is_used_when_the_network_fails(self) -> None:
        self.run_fetch()
        first = self.out.read_text()
        index = json.loads((self.cache / "index.json").read_text())
        index["fetched_at"] = 0
        (self.cache / "index.json").write_text(json.dumps(index))
        self.gh.down = True
        rc, log = self.run_fetch()
        self.assertEqual(rc, 0)
        self.assertIn("using the cached index", log)
        self.assertEqual(self.out.read_text(), first)

    def test_offline_with_nothing_cached_writes_the_empty_form(self) -> None:
        rc, log = self.run_fetch("--offline")
        self.assertEqual(rc, 0)
        self.assertEqual(self.gh.calls, [])
        doc = yaml.safe_load(self.out.read_text())
        self.assertEqual(doc, {"version": 1, "source": FETCH.RECIPES_URL, "commit": None, "seeds": {}})

    def test_no_data_leaves_an_existing_seeds_file_alone(self) -> None:
        self.out.write_text("version: 1\nseeds: {kept: {}}\n")
        self.gh.down = True
        rc, log = self.run_fetch()
        self.assertEqual(rc, 0)
        self.assertEqual(self.out.read_text(), "version: 1\nseeds: {kept: {}}\n")
        self.assertIn("left as it is", log)

    def test_the_cache_defaults_to_the_user_cache(self) -> None:
        with mock.patch.dict(os.environ, {"XDG_CACHE_HOME": "/x/cache"}, clear=False):
            os.environ.pop("DEVAI_VLLM_RECIPES_CACHE", None)
            self.assertEqual(FETCH.default_cache_dir(), Path("/x/cache/devai/vllm-recipes"))
        with mock.patch.dict(os.environ, {"DEVAI_VLLM_RECIPES_CACHE": "/y"}):
            self.assertEqual(FETCH.default_cache_dir(), Path("/y"))


class ApplicabilityAndKeepTest(FetcherRun):
    """Per-backend applicability for the picker (it cannot read
    build-vllm.sh or the image in service from inside the lab), and a row
    whose recipe cannot be fetched keeping its previous seed."""

    def setUp(self) -> None:
        super().setUp()
        self.curated = self.tmp / "recovery-flags.json"
        self.curated.write_text(json.dumps({"models": {
            "Nano": {"backends": ["vllm"], "image": "x/vllm:v0.9.0-cu129"}}}))
        patches = [
            mock.patch.object(R, "CURATED_FLAGS_PATH", self.curated),
            mock.patch.object(R, "service_vllm_version", return_value="0.28.0"),
            mock.patch.dict(os.environ, {"VLLM_IMAGE": "x/vllm:v0.22.1-cu129"}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        os.environ.pop("VLLM_DEVAI_IMAGE", None)

    def test_each_seed_records_where_it_applies(self) -> None:
        self.assertEqual(self.run_fetch()[0], 0)
        seeds = R.load_seeds(self.out)
        qwen = seeds["Qwen3.5-9B"]
        self.assertEqual(qwen["vllm_version"], {"vllm": "0.22.1", "vllm-devai": "0.28.0"})
        self.assertEqual(qwen["applies_to"], {"vllm": True, "vllm-devai": True})
        # A curated image pin is judged by its own tag, on both vLLM
        # backends (a ["vllm"] entry reaches vllm-devai by engine): 0.9.0 is
        # below the recipe's 0.10.1.
        nano = seeds["Nano"]
        self.assertEqual(nano["vllm_version"], {"vllm": "0.9.0", "vllm-devai": "0.9.0"})
        self.assertEqual(nano["applies_to"], {"vllm": False, "vllm-devai": False})

    def test_a_pin_without_a_version_tag_is_unknown_not_the_devai_build(self) -> None:
        self.curated.write_text(json.dumps({"models": {
            "Nano": {"backends": ["vllm"], "image": "x/vllm:gemma-cu130"}}}))
        self.assertIsNone(R.backend_vllm_version("Nano", "vllm-devai"))
        self.assertEqual(R.backend_vllm_version("Other", "vllm-devai"), "0.28.0")

    def test_a_row_whose_recipe_cannot_be_fetched_keeps_its_previous_seed(self) -> None:
        self.assertEqual(self.run_fetch()[0], 0)
        before = R.load_seeds(self.out)["Nano"]
        # The recipe file is gone from the cache and the network fails.
        cached = self.cache / "files" / COMMIT / "models/nvidia/NVIDIA-Nemotron-Nano-9B-v2.yaml"
        cached.unlink()
        self.gh.down = True
        rc, log = self.run_fetch("--refresh")
        self.assertEqual(rc, 0)
        seeds = R.load_seeds(self.out)
        self.assertEqual(seeds["Nano"]["recipe"], before["recipe"])
        self.assertEqual(seeds["Nano"]["commit"], before["commit"])
        self.assertIn("kept the previous seed", log)


class ShippedSeedsTest(unittest.TestCase):
    def test_the_seeds_file_parses_and_names_catalog_rows(self) -> None:
        path = REPO_ROOT / "deploy" / "recipe-seeds.yaml"
        doc = yaml.safe_load(path.read_text())
        self.assertEqual(doc["version"], 1)
        names = {r["name"] for r in yaml.safe_load(
            (REPO_ROOT / "deploy" / "models.yaml").read_text())["models"]}
        for name, seed in (doc.get("seeds") or {}).items():
            with self.subTest(row=name):
                self.assertIn(name, names)
                for key in ("recipe", "commit", "min_vllm_version", "parsers",
                            "engine_flags", "engine_env", "match"):
                    self.assertIn(key, seed)


class VersionTest(unittest.TestCase):
    def test_parse_and_admit(self) -> None:
        self.assertEqual(R.parse_version("0.28.0"), (0, 28, 0))
        self.assertEqual(R.parse_version("v0.17"), (0, 17, 0))
        self.assertIsNone(R.parse_version("nightly"))
        self.assertTrue(R.version_admits("0.17.0", "0.28.0"))
        self.assertTrue(R.version_admits("0.28.0", "0.28.0"))
        self.assertFalse(R.version_admits("0.29.0", "0.28.0"))
        self.assertFalse(R.version_admits("nightly", "0.28.0"))
        self.assertFalse(R.version_admits(None, "0.28.0"))
        self.assertFalse(R.version_admits("0.17.0", None))

    def test_build_script_version_in_every_form(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "build-vllm.sh"
            for text, want in (('VLLM_VERSION="${VLLM_VERSION:-0.28.0}"\n', "0.28.0"),
                               (': "${VLLM_VERSION:=0.30.0}"\n', "0.30.0"),
                               ('# vLLM $VLLM_VERSION\nVLLM_VERSION=0.31.0\n', "0.31.0"),
                               ('nothing here\n', None)):
                script.write_text(text)
                self.assertEqual(R.service_vllm_version(script), want, text)

    def test_the_real_build_script_names_a_version(self) -> None:
        self.assertIsNotNone(R.parse_version(R.service_vllm_version()))

    def test_image_tags(self) -> None:
        self.assertEqual(R.image_vllm_version("docker.io/vllm/vllm-openai:v0.22.1-x86_64-cu129"), "0.22.1")
        self.assertIsNone(R.image_vllm_version("docker.io/vllm/vllm-openai:latest-x86_64"))
        self.assertIsNone(R.image_vllm_version("localhost:5000/vllm"))


class ExcludedFlagsTest(unittest.TestCase):
    def test_the_set_is_the_golden_corpus_minus_injected_recovery_flags(self) -> None:
        excluded = APPLY.excluded_flags()
        for flag in ("--max-model-len", "--kv-cache-dtype", "--gpu-memory-utilization",
                     "--mem-fraction-static", "--tool-call-parser", "--reasoning-parser",
                     "--enable-auto-tool-choice", "--trust-remote-code", "--served-model-name",
                     "--max-num-seqs", "--max-running-requests", "--context-length",
                     "--speculative-config", "--speculative-algorithm", "--speculative-num-steps",
                     "--speculative-num-draft-tokens", "--speculative-eagle-topk",
                     "--speculative-draft-model-path", "--enable-sleep-mode",
                     "--enable-memory-saver"):
            self.assertIn(flag, excluded)
        self.assertNotIn("--enforce-eager", excluded)
        self.assertNotIn("--async-scheduling", excluded)

    def test_injected_recovery_flags_match_the_go_golden_cases(self) -> None:
        src = (REPO_ROOT / "gpu-arbiter" / "argv_golden_test.go").read_text()
        injected = set()
        for body in re.findall(r"RecoveryFlags:\s*\[\]string\{([^}]*)\}", src):
            injected |= set(re.findall(r'"([^"]+)"', body))
        self.assertEqual(injected, set(APPLY.GOLDEN_RECOVERY_FLAGS))


class ApplyRun(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.seeds = self.tmp / "recipe-seeds.yaml"
        self.generated = self.tmp / "recovery-flags.generated.json"
        self.write_seed(["--enforce-eager", "--max-model-len", "1010000", "--async-scheduling"])

    def write_seed(self, flags: list[str], *, minimum: str = "0.17.0",
                   env: dict | None = None) -> None:
        self.seeds.write_text(yaml.safe_dump({"version": 1, "commit": "c" * 40, "seeds": {
            "M": {"recipe": "models/Q/M.yaml", "commit": "c" * 40, "match": "exact",
                  "min_vllm_version": minimum, "parsers": {}, "engine_flags": flags,
                  "engine_env": env or {}}}}))

    def apply(self, *, version: str | None = "0.28.0", dry_run: bool = False) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            rc = APPLY.apply("M", seeds_path=self.seeds, generated_path=self.generated,
                             golden=APPLY.ARGV_GOLDEN, vllm_version=version,
                             backend="vllm-devai", catalog=None, dry_run=dry_run)
        return rc, out.getvalue()

    def groups(self) -> list[dict]:
        return json.loads(self.generated.read_text())["models"]["M"]


class RecipeApplyTest(ApplyRun):
    def test_enforce_eager_is_written_and_max_model_len_reported(self) -> None:
        rc, log = self.apply()
        self.assertEqual(rc, 0, log)
        [group] = self.groups()
        self.assertEqual(group["engine_flags"], ["--enforce-eager", "--async-scheduling"])
        self.assertEqual(group["backends"], ["vllm-devai"])
        self.assertEqual(group["source"], "recipe")
        self.assertIn("models/Q/M.yaml", group["reason"])
        self.assertRegex(group["added"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertIn("reported, not written", log)
        self.assertIn("--max-model-len 1010000", log)
        # The written file is the generated file's shape, readable by the prober.
        self.assertEqual(P._read_generated_recovery(self.generated)["M"][0]["engine_flags"],
                         ["--enforce-eager", "--async-scheduling"])

    def test_the_version_gate(self) -> None:
        for minimum in ("0.29.0", "nightly"):
            self.write_seed(["--enforce-eager"], minimum=minimum)
            rc, log = self.apply()
            self.assertEqual(rc, 1, minimum)
            self.assertIn("refused", log)
            self.assertFalse(self.generated.exists(), minimum)
        rc, _ = self.apply(version=None)
        self.assertEqual(rc, 1)
        self.write_seed(["--enforce-eager"], minimum="0.28.0")
        self.assertEqual(self.apply()[0], 0)

    def test_the_write_is_atomic(self) -> None:
        self.generated.write_text('{"version": 1, "models": {"Other": []}}\n')
        replaced = []
        real_replace = os.replace

        def spy(src, dst):
            replaced.append((Path(src), Path(dst)))
            return real_replace(src, dst)

        with mock.patch.object(APPLY.os, "replace", spy):
            self.assertEqual(self.apply()[0], 0)
        [(src, dst)] = replaced
        self.assertEqual(dst, self.generated)
        self.assertEqual(src.parent, self.generated.parent)
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()),
                         ["recipe-seeds.yaml", "recovery-flags.generated.json"])
        # A failed rename leaves the old file intact and no temp behind.
        before = self.generated.read_text()
        self.write_seed(["--enforce-eager", "--foo"])
        with mock.patch.object(APPLY.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.apply()
        self.assertEqual(self.generated.read_text(), before)
        self.assertEqual(len(list(self.tmp.iterdir())), 2)

    def test_re_applying_replaces_the_recipe_group_in_place(self) -> None:
        self.generated.write_text(json.dumps({"version": 1, "models": {
            "M": [{"backends": ["vllm-devai"], "engine_flags": ["--x"], "source": "auto-fix",
                   "reason": "r", "added": "a"}],
            "Other": [{"backends": ["vllm"], "engine_flags": ["--y"], "source": "recipe"}]}}))
        self.assertEqual(self.apply()[0], 0)
        self.write_seed(["--enforce-eager", "--new-flag"])
        self.assertEqual(self.apply()[0], 0)
        groups = self.groups()
        self.assertEqual([g["source"] for g in groups], ["auto-fix", "recipe"])
        self.assertEqual(groups[1]["engine_flags"], ["--enforce-eager", "--new-flag"])
        doc = json.loads(self.generated.read_text())
        self.assertEqual(doc["models"]["Other"][0]["engine_flags"], ["--y"])
        # The same seed again changes nothing, not even the date.
        before = self.generated.read_text()
        rc, log = self.apply()
        self.assertEqual(rc, 0)
        self.assertIn("unchanged", log)
        self.assertEqual(self.generated.read_text(), before)

    def test_dry_run_and_nothing_to_write(self) -> None:
        rc, log = self.apply(dry_run=True)
        self.assertEqual(rc, 0)
        self.assertIn("dry run", log)
        self.assertFalse(self.generated.exists())
        self.write_seed(["--max-model-len", "1", "--kv-cache-dtype", "fp8"])
        rc, log = self.apply()
        self.assertEqual(rc, 0)
        self.assertIn("nothing to write", log)
        self.assertFalse(self.generated.exists())

    def test_env_is_written_and_an_unknown_row_refused(self) -> None:
        self.write_seed([], env={"VLLM_X": "1"})
        self.assertEqual(self.apply()[0], 0)
        self.assertEqual(self.groups()[0]["engine_env"], {"VLLM_X": "1"})
        out = io.StringIO()
        with contextlib.redirect_stderr(out):
            rc = APPLY.apply("Nope", seeds_path=self.seeds, generated_path=self.generated,
                             golden=APPLY.ARGV_GOLDEN, vllm_version="0.28.0",
                             backend="vllm-devai", catalog=None, dry_run=False)
        self.assertEqual(rc, 1)
        self.assertIn("make fetch-vllm-recipes", out.getvalue())

    def test_a_malformed_generated_file_is_never_written_over(self) -> None:
        self.generated.write_text('{"version": 1, "models": ')
        with self.assertRaises(SystemExit):
            self.apply()
        self.assertEqual(self.generated.read_text(), '{"version": 1, "models": ')

    def test_the_backend_is_the_one_build_vllm_serves(self) -> None:
        self.assertEqual(R.SERVICE_BACKEND, "vllm-devai")
        src = (REPO_ROOT / "scripts" / "recipe-apply.py").read_text()
        self.assertIn("backend=_vllm_recipes.SERVICE_BACKEND", src)
        self.assertIn("version = _vllm_recipes.service_vllm_version()", src)


class ExcludedFlagBypassTest(ApplyRun):
    """recipe-apply compares the name an engine resolves a flag to, so a
    spelling variant cannot slip a builder-emitted flag (the last flag wins)
    into every launch of a model."""

    def written_and_reported(self, flags: list[str], **kw) -> tuple[list[str], str]:
        self.write_seed(flags, **kw)
        rc, log = self.apply()
        self.assertEqual(rc, 0, log)
        written = self.groups()[0]["engine_flags"] if self.generated.exists() else []
        return written, log

    def test_spelling_variants_of_an_excluded_flag_are_reported(self) -> None:
        for variant in (["--max_model_len", "5"], ["--max-model-len=5"],
                        ["--kv-cache-dtype.sub=fp8"], ["-tp", "2"], ["--max-model", "5"],
                        ["--gpu-memory", "0.9"]):
            with self.subTest(variant=variant):
                if self.generated.exists():
                    self.generated.unlink()
                written, log = self.written_and_reported(["--enforce-eager", *variant])
                self.assertEqual(written, ["--enforce-eager"])
                self.assertIn("reported, not written", log)
                self.assertIn(variant[0], log)

    def test_normalisation(self) -> None:
        self.assertEqual(APPLY.normalise_flag("--max_model_len=5"), "--max-model-len")
        self.assertEqual(APPLY.normalise_flag("--compilation-config.cudagraph_mode=FULL"),
                         "--compilation-config")
        self.assertEqual(APPLY.normalise_flag("-cc.pass_config.x=True"), "-cc")
        self.assertEqual(APPLY.normalise_flag("-tp"), "--tensor-parallel-size")
        self.assertIn("abbreviation", APPLY.why_not_written("--max-num", APPLY.excluded_flags()))
        self.assertIsNone(APPLY.why_not_written("--async-scheduling", APPLY.excluded_flags()))

    def test_a_negative_number_stays_with_its_flag(self) -> None:
        self.assertEqual(APPLY.split_flags(["--x", "-1", "--y", "-0.5", "--z"]),
                         [("--x", ["-1"]), ("--y", ["-0.5"]), ("--z", [])])
        written, _ = self.written_and_reported(["--some-offset", "-1"])
        self.assertEqual(written, ["--some-offset", "-1"])

    def test_rope_and_config_overrides_are_never_written(self) -> None:
        written, log = self.written_and_reported(
            ["--enforce-eager", "--hf-overrides", '{"rope_parameters": {}}',
             "--rope-scaling", '{"type": "yarn"}', "--rope_scaling={}"])
        self.assertEqual(written, ["--enforce-eager"])
        self.assertIn("used as delivered", log)

    def test_a_flag_or_env_the_curated_entry_sets_is_reported(self) -> None:
        curated = self.tmp / "curated.json"
        curated.write_text(json.dumps({"models": {"M": {
            "backends": ["vllm"], "engine_flags": ["--async-scheduling", "--max-num-batched-tokens", "8"],
            "engine_env": {"VLLM_X": "curated"}}}}))
        self.write_seed(["--enforce-eager", "--async-scheduling", "--max_num_batched_tokens=4"],
                        env={"VLLM_X": "recipe", "VLLM_Y": "1"})
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            rc = APPLY.apply("M", seeds_path=self.seeds, generated_path=self.generated,
                             golden=APPLY.ARGV_GOLDEN, vllm_version="0.28.0",
                             backend="vllm-devai", catalog=None, dry_run=False,
                             curated_path=curated)
        self.assertEqual(rc, 0, out.getvalue())
        [group] = self.groups()
        self.assertEqual(group["engine_flags"], ["--enforce-eager"])
        self.assertEqual(group["engine_env"], {"VLLM_Y": "1"})
        self.assertIn("curated entry already sets it", out.getvalue())
        self.assertIn("VLLM_X=recipe", out.getvalue())
        # A curated entry scoped elsewhere does not count.
        self.assertEqual(APPLY.curated_flag_names("M", "sglang", curated), (frozenset(), frozenset()))


class ParserOrderTest(unittest.TestCase):
    """Curated, card-derived, recipe seed, name heuristic, none."""

    def resolve(self, *, curated=(None, None), card=None, recipe=None, name="Qwen3.5-9B-NVFP4",
                backend="vllm-devai", ignore=False):
        def derive(n, d, b, kind):
            return (card or {}).get(kind)
        with mock.patch.object(P, "derive_parser_for", side_effect=derive), \
                contextlib.redirect_stderr(io.StringIO()):
            return P.resolve_parsers(name, Path("/m"), backend, curated[0], curated[1],
                                     recipe_parsers=recipe, ignore_curated=ignore)

    def test_each_step_in_order(self) -> None:
        both = {"tool": "t", "reasoning": "r"}
        r, t, src, _ = self.resolve(curated=("cr", "ct"), card={"tool": "kt", "reasoning": "kr"},
                                    recipe=both)
        self.assertEqual((r, t, src), ("cr", "ct", {"tool": "curated", "reasoning": "curated"}))
        r, t, src, _ = self.resolve(card={"tool": "kt"}, recipe=both)
        self.assertEqual((t, src["tool"], r, src["reasoning"]), ("kt", "card", "r", "recipe"))
        r, t, src, _ = self.resolve()
        self.assertEqual((r, t, src), ("qwen3", "qwen3_xml",
                                       {"tool": "heuristic", "reasoning": "heuristic"}))
        r, t, src, _ = self.resolve(name="Ornith-1.0-9B-NVFP4")
        self.assertEqual((r, t, src), (None, None, {"tool": None, "reasoning": None}))

    def test_a_card_disagreement_is_recorded_and_curated_still_wins(self) -> None:
        r, t, src, dis = self.resolve(curated=(None, "qwen3_xml"), card={"tool": "hermes"})
        self.assertEqual((t, src["tool"]), ("qwen3_xml", "curated"))
        self.assertEqual(dis, {"tool": {"curated": "qwen3_xml", "derived": "hermes"}})

    def test_ignore_curated_leaves_the_card_derivation_alone(self) -> None:
        r, t, src, _ = self.resolve(curated=("cr", "ct"), card={"tool": "openai"},
                                    recipe={"tool": "x", "reasoning": "y"}, ignore=True,
                                    name="gpt-oss-20b")
        self.assertEqual((r, t), (None, "openai"))
        self.assertEqual(src, {"tool": "card", "reasoning": None})

    def test_the_heuristic_is_per_engine_and_first_match_wins(self) -> None:
        self.assertEqual(P.heuristic_parser("gpt-oss-20b", "sglang", "tool"), "gpt-oss")
        self.assertEqual(P.heuristic_parser("gpt-oss-20b", "vllm-devai", "reasoning"), "openai_gptoss")
        self.assertEqual(P.heuristic_parser("Qwen3-Coder-30B", "vllm", "tool"), "qwen3_coder")
        self.assertIsNone(P.heuristic_parser("Qwen3-Coder-30B", "vllm", "reasoning"))
        self.assertEqual(P.heuristic_parser("Qwen3-8B-NVFP4", "vllm", "tool"), "hermes")
        self.assertEqual(P.heuristic_parser("Qwen3-8B-NVFP4", "sglang", "tool"), "qwen")
        self.assertIsNone(P.heuristic_parser("Gemma-4-31B-IT", "sglang", "tool"))
        # Only values the caches recorded in AUTO mode: SGLang's `qwen` for
        # the Qwen3.5+ markup passed only the forced call, the Llama tool
        # parsers likewise, and SGLang qwen3_coder was never verified.
        self.assertIsNone(P.heuristic_parser("Qwen3.6-27B-New", "sglang", "tool"))
        self.assertEqual(P.heuristic_parser("Qwen3.6-27B-New", "sglang", "reasoning"), "qwen3")
        self.assertIsNone(P.heuristic_parser("Llama-3.1-8B-Instruct", "vllm", "tool"))
        self.assertIsNone(P.heuristic_parser("Llama-3.1-8B-Instruct", "sglang", "tool"))
        self.assertIsNone(P.heuristic_parser("Qwen3-Coder-30B", "sglang", "tool"))
        self.assertEqual(P.heuristic_parser("DeepSeek-R1-0528-Qwen3-8B", "vllm", "reasoning"),
                         "deepseek_r1")
        plugins = set(json.loads((REPO_ROOT / "deploy" / "vllm-plugins.json").read_text())["plugins"])
        for _pattern, by_engine in P.NAME_PARSER_HEURISTICS:
            for parsers in by_engine.values():
                self.assertFalse(set(parsers.values()) & plugins)

    def test_the_recipe_seed_is_vllm_only_and_version_gated(self) -> None:
        seeds = {"M": {"min_vllm_version": "0.17.0", "parsers": {"tool": "qwen3_coder"}},
                 "New": {"min_vllm_version": "0.99.0", "parsers": {"tool": "x"}}}
        with mock.patch.object(R, "service_vllm_version", return_value="0.28.0"):
            self.assertEqual(P.recipe_seed_parsers("M", "vllm-devai", None, seeds),
                             {"tool": "qwen3_coder"})
            self.assertEqual(P.recipe_seed_parsers("New", "vllm-devai", None, seeds), {})
        self.assertEqual(P.recipe_seed_parsers("M", "sglang", "img:v0.5.16", seeds), {})
        self.assertEqual(P.recipe_seed_parsers("M", "vllm", "x/vllm:v0.22.1-cu129", seeds),
                         {"tool": "qwen3_coder"})
        # Stock vLLM tagged `latest` names no version: nothing is seeded.
        self.assertEqual(P.recipe_seed_parsers("M", "vllm", "x/vllm:latest-x86", seeds), {})
        self.assertEqual(P.recipe_seed_parsers("Absent", "vllm-devai", None, seeds), {})

    def test_the_switch_reaches_the_prober_from_make(self) -> None:
        spec = mock.Mock(cache_path=REPO_ROOT / "deploy" / "c.json", image="i",
                         container_name="n", probe_port=1)
        with mock.patch.dict(os.environ, {"PROBE_IGNORE_CURATED_PARSERS": "1"}):
            self.assertTrue(P.build_argparser(spec, "doc").parse_args([]).ignore_curated_parsers)
        with mock.patch.dict(os.environ, {"PROBE_IGNORE_CURATED_PARSERS": ""}):
            ap = P.build_argparser(spec, "doc")
            self.assertFalse(ap.parse_args([]).ignore_curated_parsers)
            self.assertTrue(ap.parse_args(["--ignore-curated-parsers"]).ignore_curated_parsers)
        import inspect
        self.assertIn('getattr(args, "ignore_curated_parsers", False)',
                      inspect.getsource(P.check_ignore_curated))
        src = inspect.getsource(P.run_probe_pass)
        self.assertIn("ignore_curated = check_ignore_curated(spec, args)", src)
        self.assertIn("recipe_parsers=recipe_parsers, ignore_curated=ignore_curated", src)
        mk = (REPO_ROOT / "Makefile").read_text()
        for target in ("probe-vllm", "probe-vllm-devai"):   # probe-sglang left with M16
            body = re.search(rf"^{re.escape(target)}:.*?(?=^\S)", mk, re.M | re.S).group(0)
            with self.subTest(target=target):
                self.assertIn('PROBE_IGNORE_CURATED_PARSERS="$(PROBE_IGNORE_CURATED_PARSERS)"', body)


class GuessesAndCurationTest(unittest.TestCase):
    def test_a_curated_block_keeps_its_omissions(self) -> None:
        # gemma4 curates `tool: gemma4` and deliberately no reasoning parser
        # (measured; scripts/model-families.yaml). A recipe naming one must
        # not fill that gap; the card derivation still may.
        row = {"name": "Gemma-4-31B-IT-NVFP4", "parsers": {"vllm": {"tool": "gemma4"}}}
        self.assertTrue(P.has_curated_block(row, "vllm-devai"))
        self.assertFalse(P.has_curated_block({"parsers": {"vllm": {"tool": "x"}}}, "sglang"))
        self.assertFalse(P.has_curated_block({"name": "x"}, "vllm"))
        with mock.patch.object(P, "derive_parser_for", return_value=None), \
                contextlib.redirect_stderr(io.StringIO()):
            r, t, src, _ = P.resolve_parsers(
                row["name"], Path("/m"), "vllm-devai", None, "gemma4",
                recipe_parsers={"reasoning": "gemma4"}, curated_block=True)
        self.assertEqual((r, t), (None, "gemma4"))
        self.assertEqual(src, {"tool": "curated", "reasoning": None})

    def test_an_engine_rejecting_a_parser_name_is_a_command_line_failure(self) -> None:
        for line in ("KeyError: 'invalid tool call parser: foo (chose from { hermes })'",
                     "KeyError: 'invalid reasoning parser: bar (chose from { qwen3 })'"):
            self.assertEqual(P.classify_failure_logs("x\n" + line + "\n")["kind"], "launch_args")

    def test_ignore_curated_needs_a_scoped_scratch_run(self) -> None:
        # Scoped rows, a scratch cache AND (since 4v) a scratch ledger: the
        # run's cells and its exclusion verdicts both come from card-derived
        # parsers only (plan I24).
        spec = mock.Mock(cache_path=REPO_ROOT / "deploy" / ".vllm-reasoning-cache.json")
        import argparse
        real_ledger = REPO_ROOT / "deploy" / ".model-status.json"
        scratch = {"cache": Path("/tmp/scratch.json"), "ledger": Path("/tmp/scratch-ledger.json")}
        base = dict(ignore_curated_parsers=True, repo="gpt-oss", no_cache_write=False,
                    cache=spec.cache_path, ledger=None)
        with contextlib.redirect_stderr(io.StringIO()):
            for bad in ({}, {"repo": ""}, {"ledger": scratch["ledger"]},
                        {"cache": scratch["cache"]},
                        {"cache": scratch["cache"], "ledger": real_ledger},
                        {**scratch, "repo": ""}):
                with self.subTest(bad=bad), self.assertRaises(SystemExit):
                    P.check_ignore_curated(spec, argparse.Namespace(**{**base, **bad}))
            self.assertTrue(P.check_ignore_curated(
                spec, argparse.Namespace(**{**base, **scratch})))
            # --no-cache-write makes an already-fitting row look fully
            # cached, so the run would probe nothing: refused.
            for bad in ({"no_cache_write": True},
                        {"no_cache_write": True, **scratch}):
                with self.assertRaises(SystemExit):
                    P.check_ignore_curated(spec, argparse.Namespace(**{**base, **bad}))
            self.assertFalse(P.check_ignore_curated(
                spec, argparse.Namespace(**{**base, "ignore_curated_parsers": False, "repo": ""})))


class GuessedParserFallbackTest(unittest.TestCase):
    """run_probe_pass with the launcher replaced by a scripted fake (the
    pattern of test_probe_exact_ctx.py): an engine that rejects a GUESSED
    parser at startup must cost the model one extra launch, not its cell."""

    def _run(self, *, row: dict, rejects: bool, cache: dict | None = None,
             force: bool = False, failure_kind: str = "launch_args",
             always_fail_first: bool = False, ctx_exact: int | None = None,
             fits_below: int | None = None) -> tuple[dict, list, int, str]:
        import argparse
        import types
        calls: list[tuple[int, str | None, str | None]] = []
        cache = cache if cache is not None else {}
        names = ("assert_no_active_backends", "install_probe_cleanup", "load_catalog_hf_rows",
                 "is_downloaded", "model_kind_from_disk", "model_size_gb_from_row", "load_cache",
                 "save_cache", "image_digest_via_cli", "effective_position_limit", "probe_one_cell",
                 "_load_ledger", "_save_ledger", "_ledger_reason", "_ledger_record",
                 "_ledger_clear", "_ledger_is_excluded", "derive_parser_for")
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
        P.effective_position_limit = lambda name, models_dir: 65536
        P._load_ledger = lambda: {}
        P._save_ledger = lambda ledger, host_vram_gb=None: None
        P._ledger_reason = lambda ledger, name, backend: None
        P._ledger_record = lambda ledger, name, backend, reason, **kw: None
        P._ledger_clear = lambda ledger, name, backend: None
        P._ledger_is_excluded = lambda ledger, name, backend, **kw: False
        P.derive_parser_for = lambda name, models_dir, backend, kind: None

        def fake_probe(spec, *, requested_ctx, reasoning_parser=None, tool_parser=None, **kw):
            calls.append((requested_ctx, reasoning_parser, tool_parser))
            fail = (rejects and (tool_parser or reasoning_parser)) \
                or (always_fail_first and len(calls) <= 2) \
                or (fits_below is not None and requested_ctx >= fits_below)
            if fail:
                return {"ctx": requested_ctx, "vram_gb": 24, "fits": False,
                        "capability": "error", "probed_at": "t",
                        "evidence": {"kind": failure_kind}}
            return {"ctx": requested_ctx, "vram_gb": 24, "fits": True, "capability": "structured",
                    "actual_context": requested_ctx, "actual_vram_gb": 20.0,
                    "reasoning_parser": reasoning_parser, "tool_parser": tool_parser,
                    "disable_verified": False, "probed_at": "t", "startup_seconds": 1.0,
                    "evidence": {}}
        P.probe_one_cell = fake_probe
        with tempfile.TemporaryDirectory() as td:
            args = argparse.Namespace(
                runtime="podman", repo=row["repo"], catalog=Path(td) / "models.yaml",
                cache=Path(td) / "cache.json", models_dir=td, vram="", ctx="32K,64K",
                host_vram_gb=24, image="img:latest", container_name="c", probe_port=18000,
                prompt="hi", force=force, force_arch=False, no_mtp=True,
                no_cache_write=False, ctx_exact=ctx_exact)
            spec = types.SimpleNamespace(name="sglang", schema_version=2)
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
        entry = next(v for k, v in cache.items() if not k.startswith("_"))
        return entry, calls, rc, err.getvalue()

    ROW = {"name": "Qwen3.6-27B-New", "repo": "org/Qwen3.6-27B-New", "sha": "s1", "parsers": {}}

    def test_a_failed_first_launch_with_a_guess_is_retried_without_it(self) -> None:
        # Any failure kind counts (here the engine dies in Probe A), but the
        # guess is blamed only because the same context then fits without it.
        entry, calls, rc, err = self._run(row=dict(self.ROW), rejects=True,
                                          failure_kind="oom_chat")
        self.assertEqual(rc, 0, err)
        self.assertEqual(calls[0][1:], ("qwen3", None))   # the heuristic's guess
        self.assertEqual(calls[1], (calls[0][0], None, None))
        self.assertTrue(all(c[1:] == (None, None) for c in calls[1:]))
        self.assertTrue(any(c.get("fits") for c in entry["probes"]["24"].values()))
        self.assertIsNone(entry["reasoning_parser_source"])
        fb = entry["parser_fallback"]["reasoning"]
        self.assertEqual((fb["parser"], fb["source"], fb["failure"]),
                         ("qwen3", "heuristic", "oom_chat"))
        self.assertIn("retrying without them", err)

    def test_a_failure_without_the_guess_too_is_not_blamed_on_it(self) -> None:
        # e.g. a recovery flag the engine rejects: the run without the guess
        # fails the same way, so the guess stays and nothing is recorded.
        entry, calls, rc, err = self._run(row=dict(self.ROW), rejects=False,
                                          always_fail_first=True)
        self.assertEqual(rc, 0, err)
        self.assertEqual(calls[0][1:], ("qwen3", None))
        self.assertEqual(calls[1][1:], (None, None))
        self.assertTrue(calls[2:] and all(c[1:] == ("qwen3", None) for c in calls[2:]), calls)
        self.assertTrue(any(c.get("fits") for c in entry["probes"]["24"].values()))
        self.assertNotIn("parser_fallback", entry)
        self.assertEqual(entry["reasoning_parser_source"], "heuristic")
        self.assertIn("keeping the guesses", err)

    def test_a_recorded_fallback_is_honoured_by_the_next_run(self) -> None:
        entry, calls, rc, err = self._run(row=dict(self.ROW), rejects=True, failure_kind="infra")
        recorded = entry["parser_fallback"]
        cache = {"org/Qwen3.6-27B-New@s1": entry}
        entry2, calls2, rc2, err2 = self._run(row=dict(self.ROW), rejects=True,
                                              cache=cache, force=True)
        self.assertEqual(rc2, 0, err2)
        self.assertTrue(all(c[1:] == (None, None) for c in calls2), calls2)
        self.assertEqual(entry2["parser_fallback"], recorded, "kept while it keeps the guess out")
        self.assertIn("an earlier run dropped it", err2)

    def test_an_accepted_guess_is_probed_and_labelled(self) -> None:
        entry, calls, rc, err = self._run(row=dict(self.ROW), rejects=False)
        self.assertEqual(rc, 0, err)
        self.assertTrue(all(c[1:] == ("qwen3", None) for c in calls))
        self.assertEqual(entry["reasoning_parser_source"], "heuristic")
        self.assertNotIn("parser_fallback", entry)
        self.assertIn(P.HEURISTIC_LABEL, err)

    def test_a_cached_row_keeps_the_provenance_of_its_cells(self) -> None:
        cache = {"org/Qwen3.6-27B-New@s1": {
            "schema_version": 2, "repo": "org/Qwen3.6-27B-New", "sha": "s1",
            "aliases": ["Qwen3.6-27B-New"], "capability": "structured",
            "image_digest": "sha256:deadbeef", "tool_parser_source": None,
            "probes": {"24": {"65536": {"ctx": 65536, "fits": True, "capability": "structured",
                                        "probed_at": "t", "evidence": {}}}}}}
        entry, calls, rc, err = self._run(row=dict(self.ROW), rejects=False, cache=cache)
        self.assertEqual((rc, calls), (0, []), err)
        self.assertIsNone(entry["tool_parser_source"],
                          "a row not probed in this run must not acquire a new source")

    def test_an_exact_miss_restores_the_provenance_with_the_band(self) -> None:
        prior = {"ctx": 32768, "fits": True, "capability": "structured",
                 "probed_at": "t", "evidence": {}}
        cache = {"org/Qwen3.6-27B-New@s1": {
            "schema_version": 2, "repo": "org/Qwen3.6-27B-New", "sha": "s1",
            "aliases": ["Qwen3.6-27B-New"], "capability": "structured",
            "image_digest": "sha256:deadbeef", "reasoning_parser_source": "card",
            "parser_disagreement": {"tool": {"curated": "a", "derived": "b"}},
            "probes": {"24": {"32768": dict(prior)}}}}
        entry, calls, rc, err = self._run(row=dict(self.ROW), rejects=False, cache=cache,
                                          ctx_exact=60000, fits_below=40000)
        self.assertNotEqual(rc, 0)
        self.assertEqual(entry["probes"]["24"], {"32768": prior})
        self.assertEqual(entry["reasoning_parser_source"], "card")
        self.assertEqual(entry["parser_disagreement"], {"tool": {"curated": "a", "derived": "b"}})
        self.assertNotIn("tool_parser_source", entry)


class ToolProbeAutoOnlyTest(unittest.TestCase):
    """Probe B (run_tool_probe): a guessed tool parser is recorded only when
    the auto call verifies it. A named tool_choice is satisfied by
    constrained decoding without the parser, so the forced call proves
    nothing about a guess; a row recorded tool_mode=forced would answer every
    multi-tool request 400 at the router."""

    CALL = {"choices": [{"message": {"tool_calls": [
        {"type": "function", "function": {"name": "get_time", "arguments": "{}"}}]}}]}
    NONE = {"choices": [{"message": {"content": "It is noon."}}]}

    def probe(self, *, auto_ok: bool, forced_ok: bool, auto_only: bool):
        replies = [self.CALL if auto_ok else self.NONE, self.CALL if forced_ok else self.NONE]
        with mock.patch.object(P, "_post_chat", side_effect=lambda url, body: replies.pop(0)):
            return P.run_tool_probe("http://x", "m", "hermes", "none", auto_only=auto_only)

    def test_a_guess_passing_only_the_forced_call_is_not_recorded(self) -> None:
        verified, ev = self.probe(auto_ok=False, forced_ok=True, auto_only=True)
        self.assertIsNone(verified)
        self.assertEqual((ev["verified"], ev["mode"], ev["auto_only"]), (False, "forced", True))
        self.assertIn("constrained decoding", ev["reason"])

    def test_a_guess_passing_the_auto_call_is_recorded(self) -> None:
        self.assertEqual(self.probe(auto_ok=True, forced_ok=False, auto_only=True),
                         ("hermes", {"verified": True, "mode": "auto"}))

    def test_curated_and_card_keep_todays_rule(self) -> None:
        verified, ev = self.probe(auto_ok=False, forced_ok=True, auto_only=False)
        self.assertEqual((verified, ev["mode"], ev["verified"]), ("hermes", "forced", True))

    def test_the_probe_passes_the_flag_for_guessed_sources_only(self) -> None:
        import inspect
        src = inspect.getsource(P.run_probe_pass)
        # The fit launch and the MTP launch both pass it.
        self.assertEqual(src.count('parser_sources.get("tool") in GUESSED_PARSER_SOURCES'), 2)
        self.assertIn("tool_parser_auto_only", inspect.getsource(P.probe_one_cell))


class LoadPassTest(unittest.TestCase):
    def test_the_load_pass_refuses_the_i24_switch(self) -> None:
        import argparse
        import _probe_load as LP
        with mock.patch.object(LP, "assert_no_active_backends") as guard, \
                self.assertRaises(SystemExit) as cm:
            LP.run_load_probe_pass(mock.Mock(name="vllm"),
                                   argparse.Namespace(ignore_curated_parsers=True))
        guard.assert_not_called()
        self.assertIn("fit pass only", str(cm.exception))


class MakeTargetsTest(unittest.TestCase):
    def test_targets_are_documented(self) -> None:
        mk = (REPO_ROOT / "Makefile").read_text()
        self.assertRegex(mk, r"(?m)^fetch-vllm-recipes: ## ")
        self.assertRegex(mk, r"(?m)^recipe-apply: ## ")
        body = re.search(r"^recipe-apply:.*?(?=^\S)", mk, re.M | re.S).group(0)
        self.assertIn("scripts/recipe-apply.py --name '$(NAME)'", body)


if __name__ == "__main__":
    unittest.main()
