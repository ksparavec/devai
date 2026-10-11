"""devai-vllm is built here, from source, and from nothing upstream ships.

Operator decision 2026-09-20 (same shape as devai-ollama): vLLM is compiled on
the host with the HyperQwen patch series applied, and the image is
debian:trixie-slim plus that build. No upstream vllm/vllm-openai image is
used, not even as a build input.

What these tests pin is every way the image could come out looking right and
being wrong:

  * an online `pip install` would quietly pull PyPI's vLLM over ours,
  * a wheelhouse still carrying PyPI's vLLM wheel would do the same offline,
  * an install without the patches starts fine and behaves like stock vLLM,
  * a runtime set other than the build record's hash lock (platform-rebuild
    4v: the lock with --require-hashes, then our wheel with --no-deps, then
    `pip check`),
  * a `python3` that is not the venv's breaks the router's launch command,
    which is `python3 -m vllm.entrypoints.openai.api_server ...`.

Text-level checks only; nothing is built.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCKERFILE = (REPO_ROOT / "deploy" / "Dockerfile.vllm").read_text()
MAKEFILE = (REPO_ROOT / "Makefile").read_text()
SCRIPT = (REPO_ROOT / "scripts" / "build-vllm.sh").read_text()
RECORDS = sorted((REPO_ROOT / "deploy" / "vllm-builds").glob("*.sh"))
ROUTER = (REPO_ROOT / "gpu-arbiter" / "main.go").read_text()
WORKFLOW = (REPO_ROOT / ".github" / "workflows" / "security-blocking.yml").read_text()


def _executed(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


class DockerfileTest(unittest.TestCase):
    def test_base_is_trixie_slim_and_no_upstream_vllm_image_is_an_input(self) -> None:
        froms = re.findall(r"(?m)^FROM\s+(\S+)", DOCKERFILE)
        self.assertEqual(froms, ["${BASE_IMAGE}"])
        self.assertIn("ARG BASE_IMAGE=docker.io/library/debian:trixie-slim", DOCKERFILE)
        self.assertNotRegex(_executed(DOCKERFILE), r"vllm/vllm-openai|--from=")

    def test_install_is_offline_from_the_mounted_wheelhouse(self) -> None:
        body = _executed(DOCKERFILE)
        self.assertIn("--no-index", body)
        self.assertIn('--find-links "$D/wheels"', body)
        self.assertNotRegex(body, r"pip install(?![^\n]*--no-index)")

    def test_a_pypi_vllm_wheel_in_the_wheelhouse_is_refused(self) -> None:
        body = _executed(DOCKERFILE)
        self.assertIn("*manylinux*)", body)
        self.assertIn("expected exactly one vLLM wheel", body)

    def test_build_fails_without_the_patches_or_the_compiled_extension(self) -> None:
        body = _executed(DOCKERFILE)
        self.assertIn("grep -q draft_vocab_ids", body)
        self.assertIn("lacks the HyperQwen patches", body)
        self.assertIn("no compiled _C extension", body)

    def test_the_hyperqwen_check_runs_only_when_build_info_names_a_commit(self) -> None:
        # A record without HyperQwen (Phase 23's stock 0.31.0, I29) writes
        # `hyperqwen=none`; its image must build without the check.
        body = _executed(DOCKERFILE)
        guard = body.index("""if grep -Eq '^hyperqwen=[0-9a-f]{40}$' "$D/BUILD_INFO"; then""")
        self.assertLess(guard, body.index("grep -q draft_vocab_ids"))
        self.assertIn('echo "hyperqwen=${HYPERQWEN_COMMIT:-none}"', SCRIPT)

    def test_runtime_set_is_the_lock_then_our_wheel_then_pip_check(self) -> None:
        body = _executed(DOCKERFILE)
        lock = body.index('--require-hashes -r "$D/requirements.txt"')
        wheel = body.index('--no-deps "$D"/wheels/vllm-*.whl')
        check = body.index("/opt/vllm/bin/pip check")
        self.assertLess(lock, wheel)
        self.assertLess(wheel, check)
        # ... and the dist's requirements.txt IS the record's lock.
        self.assertIn('cp "$VLLM_RECORD_DIR/$VLLM_VERSION.lock" "$dist/requirements.txt"', SCRIPT)

    def test_python3_on_path_is_the_venv_the_router_launches_through(self) -> None:
        # The venv must come FIRST: Debian's own python3 is installed too.
        self.assertRegex(DOCKERFILE, r"(?m)^ENV PATH=/opt/vllm/bin:")
        self.assertIn('"python3", "-m", "vllm.entrypoints.openai.api_server"', ROUTER)

    def test_run_time_compilers_are_in_the_image(self) -> None:
        # Triton needs gcc + Python.h; FlashInfer's JIT needs g++ and nvcc.
        # Without nvcc the engine loads 18 GiB of weights and THEN dies in
        # KV-cache init (measured 2026-09-21) -- fp8 KV selects FLASHINFER.
        body = _executed(DOCKERFILE)
        for pkg in ("gcc", "g++", "python3-dev"):
            self.assertRegex(body, rf"(?<![\w+-]){re.escape(pkg)}(?![\w+-])")
        self.assertIn("/usr/local/cuda/bin/nvcc --version", body)
        self.assertIn("CUDA_HOME=/usr/local/cuda", body)

    def test_cuda_compiler_comes_from_the_host_toolkit_not_from_apt(self) -> None:
        body = _executed(DOCKERFILE)
        self.assertIn("/var/cache/cuda-toolkit", body)
        self.assertNotRegex(body, r"apt-get install[^;]*cuda-")
        self.assertIn("no CUDA toolkit mounted", body)

    def test_a_rebuilt_wheelhouse_invalidates_the_install_layer(self) -> None:
        # podman caches a RUN layer on its instruction text and does not look
        # inside a bind mount. On 2026-09-21 a rebuilt wheel (one more patch)
        # produced an image that still held the OLD vLLM, with exit code 0.
        # The content hash must be declared before the install layer, used
        # inside it, and computed from the wheel itself by the Makefile.
        before_install = DOCKERFILE.split("pip install", 1)[0]
        self.assertIn("ARG DIST_ID", before_install)
        self.assertIn("$DIST_ID", before_install.split("ARG DIST_ID", 1)[1])
        recipe = MAKEFILE.split("\nbuild-vllm-image:", 1)[1].split("\n\n", 1)[0]
        self.assertRegex(recipe, r"--build-arg DIST_ID=\$\$\(cat \$\(VLLM_DIST\)/wheels/vllm-\*\.whl")
        self.assertIn("sha256sum", recipe)
        # A new lock changes the install as much as a new wheel does.
        self.assertIn("$(VLLM_DIST)/requirements.txt | sha256sum", recipe)

    def test_the_ollama_image_has_the_same_protection(self) -> None:
        dockerfile = (REPO_ROOT / "deploy" / "Dockerfile.ollama").read_text()
        before_copy = dockerfile.split("install -m 0755", 1)[0]
        self.assertIn("ARG DIST_ID", before_copy)
        self.assertIn("$DIST_ID", before_copy.split("ARG DIST_ID", 1)[1])
        recipe = MAKEFILE.split("\nbuild-ollama-image:", 1)[1].split("\n\n", 1)[0]
        self.assertIn("--build-arg DIST_ID=", recipe)
        self.assertIn("$(OLLAMA_DIST)/bin/ollama", recipe)

    def test_dockerfile_is_linted_in_ci(self) -> None:
        self.assertIn("deploy/Dockerfile.vllm", WORKFLOW)


class MakefileTest(unittest.TestCase):
    def _recipe(self, target: str) -> str:
        return MAKEFILE.split(f"\n{target}:", 1)[1].split("\n\n", 1)[0]

    def test_image_target_mounts_the_host_build_read_only(self) -> None:
        recipe = self._recipe("build-vllm-image")
        self.assertIn("-v $(VLLM_DIST):/var/cache/vllm-dist:ro", recipe)
        self.assertIn("-v $(VLLM_CUDA_HOME):/var/cache/cuda-toolkit:ro", recipe)
        self.assertIn("-f deploy/Dockerfile.vllm", recipe)
        self.assertIn("run 'make build-vllm-dist' first", recipe)

    def test_dist_lives_outside_the_volume_backed_cache_tree_one_root_per_version(self) -> None:
        # One build root per version (4v): a 0.30.0 build deletes only its
        # own dist/, never 0.28.0's wheelhouse, which is the rollback.
        self.assertRegex(MAKEFILE, r"(?m)^VLLM_BUILD_ROOT \?= \$\(HOME\)/\.cache/devai/vllm-build/\$\(VLLM_VERSION\)$")
        self.assertRegex(MAKEFILE, r"(?m)^VLLM_DIST \?= \$\(VLLM_BUILD_ROOT\)/dist$")
        self.assertIn('BUILD_ROOT="${VLLM_BUILD_ROOT:-$HOME/.cache/devai/vllm-build/$VLLM_VERSION}"', SCRIPT)

    def test_dist_target_builds_the_version_it_is_given(self) -> None:
        recipe = self._recipe("build-vllm-dist")
        self.assertIn("VLLM_VERSION=$(VLLM_VERSION) VLLM_BUILD_ROOT=$(VLLM_BUILD_ROOT)", recipe)
        self.assertRegex(MAKEFILE, r"(?m)^VLLM_VERSION \?= 0\.30\.0$")

    def test_build_vllm_does_both_steps_in_order(self) -> None:
        self.assertRegex(MAKEFILE, r"(?m)^build-vllm: build-vllm-dist build-vllm-image\b")


class BuildScriptShapeTest(unittest.TestCase):
    def test_builds_for_this_gpu_only(self) -> None:
        self.assertIn('CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-12.0}"', SCRIPT)
        self.assertIn('TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST"', SCRIPT)

    def test_sdist_is_hash_pinned(self) -> None:
        for record in RECORDS:
            self.assertRegex(record.read_text(), r"(?m)^VLLM_SDIST_SHA256=[0-9a-f]{64}$", record.name)
        self.assertIn("sha256sum", SCRIPT)
        self.assertIn('[ "$have" = "$VLLM_SDIST_SHA256" ]', SCRIPT)

    def test_hyperqwen_is_pinned_to_a_full_commit_or_absent(self) -> None:
        for record in RECORDS:
            self.assertRegex(record.read_text(), r"(?m)^HYPERQWEN_COMMIT=([0-9a-f]{40})?$", record.name)
        self.assertIn("HYPERQWEN_COMMIT must be a full 40-hex commit or empty", SCRIPT)

    def test_patches_are_applied_without_fuzz(self) -> None:
        self.assertIn("patch -p1 --fuzz 0", _executed(SCRIPT))

    def test_every_external_has_a_source_dir_override_and_a_pin_file(self) -> None:
        counts = {}
        for record in RECORDS:
            rows = re.findall(r'(?m)^  "([^"]+)"$', record.read_text())
            counts[record.stem] = len(rows)
            for row in rows:
                name, var, repo, ref, pinfile = row.split("|")[:5]
                self.assertTrue(var.endswith("_SRC_DIR"), row)
                self.assertTrue(repo.startswith("https://"), row)
                self.assertTrue(ref and pinfile, row)
        # 0.30.0 adds DeepSelect (CMakeLists.txt:1575), through DEEPSELECT_SRC_DIR.
        self.assertEqual(counts["0.28.0"], 9)
        self.assertEqual(counts["0.30.0"], 10)
        self.assertNotRegex(SCRIPT, r'(?m)^  "[^"]+\|[A-Z_]+_SRC_DIR\|', "the table lives in the records")

    def test_the_published_wheelhouse_drops_the_pypi_vllm_wheel(self) -> None:
        body = _executed(SCRIPT)
        self.assertIn('rm -f "$dist"/wheels/vllm-*.whl', body)


if __name__ == "__main__":
    unittest.main()
