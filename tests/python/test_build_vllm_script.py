"""scripts/build-vllm.sh: fetches retry 3 times, fail loudly, and stay pinned.

devai-vllm is compiled from source on the host with the HyperQwen patch
series applied (operator decision 2026-09-20, same shape as devai-ollama: no
upstream image, not even as a build input). The script downloads a lot --
the vLLM sdist, the external kernel repositories vLLM's CMake would
otherwise clone DURING CONFIGURE with no retry, HyperQwen itself, and the
wheels -- and the operator's rule for anything that downloads is: try 3
times, then bail out with an error, never a quiet success.

Since platform-rebuild Phase 4 step 4v what is built is a BUILD RECORD,
deploy/vllm-builds/<version>.sh, chosen by VLLM_VERSION (default 0.30.0, E1).
Beside each record lie its runtime set (<version>.in, which the script
writes) and that set's hash lock (<version>.lock, `make vllm-lock`).

Beyond retries, several things here can silently produce a WRONG build, and
each is pinned below:

  * a source directory left by an older run at a different revision,
  * an external pinned to a revision, or a repository, the vLLM source tree
    no longer names (a record is only right for the release it was written
    for; DeepGEMM moved to a vllm-project fork in 0.30.0),
  * a patch that does not apply. HyperQwen applies its series with
    `--fuzz 0`; a fuzzy or skipped patch would yield a vLLM that starts fine
    and merely lacks the behaviour the image exists for,
  * a runtime set other than the reviewed lock (a stale .in, unhashed
    downloads, a vllm or flashinfer-cubin line).

The script is SOURCED (it guards `main` behind a BASH_SOURCE check), so its
real functions run against a fake git, fake records and fake trees. No
network, no compiler.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "build-vllm.sh"
RECORDS = REPO_ROOT / "deploy" / "vllm-builds"
MAKEFILE = (REPO_ROOT / "Makefile").read_text()

# Fake git. `fetch` logs a call, leaves a half-written tree behind (as an
# interrupted fetch does) and fails while the call count is <= $FAIL_FIRST.
# Everything else (init, remote, checkout, submodule) succeeds.
_FAKE_GIT = r"""#!/usr/bin/env bash
case "$1" in
  fetch)
    echo fetch >> "$CALLS"
    n=$(wc -l < "$CALLS")
    echo partial > half-written
    [ "$n" -le "${FAIL_FIRST:-0}" ] && exit 128
    exit 0;;
esac
exit 0
"""

# Fake PYTHON for fetch_wheels: logs its argv, one line per call.
_FAKE_PYTHON = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >> "$PYCALLS"
exit 0
"""

_SHA = "f3e1a4f74c99145c0717709860bf765de1703779"
_HQ = "0123456789abcdef0123456789abcdef01234567"

# What build-vllm.sh hardcoded before 4v: the 0.28.0 record must reproduce it.
_PRE_4V_SDIST = "ac96dd0ec5be9c13f2aa4bfb50498c4727110406f6e4980b58a4097e4d18634a"
_PRE_4V_HYPERQWEN = "c0c81bbbbf91f11b54af7b95f49bd6d1570c2ba0"
_PRE_4V_EXTERNALS = [
    "cutlass|VLLM_CUTLASS_SRC_DIR|https://github.com/nvidia/cutlass.git|v4.4.2|CMakeLists.txt",
    "flash-attention|VLLM_FLASH_ATTN_SRC_DIR|https://github.com/vllm-project/flash-attention.git|f3e1a4f74c99145c0717709860bf765de1703779|cmake/external_projects/vllm_flash_attn.cmake",
    "FlashMLA|FLASH_MLA_SRC_DIR|https://github.com/vllm-project/FlashMLA|a8f794d1251cbfd88a5011445dd5582289c727e4|cmake/external_projects/flashmla.cmake",
    "MSA|FMHA_SM100_SRC_DIR|https://github.com/vllm-project/MSA.git|087c161814d4d9c735b46c21212a09e5f8eb92fa|cmake/external_projects/fmha_sm100.cmake",
    "tml-fa4|TML_FA4_SRC_DIR|https://github.com/vllm-project/tml-fa4.git|b206834606ed5b5f21f8eed6b0683f528ea9cf7d|cmake/external_projects/tml_fa4.cmake",
    "FlashKDA|FLASH_KDA_SRC_DIR|https://github.com/vllm-project/FlashKDA.git|053de1b716ef3255873e02d2d28f4adf09951978|cmake/external_projects/flashkda.cmake",
    "qutlass|QUTLASS_SRC_DIR|https://github.com/IST-DASLab/qutlass.git|e74319e3405ce6d71965732880f5dc1f52371f64|cmake/external_projects/qutlass.cmake",
    "DeepGEMM|DEEPGEMM_SRC_DIR|https://github.com/deepseek-ai/DeepGEMM.git|8b1392b978f5a03c828dd1711090d7fb50958b8a|cmake/external_projects/deepgemm.cmake",
    "triton|TRITON_KERNELS_SRC_DIR|https://github.com/triton-lang/triton.git|v3.5.1|cmake/external_projects/triton_kernels.cmake|python/triton_kernels/triton_kernels",
]


def _record(*, sdist: str = "0" * 64, hq: str = _HQ, skip: str = "dflash2-backport.patch",
            devai: str = "", externals: str = '"x|X_SRC_DIR|https://x/x.git|v1|CMakeLists.txt"',
            torch: str = "2.13.0") -> str:
    return (f"VLLM_SDIST_SHA256={sdist}\nVLLM_TORCH_PIN={torch}\n"
            f"HYPERQWEN_COMMIT={hq}\nHYPERQWEN_SKIP=({skip})\n"
            f"DEVAI_PATCHES={devai}\nEXTERNALS=({externals})\n")


class _ScriptCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.calls = self.tmp / "calls"
        self.calls.write_text("")
        bindir = self.tmp / "bin"
        bindir.mkdir()
        git = bindir / "git"
        git.write_text(_FAKE_GIT)
        git.chmod(git.stat().st_mode | stat.S_IEXEC)
        self.path = f"{bindir}:{os.environ['PATH']}"
        self.records = self.tmp / "records"
        self.records.mkdir()

    def _write_record(self, version: str, text: str) -> None:
        (self.records / f"{version}.sh").write_text(text)

    def _sh(self, body: str, **env: str) -> subprocess.CompletedProcess:
        full = dict(os.environ, PATH=self.path, CALLS=str(self.calls),
                    FETCH_RETRY_DELAY="0", **env)
        return subprocess.run(
            ["bash", "-c", f'source "{SCRIPT}"; {body}'],
            env=full, capture_output=True, text=True, timeout=60)

    def _fetches(self) -> int:
        return len(self.calls.read_text().split())


class BuildVllmScriptTest(_ScriptCase):
    # ── retry3 ───────────────────────────────────────────────────────────
    def test_retry3_gives_up_after_exactly_three_attempts_with_an_error(self) -> None:
        r = self._sh(f'retry3 "fetching thing" sh -c \'echo x >> "{self.calls}"; exit 1\'')
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(self._fetches(), 3)
        self.assertIn("fetching thing failed after 3 attempts", r.stderr)

    # ── fetch_source ─────────────────────────────────────────────────────
    def test_transient_fetch_failure_is_retried_and_recovers(self) -> None:
        dest = self.tmp / "ext"
        r = self._sh(f'fetch_source "flash-attention" https://x/y.git {_SHA} "{dest}"',
                     FAIL_FIRST="2")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self._fetches(), 3)
        self.assertEqual((dest / ".devai-ref").read_text().strip(), _SHA)

    def test_failed_fetch_leaves_nothing_that_looks_complete(self) -> None:
        dest = self.tmp / "ext"
        r = self._sh(f'fetch_source "flash-attention" https://x/y.git {_SHA} "{dest}"',
                     FAIL_FIRST="99")
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(self._fetches(), 3)
        self.assertFalse(dest.exists())
        self.assertFalse(Path(f"{dest}.partial").exists())

    def test_present_source_is_not_fetched_again(self) -> None:
        dest = self.tmp / "ext"
        dest.mkdir()
        (dest / ".devai-ref").write_text(_SHA + "\n")
        r = self._sh(f'fetch_source "flash-attention" https://x/y.git {_SHA} "{dest}"')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self._fetches(), 0)

    def test_source_at_another_revision_is_an_error(self) -> None:
        dest = self.tmp / "ext"
        dest.mkdir()
        (dest / ".devai-ref").write_text("v4.3.0\n")
        r = self._sh(f'fetch_source "cutlass" https://x/y.git v4.4.2 "{dest}"')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("v4.3.0", r.stderr)
        self.assertIn("v4.4.2", r.stderr)

    def test_directory_without_a_completion_marker_is_refetched(self) -> None:
        # An interrupted older run, or a hand-made directory: not trusted.
        dest = self.tmp / "ext"
        dest.mkdir()
        (dest / "stale").write_text("x")
        r = self._sh(f'fetch_source "cutlass" https://x/y.git v4.4.2 "{dest}"')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self._fetches(), 1)
        self.assertFalse((dest / "stale").exists())

    # ── pins and repositories, per record ────────────────────────────────
    def _tree_naming_every_pin(self, version: str) -> Path:
        src = self.tmp / f"vllm-src-{version}"
        r = self._sh('for row in "${EXTERNALS[@]}"; do IFS="|" read -r _ _ repo ref file _ <<< "$row"; '
                     f'mkdir -p "{src}/$(dirname "$file")"; '
                     f'printf "GIT_REPOSITORY %s\\nGIT_TAG %s\\n" "$repo" "$ref" >> "{src}/$file"; done',
                     VLLM_VERSION=version)
        self.assertEqual(r.returncode, 0, r.stderr)
        return src

    def test_pins_named_by_the_source_tree_pass_for_every_record(self) -> None:
        for version in ("0.28.0", "0.30.0"):
            with self.subTest(version=version):
                src = self._tree_naming_every_pin(version)
                r = self._sh(f'check_pins "{src}"', VLLM_VERSION=version)
                self.assertEqual(r.returncode, 0, r.stderr)

    def test_a_drifted_pin_is_refused(self) -> None:
        src = self._tree_naming_every_pin("0.28.0")
        cmake = src / "cmake" / "external_projects" / "vllm_flash_attn.cmake"
        cmake.write_text("GIT_REPOSITORY https://github.com/vllm-project/flash-attention.git\n"
                         "GIT_TAG 0000000000000000000000000000000000000000\n")
        r = self._sh(f'check_pins "{src}"', VLLM_VERSION="0.28.0")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("flash-attention", r.stderr)
        self.assertIn(_SHA, r.stderr)

    def test_a_moved_repository_is_refused_even_at_the_same_pin(self) -> None:
        # 0.30.0 moved DeepGEMM to a vllm-project fork (vllm#56876): a record
        # still naming deepseek-ai's repository must not build.
        src = self._tree_naming_every_pin("0.28.0")
        cmake = src / "cmake" / "external_projects" / "deepgemm.cmake"
        cmake.write_text(cmake.read_text().replace("deepseek-ai/DeepGEMM", "vllm-project/DeepGEMM"))
        r = self._sh(f'check_pins "{src}"', VLLM_VERSION="0.28.0")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("DeepGEMM repository https://github.com/deepseek-ai/DeepGEMM.git", r.stderr)

    # ── patches ──────────────────────────────────────────────────────────
    def _patch_fixture(self, *, target_text: str) -> Path:
        root = self.tmp / "build"
        hq = root / "HyperQwen" / "patches"
        hq.mkdir(parents=True)
        (hq / "series").write_text("# order matters\ndflash2-backport.patch\n\ngood.patch  # note\n")
        (hq / "dflash2-backport.patch").write_text("this is not even a patch\n")
        (hq / "good.patch").write_text(
            "--- a/mod.py\n+++ b/mod.py\n@@ -1,3 +1,3 @@\n line one\n-line two\n+line 2\n line three\n")
        # devai's own series comes second and is written against the tree the
        # HyperQwen series leaves behind ("line 2", not "line two").
        ours = root / "devai-patches"
        ours.mkdir()
        (ours / "series").write_text("ours.patch\n")
        (ours / "ours.patch").write_text(
            "--- a/mod.py\n+++ b/mod.py\n@@ -1,3 +1,3 @@\n line one\n line 2\n-line three\n+line 3\n")
        src = root / "src"
        (src / "vllm").mkdir(parents=True)
        (src / "vllm" / "mod.py").write_text(target_text)
        return root

    def _apply(self, root: Path, *, record: str | None = None,
               devai: bool = True) -> subprocess.CompletedProcess:
        self._write_record("9.9.9", record if record is not None else _record())
        env = dict(VLLM_VERSION="9.9.9", VLLM_RECORD_DIR=str(self.records),
                   VLLM_BUILD_ROOT=str(root))
        if devai:
            env["DEVAI_PATCH_DIR"] = str(root / "devai-patches")
        return self._sh(f'apply_patches "{root}/src"', **env)

    def test_both_series_apply_in_order_and_the_recorded_skip_is_skipped(self) -> None:
        root = self._patch_fixture(target_text="line one\nline two\nline three\n")
        r = self._apply(root)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertEqual((root / "src" / "vllm" / "mod.py").read_text(),
                         "line one\nline 2\nline 3\n")
        self.assertEqual((root / "src" / "PATCHES.applied").read_text().split(),
                         ["hyperqwen/good.patch", "devai/ours.patch"])

    def test_without_a_recorded_skip_the_whole_series_is_applied(self) -> None:
        # 0.30.0's record skips nothing: a patch in the series that does not
        # apply is then an error, never silently passed over.
        root = self._patch_fixture(target_text="line one\nline two\nline three\n")
        r = self._apply(root, record=_record(skip=""))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("hyperqwen/dflash2-backport.patch did not apply", r.stderr)

    def test_a_skip_the_series_does_not_list_is_an_error(self) -> None:
        root = self._patch_fixture(target_text="line one\nline two\nline three\n")
        r = self._apply(root, record=_record(skip="gone.patch"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("skips gone.patch", r.stderr)

    def test_a_record_without_hyperqwen_applies_only_devais_series(self) -> None:
        root = self._patch_fixture(target_text="line one\nline 2\nline three\n")
        r = self._apply(root, record=_record(hq="", skip=""))
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertEqual((root / "src" / "PATCHES.applied").read_text().split(), ["devai/ours.patch"])

    def test_a_record_with_no_series_at_all_applies_nothing(self) -> None:
        # Phase 23's stock 0.31.0 record (I29): no HyperQwen, no devai series.
        root = self._patch_fixture(target_text="line one\nline two\nline three\n")
        r = self._apply(root, record=_record(hq="", skip=""), devai=False)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertEqual((root / "src" / "PATCHES.applied").read_text(), "")
        self.assertEqual((root / "src" / "vllm" / "mod.py").read_text(),
                         "line one\nline two\nline three\n")

    def test_a_record_without_devais_series_applies_only_hyperqwens(self) -> None:
        root = self._patch_fixture(target_text="line one\nline two\nline three\n")
        r = self._apply(root, devai=False)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertEqual((root / "src" / "PATCHES.applied").read_text().split(), ["hyperqwen/good.patch"])

    def test_a_patch_that_needs_fuzz_is_an_error(self) -> None:
        # Context line differs -> GNU patch would still apply it with its
        # default fuzz of 2. HyperQwen forbids that, and so do we.
        root = self._patch_fixture(target_text="line ONE\nline two\nline three\n")
        r = self._apply(root)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("good.patch", r.stderr)

    def test_a_devai_patch_that_does_not_apply_is_an_error(self) -> None:
        root = self._patch_fixture(target_text="line one\nline two\nline three\n")
        (root / "devai-patches" / "ours.patch").write_text(
            "--- a/mod.py\n+++ b/mod.py\n@@ -1,3 +1,3 @@\n line one\n line 2\n-line THREE\n+line 3\n")
        r = self._apply(root)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("devai/ours.patch", r.stderr)

    def test_an_empty_hyperqwen_series_is_an_error(self) -> None:
        root = self._patch_fixture(target_text="x\n")
        (root / "HyperQwen" / "patches" / "series").write_text("# nothing\n")
        r = self._apply(root, record=_record(skip=""))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no patch was applied", r.stderr)

    def test_a_missing_devai_series_file_is_an_error(self) -> None:
        root = self._patch_fixture(target_text="line one\nline two\nline three\n")
        (root / "devai-patches" / "series").unlink()
        r = self._apply(root)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no series file", r.stderr)

    # ── devai's own patches, as shipped ──────────────────────────────────
    def test_every_shipped_series_entry_exists_and_names_its_target(self) -> None:
        pdir = REPO_ROOT / "deploy" / "vllm-patches"
        names = [ln.split("#")[0].strip() for ln in (pdir / "series").read_text().splitlines()]
        names = [n for n in names if n]
        self.assertIn("qwen3_5-mtp-share-vocab-at-init.patch", names)
        for n in names:
            body = (pdir / n).read_text()
            self.assertRegex(body, r"(?m)^--- a/\S+\n\+\+\+ b/\S+", n)

    def test_the_mtp_patch_keeps_a_kill_switch_and_the_stock_path(self) -> None:
        body = (REPO_ROOT / "deploy" / "vllm-patches"
                / "qwen3_5-mtp-share-vocab-at-init.patch").read_text()
        self.assertIn("DEVAI_MTP_SHARE_VOCAB_AT_INIT", body)
        self.assertIn("get_pp_group().world_size == 1", body)
        self.assertIn("+            self.embed_tokens = VocabParallelEmbedding(", body)

    def test_the_mtp_patch_names_both_drafter_sharing_paths(self) -> None:
        # Model Runner V2 is the default since 0.29 (vllm#53183) and shares
        # the drafter's vocabulary layers in its own place; the patch's
        # comment, and the code comment it adds, name both runners' paths.
        body = (REPO_ROOT / "deploy" / "vllm-patches"
                / "qwen3_5-mtp-share-vocab-at-init.patch").read_text()
        added = "\n".join(ln for ln in body.splitlines() if ln.startswith("+"))
        for text in (body.split("\n--- a/", 1)[0], added):
            self.assertIn("llm_base_proposer.py", text)
            self.assertIn("_maybe_share_embeddings", text)
            self.assertIn("v1/worker/gpu/spec_decode/eagle/utils.py", text)
            self.assertIn("load_eagle_model", text)
            self.assertIn("_should_share", text)

    # ── the script's own safety net ──────────────────────────────────────
    def test_missing_cuda_toolkit_is_a_clear_error(self) -> None:
        r = self._sh("preflight", CUDA_HOME=str(self.tmp / "no-cuda-here"))
        self.assertNotEqual(r.returncode, 0)
        self.assertTrue("nvcc" in r.stderr or "not found on PATH" in r.stderr, r.stderr)


class BuildRecordTest(_ScriptCase):
    """The records themselves, as the script loads them."""

    def _load(self, version: str, query: str, **env: str) -> subprocess.CompletedProcess:
        return self._sh(query, VLLM_VERSION=version, **env)

    def test_the_default_version_is_0_30_0(self) -> None:
        r = self._sh('echo "$VLLM_VERSION|$HYPERQWEN_COMMIT"')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "0.30.0|7af097b5ea3acfecd60aabbff5bc457e503b2dc5")

    def test_the_0_28_0_record_reproduces_the_pre_4v_build(self) -> None:
        r = self._load("0.28.0", 'printf "%s\\n" "$VLLM_SDIST_SHA256" "$HYPERQWEN_COMMIT" '
                       '"${HYPERQWEN_SKIP[*]}" "$VLLM_TORCH_PIN" "$DEVAI_PATCH_DIR" "${EXTERNALS[@]}"')
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.splitlines()
        self.assertEqual(lines[:5], [_PRE_4V_SDIST, _PRE_4V_HYPERQWEN, "dflash2-backport.patch",
                                     "2.13.0", str(REPO_ROOT / "deploy" / "vllm-patches")])
        self.assertEqual(lines[5:], _PRE_4V_EXTERNALS)

    def test_the_0_30_0_record(self) -> None:
        r = self._load("0.30.0", 'printf "%s\\n" "$VLLM_SDIST_SHA256" "$HYPERQWEN_COMMIT" '
                       '"${#HYPERQWEN_SKIP[@]}" "$VLLM_TORCH_PIN" "$DEVAI_PATCH_DIR" "${EXTERNALS[@]}"')
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.splitlines()
        # The plan's value, read from PyPI's JSON (2026-10-06).
        self.assertEqual(lines[0], "5f8f4e890c042ffa1c3e103f81c35e2d96f60a0a175ac43a4adbae7006bef62b")
        self.assertEqual(lines[1], "7af097b5ea3acfecd60aabbff5bc457e503b2dc5")
        self.assertEqual(lines[2], "0")
        self.assertEqual(lines[3], "2.13.0")
        self.assertEqual(lines[4], str(REPO_ROOT / "deploy" / "vllm-patches"))
        rows = {row.split("|")[0]: row.split("|") for row in lines[5:]}
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows["cutlass"][3], "v4.7.1")
        self.assertEqual(rows["DeepGEMM"][2], "https://github.com/vllm-project/DeepGEMM.git")
        self.assertEqual(rows["DeepSelect"][1], "DEEPSELECT_SRC_DIR")
        self.assertEqual(rows["triton"][5], "python/triton_kernels/triton_kernels")

    def test_every_external_has_a_source_dir_override_a_repository_and_a_pin_file(self) -> None:
        for record in sorted(RECORDS.glob("*.sh")):
            rows = re.findall(r'(?m)^  "([^"]+)"$', record.read_text())
            self.assertGreaterEqual(len(rows), 9, record.name)
            for row in rows:
                name, var, repo, ref, pinfile = row.split("|")[:5]
                self.assertTrue(var.endswith("_SRC_DIR"), row)
                self.assertTrue(repo.startswith("https://github.com/"), row)
                self.assertTrue(ref and pinfile, row)

    def test_a_missing_record_is_an_error(self) -> None:
        r = self._sh("true", VLLM_VERSION="0.0.1")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no build record for vLLM 0.0.1", r.stderr)

    def test_an_invalid_record_is_refused(self) -> None:
        for bad, why in ((_record(sdist="abc"), "VLLM_SDIST_SHA256"),
                         (_record(hq="c0c81bb"), "full 40-hex commit"),
                         (_record(hq="", skip="x.patch"), "HYPERQWEN_SKIP without"),
                         (_record(externals=""), "no EXTERNALS"),
                         (_record(torch="2.13"), "VLLM_TORCH_PIN")):
            with self.subTest(why=why):
                self._write_record("9.9.9", bad)
                r = self._sh("true", VLLM_VERSION="9.9.9", VLLM_RECORD_DIR=str(self.records))
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(why, r.stderr)

    def test_the_record_wins_over_the_environment(self) -> None:
        r = self._sh('echo "$VLLM_SDIST_SHA256 $HYPERQWEN_COMMIT"', VLLM_VERSION="0.28.0",
                     VLLM_SDIST_SHA256="f" * 64, HYPERQWEN_COMMIT="e" * 40)
        self.assertEqual(r.stdout.split(), [_PRE_4V_SDIST, _PRE_4V_HYPERQWEN])

    def test_one_build_root_per_version(self) -> None:
        for version in ("0.28.0", "0.30.0"):
            r = self._sh('echo "$BUILD_ROOT"', VLLM_VERSION=version, HOME=str(self.tmp))
            self.assertEqual(r.stdout.strip(), f"{self.tmp}/.cache/devai/vllm-build/{version}")

    def test_a_record_without_hyperqwen_fetches_no_hyperqwen(self) -> None:
        root = self.tmp / "root"
        (root / "src").mkdir(parents=True)
        tarball = root / "src" / "vllm-9.9.9.tar.gz"
        tarball.write_bytes(b"not really a tarball")
        sha = hashlib.sha256(tarball.read_bytes()).hexdigest()
        self._write_record("9.9.9", _record(sdist=sha, hq="", skip=""))
        r = self._sh("fetch_record_sources", VLLM_VERSION="9.9.9",
                     VLLM_RECORD_DIR=str(self.records), VLLM_BUILD_ROOT=str(root))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self._fetches(), 0)
        self.assertFalse((root / "HyperQwen").exists())

    def test_the_sdist_is_fetched_as_a_file_from_pypi_and_retried(self) -> None:
        # Not `pip download`: pip prepares the sdist's metadata, which runs
        # its setup.py, and 0.30.0's needs setuptools_rust in the host
        # Python (the first 0.30.0 build died there, 2026-10-07).
        payload = b"sdist bytes"
        sha = hashlib.sha256(payload).hexdigest()
        curl = self.tmp / "bin" / "curl"
        curl.write_text(f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >> "{self.calls}"
n=$(wc -l < "{self.calls}")
out=""
while [ $# -gt 0 ]; do [ "$1" = -o ] && {{ out="$2"; shift; }}; shift; done
printf 'partial' > "$out"
[ "$n" -le 1 ] && exit 22
printf 'sdist bytes' > "$out"
""")
        curl.chmod(curl.stat().st_mode | stat.S_IEXEC)
        root = self.tmp / "root"
        self._write_record("9.9.9", _record(sdist=sha, hq="", skip=""))
        r = self._sh("fetch_sdist", VLLM_VERSION="9.9.9",
                     VLLM_RECORD_DIR=str(self.records), VLLM_BUILD_ROOT=str(root))
        self.assertEqual(r.returncode, 0, r.stderr)
        calls = self.calls.read_text().splitlines()
        self.assertEqual(len(calls), 2)
        self.assertIn("https://files.pythonhosted.org/packages/source/v/vllm/vllm-9.9.9.tar.gz",
                      calls[-1])
        self.assertEqual((root / "src" / "vllm-9.9.9.tar.gz").read_bytes(), payload)
        self.assertEqual(sorted(p.name for p in (root / "src").iterdir()), ["vllm-9.9.9.tar.gz"])

    def test_a_sdist_with_another_hash_is_refused(self) -> None:
        root = self.tmp / "root"
        (root / "src").mkdir(parents=True)
        (root / "src" / "vllm-9.9.9.tar.gz").write_bytes(b"tampered")
        self._write_record("9.9.9", _record(sdist="1" * 64, hq="", skip=""))
        r = self._sh("fetch_sdist", VLLM_VERSION="9.9.9",
                     VLLM_RECORD_DIR=str(self.records), VLLM_BUILD_ROOT=str(root))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("expected " + "1" * 64, r.stderr)


class RuntimeSetTest(_ScriptCase):
    """<version>.in: the release's runtime set plus HyperQwen's, cleaned."""

    _COMMON = ("regex # comment\nnumpy\ntransformers >= 5.10.4  # why\n"
               "six>=1.16.0; python_version > '3.11' # transitive\n")
    _CUDA = ("# Common dependencies\n-r common.txt\n\ntorch==2.13.0\n"
             "--extra-index-url https://flashinfer.ai/whl/\n"
             "flashinfer-python==0.6.18.post1\nflashinfer-cubin==0.6.18.post1\n"
             "tokenspeed-mla==0.1.8; platform_system == \"Linux\"\n")
    _HQ_REQ = ("# header\nvllm==0.30.0\ntransformers==5.15.0\n"
               "--extra-index-url https://download.pytorch.org/whl/cu130\n"
               "flashinfer-cubin==0.6.18.post1\npandas==3.0.5\n")

    def _tree(self, *, cuda: str | None = None, hq_req: str | None = None) -> Path:
        root = self.tmp / "root"
        src = root / "src" / "vllm-9.9.9"
        (src / "requirements").mkdir(parents=True)
        (src / "requirements" / "common.txt").write_text(self._COMMON)
        (src / "requirements" / "cuda.txt").write_text(cuda if cuda is not None else self._CUDA)
        if hq_req is not None:
            (root / "HyperQwen" / "docker").mkdir(parents=True)
            (root / "HyperQwen" / "docker" / "requirements.txt").write_text(hq_req)
        return root

    def _runtime_in(self, root: Path, record: str) -> subprocess.CompletedProcess:
        self._write_record("9.9.9", record)
        return self._sh(f'runtime_in "{root}/src/vllm-9.9.9"', VLLM_VERSION="9.9.9",
                        VLLM_RECORD_DIR=str(self.records), VLLM_BUILD_ROOT=str(root))

    @staticmethod
    def _reqs(text: str) -> list[str]:
        return [ln for ln in text.splitlines() if ln and not ln.startswith("#")]

    def test_a_hyperqwen_record_gets_the_release_set_then_hyperqwens(self) -> None:
        r = self._runtime_in(self._tree(hq_req=self._HQ_REQ), _record())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self._reqs(r.stdout), [
            "regex", "numpy", "transformers >= 5.10.4", "six>=1.16.0; python_version > '3.11'",
            "torch==2.13.0", "flashinfer-python==0.6.18.post1",
            'tokenspeed-mla==0.1.8; platform_system == "Linux"',
            "transformers==5.15.0", "pandas==3.0.5"])
        self.assertIn(f"# HyperQwen {_HQ}: docker/requirements.txt", r.stdout)

    def test_no_vllm_no_cubin_no_index_line_ever(self) -> None:
        r = self._runtime_in(self._tree(hq_req=self._HQ_REQ), _record())
        reqs = self._reqs(r.stdout)
        self.assertFalse([ln for ln in reqs if re.match(r"(vllm|flashinfer[-_]cubin)\b", ln, re.I)])
        self.assertFalse([ln for ln in reqs if ln.startswith("-")])

    def test_a_record_without_hyperqwen_gets_the_release_set_alone(self) -> None:
        r = self._runtime_in(self._tree(hq_req=self._HQ_REQ), _record(hq="", skip=""))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("HyperQwen", r.stdout)
        self.assertNotIn("pandas==3.0.5", r.stdout)
        self.assertIn("torch==2.13.0", r.stdout)

    def test_a_release_torch_pin_other_than_the_records_is_refused(self) -> None:
        r = self._runtime_in(self._tree(hq_req=self._HQ_REQ), _record(torch="2.12.0"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("the build record pins torch==2.12.0", r.stderr)

    def test_a_release_without_a_torch_pin_gets_the_records(self) -> None:
        r = self._runtime_in(self._tree(cuda="-r common.txt\nnumba\n", hq_req=None),
                             _record(hq="", skip=""))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self._reqs(r.stdout)[-1], "torch==2.13.0")

    def test_an_unknown_option_line_is_an_error_not_a_silent_drop(self) -> None:
        r = self._runtime_in(self._tree(cuda="-c constraints.txt\ntorch==2.13.0\n", hq_req=None),
                             _record(hq="", skip=""))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("unexpected option line", r.stderr)

    def test_a_stale_runtime_set_is_refused_by_a_build(self) -> None:
        root = self._tree(hq_req=self._HQ_REQ)
        self._write_record("9.9.9", _record())
        (self.records / "9.9.9.in").write_text("numpy\n")
        (self.records / "9.9.9.lock").write_text("numpy==1 --hash=sha256:00\n")
        r = self._sh(f'check_runtime_in "{root}/src/vllm-9.9.9"', VLLM_VERSION="9.9.9",
                     VLLM_RECORD_DIR=str(self.records), VLLM_BUILD_ROOT=str(root))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("make vllm-lock VLLM_VERSION=9.9.9", r.stderr)

    def test_write_in_rewrites_the_runtime_set_from_the_records_sources(self) -> None:
        root = self._tree(hq_req=self._HQ_REQ)
        tarball = root / "src" / "vllm-9.9.9.tar.gz"
        with tarfile.open(tarball, "w:gz") as tar:
            tar.add(root / "src" / "vllm-9.9.9", arcname="vllm-9.9.9")
        sha = hashlib.sha256(tarball.read_bytes()).hexdigest()
        self._write_record("9.9.9", _record(sdist=sha))
        hq = root / "HyperQwen"
        (hq / ".devai-ref").write_text(_HQ + "\n")
        r = self._sh("write_in", VLLM_VERSION="9.9.9", VLLM_RECORD_DIR=str(self.records),
                     VLLM_BUILD_ROOT=str(root))
        self.assertEqual(r.returncode, 0, r.stderr)
        written = (self.records / "9.9.9.in").read_text()
        self.assertIn("pandas==3.0.5", written)
        self.assertFalse((self.records / "9.9.9.in.tmp").exists())
        # ... and a build accepts exactly that file.
        (self.records / "9.9.9.lock").write_text("x\n")
        r = self._sh(f'check_runtime_in "{root}/src/vllm-9.9.9"', VLLM_VERSION="9.9.9",
                     VLLM_RECORD_DIR=str(self.records), VLLM_BUILD_ROOT=str(root))
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_fetch_wheels_downloads_the_lock_with_require_hashes(self) -> None:
        fake = self.tmp / "bin" / "fakepython"
        fake.write_text(_FAKE_PYTHON)
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        pycalls = self.tmp / "pycalls"
        root = self.tmp / "root"
        r = self._sh("fetch_wheels", VLLM_VERSION="0.30.0", VLLM_BUILD_ROOT=str(root),
                     PYTHON=str(fake), PYCALLS=str(pycalls))
        self.assertEqual(r.returncode, 0, r.stderr)
        runtime = pycalls.read_text().splitlines()[0]
        self.assertIn(f"-r {RECORDS}/0.30.0.lock --require-hashes", runtime)
        self.assertIn("--only-binary=:all:", runtime)
        self.assertNotIn("HyperQwen", runtime)


class WheelCheckTest(_ScriptCase):
    def _wheel(self, mtp: str) -> Path:
        whl = self.tmp / "vllm-9.9.9-cp313-cp313-linux_x86_64.whl"
        with zipfile.ZipFile(whl, "w") as z:
            z.writestr("vllm/_C.abi3.so", b"")
            z.writestr("vllm/model_executor/models/qwen3_5_mtp.py", mtp)
        return whl

    def _verify(self, mtp: str, record: str, devai_dir: str = "") -> subprocess.CompletedProcess:
        self._write_record("9.9.9", record)
        env = dict(VLLM_VERSION="9.9.9", VLLM_RECORD_DIR=str(self.records))
        if devai_dir:
            env["DEVAI_PATCH_DIR"] = devai_dir
        return self._sh(f'verify_wheel "{self._wheel(mtp)}"', **env)

    def test_each_series_the_record_names_must_be_in_the_wheel(self) -> None:
        both = "draft_vocab_ids shares_target_vocab"
        self.assertEqual(self._verify(both, _record(), "/x").returncode, 0)
        self.assertNotEqual(self._verify("shares_target_vocab", _record(), "/x").returncode, 0)
        self.assertNotEqual(self._verify("draft_vocab_ids", _record(), "/x").returncode, 0)

    def test_a_series_the_record_does_not_name_is_not_looked_for(self) -> None:
        stock = _record(hq="", skip="")
        self.assertEqual(self._verify("", stock).returncode, 0)
        self.assertEqual(self._verify("shares_target_vocab", stock, "/x").returncode, 0)


class CommittedRuntimeSetsTest(unittest.TestCase):
    """Every record keeps its runtime set and its hash lock beside it."""

    @staticmethod
    def _record_value(version: str, key: str) -> str:
        text = (RECORDS / f"{version}.sh").read_text()
        m = re.search(rf"(?m)^{key}=(\S*)$", text)
        return m.group(1) if m else ""

    def _versions(self) -> list[str]:
        versions = sorted(p.stem for p in RECORDS.glob("*.sh"))
        self.assertIn("0.28.0", versions)
        self.assertIn("0.30.0", versions)
        return versions

    def test_every_record_has_its_in_and_its_lock(self) -> None:
        for v in self._versions():
            self.assertTrue((RECORDS / f"{v}.in").is_file(), v)
            self.assertTrue((RECORDS / f"{v}.lock").is_file(), v)

    def test_every_in_is_the_release_set_plus_hyperqwens_and_nothing_else(self) -> None:
        for v in self._versions():
            text = (RECORDS / f"{v}.in").read_text()
            reqs = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
            with self.subTest(version=v):
                self.assertTrue(text.startswith(f"# vLLM {v} runtime set, written by scripts/build-vllm.sh"))
                self.assertFalse([ln for ln in reqs if ln.startswith("-")], "index or option line")
                self.assertFalse([ln for ln in reqs if re.match(r"(vllm|flashinfer[-_]cubin)\b", ln, re.I)])
                self.assertIn(f"torch=={self._record_value(v, 'VLLM_TORCH_PIN')}", reqs)
                hq = self._record_value(v, "HYPERQWEN_COMMIT")
                if hq:
                    self.assertIn(f"# HyperQwen {hq}: docker/requirements.txt,", text)
                else:
                    self.assertNotIn("# HyperQwen", text)

    def test_every_lock_pins_the_records_torch_and_flashinfer_and_hashes_everything(self) -> None:
        for v in self._versions():
            text = (RECORDS / f"{v}.lock").read_text()
            pins = re.findall(r"(?m)^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)", text)
            names = {re.sub(r"[-_.]+", "-", n).lower(): ver for n, ver in pins}
            with self.subTest(version=v):
                self.assertIn(f"make vllm-lock VLLM_VERSION={v}", text.split("\n", 3)[1])
                self.assertEqual(names.get("torch"), self._record_value(v, "VLLM_TORCH_PIN"))
                self.assertIn("flashinfer-python", names)
                self.assertNotIn("flashinfer-cubin", names)
                self.assertNotIn("vllm", names)
                # Every pinned requirement carries at least one hash.
                blocks = re.split(r"(?m)^(?=[A-Za-z0-9])", text)
                for block in blocks:
                    if re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*==", block):
                        self.assertIn("--hash=sha256:", block, block.split()[0])

    def test_the_0_30_0_lock_has_flashinfer_0_6_18_post1(self) -> None:
        # HyperQwen's pin for 0.30.0 and vLLM 0.30.0's own (plan 4v).
        self.assertRegex((RECORDS / "0.30.0.lock").read_text(),
                         r"(?m)^flashinfer-python==0\.6\.18\.post1 ")


class VllmLockTargetTest(unittest.TestCase):
    """`make vllm-lock VLLM_VERSION=<v>`: PyPI alone, hashes, the build's Python."""

    def _recipe(self) -> str:
        return MAKEFILE.split("\nvllm-lock:", 1)[1].split("\n\n", 1)[0]

    def test_documented_target(self) -> None:
        self.assertRegex(MAKEFILE, r"(?m)^vllm-lock: ## \S")
        self.assertRegex(MAKEFILE, r"(?m)^\.PHONY:.*\bvllm-lock\b")

    def test_command(self) -> None:
        recipe = self._recipe()
        self.assertIn("bash scripts/build-vllm.sh --write-in", recipe)
        for part in ("uv pip compile deploy/vllm-builds/$(VLLM_VERSION).in", "--no-config",
                     "--default-index https://pypi.org/simple", "--only-binary :all:",
                     "--python $(VLLM_PYTHON)", "--generate-hashes", "$(if $(UPGRADE),--upgrade,)",
                     '--custom-compile-command "make vllm-lock VLLM_VERSION=$(VLLM_VERSION)"',
                     "-o deploy/vllm-builds/$(VLLM_VERSION).lock"):
            self.assertIn(part, recipe)
        self.assertNotIn("extra-index", recipe)
        for var in ("UV_INDEX", "UV_INDEX_URL", "UV_DEFAULT_INDEX", "UV_EXTRA_INDEX_URL"):
            self.assertIn(f"-u {var} ", recipe)
        self.assertRegex(MAKEFILE, r"(?m)^VLLM_PYTHON \?= /usr/bin/python3$")

    def test_make_n_renders_it_for_a_version(self) -> None:
        r = subprocess.run(["make", "-n", "vllm-lock", "VLLM_VERSION=0.30.0"], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("VLLM_VERSION=0.30.0", r.stdout)
        self.assertIn("deploy/vllm-builds/0.30.0.in", r.stdout)
        self.assertIn("-o deploy/vllm-builds/0.30.0.lock", r.stdout)


if __name__ == "__main__":
    unittest.main()
