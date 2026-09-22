"""scripts/build-vllm.sh: fetches retry 3 times, fail loudly, and stay pinned.

devai-vllm is compiled from source on the host with the HyperQwen patch
series applied (operator decision 2026-09-20, same shape as devai-ollama: no
upstream image, not even as a build input). The script downloads a lot --
the vLLM sdist, nine external kernel repositories vLLM's CMake would
otherwise clone DURING CONFIGURE with no retry, HyperQwen itself, and the
wheels -- and the operator's rule for anything that downloads is: try 3
times, then bail out with an error, never a quiet success.

Beyond retries, three things here can silently produce a WRONG build, and
each is pinned below:

  * a source directory left by an older run at a different revision,
  * an external pinned to a revision the vLLM source tree no longer names
    (the pin table is only right for the release it was written for),
  * a patch that does not apply. HyperQwen applies its series with
    `--fuzz 0`; a fuzzy or skipped patch would yield a vLLM that starts fine
    and merely lacks the behaviour the image exists for.

The script is SOURCED (it guards `main` behind a BASH_SOURCE check), so its
real functions run against a fake git. No network, no compiler.
"""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "build-vllm.sh"

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

_SHA = "f3e1a4f74c99145c0717709860bf765de1703779"


class BuildVllmScriptTest(unittest.TestCase):
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

    def _sh(self, body: str, **env: str) -> subprocess.CompletedProcess:
        full = dict(os.environ, PATH=self.path, CALLS=str(self.calls),
                    FETCH_RETRY_DELAY="0", **env)
        return subprocess.run(
            ["bash", "-c", f'source "{SCRIPT}"; {body}'],
            env=full, capture_output=True, text=True, timeout=60)

    def _fetches(self) -> int:
        return len(self.calls.read_text().split())

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

    # ── pins ─────────────────────────────────────────────────────────────
    def _tree_naming_every_pin(self) -> Path:
        src = self.tmp / "vllm-src"
        r = self._sh('for row in "${EXTERNALS[@]}"; do IFS="|" read -r _ _ _ ref file _ <<< "$row"; '
                     f'mkdir -p "{src}/$(dirname "$file")"; echo "GIT_TAG $ref" >> "{src}/$file"; done')
        self.assertEqual(r.returncode, 0, r.stderr)
        return src

    def test_pins_named_by_the_source_tree_pass(self) -> None:
        src = self._tree_naming_every_pin()
        r = self._sh(f'check_pins "{src}"')
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_a_drifted_pin_is_refused(self) -> None:
        src = self._tree_naming_every_pin()
        cmake = src / "cmake" / "external_projects" / "vllm_flash_attn.cmake"
        cmake.write_text("GIT_TAG 0000000000000000000000000000000000000000\n")
        r = self._sh(f'check_pins "{src}"')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("flash-attention", r.stderr)
        self.assertIn(_SHA, r.stderr)

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

    def _apply(self, root: Path) -> subprocess.CompletedProcess:
        return self._sh(f'apply_patches "{root}/src"', VLLM_BUILD_ROOT=str(root),
                        DEVAI_PATCH_DIR=str(root / "devai-patches"))

    def test_both_series_apply_in_order_and_the_backport_is_skipped(self) -> None:
        root = self._patch_fixture(target_text="line one\nline two\nline three\n")
        r = self._apply(root)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertEqual((root / "src" / "vllm" / "mod.py").read_text(),
                         "line one\nline 2\nline 3\n")
        self.assertEqual((root / "src" / "PATCHES.applied").read_text().split(),
                         ["hyperqwen/good.patch", "devai/ours.patch"])

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
        r = self._apply(root)
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

    # ── the script's own safety net ──────────────────────────────────────
    def test_missing_cuda_toolkit_is_a_clear_error(self) -> None:
        r = self._sh("preflight", CUDA_HOME=str(self.tmp / "no-cuda-here"))
        self.assertNotEqual(r.returncode, 0)
        self.assertTrue("nvcc" in r.stderr or "not found on PATH" in r.stderr, r.stderr)


if __name__ == "__main__":
    unittest.main()
