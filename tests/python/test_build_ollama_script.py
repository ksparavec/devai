"""scripts/build-ollama.sh: the network steps fail loudly and retry 3 times.

The slim devai-ollama image is compiled from source on the host. Its build
script fetches the Ollama and llama.cpp sources and the Go modules, and the
operator's rule for anything that downloads is: try 3 times, then bail out
with an error -- never a quiet success. Both failures that rule answers
happened on 2026-09-19: a pull target ending in `|| true`, and a hand-written
`hf download` that fetched nothing and exited 0.

The script is SOURCED here (it guards `main` behind a BASH_SOURCE check), so
its real functions run against fake commands. No network, no compiler.
"""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "build-ollama.sh"

# Fake git. `clone` logs a call, leaves a half-written destination behind
# (as an interrupted clone does) and fails while the call count is
# <= $FAIL_FIRST; afterwards it creates a plausible checkout. `describe`
# answers with $FAKE_TAG.
_FAKE_GIT = r"""#!/usr/bin/env bash
case "$1" in
  clone)
    dest="${@: -1}"
    echo clone >> "$CALLS"
    n=$(wc -l < "$CALLS")
    mkdir -p "$dest"; echo partial > "$dest/half-written"
    [ "$n" -le "${FAIL_FIRST:-0}" ] && exit 128
    mkdir -p "$dest/.git"; exit 0;;
  -C) shift 2; [ "$1" = describe ] && { echo "${FAKE_TAG:-}"; exit 0; };;
esac
exit 0
"""


class BuildOllamaScriptTest(unittest.TestCase):
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

    def _clones(self) -> int:
        return len(self.calls.read_text().split())

    # ── retry3 ──────────────────────────────────────────────────────────────
    def test_retry3_runs_a_succeeding_command_once(self) -> None:
        r = self._sh(f'retry3 "step" sh -c \'echo x >> "{self.calls}"\'')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self._clones(), 1)

    def test_retry3_gives_up_after_exactly_three_attempts_with_an_error(self) -> None:
        r = self._sh(f'retry3 "fetching thing" sh -c \'echo x >> "{self.calls}"; exit 1\'')
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(self._clones(), 3)
        self.assertIn("fetching thing failed after 3 attempts", r.stderr)

    # ── fetch_source ────────────────────────────────────────────────────────
    def test_transient_clone_failure_is_retried_and_recovers(self) -> None:
        dest = self.tmp / "src"
        r = self._sh(f'fetch_source "Ollama" https://x/y.git v1.2.3 "{dest}"',
                     FAIL_FIRST="2", FAKE_TAG="v1.2.3")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self._clones(), 3)
        self.assertTrue((dest / ".git").is_dir())

    def test_failed_clone_leaves_nothing_that_looks_complete(self) -> None:
        dest = self.tmp / "src"
        r = self._sh(f'fetch_source "Ollama" https://x/y.git v1.2.3 "{dest}"',
                     FAIL_FIRST="99", FAKE_TAG="v1.2.3")
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(self._clones(), 3)
        self.assertFalse(dest.exists(),
                         "a failed fetch must not leave the destination behind")
        self.assertFalse(Path(f"{dest}.partial").exists(),
                         "the half-written clone must be cleaned up")

    def test_present_source_is_not_fetched_again(self) -> None:
        dest = self.tmp / "src"
        (dest / ".git").mkdir(parents=True)
        r = self._sh(f'fetch_source "Ollama" https://x/y.git v1.2.3 "{dest}"',
                     FAKE_TAG="v1.2.3")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self._clones(), 0)

    def test_source_at_the_wrong_tag_is_an_error(self) -> None:
        # A directory left by an older build must not be compiled as if it
        # were the requested version.
        dest = self.tmp / "src"
        (dest / ".git").mkdir(parents=True)
        r = self._sh(f'fetch_source "Ollama" https://x/y.git v1.2.3 "{dest}"',
                     FAKE_TAG="v0.9.0")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("v0.9.0", r.stderr)
        self.assertIn("v1.2.3", r.stderr)

    # ── the script's own safety net ─────────────────────────────────────────
    def test_missing_cuda_toolkit_is_a_clear_error(self) -> None:
        r = self._sh("preflight", CUDA_HOME=str(self.tmp / "no-cuda-here"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("nvcc", r.stderr)


if __name__ == "__main__":
    unittest.main()
