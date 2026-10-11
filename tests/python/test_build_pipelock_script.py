"""scripts/build-pipelock.sh: the pinned source, fetched loudly, or nothing.

devai-pipelock is compiled from source on the host (platform-rebuild
Phase 2), at the version that ran from ghcr.io before: tag v3.0.0, commit
9d2e37d. The fetch follows the operator's rule for anything that downloads
(3 attempts, then an error), and a tree at the wrong tag -- or a tag that
moved to another commit upstream -- is refused rather than built.

The script is SOURCED (it guards `main` behind a BASH_SOURCE check), so its
real functions run against a fake git. No network, no compiler.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "build-pipelock.sh"
COMMIT = "9d2e37d296359c38e599525a0424cdc189a812e1"

# Fake git. `clone` logs a call, leaves a half-written destination behind and
# fails while the call count is <= $FAIL_FIRST; afterwards it creates a
# plausible checkout. `-C <dir> describe` answers $FAKE_TAG, `-C <dir>
# rev-parse HEAD` answers $FAKE_COMMIT.
_FAKE_GIT = r"""#!/usr/bin/env bash
[ "$1" = -c ] && shift 2
case "$1" in
  clone)
    dest="${@: -1}"
    echo clone >> "$CALLS"
    n=$(wc -l < "$CALLS")
    mkdir -p "$dest"; echo partial > "$dest/half-written"
    [ "$n" -le "${FAIL_FIRST:-0}" ] && exit 128
    mkdir -p "$dest/.git"; exit 0;;
  -C) shift 2
      [ "$1" = describe ] && { echo "${FAKE_TAG:-}"; exit 0; }
      [ "$1" = rev-parse ] && { echo "${FAKE_COMMIT:-}"; exit 0; };;
esac
exit 0
"""


class BuildPipelockScriptTest(unittest.TestCase):
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
        self.dest = self.tmp / "src"

    def _fetch(self, **env: str) -> subprocess.CompletedProcess:
        full = dict(os.environ, PATH=self.path, CALLS=str(self.calls),
                    FETCH_RETRY_DELAY="0", **env)
        return subprocess.run(
            ["bash", "-c", f'source "{SCRIPT}"; fetch_source "{self.dest}"'],
            env=full, capture_output=True, text=True, timeout=60)

    def _clones(self) -> int:
        return len(self.calls.read_text().split())

    def test_pins_the_version_that_ran_from_ghcr(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn('PIPELOCK_VERSION="v3.0.0"', text)
        self.assertIn(f'PIPELOCK_COMMIT="{COMMIT}"', text)
        # Built as upstream builds its release.
        self.assertIn("CGO_ENABLED=0", text)
        self.assertIn("-tags enterprise", text)

    def test_transient_clone_failure_is_retried_and_recovers(self) -> None:
        r = self._fetch(FAIL_FIRST="2", FAKE_TAG="v3.0.0", FAKE_COMMIT=COMMIT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self._clones(), 3)
        self.assertTrue((self.dest / ".git").is_dir())

    def test_failed_clone_is_an_error_and_leaves_nothing(self) -> None:
        r = self._fetch(FAIL_FIRST="99", FAKE_TAG="v3.0.0", FAKE_COMMIT=COMMIT)
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(self._clones(), 3)
        self.assertIn("failed after 3 attempts", r.stderr)
        self.assertFalse(self.dest.exists())
        self.assertFalse(Path(f"{self.dest}.partial").exists())

    def test_present_source_is_not_fetched_again(self) -> None:
        (self.dest / ".git").mkdir(parents=True)
        r = self._fetch(FAKE_TAG="v3.0.0", FAKE_COMMIT=COMMIT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self._clones(), 0)

    def test_source_at_another_tag_is_refused(self) -> None:
        (self.dest / ".git").mkdir(parents=True)
        r = self._fetch(FAKE_TAG="v3.5.0", FAKE_COMMIT=COMMIT)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("v3.5.0", r.stderr)

    def test_a_tag_moved_upstream_is_refused(self) -> None:
        (self.dest / ".git").mkdir(parents=True)
        r = self._fetch(FAKE_TAG="v3.0.0", FAKE_COMMIT="0" * 40)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("the tag moved upstream", r.stderr)

    def test_make_targets(self) -> None:
        mk = (REPO_ROOT / "Makefile").read_text()
        for target in ("build-pipelock", "build-pipelock-dist", "build-pipelock-image", "build-apt-cache"):
            self.assertRegex(mk, rf"(?m)^{target}:.*## ", target)
        self.assertRegex(mk, r"(?m)^build-pipelock-dist:.*\n\tbash scripts/build-pipelock\.sh$")
        self.assertIsNotNone(re.search(r"run --rm .*localhost/devai-pipelock:latest", mk),
                             "pipelock-ca-init uses the image built here")


if __name__ == "__main__":
    unittest.main()
