"""`make pull-images` must fail loudly and retry before giving up.

The recipe used to be `podman pull "$img" || true`: a failed pull was
swallowed, the loop moved on, and make exited 0. A pull that did not happen
was indistinguishable from one that did -- the same silent-success shape as
the hand-written `hf download` that fetched nothing and reported OK on
2026-09-19.

Contract pinned here:
  - every pull is attempted up to 3 times;
  - an image that still fails after 3 attempts aborts the target, non-zero;
  - IMAGES="a b" limits the run to those images (upgrading one image must not
    require pulling every base and infrastructure image).

Behavioural, not textual: make is run for real against a FAKE container
runtime that fails on demand and counts its invocations. No network, no
podman, no images.
"""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_FAKE_RUNTIME = """#!/usr/bin/env bash
# args: pull <image>. Fails the first $FAIL_FIRST calls, then succeeds.
# The Makefile also calls the runtime for unrelated things while it is being
# parsed; only `pull` is under test.
[ "$1" = "pull" ] || exit 0
echo "$2" >> "$CALLS_FILE"
n=$(wc -l < "$CALLS_FILE")
[ "$n" -le "${FAIL_FIRST:-0}" ] && exit 1
exit 0
"""


class PullImagesTargetTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        self.calls = tmp / "calls"
        self.calls.write_text("")
        self.runtime = tmp / "fake-runtime"
        self.runtime.write_text(_FAKE_RUNTIME)
        self.runtime.chmod(self.runtime.stat().st_mode | stat.S_IEXEC)

    def _make(self, images: str, fail_first: int) -> subprocess.CompletedProcess:
        env = dict(os.environ, CALLS_FILE=str(self.calls),
                   FAIL_FIRST=str(fail_first))
        return subprocess.run(
            ["make", "--no-print-directory", "pull-images",
             f"IMAGES={images}", f"CONTAINER_RUNTIME={self.runtime}",
             "PULL_RETRY_DELAY=0"],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60)

    def _pulled(self) -> list[str]:
        return self.calls.read_text().split()

    def test_success_pulls_each_image_once(self) -> None:
        r = self._make("img/a:1 img/b:2", fail_first=0)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self._pulled(), ["img/a:1", "img/b:2"])

    def test_images_limits_the_run(self) -> None:
        # Nothing from BASE_IMAGES / CACHE_IMAGES may be touched.
        r = self._make("img/only:1", fail_first=0)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self._pulled(), ["img/only:1"])

    def test_transient_failure_is_retried_and_recovers(self) -> None:
        r = self._make("img/a:1", fail_first=2)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self._pulled(), ["img/a:1"] * 3)

    def test_gives_up_after_exactly_three_attempts_with_an_error(self) -> None:
        r = self._make("img/a:1", fail_first=99)
        self.assertNotEqual(r.returncode, 0,
                            "a pull that never succeeded must fail the target")
        self.assertEqual(self._pulled(), ["img/a:1"] * 3)
        self.assertIn("img/a:1", r.stderr)
        self.assertIn("3 attempts", r.stderr)

    def test_a_failed_image_stops_the_run(self) -> None:
        # Bail out: do not carry on to later images as if nothing happened.
        r = self._make("img/bad:1 img/never:2", fail_first=99)
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("img/never:2", self._pulled())


if __name__ == "__main__":
    unittest.main()
