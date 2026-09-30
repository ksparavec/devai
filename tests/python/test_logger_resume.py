"""The logger sidecar resumes each log where the file ends.

`podman logs --follow` without --since streams a container's WHOLE history.
deploy/logging.sh did that on every follower's first connect, and the logger
is recreated by every `make cache-up`, so each restart appended the full
history again: the persisted router log held 3,245 launch lines for 742
distinct launches (2026-09-27 re-analysis). The first connect now passes
--since <newest podman stamp in the file>.

Runs the script's real shell function with /bin/sh; stdlib unittest only.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = (REPO_ROOT / "deploy" / "logging.sh").read_text()
FUNC = re.search(r"(?ms)^last_podman_stamp\(\) \{.*?^\}", SCRIPT).group(0)


def _last_stamp(content: str | None) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "svc.log"
        if content is not None:
            f.write_text(content)
        r = subprocess.run(["sh", "-c", FUNC + '\nlast_podman_stamp "$1"', "sh", str(f)],
                           capture_output=True, text=True, check=True)
        return r.stdout.strip()


class LastStampTest(unittest.TestCase):
    def test_absent_file_gives_nothing(self) -> None:
        self.assertEqual(_last_stamp(None), "")

    def test_newest_podman_stamp_wins_over_trailing_logger_lines(self) -> None:
        log = ("2026-09-27T19:27:02+02:00 a\n"
               "2026-09-27T19:27:06+02:00 b\n"
               "[2026-09-27T17:30:00Z] [logger] follower for devai-router exited; retry in 5s\n")
        self.assertEqual(_last_stamp(log), "2026-09-27T19:27:06+02:00")

    def test_fractional_seconds_and_zulu(self) -> None:
        self.assertEqual(_last_stamp("2026-09-27T17:27:01.492048000Z x\n"),
                         "2026-09-27T17:27:01.492048000Z")

    def test_only_logger_lines_give_nothing(self) -> None:
        self.assertEqual(_last_stamp("[2026-09-27T17:30:00Z] [logger] starting\n"), "")


class FollowerWiringTest(unittest.TestCase):
    def test_first_connect_resumes_from_the_file(self) -> None:
        body = SCRIPT[SCRIPT.index("follow() {"):]
        self.assertIn('since="$(last_podman_stamp "$out")"', body)
        self.assertNotRegex(body, r'(?m)^\s*since=""\s*$')

    def test_since_is_passed_whenever_set(self) -> None:
        self.assertIn('if [ -n "$since" ]; then set -- --since "$since"', SCRIPT)


if __name__ == "__main__":
    unittest.main()
