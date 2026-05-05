#!/usr/bin/env python3
"""Sanity-check the pearls problem set by running each problem's
canonical solution through the same subprocess sandbox the bench
scorer uses.

Catches the cases that silently break a real bench run:

  - test asserts a different shape than the canonical solution returns
  - canonical solution exceeds the 10 s timeout under rlimit
  - test imports a module not in stdlib
  - prompt+solution+test combine into invalid Python (mismatched indent
    in a triple-quoted string, etc.)

Exit 0 on all-pass, 1 on any failure. Prints one line per problem.

Run from inside the lab container (nothing fancy required, just a
recent Python). Outside the container also works; the rlimit + alarm
mechanics are the same on any Linux host.
"""

from __future__ import annotations

import json
import resource  # noqa: F401  # imported by the wrapper template
import subprocess
import sys
import textwrap
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JSONL_PATH = HERE / "data" / "pearls_problems.jsonl"

# Match scripts/bench/tasks/humaneval.py limits exactly so a "passes
# here" verdict means "passes in the bench scorer".
_RUN_TIMEOUT_S = 10.0
_MEM_LIMIT_MB = 256


def _run_in_subprocess(program: str) -> tuple[bool, str, float]:
    """Exec ``program`` with rlimit + alarm. Returns ``(ok, err, secs)``.

    Mirrors scripts/bench/tasks/humaneval.py:_run_check_in_subprocess
    so a sanity check here is faithful to what the bench will see.
    """
    wrapper = textwrap.dedent(f"""
        import resource, signal, sys
        try:
            resource.setrlimit(
                resource.RLIMIT_AS,
                ({_MEM_LIMIT_MB * 1024 * 1024}, {_MEM_LIMIT_MB * 1024 * 1024}),
            )
        except (ValueError, OSError):
            pass
        def _timeout(signum, frame):
            raise TimeoutError("sample exceeded {int(_RUN_TIMEOUT_S)}s")
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm({int(_RUN_TIMEOUT_S)})
    """)
    full = wrapper + "\n" + program
    t0 = time.monotonic()
    try:
        r = subprocess.run(
            [sys.executable, "-c", full],
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_S + 2,
        )
    except subprocess.TimeoutExpired:
        return False, "subprocess timeout (outer)", time.monotonic() - t0
    secs = time.monotonic() - t0
    if r.returncode == 0:
        return True, "", secs
    return False, (r.stderr or r.stdout or "").strip()[-1500:], secs


def _build_program(problem: dict) -> str:
    """The exact form the bench scorer constructs:
    prompt + canonical_solution + test + check(<entry_point>)
    """
    return (
        problem["prompt"]
        + "\n"
        + problem["canonical_solution"]
        + "\n\n"
        + problem["test"]
        + f"\n\ncheck({problem['entry_point']})\n"
    )


def main() -> int:
    if not JSONL_PATH.is_file():
        print(f"ERROR: {JSONL_PATH} missing -- run _build_problems.py first",
              file=sys.stderr)
        return 1
    problems: list[dict] = []
    for line in JSONL_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        problems.append(json.loads(line))

    failures = 0
    for p in problems:
        prog = _build_program(p)
        ok, err, secs = _run_in_subprocess(prog)
        status = "PASS" if ok else "FAIL"
        print(f"  {status:4s}  {p['task_id']:50s}  {secs:5.2f}s")
        if not ok:
            failures += 1
            # Indent the captured stderr so it visually attaches.
            for line in err.splitlines():
                print(f"        | {line}")
    if failures:
        print(f"\n{failures}/{len(problems)} problems failed -- "
              f"fix _build_problems.py and rerun", file=sys.stderr)
        return 1
    print(f"\nall {len(problems)} problems pass canonical-solution check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
