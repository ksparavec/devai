"""inspect_ai task wrapper for the Programming Pearls problem set.

Mirrors scripts/bench/tasks/humaneval.py exactly: same Pass@1 scoring,
same subprocess sandbox (rlimit + signal.alarm), same completion
cleaner (think-strip + last fence), same Sample shape with ``test`` and
``entry_point`` rolled into metadata. The only differences are:

  - dataset source: a bundled JSONL of custom problems (Bentley's
    *Programming Pearls*, columns 1-15) loaded via ``MemoryDataset``
    instead of HuggingFace's ``hf_dataset``;
  - per-row metadata carries ``column`` and ``difficulty`` so a future
    consumer (the leaderboard report, the picker badge) can break
    pass@1 down by Bentley column or difficulty band.

The bench runner doesn't need to know about Bentley -- it just sees a
task that takes ``n`` and returns an inspect_ai ``Task``. The plug-in
path into the main bench is moving this file into
``scripts/bench/tasks/`` and wiring a branch in ``bench_runner.py``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate, system_message

PROBLEMS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "pearls_problems.jsonl"
)

SYSTEM_PROMPT = (
    "You are a Python coding assistant. Complete the function below. "
    "Output ONLY the function body (or the full function definition) -- "
    "no explanations, no markdown fences, no extra prose. Your output "
    "will be concatenated directly to the prompt and executed."
)

# Same limits as humaneval.py so a "passes" verdict means the same thing
# whether the task is humaneval, pearls, or any future Pass@1 task.
_RUN_TIMEOUT_S = 10.0
_MEM_LIMIT_MB = 256

# Identical regexes to the humaneval cleaner. The shared cleaning logic
# is duplicated rather than imported because the two task modules are
# allowed to drift independently (different problem styles may need
# different cleaners). When they re-converge we can lift this into
# ``_bench_core``.
_FENCE_BLOCK_RX = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)
_THINK_RX = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _clean_completion(text: str, entry_point: str = "") -> str:
    """Extract executable Python from a model completion.

    Identical strategy to ``scripts/bench/tasks/humaneval.py``:
    1. Strip ``<think>...</think>`` reasoning blocks.
    2. If fenced code blocks exist, return the body of the LAST one
       (last fence is empirically the model's final answer).
    3. Otherwise, if ``entry_point`` is known, slice from the first
       ``def <entry_point>(`` to end-of-string.
    4. Fall through: return the think-stripped text as-is.

    Why the same logic instead of an import: keeps each task module
    self-contained, so moving ``pearls.py`` into the main bench's
    ``tasks/`` later is a copy with no further plumbing.
    """
    if not text:
        return ""
    cleaned = _THINK_RX.sub("", text).strip()
    fences = _FENCE_BLOCK_RX.findall(cleaned)
    if fences:
        return fences[-1].rstrip("\n")
    if entry_point:
        m = re.search(
            rf"^def\s+{re.escape(entry_point)}\s*\(",
            cleaned,
            flags=re.MULTILINE,
        )
        if m:
            return cleaned[m.start():]
    return cleaned


def _record_to_sample(record: dict) -> Sample:
    """One JSONL row -> one inspect_ai Sample.

    The ``test`` and ``entry_point`` ride in metadata so the scorer can
    construct the test harness without re-reading the JSONL. ``column``
    and ``difficulty`` ride along too -- not consumed by the scorer
    today but available to any future per-column or per-difficulty
    breakdown without a schema migration.
    """
    return Sample(
        input=record["prompt"],
        target=record.get("canonical_solution") or "",
        metadata={
            "task_id": record.get("task_id", ""),
            "test": record.get("test", ""),
            "entry_point": record.get("entry_point", ""),
            "column": record.get("column"),
            "difficulty": record.get("difficulty", ""),
        },
    )


def _run_check_in_subprocess(program: str) -> tuple[bool, str]:
    """Exec ``program`` in a fresh Python process with rlimit + alarm.

    Identical mechanics to the humaneval scorer. Returns
    ``(passed, stderr_excerpt)``. The stderr excerpt is capped at 400
    chars so a verbose AssertionError doesn't balloon the cache row.
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
            raise TimeoutError("pearls sample exceeded {int(_RUN_TIMEOUT_S)}s")
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm({int(_RUN_TIMEOUT_S)})
    """)
    full = wrapper + "\n" + program
    try:
        r = subprocess.run(
            [sys.executable, "-c", full],
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_S + 2,
        )
    except subprocess.TimeoutExpired:
        return False, "subprocess timeout"
    if r.returncode == 0:
        return True, ""
    err = (r.stderr or r.stdout or "").strip()
    return False, err[-400:]


def _load_problems() -> list[dict]:
    """Read the bundled JSONL into a list of dicts. Keeping the loader
    in this module (rather than a shared helper) means the plug-in
    path later is a single-file move.
    """
    out: list[dict] = []
    for raw in PROBLEMS_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(json.loads(line))
    return out


@scorer(metrics=[accuracy(), stderr()])
def pearls_pass_at_1():
    """Pass@1 scorer. Concatenates prompt + completion + test, runs in
    a subprocess with limits, scores 1.0 on exit 0, 0.0 otherwise.
    """

    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata or {}
        test = meta.get("test", "")
        entry_point = meta.get("entry_point", "")
        if not test or not entry_point:
            return Score(
                value=0.0,
                answer="",
                explanation="missing test or entry_point",
            )

        completion = _clean_completion(
            state.output.completion or "", entry_point=entry_point
        )
        program = (
            state.input_text
            + "\n"
            + completion
            + "\n\n"
            + test
            + f"\n\ncheck({entry_point})\n"
        )
        passed, err = _run_check_in_subprocess(program)
        return Score(
            value=1.0 if passed else 0.0,
            answer=completion[:200],
            explanation=err if not passed else "passed",
        )

    return score


@task
def pearls_task(n: int = 12) -> Task:
    """Build the Programming Pearls task with a configurable subset size.

    ``n`` defaults to 12 (the full set). The first ``n`` problems are
    used in JSONL order, which is column order (1, 2, 2, 4, 8, 9, 11,
    12, 13, 14, 15, 15) -- so smaller ``n`` skews to easier columns.
    For random sampling, shuffle the JSONL and pick a slice.
    """
    items = _load_problems()[:n]
    samples = [_record_to_sample(item) for item in items]
    return Task(
        dataset=MemoryDataset(samples=samples, name="pearls"),
        solver=[system_message(SYSTEM_PROMPT), generate()],
        scorer=pearls_pass_at_1(),
    )
