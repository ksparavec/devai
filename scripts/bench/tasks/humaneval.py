"""inspect_ai task wrapper for HumanEval (Python code completion).

Tests coding ability on 164 hand-written problems. Source:
openai_humaneval on HuggingFace. Each sample has a function signature
+ docstring (``prompt``) and a hidden test (``test``) that asserts on
the function's behaviour.

Pass@1 scoring: the model's first completion is concatenated to the
prompt, the resulting code is exec'd, then the test code is run
which calls ``check(<entry_point>)``. A subprocess sandbox provides
timeout + memory limits without nested-container plumbing.

Why "local" (subprocess) instead of inspect_ai's docker sandbox: the
bench harness already runs inside the lab container; running another
container nested for each sample requires podman socket forwarding
and adds 1-2 seconds of overhead per problem. A plain subprocess
with ``resource.setrlimit`` + ``signal.alarm`` is sufficient
isolation for trusted-but-buggy generated code (we trust the model
not to be malicious; we don't trust it not to fork-bomb).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate, system_message

SYSTEM_PROMPT = (
    "You are a Python coding assistant. Complete the function below. "
    "Output ONLY the function body (or the full function definition) — "
    "no explanations, no markdown fences, no extra prose. Your output "
    "will be concatenated directly to the prompt and executed."
)

# Subprocess time budget per sample. HumanEval problems are tiny;
# anything taking >10s is a model bug or an infinite loop.
_RUN_TIMEOUT_S = 10.0
# 256 MB cap on the child process. Far above what any HumanEval
# test needs; protects the host if the model writes a runaway list.
_MEM_LIMIT_MB = 256

# Strip markdown fences and any "Here is the implementation:" preamble
# that smaller models like to add despite the system prompt.
_FENCE_RX = re.compile(r"^```(?:python)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def _clean_completion(text: str) -> str:
    """Strip markdown fences from a completion. Keep the rest as-is."""
    if not text:
        return ""
    m = _FENCE_RX.search(text.strip())
    if m:
        return m.group(1)
    return text


def _record_to_sample(record: dict) -> Sample:
    """HumanEval row → inspect_ai Sample. We carry ``test`` and
    ``entry_point`` through ``metadata`` so the scorer can construct
    the test harness without needing to re-fetch the dataset.
    """
    return Sample(
        input=record["prompt"],
        target=record.get("canonical_solution") or "",
        metadata={
            "task_id": record.get("task_id", ""),
            "test": record.get("test", ""),
            "entry_point": record.get("entry_point", ""),
        },
    )


def _run_check_in_subprocess(program: str) -> tuple[bool, str]:
    """Exec ``program`` in a fresh Python process. Returns
    ``(passed, stderr_excerpt)``.

    The child sets a memory rlimit and an alarm before exec'ing the
    code. ``passed`` is True iff the child exits 0. Stderr is captured
    so failures get a readable explanation in the score record.
    """
    # Embed the program in a wrapper that installs limits before
    # running. Keeps the subprocess invocation a single argv list.
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
            raise TimeoutError("HumanEval sample exceeded {int(_RUN_TIMEOUT_S)}s")
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
    # Cap the excerpt so cache rows don't balloon.
    return False, err[-400:]


@scorer(metrics=[accuracy(), stderr()])
def humaneval_pass_at_1():
    """Pass@1 scorer. Concatenates prompt + completion + test, runs
    in a subprocess with limits, scores 1.0 on exit 0, 0.0 otherwise.
    """

    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata or {}
        test = meta.get("test", "")
        entry_point = meta.get("entry_point", "")
        if not test or not entry_point:
            return Score(value=0.0, answer="", explanation="missing test or entry_point")

        completion = _clean_completion(state.output.completion or "")
        # The HumanEval convention: program = prompt + completion + test +
        # `check(<entry>)` invocation. The test module defines `check(fn)`.
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
def humaneval_task(n: int = 50) -> Task:
    """Build the HumanEval Task with a configurable subset size.

    ``n`` defaults to 50 (out of 164). Smaller subset still
    distinguishes models well enough for routing decisions, runs in
    ~3-7 min depending on completion length.
    """
    dataset = hf_dataset(
        path="openai/openai_humaneval",
        split="test",
        sample_fields=_record_to_sample,
        limit=n,
        shuffle=False,
    )
    return Task(
        dataset=dataset,
        solver=[system_message(SYSTEM_PROMPT), generate()],
        scorer=humaneval_pass_at_1(),
    )
