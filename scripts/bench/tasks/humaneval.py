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
with ``resource.setrlimit`` (RLIMIT_AS for memory, RLIMIT_NPROC for
fork bombs) + ``signal.alarm`` is sufficient isolation for
trusted-but-buggy generated code (we trust the model not to be
malicious; we don't trust it not to fork-bomb).
"""

from __future__ import annotations

import json
import os
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
# 2 GB address-space cap on the child process -- still a runaway-list
# guard, but high enough for numpy/OpenBLAS. HumanEval+ (EvalPlus) tests
# are numpy-heavy; at the old 256 MB cap OpenBLAS failed every allocation
# ("Memory allocation still failed after 10 retries") and every sample
# scored 0. We also pin BLAS to a single thread below (see _BLAS_ENV) so
# OpenBLAS doesn't reserve a per-core buffer that scales with CPU count.
_MEM_LIMIT_MB = 2048

# Fork-bomb guard headroom. RLIMIT_NPROC is a ceiling on the total
# number of tasks (processes AND threads) the *real uid* may hold, not a
# per-process quota, so it can only be set relative to what is already
# running -- a fixed constant would either be uselessly high on a busy
# host or make a legitimate `import` fail on a quiet one. The child needs
# no subprocesses of its own; the headroom just covers interpreter
# start-up and the odd helper thread an imported library spawns.
_NPROC_HEADROOM = 32

# Force single-threaded BLAS in the test subprocess: OpenBLAS otherwise
# allocates thread-local buffers sized to the host core count, which both
# wastes the RLIMIT_AS budget and makes execution nondeterministic.
_BLAS_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

# Match any fenced code block (with optional ``python|py`` info-string)
# anywhere in the text. Non-greedy body so multiple fences round-trip
# correctly via ``findall`` (caller takes the last match).
_FENCE_BLOCK_RX = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)
# Strip ``<think>...</think>`` reasoning preambles that inline-reasoning
# models (Nemotron-Nano, R1-Distill-Llama-8B) emit before the actual
# code. Models with a probe-verified reasoning parser (Qwen3, deepseek_r1
# applied to Qwen tokenizer, harmony) split ``reasoning_content`` off
# server-side, so this regex is a no-op for them; it only fires for the
# inline-reasoning case.
_THINK_RX = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _clean_completion(text: str, entry_point: str = "") -> str:
    """Extract executable Python from a model completion.

    Strategy (in order):

    1. Strip any ``<think>...</think>`` reasoning blocks. Inline-reasoning
       models stuff CoT prose, sometimes pseudocode, into these blocks
       before the real answer. Leaving them in causes the subprocess
       to choke on ``<`` as a syntax error.
    2. If one or more fenced code blocks are present, return the body
       of the **last** one. The last fence is empirically the model's
       final answer — earlier fences are often draft attempts the model
       then revised.
    3. If no fence is present and an ``entry_point`` is known, find the
       first line that starts with ``def <entry_point>(`` and slice from
       there to the end. Catches models that emit raw code after a
       ``<think>`` block without wrapping it.
    4. Otherwise return the think-stripped text as-is. Same fall-through
       behaviour as the original strict fence-only cleaner for non-
       reasoning models with clean output.

    The strict v1 cleaner only matched a fence enclosing the entire
    completion (after ``strip()``) and otherwise returned the raw text.
    That returned ``<think>...</think>\\n```python\\n...\\n``` `` verbatim
    to the subprocess, which failed every Nemotron-Nano and R1-Distill-
    Llama-8B sample with a parse error -- not because the code was bad
    but because the wrapper text wasn't valid Python.
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
            return cleaned[m.start() :]
    return cleaned


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

    The child sets a memory rlimit, a process-count rlimit and an alarm
    before exec'ing the code. ``passed`` is True iff the child exits 0.
    Stderr is captured so failures get a readable explanation in the
    score record.
    """
    # Embed the program in a wrapper that installs limits before
    # running. Keeps the subprocess invocation a single argv list.
    wrapper = textwrap.dedent(f"""
        import os, resource, signal, sys
        try:
            resource.setrlimit(
                resource.RLIMIT_AS,
                ({_MEM_LIMIT_MB * 1024 * 1024}, {_MEM_LIMIT_MB * 1024 * 1024}),
            )
        except (ValueError, OSError):
            pass
        # Fork-bomb guard: count the tasks this uid already holds, then
        # cap RLIMIT_NPROC just above that. fork() past the cap fails
        # with BlockingIOError instead of multiplying without bound.
        # Best-effort -- if /proc is unreadable we leave the inherited
        # limit rather than guessing a number that breaks imports.
        try:
            _uid = os.getuid()
            _live = 0
            for _e in os.scandir("/proc"):
                if not _e.name.isdigit():
                    continue
                try:
                    if _e.stat().st_uid != _uid:
                        continue
                    _live += len(os.listdir(_e.path + "/task"))
                except OSError:
                    continue
            _hard = resource.getrlimit(resource.RLIMIT_NPROC)[1]
            _cap = _live + {_NPROC_HEADROOM}
            if _hard != resource.RLIM_INFINITY:
                _cap = min(_cap, _hard)
            resource.setrlimit(resource.RLIMIT_NPROC, (_cap, _hard))
        except (ValueError, OSError):
            pass
        def _timeout(signum, frame):
            raise TimeoutError("HumanEval sample exceeded {int(_RUN_TIMEOUT_S)}s")
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm({int(_RUN_TIMEOUT_S)})
    """)
    full = wrapper + "\n" + program
    try:
        # Feed the program on stdin (`python -`), NOT as `-c <program>`:
        # EvalPlus's hardened test strings are hundreds of KB and overflow
        # the OS argv limit (ARG_MAX) as a `-c` argument, raising
        # OSError(7, 'Argument list too long') and aborting the whole eval.
        r = subprocess.run(
            [sys.executable, "-"],
            input=full,
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_S + 2,
            env={**os.environ, **_BLAS_ENV},
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

        completion = _clean_completion(
            state.output.completion or "", entry_point=entry_point
        )
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
