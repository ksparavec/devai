"""inspect_ai task wrapper for GSM8K (grade-school math word problems).

Tests reasoning on multi-step arithmetic. Source: openai/gsm8k on
HuggingFace. Each sample has a question and an answer that ends with
``#### <integer>``. Scoring is exact-match on the final integer.

Why this is in v1: it's the cheapest reasoning bench that actually
distinguishes models. Single-turn, deterministic, no judge needed.
A 100-question subset takes ~3-5 minutes per model and gives a stable
score with low variance.
"""

from __future__ import annotations

import re

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate, system_message

# GSM8K answers always end with `#### <integer>`. Models tend to scatter
# the final number through prose; pull the LAST integer that appears
# after `####` if present, else the last integer in the response.
_FINAL_ANSWER_RX = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
_LAST_NUMBER_RX = re.compile(r"-?\d+(?:\.\d+)?")

SYSTEM_PROMPT = (
    "Solve the math problem step by step. After reasoning, output the "
    "final numeric answer on a line by itself prefixed with '#### '. "
    "Example: '#### 42'."
)


def _record_to_sample(record: dict) -> Sample:
    """Convert an HF GSM8K row to an inspect_ai Sample.

    The dataset's `answer` field is a chain-of-thought ending in
    `#### <int>`. Strip everything before `####` so the target is
    just the final number — that's what the scorer compares against.
    """
    raw_answer = record.get("answer", "")
    m = _FINAL_ANSWER_RX.search(raw_answer)
    target = m.group(1) if m else raw_answer.strip()
    return Sample(
        input=record["question"],
        target=target,
        metadata={"source": "openai/gsm8k", "split": "test"},
    )


@scorer(metrics=[accuracy(), stderr()])
def gsm8k_match():
    """Score by extracting the final integer from the model output and
    comparing to the gold answer. Tolerates trailing punctuation and
    answers buried in prose. Rejects answers that don't contain any
    digits at all (model declined or wandered).
    """

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion or ""
        # Prefer the explicit '#### N' shape if the model followed
        # the system prompt; fall back to the last integer otherwise.
        m = _FINAL_ANSWER_RX.search(completion)
        if m:
            answer = m.group(1)
        else:
            nums = _LAST_NUMBER_RX.findall(completion)
            answer = nums[-1] if nums else ""
        gold = (target.text or "").strip()
        # Numeric equality with tolerance for floats vs ints.
        try:
            ok = float(answer) == float(gold)
        except ValueError:
            ok = answer == gold
        return Score(
            value=1.0 if ok else 0.0,
            answer=answer,
            explanation=(
                f"extracted={answer!r} gold={gold!r}"
            ),
        )

    return score


@task
def gsm8k_task(n: int = 100) -> Task:
    """Build the GSM8K Task with a configurable subset size.

    ``n`` defaults to 100 — large enough for stable signal, small
    enough that a 7B model finishes in 3-5 min on a 24GB GPU.
    """
    dataset = hf_dataset(
        path="openai/gsm8k",
        name="main",
        split="test",
        sample_fields=_record_to_sample,
        limit=n,
        shuffle=False,
    )
    return Task(
        dataset=dataset,
        solver=[system_message(SYSTEM_PROMPT), generate()],
        scorer=gsm8k_match(),
    )
