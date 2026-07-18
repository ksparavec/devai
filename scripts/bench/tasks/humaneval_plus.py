"""inspect_ai task for HumanEval+ (EvalPlus hardened test suite).

Same 164 problems as HumanEval, but ``evalplus/humanevalplus`` ships a
much larger ``test`` suite (~80x more assertions) that catches
completions which pass HumanEval's weak original tests. The dataset row
schema is identical to ``openai/openai_humaneval`` (``task_id``,
``prompt``, ``canonical_solution``, ``entry_point``, ``test``) and the
``test`` field is the same ``check(candidate)`` harness, so we reuse
humaneval.py's completion cleaner and pass@1 subprocess scorer verbatim
and only swap the dataset path.

Run it alongside HumanEval (``--tasks humaneval,humaneval_plus``) to see
the gap open up: the delta is exactly the set of solutions that passed
the toy tests but fail under adversarial inputs.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset
from inspect_ai.solver import generate, system_message

# Reuse the HumanEval building blocks -- same schema, same harness.
from bench.tasks.humaneval import (
    SYSTEM_PROMPT,
    _record_to_sample,
    humaneval_pass_at_1,
)


@task
def humaneval_plus_task(n: int = 50) -> Task:
    """Build the HumanEval+ Task with a configurable subset size.

    Defaults to the same ``n`` as HumanEval so the two are directly
    comparable (identical problems, harder tests) -- the score delta
    isolates the effect of the expanded test suite, not a different
    problem sample.
    """
    dataset = hf_dataset(
        path="evalplus/humanevalplus",
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
