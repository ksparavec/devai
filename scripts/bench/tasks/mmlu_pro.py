"""inspect_ai task for MMLU-Pro (harder, reasoning-heavy MMLU).

``TIGER-Lab/MMLU-Pro``: ~12k questions across 14 categories with up to
10 options each (vs 4 in MMLU) and distractors chosen to defeat shallow
pattern-matching. Ungated. A strong discriminator where plain MMLU and
GSM8K saturate for top models.

Scored with inspect_ai's ``multiple_choice`` solver (chain-of-thought
enabled -- MMLU-Pro is designed to be answered with reasoning) and the
``choice`` scorer (parses the model's ``ANSWER: X`` and compares to the
correct letter).

Sampling: the dataset is ordered by category, so a naive ``limit`` would
draw one category only. We ``shuffle`` with a fixed ``seed`` so the
subset is representative AND reproducible across re-benches.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

# Up to 10 options in MMLU-Pro; keep headroom in case the set grows.
_LETTERS = "ABCDEFGHIJKLMNOP"

# Fixed so the subset is identical run-to-run (comparability matters for
# a re-benchable leaderboard).
_SAMPLE_SEED = 42


def _record_to_sample(record: dict) -> Sample:
    """MMLU-Pro row -> inspect_ai Sample.

    ``options`` is the choice list; the correct answer is given both as a
    letter (``answer``) and an index (``answer_index``). We trust the
    letter and fall back to deriving it from the index so a schema quirk
    never silently mislabels the target.
    """
    options = list(record.get("options") or [])
    target = (record.get("answer") or "").strip()
    idx = record.get("answer_index")
    if not target and isinstance(idx, int) and 0 <= idx < len(options):
        target = _LETTERS[idx]
    return Sample(
        input=record["question"],
        choices=options,
        target=target,
        metadata={
            "category": record.get("category", ""),
            "question_id": record.get("question_id", ""),
        },
    )


@task
def mmlu_pro_task(n: int = 100) -> Task:
    """Build the MMLU-Pro Task with a representative, seeded subset."""
    dataset = hf_dataset(
        path="TIGER-Lab/MMLU-Pro",
        split="test",
        sample_fields=_record_to_sample,
        shuffle=True,
        seed=_SAMPLE_SEED,
        limit=n,
    )
    return Task(
        dataset=dataset,
        solver=[multiple_choice(cot=True)],
        scorer=choice(),
    )
