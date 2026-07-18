"""inspect_ai task for GPQA-Diamond (graduate-level, Google-proof science).

``Idavidrein/gpqa`` config ``gpqa_diamond``: 198 hard physics/chemistry/
biology multiple-choice questions written and validated by domain PhDs,
designed so non-experts with web access still score near chance. The single
best discriminator among strong models where GSM8K/HumanEval/MMLU saturate.

Gated dataset: needs HF access granted to the token (accept the terms at
huggingface.co/datasets/Idavidrein/gpqa).

Each row gives one Correct Answer + three Incorrect Answers. We build a
4-way choice, shuffle the options with a per-question deterministic seed
(so the correct answer isn't always "A", yet the layout is reproducible
across runs), and score with inspect_ai's multiple_choice (CoT) + choice.
"""

from __future__ import annotations

import hashlib
import random

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

_LETTERS = "ABCD"


def _record_to_sample(record: dict) -> Sample:
    """GPQA row -> Sample with a deterministically-shuffled 4-way choice.

    The correct answer is index 0 before the shuffle; after shuffling we
    record the letter of its new position as the target. The shuffle is
    seeded from the question text so the same question always yields the
    same option order (reproducible re-benches) without biasing the answer
    toward a fixed position.
    """
    question = (record.get("Question") or "").strip()
    correct = (record.get("Correct Answer") or "").strip()
    incorrect = [
        (record.get(f"Incorrect Answer {i}") or "").strip() for i in (1, 2, 3)
    ]
    options = [correct] + incorrect  # correct is index 0
    order = list(range(4))
    random.Random(hashlib.sha256(question.encode()).hexdigest()).shuffle(order)
    shuffled = [options[i] for i in order]
    target = _LETTERS[order.index(0)]  # where the correct option landed
    return Sample(
        input=question,
        choices=shuffled,
        target=target,
        metadata={"subdomain": record.get("Subdomain", "")},
    )


@task
def gpqa_task(n: int = 100) -> Task:
    """Build the GPQA-Diamond Task. Default subset of 100 of the 198
    questions (shuffle+seed for a representative, reproducible sample);
    pass n>=198 to run the full set."""
    dataset = hf_dataset(
        path="Idavidrein/gpqa",
        name="gpqa_diamond",
        split="train",
        sample_fields=_record_to_sample,
        shuffle=True,
        seed=42,
        limit=n,
    )
    return Task(
        dataset=dataset,
        solver=[multiple_choice(cot=True)],
        scorer=choice(),
    )
