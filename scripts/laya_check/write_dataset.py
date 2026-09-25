"""Write the laya reference dataset with aiagent's own writer.

Runs INSIDE the lab image, with aiagent's bundled Python (the only place
aiagent's `distill` package is installed), piped to `python -I -` by
scripts/laya-check.py. Argument: the distill directory (/laya), whose
inbox/ must be writable. Prints one JSON line: the dataset id and counts.

The dataset is synthetic: 80 short multilingual sentences for aiagent's
`polarity` skill, labelled by construction, with placeholder binds and
teacher "interop". It is the reference for `make laya-check` (first run
2026-09-25, id ds-a6c9c8248242). Nothing here may change without updating
REFERENCE in scripts/laya-check.py: the id is a hash of the content, so a
different sentence, split or label is a different dataset. A writer that
produces a different id from this same input means aiagent's dataset output
changed -- a contract change the check reports rather than absorbs.
"""

import itertools
import json
import sys
from pathlib import Path

from aiagent.distill import dataset as D
from aiagent.distill.splits import assign_split, document_id

KEYS = ["negative", "neutral", "mixed", "positive"]
QUESTIONS = {"polarity": {"type": "choice",
                          "instructions": "Overall sentiment polarity of the passage.",
                          "criteria": KEYS}}
SUBJECTS = ["The delivery", "Customer support", "The new update", "Die Lieferung",
            "Der Kundendienst", "Dostava", "Korisnička podrška", "The hotel room",
            "Das Essen", "Aplikacija"]
VERDICTS = [("was excellent and fast.", "positive"),
            ("was late and the box was broken.", "negative"),
            ("was okay, nothing special.", "neutral"),
            ("war großartig, aber teuer.", "mixed"),
            ("je bila izvrsna, ali spora.", "mixed"),
            ("war völlig enttäuschend.", "negative"),
            ("je bila odlična!", "positive"),
            ("was fine but the staff were rude.", "mixed")]


def build_rows(tok, base):
    rows = []
    for i, (subj, (tail, label)) in enumerate(itertools.product(SUBJECTS, VERDICTS)):
        text = f"{subj} {tail} ({i})"
        gid = document_id(text)
        split = assign_split(gid)
        kw = dict(group_id=gid, index=0, text=text, input_field="text", questions=QUESTIONS,
                  split=split, tokenizer=tok, max_len=base.max_len, head_max_len=base.head_max_len)
        if split != "pool":
            votes = {label: 6, "neutral" if label != "neutral" else "mixed": 2}
            kw["gold"] = {"polarity": D.gold_from_votes(KEYS, votes)}
            kw["teacher"] = D.TeacherInfo(k=8, parse_failures={"polarity": 0})
        rows.append(D.build_row(**kw))
    return rows


def main(distill: Path) -> None:
    base = D.load_base_checkpoint(distill, D.DEFAULT_BASE)
    rows = build_rows(base.tokenizer(), base)
    producer = D.Producer(
        version="interop", created_at="2026-09-24T21:00:00Z", skill="polarity",
        predictor="classify", input_field="text", derive_version=1,
        binds=D.ProducerBinds(signature_sha256="a" * 64, question_set_sha256="b" * 64,
                              skill_source_sha256="c" * 64),
        questions=QUESTIONS,
        teacher=D.TeacherSpec(model="interop", k=8, temperature=0.7, dspy="3.2.1"),
        campaign="c-interop", round=0, parent_dataset=None,
        sources=[D.SourceInfo(doc_id=r.group_id, origin=f"interop:{n}", segments=1)
                 for n, r in enumerate(rows)],
        label_stats=D.compute_label_stats(rows, unlabeled=0))
    ref = D.write_dataset(distill / "inbox", base=base, producer=producer, rows=rows)
    counts = {}
    for r in rows:
        counts[r.split] = counts.get(r.split, 0) + 1
    print(json.dumps({"dataset": ref.id, "counts": counts}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
