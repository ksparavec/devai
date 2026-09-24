"""golden.jsonl: answers aiagent's runtime must reproduce from the artifact.

One row per held-out dataset row (at most 40), in aiagent's shape:

    {"id", "state", "questions": {qid: laya question},
     "expected": {qid: {"input_ids", "markers", "answer"}}}

`input_ids` / `markers` are laya's tokenization of the row; `answer` is laya's
own `Agent.predict(...)["answers"][qid]` without "action", computed on CPU in
fp32 (LAYA_CPU_AMP unset) from the calibrated checkpoint -- laya itself is the
reference. aiagent checks its tokenization against `input_ids` and its
torch-free ONNX decode against `answer` (1e-3), so any drift between the two
repos -- tokenizer, laya version, export, temperature handling -- shows up as
a golden mismatch, not as silently worse decisions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

GOLDEN_N = 40
TOLERANCE = 1e-3


def golden_rows(checkpoint_dir: Path, items: list[dict]) -> list[dict]:
    """The first GOLDEN_N held-out rows, answered by laya's Agent."""
    rows: dict[str, dict] = {}
    for it in items:
        if it["row_id"] not in rows:
            if len(rows) == GOLDEN_N:
                break
            rows[it["row_id"]] = {"id": it["row_id"], "state": it["state"],
                                  "questions": {}, "expected": {}}
        r = rows[it["row_id"]]
        r["questions"][it["qid"]] = it["question"]
        r["expected"][it["qid"]] = {"input_ids": it["ids"], "markers": it["markers"]}
    os.environ.pop("LAYA_CPU_AMP", None)  # fp32: bf16 autocast would miss aiagent's 1e-3
    from laya.agent import Agent
    agent = Agent(str(checkpoint_dir), device="cpu")
    for r in rows.values():
        answers = agent.predict(r["state"], r["questions"])["answers"]
        for qid, expected in r["expected"].items():
            expected["answer"] = {k: v for k, v in answers[qid].items() if k != "action"}
    return list(rows.values())


def write_golden(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
