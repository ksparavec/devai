"""A tiny laya checkpoint and a matching dataset, for devai's own tests.

The checkpoint is a real laya DecisionModel on a ModernBERT encoder with a
32-wide hidden size and a word-level tokenizer of a few hundred entries (well
under 1 MB in total), so the whole pipeline -- import, validation, training,
ONNX export, calibration, parity, golden answers -- runs on CPU in seconds.
(aiagent builds its own fixture.) `python -m laya_trainer.fixture <out_dir>`
writes one.

The dataset builder writes what aiagent writes (docs/laya-trainer.md, dataset
contract): laya's build_sequence and H(ids) for student_tokens, a `files`
table, pool rows without gold or teacher.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

from .contract import canonical_hash, write_json_atomic, write_sha256sums

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
WORDS = sorted(set("""
    the a an and or but not very quite rather really so too of to in on at for with from by
    is was were be been are it this that these those there here i you we they he she
    good great excellent wonderful fine nice happy pleased love liked enjoyed fast friendly
    bad terrible awful poor slow late broken rude sad angry hate disliked noisy dirty
    okay average mixed neutral decent acceptable ordinary usual normal plain
    delivery support service product price food room staff screen battery update app
    arrived came worked failed broke helped answered waited replied shipped fixed
    question overall sentiment polarity passage text statement holds does level
    negative positive mostly both yes no score choice noul true false instructions
    one two three four five weeks days hours minutes today yesterday again never always
    """.split()))
POLARITY_WORDS = {
    "positive": "good great excellent wonderful happy pleased love liked enjoyed fast friendly".split(),
    "negative": "bad terrible awful poor slow late broken rude sad angry hate disliked".split(),
    "mixed": "okay average mixed decent acceptable ordinary".split(),
}
FILLER = "the delivery support service product price food room staff app arrived came worked".split()

QUESTIONS = {
    "polarity": {"type": "choice", "instructions": "Overall sentiment polarity of the passage.",
                 "criteria": {"negative": "mostly negative", "mixed": "both positive and negative",
                              "positive": "mostly positive"}},
    "positive": {"type": "noul", "instructions": "Is the passage positive?"},
    "stars": {"type": "score", "instructions": "Score the sentiment of the passage.",
              "criteria": ["very negative", "negative", "neutral", "positive", "very positive"]},
}


def build_tokenizer(tok_dir: Path):
    """A word-level PreTrainedTokenizerFast saved as tokenizer.json + tokenizer_config.json."""
    from laya.agent import _fix_tokenizer_config
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    vocab = {t: i for i, t in enumerate(SPECIAL + WORDS)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.normalizer = normalizers.Lowercase()
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    tok_dir = Path(tok_dir)
    fast.save_pretrained(str(tok_dir))
    for p in tok_dir.iterdir():
        if p.name not in ("tokenizer.json", "tokenizer_config.json"):
            p.unlink()
    # The form laya's loader expects, so a read-only fixture never needs a rewrite.
    _fix_tokenizer_config(str(tok_dir.parent))
    return AutoTokenizer.from_pretrained(str(tok_dir))


def build_checkpoint(out_dir: Path, *, seed: int = 0, name: str = "laya-fixture") -> dict:
    """Write a loadable laya checkpoint and return its laya catalog row."""
    import torch
    from laya.common import build_model
    from safetensors.torch import save_file
    from transformers import ModernBertConfig

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = build_tokenizer(out_dir / "tokenizer")
    ecfg = ModernBertConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, max_position_embeddings=1024, local_attention=64,
        global_attn_every_n_layers=2, pad_token_id=tok.pad_token_id, cls_token_id=tok.cls_token_id,
        sep_token_id=tok.sep_token_id, bos_token_id=tok.cls_token_id, eos_token_id=tok.sep_token_id)
    ecfg.save_pretrained(str(out_dir / "encoder"))
    cfg = {"encoder": "devai/laya-fixture", "head_layers": 2, "max_len": 256, "head_max_len": 64,
           "max_prefixes": 6, "act_costs": {"escalate": 0.5}, "cost_wrong_act": 3.0,
           "amp_dtype": "bf16", "model_name": name, "temperature": [1.0, 1.0, 1.0],
           "temperature_by_options": {}}
    torch.manual_seed(seed)
    model = build_model(cfg, encoder_dir=str(out_dir / "encoder"), pretrained=False)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              str(out_dir / "model.safetensors"))
    write_json_atomic(out_dir / "rl_agent_config.json", cfg)
    return catalog_row(out_dir, name=name)


def catalog_row(ckpt_dir: Path, *, name: str, default: bool = True) -> dict:
    from .catalog import REQUIRED_FILES
    files = {}
    for rel in REQUIRED_FILES:
        data = (Path(ckpt_dir) / rel).read_bytes()
        files[rel] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    return {"name": name, "default": default, "repo": "devai/laya-fixture",
            "revision": "0" * 40, "subfolder": "", "license": "Apache-2.0",
            "encoder": "devai/laya-fixture", "files": files}


def _state(rng: random.Random, label: str) -> str:
    words = [rng.choice(FILLER) for _ in range(rng.randint(3, 8))]
    if label == "mixed":
        words += [rng.choice(POLARITY_WORDS["positive"]), "but", rng.choice(POLARITY_WORDS["negative"])]
    else:
        words += [rng.choice(POLARITY_WORDS[label]) for _ in range(rng.randint(1, 3))]
    rng.shuffle(words)
    return " ".join(words)


def _gold(label: str) -> dict:
    polarity = {"negative": 0.1, "mixed": 0.1, "positive": 0.1}
    polarity[label] = 0.8
    pos = {"positive": 0.9, "mixed": 0.4, "negative": 0.1}[label]
    stars = {"negative": [0.4, 0.4, 0.1, 0.05, 0.05], "mixed": [0.05, 0.15, 0.6, 0.15, 0.05],
             "positive": [0.05, 0.05, 0.1, 0.4, 0.4]}[label]
    return {
        "polarity": {"label": label, "probabilities": polarity},
        "positive": {"label": "true" if pos >= 0.5 else "false",
                     "probabilities": {"false": round(1 - pos, 10), "true": pos}},
        "stars": {"label": str(stars.index(max(stars))),
                  "probabilities": {str(i): p for i, p in enumerate(stars)}},
    }


def student_tokens(tok, state, qdef: dict, max_len: int, head_max_len: int) -> dict:
    """What aiagent writes per question: n and H(ids) of laya's own sequence."""
    from laya.agent import Agent
    from laya.common import build_sequence, serialize_state

    q = Agent._to_internal(qdef)
    state_ids = tok(serialize_state(state).replace(tok.mask_token, " "),
                    add_special_tokens=False)["input_ids"]
    ids, _ = build_sequence(tok, state, q, max_len, head_max_len,
                            truncate_left=isinstance(state, list), state_ids=state_ids)
    return {"n": len(ids), "ids_sha256": canonical_hash(ids)}


def build_dataset(store: Path, row: dict, tok, *, counts: dict | None = None, seed: int = 0,
                  max_len: int = 128, head_max_len: int = 48,
                  questions: dict | None = None) -> str:
    """Write inbox/<ds-id>/ for `row` and return the id."""
    counts = counts or {"train": 24, "calib": 30, "heldout": 12, "pool": 6}
    questions = QUESTIONS if questions is None else questions
    rng = random.Random(seed)
    labels = ["negative", "mixed", "positive"]
    splits: dict[str, list[dict]] = {}
    seen_states: set[str] = set()
    n = 0
    for split, count in counts.items():
        splits[split] = []
        for _ in range(count):
            label = labels[n % 3]
            state = _state(rng, label)
            while state in seen_states:  # one document per group: splits stay disjoint
                state = _state(rng, label)
            seen_states.add(state)
            group = hashlib.sha256(state.encode()).hexdigest()
            r = {"id": f"{group[:12]}:{n:04d}", "group_id": f"sha256:{group}", "split": split,
                 "synthetic": False, "state": {"text": state}, "questions": questions}
            if split != "pool":
                r["gold"] = {q: g for q, g in _gold(label).items() if q in questions}
                r["teacher"] = {"k": 4, "parse_failures": {q: 0 for q in questions}}
            r["student_tokens"] = {qid: student_tokens(tok, {"text": state}, q, max_len, head_max_len)
                                   for qid, q in questions.items()}
            splits[split].append(r)
            n += 1
    bodies = {f"{s}.jsonl": "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                                    for r in rows).encode("utf-8")
              for s, rows in splits.items()}
    manifest = {
        "schema_version": 1,
        "base_checkpoint": {"name": row["name"], "revision": row["revision"],
                            "weights_sha256": row["files"]["model.safetensors"]["sha256"],
                            "tokenizer_sha256": row["files"]["tokenizer/tokenizer.json"]["sha256"]},
        "max_len": max_len, "head_max_len": head_max_len,
        "splits": {"method": "fixture", "counts": dict(counts)},
        "files": {name: {"sha256": hashlib.sha256(b).hexdigest(), "rows": counts[name[:-6]]}
                  for name, b in bodies.items()},
        "producer": {"name": "laya_trainer.fixture",
                     "binds": {"signature_sha256": "a" * 64, "question_set_sha256": canonical_hash(questions),
                               "skill_source_sha256": "b" * 64}},
    }
    body = (json.dumps(manifest, indent=2) + "\n").encode()
    ds_id = "ds-" + hashlib.sha256(body).hexdigest()[:12]
    ds_dir = Path(store) / "inbox" / ds_id
    ds_dir.mkdir(parents=True)
    (ds_dir / "manifest.json").write_bytes(body)
    for name, b in bodies.items():
        (ds_dir / name).write_bytes(b)
    write_sha256sums(ds_dir, ["manifest.json", *bodies])
    return ds_id


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m laya_trainer.fixture <out_dir>", file=sys.stderr)
        return 2
    row = build_checkpoint(Path(args[0]))
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
