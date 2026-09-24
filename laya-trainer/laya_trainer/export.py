"""ONNX export, and running a set of items through torch or ONNX Runtime.

The graph has laya's input/output names (scripts/export_onnx.py), which is what
aiagent's onnxruntime port feeds. Upstream traces with batch 1, sequence 16 and
2 markers; that specialises the batch dimension (ORT then warns "Expected shape
{1,2} ... actual {2,2}" on every batched call, and the batched graph ran about
5x slower on this host). Here the trace uses batch 2, sequence 320, 3 markers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUT_NAMES = ["logits", "act_logits"]
DYNAMIC_AXES = {
    "input_ids": {0: "batch_size", 1: "seq_len"},
    "attention_mask": {0: "batch_size", 1: "seq_len"},
    "marker_pos": {0: "batch_size", 1: "num_markers"},
    "marker_mask": {0: "batch_size", 1: "num_markers"},
    "qtype": {0: "batch_size"},
    "logits": {0: "batch_size", 1: "num_markers"},
    "act_logits": {0: "batch_size"},
}
OPSET = 18
TRACE_BATCH, TRACE_SEQ, TRACE_MARKERS = 2, 320, 3
ONNX_NAME = "model.onnx"


def export_onnx(model: torch.nn.Module, out_dir: Path) -> dict:
    """Write model.onnx (plus any external-data file) and describe the export."""
    model = model.eval().cpu().float()
    vocab = int(model.encoder.config.vocab_size)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(5, vocab, (TRACE_BATCH, TRACE_SEQ), generator=g)
    att = torch.ones(TRACE_BATCH, TRACE_SEQ, dtype=torch.long)
    att[1, TRACE_SEQ - 37:] = 0  # one padded row, as in any real batch
    mpos = torch.tensor([[1, 5, 9]] * TRACE_BATCH)
    mmask = torch.ones(TRACE_BATCH, TRACE_MARKERS, dtype=torch.bool)
    qtype = torch.tensor([0, 2])
    out_dir = Path(out_dir)
    before = {p.name for p in out_dir.iterdir()}
    with torch.no_grad():
        # dynamo=True explicitly (it is torch 2.14's default): the TorchScript
        # exporter bakes the traced sequence length into the head's Reshape
        # (reproduced by aiagent on a tiny ModernBERT DecisionModel). External
        # data stays at the default size threshold -- 0 makes ORT refuse the
        # graph.
        torch.onnx.export(model, (ids, att, mpos, mmask, qtype), str(out_dir / ONNX_NAME),
                          input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
                          dynamic_axes=DYNAMIC_AXES, opset_version=OPSET,
                          do_constant_folding=True, dynamo=True)
    written = sorted({p.name for p in out_dir.iterdir()} - before)
    return {
        "opset": OPSET,
        "precision": "fp32",
        "torch": torch.__version__,
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "dynamic_axes": {k: {str(a): n for a, n in v.items()} for k, v in DYNAMIC_AXES.items()},
        "trace_shape": {"batch": TRACE_BATCH, "seq": TRACE_SEQ, "markers": TRACE_MARKERS},
        # DecisionModel.forward branches on the marker count in Python (topk(2)
        # needs >= 2 options); the traced graph keeps the >= 2 branch only.
        "min_markers": 2,
        "files": written,
    }


def ort_session(path: Path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def collate_np(items: list[dict], pad_id: int) -> dict[str, np.ndarray]:
    """laya's collate_items, as numpy arrays in the dtypes the ONNX graph takes."""
    n, width = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = np.full((n, width), pad_id, dtype=np.int64)
    att = np.zeros((n, width), dtype=np.int64)
    mpos = np.zeros((n, kmax), dtype=np.int64)
    mmask = np.zeros((n, kmax), dtype=bool)
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = it["ids"]
        att[i, :len(it["ids"])] = 1
        mpos[i, :len(it["markers"])] = it["markers"]
        mmask[i, :len(it["markers"])] = True
    qtype = np.array([it["qtype"] for it in items], dtype=np.int64)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos,
            "marker_mask": mmask, "qtype": qtype}


def ort_logits(session, items: list[dict], pad_id: int, batch_size: int = 16) -> list[np.ndarray]:
    """Per item, the logits of its own options (float64)."""
    out = []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        logits = session.run(["logits"], collate_np(chunk, pad_id))[0]
        out += [logits[i, :len(it["markers"])].astype(np.float64) for i, it in enumerate(chunk)]
    return out


def torch_logits(model: torch.nn.Module, items: list[dict], pad_id: int,
                 batch_size: int = 16) -> list[np.ndarray]:
    """The same, through the torch model in fp32 on CPU."""
    model = model.eval().cpu().float()
    out = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            b = {k: torch.from_numpy(v) for k, v in collate_np(chunk, pad_id).items()}
            logits, _ = model(b["input_ids"], b["attention_mask"], b["marker_pos"],
                              b["marker_mask"], b["qtype"])
            z = logits.float().numpy()
            out += [z[i, :len(it["markers"])].astype(np.float64) for i, it in enumerate(chunk)]
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()
