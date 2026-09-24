"""Temperature calibration on the ONNX graph's own logits.

One temperature per question type (choice, score, noul), fitted on the
dataset's `calib` split -- rows the student never trained on -- by minimising
the cross-entropy against the teacher's distribution, as laya's notebook does.
Fitted on ONNX fp32 logits because that graph is what aiagent runs.

laya clamps every temperature it applies to [0.5, 5.0] (a fit below 0.5
sharpens a coin flip into a certainty). The raw fit and the applied value are
both recorded; the artifact's rl_agent_config.json carries the applied one, and
`temperature_by_options` is dropped (a per-type fit would be hidden by it).
"""

from __future__ import annotations

import numpy as np
import torch

MIN_SAMPLES = 10


def fit_temperature(logits: list[np.ndarray], targets: list[list[float]]) -> float | None:
    """The raw temperature for one question type, or None if too few samples."""
    if len(logits) < MIN_SAMPLES:
        return None
    kmax = max(len(z) for z in logits)
    Z = torch.full((len(logits), kmax), -1e4, dtype=torch.float64)
    T = torch.zeros((len(logits), kmax), dtype=torch.float64)
    for i, (z, t) in enumerate(zip(logits, targets)):
        Z[i, :len(z)] = torch.from_numpy(np.asarray(z, dtype=np.float64))
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    raw = float(log_t.exp().item())
    return raw if np.isfinite(raw) else None


def calibrate(items: list[dict], logits: list[np.ndarray]) -> dict:
    from laya.common import TEMP_MAX, TEMP_MIN, clamp_temperature

    raw, applied, per_type = [], [], []
    for qt in range(3):
        sel = [(z, it["target"]) for z, it in zip(logits, items) if it["qtype"] == qt]
        per_type.append(len(sel))
        r = fit_temperature([z for z, _ in sel], [t for _, t in sel])
        raw.append(r)
        applied.append(clamp_temperature(r) if r is not None else 1.0)
    return {
        "split": "calib",
        "n": len(items),
        "per_type_n": per_type,
        "types": ["choice", "score", "noul"],
        "fitted_on": "onnx_fp32_logits",
        "min_samples": MIN_SAMPLES,
        "temperature_raw": raw,
        "temperature_applied": applied,
        "clamp": [TEMP_MIN, TEMP_MAX],
    }
