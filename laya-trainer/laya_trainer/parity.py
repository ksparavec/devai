"""The torch-vs-ONNX parity gate on the held-out split.

The artifact is only as good as its export: aiagent runs the ONNX graph, never
the torch model, so a graph that drifts from the model it came from would ship
an untested model. Every held-out item goes through both; the largest absolute
difference between their option probabilities must stay within 1e-3, or the
job fails with exit 7.
"""

from __future__ import annotations

import numpy as np

from .contract import ExitCode, JobError
from .export import softmax

TOLERANCE = 1e-3


def check_parity(items: list[dict], torch_z: list[np.ndarray], onnx_z: list[np.ndarray]) -> dict:
    if not items:
        raise JobError(ExitCode.EXPORT, "no held-out items to check parity on")
    max_diff, agree = 0.0, 0
    for a, b in zip(torch_z, onnx_z):
        pa, pb = softmax(a), softmax(b)
        max_diff = max(max_diff, float(np.max(np.abs(pa - pb))))
        agree += int(np.argmax(pa) == np.argmax(pb))
    report = {"split": "heldout", "n": len(items), "max_abs_prob_diff": max_diff,
              "argmax_agreement": f"{agree}/{len(items)}", "tolerance": TOLERANCE}
    if not max_diff <= TOLERANCE:
        raise JobError(ExitCode.EXPORT, f"ONNX parity failed: max |p_torch - p_onnx| = "
                                        f"{max_diff:.3g} > {TOLERANCE} ({agree}/{len(items)} argmax agree)")
    return report
