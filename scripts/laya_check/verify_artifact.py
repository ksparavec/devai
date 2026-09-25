"""Verify a laya trainer artifact with aiagent's own verifier.

Runs INSIDE the lab image, with aiagent's bundled Python, piped to
`python -I -` by scripts/laya-check.py. Arguments: the run directory and
the dataset directory (both under /laya, read-only). Prints one JSON line.

This is the consumer side of the artifact contract: `verify_artifact`
checks the manifest, the hash bindings (the dataset's producer binds and the
dataset manifest's sha256) and every file hash; `check_golden` runs the
student with aiagent's own runtime and compares it with golden.jsonl.
A rejection is reported, not raised, so the host sees why.
"""

import json
import sys
from pathlib import Path

from aiagent.system1 import artifacts as A
from aiagent.system1.contract import Binds, file_sha256


def main(run: Path, ds: Path) -> dict:
    manifest = json.loads((ds / "manifest.json").read_text(encoding="utf-8"))
    binds = manifest["producer"]["binds"]
    try:
        v = A.verify_artifact(
            run,
            expected=Binds(signature_sha256=binds["signature_sha256"],
                           question_set_sha256=binds["question_set_sha256"],
                           skill_source_sha256=binds["skill_source_sha256"]),
            dataset_manifest_sha256=file_sha256(ds / "manifest.json"),
            max_len=manifest["max_len"], head_max_len=manifest["head_max_len"])
    except Exception as e:  # the verifier's refusal is the result
        return {"accepted": False, "error": f"{type(e).__name__}: {e}"}
    try:
        golden = A.check_golden(v.runtime, run / "golden.jsonl")
    except Exception as e:
        return {"accepted": True, "artifact_id": v.manifest.artifact_id,
                "golden": None, "error": f"{type(e).__name__}: {e}"}
    return {"accepted": True, "artifact_id": v.manifest.artifact_id, "golden": golden,
            "error": None}


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1]), Path(sys.argv[2]))))
