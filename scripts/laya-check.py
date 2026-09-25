#!/usr/bin/env python3
"""`make laya-check`: the reference run of teacher/trainer coordination.

One small, known laya job (the synthetic dataset ds-a6c9c8248242, kept in the
laya store as the reference since 2026-09-25) goes through the router's
trainer port the way aiagent's does, and every hand-off between the teacher
and the trainer is checked:

  swap      a teacher was resident, and the job request evicts it and starts
            the trainer
  hold      a teacher request while the job runs is refused (503,
            Retry-After, gpu_held_by_job) -- a 200 means the router evicted
            a running job
  job       the job succeeds on the GPU
  data      the trainer read the reference dataset as before (id, split
            counts, trained tokens)
  parity    the ONNX export matches torch within the artifact's tolerance
  aiagent   the lab's aiagent accepts the artifact and reproduces every
            golden row -- the consumer side of the artifact contract
  teacher   the teacher comes back in the configuration it had before (the
            warm-up waits out the router's post-job hold per Retry-After)

GPU-EXCLUSIVE: it evicts the teacher for about three minutes (the job
~1 min, the teacher's cold start ~2 min). Requests to the teacher in that
window get 503 with Retry-After.

Nothing here talks to the network from the host: the router publishes no
host ports, so the job is driven from a container on devai-net
(scripts/laya_check/drive_job.py, in the trainer image, which has Python),
and aiagent's steps run in the lab image with aiagent's own Python
(scripts/laya_check/write_dataset.py, verify_artifact.py). Each script is
piped to `python -I -`; nothing from the repo is mounted.

The dataset is rewritten with aiagent's writer only when it is gone from
both inbox/ and datasets/; the rewrite must come out under the reference id
(its id is a hash of its content), otherwise the check stops before touching
the GPU -- a different id means aiagent's dataset output changed.

A passing run deletes its own run directory (~1.3 GB: the ONNX export and
the checkpoint); a failing one keeps it for diagnosis, as does --keep-run.
The reference run itself (runs/ftjob-f91b6d747c913af35c209ad9) is never
touched: only the directory of the job this run created is removed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent / "laya_check"

# The laya store, written out per the storage-layout rule (scripts/select-models.py).
LAYA_STORE = Path("/var/cache/devai/laya")
NETWORK = "devai-net"
ROUTER = "http://devai-router"
TRAINER_PORT = 11438
DEFAULT_TEACHER_PORT = 11437
JOB_ID_RE = re.compile(r"^ftjob-[0-9a-f]{24}$")
TERMINAL = ("succeeded", "failed", "cancelled")

# The job as it was first run. Passed explicitly so a change of the
# trainer's defaults does not silently change the reference.
HYPERPARAMETERS = {"n_epochs": 4, "batch_size": 32, "learning_rate_multiplier": 1.0,
                   "micro_batch_size": 8, "memory_mode": "lean"}
SEED = 42

# Measured on 2026-09-25 (RTX PRO 4000 Blackwell 24 GB, trainer image with
# Python 3.14.7 / torch 2.14.0+cu130, lab aiagent 0.5.0; teacher
# Qwen3.8-27B-MTP-devai-NVFP4 on vllm-devai at 118784 with MTP).
# Checked: dataset, counts, trained_tokens, golden_rows. Printed for
# comparison only (hardware-, load- and cache-dependent): the rest.
REFERENCE = {
    "measured": "2026-09-25",
    "dataset": "ds-a6c9c8248242",
    "base": "laya-multilingual",
    "counts": {"train": 50, "calib": 10, "heldout": 12, "pool": 8},
    "trained_tokens": 8172,
    "golden_rows": 12,
    "parity_max_abs_prob_diff": 5.7e-07,
    "peak_vram_gib": 3.17,
    "timings_s": {"validating": 5.46, "training": 4.13, "exporting": 12.23,
                  "calibrating": 1.82, "checking_parity": 4.91, "packaging": 2.12,
                  "total": 30.67},
    "post_s": 13.6,
    "warm_s": 126.6,
}

# aiagent's Python is whatever its launcher's shebang names (the bundle
# carries its own CPython under /opt/aiagent); ask the launcher rather than
# hard-coding the interpreter's version.
AIAGENT_PYTHON = ('exec "$(sed -n "1s/^#! *\\([^ ]*\\).*/\\1/p" /opt/aiagent/bin/aiagent)"'
                  ' -I - "$@"')

# (argv, stdin) -> (returncode, stdout). stderr is not captured: the
# containers' progress lines reach the terminal as they happen.
Runner = Callable[[list[str], str], "tuple[int, str]"]


def _run(argv: list[str], stdin: str) -> tuple[int, str]:
    p = subprocess.run(argv, input=stdin, stdout=subprocess.PIPE, text=True)
    return p.returncode, p.stdout


def lab_argv(lab_image: str, store: Path, args: list[str], *, write_inbox: bool) -> list[str]:
    """aiagent's Python in the lab image: no network, no GPU, the lab's user
    mapping, /laya read-only (inbox/ read-write only for the writer)."""
    mounts = ["-v", f"{store}:/laya:ro"]
    if write_inbox:
        mounts += ["-v", f"{store / 'inbox'}:/laya/inbox"]
    return ["podman", "run", "--rm", "-i", "--network", "none",
            "--userns=keep-id", "--user", "devai", "-e", "HOME=/tmp", *mounts,
            "--entrypoint", "/bin/sh", lab_image, "-c", AIAGENT_PYTHON, "sh", *args]


def driver_argv(trainer_image: str, cfg: dict) -> list[str]:
    return ["podman", "run", "--rm", "-i", "--network", NETWORK,
            "--entrypoint", "python3", trainer_image, "-I", "-", json.dumps(cfg)]


def _last_json(stdout: str) -> dict | None:
    for line in reversed(stdout.strip().splitlines()):
        try:
            got = json.loads(line)
        except ValueError:
            continue
        return got if isinstance(got, dict) else None
    return None


REASONING_TOKENS = ("nothink", "think", "off", "auto", "low", "medium", "high")


def parse_teacher(name: str) -> tuple[str, int | None, bool]:
    """(model, ctx or None, MTP requested) from a router model string.

    Mirrors the router's peelControlSuffixes (gpu-arbiter/main.go): trailing
    `@<ctx>`, `::mtp|nomtp` and `::<reasoning>` are peeled in any order, the
    leftmost of a repeated kind winning, until none is left.
    """
    clean, ctx, mtp = name, None, None
    while True:
        at = clean.rfind("@")
        tail = clean[at + 1:].strip() if at >= 0 else ""
        if tail.isdigit() and int(tail) > 0:
            clean, ctx = clean[:at], int(tail)
            continue
        idx = clean.rfind("::")
        token = clean[idx + 2:].strip().lower() if idx >= 0 else ""
        if token in ("mtp", "nomtp"):
            clean, mtp = clean[:idx], token == "mtp"
            continue
        if token in REASONING_TOKENS:
            clean = clean[:idx]
            continue
        return clean, ctx, bool(mtp)


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def evaluate(drive: dict, manifest: dict | None, verify: dict | None) -> list[dict]:
    """The pass/fail verdicts. Pure: everything it needs is passed in."""
    ref = REFERENCE
    out = []
    post = drive.get("post") or {}
    before = drive.get("teacher_health_before") or {}
    # Without a resident teacher the job request evicts nothing, and the
    # swap this check exists for is not exercised.
    resident = bool(before.get("running")) and bool(before.get("current_model"))
    out.append(_check(
        "swap", post.get("status") == 200 and resident,
        f"job request -> {post.get('status', 'not sent')}"
        + (f" after {post.get('seconds')} s" if "seconds" in post else "")
        + ("" if resident else "; no teacher was resident on the teacher port before it, so "
           "nothing was evicted (load the teacher first)")
        + (f": {drive.get('error')}" if drive.get("error") else "")))
    hold = drive.get("hold")
    out.append(_check(
        "hold",
        bool(hold) and hold.get("status") == 503 and hold.get("code") == "gpu_held_by_job"
        and bool(hold.get("retry_after")),
        "teacher request not sent (the job was never seen running)" if not hold else
        f"teacher request during the job -> {hold.get('status')} "
        f"code={hold.get('code')} Retry-After={hold.get('retry_after')}"))
    job = drive.get("job") or {}
    devai = job.get("devai") or {}
    out.append(_check(
        "job", job.get("status") == "succeeded" and devai.get("exit_code") == 0
        and not drive.get("timed_out"),
        f"status={job.get('status')} exit_code={devai.get('exit_code')}"
        + (" (read from the volume)" if drive.get("job_source") == "volume" else "")
        + (" (timed out, cancelled)" if drive.get("timed_out") else "")
        + (f" error={job.get('error')}" if job.get("error") else "")
        + (f" poll_error={drive.get('poll_error')}" if drive.get("poll_error") else "")
        + (f" exception={drive.get('exception')}" if drive.get("exception") else "")))
    m = manifest or {}
    ds = m.get("dataset") or {}
    tokens = (m.get("metrics") or {}).get("trained_tokens")
    out.append(_check(
        "data", ds.get("id") == ref["dataset"] and ds.get("counts") == ref["counts"]
        and tokens == ref["trained_tokens"],
        "no manifest" if not manifest else
        f"dataset={ds.get('id')} counts={ds.get('counts')} trained_tokens={tokens} "
        f"(reference {ref['dataset']}, {ref['counts']}, {ref['trained_tokens']})"))
    par = m.get("parity") or {}
    n = par.get("n")
    diff, tol = par.get("max_abs_prob_diff"), par.get("tolerance")
    out.append(_check(
        "parity", diff is not None and tol is not None and diff <= tol
        and par.get("argmax_agreement") == f"{n}/{n}",
        "no manifest" if not manifest else
        f"max_abs_prob_diff={diff} (tolerance {tol}), argmax {par.get('argmax_agreement')}"))
    v = verify or {}
    golden_n = (m.get("golden") or {}).get("n")
    out.append(_check(
        "aiagent", v.get("accepted") is True and v.get("golden") == golden_n == ref["golden_rows"],
        "not run" if not verify else
        f"accepted={v.get('accepted')} golden reproduced={v.get('golden')} of {golden_n}"
        + (f": {v.get('error')}" if v.get("error") else "")))
    warm = drive.get("warm") or {}
    after = drive.get("teacher_health_after") or {}
    teacher = drive.get("teacher") or ""
    model, ctx, mtp = parse_teacher(teacher) if teacher else ("", None, False)
    restored = (warm.get("status") == 200 and after.get("current_model") == model
                and (ctx is None or after.get("current_context") == ctx)
                and ((after.get("current_spec") or "off") != "off") == mtp)
    out.append(_check(
        "teacher", bool(teacher) and restored,
        "no teacher string" if not teacher else
        f"warm-up {warm.get('status', 'not sent')}"
        + (f" after {warm.get('seconds')} s" if "seconds" in warm else "")
        + (f", {warm.get('attempts')} attempts" if (warm.get("attempts") or 1) > 1 else "")
        + f"; now model={after.get('current_model')} context={after.get('current_context')} "
          f"spec={after.get('current_spec')} (wanted {teacher})"))
    return out


def comparison(drive: dict, manifest: dict | None) -> list[str]:
    """Measured next to the reference; informational, never a verdict."""
    ref, m = REFERENCE, manifest or {}
    t = (m.get("trainer") or {})
    lines = [f"peak VRAM (torch) {t.get('peak_vram_gib')} GiB (reference {ref['peak_vram_gib']})",
             f"parity max diff {(m.get('parity') or {}).get('max_abs_prob_diff')} "
             f"(reference {ref['parity_max_abs_prob_diff']})",
             f"job request (swap) {(drive.get('post') or {}).get('seconds')} s "
             f"(reference {ref['post_s']})",
             f"teacher warm-up {(drive.get('warm') or {}).get('seconds')} s "
             f"(reference {ref['warm_s']})"]
    timings = t.get("timings") or {}
    for phase, want in ref["timings_s"].items():
        lines.append(f"{phase:16s} {timings.get(phase)} s (reference {want})")
    return lines


def remove_run(store: Path, job_id: str) -> Path:
    """Delete runs/<job_id> -- the directory of the job this run created."""
    if not JOB_ID_RE.match(job_id or ""):
        raise ValueError(f"not a job id: {job_id!r}")
    runs = (store / "runs").resolve()
    if (runs / job_id).is_symlink():
        raise ValueError(f"{runs / job_id} is a symlink; not following it")
    target = (runs / job_id).resolve()
    if target.parent != runs or target.name != job_id:
        raise ValueError(f"{target} is not {runs / job_id}")
    for p in [target, *target.rglob("*")]:  # the trainer leaves it read-only
        if p.is_dir() and not p.is_symlink():
            p.chmod(0o755)
    shutil.rmtree(target)
    return target


def ensure_dataset(store: Path, lab_image: str, runner: Runner) -> tuple[bool, str]:
    ds_id = REFERENCE["dataset"]
    for sub in ("datasets", "inbox"):
        if (store / sub / ds_id).is_dir():
            return True, f"{ds_id} ({sub}/)"
    rc, stdout = runner(lab_argv(lab_image, store, ["/laya"], write_inbox=True),
                        (HERE / "write_dataset.py").read_text(encoding="utf-8"))
    got = _last_json(stdout) or {}
    if rc != 0 or not got.get("dataset"):
        return False, f"{ds_id} is missing and aiagent's writer failed (rc={rc})"
    if got["dataset"] != ds_id:
        return False, (f"{ds_id} is missing and aiagent's writer now produces "
                       f"{got['dataset']} from the same input: aiagent's dataset output "
                       f"changed (review it, then update REFERENCE)")
    return True, f"{ds_id} (rewritten into inbox/ by aiagent's writer)"


def base_present(store: Path) -> tuple[bool, str]:
    """The base checkpoint the reference dataset was built for, staged in base/.
    Checked before the job request: the trainer would refuse the job only
    after the router had already evicted the teacher."""
    ds_id = REFERENCE["dataset"]
    for sub in ("datasets", "inbox"):
        manifest = store / sub / ds_id / "manifest.json"
        if manifest.is_file():
            base = json.loads(manifest.read_text(encoding="utf-8"))["base_checkpoint"]
            name = f"{base['name']}@{base['revision'][:12]}"
            if (store / "base" / name).is_dir():
                return True, name
            return False, (f"base/{name} is not staged "
                           f"(make model-pull NAME={base['name']})")
    return False, f"{ds_id} has no manifest.json"


def main(argv: list[str] | None = None, *, runner: Runner | None = None,
         store: Path = LAYA_STORE) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--teacher", default="",
                    help="teacher model string (default: what the teacher port has loaded)")
    ap.add_argument("--teacher-port", type=int, default=DEFAULT_TEACHER_PORT)
    ap.add_argument("--lab-image", default="devai-lab-gpu")
    ap.add_argument("--trainer-image", default="localhost/devai-laya-trainer:latest")
    ap.add_argument("--job-timeout", type=int, default=1800, help="seconds (default 1800)")
    ap.add_argument("--keep-run", action="store_true",
                    help="keep this run's directory even when the check passes")
    args = ap.parse_args(argv)
    runner = runner or _run

    for image in (args.lab_image, args.trainer_image):
        if runner(["podman", "image", "exists", image], "")[0] != 0:
            print(f"laya-check: image {image} not found", file=sys.stderr)
            return 2
    ok, detail = ensure_dataset(store, args.lab_image, runner)
    if ok:
        ok, base = base_present(store)
        detail += f", base {base}"
    checks = [_check("dataset", ok, detail)]
    if not ok:
        return report(checks, [], None)

    print("laya-check: GPU-exclusive -- the teacher is evicted for about 3 minutes",
          file=sys.stderr, flush=True)
    cfg = {"router": ROUTER, "trainer_port": TRAINER_PORT, "teacher_port": args.teacher_port,
           "teacher": args.teacher, "base": REFERENCE["base"], "dataset": REFERENCE["dataset"],
           "suffix": "laya-check", "hyperparameters": HYPERPARAMETERS, "seed": SEED,
           "job_timeout_s": args.job_timeout, "launch_timeout_s": 900}
    rc, stdout = runner(driver_argv(args.trainer_image, cfg),
                        (HERE / "drive_job.py").read_text(encoding="utf-8"))
    drive = _last_json(stdout) or {"error": f"the job driver failed (rc={rc})"}

    job_id = drive.get("job_id") or ""
    run_dir = store / "runs" / job_id if JOB_ID_RE.match(job_id) else None
    # Polling stopped before a final status (the trainer was evicted or
    # stopped answering): the job's own record on the volume is the truth.
    if run_dir and (drive.get("job") or {}).get("status") not in TERMINAL \
            and (run_dir / "job.json").is_file():
        drive["job"] = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
        drive["job_source"] = "volume"
    manifest = verify = None
    if run_dir and (run_dir / "manifest.json").is_file():
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        rc, stdout = runner(
            lab_argv(args.lab_image, store,
                     [f"/laya/runs/{job_id}", f"/laya/datasets/{REFERENCE['dataset']}"],
                     write_inbox=False),
            (HERE / "verify_artifact.py").read_text(encoding="utf-8"))
        verify = _last_json(stdout) or {"accepted": False,
                                        "error": f"the verifier failed (rc={rc})"}
    checks += evaluate(drive, manifest, verify)
    passed = all(c["ok"] for c in checks)
    if passed and run_dir and not args.keep_run:
        remove_run(store, job_id)
        run_dir = None
    return report(checks, comparison(drive, manifest), run_dir)


def report(checks: list[dict], compare: list[str], run_dir: Path | None) -> int:
    passed = all(c["ok"] for c in checks)
    print(f"laya-check: {'PASS' if passed else 'FAIL'} "
          f"(reference {REFERENCE['dataset']}, measured {REFERENCE['measured']})")
    for c in checks:
        print(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['name']:8s} {c['detail']}")
    if compare:
        print("  measured vs reference (informational):")
        for line in compare:
            print(f"    {line}")
    if run_dir:
        print(f"  run kept: {run_dir}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
