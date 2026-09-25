"""Drive the reference laya job through the router and restore the teacher.

Runs INSIDE a container on devai-net -- the router publishes no host ports --
piped to `python3 -I -` by scripts/laya-check.py. Standard library only.
Progress goes to stderr; the LAST stdout line is the JSON result, which the
host evaluates. This script decides nothing: it records what the router and
the trainer answered, and it always prints that record, whatever went wrong.

Sequence:
  1. Resolve the teacher: the given model string, else what the teacher port's
     /health reports as loaded (model, context, MTP spec). Nothing known and
     nothing given, or a job already running on the trainer: stop before
     evicting anything.
  2. POST the job to the trainer port. The router drains and stops the
     teacher and starts the trainer inside this request.
  3. Poll the job. The first time it is `running`, send one teacher request:
     the router must refuse it with 503, Retry-After and gpu_held_by_job.
     A 200 there means the router evicted the running job; polling stops, and
     the host reads the job's final record from the volume. Polling also
     stops after MAX_POLL_FAILURES answers in a row that are not the job.
  4. Whatever happened after the job request -- success, failure, timeout
     (which cancels the job), an exception -- send the teacher request again
     as the warm-up and read the teacher port's /health: the teacher must be
     back in the configuration of step 1. The router keeps a hold verdict for
     Retry-After seconds after the job has ended, so a 503 gpu_held_by_job is
     waited out per Retry-After, within launch_timeout_s. The only exception
     is a job request refused with 409: another job holds the trainer, and
     nothing of ours evicted the teacher.
"""

import json
import sys
import time
import urllib.error
import urllib.request

POLL_S = 2
MAX_POLL_FAILURES = 5
TERMINAL = ("succeeded", "failed", "cancelled")
HEALTH_KEYS = ("running", "current_model", "current_context", "current_spec")
HOLD_CODE = "gpu_held_by_job"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def call(method: str, url: str, body=None, timeout: float = 60):
    """(status, headers, parsed body); a transport failure is status 0."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.headers, _json(resp.read())
    except urllib.error.HTTPError as e:
        with e:
            return e.code, e.headers, _json(e.read())
    except (urllib.error.URLError, OSError) as e:
        return 0, {}, {"error": {"message": f"{type(e).__name__}: {e}"}}


def _json(raw: bytes):
    try:
        return json.loads(raw or b"{}")
    except ValueError:
        return {"raw": raw[:300].decode(errors="replace")}


def error_code(body):
    """`error.code`, or None. The router also writes `{"error": "<text>"}`."""
    err = body.get("error") if isinstance(body, dict) else None
    return err.get("code") if isinstance(err, dict) else None


def retry_after(headers) -> int:
    try:
        return min(60, max(1, int(headers.get("Retry-After") or 30)))
    except (TypeError, ValueError):
        return 30


def teacher_from_health(health: dict):
    """The model string that relaunches what /health reports, or None."""
    model = health.get("current_model") or ""
    if not health.get("running") or not model:
        return None
    name = model
    if (health.get("current_spec") or "off") != "off":
        name += "::mtp"
    ctx = health.get("current_context") or 0
    if ctx > 0:
        name += f"@{ctx}"
    return name


def chat(url: str, teacher: str, max_tokens: int, timeout: float):
    return call("POST", url, {"model": teacher, "max_tokens": max_tokens,
                              "messages": [{"role": "user", "content": "Say OK."}]},
                timeout=timeout)


def health_of(url: str) -> dict:
    s, _, h = call("GET", url, timeout=30)
    if s != 200 or not isinstance(h, dict):
        return {"error": s}
    return h


def follow(cfg: dict, out: dict, trainer: str, teacher_chat: str, teacher: str) -> None:
    """Poll the job to a terminal status; probe the hold once while it runs."""
    jid = out["job_id"]
    deadline = time.monotonic() + cfg["job_timeout_s"]
    failures, seen = 0, None
    while True:
        s, _, got = call("GET", f"{trainer}/v1/fine_tuning/jobs/{jid}")
        if s == 200 and isinstance(got, dict) and got.get("id") == jid:
            out["job"], failures = got, 0
        else:
            failures += 1
            if failures >= MAX_POLL_FAILURES:
                out["poll_error"] = {"status": s, "body": got}
                log(f"job {jid}: {failures} polls in a row answered {s}; stopped polling")
                return
        job = out["job"]
        state = (job.get("status"), (job.get("devai") or {}).get("phase"))
        if state != seen:
            log(f"job {jid}: {state[0]} / {state[1]}")
            seen = state
        if job.get("status") in TERMINAL:
            return
        if out["hold"] is None and job.get("status") == "running":
            hs, hh, hj = chat(teacher_chat, teacher, 1, cfg["launch_timeout_s"])
            out["hold"] = {"status": hs, "retry_after": hh.get("Retry-After"),
                           "code": error_code(hj)}
            log(f"teacher request during the job -> {hs} {out['hold']}")
            if hs == 200:
                out["poll_error"] = {"status": None, "body": "the router served the teacher "
                                     "while the job was running: the job was evicted"}
                return
        if time.monotonic() > deadline:
            out["timed_out"] = True
            log(f"job still {job.get('status')} after {cfg['job_timeout_s']} s; cancelling")
            call("POST", f"{trainer}/v1/fine_tuning/jobs/{jid}/cancel", timeout=60)
            return
        time.sleep(POLL_S)


def restore_teacher(cfg: dict, out: dict, teacher_chat: str, teacher: str) -> None:
    t0 = time.monotonic()
    budget = t0 + cfg["launch_timeout_s"]
    attempts = 0
    while True:
        attempts += 1
        ws, wh, wj = chat(teacher_chat, teacher, 8, cfg["launch_timeout_s"])
        if ws == 503 and error_code(wj) == HOLD_CODE and time.monotonic() < budget:
            wait = retry_after(wh) + 1
            log(f"teacher warm-up -> 503 {HOLD_CODE}; retrying in {wait} s")
            time.sleep(wait)
            continue
        break
    out["warm"] = {"status": ws, "seconds": round(time.monotonic() - t0, 1),
                   "attempts": attempts}
    if ws != 200:
        out["warm"]["body"] = wj
    log(f"teacher warm-up -> {ws} after {attempts} attempt(s), {out['warm']['seconds']} s")


def main(cfg: dict) -> dict:
    trainer = f"{cfg['router']}:{cfg['trainer_port']}"
    teacher_base = f"{cfg['router']}:{cfg['teacher_port']}"
    teacher_chat = f"{teacher_base}/v1/chat/completions"
    before = health_of(f"{teacher_base}/health")
    teacher = cfg.get("teacher") or teacher_from_health(before)
    out = {"teacher": teacher,
           "teacher_health_before": {k: before.get(k) for k in HEALTH_KEYS}}
    if not teacher:
        out["error"] = ("no teacher string: nothing is loaded on the teacher port and "
                        "none was given (TEACHER=...)")
        return out
    runner = (health_of(f"{trainer}/health").get("job_runner") or {})
    if runner.get("status") == "busy":
        out["error"] = f"a job is already running on the trainer: {runner.get('job')}"
        return out

    restore = True
    try:
        t0 = time.monotonic()
        s, _, job = call("POST", f"{trainer}/v1/fine_tuning/jobs", {
            "model": cfg["base"], "training_file": cfg["dataset"], "suffix": cfg["suffix"],
            "hyperparameters": cfg["hyperparameters"], "seed": cfg["seed"]},
            timeout=cfg["launch_timeout_s"])
        out["post"] = {"status": s, "seconds": round(time.monotonic() - t0, 1)}
        log(f"POST job -> {s} in {out['post']['seconds']} s")
        if s == 200 and isinstance(job, dict) and job.get("id"):
            out.update(job_id=job["id"], job=job, hold=None)
            follow(cfg, out, trainer, teacher_chat, teacher)
            out["job_seconds"] = round(time.monotonic() - t0, 1)
            s, _, ev = call("GET", f"{trainer}/v1/fine_tuning/jobs/{out['job_id']}/events?limit=100")
            if s == 200 and isinstance(ev, dict):
                out["events"] = [[e.get("created_at"), e.get("message")]
                                 for e in reversed(ev.get("data") or [])]
        else:
            out["post"]["body"] = job
            restore = s != 409
    except Exception as e:  # recorded; the teacher is still restored below
        out["exception"] = f"{type(e).__name__}: {e}"
    if restore:
        try:
            restore_teacher(cfg, out, teacher_chat, teacher)
        except Exception as e:
            out.setdefault("exception", f"{type(e).__name__}: {e}")
    after = health_of(f"{teacher_base}/health")
    out["teacher_health_after"] = {k: after.get(k) for k in HEALTH_KEYS}
    return out


if __name__ == "__main__":
    try:
        result = main(json.loads(sys.argv[1]))
    except Exception as e:
        result = {"error": f"{type(e).__name__}: {e}"}
    print(json.dumps(result))
