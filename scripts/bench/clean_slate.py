#!/usr/bin/env python3
"""Clean slate before each model's benchmark (owner's rule, 2026-09-28).

Before every model is benched:

  1. any other bench run is stopped -- another bench-sync process group
     (orchestrator, its make and its podman client) and every bench
     container process (`bench_runner.py`);
  2. every inference engine is stopped (the containers below), and
     whatever still holds the GPU afterwards is killed too -- including a
     laya-trainer job or anything else, by the owner's choice;
  3. the result is VERIFIED with the host's own `ps` and `nvidia-smi`, not
     with `podman ps`: no bench_runner process, no engine process, no GPU
     compute process, VRAM back to idle. If that cannot be reached, the
     bench does not start (CleanSlateError).

Stopping is SIGTERM, then SIGKILL after GRACE_S -- whatever does not stop
gracefully is killed hard. The router notices stopped engines on its own
(backendVanished: a container that is not running, or an engine that does
not answer /health) and recreates the right one on the next request, so the
first request of each model pays one cold start.

Why: on 2026-09-27/28 an interrupted bench left its container running and a
second bench ran beside it for nine hours. The router serves one model at a
time, so each evicted the other's model, requests failed, and both rewrote
the cache from their own copies. See docs/bench-results.md.

Usage (host, not in a container): python3 scripts/bench/clean_slate.py
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time

ENGINE_CONTAINERS = ("devai-ollama", "devai-vllm", "devai-vllm-devai",
                     "devai-sglang", "devai-laya-trainer")
# Patterns are anchored on the COMMAND (argv[0], or the interpreter plus its
# script), so an editor or a grep that merely names one of these files is
# never matched -- and never killed.
_PY = r"^\S*python[\d.]*\s+(?:-\S+\s+)*"
BENCH_RUNNER_RX = re.compile(_PY + r"\S*bench_runner\.py\b")
ORCHESTRATOR_RX = re.compile(_PY + r"\S*bench-sync\.py\b")
# Engine processes as the host's ps shows them (rootless containers run as
# ordinary host processes). Ollama's model runner is llama-server; `ollama
# serve` itself holds no VRAM and goes with its container.
ENGINE_RXS = (
    re.compile(r"^\S*llama-server\b"),
    re.compile(_PY + r"-m\s+vllm\.entrypoints|^\S*vllm\s+serve\b|^VLLM::"),
    re.compile(_PY + r"-m\s+sglang\.launch_server|^sglang::"),
    re.compile(_PY + r"-m\s+laya_trainer|" + _PY + r"\S*laya_trainer/"),
)
GRACE_S = 10.0
SETTLE_S = 60.0         # how long VRAM may take to drain after the last kill
VRAM_IDLE_MIB = 512     # an idle card here reports a few MiB used


class CleanSlateError(RuntimeError):
    """The GPU could not be brought to a clean slate; do not bench."""


def _say(msg: str) -> None:
    print(f"  [clean-slate] {msg}", file=sys.stderr, flush=True)


def ps_table() -> list[tuple[int, int, str]]:
    """(pid, pgid, args) of every host process."""
    out = subprocess.run(["ps", "-eo", "pid=,pgid=,args="], capture_output=True,
                         text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            rows.append((int(parts[0]), int(parts[1]), parts[2] if len(parts) > 2 else ""))
    return rows


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def kill_hard(pids: list[int], grace_s: float = GRACE_S) -> list[int]:
    """SIGTERM, wait up to grace_s, SIGKILL whatever is left. Returns survivors."""
    pids = [p for p in dict.fromkeys(pids) if p != os.getpid()]
    for p in pids:
        _signal(p, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and any(_alive(p) for p in pids):
        time.sleep(0.25)
    left = [p for p in pids if _alive(p)]
    for p in left:
        _signal(p, signal.SIGKILL)
    time.sleep(1.0)
    return [p for p in left if _alive(p)]


def stop_bench_runners(grace_s: float = 30.0) -> list[int]:
    """SIGINT every bench_runner process, SIGKILL whatever is left after grace_s.

    SIGINT, not SIGTERM: the runner is PID 1 in its container, and PID 1
    gets only the signals it has a handler for -- Python handles SIGINT
    (KeyboardInterrupt) and not SIGTERM, which is why `podman stop` waited
    out its timeout on 2026-09-28. On SIGINT inspect cancels the eval and
    writes a `cancelled` log holding every finished question (checked with
    inspect 0.3.271: exit in ~2 s); SIGKILL loses the ones not yet flushed.
    Returns the pids that survived SIGKILL.
    """
    pids = bench_runner_pids(ps_table())
    for p in pids:
        _signal(p, signal.SIGINT)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and any(_alive(p) for p in pids):
        time.sleep(0.25)
    left = [p for p in pids if _alive(p)]
    for p in left:
        _signal(p, signal.SIGKILL)
    time.sleep(1.0)
    return [p for p in left if _alive(p)]


def other_orchestrator_groups(table, own_pgid: int) -> list[int]:
    """Process groups of OTHER bench-sync runs (their wrapper, make, podman client)."""
    return sorted({pgid for pid, pgid, args in table
                   if ORCHESTRATOR_RX.search(args) and pgid != own_pgid and pid != os.getpid()})


def bench_runner_pids(table) -> list[int]:
    return [pid for pid, _, args in table if BENCH_RUNNER_RX.search(args) and pid != os.getpid()]


def engine_pids(table) -> list[int]:
    return [pid for pid, _, args in table if any(rx.search(args) for rx in ENGINE_RXS)]


def gpu_compute_pids() -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    return [int(x) for x in out.split() if x.strip().isdigit()]


def gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return max(int(x) for x in out.split() if x.strip().isdigit())


def kill_process_groups(pgids: list[int], grace_s: float = GRACE_S) -> None:
    for g in pgids:
        try:
            os.killpg(g, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not any(pgid in pgids for _, pgid, _ in ps_table()):
            return
        time.sleep(0.25)
    for g in pgids:
        try:
            os.killpg(g, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    time.sleep(1.0)


def stop_engine_containers(runtime: str) -> None:
    """podman stop = SIGTERM, then SIGKILL after -t seconds. Its answer is
    not trusted; the ps / nvidia-smi checks below decide."""
    subprocess.run([runtime, "stop", "-t", str(int(GRACE_S)), *ENGINE_CONTAINERS],
                   capture_output=True, text=True)


def state() -> dict:
    table = ps_table()
    return {"bench_runner": bench_runner_pids(table), "engines": engine_pids(table),
            "gpu_pids": gpu_compute_pids(), "gpu_used_mib": gpu_used_mib()}


def is_clean(s: dict) -> bool:
    return (not s["bench_runner"] and not s["engines"] and not s["gpu_pids"]
            and s["gpu_used_mib"] <= VRAM_IDLE_MIB)


def clean_slate(runtime: str | None = None) -> dict:
    """Bring the host to a clean slate or raise CleanSlateError. Returns the final state."""
    runtime = runtime or os.environ.get("CONTAINER_RUNTIME") or "podman"
    own_pgid = os.getpgid(0)

    groups = other_orchestrator_groups(ps_table(), own_pgid)
    if groups:
        _say(f"stopping other bench runs (process groups {groups})")
        kill_process_groups(groups)

    runners = bench_runner_pids(ps_table())
    if runners:
        _say(f"stopping bench containers' processes {runners}")
        left = stop_bench_runners(grace_s=GRACE_S)
        if left:
            raise CleanSlateError(f"bench_runner processes survived SIGKILL: {left}")

    _say("stopping inference engines: " + ", ".join(ENGINE_CONTAINERS))
    stop_engine_containers(runtime)

    leftovers = engine_pids(ps_table()) + gpu_compute_pids()
    if leftovers:
        _say(f"killing processes still running an engine or holding the GPU: {sorted(set(leftovers))}")
        kill_hard(leftovers)

    deadline = time.monotonic() + SETTLE_S
    s = state()
    while not is_clean(s) and time.monotonic() < deadline:
        time.sleep(1.0)
        s = state()
    if not is_clean(s):
        raise CleanSlateError(f"no clean slate after {SETTLE_S:.0f} s: {s}")
    _say(f"clean: no bench or engine process, no GPU process, {s['gpu_used_mib']} MiB used")
    return s


def main() -> int:
    try:
        clean_slate()
    except (CleanSlateError, subprocess.CalledProcessError, OSError) as e:
        print(f"clean-slate FAILED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
