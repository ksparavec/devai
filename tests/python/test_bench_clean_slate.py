"""scripts/bench/clean_slate.py: a clean GPU before every model's benchmark.

Owner's rule (2026-09-28): stop any other bench, stop every engine, kill
whatever still holds the GPU, and verify with the host's ps and nvidia-smi,
not podman. Stdlib unittest; ps, signals, podman and nvidia-smi are faked.
"""

from __future__ import annotations

import signal
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "bench"))

import clean_slate as C  # noqa: E402

LINES = {
    "runner": "python3 /scripts/bench/bench_runner.py --backend vllm-devai --tasks gpqa",
    "orch": "python3 scripts/bench-sync.py --repo x",
    "ollama_runner": "/usr/lib/ollama/llama-server --model /root/.ollama/models/blobs/sha256-3f",
    "vllm": "python3 -m vllm.entrypoints.openai.api_server --model /models/m",
    "vllm_core": "VLLM::EngineCore",
    "sglang": "python3 -m sglang.launch_server --model-path /models/m",
    "laya": "python -m laya_trainer.controller --port 11434",
    # must never match
    "editor": "vim scripts/bench-sync.py",
    "grep": "grep bench_runner.py notes.txt",
    "logger": "podman --remote logs --follow --timestamps devai-vllm",
    "pager": "less laya-trainer/laya_trainer/controller.py",
    "ollama_serve": "/bin/ollama serve",
    "placeholder": "sleep infinity",
    "client": "podman run --rm --entrypoint python3 devai-lab-gpu /scripts/bench/bench_runner.py",
}


def _table():
    return [(i, 1000 + i, args) for i, args in enumerate(LINES.values(), start=10)]


def _names(pids):
    by_pid = {pid: name for pid, name in zip(range(10, 10 + len(LINES)), LINES)}
    return sorted(by_pid[p] for p in pids)


class PatternTest(unittest.TestCase):
    def test_bench_runner_is_the_container_process_only(self) -> None:
        self.assertEqual(_names(C.bench_runner_pids(_table())), ["runner"])

    def test_engines_are_matched_by_their_command(self) -> None:
        self.assertEqual(_names(C.engine_pids(_table())),
                         ["laya", "ollama_runner", "sglang", "vllm", "vllm_core"])

    def test_other_orchestrators_are_process_groups_not_our_own(self) -> None:
        t = _table()
        orch_pgid = next(g for _, g, a in t if a == LINES["orch"])
        self.assertEqual(C.other_orchestrator_groups(t, own_pgid=1), [orch_pgid])
        self.assertEqual(C.other_orchestrator_groups(t, own_pgid=orch_pgid), [])


class KillTest(unittest.TestCase):
    def test_term_first_then_kill_for_what_ignores_it(self) -> None:
        sent = []
        alive = {5: True}

        def fake_kill(pid, sig):
            sent.append((pid, sig))
            if sig == signal.SIGKILL:
                alive[pid] = False

        with mock.patch.object(C.os, "kill", side_effect=fake_kill), \
                mock.patch.object(C, "_alive", side_effect=lambda p: alive.get(p, False)), \
                mock.patch.object(C.time, "sleep"):
            left = C.kill_hard([5], grace_s=0.0)
        self.assertEqual(sent, [(5, signal.SIGTERM), (5, signal.SIGKILL)])
        self.assertEqual(left, [])

    def test_bench_runners_get_sigint_first(self) -> None:
        # PID 1 in a container ignores SIGTERM; Python handles SIGINT, and
        # inspect then writes a `cancelled` log with every finished question.
        sent = []
        alive = {7: True}

        def fake_kill(pid, sig):
            sent.append(sig)
            if sig == signal.SIGINT:
                alive[pid] = False

        with mock.patch.object(C, "ps_table", return_value=[(7, 7, LINES["runner"])]), \
                mock.patch.object(C.os, "kill", side_effect=fake_kill), \
                mock.patch.object(C, "_alive", side_effect=lambda p: alive.get(p, False)), \
                mock.patch.object(C.time, "sleep"):
            self.assertEqual(C.stop_bench_runners(grace_s=5.0), [])
        self.assertEqual(sent, [signal.SIGINT])


class CleanSlateTest(unittest.TestCase):
    def _run(self, gpu_pids_seq, used_seq):
        calls = []
        gpu = iter(gpu_pids_seq)
        used = iter(used_seq)
        with mock.patch.object(C, "ps_table", return_value=[]), \
                mock.patch.object(C, "stop_engine_containers", side_effect=lambda rt: calls.append(("stop", rt))), \
                mock.patch.object(C, "kill_hard", side_effect=lambda pids, grace_s=C.GRACE_S: calls.append(("kill", sorted(pids))) or []), \
                mock.patch.object(C, "gpu_compute_pids", side_effect=lambda: next(gpu)), \
                mock.patch.object(C, "gpu_used_mib", side_effect=lambda: next(used)), \
                mock.patch.object(C, "SETTLE_S", 0.0), \
                mock.patch.object(C.time, "sleep"), \
                mock.patch.object(C, "_say"):
            result = C.clean_slate(runtime="podman")
        return calls, result

    def test_engines_stopped_and_gpu_holders_killed(self) -> None:
        calls, result = self._run(gpu_pids_seq=[[4242], []], used_seq=[3])
        self.assertEqual(calls, [("stop", "podman"), ("kill", [4242])])
        self.assertEqual(result["gpu_used_mib"], 3)

    def test_a_gpu_that_stays_busy_is_an_error(self) -> None:
        with self.assertRaises(C.CleanSlateError):
            self._run(gpu_pids_seq=[[4242], [4242], [4242]], used_seq=[20000, 20000])

    def test_vram_above_idle_is_an_error_even_without_a_process(self) -> None:
        with self.assertRaises(C.CleanSlateError):
            self._run(gpu_pids_seq=[[], [], []], used_seq=[C.VRAM_IDLE_MIB + 1] * 3)

    def test_every_engine_container_is_named(self) -> None:
        self.assertEqual(set(C.ENGINE_CONTAINERS),
                         {"devai-ollama", "devai-vllm", "devai-vllm-devai", "devai-sglang",
                          "devai-laya-trainer"})


if __name__ == "__main__":
    unittest.main()
