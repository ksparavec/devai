"""deploy/slurm/devai-gpu and the GPU guard's decisions, with DEVAI_GPU_FAKE
(platform-rebuild Phase 5, step 6; docs/slurm.md Sec. 6).

The guard runs in devai-slurm as root with --pid=host. Here NVML is a JSON
file (DEVAI_GPU_FAKE), /proc a directory tree (Guard's Proc root), the bus a
recording fake and kill() a fake that makes a holder vanish -- so every
decision is checked without a card. The sampler tests run the real CLI as
subprocesses, with a stub devai-bus.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "deploy" / "slurm" / "devai-gpu"


def load_module():
    loader = importlib.machinery.SourceFileLoader("devai_gpu", str(SCRIPT))
    spec = importlib.util.spec_from_loader("devai_gpu", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["devai_gpu"] = mod  # dataclasses look their module up here
    keep = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # no __pycache__ in deploy/slurm (the image's build context)
    try:
        loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = keep
    return mod


g = load_module()

ENGINES_SCOPE = "0::/user.slice/user-1000.slice/user@1000.service/user.slice/libpod-" + "a" * 64 + ".scope/container"
OPERATOR_SCOPE = "0::/user.slice/user-1000.slice/user@1000.service/user.slice/libpod-" + "b" * 64 + ".scope/container"
TRAINER_SCOPE = "0::/user.slice/user-1000.slice/user@1000.service/user.slice/libpod-" + "c" * 64 + ".scope/container"
OTHER_SCOPE = "0::/user.slice/user-1000.slice/user@1000.service/user.slice/libpod-" + "d" * 64 + ".scope/container"


class World:
    """A fake card, /proc, bus and clock."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.nvml_file = tmp / "nvml.json"
        self.proc_root = tmp / "proc"
        self.proc_root.mkdir()
        self.card = {"present": True, "used_mib": 2, "util_pct": 0, "energy_mj": 1000, "processes": []}
        self.save()
        self.requests: list[tuple[str, dict]] = []
        self.published: list[tuple[str, dict]] = []
        self.kills: list[tuple[int, int]] = []
        self.states: dict[str, dict] = {}
        # what a bus request does to the card: subject suffix -> pids it frees
        self.frees: dict[str, list[int]] = {}
        self.unkillable: set[int] = set()
        self.now = 1000.0

    def save(self):
        self.nvml_file.write_text(json.dumps(self.card))

    def process(self, pid, cgroup, cmdline=("python3", "x"), uid=0, environ=None, used=0):
        d = self.proc_root / str(pid)
        d.mkdir()
        (d / "cgroup").write_text(cgroup + "\n")
        (d / "cmdline").write_bytes(b"\0".join(c.encode() for c in cmdline) + b"\0")
        (d / "status").write_text(f"Name:\tx\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
        env = environ or {}
        (d / "environ").write_bytes(b"\0".join(f"{k}={v}".encode() for k, v in env.items()))
        if used:
            self.card["processes"].append({"pid": pid, "used_mib": used})
            self.card["used_mib"] += used
            self.save()

    def free(self, pid):
        for p in list(self.card["processes"]):
            if p["pid"] == pid:
                self.card["processes"].remove(p)
                self.card["used_mib"] -= p["used_mib"]
        self.save()
        d = self.proc_root / str(pid)
        if d.exists():
            for f in d.iterdir():
                f.unlink()
            d.rmdir()

    # the injected parts
    def kill(self, pid, sig):
        self.kills.append((pid, sig))
        if pid in self.unkillable:
            raise PermissionError(1, "Operation not permitted")
        if sig in (signal.SIGTERM, signal.SIGKILL):
            self.free(pid)

    def sleep(self, s):
        self.now += s

    def clock(self):
        return self.now


class FakeBus:
    def __init__(self, world: World, job: int):
        self.w = world
        self.job = job

    def request(self, subject, body, timeout_s):
        self.w.requests.append((subject, body))
        svc_verb = subject.split(".", 1)[1]
        for pid in self.w.frees.get(svc_verb, []):
            self.w.free(pid)
        if subject.endswith(".state"):
            svc = subject.split(".")[2]
            if svc in self.w.states:
                return {"v": 1, "ok": True, "state": self.w.states[svc]}
            return None
        return {"v": 1, "ok": True, "state": {}}

    def publish(self, subject, body):
        self.w.published.append((subject, body))


class GuardDecisions(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.w = World(Path(self._tmp.name))
        os.environ["DEVAI_GUARD_ENGINE_SERVICES"] = "fake vllm"

    def tearDown(self):
        self._tmp.cleanup()
        os.environ.pop("DEVAI_GUARD_ENGINE_SERVICES", None)

    def guard(self, kind="engine", backend="fake", gpu=False, job=42, licenses="engine"):
        info = g.JobInfo(job=job, kind=kind, backend=backend, gpu=gpu, licenses=licenses)
        return g.Guard(g.FakeNVML(str(self.w.nvml_file)), g.Proc(str(self.w.proc_root)), FakeBus(self.w, job), info,
                       kill=self.w.kill, sleep=self.w.sleep, clock=self.w.clock)

    def control(self, pid, container, scope):
        self.w.process(pid, scope, cmdline=("/usr/local/bin/devai-control", "--container", container,
                                            "--config", "/etc/devai/control.yaml"))

    def test_idle_card(self):
        rc, v = self.guard().run()
        self.assertEqual(rc, 0)
        self.assertEqual(v["verdict"], "idle")
        self.assertEqual(self.w.kills, [])

    def test_devai_holder_is_asked_to_unload(self):
        self.control(100, "engines", ENGINES_SCOPE)
        self.w.process(101, ENGINES_SCOPE, used=2048)
        self.w.states["fake"] = {"status": "serving", "job": 7, "sleeper": None}
        self.w.frees["engines.fake.unload"] = [101]
        rc, v = self.guard().run()
        self.assertEqual(rc, 0, v)
        self.assertEqual(v["verdict"], "unloaded")
        self.assertIn(("devai.engines.fake.unload", {"force": True}), self.w.requests)
        self.assertEqual(self.w.kills, [], "a devai engine is asked, never killed")

    def test_foreign_holder_is_killed(self):
        self.control(100, "engines", ENGINES_SCOPE)
        self.w.process(200, OTHER_SCOPE, used=4000)
        rc, v = self.guard().run()
        self.assertEqual(rc, 0, v)
        self.assertEqual(v["verdict"], "killed")
        self.assertEqual(self.w.kills[0], (200, signal.SIGTERM))

    def test_holder_of_another_host_user_fails_the_prolog(self):
        self.w.process(300, OTHER_SCOPE, uid=65534, used=2048)
        self.w.unkillable.add(300)
        rc, v = self.guard().run()
        self.assertEqual(rc, 1)
        self.assertEqual(v["verdict"], "busy")
        self.assertIn("another host user", v["reason"])

    def test_a_mapped_nobody_is_killed_not_mistaken_for_another_user(self):
        self.w.process(301, OTHER_SCOPE, uid=65534, used=2048)
        rc, v = self.guard().run()
        self.assertEqual((rc, v["verdict"]), (0, "killed"))

    def test_stale_holder_in_operator_is_stopped_then_killed(self):
        self.control(110, "operator", OPERATOR_SCOPE)
        self.w.process(111, OPERATOR_SCOPE, environ={"DEVAI_JOB_ID": "77"}, used=3000)
        rc, v = self.guard(kind="probe", backend="vllm").run()
        self.assertEqual(rc, 0, v)
        stop = [b for s, b in self.w.requests if s == "devai.operator.actions.stop"]
        self.assertEqual(stop, [{"force": True, "run_job": 77, "signal": "TERM", "grace_s": 5}])
        self.assertIn((111, signal.SIGTERM), self.w.kills, "still there after the stop: killed")
        self.assertEqual(v["holders"][0]["action"], "stop+terminated")

    def test_stale_holder_that_the_stop_ends_is_not_killed(self):
        self.control(110, "operator", OPERATOR_SCOPE)
        self.w.process(112, OPERATOR_SCOPE, environ={"DEVAI_JOB_ID": "78"}, used=3000)
        self.w.frees["operator.actions.stop"] = [112]
        rc, v = self.guard(kind="action", gpu=True, backend="").run()
        self.assertEqual((rc, v["verdict"]), (0, "unloaded"))
        self.assertEqual(self.w.kills, [])

    def test_operator_holder_without_job_id_is_foreign(self):
        self.control(110, "operator", OPERATOR_SCOPE)
        self.w.process(113, OPERATOR_SCOPE, used=3000)
        rc, v = self.guard().run()
        self.assertEqual((rc, v["verdict"]), (0, "killed"))
        self.assertEqual([s for s, _ in self.w.requests if "actions" in s], [])

    def test_two_controls_in_one_cgroup_is_a_configuration_error(self):
        self.control(120, "engines", ENGINES_SCOPE)
        self.control(121, "operator", ENGINES_SCOPE)
        self.w.process(122, ENGINES_SCOPE, used=1000)
        rc, v = self.guard().run()
        self.assertEqual(rc, 1)
        self.assertIn("configuration error", v["reason"])

    def test_trainer_holder_gets_a_forced_cancel(self):
        self.control(130, "laya-trainer", TRAINER_SCOPE)
        self.w.process(131, TRAINER_SCOPE, used=3000)
        self.w.frees["laya-trainer.trainer.cancel"] = [131]
        rc, v = self.guard().run()
        self.assertEqual((rc, v["verdict"]), (0, "unloaded"))
        cancels = [b for s, b in self.w.requests if s == "devai.laya-trainer.trainer.cancel"]
        self.assertTrue(cancels and cancels[0]["force"])

    def test_sleeper_is_tolerated_by_an_engine_job(self):
        self.control(100, "engines", ENGINES_SCOPE)
        self.w.process(101, ENGINES_SCOPE, used=1500)
        self.w.states["vllm"] = {"status": "sleeping", "job": None, "sleeper": {"job": 3, "used_mib": 1500}}
        rc, v = self.guard().run()
        self.assertEqual((rc, v["verdict"]), (0, "idle"))
        self.assertEqual([s for s, _ in self.w.requests if s.endswith(".unload")], [])

    def test_probe_job_unloads_the_sleeper_first(self):
        self.control(100, "engines", ENGINES_SCOPE)
        self.w.process(101, ENGINES_SCOPE, used=1500)
        self.w.states["vllm"] = {"status": "sleeping", "job": None, "sleeper": {"job": 3, "used_mib": 1500}}
        self.w.frees["engines.vllm.unload"] = [101]
        rc, v = self.guard(kind="probe", backend="vllm").run()
        self.assertIn(("devai.engines.vllm.unload", {"target": "sleeper", "force": True}), self.w.requests)
        self.assertEqual(rc, 0, v)

    def test_bench_and_plain_action_jobs_are_not_gpu_jobs(self):
        for kind, lic, gpu in (("bench", "bench", False), ("action", "", False)):
            info = g.JobInfo(job=1, kind=kind, licenses=lic, gpu=gpu)
            self.assertFalse(info.gpu_job, kind)
        self.assertTrue(g.JobInfo(job=1, kind="action", gpu=True, licenses="engine").gpu_job)
        self.assertTrue(g.JobInfo(job=1, kind="trainer").gpu_job)

    def test_job_info_reads_slurmds_environment(self):
        os.environ["SLURM_JOB_COMMENT"] = json.dumps({"v": 1, "kind": "engine", "backend": "fake"})
        os.environ["SLURM_JOB_LICENSES"] = "engine"
        try:
            info = g.job_info(5)
        finally:
            del os.environ["SLURM_JOB_COMMENT"], os.environ["SLURM_JOB_LICENSES"]
        self.assertEqual((info.kind, info.backend, info.gpu_job), ("engine", "fake", True))


def stub_bus(dirpath: Path) -> Path:
    """A devai-bus that logs its arguments and answers ok."""
    log = dirpath / "bus.log"
    stub = dirpath / "devai-bus"
    stub.write_text(f"""#!/bin/sh
printf '%s\\n' "$*" >> {log}
case "$2" in
  *.state) echo '{{"v":1,"ok":true,"state":{{"status":"idle","job":null}}}}' ;;
  *) echo '{{"v":1,"ok":true,"state":{{}}}}' ;;
esac
""")
    stub.chmod(0o755)
    return log


class SamplerAndEpilog(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        t = Path(self._tmp.name)
        self.t = t
        self.nvml = t / "nvml.json"
        self.card(used=2, energy=5_000_000)
        bindir = t / "bin"
        bindir.mkdir()
        self.buslog = stub_bus(bindir)
        self.env = dict(os.environ,
                        DEVAI_GPU_FAKE=str(self.nvml), DEVAI_JOBS_DIR=str(t / "jobs"),
                        DEVAI_SAMPLER_DIR=str(t / "sampler"), DEVAI_BUS_BIN=str(bindir / "devai-bus"),
                        DEVAI_GPU_BIN=str(SCRIPT), DEVAI_GPU_SAMPLE_INTERVAL="0.2",
                        DEVAI_TEST_EPILOG_CONF=str(t / "epilog.conf"),
                        DEVAI_GUARD_ENGINE_SERVICES="fake",
                        SLURM_JOB_COMMENT=json.dumps({"v": 1, "kind": "engine", "backend": "fake"}),
                        SLURM_JOB_LICENSES="engine")

    def tearDown(self):
        pidfile = self.t / "sampler" / "9.pid"
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), signal.SIGKILL)
            except (OSError, ValueError):
                pass
        self._tmp.cleanup()

    def card(self, used=2, energy=0, util=10):
        self.nvml.write_text(json.dumps({"present": True, "used_mib": used, "util_pct": util,
                                         "energy_mj": energy, "processes": []}))

    def run_cli(self, *args, env=None):
        return subprocess.run([sys.executable, str(SCRIPT), *args], env=env or self.env,
                              capture_output=True, text=True, timeout=60)

    def sampler_pid(self):
        return int((self.t / "sampler" / "9.pid").read_text())

    def test_cli_reads_the_fake_card(self):
        self.card(used=1234, energy=7_500)
        self.assertEqual(self.run_cli("used").stdout.strip(), "1234")
        self.assertEqual(self.run_cli("energy").stdout.strip(), "7.5")
        self.assertEqual(self.run_cli("present").returncode, 0)
        self.nvml.write_text(json.dumps({"present": False}))
        self.assertEqual(self.run_cli("present").returncode, 1)
        self.nvml.write_text(json.dumps({"present": True, "processes": [{"pid": 5, "used_mib": 100}]}))
        self.assertEqual(json.loads(self.run_cli("processes", "--json").stdout), [{"pid": 5, "used_mib": 100}])

    def test_prolog_returns_with_the_sampler_running_and_epilog_writes_gpu_json(self):
        start = time.monotonic()
        p = self.run_cli("guard", "--job", "9")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertLess(time.monotonic() - start, 15, "the prolog must return within 15 s")
        pid = self.sampler_pid()
        os.kill(pid, 0)  # alive after the prolog exited
        guard = json.loads((self.t / "jobs" / "results" / "9" / "guard.json").read_text())
        self.assertEqual(guard["verdict"], "idle")
        self.assertIn("devai.events.slurm.guard", self.buslog.read_text())
        time.sleep(0.8)
        self.card(used=2050, energy=5_000_000 + 12_345_000)
        time.sleep(0.6)
        self.card(used=2, energy=5_000_000 + 12_345_000)
        e = self.run_cli("epilog", "--job", "9")
        self.assertEqual(e.returncode, 0, e.stderr)
        gj = json.loads((self.t / "jobs" / "results" / "9" / "gpu.json").read_text())
        self.assertGreaterEqual(len(gj["samples"]), 1)
        self.assertEqual(gj["peak_mib"], 2050)
        self.assertAlmostEqual(gj["energy_j"], 12345.0, places=1)
        self.assertNotIn("sampler_lost", gj)
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            self.fail("the epilog did not stop the sampler")
        self.assertIn("devai.engines.fake.unload", self.buslog.read_text())
        self.assertFalse((self.t / "sampler" / "9.jsonl").exists())

    def test_a_killed_sampler_gives_sampler_lost_and_the_epilog_still_passes(self):
        self.assertEqual(self.run_cli("guard", "--job", "9").returncode, 0)
        time.sleep(0.5)
        os.kill(self.sampler_pid(), signal.SIGKILL)
        time.sleep(0.2)
        e = self.run_cli("epilog", "--job", "9")
        self.assertEqual(e.returncode, 0, e.stderr)
        gj = json.loads((self.t / "jobs" / "results" / "9" / "gpu.json").read_text())
        self.assertTrue(gj["sampler_lost"])

    def test_epilog_fails_when_the_card_stays_busy(self):
        self.assertEqual(self.run_cli("guard", "--job", "9").returncode, 0)
        self.nvml.write_text(json.dumps({"present": True, "used_mib": 4000, "energy_mj": 1,
                                         "processes": [{"pid": 999999, "used_mib": 4000}]}))
        e = self.run_cli("epilog", "--job", "9")
        self.assertEqual(e.returncode, 1, "a busy card after the job drains the node")

    def test_trainer_epilog_sends_cancel_not_unload(self):
        env = dict(self.env, SLURM_JOB_COMMENT=json.dumps({"v": 1, "kind": "trainer", "ft_job": "x"}))
        self.assertEqual(self.run_cli("guard", "--job", "9", env=env).returncode, 0)
        e = self.run_cli("epilog", "--job", "9", env=env)
        self.assertEqual(e.returncode, 0, e.stderr)
        log = self.buslog.read_text()
        self.assertIn("devai.laya-trainer.trainer.cancel", log)
        self.assertIn('"reason": "epilog"', log)
        self.assertNotIn(".unload", log)

    def test_bench_and_action_jobs_leave_the_card_alone_even_when_busy(self):
        self.nvml.write_text(json.dumps({"present": True, "used_mib": 9000, "energy_mj": 1,
                                         "processes": [{"pid": 999999, "used_mib": 9000}]}))
        for comment, lic in (({"v": 1, "kind": "bench"}, "bench"), ({"v": 1, "kind": "action", "gpu": False}, "")):
            env = dict(self.env, SLURM_JOB_COMMENT=json.dumps(comment), SLURM_JOB_LICENSES=lic)
            self.assertEqual(self.run_cli("guard", "--job", "9", env=env).returncode, 0)
            self.assertEqual(self.run_cli("epilog", "--job", "9", env=env).returncode, 0)
        self.assertFalse(self.buslog.exists(), "nothing was sent for bench or action jobs")
        self.assertFalse((self.t / "jobs" / "results" / "9" / "guard.json").exists())

    def test_epilog_test_switch_writes_gpu_json_only(self):
        self.assertEqual(self.run_cli("guard", "--job", "9").returncode, 0)
        (self.t / "epilog.conf").write_text("DEVAI_TEST_EPILOG=off\n")
        before = self.buslog.read_text()
        self.nvml.write_text(json.dumps({"present": True, "used_mib": 4000, "energy_mj": 1,
                                         "processes": [{"pid": 999999, "used_mib": 4000}]}))
        e = self.run_cli("epilog", "--job", "9")
        self.assertEqual(e.returncode, 0, "with the switch on, the epilog exits 0 on a busy card")
        self.assertTrue((self.t / "jobs" / "results" / "9" / "gpu.json").exists())
        self.assertEqual(self.buslog.read_text(), before, "no unload, cancel or stop")

    def test_no_gpu_prolog_skips_nvml_and_epilog_only_unloads(self):
        self.nvml.write_text(json.dumps({"present": False}))
        p = self.run_cli("guard", "--job", "9")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertFalse((self.t / "sampler" / "9.pid").exists(), "no sampler without a GPU")
        guard = json.loads((self.t / "jobs" / "results" / "9" / "guard.json").read_text())
        self.assertIs(guard["gpu"], False)
        e = self.run_cli("epilog", "--job", "9")
        self.assertEqual(e.returncode, 0, e.stderr)
        self.assertIn("devai.engines.fake.unload", self.buslog.read_text())
        self.assertFalse((self.t / "jobs" / "results" / "9" / "gpu.json").exists())


if __name__ == "__main__":
    unittest.main()
