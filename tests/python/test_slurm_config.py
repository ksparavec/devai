"""Static checks of devai-slurm's configuration (platform-rebuild Phase 5):

- the C1 users table of the plan equals deploy/slurm/nats-server.conf.in,
  `expires: "15m"` on the control users included;
- the slurm.conf.in keys of the Phase 5 deliverable, and their rendering
  (the node's CPUs and memory from the host, I4);
- the compose service: profile slurm, pid host, devai-net only, no podman
  socket, nothing published, podman's default user namespace;
- the epilog test switch is absent from the production slurm.conf.in, the
  entrypoint, the Dockerfile and compose;
- the job scripts are C3's, line for line;
- the image pins Debian by the Makefile's DEBIAN_IMAGE digest and Slurm by
  version, and refuses to build without job_submit_lua.so;
- scripts/slurm-init.sh creates C7's files 0600 and never overwrites one.
"""

from __future__ import annotations

import configparser
import json
import os
import re
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SLURM = REPO / "deploy" / "slurm"
PLAN = REPO / "docs" / "plans" / "platform-rebuild.md"

CONTROL_CONTAINERS = {"control-engines": "engines", "control-laya": "laya-trainer", "control-operator": "operator"}


def c1_table() -> dict[str, dict]:
    """The plan's C1 'Users and permissions' table, one entry per user."""
    text = PLAN.read_text()
    block = text.split("**Users and permissions.**", 1)[1].split("**Envelope.**", 1)[0]
    users = {}
    for line in block.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        names = re.findall(r"`([^`]+)`", cells[0])
        pub = [t for t in re.findall(r"`([^`]+)`", cells[2]) if t.startswith(("devai.", "_INBOX."))]
        sub = [t for t in re.findall(r"`([^`]+)`", cells[3]) if t.startswith(("devai.", "_INBOX."))]
        responses = "allow_responses" in cells[3]
        if responses:
            assert 'allow_responses: {max: 1, expires: "15m"}' in cells[3], cells[3]
        for name in names:
            container = CONTROL_CONTAINERS.get(name)
            users[name] = {
                "publish": sorted(p.replace("<container>", container) if container else p for p in pub),
                "subscribe": sorted(s.replace("<container>", container) if container else s for s in sub),
                "responses": responses,
                "nothing": cells[2] == "nothing",
            }
    return users


def conf_users() -> list[dict]:
    text = (SLURM / "nats-server.conf.in").read_text()
    body = text.split("users = ", 1)[1]
    depth, end = 0, 0
    for i, ch in enumerate(body):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return json.loads(body[:end])


class NatsPermissionsEqualC1(unittest.TestCase):
    def test_every_c1_user_has_exactly_its_permissions(self):
        table = c1_table()
        self.assertEqual(sorted(table), ["buslog", "control-engines", "control-laya", "control-operator",
                                         "jobs", "operator", "router"])
        conf = {u["user"]: u for u in conf_users()}
        self.assertEqual(sorted(conf), sorted(table))
        for name, want in table.items():
            perms = conf[name]["permissions"]
            with self.subTest(user=name):
                self.assertEqual(conf[name]["password"], f"@PASS:{name}@")
                if want["nothing"]:
                    self.assertEqual(perms["publish"], {"deny": [">"]}, "publishes nothing")
                else:
                    self.assertEqual(sorted(perms["publish"]["allow"]), want["publish"])
                self.assertEqual(sorted(perms["subscribe"]["allow"]), want["subscribe"])
                if want["responses"]:
                    self.assertEqual(perms["allow_responses"], {"max": 1, "expires": "15m"})
                else:
                    self.assertNotIn("allow_responses", perms)

    def test_no_jetstream_no_published_monitoring(self):
        text = (SLURM / "nats-server.conf.in").read_text()
        body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
        self.assertNotIn("jetstream", body)
        self.assertNotIn("http", body)
        self.assertIn("max_payload: 65536", body)


SLURM_KEYS = {
    "ClusterName": "devai",
    "SlurmctldHost": "localhost",
    "SelectType": "select/cons_tres",
    "SelectTypeParameters": "CR_CPU",
    "Licenses": "engine:1,bench:1",
    "AuthType": "auth/slurm",
    "CredType": "cred/slurm",
    "AuthAltTypes": "auth/jwt",
    "AuthAltParameters": "jwt_key=/etc/slurm/jwt_hs256.key",
    "ProctrackType": "proctrack/linuxproc",
    "TaskPlugin": "task/none",
    "JobAcctGatherType": "jobacct_gather/none",
    "AccountingStorageType": "accounting_storage/slurmdbd",
    "AccountingStoreFlags": "job_comment,job_script",
    "PrologFlags": "Alloc,ForceRequeueOnFail",
    "JobRequeue": "0",
    "JobSubmitPlugins": "lua",
    "Prolog": "/usr/local/lib/devai/prolog.sh",
    "Epilog": "/usr/local/lib/devai/epilog.sh",
    "HealthCheckProgram": "/usr/local/lib/devai/healthcheck.sh",
    "HealthCheckInterval": "30",
    "PrologEpilogTimeout": "180",
    "BatchStartTimeout": "240",
    "KillWait": "30",
    "SlurmdParameters": "config_overrides",
    "ReturnToService": "2",
    "StateSaveLocation": "/var/cache/devai/jobs/slurmctld",
}


def slurm_conf(text: str) -> tuple[dict, list[str]]:
    keys, lines = {}, []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        lines.append(line)
        if not line.startswith(("NodeName=", "PartitionName=")):
            k, _, v = line.partition("=")
            keys[k] = v
    return keys, lines


def render(text: str, cpus: int, mem: int) -> str:
    """entrypoint.sh's substitution."""
    return text.replace("@CPUS@", str(cpus)).replace("@MEMORY_MB@", str(mem))


class SlurmConf(unittest.TestCase):
    def setUp(self):
        self.text = (SLURM / "slurm.conf.in").read_text()

    def test_keys(self):
        keys, _ = slurm_conf(self.text)
        for k, v in SLURM_KEYS.items():
            self.assertEqual(keys.get(k), v, k)

    def test_node_and_partition_render_from_the_host(self):
        _, lines = slurm_conf(render(self.text, 24, 63675))
        node = [line for line in lines if line.startswith("NodeName=")]
        part = [line for line in lines if line.startswith("PartitionName=")]
        self.assertEqual(node, ["NodeName=devai NodeAddr=127.0.0.1 CPUs=24 RealMemory=63675 State=UNKNOWN"])
        self.assertEqual(part, ["PartitionName=devai Nodes=devai Default=YES MaxTime=INFINITE State=UP"])
        self.assertNotIn("@", render(self.text, 24, 63675).split("NodeName=", 1)[1])
        entry = (SLURM / "entrypoint.sh").read_text()
        self.assertIn("nproc --all", entry)
        self.assertIn("MemTotal", entry)
        self.assertIn('s/@CPUS@/$CPUS/g', entry)
        self.assertIn('s/@MEMORY_MB@/$MEMORY_MB/g', entry)

    def test_no_cgroups_at_all(self):
        self.assertIn("CgroupPlugin=disabled", (SLURM / "cgroup.conf").read_text())

    def test_slurmdbd_and_lua(self):
        dbd = (SLURM / "slurmdbd.conf.in").read_text()
        self.assertIn("StoragePass=@MARIADB_PASS@", dbd)
        self.assertIn("AuthAltTypes=auth/jwt", dbd)
        lua = (SLURM / "job_submit.lua").read_text()
        self.assertIn("DEVAI_UID = 1000", lua)
        self.assertIn("submit_uid ~= DEVAI_UID", lua)
        self.assertIn("job_desc.user_id", lua)
        self.assertIn("devai jobs run as devai only", lua)
        self.assertIn("function slurm_job_modify", lua)

    def test_supervisord_programs(self):
        cp = configparser.RawConfigParser()
        cp.read(SLURM / "supervisord.conf")
        progs = {s.split(":", 1)[1]: dict(cp[s]) for s in cp.sections() if s.startswith("program:")}
        self.assertEqual(sorted(progs), ["accounting-init", "buslog", "mariadb", "nats-server",
                                         "slurmctld", "slurmd", "slurmdbd", "slurmrestd"])
        self.assertIn("--innodb-snapshot-isolation=OFF", progs["mariadb"]["command"])
        rest = progs["slurmrestd"]
        self.assertEqual(rest["user"], "slurmrest")
        self.assertIn("-a rest_auth/jwt", rest["command"])
        self.assertIn(":6820", rest["command"])
        self.assertIn("-N devai", progs["slurmd"]["command"])
        acct = progs["accounting-init"]
        self.assertEqual((acct["autorestart"], acct["startsecs"], acct["exitcodes"]), ("unexpected", "0", "0"))
        self.assertIn("devai-bus log --dir /var/cache/devai/jobs/bus", progs["buslog"]["command"])


class Compose(unittest.TestCase):
    def setUp(self):
        try:
            import yaml
        except ImportError:  # pragma: no cover
            self.skipTest("PyYAML not installed")
        self.svc = yaml.safe_load((REPO / "deploy" / "docker-compose.yaml").read_text())["services"]["slurm"]

    def test_service_shape(self):
        s = self.svc
        self.assertEqual(s["profiles"], ["slurm"])
        self.assertEqual(s["container_name"], "devai-slurm")
        self.assertEqual(s["hostname"], "devai-slurm")
        self.assertEqual(s["pid"], "host")
        self.assertEqual(s["networks"], ["default"], "devai-net only")
        self.assertNotIn("ports", s, "devai-slurm publishes nothing")
        self.assertNotIn("userns_mode", s, "podman's default user namespace (I26)")
        self.assertNotIn("privileged", s)
        self.assertEqual(s["tmpfs"], ["/run/devai"])
        env = dict(e.split("=", 1) for e in s["environment"])
        self.assertEqual(env, {"SLURM_REST_URL": "http://127.0.0.1:6820",
                               "SLURM_JWT_KEY_FILE": "/etc/slurm/jwt_hs256.key", "SLURM_USER": "devai"})

    def test_no_podman_socket(self):
        for v in self.svc["volumes"]:
            self.assertNotIn("podman.sock", v)
            self.assertNotIn("docker.sock", v)
        self.assertIn("/var/cache/devai/jobs:/var/cache/devai/jobs", self.svc["volumes"])


class EpilogTestSwitch(unittest.TestCase):
    def test_absent_from_production_files(self):
        for path in (SLURM / "slurm.conf.in", SLURM / "entrypoint.sh", SLURM / "Dockerfile",
                     REPO / "deploy" / "docker-compose.yaml"):
            text = path.read_text()
            with self.subTest(path=path.name):
                self.assertNotIn("DEVAI_TEST_EPILOG", text)
                self.assertNotIn("/run/devai/test", text)

    def test_only_the_test_script_writes_it(self):
        self.assertIn("DEVAI_TEST_EPILOG=off", (REPO / "tests" / "slurm" / "test-slurm.sh").read_text())
        self.assertIn("/run/devai/test/epilog.conf", (SLURM / "devai-gpu").read_text())


def plan_block(after: str) -> list[str]:
    """The bash block that follows `after` in the plan, without comment lines."""
    text = PLAN.read_text().split(after, 1)[1]
    code = text.split("```bash\n", 1)[1].split("```", 1)[0]
    return [line for line in code.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def script_lines(name: str) -> list[str]:
    text = (SLURM / "jobs" / name).read_text()
    return [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


class JobScriptsAreC3s(unittest.TestCase):
    def test_engine_trainer_action(self):
        for name, after in (("engine.sh", "The engine job script, `deploy/slurm/jobs/engine.sh`"),
                            ("trainer.sh", "The trainer job script, `deploy/slurm/jobs/trainer.sh`"),
                            ("action.sh", "The action job script, `deploy/slurm/jobs/action.sh`")):
            with self.subTest(script=name):
                self.assertEqual(script_lines(name), plan_block(after))

    def test_bench_and_probe(self):
        bench = "\n".join(script_lines("bench.sh"))
        self.assertIn('"kind\\":\\"bench\\"', bench)
        self.assertIn("trap 'stop INT 110' INT", bench)
        probe = "\n".join(script_lines("probe.sh"))
        self.assertIn("--lease-only --parent $$ &", probe)
        self.assertIn("devai-bus request $S.unload '{}' --job \"$J\" --timeout 25s", probe)
        self.assertIn('kill -HUP "$L"', probe)
        self.assertIn('"kind\\":\\"probe\\"', probe)


class Image(unittest.TestCase):
    def test_debian_pin_matches_the_makefile(self):
        mk = (REPO / "Makefile").read_text()
        pin = re.search(r"^DEBIAN_IMAGE = (\S+)$", mk, re.M).group(1)
        self.assertRegex(pin, r"^docker\.io/library/debian:trixie-slim@sha256:[0-9a-f]{64}$")
        for df in (SLURM / "Dockerfile", REPO / "tests" / "slurm" / "Dockerfile.control-test"):
            self.assertIn(f"ARG BASE_IMAGE={pin}", df.read_text(), df.name)
        self.assertIn("--build-arg BASE_IMAGE=$(DEBIAN_IMAGE)", mk)

    def test_slurm_pinned_and_lua_required(self):
        df = (SLURM / "Dockerfile").read_text()
        self.assertIn("ARG SLURM_VERSION=24.11.5-4+deb13u1", df)
        for pkg in ("slurmctld", "slurmd", "slurmdbd", "slurmrestd", "slurm-wlm-jwt-plugin", "slurm-wlm-mysql-plugin"):
            self.assertIn(f"{pkg}=$SLURM_VERSION", df)
        self.assertIn("job_submit_lua", df)
        self.assertIn('ENTRYPOINT ["/usr/bin/tini", "-s", "--"', df)

    def test_documented_targets(self):
        mk = (REPO / "Makefile").read_text()
        for target in ("build-slurm", "slurm-init", "setup-jobs", "slurm-up", "slurm-down", "test-slurm"):
            self.assertRegex(mk, rf"(?m)^{target}:.*## ", target)
        self.assertIn("LV=cache_jobs MOUNTPOINT=$(JOBS_DIR) SIZE=$${SIZE:-50G}", mk)


class SlurmInit(unittest.TestCase):
    USERS = ("router", "jobs", "operator", "control-engines", "control-laya", "control-operator", "buslog")

    def run_init(self, home: Path):
        env = dict(os.environ, HOME=str(home))
        env.pop("DEVAI_SLURM_CONFIG_DIR", None)
        return subprocess.run(["bash", str(REPO / "scripts" / "slurm-init.sh")], env=env,
                              capture_output=True, text=True, timeout=60)

    def test_creates_c7_files_0600_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            p = self.run_init(home)
            self.assertEqual(p.returncode, 0, p.stderr)
            d = home / ".config" / "devai" / "slurm"
            files = {"slurm.key": 1024, "jwt_hs256.key": 32, "mariadb.pass": None}
            files.update({f"nats/{u}.pass": None for u in self.USERS})
            for rel, size in files.items():
                f = d / rel
                self.assertEqual(stat.S_IMODE(f.stat().st_mode), 0o600, rel)
                if size:
                    self.assertEqual(f.stat().st_size, size, rel)
                else:
                    self.assertRegex(f.read_text(), r"^[0-9a-f]{64}\n$", rel)
            self.assertEqual(stat.S_IMODE(d.stat().st_mode), 0o700)
            before = {rel: (d / rel).read_bytes() for rel in files}
            p = self.run_init(home)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("0 created", p.stdout)
            for rel in files:
                self.assertEqual((d / rel).read_bytes(), before[rel], f"{rel} was overwritten")

    def test_every_bus_user_of_the_template_gets_a_password(self):
        template = set(re.findall(r"@PASS:([a-z0-9-]+)@", (SLURM / "nats-server.conf.in").read_text()))
        init = (REPO / "scripts" / "slurm-init.sh").read_text()
        entry = (SLURM / "entrypoint.sh").read_text()
        for u in template:
            self.assertIn(u, init)
            self.assertIn(u, entry)
        self.assertEqual(template, set(self.USERS))


if __name__ == "__main__":
    unittest.main()
