"""The machine-made recovery file and its re-read (plan Phase 17a, I25).

deploy/recovery-flags.generated.json holds the flags a program added
(scripts/recipe-apply.py; Phase 18's auto-fix later), as groups
{backends (required), engine_flags, engine_env, source, reason, added}. The
probers (scripts/_probe_hf_common.py) and the router
(gpu-arbiter/recovery_flags.go) append, for a launch on backend B, the
curated entry's flags when it applies to B, then every generated group whose
`backends` names B or B's engine, in file order. Both files are re-read before
each launch when either changed (device, inode, mtime, size), so a file
replaced with a new inode is seen without a restart; a file that fails to
parse keeps what was loaded, and so does a curated file that disappears. The
router reads both through a read-only mount of the repository's deploy/
DIRECTORY, because a single-file bind mount pins the old inode.

The Go side has the same cases in gpu-arbiter/recovery_flags_test.go.
Stdlib unittest only; nothing is launched.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _probe_hf_common as P  # noqa: E402

CURATED = {"models": {
    "M": {"backends": ["vllm"], "engine_flags": ["--language-model-only"],
          "engine_env": {"A": "curated"}},
    "Everywhere": {"engine_flags": ["--trust-remote-code"]},
}}


def _write_by_rename(path: Path, doc: object) -> None:
    """The writers' way: a temporary file in the same directory, renamed."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(doc if isinstance(doc, str) else json.dumps(doc))
    os.replace(tmp, path)


class _TempRegistry(unittest.TestCase):
    """Points the prober's reader at a temp dir; restores it afterwards."""

    def setUp(self) -> None:
        self._saved = (P.RECOVERY_FLAGS_PATH, P.RECOVERY_GENERATED_PATH,
                       P._RECOVERY_REGISTRY, P._RECOVERY_GENERATED,
                       list(P._RECOVERY_SIGS))
        self.dir = Path(tempfile.mkdtemp())
        self.curated = self.dir / "recovery-flags.json"
        self.generated = self.dir / "recovery-flags.generated.json"
        _write_by_rename(self.curated, CURATED)
        P.RECOVERY_FLAGS_PATH = self.curated
        P.RECOVERY_GENERATED_PATH = self.generated
        P._RECOVERY_SIGS[:] = [P._UNSEEN, P._UNSEEN]
        self.err = io.StringIO()
        self._redirect = contextlib.redirect_stderr(self.err)
        self._redirect.__enter__()

    def tearDown(self) -> None:
        self._redirect.__exit__(None, None, None)
        (P.RECOVERY_FLAGS_PATH, P.RECOVERY_GENERATED_PATH, P._RECOVERY_REGISTRY,
         P._RECOVERY_GENERATED, sigs) = self._saved
        P._RECOVERY_SIGS[:] = sigs
        for f in self.dir.iterdir():
            f.unlink()
        self.dir.rmdir()

    def gen(self, models: dict, version: int = 1) -> None:
        _write_by_rename(self.generated, {"version": version, "models": models})


class GroupScopingTest(_TempRegistry):
    def test_a_group_applies_only_to_its_backends(self) -> None:
        self.gen({"G": [
            {"backends": ["vllm"], "engine_flags": ["--enforce-eager"], "source": "recipe"},
            {"backends": ["vllm-devai"], "engine_flags": ["--max-num-batched-tokens", "2048"]},
            {"backends": ["sglang"], "engine_flags": ["--disable-radix-cache"]},
        ]})
        # A ["vllm"] group reaches a vllm-devai launch (same engine) ...
        self.assertEqual(P.recovery_overrides("G", "vllm-devai")[0],
                         ["--enforce-eager", "--max-num-batched-tokens", "2048"])
        # ... a ["vllm-devai"] group does not reach stock vllm.
        self.assertEqual(P.recovery_overrides("G", "vllm")[0], ["--enforce-eager"])
        self.assertEqual(P.recovery_overrides("G", "sglang")[0], ["--disable-radix-cache"])
        self.assertEqual(P.recovery_overrides("G", "ollama"), ([], {}))

    def test_curated_first_then_groups_in_file_order(self) -> None:
        self.gen({"M": [
            {"backends": ["vllm"], "engine_flags": ["--max-num-seqs", "8"],
             "engine_env": {"A": "first", "B": "1"}},
            {"backends": ["vllm"], "engine_flags": ["--max-num-seqs", "4"],
             "engine_env": {"A": "second"}},
        ]})
        flags, env = P.recovery_overrides("M", "vllm")
        self.assertEqual(flags, ["--language-model-only", "--max-num-seqs", "8",
                                 "--max-num-seqs", "4"])
        self.assertEqual(env, {"A": "second", "B": "1"})
        self.assertEqual(P.recovery_overrides("M", "sglang"), ([], {}))

    def test_a_group_without_backends_is_skipped(self) -> None:
        self.gen({"M": [
            {"engine_flags": ["--absent"]},
            {"backends": "vllm", "engine_flags": ["--not-a-list"]},
            {"backends": ["vllm"], "engine_flags": ["--kept"], "image": "x/y:z"},
        ], "Broken": "not a list"})
        self.assertEqual(P.recovery_overrides("M", "vllm")[0],
                         ["--language-model-only", "--kept"])
        # A generated group never sets an image; the curated entry has none.
        self.assertIsNone(P.recovery_image("M", "vllm"))
        self.assertIn("backends", self.err.getvalue())


class ReReadTest(_TempRegistry):
    def test_removing_a_group_reverts_exactly_its_flags(self) -> None:
        self.gen({"M": [
            {"backends": ["vllm"], "engine_flags": ["--enforce-eager"], "source": "recipe"},
            {"backends": ["vllm"], "engine_flags": ["--max-num-batched-tokens", "4096"],
             "engine_env": {"X": "1"}, "source": "auto-fix"},
        ]})
        self.assertEqual(P.recovery_overrides("M", "vllm"),
                         (["--language-model-only", "--enforce-eager",
                           "--max-num-batched-tokens", "4096"],
                          {"A": "curated", "X": "1"}))
        self.gen({"M": [{"backends": ["vllm"], "engine_flags": ["--enforce-eager"],
                         "source": "recipe"}]})
        self.assertEqual(P.recovery_overrides("M", "vllm"),
                         (["--language-model-only", "--enforce-eager"], {"A": "curated"}))
        self.generated.unlink()
        self.assertEqual(P.recovery_overrides("M", "vllm"),
                         (["--language-model-only"], {"A": "curated"}))

    def test_a_rename_replaced_file_is_seen_at_the_next_launch(self) -> None:
        self.gen({})
        self.assertEqual(P.recovery_overrides("New", "vllm"), ([], {}))
        self.gen({"New": [{"backends": ["vllm"], "engine_flags": ["--enforce-eager"]}]})
        self.assertEqual(P.recovery_overrides("New", "vllm")[0], ["--enforce-eager"])
        _write_by_rename(self.curated, {"models": {"New": {"engine_flags": ["--curated"]}}})
        self.assertEqual(P.recovery_overrides("New", "vllm")[0],
                         ["--curated", "--enforce-eager"])

    def test_a_missing_generated_file_reads_as_empty(self) -> None:
        self.assertFalse(self.generated.exists())
        self.assertEqual(P.recovery_overrides("M", "vllm"),
                         (["--language-model-only"], {"A": "curated"}))

    def test_a_malformed_file_keeps_the_previous_registry(self) -> None:
        self.gen({"M": [{"backends": ["vllm"], "engine_flags": ["--enforce-eager"]}]})
        want = (["--language-model-only", "--enforce-eager"], {"A": "curated"})
        self.assertEqual(P.recovery_overrides("M", "vllm"), want)
        for bad in ('{"version": 1, "models": {', '{"version": 2, "models": {}}',
                    '{"models": {}}'):
            _write_by_rename(self.generated, bad)
            self.assertEqual(P.recovery_overrides("M", "vllm"), want, bad)
        _write_by_rename(self.curated, '{"models": ')
        self.assertEqual(P.recovery_overrides("M", "vllm"), want)
        self.assertIn("keeping the previously loaded", self.err.getvalue())
        # Fixed files are picked up again.
        self.gen({})
        _write_by_rename(self.curated, CURATED)
        self.assertEqual(P.recovery_overrides("M", "vllm"),
                         (["--language-model-only"], {"A": "curated"}))


class SharedFixtureTest(_TempRegistry):
    """The same files gpu-arbiter/recovery_flags_test.go loads: both readers
    must make the same call on every malformed input, curated and generated
    alike, or a probe and a serve-time launch get different flags."""

    FIXTURES = REPO_ROOT / "tests" / "fixtures" / "recovery-flags"

    def expected(self) -> dict:
        return json.loads((self.FIXTURES / "expected.json").read_text())

    def lookup(self, key: str) -> dict:
        backend, model = key.split("/", 1)
        flags, env = P.recovery_overrides(model, backend)
        return {"flags": flags, "env": env, "image": P.recovery_image(model, backend) or ""}

    def test_both_readers_agree(self) -> None:
        P.RECOVERY_FLAGS_PATH = self.FIXTURES / "curated.json"
        P.RECOVERY_GENERATED_PATH = self.FIXTURES / "generated.json"
        P._RECOVERY_SIGS[:] = [P._UNSEEN, P._UNSEEN]
        for key, want in self.expected()["lookups"].items():
            with self.subTest(key=key):
                self.assertEqual(self.lookup(key), want)

    def test_every_broken_file_keeps_the_previous_registry(self) -> None:
        exp = self.expected()
        _write_by_rename(self.curated, (self.FIXTURES / "curated.json").read_text())
        _write_by_rename(self.generated, (self.FIXTURES / "generated.json").read_text())
        for key, want in exp["lookups"].items():  # the good load comes first
            self.assertEqual(self.lookup(key), want, key)
        targets = {"curated": self.curated, "generated": self.generated}
        for which, files in exp["keeps_previous"].items():
            self.assertTrue(files, which)
            for name in files:
                _write_by_rename(targets[which], (self.FIXTURES / "broken" / name).read_text())
                for key, want in exp["lookups"].items():
                    with self.subTest(broken=name, key=key):
                        self.assertEqual(self.lookup(key), want)
            _write_by_rename(targets[which], (self.FIXTURES / f"{which}.json").read_text())
            for key, want in exp["lookups"].items():
                self.assertEqual(self.lookup(key), want, f"restored {which}: {key}")

    def test_a_disappeared_curated_file_keeps_its_entries(self) -> None:
        # git checkout replaces a file by unlink and create; a launch in
        # between must not lose the curated flags. A removed generated file
        # reverts what it added (I25).
        self.gen({"M": [{"backends": ["vllm"], "engine_flags": ["--enforce-eager"]}]})
        self.assertEqual(P.recovery_overrides("M", "vllm")[0],
                         ["--language-model-only", "--enforce-eager"])
        self.curated.unlink()
        self.assertEqual(P.recovery_overrides("M", "vllm")[0],
                         ["--language-model-only", "--enforce-eager"])
        self.assertIn("disappeared", self.err.getvalue())
        _write_by_rename(self.curated, {"models": {"M": {"engine_flags": ["--back"]}}})
        self.assertEqual(P.recovery_overrides("M", "vllm")[0], ["--back", "--enforce-eager"])
        self.generated.unlink()
        self.assertEqual(P.recovery_overrides("M", "vllm")[0], ["--back"])

    def test_a_curated_file_missing_at_the_first_load_is_empty(self) -> None:
        self.curated.unlink()
        P._RECOVERY_SIGS[:] = [P._UNSEEN, P._UNSEEN]
        self.assertEqual(P.recovery_overrides("M", "vllm"), ([], {}))
        self.assertIn("not present", self.err.getvalue())
        _write_by_rename(self.curated, {"models": {"M": {"engine_flags": ["--late"]}}})
        self.assertEqual(P.recovery_overrides("M", "vllm")[0], ["--late"])


class ShippedFileTest(unittest.TestCase):
    def test_the_generated_file_is_tracked_in_the_i25_shape(self) -> None:
        path = REPO_ROOT / "deploy" / "recovery-flags.generated.json"
        doc = json.loads(path.read_text())
        self.assertEqual(doc.get("version"), 1)
        self.assertIsInstance(doc.get("models"), dict)
        for name, groups in doc["models"].items():
            for g in groups:
                self.assertIsInstance(g.get("backends"), list, name)
                self.assertIn(g.get("source"), ("recipe", "auto-fix"), name)
                self.assertNotIn("image", g, name)

    def test_the_default_path_is_beside_the_curated_file(self) -> None:
        self.assertEqual(P.RECOVERY_GENERATED_PATH.parent, P.RECOVERY_FLAGS_PATH.parent)
        self.assertEqual(P.RECOVERY_GENERATED_PATH.name, "recovery-flags.generated.json")


class DeployDirectoryMountTest(unittest.TestCase):
    """The router reads both files from a read-only DIRECTORY mount of
    ${DEVAI_REPO}/deploy (I25, C6), and everything that runs compose has
    DEVAI_REPO."""

    def setUp(self) -> None:
        import yaml
        self.compose = yaml.safe_load((REPO_ROOT / "deploy" / "docker-compose.yaml").read_text())
        self.router = self.compose["services"]["router"]

    def test_router_mounts_the_deploy_directory_read_only(self) -> None:
        self.assertIn("${DEVAI_REPO:?", "".join(self.router["volumes"]))
        mounts = [v for v in self.router["volumes"] if "/etc/devai/deploy" in v]
        self.assertEqual(len(mounts), 1, mounts)
        self.assertRegex(mounts[0], r"^\$\{DEVAI_REPO:\?[^}]*\}/deploy:/etc/devai/deploy:ro$")
        # No single-file mount of either recovery file is left: it would pin
        # the old inode, and a file replaced with a new one would stay unseen.
        self.assertFalse([v for v in self.router["volumes"] if "recovery-flags" in v])

    def test_router_reads_both_files_there(self) -> None:
        env = dict(e.split("=", 1) for e in self.router["environment"])
        self.assertEqual(env["RECOVERY_FLAGS_REGISTRY"], "/etc/devai/deploy/recovery-flags.json")
        self.assertEqual(env["RECOVERY_FLAGS_GENERATED"],
                         "/etc/devai/deploy/recovery-flags.generated.json")

    def test_the_makefile_exports_devai_repo(self) -> None:
        mk = (REPO_ROOT / "Makefile").read_text()
        self.assertRegex(mk, r"(?m)^export DEVAI_REPO := \$\(CURDIR\)$")

    def test_scripts_that_run_compose_default_devai_repo_first(self) -> None:
        line = ': "${DEVAI_REPO:=$(cd "$(dirname "$0")/.." && pwd)}"; export DEVAI_REPO'
        for script in ("tests/test-router-idle.sh", "tests/test-model-matrix.sh"):
            text = (REPO_ROOT / script).read_text()
            with self.subTest(script=script):
                self.assertIn(line, text)
                first_compose = min(m.start() for m in re.finditer(r"compose -f", text))
                self.assertLess(text.index(line), first_compose)

    def test_the_systemd_unit_gets_the_repository_path(self) -> None:
        import subprocess
        unit = REPO_ROOT / "deploy" / "systemd" / "devai-infra.service"
        self.assertIn('Environment="DEVAI_REPO=@DEVAI_REPO@"', unit.read_text())
        mk = (REPO_ROOT / "Makefile").read_text()
        install = re.search(r"^install-systemd:.*?(?=^\S)", mk, re.M | re.S).group(0)
        # The substitution reads the path from the environment and escapes it
        # for the quoted value; run exactly that code on awkward paths.
        code = re.search(r"python3 -c '(.*?)' deploy/systemd/devai-infra.service", install).group(1)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "unit.service"
            for repo, want in (("/srv/a b|c&d", '"DEVAI_REPO=/srv/a b|c&d"'),
                               ('/srv/q"x\\y%z', '"DEVAI_REPO=/srv/q\\"x\\\\y%%z"')):
                r = subprocess.run([sys.executable, "-c", code, str(unit), str(out)],
                                   env={**os.environ, "DEVAI_REPO": repo},
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertIn(f"Environment={want}\n", out.read_text(), repo)
            r = subprocess.run([sys.executable, "-c", code, str(unit), str(out)],
                               env={**os.environ, "DEVAI_REPO": "/srv/a\nb"},
                               capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("newline", r.stderr)


if __name__ == "__main__":
    unittest.main()
