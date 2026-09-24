"""The laya store: deploy/laya-models.yaml and `select-models.py --name <laya row>`.

What this pins (docs/plans/laya-trainer.md, Phase 1):

- The catalog pins a full commit and a sha256 + size for every file, and the
  loader refuses anything looser: a branch or tag can move, and a moved base
  silently changes every student trained on it.
- A pull downloads exactly the pinned files at the pinned revision, verifies
  every one BEFORE anything lands in base/, strips the repo subfolder, and
  leaves the checkpoint read-only.
- A verified checkpoint is not downloaded again; a mismatching one is refused
  rather than overwritten.
- laya rows stay out of deploy/models.yaml, whose LLM readers would choke on
  them; `--name` finds them in the laya catalog instead.

Stdlib unittest only. The real /var/cache/devai is never touched: every laya
path constant is redirected at a tmpdir, and the download is faked.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = REPO_ROOT / "scripts" / "select-models.py"
sys.path.insert(0, str(REPO_ROOT / "laya-trainer"))

from laya_trainer import catalog as lc  # noqa: E402


def _load_select_models():
    spec = importlib.util.spec_from_file_location("select_models_laya", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["select_models_laya"] = mod
    spec.loader.exec_module(mod)
    return mod


sm = _load_select_models()

REV = "0123456789abcdef0123456789abcdef01234567"
PAYLOAD = {
    "model.safetensors": b"weights",
    "rl_agent_config.json": b'{"encoder": "x", "head_layers": 2}',
    "encoder/config.json": b"{}",
    "tokenizer/tokenizer.json": b'{"model": {}}',
    "tokenizer/tokenizer_config.json": b'{"tokenizer_class": "PreTrainedTokenizerFast"}',
}


def _row(name: str = "laya-test", subfolder: str = "multilingual",
         payload: dict[str, bytes] | None = None, default: bool = True) -> dict:
    payload = PAYLOAD if payload is None else payload
    return {
        "name": name, "default": default, "repo": "org/laya", "revision": REV,
        "subfolder": subfolder,
        "files": {rel: {"sha256": hashlib.sha256(b).hexdigest(), "size": len(b)}
                  for rel, b in payload.items()},
    }


def _doc(*rows: dict) -> dict:
    return {"schema_version": 1, "models": list(rows)}


class RealCatalogTest(unittest.TestCase):
    """deploy/laya-models.yaml itself."""

    def setUp(self) -> None:
        self.rows = lc.load_catalog(REPO_ROOT / "deploy" / "laya-models.yaml")

    def test_loads_and_has_the_two_planned_rows(self) -> None:
        self.assertEqual({r["name"] for r in self.rows},
                         {"laya-multilingual", "laya-english"})

    def test_the_default_student_is_multilingual(self) -> None:
        self.assertEqual(lc.default_row(self.rows)["name"], "laya-multilingual")

    def test_both_rows_pin_the_planned_revision(self) -> None:
        for r in self.rows:
            self.assertEqual(r["revision"], "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851")

    def test_checkpoint_dirname_matches_the_plan(self) -> None:
        row = lc.find(self.rows, "laya-multilingual")
        self.assertEqual(lc.checkpoint_dirname(row), "laya-multilingual@55cf4c4ebb4e")

    def test_english_downloads_from_the_repo_root(self) -> None:
        row = lc.find(self.rows, "laya-english")
        self.assertIn("model.safetensors", lc.repo_paths(row))
        # Never the sibling checkpoints that share the repo.
        self.assertFalse([p for p in lc.repo_paths(row) if "/" in p
                          and p.split("/")[0] not in ("encoder", "tokenizer")])

    def test_multilingual_downloads_only_its_subfolder(self) -> None:
        row = lc.find(self.rows, "laya-multilingual")
        self.assertTrue(all(p.startswith("multilingual/") for p in lc.repo_paths(row)))

    def test_not_in_the_llm_catalog(self) -> None:
        models = yaml.safe_load((REPO_ROOT / "deploy" / "models.yaml").read_text())
        names = {m.get("name") for m in models.get("models", [])}
        self.assertFalse(names & {r["name"] for r in self.rows})


class CatalogValidationTest(unittest.TestCase):

    def assertRefused(self, doc: dict, fragment: str) -> None:
        with self.assertRaises(lc.CatalogError) as cm:
            lc.parse_catalog(doc)
        self.assertIn(fragment, str(cm.exception))

    def test_a_valid_row_passes(self) -> None:
        self.assertEqual(len(lc.parse_catalog(_doc(_row()))), 1)

    def test_a_branch_name_is_not_a_pin(self) -> None:
        self.assertRefused(_doc({**_row(), "revision": "main"}), "40-hex")

    def test_a_short_commit_is_not_a_pin(self) -> None:
        self.assertRefused(_doc({**_row(), "revision": REV[:12]}), "40-hex")

    def test_every_file_needs_a_sha256(self) -> None:
        row = _row()
        row["files"]["model.safetensors"] = {"size": 7}
        self.assertRefused(_doc(row), "sha256")

    def test_required_files_are_enforced(self) -> None:
        payload = dict(PAYLOAD)
        del payload["encoder/config.json"]
        self.assertRefused(_doc(_row(payload=payload)), "encoder/config.json")

    def test_a_file_path_cannot_escape(self) -> None:
        row = _row()
        row["files"]["../evil"] = row["files"]["model.safetensors"]
        self.assertRefused(_doc(row), "relative")

    def test_names_are_unique(self) -> None:
        self.assertRefused(_doc(_row(), _row(default=False)), "duplicate")

    def test_exactly_one_default(self) -> None:
        self.assertRefused(_doc(_row(), _row(name="other")), "exactly one")
        self.assertRefused(_doc(_row(default=False)), "exactly one")

    def test_schema_version_is_checked(self) -> None:
        self.assertRefused({"schema_version": 2, "models": [_row()]}, "schema_version")


class _LayaStoreCase(unittest.TestCase):
    """Redirect every laya path at a tmpdir and fake the hf download."""

    def setUp(self) -> None:
        tmp = tempfile.mkdtemp(prefix="laya-store-")
        self.addCleanup(self._rmtree, tmp)
        self.store = Path(tmp) / "laya"
        for attr, value in (("LAYA_STORE", self.store),
                            ("LAYA_BASE", self.store / "base"),
                            ("LAYA_STAGING", self.store / ".staging")):
            patcher = mock.patch.object(sm, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.calls: list[list[str]] = []
        self.serve = dict(PAYLOAD)
        patcher = mock.patch.object(sm, "run_download", self._fake_download)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _rmtree(path: str) -> None:
        lc.make_writable(Path(path))
        shutil.rmtree(path)

    def _fake_download(self, cmd: list[str], what: str) -> None:
        """Write the requested repo files into --local-dir, like `hf download`."""
        self.calls.append(cmd)
        local = Path(cmd[cmd.index("--local-dir") + 1])
        (local / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        for repo_path in cmd[3:cmd.index("--revision")]:
            rel = repo_path.split("/", 1)[1] if repo_path.startswith("multilingual/") else repo_path
            dest = local / repo_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(self.serve[rel])


class PullTest(_LayaStoreCase):

    def test_pull_lands_a_verified_read_only_checkpoint(self) -> None:
        target = sm.pull_laya(_row())
        self.assertEqual(target, self.store / "base" / f"laya-test@{REV[:12]}")
        self.assertEqual(lc.verify_dir(target, _row()), [])
        # Subfolder stripped, hf's own metadata left behind.
        self.assertTrue((target / "model.safetensors").is_file())
        self.assertFalse((target / "multilingual").exists())
        self.assertFalse((target / ".cache").exists())
        mode = stat.S_IMODE((target / "model.safetensors").stat().st_mode)
        self.assertEqual(mode & 0o222, 0, "checkpoint files must be read-only")
        self.assertEqual(stat.S_IMODE(target.stat().st_mode) & 0o222, 0)

    def test_pull_asks_hf_for_the_pinned_files_at_the_pinned_revision(self) -> None:
        sm.pull_laya(_row())
        cmd = self.calls[0]
        self.assertEqual(cmd[:3], [sm.HF_CLI, "download", "org/laya"])
        self.assertEqual(cmd[cmd.index("--revision") + 1], REV)
        self.assertEqual(sorted(cmd[3:cmd.index("--revision")]),
                         sorted(f"multilingual/{rel}" for rel in PAYLOAD))
        self.assertNotIn("--include", cmd)

    def test_pull_creates_the_store_layout(self) -> None:
        sm.pull_laya(_row())
        for sub in ("base", "inbox", "datasets", "runs"):
            self.assertTrue((self.store / sub).is_dir(), sub)

    def test_staging_is_removed_after_a_pull(self) -> None:
        sm.pull_laya(_row())
        self.assertEqual(list((self.store / ".staging").iterdir()), [])

    def test_a_verified_checkpoint_is_not_downloaded_again(self) -> None:
        sm.pull_laya(_row())
        sm.pull_laya(_row())
        self.assertEqual(len(self.calls), 1)

    def test_a_leftover_staging_dir_is_cleaned_on_the_next_pull(self) -> None:
        sm.pull_laya(_row())
        leftover = self.store / ".staging" / f"laya-test@{REV[:12]}"
        leftover.mkdir(parents=True)  # a run killed after its rename
        sm.pull_laya(_row())
        self.assertFalse(leftover.exists())

    def test_a_hash_mismatch_is_refused_and_leaves_nothing(self) -> None:
        self.serve["model.safetensors"] = b"weightz"  # same size, other bytes
        with self.assertRaises(SystemExit) as cm:
            sm.pull_laya(_row())
        self.assertIn("model.safetensors", str(cm.exception.code))
        self.assertFalse((self.store / "base" / f"laya-test@{REV[:12]}").exists())
        self.assertEqual(list((self.store / ".staging").iterdir()), [])

    def test_a_mismatching_existing_checkpoint_is_refused_not_overwritten(self) -> None:
        target = sm.pull_laya(_row())
        lc.make_writable(target)
        (target / "rl_agent_config.json").write_bytes(b"{}")
        with self.assertRaises(SystemExit) as cm:
            sm.pull_laya(_row())
        self.assertIn("does not match", str(cm.exception.code))
        self.assertEqual(len(self.calls), 1)

    def test_a_root_level_row_downloads_without_a_prefix(self) -> None:
        target = sm.pull_laya(_row(subfolder=""))
        self.assertEqual(sorted(self.calls[0][3:self.calls[0].index("--revision")]),
                         sorted(PAYLOAD))
        self.assertEqual(lc.verify_dir(target, _row(subfolder="")), [])


class NameDispatchTest(_LayaStoreCase):
    """`make model-pull NAME=<laya row>` goes through select-models.py --name."""

    def _main(self, *argv: str) -> None:
        with mock.patch.object(sys, "argv", ["select-models.py", *argv]):
            sm.main()

    def test_a_laya_name_pulls_from_the_laya_catalog(self) -> None:
        with mock.patch.object(sm, "pull_laya") as pull:
            self._main("--name", "laya-multilingual", "--download")
        self.assertEqual(pull.call_args.args[0]["name"], "laya-multilingual")

    def test_a_laya_dry_run_downloads_nothing(self) -> None:
        with mock.patch.object(sm, "pull_laya") as pull:
            self._main("--name", "laya-english", "--download", "--dry-run")
        pull.assert_not_called()

    def test_an_unknown_name_names_both_catalogs(self) -> None:
        with self.assertRaises(SystemExit) as cm:
            self._main("--name", "no-such-model", "--download")
        self.assertIn("laya-models.yaml", str(cm.exception.code))

    def test_an_unreadable_laya_catalog_is_an_error_message_not_a_traceback(self) -> None:
        with mock.patch.object(sm, "LAYA_CATALOG", self.store / "missing.yaml"), \
                self.assertRaises(SystemExit) as cm:
            self._main("--name", "no-such-model", "--download")
        self.assertIn("could not be read", str(cm.exception.code))


def _load_launcher():
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader("devai_agent_laya", str(REPO_ROOT / "bin" / "devai-agent"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


class LabMountTest(unittest.TestCase):
    """The lab sees the store at /laya, read-only except inbox/.

    Both places the lab starts from must agree: devai-agent's optional_mounts
    and the Makefile's MODEL_CACHE_MOUNT (make lab-*/shell-*).
    """

    def setUp(self) -> None:
        self.agent = _load_launcher()
        tmp = tempfile.mkdtemp(prefix="laya-mounts-")
        self.addCleanup(shutil.rmtree, tmp)
        self.store = Path(tmp) / "laya"
        self.agent.LAYA_STORE_DIR = self.store

    def _laya_mounts(self) -> list[str]:
        out = self.agent.optional_mounts()
        return [out[i + 1] for i, a in enumerate(out) if a == "-v" and "laya" in out[i + 1]]

    def test_store_read_only_and_inbox_read_write(self) -> None:
        (self.store / "inbox").mkdir(parents=True)
        self.assertEqual(self._laya_mounts(),
                         [f"{self.store}:/laya:ro", f"{self.store / 'inbox'}:/laya/inbox"])

    def test_no_inbox_no_writable_mount(self) -> None:
        self.store.mkdir()
        self.assertEqual(self._laya_mounts(), [f"{self.store}:/laya:ro"])

    def test_no_store_no_mount(self) -> None:
        self.assertEqual(self._laya_mounts(), [])

    def test_the_launcher_default_is_the_real_store(self) -> None:
        self.assertEqual(_load_launcher().LAYA_STORE_DIR, Path("/var/cache/devai/laya"))

    def test_makefile_mounts_the_same_way(self) -> None:
        mk = (REPO_ROOT / "Makefile").read_text()
        block = mk.split("\nMODEL_CACHE_MOUNT = ", 1)[1].split("\n\n", 1)[0]
        self.assertIn("-v $(CACHE_DIR)/laya:/laya:ro", block)
        self.assertIn("-v $(CACHE_DIR)/laya/inbox:/laya/inbox)", block)
        # Only inbox/ is writable: every other laya mount line ends in :ro.
        for line in block.splitlines():
            if "laya" in line and ":/laya/inbox)" not in line:
                self.assertIn(":ro)", line)


if __name__ == "__main__":
    unittest.main()
