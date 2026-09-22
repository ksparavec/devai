"""devai-agent must say so when a backend's probe cache is not staged.

The launcher bind-mounts `~/.devai/.<backend>-reasoning-cache.json` for
every backend it knows and silently skipped any file that did not exist
("absent backend caches are tolerated downstream"). Downstream, the picker
then simply had no rows for that backend. On 2026-09-22 the vllm-devai
symlink had not been staged (`make install` not re-run after the backend
was added), and the operator's report was "model picker is not showing
devai models at all" -- with nothing anywhere saying why.
"""

from __future__ import annotations

import contextlib
import re
import importlib.util
import io
import tempfile
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_launcher():
    loader = SourceFileLoader("devai_agent_cache_warning", str(REPO_ROOT / "bin" / "devai-agent"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


class MissingCacheWarningTest(unittest.TestCase):
    def test_missing_backend_cache_is_named_on_stderr(self) -> None:
        mod = _load_launcher()
        with tempfile.TemporaryDirectory() as td:
            staged = Path(td) / ".vllm-reasoning-cache.json"
            staged.write_text("{}")
            mod.PROBE_CACHES = {"vllm": staged, "vllm-devai": Path(td) / ".vllm-devai-reasoning-cache.json"}
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                cmd = mod.build_run_cmd(
                    runtime="podman", image="devai-lab-gpu", cpu_only=True,
                    work_dir=Path(td), prefs=dict(mod.DEFAULTS),
                    pref_model=None, pref_agent=None,
                )
        joined = " ".join(cmd)
        self.assertIn("/etc/devai/.vllm-reasoning-cache.json", joined)
        self.assertNotIn("/etc/devai/.vllm-devai-reasoning-cache.json", joined)
        self.assertIn("vllm-devai", err.getvalue())
        self.assertIn("make install", err.getvalue())


if __name__ == "__main__":
    unittest.main()


class CatalogMountTest(unittest.TestCase):
    """The launcher must bind-mount the repo catalog over the image's copy.

    The picker's MTP offer, its PARAMS/TYPE/FORMAT columns and its knowledge
    of rows added after the image was built all come from deploy/models.yaml.
    The launcher mounted the probe caches and the picker itself but not the
    catalog, so the container read the copy baked at image build time --
    2026-07-27 on this host: zero derived rows, zero `mtp:` blocks -- and
    every MTP-capable row showed MTP "No" in the picker (2026-09-22).
    """

    def test_staged_catalog_is_mounted_at_the_path_the_picker_reads(self) -> None:
        mod = _load_launcher()
        with tempfile.TemporaryDirectory() as td:
            cat = Path(td) / "models.yaml"
            cat.write_text("models: []\n")
            mod.PROBE_CACHES = {}
            mod.CATALOG = cat
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                cmd = mod.build_run_cmd(
                    runtime="podman", image="devai-lab-gpu", cpu_only=True,
                    work_dir=Path(td), prefs=dict(mod.DEFAULTS),
                    pref_model=None, pref_agent=None,
                )
            self.assertIn(f"{cat}:/etc/devai/models.yaml:ro", " ".join(cmd))
            mod.CATALOG = Path(td) / "absent.yaml"
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                cmd = mod.build_run_cmd(
                    runtime="podman", image="devai-lab-gpu", cpu_only=True,
                    work_dir=Path(td), prefs=dict(mod.DEFAULTS),
                    pref_model=None, pref_agent=None,
                )
            self.assertNotIn("/etc/devai/models.yaml", " ".join(cmd))
            self.assertIn("models.yaml", err.getvalue())
            self.assertIn("make install", err.getvalue())

    def test_make_install_stages_and_uninstall_removes_the_catalog_link(self) -> None:
        mk = (REPO_ROOT / "Makefile").read_text()
        install = re.search(r"^install:.*?(?=^\S)", mk, re.M | re.S).group(0)
        uninstall = re.search(r"^uninstall:.*?(?=^\S)", mk, re.M | re.S).group(0)
        self.assertRegex(install, r'ln -sf "\$\(CURDIR\)/deploy/models\.yaml" \$\(DEVAI_HOME\)/models\.yaml')
        self.assertIn("rm -f $(DEVAI_HOME)/models.yaml", uninstall)
