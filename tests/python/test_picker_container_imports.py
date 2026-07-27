"""The picker must start inside the container, not just in the repo.

`bin/devai-agent` bind-mounts the HOST copy of model-picker.py over the
baked-in /usr/local/bin/model-picker, so a freshly edited picker
routinely runs inside an OLDER image. Adding a plain
`import _model_status` therefore took the picker down with
ModuleNotFoundError for every user until the image was rebuilt -- the
module existed in scripts/ and nowhere else.

Two independent things have to hold, and this file pins both:

  1. Any helper the picker imports must be installed next to it -- baked
     into the image AND bind-mounted by devai-agent AND symlinked by
     `make install`. Miss any one and it breaks in that layout only.

  2. Optional helpers must degrade rather than abort. Requirement 1 is
     satisfiable only after a rebuild; requirement 2 is what keeps the
     picker usable in the meantime.

Also pinned: the ledger path. _model_status derives LEDGER_PATH from
__file__, which is /usr/local/bin inside the container, so the lookup
lands on /usr/deploy and reads an EMPTY ledger -- silently, because the
ledger fails open. The picker resolves /etc/devai first, like the probe
caches.

Stdlib unittest only.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = REPO_ROOT / "scripts"

# Helpers the picker imports that must travel with it.
CO_LOCATED = ("_capability.py", "_model_status.py")


class PickerStartsWithoutOptionalHelpersTest(unittest.TestCase):
    """Simulates the exact break: new picker, old image."""

    def test_missing_model_status_does_not_abort(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copy(SCRIPTS / "model-picker.py", os.path.join(tmp, "model-picker"))
        shutil.copy(SCRIPTS / "_capability.py", os.path.join(tmp, "_capability.py"))
        # _model_status.py deliberately absent -- this is the older image.
        r = subprocess.run([sys.executable, os.path.join(tmp, "model-picker")],
                           capture_output=True, text=True, cwd=tmp)
        self.assertNotIn("ModuleNotFoundError", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_capability_is_still_a_hard_requirement(self):
        """Not everything should be optional. Capability constants are
        load-bearing for the menu; failing loudly there is correct."""
        src = (SCRIPTS / "model-picker.py").read_text()
        self.assertIn("from _capability import Capability", src)
        self.assertNotIn("try:\n    from _capability", src)


class HelpersAreInstalledEverywhereTest(unittest.TestCase):
    """One missing install path breaks exactly one layout, which is how
    this went unnoticed until a user ran devai-agent."""

    def test_baked_into_the_image(self):
        dockerfile = (REPO_ROOT / "deploy" / "Dockerfile.lab").read_text()
        for helper in CO_LOCATED:
            with self.subTest(helper=helper):
                self.assertIn(f"COPY scripts/{helper} /usr/local/bin/{helper}",
                              dockerfile)

    def test_bind_mounted_by_devai_agent(self):
        agent = (REPO_ROOT / "bin" / "devai-agent").read_text()
        for helper in CO_LOCATED:
            with self.subTest(helper=helper):
                self.assertIn(f"/usr/local/bin/{helper}", agent)

    def test_symlinked_by_make_install(self):
        mk = (REPO_ROOT / "Makefile").read_text()
        for helper in CO_LOCATED:
            with self.subTest(helper=helper):
                self.assertIn(f"scripts/{helper}", mk)

    def test_removed_by_make_uninstall(self):
        mk = (REPO_ROOT / "Makefile").read_text()
        for helper in CO_LOCATED:
            with self.subTest(helper=helper):
                self.assertIn(f"rm -f $(DEVAI_HOME)/{helper}", mk)


class LedgerPathResolutionTest(unittest.TestCase):
    """The container path must be tried first, or the ledger silently
    reads empty and every bench verdict is ignored."""

    def test_picker_resolves_etc_devai_before_the_repo(self):
        import importlib.util
        sys.path.insert(0, str(SCRIPTS))
        spec = importlib.util.spec_from_file_location(
            "mp_paths", SCRIPTS / "model-picker.py")
        mp = importlib.util.module_from_spec(spec)
        sys.modules["mp_paths"] = mp
        spec.loader.exec_module(mp)
        self.assertEqual(mp._MODEL_STATUS_PATHS[0],
                         "/etc/devai/.model-status.json")
        self.assertTrue(mp._MODEL_STATUS_PATHS[1].endswith(
            "deploy/.model-status.json"))

    def test_picker_does_not_use_the_modules_own_ledger_path(self):
        """_model_status.LEDGER_PATH is derived from __file__ and is wrong
        inside the container."""
        src = (SCRIPTS / "model-picker.py").read_text()
        self.assertNotIn("_MS.LEDGER_PATH", src)

    def test_ledger_is_mounted_into_the_container(self):
        agent = (REPO_ROOT / "bin" / "devai-agent").read_text()
        mk = (REPO_ROOT / "Makefile").read_text()
        for name, text in (("devai-agent", agent), ("Makefile", mk)):
            with self.subTest(where=name):
                self.assertIn("/etc/devai/.model-status.json", text)


if __name__ == "__main__":
    unittest.main()
