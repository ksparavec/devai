"""Staging a model for the second HF backend must link, not re-download.

vLLM and SGLang read separate directories. Before the 2026-07 storage
change they were separate volumes on separate filesystems, so the only
way to serve a model from both was to download it twice -- and CLAUDE.md
plus docs/backends.md both said so in as many words. They now share one
filesystem (`/var/cache/devai`, vgais-cache), which makes the second copy
pure waste: 57.5 GiB across the five kept models on this fleet, and 13 GiB
of network for gpt-oss-20b alone.

Hard links, not symlinks. Each engine's container bind-mounts only its
OWN store, so a symlink pointing into the peer store resolves to nothing
inside the container. A hard link is a second name for the same inode and
needs no cooperation from the mount namespace.

Three things are pinned here:
  1. link_tree's accounting -- specifically that a re-run reports zero
     reclaimed rather than the full model size. The dry run exists to
     tell an operator how much space they will get back; a number that
     ignores existing links is worse than no number.
  2. The excludes, which must match what select-models passes to
     `hf download` or the linked tree misrepresents the store.
  3. That select-models' download path actually consults the peer store,
     and that every way the link can be inapplicable degrades to a
     download instead of an error.

Stdlib unittest only.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load(name: str, rel: str):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lk = _load("link_hf_store", "scripts/link-hf-store.py")


def _model(root: Path, name: str, files: dict[str, bytes]) -> Path:
    d = root / name
    for rel, blob in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(blob)
    (d / "config.json").write_text("{}")
    return d


class LinkTreeAccountingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.src = _model(self.root / "vllm", "M", {
            "model.safetensors": b"x" * 4096,
            "tokenizer.json": b"y" * 512,
        })
        self.dst = self.root / "sglang" / "M"

    def test_first_run_reclaims_every_byte(self):
        files, reclaimed, shared = lk.link_tree(
            self.src, self.dst, self.dst, dry_run=False)
        self.assertEqual(files, 3)          # 2 + config.json
        self.assertEqual(shared, 0)
        self.assertEqual(reclaimed, 4096 + 512 + 2)

    def test_rerun_reports_nothing_reclaimed(self):
        """The regression that motivated the accounting split: a dry run
        against an already-linked store claimed the full 57.5 GiB."""
        lk.link_tree(self.src, self.dst, self.dst, dry_run=False)
        files, reclaimed, shared = lk.link_tree(
            self.src, self.dst, self.dst, dry_run=True)
        self.assertEqual(files, 3)
        self.assertEqual(shared, 3, "existing hard links were not detected")
        self.assertEqual(reclaimed, 0)

    def test_linked_files_share_an_inode(self):
        lk.link_tree(self.src, self.dst, self.dst, dry_run=False)
        a = self.src / "model.safetensors"
        b = self.dst / "model.safetensors"
        self.assertTrue(b.exists())
        self.assertFalse(b.is_symlink(), "must be a hard link, not a symlink")
        self.assertTrue(os.path.samefile(a, b))
        self.assertGreaterEqual(b.stat().st_nlink, 2)

    def test_a_stale_destination_copy_is_replaced_and_counted(self):
        """A real (unlinked) copy already sitting there IS reclaimable --
        it must be counted, and replaced by the link."""
        self.dst.mkdir(parents=True)
        (self.dst / "model.safetensors").write_bytes(b"z" * 4096)
        _, reclaimed, shared = lk.link_tree(
            self.src, self.dst, self.dst, dry_run=False)
        self.assertEqual(shared, 0)
        self.assertEqual(reclaimed, 4096 + 512 + 2)
        self.assertTrue(os.path.samefile(
            self.src / "model.safetensors", self.dst / "model.safetensors"))

    def test_dry_run_writes_nothing(self):
        lk.link_tree(self.src, self.dst, self.dst, dry_run=True)
        self.assertFalse(self.dst.exists())


class ExcludeParityTest(unittest.TestCase):
    """The linked tree must hold what a download would have produced."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.src = _model(self.root / "vllm", "M", {
            "model.safetensors": b"x" * 16,
            "original/consolidated.pth": b"big",
            "metal/model.bin": b"big",
            ".cache/junk": b"j",
            "model.gguf": b"g",
        })
        self.dst = self.root / "sglang" / "M"

    def test_alternative_format_trees_are_not_linked(self):
        lk.link_tree(self.src, self.dst, self.dst, dry_run=False)
        for skipped in ("original", "metal", ".cache", "model.gguf"):
            self.assertFalse((self.dst / skipped).exists(),
                             f"{skipped} should not be in the linked tree")
        self.assertTrue((self.dst / "model.safetensors").exists())

    def test_excludes_match_the_download_path(self):
        sm = (REPO_ROOT / "scripts" / "select-models.py").read_text()
        for pat in ("original/*", "metal/*", "*.gguf"):
            self.assertIn(pat, sm,
                          "link excludes drifted from the hf download excludes")


class SelectModelsUsesThePeerStoreTest(unittest.TestCase):
    """The one-off script is not enough -- the normal download path has
    to reach for it, or the next operator downloads 57 GiB again."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.vllm = self.root / "vllm"
        self.sglang = self.root / "sglang"
        self.sglang.mkdir(parents=True)
        _model(self.vllm, "Present-9B", {"model.safetensors": b"x" * 32})

        self.sm = _load("select_models", "scripts/select-models.py")
        self.sm.HF_STORES.update({"vllm": self.vllm, "sglang": self.sglang})
        self.sm.VLLM_MODELS, self.sm.SGLANG_MODELS = self.vllm, self.sglang
        self.sm.HF_STORE = "sglang"

    def test_links_when_the_peer_has_it(self):
        self.assertTrue(
            self.sm.try_link_from_peer_store("Present-9B", "sglang"))
        self.assertTrue(os.path.samefile(
            self.vllm / "Present-9B" / "model.safetensors",
            self.sglang / "Present-9B" / "model.safetensors"))

    def test_falls_back_to_download_when_the_peer_lacks_it(self):
        self.assertFalse(
            self.sm.try_link_from_peer_store("Absent-9B", "sglang"))

    def test_falls_back_when_the_peer_copy_is_half_downloaded(self):
        """A directory with no config.json is not loadable; linking it
        would produce a store gap that looks like a real model."""
        (self.vllm / "Partial-9B").mkdir()
        self.assertFalse(
            self.sm.try_link_from_peer_store("Partial-9B", "sglang"))

    def test_falls_back_across_filesystems_rather_than_erroring(self):
        """The pre-2026-07 layout, and any host that splits the stores
        again. A slow download beats a failed provision."""
        real_stat = Path.stat
        vllm = self.vllm

        class FakeStat:
            st_dev = 999999

        def fake_stat(self, *a, **kw):
            if self == vllm:
                return FakeStat()
            return real_stat(self, *a, **kw)

        Path.stat = fake_stat
        self.addCleanup(setattr, Path, "stat", real_stat)
        self.assertFalse(
            self.sm.try_link_from_peer_store("Present-9B", "sglang"))

    def test_pull_hf_short_circuits_on_a_successful_link(self):
        """If the link happened, no download may be attempted."""
        called = []
        self.sm.pull_hf("Present-9B", "org/Present-9B")
        orig = self.sm.try_link_from_peer_store
        self.sm.try_link_from_peer_store = lambda *a: called.append(a) or True
        self.addCleanup(setattr, self.sm, "try_link_from_peer_store", orig)
        self.sm.pull_hf("Present-9B", "org/Present-9B")
        self.assertEqual(len(called), 1)


class DocsMatchRealityTest(unittest.TestCase):
    """Both files asserted the stores could not be hard-linked. That was
    true when written and is now false; a stale storage claim sends the
    next operator down a 57 GiB detour."""

    def test_no_doc_still_claims_hardlinks_are_impossible(self):
        for rel in ("CLAUDE.md", "docs/backends.md"):
            text = (REPO_ROOT / rel).read_text()
            with self.subTest(doc=rel):
                self.assertNotIn("cannot be hardlinked", text)
                self.assertIn("link-hf-store.py", text,
                              f"{rel} does not mention the linking path")

    def test_makefile_exposes_the_target(self):
        mk = (REPO_ROOT / "Makefile").read_text()
        self.assertIn("hf-link:", mk)
        self.assertIn("link-hf-store.py", mk)


# Variables the router reads that must NOT be forwarded, each with the
# reason. An exemption list is only honest if every entry justifies
# itself -- otherwise it becomes the place bugs go to hide.
EXEMPT_FROM_COMPOSE = {
    # Only a fallback for the --mode flag, and single is the ONLY
    # supported topology: any other value exits with a pointer to
    # attic/README.md. Leaving it unset means the router takes its
    # built-in "single" default, which is what we want. Forwarding it
    # would let a stale DEVAI_MODE=worker in someone's .env turn every
    # router start into a fatal error.
    "DEVAI_MODE",
}


class RouterEnvKnobsAreForwardedTest(unittest.TestCase):
    """A knob the router reads but compose does not forward is inert --
    and silently so, which is the worst shape for a documented setting.

    The repo already carries one instance of this class of bug (the
    router container is never handed DEVAI_GPU_DEVICE, per
    docs/gpu-vendors.md), so it is worth a test rather than a habit.
    """

    def _router_env(self) -> list[str]:
        import yaml
        c = yaml.safe_load(
            (REPO_ROOT / "deploy" / "docker-compose.yaml").read_text())
        return [str(e) for e in c["services"]["router"]["environment"]]

    def test_every_devai_env_knob_the_router_reads_is_forwarded(self):
        """Generalised deliberately. Scoping this to DEVAI_SSE_* would
        have passed while DEVAI_GPU_DEVICE -- the first instance of this
        bug, and the more damaging one -- stayed broken."""
        go = (REPO_ROOT / "gpu-arbiter" / "main.go").read_text()
        env = "\n".join(self._router_env())
        import re
        names = set(re.findall(r'env\w*\("(DEVAI_[A-Z_]+)"', go))
        self.assertTrue(names, "no DEVAI_* env knobs found in main.go")
        for name in sorted(names - EXEMPT_FROM_COMPOSE):
            with self.subTest(var=name):
                self.assertIn(name, env,
                              f"{name} is read by the router but never "
                              f"forwarded by compose -- it would be inert")

    def test_gpu_device_reaches_the_router(self):
        """vLLM and SGLang start as `sleep infinity` placeholders and are
        ALWAYS recreated by the router, so this variable not reaching the
        router meant the amd overlay never reached the two services it
        matters most for."""
        self.assertTrue(
            any(e.startswith("DEVAI_GPU_DEVICE=") for e in self._router_env()),
            "the router recreates every HF backend container; without this "
            "it stamps the hardcoded nvidia.com/gpu=all on all of them")

    def test_knobs_are_documented_for_operators(self):
        example = (REPO_ROOT / ".env.example").read_text()
        for name in ("DEVAI_SSE_KEEPALIVE_SECONDS",
                     "DEVAI_SSE_KEEPALIVE_GRACE_SECONDS"):
            self.assertIn(name, example)


if __name__ == "__main__":
    unittest.main()
