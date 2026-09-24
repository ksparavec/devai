"""Shared test helpers."""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from .. import catalog as cat

HAS_LAYA = all(importlib.util.find_spec(m) is not None
               for m in ("torch", "laya", "transformers", "onnxruntime", "onnxscript"))
needs_laya = unittest.skipUnless(HAS_LAYA, "torch/laya/onnx not installed (run make test-laya-trainer)")


def tmpdir(case: unittest.TestCase, prefix: str = "laya-") -> Path:
    d = Path(tempfile.mkdtemp(prefix=prefix))

    def cleanup() -> None:
        cat.make_writable(d)
        shutil.rmtree(d)

    case.addCleanup(cleanup)
    return d


def write_catalog(path: Path, *rows: dict) -> Path:
    path.write_text(yaml.safe_dump({"schema_version": 1, "models": list(rows)}, sort_keys=False))
    return path


class FixtureStore:
    """A laya store with the tiny fixture checkpoint installed read-only in base/."""

    def __init__(self, root: Path) -> None:
        from .. import fixture

        self.store = root / "laya"
        for sub in ("base", "inbox", "datasets", "runs"):
            (self.store / sub).mkdir(parents=True)
        build = root / "build"
        self.row = fixture.build_checkpoint(build)
        self.base_dir = self.store / "base" / cat.checkpoint_dirname(self.row)
        shutil.copytree(build, self.base_dir)
        cat.make_read_only(self.base_dir)
        self.catalog = write_catalog(root / "laya-models.yaml", self.row)
        from ..dataset import load_tokenizer
        import json
        self.cfg = json.loads((self.base_dir / "rl_agent_config.json").read_text())
        self.tok = load_tokenizer(self.base_dir, self.cfg)

    def dataset(self, **kw) -> str:
        from .. import fixture
        return fixture.build_dataset(self.store, self.row, self.tok, **kw)
