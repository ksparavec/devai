"""Model downloads are retried 3 times, then fail loudly.

Operator rule: every failure is an error, and a download is repeated 3 times
before bailing out. `make pull-images` and scripts/build-ollama.sh already
obeyed it; the model downloads in scripts/select-models.py -- the ONLY
sanctioned way to fetch a model -- failed loudly but tried exactly once.

It mattered on 2026-09-19: `make model-pull NAME=qwen3.8:27b-mtp-q4_K_M` was
running while a benchmark sent the router an explicit `<model>@<ctx>` pin. The
router recreated devai-ollama to move the tier, which killed the `ollama pull`
running inside it (`Error: unexpected EOF`, 39% into the first layer). Partial
blobs live in the store, so a retry simply resumes -- one attempt did not.

`ollama create` is deliberately NOT retried: it registers an already-staged
file, downloads nothing, and fails deterministically (a bad Modelfile does not
get better on the third try).

Stdlib unittest only; subprocess is faked, nothing is downloaded.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = REPO_ROOT / "scripts" / "select-models.py"


def _load():
    spec = importlib.util.spec_from_file_location("select_models_retries", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["select_models_retries"] = mod
    spec.loader.exec_module(mod)
    return mod


sm = _load()


class ModelPullRetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self._saved = (sm.subprocess.call, sm.PULL_RETRY_DELAY_SECONDS,
                       sm.GGUF_STAGING, sm.OLLAMA_STORE, sm.HF_STORES, sm.HF_STORE)
        sm.PULL_RETRY_DELAY_SECONDS = 0
        sm.OLLAMA_STORE = root / "ollama"
        sm.GGUF_STAGING = sm.OLLAMA_STORE / "models" / "_gguf"
        sm.HF_STORES = {"vllm": root / "vllm", "sglang": root / "sglang"}
        sm.HF_STORE = "vllm"
        self.addCleanup(self._restore)
        self.calls: list[list[str]] = []

    def _restore(self) -> None:
        (sm.subprocess.call, sm.PULL_RETRY_DELAY_SECONDS, sm.GGUF_STAGING,
         sm.OLLAMA_STORE, sm.HF_STORES, sm.HF_STORE) = self._saved

    def _fake(self, fail_first: int, *, kind: str):
        """subprocess.call that fails the first `fail_first` DOWNLOADS."""
        def call(argv, **kw):
            self.calls.append(list(argv))
            is_download = ("pull" in argv) or ("download" in argv)
            if not is_download:
                return self.create_rc
            n = sum(1 for c in self.calls if ("pull" in c) or ("download" in c))
            if n <= fail_first:
                return 1
            if kind == "gguf":   # a successful hf download leaves the file behind
                d = Path(argv[argv.index("--local-dir") + 1])
                d.mkdir(parents=True, exist_ok=True)
                (d / argv[3]).write_bytes(b"gguf")
            return 0
        self.create_rc = 0
        sm.subprocess.call = call

    def _downloads(self) -> int:
        return sum(1 for c in self.calls if ("pull" in c) or ("download" in c))

    # ── ollama library tags ─────────────────────────────────────────────────
    def test_ollama_pull_recovers_from_a_killed_attempt(self) -> None:
        self._fake(2, kind="ollama")
        sm.pull_ollama("qwen3.8:27b-mtp-q4_K_M")
        self.assertEqual(self._downloads(), 3)

    def test_ollama_pull_gives_up_after_exactly_three_attempts(self) -> None:
        self._fake(99, kind="ollama")
        with self.assertRaises(SystemExit) as cm:
            sm.pull_ollama("qwen3.8:27b-mtp-q4_K_M")
        self.assertEqual(self._downloads(), 3)
        self.assertIn("3 attempts", str(cm.exception))
        self.assertIn("qwen3.8:27b-mtp-q4_K_M", str(cm.exception))

    def test_a_clean_pull_runs_once(self) -> None:
        self._fake(0, kind="ollama")
        sm.pull_ollama("qwen3.8:27b-q4_K_M")
        self.assertEqual(self._downloads(), 1)

    # ── HF safetensors ──────────────────────────────────────────────────────
    def test_hf_download_is_retried_then_fails_loudly(self) -> None:
        self._fake(99, kind="hf")
        with self.assertRaises(SystemExit) as cm:
            sm.pull_hf("Some-NVFP4", "org/Some-NVFP4")
        self.assertEqual(self._downloads(), 3)
        self.assertIn("3 attempts", str(cm.exception))

    # ── GGUF: download retried, registration not ────────────────────────────
    def test_gguf_download_is_retried(self) -> None:
        self._fake(2, kind="gguf")
        sm.pull_gguf("fam:27b-x", "unsloth/M-GGUF", "M.gguf", "fam")
        self.assertEqual(self._downloads(), 3)

    def test_ollama_create_is_not_retried(self) -> None:
        self._fake(0, kind="gguf")
        self.create_rc = 1
        with self.assertRaises(SystemExit):
            sm.pull_gguf("fam:27b-x", "unsloth/M-GGUF", "M.gguf", "fam")
        creates = [c for c in self.calls if "create" in c]
        self.assertEqual(len(creates), 1,
                         "a registration failure is deterministic; retrying it "
                         "only delays the error")


if __name__ == "__main__":
    unittest.main()
