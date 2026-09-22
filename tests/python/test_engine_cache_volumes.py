"""The HF engines' JIT caches must survive a container recreate.

vLLM and SGLang JIT-compile at first use -- FlashInfer builds its attention
kernels with nvcc under ~/.cache/flashinfer/<version>/, vLLM keeps
torch.compile artifacts under ~/.cache/vllm, SGLang under ~/.cache/sglang
-- and the router recreates a backend container on every model or context
switch. With nothing bound under ~/.cache every launch re-paid the whole
compile: 269 s cold start measured 2026-09-22 for
Qwen3.8-27B-MTP-devai-NVFP4 on vllm-devai. Both launch paths -- the
router's libpod spec (gpu-arbiter/engine_cache.go) and the probers'
`podman run` (scripts/_probe_hf_common.py) -- now mount one podman named
volume per cache and backend. The name format and the container paths
must agree across the two languages or a probe warms a cache the router
never reads; this file pins both sides.

Stdlib unittest only.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _probe_hf_common as hf  # noqa: E402

GO_SRC = (REPO_ROOT / "gpu-arbiter" / "engine_cache.go").read_text()
GO_MAIN = (REPO_ROOT / "gpu-arbiter" / "main.go").read_text()


class EngineCacheVolumesTest(unittest.TestCase):
    def test_hf_backends_get_flashinfer_and_engine_volumes(self) -> None:
        self.assertEqual(hf.engine_cache_volumes("vllm-devai"), [
            ("devai-engine-cache-vllm-devai-flashinfer", "/root/.cache/flashinfer"),
            ("devai-engine-cache-vllm-devai-vllm", "/root/.cache/vllm"),
        ])
        self.assertEqual(hf.engine_cache_volumes("sglang"), [
            ("devai-engine-cache-sglang-flashinfer", "/root/.cache/flashinfer"),
            ("devai-engine-cache-sglang-sglang", "/root/.cache/sglang"),
        ])
        self.assertEqual(hf.engine_cache_volumes("vllm")[1],
                         ("devai-engine-cache-vllm-vllm", "/root/.cache/vllm"))

    def test_ollama_gets_none(self) -> None:
        self.assertEqual(hf.engine_cache_volumes("ollama"), [])

    def test_stock_and_custom_vllm_never_share_a_volume(self) -> None:
        stock = {v for v, _ in hf.engine_cache_volumes("vllm")}
        custom = {v for v, _ in hf.engine_cache_volumes("vllm-devai")}
        self.assertFalse(stock & custom)

    def test_probe_run_args_mount_the_volumes(self) -> None:
        args = hf.container_run_args(
            "podman", "devai-probe-sglang", "img", 18000, "/var/cache/devai/sglang",
            {"X": "1"}, "python3", ["-m", "sglang"], backend="sglang")
        vols = [args[i + 1] for i, a in enumerate(args) if a == "--volume"]
        self.assertIn("devai-engine-cache-sglang-flashinfer:/root/.cache/flashinfer:rw", vols)
        self.assertIn("devai-engine-cache-sglang-sglang:/root/.cache/sglang:rw", vols)
        self.assertEqual(vols[0], "/var/cache/devai/sglang:/models:ro")
        self.assertEqual(args[-3:], ["img", "-m", "sglang"])

    def test_probe_run_args_without_a_backend_are_unchanged(self) -> None:
        args = hf.container_run_args(
            "podman", "n", "img", 18000, "/m", {}, "e", [])
        self.assertNotIn("devai-engine-cache", " ".join(args))

    def test_both_probers_pass_their_backend(self) -> None:
        for f in ("_probe_hf_common.py", "_probe_load.py"):
            src = (REPO_ROOT / "scripts" / f).read_text()
            call = re.search(
                r"container_run_detached\(\n\s+runtime, container_name.*?\)\n", src, re.S)
            self.assertIsNotNone(call, f)
            self.assertIn("backend=spec.name", call.group(0), f)

    def test_go_side_uses_the_same_names_and_paths(self) -> None:
        m = re.search(r'engineCacheVolumePrefix = "([^"]+)"', GO_SRC)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), hf.ENGINE_CACHE_VOLUME_PREFIX)
        self.assertIn('"Dest": "/root/.cache/flashinfer"', GO_SRC)
        self.assertIn('"Dest": "/root/.cache/" + engine', GO_SRC)
        self.assertIn('backend + "-" + cache', GO_SRC)
        self.assertRegex(GO_MAIN, r'"volumes":\s+engineCacheVolumes\(cfg\.Name\)')


if __name__ == "__main__":
    unittest.main()
