"""The laya trainer's wiring outside the router code (docs/plans/laya-trainer.md, Phase 3).

- compose runs devai-laya-trainer as a `sleep infinity` placeholder, like the
  other router-recreated backends: the logger follows it, and cache-up/-down
  know it;
- the router gets the trainer's env and the laya catalog (its allowlist);
- `make cache-down` removes the router-recreated container, so a job does
  not survive it holding the GPU; `make cache-up` skips the placeholder while
  the image is not built, so hosts without it keep working;
- the HF probers refuse to run while ANY GPU-holding container is up,
  including the trainer (and vllm-devai / ollama, which the list lacked).
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COMPOSE = yaml.safe_load((REPO_ROOT / "deploy" / "docker-compose.yaml").read_text())
MAKEFILE = (REPO_ROOT / "Makefile").read_text()


def _makefile_var(name: str) -> set[str]:
    line = MAKEFILE.split(f"\n{name} =", 1)[1]
    out = []
    for raw in line.splitlines():
        out.append(raw.rstrip().rstrip("\\").strip())
        if not raw.rstrip().endswith("\\"):
            break
    return set(" ".join(out).split())


def _recipe(target: str) -> str:
    return MAKEFILE.split(f"\n{target}:", 1)[1].split("\n\n", 1)[0]


class ComposeTest(unittest.TestCase):

    def test_placeholder_service(self) -> None:
        svc = COMPOSE["services"]["laya-trainer"]
        self.assertEqual(svc["container_name"], "devai-laya-trainer")
        self.assertEqual(svc["hostname"], "laya-trainer")
        self.assertEqual(svc["entrypoint"], ["sleep", "infinity"])
        self.assertEqual(svc["pull_policy"], "never")
        self.assertIn("localhost/devai-laya-trainer:latest", svc["image"])
        self.assertIn("/var/cache/devai/laya:/laya", svc["volumes"])
        self.assertTrue(any("DEVAI_GPU_DEVICE" in d for d in svc["devices"]))

    def test_router_env(self) -> None:
        env = dict(e.split("=", 1) for e in COMPOSE["services"]["router"]["environment"])
        self.assertEqual(env["LAYA_TRAINER_URL"], "http://laya-trainer:11434")
        self.assertEqual(env["LAYA_TRAINER_PORT"], "11438")
        self.assertEqual(env["LAYA_TRAINER_CONTAINER"], "devai-laya-trainer")
        self.assertIn("devai-laya-trainer", env["LAYA_TRAINER_IMAGE"])
        self.assertEqual(env["LAYA_STORE_DIR"], "/var/cache/devai/laya")
        self.assertEqual(env["LAYA_CATALOG_FILE"], "/etc/devai/laya-models.yaml")
        self.assertEqual(env["LAYA_MAX_HOLD_S"], "${LAYA_MAX_HOLD_S:-900}")

    def test_router_reads_the_laya_catalog(self) -> None:
        self.assertIn("./laya-models.yaml:/etc/devai/laya-models.yaml:ro",
                      COMPOSE["services"]["router"]["volumes"])


class MakefileTest(unittest.TestCase):

    def test_cache_up_and_down_know_the_trainer(self) -> None:
        self.assertIn("laya-trainer", _makefile_var("CACHE_SERVICES"))
        self.assertIn("laya-trainer", _makefile_var("CACHE_BACKEND_SERVICES"))
        rm_line = next(ln for ln in _recipe("cache-down").splitlines() if "for name in" in ln)
        self.assertIn("devai-laya-trainer", rm_line)

    def test_cache_up_skips_the_placeholder_without_its_image(self) -> None:
        recipe = _recipe("cache-up")
        self.assertIn("image exists $(LAYA_TRAINER_IMAGE)", recipe)
        self.assertIn("make build-laya-trainer", recipe)


class ProberMutexTest(unittest.TestCase):

    def test_every_gpu_holder_is_in_the_mutex_list(self) -> None:
        src = (REPO_ROOT / "scripts" / "_probe_hf_common.py").read_text()
        node = next(n for n in ast.walk(ast.parse(src))
                    if isinstance(n, ast.Assign)
                    and any(getattr(t, "id", "") == "MUTEX_CONTAINERS" for t in n.targets))
        names = set(ast.literal_eval(node.value))
        for c in ("devai-router", "devai-vllm", "devai-sglang", "devai-vllm-devai",
                  "devai-ollama", "devai-laya-trainer"):
            self.assertIn(c, names)


class LoggerTest(unittest.TestCase):

    def test_fallback_targets_include_the_trainer(self) -> None:
        src = (REPO_ROOT / "deploy" / "logging.sh").read_text()
        fallback = next(ln for ln in src.splitlines() if ln.strip().startswith("LOG_TARGETS=\"devai-"))
        self.assertIn("devai-laya-trainer", fallback)


if __name__ == "__main__":
    unittest.main()
