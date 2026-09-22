"""The picker's Claude Code environment must not put Claude Code into
gateway mode.

Claude Code 2.1.278 (the version `make fetch-cli` pulled on 2026-09-22)
validates ANTHROPIC_BASE_URL at startup whenever CLAUDE_CODE_USE_GATEWAY is
set: plain http:// is refused unless the hostname is literally `localhost`,
`127.0.0.1` or `[::1]`, with no override --

    CLAUDE_CODE_USE_GATEWAY is set but ANTHROPIC_BASE_URL is invalid:
    Gateway URL must use https:// (got http://). Plain HTTP is only
    allowed for localhost during development.

-- and the router is `http://devai-router:<port>` from inside the lab, so
every Claude launch died before its first request. The variable had been
set because on 2.1.220 the /v1/models discovery was gated on it; from
2.1.265 it selects the enterprise Cloud-gateway sign-in instead, and on
2.1.278 discovery runs under the first-party provider with only
CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY, a non-Anthropic
ANTHROPIC_BASE_URL and a token. Verified on the wire: without the gateway
variable the router logged the discovery GET from the claude-code UA and
the turn completed.

Stdlib unittest only.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PICKER_SRC = REPO_ROOT / "scripts" / "model-picker.py"

_ENV_KEYS = (
    "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_SMALL_FAST_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL", "CLAUDE_CODE_USE_GATEWAY",
    "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
    "CONTEXT",
)


def _load():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("model_picker_claude", PICKER_SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_picker_claude"] = mod
    spec.loader.exec_module(mod)
    return mod


PICKER = _load()


class ClaudeEnvContractTest(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in _ENV_KEYS}
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        self.addCleanup(self._restore)

    def _restore(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_gateway_mode_is_not_set(self):
        PICKER._build("claude", "Qwen3.8-27B-MTP-devai-NVFP4::mtp@118784", "vllm-devai")
        self.assertNotIn("CLAUDE_CODE_USE_GATEWAY", os.environ)

    def test_discovery_stays_on_with_a_plain_http_router_url(self):
        cmd = PICKER._build("claude", "Qwen3.8-27B-MTP-devai-NVFP4::mtp@118784", "vllm-devai")
        self.assertEqual(os.environ["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"], "1")
        self.assertTrue(os.environ["ANTHROPIC_BASE_URL"].startswith("http://"))
        self.assertTrue(os.environ["ANTHROPIC_BASE_URL"].endswith(":11437"))
        self.assertEqual(os.environ["ANTHROPIC_AUTH_TOKEN"], "local")
        self.assertEqual(cmd[:2], ["claude", "--model"])

    def test_context_window_is_declared_from_the_selected_tier(self):
        # 2.1.278 assumes 200k for an unknown model id; on a 116K row that
        # means no auto-compact before the engine rejects the prompt. The
        # picker sets CONTEXT right before _build.
        os.environ["CONTEXT"] = "118784"
        PICKER._build("claude", "Qwen3.8-27B-MTP-devai-NVFP4::mtp@118784", "vllm-devai")
        self.assertEqual(os.environ["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "118784")

    def test_context_window_is_left_alone_without_a_tier_or_when_preset(self):
        PICKER._build("claude", "gpt-oss-20b", "ollama")
        self.assertNotIn("CLAUDE_CODE_MAX_CONTEXT_TOKENS", os.environ)
        os.environ["CONTEXT"] = "32768"
        os.environ["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = "200000"
        PICKER._build("claude", "gpt-oss-20b", "ollama")
        self.assertEqual(os.environ["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "200000")

    def test_source_never_sets_the_gateway_variable(self):
        src = PICKER_SRC.read_text()
        for line in src.splitlines():
            code = line.split("#", 1)[0]
            self.assertNotIn('"CLAUDE_CODE_USE_GATEWAY"', code, line)


if __name__ == "__main__":
    unittest.main()
