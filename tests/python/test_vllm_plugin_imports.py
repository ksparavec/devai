"""The two tool-parser plugins import on every vLLM devai serves.

vLLM 0.29 moved the protocol classes the plugins import (DeltaMessage,
ToolCall, ...) to vllm.entrypoints.generate.base.protocol and removed
vllm.entrypoints.openai.engine.protocol (vllm#54492). vLLM imports a plugin
file and logs a failure instead of raising (0.30.0
vllm/utils/import_utils.py:93-103), so a plugin that cannot import simply
never registers its parser, and the server then refuses
`--tool-call-parser deepseek_string` / `nemotron_json` at startup, before the
model loads (platform-rebuild Phase 4 step 4v).

Each plugin imports from the new module and falls back to the old one, so
one file serves stock 0.22.1, 0.28.0 and 0.30.0. Pinned here against stub
`vllm` packages in both layouts (no vLLM on the host): the parser registers
in each, and its classes come from that layout's module. The real check
belongs to 4v: each plugin file imported in the new image.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PLUGINS = REPO_ROOT / "scripts" / "vllm_plugins"
NEW_PROTOCOL = "vllm.entrypoints.generate.base.protocol"    # vLLM >= 0.29
OLD_PROTOCOL = "vllm.entrypoints.openai.engine.protocol"    # vLLM <= 0.28
PROTOCOL_CLASSES = ("DeltaFunctionCall", "DeltaMessage", "DeltaToolCall",
                    "ExtractedToolCallInformation", "FunctionCall", "ToolCall")


def _package(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []           # a package, so submodules resolve through it
    return mod


def _stub_vllm(protocol_module: str) -> tuple[dict[str, types.ModuleType], dict[str, type]]:
    """sys.modules entries for a vllm whose protocol classes live in
    `protocol_module`, plus the registry the stub ToolParserManager fills."""
    registered: dict[str, type] = {}

    class ToolParserManager:
        @staticmethod
        def register_module(names):
            def deco(cls):
                for n in ([names] if isinstance(names, str) else names):
                    registered[n] = cls
                return cls
            return deco

    class ToolParser:
        def __init__(self, tokenizer=None, tools=None):
            self.model_tokenizer = tokenizer

    mods: dict[str, types.ModuleType] = {}
    parts = protocol_module.split(".")
    for i in range(1, len(parts)):
        name = ".".join(parts[:i])
        mods[name] = _package(name)
    for name in ("vllm.entrypoints.openai", "vllm.entrypoints.openai.chat_completion",
                 "vllm.tool_parsers", "vllm.tokenizers"):
        mods.setdefault(name, _package(name))
    protocol = types.ModuleType(protocol_module)
    for cls in PROTOCOL_CLASSES:
        setattr(protocol, cls, type(cls, (), {"__module__": protocol_module}))
    mods[protocol_module] = protocol

    leaf = {
        "vllm.logger": {"init_logger": logging.getLogger},
        "vllm.tool_parsers.abstract_tool_parser": {
            "ToolParser": ToolParser, "ToolParserManager": ToolParserManager},
        "vllm.entrypoints.openai.chat_completion.protocol": {
            "ChatCompletionRequest": type("ChatCompletionRequest", (), {})},
        "vllm.tokenizers.mistral": {"MistralTokenizer": type("MistralTokenizer", (), {})},
    }
    for name, attrs in leaf.items():
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        mods[name] = mod

    # nemotron_json's third-party imports, as stubs (not all on the host).
    pjp = _package("partial_json_parser")
    core = _package("partial_json_parser.core")
    options = types.ModuleType("partial_json_parser.core.options")
    options.Allow = types.SimpleNamespace(ALL=7, STR=1)
    pydantic = types.ModuleType("pydantic")
    pydantic.Field = lambda *a, **k: None
    mods.update({"partial_json_parser": pjp, "partial_json_parser.core": core,
                 "partial_json_parser.core.options": options, "regex": re,
                 "pydantic": pydantic})
    return mods, registered


def _load(plugin: str, protocol_module: str):
    mods, registered = _stub_vllm(protocol_module)
    # Drop any real vllm / stub left in sys.modules so only this layout exists.
    clean = {k: v for k, v in sys.modules.items() if k != "vllm" and not k.startswith("vllm.")}
    with mock.patch.dict(sys.modules, clean, clear=True):
        sys.modules.update(mods)
        spec = importlib.util.spec_from_file_location(f"_devai_plugin_{plugin}", PLUGINS / plugin)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module, registered


class PluginImportTest(unittest.TestCase):
    CASES = (("deepseek_string_tool_parser.py", "deepseek_string"),
             ("nemotron_json_tool_parser.py", "nemotron_json"))

    def test_each_plugin_registers_on_the_new_layout(self) -> None:
        for plugin, parser in self.CASES:
            with self.subTest(plugin=plugin):
                module, registered = _load(plugin, NEW_PROTOCOL)
                self.assertIn(parser, registered)
                self.assertEqual(module.ToolCall.__module__, NEW_PROTOCOL)

    def test_each_plugin_falls_back_to_the_old_layout(self) -> None:
        for plugin, parser in self.CASES:
            with self.subTest(plugin=plugin):
                module, registered = _load(plugin, OLD_PROTOCOL)
                self.assertIn(parser, registered)
                self.assertEqual(module.ToolCall.__module__, OLD_PROTOCOL)

    def test_the_fallback_names_the_same_classes_as_the_new_import(self) -> None:
        for plugin, _ in self.CASES:
            src = (PLUGINS / plugin).read_text()
            new = re.search(rf"from {re.escape(NEW_PROTOCOL)} import \(([^)]*)\)", src)
            old = re.search(rf"from {re.escape(OLD_PROTOCOL)} import \(([^)]*)\)", src)
            self.assertIsNotNone(new, plugin)
            self.assertIsNotNone(old, plugin)
            self.assertEqual(re.findall(r"\w+", new.group(1)), re.findall(r"\w+", old.group(1)), plugin)
            self.assertLess(new.start(), old.start(), f"{plugin}: the new module goes first")


if __name__ == "__main__":
    unittest.main()
