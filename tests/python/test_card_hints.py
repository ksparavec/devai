"""Card-derived backend hints (card-derived-hints-and-bench-sync Phase 1).

The discriminator rules in `scripts/_card_hints.py` were written after
reading five downloaded checkpoints, so their 5/5 agreement on those five
was in-sample and proved nothing. These tests pin the rules against
minimal fixtures -- including the families that were NOT used to write
them -- so a rule change has to stay right on all of them.

The load-bearing case is `qwen3_xml_family`: four probe-verified
checkpoints ship byte-identical tool markup and split evenly between two
different parsers, so the class derives NOTHING. A test asserts that,
because the tempting "fix" is to guess one, and guessing wrong means
mis-parsing live tool calls.

Stdlib unittest only; no network, no container, no GPU.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import _card_hints as CH  # noqa: E402

FIX = REPO_ROOT / "tests" / "fixtures" / "card-hints"


def _tpl(name: str) -> str:
    return (FIX / name).read_text()


class TestToolFormatRules(unittest.TestCase):
    def test_gemma4_inverted_delimiters(self):
        fmt, ev = CH.predict_tool_format(_tpl("gemma4.jinja"))
        self.assertEqual(fmt, "gemma4")
        self.assertIn("<|tool_call>", ev)

    def test_harmony_channel_commentary(self):
        fmt, _ = CH.predict_tool_format(_tpl("harmony.jinja"))
        self.assertEqual(fmt, "harmony")

    def test_nemotron_json_toolcall_block(self):
        fmt, _ = CH.predict_tool_format(_tpl("nemotron_json.jinja"))
        self.assertEqual(fmt, "nemotron_json")

    def test_hermes_is_tool_call_without_function(self):
        fmt, ev = CH.predict_tool_format(_tpl("hermes.jinja"))
        self.assertEqual(fmt, "hermes")
        self.assertIn("without", ev, "the evidence must state what is ABSENT")

    def test_qwen3_family_xml_is_recognised_but_ambiguous(self):
        for name in ("qwen3_xml_family_reasoning.jinja",
                     "qwen3_xml_family_coder.jinja"):
            fmt, _ = CH.predict_tool_format(_tpl(name))
            self.assertEqual(fmt, "qwen3_xml_family", name)

    def test_no_tool_markup_predicts_nothing(self):
        # Llama-3.1 and DeepSeek-R1-Distill ship no tool markup; their
        # parsers are engine-side formats. Inventing one here is how a
        # launch starts failing with "auto tool choice requires ...".
        fmt, ev = CH.predict_tool_format(_tpl("no_tool_markup.jinja"))
        self.assertIsNone(fmt)
        self.assertEqual(ev, "")

    def test_gemma4_precedes_the_generic_rules(self):
        # `<|tool_call>` CONTAINS `tool_call`; if ordering regressed, a
        # Gemma-4 template could be classified as hermes.
        fmt, _ = CH.predict_tool_format(
            _tpl("gemma4.jinja") + "\n<tool_call>{}</tool_call>\n")
        self.assertEqual(fmt, "gemma4")

    def test_empty_template_is_safe(self):
        self.assertEqual(CH.predict_tool_format(""), (None, ""))


class TestReasoningFormatRules(unittest.TestCase):
    def test_harmony_analysis_channel(self):
        fmt, _ = CH.predict_reasoning_format(_tpl("harmony.jinja"))
        self.assertEqual(fmt, "harmony")

    def test_gemma4_pipe_think_token(self):
        fmt, _ = CH.predict_reasoning_format(_tpl("gemma4.jinja"))
        self.assertEqual(fmt, "gemma4_think")

    def test_bare_think_is_think_delimited(self):
        fmt, _ = CH.predict_reasoning_format(_tpl("think_only.jinja"))
        self.assertEqual(fmt, "think_delimited")

    def test_no_reasoning_markup(self):
        self.assertEqual(
            CH.predict_reasoning_format(_tpl("no_tool_markup.jinja")),
            (None, ""))


class TestParserMapping(unittest.TestCase):
    def test_tool_and_reasoning_tables_are_separate(self):
        # Regression for the first implementation, which used one table:
        # harmony's vLLM TOOL parser is `openai` but its vLLM REASONING
        # parser is `openai_gptoss`, and a single map got one of them wrong.
        self.assertEqual(CH.parser_for("harmony", "vllm", "tool"), "openai")
        self.assertEqual(CH.parser_for("harmony", "vllm", "reasoning"),
                         "openai_gptoss")

    def test_sglang_harmony_names(self):
        self.assertEqual(CH.parser_for("harmony", "sglang", "tool"), "gpt-oss")
        self.assertEqual(CH.parser_for("harmony", "sglang", "reasoning"),
                         "gpt-oss")

    def test_qwen3_xml_family_derives_nothing(self):
        """The central negative result -- do not "fix" this by guessing.

        Qwen3.5-9B and Ornith are probe-verified `qwen3_xml`;
        Qwen3-Coder-30B and Nemotron-3-Nano-30B are probe-verified
        `qwen3_coder`. All four ship the same markup, so either guess is
        wrong half the time, and a wrong tool parser mis-parses live tool
        calls while no parser merely means the router strips tools.
        """
        for backend in ("vllm", "sglang"):
            self.assertIsNone(
                CH.parser_for("qwen3_xml_family", backend, "tool"),
                f"{backend}: ambiguous markup must derive no tool parser")

    def test_think_delimited_derives_no_reasoning_parser(self):
        # qwen3, deepseek-r1 and nemotron templates all emit bare <think>
        # but need qwen3 / deepseek_r1 / nemotron_v3 respectively.
        for backend in ("vllm", "sglang"):
            self.assertIsNone(
                CH.parser_for("think_delimited", backend, "reasoning"))

    def test_sglang_gaps_are_none_not_a_vllm_name(self):
        # SGLang has no nemotron_json / gemma4 analogue; emitting one
        # produces a launch argparse rejects.
        self.assertIsNone(CH.parser_for("nemotron_json", "sglang", "tool"))
        self.assertIsNone(CH.parser_for("gemma4", "sglang", "tool"))

    def test_unknown_class_is_none(self):
        self.assertIsNone(CH.parser_for(None, "vllm", "tool"))
        self.assertIsNone(CH.parser_for("not-a-format", "vllm", "tool"))


class TestTemplateLoading(unittest.TestCase):
    def test_prefers_standalone_jinja(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "chat_template.jinja").write_text("FROM_JINJA")
            (p / "tokenizer_config.json").write_text(
                json.dumps({"chat_template": "FROM_TOKENIZER"}))
            text, src = CH.load_chat_template(p)
            self.assertEqual(text, "FROM_JINJA")
            self.assertEqual(src, "chat_template.jinja")

    def test_falls_back_to_tokenizer_config_string(self):
        # NVIDIA-Nemotron-Nano-9B-v2-NVFP4 ships no .jinja at all.
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "tokenizer_config.json").write_text(
                json.dumps({"chat_template": "EMBEDDED"}))
            text, src = CH.load_chat_template(p)
            self.assertEqual(text, "EMBEDDED")
            self.assertEqual(src, "tokenizer_config.json")

    def test_handles_named_template_list_shape(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "tokenizer_config.json").write_text(json.dumps(
                {"chat_template": [
                    {"name": "tool_use", "template": "TOOLS"},
                    {"name": "default", "template": "DEFAULT"},
                ]}))
            text, _ = CH.load_chat_template(p)
            self.assertEqual(text, "DEFAULT", "the default entry wins")

    def test_absent_metadata_degrades_quietly(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(CH.load_chat_template(d), ("", ""))

    def test_malformed_json_does_not_raise(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "tokenizer_config.json").write_text("{not json")
            self.assertEqual(CH.load_chat_template(d), ("", ""))

    def test_missing_directory_does_not_raise(self):
        self.assertEqual(
            CH.load_chat_template("/nonexistent/path/xyz"), ("", ""))


class TestSamplingDefaults(unittest.TestCase):
    def test_reads_only_the_actionable_keys(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "generation_config.json").write_text(json.dumps({
                "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                "eos_token_id": [1, 2], "transformers_version": "4.1.0",
            }))
            got = CH.sampling_defaults(d)
            self.assertEqual(set(got),
                             {"temperature", "top_p", "top_k", "eos_token_id"})

    def test_absent_file_is_empty_dict(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(CH.sampling_defaults(d), {})


class TestHintsForModel(unittest.TestCase):
    def test_end_to_end_on_a_gemma4_shaped_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "chat_template.jinja").write_text(_tpl("gemma4.jinja"))
            (p / "generation_config.json").write_text(
                json.dumps({"temperature": 1.0}))
            h = CH.hints_for_model(p)
            self.assertEqual(h["tool_format"], "gemma4")
            self.assertEqual(h["tool_parser"]["vllm"], "gemma4")
            self.assertIsNone(h["tool_parser"]["sglang"])
            self.assertEqual(h["reasoning_format"], "gemma4_think")
            self.assertEqual(h["sampling"]["temperature"], 1.0)

    def test_end_to_end_on_a_bare_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            h = CH.hints_for_model(d)
            self.assertIsNone(h["tool_format"])
            self.assertEqual(h["template_bytes"], 0)
            self.assertEqual(h["sampling"], {})


if __name__ == "__main__":
    unittest.main()
