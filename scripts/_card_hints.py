"""Derive backend hints from a checkpoint's own metadata.

Every downloaded HF checkpoint ships the markup it was trained to emit --
in `chat_template.jinja` (or embedded in `tokenizer_config.json`) -- plus
its sampling defaults in `generation_config.json`. The probers read
`config.json` and nothing else, so all of that is discarded and the same
information is instead hand-curated into the `parsers:` blocks of
`scripts/model-families.yaml`.

The cost of that is recorded in the repo: `model-families.yaml` notes the
Gemma-4 tool parser had to be inferred from a sibling family, and until it
was, "the router strips tools/tool_choice and the model scored 0 on the
tools bench despite tool-calling fine (its Ollama GGUF twin scored
tools=1.00)". That value is mechanically recoverable from the template.

This module is READ-ONLY and changes no launch argument. It exists so the
prediction can be measured against the curated values before anything
depends on it (plan Phase 1); wiring it in as a fallback is Phase 3, and
is explicitly gated on the out-of-sample result.

Design constraints:

- **Never raise.** Absent or malformed metadata must degrade to today's
  behaviour, which is "no hint". A checkpoint that ships nothing is
  normal, not an error.
- **Evidence with every prediction.** A bare parser name is unreviewable.
  Each prediction carries the substring that produced it, so a
  disagreement with the curated value can be judged rather than argued.
- **Ordered rules, first match wins.** The classes are not mutually
  exclusive in general -- a Qwen3 template contains `<tool_call>` and so
  does a Hermes one -- so ordering is part of the specification, not an
  implementation detail. See TOOL_RULES.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = [
    "load_chat_template",
    "predict_tool_format",
    "predict_reasoning_format",
    "FORMAT_TO_PARSER",
    "TOOL_FORMAT_TO_PARSER",
    "REASONING_FORMAT_TO_PARSER",
    "parser_for",
    "sampling_defaults",
    "derive_parser",
    "hints_for_model",
]


# ── Template loading ─────────────────────────────────────────────────────────

def load_chat_template(model_dir: str | Path) -> tuple[str, str]:
    """Return ``(template_text, source)`` for a checkpoint directory.

    Prefers the standalone `chat_template.jinja`; falls back to
    `tokenizer_config.json["chat_template"]`, which may itself be a plain
    string OR the list-of-named-templates shape
    ``[{"name": "default", "template": "..."}]``. Returns ``("", "")``
    when nothing is available -- e.g. NVIDIA-Nemotron-Nano-9B-v2-NVFP4
    ships no .jinja at all and only the embedded form.
    """
    d = Path(model_dir)

    jinja = d / "chat_template.jinja"
    try:
        if jinja.is_file():
            text = jinja.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                return text, "chat_template.jinja"
    except OSError:
        pass

    tok = d / "tokenizer_config.json"
    try:
        data = json.loads(tok.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError, ValueError):
        return "", ""

    raw = data.get("chat_template")
    if isinstance(raw, str) and raw.strip():
        return raw, "tokenizer_config.json"
    if isinstance(raw, list):
        # Named-template shape. Prefer "default"; otherwise take the first
        # entry that carries a template body.
        chosen = None
        for item in raw:
            if not isinstance(item, dict):
                continue
            body = item.get("template")
            if not isinstance(body, str) or not body.strip():
                continue
            if item.get("name") == "default":
                chosen = body
                break
            if chosen is None:
                chosen = body
        if chosen:
            return chosen, "tokenizer_config.json[chat_template]"
    return "", ""


# ── Tool-call format discrimination ──────────────────────────────────────────

# (format_class, required_markers, forbidden_markers)
#
# Order is load-bearing and each rule is validated against an on-disk
# checkpoint:
#
#   gemma4        Gemma-4-26B-A4B-it-NVFP4    inverted `<|tool_call>` delimiters
#   harmony       gpt-oss-20b                 `<|channel|>commentary`
#   nemotron_json NVIDIA-Nemotron-Nano-9B-v2  `<TOOLCALL>`
#   qwen3_xml_family                          `<tool_call>` + `<function=`
#                 -- AMBIGUOUS, see below
#   hermes        (Qwen3 base family)         `<tool_call>` WITHOUT `<function=`
#
# gemma4 must precede everything because its delimiters are a superstring
# trap: `<|tool_call>` contains `tool_call`. hermes must come LAST because
# it is the weakest rule -- `<tool_call>` alone -- and would otherwise
# swallow qwen3_xml, whose templates contain exactly that token four times.
TOOL_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("gemma4", ("<|tool_call>",), ()),
    ("harmony", ("<|channel|>commentary",), ()),
    ("harmony", ("to=functions.",), ()),
    ("nemotron_json", ("<TOOLCALL>",), ()),
    ("qwen3_xml_family", ("<tool_call>", "<function="), ()),
    ("hermes", ("<tool_call>",), ("<function=",)),
)


def predict_tool_format(template: str) -> tuple[str | None, str]:
    """Classify the tool-call markup. Returns ``(format_class, evidence)``.

    ``(None, "")`` means no rule matched, which is a real answer: plenty of
    checkpoints ship no tool markup, and inventing a parser for one is how
    a launch starts failing with "auto tool choice requires ...".
    """
    if not template:
        return None, ""
    for fmt, required, forbidden in TOOL_RULES:
        if all(m in template for m in required) and not any(
                m in template for m in forbidden):
            ev = " + ".join(required)
            if forbidden:
                ev += " (without " + ", ".join(forbidden) + ")"
            return fmt, ev
    return None, ""


# ── Reasoning format discrimination ──────────────────────────────────────────

REASONING_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("harmony", ("<|channel|>analysis",)),
    ("harmony", ("<|channel|>commentary",)),
    # Gemma-4's own pipe-delimited think token, distinct from the bare
    # `<think>` the Qwen-family templates use.
    ("gemma4_think", ("<|think|>",)),
    ("think_delimited", ("<think>",)),
    ("enable_thinking_only", ("enable_thinking",)),
)


def predict_reasoning_format(template: str) -> tuple[str | None, str]:
    """Classify the reasoning markup. Returns ``(format_class, evidence)``."""
    if not template:
        return None, ""
    for fmt, required in REASONING_RULES:
        if all(m in template for m in required):
            return fmt, " + ".join(required)
    return None, ""


# ── Format -> backend parser name ────────────────────────────────────────────

# (format_class, backend) -> parser flag value, or None where the engine
# has no equivalent. Static because these are CLI vocabularies, not
# derived facts: SGLang has no analogue of vLLM's nemotron_json,
# deepseek_string, qwen3_xml, gemma4 or openai (verified against
# v0.5.10.post1-cu130), so predicting one would produce a launch that
# argparse rejects.
#
# TOOL and REASONING need SEPARATE tables even though they share format
# classes. One table was the first implementation and the report caught it
# immediately: harmony's vLLM TOOL parser is `openai` but its vLLM
# REASONING parser is `openai_gptoss`. A single map predicted `openai` for
# both and disagreed with the probe-verified value.
TOOL_FORMAT_TO_PARSER: dict[tuple[str, str], str | None] = {
    ("gemma4", "vllm"): "gemma4",
    ("gemma4", "sglang"): None,
    ("harmony", "vllm"): "openai",
    ("harmony", "sglang"): "gpt-oss",
    ("nemotron_json", "vllm"): "nemotron_json",
    ("nemotron_json", "sglang"): None,
    # AMBIGUOUS on purpose -- this is the main negative result of the
    # out-of-sample validation. Four checkpoints carry identical
    # `<tool_call>` + `<function=` + `<parameter=` markup and split evenly
    # between two DIFFERENT probe-verified parsers:
    #
    #   Qwen3.5-9B-NVFP4                      -> qwen3_xml
    #   Ornith-1.0-9B-NVFP4                   -> qwen3_xml
    #   Qwen3-Coder-30B-A3B-Instruct-FP4      -> qwen3_coder
    #   NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4  -> qwen3_coder
    #
    # The token SETS are identical; only counts differ, which no template
    # refactor would preserve. `<think>` does not separate them either
    # (coder=0 but nemotron-3-nano=12). So predicting either name is a coin
    # flip, and a WRONG tool parser is worse than none: no parser means the
    # router strips tools and chat still works, while a wrong one
    # mis-parses live tool calls. This class therefore derives nothing and
    # the distinction stays curated -- which is the documented outcome
    # Phase 1's exit criteria allow for `qwen3_coder`.
    ("qwen3_xml_family", "vllm"): None,
    ("qwen3_xml_family", "sglang"): None,
    ("hermes", "vllm"): "hermes",
    ("hermes", "sglang"): "qwen",
}

REASONING_FORMAT_TO_PARSER: dict[tuple[str, str], str | None] = {
    ("harmony", "vllm"): "openai_gptoss",
    ("harmony", "sglang"): "gpt-oss",
    # AMBIGUOUS, on the same principle as qwen3_xml_family and settled by
    # the Phase 2 measurement. `Gemma-4-26B-A4B-it-NVFP4` and
    # `diffusiongemma-26B-A4B-it-NVFP4` have IDENTICAL reasoning markup --
    # one `<|think|>`, two `<channel|>`, three `<|channel>`, three
    # `strip_thinking` each -- yet the gemma4 family curates NO reasoning
    # parser and diffusiongemma curates `gemma4`.
    #
    # Measured on Gemma-4-26B-A4B-it-NVFP4 (2026-07-28, one vLLM launch):
    # `enable_thinking=true` genuinely changes behaviour (1250 chars of
    # deliberative output vs 393 without) but emits NO `<|think|>` and no
    # channel delimiters, and leaves `reasoning_content` empty. `<|think|>`
    # is not even a special token in the checkpoint -- it is absent from
    # added_tokens_decoder, so the template writes it as plain PROMPT text
    # to instruct the model, and the model answers in prose beginning
    # "thought". There is nothing delimited for a reasoning parser to
    # extract.
    #
    # So the markup cannot distinguish the two families, and the one
    # family measured emits nothing parseable. Deriving a reasoning parser
    # here would be a guess dressed as evidence.
    ("gemma4_think", "vllm"): None,
    ("gemma4_think", "sglang"): None,
    # `think_delimited` deliberately resolves to None. A bare `<think>`
    # block is emitted by qwen3, deepseek-r1 and nemotron templates alike,
    # and their vLLM reasoning parsers are qwen3, deepseek_r1 and
    # nemotron_v3 respectively -- the markup does not distinguish them.
    # Guessing here would silently mis-parse a reasoning stream, so this
    # stays curated. This is the single biggest limit on what the
    # derivation can do, and it is deliberate.
    ("think_delimited", "vllm"): None,
    ("think_delimited", "sglang"): None,
    ("enable_thinking_only", "vllm"): None,
    ("enable_thinking_only", "sglang"): None,
}

# Back-compat alias for readers that want one view of both.
FORMAT_TO_PARSER = {**REASONING_FORMAT_TO_PARSER, **TOOL_FORMAT_TO_PARSER}


def parser_for(format_class: str | None, backend: str,
               kind: str = "tool") -> str | None:
    """Backend parser name for a format class, or None when unmapped.

    `kind` is "tool" or "reasoning"; they are NOT interchangeable (see the
    tables above).
    """
    if not format_class:
        return None
    table = (TOOL_FORMAT_TO_PARSER if kind == "tool"
             else REASONING_FORMAT_TO_PARSER)
    return table.get((format_class, backend))


# ── Sampling defaults (used by Phase 4) ──────────────────────────────────────

_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "eos_token_id")


def sampling_defaults(model_dir: str | Path) -> dict:
    """Sampling fields from `generation_config.json`, or {} when absent.

    Only the keys the bench harness can act on are returned; everything
    else in that file is the engine's business.
    """
    path = Path(model_dir) / "generation_config.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: data[k] for k in _SAMPLING_KEYS if k in data}


def derive_parser(model_name: str, models_dir: str | Path,
                  backend: str, kind: str) -> str | None:
    """Parser name derived from a checkpoint's own template, or None.

    The single entry point the prober uses. `kind` is "tool" or
    "reasoning". Returns None whenever the evidence does not support a
    confident answer -- absent metadata, unrecognised markup, or a format
    class the validation showed to be ambiguous. None is the safe value:
    it reproduces today's behaviour exactly (no parser flag emitted).
    """
    model_dir = Path(models_dir) / model_name
    template, _ = load_chat_template(model_dir)
    if not template:
        return None
    if kind == "tool":
        fmt, _ = predict_tool_format(template)
    else:
        fmt, _ = predict_reasoning_format(template)
    return parser_for(fmt, backend, kind)


def hints_for_model(model_dir: str | Path) -> dict:
    """Everything this module can say about one checkpoint, in one call."""
    template, source = load_chat_template(model_dir)
    tool_fmt, tool_ev = predict_tool_format(template)
    reas_fmt, reas_ev = predict_reasoning_format(template)
    return {
        "template_source": source,
        "template_bytes": len(template),
        "tool_format": tool_fmt,
        "tool_evidence": tool_ev,
        "reasoning_format": reas_fmt,
        "reasoning_evidence": reas_ev,
        "tool_parser": {
            "vllm": parser_for(tool_fmt, "vllm", "tool"),
            "sglang": parser_for(tool_fmt, "sglang", "tool"),
        },
        "reasoning_parser": {
            "vllm": parser_for(reas_fmt, "vllm", "reasoning"),
            "sglang": parser_for(reas_fmt, "sglang", "reasoning"),
        },
        "sampling": sampling_defaults(model_dir),
    }
