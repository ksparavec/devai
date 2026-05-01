"""vLLM tool-parser plugin: DeepSeek-V3 chat-template format, string-only.

Why this exists
---------------
vLLM ships `deepseek_v3`, `deepseek_v31`, `deepseek_v32` tool parsers.
All three look up the boundary tokens in the model's vocabulary at
`__init__`:

    self.tool_calls_start_token_id = self.vocab.get(self.tool_calls_start_token)
    if self.tool_calls_start_token_id is None or ...:
        raise RuntimeError("DeepSeek-V3 Tool parser could not locate "
                           "tool call start/end tokens in the tokenizer!")

That works for the original DeepSeek-V3 / R1 weights because their
custom tokenizer has `<｜tool▁call▁begin｜>` and friends as atomic
vocabulary entries.

It DOES NOT work for the official R1 distills onto Qwen / Llama:

  - `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`   (Qwen2 tokenizer)
  - `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`  (Llama-3 tokenizer)

These distills inherit the BASE model's tokenizer but copy the
DeepSeek-V3 chat template, which still emits `<｜tool▁call▁begin｜>`
etc. literally in the output stream. The base tokenizer encodes those
markers as multi-token sequences (each full-width character a token),
so `vocab.get(...)` returns None and the parser refuses to start —
HTTP 500 on every tool-using request.

This plugin re-implements the same parser using *only string
operations* (no vocab-id lookups), so it works on any tokenizer that
encodes the markers consistently — atomic or multi-token. The
extraction regex and output shape match `deepseek_v3` exactly so
existing R1-trained tool-call sequences round-trip unchanged.

Wiring (see docs/backends.md once landed):

    --tool-parser-plugin /plugins/deepseek_string_tool_parser.py
    --tool-call-parser   deepseek_string

The plugin file must be readable inside the vllm container; bind-mount
the `scripts/vllm_plugins/` directory at `/plugins/` (or any path; the
flag takes the absolute path inside the container).

Limitations vs. the upstream parser
-----------------------------------
- Streaming uses string substring counts in `current_text` rather
  than `current_token_ids.count(...)`. For the chat-completion shape
  this is equivalent: the markers always appear as full literal
  strings in the decoded text. For weird tokenization splits it
  could in theory miss a count, but the markers are 6+ characters of
  full-width punctuation that no tokenizer will fragment in a way
  that hides them from a substring match.
- No structured-output / json-schema enforcement (`adjust_request`
  inherits the base implementation, which is fine).
- Tested against `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` only. The
  Llama-8B distill should work identically since the chat template
  is the same.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

logger = init_logger(__name__)


@ToolParserManager.register_module(["deepseek_string"])
class DeepSeekStringToolParser(ToolParser):
    """Drop-in replacement for `deepseek_v3` that doesn't require the
    boundary markers to be atomic tokens in the tokenizer's vocabulary.
    """

    # Matches the upstream `deepseek_v3` parser's literal markers so the
    # output of the same chat template parses identically.
    tool_calls_start_token = "<｜tool▁calls▁begin｜>"
    tool_calls_end_token = "<｜tool▁calls▁end｜>"
    tool_call_start_token = "<｜tool▁call▁begin｜>"
    tool_call_end_token = "<｜tool▁call▁end｜>"

    _tool_call_regex = re.compile(
        r"<｜tool▁call▁begin｜>(?P<type>[^<]*)<｜tool▁sep｜>(?P<name>[^\n]*)\n"
        r"```json\n(?P<args>.*?)\n```<｜tool▁call▁end｜>",
        re.DOTALL,
    )

    def __init__(self, tokenizer, tools=None):
        super().__init__(tokenizer, tools)
        # IMPORTANT: no vocab.get() lookup. The whole point of this
        # plugin is to skip that check.
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []

    def extract_tool_calls(
        self,
        model_output: str,
        request: "ChatCompletionRequest",
    ) -> ExtractedToolCallInformation:
        # Fast path: no boundary marker → plain text response.
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            tool_calls: list[ToolCall] = []
            for m in self._tool_call_regex.finditer(model_output):
                tool_type = (m.group("type") or "function").strip()
                name = (m.group("name") or "").strip()
                args = (m.group("args") or "").strip()
                # Validate JSON early; if the model emitted a malformed
                # block, prefer leaving it in `content` over forwarding
                # garbage to the agent.
                try:
                    json.loads(args)
                except json.JSONDecodeError:
                    logger.warning(
                        "deepseek_string: malformed JSON in tool call "
                        "for %s; skipping. args=%r", name, args[:200],
                    )
                    continue
                tool_calls.append(
                    ToolCall(
                        type=tool_type,
                        function=FunctionCall(name=name, arguments=args),
                    )
                )

            # `content` is everything before the first opening marker.
            # Empty string → emit None per OpenAI shape.
            content = model_output[
                : model_output.find(self.tool_calls_start_token)
            ]
            return ExtractedToolCallInformation(
                tools_called=bool(tool_calls),
                tool_calls=tool_calls,
                content=content if content else None,
            )
        except Exception:
            logger.exception("deepseek_string: extract_tool_calls failed")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: "ChatCompletionRequest",
    ) -> DeltaMessage | None:
        """Streaming variant. Uses substring counts in `current_text`
        instead of token-id counts in `current_token_ids` because the
        boundary markers are not atomic tokens in our target tokenizers.
        """
        # No tool calls yet — pass delta through as plain content.
        if self.tool_calls_start_token not in current_text:
            return DeltaMessage(content=delta_text)

        # Strip top-level wrappers from any text we forward — they're
        # parser markers, not content.
        sanitized_delta = delta_text.replace(
            self.tool_calls_start_token, ""
        ).replace(self.tool_calls_end_token, "")

        try:
            prev_start_count = previous_text.count(self.tool_call_start_token)
            prev_end_count = previous_text.count(self.tool_call_end_token)
            cur_start_count = current_text.count(self.tool_call_start_token)
            cur_end_count = current_text.count(self.tool_call_end_token)

            # Generating text content (not inside an open tool call).
            if (
                cur_start_count == cur_end_count
                and prev_end_count == cur_end_count
                and self.tool_call_end_token not in delta_text
            ):
                return DeltaMessage(content=sanitized_delta)

            # A new tool call just opened in this delta.
            if (
                cur_start_count > cur_end_count
                and cur_start_count > prev_start_count
            ):
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")

            # When we see the close marker, run the non-streaming
            # extractor over the just-completed call and emit a
            # finished DeltaToolCall. This is simpler than tracking
            # function-name and JSON-arg deltas mid-stream and matches
            # what most agent loops expect for short tool calls.
            if self.tool_call_end_token in delta_text:
                full_call_text = current_text + delta_text
                # Pull the most recent complete `<begin>...<end>` slice.
                m = list(
                    self._tool_call_regex.finditer(full_call_text)
                )
                if not m:
                    return DeltaMessage(content=None)
                last = m[-1]
                tool_type = (last.group("type") or "function").strip()
                name = (last.group("name") or "").strip()
                args = (last.group("args") or "").strip()
                try:
                    json.loads(args)
                except json.JSONDecodeError:
                    logger.warning(
                        "deepseek_string: malformed streaming tool "
                        "call for %s; dropping.", name,
                    )
                    return DeltaMessage(content=None)
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            type=tool_type,
                            function=DeltaFunctionCall(
                                name=name, arguments=args
                            ),
                        )
                    ]
                )

            # Mid-call delta with no closing marker yet — buffer; the
            # closing-marker branch above will emit the full call.
            return DeltaMessage(content=None)
        except Exception:
            logger.exception(
                "deepseek_string: extract_tool_calls_streaming failed"
            )
            return DeltaMessage(content=delta_text)
