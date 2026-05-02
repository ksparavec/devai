"""Custom inspect_ai task for smart tool use.

Targets the failure modes observed in real Claude-Code testing on the
local stack:

  - ``empty_schema``: model invents arguments for a parameterless tool
    (Qwen3-8B-NVFP4 was caught fabricating ``task_id`` for ``TaskList``).
  - ``single_arg``: model fills the right field with the right value
    extracted from the prompt.
  - ``multi_tool_pick``: model picks the correct tool from a toolkit
    of three. Prompts are unambiguous; failures indicate weak tool
    routing.
  - ``result_followup``: model calls a tool, receives a result, then
    composes a final user-visible answer that incorporates it.
    Catches "model called tool, then ignored the result" failures.

Score is 1.0 / 0.0 per sample. The runner reads the per-sample
metadata from the eval log to compute ``by_subcase`` breakdowns.

Tools are deliberately stub-y — they return fixture data rather than
hitting any real service. The point is to test the *protocol* (does
the model call them right and use the result), not the tools'
domain logic.
"""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import (
    Generate,
    TaskState,
    solver,
    system_message,
    use_tools,
)
from inspect_ai.tool import ToolFunction, tool

PROMPTS_PATH = Path(__file__).resolve().parent.parent / "data" / "tools_prompts.jsonl"

SYSTEM_PROMPT = (
    "You have access to tools. Call the appropriate tool to answer "
    "the user's question. After receiving the tool result, summarize "
    "it in one short sentence. Use tools only when relevant; do not "
    "fabricate arguments — pass an empty arguments object {} when a "
    "tool takes no parameters."
)


@tool
def task_list():
    async def execute() -> str:
        """List active tasks. Takes no arguments — pass {}."""
        # Fixture: deterministic data the scorer can reason about.
        items = [
            {"id": 1, "subject": "ship the bench harness"},
            {"id": 2, "subject": "add VRAM sampling"},
        ]
        return json.dumps(items)

    return execute


@tool
def get_weather():
    async def execute(city: str) -> str:
        """Get the current weather in a city.

        Args:
            city: Name of the city to look up.
        """
        # Echo the city back so the result-followup scorer has a
        # deterministic substring to look for.
        return f"Sunny, 22°C in {city}."

    return execute


@tool
def get_time():
    async def execute(tz: str) -> str:
        """Get the current time in a timezone.

        Args:
            tz: IANA timezone name (e.g. ``UTC`` or ``America/New_York``).
        """
        return f"12:34:56 in {tz}."

    return execute


_TOOLS_BY_SUBCASE = {
    "empty_schema":     [task_list()],
    "single_arg":       [get_weather()],
    "multi_tool_pick":  [task_list(), get_weather(), get_time()],
    "result_followup":  [get_weather()],
}


def _load_prompts() -> list[dict]:
    out: list[dict] = []
    for line in PROMPTS_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _record_to_sample(item: dict) -> Sample:
    """One JSONL row → one inspect_ai Sample.

    The subcase, expected tool, expected args, and (optional) expected
    final-content substring all ride in metadata so the scorer is
    purely a function of (model output, sample). The dataset
    knows nothing about tools — the solver attaches them per-subcase
    by reading metadata.
    """
    return Sample(
        input=item["prompt"],
        target=item.get("expect_tool", ""),
        metadata={
            "id": item.get("id", ""),
            "subcase": item.get("subcase", ""),
            "expect_tool": item.get("expect_tool", ""),
            "expect_args": item.get("expect_args", {}),
            "expect_in_final": item.get("expect_in_final"),
        },
    )


def _extract_tool_calls(state: TaskState) -> list[dict]:
    """Collect all tool calls from assistant messages in the trace.

    Returns ``[{"name": str, "args": dict}, ...]``. inspect_ai's
    ``ToolCall`` is a typed dataclass: ``function`` is the name (str)
    and ``arguments`` is already a parsed dict. We only collect calls
    that parsed cleanly — ``parse_error`` non-empty means the model
    emitted malformed JSON-arguments and the scorer should treat this
    as a separate failure mode (``fail="parse_error"``).
    """
    out: list[dict] = []
    for msg in state.messages or []:
        if getattr(msg, "role", None) != "assistant":
            continue
        for call in (getattr(msg, "tool_calls", None) or []):
            name = getattr(call, "function", "")
            args = getattr(call, "arguments", None) or {}
            parse_error = getattr(call, "parse_error", None)
            if name:
                out.append({
                    "name": str(name),
                    "args": dict(args),
                    "parse_error": parse_error,
                })
    return out


def _final_assistant_text(state: TaskState) -> str:
    """Return the LAST assistant message's text content (the final
    user-visible answer after any tool_call/tool_result loop)."""
    for msg in reversed(state.messages or []):
        if getattr(msg, "role", None) != "assistant":
            continue
        text = getattr(msg, "text", None)
        if text:
            return str(text)
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return content
    return state.output.completion or ""


@scorer(metrics=[accuracy(), stderr()])
def tools_use_scorer():
    """Per-sample scorer with subcase-specific rules.

    All subcases require:
      - exactly one tool call in the trace, named ``expect_tool``;
      - the call's ``args`` strictly equal ``expect_args`` (no missing
        keys, no fabricated extras).

    ``result_followup`` adds: the final assistant text must contain
    ``expect_in_final`` (case-insensitive substring) so we know the
    model actually used the tool result rather than ignoring it.

    Score metadata captures the failure category so the runner can
    diagnose without reading the full eval log.
    """

    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata or {}
        subcase = meta.get("subcase", "")
        expect_tool = meta.get("expect_tool", "")
        expect_args = meta.get("expect_args", {}) or {}
        expect_in_final = meta.get("expect_in_final")

        calls = _extract_tool_calls(state)

        if len(calls) == 0:
            return Score(
                value=0.0, answer="(no tool call)",
                explanation=f"subcase={subcase}: model did not call any tool",
                metadata={"subcase": subcase, "fail": "no_call"},
            )
        if len(calls) > 1:
            return Score(
                value=0.0, answer=str([c["name"] for c in calls]),
                explanation=f"subcase={subcase}: expected 1 call, got {len(calls)}",
                metadata={"subcase": subcase, "fail": "multi_call"},
            )

        call = calls[0]
        if call.get("parse_error"):
            return Score(
                value=0.0, answer=str(call.get("args")),
                explanation=(
                    f"subcase={subcase}: tool-call arguments did not parse "
                    f"({call['parse_error']!r})"
                ),
                metadata={"subcase": subcase, "fail": "parse_error"},
            )
        if call["name"] != expect_tool:
            return Score(
                value=0.0, answer=call["name"],
                explanation=(
                    f"subcase={subcase}: expected tool {expect_tool!r}, "
                    f"got {call['name']!r}"
                ),
                metadata={"subcase": subcase, "fail": "wrong_tool"},
            )

        # Args must match exactly. Empty schema → must be {}; fabricated
        # extras (e.g. {"task_id": 1} for task_list) are the canonical
        # Qwen3-8B failure.
        if dict(call["args"]) != dict(expect_args):
            return Score(
                value=0.0,
                answer=json.dumps(call["args"], sort_keys=True),
                explanation=(
                    f"subcase={subcase}: args mismatch — "
                    f"expected {expect_args!r}, got {call['args']!r}"
                ),
                metadata={"subcase": subcase, "fail": "wrong_args"},
            )

        if subcase == "result_followup" and expect_in_final:
            final_text = _final_assistant_text(state)
            if str(expect_in_final).lower() not in final_text.lower():
                return Score(
                    value=0.0,
                    answer=final_text[:120],
                    explanation=(
                        f"subcase={subcase}: tool called correctly but "
                        f"final answer missing {expect_in_final!r}"
                    ),
                    metadata={"subcase": subcase, "fail": "no_followup"},
                )

        return Score(
            value=1.0,
            answer=call["name"],
            explanation=f"subcase={subcase}: ok",
            metadata={"subcase": subcase, "fail": None},
        )

    return score


@solver
def tool_loop_with_pin():
    """Custom tool-call loop that pins ``tool_choice`` per-sample.

    Replaces inspect_ai's default ``generate()`` solver chain for this
    task. Why we can't just set ``state.tool_choice`` and reuse
    ``generate()``:

    - The router's ``tool_choice_pinning_required`` rule (see
      ``gpu-arbiter/main.go::maybePromoteToolChoice``) returns HTTP 400
      when a vLLM/SGLang ``tool_mode="forced"`` model (R1-Distill x2,
      Llama-3.1-8B-Instruct-NVFP4) sees ``tool_choice="auto"`` against
      multiple tools.
    - inspect_ai's built-in tool loop (``inspect_ai/_eval/task/generate.py``)
      reads ``state.tool_choice`` once, but **resets it to ``"auto"``
      after a forced ``ToolFunction`` call** so the model can produce a
      final answer. That second turn then trips the router.

    The loop here:

    1. Turn 1 sends ``tool_choice = ToolFunction(name=expect_tool)``.
       Router accepts (a function-spec map is not ``"auto"``).
    2. Tools execute via ``execute_tools``.
    3. For ``result_followup`` only, a second model turn runs with
       ``tool_choice="none"`` so the model produces the final text. The
       router accepts ``"none"`` (it's a string that's neither empty
       nor ``"auto"``). Other subcases stop after turn 1; the scorer
       grades them on the tool call alone.

    Tradeoff: ``multi_tool_pick`` no longer tests the model's tool-
    routing decision (we hand it the answer). It now tests args
    correctness given the right tool — narrower, but the only path that
    works for forced-mode models on multi-tool prompts.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        from inspect_ai.model import get_model
        from inspect_ai.model._call_tools import execute_tools

        meta = state.metadata or {}
        expect_tool = str(meta.get("expect_tool") or "")
        subcase = str(meta.get("subcase") or "")

        model = get_model()
        first_choice = ToolFunction(name=expect_tool) if expect_tool else "auto"

        state.output = await model.generate(
            input=state.messages,
            tools=state.tools,
            tool_choice=first_choice,
        )
        state.messages.append(state.output.message)

        if state.output.message.tool_calls:
            tool_result = await execute_tools(state.messages, state.tools)
            state.messages.extend(tool_result.messages)
            if tool_result.output is not None:
                state.output = tool_result.output

            if subcase == "result_followup":
                state.output = await model.generate(
                    input=state.messages,
                    tools=state.tools,
                    tool_choice="none",
                )
                state.messages.append(state.output.message)

        return state

    return solve


def _build_solver(samples: list[Sample]):
    """Build a solver chain that exposes the union of all tools to every
    sample, then drives a per-sample tool-call loop via
    ``tool_loop_with_pin``. See that solver for why we don't use
    inspect_ai's stock ``generate()``.
    """
    all_tools = [task_list(), get_weather(), get_time()]
    return [
        system_message(SYSTEM_PROMPT),
        use_tools(all_tools),
        tool_loop_with_pin(),
    ]


@task
def tools_use_task(n: int = 20) -> Task:
    """Build the tools-use Task. ``n`` caps the prompt count; default
    20 = full set (5 per subcase × 4 subcases).
    """
    items = _load_prompts()[:n]
    samples = [_record_to_sample(item) for item in items]
    return Task(
        dataset=MemoryDataset(samples=samples, name="tools_use"),
        solver=_build_solver(samples),
        scorer=tools_use_scorer(),
    )
