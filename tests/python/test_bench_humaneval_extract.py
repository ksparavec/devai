"""HumanEval completion extraction keeps the code executable.

The v2 fence regex (``` ```(?:python|py)?\\s*\\n?(.*?)``` ```) let ``\\s*``
consume the newline after the info-string AND the first body line's
indentation. A model that answered with a fenced function BODY -- which
HumanEval's prompt invites, since the completion is appended to the
signature and docstring -- then had its first line dedented to column 0
and failed with a syntax error, scored as a wrong answer. The 2026-09-27
re-analysis counted 30 candidate false failures (docs/bench-results.md).
``str.strip()`` on the whole completion did the same to an UNFENCED body.

Stdlib unittest only; inspect_ai is faked (the module imports it at top
level), and the scorer's subprocess runs real Python on tiny programs.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _passthrough_decorator(*args, **kwargs):
    if args and callable(args[0]) and not kwargs:
        return args[0]
    return lambda fn: fn


def _fake_inspect_modules() -> dict[str, types.ModuleType]:
    names = {
        "inspect_ai": ("Task", "task"),
        "inspect_ai.dataset": ("Sample", "hf_dataset"),
        "inspect_ai.scorer": ("Score", "Target", "accuracy", "scorer", "stderr"),
        "inspect_ai.solver": ("TaskState", "generate", "system_message"),
    }
    mods = {}
    for mod, attrs in names.items():
        m = types.ModuleType(mod)
        for a in attrs:
            setattr(m, a, _passthrough_decorator if a in ("task", "scorer") else object)
        mods[mod] = m
    return mods


with mock.patch.dict(sys.modules, _fake_inspect_modules()):
    sys.modules.pop("bench.tasks.humaneval", None)
    humaneval = importlib.import_module("bench.tasks.humaneval")

PROMPT = 'def add(a, b):\n    """Return a + b."""\n'
TEST = "def check(f):\n    assert f(2, 3) == 5\n    assert f(-1, 1) == 0\n"


def _passes(completion_text: str) -> bool:
    code = humaneval._clean_completion(completion_text, entry_point="add")
    program = PROMPT + "\n" + code + "\n\n" + TEST + "\n\ncheck(add)\n"
    passed, _ = humaneval._run_check_in_subprocess(program)
    return passed


class FencedBodyTest(unittest.TestCase):
    def test_fenced_body_keeps_its_first_indent(self) -> None:
        text = "```python\n    return a + b\n```"
        self.assertEqual(humaneval._clean_completion(text, "add"), "    return a + b")
        self.assertTrue(_passes(text))

    def test_multi_line_fenced_body_runs(self) -> None:
        self.assertTrue(_passes("```python\n    s = a\n    s += b\n    return s\n```"))

    def test_fenced_full_definition_runs(self) -> None:
        self.assertTrue(_passes("```python\ndef add(a, b):\n    return a + b\n```"))

    def test_python3_and_capitalised_info_strings(self) -> None:
        for info in ("python3", "Python", "py", "", "python  "):
            text = f"```{info}\n    return a + b\n```"
            self.assertEqual(humaneval._clean_completion(text, "add"),
                             "    return a + b", info)

    def test_crlf_line_endings(self) -> None:
        self.assertTrue(_passes("```python\r\n    return a + b\r\n```"))

    def test_last_fence_wins(self) -> None:
        text = "draft:\n```python\n    return a - b\n```\nfixed:\n```python\n    return a + b\n```"
        self.assertTrue(_passes(text))

    def test_think_block_then_fence(self) -> None:
        self.assertTrue(_passes("<think>add them</think>\n```python\n    return a + b\n```"))

    def test_unfenced_full_definition_runs(self) -> None:
        self.assertTrue(_passes("def add(a, b):\n    return a + b"))

    def test_unfenced_body_keeps_its_first_indent(self) -> None:
        # str.strip() used to dedent it, the same defect as the fence regex.
        self.assertTrue(_passes("    return a + b"))
        self.assertTrue(_passes("\n\n    return a + b\n\n"))
        self.assertTrue(_passes("<think>x</think>\n    return a + b"))

    def test_a_wrong_answer_still_fails(self) -> None:
        self.assertFalse(_passes("```python\n    return a - b\n```"))


class SharedWithHumanEvalPlusTest(unittest.TestCase):
    def test_humaneval_plus_uses_the_same_cleaner(self) -> None:
        src = (REPO_ROOT / "scripts" / "bench" / "tasks" / "humaneval_plus.py").read_text()
        self.assertIn("from bench.tasks.humaneval import (", src)
        self.assertIn("humaneval_pass_at_1", src)  # whose scorer calls _clean_completion


if __name__ == "__main__":
    unittest.main()
