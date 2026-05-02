#!/usr/bin/env python3
"""Minimal reproducer for the byte-level BPE leak in
DeepSeek-R1-Distill-Llama-8B on vLLM with ``--reasoning-parser deepseek_r1``.

Symptom: ``content`` chunks contain literal ``Ġ`` (U+0120, the GPT-2 /
Llama-3 byte-level BPE marker for a leading space) and ``Ċ`` (U+010A,
the marker for newline) instead of decoded whitespace. Sample output
looks like::

    ĊĊToĠsolveĠthisĠproblem,ĠweĠneedĠtoĠdetermine...

The model produces real Python at the token level; the decoded UTF-8
view returned to clients is wrong, so any downstream parser that
expects ``def foo(x):\\n    return x`` sees ``defĠfoo(x):Ċ    returnĠx``
and rejects it as syntax-invalid.

The bug does **not** fire on ``DeepSeek-R1-Distill-Qwen-7B`` (same
``--reasoning-parser deepseek_r1``, different tokenizer family). That
isolation is the strongest pointer at a parser × tokenizer interaction
in vLLM.

Usage (against this project's local stack)::

    podman run --rm --network devai-net --entrypoint python3 \\
        -v $PWD/scripts/repro:/repro:ro \\
        devai-lab-cpu /repro/r1_distill_llama_bpe.py \\
        --base-url http://devai-router:11435/v1 \\
        --model DeepSeek-R1-Distill-Llama-8B@32768

Usage (against any vLLM that has the same flags)::

    python3 r1_distill_llama_bpe.py \\
        --base-url http://your-vllm:8000/v1 \\
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

DEFAULT_PROMPT = (
    "Write a single Python function `add(a, b)` that returns the sum. "
    "Output only the function definition, no prose, no markdown fences."
)

# Byte-level BPE markers used by GPT-2 / Llama-3 tokenizers. Ġ encodes
# a leading space, Ċ encodes a newline. Both should be decoded to
# whitespace before reaching the client.
MARKER_SPACE = "Ġ"   # Ġ
MARKER_NEWLINE = "Ċ"  # Ċ


def stream_completion(base_url: str, model: str, prompt: str) -> tuple[str, str]:
    """Hit ``<base_url>/chat/completions`` with stream=true. Return a
    ``(content, reasoning_content)`` pair, each accumulated across SSE
    deltas. ``reasoning_content`` is what vLLM separates out under
    ``--reasoning-parser deepseek_r1``; ``content`` is the post-reasoning
    answer that should be clean, decoded text.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": 256,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy",
        },
    )
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in obj.get("choices") or []:
                delta = choice.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    content_parts.append(delta["content"])
                if isinstance(delta.get("reasoning_content"), str):
                    reasoning_parts.append(delta["reasoning_content"])
    return "".join(content_parts), "".join(reasoning_parts)


def report(label: str, content: str, reasoning: str) -> bool:
    """Print marker counts and a short raw-repr excerpt. Return True
    iff the bug is present (markers found in ``content``)."""
    n_space = content.count(MARKER_SPACE)
    n_newline = content.count(MARKER_NEWLINE)
    leaked = n_space > 0 or n_newline > 0
    print(f"=== {label} ===")
    print(f"  content len:           {len(content)}")
    print(f"  reasoning_content len: {len(reasoning)}")
    print(f"  Ġ (U+0120) in content: {n_space}")
    print(f"  Ċ (U+010A) in content: {n_newline}")
    print(f"  bug present:           {leaked}")
    print("  --- first 200 chars (raw repr) ---")
    print(f"  {content[:200]!r}")
    print()
    return leaked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--base-url",
        default="http://devai-router:11435/v1",
        help="OpenAI-compatible base URL (default: this project's vLLM router port)",
    )
    ap.add_argument(
        "--model",
        default="DeepSeek-R1-Distill-Llama-8B@32768",
        help="model name; bug-affected model",
    )
    ap.add_argument(
        "--compare-with",
        default="DeepSeek-R1-Distill-Qwen-7B@65536",
        help=(
            "second model to compare against (different tokenizer family). "
            "Pass empty string to skip."
        ),
    )
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = ap.parse_args()

    affected_content, affected_reasoning = stream_completion(
        args.base_url, args.model, args.prompt
    )
    affected_leaked = report(
        f"AFFECTED: {args.model}", affected_content, affected_reasoning
    )

    control_leaked = False
    if args.compare_with:
        control_content, control_reasoning = stream_completion(
            args.base_url, args.compare_with, args.prompt
        )
        control_leaked = report(
            f"CONTROL:  {args.compare_with}", control_content, control_reasoning
        )

    print("=== Summary ===")
    print(f"  affected leaked: {affected_leaked}")
    if args.compare_with:
        print(f"  control  leaked: {control_leaked}")
        if affected_leaked and not control_leaked:
            print(
                "  → reproduces the documented bug: "
                "Llama-3-tokenizer model leaks BPE markers, "
                "Qwen-tokenizer model with the same parser does not."
            )
            return 0
    if affected_leaked:
        return 0  # bug confirmed even without control
    return 2  # could not reproduce — no markers found


if __name__ == "__main__":
    sys.exit(main())
