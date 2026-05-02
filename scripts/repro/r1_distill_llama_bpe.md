# Draft vLLM issue: byte-level BPE leak in DeepSeek-R1-Distill-Llama-8B with `--reasoning-parser deepseek_r1`

> Status: not yet filed. This is a local draft. When ready, paste into
> https://github.com/vllm-project/vllm/issues/new (Bug Report).
> Reproducer: `scripts/repro/r1_distill_llama_bpe.py` in this repo.
>
> Note on character notation: this document uses `\u0120` and `\u010A`
> to refer to the leaked codepoints rather than pasting the glyphs
> directly. They are the GPT-2 / Llama-3 byte-level BPE markers for
> a leading space and a newline respectively.

## Summary

When `vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B` is launched
with `--reasoning-parser deepseek_r1`, the streamed `delta.content`
returned to OpenAI-compatible clients contains literal byte-level BPE
markers `\u0120` (Llama-3 tokenizer encoding for a leading space) and
`\u010A` (encoding for newline) instead of decoded whitespace. The
same prompt on `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` -- same
vLLM version, same `--reasoning-parser deepseek_r1`, only the
tokenizer family differs -- returns clean decoded UTF-8 text. This
isolates the bug to the deepseek_r1 reasoning parser interacting with
the Llama-3 tokenizer, not to the parser alone or the Llama-3
tokenizer alone.

## Environment

- vLLM image: `vllm/vllm-openai:latest-cu130-ubuntu2404`
  (digest: `<fill in: docker inspect | jq -r .[].Id>`)
- GPU: NVIDIA RTX 4000 PRO Blackwell (CUDA 13.0)
- Affected model: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
  (BF16 weights, Llama-3 tokenizer, sha `6a6f4aa41979`)
- Control model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
  (BF16 weights, Qwen-2 tokenizer, sha `916b56a44061`)
- Launch flags relevant to the bug:
  `--reasoning-parser deepseek_r1`
  (full launch line provided by a reverse proxy that recreates the
  container per-request; the proxy is not in the loop -- bug is
  reproducible against vLLM directly with the same flag)

## Steps to reproduce

1. Launch vLLM with the affected model and the deepseek_r1 reasoning
   parser:

   ```bash
   vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
     --reasoning-parser deepseek_r1
   ```

2. Send a streamed chat completion. Minimal Python (uses only the
   stdlib):

   ```bash
   python3 scripts/repro/r1_distill_llama_bpe.py \
     --base-url http://localhost:8000/v1 \
     --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
     --compare-with deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
   ```

   The reproducer hits `/v1/chat/completions` with `stream=true` and a
   trivial prompt ("Write a single Python function `add(a, b)` that
   returns the sum."), accumulates `delta.content` chunks across SSE
   events, and counts occurrences of `\u0120` and `\u010A`.

## Expected vs. actual

Expected: `delta.content` contains decoded UTF-8 text:

    "\n\ndef add(a, b):\n    return a + b"

Actual on `R1-Distill-Llama-8B` (showing each leaked codepoint in
Python-escape form):

    "\u010A\u010Adef\u0120add(a,\u0120b):\u010A\u0120\u0120\u0120\u0120return\u0120a\u0120+\u0120b"

Actual on `R1-Distill-Qwen-7B` (control, same parser, different
tokenizer family):

    "\n\n```python\ndef add(a, b): return a + b\n```"

Reproducer output captured in this repo on 2026-05-02:

```
=== AFFECTED: DeepSeek-R1-Distill-Llama-8B ===
  content len:                  33
  reasoning_content len:        0
  \u0120 (leading-space) count: 9
  \u010A (newline)       count: 3
  bug present:                  True

=== CONTROL:  DeepSeek-R1-Distill-Qwen-7B ===
  content len:                  43
  reasoning_content len:        0
  \u0120 in content:            0
  \u010A in content:            0
  bug present:                  False
```

The same shape was observed across all 50 samples of a HumanEval run
(11,688 occurrences of `\u0120` and 1,071 of `\u010A` across the run).

## Why we think this is in vLLM, not the model or client

- The Qwen-7B distill (which uses a Qwen-2 tokenizer family) does
  **not** show the leak under the **same** `--reasoning-parser
  deepseek_r1` flag. The parser code path is therefore being exercised
  in both runs, but only one model leaks.
- The leaked characters are exactly the GPT-2 / Llama-3 byte-level
  BPE markers (`\u0120` for leading space, `\u010A` for newline),
  which are an internal tokenizer encoding -- they should never appear
  in decoded UTF-8 returned to a client.
- We have not seen this with the Llama-3.1-8B-Instruct base (no
  reasoning parser), only with the R1-Distill flavour using
  `--reasoning-parser deepseek_r1`. That suggests the parser is
  extracting text at the token-id layer for the part of the stream
  it classifies as `content`, while the Qwen tokenizer happens to
  not surface these specific marker codepoints.

## Suggested investigation starting points

- `vllm/reasoning/deepseek_r1_reasoning_parser.py` (or whatever the
  current registry path is in vLLM HEAD) -- check whether the parser
  re-tokenizes / decodes the post-`</think>` segment with the model's
  tokenizer or assembles text at the token-id boundary.
- `vllm/transformers_utils/tokenizers/` -- whether the Llama-3
  fast-tokenizer's `convert_tokens_to_string` is being bypassed for
  the post-reasoning segment.

Happy to test patches.

## Workaround in our deployment

Until upstream fix:

- Route Llama-3 reasoning workloads to the non-reasoning
  `Llama-3.1-8B-Instruct-NVFP4` (no `--reasoning-parser`).
- Use `R1-Distill-Qwen-7B` instead of `R1-Distill-Llama-8B` for any
  workflow that depends on clean decoded `content`.
- Document the affected model in our routing tier as "avoid".

## Attachments

- `r1_distill_llama_bpe.py` -- minimal reproducer (this repo).
- Eval log with 50 affected samples available on request.
