# Reasoning, tool calling, and chat templates

This page covers the three things that make a modern LLM feel
"agentic" -- and the three things this project's router code spends
most of its complexity on. They are different mechanisms but they
interact tightly, which is why beginners conflate them.

If you understand
[`attention-and-the-transformer.md`](attention-and-the-transformer.md)
already, you know how the model emits one token at a time. This doc
explains:

1. **Chat templates** -- how a stream of role-tagged messages becomes
   a single token sequence the model can read, and how the model's
   raw token output gets parsed back into structured turns.
2. **Tool calling** -- how the model "invokes" an external function
   when it has no tools, no internet, and no side-channel -- using
   nothing but tokens.
3. **Reasoning models** -- what `<think>` tokens are, why
   `--reasoning-parser` exists, and why the project's bench harness
   had a TPS-counting bug for half a release cycle because nobody
   on the planet had cleanly explained the abstraction yet.

Each section maps directly to a real entry in
`scripts/model-families.yaml`, a real rewrite step in
[`router.md`](router.md), or a real bug fix in
[`bench-results.md`](bench-results.md).

---

## 1. Chat templates -- turning messages into tokens

A chat-tuned LLM is, mechanically, the same next-token predictor as
the base model
([`attention-and-the-transformer.md`](attention-and-the-transformer.md))
but trained on a corpus of *formatted multi-turn conversations*. The
formatting is the chat template: a deterministic recipe for turning a
list of `{role, content}` messages into one big string that gets
tokenised the usual way.

### 1.1 Why a template at all?

The model has no built-in concept of "user" vs "assistant". To it,
every input is an undifferentiated stream of token IDs. Telling the
model "this is what the user said, this is what you said, now
respond" requires *reserved control tokens* that mark turn
boundaries. The chat template inserts them.

### 1.2 The Qwen3 / ChatML format

Qwen3 (and many other modern models -- DeepSeek, Mistral, gpt-oss)
use a variant of the **ChatML** format, originally from OpenAI:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2 + 2?<|im_end|>
<|im_start|>assistant
4<|im_end|>
```

The visible markers are *control tokens* with reserved IDs in the
tokeniser's vocabulary. They are not multi-character strings the
model interprets via parsing -- they are *single tokens* whose role
the model learned during fine-tuning. For Qwen3:

- `<|im_start|>` -- token ID 151644 (single token)
- `<|im_end|>`   -- token ID 151645 (single token)
- `<|endoftext|>` -- token ID 151643 (single token, end of sequence)
- ... and dozens more for tool calls, reasoning, etc.

The roles `system`, `user`, `assistant` are *normal text tokens*
following the `<|im_start|>` marker. The newline after the role and
before the content is significant -- it is part of the trained format.

### 1.3 Where the template lives

Every HuggingFace model ships its template as a Jinja2 string in
`tokenizer_config.json` under the `chat_template` field. You can
inspect it directly:

```bash
jq -r '.chat_template' /var/cache/devai/.../tokenizer_config.json
```

For Qwen3 the template renders messages plus optional system prompt
plus a trailing `<|im_start|>assistant\n` to *prime the model to
respond*. When the user calls `/v1/chat/completions` with messages,
the inference engine (vLLM / SGLang / Ollama) runs this Jinja
template, then tokenises the resulting string.

### 1.4 The "generation prompt" and where you stop

A subtle but critical point. After rendering all the prior messages,
the template ends with:

```
<|im_start|>assistant
```

(no `<|im_end|>` yet -- the assistant turn is *open*). The model now
generates token after token until it produces `<|im_end|>` itself,
at which point the inference engine stops generating, strips the
template scaffolding, and returns the assistant's content to the
caller.

This is why **template token leakage is a real bug**: if the
inference engine fails to stop on `<|im_end|>` (wrong tokeniser, model
emits the *string* "<|im_end|>" instead of the *token*), the user
sees `<|im_end|>` in their output. The bench harness explicitly
probes for this -- see
[`bench-results.md`](bench-results.md)'s issue #3 for a real case
(`Nemotron-Nano-9B-v2-NVFP4` leaked `</think>` 3 times in 40 prompts
because vLLM was launched without `--reasoning-parser`).

### 1.5 Tokeniser != model -- keep them paired

A model trained with the Qwen3 tokeniser **must** be served with the
Qwen3 tokeniser. Cross-tokeniser inference produces nonsense -- the
model expects token ID 151644 to mean `<|im_start|>` because that is
what it learned, but a Llama tokeniser will emit different IDs for
the same string. This is why every checkpoint ships its full
tokeniser bundle (`tokenizer.json`, `tokenizer_config.json`,
`merges.txt`, `vocab.json` for byte-pair-encoding tokenisers) and
why mixing pieces from different families is forbidden.

---

## 2. Tool calling -- making the model invoke functions

The model has no internet, no shell, no databases, and no eyes. It
emits tokens. "Tool calling" is the convention by which a chat-tuned
LLM emits tokens that an external runtime parses into a structured
function invocation, executes, and feeds the result back in another
turn. The whole loop is *token in, token out* on the model's side.

### 2.1 What a tool call actually looks like over the wire

The OpenAI-compatible tool-call protocol (everyone copies this now -- 
see [`openai-api-and-streaming.md`](openai-api-and-streaming.md))
expects the *caller* to declare available tools via JSON Schema:

```json
"tools": [
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather for a city",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {"type": "string"}
        },
        "required": ["city"]
      }
    }
  }
]
```

The inference engine renders these into the chat template (each
model has its own way -- see Sec. 2.3). The model is fine-tuned to
*recognise* this schema and *emit* a structured response when it
decides to call a tool. The response comes back to the client as:

```json
"choices": [{
  "message": {
    "role": "assistant",
    "content": null,
    "tool_calls": [{
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"city\": \"Paris\"}"
      }
    }]
  },
  "finish_reason": "tool_calls"
}]
```

The client (an agent like Claude Code or Aider) executes the actual
function, then sends the result back as a new message:

```json
{"role": "tool", "tool_call_id": "call_abc123", "content": "{\"temp\":18,\"unit\":\"C\"}"}
```

The model takes another turn, sees the tool result in context, and
continues -- possibly with another tool call, possibly with a final
text answer.

### 2.2 But how does the model "emit" a JSON object?

The model emits **tokens**. The tool-call object you see in the API
response is the result of *parsing* those tokens after the model is
done generating.

Different model families learned to emit tool calls in different
formats during fine-tuning. There is no universal convention. Common
patterns:

- **Hermes / OpenHermes** (used by some Qwen3 variants):

  ```
  <tool_call>
  {"name": "get_weather", "arguments": {"city": "Paris"}}
  </tool_call>
  ```

- **Qwen3 XML** (Qwen3 family default):

  ```
  <tool_call>
  <function=get_weather>
  <parameter=city>Paris</parameter>
  </function>
  </tool_call>
  ```

- **Llama 3 JSON**:

  ```
  <|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}<|eom_id|>
  ```

- **Harmony channels** (gpt-oss):

  ```
  <|channel|>commentary
  to=functions.get_weather code: { "city": "Paris" }
  <|return|>
  ```

- **DeepSeek string** (R1-Distill family):

  ```
  <|tool_calls_begin|><|tool_call_begin|>function<|tool_sep|>get_weather
  ```json
  {"city": "Paris"}
  ```<|tool_call_end|><|tool_calls_end|>
  ```

Each format uses a different combination of reserved control tokens
to mark "I am calling a tool now" and to delimit the function name
and arguments.

### 2.3 The role of `--tool-call-parser`

vLLM and SGLang ship pluggable parsers -- `hermes`, `qwen3_xml`,
`llama3`, `llama3_json`, `deepseek_string`, `openai` (harmony), etc.
 -- each of which knows how to:

1. Render the tool definitions into the model's expected format when
   building the prompt.
2. Detect the model's tool-call output in the streamed tokens.
3. Convert the model's format back into the OpenAI-compatible
   `tool_calls` JSON.

When you launch vLLM with
`--tool-call-parser qwen3_xml --enable-auto-tool-choice`, you are
saying *"this model uses Qwen3's XML format for tool calls; please
parse them and surface them as standard OpenAI `tool_calls` in the
response"*. Without the parser flag, vLLM either rejects the request
("auto tool choice requires --enable-auto-tool-choice and
--tool-call-parser", a real error you have probably hit) or returns
the raw XML in `content` and the agent gets confused.

### 2.4 How this project picks the right parser

`scripts/model-families.yaml` curates the *expected* parser per
model family per backend. Excerpt for the Qwen3 NVFP4 family:

```yaml
- family: qwen3
  ...
  parsers:
    vllm:
      reasoning: qwen3
      # qwen3_xml is the format Qwen3 chat templates actually emit,
      # not plain hermes. Probe confirmed `hermes` left tool_calls
      # empty on Qwen3.5-9B-NVFP4 -> switched to qwen3_xml.
      tool: qwen3_xml
    sglang:
      reasoning: qwen3
      # SGLang has no qwen3_xml parser -- `qwen` covers the same
      # format under a different name.
      tool: qwen
```

The probe drivers (`scripts/probe-vllm-reasoning.py`,
`probe-sglang-reasoning.py`) launch each model with these flags and
do a single round-trip with both a structured tool-call probe and a
plain text probe. If the parser actually works, the probe stores the
verified flag in `deploy/.vllm-reasoning-cache.json` /
`deploy/.sglang-reasoning-cache.json`. The router then reads the
*verified* values back when it recreates a container -- so a model
that fails its parser probe simply gets launched without the parser
flags, falling back to no-tools mode.

This indirection (curated hint -> probe -> cache -> router) means: when
a new model lands and the parser hint is wrong, the probe surfaces
it as a probe failure rather than a runtime crash. Read
[`backends.md`](backends.md) for the lifecycle.

### 2.5 `tool_choice` -- auto, none, required, forced

Three variants of "what should the model do with the tools I
declared":

- `tool_choice: "auto"` (default) -- the model decides whether to
  call a tool or just respond. Most flexible; works on every parser.
- `tool_choice: "none"` -- explicitly disable tools for this turn.
  Same as not passing tools at all.
- `tool_choice: "required"` -- force the model to call *some* tool,
  not just respond.
- `tool_choice: {"type": "function", "function": {"name": "X"}}` -- 
  **force** the model to call this specific function. The inference
  engine constrains the model's output during generation so it can
  only emit the tool-call format for `X`.

The "forced" variant is the trickiest. It is implemented either as
**guided generation** (the engine biases logits at each step so only
tokens consistent with the schema can be emitted), or as **prompt
injection** (the engine writes the tool-call opening tokens into the
prompt itself so the model "starts" the call and only has to fill
in arguments). Different parsers/engines handle this differently.

This is also where the project's `tool_choice_pinning_required`
router rule lives. Some agents send `tool_choice: "auto"` against a
model that the bench identified as "only reliable in forced mode".
The router refuses with HTTP 400 ("`tool_choice` must be pinned for
this model") rather than letting the agent get garbage. That refusal
is what surfaced as bench issue #2 in
[`bench-results.md`](bench-results.md) -- exactly the router doing
its job, breaking a bench task that didn't know to pin.

---

## 3. Reasoning models -- thinking out loud, in tokens

A "reasoning" or "thinking" model is one that was fine-tuned to
**emit a long internal chain of thought before its final answer**.
The chain of thought is not magical or special -- it is just more
tokens, generated by the same next-token loop. The novelty is the
training objective rewarded the model for producing a high-quality
reasoning trace, and the *interface* that separates reasoning tokens
from final-answer tokens.

### 3.1 The `<think>` block convention

Reasoning models emit their chain of thought wrapped in markers:

```
<think>
The user asked what 2+2 is. Let me work this out.
2+2 = 4. That's a basic arithmetic fact. I should just say "4".
</think>4
```

The `<think>` and `</think>` tags are reserved control tokens (just
like `<|im_start|>` from Sec. 1) for some models, and plain text patterns
for others. Different families:

- **DeepSeek-R1 / R1-Distill**: literal `<think>...</think>` text
  with reserved tokens.
- **Qwen3** (with `--enable-thinking`): `<think>...</think>` text,
  parser pulls it out.
- **gpt-oss / harmony**: uses *channels* -- the model emits
  `<|channel|>analysis ... <|message|> ... <|return|>` for reasoning
  and `<|channel|>final ... <|message|> ... <|return|>` for the
  user-visible answer.
- **Nemotron-Nano-v2**: inline `<think>...</think>` without
  reserved tokens (string-pattern parsing only).

Inside the `<think>` block the model can be wrong, change its mind,
write 1000 tokens of self-doubt, derive the same wrong answer five
ways, etc. The training rewards arriving at the *right* answer
*after* the trace. The trace is "throwaway" reasoning: the user
should not see it; the agent should not act on it; only the
post-`</think>` content is the actual answer.

### 3.2 The `--reasoning-parser` flag

Just as `--tool-call-parser` separates tool calls from content, the
`--reasoning-parser` flag tells vLLM/SGLang to extract `<think>`
content and surface it under a separate `reasoning_content` field
in the API response, rather than mixing it into `content`:

```json
"choices": [{
  "message": {
    "role": "assistant",
    "content": "4",
    "reasoning_content": "The user asked what 2+2 is. Let me work this out.\n2+2 = 4..."
  }
}]
```

Streaming SSE works the same way -- `delta.reasoning_content` arrives
piecemeal as the `<think>` block streams, then `delta.content`
arrives once the model emits `</think>`.

`scripts/model-families.yaml` curates per-family reasoning parser
hints just like tool parser hints. Excerpt:

```yaml
- family: deepseek-r1
  parsers:
    vllm:
      reasoning: deepseek_r1
      tool: deepseek_string

- family: gpt-oss
  parsers:
    vllm:
      reasoning: openai_gptoss   # vLLM registers harmony parser under this name
      tool: openai

- family: nemotron
  parsers:
    vllm:
      reasoning: nemotron_v3
      tool: hermes
```

### 3.3 Capability classification -- `structured` / `inline` / `unsupported`

The probe driver classifies each model into one of three
**capability** buckets, stored in
`deploy/.vllm-reasoning-cache.json`:

- `structured` -- model + parser produces a clean
  `reasoning_content` separated from `content`. Best case; the
  router's "off" mode (`::nothink`) can suppress the trace by
  setting `enable_thinking=false`.
- `inline` -- model emits `<think>...</think>` as plain text without
  parser support, OR the parser exists but doesn't separate the two
  channels. Result: `content` contains both the trace and the
  answer. The router's "off" mode is meaningless here (no parser to
  toggle); the picker offers a separate `::nothink` row that suppresses
  thinking via prompt-side instruction.
- `unsupported` -- the model is not reasoning-capable. No `<think>`
  emission expected. The router's reasoning-policy logic is a no-op.

The picker UI uses these classifications to decide whether to show a
separate "Reasoning off" row for each model.

### 3.4 The TPS-counting bug -- a worked example of why this matters

This is the cleanest empirical illustration of why understanding
reasoning matters operationally.

**The bug**: the bench harness measured tokens-per-second by reading
`usage.completion_tokens` from the final SSE event and dividing by
`(t_done - t_first_token)`. For a 200-token answer over 2 seconds,
that gives 100 tok/s. Sensible.

**The catch**: vLLM with `--reasoning-parser qwen3` populates
`usage.completion_tokens` with **only the tokens in `content`** -- 
the *reasoning_content* tokens (which can be 1000+ for a thinking
model on a hard prompt) are excluded. So for a Qwen3 reasoning
trace of 1500 reasoning tokens + 50 content tokens streamed over
2 seconds:

- Wall-clock decode time: ~2.0 s
- Stream produced ~1550 tokens of actual decode work
- `usage.completion_tokens` reports only 50 (just the content)
- bench computes: 50 / 2.0 = **25 tok/s**
- True throughput: 1550 / 2.0 = **775 tok/s**

The first bench pass on Qwen3-14B-NVFP4 reported 0.66 tok/s,
underestimating the true rate by roughly **94x** for thinking-heavy
prompts where 99 % of the stream was reasoning. This was clearly
wrong on inspection -- the GPU was working hard, the user saw tokens
flowing rapidly -- but the metric said otherwise.

**The fix** (`scripts/bench/_bench_core.py::stream_chat_completion`):
accumulate `delta.reasoning_content` chars alongside
`delta.content`, then derive

```
   effective_tokens = max(usage.completion_tokens,
                          (content_chars + reasoning_chars) // 4)
```

The `// 4` is the standard ~4-chars-per-token rule of thumb (see
[`llm-tokens-and-speed.md`](llm-tokens-and-speed.md) Sec. 4). The `max()`
keeps accurate parsers from being penalised on short outputs where
the heuristic might underestimate.

The fix moved Qwen3-8B-NVFP4 from 14.53 tok/s to **98.3 tok/s**, the
number cited everywhere else in this project's docs. Validation: the
non-reasoning Llama-3.1-8B-Instruct-NVFP4 row moved from 95.53 to
95.88 tok/s -- a 0.4 % drift consistent with measurement noise,
proving the fix was a no-op for non-reasoning models.

### 3.5 Reasoning policies in the router

The router exposes three knobs for handling reasoning traces:

- **Global `DEVAI_REASONING` env**: `auto` (default), `off`, `low`,
  `medium`, `high`. Sets the per-request `reasoning_effort` field
  for engines that support it (vLLM with reasoning parser, harmony,
  etc.).
- **Per-request `::<reasoning>` suffix** in the model name (e.g.
  `Qwen3-8B-NVFP4::nothink@131072`): overrides the global policy
  for this one request only. Triggers a container recreate if the
  override changes (see
  [`nvfp4-coldstart.md`](nvfp4-coldstart.md) Sec. 1's "What triggers a
  full cold start").
- **Capability-gated rewrite**: the router only emits
  `enable_thinking=false` on models the probe classified as
  `structured` (and `disable_verified=true`). On `inline` models the
  rewrite is a no-op because the parser cannot separate the streams
  anyway.

This three-layer setup matches the picker's UX: users see one row
per (model, ctx, reasoning_override) combination, and switching rows
trips the right cold-start path.

---

## 4. How they all interact -- one full agent loop

Putting chat templates, tool calling, and reasoning together -- here
is what a single agent turn looks like end-to-end:

```
1. Agent (Claude Code) sends:
     POST /v1/chat/completions
     {model: "Qwen3-8B-NVFP4@131072",
      messages: [...history...],
      tools: [...declared tools...],
      tool_choice: "auto",
      stream: true}

2. Router parses model name -> model = Qwen3-8B-NVFP4, ctx = 131072.
   Checks current loaded model; recreates vLLM container if needed.
   Forwards to vLLM with --reasoning-parser qwen3
   --tool-call-parser qwen3_xml --max-model-len 131072.

3. vLLM applies Qwen3 chat template:
     <|im_start|>system\nYou are...\n<|im_end|>
     <|im_start|>user\n...message...\n<|im_end|>
     [each tool rendered as Qwen3 XML in the system message]
     <|im_start|>assistant\n
   then tokenises and runs inference.

4. Model generates:
     <think>The user wants the weather. I should call get_weather.</think>
     <tool_call>
     <function=get_weather><parameter=city>Paris</parameter></function>
     </tool_call>
     <|im_end|>

5. vLLM streams SSE events:
     a. delta.reasoning_content = "The user wants..." (parsed from <think>)
     b. delta.tool_calls[0].function.name = "get_weather"
     c. delta.tool_calls[0].function.arguments += "{\"city\":\"Paris\"}"
     d. finish_reason = "tool_calls"

6. Agent receives parsed tool call, executes get_weather("Paris"),
   sends result back as new message:
     {"role": "tool", "tool_call_id": "...", "content": "{\"temp\":18}"}

7. Loop: back to step 1 with extended history. Router does NOT
   recreate (same model + ctx + reasoning_override); vLLM extends
   the KV cache with the new turn and continues from the existing
   state.
```

Every numbered step has a concept covered above. The reason this
project has so much router code, so much YAML curation, and so much
probe machinery is that **every model implements every step
slightly differently**, and getting one detail wrong (parser name,
tool format, reasoning override semantics) means broken agents
silently producing garbage instead of clean errors.

---

## 5. Practical implications

- **Pick a model whose parsers your serving stack supports.** A
  reasoning model with no parser plugin will leak `<think>` tokens
  into `content`. A tool model with no parser plugin will not call
  tools at all.
- **Verify with the bench, not the model card.** Model cards are
  optimistic. The probe + bench in this project is what tells you
  whether parsers actually work for *your* serving stack version.
- **Don't conflate `reasoning_content` with `<think>`.** The model
  emits `<think>` *tokens*; the parser separates them into the
  `reasoning_content` *field*. If you see literal `<think>` in your
  app's output, the parser failed.
- **Don't conflate "tools" with "function calling" with
  "agentic".** Tools are a JSON schema convention. Function calling
  is what tools-aware models do. "Agentic" is the higher-level
  loop where a host program actually executes the tools and feeds
  results back. The model is responsible only for emitting the
  middle bit.
- **Cold-start cost includes parser flags.** Changing reasoning
  override or tool-call mode triggers a container recreate (see
  [`nvfp4-coldstart.md`](nvfp4-coldstart.md) Sec. 1). The picker
  surfaces these as separate rows so users opt-in deliberately.
- **Streaming complicates everything.** `tool_calls.arguments`
  arrives one character at a time and must be JSON-validated only
  after the stream closes. Same for `reasoning_content`. See
  [`openai-api-and-streaming.md`](openai-api-and-streaming.md) for
  the wire format.

---

## 6. References

### Chat templates

- HuggingFace chat templating documentation:
  <https://huggingface.co/docs/transformers/chat_templating>. The
  Jinja2 template format reference, with examples for ChatML, Llama
  3, Mistral, and Phi.
- OpenAI ChatML format (the original):
  <https://github.com/openai/openai-python/blob/main/chatml.md>.
  Historical reference -- most modern models use a variant.
- Qwen3 tokeniser config (chat_template, special tokens):
  <https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer_config.json>.

### Tool calling

- OpenAI function-calling guide (the *de facto* protocol):
  <https://platform.openai.com/docs/guides/function-calling>.
- vLLM tool-calling documentation, including parser plugins:
  <https://docs.vllm.ai/en/latest/features/tool_calling.html>.
- HuggingFace text-generation-inference tool docs (a separate but
  similar implementation):
  <https://huggingface.co/docs/text-generation-inference/index>.
- Hermes Function Calling spec (used by many open models):
  <https://github.com/NousResearch/Hermes-Function-Calling>.

### Reasoning models

- DeepSeek-R1 paper (2025): *DeepSeek-R1: Incentivizing Reasoning
  Capability in LLMs via Reinforcement Learning.*
  [arXiv:2501.12948](https://arxiv.org/abs/2501.12948). Introduces
  the `<think>` block convention as an RL training target.
- Qwen3 technical report -- section on the thinking mode and
  `enable_thinking` flag.
- gpt-oss model card (openai/gpt-oss-20b on HuggingFace) -- describes
  the harmony channel format.
- vLLM reasoning parser docs:
  <https://docs.vllm.ai/en/latest/features/reasoning_outputs.html>.

### Project-internal

- [`router.md`](router.md) -- request rewrite chain, including
  `tool_choice_pinning_required` and reasoning-override propagation.
- [`backends.md`](backends.md) -- probe lifecycle that verifies
  parser flags before the router uses them.
- [`bench-results.md`](bench-results.md) -- TPS-counting fix history,
  template-leak issue with Nemotron, BPE-decode bug with
  R1-Distill-Llama-8B.
- [`attention-and-the-transformer.md`](attention-and-the-transformer.md)
  -- what the model *does* with the tokens the chat template
  produced.
- [`openai-api-and-streaming.md`](openai-api-and-streaming.md) -- 
  the wire format these structured outputs travel over.
- `scripts/model-families.yaml` -- per-family curated parser hints.
- `deploy/.vllm-reasoning-cache.json` -- probe-verified parser
  results.
