# The OpenAI-compatible API and SSE streaming

This page documents what is *actually on the wire* when an agent talks
to vLLM, SGLang, Ollama, or any other server claiming OpenAI
compatibility. Everything in this project's router code, bench
harness, agent integrations (Claude Code, Aider, Codex, Open WebUI),
and probe drivers ultimately speaks this protocol.

If you can read JSON and you have ever sent an HTTP POST, you have
the prerequisites. If you have read
[`reasoning-tool-calling-chat-templates.md`](reasoning-tool-calling-chat-templates.md),
the structured-output fields below will already feel familiar.

---

## 1. The endpoint surface

The OpenAI-compatible servers in this project expose a small set of
HTTP endpoints. The vast majority of traffic hits exactly one:

| Endpoint | What it does | Used by |
|---|---|---|
| `POST /v1/chat/completions` | take a list of role-tagged messages, return an assistant message (or stream it) | every chat agent |
| `POST /v1/completions` | legacy raw-text completion (no chat structure) | rarely used; some autocomplete tools |
| `POST /v1/embeddings` | turn text into a vector for similarity search | RAG pipelines |
| `GET  /v1/models` | list models the server is currently serving | picker, healthchecks, agents on startup |
| `GET  /health` | "am I alive and ready?" | the router's polling loop after a container recreate |
| `POST /v1/messages` | Anthropic's variant of /v1/chat/completions | Claude SDK, Claude Code |

Ports in this project:

- `11434` -- Ollama (and the router's Ollama-compat proxy)
- `11435` -- vLLM (via router)
- `11436` -- SGLang (via router)

All three speak the OpenAI surface plus health endpoints. Ollama
also speaks its native `/api/chat` and `/api/generate`; Anthropic-
compatible `/v1/messages` is wired through the router.

---

## 2. Anatomy of a chat completion request

A minimal request:

```json
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "Qwen3-8B-NVFP4@131072",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is 2 + 2?"}
  ]
}
```

That is it. Every other field has a sensible default. The complete
field reference for the request body:

### 2.1 Routing fields

| Field | Type | Purpose |
|---|---|---|
| `model` | string | required. Names the model. The picker's `<name>::<reasoning>@<ctx>` syntax is parsed by the router (this project's extension); raw OpenAI servers see only the prefix before the first `::` or `@`. |
| `stream` | bool | `false` (default) -> return one big response. `true` -> return a Server-Sent Events stream. See Sec. 4. |

### 2.2 Generation control

| Field | Type | Default | Purpose |
|---|---|---|---|
| `temperature` | float >= 0 | 1.0 | sampling temperature; 0 = greedy, higher = more random. See [`sampling-strategies.md`](sampling-strategies.md). |
| `top_p` | float (0,1] | 1.0 | nucleus sampling cutoff. |
| `top_k` | int | -1 | (vLLM/SGLang extension) keep top-k logits. -1 = no cutoff. |
| `min_p` | float | 0.0 | (vLLM extension) minimum probability filter. |
| `max_tokens` | int | server default (often 16 or model max) | hard cap on generated tokens. |
| `max_completion_tokens` | int | -- | OpenAI's newer name for the same thing; servers accept either. |
| `stop` | string \| string[] | none | stop generation when one of these strings appears. |
| `frequency_penalty` | float | 0 | discourage repeated tokens proportional to past count. |
| `presence_penalty` | float | 0 | discourage tokens that have appeared at all. |
| `seed` | int | none | for repeatability under sampling -- same seed + same model -> same output. |
| `logprobs` | bool | false | return per-token log-probabilities in the response. |
| `top_logprobs` | int | 0 | how many alternatives per position. |

### 2.3 Tool calling (see also [`reasoning-tool-calling-chat-templates.md`](reasoning-tool-calling-chat-templates.md) Sec. 2)

| Field | Type | Purpose |
|---|---|---|
| `tools` | array | declared functions, each with `name`, `description`, JSON-schema `parameters`. |
| `tool_choice` | string \| object | `"auto"` / `"none"` / `"required"` / `{type: "function", function: {name: "X"}}`. |
| `parallel_tool_calls` | bool | allow the model to emit multiple tool calls in one assistant turn. |

### 2.4 Reasoning (vLLM/SGLang extensions for thinking models)

| Field | Type | Purpose |
|---|---|---|
| `reasoning_effort` | enum | `"low"` / `"medium"` / `"high"`. Hints the model to spend more or fewer tokens reasoning. |
| `extra_body.chat_template_kwargs.enable_thinking` | bool | per-request override of the chat template's `enable_thinking` slot. The router uses this to implement `::nothink`. |
| `extra_body.chat_template_kwargs.separate_reasoning` | bool | (SGLang) split reasoning vs final into separate channels. |

### 2.5 Response shaping

| Field | Type | Purpose |
|---|---|---|
| `response_format` | object | force structured output. `{"type": "json_object"}` (any JSON), `{"type": "json_schema", "json_schema": {...}}` (specific schema). |
| `n` | int | how many independent completions to generate. Almost always 1 in practice. |
| `user` | string | client-supplied user identifier (logged for abuse tracking on hosted providers; no-op on local servers). |
| `stream_options.include_usage` | bool | when streaming, include the final `usage` block in the last event. **Critical** for accurate metering -- bench harness uses this. |

---

## 3. Anatomy of a non-streaming response

For `stream: false` (or omitted), you get one JSON document:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1735000000,
  "model": "Qwen3-8B-NVFP4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "4",
        "reasoning_content": "The user asked what 2+2 is..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 1,
    "total_tokens": 25
  }
}
```

### 3.1 The `choices` array

Always length-1 unless you set `n > 1`. The interesting fields:

- `message.role` -- always `"assistant"` for non-tool responses.
- `message.content` -- the text content. May be `null` if the model
  emitted only tool calls.
- `message.reasoning_content` -- the `<think>` content extracted by
  the reasoning parser, if any.
- `message.tool_calls` -- array of structured tool-call objects, if
  the model decided to call tools.
- `finish_reason` -- why generation stopped:
  - `"stop"` -- the model emitted its end-of-turn token (e.g.
    `<|im_end|>`) or hit a `stop` string.
  - `"length"` -- hit `max_tokens`.
  - `"tool_calls"` -- finished with tool calls; agent should execute
    and continue.
  - `"content_filter"` -- refused by safety filter.

### 3.2 The `usage` block

Token accounting from the model's perspective:

- `prompt_tokens` -- input length after chat-template rendering and
  tokenisation. **Includes** the system prompt, all prior turns,
  and the chat template scaffolding tokens.
- `completion_tokens` -- output length. **Subject to the gotcha
  documented in
  [`reasoning-tool-calling-chat-templates.md`](reasoning-tool-calling-chat-templates.md)
  Sec. 3.4** -- vLLM's qwen3 reasoning parser excludes `reasoning_content`
  tokens from this count. The bench harness's TPS fix works around
  this.
- `total_tokens` -- sum of the above. What hosted providers bill on.

---

## 4. Streaming with Server-Sent Events (SSE)

When `stream: true`, the server replies with `Content-Type:
text/event-stream` and emits a series of events as it generates. Each
event is a single line of the form:

```
data: <json>

```

(with a literal blank line after each event -- the SSE framing). The
final event is the literal:

```
data: [DONE]

```

A typical decode of "What is 2 + 2?" -> "4" looks like:

```
data: {"id":"chatcmpl-x","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}

data: {"id":"chatcmpl-x","choices":[{"index":0,"delta":{"content":"4"}}]}

data: {"id":"chatcmpl-x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-x","choices":[],"usage":{"prompt_tokens":24,"completion_tokens":1,"total_tokens":25}}

data: [DONE]

```

Notice:

- Each event has a **`delta`** instead of a full `message`. The
  client must accumulate `delta.content`, `delta.tool_calls`, etc.
  across events.
- The first event typically carries `role: "assistant"` and an
  empty content (signals "stream open, expect deltas").
- Subsequent events carry incremental content fragments.
- `finish_reason` arrives in the second-to-last event with empty
  delta.
- The `usage` block (if `stream_options.include_usage: true`)
  arrives in the last event before `[DONE]`. Without this option,
  you get *no* usage data when streaming -- a common surprise.

### 4.1 Why SSE and not WebSockets

SSE is HTTP-native. It works through every proxy, load-balancer,
reverse-proxy, and CDN that handles HTTP/1.1 chunked transfer
encoding. WebSockets require explicit upgrade handshakes that many
infrastructure layers do not pass through. For server -> client
streaming with no client -> server back-channel, SSE is strictly
simpler.

The cost: SSE only flows one direction (server -> client). The
client must reopen a new request to send anything new. For LLM
inference this is fine -- each agent turn is one request.

### 4.2 Time-to-first-token in practice

The wall-clock delta `t_first_event - t_request_open` is the
**TTFT** that matters for user experience. It bundles network
latency, prefill cost, and the time to emit the first non-empty
delta.

The bench harness in `scripts/bench/_bench_core.py` records this as
follows (excerpt):

```python
t_open = time.time()
t_first_token = None
for t_event, payload in http_post_stream(url, body):
    if payload == "[DONE]":
        break
    obj = json.loads(payload)
    delta = obj.get("choices",[{}])[0].get("delta", {})
    if t_first_token is None and (delta.get("content") or
                                   delta.get("reasoning_content")):
        t_first_token = t_event
        # -> ttft_ms = (t_first_token - t_open) * 1000
```

Crucially, the bench counts *any* generated token toward TTFT -- 
content or reasoning. From the user's perspective, "the model started
producing" is the relevant event. This is documented in
[`bench-results.md`](bench-results.md)'s methodology and matches how
agents like Claude Code measure their own latency.

### 4.3 Tool-call streaming -- the trickiest case

Tool calls stream piecemeal too. A typical pattern:

```
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"city"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\":\""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Paris"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"}"}}]}}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}
```

Things to notice:

- `id`, `name`, and `type` appear once (in the first event for that
  tool call); subsequent events for the same `index` only carry
  `arguments` deltas.
- `arguments` arrive as **JSON-encoded string fragments**, not
  pre-parsed JSON. The client must concatenate them and only
  `JSON.parse` once `finish_reason: "tool_calls"` arrives.
- Multiple tool calls in one assistant turn use distinct
  `tool_calls[i].index` values. Streaming events can interleave
  them (rare in practice -- most engines emit sequentially).

### 4.4 Reasoning-content streaming

For models with `--reasoning-parser`, reasoning tokens arrive in
`delta.reasoning_content` (or `delta.reasoning` on older vLLM
versions). The bench's `stream_chat_completion` handles both:

```python
reasoning_piece = (
    delta.get("reasoning_content")
    or delta.get("reasoning")
)
```

A typical reasoning-then-content stream looks like:

```
data: {"choices":[{"delta":{"reasoning_content":"The user asked..."}}]}
data: {"choices":[{"delta":{"reasoning_content":" let me work this out."}}]}
... (long reasoning trace) ...
data: {"choices":[{"delta":{"reasoning_content":"So the answer is 4."}}]}
data: {"choices":[{"delta":{"content":"4"}}]}
data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
data: {"usage":{"prompt_tokens":24,"completion_tokens":2,"total_tokens":26}}
data: [DONE]
```

Notice the `usage.completion_tokens: 2` -- only the content tokens.
This is the bug
[`bench-results.md`](bench-results.md)'s "TPS counting fix" worked
around.

---

## 5. The Anthropic /v1/messages variant

Claude SDK (and therefore Claude Code) talks Anthropic's protocol,
which has a *very similar* but not identical schema:

```json
POST /v1/messages
Content-Type: application/json
anthropic-version: 2023-06-01

{
  "model": "claude-opus-4-7",
  "max_tokens": 1024,
  "messages": [
    {"role": "user", "content": "What is 2 + 2?"}
  ],
  "system": "You are a helpful assistant."
}
```

Differences from `/v1/chat/completions`:

- System prompt is a **top-level `system` field**, not a message
  with `role: "system"`.
- `max_tokens` is **required**, not defaulted.
- Response uses `content: [{"type": "text", "text": "..."}]`
  (a typed array, supporting mixed text/image/tool blocks) instead
  of `message.content: "..."`.
- Tool use uses `content: [{"type": "tool_use", ...}]` blocks rather
  than a separate `tool_calls` field.
- Streaming uses **named SSE events** (`event: message_start`,
  `event: content_block_delta`, etc.) instead of unnamed `data:`
  events.

The router in this project translates between the two so Claude
Code can talk to local vLLM/SGLang/Ollama models served through any
of the OpenAI-compatible engines.

---

## 6. Practical implications

- **`stream_options.include_usage: true` is almost always what you
  want.** Without it, streaming responses give you no token counts -- 
  no metering, no rate calculation, no bench TPS.
- **Tool-call arguments are not valid JSON until the stream
  closes.** Don't try to parse them mid-stream; you will see
  partial strings.
- **`message.content` can be `null`** when the model only emits tool
  calls. Code that does `message.content.length` will crash.
- **`usage.completion_tokens` undercounts reasoning models.** Use
  the bench's chars-/-4 fallback if you need parser-agnostic
  token-rate measurements.
- **The `model` field in the response may differ from the request.**
  vLLM echoes back `--served-model-name`, which the router sets to
  the bare model name without `@<ctx>` or `::<reasoning>` suffixes.
- **Different servers expose different extensions.** vLLM has
  `top_k`, `min_p`, `extra_body`. SGLang has `separate_reasoning`.
  Ollama has `options.num_ctx`. Pure-OpenAI clients see only the
  baseline; the router is what bridges these.
- **Pure raw `httpx` / `urllib` is enough.** You do not need the
  `openai` SDK to speak this protocol -- `scripts/bench/_bench_core.py`
  uses stdlib `urllib.request` and works fine. The SDK adds typed
  models and retry logic, no protocol magic.

---

## 7. Worked example -- full request/response with curl

Inspect the wire format yourself against the project's running
router:

```bash
# Non-streaming
curl -s http://devai-router:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-8B-NVFP4",
    "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    "max_tokens": 16
  }' | jq .

# Streaming, with usage in the final event
curl -N http://devai-router:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-8B-NVFP4",
    "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    "max_tokens": 16,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

The first command returns one JSON document. The second emits the
SSE stream described in Sec. 4. Both go through the router which applies
the request rewrite chain ([`router.md`](router.md)) before
proxying.

---

## 8. References

### Protocol specifications

- OpenAI Chat Completions API reference:
  <https://platform.openai.com/docs/api-reference/chat/create>.
  The canonical schema everyone copies.
- OpenAI Streaming guide:
  <https://platform.openai.com/docs/api-reference/streaming>.
- Anthropic Messages API reference:
  <https://docs.anthropic.com/en/api/messages>. The
  `/v1/messages` variant.
- Server-Sent Events (SSE) HTML standard:
  <https://html.spec.whatwg.org/multipage/server-sent-events.html>.
  Defines the `data:` framing.

### Engine-specific extensions

- vLLM OpenAI-compatible server documentation:
  <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>.
  Lists every vLLM-specific request extension (`top_k`, `min_p`,
  `extra_body`, etc.).
- SGLang server arguments:
  <https://docs.sglang.ai/router/router.html>.
- Ollama API documentation:
  <https://github.com/ollama/ollama/blob/main/docs/api.md>.
  Native `/api/chat` and `/api/generate`, plus the OpenAI-compat
  proxy.

### Project-internal

- [`router.md`](router.md) -- the request rewrite chain, including
  override parsing, reasoning-policy application, tool stripping,
  and ctx injection. The router is *the* place where one request
  morphs across these protocols.
- [`backends.md`](backends.md) -- backend-specific lifecycle and
  parser registration.
- [`reasoning-tool-calling-chat-templates.md`](reasoning-tool-calling-chat-templates.md)
  -- the structured outputs (tool calls, reasoning content) this
  protocol carries.
- [`sampling-strategies.md`](sampling-strategies.md) -- the
  generation-control fields (`temperature`, `top_p`, etc.) in Sec. 2.2.
- `scripts/bench/_bench_core.py::stream_chat_completion` -- a
  worked, working SSE consumer in 80 lines of stdlib Python.
