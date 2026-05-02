# Ollama Reasoning Semantics

This document defines the proposed semantics for identifying and controlling
reasoning, thinking, and chain-of-thought style behavior for models served by
the Ollama backend. It intentionally covers only Ollama for now. vLLM and
SGLang should not reuse these rules until their behavior is tested separately.

## Goals

- Distinguish model capability from activation mechanics.
- Treat the local Ollama runtime as the source of truth for downloaded models.
- Avoid relying on model names, families, README text, or upstream claims as
  final proof of reasoning support.
- Give every agent the same reasoning policy, even when agents use different
  API protocols.
- Make reasoning control explicit, reproducible, and testable.

## Core Concepts

Use three separate concepts.

### Capability

Capability describes what a specific local model can do when served by the
current Ollama runtime.

Examples:

```yaml
reasoning:
  capability: structured
```

Valid values:

- `structured`: the model emits a separate reasoning trace field.
- `inline`: the model emits reasoning text inline, for example in `<think>`
  blocks, but not in a separate structured response field.
- `unsupported`: no reasoning behavior was observed or the runtime cannot
  control it.
- `unknown`: the model has not been probed yet.
- `error`: probing failed.

Only `structured` should be considered first-class reasoning support.
`inline` can be displayed or filtered, but it should not be treated as clean
agent-compatible reasoning because it can contaminate normal answer text.

### Protocol Control

Protocol control describes how a request to a specific API protocol enables or
disables reasoning.

For Ollama native APIs, use this global protocol recipe:

```yaml
reasoning:
  native_api:
    on_request: { think: true }
    off_request: { think: false }
    trace_field: message.thinking
```

Meaning:

- `on_request` is the JSON field the router injects when policy says reasoning
  should be enabled.
- `off_request` is the JSON field the router injects when policy says reasoning
  should be disabled.
- `trace_field` is the response location where structured reasoning is expected.

These fields are protocol facts, not model facts. Do not repeat them for every
model unless a model requires a confirmed exception.

### Agent Transport

Agent transport describes which API protocol an agent uses to reach Ollama.

Examples:

- Open Interpreter may use Ollama native or OpenAI-compatible requests,
  depending on invocation.
- Aider usually uses OpenAI-compatible chat completions for non-native setups.
- Codex uses OpenAI-compatible chat or responses semantics depending on its
  configured provider.
- Claude Code uses Ollama's Anthropic-compatible messages endpoint.

Agents should not own model-specific reasoning activation logic. The router
should normalize requests according to model capability, protocol, and global
reasoning policy.

## Recommended Data Model

Store protocol recipes globally:

```yaml
reasoning_protocols:
  ollama_native:
    on_request: { think: true }
    off_request: { think: false }
    trace_field: message.thinking

  openai_chat:
    on_request: { reasoning_effort: medium }
    off_request: { reasoning_effort: none }
    trace_field: choices[].message.reasoning

  anthropic_messages:
    on_request:
      thinking:
        type: enabled
        budget_tokens: 2048
    off_request: {}
    trace_field: content[type=thinking]
```

Store per-model facts separately:

```yaml
models:
  - name: "qwen3.5:9b"
    backend: [ollama]
    source: ollama
    reasoning:
      capability: structured
      detected_by: ollama_probe
      probed_at: "2026-04-26T00:00:00Z"
      model_digest: "sha256:..."
      native_api:
        supported: true
        enable_verified: true
        disable_verified: true
        trace_observed: true
```

Do not store this per model:

```yaml
reasoning:
  native_api:
    on_request: { think: true }
    off_request: { think: false }
```

That duplicates global protocol behavior and makes the catalog harder to
maintain.

## Detection Algorithm

Detection must be performed against local downloaded models.

### Step 1: Discover Local Ollama Models

Use Ollama as the source of truth:

- `/api/tags` for local model names and digests when available.
- `/api/show` for model metadata, details, capabilities, parameters, template,
  and model_info.

Catalog data may enrich display output, but it must not decide capability.

### Step 2: Native Positive Probe

For each downloaded model, call `/api/chat` with:

```json
{
  "model": "<model>",
  "messages": [
    {
      "role": "user",
      "content": "Answer with only the final number: What is 17 + 25?"
    }
  ],
  "think": true,
  "stream": false,
  "options": {
    "temperature": 0,
    "num_predict": 128
  }
}
```

Classify:

- If `message.thinking` is present and non-empty, capability is `structured`.
- If `message.content` contains visible reasoning markers such as `<think>`,
  capability is `inline`.
- If neither appears, capability is `unsupported`.
- If the request fails or times out, capability is `error`.

### Step 3: Native Negative Probe

For models classified as `structured`, call `/api/chat` with the same prompt
and:

```json
{
  "think": false,
  "stream": false
}
```

Classify disable support:

- `disable_verified: true` if `message.thinking` is absent or empty.
- `disable_verified: false` if reasoning still appears.
- `disable_verified: unknown` if the request fails.

This matters because some models default reasoning on. Omitting the control
field is not equivalent to disabling reasoning.

### Step 4: Cache by Digest, Not Name

The probe cache (`deploy/.ollama-reasoning-cache.json`, schema v3) is
keyed by `digest` -- one record per set of weights. Each record carries:

- `aliases`: every name pointing at this digest (`ollama:latest`,
  `ollama:9b`, `ollama:9b-q4_K_M` ...)
- `max_context`: the architecture's design ceiling from `/api/show`
- `capability` and `disable_verified`: canonical, taken at the smallest
  fitting tier (most reliable signal)
- `probes`: a 2-D map nested by VRAM band then context tier, e.g.
  `probes["16"]["32768"]` and `probes["24"]["131072"]`. Each cell is a
  measurement record with `vram_gb`, `ctx`, `actual_total_gb`,
  `actual_vram_gb`, `fully_on_gpu`, per-cell capability, and timestamp.

Probing is incremental and never destructive. A new (band, tier) cell
only fills a gap; existing cells are immutable unless `--force-ctx`
or `--force` is passed. Two aliases of the same digest probe at most
once per cell -- the probe driver dedups before issuing chat calls. If
a tier is above `max_context`, it's silently capped: a 128K-only model
with tiers `[32K, 64K, 128K, 256K]` records probes at `[32768, 65536,
131072]` and is shown at higher tiers as "limited to 128K".

The orchestrator (`make probe`) loops over VRAM bands. Before each pass
it recreates devai-ollama with `OLLAMA_GPU_OVERHEAD` set to
`(host_vram - target_vram) * 1024^3` bytes, so the daemon behaves as
if it had only the target VRAM. This lets a 24G host produce cache
cells valid for 16G targets without needing physical hardware swaps.

When the digest disappears from `/api/tags` (model deleted), the entry
is dropped on the next probe run. When an alias disappears (retag), it's
removed from `aliases` but the digest entry stays.

## Runtime Policy

Expose one global policy:

```bash
DEVAI_REASONING=auto|off|low|medium|high
```

Semantics:

- `auto`: enable reasoning for `structured` models, leave unsupported models
  alone.
- `off`: inject each protocol's `off_request` when disable support is known.
- `low`: enable reasoning with a small budget or low effort where the protocol
  supports levels.
- `medium`: default explicit reasoning mode for coding agents.
- `high`: explicit high-effort reasoning mode.

For Ollama native APIs, `low`, `medium`, and `high` all map to:

```yaml
on_request: { think: true }
```

Ollama native boolean `think` does not express effort levels. Effort levels are
only meaningful for protocols that support them.

## Router Responsibilities

The router should apply reasoning control before forwarding requests to Ollama.

For Ollama native `/api/chat` and `/api/generate`:

- If policy enables reasoning and model capability is `structured`, inject
  `think: true`.
- If policy disables reasoning and disable support is verified, inject
  `think: false`.
- If model capability is `unsupported`, do not inject reasoning fields unless
  explicitly requested.
- If the client already sent an explicit `think` field, client input should win
  unless an administrative override is configured.

For OpenAI-compatible `/v1/chat/completions`:

- Map enabled reasoning to the configured OpenAI-compatible request field.
- Map disabled reasoning to the configured OpenAI-compatible off field.
- Prefer protocol fields over prompt text tricks.

For Anthropic-compatible `/v1/messages`:

- Map enabled reasoning to the configured Anthropic-compatible `thinking`
  request field.
- Do not use prompt injection as the primary activation mechanism.

## Agent Responsibilities

Agents should not hard-code per-model reasoning activation.

Agents should only provide:

- selected model
- selected backend
- optional user policy override

The router should handle protocol-specific activation.

## Picker Behavior

The picker should not hide all non-reasoning models by default.

Recommended display:

```text
qwen3.5:9b        Native reasoning   policy: auto
deepseek-r1:8b    Native reasoning   policy: auto
gemma4:e4b        No reasoning       standard chat
qwen3.5:27b       CPU offload        slow / not preferred
```

Filtering should be explicit:

- show all measured labels by default
- optional filter: reasoning-capable only
- optional filter: Native reasoning only
- optional filter: probe errors

## Avoid These Patterns

Do not infer capability only from:

- model name
- family name
- Ollama library tags such as "thinking"
- upstream README text
- HuggingFace tokenizer metadata
- hard-coded `/think` or `<|think|>` prompt prefixes

These sources are useful hints, but local runtime behavior must win.

Do not activate reasoning primarily by prompt text when a protocol field exists.
Prompt text activation is brittle across templates, agents, and model versions.

Do not store duplicate YAML keys such as:

```yaml
thinking: false
thinking: true
```

Use one structured object instead:

```yaml
reasoning:
  capability: structured
```

## Migration Plan

1. Add global `reasoning_protocols` metadata.
2. Add an Ollama probe command that writes per-model `reasoning` facts for
   downloaded Ollama models.
3. Change the picker to read `reasoning.capability` instead of `thinking`.
4. Change the router to inject protocol request fields based on policy.
5. Remove duplicate and manually assigned `thinking` keys from active catalogs.
6. Keep family-level hints only as pre-probe hints, not final capability.

## Minimal First Implementation

The smallest useful implementation is:

- probe only `/api/chat`
- support only `structured`, `unsupported`, and `error`
- store only `capability`, `model_digest`, `probed_at`, and native verification
- support only `DEVAI_REASONING=auto|off`
- inject only native `{ think: true }` and `{ think: false }`

After that works, add OpenAI-compatible and Anthropic-compatible request
rewriting for the agents that need them.
