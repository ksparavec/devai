# DevAI Router (`gpu-arbiter`)

The router is the central component of the DevAI stack. Every model
request from agents (Claude Code, Aider, Codex, Open WebUI, raw curl)
goes through it. The router is responsible for: which backend serves a
model, which model is currently loaded in that backend, what context
length and parser flags the backend was launched with, and how every
incoming request is rewritten on the way through.

This document is the source of truth for the router. It groups
behaviour by concern: ports, lifecycle, request rewrites, configuration,
caches consumed, and failure modes.

---

## Table of contents

- [Architecture](#architecture)
- [Port layout](#port-layout)
- [What the router advertises](#what-the-router-advertises)
- [Backend lifecycle](#backend-lifecycle)
- [Request rewrite chain](#request-rewrite-chain)
  - [1. Override parsing](#1-override-parsing-namectxreasoning)
  - [2. Anthropic /v1/messages normalisation](#2-anthropic-v1messages-normalisation-vllmsglang)
  - [2b. Responses API reasoning](#2b-responses-api-reasoning-v1responses-vllmsglang)
  - [3. Reasoning policy](#3-reasoning-policy)
  - [4. Tool-choice promotion](#4-tool-choice-promotion-vllmsglang)
  - [5. Tool stripping](#5-tool-stripping-vllmsglang)
  - [6. Context injection](#6-context-injection-ollama-only)
- [vLLM plugin registry](#vllm-plugin-registry)
- [Caches consumed](#caches-consumed)
- [Configuration (env vars)](#configuration-env-vars)
- [Failure modes](#failure-modes)
- [Operator tasks](#operator-tasks)
- [Benchmark harness](#benchmark-harness)

---

## Architecture

The router is a Go reverse proxy in front of three inference backends.
It listens on three ports inside the `devai-net` network, one per
backend, and proxies requests after rewriting them.

`--mode=single` (or `DEVAI_MODE=single`) is the only supported mode.
The flag is still accepted so a stale compose file or an old habit
produces a clear error instead of being silently ignored: any other
value exits with a pointer to `attic/README.md`.

The `worker` and `head` modes were **frozen on 2026-07-25** and their
implementation now lives under `attic/cluster-mode/`, behind a
`//go:build devai_frozen_cluster` tag and outside every Go module, so
it is not compiled into this binary. They were never made to work --
head mode called `log.Fatalf` on a bearer-token file that compose never
mounted, and its control plane was never published off the container
network. See `attic/README.md` for the reasoning and
`attic/cluster-mode/RESTORE.md` for the defects still open against
them.

This document covers single mode, which is all of it.

```
+--------------+                         +------------------------------+
|   agents     |                         |        devai-router          |
| (Claude Code,|                         |                              |
|  Aider,      |  POST /v1/...           |  port 11434  -->  ollama     |
|  Codex,      | ------------------>     |  port 11435  -->  vllm       |
|  Open WebUI, |                         |  port 11436  -->  sglang     |
|  curl, ...)  |                         |                              |
+--------------+                         |  GPU mutex * request rewrite |
                                         |  container lifecycle         |
                                         +------------------------------+
                                                       |
                                                       v
                                         +------------------------------+
                                         |  devai-ollama (always on)    |
                                         |  devai-vllm   (sleep > live) |
                                         |  devai-sglang (sleep > live) |
                                         +------------------------------+
```

The vLLM and SGLang containers start as `sleep infinity` placeholders
(see `deploy/docker-compose.yaml`). The router replaces them with a
live engine on the first request to their port via the Podman libpod
API, with the right `--model`, `--max-model-len`, parser flags, and
plugin mounts derived from the probe cache.

### Cluster mode (frozen)

Head and worker modes are no longer part of this binary. The design and
the code are preserved under `attic/cluster-mode/` and described in
`attic/README.md`; do not treat them as available behaviour. If they are
ever thawed, `attic/cluster-mode/RESTORE.md` lists what was still broken
when they were parked.

### What the router IS

- A multi-port reverse proxy with **deterministic routing by port**
  (no message inspection -- port determines backend).
- A **GPU mutex**: only one backend uses the GPU at a time. Switching
  drains in-flight requests, stops the loser, recreates the winner.
- A **container lifecycle manager** for vLLM/SGLang: stop, recreate,
  health-poll, idle-timeout.
- A **request rewriter**: parses suffix overrides, applies reasoning
  policy, promotes/strips tool_choice, injects num_ctx for Ollama.
- A **probe-cache reader**: synthesises serving rows from three caches
  on boot and feeds the same lookup maps as the legacy
  `active-models.yaml` did.

### What the router is NOT

- Not a model server. It owns no inference state.
- Not aware of message content. It does not parse prompts.
- Not a load balancer. There's exactly one of each backend.

---

## Port layout

| Port  | Backend | Listener | Models                          |
|-------|---------|----------|---------------------------------|
| 11434 | Ollama  | always live  | GGUF only                   |
| 11435 | vLLM    | recreated on demand | NVFP4, FP8, AWQ, BF16 safetensors |
| 11436 | SGLang  | recreated on demand | NVFP4 (arch-dependent), BF16 safetensors |

The router is internal to `devai-net`. It is **not** published to the
host. Reach it from sibling containers (`devai-open-webui`,
`devai-lab-*`) or from your own container with `--network devai-net`.

---

## Backend lifecycle

### Boot

1. Read three probe caches:
   - `/etc/devai/.ollama-reasoning-cache.json` (digest-keyed v3)
   - `/etc/devai/.vllm-reasoning-cache.json` (repo+sha-keyed v2)
   - `/etc/devai/.sglang-reasoning-cache.json` (repo+sha-keyed v2)
2. Read `/etc/devai/vllm-plugins.json` (parser plugin registry).
3. Synthesise one `configModel` per cache entry whose host-VRAM band
   has a `fits=true` (HF) / `fully_on_gpu=true` (Ollama) cell.
4. Build per-backend lookup maps: model size, declared context, probed
   max ctx, tool parser, reasoning parser, tool mode, capability,
   disable_verified.
5. Start one HTTP listener per backend.

### First request to vLLM/SGLang (cold start)

```
agent --> router (port 11435 or 11436)
              |
              +- parse `<model>@<ctx>` and `<model>::<reasoning>` overrides
              +- identify other backends holding the GPU
              |   +- drain their in-flight requests (DRAIN_TIMEOUT)
              |   +- stop their containers (or unload Ollama models)
              |   +- release GPU
              +- build launchConfig from probe cache + overrides
              |   +- compute MemFraction for the host VRAM
              |   +- resolve parser plugin paths (vllm-plugins.json)
              |   +- apply probe-verified ctx ceiling
              +- stop + remove placeholder container
              +- podman libpod create + start with full launch flags
              +- poll /health (HEALTH_TIMEOUT_SECONDS, default 600s);
              |   fail fast if the container exits or its logs show a
              |   terminal error (no more full-timeout waits on a crash)
              +- proxy the original request through
```

Cold start typically takes 60-90s for BF16 weights, up to 300s for
NVFP4 with CUDA graph capture.

### Subsequent requests, same model

Direct proxy. No recreate.

### Request for a different model on the same backend

`currentModel` and `currentContext` are tracked per backend. If either
differs from the request, the router stops and recreates. Concurrent
requests for the same target model coalesce on `recreateCond` so only
one container start happens.

### Idle

`idleWatcher` polls every 30s and delegates to `idleSweepOnce`. With
`IDLE_TIMEOUT=0` (the default -- keep-warm) it never auto-unloads: a
loaded model stays resident until a *different* model is requested, so a
cold start is paid once per model, not again after every idle gap. Set
`IDLE_TIMEOUT` to a positive value to restore time-based unloading, where
a backend idle longer than the timeout is stopped and replaced with the
`sleep infinity` placeholder.

### Router restart while a backend is serving

GPU exclusion is driven entirely by in-memory state: `stopOtherBackends`
skips any backend whose `running` and `containerLaunched` are both false.
Both start false in a fresh process, so a router restarted while a
backend held the GPU used to believe the GPU was free -- and the next
request to a *different* backend launched an engine into an
already-committed card. The engine dies with "Engine core initialization
failed", which reads like a bad model or a bad context rather than what
it is. Observed on a routine `make build-router` + recreate while
`devai-sglang` was serving: 22.3 of 24.5 GB were already taken and the
vLLM launch had no chance.

`reconcileBackendState` runs once at startup and adopts any HF backend
that answers `/health`. It is a health probe rather than a container
check on purpose: `devai-vllm` and `devai-sglang` exist as `sleep
infinity` placeholders whenever compose has run, so "container running"
says nothing -- only an answer on `/health` separates a live engine from
a placeholder. Ollama is skipped: its container is always up and always
answers, and `running` for Ollama means "a model is loaded", which
`unloadOllama` establishes from `/api/ps` at switch time.

The adopted backend's `currentModel` is deliberately left empty. The
router knows a backend is live but not what it loaded; the cost is one
extra recreate if the next request happens to want the resident model,
and the alternative -- trusting a guess -- risks serving from the wrong
weights.

### Backend switch (GPU exclusion)

When a request hits a different backend than the one currently on the
GPU:
1. Wait for the active backend's in-flight requests to drain
   (`DRAIN_TIMEOUT`, default 30s). "In-flight" means *already proxied
   upstream* (`upstreamReqs`), not every request the arbiter has
   accepted: requests still parked on the arbiter mutex are not waited
   on. They cannot drain while the switch itself holds that mutex, so
   counting them would stall every switch under load for the full
   `DRAIN_TIMEOUT`.
2. Ollama: send `keep_alive=0` to all loaded models so it releases
   VRAM. Other backends: stop their container.
3. Recreate the target backend with the new model.

---

## What the router advertises

Two endpoints, registered per backend port: `/v1/models` (OpenAI shape)
and `/api/tags` (Ollama shape). Both answer from a **vetted** subset, and
that subset is deliberately NOT the same as what the router will serve.

| Set | Contents | Read by |
| --- | --- | --- |
| **advertised** | probed AND benched AND weights in THIS backend's store AND no bench verdict | `/v1/models`, `/api/tags` |
| **serveable** (`modelNames`) | has a fitting probe cell | the request handler's allowlist |

**Why they must differ.** The bench harness is itself a router client --
`bench_runner.py` drives every scored task through `router_url + "/v1"`.
If the serving allowlist required a bench row, a newly probed model could
never earn its first one: the router would refuse it, so it would never
be benched, so the router would keep refusing it. Gating only the
*advertisement* breaks that loop and still satisfies "never advertise
anything un-vetted", because the bench names its target explicitly and
never reads the listing.

Concretely, a withheld model behaves like this:

```
POST /v1/chat/completions {"model":"Qwen3-8B-NVFP4"}   -> 503, "has no weights on disk at
                                                          /var/cache/devai/sglang/... run
                                                          `make model-pull NAME=...`"
POST /v1/chat/completions {"model":"does-not-exist"}   -> 404, "unknown model"
GET  /v1/models                                        -> does not list it
```

The 503 is `checkModelWeights` failing fast with an actionable message;
the 404 is the allowlist. A withheld model is reachable, a nonexistent
one is not.

**The live passthrough is filtered too.** Both handlers first ask the
running engine for its own model list and merge the result. That echoes
whatever is currently resident, which is exactly when an un-vetted model
would be most visible, so it is intersected with the vetted set before
merging.

**Vetting inputs** are read once at startup, like every other cache the
router consumes -- `BENCH_CACHE` (default `/etc/devai/.bench-cache.json`)
and `MODEL_STATUS_LEDGER` (default `/etc/devai/.model-status.json`), both
mounted read-only. A missing file means "nothing benched", so nothing is
advertised: the safe direction for a display decision, and loud, because
each backend logs its gap at startup:

```
vllm: advertising 5 of 16 probed model(s) -- withheld: 11 no weights in this
backend's store, 0 never benched, 0 bench-dropped. Withheld models are still
SERVEABLE by explicit name ...
```

Only `bench_*` verdicts gate advertisement. A probe-side verdict (`oom`,
`unsupported_arch`, `manual`) already prevented the model reaching
`modelNames`, and re-applying it here would conflate two different
questions. The ctx rule matches the Python `is_bench_excluded`: a verdict
recorded at ctx N applies at N and above, and one with no recorded ctx
applies everywhere.

See `gpu-arbiter/advertise.go`.

## Request rewrite chain

Every non-trivial request goes through this chain in order. Each step
inspects and may mutate the JSON body. Steps are skipped when not
applicable.

### 1. Override parsing (`<name>@<ctx>` / `<name>::<reasoning>` / `<name>::<mtp>`)

The model name in the request body may carry up to three suffixes.
`peelControlSuffixes` peels whichever recognised suffix is currently
trailing and loops until none remain, so **the order the client uses
does not matter**:

| Suffix form           | Meaning                                                |
|-----------------------|--------------------------------------------------------|
| `<name>@<ctx>`        | per-session `--max-model-len` for vLLM/SGLang          |
| `<name>::mtp`         | enable multi-token-prediction (`::nomtp` to force off) |
| `<name>::<reasoning>` | per-request reasoning policy override                  |
| `<name>::nothink`     | shortcut for `enable_thinking=false`                   |

Picker convention: `<name>::<reasoning>::<mtp>@<ctx>` (each suffix
optional, ctx last). But some clients append their own suffix out of
order -- e.g. aiagent/litellm carries a `default_reasoning` and appends
`::<reasoning>` AFTER the picker's `@<ctx>`, yielding
`<name>@<ctx>::<reasoning>`. A strict ctx-last strip choked on that
(`Atoi("<ctx>::nothink")` fails, leaving `@<ctx>` glued to the name so
the allowlist rejected it as unknown). The order-independent peel handles
either ordering: each sub-parser (`parseCtxOverride` / `parseMTPOverride`
/ `parseReasoningOverride`) strips only a token it recognises (integer
ctx / mtp keyword / reasoning keyword) and otherwise leaves the name
unchanged, so a name that legitimately contains `::` or `@` survives
intact, and every peel shortens the name so the loop always terminates.

**Ollama: `@<ctx>` is what pins a tier.** A bare `<name>` is served
from whatever tier is already loaded -- the router does not re-derive a
tier per request and does not recreate the container to move to one.
Only an explicit `<name>@<ctx>` (`ctxPinned`) can force a tier switch,
and then only when the pinned tier differs from the running one. This
matters for mixed-KV models, where different tiers were probed under
different KV dtypes: the picker emits `@<ctx>` for exactly those, and a
bare name simply reuses the resident tier.

Examples:
- `Qwen3-8B-NVFP4@65536` -> recreate vLLM with `--max-model-len 65536`,
  request body's `model` rewritten to `Qwen3-8B-NVFP4`.
- `qwen3.5:9b-q8_0::nothink` -> set Ollama's `think:false` for this
  request only.
- `gpt-oss-20b::low@131072` -> vLLM gets `--max-model-len 131072`,
  request gets `reasoning_effort: low`.
- `Nemotron-3-Nano-30B-A3B-NVFP4@131072::nothink` (aiagent/litellm
  order, reasoning appended after ctx) -> identical result to
  `::nothink@131072`: recreate vLLM with `--max-model-len 131072` and
  `enable_thinking=false`. The order-independent peel makes the two
  spellings equivalent.
- `Gemma-4-26B-A4B-NVFP4::mtp@32768` -> vLLM recreate with
  `--max-model-len 32768` AND `--speculative-config '{"method":"mtp",
  "model":"/models/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}'`.
  Catalog must declare `mtp:` for the model (see
  [`multi-token-prediction.md`](multi-token-prediction.md) Sec. 7.2).
- `Qwen3-14B-NVFP4::think::mtp@65536` is **rejected** with HTTP 400 --
  the reasoning+MTP+inline-reasoning combo triggers
  [vllm#34650](https://github.com/vllm-project/vllm/issues/34650).
  Use `::nothink::mtp@65536` or omit `::mtp`.

### 2. Anthropic `/v1/messages` normalisation (vLLM/SGLang)

Claude Code sends a `role:"system"` message **inside** `messages[]`, in
addition to a correct top-level `system`. The Anthropic Messages API
defines message roles as `user` | `assistant` only, and both engines'
compat shims implement that stricter schema, so every turn was rejected:

```
400 1 validation error:
  {'type': 'literal_error', 'loc': ('body', 'messages', 1, 'role'),
   'msg': "Input should be 'user' or 'assistant'", 'input': 'system'}
```

`normaliseAnthropicMessages` (`gpu-arbiter/anthropic_compat.go`) moves
every message whose role is not `user`/`assistant` into the top-level
`system` block list, preserving order, and leaves the rest of the body
alone. A bare-string `system` or `content` is promoted to a text block
first. When there is nothing to move the original bytes are returned
**unchanged**, so the common path costs no re-serialisation. Each rewrite
logs once, naming how many messages moved -- a silent body rewrite is
very hard to debug from the client side.

**Gating: backend `vllm` or `sglang`, AND path exactly `/v1/messages`.**
Ollama is verified tolerant of the exact shape Claude Code sends, and
rewriting a path that already works is unnecessary risk.

Scope was fixed by replay against the live engines, not by reading
schemas. Using a real captured Claude Code body (183 KB, 25 tools, every
beta field present) against vLLM:

| Request shape                                | Result |
| -------------------------------------------- | ------ |
| as-is                                        | 400    |
| folded, beta fields KEPT                     | 200    |
| folded, beta fields stripped                 | 200    |

So folding alone is sufficient and **no field filtering is needed** --
`context_management`, `output_config`, `thinking`, `metadata` and `tools`
are not the blocker and are passed through untouched. SGLang was verified
independently and behaves identically (see
`docs/plans/sglang-backend-remediation.md` Phase 0, finding 11).

This is a client/server API-version mismatch rather than a devai defect:
Claude Code emits a newer Anthropic beta wire format (note the
`?beta=true` query) that the pinned engine images do not implement. The
rejected alternative was bumping vLLM until its shim accepts the newer
schema -- not under our control, invalidates the probe and bench caches
for every HF row, and would have to be repeated every time the client's
format moves again.

### 2b. Responses API reasoning (`/v1/responses`, vLLM/SGLang)

Codex speaks **only** this wire -- `wire_api = "chat"` was removed
upstream and `responses` is now the sole variant -- and the reasoning
rewrite used to be gated to `/v1/chat/completions` and `/v1/messages`, so
every Codex request went through with no reasoning policy at all.

Both engines implement the endpoint: vLLM v0.22.1 verified live, SGLang
v0.5.10 registers it at `http_server.py:1563`.

**The shape differs, and the wrong one is accepted silently.** Measured
against the live engine serving `gpt-oss-20b`:

| Request field | Result |
| --- | --- |
| `reasoning: {"effort": "low"}` | 200 -- reasoning_tokens **298 -> 37** |
| `reasoning_effort: "low"` | 200 -- reasoning_tokens **282** (ignored) |

So simply widening the path gate and reusing `applyVLLMPolicy` would have
produced a fix that looked right and did nothing. `applyResponsesPolicy`
emits the Responses-native `reasoning.effort` instead.

Verified end-to-end through the router afterwards, same prompt:

```
bare name (policy=auto)   reasoning_tokens=249     <- no injection, model default
<name>::low               reasoning_tokens=59
<name>::high              reasoning_tokens=298
```

Rules:

- `low` / `medium` / `high` -> `reasoning.effort` = that level.
- `auto` injects **nothing**. The checkpoint's own default is the right
  answer; inventing `medium` would silently override it. The body comes
  out byte-identical.
- `off` -> `effort: "none"`, but only when the probe verified the model
  honours a disable directive. Harmony models reject `none` with
  `400 "Supported values are: high, medium, low"` -- and they also probe
  as `disable_verified: false` for exactly that reason, so the existing
  gate keeps them out. **Known limitation:** that verification was
  performed on `/v1/chat/completions` and does not strictly transfer here.
- A client-supplied `reasoning` object is never overridden.

**Tool handling on this surface.** Tool *stripping* needs no change:
`maybeStripTools` is not path-gated and operates on the top-level
`tools` / `tool_choice` keys, which the Responses API also uses. Tool
*choice promotion* is now explicitly skipped, because the engine will not
accept a pin here at all:

| `tool_choice` sent | Result |
| --- | --- |
| `"auto"` | 200 |
| `{"type":"function","name":...}` (flat) | 501 "Only 'auto' or 'none' tool_choice is supported in response API with Harmony" |
| `{"type":"function","function":{"name":...}}` (what the router emits) | 400 "Tool choice 'function' not found in 'tools' parameter" |

It already no-opped by accident -- `toolNameAt` reads the nested Chat
Completions tools shape while the Responses API requires a **flattened**
one (`{"type":"function","name":...}`; the nested form is rejected with
400 and 25 validation errors) -- so the skip is now deliberate rather
than a happy consequence of a shape mismatch.

See `gpu-arbiter/responses_compat.go`.

### 3. Reasoning policy

Driven by the `DEVAI_REASONING` env (default `auto`) plus the
`X-DevAI-Reasoning` header plus the `::<reasoning>` suffix override
(suffix wins over header wins over env). Values: `auto | off | low |
medium | high`.

The action depends on the model's capability (from the probe) and the
backend's protocol path:

| Backend / Path | Capability   | Action                                   |
|----------------|--------------|------------------------------------------|
| Ollama `/api/chat`, `/api/generate` | structured | inject `think: <true\|false>` |
| Ollama `/v1/chat/completions`       | structured | inject `reasoning_effort` (`low`/`medium`/`high`, or `none` to disable) |
| Ollama `/v1/messages`               | structured | inject `thinking.type` |
| vLLM `/v1/chat/completions`         | structured | inject `extra_body.chat_template_kwargs.enable_thinking` + `reasoning_effort` |
| SGLang `/v1/chat/completions`       | structured | inject `extra_body.chat_template_kwargs.enable_thinking` + `separate_reasoning` |
| Any                                 | inline + policy=off | log `reasoningDisable` (explicit user opt-out) |
| Any                                 | none / unsupported  | noop |

Models with `disable_verified=False` can't reliably suppress
reasoning -- the directive is sent, but the model may emit reasoning
anyway (R1-Distill family is the standing example).

### 4. Tool-choice promotion (vLLM/SGLang)

Fires when the probe's `tool_mode == "forced"` -- the model only
verified tool calls when `tool_choice` was pinned to a specific
function, not under `auto`. Applies the rule:

| `tool_choice` in request | `len(tools)` | Router action |
|---|---|---|
| absent or `"auto"` | 1 | rewrite to `{"type":"function","function":{"name":tools[0].function.name}}` |
| absent or `"auto"` | >1 | reject with HTTP 400, structured error (see below) |
| `"required"`       | any | pass through (agent took ownership) |
| `{type:function,...}` | any | pass through (already pinned) |
| `"none"`           | any | pass through (agent disabled tools) |

Models with `tool_mode == "auto"` (probe verified spontaneous calls
work) skip this step entirely. Models without a verified `tool_parser`
fall through to step 5 (tool stripping).

**Multi-tool reject payload:**

```json
{
  "error": {
    "type": "invalid_request_error",
    "code": "tool_choice_pinning_required",
    "message": "Model \"DeepSeek-R1-Distill-Qwen-7B\" requires tool_choice to be pinned to a specific function when called with multiple tools. Set tool_choice to {\"type\":\"function\",\"function\":{\"name\":\"<one of: get_time, search>\"}}, or route this request to a non-reasoning model (e.g. Qwen3.5-9B-Q8, Llama-3.1-8B-Instruct) that handles auto tool_choice reliably.",
    "param": "tool_choice"
  }
}
```

**Verified tool modes (vLLM, 24G host):**

| Model | Tool parser | `tool_mode` | Use case |
|---|---|---|---|
| `Qwen3-8B-NVFP4`             | `hermes`         | `auto`   | best general-purpose tool caller (auto works) |
| `Qwen3-14B-NVFP4`            | `hermes`         | `auto`   | larger Qwen3 with same auto behaviour |
| `Qwen3.5-9B-NVFP4`           | `qwen3_xml`      | `auto`   | Qwen3.5 with XML-style tool format |
| `gpt-oss-20b`                | `openai`         | `auto`   | harmony format; auto works |
| `Llama-3.1-8B-Instruct-NVFP4`| `llama3_json`    | `forced` | non-reasoning, cap=none; forced single-tool works |
| `DeepSeek-R1-Distill-Llama-8B` | `deepseek_string` | `forced` | reasoning, but **5 tokens** to call when forced |
| `DeepSeek-R1-Distill-Qwen-7B`  | `deepseek_string` | `forced` | reasoning; 500+ tokens of CoT before the call |

`auto` rows pass through the router unchanged. `forced` rows hit the
single-tool promote / multi-tool reject rules above. See
`docs/backends.md` "Operational notes -- R1-Distill family" for the
behavioural difference between the two distill variants.

### 5. Tool stripping (vLLM/SGLang)

Fires when the model has **no** verified `tool_parser` in the cache.
Without an engine-level `--tool-call-parser` flag, vLLM rejects every
request that carries `tool_choice` other than `"none"` with
`BadRequestError: "auto" tool choice requires --enable-auto-tool-choice
and --tool-call-parser`. The router strips `tools` and `tool_choice`
so the request becomes a plain chat. Cost: tool-calling silently
unavailable for that model. Benefit: chat works without backend errors.

Ollama is unaffected -- its protocol negotiates tool support per
request and tolerates `tools=[]` without launch flags.

### 6. Context injection (Ollama only)

For Ollama's native `/api/chat` and `/api/generate` paths, the router
injects `options.num_ctx = <effective ctx>` into the body. The
effective context is `min(model.declared_context, MAX_CONTEXT_LEN)`
clamped against the probe-verified ceiling. This makes per-session
context binding work without minting Modelfile-derived tags.

Ollama's `/v1/chat/completions` and `/v1/messages` compat paths
ignore `options.num_ctx` -- for those the global
`OLLAMA_CONTEXT_LENGTH` is the only knob.

vLLM/SGLang use `--max-model-len` / `--context-length` baked into the
container at recreate time, not per-request injection.

The vLLM entrypoint also passes `--kv-cache-dtype`, but the value is
**per-model, not a global hardcode**. It is resolved from the probe
cell covering the launch context (`resolveKVCacheType`): each cell is
stamped with the KV dtype it was measured under, and a legacy cell with
no stamp decodes to `fp8` -- the historical hardcode it was factually
probed with. For SGLang an unstamped cell emits no flag at all (engine
default). Probe-time and serve-time therefore always agree, which is
the point: the fit data in the cache is only valid for the dtype it was
measured under.

fp8 KV is what lets NVFP4 checkpoints (~18 GiB weights) cohabit with
128K-context KV on the project's 24 GiB reference GPU. Default fp16 KV
adds ~7 GiB at 128K -- enough to push the total past 24 GiB. fp8 halves
that to ~3.5 GiB, leaving room for activations and the engine's CUDA
graph workspace. Blackwell exposes native fp8 so there is no
throughput cost; older GPUs fall back to vLLM's fp8 emulation. A model
with VRAM slack can be re-probed with `PROBE_KV_CACHE_TYPE=auto` to
serve unquantized KV instead -- there is no global dtype policy.

On the Ollama side the same smallest-covering-tier rule now supplies
`OLLAMA_FLASH_ATTENTION` as well, from the probe cell's own
`flash_attention` stamp (`resolveFlashAttention`, sibling of
`resolveKVCacheType`). It used to be derived purely from the KV dtype.
That derivation is only half-true -- quantized KV requires flash
attention, but a cell can equally have been probed with flash
attention ON under the default `f16` dtype, and the dtype-derived
value would then serve it WITHOUT flash, i.e. in a different
environment from the one its fit was measured in. The stamp wins
whenever present; pre-stamp cells keep the historical dtype-derived
value. Both stamps are read from the SAME covering tier, so a launch
always reproduces one probe cell.

## Per-model recovery flags

For checkpoints whose CUDA-graph workspace alone pushes them past 24
GiB at high context (Nemotron-3-Nano-30B-A3B-NVFP4 at 128K is the
canonical case), the router appends additional flags from
`deploy/recovery-flags.json` after the parser block:

```json
{
  "models": {
    "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4": {
      "backends": ["vllm"],
      "engine_flags": ["--max-num-seqs", "8",
                       "--trust-remote-code",
                       "--enforce-eager"],
      "engine_env": {
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "VLLM_USE_FLASHINFER_MOE_FP4": "1",
        "VLLM_FLASHINFER_MOE_BACKEND": "throughput"
      }
    }
  }
}
```

Per-entry keys:

| Key            | Meaning                                                                 |
|----------------|--------------------------------------------------------------------------|
| `engine_flags` | CLI args appended after the parser block (and after `--max-num-seqs`, so a per-model value wins). |
| `engine_env`   | Env vars merged into the recreated container.                             |
| `backends`     | Optional allow-list (`["vllm"]`). Four cases, see below. |
| `image`        | Optional per-model container image override, falling back to `$VLLM_IMAGE`. Needed when one checkpoint requires a different engine build than the global default. |

`backends` exists because the rescue flags are mostly vLLM-only
(`--language-model-only`, `--quantization modelopt`, `--max-num-seqs`,
`VLLM_*` env) and were previously appended verbatim to SGLang launches,
where those flags do not exist. All 10 current entries are scoped to
`["vllm"]`.

The full contract -- identical in `gpu-arbiter/recovery_flags.go` and
`scripts/_probe_hf_common.py`, so probe and serve-time launches always
agree on which entry applies:

| `backends` value          | Meaning                                                     |
|---------------------------|-------------------------------------------------------------|
| key **absent** (or `null`) | Applies to EVERY backend. Backward compatible with pre-`backends` entries. |
| key present, `[]`          | Applies to NO backend. This is the operator's disable switch for an entry -- absent and empty deliberately mean opposite things, which is why the Go decoder holds the field as a pointer. |
| key present, list          | Applies only to the named backends.                          |
| key present, **non-list**  | Malformed: logs a warning naming the model and is treated as ABSENT (applies everywhere). An explicit `null` is not warned about -- it is the absent case. |

Decoding is **per-entry**: one malformed entry is skipped or degraded
with a warning and never discards the rest of the registry. The
canonical wording of this contract lives in the `_comment` header of
`deploy/recovery-flags.json` and in the `recoveryEntry.Backends` doc
comment.

A skipped entry logs:

```
recovery registry: entry for <model> is scoped to [vllm] -- not applied to sglang
```

`--enforce-eager` disables CUDA graph capture entirely, reclaiming the
~4 GiB workspace vLLM otherwise pre-reserves. The model then loads
with the fp8 KV cache fitting comfortably and 128K-context Q&A works
end-to-end (probe records `fits=true, vram=22.79 GiB`). Cost is a
~10-20% decode throughput hit from losing graph batching -- the going
trade to reach 128K on a 24 GiB card.

`scripts/_probe_hf_common.py` reads the same JSON so the vLLM/SGLang
probers launch with the same flags. Without this symmetry the probe
would record `fits=false` at 128K, the picker would hide the cell,
and the router would never get a chance to use the recovery flags.

---

## vLLM plugin registry

`deploy/vllm-plugins.json` is the single source of truth for custom
parsers. Format:

```json
{
  "container_dir": "/etc/devai/vllm-plugins",
  "plugins": {
    "deepseek_string": {
      "kind": "tool",
      "file": "deepseek_string_tool_parser.py"
    }
  }
}
```

Both the prober and the router consult this map. When a parser name
matches an entry:

- The router bind-mounts `VLLM_PLUGINS_HOST_DIR` (default
  `scripts/vllm_plugins`) into the recreated vLLM container at
  `container_dir`.
- The launch args gain `--tool-parser-plugin <abs>` (or
  `--reasoning-parser-plugin <abs>`) **before** the parser-name flag -- 
  vLLM resolves parser names at flag-parse time.
- An empty `VLLM_PLUGINS_HOST_DIR` for a plugin model fails the
  recreate with an actionable error.

Names absent from the registry pass through as built-in vLLM parsers.

Adding a new plugin: drop the file in `scripts/vllm_plugins/`, add one
entry to `deploy/vllm-plugins.json`, reference the parser name from a
family's `parsers.vllm.tool` (or `.reasoning`) in
`scripts/model-families.yaml`. No router code change.

See `docs/backends.md` "Custom vLLM parser plugins" for the full
adding-a-plugin recipe.

---

## Caches consumed

The router reads only -- never writes. Keep the writers single-source.

| File                                       | Schema | Writer                       | Keys              |
|--------------------------------------------|--------|------------------------------|-------------------|
| `deploy/.ollama-reasoning-cache.json`      | v3     | `probe-ollama-reasoning.py`  | digest            |
| `deploy/.vllm-reasoning-cache.json`        | v2     | `probe-vllm-reasoning.py`    | `<repo>@<sha>`    |
| `deploy/.sglang-reasoning-cache.json`      | v2     | `probe-sglang-reasoning.py`  | `<repo>@<sha>`    |
| `deploy/vllm-plugins.json`                 | v1     | hand-edited                  | parser name       |

Top-level fields used by the router:

| Field             | From                                  | Used by                                 |
|-------------------|---------------------------------------|-----------------------------------------|
| `aliases`         | catalog row name + ollama tags        | model name lookup                       |
| `size_gb`         | catalog row                           | `memFraction` math                      |
| `max_context`     | largest clean probe `actual_context`  | declared context cap                    |
| `capability`      | smallest clean probe                  | reasoning policy                        |
| `tool_parser`     | latest cell with `tool_parser` set    | engine launch flag, strip-tools gate    |
| `reasoning_parser`| latest cell with `reasoning_parser` set | engine launch flag                    |
| `disable_verified`| latest cell with `disable_verified`   | reasoning-disable gate. A non-boolean value (e.g. the string `"error"` an older prober wrote on a failed disable probe) no longer fails the whole cache parse -- that one model degrades to "unknown" and the disable rewrite is simply not applied. |
| `tool_mode`       | same cell as `tool_parser` (`evidence.tool.mode`) | tool-choice promotion       |
| `probes[vram][ctx].fits` | per-cell verdict               | row eligibility, ProbedMaxCtx           |

**Top-level field derivation.** `capability` comes from the
*smallest-tier* clean probe (most conservative classification).
Parser fields and `tool_mode` come from the **most-recently-probed
clean cell that has them populated** -- `_latest_cell_with` in
`scripts/_probe_hf_common.py`. This split lets a partial `--force`
re-probe of a single cell update the top-level row without requiring
a full matrix re-probe; older cells with stale `None`s no longer
shadow new evidence (e.g. when a curated parser hint is added to a
family that already had probed cells).

A missing probe cache is non-fatal: the corresponding backend exposes
zero models. Run `make probe` (Ollama) or `make probe-vllm` /
`make probe-sglang` (HF) to populate.

**`_meta` image-digest block (vLLM/SGLang).** Each HF cache carries a
top-level `_meta` key (NOT a model row) written by the prober:

```json
"_meta": {
  "current_image_digest": "sha256:...",
  "current_image_ref": "docker.io/vllm/vllm-openai:v0.22.1-...",
  "image_history": { "sha256:...": { "image_ref": "...", "first_seen": "..." } }
}
```

At boot the router reads `_meta.current_image_digest` (via
`readProbedImageDigest`) and compares it against the digest of the
running backend image (`imageDigestFromLibpod`, a libpod
`/images/{name}/json` query). A mismatch means the cache's fit /
`serving_ok` / parser data was measured on a **different image** -- see
"Backend image drift" under Failure modes. Every consumer skips the
`_meta` key (the Go synthesizer filters it via the schema-version /
aliases guards; Python readers skip `_`-prefixed keys).

---

## Configuration (env vars)

Set in `deploy/docker-compose.yaml` under the `router` service or on
the shell when invoking compose.

### Backend wiring

| Variable             | Default                                                    | Purpose                       |
|----------------------|------------------------------------------------------------|-------------------------------|
| `OLLAMA_URL`         | `http://devai-ollama:11434`                                | upstream                      |
| `OLLAMA_PORT`        | `11434`                                                    | router listen                 |
| `VLLM_URL`           | `http://devai-vllm:11434`                                  | upstream                      |
| `VLLM_PORT`          | `11435`                                                    | router listen                 |
| `VLLM_CONTAINER`     | `devai-vllm`                                               | name to recreate              |
| `VLLM_IMAGE`         | `docker.io/vllm/vllm-openai:v0.22.1-x86_64-cu129-ubuntu2404`       | image to launch               |
| `VLLM_MODELS_DIR`    | `/var/cache/devai/vllm`                                    | host path bound to `/models`  |
| `SGLANG_URL`         | `http://devai-sglang:11434`                                | upstream                      |
| `SGLANG_PORT`        | `11436`                                                    | router listen                 |
| `SGLANG_CONTAINER`   | `devai-sglang`                                             | name to recreate              |
| `SGLANG_IMAGE`       | `docker.io/lmsysorg/sglang:v0.5.10.post1-cu130`            | image to launch               |
| `SGLANG_MODELS_DIR`  | `/var/cache/devai/sglang`                                  | host path bound to `/models`  |
| `NETWORK`            | `devai-net`                                                | podman network name           |
| `PODMAN_SOCKET`      | `/run/podman/podman.sock`                                  | libpod socket inside router   |

### Lifecycle

| Variable                  | Default | Purpose                                                            |
|---------------------------|---------|--------------------------------------------------------------------|
| `IDLE_TIMEOUT`            | `0`     | seconds before an idle backend is auto-unloaded; `0` = never (keep-warm) |
| `DRAIN_TIMEOUT`           | `30`    | seconds to wait for in-flight requests when switching backends     |
| `HEALTH_TIMEOUT_SECONDS`  | `600`   | health-poll deadline for a *hung* load; a crashed engine fails fast via log/exit detection |
| `MAX_CONCURRENT_REQUESTS`| `32`    | max in-flight requests per backend before HTTP 429; `0` = unlimited **and** omits `--max-num-seqs` / `--max-running-requests` entirely (engine default). Any positive value is also passed to the engine as that flag. |
| `DEVAI_SSE_KEEPALIVE_SECONDS` | `10` | interval between `: keepalive` SSE comment frames during a slow launch; `0` disables the feature |
| `DEVAI_SSE_KEEPALIVE_GRACE_SECONDS` | `5` | how long a launch may take before the first frame is sent (and the response is committed as SSE) |
| `DEVAI_MAX_FAILED_LAUNCHES` | `3` | consecutive launches of the same `(model, ctx)` that may fail to produce a real engine response before the router refuses; `0` disables the breaker. See [Launch circuit breaker](#launch-circuit-breaker-engine-dies-after-passing-health). |

### SSE keepalive during cold start

The router holds a client for the whole launch window before a single
byte moves: `makeRequestHandler -> ensureBackendRunning ->
containerRecreate -> waitForHealthy -> proxy.ServeHTTP`. An NVFP4 cold
start is bounded by `HEALTH_TIMEOUT_SECONDS` (default 600s). A browser or
corporate proxy with a 30-60s idle timeout drops the connection long
before that, and the client's retry lands on a router that is still
loading -- so the expensive load is wasted and the cycle repeats. devai
ships explicit `HTTP_PROXY`/`HTTPS_PROXY` support, so intermediaries are
an expected part of the deployment.

While a launch is in progress the router therefore writes SSE comment
lines (`: keepalive <n>`), which every OpenAI/Anthropic client ignores and
which reset an intermediary's idle timer. Three properties are worth
knowing:

- **SSE surfaces only.** The gate requires an explicit `"stream": true`
  **and** a `/v1/` path. Ollama's native `/api/chat` and `/api/generate`
  answer in newline-delimited JSON *and* default `stream` to true when
  the field is absent -- a comment frame there is a parse error for every
  Ollama-native client. Ollama's own OpenAI-compat `/v1/` endpoint is
  SSE and does participate.
- **Nothing happens on the warm path.** No byte is written, and no header
  is committed, until the launch has already outrun
  `DEVAI_SSE_KEEPALIVE_GRACE_SECONDS`. The overwhelming majority of
  requests find the backend warm and keep their exact status-code
  behaviour.
- **Committing is one-way.** Once the first frame is written the response
  is a `200 text/event-stream`, so a launch failure after that point can
  no longer be a 5xx. It is reported in-band instead -- a `data:` frame
  plus `data: [DONE]` on the OpenAI surfaces, an `event: error` frame on
  `/v1/messages` -- because a stream that merely stops is
  indistinguishable from a hang and would make the client wait out its
  own timeout.

One ordering constraint is load-bearing. The keepalive is armed only
*after* `makeRequestHandler` has read the POST body and replaced it with
an in-memory reader. Writing the first frame commits the response, at
which point Go's server closes the ORIGINAL request body -- so a
keepalive armed any earlier makes `ReverseProxy` fail to forward the
request (`http: invalid Read on closed Body`), and because the response
is already a committed 200 the client receives heartbeats followed by
nothing at all. This was verified by writing it the wrong way round
first, and is pinned by
`TestKeepaliveIsArmedAfterTheBodyIsReplaced`.

Implementation and rationale: `gpu-arbiter/sse_keepalive.go`.

### Memory and context

| Variable           | Default  | Purpose                                                          |
|--------------------|----------|------------------------------------------------------------------|
| `GPU_MEMORY_GB`    | `24`     | total GPU VRAM; drives memory fraction calc                      |
| `MAX_CONTEXT_LEN`  | `262144` | global ceiling clamping any model's per-name context             |

### Reasoning

| Variable              | Default | Purpose                                                                  |
|-----------------------|---------|--------------------------------------------------------------------------|
| `DEVAI_REASONING`     | `auto`  | global policy: `auto\|off\|low\|medium\|high`. Per-request override via `::<token>` suffix or `X-DevAI-Reasoning` header. |

### Probe caches

| Variable               | Default                                           | Purpose         |
|------------------------|---------------------------------------------------|-----------------|
| `PROBE_CACHE`          | `/etc/devai/.ollama-reasoning-cache.json`         | Ollama cache    |
| `VLLM_PROBE_CACHE`     | `/etc/devai/.vllm-reasoning-cache.json`           | vLLM cache      |
| `SGLANG_PROBE_CACHE`   | `/etc/devai/.sglang-reasoning-cache.json`         | SGLang cache    |

### Plugin registry

| Variable                  | Default                            | Purpose                                                     |
|---------------------------|------------------------------------|-------------------------------------------------------------|
| `VLLM_PLUGINS_REGISTRY`   | `/etc/devai/vllm-plugins.json`     | path the router reads inside its own container              |
| `VLLM_PLUGINS_HOST_DIR`   | `""` (set by Makefile)             | host path to bind-mount into recreated vLLM containers      |

The Makefile sets `VLLM_PLUGINS_HOST_DIR = $(abspath scripts/vllm_plugins)` and exports it for compose interpolation.

---

## Failure modes

### Plugin required but `VLLM_PLUGINS_HOST_DIR` empty

```
podman create devai-vllm: vllm plugin required (tool="deepseek_string" reasoning="") but VLLM_PLUGINS_HOST_DIR is empty
```

Cause: a model's verified `tool_parser` is in the plugin registry but
the router doesn't know the host path to mount. Fix: ensure compose
runs through `make cache-up` (which sets the env via Makefile export),
or set `VLLM_PLUGINS_HOST_DIR` manually.

### Multi-tool reject (HTTP 400)

```
{"error":{"type":"invalid_request_error","code":"tool_choice_pinning_required",...}}
```

Cause: model has `tool_mode=forced`, request has `tool_choice="auto"`
or absent, and `len(tools) > 1`. The router can't pick a tool for the
agent. Fix: pin `tool_choice` client-side to a specific function, or
route to a non-reasoning model with `tool_mode=auto` (Qwen3.5,
Llama-3.1).

### Backend cold-start timeout

```
devai-vllm did not become ready within 600s
```

Cause: NVFP4 weights with CUDA graph capture can take 5+ min on
consumer GPUs; sometimes longer on first-ever load (kernel
JIT-compilation cached afterwards). Fix: bump
`HEALTH_TIMEOUT_SECONDS=900` and retry, or rerun the request -- the
second attempt usually hits warm caches.

### Launch circuit breaker (engine dies after passing `/health`)

```
sglang: Ornith-1.0-9B-NVFP4 failed to serve 3 consecutive launches at
ctx=131072 and is now refused. The engine started and passed /health but
died before completing a request ...
```

Cause: an engine that launches cleanly, answers `/health`, and then dies
serving. `detectLaunchFailure` cannot see this -- it catches an engine
that dies *during* launch. Without a breaker the router relaunches
forever: one SGLang model was recreated **72 times in a single day**,
completing zero work and holding the GPU throughout. `podman inspect`
reports `RestartCount: 0` the whole time, because each cycle is a brand
new container, so that counter is no use for spotting it.

Behaviour: each launch of a `(model, ctx)` spends one unit of a budget
(`DEVAI_MAX_FAILED_LAUNCHES`, default 3; `0` disables the breaker). The
budget is repaid **only when the engine returns a real, non-5xx
response**, which is the only evidence a launch was actually good.
Repayment fires from the proxy's `ModifyResponse`, i.e. when the
response headers arrive -- so a long generation still counts, while an
engine that never answers gets no credit (the proxy's default error
handler synthesises a 502 without ever reaching that hook). On
exhaustion the router refuses with the message above, naming the model,
the context, and both remedies.

Two deliberate details:

- **`/health` is not proof.** SGLang's HTTP server outlives its
  scheduler, so a corpse keeps answering `200 OK` on `/health` while
  every real request fails. Crediting there is what made the 72-recreate
  loop possible.
- **The reset is unkeyed.** Whatever is running right now just answered,
  so the running launch is good, full stop. Keying it would reintroduce
  the bug it prevents: the attempt is charged against the *resolved*
  ctx while the request is served at the *launched* ctx, and any drift
  would leave the budget permanently unreset and eventually refuse a
  working model.

Fix: re-measure the model (`make probe-load-vllm` / `probe-load-sglang`)
or request a smaller `@<ctx>`. Requesting a different model clears the
budget.

### Backend image drift (serve-with-warning)

```
WARNING: vllm image drift -- probe cache captured on sha256:AAA but running
image docker.io/vllm/vllm-openai:... is sha256:BBB; serving with
X-DevAI-Warning. Re-run `make probe-vllm`.
```

Cause: the backend image tag moved (e.g. a floating `latest`, or an
operator `podman pull`) after the probe cache was captured, so the
cache's fit / `serving_ok` / parser data no longer describes the image
that will actually serve. This is the exact rot that made a
previously-working NVFP4 model start crashing at load.

Behaviour (Phase C, decided policy -- "serve anyway, it probably
works"): the router does **not** refuse the model. It still launches it
(a genuine crash is then failed hard and fast by the crash-detection in
`waitForHealthy`), but:

- logs the loud WARNING above once at boot per drifted backend;
- sets an `X-DevAI-Warning` response header on every response from that
  backend (advisory, non-blocking -- the body and status are untouched);
- reports `image_stale: true` plus `probed_image_digest` /
  `running_image_digest` in that backend's `/health`.

Detection fails **open**: an unreachable podman, an absent image, or a
pre-Phase-C cache with no `_meta` all yield "no baseline" and no
warning (never a false positive). Ollama is exempt -- it runs an
unmodified upstream image with no PyTorch cold-start surface, so its
cache is not stamped and never flagged.

Fix: refresh the cache against the running image --
`make cache-down && make probe-vllm && make probe-load-vllm` (or the
`sglang` equivalents). Preview which backends are stale without
restarting anything: `make probe-check`.

### Recreate race on concurrent requests

The first concurrent request observes `running=false` and starts a
recreate; subsequent requests for the same target wait on
`recreateCond` rather than launching a duplicate. If the lead recreate
fails, all waiters wake and propagate the same error. No duplicate
`podman create` calls.

### Stuck Ollama models holding GPU

Ollama's `keep_alive=0` directive normally releases VRAM. If a model
gets stuck loaded (rare; usually the daemon is wedged), the router's
`unloadOllama` loops over `/api/ps` and unloads each. Worst case:
restart `devai-ollama`.

---

## Operator tasks

### Reload the router after a code or cache change

```bash
make build-router      # if Go code changed
make cache-down
make cache-up
```

Compose mounts caches read-only and re-reads them on every router
boot. No restart-on-write -- the router holds its working set in
memory.

### Inspect what the router is doing

```bash
make logs SERVICE=devai-router LINES=80
make logs SERVICE=devai-vllm LINES=120     # backend that's currently active
```

Useful log lines:

| Pattern                                        | Meaning                                  |
|------------------------------------------------|------------------------------------------|
| `probe cache: ... loaded (N entries -> M serving rows)` | per-cache load summary           |
| `vllm plugin registry: ... loaded`             | plugin map consumed                       |
| `vllm launch: model=X GB ... tool="Y" tool_plugin="Z"` | recreate spec                  |
| `stopping <other-backend> (switching to <target>)` | GPU mutex switch                      |
| `draining <backend> (N requests in flight upstream)` | drain-on-switch (counts only requests already proxied upstream) |
| `<backend> idle, stopping`                     | idle timeout fired                       |
| `ERROR: <cache> failed to parse: ... -- ZERO <backend> models registered` | probe cache unreadable; the router will 404 every request for that backend and the picker lists none. Replaces the older `warning: probe cache parse failed`. |
| `error: unloadOllama: keep_alive=0 for <model> failed/returned <status> -- GPU may still be held by ollama` | the Ollama unload that precedes a backend switch did not take; VRAM may still be held |
| `recovery registry: entry for <model> is scoped to [vllm] -- not applied to sglang` | a per-model recovery entry was skipped because of its `backends` allow-list |

### Verify a single request end-to-end

From a sibling container on `devai-net`:

```bash
podman run --rm --network devai-net curlimages/curl:latest \
  -X POST http://devai-router:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-8B-NVFP4@32768",
    "messages": [{"role":"user","content":"hi"}],
    "max_tokens": 64
  }'
```

### Test the tool-choice rules

Single tool + auto (forced model) -> should rewrite + extract:

```bash
podman run --rm --network devai-net curlimages/curl:latest \
  -X POST http://devai-router:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-R1-Distill-Qwen-7B@32768",
    "messages": [{"role":"user","content":"What time is it?"}],
    "tools": [{"type":"function","function":{"name":"get_time","parameters":{"type":"object","properties":{}}}}],
    "tool_choice": "auto",
    "max_tokens": 4096
  }'
```

Multi-tool + auto (forced model) -> should return HTTP 400 with
`tool_choice_pinning_required` code.

---

## Benchmark harness

Lives at `scripts/bench/`, runs in the lab container, talks to the
router exactly the way Claude Code does. Built to answer "which model
should I use for X?" with evidence rather than vibes.

### What it measures

Per (model, backend) pair, per run:

| Axis | Metric | Source |
|---|---|---|
| Reasoning | GSM8K accuracy | inspect_ai task `gsm8k_subset_<n>` |
| Coding | HumanEval pass@1 | inspect_ai task `humaneval_subset_<n>`, local subprocess sandbox |
| Coding (hardened) | HumanEval+ pass@1 | inspect_ai task `humaneval_plus_subset_<n>` (EvalPlus test set) |
| Knowledge | MMLU-Pro accuracy | inspect_ai task `mmlu_pro_subset_<n>` |
| Hard reasoning | GPQA-Diamond accuracy | inspect_ai task `gpqa_subset_<n>` |
| Tool use | Score + per-subcase breakdown | inspect_ai task `tools_use_<n>` (empty-schema, single-arg, multi-tool pick, result follow-up) |
| Output cleanliness | Leak rate + per-marker hits | regex sweep over response bodies via `bench_latency_leak.py` |
| Cold start | `ttft_ms_first` | First request to a freshly-recreated backend (cold container + weight load + KV alloc + prefill + first token) |
| Steady-state latency | `ttft_ms_steady_p50/p95` | Subsequent prompts in the same model session |
| Throughput | `tps_sustained_p50` | Tokens-per-second during streamed body |
| Memory | `peak_vram_gb`, `mean_vram_gb` | nvidia-smi sampler thread, 1Hz |

### Cache file

`deploy/.bench-cache.json`, schema v3, sorted-keys for diff-
friendliness. Top-level layout:

- `_meta.host_env_history` -- map keyed by 12-char SHA-256 id of
  `(kernel, driver_version, gpu_name, gpu_memory_gb, cuda_version)`,
  values include `captured_at` (ISO-8601 UTC). Same hardware on
  different days -> same id, so the table accumulates one entry per
  distinct environment.
- `_meta.current_host_env_id` -- pointer to the most-recent run's id.
- Bench rows -- keys `<repo>@<sha>::<backend>::<ctx>` for HF
  (vllm/sglang) or `<digest>::<backend>::<ctx>` for Ollama, so the
  same model benched at two contexts lands in two rows. Each row
  stamps `host_env_id`, **and so does each task** -- the row-level id
  describes the run that produced `metrics`, while a task keeps the id
  of the run that actually produced it. The leaderboard's `Env` column
  therefore renders every distinct id present in a row (comma-joined),
  not just the row-level one. The `::<backend>` suffix is mandatory;
  legacy keys without it are migrated in-memory by
  `migrate_bench_cache_keys` on every load.

Iterate with `is_row_key(key)` from `_bench_core.py` so `_meta` (and
any future top-level meta blocks) are skipped. Schema is documented
at the top of `scripts/bench/_bench_core.py::update_row`.

Re-runs merge in (don't overwrite) so partial benches accumulate.
`update_row` is a **pure merge**: it does not clear anything.
`BENCH_FORCE=1` re-runs every task, but only the tasks that actually
completed are overwritten -- a task that errors out leaves its previous
result (and that result's own `host_env_id`) in place. `first_benched_at`
is preserved, `last_benched_at` refreshed, and the current
`host_env_id` stamped on the row and on each re-run task. Without
`--force`, individual tasks are skipped only when an entry with that
exact subset name (e.g. `gsm8k_subset_100`) already exists.

### Make targets

```
make bench           # all backends
make bench-vllm      # only vLLM models
make bench-sglang    # only SGLang models
make bench-ollama    # only Ollama models
make bench-report    # Markdown leaderboard from .bench-cache.json
make test-bench-smoke # 1-model tiny-subset sanity check
```

Knobs (env vars; same idiom as `PROBE_*`):

| Variable | Default | Purpose |
|---|---|---|
| `BENCH_TASKS` | `gsm8k,humaneval,humaneval_plus,mmlu_pro,gpqa,tools,leak` | comma-separated subset |
| `BENCH_REPO` | unset | regex filter on probe-cache top-level key |
| `BENCH_FORCE` | unset | re-run tasks already cached |
| `BENCH_N_GSM8K` | `100` | GSM8K subset size |
| `BENCH_N_HUMANEVAL` | `50` | HumanEval subset size |
| `BENCH_N_MMLU_PRO` | `100` | MMLU-Pro subset size |
| `BENCH_N_GPQA` | `100` | GPQA-Diamond subset size |
| `BENCH_N_TOOLS` | `20` | tools_use prompts (5 per subcase x 4 subcases) |
| `BENCH_N_LEAK_PROMPTS` | `40` | latency/leak sweep prompts |

> **Wall time.** The default task set grew: a plain `make bench` now
> also runs `humaneval_plus` (n=50), `mmlu_pro` (n=100) and `gpqa`
> (n=100) for every (model, backend, ctx), which is materially longer
> than the old four-task sweep. The new total has not been measured.
> `BENCH_TASKS=gsm8k,humaneval,tools,leak` restores the previous cheap
> set.

### Adding a leak marker

Drop a Python regex line into
`scripts/bench/data/leak_markers.txt`. Backslash-escape pipes inside
`<|...|>` tokens so they're treated as literals (e.g.
`<\|new_token\|>`). The sweeper recompiles on every run; no code
changes needed.

### Reading a result

```bash
make bench-report
```

prints a leaderboard sorted by aggregate correctness score. The
top of the report carries a `_Host env_` header with the active
`host_env_id`, kernel, driver, GPU, CUDA version, and capture
timestamp; the per-row `Env` column joins each row to its history
entry so a re-bench after a driver upgrade shows clearly mixed
provenance. For a single model:

```bash
jq '.["nvidia/Qwen3-8B-NVFP4@ccd10a893cbc::vllm"]' deploy/.bench-cache.json
jq '.["_meta"].host_env_history' deploy/.bench-cache.json
```

Per-task `inspect_log_dir` paths point at `.eval` files under
`/var/cache/devai/bench/inspect-logs/` -- load them in the inspect
viewer (`inspect view start --log-dir <path>`) for full per-sample
forensics.
