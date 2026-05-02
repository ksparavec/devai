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
- [Backend lifecycle](#backend-lifecycle)
- [Request rewrite chain](#request-rewrite-chain)
  - [1. Override parsing](#1-override-parsing-namectxreasoning)
  - [2. Reasoning policy](#2-reasoning-policy)
  - [3. Tool-choice promotion](#3-tool-choice-promotion-vllmsglang)
  - [4. Tool stripping](#4-tool-stripping-vllmsglang)
  - [5. Context injection](#5-context-injection-ollama-only)
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
| 11436 | SGLang  | recreated on demand | NVFP4 (broken), BF16 safetensors |

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
              +- poll /health (HEALTH_TIMEOUT_SECONDS, default 600s)
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

`idleWatcher` polls every 30s. A backend whose `lastRequest` is older
than `IDLE_TIMEOUT` (default 300s) gets stopped and replaced with the
`sleep infinity` placeholder. The next request triggers a cold start.

### Backend switch (GPU exclusion)

When a request hits a different backend than the one currently on the
GPU:
1. Wait for the active backend's in-flight requests to drain
   (`DRAIN_TIMEOUT`, default 30s).
2. Ollama: send `keep_alive=0` to all loaded models so it releases
   VRAM. Other backends: stop their container.
3. Recreate the target backend with the new model.

---

## Request rewrite chain

Every non-trivial request goes through this chain in order. Each step
inspects and may mutate the JSON body. Steps are skipped when not
applicable.

### 1. Override parsing (`<name>@<ctx>` / `<name>::<reasoning>`)

The model name in the request body may carry suffixes:

| Suffix form               | Meaning                                          | Strip order |
|---------------------------|--------------------------------------------------|-------------|
| `<name>@<ctx>`            | per-session `--max-model-len` for vLLM/SGLang    | 1 (first)   |
| `<name>::<reasoning>`     | per-request reasoning policy override            | 2           |
| `<name>::nothink`         | shortcut for `enable_thinking=false`             | 2           |
| `<name>::<reasoning>@<ctx>` | both, in either order; `@<ctx>` strips first    | 1 then 2    |

Examples:
- `Qwen3-8B-NVFP4@65536` -> recreate vLLM with `--max-model-len 65536`,
  request body's `model` rewritten to `Qwen3-8B-NVFP4`.
- `qwen3.5:9b-q8_0::nothink` -> set Ollama's `think:false` for this
  request only.
- `gpt-oss-20b::low@131072` -> vLLM gets `--max-model-len 131072`,
  request gets `reasoning_effort: low`.

### 2. Reasoning policy

Driven by the `DEVAI_REASONING` env (default `auto`) plus the
`X-DevAI-Reasoning` header plus the `::<reasoning>` suffix override
(suffix wins over header wins over env). Values: `auto | off | low |
medium | high`.

The action depends on the model's capability (from the probe) and the
backend's protocol path:

| Backend / Path | Capability   | Action                                   |
|----------------|--------------|------------------------------------------|
| Ollama `/api/chat`, `/api/generate` | structured | inject `think: <true\|false>` |
| Ollama `/v1/chat/completions`       | structured | inject `reasoning: {enabled: ...}` |
| Ollama `/v1/messages`               | structured | inject `thinking.type` |
| vLLM `/v1/chat/completions`         | structured | inject `extra_body.chat_template_kwargs.enable_thinking` + `reasoning_effort` |
| SGLang `/v1/chat/completions`       | structured | inject `extra_body.chat_template_kwargs.enable_thinking` + `separate_reasoning` |
| Any                                 | inline + policy=off | log `reasoningDisable` (explicit user opt-out) |
| Any                                 | none / unsupported  | noop |

Models with `disable_verified=False` can't reliably suppress
reasoning -- the directive is sent, but the model may emit reasoning
anyway (R1-Distill family is the standing example).

### 3. Tool-choice promotion (vLLM/SGLang)

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
fall through to step 4 (tool stripping).

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

### 4. Tool stripping (vLLM/SGLang)

Fires when the model has **no** verified `tool_parser` in the cache.
Without an engine-level `--tool-call-parser` flag, vLLM rejects every
request that carries `tool_choice` other than `"none"` with
`BadRequestError: "auto" tool choice requires --enable-auto-tool-choice
and --tool-call-parser`. The router strips `tools` and `tool_choice`
so the request becomes a plain chat. Cost: tool-calling silently
unavailable for that model. Benefit: chat works without backend errors.

Ollama is unaffected -- its protocol negotiates tool support per
request and tolerates `tools=[]` without launch flags.

### 5. Context injection (Ollama only)

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
| `disable_verified`| latest cell with `disable_verified`   | reasoning-disable gate                  |
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
| `VLLM_IMAGE`         | `docker.io/vllm/vllm-openai:latest-cu130-ubuntu2404`       | image to launch               |
| `VLLM_MODELS_DIR`    | `/var/cache/devai/ollama/models/vllm`                      | host path bound to `/models`  |
| `SGLANG_URL`         | `http://devai-sglang:11434`                                | upstream                      |
| `SGLANG_PORT`        | `11436`                                                    | router listen                 |
| `SGLANG_CONTAINER`   | `devai-sglang`                                             | name to recreate              |
| `SGLANG_IMAGE`       | `docker.io/lmsysorg/sglang:v0.5.10.post1-cu130`            | image to launch               |
| `SGLANG_MODELS_DIR`  | `/var/cache/devai/ollama/models/vllm`                      | host path bound to `/models`  |
| `NETWORK`            | `devai-net`                                                | podman network name           |
| `PODMAN_SOCKET`      | `/run/podman/podman.sock`                                  | libpod socket inside router   |

### Lifecycle

| Variable                  | Default | Purpose                                                            |
|---------------------------|---------|--------------------------------------------------------------------|
| `IDLE_TIMEOUT`            | `300`   | seconds before idle backend is replaced with placeholder           |
| `DRAIN_TIMEOUT`           | `30`    | seconds to wait for in-flight requests when switching backends     |
| `HEALTH_TIMEOUT_SECONDS`  | `600`   | cold-start health-poll deadline (NVFP4 + CUDA graph can need 5min) |

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
| `draining <backend> (N active requests)`       | drain-on-switch                          |
| `<backend> idle, stopping`                     | idle timeout fired                       |

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
| Tool use | Score + per-subcase breakdown | inspect_ai task `tools_use_<n>` (empty-schema, single-arg, multi-tool pick, result follow-up) |
| Output cleanliness | Leak rate + per-marker hits | regex sweep over response bodies via `bench_latency_leak.py` |
| Cold start | `ttft_ms_first` | First request to a freshly-recreated backend (cold container + weight load + KV alloc + prefill + first token) |
| Steady-state latency | `ttft_ms_steady_p50/p95` | Subsequent prompts in the same model session |
| Throughput | `tps_sustained_p50` | Tokens-per-second during streamed body |
| Memory | `peak_vram_gb`, `mean_vram_gb` | nvidia-smi sampler thread, 1Hz |

### Cache file

`deploy/.bench-cache.json`, schema-versioned and sorted-keys for
diff-friendliness. Top-level key matches the probe caches -- 
`<repo@sha>` for HF, `<digest>` for Ollama -- so a row joins to its
source probe row by key. Schema is documented at the top of
`scripts/bench/_bench_core.py::update_row`.

Re-runs merge in (don't overwrite) so partial benches accumulate.
`BENCH_FORCE=1` re-runs every task even if cached.

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
| `BENCH_TASKS` | `gsm8k,humaneval,tools,leak` | comma-separated subset |
| `BENCH_REPO` | unset | regex filter on probe-cache top-level key |
| `BENCH_FORCE` | unset | re-run tasks already cached |
| `BENCH_N_GSM8K` | `100` | GSM8K subset size |
| `BENCH_N_HUMANEVAL` | `50` | HumanEval subset size |
| `BENCH_N_TOOLS` | `20` | tools_use prompts (5 per subcase x 4 subcases) |
| `BENCH_N_LEAK_PROMPTS` | `40` | latency/leak sweep prompts |

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

prints a leaderboard sorted by aggregate correctness score. For a
single model:

```bash
jq '.["meta-llama/Llama-3.1-8B@abc123"]' deploy/.bench-cache.json
```

Per-task `inspect_log_dir` paths point at `.eval` files under
`/var/cache/devai/bench/inspect-logs/` -- load them in the inspect
viewer (`inspect view start --log-dir <path>`) for full per-sample
forensics.
