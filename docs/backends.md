# Inference backends -- Ollama, vLLM, SGLang

DevAI exposes three inference backends behind a single multi-port router
(`gpu-arbiter`). Each backend serves a distinct port; the router enforces
GPU mutual exclusion (only one backend uses the GPU at a time), manages
the per-request context cap, applies the reasoning policy, and emits the
correct backend startup flags (`--reasoning-parser`, `--tool-call-parser`)
when the probe cache has confirmed values for the model.

| Backend | Port | Image | Models |
|---|---|---|---|
| Ollama  | 11434 | `ollama/ollama:latest` | GGUF (Q3/Q4/Q5/Q8/etc.) |
| vLLM    | 11435 | `vllm/vllm-openai:latest-cu130-ubuntu2404` | NVFP4, FP8, BF16/FP16 safetensors |
| SGLang  | 11436 | `lmsysorg/sglang:v0.5.10.post1-cu130` | NVFP4, FP8, BF16/FP16 safetensors (RadixAttention multi-turn) |

Backend launch-flag *names* are pinned in `deploy/backend-flags.yaml`.
Run `make verify-backend-flags` after bumping either image -- it dumps
`--help` from the pinned image and asserts every named flag is present.

All three are reachable via the router from inside the `devai-net` Podman
network. Agents (Claude Code, Aider, Codex, LATE, Open WebUI) talk to the
router on the appropriate port; the picker emits the right port based on
the chosen model's backend.

## Lifecycle

`make cache-up` brings up:

- **Always running**: `devai-ollama`, `devai-router`, `devai-open-webui`,
  `devai-webui-proxy`, `devai-apt-cache`, `devai-registry-cache`,
  `devai-logger`.
- **Idle placeholders**: `devai-vllm` and `devai-sglang` start with
  `entrypoint: ["sleep", "infinity"]`. They hold the service definition
  (image, mounts, GPU device, network) but consume no GPU resources.

When the first request hits port 11435 (vLLM) or 11436 (SGLang), the
router:

1. Stops the other GPU-using backends -- `devai-ollama` is unloaded via
   `/api/generate` with `keep_alive=0`; vLLM/SGLang containers are
   stopped via libpod.
2. Removes the placeholder `devai-vllm` / `devai-sglang` container.
3. Recreates it via libpod with a dynamic entrypoint that bakes in the
   chosen model path, `--max-model-len` / `--context-length`, and
   `--gpu-memory-utilization` / `--mem-fraction-static` derived from the
   probe cache and `MAX_CONTEXT_LEN`. For vLLM the entrypoint also
   always passes `--kv-cache-dtype fp8` -- fp16 KV at 128K would push
   NVFP4 checkpoints past 24 GiB on the reference card. If the model
   has probe-verified `reasoning_parser` and/or `tool_parser`, the
   router injects `--reasoning-parser <value>` (vLLM also adds
   `--enable-auto-tool-choice`) and `--tool-call-parser <value>`.
   Finally, any `engine_flags` / `engine_env` from
   `deploy/recovery-flags.json` keyed by the canonical model name are
   appended (e.g. `--enforce-eager` for Nemotron-3-Nano at 128K --
   see docs/router.md "Per-model recovery flags").
4. Polls `/health` until the container becomes ready (default 600s for
   NVFP4 cold-start with CUDA graph compilation, override via
   `HEALTH_TIMEOUT_SECONDS` env on the router).
5. Applies the reasoning policy and tool-stripping rules to the request.
6. Proxies the original request through.

A second request that switches the **model**, **context cap**, or **reasoning
override** triggers another recreate. The router tracks `currentModel`,
`currentContext`, and `currentReasoningOverride` per backend; any change -> recreate.

## Probing -- building the cache

Each backend has its own probe cache:

- `deploy/.ollama-reasoning-cache.json` -- schema v3, digest-keyed.
- `deploy/.vllm-reasoning-cache.json` -- schema v2, repo+sha-keyed.
- `deploy/.sglang-reasoning-cache.json` -- schema v2, repo+sha-keyed.

Schema v2 (vLLM/SGLang) added three top-level fields per entry:

- `reasoning_parser` -- backend startup flag value (e.g. `qwen3`) that
  produced a `structured` round-trip in Probe A. Null when the curated
  family hint did not pan out, when no hint was supplied, or when the
  model's capability is `inline`/`unsupported`.
- `tool_parser` -- backend startup flag value (e.g. `hermes`, `qwen25`)
  that produced a parseable tool call in Probe B. Null when no curated
  hint, or when the round-trip failed.
- `disable_verified` -- true iff Probe C suppressed `reasoning_content`
  on a structured-capable model. Mirrors Ollama's `disable_verified`;
  gates the router's "off" rewrite.

The picker hides any model that lacks a `fits=true` probe at the host
VRAM band. The router synthesizes `/v1/models` rows from these caches.
Without probes, models are invisible.

### Procedure

```bash
# 1. Ollama probing (Make-orchestrated, runs live with Ollama container)
#    Each PROBE_VRAMS band recreates devai-ollama with OLLAMA_GPU_OVERHEAD
#    set so the daemon behaves as if it had only that VRAM available.
make probe                                          # all bands x all contexts
make probe PROBE_VRAMS=24G PROBE_CONTEXTS=32K      # one band, one tier
make probe PROBE_FORCE=1                           # re-probe everything

# 2. HF probing (vLLM/SGLang) -- requires exclusive GPU access
#    Stop the live router and ollama first.
make cache-down

# 3. For each HF backend, launch a probe container and run all cells.
#    For each (model, vram_band, ctx_tier) cell:
#      A) fit + reasoning      -- classify capability, snapshot nvidia-smi
#      B) tool-call            -- only when parsers.<backend>.tool is set
#      C) disable verification -- only when Probe A produced `structured`;
#                                verifies suppression of `reasoning_content`
#    Each cell takes 1-3 minutes; extra probes add a few seconds each.
make probe-vllm                                     # all vLLM models, all cells
make probe-sglang                                   # all SGLang models, all cells
make probe-vllm PROBE_REPO=Llama                   # filter to matching models
make probe-sglang PROBE_CONTEXTS=128K              # single context tier

# 4. Restart the stack -- router reloads all three caches at boot.
make cache-up
```

### Curating parser hints

Reasoning and tool-call parsers are per-architecture. The curated
choices live in `scripts/model-families.yaml` under each family's
`parsers:` block:

```yaml
- name: qwen3.5
  ...
  parsers:
    vllm:
      reasoning: qwen3
      tool: hermes
    sglang:
      reasoning: qwen3
      tool: qwen25
```

`make catalog-regen` propagates these into per-row `parsers:` blocks
in `deploy/models.yaml`. The probers read the row's block and pass
`--reasoning-parser` / `--tool-call-parser` (vLLM also adds
`--enable-auto-tool-choice`) to the launch. A field is only confirmed
in the cache when the corresponding round-trip succeeds -- a curated
hint that the model doesn't actually honour produces a null cache
entry, and the router launches without the flag.

### Probe knobs

| Env / Make var | Effect |
|---|---|
| `PROBE_VRAMS=16G,24G` | Ollama target bands |
| `PROBE_VRAMS_VLLM=24G` | vLLM target bands |
| `PROBE_VRAMS_SGLANG=24G` | SGLang target bands |
| `PROBE_CONTEXTS=32K,64K,128K,256K` | Context tiers (all backends) |
| `PROBE_REPO=Llama-3.1-8B` | Regex filter on catalog rows (HF probers only) |
| `PROBE_FORCE=1` | Re-probe every cell even if cached |
| `PROBE_FORCE_ARCH=1` | Re-probe top-level capability/arch fields |

### Custom vLLM parser plugins

Some models emit tool calls or reasoning in a format that no built-in
vLLM parser handles. The DeepSeek R1 distills are the standing
example: their chat template uses DeepSeek-V3 boundary markers
(`<|tool_call_begin|>` etc.), but they inherit the Qwen2 / Llama-3
tokenizer where those markers aren't atomic vocab entries. vLLM's
built-in `deepseek_v3` / `_v31` / `_v32` parsers do
`vocab.get(<token>)` at startup and crash with HTTP 500 on every
tool-using request.

The fix is a parser plugin: a Python file that registers a parser
with vLLM's `ToolParserManager`, loaded via the
`--tool-parser-plugin <abs-path>` flag. DevAI handles the wiring so
adding a new plugin is a two-step change:

1. Drop the parser file in `scripts/vllm_plugins/`.
2. Add one entry in `deploy/vllm-plugins.json`:

   ```json
   {
     "plugins": {
       "<parser_name>": {
         "kind": "tool",          // or "reasoning"
         "file": "<basename>.py"
       }
     }
   }
   ```

3. Reference `<parser_name>` from a family's `parsers.vllm.tool` (or
   `parsers.vllm.reasoning`) in `scripts/model-families.yaml`, then
   `make catalog-regen` and re-probe.

Both the prober (`scripts/_probe_hf_common.py`) and the router
(`gpu-arbiter/main.go`) read the registry. When a parser name resolves
to a plugin entry they:

- bind-mount `scripts/vllm_plugins/` into the launched vLLM container
  at the registry's `container_dir` (default `/etc/devai/vllm-plugins`);
- emit `--tool-parser-plugin <abs>` (or `--reasoning-parser-plugin`)
  *before* the matching `--tool-call-parser <name>` flag -- vLLM
  resolves parser names at flag-parse time, so the plugin module has
  to be loaded by then.

Names absent from the registry pass through as built-in vLLM parsers
(no plugin flag, no bind-mount). The behaviour for built-ins is
identical to pre-plugin builds.

The router learns the host path of the plugin directory via
`VLLM_PLUGINS_HOST_DIR` (set by the Makefile to
`$(abspath scripts/vllm_plugins)` and exported into compose). When
that env is empty and a model still resolves to a plugin, the router
fails the recreate with an actionable error rather than launching
without the plugin file accessible.

SGLang has no equivalent: SGLang's plugin model is Python-import based
(register a class via SGLang's detector framework), not file-path
based, so a separate plugin would be needed per model. SGLang traffic
for plugin-only families runs without tool support until that lands.

#### Operational notes -- R1-Distill family

Both R1 distills share the same chat template and the same plugin, but
their tool-calling **behaviour** differs sharply because of base-model
training. Verified end-to-end through the router with `tool_choice:
"auto"` and one tool:

| Model | Base tokenizer | `tool_mode` | Completion tokens to call | Reasoning preamble |
|---|---|---|---|---|
| `DeepSeek-R1-Distill-Llama-8B` | Llama-3 | forced | **5** | none -- calls immediately |
| `DeepSeek-R1-Distill-Qwen-7B` | Qwen-2  | forced | ~525 | yes -- long CoT before the call |

Both ended up `tool_mode=forced` (the auto-choice probe didn't elicit
a call), so the router's promote rule kicks in for either. The
**Llama-8B distill is much more usable for tool-calling agents** -- 
it's effectively non-reasoning when handed a single tool. Prefer it
over the Qwen-7B distill when latency matters and the use case
doesn't need reasoning depth. The Qwen-7B distill is better when you
want explicit chain-of-thought, but agents must budget for ~500-token
preamble per tool call.

For multi-tool use cases on either distill, the router rejects with
HTTP 400 (`tool_choice_pinning_required`). Pin client-side or route
to a `tool_mode=auto` model (Qwen3.5-9B-Q8, Qwen3-8B-NVFP4,
Llama-3.1-8B-Instruct-NVFP4) that handles auto choice reliably.

## Cache hygiene

### vLLM / SGLang -- re-probe when sha changes

The HF probe cache is keyed on `<repo>@<sha>` where `sha` is the first
12 chars of HuggingFace's commit SHA at generation time
(`make catalog-regen`). When upstream rebases the repo:

1. `make catalog-regen` produces a new sha.
2. The old cache entry is now an orphan (different key) and ignored.
3. Run `make probe-vllm` / `make probe-sglang` to populate the new key.

Old entries persist until manually pruned. The router synthesizes only
from the latest catalog sha -- old entries don't affect serving.

### Ollama -- re-probe when digest changes

Ollama models are identified by their manifest digest. Pulling a new
quantization or alias changes the digest; the prober keys on digest
directly, so re-running `make probe` populates new entries.

## Failure-mode taxonomy

When a probe records `fits: false`, `evidence.kind` tells you why:

| `kind` | Meaning | Action |
|---|---|---|
| `arch` | Model's architecture (`config.json`) is rejected by the backend's runtime. Custom-code archs (e.g. `auto_map`-only models) hit this. | Wait for upstream support, or pick a different model. The probe records `capability: "unsupported_arch"` and the picker hides the row permanently. |
| `quant` | Quantization scheme (FP8/GPTQ/AWQ) not supported on this hardware. | Pick a different quant of the same model. |
| `oom_startup` | Container failed during model load -- weights + KV at requested ctx exceed the GPU memory budget. | Reduce `MAX_CONTEXT_LEN`, pick a smaller quant, or run on a larger GPU. |
| `oom_chat` | Container started but failed on the first chat round-trip -- typically CUDA-graph capture OOM. | Same as `oom_startup`; the budget is too tight for the model + ctx. |
| `clamped_ctx` | Backend silently capped `actual_max_model_len` below the requested ctx -- typically a model with a hard architectural ceiling lower than the operator-requested tier. | Lower the requested ctx tier or accept the cap. |
| `infra` | Container failed for non-model reasons -- image missing nvcc, network error, tokenizer download stall, podman issue. The log excerpt usually shows what. | Fix the environment; this is not a model-fitness signal. |
| `implied_spill` | Larger ctx tier filled in by the prober without launching -- set when a smaller ctx at the same VRAM band already failed. | Skip; smaller ctx fit is the actionable upper bound. |

The `evidence.matched_pattern` field (when present) names the substring
that triggered the classification -- useful for auditing why a particular
launch was tagged a particular way.

## Coordination -- only one backend at a time

The router serializes GPU access via `stopOtherBackends`. A request for
backend X drains and stops all other GPU-using backends before X starts.
Concurrent vLLM + SGLang on the same GPU is not supported.

For probing, this constraint is enforced by the prober itself: each
probe driver refuses to run if `devai-router`, `devai-vllm`, or
`devai-sglang` is up. Always `make cache-down` before probing.

## Per-session context binding & reasoning overrides

When the picker selects a model + context tier (+ reasoning override), each backend handles the binding differently:

- **Ollama**: the picker emits the parent name (e.g., `qwen3.5:9b-q8_0`), or with a reasoning override suffix (e.g., `qwen3.5:9b-q8_0::nothink` to suppress thinking even if the model supports it). KV cache is allocated dynamically per request from the `OLLAMA_CONTEXT_LENGTH` global ceiling (default 256K). The `/api/chat` and `/api/generate` endpoints honour `options.num_ctx` injected by the router; the OpenAI- and Anthropic-compat layers ignore it and use the global ceiling.
- **vLLM / SGLang**: the picker emits `<name>@<ctx>` (e.g., `Llama-3.1-8B-Instruct-NVFP4@32768`), or with a reasoning override prefix (e.g., `Llama-3.1-8B-Instruct-NVFP4::low@32768` to set reasoning effort to `low`). The router's `parseCtxOverride` strips `@<ctx>` first, then `parseReasoningOverride` strips `::<reasoning>`, rewrites the body's `model` field to the clean name, applies the reasoning policy, and triggers a recreate when `currentContext` or `currentReasoningOverride` differs. No client-side tag materialization needed.

Valid reasoning suffixes: `::off`, `::auto`, `::low`, `::medium`, `::high`, `::nothink` (Ollama only; suppresses thinking).

Both flows are transparent to the agent CLI -- the picker emits the right serving name and the router handles parsing and lifecycle management.
