# Inference backends — Ollama, vLLM, SGLang

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
Run `make verify-backend-flags` after bumping either image — it dumps
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

1. Stops the other GPU-using backends — `devai-ollama` is unloaded via
   `/api/generate` with `keep_alive=0`; vLLM/SGLang containers are
   stopped via libpod.
2. Removes the placeholder `devai-vllm` / `devai-sglang` container.
3. Recreates it via libpod with a dynamic entrypoint that bakes in the
   chosen model path, `--max-model-len` / `--context-length`, and
   `--gpu-memory-utilization` / `--mem-fraction-static` derived from the
   probe cache and `MAX_CONTEXT_LEN`. If the model has probe-verified
   `reasoning_parser` and/or `tool_parser`, the router injects
   `--reasoning-parser <value>` (vLLM also adds `--enable-auto-tool-choice`)
   and `--tool-call-parser <value>`.
4. Polls `/health` until the container becomes ready (default 600s for
   NVFP4 cold-start with CUDA graph compilation, override via
   `HEALTH_TIMEOUT_SECONDS` env on the router).
5. Applies the reasoning policy and tool-stripping rules to the request.
6. Proxies the original request through.

A second request that switches the **model**, **context cap**, or **reasoning
override** triggers another recreate. The router tracks `currentModel`,
`currentContext`, and `currentReasoningOverride` per backend; any change → recreate.

## Probing — building the cache

Each backend has its own probe cache:

- `deploy/.ollama-reasoning-cache.json` — schema v3, digest-keyed.
- `deploy/.vllm-reasoning-cache.json` — schema v2, repo+sha-keyed.
- `deploy/.sglang-reasoning-cache.json` — schema v2, repo+sha-keyed.

Schema v2 (vLLM/SGLang) added three top-level fields per entry:

- `reasoning_parser` — backend startup flag value (e.g. `qwen3`) that
  produced a `structured` round-trip in Probe A. Null when the curated
  family hint did not pan out, when no hint was supplied, or when the
  model's capability is `inline`/`unsupported`.
- `tool_parser` — backend startup flag value (e.g. `hermes`, `qwen25`)
  that produced a parseable tool call in Probe B. Null when no curated
  hint, or when the round-trip failed.
- `disable_verified` — true iff Probe C suppressed `reasoning_content`
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
make probe                                          # all bands × all contexts
make probe PROBE_VRAMS=24G PROBE_CONTEXTS=32K      # one band, one tier
make probe PROBE_FORCE=1                           # re-probe everything

# 2. HF probing (vLLM/SGLang) — requires exclusive GPU access
#    Stop the live router and ollama first.
make cache-down

# 3. For each HF backend, launch a probe container and run all cells.
#    For each (model, vram_band, ctx_tier) cell:
#      A) fit + reasoning      — classify capability, snapshot nvidia-smi
#      B) tool-call            — only when parsers.<backend>.tool is set
#      C) disable verification — only when Probe A produced `structured`;
#                                verifies suppression of `reasoning_content`
#    Each cell takes 1–3 minutes; extra probes add a few seconds each.
make probe-vllm                                     # all vLLM models, all cells
make probe-sglang                                   # all SGLang models, all cells
make probe-vllm PROBE_REPO=Llama                   # filter to matching models
make probe-sglang PROBE_CONTEXTS=128K              # single context tier

# 4. Restart the stack — router reloads all three caches at boot.
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
in the cache when the corresponding round-trip succeeds — a curated
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

## Cache hygiene

### vLLM / SGLang — re-probe when sha changes

The HF probe cache is keyed on `<repo>@<sha>` where `sha` is the first
12 chars of HuggingFace's commit SHA at generation time
(`make catalog-regen`). When upstream rebases the repo:

1. `make catalog-regen` produces a new sha.
2. The old cache entry is now an orphan (different key) and ignored.
3. Run `make probe-vllm` / `make probe-sglang` to populate the new key.

Old entries persist until manually pruned. The router synthesizes only
from the latest catalog sha — old entries don't affect serving.

### Ollama — re-probe when digest changes

Ollama models are identified by their manifest digest. Pulling a new
quantization or alias changes the digest; the prober keys on digest
directly, so re-running `make probe` populates new entries.

## Failure-mode taxonomy

When a probe records `fits: false`, `evidence.kind` tells you why:

| `kind` | Meaning | Action |
|---|---|---|
| `arch` | Model's architecture (`config.json`) is rejected by the backend's runtime. Custom-code archs (e.g. `auto_map`-only models) hit this. | Wait for upstream support, or pick a different model. The probe records `capability: "unsupported_arch"` and the picker hides the row permanently. |
| `quant` | Quantization scheme (FP8/GPTQ/AWQ) not supported on this hardware. | Pick a different quant of the same model. |
| `oom_startup` | Container failed during model load — weights + KV at requested ctx exceed the GPU memory budget. | Reduce `MAX_CONTEXT_LEN`, pick a smaller quant, or run on a larger GPU. |
| `oom_chat` | Container started but failed on the first chat round-trip — typically CUDA-graph capture OOM. | Same as `oom_startup`; the budget is too tight for the model + ctx. |
| `clamped_ctx` | Backend silently capped `actual_max_model_len` below the requested ctx — typically a model with a hard architectural ceiling lower than the operator-requested tier. | Lower the requested ctx tier or accept the cap. |
| `infra` | Container failed for non-model reasons — image missing nvcc, network error, tokenizer download stall, podman issue. The log excerpt usually shows what. | Fix the environment; this is not a model-fitness signal. |
| `implied_spill` | Larger ctx tier filled in by the prober without launching — set when a smaller ctx at the same VRAM band already failed. | Skip; smaller ctx fit is the actionable upper bound. |

The `evidence.matched_pattern` field (when present) names the substring
that triggered the classification — useful for auditing why a particular
launch was tagged a particular way.

## Coordination — only one backend at a time

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

Both flows are transparent to the agent CLI — the picker emits the right serving name and the router handles parsing and lifecycle management.
