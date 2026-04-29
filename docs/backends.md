# Inference backends — Ollama, vLLM, SGLang

DevAI exposes three inference backends behind a single multi-port router
(`gpu-arbiter`). Each backend serves a distinct port; the router enforces
GPU mutual exclusion (only one backend uses the GPU at a time), manages
the per-request context cap, and applies the reasoning policy.

| Backend | Port | Image | Models |
|---|---|---|---|
| Ollama  | 11434 | `ollama/ollama:latest` | GGUF (Q3/Q4/Q5/Q8/etc.) |
| vLLM    | 11435 | `vllm/vllm-openai:latest-cu130-ubuntu2404` | NVFP4, FP8, BF16/FP16 safetensors |
| SGLang  | 11436 | `lmsysorg/sglang:latest` | NVFP4, FP8, BF16/FP16 safetensors (RadixAttention multi-turn) |

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
   probe cache and `MAX_CONTEXT_LEN`.
4. Polls `/health` until the container becomes ready (default 300s,
   override via `HEALTH_TIMEOUT_SECONDS` env on the router).
5. Proxies the original request through.

A second request that switches the **model** or the **context cap**
triggers another recreate. The router tracks `currentModel` and
`currentContext` per backend; any change → recreate.

## Probing — building the cache

Each backend has its own probe cache:

- `deploy/.ollama-reasoning-cache.json` — schema v3, digest-keyed.
- `deploy/.vllm-reasoning-cache.json` — schema v1, repo+sha-keyed.
- `deploy/.sglang-reasoning-cache.json` — schema v1, repo+sha-keyed.

The picker hides any model that lacks a `fits=true` probe at the host
VRAM band. The router synthesizes `/v1/models` rows from these caches.
Without probes, models are invisible.

### Procedure

```bash
# 1. Stop the live router so the prober has exclusive GPU access.
make cache-down

# 2. Run the prober for one backend at a time. Each run iterates the
#    catalog rows of source: hf for that backend, launches the
#    container, polls /health, sends a chat probe, snapshots
#    nvidia-smi, classifies fit, then tears the container down.
#    A full sweep takes ~1-3 minutes per (model, vram_band, ctx_tier).
make probe                                          # Ollama (Make-orchestrated bands)
make probe-vllm                                     # vLLM
make probe-sglang                                   # SGLang

# 3. Restart the stack — router reloads all three caches at boot.
make cache-up
```

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

## Per-session context binding

When the picker selects a model + context tier, each backend handles
the context binding differently:

- **Ollama**: the picker materializes a `<parent>-ctx<N>` Modelfile
  derived tag with `PARAMETER num_ctx` baked in. Required because
  Ollama 0.21+ honours `options.num_ctx` only on its native `/api/chat`
  path; the OpenAI- and Anthropic-compat layers ignore it.
- **vLLM / SGLang**: the picker emits `<name>@<ctx>` (e.g.
  `Llama-3.1-8B-Instruct-NVFP4@32768`). The router parses the suffix
  via `parseCtxOverride`, rewrites the body's `model` field to the
  clean name, and triggers a recreate when `currentContext` differs.
  No client-side tag materialization needed.

Both flows are transparent to the agent CLI — the picker emits the
right serving name and the router handles the rest.
