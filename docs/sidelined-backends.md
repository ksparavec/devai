# Sidelined backends — vLLM and SGLang

DevAI ships with **three** inference-backend code paths in the GPU arbiter
(`gpu-arbiter/main.go`): Ollama (GGUF), vLLM (NVFP4), and SGLang
(NVFP4 + RadixAttention). At the time of writing only Ollama is wired up
for everyday use. vLLM and SGLang are deliberately dormant while Ollama-side
behaviour is being stabilized — the work to bring them back is small but
unfinished, so the docs here describe the actual state rather than the
intended end state.

## What "dormant" means in practice

| Surface | State |
|---|---|
| `gpu-arbiter/main.go` (router source) | Full vLLM + SGLang lifecycle (`containerRecreate`, `vllmEntrypoint`, `sglangEntrypoint`, port 11435/11436 listeners) is compiled in and passes `make test-router` |
| `deploy/docker-compose.yaml` | `vllm` and `sglang` services are tagged `profiles: ["backends-disabled"]` — `compose up -d` (used by `make cache-up`) skips them |
| `make cache-up` | Starts ollama, router, open-webui, webui-proxy, apt-cache, registry-cache only |
| `scripts/model-picker.py` | Shows Ollama rows only while vLLM/SGLang are dormant. Rows are grouped by family, context tier, and user-facing reasoning/offload label. There is no probe runner for vLLM/SGLang yet, so HF entries stay `capability: unknown` and are hidden. The picker prints a footer line saying how many were hidden and points back here |
| `make test-vllm`, `make test-idle` | Recipes removed from `.PHONY` and from the aggregate `make test`. The shell scripts under `tests/` still exist and can be invoked manually once vLLM is running again |

## Why it was sidelined

A debugging round on Ollama's reasoning behaviour (`docs/ollama_models.md`,
`scripts/probe-ollama-reasoning.py`) needed a small, controllable runtime
surface. Multi-backend lifecycle, container-API plumbing, and per-backend
VRAM math made every reproduction noisy. The decision was to leave the
vLLM/SGLang code intact in `gpu-arbiter` (so the router compiles to the
same binary as before), but stop starting the auxiliary containers and
stop pretending the picker can route to them.

This is a freeze, not a removal. Nothing was deleted from the router or
the picker's `_BACKENDS` dict.

## How to bring them back

In rough order, smallest task first:

1. **Re-enable the compose services.** Drop the profile guard from
   `deploy/docker-compose.yaml` (or pass `--profile backends-disabled`
   to `compose up`). Today the vLLM service still hardcodes a placeholder
   `--model /models/NVIDIA-Nemotron-Nano-9B-v2-NVFP4` in its entrypoint —
   the router's `containerRecreate` path overwrites this at runtime, so
   the compose-side hardcode only matters if you start the service
   directly without going through the router.

2. **Add a probe runner for vLLM/SGLang.** `scripts/probe-ollama-reasoning.py`
   speaks `/api/chat` + `/api/ps` (Ollama-specific). The vLLM/SGLang
   equivalent uses `/v1/chat/completions` plus `nvidia-smi` (or vLLM's
   prometheus metrics) for VRAM. Output must conform to schema v2: one
   digest-keyed entry per set of weights, with a `probes` map keyed by
   context tier and per-tier `{actual_total_gb, actual_vram_gb,
   fully_on_gpu, capability}`. No interpolation — each tier is measured
   directly. Until such a runner exists, every HF entry stays
   `capability: unknown` and the picker hides them.

3. **Rerun selection.** Once probes succeed, `make model-select` reads
   the v2 cache and writes one row per (family, effective_context,
   capability) bucket into `deploy/active-models.yaml`, alias names
   collapsed under `aliases:`. The picker filter starts admitting the
   new backend's rows and the existing router path serves them.

4. **Optional: relax the picker filter** if you want to expose
   not-yet-probed vLLM/SGLang models in the UI before step 2 lands.
   The check in `scripts/model-picker.py:_build_menu` is one
   conditional; relaxing it loses the "only show models we've actually
   measured" guarantee.

## Quick reactivation for one-off testing

```bash
# bring vllm and sglang up alongside the regular stack
podman compose -f deploy/docker-compose.yaml --profile backends-disabled up -d vllm sglang

# the router was already started by `make cache-up`; it auto-detects the
# now-running services on its existing 11435/11436 listeners. requests to
# those ports drive containerRecreate as usual.
```

This is enough to exercise the lifecycle code interactively without
unwinding the freeze.
