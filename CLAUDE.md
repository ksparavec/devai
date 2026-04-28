# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **Dev AI Lab** — a containerized development environment for AI experimentation featuring JupyterLab and multiple AI CLIs (Gemini, Claude, OpenAI, Ollama). Built on Debian Trixie with Python 3.13 (uv-managed), Node.js 22 LTS. Two-layer image build for fast iteration. Compatible with Podman and Docker. GPU/CUDA support.

**Active backend:** Ollama (GGUF) only. vLLM and SGLang lifecycle code remains compiled into `gpu-arbiter` and is still covered by `make test-router`, but the compose services are behind a `backends-disabled` profile while Ollama-side behaviour is stabilized — see `docs/sidelined-backends.md` for what's dormant and how to reactivate.

## Build and Run Commands

```bash
# Build
make build-cpu       # Build base + lab image (CPU)
make build-gpu       # Build base + lab image (GPU/CUDA)
make build-router    # Build gpu-arbiter router image
make build           # Build all (CPU + GPU + router)

# Run
make lab-gpu         # Run JupyterLab with GPU
make lab-cpu         # Run JupyterLab CPU-only
make shell-gpu       # Interactive shell (GPU)
make shell-cpu       # Interactive shell (CPU)

# Infrastructure
make cache-up        # Start active services (Ollama, router, Open WebUI). vLLM/SGLang dormant
make cache-down      # Stop all services
make cache-status    # Show status, models, disk usage

# Models — single download path through model-select
make model-select               # Probe + write active-models.yaml
make model-select DOWNLOAD=1    # …also pull missing fitting variants
make model-select FAMILY=qwen3.5 DOWNLOAD=1   # scope to one family
make ollama-list                # List downloaded Ollama models
make vllm-list                  # List on-disk vLLM weights (backend currently dormant)

# Maintenance
make clean           # Remove all images (CPU + GPU + router)
make prune           # Prune dangling images
make help            # Show all targets
```

## Configuration

Copy `.env.example` to `.env` before first use. Key settings:

- `LAB_PORT` — JupyterLab port (default: 8888)
- `WEBUI_PORT` — Open WebUI HTTPS port (default: 8443)
- `CONTAINER_RUNTIME` — `podman` (default) or `docker`
- `HOST_HOME_DIR` — Host home directory for .gitconfig/.ssh mounting
- `HOME_VOLUME` — Persistent home directory path
- `JUPYTER_TOKEN` — Fixed JupyterLab access token
- `HTTP_PROXY`/`HTTPS_PROXY` — Proxy settings for corporate environments

Model catalog is in `deploy/models.yaml` (single source of truth for all models). GPU inference settings:

- `GPU_MEMORY_GB` — Total GPU VRAM in GB (default: 24). Used by router to calculate memory fractions and context limits.
- `MAX_CONTEXT_LEN` — Default max context length in tokens (default: 131072 = 128K). Auto-reduced when KV cache can't fit it. Per-model `context:` overrides are written into `deploy/active-models.yaml` by `select-models.py` from probe data — not hand-edited.

## Architecture

### Lab container (devai-lab-cpu / devai-lab-gpu)

Two-layer image build (base rarely changes, lab layer for fast iteration):

**Layer 1: Dockerfile.base** — System packages, Python 3.13 (via uv), Node.js 22 LTS
**Layer 2: Dockerfile.lab** — CLI binaries (Claude, Codex, Ollama, Gemini, code-server), PyTorch, Python packages, JupyterLab

Build cache: CLI binaries pre-downloaded to `/var/cache/devai/pip/bin/` via `make fetch-cli` (ETag-based updates). Mounted into build — no network downloads during rebuild.

### Inference stack (deploy/docker-compose.yaml)

```
Agent → devai-router:11434 → devai-ollama:11434 (GGUF models)               ← active
Agent → devai-router:11435 → devai-vllm:11434   (NVFP4 models via vLLM)     ← dormant
Agent → devai-router:11436 → devai-sglang:11434 (NVFP4 + HF via SGLang)     ← dormant
```

- **devai-router** — Multi-port GPU-aware reverse proxy. One port per backend. No message inspection — port determines backend. Manages GPU exclusion (only one backend uses GPU at a time), graceful drain on switch, idle timeout, dynamic GPU memory allocation (`--gpu-memory-utilization` for vLLM, `--mem-fraction-static` for SGLang). Context length per request comes from the per-model `context:` in `active-models.yaml`, falling back to `MAX_CONTEXT_LEN`. All vLLM/SGLang lifecycle code stays compiled in (~1070 lines total in `gpu-arbiter/main.go`); only the auxiliary containers are dormant.
- **devai-ollama** — Unmodified `ollama/ollama:latest`. GGUF models, GPU auto-detected. `OLLAMA_MAX_LOADED_MODELS=1` ensures clean model switching. `OLLAMA_CONTEXT_LENGTH` defaults to 262144 (compose env).
- **devai-vllm** *(dormant)* — `vllm/vllm-openai` image. NVFP4 models for Blackwell GPUs. Container lifecycle managed by router via Podman API. Behind `profiles: ["backends-disabled"]` in compose.
- **devai-sglang** *(dormant)* — `lmsysorg/sglang` image. NVFP4 + HuggingFace models, RadixAttention for multi-turn speedup. Same profile, same lifecycle hookup.
- **devai-webui-proxy** — nginx TLS proxy for Open WebUI (mkcert certs or self-signed fallback).
- **devai-open-webui** — Web chat interface, connects to router's ollama port (:11434).

### Supporting services

- **apt-cacher-ng** — APT package cache (port 3142)
- **Registry mirror** — Docker Hub pull-through cache (port 5000)

All services share `devai-net` network. Model data stored under `/var/cache/devai/`.

### SSL / HTTPS

- JupyterLab: auto-detects mkcert certs in `~/.jupyter/ssl/`
- Open WebUI: nginx proxy with mkcert certs or self-signed fallback
- Generate with `mkcert <IP>` on browser workstation, copy to container host

### Model picker (shell + Jupyter)

Interactive model → agent selection via fzf. Used by both `make shell-*` (via `agent-picker`) and JupyterLab launcher cards.

- `scripts/model-picker.py` — Python TUI, two-step fzf picker. Reads `deploy/active-models.yaml` for the run-time active set and `deploy/.ollama-reasoning-cache.json` (digest-keyed, schema v2) for per-tier probe data. Falls back to `deploy/models.yaml` for catalog metadata only. Backend is derived from the chosen model's entry — there's no explicit backend step.
- `scripts/agent-picker.sh` — Shell wrapper, execs model-picker.py.
- `packages/jupyter-ai-launchers/src/index.ts` — JupyterLab extension, each card runs `model-picker --agent <name>`.

**Filter:** the picker shows one row per (family, context tier, reasoning status) bucket. A model is eligible at a tier only when the probe cache contains a measurement at exactly that tier with `fully_on_gpu: true`. There is no interpolation — gaps mean "re-run `make probe-reasoning`". HF entries stay `capability: unknown` because no probe runner exists for vLLM/SGLang yet (see `docs/sidelined-backends.md`); the picker hides them.

### Building the JupyterLab extension

Never build on the host. Build inside a container:

```bash
podman run --rm -it \
  -v "$(pwd)/packages/jupyter-ai-launchers:/src:z" \
  -w /src \
  devai-lab-cpu bash -c "jlpm install && jlpm build:prod"
```

Pre-built output lives in `packages/jupyter-ai-launchers/jupyter_ai_launchers/labextension/` and is copied into the image by Dockerfile.lab. Rebuild after changing `src/index.ts`.

### Key files

```
deploy/models.yaml            — Auto-generated catalog (every variant the upstream catalog declares)
deploy/active-models.yaml     — Auto-generated active subset (downloaded ∩ fits ∩ probed) — what router and picker read
deploy/docker-compose.yaml    — Infrastructure services (vllm/sglang in `backends-disabled` profile)
deploy/Dockerfile.base        — Base image
deploy/Dockerfile.lab         — Lab image
deploy/Dockerfile.router      — Router image (distroless)
gpu-arbiter/main.go           — GPU arbiter source (multi-port proxy, ~1070 lines Go)
scripts/generate-catalog.py   — Refresh deploy/models.yaml from upstream (HF + Ollama registry)
scripts/probe-ollama-reasoning.py — Per-tier probe: capability + measured VRAM at each context tier (schema v2, digest-keyed)
scripts/select-models.py      — Combine catalog + probe + disk → active-models.yaml
scripts/model-picker.py       — Two-step interactive picker (model → agent)
tests/test-router.sh          — Ollama-side integration tests
docs/sidelined-backends.md    — Why vLLM/SGLang are dormant + how to reactivate
```
# AK's CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Don't Assume — If You Don't Know, Say So

**Never claim something works, exists, or behaves a certain way unless you've verified it. If you haven't, say "I don't know" or "I haven't tested that".**

Banned phrases when not verified:
- "this works"
- "verified"
- "should work"
- "this is wired up"
- "X is fine"

Required phrases when uncertain:
- "I don't know"
- "I haven't tested this"
- "I'm assuming X but haven't confirmed"
- "This is unverified"

Before reporting status, audit each claim:
- Did I run it end-to-end? → can claim "works"
- Did I only check it parses / imports / starts? → say "starts cleanly, full round-trip not tested"
- Did I infer from documentation, help text, or another agent's behavior? → say "I'm assuming based on X, not verified"

A "PASS" in a test harness means PASS only for what the harness actually checked. If the harness checks "non-empty output", report that, not "works".

When the user asks "does X work?", the only honest answers are:
- "Yes — verified by [specific test/command/output]"
- "No — fails at [specific point], log/error attached"
- "I don't know — haven't tested"

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, fewer false-confidence claims that have to be retracted later, and clarifying questions come before implementation rather than after mistakes.
