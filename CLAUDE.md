# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **Dev AI Lab** — a containerized development environment for AI experimentation featuring JupyterLab and multiple AI CLIs (Gemini, Claude, OpenAI, Ollama). Built on Debian Trixie with Python 3.13 (uv-managed), Node.js 22 LTS. Two-layer image build for fast iteration. Compatible with Podman and Docker. GPU/CUDA support with automatic GPU arbitration between Ollama (GGUF) and vLLM (NVFP4) backends.

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
make cache-up        # Start all services (Ollama, vLLM, router, Open WebUI)
make cache-down      # Stop all services
make cache-status    # Show status, models, disk usage

# Models
make ollama-list     # List Ollama (GGUF) models with status
make ollama-pull     # Pull all Ollama models from models.yaml
make vllm-list       # List vLLM (NVFP4) models with status
make vllm-pull       # Download all vLLM models from HuggingFace

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
- `MAX_CONTEXT_LEN` — Default max context length in tokens (default: 131072 = 128K). Auto-reduced when KV cache can't fit it. Per-model override via `context` field in models.yaml.

## Architecture

### Lab container (devai-lab-cpu / devai-lab-gpu)

Two-layer image build (base rarely changes, lab layer for fast iteration):

**Layer 1: Dockerfile.base** — System packages, Python 3.13 (via uv), Node.js 22 LTS
**Layer 2: Dockerfile.lab** — CLI binaries (Claude, Codex, Ollama, Gemini, code-server), PyTorch, Python packages, JupyterLab

Build cache: CLI binaries pre-downloaded to `/var/cache/devai/pip/bin/` via `make fetch-cli` (ETag-based updates). Mounted into build — no network downloads during rebuild.

### Inference stack (deploy/docker-compose.yaml)

```
Agent → devai-router:11434 → devai-ollama:11434 (GGUF models)
Agent → devai-router:11435 → devai-vllm:11434   (NVFP4 models via vLLM)
Agent → devai-router:11436 → devai-sglang:11434  (NVFP4 models via SGLang)
```

- **devai-router** — Multi-port GPU-aware reverse proxy (~800 lines Go, 9 MB distroless binary). One port per backend. No message inspection — port determines backend. Manages GPU exclusion: only one backend uses GPU at a time. Graceful drain waits for in-flight requests before stopping a backend. Idle timeout auto-stops backends. Dynamic GPU memory allocation: calculates `--gpu-memory-utilization` (vLLM) and `--mem-fraction-static` (SGLang) from model weight size vs total VRAM, with backend-specific reservations for CUDA graphs and RadixAttention. Context length auto-sized to fit available KV cache (128K default, reduced for tight fits).
- **devai-ollama** — Unmodified `ollama/ollama:latest`. GGUF models, GPU auto-detected. `OLLAMA_MAX_LOADED_MODELS=1` ensures clean model switching.
- **devai-vllm** — `vllm/vllm-openai` image. NVFP4 models for Blackwell GPUs. Container lifecycle managed by router via Podman API.
- **devai-sglang** — `lmsysorg/sglang` image. NVFP4 + HuggingFace models. RadixAttention for multi-turn speedup. Container lifecycle managed by router via Podman API.
- **devai-webui-proxy** — nginx TLS proxy for Open WebUI (mkcert certs or self-signed fallback).
- **devai-open-webui** — Web chat interface, connects to Ollama port (:11434).

### Supporting services

- **apt-cacher-ng** — APT package cache (port 3142)
- **Registry mirror** — Docker Hub pull-through cache (port 5000)

All services share `devai-net` network. Model data stored under `/var/cache/devai/`.

### SSL / HTTPS

- JupyterLab: auto-detects mkcert certs in `~/.jupyter/ssl/`
- Open WebUI: nginx proxy with mkcert certs or self-signed fallback
- Generate with `mkcert <IP>` on browser workstation, copy to container host

### Model picker (shell + Jupyter)

Interactive model → backend → agent selection via fzf. Used by both `make shell-*` (via `agent-picker`) and JupyterLab launcher cards.

- `scripts/model-picker.py` — Python TUI: reads `deploy/models.yaml`, three-step fzf picker
- `scripts/agent-picker.sh` — Shell wrapper, calls model-picker.py
- `packages/jupyter-ai-launchers/src/index.ts` — JupyterLab extension, each card runs `model-picker --agent <name>`

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
deploy/models.yaml            — Model catalog (flat list, each model declares backend: [ollama/vllm/sglang])
deploy/docker-compose.yaml    — Infrastructure services
deploy/Dockerfile.base        — Base image
deploy/Dockerfile.lab         — Lab image
deploy/Dockerfile.router      — Router image (distroless, 9 MB)
gpu-arbiter/main.go           — GPU arbiter source (multi-port proxy, ~720 lines Go)
scripts/model-picker.py       — Interactive model/backend/agent picker (fzf-based)
tests/test-router.sh          — Integration tests
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

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
