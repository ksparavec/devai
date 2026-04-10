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

Model catalog is in `deploy/models.yaml` (single source of truth for all models).

## Architecture

### Lab container (devai-lab-cpu / devai-lab-gpu)

Two-layer image build (base rarely changes, lab layer for fast iteration):

**Layer 1: Dockerfile.base** — System packages, Python 3.13 (via uv), Node.js 22 LTS
**Layer 2: Dockerfile.lab** — CLI binaries (Claude, Codex, Ollama, Gemini, code-server), PyTorch, Python packages, JupyterLab

Build cache: CLI binaries pre-downloaded to `/var/cache/devai/pip/bin/` via `make fetch-cli` (ETag-based updates). Mounted into build — no network downloads during rebuild.

### Inference stack (deploy/docker-compose.yaml)

```
Client → devai-router:11434 [gpu-arbiter, 9 MB Go binary]
              ├─ GGUF model → devai-ollama:11434 (always running, auto load/unload)
              └─ NVFP4 model → devai-vllm:11434 (container auto-created per model)
```

- **devai-router** — Routes by model name (NVFP4 → vLLM, everything else → Ollama). Manages GPU exclusion: unloads Ollama before starting vLLM, stops vLLM on GGUF request or idle timeout. Translates Ollama API ↔ OpenAI API for vLLM. Recreates vLLM container with correct model on switch.
- **devai-ollama** — Unmodified `ollama/ollama:latest`. GGUF models, GPU auto-detected. `OLLAMA_MAX_LOADED_MODELS=1` ensures clean model switching.
- **devai-vllm** — `vllm/vllm-openai` image. NVFP4 models for Blackwell GPUs. Container lifecycle managed by router via Podman API.
- **devai-webui-proxy** — nginx TLS proxy for Open WebUI (mkcert certs or self-signed fallback).
- **devai-open-webui** — Web chat interface, sees all models (Ollama + vLLM) via router.

### Supporting services

- **apt-cacher-ng** — APT package cache (port 3142)
- **Registry mirror** — Docker Hub pull-through cache (port 5000)

All services share `devai-net` network. Model data stored under `/var/cache/devai/`.

### SSL / HTTPS

- JupyterLab: auto-detects mkcert certs in `~/.jupyter/ssl/`
- Open WebUI: nginx proxy with mkcert certs or self-signed fallback
- Generate with `mkcert <IP>` on browser workstation, copy to container host

### Key files

```
deploy/models.yaml            — Model catalog (ollama + vllm models)
deploy/docker-compose.yaml    — Infrastructure services
deploy/Dockerfile.base        — Base image
deploy/Dockerfile.lab         — Lab image
deploy/Dockerfile.router      — Router image (distroless, 9 MB)
gpu-arbiter/main.go           — Router source (~650 lines Go)
tests/test-router.sh          — Integration tests
```
