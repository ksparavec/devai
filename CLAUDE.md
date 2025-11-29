# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **Dev AI Lab** - a containerized development environment for AI experimentation featuring JupyterLab and multiple AI CLIs (Gemini, Claude, OpenAI, Ollama). The container is built on Python 3.11-slim with Node.js 20.x, and is optimized for Podman (rootless mode) but compatible with Docker. GPU/CUDA support is available for local model inference.

## Build and Run Commands

```bash
# CPU version
make build          # Build the container image
make run            # Run JupyterLab
make shell          # Interactive shell without JupyterLab

# GPU version (requires NVIDIA Container Toolkit)
make build-gpu      # Build GPU image with CUDA support
make run-gpu        # Run with GPU acceleration

# Cleanup
make clean          # Remove CPU image
make clean-gpu      # Remove GPU image
make help           # Show all targets
```

## Configuration

Copy `.env.example` to `.env` before first use. Key settings:

- `HOST_HOME_DIR` - Host home directory to mount (enables access to .gitconfig, .ssh, etc.)
- `HOST_WORK_DIR` - Working directory mounted to /home/devai/work (default: current dir)
- `CONTAINER_RUNTIME` - `podman` (default) or `docker`
- `PORT` - JupyterLab port (default: 8888)
- `OLLAMA_HOST` - Ollama server URL (default: host machine at port 11434)
- `HTTP_PROXY`/`HTTPS_PROXY` - Proxy settings for corporate environments

To add Python packages, create `requirements.txt` from `requirements.txt.example` and rebuild.

## Architecture

- **Dockerfile** - CPU image with Python, Node.js, Rust, AI CLIs (Gemini, Claude, OpenAI), Ollama client, ChromaDB
- **Dockerfile.gpu** - GPU image based on nvidia/cuda with PyTorch and CUDA support
- **entrypoint.sh** - Handles UID/GID mapping for rootless container operation using `gosu`
- **Makefile** - Build orchestration with Podman/Docker-specific flags and host connectivity for Ollama

The container connects to host services via `host.containers.internal` for Ollama integration.
