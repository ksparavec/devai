# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **Dev AI Lab** - a containerized development environment for AI experimentation featuring JupyterLab and multiple AI CLIs (Gemini, Claude, OpenAI, Ollama). Built on Debian Trixie with Python 3.13 (apt), Node.js 22 LTS, and uv. Two-layer image build for fast iteration. Compatible with Podman and Docker. GPU/CUDA support available for local model inference.

## Build and Run Commands

```bash
# First time: build both layers
make build          # Build base + lab image (CPU)
make build-gpu      # Build base + lab image (GPU/CUDA)

# Fast iteration: rebuild lab layer only (skips base)
make rebuild        # Rebuild lab image (CPU)
make rebuild-gpu    # Rebuild lab image (GPU)

# Run
make run            # Run JupyterLab
make run-gpu        # Run with GPU acceleration
make shell          # Interactive shell without JupyterLab

# Standalone launcher (GPU, auto-detects free port, names container after git repo)
make install        # Install devai.sh + ollama.sh to ~/.local/bin
devai.sh            # Run from any git repo

# Infrastructure (Ollama + Open WebUI + caches)
make cache-up       # Start all infrastructure services
make cache-down     # Stop all infrastructure services
make ollama-pull MODEL=llama3.2  # Pull a model
make ollama-list    # List downloaded models
make ollama-unload  # Unload all models from GPU VRAM
make install-systemd  # Enable auto-start at boot

# Cleanup
make clean          # Remove lab image
make clean-base     # Remove base image
make clean-home     # Remove persistent home volume
make clean-all      # Remove all (images + home volume)
make help           # Show all targets
```

## Configuration

Copy `.env.example` to `.env` before first use. Key settings:

- `HOST_HOME_DIR` - Host home directory — enables .gitconfig and .ssh in container
- `HOST_WORK_DIR` - Working directory mounted to /home/devai/work (default: current dir)
- `HOME_VOLUME` - Named volume for persistent /home/devai (default: devai-lab-home)
- `CONTAINER_RUNTIME` - `podman` (default) or `docker`
- `PORT` - JupyterLab port (default: 8888)
- `OLLAMA_HOST` - Ollama server URL (default: containerized at devai-ollama:11434)
- `OLLAMA_DEFAULT_MODEL` - Default model for interactive chat (default: llama3.2)
- `JUPYTER_TOKEN` - Fixed JupyterLab access token (default: devai)
- `HTTP_PROXY`/`HTTPS_PROXY` - Proxy settings for corporate environments

To add Python packages, create `requirements.txt` from `requirements.txt.example` and rebuild.

## Architecture

Two-layer image build (base rarely changes, lab layer for fast iteration):

### Layer 1: Dockerfile.base (devai-base)
- **apt-get**: Debian Trixie full, Python 3.13, compilers (build-essential, rustc, cargo), system utilities (curl, git, gosu, vim, nvtop, zstd)
- **uv**: Python package manager (installed via astral.sh)
- **Node.js 22 LTS**: Installed from official binary tarball

### Layer 2: Dockerfile (devai-lab)
- **Binary installs** (from local cache, populated by `make fetch`):
  - **Claude Code**: Binary from official distribution (claude.ai)
  - **OpenAI Codex**: Prebuilt binary from GitHub releases
  - **Ollama CLI**: Client binary (connects to containerized Ollama server)
  - **code-server**: VS Code in the browser (with jupyter-vscode-proxy integration)
  - **Google Gemini CLI**: Pre-installed npm package
- **PyTorch**: CPU-only or CUDA (controlled by GPU_BUILD arg)
- **.default-python-packages**: jupyterlab, openai, ollama, chromadb, llm, jupyter-server-proxy, jupyter-vscode-proxy, ML/data science stack
- **jupyter-ai-launchers**: JupyterLab extension adding Claude, Codex, Gemini, Ollama to launcher (pre-built)
- **requirements.txt**: Optional project-specific Python packages
- **entrypoint.sh**: Copies host config (.gitconfig, .ssh) from staging mount; auto-detects SSL certs (mkcert); for Docker, remaps UID/GID and uses `gosu` to drop privileges

### Build cache (`make fetch`)
All external binaries are pre-downloaded to `/var/cache/devai/pip/bin/` and mounted into the build. No network downloads during `make rebuild`. The `fetch` target is a dependency of all build targets.

### Runtime volumes
- **Named volume** (`HOME_VOLUME`): Persistent `/home/devai` — survives container restarts
- **Bind mount**: `HOST_WORK_DIR` → `/home/devai/work`
- **Staging mount**: Host `.gitconfig`/`.ssh` → `/tmp/host-config/` (copied into home by entrypoint)

### User identity
- **Podman (rootless)**: Runs as container root (= host user). No user switching needed.
- **Docker**: Entrypoint remaps `devai` user to host UID/GID via `gosu`. Handles UID conflicts in GPU base images.

### Infrastructure services (docker-compose.cache.yml)
- **apt-cacher-ng**: APT package cache (port 3142)
- **Registry mirror**: Docker Hub pull-through cache (port 5000)
- **Ollama**: Local LLM server with GPU support (port 11434, `/var/cache/devai/ollama` bind mount)
- **Open WebUI**: Web chat interface for Ollama (port 3000, `/var/cache/devai/open-webui` bind mount)
- All services share `devai-net` network; agent containers join the same network
- All data stored on LVM thin-provisioned volumes under `/var/cache/devai/`

Managed via `make cache-up`/`cache-down` or systemd (`make install-systemd`).
Installed to `~/.config/devai/` (independent of repo location).

### SSL / HTTPS
- Entrypoint auto-detects mkcert certificates in `~/.jupyter/ssl/`
- Looks for `<HOST_IP>.pem` or `cert.pem`
- HTTPS enables secure context for code-server webviews (Continue, Cline extensions)
- Generate with `mkcert <IP>` on the browser workstation, copy to container host

### Scripts
- **devai.sh**: Standalone GPU launcher — auto-detects free port, names container after git repo, joins devai-net, passes OLLAMA_HOST/JUPYTER_TOKEN
- **ollama.sh**: Model and GPU management — pull/list/rm/load/unload/gpu/top
- **ollama-chat.sh**: Wrapper for JupyterLab launcher — runs `interpreter` with default model
- **ollama-bench.py**: Quick model benchmarking with timing stats
