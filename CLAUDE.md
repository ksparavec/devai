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
make install        # Install devai.sh to ~/.local/bin
devai.sh            # Run from any git repo

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
- `OLLAMA_HOST` - Ollama server URL (default: host machine at port 11434)
- `HTTP_PROXY`/`HTTPS_PROXY` - Proxy settings for corporate environments

To add Python packages, create `requirements.txt` from `requirements.txt.example` and rebuild.

## Architecture

Two-layer image build (base rarely changes, lab layer for fast iteration):

### Layer 1: Dockerfile.base (devai-base)
- **apt-get**: Debian Trixie full, Python 3.13, compilers (build-essential, rustc, cargo), system utilities (curl, git, gosu)
- **uv**: Python package manager (installed via astral.sh)
- **Node.js 22 LTS**: Installed from official binary tarball

### Layer 2: Dockerfile (devai-lab)
- **PyTorch**: CPU-only or CUDA (controlled by GPU_BUILD arg)
- **.default-python-packages**: jupyterlab, openai, ollama, chromadb, ML/data science stack
- **Claude Code**: Binary from official distribution (claude.ai)
- **OpenAI Codex**: Prebuilt binary from GitHub releases + bubblewrap sandbox
- **.default-npm-packages**: @google/gemini-cli (npm, no native installer available)
- **jupyter-ai-launchers**: JupyterLab extension adding Claude, Codex, Gemini to launcher
- **requirements.txt**: Optional project-specific Python packages
- **entrypoint.sh**: Copies host config (.gitconfig, .ssh) from staging mount; for Docker, remaps UID/GID and uses `gosu` to drop privileges

### Runtime volumes
- **Named volume** (`HOME_VOLUME`): Persistent `/home/devai` — survives container restarts
- **Bind mount**: `HOST_WORK_DIR` → `/home/devai/work`
- **Staging mount**: Host `.gitconfig`/`.ssh` → `/tmp/host-config/` (copied into home by entrypoint)

### User identity
- **Podman (rootless)**: Runs as container root (= host user). No user switching needed.
- **Docker**: Entrypoint remaps `devai` user to host UID/GID via `gosu`. Handles UID conflicts in GPU base images.

The container connects to host services via `host.containers.internal` for Ollama integration.
