# Dev AI Lab

**Run AI models entirely on your own hardware.** No cloud APIs, no data leaving your network, no per-token costs.

Dev AI Lab is a containerized development environment that brings together JupyterLab, multiple AI coding assistants, and local model inference into a single, reproducible setup. It runs open-weight LLMs on your GPU — from 4B parameter models for quick tasks to 70B models for complex reasoning — all served through a unified API with automatic GPU management.

### Why local inference?

Cloud AI services (ChatGPT, Claude API, Gemini API) are convenient but come with trade-offs that matter in professional settings:

- **Data sovereignty** — your code, documents, and conversations never leave your machine. No third-party data processing agreements needed. No risk of training data leakage.
- **Regulatory compliance** — meet data residency requirements (GDPR, HIPAA, financial regulations) without complex cloud configurations. The data stays where you control it.
- **Cost predictability** — no per-token billing, no surprise invoices. One-time hardware investment, unlimited inference. A single GPU pays for itself after a few months of heavy API usage.
- **No internet dependency** — works offline, on air-gapped networks, behind restrictive firewalls. Essential for secure environments and field work.
- **Full control** — choose your models, quantization, context length, and serving parameters. No vendor lock-in, no deprecated APIs, no terms-of-service changes.
- **Privacy by design** — conversations with AI about proprietary code, internal architecture, or sensitive business logic stay completely private.

Dev AI Lab makes local inference practical by handling the operational complexity: container builds, model management, GPU arbitration between multiple backends, and a web chat UI — all through a single `make` command.

For tasks where cloud AI is appropriate, the JupyterLab environment also includes **Claude Code**, **OpenAI Codex**, and **Google Gemini CLI** — giving you the flexibility to use local models for sensitive work and cloud models when you need their capabilities, all from the same workspace.

## Features

- **JupyterLab workspace** — full data science environment with launcher icons for Claude, Codex, Gemini, and Ollama. One click to start a conversation with any AI from within your notebook workflow.
- **Multiple AI CLIs pre-installed** — Claude Code, OpenAI Codex, Google Gemini CLI, and Ollama are all ready to use from the terminal. No manual installation, no version conflicts.
- **VS Code in the browser** — code-server provides a full Visual Studio Code experience accessible from any browser, with Jupyter integration via extensions.
- **Automatic GPU-arbitrated model serving** — run both GGUF models (via Ollama) and NVFP4 models (via vLLM) on a single GPU. The gpu-arbiter router transparently switches between backends based on the model you request — no manual intervention, no OOM errors.
- **Open WebUI chat interface** — web-based chat UI over HTTPS that sees all available models (both Ollama and vLLM). Select any model from the dropdown and start chatting.
- **Fast iteration** — two-layer container build separates rarely-changing system packages (base) from frequently-updated tools and Python packages (lab). Rebuilds take minutes, not hours.
- **Aggressive caching** — CLI binaries are updated via ETags (only downloads when upstream changes). APT proxy and Docker Hub mirror eliminate redundant network traffic across rebuilds.
- **Works with Podman and Docker** — rootless Podman is the default for security and simplicity. Docker is fully supported as an alternative.

## Quick Start

### Prerequisites (Debian 13 Trixie)

```bash
# Podman + Compose
sudo apt install podman python3-podman-compose

# Python tools (for model management scripts)
sudo apt install python3-yaml

# HuggingFace CLI (for vLLM model downloads)
pip install --user huggingface-hub

# NVIDIA GPU drivers (if not already installed)
sudo apt install nvidia-driver nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=podman --config=$HOME/.config/containers/containers.conf
```

### 1. Configure

```bash
cp .env.example .env
# Edit .env: set JUPYTER_TOKEN, adjust ports if needed
```

### 2. Build

```bash
make build           # Build all images (CPU + GPU + router)
```

### 3. Start Infrastructure

```bash
make cache-up        # Start Ollama, vLLM, router, Open WebUI, caches
```

### 4. Pull Models

```bash
make ollama-pull     # Pull all GGUF models from deploy/models.yaml
make vllm-pull       # Download all NVFP4 models from HuggingFace
```

### 5. Run

```bash
make lab-gpu         # Start JupyterLab with GPU (or make lab-cpu)
```

Access:
- **JupyterLab**: `https://<HOST_IP>:8888`
- **Open WebUI**: `https://<HOST_IP>:8443`

## Architecture

```
  Browser / API clients
  ─────────────────────────────────────────────────────────
  https://<HOST_IP>:8888        https://<HOST_IP>:8443
         │                              │
  ───────┼──────────────────────────────┼──────────────────
         │         devai-net            │
         │       (internal)             │
         ▼                              ▼
  devai-lab-gpu              devai-webui-proxy
  (JupyterLab + AI CLIs)    (nginx TLS termination)
         │                              │
         │                              ▼
         │                      devai-open-webui
         │                       (chat UI)
         │                              │
         └──────────┬───────────────────┘
                    │
                    ▼
             devai-router :11434
             (gpu-arbiter, 9 MB)
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
  devai-ollama          devai-vllm
  (GGUF models)         (NVFP4 models)
  always running         on-demand

  devai-apt-cache    devai-registry-cache
  (APT cache)        (Docker Hub mirror)
```

**External access** (host ports): JupyterLab `:8888`, Open WebUI `:8443`
**Internal only** (devai-net): router, Ollama, vLLM, caches — no host ports exposed

### GPU Arbitration

The gpu-arbiter router (`devai-router`) automatically manages GPU exclusion:

- **GGUF model requested** → routes to Ollama (auto-loads model)
- **NVFP4 model requested** → unloads Ollama, starts vLLM container, proxies request
- **Model switch** → recreates vLLM container with the new model
- **Idle timeout** → stops vLLM after 5 minutes of inactivity
- **API translation** → Ollama API ↔ OpenAI API for transparent Open WebUI support

Only one backend uses the GPU at a time. No manual switching required.

## Configuration

### `.env` — Host/runtime settings

| Setting | Default | Description |
|---------|---------|-------------|
| `LAB_PORT` | 8888 | JupyterLab port |
| `WEBUI_PORT` | 8443 | Open WebUI HTTPS port |
| `CONTAINER_RUNTIME` | podman | `podman` or `docker` |
| `HOST_HOME_DIR` | `$HOME` | Enables .gitconfig/.ssh in container |
| `HOME_VOLUME` | `$HOME/devai-home` | Persistent home directory |
| `JUPYTER_TOKEN` | — | Fixed access token (set in .env) |
| `APT_PROXY` | — | APT cache URL (e.g. `http://localhost:3142`) |

### `deploy/models.yaml` — Model catalog

Single source of truth for all models. Defines Ollama (GGUF) and vLLM (NVFP4) models with names, sizes, and purposes. Used by `make ollama-list`, `make vllm-list`, and the gpu-arbiter router. The included models have been selected for systems with consumer-grade GPUs (up to 24 GB VRAM) and up to 64 GB main RAM.

### Adding Python Packages

Create `requirements.txt` in the repo root and rebuild:

```bash
echo "langchain" > requirements.txt
make build-gpu
```

## Make Targets

Run `make help` for the full list:

```
BUILD                                       INFRASTRUCTURE                              RUN
fetch-cli        Update CLI binaries        cache-up         Start services             lab-cpu          JupyterLab (CPU)
pull-images      Pull base images           cache-down       Stop services              lab-gpu          JupyterLab (GPU)
build-cpu        Build image (CPU)          cache-status     Show status                shell-cpu        Shell (CPU)
build-gpu        Build image (GPU)          cache-clean      Remove cached data         shell-gpu        Shell (GPU)
build-router     Build router image
build            Build all (CPU+GPU+router)

OLLAMA (GGUF)                               vLLM (NVFP4)                                MAINTENANCE
ollama-list      List models                vllm-list        List models                 clean-cpu        Remove image (CPU)
ollama-pull      Pull model(s)              vllm-pull        Pull model(s)               clean-gpu        Remove image (GPU)
ollama-rm        Remove model               vllm-rm          Remove model                clean-router     Remove router image
ollama-status    Show status                vllm-status      Show status                 clean            Remove all images
ollama-df        Disk usage                 vllm-df          Disk usage                  prune            Prune dangling images
ollama-clean     Clean partials                                                          test             Run integration tests
```

## GPU Support

### Requirements

- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed
- For NVFP4 models: Blackwell architecture

### GPU images

The GPU lab image (`devai-lab-gpu`) includes PyTorch with CUDA. The base image uses `nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04`. Python is installed via uv (not system apt) to ensure a single Python version across CPU and GPU images.

## SSL / HTTPS

Both JupyterLab and Open WebUI support HTTPS:

- **JupyterLab**: Auto-detects mkcert certificates in `~/.jupyter/ssl/`
- **Open WebUI**: nginx proxy with mkcert certs or self-signed fallback

Generate certificates on the browser workstation:
```bash
mkcert <HOST_IP>
# Copy <HOST_IP>.pem and <HOST_IP>-key.pem to the container host
```

## Podman Registry Mirror

To route Docker Hub pulls through the local cache (saves bandwidth on rebuilds):

```bash
cp deploy/registries.conf ~/.config/containers/registries.conf
```

This tells Podman to try `localhost:5000` (the registry mirror started by `make cache-up`) before going to Docker Hub directly.

## Auto-Start at Boot

```bash
make install-systemd
```

Installs a systemd user service that starts all infrastructure containers (Ollama, router, Open WebUI, caches) on login. Uses `loginctl enable-linger` to keep services running after logout.

## Updating

```bash
make fetch-cli       # Update CLI binaries (Claude, Codex, Ollama, Gemini) via ETags
make pull-images     # Pull latest base and infrastructure images
make ollama-pull     # Sync Ollama models (skips unchanged)
make vllm-pull       # Sync vLLM models from HuggingFace
make build           # Rebuild all images with updated binaries/packages
```

## Storage Layout (LVM2)

All persistent data is stored on dedicated LVM2 thin-provisioned logical volumes under `/var/cache/devai/`. This provides:
- **Independent sizing** — each volume can be extended without affecting others
- **Thin provisioning** — space is allocated on demand from a shared pool
- **Clean separation** — models, container images, and caches don't compete for space

Reference implementation (volume group `vgais`):

| Volume | Mount | Size | Purpose |
|--------|-------|------|---------|
| `cache_ollama` | `/var/cache/devai/ollama` | 200G | Ollama GGUF models + vLLM NVFP4 models |
| `cache_registry` | `/var/cache/devai/registry` | 200G | Podman container image storage + Docker Hub mirror |
| `cache_pip` | `/var/cache/devai/pip` | 30G | Python package cache (uv) + CLI binaries |
| `cache_apt` | `/var/cache/devai/apt` | 10G | APT package cache (apt-cacher-ng) |
| `cache_npm` | `/var/cache/devai/npm` | 10G | npm package cache |
| `cache_open_webui` | `/var/cache/devai/open-webui` | 5G | Open WebUI application data |

Create the volumes:

```bash
# Create thin pool (adjust size for your disk)
sudo lvcreate -L 500G -T vgais/cachepool

# Create thin volumes
sudo lvcreate -V 200G -T vgais/cachepool -n cache_ollama
sudo lvcreate -V 200G -T vgais/cachepool -n cache_registry
sudo lvcreate -V 30G  -T vgais/cachepool -n cache_pip
sudo lvcreate -V 10G  -T vgais/cachepool -n cache_apt
sudo lvcreate -V 10G  -T vgais/cachepool -n cache_npm
sudo lvcreate -V 5G   -T vgais/cachepool -n cache_open_webui

# Format and mount
for vol in cache_ollama cache_registry cache_pip cache_apt cache_npm cache_open_webui; do
    sudo mkfs.xfs /dev/vgais/$vol
done

# Add to /etc/fstab
cat <<'EOF' | sudo tee -a /etc/fstab
/dev/vgais/cache_ollama     /var/cache/devai/ollama     xfs defaults 0 0
/dev/vgais/cache_registry   /var/cache/devai/registry   xfs defaults 0 0
/dev/vgais/cache_pip        /var/cache/devai/pip         xfs defaults 0 0
/dev/vgais/cache_apt        /var/cache/devai/apt         xfs defaults 0 0
/dev/vgais/cache_npm        /var/cache/devai/npm         xfs defaults 0 0
/dev/vgais/cache_open_webui /var/cache/devai/open-webui  xfs defaults 0 0
EOF

sudo mkdir -p /var/cache/devai/{ollama,registry,pip,apt,npm,open-webui}
sudo mount -a
sudo chown -R $USER:$USER /var/cache/devai
```

To extend a volume (e.g. when models fill up):
```bash
sudo lvextend -L 300G /dev/vgais/cache_ollama
sudo xfs_growfs /var/cache/devai/ollama
```

## Key Files

```
.env                          — Host/runtime configuration
.env.example                  — Configuration template
deploy/
  models.yaml                 — Model catalog (ollama + vllm)
  docker-compose.yaml         — Infrastructure services
  Dockerfile.base             — Base image (system packages, Python, Node)
  Dockerfile.lab              — Lab image (CLI tools, packages, JupyterLab)
  Dockerfile.router           — Router image (distroless, 9 MB)
  webui-proxy/                — nginx TLS proxy for Open WebUI
  systemd/                    — Auto-start service
gpu-arbiter/                  — Router Go source
scripts/                      — Python helpers for model management
tests/                        — Integration tests
requirements-base.txt         — Base Python packages (always installed)
requirements.txt              — Optional project-specific packages
packages/jupyter-ai-launchers — JupyterLab launcher extension
```

## License

MIT
