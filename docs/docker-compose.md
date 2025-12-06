# Docker Compose Targets

These targets use Docker Compose for multi-service orchestration, running both Dev AI Lab and Ollama together.

## Targets

### `make compose-up`

Start all services with Docker Compose.

```bash
make compose-up
```

**What it does:**
- Starts `devai` (JupyterLab) and `ollama` containers
- Creates a shared network for service communication
- Runs in detached mode (background)
- Creates persistent volumes for data

**Services started:**
| Service | Port | Description |
|---------|------|-------------|
| devai | 8888 | JupyterLab with AI CLIs |
| ollama | 11434 | Local LLM inference server |

**Access:**
```
http://localhost:8888
```

---

### `make compose-up-gpu`

Start all services with GPU support.

```bash
make compose-up-gpu
```

**What it does:**
- Same as `compose-up` but enables GPU for both services
- Uses `docker-compose.gpu.yml` overlay
- Both devai and ollama get GPU access

**Requirements:**
- NVIDIA GPU with drivers
- NVIDIA Container Toolkit
- Docker Compose v2.x with GPU support

---

### `make compose-down`

Stop and remove all containers.

```bash
make compose-down
```

**What it does:**
- Stops running containers
- Removes containers
- Preserves volumes (data persists)

---

### `make compose-logs`

View container logs.

```bash
make compose-logs
```

**What it does:**
- Shows logs from all services
- Follows log output (Ctrl+C to exit)
- Displays JupyterLab token on startup

---

### `make compose-build`

Build images using Docker Compose.

```bash
make compose-build
```

**What it does:**
- Builds the devai image from Dockerfile
- Pulls the latest ollama image
- Uses build cache when possible

---

### `make compose-ps`

Show running containers.

```bash
make compose-ps
```

---

## Configuration

### Environment File

Copy the example environment file:

```bash
cp deploy/compose/.env.example deploy/compose/.env
```

Edit `deploy/compose/.env`:

```bash
# Image settings
IMAGE_NAME=devai-lab
IMAGE_TAG=latest

# Ports
PORT=8888
OLLAMA_PORT=11434

# Working directory
HOST_WORK_DIR=./work

# API Keys (optional)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

### Local Overrides

For local customizations, create `deploy/compose/docker-compose.override.yml`:

```yaml
services:
  devai:
    volumes:
      - ~/.ssh:/home/devai/.ssh:ro
      - ~/.gitconfig:/home/devai/.gitconfig:ro
```

This file is automatically loaded by Docker Compose.

---

## Architecture

```
┌─────────────────────────────────────────┐
│              devai-network              │
│  ┌─────────────┐    ┌─────────────────┐ │
│  │   devai     │    │     ollama      │ │
│  │  :8888      │───▶│    :11434       │ │
│  │ JupyterLab  │    │  LLM Server     │ │
│  └─────────────┘    └─────────────────┘ │
└─────────────────────────────────────────┘
         │                   │
    localhost:8888      localhost:11434
```

### Volumes

| Volume | Purpose |
|--------|---------|
| `devai-local` | Python packages, user data |
| `devai-cache` | Build caches |
| `ollama-models` | Downloaded LLM models |

---

## Common Workflows

### First Time Setup

```bash
# Copy environment file
cp deploy/compose/.env.example deploy/compose/.env

# Edit configuration
vim deploy/compose/.env

# Build and start
make compose-build
make compose-up

# View logs to get JupyterLab token
make compose-logs
```

### Pull a Model in Ollama

```bash
# Start services
make compose-up

# Pull a model
docker exec -it ollama ollama pull llama2

# Or from JupyterLab terminal
ollama pull mistral
```

### Using External Ollama

If you already have Ollama running on your host:

```bash
# In deploy/compose/.env
OLLAMA_HOST=http://host.docker.internal:11434
```

Then disable the ollama service by creating an override:

```yaml
# deploy/compose/docker-compose.override.yml
services:
  ollama:
    profiles:
      - disabled
```

### GPU Workflow

```bash
# Verify GPU is available
nvidia-smi

# Start with GPU
make compose-up-gpu

# Verify GPU in container
docker exec -it devai-lab nvidia-smi
```

---

## Troubleshooting

### Port Already in Use

```bash
# Check what's using the port
lsof -i :8888

# Use a different port
PORT=9999 make compose-up
```

### Permission Denied on Volumes

```bash
# Check volume permissions
docker volume inspect devai-local

# Fix by removing and recreating
docker volume rm devai-local
make compose-up
```

### Ollama Connection Failed

```bash
# Check if ollama is running
docker ps | grep ollama

# Check ollama logs
docker logs ollama

# Test connectivity from devai
docker exec -it devai-lab curl http://ollama:11434/api/tags
```
