# Local Container Targets

These targets build and run the Dev AI Lab container directly using Podman or Docker.

## Targets

### `make build`

Build the CPU container image.

```bash
make build
```

**What it does:**
- Builds the container from `Dockerfile`
- Tags as `devai-lab:latest` (or custom `IMAGE_NAME`)
- Passes proxy settings if configured

**Environment variables:**
- `IMAGE_NAME` - Custom image name (default: `devai-lab`)
- `HTTP_PROXY` / `HTTPS_PROXY` - Proxy for build process

---

### `make build-gpu`

Build the GPU/CUDA container image.

```bash
make build-gpu
```

**What it does:**
- Builds from `Dockerfile.gpu` with NVIDIA CUDA support
- Tags as `devai-lab-gpu:latest`
- Includes PyTorch and CUDA runtime

**Requirements:**
- NVIDIA Container Toolkit (for running, not building)

---

### `make run`

Run the CPU container with JupyterLab.

```bash
make run
```

**What it does:**
- Starts container in interactive mode
- Exposes JupyterLab on port 8888
- Mounts working directory to `/home/devai/work`
- Connects to host Ollama instance
- Maps host user UID/GID for file permissions

**Environment variables:**
- `PORT` - JupyterLab port (default: `8888`)
- `HOST_WORK_DIR` - Directory to mount (default: current directory)
- `HOST_HOME_DIR` - Optional: mount home for .ssh, .gitconfig
- `OLLAMA_HOST` - Ollama server URL (default: `http://host.containers.internal:11434`)

**Access:**
```
http://localhost:8888/lab?token=<token>
```
The token is displayed in the terminal output.

---

### `make run-gpu`

Run the GPU container with JupyterLab.

```bash
make run-gpu
```

**What it does:**
- Same as `make run` but with GPU passthrough
- Uses `--gpus all` (Docker) or `--device nvidia.com/gpu=all` (Podman)

**Requirements:**
- NVIDIA GPU with drivers installed
- NVIDIA Container Toolkit configured

---

### `make shell`

Start an interactive shell without JupyterLab.

```bash
make shell
```

**What it does:**
- Starts container with `/bin/bash`
- Useful for debugging or running CLI tools directly
- Same volume mounts as `make run`

**Example uses:**
```bash
# Inside container
claude          # Run Claude CLI
gemini          # Run Gemini CLI
ollama list     # Check Ollama models
python          # Python REPL
```

---

### `make clean`

Remove the CPU container image.

```bash
make clean
```

---

### `make clean-gpu`

Remove the GPU container image.

```bash
make clean-gpu
```

---

### `make prune`

Clean up dangling images.

```bash
make prune
```

**What it does:**
- Removes untagged/dangling images
- Keeps tagged images and volumes
- Frees disk space

---

## Configuration

All targets read from `.env` file if present. Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

### Container Runtime

Set `CONTAINER_RUNTIME` to switch between Podman and Docker:

```bash
# In .env
CONTAINER_RUNTIME=podman  # or docker
```

### Volume Mounts

```bash
# Mount current directory
HOST_WORK_DIR=.

# Mount specific project
HOST_WORK_DIR=/path/to/project

# Also mount home directory (for .ssh, .gitconfig)
HOST_HOME_DIR=/home/username
```

## Examples

### Basic Usage

```bash
# Build and run
make build
make run
```

### GPU Workflow

```bash
# Build GPU image
make build-gpu

# Run with GPU
make run-gpu
```

### Custom Port

```bash
PORT=9999 make run
# Access at http://localhost:9999
```

### Behind Corporate Proxy

```bash
# In .env
HTTP_PROXY=http://proxy.company.com:8080
HTTPS_PROXY=http://proxy.company.com:8080

make build
make run
```
