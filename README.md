# Dev AI Lab

A containerized environment designed for AI experimentation and development, featuring **JupyterLab** and multiple AI CLIs. This setup provides a consistent, isolated workspace with essential tools pre-installed.

## Features

*   **Base Environment**: Debian Trixie (slim) with Python 3.13 via uv.
*   **Tool Management**: **mise** for version-controlled tool installation.
*   **Interactive Computing**: **JupyterLab** pre-installed and configured.
*   **AI Tools**:
    *   **Google Gemini CLI** (`@google/gemini-cli`)
    *   **Claude Code CLI** (`@anthropic-ai/claude-code`)
    *   **OpenAI Codex CLI** (`@openai/codex`)
    *   **OpenAI Python SDK** (`openai`)
    *   **Ollama** for local model inference
*   **Vector Database**: **ChromaDB** for embeddings and RAG experiments.
*   **GPU Support**: NVIDIA CUDA support for accelerated inference (optional).
*   **Package Management**: **mise** manages `uv` and `node`; Python packages via `uv pip`.
*   **Development Tools**: `git`, `build-essential`, `rustc`, `cargo` (via apt).
*   **Runtime**: Optimized for **Podman** (supports rootless mode) but fully compatible with Docker. Both can be auto-installed via bootstrap.
*   **Multi-Cloud Deployment**: AWS, Azure, and GCP via Terraform and Kubernetes.

## Workflow Overview

Run `make` to see the recommended workflow:

```
1. SETUP BUILD ENVIRONMENT  - Install container runtimes and cloud CLIs
2. BUILD CONTAINER IMAGES   - Build CPU or GPU container images
3. DEPLOY TO TARGET RUNTIME - Deploy via Terraform or Kubernetes
4. RUN LOCALLY              - Run container with JupyterLab
5. MAINTENANCE              - Clean up images and resources
```

## Quick Start: Local Development

### Prerequisites

Ensure you have Podman or Docker installed. If not, run:

```bash
make setup-build-env         # Install runtimes + cloud tools (recommended)
# Or install just runtimes:
make setup-runtime           # Both podman and docker
make setup-runtime-podman    # Podman only
make setup-runtime-docker    # Docker only
```

### 1. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set:
*   `HOST_HOME_DIR`: Your home directory (e.g., `/home/username`)
*   `CONTAINER_RUNTIME`: `podman` (default) or `docker`

### 2. Build and Run

```bash
make build
make run
```

Access JupyterLab at the URL shown in the console output.

## Quick Start: Cloud Deployment

### 1. Setup Build Environment

Install container runtimes and cloud management tools:

```bash
make setup-build-env
```

This installs:
*   **Container Runtimes**: Podman and Docker (both by default)
*   **Cloud CLIs**: AWS CLI, Azure CLI, gcloud, Terraform, kubectl, kustomize

To install only cloud tools (if you already have a container runtime):

```bash
make setup-cloud-tools
```

All tools are installed to `~/.local/bin`. Add to your PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Verify installation:

```bash
make check-cloud-tools
```

### 2. Configure Cloud Provider

```bash
# AWS
aws configure

# Azure
az login

# GCP
gcloud init
```

### 3. Build Container Image

```bash
make build
```

### 4. Deploy Infrastructure

Example for AWS:

```bash
# Configure
cp deploy/terraform/aws/terraform.tfvars.example deploy/terraform/aws/terraform.tfvars
vim deploy/terraform/aws/terraform.tfvars

# Deploy
make tf-init-aws
make tf-apply-aws

# Push image
./scripts/push-image.sh aws
```

See [Cloud Deployment](#cloud-deployment) for Azure, GCP, and Kubernetes options.

---

## Using AI Tools

All AI CLIs are available from a terminal within JupyterLab (File -> New -> Terminal).

### Google Gemini CLI

```bash
gemini prompt "Hello"
```

On first run, follow the browser authentication flow.

### Claude Code CLI

```bash
claude
```

Requires `ANTHROPIC_API_KEY` environment variable or interactive login.

### OpenAI Codex CLI

```bash
codex
```

Requires `OPENAI_API_KEY` environment variable.

### Ollama (Local Models)

The container connects to Ollama running on your host machine:

1.  Install Ollama on host: https://ollama.ai
2.  Start Ollama: `ollama serve`
3.  Pull a model: `ollama pull llama3.2`
4.  Use from container:

```python
import ollama
response = ollama.chat(model='llama3.2', messages=[
    {'role': 'user', 'content': 'Hello'}
])
```

### ChromaDB (Vector Database)

```python
import chromadb
client = chromadb.Client()
collection = client.create_collection("my_embeddings")
collection.add(documents=["doc1", "doc2"], ids=["id1", "id2"])
results = collection.query(query_texts=["search query"], n_results=2)
```

---

## GPU Support

For NVIDIA GPU acceleration:

1.  Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2.  Build and run:
    ```bash
    make build-gpu
    make run-gpu
    ```

---

## Cloud Deployment

Dev AI Lab supports deployment to major cloud providers via Terraform and Kubernetes.

### Supported Clouds

| Cloud | Terraform | Kubernetes |
|-------|-----------|------------|
| **AWS** | ECS Fargate, ECR, ALB | EKS with ALB Ingress |
| **Azure** | Container Instances, ACR | AKS with AGIC |
| **GCP** | Cloud Run, Artifact Registry | GKE with GCE Ingress |

### Terraform Deployment

```bash
# AWS
make tf-init-aws && make tf-plan-aws && make tf-apply-aws

# Azure
make tf-init-azure && make tf-plan-azure && make tf-apply-azure

# GCP
make tf-init-gcp && make tf-plan-gcp && make tf-apply-gcp
```

### Kubernetes Deployment

```bash
make k8s-build KUSTOMIZE_OVERLAY=dev
make k8s-apply KUSTOMIZE_OVERLAY=dev
```

### Docker Compose (Local Multi-Service)

```bash
make compose-up      # Start devai + ollama
make compose-logs    # View logs (includes JupyterLab token)
make compose-down    # Stop services
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [ansible/README.md](ansible/README.md) | Cloud tools setup details |
| [docs/local-container.md](docs/local-container.md) | Local build/run details |
| [docs/docker-compose.md](docs/docker-compose.md) | Docker Compose setup |
| [docs/terraform.md](docs/terraform.md) | Terraform deployment |
| [docs/kubernetes.md](docs/kubernetes.md) | Kubernetes deployment |
| [docs/utilities.md](docs/utilities.md) | Config generation scripts |

---

## All Make Targets

### Setup Build Environment

| Target | Description |
|--------|-------------|
| `make setup-build-env` | Install container runtimes + all cloud CLIs (recommended) |
| `make setup-cloud-tools` | Install cloud CLIs only (no runtimes) |
| `make check-cloud-tools` | Verify installed tool versions |

See [Selective Installation Options](#selective-installation-options) in Appendix A for partial installation.

### Build Container Images

| Target | Description |
|--------|-------------|
| `make build` | Build CPU container image |
| `make build-gpu` | Build GPU/CUDA container image |
| `make compose-build` | Build images with Docker Compose |

### Deploy to Cloud (Terraform)

| Target | Description |
|--------|-------------|
| `make tf-init-aws` | Initialize Terraform for AWS |
| `make tf-plan-aws` | Plan AWS deployment |
| `make tf-apply-aws` | Deploy to AWS |
| `make tf-destroy-aws` | Destroy AWS resources |
| `make tf-init-azure` | Initialize Terraform for Azure |
| `make tf-plan-azure` | Plan Azure deployment |
| `make tf-apply-azure` | Deploy to Azure |
| `make tf-destroy-azure` | Destroy Azure resources |
| `make tf-init-gcp` | Initialize Terraform for GCP |
| `make tf-plan-gcp` | Plan GCP deployment |
| `make tf-apply-gcp` | Deploy to GCP |
| `make tf-destroy-gcp` | Destroy GCP resources |

### Deploy to Kubernetes

| Target | Description |
|--------|-------------|
| `make k8s-build` | Build Kubernetes manifests |
| `make k8s-apply` | Apply manifests to cluster |
| `make k8s-delete` | Delete Kubernetes resources |

### Run Locally

| Target | Description |
|--------|-------------|
| `make run` | Run container with JupyterLab (CPU) |
| `make run-gpu` | Run container with JupyterLab (GPU) |
| `make shell` | Interactive shell without JupyterLab |
| `make compose-up` | Start services with Docker Compose |
| `make compose-up-gpu` | Start with GPU support |
| `make compose-down` | Stop and remove containers |
| `make compose-logs` | View container logs |
| `make compose-ps` | Show running containers |

### Maintenance

| Target | Description |
|--------|-------------|
| `make clean` | Remove CPU container image |
| `make clean-gpu` | Remove GPU container image |
| `make prune` | Clean up dangling images |
| `make config-generate` | Generate configs from YAML profiles |
| `make help` | Show workflow and all targets |

---

## Appendix A: Detailed Configuration

### Environment Variables (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST_HOME_DIR` | - | Host home directory to mount |
| `HOST_WORK_DIR` | `.` | Working directory mounted to /home/devai/work |
| `CONTAINER_RUNTIME` | `podman` | Container runtime for running containers and installation preference |
| `PORT` | `8888` | JupyterLab port |
| `OLLAMA_HOST` | `http://host.containers.internal:11434` | Ollama server URL |
| `HTTP_PROXY` / `HTTPS_PROXY` | - | Proxy settings |

### Cloud Tools Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVAI_PYTHON_VERSION` | `3.12` | Python version for tools venv |
| `DEVAI_VENV_DIR` | `~/.local/devai-venv` | Virtual environment path |
| `DEVAI_BIN_DIR` | `~/.local/bin` | Tool binary directory |

### Selective Installation Options

By default, `make setup-build-env` installs everything. Use these options for selective installation:

```bash
# Via Makefile - runtime selection
make setup-build-env INSTALL_RUNTIME=podman     # Runtimes: Podman only + all cloud tools
make setup-build-env INSTALL_RUNTIME=docker     # Runtimes: Docker only + all cloud tools
make setup-build-env INSTALL_RUNTIME=both       # Runtimes: Both (default) + all cloud tools
make setup-build-env INSTALL_RUNTIME=none       # Cloud tools only (same as setup-cloud-tools)

# Via Makefile - runtime only (no cloud tools)
make setup-runtime                              # Both runtimes
make setup-runtime-podman                       # Podman only
make setup-runtime-docker                       # Docker only

# Via Makefile - specific cloud tools only (no runtimes)
make setup-cloud-tools-aws                      # AWS CLI only
make setup-cloud-tools-azure                    # Azure CLI only
make setup-cloud-tools-gcp                      # Google Cloud CLI only
make setup-cloud-tools-terraform                # Terraform only
make setup-cloud-tools-k8s                      # kubectl + kustomize only

# Via bootstrap script directly
./ansible/bootstrap.sh --install-runtime podman
./ansible/bootstrap.sh --install-runtime docker
./ansible/bootstrap.sh --install-runtime both
./ansible/bootstrap.sh --install-runtime none
./ansible/bootstrap.sh -- --tags aws            # AWS CLI only
./ansible/bootstrap.sh -- --tags terraform      # Terraform only
```

If `.env` exists with `CONTAINER_RUNTIME` set, that value determines which runtime(s) to install. Existing installations are updated if newer versions are available.

### Adding Python Packages

```bash
cp requirements.txt.example requirements.txt
# Edit requirements.txt
make build
```

## Appendix B: Podman Configuration

### Storage Driver Setup

For better performance, use the `overlay` storage driver:

1.  Check current driver:
    ```bash
    podman info --format '{{.Store.GraphDriverName}}'
    ```

2.  Update configuration (`~/.config/containers/storage.conf`):
    ```ini
    [storage]
    driver = "overlay"
    ```

3.  Reset storage (warning: deletes all images/containers):
    ```bash
    podman system reset
    ```

## Appendix C: GPU Image Selection

Use the selector script to choose CUDA base image:

```bash
# List available cuDNN images
./scripts/select-cuda-image.sh --list

# Auto-select recommended version
./scripts/select-cuda-image.sh --auto

# Interactive selection
./scripts/select-cuda-image.sh
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
