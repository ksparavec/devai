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
*   **Runtime**: Optimized for **Podman** (supports rootless mode) but fully compatible with Docker.

## Prerequisites

*   **Podman** (recommended) or Docker
*   **Make** (GNU Make)

## Quick Start

### 1. Configuration

#### Podman Storage Driver (Recommended)
Before building, ensure you are using the `overlay` storage driver for better performance and disk usage. The default `vfs` driver can be slow and space-consuming.

1.  **Check current driver**:
    ```bash
    podman info --format '{{.Store.GraphDriverName}}'
    ```

2.  **Update Configuration** (if the output is not `overlay`):
    Create or edit `~/.config/containers/storage.conf`:
    ```ini
    [storage]
    driver = "overlay"
    ```

3.  **Reset Storage** (Warning: This deletes all existing images/containers):
    ```bash
    podman system reset
    ```

#### Environment Setup
Copy the example configuration file to create your local environment settings:

```bash
cp .env.example .env
```

Open `.env` and you **must** adjust the following settings:

*   `HOST_HOME_DIR`: **Required.** Set this to your local user's home directory (e.g., `/home/username` or `/Users/username`).
    *   *Why?* Mounting your home directory allows the container to access your existing configuration files (like `.gitconfig`, `.ssh/`, or shell aliases), ensuring the container environment feels familiar and fully functional.
*   `CONTAINER_RUNTIME`: Defaults to `podman`. Change to `docker` if preferred.
*   `PORT`: Local port to access JupyterLab (default: `8888`).
*   **Proxy Settings**: Configure `HTTP_PROXY` and `HTTPS_PROXY` if you are behind a corporate firewall.

#### Optional: Add Python Packages
To install additional Python modules into the image:

1.  Copy the example requirements file:
    ```bash
    cp requirements.txt.example requirements.txt
    ```
2.  Edit `requirements.txt` and add your desired packages (one per line).
3.  Build the image (the build process will automatically detect and install these packages):
    ```bash
    make build
    ```

### 2. Build the Image

Use the `make` command to build the container image:

```bash
make build
```

This command passes proxy settings from your `.env` file (or environment variables) to the build process.

### 3. Run the Environment

Start the container with:

```bash
make run
```

*   The current directory (`.`) is mounted to `/home/devai/work` inside the container. Any files created in the `work/` folder inside JupyterLab will persist on your host machine.
*   The container handles user permissions automatically, mapping the internal user (`devai`) to your host user ID to avoid permission issues with mounted files.

### 4. Access JupyterLab

After running `make run`, the console will display access URLs. You will typically see links for both your **Host IP** and **localhost**:

```text
http://192.168.1.10:8888/lab?token=<long-token-string>
http://127.0.0.1:8888/lab?token=<long-token-string>
```

Copy and paste either URL into your browser to access the JupyterLab interface.

## Using AI Tools

All AI CLIs are available from a terminal within JupyterLab (File -> New -> Terminal).

### Google Gemini CLI

```bash
gemini prompt "Hello"
```

On first run, follow the browser authentication flow. Credentials are saved in your mounted home directory.

### Claude Code CLI

```bash
claude
```

Requires `ANTHROPIC_API_KEY` environment variable or interactive login.

### OpenAI Codex CLI

```bash
codex
```

Requires `OPENAI_API_KEY` environment variable or ChatGPT Plus/Pro/Business login.

### OpenAI Python SDK

```python
# In Python/Jupyter
from openai import OpenAI
client = OpenAI()  # Uses OPENAI_API_KEY env var
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
```

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
collection.add(
    documents=["doc1", "doc2"],
    ids=["id1", "id2"]
)
results = collection.query(query_texts=["search query"], n_results=2)
```

## GPU Support

For NVIDIA GPU acceleration:

1.  Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2.  Build the GPU image:
    ```bash
    make build-gpu
    ```
3.  Run with GPU:
    ```bash
    make run-gpu
    ```

The GPU image is based on NVIDIA CUDA with cuDNN and includes PyTorch with CUDA support.

### Selecting a CUDA Base Image

Use the included selector script to choose and update the CUDA base image:

```bash
# List available cuDNN images
./scripts/select-cuda-image.sh --list

# Auto-select recommended version and update Dockerfile.gpu
./scripts/select-cuda-image.sh --auto

# Interactive selection with ncurses UI (if dialog is installed)
./scripts/select-cuda-image.sh

# Specify Ubuntu version
./scripts/select-cuda-image.sh --auto 22.04
```

The script queries Docker Hub for available NVIDIA CUDA images with cuDNN support and recommends the latest stable version from the previous major release for best compatibility with PyTorch and other ML libraries.

## Additional Commands

```bash
make shell      # Interactive shell without JupyterLab
make clean      # Remove CPU image
make clean-gpu  # Remove GPU image
make prune      # Clean up dangling images (keeps tagged images and volumes)
make help       # Show all available targets
```

## Cloud Deployment

Dev AI Lab supports deployment to major cloud providers via Docker Compose, Terraform, and Kubernetes.

### Deployment Options

| Method | Use Case | Documentation |
|--------|----------|---------------|
| **Docker Compose** | Local multi-service orchestration | [docs/docker-compose.md](docs/docker-compose.md) |
| **Terraform** | Cloud infrastructure provisioning | [docs/terraform.md](docs/terraform.md) |
| **Kubernetes** | Container orchestration at scale | [docs/kubernetes.md](docs/kubernetes.md) |

### Supported Clouds

| Cloud | Terraform | Kubernetes |
|-------|-----------|------------|
| **AWS** | ECS Fargate, ECR, ALB | EKS with ALB Ingress |
| **Azure** | Container Instances, ACR | AKS with AGIC |
| **GCP** | Cloud Run, Artifact Registry | GKE with GCE Ingress |

### Quick Start (Docker Compose)

```bash
# Start devai + ollama locally
make compose-up

# View logs (includes JupyterLab token)
make compose-logs

# Stop services
make compose-down
```

### Quick Start (Cloud - AWS Example)

```bash
# 1. Configure
cp deploy/terraform/aws/terraform.tfvars.example deploy/terraform/aws/terraform.tfvars
vim deploy/terraform/aws/terraform.tfvars

# 2. Deploy infrastructure
make tf-init-aws
make tf-apply-aws

# 3. Push container image
./scripts/push-image.sh aws

# 4. Get service URL
cd deploy/terraform/aws && terraform output service_url
```

### Configuration Management

All deployment methods share configuration from `config/`:

```bash
# Generate configs for all targets
make config-generate PROFILE=dev

# Generate for specific profile
make config-generate PROFILE=prod
```

See [docs/utilities.md](docs/utilities.md) for details on the configuration system.

## Documentation

| Document | Description |
|----------|-------------|
| [docs/local-container.md](docs/local-container.md) | Local build/run targets (build, run, shell) |
| [docs/docker-compose.md](docs/docker-compose.md) | Docker Compose multi-service setup |
| [docs/terraform.md](docs/terraform.md) | Cloud deployment with Terraform |
| [docs/kubernetes.md](docs/kubernetes.md) | Kubernetes deployment with Kustomize |
| [docs/utilities.md](docs/utilities.md) | Config generation and utility scripts |

## All Make Targets

### Local Container
| Target | Description |
|--------|-------------|
| `make build` | Build CPU container image |
| `make build-gpu` | Build GPU/CUDA container image |
| `make run` | Run container with JupyterLab |
| `make run-gpu` | Run GPU container with JupyterLab |
| `make shell` | Interactive shell without JupyterLab |
| `make clean` | Remove CPU image |
| `make clean-gpu` | Remove GPU image |
| `make prune` | Clean dangling images |

### Docker Compose
| Target | Description |
|--------|-------------|
| `make compose-up` | Start all services |
| `make compose-up-gpu` | Start with GPU support |
| `make compose-down` | Stop and remove containers |
| `make compose-logs` | View container logs |
| `make compose-build` | Build images |
| `make compose-ps` | Show running containers |

### Terraform
| Target | Description |
|--------|-------------|
| `make tf-init-aws` | Initialize AWS |
| `make tf-plan-aws` | Plan AWS deployment |
| `make tf-apply-aws` | Deploy to AWS |
| `make tf-destroy-aws` | Destroy AWS resources |
| `make tf-init-azure` | Initialize Azure |
| `make tf-plan-azure` | Plan Azure deployment |
| `make tf-apply-azure` | Deploy to Azure |
| `make tf-destroy-azure` | Destroy Azure resources |
| `make tf-init-gcp` | Initialize GCP |
| `make tf-plan-gcp` | Plan GCP deployment |
| `make tf-apply-gcp` | Deploy to GCP |
| `make tf-destroy-gcp` | Destroy GCP resources |

### Kubernetes
| Target | Description |
|--------|-------------|
| `make k8s-build` | Build K8s manifests |
| `make k8s-apply` | Apply to cluster |
| `make k8s-delete` | Delete resources |

### Utilities
| Target | Description |
|--------|-------------|
| `make config-generate` | Generate configs from YAML |
| `make help` | Show all targets |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
