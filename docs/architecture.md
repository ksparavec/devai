# Architecture Overview

This document provides a comprehensive architectural analysis of Dev AI Lab, covering structure, technologies, design patterns, and deployment capabilities.

## Repository Structure

```
devai/
├── ansible/                    # Build environment setup (tools & runtimes)
│   ├── bootstrap.sh           # Main entry point for tool installation
│   ├── roles/                 # Installation roles (8 cloud tools + 2 runtimes)
│   └── playbooks/             # Ansible playbook for orchestration
├── config/                     # Centralized configuration source
│   ├── common/                # Shared config (env, ports, resources, labels)
│   ├── profiles/              # Environment overlays (dev, staging, prod)
│   └── secrets.example.yaml   # Secrets template
├── deploy/                     # Multi-environment deployment
│   ├── compose/               # Docker Compose (Phase 1)
│   ├── terraform/             # Cloud IaC (Phase 2)
│   │   ├── aws/              # ECS Fargate, ECR, ALB
│   │   ├── azure/            # Container Instances, ACR
│   │   ├── gcp/              # Cloud Run, Artifact Registry
│   │   └── modules/          # Reusable cloud-agnostic modules
│   └── kubernetes/            # Kubernetes orchestration (Phase 3)
│       ├── base/             # Kustomize base manifests
│       └── overlays/         # Environment & cloud-specific patches
├── docs/                       # Deployment documentation
├── scripts/                    # Utility scripts
│   ├── generate-config.sh     # Config generation from YAML
│   ├── push-image.sh          # Push image to cloud registries
│   └── select-cuda-image.sh   # CUDA image selection utility
├── Dockerfile                  # Unified CPU/GPU container image
├── Makefile                    # Build orchestration (67+ targets)
├── mise.toml                   # Tool version management
└── entrypoint.sh              # Container initialization script
```

## Technology Stack

### Base Environment

| Component | Technology | Purpose |
|-----------|------------|---------|
| OS | Debian Trixie (slim) | Minimal container base |
| Python | 3.13 via uv | Primary runtime |
| Node.js | v25 | npm-based AI CLIs |
| System Tools | apt (git, build-essential, rustc, cargo, curl, gnupg, gosu) | Build dependencies |

### Tool Management

**mise** serves as the version-controlled tool installer, replacing traditional tools like asdf/nvm/pyenv:

```toml
[tools]
uv = "latest"           # Python package manager
node = "25"             # Node.js runtime
"python:uv" = "3.13"   # Python via uv backend
```

### AI and Compute Tools

**npm packages (3)**:
- `@google/gemini-cli` - Google's Gemini API CLI
- `@anthropic-ai/claude-code` - Anthropic's Claude Code CLI
- `@openai/codex` - OpenAI Codex CLI

**Python packages (38)** in CPU version:

| Category | Packages |
|----------|----------|
| Jupyter | jupyterlab, ipywidgets |
| AI/ML | openai, ollama, chromadb, transformers, datasets, accelerate |
| Data Science | numpy, pandas, scipy, scikit-learn, matplotlib, seaborn |
| NLP | nltk, spacy |
| Vision | pillow, opencv-python |
| Utilities | tqdm, requests, python-dotenv, pyyaml, jsonlines, pydantic |

**GPU extras** (installed conditionally with NVIDIA CUDA):
- torch, torchvision, torchaudio

### Container Runtimes

| Runtime | Description |
|---------|-------------|
| Podman | Rootless container runtime (default) |
| Docker | Standard Docker runtime |

Both can be installed via the Ansible bootstrap system.

### Cloud Deployment Tools (via Ansible Bootstrap)

- **AWS CLI v2**: ECR, ECS, ALB, IAM
- **Azure CLI**: ACR, Container Instances, AKS
- **Google Cloud SDK**: Artifact Registry, Cloud Run, GKE
- **Terraform**: Infrastructure as Code
- **kubectl**: Kubernetes CLI
- **kustomize**: Kubernetes configuration management

## Build System

### Makefile Organization

The Makefile provides 67+ targets organized into 5 workflows:

**1. Setup Build Environment**
- `setup-build-env` - Install container runtimes + all cloud CLIs
- `setup-cloud-tools` - Cloud tools only
- `check-cloud-tools` - Verify installations

**2. Build Container Images**
- `build` - CPU image from Dockerfile
- `build-gpu` - GPU/CUDA image with nvidia/cuda base
- `compose-build` - Build via Docker Compose

**3. Deploy to Cloud**
- Terraform: init, plan, apply, destroy for AWS/Azure/GCP (12 targets)
- Kubernetes: build, apply, delete (3 targets)

**4. Run Locally**
- `run` / `run-gpu` - JupyterLab with optional GPU
- `shell` - Interactive shell
- `compose-up` / `compose-down` - Docker Compose orchestration

**5. Maintenance**
- `clean` / `clean-gpu` - Remove images
- `prune` - Clean dangling images
- `config-generate` - Generate deployment configs

### Dockerfile Design

A single unified Dockerfile supports both CPU and GPU builds via build arguments:

| Argument | Purpose |
|----------|---------|
| `BASE_IMAGE` | Container base (Debian or NVIDIA CUDA) |
| `GPU_BUILD` | Boolean flag to include GPU packages |
| `HTTP_PROXY` / `HTTPS_PROXY` | Proxy support for corporate environments |

Key design decisions:
- mise installs tools system-wide in `/opt/mise`
- Python packages installed globally with `uv pip`
- User `devai` created with UID 1000 for rootless operation
- Cache cleanup to minimize image size

## Architecture Patterns

### 3-Phase Deployment Strategy

The project implements a progressive deployment architecture:

```
Phase 1: Docker Compose (Local)
    ↓
Phase 2: Terraform (Cloud Infrastructure)
    ↓
Phase 3: Kubernetes (Portable Orchestration)
```

**Phase 1: Docker Compose**
- Local development orchestration
- devai + Ollama services
- Shared volumes and networking
- GPU support via override file

**Phase 2: Terraform**
- Modular design with cloud-agnostic interfaces
- Shared modules: networking, compute, storage, registries
- Cloud-specific implementations for AWS, Azure, GCP
- Unified `variables.tf` interface across all clouds

**Phase 3: Kubernetes**
- Kustomize-based configuration management
- Base manifests (deployment, service, configmap, PVC, ingress)
- Environment overlays (dev/staging/prod)
- Cloud-specific patches (storage classes, ingress controllers)

### Single-Source Configuration

The project uses a layered configuration architecture to eliminate duplication:

```
config/common/*.yaml
       ↓
  Profile overlay (dev/staging/prod)
       ↓
  ┌────┴────┬────────────┐
  ↓         ↓            ↓
Compose   Terraform    Kubernetes
```

`scripts/generate-config.sh` produces:
- `.env` for Docker Compose
- `terraform.tfvars` for Terraform
- ConfigMaps for Kubernetes

### Container User Management

The `entrypoint.sh` script handles:
- Dynamic user creation based on environment variables
- UID/GID mapping for rootless containers
- Directory initialization and permissions
- gosu for user switching (safe alternative to `su`)
- Custom Jupyter URL injection with host IP

## Ansible Bootstrap System

**Purpose**: Automated, idempotent installation of development tools and cloud CLIs.

**Architecture**:
1. Bootstrap script installs prerequisites (curl, tar, unzip, etc.)
2. Creates Python 3.12 venv with uv
3. Installs Ansible into venv
4. Runs Ansible playbook with 8 roles

**Available Roles** (tag-based targeting):

| Role | Description |
|------|-------------|
| `podman` | Rootless container runtime |
| `docker` | Standard Docker runtime |
| `aws_cli` | AWS CLI v2 |
| `azure_cli` | Azure CLI |
| `gcloud` | Google Cloud SDK |
| `terraform` | HashiCorp Terraform |
| `kubectl` | Kubernetes CLI |
| `kustomize` | Kubernetes customization tool |

**Features**:
- Idempotent (safe to run multiple times)
- Cross-platform (Linux distros + macOS)
- Selective installation via tags
- Custom version configuration
- Stores tools in `~/.local/bin` (no sudo after setup)

## Cloud Deployment Capabilities

### Supported Platforms

| Feature | AWS | Azure | GCP |
|---------|-----|-------|-----|
| **Container Registry** | ECR | ACR | Artifact Registry |
| **Compute (Terraform)** | ECS Fargate + ALB | Container Instances | Cloud Run |
| **Kubernetes** | EKS | AKS | GKE |
| **Storage** | EFS | Azure Files | Filestore |
| **GPU Support** | p3/g4dn instances | GPU SKUs | GKE GPU nodes |
| **Load Balancing** | ALB | Application Gateway | Cloud Load Balancer |

### Kubernetes Deployment Options

- **Base manifests**: Standard deployment, service, configmap, PVC, ingress
- **Environment overlays**: dev (1 replica, low resources) -> staging -> prod (3 replicas, HPA)
- **Cloud overlays**: Storage class patches, ingress controller annotations, service account configs
- **Composable**: `kustomize build overlays/prod overlays/aws` for production on AWS

## Security Features

- **No hardcoded secrets**: `secrets.example.yaml` template only
- **Environment-specific overrides**: Profile-based configuration
- **Rootless containers**: Podman runs without privileged access
- **Cloud IAM integration**: IRSA (AWS), Workload Identity (Azure/GCP)
- **Network policies**: Kubernetes NetworkPolicy resources
- **HTTPS by default**: Terraform configurations include TLS support

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Unified Dockerfile | Single image for CPU/GPU via build arguments reduces maintenance |
| mise over traditional tools | Consistent version management across distros |
| Rootless containers | Podman configured for security without elevated privileges |
| Kustomize over Helm | Simpler configuration without templating complexity |
| Layered configuration | YAML-based profiles enable easy environment switching |
| Ansible for bootstrap | Ensures reproducibility across different machines |
| gosu for user switching | Safer than `su`, prevents signal handling issues |
| Named volumes | Better persistence and backup strategy than bind mounts |

## Utility Scripts

| Script | Purpose |
|--------|---------|
| `generate-config.sh` | Reads YAML config, generates deployment-specific files |
| `push-image.sh` | Pushes images to AWS ECR, Azure ACR, or GCP Artifact Registry |
| `select-cuda-image.sh` | Lists and selects NVIDIA CUDA base images |

## Related Documentation

- [PLAN.md](../PLAN.md) - Strategic design document for deployment phases
- [local-container.md](local-container.md) - Local build/run details
- [docker-compose.md](docker-compose.md) - Docker Compose setup
- [terraform.md](terraform.md) - Terraform deployment
- [kubernetes.md](kubernetes.md) - Kubernetes deployment
- [utilities.md](utilities.md) - Utility script documentation
- [ansible/README.md](../ansible/README.md) - Cloud tools bootstrap
