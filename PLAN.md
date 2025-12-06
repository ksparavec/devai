# Cloud Deployment Extension Plan

## Overview

This plan extends Dev AI Lab to support deployment on AWS, Azure, and Google Cloud Platform. The strategy prioritizes **configuration reuse** and follows a progressive complexity approach:

1. **Phase 1**: Docker Compose (local orchestration foundation)
2. **Phase 2**: Terraform (cloud infrastructure provisioning)
3. **Phase 3**: Kubernetes (portable container orchestration)

## Design Principles

### Shared Configuration Strategy

To avoid duplication, we establish a **layered configuration architecture**:

```
config/
├── common/                      # Shared across ALL deployment methods
│   ├── env.yaml                 # Canonical environment variables
│   ├── ports.yaml               # Port mappings
│   ├── resources.yaml           # CPU/memory defaults
│   └── labels.yaml              # Standard labels/tags
├── secrets.example.yaml         # Template for sensitive values
└── profiles/                    # Environment-specific overrides
    ├── dev.yaml
    ├── staging.yaml
    └── prod.yaml
```

**Configuration Flow**:
```
config/common/*.yaml
        ↓
   Profile overlay (dev/staging/prod)
        ↓
   ┌────┴────┬────────────┐
   ↓         ↓            ↓
Compose   Terraform    Kubernetes
```

### Code Generation Approach

A single `scripts/generate-config.sh` script will:
- Read YAML configuration files
- Generate deployment-specific files:
  - `.env` files for Docker Compose
  - `terraform.tfvars` for Terraform
  - `ConfigMaps` for Kubernetes
- Ensure consistency across all deployment targets

---

## Directory Structure

```
devai/
├── Dockerfile                    # (existing)
├── Dockerfile.gpu                # (existing)
├── Makefile                      # (extended with deploy targets)
├── config/
│   ├── common/
│   │   ├── env.yaml              # Environment variables
│   │   ├── ports.yaml            # Port definitions
│   │   ├── resources.yaml        # Resource limits
│   │   └── labels.yaml           # Tags/labels
│   ├── secrets.example.yaml      # Secrets template
│   └── profiles/
│       ├── dev.yaml
│       ├── staging.yaml
│       └── prod.yaml
├── deploy/
│   ├── compose/                  # Phase 1: Docker Compose
│   │   ├── docker-compose.yml
│   │   ├── docker-compose.gpu.yml
│   │   ├── docker-compose.override.yml.example
│   │   └── .env.example
│   ├── terraform/                # Phase 2: Terraform
│   │   ├── modules/              # Reusable modules
│   │   │   ├── container-registry/
│   │   │   ├── networking/
│   │   │   ├── compute/
│   │   │   └── storage/
│   │   ├── aws/
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   ├── outputs.tf
│   │   │   └── terraform.tfvars.example
│   │   ├── azure/
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   ├── outputs.tf
│   │   │   └── terraform.tfvars.example
│   │   └── gcp/
│   │       ├── main.tf
│   │       ├── variables.tf
│   │       ├── outputs.tf
│   │       └── terraform.tfvars.example
│   └── kubernetes/               # Phase 3: Kubernetes
│       ├── base/                 # Kustomize base
│       │   ├── kustomization.yaml
│       │   ├── namespace.yaml
│       │   ├── deployment.yaml
│       │   ├── service.yaml
│       │   ├── configmap.yaml
│       │   ├── pvc.yaml
│       │   └── ingress.yaml
│       └── overlays/             # Environment overlays
│           ├── dev/
│           ├── staging/
│           ├── prod/
│           ├── aws/              # Cloud-specific patches
│           ├── azure/
│           └── gcp/
└── scripts/
    ├── generate-config.sh        # Config generator
    ├── push-image.sh             # Push to cloud registries
    └── deploy.sh                 # Unified deployment script
```

---

## Phase 1: Docker Compose (Local Orchestration)

### Purpose
- Bridge between current Makefile and cloud deployments
- Enable multi-service local development (devai + ollama)
- Serve as reference implementation for cloud configs

### Files to Create

#### `deploy/compose/docker-compose.yml`
```yaml
services:
  devai:
    build:
      context: ../..
      dockerfile: Dockerfile
    image: ${IMAGE_NAME:-devai-lab}:${IMAGE_TAG:-latest}
    container_name: devai-lab
    ports:
      - "${PORT:-8888}:8888"
    environment:
      - OLLAMA_HOST=${OLLAMA_HOST:-http://ollama:11434}
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY:-}
    volumes:
      - ${HOST_WORK_DIR:-.}:/home/devai/work
      - devai-home:/home/devai/.local
    depends_on:
      - ollama
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama-models:/root/.ollama
    restart: unless-stopped

volumes:
  devai-home:
  ollama-models:
```

#### `deploy/compose/docker-compose.gpu.yml`
Override for GPU support (used with `-f docker-compose.yml -f docker-compose.gpu.yml`).

### Makefile Additions
```makefile
compose-up:       ## Start services with Docker Compose
compose-up-gpu:   ## Start services with GPU support
compose-down:     ## Stop services
compose-logs:     ## View logs
```

---

## Phase 2: Terraform (Cloud Infrastructure)

### Shared Terraform Modules

Located in `deploy/terraform/modules/`, these are cloud-agnostic interfaces:

#### `modules/container-registry/`
- **Purpose**: Create container registry for storing images
- **AWS**: ECR
- **Azure**: ACR  
- **GCP**: Artifact Registry

#### `modules/networking/`
- **Purpose**: VPC/VNet, subnets, security groups
- **AWS**: VPC, Subnets, Security Groups
- **Azure**: VNet, Subnets, NSG
- **GCP**: VPC, Subnets, Firewall Rules

#### `modules/compute/`
- **Purpose**: Container runtime environment
- **AWS**: ECS Fargate (simple) or EKS (advanced)
- **Azure**: Container Instances (simple) or AKS (advanced)
- **GCP**: Cloud Run (simple) or GKE (advanced)

#### `modules/storage/`
- **Purpose**: Persistent storage for notebooks/models
- **AWS**: EFS
- **Azure**: Azure Files
- **GCP**: Filestore

### Cloud-Specific Implementations

Each cloud directory (`aws/`, `azure/`, `gcp/`) contains:

| File | Purpose |
|------|---------|
| `main.tf` | Resource definitions using shared modules |
| `variables.tf` | Input variables (common interface) |
| `outputs.tf` | Output values (URL, registry, etc.) |
| `terraform.tfvars.example` | Example configuration |
| `backend.tf` | Remote state configuration |

### Common Variable Interface

All clouds share the same variable names for consistency:

```hcl
# variables.tf (common across all clouds)
variable "project_name" { default = "devai-lab" }
variable "environment" { default = "dev" }
variable "region" { }
variable "enable_gpu" { default = false }
variable "cpu" { default = 2 }
variable "memory" { default = 4096 }
variable "storage_size_gb" { default = 50 }
variable "enable_https" { default = true }
variable "allowed_cidrs" { default = ["0.0.0.0/0"] }
```

### AWS Implementation (`deploy/terraform/aws/`)

**Services Used**:
- ECR for container registry
- ECS Fargate for serverless containers (or EKS for K8s)
- EFS for persistent storage
- ALB for load balancing
- ACM for SSL certificates
- IAM for permissions

**GPU Support**: ECS on EC2 with GPU instances (p3/g4dn)

### Azure Implementation (`deploy/terraform/azure/`)

**Services Used**:
- ACR for container registry
- Container Instances for simple deployment (or AKS for K8s)
- Azure Files for persistent storage
- Application Gateway for ingress
- Key Vault for secrets
- Managed Identity for authentication

**GPU Support**: Container Instances with GPU SKU or AKS with GPU node pool

### GCP Implementation (`deploy/terraform/gcp/`)

**Services Used**:
- Artifact Registry for container images
- Cloud Run for serverless containers (or GKE for K8s)
- Filestore for persistent storage
- Cloud Load Balancing
- Certificate Manager for SSL
- IAM for permissions

**GPU Support**: GKE with GPU node pool (Cloud Run doesn't support GPU)

### Makefile Additions
```makefile
tf-init-aws:      ## Initialize Terraform for AWS
tf-plan-aws:      ## Plan AWS deployment
tf-apply-aws:     ## Apply AWS deployment
tf-destroy-aws:   ## Destroy AWS resources
# (similar for azure and gcp)
```

---

## Phase 3: Kubernetes (Portable Orchestration)

### Kustomize Structure

Using Kustomize for configuration management without templating:

#### Base Resources (`deploy/kubernetes/base/`)

| File | Purpose |
|------|---------|
| `namespace.yaml` | Dedicated namespace |
| `deployment.yaml` | Pod specification |
| `service.yaml` | ClusterIP service |
| `configmap.yaml` | Environment configuration |
| `secret.yaml` | API keys (template) |
| `pvc.yaml` | Persistent volume claim |
| `ingress.yaml` | Ingress resource |
| `networkpolicy.yaml` | Network isolation |

#### Environment Overlays (`deploy/kubernetes/overlays/`)

```
overlays/
├── dev/
│   ├── kustomization.yaml
│   ├── replicas-patch.yaml      # 1 replica
│   └── resources-patch.yaml     # Lower limits
├── staging/
│   └── ...
├── prod/
│   ├── kustomization.yaml
│   ├── replicas-patch.yaml      # 3 replicas
│   ├── resources-patch.yaml     # Higher limits
│   └── hpa.yaml                 # Autoscaling
├── aws/
│   ├── kustomization.yaml
│   ├── ingress-patch.yaml       # ALB annotations
│   ├── storage-class.yaml       # EFS CSI
│   └── service-account.yaml     # IRSA
├── azure/
│   ├── kustomization.yaml
│   ├── ingress-patch.yaml       # AGIC annotations
│   ├── storage-class.yaml       # Azure Files CSI
│   └── service-account.yaml     # Workload Identity
└── gcp/
    ├── kustomization.yaml
    ├── ingress-patch.yaml       # GCE ingress annotations
    ├── storage-class.yaml       # Filestore CSI
    └── service-account.yaml     # Workload Identity
```

#### GPU Support

GPU overlay applied in combination with environment:
```bash
# Dev + GPU on AWS
kustomize build overlays/dev --load-restrictor=none \
  | kustomize build overlays/aws --load-restrictor=none \
  | kustomize build overlays/gpu
```

Or using Kustomize components:
```yaml
# overlays/aws-dev-gpu/kustomization.yaml
resources:
  - ../dev
components:
  - ../../components/gpu
  - ../../components/aws
```

### Makefile Additions
```makefile
k8s-build:        ## Build Kubernetes manifests
k8s-apply:        ## Apply to current context
k8s-delete:       ## Delete resources
```

---

## Configuration Reuse Matrix

| Configuration | Compose | Terraform | Kubernetes |
|--------------|---------|-----------|------------|
| Image name | `.env` | `tfvars` | `kustomization.yaml` |
| Port | `.env` | `tfvars` | `configmap.yaml` |
| CPU/Memory | `deploy.resources` | `tfvars` | `deployment.yaml` |
| Environment vars | `.env` | `tfvars` → user_data | `configmap.yaml` |
| Secrets | `.env` (local) | Cloud secrets manager | `secret.yaml` / External Secrets |
| Storage | `volumes` | EFS/Files/Filestore | `pvc.yaml` + `storageclass.yaml` |
| Networking | `networks` | VPC module | `ingress.yaml` + `networkpolicy.yaml` |

**Source of Truth**: `config/common/*.yaml` files generate all of the above.

---

## Scripts

### `scripts/generate-config.sh`

Reads `config/` YAML files and generates:
- `deploy/compose/.env`
- `deploy/terraform/{aws,azure,gcp}/terraform.tfvars`
- `deploy/kubernetes/base/configmap.yaml`

### `scripts/push-image.sh`

```bash
./scripts/push-image.sh aws    # Push to ECR
./scripts/push-image.sh azure  # Push to ACR
./scripts/push-image.sh gcp    # Push to Artifact Registry
```

### `scripts/deploy.sh`

Unified deployment interface:
```bash
./scripts/deploy.sh compose up
./scripts/deploy.sh terraform aws apply
./scripts/deploy.sh kubernetes aws-prod apply
```

---

## Implementation Order

### Phase 1: Docker Compose (Foundation)
1. Create `config/` directory structure with YAML schemas
2. Create `deploy/compose/` with docker-compose files
3. Create `scripts/generate-config.sh` for Compose
4. Extend Makefile with compose targets
5. Test local deployment

### Phase 2: Terraform (Cloud Infrastructure)
1. Create shared Terraform modules (`modules/`)
2. Implement AWS provider (`deploy/terraform/aws/`)
3. Implement Azure provider (`deploy/terraform/azure/`)
4. Implement GCP provider (`deploy/terraform/gcp/`)
5. Create `scripts/push-image.sh`
6. Extend `scripts/generate-config.sh` for Terraform
7. Extend Makefile with terraform targets
8. Test each cloud deployment

### Phase 3: Kubernetes (Orchestration)
1. Create Kustomize base (`deploy/kubernetes/base/`)
2. Create environment overlays (`dev/`, `staging/`, `prod/`)
3. Create cloud-specific overlays (`aws/`, `azure/`, `gcp/`)
4. Create GPU component
5. Extend `scripts/generate-config.sh` for Kubernetes
6. Extend Makefile with kubernetes targets
7. Test on each managed Kubernetes service

---

## Security Considerations

### All Phases
- Never commit secrets (`.gitignore` patterns)
- Use environment-specific secret management
- Enable HTTPS by default
- Restrict network access to known CIDRs

### Cloud-Specific
- **AWS**: IAM roles, IRSA for K8s, Security Groups
- **Azure**: Managed Identity, Workload Identity for AKS, NSG
- **GCP**: Service Accounts, Workload Identity for GKE, Firewall Rules

---

## Estimated File Count

| Phase | New Files | Modified Files |
|-------|-----------|----------------|
| Config | 7 | 0 |
| Compose | 5 | 1 (Makefile) |
| Terraform | ~25 | 1 (Makefile) |
| Kubernetes | ~20 | 1 (Makefile) |
| Scripts | 3 | 0 |
| **Total** | **~60** | **1** |

---

## Success Criteria

1. **Docker Compose**: `make compose-up` starts JupyterLab + Ollama locally
2. **AWS**: `make tf-apply-aws` deploys working environment to AWS
3. **Azure**: `make tf-apply-azure` deploys working environment to Azure
4. **GCP**: `make tf-apply-gcp` deploys working environment to GCP
5. **Kubernetes**: `make k8s-apply ENV=prod CLOUD=aws` deploys to EKS
6. **Configuration**: Single edit in `config/` propagates to all targets
