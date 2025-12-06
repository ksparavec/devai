# Utility Targets and Scripts

These targets and scripts support configuration management and deployment workflows.

## Make Targets

### `make config-generate`

Generate deployment configuration files from shared YAML sources.

```bash
make config-generate
```

**With profile:**

```bash
make config-generate PROFILE=prod
```

**What it does:**
- Reads `config/common/*.yaml` and `config/profiles/*.yaml`
- Generates:
  - `deploy/compose/.env`
  - `deploy/terraform/*/terraform.tfvars`
  - `deploy/kubernetes/base/configmap.yaml`

**Profiles available:**
- `dev` - Development settings
- `staging` - Staging settings
- `prod` - Production settings

---

### `make help`

Show all available Make targets.

```bash
make help
```

**Output:**
```
build                          Build the container image (CPU)
build-gpu                      Build the container image (GPU/CUDA)
clean                          Remove the container image (CPU)
clean-gpu                      Remove the container image (GPU)
compose-build                  Build images with Docker Compose
compose-down                   Stop and remove containers
compose-logs                   View container logs
compose-ps                     Show running containers
compose-up                     Start services with Docker Compose
compose-up-gpu                 Start services with GPU support
config-generate                Generate config files from YAML
...
```

---

## Scripts

### `scripts/generate-config.sh`

Generate configuration files from YAML sources.

```bash
./scripts/generate-config.sh [target] [profile]
```

**Arguments:**
- `target` - `compose`, `terraform`, `kubernetes`, or `all` (default: `all`)
- `profile` - `dev`, `staging`, or `prod` (default: `dev`)

**Examples:**

```bash
# Generate all configs for dev
./scripts/generate-config.sh all dev

# Generate only Compose config for production
./scripts/generate-config.sh compose prod

# Generate only Terraform configs
./scripts/generate-config.sh terraform staging
```

**Requirements:**
- `yq` - YAML processor (install: `pip install yq` or `brew install yq`)

---

### `scripts/push-image.sh`

Build and push container image to cloud registries.

```bash
./scripts/push-image.sh [cloud] [--gpu]
```

**Arguments:**
- `cloud` - `aws`, `azure`, or `gcp`
- `--gpu` - Build and push GPU image

**Examples:**

```bash
# Push to AWS ECR
./scripts/push-image.sh aws

# Push GPU image to Azure ACR
./scripts/push-image.sh azure --gpu

# Push to GCP Artifact Registry
./scripts/push-image.sh gcp
```

**Environment variables:**
- `IMAGE_TAG` - Tag for the image (default: `latest`)
- `PROJECT_NAME` - Image name (default: `devai-lab`)

**Prerequisites:**
- Cloud infrastructure must be deployed first (`make tf-apply-*`)
- Cloud CLI must be authenticated (`aws`, `az`, or `gcloud`)

---

### `scripts/select-cuda-image.sh`

Utility to select NVIDIA CUDA base image version.

```bash
./scripts/select-cuda-image.sh
```

**What it does:**
- Lists available CUDA base images
- Helps select appropriate version for GPU builds

---

## Configuration System

### Directory Structure

```
config/
├── common/                    # Shared across all deployments
│   ├── env.yaml              # Environment variables
│   ├── ports.yaml            # Port mappings
│   ├── resources.yaml        # CPU/memory defaults
│   └── labels.yaml           # Resource labels/tags
├── profiles/                  # Environment-specific
│   ├── dev.yaml
│   ├── staging.yaml
│   └── prod.yaml
└── secrets.example.yaml      # Secrets template
```

### Configuration Flow

```
config/common/*.yaml
        │
        ▼
   Profile overlay (dev/staging/prod)
        │
        ▼
   ┌────┴────┬────────────┐
   │         │            │
   ▼         ▼            ▼
Compose   Terraform    Kubernetes
 .env     .tfvars     ConfigMap
```

### Adding New Configuration

1. Add to common config:

```yaml
# config/common/env.yaml
app:
  name: devai-lab
  new_setting: value
```

2. Add profile overrides if needed:

```yaml
# config/profiles/prod.yaml
app:
  new_setting: production-value
```

3. Update generator script to use the new setting:

```bash
# scripts/generate-config.sh
local new_setting=$(yaml_get "$CONFIG_DIR/common/env.yaml" ".app.new_setting" "default")
```

4. Regenerate configs:

```bash
make config-generate PROFILE=prod
```

---

## Environment Variables Reference

### Common Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `IMAGE_NAME` | Container image name | `devai-lab` |
| `IMAGE_TAG` | Container image tag | `latest` |
| `PORT` | JupyterLab port | `8888` |
| `CONTAINER_USER` | User inside container | `devai` |
| `OLLAMA_HOST` | Ollama server URL | varies |

### Cloud Variables

| Variable | Description |
|----------|-------------|
| `AWS_REGION` | AWS region |
| `AWS_ACCESS_KEY_ID` | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials |
| `ARM_SUBSCRIPTION_ID` | Azure subscription |
| `GOOGLE_PROJECT` | GCP project ID |

### API Keys

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` | Google AI API key |

---

## Workflow Examples

### Setting Up a New Environment

```bash
# 1. Generate configs for the environment
make config-generate PROFILE=staging

# 2. Review generated files
cat deploy/compose/.env
cat deploy/terraform/aws/terraform.tfvars

# 3. Add secrets (don't commit!)
vim deploy/compose/.env  # Add API keys
vim deploy/terraform/aws/terraform.tfvars  # Add region, etc.
```

### Updating Configuration Across All Targets

```bash
# 1. Edit the source config
vim config/common/resources.yaml

# 2. Regenerate all targets
make config-generate PROFILE=prod

# 3. Review changes
git diff deploy/

# 4. Deploy updates
make compose-up  # or tf-apply-*, k8s-apply
```

### CI/CD Integration

```bash
#!/bin/bash
# Example CI script

# Set environment from CI variables
export PROFILE="${CI_ENVIRONMENT_NAME:-dev}"
export IMAGE_TAG="${CI_COMMIT_SHA:-latest}"

# Generate configs
./scripts/generate-config.sh all "$PROFILE"

# Build and push
docker build -t devai-lab:$IMAGE_TAG .
./scripts/push-image.sh "$CLOUD"

# Deploy
make tf-apply-"$CLOUD"
```
