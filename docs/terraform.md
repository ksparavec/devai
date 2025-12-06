# Terraform Targets

These targets deploy Dev AI Lab to cloud providers using Terraform infrastructure as code.

## Supported Clouds

| Cloud | Service | Registry | Storage |
|-------|---------|----------|---------|
| AWS | ECS Fargate | ECR | EFS |
| Azure | Container Instances | ACR | Azure Files |
| GCP | Cloud Run | Artifact Registry | Filestore |

---

## AWS Targets

### `make tf-init-aws`

Initialize Terraform for AWS.

```bash
make tf-init-aws
```

**What it does:**
- Downloads AWS provider
- Initializes backend (local by default)
- Prepares workspace

---

### `make tf-plan-aws`

Preview AWS deployment changes.

```bash
make tf-plan-aws
```

**What it does:**
- Shows resources to be created/modified/destroyed
- Validates configuration
- Does not make any changes

---

### `make tf-apply-aws`

Deploy to AWS.

```bash
make tf-apply-aws
```

**What it does:**
- Creates VPC with public subnets
- Creates ECR repository
- Creates ECS cluster and service
- Sets up Application Load Balancer
- Configures IAM roles and security groups

**Resources created:**
- VPC, Subnets, Internet Gateway
- ECR Repository
- ECS Cluster, Task Definition, Service
- Application Load Balancer
- CloudWatch Log Group
- IAM Roles (execution, task)
- Security Groups

---

### `make tf-destroy-aws`

Destroy all AWS resources.

```bash
make tf-destroy-aws
```

**Warning:** This permanently deletes all resources.

---

## Azure Targets

### `make tf-init-azure`

Initialize Terraform for Azure.

```bash
make tf-init-azure
```

---

### `make tf-plan-azure`

Preview Azure deployment changes.

```bash
make tf-plan-azure
```

---

### `make tf-apply-azure`

Deploy to Azure.

```bash
make tf-apply-azure
```

**What it does:**
- Creates Resource Group
- Creates Virtual Network and Subnet
- Creates Azure Container Registry
- Deploys Container Instance
- Sets up Log Analytics

**Resources created:**
- Resource Group
- VNet, Subnet, NSG
- Container Registry (ACR)
- Container Instance
- Log Analytics Workspace

---

### `make tf-destroy-azure`

Destroy all Azure resources.

```bash
make tf-destroy-azure
```

---

## GCP Targets

### `make tf-init-gcp`

Initialize Terraform for GCP.

```bash
make tf-init-gcp
```

---

### `make tf-plan-gcp`

Preview GCP deployment changes.

```bash
make tf-plan-gcp
```

---

### `make tf-apply-gcp`

Deploy to GCP.

```bash
make tf-apply-gcp
```

**What it does:**
- Enables required APIs
- Creates Artifact Registry repository
- Creates Service Account
- Deploys Cloud Run service
- Configures IAM for public access

**Resources created:**
- Artifact Registry Repository
- Service Account
- Cloud Run Service
- IAM bindings

---

### `make tf-destroy-gcp`

Destroy all GCP resources.

```bash
make tf-destroy-gcp
```

---

## Configuration

### AWS Configuration

```bash
# Copy example
cp deploy/terraform/aws/terraform.tfvars.example deploy/terraform/aws/terraform.tfvars
```

Edit `deploy/terraform/aws/terraform.tfvars`:

```hcl
# Required
region = "us-east-1"

# Project
project_name = "devai-lab"
environment  = "dev"

# Resources
cpu             = 2      # vCPUs
memory          = 4096   # MB
storage_size_gb = 50
replicas        = 1

# Features
enable_gpu   = false
enable_https = false

# Network
allowed_cidrs = ["0.0.0.0/0"]  # Restrict for production
```

### Azure Configuration

Edit `deploy/terraform/azure/terraform.tfvars`:

```hcl
# Required
region = "eastus"

# Project
project_name = "devai-lab"
environment  = "dev"

# Resources
cpu             = 2
memory          = 4096
storage_size_gb = 50
```

### GCP Configuration

Edit `deploy/terraform/gcp/terraform.tfvars`:

```hcl
# Required
project_id = "your-gcp-project-id"
region     = "us-central1"

# Project
project_name = "devai-lab"
environment  = "dev"

# Resources
cpu             = 2
memory          = 4096
storage_size_gb = 50
```

---

## Common Workflows

### Deploy to AWS

```bash
# 1. Configure AWS credentials
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
# Or use: aws configure

# 2. Copy and edit configuration
cp deploy/terraform/aws/terraform.tfvars.example deploy/terraform/aws/terraform.tfvars
vim deploy/terraform/aws/terraform.tfvars

# 3. Initialize and deploy
make tf-init-aws
make tf-plan-aws
make tf-apply-aws

# 4. Push container image
./scripts/push-image.sh aws

# 5. Get service URL
cd deploy/terraform/aws && terraform output service_url
```

### Deploy to Azure

```bash
# 1. Login to Azure
az login

# 2. Configure
cp deploy/terraform/azure/terraform.tfvars.example deploy/terraform/azure/terraform.tfvars
vim deploy/terraform/azure/terraform.tfvars

# 3. Deploy
make tf-init-azure
make tf-plan-azure
make tf-apply-azure

# 4. Push image
./scripts/push-image.sh azure

# 5. Get URL
cd deploy/terraform/azure && terraform output service_url
```

### Deploy to GCP

```bash
# 1. Authenticate
gcloud auth application-default login

# 2. Configure
cp deploy/terraform/gcp/terraform.tfvars.example deploy/terraform/gcp/terraform.tfvars
vim deploy/terraform/gcp/terraform.tfvars

# 3. Deploy
make tf-init-gcp
make tf-plan-gcp
make tf-apply-gcp

# 4. Push image
./scripts/push-image.sh gcp

# 5. Get URL
cd deploy/terraform/gcp && terraform output service_url
```

---

## Pushing Container Images

After infrastructure is deployed, push your container image:

```bash
# AWS
./scripts/push-image.sh aws

# Azure
./scripts/push-image.sh azure

# GCP
./scripts/push-image.sh gcp

# GPU image
./scripts/push-image.sh aws --gpu
```

---

## Remote State (Production)

For team environments, configure remote state backends.

### AWS S3 Backend

Create `deploy/terraform/aws/backend.tf`:

```hcl
terraform {
  backend "s3" {
    bucket         = "your-terraform-state-bucket"
    key            = "devai-lab/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}
```

### Azure Storage Backend

Create `deploy/terraform/azure/backend.tf`:

```hcl
terraform {
  backend "azurerm" {
    resource_group_name  = "terraform-state-rg"
    storage_account_name = "tfstatestorage"
    container_name       = "tfstate"
    key                  = "devai-lab.tfstate"
  }
}
```

### GCP Cloud Storage Backend

Create `deploy/terraform/gcp/backend.tf`:

```hcl
terraform {
  backend "gcs" {
    bucket = "your-terraform-state-bucket"
    prefix = "devai-lab"
  }
}
```

---

## Troubleshooting

### AWS: Task Failed to Start

```bash
# Check ECS service events
aws ecs describe-services --cluster devai-lab-dev-cluster --services devai-lab-dev-service

# Check CloudWatch logs
aws logs tail /ecs/devai-lab-dev --follow
```

### Azure: Container Instance Failed

```bash
# Check container logs
az container logs --resource-group devai-lab-dev-rg --name devai-lab-dev-aci

# Check events
az container show --resource-group devai-lab-dev-rg --name devai-lab-dev-aci --query instanceView.events
```

### GCP: Cloud Run Service Unhealthy

```bash
# Check logs
gcloud run services logs read devai-lab-dev --region us-central1

# Check service status
gcloud run services describe devai-lab-dev --region us-central1
```

### Image Not Found

Ensure you've pushed the image after infrastructure deployment:

```bash
./scripts/push-image.sh <cloud>
```

Then update the service to pull the new image.
