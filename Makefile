# Configuration
-include .env

# Defaults
CONTAINER_RUNTIME ?= podman
IMAGE_NAME ?= devai-lab
CONTAINER_USER ?= devai
PORT ?= 8888
HOST_IP ?= $(shell hostname -I | awk '{print $$1}')
HOST_HOME_DIR ?=
OLLAMA_HOST ?= http://host.containers.internal:11434

# Cloud tools configuration (bootstrapped venv)
DEVAI_PYTHON_VERSION ?= 3.12
DEVAI_VENV_DIR ?= $(HOME)/.local/devai-venv
DEVAI_BIN_DIR ?= $(HOME)/.local/bin
INSTALL_RUNTIME ?= both

# Tool binaries (use bootstrapped venv)
TERRAFORM ?= $(DEVAI_BIN_DIR)/terraform
KUBECTL ?= $(DEVAI_BIN_DIR)/kubectl
KUSTOMIZE ?= $(DEVAI_BIN_DIR)/kustomize
AWS ?= $(DEVAI_BIN_DIR)/aws
GCLOUD ?= $(DEVAI_BIN_DIR)/gcloud
AZ ?= $(DEVAI_BIN_DIR)/az

# GPU build settings
GPU_BASE_IMAGE ?= docker.io/nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

HOME_MOUNT_ARG =
ifneq ($(HOST_HOME_DIR),)
	HOME_MOUNT_ARG = -v "$$(readlink -f $(HOST_HOME_DIR))":/home/$(CONTAINER_USER)
endif

RUN_FLAGS =
ifeq ($(findstring podman,$(CONTAINER_RUNTIME)),podman)
	RUN_FLAGS += --userns=keep-id:uid=1000,gid=1000
endif

# GPU runtime flags
GPU_FLAGS =
ifeq ($(findstring podman,$(CONTAINER_RUNTIME)),podman)
	GPU_FLAGS += --device nvidia.com/gpu=all --security-opt=label=disable
else
	GPU_FLAGS += --gpus all
endif

# Compose settings
COMPOSE_DIR = deploy/compose
COMPOSE_FILE = $(COMPOSE_DIR)/docker-compose.yml
COMPOSE_GPU_FILE = $(COMPOSE_DIR)/docker-compose.gpu.yml
PROFILE ?= dev

.PHONY: all build build-gpu run run-gpu clean clean-gpu prune shell help
.PHONY: compose-up compose-up-gpu compose-down compose-logs compose-build compose-ps
.PHONY: config-generate
.PHONY: tf-init-aws tf-plan-aws tf-apply-aws tf-destroy-aws
.PHONY: tf-init-azure tf-plan-azure tf-apply-azure tf-destroy-azure
.PHONY: tf-init-gcp tf-plan-gcp tf-apply-gcp tf-destroy-gcp
.PHONY: k8s-build k8s-apply k8s-delete
.PHONY: setup-build-env setup-cloud-tools setup-cloud-tools-aws setup-cloud-tools-azure setup-cloud-tools-gcp
.PHONY: setup-cloud-tools-terraform setup-cloud-tools-k8s check-cloud-tools
.PHONY: setup-runtime setup-runtime-podman setup-runtime-docker

all: help

build: ## Build the container image (CPU)
	$(CONTAINER_RUNTIME) build \
		--build-arg HTTP_PROXY=$(HTTP_PROXY) \
		--build-arg HTTPS_PROXY=$(HTTPS_PROXY) \
		-t $(IMAGE_NAME) .

build-gpu: ## Build the container image (GPU/CUDA)
	$(CONTAINER_RUNTIME) build \
		--build-arg HTTP_PROXY=$(HTTP_PROXY) \
		--build-arg HTTPS_PROXY=$(HTTPS_PROXY) \
		--build-arg BASE_IMAGE=$(GPU_BASE_IMAGE) \
		--build-arg GPU_BUILD=true \
		-t $(IMAGE_NAME)-gpu .

run: ## Run the container (CPU)
	@if [ -n "$(HOST_HOME_DIR)" ]; then mkdir -p "$(HOST_HOME_DIR)"; fi
	@echo "Starting $(IMAGE_NAME)..."
	@echo "Access JupyterLab at http://$(HOST_IP):$(PORT)/lab?token=..."
	$(CONTAINER_RUNTIME) run -it --rm \
		--name $(IMAGE_NAME) \
		$(RUN_FLAGS) \
		--add-host=host.containers.internal:host-gateway \
		-e HTTP_PROXY=$(HTTP_PROXY) \
		-e HTTPS_PROXY=$(HTTPS_PROXY) \
		-e OLLAMA_HOST=$(OLLAMA_HOST) \
		-e USER_ID=$(shell id -u) \
		-e GROUP_ID=$(shell id -g) \
		-e CONTAINER_USER=$(CONTAINER_USER) \
		-e HOST_IP=$(HOST_IP) \
		-e PORT=$(PORT) \
		-p 0.0.0.0:$(PORT):8888 \
		$(HOME_MOUNT_ARG) \
		-v "$$(readlink -f $(HOST_WORK_DIR))":/home/$(CONTAINER_USER)/work \
		$(IMAGE_NAME)

run-gpu: ## Run the container (GPU/CUDA)
	@if [ -n "$(HOST_HOME_DIR)" ]; then mkdir -p "$(HOST_HOME_DIR)"; fi
	@echo "Starting $(IMAGE_NAME)-gpu with GPU support..."
	@echo "Access JupyterLab at http://$(HOST_IP):$(PORT)/lab?token=..."
	$(CONTAINER_RUNTIME) run -it --rm \
		--name $(IMAGE_NAME)-gpu \
		$(RUN_FLAGS) \
		$(GPU_FLAGS) \
		--add-host=host.containers.internal:host-gateway \
		-e HTTP_PROXY=$(HTTP_PROXY) \
		-e HTTPS_PROXY=$(HTTPS_PROXY) \
		-e OLLAMA_HOST=$(OLLAMA_HOST) \
		-e USER_ID=$(shell id -u) \
		-e GROUP_ID=$(shell id -g) \
		-e CONTAINER_USER=$(CONTAINER_USER) \
		-e HOST_IP=$(HOST_IP) \
		-e PORT=$(PORT) \
		-p 0.0.0.0:$(PORT):8888 \
		$(HOME_MOUNT_ARG) \
		-v "$$(readlink -f $(HOST_WORK_DIR))":/home/$(CONTAINER_USER)/work \
		$(IMAGE_NAME)-gpu

shell: ## Start an interactive shell in the container
	$(CONTAINER_RUNTIME) run -it --rm \
		--name $(IMAGE_NAME)-shell \
		$(RUN_FLAGS) \
		--add-host=host.containers.internal:host-gateway \
		-e OLLAMA_HOST=$(OLLAMA_HOST) \
		-e USER_ID=$(shell id -u) \
		-e GROUP_ID=$(shell id -g) \
		-e CONTAINER_USER=$(CONTAINER_USER) \
		$(HOME_MOUNT_ARG) \
		-v "$$(readlink -f $(HOST_WORK_DIR))":/home/$(CONTAINER_USER)/work \
		$(IMAGE_NAME) /bin/bash

clean: ## Remove the container image (CPU)
	$(CONTAINER_RUNTIME) rmi $(IMAGE_NAME)

clean-gpu: ## Remove the container image (GPU)
	$(CONTAINER_RUNTIME) rmi $(IMAGE_NAME)-gpu

prune: ## Clean up dangling images only (keeps tagged images and volumes)
	$(CONTAINER_RUNTIME) image prune -f

help: ## Show this help message
	@echo ""
	@echo "\033[1mDevAI Lab - Containerized AI Development Environment\033[0m"
	@echo ""
	@echo "\033[1;33m1. SETUP BUILD ENVIRONMENT\033[0m (install container runtimes and cloud tools)"
	@echo "   \033[36msetup-build-env\033[0m            Install runtimes + all cloud CLIs (Recommended)"
	@echo "   \033[36msetup-cloud-tools\033[0m          Install cloud CLIs only (no runtimes)"
	@echo "   \033[36mcheck-cloud-tools\033[0m          Verify installed tool versions"
	@echo "   \033[2m(See README.md Appendix A for selective installation options)\033[0m"
	@echo ""
	@echo "\033[1;33m2. BUILD CONTAINER IMAGES\033[0m"
	@echo "   \033[36mbuild\033[0m                      Build container image (CPU)"
	@echo "   \033[36mbuild-gpu\033[0m                  Build container image (GPU/CUDA)"
	@echo "   \033[36mcompose-build\033[0m              Build images with Docker Compose"
	@echo ""
	@echo "\033[1;33m3. DEPLOY TO TARGET RUNTIME\033[0m"
	@echo "   \033[1mTerraform (cloud infrastructure):\033[0m"
	@echo "   \033[36mtf-init-{aws,azure,gcp}\033[0m    Initialize Terraform for target cloud"
	@echo "   \033[36mtf-plan-{aws,azure,gcp}\033[0m    Plan deployment changes"
	@echo "   \033[36mtf-apply-{aws,azure,gcp}\033[0m   Apply deployment to cloud"
	@echo "   \033[36mtf-destroy-{aws,azure,gcp}\033[0m Destroy cloud resources"
	@echo ""
	@echo "   \033[1mKubernetes:\033[0m"
	@echo "   \033[36mk8s-build\033[0m                  Build Kubernetes manifests"
	@echo "   \033[36mk8s-apply\033[0m                  Apply manifests to cluster"
	@echo "   \033[36mk8s-delete\033[0m                 Delete Kubernetes resources"
	@echo ""
	@echo "\033[1;33m4. RUN LOCALLY\033[0m"
	@echo "   \033[36mrun\033[0m                        Run container with JupyterLab (CPU)"
	@echo "   \033[36mrun-gpu\033[0m                    Run container with JupyterLab (GPU)"
	@echo "   \033[36mshell\033[0m                      Start interactive shell in container"
	@echo "   \033[36mcompose-up\033[0m                 Start services with Docker Compose"
	@echo "   \033[36mcompose-up-gpu\033[0m             Start services with GPU support"
	@echo "   \033[36mcompose-down\033[0m               Stop and remove containers"
	@echo "   \033[36mcompose-logs\033[0m               View container logs"
	@echo "   \033[36mcompose-ps\033[0m                 Show running containers"
	@echo ""
	@echo "\033[1;33m5. MAINTENANCE\033[0m"
	@echo "   \033[36mclean\033[0m                      Remove container image (CPU)"
	@echo "   \033[36mclean-gpu\033[0m                  Remove container image (GPU)"
	@echo "   \033[36mprune\033[0m                      Clean up dangling images"
	@echo "   \033[36mconfig-generate\033[0m            Generate config from YAML profiles"
	@echo ""
	@echo "\033[1mConfiguration:\033[0m"
	@echo "   Copy .env.example to .env and adjust settings before running."
	@echo "   Tool binaries: \033[33m$(DEVAI_BIN_DIR)\033[0m"
	@echo ""

# =============================================================================
# Docker Compose targets
# =============================================================================

config-generate: ## Generate config files from YAML (usage: make config-generate PROFILE=dev)
	@./scripts/generate-config.sh all $(PROFILE)

compose-build: ## Build images with Docker Compose
	docker compose -f $(COMPOSE_FILE) build

compose-up: ## Start services with Docker Compose
	@mkdir -p $(COMPOSE_DIR)/work
	docker compose -f $(COMPOSE_FILE) up -d
	@echo "JupyterLab starting at http://localhost:$(PORT)"
	@echo "Run 'make compose-logs' to see startup logs and token"

compose-up-gpu: ## Start services with GPU support
	@mkdir -p $(COMPOSE_DIR)/work
	docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_GPU_FILE) up -d
	@echo "JupyterLab (GPU) starting at http://localhost:$(PORT)"

compose-down: ## Stop and remove containers
	docker compose -f $(COMPOSE_FILE) down

compose-logs: ## View container logs
	docker compose -f $(COMPOSE_FILE) logs -f

compose-ps: ## Show running containers
	docker compose -f $(COMPOSE_FILE) ps

# =============================================================================
# Terraform targets (using bootstrapped venv tools)
# =============================================================================

tf-init-aws: ## Initialize Terraform for AWS
	cd deploy/terraform/aws && $(TERRAFORM) init

tf-plan-aws: ## Plan AWS deployment
	cd deploy/terraform/aws && $(TERRAFORM) plan

tf-apply-aws: ## Apply AWS deployment
	cd deploy/terraform/aws && $(TERRAFORM) apply

tf-destroy-aws: ## Destroy AWS resources
	cd deploy/terraform/aws && $(TERRAFORM) destroy

tf-init-azure: ## Initialize Terraform for Azure
	cd deploy/terraform/azure && $(TERRAFORM) init

tf-plan-azure: ## Plan Azure deployment
	cd deploy/terraform/azure && $(TERRAFORM) plan

tf-apply-azure: ## Apply Azure deployment
	cd deploy/terraform/azure && $(TERRAFORM) apply

tf-destroy-azure: ## Destroy Azure resources
	cd deploy/terraform/azure && $(TERRAFORM) destroy

tf-init-gcp: ## Initialize Terraform for GCP
	cd deploy/terraform/gcp && $(TERRAFORM) init

tf-plan-gcp: ## Plan GCP deployment
	cd deploy/terraform/gcp && $(TERRAFORM) plan

tf-apply-gcp: ## Apply GCP deployment
	cd deploy/terraform/gcp && $(TERRAFORM) apply

tf-destroy-gcp: ## Destroy GCP resources
	cd deploy/terraform/gcp && $(TERRAFORM) destroy

# =============================================================================
# Kubernetes targets (using bootstrapped venv tools)
# =============================================================================

KUSTOMIZE_OVERLAY ?= dev
CLOUD ?= aws

k8s-build: ## Build Kubernetes manifests (usage: make k8s-build KUSTOMIZE_OVERLAY=prod CLOUD=aws)
	@echo "Building manifests for overlay: $(KUSTOMIZE_OVERLAY), cloud: $(CLOUD)"
	$(KUSTOMIZE) build deploy/kubernetes/overlays/$(KUSTOMIZE_OVERLAY)

k8s-apply: ## Apply Kubernetes manifests to current context
	$(KUSTOMIZE) build deploy/kubernetes/overlays/$(KUSTOMIZE_OVERLAY) | $(KUBECTL) apply -f -

k8s-delete: ## Delete Kubernetes resources
	$(KUSTOMIZE) build deploy/kubernetes/overlays/$(KUSTOMIZE_OVERLAY) | $(KUBECTL) delete -f -

# =============================================================================
# Setup / Prerequisites targets
# =============================================================================

ANSIBLE_ARGS ?=

setup-build-env: ## Install container runtimes and cloud management CLIs (recommended)
	@echo "Installing container runtimes and cloud management tools via bootstrap..."
	@echo "  Python version:    $(DEVAI_PYTHON_VERSION)"
	@echo "  Venv directory:    $(DEVAI_VENV_DIR)"
	@echo "  Bin directory:     $(DEVAI_BIN_DIR)"
	@echo "  Install runtime:   $(INSTALL_RUNTIME)"
	DEVAI_PYTHON_VERSION=$(DEVAI_PYTHON_VERSION) \
	DEVAI_VENV_DIR=$(DEVAI_VENV_DIR) \
	DEVAI_BIN_DIR=$(DEVAI_BIN_DIR) \
	./ansible/bootstrap.sh --install-runtime $(INSTALL_RUNTIME) $(ANSIBLE_ARGS)

setup-cloud-tools: ## Install cloud management CLIs only (no runtimes)
	$(MAKE) setup-build-env INSTALL_RUNTIME=none

setup-cloud-tools-aws: ## Install only AWS CLI (no runtimes)
	$(MAKE) setup-build-env INSTALL_RUNTIME=none ANSIBLE_ARGS="-- --tags aws"

setup-cloud-tools-azure: ## Install only Azure CLI (no runtimes)
	$(MAKE) setup-build-env INSTALL_RUNTIME=none ANSIBLE_ARGS="-- --tags azure"

setup-cloud-tools-gcp: ## Install only Google Cloud CLI (no runtimes)
	$(MAKE) setup-build-env INSTALL_RUNTIME=none ANSIBLE_ARGS="-- --tags gcp"

setup-cloud-tools-terraform: ## Install only Terraform (no runtimes)
	$(MAKE) setup-build-env INSTALL_RUNTIME=none ANSIBLE_ARGS="-- --tags terraform"

setup-cloud-tools-k8s: ## Install only Kubernetes tools (no runtimes)
	$(MAKE) setup-build-env INSTALL_RUNTIME=none ANSIBLE_ARGS="-- --tags kubernetes"

setup-runtime: ## Install container runtimes only (podman and docker)
	$(MAKE) setup-build-env INSTALL_RUNTIME=both ANSIBLE_ARGS="-- --tags never"

setup-runtime-podman: ## Install only Podman
	$(MAKE) setup-build-env INSTALL_RUNTIME=podman ANSIBLE_ARGS="-- --tags never"

setup-runtime-docker: ## Install only Docker
	$(MAKE) setup-build-env INSTALL_RUNTIME=docker ANSIBLE_ARGS="-- --tags never"

check-cloud-tools: ## Check installed versions of cloud tools
	@echo "=== Cloud Tools Status ==="
	@echo "Checking in: $(DEVAI_BIN_DIR)"
	@echo ""
	@echo "AWS CLI:"
	@$(AWS) --version 2>/dev/null || echo "  Not installed"
	@echo "Azure CLI:"
	@$(AZ) version --output tsv 2>/dev/null | head -1 || echo "  Not installed"
	@echo "Google Cloud CLI:"
	@$(GCLOUD) version 2>/dev/null | head -1 || echo "  Not installed"
	@echo "Terraform:"
	@$(TERRAFORM) version 2>/dev/null | head -1 || echo "  Not installed"
	@echo "kubectl:"
	@$(KUBECTL) version --client 2>/dev/null | head -1 || echo "  Not installed"
	@echo "kustomize:"
	@$(KUSTOMIZE) version 2>/dev/null || echo "  Not installed"
