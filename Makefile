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

all: build

build: ## Build the container image (CPU)
	$(CONTAINER_RUNTIME) build \
		--build-arg HTTP_PROXY=$(HTTP_PROXY) \
		--build-arg HTTPS_PROXY=$(HTTPS_PROXY) \
		-t $(IMAGE_NAME) .

build-gpu: ## Build the container image (GPU/CUDA)
	$(CONTAINER_RUNTIME) build \
		--build-arg HTTP_PROXY=$(HTTP_PROXY) \
		--build-arg HTTPS_PROXY=$(HTTPS_PROXY) \
		-f Dockerfile.gpu \
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
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

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
# Terraform targets
# =============================================================================

tf-init-aws: ## Initialize Terraform for AWS
	cd deploy/terraform/aws && terraform init

tf-plan-aws: ## Plan AWS deployment
	cd deploy/terraform/aws && terraform plan

tf-apply-aws: ## Apply AWS deployment
	cd deploy/terraform/aws && terraform apply

tf-destroy-aws: ## Destroy AWS resources
	cd deploy/terraform/aws && terraform destroy

tf-init-azure: ## Initialize Terraform for Azure
	cd deploy/terraform/azure && terraform init

tf-plan-azure: ## Plan Azure deployment
	cd deploy/terraform/azure && terraform plan

tf-apply-azure: ## Apply Azure deployment
	cd deploy/terraform/azure && terraform apply

tf-destroy-azure: ## Destroy Azure resources
	cd deploy/terraform/azure && terraform destroy

tf-init-gcp: ## Initialize Terraform for GCP
	cd deploy/terraform/gcp && terraform init

tf-plan-gcp: ## Plan GCP deployment
	cd deploy/terraform/gcp && terraform plan

tf-apply-gcp: ## Apply GCP deployment
	cd deploy/terraform/gcp && terraform apply

tf-destroy-gcp: ## Destroy GCP resources
	cd deploy/terraform/gcp && terraform destroy

# =============================================================================
# Kubernetes targets
# =============================================================================

KUSTOMIZE_OVERLAY ?= dev
CLOUD ?= aws

k8s-build: ## Build Kubernetes manifests (usage: make k8s-build KUSTOMIZE_OVERLAY=prod CLOUD=aws)
	@echo "Building manifests for overlay: $(KUSTOMIZE_OVERLAY), cloud: $(CLOUD)"
	kustomize build deploy/kubernetes/overlays/$(KUSTOMIZE_OVERLAY)

k8s-apply: ## Apply Kubernetes manifests to current context
	kustomize build deploy/kubernetes/overlays/$(KUSTOMIZE_OVERLAY) | kubectl apply -f -

k8s-delete: ## Delete Kubernetes resources
	kustomize build deploy/kubernetes/overlays/$(KUSTOMIZE_OVERLAY) | kubectl delete -f -
