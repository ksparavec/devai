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

.PHONY: all build build-gpu run run-gpu clean clean-gpu prune shell help

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
