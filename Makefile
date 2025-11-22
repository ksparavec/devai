# Configuration
-include .env

# Defaults
CONTAINER_RUNTIME ?= podman
IMAGE_NAME ?= gemini-lab
CONTAINER_USER ?= devai
PORT ?= 8888
HOST_IP ?= $(shell hostname -I | awk '{print $$1}')
HOST_HOME_DIR ?=

HOME_MOUNT_ARG =
ifneq ($(HOST_HOME_DIR),)
	HOME_MOUNT_ARG = -v "$$(readlink -f $(HOST_HOME_DIR))":/home/$(CONTAINER_USER)
endif

RUN_FLAGS =
ifeq ($(findstring podman,$(CONTAINER_RUNTIME)),podman)
	RUN_FLAGS += --userns=keep-id:uid=1000,gid=1000
endif

.PHONY: all build run clean help

all: build

build: ## Build the docker image
	$(CONTAINER_RUNTIME) build \
		--build-arg HTTP_PROXY=$(HTTP_PROXY) \
		--build-arg HTTPS_PROXY=$(HTTPS_PROXY) \
		-t $(IMAGE_NAME) .

run: ## Run the container
	@if [ -n "$(HOST_HOME_DIR)" ]; then mkdir -p "$(HOST_HOME_DIR)"; fi
	@echo "Starting $(IMAGE_NAME)..."
	@echo "Access JupyterLab at http://$(HOST_IP):$(PORT)/lab?token=..."
	$(CONTAINER_RUNTIME) run -it --rm \
		--name $(IMAGE_NAME) \
		$(RUN_FLAGS) \
		-e HTTP_PROXY=$(HTTP_PROXY) \
		-e HTTPS_PROXY=$(HTTPS_PROXY) \
		-e USER_ID=$(shell id -u) \
		-e GROUP_ID=$(shell id -g) \
		-e CONTAINER_USER=$(CONTAINER_USER) \
		-e HOST_IP=$(HOST_IP) \
		-e PORT=$(PORT) \
		-p 0.0.0.0:$(PORT):8888 \
		$(HOME_MOUNT_ARG) \
		-v "$$(readlink -f $(HOST_WORK_DIR))":/home/$(CONTAINER_USER)/work \
		$(IMAGE_NAME)

clean: ## Remove the docker image
	$(CONTAINER_RUNTIME) rmi $(IMAGE_NAME)

help: ## Show this help message
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
