# Configuration
-include .env

# Defaults (override via .env)
CONTAINER_RUNTIME ?= podman
LAB_PORT ?= 8888
WEBUI_PORT ?= 8443
HOST_WORK_DIR ?= .
HOST_HOME_DIR ?= $(HOME)
HOME_VOLUME ?= $(HOME)/devai-home
APT_PROXY ?=
NO_PROXY ?=

# Derived (not configurable)
COMPOSE = $(CONTAINER_RUNTIME) compose
IMAGE_NAME = devai-lab-cpu
IMAGE_NAME_GPU = devai-lab-gpu
BASE_IMAGE_NAME = devai-base-cpu
BASE_IMAGE_NAME_GPU = devai-base-gpu
CONTAINER_USER = devai
PYTHON_VERSION = 3.13
HOST_IP = $(shell hostname -I | awk '{print $$1}')
OLLAMA_CONTAINER = devai-ollama
OLLAMA_EXEC = $(CONTAINER_RUNTIME) exec $(OLLAMA_CONTAINER) ollama
DEVAI_NETWORK = devai-net
CACHE_DIR = /var/cache/devai
CACHE_COMPOSE = deploy/docker-compose.yaml
INFERENCE_CONFIG = deploy/models.yaml
HF_CLI = hf
VLLM_MODELS_DIR = $(CACHE_DIR)/ollama/models/vllm
OLLAMA_HOST = http://devai-router:11434
OLLAMA_DEFAULT_MODEL = $(shell python3 -c "import yaml; print(yaml.safe_load(open('$(INFERENCE_CONFIG)'))['defaults']['ollama'])" 2>/dev/null || echo "qwen3.5:9b")

# Proxy build args (passed to all container builds)
PROXY_BUILD_ARGS = \
	--build-arg HTTP_PROXY=$(HTTP_PROXY) \
	--build-arg HTTPS_PROXY=$(HTTPS_PROXY) \
	--build-arg NO_PROXY=$(NO_PROXY)

# APT proxy (base image only — apt is not used in the lab layer)
APT_PROXY_ARG = --build-arg APT_PROXY=$(APT_PROXY)
PYTHON_VERSION_ARG = --build-arg PYTHON_VERSION=$(PYTHON_VERSION)

# Hash of cached binaries (invalidates build cache when binaries change)
BIN_HASH = $(shell cat $(CACHE_DIR)/pip/bin/.etags/* 2>/dev/null | md5sum | cut -c1-12)

# Cache mount args (bind host cache dirs into build for pip/uv, npm, and binaries)
CACHE_BUILD_ARGS = \
	--build-arg BIN_HASH=$(BIN_HASH) \
	-v $(CACHE_DIR)/pip:/root/.cache/uv \
	-v $(CACHE_DIR)/npm:/root/.npm \
	-v $(CACHE_DIR)/pip/bin:/var/cache/bin:ro

# Proxy runtime env (passed to all container runs)
PROXY_RUN_ENV = \
	-e HTTP_PROXY=$(HTTP_PROXY) \
	-e HTTPS_PROXY=$(HTTPS_PROXY) \
	-e NO_PROXY=$(NO_PROXY) \
	-e http_proxy=$(HTTP_PROXY) \
	-e https_proxy=$(HTTPS_PROXY) \
	-e no_proxy=$(NO_PROXY)


# GPU build settings
GPU_BASE_IMAGE ?= docker.io/nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

# Mount host config files (gitconfig, ssh) to staging dir for entrypoint to copy
HOME_MOUNT_ARG =
ifneq ($(HOST_HOME_DIR),)
	HOME_MOUNT_ARG += $(if $(wildcard $(HOME)/.gitconfig),-v "$(HOME)/.gitconfig":/tmp/host-config/.gitconfig:ro)
	HOME_MOUNT_ARG += $(if $(wildcard $(HOME)/.ssh),-v "$(HOME)/.ssh":/tmp/host-config/.ssh:ro)
endif

RUN_FLAGS =

# User switching: only needed for docker (rootless podman root = host user)
USER_ENV =
ifneq ($(findstring podman,$(CONTAINER_RUNTIME)),podman)
	USER_ENV += -e USER_ID=$(shell id -u) -e GROUP_ID=$(shell id -g)
endif

# GPU runtime flags
GPU_FLAGS =
ifeq ($(findstring podman,$(CONTAINER_RUNTIME)),podman)
	GPU_FLAGS += --device nvidia.com/gpu=all --security-opt=label=disable
else
	GPU_FLAGS += --gpus all
endif

# Compose settings

.PHONY: all build build-cpu build-gpu build-base-cpu build-base-gpu build-router
.PHONY: lab-cpu lab-gpu shell-cpu shell-gpu
.PHONY: cache-up cache-down cache-status cache-clean
.PHONY: ollama-pull ollama-rm ollama-list ollama-status ollama-clean ollama-df
.PHONY: vllm-pull vllm-list vllm-rm vllm-status vllm-df
.PHONY: clean clean-cpu clean-gpu clean-router prune
.PHONY: fetch-cli pull-images install-systemd test test-router test-ollama test-vllm test-idle help

all: help

# =============================================================================
# Fetch external dependencies to local cache (run once)
# =============================================================================


ETAG_DIR = $(CACHE_DIR)/pip/bin/.etags

fetch-cli: ## Download all external binaries and packages to local cache (uses ETags to skip unchanged)
	@mkdir -p $(CACHE_DIR)/pip/bin/gemini $(ETAG_DIR)
	@CC_BUCKET="https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/claude-code-releases" \
		&& ARCH=$$(dpkg --print-architecture) \
		&& case "$$ARCH" in amd64) CC_PLATFORM=linux-x64;; arm64) CC_PLATFORM=linux-arm64;; esac \
		&& CC_VERSION=$$(curl -fsSL "$$CC_BUCKET/latest") \
		&& HTTP_CODE=$$(curl -fsSL -w '%{http_code}' -o $(CACHE_DIR)/pip/bin/claude.tmp \
			--etag-compare $(ETAG_DIR)/claude.etag --etag-save $(ETAG_DIR)/claude.etag \
			"$$CC_BUCKET/$$CC_VERSION/$$CC_PLATFORM/claude") \
		&& if [ "$$HTTP_CODE" = "304" ] || [ ! -s $(CACHE_DIR)/pip/bin/claude.tmp ]; then \
			rm -f $(CACHE_DIR)/pip/bin/claude.tmp; echo "Claude Code: up to date"; \
		else \
			mv $(CACHE_DIR)/pip/bin/claude.tmp $(CACHE_DIR)/pip/bin/claude \
			&& chmod +x $(CACHE_DIR)/pip/bin/claude && echo "Claude Code: updated"; fi
	@ARCH=$$(dpkg --print-architecture) \
		&& case "$$ARCH" in amd64) CODEX_ARCH=x86_64;; arm64) CODEX_ARCH=aarch64;; esac \
		&& HTTP_CODE=$$(curl -fsSL -w '%{http_code}' -o $(CACHE_DIR)/pip/bin/codex.tar.gz \
			--etag-compare $(ETAG_DIR)/codex.etag --etag-save $(ETAG_DIR)/codex.etag \
			"https://github.com/openai/codex/releases/latest/download/codex-$${CODEX_ARCH}-unknown-linux-musl.tar.gz") \
		&& if [ "$$HTTP_CODE" = "304" ] || [ ! -s $(CACHE_DIR)/pip/bin/codex.tar.gz ]; then \
			rm -f $(CACHE_DIR)/pip/bin/codex.tar.gz; echo "OpenAI Codex: up to date"; \
		else \
			tar -xzf $(CACHE_DIR)/pip/bin/codex.tar.gz -C $(CACHE_DIR)/pip/bin \
			&& mv $(CACHE_DIR)/pip/bin/codex-$${CODEX_ARCH}-unknown-linux-musl $(CACHE_DIR)/pip/bin/codex \
			&& rm -f $(CACHE_DIR)/pip/bin/codex.tar.gz && echo "OpenAI Codex: updated"; fi
	@ARCH=$$(dpkg --print-architecture) \
		&& case "$$ARCH" in amd64) OL_ARCH=amd64;; arm64) OL_ARCH=arm64;; esac \
		&& HTTP_CODE=$$(curl -fsSL -w '%{http_code}' -o $(CACHE_DIR)/pip/bin/ollama.tar.zst \
			--etag-compare $(ETAG_DIR)/ollama.etag --etag-save $(ETAG_DIR)/ollama.etag \
			"https://github.com/ollama/ollama/releases/latest/download/ollama-linux-$${OL_ARCH}.tar.zst") \
		&& if [ "$$HTTP_CODE" = "304" ] || [ ! -s $(CACHE_DIR)/pip/bin/ollama.tar.zst ]; then \
			rm -f $(CACHE_DIR)/pip/bin/ollama.tar.zst; echo "Ollama CLI: up to date"; \
		else \
			tar --zstd -xf $(CACHE_DIR)/pip/bin/ollama.tar.zst -C $(CACHE_DIR)/pip/bin --strip-components=1 bin/ollama \
			&& rm -f $(CACHE_DIR)/pip/bin/ollama.tar.zst && echo "Ollama CLI: updated"; fi
	@ARCH=$$(dpkg --print-architecture) \
		&& case "$$ARCH" in amd64) CS_ARCH=amd64;; arm64) CS_ARCH=arm64;; esac \
		&& CS_VERSION=$$(curl -fsSL https://api.github.com/repos/coder/code-server/releases/latest | python3 -c "import sys,json; print(json.load(sys.stdin)['tag_name'].lstrip('v'))") \
		&& HTTP_CODE=$$(curl -fsSL -w '%{http_code}' -o $(CACHE_DIR)/pip/bin/code-server.tar.gz \
			--etag-compare $(ETAG_DIR)/code-server.etag --etag-save $(ETAG_DIR)/code-server.etag \
			"https://github.com/coder/code-server/releases/latest/download/code-server-$${CS_VERSION}-linux-$${CS_ARCH}.tar.gz") \
		&& if [ "$$HTTP_CODE" = "304" ] || [ ! -s $(CACHE_DIR)/pip/bin/code-server.tar.gz ]; then \
			rm -f $(CACHE_DIR)/pip/bin/code-server.tar.gz; echo "code-server: up to date"; \
		else \
			mkdir -p $(CACHE_DIR)/pip/bin/code-server \
			&& tar -xzf $(CACHE_DIR)/pip/bin/code-server.tar.gz -C $(CACHE_DIR)/pip/bin/code-server --strip-components=1 \
			&& rm -f $(CACHE_DIR)/pip/bin/code-server.tar.gz && echo "code-server: updated"; fi
	@HTTP_CODE=$$(curl -fsSL -w '%{http_code}' -o $(CACHE_DIR)/pip/bin/uv.tar.gz \
			--etag-compare $(ETAG_DIR)/uv.etag --etag-save $(ETAG_DIR)/uv.etag \
			"https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz") \
		&& if [ "$$HTTP_CODE" = "304" ] || [ ! -s $(CACHE_DIR)/pip/bin/uv.tar.gz ]; then \
			rm -f $(CACHE_DIR)/pip/bin/uv.tar.gz; echo "uv: up to date"; \
		else \
			tar -xzf $(CACHE_DIR)/pip/bin/uv.tar.gz -C $(CACHE_DIR)/pip/bin --strip-components=1 uv-x86_64-unknown-linux-gnu/uv uv-x86_64-unknown-linux-gnu/uvx \
			&& rm -f $(CACHE_DIR)/pip/bin/uv.tar.gz && echo "uv: updated"; fi
	@ARCH=$$(dpkg --print-architecture) \
		&& case "$$ARCH" in amd64) OC_ARCH=x86_64;; arm64) OC_ARCH=arm64;; esac \
		&& HTTP_CODE=$$(curl -fsSL -w '%{http_code}' -o $(CACHE_DIR)/pip/bin/opencode.tar.gz \
			--etag-compare $(ETAG_DIR)/opencode.etag --etag-save $(ETAG_DIR)/opencode.etag \
			"https://github.com/opencode-ai/opencode/releases/latest/download/opencode-linux-$${OC_ARCH}.tar.gz") \
		&& if [ "$$HTTP_CODE" = "304" ] || [ ! -s $(CACHE_DIR)/pip/bin/opencode.tar.gz ]; then \
			rm -f $(CACHE_DIR)/pip/bin/opencode.tar.gz; echo "OpenCode: up to date"; \
		else \
			tar -xzf $(CACHE_DIR)/pip/bin/opencode.tar.gz -C $(CACHE_DIR)/pip/bin opencode \
			&& rm -f $(CACHE_DIR)/pip/bin/opencode.tar.gz && echo "OpenCode: updated"; fi
	@META=$$(curl -fsSL "https://registry.npmjs.org/@google/gemini-cli/latest") \
		&& LATEST=$$(echo "$$META" | python3 -c "import sys,json; print(json.load(sys.stdin)['version'])") \
		&& CACHED=$$(cat $(ETAG_DIR)/gemini.version 2>/dev/null || echo "none") \
		&& if [ "$$LATEST" = "$$CACHED" ]; then echo "Gemini CLI: up to date ($$CACHED)"; else \
			echo "Fetching Gemini CLI $$LATEST..." \
			&& TARBALL=$$(echo "$$META" | python3 -c "import sys,json; print(json.load(sys.stdin)['dist']['tarball'])") \
			&& rm -rf $(CACHE_DIR)/pip/bin/gemini/lib/node_modules/@google/gemini-cli \
			&& mkdir -p $(CACHE_DIR)/pip/bin/gemini/lib/node_modules/@google/gemini-cli \
			&& curl -fsSL "$$TARBALL" | tar -xz -C $(CACHE_DIR)/pip/bin/gemini/lib/node_modules/@google/gemini-cli --strip-components=1 \
			&& mkdir -p $(CACHE_DIR)/pip/bin/gemini/bin \
			&& ln -sf ../lib/node_modules/@google/gemini-cli/bundle/gemini.js $(CACHE_DIR)/pip/bin/gemini/bin/gemini \
			&& echo "$$LATEST" > $(ETAG_DIR)/gemini.version \
			&& echo "Gemini CLI: updated to $$LATEST"; fi

# Base images used by build and infrastructure
BASE_IMAGES = debian:trixie $(GPU_BASE_IMAGE)
CACHE_IMAGES = $(shell $(COMPOSE) -f $(CACHE_COMPOSE) config --images 2>/dev/null | grep -v devai-)

pull-images: ## Pull latest versions of all base and infrastructure images
	@for img in $(BASE_IMAGES) $(CACHE_IMAGES); do \
		echo "Pulling $$img..." \
		&& $(CONTAINER_RUNTIME) pull "$$img" || true; \
	done

build-base-cpu: ## Build base image with system packages and runtimes (CPU)
	$(CONTAINER_RUNTIME) build --network=host \
		$(PROXY_BUILD_ARGS) \
		$(APT_PROXY_ARG) \
		$(PYTHON_VERSION_ARG) \
		-f deploy/Dockerfile.base \
		-t $(BASE_IMAGE_NAME) .

build-base-gpu: ## Build base image with system packages and runtimes (GPU)
	$(CONTAINER_RUNTIME) build --network=host \
		$(PROXY_BUILD_ARGS) \
		$(APT_PROXY_ARG) \
		$(PYTHON_VERSION_ARG) \
		--build-arg BASE_IMAGE=$(GPU_BASE_IMAGE) \
		-f deploy/Dockerfile.base \
		-t $(BASE_IMAGE_NAME_GPU) .

build: build-cpu build-gpu build-router ## Build all images (CPU + GPU + router)

build-cpu: build-base-cpu fetch-cli ## Build the container image (CPU)
	$(CONTAINER_RUNTIME) build --network=host \
		$(PROXY_BUILD_ARGS) \
		$(CACHE_BUILD_ARGS) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE_NAME) \
		--build-arg OLLAMA_HOST=$(OLLAMA_HOST) \
		--build-arg OLLAMA_DEFAULT_MODEL=$(OLLAMA_DEFAULT_MODEL) \
		-f deploy/Dockerfile.lab \
		-t $(IMAGE_NAME) .

build-gpu: build-base-gpu fetch-cli ## Build the container image (GPU/CUDA)
	$(CONTAINER_RUNTIME) build --network=host \
		$(PROXY_BUILD_ARGS) \
		$(CACHE_BUILD_ARGS) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE_NAME_GPU) \
		--build-arg GPU_BUILD=true \
		--build-arg OLLAMA_HOST=$(OLLAMA_HOST) \
		--build-arg OLLAMA_DEFAULT_MODEL=$(OLLAMA_DEFAULT_MODEL) \
		-f deploy/Dockerfile.lab \
		-t $(IMAGE_NAME_GPU) .


lab-cpu: ## Run the container (CPU)
	@if [ -n "$(HOST_HOME_DIR)" ]; then mkdir -p "$(HOST_HOME_DIR)"; fi
	@echo "Starting $(IMAGE_NAME)..."
	@echo "Access JupyterLab at http://$(HOST_IP):$(LAB_PORT)/lab?token=..."
	$(CONTAINER_RUNTIME) run -it --rm \
		--name $(IMAGE_NAME) \
		$(RUN_FLAGS) \
		--network $(DEVAI_NETWORK) \
		--add-host=host.containers.internal:host-gateway \
		$(PROXY_RUN_ENV) \
		-e OLLAMA_HOST=$(OLLAMA_HOST) \
		-e OLLAMA_DEFAULT_MODEL=$(OLLAMA_DEFAULT_MODEL) \
		$(USER_ENV) \
		-e CONTAINER_USER=$(CONTAINER_USER) \
		-e HOST_IP=$(HOST_IP) \
		-e PORT=$(LAB_PORT) \
		-p 0.0.0.0:$(LAB_PORT):8888 \
		-v $(HOME_VOLUME):/home/$(CONTAINER_USER) \
		$(HOME_MOUNT_ARG) \
		-v "$$(readlink -f $(HOST_WORK_DIR))":/home/$(CONTAINER_USER)/work \
		$(IMAGE_NAME)

lab-gpu: ## Run the container (GPU/CUDA)
	@if [ -n "$(HOST_HOME_DIR)" ]; then mkdir -p "$(HOST_HOME_DIR)"; fi
	@echo "Starting $(IMAGE_NAME_GPU) with GPU support..."
	@echo "Access JupyterLab at http://$(HOST_IP):$(LAB_PORT)/lab?token=..."
	$(CONTAINER_RUNTIME) run -it --rm \
		--name $(IMAGE_NAME_GPU) \
		$(RUN_FLAGS) \
		$(GPU_FLAGS) \
		--network $(DEVAI_NETWORK) \
		--add-host=host.containers.internal:host-gateway \
		$(PROXY_RUN_ENV) \
		-e OLLAMA_HOST=$(OLLAMA_HOST) \
		-e OLLAMA_DEFAULT_MODEL=$(OLLAMA_DEFAULT_MODEL) \
		$(USER_ENV) \
		-e CONTAINER_USER=$(CONTAINER_USER) \
		-e HOST_IP=$(HOST_IP) \
		-e PORT=$(LAB_PORT) \
		-p 0.0.0.0:$(LAB_PORT):8888 \
		-v $(HOME_VOLUME):/home/$(CONTAINER_USER) \
		$(HOME_MOUNT_ARG) \
		-v "$$(readlink -f $(HOST_WORK_DIR))":/home/$(CONTAINER_USER)/work \
		$(IMAGE_NAME_GPU)

shell-cpu: ## Start an interactive shell (CPU)
	$(CONTAINER_RUNTIME) run -it --rm \
		--name $(IMAGE_NAME)-shell \
		$(RUN_FLAGS) \
		--network $(DEVAI_NETWORK) \
		--add-host=host.containers.internal:host-gateway \
		$(PROXY_RUN_ENV) \
		-e OLLAMA_HOST=$(OLLAMA_HOST) \
		-e OLLAMA_DEFAULT_MODEL=$(OLLAMA_DEFAULT_MODEL) \
		$(USER_ENV) \
		-e CONTAINER_USER=$(CONTAINER_USER) \
		-v $(HOME_VOLUME):/home/$(CONTAINER_USER) \
		$(HOME_MOUNT_ARG) \
		-v "$$(readlink -f $(HOST_WORK_DIR))":/home/$(CONTAINER_USER)/work \
		$(IMAGE_NAME) agent-picker

shell-gpu: ## Start an interactive shell (GPU)
	$(CONTAINER_RUNTIME) run -it --rm \
		--name $(IMAGE_NAME_GPU)-shell \
		$(RUN_FLAGS) \
		$(GPU_FLAGS) \
		--network $(DEVAI_NETWORK) \
		--add-host=host.containers.internal:host-gateway \
		$(PROXY_RUN_ENV) \
		-e OLLAMA_HOST=$(OLLAMA_HOST) \
		-e OLLAMA_DEFAULT_MODEL=$(OLLAMA_DEFAULT_MODEL) \
		$(USER_ENV) \
		-e CONTAINER_USER=$(CONTAINER_USER) \
		-v $(HOME_VOLUME):/home/$(CONTAINER_USER) \
		$(HOME_MOUNT_ARG) \
		-v "$$(readlink -f $(HOST_WORK_DIR))":/home/$(CONTAINER_USER)/work \
		$(IMAGE_NAME_GPU) agent-picker

clean: clean-cpu clean-gpu clean-router ## Remove all images (CPU + GPU + router)

clean-cpu: ## Remove the CPU container + base images
	$(CONTAINER_RUNTIME) rmi $(IMAGE_NAME) 2>/dev/null || true
	$(CONTAINER_RUNTIME) rmi $(BASE_IMAGE_NAME) 2>/dev/null || true

clean-gpu: ## Remove the GPU container + base images
	$(CONTAINER_RUNTIME) rmi $(IMAGE_NAME_GPU) 2>/dev/null || true
	$(CONTAINER_RUNTIME) rmi $(BASE_IMAGE_NAME_GPU) 2>/dev/null || true

clean-router: ## Remove the router image
	$(CONTAINER_RUNTIME) rmi devai-router 2>/dev/null || true

prune: ## Clean up dangling images only (keeps tagged images and volumes)
	$(CONTAINER_RUNTIME) image prune -f

test-router: ## Run Go unit tests for gpu-arbiter router
	$(CONTAINER_RUNTIME) run --rm \
		--entrypoint bash \
		-v "$$(pwd)/gpu-arbiter:/src:z" \
		-w /src \
		docker.io/library/golang:1.23-bookworm \
		-c "go test -race -v -count=1 ./..."

test-ollama: cache-up ## Run Ollama-only integration tests
	./tests/test-router.sh

test-vllm: cache-up ## Run vLLM/GPU integration tests (~10min)
	./tests/test-router-vllm.sh

test-idle: cache-up ## Run vLLM idle timeout test (restarts router temporarily)
	./tests/test-router-idle.sh

test: test-router test-ollama test-vllm test-idle ## Run all tests in sequence

help: ## Show this help message
	@printf "\nDevAI Lab — Containerized AI Development Environment\n\n"
	@printf "  %-44s%s\n" "BUILD" "RUN"
	@printf "  %-44s%s\n" "fetch-cli        Update CLI binaries" "lab-cpu          JupyterLab (CPU)"
	@printf "  %-44s%s\n" "pull-images      Pull base images" "lab-gpu          JupyterLab (GPU)"
	@printf "  %-44s%s\n" "build-cpu        Build image (CPU)" "shell-cpu        Shell (CPU)"
	@printf "  %-44s%s\n" "build-gpu        Build image (GPU)" "shell-gpu        Shell (GPU)"
	@printf "  %s\n" "build-router     Build router image"
	@printf "  %s\n" "build            Build all (CPU+GPU+router)"
	@printf "\n"
	@printf "  %-44s%s\n" "INFRASTRUCTURE" "MAINTENANCE"
	@printf "  %-44s%s\n" "cache-up         Start services" "clean-cpu        Remove image (CPU)"
	@printf "  %-44s%s\n" "cache-down       Stop services" "clean-gpu        Remove image (GPU)"
	@printf "  %-44s%s\n" "cache-status     Show status" "clean-router     Remove router image"
	@printf "  %-44s%s\n" "cache-clean      Remove cached data" "clean            Remove all images"
	@printf "  %-44s%s\n" "" "prune            Prune dangling images"
	@printf "  %-44s%s\n" "" "test             Run integration tests"
	@printf "\n"
	@printf "  %-44s%s\n" "OLLAMA (GGUF)" "vLLM (NVFP4)"
	@printf "  %-44s%s\n" "ollama-list      List models" "vllm-list        List models"
	@printf "  %-44s%s\n" "ollama-pull      Pull model(s)" "vllm-pull        Pull model(s)"
	@printf "  %-44s%s\n" "ollama-rm        Remove model" "vllm-rm          Remove model"
	@printf "  %-44s%s\n" "ollama-status    Show status" "vllm-status      Show status"
	@printf "  %-44s%s\n" "ollama-df        Disk usage" "vllm-df          Disk usage"
	@printf "  %s\n" "ollama-clean     Clean partials"
	@printf "\n"
	@printf "  %s\n" "DEPLOY"
	@printf "  %s\n" "install-systemd  Auto-start infrastructure at boot"
	@printf "\n"

# =============================================================================
# Infrastructure services (caches + Ollama + vLLM + Open WebUI)
# =============================================================================

cache-up: ## Start infrastructure services (caches + Ollama + vLLM + Open WebUI)
	@if [ "$(CONTAINER_RUNTIME)" = "podman" ] && ! systemctl --user is-active --quiet podman.socket; then \
		echo "Starting Podman API socket..."; \
		systemctl --user enable --now podman.socket; \
	fi
	@$(CONTAINER_RUNTIME) network exists $(DEVAI_NETWORK) 2>/dev/null || $(CONTAINER_RUNTIME) network create $(DEVAI_NETWORK)
	$(COMPOSE) -f $(CACHE_COMPOSE) up -d
	@$(COMPOSE) -f $(CACHE_COMPOSE) stop vllm 2>/dev/null || true
	@echo "Infrastructure services started:"
	@echo "  apt-cacher-ng:     http://localhost:3142"
	@echo "  Registry mirror:   http://localhost:5000"
	@echo "  Router:            devai-router:11434 (unified endpoint)"
	@echo "  Ollama:            devai-ollama:11434 (GGUF models)"
	@echo "  vLLM:              devai-vllm (auto-managed by router)"
	@echo "  Open WebUI:        https://localhost:$(WEBUI_PORT)"
	@echo ""
	@echo "To pull a model:  make ollama-pull MODEL=llama3.2"

cache-down: ## Stop infrastructure services
	$(COMPOSE) -f $(CACHE_COMPOSE) down

cache-status: ## Show infrastructure service status and disk usage
	@$(COMPOSE) -f $(CACHE_COMPOSE) ps
	@echo ""
	@echo "Cache disk usage:"
	@du -sh $(CACHE_DIR)/*/ 2>/dev/null; true; true
	@echo ""
	@echo "Ollama models:"
	@$(OLLAMA_EXEC) list 2>/dev/null || echo "  (ollama not running)"
	@echo ""
	@echo "vLLM models:"
	@found=false; for dir in $(VLLM_MODELS_DIR)/*/; do \
		[ -f "$$dir/config.json" ] || continue; \
		if ! $$found; then \
			printf "%-46s%-16s%-10s%-20s\n" "NAME" "ID" "SIZE" "MODIFIED"; \
			found=true; \
		fi; \
		name=$$(basename "$$dir"); \
		size=$$(du -sh "$$dir" | cut -f1); \
		id=$$(sha256sum "$$dir/config.json" | cut -c1-12); \
		mod_epoch=$$(stat -c '%Y' "$$dir/config.json"); \
		now_epoch=$$(date +%s); \
		diff_sec=$$((now_epoch - mod_epoch)); \
		if [ $$diff_sec -lt 60 ]; then modified="$$diff_sec seconds ago"; \
		elif [ $$diff_sec -lt 3600 ]; then modified="$$((diff_sec / 60)) minutes ago"; \
		elif [ $$diff_sec -lt 86400 ]; then modified="$$((diff_sec / 3600)) hours ago"; \
		elif [ $$diff_sec -lt 604800 ]; then modified="$$((diff_sec / 86400)) days ago"; \
		else modified="$$((diff_sec / 604800)) weeks ago"; fi; \
		printf "%-46s%-16s%-10s%-20s\n" "$$name" "$$id" "$$size" "$$modified"; \
	done; \
	$$found || echo "  (none — run 'make vllm-pull' to download)"

cache-clean: ## Remove all cached data (keeps volumes mounted)
	$(COMPOSE) -f $(CACHE_COMPOSE) down
	@echo "Cleaning cache directories..."
	rm -rf $(CACHE_DIR)/registry/* $(CACHE_DIR)/apt/* $(CACHE_DIR)/pip/* $(CACHE_DIR)/npm/*
	@echo "Cache cleaned."

ollama-pull: ## Pull model(s) (usage: make ollama-pull MODEL=name, or make ollama-pull to pull all)
	@INFERENCE_CONFIG=$(INFERENCE_CONFIG) CONTAINER_RUNTIME=$(CONTAINER_RUNTIME) OLLAMA_CONTAINER=$(OLLAMA_CONTAINER) MODEL="$(MODEL)" \
		python3 scripts/ollama-pull.py

ollama-rm: ## Remove an Ollama model and clean up (usage: make ollama-rm MODEL=llama3.2)
	$(OLLAMA_EXEC) rm $(MODEL)
	@$(MAKE) --no-print-directory ollama-clean

ollama-list: ## List downloaded models with catalog info (sizes, purpose)
	@INFERENCE_CONFIG=$(INFERENCE_CONFIG) CONTAINER_RUNTIME=$(CONTAINER_RUNTIME) OLLAMA_CONTAINER=$(OLLAMA_CONTAINER) \
		python3 scripts/ollama-list.py


ollama-status: ## Show Ollama container status
	@$(CONTAINER_RUNTIME) inspect -f '{{.State.Status}}' $(OLLAMA_CONTAINER) 2>/dev/null || echo "not running"
	@$(OLLAMA_EXEC) ps 2>/dev/null || true

ollama-clean: ## Remove partial downloads and orphaned blobs from Ollama cache
	@echo "Removing partial downloads..."
	@rm -vf $(CACHE_DIR)/ollama/models/blobs/*-partial* 2>/dev/null || true
	@echo "Removing orphaned blobs..."
	@for b in $(CACHE_DIR)/ollama/models/blobs/sha256-*; do \
		[ -f "$$b" ] || continue; \
		case "$$b" in *-partial*) continue;; esac; \
		bname=$$(basename "$$b" | sed 's/-/:/'); \
		if ! grep -rq "$$bname" $(CACHE_DIR)/ollama/models/manifests/ 2>/dev/null; then \
			rm -v "$$b"; \
		fi; \
	done
	@echo "Removing orphaned gguf/vllm caches..."
	@for dir in $(CACHE_DIR)/ollama/models/gguf/* $(CACHE_DIR)/ollama/models/vllm/*; do \
		[ -d "$$dir" ] || continue; \
		model=$$(basename "$$dir"); \
		if ! find $(CACHE_DIR)/ollama/models/manifests/ -path "*$$model*" -print -quit 2>/dev/null | grep -q .; then \
			rm -rvf "$$dir"; \
		fi; \
	done
	@echo "Done. Restart Ollama to reclaim space: $(CONTAINER_RUNTIME) restart $(OLLAMA_CONTAINER)"

ollama-df: ## Show Ollama cache disk usage breakdown
	@echo "=== Ollama disk usage ==="
	@df -h $(CACHE_DIR)/ollama
	@echo ""
	@echo "=== Breakdown ==="
	@du -sh $(CACHE_DIR)/ollama/models/blobs/ 2>/dev/null || true
	@du -sh $(CACHE_DIR)/ollama/models/gguf/ 2>/dev/null || true
	@du -sh $(CACHE_DIR)/ollama/models/vllm/ 2>/dev/null || true
	@echo ""
	@partials=$$(ls $(CACHE_DIR)/ollama/models/blobs/*-partial 2>/dev/null | wc -l); \
	if [ "$$partials" -gt 0 ]; then \
		echo "WARNING: $$partials partial download(s) found. Run 'make ollama-clean' to remove."; \
		for p in $(CACHE_DIR)/ollama/models/blobs/*-partial; do \
			[ -f "$$p" ] || continue; \
			size=$$(du -sh "$$p" | cut -f1); \
			hash=$$(basename "$$p" | sed 's/-partial.*//; s/-/:/'); \
			model=$$(grep -rl "$$hash" $(CACHE_DIR)/ollama/models/manifests/ 2>/dev/null \
				| head -1 | sed 's|.*/library/||; s|/|:|'); \
			age=$$(( $$(date +%s) - $$(stat -c %Y "$$p") )); \
			if [ "$$age" -lt 60 ]; then status="downloading"; else status="stale"; fi; \
			echo "  $$size	$${model:-unknown}	($$status)"; \
		done; \
	else \
		echo "No partial downloads."; \
	fi

# =============================================================================
# vLLM targets (NVFP4 on Blackwell)
# =============================================================================

vllm-pull: ## Download vLLM model (usage: make vllm-pull MODEL=name or make vllm-pull to download all)
	@INFERENCE_CONFIG=$(INFERENCE_CONFIG) VLLM_MODELS_DIR=$(VLLM_MODELS_DIR) HF_CLI=$(HF_CLI) MODEL="$(MODEL)" \
		python3 scripts/vllm-pull.py

vllm-list: ## List vLLM models with status
	@INFERENCE_CONFIG=$(INFERENCE_CONFIG) VLLM_MODELS_DIR=$(VLLM_MODELS_DIR) \
		python3 scripts/vllm-list.py

vllm-status: ## Show vLLM container status
	@$(CONTAINER_RUNTIME) inspect -f '{{.State.Status}}' devai-vllm 2>/dev/null || echo "not running"
	@$(CONTAINER_RUNTIME) exec devai-vllm curl -sf http://localhost:11434/v1/models 2>/dev/null \
		| python3 -c "import sys,json; [print(f'  {m[\"id\"]}') for m in json.load(sys.stdin)['data']]" 2>/dev/null \
		|| true

vllm-rm: ## Remove a vLLM model (usage: make vllm-rm MODEL=name)
	@if [ -z "$(MODEL)" ]; then echo "Usage: make vllm-rm MODEL=<name>"; exit 1; fi
	@echo "Removing $(MODEL)..."
	rm -rf $(VLLM_MODELS_DIR)/$(MODEL)
	@echo "Done."

vllm-df: ## Show vLLM models disk usage
	@echo "=== vLLM disk usage ==="
	@df -h $(VLLM_MODELS_DIR) 2>/dev/null | tail -1 | awk '{print "  " $$3 " used / " $$2 " total (" $$5 ")"}'
	@echo ""
	@for dir in $(VLLM_MODELS_DIR)/*/; do \
		[ -f "$$dir/config.json" ] || continue; \
		printf "  %-45s %s\n" "$$(basename $$dir)" "$$(du -sh $$dir | cut -f1)"; \
	done


build-router: ## Build the gpu-arbiter router image
	$(CONTAINER_RUNTIME) build --network=host \
		-f deploy/Dockerfile.router \
		-t devai-router .

install-systemd: ## Install and enable systemd service for infrastructure
	@mkdir -p $(HOME)/.config/devai $(HOME)/.config/systemd/user
	cp deploy/docker-compose.yaml $(HOME)/.config/devai/docker-compose.yaml
	cp deploy/registry-config.yaml $(HOME)/.config/devai/registry-config.yaml
	cp deploy/systemd/devai-infra.service $(HOME)/.config/systemd/user/
	systemctl --user daemon-reload
	systemctl --user enable --now podman.socket
	systemctl --user enable --now devai-infra.service
	loginctl enable-linger $(USER)
	@echo "Installed to $(HOME)/.config/devai/ and enabled devai-infra.service"

