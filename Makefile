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
GPU_MEMORY_GB ?= 24
MAX_CONTEXT_LEN ?= 262144
# Export so recipe shells — and compose interpolation in particular —
# see these values without requiring the user to maintain a .env file.
# Shell-level env wins over compose's `${VAR:-default}` fallback, so the
# devai-router container receives the Makefile values verbatim. Anything
# the user already exported in their shell still wins (?= keeps the
# external value), and `make MAX_CONTEXT_LEN=X cache-up` overrides both.
export GPU_MEMORY_GB MAX_CONTEXT_LEN
CACHE_COMPOSE = deploy/docker-compose.yaml
INFERENCE_CONFIG = deploy/models.yaml
HF_CLI = hf
VLLM_MODELS_DIR = $(CACHE_DIR)/ollama/models/vllm
# Absolute host path to scripts/vllm_plugins. The router (running in
# its own container) bind-mounts this into the recreated vLLM
# container so models that resolve to a custom tool/reasoning parser
# in deploy/vllm-plugins.json get the plugin .py file at the path
# `--tool-parser-plugin` expects. Exported so compose interpolates it
# into the router's env.
VLLM_PLUGINS_HOST_DIR = $(abspath scripts/vllm_plugins)
export VLLM_PLUGINS_HOST_DIR
OLLAMA_HOST = http://devai-router:11434
# Pinned to a real catalog tag. The previous shell-out read defaults['ollama']
# from models.yaml, which generate-catalog.py never writes — so this always
# fell through to a tag (qwen3.5:9b) that doesn't exist in the catalog.
OLLAMA_DEFAULT_MODEL = qwen3.5:9b-q8_0

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

# Read-only mount of the model cache so the in-container picker can do
# disk-based "is downloaded?" detection (ollama manifests + vllm/sglang dirs).
MODEL_CACHE_MOUNT = $(if $(wildcard $(CACHE_DIR)/ollama),-v $(CACHE_DIR)/ollama:/var/cache/devai/ollama:ro)

# Read-only mount of the probe caches so the in-container picker can
# render the per-tier menu for every backend and the router can build
# its name → context-cap map. All three caches must be exposed under
# /etc/devai/.<backend>-reasoning-cache.json — model-picker.py looks
# them up by that path. Without the vLLM/SGLang mounts the picker
# falls back to "no HF probes" and shows Ollama rows only.
PROBE_CACHE_MOUNT = \
	$(if $(wildcard deploy/.ollama-reasoning-cache.json),-v $(CURDIR)/deploy/.ollama-reasoning-cache.json:/etc/devai/.ollama-reasoning-cache.json:ro) \
	$(if $(wildcard deploy/.vllm-reasoning-cache.json),-v $(CURDIR)/deploy/.vllm-reasoning-cache.json:/etc/devai/.vllm-reasoning-cache.json:ro) \
	$(if $(wildcard deploy/.sglang-reasoning-cache.json),-v $(CURDIR)/deploy/.sglang-reasoning-cache.json:/etc/devai/.sglang-reasoning-cache.json:ro)

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
.PHONY: cache-up cache-down cache-status cache-clean logs setup-logs
.PHONY: ollama-rm ollama-list ollama-status ollama-clean ollama-df
.PHONY: vllm-list vllm-rm vllm-status vllm-df
.PHONY: clean clean-cpu clean-gpu clean-router prune
.PHONY: fetch-cli pull-images install install-systemd uninstall test test-router test-ollama test-agents test-models test-probe-vllm test-probe-sglang test-probe-ollama-idempotent test-vllm test-sglang test-e2e test-full help
.PHONY: catalog-regen catalog-suggest probe probe-vllm probe-sglang model-fit model-pull vram-fit verify-backend-flags ollama-cleanup-ctx-variants
.PHONY: bench bench-vllm bench-sglang bench-ollama bench-report test-bench-smoke

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
			rm -f $(CACHE_DIR)/pip/bin/claude.tmp; STATE="up to date"; \
		else \
			mv $(CACHE_DIR)/pip/bin/claude.tmp $(CACHE_DIR)/pip/bin/claude \
			&& chmod +x $(CACHE_DIR)/pip/bin/claude && STATE="updated"; fi \
		&& VERSION=$$($(CACHE_DIR)/pip/bin/claude --version 2>&1 | awk '{print $$1; exit}') \
		&& echo "Claude Code: $$STATE ($$VERSION)"
	@ARCH=$$(dpkg --print-architecture) \
		&& case "$$ARCH" in amd64) CODEX_ARCH=x86_64;; arm64) CODEX_ARCH=aarch64;; esac \
		&& HTTP_CODE=$$(curl -fsSL -w '%{http_code}' -o $(CACHE_DIR)/pip/bin/codex.tar.gz \
			--etag-compare $(ETAG_DIR)/codex.etag --etag-save $(ETAG_DIR)/codex.etag \
			"https://github.com/openai/codex/releases/latest/download/codex-$${CODEX_ARCH}-unknown-linux-musl.tar.gz") \
		&& if [ "$$HTTP_CODE" = "304" ] || [ ! -s $(CACHE_DIR)/pip/bin/codex.tar.gz ]; then \
			rm -f $(CACHE_DIR)/pip/bin/codex.tar.gz; STATE="up to date"; \
		else \
			tar -xzf $(CACHE_DIR)/pip/bin/codex.tar.gz -C $(CACHE_DIR)/pip/bin \
			&& mv $(CACHE_DIR)/pip/bin/codex-$${CODEX_ARCH}-unknown-linux-musl $(CACHE_DIR)/pip/bin/codex \
			&& rm -f $(CACHE_DIR)/pip/bin/codex.tar.gz && STATE="updated"; fi \
		&& VERSION=$$($(CACHE_DIR)/pip/bin/codex --version 2>&1 | awk '{print $$2; exit}') \
		&& echo "OpenAI Codex: $$STATE ($$VERSION)"
	@ARCH=$$(dpkg --print-architecture) \
		&& case "$$ARCH" in amd64) OL_ARCH=amd64;; arm64) OL_ARCH=arm64;; esac \
		&& HTTP_CODE=$$(curl -fsSL -w '%{http_code}' -o $(CACHE_DIR)/pip/bin/ollama.tar.zst \
			--etag-compare $(ETAG_DIR)/ollama.etag --etag-save $(ETAG_DIR)/ollama.etag \
			"https://github.com/ollama/ollama/releases/latest/download/ollama-linux-$${OL_ARCH}.tar.zst") \
		&& if [ "$$HTTP_CODE" = "304" ] || [ ! -s $(CACHE_DIR)/pip/bin/ollama.tar.zst ]; then \
			rm -f $(CACHE_DIR)/pip/bin/ollama.tar.zst; STATE="up to date"; \
		else \
			tar --zstd -xf $(CACHE_DIR)/pip/bin/ollama.tar.zst -C $(CACHE_DIR)/pip/bin --strip-components=1 bin/ollama \
			&& rm -f $(CACHE_DIR)/pip/bin/ollama.tar.zst && STATE="updated"; fi \
		&& VERSION=$$(OLLAMA_HOST= $(CACHE_DIR)/pip/bin/ollama --version 2>&1 | awk '/client version/{print $$NF; exit}') \
		&& echo "Ollama CLI: $$STATE ($$VERSION)"
	@ARCH=$$(dpkg --print-architecture) \
		&& case "$$ARCH" in amd64) CS_ARCH=amd64;; arm64) CS_ARCH=arm64;; esac \
		&& CS_VERSION=$$(curl -fsSL https://api.github.com/repos/coder/code-server/releases/latest | python3 -c "import sys,json; print(json.load(sys.stdin)['tag_name'].lstrip('v'))") \
		&& HTTP_CODE=$$(curl -fsSL -w '%{http_code}' -o $(CACHE_DIR)/pip/bin/code-server.tar.gz \
			--etag-compare $(ETAG_DIR)/code-server.etag --etag-save $(ETAG_DIR)/code-server.etag \
			"https://github.com/coder/code-server/releases/latest/download/code-server-$${CS_VERSION}-linux-$${CS_ARCH}.tar.gz") \
		&& if [ "$$HTTP_CODE" = "304" ] || [ ! -s $(CACHE_DIR)/pip/bin/code-server.tar.gz ]; then \
			rm -f $(CACHE_DIR)/pip/bin/code-server.tar.gz; STATE="up to date"; \
		else \
			mkdir -p $(CACHE_DIR)/pip/bin/code-server \
			&& tar -xzf $(CACHE_DIR)/pip/bin/code-server.tar.gz -C $(CACHE_DIR)/pip/bin/code-server --strip-components=1 \
			&& rm -f $(CACHE_DIR)/pip/bin/code-server.tar.gz && STATE="updated"; fi \
		&& VERSION=$$($(CACHE_DIR)/pip/bin/code-server/bin/code-server --version 2>&1 | awk '/^[0-9]+\.[0-9]+\.[0-9]+ /{print $$1; exit}') \
		&& echo "code-server: $$STATE ($$VERSION)"
	@HTTP_CODE=$$(curl -fsSL -w '%{http_code}' -o $(CACHE_DIR)/pip/bin/uv.tar.gz \
			--etag-compare $(ETAG_DIR)/uv.etag --etag-save $(ETAG_DIR)/uv.etag \
			"https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz") \
		&& if [ "$$HTTP_CODE" = "304" ] || [ ! -s $(CACHE_DIR)/pip/bin/uv.tar.gz ]; then \
			rm -f $(CACHE_DIR)/pip/bin/uv.tar.gz; STATE="up to date"; \
		else \
			tar -xzf $(CACHE_DIR)/pip/bin/uv.tar.gz -C $(CACHE_DIR)/pip/bin --strip-components=1 uv-x86_64-unknown-linux-gnu/uv uv-x86_64-unknown-linux-gnu/uvx \
			&& rm -f $(CACHE_DIR)/pip/bin/uv.tar.gz && STATE="updated"; fi \
		&& VERSION=$$($(CACHE_DIR)/pip/bin/uv --version 2>&1 | awk '{print $$2; exit}') \
		&& echo "uv: $$STATE ($$VERSION)"
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
	@ARCH=$$(dpkg --print-architecture) \
		&& case "$$ARCH" in amd64) LATE_ARCH=amd64;; arm64) LATE_ARCH=arm64;; esac \
		&& HTTP_CODE=$$(curl -fsSL -w '%{http_code}' -o $(CACHE_DIR)/pip/bin/late.tmp \
			--etag-compare $(ETAG_DIR)/late.etag --etag-save $(ETAG_DIR)/late.etag \
			"https://github.com/mlhher/late/releases/latest/download/late-linux-$${LATE_ARCH}") \
		&& if [ "$$HTTP_CODE" = "304" ] || [ ! -s $(CACHE_DIR)/pip/bin/late.tmp ]; then \
			rm -f $(CACHE_DIR)/pip/bin/late.tmp; STATE="up to date"; \
		else \
			mv $(CACHE_DIR)/pip/bin/late.tmp $(CACHE_DIR)/pip/bin/late \
			&& chmod +x $(CACHE_DIR)/pip/bin/late && STATE="updated"; fi \
		&& VERSION=$$($(CACHE_DIR)/pip/bin/late --version 2>&1 | awk '{print $$2; exit}' | sed 's/^v//') \
		&& echo "LATE: $$STATE ($$VERSION)"

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
		-e CONTEXT=$${CONTEXT:-$(MAX_CONTEXT_LEN)} \
		-e VRAM=$${VRAM:-$(GPU_MEMORY_GB)} \
		$(USER_ENV) \
		-e CONTAINER_USER=$(CONTAINER_USER) \
		-e HOST_IP=$(HOST_IP) \
		-e PORT=$(LAB_PORT) \
		-p 0.0.0.0:$(LAB_PORT):8888 \
		-v $(HOME_VOLUME):/home/$(CONTAINER_USER) \
		$(HOME_MOUNT_ARG) \
		$(MODEL_CACHE_MOUNT) \
		$(PROBE_CACHE_MOUNT) \
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
		-e CONTEXT=$${CONTEXT:-$(MAX_CONTEXT_LEN)} \
		-e VRAM=$${VRAM:-$(GPU_MEMORY_GB)} \
		$(USER_ENV) \
		-e CONTAINER_USER=$(CONTAINER_USER) \
		-e HOST_IP=$(HOST_IP) \
		-e PORT=$(LAB_PORT) \
		-p 0.0.0.0:$(LAB_PORT):8888 \
		-v $(HOME_VOLUME):/home/$(CONTAINER_USER) \
		$(HOME_MOUNT_ARG) \
		$(MODEL_CACHE_MOUNT) \
		$(PROBE_CACHE_MOUNT) \
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
		-e CONTEXT=$${CONTEXT:-$(MAX_CONTEXT_LEN)} \
		-e VRAM=$${VRAM:-$(GPU_MEMORY_GB)} \
		$(USER_ENV) \
		-e CONTAINER_USER=$(CONTAINER_USER) \
		-v $(HOME_VOLUME):/home/$(CONTAINER_USER) \
		$(HOME_MOUNT_ARG) \
		$(MODEL_CACHE_MOUNT) \
		$(PROBE_CACHE_MOUNT) \
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
		-e CONTEXT=$${CONTEXT:-$(MAX_CONTEXT_LEN)} \
		-e VRAM=$${VRAM:-$(GPU_MEMORY_GB)} \
		$(USER_ENV) \
		-e CONTAINER_USER=$(CONTAINER_USER) \
		-v $(HOME_VOLUME):/home/$(CONTAINER_USER) \
		$(HOME_MOUNT_ARG) \
		$(MODEL_CACHE_MOUNT) \
		$(PROBE_CACHE_MOUNT) \
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

# test-vllm and test-idle remain manual-only. The recipes for
# ./tests/test-router-vllm.sh and ./tests/test-router-idle.sh stay in the
# tree but exercise containerRecreate against a live GPU — too slow for
# the default `make test` aggregate. Run them directly when changing
# vLLM/SGLang lifecycle code. See docs/backends.md.

test-probe-vllm: ## Smoke test: probe one HF model via vLLM, assert cache schema
	@# Wall time ~60-120s (one cold vLLM start). Requires `make cache-down`
	@# first — the prober self-checks for running router/vllm/sglang
	@# containers and aborts otherwise.
	./tests/test-probe-vllm.sh

test-probe-sglang: ## Smoke test: probe one HF model via SGLang, assert cache schema
	@# Wall time ~60-120s. Same precondition as test-probe-vllm.
	@# Notes that the schema is correct even when the model can't load
	@# (e.g. SGLang+FP4 on this image — the cache records `fits: false`
	@# with `evidence.kind: "infra"`).
	./tests/test-probe-sglang.sh

test-probe-ollama-idempotent: cache-up ## Verify the refactored Ollama prober is byte-idempotent on a populated cache
	@# Phase 1 byte-identical regression check. Runs `make probe` against
	@# the existing cache (which skips already-cached cells) and diffs
	@# the result. Catches drift in cache I/O, alias reconciliation,
	@# implied-spill builder shape, etc. Wall time ~10-30s.
	./tests/test-probe-ollama-idempotent.sh

test-vllm: cache-up ## Live integration tests for the vLLM backend (chat, ctx switch, GPU exclusion)
	@# Wall time ~3-5 min. Reads deploy/.vllm-reasoning-cache.json to
	@# pick models — skips cleanly when nothing's cached. See
	@# tests/test-router-vllm.sh for the test surface.
	./tests/test-router-vllm.sh

test-sglang: cache-up ## Live integration tests for the SGLang backend (mirror of test-vllm)
	@# Skips entirely when the SGLang cache has no fitting entries
	@# (e.g. when SGLang+FP4 fails on this image — see docs/backends.md).
	./tests/test-router-sglang.sh

test-e2e: cache-up ## End-to-end: picker discovery → agent command → live router chat
	@# Bridges Phase 7 picker tests and Phase 5 router tests. Imports
	@# model-picker.py, runs _discover_models + _build_menu against
	@# live caches, picks an HF row, replays the agent-emitted request
	@# through the router. Skips when no HF row is selectable.
	./tests/test-e2e-picker.sh

test-full: test ## Alias for `make test` (kept for backwards-compat; both run the full suite)
	@true


test-models: cache-up ## Matrix test: every probed model × wire protocol × scenario.
	@# Drives /api/chat, /v1/chat/completions, /v1/messages directly via curl
	@# (faster + more deterministic than spinning up real agent CLIs). For
	@# each candidate model materialises a Modelfile-derived ctx variant
	@# (matching what the picker does) and runs basic / tools / think_auto /
	@# think_off / ctx scenarios. Pin a specific subset via TEST_MODELS=...
	HOST_VRAM=$${VRAM:-$(GPU_MEMORY_GB)} \
	TEST_CTX=$${CONTEXT:-32768} \
	CONTAINER_RUNTIME=$(CONTAINER_RUNTIME) \
	  ./tests/test-model-matrix.sh

test-agents: ## Smoke-test every (agent × backend) cell against the live router
	@# Defensive cleanup: previous probe runs may have left
	@# vllm/sglang containers behind. Harmless when they don't exist.
	@$(CONTAINER_RUNTIME) rm -f devai-vllm devai-sglang 2>/dev/null || true
	@$(MAKE) cache-up
	@mkdir -p $(CURDIR)/tests/.matrix-logs
	@# Run via the default entrypoint so codex config gets seeded into
	@# /home/devai/.codex (CODEX_HOME). Image ENV provides ANTHROPIC_*
	@# and OPENAI_API_KEY so no overrides are needed here.
	$(CONTAINER_RUNTIME) run --rm \
		--name devai-test-agents \
		$(GPU_FLAGS) \
		--network $(DEVAI_NETWORK) \
		-e CONTAINER_USER=devai \
		$(MODEL_CACHE_MOUNT) \
		$(PROBE_CACHE_MOUNT) \
		-v "$(CURDIR)/tests/agent-matrix.sh":/usr/local/bin/agent-matrix:ro \
		-v "$(CURDIR)/tests/.matrix-logs":/var/log/agent-matrix \
		-v $(HOME_VOLUME):/home/devai \
		-e CELL_TIMEOUT=$${CELL_TIMEOUT:-60} \
		-e PROMPT="$${PROMPT:-reply with the single word PONG}" \
		-e ROUTER=devai-router \
		-e LOG_DIR=/var/log/agent-matrix \
		$(IMAGE_NAME_GPU) /usr/local/bin/agent-matrix
	@echo "  logs preserved at $(CURDIR)/tests/.matrix-logs/"

test: test-router test-probe-ollama-idempotent test-ollama test-e2e test-vllm test-sglang test-models ## Run every available test in sequence (Go unit + Ollama + E2E + vLLM/SGLang integration + matrix + probes; ~30-60 min)
	@# The cache-up suite runs as prerequisites above. Probe smoke tests
	@# require the live backends to be DOWN (the prober self-checks for
	@# router/vllm/sglang containers and aborts otherwise) — they run
	@# last with cache-down/cache-up bracketing so the operator's stack
	@# ends in a known-good state. Failures from probe tests are
	@# captured and re-emitted after `cache-up` so we never leave the
	@# stack down.
	@set -u; \
	echo ""; echo "=== Running cache-down probe tests (final phase) ==="; \
	$(MAKE) cache-down >/dev/null 2>&1 || true; \
	./tests/test-probe-vllm.sh;   vrc=$$?; \
	./tests/test-probe-sglang.sh; src=$$?; \
	$(MAKE) cache-up >/dev/null 2>&1 || true; \
	if [ "$$vrc" -ne 0 ] || [ "$$src" -ne 0 ]; then \
	    echo "probe-vllm rc=$$vrc, probe-sglang rc=$$src"; exit 1; \
	fi

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
	@printf "  %-44s%s\n" "MODELS" "vLLM / SGLang (lifecycle: docs/backends.md)"
	@printf "  %-44s%s\n" "probe            Populate cache for every (VRAM, ctx) tier" "vllm-list        List on-disk vLLM weights"
	@printf "  %-44s%s\n" "model-fit        Print fitting models at VRAM/CONTEXT" "vllm-rm          Remove model"
	@printf "  %-44s%s\n" "model-pull       Download best-fit candidates" "vllm-status      Show status"
	@printf "  %-44s%s\n" "  FAMILY=qwen3.5 scope to one family" ""
	@printf "  %-44s%s\n" "ollama-list      List downloaded models" "vllm-df          Disk usage"
	@printf "  %-44s%s\n" "ollama-rm        Remove model" ""
	@printf "  %-44s%s\n" "ollama-status    Show status" ""
	@printf "  %-44s%s\n" "ollama-df        Disk usage" ""
	@printf "  %s\n" "ollama-clean     Clean partials"
	@printf "\n"
	@printf "  %s\n" "DEPLOY"
	@printf "  %s\n" "install-systemd  Auto-start infrastructure at boot"
	@printf "\n"

# =============================================================================
# Infrastructure services (caches + Ollama + vLLM + Open WebUI)
# =============================================================================

cache-up: ## Start all infrastructure (caches + Ollama + router + Open WebUI; vLLM/SGLang as `sleep` placeholders, recreated on demand)
	@if [ "$(CONTAINER_RUNTIME)" = "podman" ] && ! systemctl --user is-active --quiet podman.socket; then \
		echo "Starting Podman API socket..."; \
		systemctl --user enable --now podman.socket; \
	fi
	@$(CONTAINER_RUNTIME) network exists $(DEVAI_NETWORK) 2>/dev/null || $(CONTAINER_RUNTIME) network create $(DEVAI_NETWORK)
	$(COMPOSE) -f $(CACHE_COMPOSE) up -d
	@echo "Infrastructure services started:"
	@echo "  apt-cacher-ng:     http://localhost:3142"
	@echo "  Registry mirror:   http://localhost:5000"
	@echo "  Router:            devai-router:11434 (unified endpoint)"
	@echo "  Ollama:            devai-ollama:11434 (GGUF models)"
	@echo "  vLLM/SGLang:       devai-router:11435 / 11436 (recreated on first request — see docs/backends.md)"
	@echo "  Open WebUI:        https://localhost:$(WEBUI_PORT)"
	@echo "  Logger:            $(CACHE_DIR)/logs/<container>.log (per-service stdout)"
	@echo ""
	@echo "To pull and probe models:  make model-pull [FAMILY=qwen3.5] && make probe"
	@echo "To tail a service's log:   make logs SERVICE=devai-ollama"

cache-down: ## Stop and remove ALL infrastructure services (running, stopped, orphaned)
	@# -t 0 kills immediately; --remove-orphans catches containers no longer in compose.
	-$(COMPOSE) -f $(CACHE_COMPOSE) down --remove-orphans -t 0
	@# Force-remove devai-vllm and devai-sglang explicitly. Compose launches
	@# both as `sleep infinity` placeholders, but the router (gpu-arbiter)
	@# replaces them via libpod when a request arrives — the recreated
	@# container drifts from compose's tracked spec (different entrypoint,
	@# args, no compose labels), so `compose down` may leave it behind as
	@# a zombie that blocks the next `cache-up` with "container name
	@# already in use". Remove by name, ignoring missing-container errors.
	@for name in devai-vllm devai-sglang; do \
		$(CONTAINER_RUNTIME) rm -f $$name >/dev/null 2>&1 || true; \
	done

setup-logs: ## One-time: create dedicated 100G LV at /var/cache/devai/logs (requires sudo).
	@# Stops cache so nothing holds the old logs path open, then runs the
	@# setup script under sudo (LVM + mkfs.xfs + /etc/fstab + mount). The
	@# script is idempotent — re-running on an already-set-up host does
	@# nothing destructive. Override SIZE/VG/LV via env if desired.
	@$(MAKE) cache-down
	@echo
	sudo SIZE=$${SIZE:-100G} VG=$${VG:-vgais} LV=$${LV:-cache_logs} \
	     POOL=$${POOL:-cachepool} RECREATE=$${RECREATE:-0} \
	  deploy/setup-logs-volume.sh
	@echo
	@echo "Next: run 'make cache-up' to start services on the new logs volume."

logs: ## Tail container stdout via the logger sidecar (SERVICE=devai-X, default devai-ollama; LINES=N to seed).
	@svc="$${SERVICE:-devai-ollama}"; \
	 lines="$${LINES:-50}"; \
	 file="$(CACHE_DIR)/logs/$$svc.log"; \
	 if [ ! -f "$$file" ]; then \
	    echo "no log for $$svc at $$file"; \
	    echo "available services:"; \
	    ls -1 $(CACHE_DIR)/logs 2>/dev/null | sed 's/^/  /; s/\.log$$//'; \
	    exit 1; \
	 fi; \
	 echo "tailing $$file (Ctrl-C to stop)"; \
	 tail -n $$lines -f "$$file"

cache-status: ## Show infrastructure service status and disk usage
	@$(COMPOSE) -f $(CACHE_COMPOSE) ps
	@echo ""
	@echo "Cache disk usage:"
	@du -sh $(CACHE_DIR)/*/ 2>/dev/null; true; true
	@echo ""
	@echo "Ollama models:"
	@out=$$($(OLLAMA_EXEC) list 2>/dev/null); \
	if [ -n "$$out" ]; then \
		header=$$(printf '%s' "$$out" | head -n1); \
		body=$$(printf '%s' "$$out" | tail -n +2 | sort -V -f); \
		printf '%s\n%s\n' "$$header" "$$body"; \
	else \
		echo "  (ollama not running)"; \
	fi
	@echo ""
	@echo "vLLM/SGLang models:"
	@vllm_cache=$(CURDIR)/deploy/.vllm-reasoning-cache.json; \
	sglang_cache=$(CURDIR)/deploy/.sglang-reasoning-cache.json; \
	found=false; for dir in $$(ls -d $(VLLM_MODELS_DIR)/*/ 2>/dev/null | sort -V -f); do \
		[ -f "$$dir/config.json" ] || continue; \
		if ! $$found; then \
			printf "%-46s%-16s%-14s%-10s%-20s\n" "NAME" "ID" "BACKENDS" "SIZE" "MODIFIED"; \
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
		backends=""; \
		[ -f "$$vllm_cache" ]   && grep -q "\"$$name\"" "$$vllm_cache"   && backends="vllm"; \
		[ -f "$$sglang_cache" ] && grep -q "\"$$name\"" "$$sglang_cache" && backends="$${backends:+$$backends,}sglang"; \
		[ -z "$$backends" ] && backends="-"; \
		printf "%-46s%-16s%-14s%-10s%-20s\n" "$$name" "$$id" "$$backends" "$$size" "$$modified"; \
	done; \
	$$found || echo "  (none — run 'make model-pull' to populate)"

cache-clean: ## Remove all cached data (keeps volumes mounted, preserves podman graphroot)
	$(COMPOSE) -f $(CACHE_COMPOSE) down
	@echo "Cleaning cache directories..."
	@# $(CACHE_DIR)/registry co-hosts podman's graphroot AND the registry:2 pull-through cache
	@# (see README "Storage Layout"). Wiping registry/* would destroy podman image storage,
	@# so we target only the registry:2 subtree (registry/docker) and guard every path
	@# against overlap with the container runtime's graphroot.
	@# Container-managed caches (apt-cacher-ng, registry:2) contain files owned by
	@# subuids, so the cleanup runs under `podman unshare` to access them.
	@graphroot=$$($(CONTAINER_RUNTIME) info --format '{{.Store.GraphRoot}}' 2>/dev/null \
	              || $(CONTAINER_RUNTIME) info --format '{{.DockerRootDir}}' 2>/dev/null \
	              || true); \
	targets="$(CACHE_DIR)/registry/docker $(CACHE_DIR)/apt $(CACHE_DIR)/pip $(CACHE_DIR)/npm"; \
	for target in $$targets; do \
		if [ -n "$$graphroot" ]; then \
			case "$$graphroot/" in \
				"$$target"/*) \
					echo "ERROR: refusing to clean $$target — $(CONTAINER_RUNTIME) graphroot ($$graphroot) is inside it."; \
					echo "       Relocate graphroot or clean selectively by hand."; \
					exit 1;; \
			esac; \
		fi; \
	done; \
	if [ "$(CONTAINER_RUNTIME)" = "podman" ]; then \
		wrap="$(CONTAINER_RUNTIME) unshare"; \
	else \
		wrap=""; \
	fi; \
	for target in $$targets; do \
		[ -d "$$target" ] || continue; \
		$$wrap find "$$target" -mindepth 1 -maxdepth 1 -exec rm -rf {} +; \
		echo "  cleaned $$target/"; \
	done
	@echo "Cache cleaned. Podman graphroot preserved."

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

vllm-list: ## List vLLM models with status
	@INFERENCE_CONFIG=$(INFERENCE_CONFIG) VLLM_MODELS_DIR=$(VLLM_MODELS_DIR) \
		python3 scripts/vllm-list.py

vram-fit: ## Show which models from the full catalog fit in VRAM (planning aid; use model-pull to act)
	@GPU_MEMORY_GB=$${VRAM:-$(GPU_MEMORY_GB)} MAX_CONTEXT_LEN=$${CONTEXT:-$(MAX_CONTEXT_LEN)} \
		python3 scripts/vram-fit.py \
			$(if $(VRAM),--vram $(VRAM),) \
			$(if $(CONTEXT),--context $(CONTEXT),) \
			$(if $(KV),--kv-dtype $(KV),) \
			$(if $(FAMILY),--family $(FAMILY),) \
			$(if $(FITS_ONLY),--fits-only,) \
			--models-yaml $(INFERENCE_CONFIG) \
			--vllm-dir $(VLLM_MODELS_DIR)

PROBE_VRAMS    ?= 16G,24G
PROBE_CONTEXTS ?= 32K,64K,128K,256K

probe: ## Probe every downloaded ollama digest at every (VRAM, CONTEXT) tier.
	@# Loops over PROBE_VRAMS, recreating devai-ollama with
	@# OLLAMA_GPU_OVERHEAD set so the daemon behaves as a smaller card.
	@# A 24G host can therefore produce cache entries valid for 16G targets.
	@# Each probe pass is incremental: existing (vram, ctx) cells are
	@# never overwritten unless PROBE_FORCE=1 or PROBE_FORCE_CTX=<list>.
	@set -e; \
	 for vram in $$(echo $(PROBE_VRAMS) | tr ',' ' '); do \
	    overhead_bytes=$$(python3 -c "import sys; sys.path.insert(0, 'scripts'); from _contexts import parse_vram_token, vram_overhead_bytes; print(vram_overhead_bytes($(GPU_MEMORY_GB), parse_vram_token('$$vram')))"); \
	    echo; \
	    echo ">>> probing at VRAM=$$vram (host=$(GPU_MEMORY_GB)G, OLLAMA_GPU_OVERHEAD=$$overhead_bytes bytes)"; \
	    $(CONTAINER_RUNTIME) rm -f $(OLLAMA_CONTAINER) >/dev/null 2>&1 || true; \
	    OLLAMA_GPU_OVERHEAD=$$overhead_bytes \
	      $(COMPOSE) -f $(CACHE_COMPOSE) up -d ollama; \
	    until $(CONTAINER_RUNTIME) exec $(OLLAMA_CONTAINER) ollama list >/dev/null 2>&1; do sleep 1; done; \
	    $(CONTAINER_RUNTIME) run --rm \
	        --network $(DEVAI_NETWORK) \
	        -v $(CURDIR)/scripts:/scripts:ro \
	        -v $(CURDIR)/deploy:/deploy \
	        -e OLLAMA_HOST=http://devai-ollama:11434 \
	        -e PROBE_CONTEXTS=$(PROBE_CONTEXTS) \
	        --entrypoint python3 \
	        $(IMAGE_NAME) \
	        /scripts/probe-ollama-reasoning.py \
	            --cache /deploy/.ollama-reasoning-cache.json \
	            --vram $$vram \
	            $(if $(PROBE_FORCE),--force,) \
	            $(if $(PROBE_FORCE_CTX),--force-ctx $(PROBE_FORCE_CTX),) \
	            $(PROBE_MODELS); \
	 done; \
	 echo; \
	 echo ">>> restoring devai-ollama to host VRAM (no overhead)"; \
	 $(CONTAINER_RUNTIME) rm -f $(OLLAMA_CONTAINER) >/dev/null 2>&1 || true; \
	 OLLAMA_GPU_OVERHEAD=0 $(COMPOSE) -f $(CACHE_COMPOSE) up -d ollama

probe-vllm: ## Probe every downloaded vLLM/HF model per (VRAM, CONTEXT) cell.
	@# Pre-condition: devai-router, devai-vllm, and devai-sglang must
	@# be stopped — the prober launches devai-vllm-probe with explicit
	@# GPU exclusivity. The script self-checks and aborts otherwise.
	@# Knobs:
	@#   PROBE_VRAMS_VLLM=16G,24G    target VRAM bands
	@#   PROBE_CONTEXTS=32K,...      ctx tiers
	@#   PROBE_REPO=<regex>          filter catalog rows by repo
	@#   PROBE_FORCE=1               re-probe every cell
	@#   PROBE_FORCE_ARCH=1          re-probe top-level capability/arch
	python3 scripts/probe-vllm-reasoning.py \
	    --host-vram-gb $(GPU_MEMORY_GB) \
	    $(if $(PROBE_VRAMS_VLLM),--vram $(PROBE_VRAMS_VLLM),) \
	    $(if $(PROBE_CONTEXTS),--ctx $(PROBE_CONTEXTS),) \
	    $(if $(PROBE_REPO),--repo $(PROBE_REPO),) \
	    $(if $(PROBE_FORCE),--force,) \
	    $(if $(PROBE_FORCE_ARCH),--force-arch,)

probe-sglang: ## Probe every downloaded SGLang/HF model per (VRAM, CONTEXT) cell.
	@# Pre-condition: same as probe-vllm — all GPU-owning backends down.
	@# Knobs:
	@#   PROBE_VRAMS_SGLANG=16G,24G  target VRAM bands
	@#   PROBE_CONTEXTS=32K,...      ctx tiers
	@#   PROBE_REPO=<regex>          filter catalog rows by repo
	@#   PROBE_FORCE=1               re-probe every cell
	@#   PROBE_FORCE_ARCH=1          re-probe top-level capability/arch
	python3 scripts/probe-sglang-reasoning.py \
	    --host-vram-gb $(GPU_MEMORY_GB) \
	    $(if $(PROBE_VRAMS_SGLANG),--vram $(PROBE_VRAMS_SGLANG),) \
	    $(if $(PROBE_CONTEXTS),--ctx $(PROBE_CONTEXTS),) \
	    $(if $(PROBE_REPO),--repo $(PROBE_REPO),) \
	    $(if $(PROBE_FORCE),--force,) \
	    $(if $(PROBE_FORCE_ARCH),--force-arch,)

model-fit: ## Print which models fit at the chosen (VRAM, CONTEXT) — diagnostic, no writes.
	@OLLAMA_CONTAINER=$(OLLAMA_CONTAINER) CONTAINER_RUNTIME=$(CONTAINER_RUNTIME) \
	 VLLM_MODELS_DIR=$(VLLM_MODELS_DIR) HF_CLI=$(HF_CLI) \
	 GPU_MEMORY_GB=$${VRAM:-$(GPU_MEMORY_GB)} MAX_CONTEXT_LEN=$${CONTEXT:-$(MAX_CONTEXT_LEN)} \
	 VERBOSE=$${VERBOSE:-0} \
		python3 scripts/select-models.py \
			$(if $(FAMILY),--family $(FAMILY),) \
			$(if $(VRAM),--vram $(VRAM),) \
			$(if $(CONTEXT),--context $(CONTEXT),) \
			$(if $(CONTEXTS),--contexts $(CONTEXTS),) \
			$(if $(KV),--kv-dtype $(KV),)

model-pull: ## Pull missing best-fit candidates from the catalog (catalog-driven downloads).
	@set -e; \
	 OLLAMA_CONTAINER=$(OLLAMA_CONTAINER) CONTAINER_RUNTIME=$(CONTAINER_RUNTIME) \
	 VLLM_MODELS_DIR=$(VLLM_MODELS_DIR) HF_CLI=$(HF_CLI) \
	 GPU_MEMORY_GB=$${VRAM:-$(GPU_MEMORY_GB)} MAX_CONTEXT_LEN=$${CONTEXT:-$(MAX_CONTEXT_LEN)} \
	 VERBOSE=$${VERBOSE:-0} \
		python3 scripts/select-models.py \
			$(if $(FAMILY),--family $(FAMILY),) \
			$(if $(VRAM),--vram $(VRAM),) \
			$(if $(CONTEXT),--context $(CONTEXT),) \
			$(if $(CONTEXTS),--contexts $(CONTEXTS),) \
			$(if $(KV),--kv-dtype $(KV),) \
			--download \
			$(if $(DOWNLOAD_LIMIT),--max-downloads $(DOWNLOAD_LIMIT),) \
			$(if $(PRUNE),--prune,) \
			$(if $(PRUNE_SHADOWS),--prune-shadows,) \
			$(if $(DRY_RUN),--dry-run,); \
	 echo; \
	 echo "  next: 'make probe' to populate cache cells for any newly pulled tags"

catalog-regen: ## Regenerate deploy/models.yaml from scripts/model-families.yaml using live upstream data
	python3 scripts/generate-catalog.py

catalog-suggest: ## Suggest GGUF candidates llmfit ranks well that aren't yet in scripts/model-families.yaml. Read-only; probe before adding.
	@command -v llmfit >/dev/null 2>&1 || { \
	  echo "error: llmfit not found in PATH (install: https://github.com/AlexsJones/llmfit)"; exit 1; }
	@VRAM=$${VRAM:-$(GPU_MEMORY_GB)}; \
	 CTX=$${CONTEXT:-$(MAX_CONTEXT_LEN)}; \
	 LIMIT=$${LIMIT:-30}; \
	 echo "# llmfit recommend --runtime llamacpp --memory $${VRAM}G --max-context $${CTX} -n $${LIMIT}"; \
	 echo; \
	 llmfit --memory "$${VRAM}G" --max-context "$${CTX}" \
	   recommend --runtime llamacpp --json -n "$${LIMIT}" \
	   $(if $(USE_CASE),--use-case $(USE_CASE),) \
	   $(if $(MIN_FIT),--min-fit $(MIN_FIT),) \
	 | python3 scripts/llmfit-catalog-diff.py

verify-backend-flags: ## Assert pinned vLLM/SGLang images expose every flag in deploy/backend-flags.yaml (run after image bump)
	python3 scripts/verify-backend-flags.py

ollama-cleanup-ctx-variants: ## Remove every derived `-ctx<N>` tag from Ollama. Safe — they share weight blobs with parents; only Modelfile metadata is freed.
	@$(CONTAINER_RUNTIME) exec devai-ollama sh -c 'ollama list | awk "NR>1 && \$$1 ~ /-ctx[0-9]+\$$/ {print \$$1}"' | \
	  while IFS= read -r tag; do \
	    [ -n "$$tag" ] || continue; \
	    printf '  rm %s ... ' "$$tag"; \
	    $(CONTAINER_RUNTIME) exec devai-ollama ollama rm "$$tag" >/dev/null 2>&1 \
	      && echo "ok" || echo "FAILED"; \
	  done; \
	  echo; \
	  remaining=$$($(CONTAINER_RUNTIME) exec devai-ollama sh -c 'ollama list | grep -cE -- "-ctx[0-9]+\b"' 2>/dev/null); \
	  echo "  remaining -ctx<N> tags: $$remaining"

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

INSTALL_PREFIX ?= $(HOME)/.local
DEVAI_HOME ?= $(HOME)/.devai

install: ## Install bin/devai-agent to $(INSTALL_PREFIX)/bin and stage config in $(DEVAI_HOME)
	@install -d $(INSTALL_PREFIX)/bin $(DEVAI_HOME) $(DEVAI_HOME)/sessions
	@# Symlink rather than copy — picks up edits to bin/devai-agent without
	@# re-running `make install`. argparse's prog name will show the repo
	@# path in --help; that's honest and a non-issue in practice.
	@ln -sf "$(CURDIR)/bin/devai-agent" $(INSTALL_PREFIX)/bin/devai-agent
	@chmod +x "$(CURDIR)/bin/devai-agent"
	@# Symlink the picker so devai-agent can override the in-image copy via
	@# bind-mount; re-running `make install` after picker edits picks up the
	@# new code without a full image rebuild.
	@ln -sf "$(CURDIR)/scripts/model-picker.py" $(DEVAI_HOME)/model-picker.py
	@# Symlink each backend's probe cache so it stays fresh as the prober
	@# regenerates it. If users want a frozen snapshot they can replace the
	@# link with a copy after install. Missing caches are warned but not
	@# fatal — the picker tolerates absent backend caches.
	@for cache in ollama vllm sglang; do \
		src="$(CURDIR)/deploy/.$$cache-reasoning-cache.json"; \
		dst="$(DEVAI_HOME)/.$$cache-reasoning-cache.json"; \
		if [ -f "$$src" ]; then \
			ln -sf "$$src" "$$dst"; \
			echo "  linked: $$dst"; \
		else \
			echo "  WARNING: $$src missing — run 'make probe' (or probe-$$cache) first"; \
		fi; \
	done
	@echo "  linked: $(DEVAI_HOME)/model-picker.py"
	@echo "  installed: $(INSTALL_PREFIX)/bin/devai-agent"
	@echo
	@echo "Next steps:"
	@echo "  1. Add $(INSTALL_PREFIX)/bin to PATH if not already."
	@echo "  2. devai-agent --init     # create $(DEVAI_HOME)/preferences.yaml"
	@echo "  3. devai-agent            # launch the lab + picker"

uninstall: ## Remove devai-agent launcher and the staged config dir
	@rm -f $(INSTALL_PREFIX)/bin/devai-agent
	@rm -f $(DEVAI_HOME)/.ollama-reasoning-cache.json
	@rm -f $(DEVAI_HOME)/.vllm-reasoning-cache.json
	@rm -f $(DEVAI_HOME)/.sglang-reasoning-cache.json
	@rm -f $(DEVAI_HOME)/model-picker.py
	@echo "Removed $(INSTALL_PREFIX)/bin/devai-agent and the symlinks under $(DEVAI_HOME)/."
	@echo "preferences.yaml and sessions/ are kept; remove $(DEVAI_HOME)/ manually if you want a clean slate."

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


# =============================================================================
# Bench harness — see scripts/bench/ and docs/router.md "Benchmark harness".
# =============================================================================

# Cache mounts mirror what the router uses so the runner reads the same
# probe data the live stack does.
BENCH_CACHE_MOUNTS = \
	-v $(CURDIR)/scripts:/scripts:ro \
	-v $(CURDIR)/deploy:/deploy \
	-v $(CACHE_DIR)/bench:$(CACHE_DIR)/bench

# n-knobs surface to the runner as both env (for inspect_ai's task
# constructors that read defaults) and CLI flags (the runner reads them
# either way; CLI takes precedence).
BENCH_RUN_FLAGS = \
	$(if $(BENCH_TASKS),--tasks '$(BENCH_TASKS)',) \
	$(if $(BENCH_REPO),--repo '$(BENCH_REPO)',) \
	$(if $(BENCH_FORCE),--force,) \
	$(if $(BENCH_N_GSM8K),--n-gsm8k $(BENCH_N_GSM8K),) \
	$(if $(BENCH_N_HUMANEVAL),--n-humaneval $(BENCH_N_HUMANEVAL),) \
	$(if $(BENCH_N_TOOLS),--n-tools $(BENCH_N_TOOLS),) \
	$(if $(BENCH_N_LEAK_PROMPTS),--n-leak-prompts $(BENCH_N_LEAK_PROMPTS),)

bench: bench-vllm bench-sglang bench-ollama ## Bench every probed model on every backend
	@echo
	@echo ">>> bench complete; run 'make bench-report' for the leaderboard"

bench-vllm: ## Bench every loaded vLLM/HF model via devai-router:11435
	@# Pre-condition: devai-router + devai-vllm reachable on devai-net
	@# (run 'make cache-up'). nvidia-smi must be on PATH inside the
	@# lab image so the VRAM sampler reads memory.used.
	@mkdir -p $(CACHE_DIR)/bench/inspect-logs
	$(CONTAINER_RUNTIME) run --rm \
		--network $(DEVAI_NETWORK) \
		$(BENCH_CACHE_MOUNTS) \
		$(GPU_FLAGS) \
		-e GPU_MEMORY_GB=$(GPU_MEMORY_GB) \
		--entrypoint python3 \
		$(IMAGE_NAME_GPU) \
		/scripts/bench/bench_runner.py --backend vllm \
			$(BENCH_RUN_FLAGS)

bench-sglang: ## Bench every loaded SGLang model via devai-router:11436
	@mkdir -p $(CACHE_DIR)/bench/inspect-logs
	$(CONTAINER_RUNTIME) run --rm \
		--network $(DEVAI_NETWORK) \
		$(BENCH_CACHE_MOUNTS) \
		$(GPU_FLAGS) \
		-e GPU_MEMORY_GB=$(GPU_MEMORY_GB) \
		--entrypoint python3 \
		$(IMAGE_NAME_GPU) \
		/scripts/bench/bench_runner.py --backend sglang \
			$(BENCH_RUN_FLAGS)

bench-ollama: ## Bench every loaded Ollama model via devai-router:11434
	@mkdir -p $(CACHE_DIR)/bench/inspect-logs
	$(CONTAINER_RUNTIME) run --rm \
		--network $(DEVAI_NETWORK) \
		$(BENCH_CACHE_MOUNTS) \
		$(GPU_FLAGS) \
		-e GPU_MEMORY_GB=$(GPU_MEMORY_GB) \
		--entrypoint python3 \
		$(IMAGE_NAME_GPU) \
		/scripts/bench/bench_runner.py --backend ollama \
			$(BENCH_RUN_FLAGS)

bench-report: ## Print a Markdown leaderboard from .bench-cache.json
	@$(CONTAINER_RUNTIME) run --rm \
		-v $(CURDIR)/scripts:/scripts:ro \
		-v $(CURDIR)/deploy:/deploy:ro \
		--entrypoint python3 \
		$(IMAGE_NAME) \
		/scripts/bench/bench_report.py \
			--cache /deploy/.bench-cache.json

test-bench-smoke: ## 1-model tiny-subset smoke test (CI / sanity)
	@# Picks the smallest fitting vLLM model and runs n=5 on every
	@# task type plus n=10 on the latency probe. Should finish in
	@# under 5 minutes on a 24G GPU once the model is warm.
	@mkdir -p $(CACHE_DIR)/bench/inspect-logs
	$(CONTAINER_RUNTIME) run --rm \
		--network $(DEVAI_NETWORK) \
		$(BENCH_CACHE_MOUNTS) \
		$(GPU_FLAGS) \
		-e GPU_MEMORY_GB=$(GPU_MEMORY_GB) \
		--entrypoint python3 \
		$(IMAGE_NAME_GPU) \
		/scripts/bench/bench_runner.py --backend vllm \
			--repo "Qwen3-8B-NVFP4" \
			--n-gsm8k 5 --n-humaneval 5 --n-tools 4 --n-leak-prompts 10 \
			--force
