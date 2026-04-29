# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **Dev AI Lab** — a containerized development environment for AI experimentation featuring JupyterLab and multiple AI CLIs (Gemini, Claude, OpenAI, Ollama). Built on Debian Trixie with Python 3.13 (uv-managed), Node.js 22 LTS. Two-layer image build for fast iteration. Compatible with Podman and Docker. GPU/CUDA support.

**Backends:** all three are wired — Ollama (GGUF, port 11434), vLLM (NVFP4/safetensors, port 11435), SGLang (NVFP4/safetensors, port 11436). The router enforces GPU mutual exclusion: only one backend serves at a time. vLLM and SGLang start as `sleep infinity` placeholders and are recreated on demand by the router when a request arrives. See `docs/backends.md` for the lifecycle, probing procedure, and failure-mode taxonomy.

## Build and Run Commands

```bash
# Build
make build-cpu       # Build base + lab image (CPU)
make build-gpu       # Build base + lab image (GPU/CUDA)
make build-router    # Build gpu-arbiter router image
make build           # Build all (CPU + GPU + router)

# Run
make lab-gpu         # Run JupyterLab with GPU
make lab-cpu         # Run JupyterLab CPU-only
make shell-gpu       # Interactive shell (GPU) — cwd = repo root
make shell-cpu       # Interactive shell (CPU) — cwd = repo root

# Install standalone launcher (one-time; stages launcher + config under ~/.devai/)
make install         # writes ~/.local/bin/devai-shell + ~/.devai/{model-picker.py,probe-cache symlink}
make uninstall       # removes the launcher + symlinks

# Standalone launcher (state in ~/.devai/, independent of repo cwd)
devai-shell --init           # reset ~/.devai/preferences.yaml to defaults
devai-shell                  # GPU lab; reads/writes preferences.yaml
devai-shell --cpu
devai-shell -C ~/projects/foo --model qwen3.5:9b-q8_0 --agent claude
devai-shell --show           # print resolved prefs + podman command, no run

# Infrastructure
make cache-up        # Start all services. vLLM/SGLang start as `sleep` placeholders; router recreates on demand.
make cache-down      # Stop all services
make cache-status    # Show status, models, disk usage

# Models — three orthogonal commands after the resilient-splashing-peach refactor
make probe                      # Probe each downloaded digest at every (VRAM, ctx) tier
make model-fit                  # Print which models fit at chosen VRAM/CONTEXT (no writes)
make model-pull                 # Download best-fit candidates from catalog (catalog-driven)
make model-pull FAMILY=qwen3.5  # Scope to one family
make ollama-list                # List downloaded Ollama models
make vllm-list                  # List on-disk vLLM/SGLang weights

# Logging (logger sidecar persists each container's stdout)
make logs SERVICE=devai-ollama  # Tail one container's persisted log
make logs SERVICE=devai-router LINES=200
make setup-logs                 # One-time: 100G LV at /var/cache/devai/logs (sudo)

# Tests
make test-router                # Go unit tests for arbiter
make test-ollama                # Ollama integration tests
make test-models                # Matrix: every probed digest × wire × scenario
make test-vllm                  # Live vLLM integration (chat, ctx switch, GPU exclusion)
make test-sglang                # Live SGLang integration (skips when not loadable)
make test-e2e                   # Picker → agent command → live router chat
make test-probe-vllm            # Probe smoke: cache schema assertion (requires cache-down)
make test-probe-sglang          # Same for SGLang
make test-probe-ollama-idempotent  # Byte-identical regression check on refactored Ollama prober
make test                       # All of the above in sequence (~30-60 min wall time)

# Maintenance
make clean           # Remove all images (CPU + GPU + router)
make prune           # Prune dangling images
make help            # Show all targets
```

## Configuration

Copy `.env.example` to `.env` before first use. Key settings:

- `LAB_PORT` — JupyterLab port (default: 8888)
- `WEBUI_PORT` — Open WebUI HTTPS port (default: 8443)
- `CONTAINER_RUNTIME` — `podman` (default) or `docker`
- `HOST_HOME_DIR` — Host home directory for .gitconfig/.ssh mounting
- `HOME_VOLUME` — Persistent home directory path
- `JUPYTER_TOKEN` — Fixed JupyterLab access token
- `HTTP_PROXY`/`HTTPS_PROXY` — Proxy settings for corporate environments

Model catalog is in `deploy/models.yaml` (single source of truth for all models). GPU inference settings:

- `GPU_MEMORY_GB` — Total GPU VRAM in GB (default: 24). Used by router to calculate memory fractions and context limits.
- `MAX_CONTEXT_LEN` — Default max context length in tokens (default: 131072 = 128K). The router caps each model's per-name context at `min(model.max_context, MAX_CONTEXT_LEN)`. The probe cache (`deploy/.ollama-reasoning-cache.json`) is the source of truth — `deploy/active-models.yaml` no longer exists.

## Architecture

### Lab container (devai-lab-cpu / devai-lab-gpu)

Two-layer image build (base rarely changes, lab layer for fast iteration):

**Layer 1: Dockerfile.base** — System packages, Python 3.13 (via uv), Node.js 22 LTS
**Layer 2: Dockerfile.lab** — CLI binaries (Claude, Codex, Ollama, Gemini, code-server), PyTorch, Python packages, JupyterLab

Build cache: CLI binaries pre-downloaded to `/var/cache/devai/pip/bin/` via `make fetch-cli` (ETag-based updates). Mounted into build — no network downloads during rebuild.

### Inference stack (deploy/docker-compose.yaml)

```
Agent → devai-router:11434 → devai-ollama:11434 (GGUF models)
Agent → devai-router:11435 → devai-vllm:11434   (NVFP4 / safetensors via vLLM)
Agent → devai-router:11436 → devai-sglang:11434 (NVFP4 / safetensors via SGLang)
```

- **devai-router** — Multi-port GPU-aware reverse proxy. One port per backend. No message inspection — port determines backend. Manages GPU exclusion (only one backend uses GPU at a time), graceful drain on switch, idle timeout, dynamic GPU memory allocation (`--gpu-memory-utilization` for vLLM, `--mem-fraction-static` for SGLang). Per-request context cap comes from `<name>@<ctx>` override (picker-supplied) or the probe cache row's `min(model.max_context, MAX_CONTEXT_LEN)`. Both `currentModel` and `currentContext` are tracked per backend; either change triggers a recreate.
- **devai-ollama** — Unmodified `ollama/ollama:latest`. GGUF models, GPU auto-detected. `OLLAMA_MAX_LOADED_MODELS=1` ensures clean model switching. `OLLAMA_CONTEXT_LENGTH` defaults to 262144 (compose env).
- **devai-vllm** — `vllm/vllm-openai` image. NVFP4 / safetensors models. Starts as a `sleep infinity` placeholder; the router recreates the container with the dynamic entrypoint (model path, max-model-len, gpu-memory-utilization, optional `--tool-call-parser`) on first request to port 11435.
- **devai-sglang** — `lmsysorg/sglang` image. NVFP4 / safetensors with RadixAttention for multi-turn speedup. Same `sleep infinity` placeholder + on-demand recreate pattern as vLLM.
- **devai-webui-proxy** — nginx TLS proxy for Open WebUI (mkcert certs or self-signed fallback).
- **devai-open-webui** — Web chat interface, connects to router's ollama port (:11434).
- **devai-logger** — Sidecar that streams `podman --remote logs --follow` for every devai-* container into `/var/cache/devai/logs/<service>.log`. Survives container restarts. Tail via `make logs SERVICE=<name> [LINES=N]`. Requires the `cache_logs` LV (one-time setup via `make setup-logs`).

### Supporting services

- **apt-cacher-ng** — APT package cache (port 3142)
- **Registry mirror** — Docker Hub pull-through cache (port 5000)

All services share `devai-net` network. Model data stored under `/var/cache/devai/`.

### SSL / HTTPS

- JupyterLab: auto-detects mkcert certs in `~/.jupyter/ssl/`
- Open WebUI: nginx proxy with mkcert certs or self-signed fallback
- Generate with `mkcert <IP>` on browser workstation, copy to container host

### Model picker (shell + Jupyter)

Interactive model → agent selection via fzf. Used by `make shell-*` (via `agent-picker`), the standalone `devai-shell` launcher, and JupyterLab launcher cards.

- `scripts/model-picker.py` — Python TUI, two-step fzf picker. Reads all three probe caches (Ollama digest-keyed v3; vLLM/SGLang repo+sha-keyed v1) for fit data. Falls back to `deploy/models.yaml` for catalog metadata only. Backend is derived from the chosen model's cache entry — there's no explicit backend step.
- `scripts/agent-picker.sh` — Shell wrapper, execs model-picker.py.
- `bin/devai-shell` — Host-side Python launcher. Reads/writes `~/.devai/preferences.yaml` (vram, context, last_model, last_agent, last_work_dir, agent_session_file). Bind-mounts `~/.devai/` to `/devai-host` (rw) so the picker can drop `.last-pick.json` for the launcher to consume on exit. Bind-mounts `~/.devai/model-picker.py` over `/usr/local/bin/model-picker` so picker edits don't require an image rebuild. Pre-flight checks for image + `devai-net`.
- `packages/jupyter-ai-launchers/src/index.ts` — JupyterLab extension, each card runs `model-picker --agent <name>`.

**Filter:** the picker shows one row per (family, context tier, reasoning status) bucket at the picker's VRAM band (env `VRAM` or `GPU_MEMORY_GB`). A model is eligible at a (vram, ctx) cell only when the relevant probe cache contains a measurement there with `fully_on_gpu: true` (Ollama) or `fits: true` (vLLM/SGLang). There is no interpolation — gaps mean "re-run `make probe-vllm` / `make probe-sglang`". HF rows whose backend has no fitting probe entry stay hidden until probed. See `docs/backends.md`.

**Per-session context binding.** Two paths because the backends differ:

- **Ollama**: the picker derives a session-scoped Ollama tag `<parent>-ctx<N>` (e.g. `qwen3.5:9b-q8_0-ctx32768`) by calling `/api/create` with the structured `{model, from, parameters: {num_ctx: N}}` body. Derived tags share weight blobs with the parent (sub-second creation, no extra disk). Required because Ollama 0.21.x silently ignores `options.num_ctx` on `/v1/chat/completions` and `/v1/messages` — Modelfile-baked `PARAMETER num_ctx` is the only universal mechanism. The router's policy lookup peels the `-ctx<N>` suffix (`stripCtxVariantSuffix` in `gpu-arbiter/main.go`) so the parent's reasoning entry still applies. The probe driver filters `-ctx<N>` derived tags out of `/api/tags` so they're never re-probed.
- **vLLM / SGLang**: the picker emits `<name>@<ctx>` (e.g. `Llama-3.1-8B-Instruct-NVFP4@32768`). The router's `parseCtxOverride` strips the suffix, propagates the ctx into `containerRecreate`, which sets `--max-model-len` (vLLM) or `--context-length` (SGLang) on the dynamic entrypoint. No client-side tag materialization needed — the router's `currentContext` tracking handles the rest.

### Building the JupyterLab extension

Never build on the host. Build inside a container:

```bash
podman run --rm -it \
  -v "$(pwd)/packages/jupyter-ai-launchers:/src:z" \
  -w /src \
  devai-lab-cpu bash -c "jlpm install && jlpm build:prod"
```

Pre-built output lives in `packages/jupyter-ai-launchers/jupyter_ai_launchers/labextension/` and is copied into the image by Dockerfile.lab. Rebuild after changing `src/index.ts`.

### Key files

```
deploy/models.yaml            — Auto-generated catalog (every variant the upstream catalog declares)
deploy/.ollama-reasoning-cache.json — Ollama probe cache (schema v3, digest-keyed)
deploy/.vllm-reasoning-cache.json   — vLLM probe cache (schema v1, repo+sha-keyed)
deploy/.sglang-reasoning-cache.json — SGLang probe cache (schema v1, repo+sha-keyed)
deploy/docker-compose.yaml    — Infrastructure services (vllm/sglang start as `sleep` placeholders; router recreates on demand)
deploy/Dockerfile.base        — Base image
deploy/Dockerfile.lab         — Lab image
deploy/Dockerfile.router      — Router image (distroless)
gpu-arbiter/main.go           — GPU arbiter source (multi-port proxy, ~1070 lines Go)
scripts/generate-catalog.py   — Refresh deploy/models.yaml from upstream (HF + Ollama registry)
scripts/_probe_core.py        — Backend-agnostic probe helpers (cache I/O, classifier, implied-spill propagation)
scripts/_probe_hf_common.py   — Shared scaffold for vLLM/SGLang probers (BackendSpec, podman driver, /v1 chat probe, nvidia-smi)
scripts/probe-ollama-reasoning.py — Ollama prober (Make-orchestrated VRAM bands)
scripts/probe-vllm-reasoning.py   — vLLM prober (BackendSpec wrapper)
scripts/probe-sglang-reasoning.py — SGLang prober (BackendSpec wrapper)
scripts/select-models.py      — Print fitting models / pull missing best-fit candidates (gguf path emits FROM + RENDERER + PARSER Modelfile, runs ollama create)
scripts/model-picker.py       — Two-step interactive picker (model → agent); creates per-session `<parent>-ctx<N>` tag with PARAMETER num_ctx baked in
deploy/setup-logs-volume.sh   — Idempotent LVM/XFS/fstab setup for /var/cache/devai/logs (called by `make setup-logs`)
deploy/logging.sh             — Logger sidecar entrypoint (runs `podman --remote logs --follow` per devai-* container)
tests/test-router.sh          — Ollama-side router integration tests
tests/test-model-matrix.sh    — Exhaustive matrix: every probed digest × wire × scenario
docs/backends.md              — Lifecycle, probing, cache hygiene, failure-mode taxonomy across all 3 backends
```
# AK's CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Don't Assume — If You Don't Know, Say So

**Never claim something works, exists, or behaves a certain way unless you've verified it. If you haven't, say "I don't know" or "I haven't tested that".**

Banned phrases when not verified:
- "this works"
- "verified"
- "should work"
- "this is wired up"
- "X is fine"

Required phrases when uncertain:
- "I don't know"
- "I haven't tested this"
- "I'm assuming X but haven't confirmed"
- "This is unverified"

Before reporting status, audit each claim:
- Did I run it end-to-end? → can claim "works"
- Did I only check it parses / imports / starts? → say "starts cleanly, full round-trip not tested"
- Did I infer from documentation, help text, or another agent's behavior? → say "I'm assuming based on X, not verified"

A "PASS" in a test harness means PASS only for what the harness actually checked. If the harness checks "non-empty output", report that, not "works".

When the user asks "does X work?", the only honest answers are:
- "Yes — verified by [specific test/command/output]"
- "No — fails at [specific point], log/error attached"
- "I don't know — haven't tested"

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, fewer false-confidence claims that have to be retracted later, and clarifying questions come before implementation rather than after mistakes.
