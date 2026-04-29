# Dev AI Lab

**Run AI models entirely on your own hardware.** No cloud APIs, no data leaving your network, no per-token costs.

Dev AI Lab is a containerized development environment that brings together JupyterLab, multiple AI coding assistants, and local model inference into a single, reproducible setup. It runs open-weight LLMs on your GPU — from 4B parameter models for quick tasks to 70B-class MoE models with active-parameter inference speed — all served through a unified API with automatic GPU management and runtime-verified capability detection.

### Why local inference?

Cloud AI services (ChatGPT, Claude API, Gemini API) are convenient but come with trade-offs that matter in professional settings:

- **Data sovereignty** — your code, documents, and conversations never leave your machine.
- **Regulatory compliance** — meet data residency requirements (GDPR, HIPAA, financial regulations).
- **Cost predictability** — no per-token billing, no surprise invoices.
- **No internet dependency** — works offline, on air-gapped networks, behind restrictive firewalls.
- **Full control** — choose your models, quantization, context length, and serving parameters.
- **Privacy by design** — conversations about proprietary code stay completely private.

Dev AI Lab makes local inference practical by handling the operational complexity: container builds, model management, GPU arbitration between multiple backends, runtime probing for actual VRAM use and reasoning capability, and a web chat UI — all through a single `make` command.

For tasks where cloud AI is appropriate, the JupyterLab environment also includes **Claude Code**, **OpenAI Codex**, and **Google Gemini CLI** — giving you the flexibility to use local models for sensitive work and cloud models when you need their capabilities.

## Features

- **Two-step interactive picker** — pick a model, then an agent. Arrow-key navigation via fzf in the shell (`make shell-gpu`); same flow from JupyterLab launcher cards.
- **Probe-verified model facts** — every downloaded model is probed against the live ollama runtime. Reasoning behavior (Native reasoning / Inline reasoning / No reasoning) plus CPU offload, MoE expert counts, and *actual* VRAM use at the configured context length are measured, not guessed. Cached by model digest so re-runs are fast.
- **MoE / dense awareness** — the picker labels each model with `MoE 8/128` (8 of 128 experts active per token) or `dense`. Same fit rules apply (full weights must be GPU-resident), but you can see at a glance which models give big-model-quality at small-model speed.
- **Multiple AI CLIs pre-installed** — Claude Code, OpenAI Codex, Google Gemini CLI, Aider, LATE, Open Interpreter, Ollama. All wired through the local router by default.
- **VS Code in the browser** — code-server provides a full Visual Studio Code experience accessible from any browser.
- **Automatic GPU-arbitrated model serving** — Ollama for GGUF models, vLLM/SGLang for NVFP4 models. The gpu-arbiter router transparently switches backends and exposes a single endpoint per protocol (`/api/chat`, `/v1/chat/completions`, `/v1/messages`).
- **Reasoning policy at the router** — set `DEVAI_REASONING=auto|off|low|medium|high` to control thinking mode globally; override per-request via `X-DevAI-Reasoning` header. The router maps your policy to the right native protocol field (Ollama's `think:`) based on each model's verified capability.
- **Open WebUI chat interface** — web-based chat UI over HTTPS that sees all available models.
- **Fast iteration** — two-layer container build separates rarely-changing system packages from frequently-updated tools.
- **Aggressive caching** — CLI binaries via ETags, APT proxy, Docker Hub mirror.
- **Works with Podman and Docker** — rootless Podman is the default.

## Quick Start

### Prerequisites (Debian 13 Trixie)

```bash
# Podman + Compose
sudo apt install podman python3-podman-compose

# Python tools (for model management scripts)
sudo apt install python3-yaml

# HuggingFace CLI (for vLLM model downloads)
pip install --user huggingface-hub

# NVIDIA GPU drivers (if not already installed)
sudo apt install nvidia-driver nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=podman --config=$HOME/.config/containers/containers.conf
```

### 1. Configure

```bash
cp .env.example .env
# Edit .env: set JUPYTER_TOKEN, GPU_MEMORY_GB, adjust ports if needed
```

### 2. Build

```bash
make build           # Build all images (CPU + GPU + router)
```

### 3. Start Infrastructure

```bash
make cache-up        # Start all services. vLLM/SGLang start as `sleep` placeholders;
                     # router recreates them on demand. See docs/backends.md.
```

### 4. Pull, Probe & Select

Three orthogonal commands. `make probe` populates the probe cache (single source
of truth); `make model-fit` queries it; `make model-pull` downloads.
Add `DOWNLOAD=1` to also pull missing variants in the same run.

```bash
make probe                                    # probe every (VRAM, ctx) cell
make probe PROBE_VRAMS=24G PROBE_CONTEXTS=128K # one band, one tier
make model-pull                               # download missing best-fit candidates
make model-pull FAMILY=qwen3.5                # scope to one family
make model-fit                                # print fitting models at host VRAM × MAX_CONTEXT_LEN
make model-fit VRAM=16 CONTEXT=32768          # query a different (VRAM, ctx)
```

### 5. Run

```bash
make lab-gpu         # Start JupyterLab with GPU (or make lab-cpu)
# OR
make shell-gpu       # Drop straight into the model picker (cwd = repo)
# OR (standalone host launcher — see "devai-shell" below)
make install         # one-time: stage launcher + config in ~/.local/bin and ~/.devai/
devai-shell --init   # one-time: write default ~/.devai/preferences.yaml
devai-shell          # launch with last-used model/agent/work-dir
```

Access:
- **JupyterLab**: `https://<HOST_IP>:8888`
- **Open WebUI**: `https://<HOST_IP>:8443`

### `devai-shell` — standalone host launcher

`bin/devai-shell` is the same lab container as `make shell-gpu`,
runnable from anywhere on the host without invoking Make. State lives
under `~/.devai/`; the repo is only consulted at `make install` time.

**Install once:**

```bash
make install                       # default INSTALL_PREFIX=~/.local
make install INSTALL_PREFIX=/opt   # alternative location
```

This writes:

| Target | Purpose |
|---|---|
| `~/.local/bin/devai-shell` | The launcher script. |
| `~/.devai/.ollama-reasoning-cache.json` | Symlink to the repo's probe cache so it stays fresh as `make probe` regenerates it. |
| `~/.devai/model-picker.py` | Symlink so the launcher can override the in-image picker via bind-mount (no rebuild needed). |
| `~/.devai/sessions/` | Per-`(agent, model)` session-history dir. |

Then add `~/.local/bin` to `PATH` and run `devai-shell --init` to seed
`~/.devai/preferences.yaml` with defaults.

**Run:**

```bash
devai-shell                         # GPU; reads ~/.devai/preferences.yaml
devai-shell --cpu                   # CPU lab
devai-shell -C ~/projects/my-app    # override last_work_dir for this run
devai-shell --model qwen3.5:9b-q8_0 --agent claude
devai-shell --show                  # print resolved prefs + container cmd, no run
devai-shell --init                  # reset preferences.yaml to defaults
uninstall via:  make uninstall      # removes the launcher + symlinks
```

**Preferences (`~/.devai/preferences.yaml`).** The launcher reads
these on entry and updates them on exit, so the next invocation
reuses the last known good state:

| Key | Type | Updated by |
|---|---|---|
| `vram` | int (GB) | `--init`; hand-edit |
| `context` | int (tokens) | the picker's per-row context tier |
| `last_model` | str | the picker's model selection |
| `last_agent` | str | the picker's agent selection |
| `last_work_dir` | path | `-C/--workdir` or current value |
| `agent_session_file` | path \| null | computed from `(last_agent, last_model)` for agents that support session history (claude, codex, aider) |

The picker writes its choice to `~/.devai/.last-pick.json` (one-shot,
auto-cleaned) so the launcher knows what the user actually selected.
If the user backs out of the picker, the previous values are kept.
Prerequisites: `make build-{cpu,gpu}` and `make cache-up` once from
the repo — `devai-shell` prints an actionable message if either is
missing.

## Architecture

```
  Browser / API clients
  ─────────────────────────────────────────────────────────
  https://<HOST_IP>:8888        https://<HOST_IP>:8443
         │                              │
  ───────┼──────────────────────────────┼──────────────────
         │         devai-net            │
         │       (internal)             │
         ▼                              ▼
  devai-lab-gpu              devai-webui-proxy
  (JupyterLab + AI CLIs)    (nginx TLS termination)
         │                              │
         │                              ▼
         │                      devai-open-webui
         │                       (chat UI)
         │                              │
         └──────────┬───────────────────┘
                    │
         ┌──────────┴──────────┐
         │  devai-router       │  ports per backend:
         │  (gpu-arbiter)      │    :11434 → ollama
         │                     │    :11435 → vllm
         │  reasoning policy   │    :11436 → sglang
         └──────────┬──────────┘
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
   devai-ollama  devai-vllm  devai-sglang
   (GGUF, RAM-   (NVFP4,     (NVFP4 +
    resident)    on-demand)  RadixAttn,
                              on-demand)

  devai-apt-cache    devai-registry-cache
  (APT cache)        (Docker Hub mirror)
```

**External access** (host ports): JupyterLab `:8888`, Open WebUI `:8443`
**Internal only** (devai-net): router, Ollama, vLLM, SGLang, caches.

### GPU Arbitration

The router (`devai-router`) is a small Go reverse proxy (~9 MB distroless) that:

- Routes by **port → backend** (no message inspection): `:11434` → ollama, `:11435` → vllm, `:11436` → sglang.
- Manages **GPU exclusion**: only one backend uses the GPU at a time. Switches by stopping the current backend (with graceful drain for in-flight requests) and starting the next via the Podman API.
- Sizes vLLM/SGLang launches dynamically (`--gpu-memory-utilization`, `--mem-fraction-static`, `--max-model-len`) from per-model VRAM budgets.
- Auto-stops idle backends after `IDLE_TIMEOUT` seconds.
- Applies the **reasoning policy** to each request (see below).

> **Status:** all three backends are wired. vLLM and SGLang start as `sleep infinity` placeholders (compose can't know which model the user will pick); the router replaces them on demand via libpod when a request arrives on port 11435 / 11436. The picker shows HF rows once they have a fitting probe entry. See [`docs/backends.md`](docs/backends.md) for the lifecycle, probing procedure, cache hygiene, and failure-mode taxonomy.

### Reasoning & MoE

The router doesn't *guess* whether a model can reason. `make probe` runs `scripts/probe-ollama-reasoning.py` against the live Ollama. It probes a 2-D matrix per digest: every VRAM band in `PROBE_VRAMS` (default `16G,24G`) crossed with every context tier in `PROBE_CONTEXTS` (default `32K,64K,128K,256K`). Between bands the orchestrator recreates devai-ollama with `OLLAMA_GPU_OVERHEAD` set to `(host_vram - target_vram) * 1024^3`, so the daemon behaves as if it had only the target VRAM — which lets a 24G host produce cache cells valid for 16G targets. Each cell loads the model with `options.num_ctx` set to that tier and reads `/api/ps` for actual total memory, on-GPU memory, and CPU/RAM spill. No interpolation.

The probe cache (`deploy/.ollama-reasoning-cache.json`, schema v3) is digest-keyed: one record per set of weights, with a `probes` map nested as `probes[<vram_gb>][<ctx>]`. Aliases pointing to the same digest (e.g. `qwen3.5:latest` and `qwen3.5:9b-q8_0`) live under the entry's `aliases` list and share its measurements. The model's true `max_context` ceiling (128K, 256K, 1M, …) is read independently from `/api/show`'s `<arch>.context_length`; tiers above `max_context` are silently capped, so a 128K-only model never wastes a probe on 256K.

Probe runs are incremental: existing cells are immutable. A new (band, tier) only fills a gap; existing cells are left alone. To re-probe one tier specifically, pass `PROBE_FORCE_CTX=64K`. To force a full re-probe of every cell at the current VRAM band, pass `PROBE_FORCE=1`. When a digest disappears from `/api/tags`, its entry is dropped automatically.

**Reasoning capabilities** (top-level `capability` in the probe cache entry):
- `structured` — UI label: Native reasoning. Model returns reasoning in a separate `message.thinking` field. Clean, agent-friendly. Native `think: true|false` controls it.
- `inline` — UI label: Inline reasoning. Reasoning appears as `<think>…</think>` blocks inside `message.content`. Visible but contaminates the answer.
- `unsupported` — UI label: No reasoning. No reasoning behavior observed.
- `error` — UI label: Probe failed, unless VRAM data shows CPU offload. Ollama rejected the probe, the request timed out, or the target context spilled to CPU/RAM.

**Reasoning policy** is set globally via `DEVAI_REASONING=auto|off|low|medium|high` and overridable per-request via the `X-DevAI-Reasoning` header. The arbiter rewrites the request body to set Ollama's native `think:` field (no system-prompt mangling). For Ollama, effort levels (`low|medium|high`) all collapse to `think: true` because the field is boolean.

**Dense vs MoE** is read from `/api/show`'s `model_info` and surfaced in the picker. The numbers in `MoE 8/128` mean **8 experts used per token, out of 128 total** — at every applicable transformer layer, the gating network picks the top-8 experts by score. Concretely:

| field | meaning |
|---|---|
| `expert_count: 128` | the model has a pool of 128 experts (small FFN sub-networks) |
| `expert_used_count: 8` | every token routes through 8 of them at each MoE layer |
| `expert_feed_forward_length: 704` | per-expert hidden dim |
| `block_count: 30` | 30 transformer layers (most/all are MoE) |

So one token through `gemma4:26b` (MoE 8/128) does ~30 × 8 = 240 expert activations, drawing on 6.25% of expert weights *that token*. Over many tokens with varied routing, you sample most of the pool — so all 128 experts must be in VRAM. **The MoE benefit is compute (small active fraction per token), not memory (full pool always loaded)**. Same `fully_on_gpu` rule applies as for dense models: if any weights spill to CPU, performance collapses regardless of MoE/dense.

Three MoE families currently detected from `/api/show` data: `qwen35moe` (256 experts, 8 used), `nemotron_h_moe` (128/6), `gemma4` MoE variants (128/8). Their dense cousins (e.g. `gemma4:31b`, `gemma4:e4b-it-bf16`, `qwen3.5:9b`) carry no `expert_*` fields.

### Selection pipeline

```
scripts/model-families.yaml           hand-edited (add/remove families)
        │
        ▼  python3 scripts/generate-catalog.py        ── make catalog-regen
deploy/models.yaml                    catalog: name, size, arch, purpose
        │
        │   ┌─ for each VRAM band, recreate devai-ollama with
        │   │  OLLAMA_GPU_OVERHEAD set, then probe each context tier
        ▼  python3 scripts/probe-ollama-reasoning.py  ── make probe
deploy/.ollama-reasoning-cache.json   per-digest, per-(VRAM, ctx) cells
        │                             (single source of truth for fit data)
        ├──► router (gpu-arbiter)     reads cache directly, builds
        │                             modelContexts/modelCapability maps
        ├──► picker (model-picker.py) reads cache directly, renders tiers
        └──► diagnostic (model-fit)   prints fitting models at chosen
                                       (VRAM, CONTEXT)
```

Day-to-day after pulling a new model: `make probe && podman rm -f devai-router && make cache-up`.

The interactive model picker reads the probe cache directly. It shows the best Ollama row per family, per context tier (32K, 64K, 128K, 256K), and per user-facing status: Native reasoning, Inline reasoning, No reasoning, CPU offload — all filtered to the picker's VRAM band (env `VRAM` or `GPU_MEMORY_GB`).

**Per-session context binding.** When you select a (model, context) pair in the picker, it derives a session-scoped Ollama tag `<parent>-ctx<N>` (e.g. `qwen3.5:9b-q8_0-ctx32768`) using `/api/create` so `PARAMETER num_ctx N` is baked into the Modelfile. Derived tags share weight blobs with the parent (sub-second creation, no extra disk). This makes the chosen context binding for every wire protocol — Ollama 0.21.x silently ignores `options.num_ctx` on `/v1/chat/completions` and `/v1/messages`, so per-session Modelfile overrides are the only universal mechanism. The router peels the `-ctx<N>` suffix when resolving capability/policy so the parent's reasoning entry still applies. The probe driver also filters `-ctx<N>` derived tags out of `/api/tags` so they're never re-probed.

## Configuration

### `.env` — Host/runtime settings

| Setting | Default | Description |
|---------|---------|-------------|
| `LAB_PORT` | 8888 | JupyterLab port |
| `WEBUI_PORT` | 8443 | Open WebUI HTTPS port |
| `CONTAINER_RUNTIME` | podman | `podman` or `docker` |
| `HOST_HOME_DIR` | `$HOME` | Enables .gitconfig/.ssh in container |
| `HOME_VOLUME` | `$HOME/devai-home` | Persistent home directory |
| `JUPYTER_TOKEN` | — | Fixed access token (set in .env) |
| `APT_PROXY` | — | APT cache URL (e.g. `http://localhost:3142`) |
| `GPU_MEMORY_GB` | 24 | Total GPU VRAM in GB. Picker filter ceiling. Override per-call: `VRAM=48 make shell-gpu` |
| `MAX_CONTEXT_LEN` | 131072 | Operator-side cap on per-model context, in tokens (128K). The router reads `min(model.max_context, MAX_CONTEXT_LEN)` from the probe cache at startup. `make probe` is independent and probes the full `PROBE_CONTEXTS` tier set. |
| `PROBE_VRAMS` | `16G,24G` | Comma-separated VRAM bands `make probe` cycles through. Each band recreates `devai-ollama` with `OLLAMA_GPU_OVERHEAD` set so the daemon behaves like a card of that size. |
| `PROBE_CONTEXTS` | `32K,64K,128K,256K` | Comma-separated context tiers probed inside each VRAM band. Tiers above a model's `max_context` are silently capped. |
| `PROBE_FORCE` | — | When set, `make probe` re-probes every cell in the current VRAM band, ignoring existing cache rows. |
| `PROBE_FORCE_CTX` | — | Single tier (e.g. `64K`) to re-probe; existing cells at other tiers are left alone. |
| `OLLAMA_CONTEXT_LENGTH` | 262144 | Runtime ollama cap (compose env). May cap the probe's `num_ctx` |
| `MIN_VRAM_FRACTION` | 0.5 | Drop models whose total VRAM < this × `GPU_MEMORY_GB` |
| `DEVAI_REASONING` | auto | Default reasoning policy: `auto|off|low|medium|high` |
| `IDLE_TIMEOUT` | 300 | Seconds before auto-stopping vLLM/SGLang (router env) |

### Per-request override

Any agent that talks to the router can set `X-DevAI-Reasoning: off|auto|...` on its HTTP request to override the env-level policy for that one call. Useful for testing without restart.

### `scripts/model-families.yaml` — Family definitions

Hand-maintained source of model lineages. Each family declares its `ollama_repos`, `hf_repos`, and/or `gguf_repos`, plus an `arch_ref` HF repo for architecture metadata. The `thinking: true|false` flag is a hint for humans — final reasoning capability is determined per-variant at probe time and recorded in the probe cache.

Source kinds:

- **`ollama_repos:`** — list of library names under `ollama.com/library/<name>`. Every published tag for each library becomes a catalog row served by the Ollama backend (`ollama pull` on download).
- **`hf_repos:`** — list of HuggingFace repositories (transformers safetensors / FP8 / NVFP4 / AWQ / GPTQ). Each becomes one catalog row served by vLLM/SGLang (currently dormant).
- **`gguf_repos:`** — list of HuggingFace GGUF-only repositories with one entry per file inside the repo. Each entry takes a `repo:`, optional `tag_prefix:` (anchors the local Ollama tag), and optional `include:` allowlist of quantization tokens substring-matched against the filename (e.g. `UD-Q3_K_XL`, `Q3_K_M`). `make model-pull` downloads the `.gguf` blob, writes a Modelfile that emits `FROM <file>` plus `RENDERER <family>` and `PARSER <family>` directives, and runs `ollama create` to register the imported tag. The renderer/parser pair is what makes imported GGUFs accept tool calls — without them Ollama returns "does not support tools".

### `deploy/models.yaml` — Generated catalog

Auto-generated by `make catalog-regen` from upstream HF and Ollama APIs. Has size, architecture, and purpose for every variant. Don't hand-edit.

### `deploy/.ollama-reasoning-cache.json` — Probe cache (single source of truth)

Auto-generated by `make probe`. Schema v3, digest-keyed. Each entry carries `aliases`, `max_context`, top-level `capability`, optional `disable_verified`, and a 2-D `probes` map nested as `probes[<vram_gb>][<ctx>]`. Each probe cell records `actual_total_gb`, `actual_vram_gb`, `fully_on_gpu`, per-cell capability, and timestamp. Both the router (`gpu-arbiter`) and the picker read this file directly. There is no separate active-models.yaml any more — the cache IS the active set.

### Adding Python Packages

Create `requirements.txt` in the repo root and rebuild:

```bash
echo "langchain" > requirements.txt
make build-gpu
```

## Make Targets

Run `make help` for the full list. Highlights:

```
BUILD                                     INFRASTRUCTURE                            RUN
build-cpu        Build image (CPU)        cache-up        Start services            lab-cpu        JupyterLab (CPU)
build-gpu        Build image (GPU)        cache-down      Stop services             lab-gpu        JupyterLab (GPU)
build-router     Build router image       cache-status    Show status               shell-cpu      Shell + picker (CPU)
build            Build all                cache-clean     Remove cached data        shell-gpu      Shell + picker (GPU)
fetch-cli        Update CLI binaries      logs            Tail SERVICE=devai-X
pull-images      Pull base images         setup-logs      One-time: 100G LV at /var/cache/devai/logs

OLLAMA (GGUF — active)                    vLLM (NVFP4 — dormant)                    CATALOG / FIT
ollama-list      List downloaded models   vllm-list       List on-disk weights      catalog-regen     Refresh deploy/models.yaml from upstream
ollama-rm        Remove model             vllm-rm         Remove weights            probe             Probe every (VRAM, ctx) cell
ollama-status    Show status              vllm-status     Show status               model-fit         Print fitting models at VRAM/CONTEXT
ollama-df        Disk usage               vllm-df         Disk usage                model-pull        Download best-fit candidates
ollama-clean     Clean partials                                                     vram-fit          Show what fits without acting

MAINTENANCE                                                                         TESTING
clean            Remove all images        prune           Prune dangling images     test              Run all tests (router + ollama + matrix)
                                                                                    test-router       Go unit tests for arbiter
                                                                                    test-ollama       Ollama integration tests
                                                                                    test-models       Matrix: every probed digest × wire × scenario
                                                                                    test-agents       Smoke-test all agents against ollama
```

## GPU Support

### Requirements

- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit
- For NVFP4 models: Blackwell architecture

### GPU images

The GPU lab image (`devai-lab-gpu`) includes PyTorch with CUDA. The base image uses `nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04`. Python is installed via uv (not system apt) to ensure a single Python version across CPU and GPU images.

## SSL / HTTPS

Both JupyterLab and Open WebUI support HTTPS:

- **JupyterLab**: Auto-detects mkcert certificates in `~/.jupyter/ssl/`
- **Open WebUI**: nginx proxy with mkcert certs or self-signed fallback

Generate certificates on the browser workstation:
```bash
mkcert <HOST_IP>
# Copy <HOST_IP>.pem and <HOST_IP>-key.pem to the container host
```

## Podman Registry Mirror

To route Docker Hub pulls through the local cache:

```bash
cp deploy/registries.conf ~/.config/containers/registries.conf
```

This tells Podman to try `localhost:5000` (the registry mirror started by `make cache-up`) before going to Docker Hub directly.

## Auto-Start at Boot

```bash
make install-systemd
```

Installs a systemd user service that starts all infrastructure containers on login. Uses `loginctl enable-linger` to keep services running after logout.

## Updating

```bash
make fetch-cli                 # Update CLI binaries (Claude, Codex, Ollama, Gemini) via ETags
make pull-images               # Pull latest base and infrastructure images
make catalog-regen             # Refresh deploy/models.yaml from HF + Ollama upstream
make model-pull && make probe  # Pull any new fitting variants, then probe them
make build                     # Rebuild all images with updated binaries/packages
```

## Storage Layout (LVM2)

All persistent data is stored on dedicated LVM2 thin-provisioned logical volumes under `/var/cache/devai/`. This provides:
- **Independent sizing** — each volume can be extended without affecting others
- **Thin provisioning** — space is allocated on demand from a shared pool
- **Clean separation** — models, container images, and caches don't compete for space

Reference implementation (volume group `vgais`):

| Volume | Mount | Size | Purpose |
|--------|-------|------|---------|
| `cache_ollama` | `/var/cache/devai/ollama` | 200G | Ollama GGUF models + vLLM NVFP4 models |
| `cache_registry` | `/var/cache/devai/registry` | 200G | Podman container image storage + Docker Hub mirror |
| `cache_logs` | `/var/cache/devai/logs` | 100G | Persisted container stdout (one `<service>.log` per devai-* container, written by the `logger` sidecar) |
| `cache_pip` | `/var/cache/devai/pip` | 30G | Python package cache (uv) + CLI binaries |
| `cache_apt` | `/var/cache/devai/apt` | 10G | APT package cache (apt-cacher-ng) |
| `cache_npm` | `/var/cache/devai/npm` | 10G | npm package cache |
| `cache_open_webui` | `/var/cache/devai/open-webui` | 5G | Open WebUI application data |

`cache_logs` is the only volume that can be created entirely from the
repo: `make setup-logs` carves the LV in the existing `vgais/cachepool`,
mkfs.xfs, adds an `/etc/fstab` line, and mounts it. Re-running is a
no-op; `RECREATE=1 make setup-logs` rebuilds the volume from scratch.
The remaining volumes are still set up with the manual procedure below.

Create the volumes:

```bash
# Create thin pool (adjust size for your disk)
sudo lvcreate -L 500G -T vgais/cachepool

# Create thin volumes
sudo lvcreate -V 200G -T vgais/cachepool -n cache_ollama
sudo lvcreate -V 200G -T vgais/cachepool -n cache_registry
sudo lvcreate -V 30G  -T vgais/cachepool -n cache_pip
sudo lvcreate -V 10G  -T vgais/cachepool -n cache_apt
sudo lvcreate -V 10G  -T vgais/cachepool -n cache_npm
sudo lvcreate -V 5G   -T vgais/cachepool -n cache_open_webui

# Format and mount
for vol in cache_ollama cache_registry cache_pip cache_apt cache_npm cache_open_webui; do
    sudo mkfs.xfs /dev/vgais/$vol
done

# Add to /etc/fstab
cat <<'EOF' | sudo tee -a /etc/fstab
/dev/vgais/cache_ollama     /var/cache/devai/ollama     xfs defaults 0 0
/dev/vgais/cache_registry   /var/cache/devai/registry   xfs defaults 0 0
/dev/vgais/cache_pip        /var/cache/devai/pip         xfs defaults 0 0
/dev/vgais/cache_apt        /var/cache/devai/apt         xfs defaults 0 0
/dev/vgais/cache_npm        /var/cache/devai/npm         xfs defaults 0 0
/dev/vgais/cache_open_webui /var/cache/devai/open-webui  xfs defaults 0 0
EOF

sudo mkdir -p /var/cache/devai/{ollama,registry,pip,apt,npm,open-webui}
sudo mount -a
sudo chown -R $USER:$USER /var/cache/devai

# Logs volume (handled by Make):
make setup-logs                    # creates cache_logs LV (100G default)
make setup-logs SIZE=200G          # override
```

To extend a volume (e.g. when models fill up):
```bash
sudo lvextend -L 300G /dev/vgais/cache_ollama
sudo xfs_growfs /var/cache/devai/ollama
```

## Key Files

```
.env                              — Host/runtime configuration
.env.example                      — Configuration template
bin/devai-shell                   — Standalone shell-agent launcher (no Make required)
deploy/
  models.yaml                     — Generated catalog (ollama + hf + gguf rows)
  .ollama-reasoning-cache.json    — Probe cache (schema v3, digest-keyed,
                                    probes nested by VRAM × CONTEXT) — single
                                    source of truth for fit data
  docker-compose.yaml             — Infrastructure services
  Dockerfile.base                 — Base image (system packages, Python, Node)
  Dockerfile.lab                  — Lab image (CLI tools, packages, JupyterLab)
  Dockerfile.router               — Router image (distroless, 9 MB)
  webui-proxy/                    — nginx TLS proxy for Open WebUI
  systemd/                        — Auto-start service
gpu-arbiter/
  main.go                         — Router source (multi-port proxy + reasoning policy)
  policy_test.go                  — Unit tests for the reasoning policy
scripts/
  model-families.yaml             — Hand-edited family definitions
  _contexts.py                    — Shared (VRAM, CONTEXT) tier arrays + parsers
  generate-catalog.py             — Refresh deploy/models.yaml from upstream APIs
  probe-ollama-reasoning.py       — Per-(VRAM, ctx) probe per digest (schema v3)
  select-models.py                — Print fitting models / pull catalog candidates
  model-picker.py                 — Two-step interactive picker (model/context/status → agent)
docs/
  ollama_models.md                — Reasoning detection design doc
tests/
  agent-matrix.sh                 — Smoke-test all agents against ollama
  test-router*.sh                 — Router integration tests
requirements-base.txt             — Base Python packages (always installed)
requirements.txt                  — Optional project-specific packages
packages/jupyter-ai-launchers     — JupyterLab launcher extension
```

## License

MIT
