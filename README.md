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

- **Two-step interactive picker** — pick a model (+ backend for HF repos), then an agent. Arrow-key navigation via fzf in the shell (`make shell-gpu`); same flow from JupyterLab launcher cards. Renders per-backend discovery with FORMAT, PARSER, and reasoning-off variants. Inline-reasoning models produce two rows (default + "Reasoning off").
- **Probe-verified model facts** — every downloaded model is probed against the live runtime (Ollama, vLLM, or SGLang). Reasoning behavior (Structured / Inline / Unsupported / Error) plus actual VRAM use at every (VRAM band, context tier) cell, full on-GPU confirmation, and tool-parser verification are measured, not guessed. Cells are probed independently; no interpolation.
- **Per-context bench leaderboard** — bench cache schema v3 keys rows by `(model, backend, ctx)` so the picker shows TPS / CODE% / REAS% / TOTAL% / LEAK% values at the user's chosen ctx exactly. Same model benched at 32K and 128K lands in distinct rows instead of silently overwriting (under MTP a single model can differ by 17+ tok/s between ctx tiers). See `docs/bench-results.md`.
- **Optional cluster mode** — `gpu-arbiter --mode={single,worker,head}` extends the same Go binary to multi-host fleets. Single mode (default) is byte-identical to the pre-cluster code path. Worker mode registers with a head + sends 10s heartbeats + accepts head-forwarded requests. Head mode listens on cluster control plane port 11444 + the OpenAI-compat ports (11434/5/6) and proxies to the highest-scoring registered worker (4-tier policy: exact-match / right-model / idle / different-model). On-prem multi-host or SkyPilot cloud-burst, no Kubernetes. See `docs/cluster-mode.md`.
- **MCP gateway** (opt-in, profile=`mcp`) — the Docker MCP Gateway as a peer service on port 8088. 10 Tier 1 servers with no secrets (filesystem / git / sqlite / fetch / memory / time / sequentialthinking / duckduckgo / arxiv / wikipedia) + 4 Tier 2 servers backed by sops-rendered tmpfs secrets (github / firecrawl / hugging-face / context7). Single endpoint any MCP-aware agent can target. See `docs/mcp.md`.
- **SkyPilot fleet provisioner** (opt-in, profile=`cluster`) — long-lived SkyPilot API server peer to gpu-arbiter, callable from head mode for cloud-burst provisioning across RunPod / Lambda / AWS / GCP / Azure / k8s / Slurm. The lab image also bundles the SkyPilot CLI + Agent Skill so any agent (Claude Code, Codex, Gemini) can spin up cloud GPUs through natural-language. See `docs/skypilot.md` (system-side) and `docs/skypilot-user-guide.md` (user-facing).
- **sops + age secret store** — shared encrypted-at-rest scaffold for every credential the project needs (MCP secrets, cluster bearer tokens, SkyPilot creds). Per-host age key custody, tmpfs-only render targets, idempotent rotation, multi-host onboarding via `sops updatekeys`. See `docs/secrets.md`.
- **MoE / dense awareness** — Ollama probe captures `expert_count` / `expert_used_count` from `/api/show`, surfaced in the picker as `MoE 8/128` or `dense`. Same fit rules apply (full weights must be GPU-resident), but you can see at a glance which models give big-model-quality at small-model speed.
- **Multiple AI CLIs pre-installed** — Claude Code, OpenAI Codex, Google Gemini CLI, Aider, LATE, Open Interpreter, Ollama. All wired through the local router by default.
- **VS Code in the browser** — code-server provides a full Visual Studio Code experience accessible from any browser.
- **Automatic GPU-arbitrated model serving** — Ollama for GGUF models, vLLM/SGLang for NVFP4 models. The gpu-arbiter router transparently switches backends and exposes a single endpoint per protocol (`/api/chat`, `/v1/chat/completions`, `/v1/messages`).
- **Reasoning policy at the router** — set `DEVAI_REASONING=auto|off|low|medium|high` to control thinking mode globally; override per-request via `X-DevAI-Reasoning` header or per-session via `<model>::<reasoning>` suffix (e.g., `qwen3.5:9b::nothink` forces thinking off even for structured-capable models). The router maps your policy to the right native protocol field (Ollama's `think:`, vLLM/SGLang's `enable_thinking`) based on each model's verified capability.
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

Three orthogonal commands. `make probe` (Ollama) + `make probe-vllm` / `make probe-sglang` populate the probe caches; `make model-fit` queries them; `make model-pull` downloads best-fit candidates across a (family, backend, context) matrix.

```bash
# Ollama probing (Make-orchestrated per PROBE_VRAMS bands)
make probe                                    # probe every (VRAM, ctx) cell
make probe PROBE_VRAMS=24G PROBE_CONTEXTS=128K # one band, one tier
make probe PROBE_FORCE=1                      # re-probe everything, ignore cache

# vLLM/SGLang probing (requires `make cache-down` first for exclusive GPU access)
make cache-down
make probe-vllm                               # probe vLLM across all (VRAM, ctx)
make probe-sglang                             # probe SGLang across all (VRAM, ctx)
make cache-up

# Matrix-mode downloading (iterates all context tiers per family/backend)
make model-pull                               # download best-fit (family, backend, ctx) triplets
make model-pull FAMILY=qwen3.5                # scope to one family; still iterates all contexts + backends
make model-pull CONTEXT=32768                 # single context; disables matrix, picks one best per (family, backend)
make model-pull CONTEXTS=32K,128K             # override the context tier list
make model-pull NAME=NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4   # pull one specific catalog row by exact name, bypassing the fit matrix

# Fit queries (probe-cache backed, no side effects)
make model-fit                                # print fitting models at host VRAM × MAX_CONTEXT_LEN
make model-fit VRAM=16 CONTEXT=32768          # query a different (VRAM, ctx)
```

### 5. Run

```bash
make lab-gpu         # Start JupyterLab with GPU (or make lab-cpu)
# OR
make shell-gpu       # Drop straight into the model picker (cwd = repo)
# OR (standalone host launcher — see "devai-agent" below)
make install                  # one-time: stage launcher + config in ~/.local/bin and ~/.devai/
devai-agent --init            # one-time: write default ~/.devai/preferences.yaml
cd ~/myproject && devai-agent # launch with myproject/ mounted as work dir
```

Access:
- **JupyterLab**: `https://<HOST_IP>:8888`
- **Open WebUI**: `https://<HOST_IP>:8443`

### `devai-agent` — standalone host launcher

`bin/devai-agent` is the same lab container as `make shell-gpu`,
runnable from anywhere on the host without invoking Make or being inside the repo directory. State lives
under `~/.devai/`; the repo is only consulted once at `make install` time.

**Install once:**

```bash
make install                       # default INSTALL_PREFIX=~/.local
make install INSTALL_PREFIX=/opt   # alternative location
```

This writes:

| Target | Purpose |
|---|---|
| `~/.local/bin/devai-agent` | The launcher script. |
| `~/.devai/.ollama-reasoning-cache.json` | Symlink to the repo's probe cache so it stays fresh as `make probe` regenerates it. |
| `~/.devai/model-picker.py` | Symlink so the launcher can override the in-image picker via bind-mount (no rebuild needed). |
| `~/.devai/sessions/` | Per-`(agent, model)` session-history dir. |

Then add `~/.local/bin` to `PATH` and run `devai-agent --init` to seed
`~/.devai/preferences.yaml` with defaults.

**Run.** The work directory mounted as `/home/devai/work` is the
shell's `$PWD` at the time of invocation — `cd` into the project you
want to work on first.

```bash
cd ~/projects/my-app && devai-agent # GPU; my-app/ becomes /home/devai/work
devai-agent --cpu                   # CPU lab
devai-agent -C ~/other              # mount ~/other instead of $PWD this run
devai-agent --model qwen3.5:9b-q8_0 --agent claude
devai-agent --show                  # print resolved prefs + container cmd, no run
devai-agent --init                  # reset preferences.yaml to defaults
uninstall via:  make uninstall      # removes the launcher + symlinks
```

**Preferences (`~/.devai/preferences.yaml`).** The launcher reads
these on entry and updates them on exit, so the next invocation
reuses the last known good state for model/agent and surfaces a
record of where you last ran:

| Key | Type | Updated by |
|---|---|---|
| `vram` | int (GB) | `--init`; hand-edit |
| `context` | int (tokens) | the picker's per-row context tier |
| `last_model` | str | the picker's model selection |
| `last_agent` | str | the picker's agent selection |
| `last_work_dir` | path | the actual directory mounted on the last run (informational — the work dir is always `$PWD`, never read back from this field) |
| `agent_session_file` | path \| null | computed from `(last_agent, last_model)` for agents that support session history (claude, codex, aider) |

The picker writes its choice to `~/.devai/.last-pick.json` (one-shot,
auto-cleaned) so the launcher knows what the user actually selected.
If the user backs out of the picker, the previous values are kept.
Prerequisites: `make build-{cpu,gpu}` and `make cache-up` once from
the repo — `devai-agent` prints an actionable message if either is
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

> **Status:** all three backends are wired. vLLM and SGLang start as `sleep infinity` placeholders (compose can't know which model the user will pick); the router replaces them on demand via libpod when a request arrives on port 11435 / 11436. The picker shows HF rows once they have a fitting probe entry.
>
> Two reference docs:
> - [`docs/router.md`](docs/router.md) — router architecture, ports, lifecycle, the full request rewrite chain (override parsing, reasoning policy, tool-choice promotion, tool stripping, ctx injection), config env, caches, and failure modes. Start here when reasoning about routing.
> - [`docs/backends.md`](docs/backends.md) — backend lifecycle, probing procedure, parser plugins, cache hygiene, failure-mode taxonomy.

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

The interactive model picker reads the probe caches directly. It shows one row per `(model_dir, backend)` for HF repos (both vLLM and SGLang rows appear) and per Ollama family/quantization. Each row is filtered to the picker's VRAM band (env `VRAM` or `GPU_MEMORY_GB`). Inline-reasoning models produce two rows: default mode + "Reasoning off" variant with `::nothink` suffix appended.

**Per-session context binding & reasoning overrides:**
- **Ollama**: the picker emits the parent name (or `<name>::nothink` to force reasoning off). KV cache allocation is dynamic per request from the global `OLLAMA_CONTEXT_LENGTH` (default 256K). The `/api/chat` and `/api/generate` endpoints honour `options.num_ctx` injected by the router; OpenAI- and Anthropic-compat endpoints get the global ceiling.
- **vLLM / SGLang**: the picker emits `<name>@<ctx>` (e.g., `Llama-3.1-8B@32768`) or `<name>::<reasoning>@<ctx>` (e.g., `Llama-3.1-8B::low@32768`) to bind both context and reasoning policy per session. The router's `parseCtxOverride` and `parseReasoningOverride` strip these suffixes, rewrite the request body's `model` field to the clean name, and trigger a container recreate if the context or reasoning setting differs from the previous run.

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
clean            Remove all images        prune           Prune dangling images     test              Run all tests (router + python + ollama + matrix)
                                                                                    test-router       Go unit tests for arbiter (single + cluster mode)
                                                                                    test-python       Python stdlib unittests (bench v3, picker, sops/age, MCP, SkyPilot)
                                                                                    test-cluster-preflight  cluster-mode Phase 1.5 (worker + stub head; no GPU)
                                                                                    test-ollama       Ollama integration tests
                                                                                    test-models       Matrix: every probed digest × wire × scenario
                                                                                    test-agents       Smoke-test all agents against ollama
```

### Cluster mode + MCP gateway + SkyPilot (opt-in)

```
CLUSTER (cluster-mode plan, all profile=cluster)        MCP GATEWAY (mcp-gateway plan, profile=mcp)
cluster-head-up        Start router in head mode        mcp-up               Start the MCP gateway
cluster-head-down      Stop the head router             mcp-down             Stop it
cluster-status         GET /v1/cluster/status (head)    mcp-test             Smoke-test gateway
build-worker-bootstrap Build cloud-VM bootstrap image   mcp-secrets-render   Render Tier 2 secrets (Phase 2)

SKYPILOT FLEET PROVISIONER (skypilot plan, profile=cluster)        SECRETS SCAFFOLD (sops-age plan, shared)
skypilot-up            Start API server (port 46580)    age-keygen-host      One-time per host: gen keypair
skypilot-down          Stop it                          secrets-tmpfs        Mount /run/devai tmpfs
skypilot-check         /api/v1/version + sky check      secrets-edit         Edit *.sops.env in place
skypilot-secrets-render Render cloud creds + token      secrets-rotate       Re-key after .sops.yaml change
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
make install-systemd            # stage compose + symlink caches into ~/.config/devai/, enable unit
make uninstall-systemd          # reverse: disable unit + remove staged files
```

`install-systemd` stages a systemd user service that brings the infrastructure containers up on login (`loginctl enable-linger` keeps them running after logout). It **copies** `docker-compose.yaml` into `~/.config/devai/` and **symlinks** the eight other paths the compose mounts (`registry-config.yaml`, `logging.sh`, `recovery-flags.json`, `vllm-plugins.json`, the three `.X-reasoning-cache.json` files, and the `webui-proxy/` directory) back into the repo's `deploy/`. The symlinks keep the systemd-managed stack reading the same probe caches that `make probe` writes — no duplication, no drift.

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
.sops.yaml                        — sops/age recipient list (host public keys)
bin/devai-agent                   — Standalone shell-agent launcher (no Make required)
deploy/
  models.yaml                     — Generated catalog (ollama + hf + gguf rows)
  .ollama-reasoning-cache.json    — Probe cache (schema v3, digest-keyed,
                                    probes nested by VRAM × CONTEXT)
  .bench-cache.json               — Bench cache (schema v3, per (model, backend, ctx))
  docker-compose.yaml             — Infrastructure services (router/ollama/vllm/sglang/
                                    webui + opt-in mcp-gateway + skypilot-api-server)
  compose.head.yaml               — Cluster-head overlay (zeroes local backends,
                                    sets DEVAI_MODE=head on router)
  mcp-servers.yaml                — MCP gateway catalog (10 Tier 1 + 4 Tier 2 servers)
  mcp-secrets.sops.env.example    — Operator template for Tier 2 secrets
  skypilot-credentials.sops.env.example — Operator template for cloud creds + tokens
  Dockerfile.base                 — Base image (system packages, Python, Node)
  Dockerfile.lab                  — Lab image (CLI tools incl. SkyPilot, JupyterLab)
  Dockerfile.router               — Router image (distroless, 9 MB)
  Dockerfile.worker-bootstrap     — Cloud-VM bootstrap image for SkyPilot-launched workers
  worker-cloud-init.sh            — Cloud-init entrypoint baked into bootstrap image
  setup-secrets-tmpfs.sh          — Idempotent /run/devai tmpfs mount
  webui-proxy/                    — nginx TLS proxy for Open WebUI
  systemd/                        — Auto-start service
gpu-arbiter/
  main.go                         — Router source (multi-port proxy, reasoning, --mode dispatch)
  policy_test.go                  — Unit tests for the reasoning policy
  cluster_proto.go                — Cluster wire-protocol types (Register/Heartbeat/Command)
  cluster_auth.go                 — Bearer-token TokenStore + AuthMiddleware
  parse_minimal.go                — Head-side request parser (model + @ctx + ::reasoning)
  cluster_worker.go               — Worker-mode loop (register, heartbeat, dispatchCommand)
  cluster_main.go                 — runWorkerMode + runHeadMode entrypoints
  fleet_state.go                  — Head's in-memory worker map + heartbeat-TTL expiry
  routing_policy.go               — 4-tier scoring + round-robin tiebreak
  cluster_head.go                 — Head's control plane + frontend proxy handlers
  cluster_proxy.go                — Stream-preserving HTTP proxy to chosen worker
  skypilot_client.go              — HTTP client for SkyPilot /api/v1/{launch,status,down}
  skypilot_policy.go              — Cheapest-cloud picker + IdleTeardownCoordinator
scripts/
  model-families.yaml             — Hand-edited family definitions
  _contexts.py                    — Shared (VRAM, CONTEXT) tier arrays + parsers
  generate-catalog.py             — Refresh deploy/models.yaml from upstream APIs
  probe-ollama-reasoning.py       — Per-(VRAM, ctx) probe per digest (schema v3)
  select-models.py                — Print fitting models / pull catalog candidates
  model-picker.py                 — Two-step interactive picker (model → agent)
  age-keygen-host.sh              — Per-host age keypair generator (idempotent)
  render-secret.sh                — Generic single-file decrypt to tmpfs (refuses non-tmpfs)
  mcp-health.sh                   — MCP gateway health probe
  skypilot-api-health.sh          — SkyPilot API server health probe
  sky-setup.sh                    — First-launch helper inside the lab
  bench/_bench_core.py            — Bench harness shared helpers (schema v3 keys + migrator)
  bench/bench_runner.py           — Bench driver with --ctx / --all-ctx flags
  bench/bench_report.py           — Markdown leaderboard with CTX column
docs/
  ollama_models.md                — Reasoning detection design doc
  secrets.md                      — sops/age scaffold reference (canonical)
  mcp.md                          — MCP gateway operator reference
  cluster-mode.md                 — Cluster-mode operator reference
  cluster-env.md                  — Per-env-var contract for cluster mode
  worker-bootstrap.md             — Cloud-VM bootstrap image reference
  cluster-mode-preflight.md       — Phase 1.5 preflight test report
  skypilot.md                     — System-side fleet provisioner reference
  skypilot-user-guide.md          — User-facing SkyPilot CLI guide
  plans/                          — 6 design plans + execution-order README
tests/
  agent-matrix.sh                 — Smoke-test all agents against ollama
  test-router*.sh                 — Router integration tests
  test-cluster-preflight.sh       — cluster-mode Phase 1.5 preflight (CI-runnable, no GPU)
  test-mcp.sh                     — MCP gateway end-to-end smoke
  test-fleet-routing.sh           — Fleet-routing skeleton (skips when no SkyPilot endpoint)
  fixtures/stub-head.py           — Stub head for cluster-mode preflight
  python/                         — 138 Python stdlib unittests covering bench v3,
                                    sops/age scaffold, MCP gateway P1+P2, SkyPilot
                                    fleet P1, agent-skill, stub head
requirements-base.txt             — Base Python packages (always installed)
requirements.txt                  — Optional project-specific packages
packages/jupyter-ai-launchers     — JupyterLab launcher extension
```

## License

MIT
