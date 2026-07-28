# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **Dev AI Lab** -- a containerized development environment for AI experimentation featuring JupyterLab and multiple AI CLIs (Gemini, Claude, OpenAI, Ollama). Built on Debian Trixie with Python 3.13 (uv-managed), Node.js 22 LTS. Two-layer image build for fast iteration. Compatible with Podman and Docker. GPU/CUDA support (NVIDIA default; AMD/ROCm opt-in via `DEVAI_GPU_VENDOR`, build-time-verified only so far -- see docs/gpu-vendors.md).

**Backends:** all three are wired -- Ollama (GGUF, port 11434), vLLM (NVFP4/safetensors, port 11435), SGLang (NVFP4/safetensors, port 11436). The router enforces GPU mutual exclusion: only one backend serves at a time. vLLM and SGLang start as `sleep infinity` placeholders and are recreated on demand by the router when a request arrives.

**Documentation:**
- `docs/router.md` -- router architecture, ports, lifecycle, request rewrite chain (override parsing -> Anthropic `/v1/messages` normalisation -> reasoning policy -> tool_choice promotion -> tool stripping -> ctx injection), config, caches, failure modes. **Source of truth for the router.**
- `docs/backends.md` -- backend lifecycle, probing procedure, parser plugins, cache hygiene, failure-mode taxonomy.
- `docs/nvfp4-coldstart.md` -- NVFP4 cold-start phases (graphviz timeline) and per-component VRAM budget at 32K/64K/128K/256K. Reference model: **`nvidia/Qwen3-8B-NVFP4`**, anchored to `deploy/.bench-cache.json` measurements (peak/mean VRAM, cold + steady TTFT, sustained tok/s, GSM8K/HumanEval/tools/leak scores) on RTX PRO 4000 Blackwell (24 GB GDDR7). Per-phase wall times are not instrumented -- the diagram shows phase order and bottleneck only. Anything beyond 24 GB or beyond 8B-class is paper extrapolation, not benchmark. Explains why `HEALTH_TIMEOUT_SECONDS=600`, why context-cap changes recreate the container, and why `peak_vram_gb` is not the strict minimum.
- `docs/nvfp4-number-formats.md` -- beginner's guide to FP32, FP16, BF16, FP8 (E4M3/E5M2), FP4, MXFP4, NVFP4. Covers how IEEE-754 numbers get encoded in fewer bits with worked examples, the role of per-block scales, why model weights and KV cache typically get *different* formats (NVFP4 weights + FP8 KV is the modern default), and how to read `quantization_config` in a checkpoint's `config.json`. Read this first if you're new to LLM number formats.
- `docs/llm-tokens-and-speed.md` -- beginner's guide to LLM tokens (BPE, why they're nothing like compiler tokens, why `"hello"` and `" hello"` are different) and to the *prefill vs decode* split that drives every inference benchmark. Derives the bandwidth-bound decode ceiling for Qwen3-8B-NVFP4 on RTX 4000 PRO Blackwell (640 GB/s / 5.1 GB ~ 125 tok/s peak; measured 98.3 tok/s ~ 78 % utilisation) and proves it against the BF16 R1-Distill row (16 GB -> 40 tok/s ceiling; measured 42 tok/s = at ceiling). Sourced from `bench-results.md`.
- `docs/attention-and-the-transformer.md` -- beginner's guide to the transformer architecture itself: embedding -> 36 transformer blocks -> lm_head. Walks through Q/K/V dot-product attention with worked example, GQA (why Qwen3-8B has 8 KV heads instead of 32 -- and why this is *the* reason 128 K context fits on 24 GB), RoPE positional encoding (and how YaRN extends 40 K-trained contexts to 131 K), SwiGLU MLP, RMSNorm, and the full Qwen3-8B parameter inventory mapping each weight bucket to its NVFP4/BF16 storage class. The conceptual foundation that every other doc in this folder implicitly assumes.
- `docs/reasoning-tool-calling-chat-templates.md` -- beginner's guide to the three "agentic" mechanisms above the model architecture: chat templates and the `<|im_start|>` machinery; tool calling with parser plugins (`hermes`, `qwen3_xml`, `llama3_json`, `deepseek_string`, `openai`/harmony) and `tool_choice` modes; reasoning models with `<think>` blocks, `--reasoning-parser`, and the `structured/inline/unsupported` capability classification. Includes the TPS-counting bug from `bench-results.md` as a worked example of why understanding `reasoning_content` separation matters.
- `docs/openai-api-and-streaming.md` -- the wire format itself. Endpoint surface, request/response anatomy, every generation-control field, full SSE streaming including tool-call piecemeal arguments and reasoning-content streaming, the Anthropic `/v1/messages` variant, worked curl examples against the project's router. Practical companion to anyone reading vLLM/SGLang/router code or writing a custom agent.
- `docs/sampling-strategies.md` -- beginner's guide to how the next token is actually picked from the model's logit vector. Covers greedy, temperature, top-k, top-p (nucleus), min-p, repetition/frequency/presence penalties -- each with a worked example on a tiny 5-token vocabulary distribution. Ends with sane defaults per task type (deterministic eval, code, chat, creative, reasoning, tool calling). Self-contained ~400-line read.
- `docs/paged-attention-and-vllm-internals.md` -- beginner's guide to the trick that makes the "elastic KV pool" referenced in `nvfp4-coldstart.md` actually work. Explains internal/external KV-cache fragmentation in the naive approach, the OS-style page-table indirection vLLM uses (block_size 16, fixed pool, per-sequence block tables), the five wins (zero waste, prefix caching, continuous batching, dynamic admission, copy-on-write), the kernel-level cost, and SGLang's RadixAttention extension. Computes the actual block count for Qwen3-8B-NVFP4 on the project's hardware (~11 950 blocks ~ 191 K tokens of pool capacity).
- `docs/mixture-of-experts.md` -- beginner's guide to MoE models, anchored in `gpt-oss-20b` (the project's coding specialist). Explains total vs active parameters (~21 B total / ~3.6 B active for gpt-oss-20b), why VRAM scales with total but FLOPs scale with active, the router/specialisation mechanism, why MoE is harder to serve (expert load imbalance, batching defeated, slower cold start), and how to spot an MoE model in `config.json`. Notes the open follow-up in `bench-results.md` to re-run gpt-oss-20b TPS post-fix.
- `docs/secrets.md` -- canonical reference for the shared sops + age secret-store scaffold. Its only remaining consumer is the MCP gateway's two secret-needing servers (`github-official`, `firecrawl`), and that path is unverified; the two other consumers it was built for (gpu-arbiter cluster mode, the SkyPilot fleet provisioner) were frozen on 2026-07-25 -- see `attic/README.md`. `docs/plans/README.md` accordingly carries `sops-age-secrets` as Non-functional. Setup walkthrough (install binaries, `make age-keygen-host`, add the public key to `.sops.yaml`, `make secrets-tmpfs`), the edit / render / rotate cycle, recovery posture (lost age key = lost secrets; offline backup pattern), multi-host onboarding, and the operator pre-commit checklist that catches a stale `age1xxx...` placeholder. **Source of truth for the secret scaffold.**
- `docs/mcp.md` -- operator reference for the Docker MCP Gateway (peer service to `devai-router`). The gateway loads Docker's official MCP catalog (upstream digest-pinned and maintained) and merges this repo's first-party catalog `deploy/mcp-catalog-devai.yaml` on top with `--additional-catalog`; that first-party file declares exactly one server, `devai-model-status`. 15 servers are enabled via the compose `--servers=` list. Covers bring-up, client configs (Claude Code / Gemini / Codex / Open WebUI), the security model, troubleshooting. See also the MCP gateway subsection under "Inference stack" below for what is verified and what is not.
- `attic/README.md` -- index of frozen work. **Cluster mode (`--mode=worker|head`) and the SkyPilot fleet provisioner were frozen on 2026-07-25**: 22 Go files, the compose overlay, the worker-bootstrap image, their tests, and their five docs (`cluster-mode.md`, `cluster-env.md`, `cluster-mode-preflight.md`, `worker-bootstrap.md`, `skypilot.md`) moved to `attic/cluster-mode/` behind a `//go:build devai_frozen_cluster` tag outside every Go module. Nothing was deleted and the feature is intended to return; `attic/cluster-mode/RESTORE.md` lists the defects that were open at freeze time. Single-host is now the only supported mode -- `gpu-arbiter` still accepts `--mode`, but any value other than `single` exits with a pointer to that README.
- `docs/skypilot-user-guide.md` -- user-facing SkyPilot CLI guide. The lab image bundles the `sky` CLI (and, since 2026-07-25, `/usr/local/bin/sky-setup.sh`), which any CLI agent can drive as an ordinary command-line tool. **There is no Agent Skill plugin** -- this file used to claim one was pre-installed and nothing ever installed it (skypilot-agent-skill Phase 2 was never done). Covers per-cloud credential setup, hello-world, agent-driven flow, cost guidance with $/hr table. The system-side fleet provisioner it used to pair with is FROZEN -- see `attic/README.md`.
- **Supported picker agents (5, after the 2026-07-28 cleanup):** `claude`, `aider`, `codex`, `opencode`, `aiagent`. **LATE and Open Interpreter were removed entirely** -- LATE has no model UI of any kind (its config schema holds a single scalar `openai_model`; no `/model` command, no in-session switching, and it never even displays the active model), and Open Interpreter was dead in the lab image (`ModuleNotFoundError: No module named 'pkg_resources'` at import time on modern setuptools) as well as abandoned upstream -- last Python release 2024-10, and the GitHub name now redirects to an unrelated Rust rewrite. Removal covered the picker list and `_build()` branches, the `late` fetch block in the Makefile, the `late-bin` install + `scripts/late-launcher.sh`, `open-interpreter` in `requirements-base.txt`, the `libxcb1` system dep that existed only for OI's cv2 import, and both agents' rows in `tests/agent-matrix.sh`.
- `docs/aiagent.md` -- reference for the "AIAgent (shell)" picker agent. aiagent (github.com/devitops-com/aiagent) is a DSPy CLI the user drives herself, so the picker drops to an interactive bash shell pre-configured with the router endpoint (`AIAGENT_API_BASE=<router>/v1`) + model instead of exec'ing it. Covers the env contract, the `DEVAI_AIAGENT_GPU` router-only|share toggle (default router-only hides the GPU so aiagent can't contend with the router-loaded model), build-time install of the makeself bundle to an isolated `/opt/aiagent`, the verified qwen3.6:27b-q4_K_M example, and the five upstream aiagent bugs found + fixed during integration (devitops-com/aiagent#1-5, all fixed in v0.1.2, which the lab image bakes).
- `docs/backup-restore.md` -- reference for `devai-backup` (`devai-tools/cmd/devai-backup`): what gets snapshotted (probe/bench caches, `~/.devai/` preferences+sessions, the sops/age private key) and what's deliberately excluded, the age-key "no backdoor" warning, command reference, restore semantics (path-traversal validation before any writes, rename-aside rather than delete), a recovery walkthrough.
- `docs/security-ci.md` -- reference for the two GitHub Actions workflows (`security-blocking.yml`, `security-advisory.yml`) plus `dependabot.yml`: which checks block a merge vs which are advisory-only, branch-protection setup, the containerized local pre-push command sequence, how to read advisory findings (Security tab for CodeQL, job logs for govulncheck/Trivy).
- `docs/mcp-model-status.md` -- reference for `devai-model-status`, the only devai-authored MCP server (`devai-tools/cmd/devai-mcp-modelstatus`, declared in `deploy/mcp-catalog-devai.yaml`). Three read-only tools: `list_fitting_models`, `get_model_bench`, `get_router_status` (single mode, or unreachable). Config files are baked into the image at build time, not bind-mounted -- rebuild after `make probe`/`make bench` to refresh. The operator's standing decision is that this repo authors no further MCP servers: third-party needs are met from Docker's official catalog. `devai-model-status` is grandfathered because no third-party server can know about this lab's probe/bench caches.
- `docs/gpu-vendors.md` -- reference for the NVIDIA/AMD GPU-vendor overlay (`DEVAI_GPU_VENDOR`, flipped via `make gpu-vendor VENDOR=nvidia|amd` or `devai-agent --gpu-vendor`). Every hardcoded `nvidia.com/gpu=all` call site and why 2 of the 9 are deliberately left NVIDIA-only (that gap -- the router container never being handed `DEVAI_GPU_DEVICE` in compose, so every router-recreated backend got the NVIDIA default regardless of the overlay -- was **fixed 2026-07-27**; since vLLM/SGLang start as `sleep infinity` placeholders and are ALWAYS recreated by the router, it had meant the `amd` overlay never reached the two services it matters most for), the `devai-lab-gpu` image's ROCm base + PyTorch wheel-index branch, and an honest verification-status note: NVIDIA is fully verified, AMD/ROCm is unverified even at build time (Docker Hub was unreachable in the session this shipped from) pending a real ROCm host.
- `docs/plans/` -- 14 design plans + execution-order README (`docs/plans/README.md` carries the canonical status table; consult it rather than counting files -- statuses moved substantially on 2026-07-25). Highlights: `mcp-gateway` (Done). `bench-rewrite` (In Progress -- Phases 1-5 shipped, Phase 6 deferred to live GPU). `sops-age-secrets` (Non-functional: the scaffold ships but two of its three intended consumers are frozen). `gpu-arbiter-cluster-mode`, `skypilot-fleet-provisioner` and `skypilot-agent-skill` are all **Frozen** -- the first two have their code in `attic/cluster-mode/` (see `attic/README.md`); the plans stay in `docs/plans/` as the design record. Also tracked there: `model-lifecycle-ledger`, `kv-cache-quantization`, `router-shortcircuit`, `router-fanout`, `pi-coding-agent`, `odysseus-borrowed-ideas`, `review-fixes-2026-07`, `card-derived-hints-and-bench-sync`.

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
make shell-gpu       # Interactive shell (GPU) -- cwd = repo root
make shell-cpu       # Interactive shell (CPU) -- cwd = repo root
# All interactive containers run as the devai user (uid 1000), mapped to the
# host user -- NOT container-root. Rootless podman uses --userns=keep-id --user
# devai (devai's uid 1000 maps back to your host uid, so bind-mounted home/work
# stay writable); docker drops to devai via the entrypoint gosu remap. Need root
# inside a running container? `podman exec -u 0 <container> ...`.

# Install standalone launcher (one-time; stages launcher + config under ~/.devai/)
make install         # writes ~/.local/bin/devai-agent + ~/.devai/{model-picker.py,probe-cache symlink}
make uninstall       # removes the launcher + symlinks

# Standalone launcher (state in ~/.devai/, independent of repo cwd)
devai-agent --init           # reset ~/.devai/preferences.yaml to defaults
devai-agent                  # GPU lab; reads/writes preferences.yaml
devai-agent --cpu
devai-agent -C ~/projects/foo --model qwen3.5:9b-q8_0 --agent claude
devai-agent --show           # print resolved prefs + podman command, no run

# Infrastructure
make cache-up        # Start all services. vLLM/SGLang start as `sleep` placeholders; router recreates on demand.
make cache-down      # Stop all services
make cache-status    # Show status, models, disk usage

# Models -- matrix-driven selection and probing
make probe                      # Probe every (VRAM, ctx, backend) cell; Ollama only. PROBE_READY_TIMEOUT=180 bounds the per-band wait for the recreated devai-ollama.
make probe-vllm                 # Probe every (VRAM, ctx) cell for vLLM; requires `make cache-down`
make probe-sglang               # Probe every (VRAM, ctx) cell for SGLang; requires `make cache-down`
make probe-load-vllm            # Serving-time LOAD probe: augment vLLM fit cache with serving_ok/transient/needle; ascending ctx, stop at OOM. Run after probe-vllm.
make probe-load-sglang          # Same LOAD probe against the SGLang cache. Run after probe-sglang.
make probe-check                # Report backend image drift: compare running vLLM/SGLang image digest vs each probe cache's _meta baseline. Exit 1 on drift.
make model-fit                  # Print which models fit at chosen VRAM/CONTEXT (no writes)
make model-pull                 # Download best-fit (family, backend, context) candidates (matrix mode)
make model-pull FAMILY=qwen3.5  # Scope to one family; still iterates 4-context matrix per backend
make model-pull CONTEXT=32768   # Single context; disables matrix mode, picks one best per (family, backend)
make model-pull CONTEXTS=32K,128K  # Override the context tiers from the 4-context default
make model-pull NAME=NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4  # Pull one catalog row by exact name, bypassing the (VRAM, ctx) fit matrix. NOTE: every model-pull form lands HF weights in the vLLM store; SGLang needs `python3 scripts/select-models.py --name <n> --download --hf-store sglang`.
make catalog-discover           # Read-only: find newer-version members of tracked lineages (e.g. qwen3.7 when qwen3.5 is tracked) on HF + Ollama, FILTERED to the GPU's usable VRAM band (estimated weight VRAM = params x quant bytes; hides too-big-to-load 397B, too-small-to-bother <50%-VRAM, AND base/non-chat checkpoints). Unmarked '?' formats fetch real on-disk size. FAMILY=<substr>, INCLUDE_SAME=1, INCLUDE_OVERSIZED=1, INCLUDE_UNDERSIZED=1, INCLUDE_BASE=1, VRAM_TOLERANCE=<x>, MIN_VRAM_FRAC=<f>, NO_HF=1, NO_OLLAMA=1. VRAM budget from GPU_MEMORY_GB. Probe before adding.
make catalog-discover-add       # Confirm-add a discovered candidate INTO scripts/model-families.yaml (existing families only; the only writer of that file, comment-preserving). ADD=<repo> adds one repo non-interactively; bare target walks candidates and confirms each. YES=1 skips the prompt. Then catalog-regen + probe.
make model-status               # Show the host-local exclusion ledger (deploy/.model-status.json): models marked too_big/too_small/unsupported_arch/oom/manual so they aren't re-downloaded/probed. CLEAR=<name[::backend]> removes an entry.
make model-sync                 # Closed loop: diff catalog vs probed+excluded, auto download+probe genuinely-new rows, record outcomes. DRY_RUN=1 (plan only), SYNC_MAX_DOWNLOADS=<n>, FAMILY=<f>, REGEN=1 (catalog-regen first).
make ollama-list                # List downloaded Ollama models
make vllm-list                  # List on-disk vLLM/SGLang weights

# Logging (logger sidecar persists each container's stdout)
make logs SERVICE=devai-ollama  # Tail one container's persisted log
make logs SERVICE=devai-router LINES=200
make setup-logs                 # One-time: 100G LV at /var/cache/devai/logs (sudo)

# Tests
make test-router                # Go unit tests for arbiter (single-host only; 240 tests as of 2026-07-27)
make test-devai-tools            # Go unit tests for devai-tools/ (backup, modelcache, routerclient, envfile, gpu-vendor -- 6 packages)
make test-python                # Python stdlib unittests (bench v3, picker, sops/age, MCP, catalog/model-lifecycle, hf-store linking, bench-sync -- 612 collected on 2026-07-27)
make test-backup-restore        # devai-backup Go tests + tests/test-backup-restore.sh
make test-gpu-vendor            # GPU-vendor overlay: flips DEVAI_GPU_VENDOR, asserts rendered compose config both directions
make test-ollama                # Ollama integration tests
make test-models                # Matrix: every probed digest x wire x scenario
make test-vllm                  # Live vLLM integration (chat, ctx switch, GPU exclusion)
make test-sglang                # Live SGLang integration (skips when not loadable)
make test-e2e                   # Picker -> agent command -> live router chat
make test-probe-vllm            # Probe smoke: cache schema assertion (requires cache-down)
make test-probe-sglang          # Same for SGLang
make test-probe-ollama-idempotent  # Byte-identical regression check on refactored Ollama prober
make test                       # All of the above in sequence (~30-60 min wall time)

# Backup / restore (per docs/backup-restore.md). Snapshots probe/bench caches,
# ~/.devai/ preferences+sessions, and the sops/age key.
make backup-create [DEST=...]   # Snapshot host-local state to ~/.devai/backups/<timestamp>.tar.gz
make backup-list [DEST=...]     # List existing archives (JSON: path, size, mtime, top-level dirs)
make backup-verify ARCHIVE=...  # Validate an archive without extracting
make backup-restore ARCHIVE=... YES=1  # Restore; destructive, requires YES=1

# GPU vendor (per docs/gpu-vendors.md). NVIDIA is the default; AMD/ROCm is
# build-time-only so far (no ROCm hardware verified against yet).
make gpu-vendor VENDOR=nvidia|amd  # Flip DEVAI_GPU_VENDOR + its 3 derived .env vars

# MCP gateway (per docs/mcp.md). 15 servers enabled; only devai-model-status is ours.
make mcp-up                     # Start the gateway via the 'mcp' compose profile (port 8088, path /mcp)
make mcp-down                   # Stop the gateway
make mcp-logs                   # Tail gateway log (the startup bearer token is printed here, once)
make mcp-test                   # End-to-end: real MCP handshake, tool-count floor, first-party tool names, real tools/call
make mcp-health                 # Liveness only -- it cannot tell a gateway serving 134 tools from one serving zero. Use `make mcp-test`.
make mcp-secrets-render         # Render deploy/mcp-secrets.sops.env -> /run/devai/mcp-secrets.env (needed only by github-official + firecrawl)
make build-mcp-modelstatus-image  # Build the devai-model-status MCP server image (per docs/mcp-model-status.md)
make test-mcp-modelstatus       # End-to-end test of devai-model-status against the live gateway

# Cluster mode and the SkyPilot fleet provisioner are FROZEN (2026-07-25).
# Their 9 Makefile targets (cluster-head-up/down, cluster-status,
# skypilot-up/down/check/secrets-render, build-worker-bootstrap,
# test-cluster-preflight) are preserved in
# attic/cluster-mode/Makefile.frozen-targets. See attic/README.md.
# The `sky` CLI inside the lab image is a separate thing and stays.

# Secrets scaffold (per docs/secrets.md). Only remaining consumer: the MCP gateway's
# github-official + firecrawl secrets, and that path is unverified.
make age-keygen-host            # One-time per host: generate ~/.config/sops/age/keys.txt
make secrets-tmpfs              # Mount /run/devai as 4MiB tmpfs (one-time per boot)
make secrets-edit SOPS_FILE=... # Edit a sops-encrypted file in place
make secrets-render SOPS_FILE=... DEST=...  # Generic single-file render
make secrets-rotate             # Re-key every deploy/*.sops.env after .sops.yaml changes

# Bench harness (schema v3 -- per-context rows; see docs/bench-results.md)
make bench                      # All backends. Largest fitting ctx per model by default. Default BENCH_TASKS is now gsm8k,humaneval,humaneval_plus,mmlu_pro,gpqa,tools,leak -- materially longer than the old 4-task sweep; BENCH_TASKS=gsm8k,humaneval,tools,leak restores it.
make bench-vllm                 # Just vLLM
make bench-report               # Markdown leaderboard (CTX column shows the per-row context)
make bench-plan                 # Read-only: classify every bench target as new/incomplete/stale_env/stale_image/dropped/excluded/current. Starts no container, writes nothing.
make bench-sync                 # Closed loop: run that plan (new/incomplete/stale rows, grouped by backend so a mixed queue doesn't thrash the GPU) then re-render the leaderboard. LONG + GPU-exclusive, and says so on start.
                                #   DRY_RUN=1 (plan only), BENCH_MAX_TARGETS=<n> (a capped run reports what it left undone rather than looking like full coverage), BACKEND=, RECORD_DROPS=1 (write bench_dropped verdicts; off by default -- a drop is an operator decision).

# Maintenance
make clean           # Remove all images (CPU + GPU + router)
make prune           # Prune dangling images
make help            # Show all targets
```

## Configuration

Copy `.env.example` to `.env` before first use. Key settings:

- `LAB_PORT` -- JupyterLab port (default: 8888)
- `WEBUI_PORT` -- Open WebUI HTTPS port (default: 8443)
- `CONTAINER_RUNTIME` -- `podman` (default) or `docker`
- `HOST_HOME_DIR` -- Host home directory for .gitconfig/.ssh mounting
- `HOME_VOLUME` -- Persistent home directory path
- `JUPYTER_TOKEN` -- Fixed JupyterLab access token
- `HTTP_PROXY`/`HTTPS_PROXY` -- Proxy settings for corporate environments

Model catalog is in `deploy/models.yaml` (single source of truth for all models). GPU inference settings:

- `GPU_MEMORY_GB` -- Total GPU VRAM in GB (default: 24). Used by router to calculate memory fractions and context limits.
- `MAX_CONTEXT_LEN` -- Default max context length in tokens (default: 262144 = 256K). The router caps each model's per-name context at `min(model.max_context, MAX_CONTEXT_LEN)`. The probe cache (`deploy/.ollama-reasoning-cache.json`) is the source of truth -- `deploy/active-models.yaml` no longer exists.
- `DEVAI_GPU_VENDOR` -- `nvidia` (default) or `amd`; set via `make gpu-vendor VENDOR=nvidia|amd`, not by hand (see docs/gpu-vendors.md). Derives `DEVAI_GPU_DEVICE`, `VLLM_IMAGE`, `SGLANG_IMAGE`.
- `DEVAI_GPU_DEVICE` -- CDI device string for the backend containers and `make lab-gpu`/`shell-gpu` (default: `nvidia.com/gpu=all`). AMD/ROCm is build-time-verified only so far, not yet run against real ROCm hardware.
- `DEVAI_MODE` -- accepted but effectively fixed at `single`. Any other value exits with a pointer to `attic/README.md`; the worker/head env-var contract went to `attic/cluster-mode/` with the rest of the frozen feature.

### MCP gateway env vars (opt-in; see docs/mcp.md)

- `MCP_PORT` -- host port the gateway listens on (default 8088). Published on **127.0.0.1 only**; remote clients need an SSH tunnel or an authenticating reverse proxy. The MCP endpoint is `http://127.0.0.1:${MCP_PORT}/mcp` -- the `/mcp` path is required.
- `MCP_SECRETS_FILE` -- host path of the rendered tmpfs secrets file (defaults to `/dev/null` so installs without secrets boot cleanly). This is the only secrets knob an operator sets; the container-side path is fixed at `/secrets/.env` in the compose `--secrets=` arg. Only `github-official` and `firecrawl` need it.

## Architecture

### Lab container (devai-lab-cpu / devai-lab-gpu)

Two-layer image build (base rarely changes, lab layer for fast iteration):

**Layer 1: Dockerfile.base** -- System packages, Python 3.13 (via uv), Node.js 22 LTS
**Layer 2: Dockerfile.lab** -- CLI binaries (Claude, Codex, Ollama, Gemini, code-server), PyTorch, Python packages, JupyterLab

Build cache: CLI binaries pre-downloaded to `/var/cache/devai/pip/bin/` via `make fetch-cli` (ETag-based updates). Mounted into build -- no network downloads during rebuild.

### Inference stack (deploy/docker-compose.yaml)

Single-host is the only supported topology (multi-host cluster mode is frozen -- see `attic/README.md`):

```
Agent -> devai-router:11434 -> devai-ollama:11434 (GGUF models)
Agent -> devai-router:11435 -> devai-vllm:11434   (NVFP4 / safetensors via vLLM)
Agent -> devai-router:11436 -> devai-sglang:11434 (NVFP4 / safetensors via SGLang)
```

- **devai-router** -- Multi-port GPU-aware reverse proxy. One port per backend. No message inspection -- port determines backend. Manages GPU exclusion (only one backend uses GPU at a time), graceful drain on switch, keep-warm-by-default lifecycle (`IDLE_TIMEOUT` env, default 0 = never auto-unload; a loaded model stays resident until a *different* model is requested), fail-fast launch health (`HEALTH_TIMEOUT_SECONDS` env, default 600s; a crashed engine is detected via container-exit / terminal log signatures in `waitForHealthy`->`detectLaunchFailure` and fails immediately instead of waiting the full timeout; signature matching is case-insensitive, and the set covers SGLang's `Scheduler hit an exception` / `device-side assert triggered` / `unrecognized arguments` as well as vLLM's), **launch circuit breaker** (`DEVAI_MAX_FAILED_LAUNCHES`, default 3, `0` disables): an engine that launches, answers `/health`, and then dies serving is invisible to `detectLaunchFailure`, and one such SGLang model was recreated 72 times in a day completing zero work. Each launch of a `(model, ctx)` spends one unit; the budget is repaid ONLY when the engine returns a real non-5xx response, via the proxy's `ModifyResponse` -- so credit lands on the response headers (a long generation still counts) and an engine that never answers gets none (the default error handler synthesises 502 without reaching that hook). `/health` deliberately does NOT repay: SGLang's HTTP server outlives its scheduler. Counters live under a dedicated `breakerMu`, NOT `arbiter.mu`, because `stopOtherBackends`->`drainBackend` holds `arbiter.mu` while waiting on exactly the in-flight requests that repay -- lock order is `arbiter.mu` -> `breakerMu`. Every backend is credited, Ollama included. See docs/router.md "Launch circuit breaker". Per-backend concurrency cap (`MAX_CONCURRENT_REQUESTS` env, default 32 -> HTTP 429 beyond it; also sets vLLM `--max-num-seqs` / SGLang `--max-running-requests`. `0` genuinely means unlimited now, and also omits that engine flag entirely), dynamic GPU memory allocation (`--gpu-memory-utilization` for vLLM, `--mem-fraction-static` for SGLang), **SSE keepalive during cold start** (`gpu-arbiter/sse_keepalive.go`; `DEVAI_SSE_KEEPALIVE_SECONDS` default 10, `DEVAI_SSE_KEEPALIVE_GRACE_SECONDS` default 5, interval 0 disables). A client is otherwise held for the whole launch window with zero bytes sent, and an NVFP4 cold start is bounded by `HEALTH_TIMEOUT_SECONDS`=600s -- long enough for a browser or corporate proxy idle timeout to kill the connection and waste the load. Past the grace window the router writes `: keepalive <n>` SSE comment frames, which every OpenAI/Anthropic client ignores. Gated on an explicit `"stream": true` AND a `/v1/` path: Ollama's native `/api/chat` / `/api/generate` stream newline-delimited JSON and default `stream` to true when absent, so a comment frame there would corrupt the wire format for every Ollama-native client. Nothing is written and no header is committed inside the grace window, so the warm path keeps its exact status codes; once the first frame IS written the response is a committed `200 text/event-stream` and a later launch failure is reported in-band (`data:` + `data: [DONE]` on OpenAI surfaces, `event: error` on `/v1/messages`) rather than as a 5xx. Per-request context cap comes from `<name>@<ctx>` override (picker-supplied) or the probe cache row's `min(model.max_context, MAX_CONTEXT_LEN)`. Both `currentModel` and `currentContext` are tracked per backend; either change triggers a recreate -- **with one Ollama exception**: only an explicit `<name>@<ctx>` pin may move an Ollama tier. A BARE `<name>` is served from whatever tier is already loaded and never recreates the container to re-derive one, so a pinned client and a bare-name client no longer thrash the runner between tiers (`ctxPinned` in `ensureOllamaRunning`; see docs/router.md "Ollama: `@<ctx>` is what pins a tier"). **Reasoning policy** (`DEVAI_REASONING`): global policy is `auto|off|low|medium|high` (default auto); per-request suffix `::<reasoning>` overrides (e.g., `::nothink` -> `enable_thinking=off` for inline-reasoning models). Ollama uses native `think:` field; **vLLM** injects `extra_body.chat_template_kwargs.enable_thinking` plus `reasoning_effort`. **SGLang differs and the difference is load-bearing**: `extra_body` is not a field on its `ChatCompletionRequest` and is silently DISCARDED (no 422), so the router sends `chat_template_kwargs` **top-level**; `separate_reasoning` is dropped as a policy lever entirely (it defaults True, so the old enable branch was a no-op, and its disable branch is a PARSING switch that merely stops the `<think>` trace being split into `reasoning_content` -- the model still thinks and the user still pays for it); disable is `reasoning_effort: "none"`, which SGLang expands to BOTH template key spellings (`thinking` and `enable_thinking`) and which must be sent top-level because the Harmony guard that rejects "none" for gpt-oss reads the value popped out of `chat_template_kwargs`. Capability=`inline` + policy=`off` now returns `reasoningDisable` (explicit user opt-out). **Anthropic `/v1/messages` normalisation** (`gpu-arbiter/anthropic_compat.go`, `normaliseAnthropicMessages`): Claude Code sends a `role:"system"` message INSIDE `messages[]` alongside a correct top-level `system`, which vLLM's and SGLang's stricter compat shims reject with `400 ... ('body','messages',1,'role')` on every turn -- so the picker advertised every vLLM/SGLang row while the default agent could not complete a single turn against any of them. The router folds every non-`user`/`assistant` message into the top-level `system` block list, preserving order, promoting a bare-string `system`/`content` to a text block, and returning the original bytes byte-identical when there is nothing to move. Gated on backend `vllm`|`sglang` AND path exactly `/v1/messages`; Ollama is verified tolerant and deliberately untouched. Scope was fixed by replay against the live engines with a REAL captured 183 KB Claude Code body (25 tools, all beta fields): as-is 400, folded-with-beta-fields-kept 200 -- so folding alone is sufficient and `context_management`/`output_config`/`thinking`/`metadata`/`tools` pass through untouched. **Responses API** (`gpu-arbiter/responses_compat.go`, `applyResponsesPolicy`): Codex speaks ONLY `/v1/responses` (`wire_api="chat"` was removed upstream), and the reasoning rewrite used to be gated to `/v1/chat/completions` + `/v1/messages`, so Codex traffic got no reasoning policy at all. Both engines implement the endpoint (vLLM v0.22.1 verified live; SGLang v0.5.10 registers it at `http_server.py:1563`). **The shape differs and the wrong one is accepted silently** -- measured on gpt-oss-20b: `reasoning:{"effort":"low"}` cuts reasoning_tokens 298->37, while `reasoning_effort:"low"` returns 200 and is IGNORED (282 tokens), so widening the path gate alone would have been a fake fix. `low|medium|high` map to `reasoning.effort`; `auto` injects nothing (byte-identical body -- the checkpoint's default is the right answer); `off` emits `"none"` only behind the existing `disable_verified` gate, which is what keeps Harmony models out (they 400 on `none`). Verified end-to-end through the router: auto 249 -> `::low` 59 -> `::high` 298 tokens. Tool STRIPPING already applied (not path-gated, same top-level field names); tool_choice PROMOTION is now explicitly skipped there -- the engine rejects a pin outright (501 flat / 400 nested), and it had only been no-opping by accident because `toolNameAt` reads the nested Chat shape while Responses requires a flattened one. **Tool stripping** (`maybeStripTools`): when vLLM/SGLang models have no probe-verified tool parser, the router drops `tools` and `tool_choice` from the request body to prevent "BadRequestError: auto tool choice requires --enable-auto-tool-choice and --tool-call-parser" rejections. Disable rewrite is gated on `disable_verified` (per-model probe outcome). That field was TAUTOLOGICAL for SGLang until 2026-07-28 -- the probe sent `separate_reasoning=false`, which tells SGLang not to populate `reasoning_content`, then checked `reasoning_content` was absent, so it could only ever pass (true for 8 of 9 SGLang reasoning rows, false for 0 of 11 vLLM rows on the same checkpoints). The probe now sends what the router sends and additionally requires no inline `<think>` in `content`. Re-measured on the wire: only **Qwen3.5-9B-NVFP4** genuinely disables; `gpt-oss-20b` keeps emitting `reasoning_content`, and `NVIDIA-Nemotron-Nano-9B-v2-NVFP4` empties `reasoning_content` but moves the whole trace into `content` -- the exact case the old check passed. **Model-store mutation guard** (`ollamaMutationPaths` + `makeMutationGuard` in `main.go`, registered by `newBackendMux` on *every* backend listener): `/api/pull`, `/api/create`, `/api/push`, `/api/copy` and `/api/delete` are refused with 403 and a JSON body naming the sanctioned path, for every HTTP method, before the catch-all proxy sees them. Previously the router registered only four routes and everything else proxied straight through, so any agent in the lab -- or an Open WebUI user -- could pull an unprobed model and then serve it, bypassing the probe-cache fit gate entirely. The operator's own pipeline is unaffected: `select-models.py` shells out with `podman exec devai-ollama ollama pull/create` and the probers talk to `devai-ollama:11434` directly, so neither traverses the router. Sanctioned path remains `make model-pull` followed by `make probe`.
- **devai-ollama** -- Unmodified `ollama/ollama:latest`. GGUF models, GPU auto-detected. `OLLAMA_MAX_LOADED_MODELS=1` ensures clean model switching. `OLLAMA_CONTEXT_LENGTH` defaults to 262144 (compose env).
- **devai-vllm** -- `vllm/vllm-openai:v0.22.1-x86_64-cu129-ubuntu2404` image. NVFP4 / safetensors models. Starts as a `sleep infinity` placeholder; the router recreates the container with the dynamic entrypoint on first request to port 11435. Entrypoint passes `--kv-cache-dtype <per-model>`: resolved from the probe cache cell covering the launch ctx (cells are stamped with the dtype they were measured under; unstamped legacy cells decode to fp8 -- the historical hardcode they were factually probed with). fp8 KV is what lets the pool fit 128K context on 24 GiB cards (default fp16 KV pushes Nemotron-class checkpoints past 24 GiB during model load); a model with VRAM slack can be re-probed with `PROBE_KV_CACHE_TYPE=auto` to serve unquantized KV instead -- no global dtype policy. Entrypoint also injects `--reasoning-parser` and `--enable-auto-tool-choice --tool-call-parser` when the v2 probe cache has confirmed values for the model (sourced from each family's curated `parsers:` block in `scripts/model-families.yaml`). Per-model **recovery flags** are appended after the parser flags from `deploy/recovery-flags.json` -- `engine_flags` (CLI args, e.g. `--enforce-eager` for models whose CUDA graph workspace pushes them past 24 GiB at high context), `engine_env` (env vars merged into the container), the optional `backends` allow-list (absent = every backend; all 10 current entries are scoped to `["vllm"]`, so vLLM-only rescue flags are no longer handed to SGLang launches) and the optional `image` (per-model container-image override, falling back to `$VLLM_IMAGE`). Both probe-time and serve-time launches read the same JSON -- including the same `backends` filter -- so probe-cache fit data stays consistent with serve-time memory math.
- **devai-sglang** -- `lmsysorg/sglang:v0.5.10.post1-cu130` image (pinned; bump via `deploy/backend-flags.yaml` + `make verify-backend-flags`). NVFP4 / safetensors with RadixAttention for multi-turn speedup. Same `sleep infinity` placeholder + on-demand recreate pattern as vLLM. Entrypoint injects `--reasoning-parser` / `--tool-call-parser` from the probe cache. SGLang has no `--enable-auto-tool-choice` analogue -- `--tool-call-parser` alone enables tool parsing. Entrypoint ALWAYS passes `--disable-piecewise-cuda-graph`: v0.5.10's piecewise CUDA-graph default torch.compiles the forward and Dynamo can't trace flashinfer's FP4 JIT path (`modelopt_quant.py:1482`), so without it every NVFP4 load crashes. With it, arch-supported NVFP4 models serve (Qwen3-8B/14B, Llama-3.1-8B, Nemotron-Nano-9B, gpt-oss-20b); Gemma-4/Qwen3.5-3.6-MoE/diffusiongemma still fail on genuine arch/quant gaps and are served by vLLM instead. See `scripts/model-families.yaml` SGLang NVFP4 status block.
- **devai-webui-proxy** -- nginx TLS proxy for Open WebUI (mkcert certs or self-signed fallback).
- **devai-open-webui** -- Web chat interface, connects to router's ollama port (:11434).
- **devai-logger** -- Sidecar that streams `podman --remote logs --follow` for every devai-* container into `/var/cache/devai/logs/<service>.log`. Survives container restarts. Tail via `make logs SERVICE=<name> [LINES=N]`. Requires the `cache_logs` LV (one-time setup via `make setup-logs`).
- **devai-mcp-gateway** (opt-in, profile=`mcp`) -- Docker MCP Gateway peer service, image `docker.io/docker/mcp-gateway:v0.43.3`. Single HTTP endpoint that any MCP-aware agent can target; per-call MCP server containers spawned via the host's Podman socket. `--block-secrets` + `no-new-privileges` enforced. Dual-homed on `[default, devai-lab-egress]` -- lab containers sit on `devai-lab-egress`, so a gateway published only on `devai-net` is unresolvable from every agent, which is what it used to be. `devai-mcp-gateway` and `mcp-gateway` are in both NO_PROXY lists (Makefile `PIPELOCK_NO_PROXY`, `bin/devai-agent` `NO_PROXY_HOSTS`).

  **Catalogs.** Third-party servers resolve from Docker's official MCP catalog, which the gateway loads by default and which upstream digest-pins and maintains; this repo's `deploy/mcp-catalog-devai.yaml` is merged on top with `--additional-catalog` (NOT `--catalog`, which would replace the official one) and declares only `devai-model-status`. We hand-maintain zero third-party definitions. `deploy/mcp-servers.yaml` no longer exists -- it used the wrong schema (`apiVersion` / `schemaVersion` / `servers` as a list, against the gateway's top-level `name` / `displayName` / `registry`-as-a-MAP with a required per-entry `type`), so it parsed to an empty registry and the gateway exposed zero tools regardless of anything else. It also pinned all 14 servers at `:0.7.0`, a tag that exists for none of them.

  **Enabled servers (15, via the compose `--servers=` list; names are upstream catalog keys and are CASE-SENSITIVE -- note `SQLite`, not `sqlite`):** filesystem, git, SQLite, fetch, memory, time, sequentialthinking, duckduckgo, arxiv-mcp-server, wikipedia-mcp, github-official, firecrawl, hugging-face, context7, devai-model-status.

  **Verified live on 2026-07-25:** `make mcp-up` starts the gateway (~13s to initialise); a real MCP handshake (`initialize` -> `notifications/initialized` -> `tools/list`) over streamable HTTP at `http://127.0.0.1:8088/mcp` returns **134 tools**; all three first-party tools are present and a real `tools/call` of `list_fitting_models` returns 27 models. The gateway mints a bearer token at startup and prints it to its log ONCE ("Use Bearer token: ..."); clients MUST send `Authorization: Bearer <token>`, the URL must include the `/mcp` path, and responses are SSE-framed (`event: message\ndata: {...}`).

  **Known limitations, unresolved:** (1) 2 of the 15 servers fail to start -- `filesystem` ("Error accessing directory : ENOENT, stat ''") and `arxiv-mcp-server` ("invalid docker volume ':/app/papers': source and target are required"). Both want a host path supplied through gateway config that is not set yet; the 134 tools are what the other 13 provide. (2) `get_router_status` enumerates but returns unreachable when called: the gateway spawns each server in its own container WITHOUT attaching it to `devai-net`, so the spawned container cannot resolve `devai-router`. The catalog schema has no field for a custom network (only `disableNetwork` / `extraHosts`). The other two first-party tools read caches baked into the image and work. This needs an operator decision. (3) `github-official` and `firecrawl` enumerate their tools but need secrets to call anything, and that secrets path is unverified -- the upstream secret names are `github.personal_access_token` and `firecrawl.api_key`, NOT the `GITHUB_TOKEN`-style keys the old example file used. `hugging-face` and `context7` are `type: remote` and connect anonymously, so they need no secret.

  See `docs/mcp.md`.

### Supporting services

- **apt-cacher-ng** -- APT package cache (port 3142, published on 127.0.0.1 only)
- **Registry mirror** -- Docker Hub pull-through cache (port 5000, published on 127.0.0.1 only)

**Port-publishing posture:** every compose service publishes to `127.0.0.1` except `devai-webui-proxy` (`${WEBUI_PORT:-8443}:443`, deliberately LAN-reachable for browser access over mkcert TLS). Anything unauthenticated -- apt-cacher-ng, the registry mirror -- stays on loopback, and so does the MCP gateway: it does require a bearer token, but it also holds a read-write podman socket. Remote access means an SSH tunnel or an authenticating proxy, not a widened bind.

All services share `devai-net` network. Model data stored under `/var/cache/devai/`. Secret render targets live on a tmpfs at `/run/devai/` (per `docs/secrets.md`).

**`/var/cache/devai/` mount-point convention:** the top-level folders under `/var/cache/devai/` (e.g. `ollama/`, `vllm/`, `sglang/`, `pip/`, `logs/`, `registry/`) are each an external-volume mount point (a dedicated LV/mount), not ordinary directories. Do NOT create new top-level directories directly under `/var/cache/devai/` -- a new folder there is not backed by a volume and silently lands on the root filesystem. **vLLM and SGLang safetensors live on their OWN volumes (`vllm/` = `cache_vllm`, `sglang/` = `cache_sglang`), NOT under `ollama/`.** This isolation is deliberate and load-bearing: they were formerly under `ollama/models/vllm/`, where an Ollama cache cleanup (`make ollama-clean`) once deleted the entire non-Ollama vLLM store because its orphan-pruning keys "orphaned" off Ollama manifests. Keeping non-Ollama weights off the Ollama tree makes that class of accident impossible. `VLLM_MODELS_DIR`=`/var/cache/devai/vllm`, `SGLANG_MODELS_DIR`=`/var/cache/devai/sglang`; set up via `sudo LV=cache_vllm MOUNTPOINT=/var/cache/devai/vllm SIZE=300G deploy/setup-logs-volume.sh` (the generic volume-setup script, despite the `logs` name). That script **preserves existing data**: pointed at a non-empty unmounted directory it lists the contents and aborts rather than wiping them (pass `WIPE=1` to opt into deletion), and -- the case that matters most here -- it **refuses outright** when the target is already mounted from a different device, because everything downstream of that point (lvcreate, mkfs, and above all the unconditional /etc/fstab rewrite) would displace the volume that is already serving 300 GB of weights. **Getting weights into the SGLang store is a separate, explicit step:** `make model-pull` (including `NAME=...`) always downloads into the vLLM store, so an SGLang-served model needs `python3 scripts/select-models.py --name <n> --download --hf-store sglang`. That is **not** a second download: since the 2026-07 storage change both stores live on the same filesystem (`/var/cache/devai`, `vgais-cache`), so `pull_hf` calls `try_link_from_peer_store` first and **hard-links** the weights from the vLLM store (57.5 GiB reclaimed across the 5 kept models on this fleet). Hard links, not symlinks -- each engine's container bind-mounts only its own store, so a symlink into the peer store would dangle inside the container. The path falls back to a real download, without erroring, when the peer copy is missing or the two stores are on different devices (`st_dev` mismatch, i.e. the pre-2026-07 layout). `scripts/link-hf-store.py --from vllm --to sglang` does the same job standalone for stores populated before this existed. Caches that are not external-volume-backed (e.g. the load-probe corpus, see `scripts/_probe_load.py`) belong under the user cache `~/.cache/devai/` (`DEVAI_PROBE_CORPUS_DIR` / `XDG_CACHE_HOME` aware), not here.

### SSL / HTTPS

- JupyterLab: auto-detects mkcert certs in `~/.jupyter/ssl/`
- Open WebUI: nginx proxy with mkcert certs or self-signed fallback
- Generate with `mkcert <IP>` on browser workstation, copy to container host

### Model picker (shell + Jupyter)

Interactive model -> agent selection via fzf. Used by `make shell-*` (via `agent-picker`), the standalone `devai-agent` launcher, and JupyterLab launcher cards.

- `scripts/model-picker.py` -- Python TUI, two-step fzf picker. Reads all three probe caches (Ollama digest-keyed v3; vLLM/SGLang repo+sha-keyed v2) for fit data and `deploy/.bench-cache.json` (schema v3) for the bench-score columns. Bench lookup is keyed by `(model, backend, ctx)` so the per-row TPS / CODE% / CODE+% / MMLU% / GPQA% / TOOLS / LEAK% columns reflect the user's chosen ctx exactly. When a model has bench data at other ctxs but not the selected one, the preview pane reports `Bench: not available at ctx=<N> (have ...; run \`make bench --ctx <N>\` to populate)` instead of silently substituting a different ctx's number. Falls back to `deploy/models.yaml` for catalog metadata only. Renders one row per `(model_dir, backend)`, deduplicated and ranked. Columns: `##` (1-based line number; renumbers per sort), `CTX`, `TAG` (NVIDIA- prefix stripped for display only -- `m["name"]` is preserved for agent commands and cache lookups), `BACKEND` (right-aligned), `PARAMS` (e.g. `30B/A3B` for MoE), `TYPE` (Dense/MoE; HF probes don't fill `moe.experts_total`, so `_is_moe` falls back to `/A` in `param_size` plus a known-MoE-name list to catch `gpt-oss-20b`), `FORMAT` (NVFP4/MXFP4/BF16/Q4_K_M/...), `TOOLS` (agentic `tools_use` bench score 0-100 -- right tool, exact args, no fabrication, benched in the model's native tool_choice mode; falls back to `Yes`/`No` parser-presence when unbenched), `TPS`, `CODE%` (HumanEval), `CODE+%` (HumanEval+, EvalPlus hardened), `MMLU%` (MMLU-Pro), `GPQA%` (GPQA-Diamond), `LEAK%`, `VRAM`. The saturated `REAS%`/`TOTAL%` composites were retired for these sharper discriminators. Bindings: `ctrl-s` cycles sort mode (`GPQA > TPS > CODE > CODE+ > MMLU > TOOLS > CTX`), `ctrl-r` flips direction (`desc <-> asc`), `?` (or `ctrl-p`) toggles preview pane. Active-column header gets a down/up arrow glyph (`U+25BC` / `U+25B2`); the sort-note line shows current mode + direction. Search uses `--exact` (literal substring, no fuzzy) and is labeled `Search:`. Column header / sort note / formula note are non-navigable via `--header-lines=3` so the preview always corresponds to a real model row. The always-visible formula note now reads `* = VRAM formula estimate. TOOLS = agentic tool calling. CODE = HumanEval pass@1. CODE+ = HumanEval+ (hardened). MMLU = MMLU-Pro. GPQA = GPQA-Diamond.` -- no REAS/TOTAL. Preview pane shows model details, a per-format quant explanation under `Format:` (NVFP4 / MXFP4 / FP8 / BF16 / Q*_K_*), a `Model properties` section with per-metric rank vs peer rows (including a TOOLS line -- the agentic tool-calling bench score with its own peer rank) + peak VRAM headroom + steady-state TTFT + leak caveat, a `Recommended for:` section ranking five typical use cases (Coding / Math-analysis / General-reasoning / Doc-summary / Doc-Q&A) best-first with a 0-100 score + tier label (`_use_case_ratings`; coding/math/reasoning map to direct benchmarks, while summary and doc-Q&A have no direct bench and are context-weighted proxies -- ctx is 45%/40% of the score, plus MMLU comprehension, GPQA reasoning for Q&A, and 1-leak faithfulness -- an unmeasured leak or TPS scores a neutral 0.5 rather than a free 1.0 or a punitive 0.0; tune the weights there if typical doc lengths differ), and a use-cases blurb extended with one bench-derived sentence (top-of-list callout or leak warning) when applicable. Inline-reasoning models offer an ON/OFF toggle in a sub-modal after the row is picked. Ctrl-C / Esc exits cleanly. **Cursor history on every level** (`_LAST_POS` + `_fzf(memory_key=...)`): each modal -- model list, context tier, agent, reasoning toggle, MTP toggle, aiagent GPU mode -- remembers where the cursor was and restores it on re-entry, so backing out of a sub-modal returns you to the row you were on instead of the top of the list. Implemented with fzf's `start:pos(N)`, which **requires `--sync`** (without it the binding can run before the item list is loaded and the position is silently dropped). Positions are fzf ITEM positions, not `lines` indices -- they differ by `header_lines`, since header rows are not items. A stale position from a longer list is clamped. High-contrast colour scheme (bright cyan headers, yellow pointer, dark-grey bar, light-grey legend).
- `scripts/agent-picker.sh` -- Shell wrapper, execs model-picker.py.
- `scripts/aiagent-launcher.sh` (installed as `/usr/local/bin/aiagent-shell`) -- the picker's "AIAgent (shell)" agent. aiagent (github.com/devitops-com/aiagent) is a DSPy CLI the user drives herself, so the picker does NOT exec it: `_build("aiagent", ...)` exports the router endpoint (`AIAGENT_API_BASE=<router>/v1`, per aiagent's README env-fallback contract), `AIAGENT_MODEL` (bare Ollama tag or `<name>@<ctx>` for vLLM), and OpenAI-compat fallbacks, then execs this launcher which prints a hint and drops into interactive bash. GPU policy via `DEVAI_AIAGENT_GPU`: default `router-only` sets `CUDA_VISIBLE_DEVICES=""` so aiagent can't contend with the router-loaded model for VRAM; a picker sub-modal (`_resolve_aiagent_gpu_mode`) -- or a pre-set env value, which suppresses the prompt -- opts into `share`. Installed into the lab image from a pre-fetched makeself bundle (`make fetch-cli` -> `deploy/Dockerfile.lab`) to an isolated `/opt/aiagent` so its bundled CPython stays OFF the system PATH and never mixes with the lab's `/usr/local` python (a `uv pip install --system` would otherwise bleed lab modules into it). See `docs/aiagent.md`.
- `bin/devai-agent` -- Host-side Python launcher. Reads/writes `~/.devai/preferences.yaml` (vram, context, last_model, last_agent, last_work_dir, agent_session_file). Bind-mounts `~/.devai/` to `/devai-host` (rw) so the picker can drop `.last-pick.json` for the launcher to consume on exit. Bind-mounts `~/.devai/model-picker.py` over `/usr/local/bin/model-picker` and the four cache files (`.ollama-/.vllm-/.sglang-reasoning-cache.json` plus `.bench-cache.json`) over `/etc/devai/` so picker edits and re-benches don't require an image rebuild. The bench-cache mount is what populates the picker's TOOLS / TPS / CODE% / CODE+% / MMLU% / GPQA% / LEAK% columns; absent file -> the columns render as `-`. `make install` writes the symlinks under `~/.devai/`. Pre-flight checks for image + `devai-net`.
- `packages/jupyter-ai-launchers/src/index.ts` -- JupyterLab extension, each card runs `model-picker --agent <name>`.

**Filter:** the picker hides any `(model, backend)` the exclusion ledger carries a BENCH verdict for (`_MS.is_bench_excluded`, scoped to the offered ctx). Gated on the LEDGER, never on the bench cache's raw `drop_recommendation`: a drop is advisory by design ("never edits the exclusion ledger -- that stays an explicit operator action"), so it stays visible until an operator records it with `make bench-sync RECORD_DROPS=1` or `_model_status.record_bench_verdict`. Restore with `make model-status CLEAR=<name>::<backend>`; the count shows in the picker's hidden-rows footer. Before this, `drop_recommendation` was read ONLY by the bench harness -- not the picker, not the router -- so a model the bench had recommended dropping stayed on offer indefinitely. The picker surfaces the backends in `_PICKER_BACKENDS` -- `ollama`, `vllm` and (since 2026-07-27) `sglang`. SGLang was hidden from 2026-05-01 on agent-compat grounds and re-enabled by operator decision once it was probed and benched here; the rationale and what did/didn't change is recorded at that constant. Note `_dedup_hf_rows` keys on **(name, backend)**, not name alone -- it used to key on name and drop every backend but the highest-priority one, which was a silent no-op while SGLang was hidden and swallowed all four SGLang rows the moment it was re-enabled. vLLM still sorts ahead of SGLang within a name. The picker shows one row per `(model, backend)` pair at the picker's VRAM band (env `VRAM` or `GPU_MEMORY_GB`). A model is eligible only when the relevant probe cache contains a `fits=true` (vLLM/SGLang) or `fully_on_gpu=true` (Ollama) cell at some context tier. There is no interpolation -- gaps mean "re-run `make probe-vllm` / `make probe-sglang`". HF rows whose backend has no fitting probe entry stay hidden until probed. See `docs/backends.md`.

**Per-session context binding & reasoning overrides.** Two paths:

- **Ollama**: the picker emits just the parent name (or `<name>::nothink` for reasoning-off). EXCEPTION: mixed-KV models -- when some probed tiers only fit with quantized KV (cell `kv_cache_type=q8_0`, e.g. qwen3.6:35b-a3b-mtp at 128K), the picker shows a context-tier sub-modal and pins `@<ctx>` on the emitted name so the router launches the chosen tier with its probed KV dtype (`resolveKVCacheType` -> `OLLAMA_KV_CACHE_TYPE` + `OLLAMA_FLASH_ATTENTION=1` in the recreated container); quantized tiers carry a weaker-long-form-reasoning warning (GPQA -10 pts measured). See docs/backends.md "Per-tier KV-cache dtype". Otherwise KV cache is allocated *dynamically* per request from the loaded `context_length` ceiling (set globally via `OLLAMA_CONTEXT_LENGTH` env, default 256K). Clients hitting `/api/chat` / `/api/generate` get `options.num_ctx` injected by the router's `setNumCtx` (Ollama honours it on those paths). Clients hitting `/v1/chat/completions` or `/v1/messages` get the global `OLLAMA_CONTEXT_LENGTH` -- Ollama upstream ignores `options.num_ctx` on those compat surfaces and we accept that. `::nothink` suffix forces `enable_thinking=false` even when the global `DEVAI_REASONING` policy isn't off.
- **vLLM / SGLang**: the picker emits `<name>@<ctx>` (e.g. `Llama-3.1-8B-Instruct-NVFP4@32768`) or `<name>::<reasoning>@<ctx>` for reasoning overrides. The router's `peelControlSuffixes` strips the `@<ctx>` / `::<mtp>` / `::<reasoning>` suffixes **in any order** (it peels whichever is currently trailing and loops until none remain), then propagates the ctx into `containerRecreate` which sets `--max-model-len` (vLLM) or `--context-length` (SGLang), and handles the reasoning override (e.g. `::<reasoning>` -> `enable_thinking=off` even on models with inline capability). Order-independence matters because some clients append their own suffix out of the picker's canonical `<name>::<reasoning>::<mtp>@<ctx>` order -- notably aiagent/litellm appends its `default_reasoning` as `::<reasoning>` AFTER the `@<ctx>`, producing `<name>@<ctx>::<reasoning>`; a strict ctx-last strip would leave `@<ctx>` glued to the name and the vLLM/SGLang allowlist would reject it as unknown. No client-side tag materialization needed -- the router's tracking handles the rest.

**Do not add custom tags to cached models.** In particular, do not derive `<parent>:<tag>-ctx<N>` Modelfile siblings via `ollama create` to bake `num_ctx` (or any other PARAMETER) in. Per-session context is plumbed dynamically -- via the router's `setNumCtx` injection on Ollama's `/api/chat` and via the `@<ctx>` suffix for vLLM/SGLang launch flags -- so derived tags add nothing the runtime can use. They share digests with the parent (`make cache-status` then shows duplicate-looking rows), the picker filters them via `_ctx_tag` and the prober skips them via `_CTX_VARIANT_RE`, so they're inert leftovers from the pre-3a98ed0 design. The only sanctioned `ollama create` call is `select-models.py:pull_gguf` writing the canonical catalog tag from a downloaded GGUF blob; nothing else should mint Ollama tags.

**Use models as delivered -- no operator YaRN / rope_scaling edits.** Every model serves only up to its as-shipped context ceiling (`max_position_embeddings`, rope-extended when the checkpoint already ships a `rope_scaling` block). We do NOT hand-edit a checkpoint's `config.json` or inject `--rope-scaling` / `--hf-overrides` to extend a model past what it was delivered with. A model trained short with no YaRN config (e.g. `Qwen3-8B/14B-NVFP4`, `max_position_embeddings=40960`, `rope_scaling=null`) is used at that 40K ceiling and capped to the largest standard tier within it (32K); it is NOT extended to 131K by adding YaRN. Long context comes ONLY from models that ship it -- native (e.g. Qwen3.5-9B, Llama-3.1-8B at 128K, DiffusionGemma at 256K) or YaRN baked-in (e.g. gpt-oss-20b to 131K). This is enforced automatically by the load probe's `position_limit` cap (see `_probe_hf_common.effective_position_limit`): the fit probe never advertises a context the model would device-side-assert on at serve time. Rationale: as-delivered checkpoints are reproducible and avoid the documented YaRN sub-32K quality regression; the fleet already covers long context with purpose-built models, so per-model rope surgery buys nothing.

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
deploy/models.yaml            -- Auto-generated catalog (every variant the upstream catalog declares); filters out :latest, bare-size aliases, routing variants, cloud placeholders; recognises quant markers
deploy/.ollama-reasoning-cache.json -- Ollama probe cache (schema v3, digest-keyed); per-cell: actual_total_gb, actual_vram_gb, fully_on_gpu, per-cell capability, kv_cache_type, flash_attention (the OLLAMA_FLASH_ATTENTION setting the fit was measured under), timestamp; captures capabilities array
deploy/.vllm-reasoning-cache.json   -- vLLM probe cache (schema v2, repo+sha-keyed); top-level: reasoning_parser, tool_parser, disable_verified; per-cell: fits, evidence. The LOAD probe (`make probe-load-vllm`) additively augments fitting cells with serving_ok / serving_peak_gb / transient_gb / needle_score / predicted_logits_gb -- serving-time VRAM under a near-full-context request, catching the per-step transient (softcap-logits + attention workspace) the single-shot fit probe misses. Router + picker gate on serving_ok ONLY when present (absent = pre-load-probe behaviour). No schema bump. **Single-cell (binary-search max-ctx):** the fit + LOAD probers now binary-search the largest context that both fits and serves on the 32K-multiple grid (up to `min(MAX_CONTEXT_LEN, position_limit, max(PROBE_CONTEXTS))` -- PROBE_CONTEXTS now caps the search ceiling instead of being parsed and dropped) and keep exactly ONE winner cell per (model, backend) instead of a fixed 32K/64K/128K/256K scan; consumers treat that cell as covering every smaller ctx (KV monotonicity), and a 32K-fails model writes no cell + an `oom` ledger entry. **`_meta` image-drift block:** a top-level `_meta` key (current_image_digest + current_image_ref + image_history), NOT a model row -- stamped by the prober with the backend image digest it measured against so the router can detect a moved tag; every reader skips `_`-prefixed keys. `make probe-check` reports drift; the router serves-with-`X-DevAI-Warning` on a stale backend. See docs/backends.md + docs/router.md.
deploy/.sglang-reasoning-cache.json -- SGLang probe cache (schema v2, repo+sha-keyed; same shape as vLLM)
deploy/.bench-cache.json      -- Bench cache (schema v3). Top-level `_meta` block holds `host_env_history` keyed by 12-char SHA-256 id (kernel, driver_version, gpu_name, gpu_memory_gb, cuda_version, captured_at) plus `current_host_env_id` pointer. Every row stamps `host_env_id` so re-benches against a different driver/kernel are auditable. Row keys carry a `::<ctx>` suffix: `<repo>@<sha>::<backend>::<ctx>` (HF) or `<digest>::<backend>::<ctx>` (Ollama) -- so the same model benched at 32K and 128K lands in distinct rows instead of silently overwriting (under MTP a single model can differ by 17+ tok/s between ctx tiers). Rows mirror the suffix in a `context: int` field and hold `tasks` (gsm8k/humaneval/tools_use/leak/longctx subsets) + `metrics` (peak/mean VRAM, ttft, tps_sustained_p50, vllm_kv_cache_usage_perc, etc.). v2 -> v3 migration runs idempotently on first writer invocation; the 9 historical pre-v3 rows are mapped via `_bench_core.RECOVERED_CTX_MAP` (a one-shot artefact captured from the 2026-05-05 router log -- never edited after the migration ships, future unrecognised rows land at ctx=0 and re-bench is the right answer). Gitignored.
deploy/backend-flags.yaml     -- Pinned launch-flag *names* per backend; `make verify-backend-flags` asserts presence after image bumps
deploy/docker-compose.yaml    -- Infrastructure services (vllm/sglang start as `sleep` placeholders; router recreates on demand)
deploy/Dockerfile.base        -- Base image
deploy/Dockerfile.lab         -- Lab image. Also installs the `sky` CLI offline from pre-fetched wheels, and (since 2026-07-25) actually COPYs scripts/sky-setup.sh to /usr/local/bin/sky-setup.sh -- docs/skypilot-user-guide.md had told users to run it for months while nothing put it in the image. The `sky` CLI is a user-facing tool and is unaffected by the freeze of the system-side fleet provisioner.
deploy/Dockerfile.router      -- Router image (distroless)
deploy/Dockerfile.mcp-modelstatus -- devai-model-status MCP server image (distroless, 2-stage like Dockerfile.router). Bakes models.yaml + the 4 probe/bench cache files into /etc/devai/ at build time (falls back to an empty `{}` cache for any that don't exist yet) rather than bind-mounting -- see docs/mcp-model-status.md for why. Build via `make build-mcp-modelstatus-image`.
deploy/mcp-catalog-devai.yaml -- devai's FIRST-PARTY MCP catalog and the only MCP catalog this repo maintains. Real gateway schema (top-level name / displayName / registry-as-a-MAP, `type` required per entry). Contains exactly one entry, devai-model-status, pointing at the locally built localhost/devai-mcp-modelstatus:latest (the gateway runs servers with `--pull never`, so a local-only tag is the right shape). Replaces the deleted deploy/mcp-servers.yaml. Third-party servers are NOT declared here -- they come from Docker's official catalog; to enable one, add its upstream key to the compose `--servers=` list.
deploy/mcp-gateway.env        -- REFERENCE ONLY: documents the non-secret MCP gateway knobs (MCP_PORT default 8088). Nothing sources this file -- set MCP_PORT in `.env` or the shell.
deploy/mcp-secrets.sops.env   -- Encrypted secrets for github-official + firecrawl (the only two enabled servers that need any). Would be TRACKED at this canonical name (`.gitignore` un-ignores `deploy/*.sops.env` explicitly; only render targets and `*.env.plain` are ignored), but only `mcp-secrets.sops.env.example` exists in-tree today. Operators copy the example -> fill in real values -> `sops --encrypt` -> commit. Upstream's secret NAMES are `github.personal_access_token` and `firecrawl.api_key` -- the example still carries the older `GITHUB_TOKEN`-style names, and the delivery path is unverified end to end.
deploy/setup-secrets-tmpfs.sh -- Idempotent tmpfs mount at /run/devai (4 MiB, mode 0700, nodev/nosuid/noexec). One-time per boot; `make secrets-tmpfs` wraps it.
.sops.yaml                    -- creation_rules for sops/age. Single rule covers `deploy/*.sops.env`. Operators add their host's age public key to the `age:` list, then `sops updatekeys`.
gpu-arbiter/main.go           -- GPU arbiter source (multi-port proxy, ~4100 lines Go). `--mode` is still accepted but no longer dispatches: anything other than `single` exits with a pointer to attic/README.md. Also holds the model-store mutation guard (ollamaMutationPaths, makeMutationGuard, newBackendMux).
attic/                        -- Frozen work, deliberately NOT deleted. attic/cluster-mode/ holds the 22 cluster/fleet Go files (each `//go:build devai_frozen_cluster`, and attic/ is outside every Go module, so they are excluded twice over), the compose overlay + worker-bootstrap image + cloud-init, the frozen tests and fixtures, the 5 frozen docs, the removed compose fragment (compose.skypilot-fragment.yaml) and the 9 removed Makefile targets (Makefile.frozen-targets). Read attic/README.md first, and attic/cluster-mode/RESTORE.md before any thaw -- it lists defects that were open at freeze time.
devai-tools/                  -- Sibling Go module (own go.mod) hosting first-party devai CLIs. File-per-concern + table-driven-test convention mirrors gpu-arbiter's. Both modules' go.mod pin the same consolidated current-stable Go release; devai-tools' floor is still driven by cmd/devai-mcp-modelstatus's MCP go-sdk dependency (go >= 1.25.0), just no longer split from gpu-arbiter's version. Built/tested with the host's own Go toolchain (`make build-devai-tools`, `make test-devai-tools` -- no container, needs Go on PATH), output to devai-tools/bin/ (gitignored), not the repo-root bin/ (the Python devai-agent launcher's own install target).
devai-tools/cmd/devai-backup   -- Backup/restore CLI (snapshot/list/verify/restore). See docs/backup-restore.md.
devai-tools/cmd/devai-mcp-modelstatus -- devai-model-status MCP server (stdio, official modelcontextprotocol/go-sdk). See docs/mcp-model-status.md.
devai-tools/cmd/devai-gpu-vendor -- Flips DEVAI_GPU_VENDOR + its 3 derived .env vars (add-or-replace, comment-preserving). See docs/gpu-vendors.md.
devai-tools/internal/backup/  -- Manifest + tar snapshot/verify/restore logic, path-traversal validation shared by verify and restore.
devai-tools/internal/modelcache/ -- deploy/models.yaml + the 3 probe caches + the bench cache: parsing, the list_fitting_models join (downward-covering cell resolution matching the single-cell probe invariant, and `serving_ok:false` excluded), the get_model_bench lookup (`code_pct` = `humaneval_subset_*`, `code_plus_pct` = `humaneval_plus_subset_*`; the retired REAS/TOTAL composites are gone from both sides).
devai-tools/internal/routerclient/ -- get_router_status: per-backend `/health` probe against the compose service name `devai-router` (the router publishes no host ports). Mode is `single` or `unreachable`; the `cluster-head` mode and the `cluster_error` field went away with the cluster freeze, which also removed a doomed token-file read plus a doomed HTTP round trip from every call. The `/health` handlers are unauthenticated.
devai-tools/internal/envfile/ -- Add-or-replace .env key mutation preserving comments/other lines; devai-gpu-vendor's only writer of .env.
scripts/age-keygen-host.sh    -- One-time per host: generates ~/.config/sops/age/keys.txt mode 0600 + prints public key. Idempotent.
scripts/render-secret.sh      -- Generic single-file decrypt to tmpfs. Refuses non-tmpfs destinations (override via DEVAI_RENDER_ALLOW_NON_TMPFS=1) so a missing `make secrets-tmpfs` fails loudly.
scripts/mcp-health.sh         -- /health + /servers probe for the running MCP gateway. Liveness only, and NOT a functional check: it was not rewritten alongside tests/test-mcp.sh and cannot distinguish a gateway serving 134 tools from one serving zero (which is the exact failure this repo already shipped for months). Use `make mcp-test`. See docs/mcp.md for the verified per-endpoint behaviour.
scripts/sky-setup.sh          -- First-launch helper inside the lab: enumerates detected cloud creds, runs `sky check`, prints next-step + cost guidance. Installed to /usr/local/bin/sky-setup.sh by Dockerfile.lab. User-facing `sky` CLI only -- unrelated to the frozen system-side provisioner.
scripts/generate-catalog.py   -- Refresh deploy/models.yaml from upstream (HF + Ollama registry). Fails closed: if an upstream fetch error would cost catalog rows (HF size / Ollama manifest / Ollama tag-list / GGUF-list fetch raising, or a family with no arch_ref) it exits 1 and leaves the existing models.yaml untouched, so a network blip cannot silently truncate the catalog. Deterministic skips (412 platform-gated tags, zero-weight repos, empty `include:` filters) still write normally. Both Makefile call sites (`catalog-regen`, and `REGEN=1` inside `model-sync`) abort the target on that exit code.
scripts/catalog-discover.py   -- Read-only newer-version finder for tracked lineages (`make catalog-discover`). Groups families into lineages (brand+version+sub-lineage, auto-derived from family name + tracked repos; optional per-family `discover:` override block), then searches HF (by already-trusted author) + Ollama (next-version library probing) for untracked NEWER / GAP / SAME-version repos. Structural line-membership filter drops foreign cousins (Qwen3-VL/Next/Coder, distills, finetune brands). VRAM-band filter keeps only candidates in the GPU's usable range -- estimated weight VRAM (params x quant-format bytes) between MIN_VRAM_FRAC x GPU_MEMORY_GB (floor; below it wastes the GPU) and the family max x tolerance capped at GPU_MEMORY_GB (ceiling; above it won't load); unmarked '?' formats fetch the real on-disk size (shown '=' not '~'). Base/non-chat (pretraining) checkpoints also hidden -- name says base/pretrain, or conversational tag absent and name not instruct (conservative, keeps tag-less named-instruct quants). Too-big / too-small / base all hidden by default with counts; `--include-oversized` / `--include-undersized` / `--include-base` to show. Discovery is read-only; a separate confirmed `--add <repo>` path (the ONLY writer of model-families.yaml) appends one entry under an EXISTING family via comment-preserving line insertion (NEWER versions needing a new family are refused -- they need arch_ref/parsers curation). Probe before relying on anything added. Same review contract as catalog-suggest/llmfit-catalog-diff.py.
scripts/_card_hints.py        -- Derives backend parser hints from a checkpoint's OWN metadata (chat_template.jinja, or the embedded tokenizer_config form) instead of hand-curated `parsers:` blocks. Ordered discriminator rules, first match wins, each prediction carrying the evidence substring; never raises (absent metadata degrades to "no hint" = today's behaviour). `derive_parser()` is what the prober calls: a CURATED value always wins, derivation only fills gaps. Validated against 13 checkpoints across 10 curated families -- `gemma4`, `harmony`, `nemotron_json` and `hermes` derive cleanly (10/10 agreement with curated AND probe-recorded values), while three classes deliberately derive NOTHING because the markup does not determine the answer: `qwen3_xml_family` (four probe-verified checkpoints ship byte-identical `<tool_call>`+`<function=` markup and split evenly between `qwen3_xml` and `qwen3_coder`), bare `<think>` (qwen3 / deepseek_r1 / nemotron_v3 all emit it), and Gemma-4's `<|think|>` (measured: prompt text, not an output delimiter -- see docs/backends.md). A wrong tool parser mis-parses live tool calls; no parser merely means the router strips tools, so the derivation refuses rather than guesses.
scripts/card-hints-report.py  -- `make card-hints`. READ-ONLY predicted-vs-curated-vs-probed report with evidence strings and a disagreement list. No GPU, launches nothing.
scripts/card-hints-fetch.py   -- `make card-hints-fetch`. Stages METADATA ONLY (chat template + tokenizer/generation config; a few hundred KB per repo, no weights) for curated families under `~/.cache/devai/card-hints/`, so the rules can be validated out-of-sample. NOT under /var/cache/devai -- per the mount-point convention a new top-level dir there is not volume-backed.
scripts/_probe_core.py        -- Backend-agnostic probe helpers (cache I/O, classifier, implied-spill propagation)
scripts/_probe_hf_common.py   -- Shared scaffold for vLLM/SGLang probers (BackendSpec, podman driver, single-launch + 3-chat probe, nvidia-smi). `build_argparser` carries `--load`/`--needle-depth` flags consumed by the load probe.
scripts/_probe_load.py        -- Serving-time LOAD probe (vLLM/SGLang). Reuses _probe_hf_common's container driver + BackendSpec.build_args + recovery flags and bench.VramSampler. `run_load_probe_pass` walks fitting cells ascending, relaunches each at its ctx, fills a haystack prompt to ctx-2048 tokens from a public-domain corpus with a recall needle, samples peak VRAM at 0.1s, classifies OOM, augments the cell, stops at first OOM. Driven by `probe-{vllm,sglang}-reasoning.py --load`. The corpus (Moby-Dick + War and Peace, ~1M tokens) is fetched from Project Gutenberg (#2701/#2600, boilerplate-stripped) on first `--load` run into `~/.cache/devai/probe-corpus/` (override `DEVAI_PROBE_CORPUS_DIR`; pre-populate for air-gapped hosts) -- NOT vendored in git. Also stamps each HF cache entry's `position_limit` (config.json max_position_embeddings, rope-extended) so the fit probe stops over-promising a context the model asserts on -- via `_probe_hf_common.effective_position_limit`.
scripts/verify-backend-flags.py -- Asserts `--help` of pinned vLLM/SGLang images exposes every flag in deploy/backend-flags.yaml
scripts/probe-check.py        -- Read-only image-drift report (`make probe-check`): compares each HF cache's `_meta.current_image_digest` against the locally available vLLM/SGLang image (`_probe_core.image_digest_via_cli`); exit 1 on drift. Operator companion to the router's boot-time drift check (Phase C).
scripts/probe-ollama-reasoning.py -- Ollama prober (Make-orchestrated VRAM bands)
scripts/probe-vllm-reasoning.py   -- vLLM prober (BackendSpec wrapper)
scripts/probe-sglang-reasoning.py -- SGLang prober (BackendSpec wrapper)
scripts/select-models.py      -- Print fitting models / pull missing best-fit candidates (gguf path emits FROM + RENDERER + PARSER Modelfile, runs ollama create). **KV cache is costed per backend, not fp16 everywhere:** `--kv-dtype` defaults to the `per-backend` sentinel (`KV_DTYPE_PER_BACKEND`), which `resolve_kv_dtype` maps to fp8 for vLLM/SGLang rows and fp16 for ollama-only rows -- matching what the router actually launches (an unstamped probe cell decodes to fp8, and nearly every HF cell on this host is unstamped). An explicit `--kv-dtype` still overrides. `KV_BYTES` covers fp16/bf16/auto=2.0, fp8/int8/q8_0=1.0, q4_0=0.5. Measured effect: 1-3 more models classified as fitting per context tier -- the old 2x KV overcount silently shrank the download-candidate set, so models that fit at their served dtype were never downloaded and therefore never probed. Reconciles too_big/too_small into the exclusion ledger on `--download` (Phase 3 of model-lifecycle-ledger). HF downloads land in the vLLM store by default; `--hf-store {vllm,sglang}` is the explicit opt-in for the SGLang volume. Reports SGLang store gaps -- a model with a fits=true SGLang probe cell but no weights under SGLANG_MODELS_DIR -- on stderr, naming the fix instead of leaving a silent picker advertisement. Fatal (exit 1) on an ENUMERATING `--download` run, and on `--name X --download` only when X is itself a gap row; every other run warns and proceeds. Bypass with `--ignore-store-gaps` (or `$IGNORE_STORE_GAPS`).
scripts/_model_status.py      -- Host-local model exclusion ledger (deploy/.model-status.json, gitignored). Records too_big/too_small/unsupported_arch/oom/manual verdicts keyed by catalog name+backend (sha-stable, unlike the repo@sha probe key) so unfit models aren't re-downloaded/probed/listed. is_excluded honors stability rules (decision 2): unsupported_arch/manual are vram+sha-independent; too_big/too_small re-derive on a VRAM change; oom re-checks on a new sha. Fails open; `save_ledger` writes atomically (tmp + os.replace) like the probe caches. `make model-status`; CLEAR=<name[::backend]> clears. Written by the probers (unsupported_arch) + select-models (too_big/too_small), and pruned of catalog-absent rows by model-sync. See docs/plans/model-lifecycle-ledger.md + docs/backends.md.
scripts/bench-sync.py         -- Closed-loop bench planner/driver (`make bench-plan`, `make bench-sync`). plan_bench() classifies every target from bench_runner.discover_models() into new/incomplete/stale_env/stale_image/dropped/excluded/current; execute() benches the first four grouped by backend and re-renders the leaderboard. Does NOT re-derive the target set (discover_models already diffs the probe cache, honours serving_ok and checks weights are on disk) and does NOT touch the exclusion ledger without --record-drops. Resumability is inherited from update_row's pure merge, not reimplemented. Bench verdicts go through _model_status.record_bench_verdict/is_bench_excluded, never is_excluded -- a model dropped for a leak is still downloadable and probeable.
scripts/model-sync.py         -- Closed-loop onboarder (`make model-sync`). plan_sync diffs the catalog against probe-caches + ledger into new/evaluated/excluded (a row is excluded only if ALL its backends are); execute() sequences model-pull -> cache-down -> probe-vllm/sglang -> probe (ollama) for the genuinely-new rows under SYNC_MAX_DOWNLOADS, with `make cache-up` guaranteed by a try/finally around the GPU-exclusive phase so a failing probe no longer leaves the stack offline. main() first prunes the exclusion ledger of catalog-absent models (non-dry-run only, against the unfiltered catalog, no-op on an empty catalog and refused if it would drop >50% of the ledger). `--dry-run` changes nothing. Note it pulls via `make model-pull`, which lands HF weights in the vLLM store only -- SGLang onboarding stays a manual `--hf-store sglang` step. That unattended pull path now surfaces select-models' SGLang store-gap banner (it previously did not, the check sat behind the `--name` short-circuit); it cannot wedge the loop, because gap rows are always classified `evaluated`, never `new`, so model-sync never requests one by name. Phase 4 of model-lifecycle-ledger.
scripts/model-picker.py       -- Two-step interactive picker (model -> agent); emits `<name>@<ctx>` for vLLM/SGLang (drives container-launch flag) and just `<name>` for Ollama (KV is dynamic; per-session ctx via setNumCtx on /api/chat only). `_KV_BYTES_FP16` became `_KV_BYTES_HF = 1` for the same reason as select-models' per-backend dtype: it is used ONLY on the vLLM/SGLang formula-fallback path (the `*` VRAM cells), and those engines are launched at fp8 KV, so costing the estimate at fp16 inflated it by the whole KV term. Ollama rows never reach that code. See "Model picker" section above for column layout, sort modes, and bindings.
scripts/aiagent-launcher.sh   -- Installed as /usr/local/bin/aiagent-shell; the picker's "AIAgent (shell)" agent. Configures AIAGENT_API_BASE (=<router>/v1) + AIAGENT_MODEL + AIAGENT_CONTEXT and CUDA visibility (DEVAI_AIAGENT_GPU router-only|share), prints a hint, then execs interactive bash so the user runs `aiagent` herself. DEVAI_AIAGENT_SHELL_DEBUG=1 prints resolved env and exits (test hook). See docs/aiagent.md.
scripts/bench/_bench_core.py  -- Bench harness shared helpers: cache I/O + streaming HTTP + token reconciliation, `capture_host_env` / `host_env_id` / `stamp_host_env`, `is_row_key` (skips `_meta` when iterating). `update_row` is a pure merge -- there is no row-reset helper, so `--force` only overwrites the tasks that actually re-ran, and `host_env_id` is stamped per task as well as per row.
scripts/bench/bench_runner.py -- Bench harness driver. Iterates probe-cache fitting models, runs gsm8k / humaneval / tools_use / leak / longctx tasks, captures host_env once per run, stamps every row, resets stale fields when `--force` is set.
scripts/bench/bench_report.py -- Read-only Markdown leaderboard renderer; prints host-env header + Env column joining each row to `_meta.host_env_history`.
bin/devai-agent               -- Host launcher. See "Model picker" section above.
deploy/setup-logs-volume.sh   -- Idempotent LVM/XFS/fstab setup for /var/cache/devai/logs (called by `make setup-logs`). Three refusals, all evaluated BEFORE the first destructive or persistent action (before lvcreate/mkfs/umount/lvremove and before /etc/fstab is rewritten): (1) a non-empty plain directory at the target aborts unless `WIPE=1`; (2) a target mounted from ANOTHER device aborts on the normal path -- `WIPE=1` does not override it, unmount it yourself or ask explicitly with `RECREATE=1 WIPE=1`; (3) `RECREATE=1` while `${VG}/${LV}` merely EXISTS requires `WIPE=1`, because an unmounted LV cannot be inspected from here and is therefore always assumed to hold data. `make setup-logs` forwards RECREATE but NOT WIPE, so `RECREATE=1 make setup-logs` now refuses whenever the LV exists -- rebuild by invoking the script directly (`sudo RECREATE=1 WIPE=1 ... deploy/setup-logs-volume.sh`).
deploy/logging.sh             -- Logger sidecar entrypoint (runs `podman --remote logs --follow` per devai-* container)
tests/test-router.sh          -- Ollama-side router integration tests
tests/test-model-matrix.sh    -- Exhaustive matrix: every probed digest x wire x scenario
tests/test-mcp.sh             -- MCP gateway end-to-end: real handshake (initialize -> notifications/initialized -> tools/list) against http://127.0.0.1:${MCP_PORT}/mcp with the gateway's bearer token, a tool-count floor (MCP_MIN_TOOLS, default 40, well under the 134 observed), assertion of the three first-party tools by name, and a real tools/call. Skips with exit 77 ONLY when the gateway container is absent; a reachable gateway exposing no tools FAILS. Both negative controls were verified (impossible floor -> exit 1, absent container -> exit 77). Replaces a version that probed a /health endpoint the gateway does not serve and therefore skipped unconditionally while the gateway exposed zero tools.
tests/test-mcp-modelstatus.sh -- devai-model-status end-to-end against the live gateway (builds the image, asserts its 3 tools appear in tools/list, calls get_router_status for real). Skips with exit 77 when the gateway/image isn't available. The stronger stdio-protocol-level test (real MCP client, no gateway needed) is devai-tools/cmd/devai-mcp-modelstatus/e2e_test.go, run via `make test-devai-tools`.
tests/test-backup-restore.sh  -- devai-backup end-to-end: snapshot -> list -> verify -> delete originals -> restore --yes -> diff, against temp dirs standing in for deploy/, ~/.devai/, ~/.config/sops/age/ (--repo-root/--home-dir overrides; never touches the real $HOME).
tests/test-gpu-vendor.sh      -- Flips DEVAI_GPU_VENDOR both directions, asserts the rendered `compose config` shows the right device string + backend image tags each time (and that the other vendor's values are absent).
tests/fixtures/modelstatus/   -- Hand-crafted models.yaml + probe-cache + bench-cache fixtures matching the real schemas, shared by the Go unit tests and tests/test-mcp-modelstatus.sh.
tests/python/                 -- Python stdlib-unittest cases (612 collected as of 2026-07-27) covering bench v3 schema migration + runner ctx flags + picker keying + report rendering, sops/age scaffold script gates, MCP gateway catalog/compose/Makefile shape, SkyPilot agent-skill Dockerfile + fetch-cli + docs, catalog-discover lineage/version parsing + structural line filter + VRAM-band filter (under/oversized) + base-model filter + real-size fetch + discover-block overrides + comment-preserving YAML add, model-lifecycle probe-failure classifier + sha-stable carry-forward/orphan-prune + exclusion-ledger stability rules + model-sync diff, network-stubbed end-to-end. Run via `make test-python`.
docs/backends.md              -- Lifecycle, probing, cache hygiene, failure-mode taxonomy across all 3 backends
docs/secrets.md               -- Source of truth for the sops/age scaffold (one-time setup, edit, render, rotation, recovery, multi-host onboarding, paranoid-mode pointer). Only live consumer is the MCP gateway.
docs/mcp.md                   -- Operator reference for the MCP gateway (catalog split, client configs, security model, troubleshooting).
docs/skypilot-user-guide.md   -- User-facing `sky` CLI guide (per-cloud credential setup, hello-world, agent-driven flow). The CLI stays; the system-side provisioner it used to pair with is frozen.
attic/README.md               -- Index of frozen work + why cluster-mode/SkyPilot-fleet were frozen. attic/cluster-mode/RESTORE.md carries the thaw procedure and the defects open at freeze time. The 5 frozen docs live under attic/cluster-mode/docs/.
docs/backup-restore.md        -- devai-backup reference: what's backed up/excluded, the age-key recovery warning, command reference, restore semantics.
docs/security-ci.md           -- GitHub Actions security-CI reference: blocking vs advisory checks, branch protection, local pre-push commands.
docs/mcp-model-status.md      -- devai-model-status MCP server reference: the 3 tools, image/catalog registration, verification.
docs/gpu-vendors.md           -- GPU-vendor overlay reference (NVIDIA default, AMD/ROCm opt-in): the switch, every call site touched, verification status.
docs/plans/                   -- 14 design plans + execution-order README (README.md holds the canonical status table; 3 plans are Frozen). See "Documentation" section above for the snapshot.
```

## Documentation conventions

**Markdown documents must use ASCII characters only.** Non-ASCII
characters (anything above U+007F) do not always render properly
across editors, terminal pagers, GitHub viewers, and downstream
tooling -- some drop them, some mojibake, some treat them as zero-
width. Use ASCII equivalents:

- em-dash (U+2014)             -> `--` or ` -- ` (two hyphens)
- en-dash (U+2013)             -> `-`
- right-arrow (U+2192)         -> `->`
- left-arrow (U+2190)          -> `<-`
- bullet (U+2022)              -> `-`
- multiplication sign (U+00D7) -> `x`
- ellipsis (U+2026)            -> `...`
- smart quotes ('' "")         -> straight quotes (`'`, `"`)
- non-breaking space (U+00A0)  -> regular space

This applies to `.md` files. Source code (`.py`, `.go`, `.sh`, etc.)
is exempt: it is not affected by markdown renderers, and string
literals can legitimately need any Unicode codepoint (e.g. test
fixtures, byte-level BPE markers). Inside markdown, when discussing
such codepoints, refer to them by hex name (`U+0120`) or by
Python-style escape (`\u0120`) rather than pasting the glyph itself.

## Use the codebase knowledge base before/while editing

**If a top-level `.understand-anything/` folder exists, consult it before making code changes.** It is an [Understand-Anything](https://github.com/Egonex-AI/Understand-Anything) knowledge graph of this repo -- `knowledge-graph.json` (nodes/edges for files, functions, classes; 7 architectural layers; a 13-step guided tour), `meta.json` (commit it was built at), and `fingerprints.json`. It is generated from the whole repo -- the Go GPU-arbiter (`gpu-arbiter/`), the Python probe/bench/picker tooling (`scripts/`), the JupyterLab extension (`packages/`), the deploy/ansible infrastructure, and the `docs/` knowledge base. Only generated artifacts are excluded by `.understand-anything/.understandignore` (the `graphify-out/` graph, the runtime probe/bench caches, the compiled `gpu-arbiter/gpu-arbiter` binary, local agent state, and `.understand-anything/` itself); tests, docs, and scripts are included. Treat it as a navigation aid, not ground truth -- always confirm against the actual source, and note it may be stale if `meta.json`'s `gitCommitHash` predates current `HEAD`.

Recommended commands (all from the `understand-anything` plugin; if the graph is missing, build it first with `/understand-anything:understand`):

| Need | Command | What it does |
|------|---------|--------------|
| Explain a file/function/module before touching it | `/understand-anything:understand-explain <path-or-symbol>` | Deep-dive explanation of a specific file, function, or module |
| Understand impact/risk of current changes | `/understand-anything:understand-diff` | Analyzes the git diff / PR: what changed, affected components, blast radius, risks |
| Ask free-form questions about the code | `/understand-anything:understand-chat` | Q&A over the codebase grounded in the knowledge graph |
| Extract / explore domain knowledge | `/understand-anything:understand-domain` | Builds an interactive business-domain flow graph (derives from the existing graph when present) |
| Visualize the graph | `/understand-anything:understand-dashboard` | Launches the interactive web dashboard |
| Refresh the graph after commits | `/understand-anything:understand` (add `--full` to rebuild) | Incremental update of changed files; re-baselines `meta.json`/`fingerprints.json` |

Typical loop for a change: `understand-explain` the area you're about to edit -> make the edit -> `understand-diff` to sanity-check impact and affected components -> refresh with `/understand-anything:understand` if the change was structural.

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
- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Don't Assume -- If You Don't Know, Say So

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
- Did I run it end-to-end? -> can claim "works"
- Did I only check it parses / imports / starts? -> say "starts cleanly, full round-trip not tested"
- Did I infer from documentation, help text, or another agent's behavior? -> say "I'm assuming based on X, not verified"

A "PASS" in a test harness means PASS only for what the harness actually checked. If the harness checks "non-empty output", report that, not "works".

When the user asks "does X work?", the only honest answers are:
- "Yes -- verified by [specific test/command/output]"
- "No -- fails at [specific point], log/error attached"
- "I don't know -- haven't tested"

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, fewer false-confidence claims that have to be retracted later, and clarifying questions come before implementation rather than after mistakes.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- ALWAYS read graphify-out/GRAPH_REPORT.md before reading any source files, running grep/glob searches, or answering codebase questions. The graph is your primary map of the codebase.
- IF graphify-out/wiki/index.md EXISTS, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep -- these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
