# Repository Guidelines

## Project Overview

DevAI is a containerized local AI development environment. It provides
JupyterLab, multiple AI CLIs, an interactive model picker, Open WebUI, and a
GPU-aware inference router. All three inference backends (Ollama, vLLM, SGLang)
are wired and covered by router tests. vLLM and SGLang start as `sleep infinity`
placeholders so `make cache-up` is hermetic; the router replaces them via libpod
on demand when a request arrives on the backend's port. See
`docs/backends.md` for the lifecycle, probing procedure, and failure-mode
taxonomy before changing backend behavior.

Optional opt-in surfaces (all gated behind compose profiles or the `--mode` flag,
default behaviour unchanged):

- **Cluster mode** (`gpu-arbiter --mode={single,worker,head}`) for multi-host
  fleets. Single is byte-identical to today; worker registers with a head;
  head proxies to whichever worker scores highest for the incoming
  `(model, ctx)`. See `docs/cluster-mode.md`.
- **MCP gateway** (profile=`mcp`) -- Docker MCP Gateway peer service on port
  8088 with 10 Tier 1 + 4 Tier 2 servers. See `docs/mcp.md`.
- **SkyPilot fleet provisioner** (profile=`cluster`) -- long-lived API server
  on port 46580 that head mode calls for cloud-burst provisioning. See
  `docs/skypilot.md`. The lab image also bundles the SkyPilot CLI for the
  user-facing flow (`docs/skypilot-user-guide.md`).
- **sops + age secret store** -- shared encrypted-at-rest scaffold consumed by
  the three above. Operators run `make age-keygen-host` once and append their
  public key to `.sops.yaml`. See `docs/secrets.md`.

The project supports both Podman and Docker. Most workflows are intentionally
driven through `make` targets so contributors do not need to remember container
flags, mounted paths, or runtime environment variables.

## Project Structure & Module Organization

- `gpu-arbiter/`: Go module for the GPU-aware multi-port reverse proxy. Tests
  live beside the source as `*_test.go`.
- `deploy/`: container and runtime assets, including Dockerfiles,
  `docker-compose.yaml`, generated model catalogs, systemd units, and WebUI
  proxy configuration.
- `scripts/`: operational Python and shell tooling. Important scripts include
  `generate-catalog.py`, `probe-ollama-reasoning.py`, `select-models.py`, and
  `model-picker.py`.
- `tests/`: shell-based integration and agent matrix tests.
- `config/`: configuration for bundled AI CLIs and agent providers.
- `packages/jupyter-ai-launchers/`: JupyterLab launcher extension.
- `docs/`: design notes and operational documentation.
- `ansible/`: host provisioning resources.

Do not put generated caches, local logs, model weights, or temporary test output
into the repository. The model cache lives under `/var/cache/devai/` outside the
repo.

## Architecture Notes

The inference stack is fronted by `devai-router`, implemented in
`gpu-arbiter/main.go`. The router exposes one port per backend:

- `11434`: Ollama, active
- `11435`: vLLM, dormant
- `11436`: SGLang, dormant

The router manages backend lifecycle, GPU exclusion, idle shutdown, drain
behavior, and request forwarding. Request path and port determine protocol and
backend. Reasoning activation for Ollama should be handled by the router using
protocol fields, not by prompt hacks in agents. The current reasoning semantics
are documented in `docs/ollama_models.md`.

## Build, Test, and Development Commands

Use `make help` to list supported targets. Common commands:

- `make build-cpu`: build the CPU lab image.
- `make build-gpu`: build the GPU/CUDA lab image.
- `make build-router`: build the router image.
- `make build`: build CPU, GPU, and router images.
- `make cache-up`: start active infrastructure services.
- `make cache-down`: stop infrastructure services.
- `make cache-status`: show service and model-cache status.
- `make lab-gpu`: run JupyterLab with GPU support.
- `make lab-cpu`: run JupyterLab without GPU support.
- `make shell-gpu`: start the GPU shell and open the model picker.
- `make shell-cpu`: start the CPU shell.
- `make catalog-regen`: refresh `deploy/models.yaml` from the upstream HF
  and Ollama registries (input is `scripts/model-families.yaml`).
- `make probe`: probe every downloaded Ollama digest at every
  `(VRAM, CONTEXT)` cell and write the result to
  `deploy/.ollama-reasoning-cache.json`.
- `make model-pull`: download missing best-fit catalog candidates. Add
  `FAMILY=qwen3.5` to scope to one family, `CONTEXT=32768` to bias the
  selection toward smaller-context fits, `DRY_RUN=1` to preview.
- `make model-fit`: print which models fit at the chosen `(VRAM,
  CONTEXT)`. Read-only diagnostic -- never writes a file.
- `make ollama-list`: list downloaded Ollama models.
- `make vllm-list`: list on-disk vLLM weights; backend is currently dormant.
- `make logs SERVICE=devai-ollama`: tail a service's persisted stdout
  from the logger sidecar; `LINES=N` seeds the buffer.
- `make setup-logs`: one-time, sudo. Carve out a dedicated 100 GB
  thin LV in `vgais/cachepool` for `/var/cache/devai/logs` so log
  growth never crowds the model cache. `RECREATE=1` re-creates an
  existing volume -- it destroys the LV and the mountpoint tree, so it
  aborts on a non-empty target unless `WIPE=1` is set too (and
  `make setup-logs` does not forward `WIPE`; call the script directly).
- `make test-router`: Go router unit tests in a Go container (~2s).
- `make test-ollama`: Ollama integration tests via the live router.
- `make test-models`: matrix test over every probed digest x wire
  protocol (`/api/chat`, `/v1/chat/completions`, `/v1/messages`) x
  scenario (basic, tools, reasoning auto/off, ctx). Reads the probe
  cache to enumerate cells (slow -- minutes-to-tens-of-minutes).
- `make test-vllm` / `make test-sglang`: live HF-backend integration
  tests through the router (cold-start chat, ctx switch, model switch
  when >=2 models cached, GPU exclusion, parameter forwarding).
  Skip cleanly when no fitting model is in the corresponding cache.
- `make test-e2e`: bridges picker discovery, agent-command construction,
  and a live router round-trip -- proves the full picker->agent->router
  chain serves a real chat completion.
- `make test-probe-vllm` / `make test-probe-sglang`: single-cell probe
  smoke tests with cache-schema assertion (require `cache-down` first).
- `make test-probe-ollama-idempotent`: byte-identical regression check
  on the refactored Ollama prober.
- `make test`: runs **every** available test in sequence -- Go unit +
  Python unit + Ollama integration + matrix + vLLM/SGLang integration +
  E2E + probe smoke. Wall time 30-60+ min. Each layer skips cleanly
  when its prerequisites aren't met.
- `make test-python`: stdlib-unittest cases (138 tests as of
  2026-05-15) covering bench v3 schema migration + runner ctx flags
  + picker keying + report rendering, sops/age scaffold script
  gates, MCP gateway Phase 1+2 catalog/compose/Makefile shape,
  SkyPilot agent-skill, SkyPilot fleet Phase 1 service shape, and
  the cluster-mode Phase 1.5 stub head's HTTP surface.
- `make test-cluster-preflight`: runs the 7-scenario cluster-mode
  Phase 1.5 preflight against a real arbiter binary + stub head
  (no GPU; ~55s wall time). Gates every PR touching `gpu-arbiter/`
  cluster files.

Cluster / MCP / SkyPilot / secrets targets (opt-in):

- `make cluster-head-up` / `make cluster-head-down` /
  `make cluster-status`: head-mode router lifecycle.
- `make build-worker-bootstrap`: build the minimal cloud-VM image
  (arbiter binary + cloud-init only). Requires
  `gpu-arbiter/gpu-arbiter` to exist; build via `make build-router`
  first when running outside the distroless image flow.
- `make mcp-up` / `make mcp-down` / `make mcp-test` /
  `make mcp-secrets-render`: MCP gateway (profile=`mcp`).
- `make skypilot-up` / `make skypilot-down` /
  `make skypilot-check` / `make skypilot-secrets-render`: SkyPilot
  API server (profile=`cluster`).
- `make age-keygen-host` / `make secrets-tmpfs` /
  `make secrets-edit SOPS_FILE=...` / `make secrets-rotate`:
  shared sops/age scaffold.

JupyterLab extension changes must be built inside a container, not directly on
the host. Use the documented container build flow from `CLAUDE.md` or
`README.md` when changing `packages/jupyter-ai-launchers/src/index.ts`.

## Model Catalog and Selection Workflow

`scripts/model-families.yaml` is the only hand-edited input. Each family
declares `ollama_repos`, `hf_repos`, and/or `gguf_repos`, plus an
`arch_ref` HF repo for architecture metadata. Backend is implicit:
`ollama_repos` and `gguf_repos` rows run on Ollama; `hf_repos` rows run
on vLLM/SGLang (currently dormant).

`deploy/models.yaml` is the generated full model catalog. It is
refreshed by `make catalog-regen`, not hand-edited.

`deploy/.ollama-reasoning-cache.json` (schema v3) is the single source
of truth for what fits. It is digest-keyed; each entry carries
`aliases`, `max_context`, top-level `capability`, optional
`disable_verified`, and a 2-D `probes` map nested as
`probes[<vram_gb>][<ctx>]`. Each probe cell records `vram_gb`, `ctx`,
`actual_total_gb`, `actual_vram_gb`, `fully_on_gpu`, per-cell
capability, and timestamp. There is no `deploy/active-models.yaml`
any more -- the router and picker read this cache directly.

The Ollama reasoning probe is in `scripts/probe-ollama-reasoning.py`.
It loops the `(VRAM, CONTEXT)` matrix from
`PROBE_VRAMS=16G,24G` x `PROBE_CONTEXTS=32K,64K,128K,256K`. Between
VRAM bands the orchestrator recreates `devai-ollama` with
`OLLAMA_GPU_OVERHEAD=(host_vram - target_vram) * 1024^3` so the daemon
behaves as if it had only the target card -- a 24 GB host can produce
cache cells valid for 16 GB targets without hardware swaps. Probing is
incremental and never destructive: existing cells are immutable unless
`PROBE_FORCE=1` (whole VRAM band) or `PROBE_FORCE_CTX=64K` (one tier).

`scripts/select-models.py` is the diagnostic + downloader. It reads
the catalog and the probe cache, prints the fitting set at the chosen
`(VRAM, CONTEXT)`, and (with `--download`) pulls missing best-fit
candidates. `pull_gguf` downloads a `.gguf` blob from HF, writes a
Modelfile that emits `FROM <file>` plus `RENDERER <family>` and
`PARSER <family>` directives, and runs `ollama create` to register
the imported tag. The renderer/parser pair is what makes imported
GGUFs accept tool calls; without them Ollama returns "does not
support tools".

`scripts/model-picker.py` reads the probe cache directly. For each
session it derives a per-session tag `<parent>-ctx<N>` (e.g.
`qwen3.5:9b-q8_0-ctx32768`) using Ollama's `/api/create` so
`PARAMETER num_ctx N` is baked into the Modelfile. This makes the
chosen context binding for every wire protocol -- `/v1/chat/completions`
(Ollama 0.21.x) silently ignores `options.num_ctx`, so per-session
Modelfile overrides are the only universal mechanism. The router
peels the `-ctx<N>` suffix when resolving capability/policy so the
parent's reasoning entry still applies.

Important runtime knobs:

- `GPU_MEMORY_GB`: total GPU VRAM, default `24`.
- `MAX_CONTEXT_LEN`: default per-request context cap in tokens, default `131072`.
- `CONTEXT`: per-run picker/model selection override, for example
  `CONTEXT=32768 make shell-gpu`.
- `VRAM`: per-run VRAM override, for example `VRAM=16 make model-fit`.
- `PROBE_VRAMS`, `PROBE_CONTEXTS`: comma-separated lists driving
  `make probe`. Defaults `16G,24G` and `32K,64K,128K,256K`.
- `PROBE_FORCE`, `PROBE_FORCE_CTX`: re-probe the current VRAM band /
  the named tier, ignoring existing cache cells.
- `DEVAI_REASONING`: `auto|off|low|medium|high`.

Cluster + MCP + SkyPilot env knobs (full table in `docs/cluster-env.md`):

- `DEVAI_MODE`: `single` (default), `worker`, or `head`.
- `DEVAI_HEAD_URL`, `DEVAI_WORKER_TOKEN_FILE`, `DEVAI_WORKER_NAME`,
  `DEVAI_LIFECYCLE`, `DEVAI_GPU_TYPE`, `DEVAI_BACKENDS`,
  `DEVAI_WORKER_INBOUND_PORT`: worker-side configuration.
- `DEVAI_HEAD_LISTEN_PORT`, `DEVAI_HEAD_TOKEN_FILE`,
  `DEVAI_IDLE_MINUTES`, `DEVAI_QUEUE_DEPTH_THRESHOLD`: head-side
  configuration. `DEVAI_HEAD_TOKEN_FILE` (default
  `/run/devai/cluster-token`) is the bearer token the head requires on
  `/v1/cluster/{register,heartbeat,status}` and presents on every
  worker-bound `/v1/cluster/inbound` call.
- `MCP_PORT`, `MCP_SECRETS_FILE`: MCP gateway (Phase 2 mounts the
  secrets file when set; default `/dev/null`).
- `SKYPILOT_API_PORT`, `SKYPILOT_CREDENTIALS_FILE`,
  `SKYPILOT_API_ENDPOINT`: SkyPilot fleet provisioner.
  `SKYPILOT_API_ENDPOINT` is **not read by gpu-arbiter today** --
  `skypilot_client.go` / `skypilot_policy.go` exist but
  `NewSkyPilotClient` has no non-test caller, so setting it has no
  effect and a head routes over the local fleet only. It stays as the
  reserved name for that wiring (see `docs/cluster-env.md`).

## Coding Style & Naming Conventions

Keep changes surgical. Touch only files needed for the task and avoid unrelated
cleanup unless explicitly requested. Match existing style even when a different
style would also be reasonable.

Go code must be formatted with `gofmt`. Keep tests in `gpu-arbiter/*_test.go`
and use clear `Test...` names. Prefer table tests where they simplify repeated
policy cases.

Python scripts use 4-space indentation, type hints where already present, and
clear command-line behavior. Keep scripts executable when they are intended to
run directly. Prefer structured YAML/JSON parsing over ad hoc string handling.

Shell scripts should be explicit about Bash versus POSIX shell, fail clearly,
and avoid hiding command errors unless the failure is expected and documented.
Existing naming patterns include `model-*`, `ollama-*`, `vllm-*`, and
`test-router*.sh`.

## Testing Guidelines

Run focused tests for touched code, then broader tests when feasible. For
router changes, run `go test ./...` inside `gpu-arbiter/` or use
`make test-router`. If the sandbox blocks local sockets or Go cache writes, use
a writable `GOCACHE` or the containerized `make test-router` path.

For Python script edits, at minimum run:

```bash
python3 -m py_compile scripts/<file>.py
```

For YAML changes, parse the changed files with Python or another YAML-aware
tool. Integration tests require running services and a populated model cache;
start with `make cache-up`. `make test` runs the active router and Ollama test
path. vLLM and idle tests are intentionally dormant unless those backends are
reactivated.

When reporting results, state exactly what was verified. Do not claim an
end-to-end workflow works if you only compiled, parsed, or ran a narrow unit
test.

## Commit & Pull Request Guidelines

History mostly uses imperative commit subjects, often with `feat:` or `chore:`.
Going forward, prefer a concise subject plus a commit body that explains the
main behavior changes, generated-file updates, and verification commands. For
example:

```text
feat: normalize Ollama reasoning policy

- route native, OpenAI, and Anthropic requests through protocol-specific fields
- remove agent-side thinking prompt hacks
- update picker filtering and docs

Verification:
- go test ./...
- python3 -m py_compile scripts/model-picker.py
```

Pull requests should include a short summary, linked issues when relevant, test
commands run, and notes about generated files. Call out changes to
`deploy/models.yaml`, `deploy/.ollama-reasoning-cache.json`, compose profiles,
exposed ports, or persistent cache behavior.

## Security & Configuration Tips

Do not commit secrets, API tokens, private keys, model weights, cache contents,
or generated logs. `.env` is local configuration; keep `.env.example` generic.
Proxy variables and runtime settings should be documented without embedding
private infrastructure details.

Encrypted credentials live in `deploy/*.sops.env` (committed) and are rendered
to `/run/devai/*.env` (tmpfs, gitignored) by `make *-secrets-render`. Plaintext
templates are shipped as `*.sops.env.example` so operators see the expected
variable names without decrypting anything. `.gitignore` blocks `*.env.plain`
and `/run/devai`; the `!deploy/*.sops.env` exception ensures encrypted files
stay tracked even under broader rules. Run the sops/age scaffold's pre-commit
checklist (per `docs/secrets.md`) before any commit that touches `.sops.yaml`
or a `.sops.env` file -- it catches the literal `age1xxx...` placeholder.

Prefer Make targets over raw container commands. They preserve expected mounts,
network names, user settings, and environment variables. If a workflow truly
needs a direct `podman` or `docker` command, document why and keep it scoped.
