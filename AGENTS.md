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

Single-host is the only supported topology. Multi-host cluster mode
(`--mode=worker|head`) and the SkyPilot fleet provisioner were **frozen on
2026-07-25** and moved to `attic/cluster-mode/` -- not deleted, and intended
to return. `gpu-arbiter` still accepts `--mode`, but any value other than
`single` exits with a pointer to `attic/README.md`. Read that README before
touching anything cluster- or fleet-shaped.

Optional opt-in surfaces (gated behind compose profiles, default behaviour
unchanged):

- **MCP gateway** (profile=`mcp`) -- Docker MCP Gateway peer service on port
  8088, endpoint path `/mcp`, bearer token minted at startup. Third-party
  servers come from Docker's official catalog; `deploy/mcp-catalog-devai.yaml`
  adds the one server this repo authors. 15 servers enabled, 134 tools
  verified live on 2026-07-25. See `docs/mcp.md`.
- **sops + age secret store** -- shared encrypted-at-rest scaffold. Two of its
  three intended consumers are now frozen, so the only remaining one is the
  MCP gateway's `github-official` / `firecrawl` secrets, and that path is
  unverified. Operators run `make age-keygen-host` once and append their
  public key to `.sops.yaml`. See `docs/secrets.md`.

The lab image bundles the `sky` CLI and `/usr/local/bin/sky-setup.sh` for the
user-facing cloud flow (`docs/skypilot-user-guide.md`). That is a separate
thing from the frozen system-side provisioner and it stays.

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
- `attic/`: frozen work, kept deliberately and excluded from every build.
  `attic/cluster-mode/` holds the cluster-mode and SkyPilot-fleet sources,
  infra, tests and docs; its Go files carry `//go:build devai_frozen_cluster`
  and `attic/` sits outside every Go module, so `go build ./...` cannot see
  them. Do not add new code here, do not wire anything to it, and read
  `attic/cluster-mode/RESTORE.md` before proposing a thaw.

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

The router also refuses model-store mutations. `/api/pull`, `/api/create`,
`/api/push`, `/api/copy` and `/api/delete` return 403 on every backend
listener, for every HTTP method, before the catch-all proxy sees them
(`ollamaMutationPaths` / `makeMutationGuard` / `newBackendMux` in
`gpu-arbiter/main.go`). Without that guard anything in the lab could pull an
unprobed model and then serve it, bypassing the probe-cache fit gate. The
operator pipeline does not traverse the router -- `select-models.py` uses
`podman exec devai-ollama ollama pull` -- so `make model-pull` followed by
`make probe` remains the sanctioned path.

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
- `make test-python`: stdlib-unittest cases (612 collected as of
  2026-07-27) covering bench v3 schema migration + runner ctx flags
  + picker keying + report rendering, sops/age scaffold script
  gates, MCP gateway catalog/compose/Makefile shape, SkyPilot
  agent-skill, catalog-discover, and the model-lifecycle ledger.

MCP / secrets targets (opt-in):

- `make mcp-up` / `make mcp-down` / `make mcp-logs` / `make mcp-test` /
  `make mcp-secrets-render`: MCP gateway (profile=`mcp`).
  `make mcp-test` does a real handshake, asserts a tool-count floor,
  and invokes a tool; it skips (77) only when the gateway container
  is absent. `make mcp-health` is liveness only -- do not read it as
  a functional check.
- `make build-mcp-modelstatus-image` / `make test-mcp-modelstatus`:
  the one MCP server this repo authors. The gateway runs servers with
  `--pull never`, so build the image before starting the gateway.
- `make age-keygen-host` / `make secrets-tmpfs` /
  `make secrets-edit SOPS_FILE=...` / `make secrets-rotate`:
  shared sops/age scaffold.

The cluster-mode and SkyPilot targets are gone. All 9 of them
(`cluster-head-up`, `cluster-head-down`, `cluster-status`,
`skypilot-up`, `skypilot-down`, `skypilot-check`,
`skypilot-secrets-render`, `build-worker-bootstrap`,
`test-cluster-preflight`) are preserved verbatim in
`attic/cluster-mode/Makefile.frozen-targets`.

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

Its KV-cache cost model is **per backend**, not fp16 everywhere.
`--kv-dtype` defaults to the sentinel `per-backend`, which
`resolve_kv_dtype` maps to fp8 for vLLM/SGLang rows and fp16 for
ollama-only rows -- the dtypes the router actually launches with. An
explicit `--kv-dtype` still overrides. Costing every row at fp16
double-counted the KV term for HF models and silently shrank the
download-candidate set, so models that fit at their served dtype were
never downloaded and therefore never probed. Measured effect of the
fix: 1-3 more models classified as fitting per context tier.
`scripts/model-picker.py` carries the same correction as
`_KV_BYTES_HF = 1` (was `_KV_BYTES_FP16`), used only on its
vLLM/SGLang formula-fallback path; Ollama rows never reach it.

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

MCP env knobs:

- `MCP_PORT`: gateway host port, default 8088, bound to 127.0.0.1.
  The endpoint is `http://127.0.0.1:${MCP_PORT}/mcp` -- the `/mcp`
  path is required and a bearer token is mandatory.
- `MCP_SECRETS_FILE`: host path of the rendered secrets file, mounted
  at the fixed container path `/secrets/.env`; default `/dev/null`.

`DEVAI_MODE` and the whole `DEVAI_HEAD_*` / `DEVAI_WORKER_*` /
`SKYPILOT_*` env surface went with the freeze. Their contract table is
in `attic/cluster-mode/docs/cluster-env.md`.

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
