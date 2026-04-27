# Repository Guidelines

## Project Overview

DevAI is a containerized local AI development environment. It provides
JupyterLab, multiple AI CLIs, an interactive model picker, Open WebUI, and a
GPU-aware inference router. The current active inference backend is Ollama for
GGUF models. vLLM and SGLang lifecycle code is still compiled into the router
and covered by router tests, but their compose services are dormant behind the
`backends-disabled` profile while Ollama behavior is stabilized. See
`docs/sidelined-backends.md` before changing vLLM or SGLang behavior.

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
- `make model-select`: probe downloaded Ollama models and write
  `deploy/active-models.yaml`.
- `make model-select DOWNLOAD=1`: also pull fitting missing model variants.
- `make model-select FAMILY=qwen3.5 DOWNLOAD=1`: scope selection to one family.
- `make ollama-list`: list downloaded Ollama models.
- `make vllm-list`: list on-disk vLLM weights; backend is currently dormant.
- `make test-router`: run Go router tests in a Go container.
- `make test`: run router tests plus Ollama integration tests.

JupyterLab extension changes must be built inside a container, not directly on
the host. Use the documented container build flow from `CLAUDE.md` or
`README.md` when changing `packages/jupyter-ai-launchers/src/index.ts`.

## Model Catalog and Selection Workflow

`deploy/models.yaml` is the generated full model catalog. It should normally be
updated through `make catalog-regen`, not hand-edited.

`deploy/active-models.yaml` is also generated. It is the active subset consumed
by the router and picker: downloaded models that fit current VRAM/context
constraints and include probe metadata where available. Regenerate it with
`make model-select`.

The Ollama reasoning probe is in `scripts/probe-ollama-reasoning.py`. It records
model capability, disable support, and VRAM coefficients used by
`scripts/select-models.py` and `scripts/model-picker.py`. The picker should
show fitting Ollama models and display reasoning capability as metadata. It
should not own model-specific reasoning activation logic.

Important runtime knobs:

- `GPU_MEMORY_GB`: total GPU VRAM, default `24`.
- `MAX_CONTEXT_LEN`: default context, default `131072`.
- `CONTEXT`: per-run picker/model selection override, for example
  `CONTEXT=32768 make shell-gpu`.
- `VRAM`: per-run VRAM override, for example `VRAM=48 make model-select`.
- `DEVAI_REASONING`: `auto|off|low|medium|high`.

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
`deploy/models.yaml`, `deploy/active-models.yaml`, compose profiles, exposed
ports, or persistent cache behavior.

## Security & Configuration Tips

Do not commit secrets, API tokens, private keys, model weights, cache contents,
or generated logs. `.env` is local configuration; keep `.env.example` generic.
Proxy variables and runtime settings should be documented without embedding
private infrastructure details.

Prefer Make targets over raw container commands. They preserve expected mounts,
network names, user settings, and environment variables. If a workflow truly
needs a direct `podman` or `docker` command, document why and keep it scoped.
