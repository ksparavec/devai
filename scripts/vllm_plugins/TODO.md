# `deepseek_string` plugin — wiring DONE + VERIFIED

Status: **wired, unit-tested, and verified end-to-end on both R1
distill variants** as of 2026-05-01.

- `DeepSeek-R1-Distill-Qwen-7B`  — Qwen-2 tokenizer  → `T=deepseek_string mode=forced` ✓
- `DeepSeek-R1-Distill-Llama-8B` — Llama-3 tokenizer → `T=deepseek_string mode=forced` ✓

Both verified through the live router at `http://devai-router:11435`:
single-tool `tool_choice: "auto"` requests get promoted by the router
to a specific-function pin, vLLM loads the plugin via
`--tool-parser-plugin /etc/devai/vllm-plugins/deepseek_string_tool_parser.py`,
and the plugin extracts a clean `tool_calls: [get_time(...)]` from
the model's full-width DeepSeek-V3 marker output. Multi-tool auto
requests get rejected with HTTP 400 + `tool_choice_pinning_required`.
See `docs/backends.md` "Operational notes — R1-Distill family" for
the per-model behavioural difference (5-token vs 525-token call latency).

## What's wired

### Plugin registry — single source of truth
- `deploy/vllm-plugins.json` — JSON map of `parser_name → {kind, file}`.
  Both Python (probe driver) and Go (router) read this file; adding a
  new plugin is one JSON entry plus a file in `scripts/vllm_plugins/`.
- Container path: `/etc/devai/vllm-plugins/<file>`. The host directory
  is bind-mounted into the recreated vLLM container at recreate time.

### Probe driver (`scripts/_probe_hf_common.py` + `probe-vllm-reasoning.py`)
- `BackendSpec` gained `supports_plugins` (vLLM=True, SGLang=False).
- `_resolve_plugins` looks up parser names against the registry; when
  matched it adds the host→container plugin volume and threads the
  in-container plugin path into `build_args`.
- `vllm_command_args` emits `--tool-parser-plugin <abs>` (or
  `--reasoning-parser-plugin <abs>`) immediately before the parser-name
  flag; vLLM resolves parser names at flag-parse time so the plugin
  module has to be loaded by then.
- Built-in parsers pass through unchanged — no plugin flag, no mount,
  no behaviour change vs. pre-plugin builds.
- SGLang accepts the plugin kwargs and drops them (it has no
  `--*-parser-plugin` analogue).

### Router (`gpu-arbiter/main.go` + `gpu-arbiter/vllm_plugins.go`)
- Reads `VLLM_PLUGINS_REGISTRY` (default `/etc/devai/vllm-plugins.json`,
  mounted by compose) and `VLLM_PLUGINS_HOST_DIR` (set by the Makefile
  to the host's `scripts/vllm_plugins` path) at startup.
- `arbiter.resolvePluginLaunch` is called from `containerRecreate`. It
  populates `launchConfig.{Tool,Reasoning}ParserPlugin` and emits the
  libpod bind-mount spec when at least one plugin is required.
- Empty `VLLM_PLUGINS_HOST_DIR` + a parser that needs a plugin → the
  recreate fails loudly with an actionable error (rather than silently
  launching without the plugin file accessible).
- Kind mismatch (e.g. a `kind=reasoning` entry used as a tool parser)
  is rejected with a clear error.
- 14 new unit tests in `gpu-arbiter/vllm_plugins_test.go` cover loader
  tolerance, lookup, ordering, host-dir gating, and SGLang ignoring
  plugin paths.

### Family entry (`scripts/model-families.yaml`)
- `deepseek-r1-distill` family now has `parsers.vllm.tool: deepseek_string`.
- SGLang side intentionally left empty — see top-level TODO for the
  optional follow-up.

### Compose & Makefile
- `deploy/docker-compose.yaml`: router service mounts `vllm-plugins.json`
  at `/etc/devai/vllm-plugins.json`, env passes `VLLM_PLUGINS_HOST_DIR`.
- `Makefile`: `VLLM_PLUGINS_HOST_DIR = $(abspath scripts/vllm_plugins)`,
  exported so compose interpolation picks it up.

### Docs
- `docs/backends.md` gained a "Custom vLLM parser plugins" section
  describing the registry, the wiring pattern, and how to add a new
  plugin.

## What's NOT done (deliberate)

- **Plugin source baked into an image.** Skipped: vLLM runs in the
  upstream `vllm/vllm-openai:v0.22.1-x86_64-cu129-ubuntu2404` image which we
  don't fork, and our lab/router images don't run vLLM. Bind-mount is
  the only wiring path; the bake step from the original plan was
  misaligned with how the stack is actually deployed.
- **SGLang plugin equivalent.** Different plugin model (Python import,
  not file path). Out of scope until SGLang's NVFP4 path is unbroken.

## Adding a new plugin later

1. Drop the parser file in `scripts/vllm_plugins/`.
2. Add an entry to `deploy/vllm-plugins.json`:
   ```json
   "<parser_name>": {"kind": "tool", "file": "<basename>.py"}
   ```
3. Reference `<parser_name>` from a family's `parsers.vllm.tool` (or
   `parsers.vllm.reasoning`) in `scripts/model-families.yaml`.
4. `make catalog-regen && make cache-down`.
5. `python3 scripts/probe-vllm-reasoning.py --repo "<regex>" --force`
   — expect `T=<parser_name> dis=y` in the output.
6. `make cache-up` and confirm a live tool-using chat through the router.
