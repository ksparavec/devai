# `deepseek_string` plugin — wiring TODO

The parser file `deepseek_string_tool_parser.py` in this directory is
**written and smoke-tested** but **not yet wired into the probe driver or
the router**. This file enumerates the remaining work to make it
production-usable. Pick this up in a fresh session.

## Current state — verified working in isolation

- Plugin file: `scripts/vllm_plugins/deepseek_string_tool_parser.py` (~250 lines)
- Imports cleanly inside `vllm/vllm-openai:latest-cu130-ubuntu2404`
- Registers as `deepseek_string` via `ToolParserManager.register_module(["deepseek_string"])`
- Smoke-tested 2026-05-01 with the real `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` tokenizer:
  - `vocab.get('<｜tool▁call▁begin｜>') == None` (the case that breaks upstream `deepseek_v3`)
  - Plugin extracted `ToolCall(type=function, name=get_time, args={})` from a sample DeepSeek-shape output
  - Preceding content (`'Reasoning here.\n'`) preserved correctly

## What's left

### 1. Probe-driver wiring (`scripts/_probe_hf_common.py`)
- [ ] In `container_run_detached()` — add a bind-mount of the plugin
  directory: `--volume <repo>/scripts/vllm_plugins:/plugins:ro`. Only
  needed when the resolved `tool_parser` is a plugin name (currently
  just `deepseek_string`).
- [ ] When the family's `parsers.vllm.tool` is `deepseek_string`, prepend
  `--tool-parser-plugin /plugins/deepseek_string_tool_parser.py` to the
  vLLM engine args. The flag must come BEFORE `--tool-call-parser
  deepseek_string` (vLLM loads plugin files at parser-resolution time).
- [ ] Decide how the prober knows which parser names are plugins vs.
  built-ins. Two options:
    1. Hard-coded set in `_probe_hf_common.py` (simple, explicit)
    2. Family-level field, e.g. `parsers.vllm.tool_plugin: <path>`,
       letting the family declare its own plugin path. More flexible
       but bigger schema change.
  Lean toward option 1 for now — single-element set with a comment.

### 2. Router wiring (`gpu-arbiter/main.go`)
- [ ] Mirror the bind-mount + `--tool-parser-plugin` flag injection at
  container recreate time. The router already injects `--reasoning-parser`
  and `--tool-call-parser` from the cache row; the new flag follows the
  same pattern.
- [ ] Reuse the same plugin-name allowlist source the prober uses (a
  shared text file or YAML, not duplicated logic in two languages). A
  small JSON file at `deploy/vllm-plugins.json` with
  `{"deepseek_string": "/plugins/deepseek_string_tool_parser.py"}` works
  for both Go and Python consumers.
- [ ] Add the `--volume` to the `containerCreate` libpod call. Mode
  `ro,Z` (or whatever the existing volume mounts use).

### 3. Image build (optional but cleanest)
- [ ] In `deploy/Dockerfile.lab` (and possibly `Dockerfile.router` if it
  has Python), `COPY scripts/vllm_plugins /usr/local/share/vllm-plugins`
  so the bind-mount becomes a no-op fallback. Plugins ship inside the
  image; bind-mount is only needed for hot iteration during development.
- [ ] Add a build-time test target: `python3 -c "from
  vllm.utils.import_utils import import_from_path;
  import_from_path('p', '/usr/local/share/vllm-plugins/deepseek_string_tool_parser.py')"`
  to catch import-path drift on every image build.

### 4. Family-entry restoration (`scripts/model-families.yaml`)
- [ ] Restore `parsers.vllm.tool: deepseek_string` to the
  `deepseek-r1-distill` family (replacing the current "tool calls
  intentionally NOT wired" comment block).
- [ ] Keep the comment but rewrite it to point at the plugin: "Tool
  calls go through the `deepseek_string` plugin parser; see
  `scripts/vllm_plugins/deepseek_string_tool_parser.py`. The base
  tokenizer (Qwen2 / Llama3) lacks atomic vocab entries for the
  DeepSeek-V3 boundary markers, which broke every built-in parser."
- [ ] SGLang side stays empty — SGLang's plugin model is different
  (Python registry import, not file-path arg). A separate plugin
  would be needed for SGLang. Defer.

### 5. Verification
- [ ] `make cache-down`
- [ ] Force-probe just `DeepSeek-R1-Distill-Qwen-7B` at one cell:
  `python3 scripts/probe-vllm-reasoning.py --host-vram-gb 24 --vram 24
  --ctx 32768 --repo "DeepSeek-R1-Distill-Qwen-7B" --force`
- [ ] Expect `T=deepseek_string dis=y` in the output.
- [ ] `make cache-up` and run a tool-using chat through the router to
  confirm round-trip works end-to-end (not just probe-level).

### 6. Documentation
- [ ] Add a section to `docs/backends.md` titled "Custom tool parser
  plugins" explaining the bind-mount + flag pattern, the
  `deploy/vllm-plugins.json` registry, and how to add a new plugin.
- [ ] Add an entry to the cell schema doc (wherever vLLM cache schema
  v2 is described) noting that `tool_parser` may now be a plugin
  name, not just a built-in.

### 7. Out-of-scope for this plugin (future work, separate)
- DeepSeek-R1-Distill-Llama-8B uses the same chat template; it should
  work with this plugin too. No code change needed, just add it to the
  re-probe set after wiring.
- Original DeepSeek-V3 / R1 weights have atomic vocab tokens, so they
  should keep using built-in `deepseek_v3` for the (very minor) speed
  benefit of token-id streaming counts. Don't migrate them.
- An SGLang equivalent would need its own implementation in SGLang's
  Python detector framework. Out of scope until SGLang's NVFP4 path
  is unbroken (separate issue, S1).

## Risks / things to watch

- **Streaming correctness**: the plugin's streaming path uses
  substring counts in `current_text` instead of token-id counts in
  `current_token_ids`. For full-width DeepSeek markers that no
  tokenizer fragments mid-character this is equivalent, but if a
  future model emits the markers split across an encoding boundary
  (e.g., a tokenizer that BPE-splits the full-width separator) the
  streaming path could miss a count. Non-streaming path uses regex
  on the complete output; immune to this.
- **vLLM API drift**: imports come from `vllm.entrypoints.openai.engine.protocol`
  in this image. vLLM has reorganised the `protocol` modules a few
  times; the build-time test in step 3 will catch drift on each image
  bump.
- **Plugin path inside container**: the prober and router must agree
  on the in-container path (`/plugins/...` vs.
  `/usr/local/share/vllm-plugins/...`). Pick one and put it in
  `deploy/vllm-plugins.json` so both consumers read the same source.
