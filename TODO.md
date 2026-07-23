# TODO

> Reconciled against `git log` on 2026-05-01 (covers commits afb7c59 -> b77e42b).

## Completed

### Foundation
- [x] Aider / Claude / Codex / Gemini agents in JupyterLab + shell launcher cards
- [x] Multi-port router (Ollama :11434, vLLM :11435, SGLang :11436) with GPU exclusion + graceful drain
- [x] Container liveness verification (detect externally stopped containers)
- [x] Interactive fzf model picker (shell + Jupyter launcher cards)
- [x] Dynamic GPU memory fraction + context length per model size

### Probe & catalog pipeline
- [x] Upstream-driven catalog (`make catalog-regen` -> `deploy/models.yaml`)
- [x] Per-tier probe schema v3, digest-keyed cache (Ollama)
- [x] 2-D probe matrix (VRAM x CONTEXT); `active-models.yaml` removed -- cache IS the active set
- [x] Disk-driven picker; rows hidden until `fully_on_gpu` / `fits`
- [x] Implied-spill propagation: smaller-ctx spill short-circuits larger ctx
- [x] gguf source kind: FROM + RENDERER + PARSER Modelfile, `ollama create`-driven
- [x] Per-session num_ctx via Modelfile-derived tags (Ollama, agent-agnostic)
- [x] Probe-truth picker: probe-measured `actual_vram_gb` used directly when `fits=true`
- [x] Cell-matrix selector: `(family, backend, ctx)` cells; quality rank by params then quant precision
- [x] Probe clamp removal: tiers beyond model's nominal max_context now run a real probe
- [x] `gpt-oss` family added with verified vLLM + SGLang parsers
- [x] Catalog `hf_weight_bytes` reads `safetensors.index.json` first; excludes `original/`, `metal/`, `consolidated/` mirror dirs

### Reasoning & tool parsing
- [x] Runtime reasoning probe; capability-aware policy
- [x] Ollama native `think:` field; vLLM/SGLang `enable_thinking` + `reasoning_effort` / `separate_reasoning`
- [x] Per-request reasoning override via `<name>::<token>` suffix (e.g. `::nothink`)
- [x] `maybeStripTools` drops `tools`/`tool_choice` for unverified-tool-parser HF models
- [x] `none` capability bucket distinguishes correct-by-design non-reasoning from broken parser pairings

### vLLM / SGLang parity
- [x] Probe runners (`probe-vllm-reasoning.py`, `probe-sglang-reasoning.py`) on shared `_probe_hf_common`
- [x] Schema-v2 caches (repo+sha-keyed, per-cell `fits` + evidence)
- [x] vLLM / SGLang start as `sleep infinity` placeholders; router recreates on demand
- [x] `<name>@<ctx>` per-session context binding for HF backends
- [x] HF rows synthesized into router from probe caches; `tool_parser` plumbed through launchConfig
- [x] CI flag-drift guard: `deploy/backend-flags.yaml` + `make verify-backend-flags`
- [x] SGLang NVFP4 unblocked in `v0.5.10.post1-cu130` via `--disable-piecewise-cuda-graph` (the piecewise CUDA-graph warmup torch.compiled the forward, and Dynamo choked on flashinfer's FP4 JIT at `modelopt_quant.py:1482`). 8 SGLang models now serve (Qwen3-8B/14B, Llama-3.1-8B, Nemotron-Nano-9B, gpt-oss-20b, DeepSeek distills, Qwen3-14B-FP8); genuine arch/quant gaps remain for Gemma-4 family, Qwen3.5/3.6 MoE, diffusiongemma (those serve on vLLM)
- [x] vLLM parser plugin registry (`deploy/vllm-plugins.json` + `scripts/vllm_plugins/`); prober and router both consume it. `deepseek_string` plugin wired for `deepseek-r1-distill` family. Adding new plugins = drop file + one JSON entry.
- [x] Two-phase tool probe (auto + forced fallback) so reasoning models verify tool parsers; cache row carries `tool_mode`. Router promotes single-tool auto requests, rejects multi-tool auto with HTTP 400 + actionable error. End-to-end verified on R1-Distill-Qwen-7B and R1-Distill-Llama-8B.
- [x] `refresh_top_level_from_cells` now picks parser fields (tool_parser, reasoning_parser, tool_mode, disable_verified) from the **most-recent clean cell that has them populated** rather than the smallest-tier cell. Fresh `--force` re-probes of a single cell now propagate to the top-level row without requiring a full matrix re-probe; old cells with stale `None`s no longer shadow new evidence.
- [x] `DeepSeek-R1-Distill-Llama-8B` weights downloaded and probed; cache shows `T=deepseek_string mode=forced`. End-to-end verified through router: single-tool auto -> 5-token tool call (vs 525 tokens for Qwen-7B variant -- see `docs/backends.md` operational note).

### Documentation
- [x] `docs/router.md` -- comprehensive router reference (architecture, ports, lifecycle, request rewrite chain, config, caches, failure modes, operator tasks). Linked from CLAUDE.md and README.md.

### Standalone launcher
- [x] `bin/devai-agent` (renamed from devai-shell): Python launcher, persisted prefs in `~/.devai/preferences.yaml`
- [x] `make install` symlinks launcher + all 3 probe caches + picker source under `~/.devai/`
- [x] Mounts `$PWD` (not `last_work_dir`) as work dir; back-channel `.last-pick.json` round-trip

### Logging & infra
- [x] `devai-logger` sidecar streams every `devai-*` container's stdout to `/var/cache/devai/logs/`
- [x] `make setup-logs` creates a 100 GB thin LV at `/var/cache/devai/logs`
- [x] `make logs SERVICE=<name> [LINES=N]` tails persisted logs

### Tests
- [x] Go unit tests (31+) covering parseSizeGB, memFraction, computeLaunchConfig, parseReasoningOverride, maybeStripTools, synthesizeHFFromCache
- [x] Live integration per backend (`test-router-{vllm,sglang}.sh`)
- [x] Exhaustive matrix: every probed digest x wire x scenario (`test-model-matrix.sh`)
- [x] E2E picker -> agent -> live router round-trip (`test-e2e-picker.sh`)
- [x] Probe smoke tests + Ollama-prober byte-identical regression check

### Docs
- [x] README / CLAUDE.md / AGENTS.md / `docs/backends.md` reflect steady state (matrix-mode pull, `::nothink`, devai-agent, picker columns)

## Open Items

- [ ] Family-level `engine_flags` field (e.g. `--enforce-eager`) so OOM-prone NVFP4 MoE variants like `Nemotron-3-Nano-30B-A3B-NVFP4` can free the CUDA-graph budget and re-enter `hf_repos`.
- [x] Re-evaluated SGLang NVFP4: the blocker was the piecewise CUDA-graph default (Dynamo tracing flashinfer's FP4 JIT at `modelopt_quant.py:1482`), not `fp4_quantize` itself. Fixed with `--disable-piecewise-cuda-graph` (always-on); no upstream fix needed. Drop the flag when a future SGLang image can trace the FP4 path.
- [ ] Probe context tiers above the model's native rope: 128K/256K cells of R1-Distill-Qwen-7B failed `kind=infra` because Qwen-2 base has 32K positional encodings without rope_scaling. Either skip those tiers in the prober when `max_position_embeddings < requested_ctx`, or extend implied-spill to `infra` failures of the right shape. Today they leave noisy red entries in the cache.
- [ ] Optional: SGLang `deepseek_string` analogue. SGLang's tool-parser plugin model is Python-import based (a registered class, not a file path); requires a separate implementation against SGLang's detector framework. No longer gated on NVFP4 (now unblocked) -- still optional.

### pipelock egress lockdown (network fail-closed)
- [ ] **MCP search for all agents.** opencode has Exa web search (`OPENCODE_ENABLE_EXA=1`); the other agents' built-in search is cloud-provider-coupled and doesn't work against the local router. Give them all search uniformly via the MCP gateway's `duckduckgo` server (or self-hosted SearXNG) -- connect each agent's MCP config to the gateway. Provider-agnostic, works on local models. (Decision: opencode-only for now, 2026-06-16.)
- [ ] **Make the MCP gateway reachable from the locked lab.** `devai-mcp-gateway` is on `devai-net` but not `devai-lab-egress`, so the egress-locked lab can't reach it. Dual-home it onto the egress net (like router/caches) and decide whether the gateway's own internet egress routes through pipelock or is a trusted exception.
- [ ] **Fix web fetch for the system-runtime agents (Gemini CLI, Claude Code).** Their Node 22 `fetch` (undici) ignores `HTTPS_PROXY` -> connects direct -> no route on the locked net. Bump the lab's Node to 24 + set `NODE_USE_ENV_PROXY=1` (or preload an undici `EnvHttpProxyAgent`) and install a system Bun. opencode is unaffected (bundled runtime honors the proxy).
- [ ] **pipelock DLP false-positives on signed URLs.** Partly fixed: the `JWT Token` pattern now exempts GitHub's download CDNs (`release-assets.githubusercontent.com`, `objects.githubusercontent.com`, `codeload.github.com`) -- verified 2026-06-16 that github release fetches pass (404 real upstream) while the pattern still blocks JWTs to other hosts (example.com -> 403). Remaining: other URL-token patterns (Azure SAS, Google `ya29`, "Credential in URL") could false-positive on signed URLs the same way -- generalize via per-pattern `exempt_domains`, or scope `request_body_scanning` to outbound bodies (POST/PUT) instead of GET URLs. Loosening DLP is a security tradeoff -- decide deliberately.
- [ ] **systemd boot network-create gap.** The systemd unit relies on the `external:` networks pre-existing but doesn't create them. `devai-net` AND `devai-lab-egress` need an `ExecStartPre` (or make step) so a locked lab survives a reboot without a manual `make cache-up`.

### Open WebUI enhancements (and web access to the devai backend)

> Supersedes the dropped `docs/plans/lab-ui-enhancement.md` draft. Open WebUI already runs in the stack and natively shows per-response tok/s + token counts and per-chat Advanced Params for Ollama -- so the value is in filling the gaps it doesn't cover, not rebuilding it. Reference UX: llama-server's built-in WebUI (`timings_per_token`, "Show tokens/sec" toggle, runtime sampler panel). Ideas only; nothing planned in detail yet.

- [ ] **Per-response timing cards via an Open WebUI Filter Function.** Inline TTFT / tokens-per-sec / token counts / model-load-time attributed to the `devai-router` path, delivered as a single uploaded function (no new container). Ollama's native `load_duration` / `eval_duration` already supply most fields; LM Studio's `time_to_first_token_seconds` / `tokens_per_second` / `model_load_time_seconds` triad is a good display spec; community precedent exists ("Time Token Tracker", "Chat Metrics").
- [ ] **Extend timing visibility to vLLM / SGLang.** Open WebUI's native tok/s is Ollama-only, but the expensive cold-starts are NVFP4 on vLLM/SGLang (`HEALTH_TIMEOUT_SECONDS=600`) -- surface load-time / TTFT where it actually matters.
- [ ] **Lightweight single-file chat window on the router's OpenAI-compat surface.** Potential basis: `dmeldrum6/Local-LLM-Chat` (https://github.com/dmeldrum6/Local-LLM-Chat -- single HTML, OpenAI-compatible) pointed at `devai-router` for local inference. Low-friction, editable-in-repo way to reach the devai backend from a browser; complementary to Open WebUI, not a replacement. Reference llama-server's WebUI for the timing/sampler panel.
- [ ] **Browser-reachability prerequisites for any web-facing surface.** The router emits no CORS headers and `11434` is not host-published; a browser-direct page needs either same-origin reverse-proxying of `/api` + `/v1` through the serving container or CORS on `gpu-arbiter`. Also a deliberate egress-side choice (`devai-net` vs `devai-lab-egress`) + pipelock/CA implications for a web entry point.
