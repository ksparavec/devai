# Inference backends -- Ollama, vLLM, SGLang

DevAI exposes three inference backends behind a single multi-port router
(`gpu-arbiter`). Each backend serves a distinct port; the router enforces
GPU mutual exclusion (only one backend uses the GPU at a time), manages
the per-request context cap, applies the reasoning policy, and emits the
correct backend startup flags (`--reasoning-parser`, `--tool-call-parser`)
when the probe cache has confirmed values for the model.

| Backend | Port | Image | Models |
|---|---|---|---|
| Ollama  | 11434 | `ollama/ollama:latest` | GGUF (Q3/Q4/Q5/Q8/etc.) |
| vLLM    | 11435 | `vllm/vllm-openai:v0.22.1-x86_64-cu129-ubuntu2404` | NVFP4, FP8, BF16/FP16 safetensors |
| SGLang  | 11436 | `lmsysorg/sglang:v0.5.10.post1-cu130` | NVFP4, FP8, BF16/FP16 safetensors (RadixAttention multi-turn) |

Backend launch-flag *names* are pinned in `deploy/backend-flags.yaml`.
Run `make verify-backend-flags` after bumping either image -- it dumps
`--help` from the pinned image and asserts every named flag is present.
Matching is now **exact flag-token** matching, not substring: a removed
flag whose name is a prefix of a surviving one (`--tp` vs `--tp-size`)
no longer passes silently. `--kv-cache-dtype` is pinned for both
backends. One known finding it now surfaces: SGLang's `--tp` reports
`tp=--tp (not advertised; prefix of --tp-size)` -- the pinned image
advertises only `--tensor-parallel-size` / `--tp-size`, and `--tp`
works today solely because argparse accepts it as an unambiguous
abbreviation. See the cross-unit note in the router/prober spelling
discussion: both emitters should move to `--tp-size` before upstream
adds a second `--tp*` option.

All three are reachable via the router from inside the `devai-net` Podman
network. Agents (Claude Code, Aider, Codex, LATE, Open WebUI) talk to the
router on the appropriate port; the picker emits the right port based on
the chosen model's backend.

## Lifecycle

`make cache-up` brings up:

- **Always running**: `devai-ollama`, `devai-router`, `devai-open-webui`,
  `devai-webui-proxy`, `devai-apt-cache`, `devai-registry-cache`,
  `devai-logger`.
- **Idle placeholders**: `devai-vllm` and `devai-sglang` start with
  `entrypoint: ["sleep", "infinity"]`. They hold the service definition
  (image, mounts, GPU device, network) but consume no GPU resources.

When the first request hits port 11435 (vLLM) or 11436 (SGLang), the
router:

1. Stops the other GPU-using backends -- `devai-ollama` is unloaded via
   `/api/generate` with `keep_alive=0`; vLLM/SGLang containers are
   stopped via libpod.
2. Removes the placeholder `devai-vllm` / `devai-sglang` container.
3. Recreates it via libpod with a dynamic entrypoint that bakes in the
   chosen model path, `--max-model-len` / `--context-length`, and
   `--gpu-memory-utilization` / `--mem-fraction-static` derived from the
   probe cache and `MAX_CONTEXT_LEN`. For vLLM the entrypoint also
   passes `--kv-cache-dtype <per-model>`, resolved from the probe cell
   covering the launch ctx -- there is no global KV dtype policy (see
   "Per-tier KV-cache dtype" below). If the model
   has probe-verified `reasoning_parser` and/or `tool_parser`, the
   router injects `--reasoning-parser <value>` (vLLM also adds
   `--enable-auto-tool-choice`) and `--tool-call-parser <value>`.
   Finally, any `engine_flags` / `engine_env` from
   `deploy/recovery-flags.json` keyed by the canonical model name are
   appended (e.g. `--enforce-eager` for Nemotron-3-Nano at 128K --
   see docs/router.md "Per-model recovery flags"). An entry may carry
   an optional `backends` allow-list: absent (or `null`) means "every
   backend", `[]` means "no backend" (the operator disable switch),
   a list means only those, and a non-list value warns naming the
   model and is treated as absent. Decoding is per-entry, so one
   malformed entry never discards the rest of the registry. Both the
   Go router and the Python probers implement that contract, so probe
   and serve-time launches agree.
4. Polls `/health` until the container becomes ready (default 600s,
   override via `HEALTH_TIMEOUT_SECONDS`). The poll fails fast: if the
   container exits or its logs show a terminal error (`detectLaunchFailure`
   with signatures ported from the probe classifier), the router aborts
   immediately with that error instead of waiting the full timeout on a
   crashed engine.
5. Applies the reasoning policy and tool-stripping rules to the request.
6. Proxies the original request through.

A second request that switches the **model** or the **context cap**
triggers another recreate. The router tracks `currentModel`,
`currentContext` and `currentSpec` (the speculative-decoding / MTP
config) per backend; any of those changing -> recreate. A **reasoning
override does NOT recreate anything** -- there is no
`currentReasoningOverride` field. Reasoning is a per-request body
rewrite (`::<reasoning>` suffix -> injected `think` / `reasoning_effort`
/ `enable_thinking`), so it can change on every request against the
same running container.

## Probing -- building the cache

Each backend has its own probe cache:

- `deploy/.ollama-reasoning-cache.json` -- schema v3, digest-keyed.
- `deploy/.vllm-reasoning-cache.json` -- schema v2, repo+sha-keyed.
- `deploy/.sglang-reasoning-cache.json` -- schema v2, repo+sha-keyed.

Schema v2 (vLLM/SGLang) added three top-level fields per entry:

- `reasoning_parser` -- backend startup flag value (e.g. `qwen3`) that
  produced a `structured` round-trip in Probe A. Null when the curated
  family hint did not pan out, when no hint was supplied, or when the
  model's capability is `inline`/`unsupported`.
- `tool_parser` -- backend startup flag value (e.g. `hermes`, `qwen25`)
  that produced a parseable tool call in Probe B. Null when no curated
  hint, or when the round-trip failed.
- `disable_verified` -- true iff Probe C suppressed `reasoning_content`
  on a structured-capable model. Mirrors Ollama's `disable_verified`;
  gates the router's "off" rewrite. When the disable probe itself
  fails, the field is left **absent** (it is no longer written as the
  string `"error"`), so the next `make probe` retries it; the summary
  line reads `(disable probe failed; will retry)`. The router also
  tolerates a legacy non-boolean value by degrading that one model to
  "unknown" rather than failing the whole cache parse.

The picker hides any model that lacks a `fits=true` probe at the host
VRAM band. The router synthesizes `/v1/models` rows from these caches.
Without probes, models are invisible.

**Single-cell binary search (vLLM/SGLang).** The HF probers no longer
scan a fixed 32K/64K/128K/256K grid and store a cell per tier. They
**binary-search** the largest context that both fits AND serves under a
near-full-context request, on the 32K-multiple grid up to
`min(MAX_CONTEXT_LEN, position_limit, max(PROBE_CONTEXTS))`, and keep exactly **one** cell
per `(model, backend)` -- the winner. Any candidate above the model's
as-shipped `position_limit` fails instantly with no launch; if even 32K
fails to serve, no cell is written and the model is recorded in the
exclusion ledger (`oom`). Consumers treat the single winner cell as
covering every ctx `<=` it (KV monotonicity): the picker reads the
actual recorded cell keys, and `select-models.hf_probe_at_context`
resolves a sub-winner query to the winner. Ollama still records
multiple tiers (it is not part of this change), and its reader
(`select-models.probe_at_context`) has **no** winner-cell fallback: a
miss at the requested tier means that tier was never probed.

Every consumer clamps the request to the entry's own `max_context`
first (`eff = min(ctx, max_context)`) before looking a cell up, on both
backends -- asking a 65K-ceiling model for 256K just asks it for 65K.
`devai-model-status`'s `list_fitting_models` reproduces the same two
readers; see
[docs/mcp-model-status.md](mcp-model-status.md#list_fitting_models).

### Exclusion ledger (`deploy/.model-status.json`)

A host-local overlay (gitignored, schema v1) recording models the host
should NOT bother with, so they are not re-downloaded, re-probed, or
re-listed as "not on disk". The catalog stays the host-agnostic superset;
the ledger is the negative space the probe caches do not cover. Keyed by
catalog `name` + backend (sha-stable, unlike the repo+sha probe key).
Reasons:

- `too_big` / `too_small` -- outside the host VRAM window. Written by
  `make model-pull` (`select-models.py`), which already excludes these from
  download; the ledger persists the verdict so the probers skip them
  silently. Re-derived on a GPU-VRAM change.
- `unsupported_arch` -- the engine cannot load the architecture. Written by
  the HF probers when a launch fails with `kind=arch`. Terminal, vram- and
  sha-stable (carried forward across a re-quant by `_carry_forward_terminal`
  + `prune_orphaned_shas`). Re-evaluated with `PROBE_FORCE_ARCH=1`.
- `oom` -- re-checked on a new sha (weight-specific). `manual` -- operator
  pinned.

Inspect with `make model-status`; clear an entry with
`make model-status CLEAR=<name>[::<backend>]`. The ledger fails open: a
missing or malformed file simply means "nothing excluded". Writes are
atomic (temp file + `os.replace`), like the probe caches, so an
interrupted run cannot truncate it.

`make model-sync` prunes rows whose model is no longer in the catalog
(non-dry-run only, against the unfiltered catalog). Two guards keep a
bad catalog from emptying the ledger: an empty catalog is a no-op, and
a prune that would remove more than half the ledger is refused --
clear those by hand with `make model-status CLEAR=<name>` if the
removal really is intended.

### Procedure

```bash
# 1. Ollama probing (Make-orchestrated, runs live with Ollama container)
#    Each PROBE_VRAMS band recreates devai-ollama with OLLAMA_GPU_OVERHEAD
#    set so the daemon behaves as if it had only that VRAM available.
make probe                                          # all bands x all contexts
make probe PROBE_VRAMS=24G PROBE_CONTEXTS=32K      # one band, one tier
make probe PROBE_FORCE=1                           # re-probe everything

# 2. HF probing (vLLM/SGLang) -- requires exclusive GPU access
#    Stop the live router and ollama first.
make cache-down

# 3. For each HF backend, launch a probe container and run all cells.
#    For each (model, vram_band, ctx_tier) cell:
#      A) fit + reasoning      -- classify capability, snapshot nvidia-smi
#      B) tool-call            -- only when parsers.<backend>.tool is set
#      C) disable verification -- only when Probe A produced `structured`;
#                                verifies suppression of `reasoning_content`
#    Each cell takes 1-3 minutes; extra probes add a few seconds each.
make probe-vllm                                     # all vLLM models, all cells
make probe-sglang                                   # all SGLang models, all cells
make probe-vllm PROBE_REPO=Llama                   # filter to matching models
make probe-sglang PROBE_CONTEXTS=128K              # single context tier
#    `make probe-sglang` can only probe weights that exist in
#    SGLANG_MODELS_DIR (/var/cache/devai/sglang). `make model-pull`
#    always downloads into the vLLM store, so SGLang weights are a
#    separate, explicit copy:
python3 scripts/select-models.py --name <n> --download --hf-store sglang
#    An SGLang cell marked fits=true with no weights in that store is
#    an advertised model the router cannot launch, so select-models.py
#    flags it -- see "SGLang store gaps" below.

# 4. Restart the stack -- router reloads all three caches at boot.
make cache-up
```

### SGLang store gaps (`--hf-store`, `--ignore-store-gaps`)

`devai-sglang` mounts `SGLANG_MODELS_DIR` (`/var/cache/devai/sglang`),
a different volume from `devai-vllm`'s `VLLM_MODELS_DIR`. `make
model-pull` -- including `NAME=<row>` -- always downloads into the vLLM
store, so weights are never implicitly visible to SGLang and copying
them is a second full copy of the weights on a different filesystem
(no hardlinks). The only way to populate the SGLang store is the
explicit opt-in:

```bash
python3 scripts/select-models.py --name <n> --download --hf-store sglang
```

An SGLang probe cell marked `fits=true` whose weights are absent from
that store is a **store gap**: the picker advertises the row and the
router cannot launch it. `select-models.py` detects these
(`sglang_weight_gaps`) and reports them on **stderr**, never stdout,
so `make model-fit`'s table stays machine-readable. The rule in one
sentence: **warn on every run that could be misled by a gap, and fail
only when the run is about to act on a model that IS in the gap.**

| Invocation                       | On a store gap                                    |
| -------------------------------- | ------------------------------------------------- |
| read-only (`make model-fit`, plain `select-models.py`) | Warning banner on stderr; the fit table still prints and exit status is unchanged. A read-only diagnostic must not be turned off by the condition it is diagnosing. |
| enumerating `--download` (`make model-pull`, no `NAME=`) | Same banner as an error, then **exit 1**. That run picks its trial candidates off the very probe cache that is lying about the SGLang store, so every decision downstream of it is made on the wrong picture. |
| `--name <X> --download` (`make model-pull NAME=`, `make model-sync`) | Same banner; **exit 1 only when `X` is itself a gap row**, since pulling `X` into the vLLM store cannot repair `X`'s own SGLang gap. When `X` is not a gap row the operator asked for a specific, unrelated model -- the banner is a warning and the pull proceeds. |

The `--name` row matters because `make model-sync` is the one caller
that downloads unattended: it shells out to `make model-pull
NAME=<row>`, and the Makefile forwards `NAME` as `--name`. The check
used to sit *after* `main()`'s `--name` short-circuit and was therefore
unreachable on exactly that path; it now runs on both. The
fatal-iff-requested rule cannot wedge that loop, for two independent
reasons: structurally, a gap row is by definition already in the SGLang
probe cache (that is what "advertised" means), so `plan_sync`
classifies it `evaluated` and only `new` rows are ever pulled by name;
and defensively, `model-sync`'s `execute()` logs a non-zero per-row rc
and moves on rather than aborting the run.

Override the abort with `--ignore-store-gaps`, or
`IGNORE_STORE_GAPS=1` in the environment (the make targets expose no
flag, so the env var is the way through them). The check is skipped
entirely when the run already targets `--hf-store sglang` -- that path
is the repair route, and blocking the repair with the condition it
repairs would be circular.

The other repair is to drop `sglang` from the row's backends and
re-run `make probe-sglang` to clear the stale cells.

### Catalog regeneration is all-or-nothing

`scripts/generate-catalog.py` rewrites `deploy/models.yaml` **whole**,
so an upstream fetch that transiently fails costs rows. It therefore
compares the row count it is about to write against the existing file
and refuses:

- **Transient loss** (network error, 5xx, timeout): the write is
  refused and the script exits **1** with
  `REFUSING to overwrite deploy/models.yaml`. `make catalog-regen` and
  `make model-sync REGEN=1` propagate that failure rather than
  continuing against a truncated catalog. Re-run when upstream is
  healthy.
- **Permanent loss** (HTTP 404/410/401/403): reported as
  `GONE (HTTP nnn)`, the row is dropped, and the catalog **is**
  written -- a repo that no longer exists is not a transient failure.
  Delete that repo from `scripts/model-families.yaml` so the next run
  stops asking for it.
- `--allow-partial` writes anyway, for the case where an operator has
  inspected the transient losses and accepts the smaller catalog. It
  is an opt-in, not a restored default.

### Curating parser hints

Reasoning and tool-call parsers are per-architecture. The curated
choices live in `scripts/model-families.yaml` under each family's
`parsers:` block:

```yaml
- name: qwen3.5
  ...
  parsers:
    vllm:
      reasoning: qwen3
      tool: hermes
    sglang:
      reasoning: qwen3
      tool: qwen25
```

`make catalog-regen` propagates these into per-row `parsers:` blocks
in `deploy/models.yaml`. The probers read the row's block and pass
`--reasoning-parser` / `--tool-call-parser` (vLLM also adds
`--enable-auto-tool-choice`) to the launch. A field is only confirmed
in the cache when the corresponding round-trip succeeds -- a curated
hint that the model doesn't actually honour produces a null cache
entry, and the router launches without the flag.

### Probe knobs

| Env / Make var | Effect |
|---|---|
| `PROBE_VRAMS=16G,24G` | Ollama target bands |
| `PROBE_VRAMS_VLLM=24G` | vLLM target bands |
| `PROBE_VRAMS_SGLANG=24G` | SGLang target bands |
| `PROBE_CONTEXTS=32K,64K,128K,256K` | Context tiers. For vLLM/SGLang this **caps the binary-search ceiling** (`max()` of the listed tiers) in BOTH the fit pass and the `--load` pass; the 32K-multiple grid below that ceiling is still searched, and any listed non-grid tier is unioned in. `PROBE_CONTEXTS=32K` therefore probes exactly one tier. For Ollama it is the literal list of tiers probed. **On the load pass the cap is not just a filter -- it can MOVE the winning cell down**: the load probe keeps exactly one cell at the largest ctx that actually serves within the cap, so re-running with a narrower `--ctx` shrinks that model's advertised `max_context`. Re-run at the full ceiling to restore it. |
| `PROBE_READY_TIMEOUT=180` | Seconds `make probe` waits for the recreated `devai-ollama` to answer `ollama list` before giving up. The whole `probe` recipe is one `set -e` shell, so this `exit 1` aborts the **entire probe run**, not just the current VRAM band -- the remaining bands are not attempted. The EXIT trap still restores `OLLAMA_GPU_OVERHEAD`. |
| `PROBE_REPO=Llama-3.1-8B` | Regex filter on catalog rows (HF probers only) |
| `PROBE_FORCE=1` | Re-probe every cell even if cached |
| `PROBE_FORCE_ARCH=1` | Re-probe top-level capability/arch fields |
| `PROBE_NEEDLE_DEPTH=0.5` | Load probe: fractional depth (0.0 top, 1.0 bottom) of the recall needle |

### Load probing -- serving-time VRAM under near-full context

The fit probe (steps 2-3 above) snapshots `nvidia-smi` once, right
after `/health` plus a short chat. That captures the *load ceiling* --
weights + the reserved KV pool -- but NOT the *serving transient* that
vLLM/SGLang allocate per decode step once a real, near-full-context
request arrives: the softcap-logits buffer (`max_num_batched_tokens x
vocab x 4 bytes`) plus attention/activation workspace. A model can fit
at load and still OOM on the first large prompt -- this is exactly what
DiffusionGemma did (fit=true at 256K in the cache, crash-loop on any
prompt over one prefill chunk).

The LOAD probe closes that gap. It is a SECOND pass that *layers onto*
the existing fit cache -- it never rewrites a fit cell, only augments
it:

```bash
make cache-down                 # same exclusive-GPU precondition as fit probing
make probe-vllm                 # fit cache must exist first
make probe-load-vllm            # then layer serving-time data on top
make probe-load-sglang          # likewise for SGLang
make probe-load-vllm PROBE_REPO=Gemma PROBE_CONTEXTS=32K,64K,128K
make cache-up
```

For each downloaded model, at the host VRAM band, it walks the context
tiers **ascending** (32K -> 64K -> 128K -> 256K) and, for every tier the
fit probe already marked `fits=true`, it:

1. relaunches the backend at that `--max-model-len` with the SAME
   verified parsers + recovery flags the router serves with, and under
   the KV dtype **that cell was fit-probed with** (its stamped
   `kv_cache_type`) rather than the pass-global `PROBE_KV_CACHE_TYPE` --
   an unstamped legacy cell decodes to fp8 on vLLM and to the engine
   default on SGLang, matching `synthesizeHFFromCache` in the router,
2. records a baseline VRAM reading once `/health` passes (idle),
3. builds a haystack prompt that fills the KV pool to ~99% of the window
   (target `ctx - 512` tokens, so the pool is exercised near its true
   ceiling where real OOMs happen -- a static char estimate only reaches
   ~85-88%). The size is tokenizer-verified: the prompt is grown/trimmed
   against the backend's `/tokenize` endpoint until it lands at-or-just-
   under target. vLLM exposes `/tokenize` (exact, lands at 0.98-1.00 fill);
   SGLang does NOT, so it falls back to calibrating chars/token from a
   short chat's `usage.prompt_tokens` (`serving_fill_method=calibrated`,
   lands ~0.76-0.94 -- less precise, but SGLang's models are non-tight).
   Each cell records `serving_fill_ratio` + `serving_fill_method` for
   audit. The haystack is a public-domain corpus (Moby-Dick + War and
   Peace) with a unique needle (`RHINO-7741-DELTA-VAULT`) at
   `PROBE_NEEDLE_DEPTH`; the corpus is fetched from Project Gutenberg on
   first use into `~/.cache/devai/probe-corpus/` (override with
   `DEVAI_PROBE_CORPUS_DIR`; pre-populate on an air-gapped host) -- it is
   NOT vendored in git,
4. sends ONE completion (max_tokens 256, small so prompt+output stays
   under `--max-model-len`) while a 0.1s VRAM sampler runs,
5. captures the peak, scores needle retrieval, classifies OOM
   (transport error / container exit / OOM markers in logs),
6. augments the cell with `serving_ok`, `serving_peak_gb`,
   `transient_gb` (= peak - baseline), `needle_score`, and the
   `predicted_logits_gb` diagnostic,
7. **stops ascending at the first OOM** -- a larger context cannot fit a
   transient that already overflowed -- and marks the higher tiers
   `serving_ok=false` (implied) without launching them.

The new fields are additive; no schema bump. Both consumers gate on
`serving_ok` **only when present** -- a cell the load probe never
touched (`serving_ok` absent) keeps the pre-load-probe behaviour
byte-for-byte:

- **router** (`synthesizeHFFromCache`): a `fits=true` / `serving_ok=false`
  cell is excluded from `ProbedMaxCtx`, so the per-name request ceiling
  falls back to the largest tier that both fits AND serves.
- **picker** (`_vram_from_hf_probe`): the same cell gets
  `fully_on_gpu=false`, so `_max_fitting_ctx_info` won't advertise a ctx
  the router would refuse.

The probe is vLLM/SGLang only today (the BackendSpec path). Ollama
allocates KV per-request from `num_ctx` via llama.cpp and has no LOAD
pass yet.

### Per-tier KV-cache dtype (all backends, additive field)

Ollama probe cells carry TWO stamps describing the environment they
were measured in:

- `kv_cache_type` -- the KV dtype the daemon served the cell under,
  from the `OLLAMA_KV_CACHE_TYPE` env of the probe pass (absent/empty
  = `f16`, which is also what pre-field legacy cells mean).
- `flash_attention` -- the `OLLAMA_FLASH_ATTENTION` setting of the same
  pass. Absent on pre-stamp cells.

Both are resolved at serve time from the **same** covering tier, so a
launch always reproduces one probe cell rather than mixing two. A cell
probed under `q8_0` only fits WITH quantized KV, so fit is
dtype-scoped:

- **prober**: `make probe PROBE_KV_CACHE_TYPE=q8_0 PROBE_FLASH_ATTENTION=1
  PROBE_FORCE_CTX=128K PROBE_MODELS=<tag>` re-probes one tier under
  quantized KV and stamps it; unforced tiers keep their f16 cells.
- **router** (`resolveKVCacheType` / `resolveFlashAttention`): at
  launch, the smallest probed tier covering the serving ctx supplies
  BOTH stamps -- same rule, same tier -- and the ollama `DynamicEnv`
  bakes them into the recreated container. Flash attention comes from
  the cell's own `flash_attention` stamp when present; only pre-stamp
  cells fall back to deriving it from the dtype. That fallback is
  one-directional: quantized KV requires flash attention, but a cell
  can equally have been probed with flash attention ON under the
  default `f16`, and deriving from the dtype alone would then serve it
  without flash -- a different environment from the one its fit was
  measured in. f16/unstamped tiers emit nothing -- the container spec
  is byte-identical to pre-field builds.
- **picker** (`_kv_cells` / `_kv_mixed`): a model whose fitting tiers
  span both dtypes gets a context-tier sub-modal (tier choice = quality
  choice) and pins `@<ctx>` on the emitted name so the router serves
  exactly the chosen tier; the preview pane warns that quantized tiers
  have measurably weaker long-form reasoning (qwen3.6:35b-a3b-mtp:
  GPQA 0.8667 at 64K/f16 vs 0.75-0.77 at 128K/q8_0, two runs; all
  short-chain metrics unaffected -- see `.bench-cache.json` row
  `...::ollama::131072`).

Do NOT set `OLLAMA_KV_CACHE_TYPE` globally in `.env`: fit cells are
dtype-scoped, and a global flip silently invalidates every f16 cell.
The per-tier probe flow above is the only supported path.

Force-wipe hazard: a plain `make probe PROBE_FORCE=1` re-measures every
tier under the DEFAULT f16 env -- a quantized-only tier (q8_0 128K)
then spills, gets rewritten `fully_on_gpu=false`, and disappears from
the picker/router until the targeted `PROBE_KV_CACHE_TYPE=q8_0` pass
is re-run. After any full-force re-probe of a mixed-KV model, re-run
the quantized-tier pass to restore its cell.

The same per-cell dtype contract covers vLLM and SGLang -- there is no
global KV dtype policy:

- **vLLM**: the prober launches every cell with `--kv-cache-dtype
  $PROBE_KV_CACHE_TYPE` (default fp8) and stamps it; serve time
  (`vllmEntrypoint`) reproduces the covering cell's stamp. UNSTAMPED
  legacy cells decode to fp8 on both sides -- that is the dtype they
  were factually measured under (the pre-field hardcode), so behaviour
  is byte-identical until a model is deliberately re-probed. A model
  with VRAM slack can be re-probed `make probe-vllm
  PROBE_KV_CACHE_TYPE=auto PROBE_REPO=<repo> PROBE_FORCE=1` to serve
  unquantized KV; fit cells are dtype-scoped, so never edit the stamp
  by hand.
- **SGLang**: legacy cells ran the engine default (no flag) and stay
  unstamped; `PROBE_KV_CACHE_TYPE=fp8_e5m2` (etc.) enforces + stamps a
  dtype, and `sglangEntrypoint` emits the flag only for stamped,
  non-default cells.
- **picker**: `_kv_cells` decodes unstamped cells per backend (vllm ->
  fp8, others -> f16) and the mixed-KV sub-modal/warning generalizes:
  f16/auto count as "full quality", any other dtype carries the
  weaker-long-form-reasoning caveat (with the measured GPQA numbers
  cited only for q8_0, the dtype they were measured on).

Nothing has measured vLLM's fp8-vs-auto quality delta on this fleet
yet -- every existing vLLM bench row (including GPQA) was measured
under fp8 KV, so displayed scores are honest for what serves today.
An fp8-vs-auto A/B on a slack model (fit + LOAD probe + full bench
with GPQA both ways) is the open follow-up.

### Custom vLLM parser plugins

Some models emit tool calls or reasoning in a format that no built-in
vLLM parser handles. The DeepSeek R1 distills are the standing
example: their chat template uses DeepSeek-V3 boundary markers
(`<|tool_call_begin|>` etc.), but they inherit the Qwen2 / Llama-3
tokenizer where those markers aren't atomic vocab entries. vLLM's
built-in `deepseek_v3` / `_v31` / `_v32` parsers do
`vocab.get(<token>)` at startup and crash with HTTP 500 on every
tool-using request.

The fix is a parser plugin: a Python file that registers a parser
with vLLM's `ToolParserManager`, loaded via the
`--tool-parser-plugin <abs-path>` flag. DevAI handles the wiring so
adding a new plugin is a two-step change:

1. Drop the parser file in `scripts/vllm_plugins/`.
2. Add one entry in `deploy/vllm-plugins.json`:

   ```json
   {
     "plugins": {
       "<parser_name>": {
         "kind": "tool",          // or "reasoning"
         "file": "<basename>.py"
       }
     }
   }
   ```

3. Reference `<parser_name>` from a family's `parsers.vllm.tool` (or
   `parsers.vllm.reasoning`) in `scripts/model-families.yaml`, then
   `make catalog-regen` and re-probe.

Both the prober (`scripts/_probe_hf_common.py`) and the router
(`gpu-arbiter/main.go`) read the registry. When a parser name resolves
to a plugin entry they:

- bind-mount `scripts/vllm_plugins/` into the launched vLLM container
  at the registry's `container_dir` (default `/etc/devai/vllm-plugins`);
- emit `--tool-parser-plugin <abs>` (or `--reasoning-parser-plugin`)
  *before* the matching `--tool-call-parser <name>` flag -- vLLM
  resolves parser names at flag-parse time, so the plugin module has
  to be loaded by then.

Names absent from the registry pass through as built-in vLLM parsers
(no plugin flag, no bind-mount). The behaviour for built-ins is
identical to pre-plugin builds.

The router learns the host path of the plugin directory via
`VLLM_PLUGINS_HOST_DIR` (set by the Makefile to
`$(abspath scripts/vllm_plugins)` and exported into compose). When
that env is empty and a model still resolves to a plugin, the router
fails the recreate with an actionable error rather than launching
without the plugin file accessible.

SGLang has no equivalent: SGLang's plugin model is Python-import based
(register a class via SGLang's detector framework), not file-path
based, so a separate plugin would be needed per model. SGLang traffic
for plugin-only families runs without tool support until that lands.

#### Operational notes -- R1-Distill family

Both R1 distills share the same chat template and the same plugin, but
their tool-calling **behaviour** differs sharply because of base-model
training. Verified end-to-end through the router with `tool_choice:
"auto"` and one tool:

| Model | Base tokenizer | `tool_mode` | Completion tokens to call | Reasoning preamble |
|---|---|---|---|---|
| `DeepSeek-R1-Distill-Llama-8B` | Llama-3 | forced | **5** | none -- calls immediately |
| `DeepSeek-R1-Distill-Qwen-7B` | Qwen-2  | forced | ~525 | yes -- long CoT before the call |

Both ended up `tool_mode=forced` (the auto-choice probe didn't elicit
a call), so the router's promote rule kicks in for either. The
**Llama-8B distill is much more usable for tool-calling agents** -- 
it's effectively non-reasoning when handed a single tool. Prefer it
over the Qwen-7B distill when latency matters and the use case
doesn't need reasoning depth. The Qwen-7B distill is better when you
want explicit chain-of-thought, but agents must budget for ~500-token
preamble per tool call.

For multi-tool use cases on either distill, the router rejects with
HTTP 400 (`tool_choice_pinning_required`). Pin client-side or route
to a `tool_mode=auto` model (Qwen3.5-9B-Q8, Qwen3-8B-NVFP4,
Llama-3.1-8B-Instruct-NVFP4) that handles auto choice reliably.

## Cache hygiene

### vLLM / SGLang -- re-probe when sha changes

The HF probe cache is keyed on `<repo>@<sha>` where `sha` is the first
12 chars of HuggingFace's commit SHA at generation time
(`make catalog-regen`). When upstream rebases the repo:

1. `make catalog-regen` produces a new sha.
2. The old cache entry is now an orphan (different key) and ignored.
3. Run `make probe-vllm` / `make probe-sglang` to populate the new key.

Old entries persist until manually pruned. The router synthesizes only
from the latest catalog sha -- old entries don't affect serving.

### vLLM / SGLang -- re-probe when the backend image drifts

Each HF probe run stamps the backend image digest it measured against
into a top-level `_meta` block (`current_image_digest` +
`image_history`). A vLLM/SGLang image tag that moves after probing (a
floating `latest`, or an operator `podman pull`) silently invalidates
the cache: fit, `serving_ok`, and parser values were measured on a
different engine build. This is the rot that made a previously-working
NVFP4 model start crashing at load, which is why the image is now
**pinned** (`VLLM_IMAGE`, `SGLANG_IMAGE` in `.env` / compose).

- Preview drift without touching the stack: `make probe-check` compares
  each cache's `_meta.current_image_digest` against the locally
  available image and exits non-zero if any backend is stale.
- The router does the same check at boot and, for a drifted backend,
  serves anyway but logs a loud warning, sets an `X-DevAI-Warning`
  header, and reports `image_stale` in `/health` (see docs/router.md,
  "Backend image drift").
- Refresh after an intentional image bump:
  `make cache-down && make probe-vllm && make probe-load-vllm`
  (or the `sglang` equivalents). Then re-run `make verify-backend-flags`
  if the engine version changed.

### Ollama -- re-probe when digest changes

Ollama models are identified by their manifest digest. Pulling a new
quantization or alias changes the digest; the prober keys on digest
directly, so re-running `make probe` populates new entries.

## Failure-mode taxonomy

When a probe records `fits: false`, `evidence.kind` tells you why:

| `kind` | Meaning | Action |
|---|---|---|
| `arch` | Model's architecture (`config.json`) is rejected by the backend's runtime. Custom-code archs (e.g. `auto_map`-only models) hit this. | Wait for upstream support, or pick a different model. The probe records `capability: "unsupported_arch"` and the picker hides the row permanently. |
| `quant` | Quantization scheme (FP8/GPTQ/AWQ) not supported on this hardware. | Pick a different quant of the same model. |
| `oom_startup` | Container failed during model load -- weights + KV at requested ctx exceed the GPU memory budget. | Reduce `MAX_CONTEXT_LEN`, pick a smaller quant, or run on a larger GPU. |
| `oom_chat` | Container started but failed on the first chat round-trip -- typically CUDA-graph capture OOM. | Same as `oom_startup`; the budget is too tight for the model + ctx. |
| `clamped_ctx` | Backend silently capped `actual_max_model_len` below the requested ctx -- typically a model with a hard architectural ceiling lower than the operator-requested tier. | Lower the requested ctx tier or accept the cap. |
| `infra` | Container failed for non-model reasons -- image missing nvcc, network error, tokenizer download stall, podman issue. The log excerpt usually shows what. | Fix the environment; this is not a model-fitness signal. |
| `implied_spill` | Larger ctx tier filled in by the prober without launching -- set when a smaller ctx at the same VRAM band already failed. Legacy (multi-cell) only: the single-cell binary search never writes these, since it keeps one winner cell and excludes rather than filling failed tiers. | Skip; smaller ctx fit is the actionable upper bound. |

The `evidence.matched_pattern` field (when present) names the substring
that triggered the classification -- useful for auditing why a particular
launch was tagged a particular way.

The LOAD probe records its verdict separately. A cell that fit but
failed under near-full context gets `serving_ok: false` plus a
`serving_error` string naming the cause -- `request_error:` (the
completion call itself errored), `api_error:` (the backend returned a
500 body), `container_state=exited` (the engine crashed), `oom_marker:`
(an OOM string appeared in the logs while the response carried no
choices), or `implied_by_lower_tier_oom` (a smaller ctx already OOMed,
so this larger tier was marked without launching). These live alongside
the fit-probe `evidence`, never overwriting it.

## Coordination -- only one backend at a time

The router serializes GPU access via `stopOtherBackends`. A request for
backend X drains and stops all other GPU-using backends before X starts.
Concurrent vLLM + SGLang on the same GPU is not supported.

For probing, this constraint is enforced by the prober itself: each
probe driver refuses to run if `devai-router`, `devai-vllm`, or
`devai-sglang` is up. Always `make cache-down` before probing.

## Per-session context binding & reasoning overrides

When the picker selects a model + context tier (+ reasoning override), each backend handles the binding differently:

- **Ollama**: the picker emits the parent name (e.g., `qwen3.5:9b-q8_0`), or with a reasoning override suffix (e.g., `qwen3.5:9b-q8_0::nothink` to suppress thinking even if the model supports it). KV cache is allocated dynamically per request from the `OLLAMA_CONTEXT_LENGTH` global ceiling (default 256K). The `/api/chat` and `/api/generate` endpoints honour `options.num_ctx` injected by the router; the OpenAI- and Anthropic-compat layers ignore it and use the global ceiling. A bare `<name>` is served from whatever tier is currently loaded -- only an explicit `<name>@<ctx>` pin (which the picker emits for mixed-KV models) can force a tier switch and the container recreate that goes with it.
- **vLLM / SGLang**: the picker emits `<name>@<ctx>` (e.g., `Llama-3.1-8B-Instruct-NVFP4@32768`), or with a reasoning override suffix (e.g., `Llama-3.1-8B-Instruct-NVFP4::low@32768` to set reasoning effort to `low`). The router's `peelControlSuffixes` strips `@<ctx>`, `::<mtp>` and `::<reasoning>` in whatever order they trail, rewrites the body's `model` field to the clean name, applies the reasoning policy, and triggers a recreate when `currentContext` (or the MTP spec) differs. A reasoning change alone never recreates the container. No client-side tag materialization needed.

Valid reasoning suffixes: `::off`, `::auto`, `::low`, `::medium`, `::high`, `::nothink` (Ollama only; suppresses thinking).

Both flows are transparent to the agent CLI -- the picker emits the right serving name and the router handles parsing and lifecycle management.
