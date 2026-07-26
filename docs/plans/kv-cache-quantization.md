# KV-cache quantization

> **SUPERSEDED 2026-07-25 -- do not execute this plan as written.**
>
> The engine-side substance of Phases 1-3 shipped independently, under a
> better design than this plan proposed: per-probe-cell stamped KV dtype
> with no global policy (commits 8325255, 664bc76, 38facb5).
>
> **Phase 2 must NOT be executed.** Its headline -- flip the Ollama
> default to `q8_0` globally -- is now contradicted by the project's own
> measurement: q8_0 costs roughly 12 GPQA points on long reasoning
> chains, which is why it shipped as a per-tier opt-in with a picker
> warning instead. See `docs/backends.md`.
>
> The one part that had not shipped, backend-aware fit math (Phase 1
> step 4), was extracted and shipped on 2026-07-25: `select-models.py`
> now costs KV per backend rather than assuming fp16 everywhere.


_Bring KV-cache quantization to parity across all three backends, fix the fit-math that silently assumes fp16 KV everywhere, and promote KV dtype to a first-class per-model / per-request knob -- so a 24 GiB card serves longer contexts at a measured, not assumed, quality cost._

## Status

Draft. Not yet scheduled for execution.

## Dependencies

- **None hard.** This is a single-mode router + tooling feature. It is
  independent of sops-age, MCP gateway, cluster mode, and SkyPilot.
- **Soft (Phase 3 only):** the bench-cache row-key extension reuses
  bench-rewrite's per-ctx composite-key + idempotent v2->v3 migration
  pattern (`docs/plans/bench-rewrite.md`, shipped through Phase 5 as of
  2026-05-15). Phase 3 assumes that v3 schema is already in place.

## Enables / Unblocks

- **Longer contexts on 24 GiB for Ollama and SGLang.** vLLM already gets
  the fp8-KV win (`gpu-arbiter/main.go:853`); SGLang and Ollama do not.
  This closes that gap so the picker's fitting set is not artificially
  biased toward vLLM at high context.
- **Accurate picker / selector VRAM numbers.** Today `select-models.py`
  defaults `--kv-dtype fp16` (`scripts/select-models.py:1328`) and the
  picker computes KV "at fp16" (`scripts/model-picker.py:700-712`) while
  vLLM serves at fp8 -- so every vLLM row's formula-path VRAM is
  over-counted by the full KV delta (up to ~9.4 GiB at 128K for the
  reference model). Correct math is a prerequisite for any later
  context-maximization decision.
- **A measured KV quality/VRAM/context tradeoff surface.** The Phase 2
  bench sweep produces the project's *own* q8_0 / q4_0 / fp8 numbers
  instead of trusting the upstream "~50% / ~75% cut" community claims, so
  future model onboarding can choose a KV dtype on evidence.

## Out of scope

This section is load-bearing. Adding any of the following under this
plan's banner would change its risk profile:

- **lmdeploy as a fourth backend / INT8 KV (research hint #3).** lmdeploy's
  `kv_int8` is its own engine. devai's two safetensors backends already
  expose the equivalent "compress KV" lever as fp8 (vLLM `--kv-cache-dtype
  fp8`, SGLang `--kv-cache-dtype fp8_e5m2|fp8_e4m3`); neither exposes an
  int8 KV mode. Adding lmdeploy means a new image, a new probe driver
  (`scripts/_probe_hf_common.py` BackendSpec), a new entrypoint builder,
  new parser plumbing, and a fourth GPU-exclusion path. That is a
  "add-a-backend" plan, not a KV-cache plan. Captured here so a future
  contributor does not smuggle it in.
- **KV-cache offloading to CPU / disk.** A different lever than
  quantization (it trades bandwidth, not bits). Orthogonal; separate plan.
- **Weight quantization** (NVFP4 / MXFP4 / GGUF quant). This plan touches
  only the KV cache; weight formats are unchanged.
- **Prefix-cache / RadixAttention tuning.** Orthogonal reuse mechanism.
- **Per-request KV dtype for Ollama.** `OLLAMA_KV_CACHE_TYPE` and
  `OLLAMA_FLASH_ATTENTION` are daemon-global; Ollama cannot vary KV dtype
  per request without recreating the (always-on) container. The
  per-request `::kv=` knob (Phase 3) is therefore vLLM/SGLang-only by
  construction -- see Decision D4.
- **Calibrated fp8_e4m3 scale checkpoints.** Producing per-tensor KV
  scales is a model-prep step (offline calibration), not a serve-time
  knob. e4m3 stays opt-in per-model (D3); the default fp8 paths use
  dynamic scaling, which needs no calibration.

## Decisions

- **D1 -- KV dtype is a per-backend default, not a global constant.**
  vLLM stays `fp8` (unchanged). SGLang gains `fp8_e5m2`. Ollama gains a
  `OLLAMA_KV_CACHE_TYPE` knob whose default flips from `f16` to `q8_0`
  only after the Phase 1 probe validates it (D5). The single global
  `select-models.py --kv-dtype` arg is replaced by a per-model derivation
  keyed on the model's backend.
- **D2 -- Fit-math correctness ships first (Phase 1) and is the source of
  truth.** `select-models.vram_breakdown`, the picker `_hf_kv_gb`, and the
  router `fittableContext` heuristic are all made backend-aware before any
  context-maximization work. A wrong VRAM estimate poisons every
  downstream decision (picker eligibility, probe ceilings), so it is the
  foundation, not a follow-on.
- **D3 -- SGLang fp8 variant defaults to `fp8_e5m2`.** e5m2 needs no
  calibration scales and is SGLang's documented safe default; e4m3 (better
  accuracy, needs scales) is opt-in per-model via the Phase 3 override.
  (Open Q1.)
- **D4 -- Per-request `::kv=<dtype>` is honored for vLLM/SGLang only.**
  Those backends recreate the container per (model, ctx) already, so KV
  dtype rides the same recreate. For Ollama the suffix is a documented
  no-op (daemon-global constraint); the Ollama KV dtype is set via the
  global env and surfaced in the picker as informational.
- **D5 -- The Ollama default flip to `q8_0` is gated on evidence.** Until
  a Phase 1 probe confirms flash-attention + q8_0 loads and serves every
  catalog model the picker would show, the default stays `f16`
  (byte-identical to today). The probe produces the project's own
  fit/quality data; the "~50% cut, near-lossless" claim from the hints is
  treated as a hypothesis to verify, not a fact to ship.
- **D6 -- KV dtype enters the probe-cache cell and the bench-cache row
  key.** A probe cell becomes `(vram, ctx, kv_dtype) -> fits`; a bench row
  key gains a `::<kv_dtype>` suffix. Both reuse the established
  "re-probe/re-bench, never interpolate" doctrine (`docs/backends.md`) and
  bench-rewrite's migration pattern. No silent cross-dtype substitution.
- **D7 -- The per-model override is a dedicated `kv_cache_dtype` field in
  `deploy/recovery-flags.json`, not a raw flag in `engine_flags`.** The
  router emits exactly one `--kv-cache-dtype`, and the probers read the
  same field, so probe-time and serve-time memory math cannot drift.
  Relying on argparse last-wins with a duplicated flag is rejected.
- **D8 -- Phase order is correctness -> maximization -> per-request.** Each
  phase is independently shippable and leaves the system in a coherent
  state; later phases are pure additions, never rewrites of earlier ones.

## Open questions

1. **SGLang e5m2 vs e4m3 default.** Recommendation: **e5m2** (D3) -- no
   calibration, predictable, matches SGLang's own default posture. Revisit
   per-model via the Phase 3 override if a specific checkpoint shows
   measurable accuracy loss under e5m2 in the Phase 2 bench.
2. **When to flip the Ollama default to q8_0.** Recommendation: ship the
   *knob* in Phase 1 with default `f16` (no behavior change), and make the
   flip to `q8_0` the first action of Phase 2, gated on the Phase 1 probe
   and a Phase 2 quality bench. This keeps Phase 1 a pure
   correctness/parity change with zero output-quality risk.
3. **Re-validate the existing vLLM fp8 default?** vLLM already serves fp8
   KV with an undocumented-magnitude quality cost (`nvfp4-number-formats.md`
   calls it a "small per-token quality cost"). Recommendation: include
   `fp8` in the Phase 2 bench sweep for completeness so the cost is
   *quantified*, but treat it as non-blocking -- it is already shipped.

## Context

Three external data points (the research hints that prompted this plan):

- llama.cpp / Ollama / LM Studio expose `--cache-type-k` / `--cache-type-v`
  (`-ctk` / `-ctv`); `q8_0` is the everyday near-lossless setting (~50% KV
  VRAM cut), `q4_0` a more aggressive, task-dependent one (~75% cut).
- vLLM ships production fp8 (E4M3) KV cache, ~2x compression, native on
  Hopper/Blackwell/MI300.
- lmdeploy ships INT8 KV cache.

Mapping those onto devai's current state (verified in the tree, not
assumed):

| Backend | KV dtype today | Where | Gap |
| ------- | -------------- | ----- | --- |
| vLLM    | `fp8` (hardcoded) | `gpu-arbiter/main.go:853`, `scripts/probe-vllm-reasoning.py:82` | none for serve; not per-model configurable |
| SGLang  | engine default (`auto` = model dtype, i.e. bf16/fp16) | `sglangEntrypoint` `gpu-arbiter/main.go:976-1007` -- **no `--kv-cache-dtype`** | no fp8 parity; serves less ctx than vLLM for the same model |
| Ollama  | `f16` (no quantization) | `deploy/docker-compose.yaml:23-41` -- no `OLLAMA_FLASH_ATTENTION` / `OLLAMA_KV_CACHE_TYPE` | research hint #1 entirely unrealized |

And a latent correctness bug independent of the above: the fit-math
assumes fp16 KV uniformly while vLLM serves fp8.

- `scripts/select-models.py:52` `KV_BYTES = {"fp16":2,"bf16":2,"fp8":1,"int8":1}`,
  `:1328` `--kv-dtype ... default="fp16"` -- one global arg for all
  backends.
- `scripts/model-picker.py:700-712` `_hf_kv_gb(...)` docstring: "KV cache
  size in GB for a given (arch, ctx) at fp16."
- `gpu-arbiter/main.go:737-738` and `:753-754` comment: "Both vLLM and
  SGLang store KV cache in BF16 regardless of weight quantization." This
  is **stale for vLLM** (it passes `--kv-cache-dtype fp8` 100 lines later)
  and only accidentally correct for SGLang (which has no flag yet).

The cost, in the project's own numbers (reference model
`nvidia/Qwen3-8B-NVFP4`: 36 layers x 8 KV heads x 128 head_dim x 2(K+V),
GQA -- `docs/attention-and-the-transformer.md:240-254`):

| KV dtype | bytes/elem | KB/token | KV @ 128K |
| -------- | ---------- | -------- | --------- |
| fp16/bf16 | 2 | 144 | ~18.9 GiB |
| fp8 (current vLLM) | 1 | 72 | 9.44 GiB (`nvfp4-coldstart.md:206-211`) |

GQA is "the single biggest reason a 128 K context is feasible on a 24 GB
card" (`attention-and-the-transformer.md:240-254`); fp8 KV is the second.
The paged pool sizing in `paged-attention-and-vllm-internals.md:188-192`
(~11,950 blocks, ~191K tokens capacity) is computed at 1 byte/element --
i.e. it already bakes in fp8. The Ollama GGUF models have different
architectures and the project has **no** measured GGUF KV-quant numbers
yet; producing them is exactly what Phase 1's probe and Phase 2's bench
do.

**Honesty note (this plan was written without a live GPU host).** The
following are from upstream docs, not verified against the pinned images
or live traffic, and each has an explicit validation gate below:

- the SGLang flag is `--kv-cache-dtype` with values `fp8_e5m2` / `fp8_e4m3`
  (gate: `make verify-backend-flags` against `v0.5.10.post1-cu130`);
- Ollama honors `OLLAMA_FLASH_ATTENTION=1` + `OLLAMA_KV_CACHE_TYPE=q8_0`
  and which catalog models tolerate it (gate: Phase 1 probe pass);
- the q8_0 ~50% / q4_0 ~75% cuts and their quality impact (gate: Phase 2
  bench).

Until those gates pass, the plan ships only behavior-preserving changes.

## Approach

Three phases, each independently shippable, in correctness-first order:

1. **Phase 1 -- parity + fit-math correctness.** SGLang reaches fp8
   parity; Ollama gains the KV-type knob (default unchanged); the
   selector, picker, and router KV math become backend-aware; docs are
   corrected. Default output behavior is unchanged (vLLM untouched, Ollama
   default still f16); only SGLang's served context grows.
2. **Phase 2 -- context maximization (opt-in, evidence-gated).** A global
   `DEVAI_KV_CACHE_DTYPE` knob, the Ollama q8_0 default flip, the q4_0
   aggressive mode behind a bench quality-gate, and a re-probe of the
   (VRAM, ctx) matrix so newly-fitting larger contexts become selectable.
3. **Phase 3 -- first-class per-model / per-request KV dtype.** A
   `kv_cache_dtype` field in `recovery-flags.json`, a `::kv=<dtype>`
   request suffix in the router rewrite chain (vLLM/SGLang), a picker
   column + toggle, and KV dtype threaded into the probe-cache cell and
   bench-cache row key (with migration).

---

## Phase 1 -- parity + fit-math correctness

### Goal

Make every backend's KV dtype explicit and consistent, and make the
VRAM-fit math reflect what is actually served. No change to default
output behavior: vLLM stays fp8, Ollama default stays f16; SGLang gains
fp8 KV (its only behavior change, and a strictly-more-context one).

### Deliverables

```
deploy/backend-flags.yaml          modify -- pin sglang.kv_cache_dtype: "--kv-cache-dtype" (+ vllm, for symmetry/verify coverage)
gpu-arbiter/main.go                modify -- sglangEntrypoint emits --kv-cache-dtype fp8_e5m2; fix fittableContext stale BF16 comment + make per-token KB backend-aware
gpu-arbiter/main_test.go           modify -- assert SGLang entrypoint carries the KV flag; vLLM unchanged
scripts/probe-sglang-reasoning.py  modify -- add --kv-cache-dtype fp8_e5m2 so probe fit data matches serve-time
scripts/select-models.py           modify -- KV_BYTES -> floats incl q8_0/q4_0; kv_dtype derived per-model from backend (drop single global default); update --kv-dtype to an override, not the source of truth
scripts/model-picker.py            modify -- _hf_kv_gb uses fp8 for vLLM/SGLang rows (not fp16); note Ollama KV dtype
deploy/docker-compose.yaml         modify -- add OLLAMA_FLASH_ATTENTION + OLLAMA_KV_CACHE_TYPE env (defaults: 0 / f16 = no change)
.env.example                       modify -- document OLLAMA_KV_CACHE_TYPE, OLLAMA_FLASH_ATTENTION
docs/router.md                     modify -- KV section: per-backend dtype table (not vLLM-only)
docs/backends.md                   modify -- same; note SGLang fp8 parity + Ollama knob
docs/nvfp4-number-formats.md       modify -- add a short "integer KV quant (q8_0/q4_0) -- the llama.cpp/Ollama lever" subsection so the doc is not vLLM-fp8-only
tests/python/                      modify -- fit-math unit tests: per-backend kv_dtype selection; KV_BYTES float math
```

### Detailed steps

1. **Verify the SGLang flag name before writing any launch code.** Run
   `make verify-backend-flags` after adding `kv_cache_dtype:
   "--kv-cache-dtype"` to the `sglang:` block in
   `deploy/backend-flags.yaml`. If the pinned image
   (`v0.5.10.post1-cu130`) does not expose it under that name, stop and
   reconcile -- do not guess. (This is the same drift-guard discipline the
   file already documents.)
2. **SGLang entrypoint parity.** In `sglangEntrypoint`
   (`gpu-arbiter/main.go:976`), add `"--kv-cache-dtype", "fp8_e5m2"` to the
   base args, mirroring vLLM's line at `:853`. Keep it ahead of the
   per-model recovery flags so a Phase 3 override can last-wins it.
3. **Match the probe.** Add the same flag to
   `scripts/probe-sglang-reasoning.py` (next to `--mem-fraction-static` /
   `--context-length` at `:91-92`) so SGLang probe-cache ceilings are
   measured at fp8, consistent with serve-time (exactly the invariant the
   vLLM probe already honors, `docs/router.md:356`).
4. **Fix the fit-math (the load-bearing change).**
   - `scripts/select-models.py`: change `KV_BYTES` to floats and add the
     integer-KV entries:
     ```python
     KV_BYTES = {"fp16": 2.0, "bf16": 2.0, "fp8": 1.0,
                 "int8": 1.0, "q8_0": 1.0, "q4_0": 0.56}
     ```
     (`q8_0` ~= 8 bits + a per-block fp16 scale ~= 1.06 B/elem, rounded to
     1.0; `q4_0` ~= 4.5 bits effective ~= 0.56 B/elem.) Make
     `kv_per_token_bytes` return a float. Replace the single
     `--kv-dtype`-for-everything (`:1328`) with `kv_dtype_for(model)` that
     maps backend -> default dtype (`vllm`/`sglang` -> `fp8`, `ollama` ->
     the configured `OLLAMA_KV_CACHE_TYPE`, default `f16`); keep `--kv-dtype`
     as an explicit override for what-if analysis only.
   - `scripts/model-picker.py:700-712`: `_hf_kv_gb` is HF-only -- switch
     its constant from `_KV_BYTES_FP16 = 2` to fp8 (1 byte) since both HF
     backends now serve fp8. Update the docstring.
   - `gpu-arbiter/main.go:737-786`: rewrite the stale "Both vLLM and SGLang
     store KV cache in BF16" comment (now both store fp8), and halve the
     per-token KB table for vLLM/SGLang (the heuristic is only a fallback
     when probe data is absent -- `applyProbeCeiling` still wins when it
     exists -- but it should not lie).
5. **Ollama knob, default off.** Add to the `ollama` service in
   `deploy/docker-compose.yaml:27-34`:
   ```yaml
   - OLLAMA_FLASH_ATTENTION=${OLLAMA_FLASH_ATTENTION:-0}
   - OLLAMA_KV_CACHE_TYPE=${OLLAMA_KV_CACHE_TYPE:-f16}
   ```
   Defaults reproduce today's behavior exactly. Document both in
   `.env.example` next to `OLLAMA_CONTEXT_LENGTH`, including the hard
   constraint that q8_0/q4_0 require `OLLAMA_FLASH_ATTENTION=1` and that
   both are daemon-global (apply to every loaded Ollama model).
6. **Docs.** Replace the vLLM-only KV paragraph in `docs/router.md:349-357`
   and `docs/backends.md:47` with a per-backend dtype table. Add the
   integer-KV subsection to `docs/nvfp4-number-formats.md` so its format
   tour is not silent on the llama.cpp/Ollama lever (it currently never
   mentions q8_0/q4_0/int8 KV).

### Exit criteria

- `make verify-backend-flags` passes with the new pinned SGLang flag.
- `make test-router` shows `sglangEntrypoint` emits `--kv-cache-dtype
  fp8_e5m2` and `vllmEntrypoint` is byte-identical to before.
- `make test-python` covers: `kv_dtype_for(model)` returns `fp8` for
  vLLM/SGLang rows and the configured Ollama type for Ollama rows; the
  KV_BYTES float math; and a regression asserting a vLLM row's
  formula-path VRAM at 128K dropped by the fp16->fp8 delta.
- With `OLLAMA_FLASH_ATTENTION` unset, the Ollama container env is
  identical to today (no quantization).
- Docs no longer claim "both store BF16" anywhere; the per-backend table
  is present.

### Phase 1 risks

| Risk | Mitigation |
| ---- | ---------- |
| SGLang flag name differs on the pinned image | Step 1 gate (`verify-backend-flags`) blocks before any code lands |
| SGLang fp8_e5m2 changes existing SGLang probe ceilings, invalidating cached cells | Expected and correct -- the cache is re-probed (`make probe-sglang`); flagged in the PR as a required re-probe, not silent |
| Halving the router heuristic over-promises ctx when probe data is absent | Heuristic is fallback-only; `applyProbeCeiling` overrides whenever probe cells exist, and the picker only shows probe-confirmed models |
| Ollama env added but operator sets q8_0 without flash-attention | `.env.example` states the dependency; Phase 2's probe is what actually validates the combination before any default flip |

---

## Phase 2 -- context maximization (opt-in, evidence-gated)

### Goal

Turn the now-correct math into more usable context: validate and flip the
Ollama default to q8_0, add the aggressive q4_0 mode behind a measured
quality gate, expose a global override, and re-probe so larger contexts
that now fit become selectable.

### Deliverables

```
scripts/probe-ollama-reasoning.py  modify -- probe under OLLAMA_FLASH_ATTENTION=1 at q8_0 and q4_0; record load/serve success + the kv_dtype used
scripts/_probe_core.py             modify -- probe cell carries kv_dtype (D6); fit cells become (vram, ctx, kv_dtype)
deploy/.ollama-reasoning-cache.json (data) -- repopulated by make probe at the chosen dtype(s)
scripts/bench/bench_runner.py      modify -- accept a kv-dtype dimension; sweep {f16,q8_0,q4_0} for Ollama and {fp8} for vLLM/SGLang (Open Q3)
scripts/bench/_bench_core.py       modify -- record kv_dtype in the row (precursor to the Phase 3 key change)
deploy/docker-compose.yaml         modify -- DEVAI_KV_CACHE_DTYPE global knob plumbed to the router (vLLM/SGLang default fp8) + OLLAMA_KV_CACHE_TYPE default flip to q8_0 (gated)
gpu-arbiter/main.go                modify -- read DEVAI_KV_CACHE_DTYPE as the per-backend default (overridable per-model in Phase 3)
docs/bench-results.md              modify -- add the KV-dtype quality/VRAM/context comparison table
docs/nvfp4-coldstart.md            modify -- note how the budget shifts at q8_0/q4_0 for GGUF models
TODO.md                            modify -- record the re-probe + re-bench obligation
```

### Detailed steps

1. **Probe Ollama under flash-attention.** Extend
   `scripts/probe-ollama-reasoning.py` to run a pass with
   `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE` set to `q8_0`
   (and a separate `q4_0` pass). Record, per catalog model: did it load,
   did it serve a clean 3-chat probe, and the resulting `fully_on_gpu` /
   VRAM at each (vram, ctx) cell. Models that fail flash-attention are
   recorded as such -- they keep f16.
2. **Quality gate (the q4_0 condition).** Before q4_0 is offered for a
   model, the Phase 2 bench must show its `REAS%` and `CODE%` stay within
   a declared delta of the f16/q8_0 baseline (proposed: REAS% drop <= 3
   points, CODE% drop <= 2 points -- numbers to confirm from the first
   sweep, not asserted here). q8_0 is expected to clear this trivially;
   q4_0 is where the gate earns its keep. A model that fails the gate is
   simply not offered q4_0.
3. **Flip the Ollama default (Open Q2).** Once the q8_0 probe + bench pass
   across the picker-visible catalog, change the compose default
   `OLLAMA_KV_CACHE_TYPE` from `f16` to `q8_0`. This is the one
   intentional default-output change in the plan; it is justified by the
   project's own bench, not the community claim.
4. **Global override.** Add `DEVAI_KV_CACHE_DTYPE` (per-backend default,
   parsed in the router next to `MAX_CONTEXT_LEN` / `DEVAI_REASONING`).
   Empty -> the per-backend defaults from D1.
5. **Re-probe the matrix.** Larger contexts now fit at q8_0/fp8; run
   `make probe` / `make probe-sglang` so the new ceilings populate. Per
   `docs/backends.md` doctrine there is no interpolation -- gaps mean
   "re-probe".
6. **Bench sweep.** Run the KV-dtype sweep and render the comparison into
   `docs/bench-results.md`: for each (model, ctx), the VRAM, sustained
   tok/s, and task scores at each KV dtype, so the tradeoff is a table, not
   a claim.

### Exit criteria

- `deploy/.ollama-reasoning-cache.json` has q8_0 (and where gated-in,
  q4_0) cells with `kv_dtype` recorded; the picker shows the larger
  contexts that now fit.
- `docs/bench-results.md` has a KV-dtype comparison table sourced from a
  real run on the reference host (or the phase is explicitly marked
  "pending GPU host" like bench-rewrite Phase 6).
- The Ollama default is `q8_0` only if the gate passed; otherwise the
  flip is deferred with a one-line note saying which model blocked it.
- `DEVAI_KV_CACHE_DTYPE` unset reproduces the D1 per-backend defaults.

### Phase 2 risks

| Risk | Mitigation |
| ---- | ---------- |
| q4_0 silently degrades a model's reasoning/code quality | Hard bench gate (step 2); q4_0 is opt-in per model and never the default; gate thresholds are published in `bench-results.md` |
| Flash-attention breaks a specific Ollama model/arch | Probe (step 1) records per-model FA success; failing models stay f16; default flip is catalog-wide-gated |
| Re-probe is long machine time on one GPU | Same cost profile as the existing probe matrix; scoped to the dtypes actually being offered, not a full cross-product |
| Bench numbers unavailable without a GPU host | Mark the phase pending-host (precedent: bench-rewrite Phase 6) rather than shipping fabricated numbers |

---

## Phase 3 -- first-class per-model / per-request KV dtype

### Goal

Treat KV dtype like reasoning and context: a per-model override an
operator can pin, and a per-request suffix an agent can pass -- threaded
through the router rewrite chain, the picker, and both caches, with the
same "no silent substitution" discipline.

### Deliverables

```
deploy/recovery-flags.json         modify -- add optional models.<name>.kv_cache_dtype (D7)
gpu-arbiter/recovery_flags.go      modify -- parse kv_cache_dtype; expose on launchConfig
gpu-arbiter/main.go                modify -- launchConfig.KVCacheDtype overrides the base flag in vllm/sglangEntrypoint (single emission); parse ::kv=<dtype> in the override stage
gpu-arbiter/parse_minimal.go       modify -- head-side: strip ::kv=<dtype> alongside @<ctx> / ::<reasoning> / ::<mtp>
gpu-arbiter/main_test.go           modify -- override precedence (request ::kv > recovery-flags > backend default); Ollama ::kv is a no-op
scripts/_probe_core.py             modify -- probe cell key/value already carries kv_dtype (Phase 2); finalize lookup by (vram, ctx, kv_dtype)
scripts/bench/_bench_core.py       modify -- row key gains ::<kv_dtype> suffix; idempotent migration mapping legacy rows (mirror bench-rewrite v2->v3)
scripts/bench/bench_report.py      modify -- KV column in the leaderboard
scripts/model-picker.py            modify -- KV column + post-pick sub-modal toggle (vLLM/SGLang); emits <name>::kv=<dtype>@<ctx>
docs/router.md                     modify -- document ::kv= in the rewrite-chain "Override parsing" step + precedence
docs/backends.md                   modify -- per-model kv_cache_dtype + the Ollama daemon-global caveat
tests/python/                      modify -- bench/probe schema migration tests; picker emits the suffix; report renders the column
```

### Detailed steps

1. **Per-model override (recovery-flags).** Add an optional field:
   ```json
   "models": {
     "SomeModel-NVFP4": { "kv_cache_dtype": "fp8_e4m3" }
   }
   ```
   `gpu-arbiter/recovery_flags.go` parses it onto `launchConfig`;
   `vllmEntrypoint` / `sglangEntrypoint` emit the base `--kv-cache-dtype`
   once, substituting the override when present (D7 -- one flag, never a
   duplicate). The probers read the same field, so probe and serve agree.
2. **Per-request suffix.** Extend the override-parsing stage (step 1 of
   the rewrite chain, `docs/router.md`; `parseCtxOverride` /
   `parseReasoningOverride` in `gpu-arbiter/main.go`) to also strip
   `::kv=<dtype>` (grammar: `<name>::kv=fp8_e5m2@131072`, parsed alongside
   the existing `@<ctx>` and `::<reasoning>` / `::<mtp>` suffixes; define
   and document order). Precedence: request `::kv=` > recovery-flags
   `kv_cache_dtype` > `DEVAI_KV_CACHE_DTYPE` > backend default. For Ollama
   the suffix is parsed-and-ignored with a one-line log (D4) -- it cannot
   change the daemon-global type per request.
3. **Cluster head parity.** `parse_minimal.go` must strip `::kv=` so a
   head forwards a clean `model` to the worker (mirrors how it already
   strips `@<ctx>` / `::<reasoning>`); the worker honors it on recreate.
4. **Probe-cache cell.** Finalize the Phase 2 `kv_dtype`-carrying cell so
   lookups are `(vram, ctx, kv_dtype) -> fits`. A request for a dtype with
   no probed cell falls back to the documented "re-probe" gap, never to a
   different dtype's number.
5. **Bench-cache key.** Extend the row key from
   `<repo>@<sha>::<backend>::<ctx>` to
   `<repo>@<sha>::<backend>::<ctx>::<kv_dtype>`, with an idempotent
   migration that maps existing rows to their implicit dtype (vLLM rows ->
   `fp8`, Ollama legacy rows -> `f16`). This is a direct reuse of
   bench-rewrite's v2->v3 composite-key migration; legacy rows with no
   determinable dtype land in a documented bucket, and re-bench is the
   answer (same posture as bench-rewrite's `RECOVERED_CTX_MAP`).
6. **Picker.** Add a KV column (or fold into the FORMAT/notes pane) and a
   post-pick sub-modal toggle for vLLM/SGLang rows, mirroring the existing
   reasoning ON/OFF sub-modal. The picker emits `<name>::kv=<dtype>@<ctx>`;
   for Ollama the toggle is informational (points the operator at the
   global env), per D4.

### Exit criteria

- `make test-router`: precedence chain holds (request > recovery-flags >
  global > backend default); a vLLM/SGLang launch emits exactly one
  `--kv-cache-dtype`; an Ollama `::kv=` request logs a no-op and serves
  normally.
- `make test-python`: bench/probe schema migrations are idempotent and
  byte-stable on re-run; the picker emits the suffix; the report renders
  the KV column.
- `make test-cluster-preflight` (or its unit subset): a head strips
  `::kv=` before forwarding.
- A probe lookup for an un-probed (vram, ctx, kv_dtype) returns "no cell"
  (re-probe), not a substituted number.

### Phase 3 risks

| Risk | Mitigation |
| ---- | ---------- |
| Suffix grammar collides with `@<ctx>` / `::<reasoning>` / `::<mtp>` parsing | Define and unit-test parse order explicitly; reuse the existing strip helpers rather than a parallel parser |
| Bench/probe schema migration corrupts existing rows | Idempotent migration with a unit test asserting byte-stable re-run; legacy-bucket + re-bench, exactly as bench-rewrite did |
| Operator expects per-request KV on Ollama | `::kv=` documented as a no-op for Ollama; picker toggle is informational; log line states it explicitly |
| Duplicate `--kv-cache-dtype` from a stray engine_flags entry | D7 forbids it; the override is a dedicated field, and a test asserts single emission |

---

## Combined risk register

| Risk | Phase | Mitigation |
| ---- | ----- | ---------- |
| Shipping upstream KV claims as fact | all | Every external claim (SGLang flag, Ollama FA+q8_0, q8_0/q4_0 cuts, fp8 quality cost) has an explicit gate: `verify-backend-flags`, a probe pass, or a bench, before it changes a default |
| Output-quality regression from quantized KV | 2-3 | q8_0 default flip is catalog-wide bench-gated; q4_0 is per-model gated and opt-in; vLLM fp8 is grandfathered but quantified |
| Fit-math/serve-time drift (the original bug) | 1 | Probers and entrypoints read the same dtype source (flag in P1, recovery-flags field in P3); a regression test ties formula-path VRAM to the served dtype |
| Cache staleness after a dtype change | 1-3 | "Re-probe / re-bench, never interpolate" doctrine; un-probed cells return a gap, not a guess; PRs flag required re-probes |
| Cluster head forwards a dirty model string | 3 | `parse_minimal.go` strips `::kv=`; preflight covers it |
| Scope creep into lmdeploy / INT8 KV | all | Explicit Out-of-scope entry with rationale |

## Migration / rollback story

- **Phase 1** is behavior-preserving for vLLM and Ollama (defaults
  unchanged) and strictly more-context for SGLang. The Ollama env knobs
  default to today's values. Rollback = revert the PR; the only data
  effect is a recommended SGLang re-probe (the old cells are simply
  measured at the wrong dtype, not corrupted).
- **Phase 2** introduces the one intentional default change (Ollama
  `f16 -> q8_0`), gated on evidence and reversible by setting
  `OLLAMA_KV_CACHE_TYPE=f16`. `DEVAI_KV_CACHE_DTYPE` unset = D1 defaults.
- **Phase 3** is additive: no override anywhere reproduces Phase 2
  behavior exactly. The bench/probe schema bumps carry idempotent
  migrations; rollback re-runs the migration in reverse is unnecessary
  because readers tolerate the new key (forward-compatible `omitempty`
  posture, as elsewhere in the project).

## Estimated effort

| Phase | Engineering effort | Wall-clock | GPU-host time |
| ----- | ------------------ | ---------- | ------------- |
| Phase 1 | 1-2 PRs: SGLang flag, fit-math, Ollama env, docs, tests | 1-2 days | 1 SGLang re-probe |
| Phase 2 | 1-2 PRs: probe FA passes, bench sweep, default flip, global knob | 2-4 days | re-probe matrix + bench sweep (hours-to-overnight) |
| Phase 3 | 2-3 PRs: recovery-flags field, ::kv= parse, picker, cache schema + migration, head parse | 3-5 days | re-probe per offered dtype |
| Total | 4-7 PRs | ~6-11 days code | substantial probe/bench machine time |

The code is modest; the long pole is GPU-host probe + bench time, which
(like bench-rewrite Phase 6) can be deferred and the phase marked
pending-host rather than blocking on it.

## References

- Research hints (prior art): llama.cpp/Ollama `--cache-type-k/-v`
  (`-ctk/-ctv`) q8_0/q4_0; vLLM fp8 (E4M3) KV
  (docs.vllm.ai quantized_kvcache); lmdeploy INT8 KV.
- `gpu-arbiter/main.go` -- `:853` vLLM fp8 (the parity target), `:976`
  `sglangEntrypoint` (the gap), `:704` `memFraction`, `:766`
  `fittableContext` (stale comment), `:817` `computeLaunchConfig`.
- `scripts/select-models.py:52,106,115,153,1328` -- the fp16-everywhere
  fit-math this plan corrects.
- `scripts/model-picker.py:700-712` -- the picker's fp16 KV formula.
- `scripts/probe-vllm-reasoning.py:82` / `scripts/probe-sglang-reasoning.py:91`
  -- the probe flags that must mirror serve-time.
- `deploy/backend-flags.yaml` -- the flag-name pinning + `verify-backend-flags`
  gate; `deploy/recovery-flags.json` -- the per-model override precedent (D7).
- `docs/router.md:349-357`, `docs/backends.md:47` -- the vLLM-only KV docs
  this plan generalizes to all backends.
- `docs/nvfp4-coldstart.md:193-211,369-371`, `docs/nvfp4-number-formats.md:385-398`,
  `docs/attention-and-the-transformer.md:240-254`,
  `docs/paged-attention-and-vllm-internals.md:188-192` -- the KV budget
  numbers cited in Context.
- `docs/plans/bench-rewrite.md` -- the per-ctx composite-key + idempotent
  migration pattern Phase 3 reuses for the bench-cache `::<kv_dtype>` key.
- `scripts/model-families.yaml:60-72` -- SGLang NVFP4 broken upstream
  (why SGLang fp8 parity helps only its BF16/FP16 safetensors models today).
