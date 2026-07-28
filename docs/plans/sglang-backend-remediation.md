# SGLang backend remediation

_Repair the SGLang code paths by extending the two backend contracts that already exist -- not by rewriting them -- and stop the live Ornith crash loop first._

## Status

Draft. Not yet scheduled for execution.

The pre-work evidence phase (Phase 0 below) has already been RUN, read-only, on
2026-07-27, and its results are recorded here. Nothing else has been started and no
code has been modified.

## Dependencies

None. Every phase is independently shippable and nothing here blocks on another plan.

Two adjacent plans overlap and should be reconciled, not waited on:

- [Plan: router-anthropic-messages-compat](./router-anthropic-messages-compat.md) --
  its open questions 1 and 4 (`does SGLang expose /v1/messages`, `does its Anthropic
  shim behave like vLLM's`) are ANSWERED by Phase 0 below. That plan's impact table
  row `| SGLang | same | NOT TESTED |` can be filled in without any new work.
- [Plan: card-derived-hints-and-bench-sync](./card-derived-hints-and-bench-sync.md) --
  owns `scripts/bench-sync.py`, which carries two findings in WP-D.

## Enables / Unblocks

- Honest probe data for SGLang, which is the precondition for every consumer decision
  (picker eligibility, router launch gating, MCP `list_fitting_models`, bench targets).
- A real answer to "should this lab run three backends?", which cannot currently be
  asked because the one capability that would justify SGLang -- RadixAttention prefix
  reuse -- has never been measured and the current harness cannot measure it.
- Closes the SGLang half of `router-anthropic-messages-compat` at zero cost.
- Twelve of the nineteen HIGH findings are backend-agnostic and improve vLLM too;
  they are owed whether SGLang is kept or frozen.

## Out of scope

- **A full backend rewrite.** Evaluated and rejected -- see "Why not a rewrite" below.
- **Splitting the four oversized files.** `gpu-arbiter/main.go` (4626 lines),
  `scripts/model-picker.py` (3590), `scripts/_probe_hf_common.py` (2212) and
  `scripts/select-models.py` (2184) all breach the 800-line ceiling in CLAUDE.md and
  all should be split. Bundling a ten-thousand-line mechanical move into a correctness
  remediation is how a behavioural change slips past a reviewer. File it separately.
- **Root-causing the Ornith CUDA fault upstream.** Phase 1 quarantines the model; why
  SGLang device-side-asserts on that particular in-house NVFP4 quant is an upstream
  question, and the fleet already serves the same checkpoint correctly on vLLM.
- **Wiring Open WebUI to the HF backends.** `deploy/docker-compose.yaml` gives
  `open-webui` only `OLLAMA_BASE_URL=http://router:11434`, so the only GUI cannot
  reach any SGLang or vLLM model. CLAUDE.md documents this as intentional. It is
  noted here because it makes `TODO.md:90` unreachable as written, not because this
  plan changes it.
- **Enabling SGLang MTP / speculative decoding.** The prober records no spec block for
  SGLang by design; WP-B only stops the router from emitting unprobed spec flags.

## Open questions

1. **Is the keep-or-freeze question actually open?** The `_PICKER_BACKENDS` re-enable
   on 2026-07-27 (commit a557229) reads as a keep decision made one day before this
   audit. This plan treats it as provisional-pending-evidence. If it was final, skip
   Phase 5 entirely and run Phases 1-4 as a straight backlog -- recommendation: treat
   it as final for now and run Phase 5 only if Phase 4's re-probe leaves fewer than
   four servable SGLang models.
2. **What are the Phase 5 adjudication thresholds?** They must be written down BEFORE
   the runs or the adjudication is theatre. Recommendation: KEEP requires both
   (i) SGLang steady TTFT within 3x of vLLM's on the same checkpoint, and (ii) turn-2+
   TTFT at least 25% better than vLLM's on an identical multi-turn workload, with a
   non-zero engine-reported cache hit rate and a measurable delta against
   `--disable-radix-cache`.
3. **What does the unexplained Qwen3.5-9B HumanEval result buy?** 0.86 on SGLang vs
   0.68 on vLLM, same checkpoint, and nothing in the audit explains it. Recommendation:
   record it as an open question in either branch (including in a `RESTORE.md` if
   frozen) but do not let it alone carry a keep -- a coding advantage at 20-33x TTFT is
   not usable in an agent loop.
4. **Is a multi-hour GPU-exclusive window available in the next two weeks?** Phase 4 is
   eight forced re-probes each bounded by `HEALTH_TIMEOUT_SECONDS=600`, plus load
   probes. Recommendation: if not, run Phases 1-3 (which need no GPU beyond one cold
   start) and defer Phase 4 -- say so up front rather than discovering it mid-phase.

## Context

An `ultracode` multi-agent audit on 2026-07-27 compared every SGLang code path against
the two working backends across ten dimensions (router launch, router lifecycle, router
request-rewrite, fit probe, load probe, bench, consumers, infra, tests, docs-truth),
then adversarially verified each finding. The first pass capped verification at eight
findings per dimension, leaving 24 unreviewed; a second pass reviewed all 24 with the
same refute-by-default prompt. Result across both passes: **103 findings reported,
80 survived (76 CONFIRMED, 4 PLAUSIBLE), 23 refuted as intentional-by-design or
unfounded, 0 left unverified.** By severity: 19 HIGH, 41 MEDIUM, 20 LOW, no CRITICAL.

The second pass mattered: it refuted 9 of the 24, including the only `needs-redesign`
item in the set (`sg-health-probe-under-arbiter-mutex`) and one finding whose arithmetic
was exact but whose causal story was wrong (`sg-parsers-sglang-missing-15-rows`, see
Phase 4 step 3). Had the register shipped after the first pass, those would have been its
most expensive false positives.

The audit's own premise turned out to be wrong in an important way, and correcting it
is what determines this plan's shape.

**SGLang is not broken. One model is.** Attributing every failure in the 90 MB
`devai-sglang.log` by the model that was loaded at the time:

| Model | 200 | 5xx | error rate |
| --- | --- | --- | --- |
| `Ornith-1.0-9B-NVFP4` | 263 | 899 | **77%** |
| `gpt-oss-20b` | 1014 | 57 | 5% |
| `Qwen3.5-9B-NVFP4` | 442 | 10 | 2% |
| `DeepSeek-R1-Distill-Qwen-7B` | 2602 | 0 | 0% |
| `NVIDIA-Nemotron-Nano-9B-v2-NVFP4` | 383 | 0 | 0% |
| `Qwen3-8B-NVFP4` | 3 | 0 | 0% |

Ornith accounts for **899 of the 966 server errors (93%)**. Excluding it, SGLang runs
at 67 errors in 4511 requests (1.5%). The backend serves; a single `(model, ctx)` cell
is poisoned.

**That one cell is a five-finding chain, and no single dimension saw the whole of it.**

1. `Ornith-1.0-9B-NVFP4` under SGLang raises
   `torch.AcceleratorError: CUDA error: device-side assert triggered` -- 180 times over
   three days, and **all 71 of 2026-07-27's occurrences under that model**.
2. The assert kills the scheduler: `Scheduler hit an exception` appears exactly 180
   times, the same count.
3. The engine keeps answering HTTP but generates only token 0. The probe cell records
   `full_content: ""` and a `full_reasoning_content` of thousands of identical `!`.
4. `scripts/_probe_load.py:753` computes `serving_ok = not failed`, consulting neither
   `needle_score` (0.0) nor `needle_valid` (false). The cell was certified
   **`serving_ok: true`, `fits: true`, `capability: structured`** at 131072 -- probed
   at 2026-07-27T06:49:32Z, i.e. current.
5. The picker's recall warning (`scripts/model-picker.py:1851`) fires only when
   `needle_valid` is **true**, so it is suppressed for exactly this case. The router
   has no SGLang terminal signature for `Scheduler hit an exception`
   (`gpu-arbiter/main.go:2158`), so it never recognises the corpse, and the launch
   breaker is repaid before the request completes (`main.go:3610`) -- producing
   **72 container recreates of that one model in one day**, against 4, 2 and 1 for the
   others.

This was not an oversight on one side of the design; it is a hole between two sides that
each knew about it. `docs/backends.md:472-476` already says of `needle_score`: "a failed
serve records 0.0 as well ... Nothing gates on it and nothing should until the
measurement is repaired." The repo correctly judged the recall number untrustworthy and
declined to gate on it -- but `serving_ok` was never wired to any substitute, so the one
signal that could have caught a degenerate engine was deliberately disconnected at both
ends. Phase 2's fix is therefore to add the missing signal, not to start trusting
`needle_score`.

**The reasoning path has never worked on SGLang at all.** Verified directly against
`lmsysorg/sglang:v0.5.10.post1-cu130`:

- `extra_body={'chat_template_kwargs': {...}}` -> `chat_template_kwargs is None` and
  `model_extra is None`. `extra_body` is not a field on SGLang's
  `ChatCompletionRequest`, so the router's payload is **silently discarded** -- no 422,
  no warning.
- Top-level `chat_template_kwargs` **is** honoured.
- `reasoning_effort='none'` expands to `{'thinking': False, 'enable_thinking': False}`
  -- a working one-field generation-off lever the router never sends.
- `separate_reasoning` **defaults to `True`**. So `applySGLangPolicy`'s *enable* branch
  sets a default and is a complete no-op, and its *disable* branch is the only live
  effect -- and that one is a PARSING switch, not a generation switch: it stops SGLang
  splitting `<think>` into `reasoning_content`, leaving it in `content`. The model still
  thinks; the user asked for no-think, gets the trace merged into the answer, and pays
  the tokens.

This also explains the probe's `disable_verified` verdicts, which are a tautology: any
check that confirms "disable worked" by observing an absent `reasoning_content` passes
unconditionally once `separate_reasoning=false` is sent. Measured across the caches,
`disable_verified` is **true for 8 of 9** reasoning-capable SGLang rows and **false for
0 of 11** vLLM rows on the same checkpoints. That is not a capability difference.

**Seven models carry an SGLang exclusion whose recorded evidence the lab
manufactured -- but only two of them are plausibly recoverable.** Before the
`backends` allow-list landed (2026-07-23, commit 98dc1e8), vLLM-only recovery flags
from `deploy/recovery-flags.json` reached SGLang launches. Seven cells probed
2026-07-09 record `sglang serve: error: unrecognized arguments: <that model's exact
engine_flags>`, and an eighth (`diffusiongemma`) records
`ModuleNotFoundError: No module named 'sglang'` from the entry's vLLM `image` override.
The allow-list fix shipped **without a cache-invalidation step**, so those verdicts are
still authoritative today, and `CLAUDE.md:210`, `scripts/model-families.yaml` and
`TODO.md:42` all cite them as genuine arch/quant gaps. They are not.

Two compounding details: `scripts/_probe_hf_common.py:878-884` lists the bare token
`"GPTQ"` in `_QUANT_ERROR_PATTERNS` and matches it against the whole log, so an
argparse usage dump (which enumerates `--quantization {awq,fp8,gptq,...}`) is filed as
`kind: quant` -- all seven cells record `matched_pattern: "GPTQ"`. And the exclusion
does not depend on that mislabel: `_probe_hf_common.py:1740-1763` maps **any** evidence
kind of severity >= 1 to `Capability.ERROR`, which `gpu-arbiter/main.go:578-581` treats
as terminal. Fixing the `GPTQ` pattern alone would NOT restore these models.

The second verification pass then bounded the prize, and it is smaller than the count of
seven suggests. Cross-tabbing the poisoned cells against independent clean probes: every
gemma4 row hits a `model_type: gemma4` arch wall inside `AutoConfig.from_pretrained`, and
qwen3.6 has a clean counter-probe that launched fine and then crashed in SGLang's own
gated-delta-net Triton kernel. Both gaps are genuine and unrelated to the flag bug. That
leaves **`Qwen3-Coder-30B-A3B-Instruct-FP4` and `NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`**
as the only two models whose SGLang exclusion is both manufactured and unexplained by any
independent evidence. Phase 4 should be scheduled on that expectation, not on seven.
Note also that none of the seven has weights in the SGLang store today, so none could
serve on 11436 even with a corrected verdict -- the re-probe buys honest data and a
`make hf-link` decision, not immediate capacity.

Finally, re-probing does not self-heal. `_probe_hf_common.py:1983-1988` counts a
populated band as `fully_cached` and skips it, so a bare `make probe-sglang` -- and
`make model-sync`, which drives it -- changes nothing. `PROBE_FORCE_ARCH=1` does not
help either: the skip at :1988 precedes `band.clear()` at :1989. Only
`PROBE_FORCE=1` clears the band.

### Why not a rewrite

The audit was explicitly asked to evaluate a full backend rewrite behind an explicit
backend contract, and one of the three independent design proposals argued for it. It
is rejected on evidence, not taste.

**The contract already exists, in both languages.** `gpu-arbiter/main.go` carries
`backendConfig{Name, ListenPort, BackendURL, ContainerName, Image, ModelsDir, Network,
HealthPath, Entrypoint func(modelName, launchConfig) []string, EnvVars, MountDest,
MountRW, DynamicEnv func(launchConfig) map[string]string}` -- behaviour lives in
function fields and Ollama, vLLM and SGLang are three instances of it.
`scripts/_probe_hf_common.py` carries `BackendSpec{name, image, container_name,
probe_port, cache_path, reserve_gb, entrypoint, build_args: Callable,
supports_plugins: bool, kv_cache_dtype, schema_version}` -- and `supports_plugins` is
already exactly the kind of per-engine capability flag a contract rewrite would
introduce, with the vLLM/SGLang difference documented inline.

**There is almost no string-switching to remove.** The whole 4626-line router contains
**five** `sglang` conditionals: `main.go:1040` and `:1050` (the memory reserve),
`:3746` and `:3808` (two `vllm || sglang` gates), and `:3861` (the
`applySGLangPolicy` dispatch). The shared Python has 5, 1, 2, 1, 3 and 7 across
`_probe_hf_common.py`, `_probe_load.py`, `bench_runner.py`, `_bench_core.py`,
`model-picker.py` and `select-models.py`.

**The defects are not shaped like a missing abstraction.** Of the 19 HIGH findings,
12 are backend-agnostic bugs in shared code that vLLM is equally exposed to, 4 are
missing per-engine *data* (parser curation, stale cells), and 3 are single missing
fields or flags. A rewrite would rebuild a structure that is already correct while
putting at risk the pipeline that produced the fleet's entire fit and quality picture
(16 fitting vLLM cells, 10 vLLM bench rows, 3 Ollama probe rows and 8 Ollama bench
rows) in service of a backend with two proven models. Forcing Ollama's separate 1047-line prober and its genuinely
different cell schema into a contract derived from two HF backends is the one place a
rewrite is most likely to be wrong, and it is the speculative generality CLAUDE.md
forbids.

Three ideas are grafted from the rewrite proposal without the rewrite, and appear in
the phases below: a flag-literal allowlist test, making an undeclared backend an error
in `memFraction` rather than a silent vLLM inheritance, and a frozen argv golden
corpus before any launch-path edit.

## Approach

Refactor in five phases, ordered so that nothing is built on data that cannot yet be
trusted. Phase 0 (already run) reads the evidence on disk and settles the open protocol
questions for free. Phase 1 stops the live crash loop. Phase 2 makes the probe and
reasoning paths tell the truth, which is the precondition for Phase 4's re-probe being
worth its GPU window. Phase 3 pays the backend-agnostic debt that would otherwise
corrupt Phase 4's measurements. Phase 4 re-probes honestly. Phase 5 is the optional
keep-or-freeze adjudication, which only earns its cost if Phase 4 leaves SGLang thin.

The single structural change is **extending** the two existing contracts with the
fields they lack, so per-engine knowledge stops being hardcoded to vLLM's shape:
`BackendSpec` gains `models_dir`, `allowed_kv_dtypes`, `prefill_chunk_flag`,
`tokenize_surface` and a terminal-signature set; `backendConfig` gains
`TerminalSignatures` and `NeedsServedModelName`.

---

## Phase 0 -- Read the evidence already on disk

### Goal

Settle, at zero GPU cost, the questions the audit left open -- so no later phase is
gated on an unrun experiment.

**This phase has been RUN (2026-07-27, read-only). Results are recorded below.** It is
documented rather than dropped because every later phase depends on these facts and a
future reader needs to know how they were obtained.

### Findings

1. **Failure attribution (the decision rule).** 93% of SGLang's 966 server errors
   belong to `Ornith-1.0-9B-NVFP4` (899, at a 77% per-model rate); excluding it the
   backend runs at 1.5%. The rule was "at least 90% Ornith -> one bad cell, branch
   prior KEEP". **Resolved: KEEP.** See the table in Context.
2. **The three SGLang bench rows are valid.** Their task windows
   (`gpt-oss-20b` 07-25T16:33 and 07-26T21:53-54; `Qwen3.5-9B` 07-26T22:06-22:30;
   `Nemotron-Nano-9B-v2` 07-27T01:23-01:37) overlap the crash dates, but the crashes
   were Ornith's and Ornith has no bench row. The three benched models' own error rates
   are 5%, 2% and 0%. The audit's "SGLang works end to end" verdict stands.
3. **Launch flags, against the pinned image's `--help`:** `--served-model-name`
   PRESENT (so the WP-B fix is available), `--tp` **ABSENT** (only `--tp-size`, so
   `make verify-backend-flags` is red today and `--tp` works only by argparse prefix
   matching), `--enable-metrics` PRESENT and defaulting False,
   `--disable-piecewise-cuda-graph` PRESENT, `--kv-cache-dtype` choices are
   `{auto,fp8_e5m2,fp8_e4m3,bf16,bfloat16,fp4_e2m1}` -- **no bare `fp8`**.
4. **Reasoning wire shape -- five HIGH findings specified, no precheck needed.**
   `extra_body` discarded silently; top-level `chat_template_kwargs` honoured;
   `reasoning_effort='none'` expands to `{'thinking': False, 'enable_thinking': False}`;
   `separate_reasoning` defaults `True`. See Context.
5. **Parser names.** Every curated `parsers.sglang` name in
   `scripts/model-families.yaml` is valid for this image (reasoning `qwen3`, `gpt-oss`,
   `nemotron_3`, `deepseek-r1`; tool `qwen`, `llama3`, `gpt-oss`, `qwen3_coder`). SGLang
   has **no** equivalent of vLLM's `nemotron_json`, `deepseek_string`, `qwen3_xml`,
   `gemma4` or `openai`, so the three families missing a tool parser need a substitute
   evaluated (`deepseekv3|v31|v32` for deepseek-r1-distill; `hermes` or `pythonic` for
   nemotron-nano-v2; `qwen` for qwen3.6), not a name copied across.
6. **Metrics cluster fully settled.** `GET /metrics` returned 404 eight times live;
   `--enable-metrics` exists and defaults False. One-flag fix, no investigation needed.
7. **`served_model_name='/models/<name>'`** appears in SGLang's own recorded launch args
   for all six models -- so the missing flag is not cosmetic, it is the engine's
   internal identity. Confirmed downstream: `curl devai-router:11436/v1/models` returns
   `/models/gpt-oss-20b` as its first entry, and `main.go:3438-3441` answers 404 for
   that exact id.
8. **Two audit CLEARED entries are false.** Both router-launch and router-lifecycle
   cleared the missing `--served-model-name` on the premise that SGLang defaults to the
   directory basename. It does not; it falls back to `model_path`. A remediator trusting
   the cleared list would skip a real defect.
9. **SGLang's `/health` is a STRONGER gate than vLLM's.**
   `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` defaults True, so `/health` submits a real
   one-token generation and returns 200 only when the detokenizer answers. Do **not**
   build a `warmLoadOllama`-style post-health liveness gate for SGLang; it exists.
10. **The systemd collision is not reproduced.** `podman inspect devai-sglang` shows
    `com.docker.compose.service=sglang` present on the router-recreated container, so
    `compose down` can remove it. `sg-infra-systemd-collides-on-recreated-sglang` needs
    re-derivation before any fix.
11. **`/v1/messages` on SGLang, answering another plan's open questions.** SGLang does
    expose it (`http_server.py:1658`), rejects the current Claude Code body with vLLM's
    exact `messages.1.role ... literal_error, input_value='system'` locator, and
    **folding the stray system message alone is sufficient** -- the full beta body
    (`context_management`, `output_config`, `thinking`, `metadata`, `tools`) is then
    accepted. Live counts: 633 x 200, 1 x 400.
12. **Codex is 100% broken on SGLang.** `POST /v1/responses`: 60 requests, **60 x 500**.
    The picker offers all seven agents against every SGLang row with no backend gating
    (`model-picker.py:3120` gates on reasoning and MTP only), and
    `_ensure_opencode_model()` writes a `router-sglang` provider into
    `~/.config/opencode/opencode.json` at launch.

### Exit criteria

- Met. All twelve results above are recorded, with commands, in this section.

---

## Phase 1 -- Stop the bleeding

**SHIPPED 2026-07-27.** All five steps done; exit criteria met except the
24-hour soak, which is time-based and still pending. One criterion was
met differently than written -- see "Exit criteria" below.

### Goal

End the live Ornith crash loop and make the router recognise a dead SGLang engine.
Nothing else in this plan is urgent; this is.

### Deliverables

```
deploy/.model-status.json          data   -- ledger Ornith-1.0-9B-NVFP4::sglang
deploy/.sglang-reasoning-cache.json data  -- clear the poisoned Ornith cell
gpu-arbiter/main.go                modify -- SGLang terminal signatures; breaker repayment
gpu-arbiter/lifecycle_test.go      modify -- handler-level breaker test
```

### Detailed steps

1. Record a `manual` exclusion for `Ornith-1.0-9B-NVFP4::sglang` via
   `_model_status.record_*`, with the device-side-assert evidence in `detail`, and clear
   its `serving_ok: true` cell so the picker and router stop offering it. The same
   checkpoint stays available on vLLM, where it serves at 262144.
2. Add SGLang terminal signatures to `terminalLaunchErrors`
   (`gpu-arbiter/main.go:2158`), sourced from the recorded logs rather than guessed:
   `Scheduler hit an exception` (180 occurrences, unambiguously fatal) and
   `device-side assert triggered`. Also add `unrecognized arguments` so a bad launch
   fails fast instead of burning the 600s health timeout.
3. Fix the launch-breaker repayment (`main.go:3610`): capture the launch key on the
   backend state and call `noteLaunchSucceeded()` from the proxy path after a request
   actually completes, not when `/health` first answers.
4. Make `lastErrorLine` case-insensitive, or add `error:` and
   `unrecognized arguments` to `failureAnchors`, so an argparse rejection is attributed
   rather than filed under a generic tail.
5. Add the handler-level breaker test the existing three cannot express: upstream
   answers 200 on `/health`, then closes the connection on `/v1/chat/completions`.
   Given 72 recreates in a day, this is not hypothetical.

### Exit criteria

- `Ornith-1.0-9B-NVFP4` no longer appears in the picker's SGLang rows, and a request
  for it on port 11436 is refused with a message naming the ledger entry.
  **MET, with one deviation.** The picker hides the row (footer reports it under
  "no context tier fits"), the router synthesises 9 SGLang rows instead of 10, and
  `POST /v1/chat/completions` on 11436 returns
  `404 {"error":"unknown model \"Ornith-1.0-9B-NVFP4\" for sglang"}`. The message does
  NOT name the ledger entry: **the router does not read `deploy/.model-status.json`
  at all** -- it gates on the probe cache, so `serving_ok: false` is what actually
  refuses the model. Naming the ledger would mean teaching the router to read it,
  which is not in this phase's deliverables. Filed as a Phase 3 candidate.
  `Ornith-1.0-9B-NVFP4` on **vLLM** is untouched and still serves at 262144, as intended.
- A synthetic SGLang log containing `Scheduler hit an exception` causes
  `detectLaunchFailure` to abort within seconds in a unit test. **MET** --
  `TestDetectLaunchFailure_SGLangSchedulerExceptionIsTerminal`, built from the
  captured log (note the `GET /health ... 200 OK` on the line before the assert),
  with `TestDetectLaunchFailure_SGLangHealthyLaunchReturnsNil` as the negative
  control against a real healthy cold start.
- The new breaker test fails against current `main.go` and passes after step 3.
  **MET, verified both ways.** With the repayment restored to its old position,
  `TestMakeRequestHandler_BreakerNotRepaidWhenEngineDiesServing` and
  `TestMakeRequestHandler_BreakerNotRepaidOnUpstream5xx` both fail with `spent=0`
  (i.e. repaid); after the fix both pass. 357 router tests are race-clean.
- 24 hours after deploy, `grep -c 'sglang with model' devai-router.log` for a single
  model is in single digits. **PENDING** -- time-based; router redeployed
  2026-07-27T17:39Z.

### Deviation recorded during execution

Repayment was moved to the proxy's `ModifyResponse` rather than to a
post-`ServeHTTP` check, and the breaker counters were given their own
`breakerMu` instead of continuing to rely on `arbiter.mu`. The mutex is
not cosmetic: `stopOtherBackends` -> `drainBackend` holds `arbiter.mu`
while busy-waiting for exactly the in-flight upstream requests that would
now be taking it, so repaying under `arbiter.mu` would stall every
backend switch under load for the full `DRAIN_TIMEOUT`. Lock order is
`arbiter.mu` -> `breakerMu`, never the reverse.

Credit is also given to **every** backend including Ollama. The breaker
is generic over `backendState`, so wiring the callback only into the two
HF backends would have made Ollama refuse healthy models after three
recreates.

**The first live verification of this phase was invalid and was redone.**
`podman restart devai-router` does not pick up a rebuilt image, so the
router was still running the old binary. The cache-driven effects (Ornith
vanishing from the SGLang rows, the 404 on 11436) appeared anyway,
because the probe caches are bind-mounted -- which is exactly what made
the stale binary hard to notice. Re-verified after
`podman rm -f devai-router && make cache-up`: 9 SGLang serving rows, the
404 refusal, and zero spurious breaker refusals across live vLLM and
Ollama traffic (exercising both `newSmartProxy` and `newProxy` credit
paths against real engines). See the deployment note in
`router-anthropic-messages-compat.md`.

### Phase 1 risks

| Risk | Mitigation |
| --- | --- |
| A terminal signature false-positives on a healthy SGLang banner | Both strings are exception text, not banner text; `_probe_hf_common.py:862-868` documents the banner hazard for `--trust-remote-code` and neither new string appears there. Assert against a captured healthy-launch log in the test. |
| Moving breaker repayment to request completion could stop crediting a slow-but-healthy launch | Credit on first successful proxied response, not on stream completion, so a long generation still counts. |

---

## Phase 2 -- Make the probe and the reasoning path tell the truth

### Goal

Stop the probe certifying broken engines, and make SGLang reasoning control actually
control reasoning. Both are fully specified by Phase 0; neither needs a GPU beyond one
confirmation cold start.

### Deliverables

```
scripts/_probe_hf_common.py   modify -- is_degenerate_generation; launch_args kind; disable probe
scripts/_probe_load.py        modify -- consume the degeneracy predicate; ledger kind gate
gpu-arbiter/main.go           modify -- applySGLangPolicy rewritten
scripts/model-picker.py       modify -- warn on degenerate output regardless of needle_valid
tests/python/test_probe_classify.py modify -- real argparse-dump fixture
tests/fixtures/sglang/        new    -- captured argparse dump and healthy-launch log
gpu-arbiter/policy_test.go    modify -- assert the emitted SGLang reasoning wire shape
```

### Detailed steps

1. Define `is_degenerate_generation(content, reasoning, finish_reason, output_tokens,
   cap)` **once** in `_probe_hf_common.py` -- empty content with output at the cap, or
   a single character/token run above some fraction of the text -- and consume it from
   both the fit classifier and `_probe_load.py`. Let it force `serving_ok = False`.
   Single-sourcing is the point: the load probe must not drift from the fit classifier.
2. Add a `launch_args` evidence kind, evaluated **before** the arch/quant cascade,
   matching `unrecognized arguments` and `error:`. It is a defect in our own launch
   construction, not a model verdict: it must abort the pass loudly and must never
   write a terminal capability. Independently, tighten `_QUANT_ERROR_PATTERNS` off the
   bare `GPTQ` and `AWQ kernel` tokens.
3. Rewrite `applySGLangPolicy` (`main.go:3929-3960`) against the Phase 0 facts: send
   `chat_template_kwargs` as a **top-level** field, not nested in `extra_body`; drop
   `separate_reasoning` as a policy lever entirely (its enable branch is a no-op, its
   disable branch is harmful); use `reasoning_effort` where an effort level is
   requested. Add a Go test asserting the emitted body shape, since the current
   payload is discarded silently and no test would notice.
4. Make the SGLang disable probe falsifiable: require BOTH `reasoning_content` absent
   AND no inline `<think>` markers in `content` before setting `disable_verified`.
   Then re-probe the eight rows carrying the tautological `true`.
5. Extend the picker's `_needle_failed_at` to warn when output was degenerate, not only
   when `needle_valid` is true. The current gate is suppressed by exactly the failure
   it should surface.
6. Gate the ledger write in `_probe_load.py:1033` on OOM kinds only, so an `infra`
   serving failure stops recording a false `oom` verdict.
7. Add the classifier test that would have caught all of this: feed a real recorded
   SGLang argparse dump from `tests/fixtures/` and assert `kind != "quant"`, plus a
   negative test that `GPTQ` inside help text does not match.
8. One confirmation cold start: send a disable request to `gpt-oss-20b@131072` and
   assert `<think>` is absent from both fields.

### Exit criteria

- Replaying the recorded Ornith load-probe response through the new predicate yields
  `serving_ok: false`.
- Replaying any of the seven recorded argparse dumps yields `kind: "launch_args"`, not
  `"quant"`, and writes no terminal capability.
- The confirmation cold start shows no `<think>` in either field with reasoning off,
  and a populated `reasoning_content` with it on.
- At most one of the eight re-probed SGLang rows still reports
  `disable_verified: true`, and that one is corroborated by the wire test.

### Phase 2 risks

| Risk | Mitigation |
| --- | --- |
| The degeneracy predicate rejects a legitimate reasoning model that burns its budget on `<think>` | Require EMPTY content as well as output-at-cap. The healthy comparison case (`Qwen3.5-9B`, 477 output tokens, coherent) is far from the boundary; add both as fixtures. |
| Changing the reasoning wire shape regresses vLLM | The change is inside `applySGLangPolicy` only. `applyVLLMPolicy` is untouched, and `extra_body` remains correct for vLLM. |
| Re-probing `disable_verified` needs GPU | Eight short chat probes, not full fit probes; can ride along with Phase 4's window. |

---

## Phase 3 -- Pay the backend-agnostic debt

**SHIPPED 2026-07-28.** All nine steps done. Details under "Progress".

### Goal

Fix the shared defects that would corrupt Phase 4's measurements, and close the
flag-drift class rather than its instances. Twelve of the nineteen HIGH findings live
here and are owed even if SGLang is later frozen.

### Deliverables

```
deploy/backend-flags.yaml     modify -- pin the four emitted-but-unpinned flags; tp -> tp-size
gpu-arbiter/main.go           modify -- --served-model-name; --tp-size; %.4f; memFraction guard
scripts/probe-sglang-reasoning.py modify -- --tp-size; kv dtype validation
scripts/bench/bench_runner.py modify -- refuse tools without a parser; record tool_mode
scripts/bench-sync.py         modify -- BENCH_FORCE for stale rows
scripts/select-models.py      modify -- per-(model, backend) KV dtype
scripts/catalog-discover.py   modify -- per-backend KV dtype in the VRAM band filter
scripts/model-picker.py       modify -- per-backend store gate; _has_tools; bench hint
Makefile                      modify -- export SGLANG_IMAGE, DEVAI_GPU_DEVICE
gpu-arbiter/flags_allowlist_test.go new -- emitted flags must be pinned
```

### Detailed steps

1. **Close the flag-drift class.** `deploy/backend-flags.yaml` declares itself the pin
   for every launch flag the router and probers emit, but nothing emits from it -- its
   only reader is `verify-backend-flags.py`. Ten findings descend from that. Add a Go
   test extracting the flag literals from `sglangEntrypoint`/`vllmEntrypoint` and
   asserting each appears in the YAML; pin `max_running_requests` and `max_num_seqs`;
   change `tp: "--tp"` to `--tp-size` and update both emitters so the gate goes green
   and can serve as a drift baseline again; extend the verifier to check flag *values*
   (parser and quant names) against the pinned image's `--help`.
2. Add `--served-model-name <modelName>` to `sglangEntrypoint` and to the SGLang
   prober, and pin it. This removes the phantom `/models/<name>` entry from the
   router's own `/v1/models`.
3. Align the memory fraction format: `%.2f` in Go (`main.go:1410`, and `:1275` for
   vLLM) vs `:.4f` in both probers. Use `%.4f` on both sides and add a Go/Python
   format-equality test.
4. Make an undeclared backend an error in `memFraction` rather than a silent vLLM
   inheritance -- its `default:` arm is annotated `// vllm` and
   `computeLaunchConfig` calls it with `cfg.Name` for all three backends, so Ollama
   receives vLLM's 2.0 GB reserve today (inert only because Ollama's entrypoint ignores
   `MemFraction`). Ten lines.
5. Validate the KV dtype against a per-backend allowed set before launch. SGLang
   rejects a bare `fp8`, and `PROBE_KV_CACHE_TYPE` is a single knob shared by both
   probers whose canonical vLLM value is `fp8` -- so one `make probe-sglang
   PROBE_KV_CACHE_TYPE=fp8` would manufacture more poisoned cells. No cell is stamped
   today, so this is latent, not live.
6. Bench: refuse to score `tools_use` when the row has no `tool_parser` (write an
   explicit sentinel instead of 0.0); record `tool_mode` on the task result and surface
   it in the report and picker; pass `BENCH_FORCE=1` for `stale_env`/`stale_image`
   targets so a re-bench does not clobber metrics with a partial merge.
7. Consumers: give the picker a per-backend store map and require the row's own store
   to contain the weights (six of ten advertised SGLang rows have none); drop the
   `disable_verified` term from `_has_tools` and mirror the router's actual gate
   (probe-recorded non-empty `tool_parser`); interpolate the backend into the
   `make bench-vllm` hint. Thread the backend through `resolve_kv_dtype` in
   `select-models.py` and fix the third copy of the fp8 premise in
   `catalog-discover.py:129-134`, where it drives the VRAM band filter that hides
   discovery candidates.
8. Infra: export `SGLANG_IMAGE` (with a `?=` pin) and `DEVAI_GPU_DEVICE` from the
   Makefile, or add `--env-file $(CURDIR)/.env` to the `$(COMPOSE)` macro; add
   `devai-vllm` and `devai-sglang` to `deploy/logging.sh:62`'s fallback list.
9. Freeze an argv golden corpus for all three backends **before** touching
   `--tp-size`, `%.4f` or `--served-model-name`, generalising the existing
   `make test-probe-ollama-idempotent` technique. Roughly 200 lines, and it converts
   "I believe this is a no-op" into a checked diff.

### Progress (2026-07-28)

**Done, in the plan's own order (step 9 first, deliberately):**

- **Step 9 -- argv golden corpus.** `gpu-arbiter/argv_golden_test.go` +
  `testdata/argv-golden.txt`, frozen BEFORE any launch-path edit, 9 cases x 3
  backends. It then caught exactly the three intended changes and nothing else:
  14 memory fractions reformatted, 9 `--tp` -> `--tp-size`, 9
  `--served-model-name` pairs added.
- **Step 1 -- flag-drift class closed.** `gpu-arbiter/flags_allowlist_test.go`
  extracts flag literals from the entrypoints and requires each to be pinned.
  It found two real gaps on its first run: `--reasoning-parser-plugin` and
  `--tool-parser-plugin` were emitted for months with no pin. `--tp` was
  corrected to `--tp-size` in the YAML and BOTH emitters (router and prober);
  `max_num_seqs`, `max_running_requests` and `served_model_name` are now
  pinned. `make verify-backend-flags` **exits 0**, now covering 17 vLLM +
  18 SGLang flags (was 15 + 16).
- **Step 2 -- `--served-model-name`** added to `sglangEntrypoint` and the
  SGLang prober. The prober's docstring had asserted the opposite ("default
  served-model name is the directory basename ... no --served-model-name flag
  in the router's sglangEntrypoint either") and is corrected.
- **Step 3 -- `%.4f`.** Both Go sites now match the probers' `f"{x:.4f}"`.
  This is not cosmetic: the router rounded a probe-measured `0.8836` to
  `0.88`, handing the engine ~0.09 GB less than the fit was measured with on a
  24 GB card.
- **Step 4 -- `memFraction` guard.** Reserves are a per-backend map; Ollama is
  recognised-and-ignored and an unknown backend logs a warning instead of
  silently inheriting vLLM's 2.0 GB. *Deviation:* the plan said "an error".
  Returning one would change the signature and ripple through a serving path,
  so it warns and falls back explicitly. The intent -- no silent inheritance --
  is met.
- **Step 5 -- KV dtype validation.** `BackendSpec.allowed_kv_dtypes` +
  `validate_kv_dtype`, called before launch. Sets taken from each pinned
  image's `--help`: SGLang `{auto,fp8_e5m2,fp8_e4m3,bf16,bfloat16,fp4_e2m1}`
  (**no bare `fp8`**), vLLM includes `fp8`. The error names the knob, the
  backend and the accepted set, and suggests `fp8_e4m3` for the exact
  `PROBE_KV_CACHE_TYPE=fp8` mistake the plan predicted.

**Verified live**, after `make build-router && podman rm -f devai-router &&
make cache-up` (a plain `podman restart` does NOT pick up a rebuild -- see
`router-anthropic-messages-compat.md`). The SGLang launch argv is now:

```
python3 -m sglang.launch_server --model-path /models/gpt-oss-20b \
  --served-model-name gpt-oss-20b --host 0.0.0.0 --port 11434 --tp-size 1 \
  --mem-fraction-static 0.8750 --context-length 131072 --trust-remote-code \
  --disable-piecewise-cuda-graph --max-running-requests 32 \
  --reasoning-parser gpt-oss --tool-call-parser gpt-oss
```

and `devai-sglang:11434/v1/models` returns `['gpt-oss-20b']` -- previously
`/models/gpt-oss-20b`, an id the router answered 404 for. The same request
returned 200 with a populated `reasoning_content`, which also confirms Phase
2's rewritten reasoning wire shape reaches the engine.

- **Step 6 -- bench honesty.** A vLLM/SGLang row with no probe-verified tool
  parser now records `{"score": null, "skipped": "no_tool_parser"}` instead of
  `0.0`. Running the task measured the ROUTER's own `maybeStripTools`, and a
  0.0 was indistinguishable in the leaderboard and picker from a model that
  was asked and got it wrong. Ollama is exempt (native tool negotiation, no
  probed parser by design). Both consumers already treat a null score as
  "unbenched", so the picker falls back to parser-presence rather than showing
  zero. Successful runs now also record `tool_mode` and `tool_parser` -- a
  forced-mode score and an auto-mode score are not comparable.
  `bench-sync` groups by **(backend, force)** and passes `BENCH_FORCE=1` for
  `stale_env`/`stale_image` rows only: `update_row` is a pure merge, so
  re-benching a stale row unforced leaves half the metrics from the old
  host-env/image behind under a single fresh stamp -- while forcing a
  `new`/`incomplete` row would discard the resumability the planner exists
  for. This needed a `class` field on plan rows; `needs_bench()` flattens the
  classes, so the distinction was gone by the time execute() needed it, and
  the first version of the check was silently inert.
- **Step 7 -- consumers.** The picker gained a per-backend HF store map and
  now requires the weights to be in THAT backend's store (each engine
  bind-mounts only its own), naming every gated row in the footer with the
  `link-hf-store.py` command -- a hard link, not a re-download. On this host
  the gate is currently a no-op because both stores are already in sync;
  proven non-inert by pointing `SGLANG_MODELS_DIR` at an empty directory,
  which correctly drops all four SGLang rows and names them.
  `_has_tools` no longer subtracts `disable_verified`: that field is about
  whether REASONING can be turned off and has nothing to do with tool calling.
  The conflation became actively harmful once the disable probe was made
  falsifiable, since 8 of 9 SGLang reasoning rows carry a
  `disable_verified: true` this would have read as "cannot use tools". It now
  mirrors the router's actual gate -- a non-empty probed `tool_parser`.
  **`resolve_kv_dtype` was wrong for SGLang and is corrected.** It returned
  `fp8` for every non-Ollama row, but `sglangEntrypoint` emits
  `--kv-cache-dtype` only for a STAMPED cell; all 10 SGLang fitting cells here
  are unstamped, a live launch argv carries no such flag, and the router
  itself decodes an unstamped cell to fp8 **for vLLM only**. SGLang therefore
  serves unquantized KV, and costing it at fp8 halved its KV estimate and
  classified models as fitting that will not. Now: vLLM-reachable -> fp8,
  SGLang-only -> fp16, Ollama-only -> fp16. `test_kv_fit_math` asserted the
  old premise and is corrected with that evidence.
- **Step 8 -- infra.** `SGLANG_IMAGE` is exported with a `?=` pin. It had been
  deliberately unexported because exporting it EMPTY would beat the prober's
  own default; the pin removes that hazard, so a `.env` bump can no longer
  reach compose while the host-run prober measures a different build.
  `DEVAI_GPU_DEVICE` is exported for the same reason. `deploy/logging.sh`'s
  fallback target list gains `devai-vllm` and `devai-sglang` -- both start as
  `sleep infinity` placeholders and are recreated by the router, so a logger
  starting before either was used would never follow the engine whose crash
  logs matter most.

**Incidental finding, not fixed here:** a resident Ollama model survives a
router restart invisibly. After recreating the router, `ollama ps` still
showed `gemma4:26b-a4b-it-q4_K_M` holding 21.8 GB ("Forever", keep-warm), and
the router -- which logs `adopted already-serving backend vllm` -- did not
adopt or evict it, so the next SGLang launch OOMed at 186 MiB free. The
fail-fast worked correctly (~18 s, not the 600 s timeout) and the message
named the cause, but `stopOtherBackends` cannot free a backend it does not
know is running. Worth its own item.

### Exit criteria

- `make verify-backend-flags` exits 0 against both pinned images. **MET.**
- The new allowlist test fails if a flag is added to either entrypoint without
  a pin. **MET** -- and it failed on first run against two real unpinned flags.
- `curl devai-router:11436/v1/models` contains no `/models/`-prefixed id.
  **MET** (checked at the engine: `['gpt-oss-20b']`).
- The argv golden corpus is byte-identical for Ollama and vLLM across the whole
  phase. **PARTIALLY MET, and the criterion contradicts step 3.** Ollama is
  byte-identical (verified). vLLM is NOT, and cannot be: step 3 explicitly
  changes `--gpu-memory-utilization` from `%.2f` to `%.4f`, which is a vLLM
  flag. The defensible reading is "no UNINTENDED argv change", which holds --
  the full diff is only the three intended edits.
- A bench row for a parser-less model records the sentinel, not `tools: 0.0`.
  **MET** -- `{"score": null, "n": 0, "skipped": "no_tool_parser"}`.

### Phase 3 risks

| Risk | Mitigation |
| --- | --- |
| `--tp` -> `--tp-size` changes a working launch | `--tp` currently resolves only by argparse prefix matching; `--tp-size` is the advertised spelling. The golden corpus catches any other argv change in the same commit. |
| The picker store gate hides rows a user expects | The gate names `make hf-link FROM=vllm TO=sglang NAME=<model>` in its message -- a hard-link, not a re-download. |
| `%.4f` changes the measured fit | It makes serve-time match probe-time, which is the point; re-probe in Phase 4 regardless. |

---

## Phase 4 -- Honest re-probe

### Goal

Replace every manufactured SGLang verdict with a measured one, and make the stale-cell
class self-detecting so this never needs an archaeology pass again.

GPU-exclusive. This is the phase that needs the window in open question 4.

### Deliverables

```
scripts/_probe_hf_common.py   modify -- launch fingerprint per cell; band-skip condition
scripts/probe-check.py        modify -- report fingerprint drift
scripts/model-families.yaml   modify -- parsers.sglang for the three missing families
deploy/.sglang-reasoning-cache.json data -- eight cells re-measured
```

### Detailed steps

1. Add a **launch fingerprint** to each probe cell: a hash of the emitted argv minus
   model path and ctx. Invalidate the cell when it changes, and report drift from
   `make probe-check`. About 30 lines, independently requested by four audit
   dimensions, and it is what makes this whole class self-detecting.
2. Change the band-skip condition (`_probe_hf_common.py:1983-1988`) from "band is
   non-empty" to "band contains a fitting cell", so a band of nothing but failures is
   re-probed rather than treated as complete.
3. Do **not** add `parsers.sglang` blocks for the families that lack one. The second
   verification pass refuted that item, and the refutation is correct -- it was checked
   against the caches and reproduced here.

   Measured against the generated catalog: 31 of `deploy/models.yaml`'s 328 rows declare
   `sglang`, and exactly 15 lack a `parsers.sglang` block -- gemma4 (10), qwen3.6 (4),
   diffusiongemma (1). But the omission does **not** correlate with the poisoned cells in
   either direction, which is what kills the causal story:

   - `Qwen3-Coder-30B-A3B-Instruct-FP4` and `NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` both
     carry poisoned cells **and** full `parsers.sglang` blocks. If poisoning had caused
     the omissions, these two would be missing theirs.
   - Every gemma4 row hits an arch wall at `AutoConfig.from_pretrained`
     (`ValueError: ... model type 'gemma4' but Transformers does not recognize this
     architecture`), which is reached BEFORE any engine flag is parsed and therefore
     applies to every `model_type: gemma4` checkpoint including the NVFP4 quants. The
     "arch/quant gap" label at `model-families.yaml:150` is independently correct, and
     understated -- the gap is not NVFP4-specific.
   - qwen3.6 has an independent, clean, non-poisoned counter-probe:
     `unsloth/Qwen3.6-27B-NVFP4` launched with no recovery flags, reached
     `Application startup complete` / `Uvicorn running`, then crashed on the first
     generation inside SGLang's own linear-attention path
     (`srt/layers/attention/linear/kernels/gdn_triton.py:143 -> fla/chunk.py ->
     chunk_gated_delta_rule_fwd_kkt_solve_kernel`). Verified: that cell's log excerpt
     contains `gdn_triton`, `gated_delta`, `fla/chunk` and `Application startup
     complete`, and contains neither `unrecognized arguments` nor `CUDA out of memory`.
     A genuine gated-delta-net kernel gap for the qwen3.6 hybrid arch, shared by its
     three poisoned siblings.

   So a parser block for gemma4, qwen3.6 or diffusiongemma buys nothing: those families
   cannot load on this image regardless. Adding one would be speculative curation.
   The two families whose SGLang exclusion is genuinely unexplained -- qwen3-coder and
   nemotron-3-nano -- already have their parser blocks.
4. Re-probe the eight poisoned cells with `PROBE_FORCE=1` (note: `PROBE_FORCE_ARCH=1`
   is insufficient -- the skip precedes `band.clear()`).
5. Re-run the load probe after every forced fit re-probe: a fit re-probe drops the
   `serving_*` augmentation, so skipping this would leave cells advertising a fit with
   no serving evidence.

### Exit criteria

- No SGLang cell carries an evidence kind of `quant` whose `log_excerpt` is argparse
  usage text.
- Every SGLang cell carries a launch fingerprint, and `make probe-check` reports drift
  when an entrypoint flag changes.
- The count of servable SGLang models is a measured number, and the docs in Phase 5
  cite it rather than the 2026-07-09 verdicts.

### Phase 4 risks

| Risk | Mitigation |
| --- | --- |
| The eight models genuinely do not run on SGLang, and the window buys nothing | Then the verdicts become honest, which is the deliverable. Two are known-genuine gemma4 arch gaps; the other six are unknown, which is the problem. |
| Re-probe wall time exceeds the window | `PROBE_READY_TIMEOUT` and `HEALTH_TIMEOUT_SECONDS` bound each launch; probe in descending expected-value order and record what was left undone rather than reporting full coverage. |

---

## Phase 5 -- Adjudicate, and tell the truth in the docs

### Goal

Decide whether SGLang stays, on measured evidence, and leave no stale claim behind
either way.

Run the adjudication only if open question 1 is genuinely open, or if Phase 4 leaves
fewer than four servable SGLang models.

### Detailed steps

1. **Measure the capability that would justify keeping SGLang.** RadixAttention prefix
   reuse has never been measured and the current harness cannot measure it:
   `--enable-metrics` is emitted nowhere, `/metrics` returns 404, and `DEFAULT_TASKS`
   is entirely single-turn so prefix reuse has no scorer. Emit `--enable-metrics`, pin
   it, and build a multi-turn scorer -- validating it radix-on versus
   `--disable-radix-cache` on the same engine before trusting any cross-backend number.
2. Apply the open-question-2 thresholds, recorded before the runs.
3. **Fix every prose site.** The misattributed arch/quant claim has at least three
   copies the audit's two doc findings would miss: `CLAUDE.md:210`, `TODO.md:42`,
   `AGENTS.md:76` ("`11436`: SGLang, dormant" -- injected into agent context alongside
   CLAUDE.md) and `README.md:335`. Also `scripts/model-families.yaml`'s status block,
   `docs/backends.md`, `docs/router.md`'s vLLM-only tool-mode table, and
   `docs/bench-results.md` (regenerate via `make bench-report`).
4. **Deal with `docs/html/`.** Fifteen hand-maintained files with no generator
   (`grep -rn 'docs/html' Makefile scripts/` is empty), still publishing
   `router.html:157` "NVFP4 (broken)" -- the claim `TODO.md:74` says was fixed on
   2026-07-09 -- and `bench-results.html:148` "SGLang on hold". Regenerate or delete;
   a published surface with a wrong capability claim is worse than no surface.
5. **Gate agent pairing by backend.** Codex fails 100% on SGLang
   (`/v1/responses`, 60 x 500) and the picker offers it anyway. Either gate `_AGENTS`
   on backend in `_resolve_agent`, or fix the router's `/v1/responses` handling. Fold
   the stray system message so Claude Code's `/v1/messages` works on both HF backends,
   closing `router-anthropic-messages-compat` open questions 1 and 4.
6. If the decision is freeze, follow the `attic/` precedent: remove `sglang` from
   `_PICKER_BACKENDS` and the make targets, move the code behind a build tag, and write
   a `RESTORE.md` carrying the open questions -- including question 3. Keep every fix
   from Phases 1-3; twelve of the nineteen HIGH findings stay live on vLLM.

### Exit criteria

- No file in the repo asserts an SGLang capability that the caches contradict.
- The picker never offers an agent/backend pair that is known to fail.
- The keep-or-freeze decision, its thresholds and its measured inputs are recorded in
  this plan.

---

## Defect register

103 findings were reported across the two verification passes.
**80 survived** (76 CONFIRMED, 4 PLAUSIBLE),
**23 were refuted** as intentional-by-design or unfounded, and
**0 remain unverified**.
By severity across the surviving and unverified set:
19 HIGH, 41 MEDIUM,
20 LOW. No CRITICAL survived verification.

The first pass capped verification at eight findings per dimension, leaving 24
unreviewed; a second pass reviewed all 24 with the same dual-lens refute-by-default
prompt, and refuted 9 of them. Any row still marked UNVERIFIED has had
no skeptic and should be re-checked before being acted on.

### WP-A  Shared HF probe scaffold (fixes vLLM too)

| Sev | Verdict | Finding | Location | Scope |
| --- | --- | --- | --- | --- |
| HIGH | CONFIRMED | `sg-argparse-misbucketed-as-quant` | `scripts/_probe_hf_common.py:883` | localised |
| HIGH | CONFIRMED | `sg-disable-verified-is-a-tautology` | `scripts/_probe_hf_common.py:753` | cross-cutting |
| HIGH | CONFIRMED | `sg-load-degenerate-output-passes` | `scripts/_probe_load.py:470` | localised |
| HIGH | CONFIRMED | `sg-load-infra-failure-writes-oom-ledger` | `scripts/_probe_load.py:1033` | one-line |
| MEDIUM | CONFIRMED | `sg-load-underfilled-window` | `scripts/_probe_load.py:362` | localised |
| MEDIUM | CONFIRMED | `sg-poisoned-terminal-cells-are-permanent` | `scripts/_probe_hf_common.py:1985` | localised |
| MEDIUM | CONFIRMED | `sg-probe-container-outside-mutex-set` | `scripts/_probe_hf_common.py:126` | one-line |
| MEDIUM | CONFIRMED | `sg-serving-fill-undershoot` | `scripts/_probe_load.py:294` | localised |
| LOW | CONFIRMED | `sg-load-predicted-logits-unreachable` | `scripts/_probe_load.py:563` | localised |
| LOW | PLAUSIBLE | `sg-load-reasoning-field-hardcoded` | `scripts/_probe_load.py:705` | one-line |
| LOW | CONFIRMED | `sg-test-shared-default-models-dir-untested` | `scripts/_probe_hf_common.py:86` | localised |

### WP-B  Router (gpu-arbiter)

| Sev | Verdict | Finding | Location | Scope |
| --- | --- | --- | --- | --- |
| HIGH | CONFIRMED | `sg-breaker-repaid-before-request` | `gpu-arbiter/main.go:3610` | localised |
| HIGH | CONFIRMED | `sg-chat-template-kwargs-nesting` | `gpu-arbiter/main.go:3949` | localised |
| HIGH | CONFIRMED | `sg-coalescing-bypassed-by-stopotherbackends` | `gpu-arbiter/main.go:3095` | localised |
| HIGH | CONFIRMED | `sg-disable-verified-tautology` | `gpu-arbiter/main.go:3972` | localised |
| HIGH | CONFIRMED | `sg-launch-recovery-flags-poisoned-cells` | `gpu-arbiter/main.go:1457` | cross-cutting |
| HIGH | CONFIRMED | `sg-missing-scheduler-crash-signature` | `gpu-arbiter/main.go:2158` | localised |
| HIGH | CONFIRMED | `sg-separate-reasoning-is-not-a-generation-switch` | `gpu-arbiter/main.go:3946` | localised |
| MEDIUM | CONFIRMED | `sg-argparse-error-not-anchored` | `gpu-arbiter/main.go:2172` | one-line |
| MEDIUM | CONFIRMED | `sg-keepalive-commits-200-then-upstream-error` | `gpu-arbiter/main.go:3600` | localised |
| MEDIUM | PLAUSIBLE | `sg-launch-mtp-flags-never-probed` | `gpu-arbiter/main.go:1454` | one-line |
| MEDIUM | CONFIRMED | `sg-native-admin-surface-unguarded` | `gpu-arbiter/main.go:1866` | cross-cutting |
| MEDIUM | CONFIRMED | `sg-no-reasoning-effort-control` | `gpu-arbiter/main.go:3934` | localised |
| MEDIUM | CONFIRMED | `sg-no-served-model-name` | `gpu-arbiter/main.go:1404` | one-line |
| MEDIUM | CONFIRMED | `sg-no-vram-settle-on-switch` | `gpu-arbiter/main.go:2954` | localised |
| MEDIUM | PLAUSIBLE | `sg-reconcile-single-health-probe` | `gpu-arbiter/main.go:1987` | localised |
| MEDIUM | CONFIRMED | `sg-stale-error-cells-silently-hide-models` | `gpu-arbiter/main.go:579` | cross-cutting |
| MEDIUM | CONFIRMED | `sg-stop-failure-does-not-abort-launch` | `gpu-arbiter/main.go:2955` | localised |
| MEDIUM | CONFIRMED | `sg-test-no-probe-serve-arg-parity-test` | `gpu-arbiter/main.go:1410` | cross-cutting |
| MEDIUM | CONFIRMED | `sg-weights-gate-wrong-remedy` | `gpu-arbiter/main.go:2883` | one-line |
| LOW | CONFIRMED | `sg-launch-argparse-failure-unattributed` | `gpu-arbiter/main.go:2172` | localised |
| LOW | CONFIRMED | `sg-launch-memfraction-rounding-divergence` | `gpu-arbiter/main.go:1410` | one-line |
| LOW | CONFIRMED | `sg-test-drift-map-and-recovery-lastwins-untested` | `gpu-arbiter/main.go:1815` | localised |

### WP-C  Consumers (picker, select-models, MCP)

| Sev | Verdict | Finding | Location | Scope |
| --- | --- | --- | --- | --- |
| MEDIUM | CONFIRMED | `sg-noninteractive-backend-resolves-vllm-first` | `scripts/model-picker.py:3280` | localised |
| MEDIUM | CONFIRMED | `sg-picker-has-tools-disable-verified` | `scripts/model-picker.py:1648` | one-line |
| MEDIUM | CONFIRMED | `sg-picker-missing-weights-gate` | `scripts/model-picker.py:1071` | localised |
| MEDIUM | CONFIRMED | `sg-picker-no-sglang-store-check` | `scripts/model-picker.py:1071` | localised |
| MEDIUM | CONFIRMED | `sg-select-models-kv-dtype-fp8-for-sglang` | `scripts/select-models.py:180` | localised |
| MEDIUM | CONFIRMED | `sg-test-picker-store-gate-missing` | `scripts/model-picker.py:1071` | localised |
| LOW | CONFIRMED | `sg-picker-bench-hint-names-bench-vllm` | `scripts/model-picker.py:2407` | one-line |
| LOW | CONFIRMED | `sg-picker-mtp-offered-without-any-sglang-probe` | `scripts/model-picker.py:1715` | localised |
| LOW | CONFIRMED | `sg-select-models-single-vllm-priority-verdict` | `scripts/select-models.py:1643` | cross-cutting |

### WP-D  Bench harness

| Sev | Verdict | Finding | Location | Scope |
| --- | --- | --- | --- | --- |
| HIGH | CONFIRMED | `sg-bench-stale-no-force-clobbers-metrics` | `scripts/bench-sync.py:289` | localised |
| HIGH | CONFIRMED | `sg-bench-tool-mode-not-recorded` | `scripts/bench/bench_runner.py:842` | localised |
| HIGH | CONFIRMED | `sg-bench-tools-scored-without-parser` | `scripts/bench/bench_runner.py:830` | localised |
| MEDIUM | CONFIRMED | `sg-bench-engine-metrics-never-captured` | `scripts/bench/bench_runner.py:167` | localised |
| MEDIUM | CONFIRMED | `sg-bench-ttft-steady-quantum` | `scripts/bench/bench_latency_leak.py:101` | localised |
| MEDIUM | CONFIRMED | `sg-test-bench-backend-metrics-silent` | `scripts/bench/bench_runner.py:168` | localised |
| LOW | CONFIRMED | `sg-bench-budget-starves-sglang` | `scripts/bench-sync.py:271` | localised |

### WP-E  Infrastructure (Makefile, compose, flag pins, systemd, logging)

| Sev | Verdict | Finding | Location | Scope |
| --- | --- | --- | --- | --- |
| HIGH | CONFIRMED | `sg-infra-sglang-image-not-exported` | `Makefile:51` | one-line |
| HIGH | CONFIRMED | `sg-infra-systemd-collides-on-recreated-sglang` | `deploy/systemd/devai-infra.service:10` | localised |
| MEDIUM | CONFIRMED | `sg-bench-no-sglang-smoke-test` | `Makefile:1800` | localised |
| MEDIUM | CONFIRMED | `sg-infra-gpu-device-not-exported` | `Makefile:42` | one-line |
| MEDIUM | CONFIRMED | `sg-infra-max-running-requests-unpinned` | `deploy/backend-flags.yaml:50` | one-line |
| MEDIUM | CONFIRMED | `sg-infra-no-sglang-store-maintenance-targets` | `Makefile:1164` | localised |
| MEDIUM | CONFIRMED | `sg-infra-verify-backend-flags-red-on-tp` | `deploy/backend-flags.yaml:54` | localised |
| MEDIUM | CONFIRMED | `sg-launch-max-running-requests-unpinned` | `deploy/backend-flags.yaml:50` | one-line |
| MEDIUM | CONFIRMED | `sg-max-running-requests-unpinned` | `deploy/backend-flags.yaml:50` | one-line |
| MEDIUM | CONFIRMED | `sg-test-bench-smoke-vllm-only` | `Makefile:1800` | localised |
| LOW | CONFIRMED | `sg-infra-cache-status-blind-to-sglang` | `Makefile:1014` | localised |
| LOW | CONFIRMED | `sg-infra-logger-fallback-omits-sglang` | `deploy/logging.sh:62` | one-line |
| LOW | CONFIRMED | `sg-launch-backend-flags-stale-verification-stamp` | `deploy/backend-flags.yaml:25` | localised |
| LOW | CONFIRMED | `sg-stale-nvfp4-broken-crossrefs` | `deploy/backend-flags.yaml:73` | localised |

### WP-G  Documentation truth

| Sev | Verdict | Finding | Location | Scope |
| --- | --- | --- | --- | --- |
| HIGH | CONFIRMED | `sg-arch-quant-gap-misattributed` | `CLAUDE.md:210` | cross-cutting |
| HIGH | CONFIRMED | `sg-ornith-structured-on-garbage-output` | `docs/backends.md:453` | localised |
| MEDIUM | CONFIRMED | `sg-bench-onhold-claim-stale` | `docs/bench-results.md:647` | localised |
| MEDIUM | CONFIRMED | `sg-documented-stale-cell-repair-is-a-noop` | `docs/backends.md:292` | localised |
| MEDIUM | CONFIRMED | `sg-load-probe-augmentation-lost-on-refit` | `docs/backends.md:395` | localised |
| MEDIUM | CONFIRMED | `sg-nemotron-listed-as-serving-but-ledger-dropped` | `CLAUDE.md:210` | one-line |
| MEDIUM | CONFIRMED | `sg-prometheus-metrics-never-captured` | `docs/bench-results.md:640` | localised |
| MEDIUM | CONFIRMED | `sg-tool-mode-table-vllm-only` | `docs/router.md:355` | localised |
| MEDIUM | CONFIRMED | `sg-verify-backend-flags-permanently-red` | `docs/backends.md:22` | one-line |
| LOW | CONFIRMED | `sg-bench-docs-claim-sglang-on-hold` | `docs/bench-results.md:25` | localised |
| LOW | CONFIRMED | `sg-prefix-caching-flag-claim` | `docs/backends.md:485` | one-line |
| LOW | CONFIRMED | `sg-transient-and-cell-count-drift` | `docs/backends.md:466` | one-line |

### WP-H  Tests

| Sev | Verdict | Finding | Location | Scope |
| --- | --- | --- | --- | --- |
| HIGH | CONFIRMED | `sg-test-classify-argparse-dump-untested` | `tests/python/test_probe_classify.py:73` | localised |
| MEDIUM | PLAUSIBLE | `sg-test-router-sglang-no-transient-retry` | `tests/test-router-sglang.sh:210` | one-line |
| LOW | CONFIRMED | `sg-test-probe-smoke-always-skips` | `tests/test-probe-sglang.sh:18` | one-line |
| LOW | CONFIRMED | `sg-test-prober-concurrency-partial-parity` | `tests/python/test_needle_warning.py:146` | one-line |
| LOW | CONFIRMED | `sg-test-router-sglang-wrong-sort-key` | `tests/test-router-sglang.sh:91` | one-line |

### Refuted as intentional-by-design or unfounded -- do not "fix" these

23 findings did not survive. 14 were refuted in the first
pass and 9 in the second:

`load-omits-reasoning-policy-surface`, `sg-bench-drop-flag-unclearable-via-loop`, `sg-bench-include-usage-never-set`, `sg-bench-recovered-ctx-map-backend-blind`, `sg-health-probe-under-arbiter-mutex`, `sg-image-env-not-exported-to-prober`, `sg-infra-model-pull-cannot-target-sglang-store`, `sg-infra-sglang-models-dir-split-source`, `sg-launch-recovery-image-no-backend-guard`, `sg-launch-unknown-size-memfraction-fallback`, `sg-ledger-size-verdict-written-for-every-backend`, `sg-load-calibration-chat-unaccounted`, `sg-load-needle-valid-dead-for-sglang`, `sg-mcp-listfitting-no-capability-gate`, `sg-parsers-sglang-missing-15-rows`, `sg-picker-advertises-weightless-sglang-rows`, `sg-qwen3coder-sglang-parser-unverified`, `sg-select-models-downloaded-gated-on-active-store`, `sg-store-gap-stale-precells`, `sg-test-hidden-era-assertion-pinned`, `sg-test-router-suite-skips-green-forever`, `sg-test-shape-only-inventory`, `sg-test-terminal-signature-sglang-absent`.

Notable: `sg-picker-advertises-weightless-sglang-rows` duplicates the confirmed
`sg-picker-no-sglang-store-check`; `sg-infra-model-pull-cannot-target-sglang-store` is
unfounded because `make hf-link` exists (Makefile:1168).

### Corrections the audit made to itself

- The `GPTQ` substring produces the wrong *label*, not the exclusion. Any evidence kind
  of severity >= 1 becomes `Capability.ERROR`. Fixing the pattern alone restores
  nothing.
- `gpu-arbiter/main.go:1457` (recovery-flag append) is **correct today** -- the
  allow-list works and all ten entries are scoped to `["vllm"]`. The live defect is
  un-invalidated cache state plus documentation written on top of it.
- `reconcileBackendState` clearing `currentModel` is deliberate and backend-symmetric
  (`main.go:1975-1997`), not a partial SGLang adopt.
- The `Qwen3.6-27B-NVFP4` `oom` verdict is legitimate: the cell records
  `matched_pattern: "CUDA out of memory"`.
- Two CLEARED entries were wrong about `--served-model-name` (Phase 0 finding 8).
- The systemd collision did not reproduce (Phase 0 finding 10).

## Combined risk register

| Risk | Phase | Mitigation |
| --- | --- | --- |
| A shared-scaffold fix regresses vLLM, whose data is the fleet's entire picture | 2, 3 | Argv golden corpus frozen before any launch-path edit; every shared change lands with a replay test over recorded vLLM responses. |
| Phase 4's GPU window is unavailable, leaving the plan half-done | 4 | Phases 1-3 need no window beyond one cold start and are independently shippable. Declare the deferral rather than discovering it. |
| The re-probe confirms SGLang is thin, making Phases 2-4 look wasted | 4, 5 | Twelve of nineteen HIGH findings are backend-agnostic and improve vLLM regardless. The freeze branch keeps all of them. |
| Acting on an UNVERIFIED register row that is actually intentional | all | 14 of 103 findings were refuted on exactly that basis. Re-verify any UNVERIFIED row before acting. |
| The plan's own evidence goes stale | all | Phase 0's commands are recorded and are read-only; re-run them rather than trusting the numbers. |

## Migration / rollback story

- Phases 1-3 are ordinary code changes; rollback is reverting the commit. The argv
  golden corpus makes a launch-path revert verifiable rather than hopeful.
- Phase 1's ledger and cache edits are data-only and reversible with
  `make model-status CLEAR=Ornith-1.0-9B-NVFP4::sglang` plus a re-probe.
- Phase 4 rewrites SGLang cache cells. Back the cache up first
  (`make backup-create` covers it per `docs/backup-restore.md`); the pre-Phase-4 cells
  are manufactured verdicts, so losing them is the intent, but the backup makes the
  before/after auditable.
- No user-facing interface changes. The picker's SGLang row set shrinks in Phase 3
  (store gate) and changes in Phase 4 (re-probe); both are corrections to
  over-advertisement, and the store-gate message names the one-command fix.

## Estimated effort

| Phase | Engineering effort | Wall-clock |
| --- | --- | --- |
| Phase 0 | done, read-only | ~1 hour (spent) |
| Phase 1 | 1 PR, ~150 LoC + 1 test | half a day |
| Phase 2 | 2 PRs, ~350 LoC + 4 tests | 2 days + 1 cold start |
| Phase 3 | 3-4 PRs, ~600 LoC + golden corpus (~200) | 4-5 days |
| Phase 4 | 1 PR, ~120 LoC; GPU-exclusive re-probe | 1 day code + 3-5 hour window |
| Phase 5 | 1-2 PRs, docs + scorer (~250 LoC) | 2-3 days |
| Total | 8-10 PRs | ~2 weeks + one GPU window |

## References

- Audit run 2026-07-27: 94 agents, 103 findings, 10 dimensions, adversarial
  verification per finding. Journal retained at the session's workflow transcript dir.
- `docs/backends.md` -- backend lifecycle, probing procedure, failure-mode taxonomy.
- `docs/router.md` -- request rewrite chain; source of truth for the router.
- `attic/README.md` -- the freeze precedent, if Phase 5 goes that way.
- `deploy/backend-flags.yaml` -- the flag pin file that nothing emits from.
- SGLang v0.5.10.post1: `sglang/srt/entrypoints/openai/protocol.py`
  (`ChatCompletionRequest` field set), `sglang/srt/entrypoints/http_server.py:1658`
  (`/v1/messages`), `sglang/srt/server_args.py` (`--enable-metrics`,
  `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION`).
