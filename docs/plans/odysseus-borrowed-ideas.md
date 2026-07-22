# Ideas borrowed from odysseus

_A living register of features mined from the odysseus project (Cookbook,
Compare, Chat+Agents) that could improve devai -- each independently tracked
from Proposed through Implemented or Rejected._

## Status

Draft (living register, opened 2026-07-14). This is NOT a single sequenced
plan like the others in this directory; it is a backlog of 12 independent
candidate features. Each feature below carries its own `Status` line and an
ID (CB-*, CMP-*, RT-*). Nothing here is Approved for execution yet -- the
whole document is a menu to pick from.

## How to use this document (update protocol)

1. Each feature has a stable ID and its own **Status** field. Per-feature
   status legend:
   - **Proposed** -- captured here, not yet decided on.
   - **Accepted** -- decided worth doing; ready to schedule (promote to a
     standalone plan file if it grows past a couple of PRs).
   - **In Progress** -- implementation started.
   - **Implemented** -- shipped and verified on the host. Record the commit
     or PR in the feature's Decision log line.
   - **Rejected** -- decided against. Move a one-line reason + date into the
     "Rejected after review" section; leave a tombstone line in the dashboard
     so the ID is never reused.
   - **Deferred** -- worth doing, blocked on something (named in Depends-on).
2. When a feature changes state, update BOTH the dashboard row and the
   feature's own `Status` line, and append a dated line to the Decision log.
3. Keep IDs stable. If a feature is split, suffix (CB-1a / CB-1b); never
   renumber.

## Provenance and verification (read before implementing)

Source: the odysseus repo (github.com/pewdiepie-archdaemon/odysseus), a
self-hosted AI workspace (Python/Flask, its own web UI + agent loop). It is
the analytic/predict-first mirror of devai's measure-first design, so most of
these ideas COMPOSE with devai's probe/bench caches rather than replace them.
Nothing was ported wholesale -- odysseus is a chat app, devai is serving
infrastructure with external agent CLIs.

Ideas were mined by a 6-agent read of the odysseus source cross-referenced
against devai's known capabilities, then de-duplicated and ranked. Two areas
were also read first-hand for this document. Every feature is tagged with a
**Verification** field:

- **first-hand** -- the odysseus source AND the relevant devai code path were
  read directly while writing this doc. Claims are grounded.
- **agent-read** -- the odysseus file:line anchors come from the mining agents
  and were NOT opened by hand. Treat the odysseus mechanism description as
  "reported, plausible, unverified"; re-open the cited file before building.

Value/effort ratings are estimates, not measurements. Re-scope before
promoting a feature to Accepted.

## Feature dashboard

| ID     | Feature                                                        | Area     | Value  | Effort | Verify     | Status   |
| ------ | -------------------------------------------------------------- | -------- | ------ | ------ | ---------- | -------- |
| RT-1   | SSE keepalive heartbeat during cold-start                      | Router   | high   | low-med| first-hand | Proposed |
| CB-1   | Analytic pre-probe fit/TPS/quality estimator                   | Cookbook | high   | med    | first-hand | Proposed |
| CB-2   | Serve-error -> auto-fix retry loop (grows recovery-flags.json) | Cookbook | high   | med    | agent-read | Proposed |
| CB-3   | Ingest vllm-project/recipes as authoritative launch config     | Cookbook | high   | med    | agent-read | Proposed |
| RT-2   | Conversational model-mgmt MCP write tools                      | Router   | high   | med    | agent-read | Proposed |
| RT-3   | Fail-closed tool policy for the MCP gateway                    | Router   | high   | med    | agent-read | Proposed |
| CMP-1  | Blind A/B human-preference duel + PREF% picker column          | Compare  | high   | med    | first-hand | Proposed |
| RT-4   | Foreground-priority gate (background jobs yield)               | Router   | high   | med    | agent-read | Proposed |
| RT-5   | Consolidated /health/all readout + external-GPU scan           | Router   | med-hi | med    | agent-read | Proposed |
| CMP-2  | LLM-as-judge pairwise scoring as a bench task                  | Compare  | high   | high   | agent-read | Proposed |
| CB-4   | Use-case-weighted ranking presets over bench columns           | Cookbook | med    | low    | first-hand | Proposed |
| RT-6   | Readiness probe distinct from liveness                         | Router   | med    | low    | agent-read | Proposed |

## Dependencies and relationship to existing plans

- RT-2 (MCP write tools) MUST land with or after RT-3 (fail-closed policy):
  exposing serve/download/stop tools without a policy gate is a privilege
  escalation. RT-3 is a hard prerequisite.
- CB-1 (estimator) feeds an optional predicted-TPS tiebreak into
  [Plan: gpu-arbiter-cluster-mode](./gpu-arbiter-cluster-mode.md)'s
  `routing_policy.go`, and a hypothetical-GPU sizing input into
  [Plan: skypilot-fleet-provisioner](./skypilot-fleet-provisioner.md). Both
  are optional wire-ups, not blockers.
- CB-2 (serve-error diagnosis) and CB-3 (recipe ingestion) both write into the
  existing `deploy/recovery-flags.json` + `scripts/model-families.yaml`
  machinery from [Plan: model-lifecycle-ledger](./model-lifecycle-ledger.md);
  they extend, not replace, its verdict flow.
- CMP-1 and CMP-2 both extend the bench/cache schema owned by
  [Plan: bench-rewrite](./bench-rewrite.md) (schema v3, per-ctx rows,
  `_meta.host_env_id` stamping). CMP-1 adds a sibling `.compare-cache.json`;
  CMP-2 adds a new `tasks` subset to `.bench-cache.json`.
- RT-4 (priority gate) and RT-6 (readiness) touch the same
  register/heartbeat/idle-sweep surfaces as cluster-mode; land after
  cluster-mode Phase 1 is stable.

## Recommended sequencing

1. **RT-1** first -- highest value-to-effort, self-contained in the router,
   no schema or policy dependencies.
2. **CB-4** and **RT-6** as low-effort quick wins (a picker sort key; a
   `/ready` handler).
3. **CB-1** next -- unlocks the picker `est` cells, `make model-rank`, and the
   two optional downstream wire-ups.
4. **CB-3** then **CB-2** -- recipe ingestion seeds good launch configs;
   auto-heal catches what the recipe misses. Order matters: a recipe-seeded
   flag reduces the errors the auto-heal loop has to diagnose.
5. **RT-3** then **RT-2** -- policy before write tools, always.
6. **RT-5** (aggregate health) and **CMP-1** (compare) independently.
7. **CMP-2** last -- highest effort, and it benefits from CMP-1's prompt
   corpus and the leaderboard being populated first.

---

## Cookbook features

### CB-1 -- Analytic pre-probe fit / TPS / quality estimator

- **Status:** Proposed
- **Value / Effort:** high / medium
- **Verification:** first-hand (read `services/hwfit/fit.py`;
  cross-checked devai's `docs/llm-tokens-and-speed.md` bandwidth model)
- **Depends on:** none (optional wire-ups into cluster-mode + fleet-provisioner)

**What odysseus does.** `hwfit` ranks the entire HF catalog with ZERO
launches. `estimate_memory_gb = params_b * quant_bpp + 8e-6 * active_params_b
* ctx + 0.5` -- note the KV-at-context term, which devai's discovery step
omits. A bandwidth-bound decode estimate `raw_tps = (bw / model_gb) * 0.55`
runs over an ~80-GPU bandwidth table (plus Apple-Silicon by core count and a
harmonic CPU-offload blend). Numeric quant tables (`QUANT_QUALITY_PENALTY`,
`QUANT_SPEED_MULT`: FP8/BF16 0, NVFP4 -3, Q4_K_M -5, QAT-INT4 -1,
FP4-MoE-Mixed -0.5) feed a composite quality/speed/fit/context score with
perfect/good/marginal/too_tight tiers, and a "what if I had Nx GPU of V GB"
override re-ranks against hypothetical hardware.
Source: `services/hwfit/fit.py:184-235,592-598`, `services/hwfit/models.py:26-62,192-207`,
`routes/hwfit_routes.py:29-112`.

**devai gap.** devai only surfaces a (model, backend) row when a REAL probe
cell exists; the picker shows `-` for un-benched TPS/quality. There is no way
to rank an un-probed model, estimate TPS on a GPU the box does not own, or
account for the KV/context term in `catalog-discover`'s weight-only VRAM band.
Cluster routing scores by fit tier, never by predicted throughput on a
heterogeneous worker's GPU. devai already has the theory (the bandwidth-bound
math is derived in prose in `docs/llm-tokens-and-speed.md`) but never
operationalizes it.

**Proposed change.** Add `scripts/_fit_estimate.py` with (a)
`estimate_memory_gb` including a KV-at-ctx term over a quant-BPP table, (b) a
bandwidth-bound TPS estimate keyed by `DEVAI_GPU_TYPE`, (c) a quant
quality-penalty table. Wire it three ways:
- `make model-rank` / picker pre-filter that scores every `deploy/models.yaml`
  row (perfect/good/marginal/too_tight) against `GPU_MEMORY_GB` before probing,
  with a `VRAM=` override to simulate a hypothetical or cloud-burst card.
- Fill the picker's `-` cells with a labelled `est` value.
- Feed per-worker predicted TPS into `gpu-arbiter/routing_policy.go` tiebreaks.

Probe/bench caches stay AUTHORITATIVE; estimates are labelled fallback only
and are overridden wherever a measured cell exists. This is strictly
better-grounded than odysseus, whose ranking never gets to measure.

**Deliverables.**
```
scripts/_fit_estimate.py           new    -- estimate_memory_gb + TPS + quant tables
scripts/select-models.py           modify -- optional `make model-rank` entrypoint
scripts/model-picker.py            modify -- render labelled `est` in empty TPS/quality cells
gpu-arbiter/routing_policy.go      modify -- optional predicted-TPS tiebreak (behind a flag)
docs/llm-tokens-and-speed.md       modify -- cross-link the estimator to the derivation
```

**Exit criteria.**
- `make model-rank` prints a perfect/good/marginal/too_tight tier for every
  catalog row without launching a container.
- `VRAM=80` re-ranks the same catalog as if on an 80 GB card.
- Picker shows `est ~N t/s` (clearly labelled) only where no bench value
  exists; a measured value always wins.
- The estimator's predicted TPS for an already-benched model is within a
  documented error band of the measured value (calibration check, mirroring
  odysseus's "~59 est vs 59.8 measured").

**Risks / notes.** The bandwidth table must key off a GPU label devai actually
knows (`DEVAI_GPU_TYPE`); an unknown card degrades to "no estimate," never a
wrong one. Keep the `est` label unmissable so an operator never mistakes a
guess for a measurement.

### CB-2 -- Serve-error to auto-fix retry loop

- **Status:** Proposed
- **Value / Effort:** high / medium
- **Verification:** agent-read (`routes/cookbook_helpers.py:1208-1381`,
  `src/tools/cookbook.py:334-358` NOT opened by hand -- re-verify the regex
  table before porting)
- **Depends on:** none (extends model-lifecycle-ledger machinery)

**What odysseus does.** `cookbook_helpers._diagnose_serve_output` matches ~25
regexes over the TAIL of serve output (KV/CUDA OOM, tensor-parallel-not-
divisible, ctx-too-large, missing/wrong tool-call-parser, trust-remote-code,
ModelOpt lm_head, port-in-use, missing deps, gated repo) and returns
`{message, suggestions:[{op: replace|append|remove, flag, value}]}`.
`_cookbook_apply_retry_suggestion` mechanically applies the fix and relaunches
instead of guessing.

**devai gap.** The router aborts on failure (`detectLaunchFailure`) but has no
error-signature -> concrete-fix mapping and no auto-retry. A KV OOM or wrong
tensor-parallel size just fails. The operational knowledge (drop
`--max-model-len`, add `--enforce-eager`, switch parser, lower
`--gpu-memory-utilization`) is applied BY HAND into `deploy/recovery-flags.json`
after a manual post-mortem.

**Proposed change.** Add a `devai-tools/internal/diagnose` Go package porting
the error-signature -> fix table. Call it from two places:
- The router's `waitForHealthy` / `detectLaunchFailure` path, to attach a
  structured `X-DevAI-Warning` suggestion to the failure.
- A new `make probe` retry loop that, on a matched OOM / divisibility / parser
  error, auto-appends the suggested flag to that model's
  `deploy/recovery-flags.json` and re-probes ONCE.

Turns probe pass/fail into pass / fix-and-retry.

**Deliverables.**
```
devai-tools/internal/diagnose/       new    -- regex->fix table + table-driven tests
gpu-arbiter/main.go                  modify -- attach X-DevAI-Warning on launch failure
scripts/_probe_hf_common.py          modify -- single auto-fix retry on a matched signature
deploy/recovery-flags.json           data   -- grown automatically by the retry loop
```

**Exit criteria.**
- A deliberately-too-large `--max-model-len` probe fails once, the loop
  detects "ctx too large / KV OOM," appends a corrected flag, and the second
  probe passes -- with the applied fix recorded.
- A launch failure that matches no signature is reported verbatim (no silent
  wrong-fix), preserving current fail-fast behaviour.
- The retry is capped at ONE attempt per probe so a mis-diagnosis cannot loop.

**Risks / notes.** A wrong auto-fix that silently "passes" is worse than a
clean failure. Every auto-applied flag must be logged and attributable, and
the loop must refuse to retry more than once. Confirm the exact odysseus regex
set is not over-broad before porting.

### CB-3 -- Ingest vllm-project/recipes as authoritative launch config

- **Status:** Proposed
- **Value / Effort:** high / medium
- **Verification:** agent-read (`routes/cookbook_routes.py:3823-3997`,
  `static/js/cookbook.js:342-491`)
- **Depends on:** none (pairs naturally before CB-2)

**What odysseus does.** Cookbook fetches raw
`vllm-project/recipes/main/models/<org>/<model>.yaml` live (cached) and
normalizes `base_args`/`base_env`, `tool_calling.args`, `reasoning.args`,
per-precision `vram_minimum_gb` + extra flags, `hardware_overrides`,
`min_vllm_version`. A manifest pull badges rows that have an official recipe.
Separately, `cookbook.js` maps model NAME to parser / MoE-env / KV-dtype /
spec-decoding heuristics with a human tip each, as an instant zero-probe best
guess.

**devai gap.** devai hand-curates tool/reasoning parsers in
`scripts/model-families.yaml` and per-model flags in
`deploy/recovery-flags.json`, discovered the hard way via probe failures.
There is no ingestion of any upstream authoritative launch-config source, and
a brand-new un-probed model gets NO parser at all (tools stripped by
`maybeStripTools`) until a full probe runs.

**Proposed change.** Add `scripts/fetch-vllm-recipes.py` that pulls the recipe
YAML per catalog repo (cache 6-12h) and seeds `model-families.yaml`
`parsers:` + `recovery-flags.json` `engine_flags`/`engine_env` + a
`min_vllm_version` guard, so probing STARTS from the vendor-recommended
command. Stamp a `recipe: verified|absent` field the picker shows as a badge
next to its probe-verified TOOLS column. Add a name -> parser/MoE/KV/spec
heuristic as a labelled "guessed, not probe-verified" fallback so the
resolution cascade is **recipe > heuristic > none**, with a real probe still
the final authority.

**Deliverables.**
```
scripts/fetch-vllm-recipes.py      new    -- fetch + normalize + cache recipe YAML
scripts/model-families.yaml        modify -- recipe-seeded parsers: block (marked as seed)
deploy/recovery-flags.json         data   -- recipe-seeded engine flags/env
scripts/model-picker.py            modify -- render a recipe badge on the TOOLS column
```

**Exit criteria.**
- For a model with an official recipe, `parsers:` + engine flags are populated
  from the recipe BEFORE any probe runs, and the first probe launches with the
  vendor command instead of a bare one.
- The picker shows a recipe badge; a probe-verified value always overrides a
  recipe-seeded one.
- A model with no recipe falls back to the name heuristic (labelled) and then
  to the current probe-from-scratch behaviour.

**Risks / notes.** The recipe repo is a network dependency; the fetcher must
be offline-first (cache reuse, no hard failure when unreachable), matching
devai's `make fetch-cli` posture. A recipe can lag the pinned vLLM image;
`min_vllm_version` must gate application so a recipe never injects a flag the
pinned image rejects.

### CB-4 -- Use-case-weighted ranking presets over bench columns

- **Status:** Proposed
- **Value / Effort:** medium / low
- **Verification:** first-hand (read `services/hwfit/fit.py` ranking path)
- **Depends on:** none

**What odysseus does.** `rank_models` scores
`quality*wq + speed*ws + fit*wf + context*wc` with per-use-case
`USE_CASE_WEIGHTS` (reasoning weights quality 0.55 / speed 0.15; chat weights
speed 0.35) and per-use-case `SPEED_TARGET` / `CONTEXT_TARGET` normalization
anchors (reasoning 25 t/s and 8192 ctx; embedding 200 t/s and 512 ctx), plus
use-case match bonuses. Source: `services/hwfit/fit.py:54-74,259-311,592-598`.

**devai gap.** The picker has fixed sort modes and a fixed
`TOTAL% = mean(gsm8k, humaneval, tools)`. There is no "optimize this ranking
FOR coding" vs "FOR reasoning" preset that re-weights speed/ctx/quality by
task, and no per-use-case speed/ctx normalization anchor.

**Proposed change.** Add a use-case preset (env var or a picker sub-modal)
that re-weights the EXISTING bench columns (CODE% / REAS% / TPS / CTX / VRAM)
into a task-tuned composite before sorting: "coding" weights CODE% + TPS;
"reasoning" weights REAS% + long-CTX and tolerates lower TPS; "chat" weights
TPS. Purely a new sort key in `scripts/model-picker.py` over data devai
ALREADY measures -- strictly better-grounded than odysseus's param-tier
quality proxy. Port the `SPEED_TARGET` / `CONTEXT_TARGET` anchors for
normalization.

**Deliverables.**
```
scripts/model-picker.py            modify -- use-case preset -> weighted composite sort key
```

**Exit criteria.**
- A `coding` preset re-orders the picker to favour high CODE% + high TPS rows;
  a `reasoning` preset favours high REAS% + large CTX and tolerates lower TPS.
- The preset is a new sort mode alongside the existing ones; no bench data is
  recomputed, only re-weighted.

**Risks / notes.** Weight tables are opinionated; expose them as constants at
the top of the picker so they are easy to tune. Keep the raw columns visible
so the composite never hides the underlying numbers.

---

## Compare features

### CMP-1 -- Blind A/B human-preference duel + PREF% picker column

- **Status:** Proposed
- **Value / Effort:** high / medium
- **Verification:** first-hand (read `routes/compare_routes.py` in full)
- **Depends on:** bench-cache schema conventions (bench-rewrite)

**What odysseus does.** Compare runs one prompt against N (model, endpoint)
pairs in side-by-side panes, randomizes them into neutral "Model A" / "Model
B" slots, WITHHOLDS identities from both the API response and the session
names, and reveals only AFTER the human votes. Each pane captures TTFT /
tok-s / cost with a "Fastest" badge; votes persist and a Scoreboard aggregates
per-model Win% / W-L-T / avg-cost. Confirmed detail: their server-side winner
table is WRITE-ONLY -- `/vote` records a winner but nothing reads it back into
a ranking, a loop devai can close properly.
Source: `routes/compare_routes.py:70-318`, `static/js/compare/*`.

**devai gap.** devai has only an offline OBJECTIVE bench (gsm8k / humaneval /
tools / leak). It cannot capture subjective human quality preference (tone,
instruction-following, code taste) between two locally-served models -- exactly
the axis exact-match scoring cannot measure -- and has no bias-controlled way
for an operator to judge their own fleet.

**Proposed change.** Add `make compare PROMPT=... A=<name>@<ctx>
B=<name>@<ctx>` (and a picker multi-select action) that hits the router ports,
prints "Response A" / "Response B" under a RANDOM per-duel A<->B mapping, takes
an A/B/tie vote, then reveals. Because of GPU mutual exclusion the duel runs
SEQUENTIALLY (serve A -> capture -> swap -> serve B -> vote); true-parallel
only across cluster workers. Print a per-response footer (TTFT / tok-s /
peak-VRAM + fastest badge) reusing router/bench metrics. Persist votes to
`deploy/.compare-cache.json` (same `_meta.host_env_id` stamping as
`.bench-cache.json`), add a PREF% (Wilson lower-bound or win-rate) column +
sort mode to `model-picker.py`, and a `make compare-report`. Support `RANDOM=1`
to auto-draw two fitting, non-ledger-excluded rows.

**Deliverables.**
```
scripts/compare/compare_runner.py  new    -- sequential duel driver over router ports
deploy/.compare-cache.json         data   -- preference ledger (gitignored, host_env stamped)
scripts/model-picker.py            modify -- PREF% column + sort mode
scripts/compare/compare_report.py  new    -- `make compare-report` leaderboard
Makefile                           modify -- compare / compare-report targets
```

**Exit criteria.**
- A duel serves A, captures, swaps to B, captures, presents blind, records a
  vote, and reveals identities only after the vote.
- Left/right (A/B) mapping is randomized per duel and never leaks before the
  vote (mirrors odysseus issue #1285's fix -- neutral slot names).
- Votes accumulate in `.compare-cache.json`; PREF% renders in the picker and
  `make compare-report` prints a per-model win-rate leaderboard.

**Risks / notes.** The GPU-exclusion swap makes a duel slow (two cold starts
possible); document that up front and reuse keep-warm where the two models
share a backend. Human samples are small -- use a Wilson lower bound, not raw
win%, and mark PREF% ADVISORY so it never becomes a hard picker filter.

### CMP-2 -- LLM-as-judge pairwise scoring as a bench task

- **Status:** Proposed
- **Value / Effort:** high / high
- **Verification:** agent-read (`routes/skills_routes.py:778-943`,
  `services/memory/memory_extractor.py:487-659`,
  `static/js/compare/index.js:761-869`)
- **Depends on:** CMP-1 (prompt corpus) + a populated leaderboard

**What odysseus does.** `skills_routes._audit_one_skill` runs a
test -> judge -> fix -> retry loop: a student model runs a synthesized task,
an LLM judge emits pass/fail, on fail the artifact self-edits, and a distinct
stronger TEACHER rewrites it while the student re-runs (proving the student can
now succeed) before demoting to draft. `memory_extractor.audit_memories` wraps
LLM curation in a fingerprint short-circuit (skip if unchanged) and a "refuse
if the model returned <50% of the entries" over-deletion tripwire.

**devai gap.** All devai bench tasks are objective/deterministic; there is no
signal for open-ended answer quality (chat, reasoning prose, code taste), no
teacher/student pairing, and no self-validation of machine-generated artifacts
(recovery-flags entries and future runbooks are hand-curated). This is the
CORRECT reinterpretation of odysseus's "synthesis" for a serving lab -- NOT
odysseus's search-summary feature, which is out of scope.

**Proposed change.** Add a `pairwise_judge` task to
`scripts/bench/bench_runner.py`: for a curated prompt set, generate each
model's answer via the router, then run a judge model (through the same
router, GPU-exclusion-serialized) with a rubric prompt returning A|B|tie,
POSITION-SWAPPED and averaged to cancel order bias, storing a win matrix per
(ctx, host_env_id) under a new `.bench-cache.json` tasks subset. Pick judge /
teacher and student EMPIRICALLY from the leaderboard (highest CODE% / TOTAL%
fitting model = judge). Optionally extend to a teacher/student loop that
promotes an auto-generated recovery-flags entry draft -> published only on
student success. Wrap any LLM curation with fingerprint-short-circuit +
"refuse if >X% rows removed" rails. Gate judge output as ADVISORY (like the
leak caveat) -- LLM-judge verdicts are noisy and self-preferential -- never a
hard exclusion input.

**Deliverables.**
```
scripts/bench/bench_runner.py      modify -- pairwise_judge task (position-swapped)
scripts/bench/_bench_core.py       modify -- win-matrix storage in the tasks subset
scripts/bench/bench_report.py      modify -- render judge win-matrix (advisory-flagged)
```

**Exit criteria.**
- `pairwise_judge` produces an A|B|tie verdict per prompt, averaged over both
  position orderings, stored per (ctx, host_env_id).
- The judge is auto-selected from the leaderboard, not hardcoded.
- Judge results render as ADVISORY and never feed the exclusion ledger or a
  hard picker filter.

**Risks / notes.** LLM judges are self-preferential and order-biased -- the
position swap is mandatory, and the output must stay advisory. High effort:
this is the last item to schedule, and only after CMP-1 has a prompt corpus.

---

## Router / agent-surface features

### RT-1 -- SSE keepalive heartbeat during cold-start

- **Status:** Proposed
- **Value / Effort:** high / low-medium
- **Verification:** first-hand (confirmed router ordering in
  `gpu-arbiter/main.go`: `ensureBackendRunning -> containerRecreate ->
  waitForHealthy -> bs.proxy.ServeHTTP`, main.go ~2282-2548; client is held
  the entire recreate+health window with zero bytes sent)
- **Depends on:** none

**What odysseus does.** `agent_runs.subscribe()` writes `: heartbeat N\n\n`
SSE comment lines every 10s while a run is "running" and no token has arrived,
explicitly to reset browser/proxy idle timers and survive first-token
latencies of 30s+; `subprocess_tools` does the same for long shell jobs via a
2s progress emitter. Source: `src/agent_runs.py:177-189`,
`src/agent_tools/subprocess_tools.py:39-54`.

**devai gap.** The router flushes SSE straight through but injects NO heartbeat
while a backend is cold-loading or prefilling. NVFP4 cold starts run up to
`HEALTH_TIMEOUT_SECONDS=600`, and post-recreate first-token latency is large;
during that window no bytes flow, so a browser or corporate proxy
idle-timeout (devai ships `HTTP_PROXY`/`HTTPS_PROXY` support for exactly these
environments) can kill the connection and WASTE the expensive load.

**Proposed change.** In gpu-arbiter's streaming path, start a ticker that
writes `: keepalive\n\n` every ~10s from request-accept until the first real
upstream byte arrives, and during the `containerRecreate`/`waitForHealthy`
cold-load window before the upstream even connects. SSE comment lines are
ignored by every OpenAI/Anthropic client. Purely additive.

**Deliverables.**
```
gpu-arbiter/main.go                modify -- heartbeat ticker on the streaming path
gpu-arbiter/cluster_proxy.go       modify -- same for head-forwarded SSE
gpu-arbiter/*_test.go              new    -- assert comment frames emitted, real bytes still pass
```

**Exit criteria.**
- A `stream:true` request that triggers a 60s+ cold start receives periodic
  `: keepalive` comment frames and does NOT drop through a proxy with a 30s
  idle timeout.
- Real content bytes still flush unchanged; a non-streaming request is
  untouched.
- On a launch FAILURE after heartbeats began, the client receives an in-band
  SSE error event + `[DONE]`, not a hung stream.

**Risks / notes.** The honest wrinkle: two cases, different effort.
(1) "Upstream connected, slow first token" -- trivial: tee the stream, tick
until the first byte. (2) "Cold start before upstream connects" -- must commit
to a `200 text/event-stream` response BEFORE `waitForHealthy` succeeds, so a
subsequent launch failure can no longer return a 5xx; it must emit an in-band
SSE `error` event instead. Gate the whole behaviour on `stream:true` (the
router already parses the body for `maybeStripTools`), so non-streaming
requests keep their proper HTTP error status.

### RT-2 -- Conversational model-management MCP write tools

- **Status:** Proposed
- **Value / Effort:** high / medium
- **Verification:** agent-read (`src/tools/cookbook.py:190-233,420-600,1217-1341`,
  `src/agent_tools/model_interaction_tools.py:20-210`)
- **Depends on:** **RT-3 (hard prerequisite -- do not ship write tools without
  the policy gate)**

**What odysseus does.** `src/tools/cookbook.py` is a full agent tool suite:
`do_download_model`, `do_serve_model` (auto-registers an OpenAI endpoint so
the model is instantly routable), `do_stop`/`tail`/`list_served`,
`do_serve_preset`, `do_search_hf`. `model_interaction_tools` adds
`chat_with_model` and `ask_teacher` ("auto" -> a configured teacher model with
a senior-mentor system prompt) routed through one endpoint resolver. Presets
live as named `{model, host, port, cmd}` the agent launches by name.

**devai gap.** devai's only MCP surface (`devai-model-status`) is strictly
READ-ONLY. An external agent cannot download, serve, stop, tail, or benchmark
a model conversationally, nor escalate to a stronger model -- the exact
natural-language fleet management devai's "external agents point at the router"
thesis is positioned for but does not expose. This is the single biggest
capability gap for that thesis.

**Proposed change.** Extend `devai-tools/cmd/devai-mcp-modelstatus` with WRITE
tools that shell out to existing surfaces: `pull_model` (`make model-pull
NAME=`), `serve_model` (an `@ctx` request to the router port, returns the
endpoint), `stop_model`, `tail_backend_log` (`make logs SERVICE=`),
`run_bench` (`make bench`), plus `ask_teacher` that auto-selects the
highest-TOTAL% / largest-fitting model from the probe+bench caches (reuse
`devai-tools/internal/modelcache`). Support named launch presets via a
`presets:` block in `~/.devai/preferences.yaml` that materialize a canonical
`<name>::<reasoning>@<ctx>` string. Gate EVERY write tool behind the
admin/allowlist check from RT-3, under the gateway's existing
`--block-secrets`/`no-new-privileges`.

**Deliverables.**
```
devai-tools/cmd/devai-mcp-modelstatus/  modify -- add pull/serve/stop/tail/bench/ask_teacher tools
devai-tools/internal/modelcache/        reuse  -- teacher auto-select from bench cache
~/.devai/preferences.yaml               data   -- optional presets: block
docs/mcp-model-status.md                modify -- document the new write tools + policy gating
```

**Exit criteria.**
- An MCP-aware agent can, in one conversation, pull a model, serve it, get the
  endpoint back, chat through it, and stop it -- all through the gateway.
- Every write tool is refused for a non-admin/unallowlisted caller (RT-3), with
  a single-user fast-path.
- Read tools (`list_fitting_models`, `get_model_bench`, `get_router_status`)
  are unchanged.

**Risks / notes.** This is a privilege surface. It MUST NOT ship before RT-3.
`serve_model` under GPU mutual exclusion will evict whatever is warm -- pair
with RT-4 so a conversational serve does not stomp an interactive session
unless the caller is foreground.

### RT-3 -- Fail-closed tool policy for the MCP gateway

- **Status:** Proposed
- **Value / Effort:** high / medium
- **Verification:** agent-read (`src/tool_security.py`,
  `src/task_action_policy.py`, `src/task_scheduler.py:828-848`,
  `src/mcp_manager.py:46-132`)
- **Depends on:** none (prerequisite FOR RT-2)

**What odysseus does.** `tool_security` maintains a blocked set (bash / python
/ write_file / serve / download / manage_*) and fails CLOSED -- a malformed or
unknown tool name is treated as blocked. Plan mode inverts an allowlist
THROUGH the denylist so a newly-added or import-failed tool defaults to
BLOCKED. `task_action_policy` classifies money-spending / infra actions as
admin-only and the scheduler PAUSES (not runs) a non-admin task that hits one.
`mcp_manager` sanitizes untrusted tool schemas and classifies each tool
read-only vs write, failing closed on ambiguity.

**devai gap.** The gateway runs with `--block-secrets`/`no-new-privileges` but
has NO per-tool policy: any MCP-aware agent can call any exposed tool
(including shell / filesystem-write / serve / provision) at full privilege.
There is no read-only allowlist, no fail-closed on unknown tool names, and no
privilege gate on money-spending actions (SkyPilot launch,
container-recreate-for-download, serve).

**Proposed change.** Add a gateway-side (or router pre-filter) tool policy to
`deploy/mcp-servers.yaml` / the gateway config: default-deny a static set
(shell / filesystem-write / model-serve / provisioning), require an explicit
per-caller allowlist, and FAIL CLOSED on unknown/malformed names
(allowlist-through-denylist, so a new server's tools are blocked until
classified). Add a per-tool `readOnly` annotation (devai-model-status read
tools pass; RT-2's write tools are admin-only). Classify router/head
infra-spend actions (SkyPilot launch, download-recreate, serve) as privileged
and REFUSE -- not silently run -- when unauthorized, with a single-user
fast-path.

**Deliverables.**
```
deploy/mcp-servers.yaml            modify -- per-tool readOnly + denylist annotations
deploy/mcp-gateway.env             modify -- policy toggle + single-user fast-path
docs/mcp.md                        modify -- document the policy + fail-closed semantics
tests/test-mcp.sh                  modify -- assert an unknown tool name is blocked
```

**Exit criteria.**
- A tool name not in the classification defaults to BLOCKED.
- Read tools pass; shell/write/serve/provision tools are refused without
  explicit allowlisting.
- A single-user install has a fast-path that keeps the current behaviour
  ergonomic while still failing closed on unknowns.

**Risks / notes.** Over-blocking breaks Phase 1 Tier-1 servers if the default
denylist is too broad; classify the existing 10+4 servers explicitly so the
default policy is a no-op for them and only bites new/unknown tools.

### RT-4 -- Foreground-priority gate (background jobs yield)

- **Status:** Proposed
- **Value / Effort:** high / medium
- **Verification:** agent-read (`src/interactive_gate.py`,
  `src/task_scheduler.py:850-908`)
- **Depends on:** cluster-mode Phase 1 stable (touches the same idle surfaces)

**What odysseus does.** `interactive_gate` keeps an active-request counter +
browser heartbeat + detached-stream detector; background tasks
`wait_for_interactive_quiet()` until no active request, no activity for a quiet
window, and no live stream. A 250ms monitor cancels a running background task
the instant foreground activity resumes, so background work never competes
with the user on the shared model.

**devai gap.** Every router request is peer-priority. Under GPU mutual
exclusion a model swap is catastrophic (evicts a warm model, cold-start
seconds), yet nothing stops a low-priority background caller (`make bench`, a
scheduled probe, `model-sync`) from forcing a swap that evicts the user's
interactive model mid-turn.

**Proposed change.** Add a priority signal to gpu-arbiter (an
`X-DevAI-Priority: background` header, or a dedicated low-priority port). A
background request that would require a different `currentModel`/
`currentContext` BLOCKS until the resident model has been idle for a quiet
window (reuse `IDLE_TIMEOUT`) or returns 503 / queues, instead of preempting.
Set the background flag in `scripts/bench/bench_runner.py`,
`scripts/_probe_load.py`, and `scripts/model-sync.py` so batch tooling yields
to interactive serving.

**Deliverables.**
```
gpu-arbiter/main.go                modify -- honor X-DevAI-Priority: background before a swap
scripts/bench/bench_runner.py      modify -- set the background priority header
scripts/_probe_load.py             modify -- same
scripts/model-sync.py              modify -- same
```

**Exit criteria.**
- A background request that would evict a warm interactive model waits (or
  503s), and does NOT trigger a swap while the interactive model is in use.
- A foreground request always preempts a queued background one.
- With no interactive traffic, background jobs proceed exactly as today.

**Risks / notes.** Starvation: a permanently-busy interactive model could
block a background probe indefinitely. Add a max-wait after which the
background job either proceeds or fails loudly, so `make bench` cannot hang
forever.

### RT-5 -- Consolidated /health/all readout + external-GPU scan

- **Status:** Proposed
- **Value / Effort:** medium-high / medium
- **Verification:** agent-read (`src/service_health.py`,
  `src/tools/cookbook.py:312-417`)
- **Depends on:** none

**What odysseus does.** `service_health` fans out per-subsystem probes into
ONE report with a 4-state vocabulary (ok / degraded / down / disabled), each
bounded by a per-op timeout, an N-item fan-out budget (a thread pool with
`as_completed` timeout + `shutdown(wait=False)`), a per-subsystem deadline and
an overall ceiling. Errors pass through `_classify_error` (never `str(exc)`)
and every URL through `_safe_url` (strips userinfo/query/fragment). Separately
a `/proc` scan surfaces model servers the tracker did not launch.

**devai gap.** devai has per-backend `/health` and `get_router_status` but no
single aggregate verdict across ollama + vllm + sglang + MCP gateway +
SkyPilot server + cluster workers, no degraded-vs-down distinction, no
bounded-fan-out budget, and no secret-free error classification for the status
surface. The router also cannot see a NON-devai process already holding the
GPU it enforces exclusion over.

**Proposed change.** Add a `/health/all` handler to gpu-arbiter that fans out
(bounded goroutine budget + per-probe timeout + overall ceiling) to the 3
backends + MCP gateway + SkyPilot API server + registered workers, returning
ok / degraded / down / disabled per subsystem with SAFE category tokens
(timeout / connection_refused / dns / tls) -- never raw exceptions or
credential-bearing URLs. Surface it via `devai-tools/internal/routerclient`
`get_router_status`. Fold in a `podman ps` + host `/proc` scan so the readout
reports externally-launched vLLM/SGLang/Ollama processes occupying the GPU.

**Deliverables.**
```
gpu-arbiter/main.go (or new file)  modify -- /health/all bounded fan-out handler
devai-tools/internal/routerclient/ modify -- consume /health/all in get_router_status
scripts/... (optional)             new    -- podman ps + /proc external-GPU scan helper
```

**Exit criteria.**
- `/health/all` returns one JSON with ok/degraded/down/disabled per subsystem,
  bounded by an overall deadline even if a subsystem hangs.
- No raw exception text or credential-bearing URL ever appears in the output.
- An externally-launched process holding the GPU is reported.

**Risks / notes.** The `/proc` scan must degrade gracefully where the router
container cannot see host processes (rootless podman namespace); report
"unknown" rather than falsely "clear."

### RT-6 -- Readiness probe distinct from liveness

- **Status:** Proposed
- **Value / Effort:** medium / low
- **Verification:** agent-read (`src/readiness.py`)
- **Depends on:** cluster-mode Phase 1 (to gate worker registration)

**What odysseus does.** `readiness.py` exposes `/api/ready` as a STRICTER probe
than `/health` liveness: confirms the DB is reachable, the data dir exists and
is actually WRITABLE (writes + deletes a uuid probe file), and returns
`ready=True` only when every critical check passes -- suitable as an
orchestrator readiness gate.

**devai gap.** Router/backends and cluster workers expose only liveness
(`/health`, register + heartbeat). A worker can be "up" but unable to serve:
probe cache missing, GPU not visible, or the `/var/cache/devai` volume not
mounted/writable. Nothing verifies serve-readiness preconditions as a gate
distinct from process liveness.

**Proposed change.** Add a `/ready` endpoint to gpu-arbiter and the
worker-bootstrap arbiter that verifies GPU device visibility, that
`/var/cache/devai/*` mounts are present AND writable (uuid write + delete
probe), that the relevant probe cache exists, and that backend images are
pulled. Have the cluster head gate worker registration/routing on `/ready`
(not just liveness) so it never routes to a worker that cannot actually launch.

**Deliverables.**
```
gpu-arbiter/main.go                modify -- /ready handler (GPU + mounts + cache + images)
gpu-arbiter/cluster_head.go        modify -- gate worker routing on /ready
gpu-arbiter/cluster_worker.go      modify -- report readiness in register/heartbeat
```

**Exit criteria.**
- `/ready` returns not-ready when the GPU is invisible, a cache volume is
  unwritable, or the probe cache is missing -- even while `/health` says alive.
- The head does not route to a registered-but-not-ready worker.

**Risks / notes.** Keep the writable-probe cheap (one small uuid file) so
`/ready` stays fast enough to poll.

---

## Already covered by devai (rejected as redundant)

These odysseus ideas were considered and dropped because devai already does
them, usually better (measured vs estimated).

| odysseus idea                                              | Why devai already covers it                                                                 |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Analytic weight-VRAM band pre-screen                       | `catalog-discover` already estimates params x quant-bytes to hide too-big/too-small (only the KV-at-ctx term was new -> folded into CB-1). |
| HF downloads/likes popularity sort                         | `catalog-discover` narrows by tracked lineage + usable-VRAM band; popularity is a marginal secondary sort. |
| Computed llama.cpp n_cpu_moe / KV-type serve profiles      | Not applicable: vLLM/SGLang do not expose per-expert CPU offload; Ollama hides it.          |
| Per-quant quality-penalty/speed-mult ranking tables        | devai measures real CODE%/REAS%/LEAK% via bench; the table is useful only as an un-benched fallback (folded into CB-1). |
| Time-windowed scheduled serve with auto-stop               | `IDLE_TIMEOUT` keep-warm + `DEVAI_IDLE_MINUTES` cluster idle-sweep + SkyPilot two-step teardown already release the GPU. |
| Per-family gate for trusting structured tool_calls         | Probe-cache `disable_verified` + `maybeStripTools` gate this per-model, stronger than a hand-kept name list. |
| ReDoS-hardened forward-only output scanning                | The router is Go and never regex-parses response bodies; no ReDoS surface exists.           |
| Enforce tool gating at execution not prompt                | The MCP gateway (`--block-secrets`/`no-new-privileges`) and router (GPU exclusion, 429 caps) already enforce at the boundary in code. |
| Context-window discovery with proven-vs-fallback flag      | Probe caches measure real served ctx by launch and stamp `position_limit`; `_meta` image-drift re-probe covers never-trust-stale. |
| Provenance-tiered trust gating (fail-closed machine-authored)| verified-vs-manual probe outcomes + exclusion-ledger stability rules + `serving_ok` gate-only-when-present already embody this. |
| Embedding-collection fingerprinting on drift               | devai stamps `_meta.current_image_digest` and re-probes on drift; only relevant if a vector store is ever added. |
| NPM/binary pre-cache probe with one-line fix               | `make fetch-cli` offline-first pre-fetch + `verify-backend-flags` already probe presence before use. |

## Rejected after review (out of architectural fit)

Considered and rejected because they violate a devai invariant (no
message-inspection, no own agent loop) or fall outside the user's scope
(documents/email/notes/etc). Kept here so they are not re-proposed.

| odysseus idea                                              | Reason rejected                                                                             |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Curated image-generation model registry                    | Image generation is explicitly out of scope.                                                |
| Search-result "synthesis" LLM summary                      | Web search is out of scope; the useful judge primitive is captured in CMP-2.                |
| Detached-run replay buffer (resume after disconnect)       | High effort, low payoff: external CLIs rarely reconnect; one generation per backend under GPU exclusion. |
| Multi-GPU homogeneous-pool tensor-parallel grouping        | devai is single-GPU today; parked as future cluster design input.                           |
| RAG-based per-turn tool selection + keyword force-include  | Chat-app internal; devai delegates tool selection to external agent CLIs.                    |
| Router-tap auto-extraction of durable user facts           | Directly violates the "router does no message inspection" invariant; would need an out-of-path sidecar. |
| Untrusted-content guard wrapper for tool/MCP output        | Prompt-injection defense is the external agent's job by design; devai owns no prompt construction. |
| OAuth-capable remote MCP client with encrypted token refresh| Brushes the ignore-auth scope and is high effort; Tier-2 static sops secrets already cover API-key MCP servers. |
| Structural history-trim / orphaned-tool-message repair     | Crosses into request-body inspection the router deliberately avoids; low value.             |
| Endpoint base-URL validation / per-hop SSRF re-check       | Private-network backends + fixed public fetch hosts make the threat marginal.               |
| Zero-config Tailscale/port-scan endpoint discovery         | Niche versus explicit token-authenticated worker registration; fingerprinting overlaps RT-5.|
| Cron-driven maintenance scheduler                          | High effort; manual make targets + cluster idle-sweep cover most, and it depends on RT-4.    |

## Decision log

- 2026-07-14 -- Register opened. 12 features captured (Proposed), 12 ideas
  marked already-covered, 12 rejected as out-of-fit. RT-1 recommended as the
  first item to schedule (highest value-to-effort, self-contained). No feature
  Accepted yet.

## References

- odysseus -- github.com/pewdiepie-archdaemon/odysseus (Cookbook = `services/hwfit/*`
  + `routes/cookbook_*`; Compare = `routes/compare_routes.py` + `static/js/compare/*`;
  Chat+Agents = `src/agent_*`, `src/tool_*`, `src/mcp_manager.py`, `services/memory/*`).
- vllm-project/recipes -- the authoritative per-model launch-config repo CB-3 ingests.
- devai internal: `docs/router.md`, `docs/backends.md`, `docs/nvfp4-coldstart.md`,
  `docs/llm-tokens-and-speed.md`, and the sibling plans linked under
  "Dependencies and relationship to existing plans" above.
</content>
</invoke>
