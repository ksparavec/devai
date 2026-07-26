# devai plans -- execution ordering

This file is the canonical sequencing reference for the plans
under `docs/plans/`. Each individual plan owns its own
dependency edges (in its `## Dependencies` section); this file
synthesises them into a recommended order of execution. When
the two disagree, the per-plan `Dependencies` section is
authoritative -- update this file to match.

## Status legend

- **Draft** -- design under discussion, decisions not all locked.
- **Approved** -- decisions locked; ready to schedule.
- **In Progress** -- some phase has been started.
- **Done** -- all required phases shipped and verified.
- **Done (unverified)** -- all required phases code-complete and
  unit-tested, but the live verification the plan itself requires has
  not been run. Code-complete, NOT verified-in-production.
- **Non-functional** -- code-complete and its tests pass, but the
  feature does not work when actually exercised. This status exists
  because "Done (unverified)" was hiding four such cases: it reads as
  "works, just not tested here", when the truth was "never executed".
  A plan may only leave this state by being run.
- **Frozen** -- deliberately parked. Not maintained, not compiled, not
  deleted. Sources live under `attic/` with a restore guide. See
  `attic/README.md`.
- **Superseded** -- the plan's intent shipped by a different and better
  route, or its premise expired. The plan text is kept as a record but
  must not be executed as written.

Current snapshot (as of **2026-07-25**), rewritten after a portfolio
review that verified every claim against the tree and, where possible,
by running the thing. That review found the previous snapshot
systematically over-stated: four plans marked "Done (unverified)" were
not merely untested, they **did not work**, and their test suites passed
anyway because those suites asserted the shape of YAML and Go files
rather than behaviour.

What changed on 2026-07-25:

- **`gpu-arbiter-cluster-mode` and `skypilot-fleet-provisioner` -> Frozen.**
  6,643 lines of Go written in a single day (2026-05-15, three commits)
  and touched once since, as part of a blanket review sweep rather than
  by use. The head crash-looped by construction (`log.Fatalf` on a token
  file compose never mounted, control plane never published), routing
  ignored the probe-cache/GPU-fit lookup that was supposed to be its
  point, and every SkyPilot constructor had zero non-test callers. Moved
  to `attic/cluster-mode/` behind a build tag, with the open defects
  catalogued in `attic/cluster-mode/RESTORE.md`. Nothing was deleted.
- **`mcp-gateway` -> Done.** Previously non-functional: the catalog used
  a schema the gateway does not parse (it produced an empty registry, so
  zero tools), and all 14 servers were pinned at `:0.7.0`, a tag that
  exists for none of them -- `mcp/hugging-face` is not a repository at
  all. Rebuilt to defer to Docker's official catalog and carry only the
  first-party `devai-model-status` entry. **Verified live: 134 tools over
  a real MCP handshake, including an end-to-end `tools/call`.**
- **`sops-age-secrets` -> Non-functional.** `.sops.yaml` still carries
  the literal `age1xxxx...` placeholder, so it cannot encrypt anything;
  no age key has ever been generated on this host; only `.example` files
  exist. The unit test that should catch the placeholder passes on it.
  Two of its three consumers are now frozen.
- **`kv-cache-quantization` -> Superseded.** Its engine-side substance
  shipped independently under a better per-probe-cell design
  (664bc76 / 8325255). Its Phase 2 headline -- flip Ollama to q8_0
  globally -- is now contradicted by the repo's own measured GPQA
  regression and must NOT be executed. The one part that had not
  shipped, backend-aware fit math, was extracted and shipped on
  2026-07-25.
- **`skypilot-agent-skill` -> Frozen.** Its Phase 1 shipped documentation
  that was false in three places, including a claim in `CLAUDE.md` that
  the lab image bundles an Agent Skill plugin. Nothing installed one.
  The claims are corrected; the `sky` CLI itself stays.
- **`odysseus-borrowed-ideas` -> Draft (partly frozen).** RT-1, CB-4 and
  CB-1's KV term are approved for extraction; RT-2, RT-3, CMP-1 and
  CMP-2 are rejected rather than left Proposed to accrete legitimacy.
- **`bench-rewrite` stays In Progress, but Phase 6 is struck.** The
  backfill targets a 32K/64K/128K/256K probe grid that no longer exists
  -- the probers now keep exactly one winner cell per (model, backend).
  Executing it would write bench rows the picker cannot read. Its real
  intent (14 of 27 pickable rows have no bench data at the ctx the picker
  offers; all 8 SGLang rows have none at all) is re-filed under
  bench-sync.

Note on `review-fixes-2026-07`: executed 2026-07-23 in **three** passes
(remediation, then two adversarial-review repair rounds -- each round
found defects the previous round had itself introduced). All nine phases
were worked. It stays **Done (unverified)** because the live-GPU tests it
requires before Phase 3 may merge (`make test-router`, `make test-vllm`)
have still not been run. Worth recording alongside it: that remediation
landed as a single 120-file direct commit on `main`, and
`security-blocking.yml` triggers only on `pull_request` -- so the largest
change in recent history never ran gitleaks or CodeQL, and the next day
found three more findings.

| Plan                                                                | Status            |
| ------------------------------------------------------------------- | ----------------- |
| [mcp-gateway](./mcp-gateway.md)                                     | **Done**          |
| [model-lifecycle-ledger](./model-lifecycle-ledger.md)               | Done (unverified) |
| [review-fixes-2026-07](./review-fixes-2026-07.md)                   | Done (unverified) |
| [bench-rewrite](./bench-rewrite.md)                                 | In Progress       |
| [kv-cache-quantization](./kv-cache-quantization.md)                 | Superseded        |
| [card-derived-hints-and-bench-sync](./card-derived-hints-and-bench-sync.md) | Draft     |
| [router-shortcircuit](./router-shortcircuit.md)                     | Draft             |
| [odysseus-borrowed-ideas](./odysseus-borrowed-ideas.md)             | Draft (partly frozen) |
| [pi-coding-agent](./pi-coding-agent.md)                             | Draft             |
| [router-fanout](./router-fanout.md)                                 | Draft             |
| [sops-age-secrets](./sops-age-secrets.md)                           | Non-functional    |
| [skypilot-agent-skill](./skypilot-agent-skill.md)                   | Frozen            |
| [gpu-arbiter-cluster-mode](./gpu-arbiter-cluster-mode.md)           | **Frozen**        |
| [skypilot-fleet-provisioner](./skypilot-fleet-provisioner.md)       | **Frozen**        |

## Defects found 2026-07-25/26 -- all closed

Surfaced while onboarding SGLang for the 5 kept vLLM models. Every one
was found by running the pipeline, not by reading it. Kept here because
several of the conclusions were revised more than once, and the revisions
are the useful part.

1. **CLOSED -- the router recreated a failing model without limit.**
   Fixed by the circuit breaker in `gpu-arbiter/main.go`
   (`noteLaunchAttempt` / `noteLaunchSucceeded` / `launchBudgetExhausted`,
   budget `DEVAI_MAX_FAILED_LAUNCHES`, default 3). A launch spends one
   unit; only a request actually completing upstream repays it, because
   reaching `/health` is exactly what the failing case already does. On
   exhaustion the router refuses with a message naming the model, the
   context, and both remedies. Note the reset is deliberately UNKEYED:
   the attempt is charged against the resolved ctx while the request is
   served at the launched ctx, and any drift between those would leave
   the budget permanently unreset and eventually refuse a working model.

   Original report follows. Benching
   `Ornith-1.0-9B-NVFP4` at 256K on SGLang killed the engine under
   sustained load; the router recreated it **100 times in about an hour**
   and never gave up, completing zero tasks. `detectLaunchFailure`
   catches an engine that dies *during* launch; nothing catches one that
   launches cleanly and then dies serving. Note `podman inspect` reports
   `RestartCount: 0` throughout -- each cycle is a NEW container, so that
   counter is useless for detecting this. Wants a circuit breaker: after
   N recreates of the same (model, ctx) with no completed request,
   refuse and name the model. Severity: a single client asking for a
   marginal model can pin the GPU indefinitely, no privileges needed.

2. **CLOSED -- `needle_score` was recorded and read by nothing.**
   Repaired and surfaced as a warning; deliberately NOT a gate.
   `_probe_load.py` now writes `None` (not 0.0) when the serve failed,
   captures `serving_finish_reason`, stores a response excerpt, and
   derives `needle_valid` (not failed AND finish_reason != length AND
   fill_ratio >= 0.9). `model-picker.py`'s `_needle_failed_at` shows a
   hedged `Recall:` line only when the probe itself vouches for the
   measurement; a test asserts no consumer gates on it. Original
   analysis follows -- it is why gating would have been harmful.

   `needle_score` gated nothing, and on this fleet's data it must
   not, yet. The load probe records it; the picker, the router and
   `devai-tools/internal/modelcache` all ignore it.

   An earlier revision of this entry claimed the needle "was the ONLY
   signal that predicted defect 1". **That claim was wrong and is
   withdrawn.** Defect 1 was the router destroying healthy backends, so
   nothing about the model predicted it. Worse, gating on the field as it
   stands would actively harm: across 23 load-probed cells there are 4
   zeros, and 3 of them terminated at exactly `serving_output_tokens ==
   2048` -- the output ceiling -- while none of the 19 cells scoring 1.0
   came within 4x of it. All 4 zeros are reasoning models that spent the
   answer budget on a `<think>` trace. On this host `needle_score == 0.0`
   is close to a synonym for "reasoning model truncated", not "cannot
   recall", and a gate would silently hide both DeepSeek-R1-Distill
   models on a 100% false-positive rate.

   The field is also structurally confounded: `_probe_load.py` sets
   `needle = 0.0 if failed else ...`, so every `serving_ok=false` cell
   carries a 0.0 that means nothing. Repair the measurement before
   considering any consumer -- see the plan in this file's history.

3. **The load probe tests one request, not a workload.** Status revised
   twice in one day; read the whole entry before acting.

   It was first raised on the strength of Ornith-1.0-9B-NVFP4
   "crash-looping under sustained load" at 256K on SGLang. When the
   router-recreate bug (item 1) was found, that evidence looked
   contaminated and the item was closed as not-a-defect on a code
   argument (reproduced below, and still largely sound).

   **Then it was re-tested on 2026-07-26 with the fixed router, and
   Ornith still dies.** The new per-reason teardown logging separates
   the cases cleanly: 3 of 10 teardowns during a gsm8k run were
   `container exited` -- the engine process genuinely terminating, not a
   router false positive. Meanwhile a forced LOAD probe on the same cell
   reproduced `serving_ok=True` byte-for-byte. So this model survives one
   near-full-context prefill and dies under a sustained workload, which
   is precisely the gap `serving_ok` cannot see.

   **What that does and does not overturn.** It does NOT justify adding a
   concurrency burst to the load probe -- the cost and false-OOM
   arguments below still hold, and a burst that writes an `oom` ledger
   exclusion on a queue-full 503 would hide good models. It DOES mean
   `serving_ok` must stop being described as sufficient, and that the
   bench harness is the load gate in practice. The open question is
   whether a model that fails a bench this way should be written back to
   the probe cache automatically instead of by hand.

   The code argument for why a burst is still the wrong instrument:
   `max_num_batched_tokens`
   defaults to 2048 on the pinned vLLM image, capping tokens per engine
   step regardless of how many requests are in flight, so the
   softcap-logits allocation this probe exists to catch is identical at
   concurrency 1 and 32. The paged KV pool preempts and recomputes
   rather than OOMing, and at the binary-search winning ctx the pool
   cannot hold two full windows anyway. `--enable-prefix-caching` (on in
   both prober and router) would make N copies of one needle prompt
   share KV blocks and measure nothing. And `_detect_serving_failure`
   turns any error body into `serving_ok=false`, which writes an `oom`
   ledger exclusion -- so a queue-full 503 under a burst would hide a
   good model permanently. Concurrency evidence belongs in the bench
   harness, which already drives real multi-task workloads through the
   router on models that have passed fit+load.

4. **CLOSED -- probe and serve disagreed on `--max-num-seqs`.** Both
   probers now emit it (`PROBE_MAX_NUM_SEQS`, default 32, matching the
   router's `MAX_CONCURRENT_REQUESTS`), placed before the parser flags so
   a per-model recovery override still wins last; 0 omits the flag, as
   the router's own guard does. **This invalidates existing probe cells**
   -- it changes the KV pool -- so re-probe one model and compare before
   any fleet sweep. Original report follows. Found while closing
   item 3. The router passes `--max-num-seqs` (vLLM) /
   `--max-running-requests` (SGLang) from `MAX_CONCURRENT_REQUESTS`;
   neither prober emits either flag, despite both arg builders'
   docstrings claiming they mirror the router's entrypoint. vLLM sizes
   its CUDA-graph capture set and its memory-profiling dummy forward off
   `max_num_seqs`, so the probe's KV pool does not match serve time.
   Direction unverified -- if the engine default exceeds 32 the probe
   over-reserves and is costing advertised context; if it is below 32 the
   probe under-reserves and could over-promise. Fixing it invalidates
   existing cells, so validate on one model before any fleet sweep.

4. **`PROBE_FORCE=1` does not override the exclusion ledger.** A forced
   re-probe still skips a ledger-excluded model. Defensible, but a
   July `unsupported_arch` verdict CAN go stale -- Qwen3.5-9B-NVFP4's
   did (see the stale-terminal fix below) -- so forcing should probably
   be able to retest one.

Fixed the same day, listed so the pattern is visible: the router let any
lab agent pull an unprobed model; KV fit math assumed fp16 on backends
serving fp8; bench tasks ran at backend-default sampling; the MCP
catalog used a schema that parsed to zero servers; model removal left no
ledger trace; `pull_hf` downloaded 26 GB of unusable format variants per
model; a multi-value `--exclude` silently disabled that filter entirely;
a stale terminal probe verdict survived a clean re-probe and would have
served a model with no parsers; and the bench runner benched models
whose weights were not on disk.

## Dependency graph

```
   +-------------------+     +------------------------+
   |  bench-rewrite    |     | model-lifecycle-ledger |
   |  (no deps)        |     | (no deps)              |
   +---------+---------+     +-----------+------------+
             |                           |
             |  v3 bench schema          |  exclusion ledger
             |  + host-env stamps        |  (_model_status.py)
             +------------+--------------+
                          |
                          v
          +-------------------------------+
          | card-derived-hints-and-       |  <-- both deps already
          | bench-sync                    |      satisfied; no
          | (Ph 1-4 hints, Ph 5 loop)     |      dependents
          +-------------------------------+

                    +-------------------+
                    | skypilot-agent-   |  <-- ships in parallel,
                    | skill (no deps)   |      no dependents
                    +-------------------+

                    +-------------------+
                    | sops-age-secrets  |  <-- prerequisite for
                    | (no deps)         |      three downstream
                    +---------+---------+      plans
                              |
            +-----------------+-----------------+
            |                 |                 |
            v                 v                 v
   +----------------+   +-------------+  +-----------------+
   | mcp-gateway    |   | cluster-mode|  | (fleet-provis-  |
   | Phase 2        |   | Phase 1     |  | ioner Phase 1,  |
   | (Tier 2)       |   | (worker +   |  | inherits via    |
   +----------------+   |  bootstrap) |  | cluster-mode    |
                        +------+------+  | path below)     |
                               |         +-----------------+
                               v
                        +------+------+
                        | cluster-mode|  <-- CI hard gate
                        | Phase 1.5   |      blocks Phase 2
                        | (preflight) |      and beyond
                        +------+------+
                               |
                               v
                        +------+------+
                        | cluster-mode|
                        | Phase 2     |
                        | (head +     |
                        |  routing)   |
                        +------+------+
                               |
                               v
                +--------------+---------------+
                | skypilot-fleet-provisioner   |
                | Phase 1 -> Phase 2 -> Phase 3|
                +------------------------------+

   +-------------------------------+
   | mcp-gateway Phase 1 (no deps; |
   | independent of sops-age)      |
   +-------------------------------+

   +-------------------------------+
   | cluster-mode Phase 3          |  <-- optional, no dependents
   | (probe-cache federation)      |
   +-------------------------------+
```

## Recommended execution order

The order below assumes a single operator picking up plans
sequentially. Side branches that can run in parallel are marked.
Parallelism cap is judgement-based -- a small team can spread
the side branches across people; a single operator should
probably do them in series rather than context-switching.

| Step | Plan / Phase                                  | Wall-clock | Blocking? | Why this position                                                     |
| ---- | --------------------------------------------- | ---------- | --------- | --------------------------------------------------------------------- |
| 1    | bench-rewrite (all 6 phases)                  | ~4 hours   | No        | Smallest plan, no deps, immediate picker-accuracy win. Clears the deck. |
| 2    | sops-age-secrets                              | 1-2 days   | Yes       | Hard prerequisite for three downstream plans. Should land before any consumer schedules. |
| 3    | mcp-gateway Phase 1                           | 1-2 days   | No        | Independent of sops-age; can run in parallel with step 2 or step 4. Validates Podman-socket compatibility early. |
| 4    | skypilot-agent-skill (Phases 1-2)             | 2-3 days   | No        | Independent of everything; user-facing win; one Dockerfile change. Parallel with steps 2 or 3. |
| 5    | gpu-arbiter-cluster-mode Phase 1              | 2 weeks    | Yes       | Depends on sops-age (step 2). Long pole of the cluster work. |
| 6    | gpu-arbiter-cluster-mode Phase 1.5            | ~1 day     | Yes       | CI hard gate. Blocks Phase 2 of cluster-mode AND all of fleet-provisioner. |
| 7    | mcp-gateway Phase 2                           | 2-3 days   | No        | Depends on sops-age (step 2) but not on cluster-mode. Can run in parallel with step 5 or after step 6. |
| 8    | gpu-arbiter-cluster-mode Phase 2              | 2 weeks    | Yes       | Head mode + routing. Required before any fleet-provisioner work. |
| 9    | skypilot-fleet-provisioner Phase 1            | 2-3 days   | Yes       | API server stand-alone. Inherits cluster-mode worker bootstrap + sops-age scaffold. |
| 10   | skypilot-fleet-provisioner Phase 2            | 1-2 weeks  | Yes       | Head <-> SkyPilot integration. Two-step graceful teardown per decision. |
| 11   | skypilot-fleet-provisioner Phase 3            | 1 week     | No        | Policy hardening. Can defer indefinitely. |
| 12   | gpu-arbiter-cluster-mode Phase 3 (optional)   | 1 week     | No        | Probe-cache federation. No dependents; activate only on user friction. |
| 13   | card-derived-hints-and-bench-sync             | ~1.5 weeks | No        | Both prerequisites already satisfied, so it can ship at any point. Appended rather than inserted because it is off the critical path entirely. Phases 1-4 (hints) and Phase 5 (bench loop) are independent tracks and can be split across people or releases. |

**Total elapsed time, serial single-operator path: roughly 6-9
weeks** (steps 1-10; step 11 and step 12 are optional). A small
team can shave a week or two by running steps 3, 4, 7 in
parallel with the cluster-mode long pole.

## Critical path

The longest must-be-sequential chain is:

```
sops-age-secrets
  -> cluster-mode Phase 1
  -> cluster-mode Phase 1.5
  -> cluster-mode Phase 2
  -> fleet-provisioner Phase 1
  -> fleet-provisioner Phase 2
```

Everything else can either be scheduled around this chain or
deferred. Compressing the critical path means compressing one
of these six phases -- there's no parallelism to exploit
within it.

## Parallelism map

Side branches that genuinely have no dependency on the critical
path:

- bench-rewrite (step 1) -- can ship at any point.
- skypilot-agent-skill (step 4) -- lab image change only;
  decoupled from cluster mode.
- mcp-gateway Phase 1 (step 3) -- independent of sops-age and
  cluster-mode.
- mcp-gateway Phase 2 (step 7) -- depends on sops-age only;
  parallel with cluster-mode Phase 1 onwards.
- cluster-mode Phase 3 (step 12) -- no dependents; activate on
  demand.
- router-shortcircuit (Draft) -- single-mode router feature, no deps
  and no dependents; ships at any point. Phase 1 (fingerprint logger)
  is the productized "empirical pass" and can run standalone wherever a
  live stack exists.
- router-fanout (Draft) -- NOT free-floating: Phase 1 (single-host
  demux) has no deps, but Phases 2-3 (concurrent demux + broadcast, the
  cluster-first payoff) depend on gpu-arbiter-cluster-mode Phase 2.
  Schedule Phase 1 any time; gate Phases 2-3 behind the cluster-mode
  head landing.
- pi-coding-agent (Draft) -- lab-image + picker change, no deps and no
  dependents; same shape as skypilot-agent-skill. Ships at any point.
- card-derived-hints-and-bench-sync (Draft) -- host-side probe/bench tooling
  only; touches no router, no picker, no container topology. Its two
  prerequisites (bench-rewrite's v3 schema, model-lifecycle-ledger's
  exclusion ledger) are already landed, so nothing gates it. Phases 1-4
  (card-derived hints) and Phase 5 (bench-sync loop) are separable tracks;
  Phase 3 is the only internal gate, held behind Phase 1's out-of-sample
  validation result.
- kv-cache-quantization (Draft) -- single-mode router + tooling feature,
  no hard deps and no dependents; ships at any point. Phase 1 (SGLang fp8
  parity + fit-math correctness) is behavior-preserving and standalone;
  Phase 3's bench-cache key change softly assumes bench-rewrite's v3
  schema is already landed.

## When this file becomes stale

Update this file when any of the following happens:

- A new plan lands in `docs/plans/` -- add a row to the status
  table, draw it into the dependency graph, insert it in the
  recommended-order table.
- A plan's `Dependencies` section changes -- reconcile the
  graph and the order table here.
- A plan ships -- bump its status (`Approved` -> `In Progress`
  -> `Done`) in the snapshot table.
- A decision invalidates an edge -- e.g., if cluster-mode Phase
  3 becomes a prerequisite for fleet-provisioner Phase 3 after
  a future plan amendment, the graph and the order both shift.

If this file disagrees with a plan's `Dependencies` section,
the plan wins. Fix this file.
