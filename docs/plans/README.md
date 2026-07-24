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
  unit-tested, but the live-GPU / live-secret verification the plan
  itself requires has not been run on this host (the GPU is never
  free and `make cache-down` is unavailable here). Code-complete,
  NOT verified-in-production.

Current snapshot (as of 2026-07-24). Every row was re-audited on that
date against each plan's own phase-completion markers -- the
2026-05-15 carry-over statuses no longer apply. The audit reclassified
four rows the prior snapshot carried as "In Progress"
(`sops-age-secrets`, `mcp-gateway`, `gpu-arbiter-cluster-mode`,
`model-lifecycle-ledger`) and one it carried as "Done (see note)"
(`review-fixes-2026-07`) into **Done (unverified)**: their code is
complete and unit-tested, but every live-GPU / live-secret
verification the plans require is still outstanding on this host.
The three plans still marked **In Progress** each have real remaining
work, not just verification: `bench-rewrite` (Phase 6 backfill is
required and unshipped), `skypilot-agent-skill` (Phase 2 plugin install
deferred), and `skypilot-fleet-provisioner` (Phase 2 Go code exists but
is NOT wired -- nothing constructs the client, so cloud burst is inert).
See each plan's own "Unverified" / caveats section for specifics.

Note on `review-fixes-2026-07`: executed 2026-07-23 in **three** passes
(remediation, then two adversarial-review repair rounds -- each round
found defects the previous round had itself introduced). All nine
phases were worked and its three open questions are resolved. It is
marked **Done (unverified)** rather than shipped-and-verified because the
live-GPU tests the plan itself requires before Phase 3 may merge
(`make test-router`, `make test-vllm`) could not run on the execution
host -- they need the GPU free and `make cache-down`. See that plan's
"Unverified" section.

| Plan                                                                | Status            |
| ------------------------------------------------------------------- | ----------------- |
| [sops-age-secrets](./sops-age-secrets.md)                           | Done (unverified) |
| [bench-rewrite](./bench-rewrite.md)                                 | In Progress       |
| [skypilot-agent-skill](./skypilot-agent-skill.md)                   | In Progress       |
| [mcp-gateway](./mcp-gateway.md)                                     | Done (unverified) |
| [gpu-arbiter-cluster-mode](./gpu-arbiter-cluster-mode.md)           | Done (unverified) |
| [skypilot-fleet-provisioner](./skypilot-fleet-provisioner.md)       | In Progress       |
| [router-shortcircuit](./router-shortcircuit.md)                     | Draft             |
| [router-fanout](./router-fanout.md)                                 | Draft             |
| [pi-coding-agent](./pi-coding-agent.md)                             | Draft             |
| [kv-cache-quantization](./kv-cache-quantization.md)                 | Draft             |
| [model-lifecycle-ledger](./model-lifecycle-ledger.md)               | Done (unverified) |
| [odysseus-borrowed-ideas](./odysseus-borrowed-ideas.md)             | Draft             |
| [review-fixes-2026-07](./review-fixes-2026-07.md)                   | Done (unverified) |
| [card-derived-hints-and-bench-sync](./card-derived-hints-and-bench-sync.md) | Draft     |

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
