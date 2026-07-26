# cluster-mode + fleet-provisioner -- restore notes

Read this before thawing anything in this subtree. The code here was
frozen in a **known-broken** state. It compiles and its unit tests
pass; it does not work.

## Open defects at freeze time (2026-07-25)

These were all verified against the tree, not inferred from the plans.
None were fixed before freezing.

### Blocking -- the head cannot start

1. **Head crash-loops on a token file nothing mounts.**
   `cluster_head.go` `NewClusterHead()` calls `log.Fatalf` when
   `DEVAI_HEAD_TOKEN_FILE` (default `/run/devai/cluster-token`) is
   unreadable. The compose `router` service mounts no `/run/devai`, and
   `deploy/compose.head.yaml` set only `environment:` -- no `volumes:`,
   no `ports:`. Under `restart: unless-stopped` this crash-loops.
2. **`deploy/cluster-token.sops.env` never existed.**
   `docs/cluster-mode.md` referenced it and a Makefile target told the
   operator to render it. Neither the file nor a `.example` template was
   ever committed.
3. **The control plane was unreachable.** The `router` service has no
   `ports:` block, so `:11444` and the `11434/5/6` frontends were bound
   on `devai-net` only. No off-host worker could have registered.
4. **`.sops.yaml` still carries the literal `age1xxxx...` placeholder**,
   so the sops scaffold that was supposed to deliver the bearer token
   cannot encrypt anything. The unit test that should have caught this
   passes on the placeholder -- its regex `age1[0-9a-zA-Z]{30,}` matches
   it.

### Correctness -- routing does not do what the plan claims

5. **Routing ignores GPU fit.** `routing_policy.go` filters only on
   health, advertised backend and queue depth, and scores only on
   loaded model / loaded ctx. `GPUType` and `VRAMGB` are carried in
   `fleet_state.go` and echoed by `handleStatus` but never read by any
   routing decision. The head would happily route a request to a worker
   that cannot fit the model -- which was supposed to be the whole point
   of routing over a probe-cache-aware fleet.
6. **Decision 12 unimplemented.** "A worker whose GPU type has no probe
   rows refuses to advertise that backend" -- `cluster_main.go`
   advertises whatever `DEVAI_BACKENDS` says, unconditionally.

### Cloud half -- entirely unwired

7. **Nothing constructs the SkyPilot client.** `NewSkyPilotClient`,
   `NewSkyPilotPolicy` and `NewIdleTeardownCoordinator` had zero
   non-test callers. `SKYPILOT_API_ENDPOINT` was read by no code at all.
   Thawing the fleet provisioner means *writing the integration*, not
   restoring it -- the client and policy are building blocks that were
   never assembled.
8. **The Phase 1 exit criterion was never met.** It required a real
   cloud VM launch, hello-world, and teardown with elapsed time and
   cost recorded in docs. `docs/skypilot.md` recorded no launch, no
   time, no cost.

### Security -- do not restore verbatim

9. **The plan text still prescribes `${HOME}:/root:rw`.** The
   fleet-provisioner plan's decision 1 and its Phase 1 compose YAML
   mount the operator's entire home directory into the SkyPilot
   container. That was later replaced with five narrowly-scoped
   read-only mounts precisely because it handed the container `~/.ssh`
   **and** `~/.config/sops/age/keys.txt` -- the private key behind every
   `deploy/*.sops.env`. If you thaw from the plan text rather than from
   the frozen compose file, you will reintroduce that.

### Testing -- the green suite proves nothing

10. **`tests/test-fleet-routing.sh` asserts nothing.** Its body is a
    comment block describing a hypothetical test plus one curl for
    `/api/v1/version`; it exits 77 when `SKYPILOT_API_ENDPOINT` is
    unset, which is the normal case.
11. **No live two-host test exists at any level.** The Phase 1.5
    preflight drives a 162-line Python stub head
    (`tests/fixtures/stub-head.py`). No Go test ever constructs
    `NewClusterHead`; `cluster_head_test.go` exercises `controlPlaneMux`
    directly with a fake forwarder.
12. **The Phase 1.5 CI hard gate was never wired.** The plan made
    preflight failure block merges. `.github/workflows/` contains only
    `security-advisory.yml` and `security-blocking.yml`, and the `make
    test` aggregate omitted the preflight target.

## Restore checklist

1. Fix defects 1-4 before attempting to start a head at all.
2. Decide whether defect 5 matters to you. If the fleet is homogeneous
   it may not; if it is not homogeneous, routing must read the probe
   caches, which is the unbuilt part of cluster-mode Phase 3
   (probe-cache federation).
3. Treat the SkyPilot half (defects 7-8) as unstarted work, not as
   restorable work.
4. Do not copy compose YAML out of the plan documents. Use the frozen
   `deploy/` files in this subtree, which carry the corrected scoped
   mounts.
5. Replace defect 10-11 tests before believing any green result.

## What was moved here

```
cluster-mode/
  gpu-arbiter/   22 .go files (11 source, 11 test), each tagged
                 //go:build devai_frozen_cluster
  deploy/        compose.head.yaml, Dockerfile.worker-bootstrap,
                 worker-cloud-init.sh, skypilot compose service block
  tests/         test-cluster-preflight.sh, test-fleet-routing.sh,
                 fixtures/stub-head.py
  docs/          cluster-mode.md, cluster-env.md,
                 cluster-mode-preflight.md, worker-bootstrap.md,
                 skypilot.md
  Makefile.frozen-targets  the 9 retired make targets, verbatim
```
