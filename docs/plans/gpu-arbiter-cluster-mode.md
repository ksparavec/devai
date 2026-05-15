# gpu-arbiter Cluster Mode

_Add `--mode={single,worker,head}` to gpu-arbiter so the same Go binary
serves as either a single-host scheduler (today's behavior), a fleet
worker registering with a head, or a fleet head routing requests to
workers. No K8s, no daemonized control plane -- just a thin extension
of the existing router._

## Status

Design **approved 2026-05-14** -- all open questions resolved (see
"Confirmed decisions" below). Not yet scheduled for execution.

**Amended 2026-05-14**: (1) Pre-flight verification promoted from
informal decision 13 to a numbered **Phase 1.5** with explicit
exit criteria and a CI-runnable preflight script. (2) New
**decision 14** unifies the idle-threshold knob to a single
head-side `DEVAI_IDLE_MINUTES` env var; fleet-provisioner
inherits the same variable. (3) sops/age secret-store scaffold
(used for the worker bearer token) extracted into the new
[sops-age-secrets](./sops-age-secrets.md) plan; this plan is now
a hard dependency. (4) New `docs/cluster-env.md` deliverable in
Phase 1 captures the env-var contract that downstream consumer
plans extend.

## Dependencies

- [Plan: sops-age-secrets](./sops-age-secrets.md) -- Phase 1 reuses
  the shared sops/age scaffold for the head/worker bearer token
  (decision 8). Promoted from soft to hard dependency in the
  2026-05-14 amendment above (item 3); the scaffold lives in its
  own plan precisely so this plan can inherit it without
  duplicating fetch-cli / tmpfs / render infrastructure.

Worker bootstrap image was originally listed as a peer plan; per
decision 11 below it is folded into Phase 1 of this plan.

## Enables / Unblocks

- [Plan: skypilot-fleet-provisioner](./skypilot-fleet-provisioner.md) --
  the plan that adds cloud provisioning is meaningless without head mode
  to receive the worker registrations.
- On-prem multi-host clustering: a user with two boxes (e.g., a desktop
  with a 4090 and a server with 2x A100s) can run gpu-arbiter on each
  and have one of them act as the head, no cloud involved.
- Multi-GPU-per-host as a degenerate case: a single host with N GPUs can
  run N worker containers (each bound to one GPU via
  `CUDA_VISIBLE_DEVICES`) plus a head container on the same machine.
- Heterogeneous-fleet routing: probe cache already keys on GPU type;
  fleet routing falls out naturally once head mode can see all workers.

## Out of scope

- The single-host behavior (`--mode=single`) -- explicitly preserved
  byte-for-byte; this plan must not change today's flow.
- Cloud provisioning (the SkyPilot plan; this plan only handles the
  protocol between head and workers, not how workers get spawned).
- Multi-tenant authentication / authorization. Single-trust-zone today;
  bearer token at the network boundary is the only auth.
- Geographic / multi-region routing decisions. Future plan.
- Live probe cache federation (deferred to Phase 3 of this plan; the
  initial phases assume static cache shipped in worker images).
- Web UI for fleet state. Operators read it via the head's JSON status
  endpoint or logs.

## Confirmed decisions

Confirmed 2026-05-14 before implementation. Future deviations require
an explicit plan amendment.

1. **Multi-GPU per host: N worker containers, KISS.** Each container
   pins to one GPU via `CUDA_VISIBLE_DEVICES`. No native multi-GPU
   tracking inside the arbiter. Container overhead accepted in
   exchange for keeping the single-host code path untouched.

2. **Request mutation: worker owns the full chain; head does a
   minimal parse only.** The existing mutation chain
   (`parseReasoningOverride`, `parseCtxOverride`, `maybeStripTools`,
   `setNumCtx`, reasoning-policy injection, parser-plugin selection)
   stays where it lives -- in the worker. Head extracts just the
   `model` field from the JSON body plus any `<name>@<ctx>` and
   `::<reasoning>` suffix, enough to make a routing decision.
   Roughly 30 lines of new code on the head; zero behavioural change
   on the worker side. (Resolves Tension A from the design
   discussion.)

3. **Lifecycle class on registration, not a mode flag.** Workers
   self-declare their lifecycle in the registration message:
   - `lifecycle: ephemeral` -- head MAY issue a shutdown command
     after `DEVAI_IDLE_MINUTES` (single head-side env var; see
     decision 14) with no requests. Worker honours the command.
     SkyPilot (where applicable) tears down the VM after the
     worker exits. Default for SkyPilot-launched workers.
   - `lifecycle: persistent` -- head MUST NOT issue shutdown
     commands. Workers stay up indefinitely; loaded models stay
     loaded for fast re-use. Power consumption is the only
     steady-state cost. Default for systemd-launched on-prem
     workers.

   The idle *threshold* lives on the head; the lifecycle class
   on the worker decides only whether the worker honours a
   shutdown command. This split keeps policy in one place and
   keeps the worker's behaviour declarative.

   (Resolves Tension B. Replaces the "head-coordinated idle"
   vs "always-on" mode-flag distinction with a cleaner per-worker
   attribute.)

4. **Routing scoring: ctx-wider preference.** Head scores workers
   in this order for each request:
   1. Exact match (right model AND `loaded_ctx >= requested_ctx`),
      `queue_depth < threshold` -- zero cold-start cost.
   2. Right model but `loaded_ctx < requested_ctx` -- recreate cost;
      chosen only if no exact-match worker exists.
   3. Idle (no model loaded) -- cold-load cost.
   4. Different model loaded -- recreate cost.

   Tiebreak: round-robin among equally-scored candidates. Workers
   above `queue_depth_threshold` are skipped (overloaded). The
   `queue_depth_threshold` is configurable per backend type.

5. **Heartbeat content.** Each heartbeat carries `queue_depth`,
   `utilization_pct`, currently-loaded `(model, ctx)`,
   `last_request_at`, `health_status`, and a monotonic counter so
   head can detect out-of-order messages.

6. **Control plane protocol: HTTP REST with 10s polling.** Workers
   POST `/v1/cluster/heartbeat` every 10 seconds. Head's response
   body MAY include zero or more commands (lifecycle, drain,
   serve-this-request). No gRPC streaming, no NATS, no long-poll --
   trivial to debug with `curl`, no daemon to run, no reconnect
   choreography. Commands tolerate up to 10s propagation latency,
   which is acceptable for the operations we issue (shut down idle
   ephemeral worker, drain backend, re-register after head bounce).
   If 10s ever proves too slow, gRPC bidirectional streaming is a
   clean upgrade path that preserves the same message shapes;
   revisit then.

7. **Default mode: `single`.** Existing installs see zero change.
   Cluster behaviour is opt-in via `DEVAI_MODE={worker,head}` env
   var. Worker initiates the outbound HTTP connection to the head
   so a head behind NAT or in a private network works without
   special config.

8. **Auth: bearer token via sops/age.** Uses the shared scaffold
   from [sops-age-secrets](./sops-age-secrets.md) (also consumed by
   mcp-gateway Phase 2 and the SkyPilot fleet provisioner). Token
   stored in a sops-encrypted file, rendered to tmpfs at startup,
   mounted into both head and worker containers. Worker sends it as
   `Authorization: Bearer <token>` on every heartbeat and request.
   Head validates on every inbound endpoint.

9. **Fleet state: in-memory on head.** Re-derived from worker
   heartbeats on head restart. 10s heartbeat cadence gives fast
   recovery (workers re-register within 20s of head coming back).
   No named-volume persistence; no state migration concerns.

10. **Multi-host orchestration: SkyPilot above 1 host, systemd
    + make at 1 host.** This plan defines the protocol the worker
    speaks once running. SkyPilot owns the multi-host lifecycle
    (provision -> register -> serve -> teardown). On-prem
    single-host with N worker containers is brought up by existing
    `make cache-up` / `make cache-down` / `make cache-status` and
    supervised by systemd. No new orchestration entrypoint added
    here.

11. **Worker bootstrap: folded into Phase 1.** No separate peer
    plan. Phase 1 deliverables include the minimal worker bootstrap
    image, env-var contract (`DEVAI_MODE`, `DEVAI_HEAD_URL`,
    `DEVAI_WORKER_TOKEN_FILE`, `DEVAI_WORKER_NAME`,
    `DEVAI_LIFECYCLE`), and the cloud-init script shape that the
    SkyPilot plan will consume.

12. **Probe cache strategy: static for Phases 1-2.** Workers ship
    with the same checked-in probe cache files devai uses today
    (`deploy/.ollama-reasoning-cache.json` etc.). A worker whose
    GPU type has no rows refuses to advertise that backend. Phase 3
    adds optional live federation; defer until a real user hits
    the friction.

13. **Pre-flight verification: yes, before Phase 2.** Promoted
    to a dedicated **Phase 1.5** (see below) so it carries an
    exit criteria checklist rather than living as an informal
    note. Head + two workers all on the same host (different
    containers, distinct `CUDA_VISIBLE_DEVICES` if two GPUs are
    available; stubbed backends otherwise). Issue a request to
    the head; verify routing, forwarding, streaming, and
    lifecycle command flow. ~1 day wall-clock. Confirms Phase 1
    protocol choices before Phase 2 commits to them.

14. **Single idle-threshold env var: `DEVAI_IDLE_MINUTES`,
    read on the head only.** Defaults to 10 min. Replaces the
    earlier draft's split between cluster-mode
    `idle_minutes_threshold` and fleet-provisioner
    `SKYPILOT_IDLE_MINUTES` -- the head is the only actor that
    needs to know the threshold (it is the one issuing shutdown
    commands), so one variable is enough. Fleet-provisioner reads
    the same env var when deciding when to follow a clean worker
    exit with `sky down`. Per-deployment override via the env;
    no per-worker override (a fleet-wide policy is simpler than
    per-worker tuning and matches the operator-defined budget
    posture).

## Context

The single-host gpu-arbiter (`gpu-arbiter/main.go`, ~1070 lines) is the
existing centerpiece of devai's request path: multi-port reverse proxy,
GPU mutual exclusion, drain on switch, dynamic GPU memory allocation,
reasoning policy parsing, tool stripping, context-cap injection. All of
that is per-host concerns.

The natural extension -- repeatedly discussed in the architecture
conversations -- is a fleet-aware head that dispatches to per-host
workers, without resorting to Kubernetes. Each worker runs the existing
arbiter logic unchanged; the head runs a thinner version focused on
routing.

This plan is the prerequisite that unblocks every "more than one box"
story (SkyPilot cloud burst, on-prem multi-host, multi-GPU per host).
It is a structural change to gpu-arbiter but should be additive: the
single-host code path stays exactly as it is, gated by `--mode=single`
default.

The book has no direct equivalent: Chapter 5 (Kubernetes) is the
contrast point -- K8s offers a generic scheduler we are explicitly not
using. Chapter 8 (Multi-Agent) covers complexity-based routing which is
a different axis (model selection by prompt complexity). This plan
implements **GPU-aware request routing** across hosts, which is novel
to devai and the load-bearing AI-specific decision the architecture
discussions identified as the moat.

## Approach

Add a `--mode` flag to `gpu-arbiter/main.go`. Behaviour splits into
three code paths sharing the existing scheduling/recreate/forwarding
logic:

- `--mode=single` (default): today's behaviour. Listens on 11434/5/6,
  manages local backend containers, performs drain-on-switch. No
  cluster awareness.
- `--mode=worker`: same scheduling and backend-management logic as
  single, plus an outbound HTTP client that registers with a head on
  startup, polls the head every 10 seconds with a heartbeat (current
  loaded model/ctx, queue depth, utilization), and accepts inbound
  requests from the head only. The heartbeat response may include
  zero or more commands (drain, shutdown, serve-this-request) that
  the worker executes. Worker self-declares its lifecycle class
  (`ephemeral` or `persistent`) at registration time -- see
  decision 3.
- `--mode=head`: no GPU on this host, no backend containers, no
  `sleep infinity` placeholders. Listens on 11434/5/6 with the same
  OpenAI-compat surface. Maintains in-memory fleet state from worker
  heartbeats. For each incoming request, head does a minimal parse
  (model name + `@<ctx>` / `::<reasoning>` suffix), scores the fleet
  per decision 4, picks a worker, proxies the request body and
  streams the response back.

Authentication uses a bearer token mounted from the sops/age secret
store (shared scaffold from [sops-age-secrets](./sops-age-secrets.md),
also consumed by MCP gateway Phase 2 and the SkyPilot plan). Head
exposes `/v1/cluster/register`, `/v1/cluster/heartbeat`,
`/v1/cluster/status`. Worker exposes `/v1/cluster/inbound` (head-only
ingress for forwarded requests).

```
Client                                               Worker A (GPU type X)
  |                                                        ^
  |  POST /v1/chat/completions                             |
  v                                                        |
gpu-arbiter --mode=head                                    |
  |  minimal-parse (model + @ctx / ::reasoning)            |
  |  score fleet, pick worker                              |
  |  proxy request (HTTP + bearer)  ---------------------->+
  |  stream response back  <-------------------------------+

Worker A control plane (HTTP REST, 10s poll):
  POST head/v1/cluster/register
       { name, lifecycle, gpu_type, vram_gb, backends, version }
  every 10s:
    POST head/v1/cluster/heartbeat
         { worker_id, loaded_model, loaded_ctx, queue_depth,
           utilization_pct, last_request_at, counter }
  <- response body may contain commands:
         [ { type: "drain", backend: "vllm" },
           { type: "shutdown", grace_seconds: 30 } ]
```

Backward compatibility: existing single-host workflow (`make
cache-up`, direct `curl localhost:11434/v1/chat/completions`, model
picker, etc.) must continue to work bit-identically. Achieved by
defaulting `--mode=single` and keeping that path untouched.

---

## Phase 1 -- Worker mode + bootstrap + register/heartbeat protocol

### Goal

`gpu-arbiter --mode=worker --head=<url> --worker-token-file=<path>`
brings up the existing single-host scheduler, registers with the head
(stub head is fine), sends heartbeats every 10s, honours commands from
the head's heartbeat responses. Includes the minimal worker bootstrap
image (per decision 11) so SkyPilot can launch it later. No head-side
routing yet -- this phase validates the protocol from the worker side
and shakes out the new code paths in the arbiter.

### Deliverables

```
gpu-arbiter/
  main.go                            (modify: --mode flag, mode dispatch)
  cluster_worker.go                  (new: registration, heartbeat,
                                       command execution, inbound
                                       listener)
  cluster_proto.go                   (new: shared types for
                                       register / heartbeat / command
                                       / status messages)
  cluster_auth.go                    (new: bearer token loading,
                                       validation)
  parse_minimal.go                   (new: model + @ctx + ::reasoning
                                       extraction, head-side use only;
                                       worker imports unchanged)
deploy/
  docker-compose.yaml                (modify: add MODE/HEAD_URL/TOKEN/
                                       LIFECYCLE env vars; gate cluster
                                       behaviour behind compose profile
                                       "cluster")
  cluster-token.sops.env             (new: encrypted bearer token,
                                       consumes the sops-age-secrets
                                       scaffold)
  Dockerfile.worker-bootstrap        (new: minimal image for
                                       SkyPilot-launched cloud workers
                                       -- arbiter binary + backend
                                       images pre-pulled + cloud-init
                                       entrypoint)
  worker-cloud-init.sh               (new: cloud-init shape consumed by
                                       SkyPilot plan -- pulls credentials,
                                       reads DEVAI_HEAD_URL from env,
                                       starts arbiter in worker mode)
docs/
  router.md                          (extend: cluster mode protocol
                                       description)
  cluster-mode.md                    (new: operator guide for setting up
                                       worker / head, lifecycle classes,
                                       troubleshooting)
  worker-bootstrap.md                (new: env-var contract,
                                       cloud-init template, image
                                       layout -- the contract SkyPilot
                                       consumes)
  cluster-env.md                     (new: canonical env-var contract
                                       table -- name, default,
                                       allowed values, mode/phase
                                       that reads it, owning plan.
                                       Initial entries: DEVAI_MODE,
                                       DEVAI_HEAD_URL,
                                       DEVAI_WORKER_TOKEN_FILE,
                                       DEVAI_WORKER_NAME,
                                       DEVAI_LIFECYCLE,
                                       DEVAI_IDLE_MINUTES. Downstream
                                       consumer plans
                                       (skypilot-fleet-provisioner,
                                       sops-age-secrets, mcp-gateway)
                                       extend this table when they
                                       add their own env vars --
                                       PR review checklist enforces
                                       it.)
tests/
  test-cluster-worker.sh             (new: stand up worker against a
                                       stub head, verify registration,
                                       heartbeat traffic, command
                                       execution)
Makefile                             (new targets: cluster-worker-up,
                                       cluster-worker-down,
                                       build-worker-bootstrap)
```

### Detailed steps

1. **Add `--mode` flag and dispatch**: `flag.String("mode", "single",
   "single|worker|head")`. `single` calls existing `main`; the new
   modes route to dedicated entry points. Verify single mode is
   byte-identical to today (existing test suite must pass without
   modification).
2. **Worker registration**: on startup in worker mode, POST to
   `<head>/v1/cluster/register` with a JSON body:
   ```json
   {
     "name": "<DEVAI_WORKER_NAME>",
     "lifecycle": "<DEVAI_LIFECYCLE>",
     "gpu_type": "<probed_from_nvidia-smi>",
     "vram_gb": 24,
     "backends": ["ollama","vllm","sglang"],
     "arbiter_version": "<git_sha>",
     "endpoint": "http://<worker_ip>:<port>"
   }
   ```
   `DEVAI_LIFECYCLE` defaults to `persistent` (per decision 3);
   SkyPilot-launched cloud-init sets it to `ephemeral`. Retry with
   exponential backoff if head is unreachable. Block startup until
   registration succeeds; surface clear logs.
3. **Heartbeat loop**: every 10s, POST to `/v1/cluster/heartbeat` with
   current state per decision 5:
   ```json
   {
     "worker_id": "<assigned_by_head_at_registration>",
     "loaded_model": "<name_or_null>",
     "loaded_ctx": 131072,
     "queue_depth": 0,
     "utilization_pct": 12.0,
     "last_request_at": "2026-05-14T08:53:00Z",
     "health_status": "ready",
     "counter": 421
   }
   ```
   Read commands from the response body and execute them. Initial
   commands supported: `drain { backend }`, `shutdown { grace_seconds }`,
   `serve { request_id, target_model, target_ctx, body_url }`. Unknown
   command types are logged and ignored.
4. **Inbound listener**: worker exposes a separate
   `/v1/cluster/inbound` endpoint that accepts forwarded
   OpenAI-compat requests from the head. Token-gated. The actual
   request handling reuses the existing single-host code path -- the
   inbound endpoint is a thin auth wrapper, and the existing
   request-mutation chain runs unchanged on the worker (per
   decision 2). Worker's public 11434/5/6 ports continue to serve
   direct testing requests (useful for debugging without going
   through the head).
5. **Command execution**: workers honour lifecycle commands. If
   `lifecycle=persistent`, `shutdown` commands are logged and
   refused (per decision 3); `drain` commands are still honoured.
   If `lifecycle=ephemeral`, both are honoured. This is the only
   place the lifecycle class affects behaviour.
6. **Token loading**: read bearer token from
   `/run/devai/cluster-token` (tmpfs rendered from
   `cluster-token.sops.env`). Uses the shared sops/age scaffold
   from [sops-age-secrets](./sops-age-secrets.md).
7. **Worker bootstrap image** (`Dockerfile.worker-bootstrap`):
   minimal image containing the arbiter binary, the three backend
   images pre-pulled to local podman storage, and the cloud-init
   entrypoint. Designed for SkyPilot to provision with -- no
   JupyterLab, no model picker, no Open WebUI, nothing user-facing.
   Image size target: under 5 GiB.
8. **Cloud-init contract** (`worker-cloud-init.sh`): documented
   env-var contract -- `DEVAI_MODE=worker`, `DEVAI_HEAD_URL=<url>`,
   `DEVAI_WORKER_TOKEN_FILE=<path-to-tmpfs-rendered-token>`,
   `DEVAI_WORKER_NAME=<unique>`, `DEVAI_LIFECYCLE=ephemeral`. This
   is the contract the SkyPilot plan consumes; defining it here
   means no separate planning round-trip.
9. **Stub head for testing**: a minimal Python or shell script that
   accepts register / heartbeat / inbound, logs them, returns canned
   command sequences for tests. Used in
   `tests/test-cluster-worker.sh`.

### Exit criteria

- `--mode=single` (default) behaviour verified byte-identical to today
  by running the existing test suite (router tests, model matrix
  tests).
- Worker starts, registers with stub head (with correct `lifecycle`
  field), sends heartbeats at 10s cadence with sane content.
- Worker handles head being unreachable on startup (retries with
  backoff) and head disappearing mid-run (logs warning, keeps
  retrying, continues serving direct-port requests if any).
- Worker's `/v1/cluster/inbound` accepts a request with the right
  token, rejects without one (401), and the request flows through
  the existing single-host code path unchanged.
- Worker executes `drain` and `serve` commands from heartbeat
  responses. `shutdown` is honoured when `lifecycle=ephemeral`,
  refused with a log line when `lifecycle=persistent`.
- `Dockerfile.worker-bootstrap` builds; resulting image boots in
  `--mode=worker` against a stub head from a clean VM.

### Phase 1 risks

| Risk                                                    | Mitigation                                                        |
| ------------------------------------------------------- | ----------------------------------------------------------------- |
| Refactor breaks single-host code path                   | Single mode dispatched to existing `main` unchanged; full test suite gates merge |
| 10s heartbeat causes user-visible command latency       | Documented; acceptable for our command set (shutdown, drain). gRPC upgrade path noted in decision 6 |
| Worker's GPU type detection unreliable across drivers   | Use existing nvidia-smi probe; document failure modes             |
| Probe cache shipped in worker image becomes stale       | Phase 3 federation; for now operator runs `make probe` and rebuilds |
| Bearer token leakage in logs                            | Standard mask-secrets logger pattern; review before merge         |
| Bootstrap image bloat                                   | Pre-pull only the backend images for the GPU class this worker serves; documented in worker-bootstrap.md |

---

## Phase 1.5 -- Pre-flight verification

### Goal

Validate the Phase 1 protocol choices against a real two-worker
deployment **before** Phase 2 commits any head-side code to them.
Promoted from decision 13's informal "yes, do this" to a numbered
phase so it carries an exit-criteria checklist a reviewer can
pass/fail.

Phase 1.5 ships **one production artifact** -- a CI-runnable
test script -- plus a test-report doc. The script gates every
subsequent cluster-mode PR after Phase 1.5 closes (Phase 2 and
Phase 3 included), so a regression in the Phase 1 protocol is
caught at PR time rather than at a real fleet deploy.

### Deliverables

```
tests/
  test-cluster-preflight.sh       (new: scripted scenarios 1-7
                                    below; CI-runnable; gates
                                    every cluster-mode PR after
                                    Phase 1.5 closes; depends only
                                    on the stub head from Phase 1
                                    and two worker containers --
                                    no GPU required, stubbed
                                    backends are fine)
  fixtures/stub-head.py           (new or extended: minimal stub
                                    head that scripts the canned
                                    command sequences scenarios
                                    3-7 need; reuses the Phase 1
                                    stub-head from
                                    tests/test-cluster-worker.sh
                                    if shape allows)
docs/
  cluster-mode-preflight.md       (new: one-time test report --
                                    date run, host details, per-
                                    scenario observed behaviour,
                                    any deviations; rerun whenever
                                    a new host class joins the
                                    supported matrix)
.github/                          (modify: existing CI workflow
                                    invokes
                                    tests/test-cluster-preflight.sh
                                    on every PR that touches
                                    gpu-arbiter/ or
                                    tests/test-cluster-*)
```

### Detailed steps

Use a single host with the stub head from Phase 1 and two
worker containers (distinct `CUDA_VISIBLE_DEVICES` if two GPUs
are available; stubbed backends otherwise). Per decision 13.

1. **Registration**: both workers come up, each registers with
   the stub head, each prints its assigned `worker_id`. Verify
   stub head's logged state contains both workers with the
   correct `lifecycle` field.

2. **Heartbeat cadence**: observe heartbeats every 10s for at
   least 60s. Verify `counter` increments monotonically per
   worker; verify stub head never sees out-of-order counters
   (or, if it does on a flaky network, that it logs the gap
   and ignores the stale heartbeat per decision 5).

3. **Command flow -- drain**: stub head returns a `drain {
   backend: vllm }` command in one worker's heartbeat
   response. Verify that worker drains vllm on its existing
   per-host code path; verify it does not affect the other
   worker.

4. **Command flow -- serve**: stub head returns a `serve { ...
   }` command. Verify the worker accepts a request on
   `/v1/cluster/inbound`, runs it through the unchanged
   single-host code path, streams the response back. Verify
   token gating: a request without the bearer token returns
   401.

5. **Command flow -- shutdown / lifecycle**: send a `shutdown`
   command to the `ephemeral` worker; verify it drains, exits
   the arbiter process. Send the same command to the
   `persistent` worker; verify it logs the refusal and stays
   up (per decision 3).

6. **Failure recovery**: kill the stub head mid-run. Verify
   workers log "head unreachable" and retry; verify they keep
   accepting direct-port requests on 11434/5/6. Restart the
   stub head; verify workers re-register without operator
   intervention within 20s.

7. **Token rotation**: rotate the bearer token via the
   sops-age-secrets scaffold (decryption + re-render to
   tmpfs). Verify workers pick up the new token on their next
   heartbeat without restart, or document the restart
   requirement if pickup isn't transparent.

### Exit criteria

- All seven scenarios above pass without manual intervention
  beyond starting the containers.
- Any Phase 1 bugs discovered during the exercise are fixed
  and re-verified before Phase 1.5 closes.
- `tests/test-cluster-preflight.sh` runs the scripted scenarios
  end-to-end and exits 0 on a clean two-worker host **AND in
  CI** -- no real GPU dependency; stubbed backends acceptable.
- The CI workflow invokes `tests/test-cluster-preflight.sh` on
  every PR touching `gpu-arbiter/` or
  `tests/test-cluster-*.sh`. Failure blocks merge. (This is the
  CI-runnable hard-gate exit criterion.)
- `docs/cluster-mode-preflight.md` records: date first run, host
  details (kernel, GPU type, driver), per-scenario observed
  behaviour, any deviations from expected.

### Phase 1.5 risks

| Risk                                                  | Mitigation                                            |
| ----------------------------------------------------- | ----------------------------------------------------- |
| Pre-flight passes on stubbed backends but real vLLM/SGLang behaves differently | Run scenarios 4-5 once more with real backends loaded if a GPU host is available; document gap if not |
| Token rotation (scenario 7) requires worker restart   | Acceptable; document as expected behaviour in cluster-mode.md if so |
| Phase 1.5 turns into a multi-week bug hunt            | Time-box to ~1 day wall-clock per decision 13; if scenarios consistently fail, the right move is to revisit Phase 1 design, not extend the verification window |

---

## Phase 2 -- Head mode + fleet routing

### Goal

`gpu-arbiter --mode=head` listens on the same OpenAI-compat ports
(11434/5/6), accepts client requests, picks a worker, forwards. End-to-end
flow works: client -> head -> worker -> backend -> stream back through
head -> client.

### Deliverables

```
gpu-arbiter/
  cluster_head.go                    (new: registration receiver,
                                       fleet state, routing decision)
  cluster_proxy.go                   (new: request/response streaming
                                       proxy through head)
  fleet_state.go                     (new: in-memory worker map,
                                       loaded-model tracking, GC of
                                       expired heartbeats)
  routing_policy.go                  (new: probe-cache lookup + worker
                                       scoring + selection)
deploy/
  docker-compose.yaml                (modify: head profile, no backends
                                       on head)
  compose.head.yaml                  (new: override that disables
                                       ollama/vllm/sglang on head host)
docs/
  cluster-mode.md                    (extend: head setup, routing
                                       behavior, troubleshooting)
  router.md                          (extend: head-mode behavior,
                                       streaming proxy notes)
tests/
  test-cluster-head.sh               (new: head + two workers on same
                                       host, route requests, verify
                                       streaming, verify worker
                                       failure handling)
Makefile                             (new targets: cluster-head-up,
                                       cluster-head-down, cluster-status)
```

### Detailed steps

1. **Head listener**: `--mode=head` opens 11434/5/6 with the same
   OpenAI-compat handler shape, but the handler dispatches to
   `cluster_proxy.Forward` instead of the local recreate/serve path.
2. **Registration receiver**: handle POST `/v1/cluster/register`;
   validate token; insert worker into fleet state. Handle
   `/v1/cluster/heartbeat` updates. Expire workers whose heartbeat
   stops arriving (default 30s timeout, configurable).
3. **Fleet state in memory**: map of `worker_id -> WorkerState{name,
   gpu_type, endpoint, currently_loaded_model, currently_loaded_ctx,
   last_heartbeat, ...}`. Mutex-protected. Garbage-collected on
   expiry.
4. **Routing decision** (`routing_policy.go`): given an incoming
   request, head runs `parse_minimal` to extract `(model, ctx,
   reasoning)`. Then:
   - Probe cache lookup: which GPU types fit this `(model, ctx,
     backend)`?
   - Candidate workers: those whose `gpu_type` is in the fit set
     AND `queue_depth < queue_depth_threshold` (overloaded
     workers skipped per decision 4).
   - Score each candidate per decision 4:
     1. Exact match (`loaded_model == model AND loaded_ctx >= ctx`)
        -> score 100. Zero cold-start.
     2. Right model but `loaded_ctx < ctx` -> score 50. Recreate
        cost.
     3. Idle (`loaded_model == null`) -> score 30. Cold-load cost.
     4. Different model loaded -> score 10. Recreate cost.
   - Pick the top scorer. Tiebreak: round-robin among equally-
     scored candidates (state tracked per-bucket in head memory).
   - If no worker fits: 503 with a JSON body explaining which GPU
     types are needed but absent from the fleet (and, if a
     SkyPilot-shaped head is configured, a hint that provisioning
     could be tried -- that decision is delegated to the SkyPilot
     plan).
5. **Request forwarding** (`cluster_proxy.go`): proxy the full
   request body to the chosen worker's `/v1/cluster/inbound`,
   including the `(model, ctx, backend, reasoning)` overrides in the
   URL/body (so the existing override-parsing on the worker still
   works). Stream the response back to the client. For
   `Content-Type: text/event-stream` (SSE), preserve framing.
   Bandwidth note: head sees full request/response bytes -- documented
   tradeoff per Open Question 4.
6. **Status endpoint**: `GET /v1/cluster/status` returns JSON with
   fleet state for ops/diagnostics. No auth required for read-only
   in v1; revisit if hostile networks become a concern.
7. **No-backend mode for head**: when `--mode=head`, gpu-arbiter
   should refuse to spawn any vLLM/SGLang/Ollama containers locally
   (it has no GPU contract to enforce). The compose override file
   ensures the head host's compose stack doesn't include the
   backends.
8. **Self-registration shortcut for development**: on a single host,
   one can run head + one worker in the same compose stack for
   testing. Worker registers with `head://devai-router-head:11434`
   over the compose network.

### Exit criteria

- Head + two workers (on same host, distinct compose stacks or
  distinct `CUDA_VISIBLE_DEVICES`) come up. Both workers register
  visibly in `/v1/cluster/status`.
- Client request to head returns same response as if direct against
  a worker. Streaming SSE preserved end-to-end.
- Worker disappearing (kill container) is detected within 30s,
  removed from fleet, subsequent requests routed to remaining worker.
- Re-routing on heartbeat loss does not double-serve in-flight
  requests (document the failure semantics: a request in flight when
  worker dies returns 5xx; client retries).
- No-fit case (request requires GPU type absent from fleet) returns
  503 with helpful JSON explaining what's missing.

### Phase 2 risks

| Risk                                                       | Mitigation                                                        |
| ---------------------------------------------------------- | ----------------------------------------------------------------- |
| Streaming proxy adds latency or breaks SSE framing         | Integration test asserts byte-for-byte stream parity              |
| Head becomes single point of failure                       | Documented as expected limitation v1; HA head is future work     |
| Race: two requests arrive for same model not loaded anywhere | Coalesce: second waits for first's recreate to finish; documented |
| Worker network blip causes routing to a stale GPU state    | Heartbeats include monotonic counter; head ignores out-of-order  |
| Bandwidth on head saturates from streaming large responses | Document expected throughput; recommend deploying head on a wired link |
| Probe cache disagrees between head and workers             | Head's copy is authoritative for routing; document drift detection in Phase 3 |

---

## Phase 3 -- Probe cache federation (optional)

### Goal

Workers probe their GPU on first startup if its type is absent from the
shipped cache; results federate through head to all other workers of that
type. Eliminates the "operator runs `make probe` then rebuilds images"
loop for novel GPU types.

This phase is **optional** -- Phases 1 and 2 work fine with static probe
cache. Defer until a real user reports the friction.

### Deliverables (sketch only; flesh out when phase is activated)

- Worker runs `make probe` on first startup if its GPU type has no
  cache rows; ships rows to head.
- Head merges new rows into master cache; pushes consolidated cache to
  all workers on heartbeat.
- Schema version handling so old workers don't choke on new fields.
- Drift detection: if a worker's cache disagrees with head's, head wins
  and worker re-syncs.

### Phase 3 risks

| Risk                                                     | Mitigation                                                  |
| -------------------------------------------------------- | ----------------------------------------------------------- |
| Probe takes minutes; blocks worker availability          | Probe in background, mark worker not-ready until done       |
| Concurrent probes on multiple workers of same type       | Lock via head; first worker probes, others wait             |
| Live cache mutation introduces consistency bugs          | Cache is monotonic-add-only; rotation only via operator action |

---

## Combined risk register

| Risk                                                       | Phase | Mitigation                                                        |
| ---------------------------------------------------------- | ----- | ----------------------------------------------------------------- |
| Refactor breaks single-host code path                      | 1     | Mode dispatch keeps single path untouched; full regression suite  |
| Bearer token leakage in logs                               | 1     | Mask-secrets logger; review                                       |
| Streaming proxy breaks SSE framing                         | 2     | Byte-for-byte stream integration test                             |
| Head as SPOF                                               | 2     | Documented limitation; HA head is future                          |
| Race on model-not-loaded across multiple concurrent requests | 2   | Coalesce: second waits for first; documented                      |
| Bandwidth saturation on head                               | 2     | Operator guidance: deploy head on wired link                      |
| Probe cache drift between workers and head                 | 2-3   | Head authoritative; drift detection in Phase 3                   |
| Worker churn causes thrash in fleet state                  | 2     | Heartbeat expiry tunable; documented stability guidance           |

## Migration / rollback story

- **Phase 1 rollback**: ship without setting `DEVAI_MODE=worker`;
  default `single` is byte-identical to today. Or remove the env var
  and restart; arbiter returns to single mode on next boot.
- **Phase 2 rollback**: stop the head container; existing workers
  continue accepting direct-port requests for testing. Or restart
  workers in single mode (env change).
- **Phase 3 rollback**: feature flag the federation behavior; disabled
  by default falls back to static-cache behavior of Phases 1-2.
- **Upgrade path for existing single-host installs**: zero change.
  `make cache-up` continues to work; cluster mode is opt-in via compose
  profile and env var.

## Estimated effort

| Phase    | Engineering effort                                                                  | Wall-clock              |
| -------- | ----------------------------------------------------------------------------------- | ----------------------- |
| Phase 1  | ~2 PRs, ~500-800 lines of Go + bootstrap image + cloud-init + tests + docs          | 2 weeks                 |
| Phase 1.5| ~1 PR, ~150 lines of bash + test-report doc; no production code                     | ~1 day                  |
| Phase 2  | ~2 PRs, ~600-1000 lines of Go + integration tests + docs                            | 2 weeks                 |
| Phase 3  | ~1 PR, ~300-500 lines + cache schema versioning                                     | 1 week (when activated) |
| Total    | 5-6 PRs (Phases 1, 1.5, 2 mandatory; 3 optional)                                    | 4-5 weeks elapsed       |

Phase 1 grew by ~2-3 days vs the original draft because the worker
bootstrap image and cloud-init contract were folded in (decision 11).

## References

- Existing arbiter source: `gpu-arbiter/main.go` (~1070 lines Go)
- Existing per-host scheduling docs: `docs/router.md`,
  `docs/backends.md`, `docs/nvfp4-coldstart.md`
- Probe cache schemas: `deploy/.ollama-reasoning-cache.json` (v3,
  digest-keyed), `deploy/.vllm-reasoning-cache.json` and
  `deploy/.sglang-reasoning-cache.json` (v2, repo+sha-keyed)
- Book contrast points:
  - Ch5 (Kubernetes for ML) -- the generic-scheduler approach this
    plan explicitly avoids
  - Ch8 (Multi-Agent / complexity routing) -- different axis (model
    selection by prompt) than this plan's GPU-aware request routing
  - Ch7 (Agent Controller -- registration / heartbeat / task queue
    pattern) -- structural inspiration for the head's fleet state
- Related plans:
  - Unblocks:
    [skypilot-fleet-provisioner](./skypilot-fleet-provisioner.md)
  - Reuses secret pattern:
    [mcp-gateway](./mcp-gateway.md) Phase 2
  - Worker bootstrap image (originally a peer plan) is folded into
    Phase 1 of this plan per decision 11.
