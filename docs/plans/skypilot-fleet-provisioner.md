# SkyPilot Fleet Provisioner

_Run SkyPilot API server as a devai compose service so gpu-arbiter (in head
mode) can provision cluster workers on demand across cloud, Slurm, and
on-prem._

## Status

Design **approved 2026-05-14** -- all seven open questions resolved
(see "Confirmed decisions" below). Not yet scheduled for execution
pending the gpu-arbiter cluster-mode and worker-bootstrap plans
listed under Dependencies.

**Amended 2026-05-14**: (1) Worker bootstrap dependency now points
at gpu-arbiter-cluster-mode Phase 1 (where the bootstrap image and
cloud-init contract live per that plan's decision 11); the
standalone worker-bootstrap-image plan no longer exists.
(2) sops/age promoted from soft to hard dep on the new
[sops-age-secrets](./sops-age-secrets.md) plan. (3) Decision 6
collapsed `SKYPILOT_IDLE_MINUTES` into the unified head-side
`DEVAI_IDLE_MINUTES` env var (cluster-mode decision 14). (4) Phase
2 step 4 (idle teardown) rewritten to do a two-step graceful
teardown: head sends `shutdown` command, worker drains and exits,
then head calls `sky down` to release the VM.

## Dependencies

- [Plan: gpu-arbiter-cluster-mode](./gpu-arbiter-cluster-mode.md)
  Phase 2 -- adds `--mode={single,worker,head}` to gpu-arbiter so
  it can act as a fleet router instead of a single-host scheduler.
  This plan cannot function without head mode existing.
- [Plan: gpu-arbiter-cluster-mode](./gpu-arbiter-cluster-mode.md)
  Phase 1 -- ships the worker bootstrap image and the cloud-init
  contract (`DEVAI_MODE`, `DEVAI_HEAD_URL`, `DEVAI_WORKER_TOKEN_FILE`,
  `DEVAI_WORKER_NAME`, `DEVAI_LIFECYCLE`) per decision 11 of that
  plan. SkyPilot-launched VMs consume the bootstrap image directly;
  without it, a provisioned VM has nothing to do.
- [Plan: sops-age-secrets](./sops-age-secrets.md) -- shared sops/age
  scaffold used for non-interactive cloud credentials (RunPod /
  Lambda API keys rendered to `/run/devai/skypilot-credentials.env`)
  and for the bearer token between gpu-arbiter head and the
  SkyPilot API server. This is now a hard dep, not a soft one --
  the scaffold lives in its own plan precisely so this plan can
  inherit it without duplicating fetch-cli / tmpfs / render
  infrastructure.

## Enables / Unblocks

- Cloud-burst capacity: requests that do not fit on local GPUs trigger
  on-demand provisioning of a cloud worker, served, then torn down.
- Multi-host on-prem clustering: same gpu-arbiter, multiple bare-metal hosts
  registered as workers without K8s.
- Cost-aware scheduling: probe cache fit data feeds into "cheapest GPU type
  that fits this model and context."
- Heterogeneous-fleet routing: a request can be served on a local 3090, a
  rented A100, or a borrowed H100 depending on availability and policy.

## Out of scope

- The gpu-arbiter `--mode=head` implementation itself (prerequisite plan).
- The worker bootstrap image and its registration protocol (prerequisite
  plan).
- Spot-instance preemption handling and recovery (future plan).
- Multi-region routing or latency-aware placement (future plan).
- Kubernetes / kagent integration (explicit non-goal -- devai stays K8s-free).
- User-facing agent-driven SkyPilot invocation (that is the sibling plan
  [skypilot-agent-skill](./skypilot-agent-skill.md), not this one).
- Pricing / budget UI in JupyterLab (future plan).

## Confirmed decisions

Confirmed 2026-05-14 before implementation. Future deviations
require an explicit plan amendment.

1. **API server image: use `berkeleyskypilot/skypilot:<pin>`,
   no fallback.** Verified 2026-05-14 -- upstream publishes the
   official image on Docker Hub at `berkeleyskypilot/skypilot`
   (stable) and `berkeleyskypilot/skypilot-nightly` (dev builds).
   Most recent stable as of writing: `0.12.1` (2026-04-24);
   `0.12.2rc1` available. Smoke-tested on the project's host:
   `podman pull berkeleyskypilot/skypilot:0.12.1` + the
   documented entrypoint (`tini -- sky api start --deploy
   --foreground`) brings the API server up on container port
   **46580** within ~7s. Image is ~2.9 GB uncompressed (660 MiB
   compressed).

   Consequence: **no `Dockerfile.skypilot-api` is needed** for
   this plan; no `fetch-cli` extension is required for the API
   server path. The fetch-cli mechanism established by
   `skypilot-agent-skill.md` decision 1 remains the right pattern
   for the **lab image** -- a separate concern that this plan
   does not touch.

   Container port is **46580** (upstream default); host port is
   tunable via `SKYPILOT_API_PORT` env var. Credentials surface
   via `-v $HOME:/root` per upstream's docker-run example -- this
   matches `skypilot-agent-skill.md` decision 3 (rely on `$HOME`
   mount, no new credential surface).
2. **Cloud extras: match the agent-skill broad set** --
   `skypilot[aws,gcp,azure,kubernetes,slurm,runpod,lambda]`.
   Uniform image footprint across the lab and the API server.
   System dispatches to any cloud the user has credentials for
   without rebuilding. (Deviates from the original recommendation
   of `[runpod,lambda,kubernetes,slurm]` -- chosen for image
   uniformity with `skypilot-agent-skill.md`'s confirmed decision 2.)
3. **API auth: token-based via sops/age.** Bearer token stored
   using the shared scaffold from
   [sops-age-secrets](./sops-age-secrets.md). Survives reboots;
   rotatable; no plaintext at rest. The scaffold is a hard
   dependency of this plan (see Dependencies section); mcp-gateway
   Phase 2 and gpu-arbiter cluster mode consume the same scaffold.
4. **State DB: named volume `skypilot-state`.** Preserves cluster
   registry across container restarts. Matches the project's
   pattern for ollama/vllm-state volumes.
5. **Cost-cap enforcement: head-side, in gpu-arbiter.** gpu-arbiter
   has full context (which model, why the request was made, current
   fleet utilization) and can short-circuit before any SkyPilot or
   cloud round-trip. Budget config lives alongside other gpu-arbiter
   policy (e.g., the recovery-flags.json pattern).
6. **Idle teardown threshold: 10 minutes default, env-tunable via
   `DEVAI_IDLE_MINUTES`** (head-side, single source of truth --
   see [gpu-arbiter-cluster-mode](./gpu-arbiter-cluster-mode.md)
   decision 14). This plan does NOT introduce a second
   `SKYPILOT_IDLE_MINUTES` knob; head reads `DEVAI_IDLE_MINUTES`
   to decide when to send the worker `shutdown` command, and
   this plan's Phase 2 step 4 follows that clean exit with
   `sky down`. Midpoint of the 5-10 min range. Operators can
   dial down to 5 for spot/sporadic workloads or up to 30 for
   sustained interactive use.
7. **Pre-flight verification: yes, before Phase 2.** Stand up the
   API server in isolation against RunPod (cheapest), `sky check`,
   provision a t-shirt-size VM, run hello-world, tear down. ~30
   min wall-clock, ~$1 budget. Confirms image/extras/auth choices
   land cleanly before any gpu-arbiter integration commits to them.

## Context

devai's gpu-arbiter is a single-host GPU scheduler today. The router design
discussion concluded that the natural growth path is a head/worker split
where gpu-arbiter in head mode dispatches to gpu-arbiter in worker mode
across N hosts. SkyPilot is the open-source tool for managing the
provisioning lifecycle of those hosts across cloud, Slurm, and on-prem
without involving Kubernetes.

The integration choice (Python SDK vs CLI vs API server) was decided in
favour of the API server pattern because:

- gpu-arbiter is Go distroless; embedding Python is wrong.
- The CLI parses brittle stdout output and spawns its own in-process API
  server per call.
- The API server is what SkyPilot themselves recommend for shared/automated
  deployments and is HTTP-native, which matches what gpu-arbiter needs.

The book's Chapter 4 (Docker Offload) covers the cloud-burst pattern but
through a Docker-managed proprietary tool. SkyPilot is the open-source
equivalent supporting 20+ clouds, Slurm, and on-prem with one CLI/API
surface.

## Approach

Run `devai-skypilot-api-server` as a long-lived compose service: a Python
container with SkyPilot installed plus the chosen cloud extras and
credentials mounted from the sops/age-managed secret store. gpu-arbiter
in head mode adds a thin HTTP client to call this server for `launch`,
`status`, `down`, and `exec`. Probe cache + fleet state inform the policy
layer that decides _which_ cloud, instance type, and region to ask for.
Authentication between gpu-arbiter and the API server uses a shared bearer
token, also from the secret store.

```
gpu-arbiter (Go, --mode=head)
     |  HTTP + bearer token
     v
devai-skypilot-api-server (Python container)
     |  cloud SDK calls
     v
RunPod / Lambda / on-prem Slurm / etc.
     |  VM boots, runs cloud-init, starts gpu-arbiter --mode=worker
     v
worker registers with head, accepts forwarded requests
```

---

## Phase 1 -- API server stand-up, no gpu-arbiter integration

### Goal

`devai-skypilot-api-server` runs as a compose service, can be reached from
the lab container via HTTP, can provision and tear down a real cloud VM
when invoked manually. gpu-arbiter is unchanged.

### Deliverables

```
deploy/
  docker-compose.yaml                (modify: + devai-skypilot-api-server
                                       service, profile=cluster)
  skypilot-api.env                   (new: non-secret config)
  skypilot-credentials.sops.env      (new: cloud credentials, encrypted)
scripts/
  skypilot-api-health.sh             (new: smoke test)
docs/
  skypilot.md                        (new: overview, credential setup,
                                       client patterns)
Makefile                             (new targets: skypilot-up,
                                       skypilot-down, skypilot-check)
```

No `Dockerfile.skypilot-api`: the upstream `berkeleyskypilot/skypilot`
image is used as-is per decision 1.

### Detailed steps

1. **Pin the upstream image version.** Confirmed upstream at
   `berkeleyskypilot/skypilot:0.12.1` (smoke-tested 2026-05-14;
   `sky --version` returns `skypilot, version 0.12.1`; the
   documented entrypoint binds container port 46580 within
   ~7s). For dev builds use `berkeleyskypilot/skypilot-nightly:<dev-tag>`.
   The plan's reference pin is updated quarterly alongside the
   bench-image bumps (or on demand if a security fix lands).
2. **Add compose service** with profile=cluster (opt-in,
   matching the MCP gateway's deferred-rollout pattern):
   ```yaml
   devai-skypilot-api-server:
     image: berkeleyskypilot/skypilot:0.12.1
     container_name: devai-skypilot-api-server
     restart: unless-stopped
     networks: [devai-net]
     profiles: [cluster]
     ports: ["${SKYPILOT_API_PORT:-46580}:46580"]
     volumes:
       - skypilot-state:/root/.sky
       - ${HOME}:/root:rw                         # cloud creds via $HOME mount (skypilot-agent-skill decision 3)
       - /run/devai/skypilot-credentials.env:/secrets/.env:ro
     environment:
       - SKYPILOT_GLOBAL_CONFIG=/root/.api-server-config.yaml
     entrypoint: [tini]
     command: ["--", "sky", "api", "start", "--deploy", "--foreground"]
   ```
3. **Mount cloud credentials.** Two routes, used together:
   - `$HOME` mount surfaces interactive-style credentials
     (`~/.aws/`, `~/.config/gcloud/`, `~/.config/sky/`) -- same
     mechanism the lab uses; matches `skypilot-agent-skill.md`
     decision 3.
   - sops-rendered tmpfs `/run/devai/skypilot-credentials.env`
     for non-interactive service credentials (RunPod API key,
     Lambda API key) -- uses the shared sops/age scaffold from
     [sops-age-secrets](./sops-age-secrets.md).
4. **Smoke test** -- from the lab container (note: 46580, not
   30050):
   ```bash
   curl http://devai-skypilot-api-server:46580/api/v1/version
   sky api login --endpoint http://devai-skypilot-api-server:46580
   sky check
   sky launch --cloud runpod --gpus 3090:1 -- echo hello
   sky down --all -y
   ```
5. **Documentation** -- `docs/skypilot.md` covers credential setup,
   `sky check` verification, troubleshooting.

### Exit criteria

- `make skypilot-up` brings the API server live, healthcheck passes.
- `sky check` from the lab reports at least one cloud "enabled."
- Manual provisioning test: a real cloud VM launches, runs hello-world,
  tears down. Total elapsed time and cost recorded in docs.
- gpu-arbiter is unchanged.

### Phase 1 risks

| Risk                                                  | Mitigation                                                |
| ----------------------------------------------------- | --------------------------------------------------------- |
| Credentials format varies per cloud (env vs file)     | Per-cloud setup documented in skypilot.md                 |
| Cloud quota set to zero by default for GPU instances  | Documented as setup step; smoke test catches it           |
| API server state corruption on restart                | Named volume preserves state; documented recovery         |

---

## Phase 2 -- gpu-arbiter integration

### Goal

`gpu-arbiter --mode=head` calls the SkyPilot API server to provision a
worker when no registered worker fits the request. Provisioned worker
boots, registers, accepts forwarded request, serves it. gpu-arbiter
terminates the worker after idle timeout.

### Deliverables

```
gpu-arbiter/
  main.go                            (modify: add SkyPilot client,
                                       provisioning policy, idle teardown)
  skypilot_client.go                 (new: HTTP client for SkyPilot API)
  fleet_state.go                     (new: track which workers exist,
                                       what they have loaded)
  policy.go                          (new: pick GPU type from probe cache
                                       + fit + cost)
docs/
  skypilot.md                        (extend: cluster routing section)
  router.md                          (cross-reference: head-mode behavior)
tests/
  test-fleet-routing.sh              (new: integration test against a
                                       cheap cloud worker)
Makefile                             (new targets: fleet-up, fleet-down,
                                       fleet-test)
```

### Detailed steps

1. **HTTP client for SkyPilot API** -- minimal Go client wrapping the
   endpoints gpu-arbiter actually uses: `POST /launch`, `GET /status`,
   `POST /down`. Bearer-token auth from env (loaded from sops tmpfs).
2. **Fleet state in head** -- in-memory map of registered workers, the
   GPU type each has, what model each currently serves (if any). Updated
   via heartbeats from workers.
3. **Provisioning policy** -- when a request arrives and no registered
   worker can serve it:
   - Consult probe cache for which GPU type fits this (model, ctx).
   - Pick cheapest cloud + instance type matching that GPU class.
   - Check cost budget; reject if over.
   - Call SkyPilot launch with a task spec that boots the worker bootstrap
     image and starts `gpu-arbiter --mode=worker --head=<head-url>`.
   - Block (or 202 + retry-after) the originating request until the worker
     registers, then forward.
4. **Idle teardown** -- background goroutine that periodically
   checks each SkyPilot-provisioned worker. If idle for more than
   `DEVAI_IDLE_MINUTES` (see decision 6 and cluster-mode
   decision 14), do a **two-step graceful teardown**
   to reconcile with cluster-mode decision 3 (lifecycle =
   ephemeral workers honour `shutdown` commands):
   - Step 1: send `shutdown { grace_seconds: N }` to the worker
     via its next heartbeat response. Worker drains in-flight
     requests (refusing new ones), unloads its model, exits the
     arbiter process cleanly. Head waits for either the worker's
     heartbeat to stop (clean exit) or `grace_seconds` to elapse.
   - Step 2: regardless of how step 1 ended, call SkyPilot
     `POST /down` to release the cloud VM. The VM teardown is
     the only step that actually stops the cloud bill -- the
     soft shutdown is purely to avoid killing in-flight requests.
   Default `grace_seconds = 30`. Workers launched with
   `lifecycle = persistent` (on-prem, systemd-supervised) never
   receive a shutdown command (worker-side refusal per decision 3)
   and SkyPilot is not in their picture anyway, so step 2 does
   not apply.
5. **Local-fleet-only mode** -- gpu-arbiter `--mode=head` works without
   SkyPilot configured (provisioning policy degrades to "503 if no
   registered worker fits"). SkyPilot is opt-in via env var
   `SKYPILOT_API_ENDPOINT`.
6. **Integration test** -- `tests/test-fleet-routing.sh` makes a request
   for a model that does not fit locally, waits for the cloud worker
   to come up, verifies the response, verifies idle teardown after the
   timeout.

### Exit criteria

- Request to head for a model that does not fit locally triggers a
  SkyPilot provision, eventual response, and idle teardown.
- Local-only mode (no SKYPILOT_API_ENDPOINT) works the same as today's
  gpu-arbiter for single-host operation.
- Mixed mode: a request that fits a registered on-prem worker goes there,
  not to the cloud.
- Cost-cap rejection is surfaced as a meaningful 4xx with explanation.

### Phase 2 risks

| Risk                                                   | Mitigation                                                       |
| ------------------------------------------------------ | ---------------------------------------------------------------- |
| Cold-start latency (provision + boot + model load)     | Document expected 5-15 min cold path; surface in API response    |
| Worker fails to register after VM provisions           | Timeout + auto-terminate; surface as 5xx to caller               |
| Split-brain between SkyPilot state and head fleet view | Reconcile loop: head queries `sky status` periodically           |
| Probe cache missing for a rented GPU type              | Trigger on-the-fly probe on first worker of that GPU type        |
| Race: two requests arrive simultaneously for a model not yet loaded anywhere | Coalesce: second waits for first's provisioning to complete |

---

## Phase 3 -- Policy hardening

### Goal

Cost, latency, and reliability policies that actually look like a production
fleet rather than a toy.

### Deliverables

- Budget enforcement per (user, day, cloud) with sensible defaults.
- Spot-instance preference where supported, with fallback to on-demand.
- Pre-warming: option to keep N workers of a given GPU class warm.
- Observability hooks (metrics surface for "currently provisioning",
  "idle teardown counter", "budget consumed").

### Exit criteria

- Budget overrun blocks new provisioning, surfaces clearly to caller.
- Spot preference verified against a real preemption event.
- Pre-warming reduces p50 cold-start TTFT below a documented threshold.

### Phase 3 risks

Deferred to the time we get there. This phase is exploratory; expect to
discover policy bugs only by running it.

---

## Combined risk register

| Risk                                                   | Phase | Mitigation                                                       |
| ------------------------------------------------------ | ----- | ---------------------------------------------------------------- |
| Cloud quota at zero by default                         | 1     | Documented as install step                                       |
| Cold-start latency surprises users                     | 2     | Document expected 5-15 min path                                  |
| Split-brain between gpu-arbiter and SkyPilot state     | 2     | Periodic reconcile loop                                          |
| Probe cache miss for a rented GPU type                 | 2     | On-the-fly probe trigger                                         |
| Cost runaway from a stuck request loop                 | 2-3   | Per-day budget cap, hard kill on overrun                         |
| Spot preemption mid-request                            | 3     | Retry on a non-spot instance, surface failure if both unavailable |

## Migration / rollback story

- **Phase 1 rollback**: stop the API server service (`make skypilot-down`);
  no other surface is affected.
- **Phase 2 rollback**: drop the `SKYPILOT_API_ENDPOINT` env var; head mode
  reverts to local-fleet-only routing.
- **Phase 3 rollback**: feature flags per policy (budget, pre-warming, etc.)
  -- disable individually.
- **Upgrade path**: existing single-host devai installs gain SkyPilot only
  via `--profile cluster`. Without that, nothing changes.

## Estimated effort

| Phase    | Engineering effort                                | Wall-clock        |
| -------- | ------------------------------------------------- | ----------------- |
| Phase 1  | ~1 PR, compose + dockerfile + smoke test          | 2-3 days          |
| Phase 2  | ~1-2 PRs, ~500-1000 lines of Go in gpu-arbiter    | 1-2 weeks         |
| Phase 3  | ~1 PR, policy + observability                     | 1 week            |
| Total    | 3-4 PRs                                           | 3-4 weeks elapsed |

## References

- SkyPilot Python SDK: docs.skypilot.co/en/latest/reference/api.html
- SkyPilot API server: docs.skypilot.co/en/latest/reference/api-server/
- SkyPilot v0.12 release notes (Slurm support, Agent Skills, Pool
  autoscaling): March 2026
- SkyPilot GitHub: github.com/skypilot-org/skypilot
- Book reference: "Operational AI with Docker" Chapter 4 (Docker Offload)
  covers the cloud-burst pattern via a different (proprietary) tool.
- Related plans:
  - Sibling: [skypilot-agent-skill](./skypilot-agent-skill.md) (user-facing
    SkyPilot integration; independent of this plan)
  - Reuses secret pattern: [mcp-gateway](./mcp-gateway.md) Phase 2
