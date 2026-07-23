# gpu-arbiter cluster mode

The `gpu-arbiter` binary supports three modes via the `--mode` flag
(or `DEVAI_MODE` env var):

| Mode      | Behaviour                                                                            |
| --------- | ------------------------------------------------------------------------------------ |
| `single`  | Default. Today's per-host scheduler -- byte-identical to pre-cluster-mode behaviour. |
| `worker`  | Runs the full single-host scheduler, registers with a head, sends heartbeats, serves forwarded requests. |
| `head`    | Receives registrations, scores worker fleet, proxies requests. No local backends.    |

This is the operator reference. Architecture rationale and phased
delivery live in
[docs/plans/gpu-arbiter-cluster-mode.md](plans/gpu-arbiter-cluster-mode.md).

## Quick orientation

- Single-host devai users: nothing changes. `make cache-up` /
  `make cache-down` / `make cache-status` continue to work; cluster
  mode is opt-in.
- Multi-host on-prem: pick one host as head, the rest as workers.
  All run the same `gpu-arbiter` binary; the `--mode` flag selects
  behaviour.
- Cloud burst (SkyPilot): see
  [docs/plans/skypilot-fleet-provisioner.md](plans/skypilot-fleet-provisioner.md)
  for the cloud-side provisioning. The protocol the cloud workers
  speak is what this doc describes.

## Worker mode

```bash
# On the worker host:
export DEVAI_MODE=worker
export DEVAI_HEAD_URL=http://devai-head.lan:11444
export DEVAI_WORKER_TOKEN_FILE=/run/devai/cluster-token
export DEVAI_WORKER_NAME=desktop-4090
export DEVAI_LIFECYCLE=persistent      # or ephemeral
export GPU_MEMORY_GB=24
export DEVAI_GPU_TYPE=RTX4000          # short label the head uses for routing
gpu-arbiter --mode=worker
```

A worker builds the **same arbiter a single host runs** -- the whole
rewrite chain, GPU exclusion, container recreate and idle watcher stay
on the worker (cluster-mode decision 2). It therefore needs everything
a single host needs, and honours `IDLE_TIMEOUT`, `DRAIN_TIMEOUT`,
`HEALTH_TIMEOUT_SECONDS` and `MAX_CONCURRENT_REQUESTS` identically.
What it does *not* do is mount the per-backend 11434/5/6 listeners:
`POST /v1/cluster/inbound` is a worker's only serving surface.

Pre-conditions:

1. `/run/devai/cluster-token` exists and contains the bearer token
   shared with the head. The repo does **not** ship an encrypted
   token file: create one (conventionally
   `deploy/cluster-token.sops.env`) with the shared
   [docs/secrets.md](secrets.md) scaffold, or render the token by
   any other means -- the arbiter only ever reads the plaintext path.
2. The host has its own probe cache (`make probe`,
   `make probe-vllm`, `make probe-sglang`) so the worker advertises
   only backends it can actually serve. A worker with no probe cache
   registers with **zero** models and rejects every forwarded request.
3. The podman socket and the model weights are present, exactly as on
   a single host -- the worker launches its own backend containers.
4. The worker's inbound port (`DEVAI_WORKER_INBOUND_PORT`,
   default 11444) is reachable from the head over the network. The
   endpoint the worker advertises is
   `http://$DEVAI_WORKER_HOST:$DEVAI_WORKER_INBOUND_PORT`, and
   `DEVAI_WORKER_HOST` defaults to the host's own hostname (not
   `localhost`) -- override it when the head must reach the worker
   under a different name or IP.

## Lifecycle classes

Per [plan decision 3](plans/gpu-arbiter-cluster-mode.md):

- **`ephemeral`**: head MAY send `shutdown` after
  `DEVAI_IDLE_MINUTES` (head-side env var, default 10) of no
  requests. Worker drains in-flight requests, exits the arbiter
  process. SkyPilot-launched cloud VMs default to this.
- **`persistent`**: head MUST NOT send `shutdown`. If one arrives
  anyway (head bug), the worker logs the refusal and stays up.
  Loaded models stay loaded for fast re-use. On-prem default.

## Head mode

Phase 2 shipped 2026-05-15. The head host runs the same arbiter
binary in head mode -- no GPU on the head, no backend containers,
just routing:

```bash
make cluster-head-up      # compose override disables local backends
make cluster-status       # pretty-prints /v1/cluster/status JSON
```

Or directly:

```bash
export DEVAI_MODE=head
gpu-arbiter --mode=head
```

The head:

1. Listens on the cluster control plane port (`DEVAI_HEAD_LISTEN_PORT`,
   default 11444) for `/v1/cluster/{register,heartbeat,status}`.
2. Listens on the OpenAI-compat ports (11434/5/6) for client requests
   and proxies to the highest-scoring registered worker.
3. Runs an idle-sweep loop that expires workers whose last heartbeat
   is older than `HeartbeatTTL` (30s default).
4. Optionally injects `shutdown` commands into ephemeral workers
   that have been idle for `DEVAI_IDLE_MINUTES`.

### Routing scoring

Per the plan's decision 4:

| Score | Condition                                              |
| ----- | ------------------------------------------------------ |
| 100   | `loaded_model == model AND loaded_ctx >= ctx`          |
| 50    | `loaded_model == model AND loaded_ctx <  ctx`          |
| 30    | `loaded_model == ""` (idle, no cold-load yet)          |
| 10    | `loaded_model != model` (different model loaded)       |

Tiebreak: round-robin per bucket. Workers above
`DEVAI_QUEUE_DEPTH_THRESHOLD` are skipped (overloaded), as are workers
reporting `health_status: draining` or `shutting_down` (see
[Heartbeat response](#heartbeat-response-commands)). When no
worker fits, the head returns 503 + a JSON `no_worker_fit` error
with the requested model/ctx/backend echoed back.

**Proxy timeout.** The head does not wait indefinitely for a worker to
start answering. Its `ResponseHeaderTimeout` is derived, not fixed:
`2 * HEALTH_TIMEOUT_SECONDS + DRAIN_TIMEOUT`, read from the **head's
own** environment (20.5 min at the defaults 600 + 30). Two health
timeouts because a forwarded request can queue behind one unrelated
recreate and then pay its own. An operator who raises
`HEALTH_TIMEOUT_SECONDS` on workers must raise it on the head too --
otherwise the head cuts a legitimately slow worker loose with a 502.

### compose.head.yaml override

`deploy/compose.head.yaml` is the head-host overlay. It sets
`DEVAI_MODE=head` on the router and zeroes the local backend
service replicas so the head host's compose stack doesn't include
ollama/vllm/sglang. Apply via:

```bash
podman-compose -f deploy/docker-compose.yaml \
               -f deploy/compose.head.yaml up -d router
```

## Authentication

Bearer token via `Authorization: Bearer <token>` on every cluster
control-plane request. The worker loads it from
`DEVAI_WORKER_TOKEN_FILE`, the head from `DEVAI_HEAD_TOKEN_FILE`;
both default to `/run/devai/cluster-token`, so one rendered secret
serves the whole fleet. Re-read on a 30-second cache TTL, so a
rotated token becomes effective on the next heartbeat without a
restart on either side.

Authenticated: `/v1/cluster/register`, `/v1/cluster/heartbeat`,
`/v1/cluster/status` on the head, and `/v1/cluster/inbound` on the
worker. `GET /health` -- on the control plane and on the head's
frontend ports alike -- stays unauthenticated so liveness probes need
no secret.

Both in-tree readers of `/v1/cluster/status` send the header.
`make cluster-status` reads `DEVAI_HEAD_TOKEN_FILE` (default
`/run/devai/cluster-token`) and refuses to run with a pointer to the
render commands when that file is not readable. devai-tools'
`get_router_status` resolves the token from `DEVAI_HEAD_TOKEN_FILE`,
then `DEVAI_WORKER_TOKEN_FILE`, then the same default path, and
surfaces a 401 in its `cluster_error` field rather than silently
degrading to the per-backend `/health` fallback.

The base URL `make cluster-status` targets is the make variable
`DEVAI_HEAD_STATUS_URL` (default `http://localhost:11444`). **That
default is very likely NOT reachable as shipped:** neither the `router`
service in `deploy/docker-compose.yaml` nor the
`deploy/compose.head.yaml` overlay declares a `ports:` block, so
`:11444` is bound only inside the `devai-net` container network
(`podman compose -f deploy/docker-compose.yaml -f
deploy/compose.head.yaml config` renders the router with `ports:
null`). Reach it from another container on `devai-net`, publish the
port yourself, tunnel it (`ssh -L`), or point the variable at whatever
your deployment actually exposes:

```bash
make cluster-status DEVAI_HEAD_STATUS_URL=http://devai-head.lan:11444
```

The equivalent by hand -- note the token goes to curl over a **pipe**
(`-H @-`, curl >= 7.55), never as a command-line argument, because
argv is world-readable through `/proc/<pid>/cmdline`:

```bash
printf 'Authorization: Bearer %s\n' "$(cat /run/devai/cluster-token)" \
  | curl -fsS -H @- http://devai-head.lan:11444/v1/cluster/status \
  | python3 -m json.tool
```

Rotate via the shared scaffold (the encrypted file is operator-created
-- see pre-condition 1 above):

```bash
make secrets-edit SOPS_FILE=deploy/cluster-token.sops.env
make secrets-render SOPS_FILE=deploy/cluster-token.sops.env DEST=/run/devai/cluster-token
```

## Endpoints

### Worker -> Head

| Method | Path                       | Purpose                              |
| ------ | -------------------------- | ------------------------------------ |
| POST   | `/v1/cluster/register`     | Once at startup; head returns `worker_id`. |
| POST   | `/v1/cluster/heartbeat`    | Every 10s; carries state, receives commands. |

### Head -> Worker

| Method | Path                       | Purpose                              |
| ------ | -------------------------- | ------------------------------------ |
| POST   | `/v1/cluster/inbound`      | Forwarded request body for the worker to serve. |
| GET    | `/health`                  | Cheap liveness probe.                |

`/v1/cluster/inbound` serves for real: the worker replays the body
through its own single-host request handler, so a clustered request is
rewritten exactly like a local one. Headers the head sets and the
worker honours:

| Header                  | Meaning                                                      |
| ----------------------- | ------------------------------------------------------------ |
| `Authorization`         | `Bearer <token>` -- required; 401 otherwise.                  |
| `X-Devai-Backend`       | Which backend the head's frontend received the request on (`ollama` \| `vllm` \| `sglang`). Authoritative. |
| `X-Devai-Original-Path` | The client's original path (e.g. `/v1/chat/completions`), restored onto the worker-side request. |
| `X-Devai-Worker-Id`     | The head's chosen `worker_id`; informational only -- the token is what authenticates. |

With `X-Devai-Backend` absent (a caller hitting the endpoint directly),
the worker looks the request's model name up in its own backends and
ranks the ones that carry that name, most-specific first
(`backendForModel`):

1. a backend whose probe cache has a fitting cell covering the
   request's ctx (probed max ctx >= ctx) -- the only tier with real
   evidence the request can be served; highest ceiling wins;
2. failing that, a backend with any fitting probe cell for the model
   (probed max ctx > 0), highest ceiling first -- used when ctx was
   unstated or no backend reaches it;
3. failing that, bare catalog membership in the fixed order
   `ollama`, `vllm`, `sglang`, which is only a deterministic tie-break
   and not itself a preference.

If no backend serves the name the worker answers **HTTP 400**
(`invalid_request_error`), not 503 -- the request named something this
worker does not serve.

## Heartbeat shape

```json
{
  "worker_id": "wid-abc123",
  "loaded_model": "Qwen3-8B-NVFP4",
  "loaded_ctx": 131072,
  "queue_depth": 0,
  "utilization_pct": 12.4,
  "last_request_at": "2026-05-15T10:00:00Z",
  "health_status": "ready",
  "counter": 421
}
```

`counter` is monotonic per worker so the head can drop out-of-order
heartbeats. `loaded_model` / `loaded_ctx` / `last_request_at` are
omitted when the worker has nothing loaded yet.

## Heartbeat response (commands)

```json
{
  "commands": [
    { "type": "drain", "backend": "vllm" },
    { "type": "shutdown", "grace_seconds": 30 }
  ]
}
```

Recognised commands:

| Type       | Fields                                  | Notes                                       |
| ---------- | --------------------------------------- | ------------------------------------------- |
| `drain`    | `backend`                               | Really drains: the worker's executor calls the single-host scheduler's `drainBackend` and waits out the in-flight requests (bounded by `DRAIN_TIMEOUT`). |
| `shutdown` | `grace_seconds`                         | Drain every backend, then exit the process. Persistent refuses. |
| `serve`    | `request_id`, `target_model`, `target_ctx`, `body_url`, `response_path` | Reserved. The head never emits it -- request forwarding is the synchronous `/v1/cluster/inbound` proxy. |

Unknown types are logged and ignored; forward-compatible.

Both commands are acknowledged on the heartbeat that carried them and
executed in the background, so the drain never blocks the heartbeat
goroutine -- which matters precisely because the next heartbeat is what
propagates the new `health_status`.

**A drain is transient; a shutdown is terminal.** The worker's health
state machine (`WorkerState`, `gpu-arbiter/cluster_worker.go`) is:

```
ready    --drain-->     draining  --drain complete-->  ready
ready    --shutdown-->  shutting_down                  (terminal)
draining --shutdown-->  shutting_down                  (terminal)
```

`draining` is **not** latched: when the drain finishes, the worker
compare-and-swaps `draining -> ready` and is routable again on its next
heartbeat. The CAS is what keeps a shutdown that arrived *mid-drain*
terminal -- the completing drain finds the status is no longer
`draining` and does not resurrect the worker. Leaving `draining`
latched was the old behaviour and turned a bounded drain into a
permanent removal from the fleet.

**Routing consequence.** A worker reporting `draining` or
`shutting_down` is skipped by the head's normal routing pass
(`workerAvailable`) -- that, not the drain itself, is what stops NEW
work arriving. Any other status value (including `registered` from a
worker that has not heartbeated yet, and `""` from an older worker
build) counts as available, so an unknown status never silently
removes a worker.

If that pass finds **no** candidate and at least one worker was skipped
only because it is `draining`, the head runs a second, **degraded**
pass that admits draining workers, logs

```
[head] degraded route: every non-draining worker for backend=<b> was
unavailable; forwarding model="<m>" to draining worker <id> (expect
added latency)
```

and sets `RoutingDecision.Degraded`. This is why a bounded drain no
longer 503s a single-worker fleet: a drain holds the arbiter mutex
while it waits out in-flight work, so a newly forwarded request parks
on that mutex and is served when the drain returns, bounded by
`DRAIN_TIMEOUT` (30s default). `shutting_down` is **never** admitted by
the degraded pass -- that process exits at the end of its drain, so the
request would die with it.

Only when both passes come up empty does the head answer 503, and the
`no_worker_fit` reason then says how many workers were skipped for
draining or shutting down.

**`grace_seconds` is the head's budget, not a sleep.** The worker does
not wait it out: it drains every backend and exits as soon as the
in-flight work is done, so a fast drain exits fast and a slow one is
not cut off at the grace. The head uses `grace_seconds` to decide how
long to wait before it calls `sky down` on an ephemeral worker.

## Troubleshooting

### Worker registers but never receives commands

The head is up and reachable, but no requests are landing on the
head. Verify by hitting the head's `GET /v1/cluster/status` with the
bearer token -- `make cluster-status` sends it for you (see
Authentication above for the equivalent curl form).

If the head was restarted, its in-memory fleet map is empty and it
answers the worker's next heartbeat with `410 Gone`. The worker treats
that as "re-register" and does so automatically on the next cycle -- a
head bounce no longer orphans workers permanently. Look for the
re-registration in the worker log before suspecting a network fault.

### Worker keeps retrying registration

Either the head is unreachable (network, firewall, wrong URL) or
the bearer token doesn't match. Check the worker's logs for the
HTTP status code from `/v1/cluster/register`:

- `401`: token mismatch -- verify both sides decrypt to the same
  value.
- network error: confirm the head's listen port, firewall rules,
  and DNS.

### Worker exits unexpectedly

`ephemeral` workers exit when the head sends `shutdown`. Look in
the worker's log for `[worker] shutdown grace=...`. If the worker
exits without that log line, it's a real crash -- check stderr,
podman logs.

### Heartbeat counter going backwards

Each worker process owns its own monotonic counter. If the worker
restarts, the counter resets to 1. The head should treat counter=1
as "this is a re-registered worker" and accept it; if not, that's
a head-side bug worth filing against the cluster-mode plan.

## References

- Plan: [docs/plans/gpu-arbiter-cluster-mode.md](plans/gpu-arbiter-cluster-mode.md)
- Auth scaffold: [docs/secrets.md](secrets.md)
- Per-env-var contract: [docs/cluster-env.md](cluster-env.md)
- Cloud-init for SkyPilot workers: [docs/worker-bootstrap.md](worker-bootstrap.md)
- Sibling cloud-provisioner plan:
  [docs/plans/skypilot-fleet-provisioner.md](plans/skypilot-fleet-provisioner.md)
