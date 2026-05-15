# gpu-arbiter cluster mode

The `gpu-arbiter` binary supports three modes via the `--mode` flag
(or `DEVAI_MODE` env var):

| Mode      | Behaviour                                                                            |
| --------- | ------------------------------------------------------------------------------------ |
| `single`  | Default. Today's per-host scheduler -- byte-identical to pre-cluster-mode behaviour. |
| `worker`  | Registers with a head, sends heartbeats, accepts forwarded requests.                 |
| `head`    | (Phase 2) Receives registrations, scores worker fleet, proxies requests.             |

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

Pre-conditions:

1. `/run/devai/cluster-token` exists and contains the bearer token
   shared with the head -- rendered from the sops-encrypted
   `cluster-token.sops.env` via the shared
   [docs/secrets.md](secrets.md) scaffold.
2. The host has its own probe cache (`make probe`,
   `make probe-vllm`, `make probe-sglang`) so the worker advertises
   only backends it can actually serve.
3. The worker's inbound port (`DEVAI_WORKER_INBOUND_PORT`,
   default 11444) is reachable from the head over the network.

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

Phase 2 of the cluster-mode plan; not yet implemented. When it
lands, the operator runs:

```bash
export DEVAI_MODE=head
gpu-arbiter --mode=head
```

The head listens on the same OpenAI-compat ports (11434/5/6) and
proxies to whichever registered worker scores highest for the
incoming `(model, ctx)` request.

## Authentication

Bearer token via `Authorization: Bearer <token>` on every cluster
control-plane request. Token loaded from
`DEVAI_WORKER_TOKEN_FILE`, re-read on a 30-second cache TTL so a
rotated token becomes effective on the next heartbeat without a
worker restart.

Rotate via the shared scaffold:

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

### Head -> Worker (Phase 2)

| Method | Path                       | Purpose                              |
| ------ | -------------------------- | ------------------------------------ |
| POST   | `/v1/cluster/inbound`      | Forwarded request body for the worker to serve. |
| GET    | `/health`                  | Cheap liveness probe.                |

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
| `drain`    | `backend`                               | Drain in-flight requests on `backend`.      |
| `shutdown` | `grace_seconds`                         | Drain everything, exit. Persistent refuses. |
| `serve`    | `request_id`, `target_model`, `target_ctx`, `body_url`, `response_path` | Phase 2: forward request body for serving.  |

Unknown types are logged and ignored; forward-compatible.

## Troubleshooting

### Worker registers but never receives commands

The head is up and reachable, but no requests are landing on the
head. Verify with `make cluster-status` (Phase 2 target) or by
hitting the head's `GET /v1/cluster/status` directly.

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
