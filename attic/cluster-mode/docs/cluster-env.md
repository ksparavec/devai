# Cluster-mode env-var contract

Canonical per-env-var reference. Downstream consumer plans
([sops-age-secrets](plans/sops-age-secrets.md),
[skypilot-fleet-provisioner](plans/skypilot-fleet-provisioner.md),
[mcp-gateway](plans/mcp-gateway.md)) extend this table when they
add their own variables -- PR review checklist enforces it.

## Worker-side

| Variable                       | Default          | Allowed values                  | Read by | Owning plan                       |
| ------------------------------ | ---------------- | ------------------------------- | ------- | --------------------------------- |
| `DEVAI_MODE`                   | `single`         | `single`, `worker`, `head`      | arbiter | gpu-arbiter-cluster-mode Phase 1  |
| `DEVAI_HEAD_URL`               | (required)       | URL                             | arbiter | gpu-arbiter-cluster-mode Phase 1  |
| `DEVAI_WORKER_TOKEN_FILE`      | `/run/devai/cluster-token` | path                  | arbiter | gpu-arbiter-cluster-mode Phase 1  |
| `DEVAI_WORKER_NAME`            | `$(hostname)`    | non-empty string                | arbiter | gpu-arbiter-cluster-mode Phase 1  |
| `DEVAI_LIFECYCLE`              | `persistent`     | `ephemeral`, `persistent`       | arbiter | gpu-arbiter-cluster-mode Phase 1  |
| `DEVAI_GPU_TYPE`               | `unknown`        | short label (e.g. `RTX4000`)    | arbiter | gpu-arbiter-cluster-mode Phase 1  |
| `GPU_MEMORY_GB`                | `24`             | positive int                    | arbiter | gpu-arbiter-cluster-mode Phase 1  |
| `DEVAI_BACKENDS`               | `ollama,vllm,sglang` | comma-separated subset      | arbiter | gpu-arbiter-cluster-mode Phase 1  |
| `DEVAI_WORKER_INBOUND_PORT`    | `11444`          | port                            | arbiter | gpu-arbiter-cluster-mode Phase 1  |
| `DEVAI_WORKER_HOST`            | `$(hostname)`    | hostname or IP                  | arbiter | gpu-arbiter-cluster-mode Phase 1  |
| `DEVAI_WORKER_ENDPOINT`        | (computed)       | URL                             | arbiter | gpu-arbiter-cluster-mode Phase 1  |
| `DEVAI_ARBITER_VERSION`        | `dev`            | git sha or tag                  | arbiter | gpu-arbiter-cluster-mode Phase 1  |

## Head-side (Phase 2)

| Variable                  | Default     | Allowed values | Read by | Owning plan                         |
| ------------------------- | ----------- | -------------- | ------- | ----------------------------------- |
| `DEVAI_MODE=head`         | --          | `head`         | arbiter | gpu-arbiter-cluster-mode Phase 2    |
| `DEVAI_IDLE_MINUTES`      | `10`        | positive int   | arbiter | gpu-arbiter-cluster-mode Phase 2 (decision 14) |
| `DEVAI_HEAD_LISTEN_PORT`  | `11444`     | port           | arbiter | gpu-arbiter-cluster-mode Phase 2    |
| `DEVAI_HEAD_TOKEN_FILE`   | `/run/devai/cluster-token` | path | arbiter | gpu-arbiter-cluster-mode Phase 2    |
| `DEVAI_QUEUE_DEPTH_THRESHOLD` | `0`     | non-negative int (`0` = disabled) | arbiter | gpu-arbiter-cluster-mode Phase 2 |
| `SKYPILOT_API_ENDPOINT`   | (unset)     | URL            | (not yet read -- see below) | skypilot-fleet-provisioner Phase 2  |
| `SKYPILOT_API_PORT`       | `46580`     | port           | compose | skypilot-fleet-provisioner Phase 1  |
| `DEVAI_HEAD_STATUS_URL`   | `http://localhost:11444` | base URL | Makefile (`make cluster-status`) | gpu-arbiter-cluster-mode Phase 2 |

`DEVAI_HEAD_STATUS_URL` is the odd one out: it is read by the
**Makefile**, not by the arbiter, and only by the `cluster-status`
target, which needs a base URL it can reach from the host. Its default
is deliberately optimistic -- neither `deploy/docker-compose.yaml` nor
`deploy/compose.head.yaml` publishes `:11444` to the host, so on a
default deployment the target fails with a message naming that cause
and pointing back at this variable. See
[docs/cluster-mode.md](cluster-mode.md) ("Authentication").

`DEVAI_HEAD_TOKEN_FILE` is the head-side counterpart of
`DEVAI_WORKER_TOKEN_FILE`: the head reads the same shared bearer
token from it (`gpu-arbiter/cluster_head.go`) and requires it on
`/v1/cluster/register`, `/v1/cluster/heartbeat`, `/v1/cluster/status`
and on every worker-bound `/v1/cluster/inbound` call. Both sides
default to `/run/devai/cluster-token`, so a single rendered secret
serves head and worker.

`SKYPILOT_API_ENDPOINT` is **not read by gpu-arbiter today.** The
Phase 2 client and policy code exists (`gpu-arbiter/skypilot_client.go`,
`skypilot_policy.go`) but `NewSkyPilotClient` has no non-test caller,
so `runHeadMode` never constructs it -- setting the variable on a head
currently has no effect. It stays in this table as the reserved name
the fleet provisioner will use once the head wires it in.

## Conventions

- Variables prefixed `DEVAI_` are owned by this project and never
  shadowed by upstream tools.
- Variables shared with consumer plans (e.g. `DEVAI_IDLE_MINUTES`
  is read by the head AND by the SkyPilot Phase 2 idle-teardown
  loop) have a single owning plan -- the others reference it.
- Defaults are conservative; cluster-mode is opt-in.
- Boolean-like vars accept the literal string `1` for true,
  anything else for false.

## Adding a new variable

When a downstream plan introduces a new env var:

1. Add a row to the appropriate table above with the default and
   allowed values.
2. Cite the owning plan in the last column.
3. Document the parsing rule in the variable's first reader.
4. Make a single PR that lands the table entry, the parsing code,
   and the docs together -- partial commits create drift.
