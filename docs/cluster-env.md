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
| `DEVAI_WORKER_HOST`            | `localhost`      | hostname or IP                  | arbiter | gpu-arbiter-cluster-mode Phase 1  |
| `DEVAI_WORKER_ENDPOINT`        | (computed)       | URL                             | arbiter | gpu-arbiter-cluster-mode Phase 1  |
| `DEVAI_ARBITER_VERSION`        | `dev`            | git sha or tag                  | arbiter | gpu-arbiter-cluster-mode Phase 1  |

## Head-side (Phase 2)

| Variable                  | Default     | Allowed values | Read by | Owning plan                         |
| ------------------------- | ----------- | -------------- | ------- | ----------------------------------- |
| `DEVAI_MODE=head`         | --          | `head`         | arbiter | gpu-arbiter-cluster-mode Phase 2    |
| `DEVAI_IDLE_MINUTES`      | `10`        | positive int   | arbiter | gpu-arbiter-cluster-mode Phase 2 (decision 14) |
| `DEVAI_HEAD_LISTEN_PORT`  | `11444`     | port           | arbiter | gpu-arbiter-cluster-mode Phase 2    |
| `SKYPILOT_API_ENDPOINT`   | (unset)     | URL            | arbiter | skypilot-fleet-provisioner Phase 2  |
| `SKYPILOT_API_PORT`       | `46580`     | port           | compose | skypilot-fleet-provisioner Phase 1  |

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
