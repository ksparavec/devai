# SkyPilot fleet provisioner

The `devai-skypilot-api-server` compose service runs the upstream
[SkyPilot](https://skypilot.co/) API server as a peer to
`devai-router` (in head mode). The head calls it on demand to
provision cluster workers across cloud, Slurm, and on-prem when no
locally-registered worker can serve a request.

This is the operator reference for the **system-side** integration.
The user-facing SkyPilot CLI inside the lab container is a separate
plan -- see [docs/skypilot-user-guide.md](skypilot-user-guide.md)
for the agent-driven flow.

## Status snapshot

- **Phase 1 shipped** (2026-05-15): `devai-skypilot-api-server` runs
  as a long-lived compose service on the `cluster` profile.
  Reachable on port 46580. `make skypilot-up` brings it live.
- **Phase 2 pending**: gpu-arbiter head-mode integration --
  `skypilot_client.go`, provisioning policy, two-step idle teardown.
  Worker bootstrap image (Phase 1 of cluster-mode) is the consumer
  surface.
- **Phase 3 pending**: cost-cap enforcement, spot-instance preference,
  pre-warming, observability hooks.

## Bring-up

Pre-conditions:

1. Cluster mode is opt-in. Bring up the head + workers per
   [docs/cluster-mode.md](cluster-mode.md).
2. Cloud credentials available -- either via `$HOME` mount (which
   surfaces `~/.aws`, `~/.config/gcloud`, `~/.config/sky`) or the
   sops-rendered tmpfs file (for non-interactive RunPod / Lambda
   keys).

```bash
# Optional: render service credentials.
make age-keygen-host          # one-time per host
# add the printed public key to .sops.yaml
cp deploy/skypilot-credentials.sops.env.example /tmp/sky-creds.env
$EDITOR /tmp/sky-creds.env    # fill in real keys
sops --encrypt /tmp/sky-creds.env > deploy/skypilot-credentials.sops.env
shred -u /tmp/sky-creds.env
make skypilot-secrets-render

# Bring the API server up.
SKYPILOT_CREDENTIALS_FILE=/run/devai/skypilot-credentials.env \
  make skypilot-up

make skypilot-check           # /api/v1/version + sky check
```

To stop:

```bash
make skypilot-down
```

## Without sops-rendered creds

`SKYPILOT_CREDENTIALS_FILE` defaults to `/dev/null`, so
`make skypilot-up` works on a fresh install with only the `$HOME`
mount supplying credentials. SkyPilot's `sky check` will report
which clouds it sees.

## Volume layout

| Volume                        | Purpose                                     |
| ----------------------------- | ------------------------------------------- |
| `skypilot-state` (named)      | SkyPilot's `~/.sky` registry, task logs.    |
| `${HOME}` -> `/root` bind     | Operator's interactive cloud creds.         |
| `${SKYPILOT_CREDENTIALS_FILE}` -> `/secrets/.env` ro | sops-rendered service creds. |

## Endpoints

| Endpoint                     | Purpose                                      |
| ---------------------------- | -------------------------------------------- |
| `GET  /api/v1/version`       | Liveness probe.                              |
| `POST /api/v1/launch`        | Provision a SkyPilot cluster.                |
| `GET  /api/v1/status`        | List active clusters.                        |
| `POST /api/v1/down`          | Tear down a cluster.                         |
| `POST /api/v1/exec`          | Run a command on an existing cluster.        |

The full SkyPilot API surface lives at
[docs.skypilot.co/en/latest/reference/api-server/](https://docs.skypilot.co/en/latest/reference/api-server/).

## Phase 2 preview

When Phase 2 ships, gpu-arbiter in head mode will call this server
on no-fit requests:

1. Probe-cache lookup: which GPU type fits this `(model, ctx, backend)`?
2. Pick cheapest cloud + instance type matching that GPU class.
3. Check budget; reject if over.
4. Call `/api/v1/launch` with the worker-bootstrap image as the task
   spec. Worker boots, runs cloud-init, registers with the head.
5. Forward the originating request to the new worker.
6. After `DEVAI_IDLE_MINUTES` of no requests: send `shutdown` to the
   worker via heartbeat (decision 3 lifecycle), then `/api/v1/down`
   on the cloud VM.

## Cost guidance

A 24-hour H100 you forget to tear down is ~$120. Mitigations:

- `make skypilot-check` shows enabled clouds + `sky cost-report`.
- Phase 3 adds head-side budget caps (per cluster-mode decision 5).
- Set per-cloud spending alerts directly with the cloud provider as
  belt-and-suspenders.

## Troubleshooting

### `make skypilot-up` fails with "image pull failed"

The `berkeleyskypilot/skypilot:0.12.1` image is ~660 MiB compressed.
First pull may take a few minutes. Check `podman images` to confirm
the layers are downloading; bandwidth-limited environments may need
to pre-pull manually.

### `sky check` shows zero enabled clouds

Either no `$HOME` mount surfaced cloud creds OR
`SKYPILOT_CREDENTIALS_FILE` wasn't set when starting the container.
Verify:

```bash
podman exec devai-skypilot-api-server ls -la /root/.aws /root/.config/sky 2>&1 || true
podman exec devai-skypilot-api-server ls -la /secrets/.env 2>&1 || true
```

### Container exits immediately

Check `podman logs devai-skypilot-api-server` for an upstream
SkyPilot error. The API server requires Python 3.10+ and a working
network connection on first start.

## References

- Plan: [docs/plans/skypilot-fleet-provisioner.md](plans/skypilot-fleet-provisioner.md)
- User-side CLI: [docs/skypilot-user-guide.md](skypilot-user-guide.md)
- Cluster mode: [docs/cluster-mode.md](cluster-mode.md)
- Secrets scaffold: [docs/secrets.md](secrets.md)
- Upstream: github.com/skypilot-org/skypilot
- API server image: hub.docker.com/r/berkeleyskypilot/skypilot
