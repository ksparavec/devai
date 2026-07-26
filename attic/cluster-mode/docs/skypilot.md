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
  Reachable on port 46580, published on **127.0.0.1 only** -- the
  API server ships no authentication and `POST /api/v1/launch`
  spends real money, so remote access means an SSH tunnel or an
  authenticating reverse proxy, never a widened bind. `make
  skypilot-up` brings it live.
- **Phase 2 shipped** (2026-05-15, code only): `skypilot_client.go`
  (Launch / Status / Down + bearer auth), `skypilot_policy.go`
  (cheapest-cloud picker + per-launch budget cap + LaunchRequest
  builder + IdleTeardownCoordinator implementing the two-step
  "send shutdown via heartbeat then sky down" path). Live
  cloud-burst integration (head -> SkyPilot launch -> worker
  register -> serve) is gated behind `SKYPILOT_API_ENDPOINT` --
  when unset the head degrades to local-fleet-only routing per
  plan step 5.
- **Phase 3 pending**: cost-cap enforcement (per-day, per-cloud,
  per-user budgets), spot-instance preference with on-demand
  fallback, pre-warming N workers, observability hooks.

## Bring-up

Pre-conditions:

1. Cluster mode is opt-in. Bring up the head + workers per
   [docs/cluster-mode.md](cluster-mode.md).
2. Cloud credentials available -- either via the scoped credential
   mounts (`~/.aws`, `~/.config/gcloud`, `~/.config/sky`, each bound
   read-only) or the sops-rendered tmpfs file (for non-interactive
   RunPod / Lambda keys).

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
`make skypilot-up` works on a fresh install with only the scoped
credential mounts supplying credentials. That default feeds the
service's `env_file:` key as well as the `/secrets/.env` mount;
compose parses `/dev/null` as an empty env file (verified), so a
Phase 1 install with no rendered secrets boots unaffected. SkyPilot's
`sky check` will report which clouds it sees.

## Volume layout

| Volume                        | Purpose                                     |
| ----------------------------- | ------------------------------------------- |
| `skypilot-state` (named)      | SkyPilot's `~/.sky` registry, task logs, the provisioning keypair `sky` generates for itself, and `SKYPILOT_GLOBAL_CONFIG` (`/root/.sky/api-server-config.yaml`). |
| `${HOME}/.aws` -> `/root/.aws` ro | AWS credentials.                        |
| `${HOME}/.config/gcloud` -> `/root/.config/gcloud` ro | GCP credentials.    |
| `${HOME}/.config/sky` -> `/root/.config/sky` ro | Interactive `sky` config.  |
| `${HOME}/.runpod` -> `/root/.runpod` ro | RunPod credentials (`runpod config` writes `~/.runpod/config.toml`). |
| `${HOME}/.lambda_cloud` -> `/root/.lambda_cloud` ro | Lambda credentials (`~/.lambda_cloud/lambda_keys`). |
| `${SKYPILOT_CREDENTIALS_FILE}` -> `/secrets/.env` ro | On-disk copy of the sops-rendered service creds, for inspection only. The values are delivered by the `env_file:` key on the same service, which sources the file into the process environment -- nothing in `sky api start` reads `/secrets/.env`. |

RunPod and Lambda are the only two clouds the fleet provisioner can
actually pick (`skypilot_policy.go`'s `DefaultPricing` has no aws or
gcp rows, so `PickCheapest` never returns them); their mounts are
therefore the load-bearing ones. The `.aws` / `.config/gcloud` mounts
stay for `sky check` and for operators who override the pricing table.

These read-only binds replaced a whole-`${HOME}` -> `/root:rw`
mount. `~/.ssh` and `~/.config/sops/age` are **deliberately excluded**:
the latter holds the age private key that decrypts every
`deploy/*.sops.env`, and `sky` needs neither (it generates its own
provisioning keypair under `~/.sky`). Do not re-add them.

`SKYPILOT_GLOBAL_CONFIG` also moved, from `/root/.api-server-config.yaml`
to `/root/.sky/api-server-config.yaml` -- the old path only persisted
because the whole host `$HOME` was mounted over `/root`. An operator
who had a config at `$HOME/.api-server-config.yaml` must move it into
the `skypilot-state` volume for it to be read.

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

Two details of the shipped-but-unwired Phase 2 code worth knowing:

- `defaultVRAMForGPU` advertises **L40S as 48 GB** (its datasheet
  figure), not 24.
- The idle-teardown sweep **retains** a cluster in `pending` when
  `sky down` fails, and **never abandons it**: dropping the entry
  orphans a still-BILLING cloud VM, because nothing else ever revisits
  it. Retries continue forever with **exponential backoff**, from
  `DefaultTeardownRetryBase` (10s), doubling per failure, capped at
  `DefaultTeardownRetryCap` (15 min).
- That backoff **is** the log rate limiter -- an attempt only happens
  once its backoff window elapses, and only an attempt logs. A SkyPilot
  API server that stays down therefore costs one loud
  `[teardown] ERROR: sky down <cluster> (instance "<id>") failed: ...`
  line, a handful of doubling lines, then one line per 15 min, rather
  than one line per sweep forever.
- Pending entries are keyed by **identity, not name**:
  `MarkForTeardown(clusterName, instance)`, where `instance` is the
  `worker_id` `FleetState` assigned at registration. Marking an
  already-pending `(cluster, instance)` is a no-op (the grace deadline
  stays stable across repeated shutdown commands), but a *different*
  instance reusing the same cluster name gets its own entry with a
  fresh deadline. This closes the old name-reuse hole, where a new
  cluster inherited a stuck predecessor's long-elapsed deadline and was
  torn down on its first sweep.
- A **conflict guard** sits in front of `sky down`: when a different
  live worker currently holds the cluster name, the sweep refuses to
  call `sky down <name>` (it would kill the new cluster), flags the
  entry `conflicted`, and logs once. The old VM may still be billing
  and needs manual reconciliation (`sky status` / `sky down`).
- The operator surface is `StuckEntries()` -- pending teardowns with at
  least `DefaultStuckAfterFailures` (3) consecutive failures, or
  flagged `conflicted`. These are exactly the entries that may
  correspond to a still-billing VM. The sweep also emits at most one
  summary per `RetryCap`:
  `[teardown] N cluster(s) stuck awaiting teardown and possibly still
  BILLING: <cluster>(instance="...",failures=N), ...`.
  `Entries()` gives the full per-instance snapshot, `Pending()` the
  older name-keyed convenience view (earliest deadline wins on a
  reused name).
- `MarkForTeardown` has no production caller yet, so none of the above
  runs on a live head until the head-side wiring lands.

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

Either the scoped credential mounts surfaced nothing (the host paths
do not exist) OR `SKYPILOT_CREDENTIALS_FILE` wasn't set when starting
the container. Check the two clouds the provisioner can actually pick
first:

```bash
podman exec devai-skypilot-api-server ls -la /root/.runpod /root/.lambda_cloud 2>&1 || true
podman exec devai-skypilot-api-server ls -la /root/.aws /root/.config/gcloud /root/.config/sky 2>&1 || true
podman exec devai-skypilot-api-server ls -la /secrets/.env 2>&1 || true
```

`/secrets/.env` existing is not sufficient on its own: the values
reach the process through the service's `env_file:` key, so a mount
that appeared after the container started will not be in its
environment. Recreate the service after rendering.

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
