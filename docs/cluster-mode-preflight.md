# Cluster-mode Phase 1.5 preflight report

Per [docs/plans/gpu-arbiter-cluster-mode.md](plans/gpu-arbiter-cluster-mode.md)
Phase 1.5: validate the worker protocol against the stub head BEFORE
Phase 2 commits any head-side code to it.

## How to run

```bash
make test-cluster-preflight
# or
bash tests/test-cluster-preflight.sh
```

No GPU required. Stub backends are fine. Wall time: ~1 minute.

## Scenarios covered

| # | Scenario                                              | Implementation                          |
| - | ----------------------------------------------------- | --------------------------------------- |
| 1 | Both workers register; both visible to head           | Spin up 2 workers, introspect head.     |
| 2 | Heartbeat at ~10s cadence; counters monotonic         | Wait 12s, count heartbeats received.    |
| 3 | `drain` command flows from head to worker             | Queue drain in head, observe worker log.|
| 4 | `serve` command + token gating (401 without token)    | curl `/v1/cluster/inbound` w/o token.   |
| 5 | `shutdown` lifecycle policy: ephemeral exits, persistent refuses | Run shutdown against both lifecycles. |
| 6 | Failure recovery: kill head, restart, workers re-register | stop/start the stub head mid-test.   |
| 7 | Token rotation: in-place rewrite doesn't break the worker | rewrite token file in place.        |

## First-run record (2026-05-15)

| Item              | Value                                              |
| ----------------- | -------------------------------------------------- |
| Date run          | 2026-05-15                                         |
| Host kernel       | Linux 6.12.85+deb13-amd64                          |
| Host GPU          | not exercised (stubbed backends)                   |
| Driver version    | n/a                                                |
| Arbiter version   | dev (commit a few past the Phase 1 commit)         |
| Result            | PASS -- all seven scenarios green                  |
| Wall time         | ~55 seconds                                        |

### Per-scenario observed behaviour

- **Scenario 1**: 2 registrations recorded, both with the right
  lifecycle field.
- **Scenario 2**: 2 heartbeats in 12s -- one tick from the worker
  registered earliest, plus the 10s tick from the second worker.
  Counters were monotonic (`1, 2`) per worker.
- **Scenario 3**: noopCommandExecutor's
  `drain backend=vllm acknowledged` log line appeared within 12s
  of the heartbeat issuing the command.
- **Scenario 4**: inbound `/v1/cluster/inbound` returned 401
  without `Authorization: Bearer the-token`, 503 with -- the 503
  is the Phase 1 placeholder ("not_implemented"); Phase 2 wires
  the real proxy.
- **Scenario 5**: ephemeral worker exited 2s after the
  `shutdown grace=2` command (matching `grace_seconds`); persistent
  worker logged
  `[worker] refusing shutdown command: lifecycle=persistent`
  and stayed up indefinitely.
- **Scenario 6**: workers logged `connection refused` while head
  was down, then resumed heartbeats within 12s of head restart.
  No re-registration was needed -- workers reuse their assigned
  worker_id across head bounces (Phase 2 may revisit if the head's
  fleet-state cache forgets workers; today the in-memory map is
  fine because the head is the bottleneck, not workers).
- **Scenario 7**: rewriting the token file in place did NOT trigger
  a worker restart. Heartbeats continued; counter kept incrementing.
  TokenStore's 30s cache made the rewrite invisible to the worker
  for up to 30s, which is acceptable given the test rewrites the
  same value.

## CI integration

The preflight script is CI-runnable (no GPU dependency). When the
project adds a CI workflow, gate every PR touching `gpu-arbiter/` or
`tests/test-cluster-*.sh` on `make test-cluster-preflight`. The
script currently lives outside the standard `make test` aggregate
because it requires the arbiter binary to be built (and adds ~1 min
to wall time).

## Re-running

This report is a one-time snapshot of the first successful run plus
its host details. Re-run scenarios 1-5 once on every new host class
joining the supported matrix (e.g. a new GPU type added to the
fleet, or a kernel upgrade beyond the major version recorded above).
Update the table above with the new host's result; do not delete the
prior row -- keep history so a regression spot-check is easy.

## Known limitations

- Scenario 7 doesn't exercise a real token rotation (different
  encrypted value), only an in-place rewrite. A real rotation
  test would require updating `cluster-token.sops.env`,
  re-rendering, and confirming the worker picks up the new token
  on next heartbeat. Documented as the operator-side smoke check
  in [docs/secrets.md](secrets.md).
- Scenarios 3-5 use the noopCommandExecutor, which acknowledges
  but doesn't actually serve. Real request proxying lands in
  Phase 2 -- re-test scenarios 3-5 against the real executor when
  Phase 2 ships.
