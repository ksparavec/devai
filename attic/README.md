# attic -- frozen work

Code and configuration that was built, is being kept, and is NOT
currently maintained or compiled.

This is deliberately not `git rm`. The features here are intended to
come back. What is frozen is the *investment*, not the *idea*: nothing
in this tree is deleted, and each subdirectory carries enough context
to thaw it without re-deriving the design.

## What is in here

| Subtree | What it is | Frozen on | Why |
| --- | --- | --- | --- |
| [cluster-mode/](./cluster-mode/) | Multi-host cluster mode (`--mode=worker\|head`) and the SkyPilot cloud-burst fleet provisioner | 2026-07-25 | Built for a fleet this project does not have. Never functioned end to end. See below. |

## Why cluster-mode was frozen

A plan-portfolio review on 2026-07-25 established the following, all
verified against the tree rather than against the plans:

- **It was never run.** The cluster/fleet Go was written in a single
  day (2026-05-15, three commits) and touched once afterwards, as part
  of a blanket 116-finding review sweep rather than because anyone used
  it. Over the same period the single-host core loop (router, probe,
  bench, picker) took 84 commits.
- **It could not have been run.** `NewClusterHead` calls `log.Fatalf`
  when its bearer-token file is unreadable, and the compose `router`
  service mounts no `/run/devai`, publishes no ports, and
  `deploy/cluster-token.sops.env` never existed. `make cluster-head-up`
  crash-looped by construction.
- **The cloud half was inert.** `NewSkyPilotClient`,
  `NewSkyPilotPolicy` and `NewIdleTeardownCoordinator` had zero
  non-test callers anywhere in the repo, and `SKYPILOT_API_ENDPOINT`
  was read by nothing -- it appeared only inside comments. That is 848
  lines of production Go reachable only from its own 716 lines of
  tests.
- **Its tests asserted shape, not function.**
  `tests/test-fleet-routing.sh` had exactly one executable assertion (a
  curl for `/api/v1/version`) and exited 77 whenever the endpoint was
  unset, which was always. The Phase 1.5 preflight drove a Python stub
  head, never a real one.
- **It taxed the single-host path.** Extra modes, extra auth, extra
  files, and a cluster probe on the hot path of the
  `devai-model-status` MCP server -- all carried by a deployment that
  has exactly one GPU.

None of that is a judgement on the engineering, which is competent. It
is a judgement about scope: this is a single-workstation lab, and the
fleet was solving a problem the lab does not have yet.

## What is still live

The single-host serving path is untouched and is the only supported
mode. `gpu-arbiter` still accepts `--mode`, but any value other than
`single` now exits with a pointer to this file rather than silently
doing nothing.

## How to thaw

The Go sources under `cluster-mode/gpu-arbiter/` each carry a
`//go:build devai_frozen_cluster` tag, and `attic/` sits outside every
Go module in the repo, so they are invisible to `go build ./...` twice
over.

To bring the feature back:

1. `git mv attic/cluster-mode/gpu-arbiter/*.go gpu-arbiter/`
2. Strip the `//go:build devai_frozen_cluster` line (and the blank line
   after it) from each file.
3. Restore the `--mode` dispatch in `gpu-arbiter/main.go` -- see
   `git log -S runWorkerMode -- gpu-arbiter/main.go` for the original
   switch.
4. Restore the infra under `cluster-mode/deploy/` and
   `cluster-mode/tests/`, and re-add the Makefile targets listed in
   `cluster-mode/RESTORE.md`.

Before doing any of that, read `cluster-mode/RESTORE.md`: it lists the
defects that were open at freeze time. They are real and were never
fixed, so a naive thaw restores broken code.

## The design plans

The plans themselves stay in `docs/plans/` with a `Frozen` status
rather than moving here -- they are still the design record, and
`docs/plans/README.md` is the canonical status table.
