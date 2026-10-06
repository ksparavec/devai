# devai plans

There is one plan: [platform-rebuild](./platform-rebuild.md) (Draft, 2026-10-05; revised 2026-10-06). It holds all of devai's planned work: Phases 1-14 build the new architecture ([docs/slurm.md](../slurm.md)); Phases 15-21 are the independent work folded in from earlier plans; Phases 22-23 are the laya 0.3.28 sync and System One (the operator's E1-E6 of 2026-10-06). A new plan starts from [TEMPLATE.md](./TEMPLATE.md).

## Status legend

- **Draft** -- design under discussion, decisions not all locked.
- **Approved** -- decisions locked; ready to schedule.
- **In Progress** -- some phase has been started.
- **Done** -- all required phases shipped and verified.
- **Done (unverified)** -- all required phases code-complete and
  unit-tested, but the live verification the plan itself requires has
  not been run. Code-complete, NOT verified-in-production.
- **Non-functional** -- code-complete and its tests pass, but the
  feature does not work when actually exercised. This status exists
  because "Done (unverified)" was hiding four such cases: it reads as
  "works, just not tested here", when the truth was "never executed".
  A plan may only leave this state by being run.
- **Frozen** -- deliberately parked. Not maintained, not compiled, not
  deleted. Sources live under `attic/` with a restore guide. See
  `attic/README.md`.
- **Superseded** -- the plan's intent shipped by a different and better
  route, or its premise expired. The plan text is kept as a record but
  must not be executed as written.

## Removed plans (2026-10-05)

Every earlier plan was removed on 2026-10-05, and git history keeps each one:
`git log --diff-filter=D --oneline -- docs/plans/<name>.md` finds the commit that
removed it, and `git show <commit>^:docs/plans/<name>.md` prints it. Code comments
that still cite a removed plan point there. This index's own history (the defect
records of 2026-07-25/26, the dependency graph and the execution order) is in git
history too.

| Plan | Status when removed | What happened to it |
| --- | --- | --- |
| slurm-gatekeeper | Draft | merged into platform-rebuild; its decisions are D1-D16 there |
| minimal-external-images | Draft | merged; M1-M15 |
| devai-operator | Draft | merged; O1-O9 |
| gpu-arbiter-cluster-mode | Frozen | dropped: conflicts with the rebuild; code stays in `attic/cluster-mode/` |
| skypilot-fleet-provisioner | Frozen | dropped: conflicts with the rebuild; code stays in `attic/cluster-mode/` |
| skypilot-agent-skill | Frozen | dropped: Phase 1 shipped (the `sky` CLI in the lab); Phase 2 frozen |
| sops-age-secrets | Non-functional | dropped as moot: the scaffold goes to the attic (M12) |
| mcp-gateway | Partial | dropped as moot: the gateway is removed (M10) |
| kv-cache-quantization | Superseded | dropped |
| sglang-backend-remediation | In Progress | Phases 0-3 shipped; Phase 4's re-probe absorbed (platform-rebuild Phase 4; its 7 GPTQ cells are removed instead, I17) and the Phase 1 soak check (Phase 10); Phase 5 became Phase 15 |
| router-shortcircuit | Draft | Phases 1-4 became Phase 16; Phase 5's goal met by the router's mutation guard, its leftovers in 16e |
| router-fanout | Draft | Phase 1 and the single-host polish became Phase 19; Phases 2-3 dropped (cluster mode) |
| odysseus-borrowed-ideas | Draft | RT-1, CB-4 and CB-1's KV term shipped; CB-3 and the rest of CB-1 became Phase 17, CB-2 Phase 18, RT-6 part of Phase 9; RT-4 and RT-5 absorbed; RT-2, RT-3, CMP-1, CMP-2 rejected |
| library-service | Draft | became Phase 20; its decisions D1-D28 are LS1-LS28 there |
| laya-trainer | In Progress | Phases 1-4 shipped; Phase 4 closed by the operator on 2026-10-05 (platform-rebuild I23) |
| pi-coding-agent | Done | shipped; pinning pi and two notes are in Phase 21 |
| card-derived-hints-and-bench-sync | Done | shipped; the provenance docs are in Phase 21, the card branch's first real run in Phase 4 |
| model-lifecycle-ledger | Done (unverified) | shipped; its live `make model-sync` run absorbed (Phase 10); its Gemma-4-31B re-check dropped (I17) |
| review-fixes-2026-07 | Done (unverified) | shipped; its live runs absorbed (Phases 4 and 9) |
| bench-rewrite | Done | shipped; bench coverage absorbed (Phase 10) |
| router-anthropic-messages-compat | Done | shipped |
